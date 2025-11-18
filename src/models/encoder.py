import torch
import torch.nn as nn
import timm
from torch.nn import functional as F

# small helper: conv block
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Local Feature Enhancer (LFE): a couple of residual conv blocks to boost locality
# API: LFE(in_channels, hidden_dim, num_layers)
class LFE(nn.Module):
    def __init__(self, in_channels, hidden_dim=None, num_layers=2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_channels
        
        # Projection layer if input channels don't match hidden dim
        self.input_proj = None
        if in_channels != hidden_dim:
            self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                ConvBNReLU(hidden_dim, hidden_dim, kernel=3, padding=1),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim)
            ))
        self.blocks = nn.ModuleList(layers)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        for blk in self.blocks:
            res = blk(x)
            x = self.act(x + res)
        return x

# Multi-Scale Gating (MSG): parallel conv paths with different receptive fields and gating
# API: MSG(in_dim, scales=[1,2,4], gating=True)
class MSG(nn.Module):
    def __init__(self, in_dim, scales=[1, 2, 4], gating=True):
        super().__init__()
        self.scales = scales
        self.gating = gating
        out_ch = in_dim
        
        # Multi-scale paths based on scales parameter
        self.paths = nn.ModuleList()
        for scale in scales:
            if scale == 1:
                self.paths.append(nn.Conv2d(in_dim, out_ch, kernel_size=1, padding=0))
            else:
                self.paths.append(nn.Conv2d(in_dim, out_ch, kernel_size=3, padding=scale, dilation=scale))
        
        # Global context path
        self.global_conv = nn.Conv2d(in_dim, out_ch, 1)
        
        # Fusion layer
        num_paths = len(scales) + 1  # +1 for global context
        self.fuse = nn.Conv2d(out_ch * num_paths, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        
        # Gating mechanism
        if self.gating:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, out_ch, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # Multi-scale paths
        path_outputs = []
        for path in self.paths:
            path_outputs.append(path(x))
        
        # Global context
        gp = F.adaptive_avg_pool2d(x, 1)
        gp = self.global_conv(gp)
        gp = F.interpolate(gp, size=path_outputs[0].shape[2:], mode='bilinear', align_corners=False)
        path_outputs.append(gp)
        
        # Concatenate and fuse
        cat = torch.cat(path_outputs, dim=1)
        out = self.fuse(cat)
        out = self.act(self.bn(out))
        
        # Apply gating if enabled
        if self.gating:
            gate_weights = self.gate(out)
            out = out * gate_weights
        
        return out

class DeiT_LFE_MSG_Encoder(nn.Module):
    """
    DeiT backbone with LFE and MSG modules.
    
    Outputs:
      - feat_map: (B, C, H, W) - spatial feature map
      - feat_seq: (B, N, C) where N = H*W - sequence of tokens for BiLSTM
    """
    def __init__(self, deit_model_name='deit_base_patch16_224', pretrained=True, out_ch=768, **kwargs):
        super().__init__()
        # load timm model
        self.deit = timm.create_model(deit_model_name, pretrained=pretrained, num_classes=0, global_pool='') 
        # Many timm DeiT variants expose forward_features; using create_model with num_classes=0 gives feature tokens
        self.patch_embed = getattr(self.deit, 'patch_embed', None)
        # channel dim from DeiT
        self.embed_dim = out_ch
        self.lfe = LFE(in_channels=self.embed_dim, hidden_dim=self.embed_dim, num_layers=2)
        self.msg = MSG(in_dim=self.embed_dim, scales=[1, 2, 4], gating=True)
        # keep patch size so we can reshape tokens -> map
        # assume patch_embed exists and patch_embed.patch_size
        if self.patch_embed is not None:
            ps = getattr(self.patch_embed, 'patch_size', (16,16))
            if isinstance(ps, (tuple, list)):
                self.patch_size = ps[0]
            else:
                self.patch_size = ps
        else:
            self.patch_size = 16

    def forward(self, x):
        # x: (B,3,H,W)
        # use DeiT's forward_features to get tokens: (B, N+1, C)
        # many timm models implement model.forward_features
        tokens = self.deit.forward_features(x)  # expected token output (B, 1+N, C) or (B, C, h, w) depending on model
        # If forward_features returns (B, C, H, W) leave it as-is. Otherwise handle tokens.
        if tokens.dim() == 4:
            feat_map = tokens  # already 2D
        else:
            # tokens shape (B, 1+N, C)
            cls_token = tokens[:, :1, :]      # optional; can be used or ignored
            patch_tokens = tokens[:, 1:, :]   # (B, N, C)
            B, N, C = patch_tokens.shape
            # infer spatial dims (assume square)
            s = int(N**0.5)
            feat_map = patch_tokens.transpose(1,2).reshape(B, C, s, s)
        # LFE + MSG
        feat_map = self.lfe(feat_map)
        feat_map = self.msg(feat_map)
        B, C, H, W = feat_map.shape
        feat_seq = feat_map.flatten(2).permute(0,2,1)  # (B, N, C)
        return feat_map, feat_seq