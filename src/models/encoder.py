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
class LFE(nn.Module):
    def __init__(self, ch, num_blocks=2):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Sequential(
                ConvBNReLU(ch, ch, kernel=3, padding=1),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch)
            ))
        self.blocks = nn.ModuleList(layers)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        for blk in self.blocks:
            res = blk(x)
            x = self.act(x + res)
        return x

# Multi-Scale Grouping (MSG): parallel conv paths with different receptive fields
class MSG(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.path1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.path2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.path3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2)
        self.path4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(in_ch, out_ch, 1),
                                   nn.Upsample(scale_factor=None, mode='bilinear', align_corners=False))
        self.fuse = nn.Conv2d(out_ch*3, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        # simple global context if you don't want to resize: use avg pool -> broadcast
        gp = F.adaptive_avg_pool2d(x, 1)
        gp = nn.Conv2d(x.size(1), p1.size(1), 1)(gp)
        gp = F.interpolate(gp, size=p1.shape[2:], mode='bilinear', align_corners=False)
        cat = torch.cat([p1, p2, p3], dim=1)
        out = self.fuse(cat)
        out = self.act(self.bn(out))
        # add global context
        out = out + gp
        return out

class DeiT_LFE_MSG_Encoder(nn.Module):
    """
    Outputs:
      - feat_map: (B, C, H, W)
      - feat_seq: (B, N, C) where N = H*W (tokens, to feed BiLSTM)
    """
    def __init__(self, deit_model_name='deit_base_patch16_224', pretrained=True, out_ch=768):
        super().__init__()
        # load timm model
        self.deit = timm.create_model(deit_model_name, pretrained=pretrained, num_classes=0, global_pool='') 
        # Many timm DeiT variants expose forward_features; using create_model with num_classes=0 gives feature tokens
        self.patch_embed = getattr(self.deit, 'patch_embed', None)
        # channel dim from DeiT
        self.embed_dim = out_ch
        self.lfe = LFE(self.embed_dim, num_blocks=2)
        self.msg = MSG(self.embed_dim, self.embed_dim)
        # keep patch size so we can reshape tokens -> map
        # assume patch_embed exists and patch_embed.patch_size
        if self.patch_embed is not None:
            ps = getattr(self.patch_embed, 'patch_size', (16,16))
            self.patch_size = ps[0]
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