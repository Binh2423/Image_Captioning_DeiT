import torch
import torch.nn as nn
from typing import Optional
from .encoder import DeiT_LFE_MSG_Encoder
from .decoder import BiLSTMDecoder, BiLSTM_MDSA_C_Encoder


class CaptioningModel(nn.Module):
    """
    Complete image captioning model that wires:
    DeiT (backbone) → LFE (Local Feature Enhancer) → MSG (Multi-Scale Gating) → BiLSTM decoder with MDSA-C
    
    This model supports both training (with teacher forcing) and inference (with generation).
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: Optional[int] = None,
                 hidden_dim: int = 512,
                 visual_dim: int = 768,
                 num_decoder_layers: int = 1,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 deit_model_name: str = 'deit_base_patch16_224',
                 pretrained: bool = True,
                 pad_idx: int = 0,
                 use_bilstm_encoder: bool = False,
                 device: str = 'cpu',
                 **kwargs):
        """
        Complete image captioning model combining DeiT encoder with BiLSTM decoder.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings. If None, defaults to 256.
            hidden_dim: Hidden dimension for LSTM decoder (default: 512)
            visual_dim: Output dimension from DeiT encoder (default: 768)
            num_decoder_layers: Number of LSTM layers in decoder (default: 1)
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout rate (default: 0.1)
            deit_model_name: Name of DeiT model from timm (default: 'deit_base_patch16_224')
            pretrained: Whether to use pretrained DeiT weights (default: True)
            pad_idx: Padding token index (default: 0)
            use_bilstm_encoder: Whether to use BiLSTM+MDSA-C on visual features before decoding (default: False)
            device: Device to run on (default: 'cpu')
            **kwargs: Additional keyword arguments (silently ignored for config compatibility)
        """
        super().__init__()
        
        # Set embed_dim with inference logic
        if embed_dim is None:
            # Default to 256 if not provided
            embed_dim = 256
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.visual_dim = visual_dim
        self.pad_idx = pad_idx
        self.use_bilstm_encoder = use_bilstm_encoder
        self.device = device
        
        # DeiT + LFE + MSG encoder
        self.visual_encoder = DeiT_LFE_MSG_Encoder(
            deit_model_name=deit_model_name,
            pretrained=pretrained,
            out_ch=visual_dim
        )
        
        # Optional: BiLSTM + MDSA-C encoder on visual features
        if use_bilstm_encoder:
            self.bilstm_encoder = BiLSTM_MDSA_C_Encoder(
                in_dim=visual_dim,
                hidden_dim=hidden_dim // 2,  # Will be doubled by bidirectional
                num_layers=1,
                num_heads=num_heads,
                dropout=dropout
            )
            decoder_visual_dim = hidden_dim
        else:
            self.bilstm_encoder = None
            decoder_visual_dim = visual_dim
        
        # BiLSTM decoder with MDSA-C attention
        self.decoder = BiLSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            visual_dim=decoder_visual_dim,
            num_layers=num_decoder_layers,
            dropout=dropout,
            num_heads=num_heads,
            pad_idx=pad_idx
        )
    
    def forward(self, images, captions, teacher_forcing_ratio=1.0):
        """
        Forward pass for training.
        
        Args:
            images: (B, 3, H, W) - input images
            captions: (B, max_len) - target captions with <sos> at start
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: (B, max_len-1, vocab_size) - predicted logits
        """
        # Extract visual features
        feat_map, feat_seq = self.visual_encoder(images)  # feat_seq: (B, N, visual_dim)
        
        # Optional: Apply BiLSTM + MDSA-C encoder on visual features
        if self.bilstm_encoder is not None:
            feat_seq, _ = self.bilstm_encoder(feat_seq)  # (B, N, hidden_dim)
        
        # Decode captions
        outputs = self.decoder(feat_seq, captions, teacher_forcing_ratio=teacher_forcing_ratio)
        
        return outputs
    
    def generate(self, images, max_len=50, sos_idx=1, eos_idx=2, beam_size=1):
        """
        Generate captions for images.
        
        Args:
            images: (B, 3, H, W) or pre-extracted features (B, N, visual_dim)
            max_len: Maximum caption length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            beam_size: Beam width (1 = greedy)
        
        Returns:
            captions: (B, max_len) - generated caption indices
        """
        # Check if input is images or features
        if images.dim() == 4:  # (B, 3, H, W) - raw images
            # Extract visual features
            feat_map, feat_seq = self.visual_encoder(images)
        else:  # Pre-extracted features
            feat_seq = images
        
        # Optional: Apply BiLSTM + MDSA-C encoder
        if self.bilstm_encoder is not None:
            feat_seq, _ = self.bilstm_encoder(feat_seq)
        
        # Generate captions
        captions = self.decoder.generate(
            feat_seq,
            max_len=max_len,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            beam_size=beam_size
        )
        
        return captions
    
    def extract_features(self, images):
        """
        Extract visual features from images (useful for caching).
        
        Args:
            images: (B, 3, H, W)
        
        Returns:
            feat_seq: (B, N, visual_dim) or (B, N, hidden_dim) if BiLSTM encoder is used
        """
        with torch.no_grad():
            feat_map, feat_seq = self.visual_encoder(images)
            
            if self.bilstm_encoder is not None:
                feat_seq, _ = self.bilstm_encoder(feat_seq)
        
        return feat_seq
