"""
Robust CaptioningModel implementation with backward compatibility.

This module provides a flexible captioning model that:
- Accepts config-style kwargs and ignores unknown ones
- Supports multiple input formats (raw images, features, collate tuples)
- Handles missing components gracefully with clear error messages
- Maintains backward compatibility with existing code
"""
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional
import logging

# Import encoders and decoders
from .encoder import DeiT_LFE_MSG_Encoder
from .decoder import BiLSTMDecoder, BiLSTM_MDSA_C_Encoder

logger = logging.getLogger(__name__)


class CaptioningModel(nn.Module):
    """
    Robust image captioning model supporting multiple input formats and configurations.
    
    Features:
    - Config-first construction: accepts various config parameters via kwargs
    - Multiple input formats in forward():
      * Raw images (B, C, H, W) when backbone present
      * Precomputed features (B, L, D) when backbone is None
      * Collate tuple format ((visuals, y_input), y_expected) or (visuals, y_input)
    - Graceful handling of missing components
    - Backward compatible with existing code
    """
    
    def __init__(self,
                 vocab_size=None,
                 embed_dim=256,
                 hidden_dim=512,
                 visual_dim=768,
                 num_decoder_layers=1,
                 num_heads=8,
                 dropout=0.1,
                 deit_model_name='deit_base_patch16_224',
                 pretrained=True,
                 pad_idx=0,
                 use_bilstm_encoder=False,
                 device='cpu',
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 **kwargs):
        """
        Flexible constructor accepting both direct components and config parameters.
        
        Args:
            vocab_size: Size of vocabulary (required if decoder not provided)
            embed_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension for LSTM decoder
            visual_dim: Output dimension from visual encoder
            num_decoder_layers: Number of LSTM layers in decoder
            num_heads: Number of attention heads
            dropout: Dropout rate
            deit_model_name: Name of DeiT model from timm (if using backbone)
            pretrained: Whether to use pretrained weights
            pad_idx: Padding token index
            use_bilstm_encoder: Whether to use BiLSTM+MDSA-C on visual features
            device: Device to run on
            backbone: Optional pre-built backbone (DeiT encoder)
            encoder: Optional pre-built visual encoder
            decoder: Optional pre-built decoder
            **kwargs: Additional config parameters (ignored for compatibility)
        """
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.visual_dim = visual_dim
        self.pad_idx = pad_idx
        self.use_bilstm_encoder = use_bilstm_encoder
        self.device = device
        self._config = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'visual_dim': visual_dim,
            'num_decoder_layers': num_decoder_layers,
            'num_heads': num_heads,
            'dropout': dropout,
            'deit_model_name': deit_model_name,
            'pretrained': pretrained,
            'pad_idx': pad_idx,
            'use_bilstm_encoder': use_bilstm_encoder,
        }
        
        # Visual encoder (backbone): can be None for feature-only mode
        if backbone is not None:
            self.visual_encoder = backbone
            logger.info("Using provided backbone")
        elif encoder is not None:
            self.visual_encoder = encoder
            logger.info("Using provided encoder")
        elif deit_model_name is not None:
            # Create DeiT + LFE + MSG encoder only if model name provided
            try:
                self.visual_encoder = DeiT_LFE_MSG_Encoder(
                    deit_model_name=deit_model_name,
                    pretrained=pretrained,
                    out_ch=visual_dim
                )
                logger.info(f"Created visual encoder: {deit_model_name}")
            except Exception as e:
                logger.warning(f"Could not create visual encoder: {e}. Running in feature-only mode.")
                self.visual_encoder = None
        else:
            # No backbone, encoder, or model name provided - feature-only mode
            logger.info("No visual encoder specified. Running in feature-only mode.")
            self.visual_encoder = None
        
        # Optional BiLSTM + MDSA-C encoder on visual features
        if use_bilstm_encoder:
            self.bilstm_encoder = BiLSTM_MDSA_C_Encoder(
                in_dim=visual_dim,
                hidden_dim=hidden_dim // 2,
                num_layers=1,
                num_heads=num_heads,
                dropout=dropout
            )
            decoder_visual_dim = hidden_dim
            logger.info("Created BiLSTM encoder")
        else:
            self.bilstm_encoder = None
            decoder_visual_dim = visual_dim
        
        # Decoder: can be provided or created
        if decoder is not None:
            self.decoder = decoder
            # Ensure decoder has embed_dim attribute
            if not hasattr(self.decoder, 'embed_dim'):
                self.decoder.embed_dim = embed_dim
            logger.info("Using provided decoder")
        else:
            # Create decoder if vocab_size is available
            if vocab_size is not None:
                try:
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
                    logger.info("Created BiLSTM decoder")
                except Exception as e:
                    logger.warning(f"Could not create decoder: {e}")
                    self.decoder = None
            else:
                logger.warning("No vocab_size provided and no decoder given. Decoder must be set before forward pass.")
                self.decoder = None
    
    def _extract_input_tensors(self, *args):
        """
        Extract visual inputs and captions from various input formats.
        
        Supports:
        1. (images, captions) - standard format
        2. (TensorTuple((images, captions_input)), captions_expected) - collate format
        3. (images,) - single tensor (for generation)
        
        Returns:
            visuals: Tensor of visual inputs (images or features)
            captions: Tensor of captions (or None for generation)
        """
        if len(args) == 0:
            raise ValueError("forward() requires at least one argument")
        
        # Handle TensorTuple wrapper (from Collate)
        first_arg = args[0]
        
        # Check if it's a TensorTuple or tuple-like object
        if hasattr(first_arg, 't') and isinstance(first_arg.t, tuple):
            # TensorTuple((visuals, y_input)) format
            visuals, captions = first_arg.t
            return visuals, captions
        elif isinstance(first_arg, tuple) and len(first_arg) == 2:
            # Regular tuple ((visuals, y_input), y_expected)
            inner_tuple, y_expected = first_arg
            if isinstance(inner_tuple, tuple) and len(inner_tuple) == 2:
                visuals, y_input = inner_tuple
                return visuals, y_input
            else:
                # Assume (visuals, captions)
                return first_arg
        elif torch.is_tensor(first_arg):
            # Standard tensor input
            visuals = first_arg
            captions = args[1] if len(args) > 1 else None
            return visuals, captions
        else:
            raise ValueError(f"Unsupported input format: {type(first_arg)}")
    
    def forward(self, *args, teacher_forcing_ratio=1.0, **kwargs):
        """
        Forward pass supporting multiple input formats.
        
        Input formats supported:
        1. forward(images, captions) - standard
        2. forward(features, captions) - precomputed features
        3. forward(TensorTuple((visuals, y_input))) - collate output
        4. forward(((visuals, y_input), y_expected)) - nested tuple
        
        Args:
            *args: Variable arguments (see formats above)
            teacher_forcing_ratio: Probability of using teacher forcing
            **kwargs: Additional keyword arguments
        
        Returns:
            outputs: (B, T, vocab_size) - predicted logits
        """
        # Extract inputs from various formats
        visuals, captions = self._extract_input_tensors(*args)
        
        # Check if decoder is available
        if self.decoder is None:
            raise RuntimeError(
                "No decoder available. Please set model.decoder before calling forward(). "
                "You can use SimpleBiLSTMDecoder as a fallback."
            )
        
        # Process visual inputs
        if visuals.dim() == 4:
            # Raw images (B, 3, H, W)
            if self.visual_encoder is None:
                raise RuntimeError(
                    "No visual encoder available for raw images. "
                    "Either provide precomputed features or set a backbone."
                )
            feat_map, feat_seq = self.visual_encoder(visuals)
        elif visuals.dim() == 3:
            # Precomputed features (B, N, D)
            feat_seq = visuals
        else:
            raise ValueError(
                f"Visual input must be 4D (images) or 3D (features), got shape {visuals.shape}"
            )
        
        # Optional BiLSTM encoder
        if self.bilstm_encoder is not None:
            feat_seq, _ = self.bilstm_encoder(feat_seq)
        
        # Decode captions
        if captions is None:
            raise ValueError("Captions required for forward pass. Use generate() for inference.")
        
        outputs = self.decoder(feat_seq, captions, teacher_forcing_ratio=teacher_forcing_ratio)
        
        return outputs
    
    def generate(self, images, max_len=50, sos_idx=1, eos_idx=2, beam_size=1):
        """
        Generate captions for images.
        
        Args:
            images: (B, 3, H, W) or pre-extracted features (B, N, D)
            max_len: Maximum caption length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            beam_size: Beam width (1 = greedy)
        
        Returns:
            captions: (B, max_len) - generated caption indices
        """
        if self.decoder is None:
            raise RuntimeError("No decoder available. Cannot generate captions.")
        
        # Extract features if needed
        if images.dim() == 4:
            # Raw images
            if self.visual_encoder is None:
                raise RuntimeError("No visual encoder available for raw images.")
            feat_map, feat_seq = self.visual_encoder(images)
        else:
            # Pre-extracted features
            feat_seq = images
        
        # Optional BiLSTM encoder
        if self.bilstm_encoder is not None:
            feat_seq, _ = self.bilstm_encoder(feat_seq)
        
        # Generate captions
        if hasattr(self.decoder, 'generate'):
            captions = self.decoder.generate(
                feat_seq,
                max_len=max_len,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                beam_size=beam_size
            )
        else:
            # Fallback greedy generation if decoder doesn't have generate method
            logger.warning("Decoder missing generate() method. Using basic greedy fallback.")
            captions = self._greedy_fallback(feat_seq, max_len, sos_idx, eos_idx)
        
        return captions
    
    def _greedy_fallback(self, visual_features, max_len, sos_idx, eos_idx):
        """Basic greedy generation fallback"""
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # Start with <sos>
        captions = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            # Forward pass
            try:
                outputs = self.decoder(visual_features, captions)
                next_tokens = outputs[:, -1, :].argmax(dim=-1)
            except:
                # If decoder can't handle this, break
                break
            
            # Append to captions
            captions = torch.cat([captions, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for <eos>
            finished |= (next_tokens == eos_idx)
            if finished.all():
                break
        
        return captions
    
    def extract_features(self, images):
        """
        Extract visual features from images (useful for caching).
        
        Args:
            images: (B, 3, H, W)
        
        Returns:
            feat_seq: (B, N, D) - visual feature sequence
        """
        if self.visual_encoder is None:
            raise RuntimeError("No visual encoder available.")
        
        with torch.no_grad():
            feat_map, feat_seq = self.visual_encoder(images)
            
            if self.bilstm_encoder is not None:
                feat_seq, _ = self.bilstm_encoder(feat_seq)
        
        return feat_seq
