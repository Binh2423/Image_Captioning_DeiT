"""
Unit tests for CaptioningModel forward pass with different input formats.

Tests verify:
- Model construction with dummy components
- Forward pass with raw images returns correct shape
- Forward pass with precomputed features works
- Forward pass with collate tuple format works
- Generate method works
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import unittest
from models import CaptioningModel, SimpleBiLSTMDecoder


class TensorTuple:
    """Local copy of TensorTuple for testing"""
    def __init__(self, t):
        self.t = t
    
    def __getitem__(self, index):
        return self.t[index]
    
    def to(self, device):
        r = []
        for e in self.t:
            e.to(device)
            r.append(e.to(device))
        return tuple(r)


class DummyBackbone(nn.Module):
    """Dummy visual encoder for testing"""
    def __init__(self, visual_dim=768):
        super().__init__()
        self.visual_dim = visual_dim
        self.conv = nn.Conv2d(3, visual_dim, 1)
    
    def forward(self, x):
        # x: (B, 3, H, W)
        # Return feat_map and feat_seq
        B = x.size(0)
        feat_map = self.conv(x)  # (B, visual_dim, H, W)
        H, W = feat_map.shape[2:]
        feat_seq = feat_map.flatten(2).permute(0, 2, 1)  # (B, H*W, visual_dim)
        return feat_map, feat_seq


class TestCaptioningModel(unittest.TestCase):
    """Test suite for CaptioningModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vocab_size = 1000
        self.embed_dim = 256
        self.hidden_dim = 512
        self.visual_dim = 768
        self.batch_size = 2
        self.max_len = 20
        self.img_size = 32
        self.pad_idx = 0
        self.device = 'cpu'
    
    def test_model_construction_with_decoder(self):
        """Test that model can be constructed with a decoder"""
        decoder = SimpleBiLSTMDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            pad_idx=self.pad_idx
        )
        
        model = CaptioningModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            decoder=decoder,
            backbone=None,  # Feature-only mode
            deit_model_name=None  # Don't create DeiT encoder
        )
        
        self.assertIsNotNone(model.decoder)
        self.assertEqual(model.vocab_size, self.vocab_size)
    
    def test_model_construction_config_style(self):
        """Test that model can be constructed with config parameters"""
        # This will create decoder internally but NOT encoder (backbone=None)
        # We need to suppress encoder creation entirely
        model = CaptioningModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            pad_idx=self.pad_idx,
            backbone=None,  # Feature-only mode - no encoder/backbone
            deit_model_name=None,  # Don't create DeiT
            pretrained=False  # Don't download weights in tests
        )
        
        # Decoder should be created if vocab_size is provided
        self.assertIsNotNone(model.decoder)
    
    def test_forward_with_images(self):
        """Test forward pass with raw images"""
        # Create model with dummy backbone
        backbone = DummyBackbone(self.visual_dim)
        decoder = SimpleBiLSTMDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            pad_idx=self.pad_idx
        )
        
        model = CaptioningModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            backbone=backbone,
            decoder=decoder
        )
        
        # Create dummy inputs
        images = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        captions = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        captions[:, 0] = 1  # Set <sos> token
        
        # Forward pass
        outputs = model(images, captions)
        
        # Check output shape: (B, max_len-1, vocab_size)
        self.assertEqual(outputs.shape, (self.batch_size, self.max_len - 1, self.vocab_size))
    
    def test_forward_with_features(self):
        """Test forward pass with precomputed features"""
        decoder = SimpleBiLSTMDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            pad_idx=self.pad_idx
        )
        
        model = CaptioningModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            backbone=None,  # Feature-only mode
            decoder=decoder,
            deit_model_name=None  # Don't create DeiT encoder
        )
        
        # Create dummy precomputed features
        seq_len = 49  # e.g., 7x7 patches
        features = torch.randn(self.batch_size, seq_len, self.visual_dim)
        captions = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        captions[:, 0] = 1  # Set <sos> token
        
        # Forward pass
        outputs = model(features, captions)
        
        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, self.max_len - 1, self.vocab_size))
    
    def test_forward_with_collate_tuple(self):
        """Test forward pass with collate tuple format"""
        decoder = SimpleBiLSTMDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            pad_idx=self.pad_idx
        )
        
        model = CaptioningModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            backbone=None,
            decoder=decoder,
            deit_model_name=None  # Don't create DeiT encoder
        )
        
        # Create inputs in collate format: TensorTuple((visuals, y_input))
        seq_len = 49
        features = torch.randn(self.batch_size, seq_len, self.visual_dim)
        captions = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        captions[:, 0] = 1
        
        # Wrap in TensorTuple
        collate_input = TensorTuple((features, captions))
        
        # Forward pass
        outputs = model(collate_input)
        
        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, self.max_len - 1, self.vocab_size))
    
    def test_generate(self):
        """Test caption generation"""
        decoder = SimpleBiLSTMDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            pad_idx=self.pad_idx
        )
        
        model = CaptioningModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            backbone=None,
            decoder=decoder,
            deit_model_name=None  # Don't create DeiT encoder
        )
        
        # Create dummy features
        seq_len = 49
        features = torch.randn(self.batch_size, seq_len, self.visual_dim)
        
        # Generate captions
        generated = model.generate(features, max_len=15, sos_idx=1, eos_idx=2)
        
        # Check output shape: (B, max_len)
        self.assertEqual(generated.shape[0], self.batch_size)
        self.assertLessEqual(generated.shape[1], 15)
    
    def test_model_with_missing_decoder_raises_error(self):
        """Test that forward pass without decoder raises clear error"""
        model = CaptioningModel(
            vocab_size=None,  # No vocab_size, so no decoder created
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            visual_dim=self.visual_dim,
            backbone=None,
            deit_model_name=None  # Don't create DeiT encoder
        )
        
        # Decoder should be None
        self.assertIsNone(model.decoder)
        
        # Forward pass should raise RuntimeError
        features = torch.randn(self.batch_size, 49, self.visual_dim)
        captions = torch.randint(0, self.vocab_size, (self.batch_size, self.max_len))
        
        with self.assertRaises(RuntimeError) as context:
            model(features, captions)
        
        self.assertIn("decoder", str(context.exception).lower())
    
    def test_unknown_kwargs_ignored(self):
        """Test that unknown kwargs are ignored for compatibility"""
        # Should not raise even with unknown parameters
        model = CaptioningModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            unknown_param_1="test",
            unknown_param_2=123,
            another_param=True,
            backbone=None,
            deit_model_name=None  # Don't create DeiT encoder
        )
        
        # Model should be created successfully
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
