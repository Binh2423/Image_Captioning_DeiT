"""
Unit tests for CaptioningModel constructor signature.

Tests that CaptioningModel.__init__ can handle:
- embed_dim parameter (both provided and inferred)
- Extra keyword arguments from config dicts
- No TypeError when instantiated with config-like dicts
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from models.captioning_model import CaptioningModel


def test_model_instantiation_with_embed_dim():
    """Test that CaptioningModel accepts embed_dim parameter."""
    vocab_size = 1000
    embed_dim = 512
    
    model = CaptioningModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        pretrained=False  # Don't download weights
    )
    
    assert model.embed_dim == embed_dim
    assert model.vocab_size == vocab_size


def test_model_instantiation_without_embed_dim():
    """Test that CaptioningModel infers embed_dim when not provided."""
    vocab_size = 1000
    
    model = CaptioningModel(
        vocab_size=vocab_size,
        pretrained=False  # Don't download weights
    )
    
    # Should default to 256
    assert model.embed_dim == 256
    assert model.vocab_size == vocab_size


def test_model_instantiation_with_config_dict():
    """Test that CaptioningModel can be instantiated with a config dict containing extra keys."""
    config = {
        'vocab_size': 1000,
        'embed_dim': 256,
        'hidden_dim': 512,
        'visual_dim': 768,
        'num_decoder_layers': 1,
        'num_heads': 8,
        'dropout': 0.1,
        'deit_model_name': 'deit_base_patch16_224',
        'pretrained': False,  # Use False to avoid downloading weights in test
        'pad_idx': 0,
        'use_bilstm_encoder': False,
        'device': 'cpu',
        # Extra keys that should be ignored
        'max_len': 80,
        'encoder_length': 196,
        'features_dim': 256,
        'num_layers': 6,
        'foo_param': 123,  # Arbitrary extra parameter
    }
    
    # This should NOT raise TypeError
    model = CaptioningModel(**config)
    
    assert model.embed_dim == config['embed_dim']
    assert model.vocab_size == config['vocab_size']
    assert model.hidden_dim == config['hidden_dim']
    assert hasattr(model, 'embed_dim')


def test_model_has_required_attributes():
    """Test that model has all required attributes after initialization."""
    vocab_size = 1000
    embed_dim = 256
    
    model = CaptioningModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        pretrained=False  # Don't download weights
    )
    
    # Check that model has expected attributes
    assert hasattr(model, 'embed_dim')
    assert hasattr(model, 'vocab_size')
    assert hasattr(model, 'hidden_dim')
    assert hasattr(model, 'visual_dim')
    assert hasattr(model, 'decoder')
    assert hasattr(model, 'visual_encoder')
    
    # Check attribute values
    assert model.embed_dim == embed_dim
    assert model.vocab_size == vocab_size


def test_model_with_none_embed_dim():
    """Test that explicitly passing None for embed_dim uses default."""
    vocab_size = 1000
    
    model = CaptioningModel(
        vocab_size=vocab_size,
        embed_dim=None,
        pretrained=False  # Don't download weights
    )
    
    # Should default to 256
    assert model.embed_dim == 256


if __name__ == '__main__':
    # Run tests when script is executed directly
    print("Running test_model_instantiation_with_embed_dim...")
    test_model_instantiation_with_embed_dim()
    print("✓ Passed")
    
    print("\nRunning test_model_instantiation_without_embed_dim...")
    test_model_instantiation_without_embed_dim()
    print("✓ Passed")
    
    print("\nRunning test_model_instantiation_with_config_dict...")
    test_model_instantiation_with_config_dict()
    print("✓ Passed")
    
    print("\nRunning test_model_has_required_attributes...")
    test_model_has_required_attributes()
    print("✓ Passed")
    
    print("\nRunning test_model_with_none_embed_dim...")
    test_model_with_none_embed_dim()
    print("✓ Passed")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
