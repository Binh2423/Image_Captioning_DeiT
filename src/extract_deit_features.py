#!/usr/bin/env python3
"""
Extract DeiT features from images and save them for faster training.

This script preprocesses images with the DeiT backbone (through LFE and MSG modules)
and saves per-image visual features to disk, enabling faster training by avoiding
repeated forward passes through the visual encoder.

Usage:
    python extract_deit_features.py --images_dir /path/to/images --output_dir /path/to/features
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import DeiT_LFE_MSG_Encoder, BiLSTM_MDSA_C_Encoder


class ImageDataset(Dataset):
    """Simple dataset for loading images from a directory."""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, str(image_path)


def get_deit_transforms(image_size=224):
    """
    Get image transforms consistent with DeiT preprocessing.
    
    Args:
        image_size: Target image size
    
    Returns:
        transforms.Compose object
    """
    # DeiT uses ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),  # Resize to slightly larger
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])


def extract_features(
    images_dir: str,
    output_dir: str,
    deit_model_name: str = 'deit_base_patch16_224',
    pretrained: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    device: str = 'cuda',
    overwrite: bool = False,
    use_bilstm_encoder: bool = False,
    hidden_dim: int = 512,
    save_format: str = 'pt'
):
    """
    Extract visual features from images using DeiT + LFE + MSG.
    
    Args:
        images_dir: Directory containing images
        output_dir: Directory to save features
        deit_model_name: Name of DeiT model from timm
        pretrained: Whether to use pretrained weights
        batch_size: Batch size for feature extraction
        num_workers: Number of dataloader workers
        image_size: Input image size
        device: Device to use ('cuda' or 'cpu')
        overwrite: Whether to overwrite existing features
        use_bilstm_encoder: Whether to apply BiLSTM+MDSA-C encoding
        hidden_dim: Hidden dimension for BiLSTM encoder
        save_format: Format to save features ('pt' or 'npz')
    """
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(images_dir).rglob(f'*{ext}'))
        image_paths.extend(Path(images_dir).rglob(f'*{ext.upper()}'))
    
    image_paths = sorted(list(set(image_paths)))
    print(f'Found {len(image_paths)} images')
    
    if len(image_paths) == 0:
        print('No images found! Exiting.')
        return
    
    # Create dataset and dataloader
    transform = get_deit_transforms(image_size)
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    # Create encoder
    print(f'Loading DeiT model: {deit_model_name}')
    encoder = DeiT_LFE_MSG_Encoder(
        deit_model_name=deit_model_name,
        pretrained=pretrained,
        out_ch=768  # Standard DeiT dimension
    ).to(device)
    encoder.eval()
    
    # Optional BiLSTM encoder
    bilstm_encoder = None
    if use_bilstm_encoder:
        print('Using BiLSTM + MDSA-C encoder')
        bilstm_encoder = BiLSTM_MDSA_C_Encoder(
            in_dim=768,
            hidden_dim=hidden_dim // 2,
            num_layers=1,
            num_heads=8,
            dropout=0.0
        ).to(device)
        bilstm_encoder.eval()
    
    # Extract features
    print('Extracting features...')
    num_processed = 0
    num_skipped = 0
    
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc='Processing batches'):
            images = images.to(device)
            
            # Extract features
            feat_map, feat_seq = encoder(images)  # feat_seq: (B, N, 768)
            
            # Optional BiLSTM encoding
            if bilstm_encoder is not None:
                feat_seq, _ = bilstm_encoder(feat_seq)  # (B, N, hidden_dim)
            
            # Save features for each image
            for i, image_path in enumerate(paths):
                # Determine output filename
                rel_path = Path(image_path).relative_to(images_dir)
                output_path = Path(output_dir) / rel_path.with_suffix(f'.{save_format}')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Skip if exists and not overwriting
                if output_path.exists() and not overwrite:
                    num_skipped += 1
                    continue
                
                # Get features for this image
                features = feat_seq[i].cpu()  # (N, C)
                
                # Save features
                if save_format == 'pt':
                    torch.save(features, output_path)
                elif save_format == 'npz':
                    np.savez_compressed(output_path, features=features.numpy())
                else:
                    raise ValueError(f'Unsupported save format: {save_format}')
                
                num_processed += 1
    
    print(f'\nFeature extraction complete!')
    print(f'Processed: {num_processed} images')
    print(f'Skipped (already exist): {num_skipped} images')
    print(f'Features saved to: {output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Extract DeiT features from images for faster training'
    )
    
    # Required arguments
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save extracted features'
    )
    
    # Model arguments
    parser.add_argument(
        '--deit_model_name',
        type=str,
        default='deit_base_patch16_224',
        help='DeiT model name from timm (default: deit_base_patch16_224)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained DeiT weights (default: True)'
    )
    parser.add_argument(
        '--no_pretrained',
        action='store_false',
        dest='pretrained',
        help='Do not use pretrained weights'
    )
    parser.add_argument(
        '--use_bilstm_encoder',
        action='store_true',
        help='Apply BiLSTM + MDSA-C encoding on features'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=512,
        help='Hidden dimension for BiLSTM encoder (default: 512)'
    )
    
    # Processing arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for feature extraction (default: 32)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers (default: 4)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='Input image size (default: 224)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing features'
    )
    parser.add_argument(
        '--save_format',
        type=str,
        default='pt',
        choices=['pt', 'npz'],
        help='Format to save features (default: pt)'
    )
    
    args = parser.parse_args()
    
    # Extract features
    extract_features(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        deit_model_name=args.deit_model_name,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        device=args.device,
        overwrite=args.overwrite,
        use_bilstm_encoder=args.use_bilstm_encoder,
        hidden_dim=args.hidden_dim,
        save_format=args.save_format
    )


if __name__ == '__main__':
    main()
