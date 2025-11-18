import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable, Any
from PIL import Image

class CaptionsDataset(Dataset):
    """
    Dataset for image captioning that supports both raw images and precomputed features.
    
    If features_dir is provided, loads precomputed visual features instead of images.
    Otherwise, loads and transforms images on-the-fly.
    """

    def __init__(self,
        root: str,
        captions_df: str,
        img_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        features_dir: Optional[str] = None,
        feature_format: str = 'pt'
    ):
        """
        Args:
            root: Root directory containing images
            captions_df: Path to CSV file with image and caption columns
            img_transform: Transform to apply to images (if loading raw images)
            caption_transform: Transform to apply to captions (e.g., tokenizer)
            features_dir: Directory containing precomputed features (optional)
            feature_format: Format of precomputed features ('pt' or 'npz')
        """
        super().__init__()
        
        self.root = root
        self.img_transform = img_transform
        self.caption_transform = caption_transform
        self.captions_df = captions_df
        self.features_dir = features_dir
        self.feature_format = feature_format
        
        # Flag to track if using precomputed features
        self.use_features = features_dir is not None

        self.df = pd.read_csv(self.captions_df)
        self.images = self.df['image'].tolist()
        self.captions = self.df['caption'].tolist()

    def __getitem__(self, index) -> Any:
        caption = self.captions[index]
        
        if self.use_features:
            # Load precomputed features
            image_name = self.images[index]
            
            # Determine feature file path
            if self.feature_format == 'pt':
                feature_path = os.path.join(self.features_dir, image_name).replace(
                    os.path.splitext(image_name)[1], '.pt'
                )
                features = torch.load(feature_path)
            elif self.feature_format == 'npz':
                feature_path = os.path.join(self.features_dir, image_name).replace(
                    os.path.splitext(image_name)[1], '.npz'
                )
                data = np.load(feature_path)
                features = torch.from_numpy(data['features'])
            else:
                raise ValueError(f'Unsupported feature format: {self.feature_format}')
            
            visual_input = features
        else:
            # Load raw image
            path = os.path.join(self.root, self.images[index])
            img = Image.open(path).convert('RGB')

            if self.img_transform is not None:
                img = self.img_transform(img)
            
            visual_input = img

        if self.caption_transform is not None:
            caption = self.caption_transform(caption)

        return visual_input, caption
    
    def __len__(self) -> int:
        return len(self.captions)