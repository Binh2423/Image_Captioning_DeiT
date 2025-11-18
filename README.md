# Image Captioning with DeiT + LFE + MSG + BiLSTM+MDSA-C

This repository implements an image captioning pipeline using a state-of-the-art architecture:
**DeiT (backbone) → LFE (Local Feature Enhancer) → MSG (Multi-Scale Gating) → BiLSTM decoder with MDSA-C (Multi-Head Dual-Stage Attention with Context)**

## Architecture Overview

### Visual Encoder Pipeline
1. **DeiT Backbone**: Data-efficient Image Transformer that extracts visual features from images
2. **LFE (Local Feature Enhancer)**: Refines local patch features using residual convolutional blocks
3. **MSG (Multi-Scale Gating)**: Produces multi-scale gated features with different receptive fields

### Decoder
- **BiLSTM Decoder**: Uses LSTM cells with attention mechanism to generate captions word-by-word
- **MDSA-C Attention**: Multi-Head Dual-Stage Attention with Context gating for better feature representation
- **Teacher Forcing**: Supports configurable teacher forcing ratio during training
- **Beam Search**: Supports both greedy decoding and beam search for generation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Dataset

Organize your dataset with the following structure:
```
data/
  flickr30k/
    images/
      image1.jpg
      image2.jpg
      ...
    captions.txt
```

### 2. Initialize Dataset (Build Vocabulary)

```bash
cd src
python initialize.py --dataset flickr30k
```

This will:
- Split the dataset into train/test
- Build vocabulary from training captions
- Save tokenizer with special tokens (<sos>, <eos>, <unk>, <pad>)

### 3. (Optional) Extract Visual Features

For faster training, pre-extract visual features:

```bash
cd src
python extract_deit_features.py \
  --images_dir ../data/flickr30k/images \
  --output_dir ../data/flickr30k/features \
  --deit_model_name deit_base_patch16_224 \
  --pretrained \
  --batch_size 32 \
  --device cuda
```

Options:
- `--use_bilstm_encoder`: Apply BiLSTM+MDSA-C encoding on features (slower but better)
- `--overwrite`: Overwrite existing features
- `--save_format`: Save as 'pt' (PyTorch) or 'npz' (NumPy)

### 4. Train the Model

#### Training with raw images:
```bash
cd src
python train_new.py \
  --dataset flickr30k \
  --weights-folder transformer \
  --histories-folder transformer \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --epochs 30
```

#### Training with precomputed features (faster):
```bash
cd src
python train_new.py \
  --dataset flickr30k \
  --features ../data/flickr30k/features \
  --weights-folder transformer \
  --histories-folder transformer \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --epochs 30
```

### 5. Generate Captions

```python
import torch
from models import CaptioningModel
from utils import Tokenizer
from PIL import Image
from torchvision import transforms

# Load model
model = CaptioningModel(
    vocab_size=10000,  # Your actual vocab size
    embed_dim=256,
    hidden_dim=512,
    visual_dim=768,
    pretrained=False  # Set True for first time
)
model.load_state_dict(torch.load('path/to/weights.pt'))
model.eval()

# Load tokenizer
tokenizer = Tokenizer.load('path/to/vocab.pkl')

# Prepare image
image = Image.open('path/to/image.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

# Generate caption
with torch.no_grad():
    caption_ids = model.generate(
        image_tensor,
        max_len=50,
        sos_idx=tokenizer.vocab.sos_idx,
        eos_idx=tokenizer.vocab.eos_idx,
        beam_size=3  # Use beam search
    )

# Decode caption
caption_tokens = [tokenizer.vocab.idx_2_str[idx.item()] 
                  for idx in caption_ids[0] 
                  if idx.item() in tokenizer.vocab.idx_2_str]
caption = ' '.join(caption_tokens)
print(f"Caption: {caption}")
```

## Model Components

### 1. LFE (Local Feature Enhancer)

```python
from models import LFE

lfe = LFE(
    in_channels=768,    # Input channels from DeiT
    hidden_dim=768,     # Hidden dimension
    num_layers=2        # Number of residual blocks
)
```

### 2. MSG (Multi-Scale Gating)

```python
from models import MSG

msg = MSG(
    in_dim=768,           # Input dimension
    scales=[1, 2, 4],     # Multi-scale receptive fields
    gating=True           # Enable gating mechanism
)
```

### 3. MDSA-C (Multi-Head Dual-Stage Attention with Context)

```python
from models import MDSA_C

mdsa = MDSA_C(
    embed_dim=512,     # Embedding dimension
    num_heads=8,       # Number of attention heads
    dropout=0.1        # Dropout rate
)
```

### 4. BiLSTM Decoder

```python
from models import BiLSTMDecoder

decoder = BiLSTMDecoder(
    vocab_size=10000,    # Vocabulary size
    embed_dim=256,       # Word embedding dimension
    hidden_dim=512,      # LSTM hidden dimension
    visual_dim=768,      # Visual feature dimension
    num_layers=1,        # Number of LSTM layers
    dropout=0.1,         # Dropout rate
    num_heads=8,         # Attention heads
    pad_idx=0            # Padding token index
)
```

### 5. Complete Captioning Model

```python
from models import CaptioningModel

model = CaptioningModel(
    vocab_size=10000,
    embed_dim=256,
    hidden_dim=512,
    visual_dim=768,
    num_decoder_layers=1,
    num_heads=8,
    dropout=0.1,
    deit_model_name='deit_base_patch16_224',
    pretrained=True,
    use_bilstm_encoder=False,  # Optional BiLSTM+MDSA-C on visual features
    pad_idx=0
)
```

## Configuration

Edit `models/transformer/config.json` to customize:

```json
{
    "deit_model_name": "deit_base_patch16_224",
    "pretrained": true,
    "visual_dim": 768,
    "embed_dim": 256,
    "hidden_dim": 512,
    "num_decoder_layers": 1,
    "num_heads": 8,
    "dropout": 0.1,
    "use_bilstm_encoder": false,
    "image_size": 224,
    "max_len": 80
}
```

## Available DeiT Models

From `timm` library:
- `deit_tiny_patch16_224`
- `deit_small_patch16_224`
- `deit_base_patch16_224` (default)
- `deit_base_patch16_384`
- `deit3_small_patch16_224`
- `deit3_base_patch16_224`

## Features

✅ **DeiT Backbone**: State-of-the-art vision transformer  
✅ **Local Feature Enhancement**: Refines spatial features  
✅ **Multi-Scale Gating**: Captures features at multiple scales  
✅ **MDSA-C Attention**: Advanced dual-stage attention with context  
✅ **Teacher Forcing**: Configurable during training  
✅ **Beam Search**: Better caption generation  
✅ **Feature Caching**: Pre-extract features for faster training  
✅ **Mixed Input**: Supports both raw images and precomputed features  
✅ **GPU Acceleration**: CUDA support throughout  

## API Reference

### CaptioningModel

#### Methods

- `forward(images, captions, teacher_forcing_ratio=1.0)`: Training forward pass
- `generate(images, max_len=50, sos_idx=1, eos_idx=2, beam_size=1)`: Generate captions
- `extract_features(images)`: Extract and cache visual features

### Extract Features Script

```bash
python extract_deit_features.py --help
```

Options:
- `--images_dir`: Directory containing images (required)
- `--output_dir`: Directory to save features (required)
- `--deit_model_name`: DeiT model name (default: deit_base_patch16_224)
- `--pretrained`: Use pretrained weights (default: True)
- `--batch_size`: Batch size (default: 32)
- `--num_workers`: DataLoader workers (default: 4)
- `--device`: Device (cuda/cpu, default: cuda)
- `--use_bilstm_encoder`: Apply BiLSTM+MDSA-C encoding
- `--overwrite`: Overwrite existing features
- `--save_format`: Save format (pt/npz, default: pt)

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── encoder.py           # DeiT + LFE + MSG
│   │   ├── decoder.py           # BiLSTM + MDSA-C
│   │   ├── captioning_model.py  # Complete model
│   │   └── __init__.py
│   ├── datasets/
│   │   └── captions_dataset.py  # Dataset with feature support
│   ├── utils/
│   │   ├── vocab.py             # Vocabulary builder
│   │   └── tokenizer.py         # Tokenizer
│   ├── extract_deit_features.py # Feature extraction script
│   ├── train_new.py             # Training script
│   └── initialize.py            # Dataset initialization
├── models/transformer/
│   └── config.json              # Model configuration
├── data/
│   └── flickr30k/               # Dataset
└── requirements.txt
```

## Citation

If you use this code, please cite the relevant papers:
- DeiT: "Training data-efficient image transformers & distillation through attention"
- Attention mechanisms and image captioning architectures

## License

This project is for educational and research purposes.
