import sys
sys.path.append('../..')
import argparse
import torch
import time
import os
import logging
import logging.config
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models import CaptioningModel, SimpleBiLSTMDecoder
from datasets import CaptionsDataset
from utils import Tokenizer, TensorTuple, read_json, seed_everything
from losses import Seq2SeqCrossentropy
from metrics import Seq2SeqAccuracy
from torchmetrics.text import Perplexity
from utils import DatasetDescriptor
from trainer import Trainer
from dataclasses import dataclass
from definitions import *
from transformers import get_linear_schedule_with_warmup
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Normalize
import torchvision.transforms as transforms
logging.basicConfig(level = logging.INFO, format=' %(name)s :: %(levelname)-8s :: %(message)s')

@dataclass
class Args:
    dataset : str
    features : str

    batch_size : int
    weight_decay : float
    learning_rate : float
    epochs : int

    num_workers : int
    prefetch_factor : int

    weights_folder : str
    histories_folder : str

class Collate:
    """
    Collate function that handles both raw images and precomputed features.
    """
    def __init__(self, maxlen: int, pad_idx: int):
        self.maxlen = maxlen
        self.pad_idx = pad_idx

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[tuple[Tensor, Tensor], Tensor]:
        # Stack visual inputs (images or features)
        # Features have shape (N, C), images have shape (3, H, W)
        first_item = batch[0][0]
        
        if first_item.dim() == 2:
            # Precomputed features: (N, C)
            # Pad to same sequence length
            max_seq_len = max(x[0].size(0) for x in batch)
            feat_dim = first_item.size(1)
            
            padded_features = []
            for x in batch:
                feat = x[0]
                if feat.size(0) < max_seq_len:
                    # Pad with zeros
                    padding = torch.zeros(max_seq_len - feat.size(0), feat_dim)
                    feat = torch.cat([feat, padding], dim=0)
                padded_features.append(feat.unsqueeze(0))
            
            visual_inputs = torch.cat(padded_features, dim=0)
        else:
            # Raw images: (3, H, W)
            visual_inputs = torch.cat([torch.unsqueeze(x[0], dim=0) for x in batch], dim=0)

        # Process captions
        captions = [x[1] for x in batch]
        captions = torch.nn.utils.rnn.pad_sequence(sequences=captions, batch_first=True, padding_value=self.pad_idx)

        # Ensure fixed max length
        captions_ = torch.zeros(size=(captions.size(0), self.maxlen)).type(torch.long).fill_(self.pad_idx)
        captions_[:, :min(captions.size(1), self.maxlen)] = captions[:, :min(captions.size(1), self.maxlen)]

        # Input to model includes <sos>, expected output shifts by one
        y_input = captions_[:, :-1]
        y_expected = captions_[:, 1:]

        return TensorTuple((visual_inputs, y_input)), y_expected


def get_deit_transforms(image_size=224, is_training=True):
    """
    Get transforms consistent with DeiT preprocessing.
    
    Args:
        image_size: Target image size
        is_training: Whether for training (includes augmentation)
    
    Returns:
        Compose transform
    """
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if is_training:
        return Compose([
            RandomResizedCrop(image_size),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            normalize
        ])
    else:
        return Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            ToTensor(),
            normalize
        ])


def create_datasets(
    descriptor: DatasetDescriptor,
    tokenizer: Tokenizer,
    features_dir: str = None,
    image_size: int = 224
) -> tuple[CaptionsDataset, CaptionsDataset]:
    """
    Create training and validation datasets.
    
    Args:
        descriptor: Dataset descriptor with paths
        tokenizer: Tokenizer for captions
        features_dir: Optional directory with precomputed features
        image_size: Image size for transforms
    
    Returns:
        (train_dataset, val_dataset)
    """
    if features_dir is not None and os.path.exists(features_dir):
        # Use precomputed features
        logging.info(f'Using precomputed features from: {features_dir}')
        train_data = CaptionsDataset(
            root=descriptor.images_dir,
            captions_df=descriptor.train_captions,
            caption_transform=tokenizer,
            features_dir=features_dir,
            feature_format='pt'
        )

        test_data = CaptionsDataset(
            root=descriptor.images_dir,
            captions_df=descriptor.test_captions,
            caption_transform=tokenizer,
            features_dir=features_dir,
            feature_format='pt'
        )
    else:
        # Use raw images with transforms
        logging.info('Using raw images with on-the-fly transforms')
        train_transforms = get_deit_transforms(image_size, is_training=True)
        val_transforms = get_deit_transforms(image_size, is_training=False)
        
        train_data = CaptionsDataset(
            root=descriptor.images_dir,
            captions_df=descriptor.train_captions,
            caption_transform=tokenizer,
            img_transform=train_transforms
        )

        test_data = CaptionsDataset(
            root=descriptor.images_dir,
            captions_df=descriptor.test_captions,
            caption_transform=tokenizer,
            img_transform=val_transforms
        )

    return train_data, test_data


def read_weights_folder(path: str):
    folders = os.listdir(path)
    folders = filter(lambda f : f.endswith('.pt'), folders)
    folders = map(lambda f : os.path.join(path, f), folders)
    folders = sorted(folders)
    num_epochs = len(folders)
    
    last_weights = None
    
    if num_epochs != 0:
        print(f"Loading weights: {folders[-1]}")
        last_weights = torch.load(folders[-1])
        
    return last_weights, num_epochs


def main(args: Args):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath("D:/DACN/Flickr30k")))
    HISTORIES_DIR = os.path.join(ROOT_DIR, 'histories')
    MODELS_DIR = os.path.join(ROOT_DIR, 'models')   

    ### Preparing paths
    histories_folder = os.path.join(HISTORIES_DIR, args.histories_folder)
    weights_folder = os.path.join(MODELS_DIR, args.weights_folder)
    best_weights_folder = os.path.join(MODELS_DIR, args.weights_folder, 'best_weights')

    ### Check for directories existence
    if not os.path.exists(histories_folder):
        raise Exception(f"{histories_folder} not found")
    
    if not os.path.exists(weights_folder):
        raise Exception(f"{weights_folder} not found")
    
    if not os.path.exists(best_weights_folder):
        raise Exception(f"{best_weights_folder} not found")
    
    ### Setup code to be device agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Reproducibility
    seed_everything()

    ### Creating a logger
    logger = logging

    ### Loading model config
    config = read_json(os.path.join(weights_folder, "config.json"))
    logger.info(f'config = {str(config)}')

    ### Creating training & test datasets
    logger.info('creating training & testing datasets')
    
    descriptor: DatasetDescriptor = DatasetDescriptor.get_by_name(args.dataset)
    tokenizer: Tokenizer = Tokenizer.load(descriptor.vocab_path)

    # Check if features directory specified
    features_dir = args.features if hasattr(args, 'features') and args.features else None
    
    train_data, test_data = create_datasets(
        descriptor=descriptor,
        tokenizer=tokenizer,
        features_dir=features_dir,
        image_size=config.get('image_size', 224)
    )

    ### Creating train & test data loaders
    logger.info('creating training & testing dataloaders')

    vocab_size = len(tokenizer.vocab)
    pad_idx = tokenizer.vocab.pad_idx

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=Collate(config["max_len"], pad_idx),
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Collate(config["max_len"], pad_idx),
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0
    )

    logger.info(f'vocab_size = {vocab_size}')

    ### Creating the model
    logger.info('creating model, optimizer, loss & trainer instances')

    # Create the captioning model with config parameters
    # The new CaptioningModel is robust and will create components as needed
    # If decoder creation fails, it will be None and we can set a fallback
    model = CaptioningModel(
        vocab_size=vocab_size,
        embed_dim=config.get('embed_dim', 256),
        hidden_dim=config.get('hidden_dim', 512),
        visual_dim=config.get('visual_dim', 768),
        num_decoder_layers=config.get('num_decoder_layers', 1),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        deit_model_name=config.get('deit_model_name', 'deit_base_patch16_224'),
        pretrained=config.get('pretrained', True),
        pad_idx=pad_idx,
        use_bilstm_encoder=config.get('use_bilstm_encoder', False),
        device=device
    ).to(device)
    
    # Check if decoder was created, otherwise use fallback
    if model.decoder is None:
        logger.warning("Primary decoder not available, using SimpleBiLSTMDecoder fallback")
        decoder_visual_dim = config.get('hidden_dim', 512) if model.use_bilstm_encoder else config.get('visual_dim', 768)
        model.decoder = SimpleBiLSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=config.get('embed_dim', 256),
            hidden_dim=config.get('hidden_dim', 512),
            visual_dim=decoder_visual_dim,
            num_layers=config.get('num_decoder_layers', 1),
            dropout=config.get('dropout', 0.1),
            pad_idx=pad_idx
        ).to(device)
        logger.info("SimpleBiLSTMDecoder fallback installed successfully")

    last_weights, num_epochs = read_weights_folder(weights_folder)

    if last_weights is not None:
        model.load_state_dict(last_weights)

    ### Optimizer
    optimizer = AdamW(
        [{'params': model.parameters(), 'initial_lr': args.learning_rate}],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    ### Loss & other metrics
    loss = Seq2SeqCrossentropy(ignore_index=pad_idx).to(device=device)
    accuracy = Seq2SeqAccuracy(num_classes=vocab_size, ignore_index=pad_idx).to(device=device)
    perplexity = Perplexity(ignore_index=pad_idx).to(device=device)

    ### Learning rate scheduling
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(0.05 * num_training_steps)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=num_epochs-1,
    )

    ### Trainer class
    trainer = Trainer() \
        .set_model(model) \
        .set_criteron(loss) \
        .set_scheduler(lr_scheduler) \
        .set_save_weights_every(1) \
        .set_weights_folder(weights_folder) \
        .add_metric("accuracy", accuracy) \
        .add_metric("perplexity", perplexity) \
        .set_device(device) \
        .set_optimizer(optimizer) \
        .set_score_metric("accuracy") \
        .set_weights_folder(weights_folder)
    
    ### Start training
    logger.info('training starts now')
    
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        epochs=args.epochs
    )

    logger.info('end of training')
    logger.info('saving results to the disk')

    ### Save the history
    t = time.time()

    trainer.history.to_df().to_csv(
        os.path.join(histories_folder, f"{t}.csv"),
        index=False
    )

    #### Save the model
    torch.save(model.state_dict(), os.path.join(weights_folder, f"{t}.pt"))
    torch.save(trainer.best_weights, os.path.join(best_weights_folder, f"{t}_best_weights.pt"))

    logger.info('Goodbye')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, choices=DatasetDescriptor.LOOKUP.keys(), default="flickr30k")
    parser.add_argument('--features', type=str, default=None, help='Path to precomputed features directory')

    # hyperparameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)

    # dataloader
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=2)

    # folders
    parser.add_argument('--weights-folder', type=str, required=True)
    parser.add_argument('--histories-folder', type=str, required=True)

    args = parser.parse_args()
    
    main(args)
