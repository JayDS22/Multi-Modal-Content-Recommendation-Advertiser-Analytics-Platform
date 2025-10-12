"""
Complete training script for Two-Tower recommendation model
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import Dict
import numpy as np
from PIL import Image
from transformers import BertTokenizer
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal content (images + text)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 128,
        image_size: int = 224
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to data file (JSON)
            tokenizer: BERT tokenizer
            max_length: Maximum text sequence length
            image_size: Image size for ResNet
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # User ID
        user_id = item['user_id']
        
        # Image (generate random for demo)
        # In production, load actual images
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Text
        text = item.get('text', '')
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Label (binary relevance)
        label = item.get('label', 1)
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }


def generate_synthetic_data(output_path: str, num_samples: int = 10000):
    """Generate synthetic training data"""
    
    logger.info(f"Generating {num_samples} synthetic samples")
    
    data = []
    texts = [
        "Beautiful summer dress perfect for any occasion",
        "Latest smartphone with advanced camera features",
        "Comfortable running shoes for daily workouts",
        "Delicious pasta recipe with fresh ingredients",
        "Scenic mountain view travel destination"
    ]
    
    for i in range(num_samples):
        data.append({
            'user_id': np.random.randint(0, 1000),
            'item_id': f'item_{i}',
            'text': np.random.choice(texts),
            'label': np.random.randint(0, 2)
        })
    
    # Save data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"Saved synthetic data to {output_path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config: Dict):
    """
    Main training function
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting training")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_data_path = config.get('train_data_path', 'data/train.json')
    val_data_path = config.get('val_data_path', 'data/val.json')
    
    # Generate data if it doesn't exist
    if not Path(train_data_path).exists():
        generate_synthetic_data(train_data_path, num_samples=8000)
    if not Path(val_data_path).exists():
        generate_synthetic_data(val_data_path, num_samples=2000)
    
    train_dataset = MultiModalDataset(train_data_path, tokenizer)
    val_dataset = MultiModalDataset(val_data_path, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Initialize model
    from two_tower import TwoTowerModel
    
    model = TwoTowerModel(
        num_users=config.get('num_users', 10000),
        embedding_dim=config.get('embedding_dim', 128),
        temperature=config.get('temperature', 0.07)
    )
    model = model.to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    from trainer import Trainer
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    num_epochs = config.get('epochs', 50)
    trainer.train(num_epochs)
    
    logger.info("Training completed!")
    
    # Save final model
    final_model_path = Path(config.get('checkpoint_dir', 'checkpoints')) / 'final_model.pth'
    trainer.save_checkpoint(str(final_model_path))
    
    logger.info(f"Final model saved to {final_model_path}")


def evaluate(config: Dict):
    """
    Evaluate trained model
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting evaluation")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from two_tower import TwoTowerModel
    
    model = TwoTowerModel(
        num_users=config.get('num_users', 10000),
        embedding_dim=config.get('embedding_dim', 128)
    )
    
    # Load checkpoint
    checkpoint_path = config.get('checkpoint_path', 'checkpoints/best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Best nDCG@10: {checkpoint.get('best_ndcg', 'N/A')}")
    
    # TODO: Add detailed evaluation metrics
    
    logger.info("Evaluation completed!")


def build_index(config: Dict):
    """
    Build FAISS index for inference
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Building FAISS index")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    from two_tower import TwoTowerModel
    
    model = TwoTowerModel(
        num_users=config.get('num_users', 10000),
        embedding_dim=config.get('embedding_dim', 128)
    )
    
    checkpoint_path = config.get('checkpoint_path', 'checkpoints/best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    # Generate embeddings for all content
    # In production, load actual content data
    num_items = config.get('num_items', 10000)
    embedding_dim = config.get('embedding_dim', 128)
    
    logger.info(f"Generating embeddings for {num_items} items")
    
    all_embeddings = []
    item_ids = []
    
    with torch.no_grad():
        # Generate in batches
        batch_size = 256
        for i in range(0, num_items, batch_size):
            batch_end = min(i + batch_size, num_items)
            batch_items = batch_end - i
            
            # Generate dummy images and text
            images = torch.randn(batch_items, 3, 224, 224).to(device)
            input_ids = torch.randint(0, 30522, (batch_items, 32)).to(device)
            attention_mask = torch.ones(batch_items, 32).to(device)
            
            # Get embeddings
            embeddings = model.get_content_embedding(images, input_ids, attention_mask)
            
            all_embeddings.append(embeddings.cpu().numpy())
            item_ids.extend([f'item_{j}' for j in range(i, batch_end)])
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {batch_end}/{num_items} items")
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
    
    # Build FAISS index
    from vector_search import FAISSVectorSearch
    
    search_engine = FAISSVectorSearch(
        dimension=embedding_dim,
        index_type=config.get('index_type', 'IVF1024,Flat'),
        metric=config.get('metric', 'IP')
    )
    
    search_engine.build_index(all_embeddings, item_ids)
    
    # Save index
    index_path = config.get('index_path', 'data/embeddings/faiss.index')
    metadata_path = config.get('metadata_path', 'data/embeddings/metadata.pkl')
    
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    
    search_engine.save(index_path, metadata_path)
    
    logger.info(f"FAISS index saved to {index_path}")
    logger.info("Index building completed!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Multi-Modal Recommendation Model')
    
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'evaluate', 'build_index'],
        help='Mode: train, evaluate, or build_index'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        logger.warning(f"Config file {args.config} not found, using default config")
        config = {
            'num_users': 10000,
            'embedding_dim': 128,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'checkpoint_dir': 'checkpoints'
        }
    
    # Run selected mode
    if args.mode == 'train':
        train(config)
    elif args.mode == 'evaluate':
        evaluate(config)
    elif args.mode == 'build_index':
        build_index(config)


if __name__ == "__main__":
    main()
