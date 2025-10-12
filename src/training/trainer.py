"""
Training script for Two-Tower recommendation model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime


class MetricsCalculator:
    """Calculate recommendation metrics"""
    
    @staticmethod
    def ndcg_at_k(predictions: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
        """
        Calculate nDCG@k
        Args:
            predictions: Predicted scores (batch_size, num_items)
            labels: Binary relevance labels (batch_size, num_items)
            k: Top-k items to consider
        """
        # Get top-k predictions
        top_k_idx = np.argsort(-predictions, axis=1)[:, :k]
        
        ndcg_scores = []
        for i in range(len(predictions)):
            # Get relevance of top-k items
            relevance = labels[i][top_k_idx[i]]
            
            # DCG calculation
            dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
            
            # IDCG calculation (ideal ranking)
            ideal_relevance = np.sort(labels[i])[::-1][:k]
            idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))
            
            # nDCG
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores)
    
    @staticmethod
    def hit_rate_at_k(predictions: np.ndarray, labels: np.ndarray, k: int = 20) -> float:
        """Calculate Hit Rate@k"""
        top_k_idx = np.argsort(-predictions, axis=1)[:, :k]
        
        hits = 0
        for i in range(len(predictions)):
            if np.any(labels[i][top_k_idx[i]] > 0):
                hits += 1
        
        return hits / len(predictions)
    
    @staticmethod
    def coverage(predictions: np.ndarray, num_items: int, k: int = 20) -> float:
        """Calculate catalog coverage"""
        top_k_idx = np.argsort(-predictions, axis=1)[:, :k]
        unique_items = np.unique(top_k_idx.flatten())
        
        return len(unique_items) / num_items
    
    @staticmethod
    def mrr(predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Mean Reciprocal Rank"""
        sorted_idx = np.argsort(-predictions, axis=1)
        
        mrr_scores = []
        for i in range(len(predictions)):
            relevant_items = np.where(labels[i] > 0)[0]
            if len(relevant_items) == 0:
                continue
            
            # Find rank of first relevant item
            for rank, idx in enumerate(sorted_idx[i], 1):
                if idx in relevant_items:
                    mrr_scores.append(1.0 / rank)
                    break
        
        return np.mean(mrr_scores) if mrr_scores else 0.0


class Trainer:
    """Training manager for recommendation model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        from two_tower import ContrastiveLoss
        self.criterion = ContrastiveLoss(temperature=config.get('temperature', 0.07))
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_ndcg = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_ndcg': [],
            'val_hit_rate': [],
            'val_coverage': []
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_optimizer(self):
        """Initialize optimizer"""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_name == 'adamw':
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self):
        """Initialize learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 50),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch in pbar:
            # Move data to device
            user_ids = batch['user_ids'].to(self.device)
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            user_emb, content_emb, similarity = self.model(
                user_ids, images, input_ids, attention_mask
            )
            
            # Calculate loss
            loss = self.criterion(similarity)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move data to device
            user_ids = batch['user_ids'].to(self.device)
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels']
            
            # Forward pass
            user_emb, content_emb, similarity = self.model(
                user_ids, images, input_ids, attention_mask
            )
            
            # Calculate loss
            loss = self.criterion(similarity)
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions for metrics
            all_predictions.append(similarity.cpu().numpy())
            all_labels.append(labels.numpy())
        
        # Calculate metrics
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        metrics = {
            'loss': total_loss / num_batches,
            'ndcg@10': self.metrics_calc.ndcg_at_k(predictions, labels, k=10),
            'hit_rate@20': self.metrics_calc.hit_rate_at_k(predictions, labels, k=20),
            'coverage': self.metrics_calc.coverage(predictions, num_items=predictions.shape[1], k=20),
            'mrr': self.metrics_calc.mrr(predictions, labels)
        }
        
        return metrics
    
    def train(self, num_epochs: int):
        """Full training loop"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Config: {json.dumps(self.config, indent=2)}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_ndcg'].append(val_metrics['ndcg@10'])
            self.history['val_hit_rate'].append(val_metrics['hit_rate@20'])
            self.history['val_coverage'].append(val_metrics['coverage'])
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"nDCG@10: {val_metrics['ndcg@10']:.4f} - "
                f"Hit Rate@20: {val_metrics['hit_rate@20']:.4f} - "
                f"Coverage: {val_metrics['coverage']:.4f}"
            )
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_metrics['ndcg@10'] > self.best_ndcg:
                self.best_ndcg = val_metrics['ndcg@10']
                self.save_checkpoint('best_model.pth', is_best=True)
                self.logger.info(f"New best model saved! nDCG@10: {self.best_ndcg:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best nDCG@10: {self.best_ndcg:.4f}")
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_ndcg': self.best_ndcg,
            'history': self.history,
            'config': self.config
        }
        
        filepath = checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_ndcg = checkpoint['best_ndcg']
        self.history = checkpoint['history']
        
        self.logger.info(f"Loaded checkpoint from {filepath}")
        self.logger.info(f"Resuming from epoch {self.current_epoch + 1}")


if __name__ == "__main__":
    # Example usage
    from two_tower import TwoTowerModel
    
    # Configuration
    config = {
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'epochs': 50,
        'batch_size': 256,
        'temperature': 0.07,
        'grad_clip': 1.0,
        'save_freq': 10,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs'
    }
    
    print("Trainer module ready for use!")
    print(f"Configuration: {json.dumps(config, indent=2)}")
