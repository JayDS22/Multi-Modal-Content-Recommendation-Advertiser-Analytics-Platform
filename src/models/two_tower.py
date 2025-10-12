"""
Two-Tower Neural Network for Multi-Modal Recommendation
Combines image embeddings (ResNet-50) and text features (BERT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, BertTokenizer
from typing import Dict, Tuple, Optional


class ImageEncoder(nn.Module):
    """Image encoding tower using ResNet-50"""
    
    def __init__(self, output_dim: int = 128, pretrained: bool = True):
        super(ImageEncoder, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (batch_size, 3, 224, 224)
        Returns:
            embeddings: Tensor of shape (batch_size, output_dim)
        """
        features = self.backbone(images)
        features = features.view(features.size(0), -1)  # Flatten
        embeddings = self.projection(features)
        
        # L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class TextEncoder(nn.Module):
    """Text encoding tower using BERT"""
    
    def __init__(self, output_dim: int = 128, bert_model: str = 'bert-base-uncased'):
        super(TextEncoder, self).__init__()
        
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(bert_model)
        
        # Freeze BERT layers except last 2 transformer blocks
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < 10:  # Freeze first 10 layers
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
        Returns:
            embeddings: Tensor of shape (batch_size, output_dim)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(cls_embedding)
        
        # L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class UserTower(nn.Module):
    """User encoding tower for collaborative features"""
    
    def __init__(self, num_users: int, embedding_dim: int = 64, output_dim: int = 128):
        super(UserTower, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: Tensor of shape (batch_size,)
        Returns:
            embeddings: Tensor of shape (batch_size, output_dim)
        """
        user_emb = self.user_embedding(user_ids)
        embeddings = self.projection(user_emb)
        
        # L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class TwoTowerModel(nn.Module):
    """
    Two-Tower Neural Network combining user and content (image + text) towers
    """
    
    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 128,
        temperature: float = 0.07
    ):
        super(TwoTowerModel, self).__init__()
        
        self.temperature = temperature
        
        # Initialize towers
        self.image_encoder = ImageEncoder(output_dim=embedding_dim)
        self.text_encoder = TextEncoder(output_dim=embedding_dim)
        self.user_tower = UserTower(num_users=num_users, output_dim=embedding_dim)
        
        # Fusion layer for combining image and text
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(
        self,
        user_ids: torch.Tensor,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            user_ids: Tensor of shape (batch_size,)
            images: Tensor of shape (batch_size, 3, 224, 224)
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
        
        Returns:
            user_embeddings: Tensor of shape (batch_size, embedding_dim)
            content_embeddings: Tensor of shape (batch_size, embedding_dim)
            similarity_scores: Tensor of shape (batch_size, batch_size)
        """
        # Encode user
        user_embeddings = self.user_tower(user_ids)
        
        # Encode content (image + text)
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        
        # Fuse image and text embeddings
        combined = torch.cat([image_embeddings, text_embeddings], dim=1)
        content_embeddings = self.fusion(combined)
        
        # L2 normalize fused embeddings
        content_embeddings = F.normalize(content_embeddings, p=2, dim=1)
        
        # Compute similarity scores (dot product)
        similarity_scores = torch.matmul(
            user_embeddings, 
            content_embeddings.T
        ) / self.temperature
        
        return user_embeddings, content_embeddings, similarity_scores
    
    def get_content_embedding(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get content embedding for inference"""
        with torch.no_grad():
            image_embeddings = self.image_encoder(images)
            text_embeddings = self.text_encoder(input_ids, attention_mask)
            
            combined = torch.cat([image_embeddings, text_embeddings], dim=1)
            content_embeddings = self.fusion(combined)
            content_embeddings = F.normalize(content_embeddings, p=2, dim=1)
            
            return content_embeddings
    
    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embedding for inference"""
        with torch.no_grad():
            user_embeddings = self.user_tower(user_ids)
            return user_embeddings


class ContrastiveLoss(nn.Module):
    """InfoNCE / NT-Xent loss for two-tower model"""
    
    def __init__(self, temperature: float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, similarity_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarity_scores: Tensor of shape (batch_size, batch_size)
        Returns:
            loss: Scalar tensor
        """
        batch_size = similarity_scores.size(0)
        
        # Labels are the diagonal (positive pairs)
        labels = torch.arange(batch_size).to(similarity_scores.device)
        
        # Compute loss (bi-directional)
        loss_user_to_content = self.criterion(similarity_scores, labels)
        loss_content_to_user = self.criterion(similarity_scores.T, labels)
        
        loss = (loss_user_to_content + loss_content_to_user) / 2
        
        return loss


class RankingLoss(nn.Module):
    """BPR (Bayesian Personalized Ranking) loss"""
    
    def __init__(self, margin: float = 1.0):
        super(RankingLoss, self).__init__()
        self.margin = margin
        
    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            positive_scores: Tensor of shape (batch_size,)
            negative_scores: Tensor of shape (batch_size, num_negatives)
        Returns:
            loss: Scalar tensor
        """
        # Expand positive scores to match negative scores shape
        positive_scores = positive_scores.unsqueeze(1)
        
        # BPR loss: -log(sigmoid(positive - negative))
        diff = positive_scores - negative_scores
        loss = -F.logsigmoid(diff).mean()
        
        return loss


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    num_users = 10000
    embedding_dim = 128
    
    # Create dummy data
    user_ids = torch.randint(0, num_users, (batch_size,))
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (batch_size, 32))
    attention_mask = torch.ones(batch_size, 32)
    
    # Initialize model
    model = TwoTowerModel(num_users=num_users, embedding_dim=embedding_dim)
    
    # Forward pass
    user_emb, content_emb, similarity = model(
        user_ids, images, input_ids, attention_mask
    )
    
    print(f"User embeddings shape: {user_emb.shape}")
    print(f"Content embeddings shape: {content_emb.shape}")
    print(f"Similarity scores shape: {similarity.shape}")
    
    # Test loss
    loss_fn = ContrastiveLoss()
    loss = loss_fn(similarity)
    print(f"Loss: {loss.item():.4f}")
