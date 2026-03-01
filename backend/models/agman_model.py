"""
AGMAN Model Architecture

Attention-Guided Multi-Attribute Network for fashion embedding refinement.
Refines ResNet50 embeddings (2048D) to discriminative embeddings (512D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AGMAN(nn.Module):
   
    def __init__(self, in_dim=2048, emb_dim=512, num_classes=None, dropout=0.15):
        super().__init__()
        
        # Attention module - learns feature importance
        self.attn = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, in_dim),
            nn.Sigmoid()  # Attention weights [0, 1]
        )
        
        # Projection head - dimension reduction
        self.fc = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )
        
        # Optional classification head (used during training)
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x, return_logits=False):
        """
        Forward pass
        
        Args:
            x (Tensor): Input embeddings [batch_size, in_dim]
            return_logits (bool): Whether to return classification logits
            
        Returns:
            Tensor: L2-normalized embeddings [batch_size, emb_dim]
            OR
            Tuple[Tensor, Tensor]: (normalized embeddings, logits) if return_logits=True
        """
        # Apply attention - element-wise multiplication
        attn_weights = self.attn(x)
        x = x * attn_weights
        
        # Project to embedding space
        x = self.fc(x)
        
        # L2 normalize for cosine similarity
        x_norm = F.normalize(x, p=2, dim=1)
        
        # Optionally return classification logits
        if return_logits and self.classifier is not None:
            logits = self.classifier(x)  # Use un-normalized for classification
            return x_norm, logits
        
        return x_norm


def load_agman_model(checkpoint_path, device='cpu'):
    """
    Helper function to load a trained AGMAN model
    
    Args:
        checkpoint_path (str): Path to model checkpoint (.pth file)
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        AGMAN: Loaded model in eval mode
    """
    model = AGMAN(in_dim=2048, emb_dim=512)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
