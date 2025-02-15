"""
This module implements core transformer layers including the feed-forward network
and layer normalization components.
"""

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks.
    
    This implementation follows the original transformer architecture:
    - Two linear layers with GELU activation in between
    - Hidden dimension is 4x the input dimension
    - Includes dropout for regularization
    
    Args:
        config (dict): Configuration dictionary containing:
            - emb_dim (int): Input and output dimension
            - drop_rate (float): Dropout probability
    """
    
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),  # Expansion
            nn.GELU(),  # Activation
            nn.Dropout(config["drop_rate"]),  # Added dropout for regularization
            nn.Linear(4 * config["emb_dim"], config["emb_dim"])  # Compression
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward network to the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim)
        """
        return self.layers(x)

class LayerNorm(nn.Module):
    """
    Layer normalization module.
    
    Normalizes the input tensor across the last dimension (feature dimension)
    and applies learnable scale and shift parameters.
    
    Args:
        emb_dim (int): Feature dimension to normalize over
    """
    
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # Made into proper Parameter
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Made into proper Parameter
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim)
            
        Returns:
            torch.Tensor: Normalized tensor of the same shape
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)  # unbiased=False uses N instead of N-1
        return self.scale * (x - mean) / (std + self.eps) + self.shift 