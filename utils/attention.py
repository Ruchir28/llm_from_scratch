"""
This module implements an efficient multi-head attention mechanism as used in transformer architectures.
The implementation supports causal (auto-regressive) attention with an optional dropout for regularization.
"""

import torch
import torch.nn as nn
from typing import Optional

class MultiHeadAttention(nn.Module):
    """
    Efficient implementation of Multi-Head Attention with causal masking.
    
    This implementation performs attention operations for all heads in parallel,
    making it more efficient than separate attention heads.
    
    Args:
        d_in (int): Input dimension
        d_out (int): Output dimension (must be divisible by num_heads)
        context_length (int): Maximum sequence length for attention
        dropout (float): Dropout probability (0.0 to 1.0)
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): Whether to include bias in query, key, value projections. Defaults to False.
    """
    
    def __init__(self, d_in: int, d_out: int, context_length: int, 
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.output_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head attention over the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in)
            
        Returns:
            torch.Tensor: Attention output of shape (batch_size, num_tokens, d_out)
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split d_out dimension into num_heads and head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Group by heads
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)
        context_vector = attention_weights @ values

        # Reshape back to concatenated form
        context_vector = context_vector.transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.output_proj(context_vector)
        
        return context_vector 