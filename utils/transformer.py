import torch
import torch.nn as nn
from utils.attention import MultiHeadAttention
from utils.layers import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"]
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.drop_skip_layer = nn.Dropout(config["drop_rate"])

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_skip_layer(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_skip_layer(x)
        x = x + shortcut
        return x
    

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self,in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text(model,input,max_new_tokens,context_size):
    """
    Generate text using the model
    Args:
        model: The GPT model
        input: input of shape (batch_size, n_tokens)
        max_new_tokens: The maximum number of new tokens to generate
        context_size: The size of the context window
    Returns:
        The generated text
    """

    for _ in range(max_new_tokens):
        # if input is larger than context size, truncate the input
        input_chunk = input[:, -context_size:]
        with torch.no_grad():
            logits = model(input_chunk)
        # we need to get only the last token's logits to get the next token
        logits = logits[:, -1, :]
        # apply softmax to get the probabilities
        probs = torch.softmax(logits, dim=-1)
        # sample the next token
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        # append the next token to the input
        input = torch.cat([input, next_token], dim=1)
    return input