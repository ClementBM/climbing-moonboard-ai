import math
import torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WolfBERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # T should be max_len
        B, T = x.shape

        embedding = self.token_embedding_table(x)

        # B, T, embed_dim
        embedding = self.drop(embedding)

        return self.norm(embedding)


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x, attn_mask):
        # input of size (batch, time-step, channels=embedding size)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape

        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        # Fills elements of self tensor with value where mask is one.
        wei = wei.masked_fill(attn_mask[:, :, None] == 1, -1e11)  # (B, T, T)

        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(num_heads)])

    def forward(self, x, attn_mask):
        return torch.cat([h(x, attn_mask) for h in self.heads], dim=-1)


class WolfBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, dropout, classifier_count):
        super().__init__()

        self.embedding = WolfBERTEmbedding(vocab_size, embed_dim, dropout)

        head_size = embed_dim // num_heads

        self.multi_head = MultiHeadAttention(num_heads, head_size, embed_dim)
        self.lm_head = nn.Linear(head_size * num_heads, vocab_size)
        self.sentence_classifier = nn.Linear(head_size * num_heads, classifier_count)

    def forward(self, sequence, attn_mask, masked_pos):
        x = self.embedding(sequence)  # B, T, embedding_size
        x = self.multi_head(x, attn_mask)  # B, T, head_size * num_heads

        # masked token prediction
        masked_pos = masked_pos[:, :, None].expand(-1, -1, x.size(-1))
        x_masked = torch.gather(x, dim=1, index=masked_pos)
        logits_lm = self.lm_head(x_masked)  # B, T, vocab_size

        # bould grade prediction, by first token (CLS)
        logits_clsf = self.sentence_classifier(x[:, 0, :])  # B, classifier_count

        return logits_lm, logits_clsf
