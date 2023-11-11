import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange


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


class Relative2dPositionBias(nn.Module):
    def __init__(
        self,
        scale,
        num_buckets=32,
        x_max_distance=50 * 18,
        y_max_distance=50 * 11,
    ):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.x_max_distance = x_max_distance
        self.y_max_distance = y_max_distance
        self.x_relative_attention_bias = nn.Embedding(num_buckets, 1)
        self.y_relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(relative_position, max_distance, num_buckets=32):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots, positions):
        x_positions = positions[:, :, 0]
        y_positions = positions[:, :, 1]

        rel_x_pos = x_positions[:, None, :] - x_positions[:, :, None]
        rel_x_pos = rel_x_pos.masked_fill_(x_positions[:, :, None] == -100, 0)
        rel_x_pos = rel_x_pos.masked_fill_(x_positions[:, None, :] == -100, 0)

        rel_y_pos = y_positions[:, None, :] - y_positions[:, :, None]
        rel_y_pos = rel_y_pos.masked_fill_(y_positions[:, :, None] == -100, 0)
        rel_y_pos = rel_y_pos.masked_fill_(y_positions[:, None, :] == -100, 0)

        rp_x_bucket = self._relative_position_bucket(
            rel_x_pos,
            num_buckets=self.num_buckets,
            max_distance=self.x_max_distance,
        )
        rp_y_bucket = self._relative_position_bucket(
            rel_y_pos,
            num_buckets=self.num_buckets,
            max_distance=self.y_max_distance,
        )
        x_values = self.x_relative_attention_bias(rp_x_bucket)
        y_values = self.y_relative_attention_bias(rp_y_bucket)

        x_bias = rearrange(x_values, "b i j h -> b (h i) j")
        y_bias = rearrange(y_values, "b i j h -> b (h i) j")
        return qk_dots + ((x_bias + y_bias) * self.scale)


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, embed_dim, x_max, y_max):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        dim_head = embed_dim // head_size

        self.relative_position_bias = Relative2dPositionBias(
            scale=dim_head**-0.5,
            x_max_distance=x_max,
            y_max_distance=y_max,
        )

    def forward(self, x, attn_mask, positions):
        # input of size (batch, time-step, channels=embedding size)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape

        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        wei = self.relative_position_bias(wei, positions)

        # Fills elements of self tensor with value where mask is one.
        wei = wei.masked_fill(attn_mask[:, :, None] == 1, -1e11)  # (B, T, T)

        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embed_dim, x_max, y_max):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, embed_dim, x_max, y_max) for _ in range(num_heads)]
        )
        self.projection = nn.Linear(
            embed_dim, embed_dim
        )  # projection for residual connection

    def forward(self, x, attn_mask, positions):
        out = torch.cat([h(x, attn_mask, positions) for h in self.heads], dim=-1)
        out = self.projection(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),  # projection for residual connection
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, head_size, num_heads, x_max, y_max) -> None:
        super().__init__()

        self.multihead_attention = MultiHeadAttention(
            num_heads, head_size, embed_dim, x_max, y_max
        )
        self.ffwd = FeedForward(embed_dim)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask, positions):
        out = self.layer_norm_1(x)
        out = out + self.multihead_attention(out, attn_mask, positions)

        out = self.layer_norm_2(out)
        out = out + self.ffwd(out)
        return out


class WolfBERT(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, dropout, classifier_count, x_max, y_max
    ):
        super().__init__()

        self.embedding = WolfBERTEmbedding(vocab_size, embed_dim, dropout)

        head_size = embed_dim // num_heads

        self.lm_head = nn.Linear(head_size * num_heads, vocab_size)
        self.sentence_classifier = nn.Linear(head_size * num_heads, classifier_count)

        self.transformer_block = TransformerBlock(
            embed_dim=embed_dim,
            head_size=head_size,
            num_heads=num_heads,
            x_max=x_max,
            y_max=y_max,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, sequence, attn_mask, masked_pos, positions):
        x = self.embedding(sequence)  # B, T, embedding_size

        x = self.transformer_block(
            x, attn_mask, positions
        )  # B, T, head_size * num_heads

        # masked token prediction
        masked_pos = masked_pos[:, :, None].expand(-1, -1, x.size(-1))
        x_masked = torch.gather(x, dim=1, index=masked_pos)
        logits_lm = self.lm_head(x_masked)  # B, T, vocab_size

        # bould grade prediction, by first token (CLS)
        logits_clsf = self.sentence_classifier(x[:, 0, :])  # B, classifier_count

        return logits_lm, logits_clsf
