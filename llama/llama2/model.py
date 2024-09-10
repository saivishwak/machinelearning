from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

from .utils import apply_rotary_embeddings, precompute_theta_pos_frequencies, repeat_kv


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # Number of heads for Query
    n_kv_heads: Optional[int] = None  # Number of heads for Key and Value
    vocab_size: int = -1  # Set when tokenzier is loaded
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class SelfAttentionBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads

        self.n_rep = self.n_heads_q // self.n_kv_heads

        # Indicates the dim of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads *
                            self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> None:
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim) Only one token

        # (B, 1, Dim) -> (B, 1, H_Q * Head_dim)
        xq = self.wq(x)

        # (B, 1, Dim) -> (B, 1, H_KV * Head_dim)
        xk = self.wk(x)

        # (B, 1, Dim) -> (B, 1, H_KV * Head_dim)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_dim) -> (B, 1, H_Q, Head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (B, 1, H_KV * Head_dim) -> (B, 1, H_KV, Head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply Rotary Embedding to Q and K -> Does not change the shape of tensor
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in cache for this token
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        # Retreive all the cached keys and values so far
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        # Repeat the heads of the K and V to reach the number of heads of queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_dim) --> (B, H_q,1, Head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_q, 1, Head_dim) @ (B, H_q, Head_dim, seq_len_kv) -> (B, H_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / \
            math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_q, 1, seq_len) @ (B, H_q, seq_len_kv, head_dim) -> (B, H_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # (B, H_q, 1, Hea_dim) -> (B, 1, H_q, head_dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1))

        return self.wo(output)  # (B, 1, Dim) -> (B, 1, Dim)


class FeedforwardBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # Round the hidden_dim to the nearest multiplier of the multiple_of paramter
        hidden_dim = args.multiple_of * \
            ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> None:
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swish * x_v
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttentionBlock(args)
        self.feed_forward = FeedforwardBlock(args)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> None:
        # (B, Seq_len, Dim) + (B, Seq_len, Dim) = (B, Seq_len, Dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex)
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Gamma Param
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_len, Dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> None:
        # Dim * (B, Seq_len, Dim) = (B, Seq_len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size > 0, "Vocab size cannot be less than 0"

        self.args: ModelArgs = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # RotaryEmbedding
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    # tokens -> (Batch, Seq_len)
    def forward(self, tokens: torch.Tensor, start_pos: int) -> None:
        batch, seq_len = tokens.shape
        # this is because of KV cache, for training we have seq_len itself
        assert seq_len == 1, "Seq_len should be 1, i.e only one token should be sent as input"

        # (B, Seq_len) -> (B, Seq_len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the tokens [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        # Normalization
        h = self.norm(h)

        # Output -> (B, Seq_len, Vocab_size)
        output = self.output(h).float()
        return output
