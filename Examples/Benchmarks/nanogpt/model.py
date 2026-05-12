"""nanoGPT-style decoder-only transformer in Flax NNX.

Mirrors Karpathy's nanoGPT layout: pre-LayerNorm blocks, causal
multi-head self-attention, MLP with GELU, no biases (matching the GPT-2
"no-bias" convention). Sized for char-level tinyshakespeare:
  n_layer=6, n_embd=384, n_head=6, head_dim=64, block_size=256.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention.

    A single fused QKV projection (one matmul instead of three) — same
    shape as nanoGPT's `c_attn`.
    """

    def __init__(self, n_embd: int, n_head: int, *, rngs: nnx.Rngs):
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nnx.Linear(n_embd, 3 * n_embd, use_bias=False, rngs=rngs)
        self.proj = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.float32(self.head_dim))
        # Lower-triangular causal mask
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        scores = jnp.where(mask, scores, jnp.float32(-1e30))
        attn = jax.nn.softmax(scores, axis=-1)
        y = attn @ v  # (B, n_head, T, head_dim)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(y)


class MLP(nnx.Module):
    """Two-layer feedforward block with GELU activation (4× hidden expansion)."""

    def __init__(self, n_embd: int, *, rngs: nnx.Rngs):
        self.fc = nnx.Linear(n_embd, 4 * n_embd, use_bias=False, rngs=rngs)
        self.proj = nnx.Linear(4 * n_embd, n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.proj(nnx.gelu(self.fc(x)))


class Block(nnx.Module):
    """Pre-norm transformer block (LayerNorm before attention / MLP)."""

    def __init__(self, n_embd: int, n_head: int, *, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.attn = CausalSelfAttention(n_embd, n_head, rngs=rngs)
        self.ln2 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.mlp = MLP(n_embd, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nnx.Module):
    """nanoGPT: token + position embeddings → N pre-norm blocks → LN → lm_head."""

    def __init__(
        self,
        *,
        vocab_size: int,
        n_layer: int,
        n_embd: int,
        n_head: int,
        block_size: int,
        rngs: nnx.Rngs,
    ):
        self.block_size = block_size
        self.wte = nnx.Embed(vocab_size, n_embd, rngs=rngs)
        self.wpe = nnx.Embed(block_size, n_embd, rngs=rngs)
        # nnx.data wraps a python list of submodules so the NNX pytree
        # accounting treats their parameters as graph data, not static config.
        self.blocks = nnx.data([Block(n_embd, n_head, rngs=rngs) for _ in range(n_layer)])
        self.ln_f = nnx.LayerNorm(n_embd, rngs=rngs)
        self.lm_head = nnx.Linear(n_embd, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, idx: jax.Array) -> jax.Array:
        B, T = idx.shape
        pos = jnp.arange(T)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
