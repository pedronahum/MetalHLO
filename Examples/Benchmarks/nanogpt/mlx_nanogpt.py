"""nanoGPT training benchmark — MLX (Apple GPU) reference.

Mirrors the JAX/MetalHLO model in model.py + main.py exactly:
6-layer / 384-dim / 6-head GPT, pre-norm blocks, fused QKV, GELU MLP,
no biases on the linears, LayerNorm with affine, char-level tinyshakespeare,
batch=16, block_size=256, Adam lr=3e-4. Same RandomBatchSampler + tokenizer.

Timing matches main.py: per-step wall time with a full device sync each step
(mx.eval of params + optimizer state + loss, then loss.item()), discarding the
first half of steps as warmup and averaging the rest.

    python mlx_nanogpt.py --steps 40
"""

import argparse
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from data import RandomBatchSampler, build_tokenizer, download, encode  # noqa: E402

N_LAYER = 6
N_EMBD = 384
N_HEAD = 6
BLOCK_SIZE = 256


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.head_dim, mx.float32))
        mask = mx.tril(mx.ones((T, T), dtype=mx.bool_))
        scores = mx.where(mask, scores, mx.array(-1e30, mx.float32))
        attn = mx.softmax(scores, axis=-1)
        y = attn @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        return self.proj(nn.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, block_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        pos = mx.arange(T)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    print(f"Backend: mlx  Device: {mx.default_device()}")
    text = download(os.path.join(HERE, "input.txt"))
    stoi, itos = build_tokenizer(text)
    vocab_size = len(itos)
    tokens = encode(text, stoi)
    print(f"Loaded tinyshakespeare: {len(text):,} chars, {vocab_size}-token vocab")
    sampler = RandomBatchSampler(tokens, batch_size=args.batch_size, block_size=BLOCK_SIZE, seed=0)

    model = GPT(vocab_size, N_LAYER, N_EMBD, N_HEAD, BLOCK_SIZE)
    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"num params: {n_params:,}")

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    def loss_fn(model, x, y):
        logits = model(x)
        B, T, V = logits.shape
        return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, V), y.reshape(-1)))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def train_step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        return loss

    print(f"Training for {args.steps} steps "
          f"(batch={args.batch_size}, seq={BLOCK_SIZE}, n_layer={N_LAYER}, n_embd={N_EMBD})...")
    times = []
    final_loss = float("nan")
    train_start = time.perf_counter()
    for step in range(args.steps):
        xin, yin = sampler.sample()
        x = mx.array(xin)
        y = mx.array(yin)
        mx.eval(x, y)
        step_start = time.perf_counter()
        loss = train_step(x, y)
        mx.eval(model.parameters(), optimizer.state, loss)
        l = loss.item()
        times.append(time.perf_counter() - step_start)
        final_loss = l
        if (step + 1) % max(1, args.steps // 10) == 0 or step + 1 == args.steps:
            print(f"step {step+1:4d} / {args.steps:4d} | loss {l:.4f}", flush=True)

    train_elapsed = time.perf_counter() - train_start
    steady = times[len(times) // 2:]
    mean_steady = sum(steady) / max(1, len(steady))
    print(f"\ntraining: {args.steps} steps in {train_elapsed:.2f}s "
          f"(mean step over last {len(steady)} steps: {mean_steady*1000:.1f}ms)")
    print(f"final training loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()
