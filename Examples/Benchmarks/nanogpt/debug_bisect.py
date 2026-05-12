"""Bisect-find which piece of the nanoGPT forward breaks on MetalHLO.

Runs each fragment on JAX CPU AND MetalHLO with the same input and prints
max abs diff. We expect ~zero diff (fp drift only). A blown-up diff
identifies the broken op.
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=int, default=64, help="Sequence length (smaller = faster bisect)")
parser.add_argument("--bsz", type=int, default=2, help="Batch size")
parser.add_argument("--n-embd", type=int, default=128)
parser.add_argument("--n-head", type=int, default=4)
args = parser.parse_args()

# Register MetalHLO BEFORE any jax imports
import jax._src.xla_bridge as xb
import jaxlib.xla_client as xla_client

if not xla_client.pjrt_plugin_loaded("metalhlo"):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    lib = os.path.join(repo_root, ".build", "release", "libPJRTMetalHLO.dylib")
    if not os.path.isfile(lib):
        print("ERROR: MetalHLO plugin not built"); sys.exit(1)
    xb.register_plugin("metalhlo", priority=500, library_path=lib, options=None)

import jax
import jax.numpy as jnp
import numpy as np

cpu = jax.devices("cpu")[0]
gpu = jax.devices("metalhlo")[0]

B, T = args.bsz, args.seq
C = args.n_embd
H = args.n_head
HD = C // H

rng = np.random.default_rng(0)
x_np = rng.standard_normal((B, T, C)).astype(np.float32)
mask_np = np.tril(np.ones((T, T), dtype=np.bool_))

def run_on(device, fn, *arrs):
    with jax.default_device(device):
        outs = jax.jit(fn)(*[jnp.asarray(a) for a in arrs])
    return np.asarray(outs)

def diff(name, fn, *arrs):
    out_cpu = run_on(cpu, fn, *arrs)
    out_gpu = run_on(gpu, fn, *arrs)
    d = np.max(np.abs(out_cpu - out_gpu))
    has_nan_gpu = np.any(np.isnan(out_gpu))
    print(f"{name:35s}  max|diff|={d:.4e}  nan_gpu={has_nan_gpu}  cpu_range=[{out_cpu.min():.3f},{out_cpu.max():.3f}]  gpu_range=[{out_gpu.min():.3f},{out_gpu.max():.3f}]")

# ─── Tests, in order from simplest to fullest ─────────────────────────

# 1) Just a matmul — sanity
def t_matmul(x, w): return x @ w
w1 = rng.standard_normal((C, 3*C)).astype(np.float32) * 0.02
diff("01 matmul x @ qkv_w", t_matmul, x_np, w1)

# 2) qkv split + reshape + transpose
def t_qkv(x, w):
    qkv = x @ w  # (B, T, 3C)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, H, HD).transpose(0, 2, 1, 3)  # (B, H, T, HD)
    k = k.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    return q, k, v
def t_qkv_q(x, w): return t_qkv(x, w)[0]
diff("02 qkv split+reshape+transpose Q", t_qkv_q, x_np, w1)

# 3) Q @ K^T scaled
def t_scores(x, w):
    q, k, _ = t_qkv(x, w)
    return (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.float32(HD))
diff("03 attention scores Q@K^T", t_scores, x_np, w1)

# 4) Causal mask via jnp.where with -1e30
def t_masked_scores(x, w, mask):
    s = t_scores(x, w)
    return jnp.where(mask, s, jnp.float32(-1e30))
diff("04 masked scores (-1e30)", t_masked_scores, x_np, w1, mask_np)

# 5) Softmax of masked scores
def t_attn(x, w, mask):
    s = t_masked_scores(x, w, mask)
    return jax.nn.softmax(s, axis=-1)
diff("05 softmax(masked scores)", t_attn, x_np, w1, mask_np)

# 6) attn @ V
def t_attn_v(x, w, mask):
    a = t_attn(x, w, mask)
    _, _, v = t_qkv(x, w)
    return a @ v
diff("06 attn @ V", t_attn_v, x_np, w1, mask_np)

# 7) Full attention output (reshape back)
def t_full_attn(x, w, mask):
    out = t_attn_v(x, w, mask)  # (B, H, T, HD)
    return out.transpose(0, 2, 1, 3).reshape(B, T, C)
diff("07 attention out (B,T,C)", t_full_attn, x_np, w1, mask_np)

# 8) GELU
def t_gelu(x): return jax.nn.gelu(x)
diff("08 gelu(x)", t_gelu, x_np)

# 9) LayerNorm
def t_ln(x):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)
diff("09 manual layernorm", t_ln, x_np)
