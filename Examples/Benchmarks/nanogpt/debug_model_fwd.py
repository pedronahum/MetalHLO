"""Forward-only check of the actual nanoGPT model on tiny inputs.

Builds the SAME GPT module as the benchmark, but with batch=1, seq=8,
n_layer=1, n_embd=32. Compares CPU vs MetalHLO forward output. If
output differs or contains NaN we know there's still a bug — but
debugging happens in seconds, not minutes.
"""
import os, sys
import jax._src.xla_bridge as xb
import jaxlib.xla_client as xla_client
if not xla_client.pjrt_plugin_loaded("metalhlo"):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    xb.register_plugin(
        "metalhlo", priority=500,
        library_path=os.path.join(repo_root, ".build", "release", "libPJRTMetalHLO.dylib"),
        options=None,
    )

import jax, jax.numpy as jnp, numpy as np
from flax import nnx

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
from model import GPT

cpu = jax.devices("cpu")[0]; gpu = jax.devices("metalhlo")[0]

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--vocab", type=int, default=16)
ap.add_argument("--n-layer", type=int, default=1)
ap.add_argument("--n-embd", type=int, default=32)
ap.add_argument("--n-head", type=int, default=4)
ap.add_argument("--block-size", type=int, default=8)
ap.add_argument("--batch", type=int, default=1)
opt = ap.parse_args()
VOCAB, N_LAYER, N_EMBD, N_HEAD, BLOCK_SIZE, B = opt.vocab, opt.n_layer, opt.n_embd, opt.n_head, opt.block_size, opt.batch
print(f"config: vocab={VOCAB} n_layer={N_LAYER} n_embd={N_EMBD} n_head={N_HEAD} block={BLOCK_SIZE} batch={B}")

def make_model(device):
    with jax.default_device(device):
        return GPT(
            vocab_size=VOCAB, n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
            block_size=BLOCK_SIZE, rngs=nnx.Rngs(0),
        )

# Build TWO models so each lives entirely on one device. Init RNG is the
# same (seed=0) so weights should be bit-equal.
m_cpu = make_model(cpu)
m_gpu = make_model(gpu)

# Quick param check: make sure same weights
def first_param(m):
    leaves = jax.tree.leaves(nnx.state(m, nnx.Param))
    arr = next(l for l in leaves if hasattr(l, "shape"))
    return np.asarray(arr)
print(f"first param diff: {np.max(np.abs(first_param(m_cpu) - first_param(m_gpu))):.3e}")

rng = np.random.default_rng(0)
ids_np = rng.integers(0, VOCAB, size=(B, BLOCK_SIZE), dtype=np.int32)

@nnx.jit
def fwd(model, ids):
    return model(ids)

out_cpu = np.asarray(fwd(m_cpu, jax.device_put(jnp.asarray(ids_np), cpu)))
print(f"CPU output shape={out_cpu.shape}  range=[{out_cpu.min():.4f},{out_cpu.max():.4f}]  nan={np.any(np.isnan(out_cpu))}")
out_gpu = np.asarray(fwd(m_gpu, jax.device_put(jnp.asarray(ids_np), gpu)))
print(f"GPU output shape={out_gpu.shape}  range=[{out_gpu.min():.4f},{out_gpu.max():.4f}]  nan={np.any(np.isnan(out_gpu))}")
print(f"max |diff| = {np.max(np.abs(out_cpu - out_gpu)):.4e}")
if np.any(np.isnan(out_gpu)):
    print(f"GPU finite count: {np.isfinite(out_gpu).sum()}/{out_gpu.size}")
