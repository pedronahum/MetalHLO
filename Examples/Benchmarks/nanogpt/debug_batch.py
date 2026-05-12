"""Find which op produces NaN at B>=4 but is fine at B=3."""
import os
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

cpu = jax.devices("cpu")[0]; gpu = jax.devices("metalhlo")[0]
def run(dev, fn, *arrs):
    with jax.default_device(dev):
        return np.asarray(jax.jit(fn)(*[jnp.asarray(a) for a in arrs]))
def diff(name, fn, *arrs):
    c = run(cpu, fn, *arrs); g = run(gpu, fn, *arrs)
    nan_g = np.any(np.isnan(g)); fin = int(np.isfinite(g).sum())
    d = float(np.max(np.abs(c - g))) if not nan_g else float("nan")
    print(f"{name:60s}  shape={c.shape}  diff={d}  nan={nan_g}  finite={fin}/{g.size}")

rng = np.random.default_rng(0)

# 1) embedding: indices (B, T), table (V, C) -> (B, T, C)
V, C = 16, 32
T = 8
for B in [3, 4]:
    idx = rng.integers(0, V, size=(B, T), dtype=np.int32)
    table = rng.standard_normal((V, C)).astype(np.float32)
    def fn(idx, table): return table[idx]
    diff(f"01 embed gather B={B}", fn, idx, table)

# 2) embed + add
for B in [3, 4]:
    idx = rng.integers(0, V, size=(B, T), dtype=np.int32)
    table_t = rng.standard_normal((V, C)).astype(np.float32)
    table_p = rng.standard_normal((T, C)).astype(np.float32)
    def fn(idx, t, p): return t[idx] + p[jnp.arange(T)]
    diff(f"02 embed + pos_embed B={B}", fn, idx, table_t, table_p)

# 3) LayerNorm only (gamma+beta as 1D param)
for B in [3, 4]:
    x = rng.standard_normal((B, T, C)).astype(np.float32)
    g = rng.standard_normal((C,)).astype(np.float32) * 0.1 + 1.0
    b = rng.standard_normal((C,)).astype(np.float32) * 0.1
    def fn(x, g, b):
        m = jnp.mean(x, axis=-1, keepdims=True)
        v = jnp.var(x, axis=-1, keepdims=True)
        return g * (x - m) / jnp.sqrt(v + 1e-5) + b
    diff(f"03 manual LN with gamma/beta B={B}", fn, x, g, b)

# 4) 3D matmul (B,T,C) @ (C,3C)
for B in [3, 4]:
    x = rng.standard_normal((B, T, C)).astype(np.float32)
    w = rng.standard_normal((C, 3*C)).astype(np.float32) * 0.02
    def fn(x, w): return x @ w
    diff(f"04 3D matmul (B,T,C)@(C,3C) B={B}", fn, x, w)

# 5) 4D matmul attention scores
H, HD = 4, 8
for B in [3, 4]:
    q = rng.standard_normal((B, H, T, HD)).astype(np.float32)
    k = rng.standard_normal((B, H, T, HD)).astype(np.float32)
    def fn(q, k): return q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(jnp.float32(HD))
    diff(f"05 Q@K^T scaled B={B}", fn, q, k)

# 6) softmax with -1e30 mask
for B in [3, 4]:
    s = rng.standard_normal((B, H, T, T)).astype(np.float32)
    m = np.tril(np.ones((T, T), dtype=np.bool_))
    def fn(s, m): return jax.nn.softmax(jnp.where(m, s, jnp.float32(-1e30)), axis=-1)
    diff(f"06 softmax masked B={B}", fn, s, m)
