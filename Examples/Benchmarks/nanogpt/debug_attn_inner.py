"""Drill into the attention block at B=4 — find which inner step NaNs."""
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
B, T, C = 4, 8, 32
H, HD = 4, 8

x = rng.standard_normal((B, T, C)).astype(np.float32) * 0.5  # smaller-magnitude
w = rng.standard_normal((C, 3*C)).astype(np.float32) * 0.02

# Same flow as the attention block, but each step returned to inspect

def fn_qkv_full(x, w): return x @ w                                # (B, T, 3C)
def fn_q(x, w): return jnp.split(x @ w, 3, axis=-1)[0]              # (B, T, C)
def fn_k(x, w): return jnp.split(x @ w, 3, axis=-1)[1]
def fn_v(x, w): return jnp.split(x @ w, 3, axis=-1)[2]
def fn_q_r(x, w):
    return jnp.split(x @ w, 3, axis=-1)[0].reshape(B, T, H, HD).transpose(0, 2, 1, 3)
def fn_k_r(x, w):
    return jnp.split(x @ w, 3, axis=-1)[1].reshape(B, T, H, HD).transpose(0, 2, 1, 3)
def fn_qk(x, w):
    q = jnp.split(x @ w, 3, axis=-1)[0].reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    k = jnp.split(x @ w, 3, axis=-1)[1].reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    return (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.float32(HD))
def fn_qk_masked(x, w):
    s = fn_qk(x, w)
    m = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    return jnp.where(m, s, jnp.float32(-1e30))
def fn_softmax(x, w):
    return jax.nn.softmax(fn_qk_masked(x, w), axis=-1)
def fn_av(x, w):
    a = fn_softmax(x, w)
    v = jnp.split(x @ w, 3, axis=-1)[2].reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    return a @ v
def fn_out(x, w):
    return fn_av(x, w).transpose(0, 2, 1, 3).reshape(B, T, C)

diff("01 qkv_concat (B,T,3C)", fn_qkv_full, x, w)
diff("02 Q raw (split[0])", fn_q, x, w)
diff("03 K raw (split[1])", fn_k, x, w)
diff("04 V raw (split[2])", fn_v, x, w)
diff("05 Q reshape+transpose", fn_q_r, x, w)
diff("06 K reshape+transpose", fn_k_r, x, w)
diff("07 Q @ K^T scaled", fn_qk, x, w)
diff("08 + causal mask -1e30", fn_qk_masked, x, w)
diff("09 + softmax", fn_softmax, x, w)
diff("10 + (softmax @ V)", fn_av, x, w)
diff("11 + transpose + reshape (final)", fn_out, x, w)
