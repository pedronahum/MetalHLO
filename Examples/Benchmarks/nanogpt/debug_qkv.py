"""Pinpoint the Q@K^T diff: is K from split wrong, or is the fused pattern broken?"""
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

cpu = jax.devices("cpu")[0]; gpu = jax.devices("metalhlo")[0]
def run(dev, fn, *arrs):
    with jax.default_device(dev):
        return np.asarray(jax.jit(fn)(*[jnp.asarray(a) for a in arrs]))

def diff(name, fn, *arrs):
    c, g = run(cpu, fn, *arrs), run(gpu, fn, *arrs)
    d = np.max(np.abs(c - g))
    print(f"{name:55s}  shape={c.shape}  max|diff|={d:.4e}  range=[{c.min():.3f},{c.max():.3f}]")

rng = np.random.default_rng(0)
B, T, C, H, HD = 2, 64, 128, 4, 32
x = rng.standard_normal((B, T, C)).astype(np.float32)
w = (rng.standard_normal((C, 3 * C)) * 0.02).astype(np.float32)

# Stage by stage, returning each piece in isolation
def fn_qkv_concat(x, w): return x @ w                              # (B, T, 3C)
def fn_q_raw(x, w): return jnp.split(x @ w, 3, axis=-1)[0]         # (B, T, C)
def fn_k_raw(x, w): return jnp.split(x @ w, 3, axis=-1)[1]         # (B, T, C)
def fn_v_raw(x, w): return jnp.split(x @ w, 3, axis=-1)[2]         # (B, T, C)
def fn_q_resh(x, w):
    return jnp.split(x @ w, 3, axis=-1)[0].reshape(B, T, H, HD).transpose(0, 2, 1, 3)
def fn_k_resh(x, w):
    return jnp.split(x @ w, 3, axis=-1)[1].reshape(B, T, H, HD).transpose(0, 2, 1, 3)
def fn_kT(x, w):
    k = jnp.split(x @ w, 3, axis=-1)[1].reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    return k.transpose(0, 1, 3, 2)
def fn_qkT(x, w):
    qkv = x @ w
    q, k, _ = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    return q @ k.transpose(0, 1, 3, 2)
# Variant: compute Q and K separately (no split), with two matmuls
def fn_qkT_two_mm(x, wq, wk):
    q = (x @ wq).reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    k = (x @ wk).reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    return q @ k.transpose(0, 1, 3, 2)
wq, wk = w[:, :C], w[:, C:2*C]

diff("00 concat qkv = x @ w", fn_qkv_concat, x, w)
diff("01 Q raw (split slot 0)", fn_q_raw, x, w)
diff("02 K raw (split slot 1)", fn_k_raw, x, w)
diff("03 V raw (split slot 2)", fn_v_raw, x, w)
diff("04 Q reshaped + transposed", fn_q_resh, x, w)
diff("05 K reshaped + transposed", fn_k_resh, x, w)
diff("06 K^T (final, ready for matmul)", fn_kT, x, w)
diff("07 Q @ K^T (the big diff from earlier)", fn_qkT, x, w)
diff("08 same but two separate matmuls for Q, K", fn_qkT_two_mm, x, wq, wk)
