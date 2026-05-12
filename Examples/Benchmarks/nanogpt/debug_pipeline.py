"""Trace one block's forward pass piece by piece at B=4 to find where NaN
first appears. Uses fixed weights so CPU↔GPU comparison is exact."""
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
B, T, C = 4, 8, 32      # batch=4 triggers the bug
H, HD = 4, 8

x = rng.standard_normal((B, T, C)).astype(np.float32)
g1 = rng.standard_normal((C,)).astype(np.float32) * 0.1 + 1.0
b1 = rng.standard_normal((C,)).astype(np.float32) * 0.1
g2 = rng.standard_normal((C,)).astype(np.float32) * 0.1 + 1.0
b2 = rng.standard_normal((C,)).astype(np.float32) * 0.1
wqkv = rng.standard_normal((C, 3*C)).astype(np.float32) * 0.02
wo = rng.standard_normal((C, C)).astype(np.float32) * 0.02
wfc = rng.standard_normal((C, 4*C)).astype(np.float32) * 0.02
wp = rng.standard_normal((4*C, C)).astype(np.float32) * 0.02
mask_np = np.tril(np.ones((T, T), dtype=np.bool_))

def ln(x, g, b):
    m = jnp.mean(x, axis=-1, keepdims=True)
    v = jnp.var(x, axis=-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + 1e-5) + b

def attn(x, w):
    qkv = x @ w
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    s = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.float32(HD))
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    s = jnp.where(mask, s, jnp.float32(-1e30))
    a = jax.nn.softmax(s, axis=-1)
    y = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
    return y

def attn_proj(x, wqkv, wo):
    return attn(x, wqkv) @ wo

# Stages
diff("01 ln1(x)", lambda x, g, b: ln(x, g, b), x, g1, b1)

def stage2(x, g, b, w):
    return attn(ln(x, g, b), w)
diff("02 attn(ln1(x))", stage2, x, g1, b1, wqkv)

def stage3(x, g, b, w, wo):
    return attn(ln(x, g, b), w) @ wo
diff("03 proj(attn(ln1(x)))", stage3, x, g1, b1, wqkv, wo)

def stage4(x, g, b, w, wo):
    return x + attn(ln(x, g, b), w) @ wo
diff("04 x + proj(attn(ln1(x)))   [residual]", stage4, x, g1, b1, wqkv, wo)

def stage5(x, g1, b1, w, wo, g2, b2):
    h = x + attn(ln(x, g1, b1), w) @ wo
    return ln(h, g2, b2)
diff("05 ln2 after residual", stage5, x, g1, b1, wqkv, wo, g2, b2)

def stage6(x, g1, b1, w, wo, g2, b2, wfc, wp):
    h = x + attn(ln(x, g1, b1), w) @ wo
    m = jax.nn.gelu(ln(h, g2, b2) @ wfc) @ wp
    return h + m
diff("06 full block (LN+attn+res+LN+MLP+res)", stage6, x, g1, b1, wqkv, wo, g2, b2, wfc, wp)
