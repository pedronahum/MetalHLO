"""Same B=4 attention, but input is LN(x). Does the LN output break attention?"""
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
x = rng.standard_normal((B, T, C)).astype(np.float32) * 0.5
g = rng.standard_normal((C,)).astype(np.float32) * 0.1 + 1.0
b = rng.standard_normal((C,)).astype(np.float32) * 0.1
w = rng.standard_normal((C, 3*C)).astype(np.float32) * 0.02
mask_np = np.tril(np.ones((T, T), dtype=np.bool_))

def manual_ln(x, g, b):
    m = jnp.mean(x, axis=-1, keepdims=True)
    v = jnp.var(x, axis=-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + 1e-5) + b

# Materialize LN on CPU first, then pass to attention separately
ln_out_cpu = run(cpu, manual_ln, x, g, b)
print(f"\nLN output stats: range=[{ln_out_cpu.min():.3f},{ln_out_cpu.max():.3f}] "
      f"mean={ln_out_cpu.mean():.3f} std={ln_out_cpu.std():.3f}\n")

def attn(x_in, w):
    qkv = x_in @ w
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, H, HD).transpose(0, 2, 1, 3)
    s = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.float32(HD))
    m = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    s = jnp.where(m, s, jnp.float32(-1e30))
    a = jax.nn.softmax(s, axis=-1)
    return (a @ v).transpose(0, 2, 1, 3).reshape(B, T, C)

# Case A: attention on LN output (pre-computed, passed as input)
diff("A: attn(precomputed_LN_output)", attn, ln_out_cpu, w)

# Case B: attention on raw x
diff("B: attn(raw_x)", attn, x, w)

# Case C: composed in one JIT — LN then attention
def composed(x, g, b, w):
    h = manual_ln(x, g, b)
    return attn(h, w)
diff("C: attn(LN(x))  composed in one JIT", composed, x, g, b, w)

# Case D: same composition but with a stop_gradient between LN and attn
# (mostly a no-op semantically — but might change fusion / scheduling)
def composed_stop(x, g, b, w):
    h = jax.lax.stop_gradient(manual_ln(x, g, b))
    return attn(h, w)
diff("D: attn(stop_gradient(LN(x)))", composed_stop, x, g, b, w)
