"""What does jnp.var lower to that produces NaN?

jnp.var(x, axis=-1, keepdims=True) gives NaN for 127/128 rows on
MetalHLO. Manual mean((x-m)^2) gives the same answer correctly.
"""
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
    has_nan_c, has_nan_g = np.any(np.isnan(c)), np.any(np.isnan(g))
    d = float(np.max(np.abs(c - g))) if not has_nan_g else float("nan")
    fin = np.isfinite(g).sum()
    print(f"{name:60s}  shape={c.shape}  max|diff|={d}  nan_gpu={has_nan_g}  finite={fin}/{g.size}")

rng = np.random.default_rng(0)
B, T, C = 2, 64, 128
x = rng.standard_normal((B, T, C)).astype(np.float32)

print("--- jaxpr of jnp.var ---")
print(jax.make_jaxpr(lambda x: x.var(axis=-1, keepdims=True))(x))
print()

# Test various ways to compute (x - mean)**2 reduction
def m1(x):
    m = jnp.mean(x, axis=-1, keepdims=True)
    return jnp.mean((x - m) * (x - m), axis=-1, keepdims=True)
def m2(x):
    m = jnp.mean(x, axis=-1, keepdims=True)
    return jnp.mean((x - m)**2, axis=-1, keepdims=True)
def m3(x):
    m = jnp.mean(x, axis=-1, keepdims=True)
    c = x - m
    return jnp.mean(jax.lax.square(c), axis=-1, keepdims=True)
def m4(x):
    m = jnp.mean(x, axis=-1, keepdims=True)
    c = x - m
    return jnp.mean(jax.lax.integer_pow(c, 2), axis=-1, keepdims=True)
def m5(x):
    return jnp.var(x, axis=-1, keepdims=True)

diff("m1 mean((x-m)*(x-m))", m1, x)
diff("m2 mean((x-m)**2)", m2, x)
diff("m3 mean(lax.square(x-m))", m3, x)
diff("m4 mean(lax.integer_pow(x-m, 2))", m4, x)
diff("m5 jnp.var (the buggy one)", m5, x)

# What if we test lax.square in isolation?
def f_sq_mul(x): return x * x
def f_sq_lax(x): return jax.lax.square(x)
def f_sq_pow_int(x): return jax.lax.integer_pow(x, 2)
diff("sq: x * x", f_sq_mul, x)
diff("sq: lax.square(x)", f_sq_lax, x)
diff("sq: lax.integer_pow(x, 2)", f_sq_pow_int, x)

# reduce mean over axis -1 with keepdims of squared
def red_sq(x):
    return jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
def red_sq_mul(x):
    return jnp.mean(x * x, axis=-1, keepdims=True)
diff("reduce_mean(lax.square(x), -1)", red_sq, x)
diff("reduce_mean(x*x, -1)", red_sq_mul, x)
