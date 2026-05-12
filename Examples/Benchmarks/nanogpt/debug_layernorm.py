"""Bisect the manual-LayerNorm NaN on MetalHLO.

We saw `(x - mean) / sqrt(var + 1e-5)` produce NaN on a random-gaussian
3D input. Each piece is small enough to test in isolation — narrow
which op is producing the NaN.
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
    d = np.max(np.abs(c - g)) if not np.any(np.isnan(g)) else float("nan")
    has_nan_c = np.any(np.isnan(c)); has_nan_g = np.any(np.isnan(g))
    print(f"{name:60s}  shape={c.shape}  max|diff|={d!r}  nan_cpu={has_nan_c} nan_gpu={has_nan_g}")
    if has_nan_g and not has_nan_c:
        # Show GPU output stats to diagnose
        finite_mask = np.isfinite(g)
        print(f"   GPU output: shape={g.shape}  finite_count={finite_mask.sum()}/{g.size}  "
              f"finite_range=[{g[finite_mask].min() if finite_mask.any() else 'nan'},"
              f"{g[finite_mask].max() if finite_mask.any() else 'nan'}]  "
              f"CPU range=[{c.min():.4f},{c.max():.4f}]")

rng = np.random.default_rng(0)
# Same shape as the bisect input
B, T, C = 2, 64, 128
x = rng.standard_normal((B, T, C)).astype(np.float32)

# 1) just mean
diff("01 mean(x, axis=-1, keepdims=True)",
     lambda x: x.mean(axis=-1, keepdims=True), x)

# 2) just variance (default jnp.var which is mean((x-mean)^2))
diff("02 var(x, axis=-1, keepdims=True)",
     lambda x: x.var(axis=-1, keepdims=True), x)

# 3) variance computed via E[x^2] - E[x]^2 (cancellation-prone)
diff("03 var via E[x^2]-E[x]^2",
     lambda x: jnp.mean(x*x, axis=-1, keepdims=True) - jnp.mean(x, axis=-1, keepdims=True)**2, x)

# 4) variance computed via mean(x-mean)^2 (numerically stable)
def stable_var(x):
    m = jnp.mean(x, axis=-1, keepdims=True)
    return jnp.mean((x - m)**2, axis=-1, keepdims=True)
diff("04 var via mean((x-mean)^2)", stable_var, x)

# 5) (x - mean)
diff("05 x - mean",
     lambda x: x - x.mean(axis=-1, keepdims=True), x)

# 6) sqrt(var + 1e-5) using default jnp.var
diff("06 sqrt(default_var + 1e-5)",
     lambda x: jnp.sqrt(x.var(axis=-1, keepdims=True) + 1e-5), x)

# 7) sqrt(stable_var + 1e-5)
diff("07 sqrt(stable_var + 1e-5)",
     lambda x: jnp.sqrt(stable_var(x) + 1e-5), x)

# 8) rsqrt instead of 1/sqrt
diff("08 rsqrt(default_var + 1e-5)",
     lambda x: jax.lax.rsqrt(x.var(axis=-1, keepdims=True) + 1e-5), x)

# 9) full manual LN with stable var
def stable_ln(x):
    m = jnp.mean(x, axis=-1, keepdims=True)
    v = jnp.mean((x - m)**2, axis=-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5)
diff("09 full LN with stable var", stable_ln, x)

# 10) original LN using jnp.var
def naive_ln(x):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5)
diff("10 full LN with default jnp.var", naive_ln, x)

# 11) jax.nn / flax-style — multiply by rsqrt instead of divide
def rsqrt_ln(x):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + 1e-5)
diff("11 LN using rsqrt (not divide)", rsqrt_ln, x)

# 12) Smaller shape — sometimes shape-specific
x_small = rng.standard_normal((1, 4, 16)).astype(np.float32)
diff("12 naive LN, small shape (1,4,16)", naive_ln, x_small)
diff("13 naive LN, even smaller (1,1,8)", naive_ln, rng.standard_normal((1, 1, 8)).astype(np.float32))

# 14) 2D variant
x_2d = rng.standard_normal((128, 16)).astype(np.float32)
diff("14 naive LN, 2D (128,16)", naive_ln, x_2d)

# 15) Just the divide
def divide_then(x):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True) + 1e-5
    return (x - m) / v  # divide by var directly (not sqrt)
diff("15 (x-mean) / (var+eps)  [no sqrt]", divide_then, x)
