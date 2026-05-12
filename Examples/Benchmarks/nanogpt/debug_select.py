"""Hypothesis: select(scalar_bool, arr_a, arr_b) where the bool is a 0-dim
scalar broadcast against multi-dim arrays is miscompiled on MetalHLO."""
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
    print(f"{name:65s}  shape={c.shape}  diff={d}  nan_gpu={nan_g}  finite={fin}/{g.size}")

rng = np.random.default_rng(0)
a = rng.standard_normal((2, 64, 1)).astype(np.float32)  # var-shaped
nan_arr = np.full((2, 64, 1), np.nan, dtype=np.float32)

# Scalar bool, both branches are 3D arrays of same shape
def sel_scalar_true(a, nan_arr):
    return jnp.where(jnp.bool_(True), a, nan_arr)
def sel_scalar_false(a, nan_arr):
    return jnp.where(jnp.bool_(False), a, nan_arr)
diff("01 where(True, a, NaN_arr)", sel_scalar_true, a, nan_arr)
diff("02 where(False, a, NaN_arr)", sel_scalar_false, a, nan_arr)

# Same shape bool
def sel_full_bool(a, nan_arr):
    cond = jnp.ones_like(a, dtype=jnp.bool_)
    return jnp.where(cond, a, nan_arr)
diff("03 where(full_True, a, NaN_arr)", sel_full_bool, a, nan_arr)

# What the var jaxpr literally does: gt(128.0, 0.0), broadcast, select.
def sel_var_pattern(a, nan_arr):
    j = jnp.float32(128.0)
    cond = j > 0.0  # scalar bool
    return jax.lax.select_n(cond, nan_arr, a)
diff("04 lax.select_n(scalar bool, nan_arr, a)", sel_var_pattern, a, nan_arr)

# Scalar bool + broadcast then select
def sel_bcast(a, nan_arr):
    j = jnp.float32(128.0)
    cond = j > 0.0
    cond_bc = jnp.broadcast_to(cond, a.shape)  # explicit broadcast
    return jax.lax.select_n(cond_bc, nan_arr, a)
diff("05 explicit-broadcast cond + select_n", sel_bcast, a, nan_arr)

# Plain jax.lax.select (which is the binary variant)
def sel_lax(a, nan_arr):
    cond = jnp.float32(128.0) > 0.0
    return jax.lax.select(cond, a, nan_arr)
diff("06 lax.select(scalar bool, a, nan)", sel_lax, a, nan_arr)
