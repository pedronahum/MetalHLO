"""Focused 4D-matmul reproducer.

Step 03 of the bisect found Q @ K^T (a (B, H, T, HD) @ (B, H, HD, T)) gives
a 0.33 max-abs-diff between JAX CPU and MetalHLO. This script narrows that
down: does the diff appear without the transpose? without the head split?
on smaller shapes?
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

import jax
import jax.numpy as jnp
import numpy as np

cpu = jax.devices("cpu")[0]; gpu = jax.devices("metalhlo")[0]

def diff(name, fn, *arrs):
    with jax.default_device(cpu):
        oc = np.asarray(jax.jit(fn)(*[jnp.asarray(a) for a in arrs]))
    with jax.default_device(gpu):
        og = np.asarray(jax.jit(fn)(*[jnp.asarray(a) for a in arrs]))
    d = np.max(np.abs(oc - og))
    print(f"{name:60s}  shape={oc.shape}  max|diff|={d:.4e}  cpu_range=[{oc.min():.3f},{oc.max():.3f}]")

rng = np.random.default_rng(0)

# Shapes from bisect: B=2, H=4, T=64, HD=32
B, H, T, HD = 2, 4, 64, 32
q = rng.standard_normal((B, H, T, HD)).astype(np.float32)
k = rng.standard_normal((B, H, T, HD)).astype(np.float32)

# Baseline: 4D batched matmul, K transposed to (B, H, HD, T)
def f1(q, k): return q @ k.transpose(0, 1, 3, 2)
diff("01 q @ k.T  4D batched", f1, q, k)

# 2D matmul of one (T, HD) @ (HD, T) — sanity that 2D works
q2 = q[0, 0]; k2 = k[0, 0]
def f2(q, k): return q @ k.T
diff("02 q[0,0] @ k[0,0].T  (T,HD)@(HD,T)", f2, q2, k2)

# 3D version (collapse batch+head)
qm = q.reshape(B*H, T, HD); km = k.reshape(B*H, T, HD)
def f3(q, k): return q @ k.transpose(0, 2, 1)
diff("03 (BH,T,HD) @ (BH,HD,T)  3D batched", f3, qm, km)

# Try einsum instead of @  with 4D
def f4(q, k): return jnp.einsum("bhtd,bhsd->bhts", q, k)
diff("04 einsum bhtd,bhsd->bhts", f4, q, k)

# Try with .swapaxes instead of .transpose
def f5(q, k): return q @ jnp.swapaxes(k, -1, -2)
diff("05 q @ swapaxes(k,-1,-2)", f5, q, k)

# Reshape to 3D, multiply, reshape back — workaround
def f6(q, k):
    q3 = q.reshape(B*H, T, HD)
    k3 = k.reshape(B*H, T, HD)
    out = q3 @ k3.transpose(0, 2, 1)  # (BH, T, T)
    return out.reshape(B, H, T, T)
diff("06 reshape→3D→@→reshape→4D", f6, q, k)

# Smaller shapes
for shape_name, (b, h, t, hd) in [
    ("tiny 1,1,8,8", (1, 1, 8, 8)),
    ("small 1,2,16,16", (1, 2, 16, 16)),
    ("typical 1,4,64,32", (1, 4, 64, 32)),
]:
    qx = rng.standard_normal((b, h, t, hd)).astype(np.float32)
    kx = rng.standard_normal((b, h, t, hd)).astype(np.float32)
    diff(f"4D matmul {shape_name}", f1, qx, kx)
