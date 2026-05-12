"""Minimal reproducer for the 4D transpose bug.

Hypothesis: two consecutive `.transpose` calls on a 4D tensor get fused
into a single permutation by an MetalHLO pass, but the fused permutation
is wrong.
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
    c, g = run(cpu, fn, *arrs), run(gpu, fn, *arrs)
    d = np.max(np.abs(c - g))
    print(f"{name:55s}  out_shape={c.shape}  max|diff|={d:.4e}")

rng = np.random.default_rng(0)
B, T, H, HD = 2, 64, 4, 32
x = rng.standard_normal((B, T, H, HD)).astype(np.float32)  # input layout

# Original buggy pattern
def two_xpose(x):
    return x.transpose(0, 2, 1, 3).transpose(0, 1, 3, 2)
# Equivalent single transpose
def one_xpose(x):
    return x.transpose(0, 2, 3, 1)

diff("two_xpose (broken from QKV path)", two_xpose, x)
diff("one_xpose (equivalent single)", one_xpose, x)

# Try other 4D double-transposes
def two_a(x): return x.transpose(0, 1, 2, 3).transpose(0, 1, 2, 3)  # identity x identity
def two_b(x): return x.transpose(0, 2, 1, 3).transpose(0, 2, 1, 3)  # involution
def two_c(x): return x.transpose(0, 1, 3, 2).transpose(0, 1, 3, 2)  # involution on last 2

diff("two identity", two_a, x)
diff("two_b (1<->2 twice)", two_b, x)
diff("two_c (2<->3 twice)", two_c, x)

# Without JIT
def via_eager(x):
    with jax.default_device(gpu):
        xg = jnp.asarray(x)
        out = xg.transpose(0, 2, 1, 3).transpose(0, 1, 3, 2)
    return np.asarray(out)
cpu_truth = run(cpu, two_xpose, x)
eager = via_eager(x)
print(f"eager (no jit) MetalHLO vs CPU jit: max|diff|={np.max(np.abs(cpu_truth-eager)):.4e}")

# Same shape, single transpose, varying permutation
for perm in [(0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (3, 2, 1, 0)]:
    def fn(x, _perm=perm): return x.transpose(_perm)
    diff(f"single transpose perm={perm}", fn, x)

# Double transpose: vary the two permutations
import itertools
all_perms = list(itertools.permutations(range(4)))
mismatches = 0
for p1, p2 in [
    ((0, 2, 1, 3), (0, 1, 3, 2)),  # the buggy combo
    ((0, 2, 1, 3), (0, 2, 1, 3)),  # involution
    ((0, 1, 3, 2), (0, 1, 3, 2)),
    ((0, 3, 1, 2), (0, 1, 3, 2)),
    ((0, 3, 2, 1), (0, 2, 1, 3)),
    ((1, 0, 2, 3), (0, 1, 3, 2)),
]:
    def fn(x, _p1=p1, _p2=p2): return x.transpose(_p1).transpose(_p2)
    diff(f"double {p1} then {p2}", fn, x)
