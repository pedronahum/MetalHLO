#!/usr/bin/env python3
"""
MetalHLO PJRT Plugin — JAX Example

Demonstrates running JAX computations on Apple Silicon GPUs via
the MetalHLO PJRT plugin.

Prerequisites:
    1. Build the plugin:
       cd <repo-root>
       swift build -c release --product PJRTMetalHLO

    2. Install JAX and the MetalHLO plugin package:
       pip install jax
       pip install -e python/

    3. Run this example:
       python Examples/JAXExample/jax_metalhlo_example.py
"""

import sys
import os

# ── Verify prerequisites ────────────────────────────────────────────

try:
    import jax
    import jax.numpy as jnp
    import jax._src.xla_bridge as xb
    import jaxlib.xla_client as xla_client
except ImportError:
    print("ERROR: JAX is not installed.")
    print("  Install it with: pip install jax")
    sys.exit(1)

# Register the MetalHLO plugin before any jax.devices() call
if not xla_client.pjrt_plugin_loaded("metalhlo"):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dylib_candidates = [
        os.path.join(repo_root, ".build", "release", "libPJRTMetalHLO.dylib"),
        os.path.join(repo_root, ".build", "debug", "libPJRTMetalHLO.dylib"),
    ]
    found = None
    for p in dylib_candidates:
        if os.path.isfile(p):
            found = p
            break

    if found is None:
        print("ERROR: libPJRTMetalHLO.dylib not found.")
        print("  Build it with: swift build -c release --product PJRTMetalHLO")
        sys.exit(1)

    print(f"Registering MetalHLO PJRT plugin from:\n  {found}")
    try:
        xb.register_plugin("metalhlo", priority=500, library_path=found, options=None)
    except Exception as e:
        print(f"ERROR: Failed to register MetalHLO plugin: {e}")
        sys.exit(1)

try:
    metalhlo_devices = jax.devices("metalhlo")
except Exception as e:
    print(f"ERROR: MetalHLO backend not available: {e}")
    print("  Available backends:", [d.platform for d in jax.devices()])
    sys.exit(1)

if not metalhlo_devices:
    print("ERROR: No MetalHLO devices found.")
    sys.exit(1)

device = metalhlo_devices[0]

# ── Examples ─────────────────────────────────────────────────────────

print("MetalHLO JAX Example")
print("====================")
print(f"JAX version:  {jax.__version__}")
print(f"Device:       {device}")
print()


# Example 1: Basic Arithmetic
print("Example 1: Vector Arithmetic")
print("----------------------------")

x = jax.device_put(jnp.array([1.0, 2.0, 3.0, 4.0]), device)
y = jax.device_put(jnp.array([10.0, 20.0, 30.0, 40.0]), device)

result = x + y
print(f"  x     = {x}")
print(f"  y     = {y}")
print(f"  x + y = {result}")
print()


# Example 2: Matrix Multiply
print("Example 2: Matrix Multiply")
print("--------------------------")

a = jax.device_put(jnp.array([[1.0, 2.0], [3.0, 4.0]]), device)
b = jax.device_put(jnp.array([[5.0, 6.0], [7.0, 8.0]]), device)

c = jnp.dot(a, b)
print(f"  A = {a.tolist()}")
print(f"  B = {b.tolist()}")
print(f"  A @ B = {c.tolist()}")
print()


# Example 3: JIT-compiled function
print("Example 3: JIT Compilation")
print("--------------------------")

@jax.jit
def mlp_layer(x, w, b):
    """Single MLP layer: relu(x @ w + b)"""
    return jnp.maximum(x @ w + b, 0.0)

x_in = jax.device_put(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), device)
w_in = jax.device_put(jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), device)
b_in = jax.device_put(jnp.array([-1.0, 0.5]), device)

out = mlp_layer(x_in, w_in, b_in)
print(f"  x (2x3): {x_in.tolist()}")
print(f"  w (3x2): {w_in.tolist()}")
print(f"  b (2):   {b_in.tolist()}")
print(f"  relu(x @ w + b) = {out.tolist()}")
print()


# Example 4: Element-wise Operations
print("Example 4: Element-wise Math")
print("----------------------------")

vals = jax.device_put(jnp.array([0.0, 1.0, 2.0, 3.0]), device)

print(f"  input   = {vals.tolist()}")
print(f"  exp     = {jnp.exp(vals).tolist()}")
print(f"  sqrt    = {jnp.sqrt(vals).tolist()}")
print(f"  tanh    = {jnp.tanh(vals).tolist()}")
print()


# Example 5: Reduction Operations
print("Example 5: Reductions")
print("---------------------")

mat = jax.device_put(jnp.array([[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]]), device)

print(f"  matrix     = {mat.tolist()}")
print(f"  sum        = {jnp.sum(mat)}")
print(f"  sum(axis=0)= {jnp.sum(mat, axis=0).tolist()}")
print(f"  sum(axis=1)= {jnp.sum(mat, axis=1).tolist()}")
print(f"  max        = {jnp.max(mat)}")
print(f"  mean       = {jnp.mean(mat)}")
print()


# Example 6: Softmax (tests fused pattern)
print("Example 6: Softmax")
print("------------------")

@jax.jit
def softmax(x):
    """Numerically stable softmax."""
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

logits = jax.device_put(jnp.array([[1.0, 2.0, 3.0, 4.0],
                                    [1.0, 1.0, 1.0, 1.0]]), device)

probs = softmax(logits)
print(f"  logits   = {logits.tolist()}")
print(f"  softmax  = {[[round(v, 4) for v in row] for row in probs.tolist()]}")
print(f"  row sums = {jnp.sum(probs, axis=-1).tolist()}  (should be [1.0, 1.0])")
print()


print("All examples completed successfully!")
