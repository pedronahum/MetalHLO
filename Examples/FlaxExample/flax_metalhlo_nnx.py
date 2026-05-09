#!/usr/bin/env python3
"""
MetalHLO PJRT Plugin — Flax nnx Test Suite

Covers Flax's newer nnx API (the recommended successor to flax.linen).
nnx modules carry their state as mutable nnx.Variable refs at construction
time (no init/apply split), so the test pattern is:

  1. Construct the module under jax.default_device(cpu) — params live on CPU.
  2. Run the forward pass to get the CPU reference.
  3. Use nnx.split to peel out (graphdef, state), jax.device_put the state
     to MetalHLO, nnx.merge to reconstruct.
  4. Forward on metalhlo and diff vs CPU.
"""

import os
import sys

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jax._src.xla_bridge as xb
    import jaxlib.xla_client as xla_client
except ImportError:
    print("ERROR: JAX not installed.")
    sys.exit(1)

try:
    import flax
    from flax import nnx
except ImportError:
    print("ERROR: Flax (with nnx) not installed.")
    sys.exit(1)

# ── Plugin registration ──────────────────────────────────────────────

if not xla_client.pjrt_plugin_loaded("metalhlo"):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    found = next((p for p in [
        os.path.join(repo_root, ".build", "release", "libPJRTMetalHLO.dylib"),
        os.path.join(repo_root, ".build", "debug", "libPJRTMetalHLO.dylib"),
    ] if os.path.isfile(p)), None)
    if found is None:
        print("ERROR: libPJRTMetalHLO.dylib not found.")
        sys.exit(1)
    print(f"Registering MetalHLO PJRT plugin from:\n  {found}")
    xb.register_plugin("metalhlo", priority=500, library_path=found, options=None)

cpu = jax.devices("cpu")[0]
device = jax.devices("metalhlo")[0]

# ── Test harness ─────────────────────────────────────────────────────

passed = 0
failed = 0
errors = []


def check(name, got, expected, rtol=1e-4, atol=1e-4):
    global passed, failed, errors
    try:
        g = np.asarray(got)
        e = np.asarray(expected)
        if g.shape != e.shape:
            failed += 1
            errors.append(name)
            print(f"  FAIL  {name}: shape {g.shape} vs {e.shape}")
            return
        if g.dtype.kind in ("i", "u", "b") or e.dtype.kind in ("i", "u", "b"):
            ok = np.array_equal(g, e)
        else:
            ok = np.allclose(g, e, rtol=rtol, atol=atol)
        if ok:
            d = float(np.abs(g.astype(np.float64) - e.astype(np.float64)).max())
            passed += 1
            print(f"  PASS  {name} (max diff {d:.2e})")
        else:
            failed += 1
            errors.append(name)
            d = float(np.abs(g.astype(np.float64) - e.astype(np.float64)).max())
            print(f"  FAIL  {name}: max diff {d:.4f}")
            print(f"        got [:6]:      {g.flatten()[:6]}")
            print(f"        expected [:6]: {e.flatten()[:6]}")
    except Exception as e:
        failed += 1
        errors.append(f"{name} (exception)")
        print(f"  FAIL  {name} — {e}")


def section(name, fn):
    global failed, errors
    try:
        fn()
    except Exception as e:
        failed += 1
        errors.append(f"{name} (section crashed: {e})")
        print(f"  CRASH {name}: {e}")
    print()


def to_device(tree, dev):
    return jax.tree.map(lambda a: jax.device_put(a, dev), tree)


def run_module(build_fn, x, rng_seed=0):
    """Build module on CPU, run forward there, then on MetalHLO."""
    rngs_cpu = nnx.Rngs(rng_seed)
    with jax.default_device(cpu):
        model = build_fn(rngs_cpu)
        x_cpu = jnp.asarray(x)
        expected = model(x_cpu)

    # Move state to metalhlo
    graphdef, state = nnx.split(model)
    state_m = to_device(state, device)
    model_m = nnx.merge(graphdef, state_m)
    x_m = jax.device_put(jnp.asarray(x), device)
    got = model_m(x_m)
    return got, expected


# ── Tests ────────────────────────────────────────────────────────────

print("MetalHLO Flax nnx Test Suite")
print("=" * 50)
print(f"JAX:   {jax.__version__}")
print(f"Flax:  {flax.__version__}")
print(f"CPU:   {cpu}")
print(f"GPU:   {device}")
print()


def test_linear():
    print("1. nnx.Linear")
    print("-" * 40)
    np.random.seed(0)
    x = np.random.randn(8, 16).astype(np.float32)
    got, expected = run_module(
        lambda rngs: nnx.Linear(in_features=16, out_features=8, rngs=rngs), x
    )
    check("nnx_linear", got, expected, rtol=1e-4, atol=1e-4)


section("nnx.Linear", test_linear)


def test_conv():
    print("2. nnx.Conv (2D)")
    print("-" * 40)
    np.random.seed(1)
    x = np.random.randn(2, 8, 8, 4).astype(np.float32)
    got, expected = run_module(
        lambda rngs: nnx.Conv(
            in_features=4, out_features=8, kernel_size=(3, 3),
            padding="SAME", rngs=rngs,
        ),
        x,
    )
    check("nnx_conv2d", got, expected, rtol=3e-4, atol=3e-4)


section("nnx.Conv", test_conv)


def test_layernorm():
    print("3. nnx.LayerNorm")
    print("-" * 40)
    np.random.seed(2)
    x = np.random.randn(4, 32).astype(np.float32)
    got, expected = run_module(
        lambda rngs: nnx.LayerNorm(num_features=32, rngs=rngs), x
    )
    check("nnx_layernorm", got, expected, rtol=1e-4, atol=1e-4)


section("nnx.LayerNorm", test_layernorm)


def test_rmsnorm():
    print("4. nnx.RMSNorm")
    print("-" * 40)
    np.random.seed(3)
    x = np.random.randn(4, 32).astype(np.float32)
    got, expected = run_module(
        lambda rngs: nnx.RMSNorm(num_features=32, rngs=rngs), x
    )
    check("nnx_rmsnorm", got, expected, rtol=1e-4, atol=1e-4)


section("nnx.RMSNorm", test_rmsnorm)


def test_embed():
    print("5. nnx.Embed")
    print("-" * 40)
    np.random.seed(4)
    idx = np.array([[0, 1, 2, 3], [4, 5, 6, 0]], dtype=np.int32)
    got, expected = run_module(
        lambda rngs: nnx.Embed(num_embeddings=10, features=8, rngs=rngs), idx
    )
    check("nnx_embed", got, expected, rtol=1e-5, atol=1e-5)


section("nnx.Embed", test_embed)


def test_mlp():
    print("6. nnx MLP (custom Module)")
    print("-" * 40)

    class MLP(nnx.Module):
        def __init__(self, dims, *, rngs):
            self.dense1 = nnx.Linear(dims[0], dims[1], rngs=rngs)
            self.dense2 = nnx.Linear(dims[1], dims[2], rngs=rngs)
            self.ln = nnx.LayerNorm(dims[1], rngs=rngs)

        def __call__(self, x):
            x = self.dense1(x)
            x = nnx.relu(x)
            x = self.ln(x)
            x = self.dense2(x)
            return x

    np.random.seed(5)
    x = np.random.randn(4, 16).astype(np.float32)
    got, expected = run_module(lambda rngs: MLP((16, 32, 8), rngs=rngs), x)
    check("nnx_mlp", got, expected, rtol=2e-4, atol=2e-4)


section("nnx MLP", test_mlp)


def test_attention():
    print("7. nnx.MultiHeadAttention")
    print("-" * 40)
    np.random.seed(6)
    x = np.random.randn(2, 6, 16).astype(np.float32)
    got, expected = run_module(
        lambda rngs: nnx.MultiHeadAttention(
            num_heads=4, in_features=16, qkv_features=16,
            decode=False, rngs=rngs,
        ),
        x,
    )
    check("nnx_mha", got, expected, rtol=5e-4, atol=5e-4)


section("nnx.MultiHeadAttention", test_attention)


def test_dropout_deterministic():
    print("8. nnx.Dropout (deterministic)")
    print("-" * 40)
    np.random.seed(7)
    x = np.random.randn(4, 16).astype(np.float32)

    class Net(nnx.Module):
        def __init__(self, *, rngs):
            self.dense = nnx.Linear(16, 8, rngs=rngs)
            self.drop = nnx.Dropout(rate=0.5, deterministic=True, rngs=rngs)

        def __call__(self, x):
            return self.drop(nnx.relu(self.dense(x)))

    got, expected = run_module(lambda rngs: Net(rngs=rngs), x)
    check("nnx_dropout_det", got, expected, rtol=1e-4, atol=1e-4)


section("nnx.Dropout (det)", test_dropout_deterministic)


def test_batchnorm_inference():
    print("9. nnx.BatchNorm (inference / use_running_average)")
    print("-" * 40)
    np.random.seed(8)
    x = np.random.randn(4, 16).astype(np.float32)
    got, expected = run_module(
        lambda rngs: nnx.BatchNorm(
            num_features=16, use_running_average=True, rngs=rngs
        ),
        x,
    )
    check("nnx_batchnorm_inference", got, expected, rtol=1e-4, atol=1e-4)


section("nnx.BatchNorm (inference)", test_batchnorm_inference)


def test_jit_forward():
    print("10. nnx.jit-compiled forward")
    print("-" * 40)

    class TwoLayer(nnx.Module):
        def __init__(self, *, rngs):
            self.l1 = nnx.Linear(16, 32, rngs=rngs)
            self.l2 = nnx.Linear(32, 8, rngs=rngs)

        def __call__(self, x):
            return self.l2(nnx.relu(self.l1(x)))

    np.random.seed(9)
    x = np.random.randn(4, 16).astype(np.float32)

    rngs = nnx.Rngs(9)
    with jax.default_device(cpu):
        model = TwoLayer(rngs=rngs)
        # Use nnx.jit which is the canonical jit wrapper for nnx
        jit_fwd = nnx.jit(lambda m, x: m(x))
        expected = jit_fwd(model, jnp.asarray(x))

    graphdef, state = nnx.split(model)
    model_m = nnx.merge(graphdef, to_device(state, device))
    got = nnx.jit(lambda m, x: m(x))(model_m, jax.device_put(jnp.asarray(x), device))
    check("nnx_jit_forward", got, expected, rtol=2e-4, atol=2e-4)


section("nnx.jit forward", test_jit_forward)


def test_grad():
    print("11. nnx grad — single backward")
    print("-" * 40)

    class MLP(nnx.Module):
        def __init__(self, *, rngs):
            self.l1 = nnx.Linear(16, 32, rngs=rngs)
            self.l2 = nnx.Linear(32, 8, rngs=rngs)

        def __call__(self, x):
            return self.l2(nnx.relu(self.l1(x)))

    np.random.seed(10)
    x = np.random.randn(4, 16).astype(np.float32)
    y = np.random.randn(4, 8).astype(np.float32)

    def loss_fn(model, x, y):
        return jnp.mean((model(x) - y) ** 2)

    rngs = nnx.Rngs(10)
    with jax.default_device(cpu):
        model_c = MLP(rngs=rngs)
        loss_c = loss_fn(model_c, jnp.asarray(x), jnp.asarray(y))
        grad_c = nnx.grad(loss_fn)(model_c, jnp.asarray(x), jnp.asarray(y))

    graphdef, state = nnx.split(model_c)
    model_m = nnx.merge(graphdef, to_device(state, device))
    loss_m = loss_fn(model_m, jax.device_put(jnp.asarray(x), device),
                     jax.device_put(jnp.asarray(y), device))
    grad_m = nnx.grad(loss_fn)(
        model_m,
        jax.device_put(jnp.asarray(x), device),
        jax.device_put(jnp.asarray(y), device),
    )
    check("nnx_grad_loss", loss_m, loss_c, rtol=1e-4, atol=1e-4)
    # nnx.grad returns an updated module; extract state to compare
    _, gs_c = nnx.split(grad_c)
    _, gs_m = nnx.split(grad_m)
    leaves_c, _ = jax.tree.flatten(gs_c)
    leaves_m, _ = jax.tree.flatten(gs_m)
    worst = 0.0
    for c, m in zip(leaves_c, leaves_m):
        d = float(np.abs(np.asarray(m) - np.asarray(c)).max())
        worst = max(worst, d)
    if worst <= 5e-4:
        check("nnx_grad_state_max", np.float32(worst), np.float32(0.0),
              rtol=1.0, atol=5e-4)
    else:
        check("nnx_grad_state_max", np.float32(worst), np.float32(0.0),
              rtol=1e-4, atol=1e-4)


section("nnx grad", test_grad)


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 50)
total = passed + failed
print(f"Results: {passed} passed, {failed} failed out of {total} tests")
if failed:
    print("Failed tests:", errors)
    sys.exit(1)
else:
    print("All tests passed!")
