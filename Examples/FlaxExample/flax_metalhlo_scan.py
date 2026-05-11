#!/usr/bin/env python3
"""
MetalHLO PJRT Plugin — jax.lax.scan / flax.linen.scan Test Suite

Covers the common scan / unrolled-loop patterns:
  - jax.lax.scan cumulative sum (smallest possible scan)
  - jax.lax.scan with a Dense-style step
  - flax.linen.scan with a Dense cell and shared params (RNN-style forward)
  - flax.linen.scan + grad — an actual RNN training step. Used to crash
    with a SIGTRAP because the unroller produced colliding SSA names
    between the forward and backward while loops.
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
    import flax.linen as nn
except ImportError:
    print("ERROR: Flax not installed.")
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
        g = np.asarray(got).astype(np.float32)
        e = np.asarray(expected).astype(np.float32)
        if g.shape != e.shape:
            failed += 1
            errors.append(name)
            print(f"  FAIL  {name}: shape {g.shape} vs {e.shape}")
            return
        ok = np.allclose(g, e, rtol=rtol, atol=atol)
        if ok:
            passed += 1
            print(f"  PASS  {name} (max diff "
                  f"{float(np.abs(g - e).max()):.2e})")
        else:
            failed += 1
            errors.append(name)
            d = float(np.abs(g - e).max())
            print(f"  FAIL  {name}: max diff {d:.4f}")
    except Exception as ex:
        failed += 1
        errors.append(f"{name} (exception)")
        print(f"  FAIL  {name} — {ex}")


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


# ── Tests ────────────────────────────────────────────────────────────

print("MetalHLO jax.lax.scan / flax.linen.scan Test Suite")
print("=" * 50)
print(f"JAX:    {jax.__version__}")
print(f"Flax:   {flax.__version__}")
print(f"CPU:    {cpu}")
print(f"GPU:    {device}")
print()


def test_scan_cumsum():
    print("1. jax.lax.scan — cumulative sum")
    print("-" * 40)

    @jax.jit
    def cumsum(xs):
        def step(carry, x):
            new = carry + x
            return new, new
        _, out = jax.lax.scan(step, jnp.zeros((), dtype=xs.dtype), xs)
        return out

    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    with jax.default_device(cpu):
        out_c = cumsum(jnp.asarray(xs))
    out_m = cumsum(jax.device_put(jnp.asarray(xs), device))
    check("scan_cumsum", out_m, out_c)


section("scan cumsum", test_scan_cumsum)


def test_scan_dense_step():
    print("2. jax.lax.scan — Dense-style step (matmul + bias + carry)")
    print("-" * 40)

    @jax.jit
    def run(W, b, xs):
        def step(carry, x):
            y = jnp.dot(x, W) + b + carry
            return y, y
        _, out = jax.lax.scan(step, jnp.zeros((W.shape[1],), dtype=W.dtype), xs)
        return out

    np.random.seed(0)
    W = np.random.randn(8, 4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    xs = np.random.randn(5, 8).astype(np.float32)
    with jax.default_device(cpu):
        out_c = run(jnp.asarray(W), jnp.asarray(b), jnp.asarray(xs))
    out_m = run(
        jax.device_put(jnp.asarray(W), device),
        jax.device_put(jnp.asarray(b), device),
        jax.device_put(jnp.asarray(xs), device),
    )
    check("scan_dense_step", out_m, out_c, rtol=1e-5, atol=1e-5)


section("scan Dense step", test_scan_dense_step)


def test_nn_scan_forward():
    print("3. flax.linen.scan — Dense cell, shared params (RNN forward)")
    print("-" * 40)

    class StepCell(nn.Module):
        features: int
        @nn.compact
        def __call__(self, carry, x):
            y = nn.Dense(self.features)(jnp.concatenate([carry, x], axis=-1))
            return y, y

    class Recur(nn.Module):
        features: int
        @nn.compact
        def __call__(self, init_carry, xs):
            ScanCell = nn.scan(
                StepCell, variable_broadcast="params",
                split_rngs={"params": False},
            )
            return ScanCell(features=self.features)(init_carry, xs)

    model = Recur(features=4)
    np.random.seed(1)
    xs = np.random.randn(6, 8).astype(np.float32)
    init = np.zeros((4,), dtype=np.float32)
    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(0), jnp.asarray(init), jnp.asarray(xs))
        out_carry_c, out_ys_c = model.apply(
            params_c, jnp.asarray(init), jnp.asarray(xs)
        )

    params_m = to_device(params_c, device)
    out_carry_m, out_ys_m = jax.jit(model.apply)(
        params_m,
        jax.device_put(jnp.asarray(init), device),
        jax.device_put(jnp.asarray(xs), device),
    )
    check("nn_scan_carry", out_carry_m, out_carry_c)
    check("nn_scan_ys", out_ys_m, out_ys_c)


section("nn.scan forward", test_nn_scan_forward)


def test_nn_scan_grad():
    print("4. flax.linen.scan + grad — RNN training step")
    print("-" * 40)

    class StepCell(nn.Module):
        features: int
        @nn.compact
        def __call__(self, carry, x):
            y = nn.Dense(self.features)(jnp.concatenate([carry, x], axis=-1))
            y = jnp.tanh(y)
            return y, y

    class RNN(nn.Module):
        features: int
        @nn.compact
        def __call__(self, init_carry, xs):
            ScanCell = nn.scan(
                StepCell, variable_broadcast="params",
                split_rngs={"params": False},
            )
            return ScanCell(features=self.features)(init_carry, xs)

    model = RNN(features=4)
    np.random.seed(2)
    xs = np.random.randn(5, 8).astype(np.float32)
    init = np.zeros((4,), dtype=np.float32)
    target = np.random.randn(5, 4).astype(np.float32)

    def loss(p, init_c, xs, target):
        _, ys = model.apply(p, init_c, xs)
        return jnp.mean((ys - target) ** 2)

    with jax.default_device(cpu):
        params_c = model.init(
            jax.random.PRNGKey(0), jnp.asarray(init), jnp.asarray(xs)
        )
        loss_c, grads_c = jax.value_and_grad(loss)(
            params_c, jnp.asarray(init), jnp.asarray(xs), jnp.asarray(target)
        )

    params_m = to_device(params_c, device)
    loss_m, grads_m = jax.jit(jax.value_and_grad(loss))(
        params_m,
        jax.device_put(jnp.asarray(init), device),
        jax.device_put(jnp.asarray(xs), device),
        jax.device_put(jnp.asarray(target), device),
    )
    check("rnn_loss", loss_m, loss_c)
    flat_m, _ = jax.tree.flatten(grads_m)
    flat_c, _ = jax.tree.flatten(grads_c)
    for i, (gm, gc) in enumerate(zip(flat_m, flat_c)):
        check(f"rnn_grad[{i}]", gm, gc, rtol=1e-3, atol=1e-3)


section("nn.scan + grad", test_nn_scan_grad)


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 50)
total = passed + failed
print(f"Results: {passed} passed, {failed} failed out of {total} tests")
if failed:
    print("Failed tests:", errors)
    sys.exit(1)
else:
    print("All tests passed!")
