#!/usr/bin/env python3
"""
MetalHLO PJRT Plugin — vmap Test Suite

Covers the most common `jax.vmap` patterns Flax users hit:
  - vmap of a scalar function over a 1D vector (including self-referencing
    operations like x*x + c that previously miscompiled in the fusion path)
  - vmap of model.apply for batched inference of a single-example model
  - vmap of grad for per-example gradients (the standard idiom for
    differential privacy / meta-learning / Fisher info)
  - nested vmap (vmap of vmap)
  - vmap with non-default in_axes / out_axes
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

print("MetalHLO vmap Test Suite")
print("=" * 50)
print(f"JAX:    {jax.__version__}")
print(f"Flax:   {flax.__version__}")
print(f"CPU:    {cpu}")
print(f"GPU:    {device}")
print()


def test_vmap_scalar_fn():
    print("1. vmap of a scalar fn over a 1D vector (x*x + c)")
    print("-" * 40)
    # Triggers same-operand-twice in the fusion path — historically caused
    # `undeclared identifier 'input2'` Metal compile failures.
    f = jax.jit(jax.vmap(lambda x: x * x + 1.0))
    xs = np.arange(8, dtype=np.float32)
    with jax.default_device(cpu):
        out_c = f(jnp.asarray(xs))
    out_m = f(jax.device_put(jnp.asarray(xs), device))
    check("vmap_scalar_fn", out_m, out_c)


section("vmap of scalar fn", test_vmap_scalar_fn)


def test_vmap_model_apply():
    print("2. vmap of model.apply (batched inference, single-example model)")
    print("-" * 40)

    class SingleExample(nn.Module):
        @nn.compact
        def __call__(self, x):  # x: (d,)
            x = nn.Dense(32)(x)
            x = nn.relu(x)
            return nn.Dense(8)(x)

    model = SingleExample()
    np.random.seed(0)
    x_batch = np.random.randn(4, 16).astype(np.float32)
    with jax.default_device(cpu):
        params = model.init(jax.random.PRNGKey(0), jnp.zeros((16,)))
        out_c = jax.vmap(lambda xi: model.apply(params, xi))(jnp.asarray(x_batch))

    params_m = to_device(params, device)
    out_m = jax.jit(jax.vmap(lambda xi: model.apply(params_m, xi)))(
        jax.device_put(jnp.asarray(x_batch), device)
    )
    check("vmap_model_apply", out_m, out_c)


section("vmap of model.apply", test_vmap_model_apply)


def test_vmap_of_grad():
    print("3. vmap of grad — per-example gradients")
    print("-" * 40)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):  # x: (d,)
            return nn.Dense(4)(x)

    model = Net()
    np.random.seed(1)
    x_batch = np.random.randn(8, 10).astype(np.float32)
    y_batch = np.random.randn(8, 4).astype(np.float32)

    def loss_one(p, x, y):
        return jnp.mean((model.apply(p, x) - y) ** 2)

    with jax.default_device(cpu):
        params = model.init(jax.random.PRNGKey(0), jnp.zeros((10,)))
        per_ex_grads_c = jax.vmap(jax.grad(loss_one), in_axes=(None, 0, 0))(
            params, jnp.asarray(x_batch), jnp.asarray(y_batch)
        )

    params_m = to_device(params, device)
    per_ex_grads_m = jax.jit(jax.vmap(jax.grad(loss_one), in_axes=(None, 0, 0)))(
        params_m,
        jax.device_put(jnp.asarray(x_batch), device),
        jax.device_put(jnp.asarray(y_batch), device),
    )
    flat_m, _ = jax.tree.flatten(per_ex_grads_m)
    flat_c, _ = jax.tree.flatten(per_ex_grads_c)
    for i, (gm, gc) in enumerate(zip(flat_m, flat_c)):
        check(f"per_ex_grad[{i}]", gm, gc, rtol=1e-3, atol=1e-3)


section("vmap of grad", test_vmap_of_grad)


def test_nested_vmap():
    print("4. nested vmap (matmul via dot)")
    print("-" * 40)
    np.random.seed(2)
    A = np.random.randn(4, 6).astype(np.float32)
    B = np.random.randn(6, 5).astype(np.float32)
    f = lambda x, y: jnp.dot(x, y)
    vvf = jax.jit(jax.vmap(jax.vmap(f, in_axes=(None, 1)), in_axes=(0, None)))
    with jax.default_device(cpu):
        out_c = vvf(jnp.asarray(A), jnp.asarray(B))
    out_m = vvf(
        jax.device_put(jnp.asarray(A), device),
        jax.device_put(jnp.asarray(B), device),
    )
    check("nested_vmap_matmul", out_m, out_c)


section("nested vmap", test_nested_vmap)


def test_vmap_axes():
    print("5. vmap with non-default in_axes / out_axes")
    print("-" * 40)
    np.random.seed(3)
    X = np.random.randn(3, 5, 7).astype(np.float32)  # batch along axis 1
    f = lambda x: x.sum(axis=-1)
    vf = jax.jit(jax.vmap(f, in_axes=1, out_axes=0))
    with jax.default_device(cpu):
        out_c = vf(jnp.asarray(X))
    out_m = vf(jax.device_put(jnp.asarray(X), device))
    check("vmap_axes_perm", out_m, out_c)


section("vmap with non-default axes", test_vmap_axes)


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 50)
total = passed + failed
print(f"Results: {passed} passed, {failed} failed out of {total} tests")
if failed:
    print("Failed tests:", errors)
    sys.exit(1)
else:
    print("All tests passed!")
