#!/usr/bin/env python3
"""
MetalHLO PJRT Plugin — Flax bfloat16 Test Suite

Covers the most common bfloat16 patterns real users hit:
  - Pure bf16 forward (Dense, Conv, ReLU, Sum)
  - bf16 loss + grad (Dense)
  - Mixed precision (fp32 params, bf16 compute)
  - Mixed-precision training step under Adam
  - GroupNorm / LayerNorm on bf16

Tolerances are bf16-appropriate (~5e-2) since bf16 has ~3 decimal digits
of precision; CPU and GPU may differ in summation order.
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

try:
    import optax
except ImportError:
    print("ERROR: optax not installed.")
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


def check(name, got, expected, rtol=5e-2, atol=5e-2):
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
            print(f"        got [:6]:      {g.flatten()[:6]}")
            print(f"        expected [:6]: {e.flatten()[:6]}")
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

print("MetalHLO Flax bfloat16 Test Suite")
print("=" * 50)
print(f"JAX:    {jax.__version__}")
print(f"Flax:   {flax.__version__}")
print(f"CPU:    {cpu}")
print(f"GPU:    {device}")
print()


def test_bf16_add():
    print("1. bf16 add (smoke test)")
    print("-" * 40)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.bfloat16)
    y = np.array([0.5, 1.5, 2.5, 3.5], dtype=jnp.bfloat16)

    @jax.jit
    def add(a, b):
        return a + b

    out_m = add(
        jax.device_put(jnp.asarray(x), device),
        jax.device_put(jnp.asarray(y), device),
    )
    out_c = x.astype(np.float32) + y.astype(np.float32)
    check("bf16_add", out_m, out_c)


section("bf16 add", test_bf16_add)


def test_bf16_dense_forward():
    print("2. bf16 Dense forward")
    print("-" * 40)
    np.random.seed(0)
    x = np.random.randn(4, 16).astype(jnp.bfloat16)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(32, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
            x = nn.relu(x)
            x = nn.Dense(8, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
            return x

    model = Net()
    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
        out_c = model.apply(params_c, jnp.asarray(x))
    params_m = to_device(params_c, device)
    out_m = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("bf16_dense", out_m, out_c)


section("bf16 Dense forward", test_bf16_dense_forward)


def test_bf16_conv_forward():
    print("3. bf16 Conv forward")
    print("-" * 40)
    np.random.seed(1)
    x = np.random.randn(2, 8, 8, 3).astype(jnp.bfloat16)

    class CNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Conv(8, kernel_size=(3, 3),
                        dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)
            return nn.relu(x)

    model = CNet()
    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(1), jnp.asarray(x))
        out_c = model.apply(params_c, jnp.asarray(x))
    params_m = to_device(params_c, device)
    out_m = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("bf16_conv", out_m, out_c)


section("bf16 Conv forward", test_bf16_conv_forward)


def test_bf16_grad():
    print("4. bf16 grad (Dense)")
    print("-" * 40)
    np.random.seed(2)
    x = np.random.randn(4, 16).astype(jnp.bfloat16)
    y = np.random.randn(4, 8).astype(jnp.bfloat16)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(8, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)

    model = Net()

    def loss_fn(p, x, y):
        return jnp.mean((model.apply(p, x) - y) ** 2)

    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
        loss_c, grads_c = jax.value_and_grad(loss_fn)(params_c, jnp.asarray(x), jnp.asarray(y))

    params_m = to_device(params_c, device)
    loss_m, grads_m = jax.jit(jax.value_and_grad(loss_fn))(
        params_m,
        jax.device_put(jnp.asarray(x), device),
        jax.device_put(jnp.asarray(y), device),
    )
    check("bf16_loss", loss_m, loss_c)
    flat_m, _ = jax.tree.flatten(grads_m)
    flat_c, _ = jax.tree.flatten(grads_c)
    for i, (gm, gc) in enumerate(zip(flat_m, flat_c)):
        check(f"bf16_grad[{i}]", gm, gc)


section("bf16 grad", test_bf16_grad)


def test_mixed_precision_forward():
    print("5. mixed precision (fp32 params, bf16 compute)")
    print("-" * 40)
    np.random.seed(3)
    x = np.random.randn(4, 16).astype(np.float32)

    class MixedNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(32, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
            x = nn.relu(x)
            x = nn.Dense(8, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
            return x

    model = MixedNet()
    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
        out_c = model.apply(params_c, jnp.asarray(x))
    params_m = to_device(params_c, device)
    out_m = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("mixed_precision_forward", out_m, out_c)


section("mixed precision forward", test_mixed_precision_forward)


def test_bf16_train_step():
    print("6. bf16 training step (Adam)")
    print("-" * 40)
    np.random.seed(4)
    x = np.random.randn(4, 16).astype(jnp.bfloat16)
    y = np.random.randn(4, 8).astype(jnp.bfloat16)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(8, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)

    model = Net()
    optimizer = optax.adam(learning_rate=1e-2)

    def loss_fn(p, x, y):
        return jnp.mean((model.apply(p, x) - y) ** 2)

    @jax.jit
    def step(p, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss

    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
        opt_c = optimizer.init(params_c)
        params_c, opt_c, loss_c = step(params_c, opt_c, jnp.asarray(x), jnp.asarray(y))

    params_m = to_device(params_c if False else model.init(jax.random.PRNGKey(0), jnp.asarray(x)), device)
    opt_m = optimizer.init(params_m)
    xm = jax.device_put(jnp.asarray(x), device)
    ym = jax.device_put(jnp.asarray(y), device)
    # Re-init to keep CPU/GPU starting points identical
    with jax.default_device(cpu):
        init_params = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
    params_m = to_device(init_params, device)
    opt_m = optimizer.init(params_m)
    params_m, opt_m, loss_m = step(params_m, opt_m, xm, ym)

    check("bf16_step_loss", loss_m, loss_c)
    flat_m, _ = jax.tree.flatten(params_m)
    flat_c, _ = jax.tree.flatten(params_c)
    for i, (pm, pc) in enumerate(zip(flat_m, flat_c)):
        check(f"bf16_step_param[{i}]", pm, pc)


section("bf16 training step", test_bf16_train_step)


def test_bf16_layernorm():
    print("7. bf16 LayerNorm forward")
    print("-" * 40)
    np.random.seed(5)
    x = np.random.randn(4, 32).astype(jnp.bfloat16)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.LayerNorm(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)(x)

    model = Net()
    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
        out_c = model.apply(params_c, jnp.asarray(x))
    params_m = to_device(params_c, device)
    out_m = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("bf16_layernorm", out_m, out_c)


section("bf16 LayerNorm", test_bf16_layernorm)


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 50)
total = passed + failed
print(f"Results: {passed} passed, {failed} failed out of {total} tests")
if failed:
    print("Failed tests:", errors)
    sys.exit(1)
else:
    print("All tests passed!")
