#!/usr/bin/env python3
"""
MetalHLO PJRT Plugin — Flax float16 Test Suite

Covers the most common fp16 patterns:
  - Pure fp16 forward (Dense, Conv, ReLU, Sum)
  - fp16 loss + grad (Dense, including bias reduction)
  - Mixed precision (fp32 params, fp16 compute)
  - fp16 LayerNorm

fp16 has slightly more precision than bf16 (10-bit mantissa vs 7-bit) so
tolerances here are tighter (~5e-3 vs 5e-2).
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


def check(name, got, expected, rtol=5e-3, atol=5e-3):
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

print("MetalHLO Flax float16 Test Suite")
print("=" * 50)
print(f"JAX:    {jax.__version__}")
print(f"Flax:   {flax.__version__}")
print(f"CPU:    {cpu}")
print(f"GPU:    {device}")
print()


def test_fp16_add():
    print("1. fp16 add (smoke test)")
    print("-" * 40)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    y = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float16)
    @jax.jit
    def add(a, b):
        return a + b
    out_c = x.astype(np.float32) + y.astype(np.float32)
    out_m = add(
        jax.device_put(jnp.asarray(x), device),
        jax.device_put(jnp.asarray(y), device),
    )
    check("fp16_add", out_m, out_c)


section("fp16 add", test_fp16_add)


def test_fp16_dense_forward():
    print("2. fp16 Dense forward")
    print("-" * 40)
    np.random.seed(0)
    x = np.random.randn(4, 16).astype(np.float16)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(32, dtype=jnp.float16, param_dtype=jnp.float16)(x)
            x = nn.relu(x)
            return nn.Dense(8, dtype=jnp.float16, param_dtype=jnp.float16)(x)

    model = Net()
    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
        out_c = model.apply(params_c, jnp.asarray(x))
    params_m = to_device(params_c, device)
    out_m = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("fp16_dense", out_m, out_c)


section("fp16 Dense forward", test_fp16_dense_forward)


def test_fp16_conv_forward():
    print("3. fp16 Conv forward")
    print("-" * 40)
    np.random.seed(1)
    x = np.random.randn(2, 8, 8, 3).astype(np.float16)

    class CNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Conv(8, kernel_size=(3, 3),
                        dtype=jnp.float16, param_dtype=jnp.float16)(x)
            return nn.relu(x)

    model = CNet()
    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(1), jnp.asarray(x))
        out_c = model.apply(params_c, jnp.asarray(x))
    params_m = to_device(params_c, device)
    out_m = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("fp16_conv", out_m, out_c)


section("fp16 Conv forward", test_fp16_conv_forward)


def test_fp16_grad():
    print("4. fp16 grad (Dense)")
    print("-" * 40)
    np.random.seed(2)
    x = np.random.randn(4, 16).astype(np.float16)
    y = np.random.randn(4, 8).astype(np.float16)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(8, dtype=jnp.float16, param_dtype=jnp.float16)(x)

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
    check("fp16_loss", loss_m, loss_c)
    flat_m, _ = jax.tree.flatten(grads_m)
    flat_c, _ = jax.tree.flatten(grads_c)
    for i, (gm, gc) in enumerate(zip(flat_m, flat_c)):
        check(f"fp16_grad[{i}]", gm, gc)


section("fp16 grad", test_fp16_grad)


def test_fp16_mixed_precision():
    print("5. mixed precision (fp32 params, fp16 compute)")
    print("-" * 40)
    np.random.seed(3)
    x = np.random.randn(4, 16).astype(np.float32)

    class MixedNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(32, dtype=jnp.float16, param_dtype=jnp.float32)(x)
            x = nn.relu(x)
            return nn.Dense(8, dtype=jnp.float16, param_dtype=jnp.float32)(x)

    model = MixedNet()
    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
        out_c = model.apply(params_c, jnp.asarray(x))
    params_m = to_device(params_c, device)
    out_m = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("fp16_mixed", out_m, out_c)


section("fp16 mixed precision", test_fp16_mixed_precision)


def test_fp16_layernorm():
    print("6. fp16 LayerNorm forward")
    print("-" * 40)
    np.random.seed(5)
    x = np.random.randn(4, 32).astype(np.float16)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.LayerNorm(dtype=jnp.float16, param_dtype=jnp.float16)(x)

    model = Net()
    with jax.default_device(cpu):
        params_c = model.init(jax.random.PRNGKey(0), jnp.asarray(x))
        out_c = model.apply(params_c, jnp.asarray(x))
    params_m = to_device(params_c, device)
    out_m = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("fp16_layernorm", out_m, out_c)


section("fp16 LayerNorm", test_fp16_layernorm)


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 50)
total = passed + failed
print(f"Results: {passed} passed, {failed} failed out of {total} tests")
if failed:
    print("Failed tests:", errors)
    sys.exit(1)
else:
    print("All tests passed!")
