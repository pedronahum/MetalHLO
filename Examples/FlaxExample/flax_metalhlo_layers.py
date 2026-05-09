#!/usr/bin/env python3
"""
MetalHLO PJRT Plugin — Flax Layers Test Suite (forward only)

Covers Flax layers that the main flax_metalhlo_example.py skipped:
  - BatchNorm (mutable running stats)
  - Dropout (RNG via rngs argument)
  - ConvTranspose (upsampling)
  - depthwise / grouped Conv
  - max_pool (we already test avg_pool)
  - GELU activation (uses chlo.erf composite)
  - Causal-masked self-attention
  - Simple LSTM / GRU cell (single step)
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
            passed += 1
            print(f"  PASS  {name} (max diff "
                  f"{float(np.abs(g.astype(np.float64) - e.astype(np.float64)).max()):.2e})")
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


# ── Tests ────────────────────────────────────────────────────────────

print("MetalHLO Flax Layers Test Suite")
print("=" * 50)
print(f"JAX:    {jax.__version__}")
print(f"Flax:   {flax.__version__}")
print(f"CPU:    {cpu}")
print(f"GPU:    {device}")
print()


def test_batchnorm():
    print("1. BatchNorm")
    print("-" * 40)
    np.random.seed(0)
    x = np.random.randn(8, 16).astype(np.float32)

    # Inference mode (use_running_average=True)
    model = nn.BatchNorm(use_running_average=True)
    rng = jax.random.PRNGKey(0)
    with jax.default_device(cpu):
        variables = model.init(rng, jnp.asarray(x))
        expected = model.apply(variables, jnp.asarray(x))

    variables_m = to_device(variables, device)
    got = model.apply(variables_m, jax.device_put(jnp.asarray(x), device))
    check("batchnorm_inference", got, expected, rtol=1e-4, atol=1e-4)

    # Training mode — has mutable batch_stats
    model_train = nn.BatchNorm(use_running_average=False)
    with jax.default_device(cpu):
        variables = model_train.init(rng, jnp.asarray(x))
        expected, mutated_c = model_train.apply(
            variables, jnp.asarray(x), mutable=["batch_stats"]
        )

    variables_m = to_device(variables, device)
    got, mutated_m = model_train.apply(
        variables_m, jax.device_put(jnp.asarray(x), device),
        mutable=["batch_stats"],
    )
    check("batchnorm_training_output", got, expected, rtol=1e-4, atol=1e-4)
    # Compare updated running stats
    check("batchnorm_running_mean",
          mutated_m["batch_stats"]["mean"],
          mutated_c["batch_stats"]["mean"], rtol=1e-4, atol=1e-4)
    check("batchnorm_running_var",
          mutated_m["batch_stats"]["var"],
          mutated_c["batch_stats"]["var"], rtol=1e-4, atol=1e-4)


section("BatchNorm", test_batchnorm)


def test_dropout_deterministic():
    print("2. Dropout (deterministic)")
    print("-" * 40)
    # In deterministic mode, dropout is identity. Test that the module
    # forward pass works (no crash from jax.random plumbing) and matches CPU.
    np.random.seed(1)
    x = np.random.randn(4, 16).astype(np.float32)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x, deterministic):
            x = nn.Dense(8)(x)
            x = nn.Dropout(rate=0.5, deterministic=deterministic)(x)
            x = nn.relu(x)
            x = nn.Dense(4)(x)
            return x

    model = Net()
    rng = jax.random.PRNGKey(1)
    with jax.default_device(cpu):
        params = model.init(rng, jnp.asarray(x), deterministic=True)
        expected = model.apply(params, jnp.asarray(x), deterministic=True)
    params_m = to_device(params, device)
    got = model.apply(params_m, jax.device_put(jnp.asarray(x), device), deterministic=True)
    check("dropout_deterministic", got, expected, rtol=1e-4, atol=1e-4)


section("Dropout deterministic", test_dropout_deterministic)


def test_dropout_with_rng():
    print("3. Dropout (non-deterministic, fixed RNG)")
    print("-" * 40)
    # Compare same dropout RNG on CPU and metalhlo: since the RNG is
    # deterministic given a key, both should produce identical masks.
    np.random.seed(2)
    x = np.random.randn(4, 32).astype(np.float32)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(16)(x)
            x = nn.Dropout(rate=0.3, deterministic=False)(x)
            return x

    model = Net()
    init_rng = jax.random.PRNGKey(0)
    drop_rng = jax.random.PRNGKey(42)
    with jax.default_device(cpu):
        params = model.init({"params": init_rng, "dropout": drop_rng}, jnp.asarray(x))
        expected = model.apply(params, jnp.asarray(x), rngs={"dropout": drop_rng})
    params_m = to_device(params, device)
    got = model.apply(
        params_m,
        jax.device_put(jnp.asarray(x), device),
        rngs={"dropout": drop_rng},
    )
    check("dropout_rngd", got, expected, rtol=1e-4, atol=1e-4)


section("Dropout RNG", test_dropout_with_rng)


def test_conv_transpose():
    print("4. ConvTranspose")
    print("-" * 40)
    np.random.seed(3)
    x = np.random.randn(2, 8, 8, 4).astype(np.float32)
    model = nn.ConvTranspose(features=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME")
    rng = jax.random.PRNGKey(3)
    with jax.default_device(cpu):
        params = model.init(rng, jnp.asarray(x))
        expected = model.apply(params, jnp.asarray(x))
    params_m = to_device(params, device)
    got = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("conv_transpose_2x2_stride", got, expected, rtol=5e-4, atol=5e-4)


section("ConvTranspose", test_conv_transpose)


def test_grouped_conv():
    print("5. Grouped/depthwise Conv")
    print("-" * 40)
    np.random.seed(4)
    # Depthwise: feature_group_count = input channels
    x = np.random.randn(2, 16, 16, 8).astype(np.float32)
    model = nn.Conv(
        features=8, kernel_size=(3, 3), padding="SAME", feature_group_count=8
    )
    rng = jax.random.PRNGKey(4)
    with jax.default_device(cpu):
        params = model.init(rng, jnp.asarray(x))
        expected = model.apply(params, jnp.asarray(x))
    params_m = to_device(params, device)
    got = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("depthwise_conv2d", got, expected, rtol=5e-4, atol=5e-4)

    # Grouped: feature_group_count = 4 with 16 input, 8 output
    x2 = np.random.randn(2, 16, 16, 16).astype(np.float32)
    model2 = nn.Conv(
        features=8, kernel_size=(3, 3), padding="SAME", feature_group_count=4
    )
    with jax.default_device(cpu):
        p2 = model2.init(rng, jnp.asarray(x2))
        e2 = model2.apply(p2, jnp.asarray(x2))
    p2_m = to_device(p2, device)
    g2 = model2.apply(p2_m, jax.device_put(jnp.asarray(x2), device))
    check("grouped_conv2d_g4", g2, e2, rtol=5e-4, atol=5e-4)


section("Grouped Conv", test_grouped_conv)


def test_max_pool():
    print("6. max_pool")
    print("-" * 40)
    np.random.seed(5)
    x = np.random.randn(2, 8, 8, 4).astype(np.float32)

    class M(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    rng = jax.random.PRNGKey(5)
    model = M()
    with jax.default_device(cpu):
        params = model.init(rng, jnp.asarray(x))
        expected = model.apply(params, jnp.asarray(x))
    params_m = to_device(params, device)
    got = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("max_pool_2x2", got, expected, rtol=1e-5, atol=1e-5)


section("max_pool", test_max_pool)


def test_gelu():
    print("7. GELU activation")
    print("-" * 40)
    np.random.seed(6)
    x = np.random.randn(4, 16).astype(np.float32)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(32)(x)
            x = nn.gelu(x)
            x = nn.Dense(8)(x)
            return x

    model = Net()
    rng = jax.random.PRNGKey(6)
    with jax.default_device(cpu):
        params = model.init(rng, jnp.asarray(x))
        expected = model.apply(params, jnp.asarray(x))
    params_m = to_device(params, device)
    got = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("gelu_dense", got, expected, rtol=2e-4, atol=2e-4)

    # Also test approximate gelu (tanh-based, doesn't need erf)
    class NetApprox(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(32)(x)
            x = nn.gelu(x, approximate=True)
            x = nn.Dense(8)(x)
            return x

    model_a = NetApprox()
    with jax.default_device(cpu):
        pa = model_a.init(rng, jnp.asarray(x))
        ea = model_a.apply(pa, jnp.asarray(x))
    pa_m = to_device(pa, device)
    ga = model_a.apply(pa_m, jax.device_put(jnp.asarray(x), device))
    check("gelu_approx", ga, ea, rtol=2e-4, atol=2e-4)


section("GELU", test_gelu)


def test_causal_attention():
    print("8. Causal-masked self-attention")
    print("-" * 40)
    np.random.seed(7)
    x = np.random.randn(2, 8, 32).astype(np.float32)

    class Causal(nn.Module):
        @nn.compact
        def __call__(self, x):
            mask = nn.make_causal_mask(x[..., 0])
            return nn.MultiHeadDotProductAttention(
                num_heads=4, qkv_features=32, deterministic=True
            )(x, mask=mask)

    model = Causal()
    rng = jax.random.PRNGKey(7)
    with jax.default_device(cpu):
        params = model.init(rng, jnp.asarray(x))
        expected = model.apply(params, jnp.asarray(x))
    params_m = to_device(params, device)
    got = model.apply(params_m, jax.device_put(jnp.asarray(x), device))
    check("causal_mha", got, expected, rtol=1e-3, atol=1e-3)


section("Causal MHA", test_causal_attention)


def test_lstm_cell():
    print("9. LSTMCell single step")
    print("-" * 40)
    np.random.seed(8)
    x = np.random.randn(4, 16).astype(np.float32)
    h0 = np.zeros((4, 32), dtype=np.float32)
    c0 = np.zeros((4, 32), dtype=np.float32)

    cell = nn.LSTMCell(features=32)
    rng = jax.random.PRNGKey(8)
    with jax.default_device(cpu):
        params = cell.init(rng, (jnp.asarray(c0), jnp.asarray(h0)), jnp.asarray(x))
        e_state, e_out = cell.apply(
            params, (jnp.asarray(c0), jnp.asarray(h0)), jnp.asarray(x)
        )
    params_m = to_device(params, device)
    g_state, g_out = cell.apply(
        params_m,
        (jax.device_put(jnp.asarray(c0), device), jax.device_put(jnp.asarray(h0), device)),
        jax.device_put(jnp.asarray(x), device),
    )
    check("lstm_h", g_state[1], e_state[1], rtol=2e-4, atol=2e-4)
    check("lstm_c", g_state[0], e_state[0], rtol=2e-4, atol=2e-4)
    check("lstm_out", g_out, e_out, rtol=2e-4, atol=2e-4)


section("LSTMCell", test_lstm_cell)


def test_gru_cell():
    print("10. GRUCell single step")
    print("-" * 40)
    np.random.seed(9)
    x = np.random.randn(4, 16).astype(np.float32)
    h0 = np.zeros((4, 32), dtype=np.float32)

    cell = nn.GRUCell(features=32)
    rng = jax.random.PRNGKey(9)
    with jax.default_device(cpu):
        params = cell.init(rng, jnp.asarray(h0), jnp.asarray(x))
        e_state, e_out = cell.apply(params, jnp.asarray(h0), jnp.asarray(x))
    params_m = to_device(params, device)
    g_state, g_out = cell.apply(
        params_m,
        jax.device_put(jnp.asarray(h0), device),
        jax.device_put(jnp.asarray(x), device),
    )
    check("gru_h", g_state, e_state, rtol=2e-4, atol=2e-4)
    check("gru_out", g_out, e_out, rtol=2e-4, atol=2e-4)


section("GRUCell", test_gru_cell)


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 50)
total = passed + failed
print(f"Results: {passed} passed, {failed} failed out of {total} tests")
if failed:
    print("Failed tests:", errors)
    sys.exit(1)
else:
    print("All tests passed!")
