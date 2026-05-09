#!/usr/bin/env python3
"""
MetalHLO PJRT Plugin — Flax Example

Demonstrates running Flax (linen) neural-network modules on Apple
Silicon GPUs via the MetalHLO PJRT plugin.

Each test:
  1. Initializes a Flax module on CPU and runs a forward pass there
     to produce a ground-truth reference.
  2. Moves params + input to the MetalHLO device and runs the same
     forward pass.
  3. Asserts the two outputs agree.

Prerequisites:
    1. Build the plugin:
       cd <repo-root>
       swift build -c release --product PJRTMetalHLO

    2. Install JAX, Flax, and the MetalHLO plugin:
       pip install "jax>=0.10,<0.11" flax
       pip install -e python/

    3. Run this example:
       python Examples/FlaxExample/flax_metalhlo_example.py
"""

import os
import sys
from functools import partial

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jax._src.xla_bridge as xb
    import jaxlib.xla_client as xla_client
except ImportError:
    print("ERROR: JAX is not installed.")
    print("  Install it with: pip install 'jax>=0.10,<0.11'")
    sys.exit(1)

try:
    import flax
    import flax.linen as nn
except ImportError:
    print("ERROR: Flax is not installed.")
    print("  Install it with: pip install flax")
    sys.exit(1)

# ── Plugin registration ──────────────────────────────────────────────

if not xla_client.pjrt_plugin_loaded("metalhlo"):
    repo_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dylib_candidates = [
        os.path.join(repo_root, ".build", "release", "libPJRTMetalHLO.dylib"),
        os.path.join(repo_root, ".build", "debug", "libPJRTMetalHLO.dylib"),
    ]
    found = next((p for p in dylib_candidates if os.path.isfile(p)), None)
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
    cpu_devices = jax.devices("cpu")
except Exception as e:
    print(f"ERROR: Required backends not available: {e}")
    sys.exit(1)

if not metalhlo_devices or not cpu_devices:
    print("ERROR: Need both CPU and MetalHLO devices.")
    sys.exit(1)

device = metalhlo_devices[0]
cpu = cpu_devices[0]

# ── Test harness ─────────────────────────────────────────────────────

passed = 0
failed = 0
errors = []


def _allclose(got, expected, rtol, atol):
    g = np.asarray(got)
    e = np.asarray(expected)
    if g.shape != e.shape:
        return False, f"shape mismatch got {g.shape} vs expected {e.shape}"
    if g.dtype.kind in ('i', 'u', 'b') or e.dtype.kind in ('i', 'u', 'b'):
        ok = np.array_equal(g, e)
    else:
        ok = np.allclose(g, e, rtol=rtol, atol=atol)
    return ok, None


def check(name, got, expected, rtol=1e-4, atol=1e-4):
    global passed, failed, errors
    try:
        ok, err = _allclose(got, expected, rtol, atol)
        if ok:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            errors.append(name)
            g = np.asarray(got).flatten()
            e = np.asarray(expected).flatten()
            n = min(6, g.size)
            print(f"  FAIL  {name}")
            if err:
                print(f"        {err}")
            print(f"        got     [:{n}]: {g[:n]}")
            print(f"        expected[:{n}]: {e[:n]}")
            if g.size and e.size and g.size == e.size:
                diff = np.abs(g - e)
                print(f"        max abs diff: {diff.max():.6g}")
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


def run_module(module, x, rng_seed=0, **init_kwargs):
    """Init params on CPU, return (cpu_out, metalhlo_out, expected_shape)."""
    rng = jax.random.PRNGKey(rng_seed)
    with jax.default_device(cpu):
        x_cpu = jnp.asarray(x)
        params = module.init(rng, x_cpu, **init_kwargs)
        expected = module.apply(params, x_cpu, **init_kwargs)
    params_m = to_device(params, device)
    x_m = jax.device_put(jnp.asarray(x), device)
    got = module.apply(params_m, x_m, **init_kwargs)
    return np.asarray(got), np.asarray(expected)


# ── Tests ────────────────────────────────────────────────────────────

print("MetalHLO Flax Comprehensive Test Suite")
print("=" * 50)
print(f"JAX version:    {jax.__version__}")
print(f"Flax version:   {flax.__version__}")
print(f"CPU device:     {cpu}")
print(f"GPU device:     {device}")
print()


def test_dense():
    print("1. Dense (linear layer)")
    print("-" * 40)
    np.random.seed(0)
    x = np.random.randn(8, 16).astype(np.float32)
    got, expected = run_module(nn.Dense(features=12), x)
    check("dense_8x16_to_8x12", got, expected)

    # No bias
    got, expected = run_module(nn.Dense(features=4, use_bias=False), x)
    check("dense_no_bias", got, expected)

    # Single batch
    x1 = np.random.randn(1, 8).astype(np.float32)
    got, expected = run_module(nn.Dense(features=3), x1)
    check("dense_1x8_to_1x3", got, expected)


section("Dense", test_dense)


def test_mlp():
    print("2. MLP (Dense → activation → Dense)")
    print("-" * 40)

    class MLP(nn.Module):
        hidden: int
        out: int
        act: str = "relu"

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.hidden)(x)
            if self.act == "relu":
                x = nn.relu(x)
            elif self.act == "tanh":
                x = jnp.tanh(x)
            elif self.act == "silu":
                x = nn.silu(x)
            x = nn.Dense(self.out)(x)
            return x

    np.random.seed(1)
    x = np.random.randn(4, 32).astype(np.float32)

    got, expected = run_module(MLP(hidden=64, out=10, act="relu"), x)
    check("mlp_relu_4x32_to_4x10", got, expected)

    got, expected = run_module(MLP(hidden=64, out=10, act="tanh"), x)
    check("mlp_tanh_4x32_to_4x10", got, expected)

    got, expected = run_module(MLP(hidden=64, out=10, act="silu"), x)
    check("mlp_silu_4x32_to_4x10", got, expected)


section("MLP", test_mlp)


def test_conv1d():
    print("3. Conv1D")
    print("-" * 40)
    np.random.seed(2)
    # NWC layout: (batch, length, channels)
    x = np.random.randn(2, 16, 4).astype(np.float32)
    got, expected = run_module(
        nn.Conv(features=8, kernel_size=(3,), padding="SAME"), x
    )
    check("conv1d_same_3", got, expected, rtol=2e-4, atol=2e-4)

    got, expected = run_module(
        nn.Conv(features=8, kernel_size=(3,), padding="VALID"), x
    )
    check("conv1d_valid_3", got, expected, rtol=2e-4, atol=2e-4)

    got, expected = run_module(
        nn.Conv(features=8, kernel_size=(5,), strides=(2,), padding="VALID"), x
    )
    check("conv1d_stride2_k5", got, expected, rtol=2e-4, atol=2e-4)


section("Conv1D", test_conv1d)


def test_conv2d():
    print("4. Conv2D")
    print("-" * 40)
    np.random.seed(3)
    # NHWC layout: (batch, H, W, C)
    x = np.random.randn(2, 8, 8, 3).astype(np.float32)
    got, expected = run_module(
        nn.Conv(features=4, kernel_size=(3, 3), padding="SAME"), x
    )
    check("conv2d_same_3x3", got, expected, rtol=3e-4, atol=3e-4)

    got, expected = run_module(
        nn.Conv(features=4, kernel_size=(3, 3), padding="VALID"), x
    )
    check("conv2d_valid_3x3", got, expected, rtol=3e-4, atol=3e-4)

    got, expected = run_module(
        nn.Conv(features=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME"), x
    )
    check("conv2d_stride2_3x3", got, expected, rtol=3e-4, atol=3e-4)


section("Conv2D", test_conv2d)


def test_layernorm():
    print("5. LayerNorm")
    print("-" * 40)
    np.random.seed(4)
    x = np.random.randn(4, 32).astype(np.float32)
    got, expected = run_module(nn.LayerNorm(), x)
    check("layernorm_4x32", got, expected, rtol=1e-4, atol=1e-4)

    x3 = np.random.randn(2, 5, 16).astype(np.float32)
    got, expected = run_module(nn.LayerNorm(), x3)
    check("layernorm_2x5x16", got, expected, rtol=1e-4, atol=1e-4)


section("LayerNorm", test_layernorm)


def test_rmsnorm():
    print("6. RMSNorm")
    print("-" * 40)
    np.random.seed(5)
    x = np.random.randn(4, 32).astype(np.float32)
    got, expected = run_module(nn.RMSNorm(), x)
    check("rmsnorm_4x32", got, expected, rtol=1e-4, atol=1e-4)

    x3 = np.random.randn(2, 5, 16).astype(np.float32)
    got, expected = run_module(nn.RMSNorm(), x3)
    check("rmsnorm_2x5x16", got, expected, rtol=1e-4, atol=1e-4)


section("RMSNorm", test_rmsnorm)


def test_groupnorm():
    print("7. GroupNorm")
    print("-" * 40)
    np.random.seed(6)
    # GroupNorm expects (..., features); use 2D
    x = np.random.randn(4, 16).astype(np.float32)
    got, expected = run_module(nn.GroupNorm(num_groups=4), x)
    check("groupnorm_g4_4x16", got, expected, rtol=1e-4, atol=1e-4)


section("GroupNorm", test_groupnorm)


def test_embedding():
    print("8. Embed")
    print("-" * 40)
    # Embed expects integer indices.
    # Known issue: produces NaN under MetalHLO. The Flax Embed lowering wraps
    # gather in an out-of-bounds mask (compare + and-reduce + select), and the
    # resulting select picks the NaN fill branch even for in-range indices —
    # likely a bug in how the mask reduce / select interacts with the gather.
    np.random.seed(7)
    idx = np.array([[0, 1, 2, 3], [4, 5, 6, 0]], dtype=np.int32)
    got, expected = run_module(nn.Embed(num_embeddings=10, features=8), idx)
    check("embed_lookup", got, expected, rtol=1e-5, atol=1e-5)


section("Embed", test_embedding)


def test_softmax_classifier():
    print("9. Softmax classifier (Dense + softmax)")
    print("-" * 40)

    class Classifier(nn.Module):
        num_classes: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.num_classes)(x)
            return jax.nn.softmax(x, axis=-1)

    np.random.seed(8)
    x = np.random.randn(8, 16).astype(np.float32)
    got, expected = run_module(Classifier(num_classes=5), x)
    check("classifier_softmax_8x5", got, expected, rtol=1e-4, atol=1e-4)
    # Softmax outputs sum to 1 along axis=-1
    sums = got.sum(axis=-1)
    check("softmax_rows_sum_to_1", sums, np.ones_like(sums), rtol=1e-4, atol=1e-4)


section("Softmax classifier", test_softmax_classifier)


def test_attention():
    print("10. Multi-head self-attention")
    print("-" * 40)
    # Known issue: numerical mismatch (~2.5 absolute) under MetalHLO.
    # MHA lowers to dot_general with multiple batch + contracting dims plus
    # softmax along sequence axis; one of those paths is producing wrong
    # values. Likely a bug in dot_general dim handling for high-rank inputs.
    np.random.seed(9)
    x = np.random.randn(2, 6, 16).astype(np.float32)
    mha = nn.MultiHeadDotProductAttention(num_heads=4, qkv_features=16, deterministic=True)
    got, expected = run_module(mha, x)
    check("mha_2x6x16_4heads", got, expected, rtol=5e-4, atol=5e-4)


section("MHA", test_attention)


def test_transformer_block():
    print("11. Transformer encoder block")
    print("-" * 40)
    # Known issue: fails by ~2.0 abs because it composes MHA (see test 10).

    class EncoderBlock(nn.Module):
        d_model: int
        num_heads: int
        d_ff: int

        @nn.compact
        def __call__(self, x):
            # Pre-LN style: LN → MHA → residual → LN → FFN → residual
            h = nn.LayerNorm()(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                deterministic=True,
            )(h)
            x = x + h
            h = nn.LayerNorm()(x)
            h = nn.Dense(self.d_ff)(h)
            h = nn.relu(h)
            h = nn.Dense(self.d_model)(h)
            return x + h

    np.random.seed(10)
    x = np.random.randn(2, 8, 32).astype(np.float32)
    got, expected = run_module(
        EncoderBlock(d_model=32, num_heads=4, d_ff=64), x
    )
    check("transformer_block_2x8x32", got, expected, rtol=1e-3, atol=1e-3)


section("Transformer block", test_transformer_block)


def test_small_cnn():
    print("12. Small CNN (Conv → ReLU → Pool → Conv → ReLU → Flatten → Dense)")
    print("-" * 40)

    class SmallCNN(nn.Module):
        num_classes: int = 10

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(features=8, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(self.num_classes)(x)
            return x

    np.random.seed(11)
    x = np.random.randn(2, 16, 16, 3).astype(np.float32)
    got, expected = run_module(SmallCNN(num_classes=10), x)
    check("small_cnn_2x10", got, expected, rtol=1e-3, atol=1e-3)


section("Small CNN", test_small_cnn)


def test_sequential_mlp():
    print("13. Sequential MLP")
    print("-" * 40)
    np.random.seed(12)
    x = np.random.randn(4, 16).astype(np.float32)
    seq = nn.Sequential([
        nn.Dense(32),
        nn.relu,
        nn.Dense(16),
        nn.relu,
        nn.Dense(4),
    ])
    got, expected = run_module(seq, x)
    check("sequential_4x16_to_4x4", got, expected, rtol=2e-4, atol=2e-4)


section("Sequential MLP", test_sequential_mlp)


def test_jit_inference():
    print("14. JIT-compiled inference")
    print("-" * 40)
    # Verify that wrapping the module in jax.jit doesn't break MetalHLO execution.
    np.random.seed(13)

    class TwoLayer(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(32)(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x)
            x = nn.Dense(8)(x)
            return x

    model = TwoLayer()
    x = np.random.randn(4, 16).astype(np.float32)
    rng = jax.random.PRNGKey(13)
    with jax.default_device(cpu):
        params = model.init(rng, jnp.asarray(x))
        expected = jax.jit(model.apply)(params, jnp.asarray(x))
    params_m = to_device(params, device)
    x_m = jax.device_put(jnp.asarray(x), device)
    got = jax.jit(model.apply)(params_m, x_m)
    check("jit_two_layer", got, expected, rtol=2e-4, atol=2e-4)


section("JIT inference", test_jit_inference)


def test_residual_chain():
    print("15. Residual chain")
    print("-" * 40)

    class ResidualMLP(nn.Module):
        depth: int
        width: int

        @nn.compact
        def __call__(self, x):
            for _ in range(self.depth):
                h = nn.Dense(self.width)(x)
                h = nn.relu(h)
                h = nn.Dense(self.width)(h)
                x = x + h
            return x

    np.random.seed(14)
    x = np.random.randn(4, 32).astype(np.float32)
    got, expected = run_module(ResidualMLP(depth=3, width=32), x)
    check("residual_mlp_d3_w32", got, expected, rtol=1e-3, atol=1e-3)


section("Residual chain", test_residual_chain)


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 50)
total = passed + failed
print(f"Results: {passed} passed, {failed} failed out of {total} tests")
if failed:
    print("Failed tests:", errors)
    sys.exit(1)
else:
    print("All tests passed!")
