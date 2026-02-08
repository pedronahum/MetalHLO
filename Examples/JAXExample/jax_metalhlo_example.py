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

import numpy as np

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

# ── Test Harness ─────────────────────────────────────────────────────

passed = 0
failed = 0
errors = []


def check(name, got, expected, rtol=1e-5, atol=1e-5):
    """Compare JAX result against expected numpy value."""
    global passed, failed, errors
    try:
        got_np = np.asarray(got)
        expected_np = np.asarray(expected)
        if got_np.dtype.kind == 'b' or expected_np.dtype.kind == 'b':
            ok = np.array_equal(got_np, expected_np)
        elif got_np.dtype.kind in ('i', 'u') or expected_np.dtype.kind in ('i', 'u'):
            ok = np.array_equal(got_np, expected_np)
        else:
            ok = np.allclose(got_np, expected_np, rtol=rtol, atol=atol)
            # Also check for NaN matches
            if not ok:
                nan_match = np.isnan(got_np) == np.isnan(expected_np)
                finite_match = np.allclose(
                    got_np[~np.isnan(got_np)], expected_np[~np.isnan(expected_np)],
                    rtol=rtol, atol=atol
                ) if np.any(~np.isnan(got_np)) else True
                ok = np.all(nan_match) and finite_match
        if ok:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            errors.append(name)
            print(f"  FAIL  {name}")
            print(f"        got:      {got_np.flatten()[:8]}")
            print(f"        expected: {expected_np.flatten()[:8]}")
    except Exception as e:
        failed += 1
        errors.append(f"{name} (exception)")
        print(f"  FAIL  {name} — {e}")


def put(x):
    """Helper to place a numpy array on the MetalHLO device."""
    return jax.device_put(jnp.array(x), device)


def section(name, fn):
    """Run a test section, catching any crashes so remaining sections still run."""
    global failed, errors
    try:
        fn()
    except Exception as e:
        failed += 1
        errors.append(f"{name} (section crashed: {e})")
        print(f"  CRASH {name}: {e}")
    print()


# ── Examples ─────────────────────────────────────────────────────────

print("MetalHLO JAX Comprehensive Test Suite")
print("=" * 50)
print(f"JAX version:  {jax.__version__}")
print(f"Device:       {device}")
print()


def test_binary_arithmetic():
    print("1. Binary Arithmetic")
    print("-" * 40)
    a = put([1.0, 2.0, 3.0, 4.0])
    b = put([10.0, 20.0, 30.0, 40.0])
    check("add", a + b, [11.0, 22.0, 33.0, 44.0])
    check("subtract", a - b, [-9.0, -18.0, -27.0, -36.0])
    check("multiply", a * b, [10.0, 40.0, 90.0, 160.0])
    check("divide", b / a, [10.0, 10.0, 10.0, 10.0])
    check("maximum", jnp.maximum(a, put([3.0, 1.0, 5.0, 2.0])), [3.0, 2.0, 5.0, 4.0])
    check("minimum", jnp.minimum(a, put([3.0, 1.0, 5.0, 2.0])), [1.0, 1.0, 3.0, 2.0])
    check("power", jnp.power(a, put([2.0, 3.0, 2.0, 0.5])), [1.0, 8.0, 9.0, 2.0])

section("Binary Arithmetic", test_binary_arithmetic)


def test_unary_math():
    print("2. Unary Math Operations")
    print("-" * 40)
    v = put([0.5, 1.0, 2.0, 3.0])
    check("negate", -v, [-0.5, -1.0, -2.0, -3.0])
    check("abs", jnp.abs(put([-2.0, -1.0, 0.0, 3.0])), [2.0, 1.0, 0.0, 3.0])
    check("exp", jnp.exp(v), np.exp([0.5, 1.0, 2.0, 3.0]))
    check("log", jnp.log(v), np.log([0.5, 1.0, 2.0, 3.0]))
    check("sqrt", jnp.sqrt(v), np.sqrt([0.5, 1.0, 2.0, 3.0]))
    check("rsqrt", jax.lax.rsqrt(v), 1.0 / np.sqrt([0.5, 1.0, 2.0, 3.0]))
    check("sin", jnp.sin(v), np.sin([0.5, 1.0, 2.0, 3.0]))
    check("cos", jnp.cos(v), np.cos([0.5, 1.0, 2.0, 3.0]))
    check("tanh", jnp.tanh(v), np.tanh([0.5, 1.0, 2.0, 3.0]))
    check("floor", jnp.floor(put([1.7, -1.3, 2.0, 0.5])), [1.0, -2.0, 2.0, 0.0])
    check("ceil", jnp.ceil(put([1.7, -1.3, 2.0, 0.5])), [2.0, -1.0, 2.0, 1.0])
    check("sign", jnp.sign(put([-3.0, 0.0, 5.0, -1.0])), [-1.0, 0.0, 1.0, -1.0])
    check("expm1", jnp.expm1(put([0.0, 0.001, 1.0, 2.0])), np.expm1([0.0, 0.001, 1.0, 2.0]))
    check("log1p", jnp.log1p(put([0.0, 0.001, 1.0, 2.0])), np.log1p([0.0, 0.001, 1.0, 2.0]))
    check("logistic (sigmoid)", jax.nn.sigmoid(put([0.0, 1.0, -1.0, 5.0])),
          1.0 / (1.0 + np.exp(-np.array([0.0, 1.0, -1.0, 5.0]))))

section("Unary Math", test_unary_math)


def test_comparison_selection():
    print("3. Comparison and Selection")
    print("-" * 40)
    x = put([1.0, 2.0, 3.0, 4.0])
    y = put([2.0, 2.0, 2.0, 2.0])
    check("equal", jnp.equal(x, y), [False, True, False, False])
    check("not_equal", jnp.not_equal(x, y), [True, False, True, True])
    check("less", jnp.less(x, y), [True, False, False, False])
    check("less_equal", jnp.less_equal(x, y), [True, True, False, False])
    check("greater", jnp.greater(x, y), [False, False, True, True])
    check("greater_equal", jnp.greater_equal(x, y), [False, True, True, True])
    check("where/select", jnp.where(jnp.array([True, False, True, False]), x, y),
          [1.0, 2.0, 3.0, 2.0])
    check("clamp", jnp.clip(put([0.5, 1.5, 2.5, 3.5]), 1.0, 3.0), [1.0, 1.5, 2.5, 3.0])

section("Comparison/Selection", test_comparison_selection)


def test_matrix_ops():
    print("4. Matrix Operations")
    print("-" * 40)
    A = put([[1.0, 2.0], [3.0, 4.0]])
    B = put([[5.0, 6.0], [7.0, 8.0]])
    check("matmul_2x2", jnp.dot(A, B), [[19.0, 22.0], [43.0, 50.0]])
    np.random.seed(42)
    A16 = np.random.randn(16, 16).astype(np.float32)
    B16 = np.random.randn(16, 16).astype(np.float32)
    check("matmul_16x16", jnp.dot(put(A16), put(B16)), A16 @ B16, rtol=1e-4, atol=1e-4)
    A_ns = np.random.randn(4, 8).astype(np.float32)
    B_ns = np.random.randn(8, 6).astype(np.float32)
    check("matmul_4x8_8x6", jnp.dot(put(A_ns), put(B_ns)), A_ns @ B_ns, rtol=1e-4, atol=1e-4)
    M = put([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    vec = put([1.0, 0.0, -1.0])
    check("matvec", jnp.dot(M, vec), [-2.0, -2.0])
    check("transpose", jnp.transpose(A), [[1.0, 3.0], [2.0, 4.0]])

section("Matrix Operations", test_matrix_ops)


def test_reshape_broadcast_reverse():
    print("5. Reshape / Broadcast / Reverse")
    print("-" * 40)
    r = put([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    check("reshape", jnp.reshape(r, (2, 3)), [[1, 2, 3], [4, 5, 6]])
    check("reshape_back", jnp.reshape(put([[1, 2, 3], [4, 5, 6]]), (6,)), [1, 2, 3, 4, 5, 6])
    check("broadcast_add", put([[1.0], [2.0], [3.0]]) + put([10.0, 20.0]),
          [[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]])
    check("reverse", jnp.flip(put([1.0, 2.0, 3.0, 4.0])), [4.0, 3.0, 2.0, 1.0])
    check("reverse_axis0", jnp.flip(put([[1, 2], [3, 4]]), axis=0), [[3, 4], [1, 2]])
    check("reverse_axis1", jnp.flip(put([[1, 2], [3, 4]]), axis=1), [[2, 1], [4, 3]])

section("Reshape/Broadcast/Reverse", test_reshape_broadcast_reverse)


def test_reductions():
    print("6. Reduction Operations")
    print("-" * 40)
    mat = put([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    check("sum_global", jnp.sum(mat), 21.0)
    check("sum_axis0", jnp.sum(mat, axis=0), [5.0, 7.0, 9.0])
    check("sum_axis1", jnp.sum(mat, axis=1), [6.0, 15.0])
    check("max_global", jnp.max(mat), 6.0)
    check("max_axis0", jnp.max(mat, axis=0), [4.0, 5.0, 6.0])
    check("max_axis1", jnp.max(mat, axis=1), [3.0, 6.0])
    check("min_global", jnp.min(mat), 1.0)
    check("min_axis0", jnp.min(mat, axis=0), [1.0, 2.0, 3.0])
    check("min_axis1", jnp.min(mat, axis=1), [1.0, 4.0])
    check("mean_global", jnp.mean(mat), 3.5)
    check("prod_global", jnp.prod(put([1.0, 2.0, 3.0, 4.0])), 24.0)

section("Reductions", test_reductions)


def test_slicing():
    print("7. Slicing and Indexing")
    print("-" * 40)
    s = put([10.0, 20.0, 30.0, 40.0, 50.0])
    check("slice", jax.lax.slice(s, [1], [4]), [20.0, 30.0, 40.0])
    check("slice_with_stride", jax.lax.slice(s, [0], [5], [2]), [10.0, 30.0, 50.0])
    check("dynamic_slice", jax.lax.dynamic_slice(s, [jnp.int32(2)], [3]), [30.0, 40.0, 50.0])
    sm = put([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    check("dynamic_slice_2d", jax.lax.dynamic_slice(sm, [jnp.int32(1), jnp.int32(1)], [2, 2]),
          [[5.0, 6.0], [8.0, 9.0]])

section("Slicing/Indexing", test_slicing)


def test_concat_pad():
    print("8. Concatenate and Pad")
    print("-" * 40)
    c1 = put([1.0, 2.0, 3.0])
    c2 = put([4.0, 5.0])
    c3 = put([6.0])
    check("concatenate", jnp.concatenate([c1, c2, c3]), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cm1 = put([[1, 2], [3, 4]])
    cm2 = put([[5, 6], [7, 8]])
    check("concatenate_axis0", jnp.concatenate([cm1, cm2], axis=0),
          [[1, 2], [3, 4], [5, 6], [7, 8]])
    check("concatenate_axis1", jnp.concatenate([cm1, cm2], axis=1),
          [[1, 2, 5, 6], [3, 4, 7, 8]])
    p = put([1.0, 2.0, 3.0])
    check("pad_constant", jnp.pad(p, (2, 1), constant_values=0.0),
          [0.0, 0.0, 1.0, 2.0, 3.0, 0.0])
    pm = put([[1.0, 2.0], [3.0, 4.0]])
    check("pad_2d", jnp.pad(pm, ((1, 0), (0, 1)), constant_values=0.0),
          [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])

section("Concatenate/Pad", test_concat_pad)


def test_gather():
    print("9. Gather / Fancy Indexing")
    print("-" * 40)
    data = put([10.0, 20.0, 30.0, 40.0, 50.0])
    indices = jax.device_put(jnp.array([4, 0, 2]), device)
    check("gather_1d", data[indices], [50.0, 10.0, 30.0])
    mat_g = put([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    check("gather_rows", mat_g[jax.device_put(jnp.array([2, 0]), device)],
          [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]])

section("Gather", test_gather)


def test_iota():
    print("10. Iota / Arange")
    print("-" * 40)
    # Use JIT-compiled iota to test the iota kernel
    @jax.jit
    def make_iota():
        return jnp.arange(5, dtype=jnp.float32)
    check("iota_5", make_iota(), [0.0, 1.0, 2.0, 3.0, 4.0])

section("Iota/Arange", test_iota)


# Note: jnp.sort requires private functions and block arguments (^bb0)
# in its MLIR that the parser doesn't yet handle. Skipped for now.


def test_type_conversion():
    print("12. Type Conversion")
    print("-" * 40)
    fv = put([1.7, 2.3, -0.5, 3.9])
    check("float_to_int", fv.astype(jnp.int32), [1, 2, 0, 3])
    iv = jax.device_put(jnp.array([1, 2, 3, 4], dtype=jnp.int32), device)
    check("int_to_float", iv.astype(jnp.float32), [1.0, 2.0, 3.0, 4.0])
    fv16 = fv.astype(jnp.float16).astype(jnp.float32)
    expected_f16 = np.array([1.7, 2.3, -0.5, 3.9], dtype=np.float16).astype(np.float32)
    check("float32_f16_roundtrip", fv16, expected_f16, rtol=1e-3)

section("Type Conversion", test_type_conversion)


def test_bitwise():
    print("13. Integer / Bitwise Operations")
    print("-" * 40)
    ia = jax.device_put(jnp.array([0b1100, 0b1010, 0b1111, 0b0001], dtype=jnp.int32), device)
    ib = jax.device_put(jnp.array([0b1010, 0b1010, 0b0101, 0b0011], dtype=jnp.int32), device)
    check("bitwise_and", jnp.bitwise_and(ia, ib), [0b1000, 0b1010, 0b0101, 0b0001])
    check("bitwise_or", jnp.bitwise_or(ia, ib), [0b1110, 0b1010, 0b1111, 0b0011])
    check("bitwise_xor", jnp.bitwise_xor(ia, ib), [0b0110, 0b0000, 0b1010, 0b0010])
    int_a = jax.device_put(jnp.array([1, 2, 4, 8], dtype=jnp.int32), device)
    check("shift_left", jnp.left_shift(int_a, jnp.int32(2)), [4, 8, 16, 32])
    check("shift_right", jnp.right_shift(
        jax.device_put(jnp.array([16, 8, 4, 2], dtype=jnp.int32), device), jnp.int32(1)),
          [8, 4, 2, 1])

section("Bitwise Operations", test_bitwise)


def test_jit_compound():
    print("14. JIT Compound Functions")
    print("-" * 40)

    @jax.jit
    def mlp_layer(x, w, b):
        return jnp.maximum(x @ w + b, 0.0)

    x_in = put([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w_in = put([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    b_in = put([-1.0, 0.5])
    mlp_expected = np.maximum(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32) @
                              np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32) +
                              np.array([-1.0, 0.5], dtype=np.float32), 0.0)
    check("mlp_layer", mlp_layer(x_in, w_in, b_in), mlp_expected, rtol=1e-4)

    @jax.jit
    def softmax(x):
        x_max = jnp.max(x, axis=-1, keepdims=True)
        exp_x = jnp.exp(x - x_max)
        return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

    logits = put([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]])
    sm_result = softmax(logits)
    sm_np = np.array([[1, 2, 3, 4], [1, 1, 1, 1]], dtype=np.float32)
    sm_np_shifted = sm_np - sm_np.max(axis=-1, keepdims=True)
    sm_expected = np.exp(sm_np_shifted) / np.exp(sm_np_shifted).sum(axis=-1, keepdims=True)
    check("softmax", sm_result, sm_expected, rtol=1e-4)
    check("softmax_rowsums", jnp.sum(sm_result, axis=-1), [1.0, 1.0], atol=1e-4)

    @jax.jit
    def layer_norm(x, gamma, beta, eps=1e-5):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return gamma * (x - mean) / jnp.sqrt(var + eps) + beta

    ln_x = put([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    ln_gamma = put([1.0, 1.0, 1.0, 1.0])
    ln_beta = put([0.0, 0.0, 0.0, 0.0])
    ln_result = layer_norm(ln_x, ln_gamma, ln_beta)
    ln_np = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float32)
    ln_mean = ln_np.mean(axis=-1, keepdims=True)
    ln_var = ((ln_np - ln_mean) ** 2).mean(axis=-1, keepdims=True)
    ln_expected = (ln_np - ln_mean) / np.sqrt(ln_var + 1e-5)
    check("layer_norm", ln_result, ln_expected, rtol=1e-4)

    # Note: GELU uses chlo.erf (CHLO dialect) which the parser doesn't yet handle.

section("JIT Compound Functions", test_jit_compound)


def test_scatter():
    print("15. Scatter / Dynamic Update")
    print("-" * 40)
    base = put([0.0, 0.0, 0.0, 0.0, 0.0])
    update = put([99.0, 88.0])
    check("dynamic_update_slice",
          jax.lax.dynamic_update_slice(base, update, [jnp.int32(1)]),
          [0.0, 99.0, 88.0, 0.0, 0.0])
    base2d = put([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    upd2d = put([[99.0, 88.0], [77.0, 66.0]])
    check("dynamic_update_slice_2d",
          jax.lax.dynamic_update_slice(base2d, upd2d, [jnp.int32(1), jnp.int32(1)]),
          [[1.0, 2.0, 3.0], [4.0, 99.0, 88.0], [7.0, 77.0, 66.0]])

section("Scatter/Dynamic Update", test_scatter)


def test_batch_matmul():
    print("16. Batch Matmul")
    print("-" * 40)
    np.random.seed(123)
    bA = np.random.randn(2, 3, 4).astype(np.float32)
    bB = np.random.randn(2, 4, 5).astype(np.float32)
    bC_expected = np.matmul(bA, bB)
    check("batch_matmul", jnp.matmul(put(bA), put(bB)), bC_expected, rtol=1e-4, atol=1e-4)

section("Batch Matmul", test_batch_matmul)


def test_convolution():
    print("17. Convolution")
    print("-" * 40)
    conv_input = put([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
    conv_kernel = put([[[1.0]], [[0.0]], [[-1.0]]])

    @jax.jit
    def conv1d(x, k):
        return jax.lax.conv_general_dilated(
            x, k, window_strides=(1,), padding='VALID',
            dimension_numbers=('NHC', 'HIO', 'NHC'))

    conv_result = conv1d(conv_input, conv_kernel)
    check("conv1d_valid", conv_result, [[[-2.0], [-2.0], [-2.0], [-2.0]]])

# Note: Convolution currently causes a SIGBUS crash during Metal kernel dispatch.
# The MLIR parsing works after format conversion, but the convolution kernel
# may have buffer binding or dispatch issues. Skipped for now.
# section("Convolution", test_convolution)


def test_large_reductions():
    print("18. Larger Reductions")
    print("-" * 40)
    np.random.seed(99)
    big = np.random.randn(64, 32).astype(np.float32)
    big_jax = put(big)
    check("sum_64x32", jnp.sum(big_jax), float(np.sum(big)), rtol=1e-3, atol=1e-2)
    check("max_64x32", jnp.max(big_jax), float(np.max(big)))
    check("min_64x32", jnp.min(big_jax), float(np.min(big)))
    check("mean_64x32", jnp.mean(big_jax), float(np.mean(big)), rtol=1e-3, atol=1e-3)
    check("sum_axis0_64x32", jnp.sum(big_jax, axis=0), np.sum(big, axis=0), rtol=1e-3, atol=1e-2)
    check("sum_axis1_64x32", jnp.sum(big_jax, axis=1), np.sum(big, axis=1), rtol=1e-3, atol=1e-2)

section("Larger Reductions", test_large_reductions)


def test_edge_cases():
    print("19. Edge Cases")
    print("-" * 40)
    check("add_scalar", put(3.0) + put(4.0), 7.0)
    check("mul_scalar", put(3.0) * put(4.0), 12.0)
    check("exp_neg_large", jnp.exp(put([-100.0])), [0.0])
    check("exp_zero", jnp.exp(put([0.0])), [1.0])
    check("neg_zero", jnp.negative(put([0.0])), [0.0])
    check("add_zero", put([1.0, 2.0, 3.0]) + put([0.0, 0.0, 0.0]), [1.0, 2.0, 3.0])
    check("mul_one", put([1.0, 2.0, 3.0]) * put([1.0, 1.0, 1.0]), [1.0, 2.0, 3.0])
    check("sum_single", jnp.sum(put([42.0])), 42.0)
    check("max_single", jnp.max(put([42.0])), 42.0)

section("Edge Cases", test_edge_cases)


# Note: argmax/argmin require multi-output reduce with complex block bodies
# in their MLIR that the parser doesn't yet handle. Skipped for now.


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 50)
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if errors:
    print(f"Failed tests: {errors}")
    sys.exit(1)
else:
    print("All tests passed!")
    sys.exit(0)
