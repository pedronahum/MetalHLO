# jax-metalhlo

Python package that registers MetalHLO as a JAX PJRT backend, enabling JAX computations on Apple Silicon GPUs via custom Metal kernels.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- JAX 0.4.0+ (tested with JAX 0.7.0)

## Installation

### 1. Build the PJRT plugin (Swift)

From the repository root:

```bash
swift build -c release --product PJRTMetalHLO
```

This produces `.build/release/libPJRTMetalHLO.dylib`.

### 2. Install the Python package

```bash
pip install -e python/
```

This registers `metalhlo` as a JAX plugin via the `jax_plugins` entry point. JAX will auto-discover it on import.

### 3. Verify installation

```python
import jax
print(jax.devices())  # Should include a MetalHLO device
```

## Configuration

By default the package looks for `libPJRTMetalHLO.dylib` in:

1. Alongside the installed package directory
2. `.build/release/` relative to the repo root
3. `.build/debug/` relative to the repo root

To override, set the `METALHLO_PLUGIN_PATH` environment variable:

```bash
export METALHLO_PLUGIN_PATH=/path/to/libPJRTMetalHLO.dylib
```

## Package Structure

```
python/
├── pyproject.toml                          # Package metadata and entry point
├── README.md
└── jax_plugins/
    └── metalhlo/
        ├── __init__.py                     # Plugin auto-discovery and registration
        └── bytecode_to_text.py             # MLIR bytecode-to-text converter
```

- **`__init__.py`** — Called by JAX's plugin discovery system. Locates the dylib and registers it via `xb.register_plugin()`.
- **`bytecode_to_text.py`** — Standalone script that converts MLIR bytecode (VHLO portable artifacts) to StableHLO text. Used by the Swift PJRT runtime (`PJRTExecutable.swift`) as a subprocess to deserialize programs sent by JAX. Strips JAX-specific annotations (SDY sharding, precision configs, result info) and normalizes MLIR syntax for the MetalHLO parser.

## Running Tests

### JAX integration test suite

The comprehensive test suite is at `Examples/JAXExample/jax_metalhlo_example.py`. It covers 95 tests across all supported operations:

```bash
python Examples/JAXExample/jax_metalhlo_example.py
```

#### Test sections

| # | Section | Tests | Operations covered |
|---|---------|-------|--------------------|
| 1 | Binary Arithmetic | 7 | add, subtract, multiply, divide, maximum, minimum, power |
| 2 | Unary Math | 15 | negate, abs, exp, log, sqrt, rsqrt, sin, cos, tanh, floor, ceil, sign, expm1, log1p, sigmoid |
| 3 | Comparison/Selection | 8 | equal, not_equal, less, less_equal, greater, greater_equal, where/select, clamp |
| 4 | Matrix Operations | 5 | matmul (2x2, 16x16, non-square), matvec, transpose |
| 5 | Reshape/Broadcast/Reverse | 6 | reshape, broadcast_add, reverse (1D, axis0, axis1) |
| 6 | Reductions | 11 | sum, max, min, mean, prod (global and per-axis) |
| 7 | Slicing/Indexing | 4 | slice, slice_with_stride, dynamic_slice (1D, 2D) |
| 8 | Concatenate/Pad | 5 | concatenate (1D, axis0, axis1), pad (1D, 2D) |
| 9 | Gather | 2 | gather_1d, gather_rows |
| 10 | Iota/Arange | 1 | iota via `jnp.arange` |
| 12 | Type Conversion | 3 | float-to-int, int-to-float, float16 roundtrip |
| 13 | Bitwise Operations | 5 | bitwise_and, bitwise_or, bitwise_xor, shift_left, shift_right |
| 14 | JIT Compound Functions | 4 | MLP layer (matmul+bias+relu), softmax, layer_norm |
| 15 | Scatter/Dynamic Update | 2 | dynamic_update_slice (1D, 2D) |
| 16 | Batch Matmul | 1 | batched matrix multiplication |
| 17 | Convolution | 1 | 1D convolution (VALID padding) |
| 18 | Larger Reductions | 6 | sum/max/min/mean on 64x32 matrices, per-axis |
| 19 | Edge Cases | 9 | scalar ops, exp edge values, identity operations |

### Quick smoke test

```python
import jax
import jax.numpy as jnp

device = jax.devices("metalhlo")[0]
x = jax.device_put(jnp.array([1.0, 2.0, 3.0]), device)
y = jax.device_put(jnp.array([4.0, 5.0, 6.0]), device)
print(x + y)  # [5. 7. 9.]
```

### Testing the bytecode converter directly

```bash
python -c "from jax_plugins.metalhlo.bytecode_to_text import main; print('OK')"
```

## Known Limitations

- Single-device execution only (no multi-GPU or distributed)
- Static shapes required (no dynamic shape inference at the PJRT level)
- `jnp.sort` and `argmax`/`argmin` are not yet supported (require block arguments the parser doesn't handle)
- GELU via `jax.nn.gelu` is not yet supported (uses CHLO dialect `chlo.erf`)
