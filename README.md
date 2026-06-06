# MetalHLO

**StableHLO execution on Apple Metal.**

MetalHLO compiles and executes [StableHLO](https://github.com/openxla/stablehlo) MLIR
programs on Apple Silicon GPUs. It ships Swift, C, and
[PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt/c) APIs, so it works as a JAX
backend or as a standalone library for anything that emits StableHLO.

- **JAX backend** via the standard PJRT plugin — `import jax` runs on the Apple GPU.
- **Three execution backends** — MPSGraph (broad compatibility), custom Metal kernels
  (peak performance), and heterogeneous GPU+ANE+CPU (parallel execution across all three
  compute units that share unified memory).
- **XLA-style optimizer** — simplification, canonicalization, and pattern fusion
  (attention, FFN, LayerNorm, GELU, softmax) at O0–O3.
- **Training support** — full forward and backward passes; verified end-to-end against
  JAX CPU on ResNet18, nanoGPT, and Flax models.

Tested against **JAX 0.10.0** and **Flax 0.12.7** with optax 0.2.8.

## Requirements

- **macOS** 14.0+ (Sonoma); macOS 26+ for the MetalPerformancePrimitives matmul path
- **Swift** 6.0+, **Xcode** 15.0+
- **Apple Silicon** (M1/M2/M3/M4/M5)

## Installation

### Swift Package Manager

There are no tagged releases yet, so depend on the `main` branch (or pin a specific
commit):

```swift
dependencies: [
    .package(url: "https://github.com/pedronahum/MetalHLO.git", branch: "main")
]
```

### Build from source

```bash
git clone https://github.com/pedronahum/MetalHLO.git
cd MetalHLO
swift build
swift test

# Build the PJRT plugin for JAX integration
swift build -c release --product PJRTMetalHLO
```

### JAX backend

```bash
# Registers MetalHLO as a JAX backend (builds against the PJRT plugin above)
pip install -e python/
```

If the plugin dylib isn't found automatically, point to it explicitly:

```bash
export METALHLO_PLUGIN_PATH=/path/to/libPJRTMetalHLO.dylib
```

## Quick Start

### JAX

```python
import jax
import jax.numpy as jnp

print(jax.devices())          # includes the MetalHLO device

x = jnp.array([1.0, 2.0, 3.0, 4.0])
y = jnp.array([5.0, 6.0, 7.0, 8.0])
result = x + y                # executes on the Apple GPU via MetalHLO
```

`jit`, `grad`, `vmap`, `scan`, and full optax training steps all work — see
[JAX & Flax Compatibility](#jax--flax-compatibility).

### Swift

```swift
import MetalHLO

let client = try Client.create()

let mlir = """
module @add {
  func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
    %0 = stablehlo.add %a, %b : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""
let executable = try client.compile(mlir)                  // O2 by default

let a = client.createBuffer([1, 2, 3, 4] as [Float], shape: [4])
let b = client.createBuffer([10, 20, 30, 40] as [Float], shape: [4])
let outputs = try executable.execute([a, b])
print(try outputs[0].toFloatArray())                       // [11, 22, 33, 44]
```

Compilation presets: `.debug` (O0), `.fast` (O1), `.default` (O2), `.release` (O3 +
caching). Pass `devicePolicy: .auto` to enable heterogeneous GPU+ANE+CPU execution. The C
API mirrors this surface — see [`metalhlo.h`](Sources/CMetalHLO/include/metalhlo.h).

## JAX & Flax Compatibility

MetalHLO is a fully-functional JAX backend. Each row links to the test that exercises it.

### JAX primitives

| Capability | Test |
|---|---|
| `jit`, `value_and_grad`, full training step with optax (SGD, Adam) | [flax_metalhlo_training.py](Examples/FlaxExample/flax_metalhlo_training.py) |
| `vmap`, `vmap` of `grad`, nested `vmap`, non-default `in_axes`/`out_axes` | [flax_metalhlo_vmap.py](Examples/FlaxExample/flax_metalhlo_vmap.py) |
| `jax.lax.scan`, `flax.linen.scan` forward, `nn.scan` + `grad` (RNN training) | [flax_metalhlo_scan.py](Examples/FlaxExample/flax_metalhlo_scan.py) |
| `jax.checkpoint` / `jax.remat`, `jax.lax.optimization_barrier` | [OptimizationBarrierTest.swift](Tests/MetalHLOCoreTests/OptimizationBarrierTest.swift) |
| `jax.lax.top_k`, `jnp.argmax`/`argmin` (in-`jit`), `jnp.cumsum`/`cumprod`/`cummax`/`cumlogsumexp`, `jnp.arctan2`, `jax.lax.reduce_precision`, `jnp.searchsorted`, `jnp.unique(size=K)`, `jnp.sort`/`argsort`/`lexsort` | [Tests/MetalHLOCoreTests](Tests/MetalHLOCoreTests) |
| `jax.lax.switch` (multi-branch), `jax.lax.cond` | [CaseSwitchTest.swift](Tests/MetalHLOCoreTests/CaseSwitchTest.swift) |
| `jax.scipy.linalg.cholesky` / `solve_triangular`, `jnp.linalg.svd` / `qr` / `eigh` (routed from JAX's LAPACK FFI to Accelerate), `jax.lax.lgamma` / `digamma` / `erf` | [LapackRoutingTest.swift](Tests/MetalHLOCoreTests/LapackRoutingTest.swift) |
| `jax.random` — bit-exact threefry2x32 (`bits`/`uniform` match JAX CPU exactly) | [ThreefryRngTest.swift](Tests/MetalHLOCoreTests/ThreefryRngTest.swift) |
| `jax.debug.print` / `jax.debug.callback` (side-effect-only host callbacks) | [HostCallbackTest.swift](Tests/MetalHLOCoreTests/HostCallbackTest.swift) |
| `jax.shard_map` + `jax.lax.psum` / `pmean` (data-parallel, `METALHLO_NUM_DEVICES=N`, single-process sim — experimental) | [ManualComputationTest.swift](Tests/MetalHLOCoreTests/ManualComputationTest.swift), [AllReduceTest.swift](Tests/MetalHLOCoreTests/AllReduceTest.swift) |

### Dtypes

| Dtype | Forward | Grad | Mixed precision |
|---|---|---|---|
| `float32` | ✓ | ✓ | n/a |
| `float16` | ✓ | ✓ | ✓ (fp32 params + fp16 compute) |
| `bfloat16` | ✓ | ✓ | ✓ (fp32 params + bf16 compute), Adam step |

### Flax layers (`nn.compact` and `nnx`)

| Layer family | Verified |
|---|---|
| Dense, MLP, Sequential, ReLU/Tanh/SiLU/GELU, Softmax, Embed, Classifier | [flax_metalhlo_example.py](Examples/FlaxExample/flax_metalhlo_example.py) |
| Conv1D/2D, ConvTranspose, depthwise/grouped conv, max/avg pool | [flax_metalhlo_layers.py](Examples/FlaxExample/flax_metalhlo_layers.py) |
| BatchNorm (train + inference), LayerNorm, RMSNorm, GroupNorm, Dropout | [flax_metalhlo_layers.py](Examples/FlaxExample/flax_metalhlo_layers.py) |
| MultiHeadDotProductAttention (+ causal masking), Transformer block | [flax_metalhlo_layers.py](Examples/FlaxExample/flax_metalhlo_layers.py) |
| LSTMCell, GRUCell | [flax_metalhlo_layers.py](Examples/FlaxExample/flax_metalhlo_layers.py) |
| `flax.nnx`: Linear, Conv, LayerNorm, RMSNorm, Embed, MLP, MHA, Dropout, BatchNorm | [flax_metalhlo_nnx.py](Examples/FlaxExample/flax_metalhlo_nnx.py) |
| End-to-end: Mini-BERT, Mini-ResNet, Mini-CNN, Autoencoder (multi-step Adam) | [flax_metalhlo_e2e.py](Examples/FlaxExample/flax_metalhlo_e2e.py) |
| ResNet18 on CIFAR-10 (batch 256, fp32, Adam) — **8.7× over JAX CPU** (M5 Pro) | [Examples/Benchmarks/resnet_cifar10](Examples/Benchmarks/resnet_cifar10) |
| Karpathy's atomic GPT (1-layer, names dataset) | [Examples/Benchmarks/karpathy_gpt](Examples/Benchmarks/karpathy_gpt) |

## Optimization Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| **O0** | No optimization | Debugging, fastest compilation |
| **O1** | Algebraic simplification, DCE, constant folding | Quick iteration |
| **O2** (default) | + shape canonicalization, CSE, pattern fusion (softmax/GELU/LayerNorm/attention), producer-consumer fusion | Production |
| **O3** | + multi-iteration, sibling/horizontal/cross-layer fusion, layout optimization — **experimental** | See note below |

> **⚠️ O3 is experimental and under repair.** Its pattern-fusion stack has known
> correctness/stability bugs (a fused-attention kernel that drops the causal mask, a
> fusedGELU binding crash, an incomplete cross-layer residual fusion). Requesting `-O3`
> transparently falls back to `-O2` with a one-time warning. Set `METALHLO_ALLOW_O3=1`
> to force the real O3 (may crash or miscompile). **Use O2 for production.**

The pipeline runs in phases — Simplification → Canonicalization → Pattern Fusion →
Generic Fusion → Layout & Scheduling → Metal kernel generation.

## Architecture

```
  JAX / XLA            C/C++ projects          Swift projects
      │                      │                       │
  PJRT plugin            C API                  Swift API
  GetPjrtApi()           mhlo_*                 MetalHLO.Client
      └──────────────────────┴───────────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  MetalHLOCore       │
                  │  Parser → Optimizer │
                  │      (PassManager)  │
                  └──────────┬──────────┘
            ┌────────────────┼────────────────┐
            ▼                ▼                 ▼
       ┌─────────┐     ┌──────────┐    ┌──────────────┐
       │ MPSGraph│     │  Metal   │    │ Heterogeneous│
       │(default)│     │ kernels  │    │ GPU+ANE+CPU  │
       └─────────┘     └──────────┘    └──────────────┘
            └────────────────┴─────────────────┘
                             ▼
                Apple Metal / MPSGraph / ANE / CPU
```

**Heterogeneous execution** (`devicePolicy: .auto`) is the one differentiator worth
calling out: Apple Silicon's unified memory lets the GPU, Neural Engine (via MPS), and CPU
cores read/write the same buffers with zero transfer cost. A profitability-gated 4-pass
pipeline partitions only the operations where multi-unit dispatch provably wins — in
practice, vocabulary-scale projections (≥ 10M output elements, N ≥ 32K). Everything else
falls through to single-unit execution with zero overhead. Validated on GPT-2 (124M):
1.91× on the logit projection, zero regressions. See
[docs/BENCHMARKS.md](docs/BENCHMARKS.md#heterogeneous-execution-gpt-2-validation).

## Supported Operations

Broad StableHLO coverage — the operations below span ~99% of what production ML
workloads need.

| Category | Operations |
|----------|------------|
| **Binary** | add, subtract, multiply, divide, maximum, minimum, power, atan2 |
| **Unary** | negate, abs, exp, log, sqrt, rsqrt, sin, cos, tan, tanh, floor, ceil, sign, logistic, is_finite, expm1, log1p, cbrt, round_nearest_afz, round_nearest_even |
| **Bitwise** | not, and, or, xor, shift_left, shift_right_arithmetic, shift_right_logical, popcnt |
| **Type conversion** | convert, bitcast_convert, reduce_precision |
| **Matrix** | dot, dot_general, transpose, reshape, broadcast_in_dim, reverse |
| **Dynamic shape** | dynamic_slice, dynamic_update_slice, dynamic_reshape, dynamic_broadcast_in_dim, dynamic_pad, dynamic_iota, dynamic_gather |
| **Reduction** | reduce (sum/max/min/mean), reduce_window, argmax/argmin, cumulative (cumsum/cumprod/cummax/cumlogsumexp) |
| **Normalization** | batch_norm_inference, batch_norm_training, batch_norm_grad |
| **FFT** | fft (FFT, IFFT, RFFT, IRFFT) |
| **Sorting** | sort / argsort / lexsort (stable, any axis), top_k, searchsorted |
| **Special** | erf, lgamma, digamma |
| **Comparison** | compare (EQ/NE/LT/LE/GT/GE), select, clamp |
| **Indexing** | slice, pad, concatenate, gather, scatter, select_and_scatter |
| **Convolution** | convolution |
| **RNG** | rng (uniform, normal), rng_bit_generator (threefry2x32, bit-exact with JAX) |
| **Control flow** | while, if (`cond`), case (`switch`), optimization_barrier (`remat`) |
| **Constants** | constant, iota |
| **Quantization** | uniform_quantize, uniform_dequantize |
| **Complex** | complex, real, imag |
| **Custom calls** | fused attention / depth-attention / layer_norm / rms_norm / matmul-bias-activation / softmax / gelu / rope |

**Linear algebra.** The dense `jnp.linalg` / `jax.scipy.linalg` surface is reachable
end-to-end: JAX-CPU lowers to LAPACK FFI `custom_call`s, which the parser decodes and
routes to native MPSGraph (`cholesky`, `triangular_solve`) or to Apple's Accelerate
framework run host-side over the shared buffers (`svd`, `qr`, `eigh` — the same LAPACK
JAX-CPU itself calls). Verified against JAX CPU by reconstruction.

**Collectives (experimental).** `stablehlo.all_reduce` is supported, and
`jax.shard_map` with `jax.lax.psum` / `pmean` runs end-to-end — see **Distributed
(experimental)** under [Limitations](#limitations).

**Excluded by design.** Communication primitives (infeed, outfeed) and tuple ops.

## Performance

Two campaigns, full data in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**:

- **Multi-backend (M1)** — across MetalHLO's own backends, Metal kernels beat MPSGraph
  1.2–3.4× on element-wise ops, batch norm, transpose/reshape, and MLP/FFN inference;
  MPSGraph stays ahead on large GEMMs, convolutions, layer norm, and training.
- **vs MLX (M5 Pro)** — at -O3 with `METALHLO_MATMUL_TF32=1`, all 5 model_mlp benchmarks
  beat MLX (FFN fusion, up to 4.63×), normalization reaches parity, and GEMM 4096² hits
  0.91× via the MetalPerformancePrimitives matrix coprocessor. Attention auto-routes to
  MPSGraph's native `scaledDotProductAttention` and now **beats MLX on all four ATTN
  benchmarks** (1.0–1.76×, model_transformer 0.69×→0.97×). Reductions are now ~0.85× geomean — global
  (0.27×→0.69×, 2-stage split) and column (0.48×→0.71×, coalesced kernel) reductions were
  both fixed. Convolution is at parity (~0.95×, wins 5/8): any conv-containing graph
  auto-routes to Apple's `MPSCNNConvolution` (the path that trains ResNet18 at 8.7×), so the
  naive codegen conv kernel is only reached via `METALHLO_CONV_MPSGRAPH=0`.

**End-to-end training vs JAX CPU (M5 Pro):**

- **ResNet18 / CIFAR-10** (batch 256, fp32, Adam, Flax NNX) — **0.276 s/step vs 2.40 s on
  JAX CPU = 8.7×**, three runs at zero variance. Loss tracks the CPU reference (1.729 vs
  1.725 at step 30). See [the benchmark](Examples/Benchmarks/resnet_cifar10).
- **nanoGPT** (6L/384d, batch 16, seq 256) — 74.4 ms/step; here the yardstick is MLX
  (59.5 ms, 1.25×) rather than CPU, since GPU-busy (~60.8 ms) already ≈ MLX's entire step.
  The residual is host overhead, bounded by reverse-mode autodiff making forward activations
  multi-use. Full investigation in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#end-to-end-training-gap-nanogpt--investigation-status).

**Tips:** reuse executables (compile once, execute many), batch inputs to amortize launch
overhead, enable caching, and profile with `executeWithTiming()`.

## Limitations

**Execution model.** One physical Apple GPU; static shapes only (no dynamic shape
inference); Apple Silicon + macOS only. Data parallelism runs across N *virtual*
devices on that one GPU (see Distributed below) — not real multi-GPU/multi-host yet.

**Distributed (experimental).** `jax.shard_map` with data-parallel collectives
(`jax.lax.psum` / `pmean`, i.e. `stablehlo.all_reduce`) runs end-to-end through
JAX → PJRT → MetalHLO. Set `METALHLO_NUM_DEVICES=N` to advertise N devices; JAX
places an N-device mesh and MetalHLO runs it as a **single-process simulation** —
the sharded body is desugared into a flat program (slice per shard → replay →
cross-shard combine → reassemble) and every shard executes on the one GPU. This
validates the full collective compiler+runtime path; a real multi-Mac transport
(e.g. RDMA over Thunderbolt) would replace the in-process combine. Not yet:
automatic `pjit` auto-sharding (needs XLA's SPMD partitioner), sharded outputs
(FSDP), the other collectives (`all_gather` / `reduce_scatter` / `collective_permute`
/ `all_to_all`), and non-leading-axis / multi-axis sharding.

**Numerics / known gaps.**
- `jax.random.normal` diverges ~2.4e-7 from JAX — the threefry *bits* are exact, but the
  downstream `ndtri` inverse-CDF rounds differently in fp32 (not the RNG itself).
- `jnp.linalg.qr` feeding a downstream op inside the same `jit` falls through to the
  unsupported MPSGraph path; returning the Q/R factors works. `triangular_solve` with
  `left_side=false` is not yet implemented.
- Host callbacks: `jax.debug.print`/`callback` work (side effect dropped, numerics exact);
  `pure_callback`/`io_callback` are unsupported and fail loudly.
- Multi-step CNN training may drift up to ~0.1 absolute after step 0 (MPSGraph internal
  small-op fusion); `nn.custom_vjp` and quantization training are untested.

**Unsupported types.** `i2`/`i4`/`ui2`/`ui4` (overflow semantics), exotic floats
(`f4E2M1FN`, etc.), 64-bit integer bitwise, and full complex arithmetic. For compatibility,
the runtime promotes unsupported types where it can (`f64`→`f32`, `bf16`→`f32`, small
floats→`f16`, integers→`f32`). `dot`/`convolution`/`fft` are float-only in MPSGraph.

**Operation-specific.** `gather`/`scatter` support all opt levels and add/max/min/mul
scatter modes; batching dims are best-tested at leading positions. `reduce_window` and
`convolution` cover common patterns; unusual dimension permutations may fail. `while` loops
≤ 1000 iterations are unrolled inline; larger loops fall back to MPSGraph, which can crash
on complex multi-op loop bodies (`jax.lax.scan` with very large iteration counts) — under
active investigation.

## Testing

```bash
swift test --filter 'MetalHLOCoreTests'   # core compiler/optimizer (~764 tests)
swift test                                # full suite
swift test --no-parallel                  # most reliable (avoids GPU resource contention)
```

| Target | Description | Tests |
|--------|-------------|-------|
| `MetalHLOCoreTests` | Core compiler and optimizer | ~764 |
| `MetalHLOTests` | Integration and conformance | ~400+ |
| `PJRTMetalHLOTests` | PJRT plugin API and execution | 10 |

**StableHLO conformance:** 191 of 277 official interpreter tests pass, 0 fail, 86 skipped
for fundamental MPS/Metal limitations (complex types, small/large integers, integer
matmul/conv/FFT, exotic floats).

```bash
swift test --filter "Official" --no-parallel
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## References

- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [OpenXLA/XLA Compiler](https://github.com/openxla/xla)
- [MLX by Apple ML Research](https://github.com/ml-explore/mlx)
- [MPSGraph Documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [PJRT C API Specification](https://github.com/openxla/xla/tree/main/xla/pjrt/c)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
