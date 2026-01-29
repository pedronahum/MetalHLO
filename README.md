# MetalHLO

**StableHLO Execution on Apple Metal**

MetalHLO is a standalone library that compiles and executes [StableHLO](https://github.com/openxla/stablehlo) MLIR programs on Apple Silicon GPUs. It provides both Swift and C APIs, enabling integration with any project that emits StableHLO IR.

## Design Philosophy

MetalHLO draws inspiration from both [OpenXLA](https://github.com/openxla/xla) and [MLX](https://github.com/ml-explore/mlx), combining the best of both worlds:

**From OpenXLA:**
- **StableHLO as the IR** — A stable, portable intermediate representation for ML workloads
- **Multi-phase optimization pipeline** — Simplification → Canonicalization → Fusion → Layout → Scheduling
- **Pattern-based fusion** — Recognizing and fusing common patterns like attention, layer norm, GELU
- **Cost-model driven decisions** — Using analytical cost models to guide fusion decisions

**From MLX:**
- **Lazy graph compilation** — Build the computation graph, compile once, execute many times
- **Unified memory architecture** — Zero-copy data sharing between CPU and GPU on Apple Silicon
- **Simplicity first** — Clean APIs that make common cases easy and complex cases possible
- **Single-device focus** — Optimized for one GPU rather than distributed execution

**MetalHLO's unique contribution:**
- **Dual execution backends** — MPSGraph for broad compatibility, custom Metal kernels for peak performance
- **Progressive optimization levels** — O0 to O3 matching compiler conventions
- **C API for portability** — Integrate with any language that can call C functions

## Features

- **StableHLO Conformance** — 191 of 277 conformance tests pass (86 skipped for MPS/Metal limitations)
- **88% StableHLO Coverage** — 92 of 105 operations implemented
- **~99% Practical ML Coverage** — All operations needed for production ML workloads
- **Dual API** — Native Swift API + C API for C/C++ projects
- **Configurable Optimization** — O0 to O3 levels with algebraic simplification and operator fusion
- **Full Training Support** — Forward and backward pass operations
- **Apple Silicon Optimized** — Leverages MPSGraph and custom Metal kernels

### Supported Workloads

| Workload | Support |
|----------|---------|
| CNNs | Full (convolution, pooling, batch norm) |
| Transformers/MLPs | Full (attention, linear layers, activations) |
| RNNs | Full (while loops, dynamic operations) |
| Signal Processing | Full (FFT, IFFT, RFFT, IRFFT) |
| Quantized Models | Full (quantize/dequantize) |

## Requirements

- **macOS:** 14.0+ (Sonoma)
- **Swift:** 6.0+
- **Xcode:** 15.0+
- **Hardware:** Apple Silicon (M1/M2/M3/M4)

## Installation

### Swift Package Manager

Add MetalHLO to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/pedronahum/MetalHLO.git", from: "0.2.0")
]
```

Then add the dependency to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["MetalHLO"]
)
```

### Building from Source

```bash
git clone https://github.com/pedronahum/MetalHLO.git
cd MetalHLO
swift build
swift test
```

## Quick Start

### Swift — Basic Usage

```swift
import MetalHLO

// Create client
let client = try Client.create()
print("Using device: \(client.deviceName)")

// Compile StableHLO MLIR
let mlir = """
module @add {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""
let executable = try client.compile(mlir)

// Create input buffers
let a = client.createBuffer([1.0, 2.0, 3.0, 4.0] as [Float], shape: [4])
let b = client.createBuffer([10.0, 20.0, 30.0, 40.0] as [Float], shape: [4])

// Execute
let outputs = try executable.execute([a, b])
let result = try outputs[0].toFloatArray()
print("Result: \(result)")  // [11.0, 22.0, 33.0, 44.0]
```

### Swift — With Optimization Configuration

```swift
import MetalHLO

let client = try Client.create()

// Configure aggressive optimization (O3)
let config = CompilationConfig(optimizationLevel: .O3)
let executable = try client.compile(mlir, config: config)

// Or use presets
let debugExe = try client.compile(mlir, config: .debug)    // O0, no caching, debug info
let releaseExe = try client.compile(mlir, config: .release) // O3, caching enabled
let fastExe = try client.compile(mlir, config: .fast)       // O1, quick compilation
```

### C — Basic Usage

```c
#include <metalhlo.h>
#include <stdio.h>

int main() {
    MHLOClientRef client = NULL;
    MHLOStatusCode status;

    // Create client
    status = mhlo_client_create(&client);
    if (status != MHLO_OK) {
        fprintf(stderr, "Error: %s\n", mhlo_get_last_error());
        return 1;
    }

    // Print device info
    char* device = mhlo_client_device_name(client);
    printf("Device: %s\n", device);
    mhlo_free_string(device);

    // Compile MLIR
    const char* mlir =
        "module @add {\n"
        "  func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {\n"
        "    %0 = stablehlo.add %a, %b : tensor<4xf32>\n"
        "    return %0 : tensor<4xf32>\n"
        "  }\n"
        "}\n";

    MHLOExecutableRef exe = NULL;
    status = mhlo_compile(client, mlir, &exe);
    if (status != MHLO_OK) {
        fprintf(stderr, "Compile error: %s\n", mhlo_get_last_error());
        return 1;
    }

    // Create input buffers
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
    int64_t shape[] = {4};

    MHLOBufferRef buf_a = NULL, buf_b = NULL;
    mhlo_buffer_create(client, a_data, sizeof(a_data), shape, 1, MHLO_F32, &buf_a);
    mhlo_buffer_create(client, b_data, sizeof(b_data), shape, 1, MHLO_F32, &buf_b);

    // Execute
    MHLOBufferRef inputs[] = {buf_a, buf_b};
    MHLOBufferRef outputs[1] = {NULL};
    int32_t num_outputs = 0;

    status = mhlo_execute(exe, inputs, 2, outputs, &num_outputs);
    if (status != MHLO_OK) {
        fprintf(stderr, "Execute error: %s\n", mhlo_get_last_error());
        return 1;
    }

    // Read result
    float result[4];
    mhlo_buffer_to_host(outputs[0], result, sizeof(result));
    printf("Result: [%.1f, %.1f, %.1f, %.1f]\n",
           result[0], result[1], result[2], result[3]);

    // Cleanup
    mhlo_buffer_destroy(buf_a);
    mhlo_buffer_destroy(buf_b);
    mhlo_buffer_destroy(outputs[0]);
    mhlo_executable_destroy(exe);
    mhlo_client_destroy(client);

    return 0;
}
```

### C — With Optimization Configuration

```c
#include <metalhlo.h>

// Initialize config with defaults (O2)
MHLOCompileConfig config;
mhlo_compile_config_init(&config);

// Use aggressive optimization
config.optimization_level = MHLO_OPT_O3;
config.enable_caching = true;
config.enable_debug_info = false;

// Compile with configuration
MHLOExecutableRef exe = NULL;
mhlo_compile_with_config(client, mlir, &config, &exe);

// Get execution statistics after running
MHLOExecutionStats stats;
mhlo_executable_get_stats(exe, &stats);
printf("Executions: %lld, Avg time: %.3f ms\n",
       stats.execution_count, stats.average_execution_time_ms);
```

## Optimization Levels

MetalHLO provides four optimization levels that control the aggressiveness of the compilation pipeline:

| Level | Description | Use Case |
|-------|-------------|----------|
| **O0** | No optimization | Debugging, fastest compilation |
| **O1** | Basic optimization | Quick iteration during development |
| **O2** | Standard optimization (default) | Production with balanced compile time |
| **O3** | Aggressive optimization | Maximum runtime performance |

### What Each Level Does

**O0 — No Optimization**
- Direct translation to Metal kernels
- Fastest compile time
- Useful for debugging IR issues

**O1 — Basic Optimization**
- Algebraic simplification (`x + 0 → x`, `x * 1 → x`)
- Dead code elimination
- Constant folding

**O2 — Standard Optimization** (Default)
- All O1 optimizations
- Shape canonicalization (reshape/transpose/broadcast fusion)
- Common subexpression elimination
- Pattern-based fusion (softmax, GELU, layer norm, attention)
- Producer-consumer fusion

**O3 — Aggressive Optimization**
- All O2 optimizations
- Multiple fusion iterations
- Sibling fusion (multi-output fusion)
- Horizontal fusion (batching small operations)
- Cross-layer fusion
- Layout optimization

## Optimization Pipeline

MetalHLO's optimization pipeline runs in phases, inspired by XLA's approach:

```
                    ┌─────────────────────────────────────────┐
                    │           StableHLO MLIR Input          │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │     Phase 1: SIMPLIFICATION             │
                    │  • Constant folding                     │
                    │  • Algebraic simplification             │
                    │  • Dead code elimination                │
                    │  • Common subexpression elimination     │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │     Phase 2: CANONICALIZATION           │
                    │  • Reshape canonicalization             │
                    │  • Transpose canonicalization           │
                    │  • Broadcast canonicalization           │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │     Phase 3: PATTERN FUSION             │
                    │  • Softmax detection & fusion           │
                    │  • GELU/SiLU activation fusion          │
                    │  • Layer norm / RMS norm fusion         │
                    │  • Attention pattern fusion             │
                    │  • MatMul + Bias + Activation fusion    │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │     Phase 4: GENERIC FUSION             │
                    │  • Producer-consumer fusion             │
                    │  • Sibling fusion (multi-output)        │
                    │  • Horizontal fusion (op batching)      │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │     Phase 5: LAYOUT & SCHEDULING        │
                    │  • Memory layout optimization           │
                    │  • Buffer assignment                    │
                    │  • Kernel scheduling                    │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │         Metal Kernel Generation         │
                    └─────────────────────────────────────────┘
```

### Key Optimization Passes

| Pass | Description |
|------|-------------|
| **Algebraic Simplifier** | Applies identity rules (`x+0→x`, `x*1→x`), inverse rules (`exp(log(x))→x`), and strength reduction |
| **Dead Code Elimination** | Removes operations whose results are unused |
| **CSE** | Common Subexpression Elimination — reuses identical computations |
| **Reshape Canonicalizer** | Fuses consecutive reshapes, eliminates no-op reshapes |
| **Transpose Canonicalizer** | Composes consecutive transposes, eliminates identity transposes |
| **Broadcast Canonicalizer** | Fuses broadcasts, moves broadcasts through elementwise ops |
| **Pattern Fusion** | Recognizes ML patterns (softmax, GELU, attention) and replaces with optimized kernels |
| **Producer-Consumer Fusion** | Fuses elementwise operations with their consumers to reduce memory traffic |
| **Sibling Fusion** | Fuses operations that share inputs (multi-output fusion) |
| **Horizontal Fusion** | Batches multiple small independent operations |

## Supported Operations

### Fully Implemented (92 ops)

| Category | Operations |
|----------|------------|
| **Binary Arithmetic** | add, subtract, multiply, divide, maximum, minimum, power |
| **Unary Math** | negate, abs, exp, log, sqrt, rsqrt, sin, cos, tanh, floor, ceil, sign, tan, logistic, is_finite, expm1, log1p, cbrt, round_nearest_afz, round_nearest_even |
| **Bitwise** | not, and, or, xor, shift_left, shift_right_arithmetic, shift_right_logical, popcnt |
| **Type Conversion** | convert, bitcast_convert |
| **Matrix** | dot, dot_general, transpose, reshape, broadcast_in_dim, reverse |
| **Dynamic Shape** | dynamic_slice, dynamic_update_slice, dynamic_reshape, dynamic_broadcast_in_dim, dynamic_pad, dynamic_iota, dynamic_gather |
| **Convolution** | convolution |
| **Reduction** | reduce (sum, max, min, mean), reduce_window |
| **Normalization** | batch_norm_inference, batch_norm_training, batch_norm_grad |
| **FFT** | fft (FFT, IFFT, RFFT, IRFFT) |
| **Sorting** | sort |
| **Comparison** | compare (EQ, NE, LT, LE, GT, GE), select, clamp |
| **Indexing** | slice, pad, concatenate, gather, scatter |
| **RNG** | rng (uniform, normal), rng_bit_generator |
| **Constants** | constant, iota |
| **Control Flow** | while, if |
| **Quantization** | uniform_quantize, uniform_dequantize |
| **Complex Numbers** | complex, real, imag |
| **Select/Scatter** | select_and_scatter |
| **Custom Calls** | fused_scaled_dot_product_attention, fused_layer_norm, fused_rms_norm, fused_matmul_bias_activation, fused_softmax, fused_gelu, fused_rope |

### Operations Requiring MPS Kernel Bridging (3 ops)

| Operation | Notes |
|-----------|-------|
| `triangular_solve` | Requires MPSMatrixSolveTriangular |
| `cholesky` | Requires MPSMatrixDecompositionCholesky |
| `map` | Requires JIT compilation |

### Excluded by Design (14 ops)

Multi-device operations (all_gather, all_reduce, etc.), communication operations (infeed, outfeed, etc.), and tuple operations are excluded per the single-device focus.

## API Reference

### Swift API

#### Client

```swift
public final class Client: @unchecked Sendable {
    /// Create a client for the default Metal device
    static func create() throws -> Client

    /// The underlying Metal device name
    var deviceName: String { get }

    /// Compile StableHLO MLIR (default O2 optimization)
    func compile(_ mlir: String) throws -> Executable

    /// Compile with explicit configuration
    func compile(_ mlir: String, config: CompilationConfig) throws -> Executable

    /// Create buffers from host data
    func createBuffer(_ data: [Float], shape: [Int]) -> Buffer
    func createBuffer<T: Numeric>(_ data: [T], shape: [Int], elementType: ElementType) throws -> Buffer
}
```

#### CompilationConfig

```swift
public struct CompilationConfig: Sendable {
    var optimizationLevel: OptimizationLevel  // .O0, .O1, .O2, .O3
    var enableCaching: Bool
    var generateDebugInfo: Bool

    // Presets
    static let `default`: CompilationConfig  // O2
    static let debug: CompilationConfig      // O0, no cache, debug info
    static let release: CompilationConfig    // O3, caching
    static let fast: CompilationConfig       // O1, caching
}
```

#### Executable

```swift
public final class Executable: @unchecked Sendable {
    var inputCount: Int { get }
    var outputCount: Int { get }
    var inputTypes: [TensorType] { get }
    var outputTypes: [TensorType] { get }

    func execute(_ inputs: [Buffer]) throws -> [Buffer]
    func executeWithTiming(_ inputs: [Buffer]) throws -> ([Buffer], ExecutionTiming)
}
```

#### Buffer

```swift
public final class Buffer: @unchecked Sendable {
    var shape: [Int] { get }
    var count: Int { get }
    var elementType: ElementType { get }

    func toFloatArray() throws -> [Float]
    func toInt32Array() throws -> [Int32]
    func toData() throws -> Data
}
```

### C API

```c
// Version
const char* mhlo_version(void);

// Client
MHLOStatusCode mhlo_client_create(MHLOClientRef* out_client);
void mhlo_client_destroy(MHLOClientRef client);
char* mhlo_client_device_name(MHLOClientRef client);

// Compilation
MHLOStatusCode mhlo_compile(MHLOClientRef client, const char* mlir, MHLOExecutableRef* out);
MHLOStatusCode mhlo_compile_with_config(MHLOClientRef client, const char* mlir,
                                         const MHLOCompileConfig* config, MHLOExecutableRef* out);
void mhlo_compile_config_init(MHLOCompileConfig* config);
void mhlo_executable_destroy(MHLOExecutableRef exe);

// Execution
MHLOStatusCode mhlo_execute(MHLOExecutableRef exe, const MHLOBufferRef* inputs, int32_t num_inputs,
                             MHLOBufferRef* outputs, int32_t* num_outputs);
MHLOStatusCode mhlo_executable_get_stats(MHLOExecutableRef exe, MHLOExecutionStats* stats);
void mhlo_executable_reset_stats(MHLOExecutableRef exe);

// Buffers
MHLOStatusCode mhlo_buffer_create(MHLOClientRef client, const void* data, size_t size,
                                   const int64_t* shape, int32_t rank, MHLOElementType type,
                                   MHLOBufferRef* out);
void mhlo_buffer_destroy(MHLOBufferRef buffer);
MHLOStatusCode mhlo_buffer_to_host(MHLOBufferRef buffer, void* out, size_t size);

// Utilities
const char* mhlo_get_last_error(void);
void mhlo_free_string(const char* str);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│   C/C++ Projects                        Swift Projects          │
│        │                                      │                 │
│        ▼                                      ▼                 │
│   ┌─────────────┐                    ┌───────────────────┐      │
│   │ C API       │                    │ Swift API         │      │
│   │ mhlo_*      │                    │ MetalHLO.Client   │      │
│   └──────┬──────┘                    └─────────┬─────────┘      │
│          │                                     │                │
│          └─────────────┬───────────────────────┘                │
│                        ▼                                        │
│          ┌─────────────────────────────────────────┐            │
│          │  MetalHLOCore                           │            │
│          │                                         │            │
│          │  ┌───────────┐   ┌───────────────────┐  │            │
│          │  │ Parser    │ → │ Optimizer         │  │            │
│          │  └───────────┘   │ (PassManager)     │  │            │
│          │                  └─────────┬─────────┘  │            │
│          │                            │            │            │
│          │           ┌────────────────┴────────┐   │            │
│          │           ▼                         ▼   │            │
│          │  ┌─────────────────┐  ┌──────────────┐  │            │
│          │  │ MPSGraph Backend│  │ Metal Kernel │  │            │
│          │  │ (default)       │  │ Backend (O3) │  │            │
│          │  └─────────────────┘  └──────────────┘  │            │
│          └─────────────────────────────────────────┘            │
│                              │                                  │
│                              ▼                                  │
│                ┌─────────────────────────────┐                  │
│                │  Apple Metal / MPSGraph     │                  │
│                └─────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## Examples

### Two-Layer MLP

```swift
let mlir = """
module @mlp {
  func.func @main(%x: tensor<32x784xf32>, %w1: tensor<784x256xf32>, %b1: tensor<256xf32>,
                  %w2: tensor<256x10xf32>, %b2: tensor<10xf32>) -> (tensor<32x10xf32>) {
    // Layer 1: y1 = relu(x @ w1 + b1)
    %0 = stablehlo.dot %x, %w1 : (tensor<32x784xf32>, tensor<784x256xf32>) -> tensor<32x256xf32>
    %1 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<32x256xf32>
    %zero1 = stablehlo.constant dense<0.0> : tensor<32x256xf32>
    %3 = stablehlo.maximum %2, %zero1 : tensor<32x256xf32>

    // Layer 2: y2 = x1 @ w2 + b2
    %4 = stablehlo.dot %3, %w2 : (tensor<32x256xf32>, tensor<256x10xf32>) -> tensor<32x10xf32>
    %5 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %6 = stablehlo.add %4, %5 : tensor<32x10xf32>
    return %6 : tensor<32x10xf32>
  }
}
"""

// Compile with O3 for production
let config = CompilationConfig.release
let executable = try client.compile(mlir, config: config)
```

### Transformer Attention (Auto-Fused)

```swift
// MetalHLO automatically detects and fuses this attention pattern
let mlir = """
module @attention {
  func.func @main(%q: tensor<1x8x512x64xf32>, %k: tensor<1x8x512x64xf32>,
                  %v: tensor<1x8x512x64xf32>) -> (tensor<1x8x512x64xf32>) {
    // Transpose K
    %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] : ... -> tensor<1x8x64x512xf32>

    // Q @ K^T
    %scores = stablehlo.dot_general %q, %kt, ... -> tensor<1x8x512x512xf32>

    // Scale by 1/sqrt(d_k)
    %scale = stablehlo.constant dense<0.125> : tensor<f32>
    %scaled = stablehlo.multiply %scores, %scale : tensor<1x8x512x512xf32>

    // Softmax
    %max = stablehlo.reduce %scaled, max over [3] : tensor<1x8x512x1xf32>
    %shifted = stablehlo.subtract %scaled, %max : tensor<1x8x512x512xf32>
    %exp = stablehlo.exponential %shifted : tensor<1x8x512x512xf32>
    %sum = stablehlo.reduce %exp, add over [3] : tensor<1x8x512x1xf32>
    %probs = stablehlo.divide %exp, %sum : tensor<1x8x512x512xf32>

    // Attention @ V
    %out = stablehlo.dot_general %probs, %v, ... -> tensor<1x8x512x64xf32>
    return %out : tensor<1x8x512x64xf32>
  }
}
"""

// With O2+, this pattern is detected and fused into a single optimized kernel
let exe = try client.compile(mlir, config: .release)
```

## Testing

Run the full test suite:

```bash
swift test
```

Run specific test suites:

```bash
swift test --filter "Binary"
swift test --filter "Reduction"
swift test --filter "CAPITests"
swift test --filter "AlgebraicSimplifier"
```

### Test Coverage

| Test Suite | Tests |
|------------|-------|
| Lexer & Parser | 25 |
| Binary Operations | 7 |
| Unary Operations | 12 |
| Matrix Operations | 10 |
| Activation Functions | 35 |
| CNN Operations | 10 |
| Control Flow | 8 |
| Custom Call Handlers | 46 |
| Optimizer Passes | 50 |
| C API | 15 |
| Integration Tests | 30 |
| **StableHLO Conformance** | **191** |
| **Total** | **439** |

## Limitations

### Execution Model
1. **Static Shapes Only** — Dynamic shapes require shape inference
2. **Single Device** — No multi-device or distributed execution
3. **Apple Silicon Only** — Intel Macs with AMD GPUs are not supported
4. **macOS Only** — iOS/iPadOS support is a future consideration

### Unsupported Types
The following types are not supported due to Metal/MPS limitations:

| Type | Status | Reason |
|------|--------|--------|
| `i2`, `i4`, `ui2`, `ui4` | Not supported | Incompatible overflow semantics when promoted |
| `complex64`, `complex128` | Partial | Basic operations work; full arithmetic not supported by MPS |
| `f4E2M1FN`, `f6E2M3FN`, etc. | Not supported | Exotic float types not supported by Metal |
| 64-bit integer bitwise | Limited | MPS doesn't support 64-bit integer bitwise operations |
| Distributed/collective ops | Not supported | Single-device focus (all_gather, all_reduce, etc.) |
| Token operations | Not supported | (after_all, infeed, outfeed) - I/O primitives |

### Operation-Specific Type Restrictions

Some operations only support floating-point types in MPSGraph:

| Operation | Supported Types | Excluded Types |
|-----------|-----------------|----------------|
| `dot`, `dot_general` | `f16`, `f32`, `f64`, `bf16` | All integer types (MPSGraph matmul is float-only) |
| `convolution` | `f16`, `f32`, `f64`, `bf16` | All integer types (MPSGraph conv is float-only) |
| `fft` | `f16`, `f32`, `f64`, `bf16` | All integer types (MPSGraph FFT is float-only) |
| `triangular_solve` | Not implemented | Requires MPSMatrix bridge |
| `cholesky` | Not implemented | Requires MPSMatrix bridge |

### Operation-Specific Limitations

| Operation | Status | Notes |
|-----------|--------|-------|
| `gather` | Improved | Embedding lookup and index_vector_dim handling; batching dims supported when at leading positions |
| `scatter` | Improved | Supports add/max/min/mul computation modes via MPS; batching dims require leading positions |
| `dynamic_slice` | Working | Works when slice_sizes equal input dims (start indices clamped) |
| `reduce_window` | Partial | Works for common patterns; complex regions limited |
| `convolution` | Partial | Standard patterns work; complex dimension permutations may fail |

**Scatter computation modes:**
```mlir
// Supported via MPS scatter modes
scatter %operand, %indices, %updates, computation = add   // Adds update to existing value
scatter %operand, %indices, %updates, computation = max   // Takes maximum
scatter %operand, %indices, %updates, computation = min   // Takes minimum
scatter %operand, %indices, %updates, computation = mul   // Multiplies existing by update
scatter %operand, %indices, %updates                      // Default: replaces value
```

**Batching dimension support:**
Gather and scatter parse batching dimension attributes (`operand_batching_dims`, `start_indices_batching_dims` for gather; `input_batching_dims`, `scatter_indices_batching_dims` for scatter). The implementation includes a transpose-gather/scatter-transpose pattern for arbitrary batch dimension positions. Best tested with batch dimensions at leading positions; complex non-leading configurations may require additional work.

## StableHLO Conformance

MetalHLO includes a conformance test suite based on the [official StableHLO interpreter tests](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests).

### Conformance Results

| Metric | Count |
|--------|-------|
| **Total Tests in Suite** | 277 |
| **Tests Run** | 191 |
| **Passed** | 191 |
| **Failed** | 0 |
| **Skipped** | 86 (MPS/Metal limitations) |

### Running Conformance Tests

```bash
# Run all conformance tests (recommended: use --no-parallel for stability)
swift test --filter "Official" --no-parallel

# Run specific operation tests
swift test --filter "OfficialInterpretTests/testAdd"
swift test --filter "OfficialInterpretTests/testTranspose"
```

### Type Promotion

To maximize compatibility, MetalHLO automatically promotes unsupported types to supported ones:

| Source Type | Promoted To | Reason |
|-------------|-------------|--------|
| `f64` | `f32` | MPS doesn't support f64 |
| `bf16` | `f32` | Better precision for comparison |
| `f8E3M4`, `f8E4M3`, `f8E5M2` | `f16` | Exotic float types |
| `i8`, `i16`, `i32`, `i64` | `f32` | Integer arithmetic via float |
| `ui8`, `ui16`, `ui32`, `ui64` | `f32` | Unsigned integers via float |

This allows tests written for other backends to run correctly on Metal.

### Skipped Tests (Fundamental Limitations)

The following test categories are skipped due to fundamental MPS/Metal limitations:

| Category | Reason |
|----------|--------|
| Complex types (`complex64`, `complex128`) | MPS doesn't support complex arithmetic |
| Small integers (`i2`, `i4`, `ui2`, `ui4`) | Incompatible overflow semantics when promoted |
| Integer matmul (`dot`, `dot_general` with int types) | MPSGraph matmul only supports floating-point |
| Integer convolution | MPSGraph convolution only supports floating-point |
| Integer FFT | MPSGraph FFT only supports floating-point |
| Integer overflow tests | Float arithmetic doesn't wrap on overflow like integers |
| Integer division tests | Float division doesn't truncate like integer division |
| Large integer constants | Values like `INT64_MAX` can't be exactly represented in Float32 |
| Exotic float types (`f4E2M1FN`, etc.) | Not supported by Metal |
| Unsigned integer bitwise ops | MPS treats all integers as signed |
| 64-bit integer bitwise ops | MPS doesn't support 64-bit integer bitwise operations |

## Performance Tips

1. **Use O3 for production** — Aggressive fusion significantly reduces memory bandwidth
2. **Batch your inputs** — Larger batch sizes amortize kernel launch overhead
3. **Reuse executables** — Compile once, execute many times
4. **Enable caching** — Repeated compilations of the same MLIR return cached results
5. **Profile with timing** — Use `executeWithTiming()` to identify bottlenecks

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## References

- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [OpenXLA/XLA Compiler](https://github.com/openxla/xla)
- [MLX by Apple ML Research](https://github.com/ml-explore/mlx)
- [MPSGraph Documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)

---

**MetalHLO** — Bringing StableHLO to Apple Silicon
