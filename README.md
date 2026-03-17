# MetalHLO

**StableHLO Execution on Apple Metal**

MetalHLO is a standalone library that compiles and executes [StableHLO](https://github.com/openxla/stablehlo) MLIR programs on Apple Silicon GPUs. It provides Swift, C, and [PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt/c) APIs, enabling integration with JAX, XLA, and any project that emits StableHLO IR.

## Design Philosophy

MetalHLO draws inspiration from both [OpenXLA](https://github.com/openxla/xla) and [MLX](https://github.com/ml-explore/mlx), combining the best of both worlds:

**From OpenXLA:**
- **StableHLO as the IR** — A stable, portable intermediate representation for ML workloads
- **Multi-phase optimization pipeline** — Simplification → Canonicalization → Fusion → Layout → Scheduling
- **Pattern-based fusion** — Recognizing and fusing common patterns like attention, depth attention ([Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals)), layer norm, GELU
- **Cost-model driven decisions** — Using analytical cost models to guide fusion decisions

**From MLX:**
- **Lazy graph compilation** — Build the computation graph, compile once, execute many times
- **Unified memory architecture** — Zero-copy data sharing between CPU and GPU on Apple Silicon
- **Simplicity first** — Clean APIs that make common cases easy and complex cases possible
- **Single-device focus** — Optimized for one GPU rather than distributed execution

**MetalHLO's unique contribution:**
- **Three execution backends** — MPSGraph for broad compatibility, custom Metal kernels for peak performance, and heterogeneous GPU+ANE for parallel execution
- **Heterogeneous GPU+ANE execution** — Automatically partitions workloads across the GPU and Apple Neural Engine, the first StableHLO runtime to leverage the ANE (see [how it works](#how-gpuane-heterogeneous-execution-works) below)
- **Progressive optimization levels** — O0 to O3 matching compiler conventions
- **PJRT plugin for JAX** — Standard OpenXLA plugin interface enables `import jax; jax.numpy` on Apple GPUs
- **C API for portability** — Integrate with any language that can call C functions

### How GPU+ANE Heterogeneous Execution Works

Every Apple Silicon chip contains both a GPU and a Neural Engine (ANE) — two independent accelerators that share unified memory. Most ML frameworks only use the GPU. MetalHLO is the first StableHLO runtime to use both simultaneously.

```
                  StableHLO Program
                        │
                   ┌────▼────┐
                   │  Cost   │  Estimates GPU vs ANE execution time
                   │  Model  │  per operation based on op type, shape,
                   │         │  and data movement cost
                   └────┬────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
     ┌────────────────┐  ┌────────────────┐
     │   GPU Queue    │  │   ANE Queue    │
     │                │  │                │
     │  • Large GEMM  │  │  • Elementwise │
     │  • Convolution │  │  • Batch norm  │
     │  • Complex     │  │  • Simple      │
     │    reductions  │  │    reductions  │
     └────────┬───────┘  └────────┬───────┘
              │                   │
              │   Unified Memory  │  ← Zero-copy: both accelerators
              │   (shared)        │    read/write the same buffers
              └─────────┬─────────┘
                        ▼
                     Results
```

The key insight is that because Apple Silicon uses unified memory, there is no data transfer cost between GPU and ANE — both read and write the same physical memory. The cost model assigns each operation to whichever accelerator will complete it faster, and both execute their queues concurrently. This gives up to **3-4x speedup** on element-wise heavy workloads compared to MPSGraph, with the ANE handling simple operations while the GPU focuses on compute-intensive ones.

## Features

- **StableHLO Conformance** — 191 of 277 conformance tests pass (86 skipped for MPS/Metal limitations)
- **88% StableHLO Coverage** — 92 of 105 operations implemented
- **~99% Practical ML Coverage** — All operations needed for production ML workloads
- **Triple API** — Native Swift API + C API + PJRT plugin for JAX/XLA integration
- **Configurable Optimization** — O0 to O3 levels with algebraic simplification and operator fusion
- **Heterogeneous Execution** — GPU+ANE auto mode partitions operations across GPU and Neural Engine with a cost model
- **Full Training Support** — Forward and backward pass operations
- **Apple Silicon Optimized** — Leverages MPSGraph, custom Metal kernels, and the Apple Neural Engine

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

# Build the PJRT plugin for JAX integration
swift build -c release --product PJRTMetalHLO
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

### Swift — Heterogeneous GPU+ANE Execution

```swift
import MetalHLO

let client = try Client.create()

// Enable heterogeneous execution across GPU and Apple Neural Engine
let config = CompilationConfig(
    optimizationLevel: .O3,
    devicePolicy: .auto  // Cost model decides GPU vs ANE per operation
)
let executable = try client.compile(mlir, config: config)

// Execute — operations are automatically partitioned across GPU and ANE
let outputs = try executable.execute(inputs)
```

The cost model routes operations to the most efficient accelerator:
- **GPU:** Large matmuls, convolutions, reductions with complex access patterns
- **ANE:** Element-wise operations on large tensors, batch normalization, simple reductions

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
config.device_policy = MHLO_DEVICE_AUTO;  // Enable GPU+ANE heterogeneous execution
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
                    │  • Depth attention (Attention Residuals)│
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
| **Pattern Fusion** | Recognizes ML patterns (softmax, GELU, attention, depth attention) and replaces with optimized kernels |
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
| **Custom Calls** | fused_scaled_dot_product_attention, fused_depth_attention, fused_layer_norm, fused_rms_norm, fused_matmul_bias_activation, fused_softmax, fused_gelu, fused_rope |

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
    var devicePolicy: DevicePolicy            // .gpuOnly, .auto
    var enableCaching: Bool
    var generateDebugInfo: Bool

    // Presets
    static let `default`: CompilationConfig  // O2, GPU only
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

## PJRT Plugin (JAX Integration)

MetalHLO implements the [PJRT C API](https://github.com/openxla/xla/tree/main/xla/pjrt/c) (v0.90), the standard device plugin interface for the OpenXLA ecosystem. This enables JAX, TensorFlow, and PyTorch/XLA to execute StableHLO programs on Apple Silicon GPUs via MetalHLO.

### How It Works

The plugin is a dynamic library (`libPJRTMetalHLO.dylib`) that exports a `GetPjrtApi()` symbol returning a fully populated `PJRT_Api` vtable. JAX discovers and loads this plugin automatically via Python's entry-point mechanism.

### Implemented PJRT Functions

| Category | Functions |
|----------|-----------|
| **Error** | `Error_Destroy`, `Error_Message`, `Error_GetCode` |
| **Client** | `Client_Create`, `Client_Destroy`, `Client_PlatformName`, `Client_Devices`, `Client_AddressableDevices`, `Client_Compile`, `Client_BufferFromHostBuffer` |
| **Executable** | `LoadedExecutable_Execute`, `LoadedExecutable_Destroy`, `LoadedExecutable_GetExecutable`, `Executable_Name`, `Executable_NumOutputs`, `Executable_Serialize`, `Executable_Destroy`, `Executable_Fingerprint`, `Executable_OutputElementTypes`, `Executable_OutputDimensions`, `Executable_OutputMemoryKinds`, `Executable_DeserializeAndLoad` |
| **Buffer** | `Buffer_ToHostBuffer`, `Buffer_Destroy`, `Buffer_ElementType`, `Buffer_Dimensions`, `Buffer_OnDeviceSizeInBytes`, `Buffer_Device`, `Buffer_Memory`, `Buffer_ReadyEvent`, `Buffer_CopyToDevice` |
| **Device** | `Device_GetDescription`, `DeviceDescription_Id`, `DeviceDescription_ProcessIndex`, `DeviceDescription_Attributes` |
| **Memory** | `Device_AddressableMemories`, `Device_DefaultMemory`, `Memory_Id`, `Memory_Kind`, `Memory_Kind_Id` |
| **Event** | `Event_Await`, `Event_OnReady`, `Event_Destroy`, `Event_IsReady` |

### Building the Plugin

```bash
# Build the dynamic library
swift build -c release --product PJRTMetalHLO

# The dylib is at .build/release/libPJRTMetalHLO.dylib
```

### Using with JAX

Install the Python package that registers MetalHLO as a JAX backend:

```bash
# From the repository root
pip install -e python/
```

Then use JAX as usual — MetalHLO will be available as a backend:

```python
import jax
import jax.numpy as jnp

# Check available backends
print(jax.devices())  # Should include MetalHLO device

# Run computations on MetalHLO
x = jnp.array([1.0, 2.0, 3.0, 4.0])
y = jnp.array([5.0, 6.0, 7.0, 8.0])
result = x + y  # Executes on Apple GPU via MetalHLO
```

You can also set the `METALHLO_PLUGIN_PATH` environment variable to point to the dylib if it's not in the default build locations:

```bash
export METALHLO_PLUGIN_PATH=/path/to/libPJRTMetalHLO.dylib
```

### Plugin Capabilities

- **Platform name:** `metalhlo`
- **Memory model:** Unified memory (CPU/GPU shared, zero-copy transfers)
- **Compilation:** Uses O2 optimization by default (pattern fusion, CSE, algebraic simplification)
- **Serialization:** Full executable serialize/deserialize support for caching compiled programs
- **Buffer management:** Host-to-device and device-to-host transfers with proper event synchronization
- **Device attributes:** Reports Metal GPU family, recommended max working set size, and unified memory architecture

### Current Limitations

- Single-device execution only (no multi-GPU or distributed)
- Static shapes required (no dynamic shape inference at the PJRT level)
- Token and tuple operations are not supported
- Some JAX operations may require additional PJRT functions not yet implemented

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│   JAX / XLA               C/C++ Projects           Swift Projects       │
│        │                       │                         │              │
│        ▼                       ▼                         ▼              │
│   ┌──────────────┐    ┌─────────────┐           ┌───────────────────┐  │
│   │ PJRT Plugin  │    │ C API       │           │ Swift API         │  │
│   │ GetPjrtApi() │    │ mhlo_*      │           │ MetalHLO.Client   │  │
│   └──────┬───────┘    └──────┬──────┘           └─────────┬─────────┘  │
│          │                   │                            │             │
│          └───────────────────┴────────────────────────────┘             │
│                                    ▼                                    │
│               ┌─────────────────────────────────────────┐              │
│               │  MetalHLOCore                           │              │
│               │                                         │              │
│               │  ┌───────────┐   ┌───────────────────┐  │              │
│               │  │ Parser    │ → │ Optimizer         │  │              │
│               │  └───────────┘   │ (PassManager)     │  │              │
│               │                  └─────────┬─────────┘  │              │
│               │                            │            │              │
│               │       ┌────────────┴────────────────┐   │              │
│               │       ▼              ▼              ▼   │              │
│               │  ┌──────────┐  ┌──────────┐  ┌────────────┐           │
│               │  │ MPSGraph │  │ Metal    │  │ GPU+ANE    │           │
│               │  │ Backend  │  │ Kernel   │  │ Heterogen. │           │
│               │  │ (default)│  │ (O0-O3)  │  │ (auto)     │           │
│               │  └──────────┘  └──────────┘  └────────────┘           │
│               └─────────────────────────────────────────┘              │
│                          │            │            │                    │
│                          ▼            ▼            ▼                    │
│               ┌──────────────────────────────────────────┐             │
│               │  Apple Metal / MPSGraph / Neural Engine  │             │
│               └──────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────────────┘
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

### Quick Start

Run the core unit tests (fast, ~600 tests):

```bash
swift test --filter 'MetalHLOCoreTests'
```

Run the full test suite:

```bash
swift test
```

### Recommended: Serial Execution

For the most reliable test runs, use the `--no-parallel` flag. This prevents Metal/MPSGraph resource conflicts that can occur when multiple GPU operations run concurrently:

```bash
swift test --no-parallel
```

### Running Specific Tests

```bash
# By test suite name
swift test --filter "Binary"
swift test --filter "Reduction"
swift test --filter "CAPITests"
swift test --filter "AlgebraicSimplifier"

# PJRT plugin tests
swift test --filter 'PJRTPluginTests'

# Conformance tests
swift test --filter 'OfficialInterpretTests'
swift test --filter 'Optimization'

# Specific operation tests
swift test --filter 'testAdd'
swift test --filter 'scatter'
```

### Test Organization

| Target | Description | Test Count |
|--------|-------------|------------|
| `MetalHLOCoreTests` | Core compiler and optimizer tests | ~586 |
| `MetalHLOTests` | Integration and conformance tests | ~400+ |
| `PJRTMetalHLOTests` | PJRT plugin API and execution tests | 10 |

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
| PJRT Plugin | 10 |
| **StableHLO Conformance** | **191** |
| **Total** | **449** |

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
| `gather` | Improved | Works with all optimization levels (O0-O3); embedding lookup and index_vector_dim handling; batching dims supported when at leading positions |
| `scatter` | Improved | Supports add/max/min/mul computation modes via MPS; batching dims require leading positions |
| `convert` | Working | Type conversion between numeric types works with all optimization levels |
| `slice` | Working | Static slice extraction with starts/limits/strides |
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

## Benchmarks

MetalHLO includes a comprehensive benchmarking framework with multi-backend comparison across four execution paths.

### Multi-Backend Performance Comparison

All results measured on **Apple M1 (8 GB)**, macOS 15.6, release build, quick mode (3 warmup, 10 measurements).
Times are mean in milliseconds. **Bold** indicates the fastest backend for each benchmark.

#### Matrix Operations

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| MAT-DOT-001 | GEMM 128x128 | 0.30 | **0.17** | 0.24 | 0.25 | 1.72x (O2) |
| MAT-DOT-002 | GEMM 512x512 | 1.03 | **0.77** | 1.71 | 0.85 | 1.34x (O2) |
| MAT-DOT-003 | GEMM 1024x1024 | **4.17** | 5.10 | 5.32 | 4.53 | MPSGraph best |
| MAT-DOT-004 | GEMM 2048x2048 | **19.55** | 23.00 | 22.63 | 22.17 | MPSGraph best |
| MAT-DOT-005 | GEMM 4096x4096 | **122.7** | 226.1 | 227.6 | 234.4 | MPSGraph best |
| MAT-DOT-006 | Transformer 32x4096x768 | 2.36 | 1.22 | 1.17 | **1.10** | 2.14x (ANE) |
| MAT-DOT-007 | MLP 128x768x3072 | 3.11 | 1.90 | 2.06 | **1.63** | 1.91x (ANE) |
| MAT-DOT-008 | Matvec 1x4096x4096 | **3.93** | 10.91 | 11.68 | 10.94 | MPSGraph best |
| MAT-BATCH-001 | Batched 8x512x512 | 6.10 | 5.41 | **4.69** | 5.58 | 1.30x (O3) |
| MAT-BATCH-002 | Batched 4x1024x1024 | 4.59 | **2.85** | 3.03 | 2.99 | 1.61x (O2) |
| MAT-BATCH-003 | Attention heads | 1.61 | 1.12 | **0.67** | 1.28 | 2.42x (O3) |
| MAT-BATCH-004 | Multi-head attention | 0.81 | 0.38 | 0.38 | **0.36** | 2.29x (ANE) |
| MAT-TR-001 | Transpose 1024x1024 | 1.45 | 0.84 | **0.50** | 1.27 | 2.87x (O3) |
| MAT-TR-002 | Transpose 3D 32x128x64 | 0.49 | 0.65 | **0.25** | 0.56 | 1.99x (O3) |
| MAT-RSH-001 | Reshape flatten 1024x1024 | 1.34 | 0.53 | **0.48** | 0.57 | 2.80x (O3) |
| MAT-RSH-002 | Reshape batch 32x64x128 | 0.48 | **0.41** | 0.46 | 0.42 | 1.16x (O2) |

**Takeaway:** MPSGraph wins on large GEMMs (≥1024x1024) and matvec where Apple's tuned kernels dominate. Metal O3 wins on batched operations, transpose, and reshape (1.3-2.9x faster). O2 wins on small-to-mid GEMMs. ANE excels on transformer-shaped and MLP matmuls.

#### Element-wise Arithmetic

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| ARITH-B-001 | Add 1024x1024 | 2.52 | 0.96 | **0.75** | 0.75 | 3.35x (ANE) |
| ARITH-B-002 | Add 4096x4096 | 12.20 | 10.30 | 16.43 | **8.75** | 1.40x (ANE) |
| ARITH-B-003 | Add 8192x8192 | **57.7** | 169.1 | 138.5 | 148.1 | MPSGraph best |
| ARITH-B-004 | Mul 1024x1024 | 2.66 | **0.79** | 0.83 | 0.88 | 3.38x (O2) |
| ARITH-B-005 | Mul 4096x4096 | 11.91 | **10.12** | 10.43 | 13.97 | 1.18x (O2) |
| ARITH-B-006 | Div 1024x1024 | 2.16 | 0.97 | **0.80** | 1.15 | 2.69x (O3) |
| ARITH-B-007 | Pow 1024x1024 | 2.52 | **0.97** | 0.98 | 1.03 | 2.59x (O2) |
| ARITH-B-008 | Max 4096x4096 | 11.85 | 12.46 | 9.42 | **8.32** | 1.42x (ANE) |
| ARITH-U-001 | Exp 1024x1024 | 1.23 | 0.61 | **0.57** | 0.85 | 2.14x (O3) |
| ARITH-U-002 | Log 4096x4096 | 8.07 | 5.01 | **4.90** | 5.08 | 1.65x (O3) |
| ARITH-U-003 | Tanh 1024x1024 | 1.54 | **0.52** | 0.52 | 0.81 | 2.96x (O3) |
| ARITH-U-004 | Sqrt 4096x4096 | 8.02 | **4.61** | 4.79 | 5.30 | 1.74x (O2) |
| ARITH-U-005 | Rsqrt 4096x4096 | 8.71 | 5.65 | 5.74 | **5.57** | 1.56x (ANE) |
| ARITH-U-006 | Sigmoid 1024x1024 | 1.34 | 0.70 | **0.70** | 0.79 | 1.93x (O3) |
| ARITH-BC-001 | Add row broadcast | 1.35 | 1.34 | **0.93** | 1.10 | 1.45x (O3) |
| ARITH-BC-002 | Add scalar broadcast | 1.29 | 1.00 | 0.99 | **0.81** | 1.58x (ANE) |
| ARITH-BC-003 | Mul last-dim broadcast | 0.48 | **0.39** | 0.44 | 0.54 | 1.23x (O2) |

**Takeaway:** Metal backends are **1.2-3.4x faster** than MPSGraph on element-wise operations. The advantage is largest on 1024x1024 tensors (2.1-3.4x); on 4096x4096 tensors MPSGraph has improved significantly (macOS 15.6), narrowing the gap to 1.2-1.7x. O2, O3, and ANE all trade wins.

#### Reduction Operations

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| RED-001 | Global sum 1024x1024 | 0.91 | 0.96 | 0.89 | **0.86** | 1.06x (ANE) |
| RED-002 | Row-wise sum 1024x1024 | 1.03 | **0.87** | 1.09 | 0.95 | 1.18x (O2) |
| RED-003 | Column-wise sum 1024x1024 | 0.90 | **0.82** | 0.83 | 0.84 | 1.10x (O2) |
| RED-004 | Row-wise max 4096x4096 | **4.21** | 6.25 | 5.91 | 6.17 | MPSGraph best |
| RED-005 | LayerNorm reduction 32x128x768 | 2.72 | 1.29 | **0.95** | 1.09 | 2.86x (O3) |
| RED-006 | Attention reduction 32x12x512x512 | **22.49** | 47.37 | 46.81 | 37.49 | MPSGraph best |

**Takeaway:** Metal backends win on 1024x1024 reductions (1.1-1.2x). O3 excels on LayerNorm reduction (2.9x). On larger reductions (4096x4096 and attention-shaped), MPSGraph wins thanks to macOS 15.6 improvements.

#### Convolution

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| CONV-001 | ResNet first layer | **1.92** | 1.91 | 2.24 | 2.78 | ~1.00x |
| CONV-002 | ResNet stage2 3x3 | **1.16** | 3.36 | 3.09 | 1.16 | MPSGraph best |
| CONV-003 | ResNet stage3 3x3 | **1.06** | 3.64 | 3.15 | 1.36 | MPSGraph best |
| CONV-004 | ResNet stage4 3x3 | **1.73** | 3.45 | 2.77 | 2.01 | MPSGraph best |
| CONV-005 | Batched conv | **11.79** | 34.94 | 34.42 | 20.22 | MPSGraph best |
| CONV-006 | 1x1 pointwise | 0.83 | 0.71 | **0.68** | 1.75 | 1.22x (O3) |
| CONV-007 | Depthwise-like | 1.61 | 4.88 | 5.24 | **1.54** | 1.05x (ANE) |

**Takeaway:** MPSGraph dominates convolutions thanks to Apple's highly optimized `MPSCNNConvolution` kernels. Metal O3 wins on 1x1 pointwise convolutions. ANE occasionally beats MPSGraph on depthwise patterns.

#### Normalization

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| NORM-BN-001 | ResNet BN | 0.42 | **0.24** | 0.28 | 0.39 | 1.76x (O2) |
| NORM-BN-002 | Batched ResNet BN | 8.06 | 3.83 | **2.43** | FAIL | 3.32x (O3) |
| NORM-BN-003 | Mid-layer BN | 0.35 | **0.18** | 0.21 | 0.19 | 1.90x (O2) |
| NORM-BN-004 | Late-layer BN | 0.29 | **0.17** | 0.38 | 0.19 | 1.70x (O2) |
| NORM-LN-001 | BERT-base LayerNorm | 0.57 | **0.52** | 0.56 | 0.59 | 1.09x (O2) |
| NORM-LN-002 | BERT-base batched LN | **4.42** | 6.17 | 6.87 | 6.46 | MPSGraph best |
| NORM-LN-003 | BERT-large single LN | **1.36** | 1.77 | 2.17 | 1.63 | MPSGraph best |
| NORM-LN-004 | Long sequence LN | **15.67** | 20.90 | 21.10 | 23.02 | MPSGraph best |

**Takeaway:** Metal O2 excels at batch normalization (1.7-1.9x faster). O3 wins on large batched BN (3.3x). MPSGraph wins on layer normalization, where its native implementation is well-optimized.

#### Transformer Components

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| XFMR-INF-001 | Self-attention seq=128 | 2.76 | 2.51 | **2.19** | 2.41 | 1.26x (O3) |
| XFMR-INF-002 | Self-attention seq=512 | **7.61** | 7.85 | 8.28 | 7.82 | MPSGraph best |
| XFMR-INF-003 | Self-attention BS=8 seq=128 | **8.80** | 9.45 | 9.43 | 9.22 | MPSGraph best |
| XFMR-INF-004 | Transformer FFN BS=8 | **11.27** | 18.24 | 16.71 | 16.02 | MPSGraph best |
| XFMR-INF-005 | Softmax 8x12x128x128 | 2.53 | 2.34 | 2.29 | **2.29** | 1.11x (ANE) |
| XFMR-INF-006 | Encoder block BS=1 seq=128 | 7.71 | **7.46** | 8.48 | 7.84 | 1.03x (O2) |

**Takeaway:** MPSGraph wins on 3 of 6 transformer workloads (larger sequence lengths, FFN, batched attention). O3 excels on self-attention (1.3x). ANE is competitive on softmax.

#### MLP Inference

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| MLP-INF-001 | 784->256->10 BS=1 | 0.64 | **0.45** | 0.45 | 0.80 | 1.43x (O2) |
| MLP-INF-002 | 784->256->10 BS=32 | **0.41** | 0.43 | 0.44 | 0.60 | MPSGraph best |
| MLP-INF-003 | 784->256->10 BS=128 | 0.60 | 0.72 | 0.47 | **0.45** | 1.35x (ANE) |
| MLP-INF-004 | Deep MLP 4-layer BS=32 | 1.05 | 0.91 | 0.80 | **0.75** | 1.40x (ANE) |
| MLP-INF-005 | FFN 768->3072->768 BS=32 | 4.35 | 1.77 | **1.58** | 1.84 | 2.75x (O3) |

**Takeaway:** Metal backends are 1.4-2.8x faster than MPSGraph on MLPs. O3 wins the large FFN (2.8x). ANE excels on deep and batched MLPs. MPSGraph is fastest only on small batch sizes.

#### Training

| Benchmark | Description | MPSGraph | Metal O2 | Metal O3 | GPU+ANE | Best vs MPSGraph |
|-----------|-------------|----------|----------|----------|---------|-------------------|
| TRAIN-001 | MLP fwd+bwd BS=32 | **0.69** | 0.91 | 0.90 | 1.00 | ~1.00x |
| TRAIN-003 | Attention fwd+bwd BS=8 | **7.25** | 10.29 | 10.82 | 11.12 | MPSGraph best |

**Takeaway:** MPSGraph leads on training workloads where its backward-pass graph optimization provides an advantage.

### Backend Win Summary

Overall wins across all 67 passing benchmarks:

| Backend | Wins | Best For |
|---------|------|----------|
| **MPSGraph** | 20 | Large GEMMs, convolutions, layer norm, large reductions, training |
| **Metal O2** | 18 | Batch norm, small matmuls, element-wise ops |
| **Metal O3** | 16 | Transpose, reshape, batched ops, softmax, FFN fusion |
| **GPU+ANE** | 13 | Element-wise, MLP inference, transformer-shaped matmuls |

### When to Use Each Backend

| Use Case | Recommended Backend | Why |
|----------|-------------------|-----|
| Large matrix multiply (≥1024x1024) | MPSGraph (default) | Apple's tuned `MPSMatrixMultiplication` |
| Convolution-heavy models (CNNs) | MPSGraph (default) | `MPSCNNConvolution` is highly optimized |
| Training (forward + backward) | MPSGraph (default) | Graph-level backward pass optimization |
| Element-wise heavy workloads | Metal O2/O3 | 1.2-3.4x faster than MPSGraph |
| Batch normalization | Metal O2 | 1.7-1.9x faster custom kernels |
| Transpose / reshape / batched ops | Metal O3 | Pattern fusion + scheduling (2-3x faster) |
| MLP / FFN inference | Metal O3 or ANE | O3 wins large FFN (2.8x), ANE wins deep MLPs |
| Debugging/development | MPSGraph (default) | Broadest compatibility, no compilation |

### Benchmark Categories

| Category | Benchmarks | Description |
|----------|------------|-------------|
| **Matrix Operations** | MAT-DOT-001 to MAT-DOT-008, MAT-BATCH-001 to MAT-BATCH-004, MAT-TR-*, MAT-RSH-* | GEMM, batched GEMM, transpose, reshape |
| **Arithmetic** | ARITH-B-001 to ARITH-B-008, ARITH-BC-*, ARITH-U-* | Binary, broadcast, and unary operations |
| **Reduction** | RED-001 to RED-009 | Sum, max, mean, pooling operations |
| **Convolution** | CONV-001 to CONV-007 | Standard conv2d patterns (ResNet, VGG) |
| **Normalization** | NORM-BN-*, NORM-LN-* | Batch norm, layer norm |
| **Control Flow** | CF-001 to CF-005 | While loops, conditionals |
| **Indexing** | IDX-001 to IDX-007 | Slice, gather, scatter, pad |
| **Model Inference** | MLP-INF-*, CNN-INF-*, XFMR-INF-* | MLP, CNN, Transformer components |
| **Training** | TRAIN-001 to TRAIN-003 | Forward + backward pass benchmarks |
| **Compiler Analysis** | COMP-001 to COMP-005 | Compilation time for various program sizes |
| **Fusion Analysis** | FUSION-001 to FUSION-004 | Fused vs naive execution comparison |
| **Memory** | MEM-001 to MEM-003 | Peak allocation, buffer reuse |
| **Power Efficiency** | PWR-001 to PWR-003 | Throughput per watt estimates |

### Running Benchmarks

```bash
# Build in release mode for accurate measurements
swift build -c release --product benchmark-runner

# Multi-backend comparison (recommended)
.build/release/benchmark-runner --compare -q -c matrix        # Matrix operations
.build/release/benchmark-runner --compare -q -c arithmetic    # Element-wise ops
.build/release/benchmark-runner --compare -q -c model_transformer  # Transformer
.build/release/benchmark-runner --compare -q                  # All categories

# Single-backend benchmarks
.build/release/benchmark-runner --category matrix
.build/release/benchmark-runner --all

# MLX comparison (requires MLX)
swift build -c release --product mlx-comparison
.build/release/mlx-comparison --quick
.build/release/mlx-comparison --category matrix
```

### Benchmark Framework Features

The benchmark framework provides:

- **Timing Statistics**: Mean, std dev, min, max, p95, p99
- **GPU Synchronization**: Accurate timing with Metal command buffer completion
- **Warmup Support**: Configurable warmup iterations to avoid cold-start effects
- **Reproducibility**: Seeded random data generators
- **Multiple Output Formats**: Console, JSON, CSV

```swift
// Running benchmarks programmatically
import MetalHLOBenchmarks

let config = BenchmarkConfig(warmupIterations: 10, measurementIterations: 50)
let runner = try BenchmarkRunner(config: config)

// Run all matrix benchmarks
let results = try runner.run(OperationBenchmarks.matrixBenchmarks())

// Access timing statistics
for result in results {
    print("\(result.id): \(result.timing.mean * 1000)ms ± \(result.timing.stdDev * 1000)ms")
}
```

### GPU Utilization Metrics

The framework includes utilities for measuring GPU efficiency:

```swift
import MetalHLOBenchmarks

// Detect hardware and calculate utilization
let calculator = GPUMetricsCalculator()

// After running a 1024x1024 matmul in 2.5ms
let metrics = calculator.calculateMatMulMetrics(
    m: 1024, n: 1024, k: 1024,
    executionTimeSeconds: 0.0025
)

print(metrics.formatted())
// Hardware: Apple M1
// Compute: 0.86 / 2.60 TFLOPS (33.1% utilization)
// Memory BW: 6.7 / 68.25 GB/s (9.8% utilization)
```

### Fusion Effectiveness Analysis

Measure the benefit of operation fusion:

```swift
import MetalHLOBenchmarks

let runner = FusionAnalysisRunner()
let results = try runner.runAll()

for result in results {
    print(result.formatted())
    // FUSION-001: Fused=1.23ms, Naive=3.45ms, Speedup=2.80x (expected: 2-3x)
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## References

- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [OpenXLA/XLA Compiler](https://github.com/openxla/xla)
- [MLX by Apple ML Research](https://github.com/ml-explore/mlx)
- [MPSGraph Documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [PJRT C API Specification](https://github.com/openxla/xla/tree/main/xla/pjrt/c)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)

---

**MetalHLO** — Bringing StableHLO to Apple Silicon
