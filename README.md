# MetalHLO

**StableHLO Execution on Apple Metal**

MetalHLO is a standalone library that compiles and executes [StableHLO](https://github.com/openxla/stablehlo) MLIR programs on Apple Silicon GPUs using MPSGraph. It provides both Swift and C APIs, enabling integration with any project that emits StableHLO IR.

## Features

- **88% StableHLO Coverage** — 92 of 105 operations implemented
- **~99% Practical ML Coverage** — All operations needed for production ML workloads
- **Dual API** — Native Swift API + C API for C/C++ projects
- **PJRT-like Interface** — Familiar Client/Executable/Buffer pattern
- **Full Training Support** — Forward and backward pass operations
- **Apple Silicon Optimized** — Leverages MPSGraph for optimized kernels

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

### Swift

```swift
import MetalHLO

// Create client
let client = try Client.create()
print("Using device: \(client.deviceName)")

// Compile StableHLO MLIR
let mlir = """
module @add {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
"""
let executable = try client.compile(mlir)

// Create input buffers
let a = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [2, 3], elementType: .float32)
let b = try client.createBuffer([0.1, 0.2, 0.3, 0.4, 0.5, 0.6] as [Float], shape: [2, 3], elementType: .float32)

// Execute
let outputs = try executable.execute([a, b])

// Get results
let result = try outputs[0].toFloatArray()
print("Result: \(result)")  // [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
```

### C

```c
#include <metalhlo.h>
#include <stdio.h>

int main() {
    MHLOStatus status;
    MHLOClientRef client;

    // Create client
    status = mhlo_client_create(&client);
    if (status.code != MHLO_OK) {
        fprintf(stderr, "Failed: %s\n", status.message);
        return 1;
    }

    // Compile MLIR
    const char* mlir =
        "module @add {\n"
        "  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>) {\n"
        "    %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>\n"
        "    return %0 : tensor<2x3xf32>\n"
        "  }\n"
        "}\n";

    MHLOExecutableRef executable;
    mhlo_compile(client, mlir, &executable);

    // Create buffers and execute...

    mhlo_executable_destroy(executable);
    mhlo_client_destroy(client);
    return 0;
}
```

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

### Operations Requiring MPS Kernel Bridging (3 ops)

| Operation | Notes |
|-----------|-------|
| `triangular_solve` | Requires MPSMatrixSolveTriangular |
| `cholesky` | Requires MPSMatrixDecompositionCholesky |
| `map` | Requires JIT compilation |

### Excluded by Design (14 ops)

Multi-device operations (all_gather, all_reduce, etc.), communication operations (infeed, outfeed, etc.), and tuple operations are excluded per the single-device, static-shape focus.

## API Reference

### Client

```swift
public final class Client: @unchecked Sendable {
    /// Create a client for the default Metal device
    public static func create() throws -> Client

    /// Compile StableHLO MLIR to an executable
    public func compile(_ mlir: String) throws -> Executable

    /// Create a buffer from host data
    public func createBuffer<T: Numeric>(
        _ data: [T],
        shape: [Int],
        elementType: ElementType
    ) throws -> Buffer
}
```

### Executable

```swift
public final class Executable: @unchecked Sendable {
    /// Number of inputs/outputs
    public var inputCount: Int { get }
    public var outputCount: Int { get }

    /// Input/output tensor types
    public var inputTypes: [TensorType] { get }
    public var outputTypes: [TensorType] { get }

    /// Execute the program
    public func execute(_ inputs: [Buffer]) throws -> [Buffer]

    /// Execute with timing information
    public func executeWithTiming(_ inputs: [Buffer]) throws -> ([Buffer], ExecutionTiming)
}
```

### Buffer

```swift
public final class Buffer: @unchecked Sendable {
    public var shape: [Int] { get }
    public var count: Int { get }
    public var elementType: ElementType { get }

    public func toFloatArray() throws -> [Float]
    public func toInt32Array() throws -> [Int32]
    public func toData() throws -> Data
}
```

### Element Types

```swift
public enum ElementType: String, CaseIterable, Sendable {
    case float16 = "f16"
    case float32 = "f32"
    case float64 = "f64"
    case bfloat16 = "bf16"
    case int1 = "i1"      // Boolean
    case int8 = "i8"
    case int16 = "i16"
    case int32 = "i32"
    case int64 = "i64"
    case uint8 = "ui8"
    case uint16 = "ui16"
    case uint32 = "ui32"
    case uint64 = "ui64"
}
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
│          │  │ MLIRParser│ → │ MPSGraphCompiler  │  │            │
│          │  └───────────┘   └───────────────────┘  │            │
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
```

### Convolution + Pooling

```swift
let mlir = """
module @cnn {
  func.func @main(%input: tensor<1x28x28x1xf32>, %kernel: tensor<3x3x1x32xf32>) -> (tensor<1x13x13x32xf32>) {
    // Conv2D
    %0 = stablehlo.convolution %input, %kernel
         window_strides = [1, 1], feature_group_count = 1
         : (tensor<1x28x28x1xf32>, tensor<3x3x1x32xf32>) -> tensor<1x26x26x32xf32>

    // ReLU
    %zero = stablehlo.constant dense<0.0> : tensor<1x26x26x32xf32>
    %1 = stablehlo.maximum %0, %zero : tensor<1x26x26x32xf32>

    // MaxPool 2x2
    %init = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.reduce_window %1, %init
         window_dimensions = [1, 2, 2, 1], window_strides = [1, 2, 2, 1],
         applies stablehlo.maximum
         : (tensor<1x26x26x32xf32>, tensor<f32>) -> tensor<1x13x13x32xf32>
    return %2 : tensor<1x13x13x32xf32>
  }
}
"""
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
swift test --filter "Convolution"
```

### Test Coverage

| Test Suite | Tests |
|------------|-------|
| Lexer Tests | 15 |
| Parser Tests | 10 |
| Binary Operations | 7 |
| Unary Operations | 12 |
| Matrix Operations | 10 |
| Reduction Operations | 6 |
| Control Flow | 8 |
| Activation Functions | 35 |
| CNN Operations | 10 |
| **Total** | **174** |

## Limitations

1. **Static Shapes Only** — Dynamic shapes are not supported
2. **Single Device** — No multi-device or distributed execution
3. **Apple Silicon Only** — Intel Macs with AMD GPUs are not supported
4. **macOS Only** — iOS/iPadOS support is a future consideration

### Operations with Limited Implementation

| Operation | Limitation |
|-----------|------------|
| `reduce_precision` | Returns identity (MPSGraph lacks bitcast) |
| `triangular_solve` | Requires MPS kernel bridging |
| `cholesky` | Requires MPS kernel bridging |
| `map` | Requires JIT compilation |

## Performance

MetalHLO executes entirely on GPU using MPSGraph's native compilation:

- Single CPU encode call (no graph breaks)
- No CPU-GPU synchronization overhead
- Kernels launch without pipeline bubbles
- Native MPSGraph control flow for loops and conditionals

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## References

- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [MPSGraph Documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)


---

**MetalHLO** — Bringing StableHLO to Apple Silicon
