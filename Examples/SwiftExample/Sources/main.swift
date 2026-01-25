// main.swift
// SwiftExample
//
// Example demonstrating MetalHLO usage from Swift.

import MetalHLO

print("MetalHLO Swift Example")
print("======================")
print()

do {
    // Create client
    let client = try Client.create()
    print("Using device: \(client.deviceName)")
    print()

    // Example 1: Simple Addition
    print("Example 1: Element-wise Addition")
    print("---------------------------------")

    let addMLIR = """
    module @add {
      func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
        %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
        return %0 : tensor<2x3xf32>
      }
    }
    """

    let addExecutable = try client.compile(addMLIR)
    print("Compiled 'add' module")
    print("  Inputs: \(addExecutable.inputCount)")
    print("  Outputs: \(addExecutable.outputCount)")

    let a: [Float] = [1, 2, 3, 4, 5, 6]
    let b: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    let bufferA = try client.createBuffer(a, shape: [2, 3], elementType: .float32)
    let bufferB = try client.createBuffer(b, shape: [2, 3], elementType: .float32)

    let addOutputs = try addExecutable.execute([bufferA, bufferB])
    let addResult = try addOutputs[0].toFloatArray()

    print("  Input A: \(a)")
    print("  Input B: \(b)")
    print("  Result:  \(addResult)")
    print()

    // Example 2: Chained Operations
    print("Example 2: Chained Operations (add then multiply)")
    print("-------------------------------------------------")

    let chainMLIR = """
    module @chain {
      func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
        %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
        %1 = stablehlo.multiply %0, %arg0 : tensor<4xf32>
        return %1 : tensor<4xf32>
      }
    }
    """

    let chainExecutable = try client.compile(chainMLIR)

    let x: [Float] = [1, 2, 3, 4]
    let y: [Float] = [1, 1, 1, 1]

    let bufferX = try client.createBuffer(x, shape: [4], elementType: .float32)
    let bufferY = try client.createBuffer(y, shape: [4], elementType: .float32)

    let chainOutputs = try chainExecutable.execute([bufferX, bufferY])
    let chainResult = try chainOutputs[0].toFloatArray()

    print("  Formula: (x + y) * x")
    print("  Input X: \(x)")
    print("  Input Y: \(y)")
    print("  Result:  \(chainResult)")
    print()

    // Example 3: Execution with Timing
    print("Example 3: Execution with Timing")
    print("---------------------------------")

    let size = 1000
    let largeData = [Float](repeating: 1.0, count: size * size)
    let largeBuffer1 = try client.createBuffer(largeData, shape: [size, size], elementType: .float32)
    let largeBuffer2 = try client.createBuffer(largeData, shape: [size, size], elementType: .float32)

    let largeMLIR = """
    module @large {
      func.func @main(%arg0: tensor<\(size)x\(size)xf32>, %arg1: tensor<\(size)x\(size)xf32>) -> (tensor<\(size)x\(size)xf32>) {
        %0 = stablehlo.add %arg0, %arg1 : tensor<\(size)x\(size)xf32>
        return %0 : tensor<\(size)x\(size)xf32>
      }
    }
    """

    let largeExecutable = try client.compile(largeMLIR)
    let (_, timing) = try largeExecutable.executeWithTiming([largeBuffer1, largeBuffer2])

    print("  Tensor size: \(size)x\(size) (\(size * size) elements)")
    print("  Encode time: \(String(format: "%.3f", timing.encodeTime * 1000)) ms")
    print("  GPU time:    \(String(format: "%.3f", timing.gpuTime * 1000)) ms")
    print("  Total time:  \(String(format: "%.3f", timing.totalTime * 1000)) ms")
    print()

    print("All examples completed successfully!")

} catch {
    print("Error: \(error)")
}
