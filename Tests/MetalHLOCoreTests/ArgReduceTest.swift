// ArgReduceTest.swift
// Tests argmax/argmin on the fast CodeGenerator path. jnp.argmax/argmin lower
// to a 2-input, 2-result stablehlo.reduce (input values + an iota index) whose
// reducer keeps the larger/smaller value and the smaller index on ties. The
// parser splits that op into a value `.reduce` (%r#0) and a `.reduceArg` index
// op (%r#1); we only return the index (#1), exactly as jnp.argmax does.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("ArgReduce Tests", .serialized)
struct ArgReduceTests {

    @Test("argmax over axis 1 of a 2D array returns the per-row argmax index")
    func testArgMax2D() async throws {
        // This is the exact body JAX 0.10 emits for jnp.argmax(x, axis=1),
        // inlined into @main. reducer's first compare is GT -> argmax.
        let mlir = """
        module @argmax_test {
          func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4xi32> {
            %0 = stablehlo.iota dim = 1 : tensor<4x8xi32>
            %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %c = stablehlo.constant dense<0> : tensor<i32>
            %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<4x8xf32>, tensor<4x8xi32>, tensor<f32>, tensor<i32>) -> (tensor<4xf32>, tensor<4xi32>)
             reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
              %2 = stablehlo.compare GT, %arg1, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
              %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
              %4 = stablehlo.or %2, %3 : tensor<i1>
              %5 = stablehlo.compare EQ, %arg1, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
              %6 = stablehlo.compare LT, %arg2, %arg4, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
              %7 = stablehlo.and %5, %6 : tensor<i1>
              %8 = stablehlo.or %4, %7 : tensor<i1>
              %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32>
              %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
              stablehlo.return %9, %10 : tensor<f32>, tensor<i32>
            }
            return %1#1 : tensor<4xi32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2
        let executable = try client.compile(mlir, config: config)

        // Per-row argmax (ties -> smallest index):
        // row0 [1,3,2,0,9,5,4,6] -> 4
        // row1 [7,7,0,1,2,3,4,5] -> 0 (tie at index 0,1; smallest wins)
        // row2 [-1,-2,-9,0,-3,-4,-5,-6] -> 3
        // row3 [8,1,2,3,4,5,6,7] -> 0
        let input: [Float] = [
            1, 3, 2, 0, 9, 5, 4, 6,
            7, 7, 0, 1, 2, 3, 4, 5,
            -1, -2, -9, 0, -3, -4, -5, -6,
            8, 1, 2, 3, 4, 5, 6, 7,
        ]
        let inBuffer = try client.createBuffer(input, shape: [4, 8], elementType: .float32)

        let outputs = try executable.execute([inBuffer])
        let indices = try outputs[0].toInt32Array()

        let expected: [Int32] = [4, 0, 3, 0]
        print("argmax 2D indices: \(indices)")
        #expect(indices.count == 4)
        for i in 0..<4 {
            #expect(indices[i] == expected[i], "index[\(i)] = \(indices[i]), expected \(expected[i])")
        }
    }

    @Test("argmin over axis 1 of a 2D array returns the per-row argmin index")
    func testArgMin2D() async throws {
        // jnp.argmin(x, axis=1): reducer's first compare is LT -> argmin.
        let mlir = """
        module @argmin_test {
          func.func @main(%arg0: tensor<3x4xf32>) -> tensor<3xi32> {
            %0 = stablehlo.iota dim = 1 : tensor<3x4xi32>
            %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
            %c = stablehlo.constant dense<0> : tensor<i32>
            %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<3x4xf32>, tensor<3x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<3xf32>, tensor<3xi32>)
             reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
              %2 = stablehlo.compare LT, %arg1, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
              %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
              %4 = stablehlo.or %2, %3 : tensor<i1>
              %5 = stablehlo.compare EQ, %arg1, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
              %6 = stablehlo.compare LT, %arg2, %arg4, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
              %7 = stablehlo.and %5, %6 : tensor<i1>
              %8 = stablehlo.or %4, %7 : tensor<i1>
              %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32>
              %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
              stablehlo.return %9, %10 : tensor<f32>, tensor<i32>
            }
            return %1#1 : tensor<3xi32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2
        let executable = try client.compile(mlir, config: config)

        // row0 [5,1,4,1] -> argmin 1 (tie at 1,3; smallest)
        // row1 [-2,0,3,1] -> 0
        // row2 [9,8,7,6] -> 3
        let input: [Float] = [
            5, 1, 4, 1,
            -2, 0, 3, 1,
            9, 8, 7, 6,
        ]
        let inBuffer = try client.createBuffer(input, shape: [3, 4], elementType: .float32)

        let outputs = try executable.execute([inBuffer])
        let indices = try outputs[0].toInt32Array()

        let expected: [Int32] = [1, 0, 3]
        print("argmin 2D indices: \(indices)")
        #expect(indices.count == 3)
        for i in 0..<3 {
            #expect(indices[i] == expected[i], "index[\(i)] = \(indices[i]), expected \(expected[i])")
        }
    }

    @Test("argmax over a middle axis (innerSize > 1) strides the reduce axis correctly")
    func testArgMax3DMiddleAxis() async throws {
        // jnp.argmax(x, axis=1) on (2,4,5) -> (2,5); reduce axis 1, innerSize=5.
        let mlir = """
        module @argmax_mid {
          func.func @main(%arg0: tensor<2x4x5xf32>) -> tensor<2x5xi32> {
            %0 = stablehlo.iota dim = 1 : tensor<2x4x5xi32>
            %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %c = stablehlo.constant dense<0> : tensor<i32>
            %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [1] : (tensor<2x4x5xf32>, tensor<2x4x5xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x5xf32>, tensor<2x5xi32>)
             reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i32>, %arg4: tensor<i32>)  {
              %2 = stablehlo.compare GT, %arg1, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
              %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
              %4 = stablehlo.or %2, %3 : tensor<i1>
              %5 = stablehlo.compare EQ, %arg1, %arg3, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
              %6 = stablehlo.compare LT, %arg2, %arg4, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
              %7 = stablehlo.and %5, %6 : tensor<i1>
              %8 = stablehlo.or %4, %7 : tensor<i1>
              %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32>
              %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<i32>
              stablehlo.return %9, %10 : tensor<f32>, tensor<i32>
            }
            return %1#1 : tensor<2x5xi32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2
        let executable = try client.compile(mlir, config: config)

        // Input (2,4,5): for each (b, col) find which of the 4 rows is max.
        // Reference computed below in Swift to avoid hand-arithmetic errors.
        var input = [Float](repeating: 0, count: 2 * 4 * 5)
        var rng: UInt64 = 0x1234_5678
        func next() -> Float {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int32(truncatingIfNeeded: rng >> 33)) / Float(1 << 20)
        }
        for i in 0..<input.count { input[i] = next() }

        // Reference argmax over axis 1: index into [b][row][col], stride row = 5.
        var expected = [Int32](repeating: 0, count: 2 * 5)
        for b in 0..<2 {
            for col in 0..<5 {
                var best = -Float.infinity
                var bestIdx: Int32 = 0
                for row in 0..<4 {
                    let v = input[b * 20 + row * 5 + col]
                    if v > best { best = v; bestIdx = Int32(row) }
                }
                expected[b * 5 + col] = bestIdx
            }
        }

        let inBuffer = try client.createBuffer(input, shape: [2, 4, 5], elementType: .float32)
        let outputs = try executable.execute([inBuffer])
        let indices = try outputs[0].toInt32Array()

        print("argmax 3D mid-axis indices: \(indices) expected: \(expected)")
        #expect(indices.count == 10)
        for i in 0..<10 {
            #expect(indices[i] == expected[i], "index[\(i)] = \(indices[i]), expected \(expected[i])")
        }
    }
}
