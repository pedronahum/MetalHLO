// TopKTest.swift
// Tests stablehlo.top_k (jax.lax.top_k) lowering on the fast CodeGenerator
// path. The bytecode preprocessor collapses chlo.top_k's composite to
// `stablehlo.top_k %x {k = K} : (intype) -> (vtype, itype)`, which the parser
// splits into a values op (%r#0) and an indices op (%r#1). top_k keeps the k
// largest elements of the last axis in descending order plus their positions.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("TopK Tests", .serialized)
struct TopKTests {

    @Test("top_k k=3 over a 1D array returns values and indices")
    func testTopK1D() async throws {
        let mlir = """
        module @topk_test {
          func.func @main(%arg0: tensor<8xf32>) -> (tensor<3xf32>, tensor<3xi32>) {
            %0:2 = stablehlo.top_k %arg0 {k = 3} : (tensor<8xf32>) -> (tensor<3xf32>, tensor<3xi32>)
            return %0#0, %0#1 : tensor<3xf32>, tensor<3xi32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2
        let executable = try client.compile(mlir, config: config)

        let input: [Float] = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        let inBuffer = try client.createBuffer(input, shape: [8], elementType: .float32)

        let outputs = try executable.execute([inBuffer])
        let values = try outputs[0].toFloatArray()
        let indices = try outputs[1].toInt32Array()

        // jax.lax.top_k(input, 3) == ([9, 6, 5], [5, 7, 4])
        let expectedValues: [Float] = [9.0, 6.0, 5.0]
        let expectedIndices: [Int32] = [5, 7, 4]

        print("top_k 1D values: \(values) indices: \(indices)")
        #expect(values.count == 3)
        #expect(indices.count == 3)
        for i in 0..<3 {
            #expect(abs(values[i] - expectedValues[i]) < 1e-6,
                    "value[\(i)] = \(values[i]), expected \(expectedValues[i])")
            #expect(indices[i] == expectedIndices[i],
                    "index[\(i)] = \(indices[i]), expected \(expectedIndices[i])")
        }
    }

    @Test("top_k k=2 over a 2D array operates per-row on the last axis, ties stable")
    func testTopK2D() async throws {
        let mlir = """
        module @topk_test_2d {
          func.func @main(%arg0: tensor<3x4xf32>) -> (tensor<3x2xf32>, tensor<3x2xi32>) {
            %0:2 = stablehlo.top_k %arg0 {k = 2} : (tensor<3x4xf32>) -> (tensor<3x2xf32>, tensor<3x2xi32>)
            return %0#0, %0#1 : tensor<3x2xf32>, tensor<3x2xi32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2
        let executable = try client.compile(mlir, config: config)

        // Row 0: [2, 8, 1, 5]   -> top2 [8, 5] @ [1, 3]
        // Row 1: [7, 7, 0, 7]   -> top2 [7, 7] @ [0, 1]  (ties: lowest index first)
        // Row 2: [-3, -1, -9, 0] -> top2 [0, -1] @ [3, 1]
        let input: [Float] = [
            2.0, 8.0, 1.0, 5.0,
            7.0, 7.0, 0.0, 7.0,
            -3.0, -1.0, -9.0, 0.0,
        ]
        let inBuffer = try client.createBuffer(input, shape: [3, 4], elementType: .float32)

        let outputs = try executable.execute([inBuffer])
        let values = try outputs[0].toFloatArray()
        let indices = try outputs[1].toInt32Array()

        let expectedValues: [Float] = [8.0, 5.0, 7.0, 7.0, 0.0, -1.0]
        let expectedIndices: [Int32] = [1, 3, 0, 1, 3, 1]

        print("top_k 2D values: \(values) indices: \(indices)")
        #expect(values.count == 6)
        #expect(indices.count == 6)
        for i in 0..<6 {
            #expect(abs(values[i] - expectedValues[i]) < 1e-6,
                    "value[\(i)] = \(values[i]), expected \(expectedValues[i])")
            #expect(indices[i] == expectedIndices[i],
                    "index[\(i)] = \(indices[i]), expected \(expectedIndices[i])")
        }
    }
}
