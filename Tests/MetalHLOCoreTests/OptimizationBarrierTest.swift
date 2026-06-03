// OptimizationBarrierTest.swift
// Tests stablehlo.optimization_barrier — the variadic identity/ordering op
// emitted by jax.checkpoint / jax.remat / jax.lax.optimization_barrier.
// Each result aliases its corresponding operand; only scheduling is constrained.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Optimization Barrier Tests", .serialized)
struct OptimizationBarrierTests {

    @Test("Single-result barrier round-trips the operand unchanged")
    func testSingleResultIdentity() async throws {
        // jax.lax.optimization_barrier(x) lowers to a 1-result barrier.
        let mlir = """
        module @ob_identity {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.optimization_barrier %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        let inputData: [Float] = [1, 2, 3, 4]
        let inputBuffer = try client.createBuffer(inputData, shape: [4], elementType: .float32)

        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("OB identity: \(result)")
        #expect(result == inputData, "Barrier must pass the operand through unchanged")
    }

    @Test("Multi-result barrier aliases each result to its operand")
    func testMultiResultBarrier() async throws {
        // Mirrors the exact StableHLO JAX 0.10 emits for
        //   def f(x): a,b = optimization_barrier((x, x*2)); return a + b
        let mlir = """
        module @ob_multi {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
            %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
            %1 = stablehlo.multiply %arg0, %0 : tensor<4xf32>
            %2:2 = stablehlo.optimization_barrier %arg0, %1 : tensor<4xf32>, tensor<4xf32>
            %3 = stablehlo.add %2#0, %2#1 : tensor<4xf32>
            return %3 : tensor<4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        let inputData: [Float] = [1, 2, 3, 4]
        let inputBuffer = try client.createBuffer(inputData, shape: [4], elementType: .float32)

        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        // a + b = x + 2x = 3x
        let expected: [Float] = [3, 6, 9, 12]
        print("OB multi: \(result)")
        #expect(result == expected, "Expected 3x = \(expected), got \(result)")
    }

    @Test("Barrier inside a tiny checkpointed MLP compiles and runs")
    func testCheckpointedMLP() async throws {
        // Small MLP fragment: y = relu(x @ w) with the activation guarded by a
        // barrier (the shape jax.checkpoint produces around remat boundaries).
        let mlir = """
        module @ob_mlp {
          func.func @main(%x: tensor<2x3xf32>, %w: tensor<3x2xf32>) -> (tensor<2x2xf32>) {
            %z = stablehlo.dot_general %x, %w,
              #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
              : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
            %zb = stablehlo.optimization_barrier %z : tensor<2x2xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<2x2xf32>
            %y = stablehlo.maximum %zb, %zero : tensor<2x2xf32>
            return %y : tensor<2x2xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // x = [[1,2,3],[4,5,6]], w = [[1,0],[0,1],[1,1]]
        let xData: [Float] = [1, 2, 3, 4, 5, 6]
        let wData: [Float] = [1, 0, 0, 1, 1, 1]
        let xBuf = try client.createBuffer(xData, shape: [2, 3], elementType: .float32)
        let wBuf = try client.createBuffer(wData, shape: [3, 2], elementType: .float32)

        let outputs = try executable.execute([xBuf, wBuf])
        let result = try outputs[0].toFloatArray()

        // z[0] = [1*1+3*1, 2*1+3*1] = [4, 5]; z[1] = [4+6, 5+6] = [10, 11]
        let expected: [Float] = [4, 5, 10, 11]
        print("OB MLP: \(result)")
        #expect(result == expected, "Expected \(expected), got \(result)")
    }
}
