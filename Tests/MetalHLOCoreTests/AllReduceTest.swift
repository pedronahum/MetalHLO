// AllReduceTest.swift
// First increment of distributed/collective support (M0).
//
// stablehlo.all_reduce (jax.lax.psum) is now parsed. On a single device the
// replica group has size 1, so the reducer over one value is the identity —
// the op aliases its operand. These guard that the op parses and executes with
// correct single-device semantics. Real multi-device reduction is a later
// milestone (the distributed runtime intercepts the op before this collapse).

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("All-Reduce (single-replica)", .serialized)
struct AllReduceTests {

    // all_reduce of a compute result: %t = x + x; all_reduce(%t) == %t == 2x.
    @Test("all_reduce(add) over a compute output is identity on one device")
    func allReduceComputeOutput() async throws {
        let mlir = """
        module @all_reduce_compute {
          func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32>) {
            %t = stablehlo.add %arg0, %arg0 : tensor<8xf32>
            %0 = "stablehlo.all_reduce"(%t) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0]]> : tensor<1x1xi64>, use_global_device_ids}> ({
            ^bb0(%a: tensor<f32>, %b: tensor<f32>):
              %c = stablehlo.add %a, %b : tensor<f32>
              stablehlo.return %c : tensor<f32>
            }) : (tensor<8xf32>) -> tensor<8xf32>
            return %0 : tensor<8xf32>
          }
        }
        """
        let client = try Client.create()
        let exe = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        let xin: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let x = try client.createBuffer(xin, shape: [8], elementType: .float32)
        let result = try exe.execute([x])[0].toFloatArray()
        #expect(result == xin.map { $0 * 2 }, "all_reduce(2x) should be 2x, got \(result)")
    }

    // all_reduce directly on a function input (passthrough output path).
    @Test("all_reduce(add) over a function input is identity on one device")
    func allReduceInput() async throws {
        let mlir = """
        module @all_reduce_input {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = "stablehlo.all_reduce"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0]]> : tensor<1x1xi64>, use_global_device_ids}> ({
            ^bb0(%a: tensor<f32>, %b: tensor<f32>):
              %c = stablehlo.add %a, %b : tensor<f32>
              stablehlo.return %c : tensor<f32>
            }) : (tensor<4xf32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let client = try Client.create()
        let exe = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        let xin: [Float] = [10, 20, 30, 40]
        let x = try client.createBuffer(xin, shape: [4], elementType: .float32)
        let result = try exe.execute([x])[0].toFloatArray()
        #expect(result == xin, "single-replica all_reduce should pass the input through, got \(result)")
    }
}
