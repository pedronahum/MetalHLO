// ManualComputationTest.swift
// Distributed M0 — sdy.manual_computation (shard_map) desugar.
//
// At an N>=2 mesh, JAX wraps a shard_map body in `sdy.manual_computation` (the
// body runs on per-device shapes, with collectives across the mesh axis). The
// parser captures it and desugars it into a flat single-device program: replay
// the body once per shard over sliced inputs, lower all_reduce to a cross-shard
// combine, and assemble outputs per the output sharding. This is the
// single-process simulation of an N-device shard_map (every shard on one GPU).
// These guard the math: compile the global program and check the global result.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Manual Computation (shard_map desugar)", .serialized)
struct ManualComputationTests {

    // 2-device shard_map(psum): sum over a [8]->[4]-sharded input, all_reduced
    // across the 2 shards → global sum. (Exact post-PJRT-preprocessing IR.)
    @Test("manual_computation + all_reduce (psum) desugars to the global sum")
    func psumReplicatedOutput() throws {
        let mlir = """
        module @jit_f {
          func.func @main(%arg0: tensor<8xf32>) -> (tensor<f32>) {
            %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, []>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
              %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
              %1 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
              %2 = stablehlo.all_reduce %1, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids ({
              ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                stablehlo.return %3 : tensor<f32>
              }) : (tensor<f32>) -> tensor<f32>
              sdy.return %2 : tensor<f32>
            } : (tensor<8xf32>) -> tensor<f32>
            return %0 : tensor<f32>
          }
        }
        """
        let client = try Client.create()
        let exe = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        let xin: [Float] = [0, 1, 2, 3, 4, 5, 6, 7]   // global; shard 0 sums 0..3, shard 1 sums 4..7
        let x = try client.createBuffer(xin, shape: [8], elementType: .float32)
        let result = try exe.execute([x])[0].toFloatArray()
        #expect(result == [28.0], "expected global sum 28, got \(result)")
    }

    // Sharded output (no collective): each shard multiplies its [4] by 2; the
    // [4] per-shard outputs concatenate back to the global [8]. Exercises the
    // concat-assembly path.
    @Test("manual_computation with sharded output concatenates per-shard results")
    func shardedOutputConcat() throws {
        let mlir = """
        module @jit_g {
          func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32>) {
            %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
              %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
              %1 = stablehlo.multiply %arg1, %cst : tensor<4xf32>
              sdy.return %1 : tensor<4xf32>
            } : (tensor<8xf32>) -> tensor<8xf32>
            return %0 : tensor<8xf32>
          }
        }
        """
        let client = try Client.create()
        let exe = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        let xin: [Float] = [0, 1, 2, 3, 4, 5, 6, 7]
        let x = try client.createBuffer(xin, shape: [8], elementType: .float32)
        let result = try exe.execute([x])[0].toFloatArray()
        #expect(result == xin.map { $0 * 2 }, "expected input*2 concatenated, got \(result)")
    }
}
