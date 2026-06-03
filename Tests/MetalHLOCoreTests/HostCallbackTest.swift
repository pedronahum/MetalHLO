// HostCallbackTest.swift
// Host callbacks (jax.debug.print / jax.debug.callback) lower to a
// side-effecting stablehlo.custom_call @xla_ffi_python_cpu_callback with no SSA
// result (`-> ()`). We can't run the host round-trip, but because the op
// produces no value the program depends on, dropping it is numerically exact —
// only the host-side print is lost. These tests pin that behaviour:
//   - a program containing jax.debug.print still COMPILES and RUNS, computing
//     the surrounding math correctly (the print is silently dropped); and
//   - a result-producing callback (pure_callback) still fails LOUDLY, since it
//     would feed real values back into the graph and we have no host infra.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Host Callback Tests", .serialized)
struct HostCallbackTests {

    @Test("jax.debug.print is dropped; surrounding compute is correct")
    func testDebugPrintDropped() async throws {
        // Exact form JAX 0.10 emits for:
        //   def f(x):
        //       jax.debug.print('x={x}', x=x)
        //       return x + 1
        // The custom_call carries has_side_effect = true and no result (`-> ()`).
        let mlir = """
        module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
          sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>
          func.func public @main(%arg0: tensor<3xf32>) -> (tensor<3xf32> {jax.result_info = "result"}) {
            stablehlo.custom_call @xla_ffi_python_cpu_callback(%arg0) {backend_config = "", has_side_effect = true, mhlo.backend_config = {index = 0 : ui64}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [], sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>} : (tensor<3xf32>) -> ()
            %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
            %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3xf32>
            %1 = stablehlo.add %arg0, %0 : tensor<3xf32>
            return %1 : tensor<3xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2
        let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [10, 20, 30]
        let inputBuffer = try client.createBuffer(inputData, shape: [3], elementType: .float32)
        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("debug.print dropped, x+1 = \(result)")
        let expected: [Float] = [11, 21, 31]
        #expect(result.count == 3)
        for i in 0..<3 {
            #expect(abs(result[i] - expected[i]) < 1e-4, "idx \(i): got \(result[i]) expected \(expected[i])")
        }
    }

    @Test("debug callback before AND after a value is dropped without disturbing dataflow")
    func testCallbackInterleaved() async throws {
        // A side-effecting callback sitting between producer and consumer must be
        // dropped without breaking the SSA chain around it.
        let mlir = """
        module @jit_g attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
          func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
            %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
            %1 = stablehlo.multiply %arg0, %0 : tensor<4xf32>
            stablehlo.custom_call @xla_ffi_python_cpu_callback(%1) {has_side_effect = true, mhlo.backend_config = {index = 0 : ui64}} : (tensor<4xf32>) -> ()
            %2 = stablehlo.add %1, %0 : tensor<4xf32>
            return %2 : tensor<4xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2
        let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [1, 2, 3, 4]
        let inputBuffer = try client.createBuffer(inputData, shape: [4], elementType: .float32)
        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("interleaved callback dropped, (x*2)+2 = \(result)")
        let expected: [Float] = [4, 6, 8, 10] // (x*2)+2
        #expect(result.count == 4)
        for i in 0..<4 {
            #expect(abs(result[i] - expected[i]) < 1e-4, "idx \(i): got \(result[i]) expected \(expected[i])")
        }
    }

    @Test("result-producing host callback (pure_callback) fails loudly")
    func testResultProducingCallbackRejected() async throws {
        // pure_callback returns a value the graph consumes. We have no host
        // round-trip, so faking it would corrupt numerics — this must NOT silently
        // compile. It has an SSA result, so it isn't even a result-less statement;
        // codegen has no kernel for the @xla_ffi_python_cpu_callback target and the
        // compile must throw.
        let mlir = """
        module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
          func.func public @main(%arg0: tensor<3xf32>) -> (tensor<3xf32>) {
            %0 = stablehlo.custom_call @xla_ffi_python_cpu_callback(%arg0) {mhlo.backend_config = {index = 0 : ui64}} : (tensor<3xf32>) -> tensor<3xf32>
            return %0 : tensor<3xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2
        var threw = false
        do {
            let executable = try client.compile(mlir, config: config)
            let inputBuffer = try client.createBuffer([1, 2, 3] as [Float], shape: [3], elementType: .float32)
            _ = try executable.execute([inputBuffer])
        } catch {
            threw = true
            print("result-producing callback correctly rejected: \(error)")
        }
        #expect(threw, "a result-producing host callback must not silently compile/run")
    }
}
