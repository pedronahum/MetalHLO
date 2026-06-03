// CumsumTest.swift
// Tests cumsum lowered by JAX to stablehlo.reduce_window with left padding.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Cumsum Tests", .serialized)
struct CumsumTests {

    @Test("cumsum 1D (JAX generic reduce_window form)")
    func testCumsum1D() async throws {
        // Exact form JAX 0.10 emits for jnp.cumsum on a 1-D tensor.
        let mlir = """
        module @jit_cumsum {
          func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32>) {
            %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
            %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1>, padding = dense<[[7, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 8>, window_strides = array<i64: 1>}> ({
            ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
              %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
              stablehlo.return %1 : tensor<f32>
            }) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
            return %0 : tensor<8xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2; let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let inputBuffer = try client.createBuffer(inputData, shape: [8], elementType: .float32)
        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("cumsum 1D: \(result)")
        let expected: [Float] = [1, 3, 6, 10, 15, 21, 28, 36]
        #expect(result.count == 8)
        for i in 0..<8 {
            #expect(abs(result[i] - expected[i]) < 1e-4, "idx \(i): got \(result[i]) expected \(expected[i])")
        }
    }

    @Test("cumsum 2D axis=0")
    func testCumsum2DAxis0() async throws {
        let mlir = """
        module @jit_cumsum {
          func.func @main(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
            %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
            %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[2, 0], [0, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 1, 1>}> ({
            ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
              %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
              stablehlo.return %1 : tensor<f32>
            }) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
            return %0 : tensor<3x4xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2; let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        let inputBuffer = try client.createBuffer(inputData, shape: [3, 4], elementType: .float32)
        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("cumsum 2D axis=0: \(result)")
        // column-cumsum
        let expected: [Float] = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21]
        #expect(result.count == 12)
        for i in 0..<12 {
            #expect(abs(result[i] - expected[i]) < 1e-4, "idx \(i): got \(result[i]) expected \(expected[i])")
        }
    }

    @Test("cumsum 1D — exact JAX lowering with nested call wrappers")
    func testCumsum1DFullLowering() async throws {
        // The full module JAX 0.10 emits: main -> call @cumsum -> call @cumsum_0,
        // where @cumsum_0 holds the reduce_window. Exercises nested-call inlining
        // together with the windowed prefix sum.
        let mlir = """
        module @jit__lambda {
          func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32>) {
            %0 = call @cumsum(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
            return %0 : tensor<8xf32>
          }
          func.func private @cumsum(%arg0: tensor<8xf32>) -> tensor<8xf32> {
            %0 = call @cumsum_0(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
            return %0 : tensor<8xf32>
          }
          func.func private @cumsum_0(%arg0: tensor<8xf32>) -> tensor<8xf32> {
            %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
            %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<f32>
            %1 = "stablehlo.reduce_window"(%arg0, %0) <{base_dilations = array<i64: 1>, padding = dense<[[7, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 8>, window_strides = array<i64: 1>}> ({
            ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
              %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
              stablehlo.return %2 : tensor<f32>
            }) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
            return %1 : tensor<8xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2
        let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let inputBuffer = try client.createBuffer(inputData, shape: [8], elementType: .float32)
        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("cumsum 1D (full lowering): \(result)")
        let expected: [Float] = [1, 3, 6, 10, 15, 21, 28, 36]
        #expect(result.count == 8)
        for i in 0..<8 {
            #expect(abs(result[i] - expected[i]) < 1e-4, "idx \(i): got \(result[i]) expected \(expected[i])")
        }
    }

    @Test("cumsum 2D axis=0 — plugin converter form (bracketed window_dimensions + nested calls)")
    func testCumsum2DAxis0PluginForm() async throws {
        // Exact MLIR the PJRT plugin's bytecode-to-text converter produces for
        // jnp.cumsum(x, axis=0): simplified reduce_window with bracketed
        // window_dimensions, wrapped in two private-call levels.
        let mlir = """
        module @jit__lambda {
          func.func @main(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
            %0 = call @cumsum(%arg0) : (tensor<3x4xf32>) -> tensor<3x4xf32>
            return %0 : tensor<3x4xf32>
          }
          func.func private @cumsum(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
            %0 = call @cumsum_0(%arg0) : (tensor<3x4xf32>) -> tensor<3x4xf32>
            return %0 : tensor<3x4xf32>
          }
          func.func private @cumsum_0(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
            %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
            %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<f32>
            %1 = stablehlo.reduce_window %arg0, %0, padding = dense<[[2, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = [3, 1] ({
            ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
              %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
              stablehlo.return %2 : tensor<f32>
            }) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
            return %1 : tensor<3x4xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2
        let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        let inputBuffer = try client.createBuffer(inputData, shape: [3, 4], elementType: .float32)
        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("cumsum 2D axis=0 (plugin form): \(result)")
        let expected: [Float] = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21]
        #expect(result.count == 12)
        for i in 0..<12 {
            #expect(abs(result[i] - expected[i]) < 1e-4, "idx \(i): got \(result[i]) expected \(expected[i])")
        }
    }

    @Test("cumsum 2D axis=1")
    func testCumsum2DAxis1() async throws {
        let mlir = """
        module @jit_cumsum {
          func.func @main(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
            %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
            %0 = "stablehlo.reduce_window"(%arg0, %cst) <{base_dilations = array<i64: 1, 1>, padding = dense<[[0, 0], [3, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 4>, window_strides = array<i64: 1, 1>}> ({
            ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
              %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
              stablehlo.return %1 : tensor<f32>
            }) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
            return %0 : tensor<3x4xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig(); config.optimizationLevel = .O2; let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        let inputBuffer = try client.createBuffer(inputData, shape: [3, 4], elementType: .float32)
        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("cumsum 2D axis=1: \(result)")
        // row-cumsum
        let expected: [Float] = [0, 1, 3, 6, 4, 9, 15, 22, 8, 17, 27, 38]
        #expect(result.count == 12)
        for i in 0..<12 {
            #expect(abs(result[i] - expected[i]) < 1e-4, "idx \(i): got \(result[i]) expected \(expected[i])")
        }
    }
}
