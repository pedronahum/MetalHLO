// CrossEntropyTest.swift
// Tests the cross-entropy MLIR pattern

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("CrossEntropy Tests", .serialized)
struct CrossEntropyTests {

    @Test("Cross-entropy MLIR (O3)")
    func testCrossEntropyO3() async throws {
        // Exact MLIR that Magma generates for cross-entropy
        let mlir = """
        module @ce_test {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x1xf32>, %arg2: tensor<1x3xf32>) -> (tensor<f32>) {
            %1 = stablehlo.constant dense<-3.4028235e+38> : tensor<f32>
            %0 = stablehlo.reduce %arg0, %1 applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            %2 = stablehlo.reshape %0 : (tensor<2xf32>) -> tensor<2x1xf32>
            %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x3xf32>
            %4 = stablehlo.subtract %arg0, %3 : tensor<2x3xf32>
            %5 = stablehlo.exponential %4 : tensor<2x3xf32>
            %7 = stablehlo.constant dense<0.0> : tensor<f32>
            %6 = stablehlo.reduce %5, %7 applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            %8 = stablehlo.reshape %6 : (tensor<2xf32>) -> tensor<2x1xf32>
            %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x3xf32>
            %10 = stablehlo.divide %5, %9 : tensor<2x3xf32>
            %11 = stablehlo.log %10 : tensor<2x3xf32>
            %12 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x3xf32>
            %13 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<1x3xf32>) -> tensor<2x3xf32>
            %14 = stablehlo.compare EQ, %12, %13, FLOAT : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
            %15 = stablehlo.convert %14 : (tensor<2x3xi1>) -> tensor<2x3xf32>
            %16 = stablehlo.multiply %11, %15 : tensor<2x3xf32>
            %18 = stablehlo.constant dense<0.0> : tensor<f32>
            %17 = stablehlo.reduce %16, %18 applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            %19 = stablehlo.negate %17 : tensor<2xf32>
            %21 = stablehlo.constant dense<0.0> : tensor<f32>
            %20 = stablehlo.reduce %19, %21 applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
            %22 = stablehlo.constant dense<2.0> : tensor<f32>
            %23 = stablehlo.divide %20, %22 : tensor<f32>
            return %23 : tensor<f32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig()
        config.optimizationLevel = .O3
        let executable = try client.compile(mlir, config: config)

        // logits: [[2.0, 1.0, 0.1], [0.5, 1.5, 0.3]]
        let logitsData: [Float] = [2.0, 1.0, 0.1, 0.5, 1.5, 0.3]
        // targets reshaped: [[0.0], [2.0]]
        let targetsData: [Float] = [0.0, 2.0]
        // classIndices: [[0.0, 1.0, 2.0]]
        let classData: [Float] = [0.0, 1.0, 2.0]

        let logitsBuf = try client.createBuffer(logitsData, shape: [2, 3], elementType: .float32)
        let targetsBuf = try client.createBuffer(targetsData, shape: [2, 1], elementType: .float32)
        let classBuf = try client.createBuffer(classData, shape: [1, 3], elementType: .float32)

        let outputs = try executable.execute([logitsBuf, targetsBuf, classBuf])
        let result = try outputs[0].toFloatArray()

        print("Cross-entropy (O3): \(result)")
        // Expected: ~0.965
        // Row 0: target=0, logSoftmax[0] = -0.417 → loss = 0.417
        // Row 1: target=2, logSoftmax[2] = -1.712 → loss = 1.712
        // Mean = (0.417 + 1.712) / 2 = 1.065
        #expect(result.count == 1)
        #expect(result[0] > 0.5 && result[0] < 2.0, "Expected ~1.065, got \(result[0])")
    }

    @Test("Cross-entropy MLIR (default)")
    func testCrossEntropyDefault() async throws {
        let mlir = """
        module @ce_test {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x1xf32>, %arg2: tensor<1x3xf32>) -> (tensor<f32>) {
            %1 = stablehlo.constant dense<-3.4028235e+38> : tensor<f32>
            %0 = stablehlo.reduce %arg0, %1 applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            %2 = stablehlo.reshape %0 : (tensor<2xf32>) -> tensor<2x1xf32>
            %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x3xf32>
            %4 = stablehlo.subtract %arg0, %3 : tensor<2x3xf32>
            %5 = stablehlo.exponential %4 : tensor<2x3xf32>
            %7 = stablehlo.constant dense<0.0> : tensor<f32>
            %6 = stablehlo.reduce %5, %7 applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            %8 = stablehlo.reshape %6 : (tensor<2xf32>) -> tensor<2x1xf32>
            %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x3xf32>
            %10 = stablehlo.divide %5, %9 : tensor<2x3xf32>
            %11 = stablehlo.log %10 : tensor<2x3xf32>
            %12 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x3xf32>
            %13 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<1x3xf32>) -> tensor<2x3xf32>
            %14 = stablehlo.compare EQ, %12, %13, FLOAT : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
            %15 = stablehlo.convert %14 : (tensor<2x3xi1>) -> tensor<2x3xf32>
            %16 = stablehlo.multiply %11, %15 : tensor<2x3xf32>
            %18 = stablehlo.constant dense<0.0> : tensor<f32>
            %17 = stablehlo.reduce %16, %18 applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            %19 = stablehlo.negate %17 : tensor<2xf32>
            %21 = stablehlo.constant dense<0.0> : tensor<f32>
            %20 = stablehlo.reduce %19, %21 applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
            %22 = stablehlo.constant dense<2.0> : tensor<f32>
            %23 = stablehlo.divide %20, %22 : tensor<f32>
            return %23 : tensor<f32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        let logitsData: [Float] = [2.0, 1.0, 0.1, 0.5, 1.5, 0.3]
        let targetsData: [Float] = [0.0, 2.0]
        let classData: [Float] = [0.0, 1.0, 2.0]

        let logitsBuf = try client.createBuffer(logitsData, shape: [2, 3], elementType: .float32)
        let targetsBuf = try client.createBuffer(targetsData, shape: [2, 1], elementType: .float32)
        let classBuf = try client.createBuffer(classData, shape: [1, 3], elementType: .float32)

        let outputs = try executable.execute([logitsBuf, targetsBuf, classBuf])
        let result = try outputs[0].toFloatArray()

        print("Cross-entropy (default): \(result)")
        #expect(result.count == 1)
        #expect(result[0] > 0.5 && result[0] < 2.0, "Expected ~1.065, got \(result[0])")
    }

    @Test("Reduce max + reshape (O3)")
    func testReduceMaxReshapeO3() async throws {
        let mlir = """
        module @test {
          func.func @main(%input: tensor<2x3xf32>) -> (tensor<2x1xf32>) {
            %init = stablehlo.constant dense<-3.4028235e+38> : tensor<f32>
            %max = stablehlo.reduce %input, %init applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            %reshaped = stablehlo.reshape %max : (tensor<2xf32>) -> tensor<2x1xf32>
            return %reshaped : tensor<2x1xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig()
        config.optimizationLevel = .O3
        let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [2.0, 1.0, 0.1, 0.5, 1.5, 0.3]
        let inputBuffer = try client.createBuffer(inputData, shape: [2, 3], elementType: .float32)

        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("Reduce max + reshape (O3): \(result)")
        #expect(result.count == 2)
        #expect(result[0] == 2.0, "Row 0 max should be 2.0, got \(result[0])")
        #expect(result[1] == 1.5, "Row 1 max should be 1.5, got \(result[1])")
    }
}
