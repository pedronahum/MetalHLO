// ReduceTest.swift
// Tests reduce operations on multi-row tensors

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Reduce Tests", .serialized)
struct ReduceTests {

    @Test("Reduce sum 2D along axis 1 - simplified format (default)")
    func testReduceSum2DAxis1Default() async throws {
        let mlir = """
        module @reduce_test {
          func.func @main(%input: tensor<2x4xf32>) -> (tensor<2xf32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
            return %0 : tensor<2xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let inputBuffer = try client.createBuffer(inputData, shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("Reduce sum (default): \(result)")
        #expect(result.count == 2)
        #expect(result[0] == 10.0, "Row 0 sum should be 10")
        #expect(result[1] == 26.0, "Row 1 sum should be 26, got \(result[1])")
    }

    @Test("Reduce sum 2D along axis 1 - O3")
    func testReduceSum2DAxis1O3() async throws {
        let mlir = """
        module @reduce_test {
          func.func @main(%input: tensor<2x4xf32>) -> (tensor<2xf32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
            return %0 : tensor<2xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig()
        config.optimizationLevel = .O3
        let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let inputBuffer = try client.createBuffer(inputData, shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("Reduce sum (O3): \(result)")
        #expect(result.count == 2)
        #expect(result[0] == 10.0, "Row 0 sum should be 10")
        #expect(result[1] == 26.0, "Row 1 sum should be 26, got \(result[1])")
    }

    @Test("Reduce mean 2D along axis 1 (default)")
    func testReduceMean2DAxis1Default() async throws {
        let mlir = """
        module @mean_test {
          func.func @main(%input: tensor<2x4xf32>) -> (tensor<2x1xf32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %sum = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
            %sum_reshaped = stablehlo.reshape %sum : (tensor<2xf32>) -> tensor<2x1xf32>
            %count = stablehlo.constant dense<4.0> : tensor<2x1xf32>
            %mean = stablehlo.divide %sum_reshaped, %count : tensor<2x1xf32>
            return %mean : tensor<2x1xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let inputBuffer = try client.createBuffer(inputData, shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("Mean (default): \(result)")
        #expect(result.count == 2)
        #expect(result[0] == 2.5, "Row 0 mean should be 2.5")
        #expect(result[1] == 6.5, "Row 1 mean should be 6.5, got \(result[1])")
    }

    @Test("Reduce mean 2D along axis 1 (O3)")
    func testReduceMean2DAxis1O3() async throws {
        let mlir = """
        module @mean_test {
          func.func @main(%input: tensor<2x4xf32>) -> (tensor<2x1xf32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %sum = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
            %sum_reshaped = stablehlo.reshape %sum : (tensor<2xf32>) -> tensor<2x1xf32>
            %count = stablehlo.constant dense<4.0> : tensor<2x1xf32>
            %mean = stablehlo.divide %sum_reshaped, %count : tensor<2x1xf32>
            return %mean : tensor<2x1xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig()
        config.optimizationLevel = .O3
        let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let inputBuffer = try client.createBuffer(inputData, shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("Mean (O3): \(result)")
        #expect(result.count == 2)
        #expect(result[0] == 2.5, "Row 0 mean should be 2.5")
        #expect(result[1] == 6.5, "Row 1 mean should be 6.5, got \(result[1])")
    }

    @Test("Reduce sum + reshape only (O3) - isolate reshape")
    func testReduceSumReshapeOnlyO3() async throws {
        // Test if reduce + reshape alone is broken
        let mlir = """
        module @test {
          func.func @main(%input: tensor<2x4xf32>) -> (tensor<2x1xf32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %sum = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
            %reshaped = stablehlo.reshape %sum : (tensor<2xf32>) -> tensor<2x1xf32>
            return %reshaped : tensor<2x1xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig()
        config.optimizationLevel = .O3
        let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let inputBuffer = try client.createBuffer(inputData, shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("Reduce+reshape (O3): \(result)")
        #expect(result.count == 2)
        #expect(result[0] == 10.0, "Row 0 sum should be 10, got \(result[0])")
        #expect(result[1] == 26.0, "Row 1 sum should be 26, got \(result[1])")
    }

    @Test("Reduce sum + divide without reshape (O3) - isolate divide")
    func testReduceSumDivideNoReshapeO3() async throws {
        // Test if reduce + divide (no reshape) is broken
        let mlir = """
        module @test {
          func.func @main(%input: tensor<2x4xf32>) -> (tensor<2xf32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %sum = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
            %count = stablehlo.constant dense<4.0> : tensor<2xf32>
            %mean = stablehlo.divide %sum, %count : tensor<2xf32>
            return %mean : tensor<2xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig()
        config.optimizationLevel = .O3
        let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let inputBuffer = try client.createBuffer(inputData, shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("Reduce+divide no reshape (O3): \(result)")
        #expect(result.count == 2)
        #expect(result[0] == 2.5, "Row 0 mean should be 2.5, got \(result[0])")
        #expect(result[1] == 6.5, "Row 1 mean should be 6.5, got \(result[1])")
    }

    @Test("Reduce sum 3D along axis 2 - simplified format")
    func testReduceSum3DAxis2() async throws {
        let mlir = """
        module @reduce3d_test {
          func.func @main(%input: tensor<2x3x4xf32>) -> (tensor<2x3xf32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [2] : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // 2x3x4 tensor with known values
        var inputData: [Float] = []
        for i in 0..<24 {
            inputData.append(Float(i + 1))
        }
        let inputBuffer = try client.createBuffer(inputData, shape: [2, 3, 4], elementType: .float32)

        let outputs = try executable.execute([inputBuffer])
        let result = try outputs[0].toFloatArray()

        print("3D reduce sum result: \(result)")
        // Row sums: [1+2+3+4=10, 5+6+7+8=26, 9+10+11+12=42, 13+14+15+16=58, 17+18+19+20=74, 21+22+23+24=90]
        #expect(result.count == 6)
        #expect(result[0] == 10.0, "Expected 10, got \(result[0])")
        #expect(result[1] == 26.0, "Expected 26, got \(result[1])")
        #expect(result[5] == 90.0, "Expected 90, got \(result[5])")
    }
}
