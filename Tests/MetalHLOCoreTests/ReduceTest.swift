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

    // Global (all-axes) reductions over a large input are rewritten into two
    // cooperating reduce stages (ReductionSplitTransform) so they fill the GPU
    // instead of running in a single threadgroup. These guard that the 2-stage
    // path is numerically correct. 128×128 = 16384 elements with d0=128 trips
    // the split's size/shape gates; values are chosen to be order-independent.
    @Test("Global sum of a large 2D tensor (reduction-split path)")
    func testGlobalSumSplitPath() async throws {
        let mlir = """
        module @global_sum {
          func.func @main(%input: tensor<128x128xf32>) -> (tensor<f32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [0, 1] : (tensor<128x128xf32>, tensor<f32>) -> tensor<f32>
            return %0 : tensor<f32>
          }
        }
        """
        let client = try Client.create()
        let executable = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        // All-ones → sum is exactly 16384, independent of accumulation order.
        let input = try client.createBuffer([Float](repeating: 1.0, count: 128 * 128),
                                             shape: [128, 128], elementType: .float32)
        let result = try executable.execute([input])[0].toFloatArray()
        #expect(result.count == 1)
        #expect(result[0] == 16384.0, "global sum should be 16384, got \(result[0])")
    }

    @Test("Global max of a large 2D tensor (reduction-split path)")
    func testGlobalMaxSplitPath() async throws {
        let mlir = """
        module @global_max {
          func.func @main(%input: tensor<128x128xf32>) -> (tensor<f32>) {
            %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.maximum across dimensions = [0, 1] : (tensor<128x128xf32>, tensor<f32>) -> tensor<f32>
            return %0 : tensor<f32>
          }
        }
        """
        let client = try Client.create()
        let executable = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        // Mostly 1.0 with a single planted maximum of 42.0 at a non-corner index.
        var data = [Float](repeating: 1.0, count: 128 * 128)
        data[5000] = 42.0
        let input = try client.createBuffer(data, shape: [128, 128], elementType: .float32)
        let result = try executable.execute([input])[0].toFloatArray()
        #expect(result.count == 1)
        #expect(result[0] == 42.0, "global max should be 42, got \(result[0])")
    }

    // Long-axis column reductions (reduce over axis 0, reduceSize ≥ 64) use the
    // coalesced 2D-threadgroup kernel. 256 rows trips the long-axis path; 70
    // columns (not a multiple of the 32-wide tile) exercises the boundary guard.
    @Test("Column sum over a long first axis (coalesced kernel, ragged width)")
    func testColumnSumCoalescedKernel() async throws {
        let rows = 256, cols = 70
        let mlir = """
        module @col_sum {
          func.func @main(%input: tensor<256x70xf32>) -> (tensor<70xf32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [0] : (tensor<256x70xf32>, tensor<f32>) -> tensor<70xf32>
            return %0 : tensor<70xf32>
          }
        }
        """
        let client = try Client.create()
        let executable = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        // Column c holds the value c in every row → column sum is exactly 256*c.
        var data = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows { for c in 0..<cols { data[r * cols + c] = Float(c) } }
        let input = try client.createBuffer(data, shape: [rows, cols], elementType: .float32)
        let result = try executable.execute([input])[0].toFloatArray()
        #expect(result.count == cols)
        for c in 0..<cols {
            #expect(result[c] == Float(rows * c), "col \(c) sum should be \(rows * c), got \(result[c])")
        }
    }

    @Test("Column max over a long first axis (coalesced kernel)")
    func testColumnMaxCoalescedKernel() async throws {
        let rows = 128, cols = 40
        let mlir = """
        module @col_max {
          func.func @main(%input: tensor<128x40xf32>) -> (tensor<40xf32>) {
            %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.maximum across dimensions = [0] : (tensor<128x40xf32>, tensor<f32>) -> tensor<40xf32>
            return %0 : tensor<40xf32>
          }
        }
        """
        let client = try Client.create()
        let executable = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        // Column c: all 1.0 except a planted maximum of (c + 100) on one row.
        var data = [Float](repeating: 1.0, count: rows * cols)
        for c in 0..<cols { data[(c % rows) * cols + c] = Float(c) + 100 }
        let input = try client.createBuffer(data, shape: [rows, cols], elementType: .float32)
        let result = try executable.execute([input])[0].toFloatArray()
        #expect(result.count == cols)
        for c in 0..<cols {
            #expect(result[c] == Float(c) + 100, "col \(c) max should be \(Float(c) + 100), got \(result[c])")
        }
    }
}
