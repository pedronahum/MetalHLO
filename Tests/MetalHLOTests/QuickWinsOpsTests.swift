// QuickWinsOpsTests.swift
// MetalHLOTests
//
// Tests for the Quick Wins operations: convolution, pooling, batch norm, FFT, sort, and more.

import Testing
@testable import MetalHLO

// MARK: - Math Operations (tan, logistic, is_finite, reverse)

@Suite("Quick Wins - Math Operations")
struct MathOpsTests {

    @Test("Tan operation")
    func tan() throws {
        let client = try Client.create()
        let mlir = """
        module @tan {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.tan %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let pi = Float.pi
        let a = try client.createBuffer([0, pi/4, -pi/4, pi/6] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // tan(0) = 0, tan(pi/4) = 1, tan(-pi/4) = -1, tan(pi/6) ≈ 0.577
        #expect(abs(result[0] - 0.0) < 0.001)
        #expect(abs(result[1] - 1.0) < 0.01)
        #expect(abs(result[2] - (-1.0)) < 0.01)
        #expect(abs(result[3] - 0.577) < 0.01)
    }

    @Test("Logistic (sigmoid) operation")
    func logistic() throws {
        let client = try Client.create()
        let mlir = """
        module @logistic {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.logistic %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([0, 1, -1, 10] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269, sigmoid(10) ≈ 1.0
        #expect(abs(result[0] - 0.5) < 0.001)
        #expect(abs(result[1] - 0.731) < 0.01)
        #expect(abs(result[2] - 0.269) < 0.01)
        #expect(abs(result[3] - 1.0) < 0.001)
    }

    @Test("is_finite operation")
    func isFinite() throws {
        let client = try Client.create()
        let mlir = """
        module @is_finite {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xi1>) {
            %0 = stablehlo.is_finite %arg0 : (tensor<4xf32>) -> tensor<4xi1>
            return %0 : tensor<4xi1>
          }
        }
        """
        let executable = try client.compile(mlir)
        let inf = Float.infinity
        let nan = Float.nan
        let a = try client.createBuffer([1.0, inf, -inf, nan] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toBoolArray()
        #expect(result[0] == true)   // 1.0 is finite
        #expect(result[1] == false)  // inf is not finite
        #expect(result[2] == false)  // -inf is not finite
        #expect(result[3] == false)  // nan is not finite
    }

    @Test("Reverse operation")
    func reverse() throws {
        let client = try Client.create()
        let mlir = """
        module @reverse {
          func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.reverse %arg0, dims = [1] : tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        // Input: [[1,2,3], [4,5,6]]
        let a = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [2, 3], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // Reversed along axis 1: [[3,2,1], [6,5,4]]
        #expect(result == [3, 2, 1, 6, 5, 4])
    }
}

// MARK: - Convolution Operations

@Suite("Quick Wins - Convolution")
struct ConvolutionOpsTests {

    @Test("Basic 2D convolution")
    func convolution2D() throws {
        let client = try Client.create()
        // Simple 3x3 convolution on 4x4 input (NHWC format)
        // Use simplified attribute syntax
        let mlir = """
        module @conv2d {
          func.func @main(%arg0: tensor<1x4x4x1xf32>, %arg1: tensor<3x3x1x1xf32>) -> (tensor<1x2x2x1xf32>) {
            %0 = stablehlo.convolution %arg0, %arg1 window_strides = [1, 1], feature_group_count = 1 : (tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x2x2x1xf32>
            return %0 : tensor<1x2x2x1xf32>
          }
        }
        """
        let executable = try client.compile(mlir)

        // 4x4 input (NHWC: batch=1, height=4, width=4, channels=1)
        let input = try client.createBuffer(
            [1, 2, 3, 4,
             5, 6, 7, 8,
             9, 10, 11, 12,
             13, 14, 15, 16] as [Float],
            shape: [1, 4, 4, 1],
            elementType: .float32
        )

        // 3x3 kernel (HWIO: height=3, width=3, in_channels=1, out_channels=1)
        // All ones kernel - computes sum
        let kernel = try client.createBuffer(
            [1, 1, 1,
             1, 1, 1,
             1, 1, 1] as [Float],
            shape: [3, 3, 1, 1],
            elementType: .float32
        )

        let outputs = try executable.execute([input, kernel])
        let result = try outputs[0].toFloatArray()

        // With 3x3 kernel and no padding, output is 2x2
        // Top-left: sum of 1+2+3+5+6+7+9+10+11 = 54
        // Top-right: sum of 2+3+4+6+7+8+10+11+12 = 63
        // etc.
        #expect(result.count == 4)
        #expect(abs(result[0] - 54.0) < 0.01)
        #expect(abs(result[1] - 63.0) < 0.01)
    }
}

// MARK: - Pooling Operations (reduce_window)

@Suite("Quick Wins - Pooling")
struct PoolingOpsTests {

    @Test("Max pooling 2x2")
    func maxPooling2x2() throws {
        let client = try Client.create()
        let mlir = """
        module @maxpool {
          func.func @main(%arg0: tensor<1x4x4x1xf32>, %arg1: tensor<f32>) -> (tensor<1x2x2x1xf32>) {
            %0 = stablehlo.reduce_window %arg0, %arg1
                window_dimensions = [1, 2, 2, 1],
                window_strides = [1, 2, 2, 1],
                applies stablehlo.maximum
                : (tensor<1x4x4x1xf32>, tensor<f32>) -> tensor<1x2x2x1xf32>
            return %0 : tensor<1x2x2x1xf32>
          }
        }
        """
        let executable = try client.compile(mlir)

        // 4x4 input (NHWC)
        let input = try client.createBuffer(
            [1, 2, 3, 4,
             5, 6, 7, 8,
             9, 10, 11, 12,
             13, 14, 15, 16] as [Float],
            shape: [1, 4, 4, 1],
            elementType: .float32
        )

        // Init value for max is -inf, but we'll use 0 for simplicity
        let initVal = try client.createBuffer([Float.leastNormalMagnitude] as [Float], shape: [], elementType: .float32)

        let outputs = try executable.execute([input, initVal])
        let result = try outputs[0].toFloatArray()

        // 2x2 max pooling with stride 2 on 4x4 input gives 2x2 output
        // Top-left: max(1,2,5,6) = 6
        // Top-right: max(3,4,7,8) = 8
        // Bottom-left: max(9,10,13,14) = 14
        // Bottom-right: max(11,12,15,16) = 16
        #expect(result.count == 4)
        #expect(result[0] == 6)
        #expect(result[1] == 8)
        #expect(result[2] == 14)
        #expect(result[3] == 16)
    }
}

// MARK: - Batch Normalization Operations

@Suite("Quick Wins - Batch Normalization")
struct BatchNormOpsTests {

    @Test("Batch norm inference")
    func batchNormInference() throws {
        let client = try Client.create()
        let mlir = """
        module @batchnorm {
          func.func @main(%arg0: tensor<1x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>) -> (tensor<1x2x2x2xf32>) {
            %0 = stablehlo.batch_norm_inference %arg0, %scale, %offset, %mean, %variance epsilon = 0.00001, feature_index = 3 : (tensor<1x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
            return %0 : tensor<1x2x2x2xf32>
          }
        }
        """
        let executable = try client.compile(mlir)

        // Input: 1x2x2x2 (NHWC)
        let input = try client.createBuffer(
            [1, 2, 3, 4, 5, 6, 7, 8] as [Float],
            shape: [1, 2, 2, 2],
            elementType: .float32
        )

        // Scale (gamma): [1, 1]
        let scale = try client.createBuffer([1, 1] as [Float], shape: [2], elementType: .float32)

        // Offset (beta): [0, 0]
        let offset = try client.createBuffer([0, 0] as [Float], shape: [2], elementType: .float32)

        // Mean: [4, 5] (computed from odd/even indices)
        let mean = try client.createBuffer([4, 5] as [Float], shape: [2], elementType: .float32)

        // Variance: [1, 1]
        let variance = try client.createBuffer([1, 1] as [Float], shape: [2], elementType: .float32)

        let outputs = try executable.execute([input, scale, offset, mean, variance])
        let result = try outputs[0].toFloatArray()

        // Output should be (input - mean) / sqrt(variance + epsilon) * scale + offset
        #expect(result.count == 8)
        // First channel (indices 0,2,4,6): values 1,3,5,7 normalized with mean=4, var=1
        #expect(abs(result[0] - (-3.0)) < 0.01)  // (1-4)/1 = -3
        #expect(abs(result[2] - (-1.0)) < 0.01)  // (3-4)/1 = -1
    }
}

// MARK: - Sort Operations

@Suite("Quick Wins - Sort")
struct SortOpsTests {

    @Test("Sort 1D tensor")
    func sort1D() throws {
        let client = try Client.create()
        let mlir = """
        module @sort {
          func.func @main(%arg0: tensor<6xf32>) -> (tensor<6xf32>) {
            %0 = stablehlo.sort %arg0, dimension = 0 : tensor<6xf32>
            return %0 : tensor<6xf32>
          }
        }
        """
        let executable = try client.compile(mlir)

        let input = try client.createBuffer([3, 1, 4, 1, 5, 9] as [Float], shape: [6], elementType: .float32)
        let outputs = try executable.execute([input])
        let result = try outputs[0].toFloatArray()

        // Sorted in ascending order
        #expect(result == [1, 1, 3, 4, 5, 9])
    }

    @Test("Sort 2D tensor along axis")
    func sort2DAxis() throws {
        let client = try Client.create()
        let mlir = """
        module @sort2d {
          func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.sort %arg0, dimension = 1 : tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)

        // [[3, 1, 2], [6, 4, 5]]
        let input = try client.createBuffer([3, 1, 2, 6, 4, 5] as [Float], shape: [2, 3], elementType: .float32)
        let outputs = try executable.execute([input])
        let result = try outputs[0].toFloatArray()

        // Sorted along axis 1: [[1, 2, 3], [4, 5, 6]]
        #expect(result == [1, 2, 3, 4, 5, 6])
    }
}

// MARK: - Combined Tests

@Suite("Quick Wins - Combined Operations")
struct CombinedOpsTests {

    @Test("CNN forward pass: conv + pool + normalize")
    func cnnForwardPass() throws {
        let client = try Client.create()

        // Simplified CNN: conv2d -> relu -> maxpool
        let mlir = """
        module @cnn {
          func.func @main(%input: tensor<1x4x4x1xf32>, %kernel: tensor<3x3x1x2xf32>) -> (tensor<1x1x1x2xf32>) {
            %0 = stablehlo.convolution %input, %kernel window_strides = [1, 1], feature_group_count = 1 : (tensor<1x4x4x1xf32>, tensor<3x3x1x2xf32>) -> tensor<1x2x2x2xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<1x2x2x2xf32>
            %1 = stablehlo.maximum %0, %zero : tensor<1x2x2x2xf32>
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %2 = stablehlo.reduce_window %1, %init window_dimensions = [1, 2, 2, 1], window_strides = [1, 2, 2, 1], applies stablehlo.maximum : (tensor<1x2x2x2xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
            return %2 : tensor<1x1x1x2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // 4x4 input
        let input = try client.createBuffer(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] as [Float],
            shape: [1, 4, 4, 1],
            elementType: .float32
        )

        // 3x3 kernel with 2 output channels (first channel: all 1s, second: all 0.1s)
        let kernel = try client.createBuffer(
            [1, 0.1, 1, 0.1, 1, 0.1,
             1, 0.1, 1, 0.1, 1, 0.1,
             1, 0.1, 1, 0.1, 1, 0.1] as [Float],
            shape: [3, 3, 1, 2],
            elementType: .float32
        )

        let outputs = try executable.execute([input, kernel])
        let result = try outputs[0].toFloatArray()

        #expect(result.count == 2)  // 1x1x1x2 output
        // First channel: max of 4 summed values (should be large positive)
        // Second channel: max of 4 scaled values
        #expect(result[0] > 0)
        #expect(result[1] > 0)
    }
}
