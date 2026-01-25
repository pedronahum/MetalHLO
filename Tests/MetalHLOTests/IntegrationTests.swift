// IntegrationTests.swift
// MetalHLOTests
//
// End-to-end integration tests for MetalHLO.

import Testing
@testable import MetalHLO

@Suite("Integration Tests")
struct IntegrationTests {

    // MARK: - Basic Operations

    @Test("Addition operation")
    func addition() throws {
        let client = try Client.create()
        let mlir = """
        module @add {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let a = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [2, 3], elementType: .float32)
        let b = try client.createBuffer([1, 1, 1, 1, 1, 1] as [Float], shape: [2, 3], elementType: .float32)

        let outputs = try executable.execute([a, b])

        #expect(outputs.count == 1)
        let result = try outputs[0].toFloatArray()
        #expect(result == [2, 3, 4, 5, 6, 7])
    }

    @Test("Multiplication operation")
    func multiplication() throws {
        let client = try Client.create()
        let mlir = """
        module @mul {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let a = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([2, 2, 2, 2] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([a, b])

        let result = try outputs[0].toFloatArray()
        #expect(result == [2, 4, 6, 8])
    }

    // MARK: - Constants

    @Test("Constant operation")
    func constant() throws {
        let client = try Client.create()
        let mlir = """
        module @constant {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %cst = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
            %0 = stablehlo.add %arg0, %cst : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([0, 0, 0, 0] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        #expect(result == [1, 2, 3, 4])
    }

    @Test("Constant integer operation")
    func constantInteger() throws {
        let client = try Client.create()
        let mlir = """
        module @constant_int {
          func.func @main(%arg0: tensor<4xi32>) -> (tensor<4xi32>) {
            %cst = stablehlo.constant dense<[10, 20, 30, 40]> : tensor<4xi32>
            %0 = stablehlo.add %arg0, %cst : tensor<4xi32>
            return %0 : tensor<4xi32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 2, 3, 4] as [Int32], shape: [4], elementType: .int32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toInt32Array()
        #expect(result == [11, 22, 33, 44])
    }

    // MARK: - Unary Operations

    @Test("Negate operation")
    func negate() throws {
        let client = try Client.create()
        let mlir = """
        module @negate {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.negate %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, -2, 3, -4] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])

        let result = try outputs[0].toFloatArray()
        #expect(result == [-1, 2, -3, 4])
    }

    @Test("Abs operation")
    func abs() throws {
        let client = try Client.create()
        let mlir = """
        module @abs {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.abs %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, -2, 3, -4] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])

        let result = try outputs[0].toFloatArray()
        #expect(result == [1, 2, 3, 4])
    }

    // MARK: - Chained Operations

    @Test("Chained operations")
    func chainedOperations() throws {
        let client = try Client.create()
        let mlir = """
        module @chain {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            %1 = stablehlo.multiply %0, %arg0 : tensor<4xf32>
            return %1 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let a = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([1, 1, 1, 1] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([a, b])

        let result = try outputs[0].toFloatArray()
        // (a + b) * a = [2, 3, 4, 5] * [1, 2, 3, 4] = [2, 6, 12, 20]
        #expect(result == [2, 6, 12, 20])
    }

    // MARK: - Matrix Operations

    @Test("Reshape operation")
    func reshape() throws {
        let client = try Client.create()
        let mlir = """
        module @reshape {
          func.func @main(%arg0: tensor<6xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.reshape %arg0 : (tensor<6xf32>) -> tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [6], elementType: .float32)
        let outputs = try executable.execute([a])

        #expect(outputs[0].shape == [2, 3])
        let result = try outputs[0].toFloatArray()
        #expect(result == [1, 2, 3, 4, 5, 6])
    }

    // MARK: - Phase 2 Success Criteria

    @Test("Linear layer with ReLU: y = relu(x @ W + b)")
    func linearLayerRelu() throws {
        let client = try Client.create()

        // Linear layer: y = relu(x @ W + b)
        // x: [2, 4] (batch=2, features=4)
        // W: [4, 3] (input=4, output=3)
        // b: [3] (output bias)
        // result: [2, 3]
        let mlir = """
        module @linear_relu {
          func.func @main(%x: tensor<2x4xf32>, %W: tensor<4x3xf32>, %b: tensor<3xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.dot %x, %W : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
            %1 = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
            %2 = stablehlo.add %0, %1 : tensor<2x3xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<2x3xf32>
            %3 = stablehlo.maximum %2, %zero : tensor<2x3xf32>
            return %3 : tensor<2x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Input: x = [[1,2,3,4], [5,6,7,8]]
        let x = try client.createBuffer([1,2,3,4,5,6,7,8] as [Float], shape: [2, 4], elementType: .float32)

        // Weights: W extracts first 3 features
        // [[1,0,0], [0,1,0], [0,0,1], [0,0,0]]
        let W = try client.createBuffer([1,0,0, 0,1,0, 0,0,1, 0,0,0] as [Float], shape: [4, 3], elementType: .float32)

        // Bias: b = [-5, -5, -5] (to test relu with negative values)
        let b = try client.createBuffer([-5, -5, -5] as [Float], shape: [3], elementType: .float32)

        let outputs = try executable.execute([x, W, b])
        let result = try outputs[0].toFloatArray()

        // x @ W = [[1,2,3], [5,6,7]]
        // + b = [[-4,-3,-2], [0,1,2]]
        // relu = [[0,0,0], [0,1,2]]
        #expect(result == [0, 0, 0, 0, 1, 2])
    }

    // MARK: - Executable Properties

    @Test("Executable properties")
    func executableProperties() throws {
        let client = try Client.create()
        let mlir = """
        module @test {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        #expect(executable.inputCount == 2)
        #expect(executable.outputCount == 1)
        #expect(executable.inputTypes[0].shape == [2, 3])
        #expect(executable.inputTypes[0].elementType == .float32)
        #expect(executable.outputTypes[0].shape == [2, 3])
    }

    // MARK: - Timing

    @Test("Execution timing")
    func executionTiming() throws {
        let client = try Client.create()
        let mlir = """
        module @timing {
          func.func @main(%arg0: tensor<100x100xf32>, %arg1: tensor<100x100xf32>) -> (tensor<100x100xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<100x100xf32>
            return %0 : tensor<100x100xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let data = [Float](repeating: 1.0, count: 10000)
        let a = try client.createBuffer(data, shape: [100, 100], elementType: .float32)
        let b = try client.createBuffer(data, shape: [100, 100], elementType: .float32)

        let (outputs, timing) = try executable.executeWithTiming([a, b])

        #expect(outputs.count == 1)
        #expect(timing.totalTime > 0)
        #expect(timing.totalTime >= timing.encodeTime)
    }

    // MARK: - Error Handling

    @Test("Input count mismatch throws error")
    func inputCountMismatch() throws {
        let client = try Client.create()
        let mlir = """
        module @test {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)

        // Only provide one input when two are required
        #expect(throws: MetalHLOError.self) {
            try executable.execute([a])
        }
    }

    @Test("Parse error on invalid MLIR")
    func parseError() throws {
        let client = try Client.create()
        let invalidMLIR = "this is not valid MLIR"

        #expect(throws: MetalHLOError.self) {
            try client.compile(invalidMLIR)
        }
    }

    // MARK: - Caching

    @Test("Same module name with different content compiles correctly")
    func cacheKeyIncludesContent() throws {
        let client = try Client.create()

        // First module: addition
        let mlir1 = """
        module @test {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        // Second module: same name but multiplication
        let mlir2 = """
        module @test {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let exe1 = try client.compile(mlir1)
        let exe2 = try client.compile(mlir2)

        let a = try client.createBuffer([2, 3, 4, 5] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([2, 2, 2, 2] as [Float], shape: [4], elementType: .float32)

        let result1 = try exe1.execute([a, b])
        let result2 = try exe2.execute([a, b])

        // exe1 should add: [4, 5, 6, 7]
        #expect(try result1[0].toFloatArray() == [4, 5, 6, 7])

        // exe2 should multiply: [4, 6, 8, 10] - NOT the same as exe1
        #expect(try result2[0].toFloatArray() == [4, 6, 8, 10])
    }
}
