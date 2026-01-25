// OperationTests.swift
// MetalHLOTests
//
// Comprehensive tests for Phase 2 operations.

import Testing
@testable import MetalHLO

// MARK: - Binary Operations

@Suite("Binary Operations")
struct BinaryOperationTests {

    @Test("Subtract operation")
    func subtract() throws {
        let client = try Client.create()
        let mlir = """
        module @subtract {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([10, 20, 30, 40] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()
        #expect(result == [9, 18, 27, 36])
    }

    @Test("Divide operation")
    func divide() throws {
        let client = try Client.create()
        let mlir = """
        module @divide {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.divide %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([10, 20, 30, 40] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([2, 4, 5, 8] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()
        #expect(result == [5, 5, 6, 5])
    }

    @Test("Maximum operation")
    func maximum() throws {
        let client = try Client.create()
        let mlir = """
        module @maximum {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.maximum %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 5, 3, 8] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([2, 4, 6, 7] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()
        #expect(result == [2, 5, 6, 8])
    }

    @Test("Minimum operation")
    func minimum() throws {
        let client = try Client.create()
        let mlir = """
        module @minimum {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.minimum %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 5, 3, 8] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([2, 4, 6, 7] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()
        #expect(result == [1, 4, 3, 7])
    }

    @Test("Power operation")
    func power() throws {
        let client = try Client.create()
        let mlir = """
        module @power {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.power %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([2, 3, 4, 5] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([2, 2, 2, 2] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()
        #expect(result == [4, 9, 16, 25])
    }
}

// MARK: - Unary Operations

@Suite("Unary Operations")
struct UnaryOperationTests {

    @Test("Exponential operation")
    func exponential() throws {
        let client = try Client.create()
        let mlir = """
        module @exp {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.exponential %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([0, 1, 2, -1] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // e^0 = 1, e^1 ≈ 2.718, e^2 ≈ 7.389, e^-1 ≈ 0.368
        #expect(abs(result[0] - 1.0) < 0.001)
        #expect(abs(result[1] - 2.718) < 0.01)
        #expect(abs(result[2] - 7.389) < 0.01)
        #expect(abs(result[3] - 0.368) < 0.01)
    }

    @Test("Log operation")
    func log() throws {
        let client = try Client.create()
        let mlir = """
        module @log {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.log %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 2.718281828, 10, 100] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // ln(1) = 0, ln(e) ≈ 1, ln(10) ≈ 2.303, ln(100) ≈ 4.605
        #expect(abs(result[0] - 0.0) < 0.001)
        #expect(abs(result[1] - 1.0) < 0.01)
        #expect(abs(result[2] - 2.303) < 0.01)
        #expect(abs(result[3] - 4.605) < 0.01)
    }

    @Test("Sqrt operation")
    func sqrt() throws {
        let client = try Client.create()
        let mlir = """
        module @sqrt {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.sqrt %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 4, 9, 16] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        #expect(result == [1, 2, 3, 4])
    }

    @Test("Rsqrt operation")
    func rsqrt() throws {
        let client = try Client.create()
        let mlir = """
        module @rsqrt {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.rsqrt %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 4, 16, 25] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // 1/sqrt(1)=1, 1/sqrt(4)=0.5, 1/sqrt(16)=0.25, 1/sqrt(25)=0.2
        #expect(abs(result[0] - 1.0) < 0.001)
        #expect(abs(result[1] - 0.5) < 0.001)
        #expect(abs(result[2] - 0.25) < 0.001)
        #expect(abs(result[3] - 0.2) < 0.001)
    }

    @Test("Sine operation")
    func sine() throws {
        let client = try Client.create()
        let mlir = """
        module @sine {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.sine %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let pi = Float.pi
        let a = try client.createBuffer([0, pi/2, pi, 3*pi/2] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // sin(0)=0, sin(π/2)=1, sin(π)=0, sin(3π/2)=-1
        #expect(abs(result[0] - 0.0) < 0.001)
        #expect(abs(result[1] - 1.0) < 0.001)
        #expect(abs(result[2] - 0.0) < 0.001)
        #expect(abs(result[3] - (-1.0)) < 0.001)
    }

    @Test("Cosine operation")
    func cosine() throws {
        let client = try Client.create()
        let mlir = """
        module @cosine {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.cosine %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let pi = Float.pi
        let a = try client.createBuffer([0, pi/2, pi, 2*pi] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // cos(0)=1, cos(π/2)=0, cos(π)=-1, cos(2π)=1
        #expect(abs(result[0] - 1.0) < 0.001)
        #expect(abs(result[1] - 0.0) < 0.001)
        #expect(abs(result[2] - (-1.0)) < 0.001)
        #expect(abs(result[3] - 1.0) < 0.001)
    }

    @Test("Tanh operation")
    func tanh() throws {
        let client = try Client.create()
        let mlir = """
        module @tanh {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.tanh %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([0, 1, -1, 2] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // tanh(0)=0, tanh(1)≈0.762, tanh(-1)≈-0.762, tanh(2)≈0.964
        #expect(abs(result[0] - 0.0) < 0.001)
        #expect(abs(result[1] - 0.762) < 0.01)
        #expect(abs(result[2] - (-0.762)) < 0.01)
        #expect(abs(result[3] - 0.964) < 0.01)
    }

    @Test("Floor operation")
    func floor() throws {
        let client = try Client.create()
        let mlir = """
        module @floor {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.floor %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1.5, 2.9, -1.5, -2.9] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        #expect(result == [1, 2, -2, -3])
    }

    @Test("Ceil operation")
    func ceil() throws {
        let client = try Client.create()
        let mlir = """
        module @ceil {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.ceil %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1.1, 2.9, -1.1, -2.9] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        #expect(result == [2, 3, -1, -2])
    }

    @Test("Sign operation")
    func sign() throws {
        let client = try Client.create()
        let mlir = """
        module @sign {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.sign %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([5, -3, 0, -0.5] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        #expect(result == [1, -1, 0, -1])
    }
}

// MARK: - Matrix Operations

@Suite("Matrix Operations")
struct MatrixOperationTests {

    @Test("Dot product (matrix multiplication)")
    func dot() throws {
        let client = try Client.create()
        let mlir = """
        module @dot {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> (tensor<2x2xf32>) {
            %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        // A = [[1,2,3], [4,5,6]]
        let a = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [2, 3], elementType: .float32)
        // B = [[1,2], [3,4], [5,6]]
        let b = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [3, 2], elementType: .float32)
        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()
        // [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
        // = [[22, 28], [49, 64]]
        #expect(result == [22, 28, 49, 64])
    }

    @Test("Transpose operation")
    func transpose() throws {
        let client = try Client.create()
        let mlir = """
        module @transpose {
          func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<3x2xf32>) {
            %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
            return %0 : tensor<3x2xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        // A = [[1,2,3], [4,5,6]]
        let a = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [2, 3], elementType: .float32)
        let outputs = try executable.execute([a])
        #expect(outputs[0].shape == [3, 2])
        let result = try outputs[0].toFloatArray()
        // Transposed = [[1,4], [2,5], [3,6]]
        #expect(result == [1, 4, 2, 5, 3, 6])
    }

    @Test("Broadcast in dimension")
    func broadcastInDim() throws {
        let client = try Client.create()
        let mlir = """
        module @broadcast {
          func.func @main(%arg0: tensor<3xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 2, 3] as [Float], shape: [3], elementType: .float32)
        let outputs = try executable.execute([a])
        #expect(outputs[0].shape == [2, 3])
        let result = try outputs[0].toFloatArray()
        // Broadcast [1,2,3] to [[1,2,3], [1,2,3]]
        #expect(result == [1, 2, 3, 1, 2, 3])
    }
}

// MARK: - Type Conversion

@Suite("Type Conversion")
struct TypeConversionTests {

    @Test("Convert float32 to int32")
    func convertFloat32ToInt32() throws {
        let client = try Client.create()
        let mlir = """
        module @convert {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xi32>) {
            %0 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xi32>
            return %0 : tensor<4xi32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1.5, 2.9, -1.5, 0.0] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toInt32Array()
        // Truncates toward zero
        #expect(result == [1, 2, -1, 0])
    }

    @Test("Convert int32 to float32")
    func convertInt32ToFloat32() throws {
        let client = try Client.create()
        let mlir = """
        module @convert {
          func.func @main(%arg0: tensor<4xi32>) -> (tensor<4xf32>) {
            %0 = stablehlo.convert %arg0 : (tensor<4xi32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 2, -3, 0] as [Int32], shape: [4], elementType: .int32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        #expect(result == [1.0, 2.0, -3.0, 0.0])
    }
}

// MARK: - Linear Layer End-to-End Test

@Suite("End-to-End Tests")
struct EndToEndTests {

    @Test("Linear layer: y = relu(x @ W + b)")
    func linearLayerWithRelu() throws {
        let client = try Client.create()

        // Linear layer: y = relu(x @ W + b)
        // x: [2, 3] (batch=2, features=3)
        // W: [3, 4] (in=3, out=4)
        // b: [4] (bias)
        // Output: [2, 4]
        let mlir = """
        module @linear_relu {
          func.func @main(%x: tensor<2x3xf32>, %W: tensor<3x4xf32>, %b: tensor<4xf32>) -> (tensor<2x4xf32>) {
            %0 = stablehlo.dot %x, %W : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
            %1 = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
            %2 = stablehlo.add %0, %1 : tensor<2x4xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<2x4xf32>
            %3 = stablehlo.maximum %2, %zero : tensor<2x4xf32>
            return %3 : tensor<2x4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // x = [[1,2,3], [4,5,6]]
        let x = try client.createBuffer(
            [1, 2, 3, 4, 5, 6] as [Float],
            shape: [2, 3],
            elementType: .float32
        )

        // W = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
        // Identity-like to make verification easy
        let W = try client.createBuffer(
            [1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0] as [Float],
            shape: [3, 4],
            elementType: .float32
        )

        // b = [-2, -3, -4, 10]
        let b = try client.createBuffer(
            [-2, -3, -4, 10] as [Float],
            shape: [4],
            elementType: .float32
        )

        let outputs = try executable.execute([x, W, b])
        let result = try outputs[0].toFloatArray()

        // x @ W = [[1,2,3,0], [4,5,6,0]]
        // + b   = [[-1,-1,-1,10], [2,2,2,10]]
        // relu  = [[0,0,0,10], [2,2,2,10]]
        #expect(result == [0, 0, 0, 10, 2, 2, 2, 10])
    }

    @Test("Two-layer MLP forward pass")
    func twoLayerMLP() throws {
        let client = try Client.create()

        // MLP: y = relu(relu(x @ W1 + b1) @ W2 + b2)
        let mlir = """
        module @mlp {
          func.func @main(%x: tensor<1x2xf32>, %W1: tensor<2x3xf32>, %b1: tensor<3xf32>, %W2: tensor<3x2xf32>, %b2: tensor<2xf32>) -> (tensor<1x2xf32>) {
            %0 = stablehlo.dot %x, %W1 : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
            %1 = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<3xf32>) -> tensor<1x3xf32>
            %2 = stablehlo.add %0, %1 : tensor<1x3xf32>
            %zero1 = stablehlo.constant dense<0.0> : tensor<1x3xf32>
            %3 = stablehlo.maximum %2, %zero1 : tensor<1x3xf32>
            %4 = stablehlo.dot %3, %W2 : (tensor<1x3xf32>, tensor<3x2xf32>) -> tensor<1x2xf32>
            %5 = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
            %6 = stablehlo.add %4, %5 : tensor<1x2xf32>
            %zero2 = stablehlo.constant dense<0.0> : tensor<1x2xf32>
            %7 = stablehlo.maximum %6, %zero2 : tensor<1x2xf32>
            return %7 : tensor<1x2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let x = try client.createBuffer([1, 2] as [Float], shape: [1, 2], elementType: .float32)
        // W1 identity-ish
        let W1 = try client.createBuffer([1, 0, 0, 0, 1, 0] as [Float], shape: [2, 3], elementType: .float32)
        let b1 = try client.createBuffer([0, 0, 1] as [Float], shape: [3], elementType: .float32)
        // W2
        let W2 = try client.createBuffer([1, 0, 0, 1, 1, 1] as [Float], shape: [3, 2], elementType: .float32)
        let b2 = try client.createBuffer([0, 0] as [Float], shape: [2], elementType: .float32)

        let outputs = try executable.execute([x, W1, b1, W2, b2])
        let result = try outputs[0].toFloatArray()

        // Layer 1:
        // x @ W1 = [[1,2,0]]
        // + b1 = [[1,2,1]]
        // relu = [[1,2,1]]
        //
        // Layer 2:
        // [[1,2,1]] @ W2 = [[1*1+2*0+1*1, 1*0+2*1+1*1]] = [[2, 3]]
        // + b2 = [[2, 3]]
        // relu = [[2, 3]]
        #expect(result == [2, 3])
    }
}
