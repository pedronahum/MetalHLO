// ActivationOpsTests.swift
// MetalHLOTests
//
// Tests for Phase 4 activation functions.
// All activations are composed from StableHLO primitives.

import Testing
import Foundation
@testable import MetalHLO

// MARK: - ReLU Activation

@Suite("ReLU Activation")
struct ReLUTests {

    @Test("ReLU: max(x, 0)")
    func reluBasic() throws {
        let client = try Client.create()
        let mlir = """
        module @relu {
          func.func @main(%x: tensor<6xf32>) -> (tensor<6xf32>) {
            %zero = stablehlo.constant dense<0.0> : tensor<6xf32>
            %0 = stablehlo.maximum %x, %zero : tensor<6xf32>
            return %0 : tensor<6xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-2, -1, 0, 1, 2, 3] as [Float], shape: [6], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // ReLU: [0, 0, 0, 1, 2, 3]
        #expect(result == [0, 0, 0, 1, 2, 3])
    }

    @Test("ReLU with 2D tensor")
    func relu2D() throws {
        let client = try Client.create()
        let mlir = """
        module @relu_2d {
          func.func @main(%x: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            %zero = stablehlo.constant dense<0.0> : tensor<2x3xf32>
            %0 = stablehlo.maximum %x, %zero : tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-1, 2, -3, 4, -5, 6] as [Float], shape: [2, 3], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        #expect(result == [0, 2, 0, 4, 0, 6])
    }
}

// MARK: - Sigmoid Activation

@Suite("Sigmoid Activation")
struct SigmoidTests {

    @Test("Sigmoid: 1 / (1 + exp(-x))")
    func sigmoidBasic() throws {
        let client = try Client.create()
        // sigmoid(x) = 1 / (1 + exp(-x))
        let mlir = """
        module @sigmoid {
          func.func @main(%x: tensor<5xf32>) -> (tensor<5xf32>) {
            %neg_x = stablehlo.negate %x : tensor<5xf32>
            %exp_neg_x = stablehlo.exponential %neg_x : tensor<5xf32>
            %one = stablehlo.constant dense<1.0> : tensor<5xf32>
            %one_plus_exp = stablehlo.add %one, %exp_neg_x : tensor<5xf32>
            %result = stablehlo.divide %one, %one_plus_exp : tensor<5xf32>
            return %result : tensor<5xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-2, -1, 0, 1, 2] as [Float], shape: [5], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Reference: sigmoid(x) = 1 / (1 + exp(-x))
        // sigmoid(-2) ≈ 0.1192, sigmoid(-1) ≈ 0.2689, sigmoid(0) = 0.5
        // sigmoid(1) ≈ 0.7311, sigmoid(2) ≈ 0.8808
        let expected: [Float] = [-2, -1, 0, 1, 2].map { 1.0 / (1.0 + exp(-$0)) }

        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-5, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }

    @Test("Sigmoid at extremes")
    func sigmoidExtremes() throws {
        let client = try Client.create()
        let mlir = """
        module @sigmoid_extreme {
          func.func @main(%x: tensor<4xf32>) -> (tensor<4xf32>) {
            %neg_x = stablehlo.negate %x : tensor<4xf32>
            %exp_neg_x = stablehlo.exponential %neg_x : tensor<4xf32>
            %one = stablehlo.constant dense<1.0> : tensor<4xf32>
            %one_plus_exp = stablehlo.add %one, %exp_neg_x : tensor<4xf32>
            %result = stablehlo.divide %one, %one_plus_exp : tensor<4xf32>
            return %result : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        // Test extreme values
        let x = try client.createBuffer([-10, -5, 5, 10] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // sigmoid(-10) ≈ 0, sigmoid(-5) ≈ 0.0067
        // sigmoid(5) ≈ 0.9933, sigmoid(10) ≈ 1
        #expect(result[0] < 0.001, "sigmoid(-10) should be close to 0")
        #expect(result[1] < 0.01, "sigmoid(-5) should be small")
        #expect(result[2] > 0.99, "sigmoid(5) should be close to 1")
        #expect(result[3] > 0.999, "sigmoid(10) should be very close to 1")
    }
}

// MARK: - Softmax Activation

@Suite("Softmax Activation")
struct SoftmaxTests {

    @Test("Softmax over last axis")
    func softmaxBasic() throws {
        let client = try Client.create()
        // Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
        let mlir = """
        module @softmax {
          func.func @main(%x: tensor<4xf32>) -> (tensor<4xf32>) {
            // Find max for numerical stability
            %init_max = stablehlo.constant dense<-1e38> : tensor<f32>
            %max_val = stablehlo.reduce %x, %init_max applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>

            // Broadcast max back to original shape
            %max_broadcast = stablehlo.broadcast_in_dim %max_val, dims = [] : (tensor<f32>) -> tensor<4xf32>

            // Subtract max for numerical stability
            %x_shifted = stablehlo.subtract %x, %max_broadcast : tensor<4xf32>

            // Compute exp
            %exp_x = stablehlo.exponential %x_shifted : tensor<4xf32>

            // Sum of exp
            %init_sum = stablehlo.constant dense<0.0> : tensor<f32>
            %sum_exp = stablehlo.reduce %exp_x, %init_sum applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>

            // Broadcast sum back
            %sum_broadcast = stablehlo.broadcast_in_dim %sum_exp, dims = [] : (tensor<f32>) -> tensor<4xf32>

            // Divide
            %result = stablehlo.divide %exp_x, %sum_broadcast : tensor<4xf32>
            return %result : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Reference softmax calculation
        let maxVal = Float(4)
        let shifted = [1, 2, 3, 4].map { Float($0) - maxVal }
        let exps = shifted.map { exp($0) }
        let sumExp = exps.reduce(0, +)
        let expected = exps.map { $0 / sumExp }

        // Check values sum to 1
        let sum = result.reduce(0, +)
        #expect(abs(sum - 1.0) < 1e-5, "Softmax should sum to 1, got \(sum)")

        // Check individual values
        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-5, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }

    @Test("Softmax 2D over last axis")
    func softmax2D() throws {
        let client = try Client.create()
        // Softmax over axis 1 (last axis) for a 2D tensor
        let mlir = """
        module @softmax_2d {
          func.func @main(%x: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            // Max over axis 1
            %init_max = stablehlo.constant dense<-1e38> : tensor<f32>
            %max_val = stablehlo.reduce %x, %init_max applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>

            // Broadcast max back
            %max_broadcast = stablehlo.broadcast_in_dim %max_val, dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>

            // Subtract max
            %x_shifted = stablehlo.subtract %x, %max_broadcast : tensor<2x3xf32>

            // Exp
            %exp_x = stablehlo.exponential %x_shifted : tensor<2x3xf32>

            // Sum over axis 1
            %init_sum = stablehlo.constant dense<0.0> : tensor<f32>
            %sum_exp = stablehlo.reduce %exp_x, %init_sum applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>

            // Broadcast sum back
            %sum_broadcast = stablehlo.broadcast_in_dim %sum_exp, dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>

            // Divide
            %result = stablehlo.divide %exp_x, %sum_broadcast : tensor<2x3xf32>
            return %result : tensor<2x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [2, 3], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Each row should sum to 1
        let row0Sum = result[0] + result[1] + result[2]
        let row1Sum = result[3] + result[4] + result[5]

        #expect(abs(row0Sum - 1.0) < 1e-5, "Row 0 should sum to 1, got \(row0Sum)")
        #expect(abs(row1Sum - 1.0) < 1e-5, "Row 1 should sum to 1, got \(row1Sum)")

        // Values should be monotonically increasing within each row
        #expect(result[0] < result[1] && result[1] < result[2], "Row 0 values should increase")
        #expect(result[3] < result[4] && result[4] < result[5], "Row 1 values should increase")
    }
}

// MARK: - GELU Activation

@Suite("GELU Activation")
struct GELUTests {

    @Test("GELU approximation using tanh")
    func geluTanh() throws {
        let client = try Client.create()
        // GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        // sqrt(2/π) ≈ 0.7978845608
        let mlir = """
        module @gelu {
          func.func @main(%x: tensor<6xf32>) -> (tensor<6xf32>) {
            // Constants
            %half = stablehlo.constant dense<0.5> : tensor<6xf32>
            %one = stablehlo.constant dense<1.0> : tensor<6xf32>
            %coeff = stablehlo.constant dense<0.044715> : tensor<6xf32>
            %sqrt_2_pi = stablehlo.constant dense<0.7978845608> : tensor<6xf32>

            // x³
            %x_squared = stablehlo.multiply %x, %x : tensor<6xf32>
            %x_cubed = stablehlo.multiply %x_squared, %x : tensor<6xf32>

            // 0.044715 * x³
            %coeff_x_cubed = stablehlo.multiply %coeff, %x_cubed : tensor<6xf32>

            // x + 0.044715 * x³
            %inner = stablehlo.add %x, %coeff_x_cubed : tensor<6xf32>

            // sqrt(2/π) * (x + 0.044715 * x³)
            %scaled = stablehlo.multiply %sqrt_2_pi, %inner : tensor<6xf32>

            // tanh(...)
            %tanh_val = stablehlo.tanh %scaled : tensor<6xf32>

            // 1 + tanh(...)
            %one_plus_tanh = stablehlo.add %one, %tanh_val : tensor<6xf32>

            // 0.5 * x
            %half_x = stablehlo.multiply %half, %x : tensor<6xf32>

            // 0.5 * x * (1 + tanh(...))
            %result = stablehlo.multiply %half_x, %one_plus_tanh : tensor<6xf32>
            return %result : tensor<6xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-2, -1, 0, 1, 2, 3] as [Float], shape: [6], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Reference GELU values (tanh approximation)
        func gelu(_ x: Float) -> Float {
            let sqrt2Pi: Float = 0.7978845608
            let coeff: Float = 0.044715
            return 0.5 * x * (1 + tanh(sqrt2Pi * (x + coeff * x * x * x)))
        }
        let expected = [-2, -1, 0, 1, 2, 3].map { gelu(Float($0)) }

        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-4, "Mismatch at index \(i): \(r) vs \(e)")
        }

        // GELU(0) should be 0
        #expect(abs(result[2]) < 1e-6, "GELU(0) should be 0")

        // GELU(x) ≈ x for large positive x
        #expect(result[5] > 2.9, "GELU(3) should be close to 3")
    }
}

// MARK: - Leaky ReLU Activation

@Suite("Leaky ReLU Activation")
struct LeakyReLUTests {

    @Test("Leaky ReLU with alpha=0.01")
    func leakyReluBasic() throws {
        let client = try Client.create()
        // leaky_relu(x, α) = x if x > 0, else α*x
        // Using select: select(x > 0, x, α*x)
        let mlir = """
        module @leaky_relu {
          func.func @main(%x: tensor<6xf32>) -> (tensor<6xf32>) {
            %zero = stablehlo.constant dense<0.0> : tensor<6xf32>
            %alpha = stablehlo.constant dense<0.01> : tensor<6xf32>

            // Compute α*x
            %alpha_x = stablehlo.multiply %alpha, %x : tensor<6xf32>

            // Compare x > 0
            %mask = stablehlo.compare %x, %zero, GT : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>

            // Select: if x > 0 then x, else α*x
            %result = stablehlo.select %mask, %x, %alpha_x : (tensor<6xi1>, tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
            return %result : tensor<6xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-2, -1, 0, 1, 2, 3] as [Float], shape: [6], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Expected: [-0.02, -0.01, 0, 1, 2, 3]
        let expected: [Float] = [-0.02, -0.01, 0, 1, 2, 3]

        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-5, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }

    @Test("Leaky ReLU with alpha=0.2")
    func leakyReluAlpha02() throws {
        let client = try Client.create()
        let mlir = """
        module @leaky_relu_02 {
          func.func @main(%x: tensor<4xf32>) -> (tensor<4xf32>) {
            %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
            %alpha = stablehlo.constant dense<0.2> : tensor<4xf32>

            %alpha_x = stablehlo.multiply %alpha, %x : tensor<4xf32>
            %mask = stablehlo.compare %x, %zero, GT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            %result = stablehlo.select %mask, %x, %alpha_x : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
            return %result : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-10, -5, 5, 10] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Expected: [-2, -1, 5, 10]
        let expected: [Float] = [-2, -1, 5, 10]

        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-5, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }
}

// MARK: - ELU Activation

@Suite("ELU Activation")
struct ELUTests {

    @Test("ELU with alpha=1.0")
    func eluBasic() throws {
        let client = try Client.create()
        // elu(x, α) = x if x > 0, else α * (exp(x) - 1)
        let mlir = """
        module @elu {
          func.func @main(%x: tensor<6xf32>) -> (tensor<6xf32>) {
            %zero = stablehlo.constant dense<0.0> : tensor<6xf32>
            %one = stablehlo.constant dense<1.0> : tensor<6xf32>
            %alpha = stablehlo.constant dense<1.0> : tensor<6xf32>

            // exp(x)
            %exp_x = stablehlo.exponential %x : tensor<6xf32>

            // exp(x) - 1
            %exp_minus_1 = stablehlo.subtract %exp_x, %one : tensor<6xf32>

            // α * (exp(x) - 1)
            %alpha_branch = stablehlo.multiply %alpha, %exp_minus_1 : tensor<6xf32>

            // Compare x > 0
            %mask = stablehlo.compare %x, %zero, GT : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>

            // Select
            %result = stablehlo.select %mask, %x, %alpha_branch : (tensor<6xi1>, tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
            return %result : tensor<6xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-2, -1, 0, 1, 2, 3] as [Float], shape: [6], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Reference ELU values
        func elu(_ x: Float, alpha: Float = 1.0) -> Float {
            return x > 0 ? x : alpha * (exp(x) - 1)
        }
        let expected = [-2, -1, 0, 1, 2, 3].map { elu(Float($0)) }

        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-5, "Mismatch at index \(i): \(r) vs \(e)")
        }

        // ELU approaches -α as x approaches -∞
        #expect(result[0] > -1.0, "ELU(-2) should be > -1")
        #expect(result[0] < 0, "ELU(-2) should be negative")
    }

    @Test("ELU with alpha=0.5")
    func eluAlpha05() throws {
        let client = try Client.create()
        let mlir = """
        module @elu_05 {
          func.func @main(%x: tensor<4xf32>) -> (tensor<4xf32>) {
            %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
            %one = stablehlo.constant dense<1.0> : tensor<4xf32>
            %alpha = stablehlo.constant dense<0.5> : tensor<4xf32>

            %exp_x = stablehlo.exponential %x : tensor<4xf32>
            %exp_minus_1 = stablehlo.subtract %exp_x, %one : tensor<4xf32>
            %alpha_branch = stablehlo.multiply %alpha, %exp_minus_1 : tensor<4xf32>
            %mask = stablehlo.compare %x, %zero, GT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            %result = stablehlo.select %mask, %x, %alpha_branch : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
            return %result : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-3, -1, 1, 3] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // With α=0.5, negative values approach -0.5 instead of -1
        func elu(_ x: Float, alpha: Float = 0.5) -> Float {
            return x > 0 ? x : alpha * (exp(x) - 1)
        }
        let expected = [-3, -1, 1, 3].map { elu(Float($0)) }

        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-5, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }
}

// MARK: - SiLU (Swish) Activation

@Suite("SiLU Activation")
struct SiLUTests {

    @Test("SiLU: x * sigmoid(x)")
    func siluBasic() throws {
        let client = try Client.create()
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let mlir = """
        module @silu {
          func.func @main(%x: tensor<6xf32>) -> (tensor<6xf32>) {
            // Compute sigmoid(x) = 1 / (1 + exp(-x))
            %neg_x = stablehlo.negate %x : tensor<6xf32>
            %exp_neg_x = stablehlo.exponential %neg_x : tensor<6xf32>
            %one = stablehlo.constant dense<1.0> : tensor<6xf32>
            %one_plus_exp = stablehlo.add %one, %exp_neg_x : tensor<6xf32>
            %sigmoid = stablehlo.divide %one, %one_plus_exp : tensor<6xf32>

            // Compute x * sigmoid(x)
            %result = stablehlo.multiply %x, %sigmoid : tensor<6xf32>
            return %result : tensor<6xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-2, -1, 0, 1, 2, 3] as [Float], shape: [6], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Reference SiLU values
        func silu(_ x: Float) -> Float {
            return x / (1 + exp(-x))
        }
        let expected = [-2, -1, 0, 1, 2, 3].map { silu(Float($0)) }

        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-5, "Mismatch at index \(i): \(r) vs \(e)")
        }

        // SiLU(0) = 0
        #expect(abs(result[2]) < 1e-6, "SiLU(0) should be 0")

        // SiLU is smooth and bounded below
        #expect(result[0] < 0, "SiLU(-2) should be negative")
        #expect(result[0] > -0.5, "SiLU(-2) should be > -0.5")
    }

    @Test("SiLU approaches x for large positive x")
    func siluLargePositive() throws {
        let client = try Client.create()
        let mlir = """
        module @silu_large {
          func.func @main(%x: tensor<3xf32>) -> (tensor<3xf32>) {
            %neg_x = stablehlo.negate %x : tensor<3xf32>
            %exp_neg_x = stablehlo.exponential %neg_x : tensor<3xf32>
            %one = stablehlo.constant dense<1.0> : tensor<3xf32>
            %one_plus_exp = stablehlo.add %one, %exp_neg_x : tensor<3xf32>
            %sigmoid = stablehlo.divide %one, %one_plus_exp : tensor<3xf32>
            %result = stablehlo.multiply %x, %sigmoid : tensor<3xf32>
            return %result : tensor<3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([5, 10, 15] as [Float], shape: [3], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // For large positive x, SiLU(x) ≈ x
        #expect(abs(result[0] - 5) < 0.05, "SiLU(5) should be close to 5")
        #expect(abs(result[1] - 10) < 0.001, "SiLU(10) should be very close to 10")
        #expect(abs(result[2] - 15) < 0.0001, "SiLU(15) should be extremely close to 15")
    }
}

// MARK: - Activation Gradient Tests

@Suite("Activation Gradients")
struct ActivationGradientTests {

    @Test("Sigmoid backward pass")
    func sigmoidBackward() throws {
        let client = try Client.create()
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        // For backward: grad_x = grad_y * sigmoid(x) * (1 - sigmoid(x))
        let mlir = """
        module @sigmoid_backward {
          func.func @main(%x: tensor<4xf32>, %grad_y: tensor<4xf32>) -> (tensor<4xf32>) {
            // Forward: compute sigmoid
            %neg_x = stablehlo.negate %x : tensor<4xf32>
            %exp_neg_x = stablehlo.exponential %neg_x : tensor<4xf32>
            %one = stablehlo.constant dense<1.0> : tensor<4xf32>
            %one_plus_exp = stablehlo.add %one, %exp_neg_x : tensor<4xf32>
            %sigmoid = stablehlo.divide %one, %one_plus_exp : tensor<4xf32>

            // Backward: grad_x = grad_y * sigmoid * (1 - sigmoid)
            %one_minus_sigmoid = stablehlo.subtract %one, %sigmoid : tensor<4xf32>
            %sigmoid_deriv = stablehlo.multiply %sigmoid, %one_minus_sigmoid : tensor<4xf32>
            %grad_x = stablehlo.multiply %grad_y, %sigmoid_deriv : tensor<4xf32>
            return %grad_x : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-1, 0, 1, 2] as [Float], shape: [4], elementType: .float32)
        let gradY = try client.createBuffer([1, 1, 1, 1] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([x, gradY])
        let result = try outputs[0].toFloatArray()

        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        // Maximum at x=0 where derivative = 0.25
        func sigmoidDeriv(_ x: Float) -> Float {
            let s = 1.0 / (1.0 + exp(-x))
            return s * (1 - s)
        }
        let expected = [-1, 0, 1, 2].map { sigmoidDeriv(Float($0)) }

        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-5, "Mismatch at index \(i): \(r) vs \(e)")
        }

        // Maximum gradient at x=0
        #expect(result[1] > result[0], "Gradient at x=0 should be larger than at x=-1")
        #expect(result[1] > result[2], "Gradient at x=0 should be larger than at x=1")
    }
}

// MARK: - Combined Activation Tests

@Suite("Combined Activations")
struct CombinedActivationTests {

    @Test("Two-layer MLP with different activations")
    func mlpWithActivations() throws {
        let client = try Client.create()
        // Layer 1: Linear + ReLU
        // Layer 2: Linear + Sigmoid
        let mlir = """
        module @mlp_activations {
          func.func @main(%x: tensor<2x4xf32>, %W1: tensor<4x3xf32>, %W2: tensor<3x2xf32>) -> (tensor<2x2xf32>) {
            // Layer 1: x @ W1 followed by ReLU
            %hidden = stablehlo.dot %x, %W1 : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<2x3xf32>
            %relu_out = stablehlo.maximum %hidden, %zero : tensor<2x3xf32>

            // Layer 2: relu_out @ W2 followed by Sigmoid
            %logits = stablehlo.dot %relu_out, %W2 : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
            %neg_logits = stablehlo.negate %logits : tensor<2x2xf32>
            %exp_neg = stablehlo.exponential %neg_logits : tensor<2x2xf32>
            %one = stablehlo.constant dense<1.0> : tensor<2x2xf32>
            %one_plus_exp = stablehlo.add %one, %exp_neg : tensor<2x2xf32>
            %output = stablehlo.divide %one, %one_plus_exp : tensor<2x2xf32>

            return %output : tensor<2x2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let x = try client.createBuffer([1, 2, 3, 4, 5, 6, 7, 8] as [Float], shape: [2, 4], elementType: .float32)
        let W1 = try client.createBuffer([Float](repeating: 0.1, count: 12), shape: [4, 3], elementType: .float32)
        let W2 = try client.createBuffer([Float](repeating: 0.1, count: 6), shape: [3, 2], elementType: .float32)

        let outputs = try executable.execute([x, W1, W2])
        let result = try outputs[0].toFloatArray()

        #expect(outputs[0].shape == [2, 2])

        // Output should be in (0, 1) due to sigmoid
        for value in result {
            #expect(value > 0 && value < 1, "Sigmoid output should be in (0, 1), got \(value)")
        }
    }
}
