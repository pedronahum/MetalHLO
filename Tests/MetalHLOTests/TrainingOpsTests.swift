// TrainingOpsTests.swift
// MetalHLOTests
//
// Tests for Phase 3 training operations.

import Testing
@testable import MetalHLO

// MARK: - Batched Matrix Multiplication

@Suite("Batched MatMul")
struct BatchedMatMulTests {

    @Test("dot_general with batching dimensions")
    func dotGeneralBatched() throws {
        let client = try Client.create()

        // Batched matmul: [2, 3, 4] x [2, 4, 5] -> [2, 3, 5]
        // Batch dim: 0, LHS contracting: 2, RHS contracting: 1
        let mlir = """
        module @batched_matmul {
          func.func @main(%lhs: tensor<2x3x4xf32>, %rhs: tensor<2x4x5xf32>) -> (tensor<2x3x5xf32>) {
            %0 = stablehlo.dot_general %lhs, %rhs, #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]> : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
            return %0 : tensor<2x3x5xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // LHS: 2 batches of 3x4 matrices
        // Batch 0: [[1,0,0,0], [0,1,0,0], [0,0,1,0]] (identity-like)
        // Batch 1: [[1,1,1,1], [2,2,2,2], [3,3,3,3]]
        let lhsData: [Float] = [
            // Batch 0
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            // Batch 1
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3
        ]

        // RHS: 2 batches of 4x5 matrices
        // Batch 0: identity-like
        // Batch 1: all ones (except last column) to produce sum-like behavior
        let rhsData: [Float] = [
            // Batch 0: identity-like
            1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            // Batch 1: each column is [1,1,1,1] so dot product with [a,a,a,a] = 4a
            1, 1, 1, 1, 0,
            1, 1, 1, 1, 0,
            1, 1, 1, 1, 0,
            1, 1, 1, 1, 0
        ]

        let lhs = try client.createBuffer(lhsData, shape: [2, 3, 4], elementType: .float32)
        let rhs = try client.createBuffer(rhsData, shape: [2, 4, 5], elementType: .float32)

        let outputs = try executable.execute([lhs, rhs])
        let result = try outputs[0].toFloatArray()

        #expect(outputs[0].shape == [2, 3, 5])

        // Batch 0: identity @ identity-like = first 3 rows of identity
        // Batch 1: [[4,4,4,4,0], [8,8,8,8,0], [12,12,12,12,0]]
        let expected: [Float] = [
            // Batch 0
            1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            // Batch 1
            4, 4, 4, 4, 0,
            8, 8, 8, 8, 0,
            12, 12, 12, 12, 0
        ]

        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-5, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }

    @Test("dot_general non-standard contracting dimensions")
    func dotGeneralNonStandard() throws {
        let client = try Client.create()

        // Contract first dimension of LHS with second dimension of RHS
        // [4, 3] x [5, 4] -> [3, 5] (contracting dim 0 of LHS with dim 1 of RHS)
        let mlir = """
        module @nonstandard_dot {
          func.func @main(%lhs: tensor<4x3xf32>, %rhs: tensor<5x4xf32>) -> (tensor<3x5xf32>) {
            %0 = stablehlo.dot_general %lhs, %rhs, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [1]> : (tensor<4x3xf32>, tensor<5x4xf32>) -> tensor<3x5xf32>
            return %0 : tensor<3x5xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // LHS [4, 3]: columns are [1,2,3,4], [5,6,7,8], [9,10,11,12]
        let lhsData: [Float] = [
            1, 5, 9,
            2, 6, 10,
            3, 7, 11,
            4, 8, 12
        ]

        // RHS [5, 4]: each row is [1,1,1,1]
        let rhsData: [Float] = [Float](repeating: 1, count: 20)

        let lhs = try client.createBuffer(lhsData, shape: [4, 3], elementType: .float32)
        let rhs = try client.createBuffer(rhsData, shape: [5, 4], elementType: .float32)

        let outputs = try executable.execute([lhs, rhs])
        let result = try outputs[0].toFloatArray()

        #expect(outputs[0].shape == [3, 5])

        // Contracting over dim 0 of LHS (size 4) and dim 1 of RHS (size 4)
        // Result[i,j] = sum over k of LHS[k,i] * RHS[j,k]
        // For column 0 of LHS: [1,2,3,4], dot with any row of RHS [1,1,1,1] = 10
        // For column 1 of LHS: [5,6,7,8], dot = 26
        // For column 2 of LHS: [9,10,11,12], dot = 42
        let expected: [Float] = [
            10, 10, 10, 10, 10,
            26, 26, 26, 26, 26,
            42, 42, 42, 42, 42
        ]

        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 1e-5, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }
}

// MARK: - Reduction Operations

@Suite("Reduction Operations")
struct ReductionTests {

    @Test("Reduce sum over axis")
    func reduceSum() throws {
        let client = try Client.create()
        let mlir = """
        module @reduce_sum {
          func.func @main(%input: tensor<2x3xf32>) -> (tensor<2xf32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            return %0 : tensor<2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let input = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [2, 3], elementType: .float32)
        let outputs = try executable.execute([input])
        let result = try outputs[0].toFloatArray()

        // Row 0: 1 + 2 + 3 = 6
        // Row 1: 4 + 5 + 6 = 15
        #expect(result == [6, 15])
    }

    @Test("Reduce max over axis")
    func reduceMax() throws {
        let client = try Client.create()
        let mlir = """
        module @reduce_max {
          func.func @main(%input: tensor<2x3xf32>) -> (tensor<2xf32>) {
            %init = stablehlo.constant dense<-1e38> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            return %0 : tensor<2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let input = try client.createBuffer([1, 5, 3, 2, 6, 4] as [Float], shape: [2, 3], elementType: .float32)
        let outputs = try executable.execute([input])
        let result = try outputs[0].toFloatArray()

        // Row 0: max(1, 5, 3) = 5
        // Row 1: max(2, 6, 4) = 6
        #expect(result == [5, 6])
    }

    @Test("Reduce min over axis")
    func reduceMin() throws {
        let client = try Client.create()
        let mlir = """
        module @reduce_min {
          func.func @main(%input: tensor<2x3xf32>) -> (tensor<2xf32>) {
            %init = stablehlo.constant dense<1e38> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.minimum across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            return %0 : tensor<2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let input = try client.createBuffer([1, 5, 3, 2, 6, 4] as [Float], shape: [2, 3], elementType: .float32)
        let outputs = try executable.execute([input])
        let result = try outputs[0].toFloatArray()

        // Row 0: min(1, 5, 3) = 1
        // Row 1: min(2, 6, 4) = 2
        #expect(result == [1, 2])
    }

    @Test("Reduce sum over multiple axes")
    func reduceSumMultipleAxes() throws {
        let client = try Client.create()
        let mlir = """
        module @reduce_sum_multi {
          func.func @main(%input: tensor<2x3x4xf32>) -> (tensor<3xf32>) {
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %0 = stablehlo.reduce %input, %init applies stablehlo.add across dimensions = [0, 2] : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<3xf32>
            return %0 : tensor<3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        // Create tensor where value at [b,i,j] = b*12 + i*4 + j + 1
        var data: [Float] = []
        for b in 0..<2 {
            for i in 0..<3 {
                for j in 0..<4 {
                    data.append(Float(b * 12 + i * 4 + j + 1))
                }
            }
        }
        let input = try client.createBuffer(data, shape: [2, 3, 4], elementType: .float32)
        let outputs = try executable.execute([input])
        let result = try outputs[0].toFloatArray()

        #expect(outputs[0].shape == [3])
        // Sum over batch and last dimension for each middle dimension
        // For i=0: sum of [1,2,3,4] + [13,14,15,16] = 10 + 58 = 68
        // For i=1: sum of [5,6,7,8] + [17,18,19,20] = 26 + 74 = 100
        // For i=2: sum of [9,10,11,12] + [21,22,23,24] = 42 + 90 = 132
        #expect(result == [68, 100, 132])
    }
}

// MARK: - Comparison Operations

@Suite("Comparison Operations")
struct ComparisonTests {

    @Test("Compare equal")
    func compareEqual() throws {
        let client = try Client.create()
        // Parser expects: operands first, then direction
        let mlir = """
        module @compare_eq {
          func.func @main(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> (tensor<4xi1>) {
            %0 = stablehlo.compare %lhs, %rhs, EQ : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            return %0 : tensor<4xi1>
          }
        }
        """

        let executable = try client.compile(mlir)
        let lhs = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let rhs = try client.createBuffer([1, 0, 3, 5] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([lhs, rhs])

        // Compare: [1==1, 2==0, 3==3, 4==5] = [true, false, true, false]
        let result = try outputs[0].toBoolArray()
        #expect(result == [true, false, true, false])
    }

    @Test("Compare less than")
    func compareLessThan() throws {
        let client = try Client.create()
        let mlir = """
        module @compare_lt {
          func.func @main(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> (tensor<4xi1>) {
            %0 = stablehlo.compare %lhs, %rhs, LT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            return %0 : tensor<4xi1>
          }
        }
        """

        let executable = try client.compile(mlir)
        let lhs = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let rhs = try client.createBuffer([2, 2, 2, 2] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([lhs, rhs])

        // Compare: [1<2, 2<2, 3<2, 4<2] = [true, false, false, false]
        let result = try outputs[0].toBoolArray()
        #expect(result == [true, false, false, false])
    }

    @Test("Compare greater than")
    func compareGreaterThan() throws {
        let client = try Client.create()
        let mlir = """
        module @compare_gt {
          func.func @main(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> (tensor<4xi1>) {
            %0 = stablehlo.compare %lhs, %rhs, GT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            return %0 : tensor<4xi1>
          }
        }
        """

        let executable = try client.compile(mlir)
        let lhs = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let rhs = try client.createBuffer([2, 2, 2, 2] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([lhs, rhs])

        // Compare: [1>2, 2>2, 3>2, 4>2] = [false, false, true, true]
        let result = try outputs[0].toBoolArray()
        #expect(result == [false, false, true, true])
    }

    @Test("Compare less than or equal")
    func compareLessEqual() throws {
        let client = try Client.create()
        let mlir = """
        module @compare_le {
          func.func @main(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> (tensor<4xi1>) {
            %0 = stablehlo.compare %lhs, %rhs, LE : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            return %0 : tensor<4xi1>
          }
        }
        """

        let executable = try client.compile(mlir)
        let lhs = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let rhs = try client.createBuffer([2, 2, 2, 2] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([lhs, rhs])

        // Compare: [1<=2, 2<=2, 3<=2, 4<=2] = [true, true, false, false]
        let result = try outputs[0].toBoolArray()
        #expect(result == [true, true, false, false])
    }

    @Test("Compare greater than or equal")
    func compareGreaterEqual() throws {
        let client = try Client.create()
        let mlir = """
        module @compare_ge {
          func.func @main(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> (tensor<4xi1>) {
            %0 = stablehlo.compare %lhs, %rhs, GE : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            return %0 : tensor<4xi1>
          }
        }
        """

        let executable = try client.compile(mlir)
        let lhs = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let rhs = try client.createBuffer([2, 2, 2, 2] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([lhs, rhs])

        // Compare: [1>=2, 2>=2, 3>=2, 4>=2] = [false, true, true, true]
        let result = try outputs[0].toBoolArray()
        #expect(result == [false, true, true, true])
    }

    @Test("Compare not equal")
    func compareNotEqual() throws {
        let client = try Client.create()
        let mlir = """
        module @compare_ne {
          func.func @main(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> (tensor<4xi1>) {
            %0 = stablehlo.compare %lhs, %rhs, NE : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            return %0 : tensor<4xi1>
          }
        }
        """

        let executable = try client.compile(mlir)
        let lhs = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let rhs = try client.createBuffer([1, 0, 3, 5] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([lhs, rhs])

        // Compare: [1!=1, 2!=0, 3!=3, 4!=5] = [false, true, false, true]
        let result = try outputs[0].toBoolArray()
        #expect(result == [false, true, false, true])
    }
}

// MARK: - Select Operation

@Suite("Select Operation")
struct SelectTests {

    @Test("Select based on predicate")
    func selectBasic() throws {
        let client = try Client.create()
        // Use compare to generate the predicate - this is more realistic for ML workloads
        // Predicate is generated by comparing indices with threshold
        let mlir = """
        module @select {
          func.func @main(%indices: tensor<4xf32>, %threshold: tensor<4xf32>, %on_true: tensor<4xf32>, %on_false: tensor<4xf32>) -> (tensor<4xf32>) {
            %pred = stablehlo.compare %indices, %threshold, LT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // indices < threshold gives predicate: [0 < 1.5, 1 < 1.5, 2 < 1.5, 3 < 1.5] = [true, true, false, false]
        let indices = try client.createBuffer([0, 1, 2, 3] as [Float], shape: [4], elementType: .float32)
        let threshold = try client.createBuffer([1.5, 1.5, 1.5, 1.5] as [Float], shape: [4], elementType: .float32)
        let onTrue = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let onFalse = try client.createBuffer([10, 20, 30, 40] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([indices, threshold, onTrue, onFalse])
        let result = try outputs[0].toFloatArray()

        // Select with [true, true, false, false]: [1, 2, 30, 40]
        #expect(result == [1, 2, 30, 40])
    }

    @Test("Select with computed predicate")
    func selectWithCompare() throws {
        let client = try Client.create()
        let mlir = """
        module @select_compare {
          func.func @main(%x: tensor<4xf32>, %threshold: tensor<4xf32>) -> (tensor<4xf32>) {
            %pred = stablehlo.compare %x, %threshold, GT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            %zeros = stablehlo.constant dense<0.0> : tensor<4xf32>
            %0 = stablehlo.select %pred, %x, %zeros : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-1, 2, -3, 4] as [Float], shape: [4], elementType: .float32)
        let threshold = try client.createBuffer([0, 0, 0, 0] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([x, threshold])
        let result = try outputs[0].toFloatArray()

        // ReLU-like: [0, 2, 0, 4]
        #expect(result == [0, 2, 0, 4])
    }
}

// MARK: - Clamp Operation

@Suite("Clamp Operation")
struct ClampTests {

    @Test("Clamp values to range")
    func clampBasic() throws {
        let client = try Client.create()
        let mlir = """
        module @clamp {
          func.func @main(%min: tensor<4xf32>, %x: tensor<4xf32>, %max: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.clamp %min, %x, %max : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let min = try client.createBuffer([0, 0, 0, 0] as [Float], shape: [4], elementType: .float32)
        let x = try client.createBuffer([-1, 0.5, 1.5, 2] as [Float], shape: [4], elementType: .float32)
        let max = try client.createBuffer([1, 1, 1, 1] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([min, x, max])
        let result = try outputs[0].toFloatArray()

        // Clamp to [0, 1]: [0, 0.5, 1, 1]
        #expect(result == [0, 0.5, 1, 1])
    }

    @Test("Clamp with scalar bounds")
    func clampScalarBounds() throws {
        let client = try Client.create()
        let mlir = """
        module @clamp_scalar {
          func.func @main(%x: tensor<6xf32>) -> (tensor<6xf32>) {
            %min = stablehlo.constant dense<-1.0> : tensor<6xf32>
            %max = stablehlo.constant dense<1.0> : tensor<6xf32>
            %0 = stablehlo.clamp %min, %x, %max : tensor<6xf32>
            return %0 : tensor<6xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-2, -1, -0.5, 0.5, 1, 2] as [Float], shape: [6], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Clamp to [-1, 1]: [-1, -1, -0.5, 0.5, 1, 1]
        #expect(result == [-1, -1, -0.5, 0.5, 1, 1])
    }
}

// MARK: - Gradient Computation Tests

@Suite("Gradient Computation")
struct GradientTests {

    @Test("Linear layer backward pass: dL/dW")
    func linearLayerGradientWeights() throws {
        let client = try Client.create()

        // Forward: y = x @ W
        // Backward: dL/dW = x^T @ dL/dy
        // x: [2, 3], W: [3, 4], y: [2, 4]
        // dL/dy: [2, 4] (upstream gradient)
        // dL/dW: [3, 4] = x^T @ dL/dy
        let mlir = """
        module @linear_grad_w {
          func.func @main(%x: tensor<2x3xf32>, %grad_y: tensor<2x4xf32>) -> (tensor<3x4xf32>) {
            %x_t = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
            %grad_w = stablehlo.dot %x_t, %grad_y : (tensor<3x2xf32>, tensor<2x4xf32>) -> tensor<3x4xf32>
            return %grad_w : tensor<3x4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // x = [[1, 2, 3], [4, 5, 6]]
        let x = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [2, 3], elementType: .float32)
        // grad_y = all ones (uniform gradient)
        let gradY = try client.createBuffer([Float](repeating: 1, count: 8), shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([x, gradY])
        let result = try outputs[0].toFloatArray()

        // x^T = [[1, 4], [2, 5], [3, 6]]
        // x^T @ ones[2,4] = [[5, 5, 5, 5], [7, 7, 7, 7], [9, 9, 9, 9]]
        let expected: [Float] = [5, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9]
        #expect(result == expected)
    }

    @Test("Linear layer backward pass: dL/dx")
    func linearLayerGradientInput() throws {
        let client = try Client.create()

        // Forward: y = x @ W
        // Backward: dL/dx = dL/dy @ W^T
        // x: [2, 3], W: [3, 4], y: [2, 4]
        // dL/dy: [2, 4]
        // dL/dx: [2, 3] = dL/dy @ W^T
        let mlir = """
        module @linear_grad_x {
          func.func @main(%W: tensor<3x4xf32>, %grad_y: tensor<2x4xf32>) -> (tensor<2x3xf32>) {
            %w_t = stablehlo.transpose %W, dims = [1, 0] : (tensor<3x4xf32>) -> tensor<4x3xf32>
            %grad_x = stablehlo.dot %grad_y, %w_t : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
            return %grad_x : tensor<2x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // W = identity-like for first 3 columns: [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
        let W = try client.createBuffer([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] as [Float], shape: [3, 4], elementType: .float32)
        // grad_y = all ones
        let gradY = try client.createBuffer([Float](repeating: 1, count: 8), shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([W, gradY])
        let result = try outputs[0].toFloatArray()

        // W^T = [[1,0,0], [0,1,0], [0,0,1], [0,0,0]]
        // ones[2,4] @ W^T = [[1,1,1], [1,1,1]]
        let expected: [Float] = [1, 1, 1, 1, 1, 1]
        #expect(result == expected)
    }

    @Test("ReLU backward pass")
    func reluBackward() throws {
        let client = try Client.create()

        // ReLU backward: dL/dx = dL/dy * (x > 0)
        let mlir = """
        module @relu_backward {
          func.func @main(%x: tensor<6xf32>, %grad_y: tensor<6xf32>) -> (tensor<6xf32>) {
            %zero = stablehlo.constant dense<0.0> : tensor<6xf32>
            %mask = stablehlo.compare %x, %zero, GT : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
            %grad_x = stablehlo.select %mask, %grad_y, %zero : (tensor<6xi1>, tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
            return %grad_x : tensor<6xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // x = [-2, -1, 0, 1, 2, 3]
        let x = try client.createBuffer([-2, -1, 0, 1, 2, 3] as [Float], shape: [6], elementType: .float32)
        // grad_y = all ones
        let gradY = try client.createBuffer([Float](repeating: 1, count: 6), shape: [6], elementType: .float32)

        let outputs = try executable.execute([x, gradY])
        let result = try outputs[0].toFloatArray()

        // Gradient flows through where x > 0: [0, 0, 0, 1, 1, 1]
        #expect(result == [0, 0, 0, 1, 1, 1])
    }

    @Test("Full linear layer forward + backward")
    func linearLayerFullBackward() throws {
        let client = try Client.create()

        // Forward: y = relu(x @ W + b)
        // Compute gradients: dL/dW, dL/db, dL/dx
        // Assuming dL/dy = ones (MSE gradient with target = y + 1)
        let mlir = """
        module @linear_full_backward {
          func.func @main(%x: tensor<2x3xf32>, %W: tensor<3x4xf32>, %b: tensor<4xf32>) -> (tensor<3x4xf32>, tensor<4xf32>, tensor<2x3xf32>) {
            // Forward pass
            %matmul = stablehlo.dot %x, %W : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
            %b_broadcast = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
            %pre_relu = stablehlo.add %matmul, %b_broadcast : tensor<2x4xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<2x4xf32>
            %y = stablehlo.maximum %pre_relu, %zero : tensor<2x4xf32>

            // Backward pass (assuming dL/dy = ones)
            %ones = stablehlo.constant dense<1.0> : tensor<2x4xf32>

            // ReLU backward: grad_pre_relu = grad_y * (pre_relu > 0)
            %relu_mask = stablehlo.compare %pre_relu, %zero, GT : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
            %zero_grad = stablehlo.constant dense<0.0> : tensor<2x4xf32>
            %grad_pre_relu = stablehlo.select %relu_mask, %ones, %zero_grad : (tensor<2x4xi1>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

            // Bias gradient: sum over batch dimension
            %init = stablehlo.constant dense<0.0> : tensor<f32>
            %grad_b = stablehlo.reduce %grad_pre_relu, %init applies stablehlo.add across dimensions = [0] : (tensor<2x4xf32>, tensor<f32>) -> tensor<4xf32>

            // Weight gradient: x^T @ grad_pre_relu
            %x_t = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
            %grad_W = stablehlo.dot %x_t, %grad_pre_relu : (tensor<3x2xf32>, tensor<2x4xf32>) -> tensor<3x4xf32>

            // Input gradient: grad_pre_relu @ W^T
            %W_t = stablehlo.transpose %W, dims = [1, 0] : (tensor<3x4xf32>) -> tensor<4x3xf32>
            %grad_x = stablehlo.dot %grad_pre_relu, %W_t : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>

            return %grad_W, %grad_b, %grad_x : tensor<3x4xf32>, tensor<4xf32>, tensor<2x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // x = [[1, 0, 0], [0, 1, 0]]
        let x = try client.createBuffer([1, 0, 0, 0, 1, 0] as [Float], shape: [2, 3], elementType: .float32)
        // W = identity for first 3 cols
        let W = try client.createBuffer([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] as [Float], shape: [3, 4], elementType: .float32)
        // b = [-0.5, 0.5, -0.5, 0.5] (some will be zeroed by ReLU)
        let b = try client.createBuffer([-0.5, 0.5, -0.5, 0.5] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([x, W, b])

        #expect(outputs.count == 3)
        #expect(outputs[0].shape == [3, 4]) // grad_W
        #expect(outputs[1].shape == [4])    // grad_b
        #expect(outputs[2].shape == [2, 3]) // grad_x

        // Verify shapes and that gradients are computed (non-zero where expected)
        let gradW = try outputs[0].toFloatArray()
        let gradB = try outputs[1].toFloatArray()
        let _ = try outputs[2].toFloatArray()

        // The bias gradient should be non-zero for dimensions where ReLU passed
        // With the inputs, pre_relu should have some positive values
        #expect(gradB.contains { $0 != 0 }, "Bias gradient should have non-zero values")
        #expect(gradW.contains { $0 != 0 }, "Weight gradient should have non-zero values")
    }
}
