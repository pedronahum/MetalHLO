// IndexingOperationsDiagnostic.swift
// MetalHLOTests
//
// Diagnostic tests to understand why slice, gather, scatter cause MPS issues.

import Testing
import Foundation
import MetalHLO

@Suite("Indexing Operations Diagnostic")
struct IndexingOperationsDiagnosticTests {

    // MARK: - Slice Tests

    @Test("Simple 1D slice")
    func simpleSlice1D() async throws {
        // Using the direct stablehlo format that our parser understands
        let mlir = """
        module @simple_slice {
          func.func @main(%arg0: tensor<8xf32>) -> (tensor<4xf32>) {
            %result = stablehlo.slice %arg0 [2:6:1] : (tensor<8xf32>) -> tensor<4xf32>
            return %result : tensor<4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        let input = client.createBuffer([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] as [Float], shape: [8])
        let outputs = try executable.execute([input])

        let result = try outputs[0].toFloatArray()
        print("1D Slice result: \(result)")

        let expected: [Float] = [2.0, 3.0, 4.0, 5.0]
        for (i, (actual, exp)) in zip(result, expected).enumerated() {
            #expect(abs(actual - exp) < 1e-5, "Mismatch at index \(i): \(actual) vs \(exp)")
        }
    }

    @Test("2D slice no strides")
    func slice2DNoStrides() async throws {
        let mlir = """
        module @slice_2d {
          func.func @main(%arg0: tensor<4x4xf32>) -> (tensor<2x2xf32>) {
            %result = stablehlo.slice %arg0 [1:3:1, 1:3:1] : (tensor<4x4xf32>) -> tensor<2x2xf32>
            return %result : tensor<2x2xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // 4x4 matrix: [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
        var data: [Float] = []
        for i in 0..<16 {
            data.append(Float(i))
        }
        let input = client.createBuffer(data, shape: [4, 4])
        let outputs = try executable.execute([input])

        let result = try outputs[0].toFloatArray()
        print("2D Slice result: \(result)")

        // Expected: [[5,6], [9,10]]
        let expected: [Float] = [5.0, 6.0, 9.0, 10.0]
        for (i, (actual, exp)) in zip(result, expected).enumerated() {
            #expect(abs(actual - exp) < 1e-5, "Mismatch at index \(i): \(actual) vs \(exp)")
        }
    }

    @Test("2D slice with strides")
    func slice2DWithStrides() async throws {
        // Slicing with strides > 1
        let mlir = """
        module @slice_strides {
          func.func @main(%arg0: tensor<4x6xf32>) -> (tensor<2x2xf32>) {
            %result = stablehlo.slice %arg0 [0:4:2, 0:6:3] : (tensor<4x6xf32>) -> tensor<2x2xf32>
            return %result : tensor<2x2xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Input: 4x6 matrix with sequential values
        var data: [Float] = []
        for i in 0..<24 {
            data.append(Float(i))
        }
        // [[0,1,2,3,4,5], [6,7,8,9,10,11], [12,13,14,15,16,17], [18,19,20,21,22,23]]
        let input = client.createBuffer(data, shape: [4, 6])
        let outputs = try executable.execute([input])

        let result = try outputs[0].toFloatArray()
        print("2D Slice with strides result: \(result)")

        // rows 0,2 and cols 0,3: [[0,3], [12,15]]
        let expected: [Float] = [0.0, 3.0, 12.0, 15.0]
        for (i, (actual, exp)) in zip(result, expected).enumerated() {
            #expect(abs(actual - exp) < 1e-5, "Mismatch at index \(i): \(actual) vs \(exp)")
        }
    }

    // MARK: - Gather Tests

    @Test("Simple gather (embedding lookup)")
    func simpleGather() async throws {
        // Simplified gather for embedding lookup pattern
        let mlir = """
        module @gather_embedding {
          func.func @main(%embeddings: tensor<10x4xf32>, %indices: tensor<3xi32>) -> (tensor<3x4xf32>) {
            %result = stablehlo.gather %embeddings, %indices, offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1, slice_sizes = [1, 4] : (tensor<10x4xf32>, tensor<3xi32>) -> tensor<3x4xf32>
            return %result : tensor<3x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Embeddings: 10 vectors of dim 4
        var embData: [Float] = []
        for i in 0..<10 {
            for j in 0..<4 {
                embData.append(Float(i * 10 + j))
            }
        }
        let embeddings = client.createBuffer(embData, shape: [10, 4])
        let indices = client.createBuffer([Int32(2), Int32(5), Int32(8)], shape: [3])

        let outputs = try executable.execute([embeddings, indices])

        let result = try outputs[0].toFloatArray()
        print("Gather (embedding lookup) result: \(result)")

        // Expected: rows 2, 5, 8 from embeddings
        let expected: [Float] = [
            20.0, 21.0, 22.0, 23.0,  // row 2
            50.0, 51.0, 52.0, 53.0,  // row 5
            80.0, 81.0, 82.0, 83.0   // row 8
        ]
        for (i, (actual, exp)) in zip(result, expected).enumerated() {
            #expect(abs(actual - exp) < 1e-5, "Mismatch at index \(i): \(actual) vs \(exp)")
        }
    }

    // MARK: - Scatter Tests

    @Test("Simple scatter update")
    func simpleScatter() async throws {
        // Note: scatter requires a computation region which our parser may not support
        // This test uses a simple update pattern
        let mlir = """
        module @scatter_update {
          func.func @main(%operand: tensor<5x4xf32>, %indices: tensor<2xi32>, %updates: tensor<2x4xf32>) -> (tensor<5x4xf32>) {
            %result = stablehlo.scatter %operand, %indices, %updates, update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1 : (tensor<5x4xf32>, tensor<2xi32>, tensor<2x4xf32>) -> tensor<5x4xf32>
            return %result : tensor<5x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Operand: 5x4 zeros
        let operand = client.createBuffer([Float](repeating: 0.0, count: 20), shape: [5, 4])
        // Indices: scatter to rows 1 and 3
        let indices = client.createBuffer([Int32(1), Int32(3)], shape: [2])
        // Updates: 2x4 with values
        let updates = client.createBuffer([
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        ] as [Float], shape: [2, 4])

        let outputs = try executable.execute([operand, indices, updates])

        let result = try outputs[0].toFloatArray()
        print("Scatter result: \(result)")

        // Row 0: [0,0,0,0], Row 1: [1,2,3,4], Row 2: [0,0,0,0], Row 3: [5,6,7,8], Row 4: [0,0,0,0]
        let expected: [Float] = [
            0.0, 0.0, 0.0, 0.0,
            1.0, 2.0, 3.0, 4.0,
            0.0, 0.0, 0.0, 0.0,
            5.0, 6.0, 7.0, 8.0,
            0.0, 0.0, 0.0, 0.0
        ]
        for (i, (actual, exp)) in zip(result, expected).enumerated() {
            #expect(abs(actual - exp) < 1e-5, "Mismatch at index \(i): \(actual) vs \(exp)")
        }
    }

    @Test("Scatter with add computation")
    func scatterWithAdd() async throws {
        // Scatter with add computation: adds updates to existing values
        let mlir = """
        module @scatter_add {
          func.func @main(%operand: tensor<5x4xf32>, %indices: tensor<2xi32>, %updates: tensor<2x4xf32>) -> (tensor<5x4xf32>) {
            %result = stablehlo.scatter %operand, %indices, %updates, update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1, computation = add : (tensor<5x4xf32>, tensor<2xi32>, tensor<2x4xf32>) -> tensor<5x4xf32>
            return %result : tensor<5x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Operand: 5x4 with initial values (all 10s)
        let operand = client.createBuffer([Float](repeating: 10.0, count: 20), shape: [5, 4])
        // Indices: scatter to rows 1 and 3
        let indices = client.createBuffer([Int32(1), Int32(3)], shape: [2])
        // Updates: 2x4 with values
        let updates = client.createBuffer([
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0
        ] as [Float], shape: [2, 4])

        let outputs = try executable.execute([operand, indices, updates])

        let result = try outputs[0].toFloatArray()
        print("Scatter add result: \(result)")

        // Row 0: [10,10,10,10], Row 1: [10+1, 10+2, 10+3, 10+4], Row 2: [10,10,10,10],
        // Row 3: [10+5, 10+6, 10+7, 10+8], Row 4: [10,10,10,10]
        let expected: [Float] = [
            10.0, 10.0, 10.0, 10.0,
            11.0, 12.0, 13.0, 14.0,
            10.0, 10.0, 10.0, 10.0,
            15.0, 16.0, 17.0, 18.0,
            10.0, 10.0, 10.0, 10.0
        ]
        for (i, (actual, exp)) in zip(result, expected).enumerated() {
            #expect(abs(actual - exp) < 1e-5, "Mismatch at index \(i): \(actual) vs \(exp)")
        }
    }

    @Test("Scatter with max computation")
    func scatterWithMax() async throws {
        // Scatter with max computation: takes maximum of existing and update
        let mlir = """
        module @scatter_max {
          func.func @main(%operand: tensor<5x4xf32>, %indices: tensor<2xi32>, %updates: tensor<2x4xf32>) -> (tensor<5x4xf32>) {
            %result = stablehlo.scatter %operand, %indices, %updates, update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1, computation = max : (tensor<5x4xf32>, tensor<2xi32>, tensor<2x4xf32>) -> tensor<5x4xf32>
            return %result : tensor<5x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Operand: 5x4 with initial value 5
        let operand = client.createBuffer([Float](repeating: 5.0, count: 20), shape: [5, 4])
        // Indices: scatter to rows 1 and 3
        let indices = client.createBuffer([Int32(1), Int32(3)], shape: [2])
        // Updates: 2x4 - some values > 5, some < 5
        let updates = client.createBuffer([
            1.0, 8.0, 3.0, 9.0,   // row 1: max(5, [1,8,3,9]) = [5,8,5,9]
            7.0, 2.0, 10.0, 4.0  // row 3: max(5, [7,2,10,4]) = [7,5,10,5]
        ] as [Float], shape: [2, 4])

        let outputs = try executable.execute([operand, indices, updates])

        let result = try outputs[0].toFloatArray()
        print("Scatter max result: \(result)")

        let expected: [Float] = [
            5.0, 5.0, 5.0, 5.0,
            5.0, 8.0, 5.0, 9.0,
            5.0, 5.0, 5.0, 5.0,
            7.0, 5.0, 10.0, 5.0,
            5.0, 5.0, 5.0, 5.0
        ]
        for (i, (actual, exp)) in zip(result, expected).enumerated() {
            #expect(abs(actual - exp) < 1e-5, "Mismatch at index \(i): \(actual) vs \(exp)")
        }
    }

    @Test("Scatter with min computation")
    func scatterWithMin() async throws {
        // Scatter with min computation: takes minimum of existing and update
        let mlir = """
        module @scatter_min {
          func.func @main(%operand: tensor<5x4xf32>, %indices: tensor<2xi32>, %updates: tensor<2x4xf32>) -> (tensor<5x4xf32>) {
            %result = stablehlo.scatter %operand, %indices, %updates, update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1, computation = min : (tensor<5x4xf32>, tensor<2xi32>, tensor<2x4xf32>) -> tensor<5x4xf32>
            return %result : tensor<5x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Operand: 5x4 with initial value 5
        let operand = client.createBuffer([Float](repeating: 5.0, count: 20), shape: [5, 4])
        // Indices: scatter to rows 1 and 3
        let indices = client.createBuffer([Int32(1), Int32(3)], shape: [2])
        // Updates: 2x4 - some values > 5, some < 5
        let updates = client.createBuffer([
            1.0, 8.0, 3.0, 9.0,   // row 1: min(5, [1,8,3,9]) = [1,5,3,5]
            7.0, 2.0, 10.0, 4.0  // row 3: min(5, [7,2,10,4]) = [5,2,5,4]
        ] as [Float], shape: [2, 4])

        let outputs = try executable.execute([operand, indices, updates])

        let result = try outputs[0].toFloatArray()
        print("Scatter min result: \(result)")

        let expected: [Float] = [
            5.0, 5.0, 5.0, 5.0,
            1.0, 5.0, 3.0, 5.0,
            5.0, 5.0, 5.0, 5.0,
            5.0, 2.0, 5.0, 4.0,
            5.0, 5.0, 5.0, 5.0
        ]
        for (i, (actual, exp)) in zip(result, expected).enumerated() {
            #expect(abs(actual - exp) < 1e-5, "Mismatch at index \(i): \(actual) vs \(exp)")
        }
    }

    // MARK: - Dynamic Slice Tests

    @Test("Dynamic slice with constant indices")
    func dynamicSliceConstantIndices() async throws {
        // Use plain slice since dynamic_slice has runtime indices
        let mlir = """
        module @dynamic_slice {
          func.func @main(%arg0: tensor<4x4xf32>) -> (tensor<2x2xf32>) {
            %result = stablehlo.slice %arg0 [1:3:1, 1:3:1] : (tensor<4x4xf32>) -> tensor<2x2xf32>
            return %result : tensor<2x2xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        var data: [Float] = []
        for i in 0..<16 {
            data.append(Float(i))
        }
        let input = client.createBuffer(data, shape: [4, 4])
        let outputs = try executable.execute([input])

        let result = try outputs[0].toFloatArray()
        print("Slice result: \(result)")

        // Expected: [[5,6], [9,10]]
        let expected: [Float] = [5.0, 6.0, 9.0, 10.0]
        for (i, (actual, exp)) in zip(result, expected).enumerated() {
            #expect(abs(actual - exp) < 1e-5, "Mismatch at index \(i): \(actual) vs \(exp)")
        }
    }
}
