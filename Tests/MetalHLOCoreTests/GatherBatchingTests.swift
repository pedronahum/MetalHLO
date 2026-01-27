// GatherBatchingTests.swift
// MetalHLOCoreTests
//
// Tests for gather and scatter operations with batching dimension support.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Gather/Scatter Batching Tests")
struct GatherBatchingTests {

    // MARK: - Gather Tests

    @Test("Gather without batching dimensions")
    func testGatherNoBatching() async throws {
        let mlir = """
        module @gather_test {
          func.func @main(%operand: tensor<3x4xf32>, %indices: tensor<2xi32>) -> (tensor<2x4xf32>) {
            %result = stablehlo.gather %operand, %indices, offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1, slice_sizes = [1, 4] : (tensor<3x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>
            return %result : tensor<2x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Create test data: 3x4 matrix
        var operandData: [Float] = []
        for i in 0..<3 {
            for j in 0..<4 {
                operandData.append(Float(i * 4 + j))
            }
        }
        // Indices: select rows 0 and 2
        let indicesData: [Int32] = [0, 2]

        let operandBuffer = try client.createBuffer(operandData, shape: [3, 4], elementType: .float32)
        let indicesBuffer = try client.createBuffer(indicesData, shape: [2], elementType: .int32)

        let outputs = try executable.execute([operandBuffer, indicesBuffer])
        let result = try outputs[0].toFloatArray()

        // Expected: rows 0 and 2 of the matrix
        // Row 0: [0, 1, 2, 3]
        // Row 2: [8, 9, 10, 11]
        #expect(result.count == 8)
        #expect(result[0] == 0)
        #expect(result[1] == 1)
        #expect(result[4] == 8)
        #expect(result[5] == 9)
    }

    @Test("Gather with leading batch dimensions")
    func testGatherWithLeadingBatchDims() async throws {
        // Batch size 2, gather from 3x4 per batch
        let mlir = """
        module @gather_batch_test {
          func.func @main(%operand: tensor<2x3x4xf32>, %indices: tensor<2x1xi32>) -> (tensor<2x4xf32>) {
            %result = stablehlo.gather %operand, %indices, offset_dims = [1], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1, slice_sizes = [1, 1, 4] : (tensor<2x3x4xf32>, tensor<2x1xi32>) -> tensor<2x4xf32>
            return %result : tensor<2x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Create test data: 2 batches of 3x4 matrices
        var operandData: [Float] = []
        for b in 0..<2 {
            for i in 0..<3 {
                for j in 0..<4 {
                    operandData.append(Float(b * 100 + i * 4 + j))
                }
            }
        }

        // Indices: select row 1 from batch 0, row 2 from batch 1
        let indicesData: [Int32] = [1, 2]

        let operandBuffer = try client.createBuffer(operandData, shape: [2, 3, 4], elementType: .float32)
        let indicesBuffer = try client.createBuffer(indicesData, shape: [2, 1], elementType: .int32)

        let outputs = try executable.execute([operandBuffer, indicesBuffer])
        let result = try outputs[0].toFloatArray()

        // Expected:
        // Batch 0, row 1: [4, 5, 6, 7]
        // Batch 1, row 2: [108, 109, 110, 111]
        #expect(result.count == 8)
        #expect(result[0] == 4)
        #expect(result[4] == 108)
    }

    // MARK: - Scatter Tests

    @Test("Scatter without batching - set mode")
    func testScatterNoBatchingSet() async throws {
        let mlir = """
        module @scatter_test {
          func.func @main(%operand: tensor<4x4xf32>, %indices: tensor<2xi32>, %updates: tensor<2x4xf32>) -> (tensor<4x4xf32>) {
            %result = stablehlo.scatter %operand, %indices, %updates, update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1 : (tensor<4x4xf32>, tensor<2xi32>, tensor<2x4xf32>) -> tensor<4x4xf32>
            return %result : tensor<4x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Create zero-initialized operand
        let operandData = [Float](repeating: 0, count: 16)

        // Indices: update rows 1 and 3
        let indicesData: [Int32] = [1, 3]

        // Updates: new values for rows 1 and 3
        var updatesData: [Float] = []
        for i in 0..<2 {
            for j in 0..<4 {
                updatesData.append(Float((i + 1) * 10 + j))
            }
        }

        let operandBuffer = try client.createBuffer(operandData, shape: [4, 4], elementType: .float32)
        let indicesBuffer = try client.createBuffer(indicesData, shape: [2], elementType: .int32)
        let updatesBuffer = try client.createBuffer(updatesData, shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([operandBuffer, indicesBuffer, updatesBuffer])
        let result = try outputs[0].toFloatArray()

        // Row 0 should be zeros
        #expect(result[0] == 0)
        #expect(result[1] == 0)

        // Row 1 should be [10, 11, 12, 13]
        #expect(result[4] == 10)
        #expect(result[5] == 11)

        // Row 2 should be zeros
        #expect(result[8] == 0)

        // Row 3 should be [20, 21, 22, 23]
        #expect(result[12] == 20)
        #expect(result[13] == 21)
    }

    @Test("Scatter with add computation")
    func testScatterWithAdd() async throws {
        let mlir = """
        module @scatter_add_test {
          func.func @main(%operand: tensor<3x4xf32>, %indices: tensor<2xi32>, %updates: tensor<2x4xf32>) -> (tensor<3x4xf32>) {
            %result = stablehlo.scatter %operand, %indices, %updates, update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1, computation = add : (tensor<3x4xf32>, tensor<2xi32>, tensor<2x4xf32>) -> tensor<3x4xf32>
            return %result : tensor<3x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Create operand with all 1s
        let operandData = [Float](repeating: 1, count: 12)

        // Indices: update rows 0 and 2
        let indicesData: [Int32] = [0, 2]

        // Updates: add these values
        let updatesData: [Float] = [
            10, 10, 10, 10,  // Add to row 0
            20, 20, 20, 20   // Add to row 2
        ]

        let operandBuffer = try client.createBuffer(operandData, shape: [3, 4], elementType: .float32)
        let indicesBuffer = try client.createBuffer(indicesData, shape: [2], elementType: .int32)
        let updatesBuffer = try client.createBuffer(updatesData, shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([operandBuffer, indicesBuffer, updatesBuffer])
        let result = try outputs[0].toFloatArray()

        // Row 0: 1 + 10 = 11
        #expect(result[0] == 11)

        // Row 1: unchanged = 1
        #expect(result[4] == 1)

        // Row 2: 1 + 20 = 21
        #expect(result[8] == 21)
    }

    // MARK: - Non-Leading Batch Dimension Tests

    @Test("Gather with non-leading batch dimensions")
    func testGatherWithNonLeadingBatchDims() async throws {
        // Operand shape: [3, 2, 4] with batch dim at position 1
        // - dim 0: 3 rows (spatial)
        // - dim 1: 2 batches
        // - dim 2: 4 columns (feature)
        // After transpose to move batch front: [2, 3, 4]
        let mlir = """
        module @gather_nonleading_batch {
          func.func @main(%operand: tensor<3x2x4xf32>, %indices: tensor<2x1xi32>) -> (tensor<2x4xf32>) {
            %result = stablehlo.gather %operand, %indices, offset_dims = [1], collapsed_slice_dims = [0], operand_batching_dims = [1], start_indices_batching_dims = [0], start_index_map = [0], index_vector_dim = 1, slice_sizes = [1, 1, 4] : (tensor<3x2x4xf32>, tensor<2x1xi32>) -> tensor<2x4xf32>
            return %result : tensor<2x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Create test data: [3, 2, 4] tensor
        // Layout: operand[row][batch][col]
        var operandData: [Float] = []
        for row in 0..<3 {
            for batch in 0..<2 {
                for col in 0..<4 {
                    // Value encodes position: row*100 + batch*10 + col
                    operandData.append(Float(row * 100 + batch * 10 + col))
                }
            }
        }

        // Indices: [2, 1] - select row 0 for batch 0, row 2 for batch 1
        let indicesData: [Int32] = [0, 2]

        let operandBuffer = try client.createBuffer(operandData, shape: [3, 2, 4], elementType: .float32)
        let indicesBuffer = try client.createBuffer(indicesData, shape: [2, 1], elementType: .int32)

        let outputs = try executable.execute([operandBuffer, indicesBuffer])
        let result = try outputs[0].toFloatArray()

        // Expected output: [2, 4]
        // Batch 0: operand[0][0][:] = [0*100 + 0*10 + {0,1,2,3}] = [0, 1, 2, 3]
        // Batch 1: operand[2][1][:] = [2*100 + 1*10 + {0,1,2,3}] = [210, 211, 212, 213]
        #expect(result.count == 8)
        #expect(result[0] == 0)
        #expect(result[1] == 1)
        #expect(result[2] == 2)
        #expect(result[3] == 3)
        #expect(result[4] == 210)
        #expect(result[5] == 211)
        #expect(result[6] == 212)
        #expect(result[7] == 213)
    }

    @Test("Scatter with non-leading batch dimensions")
    func testScatterWithNonLeadingBatchDims() async throws {
        // Operand shape: [3, 2, 4] with batch dim at position 1
        // Updates shape: [2, 4] with batch dim at position 0
        let mlir = """
        module @scatter_nonleading_batch {
          func.func @main(%operand: tensor<3x2x4xf32>, %indices: tensor<2x1xi32>, %updates: tensor<2x4xf32>) -> (tensor<3x2x4xf32>) {
            %result = stablehlo.scatter %operand, %indices, %updates, update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1, input_batching_dims = [1], scatter_indices_batching_dims = [0] : (tensor<3x2x4xf32>, tensor<2x1xi32>, tensor<2x4xf32>) -> tensor<3x2x4xf32>
            return %result : tensor<3x2x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Create zero-initialized operand [3, 2, 4]
        let operandData = [Float](repeating: 0, count: 24)

        // Indices: scatter to row 1 for batch 0, row 2 for batch 1
        let indicesData: [Int32] = [1, 2]

        // Updates: [2, 4] - what to scatter
        var updatesData: [Float] = []
        for batch in 0..<2 {
            for col in 0..<4 {
                updatesData.append(Float(batch * 10 + col))
            }
        }

        let operandBuffer = try client.createBuffer(operandData, shape: [3, 2, 4], elementType: .float32)
        let indicesBuffer = try client.createBuffer(indicesData, shape: [2, 1], elementType: .int32)
        let updatesBuffer = try client.createBuffer(updatesData, shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([operandBuffer, indicesBuffer, updatesBuffer])
        let result = try outputs[0].toFloatArray()

        // Expected: operand[1][0][:] = [0, 1, 2, 3], operand[2][1][:] = [10, 11, 12, 13]
        // Flattened layout: [row][batch][col]
        // operand[1][0][:] starts at index 1*8 + 0*4 = 8
        // operand[2][1][:] starts at index 2*8 + 1*4 = 20
        #expect(result.count == 24)

        // Row 1, batch 0 should have [0, 1, 2, 3]
        #expect(result[8] == 0)
        #expect(result[9] == 1)
        #expect(result[10] == 2)
        #expect(result[11] == 3)

        // Row 2, batch 1 should have [10, 11, 12, 13]
        #expect(result[20] == 10)
        #expect(result[21] == 11)
        #expect(result[22] == 12)
        #expect(result[23] == 13)

        // Other positions should remain 0
        #expect(result[0] == 0)  // row 0, batch 0
        #expect(result[4] == 0)  // row 0, batch 1
    }

    // MARK: - Permutation Integration Tests

    @Test("Permutation utilities integration")
    func testPermutationIntegration() {
        // Test that our permutation utilities work correctly for gather/scatter scenarios

        // Scenario: operand has shape [A, B, C, D] with batch dims at [2]
        // We want to move batch dims to front: [C, A, B, D]
        let operandBatchDims = [2]
        let operandRank = 4

        let operandPerm = PermutationUtils.buildPermutationMovingToFront(
            dims: operandBatchDims,
            rank: operandRank
        )
        #expect(operandPerm == [2, 0, 1, 3])

        // After permutation:
        // - Original dim 0 (A) is now at position 1
        // - Original dim 1 (B) is now at position 2
        // - Original dim 2 (C) is now at position 0
        // - Original dim 3 (D) is now at position 3

        // Slice sizes [sA, sB, sC, sD] become [sC, sA, sB, sD]
        let sliceSizes = [1, 2, 3, 4]
        let adjustedSliceSizes = PermutationUtils.applyPermutation(operandPerm, to: sliceSizes)
        #expect(adjustedSliceSizes == [3, 1, 2, 4])

        // collapsed_slice_dims [0, 1] in original become [1, 2] in transposed
        let collapsedDims = [0, 1]
        let operandInverse = PermutationUtils.invertPermutation(operandPerm)
        let adjustedCollapsedDims = collapsedDims.map { operandInverse[$0] }.sorted()
        #expect(adjustedCollapsedDims == [1, 2])

        // Batch dims are now at front
        #expect(PermutationUtils.areDimsContiguousAtFront([0], rank: operandRank))
    }
}
