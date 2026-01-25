// IndexingOpsTests.swift
// MetalHLOTests
//
// Tests for Phase 5 indexing and memory operations.
// Covers slice, pad, concatenate, and gather (embedding lookup).

import Testing
import Foundation
@testable import MetalHLO

// MARK: - Slice Operations

@Suite("Slice Operations")
struct SliceTests {

    @Test("Slice: basic 1D slice")
    func sliceBasic1D() throws {
        let client = try Client.create()
        let mlir = """
        module @slice_1d {
          func.func @main(%x: tensor<6xf32>) -> (tensor<3xf32>) {
            %0 = stablehlo.slice %x [2:5:1] : (tensor<6xf32>) -> tensor<3xf32>
            return %0 : tensor<3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([0, 1, 2, 3, 4, 5] as [Float], shape: [6], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Elements at indices 2, 3, 4 -> [2, 3, 4]
        #expect(result == [2, 3, 4])
    }

    @Test("Slice: 2D slice extracting submatrix")
    func slice2DSubmatrix() throws {
        let client = try Client.create()
        let mlir = """
        module @slice_2d {
          func.func @main(%x: tensor<4x4xf32>) -> (tensor<2x2xf32>) {
            %0 = stablehlo.slice %x [1:3, 1:3] : (tensor<4x4xf32>) -> tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        // 4x4 matrix:
        // [ 0,  1,  2,  3]
        // [ 4,  5,  6,  7]
        // [ 8,  9, 10, 11]
        // [12, 13, 14, 15]
        let x = try client.createBuffer(Array(0..<16).map { Float($0) }, shape: [4, 4], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Extract [1:3, 1:3]:
        // [5, 6]
        // [9, 10]
        #expect(result == [5, 6, 9, 10])
    }

    @Test("Slice: with stride")
    func sliceWithStride() throws {
        let client = try Client.create()
        let mlir = """
        module @slice_stride {
          func.func @main(%x: tensor<8xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.slice %x [0:8:2] : (tensor<8xf32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([0, 1, 2, 3, 4, 5, 6, 7] as [Float], shape: [8], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Every other element: [0, 2, 4, 6]
        #expect(result == [0, 2, 4, 6])
    }

    @Test("Slice: first row of matrix")
    func sliceFirstRow() throws {
        let client = try Client.create()
        let mlir = """
        module @slice_row {
          func.func @main(%x: tensor<3x4xf32>) -> (tensor<1x4xf32>) {
            %0 = stablehlo.slice %x [0:1, 0:4] : (tensor<3x4xf32>) -> tensor<1x4xf32>
            return %0 : tensor<1x4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer(Array(0..<12).map { Float($0) }, shape: [3, 4], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        #expect(result == [0, 1, 2, 3])
    }
}

// MARK: - Pad Operations

@Suite("Pad Operations")
struct PadTests {

    @Test("Pad: 1D zero padding")
    func pad1DZero() throws {
        let client = try Client.create()
        let mlir = """
        module @pad_1d {
          func.func @main(%x: tensor<3xf32>, %pad_value: tensor<f32>) -> (tensor<7xf32>) {
            %0 = stablehlo.pad %x, %pad_value, low = [2], high = [2], interior = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<7xf32>
            return %0 : tensor<7xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([1, 2, 3] as [Float], shape: [3], elementType: .float32)
        let padValue = try client.createBuffer([0] as [Float], shape: [], elementType: .float32)

        let outputs = try executable.execute([x, padValue])
        let result = try outputs[0].toFloatArray()

        // [0, 0, 1, 2, 3, 0, 0]
        #expect(result == [0, 0, 1, 2, 3, 0, 0])
    }

    @Test("Pad: 2D asymmetric padding")
    func pad2DAsymmetric() throws {
        let client = try Client.create()
        let mlir = """
        module @pad_2d_asym {
          func.func @main(%x: tensor<2x2xf32>, %pad_value: tensor<f32>) -> (tensor<4x3xf32>) {
            %0 = stablehlo.pad %x, %pad_value, low = [1, 0], high = [1, 1], interior = [0, 0] : (tensor<2x2xf32>, tensor<f32>) -> tensor<4x3xf32>
            return %0 : tensor<4x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        // Input:
        // [1, 2]
        // [3, 4]
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [2, 2], elementType: .float32)
        let padValue = try client.createBuffer([0] as [Float], shape: [], elementType: .float32)

        let outputs = try executable.execute([x, padValue])
        let result = try outputs[0].toFloatArray()

        // Output (low=[1,0], high=[1,1]):
        // [0, 0, 0]  <- pad row
        // [1, 2, 0]  <- original + pad col
        // [3, 4, 0]  <- original + pad col
        // [0, 0, 0]  <- pad row
        #expect(result == [0, 0, 0, 1, 2, 0, 3, 4, 0, 0, 0, 0])
    }

    @Test("Pad: left-only padding")
    func padLeftOnly() throws {
        let client = try Client.create()
        let mlir = """
        module @pad_left {
          func.func @main(%x: tensor<3xf32>, %pad_value: tensor<f32>) -> (tensor<5xf32>) {
            %0 = stablehlo.pad %x, %pad_value, low = [2], high = [0], interior = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<5xf32>
            return %0 : tensor<5xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([1, 2, 3] as [Float], shape: [3], elementType: .float32)
        let padValue = try client.createBuffer([0] as [Float], shape: [], elementType: .float32)

        let outputs = try executable.execute([x, padValue])
        let result = try outputs[0].toFloatArray()

        // [0, 0, 1, 2, 3]
        #expect(result == [0, 0, 1, 2, 3])
    }

    @Test("Pad: with non-zero constant pad value")
    func padWithConstantValue() throws {
        let client = try Client.create()
        // Use inline constant for pad value (common pattern in StableHLO)
        let mlir = """
        module @pad_const {
          func.func @main(%x: tensor<2xf32>) -> (tensor<4xf32>) {
            %pad_val = stablehlo.constant dense<-1.0> : tensor<f32>
            %0 = stablehlo.pad %x, %pad_val, low = [1], high = [1], interior = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([5, 6] as [Float], shape: [2], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // [-1, 5, 6, -1]
        #expect(result == [-1, 5, 6, -1])
    }
}

// MARK: - Concatenate Operations

@Suite("Concatenate Operations")
struct ConcatenateTests {

    @Test("Concatenate: 1D tensors")
    func concatenate1D() throws {
        let client = try Client.create()
        let mlir = """
        module @concat_1d {
          func.func @main(%a: tensor<3xf32>, %b: tensor<2xf32>) -> (tensor<5xf32>) {
            %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<3xf32>, tensor<2xf32>) -> tensor<5xf32>
            return %0 : tensor<5xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 2, 3] as [Float], shape: [3], elementType: .float32)
        let b = try client.createBuffer([4, 5] as [Float], shape: [2], elementType: .float32)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()

        #expect(result == [1, 2, 3, 4, 5])
    }

    @Test("Concatenate: 2D along axis 0 (row-wise)")
    func concatenate2DAxis0() throws {
        let client = try Client.create()
        let mlir = """
        module @concat_2d_axis0 {
          func.func @main(%a: tensor<2x3xf32>, %b: tensor<1x3xf32>) -> (tensor<3x3xf32>) {
            %0 = stablehlo.concatenate %a, %b, dim = 0 : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
            return %0 : tensor<3x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        // a:
        // [1, 2, 3]
        // [4, 5, 6]
        let a = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [2, 3], elementType: .float32)
        // b:
        // [7, 8, 9]
        let b = try client.createBuffer([7, 8, 9] as [Float], shape: [1, 3], elementType: .float32)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()

        // Result:
        // [1, 2, 3]
        // [4, 5, 6]
        // [7, 8, 9]
        #expect(result == [1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    @Test("Concatenate: 2D along axis 1 (column-wise)")
    func concatenate2DAxis1() throws {
        let client = try Client.create()
        let mlir = """
        module @concat_2d_axis1 {
          func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x3xf32>) -> (tensor<2x5xf32>) {
            %0 = stablehlo.concatenate %a, %b, dim = 1 : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x5xf32>
            return %0 : tensor<2x5xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        // a:
        // [1, 2]
        // [3, 4]
        let a = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [2, 2], elementType: .float32)
        // b:
        // [5, 6, 7]
        // [8, 9, 10]
        let b = try client.createBuffer([5, 6, 7, 8, 9, 10] as [Float], shape: [2, 3], elementType: .float32)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()

        // Result:
        // [1, 2, 5, 6, 7]
        // [3, 4, 8, 9, 10]
        #expect(result == [1, 2, 5, 6, 7, 3, 4, 8, 9, 10])
    }

    @Test("Concatenate: three tensors")
    func concatenateThree() throws {
        let client = try Client.create()
        let mlir = """
        module @concat_three {
          func.func @main(%a: tensor<2xf32>, %b: tensor<3xf32>, %c: tensor<2xf32>) -> (tensor<7xf32>) {
            %0 = stablehlo.concatenate %a, %b, %c, dim = 0 : (tensor<2xf32>, tensor<3xf32>, tensor<2xf32>) -> tensor<7xf32>
            return %0 : tensor<7xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 2] as [Float], shape: [2], elementType: .float32)
        let b = try client.createBuffer([3, 4, 5] as [Float], shape: [3], elementType: .float32)
        let c = try client.createBuffer([6, 7] as [Float], shape: [2], elementType: .float32)

        let outputs = try executable.execute([a, b, c])
        let result = try outputs[0].toFloatArray()

        #expect(result == [1, 2, 3, 4, 5, 6, 7])
    }
}

// MARK: - Gather Operations (Embedding Lookup)

@Suite("Gather Operations")
struct GatherTests {

    @Test("Gather: simple embedding lookup")
    func gatherEmbeddingLookup() throws {
        let client = try Client.create()
        // Embedding table: 5 embeddings of dimension 3
        // Indices: lookup embeddings at positions 0, 2, 4
        let mlir = """
        module @embedding_lookup {
          func.func @main(%embeddings: tensor<5x3xf32>, %indices: tensor<3xi32>) -> (tensor<3x3xf32>) {
            %0 = stablehlo.gather %embeddings, %indices,
              offset_dims = [1],
              collapsed_slice_dims = [0],
              start_index_map = [0],
              index_vector_dim = 1,
              slice_sizes = [1, 3]
              : (tensor<5x3xf32>, tensor<3xi32>) -> tensor<3x3xf32>
            return %0 : tensor<3x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Embedding table:
        // [0.0, 0.1, 0.2]  <- index 0
        // [1.0, 1.1, 1.2]  <- index 1
        // [2.0, 2.1, 2.2]  <- index 2
        // [3.0, 3.1, 3.2]  <- index 3
        // [4.0, 4.1, 4.2]  <- index 4
        var embeddings: [Float] = []
        for i in 0..<5 {
            embeddings.append(contentsOf: [Float(i) + 0.0, Float(i) + 0.1, Float(i) + 0.2])
        }
        let embeddingBuffer = try client.createBuffer(embeddings, shape: [5, 3], elementType: .float32)

        // Indices: lookup [0, 2, 4]
        let indices = try client.createBuffer([0, 2, 4] as [Int32], shape: [3], elementType: .int32)

        let outputs = try executable.execute([embeddingBuffer, indices])
        let result = try outputs[0].toFloatArray()

        // Expected:
        // [0.0, 0.1, 0.2]  <- embedding 0
        // [2.0, 2.1, 2.2]  <- embedding 2
        // [4.0, 4.1, 4.2]  <- embedding 4
        let expected: [Float] = [0.0, 0.1, 0.2, 2.0, 2.1, 2.2, 4.0, 4.1, 4.2]
        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 0.001, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }

    @Test("Gather: batched embedding lookup")
    func gatherBatchedEmbedding() throws {
        let client = try Client.create()
        // Batch of 2, each looking up 2 embeddings from a 4x2 embedding table
        let mlir = """
        module @batched_embedding {
          func.func @main(%embeddings: tensor<4x2xf32>, %indices: tensor<2x2xi32>) -> (tensor<2x2x2xf32>) {
            %0 = stablehlo.gather %embeddings, %indices,
              offset_dims = [2],
              collapsed_slice_dims = [0],
              start_index_map = [0],
              index_vector_dim = 2,
              slice_sizes = [1, 2]
              : (tensor<4x2xf32>, tensor<2x2xi32>) -> tensor<2x2x2xf32>
            return %0 : tensor<2x2x2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Embedding table:
        // [0.0, 0.1]  <- index 0
        // [1.0, 1.1]  <- index 1
        // [2.0, 2.1]  <- index 2
        // [3.0, 3.1]  <- index 3
        let embeddings: [Float] = [0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1]
        let embeddingBuffer = try client.createBuffer(embeddings, shape: [4, 2], elementType: .float32)

        // Indices: batch of 2, each with 2 lookups
        // Batch 0: [0, 3] -> look up embeddings 0 and 3
        // Batch 1: [1, 2] -> look up embeddings 1 and 2
        let indices = try client.createBuffer([0, 3, 1, 2] as [Int32], shape: [2, 2], elementType: .int32)

        let outputs = try executable.execute([embeddingBuffer, indices])
        let result = try outputs[0].toFloatArray()

        // Expected shape: [2, 2, 2]
        // Batch 0: [[0.0, 0.1], [3.0, 3.1]]
        // Batch 1: [[1.0, 1.1], [2.0, 2.1]]
        let expected: [Float] = [0.0, 0.1, 3.0, 3.1, 1.0, 1.1, 2.0, 2.1]
        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 0.001, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }

    @Test("Gather: vocabulary embedding for NLP")
    func gatherVocabEmbedding() throws {
        let client = try Client.create()
        // Simulate NLP embedding: vocab_size=10, embedding_dim=4, sequence_length=6
        let mlir = """
        module @vocab_embedding {
          func.func @main(%vocab_embeddings: tensor<10x4xf32>, %token_ids: tensor<6xi32>) -> (tensor<6x4xf32>) {
            %0 = stablehlo.gather %vocab_embeddings, %token_ids,
              offset_dims = [1],
              collapsed_slice_dims = [0],
              start_index_map = [0],
              index_vector_dim = 1,
              slice_sizes = [1, 4]
              : (tensor<10x4xf32>, tensor<6xi32>) -> tensor<6x4xf32>
            return %0 : tensor<6x4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Create simple embedding table where row i has values [i*10, i*10+1, i*10+2, i*10+3]
        var embeddings: [Float] = []
        for i in 0..<10 {
            let base = Float(i * 10)
            embeddings.append(contentsOf: [base, base + 1, base + 2, base + 3])
        }
        let embeddingBuffer = try client.createBuffer(embeddings, shape: [10, 4], elementType: .float32)

        // Token IDs: [3, 1, 4, 1, 5, 9] (like a sequence of tokens)
        let tokenIds = try client.createBuffer([3, 1, 4, 1, 5, 9] as [Int32], shape: [6], elementType: .int32)

        let outputs = try executable.execute([embeddingBuffer, tokenIds])
        let result = try outputs[0].toFloatArray()

        // Expected: for each token ID, get that row
        // Token 3: [30, 31, 32, 33]
        // Token 1: [10, 11, 12, 13]
        // Token 4: [40, 41, 42, 43]
        // Token 1: [10, 11, 12, 13]
        // Token 5: [50, 51, 52, 53]
        // Token 9: [90, 91, 92, 93]
        let expected: [Float] = [
            30, 31, 32, 33,
            10, 11, 12, 13,
            40, 41, 42, 43,
            10, 11, 12, 13,
            50, 51, 52, 53,
            90, 91, 92, 93
        ]
        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 0.001, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }
}

// MARK: - Scatter Operations

@Suite("Scatter Operations")
struct ScatterTests {

    @Test("Scatter: update embedding table")
    func scatterEmbeddingUpdate() throws {
        let client = try Client.create()
        // Update specific rows in an embedding table
        let mlir = """
        module @scatter_update {
          func.func @main(%table: tensor<5x3xf32>, %indices: tensor<2xi32>, %updates: tensor<2x3xf32>) -> (tensor<5x3xf32>) {
            %0 = stablehlo.scatter %table, %indices, %updates,
              update_window_dims = [1],
              inserted_window_dims = [0],
              scatter_dims_to_operand_dims = [0],
              index_vector_dim = 1
              : (tensor<5x3xf32>, tensor<2xi32>, tensor<2x3xf32>) -> tensor<5x3xf32>
            return %0 : tensor<5x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Original embedding table (5 rows of 3 elements each)
        let table: [Float] = [
            0, 0, 0,  // row 0
            1, 1, 1,  // row 1
            2, 2, 2,  // row 2
            3, 3, 3,  // row 3
            4, 4, 4   // row 4
        ]
        let tableBuffer = try client.createBuffer(table, shape: [5, 3], elementType: .float32)

        // Indices to update: rows 1 and 3
        let indices = try client.createBuffer([1, 3] as [Int32], shape: [2], elementType: .int32)

        // New values for those rows
        let updates: [Float] = [
            10, 11, 12,  // new row 1
            30, 31, 32   // new row 3
        ]
        let updatesBuffer = try client.createBuffer(updates, shape: [2, 3], elementType: .float32)

        let outputs = try executable.execute([tableBuffer, indices, updatesBuffer])
        let result = try outputs[0].toFloatArray()

        // Expected: rows 1 and 3 updated
        let expected: [Float] = [
            0, 0, 0,     // row 0 unchanged
            10, 11, 12,  // row 1 updated
            2, 2, 2,     // row 2 unchanged
            30, 31, 32,  // row 3 updated
            4, 4, 4      // row 4 unchanged
        ]
        for (i, (r, e)) in zip(result, expected).enumerated() {
            #expect(abs(r - e) < 0.001, "Mismatch at index \(i): \(r) vs \(e)")
        }
    }
}

// MARK: - Combined Indexing Test

@Suite("Combined Indexing Operations")
struct CombinedIndexingTests {

    @Test("Slice + Pad: extract and pad back")
    func sliceThenPad() throws {
        let client = try Client.create()
        let mlir = """
        module @slice_pad {
          func.func @main(%x: tensor<6xf32>, %pad_val: tensor<f32>) -> (tensor<6xf32>) {
            %sliced = stablehlo.slice %x [1:4:1] : (tensor<6xf32>) -> tensor<3xf32>
            %padded = stablehlo.pad %sliced, %pad_val, low = [1], high = [2], interior = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<6xf32>
            return %padded : tensor<6xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([10, 20, 30, 40, 50, 60] as [Float], shape: [6], elementType: .float32)
        let padVal = try client.createBuffer([0] as [Float], shape: [], elementType: .float32)

        let outputs = try executable.execute([x, padVal])
        let result = try outputs[0].toFloatArray()

        // Slice [1:4] -> [20, 30, 40]
        // Pad low=1, high=2 -> [0, 20, 30, 40, 0, 0]
        #expect(result == [0, 20, 30, 40, 0, 0])
    }

    @Test("Concatenate + Slice: build and extract")
    func concatenateThenSlice() throws {
        let client = try Client.create()
        let mlir = """
        module @concat_slice {
          func.func @main(%a: tensor<3xf32>, %b: tensor<3xf32>) -> (tensor<4xf32>) {
            %concat = stablehlo.concatenate %a, %b, dim = 0 : (tensor<3xf32>, tensor<3xf32>) -> tensor<6xf32>
            %sliced = stablehlo.slice %concat [1:5:1] : (tensor<6xf32>) -> tensor<4xf32>
            return %sliced : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 2, 3] as [Float], shape: [3], elementType: .float32)
        let b = try client.createBuffer([4, 5, 6] as [Float], shape: [3], elementType: .float32)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()

        // Concat -> [1, 2, 3, 4, 5, 6]
        // Slice [1:5] -> [2, 3, 4, 5]
        #expect(result == [2, 3, 4, 5])
    }
}
