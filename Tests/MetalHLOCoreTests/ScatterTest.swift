// ScatterTest.swift
// Tests stablehlo.scatter parsing and execution.
//
// Scatter is used for the gradient of gather (embedding lookup backward pass).
// It requires a region body to specify the update computation (e.g., identity or add).
// Currently MetalHLO fails to parse the region syntax "({ ... })".

import Testing
import Foundation
import Metal
@testable import MetalHLO
@testable import MetalHLOCore

enum ScatterTestError: Error { case noDevice }

@Suite("Scatter Tests", .serialized)
struct ScatterTests {

    // MARK: - Parser tests for scatter region syntax

    @Test("Scatter with identity update (replace) - default")
    func testScatterIdentityDefault() async throws {
        // Scatter 2 rows into a 4x3 zero tensor at indices [1, 3]
        // This is the pattern Magma emits for gather gradient (embedding backward)
        let mlir = """
        module @scatter_test {
          func.func @main(%input: tensor<4x3xf32>, %indices: tensor<2xf32>, %updates: tensor<2x3xf32>) -> (tensor<4x3xf32>) {
            %0 = stablehlo.convert %indices : (tensor<2xf32>) -> tensor<2xi64>
            %1 = "stablehlo.scatter"(%input, %0, %updates) ({
            ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
              stablehlo.return %arg1 : tensor<f32>
            }) {
              scatter_dimension_numbers = #stablehlo.scatter<
                update_window_dims = [1],
                inserted_window_dims = [0],
                scatter_dims_to_operand_dims = [0],
                index_vector_dim = 1
              >,
              indices_are_sorted = false,
              unique_indices = false
            } : (tensor<4x3xf32>, tensor<2xi64>, tensor<2x3xf32>) -> tensor<4x3xf32>
            return %1 : tensor<4x3xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // 4x3 zeros
        let inputData: [Float] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        // Scatter into rows 1 and 3
        let indicesData: [Float] = [1, 3]
        // Updates: row0=[10,20,30], row1=[40,50,60]
        let updatesData: [Float] = [10, 20, 30, 40, 50, 60]

        let inputBuf = try client.createBuffer(inputData, shape: [4, 3], elementType: .float32)
        let indicesBuf = try client.createBuffer(indicesData, shape: [2], elementType: .float32)
        let updatesBuf = try client.createBuffer(updatesData, shape: [2, 3], elementType: .float32)

        let outputs = try executable.execute([inputBuf, indicesBuf, updatesBuf])
        let result = try outputs[0].toFloatArray()

        print("Scatter identity (default): \(result)")
        // Expected: [[0,0,0], [10,20,30], [0,0,0], [40,50,60]]
        #expect(result.count == 12)
        #expect(result[0] == 0, "Row 0 should be zeros")
        #expect(result[3] == 10, "Row 1 col 0 should be 10, got \(result[3])")
        #expect(result[4] == 20, "Row 1 col 1 should be 20, got \(result[4])")
        #expect(result[5] == 30, "Row 1 col 2 should be 30, got \(result[5])")
        #expect(result[6] == 0, "Row 2 should be zeros")
        #expect(result[9] == 40, "Row 3 col 0 should be 40, got \(result[9])")
        #expect(result[10] == 50, "Row 3 col 1 should be 50, got \(result[10])")
        #expect(result[11] == 60, "Row 3 col 2 should be 60, got \(result[11])")
    }

    @Test("Scatter with add update (accumulate) - default")
    func testScatterAddDefault() async throws {
        // Scatter with stablehlo.add: accumulates instead of replacing.
        // This is needed for gradient accumulation with duplicate indices.
        let mlir = """
        module @scatter_add_test {
          func.func @main(%input: tensor<3x2xf32>, %indices: tensor<4xf32>, %updates: tensor<4x2xf32>) -> (tensor<3x2xf32>) {
            %0 = stablehlo.convert %indices : (tensor<4xf32>) -> tensor<4xi64>
            %1 = "stablehlo.scatter"(%input, %0, %updates) ({
            ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
              %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
              stablehlo.return %sum : tensor<f32>
            }) {
              scatter_dimension_numbers = #stablehlo.scatter<
                update_window_dims = [1],
                inserted_window_dims = [0],
                scatter_dims_to_operand_dims = [0],
                index_vector_dim = 1
              >,
              indices_are_sorted = false,
              unique_indices = false
            } : (tensor<3x2xf32>, tensor<4xi64>, tensor<4x2xf32>) -> tensor<3x2xf32>
            return %1 : tensor<3x2xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // 3x2 zeros
        let inputData: [Float] = [0, 0, 0, 0, 0, 0]
        // Indices with duplicate: [0, 1, 0, 2] — index 0 appears twice
        let indicesData: [Float] = [0, 1, 0, 2]
        // Updates: 4 rows of 2 cols
        let updatesData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]

        let inputBuf = try client.createBuffer(inputData, shape: [3, 2], elementType: .float32)
        let indicesBuf = try client.createBuffer(indicesData, shape: [4], elementType: .float32)
        let updatesBuf = try client.createBuffer(updatesData, shape: [4, 2], elementType: .float32)

        let outputs = try executable.execute([inputBuf, indicesBuf, updatesBuf])
        let result = try outputs[0].toFloatArray()

        print("Scatter add (default): \(result)")
        // Row 0: [1,2] + [5,6] = [6,8]  (indices 0 and 2 both map to row 0)
        // Row 1: [3,4]                    (index 1 maps to row 1)
        // Row 2: [7,8]                    (index 3 maps to row 2)
        #expect(result.count == 6)
        #expect(result[0] == 6, "Row 0 col 0: 1+5=6, got \(result[0])")
        #expect(result[1] == 8, "Row 0 col 1: 2+6=8, got \(result[1])")
        #expect(result[2] == 3, "Row 1 col 0 should be 3, got \(result[2])")
        #expect(result[3] == 4, "Row 1 col 1 should be 4, got \(result[3])")
        #expect(result[4] == 7, "Row 2 col 0 should be 7, got \(result[4])")
        #expect(result[5] == 8, "Row 2 col 1 should be 8, got \(result[5])")
    }

    @Test("Add-scatter seeds unscattered positions from operand (reuse-safe)")
    func testScatterAddSeedsFromOperand() async throws {
        // Regression for the jnp.unique corruption: the accumulating (non-SET)
        // scatter must initialize EVERY output position from its operand, not
        // just the positions it writes. The old kernel skipped the operand→output
        // copy and assumed the output slab was fresh-zero — true only when the
        // memory planner gives the result an untouched region. Under buffer reuse
        // the slot holds another tensor's stale bytes, so unscattered positions
        // came out dirty (unique([1..6],size=4) → [2,4,1,5] instead of [1,2,3,4]).
        //
        // i32 add-scatter takes the non-atomic accumulating path (the float32-only
        // atomic fast path is bypassed). We give it a NONZERO operand and scatter a
        // single update; the unscattered positions must equal the operand, proving
        // the seed happens. Pre-fix this returns 0 at those positions (the executor
        // memsets the slab to 0 and the kernel never wrote them) and FAILS.
        let mlir = """
        module @scatter_add_seed_test {
          func.func @main(%operand: tensor<4xi32>, %indices: tensor<1x1xi32>, %updates: tensor<1xi32>) -> (tensor<4xi32>) {
            %1 = "stablehlo.scatter"(%operand, %indices, %updates) ({
            ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
              %sum = stablehlo.add %arg0, %arg1 : tensor<i32>
              stablehlo.return %sum : tensor<i32>
            }) {
              scatter_dimension_numbers = #stablehlo.scatter<
                update_window_dims = [],
                inserted_window_dims = [0],
                scatter_dims_to_operand_dims = [0],
                index_vector_dim = 1
              >,
              indices_are_sorted = false,
              unique_indices = false
            } : (tensor<4xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<4xi32>
            return %1 : tensor<4xi32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Nonzero operand standing in for a reused dirty slot.
        let operandData: [Int32] = [10, 20, 30, 40]
        let indicesData: [Int32] = [1]   // scatter at position 1
        let updatesData: [Int32] = [7]   // add 7 there

        let operandBuf = try client.createBuffer(operandData, shape: [4], elementType: .int32)
        let indicesBuf = try client.createBuffer(indicesData, shape: [1, 1], elementType: .int32)
        let updatesBuf = try client.createBuffer(updatesData, shape: [1], elementType: .int32)

        let outputs = try executable.execute([operandBuf, indicesBuf, updatesBuf])
        let result = try outputs[0].toInt32Array()

        print("Add-scatter seed result: \(result)")
        // Expected: operand with +7 at position 1 = [10, 27, 30, 40].
        #expect(result.count == 4)
        #expect(result[0] == 10, "Position 0 must keep operand value 10, got \(result[0])")
        #expect(result[1] == 27, "Position 1 = 20+7 = 27, got \(result[1])")
        #expect(result[2] == 30, "Position 2 must keep operand value 30, got \(result[2])")
        #expect(result[3] == 40, "Position 3 must keep operand value 40, got \(result[3])")
    }

    @Test("Generated add-scatter kernel seeds every output from operand")
    func generatedAddScatterSeedsOperand() throws {
        // Kernel-contract regression for the jnp.unique corruption. The accumulating
        // (non-SET) scatter must be OUTPUT-centric: one thread per output element
        // seeds `result = operand[tid]` and ALWAYS writes `output[tid]`. The OLD
        // per-update kernel only wrote the touched positions and assumed the output
        // slab was fresh-zero — false under buffer reuse, which left unscattered
        // positions holding a dead tensor's stale bytes (unique([1..6],size=4) →
        // [2,4,1,5]). This asserts the generated Metal source directly so it fails
        // deterministically if the fix is reverted, independent of planner reuse
        // heuristics. i32 operand ⇒ the non-atomic accumulating path (the float32
        // atomic fast path is bypassed).
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ScatterTestError.noDevice
        }
        let gen = CodeGenerator(device: device)
        let dims = ScatterDimensionNumbers(
            updateWindowDims: [],
            insertedWindowDims: [0],
            scatterDimsToOperandDims: [0],
            indexVectorDim: 1
        )
        let source = gen.generateScatterKernel(
            entryPoint: "scatter_add_seed",
            operandShape: [4],
            indicesShape: [1, 1],
            updatesShape: [1],
            outputShape: [4],
            dimNumbers: dims,
            computationKind: .add,
            operandType: "int",
            indicesType: "int"
        )

        print("Generated add-scatter kernel:\n\(source)")
        // The seed from operand and the unconditional per-output write are the
        // exact lines the fix introduced; both are absent in the old per-update
        // kernel (which has `if (tid >= 1) return;` over the single update and a
        // bare `output[dstPos] = output[dstPos] + ...`).
        #expect(source.contains("result = operand[tid]"),
                "Accumulating scatter must seed result from operand[tid]")
        #expect(source.contains("output[tid] = result"),
                "Accumulating scatter must write every output position")
        #expect(!source.contains("output[dstPos] = output[dstPos] + updates[tid]"),
                "Accumulating scatter must not use the old per-update RMW that skips unscattered positions")
    }

    @Test("Scatter with identity update - O3")
    func testScatterIdentityO3() async throws {
        let mlir = """
        module @scatter_test {
          func.func @main(%input: tensor<4x3xf32>, %indices: tensor<2xf32>, %updates: tensor<2x3xf32>) -> (tensor<4x3xf32>) {
            %0 = stablehlo.convert %indices : (tensor<2xf32>) -> tensor<2xi64>
            %1 = "stablehlo.scatter"(%input, %0, %updates) ({
            ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
              stablehlo.return %arg1 : tensor<f32>
            }) {
              scatter_dimension_numbers = #stablehlo.scatter<
                update_window_dims = [1],
                inserted_window_dims = [0],
                scatter_dims_to_operand_dims = [0],
                index_vector_dim = 1
              >,
              indices_are_sorted = false,
              unique_indices = false
            } : (tensor<4x3xf32>, tensor<2xi64>, tensor<2x3xf32>) -> tensor<4x3xf32>
            return %1 : tensor<4x3xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig()
        config.optimizationLevel = .O3
        let executable = try client.compile(mlir, config: config)

        let inputData: [Float] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let indicesData: [Float] = [1, 3]
        let updatesData: [Float] = [10, 20, 30, 40, 50, 60]

        let inputBuf = try client.createBuffer(inputData, shape: [4, 3], elementType: .float32)
        let indicesBuf = try client.createBuffer(indicesData, shape: [2], elementType: .float32)
        let updatesBuf = try client.createBuffer(updatesData, shape: [2, 3], elementType: .float32)

        let outputs = try executable.execute([inputBuf, indicesBuf, updatesBuf])
        let result = try outputs[0].toFloatArray()

        print("Scatter identity (O3): \(result)")
        #expect(result.count == 12)
        #expect(result[3] == 10, "Row 1 col 0 should be 10, got \(result[3])")
        #expect(result[9] == 40, "Row 3 col 0 should be 40, got \(result[9])")
    }

    // MARK: - Realistic embedding gradient pattern

    @Test("Embedding gradient pattern: gather + scatter round-trip - default")
    func testEmbeddingGradientDefault() async throws {
        // This is the exact pattern used in GPT-2 training:
        // Forward:  gather rows from embedding table by token indices
        // Backward: scatter gradients back to embedding table positions
        //
        // gather: table[3x4] @ indices[2] -> output[2x4]
        // scatter: zeros[3x4].scatter(indices[2], grads[2x4]) -> grad_table[3x4]
        let mlir = """
        module @embedding_grad_test {
          func.func @main(%table: tensor<3x4xf32>, %indices: tensor<2xf32>, %upstream_grad: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<3x4xf32>) {
            %idx = stablehlo.convert %indices : (tensor<2xf32>) -> tensor<2xi64>

            %gathered = "stablehlo.gather"(%table, %idx) {
              dimension_numbers = #stablehlo.gather<
                offset_dims = [1],
                collapsed_slice_dims = [0],
                start_index_map = [0],
                index_vector_dim = 1
              >,
              slice_sizes = array<i64: 1, 4>,
              indices_are_sorted = false
            } : (tensor<3x4xf32>, tensor<2xi64>) -> tensor<2x4xf32>

            %zeros = stablehlo.constant dense<0.0> : tensor<3x4xf32>

            %grad_table = "stablehlo.scatter"(%zeros, %idx, %upstream_grad) ({
            ^bb0(%current: tensor<f32>, %update: tensor<f32>):
              %sum = stablehlo.add %current, %update : tensor<f32>
              stablehlo.return %sum : tensor<f32>
            }) {
              scatter_dimension_numbers = #stablehlo.scatter<
                update_window_dims = [1],
                inserted_window_dims = [0],
                scatter_dims_to_operand_dims = [0],
                index_vector_dim = 1
              >,
              indices_are_sorted = false,
              unique_indices = false
            } : (tensor<3x4xf32>, tensor<2xi64>, tensor<2x4xf32>) -> tensor<3x4xf32>

            return %gathered, %grad_table : tensor<2x4xf32>, tensor<3x4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Embedding table: 3 tokens, 4-dim embeddings
        let tableData: [Float] = [
            1, 2, 3, 4,    // token 0
            5, 6, 7, 8,    // token 1
            9, 10, 11, 12  // token 2
        ]
        // Look up tokens 2 and 0
        let indicesData: [Float] = [2, 0]
        // Upstream gradient for [2x4]
        let gradData: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        let tableBuf = try client.createBuffer(tableData, shape: [3, 4], elementType: .float32)
        let indicesBuf = try client.createBuffer(indicesData, shape: [2], elementType: .float32)
        let gradBuf = try client.createBuffer(gradData, shape: [2, 4], elementType: .float32)

        let outputs = try executable.execute([tableBuf, indicesBuf, gradBuf])
        let gathered = try outputs[0].toFloatArray()
        let gradTable = try outputs[1].toFloatArray()

        print("Gathered: \(gathered)")
        print("Grad table: \(gradTable)")

        // Gathered should be rows [2, 0] = [[9,10,11,12], [1,2,3,4]]
        #expect(gathered[0] == 9, "Should gather token 2: got \(gathered[0])")
        #expect(gathered[4] == 1, "Should gather token 0: got \(gathered[4])")

        // Grad table: scatter upstream_grad to positions [2, 0]
        // Row 0 gets grad[1] = [0.5, 0.6, 0.7, 0.8]
        // Row 1 gets nothing = [0, 0, 0, 0]
        // Row 2 gets grad[0] = [0.1, 0.2, 0.3, 0.4]
        #expect(abs(gradTable[0] - 0.5) < 1e-5, "Row 0 col 0: got \(gradTable[0])")
        #expect(abs(gradTable[4] - 0.0) < 1e-5, "Row 1 should be 0: got \(gradTable[4])")
        #expect(abs(gradTable[8] - 0.1) < 1e-5, "Row 2 col 0: got \(gradTable[8])")
    }
}
