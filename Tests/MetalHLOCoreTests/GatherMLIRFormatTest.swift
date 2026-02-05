// GatherMLIRFormatTest.swift
// Tests the #stablehlo.gather<> MLIR format parsing

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Gather MLIR Format Tests")
struct GatherMLIRFormatTests {

    @Test("Gather with #stablehlo.gather<> format - default")
    func testGatherDefault() async throws {
        let mlir = """
        module @gather_test {
          func.func @main(%operand: tensor<3x3xf32>, %indices: tensor<2xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.convert %indices : (tensor<2xf32>) -> tensor<2xi64>
            %1 = "stablehlo.gather"(%operand, %0) {
              dimension_numbers = #stablehlo.gather<
                offset_dims = [1],
                collapsed_slice_dims = [0],
                start_index_map = [0],
                index_vector_dim = 1
              >,
              slice_sizes = array<i64: 1, 3>,
              indices_are_sorted = false
            } : (tensor<3x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
            return %1 : tensor<2x3xf32>
          }
        }
        """

        let client = try Client.create()
        // Default - no config
        let executable = try client.compile(mlir)

        let operandData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        let indicesData: [Float] = [0, 2]

        let operandBuffer = try client.createBuffer(operandData, shape: [3, 3], elementType: .float32)
        let indicesBuffer = try client.createBuffer(indicesData, shape: [2], elementType: .float32)

        let outputs = try executable.execute([operandBuffer, indicesBuffer])
        let result = try outputs[0].toFloatArray()

        print("Default result: \(result)")
        #expect(result[0] == 1, "Expected row 0 first element to be 1")
    }

    @Test("Gather with #stablehlo.gather<> format - O3")
    func testGatherO3() async throws {
        let mlir = """
        module @gather_test {
          func.func @main(%operand: tensor<3x3xf32>, %indices: tensor<2xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.convert %indices : (tensor<2xf32>) -> tensor<2xi64>
            %1 = "stablehlo.gather"(%operand, %0) {
              dimension_numbers = #stablehlo.gather<
                offset_dims = [1],
                collapsed_slice_dims = [0],
                start_index_map = [0],
                index_vector_dim = 1
              >,
              slice_sizes = array<i64: 1, 3>,
              indices_are_sorted = false
            } : (tensor<3x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
            return %1 : tensor<2x3xf32>
          }
        }
        """

        let client = try Client.create()
        var config = CompilationConfig()
        config.optimizationLevel = .O3
        let executable = try client.compile(mlir, config: config)

        let operandData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        let indicesData: [Float] = [0, 2]

        let operandBuffer = try client.createBuffer(operandData, shape: [3, 3], elementType: .float32)
        let indicesBuffer = try client.createBuffer(indicesData, shape: [2], elementType: .float32)

        let outputs = try executable.execute([operandBuffer, indicesBuffer])
        let result = try outputs[0].toFloatArray()

        print("O3 result: \(result)")
        #expect(result[0] == 1, "Expected row 0 first element to be 1")
    }

    @Test("Gather with int32 indices directly")
    func testGatherInt32Indices() async throws {
        // Using int32 indices directly (no conversion)
        let mlir = """
        module @gather_test {
          func.func @main(%operand: tensor<3x3xf32>, %indices: tensor<2xi32>) -> (tensor<2x3xf32>) {
            %1 = "stablehlo.gather"(%operand, %indices) {
              dimension_numbers = #stablehlo.gather<
                offset_dims = [1],
                collapsed_slice_dims = [0],
                start_index_map = [0],
                index_vector_dim = 1
              >,
              slice_sizes = array<i64: 1, 3>,
              indices_are_sorted = false
            } : (tensor<3x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
            return %1 : tensor<2x3xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Create test data: 3x3 matrix with values 1-9
        let operandData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        // Indices: select rows 0 and 2
        let indicesData: [Int32] = [0, 2]

        let operandBuffer = try client.createBuffer(operandData, shape: [3, 3], elementType: .float32)
        let indicesBuffer = try client.createBuffer(indicesData, shape: [2], elementType: .int32)

        let outputs = try executable.execute([operandBuffer, indicesBuffer])
        let result = try outputs[0].toFloatArray()

        print("Gather result (int32): \(result)")

        // Expected: rows 0 and 2 of the matrix
        #expect(result.count == 6)
        #expect(result[0] == 1)
        #expect(result[1] == 2)
        #expect(result[2] == 3)
        #expect(result[3] == 7)
        #expect(result[4] == 8)
        #expect(result[5] == 9)
    }
}
