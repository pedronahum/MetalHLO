// LinalgMPSTests.swift
// MetalHLOTests
//
// Coverage for the linear-algebra / region ops that lower onto the MPSGraph
// backend: stablehlo.triangular_solve, stablehlo.cholesky, and stablehlo.map.
// Each case is checked numerically against a NumPy/JAX reference.

import Testing
@testable import MetalHLO

@Suite("Linalg + Map (MPSGraph path)", .serialized)
struct LinalgMPSTests {

    // Solve a lower-triangular system L·x = b by forward substitution.
    //   L = [[2,0,0],[6,1,0],[-8,5,3]]   (the Cholesky factor of an SPD matrix)
    //   b = [[1],[2],[3]]   →   x = [0.5, -1, 4]
    @Test("triangular_solve lower-triangular forward substitution")
    func triangularSolveLower() throws {
        let client = try Client.create()
        let mlir = """
        module @trisolve {
          func.func @main(%a: tensor<3x3xf32>, %b: tensor<3x1xf32>) -> (tensor<3x1xf32>) {
            %0 = "stablehlo.triangular_solve"(%a, %b) {
              left_side = true,
              lower = true,
              unit_diagonal = false,
              transpose_a = #stablehlo<transpose NO_TRANSPOSE>
            } : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
            return %0 : tensor<3x1xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer(
            [2, 0, 0, 6, 1, 0, -8, 5, 3] as [Float], shape: [3, 3], elementType: .float32)
        let b = try client.createBuffer([1, 2, 3] as [Float], shape: [3, 1], elementType: .float32)
        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()
        let expected: [Float] = [0.5, -1, 4]
        #expect(result.count == 3)
        for (got, want) in zip(result, expected) {
            #expect(abs(got - want) < 1e-4, "trisolve got \(got), want \(want)")
        }
    }

    // Cholesky of the SPD matrix
    //   A = [[4,12,-16],[12,37,-43],[-16,-43,98]]
    // The lower factor is L = [[2,0,0],[6,1,0],[-8,5,3]], and L·Lᵀ == A.
    @Test("cholesky lower factor of an SPD matrix")
    func choleskyLower() throws {
        let client = try Client.create()
        let mlir = """
        module @chol {
          func.func @main(%a: tensor<3x3xf32>) -> (tensor<3x3xf32>) {
            %0 = "stablehlo.cholesky"(%a) {lower = true} : (tensor<3x3xf32>) -> tensor<3x3xf32>
            return %0 : tensor<3x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer(
            [4, 12, -16, 12, 37, -43, -16, -43, 98] as [Float],
            shape: [3, 3], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // Row-major lower-triangular factor (upper entries are zero).
        let expected: [Float] = [2, 0, 0, 6, 1, 0, -8, 5, 3]
        #expect(result.count == 9)
        for (got, want) in zip(result, expected) {
            #expect(abs(got - want) < 1e-3, "cholesky got \(got), want \(want)")
        }
    }

    // map applies a scalar lambda element-wise. Here: x -> x*x + 1.
    //   [0,1,2,3] -> [1,2,5,10]
    @Test("map inlines an element-wise region")
    func mapElementwise() throws {
        let client = try Client.create()
        let mlir = """
        module @mapfn {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = "stablehlo.map"(%arg0) ({
              ^bb0(%a: tensor<f32>):
                %1 = stablehlo.multiply %a, %a : tensor<f32>
                %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                %2 = stablehlo.add %1, %cst : tensor<f32>
                stablehlo.return %2 : tensor<f32>
            }) {dimensions = array<i64: 0>} : (tensor<4xf32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let x = try client.createBuffer([0, 1, 2, 3] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()
        let expected: [Float] = [1, 2, 5, 10]
        #expect(result.count == 4)
        for (got, want) in zip(result, expected) {
            #expect(abs(got - want) < 1e-4, "map got \(got), want \(want)")
        }
    }
}
