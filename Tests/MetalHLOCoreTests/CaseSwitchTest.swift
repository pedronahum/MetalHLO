// CaseSwitchTest.swift
// Tests stablehlo.case (jax.lax.switch) lowering on the fast CodeGenerator path.
//
// jax.lax.switch(i, [f0, f1, f2], x) lowers to a clamp of the index into
// [0, N-1] followed by "stablehlo.case"(%idx) ({...}, {...}, {...}). The fast
// path has no runtime branch dispatch, so the parser expands the case into the
// inlined branches plus an index-select chain. These tests drive the exact MLIR
// JAX 0.10 emits and check that each index picks the matching branch.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Case / Switch Tests", .serialized)
struct CaseSwitchTests {

    // Mirrors jax.jit(lambda i, x: jax.lax.switch(i, [f0, f1, f2], x)) where
    //   f0(x) = x + 1, f1(x) = x * 2, f2(x) = x - 3
    // JAX clamps the index into [0, 2] before the case op.
    private let mlir = """
    module @jit_switch {
      func.func @main(%arg0: tensor<i32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
        %c = stablehlo.constant dense<0> : tensor<i32>
        %c_0 = stablehlo.constant dense<2> : tensor<i32>
        %0 = stablehlo.clamp %c, %arg0, %c_0 : tensor<i32>
        %1 = "stablehlo.case"(%0) ({
          %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
          %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
          %3 = stablehlo.add %arg1, %2 : tensor<4xf32>
          stablehlo.return %3 : tensor<4xf32>
        }, {
          %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
          %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
          %3 = stablehlo.multiply %arg1, %2 : tensor<4xf32>
          stablehlo.return %3 : tensor<4xf32>
        }, {
          %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
          %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
          %3 = stablehlo.subtract %arg1, %2 : tensor<4xf32>
          stablehlo.return %3 : tensor<4xf32>
        }) : (tensor<i32>) -> tensor<4xf32>
        return %1 : tensor<4xf32>
      }
    }
    """

    private func runSwitch(index: Int32) throws -> [Float] {
        let client = try Client.create()
        let executable = try client.compile(mlir)
        let x: [Float] = [10.0, 20.0, 30.0, 40.0]
        let idxBuffer = try client.createBuffer([index], shape: [], elementType: .int32)
        let xBuffer = try client.createBuffer(x, shape: [4], elementType: .float32)
        let outputs = try executable.execute([idxBuffer, xBuffer])
        return try outputs[0].toFloatArray()
    }

    @Test("switch selects each branch by index")
    func testSwitchBranches() async throws {
        let x: [Float] = [10.0, 20.0, 30.0, 40.0]

        // Branch 0: x + 1
        let r0 = try runSwitch(index: 0)
        for i in 0..<4 { #expect(abs(r0[i] - (x[i] + 1.0)) < 1e-5, "branch 0 @\(i): \(r0[i])") }

        // Branch 1: x * 2
        let r1 = try runSwitch(index: 1)
        for i in 0..<4 { #expect(abs(r1[i] - (x[i] * 2.0)) < 1e-5, "branch 1 @\(i): \(r1[i])") }

        // Branch 2: x - 3
        let r2 = try runSwitch(index: 2)
        for i in 0..<4 { #expect(abs(r2[i] - (x[i] - 3.0)) < 1e-5, "branch 2 @\(i): \(r2[i])") }
    }

    @Test("switch clamps out-of-range index like JAX")
    func testSwitchClampsIndex() async throws {
        let x: [Float] = [10.0, 20.0, 30.0, 40.0]

        // Negative index clamps to branch 0 (x + 1).
        let rNeg = try runSwitch(index: -5)
        for i in 0..<4 { #expect(abs(rNeg[i] - (x[i] + 1.0)) < 1e-5, "clamp-low @\(i): \(rNeg[i])") }

        // Over-range index clamps to the last branch 2 (x - 3).
        let rHigh = try runSwitch(index: 7)
        for i in 0..<4 { #expect(abs(rHigh[i] - (x[i] - 3.0)) < 1e-5, "clamp-high @\(i): \(rHigh[i])") }
    }
}
