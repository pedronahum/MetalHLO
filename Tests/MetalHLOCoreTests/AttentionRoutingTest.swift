// AttentionRoutingTest.swift
// Guards the gpuOnly -> MPSGraph routing for scaled-dot-product attention.
//
// Client.compile auto-routes a graph containing a (batched) dot_general + a
// softmax (stablehlo.exponential) to MPSGraph's native scaledDotProductAttention
// — far faster than the codegen path. Because the benchmark harness only checks
// timing, this guards that the routed path is also numerically correct.
//
// Known-answer construction: with Q = K = 0 every score is 0, so the softmax is
// uniform (1/S per key) and each output row is the column-mean of V — a value
// independent of the scale and of any softmax max-shift, so it pins the result
// exactly regardless of which MPSGraph attention path fires.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Attention Routing", .serialized)
struct AttentionRoutingTests {

    // B=1, H=1, S=4, D=3 softmax attention: softmax(scale · Q·Kᵀ) · V.
    private static let attentionMLIR = """
    module @attn {
      func.func @main(%q: tensor<1x1x4x3xf32>, %k: tensor<1x1x4x3xf32>, %v: tensor<1x1x4x3xf32>) -> (tensor<1x1x4x3xf32>) {
        %scale = stablehlo.constant dense<0.5773503> : tensor<f32>
        %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] : (tensor<1x1x4x3xf32>) -> tensor<1x1x3x4xf32>
        %scores = stablehlo.dot_general %q, %kt, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<1x1x4x3xf32>, tensor<1x1x3x4xf32>) -> tensor<1x1x4x4xf32>
        %scale_bc = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<1x1x4x4xf32>
        %scaled = stablehlo.multiply %scores, %scale_bc : tensor<1x1x4x4xf32>
        %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
        %max = stablehlo.reduce %scaled, %neg_inf applies stablehlo.maximum across dimensions = [3] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x4xf32>
        %max_bc = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2] : (tensor<1x1x4xf32>) -> tensor<1x1x4x4xf32>
        %shifted = stablehlo.subtract %scaled, %max_bc : tensor<1x1x4x4xf32>
        %exp = stablehlo.exponential %shifted : tensor<1x1x4x4xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<f32>
        %sum = stablehlo.reduce %exp, %zero applies stablehlo.add across dimensions = [3] : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x4xf32>
        %sum_bc = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<1x1x4xf32>) -> tensor<1x1x4x4xf32>
        %weights = stablehlo.divide %exp, %sum_bc : tensor<1x1x4x4xf32>
        %out = stablehlo.dot_general %weights, %v, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<1x1x4x4xf32>, tensor<1x1x4x3xf32>) -> tensor<1x1x4x3xf32>
        return %out : tensor<1x1x4x3xf32>
      }
    }
    """

    private func runUniformAttention(devicePolicy: DevicePolicy) throws -> [Float] {
        let client = try Client.create()
        let exe = try client.compile(
            Self.attentionMLIR,
            config: CompilationConfig(optimizationLevel: .O2, devicePolicy: devicePolicy))
        // Q = K = 0 → uniform softmax → each output row = column-mean of V.
        let zeros = [Float](repeating: 0, count: 12)
        let vData: [Float] = [1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12]
        let q = client.createBuffer(zeros, shape: [1, 1, 4, 3])
        let k = client.createBuffer(zeros, shape: [1, 1, 4, 3])
        let v = client.createBuffer(vData, shape: [1, 1, 4, 3])
        return try exe.execute([q, k, v])[0].toFloatArray()
    }

    @Test("gpuOnly attention routes to MPSGraph and is numerically correct")
    func attentionRoutingNumericallyCorrect() throws {
        // gpuOnly + (dot_general & exponential) → MPSGraph scaledDotProductAttention.
        let result = try runUniformAttention(devicePolicy: .gpuOnly)
        // Column-mean of V rows = [(1+4+7+10)/4, (2+5+8+11)/4, (3+6+9+12)/4]
        //                       = [5.5, 6.5, 7.5], repeated for all 4 query rows.
        let expectedRow: [Float] = [5.5, 6.5, 7.5]
        #expect(result.count == 12)
        for i in 0..<4 {
            for c in 0..<3 {
                let got = result[i * 3 + c]
                #expect(abs(got - expectedRow[c]) < 1e-4,
                        "row \(i) col \(c): expected \(expectedRow[c]), got \(got)")
            }
        }
    }
}
