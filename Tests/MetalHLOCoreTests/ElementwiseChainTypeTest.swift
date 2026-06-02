// ElementwiseChainTypeTest.swift
// Guards the fused-elementwise chain emitter against mistyped operands.
//
// When compare/select fuse into a single chain (default-on), the chain
// computes in float but an operand may be a narrower type whose buffer must be
// read at its true width before converting:
//   • an i1 mask feeding a select predicate is a 1-byte `bool` buffer, and
//   • an integer compare's operands (e.g. an iota index vs a threshold) are
//     `int` buffers.
// Reading either as `float` reinterprets the raw bytes and corrupts the result.
// These tests exercise both through the full compile→execute path.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Elementwise Chain Operand Typing", .serialized)
struct ElementwiseChainTypeTests {

    // select(compare(iota_i32, threshold_i32), x, y): the integer compare must
    // be read as `int`, not `float`. mask = idx < 2 → [T,T,F,F].
    @Test("Integer compare feeding select reads int operands at true width")
    func integerCompareSelect() async throws {
        let mlir = """
        module @int_select_test {
          func.func @main(%x: tensor<4xf32>, %y: tensor<4xf32>) -> (tensor<4xf32>) {
            %idx = stablehlo.iota dim = 0 : tensor<4xi32>
            %thresh = stablehlo.constant dense<2> : tensor<4xi32>
            %mask = stablehlo.compare LT, %idx, %thresh : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
            %r = stablehlo.select %mask, %x, %y : tensor<4xi1>, tensor<4xf32>
            return %r : tensor<4xf32>
          }
        }
        """

        let client = try Client.create()
        // O2 routes through the PassManager (producer-consumer-fusion) so the
        // compare→select actually fuses into one chain — the path under test.
        let executable = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))

        let x = try client.createBuffer([10, 20, 30, 40] as [Float], shape: [4], elementType: .float32)
        let y = try client.createBuffer([-1, -2, -3, -4] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([x, y])
        let result = try outputs[0].toFloatArray()

        // idx < 2 selects x for lanes 0,1; y for lanes 2,3.
        #expect(result == [10, 20, -3, -4], "int compare→select wrong: \(result)")
    }

    // Float compare feeding select must still match the comparison DIRECTION.
    // Regression guard for the rawValue-case bug (uppercase "LT" vs lowercase
    // match) that silently collapsed every comparison to "==".
    @Test("Float compare direction is honored in fused chain")
    func floatCompareDirection() async throws {
        let mlir = """
        module @lt_select_test {
          func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
            %mask = stablehlo.compare LT, %a, %b : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            %r = stablehlo.select %mask, %a, %b : tensor<4xi1>, tensor<4xf32>
            return %r : tensor<4xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))

        let a = try client.createBuffer([1, 5, 3, 9] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([4, 2, 8, 6] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()

        // select(a < b, a, b) = elementwise min(a, b) = [1, 2, 3, 6].
        // If LT collapsed to EQ, the predicate would be all-false → [4,2,8,6].
        #expect(result == [1, 2, 3, 6], "LT direction not honored: \(result)")
    }

    // A transpose fused into an elementwise chain reads its source via a
    // per-thread computed index. That index must use the INVERSE permutation;
    // using the forward perm only happens to work for involutions (e.g.
    // (0,2,1)). This uses a rotation (perm [2,0,1], inverse [1,2,0]) so a
    // forward-vs-inverse mistake produces wrong values. Surfaced via
    // jax.nn.dot_product_attention's forward + gradient being wrong.
    @Test("Transpose fused in elementwise chain uses inverse permutation")
    func transposeChainInversePerm() async throws {
        let mlir = """
        module @transpose_chain_test {
          func.func @main(%x: tensor<2x3x4xf32>) -> (tensor<4x2x3xf32>) {
            %t = stablehlo.transpose %x, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
            %n = stablehlo.negate %t : tensor<4x2x3xf32>
            return %n : tensor<4x2x3xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))

        // x[i,j,k] = i*12 + j*4 + k, shape [2,3,4].
        let xin = (0..<24).map { Float($0) }
        let x = try client.createBuffer(xin, shape: [2, 3, 4], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // Reference: out[a,b,c] = -x[b,c,a], shape [4,2,3] (perm [2,0,1]).
        var expected = [Float](repeating: 0, count: 24)
        for a in 0..<4 { for b in 0..<2 { for c in 0..<3 {
            expected[a * 6 + b * 3 + c] = -xin[b * 12 + c * 4 + a]
        }}}
        #expect(result == expected, "transpose perm wrong (forward instead of inverse): \(result)")
    }
}
