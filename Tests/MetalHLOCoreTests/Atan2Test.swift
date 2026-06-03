// Atan2Test.swift
// Tests stablehlo.atan2 (jnp.arctan2) lowering on the fast CodeGenerator path.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Atan2 Tests", .serialized)
struct Atan2Tests {

    @Test("atan2 elementwise across all four quadrants + axes")
    func testAtan2Quadrants() async throws {
        // stablehlo.atan2 %y, %x : atan2(y, x). %arg0 is y, %arg1 is x.
        let mlir = """
        module @atan2_test {
          func.func @main(%y: tensor<8xf32>, %x: tensor<8xf32>) -> (tensor<8xf32>) {
            %0 = stablehlo.atan2 %y, %x : tensor<8xf32>
            return %0 : tensor<8xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // y/x pairs covering Q1, Q2, Q3, Q4, the +/- axes, and a couple of
        // arbitrary points. Reference values are JAX 0.10's jnp.arctan2.
        let y: [Float] = [1.0,  1.0, -1.0, -1.0,  0.0, 1.0,  3.0, -2.0]
        let x: [Float] = [1.0, -1.0, -1.0,  1.0, -1.0, 0.0,  4.0, -3.0]
        let expected: [Float] = [
            0.7853982, 2.3561945, -2.3561945, -0.7853982,
            3.1415925, 1.5707964, 0.6435011, -2.5535901,
        ]

        let yBuffer = try client.createBuffer(y, shape: [8], elementType: .float32)
        let xBuffer = try client.createBuffer(x, shape: [8], elementType: .float32)

        let outputs = try executable.execute([yBuffer, xBuffer])
        let result = try outputs[0].toFloatArray()

        print("atan2 result: \(result)")
        #expect(result.count == 8)
        for i in 0..<8 {
            #expect(abs(result[i] - expected[i]) < 1e-5,
                    "atan2(\(y[i]), \(x[i])) = \(result[i]), expected \(expected[i])")
        }
    }
}
