// ReducePrecisionTest.swift
// Tests stablehlo.reduce_precision (jax.lax.reduce_precision) lowering on the
// fast CodeGenerator path. Verifies round-to-nearest-even mantissa narrowing
// plus exponent-range clamping against JAX 0.10 reference values.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("ReducePrecision Tests", .serialized)
struct ReducePrecisionTests {

    // bf16 narrowing: e8m7 keeps fp32's 8-bit exponent and rounds the mantissa
    // from 23 down to 7 bits (round-to-nearest-even).
    @Test("reduce_precision e8m7 (bf16) mantissa rounding")
    func testReducePrecisionBF16() async throws {
        // JAX 0.10 lowers this as `format = e8m7`.
        let mlir = """
        module @reduce_precision_e8m7 {
          func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32>) {
            %0 = stablehlo.reduce_precision %arg0, format = e8m7 : tensor<8xf32>
            return %0 : tensor<8xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        let input: [Float] = [1.0, 1.2345678, 3.14159265, -2.7182818, 0.333333333, 100000.0, 1.5, 0.0]
        // Reference values from jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7).
        let expected: [Float] = [1.0, 1.234375, 3.140625, -2.71875, 0.333984375, 99840.0, 1.5, 0.0]

        let inBuffer = try client.createBuffer(input, shape: [8], elementType: .float32)
        let outputs = try executable.execute([inBuffer])
        let result = try outputs[0].toFloatArray()

        print("reduce_precision e8m7 result: \(result)")
        #expect(result.count == 8)
        for i in 0..<8 {
            // Reduced precision must match the reference bit-exactly.
            #expect(result[i] == expected[i],
                    "reduce_precision e8m7 [\(i)] = \(result[i]), expected \(expected[i])")
        }
    }

    // fp16 narrowing: e5m10 rounds the mantissa to 10 bits AND clamps the
    // dynamic range — large values overflow to +/-inf, tiny values underflow
    // to zero (target subnormals are not produced, matching XLA).
    @Test("reduce_precision e5m10 (fp16) mantissa rounding + exponent clamp")
    func testReducePrecisionFP16() async throws {
        let mlir = """
        module @reduce_precision_e5m10 {
          func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32>) {
            %0 = stablehlo.reduce_precision %arg0, format = e5m10 : tensor<8xf32>
            return %0 : tensor<8xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        let input: [Float] = [1.0, 1.2345678, 3.14159265, -2.7182818, 0.333333333, 100000.0, 1e-40, 65520.0]
        // Reference from jax.lax.reduce_precision(x, exponent_bits=5, mantissa_bits=10).
        let expected: [Float] = [1.0, 1.234375, 3.140625, -2.71875, 0.333251953125,
                                 Float.infinity, 0.0, Float.infinity]

        let inBuffer = try client.createBuffer(input, shape: [8], elementType: .float32)
        let outputs = try executable.execute([inBuffer])
        let result = try outputs[0].toFloatArray()

        print("reduce_precision e5m10 result: \(result)")
        #expect(result.count == 8)
        for i in 0..<8 {
            #expect(result[i] == expected[i],
                    "reduce_precision e5m10 [\(i)] = \(result[i]), expected \(expected[i])")
        }
    }
}
