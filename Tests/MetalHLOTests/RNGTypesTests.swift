// RNGTypesTests.swift
// MetalHLOTests
//
// Tests for Phase 6: RNG and element types.

import Testing
import Foundation
@testable import MetalHLO

// MARK: - RNG Uniform Distribution

@Suite("RNG Uniform")
struct RNGUniformTests {

    @Test("RNG uniform: generates values in [0, 1)")
    func rngUniformBasic() throws {
        let client = try Client.create()
        let mlir = """
        module @rng_uniform {
          func.func @main() -> (tensor<100xf32>) {
            %0 = stablehlo.rng UNIFORM : tensor<100xf32>
            return %0 : tensor<100xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let outputs = try executable.execute([])
        let result = try outputs[0].toFloatArray()

        // All values should be in [0, 1)
        #expect(result.count == 100)
        for (i, val) in result.enumerated() {
            #expect(val >= 0 && val < 1, "Value \(i) out of range: \(val)")
        }

        // Values should not all be the same (statistical sanity check)
        let uniqueCount = Set(result).count
        #expect(uniqueCount > 10, "Random values should be diverse, got \(uniqueCount) unique")
    }

    @Test("RNG uniform: 2D tensor")
    func rngUniform2D() throws {
        let client = try Client.create()
        let mlir = """
        module @rng_uniform_2d {
          func.func @main() -> (tensor<10x20xf32>) {
            %0 = stablehlo.rng UNIFORM : tensor<10x20xf32>
            return %0 : tensor<10x20xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let outputs = try executable.execute([])
        let result = try outputs[0].toFloatArray()

        #expect(result.count == 200)
        for val in result {
            #expect(val >= 0 && val < 1)
        }
    }
}

// MARK: - RNG Normal Distribution

@Suite("RNG Normal")
struct RNGNormalTests {

    @Test("RNG normal: generates normally distributed values")
    func rngNormalBasic() throws {
        let client = try Client.create()
        let mlir = """
        module @rng_normal {
          func.func @main() -> (tensor<1000xf32>) {
            %0 = stablehlo.rng NORMAL : tensor<1000xf32>
            return %0 : tensor<1000xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let outputs = try executable.execute([])
        let result = try outputs[0].toFloatArray()

        #expect(result.count == 1000)

        // Compute mean and std
        let mean = result.reduce(0, +) / Float(result.count)
        let variance = result.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(result.count)
        let std = sqrt(variance)

        // Mean should be close to 0, std close to 1 for standard normal
        #expect(abs(mean) < 0.2, "Mean \(mean) should be close to 0")
        #expect(abs(std - 1.0) < 0.2, "Std \(std) should be close to 1")
    }

    @Test("RNG normal: 2D tensor")
    func rngNormal2D() throws {
        let client = try Client.create()
        let mlir = """
        module @rng_normal_2d {
          func.func @main() -> (tensor<50x20xf32>) {
            %0 = stablehlo.rng NORMAL : tensor<50x20xf32>
            return %0 : tensor<50x20xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let outputs = try executable.execute([])
        let result = try outputs[0].toFloatArray()

        #expect(result.count == 1000)

        // Should have both positive and negative values
        let positiveCount = result.filter { $0 > 0 }.count
        let negativeCount = result.filter { $0 < 0 }.count
        #expect(positiveCount > 100, "Should have positive values")
        #expect(negativeCount > 100, "Should have negative values")
    }
}

// MARK: - Dropout Layer (Success Criteria)

@Suite("Dropout Layer")
struct DropoutTests {

    @Test("Dropout: random masking with probability")
    func dropoutBasic() throws {
        let client = try Client.create()
        // Dropout: mask = rng < keep_prob, output = input * mask / keep_prob
        let mlir = """
        module @dropout {
          func.func @main(%input: tensor<100xf32>) -> (tensor<100xf32>) {
            %rng = stablehlo.rng UNIFORM : tensor<100xf32>
            %keep_prob = stablehlo.constant dense<0.8> : tensor<100xf32>
            %one = stablehlo.constant dense<1.0> : tensor<100xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<100xf32>

            // mask = rng < keep_prob (compare LT gives true where we keep)
            %mask_bool = stablehlo.compare %rng, %keep_prob, LT : (tensor<100xf32>, tensor<100xf32>) -> tensor<100xi1>

            // Convert bool mask to float (select 1.0 or 0.0)
            %mask = stablehlo.select %mask_bool, %one, %zero : (tensor<100xi1>, tensor<100xf32>, tensor<100xf32>) -> tensor<100xf32>

            // Apply mask and scale: output = input * mask / keep_prob
            %masked = stablehlo.multiply %input, %mask : tensor<100xf32>
            %scaled = stablehlo.divide %masked, %keep_prob : tensor<100xf32>

            return %scaled : tensor<100xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let input = try client.createBuffer(Array(repeating: Float(1.0), count: 100), shape: [100], elementType: .float32)

        let outputs = try executable.execute([input])
        let result = try outputs[0].toFloatArray()

        // With keep_prob=0.8, we expect:
        // - About 80% of values to be 1.0/0.8 = 1.25 (kept and scaled)
        // - About 20% of values to be 0.0 (dropped)
        let zeroCount = result.filter { $0 == 0.0 }.count
        let scaledCount = result.filter { abs($0 - 1.25) < 0.01 }.count

        // Allow for statistical variance
        #expect(zeroCount > 5 && zeroCount < 45, "Expected ~20% zeros, got \(zeroCount)")
        #expect(scaledCount > 55 && scaledCount < 95, "Expected ~80% scaled, got \(scaledCount)")
    }
}

// MARK: - Element Types

@Suite("Element Types")
struct ElementTypeTests {

    @Test("Float16 arithmetic")
    func float16Arithmetic() throws {
        let client = try Client.create()
        let mlir = """
        module @f16_test {
          func.func @main(%a: tensor<4xf16>, %b: tensor<4xf16>) -> (tensor<4xf16>) {
            %0 = stablehlo.add %a, %b : tensor<4xf16>
            return %0 : tensor<4xf16>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Create Float16 buffers
        let aData: [Float16] = [1.0, 2.0, 3.0, 4.0]
        let bData: [Float16] = [0.5, 1.5, 2.5, 3.5]
        let a = try client.createBuffer(aData, shape: [4], elementType: .float16)
        let b = try client.createBuffer(bData, shape: [4], elementType: .float16)

        let outputs = try executable.execute([a, b])

        // Read as Float16
        let f16Result = try outputs[0].toFloat16Array()
        let result = f16Result.map { Float($0) }

        #expect(abs(result[0] - 1.5) < 0.01)
        #expect(abs(result[1] - 3.5) < 0.01)
        #expect(abs(result[2] - 5.5) < 0.01)
        #expect(abs(result[3] - 7.5) < 0.01)
    }

    @Test("Int32 arithmetic")
    func int32Arithmetic() throws {
        let client = try Client.create()
        let mlir = """
        module @i32_test {
          func.func @main(%a: tensor<4xi32>, %b: tensor<4xi32>) -> (tensor<4xi32>) {
            %0 = stablehlo.add %a, %b : tensor<4xi32>
            return %0 : tensor<4xi32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let a = try client.createBuffer([1, 2, 3, 4] as [Int32], shape: [4], elementType: .int32)
        let b = try client.createBuffer([10, 20, 30, 40] as [Int32], shape: [4], elementType: .int32)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toInt32Array()

        #expect(result == [11, 22, 33, 44])
    }

    @Test("Int64 arithmetic")
    func int64Arithmetic() throws {
        let client = try Client.create()
        let mlir = """
        module @i64_test {
          func.func @main(%a: tensor<3xi64>, %b: tensor<3xi64>) -> (tensor<3xi64>) {
            %0 = stablehlo.add %a, %b : tensor<3xi64>
            return %0 : tensor<3xi64>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Use large values that don't fit in Int32
        let a = try client.createBuffer([Int64(1_000_000_000_000), 2, 3], shape: [3], elementType: .int64)
        let b = try client.createBuffer([Int64(1), 1_000_000_000_000, 1], shape: [3], elementType: .int64)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toInt64Array()

        #expect(result[0] == 1_000_000_000_001)
        #expect(result[1] == 1_000_000_000_002)
        #expect(result[2] == 4)
    }

    @Test("BFloat16 arithmetic")
    func bfloat16Arithmetic() throws {
        let client = try Client.create()
        let mlir = """
        module @bf16_test {
          func.func @main(%a: tensor<4xbf16>, %b: tensor<4xbf16>) -> (tensor<4xbf16>) {
            %0 = stablehlo.add %a, %b : tensor<4xbf16>
            return %0 : tensor<4xbf16>
          }
        }
        """

        let executable = try client.compile(mlir)

        // BFloat16 is typically handled as UInt16 for storage
        // Create Float values and execute
        let a = try client.createBuffer([1.0, 2.0, 3.0, 4.0] as [Float], shape: [4], elementType: .bfloat16)
        let b = try client.createBuffer([0.5, 1.5, 2.5, 3.5] as [Float], shape: [4], elementType: .bfloat16)

        let outputs = try executable.execute([a, b])

        // Verify the output has correct size (bf16 = 2 bytes per element)
        #expect(outputs[0].byteCount == 8) // 4 * 2 bytes
        #expect(outputs[0].count == 4)
    }

    @Test("Type conversion: i32 to f32")
    func convertI32ToF32() throws {
        let client = try Client.create()
        let mlir = """
        module @convert_test {
          func.func @main(%x: tensor<4xi32>) -> (tensor<4xf32>) {
            %0 = stablehlo.convert %x : (tensor<4xi32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([1, 2, 3, 4] as [Int32], shape: [4], elementType: .int32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        #expect(result == [1.0, 2.0, 3.0, 4.0])
    }

    @Test("Type conversion: f32 to f16")
    func convertF32ToF16() throws {
        let client = try Client.create()
        let mlir = """
        module @convert_f32_f16 {
          func.func @main(%x: tensor<4xf32>) -> (tensor<4xf16>) {
            %0 = stablehlo.convert %x : (tensor<4xf32>) -> tensor<4xf16>
            return %0 : tensor<4xf16>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([1.0, 2.0, 3.0, 4.0] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([x])

        // Verify output is f16 (2 bytes per element = 8 bytes total)
        #expect(outputs[0].byteCount == 8)

        // Read as Float16 and verify
        let f16Result = try outputs[0].toFloat16Array()
        #expect(Float(f16Result[0]) == 1.0)
        #expect(Float(f16Result[1]) == 2.0)
        #expect(Float(f16Result[2]) == 3.0)
        #expect(Float(f16Result[3]) == 4.0)
    }
}

// MARK: - Mixed Precision Operations

@Suite("Mixed Precision")
struct MixedPrecisionTests {

    @Test("F16 matmul with F32 accumulation pattern")
    func f16MatmulPattern() throws {
        let client = try Client.create()
        // Common pattern: compute in f16, accumulate/output in f32
        let mlir = """
        module @f16_matmul {
          func.func @main(%a: tensor<2x3xf16>, %b: tensor<3x2xf16>) -> (tensor<2x2xf32>) {
            // Matrix multiply in f16
            %mm = stablehlo.dot_general %a, %b,
              #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
              : (tensor<2x3xf16>, tensor<3x2xf16>) -> tensor<2x2xf16>

            // Convert result to f32 for higher precision output
            %result = stablehlo.convert %mm : (tensor<2x2xf16>) -> tensor<2x2xf32>
            return %result : tensor<2x2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Create f16 inputs
        let aData: [Float16] = [1, 2, 3, 4, 5, 6]
        let bData: [Float16] = [1, 2, 3, 4, 5, 6]
        let a = try client.createBuffer(aData, shape: [2, 3], elementType: .float16)
        let b = try client.createBuffer(bData, shape: [3, 2], elementType: .float16)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()

        // Verify output is f32 and correct
        // [1,2,3] @ [1,2; 3,4; 5,6] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // [4,5,6] @ [1,2; 3,4; 5,6] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        #expect(result.count == 4)
        #expect(abs(result[0] - 22) < 0.1)
        #expect(abs(result[1] - 28) < 0.1)
        #expect(abs(result[2] - 49) < 0.1)
        #expect(abs(result[3] - 64) < 0.1)
    }
}
