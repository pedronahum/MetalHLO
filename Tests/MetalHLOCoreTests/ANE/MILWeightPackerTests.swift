// MILWeightPackerTests.swift
// MetalHLOCoreTests

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("MIL Weight Packer")
struct MILWeightPackerTests {

    @Test("Scalar constant emits inline value")
    func scalarInline() {
        let packer = MILWeightPacker()
        let builder = MILTextBuilder()

        packer.packConstant(
            name: "c0",
            value: .scalar(2.5),
            resultType: TensorType(shape: [], elementType: .float32),
            builder: builder
        )

        #expect(!packer.hasWeights)
    }

    @Test("Small splat emits inline array")
    func smallSplatInline() {
        let packer = MILWeightPacker()
        let builder = MILTextBuilder()

        packer.packConstant(
            name: "c0",
            value: .splat(1.0, TensorType(shape: [4], elementType: .float32)),
            resultType: TensorType(shape: [4], elementType: .float32),
            builder: builder
        )

        #expect(!packer.hasWeights)
    }

    @Test("Large splat uses BLOBFILE")
    func largeSplatBlob() {
        let packer = MILWeightPacker()
        let builder = MILTextBuilder()

        let type = TensorType(shape: [64], elementType: .float32)
        packer.packConstant(
            name: "c0",
            value: .splat(1.0, type),
            resultType: type,
            builder: builder
        )

        #expect(packer.hasWeights)
        // FP16: 64 elements * 2 bytes = 128 bytes
        #expect(packer.getWeightData().count == 128)
    }

    @Test("Small dense emits inline array")
    func smallDenseInline() {
        let packer = MILWeightPacker()
        let builder = MILTextBuilder()

        let values: [Double] = [1, 2, 3, 4]
        packer.packConstant(
            name: "c0",
            value: .dense(values, TensorType(shape: [4], elementType: .float32)),
            resultType: TensorType(shape: [4], elementType: .float32),
            builder: builder
        )

        #expect(!packer.hasWeights)
    }

    @Test("Large dense uses BLOBFILE with correct size")
    func largeDenseBlob() {
        let packer = MILWeightPacker()
        let builder = MILTextBuilder()

        let values = (0..<256).map { Double($0) }
        let type = TensorType(shape: [16, 16], elementType: .float32)
        packer.packConstant(
            name: "c0",
            value: .dense(values, type),
            resultType: type,
            builder: builder
        )

        #expect(packer.hasWeights)
        // FP16: 256 elements * 2 bytes = 512 bytes
        #expect(packer.getWeightData().count == 512)
    }

    @Test("Multiple constants accumulate in blob with correct offsets")
    func multipleConstants() {
        let packer = MILWeightPacker()
        let builder = MILTextBuilder()

        // First constant: 32 elements = 64 bytes
        let type1 = TensorType(shape: [32], elementType: .float32)
        packer.packConstant(
            name: "w1",
            value: .splat(1.0, type1),
            resultType: type1,
            builder: builder
        )

        // Second constant: 64 elements = 128 bytes
        let type2 = TensorType(shape: [64], elementType: .float32)
        packer.packConstant(
            name: "w2",
            value: .splat(2.0, type2),
            resultType: type2,
            builder: builder
        )

        // Total: 64 + 128 = 192 bytes
        #expect(packer.getWeightData().count == 192)
    }

    @Test("FP16 conversion preserves values within tolerance")
    func fp16Conversion() {
        let packer = MILWeightPacker()
        let builder = MILTextBuilder()

        let values: [Double] = [0.0, 0.5, 1.0, -1.0, 3.14, 100.0]
        let type = TensorType(shape: [32], elementType: .float32)
        // Make it large enough for BLOBFILE
        var allValues = Array(repeating: 0.0, count: 32)
        for (i, v) in values.enumerated() {
            allValues[i] = v
        }

        packer.packConstant(
            name: "w",
            value: .dense(allValues, type),
            resultType: type,
            builder: builder
        )

        let data = packer.getWeightData()
        #expect(data.count == 64) // 32 * 2 bytes

        // Read back FP16 values and verify
        let fp16Values = data.withUnsafeBytes { buffer -> [Float16] in
            let ptr = buffer.bindMemory(to: Float16.self)
            return Array(ptr)
        }

        for (i, expected) in values.enumerated() {
            let actual = Double(fp16Values[i])
            #expect(Swift.abs(actual - expected) < 0.1,
                    "Value at index \(i): expected \(expected), got \(actual)")
        }
    }
}
