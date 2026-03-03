// ANEBufferTests.swift
// ANERuntimeTests

import Foundation
import Testing
@testable import ANERuntime

@Suite("ANE Buffer", .serialized)
struct ANEBufferTests {

    // MARK: - Descriptor Shape Mapping

    @Test("1D shape maps to [1, 1, 1, N]")
    func descriptor1D() {
        let desc = ANEBufferDescriptor(shape: [256])
        #expect(desc.aneShape == [1, 1, 1, 256])
        #expect(desc.elementCount == 256)
    }

    @Test("2D shape maps to [1, M, 1, N]")
    func descriptor2D() {
        let desc = ANEBufferDescriptor(shape: [4, 64])
        #expect(desc.aneShape == [1, 4, 1, 64])
        #expect(desc.elementCount == 256)
    }

    @Test("3D shape maps to [1, C, 1, B*N]")
    func descriptor3D() {
        let desc = ANEBufferDescriptor(shape: [2, 8, 32])
        #expect(desc.aneShape == [1, 8, 1, 64])  // S = 2 * 32 = 64
        #expect(desc.elementCount == 512)
    }

    @Test("4D shape maps to [1, C, 1, B*H*W]")
    func descriptor4D() {
        let desc = ANEBufferDescriptor(shape: [1, 3, 16, 16])
        #expect(desc.aneShape == [1, 3, 1, 256])  // S = 1 * 16 * 16 = 256
        #expect(desc.elementCount == 768)
    }

    @Test("Scalar shape maps to [1, 1, 1, 1]")
    func descriptorScalar() {
        let desc = ANEBufferDescriptor(shape: [])
        #expect(desc.aneShape == [1, 1, 1, 1])
        #expect(desc.elementCount == 1)
    }

    @Test("Bytes per row is 64-byte aligned")
    func bytesPerRowAlignment() {
        // 10 FP16 elements = 20 bytes → aligned to 64
        let desc = ANEBufferDescriptor(shape: [10])
        #expect(desc.bytesPerRow % 64 == 0)
        #expect(desc.bytesPerRow >= 20)

        // 32 FP16 elements = 64 bytes → already aligned
        let desc2 = ANEBufferDescriptor(shape: [32])
        #expect(desc2.bytesPerRow == 64)
    }

    // MARK: - IOSurface Creation

    @Test("IOSurface creation succeeds for valid descriptor")
    func ioSurfaceCreation() {
        let desc = ANEBufferDescriptor(shape: [128])
        let buffer = ANEBuffer(descriptor: desc)
        #expect(buffer != nil)
        #expect(buffer?.isValid == true)
    }

    @Test("IOSurface creation succeeds for multi-dimensional shapes")
    func ioSurfaceCreation2D() {
        let desc = ANEBufferDescriptor(shape: [32, 64])
        let buffer = ANEBuffer(descriptor: desc)
        #expect(buffer != nil)
    }

    @Test("IOSurface creation succeeds for large buffers")
    func ioSurfaceLargeBuffer() {
        let desc = ANEBufferDescriptor(shape: [1024, 1024])
        let buffer = ANEBuffer(descriptor: desc)
        #expect(buffer != nil)
    }

    // MARK: - Data Round-trip

    @Test("FP32 write and read roundtrip preserves values within FP16 precision")
    func fp32Roundtrip() throws {
        let desc = ANEBufferDescriptor(shape: [4])
        guard let buffer = ANEBuffer(descriptor: desc) else {
            Issue.record("Failed to create buffer")
            return
        }
        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        try buffer.writeFloat32(input, shape: [4])
        let output = try buffer.readFloat32()
        #expect(output.count == 4)
        for (a, b) in zip(input, output) {
            #expect(abs(a - b) < 0.01)
        }
    }

    @Test("Larger FP32 roundtrip preserves values")
    func fp32RoundtripLarger() throws {
        let shape = [8, 16]
        let desc = ANEBufferDescriptor(shape: shape)
        guard let buffer = ANEBuffer(descriptor: desc) else {
            Issue.record("Failed to create buffer")
            return
        }
        let input = (0..<128).map { Float($0) * 0.1 }
        try buffer.writeFloat32(input, shape: shape)
        let output = try buffer.readFloat32()
        #expect(output.count == 128)
        for (a, b) in zip(input, output) {
            // FP16 has ~3 decimal digits of precision
            #expect(abs(a - b) < max(abs(a) * 0.01, 0.01))
        }
    }

    @Test("Raw FP16 write and read roundtrip")
    func fp16Roundtrip() throws {
        let desc = ANEBufferDescriptor(shape: [4])
        guard let buffer = ANEBuffer(descriptor: desc) else {
            Issue.record("Failed to create buffer")
            return
        }

        let fp16Values: [Float16] = [1.0, 2.5, -3.0, 0.5]
        let data = fp16Values.withUnsafeBufferPointer { Data(buffer: $0) }
        try buffer.write(data)

        let readData = try buffer.readFP16()
        let readValues = readData.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float16.self))
        }
        #expect(readValues == fp16Values)
    }

    // MARK: - Validation

    @Test("Write rejects mismatched shape")
    func writeMismatchedShape() throws {
        let desc = ANEBufferDescriptor(shape: [4])
        guard let buffer = ANEBuffer(descriptor: desc) else {
            Issue.record("Failed to create buffer")
            return
        }
        #expect(throws: ANEError.self) {
            try buffer.writeFloat32([1.0, 2.0], shape: [2])
        }
    }

    @Test("Write rejects mismatched element count")
    func writeMismatchedCount() throws {
        let desc = ANEBufferDescriptor(shape: [4])
        guard let buffer = ANEBuffer(descriptor: desc) else {
            Issue.record("Failed to create buffer")
            return
        }
        #expect(throws: ANEError.self) {
            try buffer.writeFloat32([1.0, 2.0], shape: [4])  // Only 2 elements for shape [4]
        }
    }

    // MARK: - Lifecycle

    @Test("Buffer release invalidates the buffer")
    func bufferRelease() {
        let desc = ANEBufferDescriptor(shape: [8])
        let buffer = ANEBuffer(descriptor: desc)!
        #expect(buffer.isValid)
        buffer.release()
        #expect(!buffer.isValid)
    }

    @Test("Read after release throws")
    func readAfterRelease() {
        let desc = ANEBufferDescriptor(shape: [8])
        let buffer = ANEBuffer(descriptor: desc)!
        buffer.release()
        #expect(throws: ANEError.self) {
            _ = try buffer.readFP16()
        }
    }

    @Test("Write after release throws")
    func writeAfterRelease() throws {
        let desc = ANEBufferDescriptor(shape: [4])
        let buffer = ANEBuffer(descriptor: desc)!
        buffer.release()
        let data = Data(count: 8)
        #expect(throws: ANEError.self) {
            try buffer.write(data)
        }
    }
}
