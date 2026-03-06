// SharedIOSurfaceTests.swift
// MetalHLOCoreTests
//
// Tests for shared IOSurface-backed buffers: GPU↔ANE zero-copy
// memory sharing, MLMultiArray interop, and integration with
// TensorTransferManager.

import Testing
import Foundation
import Metal
import CoreML
@testable import MetalHLOCore

@Suite("Shared IOSurface Transfer")
struct SharedIOSurfaceTests {

    // MARK: - SharedIOSurfaceBuffer

    @Test("SharedIOSurfaceBuffer creates valid Metal buffer")
    func sharedBufferCreation() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device available")
            return
        }

        let shape = [4, 8]
        guard let buffer = SharedIOSurfaceBuffer(device: device, shape: shape) else {
            Issue.record("Failed to create shared buffer")
            return
        }

        #expect(buffer.count == 32)
        #expect(buffer.byteSize == 32 * MemoryLayout<Float>.size)
        #expect(buffer.shape == [4, 8])
        #expect(buffer.metalBuffer.length >= buffer.byteSize)
    }

    @Test("SharedIOSurfaceBuffer round-trips Float data")
    func sharedBufferRoundTrip() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device available")
            return
        }

        let shape = [2, 3]
        let inputData: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        guard let buffer = SharedIOSurfaceBuffer(device: device, shape: shape, data: inputData) else {
            Issue.record("Failed to create shared buffer")
            return
        }

        let output = buffer.toFloatArray()
        #expect(output.count == 6)
        for i in 0..<6 {
            #expect(output[i] == inputData[i], "Mismatch at index \(i)")
        }
    }

    @Test("SharedIOSurfaceBuffer and MLMultiArray share memory")
    func sharedBufferMLMultiArraySharing() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device available")
            return
        }

        let shape = [2, 4]
        let data: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        guard let buffer = SharedIOSurfaceBuffer(device: device, shape: shape, data: data) else {
            Issue.record("Failed to create shared buffer")
            return
        }

        let mlArray = try buffer.makeMLMultiArray()

        // Verify MLMultiArray reads the same data
        #expect(mlArray.count == 8)
        for i in 0..<8 {
            let value = mlArray[[i / 4, i % 4] as [NSNumber]].floatValue
            #expect(value == data[i], "MLMultiArray mismatch at index \(i)")
        }

        // Write through Metal buffer pointer, verify via MLMultiArray
        buffer.write([10, 20, 30, 40, 50, 60, 70, 80])
        let newMlArray = try buffer.makeMLMultiArray()
        let firstValue = newMlArray[[0, 0] as [NSNumber]].floatValue
        #expect(firstValue == 10.0, "Write through Metal buffer should be visible in MLMultiArray")
    }

    @Test("SharedIOSurfaceBuffer rejects empty shape")
    func sharedBufferEmptyShape() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device available")
            return
        }

        let buffer = SharedIOSurfaceBuffer(device: device, shape: [0])
        #expect(buffer == nil, "Empty shape should return nil")
    }

    // MARK: - TensorTransferManager Integration

    @Test("TensorTransferManager provides shared MLMultiArray from GPU storage")
    func transferManagerSharedFromGPU() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device available")
            return
        }

        let manager = TensorTransferManager(device: device)
        let data: [Float] = [1, 2, 3, 4]
        let storage = BufferStorage(floatData: data, shape: [2, 2], device: device)
        manager.storeGPUResult(name: "%x", storage: storage)

        let result = try manager.getSharedMLMultiArray(name: "%x")
        #expect(result != nil, "Should create shared MLMultiArray from GPU storage")
        #expect(result?.shape == [2, 2])
        #expect(result?.array.count == 4)

        // Verify values
        let val = result!.array[[0, 0] as [NSNumber]].floatValue
        #expect(val == 1.0)
    }

    @Test("TensorTransferManager provides shared MLMultiArray from CPU array")
    func transferManagerSharedFromCPU() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device available")
            return
        }

        let manager = TensorTransferManager(device: device)
        manager.storeCPUResult(name: "%y", data: [10, 20, 30], shape: [3])

        let result = try manager.getSharedMLMultiArray(name: "%y")
        #expect(result != nil, "Should create shared MLMultiArray from CPU array")
        #expect(result?.array.count == 3)
    }

    @Test("TensorTransferManager returns nil for unknown tensor")
    func transferManagerSharedUnknown() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device available")
            return
        }

        let manager = TensorTransferManager(device: device)
        let result = try manager.getSharedMLMultiArray(name: "%nonexistent")
        #expect(result == nil)
    }

    @Test("TensorTransferManager caches shared buffers")
    func transferManagerSharedCaching() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device available")
            return
        }

        let manager = TensorTransferManager(device: device)
        manager.storeCPUResult(name: "%z", data: [1, 2, 3, 4], shape: [4])

        let first = try manager.getSharedMLMultiArray(name: "%z")
        let second = try manager.getSharedMLMultiArray(name: "%z")

        #expect(first != nil)
        #expect(second != nil)
        // Both should return valid arrays with the same data
        #expect(first!.array.count == second!.array.count)
    }

    @Test("TensorTransferManager clear removes shared buffers")
    func transferManagerClearShared() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device available")
            return
        }

        let manager = TensorTransferManager(device: device)
        manager.storeCPUResult(name: "%w", data: [1, 2], shape: [2])
        _ = try manager.getSharedMLMultiArray(name: "%w")

        manager.clear()

        let result = try manager.getSharedMLMultiArray(name: "%w")
        #expect(result == nil, "Should be nil after clear")
    }
}
