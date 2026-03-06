// SharedIOSurfaceBuffer.swift
// MetalHLOCore
//
// Provides zero-copy shared memory between Metal (GPU) and CoreML (ANE)
// via IOSurface-backed buffers. On Apple Silicon, GPU and ANE share
// unified memory, so this eliminates intermediate Float array copies
// on the input path.
//
// Transfer path with shared buffers:
//   GPU → MTLBuffer (IOSurface-backed) → MLMultiArray (same memory) → ANE
//   = 0 copies on input (was 2: MTLBuffer→[Float]→MLMultiArray)
//
// The output path still requires 1 copy since CoreML allocates its
// own output MLMultiArray.

import Foundation
import Metal
import CoreML

/// A buffer backed by IOSurface that can be accessed as both a Metal
/// buffer and a CoreML MLMultiArray without copying.
///
/// Memory coherence: On Apple Silicon, `storageModeShared` places the
/// buffer in cache-coherent unified memory accessible by both GPU and ANE.
/// No explicit GPU→ANE memory barrier is needed beyond ensuring the GPU
/// command buffer has completed before ANE reads. In the level-based
/// concurrent scheduler, this is guaranteed by `DispatchGroup.wait()`.
final class SharedIOSurfaceBuffer: @unchecked Sendable {

    /// The underlying Metal buffer (IOSurface-backed via shared storage mode).
    let metalBuffer: MTLBuffer

    /// Shape of the tensor stored in this buffer.
    let shape: [Int]

    /// Element count.
    let count: Int

    /// Byte size.
    let byteSize: Int

    /// The Metal device that owns this buffer.
    private let device: MTLDevice

    /// Creates a shared buffer for the given shape.
    ///
    /// Uses `MTLResourceOptions.storageModeShared` which on Apple Silicon
    /// places the buffer in unified memory accessible by both GPU and ANE.
    init?(device: MTLDevice, shape: [Int]) {
        self.device = device
        self.shape = shape
        self.count = shape.reduce(1, *)
        self.byteSize = count * MemoryLayout<Float>.size

        guard byteSize > 0 else { return nil }

        // storageModeShared = unified memory on Apple Silicon
        guard let buffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            return nil
        }
        self.metalBuffer = buffer
    }

    /// Creates a shared buffer and copies initial Float data into it.
    init?(device: MTLDevice, shape: [Int], data: [Float]) {
        self.device = device
        self.shape = shape
        self.count = shape.reduce(1, *)
        self.byteSize = count * MemoryLayout<Float>.size

        guard byteSize > 0, data.count >= count else { return nil }

        guard let buffer = device.makeBuffer(
            bytes: data,
            length: byteSize,
            options: .storageModeShared
        ) else {
            return nil
        }
        self.metalBuffer = buffer
    }

    /// Creates an MLMultiArray that shares the same memory as the Metal buffer.
    ///
    /// The MLMultiArray is backed by the Metal buffer's pointer, so writes
    /// through one are visible through the other — no copy occurs.
    func makeMLMultiArray() throws -> MLMultiArray {
        let ptr = metalBuffer.contents()
        let nsShape = shape.map { NSNumber(value: $0) }

        // Compute strides (row-major)
        var strides = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        let nsStrides = strides.map { NSNumber(value: $0) }

        return try MLMultiArray(
            dataPointer: ptr,
            shape: nsShape,
            dataType: .float32,
            strides: nsStrides
        )
    }

    /// Reads the buffer contents as a Float array.
    func toFloatArray() -> [Float] {
        let ptr = metalBuffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Writes Float data into the buffer.
    func write(_ data: [Float]) {
        let copyCount = min(data.count, count)
        let ptr = metalBuffer.contents().bindMemory(to: Float.self, capacity: count)
        data.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: copyCount)
        }
    }
}
