// ANEBuffer.swift
// ANERuntime
//
// IOSurface-backed buffers in ANE's [1, C, 1, S] layout.

import Foundation
import IOSurface

/// Element types supported by ANE.
public enum ANEElementType: String, Sendable {
    case float16
}

/// Describes the shape and layout of an ANE buffer.
public struct ANEBufferDescriptor: Sendable, Equatable {

    /// Logical shape of the tensor (e.g., [batch, channels, height, width]).
    public let shape: [Int]

    /// The element type. ANE only supports FP16 for compute.
    public let elementType: ANEElementType

    /// The ANE physical layout: [1, C, 1, S].
    /// N=1, C=channels, D=1 (collapsed with H), S=spatial (product of remaining dims).
    public let aneShape: [Int]

    /// Total element count.
    public let elementCount: Int

    /// Bytes per row, aligned to 64 bytes.
    public let bytesPerRow: Int

    /// Total allocated bytes including alignment padding.
    public let allocatedBytes: Int

    /// Creates a descriptor from a logical shape.
    ///
    /// Layout mapping:
    /// - `[N]`          → `[1, 1, 1, N]`
    /// - `[M, N]`       → `[1, M, 1, N]`
    /// - `[B, M, N]`    → `[1, M, 1, B*N]`
    /// - `[B, C, H, W]` → `[1, C, 1, B*H*W]`
    public init(shape: [Int], elementType: ANEElementType = .float16) {
        self.shape = shape
        self.elementType = elementType
        self.elementCount = shape.reduce(1, *)

        // Map logical shape to ANE [1, C, 1, S]
        let (c, s): (Int, Int)
        switch shape.count {
        case 0:
            (c, s) = (1, 1)
        case 1:
            (c, s) = (1, shape[0])
        case 2:
            (c, s) = (shape[0], shape[1])
        case 3:
            (c, s) = (shape[1], shape[0] * shape[2])
        default:
            // For 4D+: second dim is C, rest fold into S
            let channels = shape[1]
            let spatial = shape.enumerated()
                .filter { $0.offset != 1 }
                .map { $0.element }
                .reduce(1, *)
            (c, s) = (channels, spatial)
        }
        self.aneShape = [1, c, 1, s]

        // FP16: 2 bytes per element
        let bytesPerElement = 2
        let rawBytesPerRow = s * bytesPerElement
        // Align to 64 bytes
        self.bytesPerRow = (rawBytesPerRow + 63) & ~63
        self.allocatedBytes = self.bytesPerRow * c
    }
}

/// An IOSurface-backed buffer for ANE execution.
///
/// ANE requires IOSurface buffers with specific alignment and layout.
/// This class manages the IOSurface lifecycle and provides methods
/// to populate and read back data.
///
/// IOSurface creation works on all macOS machines, not just those with ANE.
public final class ANEBuffer: @unchecked Sendable {

    /// The descriptor for this buffer.
    public let descriptor: ANEBufferDescriptor

    /// The underlying IOSurface.
    internal let surface: IOSurfaceRef

    private let lock = NSLock()
    private var released = false

    /// Whether this buffer is still valid (not released).
    public var isValid: Bool {
        lock.lock()
        defer { lock.unlock() }
        return !released
    }

    /// Creates a new ANE buffer with the given descriptor.
    /// Returns nil if IOSurface creation fails.
    public init?(descriptor: ANEBufferDescriptor) {
        self.descriptor = descriptor

        let s = descriptor.aneShape[3]

        let props: [String: Any] = [
            kIOSurfaceWidth as String: s,
            kIOSurfaceHeight as String: descriptor.aneShape[1],  // C planes
            kIOSurfaceBytesPerElement as String: 2,
            kIOSurfaceBytesPerRow as String: descriptor.bytesPerRow,
            kIOSurfaceAllocSize as String: descriptor.allocatedBytes,
            kIOSurfacePixelFormat as String: 0x4C303038,  // 'L008' - one component 16-bit
        ]

        guard let surface = IOSurfaceCreate(props as CFDictionary) else {
            return nil
        }
        self.surface = surface
    }

    /// Writes raw FP16 data into the IOSurface buffer.
    /// The data should contain `descriptor.elementCount` FP16 values.
    public func write(_ data: Data) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !released else {
            throw ANEError.bufferError("Buffer has been released")
        }

        let expectedBytes = descriptor.elementCount * 2
        guard data.count >= expectedBytes else {
            throw ANEError.bufferError(
                "Data too small: \(data.count) bytes, expected \(expectedBytes)")
        }

        IOSurfaceLock(surface, [], nil)
        defer { IOSurfaceUnlock(surface, [], nil) }

        let baseAddress = IOSurfaceGetBaseAddress(surface)
        let c = descriptor.aneShape[1]
        let s = descriptor.aneShape[3]
        let bytesPerElement = 2

        // Copy data respecting row stride (which may include padding)
        data.withUnsafeBytes { srcPtr in
            let src = srcPtr.baseAddress!
            for row in 0..<c {
                let srcOffset = row * s * bytesPerElement
                let dstOffset = row * descriptor.bytesPerRow
                let rowBytes = s * bytesPerElement
                memcpy(baseAddress.advanced(by: dstOffset), src.advanced(by: srcOffset), rowBytes)
            }
        }
    }

    /// Writes Float32 data, converting to FP16.
    /// - Parameters:
    ///   - data: Float32 values in logical shape order.
    ///   - shape: Logical shape (must match descriptor's shape).
    public func writeFloat32(_ data: [Float], shape: [Int]) throws {
        guard shape == descriptor.shape else {
            throw ANEError.bufferError(
                "Shape mismatch: expected \(descriptor.shape), got \(shape)")
        }
        guard data.count == descriptor.elementCount else {
            throw ANEError.bufferError(
                "Element count mismatch: expected \(descriptor.elementCount), got \(data.count)")
        }

        // Convert Float32 → Float16
        let fp16Data = data.map { Float16($0) }
        let rawData = fp16Data.withUnsafeBufferPointer { bufPtr in
            Data(buffer: bufPtr)
        }
        try write(rawData)
    }

    /// Reads raw FP16 data from the IOSurface buffer.
    public func readFP16() throws -> Data {
        lock.lock()
        defer { lock.unlock() }
        guard !released else {
            throw ANEError.bufferError("Buffer has been released")
        }

        IOSurfaceLock(surface, .readOnly, nil)
        defer { IOSurfaceUnlock(surface, .readOnly, nil) }

        let baseAddress = IOSurfaceGetBaseAddress(surface)
        let c = descriptor.aneShape[1]
        let s = descriptor.aneShape[3]
        let bytesPerElement = 2
        let totalBytes = descriptor.elementCount * bytesPerElement

        var result = Data(count: totalBytes)
        result.withUnsafeMutableBytes { dstPtr in
            let dst = dstPtr.baseAddress!
            for row in 0..<c {
                let srcOffset = row * descriptor.bytesPerRow
                let dstOffset = row * s * bytesPerElement
                let rowBytes = s * bytesPerElement
                memcpy(dst.advanced(by: dstOffset), baseAddress.advanced(by: srcOffset), rowBytes)
            }
        }
        return result
    }

    /// Reads data from the buffer, converting FP16 to Float32.
    public func readFloat32() throws -> [Float] {
        let fp16Data = try readFP16()
        return fp16Data.withUnsafeBytes { rawPtr in
            let fp16Ptr = rawPtr.bindMemory(to: Float16.self)
            return fp16Ptr.map { Float($0) }
        }
    }

    /// Releases the IOSurface. Called automatically in deinit.
    public func release() {
        lock.lock()
        defer { lock.unlock() }
        released = true
        // IOSurfaceRef is a CFTypeRef managed by ARC — setting released flag
        // prevents further use. The actual surface is freed when this object deinits.
    }

    deinit {
        released = true
    }
}
