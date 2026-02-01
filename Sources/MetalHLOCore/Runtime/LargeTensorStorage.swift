// LargeTensorStorage.swift
// MetalHLOCore
//
// Direct MTLBuffer allocation for large tensors, bypassing Swift Data overhead.

import Foundation
import Metal

/// Storage for large tensors using direct MTLBuffer allocation.
///
/// For tensors larger than `threshold` bytes, this class bypasses Swift's `Data`
/// type and allocates directly to Metal shared memory. This eliminates:
/// - Swift memory management overhead
/// - Double-copy through intermediate Data objects
/// - Allocation pressure on Swift's heap
///
/// Example:
/// ```swift
/// let storage = try LargeTensorStorage(
///     device: device,
///     shape: [16384, 16384],
///     elementType: .float32
/// )
/// // Populate data directly
/// storage.withUnsafeMutableBytes { ptr in
///     // Fill buffer directly
/// }
/// ```
public final class LargeTensorStorage: @unchecked Sendable {

    // MARK: - Threshold Configuration

    /// Threshold in bytes above which tensors use direct MTLBuffer allocation.
    /// Default: 64MB - below this, Swift Data overhead is negligible.
    public static nonisolated(unsafe) var threshold: Int = 64 * 1024 * 1024

    // MARK: - Properties

    /// The underlying Metal buffer.
    public let buffer: MTLBuffer

    /// The tensor shape.
    public let shape: [Int]

    /// The element type.
    public let elementType: ElementType

    /// The number of elements.
    public var count: Int {
        shape.isEmpty ? 1 : shape.reduce(1, *)
    }

    /// The size in bytes.
    public var byteCount: Int {
        buffer.length
    }

    // MARK: - Initialization

    /// Creates uninitialized storage for a large tensor.
    ///
    /// - Parameters:
    ///   - device: Metal device for buffer allocation.
    ///   - shape: The tensor shape.
    ///   - elementType: The element type.
    /// - Throws: `LargeTensorError.allocationFailed` if buffer allocation fails.
    public init(
        device: MTLDevice,
        shape: [Int],
        elementType: ElementType
    ) throws {
        self.shape = shape
        self.elementType = elementType

        let count = shape.isEmpty ? 1 : shape.reduce(1, *)
        let byteSize: Int
        if elementType == .int1 {
            byteSize = count  // 1 byte per bool
        } else {
            byteSize = count * elementType.byteSize
        }

        // Use storageModeShared for CPU/GPU access without explicit copy
        guard let buffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            throw LargeTensorError.allocationFailed(size: byteSize)
        }

        self.buffer = buffer
    }

    /// Creates storage from Float data, copying directly to GPU buffer.
    ///
    /// - Parameters:
    ///   - device: Metal device for buffer allocation.
    ///   - data: Float array to copy.
    ///   - shape: The tensor shape.
    /// - Throws: `LargeTensorError.allocationFailed` if buffer allocation fails.
    public init(
        device: MTLDevice,
        floatData data: [Float],
        shape: [Int]
    ) throws {
        self.shape = shape
        self.elementType = .float32

        let byteSize = data.count * MemoryLayout<Float>.stride

        guard let buffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            throw LargeTensorError.allocationFailed(size: byteSize)
        }

        // Direct copy to buffer
        _ = data.withUnsafeBytes { src in
            memcpy(buffer.contents(), src.baseAddress!, byteSize)
        }

        self.buffer = buffer
    }

    /// Creates storage from Int32 data, copying directly to GPU buffer.
    public init(
        device: MTLDevice,
        int32Data data: [Int32],
        shape: [Int]
    ) throws {
        self.shape = shape
        self.elementType = .int32

        let byteSize = data.count * MemoryLayout<Int32>.stride

        guard let buffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            throw LargeTensorError.allocationFailed(size: byteSize)
        }

        _ = data.withUnsafeBytes { src in
            memcpy(buffer.contents(), src.baseAddress!, byteSize)
        }

        self.buffer = buffer
    }

    /// Creates storage from raw bytes, copying directly to GPU buffer.
    ///
    /// - Parameters:
    ///   - device: Metal device for buffer allocation.
    ///   - bytes: Raw byte data.
    ///   - shape: The tensor shape.
    ///   - elementType: The element type.
    /// - Throws: `LargeTensorError.allocationFailed` if buffer allocation fails.
    public init(
        device: MTLDevice,
        bytes: Data,
        shape: [Int],
        elementType: ElementType
    ) throws {
        self.shape = shape
        self.elementType = elementType

        guard let buffer = device.makeBuffer(length: bytes.count, options: .storageModeShared) else {
            throw LargeTensorError.allocationFailed(size: bytes.count)
        }

        _ = bytes.withUnsafeBytes { src in
            memcpy(buffer.contents(), src.baseAddress!, bytes.count)
        }

        self.buffer = buffer
    }

    /// Creates storage wrapping an existing MTLBuffer.
    ///
    /// The buffer is NOT copied - this creates a view over the existing buffer.
    /// The caller is responsible for ensuring the buffer remains valid.
    ///
    /// - Parameters:
    ///   - buffer: Existing Metal buffer.
    ///   - shape: The tensor shape.
    ///   - elementType: The element type.
    public init(
        buffer: MTLBuffer,
        shape: [Int],
        elementType: ElementType
    ) {
        self.buffer = buffer
        self.shape = shape
        self.elementType = elementType
    }

    // MARK: - Data Access

    /// Provides direct read access to buffer contents.
    public func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        let ptr = UnsafeRawBufferPointer(start: buffer.contents(), count: buffer.length)
        return try body(ptr)
    }

    /// Provides direct write access to buffer contents.
    public func withUnsafeMutableBytes<R>(_ body: (UnsafeMutableRawBufferPointer) throws -> R) rethrows -> R {
        let ptr = UnsafeMutableRawBufferPointer(start: buffer.contents(), count: buffer.length)
        return try body(ptr)
    }

    /// Converts buffer contents to an array.
    /// Note: This creates a copy - use `withUnsafeBytes` for zero-copy access.
    public func toArray<T>(as type: T.Type) -> [T] {
        withUnsafeBytes { buffer in
            let typedBuffer = buffer.bindMemory(to: T.self)
            return Array(typedBuffer)
        }
    }

    /// Converts to Swift Data.
    /// Note: This creates a copy - use `withUnsafeBytes` for zero-copy access.
    public func toData() -> Data {
        Data(bytes: buffer.contents(), count: buffer.length)
    }

    // MARK: - Static Helpers

    /// Checks if a tensor size exceeds the threshold for large tensor storage.
    public static func shouldUseLargeTensorStorage(shape: [Int], elementType: ElementType) -> Bool {
        let count = shape.isEmpty ? 1 : shape.reduce(1, *)
        let byteSize = elementType == .int1 ? count : count * elementType.byteSize
        return byteSize >= threshold
    }
}

// MARK: - Errors

/// Errors that can occur during large tensor operations.
public enum LargeTensorError: Error, Sendable, CustomStringConvertible {
    case allocationFailed(size: Int)
    case poolExhausted
    case invalidSize(expected: Int, got: Int)

    public var description: String {
        switch self {
        case .allocationFailed(let size):
            return "Failed to allocate large tensor buffer of size \(size) bytes (\(size / (1024 * 1024)) MB)"
        case .poolExhausted:
            return "Large tensor pool exhausted and new allocation failed"
        case .invalidSize(let expected, let got):
            return "Invalid tensor size: expected \(expected) bytes, got \(got) bytes"
        }
    }
}
