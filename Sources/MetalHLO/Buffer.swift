// Buffer.swift
// MetalHLO
//
// A device-side buffer containing tensor data.

import Foundation
import MetalHLOCore

/// A device-side buffer containing tensor data.
///
/// `Buffer` wraps a Metal buffer and provides methods to transfer
/// data between the device and host.
///
/// ## Example
/// ```swift
/// let buffer = try client.createBuffer([1.0, 2.0, 3.0], shape: [3], elementType: .float32)
/// let data = try buffer.toFloatArray()
/// ```
public final class Buffer: @unchecked Sendable {

    // MARK: - Properties

    internal let storage: BufferStorage

    /// The tensor shape.
    public var shape: [Int] {
        storage.shape
    }

    /// The number of elements.
    public var count: Int {
        storage.count
    }

    /// The element type.
    public var elementType: ElementType {
        ElementType.fromCoreType(storage.elementType)
    }

    /// The size in bytes.
    public var byteCount: Int {
        storage.byteCount
    }

    // MARK: - Initialization

    internal init(storage: BufferStorage) {
        self.storage = storage
    }

    // MARK: - Data Transfer

    /// Copies buffer contents to host as a Float array.
    ///
    /// - Throws: `MetalHLOError.transferFailed` on failure.
    /// - Returns: An array of float values.
    public func toFloatArray() throws -> [Float] {
        try storage.toArray(as: Float.self)
    }

    /// Copies buffer contents to host as a Float16 array.
    ///
    /// - Throws: `MetalHLOError.transferFailed` on failure.
    /// - Returns: An array of Float16 values.
    @available(macOS 14.0, *)
    public func toFloat16Array() throws -> [Float16] {
        try storage.toArray(as: Float16.self)
    }

    /// Copies buffer contents to host as an Int32 array.
    ///
    /// - Throws: `MetalHLOError.transferFailed` on failure.
    /// - Returns: An array of Int32 values.
    public func toInt32Array() throws -> [Int32] {
        try storage.toArray(as: Int32.self)
    }

    /// Copies buffer contents to host as an Int64 array.
    ///
    /// - Throws: `MetalHLOError.transferFailed` on failure.
    /// - Returns: An array of Int64 values.
    public func toInt64Array() throws -> [Int64] {
        try storage.toArray(as: Int64.self)
    }

    /// Copies buffer contents to host as a Bool array.
    ///
    /// - Throws: `MetalHLOError.transferFailed` on failure.
    /// - Returns: An array of Bool values.
    public func toBoolArray() throws -> [Bool] {
        try storage.toBoolArray()
    }

    /// Copies buffer contents to raw bytes.
    ///
    /// - Throws: `MetalHLOError.transferFailed` on failure.
    /// - Returns: The raw byte data.
    public func toData() throws -> Data {
        try storage.toData()
    }
}
