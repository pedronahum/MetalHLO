// BufferStorage.swift
// MetalHLOCore
//
// Internal storage for device buffers.

import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Internal storage for a device buffer.
///
/// `BufferStorage` wraps the underlying Metal buffer and provides
/// methods for data transfer and access.
public final class BufferStorage: @unchecked Sendable {

    // MARK: - Properties

    /// The tensor shape.
    public let shape: [Int]

    /// The element type.
    public let elementType: ElementType

    /// The underlying data.
    internal private(set) var data: Data

    /// The Metal device.
    private let device: MTLDevice

    // MARK: - Computed Properties

    /// The number of elements.
    public var count: Int {
        shape.isEmpty ? 1 : shape.reduce(1, *)
    }

    /// The size in bytes.
    public var byteCount: Int {
        data.count
    }

    // MARK: - Initialization

    /// Creates storage from numeric data.
    public init<T: Numeric>(
        data: [T],
        shape: [Int],
        elementType: ElementType,
        device: MTLDevice
    ) throws {
        self.shape = shape
        self.elementType = elementType
        self.device = device

        // Convert data to the appropriate format
        self.data = try Self.convertToData(data, elementType: elementType)
    }

    /// Creates storage from raw bytes.
    public init(
        bytes data: Data,
        shape: [Int],
        elementType: ElementType,
        device: MTLDevice
    ) throws {
        self.shape = shape
        self.elementType = elementType
        self.device = device
        self.data = data
    }

    /// Creates uninitialized storage.
    public init(
        shape: [Int],
        elementType: ElementType,
        device: MTLDevice
    ) throws {
        self.shape = shape
        self.elementType = elementType
        self.device = device

        let count = shape.isEmpty ? 1 : shape.reduce(1, *)
        let byteSize: Int
        if elementType == .int1 {
            byteSize = count  // 1 byte per bool
        } else {
            byteSize = count * elementType.byteSize
        }

        self.data = Data(count: byteSize)
    }

    /// Creates storage from MPSGraph tensor data.
    public init(
        tensorData: MPSGraphTensorData,
        type: TensorType,
        device: MTLDevice
    ) throws {
        self.shape = type.shape
        self.elementType = type.elementType
        self.device = device

        // Extract data from tensor
        let byteCount = type.byteCount
        var bytes = [UInt8](repeating: 0, count: byteCount)
        tensorData.mpsndarray().readBytes(&bytes, strideBytes: nil)
        self.data = Data(bytes)
    }

    // MARK: - Data Conversion

    private static func convertToData<T: Numeric>(
        _ values: [T],
        elementType: ElementType
    ) throws -> Data {
        switch elementType {
        case .float32:
            var floats = values.map { floatValue($0) }
            return Data(bytes: &floats, count: floats.count * MemoryLayout<Float>.stride)

        case .float16:
            var floats = values.map { Float16(floatValue($0)) }
            return Data(bytes: &floats, count: floats.count * MemoryLayout<Float16>.stride)

        case .float64:
            var doubles = values.map { Double(floatValue($0)) }
            return Data(bytes: &doubles, count: doubles.count * MemoryLayout<Double>.stride)

        case .bfloat16:
            // BFloat16 not directly supported, convert through Float
            var floats = values.map { floatValue($0) }
            return Data(bytes: &floats, count: floats.count * MemoryLayout<Float>.stride)

        case .int1:
            var bools = values.map { intValue($0) != 0 ? UInt8(1) : UInt8(0) }
            return Data(bytes: &bools, count: bools.count)

        case .int8:
            var ints = values.map { Int8(clamping: intValue($0)) }
            return Data(bytes: &ints, count: ints.count)

        case .int16:
            var ints = values.map { Int16(clamping: intValue($0)) }
            return Data(bytes: &ints, count: ints.count * MemoryLayout<Int16>.stride)

        case .int32:
            var ints = values.map { Int32(clamping: intValue($0)) }
            return Data(bytes: &ints, count: ints.count * MemoryLayout<Int32>.stride)

        case .int64:
            var ints = values.map { Int64(intValue($0)) }
            return Data(bytes: &ints, count: ints.count * MemoryLayout<Int64>.stride)

        case .uint8:
            var ints = values.map { UInt8(clamping: intValue($0)) }
            return Data(bytes: &ints, count: ints.count)

        case .uint16:
            var ints = values.map { UInt16(clamping: intValue($0)) }
            return Data(bytes: &ints, count: ints.count * MemoryLayout<UInt16>.stride)

        case .uint32:
            var ints = values.map { UInt32(clamping: intValue($0)) }
            return Data(bytes: &ints, count: ints.count * MemoryLayout<UInt32>.stride)

        case .uint64:
            var ints = values.map { UInt64(intValue($0)) }
            return Data(bytes: &ints, count: ints.count * MemoryLayout<UInt64>.stride)
        }
    }

    private static func floatValue<T: Numeric>(_ value: T) -> Float {
        if let f = value as? Float { return f }
        if let d = value as? Double { return Float(d) }
        if let i = value as? Int { return Float(i) }
        if let i = value as? Int32 { return Float(i) }
        if let i = value as? Int64 { return Float(i) }
        if let f16 = value as? Float16 { return Float(f16) }
        return 0
    }

    private static func intValue<T: Numeric>(_ value: T) -> Int {
        if let i = value as? Int { return i }
        if let i = value as? Int32 { return Int(i) }
        if let i = value as? Int64 { return Int(i) }
        if let f = value as? Float { return Int(f) }
        if let d = value as? Double { return Int(d) }
        if let f16 = value as? Float16 { return Int(f16) }
        return 0
    }

    // MARK: - Data Access

    /// Converts buffer contents to an array of the specified type.
    public func toArray<T>(as type: T.Type) throws -> [T] {
        data.withUnsafeBytes { buffer in
            let typedBuffer = buffer.bindMemory(to: T.self)
            return Array(typedBuffer)
        }
    }

    /// Converts buffer contents to a Bool array.
    public func toBoolArray() throws -> [Bool] {
        let bytes: [UInt8] = try toArray(as: UInt8.self)
        return bytes.map { $0 != 0 }
    }

    /// Returns the raw data.
    public func toData() throws -> Data {
        data
    }
}
