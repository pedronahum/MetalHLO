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
///
/// For large tensors (> 64MB by default), storage automatically uses
/// direct MTLBuffer allocation to avoid Swift Data overhead.
public final class BufferStorage: @unchecked Sendable {

    // MARK: - Storage Mode

    /// How the data is stored.
    private enum StorageMode {
        /// Traditional Swift Data storage (for smaller tensors).
        case data(Data)
        /// Direct MTLBuffer storage (for large tensors).
        case largeTensor(LargeTensorStorage)
    }

    // MARK: - Properties

    /// The tensor shape.
    public let shape: [Int]

    /// The element type.
    public let elementType: ElementType

    /// The underlying storage.
    private let storage: StorageMode

    /// The Metal device.
    private let device: MTLDevice

    /// The underlying data (for compatibility with existing code).
    internal var data: Data {
        switch storage {
        case .data(let data):
            return data
        case .largeTensor(let largeTensor):
            return largeTensor.toData()
        }
    }

    /// Returns the underlying MTLBuffer if using large tensor storage.
    public var metalBuffer: MTLBuffer? {
        switch storage {
        case .data:
            return nil
        case .largeTensor(let largeTensor):
            return largeTensor.buffer
        }
    }

    /// Whether this storage uses direct MTLBuffer allocation.
    public var usesDirectBuffer: Bool {
        switch storage {
        case .data: return false
        case .largeTensor: return true
        }
    }

    // MARK: - Computed Properties

    /// The number of elements.
    public var count: Int {
        shape.isEmpty ? 1 : shape.reduce(1, *)
    }

    /// The size in bytes.
    public var byteCount: Int {
        switch storage {
        case .data(let data):
            return data.count
        case .largeTensor(let largeTensor):
            return largeTensor.byteCount
        }
    }

    // MARK: - Initialization

    /// Creates storage wrapping a pre-filled LargeTensorStorage.
    /// Use this to avoid creating intermediate Swift arrays for large tensors.
    public init(largeTensor: LargeTensorStorage, device: MTLDevice) {
        self.shape = largeTensor.shape
        self.elementType = largeTensor.elementType
        self.device = device
        self.storage = .largeTensor(largeTensor)
    }

    /// Creates storage from Float data (optimized fast path).
    /// This avoids expensive per-element type conversion.
    /// For large tensors, uses direct MTLBuffer allocation.
    public init(
        floatData data: [Float],
        shape: [Int],
        device: MTLDevice
    ) {
        self.shape = shape
        self.elementType = .float32
        self.device = device

        // Check if we should use large tensor storage
        let byteSize = data.count * MemoryLayout<Float>.stride
        if byteSize >= LargeTensorStorage.threshold {
            // Use direct MTLBuffer allocation for large tensors
            if let largeTensor = try? LargeTensorStorage(device: device, floatData: data, shape: shape) {
                self.storage = .largeTensor(largeTensor)
            } else {
                // Fallback to Data if allocation fails
                self.storage = .data(data.withUnsafeBytes { Data($0) })
            }
        } else {
            // Direct byte copy - no conversion needed
            self.storage = .data(data.withUnsafeBytes { Data($0) })
        }
    }

    /// Creates storage from Int32 data (optimized fast path).
    /// For large tensors, uses direct MTLBuffer allocation.
    public init(
        int32Data data: [Int32],
        shape: [Int],
        device: MTLDevice
    ) {
        self.shape = shape
        self.elementType = .int32
        self.device = device

        // Check if we should use large tensor storage
        let byteSize = data.count * MemoryLayout<Int32>.stride
        if byteSize >= LargeTensorStorage.threshold {
            if let largeTensor = try? LargeTensorStorage(device: device, int32Data: data, shape: shape) {
                self.storage = .largeTensor(largeTensor)
            } else {
                self.storage = .data(data.withUnsafeBytes { Data($0) })
            }
        } else {
            self.storage = .data(data.withUnsafeBytes { Data($0) })
        }
    }

    /// Creates storage from Int64 data (optimized fast path).
    public init(
        int64Data data: [Int64],
        shape: [Int],
        device: MTLDevice
    ) {
        self.shape = shape
        self.elementType = .int64
        self.device = device

        // Check if we should use large tensor storage
        let byteSize = data.count * MemoryLayout<Int64>.stride
        if byteSize >= LargeTensorStorage.threshold {
            if let largeTensor = try? LargeTensorStorage(device: device, bytes: data.withUnsafeBytes { Data($0) }, shape: shape, elementType: .int64) {
                self.storage = .largeTensor(largeTensor)
            } else {
                self.storage = .data(data.withUnsafeBytes { Data($0) })
            }
        } else {
            self.storage = .data(data.withUnsafeBytes { Data($0) })
        }
    }

    /// Creates storage from numeric data (generic, slower path with type conversion).
    public init<T: Numeric>(
        data: [T],
        shape: [Int],
        elementType: ElementType,
        device: MTLDevice
    ) throws {
        self.shape = shape
        self.elementType = elementType
        self.device = device

        // Determine byte size for large tensor check
        let count = shape.isEmpty ? 1 : shape.reduce(1, *)
        let byteSize = elementType == .int1 ? count : count * elementType.byteSize
        let useLargeTensor = byteSize >= LargeTensorStorage.threshold

        // Fast path: if T is already the correct type, use direct copy
        let rawData: Data
        if elementType == .float32, let floatData = data as? [Float] {
            rawData = floatData.withUnsafeBytes { Data($0) }
        } else if elementType == .int32, let int32Data = data as? [Int32] {
            rawData = int32Data.withUnsafeBytes { Data($0) }
        } else if elementType == .int64, let int64Data = data as? [Int64] {
            rawData = int64Data.withUnsafeBytes { Data($0) }
        } else if elementType == .float64, let doubleData = data as? [Double] {
            rawData = doubleData.withUnsafeBytes { Data($0) }
        } else {
            // Slow path: convert data to the appropriate format
            rawData = try Self.convertToData(data, elementType: elementType)
        }

        if useLargeTensor {
            if let largeTensor = try? LargeTensorStorage(device: device, bytes: rawData, shape: shape, elementType: elementType) {
                self.storage = .largeTensor(largeTensor)
            } else {
                self.storage = .data(rawData)
            }
        } else {
            self.storage = .data(rawData)
        }
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

        // Use large tensor storage for large data
        if data.count >= LargeTensorStorage.threshold {
            if let largeTensor = try? LargeTensorStorage(device: device, bytes: data, shape: shape, elementType: elementType) {
                self.storage = .largeTensor(largeTensor)
            } else {
                self.storage = .data(data)
            }
        } else {
            self.storage = .data(data)
        }
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

        // Use large tensor storage for large allocations
        if byteSize >= LargeTensorStorage.threshold {
            if let largeTensor = try? LargeTensorStorage(device: device, shape: shape, elementType: elementType) {
                self.storage = .largeTensor(largeTensor)
            } else {
                self.storage = .data(Data(count: byteSize))
            }
        } else {
            self.storage = .data(Data(count: byteSize))
        }
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

        // Use large tensor storage for large outputs
        if byteCount >= LargeTensorStorage.threshold {
            if let largeTensor = try? LargeTensorStorage(device: device, shape: type.shape, elementType: type.elementType) {
                // Read directly into the large tensor buffer
                tensorData.mpsndarray().readBytes(largeTensor.buffer.contents(), strideBytes: nil)
                self.storage = .largeTensor(largeTensor)
            } else {
                // Fallback to Data
                var bytes = [UInt8](repeating: 0, count: byteCount)
                tensorData.mpsndarray().readBytes(&bytes, strideBytes: nil)
                self.storage = .data(Data(bytes))
            }
        } else {
            var bytes = [UInt8](repeating: 0, count: byteCount)
            tensorData.mpsndarray().readBytes(&bytes, strideBytes: nil)
            self.storage = .data(Data(bytes))
        }
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
        switch storage {
        case .data(let data):
            return data.withUnsafeBytes { buffer in
                let typedBuffer = buffer.bindMemory(to: T.self)
                return Array(typedBuffer)
            }
        case .largeTensor(let largeTensor):
            return largeTensor.toArray(as: type)
        }
    }

    /// Converts buffer contents to a Bool array.
    public func toBoolArray() throws -> [Bool] {
        let bytes: [UInt8] = try toArray(as: UInt8.self)
        return bytes.map { $0 != 0 }
    }

    /// Returns the raw data.
    public func toData() throws -> Data {
        switch storage {
        case .data(let data):
            return data
        case .largeTensor(let largeTensor):
            return largeTensor.toData()
        }
    }

    /// Provides direct read access to buffer contents (zero-copy for large tensors).
    public func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        switch storage {
        case .data(let data):
            return try data.withUnsafeBytes(body)
        case .largeTensor(let largeTensor):
            return try largeTensor.withUnsafeBytes(body)
        }
    }
}
