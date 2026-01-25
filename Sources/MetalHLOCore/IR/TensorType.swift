// TensorType.swift
// MetalHLOCore
//
// Internal tensor type representation.

import MetalPerformanceShadersGraph

/// Describes a tensor's shape and element type.
///
/// This is the internal representation used by MetalHLOCore.
/// The public API re-exports this type.
public struct TensorType: Equatable, Sendable, CustomStringConvertible {

    // MARK: - Properties

    /// The tensor shape (dimensions).
    public let shape: [Int]

    /// The element type.
    public let elementType: ElementType

    // MARK: - Computed Properties

    /// The total number of elements.
    public var count: Int {
        shape.isEmpty ? 1 : shape.reduce(1, *)
    }

    /// The total size in bytes.
    public var byteCount: Int {
        if elementType == .int1 {
            return count  // 1 byte per bool for storage
        }
        return count * elementType.byteSize
    }

    /// The rank (number of dimensions).
    public var rank: Int {
        shape.count
    }

    /// Whether this is a scalar (rank 0).
    public var isScalar: Bool {
        shape.isEmpty
    }

    /// The shape as NSNumber array for MPSGraph.
    public var mpsShape: [NSNumber] {
        shape.map { NSNumber(value: $0) }
    }

    // MARK: - Initialization

    /// Creates a tensor type with the given shape and element type.
    public init(shape: [Int], elementType: ElementType) {
        self.shape = shape
        self.elementType = elementType
    }

    // MARK: - CustomStringConvertible

    public var description: String {
        if shape.isEmpty {
            return "tensor<\(elementType.rawValue)>"
        }
        let shapeStr = shape.map(String.init).joined(separator: "x")
        return "tensor<\(shapeStr)x\(elementType.rawValue)>"
    }
}

/// Supported element types for tensor data.
public enum ElementType: String, CaseIterable, Sendable {

    // Floating point
    case float16 = "f16"
    case float32 = "f32"
    case float64 = "f64"
    case bfloat16 = "bf16"

    // Signed integers
    case int1 = "i1"
    case int8 = "i8"
    case int16 = "i16"
    case int32 = "i32"
    case int64 = "i64"

    // Unsigned integers
    case uint8 = "ui8"
    case uint16 = "ui16"
    case uint32 = "ui32"
    case uint64 = "ui64"

    /// The size in bytes (0 for i1/bool).
    public var byteSize: Int {
        switch self {
        case .int1: return 0
        case .int8, .uint8: return 1
        case .float16, .bfloat16, .int16, .uint16: return 2
        case .float32, .int32, .uint32: return 4
        case .float64, .int64, .uint64: return 8
        }
    }

    /// The corresponding MPSDataType.
    public var mpsDataType: MPSDataType {
        switch self {
        case .float16: return .float16
        case .float32: return .float32
        case .float64: return .float32  // MPSGraph doesn't support f64 directly
        case .bfloat16: return .bFloat16
        case .int1: return .bool
        case .int8: return .int8
        case .int16: return .int16
        case .int32: return .int32
        case .int64: return .int64
        case .uint8: return .uInt8
        case .uint16: return .uInt16
        case .uint32: return .uInt32
        case .uint64: return .uInt64
        }
    }

    /// Whether this is a floating-point type.
    public var isFloatingPoint: Bool {
        switch self {
        case .float16, .float32, .float64, .bfloat16: return true
        default: return false
        }
    }

    /// Whether this is a signed integer type.
    public var isSignedInteger: Bool {
        switch self {
        case .int1, .int8, .int16, .int32, .int64: return true
        default: return false
        }
    }

    /// Whether this is an unsigned integer type.
    public var isUnsignedInteger: Bool {
        switch self {
        case .uint8, .uint16, .uint32, .uint64: return true
        default: return false
        }
    }
}
