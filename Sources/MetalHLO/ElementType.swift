// ElementType.swift
// MetalHLO
//
// Supported element types for tensor data.

import MetalPerformanceShadersGraph
import MetalHLOCore

/// Supported element types for tensor data.
///
/// MetalHLO supports a variety of floating-point, signed integer,
/// and unsigned integer types.
public enum ElementType: String, CaseIterable, Sendable {

    // MARK: - Floating Point

    /// 16-bit floating point (half precision).
    case float16 = "f16"

    /// 32-bit floating point (single precision).
    case float32 = "f32"

    /// 64-bit floating point (double precision).
    case float64 = "f64"

    /// Brain floating point (16-bit with 8-bit exponent).
    case bfloat16 = "bf16"

    // MARK: - Signed Integers

    /// 1-bit integer (boolean).
    case int1 = "i1"

    /// 8-bit signed integer.
    case int8 = "i8"

    /// 16-bit signed integer.
    case int16 = "i16"

    /// 32-bit signed integer.
    case int32 = "i32"

    /// 64-bit signed integer.
    case int64 = "i64"

    // MARK: - Unsigned Integers

    /// 8-bit unsigned integer.
    case uint8 = "ui8"

    /// 16-bit unsigned integer.
    case uint16 = "ui16"

    /// 32-bit unsigned integer.
    case uint32 = "ui32"

    /// 64-bit unsigned integer.
    case uint64 = "ui64"

    // MARK: - Properties

    /// The size in bytes (0 for i1/bool).
    public var byteSize: Int {
        switch self {
        case .int1:
            return 0  // Packed bits
        case .int8, .uint8:
            return 1
        case .float16, .bfloat16, .int16, .uint16:
            return 2
        case .float32, .int32, .uint32:
            return 4
        case .float64, .int64, .uint64:
            return 8
        }
    }

    /// The corresponding MPSDataType.
    public var mpsDataType: MPSDataType {
        switch self {
        case .float16:
            return .float16
        case .float32:
            return .float32
        case .float64:
            // MPSGraph doesn't support float64 directly, will need conversion
            return .float32
        case .bfloat16:
            return .bFloat16
        case .int1:
            return .bool
        case .int8:
            return .int8
        case .int16:
            return .int16
        case .int32:
            return .int32
        case .int64:
            return .int64
        case .uint8:
            return .uInt8
        case .uint16:
            return .uInt16
        case .uint32:
            return .uInt32
        case .uint64:
            return .uInt64
        }
    }

    /// Whether this is a floating-point type.
    public var isFloatingPoint: Bool {
        switch self {
        case .float16, .float32, .float64, .bfloat16:
            return true
        default:
            return false
        }
    }

    /// Whether this is a signed integer type.
    public var isSignedInteger: Bool {
        switch self {
        case .int1, .int8, .int16, .int32, .int64:
            return true
        default:
            return false
        }
    }

    /// Whether this is an unsigned integer type.
    public var isUnsignedInteger: Bool {
        switch self {
        case .uint8, .uint16, .uint32, .uint64:
            return true
        default:
            return false
        }
    }

    // MARK: - Core Conversion

    /// Converts to the core ElementType.
    internal func toCoreType() -> MetalHLOCore.ElementType {
        switch self {
        case .float16: return .float16
        case .float32: return .float32
        case .float64: return .float64
        case .bfloat16: return .bfloat16
        case .int1: return .int1
        case .int8: return .int8
        case .int16: return .int16
        case .int32: return .int32
        case .int64: return .int64
        case .uint8: return .uint8
        case .uint16: return .uint16
        case .uint32: return .uint32
        case .uint64: return .uint64
        }
    }

    /// Creates from a core ElementType.
    internal static func fromCoreType(_ coreType: MetalHLOCore.ElementType) -> ElementType {
        switch coreType {
        case .float16: return .float16
        case .float32: return .float32
        case .float64: return .float64
        case .bfloat16: return .bfloat16
        case .int1: return .int1
        case .int8: return .int8
        case .int16: return .int16
        case .int32: return .int32
        case .int64: return .int64
        case .uint8: return .uint8
        case .uint16: return .uint16
        case .uint32: return .uint32
        case .uint64: return .uint64
        }
    }
}
