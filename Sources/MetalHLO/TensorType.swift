// TensorType.swift
// MetalHLO
//
// Describes a tensor's shape and element type.

/// Describes a tensor's shape and element type.
///
/// `TensorType` is used to describe the expected input and output
/// types for compiled executables.
public struct TensorType: Equatable, Sendable, CustomStringConvertible {

    // MARK: - Properties

    /// The tensor shape (dimensions).
    public let shape: [Int]

    /// The element type.
    public let elementType: ElementType

    // MARK: - Computed Properties

    /// The total number of elements.
    public var count: Int {
        shape.reduce(1, *)
    }

    /// The total size in bytes.
    public var byteCount: Int {
        if elementType == .int1 {
            // Bool type: 1 byte per element for storage
            return count
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

    // MARK: - Initialization

    /// Creates a tensor type with the given shape and element type.
    ///
    /// - Parameters:
    ///   - shape: The tensor shape.
    ///   - elementType: The element type.
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
