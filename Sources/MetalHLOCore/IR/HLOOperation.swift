// HLOOperation.swift
// MetalHLOCore
//
// Internal representation of a StableHLO operation.

/// An operation in the function body.
///
/// `HLOOperation` represents a single StableHLO operation with its
/// operands, result, and attributes.
public struct HLOOperation: Sendable {

    /// The result value name (e.g., "%0").
    public let result: String

    /// The operation kind.
    public let kind: HLOOpKind

    /// The operand value names (e.g., ["%arg0", "%arg1"]).
    public let operands: [String]

    /// The result type.
    public let resultType: TensorType

    /// Operation-specific attributes.
    public let attributes: HLOAttributes

    /// Creates a new HLO operation.
    ///
    /// - Parameters:
    ///   - result: The result value name.
    ///   - kind: The operation kind.
    ///   - operands: The operand value names.
    ///   - resultType: The result type.
    ///   - attributes: The operation attributes.
    public init(
        result: String,
        kind: HLOOpKind,
        operands: [String],
        resultType: TensorType,
        attributes: HLOAttributes = HLOAttributes()
    ) {
        self.result = result
        self.kind = kind
        self.operands = operands
        self.resultType = resultType
        self.attributes = attributes
    }
}

extension HLOOperation: CustomStringConvertible {
    public var description: String {
        let operandsStr = operands.joined(separator: ", ")
        return "\(result) = stablehlo.\(kind.rawValue) \(operandsStr) : \(resultType)"
    }
}
