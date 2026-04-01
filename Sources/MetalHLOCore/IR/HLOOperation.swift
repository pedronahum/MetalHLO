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

    /// Number of results produced by this operation (1 for most ops, >1 for while/call).
    /// Multi-result ops store outputs as "%result.0", "%result.1", etc.
    public let resultCount: Int

    /// Creates a new HLO operation.
    ///
    /// - Parameters:
    ///   - result: The result value name.
    ///   - kind: The operation kind.
    ///   - operands: The operand value names.
    ///   - resultType: The result type.
    ///   - attributes: The operation attributes.
    ///   - resultCount: Number of results (default 1).
    public init(
        result: String,
        kind: HLOOpKind,
        operands: [String],
        resultType: TensorType,
        attributes: HLOAttributes = HLOAttributes(),
        resultCount: Int = 1
    ) {
        self.result = result
        self.kind = kind
        self.operands = operands
        self.resultType = resultType
        self.attributes = attributes
        self.resultCount = resultCount
    }
}

extension HLOOperation: CustomStringConvertible {
    public var description: String {
        let operandsStr = operands.joined(separator: ", ")
        return "\(result) = stablehlo.\(kind.rawValue) \(operandsStr) : \(resultType)"
    }
}
