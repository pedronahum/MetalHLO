// HLOFunction.swift
// MetalHLOCore
//
// Internal representation of a function in the HLO module.

/// A function in the HLO module.
///
/// `HLOFunction` represents a parsed StableHLO function with its
/// inputs, operations, and outputs.
public struct HLOFunction: Sendable {

    /// The function name (typically "main").
    public let name: String

    /// The input arguments.
    public let inputs: [HLOArgument]

    /// The output types.
    public let outputTypes: [TensorType]

    /// The operations in topological order.
    public let operations: [HLOOperation]

    /// The return value names (e.g., ["%7"]).
    public let returnValues: [String]

    /// Creates a new HLO function.
    ///
    /// - Parameters:
    ///   - name: The function name.
    ///   - inputs: The input arguments.
    ///   - outputTypes: The output types.
    ///   - operations: The operations.
    ///   - returnValues: The return value names.
    public init(
        name: String,
        inputs: [HLOArgument],
        outputTypes: [TensorType],
        operations: [HLOOperation],
        returnValues: [String]
    ) {
        self.name = name
        self.inputs = inputs
        self.outputTypes = outputTypes
        self.operations = operations
        self.returnValues = returnValues
    }
}

/// A function argument.
public struct HLOArgument: Sendable {

    /// The argument name (e.g., "%arg0").
    public let name: String

    /// The tensor type.
    public let type: TensorType

    /// Creates a new HLO argument.
    ///
    /// - Parameters:
    ///   - name: The argument name.
    ///   - type: The tensor type.
    public init(name: String, type: TensorType) {
        self.name = name
        self.type = type
    }
}

extension HLOFunction: CustomStringConvertible {
    public var description: String {
        let inputsStr = inputs.map { "\($0.name): \($0.type)" }.joined(separator: ", ")
        let outputsStr = outputTypes.map { "\($0)" }.joined(separator: ", ")
        let opsStr = operations.map { "    \($0)" }.joined(separator: "\n")
        let returnStr = returnValues.joined(separator: ", ")

        return """
          func.func @\(name)(\(inputsStr)) -> (\(outputsStr)) {
        \(opsStr)
            return \(returnStr)
          }
        """
    }
}
