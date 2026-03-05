// CoreMLOp.swift
// ANERuntime
//
// Simple operation descriptors for the CoreML compilation bridge.
// Defined in ANERuntime to avoid dependency on MetalHLOCore.
// The caller (e.g. MILEmitter) converts its IR to these types.

import Foundation

/// Describes a single MIL operation for CoreML model building.
public struct CoreMLOp: Sendable {
    /// Result variable name (e.g. "0", "add_result").
    public let result: String

    /// MIL operation name (e.g. "add", "matmul", "conv").
    public let op: String

    /// Output tensor shape.
    public let shape: [Int]

    /// Operation parameters in order.
    public let params: [(String, CoreMLParamValue)]

    public init(
        result: String,
        op: String,
        shape: [Int],
        params: [(String, CoreMLParamValue)]
    ) {
        self.result = result
        self.op = op
        self.shape = shape
        self.params = params
    }
}

/// A parameter value for a CoreML MIL operation.
public enum CoreMLParamValue: Sendable {
    /// Reference to another variable by name.
    case variable(String)
    /// Integer scalar.
    case intValue(Int)
    /// Integer array (e.g. shape, axes, strides).
    case intArray([Int])
    /// Float scalar.
    case floatValue(Double)
    /// Float array.
    case floatArray([Double])
    /// Boolean scalar.
    case boolValue(Bool)
    /// String literal.
    case stringValue(String)
    /// Tuple of variable names (e.g. for concat inputs).
    case variableTuple([String])
}
