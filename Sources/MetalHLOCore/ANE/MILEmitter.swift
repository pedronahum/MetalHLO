// MILEmitter.swift
// MetalHLOCore
//
// Emits MIL (Machine Learning Intermediate Language) programs from
// MetalHLO's internal IR. The output is ready to be compiled by
// ANEDevice.compile().

import Foundation

/// The result of MIL emission: text + weight blob.
public struct MILProgram: Sendable {
    /// The MIL program text (UTF-8), ready to pass to ANEDevice.compile().
    public let milText: String

    /// Weight data dictionary. Key is blob name, value is concatenated
    /// FP16 binary data. For BLOBFILE references in the MIL text.
    public let weights: [String: Data]

    /// Input shapes in function signature order (logical shapes).
    public let inputShapes: [[Int]]

    /// Output shapes (logical shapes).
    public let outputShapes: [[Int]]
}

/// Errors during MIL emission.
///
/// These are programming/IR errors, not runtime errors. Since invalid MIL
/// crashes the ANE compiler (SIGABRT), these errors prevent emission
/// rather than allowing bad MIL through.
public enum MILEmitterError: Error, Sendable, CustomStringConvertible {
    /// The operation is not supported for MIL emission.
    case unsupportedOperation(HLOOpKind, String)
    /// The element type is not supported on ANE.
    case unsupportedElementType(ElementType, String)
    /// Invalid shape for MIL emission.
    case invalidShape(String)
    /// Missing or invalid operation attributes.
    case invalidAttributes(String)
    /// Internal emitter error.
    case internalError(String)

    public var description: String {
        switch self {
        case .unsupportedOperation(let kind, let reason):
            return "Unsupported MIL operation \(kind): \(reason)"
        case .unsupportedElementType(let type, let reason):
            return "Unsupported element type \(type): \(reason)"
        case .invalidShape(let msg):
            return "Invalid shape: \(msg)"
        case .invalidAttributes(let msg):
            return "Invalid attributes: \(msg)"
        case .internalError(let msg):
            return "Internal MIL emitter error: \(msg)"
        }
    }
}

/// Emits MIL (Core ML Intermediate Language) text from MetalHLO's IR.
///
/// The emitter translates an `HLOFunction` into a complete MIL program
/// string plus weight data, suitable for `ANEDevice.compile()`.
///
/// Thread-safe. Stateless per-emission — all state lives in the `emit()` call.
///
/// ```swift
/// let emitter = MILEmitter()
/// let program = try emitter.emit(function: hloFunction)
/// let aneProgram = try device.compile(
///     milProgram: program.milText,
///     weights: program.weights,
///     inputDescriptors: ...,
///     outputDescriptors: ...
/// )
/// ```
public final class MILEmitter: @unchecked Sendable {

    public init() {}

    /// Emits a MIL program from an HLO function.
    ///
    /// - Parameter function: The HLO function to translate.
    /// - Throws: `MILEmitterError` if the function contains unsupported ops
    ///   or invalid attributes.
    /// - Returns: A `MILProgram` with MIL text and weight data.
    public func emit(function: HLOFunction) throws -> MILProgram {
        let builder = MILTextBuilder()
        let weightPacker = MILWeightPacker()

        // Track shapes for operand resolution
        var operandShapes: [String: [Int]] = [:]

        // Emit function arguments
        for arg in function.inputs {
            let name = MILTypeMapper.sanitizeName(arg.name)
            builder.addFunctionArg(name: name, shape: arg.type.shape)
            operandShapes[arg.name] = arg.type.shape
        }

        // Translate each operation
        for op in function.operations {
            try MILOpTranslator.translate(
                op: op,
                builder: builder,
                weightPacker: weightPacker,
                operandShapes: &operandShapes
            )
        }

        // Set return value
        guard let returnValue = function.returnValues.first else {
            throw MILEmitterError.internalError("Function has no return values")
        }
        builder.setReturn(name: MILTypeMapper.sanitizeName(returnValue))

        // Build final MIL text
        let milText = builder.build()

        // Build weights dictionary
        var weights: [String: Data] = [:]
        if weightPacker.hasWeights {
            weights["weights"] = weightPacker.getWeightData()
        }

        return MILProgram(
            milText: milText,
            weights: weights,
            inputShapes: function.inputs.map { $0.type.shape },
            outputShapes: function.outputTypes.map { $0.shape }
        )
    }

    /// Checks whether all operations in a function are supported for MIL emission.
    ///
    /// - Parameter function: The HLO function to check.
    /// - Returns: A list of unsupported operations with reasons. Empty if all supported.
    public func validateSupport(function: HLOFunction) -> [(HLOOpKind, String)] {
        var unsupported: [(HLOOpKind, String)] = []
        for op in function.operations {
            if !MILOpTranslator.isSupported(op.kind) {
                unsupported.append((op.kind, MILOpTranslator.unsupportedReason(op.kind)))
            }
        }
        return unsupported
    }
}
