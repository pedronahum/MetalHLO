// ANEError.swift
// ANERuntime
//
// Error types for ANE operations.

/// Errors that can occur during ANE operations.
public enum ANEError: Error, Sendable, CustomStringConvertible {

    /// ANE hardware is not available on this machine.
    case aneNotAvailable(String)

    /// Failed to load AppleNeuralEngine.framework.
    case frameworkLoadFailed(String)

    /// Failed to resolve a required class or method.
    case symbolResolutionFailed(String)

    /// MIL program compilation failed.
    case compilationFailed(String)

    /// Program execution failed.
    case executionFailed(String)

    /// IOSurface buffer creation or operation failed.
    case bufferError(String)

    /// Compilation limit reached — ANE leaks resources after ~119 compiles.
    case compilationLimitReached(current: Int, limit: Int)

    /// Data type is not supported on ANE (only FP16 for compute).
    case unsupportedDataType(String)

    /// An internal error in the ANE runtime.
    case internalError(String)

    /// CoreML model compilation failed.
    case coreMLCompilationFailed(String)

    /// Protobuf serialization failed.
    case protobufSerializationFailed(String)

    // MARK: - CustomStringConvertible

    public var description: String {
        switch self {
        case .aneNotAvailable(let reason):
            return "ANE not available: \(reason)"
        case .frameworkLoadFailed(let reason):
            return "Failed to load ANE framework: \(reason)"
        case .symbolResolutionFailed(let symbol):
            return "Failed to resolve ANE symbol: \(symbol)"
        case .compilationFailed(let reason):
            return "ANE compilation failed: \(reason)"
        case .executionFailed(let reason):
            return "ANE execution failed: \(reason)"
        case .bufferError(let reason):
            return "ANE buffer error: \(reason)"
        case .compilationLimitReached(let current, let limit):
            return "ANE compilation limit reached: \(current)/\(limit)"
        case .unsupportedDataType(let dtype):
            return "Unsupported ANE data type: \(dtype)"
        case .internalError(let reason):
            return "ANE internal error: \(reason)"
        case .coreMLCompilationFailed(let reason):
            return "CoreML compilation failed: \(reason)"
        case .protobufSerializationFailed(let reason):
            return "Protobuf serialization failed: \(reason)"
        }
    }
}
