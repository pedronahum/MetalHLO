// Errors.swift
// MetalHLO
//
// Error types for MetalHLO operations.

/// Errors that can occur during MetalHLO operations.
public enum MetalHLOError: Error, Sendable, CustomStringConvertible {

    /// No Metal device is available.
    case noMetalDevice

    /// The Metal device doesn't support required features.
    case unsupportedDevice(String)

    /// Failed to parse the MLIR input.
    case parseFailed(line: Int, column: Int, message: String)

    /// Encountered an unsupported StableHLO operation.
    case unsupportedOperation(String)

    /// Compilation failed.
    case compilationFailed(String)

    /// Buffer creation failed.
    case bufferCreationFailed(String)

    /// Input count or types don't match the executable.
    case inputMismatch(expected: Int, got: Int)

    /// Input type mismatch at a specific index.
    case inputTypeMismatch(index: Int, expected: TensorType, got: TensorType)

    /// Execution failed.
    case executionFailed(String)

    /// Data transfer between host and device failed.
    case transferFailed(String)

    // MARK: - CustomStringConvertible

    public var description: String {
        switch self {
        case .noMetalDevice:
            return "No Metal device available"
        case .unsupportedDevice(let reason):
            return "Unsupported device: \(reason)"
        case .parseFailed(let line, let column, let message):
            return "Parse error at line \(line), column \(column): \(message)"
        case .unsupportedOperation(let op):
            return "Unsupported operation: \(op)"
        case .compilationFailed(let reason):
            return "Compilation failed: \(reason)"
        case .bufferCreationFailed(let reason):
            return "Buffer creation failed: \(reason)"
        case .inputMismatch(let expected, let got):
            return "Input count mismatch: expected \(expected), got \(got)"
        case .inputTypeMismatch(let index, let expected, let got):
            return "Input type mismatch at index \(index): expected \(expected), got \(got)"
        case .executionFailed(let reason):
            return "Execution failed: \(reason)"
        case .transferFailed(let reason):
            return "Transfer failed: \(reason)"
        }
    }
}
