// Executable.swift
// MetalHLO
//
// A compiled program ready for execution.

import MetalHLOCore

/// A compiled StableHLO program ready for execution on Metal.
///
/// `Executable` encapsulates a compiled MPSGraph and provides methods
/// to execute the program with input buffers.
///
/// ## Example
/// ```swift
/// let outputs = try executable.execute([inputA, inputB])
/// let result = try outputs[0].toFloatArray()
/// ```
public final class Executable: @unchecked Sendable {

    // MARK: - Properties

    private let compiled: CompiledGraph
    private let executor: MetalExecutor

    /// The number of input arguments expected.
    public var inputCount: Int {
        compiled.inputTypes.count
    }

    /// The number of outputs produced.
    public var outputCount: Int {
        compiled.outputTypes.count
    }

    /// The input tensor types (shape and element type).
    public var inputTypes: [TensorType] {
        compiled.inputTypes.map { coreType in
            TensorType(
                shape: coreType.shape,
                elementType: ElementType.fromCoreType(coreType.elementType)
            )
        }
    }

    /// The output tensor types (shape and element type).
    public var outputTypes: [TensorType] {
        compiled.outputTypes.map { coreType in
            TensorType(
                shape: coreType.shape,
                elementType: ElementType.fromCoreType(coreType.elementType)
            )
        }
    }

    // MARK: - Initialization

    internal init(compiled: CompiledGraph, executor: MetalExecutor) {
        self.compiled = compiled
        self.executor = executor
    }

    // MARK: - Execution

    /// Executes the program with the given inputs.
    ///
    /// - Parameter inputs: Input buffers (must match `inputCount` and `inputTypes`).
    /// - Throws: `MetalHLOError.executionFailed` or `MetalHLOError.inputMismatch`.
    /// - Returns: Output buffers.
    public func execute(_ inputs: [Buffer]) throws -> [Buffer] {
        let inputStorages = inputs.map { $0.storage }
        let outputStorages: [BufferStorage]
        do {
            outputStorages = try executor.execute(compiled: compiled, inputs: inputStorages)
        } catch let error as ExecutorError {
            throw Self.convertExecutorError(error)
        }
        return outputStorages.map { Buffer(storage: $0) }
    }

    /// Executes the program with timing information.
    ///
    /// - Parameter inputs: Input buffers.
    /// - Throws: `MetalHLOError.executionFailed` or `MetalHLOError.inputMismatch`.
    /// - Returns: A tuple of (outputs, timing).
    public func executeWithTiming(_ inputs: [Buffer]) throws -> ([Buffer], ExecutionTiming) {
        let inputStorages = inputs.map { $0.storage }
        let outputStorages: [BufferStorage]
        let coreTiming: MetalHLOCore.ExecutionTiming
        do {
            (outputStorages, coreTiming) = try executor.executeWithTiming(compiled: compiled, inputs: inputStorages)
        } catch let error as ExecutorError {
            throw Self.convertExecutorError(error)
        }
        let buffers = outputStorages.map { Buffer(storage: $0) }
        let timing = ExecutionTiming(
            encodeTime: coreTiming.encodeTime,
            gpuTime: coreTiming.gpuTime,
            totalTime: coreTiming.totalTime
        )
        return (buffers, timing)
    }

    // MARK: - Error Conversion

    private static func convertExecutorError(_ error: ExecutorError) -> MetalHLOError {
        switch error {
        case .noMetalDevice:
            return .noMetalDevice
        case .unsupportedDevice(let reason):
            return .unsupportedDevice(reason)
        case .inputMismatch(let expected, let got):
            return .inputMismatch(expected: expected, got: got)
        case .inputTypeMismatch(let index, let expected, let got):
            return .inputTypeMismatch(
                index: index,
                expected: TensorType(shape: expected.shape, elementType: ElementType.fromCoreType(expected.elementType)),
                got: TensorType(shape: got.shape, elementType: ElementType.fromCoreType(got.elementType))
            )
        case .executionFailed(let reason):
            return .executionFailed(reason)
        case .bufferCreationFailed(let reason):
            return .bufferCreationFailed(reason)
        case .transferFailed(let reason):
            return .transferFailed(reason)
        case .commandQueueCreationFailed:
            return .executionFailed("Command queue creation failed")
        case .commandBufferCreationFailed:
            return .executionFailed("Command buffer creation failed")
        }
    }
}

/// Timing information for an execution.
public struct ExecutionTiming: Sendable {
    /// Time to encode commands (seconds).
    public let encodeTime: Double

    /// Time for GPU execution (seconds).
    public let gpuTime: Double

    /// Total wall-clock time (seconds).
    public let totalTime: Double

    public init(encodeTime: Double, gpuTime: Double, totalTime: Double) {
        self.encodeTime = encodeTime
        self.gpuTime = gpuTime
        self.totalTime = totalTime
    }
}
