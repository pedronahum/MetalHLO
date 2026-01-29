// Executable.swift
// MetalHLO
//
// A compiled program ready for execution.

import Metal
import MetalHLOCore

/// A compiled StableHLO program ready for execution on Metal.
///
/// `Executable` encapsulates either an MPSGraph-based executable or an
/// optimized Metal kernel-based executable, and provides methods to execute
/// the program with input buffers.
///
/// ## Example
/// ```swift
/// // Basic compilation (MPSGraph backend)
/// let executable = try client.compile(mlirString)
///
/// // Optimized compilation with custom config
/// let config = CompilationConfig(optimizationLevel: .O3)
/// let optimizedExe = try client.compile(mlirString, config: config)
///
/// let outputs = try executable.execute([inputA, inputB])
/// let result = try outputs[0].toFloatArray()
/// ```
public final class Executable: @unchecked Sendable {

    // MARK: - Backend Types

    private enum Backend {
        /// MPSGraph-based execution (default path)
        case mpsGraph(CompiledGraph, MetalExecutor)

        /// Optimized Metal kernels via integrated executor
        case integrated(CompiledExecutable, IntegratedExecutor)
    }

    // MARK: - Properties

    private let backend: Backend

    /// The number of input arguments expected.
    public var inputCount: Int {
        switch backend {
        case .mpsGraph(let compiled, _):
            return compiled.inputTypes.count
        case .integrated(let executable, _):
            return executable.inputSpecs.count
        }
    }

    /// The number of outputs produced.
    public var outputCount: Int {
        switch backend {
        case .mpsGraph(let compiled, _):
            return compiled.outputTypes.count
        case .integrated(let executable, _):
            return executable.outputSpecs.count
        }
    }

    /// The input tensor types (shape and element type).
    public var inputTypes: [TensorType] {
        switch backend {
        case .mpsGraph(let compiled, _):
            return compiled.inputTypes.map { coreType in
                TensorType(
                    shape: coreType.shape,
                    elementType: ElementType.fromCoreType(coreType.elementType)
                )
            }
        case .integrated(let executable, _):
            return executable.inputSpecs.values.map { spec in
                TensorType(
                    shape: spec.shape,
                    elementType: ElementType.fromCoreType(spec.elementType)
                )
            }
        }
    }

    /// The output tensor types (shape and element type).
    public var outputTypes: [TensorType] {
        switch backend {
        case .mpsGraph(let compiled, _):
            return compiled.outputTypes.map { coreType in
                TensorType(
                    shape: coreType.shape,
                    elementType: ElementType.fromCoreType(coreType.elementType)
                )
            }
        case .integrated(let executable, _):
            return executable.outputSpecs.values.map { spec in
                TensorType(
                    shape: spec.shape,
                    elementType: ElementType.fromCoreType(spec.elementType)
                )
            }
        }
    }

    // MARK: - Initialization

    /// Internal initializer for MPSGraph-based execution.
    internal init(compiled: CompiledGraph, executor: MetalExecutor) {
        self.backend = .mpsGraph(compiled, executor)
    }

    /// Internal initializer for integrated (optimized) execution.
    internal init(compiled: CompiledExecutable, executor: IntegratedExecutor) {
        self.backend = .integrated(compiled, executor)
    }

    // MARK: - Execution

    /// Executes the program with the given inputs.
    ///
    /// - Parameter inputs: Input buffers (must match `inputCount` and `inputTypes`).
    /// - Throws: `MetalHLOError.executionFailed` or `MetalHLOError.inputMismatch`.
    /// - Returns: Output buffers.
    public func execute(_ inputs: [Buffer]) throws -> [Buffer] {
        switch backend {
        case .mpsGraph(let compiled, let executor):
            return try executeMPSGraph(inputs, compiled: compiled, executor: executor)
        case .integrated(_, let executor):
            return try executeIntegrated(inputs, executor: executor)
        }
    }

    /// Executes the program with timing information.
    ///
    /// - Parameter inputs: Input buffers.
    /// - Throws: `MetalHLOError.executionFailed` or `MetalHLOError.inputMismatch`.
    /// - Returns: A tuple of (outputs, timing).
    public func executeWithTiming(_ inputs: [Buffer]) throws -> ([Buffer], ExecutionTiming) {
        switch backend {
        case .mpsGraph(let compiled, let executor):
            return try executeWithTimingMPSGraph(inputs, compiled: compiled, executor: executor)
        case .integrated(_, let executor):
            return try executeWithTimingIntegrated(inputs, executor: executor)
        }
    }

    // MARK: - MPSGraph Execution

    private func executeMPSGraph(
        _ inputs: [Buffer],
        compiled: CompiledGraph,
        executor: MetalExecutor
    ) throws -> [Buffer] {
        let inputStorages = inputs.map { $0.storage }
        let outputStorages: [BufferStorage]
        do {
            outputStorages = try executor.execute(compiled: compiled, inputs: inputStorages)
        } catch let error as ExecutorError {
            throw Self.convertExecutorError(error)
        }
        return outputStorages.map { Buffer(storage: $0) }
    }

    private func executeWithTimingMPSGraph(
        _ inputs: [Buffer],
        compiled: CompiledGraph,
        executor: MetalExecutor
    ) throws -> ([Buffer], ExecutionTiming) {
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

    // MARK: - Integrated Execution

    private func executeIntegrated(
        _ inputs: [Buffer],
        executor: IntegratedExecutor
    ) throws -> [Buffer] {
        let metalInputs = buildMetalInputs(inputs, executable: executor.executable, device: executor.device)
        let result: ExecutionResult
        do {
            result = try executor.execute(inputs: metalInputs)
        } catch let error as IntegratedExecutorError {
            throw Self.convertIntegratedError(error)
        }
        return buildOutputBuffers(from: result, executor: executor)
    }

    private func executeWithTimingIntegrated(
        _ inputs: [Buffer],
        executor: IntegratedExecutor
    ) throws -> ([Buffer], ExecutionTiming) {
        let metalInputs = buildMetalInputs(inputs, executable: executor.executable, device: executor.device)
        let result: ExecutionResult
        do {
            result = try executor.execute(inputs: metalInputs)
        } catch let error as IntegratedExecutorError {
            throw Self.convertIntegratedError(error)
        }
        let buffers = buildOutputBuffers(from: result, executor: executor)
        let timing = ExecutionTiming(
            encodeTime: 0,  // Not separately tracked in integrated executor
            gpuTime: result.executionTimeMs / 1000.0,
            totalTime: result.executionTimeMs / 1000.0
        )
        return (buffers, timing)
    }

    private func buildMetalInputs(_ inputs: [Buffer], executable: CompiledExecutable, device: MTLDevice) -> [String: MTLBuffer] {
        var metalInputs: [String: MTLBuffer] = [:]
        let inputNames = Array(executable.inputSpecs.keys).sorted()

        for (index, name) in inputNames.enumerated() where index < inputs.count {
            // Create Metal buffer from storage data
            let storage = inputs[index].storage
            let data = (try? storage.toData()) ?? Data()
            if let buffer = device.makeBuffer(bytes: (data as NSData).bytes, length: data.count, options: .storageModeShared) {
                metalInputs[name] = buffer
            }
        }
        return metalInputs
    }

    private func buildOutputBuffers(from result: ExecutionResult, executor: IntegratedExecutor) -> [Buffer] {
        let device = executor.device
        let outputNames = Array(executor.executable.outputSpecs.keys).sorted()
        return outputNames.compactMap { name -> Buffer? in
            guard let metalBuffer = result.outputs[name],
                  let spec = executor.executable.outputSpecs[name] else {
                return nil
            }
            // Copy data from Metal buffer to Data
            let byteCount = metalBuffer.length
            let data = Data(bytes: metalBuffer.contents(), count: byteCount)
            // Create BufferStorage with the data
            let storage = try? BufferStorage(
                bytes: data,
                shape: spec.shape,
                elementType: spec.elementType,
                device: device
            )
            return storage.map { Buffer(storage: $0) }
        }
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

    private static func convertIntegratedError(_ error: IntegratedExecutorError) -> MetalHLOError {
        switch error {
        case .commandQueueCreationFailed:
            return .executionFailed("Command queue creation failed")
        case .commandBufferCreationFailed:
            return .executionFailed("Command buffer creation failed")
        case .encoderCreationFailed:
            return .executionFailed("Command encoder creation failed")
        case .bufferAllocationFailed(let size):
            return .bufferCreationFailed("Failed to allocate buffer of size \(size) bytes")
        case .missingPipeline(let opID):
            return .executionFailed("Missing pipeline for operation: \(opID)")
        case .missingDispatch(let opID):
            return .executionFailed("Missing dispatch for operation: \(opID)")
        case .missingBindings(let opID):
            return .executionFailed("Missing bindings for operation: \(opID)")
        case .missingInput(let name):
            return .executionFailed("Missing input: \(name)")
        case .missingConstant(let id):
            return .executionFailed("Missing constant: \(id)")
        case .invalidBinding(let reason):
            return .executionFailed("Invalid binding: \(reason)")
        case .executionFailed(let reason):
            return .executionFailed(reason)
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
