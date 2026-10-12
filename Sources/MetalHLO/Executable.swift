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

        /// Heterogeneous GPU+ANE execution
        case heterogeneous(HLOFunction, HeterogeneousExecutor)
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
        case .heterogeneous(let function, _):
            return function.inputs.count
        }
    }

    /// The number of outputs produced.
    public var outputCount: Int {
        switch backend {
        case .mpsGraph(let compiled, _):
            return compiled.outputTypes.count
        case .integrated(let executable, _):
            return executable.outputSpecs.count
        case .heterogeneous(let function, _):
            return function.outputTypes.count
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
            // In signature order: inputSpecs is keyed by name.
            return executable.inputOrder.compactMap { executable.inputSpecs[$0] }.map { spec in
                TensorType(
                    shape: spec.shape,
                    elementType: ElementType.fromCoreType(spec.elementType)
                )
            }
        case .heterogeneous(let function, _):
            return function.inputs.map { arg in
                TensorType(
                    shape: arg.type.shape,
                    elementType: ElementType.fromCoreType(arg.type.elementType)
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
            // In signature order: outputSpecs is keyed by name.
            return executable.outputOrder.compactMap { executable.outputSpecs[$0] }.map { spec in
                TensorType(
                    shape: spec.shape,
                    elementType: ElementType.fromCoreType(spec.elementType)
                )
            }
        case .heterogeneous(let function, _):
            return function.outputTypes.map { coreType in
                TensorType(
                    shape: coreType.shape,
                    elementType: ElementType.fromCoreType(coreType.elementType)
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

    /// Internal initializer for heterogeneous GPU+ANE execution.
    internal init(function: HLOFunction, executor: HeterogeneousExecutor) {
        self.backend = .heterogeneous(function, executor)
    }

    // MARK: - Execution

    /// Executes the program with the given inputs.
    ///
    /// - Parameter inputs: Input buffers (must match `inputCount` and `inputTypes`).
    /// - Throws: `MetalHLOError.executionFailed` or `MetalHLOError.inputMismatch`.
    /// - Returns: Output buffers.
    public func execute(_ inputs: [Buffer]) throws -> [Buffer] {
        let inputs = try float64ForExecutor(inputs)
        let outputs: [Buffer]
        switch backend {
        case .mpsGraph(let compiled, let executor):
            outputs = try executeMPSGraph(inputs, compiled: compiled, executor: executor)
        case .integrated(_, let executor):
            outputs = try executeIntegrated(inputs, executor: executor)
        case .heterogeneous(let function, let executor):
            outputs = try executeHeterogeneous(inputs, function: function, executor: executor)
        }
        return try float64FromExecutor(outputs)
    }

    // MARK: - Float64 boundary

    /// Executors compute f64 as f32 (Metal has no double); callers' f64 buffers
    /// hold 8-byte doubles. Narrow f64 inputs on the way in...
    private func float64ForExecutor(_ inputs: [Buffer]) throws -> [Buffer] {
        guard inputs.contains(where: { $0.elementType == .float64 }) else { return inputs }
        return try inputs.map { $0.elementType == .float64 ? Buffer(storage: try $0.storage.float64ForExecutor()) : $0 }
    }

    /// ...and widen every output the program declares f64 on the way out.
    private func float64FromExecutor(_ outputs: [Buffer]) throws -> [Buffer] {
        let declared = outputTypes
        guard declared.contains(where: { $0.elementType == .float64 }) else { return outputs }
        return try outputs.enumerated().map { i, out in
            i < declared.count && declared[i].elementType == .float64
                ? Buffer(storage: try out.storage.float64FromExecutor()) : out
        }
    }

    /// Executes the program with timing information.
    ///
    /// - Parameter inputs: Input buffers.
    /// - Throws: `MetalHLOError.executionFailed` or `MetalHLOError.inputMismatch`.
    /// - Returns: A tuple of (outputs, timing).
    public func executeWithTiming(_ inputs: [Buffer]) throws -> ([Buffer], ExecutionTiming) {
        let (outputs, timing) = try executeWithTimingUnconverted(try float64ForExecutor(inputs))
        return (try float64FromExecutor(outputs), timing)
    }

    private func executeWithTimingUnconverted(_ inputs: [Buffer]) throws -> ([Buffer], ExecutionTiming) {
        switch backend {
        case .mpsGraph(let compiled, let executor):
            return try executeWithTimingMPSGraph(inputs, compiled: compiled, executor: executor)
        case .integrated(_, let executor):
            return try executeWithTimingIntegrated(inputs, executor: executor)
        case .heterogeneous(let function, let executor):
            // Reset stats before execution so we measure a single call,
            // not cumulative time across all previous executions.
            executor.resetStats()
            let buffers = try executeHeterogeneous(inputs, function: function, executor: executor)
            let stats = executor.getStats()
            let timing = ExecutionTiming(
                encodeTime: 0,
                gpuTime: stats.gpuTimeMs / 1000.0,
                totalTime: stats.totalTimeMs / 1000.0
            )
            return (buffers, timing)
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
        let prof = ProcessInfo.processInfo.environment["METALHLO_PROFILE_EXEC"] == "1"
        let t0 = prof ? DispatchTime.now().uptimeNanoseconds : 0
        let metalInputs = buildMetalInputs(inputs, executable: executor.executable, device: executor.device)
        let t1 = prof ? DispatchTime.now().uptimeNanoseconds : 0
        let result: ExecutionResult
        do {
            result = try executor.execute(inputs: metalInputs)
        } catch let error as IntegratedExecutorError {
            throw Self.convertIntegratedError(error)
        }
        let t2 = prof ? DispatchTime.now().uptimeNanoseconds : 0
        let out = try buildOutputBuffers(from: result, executor: executor, inputs: inputs)
        if prof {
            let t3 = DispatchTime.now().uptimeNanoseconds
            let ms = { (a: UInt64, b: UInt64) in Double(b - a) / 1_000_000 }
            FileHandle.standardError.write("[exec-integrated] buildInputs=\(String(format: "%.2f", ms(t0,t1)))ms execute=\(String(format: "%.2f", ms(t1,t2)))ms buildOutputs=\(String(format: "%.2f", ms(t2,t3)))ms (nOut=\(out.count))\n".data(using: .utf8)!)
        }
        return out
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
        let buffers = try buildOutputBuffers(from: result, executor: executor, inputs: inputs)
        let timing = ExecutionTiming(
            encodeTime: 0,  // Not separately tracked in integrated executor
            gpuTime: result.gpuTimeMs / 1000.0,  // GPU-only time (excludes output extraction)
            totalTime: result.executionTimeMs / 1000.0
        )
        return (buffers, timing)
    }

    private func buildMetalInputs(_ inputs: [Buffer], executable: CompiledExecutable, device: MTLDevice) -> [String: MTLBuffer] {
        var metalInputs: [String: MTLBuffer] = [:]
        // Use inputOrder to preserve function signature order (not alphabetical)
        let inputNames = executable.inputOrder

        for (index, name) in inputNames.enumerated() where index < inputs.count {
            let storage = inputs[index].storage
            // Zero-copy fast path: BufferStorage already wraps an MTLBuffer.
            // Avoids `storage.toData()` + `device.makeBuffer(bytes:)` which would
            // memcpy the input twice — a real problem when an op takes a large
            // tensor as input (e.g. `jit_dynamic_slice` on the full preloaded
            // CIFAR-10 dataset = 600 MB copied twice, ~80 ms per step).
            if let buffer = storage.metalBuffer {
                metalInputs[name] = buffer
                continue
            }
            // Slow path: small tensors stored as `.data` — copy once into a new
            // shared MTLBuffer. (Acceptable: small tensors are by construction
            // well under the LargeTensorStorage threshold.)
            let data = (try? storage.toData()) ?? Data()
            if let buffer = device.makeBuffer(bytes: (data as NSData).bytes, length: data.count, options: .storageModeShared) {
                metalInputs[name] = buffer
            }
        }
        return metalInputs
    }

    private func buildOutputBuffers(
        from result: ExecutionResult, executor: IntegratedExecutor, inputs: [Buffer]
    ) throws -> [Buffer] {
        let device = executor.device
        // Use outputOrder to preserve function signature order (not alphabetical)
        let outputNames = executor.executable.outputOrder
        // Zero-copy output handoff (default; METALHLO_ZEROCOPY_OUTPUTS=0 to
        // disable). The executor allocates a FRESH owned buffer per output in
        // this mode (extractOutputs bypasses its reuse pool), so we can wrap it
        // directly instead of the old Data → new-buffer double-copy. For
        // training, outputs are the updated params + optimizer state (~130 MB),
        // so the double-copy was ~17 ms/step — the dominant host overhead.
        let zeroCopy = ProcessInfo.processInfo.environment["METALHLO_ZEROCOPY_OUTPUTS"] != "0"
        return try outputNames.map { name -> Buffer in
            guard let metalBuffer = result.outputs[name] ?? executor.executable.constantBuffers[name],
                  let spec = executor.executable.outputSpecs[name] else {
                // No kernel writes an output the function returns straight
                // from its arguments: hand that argument back.
                if let i = executor.executable.inputOrder.firstIndex(of: name), i < inputs.count {
                    return inputs[i]
                }
                throw MetalHLOError.executionFailed("output \(name) was not produced")
            }
            if zeroCopy {
                let large = LargeTensorStorage(
                    buffer: metalBuffer, shape: spec.shape, elementType: spec.elementType)
                return Buffer(storage: BufferStorage(largeTensor: large, device: device))
            }
            let byteCount = metalBuffer.length
            let data = Data(bytes: metalBuffer.contents(), count: byteCount)
            let storage = try BufferStorage(
                bytes: data,
                shape: spec.shape,
                elementType: spec.elementType,
                device: device
            )
            return Buffer(storage: storage)
        }
    }

    // MARK: - Heterogeneous Execution

    private func executeHeterogeneous(
        _ inputs: [Buffer],
        function: HLOFunction,
        executor: HeterogeneousExecutor
    ) throws -> [Buffer] {
        // Build named input storages
        var namedInputs: [String: BufferStorage] = [:]
        for (i, arg) in function.inputs.enumerated() where i < inputs.count {
            namedInputs[arg.name] = inputs[i].storage
        }

        let result = try executor.execute(function, inputs: namedInputs)

        // Collect outputs in return value order
        var outputBuffers: [Buffer] = []
        for retVal in function.returnValues {
            if let storage = result.outputStorages[retVal] {
                outputBuffers.append(Buffer(storage: storage))
            }
        }
        return outputBuffers
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
        case .generalError(let reason):
            return .executionFailed(reason)
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
