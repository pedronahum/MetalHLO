// IntegratedExecutor.swift
// MetalHLOCore
//
// Executor for running compiled executables with unified buffer management.

import Foundation
@preconcurrency import Metal

// MARK: - Integrated Executor

/// Executes compiled executables with zero-allocation runtime.
///
/// The executor uses a single pre-allocated unified buffer for all intermediate
/// tensors, eliminating per-inference memory allocation overhead.
///
/// Example:
/// ```swift
/// let compiler = MetalHLOCompiler(device: device)
/// let executable = try compiler.compile(mlir)
/// let executor = IntegratedExecutor(device: device, executable: executable)
///
/// let outputs = try executor.execute(inputs: ["x": inputBuffer])
/// ```
public final class IntegratedExecutor: @unchecked Sendable {

    // MARK: - Properties

    /// Metal device.
    public let device: MTLDevice

    /// Command queue for execution.
    private let commandQueue: MTLCommandQueue

    /// The compiled executable.
    public let executable: CompiledExecutable

    /// Pre-allocated unified buffer for all intermediates.
    private let unifiedBuffer: MTLBuffer

    /// Constant buffers (created once).
    private var constantBuffers: [String: MTLBuffer]

    /// Configuration.
    public let config: Config

    /// Execution statistics.
    public private(set) var statistics: ExecutionStatistics

    // MARK: - Configuration

    public struct Config: Sendable {
        /// Whether to profile kernel execution times.
        public var enableProfiling: Bool

        /// Whether to wait for completion synchronously.
        public var synchronous: Bool

        /// Whether to validate inputs before execution.
        public var validateInputs: Bool

        /// Label for command buffers (for debugging).
        public var debugLabel: String?

        public init(
            enableProfiling: Bool = false,
            synchronous: Bool = true,
            validateInputs: Bool = true,
            debugLabel: String? = nil
        ) {
            self.enableProfiling = enableProfiling
            self.synchronous = synchronous
            self.validateInputs = validateInputs
            self.debugLabel = debugLabel
        }

        public static let `default` = Config()

        public static let profiling = Config(enableProfiling: true)

        public static let async = Config(synchronous: false)
    }

    // MARK: - Initialization

    /// Creates an executor for a compiled executable.
    /// - Throws: `IntegratedExecutorError.commandQueueCreationFailed` or `IntegratedExecutorError.bufferAllocationFailed`
    public init(device: MTLDevice, executable: CompiledExecutable, config: Config = .default) throws {
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw IntegratedExecutorError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue

        self.executable = executable
        self.config = config
        self.statistics = ExecutionStatistics()
        self.constantBuffers = executable.constantBuffers

        // Pre-allocate unified buffer for ALL intermediate tensors
        let bufferSize = max(executable.memoryPlan.totalBytes, 256)
        guard let unifiedBuffer = device.makeBuffer(
            length: bufferSize,
            options: .storageModeShared
        ) else {
            throw IntegratedExecutorError.bufferAllocationFailed(size: bufferSize)
        }
        self.unifiedBuffer = unifiedBuffer

        if let label = config.debugLabel {
            self.unifiedBuffer.label = "\(label)_unified"
        }
    }

    // MARK: - Execution

    /// Executes the compiled executable with the given inputs.
    ///
    /// - Parameter inputs: Dictionary mapping input names to Metal buffers.
    /// - Returns: Execution result with output buffers.
    /// - Throws: `ExecutorError` if execution fails.
    public func execute(inputs: [String: MTLBuffer]) throws -> ExecutionResult {
        let startTime = DispatchTime.now()

        // Serialize Metal execution to prevent concurrent access crashes
        metalExecutionSemaphore.wait()
        defer { metalExecutionSemaphore.signal() }

        // Validate inputs
        if config.validateInputs {
            try executable.validateInputs(inputs)
        }

        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw IntegratedExecutorError.commandBufferCreationFailed
        }

        if let label = config.debugLabel {
            commandBuffer.label = label
        }

        // Kernel timings for profiling
        var kernelTimings: [OpID: Double]?
        if config.enableProfiling {
            kernelTimings = [:]
        }

        // Encode all operations
        for opID in executable.executionOrder {
            try encodeOperation(
                opID,
                commandBuffer: commandBuffer,
                inputs: inputs,
                kernelTimings: &kernelTimings
            )
        }

        // Execute
        commandBuffer.commit()

        if config.synchronous {
            commandBuffer.waitUntilCompleted()

            if let error = commandBuffer.error {
                throw IntegratedExecutorError.executionFailed(error.localizedDescription)
            }
        }

        // Extract outputs
        let outputs = try extractOutputs(inputs: inputs)

        let executionTimeMs = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000

        // Update statistics
        statistics.executionCount += 1
        statistics.totalExecutionTimeMs += executionTimeMs
        statistics.lastExecutionTimeMs = executionTimeMs

        return ExecutionResult(
            outputs: outputs,
            executionTimeMs: executionTimeMs,
            kernelTimings: kernelTimings
        )
    }

    /// Executes asynchronously, returning immediately.
    ///
    /// - Parameters:
    ///   - inputs: Dictionary mapping input names to Metal buffers.
    ///   - completion: Called when execution completes.
    public func executeAsync(
        inputs: [String: MTLBuffer],
        completion: @escaping (Result<ExecutionResult, Error>) -> Void
    ) {
        let startTime = DispatchTime.now()

        do {
            if config.validateInputs {
                try executable.validateInputs(inputs)
            }
        } catch {
            completion(.failure(error))
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            completion(.failure(IntegratedExecutorError.commandBufferCreationFailed))
            return
        }

        var kernelTimings: [OpID: Double]?
        if config.enableProfiling {
            kernelTimings = [:]
        }

        do {
            for opID in executable.executionOrder {
                try encodeOperation(
                    opID,
                    commandBuffer: commandBuffer,
                    inputs: inputs,
                    kernelTimings: &kernelTimings
                )
            }
        } catch {
            completion(.failure(error))
            return
        }

        commandBuffer.addCompletedHandler { [weak self] buffer in
            guard let self = self else { return }

            if let error = buffer.error {
                completion(.failure(IntegratedExecutorError.executionFailed(error.localizedDescription)))
                return
            }

            do {
                let outputs = try self.extractOutputs(inputs: inputs)
                let executionTimeMs = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000

                self.statistics.executionCount += 1
                self.statistics.totalExecutionTimeMs += executionTimeMs
                self.statistics.lastExecutionTimeMs = executionTimeMs

                let result = ExecutionResult(
                    outputs: outputs,
                    executionTimeMs: executionTimeMs,
                    kernelTimings: kernelTimings
                )

                completion(.success(result))
            } catch {
                completion(.failure(error))
            }
        }

        commandBuffer.commit()
    }

    // MARK: - Private Methods

    /// Encodes a single operation to the command buffer.
    private func encodeOperation(
        _ opID: OpID,
        commandBuffer: MTLCommandBuffer,
        inputs: [String: MTLBuffer],
        kernelTimings: inout [OpID: Double]?
    ) throws {
        guard let pipeline = executable.pipelines[opID] else {
            throw IntegratedExecutorError.missingPipeline(opID)
        }

        guard let dispatch = executable.dispatches[opID] else {
            throw IntegratedExecutorError.missingDispatch(opID)
        }

        guard let bindings = executable.bindings[opID] else {
            throw IntegratedExecutorError.missingBindings(opID)
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw IntegratedExecutorError.encoderCreationFailed
        }

        if let label = config.debugLabel {
            encoder.label = "\(label)_\(opID)"
        }

        encoder.setComputePipelineState(pipeline)

        // Bind all buffers and scalars
        for binding in bindings {
            switch binding.source {
            case .scalar(let value):
                // Use setBytes for scalar uniform values
                var scalarValue = value
                encoder.setBytes(&scalarValue, length: MemoryLayout<UInt32>.size, index: binding.index)
            default:
                // Use setBuffer for all other sources
                let (buffer, offset) = try resolveBinding(binding, inputs: inputs)
                encoder.setBuffer(buffer, offset: offset, index: binding.index)
            }
        }

        // Set threadgroup memory if needed (for operations like matmul that use shared memory)
        if let sharedMemSize = executable.sharedMemorySizes[opID], sharedMemSize > 0 {
            // For matmul, we need two tile buffers: tileA at index 0, tileB at index 1
            let tileSize = sharedMemSize / 2
            encoder.setThreadgroupMemoryLength(tileSize, index: 0)  // tileA
            encoder.setThreadgroupMemoryLength(tileSize, index: 1)  // tileB
        }

        // Dispatch
        if dispatch.useNonUniform {
            let totalThreads = MTLSize(
                width: dispatch.gridSize.width * dispatch.threadgroupSize.width,
                height: dispatch.gridSize.height * dispatch.threadgroupSize.height,
                depth: dispatch.gridSize.depth * dispatch.threadgroupSize.depth
            )
            encoder.dispatchThreads(totalThreads, threadsPerThreadgroup: dispatch.threadgroupSize)
        } else {
            encoder.dispatchThreadgroups(dispatch.gridSize, threadsPerThreadgroup: dispatch.threadgroupSize)
        }

        encoder.endEncoding()
    }

    /// Resolves a buffer binding to an actual buffer and offset.
    private func resolveBinding(
        _ binding: BufferBinding,
        inputs: [String: MTLBuffer]
    ) throws -> (MTLBuffer, Int) {
        switch binding.source {
        case .input(let name):
            guard let buffer = inputs[name] else {
                throw IntegratedExecutorError.missingInput(name)
            }
            return (buffer, binding.offset)

        case .output(let name):
            // Outputs come from the unified buffer
            if let outputOffset = executable.memoryPlan.tensorOffsets[name] {
                return (unifiedBuffer, outputOffset + binding.offset)
            }
            // If not in memory plan, it's a direct output
            guard let buffer = inputs[name] else {
                // Output will be extracted later from unified buffer
                return (unifiedBuffer, binding.offset)
            }
            return (buffer, binding.offset)

        case .unified(let offset):
            return (unifiedBuffer, offset + binding.offset)

        case .constant(let id):
            guard let buffer = constantBuffers[id] else {
                throw IntegratedExecutorError.missingConstant(id)
            }
            return (buffer, binding.offset)

        case .threadgroup:
            // Threadgroup memory is handled by the encoder, not buffer binding
            throw IntegratedExecutorError.invalidBinding("Threadgroup memory cannot be bound as buffer")

        case .scalar:
            // Scalar bindings are handled separately via setBytes, not resolveBinding
            throw IntegratedExecutorError.invalidBinding("Scalar bindings should use setBytes, not buffer binding")
        }
    }

    /// Extracts output buffers from the unified buffer.
    private func extractOutputs(inputs: [String: MTLBuffer]) throws -> [String: MTLBuffer] {
        var outputs: [String: MTLBuffer] = [:]

        for (name, spec) in executable.outputSpecs {
            // Check if output is in memory plan
            if let offset = executable.memoryPlan.tensorOffsets[name] {
                // Create output buffer and copy data
                guard let outputBuffer = device.makeBuffer(
                    length: spec.byteSize,
                    options: .storageModeShared
                ) else {
                    throw IntegratedExecutorError.bufferAllocationFailed(size: spec.byteSize)
                }

                if let label = config.debugLabel {
                    outputBuffer.label = "\(label)_output_\(name)"
                }

                memcpy(
                    outputBuffer.contents(),
                    unifiedBuffer.contents().advanced(by: offset),
                    spec.byteSize
                )

                outputs[name] = outputBuffer
            } else if let inputBuffer = inputs[name] {
                // Output was written directly to input buffer (in-place)
                outputs[name] = inputBuffer
            }
        }

        return outputs
    }

    // MARK: - Utilities

    /// Resets execution statistics.
    public func resetStatistics() {
        statistics = ExecutionStatistics()
    }

    /// Returns memory usage information.
    public var memoryUsage: MemoryUsage {
        MemoryUsage(
            unifiedBufferBytes: unifiedBuffer.length,
            constantBufferBytes: constantBuffers.values.reduce(0) { $0 + $1.length },
            peakMemoryBytes: executable.memoryPlan.peakMemory
        )
    }

    public struct MemoryUsage: Sendable {
        public let unifiedBufferBytes: Int
        public let constantBufferBytes: Int
        public let peakMemoryBytes: Int

        public var totalBytes: Int {
            unifiedBufferBytes + constantBufferBytes
        }
    }
}

// MARK: - Execution Statistics

/// Statistics about executor execution.
public struct ExecutionStatistics: Sendable {
    /// Number of executions.
    public var executionCount: Int = 0

    /// Total execution time in milliseconds.
    public var totalExecutionTimeMs: Double = 0

    /// Last execution time in milliseconds.
    public var lastExecutionTimeMs: Double = 0

    /// Average execution time in milliseconds.
    public var averageExecutionTimeMs: Double {
        executionCount > 0 ? totalExecutionTimeMs / Double(executionCount) : 0
    }
}

// MARK: - Executor Errors

/// Errors that can occur during execution.
public enum IntegratedExecutorError: Error, Sendable {
    case commandQueueCreationFailed
    case commandBufferCreationFailed
    case encoderCreationFailed
    case bufferAllocationFailed(size: Int)
    case missingPipeline(OpID)
    case missingDispatch(OpID)
    case missingBindings(OpID)
    case missingInput(String)
    case missingConstant(String)
    case invalidBinding(String)
    case executionFailed(String)
}

// MARK: - Batch Executor

/// Executor optimized for batch inference.
public final class BatchExecutor: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let executable: CompiledExecutable
    private var unifiedBuffers: [MTLBuffer]
    private let config: Config

    public struct Config: Sendable {
        /// Number of buffers for double/triple buffering.
        public var bufferCount: Int

        /// Maximum batch size.
        public var maxBatchSize: Int

        public init(bufferCount: Int = 3, maxBatchSize: Int = 32) {
            self.bufferCount = bufferCount
            self.maxBatchSize = maxBatchSize
        }
    }

    /// Creates a batch executor.
    /// - Throws: `IntegratedExecutorError.commandQueueCreationFailed` or `IntegratedExecutorError.bufferAllocationFailed`
    public init(device: MTLDevice, executable: CompiledExecutable, config: Config = Config()) throws {
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw IntegratedExecutorError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue

        self.executable = executable
        self.config = config

        // Create multiple unified buffers for overlapped execution
        let bufferSize = max(executable.memoryPlan.totalBytes, 256)
        var buffers: [MTLBuffer] = []
        for i in 0..<config.bufferCount {
            guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
                throw IntegratedExecutorError.bufferAllocationFailed(size: bufferSize)
            }
            buffer.label = "unified_buffer_\(i)"
            buffers.append(buffer)
        }
        self.unifiedBuffers = buffers
    }

    /// Executes multiple batches with overlapped execution.
    public func executeBatches(
        batches: [[String: MTLBuffer]],
        completion: @escaping ([Result<ExecutionResult, Error>]) -> Void
    ) {
        var results: [Result<ExecutionResult, Error>?] = Array(repeating: nil, count: batches.count)
        let group = DispatchGroup()

        for (index, inputs) in batches.enumerated() {
            group.enter()

            let bufferIndex = index % config.bufferCount
            let unifiedBuffer = unifiedBuffers[bufferIndex]

            executeSingleBatch(
                inputs: inputs,
                unifiedBuffer: unifiedBuffer,
                completion: { result in
                    results[index] = result
                    group.leave()
                }
            )
        }

        group.notify(queue: .main) {
            completion(results.compactMap { $0 })
        }
    }

    private func executeSingleBatch(
        inputs: [String: MTLBuffer],
        unifiedBuffer: MTLBuffer,
        completion: @escaping (Result<ExecutionResult, Error>) -> Void
    ) {
        let startTime = DispatchTime.now()

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            completion(.failure(IntegratedExecutorError.commandBufferCreationFailed))
            return
        }

        // Encode operations (simplified - full implementation would mirror IntegratedExecutor)
        for opID in executable.executionOrder {
            guard let pipeline = executable.pipelines[opID],
                  let dispatch = executable.dispatches[opID] else {
                continue
            }

            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                continue
            }

            encoder.setComputePipelineState(pipeline)
            // Buffer bindings would go here
            encoder.dispatchThreadgroups(dispatch.gridSize, threadsPerThreadgroup: dispatch.threadgroupSize)
            encoder.endEncoding()
        }

        commandBuffer.addCompletedHandler { buffer in
            if let error = buffer.error {
                completion(.failure(IntegratedExecutorError.executionFailed(error.localizedDescription)))
                return
            }

            let executionTimeMs = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000

            let result = ExecutionResult(
                outputs: [:],  // Would extract from unified buffer
                executionTimeMs: executionTimeMs,
                kernelTimings: nil
            )

            completion(.success(result))
        }

        commandBuffer.commit()
    }
}
