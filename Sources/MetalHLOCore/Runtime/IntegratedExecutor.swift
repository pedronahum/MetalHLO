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

    /// Output buffer pool for reuse (optional).
    private var outputBufferPool: OutputBufferPool?

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

        /// Whether to pool output buffers for reuse across executions.
        public var enableOutputPooling: Bool

        /// Number of output buffers to pre-allocate per output (when pooling enabled).
        public var outputPoolSize: Int

        public init(
            enableProfiling: Bool = false,
            synchronous: Bool = true,
            validateInputs: Bool = true,
            debugLabel: String? = nil,
            enableOutputPooling: Bool = true,
            outputPoolSize: Int = 3
        ) {
            self.enableProfiling = enableProfiling
            self.synchronous = synchronous
            self.validateInputs = validateInputs
            self.debugLabel = debugLabel
            self.enableOutputPooling = enableOutputPooling
            self.outputPoolSize = outputPoolSize
        }

        public static let `default` = Config()

        public static let profiling = Config(enableProfiling: true)

        public static let async = Config(synchronous: false)

        public static let noPooling = Config(enableOutputPooling: false)
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

        // Initialize output buffer pool if enabled
        if config.enableOutputPooling {
            self.outputBufferPool = OutputBufferPool(
                device: device,
                outputSpecs: executable.outputSpecs,
                poolSize: config.outputPoolSize
            )
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

        // Note: Semaphore removed - Metal command queues handle concurrent submission safely.
        // Compilation still uses semaphore (in MetalHLOCompiler) but execution doesn't need it.

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

        // Create a single encoder for all operations (reduces overhead significantly)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw IntegratedExecutorError.encoderCreationFailed
        }

        if let label = config.debugLabel {
            encoder.label = label
        }

        // Encode all operations to the single encoder
        for (index, opID) in executable.executionOrder.enumerated() {
            try encodeOperationToEncoder(
                opID,
                encoder: encoder,
                inputs: inputs,
                kernelTimings: &kernelTimings
            )

            // Add memory barrier between operations for data hazard protection
            // (Skip after last operation since there's nothing following)
            if index < executable.executionOrder.count - 1 {
                encoder.memoryBarrier(scope: .buffers)
            }
        }

        encoder.endEncoding()

        // Execute
        commandBuffer.commit()

        var gpuTimeMs: Double = 0
        if config.synchronous {
            commandBuffer.waitUntilCompleted()

            // Measure GPU time immediately after completion (before output extraction)
            gpuTimeMs = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000

            if let error = commandBuffer.error {
                throw IntegratedExecutorError.executionFailed(error.localizedDescription)
            }
        }

        // Extract outputs (this adds overhead that shouldn't be counted in GPU time)
        let outputs = try extractOutputs(inputs: inputs)

        let executionTimeMs = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000

        // Update statistics
        statistics.executionCount += 1
        statistics.totalExecutionTimeMs += executionTimeMs
        statistics.lastExecutionTimeMs = executionTimeMs

        return ExecutionResult(
            outputs: outputs,
            executionTimeMs: executionTimeMs,
            gpuTimeMs: gpuTimeMs,
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

        // Create a single encoder for all operations
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            completion(.failure(IntegratedExecutorError.encoderCreationFailed))
            return
        }

        if let label = config.debugLabel {
            encoder.label = label
        }

        do {
            for (index, opID) in executable.executionOrder.enumerated() {
                try encodeOperationToEncoder(
                    opID,
                    encoder: encoder,
                    inputs: inputs,
                    kernelTimings: &kernelTimings
                )

                // Add memory barrier between operations for data hazard protection
                if index < executable.executionOrder.count - 1 {
                    encoder.memoryBarrier(scope: .buffers)
                }
            }
        } catch {
            completion(.failure(error))
            return
        }

        encoder.endEncoding()

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

    /// Executes using Swift concurrency (async/await).
    ///
    /// This is the preferred method for modern Swift code. It uses the
    /// underlying async execution and wraps it in Swift's structured concurrency.
    ///
    /// - Parameter inputs: Dictionary mapping input names to Metal buffers.
    /// - Returns: Execution result with output buffers.
    /// - Throws: `IntegratedExecutorError` if execution fails.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func execute(inputs: [String: MTLBuffer]) async throws -> ExecutionResult {
        try await withCheckedThrowingContinuation { continuation in
            executeAsync(inputs: inputs) { result in
                switch result {
                case .success(let executionResult):
                    continuation.resume(returning: executionResult)
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Private Methods

    /// Encodes a single operation to an existing encoder.
    /// This avoids the overhead of creating/ending encoders for each operation.
    private func encodeOperationToEncoder(
        _ opID: OpID,
        encoder: MTLComputeCommandEncoder,
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

        // Set pipeline state (Metal driver optimizes consecutive same-pipeline dispatches)
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

        // Set threadgroup memory if needed (for operations like matmul and transpose)
        if let sharedMemSize = executable.sharedMemorySizes[opID], sharedMemSize > 0 {
            let bufferCount = executable.threadgroupBufferCounts[opID] ?? 1
            if bufferCount == 2 {
                // Two buffers (e.g., matmul with tileA and tileB)
                let tileSize = sharedMemSize / 2
                encoder.setThreadgroupMemoryLength(tileSize, index: 0)
                encoder.setThreadgroupMemoryLength(tileSize, index: 1)
            } else {
                // Single buffer (e.g., transpose)
                encoder.setThreadgroupMemoryLength(sharedMemSize, index: 0)
            }
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

        // Note: endEncoding() is NOT called here - it's called once after all operations
    }

    /// Resolves a buffer binding to an actual buffer and offset.
    /// Handles view resolution - if a tensor is a view of another tensor,
    /// resolves to the base tensor's memory location with appropriate offset.
    private func resolveBinding(
        _ binding: BufferBinding,
        inputs: [String: MTLBuffer]
    ) throws -> (MTLBuffer, Int) {
        switch binding.source {
        case .input(let name):
            // Check if this input is actually a view of another tensor
            let (baseTensorID, viewOffset) = executable.resolveViewChain(name)

            // Try to get from inputs first (for direct inputs)
            if let buffer = inputs[baseTensorID] {
                return (buffer, binding.offset + viewOffset)
            }

            // Original input lookup
            guard let buffer = inputs[name] else {
                throw IntegratedExecutorError.missingInput(name)
            }
            return (buffer, binding.offset)

        case .output(let name):
            // Resolve view chain to get base tensor and offset
            let (baseTensorID, viewOffset) = executable.resolveViewChain(name)

            // Outputs come from the unified buffer
            // First try the base tensor's offset (for views)
            if let outputOffset = executable.memoryPlan.tensorOffsets[baseTensorID] {
                return (unifiedBuffer, outputOffset + binding.offset + viewOffset)
            }

            // Fall back to original name lookup
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
    /// Handles views - if an output is a view, extracts from the base tensor location.
    private func extractOutputs(inputs: [String: MTLBuffer]) throws -> [String: MTLBuffer] {
        var outputs: [String: MTLBuffer] = [:]

        for (name, spec) in executable.outputSpecs {
            // Resolve view chain to get base tensor and offset
            let (baseTensorID, viewOffset) = executable.resolveViewChain(name)

            // First try base tensor's offset (for views)
            if let offset = executable.memoryPlan.tensorOffsets[baseTensorID] {
                // Get output buffer from pool or allocate new one
                let outputBuffer: MTLBuffer
                if let pool = outputBufferPool, let pooled = pool.acquire(name) {
                    outputBuffer = pooled
                } else {
                    // Fallback to allocation if pool disabled or exhausted
                    guard let newBuffer = device.makeBuffer(
                        length: spec.byteSize,
                        options: .storageModeShared
                    ) else {
                        throw IntegratedExecutorError.bufferAllocationFailed(size: spec.byteSize)
                    }
                    if let label = config.debugLabel {
                        newBuffer.label = "\(label)_output_\(name)"
                    }
                    outputBuffer = newBuffer
                }

                // Copy data from unified buffer (including view offset)
                memcpy(
                    outputBuffer.contents(),
                    unifiedBuffer.contents().advanced(by: offset + viewOffset),
                    spec.byteSize
                )

                outputs[name] = outputBuffer
            } else if let offset = executable.memoryPlan.tensorOffsets[name] {
                // Fall back to direct name lookup (non-view case)
                let outputBuffer: MTLBuffer
                if let pool = outputBufferPool, let pooled = pool.acquire(name) {
                    outputBuffer = pooled
                } else {
                    guard let newBuffer = device.makeBuffer(
                        length: spec.byteSize,
                        options: .storageModeShared
                    ) else {
                        throw IntegratedExecutorError.bufferAllocationFailed(size: spec.byteSize)
                    }
                    if let label = config.debugLabel {
                        newBuffer.label = "\(label)_output_\(name)"
                    }
                    outputBuffer = newBuffer
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

    /// Releases output buffers back to the pool for reuse.
    /// Call this when done processing outputs to enable buffer reuse.
    public func releaseOutputs(_ outputs: [String: MTLBuffer]) {
        outputBufferPool?.releaseAll(outputs)
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

// MARK: - Output Buffer Pool

/// Pool of pre-allocated output buffers for reuse across executions.
///
/// The pool maintains a set of buffers for each output tensor name.
/// When `acquire()` is called, it returns an available buffer from the pool
/// or allocates a new one if the pool is exhausted. When `release()` is called,
/// the buffer is returned to the pool for reuse.
///
/// This eliminates per-execution allocation overhead for repeated inference.
public final class OutputBufferPool: @unchecked Sendable {
    private let device: MTLDevice
    private let specs: [String: TensorSpec]
    private var pools: [String: [MTLBuffer]]
    private var inUse: [String: Set<ObjectIdentifier>]
    private let lock = NSLock()

    /// Creates a new output buffer pool.
    /// - Parameters:
    ///   - device: Metal device for buffer allocation.
    ///   - outputSpecs: Specifications for each output tensor.
    ///   - poolSize: Number of buffers to pre-allocate per output.
    public init(device: MTLDevice, outputSpecs: [String: TensorSpec], poolSize: Int = 3) {
        self.device = device
        self.specs = outputSpecs
        self.pools = [:]
        self.inUse = [:]

        // Pre-allocate buffers for each output
        for (name, spec) in outputSpecs {
            var buffers: [MTLBuffer] = []
            for i in 0..<poolSize {
                if let buffer = device.makeBuffer(length: spec.byteSize, options: .storageModeShared) {
                    buffer.label = "pooled_output_\(name)_\(i)"
                    buffers.append(buffer)
                }
            }
            pools[name] = buffers
            inUse[name] = []
        }
    }

    /// Acquires an output buffer for the given output name.
    /// Returns nil if the pool is exhausted and allocation fails.
    public func acquire(_ name: String) -> MTLBuffer? {
        lock.lock()
        defer { lock.unlock() }

        guard var available = pools[name], !available.isEmpty else {
            // Pool exhausted - try to allocate new buffer
            guard let spec = specs[name] else { return nil }
            return device.makeBuffer(length: spec.byteSize, options: .storageModeShared)
        }

        let buffer = available.removeLast()
        pools[name] = available
        inUse[name]?.insert(ObjectIdentifier(buffer))
        return buffer
    }

    /// Releases a buffer back to the pool.
    public func release(_ buffer: MTLBuffer, name: String) {
        lock.lock()
        defer { lock.unlock() }

        let id = ObjectIdentifier(buffer)
        guard inUse[name]?.contains(id) == true else { return }

        inUse[name]?.remove(id)
        pools[name]?.append(buffer)
    }

    /// Releases all buffers from a result dictionary back to the pool.
    public func releaseAll(_ outputs: [String: MTLBuffer]) {
        for (name, buffer) in outputs {
            release(buffer, name: name)
        }
    }

    /// Returns the total number of buffers in the pool (available + in use).
    public var totalBufferCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return pools.values.reduce(0) { $0 + $1.count } + inUse.values.reduce(0) { $0 + $1.count }
    }

    /// Returns the number of currently available buffers.
    public var availableBufferCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return pools.values.reduce(0) { $0 + $1.count }
    }
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
