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
    /// `var` (not `let`) because the METALHLO_DEBUG_NO_BUFFER_REUSE
    /// diagnostic mode swaps a fresh allocation in at the start of every
    /// execute() to test whether cross-call buffer reuse is what's
    /// causing #18 BERT cross-call NaN.
    private var unifiedBuffer: MTLBuffer

    /// Constant buffers (created once).
    private var constantBuffers: [String: MTLBuffer]

    /// Pre-resolved op plans in execution order. `executable.bindings[opID]`
    /// walks the StableHLO view chain and does multiple dictionary lookups
    /// per binding; that all resolves to a static answer (the offsets and
    /// constant buffers are fixed by the memory planner). Caching as a flat
    /// array indexed by execution position lets the hot loop skip both the
    /// view-chain walk *and* the `Dictionary` hashing per op.
    private let resolvedPlanOrdered: [ResolvedOpPlan]

    /// Pre-resolved binding for a single kernel argument slot. Keeps the
    /// per-op encode loop branch-light and avoids per-call dictionary work.
    fileprivate enum ResolvedBinding {
        /// `setBytes` of a 32-bit scalar (compile-time constant).
        case scalar(value: UInt32, index: Int)
        /// `setBuffer(unifiedBuffer, offset:)` — offset is precomputed.
        case unified(offset: Int, index: Int)
        /// `setBuffer(constantBuffer, offset:)` — constant buffer is fixed at
        /// init time.
        case constant(buffer: MTLBuffer, offset: Int, index: Int)
        /// `setBuffer(inputs[name], offset:)` — only this one needs a
        /// per-execute lookup. `name` is the *resolved* (post-view-chain)
        /// tensor id; `extraOffset` is the view byte offset baked in.
        case input(name: String, extraOffset: Int, index: Int)
    }

    /// Per-op cached dispatch + bindings. Reading the three executable
    /// dictionaries (pipelines / dispatches / bindings) on every op is what
    /// the encode loop used to spend most of its time on.
    fileprivate struct ResolvedOpPlan {
        let pipeline: MTLComputePipelineState
        let dispatch: DispatchConfig
        let bindings: [ResolvedBinding]
        /// (threadgroupMemorySize, bufferCount) — both zero when the op
        /// doesn't ask for shared memory.
        let sharedMemoryBytes: Int
        let threadgroupBufferCount: Int
        /// Original op ID, kept for diagnostics (`METALHLO_DEBUG_DISPATCH`).
        let opID: OpID
    }

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

        // Build the per-op resolved plan once. Walk every binding through
        // the view chain + memory plan now so the encode hot path skips
        // those lookups entirely.
        var planArr: [ResolvedOpPlan] = []
        planArr.reserveCapacity(executable.executionOrder.count)
        let memoryPlan = executable.memoryPlan
        let constants = executable.constantBuffers
        for opID in executable.executionOrder {
            guard let pipeline = executable.pipelines[opID],
                  let dispatch = executable.dispatches[opID],
                  let bindings = executable.bindings[opID] else { continue }
            var resolved: [ResolvedBinding] = []
            resolved.reserveCapacity(bindings.count)
            for binding in bindings {
                switch binding.source {
                case .scalar(let value):
                    resolved.append(.scalar(value: value, index: binding.index))
                case .unified(let offset):
                    resolved.append(.unified(offset: offset + binding.offset, index: binding.index))
                case .constant(let id):
                    guard let buf = constants[id] else { continue }
                    resolved.append(.constant(buffer: buf, offset: binding.offset, index: binding.index))
                case .input(let name):
                    let (baseName, viewOffset) = executable.resolveViewChain(name)
                    resolved.append(.input(name: baseName, extraOffset: binding.offset + viewOffset, index: binding.index))
                case .output(let name):
                    let (baseName, viewOffset) = executable.resolveViewChain(name)
                    if let off = memoryPlan.tensorOffsets[baseName] {
                        resolved.append(.unified(offset: off + binding.offset + viewOffset, index: binding.index))
                    } else if let off = memoryPlan.tensorOffsets[name] {
                        resolved.append(.unified(offset: off + binding.offset, index: binding.index))
                    } else {
                        // Direct output without a plan entry — fall back to
                        // the input slot the caller will provide.
                        resolved.append(.input(name: name, extraOffset: binding.offset, index: binding.index))
                    }
                case .threadgroup:
                    continue  // never bound as a buffer
                }
            }
            planArr.append(ResolvedOpPlan(
                pipeline: pipeline,
                dispatch: dispatch,
                bindings: resolved,
                sharedMemoryBytes: executable.sharedMemorySizes[opID] ?? 0,
                threadgroupBufferCount: executable.threadgroupBufferCounts[opID] ?? 1,
                opID: opID
            ))
        }
        self.resolvedPlanOrdered = planArr
    }

    // MARK: - Execution

    /// Executes the compiled executable with the given inputs.
    ///
    /// - Parameter inputs: Dictionary mapping input names to Metal buffers.
    /// - Returns: Execution result with output buffers.
    /// - Throws: `ExecutorError` if execution fails.
    public func execute(inputs: [String: MTLBuffer]) throws -> ExecutionResult {
        // Wrap the whole body in an autoreleasepool. MTLCommandBuffer (and the
        // resources it retains — pipeline state, buffer bindings, including the
        // *previous* unifiedBuffer that we replaced earlier this call) is an
        // NSObject and only gets released when its autorelease pool drains.
        // Without an explicit pool here, every execute() leaves the just-used
        // command buffer plus its unifiedBuffer retention alive until the next
        // top-level runloop turn — which never comes when JAX is calling us in
        // a tight Python loop. Measured at full nanoGPT scale: ~2.64 GB leak
        // per step until OOM at ~step 14 in a 51 GB machine.
        return try autoreleasepool {
            try executeImpl(inputs: inputs)
        }
    }

    private func executeImpl(inputs: [String: MTLBuffer]) throws -> ExecutionResult {
        let startTime = DispatchTime.now()

        // Note: Semaphore removed - Metal command queues handle concurrent submission safely.
        // Compilation still uses semaphore (in MetalHLOCompiler) but execution doesn't need it.

        // Validate inputs
        if config.validateInputs {
            try executable.validateInputs(inputs)
        }

        // unifiedBuffer reuse vs fresh-alloc-per-execute.
        //
        // The historic prior bug ("BERT cross-call NaN", documented in the
        // earlier comment) was that reusing the same MTLBuffer across
        // execute()s left scatter / gather / atomic-add updates from the
        // previous step visible to the next. Allocating fresh dodged that.
        //
        // Since then we've fixed the underlying ops:
        //   - scatter no longer relies on the bogus in-kernel copy phase
        //     (commit d25515b)
        //   - the autoreleasepool wrapper releases the previous
        //     command-buffer-retained buffer ARC reference per step
        //   - the memory planner places each tensor at a non-overlapping
        //     offset so old values never feed a fresh op
        //
        // Reusing the persistent buffer + host-memset is cheaper than
        // allocating + zero-faulting a fresh 24-MB-to-100-MB region per
        // step. Default to reuse; `METALHLO_FRESH_UNIFIED=1` restores
        // fresh-per-execute for diagnostics.
        if ProcessInfo.processInfo.environment["METALHLO_FRESH_UNIFIED"] == "1" {
            let bufferSize = unifiedBuffer.length
            if let fresh = device.makeBuffer(length: bufferSize, options: .storageModeShared) {
                fresh.label = unifiedBuffer.label
                unifiedBuffer = fresh
            }
        }
        memset(unifiedBuffer.contents(), 0, unifiedBuffer.length)

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

        // Encode all operations to the single encoder.
        //
        // The default dispatch type for `makeComputeCommandEncoder()` is
        // `.serial`, which means consecutive dispatches on the same encoder
        // already see each other's writes — Metal serializes them on the
        // device queue. Explicit `memoryBarrier(scope: .buffers)` between
        // every op is therefore redundant and just adds CPU encoding cost
        // (measured 288ms of CPU overhead vs 162ms of GPU compute on a
        // 2360-op nanoGPT train step). Set METALHLO_BARRIER_PER_OP=1 to
        // restore the per-op barrier if a regression appears.
        let perOpBarrier = ProcessInfo.processInfo.environment["METALHLO_BARRIER_PER_OP"] == "1"
        let debugDispatch = ProcessInfo.processInfo.environment["METALHLO_DEBUG_DISPATCH"] != nil
        let unified = unifiedBuffer
        let plan = resolvedPlanOrdered
        let lastIdx = plan.count - 1
        var lastPipeline: MTLComputePipelineState? = nil
        for i in 0..<plan.count {
            let opPlan = plan[i]
            // Skip the redundant setComputePipelineState API call when two
            // adjacent ops share a kernel. Each call has measurable CPU
            // cost even though the driver no-ops the state change itself.
            if opPlan.pipeline !== lastPipeline {
                encoder.setComputePipelineState(opPlan.pipeline)
                lastPipeline = opPlan.pipeline
            }

            for binding in opPlan.bindings {
                switch binding {
                case .scalar(let value, let index):
                    var v = value
                    encoder.setBytes(&v, length: MemoryLayout<UInt32>.size, index: index)
                case .unified(let offset, let index):
                    encoder.setBuffer(unified, offset: offset, index: index)
                case .constant(let buffer, let offset, let index):
                    encoder.setBuffer(buffer, offset: offset, index: index)
                case .input(let name, let extraOffset, let index):
                    guard let inputBuffer = inputs[name] else {
                        throw IntegratedExecutorError.missingInput(name)
                    }
                    encoder.setBuffer(inputBuffer, offset: extraOffset, index: index)
                }
            }

            if opPlan.sharedMemoryBytes > 0 {
                if opPlan.threadgroupBufferCount == 2 {
                    let tileSize = opPlan.sharedMemoryBytes / 2
                    encoder.setThreadgroupMemoryLength(tileSize, index: 0)
                    encoder.setThreadgroupMemoryLength(tileSize, index: 1)
                } else {
                    encoder.setThreadgroupMemoryLength(opPlan.sharedMemoryBytes, index: 0)
                }
            }

            let dispatch = opPlan.dispatch
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

            if debugDispatch {
                FileHandle.standardError.write(
                    "[disp] op=\(opPlan.opID) pipeline=\(opPlan.pipeline.label ?? "<unlabelled>") dispatch=tg(\(dispatch.gridSize.width)x\(dispatch.gridSize.height)x\(dispatch.gridSize.depth)) x \(dispatch.threadgroupSize.width)x\(dispatch.threadgroupSize.height)x\(dispatch.threadgroupSize.depth)\n"
                        .data(using: .utf8)!)
            }

            if perOpBarrier && i < lastIdx {
                encoder.memoryBarrier(scope: .buffers)
            }
        }
        _ = kernelTimings  // not used in the fast path — kept for executeAsync caller

        encoder.endEncoding()

        if ProcessInfo.processInfo.environment["METALHLO_DUMP_DISPATCH_SUMMARY"] != nil {
            var byPipeline: [String: (count: Int, threads: Int)] = [:]
            for opID in executable.executionOrder {
                let pipeline = executable.pipelines[opID]
                let label = pipeline?.label ?? "<unlabelled>"
                let dispatch = executable.dispatches[opID]
                let threads = dispatch.map { Int($0.gridSize.width * $0.gridSize.height * $0.gridSize.depth * $0.threadgroupSize.width * $0.threadgroupSize.height * $0.threadgroupSize.depth) } ?? 0
                let prev = byPipeline[label] ?? (count: 0, threads: 0)
                byPipeline[label] = (count: prev.count + 1, threads: prev.threads + threads)
            }
            let sorted = byPipeline.sorted { $0.value.threads > $1.value.threads }
            FileHandle.standardError.write("[dispatch_summary] \(executable.executionOrder.count) ops in this execute()\n".data(using: .utf8)!)
            for (label, info) in sorted.prefix(20) {
                FileHandle.standardError.write(String(format: "  %-40s  count=%4d  total_threads=%12d\n", label.prefix(40).description, info.count, info.threads).data(using: .utf8)!)
            }
        }

        // Execute
        commandBuffer.commit()

        // extractOutputs() below reads unifiedBuffer on the host
        // (via memcpy), and the next call's memset(unifiedBuffer, ...)
        // mutates it from the host. Both require GPU writes to have
        // completed first. The synchronous path already waited; the
        // async path was missing this wait, which allowed the next
        // call's memset to race still-running kernels mid-flight. Wait
        // unconditionally — the async path no longer hides latency but
        // it is now correct (#18 BERT cross-call NaN context).
        var gpuTimeMs: Double = 0
        commandBuffer.waitUntilCompleted()
        if config.synchronous {
            gpuTimeMs = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000
        }
        if ProcessInfo.processInfo.environment["METALHLO_PROFILE_GPU"] == "1" {
            // GPUStartTime / GPUEndTime are wall-clock seconds — only the GPU
            // execution slice between commit and completion, no CPU dispatch.
            let gpuOnly = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000.0
            FileHandle.standardError.write(String(format: "[gpu_time] ops=%d wall=%.2fms gpu=%.2fms\n",
                executable.executionOrder.count, gpuTimeMs, gpuOnly).data(using: .utf8)!)
        }
        if let error = commandBuffer.error {
            throw IntegratedExecutorError.executionFailed(error.localizedDescription)
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

        // Debug logging — when METALHLO_DEBUG_DISPATCH is set, log every
        // kernel dispatch with the buffers and offsets it binds plus the
        // grid/threadgroup geometry. Lets us cross-reference an in-flight
        // GPU page fault to the offending dispatch.
        let debugDispatch = ProcessInfo.processInfo.environment["METALHLO_DEBUG_DISPATCH"] != nil

        // Bind all buffers and scalars
        for binding in bindings {
            switch binding.source {
            case .scalar(let value):
                // Use setBytes for scalar uniform values
                var scalarValue = value
                encoder.setBytes(&scalarValue, length: MemoryLayout<UInt32>.size, index: binding.index)
                if debugDispatch {
                    FileHandle.standardError.write(
                        "[disp] op=\(opID) bind[\(binding.index)]=scalar(\(scalarValue))\n"
                            .data(using: .utf8)!)
                }
            default:
                // Use setBuffer for all other sources
                let (buffer, offset) = try resolveBinding(binding, inputs: inputs)
                encoder.setBuffer(buffer, offset: offset, index: binding.index)
                if debugDispatch {
                    let bufLen = buffer.length
                    let bufId = ObjectIdentifier(buffer).hashValue & 0xFFFFFF
                    FileHandle.standardError.write(
                        "[disp] op=\(opID) bind[\(binding.index)]=buf(id=\(String(bufId, radix: 16)),len=\(bufLen),off=\(offset),source=\(binding.source))\n"
                            .data(using: .utf8)!)
                }
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
            if debugDispatch {
                FileHandle.standardError.write(
                    "[disp] op=\(opID) pipeline=\(pipeline.label ?? "<unlabelled>") dispatch=threads(\(totalThreads.width)x\(totalThreads.height)x\(totalThreads.depth)) tg=\(dispatch.threadgroupSize.width)x\(dispatch.threadgroupSize.height)x\(dispatch.threadgroupSize.depth)\n"
                        .data(using: .utf8)!)
            }
            encoder.dispatchThreads(totalThreads, threadsPerThreadgroup: dispatch.threadgroupSize)
        } else {
            if debugDispatch {
                FileHandle.standardError.write(
                    "[disp] op=\(opID) pipeline=\(pipeline.label ?? "<unlabelled>") dispatch=tg(\(dispatch.gridSize.width)x\(dispatch.gridSize.height)x\(dispatch.gridSize.depth)) x \(dispatch.threadgroupSize.width)x\(dispatch.threadgroupSize.height)x\(dispatch.threadgroupSize.depth)\n"
                        .data(using: .utf8)!)
            }
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
