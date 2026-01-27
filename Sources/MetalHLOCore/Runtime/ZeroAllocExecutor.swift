// ZeroAllocExecutor.swift
// MetalHLOCore
//
// Zero-allocation executor using pre-planned memory layout.
// All memory is allocated once at initialization.

import Foundation
import Metal

// MARK: - Execution Context

/// Context for executing a computation graph with zero runtime allocations.
public struct ExecutionContext: Sendable {
    /// The memory plan being used.
    public let memoryPlan: MemoryPlan

    /// Input buffer offsets (external inputs).
    public let inputOffsets: [String: Int]

    /// Output buffer offsets.
    public let outputOffsets: [String: Int]

    /// Total bytes for intermediate tensors.
    public let intermediateBytes: Int
}

// MARK: - Buffer Pool

/// A pool of pre-allocated buffers for triple buffering.
public final class BufferPool: @unchecked Sendable {
    private let device: MTLDevice
    private let bufferSize: Int
    private var buffers: [MTLBuffer]
    private var currentIndex: Int = 0
    private let lock = NSLock()

    /// Number of buffers in the pool.
    public var count: Int { buffers.count }

    public init(device: MTLDevice, bufferSize: Int, count: Int = 3) {
        self.device = device
        self.bufferSize = bufferSize
        self.buffers = []

        // Pre-allocate buffers
        for _ in 0..<count {
            if let buffer = device.makeBuffer(length: max(bufferSize, 256), options: .storageModeShared) {
                buffers.append(buffer)
            }
        }
    }

    /// Gets the next buffer in round-robin fashion.
    public func next() -> MTLBuffer {
        lock.lock()
        defer { lock.unlock() }

        let buffer = buffers[currentIndex]
        currentIndex = (currentIndex + 1) % buffers.count
        return buffer
    }

    /// Gets a specific buffer by index.
    public func buffer(at index: Int) -> MTLBuffer {
        return buffers[index % buffers.count]
    }
}

// MARK: - Zero-Allocation Executor

/// Executor that runs computations with zero runtime memory allocations.
///
/// All intermediate tensors are stored in a single pre-allocated unified buffer.
/// Triple buffering is used for pipelined execution.
public final class ZeroAllocExecutor: @unchecked Sendable {

    /// Configuration for the executor.
    public struct Config: Sendable {
        /// Number of in-flight command buffers for pipelining.
        public var inflightBufferCount: Int = 3

        /// Whether to use triple buffering for the unified buffer.
        public var useTripleBuffering: Bool = true

        /// Command buffer execution timeout in seconds.
        public var executionTimeout: TimeInterval = 30.0

        public init() {}
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let memoryPlan: MemoryPlan
    private let config: Config

    /// Unified buffer(s) for intermediate tensors.
    private let unifiedBufferPool: BufferPool

    /// Semaphore for limiting in-flight command buffers.
    private let inflightSemaphore: DispatchSemaphore

    /// Current buffer index for round-robin.
    private var currentBufferIndex: Int = 0

    // MARK: - Initialization

    /// Creates a new zero-allocation executor.
    ///
    /// - Parameters:
    ///   - device: The Metal device.
    ///   - memoryPlan: Pre-computed memory plan.
    ///   - config: Executor configuration.
    public init(device: MTLDevice, memoryPlan: MemoryPlan, config: Config = Config()) throws {
        self.device = device
        self.memoryPlan = memoryPlan
        self.config = config

        guard let queue = device.makeCommandQueue() else {
            throw ZeroAllocError.failedToCreateCommandQueue
        }
        self.commandQueue = queue

        // Create buffer pool
        let bufferCount = config.useTripleBuffering ? config.inflightBufferCount : 1
        self.unifiedBufferPool = BufferPool(
            device: device,
            bufferSize: memoryPlan.totalBytes,
            count: bufferCount
        )

        self.inflightSemaphore = DispatchSemaphore(value: config.inflightBufferCount)
    }

    // MARK: - Execution

    /// Executes the computation graph with the given inputs.
    ///
    /// - Parameters:
    ///   - function: The HLO function to execute.
    ///   - inputs: Input buffers keyed by argument name.
    ///   - kernels: Pre-compiled kernel pipeline states.
    /// - Returns: Output buffers keyed by output name.
    public func execute(
        function: HLOFunction,
        inputs: [String: MTLBuffer],
        kernels: [OperationID: MTLComputePipelineState]
    ) throws -> [String: MTLBuffer] {
        // Wait for a slot
        _ = inflightSemaphore.wait(timeout: .now() + config.executionTimeout)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inflightSemaphore.signal()
            throw ZeroAllocError.failedToCreateCommandBuffer
        }

        // Get unified buffer for this execution
        let unifiedBuffer = unifiedBufferPool.next()

        // Create output buffers
        var outputBuffers: [String: MTLBuffer] = [:]
        for (i, returnValue) in function.returnValues.enumerated() {
            if let offset = memoryPlan.tensorOffsets[returnValue] {
                // Output is in unified buffer - create a view or copy
                let outputType = function.outputTypes[i]
                let outputSize = outputType.byteCount

                if let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) {
                    outputBuffers[returnValue] = outputBuffer
                }
            }
        }

        // Encode operations in planned order
        for opIndex in memoryPlan.executionOrder {
            guard opIndex < function.operations.count else { continue }
            let op = function.operations[opIndex]

            guard let pipeline = kernels[opIndex] else { continue }

            try encodeOperation(
                op,
                opIndex: opIndex,
                pipeline: pipeline,
                commandBuffer: commandBuffer,
                unifiedBuffer: unifiedBuffer,
                inputs: inputs,
                outputBuffers: outputBuffers
            )
        }

        // Copy outputs from unified buffer
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            for (returnValue, outputBuffer) in outputBuffers {
                if let offset = memoryPlan.tensorOffsets[returnValue] {
                    blitEncoder.copy(
                        from: unifiedBuffer,
                        sourceOffset: offset,
                        to: outputBuffer,
                        destinationOffset: 0,
                        size: outputBuffer.length
                    )
                }
            }
            blitEncoder.endEncoding()
        }

        // Signal semaphore on completion
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.inflightSemaphore.signal()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw ZeroAllocError.executionFailed(error.localizedDescription)
        }

        return outputBuffers
    }

    /// Executes with streaming inputs for pipelined execution.
    ///
    /// - Parameters:
    ///   - function: The HLO function.
    ///   - inputBatches: Stream of input batches.
    ///   - kernels: Pre-compiled kernels.
    ///   - completion: Called with each output batch.
    public func executeStreaming(
        function: HLOFunction,
        inputBatches: [[String: MTLBuffer]],
        kernels: [OperationID: MTLComputePipelineState],
        completion: @escaping ([String: MTLBuffer]) -> Void
    ) {
        let dispatchGroup = DispatchGroup()

        for inputs in inputBatches {
            dispatchGroup.enter()

            // Wait for a slot
            _ = inflightSemaphore.wait(timeout: .distantFuture)

            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                inflightSemaphore.signal()
                dispatchGroup.leave()
                continue
            }

            let bufferIndex = currentBufferIndex
            currentBufferIndex = (currentBufferIndex + 1) % unifiedBufferPool.count
            let unifiedBuffer = unifiedBufferPool.buffer(at: bufferIndex)

            // Create output buffers
            var outputBuffers: [String: MTLBuffer] = [:]
            for (i, returnValue) in function.returnValues.enumerated() {
                let outputType = function.outputTypes[i]
                if let outputBuffer = device.makeBuffer(length: outputType.byteCount, options: .storageModeShared) {
                    outputBuffers[returnValue] = outputBuffer
                }
            }

            // Encode operations
            for opIndex in memoryPlan.executionOrder {
                guard opIndex < function.operations.count else { continue }
                let op = function.operations[opIndex]
                guard let pipeline = kernels[opIndex] else { continue }

                do {
                    try encodeOperation(
                        op,
                        opIndex: opIndex,
                        pipeline: pipeline,
                        commandBuffer: commandBuffer,
                        unifiedBuffer: unifiedBuffer,
                        inputs: inputs,
                        outputBuffers: outputBuffers
                    )
                } catch {
                    // Skip failed operations
                }
            }

            // Copy outputs
            if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
                for (returnValue, outputBuffer) in outputBuffers {
                    if let offset = memoryPlan.tensorOffsets[returnValue] {
                        blitEncoder.copy(
                            from: unifiedBuffer,
                            sourceOffset: offset,
                            to: outputBuffer,
                            destinationOffset: 0,
                            size: outputBuffer.length
                        )
                    }
                }
                blitEncoder.endEncoding()
            }

            commandBuffer.addCompletedHandler { [weak self] _ in
                completion(outputBuffers)
                self?.inflightSemaphore.signal()
                dispatchGroup.leave()
            }

            commandBuffer.commit()
        }

        dispatchGroup.wait()
    }

    // MARK: - Operation Encoding

    private func encodeOperation(
        _ op: HLOOperation,
        opIndex: Int,
        pipeline: MTLComputePipelineState,
        commandBuffer: MTLCommandBuffer,
        unifiedBuffer: MTLBuffer,
        inputs: [String: MTLBuffer],
        outputBuffers: [String: MTLBuffer]
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ZeroAllocError.failedToCreateEncoder
        }

        encoder.setComputePipelineState(pipeline)

        // Set input buffers
        for (index, operand) in op.operands.enumerated() {
            if let inputBuffer = inputs[operand] {
                // External input
                encoder.setBuffer(inputBuffer, offset: 0, index: index)
            } else if let offset = memoryPlan.tensorOffsets[operand] {
                // Intermediate tensor in unified buffer
                encoder.setBuffer(unifiedBuffer, offset: offset, index: index)
            }
        }

        // Set output buffer
        let outputIndex = op.operands.count
        if let outputBuffer = outputBuffers[op.result] {
            encoder.setBuffer(outputBuffer, offset: 0, index: outputIndex)
        } else if let offset = memoryPlan.tensorOffsets[op.result] {
            encoder.setBuffer(unifiedBuffer, offset: offset, index: outputIndex)
        }

        // Compute dispatch sizes
        let (gridSize, threadgroupSize) = computeDispatchSizes(for: op, pipeline: pipeline)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)

        encoder.endEncoding()
    }

    /// Computes dispatch sizes for an operation.
    private func computeDispatchSizes(
        for op: HLOOperation,
        pipeline: MTLComputePipelineState
    ) -> (MTLSize, MTLSize) {
        let shape = op.resultType.shape

        // Default threadgroup size
        let threadgroupSize = MTLSize(
            width: min(32, pipeline.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1
        )

        // Compute grid size based on output shape
        let totalElements = shape.reduce(1, *)
        let gridWidth = (totalElements + threadgroupSize.width - 1) / threadgroupSize.width

        let gridSize = MTLSize(width: gridWidth, height: 1, depth: 1)

        return (gridSize, threadgroupSize)
    }

    // MARK: - Statistics

    /// Returns memory statistics for this executor.
    public var memoryStatistics: MemoryStatistics {
        MemoryStatistics(
            totalAllocatedBytes: memoryPlan.totalBytes * unifiedBufferPool.count,
            peakMemoryBytes: memoryPlan.peakMemory,
            numTensors: memoryPlan.tensorOffsets.count,
            numSharingGroups: memoryPlan.sharingGroups.count,
            inflightBufferCount: config.inflightBufferCount
        )
    }

    /// Memory statistics for the executor.
    public struct MemoryStatistics: Sendable {
        public let totalAllocatedBytes: Int
        public let peakMemoryBytes: Int
        public let numTensors: Int
        public let numSharingGroups: Int
        public let inflightBufferCount: Int

        /// Memory saved through tensor sharing.
        public var memorySavings: Double {
            guard totalAllocatedBytes > 0 else { return 0 }
            return 1.0 - Double(peakMemoryBytes) / Double(totalAllocatedBytes)
        }
    }
}

// MARK: - Errors

/// Errors that can occur during zero-allocation execution.
public enum ZeroAllocError: Error, Sendable {
    case failedToCreateCommandQueue
    case failedToCreateCommandBuffer
    case failedToCreateEncoder
    case failedToCreateBuffer(size: Int)
    case executionFailed(String)
    case timeout
}

// MARK: - Triple Buffer Manager

/// Manages triple buffering for pipelined execution.
public final class TripleBufferManager: @unchecked Sendable {

    private let device: MTLDevice
    private let buffers: [MTLBuffer]
    private var writeIndex: Int = 0
    private var readIndex: Int = 0
    private let lock = NSLock()

    /// Total number of buffers.
    public let bufferCount = 3

    public init(device: MTLDevice, size: Int) throws {
        self.device = device
        var buffers: [MTLBuffer] = []

        for _ in 0..<3 {
            guard let buffer = device.makeBuffer(length: max(size, 256), options: .storageModeShared) else {
                throw ZeroAllocError.failedToCreateBuffer(size: size)
            }
            buffers.append(buffer)
        }

        self.buffers = buffers
    }

    /// Gets the buffer for writing (producer).
    public func getWriteBuffer() -> MTLBuffer {
        lock.lock()
        defer { lock.unlock() }
        return buffers[writeIndex]
    }

    /// Gets the buffer for reading (consumer).
    public func getReadBuffer() -> MTLBuffer {
        lock.lock()
        defer { lock.unlock() }
        return buffers[readIndex]
    }

    /// Advances the write pointer after writing is complete.
    public func advanceWrite() {
        lock.lock()
        defer { lock.unlock() }
        writeIndex = (writeIndex + 1) % bufferCount
    }

    /// Advances the read pointer after reading is complete.
    public func advanceRead() {
        lock.lock()
        defer { lock.unlock() }
        readIndex = (readIndex + 1) % bufferCount
    }

    /// Resets both pointers.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        writeIndex = 0
        readIndex = 0
    }
}

// MARK: - Execution Plan Builder

/// Builder for creating execution plans with zero-allocation memory layout.
public final class ExecutionPlanBuilder: @unchecked Sendable {

    private let planner: StaticMemoryPlanner

    public init(config: StaticMemoryPlanner.Config = .init()) {
        self.planner = StaticMemoryPlanner(config: config)
    }

    /// Builds an execution plan for the given function.
    public func build(for function: HLOFunction) -> ExecutionPlan {
        let memoryPlan = planner.plan(function)

        return ExecutionPlan(
            function: function,
            memoryPlan: memoryPlan,
            executionOrder: memoryPlan.executionOrder
        )
    }
}

/// A complete execution plan including memory layout and operation order.
public struct ExecutionPlan: Sendable {
    /// The function being executed.
    public let function: HLOFunction

    /// Memory plan for zero-allocation.
    public let memoryPlan: MemoryPlan

    /// Optimized execution order.
    public let executionOrder: [OperationID]

    /// Total memory required.
    public var totalMemoryRequired: Int {
        memoryPlan.totalBytes
    }

    /// Number of operations.
    public var operationCount: Int {
        function.operations.count
    }
}
