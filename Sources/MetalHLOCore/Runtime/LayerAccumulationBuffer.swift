// LayerAccumulationBuffer.swift
// MetalHLOCore
//
// Manages persistent GPU buffers that accumulate layer representations across
// function invocations for Attention Residuals (Block AttnRes).
//
// Block AttnRes requires maintaining a stack of hidden states from block
// boundaries so that the depth-attention kernel can read all prior blocks.
// Since MetalHLO executes each function independently, this buffer provides
// the cross-invocation state needed to bridge that gap.
//
// Memory budget: For a GPT-2 scale model (hidden=768, seq=1024, 8 blocks),
// the stack is 8 × 1024 × 768 × 4 bytes = 25 MB in float32. Easily fits
// on M1's unified memory.

import Metal
import Foundation

/// Configuration for a layer accumulation buffer.
public struct LayerAccumulationConfig: Sendable {
    /// Maximum number of block-boundary snapshots to store.
    public let maxDepth: Int

    /// Shape of each layer snapshot: [batch, seq, hidden] or [batch, hidden].
    public let snapshotShape: [Int]

    /// Element type for stored tensors.
    public let elementType: ElementType

    /// Total bytes per snapshot.
    public var snapshotBytes: Int {
        snapshotShape.reduce(1, *) * elementType.byteSize
    }

    /// Total bytes for the full stack at max depth.
    public var totalBytes: Int {
        maxDepth * snapshotBytes
    }

    public init(maxDepth: Int, snapshotShape: [Int], elementType: ElementType = .float32) {
        self.maxDepth = maxDepth
        self.snapshotShape = snapshotShape
        self.elementType = elementType
    }
}

/// A persistent GPU buffer that accumulates layer outputs across function calls.
///
/// Usage pattern (per inference step):
/// 1. After each block, call `push(layerOutput)` to store the block's output.
/// 2. At block boundaries, call `stackedBuffer()` to get a single buffer
///    containing all accumulated layers as [currentDepth, ...snapshotShape].
/// 3. Pass the stacked buffer as keys/values to the depth-attention kernel.
/// 4. After inference completes, call `reset()` to prepare for the next input.
///
/// The buffer uses a single contiguous MTLBuffer and writes each snapshot
/// at offset `currentDepth * snapshotBytes`. No copies or reallocations occur
/// during accumulation — the stacked buffer is just the underlying MTLBuffer
/// with an updated shape interpretation.
public final class LayerAccumulationBuffer: @unchecked Sendable {

    // MARK: - Properties

    /// The Metal device.
    private let device: MTLDevice

    /// Configuration for this buffer.
    public let config: LayerAccumulationConfig

    /// The backing MTLBuffer (contiguous, holds all snapshots).
    private let buffer: MTLBuffer

    /// Current number of stored snapshots.
    private(set) public var currentDepth: Int = 0

    /// Lock for thread safety.
    private let lock = NSLock()

    // MARK: - Initialization

    /// Creates a new accumulation buffer.
    ///
    /// - Parameters:
    ///   - config: Buffer configuration (max depth, snapshot shape, element type).
    ///   - device: Metal device for buffer allocation.
    /// - Throws: If buffer allocation fails.
    public init(config: LayerAccumulationConfig, device: MTLDevice) throws {
        self.device = device
        self.config = config

        guard let buf = device.makeBuffer(length: config.totalBytes, options: .storageModeShared) else {
            throw LayerAccumulationError.allocationFailed(config.totalBytes)
        }
        self.buffer = buf
    }

    // MARK: - Accumulation

    /// Pushes a layer snapshot onto the stack.
    ///
    /// The snapshot data is copied into the backing buffer at the current
    /// depth offset. The snapshot must match `config.snapshotShape`.
    ///
    /// - Parameter snapshot: The layer output as a BufferStorage.
    /// - Throws: If the buffer is full or the snapshot shape doesn't match.
    public func push(_ snapshot: BufferStorage) throws {
        lock.lock()
        defer { lock.unlock() }

        guard currentDepth < config.maxDepth else {
            throw LayerAccumulationError.bufferFull(config.maxDepth)
        }

        guard snapshot.shape == config.snapshotShape else {
            throw LayerAccumulationError.shapeMismatch(
                expected: config.snapshotShape, got: snapshot.shape
            )
        }

        let offset = currentDepth * config.snapshotBytes

        // Copy snapshot data into the backing buffer at the correct offset
        if let srcBuffer = snapshot.metalBuffer {
            // GPU→GPU: blit copy would be ideal, but for unified memory a memcpy
            // from the shared buffer is effectively zero-cost.
            let dst = buffer.contents().advanced(by: offset)
            let src = srcBuffer.contents()
            memcpy(dst, src, config.snapshotBytes)
        } else {
            // Copy from Data
            let dst = buffer.contents().advanced(by: offset)
            snapshot.data.withUnsafeBytes { ptr in
                memcpy(dst, ptr.baseAddress!, min(ptr.count, config.snapshotBytes))
            }
        }

        currentDepth += 1
    }

    /// Pushes a layer snapshot directly from an MTLBuffer.
    ///
    /// - Parameters:
    ///   - metalBuffer: Source buffer containing the snapshot data.
    ///   - byteCount: Number of bytes to copy (must equal snapshotBytes).
    /// - Throws: If the buffer is full or byte count doesn't match.
    public func push(metalBuffer: MTLBuffer, byteCount: Int) throws {
        lock.lock()
        defer { lock.unlock() }

        guard currentDepth < config.maxDepth else {
            throw LayerAccumulationError.bufferFull(config.maxDepth)
        }

        guard byteCount == config.snapshotBytes else {
            throw LayerAccumulationError.byteMismatch(
                expected: config.snapshotBytes, got: byteCount
            )
        }

        let offset = currentDepth * config.snapshotBytes
        let dst = buffer.contents().advanced(by: offset)
        let src = metalBuffer.contents()
        memcpy(dst, src, byteCount)

        currentDepth += 1
    }

    // MARK: - Access

    /// Returns the backing MTLBuffer containing all accumulated snapshots.
    ///
    /// The buffer is interpreted as shape [currentDepth, ...snapshotShape].
    /// Only the first `currentDepth` snapshots contain valid data.
    public var stackedBuffer: MTLBuffer {
        buffer
    }

    /// Returns the shape of the stacked tensor: [currentDepth, ...snapshotShape].
    public var stackedShape: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return [currentDepth] + config.snapshotShape
    }

    /// Returns the byte count of valid data in the stacked buffer.
    public var validByteCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return currentDepth * config.snapshotBytes
    }

    /// Wraps the stacked buffer as a BufferStorage for use with MetalExecutor.
    ///
    /// - Returns: A BufferStorage with shape [currentDepth, ...snapshotShape].
    public func asBufferStorage() -> BufferStorage {
        lock.lock()
        let depth = currentDepth
        lock.unlock()

        let shape = [depth] + config.snapshotShape
        let largeTensor = LargeTensorStorage(
            buffer: buffer,
            shape: shape,
            elementType: config.elementType
        )
        return BufferStorage(largeTensor: largeTensor, device: device)
    }

    // MARK: - Lifecycle

    /// Resets the buffer for a new inference pass.
    ///
    /// Does not deallocate — just resets the depth counter.
    /// Previous data remains in the buffer but will be overwritten.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        currentDepth = 0
    }

    /// Returns whether the buffer has any accumulated snapshots.
    public var isEmpty: Bool {
        lock.lock()
        defer { lock.unlock() }
        return currentDepth == 0
    }

    /// Returns whether the buffer is at maximum capacity.
    public var isFull: Bool {
        lock.lock()
        defer { lock.unlock() }
        return currentDepth >= config.maxDepth
    }
}

// MARK: - Batched Depth Attention Encoder

/// Encodes multiple depth-attention dispatches into a single command buffer,
/// amortizing the ~200µs Metal command buffer creation/commit overhead across
/// all calls within an inference pass.
///
/// Usage:
/// ```
/// let encoder = DepthAttentionEncoder(device: device, commandQueue: queue)
/// // During inference, at each block boundary:
/// encoder.encodeDepthAttention(query: qBuf, keys: kvBuf, values: kvBuf,
///                               output: oBuf, params: params)
/// // After all blocks:
/// encoder.commitAndWait()  // Single GPU submission
/// ```
public final class DepthAttentionEncoder: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var commandBuffer: MTLCommandBuffer?
    private var encoder: MTLComputeCommandEncoder?
    private var dispatchCount: Int = 0

    /// Creates a new batched encoder.
    /// - Parameters:
    ///   - device: Metal device.
    ///   - commandQueue: Command queue for submission.
    /// - Throws: If kernel compilation fails.
    public init(device: MTLDevice, commandQueue: MTLCommandQueue) throws {
        self.device = device
        self.commandQueue = commandQueue

        let registry = MetalKernelRegistry.shared
        guard let kernel = registry.getKernel("depth_attention") else {
            throw KernelError.kernelNotFound("depth_attention")
        }
        self.pipeline = try registry.getPipeline(for: kernel, device: device)
    }

    /// Starts a new batch. Call before encoding dispatches.
    public func begin() throws {
        guard let cb = commandQueue.makeCommandBuffer() else {
            throw KernelError.commandBufferCreationFailed
        }
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw KernelError.encoderCreationFailed
        }
        enc.setComputePipelineState(pipeline)
        self.commandBuffer = cb
        self.encoder = enc
        self.dispatchCount = 0
    }

    /// Encodes one depth-attention dispatch into the current command buffer.
    ///
    /// - Parameters:
    ///   - query: Query buffer [batch, hidden].
    ///   - keys: Keys buffer [batch, depth, hidden].
    ///   - values: Values buffer [batch, depth, hidden].
    ///   - output: Output buffer [batch, hidden].
    ///   - params: Kernel parameters.
    public func encodeDepthAttention(
        query: MTLBuffer, keys: MTLBuffer, values: MTLBuffer,
        output: MTLBuffer, params: DepthAttentionParams
    ) {
        guard let enc = encoder else {
            fatalError("DepthAttentionEncoder: call begin() before encoding")
        }

        let numSG = DepthAttentionKernel.optimalSimdgroups(hiddenDim: params.hiddenDim)
        var gpuParams = GPUDepthAttentionParamsPublic(
            batch_size: UInt32(params.batchSize),
            depth_dim: UInt32(params.depthDim),
            hidden_dim: UInt32(params.hiddenDim),
            scale: params.scale,
            num_simdgroups: UInt32(numSG)
        )

        let sharedMemSize = (numSG * 32 + 32) * MemoryLayout<Float>.size
        let simdWidth = 32
        let threadsPerGroup = numSG * simdWidth

        enc.setBuffer(query, offset: 0, index: 0)
        enc.setBuffer(keys, offset: 0, index: 1)
        enc.setBuffer(values, offset: 0, index: 2)
        enc.setBuffer(output, offset: 0, index: 3)
        enc.setBytes(&gpuParams, length: MemoryLayout<GPUDepthAttentionParamsPublic>.size, index: 4)
        enc.setThreadgroupMemoryLength(sharedMemSize, index: 0)

        let gridSize = MTLSize(width: 1, height: params.batchSize, depth: 1)
        let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)

        dispatchCount += 1
    }

    /// Commits all encoded dispatches and waits for GPU completion.
    /// This is where the single command buffer submission happens.
    public func commitAndWait() {
        encoder?.endEncoding()
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        encoder = nil
        commandBuffer = nil
    }

    /// Number of dispatches encoded in the current batch.
    public var pendingDispatches: Int { dispatchCount }
}

/// Public GPU params (must match Metal shader layout).
/// Separate from the private one in the kernel file for encoder access.
public struct GPUDepthAttentionParamsPublic {
    public var batch_size: UInt32
    public var depth_dim: UInt32
    public var hidden_dim: UInt32
    public var scale: Float
    public var num_simdgroups: UInt32

    public init(batch_size: UInt32, depth_dim: UInt32, hidden_dim: UInt32, scale: Float, num_simdgroups: UInt32) {
        self.batch_size = batch_size
        self.depth_dim = depth_dim
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.num_simdgroups = num_simdgroups
    }
}

// MARK: - Errors

/// Errors from layer accumulation operations.
public enum LayerAccumulationError: Error, CustomStringConvertible {
    case allocationFailed(Int)
    case bufferFull(Int)
    case shapeMismatch(expected: [Int], got: [Int])
    case byteMismatch(expected: Int, got: Int)

    public var description: String {
        switch self {
        case .allocationFailed(let bytes):
            return "Failed to allocate layer accumulation buffer (\(bytes) bytes)"
        case .bufferFull(let maxDepth):
            return "Layer accumulation buffer is full (max depth: \(maxDepth))"
        case .shapeMismatch(let expected, let got):
            return "Snapshot shape mismatch: expected \(expected), got \(got)"
        case .byteMismatch(let expected, let got):
            return "Byte count mismatch: expected \(expected), got \(got)"
        }
    }
}

// MARK: - MetalExecutor Extension

extension MetalExecutor {

    /// Creates a layer accumulation buffer for Attention Residuals.
    ///
    /// - Parameters:
    ///   - maxDepth: Maximum number of block boundaries (e.g. 8 for 8 blocks).
    ///   - snapshotShape: Shape of each layer output (e.g. [batch, seq, hidden]).
    ///   - elementType: Element type (default: float32).
    /// - Returns: A new accumulation buffer.
    /// - Throws: If allocation fails.
    public func createLayerAccumulationBuffer(
        maxDepth: Int,
        snapshotShape: [Int],
        elementType: ElementType = .float32
    ) throws -> LayerAccumulationBuffer {
        let config = LayerAccumulationConfig(
            maxDepth: maxDepth,
            snapshotShape: snapshotShape,
            elementType: elementType
        )
        return try LayerAccumulationBuffer(config: config, device: device)
    }

    /// Executes depth attention using an accumulation buffer as keys/values.
    ///
    /// - Parameters:
    ///   - query: The learned query vector as a BufferStorage [batch, hidden].
    ///   - accumulator: The layer accumulation buffer containing stacked block outputs.
    ///   - scale: Attention scale factor (default: 1/sqrt(hiddenDim)).
    /// - Returns: The depth-attention output [batch, hidden].
    /// - Throws: If execution fails.
    public func executeDepthAttention(
        query: BufferStorage,
        accumulator: LayerAccumulationBuffer,
        scale: Float? = nil
    ) throws -> BufferStorage {
        guard !accumulator.isEmpty else {
            throw ExecutorError.generalError("Layer accumulation buffer is empty")
        }

        guard let qBuffer = query.metalBuffer else {
            throw ExecutorError.generalError("Depth attention query requires MTLBuffer")
        }

        let hiddenDim = accumulator.config.snapshotShape.last ?? 1
        let batchSize: Int
        if accumulator.config.snapshotShape.count >= 2 {
            batchSize = accumulator.config.snapshotShape.dropLast().reduce(1, *)
        } else {
            batchSize = 1
        }

        let outByteCount = batchSize * hiddenDim * accumulator.config.elementType.byteSize
        guard let outBuffer = device.makeBuffer(length: outByteCount, options: .storageModeShared) else {
            throw ExecutorError.generalError("Failed to allocate depth attention output buffer")
        }

        let params = DepthAttentionParams(
            batchSize: batchSize,
            depthDim: accumulator.currentDepth,
            hiddenDim: hiddenDim,
            scale: scale
        )

        guard let commandQueue = device.makeCommandQueue() else {
            throw ExecutorError.commandQueueCreationFailed
        }

        try MetalKernelRegistry.shared.execute(
            kernelName: "depth_attention",
            inputs: [qBuffer, accumulator.stackedBuffer, accumulator.stackedBuffer],
            outputs: [outBuffer],
            params: params,
            device: device,
            commandQueue: commandQueue
        )

        let outShape = Array(accumulator.config.snapshotShape)
        let largeTensor = LargeTensorStorage(
            buffer: outBuffer,
            shape: outShape,
            elementType: accumulator.config.elementType
        )
        return BufferStorage(largeTensor: largeTensor, device: device)
    }
}
