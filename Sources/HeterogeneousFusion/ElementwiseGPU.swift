// ElementwiseGPU.swift
// HeterogeneousFusion
//
// Phase 5: Row-sliceable elementwise operations via Metal compute kernels.
// Supports add, mul, relu, gelu (unary/binary dispatch).
// Each thread processes one element — bandwidth-bound, not compute-bound.

import Metal
import QuartzCore

/// Which elementwise operation to execute.
public enum ElementwiseOp: String, Sendable {
    case add
    case mul
    case relu
    case gelu

    /// Whether this op takes two inputs.
    public var isBinary: Bool {
        switch self {
        case .add, .mul: return true
        case .relu, .gelu: return false
        }
    }

    /// Map from HLOOpCode to ElementwiseOp, if applicable.
    public static func from(_ opcode: HLOOpCode) -> ElementwiseOp? {
        switch opcode {
        case .elementwiseAdd: return .add
        case .elementwiseMul: return .mul
        case .elementwiseRelu: return .relu
        case .gelu: return .gelu
        default: return nil
        }
    }
}

/// Performs elementwise ops on GPU via Metal compute kernels.
/// Supports row-slicing: operates on a contiguous region of the input buffer(s).
public final class ElementwiseGPU: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelines: [ElementwiseOp: MTLComputePipelineState]

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create elementwise command queue")
        }
        self.commandQueue = queue

        let source = Self.generateKernels()
        let library = try device.makeLibrary(source: source, options: nil)

        var pipes: [ElementwiseOp: MTLComputePipelineState] = [:]
        for (op, name) in [
            (ElementwiseOp.add, "elementwise_add"),
            (.mul, "elementwise_mul"),
            (.relu, "elementwise_relu"),
            (.gelu, "elementwise_gelu"),
        ] {
            guard let fn = library.makeFunction(name: name) else {
                throw HeterogeneousError.metalSetupFailed("Missing kernel: \(name)")
            }
            pipes[op] = try device.makeComputePipelineState(function: fn)
        }
        self.pipelines = pipes
    }

    /// Expose command queue for shared use.
    public var queue: MTLCommandQueue { commandQueue }

    /// Encode a row-sliced elementwise op into the given command buffer.
    ///
    /// - Parameters:
    ///   - op: Which elementwise operation.
    ///   - A: Input buffer (required for all ops).
    ///   - B: Second input buffer (required for binary ops: add, mul).
    ///   - C: Output buffer.
    ///   - aOffset: Byte offset into A.
    ///   - bOffset: Byte offset into B (ignored for unary ops).
    ///   - cOffset: Byte offset into C.
    ///   - count: Number of elements to process.
    ///   - commandBuffer: Metal command buffer to encode into.
    public func encode(
        op: ElementwiseOp,
        A: MTLBuffer, B: MTLBuffer?, C: MTLBuffer,
        aOffset: Int, bOffset: Int, cOffset: Int,
        count: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let pipeline = pipelines[op],
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(A, offset: aOffset, index: 0)
        if op.isBinary, let b = B {
            encoder.setBuffer(b, offset: bOffset, index: 1)
        }
        encoder.setBuffer(C, offset: cOffset, index: 2)

        var n = UInt32(count)
        encoder.setBytes(&n, length: 4, index: 3)

        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let groups = (count + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Execute synchronously. Returns wall-clock time in seconds.
    public func execute(
        op: ElementwiseOp,
        A: MTLBuffer, B: MTLBuffer?, C: MTLBuffer,
        count: Int
    ) -> Double {
        guard let cb = commandQueue.makeCommandBuffer() else { return 0 }
        encode(op: op, A: A, B: B, C: C,
               aOffset: 0, bOffset: 0, cOffset: 0,
               count: count, commandBuffer: cb)
        let start = CACurrentMediaTime()
        cb.commit()
        cb.waitUntilCompleted()
        return CACurrentMediaTime() - start
    }

    // MARK: - Metal Kernel Source

    private static func generateKernels() -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void elementwise_add(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],
            device float*       C [[buffer(2)]],
            constant uint&      N [[buffer(3)]],
            uint idx [[thread_position_in_grid]])
        {
            if (idx < N) C[idx] = A[idx] + B[idx];
        }

        kernel void elementwise_mul(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],
            device float*       C [[buffer(2)]],
            constant uint&      N [[buffer(3)]],
            uint idx [[thread_position_in_grid]])
        {
            if (idx < N) C[idx] = A[idx] * B[idx];
        }

        kernel void elementwise_relu(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],  // unused
            device float*       C [[buffer(2)]],
            constant uint&      N [[buffer(3)]],
            uint idx [[thread_position_in_grid]])
        {
            if (idx < N) C[idx] = max(A[idx], 0.0f);
        }

        kernel void elementwise_gelu(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],  // unused
            device float*       C [[buffer(2)]],
            constant uint&      N [[buffer(3)]],
            uint idx [[thread_position_in_grid]])
        {
            if (idx < N) {
                float x = A[idx];
                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                C[idx] = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            }
        }
        """
    }
}
