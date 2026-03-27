// LayerNormGPU.swift
// HeterogeneousFusion
//
// Phase 5: Row-sliceable LayerNorm via Metal compute kernels.
// Two kernel variants:
//   Option A: One thread per row (N ≤ 2048) — simple, no shared memory
//   Option B: Threadgroup reduction per row (N > 2048) — simdgroup reductions
//
// Each row is independent: mean → variance → normalize+affine.
// γ and β are shared (broadcast to all units).

import Metal
import QuartzCore

/// Performs LayerNorm on GPU via Metal compute kernels.
/// Supports row-slicing: operates on a contiguous range of rows.
public final class LayerNormGPU: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineNarrow: MTLComputePipelineState   // Option A: N ≤ 2048
    private let pipelineWide: MTLComputePipelineState      // Option B: N > 2048

    /// Threshold for switching from per-row kernel to threadgroup reduction kernel.
    public static let wideKernelThreshold: Int = 2048

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create LayerNorm command queue")
        }
        self.commandQueue = queue

        let source = Self.generateKernels()
        let library = try device.makeLibrary(source: source, options: nil)

        guard let fnNarrow = library.makeFunction(name: "layer_norm_row"),
              let fnWide = library.makeFunction(name: "layer_norm_row_wide") else {
            throw HeterogeneousError.metalSetupFailed("Missing LayerNorm kernel functions")
        }

        self.pipelineNarrow = try device.makeComputePipelineState(function: fnNarrow)
        self.pipelineWide = try device.makeComputePipelineState(function: fnWide)
    }

    /// Expose command queue for shared use.
    public var queue: MTLCommandQueue { commandQueue }

    /// Encode a row-sliced LayerNorm into the given command buffer.
    ///
    /// - Parameters:
    ///   - input: Input buffer [M, N].
    ///   - gamma: Scale parameter buffer [N].
    ///   - beta: Bias parameter buffer [N].
    ///   - output: Output buffer [M, N].
    ///   - inputOffset: Byte offset into input buffer.
    ///   - outputOffset: Byte offset into output buffer.
    ///   - rowCount: Number of rows this unit processes.
    ///   - N: Hidden dimension (columns per row).
    ///   - epsilon: Small constant for numerical stability.
    ///   - commandBuffer: Metal command buffer to encode into.
    public func encode(
        input: MTLBuffer, gamma: MTLBuffer, beta: MTLBuffer, output: MTLBuffer,
        inputOffset: Int, outputOffset: Int,
        rowCount: Int, N: Int, epsilon: Float,
        commandBuffer: MTLCommandBuffer
    ) {
        let useWide = N > Self.wideKernelThreshold
        let pipeline = useWide ? pipelineWide : pipelineNarrow

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(gamma, offset: 0, index: 1)
        encoder.setBuffer(beta, offset: 0, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)

        var n = UInt32(N)
        var eps = epsilon
        encoder.setBytes(&n, length: 4, index: 4)
        encoder.setBytes(&eps, length: 4, index: 5)

        if useWide {
            // Option B: one threadgroup per row, threads cooperate within row
            let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            // Shared memory: (threadgroup_size/32 + 1) * 2 floats for reduction + broadcast
            let simdGroups = (threadsPerGroup + 31) / 32
            let sharedMemSize = (simdGroups + 1) * 2 * MemoryLayout<Float>.size
            encoder.setThreadgroupMemoryLength(sharedMemSize, index: 0)

            encoder.dispatchThreadgroups(
                MTLSize(width: rowCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
        } else {
            // Option A: one thread per row
            let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let groups = (rowCount + threadsPerGroup - 1) / threadsPerGroup
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
        }

        encoder.endEncoding()
    }

    /// Execute synchronously. Returns wall-clock time in seconds.
    public func execute(
        input: MTLBuffer, gamma: MTLBuffer, beta: MTLBuffer, output: MTLBuffer,
        rowCount: Int, N: Int, epsilon: Float = 1e-5
    ) -> Double {
        guard let cb = commandQueue.makeCommandBuffer() else { return 0 }
        encode(
            input: input, gamma: gamma, beta: beta, output: output,
            inputOffset: 0, outputOffset: 0,
            rowCount: rowCount, N: N, epsilon: epsilon,
            commandBuffer: cb
        )
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

        // Option A: One thread per row. Efficient for N ≤ 2048.
        // Each thread scans all N elements 3 times (mean, variance, normalize+affine).
        kernel void layer_norm_row(
            device const float* input   [[buffer(0)]],
            device const float* gamma   [[buffer(1)]],
            device const float* beta    [[buffer(2)]],
            device float*       output  [[buffer(3)]],
            constant uint&      N       [[buffer(4)]],
            constant float&     epsilon [[buffer(5)]],
            uint row [[thread_position_in_grid]])
        {
            device const float* x = input  + row * N;
            device float*       y = output + row * N;

            // Pass 1: mean
            float sum = 0.0f;
            for (uint j = 0; j < N; j++) sum += x[j];
            float mean = sum / float(N);

            // Pass 2: variance
            float var_sum = 0.0f;
            for (uint j = 0; j < N; j++) {
                float d = x[j] - mean;
                var_sum += d * d;
            }
            float inv_std = rsqrt(var_sum / float(N) + epsilon);

            // Pass 3: normalize + affine
            for (uint j = 0; j < N; j++) {
                y[j] = gamma[j] * (x[j] - mean) * inv_std + beta[j];
            }
        }

        // Option B: Threadgroup reduction per row. Required for N > 2048.
        // One threadgroup per row. Threads cooperate via simd_sum + shared memory.
        kernel void layer_norm_row_wide(
            device const float* input     [[buffer(0)]],
            device const float* gamma     [[buffer(1)]],
            device const float* beta      [[buffer(2)]],
            device float*       output    [[buffer(3)]],
            constant uint&      N         [[buffer(4)]],
            constant float&     epsilon   [[buffer(5)]],
            threadgroup float*  scratch   [[threadgroup(0)]],
            uint gid [[threadgroup_position_in_grid]],
            uint lid [[thread_position_in_threadgroup]],
            uint tpg [[threads_per_threadgroup]])
        {
            uint row = gid;
            device const float* x = input  + row * N;
            device float*       y = output + row * N;

            uint simd_lanes = 32;
            uint num_simds = (tpg + simd_lanes - 1) / simd_lanes;
            // scratch layout: [0..num_simds-1] = partial sums, [num_simds] = broadcast value

            // Pass 1: mean via parallel reduction
            float local_sum = 0.0f;
            for (uint j = lid; j < N; j += tpg) local_sum += x[j];
            local_sum = simd_sum(local_sum);
            if (lid % simd_lanes == 0) scratch[lid / simd_lanes] = local_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float mean = 0.0f;
            if (lid == 0) {
                for (uint i = 0; i < num_simds; i++) mean += scratch[i];
                mean /= float(N);
                scratch[num_simds] = mean;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mean = scratch[num_simds];

            // Pass 2: variance via parallel reduction
            float local_var = 0.0f;
            for (uint j = lid; j < N; j += tpg) {
                float d = x[j] - mean;
                local_var += d * d;
            }
            local_var = simd_sum(local_var);
            if (lid % simd_lanes == 0) scratch[lid / simd_lanes] = local_var;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float inv_std = 0.0f;
            if (lid == 0) {
                float var_sum = 0.0f;
                for (uint i = 0; i < num_simds; i++) var_sum += scratch[i];
                inv_std = rsqrt(var_sum / float(N) + epsilon);
                scratch[num_simds] = inv_std;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            inv_std = scratch[num_simds];

            // Pass 3: normalize + affine
            for (uint j = lid; j < N; j += tpg) {
                y[j] = gamma[j] * (x[j] - mean) * inv_std + beta[j];
            }
        }
        """
    }
}
