// SoftmaxKernel.swift
// MetalHLOCore
//
// Custom Metal kernel for numerically stable softmax.

import Metal
import Foundation

/// Custom Metal kernel for numerically stable softmax.
///
/// Implements the standard softmax formula with numerical stability:
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
///
/// This kernel performs softmax over the last dimension in a single pass
/// through the data, using online max/sum computation for efficiency.
public struct SoftmaxKernel: MetalKernel, Sendable {

    public let name = "softmax"
    public let metalFunctionName = "softmax_kernel"

    public init() {}

    public var shaderSource: String {
        """
        #include <metal_stdlib>
        using namespace metal;

        // Kernel parameters structure
        struct SoftmaxParams {
            uint batch_size;    // Number of rows (batch * other dims)
            uint seq_len;       // Length of each row (softmax dimension)
        };

        // Numerically stable softmax kernel
        // Each threadgroup processes one row of the input
        kernel void softmax_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant SoftmaxParams& params [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.seq_len;
            uint seq_len = params.seq_len;

            // Shared memory for reduction
            threadgroup float shared_max[256];
            threadgroup float shared_sum[256];

            // Step 1: Find max value in row (parallel reduction)
            float local_max = -INFINITY;
            for (uint i = local_tid; i < seq_len; i += threads_per_row) {
                local_max = max(local_max, input[row_start + i]);
            }
            shared_max[local_tid] = local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Reduce within threadgroup
            for (uint stride = threads_per_row / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    shared_max[local_tid] = max(shared_max[local_tid], shared_max[local_tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float row_max = shared_max[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step 2: Compute exp(x - max) and sum (parallel reduction)
            float local_sum = 0.0f;
            for (uint i = local_tid; i < seq_len; i += threads_per_row) {
                float exp_val = exp(input[row_start + i] - row_max);
                output[row_start + i] = exp_val;  // Store intermediate result
                local_sum += exp_val;
            }
            shared_sum[local_tid] = local_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Reduce sum within threadgroup
            for (uint stride = threads_per_row / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    shared_sum[local_tid] += shared_sum[local_tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float row_sum = shared_sum[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step 3: Normalize by dividing by sum
            float inv_sum = 1.0f / row_sum;
            for (uint i = local_tid; i < seq_len; i += threads_per_row) {
                output[row_start + i] *= inv_sum;
            }
        }

        // Half-precision version for memory efficiency
        kernel void softmax_kernel_half(
            device const half* input [[buffer(0)]],
            device half* output [[buffer(1)]],
            constant SoftmaxParams& params [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.seq_len;
            uint seq_len = params.seq_len;

            // Use float for computation even with half input
            threadgroup float shared_max[256];
            threadgroup float shared_sum[256];

            // Step 1: Find max
            float local_max = -INFINITY;
            for (uint i = local_tid; i < seq_len; i += threads_per_row) {
                local_max = max(local_max, float(input[row_start + i]));
            }
            shared_max[local_tid] = local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    shared_max[local_tid] = max(shared_max[local_tid], shared_max[local_tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float row_max = shared_max[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step 2: Compute exp and sum
            float local_sum = 0.0f;
            for (uint i = local_tid; i < seq_len; i += threads_per_row) {
                float exp_val = exp(float(input[row_start + i]) - row_max);
                output[row_start + i] = half(exp_val);
                local_sum += exp_val;
            }
            shared_sum[local_tid] = local_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    shared_sum[local_tid] += shared_sum[local_tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float row_sum = shared_sum[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step 3: Normalize
            float inv_sum = 1.0f / row_sum;
            for (uint i = local_tid; i < seq_len; i += threads_per_row) {
                output[row_start + i] = half(float(output[row_start + i]) * inv_sum);
            }
        }
        """
    }

    public func encode(
        into encoder: MTLComputeCommandEncoder,
        inputs: [MTLBuffer],
        outputs: [MTLBuffer],
        params: KernelParams,
        pipeline: MTLComputePipelineState
    ) {
        guard let softmaxParams = params as? SoftmaxParams else {
            fatalError("SoftmaxKernel requires SoftmaxParams")
        }

        // GPU-side params structure (matches Metal struct)
        var gpuParams = GPUSoftmaxParams(
            batch_size: UInt32(softmaxParams.batchSize),
            seq_len: UInt32(softmaxParams.seqLen)
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs[0], offset: 0, index: 0)
        encoder.setBuffer(outputs[0], offset: 0, index: 1)
        encoder.setBytes(&gpuParams, length: MemoryLayout<GPUSoftmaxParams>.size, index: 2)

        let (gridSize, threadgroupSize) = calculateThreadgroups(for: params, pipeline: pipeline)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    public func calculateThreadgroups(
        for params: KernelParams,
        pipeline: MTLComputePipelineState
    ) -> (gridSize: MTLSize, threadgroupSize: MTLSize) {
        guard let softmaxParams = params as? SoftmaxParams else {
            fatalError("SoftmaxKernel requires SoftmaxParams")
        }

        // One threadgroup per row, with threads working on the columns
        let maxThreads = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadsPerRow = min(maxThreads, softmaxParams.seqLen)
        // Round up to power of 2 for efficient reduction
        let roundedThreads = 1 << Int(ceil(log2(Double(threadsPerRow))))
        let finalThreads = min(roundedThreads, maxThreads)

        let threadgroupSize = MTLSize(width: finalThreads, height: 1, depth: 1)
        let gridSize = MTLSize(width: 1, height: softmaxParams.batchSize, depth: 1)

        return (gridSize, threadgroupSize)
    }
}

// GPU-side params structure (must match Metal shader)
private struct GPUSoftmaxParams {
    var batch_size: UInt32
    var seq_len: UInt32
}
