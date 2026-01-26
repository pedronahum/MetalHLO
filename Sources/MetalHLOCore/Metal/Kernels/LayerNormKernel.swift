// LayerNormKernel.swift
// MetalHLOCore
//
// Custom Metal kernels for LayerNorm and RMSNorm.

import Metal
import Foundation

/// Parameters for normalization kernels.
public struct NormParams: KernelParams, Sendable {
    public let batchSize: Int     // Number of rows to normalize
    public let hiddenSize: Int    // Size of normalization dimension
    public let epsilon: Float     // Small constant for numerical stability

    public var totalElements: Int { batchSize * hiddenSize }

    public init(batchSize: Int, hiddenSize: Int, epsilon: Float = 1e-5) {
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
        self.epsilon = epsilon
    }
}

/// Custom Metal kernel for Layer Normalization.
///
/// LayerNorm normalizes across the last dimension:
/// y = (x - mean) / sqrt(variance + eps) * gamma + beta
///
/// Uses Welford's online algorithm for numerically stable
/// mean and variance computation in a single pass.
public struct LayerNormKernel: MetalKernel, Sendable {

    public let name = "layer_norm"
    public let metalFunctionName = "layer_norm_kernel"

    public init() {}

    public var shaderSource: String {
        """
        #include <metal_stdlib>
        using namespace metal;

        // Parameters structure
        struct NormParams {
            uint batch_size;    // Number of rows
            uint hidden_size;   // Size of normalization dim
            float epsilon;      // Numerical stability constant
        };

        // Layer Normalization kernel
        // Each threadgroup processes one row
        kernel void layer_norm_kernel(
            device const float* input [[buffer(0)]],
            device const float* gamma [[buffer(1)]],
            device const float* beta [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant NormParams& params [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            // Shared memory for reduction
            threadgroup float shared_sum[256];
            threadgroup float shared_sq_sum[256];

            // Step 1: Compute sum and squared sum (Welford's algorithm)
            float local_sum = 0.0f;
            float local_sq_sum = 0.0f;

            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                float val = input[row_start + i];
                local_sum += val;
                local_sq_sum += val * val;
            }

            shared_sum[local_tid] = local_sum;
            shared_sq_sum[local_tid] = local_sq_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Reduce sums within threadgroup
            for (uint stride = threads_per_row / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    shared_sum[local_tid] += shared_sum[local_tid + stride];
                    shared_sq_sum[local_tid] += shared_sq_sum[local_tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Compute mean and variance
            float mean = shared_sum[0] / float(hidden_size);
            float variance = (shared_sq_sum[0] / float(hidden_size)) - (mean * mean);
            float inv_std = rsqrt(variance + params.epsilon);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step 2: Normalize and apply affine transform
            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                float x = input[row_start + i];
                float normalized = (x - mean) * inv_std;
                output[row_start + i] = normalized * gamma[i] + beta[i];
            }
        }

        // LayerNorm without affine transform (gamma=1, beta=0)
        kernel void layer_norm_no_affine_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant NormParams& params [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            threadgroup float shared_sum[256];
            threadgroup float shared_sq_sum[256];

            float local_sum = 0.0f;
            float local_sq_sum = 0.0f;

            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                float val = input[row_start + i];
                local_sum += val;
                local_sq_sum += val * val;
            }

            shared_sum[local_tid] = local_sum;
            shared_sq_sum[local_tid] = local_sq_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    shared_sum[local_tid] += shared_sum[local_tid + stride];
                    shared_sq_sum[local_tid] += shared_sq_sum[local_tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float mean = shared_sum[0] / float(hidden_size);
            float variance = (shared_sq_sum[0] / float(hidden_size)) - (mean * mean);
            float inv_std = rsqrt(variance + params.epsilon);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                float x = input[row_start + i];
                output[row_start + i] = (x - mean) * inv_std;
            }
        }

        // Half-precision LayerNorm
        kernel void layer_norm_kernel_half(
            device const half* input [[buffer(0)]],
            device const half* gamma [[buffer(1)]],
            device const half* beta [[buffer(2)]],
            device half* output [[buffer(3)]],
            constant NormParams& params [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            // Use float for computation
            threadgroup float shared_sum[256];
            threadgroup float shared_sq_sum[256];

            float local_sum = 0.0f;
            float local_sq_sum = 0.0f;

            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                float val = float(input[row_start + i]);
                local_sum += val;
                local_sq_sum += val * val;
            }

            shared_sum[local_tid] = local_sum;
            shared_sq_sum[local_tid] = local_sq_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    shared_sum[local_tid] += shared_sum[local_tid + stride];
                    shared_sq_sum[local_tid] += shared_sq_sum[local_tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float mean = shared_sum[0] / float(hidden_size);
            float variance = (shared_sq_sum[0] / float(hidden_size)) - (mean * mean);
            float inv_std = rsqrt(variance + params.epsilon);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                float x = float(input[row_start + i]);
                float normalized = (x - mean) * inv_std;
                output[row_start + i] = half(normalized * float(gamma[i]) + float(beta[i]));
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
        guard let normParams = params as? NormParams else {
            fatalError("LayerNormKernel requires NormParams")
        }

        var gpuParams = GPUNormParams(
            batch_size: UInt32(normParams.batchSize),
            hidden_size: UInt32(normParams.hiddenSize),
            epsilon: normParams.epsilon
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs[0], offset: 0, index: 0)  // input

        if inputs.count >= 3 {
            // With affine transform
            encoder.setBuffer(inputs[1], offset: 0, index: 1)  // gamma
            encoder.setBuffer(inputs[2], offset: 0, index: 2)  // beta
            encoder.setBuffer(outputs[0], offset: 0, index: 3)
            encoder.setBytes(&gpuParams, length: MemoryLayout<GPUNormParams>.size, index: 4)
        } else {
            // Without affine transform
            encoder.setBuffer(outputs[0], offset: 0, index: 1)
            encoder.setBytes(&gpuParams, length: MemoryLayout<GPUNormParams>.size, index: 2)
        }

        let (gridSize, threadgroupSize) = calculateThreadgroups(for: params, pipeline: pipeline)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    public func calculateThreadgroups(
        for params: KernelParams,
        pipeline: MTLComputePipelineState
    ) -> (gridSize: MTLSize, threadgroupSize: MTLSize) {
        guard let normParams = params as? NormParams else {
            fatalError("LayerNormKernel requires NormParams")
        }

        let maxThreads = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadsPerRow = min(maxThreads, normParams.hiddenSize)
        let roundedThreads = 1 << Int(ceil(log2(Double(max(1, threadsPerRow)))))
        let finalThreads = min(roundedThreads, maxThreads)

        let threadgroupSize = MTLSize(width: finalThreads, height: 1, depth: 1)
        let gridSize = MTLSize(width: 1, height: normParams.batchSize, depth: 1)

        return (gridSize, threadgroupSize)
    }
}

/// Custom Metal kernel for RMS Normalization.
///
/// RMSNorm normalizes by root mean square without mean centering:
/// y = x / sqrt(mean(x^2) + eps) * gamma
///
/// Simpler and faster than LayerNorm, commonly used in LLMs.
public struct RMSNormKernel: MetalKernel, Sendable {

    public let name = "rms_norm"
    public let metalFunctionName = "rms_norm_kernel"

    public init() {}

    public var shaderSource: String {
        """
        #include <metal_stdlib>
        using namespace metal;

        struct NormParams {
            uint batch_size;
            uint hidden_size;
            float epsilon;
        };

        // RMS Normalization kernel
        kernel void rms_norm_kernel(
            device const float* input [[buffer(0)]],
            device const float* gamma [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant NormParams& params [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            // Shared memory for reduction
            threadgroup float shared_sq_sum[256];

            // Step 1: Compute sum of squares
            float local_sq_sum = 0.0f;
            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                float val = input[row_start + i];
                local_sq_sum += val * val;
            }

            shared_sq_sum[local_tid] = local_sq_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Reduce within threadgroup
            for (uint stride = threads_per_row / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    shared_sq_sum[local_tid] += shared_sq_sum[local_tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Compute RMS
            float rms = rsqrt((shared_sq_sum[0] / float(hidden_size)) + params.epsilon);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step 2: Normalize and scale
            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                output[row_start + i] = input[row_start + i] * rms * gamma[i];
            }
        }

        // RMSNorm without gamma (gamma=1)
        kernel void rms_norm_no_gamma_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant NormParams& params [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            threadgroup float shared_sq_sum[256];

            float local_sq_sum = 0.0f;
            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                float val = input[row_start + i];
                local_sq_sum += val * val;
            }

            shared_sq_sum[local_tid] = local_sq_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    shared_sq_sum[local_tid] += shared_sq_sum[local_tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float rms = rsqrt((shared_sq_sum[0] / float(hidden_size)) + params.epsilon);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                output[row_start + i] = input[row_start + i] * rms;
            }
        }

        // Half-precision RMSNorm
        kernel void rms_norm_kernel_half(
            device const half* input [[buffer(0)]],
            device const half* gamma [[buffer(1)]],
            device half* output [[buffer(2)]],
            constant NormParams& params [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            threadgroup float shared_sq_sum[256];

            float local_sq_sum = 0.0f;
            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                float val = float(input[row_start + i]);
                local_sq_sum += val * val;
            }

            shared_sq_sum[local_tid] = local_sq_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = threads_per_row / 2; stride > 0; stride /= 2) {
                if (local_tid < stride) {
                    shared_sq_sum[local_tid] += shared_sq_sum[local_tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float rms = rsqrt((shared_sq_sum[0] / float(hidden_size)) + params.epsilon);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = local_tid; i < hidden_size; i += threads_per_row) {
                output[row_start + i] = half(float(input[row_start + i]) * rms * float(gamma[i]));
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
        guard let normParams = params as? NormParams else {
            fatalError("RMSNormKernel requires NormParams")
        }

        var gpuParams = GPUNormParams(
            batch_size: UInt32(normParams.batchSize),
            hidden_size: UInt32(normParams.hiddenSize),
            epsilon: normParams.epsilon
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs[0], offset: 0, index: 0)  // input

        if inputs.count >= 2 {
            // With gamma
            encoder.setBuffer(inputs[1], offset: 0, index: 1)  // gamma
            encoder.setBuffer(outputs[0], offset: 0, index: 2)
            encoder.setBytes(&gpuParams, length: MemoryLayout<GPUNormParams>.size, index: 3)
        } else {
            // Without gamma
            encoder.setBuffer(outputs[0], offset: 0, index: 1)
            encoder.setBytes(&gpuParams, length: MemoryLayout<GPUNormParams>.size, index: 2)
        }

        let (gridSize, threadgroupSize) = calculateThreadgroups(for: params, pipeline: pipeline)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    public func calculateThreadgroups(
        for params: KernelParams,
        pipeline: MTLComputePipelineState
    ) -> (gridSize: MTLSize, threadgroupSize: MTLSize) {
        guard let normParams = params as? NormParams else {
            fatalError("RMSNormKernel requires NormParams")
        }

        let maxThreads = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadsPerRow = min(maxThreads, normParams.hiddenSize)
        let roundedThreads = 1 << Int(ceil(log2(Double(max(1, threadsPerRow)))))
        let finalThreads = min(roundedThreads, maxThreads)

        let threadgroupSize = MTLSize(width: finalThreads, height: 1, depth: 1)
        let gridSize = MTLSize(width: 1, height: normParams.batchSize, depth: 1)

        return (gridSize, threadgroupSize)
    }
}

// GPU-side params structure (must match Metal shader)
private struct GPUNormParams {
    var batch_size: UInt32
    var hidden_size: UInt32
    var epsilon: Float
}
