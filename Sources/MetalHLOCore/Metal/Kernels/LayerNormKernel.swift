// LayerNormKernel.swift
// MetalHLOCore
//
// Custom Metal kernels for LayerNorm and RMSNorm with SIMD optimizations.

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

/// Custom Metal kernel for Layer Normalization with SIMD optimizations.
///
/// LayerNorm normalizes across the last dimension:
/// y = (x - mean) / sqrt(variance + eps) * gamma + beta
///
/// Optimizations:
/// 1. SIMD group reduction using simd_sum() for fast warp-level reductions
/// 2. Vectorized memory access (float4) for better memory throughput
/// 3. Two-pass algorithm: first pass computes stats, second normalizes
/// 4. Minimized shared memory traffic with warp-level reductions
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

        // Optimized Layer Normalization kernel with SIMD reductions
        // Each threadgroup processes one row, using SIMD groups for fast reductions
        kernel void layer_norm_kernel(
            device const float* input [[buffer(0)]],
            device const float* gamma [[buffer(1)]],
            device const float* beta [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant NormParams& params [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            // Phase 1: Compute sum and sum of squares using vectorized access + SIMD reduction
            float local_sum = 0.0f;
            float local_sq_sum = 0.0f;

            // Vectorized load (float4) when possible
            uint vec_hidden = hidden_size / 4;
            uint remaining = hidden_size % 4;

            device const float4* input_vec = (device const float4*)(input + row_start);
            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 val = input_vec[i];
                local_sum += val.x + val.y + val.z + val.w;
                local_sq_sum += dot(val, val);
            }

            // Handle remaining elements
            uint vec_offset = vec_hidden * 4;
            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                float val = input[row_start + vec_offset + i];
                local_sum += val;
                local_sq_sum += val * val;
            }

            // SIMD group reduction (fast warp-level reduction)
            local_sum = simd_sum(local_sum);
            local_sq_sum = simd_sum(local_sq_sum);

            // Shared memory for cross-SIMD-group reduction
            threadgroup float shared_sum[32];
            threadgroup float shared_sq_sum[32];

            // Only first lane of each SIMD group writes to shared memory
            if (simd_lane == 0) {
                shared_sum[simd_group] = local_sum;
                shared_sq_sum[simd_group] = local_sq_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Final reduction by first SIMD group
            uint num_simd_groups = (threads_per_row + 31) / 32;
            if (simd_group == 0 && simd_lane < num_simd_groups) {
                local_sum = shared_sum[simd_lane];
                local_sq_sum = shared_sq_sum[simd_lane];
            } else if (simd_group == 0) {
                local_sum = 0.0f;
                local_sq_sum = 0.0f;
            }

            if (simd_group == 0) {
                local_sum = simd_sum(local_sum);
                local_sq_sum = simd_sum(local_sq_sum);

                if (simd_lane == 0) {
                    shared_sum[0] = local_sum;
                    shared_sq_sum[0] = local_sq_sum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute mean and inverse standard deviation
            float mean = shared_sum[0] / float(hidden_size);
            float variance = (shared_sq_sum[0] / float(hidden_size)) - (mean * mean);
            float inv_std = rsqrt(variance + params.epsilon);

            // Phase 2: Normalize and apply affine transform with vectorized access
            device float4* output_vec = (device float4*)(output + row_start);
            device const float4* gamma_vec = (device const float4*)gamma;
            device const float4* beta_vec = (device const float4*)beta;

            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 x = input_vec[i];
                float4 g = gamma_vec[i];
                float4 b = beta_vec[i];
                float4 normalized = (x - mean) * inv_std;
                output_vec[i] = normalized * g + b;
            }

            // Handle remaining elements
            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                uint idx = vec_offset + i;
                float x = input[row_start + idx];
                float normalized = (x - mean) * inv_std;
                output[row_start + idx] = normalized * gamma[idx] + beta[idx];
            }
        }

        // LayerNorm without affine transform (gamma=1, beta=0)
        kernel void layer_norm_no_affine_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant NormParams& params [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            float local_sum = 0.0f;
            float local_sq_sum = 0.0f;

            uint vec_hidden = hidden_size / 4;
            uint remaining = hidden_size % 4;

            device const float4* input_vec = (device const float4*)(input + row_start);
            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 val = input_vec[i];
                local_sum += val.x + val.y + val.z + val.w;
                local_sq_sum += dot(val, val);
            }

            uint vec_offset = vec_hidden * 4;
            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                float val = input[row_start + vec_offset + i];
                local_sum += val;
                local_sq_sum += val * val;
            }

            local_sum = simd_sum(local_sum);
            local_sq_sum = simd_sum(local_sq_sum);

            threadgroup float shared_sum[32];
            threadgroup float shared_sq_sum[32];

            if (simd_lane == 0) {
                shared_sum[simd_group] = local_sum;
                shared_sq_sum[simd_group] = local_sq_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint num_simd_groups = (threads_per_row + 31) / 32;
            if (simd_group == 0 && simd_lane < num_simd_groups) {
                local_sum = shared_sum[simd_lane];
                local_sq_sum = shared_sq_sum[simd_lane];
            } else if (simd_group == 0) {
                local_sum = 0.0f;
                local_sq_sum = 0.0f;
            }

            if (simd_group == 0) {
                local_sum = simd_sum(local_sum);
                local_sq_sum = simd_sum(local_sq_sum);

                if (simd_lane == 0) {
                    shared_sum[0] = local_sum;
                    shared_sq_sum[0] = local_sq_sum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float mean = shared_sum[0] / float(hidden_size);
            float variance = (shared_sq_sum[0] / float(hidden_size)) - (mean * mean);
            float inv_std = rsqrt(variance + params.epsilon);

            device float4* output_vec = (device float4*)(output + row_start);
            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 x = input_vec[i];
                output_vec[i] = (x - mean) * inv_std;
            }

            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                uint idx = vec_offset + i;
                float x = input[row_start + idx];
                output[row_start + idx] = (x - mean) * inv_std;
            }
        }

        // Half-precision LayerNorm with SIMD optimizations
        kernel void layer_norm_kernel_half(
            device const half* input [[buffer(0)]],
            device const half* gamma [[buffer(1)]],
            device const half* beta [[buffer(2)]],
            device half* output [[buffer(3)]],
            constant NormParams& params [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            // Use float for computation, half for storage
            float local_sum = 0.0f;
            float local_sq_sum = 0.0f;

            // Vectorized load (half4)
            uint vec_hidden = hidden_size / 4;
            uint remaining = hidden_size % 4;

            device const half4* input_vec = (device const half4*)(input + row_start);
            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 val = float4(input_vec[i]);
                local_sum += val.x + val.y + val.z + val.w;
                local_sq_sum += dot(val, val);
            }

            uint vec_offset = vec_hidden * 4;
            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                float val = float(input[row_start + vec_offset + i]);
                local_sum += val;
                local_sq_sum += val * val;
            }

            local_sum = simd_sum(local_sum);
            local_sq_sum = simd_sum(local_sq_sum);

            threadgroup float shared_sum[32];
            threadgroup float shared_sq_sum[32];

            if (simd_lane == 0) {
                shared_sum[simd_group] = local_sum;
                shared_sq_sum[simd_group] = local_sq_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint num_simd_groups = (threads_per_row + 31) / 32;
            if (simd_group == 0 && simd_lane < num_simd_groups) {
                local_sum = shared_sum[simd_lane];
                local_sq_sum = shared_sq_sum[simd_lane];
            } else if (simd_group == 0) {
                local_sum = 0.0f;
                local_sq_sum = 0.0f;
            }

            if (simd_group == 0) {
                local_sum = simd_sum(local_sum);
                local_sq_sum = simd_sum(local_sq_sum);

                if (simd_lane == 0) {
                    shared_sum[0] = local_sum;
                    shared_sq_sum[0] = local_sq_sum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float mean = shared_sum[0] / float(hidden_size);
            float variance = (shared_sq_sum[0] / float(hidden_size)) - (mean * mean);
            float inv_std = rsqrt(variance + params.epsilon);

            device half4* output_vec = (device half4*)(output + row_start);
            device const half4* gamma_vec = (device const half4*)gamma;
            device const half4* beta_vec = (device const half4*)beta;

            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 x = float4(input_vec[i]);
                float4 g = float4(gamma_vec[i]);
                float4 b = float4(beta_vec[i]);
                float4 normalized = (x - mean) * inv_std;
                output_vec[i] = half4(normalized * g + b);
            }

            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                uint idx = vec_offset + i;
                float x = float(input[row_start + idx]);
                float normalized = (x - mean) * inv_std;
                output[row_start + idx] = half(normalized * float(gamma[idx]) + float(beta[idx]));
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

        // Use power-of-2 thread count, at least 32 (one SIMD group), up to 256
        let maxThreads = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let desiredThreads = min(maxThreads, max(32, normParams.hiddenSize / 4))
        let roundedThreads = 1 << Int(ceil(log2(Double(max(32, desiredThreads)))))
        let finalThreads = min(roundedThreads, maxThreads)

        let threadgroupSize = MTLSize(width: finalThreads, height: 1, depth: 1)
        let gridSize = MTLSize(width: 1, height: normParams.batchSize, depth: 1)

        return (gridSize, threadgroupSize)
    }
}

/// Custom Metal kernel for RMS Normalization with SIMD optimizations.
///
/// RMSNorm normalizes by root mean square without mean centering:
/// y = x / sqrt(mean(x^2) + eps) * gamma
///
/// Optimizations:
/// 1. SIMD group reduction using simd_sum()
/// 2. Vectorized memory access (float4)
/// 3. Single-pass computation (no mean subtraction needed)
/// 4. Simpler than LayerNorm - commonly used in modern LLMs (LLaMA, etc.)
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

        // Optimized RMS Normalization kernel with SIMD reductions
        kernel void rms_norm_kernel(
            device const float* input [[buffer(0)]],
            device const float* gamma [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant NormParams& params [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            // Phase 1: Compute sum of squares with vectorized access
            float local_sq_sum = 0.0f;

            uint vec_hidden = hidden_size / 4;
            uint remaining = hidden_size % 4;

            device const float4* input_vec = (device const float4*)(input + row_start);
            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 val = input_vec[i];
                local_sq_sum += dot(val, val);
            }

            uint vec_offset = vec_hidden * 4;
            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                float val = input[row_start + vec_offset + i];
                local_sq_sum += val * val;
            }

            // SIMD group reduction
            local_sq_sum = simd_sum(local_sq_sum);

            threadgroup float shared_sq_sum[32];

            if (simd_lane == 0) {
                shared_sq_sum[simd_group] = local_sq_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint num_simd_groups = (threads_per_row + 31) / 32;
            if (simd_group == 0 && simd_lane < num_simd_groups) {
                local_sq_sum = shared_sq_sum[simd_lane];
            } else if (simd_group == 0) {
                local_sq_sum = 0.0f;
            }

            if (simd_group == 0) {
                local_sq_sum = simd_sum(local_sq_sum);

                if (simd_lane == 0) {
                    shared_sq_sum[0] = local_sq_sum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute RMS scale factor
            float rms = rsqrt((shared_sq_sum[0] / float(hidden_size)) + params.epsilon);

            // Phase 2: Normalize and scale with vectorized access
            device float4* output_vec = (device float4*)(output + row_start);
            device const float4* gamma_vec = (device const float4*)gamma;

            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 x = input_vec[i];
                float4 g = gamma_vec[i];
                output_vec[i] = x * rms * g;
            }

            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                uint idx = vec_offset + i;
                output[row_start + idx] = input[row_start + idx] * rms * gamma[idx];
            }
        }

        // RMSNorm without gamma (gamma=1)
        kernel void rms_norm_no_gamma_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant NormParams& params [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            float local_sq_sum = 0.0f;

            uint vec_hidden = hidden_size / 4;
            uint remaining = hidden_size % 4;

            device const float4* input_vec = (device const float4*)(input + row_start);
            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 val = input_vec[i];
                local_sq_sum += dot(val, val);
            }

            uint vec_offset = vec_hidden * 4;
            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                float val = input[row_start + vec_offset + i];
                local_sq_sum += val * val;
            }

            local_sq_sum = simd_sum(local_sq_sum);

            threadgroup float shared_sq_sum[32];

            if (simd_lane == 0) {
                shared_sq_sum[simd_group] = local_sq_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint num_simd_groups = (threads_per_row + 31) / 32;
            if (simd_group == 0 && simd_lane < num_simd_groups) {
                local_sq_sum = shared_sq_sum[simd_lane];
            } else if (simd_group == 0) {
                local_sq_sum = 0.0f;
            }

            if (simd_group == 0) {
                local_sq_sum = simd_sum(local_sq_sum);

                if (simd_lane == 0) {
                    shared_sq_sum[0] = local_sq_sum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float rms = rsqrt((shared_sq_sum[0] / float(hidden_size)) + params.epsilon);

            device float4* output_vec = (device float4*)(output + row_start);
            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                output_vec[i] = input_vec[i] * rms;
            }

            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                uint idx = vec_offset + i;
                output[row_start + idx] = input[row_start + idx] * rms;
            }
        }

        // Half-precision RMSNorm with SIMD optimizations
        kernel void rms_norm_kernel_half(
            device const half* input [[buffer(0)]],
            device const half* gamma [[buffer(1)]],
            device half* output [[buffer(2)]],
            constant NormParams& params [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgSize [[threads_per_threadgroup]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            uint row = gid.y;
            uint local_tid = tid.x;
            uint threads_per_row = tgSize.x;

            if (row >= params.batch_size) return;

            uint row_start = row * params.hidden_size;
            uint hidden_size = params.hidden_size;

            float local_sq_sum = 0.0f;

            uint vec_hidden = hidden_size / 4;
            uint remaining = hidden_size % 4;

            device const half4* input_vec = (device const half4*)(input + row_start);
            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 val = float4(input_vec[i]);
                local_sq_sum += dot(val, val);
            }

            uint vec_offset = vec_hidden * 4;
            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                float val = float(input[row_start + vec_offset + i]);
                local_sq_sum += val * val;
            }

            local_sq_sum = simd_sum(local_sq_sum);

            threadgroup float shared_sq_sum[32];

            if (simd_lane == 0) {
                shared_sq_sum[simd_group] = local_sq_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint num_simd_groups = (threads_per_row + 31) / 32;
            if (simd_group == 0 && simd_lane < num_simd_groups) {
                local_sq_sum = shared_sq_sum[simd_lane];
            } else if (simd_group == 0) {
                local_sq_sum = 0.0f;
            }

            if (simd_group == 0) {
                local_sq_sum = simd_sum(local_sq_sum);

                if (simd_lane == 0) {
                    shared_sq_sum[0] = local_sq_sum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float rms = rsqrt((shared_sq_sum[0] / float(hidden_size)) + params.epsilon);

            device half4* output_vec = (device half4*)(output + row_start);
            device const half4* gamma_vec = (device const half4*)gamma;

            for (uint i = local_tid; i < vec_hidden; i += threads_per_row) {
                float4 x = float4(input_vec[i]);
                float4 g = float4(gamma_vec[i]);
                output_vec[i] = half4(x * rms * g);
            }

            for (uint i = local_tid; i < remaining; i += threads_per_row) {
                uint idx = vec_offset + i;
                output[row_start + idx] = half(float(input[row_start + idx]) * rms * float(gamma[idx]));
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

        // Use power-of-2 thread count, at least 32 (one SIMD group), up to 256
        let maxThreads = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let desiredThreads = min(maxThreads, max(32, normParams.hiddenSize / 4))
        let roundedThreads = 1 << Int(ceil(log2(Double(max(32, desiredThreads)))))
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
