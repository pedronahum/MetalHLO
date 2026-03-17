// DepthAttentionKernel.swift
// MetalHLOCore
//
// Optimized Metal kernel for depth-wise attention used in Attention Residuals (AttnRes).
//
// Depth attention has fundamentally different characteristics from sequence attention:
// - seqLenQ is very small (typically 1 — a single learned query vector per block)
// - seqLenKV is very small (4-16 blocks, at most 32)
// - headDim equals the model's hidden dimension (e.g. 768)
// - numHeads = 1 (no multi-head structure)
//
// This means:
// - The Q·K^T score matrix fits in registers (≤32 floats)
// - The bottleneck is the weighted sum over values, not the score computation
// - One threadgroup can handle the entire attention for one batch element
//
// The kernel computes: output = softmax(query @ keys^T * scale) @ values
// where query is [1, D], keys is [L, D], values is [L, D], output is [1, D].
//
// Multi-simdgroup design:
// - Multiple simdgroups (4-8) per batch element split the D dimension
// - Each simdgroup computes a partial dot product via simd_sum
// - Threadgroup memory holds partial sums for cross-simdgroup reduction
// - After reduction, all threads have identical scores → weighted sum is embarrassingly parallel

import Metal
import Foundation

/// Parameters for the depth attention kernel.
public struct DepthAttentionParams: KernelParams, Sendable {
    /// Batch size (number of independent depth-attention instances)
    public let batchSize: Int

    /// Number of depth positions (layers/blocks) to attend over
    public let depthDim: Int

    /// Hidden dimension of each layer representation
    public let hiddenDim: Int

    /// Scaling factor (typically 1/sqrt(hiddenDim))
    public let scale: Float

    public var totalElements: Int { batchSize * hiddenDim }

    public init(
        batchSize: Int,
        depthDim: Int,
        hiddenDim: Int,
        scale: Float? = nil
    ) {
        self.batchSize = batchSize
        self.depthDim = depthDim
        self.hiddenDim = hiddenDim
        self.scale = scale ?? (1.0 / sqrt(Float(hiddenDim)))
    }
}

/// Multi-simdgroup Metal kernel for depth-wise attention (Attention Residuals).
///
/// Uses multiple simdgroups per batch element:
/// - Each simdgroup reduces a slice of D via simd_sum → partial dot product
/// - Partials written to threadgroup memory, then reduced by simdgroup 0
/// - After scores are computed, all threads read them from threadgroup memory
/// - Weighted sum over values is split across all threads (embarrassingly parallel)
///
/// Thread organization:
///   grid:        (1, batchSize, 1) threadgroups
///   threadgroup: (numSimdgroups * 32, 1, 1) threads
///
/// Threadgroup memory layout:
///   [0 .. numSimdgroups * DA_MAX_DEPTH)  : partial dot products per simdgroup per depth
///   [numSimdgroups * DA_MAX_DEPTH .. +DA_MAX_DEPTH) : final scores (after softmax)
public struct DepthAttentionKernel: MetalKernel, Sendable {

    public let name = "depth_attention"
    public let metalFunctionName = "depth_attention_kernel"

    public init() {}

    public var shaderSource: String {
        """
        #include <metal_stdlib>
        using namespace metal;

        struct DepthAttentionParams {
            uint batch_size;
            uint depth_dim;    // L, max 32
            uint hidden_dim;   // D
            float scale;
            uint num_simdgroups;  // S, number of simdgroups per threadgroup
        };

        constant uint DA_MAX_DEPTH = 32;
        constant uint SIMD_WIDTH = 32;

        kernel void depth_attention_kernel(
            device const float* Q  [[buffer(0)]],
            device const float* K  [[buffer(1)]],
            device const float* V  [[buffer(2)]],
            device       float* O  [[buffer(3)]],
            constant DepthAttentionParams& p [[buffer(4)]],
            threadgroup float* shared [[threadgroup(0)]],
            uint3 tgpig  [[threadgroup_position_in_grid]],
            uint  tpitg  [[thread_index_in_threadgroup]],
            uint  sg_idx [[simdgroup_index_in_threadgroup]],
            uint  sg_lane [[thread_index_in_simdgroup]]
        ) {
            const uint batch = tgpig.y;
            if (batch >= p.batch_size) return;

            const uint D = p.hidden_dim;
            const uint L = min(p.depth_dim, DA_MAX_DEPTH);
            const uint S = p.num_simdgroups;
            const uint tid = tpitg;
            const uint num_threads = S * SIMD_WIDTH;

            const uint q_base = batch * D;
            const uint kv_base = batch * L * D;

            // Shared memory layout:
            //   partials: [S * DA_MAX_DEPTH] — partial dot products per simdgroup
            //   scores:   [DA_MAX_DEPTH]     — final softmax scores
            threadgroup float* partials = shared;
            threadgroup float* scores   = shared + S * DA_MAX_DEPTH;

            // ---- Step 1: Compute attention scores Q·K[l]^T ----
            // Each thread accumulates partial dot product over its D-slice.
            // simd_sum reduces within each simdgroup.
            // Simdgroup leaders write partials to shared memory.
            for (uint l = 0; l < L; l++) {
                float partial_dot = 0.0f;
                for (uint d = tid; d < D; d += num_threads) {
                    partial_dot += Q[q_base + d] * K[kv_base + l * D + d];
                }
                partial_dot = simd_sum(partial_dot);

                if (sg_lane == 0) {
                    partials[sg_idx * DA_MAX_DEPTH + l] = partial_dot;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- Step 2: Reduce partials and compute softmax (simdgroup 0) ----
            if (sg_idx == 0 && sg_lane < L) {
                float total = 0.0f;
                for (uint s = 0; s < S; s++) {
                    total += partials[s * DA_MAX_DEPTH + sg_lane];
                }
                scores[sg_lane] = total * p.scale;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Softmax: find max, then exp and sum (single thread since L ≤ 32)
            if (tid == 0) {
                float max_score = -INFINITY;
                for (uint l = 0; l < L; l++) {
                    max_score = max(max_score, scores[l]);
                }
                float sum_exp = 0.0f;
                for (uint l = 0; l < L; l++) {
                    scores[l] = exp(scores[l] - max_score);
                    sum_exp += scores[l];
                }
                float inv_sum = (sum_exp > 1e-10f) ? (1.0f / sum_exp) : 0.0f;
                for (uint l = 0; l < L; l++) {
                    scores[l] *= inv_sum;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- Step 3: Weighted sum over values (embarrassingly parallel) ----
            // Each thread computes its strided slice of the output hidden dimension.
            for (uint d = tid; d < D; d += num_threads) {
                float acc = 0.0f;
                for (uint l = 0; l < L; l++) {
                    acc += scores[l] * V[kv_base + l * D + d];
                }
                O[q_base + d] = acc;
            }
        }

        kernel void depth_attention_kernel_half(
            device const half*  Q  [[buffer(0)]],
            device const half*  K  [[buffer(1)]],
            device const half*  V  [[buffer(2)]],
            device       half*  O  [[buffer(3)]],
            constant DepthAttentionParams& p [[buffer(4)]],
            threadgroup float* shared [[threadgroup(0)]],
            uint3 tgpig  [[threadgroup_position_in_grid]],
            uint  tpitg  [[thread_index_in_threadgroup]],
            uint  sg_idx [[simdgroup_index_in_threadgroup]],
            uint  sg_lane [[thread_index_in_simdgroup]]
        ) {
            const uint batch = tgpig.y;
            if (batch >= p.batch_size) return;

            const uint D = p.hidden_dim;
            const uint L = min(p.depth_dim, DA_MAX_DEPTH);
            const uint S = p.num_simdgroups;
            const uint tid = tpitg;
            const uint num_threads = S * SIMD_WIDTH;

            const uint q_base = batch * D;
            const uint kv_base = batch * L * D;

            threadgroup float* partials = shared;
            threadgroup float* scores   = shared + S * DA_MAX_DEPTH;

            for (uint l = 0; l < L; l++) {
                float partial_dot = 0.0f;
                for (uint d = tid; d < D; d += num_threads) {
                    partial_dot += float(Q[q_base + d]) * float(K[kv_base + l * D + d]);
                }
                partial_dot = simd_sum(partial_dot);
                if (sg_lane == 0) {
                    partials[sg_idx * DA_MAX_DEPTH + l] = partial_dot;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg_idx == 0 && sg_lane < L) {
                float total = 0.0f;
                for (uint s = 0; s < S; s++) total += partials[s * DA_MAX_DEPTH + sg_lane];
                scores[sg_lane] = total * p.scale;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                float max_score = -INFINITY;
                for (uint l = 0; l < L; l++) max_score = max(max_score, scores[l]);
                float sum_exp = 0.0f;
                for (uint l = 0; l < L; l++) {
                    scores[l] = exp(scores[l] - max_score);
                    sum_exp += scores[l];
                }
                float inv_sum = (sum_exp > 1e-10f) ? (1.0f / sum_exp) : 0.0f;
                for (uint l = 0; l < L; l++) scores[l] *= inv_sum;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint d = tid; d < D; d += num_threads) {
                float acc = 0.0f;
                for (uint l = 0; l < L; l++) {
                    acc += scores[l] * float(V[kv_base + l * D + d]);
                }
                O[q_base + d] = half(acc);
            }
        }
        """
    }

    /// Choose number of simdgroups based on hidden dimension.
    /// More simdgroups = more threads splitting the D dimension.
    public static func optimalSimdgroups(hiddenDim: Int) -> Int {
        // D=768: 768/32=24 elements per thread with 1 SG, 6 with 4 SGs
        // D=4096: 4096/32=128 elements per thread with 1 SG, 16 with 8 SGs
        if hiddenDim >= 2048 { return 8 }
        if hiddenDim >= 512  { return 4 }
        return 1
    }

    public func encode(
        into encoder: MTLComputeCommandEncoder,
        inputs: [MTLBuffer],
        outputs: [MTLBuffer],
        params: KernelParams,
        pipeline: MTLComputePipelineState
    ) {
        guard let daParams = params as? DepthAttentionParams else {
            fatalError("DepthAttentionKernel requires DepthAttentionParams")
        }

        let numSG = DepthAttentionKernel.optimalSimdgroups(hiddenDim: daParams.hiddenDim)

        var gpuParams = GPUDepthAttentionParams(
            batch_size: UInt32(daParams.batchSize),
            depth_dim: UInt32(daParams.depthDim),
            hidden_dim: UInt32(daParams.hiddenDim),
            scale: daParams.scale,
            num_simdgroups: UInt32(numSG)
        )

        // Threadgroup memory: partials[S * 32] + scores[32]
        let sharedMemSize = (numSG * 32 + 32) * MemoryLayout<Float>.size

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs[0], offset: 0, index: 0)  // Q
        encoder.setBuffer(inputs[1], offset: 0, index: 1)  // K
        encoder.setBuffer(inputs[2], offset: 0, index: 2)  // V
        encoder.setBuffer(outputs[0], offset: 0, index: 3) // O
        encoder.setBytes(&gpuParams, length: MemoryLayout<GPUDepthAttentionParams>.size, index: 4)
        encoder.setThreadgroupMemoryLength(sharedMemSize, index: 0)

        let (gridSize, threadgroupSize) = calculateThreadgroups(for: params, pipeline: pipeline)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    public func calculateThreadgroups(
        for params: KernelParams,
        pipeline: MTLComputePipelineState
    ) -> (gridSize: MTLSize, threadgroupSize: MTLSize) {
        guard let daParams = params as? DepthAttentionParams else {
            fatalError("DepthAttentionKernel requires DepthAttentionParams")
        }

        let simdWidth = pipeline.threadExecutionWidth  // 32 on Apple Silicon
        let numSG = DepthAttentionKernel.optimalSimdgroups(hiddenDim: daParams.hiddenDim)
        let threadsPerGroup = numSG * simdWidth

        let gridSize = MTLSize(width: 1, height: daParams.batchSize, depth: 1)
        let threadgroupSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        return (gridSize, threadgroupSize)
    }
}

// GPU-side params structure (must match Metal shader layout)
private struct GPUDepthAttentionParams {
    var batch_size: UInt32
    var depth_dim: UInt32
    var hidden_dim: UInt32
    var scale: Float
    var num_simdgroups: UInt32
}
