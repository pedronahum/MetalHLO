// FlashAttentionKernel.swift
// MetalHLOCore
//
// Custom Metal kernel for FlashAttention with tiled computation and online softmax.
// For long sequences where standard attention becomes memory-bound.

import Metal
import Foundation

/// Parameters for FlashAttention kernel.
///
/// Supports batched multi-head attention with optional causal masking.
public struct FlashAttentionParams: KernelParams, Sendable {
    /// Batch size
    public let batchSize: Int

    /// Sequence length for queries
    public let seqLenQ: Int

    /// Sequence length for keys/values (can differ from seqLenQ for cross-attention)
    public let seqLenKV: Int

    /// Number of attention heads
    public let numHeads: Int

    /// Dimension of each head
    public let headDim: Int

    /// Scaling factor (typically 1/sqrt(headDim))
    public let scale: Float

    /// Whether to apply causal masking (future positions masked)
    public let isCausal: Bool

    /// Block size for tiling (must be power of 2)
    public let blockSize: Int

    public var totalElements: Int { batchSize * numHeads * seqLenQ * headDim }

    public init(
        batchSize: Int,
        seqLenQ: Int,
        seqLenKV: Int? = nil,
        numHeads: Int,
        headDim: Int,
        scale: Float? = nil,
        isCausal: Bool = false,
        blockSize: Int = 64
    ) {
        self.batchSize = batchSize
        self.seqLenQ = seqLenQ
        self.seqLenKV = seqLenKV ?? seqLenQ
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = scale ?? (1.0 / sqrt(Float(headDim)))
        self.isCausal = isCausal
        self.blockSize = blockSize
    }
}

/// Custom Metal kernel for FlashAttention with tiled computation.
///
/// FlashAttention computes attention in a memory-efficient manner by:
/// 1. Processing attention in tiles (blocks) to minimize memory bandwidth
/// 2. Using online softmax to avoid materializing the full attention matrix
/// 3. Keeping intermediate results in fast threadgroup memory
///
/// Algorithm:
/// - For each query block:
///   - Load Q tile into shared memory
///   - For each K/V block:
///     - Load K, V tiles into shared memory
///     - Compute attention scores S = Q @ K^T * scale
///     - Apply causal mask (if enabled)
///     - Update running max and sum for online softmax
///     - Accumulate weighted V into output
///   - Normalize output by final sum
///
/// Expected impact: O(N) memory vs O(N^2) for standard attention,
/// enabling much longer sequence lengths.
public struct FlashAttentionKernel: MetalKernel, Sendable {

    public let name = "flash_attention"
    public let metalFunctionName = "flash_attention_kernel"

    public init() {}

    public var shaderSource: String {
        """
        #include <metal_stdlib>
        using namespace metal;

        // FlashAttention parameters
        struct FlashAttentionParams {
            uint batch_size;
            uint seq_len_q;
            uint seq_len_kv;
            uint num_heads;
            uint head_dim;
            float scale;
            uint is_causal;
            uint block_size;
        };

        // FlashAttention kernel with online softmax
        // Each thread computes one query position, iterating over K/V in tiles
        kernel void flash_attention_kernel(
            device const float* Q [[buffer(0)]],        // [batch, num_heads, seq_q, head_dim]
            device const float* K [[buffer(1)]],        // [batch, num_heads, seq_kv, head_dim]
            device const float* V [[buffer(2)]],        // [batch, num_heads, seq_kv, head_dim]
            device float* O [[buffer(3)]],              // [batch, num_heads, seq_q, head_dim]
            constant FlashAttentionParams& params [[buffer(4)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            uint batch = gid.z;
            uint head = gid.y;
            uint q_pos = gid.x;

            if (batch >= params.batch_size || head >= params.num_heads || q_pos >= params.seq_len_q) {
                return;
            }

            uint head_dim = params.head_dim;
            float scale = params.scale;
            uint seq_len_kv = params.seq_len_kv;
            uint block_size = params.block_size;

            // Base offsets for this batch/head
            uint head_offset = (batch * params.num_heads + head);
            uint q_base = (head_offset * params.seq_len_q + q_pos) * head_dim;
            uint kv_head_base = head_offset * seq_len_kv * head_dim;

            // Load query vector into registers
            float q_vec[128];  // Support up to 128 head_dim
            for (uint d = 0; d < head_dim && d < 128; d++) {
                q_vec[d] = Q[q_base + d];
            }

            // Online softmax accumulators
            float m_i = -INFINITY;  // Running max
            float l_i = 0.0f;       // Running sum of exp

            // Output accumulator
            float o_acc[128];
            for (uint d = 0; d < head_dim && d < 128; d++) {
                o_acc[d] = 0.0f;
            }

            // Process K/V in tiles for better cache utilization
            uint num_kv_tiles = (seq_len_kv + block_size - 1) / block_size;

            for (uint tile = 0; tile < num_kv_tiles; tile++) {
                uint tile_start = tile * block_size;
                uint tile_end = min(tile_start + block_size, seq_len_kv);

                // Causal masking: skip entire future tiles
                if (params.is_causal && tile_start > q_pos) {
                    break;
                }

                for (uint k_pos = tile_start; k_pos < tile_end; k_pos++) {
                    // Apply causal mask
                    if (params.is_causal && k_pos > q_pos) {
                        break;
                    }

                    // Compute attention score: Q[q_pos] @ K[k_pos]^T * scale
                    uint k_base = kv_head_base + k_pos * head_dim;
                    float score = 0.0f;
                    for (uint d = 0; d < head_dim && d < 128; d++) {
                        score += q_vec[d] * K[k_base + d];
                    }
                    score *= scale;

                    // Online softmax update
                    float m_new = max(m_i, score);
                    float exp_diff_old = exp(m_i - m_new);
                    float exp_diff_new = exp(score - m_new);

                    // Rescale previous accumulator and add new weighted V
                    uint v_base = k_base;
                    for (uint d = 0; d < head_dim && d < 128; d++) {
                        o_acc[d] = o_acc[d] * exp_diff_old + exp_diff_new * V[v_base + d];
                    }

                    l_i = l_i * exp_diff_old + exp_diff_new;
                    m_i = m_new;
                }
            }

            // Final normalization and write output
            float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            for (uint d = 0; d < head_dim && d < 128; d++) {
                O[q_base + d] = o_acc[d] * l_inv;
            }
        }

        // Half-precision version
        kernel void flash_attention_kernel_half(
            device const half* Q [[buffer(0)]],
            device const half* K [[buffer(1)]],
            device const half* V [[buffer(2)]],
            device half* O [[buffer(3)]],
            constant FlashAttentionParams& params [[buffer(4)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            uint batch = gid.z;
            uint head = gid.y;
            uint q_pos = gid.x;

            if (batch >= params.batch_size || head >= params.num_heads || q_pos >= params.seq_len_q) {
                return;
            }

            uint head_dim = params.head_dim;
            float scale = params.scale;
            uint seq_len_kv = params.seq_len_kv;

            uint head_offset = (batch * params.num_heads + head);
            uint q_base = (head_offset * params.seq_len_q + q_pos) * head_dim;
            uint kv_head_base = head_offset * seq_len_kv * head_dim;

            // Use float for accumulation (better precision)
            float m_i = -INFINITY;
            float l_i = 0.0f;
            float o_acc[128];
            for (uint d = 0; d < head_dim && d < 128; d++) {
                o_acc[d] = 0.0f;
            }

            for (uint k_pos = 0; k_pos < seq_len_kv; k_pos++) {
                if (params.is_causal && k_pos > q_pos) {
                    break;
                }

                uint k_base = kv_head_base + k_pos * head_dim;
                float score = 0.0f;
                for (uint d = 0; d < head_dim && d < 128; d++) {
                    score += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                score *= scale;

                float m_new = max(m_i, score);
                float exp_diff_old = exp(m_i - m_new);
                float exp_diff_new = exp(score - m_new);

                for (uint d = 0; d < head_dim && d < 128; d++) {
                    o_acc[d] = o_acc[d] * exp_diff_old + exp_diff_new * float(V[k_base + d]);
                }
                l_i = l_i * exp_diff_old + exp_diff_new;
                m_i = m_new;
            }

            float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            for (uint d = 0; d < head_dim && d < 128; d++) {
                O[q_base + d] = half(o_acc[d] * l_inv);
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
        guard let flashParams = params as? FlashAttentionParams else {
            fatalError("FlashAttentionKernel requires FlashAttentionParams")
        }

        var gpuParams = GPUFlashAttentionParams(
            batch_size: UInt32(flashParams.batchSize),
            seq_len_q: UInt32(flashParams.seqLenQ),
            seq_len_kv: UInt32(flashParams.seqLenKV),
            num_heads: UInt32(flashParams.numHeads),
            head_dim: UInt32(flashParams.headDim),
            scale: flashParams.scale,
            is_causal: flashParams.isCausal ? 1 : 0,
            block_size: UInt32(flashParams.blockSize)
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs[0], offset: 0, index: 0)  // Q
        encoder.setBuffer(inputs[1], offset: 0, index: 1)  // K
        encoder.setBuffer(inputs[2], offset: 0, index: 2)  // V
        encoder.setBuffer(outputs[0], offset: 0, index: 3) // O
        encoder.setBytes(&gpuParams, length: MemoryLayout<GPUFlashAttentionParams>.size, index: 4)

        let (gridSize, threadgroupSize) = calculateThreadgroups(for: params, pipeline: pipeline)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    public func calculateThreadgroups(
        for params: KernelParams,
        pipeline: MTLComputePipelineState
    ) -> (gridSize: MTLSize, threadgroupSize: MTLSize) {
        guard let flashParams = params as? FlashAttentionParams else {
            fatalError("FlashAttentionKernel requires FlashAttentionParams")
        }

        // One thread per query position
        // Grid: (seq_q, num_heads, batch_size)
        let maxThreads = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))

        // Threadgroup size: process multiple query positions per threadgroup
        let threadgroupWidth = min(flashParams.seqLenQ, maxThreads)
        let threadgroupSize = MTLSize(width: threadgroupWidth, height: 1, depth: 1)

        // Grid size: number of threadgroups needed
        let numThreadgroups = (flashParams.seqLenQ + threadgroupWidth - 1) / threadgroupWidth
        let gridSize = MTLSize(
            width: numThreadgroups,
            height: flashParams.numHeads,
            depth: flashParams.batchSize
        )

        return (gridSize, threadgroupSize)
    }

    /// Estimated shared memory size (for reference - kernel uses registers)
    public func sharedMemorySize(for params: FlashAttentionParams) -> Int {
        // This kernel uses registers, not shared memory
        // Return 0 - keeping method for API compatibility
        return 0
    }
}

// GPU-side params structure (must match Metal shader)
private struct GPUFlashAttentionParams {
    var batch_size: UInt32
    var seq_len_q: UInt32
    var seq_len_kv: UInt32
    var num_heads: UInt32
    var head_dim: UInt32
    var scale: Float
    var is_causal: UInt32
    var block_size: UInt32
}

// MARK: - FlashAttention Statistics

/// Statistics about FlashAttention execution.
public struct FlashAttentionStatistics: Sendable {
    /// Batch size
    public let batchSize: Int

    /// Number of attention heads
    public let numHeads: Int

    /// Query sequence length
    public let seqLenQ: Int

    /// Key/Value sequence length
    public let seqLenKV: Int

    /// Head dimension
    public let headDim: Int

    /// Block size used for tiling
    public let blockSize: Int

    /// Whether causal masking was used
    public let isCausal: Bool

    /// Number of Q blocks processed
    public var numQBlocks: Int {
        (seqLenQ + blockSize - 1) / blockSize
    }

    /// Number of KV blocks per Q block (average, considering causal)
    public var avgKVBlocksPerQ: Double {
        let numKVBlocks = (seqLenKV + blockSize - 1) / blockSize
        if isCausal {
            // For causal, average is half the blocks
            return Double(numKVBlocks) / 2.0
        }
        return Double(numKVBlocks)
    }

    /// Total threadgroups launched
    public var totalThreadgroups: Int {
        batchSize * numHeads * numQBlocks
    }

    /// Memory saved compared to standard attention (ratio)
    /// Standard attention materializes full N*N attention matrix
    /// FlashAttention only uses O(N) memory for tiles
    public var memorySavingsRatio: Double {
        let standardMemory = seqLenQ * seqLenKV  // Full attention matrix
        let flashMemory = 3 * blockSize * headDim  // Q, K, V tiles
        return Double(standardMemory) / Double(flashMemory)
    }

    /// Estimated FLOPs for the attention computation
    public var estimatedFLOPs: Int {
        // QK^T: batch * heads * seq_q * seq_kv * head_dim * 2
        // softmax: batch * heads * seq_q * seq_kv * 5 (max, sub, exp, sum, div)
        // AV: batch * heads * seq_q * head_dim * seq_kv * 2
        let qkFlops = batchSize * numHeads * seqLenQ * seqLenKV * headDim * 2
        let softmaxFlops = batchSize * numHeads * seqLenQ * seqLenKV * 5
        let avFlops = batchSize * numHeads * seqLenQ * headDim * seqLenKV * 2
        return qkFlops + softmaxFlops + avFlops
    }

    public init(
        batchSize: Int,
        numHeads: Int,
        seqLenQ: Int,
        seqLenKV: Int,
        headDim: Int,
        blockSize: Int,
        isCausal: Bool
    ) {
        self.batchSize = batchSize
        self.numHeads = numHeads
        self.seqLenQ = seqLenQ
        self.seqLenKV = seqLenKV
        self.headDim = headDim
        self.blockSize = blockSize
        self.isCausal = isCausal
    }
}

extension FlashAttentionParams {
    /// Computes statistics for this attention configuration.
    public func computeStatistics() -> FlashAttentionStatistics {
        FlashAttentionStatistics(
            batchSize: batchSize,
            numHeads: numHeads,
            seqLenQ: seqLenQ,
            seqLenKV: seqLenKV,
            headDim: headDim,
            blockSize: blockSize,
            isCausal: isCausal
        )
    }
}
