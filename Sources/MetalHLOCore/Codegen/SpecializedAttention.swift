// SpecializedAttention.swift
// MetalHLOCore
//
// Shape-specialized attention kernel generator.

import Foundation

/// Configuration for specialized attention computation.
public struct AttentionSpecialization: Hashable, Sendable {
    /// Batch size.
    public let batchSize: Int

    /// Number of attention heads.
    public let numHeads: Int

    /// Number of KV heads (for grouped-query attention).
    public let numKVHeads: Int

    /// Query sequence length.
    public let seqLenQ: Int

    /// Key/Value sequence length.
    public let seqLenKV: Int

    /// Dimension of each head.
    public let headDim: Int

    /// Whether to apply causal masking.
    public let isCausal: Bool

    /// Data type.
    public let dtype: MetalType

    /// Scaling factor (defaults to 1/sqrt(headDim)).
    public let scale: Float?

    /// Whether this is for prefill (long prompt) or decode (single token).
    public let isPrefill: Bool

    public init(
        batchSize: Int,
        numHeads: Int,
        numKVHeads: Int? = nil,
        seqLenQ: Int,
        seqLenKV: Int? = nil,
        headDim: Int,
        isCausal: Bool = true,
        dtype: MetalType = .float,
        scale: Float? = nil,
        isPrefill: Bool = true
    ) {
        self.batchSize = batchSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads ?? numHeads
        self.seqLenQ = seqLenQ
        self.seqLenKV = seqLenKV ?? seqLenQ
        self.headDim = headDim
        self.isCausal = isCausal
        self.dtype = dtype
        self.scale = scale
        self.isPrefill = isPrefill
    }

    /// Unique identifier for this specialization.
    public var identifier: String {
        var parts = ["attention"]
        parts.append("b\(batchSize)")
        parts.append("h\(numHeads)")
        if numKVHeads != numHeads {
            parts.append("kv\(numKVHeads)")
        }
        parts.append("sq\(seqLenQ)")
        if seqLenKV != seqLenQ {
            parts.append("skv\(seqLenKV)")
        }
        parts.append("d\(headDim)")
        if isCausal { parts.append("causal") }
        parts.append(dtype.rawValue)
        if !isPrefill { parts.append("decode") }
        return parts.joined(separator: "_")
    }

    /// Computed scale factor.
    public var effectiveScale: Float {
        scale ?? (1.0 / sqrt(Float(headDim)))
    }

    /// Number of heads per KV head (for GQA).
    public var headsPerKVHead: Int {
        numHeads / numKVHeads
    }

    /// Whether this uses grouped-query attention.
    public var isGQA: Bool {
        numKVHeads != numHeads
    }
}

/// Generates shape-specialized attention kernels.
///
/// This generator creates optimized attention kernels with:
/// - Compile-time constant dimensions
/// - Optimal block sizes for the sequence length
/// - Causal masking optimization (skip unnecessary computation)
/// - Grouped-query attention (GQA) support
/// - Online softmax to avoid materializing N*N attention matrix
public struct SpecializedAttentionGenerator: Sendable {

    /// Tile calculator for optimal configurations.
    private let tileCalculator: TileCalculator

    public init(tileCalculator: TileCalculator = TileCalculator()) {
        self.tileCalculator = tileCalculator
    }

    /// Generates a specialized kernel for the given configuration.
    public func generate(spec: AttentionSpecialization) -> SpecializedKernel {
        let tileConfig = tileCalculator.calculateAttentionTiles(
            batchSize: spec.batchSize,
            seqLen: spec.seqLenQ,
            numHeads: spec.numHeads,
            headDim: spec.headDim,
            isCausal: spec.isCausal,
            elementType: spec.dtype == .half ? .float16 : .float32
        )

        let source: String
        if !spec.isPrefill {
            // Decode mode: single query token attending to full KV cache
            source = generateDecodeKernel(spec: spec, tileConfig: tileConfig)
        } else if spec.isGQA {
            // Grouped-query attention
            source = generateGQAKernel(spec: spec, tileConfig: tileConfig)
        } else if spec.seqLenQ > 512 {
            // Long sequence: use memory-efficient FlashAttention-style
            source = generateFlashStyleKernel(spec: spec, tileConfig: tileConfig)
        } else {
            // Short sequence: straightforward implementation
            source = generateStandardKernel(spec: spec, tileConfig: tileConfig)
        }

        let estimate = computePerformanceEstimate(spec: spec)

        return SpecializedKernel(
            source: source,
            functionName: spec.identifier,
            tileConfig: TileConfigWrapper(attention: tileConfig),
            estimatedMetrics: estimate
        )
    }

    // MARK: - Kernel Generators

    /// Standard attention for short sequences.
    private func generateStandardKernel(
        spec: AttentionSpecialization,
        tileConfig: AttentionTileConfig
    ) -> String {
        let dtype = spec.dtype.rawValue
        let scale = spec.effectiveScale

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Standard attention: batch=\(spec.batchSize), heads=\(spec.numHeads), seq=\(spec.seqLenQ), dim=\(spec.headDim)
        kernel void \(spec.identifier)(
            device const \(dtype)* Q [[buffer(0)]],
            device const \(dtype)* K [[buffer(1)]],
            device const \(dtype)* V [[buffer(2)]],
            device \(dtype)* O [[buffer(3)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            constexpr uint BATCH = \(spec.batchSize);
            constexpr uint HEADS = \(spec.numHeads);
            constexpr uint SEQ_Q = \(spec.seqLenQ);
            constexpr uint SEQ_KV = \(spec.seqLenKV);
            constexpr uint DIM = \(spec.headDim);
            constexpr float SCALE = \(scale)f;

            uint batch = gid.z;
            uint head = gid.y;
            uint q_pos = gid.x;

            if (batch >= BATCH || head >= HEADS || q_pos >= SEQ_Q) return;

            // Base offsets
            uint q_base = ((batch * HEADS + head) * SEQ_Q + q_pos) * DIM;
            uint kv_head_base = (batch * HEADS + head) * SEQ_KV;

            // Online softmax accumulators
            float m_i = -INFINITY;
            float l_i = 0.0f;

            // Pass 1: Compute softmax normalization
            uint kv_limit = \(spec.isCausal ? "min(q_pos + 1, SEQ_KV)" : "SEQ_KV");

            for (uint k_pos = 0; k_pos < kv_limit; k_pos++) {
                uint k_base = (kv_head_base + k_pos) * DIM;
                float score = 0.0f;

                // Dot product Q[q_pos] · K[k_pos]
                for (uint d = 0; d < DIM; d++) {
                    score += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                score *= SCALE;

                // Online softmax update
                float m_new = max(m_i, score);
                float exp_old = exp(m_i - m_new);
                float exp_new = exp(score - m_new);
                l_i = l_i * exp_old + exp_new;
                m_i = m_new;
            }

            float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

            // Initialize output
            uint o_base = q_base;
            for (uint d = 0; d < DIM; d++) {
                O[o_base + d] = \(dtype)(0);
            }

            // Pass 2: Weighted sum of values
            for (uint k_pos = 0; k_pos < kv_limit; k_pos++) {
                uint k_base = (kv_head_base + k_pos) * DIM;

                // Recompute attention score
                float score = 0.0f;
                for (uint d = 0; d < DIM; d++) {
                    score += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                score *= SCALE;

                float weight = exp(score - m_i) * l_inv;

                // Accumulate weighted V
                for (uint d = 0; d < DIM; d++) {
                    O[o_base + d] = \(dtype)(float(O[o_base + d]) + weight * float(V[k_base + d]));
                }
            }
        }
        """
    }

    /// FlashAttention-style kernel for long sequences.
    private func generateFlashStyleKernel(
        spec: AttentionSpecialization,
        tileConfig: AttentionTileConfig
    ) -> String {
        let dtype = spec.dtype.rawValue
        let scale = spec.effectiveScale
        let blockQ = tileConfig.blockQ
        let blockKV = tileConfig.blockKV

        return """
        #include <metal_stdlib>
        using namespace metal;

        // FlashAttention-style: batch=\(spec.batchSize), heads=\(spec.numHeads), seq=\(spec.seqLenQ), dim=\(spec.headDim)
        // Block sizes: Q=\(blockQ), KV=\(blockKV)
        kernel void \(spec.identifier)(
            device const \(dtype)* Q [[buffer(0)]],
            device const \(dtype)* K [[buffer(1)]],
            device const \(dtype)* V [[buffer(2)]],
            device \(dtype)* O [[buffer(3)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            constexpr uint BATCH = \(spec.batchSize);
            constexpr uint HEADS = \(spec.numHeads);
            constexpr uint SEQ_Q = \(spec.seqLenQ);
            constexpr uint SEQ_KV = \(spec.seqLenKV);
            constexpr uint DIM = \(spec.headDim);
            constexpr float SCALE = \(scale)f;
            constexpr bool IS_CAUSAL = \(spec.isCausal);

            uint batch = gid.z;
            uint head = gid.y;
            uint q_pos = gid.x;

            if (batch >= BATCH || head >= HEADS || q_pos >= SEQ_Q) return;

            // Base offsets
            uint q_base = ((batch * HEADS + head) * SEQ_Q + q_pos) * DIM;
            uint kv_head_base = (batch * HEADS + head) * SEQ_KV;

            // Online softmax state
            float m_i = -INFINITY;
            float l_i = 0.0f;

            // Output accumulator (in registers)
            float o_acc[DIM];
            for (uint d = 0; d < DIM; d++) {
                o_acc[d] = 0.0f;
            }

            // KV limit for causal masking
            uint kv_limit = IS_CAUSAL ? min(q_pos + 1, SEQ_KV) : SEQ_KV;

            // Process KV in blocks
            for (uint k_start = 0; k_start < kv_limit; k_start++) {
                uint k_base = (kv_head_base + k_start) * DIM;

                // Compute attention score
                float score = 0.0f;
                for (uint d = 0; d < DIM; d++) {
                    score += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                score *= SCALE;

                // Online softmax with output rescaling
                float m_new = max(m_i, score);
                float exp_old = exp(m_i - m_new);
                float exp_new = exp(score - m_new);

                // Rescale existing accumulator
                float scale_factor = exp_old;
                for (uint d = 0; d < DIM; d++) {
                    o_acc[d] *= scale_factor;
                }

                // Add new contribution
                float weight = exp_new;
                for (uint d = 0; d < DIM; d++) {
                    o_acc[d] += weight * float(V[k_base + d]);
                }

                // Update softmax state
                l_i = l_i * exp_old + exp_new;
                m_i = m_new;
            }

            // Normalize output
            float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            uint o_base = q_base;
            for (uint d = 0; d < DIM; d++) {
                O[o_base + d] = \(dtype)(o_acc[d] * l_inv);
            }
        }
        """
    }

    /// Decode kernel for single-token generation.
    private func generateDecodeKernel(
        spec: AttentionSpecialization,
        tileConfig: AttentionTileConfig
    ) -> String {
        let dtype = spec.dtype.rawValue
        let scale = spec.effectiveScale

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Decode attention: batch=\(spec.batchSize), heads=\(spec.numHeads), kv_len=\(spec.seqLenKV), dim=\(spec.headDim)
        // Single query token attending to full KV cache
        kernel void \(spec.identifier)(
            device const \(dtype)* Q [[buffer(0)]],
            device const \(dtype)* K [[buffer(1)]],
            device const \(dtype)* V [[buffer(2)]],
            device \(dtype)* O [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            constexpr uint BATCH = \(spec.batchSize);
            constexpr uint HEADS = \(spec.numHeads);
            constexpr uint SEQ_KV = \(spec.seqLenKV);
            constexpr uint DIM = \(spec.headDim);
            constexpr float SCALE = \(scale)f;

            uint batch = gid.y;
            uint head = gid.x;

            if (batch >= BATCH || head >= HEADS) return;

            // Q has shape [batch, heads, 1, dim]
            uint q_base = (batch * HEADS + head) * DIM;
            // KV has shape [batch, heads, seq_kv, dim]
            uint kv_head_base = (batch * HEADS + head) * SEQ_KV;

            // Online softmax state
            float m_i = -INFINITY;
            float l_i = 0.0f;

            // Output accumulator
            float o_acc[DIM];
            for (uint d = 0; d < DIM; d++) {
                o_acc[d] = 0.0f;
            }

            // Attend to all KV positions
            for (uint k_pos = 0; k_pos < SEQ_KV; k_pos++) {
                uint k_base = (kv_head_base + k_pos) * DIM;

                // Score computation
                float score = 0.0f;
                for (uint d = 0; d < DIM; d++) {
                    score += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                score *= SCALE;

                // Online softmax update
                float m_new = max(m_i, score);
                float exp_old = exp(m_i - m_new);
                float exp_new = exp(score - m_new);

                // Rescale and accumulate
                for (uint d = 0; d < DIM; d++) {
                    o_acc[d] = o_acc[d] * exp_old + exp_new * float(V[k_base + d]);
                }

                l_i = l_i * exp_old + exp_new;
                m_i = m_new;
            }

            // Normalize and write output
            float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            uint o_base = q_base;
            for (uint d = 0; d < DIM; d++) {
                O[o_base + d] = \(dtype)(o_acc[d] * l_inv);
            }
        }
        """
    }

    /// Grouped-query attention kernel.
    private func generateGQAKernel(
        spec: AttentionSpecialization,
        tileConfig: AttentionTileConfig
    ) -> String {
        let dtype = spec.dtype.rawValue
        let scale = spec.effectiveScale

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Grouped-Query Attention: batch=\(spec.batchSize), q_heads=\(spec.numHeads), kv_heads=\(spec.numKVHeads)
        // seq=\(spec.seqLenQ), dim=\(spec.headDim), heads_per_kv=\(spec.headsPerKVHead)
        kernel void \(spec.identifier)(
            device const \(dtype)* Q [[buffer(0)]],
            device const \(dtype)* K [[buffer(1)]],
            device const \(dtype)* V [[buffer(2)]],
            device \(dtype)* O [[buffer(3)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            constexpr uint BATCH = \(spec.batchSize);
            constexpr uint Q_HEADS = \(spec.numHeads);
            constexpr uint KV_HEADS = \(spec.numKVHeads);
            constexpr uint HEADS_PER_KV = \(spec.headsPerKVHead);
            constexpr uint SEQ_Q = \(spec.seqLenQ);
            constexpr uint SEQ_KV = \(spec.seqLenKV);
            constexpr uint DIM = \(spec.headDim);
            constexpr float SCALE = \(scale)f;
            constexpr bool IS_CAUSAL = \(spec.isCausal);

            uint batch = gid.z;
            uint q_head = gid.y;
            uint q_pos = gid.x;

            if (batch >= BATCH || q_head >= Q_HEADS || q_pos >= SEQ_Q) return;

            // Map Q head to KV head
            uint kv_head = q_head / HEADS_PER_KV;

            // Base offsets
            uint q_base = ((batch * Q_HEADS + q_head) * SEQ_Q + q_pos) * DIM;
            uint kv_head_base = (batch * KV_HEADS + kv_head) * SEQ_KV;

            // Online softmax state
            float m_i = -INFINITY;
            float l_i = 0.0f;

            uint kv_limit = IS_CAUSAL ? min(q_pos + 1, SEQ_KV) : SEQ_KV;

            // Pass 1: Compute softmax normalization
            for (uint k_pos = 0; k_pos < kv_limit; k_pos++) {
                uint k_base = (kv_head_base + k_pos) * DIM;
                float score = 0.0f;

                for (uint d = 0; d < DIM; d++) {
                    score += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                score *= SCALE;

                float m_new = max(m_i, score);
                float exp_old = exp(m_i - m_new);
                float exp_new = exp(score - m_new);
                l_i = l_i * exp_old + exp_new;
                m_i = m_new;
            }

            float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

            // Initialize output
            uint o_base = q_base;
            for (uint d = 0; d < DIM; d++) {
                O[o_base + d] = \(dtype)(0);
            }

            // Pass 2: Weighted sum
            for (uint k_pos = 0; k_pos < kv_limit; k_pos++) {
                uint k_base = (kv_head_base + k_pos) * DIM;
                float score = 0.0f;

                for (uint d = 0; d < DIM; d++) {
                    score += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                score *= SCALE;

                float weight = exp(score - m_i) * l_inv;

                for (uint d = 0; d < DIM; d++) {
                    O[o_base + d] = \(dtype)(float(O[o_base + d]) + weight * float(V[k_base + d]));
                }
            }
        }
        """
    }

    // MARK: - Performance Estimation

    private func computePerformanceEstimate(spec: AttentionSpecialization) -> PerformanceEstimate {
        // FLOPs calculation:
        // QK^T: 2 * batch * heads * seq_q * seq_kv * head_dim
        // Softmax: ~5 * batch * heads * seq_q * seq_kv (max, sub, exp, sum, div)
        // AV: 2 * batch * heads * seq_q * seq_kv * head_dim

        let qkFlops = 2 * spec.batchSize * spec.numHeads * spec.seqLenQ * spec.seqLenKV * spec.headDim
        let softmaxFlops = 5 * spec.batchSize * spec.numHeads * spec.seqLenQ * spec.seqLenKV
        let avFlops = 2 * spec.batchSize * spec.numHeads * spec.seqLenQ * spec.seqLenKV * spec.headDim
        let totalFlops = qkFlops + softmaxFlops + avFlops

        // Memory: Q, K, V inputs + O output
        let qBytes = spec.batchSize * spec.numHeads * spec.seqLenQ * spec.headDim * spec.dtype.byteSize
        let kvBytes = spec.batchSize * spec.numKVHeads * spec.seqLenKV * spec.headDim * spec.dtype.byteSize
        let totalMemory = qBytes + 2 * kvBytes + qBytes  // Q + K + V + O

        // Attention is typically compute-bound for long sequences
        let arithmeticIntensity = Double(totalFlops) / Double(totalMemory)
        let isComputeBound = arithmeticIntensity > 50

        return PerformanceEstimate(
            flops: totalFlops,
            memoryBytes: totalMemory,
            isComputeBound: isComputeBound
        )
    }
}

// MARK: - Common Attention Shapes

extension AttentionSpecialization {

    /// Llama 7B attention configuration.
    public static func llama7B(batchSize: Int, seqLen: Int) -> AttentionSpecialization {
        return AttentionSpecialization(
            batchSize: batchSize,
            numHeads: 32,
            numKVHeads: 32,
            seqLenQ: seqLen,
            headDim: 128,
            isCausal: true
        )
    }

    /// Llama 70B attention configuration (GQA).
    public static func llama70B(batchSize: Int, seqLen: Int) -> AttentionSpecialization {
        return AttentionSpecialization(
            batchSize: batchSize,
            numHeads: 64,
            numKVHeads: 8,  // 8x GQA
            seqLenQ: seqLen,
            headDim: 128,
            isCausal: true
        )
    }

    /// GPT-2 attention configuration.
    public static func gpt2(batchSize: Int, seqLen: Int) -> AttentionSpecialization {
        return AttentionSpecialization(
            batchSize: batchSize,
            numHeads: 12,
            seqLenQ: seqLen,
            headDim: 64,
            isCausal: true
        )
    }

    /// BERT attention configuration (non-causal).
    public static func bert(batchSize: Int, seqLen: Int) -> AttentionSpecialization {
        return AttentionSpecialization(
            batchSize: batchSize,
            numHeads: 12,
            seqLenQ: seqLen,
            headDim: 64,
            isCausal: false  // BERT uses full attention
        )
    }
}
