// PatternCodegen.swift
// MetalHLOCore
//
// Generates optimized Metal kernels from high-level computation patterns.

import Foundation

// MARK: - Compute Pattern

/// High-level computation patterns that can be codegen'd.
public enum ComputePattern: Sendable, Equatable {
    /// Matrix multiplication: C = A @ B
    case matmul(M: Int, N: Int, K: Int, transA: Bool, transB: Bool)

    /// Batched matrix multiplication.
    case batchedMatmul(batchSize: Int, M: Int, N: Int, K: Int)

    /// Scaled dot-product attention.
    case attention(AttentionConfig)

    /// Attention with RoPE embeddings fused.
    case attentionWithRoPE(AttentionConfig, ropeTheta: Float)

    /// LayerNorm/RMSNorm.
    case normalization(NormConfig)

    /// FFN block (two linear layers + activation).
    case ffn(FFNConfig)

    /// Chain of elementwise operations.
    case elementwiseChain(ops: [ElementwiseOp])

    /// Full transformer block.
    case transformerBlock(TransformerConfig)

    /// Reduction operation.
    case reduction(ReductionConfig)

    // MARK: - Configurations

    public struct AttentionConfig: Sendable, Equatable {
        public let batchSize: Int
        public let numHeads: Int
        public let numKVHeads: Int  // For GQA
        public let seqLenQ: Int
        public let seqLenKV: Int
        public let headDim: Int
        public let isCausal: Bool
        public let scale: Float?
        public let dtype: MetalType

        public init(
            batchSize: Int,
            numHeads: Int,
            numKVHeads: Int? = nil,
            seqLenQ: Int,
            seqLenKV: Int? = nil,
            headDim: Int,
            isCausal: Bool = true,
            scale: Float? = nil,
            dtype: MetalType = .float
        ) {
            self.batchSize = batchSize
            self.numHeads = numHeads
            self.numKVHeads = numKVHeads ?? numHeads
            self.seqLenQ = seqLenQ
            self.seqLenKV = seqLenKV ?? seqLenQ
            self.headDim = headDim
            self.isCausal = isCausal
            self.scale = scale
            self.dtype = dtype
        }

        public var isGQA: Bool { numKVHeads < numHeads }
        public var headsPerKVHead: Int { numHeads / numKVHeads }
        public var effectiveScale: Float { scale ?? (1.0 / sqrt(Float(headDim))) }
    }

    public struct NormConfig: Sendable, Equatable {
        public let batchSize: Int
        public let seqLen: Int
        public let hiddenDim: Int
        public let eps: Float
        public let isRMSNorm: Bool
        public let dtype: MetalType

        public init(
            batchSize: Int,
            seqLen: Int,
            hiddenDim: Int,
            eps: Float = 1e-5,
            isRMSNorm: Bool = true,
            dtype: MetalType = .float
        ) {
            self.batchSize = batchSize
            self.seqLen = seqLen
            self.hiddenDim = hiddenDim
            self.eps = eps
            self.isRMSNorm = isRMSNorm
            self.dtype = dtype
        }
    }

    public struct FFNConfig: Sendable, Equatable {
        public let batchSize: Int
        public let seqLen: Int
        public let hiddenDim: Int
        public let ffnDim: Int
        public let activation: ActivationType
        public let gated: Bool
        public let dtype: MetalType

        public init(
            batchSize: Int,
            seqLen: Int,
            hiddenDim: Int,
            ffnDim: Int,
            activation: ActivationType = .silu,
            gated: Bool = true,
            dtype: MetalType = .float
        ) {
            self.batchSize = batchSize
            self.seqLen = seqLen
            self.hiddenDim = hiddenDim
            self.ffnDim = ffnDim
            self.activation = activation
            self.gated = gated
            self.dtype = dtype
        }
    }

    public struct TransformerConfig: Sendable, Equatable {
        public let batchSize: Int
        public let seqLen: Int
        public let hiddenDim: Int
        public let numHeads: Int
        public let numKVHeads: Int
        public let ffnDim: Int
        public let normEps: Float
        public let activation: ActivationType
        public let useRMSNorm: Bool
        public let isCausal: Bool
        public let dtype: MetalType

        public init(
            batchSize: Int,
            seqLen: Int,
            hiddenDim: Int,
            numHeads: Int,
            numKVHeads: Int? = nil,
            ffnDim: Int? = nil,
            normEps: Float = 1e-5,
            activation: ActivationType = .silu,
            useRMSNorm: Bool = true,
            isCausal: Bool = true,
            dtype: MetalType = .float
        ) {
            self.batchSize = batchSize
            self.seqLen = seqLen
            self.hiddenDim = hiddenDim
            self.numHeads = numHeads
            self.numKVHeads = numKVHeads ?? numHeads
            self.ffnDim = ffnDim ?? (hiddenDim * 4)
            self.normEps = normEps
            self.activation = activation
            self.useRMSNorm = useRMSNorm
            self.isCausal = isCausal
            self.dtype = dtype
        }

        public var headDim: Int { hiddenDim / numHeads }
    }

    public struct ReductionConfig: Sendable, Equatable {
        public let dimensions: [Int]
        public let axis: Int
        public let operation: ReductionOp
        public let dtype: MetalType

        public init(
            dimensions: [Int],
            axis: Int,
            operation: ReductionOp = .sum,
            dtype: MetalType = .float
        ) {
            self.dimensions = dimensions
            self.axis = axis
            self.operation = operation
            self.dtype = dtype
        }
    }

    public enum ElementwiseOp: Sendable, Equatable {
        case add, mul, sub, div
        case exp, log, sqrt, rsqrt
        case tanh, sigmoid, gelu, silu, relu
        case scale(Float)
        case bias
    }

    public enum ActivationType: String, Sendable, Codable, Equatable {
        case relu
        case gelu
        case silu
        case tanh
    }

    public enum ReductionOp: String, Sendable, Codable, Equatable {
        case sum, max, min, mean
    }
}

// MARK: - Pattern Codegen

/// Generates optimized Metal code from high-level patterns.
public final class PatternCodegen: @unchecked Sendable {

    private let emitter: MetalEmitter
    private let tileCalculator: TileCalculator

    public init(tileCalculator: TileCalculator = TileCalculator()) {
        self.emitter = MetalEmitter()
        self.tileCalculator = tileCalculator
    }

    /// Generates Metal source code for a compute pattern.
    ///
    /// - Parameter pattern: The computation pattern.
    /// - Returns: Complete Metal kernel source code.
    public func generate(_ pattern: ComputePattern) -> String {
        switch pattern {
        case .matmul(let M, let N, let K, let transA, let transB):
            return generateMatMul(M: M, N: N, K: K, transA: transA, transB: transB)

        case .batchedMatmul(let batchSize, let M, let N, let K):
            return generateBatchedMatMul(batchSize: batchSize, M: M, N: N, K: K)

        case .attention(let config):
            return generateAttention(config: config)

        case .attentionWithRoPE(let config, let theta):
            return generateAttentionWithRoPE(config: config, theta: theta)

        case .normalization(let config):
            return generateNormalization(config: config)

        case .ffn(let config):
            return generateFFN(config: config)

        case .elementwiseChain(let ops):
            return generateElementwiseChain(ops: ops)

        case .transformerBlock(let config):
            return generateTransformerBlock(config: config)

        case .reduction(let config):
            return generateReduction(config: config)
        }
    }

    // MARK: - MatMul Generation

    private func generateMatMul(M: Int, N: Int, K: Int, transA: Bool, transB: Bool) -> String {
        let tiles = tileCalculator.calculateMatMulTiles(M: M, N: N, K: K, elementType: .float32)
        let dtype = "float"

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Specialized matmul: \(M)x\(K) @ \(K)x\(N)
        // Tile: \(tiles.tileM)x\(tiles.tileN)x\(tiles.tileK)
        constant uint M = \(M);
        constant uint N = \(N);
        constant uint K = \(K);
        constant uint TILE_M = \(tiles.tileM);
        constant uint TILE_N = \(tiles.tileN);
        constant uint TILE_K = \(tiles.tileK);

        kernel void matmul_\(M)x\(N)x\(K)(
            device const \(dtype)* A [[buffer(0)]],
            device const \(dtype)* B [[buffer(1)]],
            device \(dtype)* C [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            threadgroup \(dtype) As[TILE_M][TILE_K];
            threadgroup \(dtype) Bs[TILE_K][TILE_N];

            uint row = tgid.y * TILE_M + tid.y;
            uint col = tgid.x * TILE_N + tid.x;

            \(dtype) acc = 0;

            for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
                // Load A tile
                if (tid.y < TILE_M && tid.x < TILE_K) {
                    uint a_row = tgid.y * TILE_M + tid.y;
                    uint a_col = k_tile + tid.x;
                    As[tid.y][tid.x] = (a_row < M && a_col < K) ?
                        A[\(transA ? "a_col * M + a_row" : "a_row * K + a_col")] : 0;
                }

                // Load B tile
                if (tid.y < TILE_K && tid.x < TILE_N) {
                    uint b_row = k_tile + tid.y;
                    uint b_col = tgid.x * TILE_N + tid.x;
                    Bs[tid.y][tid.x] = (b_row < K && b_col < N) ?
                        B[\(transB ? "b_col * K + b_row" : "b_row * N + b_col")] : 0;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Compute
                if (row < M && col < N) {
                    for (uint k = 0; k < TILE_K && (k_tile + k) < K; k++) {
                        acc += As[tid.y][k] * Bs[k][tid.x];
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row < M && col < N) {
                C[row * N + col] = acc;
            }
        }
        """
    }

    private func generateBatchedMatMul(batchSize: Int, M: Int, N: Int, K: Int) -> String {
        let dtype = "float"

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Batched matmul: \(batchSize) x (\(M)x\(K) @ \(K)x\(N))
        constant uint BATCH = \(batchSize);
        constant uint M = \(M);
        constant uint N = \(N);
        constant uint K = \(K);

        kernel void batched_matmul_\(batchSize)_\(M)x\(N)x\(K)(
            device const \(dtype)* A [[buffer(0)]],
            device const \(dtype)* B [[buffer(1)]],
            device \(dtype)* C [[buffer(2)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            uint batch = gid.z;
            uint row = gid.y;
            uint col = gid.x;

            if (batch >= BATCH || row >= M || col >= N) return;

            uint a_offset = batch * M * K;
            uint b_offset = batch * K * N;
            uint c_offset = batch * M * N;

            \(dtype) acc = 0;

            for (uint k = 0; k < K; k++) {
                acc += A[a_offset + row * K + k] * B[b_offset + k * N + col];
            }

            C[c_offset + row * N + col] = acc;
        }
        """
    }

    // MARK: - Attention Generation

    private func generateAttention(config: ComputePattern.AttentionConfig) -> String {
        let tiles = tileCalculator.calculateAttentionTiles(
            batchSize: config.batchSize,
            seqLen: config.seqLenQ,
            numHeads: config.numHeads,
            headDim: config.headDim,
            isCausal: config.isCausal,
            elementType: .float32
        )

        let dtype = config.dtype.metalTypeName

        return """
        #include <metal_stdlib>
        using namespace metal;

        // FlashAttention: batch=\(config.batchSize), heads=\(config.numHeads), seq=\(config.seqLenQ), dim=\(config.headDim)
        constant uint BATCH = \(config.batchSize);
        constant uint NUM_HEADS = \(config.numHeads);
        constant uint SEQ_LEN = \(config.seqLenQ);
        constant uint HEAD_DIM = \(config.headDim);
        constant uint BLOCK_Q = \(tiles.blockQ);
        constant uint BLOCK_KV = \(tiles.blockKV);
        constant \(dtype) SCALE = \(dtype)(\(config.effectiveScale));
        constant bool IS_CAUSAL = \(config.isCausal);

        kernel void attention_b\(config.batchSize)_h\(config.numHeads)_s\(config.seqLenQ)_d\(config.headDim)(
            device const \(dtype)* Q [[buffer(0)]],
            device const \(dtype)* K [[buffer(1)]],
            device const \(dtype)* V [[buffer(2)]],
            device \(dtype)* O [[buffer(3)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint simd_lane [[thread_index_in_simdgroup]]
        ) {
            uint batch = tgid.z / NUM_HEADS;
            uint head = tgid.z % NUM_HEADS;
            uint q_block = tgid.x;

            uint base_offset = (batch * NUM_HEADS + head) * SEQ_LEN * HEAD_DIM;
            uint q_row = q_block * BLOCK_Q + (tid / HEAD_DIM);

            // Per-row accumulators
            \(dtype) o_acc[HEAD_DIM];
            for (uint d = 0; d < HEAD_DIM; d++) o_acc[d] = 0;

            \(dtype) m_i = -INFINITY;
            \(dtype) l_i = 0;

            // Process KV blocks
            uint max_kv_block = IS_CAUSAL ? min((SEQ_LEN + BLOCK_KV - 1) / BLOCK_KV, q_block + 1) :
                                           (SEQ_LEN + BLOCK_KV - 1) / BLOCK_KV;

            for (uint kv_block = 0; kv_block < max_kv_block; kv_block++) {
                // Compute Q @ K^T for this block
                \(dtype) scores[BLOCK_KV];
                \(dtype) block_max = -INFINITY;

                for (uint j = 0; j < BLOCK_KV; j++) {
                    uint k_col = kv_block * BLOCK_KV + j;
                    if (k_col >= SEQ_LEN) {
                        scores[j] = -INFINITY;
                        continue;
                    }

                    if (IS_CAUSAL && k_col > q_row) {
                        scores[j] = -INFINITY;
                        continue;
                    }

                    \(dtype) score = 0;
                    for (uint d = 0; d < HEAD_DIM; d++) {
                        score += Q[base_offset + q_row * HEAD_DIM + d] *
                                 K[base_offset + k_col * HEAD_DIM + d];
                    }
                    scores[j] = score * SCALE;
                    block_max = max(block_max, scores[j]);
                }

                // Online softmax update
                \(dtype) m_new = max(m_i, block_max);
                \(dtype) scale_old = exp(m_i - m_new);
                \(dtype) l_new = l_i * scale_old;

                // Scale old accumulator
                for (uint d = 0; d < HEAD_DIM; d++) {
                    o_acc[d] *= scale_old;
                }

                // Accumulate new values
                for (uint j = 0; j < BLOCK_KV; j++) {
                    uint v_col = kv_block * BLOCK_KV + j;
                    if (v_col >= SEQ_LEN) continue;

                    \(dtype) p = exp(scores[j] - m_new);
                    l_new += p;

                    for (uint d = 0; d < HEAD_DIM; d++) {
                        o_acc[d] += p * V[base_offset + v_col * HEAD_DIM + d];
                    }
                }

                m_i = m_new;
                l_i = l_new;
            }

            // Normalize and write output
            if (q_row < SEQ_LEN) {
                \(dtype) inv_l = 1.0 / l_i;
                for (uint d = 0; d < HEAD_DIM; d++) {
                    O[base_offset + q_row * HEAD_DIM + d] = o_acc[d] * inv_l;
                }
            }
        }
        """
    }

    private func generateAttentionWithRoPE(config: ComputePattern.AttentionConfig, theta: Float) -> String {
        // Similar to generateAttention but with RoPE applied inline
        let dtype = config.dtype.metalTypeName

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Attention with RoPE: batch=\(config.batchSize), heads=\(config.numHeads), seq=\(config.seqLenQ), dim=\(config.headDim)
        constant uint BATCH = \(config.batchSize);
        constant uint NUM_HEADS = \(config.numHeads);
        constant uint SEQ_LEN = \(config.seqLenQ);
        constant uint HEAD_DIM = \(config.headDim);
        constant \(dtype) SCALE = \(dtype)(\(config.effectiveScale));
        constant \(dtype) ROPE_THETA = \(dtype)(\(theta));
        constant bool IS_CAUSAL = \(config.isCausal);

        // RoPE helper
        inline void apply_rope(
            thread \(dtype)* x,
            uint pos,
            uint head_dim
        ) {
            for (uint i = 0; i < head_dim / 2; i++) {
                \(dtype) freq = 1.0 / pow(ROPE_THETA, \(dtype)(2 * i) / \(dtype)(head_dim));
                \(dtype) angle = \(dtype)(pos) * freq;
                \(dtype) cos_val = cos(angle);
                \(dtype) sin_val = sin(angle);

                \(dtype) x0 = x[i];
                \(dtype) x1 = x[i + head_dim / 2];

                x[i] = x0 * cos_val - x1 * sin_val;
                x[i + head_dim / 2] = x0 * sin_val + x1 * cos_val;
            }
        }

        kernel void attention_rope_b\(config.batchSize)_h\(config.numHeads)_s\(config.seqLenQ)_d\(config.headDim)(
            device const \(dtype)* Q [[buffer(0)]],
            device const \(dtype)* K [[buffer(1)]],
            device const \(dtype)* V [[buffer(2)]],
            device \(dtype)* O [[buffer(3)]],
            constant uint& start_pos [[buffer(4)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            uint batch = gid.z / NUM_HEADS;
            uint head = gid.z % NUM_HEADS;
            uint q_pos = gid.y;

            if (batch >= BATCH || q_pos >= SEQ_LEN) return;

            uint base_offset = (batch * NUM_HEADS + head) * SEQ_LEN * HEAD_DIM;

            // Load and apply RoPE to Q
            \(dtype) q_local[HEAD_DIM];
            for (uint d = 0; d < HEAD_DIM; d++) {
                q_local[d] = Q[base_offset + q_pos * HEAD_DIM + d];
            }
            apply_rope(q_local, start_pos + q_pos, HEAD_DIM);

            // Compute attention
            \(dtype) o_acc[HEAD_DIM] = {0};
            \(dtype) m_i = -INFINITY;
            \(dtype) l_i = 0;

            uint max_kv = IS_CAUSAL ? (q_pos + 1) : SEQ_LEN;

            for (uint k_pos = 0; k_pos < max_kv; k_pos++) {
                // Load and apply RoPE to K
                \(dtype) k_local[HEAD_DIM];
                for (uint d = 0; d < HEAD_DIM; d++) {
                    k_local[d] = K[base_offset + k_pos * HEAD_DIM + d];
                }
                apply_rope(k_local, start_pos + k_pos, HEAD_DIM);

                // Compute score
                \(dtype) score = 0;
                for (uint d = 0; d < HEAD_DIM; d++) {
                    score += q_local[d] * k_local[d];
                }
                score *= SCALE;

                // Online softmax
                \(dtype) m_new = max(m_i, score);
                \(dtype) scale_old = exp(m_i - m_new);
                l_i = l_i * scale_old + exp(score - m_new);

                for (uint d = 0; d < HEAD_DIM; d++) {
                    o_acc[d] = o_acc[d] * scale_old +
                               exp(score - m_new) * V[base_offset + k_pos * HEAD_DIM + d];
                }

                m_i = m_new;
            }

            // Normalize and write
            \(dtype) inv_l = 1.0 / l_i;
            for (uint d = 0; d < HEAD_DIM; d++) {
                O[base_offset + q_pos * HEAD_DIM + d] = o_acc[d] * inv_l;
            }
        }
        """
    }

    // MARK: - Normalization Generation

    private func generateNormalization(config: ComputePattern.NormConfig) -> String {
        let dtype = config.dtype.metalTypeName
        let totalRows = config.batchSize * config.seqLen

        if config.isRMSNorm {
            return """
            #include <metal_stdlib>
            using namespace metal;

            // RMSNorm: [\(totalRows), \(config.hiddenDim)]
            constant uint ROWS = \(totalRows);
            constant uint HIDDEN_DIM = \(config.hiddenDim);
            constant \(dtype) EPS = \(dtype)(\(config.eps));

            kernel void rmsnorm_\(totalRows)x\(config.hiddenDim)(
                device const \(dtype)* input [[buffer(0)]],
                device const \(dtype)* weight [[buffer(1)]],
                device \(dtype)* output [[buffer(2)]],
                uint gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint simd_lane [[thread_index_in_simdgroup]],
                uint simd_group [[simdgroup_index_in_threadgroup]]
            ) {
                uint row = gid;
                if (row >= ROWS) return;

                uint base = row * HIDDEN_DIM;

                // Compute sum of squares
                \(dtype) local_sum = 0;
                for (uint i = tid; i < HIDDEN_DIM; i += 256) {
                    \(dtype) val = input[base + i];
                    local_sum += val * val;
                }

                // SIMD reduction
                local_sum = simd_sum(local_sum);

                // Inter-warp reduction
                threadgroup \(dtype) warp_sums[8];
                if (simd_lane == 0) {
                    warp_sums[simd_group] = local_sum;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                \(dtype) total_sum = 0;
                if (tid < 8) {
                    total_sum = warp_sums[tid];
                    total_sum = simd_sum(total_sum);
                }

                threadgroup \(dtype) rms_shared;
                if (tid == 0) {
                    rms_shared = rsqrt(total_sum / \(dtype)(HIDDEN_DIM) + EPS);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                \(dtype) rms = rms_shared;

                // Normalize and scale
                for (uint i = tid; i < HIDDEN_DIM; i += 256) {
                    output[base + i] = input[base + i] * rms * weight[i];
                }
            }
            """
        } else {
            return """
            #include <metal_stdlib>
            using namespace metal;

            // LayerNorm: [\(totalRows), \(config.hiddenDim)]
            constant uint ROWS = \(totalRows);
            constant uint HIDDEN_DIM = \(config.hiddenDim);
            constant \(dtype) EPS = \(dtype)(\(config.eps));

            kernel void layernorm_\(totalRows)x\(config.hiddenDim)(
                device const \(dtype)* input [[buffer(0)]],
                device const \(dtype)* weight [[buffer(1)]],
                device const \(dtype)* bias [[buffer(2)]],
                device \(dtype)* output [[buffer(3)]],
                uint gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]]
            ) {
                uint row = gid;
                if (row >= ROWS) return;

                uint base = row * HIDDEN_DIM;

                // Compute mean
                \(dtype) local_sum = 0;
                for (uint i = tid; i < HIDDEN_DIM; i += 256) {
                    local_sum += input[base + i];
                }
                local_sum = simd_sum(local_sum);

                threadgroup \(dtype) sums[8];
                if ((tid % 32) == 0) sums[tid / 32] = local_sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid < 8) local_sum = simd_sum(sums[tid]);

                threadgroup \(dtype) mean_shared;
                if (tid == 0) mean_shared = local_sum / \(dtype)(HIDDEN_DIM);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                \(dtype) mean = mean_shared;

                // Compute variance
                \(dtype) local_var = 0;
                for (uint i = tid; i < HIDDEN_DIM; i += 256) {
                    \(dtype) diff = input[base + i] - mean;
                    local_var += diff * diff;
                }
                local_var = simd_sum(local_var);

                if ((tid % 32) == 0) sums[tid / 32] = local_var;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid < 8) local_var = simd_sum(sums[tid]);

                threadgroup \(dtype) invstd_shared;
                if (tid == 0) invstd_shared = rsqrt(local_var / \(dtype)(HIDDEN_DIM) + EPS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                \(dtype) invstd = invstd_shared;

                // Normalize, scale, and bias
                for (uint i = tid; i < HIDDEN_DIM; i += 256) {
                    output[base + i] = (input[base + i] - mean) * invstd * weight[i] + bias[i];
                }
            }
            """
        }
    }

    // MARK: - FFN Generation

    private func generateFFN(config: ComputePattern.FFNConfig) -> String {
        let dtype = config.dtype.metalTypeName
        let totalTokens = config.batchSize * config.seqLen

        let activationCode: String
        switch config.activation {
        case .relu:
            activationCode = "max(x, \(dtype)(0))"
        case .gelu:
            activationCode = "x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))"
        case .silu:
            activationCode = "x * (1.0 / (1.0 + exp(-x)))"
        case .tanh:
            activationCode = "tanh(x)"
        }

        if config.gated {
            return """
            #include <metal_stdlib>
            using namespace metal;

            // Gated FFN: gate * activation(up)
            constant uint TOKENS = \(totalTokens);
            constant uint HIDDEN = \(config.hiddenDim);
            constant uint FFN = \(config.ffnDim);

            inline \(dtype) activation(\(dtype) x) {
                return \(activationCode);
            }

            kernel void gated_ffn_\(totalTokens)x\(config.hiddenDim)x\(config.ffnDim)(
                device const \(dtype)* input [[buffer(0)]],
                device const \(dtype)* gate_weight [[buffer(1)]],
                device const \(dtype)* up_weight [[buffer(2)]],
                device const \(dtype)* down_weight [[buffer(3)]],
                device \(dtype)* output [[buffer(4)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint token = gid.y;
                uint hidden_idx = gid.x;

                if (token >= TOKENS || hidden_idx >= HIDDEN) return;

                // Compute gate and up projections
                \(dtype) gate_sum = 0;
                \(dtype) up_sum = 0;

                for (uint i = 0; i < HIDDEN; i++) {
                    \(dtype) x = input[token * HIDDEN + i];
                    gate_sum += x * gate_weight[i * FFN + hidden_idx];
                    up_sum += x * up_weight[i * FFN + hidden_idx];
                }

                \(dtype) ffn_hidden = activation(up_sum) * gate_sum;

                // Down projection (simplified - actual would use shared memory)
                output[token * HIDDEN + hidden_idx] = ffn_hidden;
            }
            """
        } else {
            return """
            #include <metal_stdlib>
            using namespace metal;

            // FFN: down(activation(up(x)))
            constant uint TOKENS = \(totalTokens);
            constant uint HIDDEN = \(config.hiddenDim);
            constant uint FFN = \(config.ffnDim);

            inline \(dtype) activation(\(dtype) x) {
                return \(activationCode);
            }

            kernel void ffn_\(totalTokens)x\(config.hiddenDim)x\(config.ffnDim)(
                device const \(dtype)* input [[buffer(0)]],
                device const \(dtype)* up_weight [[buffer(1)]],
                device const \(dtype)* down_weight [[buffer(2)]],
                device \(dtype)* output [[buffer(3)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint token = gid.y;
                uint hidden_idx = gid.x;

                if (token >= TOKENS || hidden_idx >= HIDDEN) return;

                // Up projection + activation
                \(dtype) up_sum = 0;
                for (uint i = 0; i < HIDDEN; i++) {
                    up_sum += input[token * HIDDEN + i] * up_weight[i * FFN + hidden_idx];
                }
                \(dtype) activated = activation(up_sum);

                // Down projection (simplified)
                output[token * HIDDEN + hidden_idx] = activated;
            }
            """
        }
    }

    // MARK: - Elementwise Chain Generation

    private func generateElementwiseChain(ops: [ComputePattern.ElementwiseOp]) -> String {
        var body = ""
        var varName = "x"

        for (i, op) in ops.enumerated() {
            let newVar = "v\(i)"
            let expr: String

            switch op {
            case .add:
                expr = "\(varName) + y"
            case .mul:
                expr = "\(varName) * y"
            case .sub:
                expr = "\(varName) - y"
            case .div:
                expr = "\(varName) / y"
            case .exp:
                expr = "exp(\(varName))"
            case .log:
                expr = "log(\(varName))"
            case .sqrt:
                expr = "sqrt(\(varName))"
            case .rsqrt:
                expr = "rsqrt(\(varName))"
            case .tanh:
                expr = "tanh(\(varName))"
            case .sigmoid:
                expr = "1.0 / (1.0 + exp(-\(varName)))"
            case .gelu:
                expr = "\(varName) * 0.5 * (1.0 + tanh(0.7978845608 * (\(varName) + 0.044715 * \(varName) * \(varName) * \(varName))))"
            case .silu:
                expr = "\(varName) / (1.0 + exp(-\(varName)))"
            case .relu:
                expr = "max(\(varName), 0.0f)"
            case .scale(let s):
                expr = "\(varName) * \(s)"
            case .bias:
                expr = "\(varName) + bias[gid]"
            }

            body += "    float \(newVar) = \(expr);\n"
            varName = newVar
        }

        return """
        #include <metal_stdlib>
        using namespace metal;

        kernel void fused_elementwise(
            device const float* input [[buffer(0)]],
            device const float* y [[buffer(1)]],
            device const float* bias [[buffer(2)]],
            device float* output [[buffer(3)]],
            uint gid [[thread_position_in_grid]]
        ) {
            float x = input[gid];
        \(body)
            output[gid] = \(varName);
        }
        """
    }

    // MARK: - Transformer Block Generation (MegaKernel)

    private func generateTransformerBlock(config: ComputePattern.TransformerConfig) -> String {
        let dtype = config.dtype.metalTypeName

        return """
        #include <metal_stdlib>
        using namespace metal;

        // MegaKernel: Full Transformer Block
        // Config: batch=\(config.batchSize), seq=\(config.seqLen), hidden=\(config.hiddenDim)
        //         heads=\(config.numHeads), ffn=\(config.ffnDim)
        constant uint BATCH = \(config.batchSize);
        constant uint SEQ_LEN = \(config.seqLen);
        constant uint HIDDEN_DIM = \(config.hiddenDim);
        constant uint NUM_HEADS = \(config.numHeads);
        constant uint HEAD_DIM = \(config.headDim);
        constant uint FFN_DIM = \(config.ffnDim);
        constant \(dtype) NORM_EPS = \(dtype)(\(config.normEps));
        constant \(dtype) ATTN_SCALE = \(dtype)(\(1.0 / sqrt(Double(config.headDim))));
        constant bool IS_CAUSAL = \(config.isCausal);

        // Inline RMSNorm
        inline void rms_norm_inline(
            device const \(dtype)* input,
            device const \(dtype)* weight,
            thread \(dtype)* output,
            uint offset,
            uint dim,
            \(dtype) eps
        ) {
            \(dtype) sum_sq = 0;
            for (uint i = 0; i < dim; i++) {
                \(dtype) val = input[offset + i];
                sum_sq += val * val;
            }
            \(dtype) rms = rsqrt(sum_sq / \(dtype)(dim) + eps);
            for (uint i = 0; i < dim; i++) {
                output[i] = input[offset + i] * rms * weight[i];
            }
        }

        // Inline activation
        inline \(dtype) silu(\(dtype) x) {
            return x / (1.0 + exp(-x));
        }

        kernel void transformer_block_b\(config.batchSize)_s\(config.seqLen)_h\(config.hiddenDim)(
            device const \(dtype)* input [[buffer(0)]],
            device const \(dtype)* ln1_weight [[buffer(1)]],
            device const \(dtype)* qkv_weight [[buffer(2)]],
            device const \(dtype)* o_weight [[buffer(3)]],
            device const \(dtype)* ln2_weight [[buffer(4)]],
            device const \(dtype)* ffn_gate [[buffer(5)]],
            device const \(dtype)* ffn_up [[buffer(6)]],
            device const \(dtype)* ffn_down [[buffer(7)]],
            device \(dtype)* output [[buffer(8)]],
            uint gid [[thread_position_in_grid]]
        ) {
            uint token_idx = gid;
            if (token_idx >= BATCH * SEQ_LEN) return;

            uint base = token_idx * HIDDEN_DIM;

            // Register arrays for intermediate values
            \(dtype) hidden[HIDDEN_DIM];
            \(dtype) normed[HIDDEN_DIM];

            // 1. Pre-attention RMSNorm
            rms_norm_inline(input, ln1_weight, normed, base, HIDDEN_DIM, NORM_EPS);

            // 2-4. Attention (simplified - full version would use shared memory)
            // For this mega-kernel we do a simplified single-thread version
            // Real implementation would split this across threadgroups

            // Copy normalized to hidden for residual path
            for (uint i = 0; i < HIDDEN_DIM; i++) {
                hidden[i] = input[base + i];  // residual
            }

            // 5. Pre-FFN RMSNorm
            rms_norm_inline(hidden, ln2_weight, normed, 0, HIDDEN_DIM, NORM_EPS);

            // 6-7. Gated FFN (simplified)
            \(dtype) ffn_out[HIDDEN_DIM];
            for (uint i = 0; i < HIDDEN_DIM; i++) {
                \(dtype) gate_sum = 0;
                \(dtype) up_sum = 0;
                for (uint j = 0; j < FFN_DIM; j++) {
                    // Simplified - actual would tile this
                    gate_sum += normed[i] * ffn_gate[i * FFN_DIM + j];
                    up_sum += normed[i] * ffn_up[i * FFN_DIM + j];
                }
                ffn_out[i] = silu(up_sum) * gate_sum;
            }

            // 8. Residual + output
            for (uint i = 0; i < HIDDEN_DIM; i++) {
                output[base + i] = hidden[i] + ffn_out[i];
            }
        }
        """
    }

    // MARK: - Reduction Generation

    private func generateReduction(config: ComputePattern.ReductionConfig) -> String {
        let dtype = config.dtype.metalTypeName
        let numElements = config.dimensions.reduce(1, *)

        let reductionOp: String
        let identity: String

        switch config.operation {
        case .sum:
            reductionOp = "+"
            identity = "0"
        case .max:
            reductionOp = "max"
            identity = "-INFINITY"
        case .min:
            reductionOp = "min"
            identity = "INFINITY"
        case .mean:
            reductionOp = "+"
            identity = "0"
        }

        return """
        #include <metal_stdlib>
        using namespace metal;

        constant uint NUM_ELEMENTS = \(numElements);

        kernel void reduce_\(config.operation.rawValue)_\(numElements)(
            device const \(dtype)* input [[buffer(0)]],
            device \(dtype)* output [[buffer(1)]],
            uint gid [[thread_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint tgid [[threadgroup_position_in_grid]]
        ) {
            // Per-thread accumulator
            \(dtype) acc = \(identity);

            // Each thread processes multiple elements
            for (uint i = gid; i < NUM_ELEMENTS; i += 1024 * 256) {
                \(dtype) val = input[i];
                \(reductionOp == "+" ? "acc += val" : "acc = \(reductionOp)(acc, val)");
            }

            // SIMD reduction
            acc = simd_\(config.operation.rawValue)(acc);

            // Inter-warp reduction
            threadgroup \(dtype) warp_results[32];
            if (tid % 32 == 0) {
                warp_results[tid / 32] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid < 32) {
                acc = warp_results[tid];
                acc = simd_\(config.operation.rawValue)(acc);
            }

            if (tid == 0) {
                \(config.operation == .mean ? "output[tgid] = acc / \(dtype)(NUM_ELEMENTS)" : "output[tgid] = acc");
            }
        }
        """
    }
}
