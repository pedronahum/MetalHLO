// SpecializedMatMul.swift
// MetalHLOCore
//
// Shape-specialized matrix multiplication kernel generator.

import Foundation

/// Configuration for specialized matrix multiplication.
public struct MatMulSpecialization: Hashable, Sendable {
    /// M dimension (rows of output).
    public let M: Int

    /// N dimension (columns of output).
    public let N: Int

    /// K dimension (contraction dimension).
    public let K: Int

    /// Whether A is transposed.
    public let transA: Bool

    /// Whether B is transposed.
    public let transB: Bool

    /// Data type.
    public let dtype: MetalType

    /// Batch size (0 for non-batched).
    public let batchSize: Int

    /// Alpha scaling factor.
    public let alpha: Float

    /// Beta scaling factor (for C += alpha * A @ B + beta * C).
    public let beta: Float

    public init(
        M: Int, N: Int, K: Int,
        transA: Bool = false,
        transB: Bool = false,
        dtype: MetalType = .float,
        batchSize: Int = 0,
        alpha: Float = 1.0,
        beta: Float = 0.0
    ) {
        self.M = M
        self.N = N
        self.K = K
        self.transA = transA
        self.transB = transB
        self.dtype = dtype
        self.batchSize = batchSize
        self.alpha = alpha
        self.beta = beta
    }

    /// Unique identifier for this specialization.
    public var identifier: String {
        var parts = ["matmul"]
        if batchSize > 0 { parts.append("b\(batchSize)") }
        parts.append("\(M)x\(K)x\(N)")
        if transA { parts.append("tA") }
        if transB { parts.append("tB") }
        parts.append(dtype.rawValue)
        return parts.joined(separator: "_")
    }
}

/// Generates shape-specialized matrix multiplication kernels.
///
/// This generator creates Metal kernels with:
/// - Compile-time constant dimensions (no runtime branching)
/// - Optimal tile sizes for the exact problem
/// - Unrolled loops where beneficial
/// - Vectorized memory access when dimensions align
public struct SpecializedMatMulGenerator: Sendable {

    /// Tile calculator for optimal configurations.
    private let tileCalculator: TileCalculator

    public init(tileCalculator: TileCalculator = TileCalculator()) {
        self.tileCalculator = tileCalculator
    }

    /// Generates a specialized kernel for the given configuration.
    public func generate(spec: MatMulSpecialization) -> SpecializedKernel {
        let tileConfig = tileCalculator.calculateMatMulTiles(
            M: spec.M,
            N: spec.N,
            K: spec.K,
            elementType: spec.dtype == .half ? .float16 : .float32
        )

        let source: String
        if spec.batchSize > 0 {
            source = generateBatchedKernel(spec: spec, tileConfig: tileConfig)
        } else if shouldUseSimdGroupMatmul(spec: spec, tileConfig: tileConfig) {
            source = generateSimdGroupKernel(spec: spec, tileConfig: tileConfig)
        } else if shouldUseTiledKernel(spec: spec, tileConfig: tileConfig) {
            source = generateTiledKernel(spec: spec, tileConfig: tileConfig)
        } else {
            source = generateSimpleKernel(spec: spec)
        }

        let flops = spec.batchSize > 0
            ? spec.batchSize * 2 * spec.M * spec.N * spec.K
            : 2 * spec.M * spec.N * spec.K

        let memoryBytes = spec.batchSize > 0
            ? spec.batchSize * (spec.M * spec.K + spec.K * spec.N + spec.M * spec.N) * spec.dtype.byteSize
            : (spec.M * spec.K + spec.K * spec.N + spec.M * spec.N) * spec.dtype.byteSize

        let estimate = PerformanceEstimate(
            flops: flops,
            memoryBytes: memoryBytes,
            isComputeBound: Double(flops) / Double(memoryBytes) > 10
        )

        return SpecializedKernel(
            source: source,
            functionName: spec.identifier,
            tileConfig: TileConfigWrapper(matmul: tileConfig),
            estimatedMetrics: estimate
        )
    }

    // MARK: - Kernel Selection Heuristics

    private func shouldUseSimdGroupMatmul(spec: MatMulSpecialization, tileConfig: MatMulTileConfig) -> Bool {
        // Use simdgroup matrix ops for larger problems with aligned dimensions
        return spec.M >= 32 && spec.N >= 32 && spec.K >= 8
            && spec.M % 8 == 0 && spec.N % 8 == 0
            && tileCalculator.hasSimdgroupMatrixOps
    }

    private func shouldUseTiledKernel(spec: MatMulSpecialization, tileConfig: MatMulTileConfig) -> Bool {
        // Use tiled kernel for medium to large problems
        return spec.M >= 64 || spec.N >= 64 || spec.K >= 64
    }

    // MARK: - Kernel Generators

    /// Generates a simple kernel for small matrices.
    private func generateSimpleKernel(spec: MatMulSpecialization) -> String {
        let dtype = spec.dtype.rawValue

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Simple matmul for small matrices: \(spec.M)x\(spec.K) @ \(spec.K)x\(spec.N)
        kernel void \(spec.identifier)(
            device const \(dtype)* A [[buffer(0)]],
            device const \(dtype)* B [[buffer(1)]],
            device \(dtype)* C [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            constexpr uint M = \(spec.M);
            constexpr uint N = \(spec.N);
            constexpr uint K = \(spec.K);
            constexpr \(dtype) alpha = \(dtype)(\(spec.alpha));
            constexpr \(dtype) beta = \(dtype)(\(spec.beta));

            uint row = gid.y;
            uint col = gid.x;

            if (row >= M || col >= N) return;

            \(dtype) acc = 0;

            // Fully unrolled for small K
            \(generateUnrolledDotProduct(spec: spec))

            if (beta != 0) {
                C[row * N + col] = alpha * acc + beta * C[row * N + col];
            } else {
                C[row * N + col] = alpha * acc;
            }
        }
        """
    }

    /// Generates a tiled kernel using shared memory.
    private func generateTiledKernel(spec: MatMulSpecialization, tileConfig: MatMulTileConfig) -> String {
        let dtype = spec.dtype.rawValue
        let tileM = tileConfig.tileM
        let tileN = tileConfig.tileN
        let tileK = tileConfig.tileK

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Tiled matmul: \(spec.M)x\(spec.K) @ \(spec.K)x\(spec.N)
        // Tile size: \(tileM)x\(tileN)x\(tileK)
        kernel void \(spec.identifier)(
            device const \(dtype)* A [[buffer(0)]],
            device const \(dtype)* B [[buffer(1)]],
            device \(dtype)* C [[buffer(2)]],
            threadgroup \(dtype)* shared_A [[threadgroup(0)]],
            threadgroup \(dtype)* shared_B [[threadgroup(1)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            constexpr uint M = \(spec.M);
            constexpr uint N = \(spec.N);
            constexpr uint K = \(spec.K);
            constexpr uint TILE_M = \(tileM);
            constexpr uint TILE_N = \(tileN);
            constexpr uint TILE_K = \(tileK);

            // Global output position
            uint global_row = tgid.y * TILE_M + tid.y;
            uint global_col = tgid.x * TILE_N + tid.x;

            // Accumulator
            \(dtype) acc = 0;

            // Loop over K tiles
            for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
                // Collaborative load of A tile
                uint a_row = tgid.y * TILE_M + tid.y;
                uint a_col = k_tile + tid.x;
                if (a_row < M && a_col < K) {
                    shared_A[tid.y * TILE_K + tid.x] = A[a_row * K + a_col];
                } else {
                    shared_A[tid.y * TILE_K + tid.x] = 0;
                }

                // Collaborative load of B tile
                uint b_row = k_tile + tid.y;
                uint b_col = tgid.x * TILE_N + tid.x;
                if (b_row < K && b_col < N) {
                    shared_B[tid.y * TILE_N + tid.x] = B[b_row * N + b_col];
                } else {
                    shared_B[tid.y * TILE_N + tid.x] = 0;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Compute partial dot product
                for (uint k = 0; k < TILE_K; k++) {
                    acc += shared_A[tid.y * TILE_K + k] * shared_B[k * TILE_N + tid.x];
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Write result
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = acc;
            }
        }
        """
    }

    /// Generates a kernel using simdgroup matrix operations (Apple Silicon optimized).
    private func generateSimdGroupKernel(spec: MatMulSpecialization, tileConfig: MatMulTileConfig) -> String {
        let dtype = spec.dtype.rawValue
        let simdType = spec.dtype == .half ? "simdgroup_half8x8" : "simdgroup_float8x8"

        return """
        #include <metal_stdlib>
        using namespace metal;

        // SIMD group matmul: \(spec.M)x\(spec.K) @ \(spec.K)x\(spec.N)
        // Uses Apple Silicon simdgroup_matrix operations for optimal performance
        kernel void \(spec.identifier)(
            device const \(dtype)* A [[buffer(0)]],
            device const \(dtype)* B [[buffer(1)]],
            device \(dtype)* C [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]],
            uint simd_lane_id [[thread_index_in_simdgroup]],
            uint simd_group_id [[simdgroup_index_in_threadgroup]]
        ) {
            constexpr uint M = \(spec.M);
            constexpr uint N = \(spec.N);
            constexpr uint K = \(spec.K);

            // Each simdgroup computes an 8x8 tile of the output
            uint tile_row = (gid.y / 8) * 8;
            uint tile_col = (gid.x / 8) * 8;

            if (tile_row >= M || tile_col >= N) return;

            // Accumulator matrix (8x8)
            \(simdType) acc;
            acc = make_filled_simdgroup_matrix<\(dtype), 8, 8>(0);

            // Loop over K in steps of 8
            for (uint k = 0; k < K; k += 8) {
                // Load A tile (8x8)
                \(simdType) a_tile;
                simdgroup_load(a_tile, A + tile_row * K + k, K);

                // Load B tile (8x8)
                \(simdType) b_tile;
                simdgroup_load(b_tile, B + k * N + tile_col, N);

                // Multiply-accumulate
                simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
            }

            // Store result
            simdgroup_store(acc, C + tile_row * N + tile_col, N);
        }
        """
    }

    /// Generates a batched matrix multiplication kernel.
    private func generateBatchedKernel(spec: MatMulSpecialization, tileConfig: MatMulTileConfig) -> String {
        let dtype = spec.dtype.rawValue

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Batched matmul: \(spec.batchSize) x (\(spec.M)x\(spec.K) @ \(spec.K)x\(spec.N))
        kernel void \(spec.identifier)(
            device const \(dtype)* A [[buffer(0)]],
            device const \(dtype)* B [[buffer(1)]],
            device \(dtype)* C [[buffer(2)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            constexpr uint BATCH = \(spec.batchSize);
            constexpr uint M = \(spec.M);
            constexpr uint N = \(spec.N);
            constexpr uint K = \(spec.K);

            uint batch = gid.z;
            uint row = gid.y;
            uint col = gid.x;

            if (batch >= BATCH || row >= M || col >= N) return;

            // Batch offsets
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

    // MARK: - Helper Methods

    private func generateUnrolledDotProduct(spec: MatMulSpecialization) -> String {
        // Unroll for small K values
        if spec.K <= 8 {
            var lines: [String] = []
            for k in 0..<spec.K {
                let aIdx = spec.transA ? "(\(k) * M + row)" : "(row * K + \(k))"
                let bIdx = spec.transB ? "(col * K + \(k))" : "(\(k) * N + col)"
                lines.append("acc += A[\(aIdx)] * B[\(bIdx)];")
            }
            return lines.joined(separator: "\n            ")
        } else {
            // Use loop for larger K
            let aIdx = spec.transA ? "(k * M + row)" : "(row * K + k)"
            let bIdx = spec.transB ? "(col * K + k)" : "(k * N + col)"
            return """
            for (uint k = 0; k < K; k++) {
                        acc += A[\(aIdx)] * B[\(bIdx)];
                    }
            """
        }
    }
}

// MARK: - Common MatMul Shapes

extension MatMulSpecialization {

    /// Common shapes for transformer models.
    public static func transformerQK(batchHeads: Int, seqLen: Int, headDim: Int) -> MatMulSpecialization {
        // Q @ K^T: [batch*heads, seq, dim] @ [batch*heads, dim, seq] -> [batch*heads, seq, seq]
        return MatMulSpecialization(
            M: seqLen, N: seqLen, K: headDim,
            transA: false, transB: true,
            batchSize: batchHeads
        )
    }

    public static func transformerAV(batchHeads: Int, seqLen: Int, headDim: Int) -> MatMulSpecialization {
        // Attn @ V: [batch*heads, seq, seq] @ [batch*heads, seq, dim] -> [batch*heads, seq, dim]
        return MatMulSpecialization(
            M: seqLen, N: headDim, K: seqLen,
            batchSize: batchHeads
        )
    }

    public static func transformerFFN1(batchSeq: Int, hiddenDim: Int, ffnDim: Int) -> MatMulSpecialization {
        // FFN up projection: [batch*seq, hidden] @ [hidden, ffn] -> [batch*seq, ffn]
        return MatMulSpecialization(M: batchSeq, N: ffnDim, K: hiddenDim)
    }

    public static func transformerFFN2(batchSeq: Int, ffnDim: Int, hiddenDim: Int) -> MatMulSpecialization {
        // FFN down projection: [batch*seq, ffn] @ [ffn, hidden] -> [batch*seq, hidden]
        return MatMulSpecialization(M: batchSeq, N: hiddenDim, K: ffnDim)
    }
}
