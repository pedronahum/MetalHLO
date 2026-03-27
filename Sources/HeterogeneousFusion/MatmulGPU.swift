// MatmulGPU.swift
// HeterogeneousFusion
//
// Row-sliceable matrix multiplication via a custom Metal compute kernel.
// Uses simdgroup_matrix operations (hardware-accelerated 8×8 multiply-accumulate)
// matching MetalHLO's CodeGenerator approach for competitive performance.

import Metal
import QuartzCore

/// Performs matmul on GPU via a custom Metal compute kernel.
/// Uses simdgroup 8×8 hardware matrix operations for high throughput.
/// Supports row-slicing: computes C_slice = A_slice @ B.
public final class MatmulGPU: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState

    /// 32×32 output tile per threadgroup, 4 simdgroups of 32 threads each.
    private static let tileM = 32
    private static let tileN = 32
    private static let tileK = 8
    private static let simdgroupsPerThreadgroup = 4
    private static let threadsPerThreadgroup = simdgroupsPerThreadgroup * 32  // 128

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create command queue")
        }
        self.commandQueue = queue

        let kernelSource = Self.generateSimdgroupKernel()
        let library = try device.makeLibrary(source: kernelSource, options: nil)
        guard let function = library.makeFunction(name: "matmul_simdgroup") else {
            throw HeterogeneousError.metalSetupFailed("Failed to create matmul_simdgroup function")
        }
        self.pipelineState = try device.makeComputePipelineState(function: function)
    }

    /// Encode a row-sliced matmul into the given command buffer.
    public func encode(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        sliceM: Int, K: Int, N: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        encodeInternal(A: A, B: B, C: C, aOffset: 0, cOffset: 0,
                       sliceM: sliceM, K: K, N: N, commandBuffer: commandBuffer)
    }

    /// Encode with byte offsets into shared A/C buffers (for fused execution).
    public func encode(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        aOffset: Int, cOffset: Int,
        sliceM: Int, K: Int, N: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        encodeInternal(A: A, B: B, C: C, aOffset: aOffset, cOffset: cOffset,
                       sliceM: sliceM, K: K, N: N, commandBuffer: commandBuffer)
    }

    /// Execute matmul synchronously: C = A @ B. Returns wall-clock time in seconds.
    public func execute(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        M: Int, K: Int, N: Int
    ) -> Double {
        guard let cb = commandQueue.makeCommandBuffer() else { return 0 }
        encode(A: A, B: B, C: C, sliceM: M, K: K, N: N, commandBuffer: cb)

        let start = CACurrentMediaTime()
        cb.commit()
        cb.waitUntilCompleted()
        return CACurrentMediaTime() - start
    }

    /// Expose command queue for shared use.
    public var queue: MTLCommandQueue { commandQueue }

    // MARK: - Internal

    private func encodeInternal(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        aOffset: Int, cOffset: Int,
        sliceM: Int, K: Int, N: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(A, offset: aOffset, index: 0)
        encoder.setBuffer(B, offset: 0, index: 1)
        encoder.setBuffer(C, offset: cOffset, index: 2)

        var m = UInt32(sliceM)
        var n = UInt32(N)
        var k = UInt32(K)
        encoder.setBytes(&m, length: 4, index: 3)
        encoder.setBytes(&n, length: 4, index: 4)
        encoder.setBytes(&k, length: 4, index: 5)

        // Grid: one threadgroup per 32×32 output tile
        let gridWidth = (N + Self.tileN - 1) / Self.tileN
        let gridHeight = (sliceM + Self.tileM - 1) / Self.tileM

        encoder.dispatchThreadgroups(
            MTLSize(width: gridWidth, height: gridHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerThreadgroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    // MARK: - Metal Kernel Source

    private static func generateSimdgroupKernel() -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        // Optimized matmul using simdgroup_matrix operations.
        // Each threadgroup computes a 32×32 output tile using 4 simdgroups.
        // Each simdgroup computes an 8×32 strip using 8×8 hardware multiply-accumulate.
        // No threadgroup shared memory needed — uses simdgroup registers.

        #define TILE_M 32
        #define TILE_N 32
        #define TILE_K 8

        kernel void matmul_simdgroup(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],
            device float* C       [[buffer(2)]],
            constant uint& M      [[buffer(3)]],
            constant uint& N      [[buffer(4)]],
            constant uint& K      [[buffer(5)]],
            uint2 gid             [[threadgroup_position_in_grid]],
            uint simd_lane_id     [[thread_index_in_simdgroup]],
            uint simd_group_id    [[simdgroup_index_in_threadgroup]]
        ) {
            // Each threadgroup handles a 32×32 output tile
            uint tileRowStart = gid.y * TILE_M;
            uint tileColStart = gid.x * TILE_N;

            // Each of 4 simdgroups computes 8 rows of the 32×32 tile
            uint simdRowOffset = simd_group_id * 8;

            // 4 accumulators per simdgroup: 8×8 each, covering 8 rows × 32 cols
            simdgroup_float8x8 acc0, acc1, acc2, acc3;
            acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            // Iterate over K dimension in tiles of 8
            for (uint k = 0; k < K; k += TILE_K) {
                // Load 8×8 tile from A
                simdgroup_float8x8 a_tile;
                uint aRow = tileRowStart + simdRowOffset;
                uint aCol = k;

                if (aRow + 8 <= M && aCol + 8 <= K) {
                    simdgroup_load(a_tile, A + aRow * K + aCol, K);
                } else {
                    a_tile = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    if (aRow < M && aCol < K) {
                        simdgroup_load(a_tile, A + min(aRow, M > 8 ? M - 8 : 0u) * K + min(aCol, K > 8 ? K - 8 : 0u), K);
                    }
                }

                // Load 4 8×8 tiles from B (covers 8 rows × 32 cols)
                simdgroup_float8x8 b0, b1, b2, b3;
                uint bRow = k;
                uint bCol = tileColStart;

                if (bRow + 8 <= K && bCol + 32 <= N) {
                    simdgroup_load(b0, B + bRow * N + bCol,      N);
                    simdgroup_load(b1, B + bRow * N + bCol + 8,  N);
                    simdgroup_load(b2, B + bRow * N + bCol + 16, N);
                    simdgroup_load(b3, B + bRow * N + bCol + 24, N);
                } else {
                    b0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    b1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    b2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    b3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    if (bRow < K) {
                        uint safeRow = min(bRow, K > 8 ? K - 8 : 0u);
                        simdgroup_load(b0, B + safeRow * N + min(bCol,      N > 8  ? N - 8  : 0u), N);
                        if (bCol + 8  < N) simdgroup_load(b1, B + safeRow * N + min(bCol + 8,  N > 8  ? N - 8  : 0u), N);
                        if (bCol + 16 < N) simdgroup_load(b2, B + safeRow * N + min(bCol + 16, N > 8  ? N - 8  : 0u), N);
                        if (bCol + 24 < N) simdgroup_load(b3, B + safeRow * N + min(bCol + 24, N > 8  ? N - 8  : 0u), N);
                    }
                }

                // Hardware multiply-accumulate
                simdgroup_multiply_accumulate(acc0, a_tile, b0, acc0);
                simdgroup_multiply_accumulate(acc1, a_tile, b1, acc1);
                simdgroup_multiply_accumulate(acc2, a_tile, b2, acc2);
                simdgroup_multiply_accumulate(acc3, a_tile, b3, acc3);
            }

            // Store results
            uint outRow = tileRowStart + simdRowOffset;
            uint outCol = tileColStart;

            if (outRow + 8 <= M && outCol + 32 <= N) {
                simdgroup_store(acc0, C + outRow * N + outCol,      N);
                simdgroup_store(acc1, C + outRow * N + outCol + 8,  N);
                simdgroup_store(acc2, C + outRow * N + outCol + 16, N);
                simdgroup_store(acc3, C + outRow * N + outCol + 24, N);
            } else {
                if (outRow < M && outCol < N) {
                    uint sr = min(outRow, M > 8 ? M - 8 : 0u);
                    simdgroup_store(acc0, C + sr * N + min(outCol, N > 8 ? N - 8 : 0u), N);
                }
                if (outRow < M && outCol + 8 < N) {
                    uint sr = min(outRow, M > 8 ? M - 8 : 0u);
                    simdgroup_store(acc1, C + sr * N + min(outCol + 8, N > 8 ? N - 8 : 0u), N);
                }
                if (outRow < M && outCol + 16 < N) {
                    uint sr = min(outRow, M > 8 ? M - 8 : 0u);
                    simdgroup_store(acc2, C + sr * N + min(outCol + 16, N > 8 ? N - 8 : 0u), N);
                }
                if (outRow < M && outCol + 24 < N) {
                    uint sr = min(outRow, M > 8 ? M - 8 : 0u);
                    simdgroup_store(acc3, C + sr * N + min(outCol + 24, N > 8 ? N - 8 : 0u), N);
                }
            }
        }
        """
    }
}
