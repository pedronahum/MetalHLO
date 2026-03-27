// MatmulCPU.swift
// HeterogeneousFusion
//
// Row-sliceable matrix multiplication via Accelerate (cblas_sgemm).

import Accelerate
import Metal
import QuartzCore

/// Performs matmul on CPU via Accelerate's BLAS.
/// Supports row-slicing: computes C_slice = A_slice @ B.
public final class MatmulCPU: Sendable {

    public init() {}

    /// Execute a row-sliced matmul synchronously on CPU.
    ///
    /// A_slice and B must be contiguous float32 arrays.
    /// C_slice will be written with the result.
    ///
    /// - Parameters:
    ///   - A: Pointer to A slice data [sliceM, K], float32.
    ///   - B: Pointer to B data [K, N], float32.
    ///   - C: Pointer to output [sliceM, N], float32.
    ///   - sliceM: Number of rows in the A slice.
    ///   - K: Inner dimension.
    ///   - N: Columns of B / output.
    /// - Returns: Wall-clock time in seconds.
    public func execute(
        A: UnsafePointer<Float>,
        B: UnsafePointer<Float>,
        C: UnsafeMutablePointer<Float>,
        sliceM: Int, K: Int, N: Int
    ) -> Double {
        let start = CACurrentMediaTime()

        // C = alpha * A @ B + beta * C
        // A is sliceM×K, B is K×N, C is sliceM×N
        // Use vDSP for matrix multiply (avoids deprecated cblas_sgemm warning)
        vDSP_mmul(
            A, 1,                 // A, stride
            B, 1,                 // B, stride
            C, 1,                 // C, stride
            vDSP_Length(sliceM),  // M
            vDSP_Length(N),       // N
            vDSP_Length(K)        // K
        )

        return CACurrentMediaTime() - start
    }

    /// Execute using MTLBuffer pointers (shared memory on Apple Silicon).
    /// - Returns: Wall-clock time in seconds.
    public func execute(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        sliceM: Int, K: Int, N: Int
    ) -> Double {
        let aPtr = A.contents().assumingMemoryBound(to: Float.self)
        let bPtr = B.contents().assumingMemoryBound(to: Float.self)
        let cPtr = C.contents().assumingMemoryBound(to: Float.self)
        return execute(A: aPtr, B: bPtr, C: cPtr, sliceM: sliceM, K: K, N: N)
    }
}
