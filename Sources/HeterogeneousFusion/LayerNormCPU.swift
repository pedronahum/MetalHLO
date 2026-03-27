// LayerNormCPU.swift
// HeterogeneousFusion
//
// Phase 5: Row-sliceable LayerNorm via Accelerate (vDSP).
// Each row: mean → variance → normalize → scale by γ → add β.
// Uses vDSP_meanv, vDSP_measqv, vDSP_vsmul, vDSP_vmul, vDSP_vadd.

import Accelerate
import Foundation

/// Performs LayerNorm on CPU via Accelerate/vDSP.
public final class LayerNormCPU: Sendable {

    public init() {}

    /// Execute LayerNorm on contiguous float32 arrays.
    ///
    /// - Parameters:
    ///   - input: Input pointer [M, N], row-major.
    ///   - gamma: Scale parameter pointer [N].
    ///   - beta: Bias parameter pointer [N].
    ///   - output: Output pointer [M, N], row-major.
    ///   - M: Number of rows.
    ///   - N: Hidden dimension (columns per row).
    ///   - epsilon: Small constant for numerical stability.
    public func execute(
        input: UnsafePointer<Float>,
        gamma: UnsafePointer<Float>,
        beta: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        M: Int, N: Int,
        epsilon: Float
    ) {
        let vN = vDSP_Length(N)

        for row in 0..<M {
            let x = input + row * N
            let y = output + row * N

            // Mean: vDSP_meanv
            var mean: Float = 0
            vDSP_meanv(x, 1, &mean, vN)

            // Centered = x - mean: vDSP_vsadd
            var negMean = -mean
            vDSP_vsadd(x, 1, &negMean, y, 1, vN)

            // Variance: mean(x²) - mean²
            var meanSq: Float = 0
            vDSP_measqv(x, 1, &meanSq, vN)
            let variance = max(meanSq - mean * mean, 0)
            var invStd = 1.0 / sqrtf(variance + epsilon)

            // Scale centered by invStd: vDSP_vsmul (y = centered * invStd)
            vDSP_vsmul(y, 1, &invStd, y, 1, vN)

            // Multiply by gamma: vDSP_vmul (y = y * gamma)
            vDSP_vmul(y, 1, gamma, 1, y, 1, vN)

            // Add beta: vDSP_vadd (y = y + beta)
            vDSP_vadd(y, 1, beta, 1, y, 1, vN)
        }
    }
}
