// SoftmaxCPU.swift
// HeterogeneousFusion
//
// CPU softmax using Accelerate framework.
// Row-wise softmax with optional scaling.

import Accelerate

/// CPU softmax using vDSP.
public final class SoftmaxCPU: Sendable {

    public init() {}

    /// Compute row-wise softmax in-place on a [rows, cols] float32 matrix.
    /// Optionally scales all values by `scale` before softmax.
    public func execute(
        data: UnsafeMutablePointer<Float>,
        rows: Int, cols: Int, scale: Float = 1.0
    ) {
        let n = vDSP_Length(cols)

        for r in 0..<rows {
            let row = data + r * cols

            // Scale
            if scale != 1.0 {
                var s = scale
                vDSP_vsmul(row, 1, &s, row, 1, n)
            }

            // Find max
            var maxVal: Float = 0
            vDSP_maxv(row, 1, &maxVal, n)

            // Subtract max
            var negMax = -maxVal
            vDSP_vsadd(row, 1, &negMax, row, 1, n)

            // Exponentiate
            var count = Int32(cols)
            vvexpf(row, row, &count)

            // Sum
            var sum: Float = 0
            vDSP_sve(row, 1, &sum, n)

            // Normalize
            var invSum = 1.0 / sum
            vDSP_vsmul(row, 1, &invSum, row, 1, n)
        }
    }
}
