// ElementwiseCPU.swift
// HeterogeneousFusion
//
// Phase 5: Row-sliceable elementwise operations via Accelerate/vDSP.
// Supports add, mul, relu, gelu.

import Accelerate
import Foundation

/// Performs elementwise ops on CPU via Accelerate (vDSP).
public final class ElementwiseCPU: Sendable {

    public init() {}

    /// Execute an elementwise op on contiguous float32 arrays.
    ///
    /// - Parameters:
    ///   - op: Which operation.
    ///   - A: Input pointer.
    ///   - B: Second input pointer (for binary ops). Pass nil for unary ops.
    ///   - C: Output pointer.
    ///   - count: Number of elements.
    public func execute(
        op: ElementwiseOp,
        A: UnsafePointer<Float>,
        B: UnsafePointer<Float>?,
        C: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        let n = vDSP_Length(count)

        switch op {
        case .add:
            guard let b = B else { return }
            vDSP_vadd(A, 1, b, 1, C, 1, n)

        case .mul:
            guard let b = B else { return }
            vDSP_vmul(A, 1, b, 1, C, 1, n)

        case .relu:
            // relu(x) = max(x, 0)
            var zero: Float = 0
            vDSP_vthres(A, 1, &zero, C, 1, n)

        case .gelu:
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            // Computed element-by-element; vDSP doesn't have a native GELU
            for i in 0..<count {
                let x = A[i]
                let inner = 0.7978845608 * (x + 0.044715 * x * x * x)
                C[i] = 0.5 * x * (1.0 + tanh(inner))
            }
        }
    }
}
