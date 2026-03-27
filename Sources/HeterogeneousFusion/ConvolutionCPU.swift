// ConvolutionCPU.swift
// HeterogeneousFusion
//
// Phase 5: Output-channel-sliceable 2D convolution via Accelerate.
// NCHW layout: input [B, Cin, H, W], weights [Cout, Cin, Kh, Kw],
//              output [B, Cout, outH, outW].
//
// Uses im2col + vDSP_mmul for the inner computation, which is
// significantly faster than naive nested loops for large convolutions.

import Accelerate
import QuartzCore

/// Performs 2D convolution on CPU via Accelerate (im2col + vDSP_mmul).
/// Supports output-channel slicing for partitioned execution.
public final class ConvolutionCPU: Sendable {

    public init() {}

    /// Execute a 2D convolution (or output-channel slice) on CPU.
    ///
    /// Uses im2col to transform the convolution into a matrix multiplication:
    ///   output = weights_reshaped @ im2col_matrix
    /// where weights_reshaped is [sliceOutC, inC*kH*kW]
    /// and im2col_matrix is [inC*kH*kW, outH*outW].
    ///
    /// - Parameters:
    ///   - input: Pointer to input [B, Cin, H, W], float32 NCHW.
    ///   - weights: Pointer to weight slice [sliceOutC, Cin, Kh, Kw], float32.
    ///   - output: Pointer to output region [B, sliceOutC, outH, outW], float32.
    ///   - shape: Convolution shape parameters.
    ///   - sliceOutC: Number of output channels this unit computes.
    /// - Returns: Wall-clock time in seconds.
    public func execute(
        input: UnsafePointer<Float>,
        weights: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        shape: ConvShape, sliceOutC: Int
    ) -> Double {
        let start = CACurrentMediaTime()

        let outH = shape.outHeight
        let outW = shape.outWidth
        let patchSize = shape.inChannels * shape.kernelH * shape.kernelW
        let spatialOut = outH * outW

        // Allocate im2col buffer: [patchSize, spatialOut] per batch
        let im2colCount = patchSize * spatialOut
        let im2col = UnsafeMutablePointer<Float>.allocate(capacity: im2colCount)
        defer { im2col.deallocate() }

        let inSpatial = shape.height * shape.width
        let inBatchStride = shape.inChannels * inSpatial
        let outBatchStride = sliceOutC * spatialOut

        for b in 0..<shape.batch {
            let batchInput = input + b * inBatchStride
            let batchOutput = output + b * outBatchStride

            // Build im2col matrix
            im2colTransform(
                input: batchInput, output: im2col,
                inC: shape.inChannels, H: shape.height, W: shape.width,
                kH: shape.kernelH, kW: shape.kernelW,
                strideH: shape.strideH, strideW: shape.strideW,
                padH: shape.padH, padW: shape.padW,
                outH: outH, outW: outW
            )

            // Matmul: output = weights @ im2col
            // weights: [sliceOutC, patchSize], im2col: [patchSize, spatialOut]
            // output: [sliceOutC, spatialOut]
            vDSP_mmul(
                weights, 1,
                im2col, 1,
                batchOutput, 1,
                vDSP_Length(sliceOutC),
                vDSP_Length(spatialOut),
                vDSP_Length(patchSize)
            )
        }

        return CACurrentMediaTime() - start
    }

    /// Execute using MTLBuffer pointers (shared memory on Apple Silicon).
    public func execute(
        input: MTLBuffer, weights: MTLBuffer, output: MTLBuffer,
        weightsOffset: Int, outputOffset: Int,
        shape: ConvShape, sliceOutC: Int
    ) -> Double {
        let inPtr = input.contents().assumingMemoryBound(to: Float.self)
        let wPtr = (weights.contents() + weightsOffset).assumingMemoryBound(to: Float.self)
        let outPtr = (output.contents() + outputOffset).assumingMemoryBound(to: Float.self)
        return execute(input: inPtr, weights: wPtr, output: outPtr,
                       shape: shape, sliceOutC: sliceOutC)
    }

    // MARK: - im2col

    /// Transform input patches into column matrix for GEMM-based convolution.
    /// Output layout: [patchSize, outH * outW] where patchSize = inC * kH * kW.
    private func im2colTransform(
        input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>,
        inC: Int, H: Int, W: Int,
        kH: Int, kW: Int,
        strideH: Int, strideW: Int,
        padH: Int, padW: Int,
        outH: Int, outW: Int
    ) {
        let spatialOut = outH * outW
        var colIdx = 0

        for ic in 0..<inC {
            let channelPtr = input + ic * H * W
            for kh in 0..<kH {
                for kw in 0..<kW {
                    // This row of the im2col matrix: all spatial output positions
                    // for this (ic, kh, kw) combination
                    var outIdx = 0
                    for oh in 0..<outH {
                        let ih = oh * strideH + kh - padH
                        for ow in 0..<outW {
                            let iw = ow * strideW + kw - padW
                            if ih >= 0 && ih < H && iw >= 0 && iw < W {
                                output[colIdx * spatialOut + outIdx] = channelPtr[ih * W + iw]
                            } else {
                                output[colIdx * spatialOut + outIdx] = 0
                            }
                            outIdx += 1
                        }
                    }
                    colIdx += 1
                }
            }
        }
    }
}
