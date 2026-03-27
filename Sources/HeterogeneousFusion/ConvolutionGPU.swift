// ConvolutionGPU.swift
// HeterogeneousFusion
//
// Phase 5: Output-channel-sliceable 2D convolution via Metal compute kernel.
// NCHW layout: input [B, Cin, H, W], weights [Cout, Cin, Kh, Kw],
//              output [B, Cout, outH, outW].
//
// Each thread computes one output element. Supports output-channel slicing:
// computes output[:, chanOffset:chanOffset+sliceCout, :, :].

import Metal
import QuartzCore

/// Performs 2D convolution on GPU via a custom Metal compute kernel.
/// Uses NCHW layout. Supports output-channel slicing for partitioned execution.
public final class ConvolutionGPU: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create convolution GPU command queue")
        }
        self.commandQueue = queue

        let source = Self.generateKernel()
        let library = try device.makeLibrary(source: source, options: nil)
        guard let fn = library.makeFunction(name: "conv2d_nchw") else {
            throw HeterogeneousError.metalSetupFailed("Missing kernel: conv2d_nchw")
        }
        self.pipelineState = try device.makeComputePipelineState(function: fn)
    }

    /// Expose command queue for shared use.
    public var queue: MTLCommandQueue { commandQueue }

    /// Encode a 2D convolution (or output-channel slice) into the given command buffer.
    ///
    /// - Parameters:
    ///   - input: Input buffer [B, Cin, H, W], float32 NCHW.
    ///   - weights: Full weight buffer [Cout, Cin, Kh, Kw], float32.
    ///   - output: Output buffer [B, Cout, outH, outW], float32 NCHW.
    ///   - weightsOffset: Byte offset into weights buffer for this channel slice.
    ///   - outputOffset: Byte offset into output buffer for this channel slice.
    ///   - shape: Convolution shape parameters.
    ///   - sliceOutC: Number of output channels this unit computes.
    ///   - commandBuffer: Metal command buffer to encode into.
    public func encode(
        input: MTLBuffer, weights: MTLBuffer, output: MTLBuffer,
        weightsOffset: Int, outputOffset: Int,
        shape: ConvShape, sliceOutC: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)

        // Pack parameters into a struct matching the Metal kernel
        var params = ConvParams(
            batch: UInt32(shape.batch),
            inC: UInt32(shape.inChannels),
            outC: UInt32(sliceOutC),
            H: UInt32(shape.height),
            W: UInt32(shape.width),
            outH: UInt32(shape.outHeight),
            outW: UInt32(shape.outWidth),
            kH: UInt32(shape.kernelH),
            kW: UInt32(shape.kernelW),
            strideH: UInt32(shape.strideH),
            strideW: UInt32(shape.strideW),
            padH: UInt32(shape.padH),
            padW: UInt32(shape.padW)
        )
        encoder.setBytes(&params, length: MemoryLayout<ConvParams>.size, index: 3)

        // 1 thread per output element: batch * sliceOutC * outH * outW
        let totalThreads = shape.batch * sliceOutC * shape.outHeight * shape.outWidth
        let threadsPerGroup = min(256, pipelineState.maxTotalThreadsPerThreadgroup)
        let groups = (totalThreads + threadsPerGroup - 1) / threadsPerGroup

        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Execute full convolution synchronously. Returns wall-clock time in seconds.
    public func execute(
        input: MTLBuffer, weights: MTLBuffer, output: MTLBuffer,
        shape: ConvShape
    ) -> Double {
        guard let cb = commandQueue.makeCommandBuffer() else { return 0 }
        encode(
            input: input, weights: weights, output: output,
            weightsOffset: 0, outputOffset: 0,
            shape: shape, sliceOutC: shape.outChannels,
            commandBuffer: cb
        )
        let start = CACurrentMediaTime()
        cb.commit()
        cb.waitUntilCompleted()
        return CACurrentMediaTime() - start
    }

    // MARK: - Metal Kernel Parameters (must match Swift struct)

    private struct ConvParams {
        var batch: UInt32
        var inC: UInt32
        var outC: UInt32
        var H: UInt32
        var W: UInt32
        var outH: UInt32
        var outW: UInt32
        var kH: UInt32
        var kW: UInt32
        var strideH: UInt32
        var strideW: UInt32
        var padH: UInt32
        var padW: UInt32
    }

    // MARK: - Metal Kernel Source

    private static func generateKernel() -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        struct ConvParams {
            uint batch;
            uint inC;
            uint outC;
            uint H;
            uint W;
            uint outH;
            uint outW;
            uint kH;
            uint kW;
            uint strideH;
            uint strideW;
            uint padH;
            uint padW;
        };

        // Direct 2D convolution, NCHW layout.
        // 1 thread per output element.
        // output[b][oc][oh][ow] = sum_{ic,kh,kw} input[b][ic][oh*sH+kh-padH][ow*sW+kw-padW]
        //                                         * weight[oc][ic][kh][kw]
        kernel void conv2d_nchw(
            device const float* input   [[buffer(0)]],
            device const float* weights [[buffer(1)]],
            device float*       output  [[buffer(2)]],
            constant ConvParams& p      [[buffer(3)]],
            uint gid [[thread_position_in_grid]])
        {
            uint total = p.batch * p.outC * p.outH * p.outW;
            if (gid >= total) return;

            // Decompose flat index → (b, oc, oh, ow)
            uint rem = gid;
            uint ow = rem % p.outW; rem /= p.outW;
            uint oh = rem % p.outH; rem /= p.outH;
            uint oc = rem % p.outC; rem /= p.outC;
            uint b  = rem;

            float acc = 0.0f;

            uint inSpatial = p.H * p.W;
            uint inBatchStride = p.inC * inSpatial;

            uint wFilterStride = p.inC * p.kH * p.kW;  // per output channel

            for (uint ic = 0; ic < p.inC; ic++) {
                for (uint kh = 0; kh < p.kH; kh++) {
                    int ih = int(oh * p.strideH + kh) - int(p.padH);
                    if (ih < 0 || uint(ih) >= p.H) continue;

                    for (uint kw = 0; kw < p.kW; kw++) {
                        int iw = int(ow * p.strideW + kw) - int(p.padW);
                        if (iw < 0 || uint(iw) >= p.W) continue;

                        float inVal = input[b * inBatchStride + ic * inSpatial + uint(ih) * p.W + uint(iw)];
                        float wVal  = weights[oc * wFilterStride + ic * p.kH * p.kW + kh * p.kW + kw];
                        acc += inVal * wVal;
                    }
                }
            }

            uint outSpatial = p.outH * p.outW;
            output[b * (p.outC * outSpatial) + oc * outSpatial + oh * p.outW + ow] = acc;
        }
        """
    }
}
