// GELUKernel.swift
// MetalHLOCore
//
// Custom Metal kernel for GELU activation.

import Metal
import Foundation

/// Custom Metal kernel for GELU (Gaussian Error Linear Unit) activation.
///
/// Implements GELU using the fast approximation:
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// This is the same approximation used by PyTorch and TensorFlow.
public struct GELUKernel: MetalKernel, Sendable {

    public let name = "gelu"
    public let metalFunctionName = "gelu_kernel"

    public init() {}

    public var shaderSource: String {
        """
        #include <metal_stdlib>
        using namespace metal;

        // GELU constants
        constant float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
        constant float GELU_COEF = 0.044715f;

        // Element-wise GELU kernel (float)
        kernel void gelu_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;

            float x = input[gid];
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float x_cubed = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
            output[gid] = 0.5f * x * (1.0f + tanh(inner));
        }

        // Vectorized GELU kernel (float4) for better memory throughput
        kernel void gelu_kernel_vec4(
            device const float4* input [[buffer(0)]],
            device float4* output [[buffer(1)]],
            constant uint& vec_count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= vec_count) return;

            float4 x = input[gid];
            float4 x_cubed = x * x * x;
            float4 inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
            output[gid] = 0.5f * x * (1.0f + tanh(inner));
        }

        // Half-precision GELU kernel
        kernel void gelu_kernel_half(
            device const half* input [[buffer(0)]],
            device half* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;

            float x = float(input[gid]);
            float x_cubed = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
            output[gid] = half(0.5f * x * (1.0f + tanh(inner)));
        }

        // Fused GELU + scale (common in transformers after linear projection)
        kernel void gelu_scale_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            constant float& scale [[buffer(3)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;

            float x = input[gid];
            float x_cubed = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
            output[gid] = scale * 0.5f * x * (1.0f + tanh(inner));
        }
        """
    }

    public func encode(
        into encoder: MTLComputeCommandEncoder,
        inputs: [MTLBuffer],
        outputs: [MTLBuffer],
        params: KernelParams,
        pipeline: MTLComputePipelineState
    ) {
        var count = UInt32(params.totalElements)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs[0], offset: 0, index: 0)
        encoder.setBuffer(outputs[0], offset: 0, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)

        let (gridSize, threadgroupSize) = calculateThreadgroups(for: params, pipeline: pipeline)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    public func calculateThreadgroups(
        for params: KernelParams,
        pipeline: MTLComputePipelineState
    ) -> (gridSize: MTLSize, threadgroupSize: MTLSize) {
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let threadgroupSize = MTLSize(width: min(256, maxThreads), height: 1, depth: 1)
        let numThreadgroups = (params.totalElements + threadgroupSize.width - 1) / threadgroupSize.width
        let gridSize = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        return (gridSize, threadgroupSize)
    }
}
