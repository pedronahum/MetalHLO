// UnaryKernels.swift
// MetalHLOCore
//
// Custom Metal kernels for simple unary activations: ReLU, Exp, Sigmoid, Tanh.
// All use scalar dispatch with float4 SIMD loads for bandwidth efficiency.

import Metal
import Foundation

// MARK: - ReLU Kernel

/// Custom Metal kernel for ReLU activation: max(0, x).
public struct ReLUKernel: MetalKernel, Sendable {

    public let name = "relu"
    public let metalFunctionName = "relu_kernel"

    public init() {}

    public var shaderSource: String {
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void relu_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            output[gid] = max(0.0f, input[gid]);
        }

        kernel void relu_kernel_vec4(
            device const float4* input [[buffer(0)]],
            device float4* output [[buffer(1)]],
            constant uint& vec_count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= vec_count) return;
            output[gid] = max((float4)0.0f, input[gid]);
        }

        kernel void relu_kernel_half(
            device const half* input [[buffer(0)]],
            device half* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            output[gid] = max((half)0.0f, input[gid]);
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
}

// MARK: - Exp Kernel

/// Custom Metal kernel for element-wise exponential: exp(x).
public struct ExpKernel: MetalKernel, Sendable {

    public let name = "exp_activation"
    public let metalFunctionName = "exp_kernel"

    public init() {}

    public var shaderSource: String {
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void exp_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            output[gid] = precise::exp(input[gid]);
        }

        kernel void exp_kernel_vec4(
            device const float4* input [[buffer(0)]],
            device float4* output [[buffer(1)]],
            constant uint& vec_count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= vec_count) return;
            float4 x = input[gid];
            output[gid] = float4(precise::exp(x.x), precise::exp(x.y),
                                  precise::exp(x.z), precise::exp(x.w));
        }

        kernel void exp_kernel_half(
            device const half* input [[buffer(0)]],
            device half* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            output[gid] = half(precise::exp(float(input[gid])));
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
}

// MARK: - Sigmoid Kernel

/// Custom Metal kernel for sigmoid activation: 1 / (1 + exp(-x)).
public struct SigmoidKernel: MetalKernel, Sendable {

    public let name = "sigmoid"
    public let metalFunctionName = "sigmoid_kernel"

    public init() {}

    public var shaderSource: String {
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void sigmoid_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            output[gid] = 1.0f / (1.0f + precise::exp(-input[gid]));
        }

        kernel void sigmoid_kernel_vec4(
            device const float4* input [[buffer(0)]],
            device float4* output [[buffer(1)]],
            constant uint& vec_count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= vec_count) return;
            float4 x = input[gid];
            output[gid] = float4(1.0f) / (float4(1.0f) + float4(precise::exp(-x.x),
                                                                   precise::exp(-x.y),
                                                                   precise::exp(-x.z),
                                                                   precise::exp(-x.w)));
        }

        kernel void sigmoid_kernel_half(
            device const half* input [[buffer(0)]],
            device half* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            float x = float(input[gid]);
            output[gid] = half(1.0f / (1.0f + precise::exp(-x)));
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
}

// MARK: - Tanh Kernel

/// Custom Metal kernel for hyperbolic tangent activation: tanh(x).
public struct TanhKernel: MetalKernel, Sendable {

    public let name = "tanh_activation"
    public let metalFunctionName = "tanh_kernel"

    public init() {}

    public var shaderSource: String {
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void tanh_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            output[gid] = precise::tanh(input[gid]);
        }

        kernel void tanh_kernel_vec4(
            device const float4* input [[buffer(0)]],
            device float4* output [[buffer(1)]],
            constant uint& vec_count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= vec_count) return;
            float4 x = input[gid];
            output[gid] = float4(precise::tanh(x.x), precise::tanh(x.y),
                                  precise::tanh(x.z), precise::tanh(x.w));
        }

        kernel void tanh_kernel_half(
            device const half* input [[buffer(0)]],
            device half* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count) return;
            output[gid] = half(precise::tanh(float(input[gid])));
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
}
