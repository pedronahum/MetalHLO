// MetalKernel.swift
// MetalHLOCore
//
// Protocol and types for custom Metal compute kernels.

import Metal
import Foundation

/// Parameters passed to kernel dispatch.
public protocol KernelParams: Sendable {
    /// Returns the total number of elements to process.
    var totalElements: Int { get }
}

/// Specialization constants for kernel compilation.
public struct KernelSpecialization: Hashable, Sendable {
    /// Data type (e.g., "float", "half").
    public let dataType: String

    /// Additional specialization options.
    public let options: [String: Int]

    public init(dataType: String = "float", options: [String: Int] = [:]) {
        self.dataType = dataType
        self.options = options
    }

    /// Returns a unique key for caching compiled pipelines.
    public var key: String {
        var parts = [dataType]
        for (k, v) in options.sorted(by: { $0.key < $1.key }) {
            parts.append("\(k)=\(v)")
        }
        return parts.joined(separator: "_")
    }

    /// Converts to Metal function constants.
    public func toFunctionConstants() -> MTLFunctionConstantValues {
        let constants = MTLFunctionConstantValues()
        // Index 0: data type (0 = float, 1 = half)
        var typeIndex: UInt32 = dataType == "half" ? 1 : 0
        constants.setConstantValue(&typeIndex, type: .uint, index: 0)

        // Additional constants starting at index 1
        var index: Int = 1
        for (_, value) in options.sorted(by: { $0.key < $1.key }) {
            var v = UInt32(value)
            constants.setConstantValue(&v, type: .uint, index: index)
            index += 1
        }

        return constants
    }
}

/// Protocol for custom Metal compute kernels.
///
/// Implementations provide the kernel name, Metal function name,
/// and encoding logic for dispatching the kernel.
public protocol MetalKernel: Sendable {
    /// Unique kernel identifier.
    var name: String { get }

    /// Name of the Metal function in the shader.
    var metalFunctionName: String { get }

    /// Source code of the Metal shader (for runtime compilation).
    var shaderSource: String { get }

    /// Encodes the kernel dispatch into a command encoder.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder.
    ///   - inputs: Input Metal buffers.
    ///   - outputs: Output Metal buffers.
    ///   - params: Kernel-specific parameters.
    ///   - pipeline: The compiled compute pipeline state.
    func encode(
        into encoder: MTLComputeCommandEncoder,
        inputs: [MTLBuffer],
        outputs: [MTLBuffer],
        params: KernelParams,
        pipeline: MTLComputePipelineState
    )

    /// Calculates threadgroup and grid sizes for dispatch.
    ///
    /// - Parameters:
    ///   - params: Kernel parameters.
    ///   - pipeline: The compiled pipeline (for max threads info).
    /// - Returns: Grid size and threadgroup size.
    func calculateThreadgroups(
        for params: KernelParams,
        pipeline: MTLComputePipelineState
    ) -> (gridSize: MTLSize, threadgroupSize: MTLSize)
}

/// Default implementations for MetalKernel.
extension MetalKernel {
    /// Default threadgroup calculation using 1D dispatch.
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

/// Simple parameters for element-wise kernels.
public struct ElementwiseParams: KernelParams, Sendable {
    public let totalElements: Int
    public let shape: [Int]

    public init(totalElements: Int, shape: [Int] = []) {
        self.totalElements = totalElements
        self.shape = shape
    }
}

/// Parameters for softmax kernel.
public struct SoftmaxParams: KernelParams, Sendable {
    public let batchSize: Int
    public let seqLen: Int
    public let axis: Int

    public var totalElements: Int { batchSize * seqLen }

    public init(batchSize: Int, seqLen: Int, axis: Int = -1) {
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.axis = axis
    }
}

/// Parameters for GEMM (General Matrix Multiply) kernel.
public struct GEMMParams: KernelParams, Sendable {
    public let M: Int  // Rows of A / output
    public let N: Int  // Columns of B / output
    public let K: Int  // Columns of A / rows of B
    public let alpha: Float
    public let beta: Float
    public let transA: Bool
    public let transB: Bool

    public var totalElements: Int { M * N }

    public init(M: Int, N: Int, K: Int, alpha: Float = 1.0, beta: Float = 0.0,
                transA: Bool = false, transB: Bool = false) {
        self.M = M
        self.N = N
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB
    }
}

/// Parameters for attention kernel.
public struct AttentionParams: KernelParams, Sendable {
    public let batchSize: Int
    public let seqLen: Int
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float
    public let isCausal: Bool
    public let hasMask: Bool

    public var totalElements: Int { batchSize * numHeads * seqLen * headDim }

    public init(batchSize: Int, seqLen: Int, numHeads: Int, headDim: Int,
                scale: Float? = nil, isCausal: Bool = false, hasMask: Bool = false) {
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = scale ?? (1.0 / sqrt(Float(headDim)))
        self.isCausal = isCausal
        self.hasMask = hasMask
    }
}
