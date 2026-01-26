// MetalKernelTests.swift
// MetalHLOCoreTests
//
// Tests for custom Metal compute kernels.

import Testing
import Metal
@testable import MetalHLOCore

@Suite("Metal Kernel Tests")
struct MetalKernelTests {

    // MARK: - Setup

    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.noMetalDevice
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
    }

    // MARK: - Registry Tests

    @Test("Kernel registry has built-in kernels")
    func registryHasBuiltInKernels() {
        let registry = MetalKernelRegistry.shared

        #expect(registry.getKernel("softmax") != nil)
        #expect(registry.getKernel("gelu") != nil)
        #expect(registry.getKernel("layer_norm") != nil)
        #expect(registry.getKernel("rms_norm") != nil)
    }

    @Test("Kernel registry returns nil for unknown kernel")
    func registryReturnsNilForUnknown() {
        let registry = MetalKernelRegistry.shared
        #expect(registry.getKernel("unknown_kernel") == nil)
    }

    // MARK: - Softmax Kernel Tests

    @Test("Softmax kernel compiles")
    func softmaxKernelCompiles() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("softmax")!

        let pipeline = try registry.getPipeline(for: kernel, device: device)
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
    }

    @Test("Softmax kernel produces valid output")
    func softmaxKernelProducesValidOutput() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("softmax")!
        let pipeline = try registry.getPipeline(for: kernel, device: device)

        // Create input: 2 rows of 4 elements
        let input: [Float] = [1.0, 2.0, 3.0, 4.0,
                              0.0, 0.0, 0.0, 0.0]
        let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride, options: [])!
        let outputBuffer = device.makeBuffer(length: input.count * MemoryLayout<Float>.stride, options: [])!

        let params = SoftmaxParams(batchSize: 2, seqLen: 4)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        kernel.encode(
            into: encoder,
            inputs: [inputBuffer],
            outputs: [outputBuffer],
            params: params,
            pipeline: pipeline
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results
        let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let output = Array(UnsafeBufferPointer(start: outputPtr, count: input.count))

        // Verify first row sums to 1
        let row1Sum = output[0...3].reduce(0, +)
        #expect(abs(row1Sum - 1.0) < 1e-5)

        // Verify second row (all zeros -> uniform distribution)
        let row2Sum = output[4...7].reduce(0, +)
        #expect(abs(row2Sum - 1.0) < 1e-5)
        #expect(abs(output[4] - 0.25) < 1e-5)  // Should be uniform

        // Verify monotonicity in first row (larger input -> larger output)
        #expect(output[3] > output[2])
        #expect(output[2] > output[1])
        #expect(output[1] > output[0])
    }

    // MARK: - GELU Kernel Tests

    @Test("GELU kernel compiles")
    func geluKernelCompiles() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("gelu")!

        let pipeline = try registry.getPipeline(for: kernel, device: device)
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
    }

    @Test("GELU kernel produces correct output")
    func geluKernelProducesCorrectOutput() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("gelu")!
        let pipeline = try registry.getPipeline(for: kernel, device: device)

        // Test values with known GELU outputs
        let input: [Float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
        let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride, options: [])!
        let outputBuffer = device.makeBuffer(length: input.count * MemoryLayout<Float>.stride, options: [])!

        let params = ElementwiseParams(totalElements: input.count)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        kernel.encode(
            into: encoder,
            inputs: [inputBuffer],
            outputs: [outputBuffer],
            params: params,
            pipeline: pipeline
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let output = Array(UnsafeBufferPointer(start: outputPtr, count: input.count))

        // GELU(0) = 0
        #expect(abs(output[2]) < 1e-5)

        // GELU is approximately 0 for large negative values
        #expect(output[0] < 0.1 && output[0] > -0.1)

        // GELU is approximately x for large positive values
        #expect(abs(output[4] - 2.0) < 0.1)

        // GELU is monotonic for positive values (not globally monotonic due to minimum at ~-0.58)
        // Check output[2] (x=0) through output[4] (x=2) are monotonically increasing
        for i in 2..<output.count - 1 {
            #expect(output[i + 1] >= output[i], "GELU should be monotonic for positive values")
        }
    }

    // MARK: - LayerNorm Kernel Tests

    @Test("LayerNorm kernel compiles")
    func layerNormKernelCompiles() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("layer_norm")!

        let pipeline = try registry.getPipeline(for: kernel, device: device)
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
    }

    @Test("LayerNorm kernel produces normalized output")
    func layerNormProducesNormalizedOutput() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("layer_norm")!
        let pipeline = try registry.getPipeline(
            for: kernel,
            device: device,
            specialization: KernelSpecialization(dataType: "float", options: ["no_affine": 1])
        )

        // Use the no_affine version function name
        let noAffineKernel = LayerNormNoAffineKernel()
        let noAffinePipeline = try MetalKernelRegistry.shared.getPipeline(for: noAffineKernel, device: device)

        // Create input: 2 rows of 4 elements
        let input: [Float] = [1.0, 2.0, 3.0, 4.0,  // mean=2.5, var=1.25
                              10.0, 20.0, 30.0, 40.0]  // mean=25, var=125
        let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride, options: [])!
        let outputBuffer = device.makeBuffer(length: input.count * MemoryLayout<Float>.stride, options: [])!

        let params = NormParams(batchSize: 2, hiddenSize: 4, epsilon: 1e-5)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        noAffineKernel.encode(
            into: encoder,
            inputs: [inputBuffer],
            outputs: [outputBuffer],
            params: params,
            pipeline: noAffinePipeline
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let output = Array(UnsafeBufferPointer(start: outputPtr, count: input.count))

        // For each row, check that mean ≈ 0 and std ≈ 1
        for row in 0..<2 {
            let rowStart = row * 4
            let rowEnd = rowStart + 4
            let rowValues = Array(output[rowStart..<rowEnd])

            let mean = rowValues.reduce(0, +) / Float(rowValues.count)
            let variance = rowValues.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(rowValues.count)
            let std = sqrt(variance)

            #expect(abs(mean) < 1e-4, "Row \(row) mean should be ~0, got \(mean)")
            #expect(abs(std - 1.0) < 1e-4, "Row \(row) std should be ~1, got \(std)")
        }
    }

    // MARK: - RMSNorm Kernel Tests

    @Test("RMSNorm kernel compiles")
    func rmsNormKernelCompiles() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("rms_norm")!

        let pipeline = try registry.getPipeline(for: kernel, device: device)
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
    }

    @Test("RMSNorm kernel produces valid output")
    func rmsNormProducesValidOutput() throws {
        let noGammaKernel = RMSNormNoGammaKernel()
        let pipeline = try MetalKernelRegistry.shared.getPipeline(for: noGammaKernel, device: device)

        // Create input
        let input: [Float] = [3.0, 4.0, 0.0, 0.0]  // RMS = sqrt((9+16)/4) = sqrt(6.25) = 2.5
        let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride, options: [])!
        let outputBuffer = device.makeBuffer(length: input.count * MemoryLayout<Float>.stride, options: [])!

        let params = NormParams(batchSize: 1, hiddenSize: 4, epsilon: 1e-5)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        noGammaKernel.encode(
            into: encoder,
            inputs: [inputBuffer],
            outputs: [outputBuffer],
            params: params,
            pipeline: pipeline
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let output = Array(UnsafeBufferPointer(start: outputPtr, count: input.count))

        // Expected: x / rms = [3/2.5, 4/2.5, 0, 0] = [1.2, 1.6, 0, 0]
        #expect(abs(output[0] - 1.2) < 1e-4)
        #expect(abs(output[1] - 1.6) < 1e-4)
        #expect(abs(output[2]) < 1e-5)
        #expect(abs(output[3]) < 1e-5)
    }

    // MARK: - Performance Tests

    @Test("Softmax performance on large input")
    func softmaxPerformance() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("softmax")!
        let pipeline = try registry.getPipeline(for: kernel, device: device)

        // Large input: 1024 rows of 512 elements
        let batchSize = 1024
        let seqLen = 512
        let totalElements = batchSize * seqLen

        var input = [Float](repeating: 0, count: totalElements)
        for i in 0..<totalElements {
            input[i] = Float.random(in: -10...10)
        }

        let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride, options: [])!
        let outputBuffer = device.makeBuffer(length: input.count * MemoryLayout<Float>.stride, options: [])!

        let params = SoftmaxParams(batchSize: batchSize, seqLen: seqLen)

        // Warmup
        for _ in 0..<5 {
            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            kernel.encode(into: encoder, inputs: [inputBuffer], outputs: [outputBuffer], params: params, pipeline: pipeline)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        // Timed run
        let iterations = 100
        let start = CFAbsoluteTimeGetCurrent()

        for _ in 0..<iterations {
            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            kernel.encode(into: encoder, inputs: [inputBuffer], outputs: [outputBuffer], params: params, pipeline: pipeline)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgTimeMs = (elapsed / Double(iterations)) * 1000

        print("Softmax [\(batchSize)x\(seqLen)]: \(String(format: "%.3f", avgTimeMs)) ms/iteration")

        // Should be reasonably fast (< 10ms for this size on M1+)
        #expect(avgTimeMs < 50, "Softmax took too long: \(avgTimeMs) ms")
    }
}

// MARK: - Helper Kernels for Testing

/// LayerNorm without affine transform (for testing)
struct LayerNormNoAffineKernel: MetalKernel, Sendable {
    let name = "layer_norm_no_affine"
    let metalFunctionName = "layer_norm_no_affine_kernel"

    var shaderSource: String {
        LayerNormKernel().shaderSource
    }

    func encode(
        into encoder: MTLComputeCommandEncoder,
        inputs: [MTLBuffer],
        outputs: [MTLBuffer],
        params: KernelParams,
        pipeline: MTLComputePipelineState
    ) {
        guard let normParams = params as? NormParams else {
            fatalError("Requires NormParams")
        }

        var gpuParams = GPUNormParamsTest(
            batch_size: UInt32(normParams.batchSize),
            hidden_size: UInt32(normParams.hiddenSize),
            epsilon: normParams.epsilon
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs[0], offset: 0, index: 0)
        encoder.setBuffer(outputs[0], offset: 0, index: 1)
        encoder.setBytes(&gpuParams, length: MemoryLayout<GPUNormParamsTest>.size, index: 2)

        let (gridSize, threadgroupSize) = calculateThreadgroups(for: params, pipeline: pipeline)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    func calculateThreadgroups(
        for params: KernelParams,
        pipeline: MTLComputePipelineState
    ) -> (gridSize: MTLSize, threadgroupSize: MTLSize) {
        guard let normParams = params as? NormParams else {
            fatalError("Requires NormParams")
        }

        let maxThreads = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadsPerRow = min(maxThreads, normParams.hiddenSize)
        let roundedThreads = 1 << Int(ceil(log2(Double(max(1, threadsPerRow)))))
        let finalThreads = min(roundedThreads, maxThreads)

        return (MTLSize(width: 1, height: normParams.batchSize, depth: 1),
                MTLSize(width: finalThreads, height: 1, depth: 1))
    }
}

/// RMSNorm without gamma (for testing)
struct RMSNormNoGammaKernel: MetalKernel, Sendable {
    let name = "rms_norm_no_gamma"
    let metalFunctionName = "rms_norm_no_gamma_kernel"

    var shaderSource: String {
        RMSNormKernel().shaderSource
    }

    func encode(
        into encoder: MTLComputeCommandEncoder,
        inputs: [MTLBuffer],
        outputs: [MTLBuffer],
        params: KernelParams,
        pipeline: MTLComputePipelineState
    ) {
        guard let normParams = params as? NormParams else {
            fatalError("Requires NormParams")
        }

        var gpuParams = GPUNormParamsTest(
            batch_size: UInt32(normParams.batchSize),
            hidden_size: UInt32(normParams.hiddenSize),
            epsilon: normParams.epsilon
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs[0], offset: 0, index: 0)
        encoder.setBuffer(outputs[0], offset: 0, index: 1)
        encoder.setBytes(&gpuParams, length: MemoryLayout<GPUNormParamsTest>.size, index: 2)

        let (gridSize, threadgroupSize) = calculateThreadgroups(for: params, pipeline: pipeline)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    func calculateThreadgroups(
        for params: KernelParams,
        pipeline: MTLComputePipelineState
    ) -> (gridSize: MTLSize, threadgroupSize: MTLSize) {
        guard let normParams = params as? NormParams else {
            fatalError("Requires NormParams")
        }

        let maxThreads = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadsPerRow = min(maxThreads, normParams.hiddenSize)
        let roundedThreads = 1 << Int(ceil(log2(Double(max(1, threadsPerRow)))))
        let finalThreads = min(roundedThreads, maxThreads)

        return (MTLSize(width: 1, height: normParams.batchSize, depth: 1),
                MTLSize(width: finalThreads, height: 1, depth: 1))
    }
}

// GPU params struct for tests
private struct GPUNormParamsTest {
    var batch_size: UInt32
    var hidden_size: UInt32
    var epsilon: Float
}

// MARK: - Test Errors

enum TestError: Error {
    case noMetalDevice
}
