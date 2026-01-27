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
        #expect(registry.getKernel("flash_attention") != nil)
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

    // MARK: - FlashAttention Kernel Tests

    @Test("FlashAttention kernel compiles")
    func flashAttentionKernelCompiles() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("flash_attention")!

        let pipeline = try registry.getPipeline(for: kernel, device: device)
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
    }

    @Test("FlashAttention params initialization")
    func flashAttentionParamsInit() {
        // Default initialization
        let params1 = FlashAttentionParams(
            batchSize: 2,
            seqLenQ: 128,
            numHeads: 8,
            headDim: 64
        )

        #expect(params1.batchSize == 2)
        #expect(params1.seqLenQ == 128)
        #expect(params1.seqLenKV == 128)  // Defaults to seqLenQ
        #expect(params1.numHeads == 8)
        #expect(params1.headDim == 64)
        #expect(abs(params1.scale - (1.0 / sqrt(64.0))) < 1e-6)  // Default scale
        #expect(!params1.isCausal)
        #expect(params1.blockSize == 64)

        // Custom initialization
        let params2 = FlashAttentionParams(
            batchSize: 4,
            seqLenQ: 256,
            seqLenKV: 512,
            numHeads: 12,
            headDim: 32,
            scale: 0.1,
            isCausal: true,
            blockSize: 32
        )

        #expect(params2.seqLenKV == 512)
        #expect(params2.scale == 0.1)
        #expect(params2.isCausal)
        #expect(params2.blockSize == 32)
    }

    @Test("FlashAttention statistics computation")
    func flashAttentionStatistics() {
        let params = FlashAttentionParams(
            batchSize: 2,
            seqLenQ: 256,
            seqLenKV: 256,
            numHeads: 8,
            headDim: 64,
            isCausal: false
        )

        let stats = params.computeStatistics()

        #expect(stats.numQBlocks == 4)  // 256 / 64 = 4
        #expect(stats.avgKVBlocksPerQ == 4.0)  // Non-causal: all 4 blocks
        #expect(stats.totalThreadgroups == 2 * 8 * 4)  // batch * heads * q_blocks
        #expect(stats.memorySavingsRatio > 1.0)  // Should save memory
    }

    @Test("FlashAttention statistics with causal mask")
    func flashAttentionStatisticsCausal() {
        let params = FlashAttentionParams(
            batchSize: 1,
            seqLenQ: 128,
            seqLenKV: 128,
            numHeads: 4,
            headDim: 64,
            isCausal: true
        )

        let stats = params.computeStatistics()

        #expect(stats.isCausal)
        #expect(stats.avgKVBlocksPerQ == 1.0)  // Causal: half the blocks on average
    }

    @Test("FlashAttention total elements calculation")
    func flashAttentionTotalElements() {
        let params = FlashAttentionParams(
            batchSize: 2,
            seqLenQ: 64,
            numHeads: 4,
            headDim: 32
        )

        #expect(params.totalElements == 2 * 4 * 64 * 32)
    }

    @Test("FlashAttention threadgroup calculation")
    func flashAttentionThreadgroups() throws {
        let kernel = FlashAttentionKernel()
        let pipeline = try MetalKernelRegistry.shared.getPipeline(for: kernel, device: device)

        let params = FlashAttentionParams(
            batchSize: 2,
            seqLenQ: 128,
            numHeads: 8,
            headDim: 64,
            blockSize: 64
        )

        let (gridSize, threadgroupSize) = kernel.calculateThreadgroups(for: params, pipeline: pipeline)

        // Simplified kernel: one thread per query position
        // Grid: (ceil(seq_q / threadgroup_width), num_heads, batch_size)
        #expect(gridSize.height == 8)  // num_heads
        #expect(gridSize.depth == 2)   // batch_size

        // Threadgroup width should be min(seqLen, maxThreads)
        #expect(threadgroupSize.width <= 256)  // Limited by max threads
        #expect(threadgroupSize.width > 0)
        #expect(threadgroupSize.height == 1)
        #expect(threadgroupSize.depth == 1)

        // Verify grid covers all query positions
        let totalQueries = gridSize.width * threadgroupSize.width
        #expect(totalQueries >= params.seqLenQ)
    }

    @Test("FlashAttention kernel produces valid output shape")
    func flashAttentionProducesValidOutput() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("flash_attention")!
        let pipeline = try registry.getPipeline(for: kernel, device: device)

        // Small test case: batch=1, heads=1, seq=64, dim=32
        let batchSize = 1
        let numHeads = 1
        let seqLen = 64
        let headDim = 32
        let totalElements = batchSize * numHeads * seqLen * headDim

        // Create random Q, K, V inputs
        var q = [Float](repeating: 0, count: totalElements)
        var k = [Float](repeating: 0, count: totalElements)
        var v = [Float](repeating: 0, count: totalElements)

        for i in 0..<totalElements {
            q[i] = Float.random(in: -1...1)
            k[i] = Float.random(in: -1...1)
            v[i] = Float.random(in: -1...1)
        }

        let qBuffer = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride, options: [])!
        let kBuffer = device.makeBuffer(bytes: k, length: k.count * MemoryLayout<Float>.stride, options: [])!
        let vBuffer = device.makeBuffer(bytes: v, length: v.count * MemoryLayout<Float>.stride, options: [])!
        let oBuffer = device.makeBuffer(length: totalElements * MemoryLayout<Float>.stride, options: [])!

        let params = FlashAttentionParams(
            batchSize: batchSize,
            seqLenQ: seqLen,
            numHeads: numHeads,
            headDim: headDim
        )

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        kernel.encode(
            into: encoder,
            inputs: [qBuffer, kBuffer, vBuffer],
            outputs: [oBuffer],
            params: params,
            pipeline: pipeline
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read output
        let outputPtr = oBuffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let output = Array(UnsafeBufferPointer(start: outputPtr, count: totalElements))

        // Verify output is not all zeros and not NaN
        var hasNonZero = false
        for val in output {
            #expect(!val.isNaN, "Output should not contain NaN")
            #expect(!val.isInfinite, "Output should not contain Inf")
            if abs(val) > 1e-10 {
                hasNonZero = true
            }
        }
        #expect(hasNonZero, "Output should have non-zero values")
    }

    @Test("FlashAttention with causal mask")
    func flashAttentionWithCausalMask() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("flash_attention")!
        let pipeline = try registry.getPipeline(for: kernel, device: device)

        let batchSize = 1
        let numHeads = 1
        let seqLen = 64
        let headDim = 32
        let totalElements = batchSize * numHeads * seqLen * headDim

        var q = [Float](repeating: 0, count: totalElements)
        var k = [Float](repeating: 0, count: totalElements)
        var v = [Float](repeating: 0, count: totalElements)

        for i in 0..<totalElements {
            q[i] = Float.random(in: -1...1)
            k[i] = Float.random(in: -1...1)
            v[i] = Float.random(in: -1...1)
        }

        let qBuffer = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride, options: [])!
        let kBuffer = device.makeBuffer(bytes: k, length: k.count * MemoryLayout<Float>.stride, options: [])!
        let vBuffer = device.makeBuffer(bytes: v, length: v.count * MemoryLayout<Float>.stride, options: [])!
        let oBuffer = device.makeBuffer(length: totalElements * MemoryLayout<Float>.stride, options: [])!

        let params = FlashAttentionParams(
            batchSize: batchSize,
            seqLenQ: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            isCausal: true
        )

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        kernel.encode(
            into: encoder,
            inputs: [qBuffer, kBuffer, vBuffer],
            outputs: [oBuffer],
            params: params,
            pipeline: pipeline
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = oBuffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let output = Array(UnsafeBufferPointer(start: outputPtr, count: totalElements))

        // Verify no NaN or Inf
        for val in output {
            #expect(!val.isNaN, "Output should not contain NaN")
            #expect(!val.isInfinite, "Output should not contain Inf")
        }
    }

    @Test("FlashAttention with multiple heads and batches")
    func flashAttentionMultiHeadBatch() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("flash_attention")!
        let pipeline = try registry.getPipeline(for: kernel, device: device)

        let batchSize = 2
        let numHeads = 4
        let seqLen = 64
        let headDim = 32
        let totalElements = batchSize * numHeads * seqLen * headDim

        var q = [Float](repeating: 0, count: totalElements)
        var k = [Float](repeating: 0, count: totalElements)
        var v = [Float](repeating: 0, count: totalElements)

        for i in 0..<totalElements {
            q[i] = Float.random(in: -0.5...0.5)
            k[i] = Float.random(in: -0.5...0.5)
            v[i] = Float.random(in: -0.5...0.5)
        }

        let qBuffer = device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride, options: [])!
        let kBuffer = device.makeBuffer(bytes: k, length: k.count * MemoryLayout<Float>.stride, options: [])!
        let vBuffer = device.makeBuffer(bytes: v, length: v.count * MemoryLayout<Float>.stride, options: [])!
        let oBuffer = device.makeBuffer(length: totalElements * MemoryLayout<Float>.stride, options: [])!

        let params = FlashAttentionParams(
            batchSize: batchSize,
            seqLenQ: seqLen,
            numHeads: numHeads,
            headDim: headDim
        )

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        kernel.encode(
            into: encoder,
            inputs: [qBuffer, kBuffer, vBuffer],
            outputs: [oBuffer],
            params: params,
            pipeline: pipeline
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPtr = oBuffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let output = Array(UnsafeBufferPointer(start: outputPtr, count: totalElements))

        // Verify all batches and heads produce valid output
        for b in 0..<batchSize {
            for h in 0..<numHeads {
                let offset = (b * numHeads + h) * seqLen * headDim
                var hasNonZero = false
                for i in 0..<(seqLen * headDim) {
                    let val = output[offset + i]
                    #expect(!val.isNaN, "Batch \(b) head \(h) should not contain NaN")
                    #expect(!val.isInfinite, "Batch \(b) head \(h) should not contain Inf")
                    if abs(val) > 1e-10 {
                        hasNonZero = true
                    }
                }
                #expect(hasNonZero, "Batch \(b) head \(h) should have non-zero values")
            }
        }
    }

    @Test("FlashAttention estimated FLOPs")
    func flashAttentionEstimatedFLOPs() {
        let params = FlashAttentionParams(
            batchSize: 1,
            seqLenQ: 1024,
            numHeads: 8,
            headDim: 64
        )

        let stats = params.computeStatistics()
        let flops = stats.estimatedFLOPs

        // QK^T: batch * heads * seq_q * seq_kv * head_dim * 2
        // = 1 * 8 * 1024 * 1024 * 64 * 2 = 1,073,741,824
        let expectedQKFlops = 1 * 8 * 1024 * 1024 * 64 * 2

        // Softmax: batch * heads * seq_q * seq_kv * 5
        // = 1 * 8 * 1024 * 1024 * 5 = 41,943,040
        let expectedSoftmaxFlops = 1 * 8 * 1024 * 1024 * 5

        // AV: batch * heads * seq_q * head_dim * seq_kv * 2
        // = 1 * 8 * 1024 * 64 * 1024 * 2 = 1,073,741,824
        let expectedAVFlops = 1 * 8 * 1024 * 64 * 1024 * 2

        let expectedTotal = expectedQKFlops + expectedSoftmaxFlops + expectedAVFlops
        #expect(flops == expectedTotal)
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
