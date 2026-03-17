// DepthAttentionTests.swift
// MetalHLOCoreTests
//
// Tests for depth-wise attention kernel, pattern detection, and layer accumulation buffer.
// Validates Attention Residuals (Block AttnRes) support.

import Testing
import Metal
@testable import MetalHLOCore

@Suite("Depth Attention Tests", .serialized)
struct DepthAttentionTests {

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

    // MARK: - Kernel Registration

    @Test("Depth attention kernel is registered")
    func kernelIsRegistered() {
        let registry = MetalKernelRegistry.shared
        #expect(registry.getKernel("depth_attention") != nil)
    }

    @Test("Depth attention kernel compiles")
    func kernelCompiles() throws {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("depth_attention")!
        let pipeline = try registry.getPipeline(for: kernel, device: device)
        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
    }

    // MARK: - Kernel Correctness

    @Test("Depth attention with uniform scores produces mean of values")
    func uniformScoresProduceMean() throws {
        // When all keys are identical, all scores are equal.
        // softmax(equal scores) = uniform distribution.
        // output = mean(values) along depth dimension.

        let batchSize = 1
        let depthDim = 4
        let hiddenDim = 8

        // Query: arbitrary vector (doesn't matter since all keys are the same)
        var query = [Float](repeating: 0, count: batchSize * hiddenDim)
        for i in 0..<query.count { query[i] = Float.random(in: -1...1) }

        // Keys: all identical rows → equal dot products → uniform attention
        let keyRow = [Float](repeating: 1.0, count: hiddenDim)
        var keys = [Float]()
        for _ in 0..<depthDim { keys.append(contentsOf: keyRow) }

        // Values: distinct rows so we can verify the weighted sum
        var values = [Float]()
        for l in 0..<depthDim {
            for d in 0..<hiddenDim {
                values.append(Float(l * hiddenDim + d))
            }
        }

        // Expected: mean of values along depth = (v0 + v1 + v2 + v3) / 4
        var expected = [Float](repeating: 0, count: hiddenDim)
        for l in 0..<depthDim {
            for d in 0..<hiddenDim {
                expected[d] += values[l * hiddenDim + d]
            }
        }
        for d in 0..<hiddenDim {
            expected[d] /= Float(depthDim)
        }

        let output = try executeDepthAttention(
            query: query, keys: keys, values: values,
            batchSize: batchSize, depthDim: depthDim, hiddenDim: hiddenDim
        )

        for d in 0..<hiddenDim {
            #expect(abs(output[d] - expected[d]) < 1e-3,
                    "dim \(d): expected \(expected[d]), got \(output[d])")
        }
    }

    @Test("Depth attention with one-hot scores selects single value row")
    func oneHotScoresSelectSingleRow() throws {
        // When one key has much larger dot product than others,
        // softmax concentrates weight on that key → output ≈ that value row.

        let batchSize = 1
        let depthDim = 4
        let hiddenDim = 16

        // Query: unit vector in first dimension
        var query = [Float](repeating: 0, count: hiddenDim)
        query[0] = 10.0  // Large value to create sharp softmax

        // Keys: only row 2 has large component in first dimension
        var keys = [Float](repeating: 0, count: depthDim * hiddenDim)
        keys[2 * hiddenDim + 0] = 10.0  // Row 2 has high dot product

        // Values: distinct rows
        var values = [Float](repeating: 0, count: depthDim * hiddenDim)
        for l in 0..<depthDim {
            for d in 0..<hiddenDim {
                values[l * hiddenDim + d] = Float(l == 2 ? d + 1 : 0)
            }
        }

        let output = try executeDepthAttention(
            query: query, keys: keys, values: values,
            batchSize: batchSize, depthDim: depthDim, hiddenDim: hiddenDim,
            scale: 1.0  // No scaling to keep the sharp softmax
        )

        // Output should be approximately values[2] = [1, 2, 3, ..., 16]
        for d in 0..<hiddenDim {
            let expected = Float(d + 1)
            #expect(abs(output[d] - expected) < 0.1,
                    "dim \(d): expected ~\(expected), got \(output[d])")
        }
    }

    @Test("Depth attention with batch > 1")
    func batchedExecution() throws {
        let batchSize = 4
        let depthDim = 8
        let hiddenDim = 32
        let totalQ = batchSize * hiddenDim
        let totalKV = batchSize * depthDim * hiddenDim

        var query = [Float](repeating: 0, count: totalQ)
        var keys = [Float](repeating: 0, count: totalKV)
        var values = [Float](repeating: 0, count: totalKV)

        for i in 0..<totalQ { query[i] = Float.random(in: -0.5...0.5) }
        for i in 0..<totalKV { keys[i] = Float.random(in: -0.5...0.5) }
        for i in 0..<totalKV { values[i] = Float.random(in: -0.5...0.5) }

        let output = try executeDepthAttention(
            query: query, keys: keys, values: values,
            batchSize: batchSize, depthDim: depthDim, hiddenDim: hiddenDim
        )

        #expect(output.count == batchSize * hiddenDim)

        // Verify no NaN or Inf in any batch
        for b in 0..<batchSize {
            for d in 0..<hiddenDim {
                let val = output[b * hiddenDim + d]
                #expect(!val.isNaN, "Batch \(b) dim \(d) is NaN")
                #expect(!val.isInfinite, "Batch \(b) dim \(d) is Inf")
            }
        }
    }

    @Test("Depth attention matches reference implementation")
    func matchesReference() throws {
        let batchSize = 2
        let depthDim = 6
        let hiddenDim = 16

        let totalQ = batchSize * hiddenDim
        let totalKV = batchSize * depthDim * hiddenDim

        var query = [Float](repeating: 0, count: totalQ)
        var keys = [Float](repeating: 0, count: totalKV)
        var values = [Float](repeating: 0, count: totalKV)

        // Seeded random for reproducibility
        srand48(42)
        for i in 0..<totalQ { query[i] = Float(drand48() - 0.5) }
        for i in 0..<totalKV { keys[i] = Float(drand48() - 0.5) }
        for i in 0..<totalKV { values[i] = Float(drand48() - 0.5) }

        let scale: Float = 1.0 / sqrt(Float(hiddenDim))

        // Reference: CPU implementation
        let reference = cpuDepthAttention(
            query: query, keys: keys, values: values,
            batchSize: batchSize, depthDim: depthDim, hiddenDim: hiddenDim,
            scale: scale
        )

        // GPU implementation
        let gpuOutput = try executeDepthAttention(
            query: query, keys: keys, values: values,
            batchSize: batchSize, depthDim: depthDim, hiddenDim: hiddenDim,
            scale: scale
        )

        // Compare
        for i in 0..<reference.count {
            #expect(abs(gpuOutput[i] - reference[i]) < 1e-3,
                    "index \(i): reference=\(reference[i]), gpu=\(gpuOutput[i])")
        }
    }

    @Test("Depth attention with max depth (32 blocks)")
    func maxDepth() throws {
        let batchSize = 1
        let depthDim = 32  // Maximum supported
        let hiddenDim = 64

        let totalQ = batchSize * hiddenDim
        let totalKV = batchSize * depthDim * hiddenDim

        var query = [Float](repeating: 0, count: totalQ)
        var keys = [Float](repeating: 0, count: totalKV)
        var values = [Float](repeating: 0, count: totalKV)

        for i in 0..<totalQ { query[i] = Float.random(in: -0.5...0.5) }
        for i in 0..<totalKV { keys[i] = Float.random(in: -0.5...0.5) }
        for i in 0..<totalKV { values[i] = Float.random(in: -0.5...0.5) }

        let output = try executeDepthAttention(
            query: query, keys: keys, values: values,
            batchSize: batchSize, depthDim: depthDim, hiddenDim: hiddenDim
        )

        // Verify valid output
        var hasNonZero = false
        for val in output {
            #expect(!val.isNaN, "Output contains NaN")
            #expect(!val.isInfinite, "Output contains Inf")
            if abs(val) > 1e-10 { hasNonZero = true }
        }
        #expect(hasNonZero, "Output should have non-zero values")
    }

    // MARK: - Params Tests

    @Test("DepthAttentionParams default scale")
    func paramsDefaultScale() {
        let params = DepthAttentionParams(batchSize: 2, depthDim: 8, hiddenDim: 768)
        let expected: Float = 1.0 / sqrt(768.0)
        #expect(abs(params.scale - expected) < 1e-6)
        #expect(params.totalElements == 2 * 768)
    }

    @Test("DepthAttentionParams custom scale")
    func paramsCustomScale() {
        let params = DepthAttentionParams(batchSize: 1, depthDim: 4, hiddenDim: 64, scale: 0.5)
        #expect(params.scale == 0.5)
    }

    // MARK: - Layer Accumulation Buffer Tests

    @Test("LayerAccumulationBuffer push and stack")
    func accumulationPushAndStack() throws {
        let config = LayerAccumulationConfig(
            maxDepth: 8, snapshotShape: [1, 4], elementType: .float32
        )
        let buffer = try LayerAccumulationBuffer(config: config, device: device)

        #expect(buffer.isEmpty)
        #expect(buffer.currentDepth == 0)

        // Push two snapshots
        let snap1: [Float] = [1.0, 2.0, 3.0, 4.0]
        let snap2: [Float] = [5.0, 6.0, 7.0, 8.0]

        let buf1 = device.makeBuffer(bytes: snap1, length: 16, options: .storageModeShared)!
        let buf2 = device.makeBuffer(bytes: snap2, length: 16, options: .storageModeShared)!

        try buffer.push(metalBuffer: buf1, byteCount: 16)
        #expect(buffer.currentDepth == 1)
        #expect(!buffer.isEmpty)

        try buffer.push(metalBuffer: buf2, byteCount: 16)
        #expect(buffer.currentDepth == 2)
        #expect(buffer.stackedShape == [2, 1, 4])

        // Read stacked data
        let ptr = buffer.stackedBuffer.contents().bindMemory(to: Float.self, capacity: 8)
        let data = Array(UnsafeBufferPointer(start: ptr, count: 8))

        #expect(data[0] == 1.0)
        #expect(data[1] == 2.0)
        #expect(data[4] == 5.0)
        #expect(data[5] == 6.0)
    }

    @Test("LayerAccumulationBuffer reset")
    func accumulationReset() throws {
        let config = LayerAccumulationConfig(
            maxDepth: 4, snapshotShape: [2], elementType: .float32
        )
        let buffer = try LayerAccumulationBuffer(config: config, device: device)

        let snap: [Float] = [1.0, 2.0]
        let buf = device.makeBuffer(bytes: snap, length: 8, options: .storageModeShared)!

        try buffer.push(metalBuffer: buf, byteCount: 8)
        try buffer.push(metalBuffer: buf, byteCount: 8)
        #expect(buffer.currentDepth == 2)

        buffer.reset()
        #expect(buffer.currentDepth == 0)
        #expect(buffer.isEmpty)
    }

    @Test("LayerAccumulationBuffer rejects when full")
    func accumulationRejectsWhenFull() throws {
        let config = LayerAccumulationConfig(
            maxDepth: 2, snapshotShape: [2], elementType: .float32
        )
        let buffer = try LayerAccumulationBuffer(config: config, device: device)

        let snap: [Float] = [1.0, 2.0]
        let buf = device.makeBuffer(bytes: snap, length: 8, options: .storageModeShared)!

        try buffer.push(metalBuffer: buf, byteCount: 8)
        try buffer.push(metalBuffer: buf, byteCount: 8)
        #expect(buffer.isFull)

        #expect(throws: LayerAccumulationError.self) {
            try buffer.push(metalBuffer: buf, byteCount: 8)
        }
    }

    @Test("LayerAccumulationBuffer rejects wrong byte count")
    func accumulationRejectsWrongBytes() throws {
        let config = LayerAccumulationConfig(
            maxDepth: 4, snapshotShape: [4], elementType: .float32
        )
        let buffer = try LayerAccumulationBuffer(config: config, device: device)

        let snap: [Float] = [1.0, 2.0]
        let buf = device.makeBuffer(bytes: snap, length: 8, options: .storageModeShared)!

        #expect(throws: LayerAccumulationError.self) {
            try buffer.push(metalBuffer: buf, byteCount: 8)  // Expected 16 bytes
        }
    }

    // MARK: - Pattern Detection Tests

    @Test("DepthAttentionPattern registered with correct priority")
    func patternRegistered() {
        let patterns = HLOPatternRegistry.shared.registeredPatterns
        let depthPattern = patterns.first { $0.name == "depth_attention" }
        #expect(depthPattern != nil)
        #expect(depthPattern!.priority == 120)

        // Should be higher priority than standard attention (110)
        let attentionPattern = patterns.first { $0.name == "attention" }
        #expect(depthPattern!.priority > attentionPattern!.priority)
    }

    @Test("DepthAttentionPattern matches small-dim attention")
    func patternMatchesSmallDim() {
        // Build a synthetic function with the depth attention pattern:
        // %scores = dot_general(%query, %keys)  → resultType [1, 8]
        // %max = reduce_max(%scores)
        // %broadcast = broadcast(%max)
        // %sub = subtract(%scores, %broadcast)
        // %exp = exp(%sub)
        // %sum = reduce_sum(%exp)
        // %weights = divide(%exp, %sum)
        // %output = dot_general(%weights, %values)

        let pattern = DepthAttentionPattern()

        let qkDot = HLOOperation(
            result: "%scores",
            kind: .dotGeneral,
            operands: ["%query", "%keys"],
            resultType: TensorType(shape: [1, 8], elementType: .float32),
            attributes: HLOAttributes()
        )

        var maxAttrs = HLOAttributes()
        maxAttrs.dimensions = [-1]
        let maxOp = HLOOperation(
            result: "%max", kind: .reduce, operands: ["%scores"],
            resultType: TensorType(shape: [1], elementType: .float32),
            attributes: maxAttrs
        )

        let broadcastOp = HLOOperation(
            result: "%bmax", kind: .broadcastInDim, operands: ["%max"],
            resultType: TensorType(shape: [1, 8], elementType: .float32),
            attributes: HLOAttributes()
        )

        let subOp = HLOOperation(
            result: "%sub", kind: .subtract, operands: ["%scores", "%bmax"],
            resultType: TensorType(shape: [1, 8], elementType: .float32),
            attributes: HLOAttributes()
        )

        let expOp = HLOOperation(
            result: "%exp", kind: .exponential, operands: ["%sub"],
            resultType: TensorType(shape: [1, 8], elementType: .float32),
            attributes: HLOAttributes()
        )

        var sumAttrs = HLOAttributes()
        sumAttrs.dimensions = [-1]
        let sumOp = HLOOperation(
            result: "%sum", kind: .reduce, operands: ["%exp"],
            resultType: TensorType(shape: [1], elementType: .float32),
            attributes: sumAttrs
        )

        let divideOp = HLOOperation(
            result: "%weights", kind: .divide, operands: ["%exp", "%sum"],
            resultType: TensorType(shape: [1, 8], elementType: .float32),
            attributes: HLOAttributes()
        )

        let wvDot = HLOOperation(
            result: "%output", kind: .dotGeneral, operands: ["%weights", "%values"],
            resultType: TensorType(shape: [1, 64], elementType: .float32),
            attributes: HLOAttributes()
        )

        let ops = [qkDot, maxOp, broadcastOp, subOp, expOp, sumOp, divideOp, wvDot]
        let function = HLOFunction(
            name: "depth_attn_test",
            inputs: [
                HLOArgument(name: "%query", type: TensorType(shape: [1, 64], elementType: .float32)),
                HLOArgument(name: "%keys", type: TensorType(shape: [8, 64], elementType: .float32)),
                HLOArgument(name: "%values", type: TensorType(shape: [8, 64], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [1, 64], elementType: .float32)],
            operations: ops,
            returnValues: ["%output"]
        )

        var definingOps: [String: (op: HLOOperation, index: Int)] = [:]
        for (i, op) in ops.enumerated() {
            definingOps[op.result] = (op, i)
        }

        // Match at the final dot_general (weights @ values)
        let match = pattern.match(at: wvDot, index: 7, in: function, definingOps: definingOps)

        #expect(match != nil, "DepthAttentionPattern should match small-dim attention")
        if let m = match {
            #expect(m.inputs.count == 3)  // query, keys, values
            if case .int(let depthDim) = m.metadata["depth_dim"] {
                #expect(depthDim == 8)
            } else {
                Issue.record("Missing depth_dim metadata")
            }
        }
    }

    @Test("DepthAttentionPattern rejects large-dim attention")
    func patternRejectsLargeDim() {
        let pattern = DepthAttentionPattern()

        // Build an attention pattern where scores have shape [1, 512] — too large for depth attention
        let qkDot = HLOOperation(
            result: "%scores", kind: .dotGeneral, operands: ["%query", "%keys"],
            resultType: TensorType(shape: [1, 512], elementType: .float32),
            attributes: HLOAttributes()
        )

        var maxAttrs = HLOAttributes()
        maxAttrs.dimensions = [-1]
        let maxOp = HLOOperation(
            result: "%max", kind: .reduce, operands: ["%scores"],
            resultType: TensorType(shape: [1], elementType: .float32),
            attributes: maxAttrs
        )

        let subOp = HLOOperation(
            result: "%sub", kind: .subtract, operands: ["%scores", "%max"],
            resultType: TensorType(shape: [1, 512], elementType: .float32),
            attributes: HLOAttributes()
        )

        let expOp = HLOOperation(
            result: "%exp", kind: .exponential, operands: ["%sub"],
            resultType: TensorType(shape: [1, 512], elementType: .float32),
            attributes: HLOAttributes()
        )

        var sumAttrs = HLOAttributes()
        sumAttrs.dimensions = [-1]
        let sumOp = HLOOperation(
            result: "%sum", kind: .reduce, operands: ["%exp"],
            resultType: TensorType(shape: [1], elementType: .float32),
            attributes: sumAttrs
        )

        let divideOp = HLOOperation(
            result: "%weights", kind: .divide, operands: ["%exp", "%sum"],
            resultType: TensorType(shape: [1, 512], elementType: .float32),
            attributes: HLOAttributes()
        )

        let wvDot = HLOOperation(
            result: "%output", kind: .dotGeneral, operands: ["%weights", "%values"],
            resultType: TensorType(shape: [1, 64], elementType: .float32),
            attributes: HLOAttributes()
        )

        let ops = [qkDot, maxOp, subOp, expOp, sumOp, divideOp, wvDot]
        let function = HLOFunction(
            name: "large_attn_test",
            inputs: [
                HLOArgument(name: "%query", type: TensorType(shape: [1, 64], elementType: .float32)),
                HLOArgument(name: "%keys", type: TensorType(shape: [512, 64], elementType: .float32)),
                HLOArgument(name: "%values", type: TensorType(shape: [512, 64], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [1, 64], elementType: .float32)],
            operations: ops,
            returnValues: ["%output"]
        )

        var definingOps: [String: (op: HLOOperation, index: Int)] = [:]
        for (i, op) in ops.enumerated() {
            definingOps[op.result] = (op, i)
        }

        let match = pattern.match(at: wvDot, index: 6, in: function, definingOps: definingOps)
        #expect(match == nil, "DepthAttentionPattern should NOT match large-dim attention (512)")
    }

    // MARK: - Custom Call Handler Tests

    @Test("FusedDepthAttentionHandler is registered")
    func handlerRegistered() {
        let registry = CustomCallRegistry.shared
        #expect(registry.isSupported("fused_depth_attention"))
    }

    // MARK: - Helpers

    /// Execute depth attention on GPU and return output as Float array.
    private func executeDepthAttention(
        query: [Float], keys: [Float], values: [Float],
        batchSize: Int, depthDim: Int, hiddenDim: Int,
        scale: Float? = nil
    ) throws -> [Float] {
        let registry = MetalKernelRegistry.shared
        let kernel = registry.getKernel("depth_attention")!
        let pipeline = try registry.getPipeline(for: kernel, device: device)

        let qBuffer = device.makeBuffer(bytes: query, length: query.count * 4, options: .storageModeShared)!
        let kBuffer = device.makeBuffer(bytes: keys, length: keys.count * 4, options: .storageModeShared)!
        let vBuffer = device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared)!
        let oBuffer = device.makeBuffer(length: batchSize * hiddenDim * 4, options: .storageModeShared)!

        let params = DepthAttentionParams(
            batchSize: batchSize,
            depthDim: depthDim,
            hiddenDim: hiddenDim,
            scale: scale
        )

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        kernel.encode(into: encoder, inputs: [qBuffer, kBuffer, vBuffer], outputs: [oBuffer], params: params, pipeline: pipeline)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let ptr = oBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * hiddenDim)
        return Array(UnsafeBufferPointer(start: ptr, count: batchSize * hiddenDim))
    }

    /// Reference CPU implementation of depth attention.
    private func cpuDepthAttention(
        query: [Float], keys: [Float], values: [Float],
        batchSize: Int, depthDim: Int, hiddenDim: Int,
        scale: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: batchSize * hiddenDim)

        for b in 0..<batchSize {
            let qBase = b * hiddenDim
            let kvBase = b * depthDim * hiddenDim

            // Compute scores
            var scores = [Float](repeating: 0, count: depthDim)
            for l in 0..<depthDim {
                var dot: Float = 0
                for d in 0..<hiddenDim {
                    dot += query[qBase + d] * keys[kvBase + l * hiddenDim + d]
                }
                scores[l] = dot * scale
            }

            // Softmax
            let maxScore = scores.max()!
            var expScores = scores.map { exp($0 - maxScore) }
            let sumExp = expScores.reduce(0, +)
            for l in 0..<depthDim { expScores[l] /= sumExp }

            // Weighted sum
            for d in 0..<hiddenDim {
                var acc: Float = 0
                for l in 0..<depthDim {
                    acc += expScores[l] * values[kvBase + l * hiddenDim + d]
                }
                output[qBase + d] = acc
            }
        }

        return output
    }
}
