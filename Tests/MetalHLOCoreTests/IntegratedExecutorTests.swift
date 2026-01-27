// IntegratedExecutorTests.swift
// MetalHLOCoreTests
//
// Tests for the IntegratedExecutor class.

import Testing
import Metal
@testable import MetalHLOCore

// MARK: - Integrated Executor Tests

@Suite("Integrated Executor Tests")
struct IntegratedExecutorTests {

    // MARK: - Config Tests

    @Test("Config has default values")
    func configDefaults() {
        let config = IntegratedExecutor.Config.default

        #expect(config.enableProfiling == false)
        #expect(config.synchronous == true)
        #expect(config.validateInputs == true)
        #expect(config.debugLabel == nil)
    }

    @Test("Config presets have expected values")
    func configPresets() {
        let profiling = IntegratedExecutor.Config.profiling
        #expect(profiling.enableProfiling == true)

        let async = IntegratedExecutor.Config.async
        #expect(async.synchronous == false)
    }

    @Test("Config can be customized")
    func configCustomization() {
        let config = IntegratedExecutor.Config(
            enableProfiling: true,
            synchronous: false,
            validateInputs: false,
            debugLabel: "test"
        )

        #expect(config.enableProfiling == true)
        #expect(config.synchronous == false)
        #expect(config.validateInputs == false)
        #expect(config.debugLabel == "test")
    }

    // MARK: - Memory Usage Tests

    @Test("MemoryUsage calculates total correctly")
    func memoryUsageTotal() {
        let usage = IntegratedExecutor.MemoryUsage(
            unifiedBufferBytes: 1024,
            constantBufferBytes: 256,
            peakMemoryBytes: 2048
        )

        #expect(usage.totalBytes == 1280)  // 1024 + 256
        #expect(usage.peakMemoryBytes == 2048)
    }

    // MARK: - Execution Statistics Tests

    @Test("ExecutionStatistics initializes with zeros")
    func statisticsInit() {
        let stats = ExecutionStatistics()

        #expect(stats.executionCount == 0)
        #expect(stats.totalExecutionTimeMs == 0)
        #expect(stats.lastExecutionTimeMs == 0)
        #expect(stats.averageExecutionTimeMs == 0)
    }

    @Test("ExecutionStatistics calculates average correctly")
    func statisticsAverage() {
        var stats = ExecutionStatistics()
        stats.executionCount = 4
        stats.totalExecutionTimeMs = 100

        #expect(stats.averageExecutionTimeMs == 25)
    }

    @Test("ExecutionStatistics handles zero executions")
    func statisticsZeroExecutions() {
        let stats = ExecutionStatistics()

        // Should not crash with zero division
        #expect(stats.averageExecutionTimeMs == 0)
    }

    // MARK: - Error Tests

    @Test("IntegratedExecutorError cases exist")
    func errorCases() {
        let errors: [IntegratedExecutorError] = [
            .commandBufferCreationFailed,
            .encoderCreationFailed,
            .missingPipeline("test"),
            .missingDispatch("test"),
            .missingBindings("test"),
            .missingInput("test"),
            .missingConstant("test"),
            .invalidBinding("test"),
            .executionFailed("test")
        ]

        #expect(errors.count == 9)
    }

    // MARK: - Batch Executor Config Tests

    @Test("BatchExecutor.Config has defaults")
    func batchConfigDefaults() {
        let config = BatchExecutor.Config()

        #expect(config.bufferCount == 3)
        #expect(config.maxBatchSize == 32)
    }

    @Test("BatchExecutor.Config can be customized")
    func batchConfigCustomization() {
        let config = BatchExecutor.Config(
            bufferCount: 2,
            maxBatchSize: 64
        )

        #expect(config.bufferCount == 2)
        #expect(config.maxBatchSize == 64)
    }
}

// MARK: - Execution Result Tests

@Suite("Execution Result Tests")
struct ExecutionResultTests {

    @Test("ExecutionResult stores data correctly")
    func storesData() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let buffer = device.makeBuffer(length: 256, options: .storageModeShared)!
        let outputs: [String: MTLBuffer] = ["output": buffer]
        let timings: [OpID: Double] = ["op1": 1.5, "op2": 2.5]

        let result = ExecutionResult(
            outputs: outputs,
            executionTimeMs: 10.5,
            kernelTimings: timings
        )

        #expect(result.outputs["output"] != nil)
        #expect(result.executionTimeMs == 10.5)
        #expect(result.kernelTimings?["op1"] == 1.5)
        #expect(result.kernelTimings?["op2"] == 2.5)
    }

    @Test("ExecutionResult handles nil kernel timings")
    func handlesNilTimings() {
        let result = ExecutionResult(
            outputs: [:],
            executionTimeMs: 5.0,
            kernelTimings: nil
        )

        #expect(result.kernelTimings == nil)
    }
}

// MARK: - Integration Types Tests

@Suite("Integration Types Tests")
struct IntegrationTypesTests {

    @Test("TensorInfo stores all properties")
    func tensorInfo() {
        let info = TensorInfo(
            id: "tensor1",
            shape: [2, 3, 4],
            elementType: .float32
        )

        #expect(info.id == "tensor1")
        #expect(info.shape == [2, 3, 4])
        #expect(info.elementType == .float32)
        #expect(info.byteSize == 2 * 3 * 4 * 4)  // 96 floats * 4 bytes
    }

    @Test("TensorParam stores all properties")
    func tensorParam() {
        let param = TensorParam(
            name: "input",
            type: TensorType(shape: [4, 5], elementType: .float16)
        )

        #expect(param.name == "input")
        #expect(param.type.shape == [4, 5])
        #expect(param.type.elementType == .float16)
        #expect(param.id == "input")
    }

    @Test("Lifetime stores defined and lastUsed")
    func lifetime() {
        let lifetime = Lifetime(defined: 5, lastUsed: 10)

        #expect(lifetime.defined == 5)
        #expect(lifetime.lastUsed == 10)
    }

    @Test("PatternType has all expected cases")
    func patternType() {
        let types: [PatternType] = [
            .attention,
            .multiHeadAttention,
            .layerNorm,
            .rmsNorm,
            .gelu,
            .silu,
            .ffn,
            .matmulBiasActivation,
            .transformerBlock,
            .rotaryPositionEmbedding
        ]

        #expect(types.count == 10)
    }

    @Test("DetectedPattern stores metadata")
    func detectedPattern() {
        var metadata = PatternMetadata()
        metadata.numHeads = 8
        metadata.headDim = 64

        let pattern = DetectedPattern(
            type: .attention,
            operationIndices: [0, 1, 2],
            rootIndex: 2,
            metadata: metadata
        )

        #expect(pattern.type == .attention)
        #expect(pattern.operationIndices == [0, 1, 2])
        #expect(pattern.rootIndex == 2)
        #expect(pattern.metadata.numHeads == 8)
        #expect(pattern.metadata.headDim == 64)
    }

    @Test("FusedOpType has all expected cases")
    func fusedOpType() {
        // Test original op type
        let originalType = FusedOpType.original(.add)
        switch originalType {
        case .original(let opKind):
            #expect(opKind == .add)
        default:
            Issue.record("Expected original type")
        }

        // Test fused types with associated values
        let attentionConfig = AttentionConfig(numHeads: 8, headDim: 64)
        let mhaConfig = MultiHeadAttentionConfig(numHeads: 8, headDim: 64, hiddenDim: 512)
        let normConfig = NormConfig(epsilon: 1e-5)
        let matmulConfig = MatMulConfig()
        let ffnConfig = FFNConfig(hiddenDim: 768, intermediateDim: 3072)
        let blockConfig = TransformerBlockConfig(attention: mhaConfig, ffn: ffnConfig)
        let ropeConfig = RoPEConfig(dim: 64)

        let fusedTypes: [FusedOpType] = [
            .fusedAttention(attentionConfig),
            .fusedMultiHeadAttention(mhaConfig),
            .fusedRMSNorm(normConfig),
            .fusedLayerNorm(normConfig),
            .fusedMatMulBiasAct(matmulConfig),
            .fusedGELU(approximate: true),
            .fusedSiLU,
            .fusedElementwise([.add, .multiply]),
            .fusedFFN(ffnConfig),
            .fusedTransformerBlock(blockConfig),
            .fusedRoPE(ropeConfig)
        ]

        #expect(fusedTypes.count == 11)
    }

    @Test("FusedOp stores all properties")
    func fusedOp() {
        let fusedOp = FusedOp(
            id: "fused1",
            type: .fusedGELU(approximate: true),
            inputs: ["x"],
            outputs: [TensorInfo(id: "y", shape: [4], elementType: .float32)],
            originalOps: ["op0", "op1"],
            attributes: HLOAttributes()
        )

        #expect(fusedOp.id == "fused1")
        #expect(fusedOp.inputs == ["x"])
        #expect(fusedOp.outputs.count == 1)
        #expect(fusedOp.originalOps == ["op0", "op1"])
    }
}

// MARK: - Kernel Types Tests

@Suite("Kernel Types Tests")
struct KernelTypesTests {

    @Test("DispatchConfig calculates totals correctly")
    func dispatchConfigTotals() {
        let config = DispatchConfig(
            gridSize: MTLSize(width: 4, height: 3, depth: 2),
            threadgroupSize: MTLSize(width: 8, height: 8, depth: 1)
        )

        #expect(config.totalThreadgroups == 4 * 3 * 2)  // 24
        #expect(config.totalThreads == 4 * 3 * 2 * 8 * 8 * 1)  // 1536
    }

    @Test("DispatchConfig.dispatch1D creates correct config")
    func dispatch1D() {
        let config = DispatchConfig.dispatch1D(elements: 1000, threadgroupSize: 256)

        #expect(config.threadgroupSize.width == 256)
        #expect(config.gridSize.width == 4)  // ceil(1000/256) = 4
        #expect(config.gridSize.height == 1)
        #expect(config.gridSize.depth == 1)
    }

    @Test("DispatchConfig.dispatch2D creates correct config")
    func dispatch2D() {
        let config = DispatchConfig.dispatch2D(width: 100, height: 50, tileWidth: 16, tileHeight: 16)

        #expect(config.threadgroupSize.width == 16)
        #expect(config.threadgroupSize.height == 16)
        #expect(config.gridSize.width == 7)  // ceil(100/16) = 7
        #expect(config.gridSize.height == 4)  // ceil(50/16) ≈ 4
    }

    @Test("BufferSource has all expected cases")
    func bufferSource() {
        let sources: [BufferSource] = [
            .input(name: "x"),
            .output(name: "y"),
            .unified(offset: 0),
            .constant(id: "const"),
            .threadgroup(size: 1024)
        ]

        #expect(sources.count == 5)
    }

    @Test("BufferBinding stores all properties")
    func bufferBinding() {
        let binding = BufferBinding(
            index: 0,
            source: .input(name: "x"),
            offset: 64,
            size: 256,
            access: .readWrite
        )

        #expect(binding.index == 0)
        #expect(binding.offset == 64)
        #expect(binding.size == 256)
        #expect(binding.access == .readWrite)
    }

    @Test("TuningConfig presets have expected values")
    func tuningPresets() {
        let small = TuningConfig.small
        #expect(small.blockSize == 256)
        #expect(small.useSharedMemory == false)

        let matmul = TuningConfig.matmul
        #expect(matmul.tileM == 32)
        #expect(matmul.tileN == 32)
        #expect(matmul.useSharedMemory == true)

        let attention = TuningConfig.attention
        #expect(attention.tileM == 64)
        #expect(attention.unrollFactor == 4)
    }

    @Test("TensorSpec calculates byte size correctly")
    func tensorSpec() {
        let spec = TensorSpec(
            name: "tensor",
            shape: [2, 3, 4],
            elementType: .float32
        )

        #expect(spec.name == "tensor")
        #expect(spec.byteSize == 2 * 3 * 4 * 4)  // 96 bytes
    }

    @Test("TensorSpec initializes from TensorParam")
    func tensorSpecFromParam() {
        let param = TensorParam(
            name: "param",
            type: TensorType(shape: [10, 20], elementType: .float16)
        )

        let spec = TensorSpec(from: param)

        #expect(spec.name == "param")
        #expect(spec.shape == [10, 20])
        #expect(spec.elementType == .float16)
    }
}
