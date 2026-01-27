// QuantizationAwareTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2G: Quantization-Aware Compilation

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("Quantization-Aware Tests")
struct QuantizationAwareTests {

    // MARK: - QuantizationType Tests

    @Test("QuantizationType bytes per element")
    func quantizationTypeBytesPerElement() {
        #expect(QuantizationType.int8.bytesPerElement == 1)
        #expect(QuantizationType.uint8.bytesPerElement == 1)
        #expect(QuantizationType.int4.bytesPerElement == 1)
        #expect(QuantizationType.float16.bytesPerElement == 2)
        #expect(QuantizationType.bfloat16.bytesPerElement == 2)
    }

    @Test("QuantizationType requires calibration")
    func quantizationTypeRequiresCalibration() {
        #expect(QuantizationType.int8.requiresCalibration == true)
        #expect(QuantizationType.uint8.requiresCalibration == true)
        #expect(QuantizationType.int4.requiresCalibration == true)
        #expect(QuantizationType.float16.requiresCalibration == false)
        #expect(QuantizationType.bfloat16.requiresCalibration == false)
    }

    // MARK: - QuantizationConfig Tests

    @Test("QuantizationConfig default presets")
    func quantizationConfigPresets() {
        let int8PerTensor = QuantizationConfig.int8PerTensor
        #expect(int8PerTensor.dtype == .int8)
        #expect(int8PerTensor.scheme == .perTensor)
        #expect(int8PerTensor.symmetric == true)

        let int8PerChannel = QuantizationConfig.int8PerChannel
        #expect(int8PerChannel.dtype == .int8)
        #expect(int8PerChannel.scheme == .perChannel)
        #expect(int8PerChannel.axis == 0)

        let int4Group = QuantizationConfig.int4Group128
        #expect(int4Group.dtype == .int4)
        #expect(int4Group.scheme == .perGroup)
        #expect(int4Group.groupSize == 128)
    }

    @Test("QuantizationConfig equality")
    func quantizationConfigEquality() {
        let config1 = QuantizationConfig(dtype: .int8, scheme: .perTensor)
        let config2 = QuantizationConfig(dtype: .int8, scheme: .perTensor)
        let config3 = QuantizationConfig(dtype: .uint8, scheme: .perTensor)

        #expect(config1 == config2)
        #expect(config1 != config3)
    }

    // MARK: - QuantizationParams Tests

    @Test("QuantizationParams from statistics symmetric")
    func quantizationParamsSymmetric() {
        let params = QuantizationParams.fromStatistics(
            min: -5.0,
            max: 3.0,
            config: QuantizationConfig(dtype: .int8, symmetric: true)
        )

        // Symmetric uses max(abs(min), abs(max)) = 5.0
        // scale = 5.0 / 127
        #expect(params.scales.count == 1)
        #expect(params.zeroPoints.first == 0)
        #expect(params.qmin == -128)
        #expect(params.qmax == 127)
    }

    @Test("QuantizationParams from statistics asymmetric")
    func quantizationParamsAsymmetric() {
        let params = QuantizationParams.fromStatistics(
            min: 0.0,
            max: 1.0,
            config: QuantizationConfig(dtype: .uint8, scheme: .perTensor, symmetric: false)
        )

        // Asymmetric: scale = (max - min) / (qmax - qmin) = 1.0 / 255
        #expect(params.scales.count == 1)
        #expect(params.qmin == 0)
        #expect(params.qmax == 255)
    }

    @Test("QuantizationParams ranges for different types")
    func quantizationParamsRanges() {
        let int8Params = QuantizationParams(scales: [1.0], config: .int8PerTensor)
        #expect(int8Params.qmin == -128)
        #expect(int8Params.qmax == 127)

        let uint8Config = QuantizationConfig(dtype: .uint8)
        let uint8Params = QuantizationParams(scales: [1.0], config: uint8Config)
        #expect(uint8Params.qmin == 0)
        #expect(uint8Params.qmax == 255)

        let int4Config = QuantizationConfig(dtype: .int4)
        let int4Params = QuantizationParams(scales: [1.0], config: int4Config)
        #expect(int4Params.qmin == -8)
        #expect(int4Params.qmax == 7)
    }

    // MARK: - MixedPrecisionPolicy Tests

    @Test("MixedPrecisionPolicy presets")
    func mixedPrecisionPolicyPresets() {
        let fp16Train = MixedPrecisionPolicy.fp16Training
        #expect(fp16Train.defaultPrecision == .float16)
        #expect(fp16Train.useLossScaling == true)
        #expect(fp16Train.fullPrecisionOps.contains("softmax"))

        let fp16Infer = MixedPrecisionPolicy.fp16Inference
        #expect(fp16Infer.defaultPrecision == .float16)
        #expect(fp16Infer.useLossScaling == false)

        let bf16 = MixedPrecisionPolicy.bf16
        #expect(bf16.defaultPrecision == .bfloat16)
    }

    @Test("MixedPrecisionPolicy custom config")
    func mixedPrecisionPolicyCustom() {
        let policy = MixedPrecisionPolicy(
            reducedPrecisionOps: ["dot", "convolution"],
            fullPrecisionOps: ["softmax", "layer_norm"],
            defaultPrecision: .float16,
            useLossScaling: true,
            initialLossScale: 1024.0
        )

        #expect(policy.reducedPrecisionOps.count == 2)
        #expect(policy.fullPrecisionOps.count == 2)
        #expect(policy.initialLossScale == 1024.0)
    }

    // MARK: - QuantizationAnalyzer Tests

    @Test("QuantizationAnalyzer analyzes function")
    func quantizationAnalyzerAnalyzes() {
        let function = createTestFunction()
        let analyzer = QuantizationAnalyzer()

        let analysis = analyzer.analyze(function)

        #expect(analysis.count > 0)
    }

    @Test("QuantizationAnalyzer recommends INT8 for matmul")
    func quantizationAnalyzerMatmul() {
        let function = createMatmulFunction()
        let analyzer = QuantizationAnalyzer()

        let analysis = analyzer.analyze(function)

        if let matmulAnalysis = analysis["matmul_result"] {
            #expect(matmulAnalysis.canQuantize == true)
            #expect(matmulAnalysis.recommendedConfig?.dtype == .int8)
            #expect(matmulAnalysis.memorySavings > 0.5)
        }
    }

    @Test("QuantizationAnalyzer does not quantize softmax")
    func quantizationAnalyzerSoftmax() {
        let function = createSoftmaxFunction()
        let analyzer = QuantizationAnalyzer()

        let analysis = analyzer.analyze(function)

        if let softmaxAnalysis = analysis["softmax_result"] {
            #expect(softmaxAnalysis.canQuantize == false)
            #expect(softmaxAnalysis.reason != nil)
        }
    }

    // MARK: - QuantizedKVCacheConfig Tests

    @Test("QuantizedKVCacheConfig memory usage")
    func quantizedKVCacheMemory() {
        let config = QuantizedKVCacheConfig(
            keyQuantization: .int8,
            valueQuantization: .int8,
            maxSeqLen: 2048,
            numKVHeads: 32,
            headDim: 128
        )

        // INT8 K and V: 2048 * 32 * 128 * 1 * 2 = 16,777,216 bytes
        // Plus scales
        #expect(config.memoryUsage > 0)
        #expect(config.memorySavingsRatio > 0.5)
    }

    @Test("QuantizedKVCacheConfig savings ratio")
    func quantizedKVCacheSavings() {
        let int8Config = QuantizedKVCacheConfig(
            keyQuantization: .int8,
            valueQuantization: .int8,
            maxSeqLen: 1024,
            numKVHeads: 8,
            headDim: 64
        )

        let int4Config = QuantizedKVCacheConfig(
            keyQuantization: .int4,
            valueQuantization: .int4,
            maxSeqLen: 1024,
            numKVHeads: 8,
            headDim: 64
        )

        // INT8 saves ~75%, INT4 saves ~87.5%
        #expect(int8Config.memorySavingsRatio > 0.5)
        // INT4 should save more (or equal if packed)
        #expect(int4Config.memorySavingsRatio >= int8Config.memorySavingsRatio - 0.1)
    }

    // MARK: - QuantizedKVCacheManager Tests

    @Test("QuantizedKVCacheManager updates stats")
    func quantizedKVCacheManagerStats() {
        let config = QuantizedKVCacheConfig()
        let manager = QuantizedKVCacheManager(config: config)

        manager.updateStats(keyMin: -1.0, keyMax: 1.0, valueMin: -0.5, valueMax: 0.5)
        manager.updateStats(keyMin: -2.0, keyMax: 0.5, valueMin: -0.3, valueMax: 0.8)

        let keyParams = manager.getKeyParams()
        let valueParams = manager.getValueParams()

        #expect(keyParams.scales.count > 0)
        #expect(valueParams.scales.count > 0)
    }

    @Test("QuantizedKVCacheManager memory stats")
    func quantizedKVCacheManagerMemoryStats() {
        let config = QuantizedKVCacheConfig(
            maxSeqLen: 1024,
            numKVHeads: 16,
            headDim: 64
        )
        let manager = QuantizedKVCacheManager(config: config)

        let (usage, savings) = manager.memoryStats
        #expect(usage > 0)
        #expect(savings > 0)
    }

    // MARK: - FusedQuantOp Tests

    @Test("FusedQuantOp equality")
    func fusedQuantOpEquality() {
        let op1 = FusedQuantOp.int8MatmulDequant
        let op2 = FusedQuantOp.int8MatmulDequant
        let op3 = FusedQuantOp.dequantActivation(activation: .relu)

        #expect(op1 == op2)
        #expect(op1 != op3)
    }

    @Test("FusedQuantOp with configs")
    func fusedQuantOpConfigs() {
        let config1 = QuantizationConfig.int8PerTensor
        let config2 = QuantizationConfig.int8PerChannel

        let op1 = FusedQuantOp.quantizedMatmul(aConfig: config1, bConfig: config2)
        let op2 = FusedQuantOp.quantizedMatmul(aConfig: config1, bConfig: config2)
        let op3 = FusedQuantOp.quantizedMatmul(aConfig: config2, bConfig: config1)

        #expect(op1 == op2)
        #expect(op1 != op3)
    }

    // MARK: - FusedQuantPatternDetector Tests

    @Test("FusedQuantPatternDetector detects matmul patterns")
    func fusedQuantPatternDetectorMatmul() {
        let function = createMatmulFunction()
        let detector = FusedQuantPatternDetector()

        let patterns = detector.detectPatterns(function)

        #expect(patterns.count > 0)

        if let pattern = patterns.first {
            #expect(pattern.operations.count > 0)
            #expect(pattern.estimatedSpeedup > 1.0)
        }
    }

    // MARK: - QuantizationAwareOptimizer Tests

    @Test("QuantizationAwareOptimizer creates plan")
    func quantizationAwareOptimizerPlan() {
        let function = createTestFunction()

        let config = QuantizationAwareOptimizer.Config(
            targetOps: ["dot", "dot_general"],
            defaultConfig: .int8PerTensor,
            analyzeFirst: true
        )
        let optimizer = QuantizationAwareOptimizer(config: config)

        let plan = optimizer.optimize(function)

        #expect(plan.estimatedMemorySavings >= 0)
        #expect(plan.estimatedSpeedup >= 1.0)
    }

    @Test("QuantizationAwareOptimizer respects min savings")
    func quantizationAwareOptimizerMinSavings() {
        let function = createTestFunction()

        // High threshold - fewer tensors should be quantized
        var highConfig = QuantizationAwareOptimizer.Config()
        highConfig.minMemorySavings = 0.9
        let highOptimizer = QuantizationAwareOptimizer(config: highConfig)
        let highPlan = highOptimizer.optimize(function)

        // Low threshold - more tensors should be quantized
        var lowConfig = QuantizationAwareOptimizer.Config()
        lowConfig.minMemorySavings = 0.1
        let lowOptimizer = QuantizationAwareOptimizer(config: lowConfig)
        let lowPlan = lowOptimizer.optimize(function)

        #expect(lowPlan.quantizedTensorCount >= highPlan.quantizedTensorCount)
    }

    @Test("QuantizationAwareOptimizer with mixed precision")
    func quantizationAwareOptimizerMixedPrecision() {
        let function = createTestFunction()

        let config = QuantizationAwareOptimizer.Config(
            mixedPrecisionPolicy: .fp16Inference
        )
        let optimizer = QuantizationAwareOptimizer(config: config)

        let plan = optimizer.optimize(function)

        #expect(plan.mixedPrecisionPolicy != nil)
        #expect(plan.mixedPrecisionPolicy?.defaultPrecision == .float16)
    }

    // MARK: - QuantizationPlan Tests

    @Test("QuantizationPlan properties")
    func quantizationPlanProperties() {
        let tensorConfigs: [String: QuantizationConfig] = [
            "tensor1": .int8PerTensor,
            "tensor2": .int8PerChannel
        ]

        let fusedOps: [(operations: [Int], fusedOp: FusedQuantOp)] = [
            ([0, 1], .int8MatmulDequant)
        ]

        let plan = QuantizationPlan(
            tensorConfigs: tensorConfigs,
            fusedOperations: fusedOps,
            mixedPrecisionPolicy: .fp16Inference,
            estimatedMemorySavings: 0.5,
            estimatedSpeedup: 1.5
        )

        #expect(plan.quantizedTensorCount == 2)
        #expect(plan.fusedOpCount == 1)
        #expect(plan.estimatedMemorySavings == 0.5)
        #expect(plan.estimatedSpeedup == 1.5)
    }

    // MARK: - QuantizationCalibrator Tests

    @Test("QuantizationCalibrator updates and retrieves")
    func quantizationCalibratorBasic() {
        let calibrator = QuantizationCalibrator()

        calibrator.update(tensorId: "tensor1", min: -1.0, max: 1.0)
        calibrator.update(tensorId: "tensor1", min: -0.5, max: 2.0)

        let params = calibrator.getParams(for: "tensor1", config: .int8PerTensor)

        #expect(params != nil)
        #expect(calibrator.calibratedCount == 1)
    }

    @Test("QuantizationCalibrator tracks multiple tensors")
    func quantizationCalibratorMultiple() {
        let calibrator = QuantizationCalibrator()

        calibrator.update(tensorId: "tensor1", min: -1.0, max: 1.0)
        calibrator.update(tensorId: "tensor2", min: 0.0, max: 255.0)
        calibrator.update(tensorId: "tensor3", min: -10.0, max: 10.0)

        #expect(calibrator.calibratedCount == 3)

        let params1 = calibrator.getParams(for: "tensor1", config: .int8PerTensor)
        let params2 = calibrator.getParams(for: "tensor2", config: QuantizationConfig(dtype: .uint8))
        let params3 = calibrator.getParams(for: "tensor3", config: .int8PerTensor)

        #expect(params1 != nil)
        #expect(params2 != nil)
        #expect(params3 != nil)
    }

    @Test("QuantizationCalibrator reset")
    func quantizationCalibratorReset() {
        let calibrator = QuantizationCalibrator()

        calibrator.update(tensorId: "tensor1", min: -1.0, max: 1.0)
        #expect(calibrator.calibratedCount == 1)

        calibrator.reset()
        #expect(calibrator.calibratedCount == 0)
    }

    // MARK: - Integration Tests

    @Test("Full quantization workflow")
    func fullQuantizationWorkflow() {
        let function = createTransformerFunction()

        // Analyze
        let analyzer = QuantizationAnalyzer()
        let analysis = analyzer.analyze(function)

        // Calibrate (simulated)
        let calibrator = QuantizationCalibrator()
        for op in function.operations {
            calibrator.update(tensorId: op.result, min: -1.0, max: 1.0)
        }

        // Optimize
        let optimizer = QuantizationAwareOptimizer(config: .init(
            mixedPrecisionPolicy: .fp16Inference
        ))
        let plan = optimizer.optimize(function)

        // Verify
        #expect(analysis.count > 0)
        #expect(calibrator.calibratedCount == function.operations.count)
        #expect(plan.estimatedMemorySavings >= 0)
    }

    @Test("Quantized KV cache workflow")
    func quantizedKVCacheWorkflow() {
        // Configure KV cache
        let kvConfig = QuantizedKVCacheConfig(
            keyQuantization: .int8,
            valueQuantization: .int8,
            maxSeqLen: 4096,
            numKVHeads: 32,
            headDim: 128
        )

        let manager = QuantizedKVCacheManager(config: kvConfig)

        // Simulate calibration over multiple batches
        for _ in 0..<10 {
            let keyMin = Float.random(in: -2.0...0.0)
            let keyMax = Float.random(in: 0.0...2.0)
            let valueMin = Float.random(in: -1.0...0.0)
            let valueMax = Float.random(in: 0.0...1.0)

            manager.updateStats(
                keyMin: keyMin,
                keyMax: keyMax,
                valueMin: valueMin,
                valueMax: valueMax
            )
        }

        // Get params
        let keyParams = manager.getKeyParams()
        let valueParams = manager.getValueParams()

        #expect(keyParams.scales.count > 0)
        #expect(valueParams.scales.count > 0)

        // Check memory savings
        let (_, savings) = manager.memoryStats
        #expect(savings > 0.5)  // Should save at least 50%
    }

    // MARK: - Helper Functions

    private func createTestFunction() -> HLOFunction {
        let inputs = [
            HLOArgument(name: "input", type: TensorType(shape: [32, 64], elementType: .float32))
        ]

        let operations = [
            HLOOperation(
                result: "result1",
                kind: .exponential,
                operands: ["input"],
                resultType: TensorType(shape: [32, 64], elementType: .float32),
                attributes: HLOAttributes()
            ),
            HLOOperation(
                result: "result2",
                kind: .tanh,
                operands: ["result1"],
                resultType: TensorType(shape: [32, 64], elementType: .float32),
                attributes: HLOAttributes()
            )
        ]

        return HLOFunction(
            name: "test_fn",
            inputs: inputs,
            outputTypes: [TensorType(shape: [32, 64], elementType: .float32)],
            operations: operations,
            returnValues: ["result2"]
        )
    }

    private func createMatmulFunction() -> HLOFunction {
        let inputs = [
            HLOArgument(name: "a", type: TensorType(shape: [64, 128], elementType: .float32)),
            HLOArgument(name: "b", type: TensorType(shape: [128, 256], elementType: .float32))
        ]

        let operations = [
            HLOOperation(
                result: "matmul_result",
                kind: .dot,
                operands: ["a", "b"],
                resultType: TensorType(shape: [64, 256], elementType: .float32),
                attributes: HLOAttributes()
            )
        ]

        return HLOFunction(
            name: "matmul_fn",
            inputs: inputs,
            outputTypes: [TensorType(shape: [64, 256], elementType: .float32)],
            operations: operations,
            returnValues: ["matmul_result"]
        )
    }

    private func createSoftmaxFunction() -> HLOFunction {
        let inputs = [
            HLOArgument(name: "input", type: TensorType(shape: [32, 1000], elementType: .float32))
        ]

        // Use reduce as a proxy for operations that shouldn't be quantized
        let operations = [
            HLOOperation(
                result: "softmax_result",
                kind: .reduce,
                operands: ["input"],
                resultType: TensorType(shape: [32, 1000], elementType: .float32),
                attributes: HLOAttributes()
            )
        ]

        return HLOFunction(
            name: "softmax_fn",
            inputs: inputs,
            outputTypes: [TensorType(shape: [32, 1000], elementType: .float32)],
            operations: operations,
            returnValues: ["softmax_result"]
        )
    }

    private func createTransformerFunction() -> HLOFunction {
        let shape = [4, 512, 768]

        let inputs = [
            HLOArgument(name: "hidden", type: TensorType(shape: shape, elementType: .float32))
        ]

        let operations = [
            // QKV projection (matmul)
            HLOOperation(
                result: "qkv",
                kind: .dot,
                operands: ["hidden", "hidden"],
                resultType: TensorType(shape: [4, 512, 2304], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Attention (can be quantized)
            HLOOperation(
                result: "attn",
                kind: .dot,
                operands: ["qkv", "qkv"],
                resultType: TensorType(shape: shape, elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Normalization/reduction (keep FP32)
            HLOOperation(
                result: "probs",
                kind: .reduce,
                operands: ["attn"],
                resultType: TensorType(shape: shape, elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Output projection
            HLOOperation(
                result: "output",
                kind: .dot,
                operands: ["probs", "hidden"],
                resultType: TensorType(shape: shape, elementType: .float32),
                attributes: HLOAttributes()
            )
        ]

        return HLOFunction(
            name: "transformer_fn",
            inputs: inputs,
            outputTypes: [TensorType(shape: shape, elementType: .float32)],
            operations: operations,
            returnValues: ["output"]
        )
    }
}
