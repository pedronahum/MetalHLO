// QuantizationAware.swift
// MetalHLOCore
//
// Quantization-aware compilation for reduced memory and faster inference.
// Supports INT8/INT4 quantization, mixed precision, and fused quant ops.

import Foundation

// MARK: - Quantization Configuration

/// Data type for quantized values.
public enum QuantizationType: String, Sendable, CaseIterable {
    /// 8-bit signed integer quantization.
    case int8

    /// 8-bit unsigned integer quantization.
    case uint8

    /// 4-bit integer quantization (packed).
    case int4

    /// 16-bit floating point (brain float).
    case bfloat16

    /// 16-bit floating point (IEEE).
    case float16

    /// Bytes per element (for packed types, this is the unpacked size).
    public var bytesPerElement: Int {
        switch self {
        case .int8, .uint8:
            return 1
        case .int4:
            return 1  // 2 values packed per byte when stored
        case .bfloat16, .float16:
            return 2
        }
    }

    /// Whether this type requires scale and zero-point.
    public var requiresCalibration: Bool {
        switch self {
        case .int8, .uint8, .int4:
            return true
        case .bfloat16, .float16:
            return false
        }
    }
}

/// Quantization scheme/method.
public enum QuantizationScheme: String, Sendable {
    /// Per-tensor quantization (single scale/zero-point).
    case perTensor

    /// Per-channel quantization (scale/zero-point per output channel).
    case perChannel

    /// Per-group quantization (for LLM weights, groups of K elements).
    case perGroup

    /// Dynamic quantization (calibration at runtime).
    case dynamic

    /// Static quantization (pre-calibrated).
    case `static`
}

/// Configuration for a quantized tensor.
public struct QuantizationConfig: Sendable, Equatable {
    /// Target quantization type.
    public let dtype: QuantizationType

    /// Quantization scheme.
    public let scheme: QuantizationScheme

    /// Group size for per-group quantization.
    public let groupSize: Int

    /// Symmetric quantization (zero-point = 0).
    public let symmetric: Bool

    /// Axis for per-channel quantization (-1 for per-tensor).
    public let axis: Int

    public init(
        dtype: QuantizationType = .int8,
        scheme: QuantizationScheme = .perTensor,
        groupSize: Int = 128,
        symmetric: Bool = true,
        axis: Int = -1
    ) {
        self.dtype = dtype
        self.scheme = scheme
        self.groupSize = groupSize
        self.symmetric = symmetric
        self.axis = axis
    }

    /// Default INT8 per-tensor config.
    public static let int8PerTensor = QuantizationConfig(
        dtype: .int8,
        scheme: .perTensor,
        symmetric: true
    )

    /// INT8 per-channel config (for weights).
    public static let int8PerChannel = QuantizationConfig(
        dtype: .int8,
        scheme: .perChannel,
        axis: 0
    )

    /// INT4 group quantization config (for LLM weights).
    public static let int4Group128 = QuantizationConfig(
        dtype: .int4,
        scheme: .perGroup,
        groupSize: 128,
        symmetric: false
    )
}

// MARK: - Quantization Parameters

/// Parameters for quantizing a tensor.
public struct QuantizationParams: Sendable {
    /// Scale factor(s) for dequantization.
    public let scales: [Float]

    /// Zero point(s) for asymmetric quantization.
    public let zeroPoints: [Int32]

    /// The quantization config used.
    public let config: QuantizationConfig

    /// Minimum representable value.
    public let qmin: Int32

    /// Maximum representable value.
    public let qmax: Int32

    public init(
        scales: [Float],
        zeroPoints: [Int32] = [],
        config: QuantizationConfig,
        qmin: Int32? = nil,
        qmax: Int32? = nil
    ) {
        self.scales = scales
        self.zeroPoints = zeroPoints.isEmpty ? Array(repeating: 0, count: scales.count) : zeroPoints
        self.config = config

        // Default quantization ranges
        switch config.dtype {
        case .int8:
            self.qmin = qmin ?? -128
            self.qmax = qmax ?? 127
        case .uint8:
            self.qmin = qmin ?? 0
            self.qmax = qmax ?? 255
        case .int4:
            self.qmin = qmin ?? -8
            self.qmax = qmax ?? 7
        case .bfloat16, .float16:
            self.qmin = qmin ?? 0
            self.qmax = qmax ?? 0
        }
    }

    /// Computes quantization parameters from tensor statistics.
    public static func fromStatistics(
        min: Float,
        max: Float,
        config: QuantizationConfig
    ) -> QuantizationParams {
        let qmin: Int32
        let qmax: Int32

        switch config.dtype {
        case .int8:
            qmin = -128
            qmax = 127
        case .uint8:
            qmin = 0
            qmax = 255
        case .int4:
            qmin = -8
            qmax = 7
        case .bfloat16, .float16:
            // No quantization params needed
            return QuantizationParams(scales: [1.0], config: config)
        }

        let scale: Float
        let zeroPoint: Int32

        if config.symmetric {
            let absMax = Swift.max(abs(min), abs(max))
            scale = absMax / Float(qmax)
            zeroPoint = 0
        } else {
            scale = (max - min) / Float(qmax - qmin)
            zeroPoint = Int32(round(Float(qmin) - min / scale))
        }

        return QuantizationParams(
            scales: [scale],
            zeroPoints: [zeroPoint],
            config: config,
            qmin: qmin,
            qmax: qmax
        )
    }
}

// MARK: - Mixed Precision Policy

/// Policy for mixed precision computation.
public struct MixedPrecisionPolicy: Sendable {
    /// Operations that should use reduced precision.
    public var reducedPrecisionOps: Set<String>

    /// Operations that must remain in full precision.
    public var fullPrecisionOps: Set<String>

    /// Default precision for unlisted operations.
    public var defaultPrecision: QuantizationType

    /// Whether to use loss scaling for training.
    public var useLossScaling: Bool

    /// Initial loss scale value.
    public var initialLossScale: Float

    public init(
        reducedPrecisionOps: Set<String> = [],
        fullPrecisionOps: Set<String> = [],
        defaultPrecision: QuantizationType = .float16,
        useLossScaling: Bool = false,
        initialLossScale: Float = 65536.0
    ) {
        self.reducedPrecisionOps = reducedPrecisionOps
        self.fullPrecisionOps = fullPrecisionOps
        self.defaultPrecision = defaultPrecision
        self.useLossScaling = useLossScaling
        self.initialLossScale = initialLossScale
    }

    /// Default FP16 training policy.
    public static let fp16Training = MixedPrecisionPolicy(
        fullPrecisionOps: ["batch_norm_inference", "batch_norm_training", "softmax", "log_softmax"],
        defaultPrecision: .float16,
        useLossScaling: true
    )

    /// Default FP16 inference policy.
    public static let fp16Inference = MixedPrecisionPolicy(
        fullPrecisionOps: ["softmax"],
        defaultPrecision: .float16,
        useLossScaling: false
    )

    /// BFloat16 policy (better numeric stability).
    public static let bf16 = MixedPrecisionPolicy(
        defaultPrecision: .bfloat16,
        useLossScaling: false
    )
}

// MARK: - Quantization Analyzer

/// Analyzes operations for quantization opportunities.
public final class QuantizationAnalyzer: @unchecked Sendable {

    /// Analysis result for a tensor.
    public struct TensorAnalysis: Sendable {
        /// Whether the tensor can be quantized.
        public let canQuantize: Bool

        /// Recommended quantization config.
        public let recommendedConfig: QuantizationConfig?

        /// Estimated memory savings ratio.
        public let memorySavings: Double

        /// Estimated accuracy impact (0 = no impact, 1 = severe).
        public let accuracyImpact: Double

        /// Reason if cannot quantize.
        public let reason: String?
    }

    public init() {}

    /// Analyzes a function for quantization opportunities.
    public func analyze(_ function: HLOFunction) -> [String: TensorAnalysis] {
        var results: [String: TensorAnalysis] = [:]

        for op in function.operations {
            let analysis = analyzeOperation(op)
            results[op.result] = analysis
        }

        return results
    }

    private func analyzeOperation(_ op: HLOOperation) -> TensorAnalysis {
        // Check operation type
        switch op.kind {
        case .dot, .dotGeneral, .convolution:
            // Compute-intensive ops benefit from INT8
            return TensorAnalysis(
                canQuantize: true,
                recommendedConfig: .int8PerChannel,
                memorySavings: 0.75,  // 4 bytes -> 1 byte
                accuracyImpact: 0.1,
                reason: nil
            )

        case .add, .subtract, .multiply, .divide:
            // Elementwise ops can use FP16
            return TensorAnalysis(
                canQuantize: true,
                recommendedConfig: QuantizationConfig(dtype: .float16),
                memorySavings: 0.5,  // 4 bytes -> 2 bytes
                accuracyImpact: 0.05,
                reason: nil
            )

        case .reduce, .reduceWindow:
            // Reductions need higher precision for accumulation
            return TensorAnalysis(
                canQuantize: false,
                recommendedConfig: nil,
                memorySavings: 0,
                accuracyImpact: 0.5,
                reason: "Reductions require full precision for accuracy"
            )

        case .batchNormInference, .batchNormTraining:
            // Batch norm needs FP32 for numeric stability
            return TensorAnalysis(
                canQuantize: false,
                recommendedConfig: nil,
                memorySavings: 0,
                accuracyImpact: 0.8,
                reason: "Normalization requires full precision"
            )

        default:
            // Default: allow FP16
            return TensorAnalysis(
                canQuantize: true,
                recommendedConfig: QuantizationConfig(dtype: .float16),
                memorySavings: 0.5,
                accuracyImpact: 0.1,
                reason: nil
            )
        }
    }
}

// MARK: - Quantized KV Cache

/// Configuration for quantized KV cache.
public struct QuantizedKVCacheConfig: Sendable, Equatable {
    /// Quantization type for keys.
    public var keyQuantization: QuantizationType

    /// Quantization type for values.
    public var valueQuantization: QuantizationType

    /// Maximum sequence length.
    public var maxSeqLen: Int

    /// Number of key-value heads.
    public var numKVHeads: Int

    /// Dimension per head.
    public var headDim: Int

    /// Whether to quantize per-head.
    public var perHeadQuantization: Bool

    public init(
        keyQuantization: QuantizationType = .int8,
        valueQuantization: QuantizationType = .int8,
        maxSeqLen: Int = 2048,
        numKVHeads: Int = 32,
        headDim: Int = 128,
        perHeadQuantization: Bool = true
    ) {
        self.keyQuantization = keyQuantization
        self.valueQuantization = valueQuantization
        self.maxSeqLen = maxSeqLen
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.perHeadQuantization = perHeadQuantization
    }

    /// Computes memory usage in bytes.
    public var memoryUsage: Int {
        let keySize = maxSeqLen * numKVHeads * headDim * keyQuantization.bytesPerElement
        let valueSize = maxSeqLen * numKVHeads * headDim * valueQuantization.bytesPerElement

        // Add scales and zero-points
        let scalesSize: Int
        if perHeadQuantization {
            scalesSize = maxSeqLen * numKVHeads * 4 * 2  // scale + zp for K and V
        } else {
            scalesSize = maxSeqLen * 4 * 2
        }

        return keySize + valueSize + scalesSize
    }

    /// Memory savings vs FP32.
    public var memorySavingsRatio: Double {
        let fp32Size = maxSeqLen * numKVHeads * headDim * 4 * 2  // K and V
        return 1.0 - Double(memoryUsage) / Double(fp32Size)
    }
}

/// Manager for quantized KV cache operations.
public final class QuantizedKVCacheManager: @unchecked Sendable {

    /// Statistics for calibration.
    public struct CalibrationStats: Sendable {
        public var keyMin: Float
        public var keyMax: Float
        public var valueMin: Float
        public var valueMax: Float
        public var sampleCount: Int

        public init() {
            self.keyMin = Float.infinity
            self.keyMax = -Float.infinity
            self.valueMin = Float.infinity
            self.valueMax = -Float.infinity
            self.sampleCount = 0
        }

        mutating func update(keyMin: Float, keyMax: Float, valueMin: Float, valueMax: Float) {
            self.keyMin = Swift.min(self.keyMin, keyMin)
            self.keyMax = Swift.max(self.keyMax, keyMax)
            self.valueMin = Swift.min(self.valueMin, valueMin)
            self.valueMax = Swift.max(self.valueMax, valueMax)
            self.sampleCount += 1
        }
    }

    private let config: QuantizedKVCacheConfig
    private var stats: CalibrationStats

    public init(config: QuantizedKVCacheConfig) {
        self.config = config
        self.stats = CalibrationStats()
    }

    /// Gets quantization params for keys.
    public func getKeyParams() -> QuantizationParams {
        let quantConfig = QuantizationConfig(
            dtype: config.keyQuantization,
            scheme: config.perHeadQuantization ? .perChannel : .perTensor
        )

        return QuantizationParams.fromStatistics(
            min: stats.keyMin.isFinite ? stats.keyMin : -1.0,
            max: stats.keyMax.isFinite ? stats.keyMax : 1.0,
            config: quantConfig
        )
    }

    /// Gets quantization params for values.
    public func getValueParams() -> QuantizationParams {
        let quantConfig = QuantizationConfig(
            dtype: config.valueQuantization,
            scheme: config.perHeadQuantization ? .perChannel : .perTensor
        )

        return QuantizationParams.fromStatistics(
            min: stats.valueMin.isFinite ? stats.valueMin : -1.0,
            max: stats.valueMax.isFinite ? stats.valueMax : 1.0,
            config: quantConfig
        )
    }

    /// Updates calibration statistics.
    public func updateStats(keyMin: Float, keyMax: Float, valueMin: Float, valueMax: Float) {
        stats.update(keyMin: keyMin, keyMax: keyMax, valueMin: valueMin, valueMax: valueMax)
    }

    /// Memory statistics.
    public var memoryStats: (usage: Int, savings: Double) {
        (config.memoryUsage, config.memorySavingsRatio)
    }
}

// MARK: - Fused Quantization Operations

/// Represents a fused quantization operation.
public enum FusedQuantOp: Sendable, Equatable {
    /// Quantize then matmul: Q(A) @ Q(B)
    case quantizedMatmul(aConfig: QuantizationConfig, bConfig: QuantizationConfig)

    /// Matmul then dequantize
    case matmulDequantize(outputConfig: QuantizationConfig)

    /// Full int8 matmul with dequantize
    case int8MatmulDequant

    /// Quantize, matmul, dequantize fused
    case quantMatmulDequant(aConfig: QuantizationConfig, bConfig: QuantizationConfig)

    /// Quantized attention (Q @ K^T @ V with quantized K, V)
    case quantizedAttention(kvConfig: QuantizedKVCacheConfig)

    /// Dequantize + activation fused
    case dequantActivation(activation: ActivationType)

    /// Activation type for fused ops.
    public enum ActivationType: String, Sendable {
        case relu
        case gelu
        case silu
        case none
    }
}

/// Pattern detector for fused quantization opportunities.
public final class FusedQuantPatternDetector: @unchecked Sendable {

    /// Detected pattern.
    public struct DetectedPattern: Sendable {
        /// Operations involved.
        public let operations: [Int]

        /// The fused operation type.
        public let fusedOp: FusedQuantOp

        /// Estimated speedup.
        public let estimatedSpeedup: Double

        /// Memory savings.
        public let memorySavings: Double
    }

    public init() {}

    /// Detects fused quantization patterns in a function.
    public func detectPatterns(_ function: HLOFunction) -> [DetectedPattern] {
        var patterns: [DetectedPattern] = []

        // Look for matmul chains that can be quantized
        for (i, op) in function.operations.enumerated() {
            if op.kind == .dot || op.kind == .dotGeneral {
                // Check if inputs could be quantized
                let pattern = DetectedPattern(
                    operations: [i],
                    fusedOp: .int8MatmulDequant,
                    estimatedSpeedup: 2.0,  // INT8 can be 2x faster
                    memorySavings: 0.75
                )
                patterns.append(pattern)
            }
        }

        return patterns
    }
}

// MARK: - Quantization-Aware Optimizer

/// Optimizer that applies quantization-aware transformations.
public final class QuantizationAwareOptimizer: @unchecked Sendable {

    /// Configuration for the optimizer.
    public struct Config: Sendable {
        /// Target operations for quantization.
        public var targetOps: Set<String>

        /// Default quantization config.
        public var defaultConfig: QuantizationConfig

        /// Mixed precision policy.
        public var mixedPrecisionPolicy: MixedPrecisionPolicy?

        /// Whether to analyze before optimizing.
        public var analyzeFirst: Bool

        /// Minimum memory savings to apply quantization.
        public var minMemorySavings: Double

        public init(
            targetOps: Set<String> = ["dot", "dot_general", "convolution"],
            defaultConfig: QuantizationConfig = .int8PerTensor,
            mixedPrecisionPolicy: MixedPrecisionPolicy? = nil,
            analyzeFirst: Bool = true,
            minMemorySavings: Double = 0.25
        ) {
            self.targetOps = targetOps
            self.defaultConfig = defaultConfig
            self.mixedPrecisionPolicy = mixedPrecisionPolicy
            self.analyzeFirst = analyzeFirst
            self.minMemorySavings = minMemorySavings
        }
    }

    private let config: Config
    private let analyzer: QuantizationAnalyzer
    private let patternDetector: FusedQuantPatternDetector

    public init(config: Config = Config()) {
        self.config = config
        self.analyzer = QuantizationAnalyzer()
        self.patternDetector = FusedQuantPatternDetector()
    }

    /// Optimizes a function with quantization-aware transformations.
    public func optimize(_ function: HLOFunction) -> QuantizationPlan {
        // Analyze function
        let analysis = config.analyzeFirst ? analyzer.analyze(function) : [:]

        // Detect fusion patterns
        let patterns = patternDetector.detectPatterns(function)

        // Build quantization plan
        var tensorConfigs: [String: QuantizationConfig] = [:]
        var fusedOps: [(operations: [Int], fusedOp: FusedQuantOp)] = []

        // Apply quantization to eligible tensors
        for (tensorId, tensorAnalysis) in analysis {
            if tensorAnalysis.canQuantize,
               tensorAnalysis.memorySavings >= config.minMemorySavings,
               let recommendedConfig = tensorAnalysis.recommendedConfig {
                tensorConfigs[tensorId] = recommendedConfig
            }
        }

        // Add fused operations
        for pattern in patterns {
            if pattern.memorySavings >= config.minMemorySavings {
                fusedOps.append((pattern.operations, pattern.fusedOp))
            }
        }

        // Compute overall memory savings
        let totalOriginalSize = function.operations.reduce(0) { $0 + $1.resultType.byteCount }
        var totalQuantizedSize = totalOriginalSize

        for (tensorId, quantConfig) in tensorConfigs {
            if let op = function.operations.first(where: { $0.result == tensorId }) {
                let originalSize = op.resultType.byteCount
                let quantizedSize = op.resultType.shape.reduce(1, *) * quantConfig.dtype.bytesPerElement
                totalQuantizedSize -= (originalSize - quantizedSize)
            }
        }

        let memorySavings = 1.0 - Double(totalQuantizedSize) / Double(totalOriginalSize)

        return QuantizationPlan(
            tensorConfigs: tensorConfigs,
            fusedOperations: fusedOps,
            mixedPrecisionPolicy: config.mixedPrecisionPolicy,
            estimatedMemorySavings: memorySavings,
            estimatedSpeedup: patterns.isEmpty ? 1.0 : patterns.map(\.estimatedSpeedup).reduce(1.0, *)
        )
    }
}

/// Plan for applying quantization to a function.
public struct QuantizationPlan: Sendable {
    /// Quantization config for each tensor.
    public let tensorConfigs: [String: QuantizationConfig]

    /// Fused operations to apply.
    public let fusedOperations: [(operations: [Int], fusedOp: FusedQuantOp)]

    /// Mixed precision policy.
    public let mixedPrecisionPolicy: MixedPrecisionPolicy?

    /// Estimated memory savings (0-1).
    public let estimatedMemorySavings: Double

    /// Estimated speedup factor (>1 = faster).
    public let estimatedSpeedup: Double

    /// Number of quantized tensors.
    public var quantizedTensorCount: Int {
        tensorConfigs.count
    }

    /// Number of fused operations.
    public var fusedOpCount: Int {
        fusedOperations.count
    }
}

// MARK: - Quantization Calibrator

/// Calibrates quantization parameters from sample data.
public final class QuantizationCalibrator: @unchecked Sendable {

    /// Calibration method.
    public enum Method: Sendable {
        /// Min-max calibration.
        case minMax

        /// Percentile-based calibration.
        case percentile(Float)

        /// Entropy-based calibration.
        case entropy

        /// Mean squared error minimization.
        case mse
    }

    private let method: Method
    private var tensorStats: [String: (min: Float, max: Float, samples: Int)]

    public init(method: Method = .minMax) {
        self.method = method
        self.tensorStats = [:]
    }

    /// Updates calibration with new sample.
    public func update(tensorId: String, min: Float, max: Float) {
        if var existing = tensorStats[tensorId] {
            existing.min = Swift.min(existing.min, min)
            existing.max = Swift.max(existing.max, max)
            existing.samples += 1
            tensorStats[tensorId] = existing
        } else {
            tensorStats[tensorId] = (min, max, 1)
        }
    }

    /// Gets calibrated parameters for a tensor.
    public func getParams(
        for tensorId: String,
        config: QuantizationConfig
    ) -> QuantizationParams? {
        guard let stats = tensorStats[tensorId] else { return nil }

        return QuantizationParams.fromStatistics(
            min: stats.min,
            max: stats.max,
            config: config
        )
    }

    /// Resets calibration data.
    public func reset() {
        tensorStats = [:]
    }

    /// Number of calibrated tensors.
    public var calibratedCount: Int {
        tensorStats.count
    }
}
