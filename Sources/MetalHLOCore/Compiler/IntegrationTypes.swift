// IntegrationTypes.swift
// MetalHLOCore
//
// Core data structures for the MetalHLO compilation pipeline.
// These types serve as contracts between pipeline stages.

import Foundation
import Metal

// MARK: - Type Aliases

/// Unique identifier for an operation in the computation graph.
public typealias OpID = String

// MARK: - Tensor Information

/// Complete information about a tensor.
public struct TensorInfo: Sendable, Hashable {
    /// Unique identifier.
    public let id: TensorID

    /// Shape dimensions.
    public let shape: [Int]

    /// Element data type.
    public let elementType: ElementType

    /// Size in bytes.
    public var byteSize: Int {
        shape.reduce(1, *) * elementType.byteSize
    }

    public init(id: TensorID, shape: [Int], elementType: ElementType) {
        self.id = id
        self.shape = shape
        self.elementType = elementType
    }

    public init(id: TensorID, type: TensorType) {
        self.id = id
        self.shape = type.shape
        self.elementType = type.elementType
    }
}

/// Tensor parameter (function input).
public struct TensorParam: Sendable, Hashable {
    /// Parameter name.
    public let name: String

    /// Parameter type.
    public let type: TensorType

    /// Unique identifier (same as name for inputs).
    public var id: TensorID { name }

    public init(name: String, type: TensorType) {
        self.name = name
        self.type = type
    }

    public init(from argument: HLOArgument) {
        self.name = argument.name
        self.type = argument.type
    }
}

// MARK: - Lifetime

/// Represents when a tensor is live during execution.
public struct Lifetime: Sendable, Hashable {
    /// Operation index where the tensor is created (-1 for inputs).
    public let defined: Int

    /// Operation index where the tensor is last read.
    public let lastUsed: Int

    public init(defined: Int, lastUsed: Int) {
        self.defined = defined
        self.lastUsed = lastUsed
    }

    /// Returns true if the tensor is live at the given operation index.
    public func isLiveAt(_ index: Int) -> Bool {
        index >= defined && index <= lastUsed
    }

    /// Returns true if this lifetime overlaps with another.
    public func overlaps(with other: Lifetime) -> Bool {
        defined <= other.lastUsed && other.defined <= lastUsed
    }
}

// MARK: - Analysis Results

/// Results from the analysis phase.
public struct AnalysisResults: Sendable {
    /// Inferred shapes for all tensors.
    public let shapes: [TensorID: [Int]]

    /// Dependencies: operation -> operations it depends on.
    public let dependencies: [OpID: Set<OpID>]

    /// Users: operation -> operations that use its output.
    public let users: [OpID: Set<OpID>]

    /// Tensor lifetimes.
    public let lifetimes: [TensorID: Lifetime]

    /// Detected high-level patterns.
    public let patterns: [DetectedPattern]

    /// Tensor element types.
    public let elementTypes: [TensorID: ElementType]

    public init(
        shapes: [TensorID: [Int]] = [:],
        dependencies: [OpID: Set<OpID>] = [:],
        users: [OpID: Set<OpID>] = [:],
        lifetimes: [TensorID: Lifetime] = [:],
        patterns: [DetectedPattern] = [],
        elementTypes: [TensorID: ElementType] = [:]
    ) {
        self.shapes = shapes
        self.dependencies = dependencies
        self.users = users
        self.lifetimes = lifetimes
        self.patterns = patterns
        self.elementTypes = elementTypes
    }

    /// Empty analysis results.
    public static let empty = AnalysisResults()
}

// MARK: - Detected Patterns

/// A detected high-level pattern in the computation graph.
public struct DetectedPattern: Sendable {
    /// Type of pattern detected.
    public let type: PatternType

    /// Operation indices involved in the pattern.
    public let operationIndices: [Int]

    /// Root operation index.
    public let rootIndex: Int

    /// Pattern-specific metadata.
    public let metadata: PatternMetadata

    public init(type: PatternType, operationIndices: [Int], rootIndex: Int, metadata: PatternMetadata = .init()) {
        self.type = type
        self.operationIndices = operationIndices
        self.rootIndex = rootIndex
        self.metadata = metadata
    }
}

/// Types of patterns that can be detected.
public enum PatternType: String, Sendable, CaseIterable {
    case attention
    case multiHeadAttention
    case flashAttention
    case ffn
    case gatedFFN
    case layerNorm
    case rmsNorm
    case softmax
    case gelu
    case silu
    case matmulBiasActivation
    case residualAdd
    case embeddingLookup
    case rotaryPositionEmbedding
    case transformerBlock
}

/// Metadata for detected patterns.
public struct PatternMetadata: Sendable {
    /// Number of attention heads (for attention patterns).
    public var numHeads: Int?

    /// Head dimension.
    public var headDim: Int?

    /// Hidden dimension.
    public var hiddenDim: Int?

    /// Whether the pattern uses causal masking.
    public var causalMask: Bool?

    /// Activation function type.
    public var activation: String?

    /// Epsilon for normalization.
    public var epsilon: Float?

    public init(
        numHeads: Int? = nil,
        headDim: Int? = nil,
        hiddenDim: Int? = nil,
        causalMask: Bool? = nil,
        activation: String? = nil,
        epsilon: Float? = nil
    ) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.causalMask = causalMask
        self.activation = activation
        self.epsilon = epsilon
    }
}

// MARK: - Fused Operation Types

/// Types of fused operations after optimization.
public enum FusedOpType: Sendable, Hashable {
    /// Original unfused operation.
    case original(HLOOpKind)

    /// Fused attention (Q @ K^T -> softmax -> @ V).
    case fusedAttention(AttentionConfig)

    /// Fused multi-head attention with projections.
    case fusedMultiHeadAttention(MultiHeadAttentionConfig)

    /// Fused RMS normalization.
    case fusedRMSNorm(NormConfig)

    /// Fused layer normalization.
    case fusedLayerNorm(NormConfig)

    /// Fused matmul + bias + activation.
    case fusedMatMulBiasAct(MatMulConfig)

    /// Fused GELU activation.
    case fusedGELU(approximate: Bool)

    /// Fused SiLU activation.
    case fusedSiLU

    /// Chain of fused elementwise operations.
    case fusedElementwise([HLOOpKind])

    /// Fused FFN block.
    case fusedFFN(FFNConfig)

    /// Fused transformer block (attention + FFN).
    case fusedTransformerBlock(TransformerBlockConfig)

    /// Fused rotary position embedding.
    case fusedRoPE(RoPEConfig)
}

/// Configuration for fused attention.
public struct AttentionConfig: Sendable, Hashable {
    public var numHeads: Int
    public var headDim: Int
    public var scale: Float?
    public var causalMask: Bool
    public var dropout: Float?

    public init(numHeads: Int, headDim: Int, scale: Float? = nil, causalMask: Bool = false, dropout: Float? = nil) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = scale
        self.causalMask = causalMask
        self.dropout = dropout
    }
}

/// Configuration for fused multi-head attention.
public struct MultiHeadAttentionConfig: Sendable, Hashable {
    public var numHeads: Int
    public var numKVHeads: Int
    public var headDim: Int
    public var hiddenDim: Int
    public var causalMask: Bool
    public var useRoPE: Bool

    public init(numHeads: Int, numKVHeads: Int? = nil, headDim: Int, hiddenDim: Int, causalMask: Bool = false, useRoPE: Bool = false) {
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads ?? numHeads
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.causalMask = causalMask
        self.useRoPE = useRoPE
    }
}

/// Configuration for fused normalization.
public struct NormConfig: Sendable, Hashable {
    public var epsilon: Float
    public var axis: Int
    public var elementwiseAffine: Bool

    public init(epsilon: Float = 1e-5, axis: Int = -1, elementwiseAffine: Bool = true) {
        self.epsilon = epsilon
        self.axis = axis
        self.elementwiseAffine = elementwiseAffine
    }
}

/// Configuration for fused matmul.
public struct MatMulConfig: Sendable, Hashable {
    public var transA: Bool
    public var transB: Bool
    public var hasBias: Bool
    public var activation: ActivationType?
    public var alpha: Float
    public var beta: Float

    public init(transA: Bool = false, transB: Bool = false, hasBias: Bool = false, activation: ActivationType? = nil, alpha: Float = 1.0, beta: Float = 0.0) {
        self.transA = transA
        self.transB = transB
        self.hasBias = hasBias
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
    }
}

/// Activation types for fused operations.
public enum ActivationType: String, Sendable, Hashable, CaseIterable {
    case relu
    case gelu
    case geluApproximate
    case silu
    case tanh
    case sigmoid
    case none
}

/// Configuration for fused FFN.
public struct FFNConfig: Sendable, Hashable {
    public var hiddenDim: Int
    public var intermediateDim: Int
    public var activation: ActivationType
    public var isGated: Bool

    public init(hiddenDim: Int, intermediateDim: Int, activation: ActivationType = .gelu, isGated: Bool = false) {
        self.hiddenDim = hiddenDim
        self.intermediateDim = intermediateDim
        self.activation = activation
        self.isGated = isGated
    }
}

/// Configuration for fused transformer block.
public struct TransformerBlockConfig: Sendable, Hashable {
    public var attention: MultiHeadAttentionConfig
    public var ffn: FFNConfig
    public var preNorm: Bool
    public var normType: NormType

    public enum NormType: String, Sendable, Hashable {
        case layerNorm
        case rmsNorm
    }

    public init(attention: MultiHeadAttentionConfig, ffn: FFNConfig, preNorm: Bool = true, normType: NormType = .rmsNorm) {
        self.attention = attention
        self.ffn = ffn
        self.preNorm = preNorm
        self.normType = normType
    }
}

/// Configuration for rotary position embedding.
public struct RoPEConfig: Sendable, Hashable {
    public var dim: Int
    public var maxSeqLen: Int
    public var base: Float

    public init(dim: Int, maxSeqLen: Int = 2048, base: Float = 10000.0) {
        self.dim = dim
        self.maxSeqLen = maxSeqLen
        self.base = base
    }
}

// MARK: - Fused Operation

/// An operation after fusion optimization.
public struct FusedOp: Sendable {
    /// Unique identifier.
    public let id: OpID

    /// Fused operation type.
    public let type: FusedOpType

    /// Input tensor IDs.
    public let inputs: [TensorID]

    /// Output tensor information.
    public let outputs: [TensorInfo]

    /// Original operations that were fused (for debugging).
    public let originalOps: [OpID]?

    /// HLO attributes (preserved from original).
    public let attributes: HLOAttributes

    public init(
        id: OpID,
        type: FusedOpType,
        inputs: [TensorID],
        outputs: [TensorInfo],
        originalOps: [OpID]? = nil,
        attributes: HLOAttributes = HLOAttributes()
    ) {
        self.id = id
        self.type = type
        self.inputs = inputs
        self.outputs = outputs
        self.originalOps = originalOps
        self.attributes = attributes
    }

    /// Creates a FusedOp from an original HLO operation.
    public init(from operation: HLOOperation) {
        self.id = operation.result
        self.type = .original(operation.kind)
        self.inputs = operation.operands
        self.outputs = [TensorInfo(id: operation.result, type: operation.resultType)]
        self.originalOps = nil
        self.attributes = operation.attributes
    }

    /// Primary output tensor ID.
    public var primaryOutput: TensorID {
        outputs.first?.id ?? id
    }
}

// MARK: - Optimized Module

/// The result of the optimization phase.
public struct OptimizedModule: Sendable {
    /// Operations after fusion (fewer than original).
    public let operations: [FusedOp]

    /// Input parameters (unchanged from original).
    public let inputs: [TensorParam]

    /// Output tensor IDs (unchanged from original).
    public let outputs: [TensorID]

    /// All tensor metadata.
    public let tensors: [TensorID: TensorInfo]

    /// Memory layouts for tensors.
    public let layouts: [TensorID: TensorLayout]

    /// Optimized execution order.
    public let schedule: [OpID]

    /// Analysis results used during optimization.
    public let analysis: AnalysisResults

    /// Constants embedded in the module (tensor ID -> constant value).
    public let constants: [TensorID: ConstantValue]

    public init(
        operations: [FusedOp],
        inputs: [TensorParam],
        outputs: [TensorID],
        tensors: [TensorID: TensorInfo],
        layouts: [TensorID: TensorLayout] = [:],
        schedule: [OpID]? = nil,
        analysis: AnalysisResults = .empty,
        constants: [TensorID: ConstantValue] = [:]
    ) {
        self.operations = operations
        self.inputs = inputs
        self.outputs = outputs
        self.tensors = tensors
        self.layouts = layouts
        self.schedule = schedule ?? operations.map { $0.id }
        self.analysis = analysis
        self.constants = constants
    }

    /// Creates an OptimizedModule from an HLO function (no optimization).
    public static func from(_ function: HLOFunction) -> OptimizedModule {
        let fusedOps = function.operations.map { FusedOp(from: $0) }
        let inputs = function.inputs.map { TensorParam(from: $0) }

        var tensors: [TensorID: TensorInfo] = [:]
        for input in function.inputs {
            tensors[input.name] = TensorInfo(id: input.name, type: input.type)
        }
        for op in function.operations {
            tensors[op.result] = TensorInfo(id: op.result, type: op.resultType)
        }

        // Extract constants from the function
        var constants: [TensorID: ConstantValue] = [:]
        for op in function.operations {
            if case .constant = op.kind,
               let constantValue = op.attributes.constantValue {
                constants[op.result] = constantValue
            }
        }

        return OptimizedModule(
            operations: fusedOps,
            inputs: inputs,
            outputs: function.returnValues,
            tensors: tensors,
            constants: constants
        )
    }
}
