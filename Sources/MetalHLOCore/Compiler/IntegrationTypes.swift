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
        let count = shape.reduce(1, *)
        if elementType == .int1 {
            return count  // 1 byte per bool for storage
        }
        return count * elementType.byteSize
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
    case batchNorm
    case softmax
    case gelu
    case silu
    case matmulBiasActivation
    case convBiasActivation
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

    /// Reduction axis (for softmax patterns). Negative values are interpreted
    /// relative to the input rank.
    public var axis: Int?

    /// Explicit operand tensors for norm fusion (avoids the brittle heuristic
    /// extraction in NormFusionPass — JAX folds gamma into the rsqrt scale, so
    /// scanning for "the last multiply" picks the wrong tensor).
    public var inputTensor: TensorID?
    public var gammaTensor: TensorID?
    public var betaTensor: TensorID?
    /// SSA name of the pattern's root (output) op. Robust to index shifts —
    /// the integer rootIndex goes stale across passes.
    public var outputTensor: TensorID?

    public init(
        numHeads: Int? = nil,
        headDim: Int? = nil,
        hiddenDim: Int? = nil,
        causalMask: Bool? = nil,
        activation: String? = nil,
        epsilon: Float? = nil,
        axis: Int? = nil,
        inputTensor: TensorID? = nil,
        gammaTensor: TensorID? = nil,
        betaTensor: TensorID? = nil,
        outputTensor: TensorID? = nil
    ) {
        self.outputTensor = outputTensor
        self.numHeads = numHeads
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.causalMask = causalMask
        self.activation = activation
        self.epsilon = epsilon
        self.axis = axis
        self.inputTensor = inputTensor
        self.gammaTensor = gammaTensor
        self.betaTensor = betaTensor
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

    /// Fused batch normalization (training-mode apply chain).
    case fusedBatchNorm(NormConfig)

    /// Fused matmul + bias + activation.
    case fusedMatMulBiasAct(MatMulConfig)

    /// Fused 2D convolution + bias + activation.
    case fusedConvBiasAct(ConvBiasActConfig)

    /// Fused GELU activation.
    case fusedGELU(approximate: Bool)

    /// Fused SiLU activation.
    case fusedSiLU

    /// Chain of fused elementwise operations.
    ///
    /// Carries a `FusedElementwiseChain` that records, for each chain op,
    /// exactly where its operands come from — an earlier chain result or a
    /// specific external input slot. The codegen walks the chain and emits
    /// one temporary per op, so the kernel can express an arbitrary DAG of
    /// elementwise + scalar broadcast + reshape (not just a strict left-to-
    /// right linear chain). That's what unlocks fusing the bulk of nanoGPT-
    /// class workloads' elementwise GPU time.
    case fusedElementwise(FusedElementwiseChain)

    /// Fused reduce: a pointwise transform chain applied per input element,
    /// then reduced. Captures the MLX single-pass layernorm/softmax pattern
    /// (e.g. `reduce(sum, multiply(x, x))`). The chain's externals are the
    /// op's operands; reduce metadata (kind, dims, pre-reduction shape) is
    /// baked into the kernel so binding stays identical to fusedElementwise.
    case fusedReduce(FusedReduceConfig)

    /// Fused FFN block.
    case fusedFFN(FFNConfig)

    /// Fused transformer block (attention + FFN).
    case fusedTransformerBlock(TransformerBlockConfig)

    /// Fused rotary position embedding.
    case fusedRoPE(RoPEConfig)

    /// Fused softmax (numerically stable: exp(x - max) / sum(exp(x - max))).
    case fusedSoftmax(axis: Int)
}

/// SSA-style description of a fused elementwise chain.
///
/// Each op declares where its operands come from: either an external input
/// (an entry in the customCall's `inputs` array) or an earlier op's result
/// within the same chain. The codegen then emits one `float v_k = ...;`
/// per op and writes the last op's result to `output[tid]`.
///
/// Scalar inputs are detected from the external input's shape at codegen
/// time (element count == 1 ⇒ read `input[0]`, otherwise `input[tid]`).
/// Reshape ops are no-ops in linear-index space and emit `float v_k = src;`.
public struct FusedElementwiseChain: Sendable, Hashable {
    public let ops: [Op]

    public init(ops: [Op]) { self.ops = ops }

    public struct Op: Sendable, Hashable {
        public let kind: HLOOpKind
        public let operands: [OperandSource]
        /// For `.transpose`: the input→output dim permutation (output dim
        /// i comes from input dim `dimensions[i]`).
        /// For `.broadcastInDim`: `broadcast_dimensions` — input dim d
        /// maps to output dim `dimensions[d]`. Other output dims are
        /// "broadcast" (input has size 1 there or no corresponding dim).
        public let dimensions: [Int]?
        /// For `.broadcastInDim`: the broadcast's output shape. Required
        /// because the output shape is encoded in the result type and not
        /// derivable from input shape + dimensions alone (size-1 input dims
        /// can fan out to any output size). For other ops where the output
        /// shape can be derived from the inputs, this is nil.
        public let outputShape: [Int]?
        /// For `.compare`: the comparison direction raw value (eq/ne/lt/le/gt/ge).
        /// The chain emits `(a OP b) ? 1.0f : 0.0f` in-register, so a downstream
        /// in-chain `.select` reads it as a float predicate (no bool buffer).
        public let comparison: String?
        public init(kind: HLOOpKind,
                    operands: [OperandSource],
                    dimensions: [Int]? = nil,
                    outputShape: [Int]? = nil,
                    comparison: String? = nil) {
            self.kind = kind
            self.operands = operands
            self.dimensions = dimensions
            self.outputShape = outputShape
            self.comparison = comparison
        }
    }

    public enum OperandSource: Sendable, Hashable {
        /// The customCall's `inputs[idx]` — read as `inputIdx[tid]` for
        /// full-shape tensors or `inputIdx[0]` for scalar (1-element) inputs.
        case external(Int)
        /// The result produced by an earlier op in the chain at index `idx`.
        case prior(Int)
    }
}

/// Configuration for a fused reduce (pointwise transform chain + reduction).
public struct FusedReduceConfig: Sendable, Hashable {
    /// Pointwise transform applied per input element (no transpose/broadcast).
    public let chain: FusedElementwiseChain
    /// Reduction kind as a raw string ("sum","max","min","mean","product").
    public let reductionKind: String
    /// Reduce dimensions (into the pre-reduction input shape).
    public let reduceDims: [Int]
    /// Pre-reduction input shape (= the chain's per-element output shape).
    public let inputShape: [Int]
    /// Output (post-reduction) shape.
    public let outputShape: [Int]
    public init(chain: FusedElementwiseChain, reductionKind: String,
                reduceDims: [Int], inputShape: [Int], outputShape: [Int]) {
        self.chain = chain
        self.reductionKind = reductionKind
        self.reduceDims = reduceDims
        self.inputShape = inputShape
        self.outputShape = outputShape
    }
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

/// Configuration for fused conv + bias + activation.
public struct ConvBiasActConfig: Sendable, Hashable {
    public var hasBias: Bool
    public var activation: ActivationType

    public init(hasBias: Bool = false, activation: ActivationType = .none) {
        self.hasBias = hasBias
        self.activation = activation
    }
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
        self.inputs = operation.operands
        self.outputs = [TensorInfo(id: operation.result, type: operation.resultType)]
        self.originalOps = nil
        self.attributes = operation.attributes

        // Handle custom_call operations by mapping to appropriate FusedOpType
        if operation.kind == .customCall, let targetName = operation.attributes.callTargetName {
            self.type = FusedOp.fusedOpTypeFromCustomCall(targetName: targetName, backendConfig: operation.attributes.backendConfig)
        } else {
            self.type = .original(operation.kind)
        }
    }

    /// Maps a custom_call target name to a FusedOpType.
    private static func fusedOpTypeFromCustomCall(targetName: String, backendConfig: String?) -> FusedOpType {
        // Parse backend config if available
        let config: [String: Any]
        if let configStr = backendConfig,
           let data = configStr.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            config = parsed
        } else {
            config = [:]
        }

        switch targetName {
        case FusedScaledDotProductAttentionHandler.targetName:
            let scale = (config["scale"] as? NSNumber)?.floatValue
            let isCausal = (config["is_causal"] as? Bool) ?? false
            return .fusedAttention(AttentionConfig(
                numHeads: (config["num_heads"] as? Int) ?? 1,
                headDim: (config["head_dim"] as? Int) ?? 64,
                scale: scale,
                causalMask: isCausal
            ))

        case FusedLayerNormHandler.targetName:
            let eps = (config["eps"] as? NSNumber)?.floatValue ?? 1e-5
            return .fusedLayerNorm(NormConfig(epsilon: eps, axis: -1))

        case FusedRMSNormHandler.targetName:
            let eps = (config["eps"] as? NSNumber)?.floatValue ?? 1e-5
            return .fusedRMSNorm(NormConfig(epsilon: eps, axis: -1))

        case FusedBatchNormHandler.targetName:
            let eps = (config["eps"] as? NSNumber)?.floatValue ?? 1e-5
            return .fusedBatchNorm(NormConfig(epsilon: eps, axis: -1))

        case FusedGeluHandler.targetName:
            let approximate = (config["approximate"] as? Bool) ?? true
            return .fusedGELU(approximate: approximate)

        case FusedMatMulBiasActivationHandler.targetName:
            let activationStr = config["activation"] as? String ?? "none"
            let transA = (config["trans_a"] as? Bool) ?? false
            let transB = (config["trans_b"] as? Bool) ?? false
            let activation = ActivationType(rawValue: activationStr) ?? .none
            return .fusedMatMulBiasAct(MatMulConfig(
                transA: transA,
                transB: transB,
                hasBias: true,
                activation: activation
            ))

        case FusedConvBiasActHandler.targetName:
            let activationStr = config["activation"] as? String ?? "none"
            let hasBias = (config["has_bias"] as? Bool) ?? true
            let activation = ActivationType(rawValue: activationStr) ?? .none
            return .fusedConvBiasAct(ConvBiasActConfig(
                hasBias: hasBias,
                activation: activation
            ))

        case "fused_elementwise":
            // The producer-consumer-fusion pass serializes the chain as
            //   {"chain": [
            //     {"k": "<HLOOpKind raw>", "o": [["e", idx] | ["p", idx], ...]},
            //     ...
            //   ]}
            // where "e" tags an external input slot and "p" tags a prior
            // chain-op result. The shape is intentionally compact so the
            // backend config stays short for long chains.
            if let chainArr = config["chain"] as? [[String: Any]] {
                var ops: [FusedElementwiseChain.Op] = []
                ops.reserveCapacity(chainArr.count)
                for opDict in chainArr {
                    guard let kindStr = opDict["k"] as? String,
                          let kind = HLOOpKind(rawValue: kindStr),
                          let operandsArr = opDict["o"] as? [[Any]] else {
                        return .original(.customCall)
                    }
                    var operands: [FusedElementwiseChain.OperandSource] = []
                    operands.reserveCapacity(operandsArr.count)
                    for src in operandsArr {
                        guard src.count == 2, let tag = src[0] as? String,
                              let idx = (src[1] as? Int) ?? (src[1] as? NSNumber).map({ $0.intValue })
                        else { return .original(.customCall) }
                        switch tag {
                        case "e": operands.append(.external(idx))
                        case "p": operands.append(.prior(idx))
                        default:  return .original(.customCall)
                        }
                    }
                    let dimensions: [Int]? = (opDict["t"] as? [Any])?.compactMap {
                        ($0 as? Int) ?? ($0 as? NSNumber).map { $0.intValue }
                    }
                    let outputShape: [Int]? = (opDict["s"] as? [Any])?.compactMap {
                        ($0 as? Int) ?? ($0 as? NSNumber).map { $0.intValue }
                    }
                    let comparison = opDict["c"] as? String
                    ops.append(FusedElementwiseChain.Op(kind: kind, operands: operands, dimensions: dimensions, outputShape: outputShape, comparison: comparison))
                }
                if !ops.isEmpty {
                    return .fusedElementwise(FusedElementwiseChain(ops: ops))
                }
            }
            // Legacy: just a list of op kinds → reconstruct a strict linear
            // chain (op[i] consumes prior(i-1) + external(i), op[0] consumes
            // external(0) + external(1) for binary, external(0) for unary).
            let opsArray: [String]? = (config["ops"] as? [String]) ?? (config["operations"] as? [String])
            if let ops = opsArray {
                let opKinds = ops.compactMap { HLOOpKind(rawValue: $0) }
                if !opKinds.isEmpty {
                    return .fusedElementwise(legacyLinearChain(from: opKinds))
                }
            }
            return .original(.customCall)

        case "fused_reduce":
            // {"chain": [...same as fused_elementwise...], "rk": "sum",
            //  "rd": [dims], "is": [inputShape], "os": [outputShape]}
            guard let chain = parseChain(config["chain"]),
                  let rk = config["rk"] as? String,
                  let rd = intArray(config["rd"]),
                  let ishape = intArray(config["is"]),
                  let oshape = intArray(config["os"]) else {
                return .original(.customCall)
            }
            return .fusedReduce(FusedReduceConfig(
                chain: chain, reductionKind: rk,
                reduceDims: rd, inputShape: ishape, outputShape: oshape))

        case FusedRoPEHandler.targetName:
            let baseFreq = (config["base_freq"] as? NSNumber)?.floatValue ?? 10000.0
            let dim = (config["dim"] as? Int) ?? 64
            return .fusedRoPE(RoPEConfig(dim: dim, maxSeqLen: 4096, base: baseFreq))

        case "fused_transformer_block":
            // Parse transformer block configuration
            let numHeads = (config["num_heads"] as? Int) ?? 8
            let headDim = (config["head_dim"] as? Int) ?? 64
            let hiddenDim = (config["hidden_dim"] as? Int) ?? numHeads * headDim
            let intermediateDim = (config["intermediate_dim"] as? Int) ?? hiddenDim * 4
            let preNorm = (config["pre_norm"] as? Bool) ?? true
            let normTypeStr = (config["norm_type"] as? String) ?? "rmsNorm"
            let normType: TransformerBlockConfig.NormType = normTypeStr == "layerNorm" ? .layerNorm : .rmsNorm

            let attentionConfig = MultiHeadAttentionConfig(
                numHeads: numHeads,
                headDim: headDim,
                hiddenDim: hiddenDim,
                causalMask: (config["causal_mask"] as? Bool) ?? false
            )
            let ffnConfig = FFNConfig(
                hiddenDim: hiddenDim,
                intermediateDim: intermediateDim,
                activation: .gelu
            )
            return .fusedTransformerBlock(TransformerBlockConfig(
                attention: attentionConfig,
                ffn: ffnConfig,
                preNorm: preNorm,
                normType: normType
            ))

        case "fused_residual_chain":
            // Residual chains are essentially elementwise adds.
            // Reconstruct as a single-op linear chain: op[0] = add(external(0), external(1)).
            return .fusedElementwise(legacyLinearChain(from: [.add]))

        case FusedSoftmaxHandler.targetName:
            let axis = (config["axis"] as? Int) ?? -1
            return .fusedSoftmax(axis: axis)

        case FusedFFNHandler.targetName:
            // Parse FFN configuration
            let hiddenDim = (config["hidden_dim"] as? Int) ?? 0
            let activationStr = (config["activation"] as? String) ?? "gelu"
            let isGated = (config["is_gated"] as? Bool) ?? false
            let activation = ActivationType(rawValue: activationStr) ?? .gelu
            return .fusedFFN(FFNConfig(
                hiddenDim: hiddenDim,
                intermediateDim: hiddenDim * 4,  // Default intermediate ratio
                activation: activation,
                isGated: isGated
            ))

        default:
            // Unknown custom call, keep as original
            return .original(.customCall)
        }
    }

    /// Reconstructs a strict-linear-chain `FusedElementwiseChain` from the
    /// legacy `[HLOOpKind]` serialization. Each binary op consumes the prior
    /// result (if any) on the left and a fresh external on the right; each
    /// unary consumes either the prior result or a fresh external when no
    /// prior result exists. Matches the original codegen's input-walking
    /// semantics so older serialized executables still load.
    /// Parses a `{"chain": [...]}` value (the SSA-style op list shared by
    /// fused_elementwise and fused_reduce) into a FusedElementwiseChain.
    private static func parseChain(_ value: Any?) -> FusedElementwiseChain? {
        guard let chainArr = value as? [[String: Any]] else { return nil }
        var ops: [FusedElementwiseChain.Op] = []
        ops.reserveCapacity(chainArr.count)
        for opDict in chainArr {
            guard let kindStr = opDict["k"] as? String,
                  let kind = HLOOpKind(rawValue: kindStr),
                  let operandsArr = opDict["o"] as? [[Any]] else { return nil }
            var operands: [FusedElementwiseChain.OperandSource] = []
            for src in operandsArr {
                guard src.count == 2, let tag = src[0] as? String,
                      let idx = (src[1] as? Int) ?? (src[1] as? NSNumber).map({ $0.intValue })
                else { return nil }
                switch tag {
                case "e": operands.append(.external(idx))
                case "p": operands.append(.prior(idx))
                default:  return nil
                }
            }
            let dims = intArray(opDict["t"])
            let outShape = intArray(opDict["s"])
            let comparison = opDict["c"] as? String
            ops.append(FusedElementwiseChain.Op(kind: kind, operands: operands, dimensions: dims, outputShape: outShape, comparison: comparison))
        }
        return ops.isEmpty ? nil : FusedElementwiseChain(ops: ops)
    }

    /// Parses a JSON array of ints (tolerating NSNumber boxing).
    private static func intArray(_ value: Any?) -> [Int]? {
        guard let arr = value as? [Any] else { return nil }
        return arr.compactMap { ($0 as? Int) ?? ($0 as? NSNumber).map { $0.intValue } }
    }

    private static func legacyLinearChain(from kinds: [HLOOpKind]) -> FusedElementwiseChain {
        var ops: [FusedElementwiseChain.Op] = []
        ops.reserveCapacity(kinds.count)
        var nextExternal = 0
        for (i, kind) in kinds.enumerated() {
            let isBinary: Bool
            switch kind {
            case .add, .subtract, .multiply, .divide, .remainder, .maximum, .minimum, .power,
                 .and, .or, .xor, .shiftLeft, .shiftRightArithmetic, .shiftRightLogical:
                isBinary = true
            default:
                isBinary = false
            }
            var operands: [FusedElementwiseChain.OperandSource] = []
            if isBinary {
                if i == 0 {
                    operands.append(.external(nextExternal)); nextExternal += 1
                    operands.append(.external(nextExternal)); nextExternal += 1
                } else {
                    operands.append(.prior(i - 1))
                    operands.append(.external(nextExternal)); nextExternal += 1
                }
            } else {
                if i == 0 {
                    operands.append(.external(nextExternal)); nextExternal += 1
                } else {
                    operands.append(.prior(i - 1))
                }
            }
            ops.append(FusedElementwiseChain.Op(kind: kind, operands: operands))
        }
        return FusedElementwiseChain(ops: ops)
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
