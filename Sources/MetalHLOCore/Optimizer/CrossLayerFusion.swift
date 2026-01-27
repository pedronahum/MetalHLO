// CrossLayerFusion.swift
// MetalHLOCore
//
// Cross-layer fusion pass for global optimization beyond function boundaries.

import Foundation

// MARK: - Model Structure Analysis

/// High-level structure of a neural network model.
public struct ModelStructure: Sendable {
    /// Detected transformer layers.
    public var transformerLayers: [TransformerLayer] = []

    /// Detected attention blocks.
    public var attentionBlocks: [AttentionBlock] = []

    /// Detected FFN blocks.
    public var ffnBlocks: [FFNBlock] = []

    /// Detected normalization operations.
    public var normBlocks: [NormBlock] = []

    /// Residual connections.
    public var residualConnections: [ResidualConnection] = []

    /// KV cache operations.
    public var kvCacheOps: [KVCacheOp] = []
}

/// A detected transformer layer in the model.
public struct TransformerLayer: Sendable {
    public let index: Int
    public let preAttentionNorm: Int?  // Operation index
    public let attention: AttentionBlock?
    public let postAttentionResidual: Int?
    public let preFFNNorm: Int?
    public let ffn: FFNBlock?
    public let postFFNResidual: Int?
    public let operationIndices: [Int]
}

/// A detected attention block.
public struct AttentionBlock: Sendable {
    public let startIndex: Int
    public let endIndex: Int
    public let qProjection: Int?
    public let kProjection: Int?
    public let vProjection: Int?
    public let sdpa: Int?  // Scaled dot-product attention
    public let outputProjection: Int?
    public let hasKVCache: Bool
}

/// A detected FFN block.
public struct FFNBlock: Sendable {
    public let startIndex: Int
    public let endIndex: Int
    public let upProjection: Int?
    public let gateProjection: Int?  // For gated FFN
    public let activation: Int?
    public let downProjection: Int?
    public let isGated: Bool
}

/// A detected normalization block.
public struct NormBlock: Sendable {
    public let operationIndex: Int
    public let isRMSNorm: Bool
    public let hiddenDim: Int
}

/// A detected residual connection.
public struct ResidualConnection: Sendable {
    public let addOpIndex: Int
    public let skipFromIndex: Int
    public let skipToIndex: Int
}

/// A KV cache operation.
public struct KVCacheOp: Sendable {
    public let operationIndex: Int
    public let isRead: Bool
    public let layerIndex: Int
}

// MARK: - Cross-Layer Match

/// A match of a cross-layer pattern.
public struct CrossLayerMatch: Sendable {
    /// Operation indices that matched.
    public let operationIndices: [Int]

    /// Pattern type.
    public let patternType: CrossLayerPatternType

    /// Metadata about the match.
    public let metadata: [String: Int]

    public init(operationIndices: [Int], patternType: CrossLayerPatternType, metadata: [String: Int] = [:]) {
        self.operationIndices = operationIndices
        self.patternType = patternType
        self.metadata = metadata
    }
}

/// Types of cross-layer patterns.
public enum CrossLayerPatternType: String, Sendable {
    case transformerBlock
    case attentionFFN
    case residualChain
    case consecutiveNorms
    case embeddingPlusFirstLayer
    case lastLayerPlusOutput
    case kvCacheUpdate
    case multiLayerPipeline
}

// MARK: - Cross-Layer Fusion Pass

/// Optimization pass that fuses operations across traditional layer boundaries.
///
/// This pass can:
/// - Fuse entire transformer blocks into mega-kernels
/// - Combine consecutive normalizations
/// - Optimize KV cache updates
/// - Enable multi-layer pipelining
public final class CrossLayerFusionPass: @unchecked Sendable {

    public let name = "cross-layer-fusion"

    /// Configuration for the pass.
    public struct Config: Sendable {
        /// Enable transformer block fusion.
        public var enableBlockFusion: Bool = true

        /// Enable residual chain optimization.
        public var enableResidualChainFusion: Bool = true

        /// Enable KV cache optimization.
        public var enableKVCacheOptimization: Bool = true

        /// Enable multi-layer pipelining.
        public var enableMultiLayerPipelining: Bool = false

        /// Maximum layers to pipeline together.
        public var pipelineDepth: Int = 2

        /// Minimum operations to consider for fusion.
        public var minOpsForFusion: Int = 5

        public init() {}
    }

    private let config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    /// Runs the cross-layer fusion pass.
    ///
    /// - Parameter function: The function to optimize.
    /// - Returns: The optimized function.
    public func run(on function: HLOFunction) -> HLOFunction {
        // 1. Analyze model structure
        let structure = analyzeStructure(function)

        // 2. Find fusion opportunities
        var matches: [CrossLayerMatch] = []

        if config.enableBlockFusion {
            matches.append(contentsOf: findTransformerBlockMatches(structure, function: function))
        }

        if config.enableResidualChainFusion {
            matches.append(contentsOf: findResidualChainMatches(structure, function: function))
        }

        if config.enableKVCacheOptimization {
            matches.append(contentsOf: findKVCacheMatches(structure, function: function))
        }

        // 3. Apply fusions (non-overlapping)
        var current = function
        let sortedMatches = sortAndFilterMatches(matches)

        for match in sortedMatches {
            current = applyFusion(match, to: current)
        }

        return current
    }

    // MARK: - Structure Analysis

    private func analyzeStructure(_ function: HLOFunction) -> ModelStructure {
        var structure = ModelStructure()

        // Find normalization operations
        for (index, op) in function.operations.enumerated() {
            if isNormOperation(op) {
                let normBlock = NormBlock(
                    operationIndex: index,
                    isRMSNorm: isRMSNorm(op),
                    hiddenDim: extractHiddenDim(op)
                )
                structure.normBlocks.append(normBlock)
            }
        }

        // Find attention blocks
        structure.attentionBlocks = findAttentionBlocks(function)

        // Find FFN blocks
        structure.ffnBlocks = findFFNBlocks(function)

        // Find residual connections
        structure.residualConnections = findResidualConnections(function)

        // Detect transformer layers
        structure.transformerLayers = detectTransformerLayers(
            attentionBlocks: structure.attentionBlocks,
            ffnBlocks: structure.ffnBlocks,
            normBlocks: structure.normBlocks,
            residuals: structure.residualConnections,
            function: function
        )

        return structure
    }

    private func isNormOperation(_ op: HLOOperation) -> Bool {
        // Check for LayerNorm or RMSNorm patterns
        if case .customCall = op.kind {
            if let target = op.attributes.callTargetName {
                return target.contains("layer_norm") || target.contains("rms_norm")
            }
        }

        // Check for manual norm pattern (reduce + rsqrt + mul)
        return false
    }

    private func isRMSNorm(_ op: HLOOperation) -> Bool {
        if case .customCall = op.kind {
            if let target = op.attributes.callTargetName {
                return target.contains("rms_norm")
            }
        }
        return false
    }

    private func extractHiddenDim(_ op: HLOOperation) -> Int {
        return op.resultType.shape.last ?? 768
    }

    private func findAttentionBlocks(_ function: HLOFunction) -> [AttentionBlock] {
        var blocks: [AttentionBlock] = []

        // Look for attention patterns: Q @ K^T -> softmax -> @ V
        var i = 0
        while i < function.operations.count {
            if let block = matchAttentionBlock(starting: i, in: function) {
                blocks.append(block)
                i = block.endIndex + 1
            } else {
                i += 1
            }
        }

        return blocks
    }

    private func matchAttentionBlock(starting index: Int, in function: HLOFunction) -> AttentionBlock? {
        // Simplified pattern matching for attention
        // Real implementation would be more sophisticated

        guard index + 4 < function.operations.count else { return nil }

        let ops = function.operations

        // Look for dot (Q @ K^T), softmax, dot (@ V) pattern
        var dotCount = 0
        var softmaxFound = false
        var endIndex = index

        for j in index..<min(index + 20, ops.count) {
            let op = ops[j]
            if case .dot = op.kind {
                dotCount += 1
            }
            if case .customCall = op.kind {
                if let target = op.attributes.callTargetName,
                   target.contains("softmax") {
                    softmaxFound = true
                }
            }
            if dotCount >= 2 && softmaxFound {
                endIndex = j
                break
            }
        }

        if dotCount >= 2 && softmaxFound {
            return AttentionBlock(
                startIndex: index,
                endIndex: endIndex,
                qProjection: index,
                kProjection: index + 1,
                vProjection: index + 2,
                sdpa: nil,
                outputProjection: endIndex,
                hasKVCache: false
            )
        }

        return nil
    }

    private func findFFNBlocks(_ function: HLOFunction) -> [FFNBlock] {
        var blocks: [FFNBlock] = []

        // Look for FFN patterns: linear -> activation -> linear
        var i = 0
        while i < function.operations.count {
            if let block = matchFFNBlock(starting: i, in: function) {
                blocks.append(block)
                i = block.endIndex + 1
            } else {
                i += 1
            }
        }

        return blocks
    }

    private func matchFFNBlock(starting index: Int, in function: HLOFunction) -> FFNBlock? {
        guard index + 2 < function.operations.count else { return nil }

        let ops = function.operations

        // Simple pattern: dot -> activation -> dot
        if case .dot = ops[index].kind {
            // Look for activation followed by another dot
            for j in (index + 1)..<min(index + 5, ops.count) {
                if isActivation(ops[j]) {
                    for k in (j + 1)..<min(j + 3, ops.count) {
                        if case .dot = ops[k].kind {
                            return FFNBlock(
                                startIndex: index,
                                endIndex: k,
                                upProjection: index,
                                gateProjection: nil,
                                activation: j,
                                downProjection: k,
                                isGated: false
                            )
                        }
                    }
                }
            }
        }

        return nil
    }

    private func isActivation(_ op: HLOOperation) -> Bool {
        switch op.kind {
        case .tanh, .exponential, .log:
            return true
        case .customCall:
            if let target = op.attributes.callTargetName {
                return target.contains("gelu") || target.contains("silu") ||
                       target.contains("relu") || target.contains("swish")
            }
            return false
        default:
            return false
        }
    }

    private func findResidualConnections(_ function: HLOFunction) -> [ResidualConnection] {
        var connections: [ResidualConnection] = []

        // Build a map from result name to operation index
        var resultToIndex: [String: Int] = [:]
        for input in function.inputs {
            resultToIndex[input.name] = -1  // Inputs are "before" all operations
        }
        for (index, op) in function.operations.enumerated() {
            resultToIndex[op.result] = index
        }

        for (index, op) in function.operations.enumerated() {
            if case .add = op.kind, op.operands.count >= 2 {
                // Find the source indices of both operands
                let operand0 = op.operands[0]
                let operand1 = op.operands[1]

                let idx0 = resultToIndex[operand0] ?? -1
                let idx1 = resultToIndex[operand1] ?? -1

                // A residual connection typically has one operand from much earlier
                // (the skip connection) and one from just before (the block output)
                let minIdx = min(idx0, idx1)
                let maxIdx = max(idx0, idx1)

                // Consider it a residual if the skip distance is significant (> 2 ops)
                // or if one operand is an input (-1)
                if minIdx < maxIdx - 2 || minIdx == -1 {
                    connections.append(ResidualConnection(
                        addOpIndex: index,
                        skipFromIndex: minIdx,
                        skipToIndex: maxIdx
                    ))
                }
            }
        }

        return connections
    }

    private func detectTransformerLayers(
        attentionBlocks: [AttentionBlock],
        ffnBlocks: [FFNBlock],
        normBlocks: [NormBlock],
        residuals: [ResidualConnection],
        function: HLOFunction
    ) -> [TransformerLayer] {
        var layers: [TransformerLayer] = []

        // Pair attention blocks with FFN blocks
        for (i, attn) in attentionBlocks.enumerated() {
            // Find FFN that comes after this attention
            if let ffn = ffnBlocks.first(where: { $0.startIndex > attn.endIndex }) {
                // Find norms before attention and FFN
                let preAttnNorm = normBlocks.last(where: { $0.operationIndex < attn.startIndex })
                let preFFNNorm = normBlocks.first(where: {
                    $0.operationIndex > attn.endIndex && $0.operationIndex < ffn.startIndex
                })

                // Find residuals
                let postAttnResidual = residuals.first(where: {
                    $0.addOpIndex > attn.endIndex && $0.addOpIndex < (preFFNNorm?.operationIndex ?? ffn.startIndex)
                })
                let postFFNResidual = residuals.first(where: { $0.addOpIndex > ffn.endIndex })

                var opIndices: [Int] = []
                let startIdx = preAttnNorm?.operationIndex ?? attn.startIndex
                let endIdx = postFFNResidual?.addOpIndex ?? ffn.endIndex
                for idx in startIdx...endIdx {
                    opIndices.append(idx)
                }

                layers.append(TransformerLayer(
                    index: i,
                    preAttentionNorm: preAttnNorm?.operationIndex,
                    attention: attn,
                    postAttentionResidual: postAttnResidual?.addOpIndex,
                    preFFNNorm: preFFNNorm?.operationIndex,
                    ffn: ffn,
                    postFFNResidual: postFFNResidual?.addOpIndex,
                    operationIndices: opIndices
                ))
            }
        }

        return layers
    }

    // MARK: - Pattern Matching

    private func findTransformerBlockMatches(_ structure: ModelStructure, function: HLOFunction) -> [CrossLayerMatch] {
        return structure.transformerLayers.compactMap { layer in
            guard layer.operationIndices.count >= config.minOpsForFusion else { return nil }

            return CrossLayerMatch(
                operationIndices: layer.operationIndices,
                patternType: .transformerBlock,
                metadata: ["layer_index": layer.index]
            )
        }
    }

    private func findResidualChainMatches(_ structure: ModelStructure, function: HLOFunction) -> [CrossLayerMatch] {
        var matches: [CrossLayerMatch] = []

        // Find chains of consecutive residual adds that can be combined
        var i = 0
        while i < structure.residualConnections.count - 1 {
            let current = structure.residualConnections[i]
            let next = structure.residualConnections[i + 1]

            // Check if they're consecutive
            if next.skipFromIndex == current.addOpIndex {
                matches.append(CrossLayerMatch(
                    operationIndices: [current.addOpIndex, next.addOpIndex],
                    patternType: .residualChain,
                    metadata: [:]
                ))
            }
            i += 1
        }

        return matches
    }

    private func findKVCacheMatches(_ structure: ModelStructure, function: HLOFunction) -> [CrossLayerMatch] {
        // Find KV cache update patterns that can be optimized
        var matches: [CrossLayerMatch] = []

        for (i, attn) in structure.attentionBlocks.enumerated() {
            if attn.hasKVCache {
                matches.append(CrossLayerMatch(
                    operationIndices: [attn.startIndex, attn.endIndex],
                    patternType: .kvCacheUpdate,
                    metadata: ["attention_index": i]
                ))
            }
        }

        return matches
    }

    private func sortAndFilterMatches(_ matches: [CrossLayerMatch]) -> [CrossLayerMatch] {
        // Sort by size (larger fusions first)
        var sorted = matches.sorted { $0.operationIndices.count > $1.operationIndices.count }

        // Filter overlapping matches
        var used: Set<Int> = []
        var filtered: [CrossLayerMatch] = []

        for match in sorted {
            let indices = Set(match.operationIndices)
            if indices.isDisjoint(with: used) {
                filtered.append(match)
                used.formUnion(indices)
            }
        }

        return filtered
    }

    // MARK: - Fusion Application

    private func applyFusion(_ match: CrossLayerMatch, to function: HLOFunction) -> HLOFunction {
        switch match.patternType {
        case .transformerBlock:
            return fuseTransformerBlock(match, in: function)
        case .residualChain:
            return fuseResidualChain(match, in: function)
        case .kvCacheUpdate:
            return optimizeKVCache(match, in: function)
        default:
            return function
        }
    }

    private func fuseTransformerBlock(_ match: CrossLayerMatch, in function: HLOFunction) -> HLOFunction {
        guard let layerIndex = match.metadata["layer_index"] else { return function }

        // Create fused transformer block operation
        let fusedOp = createFusedTransformerBlockOp(
            operations: match.operationIndices.map { function.operations[$0] },
            layerIndex: layerIndex
        )

        // Replace operations with fused operation
        return replaceOperations(match.operationIndices, with: fusedOp, in: function)
    }

    private func fuseResidualChain(_ match: CrossLayerMatch, in function: HLOFunction) -> HLOFunction {
        // Combine consecutive residual adds
        guard match.operationIndices.count >= 2 else { return function }

        let ops = match.operationIndices.map { function.operations[$0] }

        // Create fused residual operation
        let fusedOp = createFusedResidualOp(operations: ops)

        return replaceOperations(match.operationIndices, with: fusedOp, in: function)
    }

    private func optimizeKVCache(_ match: CrossLayerMatch, in function: HLOFunction) -> HLOFunction {
        // Find the attention block operations
        guard match.operationIndices.count >= 2 else { return function }

        let startIdx = match.operationIndices.first!
        let endIdx = match.operationIndices.last!

        // Find K/V projection operations followed by cache updates
        var kvProjections: [(projIdx: Int, cacheUpdateIdx: Int, isKey: Bool)] = []

        for i in startIdx...endIdx {
            guard i < function.operations.count else { continue }
            let op = function.operations[i]

            // Look for dot (projection) followed by dynamic-update-slice (cache append)
            if case .dot = op.kind {
                // Check if this feeds into a cache update
                for j in (i + 1)...min(i + 5, endIdx) {
                    guard j < function.operations.count else { continue }
                    let nextOp = function.operations[j]
                    if case .dynamicUpdateSlice = nextOp.kind {
                        if nextOp.operands.contains(op.result) {
                            let isKey = op.result.lowercased().contains("key") ||
                                        op.result.lowercased().contains("_k") ||
                                        nextOp.result.lowercased().contains("key")
                            kvProjections.append((i, j, isKey))
                            break
                        }
                    }
                }
            }
        }

        // If we found K/V projection + cache update pairs, fuse them
        guard !kvProjections.isEmpty else { return function }

        var newOperations = function.operations
        var indicesToRemove: Set<Int> = []

        for (projIdx, cacheIdx, isKey) in kvProjections {
            let projOp = function.operations[projIdx]
            let cacheOp = function.operations[cacheIdx]

            // Create fused projection + cache update operation
            var fusedAttrs = HLOAttributes()
            fusedAttrs.callTargetName = isKey ? "fused_key_proj_cache_update" : "fused_value_proj_cache_update"

            // Combine operands: projection weight, input, cache tensor, indices
            var fusedOperands = projOp.operands
            fusedOperands.append(contentsOf: cacheOp.operands.filter { !fusedOperands.contains($0) })

            let fusedOp = HLOOperation(
                result: cacheOp.result,
                kind: .customCall,
                operands: fusedOperands,
                resultType: cacheOp.resultType,
                attributes: fusedAttrs
            )

            // Replace cache update with fused op, mark projection for removal
            newOperations[cacheIdx] = fusedOp
            indicesToRemove.insert(projIdx)
        }

        // Remove the projection operations that were fused
        let filteredOps = newOperations.enumerated()
            .filter { !indicesToRemove.contains($0.offset) }
            .map { $0.element }

        return HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: filteredOps,
            returnValues: function.returnValues
        )
    }

    private func createFusedTransformerBlockOp(operations: [HLOOperation], layerIndex: Int) -> HLOOperation {
        // Create a custom_call operation representing the fused transformer block
        let resultType = operations.last?.resultType ?? TensorType(shape: [1], elementType: .float32)

        var attributes = HLOAttributes()
        attributes.callTargetName = "fused_transformer_block"

        // Collect operand names from all operations
        var allOperands: [String] = []
        for op in operations {
            allOperands.append(contentsOf: op.operands)
        }
        // Remove duplicates while preserving order
        var seen = Set<String>()
        let uniqueOperands = allOperands.filter { seen.insert($0).inserted }

        return HLOOperation(
            result: "fused_block_\(layerIndex)",
            kind: .customCall,
            operands: uniqueOperands,
            resultType: resultType,
            attributes: attributes
        )
    }

    private func createFusedResidualOp(operations: [HLOOperation]) -> HLOOperation {
        // Create a fused residual add operation
        let resultType = operations.last?.resultType ?? TensorType(shape: [1], elementType: .float32)

        var allOperands: [String] = []
        for op in operations {
            allOperands.append(contentsOf: op.operands)
        }
        var seen = Set<String>()
        let uniqueOperands = allOperands.filter { seen.insert($0).inserted }

        var attributes = HLOAttributes()
        attributes.callTargetName = "fused_residual_chain"

        return HLOOperation(
            result: "fused_residual",
            kind: .customCall,
            operands: uniqueOperands,
            resultType: resultType,
            attributes: attributes
        )
    }

    private func replaceOperations(_ indices: [Int], with newOp: HLOOperation, in function: HLOFunction) -> HLOFunction {
        let indexSet = Set(indices)

        // Keep operations not in the fusion set
        var newOperations: [HLOOperation] = []
        var insertionDone = false

        for (i, op) in function.operations.enumerated() {
            if indexSet.contains(i) {
                // Insert the fused operation at the position of the first replaced operation
                if !insertionDone && i == indices.min() {
                    newOperations.append(newOp)
                    insertionDone = true
                }
                // Skip the replaced operation
            } else {
                newOperations.append(op)
            }
        }

        return HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: newOperations,
            returnValues: function.returnValues
        )
    }
}

// MARK: - Multi-Layer Pipelining

/// Pass for pipelining multiple transformer layers together.
public final class MultiLayerPipeliningPass: @unchecked Sendable {

    public let name = "multi-layer-pipelining"

    /// Number of layers to pipeline together.
    public var pipelineDepth: Int

    public init(pipelineDepth: Int = 2) {
        self.pipelineDepth = pipelineDepth
    }

    /// Creates a pipelined version of consecutive transformer layers.
    ///
    /// - Parameters:
    ///   - layers: The transformer layers to pipeline.
    ///   - function: The original function.
    /// - Returns: Pipelined layer configuration.
    public func createPipelinedLayers(_ layers: [TransformerLayer], function: HLOFunction) -> PipelinedLayerGroup? {
        guard layers.count >= pipelineDepth else { return nil }

        // Group layers into pipeline stages
        var stages: [[TransformerLayer]] = []
        for i in stride(from: 0, to: layers.count, by: pipelineDepth) {
            let end = min(i + pipelineDepth, layers.count)
            stages.append(Array(layers[i..<end]))
        }

        return PipelinedLayerGroup(
            stages: stages,
            pipelineDepth: pipelineDepth
        )
    }
}

/// A group of layers configured for pipelined execution.
public struct PipelinedLayerGroup: Sendable {
    /// Pipeline stages, each containing layers that execute together.
    public let stages: [[TransformerLayer]]

    /// Number of layers pipelined in each stage.
    public let pipelineDepth: Int

    /// Total number of layers.
    public var totalLayers: Int {
        stages.flatMap { $0 }.count
    }
}

// MARK: - KV Cache Optimizer

/// Optimizer for KV cache operations.
public final class KVCacheOptimizer: @unchecked Sendable {

    public let name = "kv-cache-optimizer"

    /// Optimizes KV cache operations in a function.
    ///
    /// Optimizations include:
    /// - Fusing K/V projections with cache append
    /// - Using paged attention patterns
    /// - Optimizing cache memory layout
    ///
    /// - Parameter function: The function to optimize.
    /// - Returns: The optimized function.
    public func optimize(_ function: HLOFunction) -> HLOFunction {
        var current = function

        // Find KV cache patterns
        let cacheOps = findKVCacheOperations(current)

        // Apply optimizations
        for cacheOp in cacheOps {
            current = optimizeCacheOp(cacheOp, in: current)
        }

        return current
    }

    private func findKVCacheOperations(_ function: HLOFunction) -> [KVCachePattern] {
        var patterns: [KVCachePattern] = []

        for (i, op) in function.operations.enumerated() {
            // Look for dynamic-update-slice (cache append) patterns
            if case .dynamicUpdateSlice = op.kind {
                patterns.append(KVCachePattern(
                    updateIndex: i,
                    cacheOperand: op.operands.first ?? "",
                    updateOperand: op.operands.count > 1 ? op.operands[1] : "",
                    isKeyCache: guessIsKeyCache(op)
                ))
            }
        }

        return patterns
    }

    private func guessIsKeyCache(_ op: HLOOperation) -> Bool {
        // Heuristic: check result name or operand names
        let name = op.result.lowercased()
        return name.contains("key") || name.contains("_k_")
    }

    private func optimizeCacheOp(_ pattern: KVCachePattern, in function: HLOFunction) -> HLOFunction {
        // Find the projection operation that produces the update operand
        var projectionIdx: Int? = nil

        for (i, op) in function.operations.enumerated() {
            if op.result == pattern.updateOperand {
                // Check if this is a dot (linear projection)
                if case .dot = op.kind {
                    projectionIdx = i
                }
                break
            }
        }

        // If we found a projection feeding the cache update, fuse them
        guard let projIdx = projectionIdx else { return function }

        let projOp = function.operations[projIdx]
        let cacheOp = function.operations[pattern.updateIndex]

        // Create fused operation: projection + cache update in one kernel
        var fusedAttrs = HLOAttributes()
        fusedAttrs.callTargetName = pattern.isKeyCache ?
            "fused_key_projection_cache_update" :
            "fused_value_projection_cache_update"

        // The fused op takes: input tensor, projection weights, cache tensor, position indices
        var fusedOperands = projOp.operands  // [input, weights]
        // Add cache tensor and indices from the dynamic-update-slice
        for operand in cacheOp.operands {
            if !fusedOperands.contains(operand) {
                fusedOperands.append(operand)
            }
        }

        let fusedOp = HLOOperation(
            result: cacheOp.result,
            kind: .customCall,
            operands: fusedOperands,
            resultType: cacheOp.resultType,
            attributes: fusedAttrs
        )

        // Build new operations list: remove projection, replace cache update with fused
        var newOps: [HLOOperation] = []
        for (i, op) in function.operations.enumerated() {
            if i == projIdx {
                // Skip the projection (it's now fused)
                continue
            } else if i == pattern.updateIndex {
                // Replace cache update with fused operation
                newOps.append(fusedOp)
            } else {
                newOps.append(op)
            }
        }

        return HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: newOps,
            returnValues: function.returnValues
        )
    }
}

/// A detected KV cache update pattern.
private struct KVCachePattern {
    let updateIndex: Int
    let cacheOperand: String
    let updateOperand: String
    let isKeyCache: Bool
}
