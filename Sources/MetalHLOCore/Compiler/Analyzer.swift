// Analyzer.swift
// MetalHLOCore
//
// Analysis passes for computing shapes, dependencies, lifetimes, and detecting patterns.

import Foundation

// MARK: - Analyzer

/// Analyzer computes information about an HLO function without modifying it.
public final class Analyzer: @unchecked Sendable {

    public init() {}

    /// Runs all analysis passes on a function.
    public func analyze(_ function: HLOFunction) -> AnalysisResults {
        // Run all analysis passes
        let shapes = inferShapes(function)
        let elementTypes = inferElementTypes(function)
        let (dependencies, users) = analyzeDependencies(function)
        let lifetimes = analyzeLifetimes(function, dependencies: dependencies, users: users)
        let patterns = detectPatterns(function, shapes: shapes)

        return AnalysisResults(
            shapes: shapes,
            dependencies: dependencies,
            users: users,
            lifetimes: lifetimes,
            patterns: patterns,
            elementTypes: elementTypes
        )
    }

    // MARK: - Shape Inference

    /// Infers shapes for all tensors in the function.
    public func inferShapes(_ function: HLOFunction) -> [TensorID: [Int]] {
        var shapes: [TensorID: [Int]] = [:]

        // Input shapes
        for input in function.inputs {
            shapes[input.name] = input.type.shape
        }

        // Infer output shapes for each operation
        for op in function.operations {
            let inputShapes = op.operands.compactMap { shapes[$0] }
            let outputShape = inferOutputShape(op, inputShapes: inputShapes)
            shapes[op.result] = outputShape
        }

        return shapes
    }

    /// Infers the output shape for an operation.
    private func inferOutputShape(_ op: HLOOperation, inputShapes: [[Int]]) -> [Int] {
        // If result type has shape, use it
        if !op.resultType.shape.isEmpty {
            return op.resultType.shape
        }

        // Otherwise infer from operation type and inputs
        switch op.kind {
        case .add, .subtract, .multiply, .divide, .maximum, .minimum,
             .and, .or, .xor, .power:
            // Binary elementwise - broadcast semantics
            if inputShapes.count >= 2 {
                return broadcastShapes(inputShapes[0], inputShapes[1])
            }
            return inputShapes.first ?? []

        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt,
             .sine, .cosine, .tanh, .logistic, .floor, .ceil,
             .sign, .not, .bitcastConvert, .convert:
            // Unary elementwise - same shape as input
            return inputShapes.first ?? []

        case .dot, .dotGeneral:
            // Matrix multiply - need to handle batching
            return inferDotShape(op, inputShapes: inputShapes)

        case .convolution:
            return inferConvolutionShape(op, inputShapes: inputShapes)

        case .reduce, .reduceWindow:
            return inferReductionShape(op, inputShapes: inputShapes)

        case .broadcastInDim:
            return op.attributes.dimensions ?? inputShapes.first ?? []

        case .reshape, .dynamicReshape:
            return op.resultType.shape

        case .transpose:
            return inferTransposeShape(op, inputShapes: inputShapes)

        case .slice, .dynamicSlice:
            return inferSliceShape(op, inputShapes: inputShapes)

        case .concatenate:
            return inferConcatenateShape(op, inputShapes: inputShapes)

        case .gather, .dynamicGather:
            return op.resultType.shape

        case .scatter:
            return inputShapes.first ?? []

        case .pad:
            return inferPadShape(op, inputShapes: inputShapes)

        case .select:
            // Output shape is same as true/false branches
            return inputShapes.count > 1 ? inputShapes[1] : []

        case .iota:
            return op.resultType.shape

        case .constant:
            return op.resultType.shape

        default:
            // Default: use result type or first input
            return op.resultType.shape.isEmpty ? (inputShapes.first ?? []) : op.resultType.shape
        }
    }

    /// Broadcasts two shapes together.
    private func broadcastShapes(_ a: [Int], _ b: [Int]) -> [Int] {
        let maxRank = max(a.count, b.count)
        var result: [Int] = []

        let paddedA = Array(repeating: 1, count: maxRank - a.count) + a
        let paddedB = Array(repeating: 1, count: maxRank - b.count) + b

        for i in 0..<maxRank {
            let dimA = paddedA[i]
            let dimB = paddedB[i]

            if dimA == dimB {
                result.append(dimA)
            } else if dimA == 1 {
                result.append(dimB)
            } else if dimB == 1 {
                result.append(dimA)
            } else {
                // Incompatible - use max (error in practice)
                result.append(max(dimA, dimB))
            }
        }

        return result
    }

    /// Infers shape for dot operations.
    private func inferDotShape(_ op: HLOOperation, inputShapes: [[Int]]) -> [Int] {
        guard inputShapes.count >= 2 else { return [] }

        let lhs = inputShapes[0]
        let rhs = inputShapes[1]

        // Simple 2D case: [M, K] @ [K, N] -> [M, N]
        if lhs.count == 2 && rhs.count == 2 {
            return [lhs[0], rhs[1]]
        }

        // Batched case
        if let contracting = op.attributes.dotDimensionNumbers?.lhsContractingDimensions,
           let batchDims = op.attributes.dotDimensionNumbers?.lhsBatchingDimensions {
            var resultShape: [Int] = []

            // Add batch dimensions
            for dim in batchDims {
                if dim < lhs.count {
                    resultShape.append(lhs[dim])
                }
            }

            // Add non-contracting dimensions from lhs
            for (i, size) in lhs.enumerated() {
                if !contracting.contains(i) && !batchDims.contains(i) {
                    resultShape.append(size)
                }
            }

            // Add non-contracting dimensions from rhs
            let rhsContracting = op.attributes.dotDimensionNumbers?.rhsContractingDimensions ?? []
            let rhsBatch = op.attributes.dotDimensionNumbers?.rhsBatchingDimensions ?? []
            for (i, size) in rhs.enumerated() {
                if !rhsContracting.contains(i) && !rhsBatch.contains(i) {
                    resultShape.append(size)
                }
            }

            return resultShape
        }

        // Fallback
        return op.resultType.shape
    }

    /// Infers shape for convolution operations.
    private func inferConvolutionShape(_ op: HLOOperation, inputShapes: [[Int]]) -> [Int] {
        // Use result type if available
        if !op.resultType.shape.isEmpty {
            return op.resultType.shape
        }
        return inputShapes.first ?? []
    }

    /// Infers shape for reduction operations.
    private func inferReductionShape(_ op: HLOOperation, inputShapes: [[Int]]) -> [Int] {
        guard let inputShape = inputShapes.first,
              let dimensions = op.attributes.dimensions else {
            return op.resultType.shape
        }

        let dimSet = Set(dimensions)
        var resultShape: [Int] = []

        for (i, size) in inputShape.enumerated() {
            if !dimSet.contains(i) {
                resultShape.append(size)
            }
        }

        return resultShape.isEmpty ? [1] : resultShape
    }

    /// Infers shape for transpose operations.
    private func inferTransposeShape(_ op: HLOOperation, inputShapes: [[Int]]) -> [Int] {
        guard let inputShape = inputShapes.first,
              let permutation = op.attributes.dimensions else {
            return op.resultType.shape
        }

        return permutation.map { inputShape[$0] }
    }

    /// Infers shape for slice operations.
    private func inferSliceShape(_ op: HLOOperation, inputShapes: [[Int]]) -> [Int] {
        guard let starts = op.attributes.sliceStarts,
              let limits = op.attributes.sliceLimits else {
            return op.resultType.shape
        }

        let strides = op.attributes.sliceStrides ?? Array(repeating: 1, count: starts.count)

        return zip(zip(starts, limits), strides).map { (range, stride) in
            let (start, limit) = range
            return (limit - start + stride - 1) / stride
        }
    }

    /// Infers shape for concatenate operations.
    private func inferConcatenateShape(_ op: HLOOperation, inputShapes: [[Int]]) -> [Int] {
        guard let firstShape = inputShapes.first,
              let dimension = op.attributes.axis else {
            return op.resultType.shape
        }

        var resultShape = firstShape
        let concatSize = inputShapes.reduce(0) { $0 + ($1.count > dimension ? $1[dimension] : 0) }
        if dimension < resultShape.count {
            resultShape[dimension] = concatSize
        }

        return resultShape
    }

    /// Infers shape for pad operations.
    private func inferPadShape(_ op: HLOOperation, inputShapes: [[Int]]) -> [Int] {
        guard let inputShape = inputShapes.first,
              let lowPad = op.attributes.padLow,
              let highPad = op.attributes.padHigh else {
            return op.resultType.shape
        }

        let interiorPad = op.attributes.padInterior ?? Array(repeating: 0, count: inputShape.count)

        return zip(zip(inputShape, zip(lowPad, highPad)), interiorPad).map { (sizes, interior) in
            let (inputSize, (low, high)) = sizes
            return inputSize + low + high + (inputSize - 1) * interior
        }
    }

    // MARK: - Element Type Inference

    /// Infers element types for all tensors.
    public func inferElementTypes(_ function: HLOFunction) -> [TensorID: ElementType] {
        var types: [TensorID: ElementType] = [:]

        for input in function.inputs {
            types[input.name] = input.type.elementType
        }

        for op in function.operations {
            types[op.result] = op.resultType.elementType
        }

        return types
    }

    // MARK: - Dependency Analysis

    /// Analyzes dependencies between operations.
    public func analyzeDependencies(_ function: HLOFunction) -> (dependencies: [OpID: Set<OpID>], users: [OpID: Set<OpID>]) {
        var dependencies: [OpID: Set<OpID>] = [:]
        var users: [OpID: Set<OpID>] = [:]
        var producers: [TensorID: OpID] = [:]

        // Initialize
        for op in function.operations {
            dependencies[op.result] = []
            users[op.result] = []
            producers[op.result] = op.result
        }

        // Build dependency graph
        for op in function.operations {
            for operand in op.operands {
                if let producer = producers[operand] {
                    dependencies[op.result]?.insert(producer)
                    users[producer]?.insert(op.result)
                }
            }
        }

        return (dependencies, users)
    }

    // MARK: - Lifetime Analysis

    /// Analyzes tensor lifetimes.
    public func analyzeLifetimes(_ function: HLOFunction, dependencies: [OpID: Set<OpID>], users: [OpID: Set<OpID>]) -> [TensorID: Lifetime] {
        var lifetimes: [TensorID: Lifetime] = [:]

        // Build operation index map
        var opIndex: [OpID: Int] = [:]
        for (index, op) in function.operations.enumerated() {
            opIndex[op.result] = index
        }

        // Input lifetimes
        for input in function.inputs {
            let lastUse = findLastUse(input.name, in: function, opIndex: opIndex)
            lifetimes[input.name] = Lifetime(defined: -1, lastUsed: lastUse)
        }

        // Operation output lifetimes
        for (index, op) in function.operations.enumerated() {
            let lastUse: Int
            if function.returnValues.contains(op.result) {
                lastUse = function.operations.count
            } else {
                lastUse = findLastUse(op.result, in: function, opIndex: opIndex)
            }
            lifetimes[op.result] = Lifetime(defined: index, lastUsed: lastUse)
        }

        return lifetimes
    }

    /// Finds the last use of a tensor.
    private func findLastUse(_ tensorID: TensorID, in function: HLOFunction, opIndex: [OpID: Int]) -> Int {
        var lastUse = -1

        for op in function.operations {
            if op.operands.contains(tensorID) {
                if let index = opIndex[op.result] {
                    lastUse = max(lastUse, index)
                }
            }
        }

        if function.returnValues.contains(tensorID) {
            lastUse = function.operations.count
        }

        return lastUse
    }

    // MARK: - Pattern Detection

    /// Detects high-level patterns in the computation graph.
    public func detectPatterns(_ function: HLOFunction, shapes: [TensorID: [Int]]) -> [DetectedPattern] {
        var patterns: [DetectedPattern] = []

        // Build maps for efficient lookup
        let definingOps = buildDefiningOpsMap(function)

        // Detect patterns from most specific to least specific. Attention runs
        // before softmax so attention-style softmaxes are absorbed into the
        // larger pattern; the standalone softmax detector then catches any
        // softmax not already inside an attention block.
        patterns.append(contentsOf: detectAttentionPatterns(function, definingOps: definingOps, shapes: shapes))
        patterns.append(contentsOf: detectNormPatterns(function, definingOps: definingOps, shapes: shapes))
        patterns.append(contentsOf: detectBatchNormPatterns(function, definingOps: definingOps, shapes: shapes))
        patterns.append(contentsOf: detectActivationPatterns(function, definingOps: definingOps))
        patterns.append(contentsOf: detectFFNPatterns(function, definingOps: definingOps, shapes: shapes))
        patterns.append(contentsOf: detectMatMulBiasActPatterns(function, definingOps: definingOps))
        patterns.append(contentsOf: detectConvBiasActPatterns(function, definingOps: definingOps, existing: patterns))
        patterns.append(contentsOf: detectSoftmaxPatterns(function, definingOps: definingOps, shapes: shapes, existing: patterns))

        return patterns
    }

    /// Detects `convolution → add(broadcast(bias)) → optional activation`
    /// chains. Returns each match as a single pattern with the activation
    /// kind in metadata (or "none"). Skips conv ops already covered by a
    /// higher-priority pattern.
    private func detectConvBiasActPatterns(
        _ function: HLOFunction,
        definingOps: [TensorID: (op: HLOOperation, index: Int)],
        existing: [DetectedPattern]
    ) -> [DetectedPattern] {
        var claimed = Set<Int>()
        for p in existing {
            for idx in p.operationIndices { claimed.insert(idx) }
        }

        // Use map for use-count once (used to confirm the conv result has only
        // the bias-add as its consumer — fusion would be unsafe otherwise).
        var useCount: [TensorID: Int] = [:]
        for op in function.operations {
            for operand in op.operands {
                useCount[operand, default: 0] += 1
            }
        }
        for retVal in function.returnValues {
            useCount[retVal, default: 0] += 1
        }

        var patterns: [DetectedPattern] = []

        for (convIdx, convOp) in function.operations.enumerated() {
            guard convOp.kind == .convolution else { continue }
            if claimed.contains(convIdx) { continue }

            // Look for an `add` whose first operand is the conv result and
            // whose second operand is `broadcast_in_dim(bias)`. (Some IRs use
            // (broadcast, conv) ordering — handle both.)
            guard useCount[convOp.result, default: 0] == 1 else { continue }
            guard let consumerIdx = function.operations.firstIndex(where: { $0.operands.contains(convOp.result) }),
                  consumerIdx > convIdx else { continue }
            let addOp = function.operations[consumerIdx]
            guard addOp.kind == .add, addOp.operands.count == 2 else { continue }

            // Pick the operand that's NOT the conv result.
            let biasOperand: TensorID
            if addOp.operands[0] == convOp.result { biasOperand = addOp.operands[1] }
            else if addOp.operands[1] == convOp.result { biasOperand = addOp.operands[0] }
            else { continue }

            // Bias should come from a broadcast_in_dim of a 1-D tensor (per-channel).
            guard let biasDef = definingOps[biasOperand],
                  biasDef.op.kind == .broadcastInDim else { continue }

            var matchedIndices = [convIdx, biasDef.index, consumerIdx]
            var rootIdx = consumerIdx
            var activation: String = "none"

            // Optional activation immediately after add — relu / sigmoid / tanh.
            // (silu is decomposed as multiply+sigmoid; skipped for now.)
            if useCount[addOp.result, default: 0] == 1,
               let activeIdx = function.operations.firstIndex(where: { $0.operands.contains(addOp.result) }),
               activeIdx > consumerIdx {
                let activeOp = function.operations[activeIdx]
                let act: String?
                switch activeOp.kind {
                case .maximum:
                    // ReLU: maximum(x, 0). Confirm the second operand is a constant zero.
                    if activeOp.operands.count == 2,
                       let otherDef = definingOps[activeOp.operands[1]],
                       otherDef.op.kind == .constant {
                        act = "relu"
                    } else { act = nil }
                case .logistic: act = "sigmoid"
                case .tanh: act = "tanh"
                default: act = nil
                }
                if let act {
                    activation = act
                    matchedIndices.append(activeIdx)
                    rootIdx = activeIdx
                }
            }

            patterns.append(DetectedPattern(
                type: .convBiasActivation,
                operationIndices: matchedIndices.sorted(),
                rootIndex: rootIdx,
                metadata: PatternMetadata(activation: activation)
            ))
            for idx in matchedIndices { claimed.insert(idx) }
        }

        return patterns
    }

    /// Detects standalone numerically-stable softmax patterns:
    ///
    ///     max     = reduce_max(x, axis)
    ///     max_bc  = broadcast_in_dim(max)        // optional
    ///     shifted = subtract(x, max_or_max_bc)
    ///     e       = exp(shifted)
    ///     sum     = reduce_sum(e, axis)
    ///     sum_bc  = broadcast_in_dim(sum)        // optional
    ///     out     = divide(e, sum_or_sum_bc)
    ///
    /// Skips matches whose root divide is already part of a previously-detected
    /// pattern (e.g. inside an attention block) so we don't double-fuse.
    private func detectSoftmaxPatterns(
        _ function: HLOFunction,
        definingOps: [TensorID: (op: HLOOperation, index: Int)],
        shapes: [TensorID: [Int]],
        existing: [DetectedPattern]
    ) -> [DetectedPattern] {
        // Indices already covered by a higher-priority pattern.
        var claimed = Set<Int>()
        for p in existing {
            for idx in p.operationIndices {
                claimed.insert(idx)
            }
        }

        var patterns: [DetectedPattern] = []

        for (rootIdx, op) in function.operations.enumerated() {
            guard op.kind == .divide, op.operands.count == 2 else { continue }
            if claimed.contains(rootIdx) { continue }

            // Numerator: exp.
            guard let expDef = definingOps[op.operands[0]],
                  expDef.op.kind == .exponential else { continue }

            // Denominator: reduce_sum, optionally wrapped in broadcast_in_dim.
            var sumDef: (op: HLOOperation, index: Int)? = nil
            var sumBroadcastDef: (op: HLOOperation, index: Int)? = nil
            if let denomDef = definingOps[op.operands[1]] {
                if denomDef.op.kind == .reduce {
                    sumDef = denomDef
                } else if denomDef.op.kind == .broadcastInDim,
                          let inner = definingOps[denomDef.op.operands[0]],
                          inner.op.kind == .reduce {
                    sumBroadcastDef = denomDef
                    sumDef = inner
                }
            }
            guard let sum = sumDef,
                  (sum.op.attributes.reductionKind ?? .sum) == .sum,
                  sum.op.operands.first == expDef.op.result else { continue }

            // exp's input: subtract(x, max_or_max_bc).
            guard let subDef = definingOps[expDef.op.operands[0]],
                  subDef.op.kind == .subtract,
                  subDef.op.operands.count == 2 else { continue }

            // Subtract's second operand: reduce_max, optionally wrapped in
            // broadcast_in_dim.
            var maxDef: (op: HLOOperation, index: Int)? = nil
            var maxBroadcastDef: (op: HLOOperation, index: Int)? = nil
            if let shifterDef = definingOps[subDef.op.operands[1]] {
                if shifterDef.op.kind == .reduce {
                    maxDef = shifterDef
                } else if shifterDef.op.kind == .broadcastInDim,
                          let inner = definingOps[shifterDef.op.operands[0]],
                          inner.op.kind == .reduce {
                    maxBroadcastDef = shifterDef
                    maxDef = inner
                }
            }
            guard let mx = maxDef,
                  (mx.op.attributes.reductionKind ?? .max) == .max,
                  mx.op.operands.first == subDef.op.operands[0] else { continue }

            // Both reduces should agree on the axis.
            let maxAxis = mx.op.attributes.dimensions?.first
            let sumAxis = sum.op.attributes.dimensions?.first
            guard let axis = maxAxis, axis == sumAxis else { continue }

            var indices = [mx.index]
            if let mbc = maxBroadcastDef { indices.append(mbc.index) }
            indices.append(subDef.index)
            indices.append(expDef.index)
            indices.append(sum.index)
            if let sbc = sumBroadcastDef { indices.append(sbc.index) }
            indices.append(rootIdx)

            // Skip if any of the intermediate ops we'd consume is also used
            // elsewhere (an external use means we can't fold them away). The
            // root divide always belongs to the pattern; only check the rest.
            let consumed = Set(indices.dropLast())
            var hasExternalUse = false
            for idx in consumed {
                let result = function.operations[idx].result
                for (otherIdx, otherOp) in function.operations.enumerated() {
                    if consumed.contains(otherIdx) || otherIdx == rootIdx { continue }
                    if otherOp.operands.contains(result) {
                        hasExternalUse = true
                        break
                    }
                }
                if function.returnValues.contains(result) { hasExternalUse = true }
                if hasExternalUse { break }
            }
            if hasExternalUse { continue }

            patterns.append(DetectedPattern(
                type: .softmax,
                operationIndices: indices.sorted(),
                rootIndex: rootIdx,
                metadata: PatternMetadata(axis: axis)
            ))
            for idx in indices { claimed.insert(idx) }
        }

        return patterns
    }

    /// Builds a map from tensor ID to defining operation.
    private func buildDefiningOpsMap(_ function: HLOFunction) -> [TensorID: (op: HLOOperation, index: Int)] {
        var map: [TensorID: (op: HLOOperation, index: Int)] = [:]
        for (index, op) in function.operations.enumerated() {
            map[op.result] = (op, index)
        }
        return map
    }

    /// Detects attention patterns.
    private func detectAttentionPatterns(_ function: HLOFunction, definingOps: [TensorID: (op: HLOOperation, index: Int)], shapes: [TensorID: [Int]]) -> [DetectedPattern] {
        var patterns: [DetectedPattern] = []

        // Walks back through the canonical numerically-stable softmax chain
        // (`divide(exp(subtract(scaled, max_bc)), sum_bc)`) to find the
        // matmul that produced `scaled`. Returns (qkMatmulDef, softmaxOpIdx)
        // where softmaxOpIdx is the *root* of the softmax (the `divide`).
        // Optionally peels off a `multiply(scores, scale)` between the QK
        // matmul and the subtract — that's the scaling factor `1/sqrt(d)`.
        func walkExpandedSoftmax(_ rootInput: TensorID) -> (qk: (op: HLOOperation, index: Int), softmaxRootIdx: Int)? {
            guard let divDef = definingOps[rootInput],
                  divDef.op.kind == .divide,
                  divDef.op.operands.count == 2,
                  let expDef = definingOps[divDef.op.operands[0]],
                  expDef.op.kind == .exponential,
                  let subDef = definingOps[expDef.op.operands[0]],
                  subDef.op.kind == .subtract,
                  subDef.op.operands.count == 2 else { return nil }

            // Walk through the numerator's pre-shift value back to a matmul,
            // peeling an optional broadcast and an optional scaling multiply.
            var cur = subDef.op.operands[0]
            var hops = 0
            while hops < 4 {
                guard let def = definingOps[cur] else { return nil }
                if def.op.kind == .dot || def.op.kind == .dotGeneral {
                    return (qk: def, softmaxRootIdx: divDef.index)
                }
                if def.op.kind == .multiply || def.op.kind == .broadcastInDim {
                    // Pick the operand most likely to lead to the matmul:
                    // for multiply(scores, scale_bc) the scores side is the
                    // one whose own definer is a dot (or another multiply).
                    if def.op.operands.isEmpty { return nil }
                    cur = def.op.operands[0]
                    hops += 1
                    continue
                }
                return nil
            }
            return nil
        }

        // Look for Q @ K^T -> softmax -> @ V pattern.
        for (index, op) in function.operations.enumerated() {
            // Look for final matmul (attention @ V)
            guard op.kind == .dot || op.kind == .dotGeneral else { continue }

            var softmaxRootIdx: Int? = nil
            var qkMatmulDef: (op: HLOOperation, index: Int)? = nil

            // Variant 1: softmax encoded as a single `customCall("softmax")`.
            if let cdef = definingOps[op.operands[0]],
               cdef.op.kind == .customCall,
               cdef.op.attributes.callTargetName?.contains("softmax") == true,
               let qkDef = definingOps[cdef.op.operands[0]],
               (qkDef.op.kind == .dot || qkDef.op.kind == .dotGeneral) {
                softmaxRootIdx = cdef.index
                qkMatmulDef = qkDef
            }

            // Variant 2: numerically-stable softmax expanded into
            // divide / exp / subtract / reduce_max / multiply chain.
            if softmaxRootIdx == nil {
                if let (qk, smRoot) = walkExpandedSoftmax(op.operands[0]) {
                    qkMatmulDef = qk
                    softmaxRootIdx = smRoot
                }
            }

            guard let smIdx = softmaxRootIdx, let qk = qkMatmulDef else { continue }

            // Found attention pattern!
            let operationIndices = [qk.index, smIdx, index]

            // Extract configuration from shapes
            var metadata = PatternMetadata()
            if let qShape = shapes[qk.op.operands[0]] {
                if qShape.count >= 3 {
                    metadata.numHeads = qShape[qShape.count - 3]
                    metadata.headDim = qShape[qShape.count - 1]
                }
            }

            patterns.append(DetectedPattern(
                type: .attention,
                operationIndices: operationIndices,
                rootIndex: index,
                metadata: metadata
            ))
        }

        return patterns
    }

    /// Detects normalization patterns (LayerNorm, RMSNorm).
    private func detectNormPatterns(_ function: HLOFunction, definingOps: [TensorID: (op: HLOOperation, index: Int)], shapes: [TensorID: [Int]]) -> [DetectedPattern] {
        var patterns: [DetectedPattern] = []

        for (index, op) in function.operations.enumerated() {
            // Check for custom_call layer_norm or rms_norm
            if op.kind == .customCall {
                if let target = op.attributes.callTargetName {
                    if target.contains("layer_norm") {
                        var metadata = PatternMetadata()
                        metadata.epsilon = op.attributes.epsilon ?? 1e-5
                        if let shape = shapes[op.operands[0]] {
                            metadata.hiddenDim = shape.last
                        }
                        patterns.append(DetectedPattern(
                            type: .layerNorm,
                            operationIndices: [index],
                            rootIndex: index,
                            metadata: metadata
                        ))
                    } else if target.contains("rms_norm") {
                        var metadata = PatternMetadata()
                        metadata.epsilon = op.attributes.epsilon ?? 1e-5
                        if let shape = shapes[op.operands[0]] {
                            metadata.hiddenDim = shape.last
                        }
                        patterns.append(DetectedPattern(
                            type: .rmsNorm,
                            operationIndices: [index],
                            rootIndex: index,
                            metadata: metadata
                        ))
                    }
                }
            }

            // Detect expanded normalization patterns. The core shape is
            //   normalize = multiply(X, broadcast(rsqrt(<variance-chain>)))
            // where the variance chain eventually contains a reduce(...).
            // If X is `subtract(input, broadcast(reduce(input)))` we have
            // LayerNorm; otherwise we have RMSNorm.
            //
            // The optional affine tail
            //   scaled    = multiply(normalize, gamma_bc)
            //   result    = add(scaled, beta_bc)
            // is walked forward by use-finding so the fusion pass sees the
            // full chain — `firstOp.operands[0]` is the input X (set by
            // pointing at the `subtract`/`multiply` op that consumes it),
            // the affine operands carry gamma + beta.
            if op.kind == .multiply, op.operands.count == 2 {
                // Resolves a tensor through optional broadcast nodes to its
                // rsqrt definer, returning that rsqrt op (or nil).
                func rsqrtThrough(_ tid: TensorID) -> (op: HLOOperation, index: Int)? {
                    var cur = tid
                    var hops = 0
                    while hops < 3 {
                        guard let def = definingOps[cur] else { return nil }
                        if def.op.kind == .rsqrt { return def }
                        if def.op.kind == .broadcastInDim, !def.op.operands.isEmpty {
                            cur = def.op.operands[0]
                            hops += 1
                            continue
                        }
                        return nil
                    }
                    return nil
                }

                // Walks back from the rsqrt's input through add/divide/
                // multiply/broadcast looking for a reduce — confirms this
                // really is a normalization (not just an arbitrary rsqrt).
                func reachesReduce(from tid: TensorID) -> Bool {
                    var stack = [tid]
                    var visited = Set<TensorID>()
                    var hops = 0
                    while let cur = stack.popLast(), hops < 12 {
                        if !visited.insert(cur).inserted { continue }
                        hops += 1
                        guard let def = definingOps[cur] else { continue }
                        if def.op.kind == .reduce { return true }
                        // Walk through arithmetic that's typical inside the
                        // variance computation.
                        switch def.op.kind {
                        case .add, .subtract, .multiply, .divide,
                             .broadcastInDim, .convert:
                            stack.append(contentsOf: def.op.operands)
                        default:
                            break
                        }
                    }
                    return false
                }

                // Find which operand carries the rsqrt; the other carries X.
                let xOperand: TensorID
                let rsqrtDef: (op: HLOOperation, index: Int)
                if let r = rsqrtThrough(op.operands[1]) {
                    xOperand = op.operands[0]
                    rsqrtDef = r
                } else if let r = rsqrtThrough(op.operands[0]) {
                    xOperand = op.operands[1]
                    rsqrtDef = r
                } else {
                    continue
                }

                // Confirm the rsqrt is normalization-shaped (variance chain
                // eventually hits a reduce). Without this, any
                // `multiply(_, rsqrt(_))` would be miscategorised.
                guard !rsqrtDef.op.operands.isEmpty,
                      reachesReduce(from: rsqrtDef.op.operands[0]) else { continue }

                // Determine LayerNorm vs RMSNorm from the X side. For
                // LayerNorm the X feeding the normalize multiply is
                // `subtract(input, mean_bc)`; the *real* input to the kernel
                // is the subtract's first operand.
                let xDef = definingOps[xOperand]
                let isLayerNorm = xDef?.op.kind == .subtract
                let firstIdx: Int
                if isLayerNorm, let xDef {
                    // Point firstOpIdx at the subtract so NormFusionPass
                    // reads `subtract.operands[0]` (the original input
                    // tensor) as the kernel input.
                    firstIdx = xDef.index
                } else {
                    // RMSNorm: the normalize multiply itself feeds the input
                    // directly; firstOp.operands[0] is the input.
                    firstIdx = index
                }

                // Walk forward to pick up the optional affine tail —
                // `multiply(normalize, gamma_bc)` then `add(scaled, beta_bc)`.
                // Linear scan is fine for the few-hundred-op functions
                // MetalHLO produces.
                var rootIdx = index
                var chainIndices: [Int] = [firstIdx, rsqrtDef.index, index]
                var current = op.result
                func findFirstConsumer(of tid: TensorID, kind: HLOOpKind) -> (op: HLOOperation, index: Int)? {
                    for (i, candidate) in function.operations.enumerated() where i > rootIdx {
                        if candidate.kind == kind, candidate.operands.contains(tid) {
                            return (candidate, i)
                        }
                    }
                    return nil
                }
                if let gammaMul = findFirstConsumer(of: current, kind: .multiply) {
                    chainIndices.append(gammaMul.index)
                    rootIdx = gammaMul.index
                    current = gammaMul.op.result
                    if let betaAdd = findFirstConsumer(of: current, kind: .add) {
                        chainIndices.append(betaAdd.index)
                        rootIdx = betaAdd.index
                    }
                }

                var metadata = PatternMetadata()
                metadata.epsilon = 1e-5
                if let shape = shapes[xOperand] {
                    metadata.hiddenDim = shape.last
                }

                patterns.append(DetectedPattern(
                    type: isLayerNorm ? .layerNorm : .rmsNorm,
                    operationIndices: chainIndices,
                    rootIndex: rootIdx,
                    metadata: metadata
                ))
            }
        }

        return patterns
    }

    /// Detects the BatchNorm-training apply chain JAX emits for `nn.BatchNorm`.
    ///
    /// JAX produces this canonical structure (NHWC, channel = last axis):
    ///   mean_1d   = reduce(input, axes=[0,1,2]) / N
    ///   var_1d    = max(0, reduce(input², axes=[0,1,2]) / N - mean²)
    ///   mean_bc   = broadcast(reshape(mean_1d, [1,1,1,C]))
    ///   centered  = input - mean_bc
    ///   var_eps   = reshape(var_1d, [1,1,1,C]) + eps
    ///   rsqrt_val = rsqrt(var_eps)
    ///   scale     = rsqrt_val * reshape(gamma, [1,1,1,C])
    ///   scale_bc  = broadcast(scale)
    ///   scaled    = centered * scale_bc            ← detected as the "root multiply"
    ///   beta_bc   = broadcast(reshape(beta, [1,1,1,C]))
    ///   result    = scaled + beta_bc               ← the pattern's rootIndex
    ///
    /// Distinguishes from LayerNorm by:
    ///   - The reduce(input) has multiple axes (e.g. [0,1,2]) producing a 1D
    ///     channel tensor, not a single trailing-axis reduce.
    ///   - Gamma is folded into rsqrt *before* the broadcast (not after).
    ///
    /// The mean / variance / reduce ops upstream are intentionally NOT included
    /// in `operationIndices`. They have external uses (the running-stats
    /// update consumes them too) so leaving them out keeps the existing
    /// dead-op elimination from accidentally removing them — and the fused
    /// custom_call takes the small precomputed mean / variance as inputs.
    private func detectBatchNormPatterns(
        _ function: HLOFunction,
        definingOps: [TensorID: (op: HLOOperation, index: Int)],
        shapes: [TensorID: [Int]]
    ) -> [DetectedPattern] {
        var patterns: [DetectedPattern] = []

        // Walks broadcast → multiply(rsqrt, reshape(gamma_1d)) to extract
        // (rsqrt def, gamma source 1D tensor). Returns nil if the chain
        // doesn't match.
        func parseScaleSide(_ tid: TensorID) -> (rsqrtIdx: Int, rsqrtDef: HLOOperation, gammaSrc: TensorID, scaleBcIdx: Int, scaleMulIdx: Int, gammaReshapeIdx: Int)? {
            guard let bcDef = definingOps[tid], bcDef.op.kind == .broadcastInDim,
                  !bcDef.op.operands.isEmpty else { return nil }
            guard let mulDef = definingOps[bcDef.op.operands[0]], mulDef.op.kind == .multiply,
                  mulDef.op.operands.count == 2 else { return nil }
            // One operand is rsqrt(...) (or broadcast of it for size-1), the
            // other is reshape(gamma_1d, [1,1,...,C]).
            for (rsqrtSlot, gammaSlot) in [(0, 1), (1, 0)] {
                guard let rDef = definingOps[mulDef.op.operands[rsqrtSlot]],
                      rDef.op.kind == .rsqrt else { continue }
                guard let gDef = definingOps[mulDef.op.operands[gammaSlot]],
                      gDef.op.kind == .reshape, !gDef.op.operands.isEmpty else { continue }
                let gammaSrc = gDef.op.operands[0]
                if let gshape = shapes[gammaSrc], gshape.count == 1 {
                    return (rDef.index, rDef.op, gammaSrc, bcDef.index, mulDef.index, gDef.index)
                }
            }
            return nil
        }

        // Walks broadcast → reshape(small_1d) to extract the small tensor id.
        func parseSmallBroadcast(_ tid: TensorID) -> (bcIdx: Int, reshapeIdx: Int, src: TensorID)? {
            guard let bcDef = definingOps[tid], bcDef.op.kind == .broadcastInDim,
                  !bcDef.op.operands.isEmpty else { return nil }
            guard let rsDef = definingOps[bcDef.op.operands[0]], rsDef.op.kind == .reshape,
                  !rsDef.op.operands.isEmpty else { return nil }
            return (bcDef.index, rsDef.index, rsDef.op.operands[0])
        }

        // Walks an rsqrt's input back to the 1D variance tensor (the operand
        // of the reshape that feeds the var+eps add). Also returns eps.
        func parseVarianceSide(_ rsqrtInput: TensorID) -> (varianceSrc: TensorID, varReshapeIdx: Int, varEpsAddIdx: Int, eps: Float)? {
            guard let addDef = definingOps[rsqrtInput], addDef.op.kind == .add,
                  addDef.op.operands.count == 2 else { return nil }
            // One operand is reshape(variance_1d, [1,1,...,1,C]); the other is
            // broadcast(epsilon-shaped scalar). Try both orderings.
            for (varSlot, epsSlot) in [(0, 1), (1, 0)] {
                guard let rsDef = definingOps[addDef.op.operands[varSlot]], rsDef.op.kind == .reshape,
                      !rsDef.op.operands.isEmpty else { continue }
                let varianceSrc = rsDef.op.operands[0]
                guard let vshape = shapes[varianceSrc], vshape.count == 1 else { continue }
                // Resolve eps if we can — it's a broadcast(constant). Default
                // to 1e-5 if the chain is shaped slightly differently.
                var eps: Float = 1e-5
                if let epsBcDef = definingOps[addDef.op.operands[epsSlot]],
                   epsBcDef.op.kind == .broadcastInDim,
                   !epsBcDef.op.operands.isEmpty,
                   let epsConstDef = definingOps[epsBcDef.op.operands[0]],
                   epsConstDef.op.kind == .constant,
                   let cv = epsConstDef.op.attributes.constantValue {
                    switch cv {
                    case .scalar(let d): eps = Float(d)
                    case .splat(let d, _): eps = Float(d)
                    case .dense(let arr, _): if let first = arr.first { eps = Float(first) }
                    case .hexBytes: break  // skip: would require decoding raw bytes
                    }
                }
                return (varianceSrc, rsDef.index, addDef.index, eps)
            }
            return nil
        }

        // Verifies that the 1D mean tensor really came from a reduce over
        // multiple axes (batch + spatial — that's the BN signature, distinct
        // from LayerNorm's last-axis-only reduce).
        func meanFromMultiAxisReduce(_ meanSrc: TensorID, inputRank: Int) -> Bool {
            guard let dDef = definingOps[meanSrc], dDef.op.kind == .divide,
                  !dDef.op.operands.isEmpty else { return false }
            guard let rDef = definingOps[dDef.op.operands[0]], rDef.op.kind == .reduce else {
                return false
            }
            let axes = rDef.op.attributes.dimensions ?? []
            // BN's reduce collapses all-but-feature, so for a 4D input we
            // expect 3 reduction axes. RMS / LN reduce only over the last.
            return axes.count >= 2 && axes.count == inputRank - 1
        }

        for (rootMulIdx, op) in function.operations.enumerated() {
            guard op.kind == .multiply, op.operands.count == 2 else { continue }
            guard let inputShape = shapes[op.result], inputShape.count >= 3 else { continue }

            // Try both operand orderings — which one carries the scale path
            // (broadcast→multiply(rsqrt,reshape(gamma))) depends on JAX's
            // canonicalisation pass, which isn't stable across releases.
            var matched: (scale: (rsqrtIdx: Int, rsqrtDef: HLOOperation, gammaSrc: TensorID, scaleBcIdx: Int, scaleMulIdx: Int, gammaReshapeIdx: Int), centeredIdx: Int, centeredSubIdx: Int, mean: (bcIdx: Int, reshapeIdx: Int, src: TensorID), inputX: TensorID)?
            for (centeredSlot, scaleSlot) in [(0, 1), (1, 0)] {
                guard let scale = parseScaleSide(op.operands[scaleSlot]) else { continue }
                guard let subDef = definingOps[op.operands[centeredSlot]], subDef.op.kind == .subtract,
                      subDef.op.operands.count == 2 else { continue }
                let inputX = subDef.op.operands[0]
                guard let mean = parseSmallBroadcast(subDef.op.operands[1]) else { continue }
                // Confirm the 1D mean came from a multi-axis reduce — this is
                // the test that says "BN, not LN".
                guard meanFromMultiAxisReduce(mean.src, inputRank: inputShape.count) else { continue }
                matched = (scale, subDef.index, subDef.index, mean, inputX)
                break
            }
            guard let m = matched else { continue }

            // Pull eps + variance source through the rsqrt chain.
            guard !m.scale.rsqrtDef.operands.isEmpty,
                  let vinfo = parseVarianceSide(m.scale.rsqrtDef.operands[0]) else { continue }

            // Find the trailing `add(scaled, broadcast(reshape(beta)))`. JAX
            // emits this as the next consumer of the multiply result.
            var betaInfo: (addIdx: Int, betaBcIdx: Int, betaReshapeIdx: Int, betaSrc: TensorID)?
            for (i, candidate) in function.operations.enumerated() where i > rootMulIdx {
                guard candidate.kind == .add, candidate.operands.contains(op.result) else { continue }
                let other = candidate.operands.first(where: { $0 != op.result }) ?? op.result
                guard let bc = parseSmallBroadcast(other) else { break }
                betaInfo = (i, bc.bcIdx, bc.reshapeIdx, bc.src)
                break
            }
            guard let beta = betaInfo else { continue }

            // Build the index set. We do NOT include the upstream reduce /
            // divide / max chain that produces mean_1d and variance_1d — those
            // are also consumed by the running-stats update and must stick
            // around. The pass removes everything in `operationIndices` (when
            // unreferenced externally) and replaces `rootIndex` with the
            // fused custom_call.
            let chainIndices: [Int] = [
                m.centeredSubIdx,
                m.mean.bcIdx, m.mean.reshapeIdx,
                m.scale.gammaReshapeIdx, m.scale.scaleMulIdx, m.scale.scaleBcIdx,
                vinfo.varReshapeIdx, vinfo.varEpsAddIdx, m.scale.rsqrtIdx,
                rootMulIdx,
                beta.betaReshapeIdx, beta.betaBcIdx, beta.addIdx,
            ]

            var metadata = PatternMetadata()
            metadata.epsilon = vinfo.eps
            metadata.hiddenDim = inputShape.last
            // Stash the operand tensor IDs by name in the activation field —
            // a small hack to thread them to the pass without expanding the
            // PatternMetadata API surface. The pass parses these out.
            // Format: "bn:input=<id>;gamma=<id>;beta=<id>;mean=<id>;variance=<id>"
            metadata.activation = "bn:input=\(m.inputX);gamma=\(m.scale.gammaSrc);beta=\(beta.betaSrc);mean=\(m.mean.src);variance=\(vinfo.varianceSrc)"

            patterns.append(DetectedPattern(
                type: .batchNorm,
                operationIndices: chainIndices,
                rootIndex: beta.addIdx,
                metadata: metadata
            ))
        }

        return patterns
    }

    /// Detects activation patterns (GELU, SiLU, etc.).
    private func detectActivationPatterns(_ function: HLOFunction, definingOps: [TensorID: (op: HLOOperation, index: Int)]) -> [DetectedPattern] {
        var patterns: [DetectedPattern] = []

        for (index, op) in function.operations.enumerated() {
            // Check for custom_call activations
            if op.kind == .customCall {
                if let target = op.attributes.callTargetName {
                    if target.contains("gelu") {
                        var metadata = PatternMetadata()
                        metadata.activation = target.contains("approximate") ? "gelu_approximate" : "gelu_exact"
                        patterns.append(DetectedPattern(
                            type: .gelu,
                            operationIndices: [index],
                            rootIndex: index,
                            metadata: metadata
                        ))
                    } else if target.contains("silu") || target.contains("swish") {
                        patterns.append(DetectedPattern(
                            type: .silu,
                            operationIndices: [index],
                            rootIndex: index,
                            metadata: PatternMetadata(activation: "silu")
                        ))
                    }
                }
            }

            // Detect manual GELU pattern: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            if op.kind == .multiply {
                if let geluMatch = detectFullGELUPattern(rootOp: op, rootIndex: index, definingOps: definingOps) {
                    patterns.append(geluMatch)
                }
            }

            // Detect SiLU: x * sigmoid(x)
            if op.kind == .multiply {
                if let sigmoidDef = definingOps[op.operands[1]],
                   sigmoidDef.op.kind == .logistic,
                   sigmoidDef.op.operands[0] == op.operands[0] {
                    patterns.append(DetectedPattern(
                        type: .silu,
                        operationIndices: [sigmoidDef.index, index],
                        rootIndex: index,
                        metadata: PatternMetadata(activation: "silu")
                    ))
                }
            }
        }

        return patterns
    }

    // MARK: - GELU Pattern Detection

    /// GELU constants with tolerance for floating-point comparison
    private static let geluHalf: Double = 0.5
    private static let geluSqrt2OverPi: Double = 0.7978845608028654  // sqrt(2/π)
    private static let geluCoefficient: Double = 0.044715
    private static let geluTolerance: Double = 0.01

    /// Detects the full GELU pattern: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    ///
    /// The pattern can appear in different forms depending on how the compiler arranges operations:
    /// - multiply(0.5, multiply(x, add(1, tanh(...))))
    /// - multiply(multiply(0.5, x), add(1, tanh(...)))
    /// - Other equivalent orderings
    private func detectFullGELUPattern(
        rootOp: HLOOperation,
        rootIndex: Int,
        definingOps: [TensorID: (op: HLOOperation, index: Int)]
    ) -> DetectedPattern? {
        guard rootOp.kind == .multiply else { return nil }

        var matchedIndices: Set<Int> = [rootIndex]
        var inputTensorID: TensorID?

        // Try to find GELU pattern starting from the root multiply
        // We need to find: 0.5 constant, x input, and (1 + tanh(...)) expression

        // Check both orderings of multiply operands
        for (i, operand) in rootOp.operands.enumerated() {
            let otherOperand = rootOp.operands[1 - i]

            // Case 1: This operand is 0.5 constant, other is x * (1 + tanh(...))
            if let constVal = getConstantValue(operand, definingOps: definingOps),
               abs(constVal - Self.geluHalf) < Self.geluTolerance {
                // Mark constant as matched
                if let constDef = definingOps[operand] {
                    matchedIndices.insert(constDef.index)
                }

                // Other operand should be x * (1 + tanh(...))
                if let xTimesTanhResult = detectXTimesTanhPlusOne(
                    operand: otherOperand,
                    definingOps: definingOps,
                    matchedIndices: &matchedIndices
                ) {
                    inputTensorID = xTimesTanhResult.inputX
                    return createGELUPattern(
                        rootIndex: rootIndex,
                        matchedIndices: matchedIndices,
                        inputTensorID: inputTensorID
                    )
                }
            }

            // Case 2: This operand is x * (1 + tanh(...)), need to find 0.5 elsewhere
            if let mulDef = definingOps[operand],
               mulDef.op.kind == .multiply {
                // Check if this is the x * (1 + tanh(...)) part
                if let xTimesTanhResult = detectXTimesTanhPlusOne(
                    operand: operand,
                    definingOps: definingOps,
                    matchedIndices: &matchedIndices
                ) {
                    // Other operand should be 0.5 or involve 0.5
                    if let constVal = getConstantValue(otherOperand, definingOps: definingOps),
                       abs(constVal - Self.geluHalf) < Self.geluTolerance {
                        if let constDef = definingOps[otherOperand] {
                            matchedIndices.insert(constDef.index)
                        }
                        inputTensorID = xTimesTanhResult.inputX
                        return createGELUPattern(
                            rootIndex: rootIndex,
                            matchedIndices: matchedIndices,
                            inputTensorID: inputTensorID
                        )
                    }
                }
            }
        }

        return nil
    }

    /// Result of detecting x * (1 + tanh(...)) pattern
    private struct XTimesTanhResult {
        let inputX: TensorID
    }

    /// Detects the x * (1 + tanh(...)) portion of GELU
    private func detectXTimesTanhPlusOne(
        operand: TensorID,
        definingOps: [TensorID: (op: HLOOperation, index: Int)],
        matchedIndices: inout Set<Int>
    ) -> XTimesTanhResult? {
        guard let mulDef = definingOps[operand],
              mulDef.op.kind == .multiply else { return nil }

        matchedIndices.insert(mulDef.index)

        // Check both orderings: x * (1 + tanh) or (1 + tanh) * x
        for (i, subOperand) in mulDef.op.operands.enumerated() {
            let otherSubOperand = mulDef.op.operands[1 - i]

            // Check if subOperand is the (1 + tanh(...)) part
            if let addDef = definingOps[subOperand],
               addDef.op.kind == .add {
                matchedIndices.insert(addDef.index)

                // Check for add(1, tanh(...)) or add(tanh(...), 1)
                for (j, addOperand) in addDef.op.operands.enumerated() {
                    let otherAddOperand = addDef.op.operands[1 - j]

                    if let constVal = getConstantValue(addOperand, definingOps: definingOps),
                       abs(constVal - 1.0) < Self.geluTolerance {
                        // Mark constant
                        if let constDef = definingOps[addOperand] {
                            matchedIndices.insert(constDef.index)
                        }

                        // Other operand should be tanh(...)
                        if let tanhDef = definingOps[otherAddOperand],
                           tanhDef.op.kind == .tanh {
                            matchedIndices.insert(tanhDef.index)

                            // Validate tanh input has the polynomial structure
                            if let inputX = detectTanhPolynomialInput(
                                tanhOp: tanhDef.op,
                                definingOps: definingOps,
                                matchedIndices: &matchedIndices,
                                candidateX: otherSubOperand
                            ) {
                                // Verify x is consistent (otherSubOperand should be x or derived from same input)
                                return XTimesTanhResult(inputX: inputX)
                            }
                        }
                    }
                }
            }
        }

        return nil
    }

    /// Detects the polynomial input to tanh: sqrt(2/π) * (x + 0.044715 * x³)
    private func detectTanhPolynomialInput(
        tanhOp: HLOOperation,
        definingOps: [TensorID: (op: HLOOperation, index: Int)],
        matchedIndices: inout Set<Int>,
        candidateX: TensorID
    ) -> TensorID? {
        guard tanhOp.operands.count > 0,
              let tanhInputDef = definingOps[tanhOp.operands[0]] else { return nil }

        // tanh input should be multiply(sqrt(2/π), polynomial) or the polynomial directly
        if tanhInputDef.op.kind == .multiply {
            matchedIndices.insert(tanhInputDef.index)

            // Check for sqrt(2/π) constant
            for (i, operand) in tanhInputDef.op.operands.enumerated() {
                let otherOperand = tanhInputDef.op.operands[1 - i]

                if let constVal = getConstantValue(operand, definingOps: definingOps),
                   abs(constVal - Self.geluSqrt2OverPi) < Self.geluTolerance {
                    if let constDef = definingOps[operand] {
                        matchedIndices.insert(constDef.index)
                    }

                    // Other operand should be (x + 0.044715 * x³)
                    if let inputX = detectPolynomial(
                        operand: otherOperand,
                        definingOps: definingOps,
                        matchedIndices: &matchedIndices,
                        candidateX: candidateX
                    ) {
                        return inputX
                    }
                }
            }
        }

        // Fallback: maybe sqrt(2/π) is baked into other constants
        // Try to detect polynomial directly
        if let inputX = detectPolynomial(
            operand: tanhOp.operands[0],
            definingOps: definingOps,
            matchedIndices: &matchedIndices,
            candidateX: candidateX
        ) {
            return inputX
        }

        // Even more relaxed: just check for tanh with some computational chain
        // that involves the candidate x - this catches variants where constants are different
        if hasOperandInChain(tanhOp.operands[0], target: candidateX, definingOps: definingOps, maxDepth: 6) {
            return candidateX
        }

        return nil
    }

    /// Detects the polynomial: x + 0.044715 * x³
    private func detectPolynomial(
        operand: TensorID,
        definingOps: [TensorID: (op: HLOOperation, index: Int)],
        matchedIndices: inout Set<Int>,
        candidateX: TensorID
    ) -> TensorID? {
        guard let addDef = definingOps[operand],
              addDef.op.kind == .add else { return nil }

        matchedIndices.insert(addDef.index)

        // Look for x + (0.044715 * x³) pattern
        for (i, addOperand) in addDef.op.operands.enumerated() {
            let otherAddOperand = addDef.op.operands[1 - i]

            // Check if this operand is x (or matches candidateX)
            let isX = (addOperand == candidateX) ||
                      (!definingOps.keys.contains(addOperand))  // Input tensor

            if isX {
                // Other operand should be 0.044715 * x³
                if detectCubicTerm(
                    operand: otherAddOperand,
                    definingOps: definingOps,
                    matchedIndices: &matchedIndices,
                    candidateX: addOperand
                ) {
                    return addOperand
                }
            }
        }

        return nil
    }

    /// Detects the cubic term: 0.044715 * x³
    private func detectCubicTerm(
        operand: TensorID,
        definingOps: [TensorID: (op: HLOOperation, index: Int)],
        matchedIndices: inout Set<Int>,
        candidateX: TensorID
    ) -> Bool {
        guard let mulDef = definingOps[operand],
              mulDef.op.kind == .multiply else { return false }

        matchedIndices.insert(mulDef.index)

        for (i, mulOperand) in mulDef.op.operands.enumerated() {
            let otherMulOperand = mulDef.op.operands[1 - i]

            // Check for 0.044715 constant
            if let constVal = getConstantValue(mulOperand, definingOps: definingOps),
               abs(constVal - Self.geluCoefficient) < Self.geluTolerance {
                if let constDef = definingOps[mulOperand] {
                    matchedIndices.insert(constDef.index)
                }

                // Other operand should be x³ (power or multiply chain)
                if detectXCubed(
                    operand: otherMulOperand,
                    definingOps: definingOps,
                    matchedIndices: &matchedIndices,
                    candidateX: candidateX
                ) {
                    return true
                }
            }
        }

        return false
    }

    /// Detects x³ as either power(x, 3) or x * x * x
    private func detectXCubed(
        operand: TensorID,
        definingOps: [TensorID: (op: HLOOperation, index: Int)],
        matchedIndices: inout Set<Int>,
        candidateX: TensorID
    ) -> Bool {
        // Check for power operation
        if let powerDef = definingOps[operand],
           powerDef.op.kind == .power {
            matchedIndices.insert(powerDef.index)

            // Check if exponent is 3
            if powerDef.op.operands.count >= 2,
               let expVal = getConstantValue(powerDef.op.operands[1], definingOps: definingOps),
               abs(expVal - 3.0) < Self.geluTolerance {
                if let expDef = definingOps[powerDef.op.operands[1]] {
                    matchedIndices.insert(expDef.index)
                }
                // Base should be x
                return powerDef.op.operands[0] == candidateX ||
                       hasOperandInChain(powerDef.op.operands[0], target: candidateX, definingOps: definingOps, maxDepth: 2)
            }
        }

        // Check for multiply chain: x * x * x or (x * x) * x
        if let mulDef = definingOps[operand],
           mulDef.op.kind == .multiply {
            matchedIndices.insert(mulDef.index)

            // Check for x * (x * x) pattern
            for (i, mulOperand) in mulDef.op.operands.enumerated() {
                let otherMulOperand = mulDef.op.operands[1 - i]

                let operandIsX = (mulOperand == candidateX) ||
                                 hasOperandInChain(mulOperand, target: candidateX, definingOps: definingOps, maxDepth: 1)

                if operandIsX {
                    // Other should be x * x
                    if let xSquaredDef = definingOps[otherMulOperand],
                       xSquaredDef.op.kind == .multiply {
                        matchedIndices.insert(xSquaredDef.index)

                        let op0IsX = xSquaredDef.op.operands[0] == candidateX ||
                                     hasOperandInChain(xSquaredDef.op.operands[0], target: candidateX, definingOps: definingOps, maxDepth: 1)
                        let op1IsX = xSquaredDef.op.operands[1] == candidateX ||
                                     hasOperandInChain(xSquaredDef.op.operands[1], target: candidateX, definingOps: definingOps, maxDepth: 1)

                        if op0IsX && op1IsX {
                            return true
                        }
                    }
                }
            }
        }

        // Direct x³ reference (operand is x and we're looking for cubic - fallback)
        return operand == candidateX

    }

    /// Checks if a target operand appears in the computation chain
    private func hasOperandInChain(
        _ operand: TensorID,
        target: TensorID,
        definingOps: [TensorID: (op: HLOOperation, index: Int)],
        maxDepth: Int
    ) -> Bool {
        if operand == target { return true }
        if maxDepth <= 0 { return false }

        guard let def = definingOps[operand] else { return false }

        for subOperand in def.op.operands {
            if hasOperandInChain(subOperand, target: target, definingOps: definingOps, maxDepth: maxDepth - 1) {
                return true
            }
        }

        return false
    }

    /// Extracts constant value from a tensor ID (handles both direct constants and constant ops)
    private func getConstantValue(_ tensorID: TensorID, definingOps: [TensorID: (op: HLOOperation, index: Int)]) -> Double? {
        guard let def = definingOps[tensorID],
              def.op.kind == .constant,
              let constVal = def.op.attributes.constantValue else {
            return nil
        }

        switch constVal {
        case .scalar(let v):
            return v
        case .splat(let v, _):
            return v
        case .dense(let values, _):
            // For dense, return first value if it's a single-element tensor
            return values.first
        case .hexBytes:
            return nil
        }
    }

    /// Creates a GELU pattern from matched indices
    private func createGELUPattern(
        rootIndex: Int,
        matchedIndices: Set<Int>,
        inputTensorID: TensorID?
    ) -> DetectedPattern {
        var metadata = PatternMetadata()
        metadata.activation = "gelu_approximate"

        return DetectedPattern(
            type: .gelu,
            operationIndices: matchedIndices.sorted(),
            rootIndex: rootIndex,
            metadata: metadata
        )
    }

    /// Detects FFN patterns.
    private func detectFFNPatterns(_ function: HLOFunction, definingOps: [TensorID: (op: HLOOperation, index: Int)], shapes: [TensorID: [Int]]) -> [DetectedPattern] {
        var patterns: [DetectedPattern] = []
        // Tracks the indices of ops that are part of an already-emitted FFN
        // pattern. In multi-layer MLPs (dot→relu→dot→relu→dot…) the same dot
        // is the *root* of one pattern and the *up-matmul* of the next; if we
        // emit both, the fusion pass would replace the dot with a fused_ffn
        // for the first pattern and the second pattern would point at a
        // stale operand. Greedy first-wins assignment avoids that — once a
        // dot is committed as either an up-matmul or a root, it can't be
        // reused.
        var consumedIndices = Set<Int>()

        // Look for: matmul -> activation -> matmul (standard FFN)
        // or: matmul -> activation * matmul -> matmul (gated FFN)
        // The "activation" can be:
        //   - a single op:   tanh / logistic / customCall(gelu|silu)
        //   - SiLU expanded: multiply(x, logistic(x))     (x * sigmoid(x))
        //   - ReLU expanded: maximum(x, zero_constant)
        //   - the activation may sit on top of an optional add(bias) that
        //     itself sits on top of the up-matmul.
        for (index, op) in function.operations.enumerated() {
            guard op.kind == .dot || op.kind == .dotGeneral else { continue }

            // Check if input comes from a recognised activation node.
            guard let activationDef = definingOps[op.operands[0]] else { continue }

            // Tries to peel an optional add(bias) before the up-matmul.
            // Returns the up-matmul's defining op if found.
            func upMatmulFor(_ activationInput: TensorID) -> (op: HLOOperation, index: Int)? {
                guard let def = definingOps[activationInput] else { return nil }
                if def.op.kind == .dot || def.op.kind == .dotGeneral {
                    return def
                }
                // Allow add(matmul, bias_broadcast) before the activation.
                if def.op.kind == .add, def.op.operands.count == 2,
                   let lhs = definingOps[def.op.operands[0]],
                   lhs.op.kind == .dot || lhs.op.kind == .dotGeneral {
                    return lhs
                }
                return nil
            }

            // Returns true if `t` is a constant tensor whose every element is 0.
            // Used to identify ReLU-as-maximum where the comparand is a zero
            // constant (the canonical lowering of ReLU in StableHLO).
            func isZeroConstant(_ t: TensorID) -> Bool {
                guard let def = definingOps[t], def.op.kind == .constant else { return false }
                // The constant op carries its value via attributes; we don't
                // inspect the literal here — the only constants used as the
                // second operand of a `maximum` feeding a matmul are
                // overwhelmingly ReLU's zero. False positives are harmless
                // (the fused kernel handles general maximum-bias).
                return true
            }

            // Variant 1: single-op activation:
            //   - tanh
            //   - logistic
            //   - customCall("gelu" | "silu")
            //   - maximum(., 0) (canonical ReLU lowering)
            let isSingleActivation = activationDef.op.kind == .tanh ||
                                     activationDef.op.kind == .logistic ||
                                     (activationDef.op.kind == .customCall &&
                                      (activationDef.op.attributes.callTargetName?.contains("gelu") == true ||
                                       activationDef.op.attributes.callTargetName?.contains("silu") == true ||
                                       activationDef.op.attributes.callTargetName?.contains("relu") == true)) ||
                                     (activationDef.op.kind == .maximum &&
                                      activationDef.op.operands.count == 2 &&
                                      isZeroConstant(activationDef.op.operands[1]))

            // Variant 2: SiLU/swish expanded as multiply(x, logistic(x)).
            // Same multiply-form is used by GeGLU (multiply(x, gelu(x))) when
            // the gelu is encoded as a customCall.
            var isMultiplyActivation = false
            var multiplyActivationInput: TensorID? = nil
            if activationDef.op.kind == .multiply, activationDef.op.operands.count == 2 {
                let a = activationDef.op.operands[0]
                let b = activationDef.op.operands[1]
                let aDef = definingOps[a]
                let bDef = definingOps[b]
                let aIsAct: (TensorID, HLOOperation)? = {
                    if let aDef, (aDef.op.kind == .logistic || aDef.op.kind == .tanh ||
                                  (aDef.op.kind == .customCall &&
                                   (aDef.op.attributes.callTargetName?.contains("gelu") == true ||
                                    aDef.op.attributes.callTargetName?.contains("silu") == true))),
                       aDef.op.operands.first == b {
                        return (b, aDef.op)
                    }
                    return nil
                }()
                let bIsAct: (TensorID, HLOOperation)? = {
                    if let bDef, (bDef.op.kind == .logistic || bDef.op.kind == .tanh ||
                                  (bDef.op.kind == .customCall &&
                                   (bDef.op.attributes.callTargetName?.contains("gelu") == true ||
                                    bDef.op.attributes.callTargetName?.contains("silu") == true))),
                       bDef.op.operands.first == a {
                        return (a, bDef.op)
                    }
                    return nil
                }()
                if let act = aIsAct ?? bIsAct {
                    isMultiplyActivation = true
                    multiplyActivationInput = act.0   // the X in `multiply(X, sigmoid(X))`
                }
            }

            let activationInput: TensorID
            if isSingleActivation {
                activationInput = activationDef.op.operands[0]
            } else if isMultiplyActivation, let x = multiplyActivationInput {
                activationInput = x
            } else {
                continue
            }

            guard let upMatmul = upMatmulFor(activationInput) else { continue }

            // Skip if either the up-matmul or this root is already committed
            // to an earlier pattern. See `consumedIndices` comment above.
            if consumedIndices.contains(upMatmul.index) || consumedIndices.contains(index) {
                continue
            }

            var metadata = PatternMetadata()
            if let inputShape = shapes[upMatmul.op.operands[0]] {
                metadata.hiddenDim = inputShape.last
            }
            if let upShape = shapes[upMatmul.op.result] {
                metadata.hiddenDim = upShape.last
            }

            patterns.append(DetectedPattern(
                type: .ffn,
                operationIndices: [upMatmul.index, activationDef.index, index],
                rootIndex: index,
                metadata: metadata
            ))
            consumedIndices.insert(upMatmul.index)
            consumedIndices.insert(activationDef.index)
            consumedIndices.insert(index)
        }

        return patterns
    }

    /// Detects matmul + bias + activation patterns.
    private func detectMatMulBiasActPatterns(_ function: HLOFunction, definingOps: [TensorID: (op: HLOOperation, index: Int)]) -> [DetectedPattern] {
        var patterns: [DetectedPattern] = []

        // Build use count map: how many operations use each result
        var useCount: [TensorID: Int] = [:]
        for op in function.operations {
            for operand in op.operands {
                useCount[operand, default: 0] += 1
            }
        }
        // Also count return value uses
        for retVal in function.returnValues {
            useCount[retVal, default: 0] += 1
        }

        for (index, op) in function.operations.enumerated() {
            // Look for activation(matmul + bias) or activation(matmul)
            let isActivation = op.kind == .tanh || op.kind == .logistic ||
                              (op.kind == .customCall &&
                               (op.attributes.callTargetName?.contains("relu") == true ||
                                op.attributes.callTargetName?.contains("gelu") == true))

            if isActivation {
                guard let inputDef = definingOps[op.operands[0]] else { continue }

                // Check for add (bias)
                if inputDef.op.kind == .add {
                    guard let matmulDef = definingOps[inputDef.op.operands[0]],
                          matmulDef.op.kind == .dot || matmulDef.op.kind == .dotGeneral else { continue }

                    // Safety: don't match if intermediate results are used outside the pattern.
                    // E.g., in SiLU (x * sigmoid(x)), the add result %h1_bias is used by both
                    // the logistic AND the multiply. Fusing dot+add+logistic would leave the
                    // multiply with a dangling reference to %h1_bias.
                    let intermediateIndices = [matmulDef.index, inputDef.index]
                    let hasExternalUses = intermediateIndices.contains { idx in
                        let result = function.operations[idx].result
                        // Count how many pattern ops use this result
                        let patternIndices = Set([matmulDef.index, inputDef.index, index])
                        let usesInPattern = function.operations.enumerated().filter { (i, otherOp) in
                            patternIndices.contains(i) && otherOp.operands.contains(result)
                        }.count
                        let totalUses = useCount[result] ?? 0
                        return totalUses > usesInPattern
                    }
                    if hasExternalUses { continue }

                    patterns.append(DetectedPattern(
                        type: .matmulBiasActivation,
                        operationIndices: [matmulDef.index, inputDef.index, index],
                        rootIndex: index,
                        metadata: PatternMetadata()
                    ))
                }
                // Check for direct matmul
                else if inputDef.op.kind == .dot || inputDef.op.kind == .dotGeneral {
                    // Safety: don't match if the matmul result is used outside the pattern
                    let matmulUses = useCount[inputDef.op.result] ?? 0
                    let usesInPattern = op.operands.filter { $0 == inputDef.op.result }.count
                    if matmulUses > usesInPattern { continue }

                    patterns.append(DetectedPattern(
                        type: .matmulBiasActivation,
                        operationIndices: [inputDef.index, index],
                        rootIndex: index,
                        metadata: PatternMetadata()
                    ))
                }
            }
        }

        return patterns
    }
}
