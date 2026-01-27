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

        // Detect patterns from most specific to least specific
        patterns.append(contentsOf: detectAttentionPatterns(function, definingOps: definingOps, shapes: shapes))
        patterns.append(contentsOf: detectNormPatterns(function, definingOps: definingOps, shapes: shapes))
        patterns.append(contentsOf: detectActivationPatterns(function, definingOps: definingOps))
        patterns.append(contentsOf: detectFFNPatterns(function, definingOps: definingOps, shapes: shapes))
        patterns.append(contentsOf: detectMatMulBiasActPatterns(function, definingOps: definingOps))

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

        // Look for Q @ K^T -> softmax -> @ V pattern
        for (index, op) in function.operations.enumerated() {
            // Look for final matmul (attention @ V)
            guard op.kind == .dot || op.kind == .dotGeneral else { continue }

            // Check if first operand comes from softmax
            guard let softmaxDef = definingOps[op.operands[0]],
                  softmaxDef.op.kind == .customCall,
                  softmaxDef.op.attributes.callTargetName?.contains("softmax") == true else { continue }

            // Check if softmax input comes from matmul (Q @ K^T)
            guard let qkMatmulDef = definingOps[softmaxDef.op.operands[0]],
                  qkMatmulDef.op.kind == .dot || qkMatmulDef.op.kind == .dotGeneral else { continue }

            // Found attention pattern!
            let operationIndices = [qkMatmulDef.index, softmaxDef.index, index]

            // Extract configuration from shapes
            var metadata = PatternMetadata()
            if let qShape = shapes[qkMatmulDef.op.operands[0]] {
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

            // Also detect manual norm patterns: reduce -> rsqrt -> mul
            if op.kind == .multiply {
                guard let rsqrtDef = definingOps[op.operands[1]],
                      rsqrtDef.op.kind == .rsqrt else { continue }
                guard let reduceDef = definingOps[rsqrtDef.op.operands[0]],
                      reduceDef.op.kind == .reduce else { continue }

                patterns.append(DetectedPattern(
                    type: .rmsNorm,
                    operationIndices: [reduceDef.index, rsqrtDef.index, index],
                    rootIndex: index,
                    metadata: PatternMetadata()
                ))
            }
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

            // Detect manual GELU: x * 0.5 * (1 + tanh(...))
            // This is a simplified check
            if op.kind == .multiply {
                if let tanhDef = definingOps[op.operands[1]],
                   tanhDef.op.kind == .tanh {
                    patterns.append(DetectedPattern(
                        type: .gelu,
                        operationIndices: [tanhDef.index, index],
                        rootIndex: index,
                        metadata: PatternMetadata(activation: "gelu_approximate")
                    ))
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

    /// Detects FFN patterns.
    private func detectFFNPatterns(_ function: HLOFunction, definingOps: [TensorID: (op: HLOOperation, index: Int)], shapes: [TensorID: [Int]]) -> [DetectedPattern] {
        var patterns: [DetectedPattern] = []

        // Look for: matmul -> activation -> matmul (standard FFN)
        // or: matmul -> activation * matmul -> matmul (gated FFN)
        for (index, op) in function.operations.enumerated() {
            guard op.kind == .dot || op.kind == .dotGeneral else { continue }

            // Check if input comes from activation
            guard let activationDef = definingOps[op.operands[0]] else { continue }

            let isActivation = activationDef.op.kind == .tanh ||
                              activationDef.op.kind == .logistic ||
                              (activationDef.op.kind == .customCall &&
                               (activationDef.op.attributes.callTargetName?.contains("gelu") == true ||
                                activationDef.op.attributes.callTargetName?.contains("silu") == true))

            guard isActivation else { continue }

            // Check if activation input comes from matmul
            guard let upMatmulDef = definingOps[activationDef.op.operands[0]],
                  upMatmulDef.op.kind == .dot || upMatmulDef.op.kind == .dotGeneral else { continue }

            var metadata = PatternMetadata()
            if let inputShape = shapes[upMatmulDef.op.operands[0]] {
                metadata.hiddenDim = inputShape.last
            }
            if let upShape = shapes[upMatmulDef.op.result] {
                metadata.hiddenDim = upShape.last
            }

            patterns.append(DetectedPattern(
                type: .ffn,
                operationIndices: [upMatmulDef.index, activationDef.index, index],
                rootIndex: index,
                metadata: metadata
            ))
        }

        return patterns
    }

    /// Detects matmul + bias + activation patterns.
    private func detectMatMulBiasActPatterns(_ function: HLOFunction, definingOps: [TensorID: (op: HLOOperation, index: Int)]) -> [DetectedPattern] {
        var patterns: [DetectedPattern] = []

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

                    patterns.append(DetectedPattern(
                        type: .matmulBiasActivation,
                        operationIndices: [matmulDef.index, inputDef.index, index],
                        rootIndex: index,
                        metadata: PatternMetadata()
                    ))
                }
                // Check for direct matmul
                else if inputDef.op.kind == .dot || inputDef.op.kind == .dotGeneral {
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
