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
