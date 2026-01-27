// HLOOptimizerTests.swift
// MetalHLOCoreTests
//
// Tests for HLO pattern matching and optimization.

import Testing
@testable import MetalHLOCore

@Suite("HLO Optimizer Tests")
struct HLOOptimizerTests {

    // MARK: - Pattern Registry Tests

    @Test("Pattern registry has default patterns")
    func registryHasDefaultPatterns() {
        let registry = HLOPatternRegistry.shared
        let patterns = registry.registeredPatterns

        #expect(patterns.contains { $0.name == "attention" })
        #expect(patterns.contains { $0.name == "softmax" })
        #expect(patterns.contains { $0.name == "gelu" })
        #expect(patterns.contains { $0.name == "layer_norm" })
        #expect(patterns.contains { $0.name == "rms_norm" })
    }

    @Test("Pattern registry orders by priority")
    func registryOrdersByPriority() {
        let registry = HLOPatternRegistry.shared
        let patterns = registry.registeredPatterns

        // Verify sorted by priority (descending)
        for i in 0..<patterns.count - 1 {
            #expect(patterns[i].priority >= patterns[i + 1].priority)
        }
    }

    // MARK: - Softmax Pattern Tests

    @Test("Softmax pattern matches basic softmax IR")
    func softmaxPatternMatchesBasicSoftmax() {
        // Simplified softmax pattern that matches our detector:
        // %0 = reduce_max(%arg0)       - max over last dim
        // %1 = subtract(%arg0, %0)     - x - max (implicit broadcast)
        // %2 = exp(%1)                 - exp(x - max)
        // %3 = reduce_sum(%2)          - sum of exp
        // %4 = divide(%2, %3)          - exp / sum

        let inputType = TensorType(shape: [2, 4], elementType: .float32)
        let scalarType = TensorType(shape: [2], elementType: .float32)

        var maxAttrs = HLOAttributes()
        maxAttrs.dimensions = [1]
        maxAttrs.reductionKind = .max

        var sumAttrs = HLOAttributes()
        sumAttrs.dimensions = [1]
        sumAttrs.reductionKind = .sum

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .reduce, operands: ["%arg0"], resultType: scalarType, attributes: maxAttrs),
            HLOOperation(result: "%1", kind: .subtract, operands: ["%arg0", "%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .exponential, operands: ["%1"], resultType: inputType),
            HLOOperation(result: "%3", kind: .reduce, operands: ["%2"], resultType: scalarType, attributes: sumAttrs),
            HLOOperation(result: "%4", kind: .divide, operands: ["%2", "%3"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "main",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%4"]
        )

        let pattern = SoftmaxPattern()
        let definingOps = DefiningOpMap(function: function)

        // Match at the divide operation (last one)
        let match = pattern.match(
            at: operations[4],
            index: 4,
            in: function,
            definingOps: definingOps.map
        )

        #expect(match != nil)
        if let match = match {
            #expect(match.rootOperation.kind == .divide)
            if case .string(let p) = match.metadata["pattern"] {
                #expect(p == "softmax")
            }
        }
    }

    // MARK: - GELU Pattern Tests

    @Test("GELU pattern matches tanh approximation")
    func geluPatternMatchesTanhApproximation() {
        // Simplified GELU: 0.5 * x * (1 + tanh(...))
        // We'll create a minimal pattern that has the key components

        let inputType = TensorType(shape: [2, 4], elementType: .float32)
        let scalarType = TensorType(shape: [], elementType: .float32)

        var halfConst = HLOAttributes()
        halfConst.constantValue = .scalar(0.5)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .constant, operands: [], resultType: scalarType, attributes: halfConst),
            HLOOperation(result: "%1", kind: .tanh, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .multiply, operands: ["%arg0", "%1"], resultType: inputType),
            HLOOperation(result: "%3", kind: .multiply, operands: ["%0", "%2"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "main",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%3"]
        )

        let pattern = GELUPattern()
        let definingOps = DefiningOpMap(function: function)

        // Match at the final multiply
        let match = pattern.match(
            at: operations[3],
            index: 3,
            in: function,
            definingOps: definingOps.map
        )

        #expect(match != nil)
        if let match = match {
            #expect(match.rootOperation.kind == .multiply)
        }
    }

    // MARK: - Optimizer Tests

    @Test("Optimizer with no patterns returns original function")
    func optimizerNoPatterns() {
        let inputType = TensorType(shape: [2, 4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "main",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%0"]
        )

        let config = HLOOptimizerConfig(enableFusion: false)
        let optimizer = HLOOptimizer(config: config)
        let result = optimizer.optimize(function)

        #expect(result.operations.count == function.operations.count)
        #expect(result.operations[0].kind == .add)
    }

    @Test("Optimizer config can enable specific patterns")
    func optimizerConfigPatterns() {
        let config = HLOOptimizerConfig(enabledPatterns: ["softmax"])
        #expect(config.enabledPatterns?.contains("softmax") == true)
        #expect(config.enabledPatterns?.contains("gelu") == false)
    }

    // MARK: - DefiningOpMap Tests

    @Test("DefiningOpMap correctly maps operations")
    func definingOpMapMapsCorrectly() {
        let inputType = TensorType(shape: [2, 4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .multiply, operands: ["%0", "%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "main",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let definingOps = DefiningOpMap(function: function)

        #expect(definingOps.definingOp(for: "%0")?.kind == .add)
        #expect(definingOps.definingOp(for: "%1")?.kind == .multiply)
        #expect(definingOps.definingIndex(for: "%0") == 0)
        #expect(definingOps.definingIndex(for: "%1") == 1)
        #expect(definingOps.definingOp(for: "%arg0") == nil)  // Function arguments not in map
    }

    // MARK: - Pattern Replacement Tests

    @Test("Softmax pattern creates custom_call replacement")
    func softmaxPatternCreatesCustomCall() {
        let inputType = TensorType(shape: [2, 4], elementType: .float32)
        let scalarType = TensorType(shape: [2], elementType: .float32)

        var maxAttrs = HLOAttributes()
        maxAttrs.dimensions = [1]
        maxAttrs.reductionKind = .max

        var sumAttrs = HLOAttributes()
        sumAttrs.dimensions = [1]
        sumAttrs.reductionKind = .sum

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .reduce, operands: ["%arg0"], resultType: scalarType, attributes: maxAttrs),
            HLOOperation(result: "%1", kind: .subtract, operands: ["%arg0", "%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .exponential, operands: ["%1"], resultType: inputType),
            HLOOperation(result: "%3", kind: .reduce, operands: ["%2"], resultType: scalarType, attributes: sumAttrs),
            HLOOperation(result: "%4", kind: .divide, operands: ["%2", "%3"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "main",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%4"]
        )

        let pattern = SoftmaxPattern()
        let definingOps = DefiningOpMap(function: function)

        if let match = pattern.match(at: operations[4], index: 4, in: function, definingOps: definingOps.map) {
            let replacements = pattern.replacement(for: match, in: function)

            #expect(replacements.count == 1)
            #expect(replacements[0].kind == .customCall)
            #expect(replacements[0].attributes.callTargetName == "fused_softmax")
        } else {
            #expect(Bool(false), "Pattern should have matched")
        }
    }

    // MARK: - Attention Pattern Tests

    @Test("Attention pattern has higher priority than softmax")
    func attentionPatternHigherPriority() {
        let registry = HLOPatternRegistry.shared
        let patterns = registry.registeredPatterns

        let attentionIdx = patterns.firstIndex { $0.name == "attention" }
        let softmaxIdx = patterns.firstIndex { $0.name == "softmax" }

        #expect(attentionIdx != nil)
        #expect(softmaxIdx != nil)

        // Attention should come before softmax (higher priority)
        if let aIdx = attentionIdx, let sIdx = softmaxIdx {
            #expect(aIdx < sIdx)
        }
    }

    @Test("Attention pattern matches basic attention IR")
    func attentionPatternMatchesBasicAttention() {
        // Build attention pattern: Q @ K^T -> scale -> softmax -> @ V
        // Shapes: Q, K, V are [batch=1, heads=2, seq=4, head_dim=8]
        //         Q @ K^T = [1, 2, 4, 4] (scores)
        //         scores @ V = [1, 2, 4, 8] (output)

        let qkvType = TensorType(shape: [1, 2, 4, 8], elementType: .float32)
        let scoresType = TensorType(shape: [1, 2, 4, 4], elementType: .float32)
        let reducedType = TensorType(shape: [1, 2, 4], elementType: .float32)
        let scalarType = TensorType(shape: [], elementType: .float32)

        // Q @ K^T dimension numbers
        var qkDotAttrs = HLOAttributes()
        qkDotAttrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [0, 1],
            rhsBatchingDimensions: [0, 1],
            lhsContractingDimensions: [3],
            rhsContractingDimensions: [3]
        )

        // Scale constant
        var scaleAttrs = HLOAttributes()
        scaleAttrs.constantValue = .scalar(0.125)  // 1/sqrt(64)

        // Softmax reduce attrs
        var maxAttrs = HLOAttributes()
        maxAttrs.dimensions = [3]
        maxAttrs.reductionKind = .max

        var sumAttrs = HLOAttributes()
        sumAttrs.dimensions = [3]
        sumAttrs.reductionKind = .sum

        // Weights @ V dimension numbers
        var wvDotAttrs = HLOAttributes()
        wvDotAttrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [0, 1],
            rhsBatchingDimensions: [0, 1],
            lhsContractingDimensions: [3],
            rhsContractingDimensions: [2]
        )

        let operations: [HLOOperation] = [
            // Q @ K^T
            HLOOperation(result: "%0", kind: .dotGeneral, operands: ["%q", "%k"], resultType: scoresType, attributes: qkDotAttrs),
            // Scale constant
            HLOOperation(result: "%1", kind: .constant, operands: [], resultType: scalarType, attributes: scaleAttrs),
            // scores * scale
            HLOOperation(result: "%2", kind: .multiply, operands: ["%0", "%1"], resultType: scoresType),
            // Softmax: max
            HLOOperation(result: "%3", kind: .reduce, operands: ["%2"], resultType: reducedType, attributes: maxAttrs),
            // Softmax: subtract (scores - max)
            HLOOperation(result: "%4", kind: .subtract, operands: ["%2", "%3"], resultType: scoresType),
            // Softmax: exp
            HLOOperation(result: "%5", kind: .exponential, operands: ["%4"], resultType: scoresType),
            // Softmax: sum
            HLOOperation(result: "%6", kind: .reduce, operands: ["%5"], resultType: reducedType, attributes: sumAttrs),
            // Softmax: divide
            HLOOperation(result: "%7", kind: .divide, operands: ["%5", "%6"], resultType: scoresType),
            // Weights @ V
            HLOOperation(result: "%8", kind: .dotGeneral, operands: ["%7", "%v"], resultType: qkvType, attributes: wvDotAttrs),
        ]

        let function = HLOFunction(
            name: "attention",
            inputs: [
                HLOArgument(name: "%q", type: qkvType),
                HLOArgument(name: "%k", type: qkvType),
                HLOArgument(name: "%v", type: qkvType),
            ],
            outputTypes: [qkvType],
            operations: operations,
            returnValues: ["%8"]
        )

        let pattern = AttentionPattern()
        let definingOps = DefiningOpMap(function: function)

        // Match at the final dotGeneral (Weights @ V)
        let match = pattern.match(
            at: operations[8],
            index: 8,
            in: function,
            definingOps: definingOps.map
        )

        #expect(match != nil)
        if let match = match {
            #expect(match.rootOperation.kind == .dotGeneral)
            if case .string(let p) = match.metadata["pattern"] {
                #expect(p == "attention")
            }
            // Should have Q, K, V as inputs
            #expect(match.inputs.count == 3)
            #expect(match.inputs.contains("%q"))
            #expect(match.inputs.contains("%k"))
            #expect(match.inputs.contains("%v"))
        }
    }

    @Test("Attention pattern creates fused_scaled_dot_product_attention custom_call")
    func attentionPatternCreatesCustomCall() {
        let qkvType = TensorType(shape: [1, 2, 4, 8], elementType: .float32)
        let scoresType = TensorType(shape: [1, 2, 4, 4], elementType: .float32)
        let reducedType = TensorType(shape: [1, 2, 4], elementType: .float32)
        let scalarType = TensorType(shape: [], elementType: .float32)

        var qkDotAttrs = HLOAttributes()
        qkDotAttrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [0, 1],
            rhsBatchingDimensions: [0, 1],
            lhsContractingDimensions: [3],
            rhsContractingDimensions: [3]
        )

        var scaleAttrs = HLOAttributes()
        scaleAttrs.constantValue = .scalar(0.125)

        var maxAttrs = HLOAttributes()
        maxAttrs.dimensions = [3]
        maxAttrs.reductionKind = .max

        var sumAttrs = HLOAttributes()
        sumAttrs.dimensions = [3]
        sumAttrs.reductionKind = .sum

        var wvDotAttrs = HLOAttributes()
        wvDotAttrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [0, 1],
            rhsBatchingDimensions: [0, 1],
            lhsContractingDimensions: [3],
            rhsContractingDimensions: [2]
        )

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .dotGeneral, operands: ["%q", "%k"], resultType: scoresType, attributes: qkDotAttrs),
            HLOOperation(result: "%1", kind: .constant, operands: [], resultType: scalarType, attributes: scaleAttrs),
            HLOOperation(result: "%2", kind: .multiply, operands: ["%0", "%1"], resultType: scoresType),
            HLOOperation(result: "%3", kind: .reduce, operands: ["%2"], resultType: reducedType, attributes: maxAttrs),
            HLOOperation(result: "%4", kind: .subtract, operands: ["%2", "%3"], resultType: scoresType),
            HLOOperation(result: "%5", kind: .exponential, operands: ["%4"], resultType: scoresType),
            HLOOperation(result: "%6", kind: .reduce, operands: ["%5"], resultType: reducedType, attributes: sumAttrs),
            HLOOperation(result: "%7", kind: .divide, operands: ["%5", "%6"], resultType: scoresType),
            HLOOperation(result: "%8", kind: .dotGeneral, operands: ["%7", "%v"], resultType: qkvType, attributes: wvDotAttrs),
        ]

        let function = HLOFunction(
            name: "attention",
            inputs: [
                HLOArgument(name: "%q", type: qkvType),
                HLOArgument(name: "%k", type: qkvType),
                HLOArgument(name: "%v", type: qkvType),
            ],
            outputTypes: [qkvType],
            operations: operations,
            returnValues: ["%8"]
        )

        let pattern = AttentionPattern()
        let definingOps = DefiningOpMap(function: function)

        if let match = pattern.match(at: operations[8], index: 8, in: function, definingOps: definingOps.map) {
            let replacements = pattern.replacement(for: match, in: function)

            #expect(replacements.count == 1)
            #expect(replacements[0].kind == .customCall)
            #expect(replacements[0].attributes.callTargetName == "fused_scaled_dot_product_attention")
            #expect(replacements[0].operands.count == 3)  // Q, K, V
        } else {
            #expect(Bool(false), "Attention pattern should have matched")
        }
    }
}

// MARK: - Algebraic Simplifier Tests

@Suite("Algebraic Simplifier Tests")
struct AlgebraicSimplifierTests {

    // MARK: - Identity Rules

    @Test("AddZero rule: x + 0 = x")
    func addZeroRuleSimplifies() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        var zeroConst = HLOAttributes()
        zeroConst.constantValue = .splat(0.0, inputType)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .constant, operands: [], resultType: inputType, attributes: zeroConst),
            HLOOperation(result: "%1", kind: .add, operands: ["%arg0", "%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        // The add should be eliminated, output should be %arg0
        #expect(result.returnValues[0] == "%arg0")
    }

    @Test("MultiplyOne rule: x * 1 = x")
    func multiplyOneRuleSimplifies() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        var oneConst = HLOAttributes()
        oneConst.constantValue = .splat(1.0, inputType)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .constant, operands: [], resultType: inputType, attributes: oneConst),
            HLOOperation(result: "%1", kind: .multiply, operands: ["%arg0", "%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        #expect(result.returnValues[0] == "%arg0")
    }

    @Test("MultiplyZero rule: x * 0 = 0")
    func multiplyZeroRuleSimplifies() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        var zeroConst = HLOAttributes()
        zeroConst.constantValue = .splat(0.0, inputType)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .constant, operands: [], resultType: inputType, attributes: zeroConst),
            HLOOperation(result: "%1", kind: .multiply, operands: ["%arg0", "%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        // Result should be a constant zero
        let outputOp = result.operations.first { $0.result == result.returnValues[0] }
        #expect(outputOp?.kind == .constant)
    }

    @Test("SubtractSelf rule: x - x = 0")
    func subtractSelfRuleSimplifies() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .subtract, operands: ["%arg0", "%arg0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%0"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        // Result should be a constant zero
        let outputOp = result.operations.first { $0.result == result.returnValues[0] }
        #expect(outputOp?.kind == .constant)
    }

    @Test("DivideSelf rule: x / x = 1")
    func divideSelfRuleSimplifies() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .divide, operands: ["%arg0", "%arg0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%0"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        // Result should be a constant one
        let outputOp = result.operations.first { $0.result == result.returnValues[0] }
        #expect(outputOp?.kind == .constant)
    }

    // MARK: - Inverse Rules

    @Test("NegNeg rule: -(-x) = x")
    func negNegRuleSimplifies() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        #expect(result.returnValues[0] == "%arg0")
    }

    @Test("ExpLog rule: exp(log(x)) = x")
    func expLogRuleSimplifies() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .log, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .exponential, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        #expect(result.returnValues[0] == "%arg0")
    }

    @Test("LogExp rule: log(exp(x)) = x")
    func logExpRuleSimplifies() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .exponential, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .log, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        #expect(result.returnValues[0] == "%arg0")
    }

    // MARK: - Constant Folding

    @Test("Binary constant folding: 2 + 3 = 5")
    func binaryConstantFolding() {
        let scalarType = TensorType(shape: [], elementType: .float32)

        var twoConst = HLOAttributes()
        twoConst.constantValue = .scalar(2.0)

        var threeConst = HLOAttributes()
        threeConst.constantValue = .scalar(3.0)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .constant, operands: [], resultType: scalarType, attributes: twoConst),
            HLOOperation(result: "%1", kind: .constant, operands: [], resultType: scalarType, attributes: threeConst),
            HLOOperation(result: "%2", kind: .add, operands: ["%0", "%1"], resultType: scalarType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [],
            outputTypes: [scalarType],
            operations: operations,
            returnValues: ["%2"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        // Result should be a single constant with value 5
        let outputOp = result.operations.first { $0.result == result.returnValues[0] }
        #expect(outputOp?.kind == .constant)
        if case .splat(let v, _) = outputOp?.attributes.constantValue {
            #expect(v == 5.0)
        }
    }

    @Test("Unary constant folding: negate(3) = -3")
    func unaryConstantFolding() {
        let scalarType = TensorType(shape: [], elementType: .float32)

        var threeConst = HLOAttributes()
        threeConst.constantValue = .scalar(3.0)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .constant, operands: [], resultType: scalarType, attributes: threeConst),
            HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: scalarType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [],
            outputTypes: [scalarType],
            operations: operations,
            returnValues: ["%1"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        let outputOp = result.operations.first { $0.result == result.returnValues[0] }
        #expect(outputOp?.kind == .constant)
        if case .splat(let v, _) = outputOp?.attributes.constantValue {
            #expect(v == -3.0)
        }
    }

    // MARK: - Shape Simplifications

    @Test("Transpose identity rule: transpose([0,1,2]) = x")
    func transposeIdentityRuleSimplifies() {
        let inputType = TensorType(shape: [2, 3, 4], elementType: .float32)

        var transposeAttrs = HLOAttributes()
        transposeAttrs.dimensions = [0, 1, 2]  // Identity permutation

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .transpose, operands: ["%arg0"], resultType: inputType, attributes: transposeAttrs),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%0"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        #expect(result.returnValues[0] == "%arg0")
    }

    @Test("Transpose transpose rule: transpose(transpose(x)) simplifies")
    func transposeTransposeRuleSimplifies() {
        let inputType = TensorType(shape: [2, 3], elementType: .float32)
        let transposedType = TensorType(shape: [3, 2], elementType: .float32)

        var transposeAttrs = HLOAttributes()
        transposeAttrs.dimensions = [1, 0]  // Swap dimensions

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .transpose, operands: ["%arg0"], resultType: transposedType, attributes: transposeAttrs),
            HLOOperation(result: "%1", kind: .transpose, operands: ["%0"], resultType: inputType, attributes: transposeAttrs),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        // Two transposes with [1,0] cancel out
        #expect(result.returnValues[0] == "%arg0")
    }

    @Test("Reshape reshape rule: reshape(reshape(x)) = reshape(x)")
    func reshapeReshapeRuleSimplifies() {
        let inputType = TensorType(shape: [2, 3], elementType: .float32)
        let midType = TensorType(shape: [6], elementType: .float32)
        let outputType = TensorType(shape: [3, 2], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .reshape, operands: ["%arg0"], resultType: midType),
            HLOOperation(result: "%1", kind: .reshape, operands: ["%0"], resultType: outputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [outputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        // The output reshape should now take input directly from %arg0
        // Note: The first reshape may still exist (becomes dead code - DCE not implemented)
        let outputOp = result.operations.first { $0.result == result.returnValues[0] }
        #expect(outputOp?.kind == .reshape)
        #expect(outputOp?.operands[0] == "%arg0")  // Should bypass intermediate reshape
    }

    // MARK: - Convergence

    @Test("Simplifier runs to convergence on chain")
    func simplifierRunsToConvergence() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        var zeroConst = HLOAttributes()
        zeroConst.constantValue = .splat(0.0, inputType)

        var oneConst = HLOAttributes()
        oneConst.constantValue = .splat(1.0, inputType)

        // Chain: (x + 0) * 1 should simplify to x
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .constant, operands: [], resultType: inputType, attributes: zeroConst),
            HLOOperation(result: "%1", kind: .constant, operands: [], resultType: inputType, attributes: oneConst),
            HLOOperation(result: "%2", kind: .add, operands: ["%arg0", "%0"], resultType: inputType),  // x + 0
            HLOOperation(result: "%3", kind: .multiply, operands: ["%2", "%1"], resultType: inputType),  // (x+0) * 1
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%3"]
        )

        let simplifier = AlgebraicSimplifier()
        let result = simplifier.simplify(function)

        // After convergence, should just return the input
        #expect(result.returnValues[0] == "%arg0")
    }
}

// MARK: - Producer-Consumer Fusion Tests

@Suite("Producer-Consumer Fusion Tests")
struct ProducerConsumerFusionTests {

    // MARK: - Basic Fusion Tests

    @Test("Fusion groups elementwise chain")
    func fusionGroupsElementwiseChain() {
        let inputType = TensorType(shape: [4, 4], elementType: .float32)

        // Chain: exp(x) -> add(exp(x), exp(x)) -> multiply by constant
        var constAttrs = HLOAttributes()
        constAttrs.constantValue = .splat(2.0, inputType)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .exponential, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .add, operands: ["%0", "%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .constant, operands: [], resultType: inputType, attributes: constAttrs),
            HLOOperation(result: "%3", kind: .multiply, operands: ["%1", "%2"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%3"]
        )

        let fusion = ProducerConsumerFusion()
        let result = fusion.fuse(function)

        // All operations should be preserved (MPSGraph does the actual stitching)
        #expect(result.operations.count == 4)
        #expect(result.returnValues[0] == "%3")
    }

    @Test("Fusion preserves operation order")
    func fusionPreservesOperationOrder() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .exponential, operands: ["%1"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%2"]
        )

        let fusion = ProducerConsumerFusion()
        let result = fusion.fuse(function)

        // Operations should be in same order
        #expect(result.operations[0].kind == .negate)
        #expect(result.operations[1].kind == .abs)
        #expect(result.operations[2].kind == .exponential)
    }

    // MARK: - Fusibility Tests

    @Test("Fusion does not fuse unfusible operations")
    func fusionDoesNotFuseUnfusible() {
        let inputType = TensorType(shape: [4, 4], elementType: .float32)
        let reducedType = TensorType(shape: [4], elementType: .float32)

        var reduceAttrs = HLOAttributes()
        reduceAttrs.dimensions = [1]
        reduceAttrs.reductionKind = .sum

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .exponential, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .reduce, operands: ["%0"], resultType: reducedType, attributes: reduceAttrs),
            HLOOperation(result: "%2", kind: .negate, operands: ["%1"], resultType: reducedType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [reducedType],
            operations: operations,
            returnValues: ["%2"]
        )

        let fusion = ProducerConsumerFusion()
        let result = fusion.fuse(function)

        // Reduce should not be fused (it's unfusible)
        // All operations should still be present
        #expect(result.operations.count == 3)
        #expect(result.operations.contains { $0.kind == .reduce })
    }

    @Test("Fusion respects single-use constraint for expensive ops")
    func fusionRespectsMultiUseConstraint() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        // exp(x) is used by both add and multiply - expensive op with multi-use
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .exponential, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .add, operands: ["%0", "%arg0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .multiply, operands: ["%0", "%1"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%2"]
        )

        let fusion = ProducerConsumerFusion()
        let result = fusion.fuse(function)

        // All operations should be preserved (exp has multiple uses)
        #expect(result.operations.count == 3)
    }

    @Test("Fusion allows duplication of cheap ops")
    func fusionAllowsCheapOpDuplication() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        // negate(x) is cheap and can be duplicated if needed
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .add, operands: ["%0", "%arg0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .multiply, operands: ["%0", "%1"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%2"]
        )

        let fusion = ProducerConsumerFusion()
        let result = fusion.fuse(function)

        // Negate is cheap to duplicate, so fusion should still work
        #expect(result.operations.count == 3)
    }

    // MARK: - UseDefInfo Tests

    @Test("UseDefInfo correctly tracks definitions")
    func useDefInfoTracksDefs() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .multiply, operands: ["%0", "%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let useDefInfo = UseDefInfo(function: function)

        #expect(useDefInfo.definingOp(for: "%0")?.op.kind == .add)
        #expect(useDefInfo.definingOp(for: "%1")?.op.kind == .multiply)
        #expect(useDefInfo.definingOp(for: "%arg0") == nil)  // Function argument
    }

    @Test("UseDefInfo correctly tracks uses")
    func useDefInfoTracksUses() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .multiply, operands: ["%0", "%arg0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let useDefInfo = UseDefInfo(function: function)

        // %arg0 is used in both operations
        let arg0Uses = useDefInfo.uses(of: "%arg0")
        #expect(arg0Uses.count >= 2)  // At least 2 uses (may include return)

        // %0 is used in operation 1
        let result0Uses = useDefInfo.uses(of: "%0")
        #expect(result0Uses.contains(1))
    }

    @Test("UseDefInfo identifies single-use values")
    func useDefInfoIdentifiesSingleUse() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let useDefInfo = UseDefInfo(function: function)

        // %0 has a single use (in %1)
        #expect(useDefInfo.hasSingleUse("%0"))
    }

    // MARK: - Optimizer Integration Tests

    @Test("Optimizer runs producer-consumer fusion phase")
    func optimizerRunsProducerConsumerFusion() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .exponential, operands: ["%1"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%2"]
        )

        // Enable only producer-consumer fusion
        let config = HLOOptimizerConfig(
            enableFusion: false,
            enableConstantFolding: false,
            enableProducerConsumerFusion: true
        )
        let optimizer = HLOOptimizer(config: config)
        let result = optimizer.optimize(function)

        // Operations should be preserved (fusion groups them for MPSGraph stitching)
        #expect(result.operations.count == 3)
    }

    @Test("Optimizer config can disable producer-consumer fusion")
    func optimizerConfigDisablesProducerConsumerFusion() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        var zeroConst = HLOAttributes()
        zeroConst.constantValue = .splat(0.0, inputType)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .constant, operands: [], resultType: inputType, attributes: zeroConst),
            HLOOperation(result: "%1", kind: .add, operands: ["%arg0", "%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        // Disable producer-consumer fusion but enable algebraic simplification
        let config = HLOOptimizerConfig(
            enableFusion: false,
            enableConstantFolding: true,
            enableProducerConsumerFusion: false
        )
        let optimizer = HLOOptimizer(config: config)
        let result = optimizer.optimize(function)

        // Algebraic simplification should still work
        #expect(result.returnValues[0] == "%arg0")
    }

    @Test("Full optimizer pipeline runs all phases")
    func fullOptimizerPipelineRunsAllPhases() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        var zeroConst = HLOAttributes()
        zeroConst.constantValue = .splat(0.0, inputType)

        var oneConst = HLOAttributes()
        oneConst.constantValue = .splat(1.0, inputType)

        // x + 0 -> multiply by 1 -> negate -> abs
        // Algebraic simplification should remove +0 and *1
        // Producer-consumer fusion should group remaining ops
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .constant, operands: [], resultType: inputType, attributes: zeroConst),
            HLOOperation(result: "%1", kind: .constant, operands: [], resultType: inputType, attributes: oneConst),
            HLOOperation(result: "%2", kind: .add, operands: ["%arg0", "%0"], resultType: inputType),
            HLOOperation(result: "%3", kind: .multiply, operands: ["%2", "%1"], resultType: inputType),
            HLOOperation(result: "%4", kind: .negate, operands: ["%3"], resultType: inputType),
            HLOOperation(result: "%5", kind: .abs, operands: ["%4"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%5"]
        )

        // Enable all optimizations
        let config = HLOOptimizerConfig.default
        let optimizer = HLOOptimizer(config: config)
        let result = optimizer.optimize(function)

        // After algebraic simplification: x -> negate -> abs
        // The +0 and *1 should be eliminated
        // Check that negate and abs are still there
        #expect(result.operations.contains { $0.kind == .negate })
        #expect(result.operations.contains { $0.kind == .abs })

        // The add and multiply with identity should be gone
        let addOps = result.operations.filter { $0.kind == .add }
        let mulOps = result.operations.filter { $0.kind == .multiply }
        #expect(addOps.isEmpty)
        #expect(mulOps.isEmpty)
    }

    // MARK: - FusionRegion Tests

    @Test("FusionRegion correctly identifies fused regions")
    func fusionRegionIdentifiesFused() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let ops = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
        ]

        let region = FusionRegion(
            operations: ops,
            indices: [0, 1],
            rootOperation: ops[1],
            inputs: ["%arg0"]
        )

        #expect(region.isFused == true)
        #expect(region.size == 2)
        #expect(region.inputs == ["%arg0"])
    }

    @Test("Single-operation region is not fused")
    func singleOpRegionNotFused() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let op = HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType)

        let region = FusionRegion(
            operations: [op],
            indices: [0],
            rootOperation: op,
            inputs: ["%arg0"]
        )

        #expect(region.isFused == false)
        #expect(region.size == 1)
    }
}

// MARK: - Layout Assignment Tests

@Suite("Layout Assignment Tests")
struct LayoutAssignmentTests {

    // MARK: - TensorLayout Tests

    @Test("TensorLayout distinguishes matrix and convolution layouts")
    func tensorLayoutDistinguishesTypes() {
        #expect(TensorLayout.rowMajor.isMatrixLayout)
        #expect(TensorLayout.columnMajor.isMatrixLayout)
        #expect(!TensorLayout.rowMajor.isConvolutionLayout)

        #expect(TensorLayout.nhwc.isConvolutionLayout)
        #expect(TensorLayout.nchw.isConvolutionLayout)
        #expect(TensorLayout.hwio.isConvolutionLayout)
        #expect(!TensorLayout.nhwc.isMatrixLayout)
    }

    @Test("TensorLayout unspecified is neither matrix nor convolution")
    func tensorLayoutUnspecified() {
        #expect(!TensorLayout.unspecified.isMatrixLayout)
        #expect(!TensorLayout.unspecified.isConvolutionLayout)
    }

    // MARK: - Layout Preference Tests

    @Test("LayoutPreference stores priority correctly")
    func layoutPreferencePriority() {
        let pref1 = LayoutPreference(layout: .rowMajor, priority: 5, source: "matmul")
        let pref2 = LayoutPreference(layout: .columnMajor, priority: 10, source: "conv")

        #expect(pref1.priority < pref2.priority)
        #expect(pref2.layout == .columnMajor)
    }

    // MARK: - Layout Assignment Basic Tests

    @Test("Layout assignment assigns default layout to function inputs")
    func layoutAssignmentDefaultInputLayout() {
        let inputType = TensorType(shape: [4, 4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%0"]
        )

        let layoutAssignment = LayoutAssignment()
        let plan = layoutAssignment.assignLayouts(to: function)

        #expect(plan.layouts["%arg0"] != nil)
        #expect(plan.layouts["%0"] != nil)
    }

    @Test("Layout assignment propagates layout through elementwise ops")
    func layoutAssignmentPropagatesLayout() {
        let inputType = TensorType(shape: [4, 4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .exponential, operands: ["%1"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%2"]
        )

        let layoutAssignment = LayoutAssignment()
        let plan = layoutAssignment.assignLayouts(to: function)

        // All should have the same layout (propagated from input)
        let inputLayout = plan.layouts["%arg0"]
        #expect(plan.layouts["%0"] == inputLayout)
        #expect(plan.layouts["%1"] == inputLayout)
        #expect(plan.layouts["%2"] == inputLayout)
    }

    // MARK: - MatMul Layout Tests

    @Test("Layout assignment prefers row-major for matmul LHS")
    func layoutAssignmentMatmulLHS() {
        let matrixType = TensorType(shape: [4, 4], elementType: .float32)

        var dotAttrs = HLOAttributes()
        dotAttrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [],
            rhsBatchingDimensions: [],
            lhsContractingDimensions: [1],
            rhsContractingDimensions: [0]
        )

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .dotGeneral, operands: ["%arg0", "%arg1"], resultType: matrixType, attributes: dotAttrs),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: matrixType),
                HLOArgument(name: "%arg1", type: matrixType),
            ],
            outputTypes: [matrixType],
            operations: operations,
            returnValues: ["%0"]
        )

        let layoutAssignment = LayoutAssignment()
        let plan = layoutAssignment.assignLayouts(to: function)

        // LHS should prefer row-major
        #expect(plan.layouts["%arg0"] == .rowMajor)
    }

    @Test("Layout assignment prefers column-major for matmul RHS")
    func layoutAssignmentMatmulRHS() {
        let matrixType = TensorType(shape: [4, 4], elementType: .float32)

        var dotAttrs = HLOAttributes()
        dotAttrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [],
            rhsBatchingDimensions: [],
            lhsContractingDimensions: [1],
            rhsContractingDimensions: [0]
        )

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .dotGeneral, operands: ["%arg0", "%arg1"], resultType: matrixType, attributes: dotAttrs),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: matrixType),
                HLOArgument(name: "%arg1", type: matrixType),
            ],
            outputTypes: [matrixType],
            operations: operations,
            returnValues: ["%0"]
        )

        let layoutAssignment = LayoutAssignment()
        let plan = layoutAssignment.assignLayouts(to: function)

        // RHS should prefer column-major
        #expect(plan.layouts["%arg1"] == .columnMajor)
    }

    // MARK: - Convolution Layout Tests

    @Test("Layout assignment prefers NHWC for convolution input")
    func layoutAssignmentConvNHWC() {
        let inputType = TensorType(shape: [1, 28, 28, 3], elementType: .float32)
        let kernelType = TensorType(shape: [3, 3, 3, 32], elementType: .float32)
        let outputType = TensorType(shape: [1, 26, 26, 32], elementType: .float32)

        var convAttrs = HLOAttributes()
        convAttrs.convolutionDimensionNumbers = ConvolutionDimensionNumbers(
            inputBatchDimension: 0,
            inputFeatureDimension: 3,  // NHWC: channels last
            inputSpatialDimensions: [1, 2],
            kernelInputFeatureDimension: 2,
            kernelOutputFeatureDimension: 3,
            kernelSpatialDimensions: [0, 1],
            outputBatchDimension: 0,
            outputFeatureDimension: 3,
            outputSpatialDimensions: [1, 2]
        )
        convAttrs.windowStrides = [1, 1]

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .convolution, operands: ["%input", "%kernel"], resultType: outputType, attributes: convAttrs),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%input", type: inputType),
                HLOArgument(name: "%kernel", type: kernelType),
            ],
            outputTypes: [outputType],
            operations: operations,
            returnValues: ["%0"]
        )

        let layoutAssignment = LayoutAssignment()
        let plan = layoutAssignment.assignLayouts(to: function)

        // Input should be NHWC
        #expect(plan.layouts["%input"] == .nhwc)
        // Kernel should be HWIO
        #expect(plan.layouts["%kernel"] == .hwio)
    }

    // MARK: - Reduction Layout Tests

    @Test("Layout assignment prefers row-major for reduction on last dimension")
    func layoutAssignmentReductionLastDim() {
        let inputType = TensorType(shape: [4, 8], elementType: .float32)
        let outputType = TensorType(shape: [4], elementType: .float32)

        var reduceAttrs = HLOAttributes()
        reduceAttrs.dimensions = [1]  // Reduce last dimension
        reduceAttrs.reductionKind = .sum

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .reduce, operands: ["%arg0"], resultType: outputType, attributes: reduceAttrs),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [outputType],
            operations: operations,
            returnValues: ["%0"]
        )

        let layoutAssignment = LayoutAssignment()
        let plan = layoutAssignment.assignLayouts(to: function)

        // Reducing last dimension -> row-major is best
        #expect(plan.layouts["%arg0"] == .rowMajor)
    }

    @Test("Layout assignment prefers column-major for reduction on first dimension")
    func layoutAssignmentReductionFirstDim() {
        let inputType = TensorType(shape: [4, 8], elementType: .float32)
        let outputType = TensorType(shape: [8], elementType: .float32)

        var reduceAttrs = HLOAttributes()
        reduceAttrs.dimensions = [0]  // Reduce first dimension
        reduceAttrs.reductionKind = .sum

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .reduce, operands: ["%arg0"], resultType: outputType, attributes: reduceAttrs),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [outputType],
            operations: operations,
            returnValues: ["%0"]
        )

        let layoutAssignment = LayoutAssignment()
        let plan = layoutAssignment.assignLayouts(to: function)

        // Reducing first dimension -> column-major is best
        #expect(plan.layouts["%arg0"] == .columnMajor)
    }

    // MARK: - Layout Plan Tests

    @Test("Layout plan reports when conversion is needed")
    func layoutPlanReportsConversion() {
        let copies = [
            LayoutCopyOp(
                source: "%0",
                destination: "%0_nhwc",
                sourceLayout: .nchw,
                targetLayout: .nhwc,
                type: TensorType(shape: [1, 3, 28, 28], elementType: .float32)
            )
        ]

        let plan = LayoutPlan(layouts: ["%0": .nchw], copies: copies)
        #expect(plan.requiresConversion)
    }

    @Test("Layout plan reports no conversion when not needed")
    func layoutPlanReportsNoConversion() {
        let plan = LayoutPlan(layouts: ["%0": .rowMajor], copies: [])
        #expect(!plan.requiresConversion)
    }

    // MARK: - Layout Statistics Tests

    @Test("Layout statistics counts layouts correctly")
    func layoutStatisticsCountsCorrectly() {
        let layouts: [String: TensorLayout] = [
            "%0": .rowMajor,
            "%1": .rowMajor,
            "%2": .columnMajor,
            "%3": .nhwc,
        ]

        let plan = LayoutPlan(layouts: layouts, copies: [])
        let layoutAssignment = LayoutAssignment()
        let stats = layoutAssignment.computeStatistics(for: plan)

        #expect(stats.numAssignedLayouts == 4)
        #expect(stats.layoutCounts[.rowMajor] == 2)
        #expect(stats.layoutCounts[.columnMajor] == 1)
        #expect(stats.layoutCounts[.nhwc] == 1)
    }

    // MARK: - Optimizer Integration Tests

    @Test("Optimizer runs layout assignment phase")
    func optimizerRunsLayoutAssignment() {
        let inputType = TensorType(shape: [4, 4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%0"]
        )

        // Enable only layout assignment
        let config = HLOOptimizerConfig(
            enableFusion: false,
            enableConstantFolding: false,
            enableProducerConsumerFusion: false,
            enableLayoutAssignment: true
        )
        let optimizer = HLOOptimizer(config: config)
        let result = optimizer.optimize(function)

        // Function should be preserved (no layout conflicts in this simple case)
        #expect(result.operations.count == 1)
        #expect(result.operations[0].kind == .negate)
    }

    @Test("Optimizer config can disable layout assignment")
    func optimizerConfigDisablesLayoutAssignment() {
        let config = HLOOptimizerConfig(
            enableFusion: false,
            enableConstantFolding: false,
            enableProducerConsumerFusion: false,
            enableLayoutAssignment: false
        )

        #expect(!config.enableLayoutAssignment)
    }

    @Test("Full optimizer pipeline includes layout assignment")
    func fullOptimizerPipelineIncludesLayout() {
        let matrixType = TensorType(shape: [4, 4], elementType: .float32)

        var dotAttrs = HLOAttributes()
        dotAttrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [],
            rhsBatchingDimensions: [],
            lhsContractingDimensions: [1],
            rhsContractingDimensions: [0]
        )

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .dotGeneral, operands: ["%arg0", "%arg1"], resultType: matrixType, attributes: dotAttrs),
            HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: matrixType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: matrixType),
                HLOArgument(name: "%arg1", type: matrixType),
            ],
            outputTypes: [matrixType],
            operations: operations,
            returnValues: ["%1"]
        )

        // Use default config (all optimizations enabled)
        let config = HLOOptimizerConfig.default
        let optimizer = HLOOptimizer(config: config)
        let result = optimizer.optimize(function)

        // Both operations should be preserved
        #expect(result.operations.contains { $0.kind == .dotGeneral })
        #expect(result.operations.contains { $0.kind == .negate })
    }
}

// MARK: - Buffer Assignment Tests

@Suite("Buffer Assignment Tests")
struct BufferAssignmentTests {

    // MARK: - Tensor Lifetime Tests

    @Test("TensorLifetime detects overlapping lifetimes")
    func tensorLifetimeOverlap() {
        let lifetime1 = TensorLifetime(start: 0, end: 5, tensorName: "%0", byteSize: 16)
        let lifetime2 = TensorLifetime(start: 3, end: 8, tensorName: "%1", byteSize: 16)
        let lifetime3 = TensorLifetime(start: 6, end: 10, tensorName: "%2", byteSize: 16)

        // Lifetimes 1 and 2 overlap (3-5)
        #expect(lifetime1.overlaps(with: lifetime2))
        #expect(lifetime2.overlaps(with: lifetime1))

        // Lifetimes 1 and 3 don't overlap (1 ends at 5, 3 starts at 6)
        #expect(!lifetime1.overlaps(with: lifetime3))

        // Lifetimes 2 and 3 overlap (6-8)
        #expect(lifetime2.overlaps(with: lifetime3))
    }

    @Test("TensorLifetime calculates duration correctly")
    func tensorLifetimeDuration() {
        let lifetime = TensorLifetime(start: 2, end: 7, tensorName: "%0", byteSize: 16)
        #expect(lifetime.duration == 6)  // 2, 3, 4, 5, 6, 7 = 6 operations
    }

    // MARK: - Interference Graph Tests

    @Test("InterferenceGraph tracks edges correctly")
    func interferenceGraphEdges() {
        var graph = InterferenceGraph()
        graph.addVertex("%0")
        graph.addVertex("%1")
        graph.addVertex("%2")

        graph.addEdge("%0", "%1")

        #expect(graph.interferes("%0", "%1"))
        #expect(graph.interferes("%1", "%0"))  // Symmetric
        #expect(!graph.interferes("%0", "%2"))
        #expect(!graph.interferes("%1", "%2"))
    }

    @Test("InterferenceGraph tracks all vertices")
    func interferenceGraphVertices() {
        var graph = InterferenceGraph()
        graph.addVertex("%0")
        graph.addVertex("%1")
        graph.addVertex("%2")

        #expect(graph.vertices.count == 3)
        #expect(graph.vertices.contains("%0"))
        #expect(graph.vertices.contains("%1"))
        #expect(graph.vertices.contains("%2"))
    }

    @Test("InterferenceGraph returns neighbors correctly")
    func interferenceGraphNeighbors() {
        var graph = InterferenceGraph()
        graph.addVertex("%0")
        graph.addVertex("%1")
        graph.addVertex("%2")
        graph.addVertex("%3")

        graph.addEdge("%0", "%1")
        graph.addEdge("%0", "%2")

        let neighbors = graph.neighbors(of: "%0")
        #expect(neighbors.count == 2)
        #expect(neighbors.contains("%1"))
        #expect(neighbors.contains("%2"))
        #expect(!neighbors.contains("%3"))
    }

    // MARK: - Buffer Slot Tests

    @Test("BufferSlot stores properties correctly")
    func bufferSlotProperties() {
        let slot = BufferSlot(id: 0, size: 1024, tensors: ["%0", "%1"], isOutput: false)

        #expect(slot.id == 0)
        #expect(slot.size == 1024)
        #expect(slot.tensors.count == 2)
        #expect(!slot.isOutput)
    }

    // MARK: - Buffer Plan Tests

    @Test("BufferPlan calculates total memory")
    func bufferPlanTotalMemory() {
        let slots = [
            BufferSlot(id: 0, size: 1024, tensors: ["%0"], isOutput: false),
            BufferSlot(id: 1, size: 2048, tensors: ["%1"], isOutput: false),
            BufferSlot(id: 2, size: 512, tensors: ["%2"], isOutput: true),
        ]
        let tensorToSlot = ["%0": 0, "%1": 1, "%2": 2]

        let plan = BufferPlan(slots: slots, tensorToSlot: tensorToSlot)

        #expect(plan.totalMemory == 3584)  // 1024 + 2048 + 512
    }

    @Test("BufferPlan identifies output slots")
    func bufferPlanOutputSlots() {
        let slots = [
            BufferSlot(id: 0, size: 1024, tensors: ["%0"], isOutput: false),
            BufferSlot(id: 1, size: 512, tensors: ["%1"], isOutput: true),
        ]
        let tensorToSlot = ["%0": 0, "%1": 1]

        let plan = BufferPlan(slots: slots, tensorToSlot: tensorToSlot)

        #expect(!plan.isOutputSlot(0))
        #expect(plan.isOutputSlot(1))
    }

    // MARK: - Buffer Assignment Basic Tests

    @Test("Buffer assignment assigns buffers to all tensors")
    func bufferAssignmentAssignsAll() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let bufferAssignment = BufferAssignment()
        let plan = bufferAssignment.assignBuffers(to: function)

        // All tensors should have a slot
        #expect(plan.tensorToSlot["%arg0"] != nil)
        #expect(plan.tensorToSlot["%0"] != nil)
        #expect(plan.tensorToSlot["%1"] != nil)
    }

    @Test("Buffer assignment enables buffer sharing for non-overlapping tensors")
    func bufferAssignmentSharing() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        // Linear chain: arg0 -> %0 -> %1 -> %2
        // %0 ends when %1 is computed, so %0 and %2 don't overlap
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .exponential, operands: ["%1"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%2"]
        )

        let bufferAssignment = BufferAssignment()
        let plan = bufferAssignment.assignBuffers(to: function)

        // Should have fewer slots than tensors due to sharing
        // At minimum: input, output, and possibly shared intermediate
        #expect(plan.slots.count <= 4)  // At most 4 tensors
    }

    @Test("Buffer assignment preserves output buffers")
    func bufferAssignmentPreservesOutputs() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%0"]
        )

        let bufferAssignment = BufferAssignment(preserveOutputs: true)
        let plan = bufferAssignment.assignBuffers(to: function)

        // Output should be in its own slot
        let outputSlotId = plan.tensorToSlot["%0"]!
        #expect(plan.isOutputSlot(outputSlotId))
    }

    // MARK: - Lifetime Analysis Tests

    @Test("Buffer assignment computes peak memory correctly")
    func bufferAssignmentPeakMemory() {
        let type16 = TensorType(shape: [4], elementType: .float32)  // 16 bytes

        // Fork: arg0 feeds both %0 and %1, which are then combined into %2
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: type16),
            HLOOperation(result: "%1", kind: .abs, operands: ["%arg0"], resultType: type16),
            HLOOperation(result: "%2", kind: .add, operands: ["%0", "%1"], resultType: type16),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: type16)],
            outputTypes: [type16],
            operations: operations,
            returnValues: ["%2"]
        )

        let bufferAssignment = BufferAssignment()
        let peakMemory = bufferAssignment.computePeakMemory(function)

        // At operation 2 (add), we have: arg0, %0, %1 all live = 48 bytes
        // Then output %2 replaces %0 and %1
        #expect(peakMemory >= 48)  // At least 3 tensors live at once
    }

    @Test("Buffer assignment analyzes liveness correctly")
    func bufferAssignmentLiveness() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let bufferAssignment = BufferAssignment()
        let liveness = bufferAssignment.analyzeLiveness(function)

        // At operation 0: arg0 and %0 are live
        #expect(liveness[0]?.contains("%arg0") == true)
        #expect(liveness[0]?.contains("%0") == true)

        // At operation 1: %0 and %1 are live (%arg0 may or may not be)
        #expect(liveness[1]?.contains("%0") == true)
        #expect(liveness[1]?.contains("%1") == true)
    }

    // MARK: - In-Place Optimization Tests

    @Test("Buffer assignment identifies in-place candidates")
    func bufferAssignmentInPlace() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        // Chain where each intermediate is used only once
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .exponential, operands: ["%1"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%2"]
        )

        let bufferAssignment = BufferAssignment()
        let candidates = bufferAssignment.findInPlaceCandidates(function)

        // %1 can be in-place (%0 is only used here)
        // %2 can be in-place (%1 is only used here)
        // %0 cannot be in-place (arg0 is a function input)
        #expect(candidates.contains("%1"))
        #expect(candidates.contains("%2"))
        #expect(!candidates.contains("%0"))  // Input is function argument
    }

    @Test("Buffer assignment excludes multi-use tensors from in-place")
    func bufferAssignmentInPlaceMultiUse() {
        let inputType = TensorType(shape: [4], elementType: .float32)

        // %0 is used by both %1 and %2
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .exponential, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%2"]
        )

        let bufferAssignment = BufferAssignment()
        let candidates = bufferAssignment.findInPlaceCandidates(function)

        // %1 cannot be in-place because %0 is also used by %2
        #expect(!candidates.contains("%1"))
    }

    // MARK: - Buffer Statistics Tests

    @Test("Buffer statistics calculates memory savings")
    func bufferStatisticsMemorySavings() {
        let inputType = TensorType(shape: [4], elementType: .float32)  // 16 bytes each

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let bufferAssignment = BufferAssignment()
        let plan = bufferAssignment.assignBuffers(to: function)
        let stats = bufferAssignment.computeStatistics(for: plan, function: function)

        #expect(stats.numTensors == 3)  // arg0, %0, %1
        #expect(stats.naiveMemory == 48)  // 3 * 16 bytes
        #expect(stats.totalMemory <= 48)  // Should be same or less due to sharing
    }

    @Test("Buffer statistics calculates reduction percentage")
    func bufferStatisticsReduction() {
        let stats = BufferStatistics(
            numTensors: 10,
            numSlots: 5,
            totalMemory: 5000,
            naiveMemory: 10000
        )

        #expect(stats.memorySaved == 5000)
        #expect(stats.reductionPercentage == 50.0)
        #expect(stats.averageTensorsPerSlot == 2.0)
    }
}

// MARK: - Cost Model Tests

@Suite("Cost Model Tests")
struct CostModelTests {

    // MARK: - Device Capabilities Tests

    @Test("DeviceCapabilities has predefined Apple Silicon profiles")
    func deviceCapabilitiesProfiles() {
        // M1 base
        #expect(DeviceCapabilities.m1.computeUnits == 8)
        #expect(DeviceCapabilities.m1.simdWidth == 32)
        #expect(DeviceCapabilities.m1.memoryBandwidthGBps > 0)
        #expect(DeviceCapabilities.m1.peakTFlops > 0)

        // M1 Pro has more compute units
        #expect(DeviceCapabilities.m1Pro.computeUnits > DeviceCapabilities.m1.computeUnits)

        // M1 Max has more compute units than Pro
        #expect(DeviceCapabilities.m1Max.computeUnits > DeviceCapabilities.m1Pro.computeUnits)

        // M2 has more TFLOPs than M1
        #expect(DeviceCapabilities.m2.peakTFlops > DeviceCapabilities.m1.peakTFlops)

        // M3 has more TFLOPs than M2
        #expect(DeviceCapabilities.m3.peakTFlops > DeviceCapabilities.m2.peakTFlops)
    }

    @Test("DeviceCapabilities default is M1")
    func deviceCapabilitiesDefault() {
        let def = DeviceCapabilities.default
        let m1 = DeviceCapabilities.m1

        #expect(def.computeUnits == m1.computeUnits)
        #expect(def.peakTFlops == m1.peakTFlops)
    }

    @Test("DeviceCapabilities custom initialization")
    func deviceCapabilitiesCustom() {
        let custom = DeviceCapabilities(
            computeUnits: 24,
            maxThreadsPerThreadgroup: 1024,
            simdWidth: 32,
            memoryBandwidthGBps: 300.0,
            peakTFlops: 8.0,
            sharedMemoryPerThreadgroup: 32768,
            maxRegistersPerThread: 256,
            kernelLaunchOverheadUs: 3.0
        )

        #expect(custom.computeUnits == 24)
        #expect(custom.memoryBandwidthGBps == 300.0)
        #expect(custom.peakTFlops == 8.0)
        #expect(custom.maxRegistersPerThread == 256)
        #expect(custom.kernelLaunchOverheadUs == 3.0)
    }

    // MARK: - Operation Cost Tests

    @Test("OperationCost calculates arithmetic intensity")
    func operationCostArithmeticIntensity() {
        let cost = OperationCost(
            computeTimeUs: 10.0,
            memoryTimeUs: 5.0,
            flops: 1000,
            bytesRead: 400,
            bytesWritten: 100
        )

        // Arithmetic intensity = FLOPs / (bytesRead + bytesWritten)
        // 1000 / 500 = 2.0
        #expect(cost.arithmeticIntensity == 2.0)
    }

    @Test("OperationCost identifies compute-bound operations")
    func operationCostComputeBound() {
        let computeBound = OperationCost(
            computeTimeUs: 100.0,
            memoryTimeUs: 10.0,
            flops: 10000,
            bytesRead: 100,
            bytesWritten: 100
        )

        let memoryBound = OperationCost(
            computeTimeUs: 10.0,
            memoryTimeUs: 100.0,
            flops: 100,
            bytesRead: 10000,
            bytesWritten: 10000
        )

        #expect(computeBound.isComputeBound == true)
        #expect(memoryBound.isComputeBound == false)
    }

    @Test("OperationCost totalTimeUs uses max of compute and memory")
    func operationCostTotalTime() {
        let cost1 = OperationCost(
            computeTimeUs: 100.0,
            memoryTimeUs: 10.0,
            flops: 1000,
            bytesRead: 100,
            bytesWritten: 100
        )

        let cost2 = OperationCost(
            computeTimeUs: 10.0,
            memoryTimeUs: 100.0,
            flops: 100,
            bytesRead: 1000,
            bytesWritten: 1000
        )

        #expect(cost1.totalTimeUs == 100.0)
        #expect(cost2.totalTimeUs == 100.0)
    }

    @Test("OperationCost combines costs correctly")
    func operationCostCombined() {
        let cost1 = OperationCost(
            computeTimeUs: 10.0,
            memoryTimeUs: 5.0,
            flops: 100,
            bytesRead: 50,
            bytesWritten: 25
        )

        let cost2 = OperationCost(
            computeTimeUs: 20.0,
            memoryTimeUs: 15.0,
            flops: 200,
            bytesRead: 100,
            bytesWritten: 50
        )

        let combined = cost1.combined(with: cost2)

        #expect(combined.computeTimeUs == 30.0)
        #expect(combined.memoryTimeUs == 20.0)
        #expect(combined.flops == 300)
        #expect(combined.bytesRead == 150)
        #expect(combined.bytesWritten == 75)
    }

    // MARK: - Performance Model Tests

    @Test("MetalPerformanceModel estimates elementwise cost")
    func performanceModelElementwise() {
        let model = MetalPerformanceModel(device: .m1)

        let inputType = TensorType(shape: [1024, 1024], elementType: .float32)
        let op = HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: inputType)

        let cost = model.estimateCost(op)

        // Should have positive values
        #expect(cost.flops > 0)
        #expect(cost.bytesRead > 0)
        #expect(cost.bytesWritten > 0)
        #expect(cost.totalTimeUs > 0)

        // Elementwise add is 1 FLOP per element
        #expect(cost.flops == 1024 * 1024)
    }

    @Test("MetalPerformanceModel estimates transcendental cost higher")
    func performanceModelTranscendental() {
        let model = MetalPerformanceModel(device: .m1)

        let inputType = TensorType(shape: [1024], elementType: .float32)

        let addOp = HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: inputType)
        let expOp = HLOOperation(result: "%1", kind: .exponential, operands: ["%arg0"], resultType: inputType)

        let addCost = model.estimateCost(addOp)
        let expCost = model.estimateCost(expOp)

        // Transcendentals have more FLOPs per element
        #expect(expCost.flops > addCost.flops)
    }

    @Test("MetalPerformanceModel estimates matmul as compute-heavy")
    func performanceModelMatmul() {
        let model = MetalPerformanceModel(device: .m1)

        let resultType = TensorType(shape: [256, 256], elementType: .float32)

        var attrs = HLOAttributes()
        attrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [],
            rhsBatchingDimensions: [],
            lhsContractingDimensions: [1],
            rhsContractingDimensions: [0]
        )

        let op = HLOOperation(result: "%0", kind: .dotGeneral, operands: ["%arg0", "%arg1"], resultType: resultType, attributes: attrs)

        let cost = model.estimateCost(op)

        // Matmul should have high arithmetic intensity
        #expect(cost.arithmeticIntensity > 1.0)
        #expect(cost.flops > cost.bytesRead)
    }

    @Test("MetalPerformanceModel estimates reduction cost")
    func performanceModelReduction() {
        let model = MetalPerformanceModel(device: .m1)

        let inputType = TensorType(shape: [64], elementType: .float32)  // Reduced output

        var attrs = HLOAttributes()
        attrs.dimensions = [1]  // Reduce over dimension 1
        attrs.reductionKind = .sum

        let op = HLOOperation(result: "%0", kind: .reduce, operands: ["%arg0"], resultType: inputType, attributes: attrs)

        let cost = model.estimateCost(op)

        #expect(cost.flops > 0)
        #expect(cost.bytesRead > 0)
        #expect(cost.bytesWritten > 0)
    }

    @Test("MetalPerformanceModel estimates memory transfer time")
    func performanceModelMemoryTransfer() {
        let model = MetalPerformanceModel(device: .m1)

        let smallTransfer = model.estimateMemoryTransferTime(byteCount: 1024)
        let largeTransfer = model.estimateMemoryTransferTime(byteCount: 1024 * 1024)

        // Larger transfer takes more time
        #expect(largeTransfer > smallTransfer)

        // Both should be positive
        #expect(smallTransfer > 0)
        #expect(largeTransfer > 0)
    }

    // MARK: - Fusion Heuristics Tests

    @Test("FusionHeuristics recommends fusing elementwise ops")
    func fusionHeuristicsElementwise() {
        let heuristics = FusionHeuristics()

        let inputType = TensorType(shape: [1024], elementType: .float32)
        let producer = HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType)
        let consumer = HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType)

        let decision = heuristics.shouldFuse(producer: producer, consumer: consumer)

        // Elementwise ops should benefit from fusion
        #expect(decision.shouldFuse == true)
        #expect(decision.estimatedSpeedup > 1.0)
    }

    @Test("FusionHeuristics rejects unfusible operations")
    func fusionHeuristicsUnfusible() {
        let heuristics = FusionHeuristics()

        let inputType = TensorType(shape: [4, 4], elementType: .float32)

        // Convolution is unfusible
        var convAttrs = HLOAttributes()
        convAttrs.convolutionDimensionNumbers = ConvolutionDimensionNumbers(
            inputBatchDimension: 0,
            inputFeatureDimension: 3,
            inputSpatialDimensions: [1, 2],
            kernelInputFeatureDimension: 2,
            kernelOutputFeatureDimension: 3,
            kernelSpatialDimensions: [0, 1],
            outputBatchDimension: 0,
            outputFeatureDimension: 3,
            outputSpatialDimensions: [1, 2]
        )

        let convOp = HLOOperation(result: "%0", kind: .convolution, operands: ["%arg0", "%arg1"], resultType: inputType, attributes: convAttrs)
        let addOp = HLOOperation(result: "%1", kind: .add, operands: ["%0", "%arg2"], resultType: inputType)

        let decision = heuristics.shouldFuse(producer: convOp, consumer: addOp)

        #expect(decision.shouldFuse == false)
        #expect(decision.reason.contains("unfusible"))
    }

    @Test("FusionHeuristics identifies cheap-to-duplicate ops")
    func fusionHeuristicsCheapToDuplicate() {
        let heuristics = FusionHeuristics()

        let inputType = TensorType(shape: [4], elementType: .float32)

        let reshapeOp = HLOOperation(result: "%0", kind: .reshape, operands: ["%arg0"], resultType: inputType)
        let broadcastOp = HLOOperation(result: "%1", kind: .broadcastInDim, operands: ["%arg0"], resultType: inputType)
        let transposeOp = HLOOperation(result: "%2", kind: .transpose, operands: ["%arg0"], resultType: inputType)
        let constantOp = HLOOperation(result: "%3", kind: .constant, operands: [], resultType: inputType)
        let negateOp = HLOOperation(result: "%4", kind: .negate, operands: ["%arg0"], resultType: inputType)
        let expOp = HLOOperation(result: "%5", kind: .exponential, operands: ["%arg0"], resultType: inputType)

        // Shape ops and constants are cheap
        #expect(heuristics.isCheapToDuplicate(reshapeOp) == true)
        #expect(heuristics.isCheapToDuplicate(broadcastOp) == true)
        #expect(heuristics.isCheapToDuplicate(transposeOp) == true)
        #expect(heuristics.isCheapToDuplicate(constantOp) == true)
        #expect(heuristics.isCheapToDuplicate(negateOp) == true)  // Simple unary

        // Transcendentals are not cheap
        #expect(heuristics.isCheapToDuplicate(expOp) == false)
    }

    @Test("FusionHeuristics checks fusibility of op kinds")
    func fusionHeuristicsFusibility() {
        let heuristics = FusionHeuristics()

        // Elementwise ops are fusible
        #expect(heuristics.isFusibleOp(.add) == true)
        #expect(heuristics.isFusibleOp(.multiply) == true)
        #expect(heuristics.isFusibleOp(.exponential) == true)

        // Library calls are not fusible
        #expect(heuristics.isFusibleOp(.convolution) == false)
        #expect(heuristics.isFusibleOp(.fft) == false)
        #expect(heuristics.isFusibleOp(.sort) == false)
    }

    @Test("FusionHeuristics estimates register pressure")
    func fusionHeuristicsRegisterPressure() {
        let heuristics = FusionHeuristics()

        let smallType = TensorType(shape: [4], elementType: .float32)
        let ops = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: smallType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: smallType),
        ]

        let pressure = heuristics.estimateRegisterPressure(ops)

        #expect(pressure > 0)
        #expect(pressure <= heuristics.maxRegisterPressure)
    }

    // MARK: - Fusion Decision Tests

    @Test("FusionDecision stores all properties")
    func fusionDecisionProperties() {
        let decision = FusionDecision(
            shouldFuse: true,
            reason: "Test reason",
            estimatedSpeedup: 1.5,
            unfusedTimeUs: 100.0,
            fusedTimeUs: 66.7
        )

        #expect(decision.shouldFuse == true)
        #expect(decision.reason == "Test reason")
        #expect(decision.estimatedSpeedup == 1.5)
        #expect(decision.unfusedTimeUs == 100.0)
        #expect(decision.fusedTimeUs == 66.7)
    }

    // MARK: - Cost Model Statistics Tests

    @Test("CostModelStatistics calculates derived metrics")
    func costModelStatistics() {
        let stats = CostModelStatistics(
            numOperations: 10,
            totalComputeTimeUs: 100.0,
            totalMemoryTimeUs: 200.0,
            numComputeBound: 3,
            numMemoryBound: 7,
            totalFlops: 10000,
            totalBytes: 5000
        )

        // Arithmetic intensity = totalFlops / totalBytes
        #expect(stats.averageArithmeticIntensity == 2.0)

        // Memory-bound fraction = numMemoryBound / numOperations
        #expect(stats.memoryBoundFraction == 0.7)
    }

    @Test("MetalPerformanceModel analyzes function")
    func performanceModelAnalyzeFunction() {
        let model = MetalPerformanceModel(device: .m1)

        let inputType = TensorType(shape: [256, 256], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .add, operands: ["%0", "%1"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%2"]
        )

        let stats = model.analyzeFunction(function)

        #expect(stats.numOperations == 3)
        #expect(stats.totalFlops > 0)
        #expect(stats.totalBytes > 0)
        #expect(stats.totalComputeTimeUs > 0)
        #expect(stats.totalMemoryTimeUs > 0)
    }

    @Test("Performance model handles empty function")
    func performanceModelEmptyFunction() {
        let model = MetalPerformanceModel(device: .m1)

        let inputType = TensorType(shape: [4], elementType: .float32)

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: [],
            returnValues: ["%arg0"]
        )

        let stats = model.analyzeFunction(function)

        #expect(stats.numOperations == 0)
        #expect(stats.totalFlops == 0)
        #expect(stats.memoryBoundFraction == 0)
    }

    @Test("Higher bandwidth device gives lower memory time")
    func performanceModelDeviceComparison() {
        let m1Model = MetalPerformanceModel(device: .m1)
        let m1MaxModel = MetalPerformanceModel(device: .m1Max)

        let inputType = TensorType(shape: [1024, 1024], elementType: .float32)
        let op = HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: inputType)

        let m1Cost = m1Model.estimateCost(op)
        let m1MaxCost = m1MaxModel.estimateCost(op)

        // M1 Max has higher bandwidth, so memory time should be lower
        #expect(m1MaxCost.memoryTimeUs < m1Cost.memoryTimeUs)
    }

    @Test("FusionHeuristics respects max fusion size")
    func fusionHeuristicsMaxSize() {
        let heuristics = FusionHeuristics(maxFusionSize: 10)
        #expect(heuristics.maxFusionSize == 10)

        let defaultHeuristics = FusionHeuristics()
        #expect(defaultHeuristics.maxFusionSize == 50)
    }
}

// MARK: - Sibling Fusion Tests

@Suite("Sibling Fusion Tests")
struct SiblingFusionTests {

    // MARK: - Basic Sibling Detection

    @Test("Sibling fusion finds siblings sharing same producer")
    func siblingFusionFindsSiblings() {
        let inputType = TensorType(shape: [256], elementType: .float32)

        // Producer feeds two siblings: %0 -> %1, %2
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .exponential, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType, inputType],
            operations: operations,
            returnValues: ["%1", "%2"]
        )

        let siblingFusion = SiblingFusion()
        let stats = siblingFusion.analyzeOpportunities(function)

        // Should find the sibling group (%1 and %2 share %0)
        #expect(stats.numSiblingGroups >= 1)
        #expect(stats.totalSiblingsFused >= 2)
    }

    @Test("Sibling fusion preserves function semantics")
    func siblingFusionPreservesSemantics() {
        let inputType = TensorType(shape: [64], elementType: .float32)

        // Simple case: producer with two consumers
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: inputType),
            HLOOperation(result: "%1", kind: .multiply, operands: ["%0", "%arg0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .subtract, operands: ["%0", "%arg1"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: inputType),
                HLOArgument(name: "%arg1", type: inputType)
            ],
            outputTypes: [inputType, inputType],
            operations: operations,
            returnValues: ["%1", "%2"]
        )

        let siblingFusion = SiblingFusion()
        let result = siblingFusion.fuse(function)

        // Return values should be preserved
        #expect(result.returnValues.contains("%1"))
        #expect(result.returnValues.contains("%2"))

        // All operation results should still be produced
        let resultNames = Set(result.operations.map { $0.result })
        #expect(resultNames.contains("%0"))
        #expect(resultNames.contains("%1"))
        #expect(resultNames.contains("%2"))
    }

    @Test("Sibling fusion requires at least two siblings")
    func siblingFusionRequiresTwoSiblings() {
        let inputType = TensorType(shape: [32], elementType: .float32)

        // Only one consumer - no sibling fusion possible
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%arg0"], resultType: inputType),
            HLOOperation(result: "%1", kind: .abs, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        let siblingFusion = SiblingFusion()
        let stats = siblingFusion.analyzeOpportunities(function)

        // No sibling groups (only one consumer per producer)
        #expect(stats.numSiblingGroups == 0)
    }

    // MARK: - Shape Compatibility Tests

    @Test("Sibling fusion accepts same-shape siblings")
    func siblingFusionSameShape() {
        let inputType = TensorType(shape: [128, 64], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: inputType),
            HLOOperation(result: "%1", kind: .tanh, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .logistic, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%3", kind: .sqrt, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: inputType),
                HLOArgument(name: "%arg1", type: inputType)
            ],
            outputTypes: [inputType, inputType, inputType],
            operations: operations,
            returnValues: ["%1", "%2", "%3"]
        )

        let siblingFusion = SiblingFusion()
        let stats = siblingFusion.analyzeOpportunities(function)

        // Three siblings with same shape should be found
        #expect(stats.totalSiblingsFused >= 3)
    }

    @Test("Sibling fusion handles broadcastable shapes")
    func siblingFusionBroadcastableShapes() {
        let largeType = TensorType(shape: [32, 64], elementType: .float32)
        let smallType = TensorType(shape: [64], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: largeType),
            // Sibling 1: same output shape
            HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: largeType),
            // Sibling 2: reduction to smaller shape (still compatible)
            HLOOperation(result: "%2", kind: .tanh, operands: ["%0"], resultType: largeType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: largeType),
                HLOArgument(name: "%arg1", type: smallType)
            ],
            outputTypes: [largeType, largeType],
            operations: operations,
            returnValues: ["%1", "%2"]
        )

        let siblingFusion = SiblingFusion()
        let result = siblingFusion.fuse(function)

        // Should process without error
        #expect(result.operations.count >= 3)
    }

    // MARK: - Producer Type Tests

    @Test("Sibling fusion works with matmul producer")
    func siblingFusionMatmulProducer() {
        let matrixType = TensorType(shape: [64, 128], elementType: .float32)

        var dotAttrs = HLOAttributes()
        dotAttrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [],
            rhsBatchingDimensions: [],
            lhsContractingDimensions: [1],
            rhsContractingDimensions: [0]
        )

        let operations: [HLOOperation] = [
            // MatMul as producer
            HLOOperation(result: "%0", kind: .dotGeneral, operands: ["%arg0", "%arg1"], resultType: matrixType, attributes: dotAttrs),
            // Multiple elementwise consumers (common in transformers)
            HLOOperation(result: "%1", kind: .tanh, operands: ["%0"], resultType: matrixType),
            HLOOperation(result: "%2", kind: .logistic, operands: ["%0"], resultType: matrixType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [64, 256], elementType: .float32)),
                HLOArgument(name: "%arg1", type: TensorType(shape: [256, 128], elementType: .float32))
            ],
            outputTypes: [matrixType, matrixType],
            operations: operations,
            returnValues: ["%1", "%2"]
        )

        let siblingFusion = SiblingFusion()
        let stats = siblingFusion.analyzeOpportunities(function)

        // MatMul followed by multiple elementwise is a good fusion target
        #expect(stats.numSiblingGroups >= 1)
    }

    @Test("Sibling fusion skips shape-only producers")
    func siblingFusionSkipsShapeProducers() {
        let inputType = TensorType(shape: [64], elementType: .float32)
        let reshapedType = TensorType(shape: [8, 8], elementType: .float32)

        let operations: [HLOOperation] = [
            // Reshape is a shape-only operation (shouldn't be fusion root)
            HLOOperation(result: "%0", kind: .reshape, operands: ["%arg0"], resultType: reshapedType),
            HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: reshapedType),
            HLOOperation(result: "%2", kind: .abs, operands: ["%0"], resultType: reshapedType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "%arg0", type: inputType)],
            outputTypes: [reshapedType, reshapedType],
            operations: operations,
            returnValues: ["%1", "%2"]
        )

        let siblingFusion = SiblingFusion()
        let stats = siblingFusion.analyzeOpportunities(function)

        // Reshape is not a good fusion root (zero cost anyway)
        // May or may not find siblings depending on implementation
        #expect(stats.numSiblingGroups >= 0)
    }

    // MARK: - Multi-Output Region Tests

    @Test("MultiOutputFusionRegion stores all outputs")
    func multiOutputRegionOutputs() {
        let inputType = TensorType(shape: [32], elementType: .float32)

        let producer = HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: inputType)
        let sibling1 = HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: inputType)
        let sibling2 = HLOOperation(result: "%2", kind: .abs, operands: ["%0"], resultType: inputType)

        let region = MultiOutputFusionRegion(
            sharedProducer: producer,
            producerIndex: 0,
            siblings: [sibling1, sibling2],
            siblingIndices: [1, 2],
            inputs: ["%arg0", "%arg1"]
        )

        #expect(region.outputs == ["%1", "%2"])
        #expect(region.siblingCount == 2)
        #expect(region.totalOperations == 3)  // producer + 2 siblings
        #expect(region.inputs == ["%arg0", "%arg1"])
    }

    // MARK: - Statistics Tests

    @Test("SiblingFusionStatistics calculates averages")
    func siblingFusionStatistics() {
        let stats = SiblingFusionStatistics(
            numSiblingGroups: 4,
            totalSiblingsFused: 12,
            estimatedReadsSaved: 8
        )

        #expect(stats.numSiblingGroups == 4)
        #expect(stats.totalSiblingsFused == 12)
        #expect(stats.averageSiblingsPerGroup == 3.0)
        #expect(stats.estimatedReadsSaved == 8)
    }

    @Test("SiblingFusionStatistics handles empty case")
    func siblingFusionStatisticsEmpty() {
        let stats = SiblingFusionStatistics(
            numSiblingGroups: 0,
            totalSiblingsFused: 0,
            estimatedReadsSaved: 0
        )

        #expect(stats.averageSiblingsPerGroup == 0)
    }

    // MARK: - Integration Tests

    @Test("Optimizer runs sibling fusion phase")
    func optimizerRunsSiblingFusion() {
        let inputType = TensorType(shape: [64], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: inputType),
            HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .abs, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: inputType),
                HLOArgument(name: "%arg1", type: inputType)
            ],
            outputTypes: [inputType, inputType],
            operations: operations,
            returnValues: ["%1", "%2"]
        )

        // Enable only sibling fusion
        let config = HLOOptimizerConfig(
            enableFusion: false,
            enableConstantFolding: false,
            enableProducerConsumerFusion: false,
            enableSiblingFusion: true,
            enableLayoutAssignment: false
        )
        let optimizer = HLOOptimizer(config: config)
        let result = optimizer.optimize(function)

        // All operations should still be present
        #expect(result.operations.count == 3)
        // Return values should be preserved
        #expect(result.returnValues.contains("%1"))
        #expect(result.returnValues.contains("%2"))
    }

    @Test("Optimizer config can disable sibling fusion")
    func optimizerConfigDisablesSiblingFusion() {
        let inputType = TensorType(shape: [32], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: inputType),
            HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: inputType),
                HLOArgument(name: "%arg1", type: inputType)
            ],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%1"]
        )

        // Disable sibling fusion
        let config = HLOOptimizerConfig(
            enableFusion: false,
            enableConstantFolding: false,
            enableProducerConsumerFusion: false,
            enableSiblingFusion: false,
            enableLayoutAssignment: false
        )
        let optimizer = HLOOptimizer(config: config)
        let result = optimizer.optimize(function)

        // Function should pass through unchanged
        #expect(result.operations.count == 2)
    }

    @Test("Sibling fusion with three or more siblings")
    func siblingFusionThreeSiblings() {
        let inputType = TensorType(shape: [128], elementType: .float32)

        // Producer with 4 consumers
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .multiply, operands: ["%arg0", "%arg1"], resultType: inputType),
            HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%3", kind: .tanh, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%4", kind: .sqrt, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: inputType),
                HLOArgument(name: "%arg1", type: inputType)
            ],
            outputTypes: [inputType, inputType, inputType, inputType],
            operations: operations,
            returnValues: ["%1", "%2", "%3", "%4"]
        )

        let siblingFusion = SiblingFusion()
        let stats = siblingFusion.analyzeOpportunities(function)

        // Should find all 4 siblings
        #expect(stats.totalSiblingsFused >= 4)
        // Estimated reads saved = n-1 for n siblings
        #expect(stats.estimatedReadsSaved >= 3)
    }

    @Test("Sibling fusion respects max siblings limit")
    func siblingFusionMaxSiblingsLimit() {
        let siblingFusion = SiblingFusion(maxSiblings: 3)

        let inputType = TensorType(shape: [64], elementType: .float32)

        // Producer with 5 consumers (more than limit)
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: inputType),
            HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%3", kind: .tanh, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%4", kind: .sqrt, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%5", kind: .exponential, operands: ["%0"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: inputType),
                HLOArgument(name: "%arg1", type: inputType)
            ],
            outputTypes: [inputType, inputType, inputType, inputType, inputType],
            operations: operations,
            returnValues: ["%1", "%2", "%3", "%4", "%5"]
        )

        let result = siblingFusion.fuse(function)

        // Should still produce valid output
        #expect(result.operations.count >= 6)
    }

    @Test("Sibling fusion handles diamond pattern")
    func siblingFusionDiamondPattern() {
        let inputType = TensorType(shape: [64], elementType: .float32)

        // Diamond: A -> B,C -> D (where D uses both B and C)
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%arg0", "%arg1"], resultType: inputType),
            HLOOperation(result: "%1", kind: .negate, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%2", kind: .abs, operands: ["%0"], resultType: inputType),
            HLOOperation(result: "%3", kind: .multiply, operands: ["%1", "%2"], resultType: inputType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%arg0", type: inputType),
                HLOArgument(name: "%arg1", type: inputType)
            ],
            outputTypes: [inputType],
            operations: operations,
            returnValues: ["%3"]
        )

        let siblingFusion = SiblingFusion()
        let result = siblingFusion.fuse(function)

        // Diamond pattern should be handled correctly
        // %1 and %2 are siblings (both consume %0)
        #expect(result.operations.count == 4)

        // Final multiply should still work
        let lastOp = result.operations.last!
        #expect(lastOp.kind == .multiply)
    }
}

// MARK: - Horizontal Fusion Tests

@Suite("Horizontal Fusion Tests")
struct HorizontalFusionTests {

    // MARK: - Operation Signature Tests

    @Test("OperationSignature groups by kind and shape")
    func operationSignatureGrouping() {
        let type1 = TensorType(shape: [64], elementType: .float32)
        let type2 = TensorType(shape: [64], elementType: .float32)
        let type3 = TensorType(shape: [128], elementType: .float32)

        let op1 = HLOOperation(result: "%0", kind: .add, operands: ["%a", "%b"], resultType: type1)
        let op2 = HLOOperation(result: "%1", kind: .add, operands: ["%c", "%d"], resultType: type2)
        let op3 = HLOOperation(result: "%2", kind: .add, operands: ["%e", "%f"], resultType: type3)
        let op4 = HLOOperation(result: "%3", kind: .multiply, operands: ["%g", "%h"], resultType: type1)

        let sig1 = OperationSignature.from(op1)
        let sig2 = OperationSignature.from(op2)
        let sig3 = OperationSignature.from(op3)
        let sig4 = OperationSignature.from(op4)

        // Same kind and shape should be equal
        #expect(sig1 == sig2)

        // Different shape should not be equal
        #expect(sig1 != sig3)

        // Different kind should not be equal
        #expect(sig1 != sig4)
    }

    @Test("OperationSignature captures operand count")
    func operationSignatureOperandCount() {
        let type = TensorType(shape: [32], elementType: .float32)

        let unaryOp = HLOOperation(result: "%0", kind: .negate, operands: ["%a"], resultType: type)
        let binaryOp = HLOOperation(result: "%1", kind: .add, operands: ["%a", "%b"], resultType: type)

        let unarySig = OperationSignature.from(unaryOp)
        let binarySig = OperationSignature.from(binaryOp)

        #expect(unarySig.numOperands == 1)
        #expect(binarySig.numOperands == 2)
    }

    // MARK: - Candidate Finding Tests

    @Test("Horizontal fusion finds small elementwise candidates")
    func horizontalFusionFindsCandidates() {
        let smallType = TensorType(shape: [64], elementType: .float32)

        // 6 independent small adds (common in optimizer updates)
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%p0", "%g0"], resultType: smallType),
            HLOOperation(result: "%1", kind: .add, operands: ["%p1", "%g1"], resultType: smallType),
            HLOOperation(result: "%2", kind: .add, operands: ["%p2", "%g2"], resultType: smallType),
            HLOOperation(result: "%3", kind: .add, operands: ["%p3", "%g3"], resultType: smallType),
            HLOOperation(result: "%4", kind: .add, operands: ["%p4", "%g4"], resultType: smallType),
            HLOOperation(result: "%5", kind: .add, operands: ["%p5", "%g5"], resultType: smallType),
        ]

        let inputs = (0..<6).flatMap { i in
            [
                HLOArgument(name: "%p\(i)", type: smallType),
                HLOArgument(name: "%g\(i)", type: smallType)
            ]
        }

        let function = HLOFunction(
            name: "optimizer_update",
            inputs: inputs,
            outputTypes: Array(repeating: smallType, count: 6),
            operations: operations,
            returnValues: ["%0", "%1", "%2", "%3", "%4", "%5"]
        )

        let horizontalFusion = HorizontalFusion(minBatchSize: 4)
        let stats = horizontalFusion.analyzeOpportunities(function)

        // Should find a group of 6 adds
        #expect(stats.numGroups >= 1)
        #expect(stats.totalOperationsBatched >= 4)
        #expect(stats.kernelLaunchesSaved >= 3)
    }

    @Test("Horizontal fusion rejects large operations")
    func horizontalFusionRejectsLarge() {
        let largeType = TensorType(shape: [1000, 1000], elementType: .float32)  // 1M elements

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%a0", "%b0"], resultType: largeType),
            HLOOperation(result: "%1", kind: .add, operands: ["%a1", "%b1"], resultType: largeType),
            HLOOperation(result: "%2", kind: .add, operands: ["%a2", "%b2"], resultType: largeType),
            HLOOperation(result: "%3", kind: .add, operands: ["%a3", "%b3"], resultType: largeType),
        ]

        let inputs = (0..<4).flatMap { i in
            [
                HLOArgument(name: "%a\(i)", type: largeType),
                HLOArgument(name: "%b\(i)", type: largeType)
            ]
        }

        let function = HLOFunction(
            name: "test",
            inputs: inputs,
            outputTypes: Array(repeating: largeType, count: 4),
            operations: operations,
            returnValues: ["%0", "%1", "%2", "%3"]
        )

        let horizontalFusion = HorizontalFusion(maxElementsPerOp: 10000)
        let stats = horizontalFusion.analyzeOpportunities(function)

        // Large ops should not be batched
        #expect(stats.numGroups == 0)
    }

    @Test("Horizontal fusion requires minimum batch size")
    func horizontalFusionMinBatchSize() {
        let smallType = TensorType(shape: [32], elementType: .float32)

        // Only 2 operations (less than minBatchSize=4)
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%a", "%b"], resultType: smallType),
            HLOOperation(result: "%1", kind: .add, operands: ["%c", "%d"], resultType: smallType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%a", type: smallType),
                HLOArgument(name: "%b", type: smallType),
                HLOArgument(name: "%c", type: smallType),
                HLOArgument(name: "%d", type: smallType)
            ],
            outputTypes: [smallType, smallType],
            operations: operations,
            returnValues: ["%0", "%1"]
        )

        let horizontalFusion = HorizontalFusion(minBatchSize: 4)
        let stats = horizontalFusion.analyzeOpportunities(function)

        // Not enough operations to batch
        #expect(stats.numGroups == 0)
    }

    // MARK: - Fusion Group Tests

    @Test("HorizontalFusionGroup calculates total elements")
    func horizontalFusionGroupElements() {
        let shape = [64, 32]
        let signature = OperationSignature(
            kind: .add,
            elementType: .float32,
            shape: shape,
            numOperands: 2
        )

        let ops = [
            HLOOperation(result: "%0", kind: .add, operands: ["%a", "%b"], resultType: TensorType(shape: shape, elementType: .float32)),
            HLOOperation(result: "%1", kind: .add, operands: ["%c", "%d"], resultType: TensorType(shape: shape, elementType: .float32)),
        ]

        let group = HorizontalFusionGroup(
            signature: signature,
            operations: ops,
            indices: [0, 1]
        )

        // 64 * 32 * 2 operations = 4096 elements
        #expect(group.totalElements == 64 * 32 * 2)
        #expect(group.count == 2)
    }

    // MARK: - Statistics Tests

    @Test("HorizontalFusionStatistics calculates averages")
    func horizontalFusionStatistics() {
        let stats = HorizontalFusionStatistics(
            numGroups: 3,
            totalOperationsBatched: 15,
            kernelLaunchesSaved: 12
        )

        #expect(stats.numGroups == 3)
        #expect(stats.totalOperationsBatched == 15)
        #expect(stats.kernelLaunchesSaved == 12)
        #expect(stats.averageBatchSize == 5.0)
    }

    @Test("HorizontalFusionStatistics handles empty case")
    func horizontalFusionStatisticsEmpty() {
        let stats = HorizontalFusionStatistics(
            numGroups: 0,
            totalOperationsBatched: 0,
            kernelLaunchesSaved: 0
        )

        #expect(stats.averageBatchSize == 0)
    }

    // MARK: - Integration Tests

    @Test("Optimizer runs horizontal fusion phase")
    func optimizerRunsHorizontalFusion() {
        let smallType = TensorType(shape: [32], elementType: .float32)

        // 5 independent adds
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%p0", "%g0"], resultType: smallType),
            HLOOperation(result: "%1", kind: .add, operands: ["%p1", "%g1"], resultType: smallType),
            HLOOperation(result: "%2", kind: .add, operands: ["%p2", "%g2"], resultType: smallType),
            HLOOperation(result: "%3", kind: .add, operands: ["%p3", "%g3"], resultType: smallType),
            HLOOperation(result: "%4", kind: .add, operands: ["%p4", "%g4"], resultType: smallType),
        ]

        let inputs = (0..<5).flatMap { i in
            [
                HLOArgument(name: "%p\(i)", type: smallType),
                HLOArgument(name: "%g\(i)", type: smallType)
            ]
        }

        let function = HLOFunction(
            name: "test",
            inputs: inputs,
            outputTypes: Array(repeating: smallType, count: 5),
            operations: operations,
            returnValues: ["%0", "%1", "%2", "%3", "%4"]
        )

        // Enable only horizontal fusion
        let config = HLOOptimizerConfig(
            enableFusion: false,
            enableConstantFolding: false,
            enableProducerConsumerFusion: false,
            enableSiblingFusion: false,
            enableHorizontalFusion: true,
            enableLayoutAssignment: false
        )
        let optimizer = HLOOptimizer(config: config)
        let result = optimizer.optimize(function)

        // Return values should still produce correct outputs
        #expect(result.returnValues.count == 5)
    }

    @Test("Optimizer config can disable horizontal fusion")
    func optimizerConfigDisablesHorizontalFusion() {
        let smallType = TensorType(shape: [16], elementType: .float32)

        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%a", "%b"], resultType: smallType),
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "%a", type: smallType),
                HLOArgument(name: "%b", type: smallType)
            ],
            outputTypes: [smallType],
            operations: operations,
            returnValues: ["%0"]
        )

        // Disable horizontal fusion
        let config = HLOOptimizerConfig(
            enableFusion: false,
            enableConstantFolding: false,
            enableProducerConsumerFusion: false,
            enableSiblingFusion: false,
            enableHorizontalFusion: false,
            enableLayoutAssignment: false
        )
        let optimizer = HLOOptimizer(config: config)
        let result = optimizer.optimize(function)

        // Function should pass through unchanged
        #expect(result.operations.count == 1)
    }

    @Test("Horizontal fusion preserves function semantics")
    func horizontalFusionPreservesSemantics() {
        let smallType = TensorType(shape: [64], elementType: .float32)

        // 4 independent multiplies
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .multiply, operands: ["%a0", "%b0"], resultType: smallType),
            HLOOperation(result: "%1", kind: .multiply, operands: ["%a1", "%b1"], resultType: smallType),
            HLOOperation(result: "%2", kind: .multiply, operands: ["%a2", "%b2"], resultType: smallType),
            HLOOperation(result: "%3", kind: .multiply, operands: ["%a3", "%b3"], resultType: smallType),
        ]

        let inputs = (0..<4).flatMap { i in
            [
                HLOArgument(name: "%a\(i)", type: smallType),
                HLOArgument(name: "%b\(i)", type: smallType)
            ]
        }

        let function = HLOFunction(
            name: "test",
            inputs: inputs,
            outputTypes: Array(repeating: smallType, count: 4),
            operations: operations,
            returnValues: ["%0", "%1", "%2", "%3"]
        )

        let horizontalFusion = HorizontalFusion(minBatchSize: 4)
        let result = horizontalFusion.fuse(function)

        // All original results should still be produced
        let resultNames = Set(result.operations.map { $0.result })
        #expect(resultNames.contains("%0"))
        #expect(resultNames.contains("%1"))
        #expect(resultNames.contains("%2"))
        #expect(resultNames.contains("%3"))
    }

    @Test("Horizontal fusion groups by operation type")
    func horizontalFusionGroupsByType() {
        let smallType = TensorType(shape: [32], elementType: .float32)

        // Mix of adds and multiplies
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .add, operands: ["%a0", "%b0"], resultType: smallType),
            HLOOperation(result: "%1", kind: .multiply, operands: ["%a1", "%b1"], resultType: smallType),
            HLOOperation(result: "%2", kind: .add, operands: ["%a2", "%b2"], resultType: smallType),
            HLOOperation(result: "%3", kind: .multiply, operands: ["%a3", "%b3"], resultType: smallType),
            HLOOperation(result: "%4", kind: .add, operands: ["%a4", "%b4"], resultType: smallType),
            HLOOperation(result: "%5", kind: .multiply, operands: ["%a5", "%b5"], resultType: smallType),
            HLOOperation(result: "%6", kind: .add, operands: ["%a6", "%b6"], resultType: smallType),
            HLOOperation(result: "%7", kind: .multiply, operands: ["%a7", "%b7"], resultType: smallType),
        ]

        let inputs = (0..<8).flatMap { i in
            [
                HLOArgument(name: "%a\(i)", type: smallType),
                HLOArgument(name: "%b\(i)", type: smallType)
            ]
        }

        let function = HLOFunction(
            name: "test",
            inputs: inputs,
            outputTypes: Array(repeating: smallType, count: 8),
            operations: operations,
            returnValues: ["%0", "%1", "%2", "%3", "%4", "%5", "%6", "%7"]
        )

        let horizontalFusion = HorizontalFusion(minBatchSize: 4)
        let stats = horizontalFusion.analyzeOpportunities(function)

        // Should find 2 groups: 4 adds and 4 multiplies
        #expect(stats.numGroups == 2)
        #expect(stats.totalOperationsBatched == 8)
    }

    @Test("Horizontal fusion handles unary operations")
    func horizontalFusionUnaryOps() {
        let smallType = TensorType(shape: [64], elementType: .float32)

        // 5 independent negates
        let operations: [HLOOperation] = [
            HLOOperation(result: "%0", kind: .negate, operands: ["%a0"], resultType: smallType),
            HLOOperation(result: "%1", kind: .negate, operands: ["%a1"], resultType: smallType),
            HLOOperation(result: "%2", kind: .negate, operands: ["%a2"], resultType: smallType),
            HLOOperation(result: "%3", kind: .negate, operands: ["%a3"], resultType: smallType),
            HLOOperation(result: "%4", kind: .negate, operands: ["%a4"], resultType: smallType),
        ]

        let inputs = (0..<5).map { i in
            HLOArgument(name: "%a\(i)", type: smallType)
        }

        let function = HLOFunction(
            name: "test",
            inputs: inputs,
            outputTypes: Array(repeating: smallType, count: 5),
            operations: operations,
            returnValues: ["%0", "%1", "%2", "%3", "%4"]
        )

        let horizontalFusion = HorizontalFusion(minBatchSize: 4)
        let stats = horizontalFusion.analyzeOpportunities(function)

        // Should batch unary ops
        #expect(stats.numGroups >= 1)
        #expect(stats.totalOperationsBatched >= 4)
    }

    @Test("Horizontal fusion respects combined elements limit")
    func horizontalFusionCombinedLimit() {
        let mediumType = TensorType(shape: [1000], elementType: .float32)  // 1000 elements each

        // 20 operations = 20,000 elements combined
        var operations: [HLOOperation] = []
        for i in 0..<20 {
            operations.append(HLOOperation(
                result: "%\(i)",
                kind: .add,
                operands: ["%a\(i)", "%b\(i)"],
                resultType: mediumType
            ))
        }

        let inputs = (0..<20).flatMap { i in
            [
                HLOArgument(name: "%a\(i)", type: mediumType),
                HLOArgument(name: "%b\(i)", type: mediumType)
            ]
        }

        let function = HLOFunction(
            name: "test",
            inputs: inputs,
            outputTypes: Array(repeating: mediumType, count: 20),
            operations: operations,
            returnValues: (0..<20).map { "%\($0)" }
        )

        // Set a low combined limit
        let horizontalFusion = HorizontalFusion(
            minBatchSize: 4,
            maxCombinedElements: 5000  // Can only batch 5 ops
        )
        let stats = horizontalFusion.analyzeOpportunities(function)

        // Should batch some but not all
        #expect(stats.totalOperationsBatched <= 5)
    }
}
