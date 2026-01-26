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
}
