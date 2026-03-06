// MILFusionTests.swift
// MetalHLOCoreTests
//
// Tests for MIL graph fusion: rewriter pattern matching,
// fused CoreML op emission, and numerical equivalence.

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("MIL Fusion")
struct MILFusionTests {

    // MARK: - Test Helpers

    private func makeFunction(
        inputs: [(String, [Int])],
        ops: [(String, HLOOpKind, [String], [Int])],
        returnValues: [String]
    ) -> HLOFunction {
        HLOFunction(
            name: "main",
            inputs: inputs.map { name, shape in
                HLOArgument(name: name, type: TensorType(shape: shape, elementType: .float32))
            },
            outputTypes: [TensorType(shape: ops.last!.3, elementType: .float32)],
            operations: ops.map { result, kind, operands, shape in
                HLOOperation(
                    result: result,
                    kind: kind,
                    operands: operands,
                    resultType: TensorType(shape: shape, elementType: .float32)
                )
            },
            returnValues: returnValues
        )
    }

    private func makeOp(
        result: String,
        kind: HLOOpKind,
        operands: [String],
        shape: [Int],
        attributes: HLOAttributes = HLOAttributes()
    ) -> HLOOperation {
        HLOOperation(
            result: result,
            kind: kind,
            operands: operands,
            resultType: TensorType(shape: shape, elementType: .float32),
            attributes: attributes
        )
    }

    // MARK: - Rewriter Tests

    @Test("Rewriter passes through ops when no patterns detected")
    func rewriterNoPatterns() {
        let function = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [
                ("%0", .add, ["%x", "%y"], [64, 64]),
                ("%1", .multiply, ["%0", "%y"], [64, 64]),
            ],
            returnValues: ["%1"]
        )

        let result = MILFusionRewriter.rewrite(function: function, patterns: [])
        #expect(result.count == 2, "Should pass through all ops")

        for entry in result {
            if case .original = entry {
                // Expected
            } else {
                Issue.record("Expected .original, got .fused")
            }
        }
    }

    @Test("Rewriter fuses GELU pattern")
    func rewriterFusesGELU() {
        // GELU pattern is typically a custom_call, but for testing we can simulate
        // a single-op pattern at index 0
        let function = makeFunction(
            inputs: [("%x", [64, 64])],
            ops: [
                ("%0", .tanh, ["%x"], [64, 64]),  // Placeholder for GELU op
            ],
            returnValues: ["%0"]
        )

        let pattern = DetectedPattern(
            type: .gelu,
            operationIndices: [0],
            rootIndex: 0,
            metadata: PatternMetadata(activation: "gelu_exact")
        )

        let result = MILFusionRewriter.rewrite(function: function, patterns: [pattern])
        #expect(result.count == 1)

        if case .fused(let p, _, _) = result[0] {
            #expect(p.type == .gelu)
        } else {
            Issue.record("Expected fused GELU")
        }
    }

    @Test("Rewriter fuses multi-op RMSNorm pattern")
    func rewriterFusesRMSNorm() {
        // RMSNorm: reduce → rsqrt → mul
        let function = makeFunction(
            inputs: [("%x", [4, 64])],
            ops: [
                ("%0", .reduce, ["%x"], [4, 64]),      // reduce (mean of squares)
                ("%1", .rsqrt, ["%0"], [4, 64]),        // rsqrt
                ("%2", .multiply, ["%x", "%1"], [4, 64]), // x * rsqrt(...)
            ],
            returnValues: ["%2"]
        )

        let pattern = DetectedPattern(
            type: .rmsNorm,
            operationIndices: [0, 1, 2],
            rootIndex: 2,
            metadata: PatternMetadata(epsilon: 1e-5)
        )

        let result = MILFusionRewriter.rewrite(function: function, patterns: [pattern])
        // reduce (0) and rsqrt (1) should be consumed; only the fused root remains
        #expect(result.count == 1)

        if case .fused(let p, let root, let consumed) = result[0] {
            #expect(p.type == .rmsNorm)
            #expect(root.result == "%2")
            #expect(consumed.count == 2)
        } else {
            Issue.record("Expected fused RMSNorm")
        }
    }

    @Test("Rewriter skips unsupported pattern types")
    func rewriterSkipsUnsupported() {
        let function = makeFunction(
            inputs: [("%x", [4, 64])],
            ops: [
                ("%0", .add, ["%x", "%x"], [4, 64]),
            ],
            returnValues: ["%0"]
        )

        // attention is not in the supported set for CoreML fusion
        let pattern = DetectedPattern(
            type: .attention,
            operationIndices: [0],
            rootIndex: 0,
            metadata: PatternMetadata()
        )

        let result = MILFusionRewriter.rewrite(function: function, patterns: [pattern])
        #expect(result.count == 1)

        if case .original = result[0] {
            // Expected — attention not fused
        } else {
            Issue.record("Attention pattern should not be fused into CoreML ops")
        }
    }

    @Test("Rewriter handles overlapping patterns by first-come-first-served")
    func rewriterOverlappingPatterns() {
        let function = makeFunction(
            inputs: [("%x", [64])],
            ops: [
                ("%0", .logistic, ["%x"], [64]),        // sigmoid
                ("%1", .multiply, ["%x", "%0"], [64]),  // x * sigmoid(x) = SiLU
                ("%2", .add, ["%1", "%x"], [64]),       // something else using %1
            ],
            returnValues: ["%2"]
        )

        // SiLU pattern uses ops 0 and 1
        let siluPattern = DetectedPattern(
            type: .silu,
            operationIndices: [0, 1],
            rootIndex: 1,
            metadata: PatternMetadata(activation: "silu")
        )

        let result = MILFusionRewriter.rewrite(function: function, patterns: [siluPattern])
        // Op 0 consumed, op 1 is fused root, op 2 is original
        #expect(result.count == 2)

        if case .fused(let p, _, _) = result[0] {
            #expect(p.type == .silu)
        } else {
            Issue.record("First entry should be fused SiLU")
        }

        if case .original(let op) = result[1] {
            #expect(op.result == "%2")
        } else {
            Issue.record("Second entry should be original add")
        }
    }

    // MARK: - CoreMLOpBuilder Fusion Tests

    @Test("buildWithFusion emits gelu op for GELU pattern")
    func buildWithFusionGELU() throws {
        let function = HLOFunction(
            name: "main",
            inputs: [HLOArgument(name: "%x", type: TensorType(shape: [4, 64], elementType: .float32))],
            outputTypes: [TensorType(shape: [4, 64], elementType: .float32)],
            operations: [
                makeOp(result: "%0", kind: .tanh, operands: ["%x"], shape: [4, 64]),
            ],
            returnValues: ["%0"]
        )

        let pattern = DetectedPattern(
            type: .gelu,
            operationIndices: [0],
            rootIndex: 0,
            metadata: PatternMetadata()
        )

        let builder = CoreMLOpBuilder()
        let (_, ops, _) = try builder.buildWithFusion(function: function, patterns: [pattern])

        // Should emit a single gelu op
        #expect(ops.count == 1)
        #expect(ops[0].op == "gelu")
    }

    @Test("buildWithFusion emits SiLU ops for SiLU pattern")
    func buildWithFusionSiLU() throws {
        let function = HLOFunction(
            name: "main",
            inputs: [HLOArgument(name: "%x", type: TensorType(shape: [4, 64], elementType: .float32))],
            outputTypes: [TensorType(shape: [4, 64], elementType: .float32)],
            operations: [
                makeOp(result: "%sig", kind: .logistic, operands: ["%x"], shape: [4, 64]),
                makeOp(result: "%0", kind: .multiply, operands: ["%x", "%sig"], shape: [4, 64]),
            ],
            returnValues: ["%0"]
        )

        let pattern = DetectedPattern(
            type: .silu,
            operationIndices: [0, 1],
            rootIndex: 1,
            metadata: PatternMetadata(activation: "silu")
        )

        let builder = CoreMLOpBuilder()
        let (_, ops, _) = try builder.buildWithFusion(function: function, patterns: [pattern])

        // SiLU emits sigmoid + mul
        #expect(ops.count == 2)
        #expect(ops[0].op == "sigmoid")
        #expect(ops[1].op == "mul")
    }

    @Test("buildWithFusion emits RMSNorm ops for RMSNorm pattern")
    func buildWithFusionRMSNorm() throws {
        var reduceAttrs = HLOAttributes()
        reduceAttrs.reductionKind = .mean
        reduceAttrs.dimensions = [1]

        let function = HLOFunction(
            name: "main",
            inputs: [HLOArgument(name: "%x", type: TensorType(shape: [4, 64], elementType: .float32))],
            outputTypes: [TensorType(shape: [4, 64], elementType: .float32)],
            operations: [
                makeOp(result: "%red", kind: .reduce, operands: ["%x"], shape: [4, 64], attributes: reduceAttrs),
                makeOp(result: "%rsq", kind: .rsqrt, operands: ["%red"], shape: [4, 64]),
                makeOp(result: "%0", kind: .multiply, operands: ["%x", "%rsq"], shape: [4, 64]),
            ],
            returnValues: ["%0"]
        )

        let pattern = DetectedPattern(
            type: .rmsNorm,
            operationIndices: [0, 1, 2],
            rootIndex: 2,
            metadata: PatternMetadata(epsilon: 1e-6)
        )

        let builder = CoreMLOpBuilder()
        let (_, ops, _) = try builder.buildWithFusion(function: function, patterns: [pattern])

        // RMSNorm emits: mul(sq), reduce_mean, const(eps), add, rsqrt, mul
        #expect(ops.count == 6)
        let opNames = ops.map { $0.op }
        #expect(opNames.contains("rsqrt"))
        #expect(opNames.contains("reduce_mean"))
        #expect(opNames.last == "mul")
    }

    @Test("buildWithFusion falls back for unsupported patterns")
    func buildWithFusionFallback() throws {
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%x", type: TensorType(shape: [4, 64], elementType: .float32)),
                HLOArgument(name: "%y", type: TensorType(shape: [4, 64], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [4, 64], elementType: .float32)],
            operations: [
                makeOp(result: "%0", kind: .add, operands: ["%x", "%y"], shape: [4, 64]),
            ],
            returnValues: ["%0"]
        )

        // No patterns — should produce same result as standard build
        let builder = CoreMLOpBuilder()
        let (_, fusedOps, _) = try builder.buildWithFusion(function: function, patterns: [])
        let (_, standardOps, _) = try builder.build(function: function)

        #expect(fusedOps.count == standardOps.count)
        #expect(fusedOps[0].op == standardOps[0].op)
    }

    @Test("buildWithFusion preserves unfused ops alongside fused ones")
    func buildWithFusionMixed() throws {
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%x", type: TensorType(shape: [4, 64], elementType: .float32)),
                HLOArgument(name: "%y", type: TensorType(shape: [4, 64], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [4, 64], elementType: .float32)],
            operations: [
                makeOp(result: "%add", kind: .add, operands: ["%x", "%y"], shape: [4, 64]),
                // GELU on the add result
                makeOp(result: "%gelu", kind: .tanh, operands: ["%add"], shape: [4, 64]),
                // Another add after
                makeOp(result: "%out", kind: .add, operands: ["%gelu", "%x"], shape: [4, 64]),
            ],
            returnValues: ["%out"]
        )

        let pattern = DetectedPattern(
            type: .gelu,
            operationIndices: [1],
            rootIndex: 1,
            metadata: PatternMetadata()
        )

        let builder = CoreMLOpBuilder()
        let (_, ops, returnVar) = try builder.buildWithFusion(function: function, patterns: [pattern])

        // Should have: add, gelu, add = 3 ops
        #expect(ops.count == 3)
        #expect(ops[0].op == "add")
        #expect(ops[1].op == "gelu")
        #expect(ops[2].op == "add")
        #expect(returnVar == "out")
    }
}
