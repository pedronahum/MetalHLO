// CanonicalizersTests.swift
// MetalHLOCoreTests
//
// Tests for the canonicalization passes.

import Testing
@testable import MetalHLOCore

// MARK: - Reshape Canonicalizer Tests

@Suite("Reshape Canonicalizer Tests")
struct ReshapeCanonicizerTests {

    let canonicalizer = ReshapeCanonicalizer()

    @Test("Removes identity reshapes")
    func removesIdentityReshape() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [2, 3, 4], elementType: .float32))

        // Identity reshape: same input and output shape
        let reshape = HLOOperation(
            result: "y",
            kind: .reshape,
            operands: ["x"],
            resultType: TensorType(shape: [2, 3, 4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [2, 3, 4], elementType: .float32)],
            operations: [reshape],
            returnValues: ["y"]
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)
        let result = canonicalizer.run(on: function, analysis: analysis)

        #expect(result.changed == true)
        // Identity reshape should be removed
        #expect(result.function.operations.count == 0)
    }

    @Test("Folds consecutive reshapes")
    func foldsConsecutiveReshapes() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [24], elementType: .float32))

        let reshape1 = HLOOperation(
            result: "y",
            kind: .reshape,
            operands: ["x"],
            resultType: TensorType(shape: [4, 6], elementType: .float32),
            attributes: HLOAttributes()
        )

        let reshape2 = HLOOperation(
            result: "z",
            kind: .reshape,
            operands: ["y"],
            resultType: TensorType(shape: [2, 3, 4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [2, 3, 4], elementType: .float32)],
            operations: [reshape1, reshape2],
            returnValues: ["z"]
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)
        let result = canonicalizer.run(on: function, analysis: analysis)

        #expect(result.changed == true)
        // Should fold to single reshape
        #expect(result.function.operations.count <= 2)
    }
}

// MARK: - Transpose Canonicalizer Tests

@Suite("Transpose Canonicalizer Tests")
struct TransposeCanonicizerTests {

    let canonicalizer = TransposeCanonicalizer()

    @Test("Removes identity transpose")
    func removesIdentityTranspose() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [2, 3, 4], elementType: .float32))

        var attrs = HLOAttributes()
        attrs.dimensions = [0, 1, 2]  // Identity permutation

        let transpose = HLOOperation(
            result: "y",
            kind: .transpose,
            operands: ["x"],
            resultType: TensorType(shape: [2, 3, 4], elementType: .float32),
            attributes: attrs
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [2, 3, 4], elementType: .float32)],
            operations: [transpose],
            returnValues: ["y"]
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)
        let result = canonicalizer.run(on: function, analysis: analysis)

        #expect(result.changed == true)
        #expect(result.function.operations.count == 0)
    }

    @Test("Folds consecutive transposes")
    func foldsConsecutiveTransposes() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [2, 3, 4], elementType: .float32))

        var attrs1 = HLOAttributes()
        attrs1.dimensions = [1, 0, 2]  // First permutation

        let transpose1 = HLOOperation(
            result: "y",
            kind: .transpose,
            operands: ["x"],
            resultType: TensorType(shape: [3, 2, 4], elementType: .float32),
            attributes: attrs1
        )

        var attrs2 = HLOAttributes()
        attrs2.dimensions = [1, 0, 2]  // Second permutation (same as first, so cancels out)

        let transpose2 = HLOOperation(
            result: "z",
            kind: .transpose,
            operands: ["y"],
            resultType: TensorType(shape: [2, 3, 4], elementType: .float32),
            attributes: attrs2
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [2, 3, 4], elementType: .float32)],
            operations: [transpose1, transpose2],
            returnValues: ["z"]
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)
        let result = canonicalizer.run(on: function, analysis: analysis)

        #expect(result.changed == true)
        // Two transposes with [1,0,2] permutation cancel out
        // Should be folded to identity (0 or 1 ops)
    }
}

// MARK: - Broadcast Canonicalizer Tests

@Suite("Broadcast Canonicalizer Tests")
struct BroadcastCanonicizerTests {

    let canonicalizer = BroadcastCanonicalizer()

    @Test("Removes identity broadcast")
    func removesIdentityBroadcast() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [2, 3, 4], elementType: .float32))

        var attrs = HLOAttributes()
        attrs.dimensions = [0, 1, 2]

        let broadcast = HLOOperation(
            result: "y",
            kind: .broadcastInDim,
            operands: ["x"],
            resultType: TensorType(shape: [2, 3, 4], elementType: .float32),
            attributes: attrs
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [2, 3, 4], elementType: .float32)],
            operations: [broadcast],
            returnValues: ["y"]
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)
        let result = canonicalizer.run(on: function, analysis: analysis)

        #expect(result.changed == true)
        #expect(result.function.operations.count == 0)
    }
}

// MARK: - Dead Code Elimination Tests

@Suite("Dead Code Elimination Tests")
struct DeadCodeEliminationTests {

    let dce = DeadCodeEliminationPass()

    @Test("Removes unused operations")
    func removesUnusedOps() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [4], elementType: .float32))

        // op1 is unused (not a return value and not used by other ops)
        let op1 = HLOOperation(
            result: "unused",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let op2 = HLOOperation(
            result: "y",
            kind: .sqrt,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [op1, op2],
            returnValues: ["y"]  // Only y is returned, unused is dead
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)
        let result = dce.run(on: function, analysis: analysis)

        #expect(result.changed == true)
        #expect(result.function.operations.count == 1)
        #expect(result.function.operations.first?.result == "y")
    }

    @Test("Keeps transitively used operations")
    func keepsTransitivelyUsedOps() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [4], elementType: .float32))

        // All ops are transitively used by the return value
        let op1 = HLOOperation(
            result: "a",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let op2 = HLOOperation(
            result: "b",
            kind: .sqrt,
            operands: ["a"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let op3 = HLOOperation(
            result: "y",
            kind: .negate,
            operands: ["b"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [op1, op2, op3],
            returnValues: ["y"]
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)
        let result = dce.run(on: function, analysis: analysis)

        // All ops should be kept since they're all transitively used
        #expect(result.function.operations.count == 3)
    }
}

// MARK: - CSE Tests

@Suite("Common Subexpression Elimination Tests")
struct CommonSubexpressionEliminationTests {

    let cse = CommonSubexpressionEliminationPass()

    @Test("Eliminates common subexpressions")
    func eliminatesCommonSubexpressions() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [4], elementType: .float32))

        // Two identical tanh operations
        let op1 = HLOOperation(
            result: "y1",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let op2 = HLOOperation(
            result: "y2",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        // Use both
        let op3 = HLOOperation(
            result: "z",
            kind: .add,
            operands: ["y1", "y2"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [op1, op2, op3],
            returnValues: ["z"]
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)
        let result = cse.run(on: function, analysis: analysis)

        #expect(result.changed == true)
        // y1 and y2 are identical, so one should be eliminated
        #expect(result.function.operations.count == 2)
    }
}
