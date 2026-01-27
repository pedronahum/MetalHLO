// AnalyzerTests.swift
// MetalHLOCoreTests
//
// Tests for the Analyzer class.

import Testing
@testable import MetalHLOCore

// MARK: - Analyzer Tests

@Suite("Analyzer Tests")
struct AnalyzerTests {

    let analyzer = Analyzer()

    // MARK: - Shape Inference Tests

    @Test("Shape inference for unary operations")
    func shapeInferenceUnary() {
        // Create a simple function with a unary operation
        let input = HLOArgument(name: "x", type: TensorType(shape: [2, 3, 4], elementType: .float32))
        let op = HLOOperation(
            result: "y",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [2, 3, 4], elementType: .float32),
            attributes: HLOAttributes()
        )
        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [2, 3, 4], elementType: .float32)],
            operations: [op],
            returnValues: ["y"]
        )

        let shapes = analyzer.inferShapes(function)

        #expect(shapes["x"] == [2, 3, 4])
        #expect(shapes["y"] == [2, 3, 4])
    }

    @Test("Shape inference for binary operations")
    func shapeInferenceBinary() {
        // Create function with binary operation
        let inputA = HLOArgument(name: "a", type: TensorType(shape: [4, 5], elementType: .float32))
        let inputB = HLOArgument(name: "b", type: TensorType(shape: [4, 5], elementType: .float32))
        let op = HLOOperation(
            result: "c",
            kind: .add,
            operands: ["a", "b"],
            resultType: TensorType(shape: [4, 5], elementType: .float32),
            attributes: HLOAttributes()
        )
        let function = HLOFunction(
            name: "test",
            inputs: [inputA, inputB],
            outputTypes: [TensorType(shape: [4, 5], elementType: .float32)],
            operations: [op],
            returnValues: ["c"]
        )

        let shapes = analyzer.inferShapes(function)

        #expect(shapes["a"] == [4, 5])
        #expect(shapes["b"] == [4, 5])
        #expect(shapes["c"] == [4, 5])
    }

    @Test("Shape inference for reduction operations")
    func shapeInferenceReduction() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [2, 3, 4], elementType: .float32))
        var attrs = HLOAttributes()
        attrs.dimensions = [1]  // Reduce along dimension 1

        let op = HLOOperation(
            result: "y",
            kind: .reduce,
            operands: ["x"],
            resultType: TensorType(shape: [2, 4], elementType: .float32),
            attributes: attrs
        )
        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [2, 4], elementType: .float32)],
            operations: [op],
            returnValues: ["y"]
        )

        let shapes = analyzer.inferShapes(function)

        #expect(shapes["x"] == [2, 3, 4])
        #expect(shapes["y"] == [2, 4])
    }

    @Test("Shape inference for transpose operations")
    func shapeInferenceTranspose() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [2, 3, 4], elementType: .float32))
        var attrs = HLOAttributes()
        attrs.dimensions = [2, 0, 1]  // Permutation

        let op = HLOOperation(
            result: "y",
            kind: .transpose,
            operands: ["x"],
            resultType: TensorType(shape: [4, 2, 3], elementType: .float32),
            attributes: attrs
        )
        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [4, 2, 3], elementType: .float32)],
            operations: [op],
            returnValues: ["y"]
        )

        let shapes = analyzer.inferShapes(function)

        #expect(shapes["y"] == [4, 2, 3])
    }

    // MARK: - Element Type Inference Tests

    @Test("Element type inference")
    func elementTypeInference() {
        let inputA = HLOArgument(name: "a", type: TensorType(shape: [4], elementType: .float16))
        let inputB = HLOArgument(name: "b", type: TensorType(shape: [4], elementType: .float32))

        let op1 = HLOOperation(
            result: "c",
            kind: .tanh,
            operands: ["a"],
            resultType: TensorType(shape: [4], elementType: .float16),
            attributes: HLOAttributes()
        )
        let op2 = HLOOperation(
            result: "d",
            kind: .sqrt,
            operands: ["b"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [inputA, inputB],
            outputTypes: [TensorType(shape: [4], elementType: .float16), TensorType(shape: [4], elementType: .float32)],
            operations: [op1, op2],
            returnValues: ["c", "d"]
        )

        let types = analyzer.inferElementTypes(function)

        #expect(types["a"] == .float16)
        #expect(types["b"] == .float32)
        #expect(types["c"] == .float16)
        #expect(types["d"] == .float32)
    }

    // MARK: - Dependency Analysis Tests

    @Test("Dependency analysis basic")
    func dependencyAnalysisBasic() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [4], elementType: .float32))

        // x -> y -> z chain
        let op1 = HLOOperation(
            result: "y",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )
        let op2 = HLOOperation(
            result: "z",
            kind: .sqrt,
            operands: ["y"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [op1, op2],
            returnValues: ["z"]
        )

        let (dependencies, users) = analyzer.analyzeDependencies(function)

        // y depends on x (but x is an input, not an op result)
        #expect(dependencies["y"]?.isEmpty == true)
        #expect(dependencies["z"]?.contains("y") == true)

        // y is used by z
        #expect(users["y"]?.contains("z") == true)
    }

    @Test("Dependency analysis diamond pattern")
    func dependencyAnalysisDiamond() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [4], elementType: .float32))

        // Diamond: x -> y, x -> z, y+z -> w
        let op1 = HLOOperation(
            result: "y",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )
        let op2 = HLOOperation(
            result: "z",
            kind: .sqrt,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )
        let op3 = HLOOperation(
            result: "w",
            kind: .add,
            operands: ["y", "z"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [op1, op2, op3],
            returnValues: ["w"]
        )

        let (dependencies, users) = analyzer.analyzeDependencies(function)

        // w depends on both y and z
        #expect(dependencies["w"]?.contains("y") == true)
        #expect(dependencies["w"]?.contains("z") == true)

        // y and z are both used by w
        #expect(users["y"]?.contains("w") == true)
        #expect(users["z"]?.contains("w") == true)
    }

    // MARK: - Lifetime Analysis Tests

    @Test("Lifetime analysis basic")
    func lifetimeAnalysisBasic() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [4], elementType: .float32))

        let op1 = HLOOperation(
            result: "y",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )
        let op2 = HLOOperation(
            result: "z",
            kind: .sqrt,
            operands: ["y"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [op1, op2],
            returnValues: ["z"]
        )

        let (dependencies, users) = analyzer.analyzeDependencies(function)
        let lifetimes = analyzer.analyzeLifetimes(function, dependencies: dependencies, users: users)

        // Input x is used in op at index 0
        #expect(lifetimes["x"]?.defined == -1)  // Inputs defined at -1
        #expect(lifetimes["x"]?.lastUsed == 0)

        // y is defined at index 0, used in op at index 1
        #expect(lifetimes["y"]?.defined == 0)
        #expect(lifetimes["y"]?.lastUsed == 1)

        // z is defined at index 1, it's a return value
        #expect(lifetimes["z"]?.defined == 1)
        #expect(lifetimes["z"]?.lastUsed == 2)  // Return values live until end
    }

    // MARK: - Full Analysis Tests

    @Test("Full analysis returns all components")
    func fullAnalysis() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [2, 3], elementType: .float32))

        let op = HLOOperation(
            result: "y",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [2, 3], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [2, 3], elementType: .float32)],
            operations: [op],
            returnValues: ["y"]
        )

        let results = analyzer.analyze(function)

        // Verify all analysis components are populated
        #expect(!results.shapes.isEmpty)
        #expect(!results.elementTypes.isEmpty)
        #expect(!results.lifetimes.isEmpty)
        // Patterns may be empty for simple functions
    }

    // MARK: - Pattern Detection Tests

    @Test("Pattern detection finds attention pattern")
    func patternDetectionAttention() {
        // Create a simplified attention-like pattern
        let input = HLOArgument(name: "q", type: TensorType(shape: [1, 8, 64, 64], elementType: .float32))

        // Q @ K^T -> softmax -> @ V (simplified)
        let qkMatmul = HLOOperation(
            result: "qk",
            kind: .dot,
            operands: ["q", "q"],  // Simplified: Q @ Q^T
            resultType: TensorType(shape: [1, 8, 64, 64], elementType: .float32),
            attributes: HLOAttributes()
        )

        var softmaxAttrs = HLOAttributes()
        softmaxAttrs.callTargetName = "softmax"
        let softmax = HLOOperation(
            result: "scores",
            kind: .customCall,
            operands: ["qk"],
            resultType: TensorType(shape: [1, 8, 64, 64], elementType: .float32),
            attributes: softmaxAttrs
        )

        let attnOutput = HLOOperation(
            result: "attn",
            kind: .dot,
            operands: ["scores", "q"],
            resultType: TensorType(shape: [1, 8, 64, 64], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "attention",
            inputs: [input],
            outputTypes: [TensorType(shape: [1, 8, 64, 64], elementType: .float32)],
            operations: [qkMatmul, softmax, attnOutput],
            returnValues: ["attn"]
        )

        let shapes = analyzer.inferShapes(function)
        let patterns = analyzer.detectPatterns(function, shapes: shapes)

        // Should detect attention pattern
        let attentionPatterns = patterns.filter { $0.type == .attention }
        #expect(attentionPatterns.count == 1)
    }

    @Test("Pattern detection finds RMSNorm pattern")
    func patternDetectionRMSNorm() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [1, 512, 768], elementType: .float32))

        var rmsNormAttrs = HLOAttributes()
        rmsNormAttrs.callTargetName = "rms_norm"
        rmsNormAttrs.epsilon = 1e-5

        let rmsNorm = HLOOperation(
            result: "y",
            kind: .customCall,
            operands: ["x"],
            resultType: TensorType(shape: [1, 512, 768], elementType: .float32),
            attributes: rmsNormAttrs
        )

        let function = HLOFunction(
            name: "rms_norm",
            inputs: [input],
            outputTypes: [TensorType(shape: [1, 512, 768], elementType: .float32)],
            operations: [rmsNorm],
            returnValues: ["y"]
        )

        let shapes = analyzer.inferShapes(function)
        let patterns = analyzer.detectPatterns(function, shapes: shapes)

        let rmsNormPatterns = patterns.filter { $0.type == .rmsNorm }
        #expect(rmsNormPatterns.count == 1)
        #expect(rmsNormPatterns.first?.metadata.epsilon == 1e-5)
    }

    @Test("Pattern detection finds GELU pattern")
    func patternDetectionGELU() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [1, 512, 768], elementType: .float32))

        var geluAttrs = HLOAttributes()
        geluAttrs.callTargetName = "gelu_approximate"

        let gelu = HLOOperation(
            result: "y",
            kind: .customCall,
            operands: ["x"],
            resultType: TensorType(shape: [1, 512, 768], elementType: .float32),
            attributes: geluAttrs
        )

        let function = HLOFunction(
            name: "gelu",
            inputs: [input],
            outputTypes: [TensorType(shape: [1, 512, 768], elementType: .float32)],
            operations: [gelu],
            returnValues: ["y"]
        )

        let shapes = analyzer.inferShapes(function)
        let patterns = analyzer.detectPatterns(function, shapes: shapes)

        let geluPatterns = patterns.filter { $0.type == .gelu }
        #expect(geluPatterns.count == 1)
    }

    @Test("Pattern detection finds SiLU pattern")
    func patternDetectionSiLU() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [1, 512, 768], elementType: .float32))

        // SiLU = x * sigmoid(x)
        let sigmoid = HLOOperation(
            result: "sig",
            kind: .logistic,
            operands: ["x"],
            resultType: TensorType(shape: [1, 512, 768], elementType: .float32),
            attributes: HLOAttributes()
        )

        let silu = HLOOperation(
            result: "y",
            kind: .multiply,
            operands: ["x", "sig"],
            resultType: TensorType(shape: [1, 512, 768], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "silu",
            inputs: [input],
            outputTypes: [TensorType(shape: [1, 512, 768], elementType: .float32)],
            operations: [sigmoid, silu],
            returnValues: ["y"]
        )

        let shapes = analyzer.inferShapes(function)
        let patterns = analyzer.detectPatterns(function, shapes: shapes)

        let siluPatterns = patterns.filter { $0.type == .silu }
        #expect(siluPatterns.count == 1)
    }
}
