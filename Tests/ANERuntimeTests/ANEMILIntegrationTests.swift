// ANEMILIntegrationTests.swift
// ANERuntimeTests
//
// Integration tests that verify the full HLO → CoreMLOp → CoreML pipeline.
// Uses CoreMLBridge (pure Swift, no Python) to compile and execute models.

import Testing
import Foundation
@testable import MetalHLOCore
@testable import ANERuntime

@Suite("ANE MIL Integration", .serialized)
struct ANEMILIntegrationTests {

    private let bridge = CoreMLBridge()

    // MARK: - MIL Text Format Verification

    @Test("MIL text has correct program structure")
    func milStructure() throws {
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [4], elementType: .float16)),
                HLOArgument(name: "%arg1", type: TensorType(shape: [4], elementType: .float16)),
            ],
            outputTypes: [TensorType(shape: [4], elementType: .float16)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .add,
                    operands: ["%arg0", "%arg1"],
                    resultType: TensorType(shape: [4], elementType: .float16)
                ),
            ],
            returnValues: ["%0"]
        )

        let emitter = MILEmitter()
        let program = try emitter.emit(function: function)
        let text = program.milText

        // Verify structural elements of MIL text
        #expect(text.contains("program(1.0)"), "Missing program header")
        #expect(text.contains("[buildInfo = {}]"), "Missing buildInfo")
        #expect(text.contains("func main("), "Missing func main")
        #expect(text.contains("tensor<fp16, [4]>"), "Missing tensor type annotation")
        #expect(text.contains("add(x=arg0, y=arg1)"), "Missing add operation")
        #expect(text.contains("-> (0);"), "Missing return statement")
    }

    // MARK: - CoreMLOpBuilder Conversion

    @Test("CoreMLOpBuilder produces correct ops from HLO add")
    func coreMLOpBuilderAdd() throws {
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%x", type: TensorType(shape: [4], elementType: .float16)),
                HLOArgument(name: "%y", type: TensorType(shape: [4], elementType: .float16)),
            ],
            outputTypes: [TensorType(shape: [4], elementType: .float16)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .add,
                    operands: ["%x", "%y"],
                    resultType: TensorType(shape: [4], elementType: .float16)
                ),
            ],
            returnValues: ["%0"]
        )

        let builder = CoreMLOpBuilder()
        let (inputs, ops, returnVar) = try builder.build(function: function)

        #expect(inputs.count == 2)
        #expect(inputs[0].name == "x")
        #expect(inputs[1].name == "y")
        #expect(ops.count == 1)
        #expect(ops[0].op == "add")
        #expect(ops[0].result == "v0")
        #expect(ops[0].shape == [4])
        #expect(returnVar == "v0")
    }

    // MARK: - End-to-End: HLO → CoreMLOps → CoreML Execution

    @Test("Add: HLO → CoreML end-to-end")
    func addEndToEnd() throws {
        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [("%0", .add, ["%x", "%y"], [4])],
            returnVar: "%0"
        )

        let result = try compileAndExecute(
            function: function,
            inputs: [
                ("x", [1, 2, 3, 4], [4]),
                ("y", [5, 6, 7, 8], [4]),
            ]
        )

        assertClose(result, expected: [6, 8, 10, 12], tolerance: 0.1)
    }

    @Test("Multiply: HLO → CoreML end-to-end")
    func multiplyEndToEnd() throws {
        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [("%0", .multiply, ["%x", "%y"], [4])],
            returnVar: "%0"
        )

        let result = try compileAndExecute(
            function: function,
            inputs: [
                ("x", [2, 3, 4, 5], [4]),
                ("y", [10, 20, 30, 40], [4]),
            ]
        )

        assertClose(result, expected: [20, 60, 120, 200], tolerance: 1.0)
    }

    @Test("Sigmoid: HLO → CoreML end-to-end")
    func sigmoidEndToEnd() throws {
        let function = makeFunction(
            inputs: [("%x", [4])],
            ops: [("%0", .logistic, ["%x"], [4])],
            returnVar: "%0"
        )

        let result = try compileAndExecute(
            function: function,
            inputs: [("x", [0, 1, -1, 10], [4])]
        )

        assertClose(result, expected: [0.5, 0.731, 0.269, 1.0], tolerance: 0.05)
    }

    @Test("Exp: HLO → CoreML end-to-end")
    func expEndToEnd() throws {
        let function = makeFunction(
            inputs: [("%x", [4])],
            ops: [("%0", .exponential, ["%x"], [4])],
            returnVar: "%0"
        )

        let result = try compileAndExecute(
            function: function,
            inputs: [("x", [0, 1, -1, 2], [4])]
        )

        assertClose(result, expected: [1.0, 2.718, 0.368, 7.389], tolerance: 0.05)
    }

    @Test("Chain: multiply then add (HLO → CoreML)")
    func chainMulAddEndToEnd() throws {
        // y = x0 * x1 + x2
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%x0", type: TensorType(shape: [4], elementType: .float16)),
                HLOArgument(name: "%x1", type: TensorType(shape: [4], elementType: .float16)),
                HLOArgument(name: "%x2", type: TensorType(shape: [4], elementType: .float16)),
            ],
            outputTypes: [TensorType(shape: [4], elementType: .float16)],
            operations: [
                HLOOperation(result: "%0", kind: .multiply,
                             operands: ["%x0", "%x1"],
                             resultType: TensorType(shape: [4], elementType: .float16)),
                HLOOperation(result: "%1", kind: .add,
                             operands: ["%0", "%x2"],
                             resultType: TensorType(shape: [4], elementType: .float16)),
            ],
            returnValues: ["%1"]
        )

        let result = try compileAndExecute(
            function: function,
            inputs: [
                ("x0", [2, 3, 4, 5], [4]),
                ("x1", [10, 10, 10, 10], [4]),
                ("x2", [1, 1, 1, 1], [4]),
            ]
        )

        assertClose(result, expected: [21, 31, 41, 51], tolerance: 1.0)
    }

    @Test("Subtract: HLO → CoreML end-to-end")
    func subtractEndToEnd() throws {
        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [("%0", .subtract, ["%x", "%y"], [4])],
            returnVar: "%0"
        )

        let result = try compileAndExecute(
            function: function,
            inputs: [
                ("x", [10, 20, 30, 40], [4]),
                ("y", [1, 2, 3, 4], [4]),
            ]
        )

        assertClose(result, expected: [9, 18, 27, 36], tolerance: 0.1)
    }

    @Test("Tanh: HLO → CoreML end-to-end")
    func tanhEndToEnd() throws {
        let function = makeFunction(
            inputs: [("%x", [4])],
            ops: [("%0", .tanh, ["%x"], [4])],
            returnVar: "%0"
        )

        let result = try compileAndExecute(
            function: function,
            inputs: [("x", [0, 1, -1, 2], [4])]
        )

        assertClose(result, expected: [0.0, 0.762, -0.762, 0.964], tolerance: 0.05)
    }

    // MARK: - Helpers

    /// Creates a simple HLO function for testing.
    private func makeFunction(
        inputs: [(String, [Int])],
        ops: [(String, HLOOpKind, [String], [Int])],
        returnVar: String
    ) -> HLOFunction {
        HLOFunction(
            name: "main",
            inputs: inputs.map { name, shape in
                HLOArgument(name: name, type: TensorType(shape: shape, elementType: .float16))
            },
            outputTypes: [TensorType(shape: ops.last!.3, elementType: .float16)],
            operations: ops.map { result, kind, operands, shape in
                HLOOperation(result: result, kind: kind, operands: operands,
                             resultType: TensorType(shape: shape, elementType: .float16))
            },
            returnValues: [returnVar]
        )
    }

    /// Full pipeline: HLO → CoreMLOpBuilder → CoreMLBridge → results.
    private func compileAndExecute(
        function: HLOFunction,
        inputs: [(name: String, data: [Float], shape: [Int])]
    ) throws -> [Float] {
        let builder = CoreMLOpBuilder()
        let (coremlInputs, ops, returnVar) = try builder.build(function: function)

        let program = try bridge.compile(
            inputs: coremlInputs,
            operations: ops,
            returnVar: returnVar
        )

        return try bridge.execute(program, inputs: inputs)
    }

    /// Asserts that actual values are close to expected values.
    private func assertClose(_ actual: [Float], expected: [Float], tolerance: Float) {
        #expect(actual.count == expected.count,
               "Count mismatch: got \(actual.count), expected \(expected.count)")
        for (a, b) in zip(actual, expected) {
            #expect(abs(a - b) < tolerance, "Expected \(b), got \(a)")
        }
    }
}
