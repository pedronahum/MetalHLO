// MILEmitterTests.swift
// MetalHLOCoreTests

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("MIL Emitter")
struct MILEmitterTests {

    // MARK: - Simple Binary Ops

    @Test("Add emits correct MIL")
    func simpleAdd() throws {
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [2, 3], elementType: .float32)),
                HLOArgument(name: "%arg1", type: TensorType(shape: [2, 3], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [2, 3], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .add,
                    operands: ["%arg0", "%arg1"],
                    resultType: TensorType(shape: [2, 3], elementType: .float32)
                ),
            ],
            returnValues: ["%0"]
        )

        let emitter = MILEmitter()
        let program = try emitter.emit(function: function)

        #expect(program.milText.contains("program(1.0)"))
        #expect(program.milText.contains("func main("))
        #expect(program.milText.contains("add("))
        #expect(program.milText.contains("x=arg0"))
        #expect(program.milText.contains("y=arg1"))
        #expect(program.milText.contains("tensor<fp16"))
        #expect(program.milText.contains("-> (0);"))
        #expect(program.weights.isEmpty)
        #expect(program.inputShapes == [[2, 3], [2, 3]])
        #expect(program.outputShapes == [[2, 3]])
    }

    @Test("Subtract emits sub op")
    func subtract() throws {
        let function = makeUnaryBinaryFunction(.subtract)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("sub("))
    }

    @Test("Multiply emits mul op")
    func multiply() throws {
        let function = makeUnaryBinaryFunction(.multiply)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("mul("))
    }

    @Test("Divide emits real_div op")
    func divide() throws {
        let function = makeUnaryBinaryFunction(.divide)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("real_div("))
    }

    @Test("Maximum emits maximum op")
    func maximumOp() throws {
        let function = makeUnaryBinaryFunction(.maximum)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("maximum("))
    }

    @Test("Minimum emits minimum op")
    func minimumOp() throws {
        let function = makeUnaryBinaryFunction(.minimum)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("minimum("))
    }

    // MARK: - Unary Ops

    @Test("Exponential emits exp op")
    func exponential() throws {
        let function = makeUnaryFunction(.exponential)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("exp("))
    }

    @Test("Tanh emits tanh op")
    func tanhOp() throws {
        let function = makeUnaryFunction(.tanh)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("tanh("))
    }

    @Test("Logistic emits sigmoid op")
    func logistic() throws {
        let function = makeUnaryFunction(.logistic)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("sigmoid("))
    }

    @Test("Abs emits abs op")
    func absOp() throws {
        let function = makeUnaryFunction(.abs)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("abs("))
    }

    @Test("Negate emits mul with -1")
    func negate() throws {
        let function = makeUnaryFunction(.negate)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("mul("))
        #expect(program.milText.contains("-1.0"))
    }

    @Test("Sqrt emits sqrt op")
    func sqrtOp() throws {
        let function = makeUnaryFunction(.sqrt)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("sqrt("))
    }

    @Test("Rsqrt emits rsqrt op")
    func rsqrtOp() throws {
        let function = makeUnaryFunction(.rsqrt)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("rsqrt("))
    }

    @Test("Log emits log op")
    func logOp() throws {
        let function = makeUnaryFunction(.log)
        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("log("))
    }

    // MARK: - Constants

    @Test("Scalar constant emits inline")
    func scalarConstant() throws {
        var attrs = HLOAttributes()
        attrs.constantValue = .scalar(3.14)

        let function = HLOFunction(
            name: "main",
            inputs: [],
            outputTypes: [TensorType(shape: [], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .constant,
                    operands: [],
                    resultType: TensorType(shape: [], elementType: .float32),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("const(val="))
        #expect(program.milText.contains("fp16"))
    }

    @Test("Small dense constant emits inline array")
    func smallDenseConstant() throws {
        var attrs = HLOAttributes()
        attrs.constantValue = .dense([1.0, 2.0, 3.0, 4.0], TensorType(shape: [4], elementType: .float32))

        let function = HLOFunction(
            name: "main",
            inputs: [],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .constant,
                    operands: [],
                    resultType: TensorType(shape: [4], elementType: .float32),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("["))
        #expect(program.weights.isEmpty)
    }

    @Test("Large dense constant uses BLOBFILE")
    func largeDenseConstant() throws {
        let values = (0..<1024).map { Double($0) }
        var attrs = HLOAttributes()
        attrs.constantValue = .dense(values, TensorType(shape: [32, 32], elementType: .float32))

        let function = HLOFunction(
            name: "main",
            inputs: [],
            outputTypes: [TensorType(shape: [32, 32], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .constant,
                    operands: [],
                    resultType: TensorType(shape: [32, 32], elementType: .float32),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("BLOBFILE"))
        #expect(!program.weights.isEmpty)
        // FP16: 1024 elements * 2 bytes = 2048 bytes
        let totalBytes = program.weights.values.reduce(0) { $0 + $1.count }
        #expect(totalBytes == 2048)
    }

    // MARK: - Reshape and Transpose

    @Test("Reshape emits reshape with shape")
    func reshape() throws {
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [2, 3], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [6], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .reshape,
                    operands: ["%arg0"],
                    resultType: TensorType(shape: [6], elementType: .float32)
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("reshape("))
        #expect(program.milText.contains("shape=[6]"))
    }

    @Test("Transpose emits transpose with perm")
    func transpose() throws {
        var attrs = HLOAttributes()
        attrs.dimensions = [1, 0]

        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [2, 3], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [3, 2], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .transpose,
                    operands: ["%arg0"],
                    resultType: TensorType(shape: [3, 2], elementType: .float32),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("transpose("))
        #expect(program.milText.contains("perm=[1, 0]"))
    }

    // MARK: - Dot / Matmul

    @Test("Dot emits matmul")
    func dot() throws {
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [2, 3], elementType: .float32)),
                HLOArgument(name: "%arg1", type: TensorType(shape: [3, 4], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [2, 4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .dot,
                    operands: ["%arg0", "%arg1"],
                    resultType: TensorType(shape: [2, 4], elementType: .float32)
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("matmul("))
    }

    @Test("DotGeneral emits matmul with transpose flags")
    func dotGeneral() throws {
        var attrs = HLOAttributes()
        attrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [],
            rhsBatchingDimensions: [],
            lhsContractingDimensions: [1],
            rhsContractingDimensions: [0]
        )

        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [2, 3], elementType: .float32)),
                HLOArgument(name: "%arg1", type: TensorType(shape: [3, 4], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [2, 4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .dotGeneral,
                    operands: ["%arg0", "%arg1"],
                    resultType: TensorType(shape: [2, 4], elementType: .float32),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("matmul("))
        #expect(program.milText.contains("transpose_x"))
        #expect(program.milText.contains("transpose_y"))
    }

    // MARK: - Reduce

    @Test("Reduce sum emits reduce_sum")
    func reduceSum() throws {
        var attrs = HLOAttributes()
        attrs.reductionKind = .sum
        attrs.dimensions = [1]

        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [2, 3], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [2], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .reduce,
                    operands: ["%arg0"],
                    resultType: TensorType(shape: [2], elementType: .float32),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("reduce_sum("))
        #expect(program.milText.contains("axes=[1]"))
    }

    @Test("Reduce max emits reduce_max")
    func reduceMax() throws {
        var attrs = HLOAttributes()
        attrs.reductionKind = .max
        attrs.dimensions = [0]

        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [4, 3], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [3], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .reduce,
                    operands: ["%arg0"],
                    resultType: TensorType(shape: [3], elementType: .float32),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("reduce_max("))
    }

    // MARK: - Concatenate

    @Test("Concatenate emits concat")
    func concatenate() throws {
        var attrs = HLOAttributes()
        attrs.axis = 0

        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [2, 3], elementType: .float32)),
                HLOArgument(name: "%arg1", type: TensorType(shape: [4, 3], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [6, 3], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .concatenate,
                    operands: ["%arg0", "%arg1"],
                    resultType: TensorType(shape: [6, 3], elementType: .float32),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("concat("))
        #expect(program.milText.contains("axis=0"))
    }

    // MARK: - Slice

    @Test("Slice emits slice_by_index")
    func sliceOp() throws {
        var attrs = HLOAttributes()
        attrs.sliceStarts = [1, 0]
        attrs.sliceLimits = [3, 3]
        attrs.sliceStrides = [1, 1]

        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [4, 3], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [2, 3], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .slice,
                    operands: ["%arg0"],
                    resultType: TensorType(shape: [2, 3], elementType: .float32),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("slice_by_index("))
        #expect(program.milText.contains("begin=[1, 0]"))
        #expect(program.milText.contains("end=[3, 3]"))
    }

    // MARK: - Compare and Select

    @Test("Compare EQ emits equal")
    func compareEQ() throws {
        var attrs = HLOAttributes()
        attrs.comparisonDirection = .eq

        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [4], elementType: .float32)),
                HLOArgument(name: "%arg1", type: TensorType(shape: [4], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [4], elementType: .int1)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .compare,
                    operands: ["%arg0", "%arg1"],
                    resultType: TensorType(shape: [4], elementType: .int1),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("equal("))
    }

    @Test("Select emits select")
    func selectOp() throws {
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [4], elementType: .int1)),
                HLOArgument(name: "%arg1", type: TensorType(shape: [4], elementType: .float32)),
                HLOArgument(name: "%arg2", type: TensorType(shape: [4], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .select,
                    operands: ["%arg0", "%arg1", "%arg2"],
                    resultType: TensorType(shape: [4], elementType: .float32)
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("select("))
        #expect(program.milText.contains("cond="))
    }

    // MARK: - Broadcast

    @Test("BroadcastInDim emits reshape + tile")
    func broadcastInDim() throws {
        var attrs = HLOAttributes()
        attrs.dimensions = [1]  // Map input dim 0 to output dim 1

        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [3], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [2, 3], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .broadcastInDim,
                    operands: ["%arg0"],
                    resultType: TensorType(shape: [2, 3], elementType: .float32),
                    attributes: attrs
                ),
            ],
            returnValues: ["%0"]
        )

        let program = try MILEmitter().emit(function: function)
        // Should contain reshape (to insert size-1 dim) and tile (to broadcast)
        #expect(program.milText.contains("reshape("))
        #expect(program.milText.contains("tile("))
    }

    // MARK: - Composite Tests

    @Test("Linear layer: dot + add")
    func linearLayer() throws {
        var dotAttrs = HLOAttributes()
        dotAttrs.dotDimensionNumbers = DotDimensionNumbers(
            lhsBatchingDimensions: [],
            rhsBatchingDimensions: [],
            lhsContractingDimensions: [1],
            rhsContractingDimensions: [0]
        )

        var biasAttrs = HLOAttributes()
        biasAttrs.constantValue = .splat(0.0, TensorType(shape: [4], elementType: .float32))

        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [2, 3], elementType: .float32)),
                HLOArgument(name: "%arg1", type: TensorType(shape: [3, 4], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [2, 4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .dotGeneral,
                    operands: ["%arg0", "%arg1"],
                    resultType: TensorType(shape: [2, 4], elementType: .float32),
                    attributes: dotAttrs
                ),
                HLOOperation(
                    result: "%1",
                    kind: .constant,
                    operands: [],
                    resultType: TensorType(shape: [4], elementType: .float32),
                    attributes: biasAttrs
                ),
                HLOOperation(
                    result: "%2",
                    kind: .add,
                    operands: ["%0", "%1"],
                    resultType: TensorType(shape: [2, 4], elementType: .float32)
                ),
            ],
            returnValues: ["%2"]
        )

        let program = try MILEmitter().emit(function: function)
        #expect(program.milText.contains("matmul("))
        #expect(program.milText.contains("const("))
        #expect(program.milText.contains("add("))
    }

    // MARK: - Validation

    @Test("Unsupported op throws error")
    func unsupportedOp() {
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [4], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .fft,
                    operands: ["%arg0"],
                    resultType: TensorType(shape: [4], elementType: .float32)
                ),
            ],
            returnValues: ["%0"]
        )

        #expect(throws: MILEmitterError.self) {
            _ = try MILEmitter().emit(function: function)
        }
    }

    @Test("validateSupport reports unsupported ops")
    func validateSupportReports() {
        let function = HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [4], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .add,
                    operands: ["%arg0", "%arg0"],
                    resultType: TensorType(shape: [4], elementType: .float32)
                ),
                HLOOperation(
                    result: "%1",
                    kind: .fft,
                    operands: ["%0"],
                    resultType: TensorType(shape: [4], elementType: .float32)
                ),
            ],
            returnValues: ["%1"]
        )

        let emitter = MILEmitter()
        let unsupported = emitter.validateSupport(function: function)
        #expect(unsupported.count == 1)
        #expect(unsupported[0].0 == .fft)
    }

    @Test("validateSupport returns empty for supported function")
    func validateSupportAllGood() {
        let function = makeUnaryFunction(.exponential)
        let unsupported = MILEmitter().validateSupport(function: function)
        #expect(unsupported.isEmpty)
    }

    // MARK: - MIL Structure Tests

    @Test("MIL text has proper structure")
    func milStructure() throws {
        let function = makeUnaryFunction(.exponential)
        let program = try MILEmitter().emit(function: function)

        let text = program.milText
        // Must start with program header
        #expect(text.hasPrefix("program(1.0)"))
        // Must contain func main
        #expect(text.contains("func main("))
        // Must end with }
        #expect(text.hasSuffix("}"))
        // Must have return (via -> syntax)
        #expect(text.contains("-> ("))
    }

    // MARK: - Helpers

    private func makeUnaryFunction(_ kind: HLOOpKind) -> HLOFunction {
        HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [4], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: kind,
                    operands: ["%arg0"],
                    resultType: TensorType(shape: [4], elementType: .float32)
                ),
            ],
            returnValues: ["%0"]
        )
    }

    private func makeUnaryBinaryFunction(_ kind: HLOOpKind) -> HLOFunction {
        HLOFunction(
            name: "main",
            inputs: [
                HLOArgument(name: "%arg0", type: TensorType(shape: [4], elementType: .float32)),
                HLOArgument(name: "%arg1", type: TensorType(shape: [4], elementType: .float32)),
            ],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: kind,
                    operands: ["%arg0", "%arg1"],
                    resultType: TensorType(shape: [4], elementType: .float32)
                ),
            ],
            returnValues: ["%0"]
        )
    }
}
