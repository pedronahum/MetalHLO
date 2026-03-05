// CoreMLModelBuilder.swift
// ANERuntime
//
// Translates CoreMLOp arrays into CoreML protobuf model specs.
// Produces binary protobuf data for .mlmodel files.

import Foundation

/// Builds CoreML MIL model protobuf from structured operation descriptions.
internal struct CoreMLModelBuilder {

    /// Builds a complete CoreML model protobuf.
    ///
    /// - Parameters:
    ///   - inputs: Function input names and shapes.
    ///   - operations: The MIL operations in topological order.
    ///   - returnVar: Name of the return variable.
    ///   - weightsFileName: Name of the weights file for blob references (default: "@model_path/weights/weight.bin").
    /// - Returns: Binary protobuf data for the .mlmodel file.
    static func build(
        inputs: [(name: String, shape: [Int])],
        operations: [CoreMLOp],
        returnVar: String,
        weightsFileName: String = "@model_path/weights/weight.bin"
    ) -> Data {
        // Determine output shape from the return variable
        let outputShape = findOutputShape(returnVar: returnVar, operations: operations)

        return CoreMLProtoBuilder.buildModel(
            specVersion: 8,
            inputs: inputs,
            outputs: [(name: returnVar, shape: outputShape)],
            program: { pw in
                CoreMLProtoBuilder.writeMILProgram(pw, functions: [
                    (name: "main", builder: { fw in
                        CoreMLProtoBuilder.writeMILFunction(
                            fw,
                            inputs: inputs.map { ($0.name, $0.shape, MILDataType.float32) },
                            block: { bw in
                                let opWriters = operations.map { op in
                                    return { (w: ProtobufWriter) in
                                        writeOperation(w, op: op, weightsFileName: weightsFileName)
                                    }
                                }
                                CoreMLProtoBuilder.writeMILBlock(
                                    bw,
                                    outputs: [returnVar],
                                    operations: opWriters
                                )
                            }
                        )
                    })
                ])
            }
        )
    }

    // MARK: - Operation translation

    private static func writeOperation(
        _ w: ProtobufWriter,
        op: CoreMLOp,
        weightsFileName: String
    ) {
        let milInputs: [(name: String, argument: (ProtobufWriter) -> Void)] =
            op.params.map { (name, value) in
                (name: name, argument: { aw in
                    writeParamAsArgument(aw, value: value, weightsFileName: weightsFileName)
                })
            }

        CoreMLProtoBuilder.writeMILOperation(
            w,
            type: op.op,
            inputs: milInputs,
            outputs: [(name: op.result, shape: op.shape, dataType: .float32)]
        )
    }

    private static func writeParamAsArgument(
        _ w: ProtobufWriter,
        value: CoreMLParamValue,
        weightsFileName: String
    ) {
        switch value {
        case .variable(let name):
            CoreMLProtoBuilder.writeVariableArgument(w, name: name)

        case .intValue(let v):
            CoreMLProtoBuilder.writeImmediateArgument(w, name: String(v)) { vw in
                CoreMLProtoBuilder.writeIntValue(vw, shape: [], values: [Int32(v)])
            }

        case .intArray(let arr):
            CoreMLProtoBuilder.writeImmediateArgument(w, name: "value") { vw in
                CoreMLProtoBuilder.writeIntValue(vw, shape: [arr.count], values: arr.map { Int32($0) })
            }

        case .floatValue(let v):
            CoreMLProtoBuilder.writeImmediateArgument(w, name: "value") { vw in
                CoreMLProtoBuilder.writeFloatValue(vw, shape: [], values: [Float(v)])
            }

        case .floatArray(let arr):
            CoreMLProtoBuilder.writeImmediateArgument(w, name: "value") { vw in
                CoreMLProtoBuilder.writeFloatValue(vw, shape: [arr.count], values: arr.map { Float($0) })
            }

        case .boolValue(let v):
            CoreMLProtoBuilder.writeImmediateArgument(w, name: "value") { vw in
                CoreMLProtoBuilder.writeBoolValue(vw, shape: [], values: [v])
            }

        case .stringValue(let s):
            CoreMLProtoBuilder.writeImmediateArgument(w, name: "value") { vw in
                CoreMLProtoBuilder.writeStringValue(vw, value: s)
            }

        case .variableTuple(let names):
            // For concat etc: write as multiple bindings in the argument
            CoreMLProtoBuilder.writeArgument(w, bindings: names.map { (name: $0, value: nil) })
        }
    }

    // MARK: - Helpers

    private static func findOutputShape(returnVar: String, operations: [CoreMLOp]) -> [Int] {
        if let op = operations.last(where: { $0.result == returnVar }) {
            return op.shape
        }
        return []
    }
}
