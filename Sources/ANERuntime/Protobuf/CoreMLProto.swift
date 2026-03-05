// CoreMLProto.swift
// ANERuntime
//
// CoreML-specific protobuf message builders for MIL model specs.
// Field numbers extracted from Apple's coremltools proto definitions:
// - Model.proto: Model, ModelDescription, FeatureDescription
// - FeatureTypes.proto: FeatureType, ArrayFeatureType
// - MIL.proto: Program, Function, Block, Operation, etc.
//
// Reference: https://github.com/apple/coremltools/tree/main/mlmodel/format

import Foundation

// MARK: - Data type enums

/// CoreML MIL DataType enum values (from MIL.proto).
internal enum MILDataType: Int {
    case float16 = 10
    case float32 = 11
    case float64 = 12
    case int8 = 21
    case int16 = 22
    case int32 = 23
    case int64 = 24
    case bool = 1
    case string = 2
}

/// CoreML ArrayDataType enum values (from FeatureTypes.proto).
/// Used in model description input/output feature types.
internal enum ArrayDataType: Int {
    case float16 = 65552
    case float32 = 65568
    case float64 = 65600
    case int32 = 131104
}

// MARK: - Top-level model

/// Builds a complete CoreML Model protobuf.
///
/// Model proto layout:
///   specificationVersion: int32 (field 1)
///   description: ModelDescription (field 2)
///   mlProgram: MILSpec.Program (field 502)
internal struct CoreMLProtoBuilder {

    /// Builds a complete Model protobuf binary.
    static func buildModel(
        specVersion: Int32,
        inputs: [(name: String, shape: [Int])],
        outputs: [(name: String, shape: [Int])],
        program: (ProtobufWriter) -> Void
    ) -> Data {
        let writer = ProtobufWriter()
        writer.writeInt32(1, value: specVersion)
        writer.writeMessage(2) { desc in
            writeModelDescription(desc, inputs: inputs, outputs: outputs)
        }
        writer.writeMessage(502, program)
        return writer.data
    }

    // MARK: - ModelDescription

    /// ModelDescription: input=1, output=10
    private static func writeModelDescription(
        _ w: ProtobufWriter,
        inputs: [(name: String, shape: [Int])],
        outputs: [(name: String, shape: [Int])]
    ) {
        for input in inputs {
            w.writeMessage(1) { fd in
                writeFeatureDescription(fd, name: input.name, shape: input.shape, dataType: .float32)
            }
        }
        for output in outputs {
            w.writeMessage(10) { fd in
                writeFeatureDescription(fd, name: output.name, shape: output.shape, dataType: .float32)
            }
        }
    }

    /// FeatureDescription: name=1, type=3
    private static func writeFeatureDescription(
        _ w: ProtobufWriter,
        name: String,
        shape: [Int],
        dataType: ArrayDataType
    ) {
        w.writeString(1, value: name)
        w.writeMessage(3) { ft in
            // FeatureType: multiArrayType=5
            ft.writeMessage(5) { arr in
                // ArrayFeatureType: shape=1, dataType=2
                for dim in shape {
                    arr.writeInt64(1, value: Int64(dim))
                }
                arr.writeEnum(2, value: dataType.rawValue)
            }
        }
    }

    // MARK: - MIL Program

    /// MILSpec.Program: version=1, functions=2
    static func writeMILProgram(
        _ w: ProtobufWriter,
        version: String = "1",
        functions: [(name: String, builder: (ProtobufWriter) -> Void)]
    ) {
        w.writeString(1, value: version)
        for fn in functions {
            w.writeMapEntry(2, key: fn.name, value: fn.builder)
        }
    }

    /// MILSpec.Function: inputs=1, opset=2, block_specializations=3
    static func writeMILFunction(
        _ w: ProtobufWriter,
        inputs: [(name: String, shape: [Int], dataType: MILDataType)],
        opset: String = "CoreML7",
        block: (ProtobufWriter) -> Void
    ) {
        for input in inputs {
            w.writeMessage(1) { nvt in
                writeNamedValueType(nvt, name: input.name, shape: input.shape, dataType: input.dataType)
            }
        }
        w.writeString(2, value: opset)
        w.writeMapEntry(3, key: opset, value: block)
    }

    /// MILSpec.Block: inputs=1, outputs=2, operations=3
    static func writeMILBlock(
        _ w: ProtobufWriter,
        outputs: [String],
        operations: [(ProtobufWriter) -> Void]
    ) {
        for op in operations {
            w.writeMessage(3, op)
        }
        for output in outputs {
            // Block.outputs is repeated string (field 2)
            w.writeString(2, value: output)
        }
    }

    // MARK: - Operations

    /// MILSpec.Operation: type=1, inputs=2, outputs=3, attributes=5
    static func writeMILOperation(
        _ w: ProtobufWriter,
        type: String,
        inputs: [(name: String, argument: (ProtobufWriter) -> Void)],
        outputs: [(name: String, shape: [Int], dataType: MILDataType)]
    ) {
        w.writeString(1, value: type)
        for input in inputs {
            w.writeMapEntry(2, key: input.name, value: input.argument)
        }
        for output in outputs {
            w.writeMessage(3) { nvt in
                writeNamedValueType(nvt, name: output.name, shape: output.shape, dataType: output.dataType)
            }
        }
    }

    // MARK: - Arguments and Values

    /// MILSpec.Argument: arguments=1 (repeated Binding)
    static func writeArgument(
        _ w: ProtobufWriter,
        bindings: [(name: String, value: ((ProtobufWriter) -> Void)?)]
    ) {
        for binding in bindings {
            w.writeMessage(1) { b in
                // Binding: name=1, value=2
                b.writeString(1, value: binding.name)
                if let valueWriter = binding.value {
                    b.writeMessage(2, valueWriter)
                }
            }
        }
    }

    /// Writes a variable reference argument (just the name, no value).
    static func writeVariableArgument(_ w: ProtobufWriter, name: String) {
        writeArgument(w, bindings: [(name: name, value: nil)])
    }

    /// Writes an immediate value argument with the given value.
    static func writeImmediateArgument(
        _ w: ProtobufWriter,
        name: String,
        value: @escaping (ProtobufWriter) -> Void
    ) {
        writeArgument(w, bindings: [(name: name, value: value)])
    }

    // MARK: - Values

    /// MILSpec.Value: type=2, immediateValue=3
    static func writeValue(
        _ w: ProtobufWriter,
        shape: [Int],
        dataType: MILDataType,
        immediateValue: (ProtobufWriter) -> Void
    ) {
        // type (field 2)
        w.writeMessage(2) { vt in
            writeValueType(vt, shape: shape, dataType: dataType)
        }
        // immediateValue (field 3)
        w.writeMessage(3, immediateValue)
    }

    /// Writes a Value with float immediate data.
    static func writeFloatValue(_ w: ProtobufWriter, shape: [Int], values: [Float]) {
        writeValue(w, shape: shape, dataType: .float32) { iv in
            // ImmediateValue: tensor=1
            iv.writeMessage(1) { tv in
                // TensorValue: floats=1
                tv.writeMessage(1) { rf in
                    // RepeatedFloats: values=1 (packed)
                    rf.writePackedFloats(1, values: values)
                }
            }
        }
    }

    /// Writes a Value with int immediate data.
    static func writeIntValue(_ w: ProtobufWriter, shape: [Int], values: [Int32]) {
        writeValue(w, shape: shape, dataType: .int32) { iv in
            iv.writeMessage(1) { tv in
                // TensorValue: ints=2
                tv.writeMessage(2) { ri in
                    ri.writePackedInt32s(1, values: values)
                }
            }
        }
    }

    /// Writes a Value with bool immediate data.
    static func writeBoolValue(_ w: ProtobufWriter, shape: [Int], values: [Bool]) {
        writeValue(w, shape: shape, dataType: .bool) { iv in
            iv.writeMessage(1) { tv in
                // TensorValue: bools=3
                tv.writeMessage(3) { rb in
                    rb.writePackedBools(1, values: values)
                }
            }
        }
    }

    /// Writes a Value with string immediate data.
    static func writeStringValue(_ w: ProtobufWriter, value: String) {
        writeValue(w, shape: [], dataType: .string) { iv in
            iv.writeMessage(1) { tv in
                // TensorValue: strings=4
                tv.writeMessage(4) { rs in
                    rs.writeString(1, value: value)
                }
            }
        }
    }

    /// MILSpec.Value with BlobFileValue: fileName=1, offset=2
    static func writeBlobFileValue(
        _ w: ProtobufWriter,
        shape: [Int],
        dataType: MILDataType,
        fileName: String,
        offset: UInt64
    ) {
        w.writeMessage(2) { vt in
            writeValueType(vt, shape: shape, dataType: dataType)
        }
        w.writeMessage(5) { bfv in
            // BlobFileValue: fileName=1, offset=2
            bfv.writeString(1, value: fileName)
            bfv.writeUInt64(2, value: offset)
        }
    }

    // MARK: - Type helpers

    /// MILSpec.NamedValueType: name=1, type=2
    static func writeNamedValueType(
        _ w: ProtobufWriter,
        name: String,
        shape: [Int],
        dataType: MILDataType
    ) {
        w.writeString(1, value: name)
        w.writeMessage(2) { vt in
            writeValueType(vt, shape: shape, dataType: dataType)
        }
    }

    /// MILSpec.ValueType: tensorType=1
    static func writeValueType(
        _ w: ProtobufWriter,
        shape: [Int],
        dataType: MILDataType
    ) {
        w.writeMessage(1) { tt in
            writeTensorType(tt, shape: shape, dataType: dataType)
        }
    }

    /// MILSpec.TensorType: dataType=1, rank=2, dimensions=3
    static func writeTensorType(
        _ w: ProtobufWriter,
        shape: [Int],
        dataType: MILDataType
    ) {
        w.writeEnum(1, value: dataType.rawValue)
        w.writeInt64(2, value: Int64(shape.count))
        for dim in shape {
            // Dimension: constant=1 -> ConstantDimension: size=1
            w.writeMessage(3) { d in
                d.writeMessage(1) { cd in
                    cd.writeInt64(1, value: Int64(dim))
                }
            }
        }
    }
}
