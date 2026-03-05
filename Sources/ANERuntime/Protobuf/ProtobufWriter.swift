// ProtobufWriter.swift
// ANERuntime
//
// Minimal protobuf wire format encoder for generating CoreML model specs.
// Supports varint, length-delimited (strings, bytes, sub-messages),
// fixed32/64 (floats, doubles), and packed repeated fields.
// No external dependencies.

import Foundation

/// Encodes data in protobuf binary wire format.
///
/// Wire types:
/// - 0: varint (int32, int64, uint32, uint64, bool, enum)
/// - 1: 64-bit (double, fixed64)
/// - 2: length-delimited (string, bytes, sub-messages, packed repeated)
/// - 5: 32-bit (float, fixed32)
internal final class ProtobufWriter {

    private(set) var data = Data()

    init() {}

    // MARK: - Low-level encoding

    /// Encodes a varint (variable-length integer).
    func writeRawVarint(_ value: UInt64) {
        var v = value
        while v > 0x7F {
            data.append(UInt8(v & 0x7F) | 0x80)
            v >>= 7
        }
        data.append(UInt8(v))
    }

    /// Encodes a field tag (field number + wire type).
    func writeTag(fieldNumber: Int, wireType: Int) {
        writeRawVarint(UInt64(fieldNumber << 3 | wireType))
    }

    // MARK: - Varint fields (wire type 0)

    func writeUInt64(_ fieldNumber: Int, value: UInt64) {
        writeTag(fieldNumber: fieldNumber, wireType: 0)
        writeRawVarint(value)
    }

    func writeInt64(_ fieldNumber: Int, value: Int64) {
        writeUInt64(fieldNumber, value: UInt64(bitPattern: value))
    }

    func writeInt32(_ fieldNumber: Int, value: Int32) {
        writeUInt64(fieldNumber, value: UInt64(UInt32(bitPattern: value)))
    }

    func writeBool(_ fieldNumber: Int, value: Bool) {
        writeUInt64(fieldNumber, value: value ? 1 : 0)
    }

    func writeEnum(_ fieldNumber: Int, value: Int) {
        writeUInt64(fieldNumber, value: UInt64(value))
    }

    // MARK: - Fixed-width fields

    /// Writes a float (wire type 5 = 32-bit).
    func writeFloat(_ fieldNumber: Int, value: Float) {
        writeTag(fieldNumber: fieldNumber, wireType: 5)
        var v = value
        withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
    }

    /// Writes a double (wire type 1 = 64-bit).
    func writeDouble(_ fieldNumber: Int, value: Double) {
        writeTag(fieldNumber: fieldNumber, wireType: 1)
        var v = value
        withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
    }

    // MARK: - Length-delimited fields (wire type 2)

    func writeString(_ fieldNumber: Int, value: String) {
        let bytes = Data(value.utf8)
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeRawVarint(UInt64(bytes.count))
        data.append(bytes)
    }

    func writeBytes(_ fieldNumber: Int, value: Data) {
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeRawVarint(UInt64(value.count))
        data.append(value)
    }

    /// Writes a nested message built by the closure.
    func writeMessage(_ fieldNumber: Int, _ builder: (ProtobufWriter) -> Void) {
        let sub = ProtobufWriter()
        builder(sub)
        writeBytes(fieldNumber, value: sub.data)
    }

    // MARK: - Map entries

    /// Writes a map<string, message> entry.
    /// Maps are encoded as repeated messages with key=1, value=2.
    func writeMapEntry(
        _ fieldNumber: Int,
        key: String,
        value: (ProtobufWriter) -> Void
    ) {
        writeMessage(fieldNumber) { entry in
            entry.writeString(1, value: key)
            entry.writeMessage(2, value)
        }
    }

    // MARK: - Packed repeated fields

    /// Writes packed repeated float values.
    func writePackedFloats(_ fieldNumber: Int, values: [Float]) {
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeRawVarint(UInt64(values.count * 4))
        for var v in values {
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }
    }

    /// Writes packed repeated int32 values.
    func writePackedInt32s(_ fieldNumber: Int, values: [Int32]) {
        // Compute total size (each int32 as varint)
        let sub = ProtobufWriter()
        for v in values {
            sub.writeRawVarint(UInt64(UInt32(bitPattern: v)))
        }
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeRawVarint(UInt64(sub.data.count))
        data.append(sub.data)
    }

    /// Writes packed repeated int64 values.
    func writePackedInt64s(_ fieldNumber: Int, values: [Int64]) {
        let sub = ProtobufWriter()
        for v in values {
            sub.writeRawVarint(UInt64(bitPattern: v))
        }
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeRawVarint(UInt64(sub.data.count))
        data.append(sub.data)
    }

    /// Writes packed repeated bool values.
    func writePackedBools(_ fieldNumber: Int, values: [Bool]) {
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeRawVarint(UInt64(values.count))
        for v in values {
            data.append(v ? 1 : 0)
        }
    }

    /// Writes packed repeated double values.
    func writePackedDoubles(_ fieldNumber: Int, values: [Double]) {
        writeTag(fieldNumber: fieldNumber, wireType: 2)
        writeRawVarint(UInt64(values.count * 8))
        for var v in values {
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }
    }
}
