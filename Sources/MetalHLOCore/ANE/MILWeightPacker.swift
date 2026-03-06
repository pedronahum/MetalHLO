// MILWeightPacker.swift
// MetalHLOCore
//
// Packs constant data into a single weight blob with BLOBFILE references.
// Small constants are emitted inline in MIL text. Large constants are
// packed into a binary FP16 blob.

import Foundation

/// Packs constant data into a single weight blob with BLOBFILE references.
///
/// Small constants (scalars, small arrays ≤16 elements) are emitted inline.
/// Large constants are packed into a binary FP16 blob and referenced via
/// `BLOBFILE(offset=X, size=Y)`.
internal final class MILWeightPacker {

    /// A weight entry recording name, offset, and size within the blob.
    struct WeightEntry {
        let name: String
        let offset: Int
        let size: Int
    }

    /// Constants with more elements than this use BLOBFILE instead of inline.
    static let inlineThreshold = 16

    private var blobData = Data()
    private var currentOffset = 0
    private var weightEntries: [WeightEntry] = []

    /// Packs a ConstantValue and emits the appropriate MIL statement.
    ///
    /// - Parameters:
    ///   - name: The MIL variable name (without %).
    ///   - value: The constant value from HLO IR.
    ///   - resultType: The result tensor type.
    ///   - builder: The MIL text builder.
    func packConstant(
        name: String,
        value: ConstantValue,
        resultType: TensorType,
        builder: MILTextBuilder
    ) {
        let shape = resultType.shape

        switch value {
        case .scalar(let v):
            let fp16Val = Double(Float16(Float(v)))
            builder.emitInlineConst(
                name: name,
                shape: shape,
                value: formatFloat(fp16Val)
            )

        case .splat(let v, let type):
            let fp16Val = Double(Float16(Float(v)))
            if type.count <= Self.inlineThreshold {
                let arr = Array(repeating: formatFloat(fp16Val), count: type.count)
                builder.emitInlineConst(
                    name: name,
                    shape: shape,
                    value: "[\(arr.joined(separator: ", "))]"
                )
            } else {
                let data = makeFP16Data(Array(repeating: Float(v), count: type.count))
                let offset = appendBlob(data, name: name)
                builder.emitBlobConst(
                    name: name,
                    shape: shape,
                    offset: offset,
                    size: data.count
                )
            }

        case .dense(let values, let type):
            if values.count <= Self.inlineThreshold {
                let fp16Strs = values.map { formatFloat(Double(Float16(Float($0)))) }
                builder.emitInlineConst(
                    name: name,
                    shape: shape,
                    value: "[\(fp16Strs.joined(separator: ", "))]"
                )
            } else {
                let data = makeFP16Data(values.map { Float($0) })
                let offset = appendBlob(data, name: name)
                builder.emitBlobConst(
                    name: name,
                    shape: shape,
                    offset: offset,
                    size: data.count
                )
            }
        }
    }

    /// Returns the accumulated weight blob data.
    func getWeightData() -> Data {
        return blobData
    }

    /// Returns the layout of all BLOBFILE weight entries.
    /// Used by MILWeightTemplate to patch weights without recompilation.
    func getWeightLayout() -> [WeightEntry] {
        return weightEntries
    }

    /// Returns true if any weight data has been accumulated.
    var hasWeights: Bool {
        return !blobData.isEmpty
    }

    // MARK: - Private

    private func appendBlob(_ data: Data, name: String) -> Int {
        let offset = currentOffset
        blobData.append(data)
        currentOffset += data.count
        weightEntries.append(WeightEntry(name: name, offset: offset, size: data.count))
        return offset
    }

    private func makeFP16Data(_ floats: [Float]) -> Data {
        var fp16Values = floats.map { Float16($0) }
        return Data(bytes: &fp16Values, count: fp16Values.count * MemoryLayout<Float16>.size)
    }

    private func formatFloat(_ v: Double) -> String {
        if v == 0.0 { return "0.0" }
        if v == 1.0 { return "1.0" }
        if v == -1.0 { return "-1.0" }
        return String(format: "%.6g", v)
    }
}
