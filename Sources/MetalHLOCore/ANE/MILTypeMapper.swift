// MILTypeMapper.swift
// MetalHLOCore
//
// Maps MetalHLO types and shapes to MIL type annotations.
// MIL uses `tensor<dtype, [shape]>` format.

/// Maps MetalHLO types and shapes to MIL type annotations.
///
/// All types are coerced to fp16 for ANE execution. Shape annotations
/// use logical shapes — the ANE compiler handles physical layout internally.
internal struct MILTypeMapper {

    /// Formats a shape as MIL tensor type: `tensor<fp16, [d0, d1, ...]>`
    /// Scalar: `fp16`
    static func tensorType(shape: [Int]) -> String {
        if shape.isEmpty {
            return "fp16"
        }
        let shapeStr = shape.map(String.init).joined(separator: ", ")
        return "tensor<fp16, [\(shapeStr)]>"
    }

    /// Sanitizes an SSA name for MIL usage.
    ///
    /// HLO uses "%0", "%arg0", etc. MIL uses plain names without %.
    /// Strips the leading "%" and replaces characters that might be
    /// invalid in MIL identifiers.
    static func sanitizeName(_ name: String) -> String {
        var clean = name
        if clean.hasPrefix("%") {
            clean = String(clean.dropFirst())
        }
        // Replace characters that might be invalid in MIL identifiers
        clean = clean.replacingOccurrences(of: "#", with: "_")
        clean = clean.replacingOccurrences(of: ".", with: "_")
        return clean
    }
}
