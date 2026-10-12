// NarrowFloat.swift
// MetalHLOCore
//
// Host-side codecs for the narrow float storage types. Metal has no 8- or 4-bit
// float arithmetic: programs compute them as f32 holding exactly representable
// values (NarrowFloatLegalization), and buffers are converted at the
// Executable boundary with these functions.

import Foundation

public enum NarrowFloat {

    // MARK: f8E4M3FN — 1 sign, 4 exponent (bias 7), 3 mantissa bits; no
    // infinities, NaN = S.1111.111, largest finite 448.

    /// The value of an f8E4M3FN byte.
    public static func decodeF8E4M3FN(_ byte: UInt8) -> Float {
        let sign: Float = byte & 0x80 != 0 ? -1 : 1
        let exponent = Int((byte >> 3) & 0xF)
        let mantissa = Float(byte & 0x7)
        if exponent == 0xF && byte & 0x7 == 0x7 { return .nan }
        if exponent == 0 { return sign * mantissa * 0x1p-9 }
        return sign * (1 + mantissa / 8) * Float(sign: .plus, exponent: exponent - 7, significand: 1)
    }

    /// `x` rounded to the nearest f8E4M3FN value (ties to even), as a Float:
    /// NaN for NaN, ±∞ and magnitudes that round past 448 (XLA's
    /// non-saturating conversion). The same arithmetic as the in-graph
    /// rounding NarrowFloatLegalization emits.
    public static func roundToF8E4M3FN(_ x: Float) -> Float {
        if x.isNaN { return .nan }
        let a = abs(x)
        let exponent = max(Int(a.bitPattern >> 23) - 127, -6)
        let quantum = Float(sign: .plus, exponent: exponent - 3, significand: 1)
        let r = (a / quantum).rounded(.toNearestOrEven) * quantum
        if r > 448 { return .nan }
        return x < 0 ? -r : r
    }

    /// The f8E4M3FN byte for `x` (rounded as `roundToF8E4M3FN`).
    public static func encodeF8E4M3FN(_ x: Float) -> UInt8 {
        let r = roundToF8E4M3FN(x)
        let sign: UInt8 = (x.sign == .minus) ? 0x80 : 0
        if r.isNaN { return sign | 0x7F }
        let a = abs(r)
        if a == 0 { return sign }
        if a < 0x1p-6 { return sign | UInt8(a / 0x1p-9) }  // subnormal: mantissa · 2⁻⁹
        let exponent = Int(a.exponent)
        let mantissa = UInt8(((a / Float(sign: .plus, exponent: exponent, significand: 1)) - 1) * 8)
        return sign | UInt8(exponent + 7) << 3 | mantissa
    }

    // MARK: f4E2M1FN — 1 sign, 2 exponent (bias 1), 1 mantissa bit; values
    // 0, 0.5, 1, 1.5, 2, 3, 4, 6; no infinities or NaN.

    /// The value of the low 4 bits of `nibble` as f4E2M1FN.
    public static func decodeF4E2M1FN(_ nibble: UInt8) -> Float {
        let sign: Float = nibble & 0x8 != 0 ? -1 : 1
        let exponent = Int((nibble >> 1) & 0x3)
        let mantissa = Float(nibble & 0x1)
        if exponent == 0 { return sign * mantissa * 0.5 }
        return sign * (1 + mantissa / 2) * Float(sign: .plus, exponent: exponent - 1, significand: 1)
    }

    /// The f4E2M1FN nibble nearest `x` (ties to even), saturating at ±6.
    public static func encodeF4E2M1FN(_ x: Float) -> UInt8 {
        let sign: UInt8 = x.sign == .minus ? 0x8 : 0
        let magnitudes: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        let a = x.isNaN ? 0 : min(abs(x), 6)
        var best = 0
        for (i, m) in magnitudes.enumerated() {
            let d = abs(a - m), bestD = abs(a - magnitudes[best])
            if d < bestD || (d == bestD && i % 2 == 0 && best % 2 == 1) { best = i }
        }
        return sign | UInt8(best)
    }
}
