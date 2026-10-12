// ChloExpansion.swift
// MetalHLOCore
//
// Expands the CHLO special functions a frontend may emit directly
// (`chlo.lgamma`, `chlo.digamma`, `chlo.polygamma`) into elementwise StableHLO
// ops, the way XLA legalizes them, so both backends run them with no special
// kernel. Lanczos (g = 7) for lgamma/digamma with the reflection formula below
// 0.5; Hurwitz zeta by Euler–Maclaurin for polygamma.

import Foundation

/// Builds a chain of elementwise ops over one tensor type. Every value it
/// creates is named `<prefix>_chloN`; `finish(as:)` renames the last op.
struct ChloExpander {
    private(set) var ops: [HLOOperation] = []
    let type: TensorType
    let prefix: String
    private var counter = 0

    init(type: TensorType, prefix: String) {
        self.type = type
        self.prefix = prefix
    }

    private var boolType: TensorType { TensorType(shape: type.shape, elementType: .int1) }

    private mutating func emit(_ kind: HLOOpKind, _ operands: [String], type t: TensorType? = nil,
                               attributes: HLOAttributes = HLOAttributes()) -> String {
        counter += 1
        let name = "\(prefix)_chlo\(counter)"
        ops.append(HLOOperation(result: name, kind: kind, operands: operands, resultType: t ?? type,
                                attributes: attributes))
        return name
    }

    mutating func const(_ value: Double) -> String {
        var a = HLOAttributes()
        a.constantValue = .splat(value, type)
        return emit(.constant, [], attributes: a)
    }

    mutating func unary(_ kind: HLOOpKind, _ x: String) -> String {
        var a = HLOAttributes()
        a.inputElementTypes = [type.elementType]
        return emit(kind, [x], attributes: a)
    }

    mutating func binary(_ kind: HLOOpKind, _ x: String, _ y: String) -> String {
        var a = HLOAttributes()
        a.inputElementTypes = [type.elementType, type.elementType]
        return emit(kind, [x, y], attributes: a)
    }

    mutating func add(_ x: String, _ y: String) -> String { binary(.add, x, y) }
    mutating func sub(_ x: String, _ y: String) -> String { binary(.subtract, x, y) }
    mutating func mul(_ x: String, _ y: String) -> String { binary(.multiply, x, y) }
    mutating func div(_ x: String, _ y: String) -> String { binary(.divide, x, y) }

    mutating func compare(_ direction: ComparisonDirection, _ x: String, _ y: String) -> String {
        var a = HLOAttributes()
        a.comparisonDirection = direction
        a.inputElementTypes = [type.elementType, type.elementType]
        return emit(.compare, [x, y], type: boolType, attributes: a)
    }

    mutating func and(_ p: String, _ q: String) -> String {
        var a = HLOAttributes()
        a.inputElementTypes = [.int1, .int1]
        return emit(.and, [p, q], type: boolType, attributes: a)
    }

    mutating func select(_ p: String, _ x: String, _ y: String) -> String {
        var a = HLOAttributes()
        a.inputElementTypes = [.int1, type.elementType, type.elementType]
        return emit(.select, [p, x, y], attributes: a)
    }

    /// Renames the last op's result to `name` (the op being expanded).
    mutating func finish(as name: String) {
        let last = ops.removeLast()
        ops.append(HLOOperation(result: name, kind: last.kind, operands: last.operands.map { $0 },
                                resultType: last.resultType, attributes: last.attributes))
    }

    // MARK: - Special functions

    private static let lanczosGamma = 7.0
    private static let lanczosBase = 0.99999999999980993227684700473478
    private static let lanczosCoefficients: [Double] = [
        676.520368121885098567009190444019, -1259.13921672240287047156078755283,
        771.3234287776530788486528258894, -176.61502916214059906584551354,
        12.507343278686904814458936853, -0.13857109526572011689554707,
        9.984369578019570859563e-6, 1.50563273514931155834e-7,
    ]

    /// log Γ(x).
    mutating func lgamma(_ x: String) -> String {
        let half = const(0.5), one = const(1)
        let reflect = compare(.lt, x, half)
        let z = select(reflect, unary(.negate, x), sub(x, one))
        var sum = const(Self.lanczosBase)
        for (i, c) in Self.lanczosCoefficients.enumerated() {
            sum = add(sum, div(const(c), add(z, const(Double(i + 1)))))
        }
        let gHalf = Self.lanczosGamma + 0.5
        let t = add(z, const(gHalf))
        let logT = add(const(Foundation.log(gHalf)), unary(.log1p, div(z, const(gHalf))))
        // log y = log √(2π) + (z + ½ − t / log t) · log t + log(sum)
        let logY = add(add(const(0.5 * Foundation.log(2 * Double.pi)),
                           mul(sub(add(z, half), div(t, logT)), logT)),
                       unary(.log, sum))
        // Reflection: log Γ(x) = log π − log|sin(π x)| − log Γ(1 − x).
        let absX = unary(.abs, x)
        let frac = sub(absX, unary(.floor, absX))
        let reduced = select(compare(.gt, frac, half), sub(one, frac), frac)
        let denom = unary(.log, unary(.sine, mul(const(Double.pi), reduced)))
        let finiteDenom = compare(.lt, unary(.abs, denom), const(.infinity))
        let reflection = select(finiteDenom, sub(sub(const(Foundation.log(Double.pi)), denom), logY),
                                unary(.negate, denom))
        let result = select(reflect, reflection, logY)
        // ±∞ → +∞.
        return select(compare(.eq, absX, const(.infinity)), const(.infinity), result)
    }

    /// ψ(x) = d/dx log Γ(x).
    mutating func digamma(_ x: String) -> String {
        let half = const(0.5), one = const(1)
        let reflect = compare(.lt, x, half)
        let z = select(reflect, unary(.negate, x), sub(x, one))
        var num = const(0)
        var denom = const(Self.lanczosBase)
        for (i, c) in Self.lanczosCoefficients.enumerated() {
            let zi = add(z, const(Double(i + 1)))
            num = sub(num, div(const(c), mul(zi, zi)))
            denom = add(denom, div(const(c), zi))
        }
        let gHalf = Self.lanczosGamma + 0.5
        let t = add(z, const(gHalf))
        let logT = add(const(Foundation.log(gHalf)), unary(.log1p, div(z, const(gHalf))))
        let y = sub(add(logT, div(num, denom)), div(const(Self.lanczosGamma), t))
        // Reflection: ψ(x) = ψ(1 − x) − π / tan(π x).
        let reduced = add(x, unary(.abs, unary(.floor, add(x, half))))
        let piR = mul(const(Double.pi), reduced)
        let reflection = sub(y, div(mul(const(Double.pi), unary(.cosine, piR)), unary(.sine, piR)))
        let result = select(reflect, reflection, y)
        // Poles at 0, −1, −2, …
        let pole = and(compare(.le, x, const(0)), compare(.eq, x, unary(.floor, x)))
        return select(pole, const(.nan), result)
    }

    /// B₂ₖ / (2k)! for k = 1…12.
    private static let bernoulliOverFactorial: [Double] = {
        let b: [Double] = [1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66, -691.0 / 2730, 7.0 / 6,
                           -3617.0 / 510, 43867.0 / 798, -174611.0 / 330, 854513.0 / 138,
                           -236364091.0 / 2730]
        var factorial = 1.0
        return b.enumerated().map { k, bk in
            let n = 2 * (k + 1)
            factorial *= Double(n - 1) * Double(n)
            return bk / factorial
        }
    }()

    /// Hurwitz ζ(s, q) = Σₖ (q + k)^−s, by Euler–Maclaurin: ten terms of the
    /// series, then the integral and Bernoulli corrections at a = q + 10.
    mutating func zeta(_ s: String, _ q: String) -> String {
        let one = const(1)
        var sum = binary(.power, q, unary(.negate, s))
        var a = q
        for _ in 0..<9 {
            a = add(a, one)
            sum = add(sum, binary(.power, a, unary(.negate, s)))
        }
        a = add(a, one)
        let negPower = binary(.power, a, unary(.negate, s))  // a^−s
        // a^(1−s) / (s − 1) + a^−s / 2
        var total = add(sum, div(mul(negPower, a), sub(s, one)))
        // a^−s · (s / a) · Σⱼ cⱼ (s+1)…(s+2j−2) / a^(2j−2), by Horner from j = 12.
        let invA2 = div(one, mul(a, a))
        let c = Self.bernoulliOverFactorial
        var horner = const(0)
        for j in stride(from: c.count, through: 2, by: -1) {
            let f = mul(mul(add(s, const(Double(2 * j - 3))), add(s, const(Double(2 * j - 2)))), invA2)
            horner = mul(f, add(const(c[j - 1]), horner))
        }
        let correction = mul(negPower, add(const(0.5), mul(div(s, a), add(const(c[0]), horner))))
        total = add(total, correction)
        // Undefined for s < 1; for q <= 0 only at integer s, and a pole at integer q <= 0.
        let nan = const(.nan)
        total = select(compare(.lt, s, one), nan, total)
        let qNonPositive = compare(.le, q, const(0))
        total = select(and(qNonPositive, compare(.ne, s, unary(.floor, s))), nan, total)
        total = select(and(qNonPositive, compare(.eq, q, unary(.floor, q))), const(.infinity), total)
        return total
    }

    /// ψ⁽ⁿ⁾(x): ψ(x) for n = 0, else (−1)ⁿ⁺¹ n! ζ(n + 1, x); NaN for n not a
    /// non-negative integer.
    mutating func polygamma(_ n: String, _ x: String) -> String {
        let one = const(1)
        let nPlusOne = add(n, one)
        let factorial = unary(.exponential, lgamma(nPlusOne))
        // (−1)ⁿ⁺¹: +1 for odd n, −1 for even n.
        let two = const(2)
        let parity = sub(n, mul(two, unary(.floor, div(n, two))))
        let sign = sub(mul(two, parity), one)
        let higher = mul(mul(sign, factorial), zeta(nPlusOne, x))
        var result = select(compare(.eq, n, const(0)), digamma(x), higher)
        let notNatural = compare(.ne, n, unary(.floor, n))
        result = select(notNatural, const(.nan), result)
        return select(compare(.lt, n, const(0)), const(.nan), result)
    }
}
