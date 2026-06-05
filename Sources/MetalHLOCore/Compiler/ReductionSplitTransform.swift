// ReductionSplitTransform.swift
// MetalHLOCore
//
// HLO-level transform that splits an inefficient *global* reduction (one that
// reduces every axis to a scalar) into two cooperating reduce stages.
//
// A reduce-all has a single output element, so the codegen runs it in ONE
// threadgroup (the executor is strictly one-op → one-dispatch). For a large
// input (e.g. RED-001: sum of a 1024×1024 tensor) that leaves all but one of
// the GPU's threadgroups idle while 1024 threads chew through ~1M elements
// sequentially — ~0.27× of MLX.
//
// We rewrite `%out = reduce(%x) over all axes` into:
//
//     %partial = reduce(%x) over the trailing axes  →  [d0]   (stage 1)
//     %out     = reduce(%partial) over axis 0        →  scalar (stage 2)
//
// Stage 1 has `d0` outputs, so it fills `d0` threadgroups (good occupancy) and
// each output reduces a *contiguous* block (coalesced reads). Stage 2 reduces
// only the `d0` partials in one cheap threadgroup. The decomposition is exact
// for any associative reducer whose init is the identity (sum/max/min/product),
// and is order-stable (a fixed partial count → a fixed combine order), so it
// preserves the reproducibility the loss-exactness gates rely on.
//
// Activated by `METALHLO_SPLIT_REDUCE` (default ON; set to `0` to opt out).

import Foundation

/// Splits large global reductions into two stages when enabled. Off only when
/// `METALHLO_SPLIT_REDUCE=0`; otherwise applied.
public func applyReductionSplitIfEnabled(_ function: HLOFunction) -> HLOFunction {
    if ProcessInfo.processInfo.environment["METALHLO_SPLIT_REDUCE"] == "0" {
        return function
    }
    return applyReductionSplit(function)
}

/// Minimum total input elements for a split to pay off — below this the
/// single-threadgroup kernel is already fast enough that two dispatches lose.
private let reductionSplitMinElements = 1 << 14   // 16384

/// `d0` (the surviving axis / stage-2 reduce length) must sit in this range:
/// large enough that stage 1 fills enough threadgroups, small enough that the
/// stage-2 single-threadgroup reduce stays cheap.
private let reductionSplitMinD0 = 64
private let reductionSplitMaxD0 = 1 << 16          // 65536

public func applyReductionSplit(_ function: HLOFunction) -> HLOFunction {
    // result-name → type (for operand shapes) and constant init values.
    var typeMap: [String: TensorType] = [:]
    for input in function.inputs { typeMap[input.name] = input.type }
    for op in function.operations { typeMap[op.result] = op.resultType }

    var constMap: [String: ConstantValue] = [:]
    for op in function.operations where op.kind == .constant {
        if let cv = op.attributes.constantValue { constMap[op.result] = cv }
    }

    let debug = ProcessInfo.processInfo.environment["METALHLO_DEBUG_REDUCE_SPLIT"] == "1"

    var newOps: [HLOOperation] = []
    newOps.reserveCapacity(function.operations.count + 4)
    var splitCount = 0

    for (index, op) in function.operations.enumerated() {
        guard let inType = typeMap[op.operands.first ?? ""],
              shouldSplit(op, inputType: inType, constMap: constMap) else {
            newOps.append(op)
            continue
        }

        let inShape = inType.shape
        let rank = inShape.count
        let d0 = inShape[0]
        let elementType = inType.elementType
        let initOperand = op.operands[1]
        let kind = op.attributes.reductionKind ?? .sum

        // Stage 1: reduce the trailing axes [1 ..< rank] → [d0].
        let partialName = "\(op.result)_rsplit_\(index)"
        let partialType = TensorType(shape: [d0], elementType: elementType)
        var s1 = HLOAttributes()
        s1.dimensions = Array(1..<rank)
        s1.reductionKind = kind
        newOps.append(HLOOperation(
            result: partialName,
            kind: .reduce,
            operands: [op.operands[0], initOperand],
            resultType: partialType,
            attributes: s1
        ))
        typeMap[partialName] = partialType

        // Stage 2: reduce the [d0] partials → original scalar result.
        var s2 = HLOAttributes()
        s2.dimensions = [0]
        s2.reductionKind = kind
        newOps.append(HLOOperation(
            result: op.result,
            kind: .reduce,
            operands: [partialName, initOperand],
            resultType: op.resultType,
            attributes: s2
        ))

        splitCount += 1
        if debug {
            FileHandle.standardError.write(Data(
                "[reduce-split] \(op.result): \(inShape) all-axes \(kind) → stage1 [d0=\(d0)] + stage2 scalar\n".utf8))
        }
    }

    if splitCount == 0 { return function }

    return HLOFunction(
        name: function.name,
        isPrivate: function.isPrivate,
        inputs: function.inputs,
        outputTypes: function.outputTypes,
        operations: newOps,
        returnValues: function.returnValues
    )
}

/// A reduce qualifies for the 2-stage split when it reduces every axis of a
/// rank ≥ 2 input to a scalar, uses an associative reducer with an identity
/// init, the input is large, and the surviving first axis `d0` is in a range
/// that keeps both stages efficient.
private func shouldSplit(
    _ op: HLOOperation,
    inputType: TensorType,
    constMap: [String: ConstantValue]
) -> Bool {
    guard op.kind == .reduce, op.operands.count == 2 else { return false }

    let kind = op.attributes.reductionKind ?? .sum
    switch kind {
    case .sum, .max, .min, .product: break
    default: return false   // mean (needs a final scale) and bitwise: skip.
    }

    let shape = inputType.shape
    let rank = shape.count
    guard rank >= 2 else { return false }

    // Must reduce *every* axis (scalar output).
    let dims = Set(op.attributes.dimensions ?? [])
    guard dims == Set(0..<rank) else { return false }

    let total = shape.reduce(1, *)
    guard total >= reductionSplitMinElements else { return false }

    let d0 = shape[0]
    guard d0 >= reductionSplitMinD0, d0 <= reductionSplitMaxD0 else { return false }

    // Init must be the reducer's identity so stage1+stage2 compose exactly.
    guard let initCV = constMap[op.operands[1]],
          isIdentityInit(initCV, for: kind) else { return false }

    return true
}

/// True when `cv` is the identity element for `kind` (sum→0, product→1,
/// max→−∞, min→+∞), tolerating the finite sentinels graph emitters use for ±∞.
private func isIdentityInit(_ cv: ConstantValue, for kind: ReductionKind) -> Bool {
    let v: Double
    switch cv {
    case .scalar(let d): v = d
    case .splat(let d, _): v = d
    case .dense(let arr, _) where arr.count == 1: v = arr[0]
    default: return false
    }
    switch kind {
    case .sum: return v == 0
    case .product: return v == 1
    case .max: return v == -Double.infinity || v <= -3.0e38
    case .min: return v == Double.infinity || v >= 3.0e38
    default: return false
    }
}
