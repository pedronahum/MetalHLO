// TF32Transform.swift
// MetalHLOCore
//
// HLO-level transformation that wraps fp32 dot/dot_general ops with convert
// operations so that the matmul executes at fp16 internally — analogous to
// JAX's `jax_default_matmul_precision='tf32'`. Apple Silicon's matrix
// coprocessors run fp16 matmul at ~2x fp32 throughput.
//
// For each operation `%c = dot %a, %b` where a/b/c are fp32, we rewrite to:
//
//     %a_tf32h = convert %a   : fp32 -> fp16
//     %b_tf32h = convert %b   : fp32 -> fp16
//     %c_tf32h = dot %a_tf32h, %b_tf32h   : fp16 -> fp16
//     %c       = convert %c_tf32h         : fp16 -> fp32
//
// The original SSA name `%c` is preserved so downstream consumers don't need
// to be rewritten. The intermediate fp16 dot uses a fresh `%c_tf32h_<index>`
// name that is guaranteed unique within the function.
//
// Activated by env var `METALHLO_MATMUL_TF32=1`. Off by default.

import Foundation

/// Applies the TF32 matmul transformation to a function if
/// `METALHLO_MATMUL_TF32=1` is set in the environment. Otherwise returns the
/// function unchanged.
///
/// `fuseOutputConvert` (gate predicate, M/N/K/batchSize → Bool): when the
/// predicate returns true for a given dot, the trailing fp16→fp32 convert is
/// dropped and the dot is emitted as mixed-precision (fp16 inputs, fp32
/// output). The codegen then routes that dot to a `half × half → float` MPP
/// kernel which fuses the output cast in-register, eliminating one device-
/// memory round-trip. Predicate defaults to never-fuse for safety; only the
/// MetalHLO compiler turns it on, gated on Apple9 GPU support.
public func applyTF32MatmulTransformIfEnabled(
    _ function: HLOFunction,
    fuseOutputConvert: @Sendable (Int, Int, Int, Int) -> Bool = { _, _, _, _ in false }
) -> HLOFunction {
    guard ProcessInfo.processInfo.environment["METALHLO_MATMUL_TF32"] == "1" else {
        return function
    }
    return applyTF32MatmulTransform(function, fuseOutputConvert: fuseOutputConvert)
}

/// Wraps every fp32 `.dot` / `.dotGeneral` op in the function with
/// fp32→fp16→fp32 convert ops so the matmul itself runs at fp16.
///
/// When `fuseOutputConvert(M, N, K, batchSize)` returns true for a dot, the
/// trailing fp16→fp32 convert is omitted and the dot emits a mixed-precision
/// result (fp16 inputs, fp32 output) — the codegen's MPP path then handles
/// the cast in-register.
public func applyTF32MatmulTransform(
    _ function: HLOFunction,
    fuseOutputConvert: @Sendable (Int, Int, Int, Int) -> Bool = { _, _, _, _ in false }
) -> HLOFunction {
    // Build a value-name → TensorType map covering function inputs and every
    // op result. Used to look up operand types when deciding whether a cast
    // is needed.
    var typeMap: [String: TensorType] = [:]
    for input in function.inputs {
        typeMap[input.name] = input.type
    }
    for op in function.operations {
        typeMap[op.result] = op.resultType
    }

    var newOps: [HLOOperation] = []
    newOps.reserveCapacity(function.operations.count + 8)
    var transformCount = 0

    for (index, op) in function.operations.enumerated() {
        // Only transform fp32 dot/dot_general. Other ops, and dots that already
        // produce non-fp32 output, pass through untouched.
        guard op.kind == .dot || op.kind == .dotGeneral else {
            newOps.append(op)
            continue
        }
        guard op.resultType.elementType == .float32 else {
            newOps.append(op)
            continue
        }

        // Size gate: only apply TF32 when the compute savings dwarf the
        // fp32→fp16 / fp16→fp32 convert overhead. Empirically on M5 Pro
        // (9.5 TFLOPS fp32, ~300 GB/s memory bandwidth), the codegen
        // simdgroup_half kernel is only ~2x faster than fp32, so TF32 is a
        // win only when M*N*K (compute) substantially exceeds the operand+
        // output surface area (memory). Threshold of 200 corresponds to
        // square shapes with D ≳ 600 — protects small batched matmuls
        // (attention, small FFN) from convert-overhead-dominated regressions.
        if let surfaceRatio = computeSurfaceRatio(op: op, typeMap: typeMap),
           surfaceRatio < 200 {
            newOps.append(op)
            continue
        }

        // Cast each fp32 operand to fp16. Operands that are already non-fp32
        // pass through. If we can't determine an operand's type, bail and
        // keep the op untransformed (safer than emitting a malformed graph).
        var newOperands: [String] = []
        var convertOps: [HLOOperation] = []
        var bailOut = false
        for operand in op.operands {
            guard let operandType = typeMap[operand] else {
                bailOut = true
                break
            }
            if operandType.elementType != .float32 {
                newOperands.append(operand)
                continue
            }
            let castName = "\(operand)_tf32h_\(index)"
            let castType = TensorType(shape: operandType.shape, elementType: .float16)
            convertOps.append(HLOOperation(
                result: castName,
                kind: .convert,
                operands: [operand],
                resultType: castType
            ))
            typeMap[castName] = castType
            newOperands.append(castName)
        }
        if bailOut {
            newOps.append(op)
            continue
        }

        // Insert the operand convert ops.
        newOps.append(contentsOf: convertOps)

        // Decide whether to fuse the output convert into the matmul kernel.
        // The fusion path emits the dot directly producing fp32 from fp16
        // inputs; the codegen routes this to MPP `half × half → float` which
        // does the cast in-register.
        let dims = matmulDims(op: op, typeMap: typeMap)
        let fuse: Bool
        if let dims {
            fuse = fuseOutputConvert(dims.M, dims.N, dims.K, dims.batchSize)
        } else {
            fuse = false
        }

        if fuse {
            // Single mixed-precision dot — original SSA name preserved, no
            // trailing convert needed.
            let mixedDot = HLOOperation(
                result: op.result,
                kind: op.kind,
                operands: newOperands,
                resultType: op.resultType,    // stays fp32
                attributes: op.attributes,
                resultCount: op.resultCount
            )
            newOps.append(mixedDot)
            typeMap[op.result] = op.resultType
        } else {
            // Original sequence: fp16 dot + fp16→fp32 convert.
            let fp16ResultName = "\(op.result)_tf32h_\(index)"
            let fp16ResultType = TensorType(shape: op.resultType.shape, elementType: .float16)
            let fp16Dot = HLOOperation(
                result: fp16ResultName,
                kind: op.kind,
                operands: newOperands,
                resultType: fp16ResultType,
                attributes: op.attributes,
                resultCount: op.resultCount
            )
            newOps.append(fp16Dot)
            typeMap[fp16ResultName] = fp16ResultType

            newOps.append(HLOOperation(
                result: op.result,                  // original name preserved
                kind: .convert,
                operands: [fp16ResultName],
                resultType: op.resultType
            ))
        }

        transformCount += 1
    }

    if transformCount == 0 {
        return function
    }

    return HLOFunction(
        name: function.name,
        isPrivate: function.isPrivate,
        inputs: function.inputs,
        outputTypes: function.outputTypes,
        operations: newOps,
        returnValues: function.returnValues
    )
}

/// Returns the compute-to-memory ratio M*N*K / (M*K + K*N + M*N) for a
/// dot/dot_general op, or nil if it cannot be determined (missing operand
/// types, malformed shapes). Higher ratio = more compute-bound = fp16
/// convert overhead is amortized.
private func computeSurfaceRatio(op: HLOOperation, typeMap: [String: TensorType]) -> Double? {
    guard op.operands.count == 2,
          let lhsType = typeMap[op.operands[0]],
          let rhsType = typeMap[op.operands[1]] else {
        return nil
    }
    let lhsShape = lhsType.shape
    let rhsShape = rhsType.shape
    let outShape = op.resultType.shape

    let m, k, n: Int

    if op.kind == .dot {
        // Standard matmul: lhs = [M, K], rhs = [K, N], out = [M, N]
        guard lhsShape.count == 2, rhsShape.count == 2, outShape.count == 2 else {
            return nil
        }
        m = lhsShape[0]
        k = lhsShape[1]
        n = rhsShape[1]
    } else if op.kind == .dotGeneral, let dimNumbers = op.attributes.dotDimensionNumbers {
        // Generic dot: derive M, K, N from dim numbers.
        let lhsBatching = Set(dimNumbers.lhsBatchingDimensions)
        let lhsContracting = Set(dimNumbers.lhsContractingDimensions)
        let rhsBatching = Set(dimNumbers.rhsBatchingDimensions)
        let rhsContracting = Set(dimNumbers.rhsContractingDimensions)

        var mProd = 1
        for (i, dim) in lhsShape.enumerated() where !lhsBatching.contains(i) && !lhsContracting.contains(i) {
            mProd *= dim
        }
        var kProd = 1
        for i in dimNumbers.lhsContractingDimensions {
            kProd *= lhsShape[i]
        }
        var nProd = 1
        for (i, dim) in rhsShape.enumerated() where !rhsBatching.contains(i) && !rhsContracting.contains(i) {
            nProd *= dim
        }
        m = mProd
        k = kProd
        n = nProd
    } else {
        return nil
    }

    let surface = m * k + k * n + m * n
    guard surface > 0 else { return nil }
    return Double(m * n * k) / Double(surface)
}

/// Extracts (M, N, K, batchSize) for a dot / dot_general op, or nil if it
/// cannot be determined. Used by the output-convert fusion gate to decide
/// whether the dot will go through the MPP path at codegen time.
private func matmulDims(
    op: HLOOperation,
    typeMap: [String: TensorType]
) -> (M: Int, N: Int, K: Int, batchSize: Int)? {
    guard op.operands.count == 2,
          let lhsType = typeMap[op.operands[0]],
          let rhsType = typeMap[op.operands[1]] else {
        return nil
    }
    let lhsShape = lhsType.shape
    let rhsShape = rhsType.shape

    if op.kind == .dot {
        guard lhsShape.count == 2, rhsShape.count == 2 else { return nil }
        return (M: lhsShape[0], N: rhsShape[1], K: lhsShape[1], batchSize: 1)
    }

    guard op.kind == .dotGeneral, let dimNumbers = op.attributes.dotDimensionNumbers else {
        return nil
    }
    let lhsBatching = Set(dimNumbers.lhsBatchingDimensions)
    let lhsContracting = Set(dimNumbers.lhsContractingDimensions)
    let rhsBatching = Set(dimNumbers.rhsBatchingDimensions)
    let rhsContracting = Set(dimNumbers.rhsContractingDimensions)

    var m = 1
    for (i, d) in lhsShape.enumerated() where !lhsBatching.contains(i) && !lhsContracting.contains(i) { m *= d }
    var k = 1
    for i in dimNumbers.lhsContractingDimensions { k *= lhsShape[i] }
    var n = 1
    for (i, d) in rhsShape.enumerated() where !rhsBatching.contains(i) && !rhsContracting.contains(i) { n *= d }
    let batch = lhsBatching.isEmpty
        ? (lhsShape.count > 2 ? lhsShape.dropLast(2).reduce(1, *) : 1)
        : dimNumbers.lhsBatchingDimensions.reduce(1) { $0 * lhsShape[$1] }
    return (M: m, N: n, K: k, batchSize: batch)
}
