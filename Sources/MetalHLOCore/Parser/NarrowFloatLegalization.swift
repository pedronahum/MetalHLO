// NarrowFloatLegalization.swift
// MetalHLOCore
//
// Metal has no 8- or 4-bit float arithmetic. This pass rewrites a parsed module
// so no value has a narrow float type (f8E4M3FN, f4E2M1FN): such values become
// f32 tensors holding exactly representable values.
//
//   convert X -> f8E4M3FN        round to the f8 grid in f32 (ties to even; NaN
//                                past 448, as XLA's non-saturating conversion)
//   convert f8/f4 -> X           a convert from f32
//   bitcast_convert ui8 -> f8    decode the byte's bit fields
//   bitcast_convert ui8[..] -> f4[.., 2]  decode both nibbles, low first
//   constants                    values rounded to the narrow grid
//   everything else              retyped to f32
//
// Main's narrow-float arguments and results stay narrow at the API: Executable
// decodes such input buffers to f32 and encodes declared narrow outputs back
// (see NarrowFloat for the host codecs).

import Foundation

public enum NarrowFloatLegalization {

    static func isNarrow(_ t: ElementType) -> Bool { t == .float8E4M3FN || t == .float4E2M1FN }

    static func widened(_ t: TensorType) -> TensorType {
        isNarrow(t.elementType) ? TensorType(shape: t.shape, elementType: .float32) : t
    }

    /// The module with no narrow-float values, and the declared narrow element
    /// type of each of main's results (nil where the result is not narrow).
    public static func apply(_ module: HLOModule) throws -> (module: HLOModule, narrowOutputs: [ElementType?]) {
        let usesNarrow = module.functions.contains { f in
            f.inputs.contains { isNarrow($0.type.elementType) } || f.outputTypes.contains { isNarrow($0.elementType) }
                || containsNarrow(f.operations)
        }
        let narrowOutputs = module.function.outputTypes.map { isNarrow($0.elementType) ? $0.elementType : nil }
        guard usesNarrow else { return (module, narrowOutputs) }
        let functions = try module.functions.map { f -> HLOFunction in
            var types: [String: TensorType] = [:]
            for arg in f.inputs { types[arg.name] = arg.type }
            return HLOFunction(
                name: f.name, isPrivate: f.isPrivate,
                inputs: f.inputs.map { HLOArgument(name: $0.name, type: widened($0.type)) },
                outputTypes: f.outputTypes.map(widened),
                operations: try legalize(f.operations, types: &types),
                returnValues: f.returnValues)
        }
        return (HLOModule(name: module.name, functions: functions), narrowOutputs)
    }

    private static func containsNarrow(_ ops: [HLOOperation]) -> Bool {
        ops.contains { op in
            if isNarrow(op.resultType.elementType) { return true }
            if let w = op.attributes.whileRegions, containsNarrow(w.condition) || containsNarrow(w.body) { return true }
            if let r = op.attributes.ifRegions,
               containsNarrow(r.thenBranch) || (r.elseBranch.map(containsNarrow) ?? false) { return true }
            return false
        }
    }

    private static func containsNarrow(_ region: Region) -> Bool {
        region.arguments.contains { isNarrow($0.type.elementType) } || containsNarrow(region.operations)
    }

    /// `types` maps every value seen so far to its declared (pre-legalization) type.
    private static func legalize(_ ops: [HLOOperation], types: inout [String: TensorType]) throws -> [HLOOperation] {
        var out: [HLOOperation] = []
        for op in ops {
            let operandType = op.operands.first.flatMap { types[$0] }
            types[op.result] = op.resultType
            let result = op.resultType.elementType
            if op.kind == .convert, isNarrow(result), let source = op.operands.first {
                guard result == .float8E4M3FN else {
                    throw CompilationError.unsupportedOperation("convert to \(result.rawValue)")
                }
                if let operandType, isNarrow(operandType.elementType) {
                    out.append(retyped(op, regions: nil))  // already on a narrow grid
                } else {
                    var b = NarrowBuilder(shape: op.resultType.shape, prefix: op.result)
                    let x = operandType?.elementType == .float32 ? source : b.convert(source, to: .float32)
                    _ = b.roundToF8E4M3FN(x)
                    out.append(contentsOf: b.finish(as: op.result))
                }
            } else if op.kind == .bitcastConvert, isNarrow(result), let source = op.operands.first {
                var b = NarrowBuilder(shape: operandType?.shape ?? op.resultType.shape, prefix: op.result)
                let bytes = b.convert(source, to: .int32)
                if result == .float8E4M3FN {
                    _ = b.decodeF8E4M3FN(bytes)
                } else {
                    _ = b.decodeF4E2M1FNPairs(bytes, resultShape: op.resultType.shape)
                }
                out.append(contentsOf: b.finish(as: op.result))
            } else if op.kind == .bitcastConvert, let operandType, isNarrow(operandType.elementType) {
                throw CompilationError.unsupportedOperation("bitcast_convert from \(operandType.elementType.rawValue)")
            } else {
                var regionsChanged: HLOAttributes? = nil
                if let w = op.attributes.whileRegions {
                    var a = op.attributes
                    a.whileRegions = WhileRegions(condition: try legalize(w.condition, types: &types),
                                                  body: try legalize(w.body, types: &types))
                    regionsChanged = a
                }
                if let r = op.attributes.ifRegions {
                    var a = regionsChanged ?? op.attributes
                    a.ifRegions = IfRegions(thenBranch: try legalize(r.thenBranch, types: &types),
                                            elseBranch: try r.elseBranch.map { try legalize($0, types: &types) })
                    regionsChanged = a
                }
                out.append(retyped(op, regions: regionsChanged))
            }
        }
        return out
    }

    private static func legalize(_ region: Region, types: inout [String: TensorType]) throws -> Region {
        for arg in region.arguments { types[arg.name] = arg.type }
        return Region(arguments: region.arguments.map { RegionArgument(name: $0.name, type: widened($0.type)) },
                      operations: try legalize(region.operations, types: &types),
                      returnValues: region.returnValues)
    }

    /// `op` with narrow types widened and narrow constants rounded to their grid.
    private static func retyped(_ op: HLOOperation, regions: HLOAttributes?) -> HLOOperation {
        var a = regions ?? op.attributes
        a.inputElementTypes = a.inputElementTypes?.map { isNarrow($0) ? .float32 : $0 }
        if let c = a.constantValue, isNarrow(op.resultType.elementType) {
            let narrow = op.resultType.elementType
            let grid: (Double) -> Double = { v in
                narrow == .float8E4M3FN
                    ? Double(NarrowFloat.roundToF8E4M3FN(Float(v)))
                    : Double(NarrowFloat.decodeF4E2M1FN(NarrowFloat.encodeF4E2M1FN(Float(v))))
            }
            let t = widened(op.resultType)
            switch c {
            case .scalar(let v): a.constantValue = .scalar(grid(v))
            case .splat(let v, _): a.constantValue = .splat(grid(v), t)
            case .dense(let vs, _): a.constantValue = .dense(vs.map(grid), t)
            default: break
            }
        }
        return HLOOperation(result: op.result, kind: op.kind, operands: op.operands,
                            resultType: widened(op.resultType), attributes: a, resultCount: op.resultCount)
    }
}

/// Emits elementwise ops of one shape for the narrow-float expansions; each
/// op names its element type.
struct NarrowBuilder {
    private(set) var ops: [HLOOperation] = []
    var shape: [Int]
    let prefix: String
    private var counter = 0
    private var typeOf: [String: ElementType] = [:]

    init(shape: [Int], prefix: String) {
        self.shape = shape
        self.prefix = prefix
    }

    private mutating func emit(_ kind: HLOOpKind, _ operands: [String], _ type: ElementType,
                               shape s: [Int]? = nil, _ configure: (inout HLOAttributes) -> Void = { _ in }) -> String {
        counter += 1
        let name = "\(prefix)_narrow\(counter)"
        var a = HLOAttributes()
        a.inputElementTypes = operands.map { typeOf[$0] ?? .float32 }
        configure(&a)
        ops.append(HLOOperation(result: name, kind: kind, operands: operands,
                                resultType: TensorType(shape: s ?? shape, elementType: type), attributes: a))
        typeOf[name] = type
        return name
    }

    mutating func const(_ v: Double, _ type: ElementType) -> String {
        let constantType = TensorType(shape: shape, elementType: type)
        return emit(.constant, [], type) { $0.constantValue = .splat(v, constantType) }
    }

    mutating func convert(_ x: String, to type: ElementType) -> String { emit(.convert, [x], type) }
    mutating func bitcast(_ x: String, to type: ElementType) -> String { emit(.bitcastConvert, [x], type) }

    private func typeOrFloat(_ x: String) -> ElementType { typeOf[x] ?? .float32 }

    mutating func op(_ kind: HLOOpKind, _ x: String, _ y: String) -> String { emit(kind, [x, y], typeOrFloat(x)) }
    mutating func unary(_ kind: HLOOpKind, _ x: String) -> String { emit(kind, [x], typeOrFloat(x)) }

    mutating func compare(_ d: ComparisonDirection, _ x: String, _ y: String) -> String {
        emit(.compare, [x, y], .int1) { $0.comparisonDirection = d }
    }

    mutating func select(_ p: String, _ x: String, _ y: String) -> String { emit(.select, [p, x, y], typeOrFloat(x)) }

    /// 2^e for an i32 exponent tensor e (normal range), as f32.
    mutating func pow2(_ e: String) -> String {
        let biased = op(.add, e, const(127, .int32))
        return bitcast(op(.shiftLeft, biased, const(23, .int32)), to: .float32)
    }

    /// f32 x rounded to the f8E4M3FN grid: quantum 2^(max(e, −6) − 3), ties to
    /// even, NaN past 448 (and for NaN/±∞).
    mutating func roundToF8E4M3FN(_ x: String) -> String {
        let a = unary(.abs, x)
        let bits = bitcast(a, to: .int32)
        let e = op(.subtract, op(.shiftRightLogical, bits, const(23, .int32)), const(127, .int32))
        let quantum = pow2(op(.subtract, op(.maximum, e, const(-6, .int32)), const(3, .int32)))
        let r = op(.multiply, unary(.roundNearestEven, op(.divide, a, quantum)), quantum)
        let finite = select(compare(.gt, r, const(448, .float32)), const(.nan, .float32), r)
        return select(compare(.lt, x, const(0, .float32)), unary(.negate, finite), finite)
    }

    /// i32 bytes (0…255) decoded as f8E4M3FN, as f32.
    mutating func decodeF8E4M3FN(_ b: String) -> String {
        let sign = op(.shiftRightLogical, b, const(7, .int32))
        let e = op(.and, op(.shiftRightLogical, b, const(3, .int32)), const(15, .int32))
        let m = op(.and, b, const(7, .int32))
        let mf = convert(m, to: .float32)
        let normal = op(.multiply, op(.add, const(1, .float32), op(.divide, mf, const(8, .float32))),
                        pow2(op(.subtract, e, const(7, .int32))))
        let subnormal = op(.multiply, mf, const(0x1p-9, .float32))
        var v = select(compare(.eq, e, const(0, .int32)), subnormal, normal)
        let nan = op(.and, compare(.eq, e, const(15, .int32)), compare(.eq, m, const(7, .int32)))
        v = select(nan, const(.nan, .float32), v)
        return select(compare(.eq, sign, const(1, .int32)), unary(.negate, v), v)
    }

    /// i32 nibbles (0…15) decoded as f4E2M1FN, as f32.
    mutating func decodeF4E2M1FN(_ n: String) -> String {
        let sign = op(.shiftRightLogical, n, const(3, .int32))
        let e = op(.and, op(.shiftRightLogical, n, const(1, .int32)), const(3, .int32))
        let mf = convert(op(.and, n, const(1, .int32)), to: .float32)
        let normal = op(.multiply, op(.add, const(1, .float32), op(.divide, mf, const(2, .float32))),
                        pow2(op(.subtract, e, const(1, .int32))))
        let subnormal = op(.multiply, mf, const(0.5, .float32))
        let v = select(compare(.eq, e, const(0, .int32)), subnormal, normal)
        return select(compare(.eq, sign, const(1, .int32)), unary(.negate, v), v)
    }

    /// Bytes [..] → f4E2M1FN pairs [.., 2]: element 0 is the low nibble.
    mutating func decodeF4E2M1FNPairs(_ b: String, resultShape: [Int]) -> String {
        let low = decodeF4E2M1FN(op(.and, b, const(15, .int32)))
        let high = decodeF4E2M1FN(op(.shiftRightLogical, b, const(4, .int32)))
        let column = shape + [1]
        let lowColumn = emit(.reshape, [low], .float32, shape: column)
        let highColumn = emit(.reshape, [high], .float32, shape: column)
        let axis = column.count - 1
        return emit(.concatenate, [lowColumn, highColumn], .float32, shape: resultShape) { $0.axis = axis }
    }

    /// The ops, with the last one renamed to `name`.
    func finish(as name: String) -> [HLOOperation] {
        var result = ops
        let last = result.removeLast()
        result.append(HLOOperation(result: name, kind: last.kind, operands: last.operands,
                                   resultType: last.resultType, attributes: last.attributes))
        return result
    }
}
