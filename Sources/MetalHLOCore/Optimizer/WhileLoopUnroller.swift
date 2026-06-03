// WhileLoopUnroller.swift
// MetalHLOCore
//
// Unrolls statically-bounded counted `while` loops into straight-line ops.
//
// The fast (IntegratedExecutor) compile path has no runtime control flow — it
// lowers a single flat list of ops to Metal kernels. JAX, however, lowers some
// composites through bounded counted loops. The most important is
// jax.random.bits / jax.random.{uniform,normal,...}: every draw runs the
// threefry2x32 block, which JAX emits as a `while` with a static trip count of
// 5 around a `func.call @closed_call` (one Threefry round group + key
// injection per iteration). Without unrolling, the loop is opaque to the fast
// path and the RNG output is wrong (or the program fails outright).
//
// This pass recognizes the counted-loop shape JAX emits:
//   %r:N = while(%i = c0, %a = ..., ...) : ...
//     cond { compare LT, %i, cBound  ->  i1 }
//     do   { ...body...; %i' = add %i, cStep; return %i', ... }
// where the induction variable %i starts at a constant, the condition tests it
// against a constant bound with LT, and the body advances it by a constant
// step. The trip count is then known at compile time and the loop is replaced
// by `tripCount` copies of the body, each with freshly renamed SSA values and
// the loop-carried values threaded between iterations. Body `func.call`s are
// left in place; the subsequent FunctionInliner flattens them.
//
// Loops that don't match this shape (data-dependent bounds, non-LT tests,
// non-affine induction) are left untouched — they fall through to the MPSGraph
// path which has native control flow.

/// Unrolls statically-bounded counted `while` loops in a module's entry
/// function so the fast codegen path sees only straight-line ops.
public struct WhileLoopUnroller {

    /// A conservative cap on the number of unrolled iterations. Threefry uses
    /// 5; anything wildly larger is almost certainly not the counted-loop
    /// pattern we want to flatten (and would explode the op count), so we bail
    /// and leave such loops for the MPSGraph path.
    private static let maxTripCount = 256

    /// Returns a copy of `module` whose entry function has every recognizable
    /// counted `while` loop unrolled. Private helper functions are preserved so
    /// a later inlining pass can still resolve calls left inside unrolled
    /// bodies. No-op (returns the module unchanged) when the entry has no
    /// `while` ops.
    public static func unroll(_ module: HLOModule) -> HLOModule {
        let entry = module.function
        guard entry.operations.contains(where: { $0.kind == .whileOp }) else {
            return module
        }

        var counter = 0
        let newOps = unrollOperations(entry.operations, counter: &counter)

        let newEntry = HLOFunction(
            name: entry.name,
            isPrivate: entry.isPrivate,
            inputs: entry.inputs,
            outputTypes: entry.outputTypes,
            operations: newOps,
            returnValues: entry.returnValues
        )

        // Keep the other functions (e.g. @closed_call) intact for the inliner.
        var functions = module.functions
        if let idx = functions.firstIndex(where: { $0.name == entry.name }) {
            functions[idx] = newEntry
        } else {
            functions.insert(newEntry, at: 0)
        }
        return HLOModule(name: module.name, functions: functions)
    }

    /// Rewrites a flat op list, replacing each recognizable counted `while`
    /// with its unrolled body. Operations are emitted in order; an unrolled
    /// loop expands in place. Renames carry forward via `renames` so that
    /// later ops referencing the while's results resolve to the final
    /// loop-carried values.
    private static func unrollOperations(
        _ ops: [HLOOperation],
        counter: inout Int
    ) -> [HLOOperation] {
        var renames: [String: String] = [:]
        var result: [HLOOperation] = []

        // Index of constants defined in this op list, so the induction var's
        // initial value (a while operand) can be read by name.
        var enclosingConstants: [String: Double] = [:]
        for op in ops where op.kind == .constant {
            if let v = scalarIntConstantValue(op.attributes.constantValue) {
                enclosingConstants[op.result] = v
            }
        }

        func remap(_ name: String) -> String { renames[name] ?? name }

        for op in ops {
            guard op.kind == .whileOp,
                  let regions = op.attributes.whileRegions,
                  let trip = staticTripCount(
                      condition: regions.condition,
                      body: regions.body,
                      whileOperands: op.operands,
                      enclosingConstants: enclosingConstants
                  )
            else {
                // Non-loop (or non-counted loop): just remap operands so any
                // upstream unrolled-while results thread through.
                result.append(HLOOperation(
                    result: op.result,
                    kind: op.kind,
                    operands: op.operands.map { remap($0) },
                    resultType: op.resultType,
                    attributes: op.attributes,
                    resultCount: op.resultCount
                ))
                continue
            }

            let body = regions.body
            let bodyArgs = body.arguments.map { $0.name }

            // Initial loop-carried values are the while operands, remapped into
            // the current scope.
            var carried: [String] = op.operands.map { remap($0) }

            for _ in 0..<trip {
                // Fresh prefix for this iteration so repeat copies of the body
                // don't collide.
                let prefix = "_wu\(counter)_"
                counter += 1

                // Map body block args -> current carried values, and every
                // value the body defines -> a unique prefixed name.
                var local: [String: String] = [:]
                for (i, arg) in bodyArgs.enumerated() where i < carried.count {
                    local[arg] = carried[i]
                }
                for bodyOp in body.operations {
                    local[bodyOp.result] = prefix + bodyOp.result
                    if bodyOp.resultCount > 1 {
                        for k in 0..<bodyOp.resultCount {
                            let multi = "\(bodyOp.result).\(k)"
                            local[multi] = prefix + multi
                        }
                    }
                }
                func remapLocal(_ name: String) -> String { local[name] ?? name }

                // Emit the body's ops with renamed operands/results. Calls are
                // kept (the inliner flattens them after this pass).
                for bodyOp in body.operations {
                    result.append(HLOOperation(
                        result: remapLocal(bodyOp.result),
                        kind: bodyOp.kind,
                        operands: bodyOp.operands.map { remapLocal($0) },
                        resultType: bodyOp.resultType,
                        attributes: bodyOp.attributes,
                        resultCount: bodyOp.resultCount
                    ))
                }

                // The body's return values become the carried values for the
                // next iteration.
                carried = body.returnValues.map { remapLocal($0) }
            }

            // Wire the while's results (%w, %w.0, %w.1, ...) to the final
            // carried values so later consumers resolve.
            if op.resultCount > 1 {
                for k in 0..<op.resultCount where k < carried.count {
                    renames["\(op.result).\(k)"] = carried[k]
                }
            } else if let first = carried.first {
                renames[op.result] = first
            }
        }

        return result
    }

    /// Computes the static trip count of a counted loop, or nil if the loop
    /// doesn't match the recognizable shape.
    ///
    /// Recognized shape:
    ///   cond: `%pred = compare LT, %iv, %bound` where %iv is a body block arg
    ///         and %bound is a scalar integer constant (in the cond region).
    ///   body: the same induction arg %iv is returned as `add %iv, %step` with
    ///         %step a scalar integer constant, and %iv starts at a scalar
    ///         integer constant init (the while operand for that arg slot).
    private static func staticTripCount(
        condition: Region,
        body: Region,
        whileOperands: [String],
        enclosingConstants: [String: Double]
    ) -> Int? {
        // The cond region returns a single i1 from a compare LT.
        guard condition.returnValues.count == 1,
              let predName = condition.returnValues.first
        else { return nil }

        guard let cmp = condition.operations.first(where: { $0.result == predName }),
              cmp.kind == .compare,
              cmp.attributes.comparisonDirection == .lt,
              cmp.operands.count == 2
        else { return nil }

        // Resolve the induction variable (a block arg) and the bound constant.
        let condArgNames = Set(condition.arguments.map { $0.name })
        let lhs = cmp.operands[0]
        let rhs = cmp.operands[1]

        // The bound / step constants are normally defined inside the cond /
        // body regions, but a prior text-level inliner can hoist them into the
        // enclosing function (e.g. JAX's searchsorted lowering after the PJRT
        // text transforms move the `dense<3>` bound and `dense<1>` step ahead of
        // the while). Resolve a constant by name from the region first, then
        // fall back to the enclosing-scope constant table so a hoisted literal
        // is still recognized.
        func constant(named name: String, in region: Region) -> Double? {
            scalarIntConstant(named: name, in: region)
                ?? enclosingConstants[name]
        }

        // Find which side is the block-arg induction var and which is the bound.
        let ivArg: String
        let bound: Double
        if condArgNames.contains(lhs), let b = constant(named: rhs, in: condition) {
            ivArg = lhs
            bound = b
        } else {
            return nil
        }

        // The induction var's slot index among block args.
        guard let ivIndex = condition.arguments.firstIndex(where: { $0.name == ivArg })
        else { return nil }
        // Body must have a matching arg slot.
        guard ivIndex < body.arguments.count else { return nil }
        let bodyIVArg = body.arguments[ivIndex].name

        // The body must return `add bodyIVArg, step` in that slot.
        guard ivIndex < body.returnValues.count else { return nil }
        let returnedIV = body.returnValues[ivIndex]
        guard let inc = body.operations.first(where: { $0.result == returnedIV }),
              inc.kind == .add, inc.operands.count == 2
        else { return nil }

        let step: Double
        if inc.operands[0] == bodyIVArg, let s = constant(named: inc.operands[1], in: body) {
            step = s
        } else if inc.operands[1] == bodyIVArg, let s = constant(named: inc.operands[0], in: body) {
            step = s
        } else {
            return nil
        }
        guard step > 0 else { return nil }

        // Initial value of the induction var = the while operand at ivIndex,
        // which must be a scalar-int constant defined in the enclosing scope.
        // The iteration count for a `LT bound` loop starting at `init` stepping
        // by `step` is ceil((bound - init) / step).
        guard ivIndex < whileOperands.count,
              let initValue = enclosingConstants[whileOperands[ivIndex]]
        else { return nil }

        let count = Int(((bound - initValue) + step - 1) / step)
        guard count >= 0, count <= maxTripCount else { return nil }
        return count
    }

    /// Reads a scalar integer constant value defined inside `region` by name,
    /// or nil if `name` is not a scalar-integer constant local to the region.
    private static func scalarIntConstant(named name: String, in region: Region) -> Double? {
        guard let def = region.operations.first(where: { $0.result == name }),
              def.kind == .constant
        else { return nil }
        return scalarIntConstantValue(def.attributes.constantValue)
    }

    /// Extracts a scalar value from a (scalar / 1-element) constant, or nil.
    private static func scalarIntConstantValue(_ cv: ConstantValue?) -> Double? {
        guard let cv = cv else { return nil }
        switch cv {
        case .scalar(let v):
            return v
        case .splat(let v, let t) where t.shape.isEmpty || t.shape == [1]:
            return v
        case .dense(let vals, let t) where (t.shape.isEmpty || t.shape == [1]) && vals.count == 1:
            return vals[0]
        default:
            return nil
        }
    }
}
