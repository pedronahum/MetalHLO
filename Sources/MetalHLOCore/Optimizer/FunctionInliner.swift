// FunctionInliner.swift
// MetalHLOCore
//
// Flattens func.call operations into the entry function by inlining the
// callee bodies. The fast (IntegratedExecutor) compile path operates on a
// single function and has no notion of cross-function calls, so any `call`
// left in the graph silently produces no output.
//
// JAX 0.10 lowers a number of composite ops through small private helper
// functions reached via `call` — e.g. jnp.cumsum lowers to
//   main -> call @cumsum -> call @cumsum_0(reduce_window)
// Inlining (recursively, so nested calls are flattened too) lets the rest of
// the pipeline see one self-contained function.

/// Inlines `call` operations into a module's entry function.
public struct FunctionInliner {

    /// Returns a copy of `module` whose entry function contains no `call`
    /// operations — every call has been replaced by a renamed copy of the
    /// callee's body. Private helper functions are dropped from the result.
    ///
    /// If the entry function has no calls (the common case), the module is
    /// returned unchanged so this is effectively free.
    public static func inline(_ module: HLOModule) -> HLOModule {
        let entry = module.function
        // A `call` may also sit inside a while body (the threefry pattern:
        // entry -> @threefry2x32 holds a while whose body calls @closed_call).
        // Such nested calls aren't flattened here (regions are opaque to this
        // pass), but the helper functions they reach must survive so a later
        // pass (the while unroller, then a second inline) can still resolve
        // them — hence we retain the module's other functions in the result.
        guard entry.operations.contains(where: { $0.kind == .call }) else {
            return module
        }

        var counter = 0
        var topRenames: [String: String] = [:]
        let inlinedOps = inlineOperations(
            entry.operations,
            module: module,
            valueRenames: [:],
            prefix: "",
            counter: &counter,
            outRenames: &topRenames
        )

        // The entry's return values may name a call result (e.g. main returns
        // %0 = call @cumsum(...)); rewrite them through the rename map so they
        // point at the inlined producer.
        let newReturns = entry.returnValues.map { topRenames[$0] ?? $0 }

        let newEntry = HLOFunction(
            name: entry.name,
            isPrivate: entry.isPrivate,
            inputs: entry.inputs,
            outputTypes: entry.outputTypes,
            operations: inlinedOps,
            returnValues: newReturns
        )

        // Retain non-entry functions so calls left inside while bodies remain
        // resolvable for a subsequent inline pass.
        var functions = module.functions
        if let idx = functions.firstIndex(where: { $0.name == entry.name }) {
            functions[idx] = newEntry
        } else {
            functions.insert(newEntry, at: 0)
        }
        return HLOModule(name: module.name, functions: functions)
    }

    /// Recursively rewrites a list of operations, inlining any `call` ops.
    ///
    /// - Parameters:
    ///   - ops: operations to process.
    ///   - valueRenames: map from a value name as written in `ops` to the name
    ///     it should take in the flattened output (used to remap a callee's
    ///     argument/SSA names onto the caller's values + fresh names).
    ///   - prefix: unique prefix applied to freshly minted SSA names so repeat
    ///     inlines of the same function don't collide.
    ///   - counter: monotonically increasing source of unique prefixes.
    private static func inlineOperations(
        _ ops: [HLOOperation],
        module: HLOModule,
        valueRenames: [String: String],
        prefix: String,
        counter: inout Int,
        outRenames: inout [String: String]
    ) -> [HLOOperation] {
        var renames = valueRenames
        var result: [HLOOperation] = []
        defer { outRenames = renames }

        func remap(_ name: String) -> String {
            renames[name] ?? name
        }

        for op in ops {
            if op.kind == .call, let target = op.attributes.functionCallTarget,
               let callee = module.getFunction(named: target) {
                // Fresh prefix for this call site.
                let sitePrefix = "\(prefix)_inl\(counter)_"
                counter += 1

                // Map callee formal args -> caller actual operands (already
                // remapped into the current scope).
                var calleeRenames: [String: String] = [:]
                for (i, arg) in callee.inputs.enumerated() where i < op.operands.count {
                    calleeRenames[arg.name] = remap(op.operands[i])
                }
                // Every value the callee defines gets a unique prefixed name.
                for bodyOp in callee.operations {
                    calleeRenames[bodyOp.result] = sitePrefix + bodyOp.result
                    if bodyOp.resultCount > 1 {
                        for k in 0..<bodyOp.resultCount {
                            let multi = "\(bodyOp.result).\(k)"
                            calleeRenames[multi] = sitePrefix + multi
                        }
                    }
                }

                // Recursively inline the callee body under the new scope.
                var nestedRenames: [String: String] = [:]
                let inlinedBody = inlineOperations(
                    callee.operations,
                    module: module,
                    valueRenames: calleeRenames,
                    prefix: sitePrefix,
                    counter: &counter,
                    outRenames: &nestedRenames
                )
                result.append(contentsOf: inlinedBody)

                // Wire the callee's returns to the call's result names so later
                // consumers (which reference %call or %call.k) resolve. A
                // returned value may itself be a (now-inlined) nested call
                // result, so consult the nested scope's final renames first.
                func resolveReturn(_ rv: String) -> String {
                    nestedRenames[rv] ?? calleeRenames[rv] ?? rv
                }
                if op.resultCount > 1 {
                    for k in 0..<op.resultCount {
                        if k < callee.returnValues.count {
                            renames["\(op.result).\(k)"] = resolveReturn(callee.returnValues[k])
                        }
                    }
                } else if let rv = callee.returnValues.first {
                    renames[op.result] = resolveReturn(rv)
                }
                continue
            }

            // Non-call op: remap operands (and its own result if it was renamed
            // to flow a callee value through).
            let newOperands = op.operands.map { remap($0) }
            let newResult = remap(op.result)
            result.append(HLOOperation(
                result: newResult,
                kind: op.kind,
                operands: newOperands,
                resultType: op.resultType,
                attributes: op.attributes,
                resultCount: op.resultCount
            ))
        }

        return result
    }
}
