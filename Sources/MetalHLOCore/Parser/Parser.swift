// Parser.swift
// MetalHLOCore
//
// MLIR parser for StableHLO modules.

/// Parses MLIR text into an HLOModule.
public final class Parser {

    // MARK: - Properties

    private let lexer: Lexer
    private var currentToken: Token
    private var peekedToken: Token?

    // MARK: - Initialization

    /// Creates a new parser for the given source text.
    ///
    /// - Parameter source: The MLIR source text.
    public init(source: String) {
        self.lexer = Lexer(source: source)
        self.currentToken = lexer.nextToken()
    }

    // MARK: - Public API

    /// Parses the source and returns an HLOModule.
    ///
    /// - Throws: `ParseError` if parsing fails.
    /// - Returns: The parsed HLOModule.
    public func parse() throws -> HLOModule {
        try parseModule()
    }

    // MARK: - Module Parsing

    private func parseModule() throws -> HLOModule {
        // Skip any leading newlines
        skipNewlines()

        // Expect: module @name { or module "name" { or module {
        try expect(.keyword(.module))
        let moduleName: String
        if check(.atIdentifier) {
            moduleName = try parseAtIdentifier()
        } else if case .string(let name) = currentToken.kind {
            moduleName = name
            advance()
        } else {
            moduleName = "module"
        }
        // Optional module attribute dict: `attributes { ... }` (JAX emits
        // mhlo.num_partitions/num_replicas here). Skip the balanced braces.
        //
        // While skipping, watch for `mhlo.num_partitions` / `mhlo.num_replicas`.
        // MetalHLO is single-device by design: pjit-with-mesh / shard_map /
        // nn.partitioning are supported only when they collapse to a single
        // device (num_partitions == num_replicas == 1, which is what JAX emits
        // for a 1-device mesh — the sharding annotations are then no-ops and we
        // ignore them safely). A module that XLA partitioned across multiple
        // devices (num_partitions > 1 or num_replicas > 1) would otherwise be
        // executed as a full, unpartitioned computation on one device — a
        // silently wrong result. Reject it early and loudly instead.
        if checkIdentifier("attributes") {
            advance()
            if check(.leftBrace) {
                advance()
                var depth = 1
                // Name of the partition/replica attribute whose integer value we
                // are about to read (`= N : i32`), or nil when not inside one.
                var pendingShardingAttr: String?
                let attrStart = currentToken.location
                while depth > 0 && !check(.eof) {
                    if check(.leftBrace) { depth += 1 }
                    else if check(.rightBrace) { depth -= 1 }
                    else if currentToken.kind == .identifier
                                && (currentToken.text == "mhlo.num_partitions"
                                    || currentToken.text == "mhlo.num_replicas") {
                        pendingShardingAttr = currentToken.text
                    } else if let attr = pendingShardingAttr,
                              case .integer(let count) = currentToken.kind {
                        // First integer after the attribute name is its value.
                        if count > 1 {
                            let kind = attr == "mhlo.num_partitions"
                                ? "partitions" : "replicas"
                            throw ParseError.unsupportedFeature(
                                "module declares \(attr) = \(count); MetalHLO is "
                                + "single-device only and cannot execute a program "
                                + "partitioned across \(count) \(kind). Run this "
                                + "computation on a single device (a 1-device mesh, "
                                + "or without pjit in/out_shardings / shard_map over "
                                + "multiple devices).",
                                location: attrStart)
                        }
                        pendingShardingAttr = nil
                    }
                    advance()
                }
            }
            skipNewlines()
        }
        try expect(.leftBrace)
        skipNewlines()

        // Parse all functions in the module
        var functions: [HLOFunction] = []
        while !check(.rightBrace) && !check(.eof) {
            if checkIdentifier("func.func") || checkIdentifier("func") {
                let function = try parseFunction()
                functions.append(function)
            } else {
                // Skip unknown top-level declarations (e.g., sdy.mesh)
                advance()
            }
            skipNewlines()
        }

        // Expect: }
        try expect(.rightBrace)

        return HLOModule(name: moduleName, functions: functions)
    }

    // MARK: - Function Parsing

    private func parseFunction() throws -> HLOFunction {
        // Expect: func.func [private] @name(args) -> (return_types) {
        try expectIdentifier("func.func")

        // Check for visibility: `private` or `public` (JAX marks the entry
        // `public`). Only `private` affects how we treat the function.
        let isPrivate = checkIdentifier("private")
        if isPrivate || checkIdentifier("public") { advance() }

        let funcName = try parseAtIdentifier()
        let inputs = try parseFunctionArguments()
        // Return types are optional — some functions (e.g., scatter computations)
        // have no explicit return type: func.func @name(...) { ... }
        let outputTypes: [TensorType]
        if check(.arrow) {
            advance()
            outputTypes = try parseReturnTypes()
        } else {
            outputTypes = []
        }
        try expect(.leftBrace)
        skipNewlines()

        // Parse operations
        var operations: [HLOOperation] = []
        var returnValues: [String] = []

        // Tuple elimination: track tuple compositions and value aliases
        var tupleOperands: [String: [String]] = [:]
        var valueAliases: [String: String] = [:]
        // Most recent @lapack_sgeqrf_ffi input (matrix A), so the paired
        // @lapack_sorgqr_ffi (which materializes Q) can be re-rooted on A and the
        // whole QR decomposition recomputed host-side from the original matrix.
        var lastGeqrfInput: String? = nil

        // Resolve the tensor type of an already-defined value (last op writing it,
        // or a function input). Used to give each split sort result the dtype of
        // its source operand.
        func typeOf(_ name: String) -> TensorType? {
            if let op = operations.last(where: { $0.result == name }) { return op.resultType }
            if let arg = inputs.first(where: { $0.name == name }) { return arg.type }
            return nil
        }

        while !check(.rightBrace) && !check(.eof) {
            if checkKeyword(.return) {
                returnValues = try parseReturn()
                // Resolve aliases in return values
                returnValues = returnValues.map { resolveAlias($0, aliases: valueAliases) }
                break
            }
            // Result-less side-effecting statement (no `%result =` prefix), e.g.
            //   stablehlo.custom_call @xla_ffi_python_cpu_callback(%x) {...} : (...) -> ()
            // JAX lowers host callbacks (jax.debug.print / debug_callback) to such
            // ops. They have no SSA result and produce no value the program depends
            // on, so dropping them is numerically exact — only the host-side
            // side effect (the print) is lost. parseResultlessStatement consumes
            // the statement and returns true if it was a known-droppable callback;
            // anything else still throws loudly.
            if !check(.percentIdentifier) {
                try parseResultlessStatement()
                skipNewlines()
                continue
            }
            let op = try parseOperation()

            if op.kind == .tuple {
                // Record tuple composition — resolve any aliases in operands
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                tupleOperands[op.result] = resolvedOperands
            } else if op.kind == .optimizationBarrier {
                // Identity barrier: result i aliases operand i. Resolve any
                // aliases in the operands first, then map each result name to
                // its corresponding input. Downstream references use %name#i,
                // which the lexer rewrites to %name.i; the single-result form
                // %name maps to operand 0.
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                for (i, src) in resolvedOperands.enumerated() {
                    valueAliases["\(op.result).\(i)"] = src
                }
                if let first = resolvedOperands.first {
                    valueAliases[op.result] = first
                }
            } else if op.kind == .allReduce {
                // Cross-replica all-reduce. On a single device the replica group
                // has size 1, so the reducer over one value is the identity:
                // alias the result to the operand (like optimization_barrier).
                // The multi-device reduction is performed by the distributed
                // runtime, which intercepts this op before this single-device
                // alias collapse. (Multi-partition modules are still rejected
                // up front, so only single-device programs reach here today.)
                let src = resolveAlias(op.operands.first ?? "", aliases: valueAliases)
                valueAliases[op.result] = src
            } else if op.kind == .reduce && op.resultCount >= 2
                        && op.attributes.argReduceIndexType != nil {
                // Multi-input argmax/argmin reduce: a single 2-result op
                // (max/min value, argmax/argmin index). Split it into a normal
                // value `.reduce` (result %name.0) and a `.reduceArg` index op
                // (result %name.1). Both read operand 0 (the input values);
                // downstream %name#0 / %name#1 refs resolve straight to these
                // results. reductionKind (.max / .min, detected from the
                // reducer body) selects argmax vs argmin.
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                // Original operand order is (in_value, in_index, init_value,
                // init_index). The value reduce matches the single-input reduce
                // codegen which takes (input, init), so feed it the input values
                // and the value init.
                var valueReduceOperands = [resolvedOperands[0]]
                if resolvedOperands.count > 2 { valueReduceOperands.append(resolvedOperands[2]) }
                operations.append(HLOOperation(
                    result: "\(op.result).0",
                    kind: .reduce,
                    operands: valueReduceOperands,
                    resultType: op.resultType,
                    attributes: op.attributes
                ))
                if let indexType = op.attributes.argReduceIndexType {
                    // The index op only needs the input values (operand 0); it
                    // scans the reduce axis and writes the argmax/argmin index.
                    operations.append(HLOOperation(
                        result: "\(op.result).1",
                        kind: .reduceArg,
                        operands: [resolvedOperands[0]],
                        resultType: indexType,
                        attributes: op.attributes
                    ))
                }
            } else if op.kind == .topKValues {
                // top_k is a single 2-result op. Split it into a values op
                // (result %name.0) and an indices op (result %name.1), both
                // reading the same input. Downstream %name#0 / %name#1 refs are
                // rewritten by the lexer to %name.0 / %name.1, so they resolve
                // directly to these results — no alias needed. Each op carries
                // the shared `k` attribute; the values op keeps the values
                // (f32) result type and the indices op takes the stashed i32
                // index type.
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                operations.append(HLOOperation(
                    result: "\(op.result).0",
                    kind: .topKValues,
                    operands: resolvedOperands,
                    resultType: op.resultType,
                    attributes: op.attributes
                ))
                if let indexType = op.attributes.topKIndexType {
                    operations.append(HLOOperation(
                        result: "\(op.result).1",
                        kind: .topKIndices,
                        operands: resolvedOperands,
                        resultType: indexType,
                        attributes: op.attributes
                    ))
                }
            } else if op.kind == .sort && op.resultCount >= 2 {
                // Multi-operand sort (argsort / lexsort): `%r:N = sort(o0..oN-1)`
                // sorts every operand by the comparator on the keys and returns
                // each reordered. Split into N `sortResult` ops — result i ranks
                // by the shared key (operand 0) and reorders operand i. Result i's
                // dtype == operand i's dtype (sort preserves dtype). Downstream
                // %r#i refs resolve to %r.i directly.
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                let n = resolvedOperands.count
                if n >= 2 {
                    // JAX appends an iota index as the LAST operand for arg/lex
                    // sorts; the leading operands are the comparison keys. So the
                    // rank is lexicographic over keys[0..K-1] = operands[0..N-2]
                    // (argsort: K=1; lexsort: K=N-1). Each sortResult carries all
                    // K keys plus its own payload operand.
                    let k = max(1, n - 1)
                    let keys = Array(resolvedOperands.prefix(k))
                    for i in 0..<n {
                        let payload = resolvedOperands[i]
                        operations.append(HLOOperation(
                            result: "\(op.result).\(i)",
                            kind: .sortResult,
                            operands: keys + [payload],
                            resultType: typeOf(payload) ?? op.resultType,
                            attributes: op.attributes
                        ))
                    }
                }
            } else if op.kind == .getTupleElement {
                // Resolve get_tuple_element to the actual tensor
                let tupleRef = resolveAlias(op.operands.first ?? "", aliases: valueAliases)
                let index = op.attributes.tupleIndex ?? 0
                if let members = tupleOperands[tupleRef], index < members.count {
                    valueAliases[op.result] = members[index]
                }
            } else if op.kind == .customCall,
                      let target = op.attributes.callTargetName,
                      target.hasPrefix("lapack_spotrf") || target.hasPrefix("lapack_dpotrf") {
                // Cholesky LAPACK FFI: `@lapack_spotrf_ffi(%a)` returns a 2-tuple
                // (factor, info). JAX's `_cholesky` wrapper masks the factor to
                // NaN when info != 0 and applies the upper/lower triangular mask
                // itself, so the custom_call only needs to produce the raw factor
                // and a success code. Route the factor to the native .cholesky
                // op (result %name.0) and emit a constant info=0 (result %name.1).
                // The native solver always returns a factor; for the SPD inputs
                // Cholesky targets, info=0 is correct and the wrapper passes it
                // through untouched.
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                operations.append(HLOOperation(
                    result: "\(op.result).0",
                    kind: .cholesky,
                    operands: resolvedOperands,
                    resultType: op.resultType,
                    attributes: op.attributes
                ))
                // info = 0 (success), tensor<i32>. The wrapper compares this to 0.
                var infoAttrs = HLOAttributes()
                infoAttrs.constantValue = .scalar(0)
                operations.append(HLOOperation(
                    result: "\(op.result).1",
                    kind: .constant,
                    operands: [],
                    resultType: TensorType(shape: [], elementType: .int32),
                    attributes: infoAttrs
                ))
            } else if op.kind == .customCall,
                      let target = op.attributes.callTargetName,
                      target.hasPrefix("lapack_strsm") || target.hasPrefix("lapack_dtrsm") {
                // Triangular-solve LAPACK FFI: `@lapack_strsm_ffi(%a, %b)` returns
                // the solution X of the triangular system. The uplo/side/trans/diag
                // flags were decoded onto lower/leftSide/transposeA/unitDiagonal
                // during custom_call parsing. Route straight to the native
                // .triangularSolve op (single result, same name).
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                operations.append(HLOOperation(
                    result: op.result,
                    kind: .triangularSolve,
                    operands: resolvedOperands,
                    resultType: op.resultType,
                    attributes: op.attributes
                ))
            } else if op.kind == .customCall,
                      let target = op.attributes.callTargetName,
                      target.hasPrefix("lapack_sgesdd") || target.hasPrefix("lapack_dgesdd") {
                // SVD LAPACK FFI: `@lapack_sgesdd_ffi(%a)` returns a 5-tuple
                //   (#0 input-scratch, #1 S, #2 U, #3 Vt, #4 info).
                // JAX's `@svd` wrapper masks U/S/Vt to NaN when info != 0 and
                // returns (U, S, Vh). The decomposition itself runs host-side via
                // Accelerate LAPACK (see MetalExecutor's host-SVD shortcut), so the
                // job here is to route the consumed results (#1 S, #2 U, #3 Vt) to
                // native `.svd` ops and emit info = 0 for #4 (LAPACK always
                // succeeds for the well-conditioned inputs SVD targets; the wrapper
                // passes value through untouched when info == 0).
                //
                // Result-component encoding (op.attributes.tupleIndex):
                //   0 = U, 1 = S, 2 = Vh. Shapes are derived from the input M×N
                //   matrix and the `full_matrices` flag decoded from the mode byte.
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                let inputShape = op.resultType.shape   // result #0 == input shape M×N
                let m = inputShape.count >= 1 ? inputShape[0] : 0
                let n = inputShape.count >= 2 ? inputShape[1] : 0
                let minMN = Swift.min(m, n)
                let full = op.attributes.fullMatrices ?? true
                let elem = op.resultType.elementType

                // U: M×M (full) or M×min (thin)  -> custom_call result #2
                var uAttrs = op.attributes
                uAttrs.tupleIndex = 0
                operations.append(HLOOperation(
                    result: "\(op.result).2",
                    kind: .svd,
                    operands: resolvedOperands,
                    resultType: TensorType(shape: [m, full ? m : minMN], elementType: elem),
                    attributes: uAttrs
                ))
                // S: [min]  -> custom_call result #1
                var sAttrs = op.attributes
                sAttrs.tupleIndex = 1
                operations.append(HLOOperation(
                    result: "\(op.result).1",
                    kind: .svd,
                    operands: resolvedOperands,
                    resultType: TensorType(shape: [minMN], elementType: elem),
                    attributes: sAttrs
                ))
                // Vh: N×N (full) or min×N (thin)  -> custom_call result #3
                var vAttrs = op.attributes
                vAttrs.tupleIndex = 2
                operations.append(HLOOperation(
                    result: "\(op.result).3",
                    kind: .svd,
                    operands: resolvedOperands,
                    resultType: TensorType(shape: [full ? n : minMN, n], elementType: elem),
                    attributes: vAttrs
                ))
                // info = 0 (success), tensor<i32>. The wrapper compares this to 0.
                var infoAttrs = HLOAttributes()
                infoAttrs.constantValue = .scalar(0)
                operations.append(HLOOperation(
                    result: "\(op.result).4",
                    kind: .constant,
                    operands: [],
                    resultType: TensorType(shape: [], elementType: .int32),
                    attributes: infoAttrs
                ))
                // #0 is the input-aliased scratch buffer (output_operand_alias to
                // operand 0); nothing consumes it, but alias it to the input so any
                // stray reference resolves.
                if let firstOperand = resolvedOperands.first {
                    valueAliases["\(op.result).0"] = firstOperand
                }
            } else if op.kind == .customCall,
                      let target = op.attributes.callTargetName,
                      target.hasPrefix("lapack_ssyevd") || target.hasPrefix("lapack_dsyevd")
                        || target.hasPrefix("lapack_cheevd") || target.hasPrefix("lapack_zheevd") {
                // Symmetric eigendecomposition LAPACK FFI: `@lapack_ssyevd_ffi(%a)`
                // returns a 3-tuple
                //   (#0 eigenvectors v [N×N], #1 eigenvalues w [N], #2 info).
                // JAX's `@eigh` wrapper masks v/w to NaN when info != 0 and returns
                // (w, v). The decomposition itself runs host-side via Accelerate
                // LAPACK (see MetalExecutor's host-eigh shortcut), so the job here is
                // to route the consumed results (#0 v, #1 w) to native `.eigh` ops
                // and emit info = 0 for #2 (LAPACK always succeeds for the
                // well-conditioned symmetric inputs eigh targets; the wrapper passes
                // value through untouched when info == 0).
                //
                // Result-component encoding (op.attributes.tupleIndex):
                //   0 = w (eigenvalues, ascending), 1 = v (eigenvectors, columns).
                // Shapes derive from the input N×N matrix; the UPLO triangle the
                // routine reads is decoded onto `attributes.lower` from the
                // backend_config `uplo` byte.
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                let inputShape = op.resultType.shape   // result #0 == input shape N×N
                let n = inputShape.count >= 1 ? inputShape[0] : 0
                let elem = op.resultType.elementType

                // v: N×N eigenvectors (columns)  -> custom_call result #0
                var vAttrs = op.attributes
                vAttrs.tupleIndex = 1
                operations.append(HLOOperation(
                    result: "\(op.result).0",
                    kind: .eigh,
                    operands: resolvedOperands,
                    resultType: TensorType(shape: [n, n], elementType: elem),
                    attributes: vAttrs
                ))
                // w: [N] eigenvalues (ascending)  -> custom_call result #1
                var wAttrs = op.attributes
                wAttrs.tupleIndex = 0
                operations.append(HLOOperation(
                    result: "\(op.result).1",
                    kind: .eigh,
                    operands: resolvedOperands,
                    resultType: TensorType(shape: [n], elementType: elem),
                    attributes: wAttrs
                ))
                // info = 0 (success), tensor<i32>. The wrapper compares this to 0.
                var infoAttrs = HLOAttributes()
                infoAttrs.constantValue = .scalar(0)
                operations.append(HLOOperation(
                    result: "\(op.result).2",
                    kind: .constant,
                    operands: [],
                    resultType: TensorType(shape: [], elementType: .int32),
                    attributes: infoAttrs
                ))
            } else if op.kind == .customCall,
                      let target = op.attributes.callTargetName,
                      target.hasPrefix("lapack_sgeqrf") || target.hasPrefix("lapack_dgeqrf") {
                // QR factorization LAPACK FFI: `@lapack_sgeqrf_ffi(%A)` returns a
                // 2-tuple (#0 factored A [M×N], #1 tau [min(M,N)]). Result #0 holds
                // R in its upper triangle and the Householder reflectors below the
                // diagonal; the surrounding StableHLO wrapper slices it to the
                // top-left block and zeroes the strict lower triangle to form R.
                // Result #1 (tau) is consumed only by the paired @lapack_sorgqr_ffi
                // call, which we re-root on A below, so tau is never materialized.
                //
                // The decomposition runs host-side via Accelerate LAPACK (see
                // MetalExecutor's host-QR shortcut). Here we route result #0 to a
                // native `.qr` op tagged component 1 (the factored matrix), which
                // feeds the R-forming slice/select ops, and remember A so the
                // sorgqr call can recompute Q from it.
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                let inputA = resolvedOperands.first
                lastGeqrfInput = inputA
                var factoredAttrs = op.attributes
                factoredAttrs.tupleIndex = 1   // 1 = geqrf factored matrix
                operations.append(HLOOperation(
                    result: "\(op.result).0",
                    kind: .qr,
                    operands: resolvedOperands,
                    resultType: op.resultType,   // result #0 == input shape M×N
                    attributes: factoredAttrs
                ))
                // tau (#1) is unused once sorgqr is re-rooted on A; alias it to the
                // input so any stray reference resolves to a valid value.
                if let inputA {
                    valueAliases["\(op.result).1"] = inputA
                }
            } else if op.kind == .customCall,
                      let target = op.attributes.callTargetName,
                      target.hasPrefix("lapack_sorgqr") || target.hasPrefix("lapack_dorgqr") {
                // Materialize Q LAPACK FFI: `@lapack_sorgqr_ffi(%factored, %tau)`
                // returns Q. Its result type carries Q's shape (M×min for reduced
                // mode, M×M for complete), which the host shortcut reads to pick the
                // sorgqr column count. The decomposition runs host-side from the
                // original matrix A (captured at the paired geqrf call), so route
                // the single result to a native `.qr` op tagged component 0 (Q),
                // re-rooted on A rather than the geqrf intermediate.
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                let qOperand = lastGeqrfInput ?? resolvedOperands.first
                var qAttrs = op.attributes
                qAttrs.tupleIndex = 0   // 0 = Q
                operations.append(HLOOperation(
                    result: op.result,
                    kind: .qr,
                    operands: qOperand.map { [$0] } ?? resolvedOperands,
                    resultType: op.resultType,   // Q shape (M×min or M×M)
                    attributes: qAttrs
                ))
            } else if op.kind == .caseOp, let branches = op.attributes.caseRegions {
                // stablehlo.case (jax.lax.switch): an integer index operand selects
                // among N branch regions. The fast CodeGenerator path has no runtime
                // branch dispatch, so expand the case into a flat op sequence:
                //   1. Inline every branch's operations (renamed to avoid SSA
                //      collisions); branch-local references are rewritten, outer
                //      references (function args / earlier results) pass through.
                //   2. Build an index-select chain over the branch results:
                //        acc = result[N-1]                       (default = last)
                //        for b in (N-2 ... 0):
                //          pred_b = compare(idx == b)
                //          acc    = select(pred_b, result[b], acc)
                //      The index is already clamped to [0, N-1] by JAX, so the
                //      last branch is the correct default and "first match wins".
                // The selection is branch-free (every branch is always computed and
                // the right value picked), matching how the codegen lowers select.
                let resolvedIndex = resolveAlias(op.operands.first ?? "", aliases: valueAliases)
                let indexType = TensorType(shape: [], elementType: .int32)

                // Inline each branch and record the name carrying its return value.
                var branchResults: [String] = []
                for (b, branch) in branches.enumerated() {
                    let prefix = "\(op.result)_case\(b)_"

                    // Names defined locally inside this branch. Only these are
                    // renamed; anything else is an outer SSA value left untouched.
                    var localDefs = Set(branch.operations.map { $0.result })
                    for arg in branch.arguments { localDefs.insert(arg.name) }

                    func rename(_ name: String) -> String {
                        // Resolve outer aliases first, then prefix branch-local names.
                        if localDefs.contains(name) { return prefix + name }
                        return resolveAlias(name, aliases: valueAliases)
                    }

                    for branchOp in branch.operations {
                        operations.append(HLOOperation(
                            result: prefix + branchOp.result,
                            kind: branchOp.kind,
                            operands: branchOp.operands.map(rename),
                            resultType: branchOp.resultType,
                            attributes: branchOp.attributes,
                            resultCount: branchOp.resultCount
                        ))
                    }

                    // A branch returns its single result value (multi-result
                    // switch is not emitted by jax.lax.switch in the common case).
                    let ret = branch.returnValues.first ?? ""
                    branchResults.append(rename(ret))
                }

                // Index-select chain, last branch as the default accumulator.
                var acc = branchResults.last ?? resolvedIndex
                if branchResults.count >= 2 {
                    for b in stride(from: branchResults.count - 2, through: 0, by: -1) {
                        // Constant b (scalar i32) matching the index type.
                        var constAttrs = HLOAttributes()
                        constAttrs.constantValue = .scalar(Double(b))
                        let constName = "\(op.result)_caseidx\(b)"
                        operations.append(HLOOperation(
                            result: constName,
                            kind: .constant,
                            operands: [],
                            resultType: indexType,
                            attributes: constAttrs
                        ))

                        // pred_b = (idx == b), scalar i1.
                        var cmpAttrs = HLOAttributes()
                        cmpAttrs.comparisonDirection = .eq
                        cmpAttrs.inputElementTypes = [.int32, .int32]
                        let predName = "\(op.result)_casecmp\(b)"
                        operations.append(HLOOperation(
                            result: predName,
                            kind: .compare,
                            operands: [resolvedIndex, constName],
                            resultType: TensorType(shape: [], elementType: .int1),
                            attributes: cmpAttrs
                        ))

                        // acc = select(pred_b, result[b], acc). The scalar
                        // predicate is broadcast across the result by codegen.
                        var selAttrs = HLOAttributes()
                        let valueElem = op.resultType.elementType
                        selAttrs.inputElementTypes = [.int1, valueElem, valueElem]
                        let selName = (b == 0) ? op.result : "\(op.result)_casesel\(b)"
                        operations.append(HLOOperation(
                            result: selName,
                            kind: .select,
                            operands: [predName, branchResults[b], acc],
                            resultType: op.resultType,
                            attributes: selAttrs
                        ))
                        acc = selName
                    }
                } else {
                    // Single-branch case: result aliases the only branch output.
                    valueAliases[op.result] = acc
                }
            } else {
                // Substitute aliases in operands for regular operations
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                if resolvedOperands != op.operands {
                    operations.append(HLOOperation(
                        result: op.result,
                        kind: op.kind,
                        operands: resolvedOperands,
                        resultType: op.resultType,
                        attributes: op.attributes
                    ))
                } else {
                    operations.append(op)
                }
            }
            skipNewlines()
        }

        // Debug: uncomment to trace parser output
        // print("[MetalHLO Parser] Parsed \(operations.count) ops in '\(funcName)'")

        skipNewlines()
        try expect(.rightBrace)

        return HLOFunction(
            name: funcName,
            isPrivate: isPrivate,
            inputs: inputs,
            outputTypes: outputTypes,
            operations: operations,
            returnValues: returnValues
        )
    }

    private func parseFunctionArguments() throws -> [HLOArgument] {
        try expect(.leftParen)
        var args: [HLOArgument] = []

        if !check(.rightParen) {
            repeat {
                let name = try parsePercentIdentifier()
                try expect(.colon)
                let type = try parseTensorType()
                // Optional per-arg attribute dict, e.g. `{mhlo.sharding = ...}`.
                skipOptionalAttributeDict()
                args.append(HLOArgument(name: name, type: type))
            } while match(.comma)
        }

        try expect(.rightParen)
        return args
    }

    private func parseReturnTypes() throws -> [TensorType] {
        // Handle both parenthesized: -> (type1, type2)
        // and bare: -> type
        if match(.leftParen) {
            var types: [TensorType] = []

            if !check(.rightParen) {
                repeat {
                    let type = try parseTensorType()
                    // Optional per-result attribute dict, e.g.
                    // `{jax.result_info = "result"}` on the entry's outputs.
                    skipOptionalAttributeDict()
                    types.append(type)
                } while match(.comma)
            }

            try expect(.rightParen)
            return types
        } else {
            // Bare return type (e.g., -> tensor<3x3xf32>)
            let type = try parseTensorType()
            return [type]
        }
    }

    /// Skips an optional MLIR attribute dictionary `{ ... }` (balanced braces)
    /// when one immediately follows. Used for per-argument / per-result
    /// attributes that JAX attaches to entry-function signatures (sharding,
    /// jax.result_info, …) which carry no semantics for the fast path.
    private func skipOptionalAttributeDict() {
        guard check(.leftBrace) else { return }
        advance()
        var depth = 1
        while depth > 0 && !check(.eof) {
            if check(.leftBrace) { depth += 1 }
            else if check(.rightBrace) { depth -= 1 }
            advance()
        }
    }

    private func parseReturn() throws -> [String] {
        try expectKeyword(.return)
        var values: [String] = []

        // Check for empty return (void function) - no values and no type annotation
        if check(.rightBrace) || check(.newline) || check(.eof) {
            return values
        }

        // Check for return with values
        if check(.percentIdentifier) {
            repeat {
                let value = try parsePercentIdentifier()
                values.append(value)
            } while match(.comma)
        }

        // Parse type annotation if present
        if match(.colon) {
            // Skip the return type(s)
            _ = try parseTensorType()
            while match(.comma) {
                _ = try parseTensorType()
            }
        }

        return values
    }

    // MARK: - Operation Parsing

    private func parseOperation() throws -> HLOOperation {
        // Format: %result = stablehlo.op operands attributes : type
        //     or: %result:N = stablehlo.op ... (multi-result, e.g., while, call)
        let result = try parsePercentIdentifier()

        // Check for multi-result syntax: %name:N
        var resultCount = 1
        if match(.colon) {
            if case .integer(let count) = currentToken.kind {
                resultCount = Int(count)
                advance()
            }
        }

        try expect(.equal)

        // Parse operation name (e.g., stablehlo.add)
        let opName = try parseOperationName()
        let kind = try parseOpKind(from: opName)

        // Parse operands and attributes based on operation
        let (operands, attributes, resultType) = try parseOperationBody(kind: kind)

        // Result-producing host callbacks (jax.pure_callback / io_callback) lower
        // to a host-callback custom_call that DOES have an SSA result the graph
        // consumes. We have no host round-trip to satisfy it, so silently
        // compiling would corrupt numerics — reject it loudly here. (The
        // result-less, side-effecting variants are dropped earlier in
        // parseResultlessStatement.)
        if kind == .customCall,
           let target = attributes.callTargetName,
           Parser.droppableHostCallbackTargets.contains(target) {
            throw ParseError.invalidOperation(
                "result-producing host callback custom_call '@\(target)' is not "
                + "supported (jax.pure_callback / io_callback require a host "
                + "round-trip the MetalHLO backend does not implement)",
                location: currentToken.location)
        }

        return HLOOperation(
            result: result,
            kind: kind,
            operands: operands,
            resultType: resultType,
            attributes: attributes,
            resultCount: resultCount
        )
    }

    /// Host-callback custom_call targets that JAX emits for side-effecting host
    /// callbacks (jax.debug.print, jax.debug.callback). These produce no SSA
    /// result and the computation never consumes their (absent) output, so the
    /// device program is numerically identical with them removed — only the
    /// host-side effect (the print) is dropped. We do NOT silently swallow
    /// result-producing callbacks (pure_callback / io_callback): those feed real
    /// values back into the graph and require host round-trip infrastructure we
    /// don't have, so they are left to fail loudly elsewhere.
    private static let droppableHostCallbackTargets: Set<String> = [
        "xla_ffi_python_cpu_callback",
        "xla_python_cpu_callback",
        "xla_python_gpu_callback",
        "xla_ffi_python_gpu_callback",
    ]

    /// Parses a result-less (no `%result =`) operation statement. The only such
    /// statements JAX emits are side-effecting custom_calls — chiefly host
    /// callbacks like jax.debug.print, which lower to
    ///   stablehlo.custom_call @xla_ffi_python_cpu_callback(%x) {has_side_effect = true, ...} : (...) -> ()
    /// When the target is a known droppable host callback we consume and discard
    /// the statement (the print is dropped; numerics are unchanged). Any other
    /// result-less operation throws ParseError.invalidOperation so genuinely
    /// unsupported side effects still surface loudly.
    private func parseResultlessStatement() throws {
        let location = currentToken.location
        let opName = try parseOperationName()
        let kind = try parseOpKind(from: opName)

        guard kind == .customCall else {
            throw ParseError.invalidOperation(
                "unsupported result-less operation '\(opName)' (no SSA result)",
                location: location
            )
        }

        // Capture the target name: @target_name.
        var target = ""
        if check(.atIdentifier) {
            target = String(currentToken.text.dropFirst())  // drop '@'
            advance()
        }

        // Skip the operand list: (%op1, %op2, ...).
        if check(.leftParen) {
            try skipBalancedParens()
        }

        // Skip the attribute block: { ... }. JAX host callbacks carry nested
        // attribute dicts (e.g. `mhlo.backend_config = {index = 0 : ui64}`), so
        // track brace depth rather than stopping at the first `}`.
        if check(.leftBrace) {
            try skipBalancedBraces()
        }

        // Consume the trailing type signature, including the `-> ()` empty result
        // list that parseTypeSignature does not handle.
        try expect(.colon)
        try skipResultlessTypeSignature()

        guard Parser.droppableHostCallbackTargets.contains(target) else {
            throw ParseError.invalidOperation(
                "unsupported side-effecting custom_call target '@\(target)'. "
                + "Host callbacks that return values (jax.pure_callback / io_callback) "
                + "require host round-trip support that is not implemented.",
                location: location
            )
        }
        // Known host-callback side effect: drop it. The computation is unchanged.
    }

    /// Skips the type signature of a result-less custom_call. Same operand-list
    /// form as parseTypeSignature, but the return is the empty tuple `()` (or, in
    /// principle, a normal type) rather than a tensor. parseTypeSignature would
    /// choke on the `()`, so handle the empty result list explicitly.
    private func skipResultlessTypeSignature() throws {
        if check(.leftParen) {
            try expect(.leftParen)
            var depth = 1
            while depth > 0 && !check(.eof) {
                if check(.leftParen) { depth += 1 }
                if check(.rightParen) { depth -= 1 }
                advance()
            }
            try expect(.arrow)
            // Empty result list: `-> ()`. Consume the parens and return.
            if check(.leftParen) {
                try expect(.leftParen)
                var rdepth = 1
                while rdepth > 0 && !check(.eof) {
                    if check(.leftParen) { rdepth += 1 }
                    if check(.rightParen) { rdepth -= 1 }
                    advance()
                }
                return
            }
            // Non-empty result: a normal type follows; consume it.
            _ = try parseTensorType()
            return
        }
        // Bare return type (no parenthesised operand list).
        _ = try parseTensorType()
    }

    /// Consumes a balanced `( ... )` group, including nested parens. Assumes the
    /// current token is the opening `(`.
    private func skipBalancedParens() throws {
        try expect(.leftParen)
        var depth = 1
        while depth > 0 && !check(.eof) {
            if check(.leftParen) { depth += 1 }
            if check(.rightParen) { depth -= 1 }
            advance()
        }
    }

    /// Consumes a balanced `{ ... }` group, including nested braces. Assumes the
    /// current token is the opening `{`. Used to skip attribute dictionaries that
    /// contain nested dicts (e.g. `mhlo.backend_config = {index = 0 : ui64}`).
    private func skipBalancedBraces() throws {
        try expect(.leftBrace)
        var depth = 1
        while depth > 0 && !check(.eof) {
            if check(.leftBrace) { depth += 1 }
            if check(.rightBrace) { depth -= 1 }
            advance()
        }
    }

    /// Tracks whether the last parsed operation name was in generic form (quoted)
    private var isGenericForm = false

    /// Set by parseOperationRegion when the region it just consumed is the
    /// numerically-stable log-add-exp combiner jax.lax.cumlogsumexp emits
    /// (signature: a `log_plus_one` op inside the reducer). The reducer's first
    /// op is `maximum`, so the generic computation-kind scan returns `.max`;
    /// this flag lets the reduce_window caller override that to `.logAddExp`.
    private var lastRegionWasLogAddExp = false

    private func parseOperationName() throws -> String {
        var name = ""
        isGenericForm = false

        // Handle quoted operation names (generic form): "stablehlo.rng"
        if case .string(let stringValue) = currentToken.kind {
            name = stringValue
            isGenericForm = true
            advance()
            return name
        }

        // Handle identifiers that may contain dots (e.g., stablehlo.add, func.func)
        if case .identifier = currentToken.kind {
            name = currentToken.text
            advance()

            // Handle dotted names - only continue if name ends with dot and next is identifier
            while name.hasSuffix(".") && currentToken.kind == .identifier {
                name += currentToken.text
                advance()
            }
        } else if case .keyword(.stablehlo) = currentToken.kind {
            name = "stablehlo"
            advance()
            // Continue parsing if there's more to the dotted name
            while name.hasSuffix(".") && currentToken.kind == .identifier {
                name += currentToken.text
                advance()
            }
        } else {
            throw ParseError.unexpectedToken(expected: "operation name", got: currentToken)
        }

        return name
    }

    private func parseOpKind(from name: String) throws -> HLOOpKind {
        // Handle function calls: "call" or "func.call"
        if name == "call" || name == "func.call" {
            return .call
        }

        // Extract the operation name after "stablehlo."
        let opName: String
        if name.hasPrefix("stablehlo.") {
            opName = String(name.dropFirst("stablehlo.".count))
        } else {
            opName = name
        }

        // top_k arrives as a single 2-result op (values, indices). We model it
        // with two single-result kinds; parse it under .topKValues here and the
        // function-body loop splits off the .topKIndices companion op.
        if opName == "top_k" {
            return .topKValues
        }

        guard let kind = HLOOpKind(rawValue: opName) else {
            throw ParseError.invalidOperation(
                "Unknown operation '\(name)'",
                location: currentToken.location
            )
        }

        return kind
    }

    private func parseOperationBody(kind: HLOOpKind) throws -> ([String], HLOAttributes, TensorType) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Handle generic form: "stablehlo.op"(%op1, %op2) {attributes} : type
        if isGenericForm {
            (operands, attributes) = try parseGenericFormOperandsAndAttributes(kind: kind)
        } else if kind == .customCall {
            // Special handling for custom_call which has @target(operands) format
            (operands, attributes) = try parseCustomCallOperandsAndAttributes()
        } else if kind == .reduce {
            // Special handling for reduce: (%input init: %init) applies stablehlo.add across dimensions = [0]
            (operands, attributes) = try parseReduceOperandsAndAttributes()
        } else if kind == .compare {
            // Special handling for compare: EQ, %arg0, %arg1, FLOAT : types
            (operands, attributes) = try parseCompareOperandsAndAttributes()
        } else if kind == .select {
            // Special handling for select: %pred, %on_true, %on_false : pred_type, result_type
            (operands, attributes) = try parseSelectOperandsAndAttributes()
        } else if kind == .iota {
            // Special handling for iota: dim = N : tensor<...>
            (operands, attributes) = try parseIotaOperandsAndAttributes()
        } else if kind == .optimizationBarrier {
            // optimization_barrier %a, %b : t0, t1  (variadic identity).
            // Parse the operands, then consume the comma-separated type list
            // (one type per result). The op is eliminated via alias resolution
            // in the main loop, so the result type here is informational.
            while check(.percentIdentifier) {
                let operand = try parsePercentIdentifier()
                operands.append(operand)
                _ = match(.comma)
            }
            try expect(.colon)
            let firstType = try parseTensorType()
            while match(.comma) {
                _ = try parseTensorType()
            }
            return (operands, attributes, firstType)
        } else if kind == .topKValues {
            // top_k %x {k = K} : (intype) -> (vtype, itype)
            // Parsed under .topKValues; the values type becomes resultType and
            // the indices type is stashed so the main loop can split off the
            // .topKIndices op. top_k always operates on the last axis.
            let operand = try parsePercentIdentifier()
            operands.append(operand)
            _ = match(.comma)
            // Attribute block: { k = K }
            if match(.leftBrace) {
                while !check(.rightBrace) && !check(.eof) {
                    if checkIdentifier("k") {
                        try expectIdentifier("k")
                        try expect(.equal)
                        attributes.topK = try parseInteger()
                    } else {
                        advance()
                    }
                    _ = match(.comma)
                }
                try expect(.rightBrace)
            }
            // Type signature: (intype) -> (vtype, itype)
            try expect(.colon)
            try expect(.leftParen)
            _ = try parseTensorType()  // input type (operand type)
            try expect(.rightParen)
            try expect(.arrow)
            try expect(.leftParen)
            let valuesType = try parseTensorType()
            try expect(.comma)
            let indicesType = try parseTensorType()
            try expect(.rightParen)
            attributes.topKIndexType = indicesType
            return (operands, attributes, valuesType)
        } else if kind == .getTupleElement {
            // Special handling for get_tuple_element: %tuple[index]
            (operands, attributes) = try parseGetTupleElementOperandsAndAttributes()
        } else if kind == .call {
            // func.call @name(%arg0, %arg1) : (input_types) -> (output_types)
            (operands, attributes) = try parseCallOperandsAndAttributes()
        } else if kind == .whileOp && check(.leftParen) {
            // JAX-emitted while format:
            //   stablehlo.while(%iterArg = %c_0, %iterArg_1 = %arg0) : tensor<i32>, tensor<i32>
            //   cond { ... } do { ... }
            // This format includes type annotation and regions inline, so return directly.
            let (whileOperands, whileAttrs, whileType) = try parseWhileBindingsAndRegions()
            return (whileOperands, whileAttrs, whileType)
        } else {
            // Parse operands (% identifiers before attributes/type)
            while check(.percentIdentifier) {
                let operand = try parsePercentIdentifier()
                operands.append(operand)
                _ = match(.comma)
            }

            // Parse attributes based on operation kind
            attributes = try parseAttributes(for: kind)
        }

        // Skip scatter/selectAndScatter reducer regions: ({^bb0(...): ...stablehlo.return...})
        // These appear after attributes and before the type signature.
        // Also handles reduce_window in generic form, where the reduction op is
        // expressed as an inline region (Flax avg_pool / max_pool emit this).
        if (kind == .scatter || kind == .selectAndScatter || kind == .reduceWindow
            || kind == .allReduce)
            && check(.leftParen) {
            // Skip one or more inline regions: ({...}) or ({...}, {...}).
            // select_and_scatter has TWO regions (select predicate + scatter
            // combiner) separated by `,`; the others have one.
            try expect(.leftParen)
            while check(.leftBrace) {
                var depth = 1
                advance() // skip {
                while depth > 0 && !check(.eof) {
                    if check(.leftBrace) { depth += 1 }
                    if check(.rightBrace) { depth -= 1 }
                    // Detect reduction op for reduce_window from body content.
                    // Body uses dotted identifiers like "stablehlo.add" — match by suffix.
                    if kind == .reduceWindow && attributes.reductionKind == nil
                        && currentToken.kind == .identifier {
                        let t = currentToken.text
                        if t == "stablehlo.add" || t == "add" {
                            attributes.reductionKind = .sum
                        } else if t == "stablehlo.maximum" || t == "maximum" {
                            attributes.reductionKind = .max
                        } else if t == "stablehlo.minimum" || t == "minimum" {
                            attributes.reductionKind = .min
                        } else if t == "stablehlo.multiply" || t == "multiply" {
                            attributes.reductionKind = .product
                        }
                    }
                    // jax.lax.cumlogsumexp emits a reduce_window whose reducer is
                    // a numerically-stable log-add-exp, not a plain reduction:
                    //   max(a,b); d=a-b; nan=(d!=d); s=a+b; e=exp(-|d|);
                    //   l=log1p(e); r=max+l; select(nan, s, r)
                    // The first op in that region is `maximum`, so the matcher
                    // above mis-detects it as `.max`. The `log_plus_one`
                    // (log1p) op appears only in this logaddexp combiner, so use
                    // it as the unambiguous signature and override the earlier
                    // guess. Without this, cumlogsumexp computes a cumulative
                    // max instead of a cumulative logsumexp.
                    if kind == .reduceWindow && currentToken.kind == .identifier {
                        let t = currentToken.text
                        if t == "stablehlo.log_plus_one" || t == "log_plus_one" {
                            attributes.reductionKind = .logAddExp
                        }
                    }
                    // Detect scatter combine op the same way — JAX-emitted
                    // scatter (e.g. embedding-table backward) puts the
                    // reducer inline as a region after the attribute block.
                    // Without this, computation kind silently defaults to
                    // `.set`, so scatter-add of repeated indices keeps only
                    // one update — embedding gradients off by ~the average
                    // occurrence count per token.
                    if kind == .scatter && attributes.scatterComputationKind == nil
                        && currentToken.kind == .identifier {
                        let t = currentToken.text
                        if t == "stablehlo.add" || t == "add" {
                            attributes.scatterComputationKind = .add
                        } else if t == "stablehlo.maximum" || t == "maximum" {
                            attributes.scatterComputationKind = .max
                        } else if t == "stablehlo.minimum" || t == "minimum" {
                            attributes.scatterComputationKind = .min
                        } else if t == "stablehlo.multiply" || t == "multiply" {
                            attributes.scatterComputationKind = .mul
                        }
                    }
                    advance()
                }
                // After consuming this region, optionally skip a comma to
                // pick up the next region for select_and_scatter.
                _ = match(.comma)
            }
            try expect(.rightParen)
        }

        // Parse type signature
        try expect(.colon)
        let resultType: TensorType
        if kind == .reduce && operands.count >= 4 {
            // Multi-input reduce (argmax/argmin): the signature has two result
            // types — the value type and the i32/i64 index type, e.g.
            //   (tensor<4x8xf32>, tensor<4x8xi32>, tensor<f32>, tensor<i32>)
            //       -> (tensor<4xf32>, tensor<4xi32>)
            // Capture both so the main loop can split off the .reduceArg index
            // op. parseTypeSignature would discard the index type.
            let (valueType, indexType) = try parseReduceResultTypes()
            resultType = valueType
            attributes.argReduceIndexType = indexType
        } else {
            resultType = try parseTypeSignature(for: kind)
        }

        // For reduce operations, skip the optional reducer block that follows the type.
        // Format: ... : (types) -> type\n  reducer(%args) (%args) { body }
        if kind == .reduce {
            skipNewlines()
            if checkIdentifier("reducer") {
                try expectIdentifier("reducer")
                // Skip argument groups: (%arg1: type, ...) (%arg2: type, ...)
                while check(.leftParen) {
                    try expect(.leftParen)
                    var depth = 1
                    while depth > 0 && !check(.eof) {
                        if check(.leftParen) { depth += 1 }
                        if check(.rightParen) { depth -= 1 }
                        advance()
                    }
                }
                skipNewlines()
                // Skip the reducer body block { ... }
                if check(.leftBrace) {
                    try expect(.leftBrace)
                    var braceDepth = 1
                    // Detect reduction kind from body
                    while braceDepth > 0 && !check(.eof) {
                        if check(.leftBrace) { braceDepth += 1 }
                        if check(.rightBrace) {
                            braceDepth -= 1
                            if braceDepth == 0 { break }
                        }
                        // Detect comparison-based reduce (argmax/argmin)
                        if checkIdentifier("GT") || checkIdentifier("GE") {
                            if attributes.reductionKind == nil { attributes.reductionKind = .max }
                        } else if checkIdentifier("LT") || checkIdentifier("LE") {
                            if attributes.reductionKind == nil { attributes.reductionKind = .min }
                        }
                        advance()
                    }
                    try expect(.rightBrace)
                }
            }
        }

        return (operands, attributes, resultType)
    }

    /// Parse func.call operands: @name(%arg0, %arg1) : (in_types) -> (out_types)
    private func parseCallOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Parse target function name: @name
        let targetName = try parseAtIdentifier()
        attributes.functionCallTarget = targetName

        // Parse operands: (%arg0, %arg1, ...)
        try expect(.leftParen)
        if !check(.rightParen) {
            repeat {
                let operand = try parsePercentIdentifier()
                operands.append(operand)
            } while match(.comma)
        }
        try expect(.rightParen)

        return (operands, attributes)
    }

    /// Parse custom_call operands and attributes in the format: @target(operands) { backend_config = "..." }
    private func parseCustomCallOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Parse target name: @target_name
        if check(.atIdentifier) {
            attributes.callTargetName = String(currentToken.text.dropFirst())  // Remove @
            advance()
        }

        // Parse operands: (%op1, %op2, ...)
        if match(.leftParen) {
            if !check(.rightParen) {
                repeat {
                    let operand = try parsePercentIdentifier()
                    operands.append(operand)
                } while match(.comma)
            }
            try expect(.rightParen)
        }

        // Only LAPACK FFI custom_calls (@lapack_*_ffi) need their nested
        // `mhlo.backend_config = {flag = N : ui8}` dict decoded. Every other
        // custom_call is skipped with the original token-by-token logic so this
        // change can't alter how unrelated targets parse.
        let isLapackTarget = (attributes.callTargetName?.hasPrefix("lapack_") ?? false)

        // Parse attribute block: { backend_config = "..." }
        if match(.leftBrace) {
            skipNewlines()
            while !check(.rightBrace) && !check(.eof) {
                if checkIdentifier("backend_config") {
                    try expectIdentifier("backend_config")
                    try expect(.equal)
                    if case .string(let value) = currentToken.kind {
                        attributes.backendConfig = value
                        advance()
                    }
                } else if isLapackTarget
                            && (currentToken.kind == .identifier)
                            && (currentToken.text == "mhlo.backend_config") {
                    // LAPACK FFI custom_calls carry their operation flags in a
                    // nested `mhlo.backend_config = {uplo = 76 : ui8, ...}` dict.
                    // The flags are single ASCII characters encoded as ui8:
                    //   uplo  : 'L'(76) lower / 'U'(85) upper
                    //   side  : 'L'(76) left  / 'R'(82) right
                    //   trans : 'N'(78) none  / 'T'(84) transpose / 'C'(67) adjoint
                    //   diag  : 'N'(78) non-unit / 'U'(85) unit diagonal
                    // Decode them onto the existing triangular_solve / cholesky
                    // attribute fields so the LAPACK calls route to the native
                    // MPSGraph implementations.
                    advance()
                    if match(.equal), check(.leftBrace) {
                        try parseLapackBackendConfig(into: &attributes)
                    }
                } else {
                    // Skip an unknown attribute value, balancing nested brackets.
                    // StableHLO attribute values can contain commas/colons inside
                    // `<...>`, `[...]`, `(...)`, `{...}` (e.g.
                    // `output_operand_aliases = [#stablehlo.output_operand_alias<
                    // output_tuple_indices = [0], operand_index = 0, ...>]` or
                    // `result_layouts = [dense<[0, 1]> : tensor<2xindex>, ...]`),
                    // so a naive token-at-a-time skip stops at the first inner
                    // comma and corrupts the parse. Skip whole balanced groups.
                    skipBalancedAttributeValue()
                }
                _ = match(.comma)
                skipNewlines()
            }
            try expect(.rightBrace)
        }

        return (operands, attributes)
    }

    /// Skips a single unknown attribute value (identifier/value up to the next
    /// top-level `,` or `}`), consuming any nested `<...>`, `[...]`, `(...)`,
    /// `{...}` groups as balanced units so embedded commas/colons don't terminate
    /// the skip early. Stops with the cursor on the delimiting `,`/`}`.
    private func skipBalancedAttributeValue() {
        var depth = 0
        while !check(.eof) {
            switch currentToken.kind {
            case .leftAngle, .leftBracket, .leftParen, .leftBrace:
                depth += 1
            case .rightAngle, .rightBracket, .rightParen:
                if depth > 0 { depth -= 1 }
            case .rightBrace:
                if depth == 0 { return }   // end of the attribute block
                depth -= 1
            case .comma:
                if depth == 0 { return }   // attribute separator
            default:
                break
            }
            advance()
        }
    }

    /// Parses a LAPACK FFI `mhlo.backend_config` dict `{ flag = N : ui8, ... }`
    /// and decodes the ASCII-coded flags onto triangular_solve / cholesky
    /// attribute fields. Assumes the current token is the opening `{`.
    private func parseLapackBackendConfig(into attributes: inout HLOAttributes) throws {
        try expect(.leftBrace)
        skipNewlines()
        while !check(.rightBrace) && !check(.eof) {
            if currentToken.kind == .identifier {
                let flag = currentToken.text
                advance()
                if match(.equal), case .integer(let code) = currentToken.kind {
                    advance()
                    applyLapackFlag(flag, code: Int(code), into: &attributes)
                    // Consume the trailing `: ui8` type annotation if present.
                    if match(.colon) { advance() }
                }
            } else {
                advance()
            }
            _ = match(.comma)
            skipNewlines()
        }
        try expect(.rightBrace)
    }

    /// Maps a single decoded LAPACK flag (ASCII ui8 code) onto the matching
    /// triangular_solve / cholesky attribute.
    private func applyLapackFlag(_ flag: String, code: Int, into attributes: inout HLOAttributes) {
        let ch = Character(UnicodeScalar(UInt8(truncatingIfNeeded: code)))
        switch flag {
        case "uplo":
            // 'L' lower, 'U' upper.
            attributes.lower = (ch == "L")
        case "side":
            // 'L' left, 'R' right.
            attributes.leftSide = (ch == "L")
        case "trans_x", "trans":
            // 'N' none, 'T' transpose, 'C' conjugate-transpose.
            switch ch {
            case "T": attributes.transposeA = .transpose
            case "C": attributes.transposeA = .adjoint
            default:  attributes.transposeA = .noTranspose
            }
        case "diag":
            // 'U' unit diagonal, 'N' non-unit.
            attributes.unitDiagonal = (ch == "U")
        case "mode":
            // sgesdd jobz: 'A' (65) full_matrices, 'S' (83) reduced/thin.
            attributes.fullMatrices = (ch == "A")
        default:
            break
        }
    }

    /// Parse generic form operands and attributes
    /// Format: (%op1, %op2, ...) ({region}) {attr = value, ...}
    /// The region is optional and used by operations like scatter and reduce.
    private func parseGenericFormOperandsAndAttributes(kind: HLOOpKind) throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Parse operands: (%op1, %op2, ...)
        if match(.leftParen) {
            if !check(.rightParen) {
                repeat {
                    let operand = try parsePercentIdentifier()
                    operands.append(operand)
                } while match(.comma)
            }
            try expect(.rightParen)
        }

        // Parse the inherent-attributes (properties) block emitted in the
        // newer generic form: <{ attr = value, ... }>. JAX 0.10 lowers
        // jnp.cumsum / jax.lax.cumsum to a generic stablehlo.reduce_window with
        // the window/padding/dilation attributes inside this `<{...}>` dict and
        // the reducer expressed as a trailing region. Route reduce_window to its
        // dedicated attribute parser so window_dimensions / padding / dilations
        // are captured instead of silently skipped.
        if check(.leftAngle) {
            advance() // consume '<'
            if match(.leftBrace) {
                if kind == .reduceWindow {
                    let windowAttrs = try parseReduceWindowAttributes()
                    attributes.windowDimensions = windowAttrs.windowDimensions
                    attributes.windowStrides = windowAttrs.windowStrides
                    attributes.convPadding = windowAttrs.convPadding
                    attributes.baseDilations = windowAttrs.baseDilations
                    attributes.windowDilations = windowAttrs.windowDilations
                } else if kind == .sort {
                    // JAX lowers jnp.sort/argsort/lexsort to the generic form
                    //   "stablehlo.sort"(ops) <{dimension = N : i64, is_stable = b}> ({comparator})
                    // Capture the sort axis (into `axis`) and stability; the
                    // direction comes from the comparator region (handled below).
                    while !check(.rightBrace) && !check(.eof) {
                        if checkIdentifier("dimension") {
                            try expectIdentifier("dimension")
                            try expect(.equal)
                            attributes.axis = try parseInteger()
                            if match(.colon) { advance() }  // skip ': i64' type suffix
                        } else if checkIdentifier("is_stable") {
                            try expectIdentifier("is_stable")
                            try expect(.equal)
                            if checkBool(true) { attributes.isStable = true; advance() }
                            else if checkBool(false) { attributes.isStable = false; advance() }
                        } else {
                            advance()
                        }
                        _ = match(.comma)
                    }
                }
                // Skip any remaining/unknown attributes up to the closing brace.
                while !check(.rightBrace) && !check(.eof) {
                    advance()
                }
                try expect(.rightBrace)
            }
            if check(.rightAngle) {
                advance() // consume '>'
            }
        }

        // Parse region if present: ({^bb0(%arg0: type, %arg1: type): ops... })
        // This appears between operands and attributes for ops like scatter and reduce.
        // The region body determines the computation kind (e.g., identity, add, max).
        if kind == .caseOp && check(.leftParen) {
            // stablehlo.case carries one full region per branch:
            //   "stablehlo.case"(%idx) ({...}, {...}, {...}) : (tensor<i32>) -> T
            // Capture every branch region; the main parse loop expands them into
            // inlined ops plus an index-select chain (no runtime branch dispatch
            // exists on the fast path).
            try expect(.leftParen)
            var branches: [Region] = []
            repeat {
                branches.append(try parseRegion())
                skipNewlines()
            } while match(.comma)
            try expect(.rightParen)
            attributes.caseRegions = branches
            skipNewlines()
        } else if kind == .map && check(.leftParen) {
            // stablehlo.map carries a full per-element computation region, not a
            // single reduction kind. Capture the whole region so codegen can
            // inline its body over the broadcasted element-wise inputs.
            try expect(.leftParen)
            attributes.mapComputation = try parseRegion()
            try expect(.rightParen)
            skipNewlines()
        } else if kind == .sort && check(.leftParen) {
            // stablehlo.sort's comparator region returns true when arg1 should
            // precede arg2. We don't execute the region; we only need the sort
            // DIRECTION. JAX wraps it in NaN/zero canonicalization (FLOAT EQ/NE
            // compares) followed by the order-determining compare over the
            // canonicalized values, tagged TOTALORDER: LT => ascending,
            // GT => descending. Scan the region for that compare.
            try expect(.leftParen)
            let comparator = try parseRegion()
            var descending = false
            for regionOp in comparator.operations where regionOp.kind == .compare {
                if let dir = regionOp.attributes.comparisonDirection {
                    // The canonicalization compares are EQ/NE; the order compare
                    // is LT or GT — the last such one decides direction.
                    if dir == .gt || dir == .ge { descending = true }
                    else if dir == .lt || dir == .le { descending = false }
                }
            }
            attributes.sortDescending = descending
            try expect(.rightParen)
            skipNewlines()
        } else if check(.leftParen) {
            let computationKind = try parseOperationRegion()
            if kind == .scatter {
                attributes.scatterComputationKind = computationKind
            } else if kind == .reduceWindow {
                // Map the reducer region (stablehlo.add for cumsum) onto the
                // reduce_window reduction kind so codegen sums the window.
                if lastRegionWasLogAddExp {
                    // cumlogsumexp's stable log-add-exp reducer leads with a
                    // `maximum` op, so computationKind is .max here; the
                    // log_plus_one signature corrects it to the real combiner.
                    attributes.reductionKind = .logAddExp
                } else {
                    switch computationKind {
                    case .add: attributes.reductionKind = .sum
                    case .max: attributes.reductionKind = .max
                    case .min: attributes.reductionKind = .min
                    case .mul: attributes.reductionKind = .product
                    default: break
                    }
                }
            }
        }

        // Parse attribute block: {attr = value, ...}
        if match(.leftBrace) {
            skipNewlines()

            // For gather operations, use specialized parsing
            if kind == .gather {
                attributes.gatherDimensionNumbers = try parseGatherDimensionNumbers()
                // Skip to end of brace
                while !check(.rightBrace) && !check(.eof) {
                    advance()
                }
            } else if kind == .scatter {
                // The outer `{ … }` attribute block is consumed here (matched at
                // the top of this branch and expected closed just below), so the
                // scatter attribute parser must stop at — not swallow — its `}`.
                let (dimNumbers, attrComputationKind) = try parseScatterAttributes(stopAtOuterBrace: true)
                attributes.scatterDimensionNumbers = dimNumbers
                // Only overwrite computation kind if the attribute block had one;
                // otherwise preserve what the region parsing already set.
                if attrComputationKind != nil {
                    attributes.scatterComputationKind = attrComputationKind
                }
                while !check(.rightBrace) && !check(.eof) {
                    advance()
                }
            } else if kind == .triangularSolve {
                attributes = try parseTriangularSolveAttributes()
                while !check(.rightBrace) && !check(.eof) {
                    advance()
                }
            } else {
                // Generic attribute parsing
                while !check(.rightBrace) && !check(.eof) {
                    // Parse attribute name
                    if checkIdentifier("index") && kind == .getTupleElement {
                        // Parse index attribute: index = 0 : i32
                        try expectIdentifier("index")
                        try expect(.equal)
                        if case .integer(let indexValue) = currentToken.kind {
                            attributes.tupleIndex = Int(indexValue)
                            advance()
                        }
                        // Skip optional type annotation ": i32"
                        if match(.colon) {
                            while !check(.comma) && !check(.rightBrace) && !check(.eof) {
                                advance()
                            }
                        }
                    } else if checkIdentifier("rng_distribution") {
                        try expectIdentifier("rng_distribution")
                        try expect(.equal)
                        // Parse #stablehlo<rng_distribution UNIFORM> or #stablehlo<rng_distribution NORMAL>
                        if check(.hashIdentifier) {
                            // Skip #stablehlo
                            advance()
                            // Parse <rng_distribution UNIFORM>
                            if match(.leftAngle) {
                                // Skip "rng_distribution" identifier
                                if check(.identifier) {
                                    advance()
                                }
                                // Parse UNIFORM or NORMAL
                                if checkIdentifier("UNIFORM") {
                                    attributes.rngDistribution = .uniform
                                    advance()
                                } else if checkIdentifier("NORMAL") {
                                    attributes.rngDistribution = .normal
                                    advance()
                                }
                                try expect(.rightAngle)
                            }
                        }
                    } else {
                        // Skip unknown attributes
                        advance()
                    }
                    _ = match(.comma)
                    skipNewlines()
                }
            }
            try expect(.rightBrace)
        }

        // Parse region if present AFTER attribute block (e.g., scatter in JAX-emitted MLIR).
        // JAX emits: operands, attributes, region — unlike the MLIR spec order (operands, region, attributes).
        if check(.leftParen) {
            let computationKind = try parseOperationRegion()
            if kind == .scatter {
                // Only set if not already determined by the attribute block
                if attributes.scatterComputationKind == nil {
                    attributes.scatterComputationKind = computationKind
                }
            }
        }

        return (operands, attributes)
    }

    /// Parse reduce operands and attributes
    /// Supports two formats:
    ///   1. (%input init: %init) applies stablehlo.add across dimensions = [0]
    ///   2. %input, %init applies stablehlo.add across dimensions = [0]
    private func parseReduceOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Check which format we're parsing
        if check(.leftParen) {
            // Format 1: (%input init: %init) [, (%input2 init: %init2)]
            // Supports multi-input reduce used by argmax/argmin
            repeat {
                try expect(.leftParen)

                // Parse input operand
                let inputOperand = try parsePercentIdentifier()
                operands.append(inputOperand)

                // Parse "init:" and init operand
                try expectIdentifier("init")
                try expect(.colon)
                let initOperand = try parsePercentIdentifier()
                operands.append(initOperand)

                try expect(.rightParen)
            } while match(.comma) && check(.leftParen)
        } else {
            // Format 2: %input, %init
            // Parse input operand
            let inputOperand = try parsePercentIdentifier()
            operands.append(inputOperand)

            _ = match(.comma)

            // Parse init operand
            let initOperand = try parsePercentIdentifier()
            operands.append(initOperand)
        }

        // Parse: applies stablehlo.add across dimensions = [0]
        if checkIdentifier("applies") {
            try expectIdentifier("applies")
            let reductionOp = try parseOperationName()

            if reductionOp.contains("add") {
                attributes.reductionKind = .sum
            } else if reductionOp.contains("max") {
                attributes.reductionKind = .max
            } else if reductionOp.contains("min") {
                attributes.reductionKind = .min
            }
            if reductionOp.contains("multiply") {
                attributes.reductionKind = .product
            } else if reductionOp.contains("and") {
                attributes.reductionKind = .and
            } else if reductionOp.contains("or") {
                attributes.reductionKind = .or
            }

            if checkIdentifier("across") {
                try expectIdentifier("across")
                try expectIdentifier("dimensions")
                try expect(.equal)
                attributes.dimensions = try parseDimensionList()
            }
        }

        // Parse: across dimensions = [0]
        if checkIdentifier("across") {
            try expectIdentifier("across")
            try expectIdentifier("dimensions")
            try expect(.equal)
            attributes.dimensions = try parseDimensionList()
        }

        return (operands, attributes)
    }

    /// Parse compare operands and attributes
    /// Supports two formats:
    ///   1. Direction first: EQ, %arg0, %arg1, FLOAT : types (official StableHLO test format)
    ///   2. Operands first: %arg0, %arg1, EQ, FLOAT : types (common MLIR format)
    private func parseCompareOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        let directions = ["EQ", "NE", "LT", "LE", "GT", "GE"]

        // Check if direction comes first (official StableHLO format)
        var directionFirst = false
        for dir in directions {
            if checkIdentifier(dir) {
                directionFirst = true
                attributes.comparisonDirection = ComparisonDirection(rawValue: dir)
                advance()
                _ = match(.comma)
                break
            }
        }

        // Parse operands
        while check(.percentIdentifier) {
            let operand = try parsePercentIdentifier()
            operands.append(operand)
            _ = match(.comma)
        }

        // If direction wasn't first, parse it after operands
        if !directionFirst {
            var foundDirection = false
            for dir in directions {
                if checkIdentifier(dir) {
                    attributes.comparisonDirection = ComparisonDirection(rawValue: dir)
                    advance()
                    _ = match(.comma)
                    foundDirection = true
                    break
                }
            }

            if !foundDirection {
                throw ParseError.unexpectedToken(expected: "comparison direction (EQ, NE, LT, LE, GT, GE)", got: currentToken)
            }
        }

        // Skip optional comparison type (FLOAT, SIGNED, UNSIGNED, TOTALORDER)
        let compTypes = ["FLOAT", "SIGNED", "UNSIGNED", "TOTALORDER"]
        for compType in compTypes {
            if checkIdentifier(compType) {
                advance()
                break
            }
        }

        return (operands, attributes)
    }

    /// Parse select operands in the format: %pred, %on_true, %on_false
    private func parseSelectOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        let attributes = HLOAttributes()

        // Parse the three operands: predicate, true value, false value
        while check(.percentIdentifier) {
            let operand = try parsePercentIdentifier()
            operands.append(operand)
            _ = match(.comma)
        }

        return (operands, attributes)
    }

    /// Parse iota operands and attributes in the format: dim = N
    /// iota has no operands, just a dimension attribute
    private func parseIotaOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        let operands: [String] = []
        var attributes = HLOAttributes()

        // Parse: dim = N
        if checkIdentifier("dim") {
            try expectIdentifier("dim")
            try expect(.equal)
            if case .integer(let dimValue) = currentToken.kind {
                attributes.iotaDimension = Int(dimValue)
                advance()
            } else {
                throw ParseError.unexpectedToken(expected: "integer dimension", got: currentToken)
            }
        }

        return (operands, attributes)
    }

    /// Parse get_tuple_element operands and attributes.
    /// Pretty-printed form: %tuple[index]
    private func parseGetTupleElementOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Parse the tuple operand
        let operand = try parsePercentIdentifier()
        operands.append(operand)

        // Parse [index] if present (pretty-printed form)
        if match(.leftBracket) {
            if case .integer(let indexValue) = currentToken.kind {
                attributes.tupleIndex = Int(indexValue)
                advance()
            }
            try expect(.rightBracket)
        }

        return (operands, attributes)
    }

    private func parseAttributes(for kind: HLOOpKind) throws -> HLOAttributes {
        var attributes = HLOAttributes()

        // Handle operation-specific attribute parsing
        switch kind {
        case .constant:
            attributes.constantValue = try parseConstantValue()

        case .transpose:
            if match(.comma) || checkIdentifier("dims") {
                if checkIdentifier("dims") {
                    try expectIdentifier("dims")
                    try expect(.equal)
                }
                attributes.dimensions = try parseDimensionList()
            }

        case .broadcastInDim:
            if match(.comma) || checkIdentifier("dims") {
                if checkIdentifier("dims") {
                    try expectIdentifier("dims")
                    try expect(.equal)
                }
                attributes.dimensions = try parseDimensionList()
            }

        case .reduce:
            attributes = try parseReduceAttributes()

        case .dotGeneral:
            attributes.dotDimensionNumbers = try parseDotDimensionNumbers()

        case .gather:
            attributes.gatherDimensionNumbers = try parseGatherDimensionNumbers()

        case .scatter:
            (attributes.scatterDimensionNumbers, attributes.scatterComputationKind) = try parseScatterAttributes()

        case .compare:
            attributes.comparisonDirection = try parseComparisonDirection()

        case .slice:
            attributes = try parseSliceAttributes()

        case .pad:
            attributes = try parsePadAttributes()

        case .concatenate:
            if match(.comma) || checkIdentifier("dim") || checkIdentifier("dimension") {
                if checkIdentifier("dim") {
                    try expectIdentifier("dim")
                } else if checkIdentifier("dimension") {
                    try expectIdentifier("dimension")
                }
                try expect(.equal)
                attributes.axis = try parseInteger()
            }

        case .rng:
            // RNG has format: shape = [...], distribution = UNIFORM/NORMAL
            // Skip shape attribute (output shape comes from result type)
            while !check(.colon) && !check(.eof) {
                if checkIdentifier("shape") {
                    try expectIdentifier("shape")
                    try expect(.equal)
                    // Skip the shape array [...]
                    if match(.leftBracket) {
                        while !check(.rightBracket) && !check(.eof) {
                            advance()
                        }
                        try expect(.rightBracket)
                    }
                    _ = match(.comma)
                } else if checkIdentifier("distribution") {
                    try expectIdentifier("distribution")
                    try expect(.equal)
                    attributes.rngDistribution = try parseRNGDistribution()
                    _ = match(.comma)
                } else if checkIdentifier("UNIFORM") || checkIdentifier("NORMAL") {
                    // Direct distribution without "distribution =" prefix
                    attributes.rngDistribution = try parseRNGDistribution()
                    _ = match(.comma)
                } else {
                    break
                }
            }

        case .whileOp:
            attributes.whileRegions = try parseWhileRegions()

        case .ifOp:
            attributes.ifRegions = try parseIfRegions()

        case .reverse:
            if match(.comma) || checkIdentifier("dims") || checkIdentifier("dimensions") {
                if checkIdentifier("dims") {
                    try expectIdentifier("dims")
                } else if checkIdentifier("dimensions") {
                    try expectIdentifier("dimensions")
                }
                try expect(.equal)
                attributes.dimensions = try parseDimensionList()
            }

        case .convolution:
            attributes = try parseConvolutionAttributes()

        case .reduceWindow:
            attributes = try parseReduceWindowAttributes()

        case .batchNormInference, .batchNormTraining, .batchNormGrad:
            attributes = try parseBatchNormAttributes()

        case .fft:
            attributes = try parseFFTAttributes()

        case .sort:
            attributes = try parseSortAttributes()

        case .tan, .logistic, .isFinite, .expm1, .log1p, .cbrt,
             .roundNearestAfz, .roundNearestEven, .popcnt, .real, .imag:
            // No special attributes needed - simple unary ops
            break

        case .shiftLeft, .shiftRightArithmetic, .shiftRightLogical, .complex:
            // Binary ops with no special attributes
            break

        case .uniformQuantize, .uniformDequantize, .bitcastConvert:
            // Type conversion operations
            break

        case .dynamicSlice:
            attributes = try parseDynamicSliceAttributes()

        case .dynamicUpdateSlice, .dynamicPad, .dynamicGather:
            // Dynamic indexing operations - operands contain dynamic info
            break

        case .dynamicReshape, .dynamicBroadcastInDim, .dynamicIota:
            // Dynamic shape operations - output shape from operand
            break

        case .triangularSolve:
            attributes = try parseTriangularSolveAttributes()

        case .cholesky:
            attributes = try parseCholeskyAttributes()

        case .reducePrecision:
            attributes = try parseReducePrecisionAttributes()

        case .rngBitGenerator:
            attributes = try parseRngBitGeneratorAttributes()

        case .selectAndScatter:
            attributes = try parseSelectAndScatterAttributes()

        case .map:
            // Map has a computation region - simplified parsing
            break

        case .customCall:
            // custom_call is handled specially in parseOperationBody via parseCustomCallOperandsAndAttributes()
            break

        default:
            // No special attributes needed
            break
        }

        return attributes
    }

    // MARK: - Convolution Attribute Parsing

    private func parseConvolutionAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("window_strides") {
                try expectIdentifier("window_strides")
                try expect(.equal)
                attributes.windowStrides = try parseArrayAttribute()
            } else if checkIdentifier("padding") {
                try expectIdentifier("padding")
                try expect(.equal)
                attributes.convPadding = try parsePaddingArray()
            } else if checkIdentifier("lhs_dilation") {
                try expectIdentifier("lhs_dilation")
                try expect(.equal)
                attributes.lhsDilation = try parseArrayAttribute()
            } else if checkIdentifier("rhs_dilation") {
                try expectIdentifier("rhs_dilation")
                try expect(.equal)
                attributes.rhsDilation = try parseArrayAttribute()
            } else if checkIdentifier("feature_group_count") {
                try expectIdentifier("feature_group_count")
                try expect(.equal)
                attributes.featureGroupCount = try parseInteger()
            } else if checkIdentifier("batch_group_count") {
                try expectIdentifier("batch_group_count")
                try expect(.equal)
                attributes.batchGroupCount = try parseInteger()
            } else if checkIdentifier("dimension_numbers") || check(.hashIdentifier) {
                attributes.convolutionDimensionNumbers = try parseConvolutionDimensionNumbers()
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    private func parseConvolutionDimensionNumbers() throws -> ConvolutionDimensionNumbers? {
        // Skip the #stablehlo.conv< prefix if present
        if check(.hashIdentifier) {
            advance()
        } else if checkIdentifier("dimension_numbers") {
            try expectIdentifier("dimension_numbers")
            try expect(.equal)
            if check(.hashIdentifier) {
                advance()
            }
        }

        // Skip angle bracket if present
        let hasAngleBrackets = check(.leftAngle)
        if hasAngleBrackets {
            advance()
        }

        // Collect the format string: [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
        // When wrapped in <...>, stop at '>'. Otherwise, the format has exactly
        // 3 closing brackets ']' (input, kernel, output) — stop after the third.
        var formatString = ""
        var closeBracketCount = 0
        while !check(.rightAngle) && !check(.colon) && !check(.eof) {
            let tokenText = currentToken.text
            formatString += tokenText
            if tokenText == "]" {
                closeBracketCount += 1
            }
            advance()
            // Without angle brackets, stop after the 3rd ']' (end of output layout)
            if !hasAngleBrackets && closeBracketCount >= 3 {
                break
            }
        }

        if check(.rightAngle) {
            advance()
        }

        // Parse the dimension format
        // Format: input_layout x kernel_layout -> output_layout
        // Example: [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
        return parseDimensionFormat(formatString)
    }

    /// Parse dimension format string like [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
    private func parseDimensionFormat(_ format: String) -> ConvolutionDimensionNumbers {
        // Default NHWC layout
        var inputBatch = 0
        var inputFeature = 3
        var inputSpatial = [1, 2]
        var kernelInput = 2
        var kernelOutput = 3
        var kernelSpatial = [0, 1]
        var outputBatch = 0
        var outputFeature = 3
        var outputSpatial = [1, 2]

        // Clean up the format string
        let cleaned = format.replacingOccurrences(of: " ", with: "")
                           .replacingOccurrences(of: ",", with: "")

        // Split into parts: input x kernel -> output
        // Try to find the 'x' separator between input and kernel
        let xParts = cleaned.components(separatedBy: "x")
        guard xParts.count >= 2 else {
            // Malformed, return defaults
            return ConvolutionDimensionNumbers(
                inputBatchDimension: inputBatch,
                inputFeatureDimension: inputFeature,
                inputSpatialDimensions: inputSpatial,
                kernelInputFeatureDimension: kernelInput,
                kernelOutputFeatureDimension: kernelOutput,
                kernelSpatialDimensions: kernelSpatial,
                outputBatchDimension: outputBatch,
                outputFeatureDimension: outputFeature,
                outputSpatialDimensions: outputSpatial
            )
        }

        let inputPart = xParts[0]
        let restPart = xParts[1]

        // Split kernel and output by ->
        let arrowParts = restPart.components(separatedBy: "->")
        let kernelPart = arrowParts[0]
        let outputPart = arrowParts.count > 1 ? arrowParts[1] : inputPart

        // Parse each layout part
        (inputBatch, inputFeature, inputSpatial) = parseLayoutPart(inputPart, isKernel: false)
        (kernelInput, kernelOutput, kernelSpatial) = parseKernelLayoutPart(kernelPart)
        (outputBatch, outputFeature, outputSpatial) = parseLayoutPart(outputPart, isKernel: false)

        return ConvolutionDimensionNumbers(
            inputBatchDimension: inputBatch,
            inputFeatureDimension: inputFeature,
            inputSpatialDimensions: inputSpatial,
            kernelInputFeatureDimension: kernelInput,
            kernelOutputFeatureDimension: kernelOutput,
            kernelSpatialDimensions: kernelSpatial,
            outputBatchDimension: outputBatch,
            outputFeatureDimension: outputFeature,
            outputSpatialDimensions: outputSpatial
        )
    }

    /// Parse a layout part like [b, 0, 1, f] for input/output tensors
    /// Returns (batchDim, featureDim, spatialDims)
    private func parseLayoutPart(_ part: String, isKernel: Bool) -> (Int, Int, [Int]) {
        // Remove brackets
        let cleaned = part.replacingOccurrences(of: "[", with: "")
                         .replacingOccurrences(of: "]", with: "")

        var batchDim = 0
        var featureDim = 3
        var spatialDims: [Int] = []

        // Parse each character/token
        var position = 0
        for char in cleaned {
            switch char {
            case "b", "B":
                batchDim = position
            case "f", "F":
                featureDim = position
            case "0"..."9":
                // Spatial dimension numbered 0, 1, 2, etc.
                spatialDims.append(position)
            default:
                continue
            }
            position += 1
        }

        // Ensure spatialDims is sorted
        spatialDims.sort()

        return (batchDim, featureDim, spatialDims)
    }

    /// Parse a kernel layout part like [0, 1, i, o]
    /// Returns (inputFeatureDim, outputFeatureDim, spatialDims)
    private func parseKernelLayoutPart(_ part: String) -> (Int, Int, [Int]) {
        // Remove brackets
        let cleaned = part.replacingOccurrences(of: "[", with: "")
                         .replacingOccurrences(of: "]", with: "")

        var inputDim = 2
        var outputDim = 3
        var spatialDims: [Int] = []

        // Parse each character/token
        var position = 0
        for char in cleaned {
            switch char {
            case "i", "I":
                inputDim = position
            case "o", "O":
                outputDim = position
            case "0"..."9":
                // Spatial dimension
                spatialDims.append(position)
            default:
                continue
            }
            position += 1
        }

        // Ensure spatialDims is sorted by their numeric label, not position
        spatialDims.sort()

        return (inputDim, outputDim, spatialDims)
    }

    private func parseArrayAttribute() throws -> [Int] {
        if checkIdentifier("array") {
            try expectIdentifier("array")
            if check(.leftAngle) {
                advance()
                // Skip type like i64:
                while !check(.colon) && !check(.eof) {
                    advance()
                }
                if check(.colon) {
                    advance()
                }
                // MLIR dense-array syntax lists elements bare (no brackets):
                //   array<i64: 1, 2, 3>  or  array<i64>  (empty).
                // The element list is terminated by the closing '>'.
                var dims: [Int] = []
                if !check(.rightAngle) {
                    repeat {
                        dims.append(try parseInteger())
                    } while match(.comma)
                }
                if check(.rightAngle) {
                    advance()
                }
                return dims
            }
        }
        // Fallback for the bracketed form: [1, 2, 3]
        return try parseDimensionList()
    }

    private func parsePaddingArray() throws -> [[Int]] {
        var result: [[Int]] = []

        if checkIdentifier("dense") || checkKeyword(.dense) {
            // dense<0> or dense<[[0,0],[0,0]]> format.
            // `dense` is tokenized as a keyword (constant literals use it), so
            // accept either spelling — the generic reduce_window form JAX emits
            // for cumsum writes `padding = dense<[[7, 0]]> : tensor<1x2xi64>`.
            advance()
            try expect(.leftAngle)

            if case .integer(_) = currentToken.kind {
                // Simple dense<0> - all zeros
                advance()
                try expect(.rightAngle)
                // Return empty, will use default
                return result
            }

            if check(.leftBracket) {
                // Nested array format
                try expect(.leftBracket)
                repeat {
                    try expect(.leftBracket)
                    var pair: [Int] = []
                    repeat {
                        pair.append(try parseInteger())
                    } while match(.comma)
                    try expect(.rightBracket)
                    result.append(pair)
                } while match(.comma)
                try expect(.rightBracket)
            }

            try expect(.rightAngle)

            // Skip the dense literal's element type if present, e.g.
            //   dense<[[7, 0]]> : tensor<1x2xi64>
            // The `tensor<...>` itself contains angle brackets, so track depth
            // and consume the matching closing '>' rather than stopping at the
            // first one (which would leak a stray '>' into the surrounding
            // attribute dict and silently drop later attributes).
            if check(.colon) {
                advance()
                var angleDepth = 0
                while !check(.eof) {
                    if check(.leftAngle) {
                        angleDepth += 1
                        advance()
                    } else if check(.rightAngle) {
                        if angleDepth == 0 { break }
                        angleDepth -= 1
                        advance()
                    } else if angleDepth == 0 && check(.comma) {
                        break
                    } else {
                        advance()
                    }
                }
            }
        } else if check(.leftBracket) {
            // Bare nested-array form, e.g. padding = [[1, 1], [2, 2]]
            // emitted by StableHLO when convolutions have explicit padding
            // (Flax SAME-padded convs).
            try expect(.leftBracket)
            if check(.rightBracket) {
                advance()
                return result
            }
            repeat {
                try expect(.leftBracket)
                var pair: [Int] = []
                repeat {
                    pair.append(try parseInteger())
                } while match(.comma)
                try expect(.rightBracket)
                result.append(pair)
            } while match(.comma)
            try expect(.rightBracket)
        }

        return result
    }

    // MARK: - Reduce Window Attribute Parsing

    private func parseReduceWindowAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("window_dimensions") {
                try expectIdentifier("window_dimensions")
                try expect(.equal)
                attributes.windowDimensions = try parseArrayAttribute()
            } else if checkIdentifier("window_strides") {
                try expectIdentifier("window_strides")
                try expect(.equal)
                attributes.windowStrides = try parseArrayAttribute()
            } else if checkIdentifier("padding") {
                try expectIdentifier("padding")
                try expect(.equal)
                attributes.convPadding = try parsePaddingArray()
            } else if checkIdentifier("base_dilations") {
                try expectIdentifier("base_dilations")
                try expect(.equal)
                attributes.baseDilations = try parseArrayAttribute()
            } else if checkIdentifier("window_dilations") {
                try expectIdentifier("window_dilations")
                try expect(.equal)
                attributes.windowDilations = try parseArrayAttribute()
            } else if checkIdentifier("applies") {
                // Same as regular reduce
                try expectIdentifier("applies")
                let reductionOp = try parseOperationName()
                if reductionOp.contains("add") {
                    attributes.reductionKind = .sum
                } else if reductionOp.contains("max") {
                    attributes.reductionKind = .max
                } else if reductionOp.contains("min") {
                    attributes.reductionKind = .min
                } else if reductionOp.contains("multiply") {
                    attributes.reductionKind = .product
                } else if reductionOp.contains("and") {
                    attributes.reductionKind = .and
                } else if reductionOp.contains("or") {
                    attributes.reductionKind = .or
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Batch Norm Attribute Parsing

    private func parseBatchNormAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("epsilon") {
                try expectIdentifier("epsilon")
                try expect(.equal)
                if case .float(let val) = currentToken.kind {
                    attributes.epsilon = Float(val)
                    advance()
                } else if case .integer(let val) = currentToken.kind {
                    attributes.epsilon = Float(val)
                    advance()
                }
            } else if checkIdentifier("feature_index") {
                try expectIdentifier("feature_index")
                try expect(.equal)
                attributes.featureIndex = try parseInteger()
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - FFT Attribute Parsing

    private func parseFFTAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("fft_type") {
                try expectIdentifier("fft_type")
                try expect(.equal)
                if check(.hashIdentifier) {
                    advance()  // Skip #stablehlo<fft_type ...>
                }
                if checkIdentifier("FFT") {
                    attributes.fftType = .fft
                    advance()
                } else if checkIdentifier("IFFT") {
                    attributes.fftType = .ifft
                    advance()
                } else if checkIdentifier("RFFT") {
                    attributes.fftType = .rfft
                    advance()
                } else if checkIdentifier("IRFFT") {
                    attributes.fftType = .irfft
                    advance()
                }
                if check(.rightAngle) {
                    advance()
                }
            } else if checkIdentifier("fft_length") {
                try expectIdentifier("fft_length")
                try expect(.equal)
                attributes.fftLength = try parseArrayAttribute()
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Sort Attribute Parsing

    private func parseSortAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        // `dimension = N : i64, is_stable = bool`. The `: i64` type suffix must
        // be consumed here — it is NOT the op's result-type colon (the old
        // `while !check(.colon)` guard stopped on it, leaving `i64,...` to be
        // mis-read as the result type).
        while true {
            if checkIdentifier("dimension") {
                try expectIdentifier("dimension")
                try expect(.equal)
                attributes.axis = try parseInteger()
                // Optional `: i64` type annotation on the dimension attribute.
                // Distinguish from the op's RESULT-type colon in the short form
                // `stablehlo.sort %a, dimension = 0 : tensor<...>` (no comparator
                // region) — only consume when an integer type follows; otherwise
                // leave the colon for the main parser's result-type parsing.
                if check(.colon) {
                    let next = peekNext()
                    let intTypes: Set<String> = ["i1", "i8", "i16", "i32", "i64", "index",
                                                 "ui8", "ui16", "ui32", "ui64"]
                    if next.kind == .identifier, intTypes.contains(next.text) {
                        advance()  // ':'
                        advance()  // 'i64'
                    }
                }
            } else if checkIdentifier("is_stable") {
                try expectIdentifier("is_stable")
                try expect(.equal)
                if checkBool(true) {
                    attributes.isStable = true
                    advance()
                } else if checkBool(false) {
                    attributes.isStable = false
                    advance()
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        // Comparator region `({ ^bb0(...): ...; stablehlo.return %p })`. We do
        // not execute it — only the sort DIRECTION matters. JAX canonicalizes
        // NaN/±0 (FLOAT EQ/NE compares) then does the order compare over the
        // canonicalized values (LT => ascending, GT => descending, TOTALORDER).
        if check(.leftParen) {
            advance()  // consume `(`
            let comparator = try parseRegion()
            var descending = false
            for regionOp in comparator.operations where regionOp.kind == .compare {
                if let dir = regionOp.attributes.comparisonDirection {
                    if dir == .gt || dir == .ge { descending = true }
                    else if dir == .lt || dir == .le { descending = false }
                }
            }
            attributes.sortDescending = descending
            try expect(.rightParen)
        }

        return attributes
    }

    // MARK: - Dynamic Slice Attribute Parsing

    private func parseDynamicSliceAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("slice_sizes") {
                try expectIdentifier("slice_sizes")
                try expect(.equal)
                attributes.dynamicSliceSizes = try parseArrayAttribute()
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Triangular Solve Attribute Parsing

    private func parseTriangularSolveAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("left_side") {
                try expectIdentifier("left_side")
                try expect(.equal)
                if checkBool(true) {
                    attributes.leftSide = true
                    advance()
                } else if checkBool(false) {
                    attributes.leftSide = false
                    advance()
                }
            } else if checkIdentifier("lower") {
                try expectIdentifier("lower")
                try expect(.equal)
                if checkBool(true) {
                    attributes.lower = true
                    advance()
                } else if checkBool(false) {
                    attributes.lower = false
                    advance()
                }
            } else if checkIdentifier("unit_diagonal") {
                try expectIdentifier("unit_diagonal")
                try expect(.equal)
                if checkBool(true) {
                    attributes.unitDiagonal = true
                    advance()
                } else if checkBool(false) {
                    attributes.unitDiagonal = false
                    advance()
                }
            } else if checkIdentifier("transpose_a") {
                try expectIdentifier("transpose_a")
                try expect(.equal)
                // Handle both plain identifier and #stablehlo<transpose ...> format
                if check(.hashIdentifier) {
                    advance() // skip #stablehlo
                    if match(.leftAngle) {
                        // Skip "transpose" keyword if present
                        if checkIdentifier("transpose") { advance() }
                        if checkIdentifier("NO_TRANSPOSE") {
                            attributes.transposeA = .noTranspose
                            advance()
                        } else if checkIdentifier("TRANSPOSE") {
                            attributes.transposeA = .transpose
                            advance()
                        } else if checkIdentifier("ADJOINT") {
                            attributes.transposeA = .adjoint
                            advance()
                        }
                        _ = match(.rightAngle)
                    }
                } else if checkIdentifier("NO_TRANSPOSE") {
                    attributes.transposeA = .noTranspose
                    advance()
                } else if checkIdentifier("TRANSPOSE") {
                    attributes.transposeA = .transpose
                    advance()
                } else if checkIdentifier("ADJOINT") {
                    attributes.transposeA = .adjoint
                    advance()
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Cholesky Attribute Parsing

    private func parseCholeskyAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("lower") {
                try expectIdentifier("lower")
                try expect(.equal)
                if checkBool(true) {
                    attributes.lower = true
                    advance()
                } else if checkBool(false) {
                    attributes.lower = false
                    advance()
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Reduce Precision Attribute Parsing

    private func parseReducePrecisionAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("format") {
                // JAX 0.10 lowers reduce_precision as `format = eNmM`
                // (e.g. e8m7, e5m10) — a single identifier token encoding the
                // target exponent/mantissa widths. Parse the N and M out of it.
                try expectIdentifier("format")
                try expect(.equal)
                guard currentToken.kind == .identifier else {
                    throw ParseError.unexpectedToken(expected: "eNmM format", got: currentToken)
                }
                let (eb, mb) = try parseExponentMantissaFormat(currentToken.text)
                attributes.exponentBits = eb
                attributes.mantissaBits = mb
                advance()
            } else if checkIdentifier("exponent_bits") {
                try expectIdentifier("exponent_bits")
                try expect(.equal)
                attributes.exponentBits = try parseInteger()
            } else if checkIdentifier("mantissa_bits") {
                try expectIdentifier("mantissa_bits")
                try expect(.equal)
                attributes.mantissaBits = try parseInteger()
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    /// Parses an `eNmM` format string (e.g. "e8m7") into (exponentBits, mantissaBits).
    private func parseExponentMantissaFormat(_ text: String) throws -> (Int, Int) {
        guard text.hasPrefix("e"), let mIndex = text.firstIndex(of: "m") else {
            throw ParseError.invalidOperation("Invalid reduce_precision format: \(text)", location: currentToken.location)
        }
        let expPart = text[text.index(after: text.startIndex)..<mIndex]
        let mantPart = text[text.index(after: mIndex)...]
        guard let eb = Int(expPart), let mb = Int(mantPart) else {
            throw ParseError.invalidOperation("Invalid reduce_precision format: \(text)", location: currentToken.location)
        }
        return (eb, mb)
    }

    // MARK: - RNG Bit Generator Attribute Parsing

    private func parseRngBitGeneratorAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("rng_algorithm") {
                try expectIdentifier("rng_algorithm")
                try expect(.equal)
                if check(.hashIdentifier) {
                    advance()  // Skip #rng_algorithm<...>
                }
                if checkIdentifier("DEFAULT") {
                    attributes.rngAlgorithm = .defaultAlgorithm
                    advance()
                } else if checkIdentifier("THREE_FRY") {
                    attributes.rngAlgorithm = .threeFry
                    advance()
                } else if checkIdentifier("PHILOX") {
                    attributes.rngAlgorithm = .philox
                    advance()
                }
                if check(.rightAngle) {
                    advance()
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Select and Scatter Attribute Parsing

    private func parseSelectAndScatterAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        var windowDims: [Int] = []
        var windowStrides: [Int] = []
        var padding: [[Int]]? = nil

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("window_dimensions") {
                try expectIdentifier("window_dimensions")
                try expect(.equal)
                windowDims = try parseArrayAttribute()
            } else if checkIdentifier("window_strides") {
                try expectIdentifier("window_strides")
                try expect(.equal)
                windowStrides = try parseArrayAttribute()
            } else if checkIdentifier("padding") {
                try expectIdentifier("padding")
                try expect(.equal)
                padding = try parsePaddingArray()
            } else if checkIdentifier("select") || checkIdentifier("scatter") {
                // Skip select and scatter regions for now
                while !check(.comma) && !check(.colon) && !check(.eof) {
                    if check(.leftBrace) {
                        var depth = 1
                        advance()
                        while depth > 0 && !check(.eof) {
                            if check(.leftBrace) { depth += 1 }
                            if check(.rightBrace) { depth -= 1 }
                            advance()
                        }
                    } else {
                        advance()
                    }
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        if !windowDims.isEmpty {
            attributes.selectAndScatterDimensionNumbers = SelectAndScatterDimensionNumbers(
                windowDimensions: windowDims,
                windowStrides: windowStrides,
                padding: padding
            )
        }

        return attributes
    }

    // MARK: - Control Flow Region Parsing

    /// Parses while loop regions.
    /// Format: (%init) cond { ... stablehlo.return %pred } do { ... stablehlo.return %val }
    private func parseWhileRegions() throws -> WhileRegions {
        // Parse condition region
        // Format: cond { ^bb(%args): ops... stablehlo.return %pred }
        try expectIdentifier("cond")
        let condRegion = try parseRegion()

        // Parse body region
        // Format: do { ^bb(%args): ops... stablehlo.return %vals }
        skipNewlines()
        try expectIdentifier("do")
        let bodyRegion = try parseRegion()

        return WhileRegions(condition: condRegion, body: bodyRegion)
    }

    /// Parses JAX-emitted while format with inline bindings.
    /// Format: (%iterArg = %val0, %iterArg_1 = %val1) : T0, T1
    ///         cond { ops using %iterArg... } do { ops using %iterArg... }
    /// Returns (operands, attributes, resultType) — the type of the first result.
    private func parseWhileBindingsAndRegions() throws -> ([String], HLOAttributes, TensorType) {
        var operands: [String] = []
        var bindingNames: [String] = []
        var attributes = HLOAttributes()

        // Parse bindings: (%iterArg = %val, ...)
        try expect(.leftParen)
        if !check(.rightParen) {
            repeat {
                let bindingName = try parsePercentIdentifier()
                bindingNames.append(bindingName)
                try expect(.equal)
                let value = try parsePercentIdentifier()
                operands.append(value)
            } while match(.comma)
        }
        try expect(.rightParen)

        // Parse type annotations: : T0, T1, ...
        try expect(.colon)
        var types: [TensorType] = []
        repeat {
            let type = try parseTensorType()
            types.append(type)
        } while match(.comma)

        // Build region arguments from bindings + types
        var regionArgs: [RegionArgument] = []
        for (i, name) in bindingNames.enumerated() {
            let type = i < types.count ? types[i] : types.last ?? TensorType(shape: [], elementType: .float32)
            regionArgs.append(RegionArgument(name: name, type: type))
        }

        skipNewlines()

        // Parse cond region (no ^bb header — uses binding names directly)
        try expectIdentifier("cond")
        let condRegion = try parseRegion(implicitArguments: regionArgs)

        // Parse body region
        skipNewlines()
        try expectIdentifier("do")
        let bodyRegion = try parseRegion(implicitArguments: regionArgs)

        attributes.whileRegions = WhileRegions(condition: condRegion, body: bodyRegion)

        let resultType = types.first ?? TensorType(shape: [], elementType: .float32)
        return (operands, attributes, resultType)
    }

    /// Parses a region block, optionally injecting implicit arguments when no ^bb header is present.
    private func parseRegion(implicitArguments: [RegionArgument]? = nil) throws -> Region {
        try expect(.leftBrace)
        skipNewlines()

        var arguments: [RegionArgument] = []
        var operations: [HLOOperation] = []
        var returnValues: [String] = []

        // Check for block label with arguments: ^bb(%arg0: type, ...)
        if currentToken.text == "^" {
            advance()
            if check(.identifier) { advance() }
            if check(.leftParen) {
                try expect(.leftParen)
                if !check(.rightParen) {
                    repeat {
                        let name = try parsePercentIdentifier()
                        try expect(.colon)
                        let type = try parseTensorType()
                        arguments.append(RegionArgument(name: name, type: type))
                    } while match(.comma)
                }
                try expect(.rightParen)
                _ = match(.colon)
            }
        } else if let implicit = implicitArguments {
            // No ^bb header — use implicit arguments from while bindings
            arguments = implicit
        }

        skipNewlines()

        // Parse operations until stablehlo.return or }
        while !check(.rightBrace) && !check(.eof) {
            if checkIdentifier("stablehlo.return") {
                returnValues = try parseStablehloReturn()
                break
            }
            let op = try parseOperation()
            operations.append(op)
            skipNewlines()
        }

        skipNewlines()
        try expect(.rightBrace)

        return Region(
            arguments: arguments,
            operations: operations,
            returnValues: returnValues
        )
    }

    /// Parses if conditional regions.
    /// Format: then { ... } else { ... }
    private func parseIfRegions() throws -> IfRegions {
        // Parse then branch
        try expectIdentifier("then")
        let thenRegion = try parseRegion()

        // Parse optional else branch
        skipNewlines()
        var elseRegion: Region? = nil
        if checkIdentifier("else") {
            try expectIdentifier("else")
            elseRegion = try parseRegion()
        }

        return IfRegions(thenBranch: thenRegion, elseBranch: elseRegion)
    }

    /// Parses an inline operation region wrapped in parentheses.
    /// Format: ({ ^bb0(%arg0: type, %arg1: type): ops... stablehlo.return %val })
    ///
    /// Used by generic-form operations like scatter and reduce that embed
    /// a computation region between the operand list and the attribute dict.
    /// Returns the computation kind determined by the operation in the region body:
    /// - `stablehlo.return %arg1` (identity) → `.set`
    /// - `stablehlo.add` → `.add`
    /// - `stablehlo.maximum` → `.max`
    /// - `stablehlo.minimum` → `.min`
    /// - `stablehlo.multiply` → `.mul`
    private func parseOperationRegion() throws -> ScatterComputationKind? {
        try expect(.leftParen)
        try expect(.leftBrace)
        skipNewlines()

        var computationKind: ScatterComputationKind? = nil
        lastRegionWasLogAddExp = false

        // Track brace depth to handle nested braces
        var braceDepth = 1

        // Scan through the region body looking for known operations
        while braceDepth > 0 && !check(.eof) {
            if check(.leftBrace) {
                braceDepth += 1
                advance()
            } else if check(.rightBrace) {
                braceDepth -= 1
                if braceDepth == 0 {
                    break
                }
                advance()
            } else if check(.identifier) || check(.percentIdentifier) {
                let text = currentToken.text

                // Check for stablehlo operations that determine computation kind
                if text == "stablehlo.add" || text == "add" {
                    computationKind = .add
                } else if text == "stablehlo.maximum" || text == "maximum" {
                    computationKind = .max
                } else if text == "stablehlo.minimum" || text == "minimum" {
                    computationKind = .min
                } else if text == "stablehlo.multiply" || text == "multiply" {
                    computationKind = .mul
                } else if text == "stablehlo.log_plus_one" || text == "log_plus_one" {
                    // log_plus_one appears only in cumlogsumexp's reduce_window
                    // reducer (a stable log-add-exp), never in a plain scatter/
                    // reduce combiner. Flag it so the reduce_window caller maps
                    // this region to .logAddExp instead of the .max it would
                    // otherwise infer from the leading `maximum` op.
                    lastRegionWasLogAddExp = true
                }
                // If we see stablehlo.return without having found an operation,
                // the region is identity (just returns the update argument)
                advance()
            } else {
                advance()
            }
        }

        // Consume closing } and )
        try expect(.rightBrace)
        try expect(.rightParen)
        skipNewlines()

        // If no computation operation was found, it's an identity/set operation
        // (region just does: stablehlo.return %arg1)
        if computationKind == nil {
            computationKind = .set
        }

        return computationKind
    }

    /// Parses stablehlo.return statement.
    /// Format: stablehlo.return %val1, %val2, ... : types
    private func parseStablehloReturn() throws -> [String] {
        try expectIdentifier("stablehlo.return")

        var values: [String] = []

        // Parse return values
        if check(.percentIdentifier) {
            repeat {
                let value = try parsePercentIdentifier()
                values.append(value)
            } while match(.comma)
        }

        // Skip optional type annotation
        if match(.colon) {
            _ = try parseTensorType()
            while match(.comma) {
                _ = try parseTensorType()
            }
        }

        return values
    }

    // MARK: - Type Parsing

    private func parseTensorType() throws -> TensorType {
        // Handle tuple type (skip and return placeholder — tuples are eliminated during parsing)
        if checkIdentifier("tuple") {
            return try skipTupleType()
        }
        try expectKeyword(.tensor)
        try expect(.leftAngle)

        var shape: [Int] = []
        var elementType: ElementType?

        // Parse shape dimensions and element type
        while !check(.rightAngle) {
            if let et = tryParseElementType() {
                elementType = et
                break
            }

            if case .integer(let dim) = currentToken.kind {
                shape.append(Int(dim))
                advance()
                if check(.identifier) && currentToken.text == "x" {
                    advance()
                }
            } else if currentToken.text == "x" {
                advance()
            } else {
                break
            }
        }

        try expect(.rightAngle)

        guard let et = elementType else {
            throw ParseError.invalidTensorType(
                "Missing element type",
                location: currentToken.location
            )
        }

        return TensorType(shape: shape, elementType: et)
    }

    private func tryParseElementType() -> ElementType? {
        let text = currentToken.text
        let elementType: ElementType?

        switch text {
        case "f16": elementType = .float16
        case "f32": elementType = .float32
        case "f64": elementType = .float64
        case "bf16": elementType = .bfloat16
        case "i1": elementType = .int1
        case "i8": elementType = .int8
        case "i16": elementType = .int16
        case "i32": elementType = .int32
        case "i64": elementType = .int64
        case "ui8": elementType = .uint8
        case "ui16": elementType = .uint16
        case "ui32": elementType = .uint32
        case "ui64": elementType = .uint64
        default: elementType = nil
        }

        if elementType != nil {
            advance()
        }

        return elementType
    }

    private func parseTypeSignature(for kind: HLOOpKind = .add) throws -> TensorType {
        // Handle both simple types: tensor<2x3xf32>
        // and function-like types: (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
        //                      or: (tensor<i32>) -> (tensor<i32>, tensor<i32>)

        if check(.leftParen) {
            // Function-like signature, skip to return type
            try expect(.leftParen)
            var depth = 1
            while depth > 0 && !check(.eof) {
                if check(.leftParen) { depth += 1 }
                if check(.rightParen) { depth -= 1 }
                advance()
            }
            try expect(.arrow)

            // Handle multi-result return: (type1, type2, ...)
            if check(.leftParen) {
                try expect(.leftParen)
                let firstType = try parseTensorType()
                // Skip remaining types
                while match(.comma) {
                    _ = try parseTensorType()
                }
                try expect(.rightParen)
                return firstType
            }

            return try parseTensorType()
        }

        // For select: pred_type, result_type - skip predicate type and get result type
        if kind == .select {
            // Skip the predicate type (e.g., tensor<2x3xi1>)
            _ = try parseTensorType()
            _ = match(.comma)
            // Parse the result type
            return try parseTensorType()
        }

        return try parseTensorType()
    }

    /// Parses the type signature of a multi-input (argmax/argmin) reduce and
    /// returns the (value, index) result types. Format:
    ///   (in_value, in_index, init_value, init_index) -> (out_value, out_index)
    /// The leading operand-type list is skipped; both return types are captured.
    private func parseReduceResultTypes() throws -> (TensorType, TensorType) {
        // Skip the parenthesised operand-type list.
        try expect(.leftParen)
        var depth = 1
        while depth > 0 && !check(.eof) {
            if check(.leftParen) { depth += 1 }
            if check(.rightParen) { depth -= 1 }
            advance()
        }
        try expect(.arrow)
        // Return types: (value_type, index_type).
        try expect(.leftParen)
        let valueType = try parseTensorType()
        try expect(.comma)
        let indexType = try parseTensorType()
        // Skip any further (unexpected) result types.
        while match(.comma) {
            _ = try parseTensorType()
        }
        try expect(.rightParen)
        return (valueType, indexType)
    }

    /// Skips a `tuple<...>` type and returns a placeholder TensorType.
    /// Tuple operations are eliminated during parsing, so the type is not used.
    private func skipTupleType() throws -> TensorType {
        // Skip "tuple" identifier
        if checkIdentifier("tuple") {
            advance()
        }
        // Skip <...> with nested angle brackets
        if check(.leftAngle) {
            advance()
            var depth = 1
            while depth > 0 && !check(.eof) {
                if check(.leftAngle) { depth += 1 }
                if check(.rightAngle) { depth -= 1 }
                advance()
            }
        }
        return TensorType(shape: [], elementType: .int32)
    }

    /// Resolves a value through the alias chain (for tuple elimination).
    private func resolveAlias(_ value: String, aliases: [String: String]) -> String {
        var current = value
        var seen = Set<String>()
        while let alias = aliases[current], !seen.contains(current) {
            seen.insert(current)
            current = alias
        }
        return current
    }

    // MARK: - Attribute Parsing Helpers

    private func parseConstantValue() throws -> ConstantValue {
        try expectKeyword(.dense)
        try expect(.leftAngle)

        // Parse the constant data
        let value: ConstantValue

        if check(.rightAngle) {
            // Empty tensor: dense<> for zero-element tensors
            value = .dense([], TensorType(shape: [0], elementType: .int32))
        } else if case .string(let hexStr) = currentToken.kind {
            // Hex-encoded byte buffer: dense<"0x6F12833A..."> for raw float data
            advance()
            value = .hexBytes(hexStr)
        } else if checkKeyword(.true_) {
            advance()
            value = .scalar(1.0)
        } else if checkKeyword(.false_) {
            advance()
            value = .scalar(0.0)
        } else if check(.leftBracket) {
            // Dense array
            let values = try parseDenseArray()
            // We'll get the type from the type signature later
            value = .dense(values, TensorType(shape: [], elementType: .float32))
        } else if case .float(let f) = currentToken.kind {
            advance()
            value = .scalar(f)
        } else if case .integer(let i) = currentToken.kind {
            // In MLIR, hex integers in dense<> represent float bit patterns
            // (e.g., 0xFF800000 is -inf as f32, not the integer 4286578688)
            let doubleValue = integerToConstantDouble(i, text: currentToken.text)
            advance()
            value = .scalar(doubleValue)
        } else {
            throw ParseError.invalidConstant(
                "Expected constant value",
                location: currentToken.location
            )
        }

        try expect(.rightAngle)
        return value
    }

    /// Converts an integer token to a Double for use in a constant value.
    ///
    /// In MLIR text format, hex integers in `dense<>` represent IEEE 754 float
    /// bit patterns: `0xFF800000` is `-inf` (f32), not the integer 4286578688.
    /// Decimal integers are plain numeric values (e.g., `dense<0>` is 0.0).
    private func integerToConstantDouble(_ value: Int64, text: String) -> Double {
        let isHex = text.hasPrefix("0x") || text.hasPrefix("0X")
            || text.hasPrefix("-0x") || text.hasPrefix("-0X")
        guard isHex else {
            return Double(value)
        }
        // Reinterpret hex value as a float bit pattern.
        // f32 bit patterns fit in 32 bits; f64 patterns use 64 bits.
        let bits = UInt64(bitPattern: value)
        if bits <= UInt64(UInt32.max) {
            return Double(Float(bitPattern: UInt32(bits)))
        } else {
            return Double(bitPattern: bits)
        }
    }

    private func parseDenseArray() throws -> [Double] {
        try expect(.leftBracket)
        var values: [Double] = []

        if !check(.rightBracket) {
            repeat {
                if check(.leftBracket) {
                    // Nested array
                    let nested = try parseDenseArray()
                    values.append(contentsOf: nested)
                } else if case .float(let f) = currentToken.kind {
                    values.append(f)
                    advance()
                } else if case .integer(let i) = currentToken.kind {
                    values.append(integerToConstantDouble(i, text: currentToken.text))
                    advance()
                } else if check(.minus) {
                    advance()
                    if case .float(let f) = currentToken.kind {
                        values.append(-f)
                        advance()
                    } else if case .integer(let i) = currentToken.kind {
                        values.append(Double(-i))
                        advance()
                    }
                }
            } while match(.comma)
        }

        try expect(.rightBracket)
        return values
    }

    private func parseDimensionList() throws -> [Int] {
        try expect(.leftBracket)
        var dims: [Int] = []

        if !check(.rightBracket) {
            repeat {
                dims.append(try parseInteger())
            } while match(.comma)
        }

        try expect(.rightBracket)
        return dims
    }

    private func parseInteger() throws -> Int {
        if case .integer(let value) = currentToken.kind {
            advance()
            return Int(value)
        }
        throw ParseError.unexpectedToken(expected: "integer", got: currentToken)
    }

    private func parseReduceAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        // Parse: applies stablehlo.add across dimensions = [1]
        // or other reduce patterns

        if checkIdentifier("applies") {
            try expectIdentifier("applies")
            let reductionOp = try parseOperationName()

            if reductionOp.contains("add") {
                attributes.reductionKind = .sum
            } else if reductionOp.contains("max") {
                attributes.reductionKind = .max
            } else if reductionOp.contains("min") {
                attributes.reductionKind = .min
            } else if reductionOp.contains("multiply") {
                attributes.reductionKind = .product
            } else if reductionOp.contains("and") {
                attributes.reductionKind = .and
            } else if reductionOp.contains("or") {
                attributes.reductionKind = .or
            }

            if checkIdentifier("across") {
                try expectIdentifier("across")
                try expectIdentifier("dimensions")
                try expect(.equal)
                attributes.dimensions = try parseDimensionList()
            }
        }

        return attributes
    }

    private func parseDotDimensionNumbers() throws -> DotDimensionNumbers? {
        var lhsBatching: [Int] = []
        var rhsBatching: [Int] = []
        var lhsContracting: [Int] = []
        var rhsContracting: [Int] = []

        // Handle verbose format: #stablehlo.dot<lhs_batching_dimensions = [], ...>
        if check(.hashIdentifier) && currentToken.text == "#stablehlo.dot" {
            advance()
            try expect(.leftAngle)

            while !check(.rightAngle) {
                if checkIdentifier("lhs_batching_dimensions") {
                    try expectIdentifier("lhs_batching_dimensions")
                    try expect(.equal)
                    lhsBatching = try parseDimensionList()
                } else if checkIdentifier("rhs_batching_dimensions") {
                    try expectIdentifier("rhs_batching_dimensions")
                    try expect(.equal)
                    rhsBatching = try parseDimensionList()
                } else if checkIdentifier("lhs_contracting_dimensions") {
                    try expectIdentifier("lhs_contracting_dimensions")
                    try expect(.equal)
                    lhsContracting = try parseDimensionList()
                } else if checkIdentifier("rhs_contracting_dimensions") {
                    try expectIdentifier("rhs_contracting_dimensions")
                    try expect(.equal)
                    rhsContracting = try parseDimensionList()
                }
                _ = match(.comma)
            }

            try expect(.rightAngle)
        }
        // Handle simple format: contracting_dims = [1] x [0]
        // and optionally: batching_dims = [0] x [0]
        else if checkIdentifier("contracting_dims") || checkIdentifier("batching_dims") {
            // Parse optional batching_dims = [lhs] x [rhs]
            if checkIdentifier("batching_dims") {
                try expectIdentifier("batching_dims")
                try expect(.equal)
                lhsBatching = try parseDimensionList()
                try expectIdentifier("x")
                rhsBatching = try parseDimensionList()
                _ = match(.comma)
            }

            // Parse contracting_dims = [lhs] x [rhs]
            if checkIdentifier("contracting_dims") {
                try expectIdentifier("contracting_dims")
                try expect(.equal)
                lhsContracting = try parseDimensionList()
                try expectIdentifier("x")
                rhsContracting = try parseDimensionList()
            }
        } else {
            // No dimension numbers provided
            return nil
        }

        return DotDimensionNumbers(
            lhsBatchingDimensions: lhsBatching,
            rhsBatchingDimensions: rhsBatching,
            lhsContractingDimensions: lhsContracting,
            rhsContractingDimensions: rhsContracting
        )
    }

    private func parseGatherDimensionNumbers() throws -> GatherDimensionNumbers? {
        // Parse gather dimension numbers in StableHLO MLIR format:
        // dimension_numbers = #stablehlo.gather<
        //   offset_dims = [...],
        //   collapsed_slice_dims = [...],
        //   start_index_map = [...],
        //   index_vector_dim = N
        // >,
        // slice_sizes = array<i64: ...>,
        // indices_are_sorted = false

        var offsetDims: [Int] = []
        var collapsedSliceDims: [Int] = []
        var startIndexMap: [Int] = []
        var indexVectorDim: Int = 0
        var sliceSizes: [Int] = []
        var operandBatchingDims: [Int] = []
        var startIndicesBatchingDims: [Int] = []

        while !check(.colon) && !check(.eof) {
            skipNewlines()

            if checkIdentifier("dimension_numbers") {
                // Parse: dimension_numbers = #stablehlo.gather<...>
                try expectIdentifier("dimension_numbers")
                try expect(.equal)

                // Skip #stablehlo.gather or similar prefix
                if check(.hashIdentifier) {
                    advance()
                }

                // Parse the inner content between <...> or {...}
                let useBraces = check(.leftBrace)
                let openToken: TokenKind = useBraces ? .leftBrace : .leftAngle
                let closeToken: TokenKind = useBraces ? .rightBrace : .rightAngle
                if match(openToken) {
                    while !check(closeToken) && !check(.eof) {
                        skipNewlines()

                        if checkIdentifier("offset_dims") {
                            try expectIdentifier("offset_dims")
                            try expect(.equal)
                            offsetDims = try parseDimensionList()
                        } else if checkIdentifier("collapsed_slice_dims") {
                            try expectIdentifier("collapsed_slice_dims")
                            try expect(.equal)
                            collapsedSliceDims = try parseDimensionList()
                        } else if checkIdentifier("start_index_map") {
                            try expectIdentifier("start_index_map")
                            try expect(.equal)
                            startIndexMap = try parseDimensionList()
                        } else if checkIdentifier("index_vector_dim") {
                            try expectIdentifier("index_vector_dim")
                            try expect(.equal)
                            indexVectorDim = try parseInteger()
                        } else if checkIdentifier("operand_batching_dims") {
                            try expectIdentifier("operand_batching_dims")
                            try expect(.equal)
                            operandBatchingDims = try parseDimensionList()
                        } else if checkIdentifier("start_indices_batching_dims") {
                            try expectIdentifier("start_indices_batching_dims")
                            try expect(.equal)
                            startIndicesBatchingDims = try parseDimensionList()
                        } else {
                            // Skip unknown tokens
                            advance()
                        }
                        _ = match(.comma)
                        skipNewlines()
                    }
                    _ = match(closeToken)
                }
            } else if checkIdentifier("slice_sizes") {
                // Parse: slice_sizes = array<i64: 1, 3>
                try expectIdentifier("slice_sizes")
                try expect(.equal)

                // Handle array<i64: ...> format
                if checkIdentifier("array") {
                    try expectIdentifier("array")
                    if match(.leftAngle) {
                        // Skip type like "i64:"
                        while !check(.colon) && !check(.rightAngle) && !check(.eof) {
                            advance()
                        }
                        _ = match(.colon)

                        // Parse the integers
                        var sizes: [Int] = []
                        while !check(.rightAngle) && !check(.eof) {
                            if case .integer(let val) = currentToken.kind {
                                sizes.append(Int(val))
                                advance()
                            } else if check(.minus) {
                                advance()
                                if case .integer(let val) = currentToken.kind {
                                    sizes.append(-Int(val))
                                    advance()
                                }
                            } else {
                                _ = match(.comma)
                            }
                        }
                        _ = match(.rightAngle)
                        sliceSizes = sizes
                    }
                } else {
                    sliceSizes = try parseDimensionList()
                }
            } else if checkIdentifier("offset_dims") {
                // Fallback: flat format without dimension_numbers wrapper
                try expectIdentifier("offset_dims")
                try expect(.equal)
                offsetDims = try parseDimensionList()
            } else if checkIdentifier("collapsed_slice_dims") {
                try expectIdentifier("collapsed_slice_dims")
                try expect(.equal)
                collapsedSliceDims = try parseDimensionList()
            } else if checkIdentifier("start_index_map") {
                try expectIdentifier("start_index_map")
                try expect(.equal)
                startIndexMap = try parseDimensionList()
            } else if checkIdentifier("index_vector_dim") {
                try expectIdentifier("index_vector_dim")
                try expect(.equal)
                indexVectorDim = try parseInteger()
            } else if checkIdentifier("operand_batching_dims") {
                try expectIdentifier("operand_batching_dims")
                try expect(.equal)
                operandBatchingDims = try parseDimensionList()
            } else if checkIdentifier("start_indices_batching_dims") {
                try expectIdentifier("start_indices_batching_dims")
                try expect(.equal)
                startIndicesBatchingDims = try parseDimensionList()
            } else if checkIdentifier("slice_sizes") {
                try expectIdentifier("slice_sizes")
                try expect(.equal)
                sliceSizes = try parseDimensionList()
            } else if checkIdentifier("indices_are_sorted") {
                // Skip this attribute
                try expectIdentifier("indices_are_sorted")
                try expect(.equal)
                // Skip true/false (tokenized as keywords, not identifiers)
                if checkKeyword(.true_) || checkKeyword(.false_) {
                    advance()
                }
            } else {
                // Skip unknown tokens
                break
            }
            _ = match(.comma)
            skipNewlines()
        }

        // Only return if we parsed the essential fields
        guard !sliceSizes.isEmpty else {
            return nil
        }

        return GatherDimensionNumbers(
            offsetDims: offsetDims,
            collapsedSliceDims: collapsedSliceDims,
            startIndexMap: startIndexMap,
            indexVectorDim: indexVectorDim,
            sliceSizes: sliceSizes,
            operandBatchingDims: operandBatchingDims,
            startIndicesBatchingDims: startIndicesBatchingDims
        )
    }

    private func parseScatterAttributes(stopAtOuterBrace: Bool = false) throws -> (ScatterDimensionNumbers?, ScatterComputationKind?) {
        // Parse scatter attributes in inline format:
        // update_window_dims = [...], inserted_window_dims = [...],
        // scatter_dims_to_operand_dims = [...], index_vector_dim = N,
        // input_batching_dims = [...], scatter_indices_batching_dims = [...],
        // computation = add/max/min/mul
        //
        // `stopAtOuterBrace`: when the dimension numbers sit inside a caller-owned
        // attribute block — `"stablehlo.scatter"(...) ({region}) { <here> } : sig`,
        // the generic-form path consumes that outer `{`/`}` itself — this parser
        // must NOT consume the closing `}`. Without this it eats the block's `}`
        // and the caller then over-runs the type signature and the func body,
        // surfacing as a spurious "expected colon, got '}'" at the module close.

        var updateWindowDims: [Int] = []
        var insertedWindowDims: [Int] = []
        var scatterDimsToOperandDims: [Int] = []
        var indexVectorDim: Int = 0
        var inputBatchingDims: [Int] = []
        var scatterIndicesBatchingDims: [Int] = []
        var computationKind: ScatterComputationKind? = nil
        // Depth of brace-wrapped dimension numbers (the `{attrs}` regex-conversion
        // form) opened inside this parser; their `}` is ours to consume, the
        // caller-owned outer block's `}` is not.
        var innerBraceDepth = 0

        while !check(.colon) && !check(.eof) && !check(.leftParen) {
            // Handle #stablehlo.scatter<...> wrapper format:
            // scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = ..., ...>
            if checkIdentifier("scatter_dimension_numbers") {
                try expectIdentifier("scatter_dimension_numbers")
                try expect(.equal)
                // Handle multiple formats:
                // 1. #stablehlo.scatter<attrs> (hash identifier + angle brackets)
                // 2. {attrs} (brace-wrapped, from our regex conversion)
                if check(.hashIdentifier) {
                    advance() // skip #stablehlo.scatter
                }
                if check(.leftAngle) {
                    advance() // skip <
                    skipNewlines()
                } else if check(.leftBrace) {
                    advance() // skip {
                    innerBraceDepth += 1
                    skipNewlines()
                }
                continue
            } else if checkIdentifier("indices_are_sorted") || checkIdentifier("unique_indices") {
                // Skip boolean attributes not needed for compilation
                advance() // name
                if match(.equal) {
                    advance() // value (true/false)
                }
                _ = match(.comma)
                skipNewlines()
                continue
            } else if check(.rightBrace) {
                // A `}` closing a brace-wrapped dimension-number block is ours.
                // A `}` we did not open is the caller's outer attribute block —
                // stop and leave it for the caller to consume.
                if innerBraceDepth > 0 {
                    innerBraceDepth -= 1
                    advance()
                    _ = match(.comma)
                    skipNewlines()
                    continue
                }
                if stopAtOuterBrace {
                    break
                }
                advance()
                _ = match(.comma)
                skipNewlines()
                continue
            } else if check(.rightAngle) {
                // End of #stablehlo.scatter<...> angle-bracket block.
                advance()
                _ = match(.comma)
                skipNewlines()
                continue
            } else if checkIdentifier("update_window_dims") {
                try expectIdentifier("update_window_dims")
                try expect(.equal)
                updateWindowDims = try parseDimensionList()
            } else if checkIdentifier("inserted_window_dims") {
                try expectIdentifier("inserted_window_dims")
                try expect(.equal)
                insertedWindowDims = try parseDimensionList()
            } else if checkIdentifier("scatter_dims_to_operand_dims") {
                try expectIdentifier("scatter_dims_to_operand_dims")
                try expect(.equal)
                scatterDimsToOperandDims = try parseDimensionList()
            } else if checkIdentifier("index_vector_dim") {
                try expectIdentifier("index_vector_dim")
                try expect(.equal)
                indexVectorDim = try parseInteger()
            } else if checkIdentifier("input_batching_dims") {
                try expectIdentifier("input_batching_dims")
                try expect(.equal)
                inputBatchingDims = try parseDimensionList()
            } else if checkIdentifier("scatter_indices_batching_dims") {
                try expectIdentifier("scatter_indices_batching_dims")
                try expect(.equal)
                scatterIndicesBatchingDims = try parseDimensionList()
            } else if checkIdentifier("computation") {
                try expectIdentifier("computation")
                try expect(.equal)
                // Parse computation kind: add, max, min, mul
                if checkIdentifier("add") {
                    try expectIdentifier("add")
                    computationKind = .add
                } else if checkIdentifier("max") {
                    try expectIdentifier("max")
                    computationKind = .max
                } else if checkIdentifier("min") {
                    try expectIdentifier("min")
                    computationKind = .min
                } else if checkIdentifier("mul") {
                    try expectIdentifier("mul")
                    computationKind = .mul
                } else if checkIdentifier("set") {
                    try expectIdentifier("set")
                    computationKind = .set
                }
            } else {
                // Skip unknown tokens
                break
            }
            _ = match(.comma)
        }

        // Return even if some fields are empty (they have defaults)
        let dimNumbers = ScatterDimensionNumbers(
            updateWindowDims: updateWindowDims,
            insertedWindowDims: insertedWindowDims,
            scatterDimsToOperandDims: scatterDimsToOperandDims,
            indexVectorDim: indexVectorDim,
            inputBatchingDims: inputBatchingDims,
            scatterIndicesBatchingDims: scatterIndicesBatchingDims
        )
        return (dimNumbers, computationKind)
    }

    private func parseComparisonDirection() throws -> ComparisonDirection? {
        let directions = ["EQ", "NE", "LT", "LE", "GT", "GE"]
        for dir in directions {
            if checkIdentifier(dir) {
                advance()
                _ = match(.comma)
                return ComparisonDirection(rawValue: dir)
            }
        }
        return nil
    }

    private func parseSliceAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        // Parse slice notation: [start:end:stride, ...]
        if check(.leftBracket) {
            advance()
            var starts: [Int] = []
            var limits: [Int] = []
            var strides: [Int] = []

            repeat {
                let start = try parseInteger()
                starts.append(start)
                try expect(.colon)
                let limit = try parseInteger()
                limits.append(limit)
                if match(.colon) {
                    let stride = try parseInteger()
                    strides.append(stride)
                } else {
                    strides.append(1)
                }
            } while match(.comma)

            try expect(.rightBracket)

            attributes.sliceStarts = starts
            attributes.sliceLimits = limits
            attributes.sliceStrides = strides
        }

        return attributes
    }

    private func parsePadAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        // Parse: low = [...], high = [...], interior = [...]
        while !check(.colon) && !check(.eof) {
            if checkIdentifier("low") {
                try expectIdentifier("low")
                try expect(.equal)
                attributes.padLow = try parseDimensionList()
            } else if checkIdentifier("high") {
                try expectIdentifier("high")
                try expect(.equal)
                attributes.padHigh = try parseDimensionList()
            } else if checkIdentifier("interior") {
                try expectIdentifier("interior")
                try expect(.equal)
                attributes.padInterior = try parseDimensionList()
            }
            _ = match(.comma)
        }

        return attributes
    }

    private func parseRNGDistribution() throws -> RNGDistribution? {
        if checkIdentifier("UNIFORM") {
            advance()
            return .uniform
        } else if checkIdentifier("NORMAL") {
            advance()
            return .normal
        }
        return nil
    }

    // MARK: - Token Helpers

    private func advance() {
        if let peeked = peekedToken {
            currentToken = peeked
            peekedToken = nil
        } else {
            currentToken = lexer.nextToken()
        }
        // Skip newlines automatically
        while currentToken.kind == .newline {
            currentToken = lexer.nextToken()
        }
    }

    private func skipNewlines() {
        while currentToken.kind == .newline {
            currentToken = lexer.nextToken()
        }
    }

    /// Returns the token after `currentToken` without consuming it.
    private func peekNext() -> Token {
        if peekedToken == nil {
            var t = lexer.nextToken()
            while t.kind == .newline { t = lexer.nextToken() }
            peekedToken = t
        }
        return peekedToken!
    }

    private func check(_ kind: TokenKind) -> Bool {
        currentToken.kind == kind
    }

    private func checkKeyword(_ keyword: Keyword) -> Bool {
        if case .keyword(let kw) = currentToken.kind {
            return kw == keyword
        }
        return false
    }

    private func checkIdentifier(_ name: String) -> Bool {
        currentToken.kind == .identifier && currentToken.text == name
    }

    /// Check for true/false which are tokenized as keywords, not identifiers.
    private func checkBool(_ value: Bool) -> Bool {
        if value {
            return checkKeyword(.true_) || (currentToken.kind == .identifier && currentToken.text == "true")
        } else {
            return checkKeyword(.false_) || (currentToken.kind == .identifier && currentToken.text == "false")
        }
    }

    private func match(_ kind: TokenKind) -> Bool {
        if check(kind) {
            advance()
            return true
        }
        return false
    }

    private func expect(_ kind: TokenKind) throws {
        if !match(kind) {
            throw ParseError.unexpectedToken(expected: "\(kind)", got: currentToken)
        }
    }

    private func expectKeyword(_ keyword: Keyword) throws {
        if !checkKeyword(keyword) {
            throw ParseError.unexpectedToken(expected: keyword.rawValue, got: currentToken)
        }
        advance()
    }

    private func expectIdentifier(_ name: String) throws {
        if !checkIdentifier(name) {
            throw ParseError.unexpectedToken(expected: name, got: currentToken)
        }
        advance()
    }

    private func parseAtIdentifier() throws -> String {
        guard case .atIdentifier = currentToken.kind else {
            throw ParseError.unexpectedToken(expected: "@identifier", got: currentToken)
        }
        let name = String(currentToken.text.dropFirst())  // Remove @
        advance()
        return name
    }

    private func parsePercentIdentifier() throws -> String {
        guard case .percentIdentifier = currentToken.kind else {
            throw ParseError.unexpectedToken(expected: "%identifier", got: currentToken)
        }
        var name = currentToken.text
        advance()

        // Handle element access syntax: %name#N → %name.N
        // Used by multi-result operations (while, call) to reference individual outputs.
        // The compiler stores multi-result outputs as "%name.0", "%name.1", etc.
        if case .hashIdentifier = currentToken.kind {
            let indexStr = String(currentToken.text.dropFirst())  // Remove '#'
            name = "\(name).\(indexStr)"
            advance()
        }

        return name
    }
}
