// ProducerConsumerFusion.swift
// MetalHLOCore
//
// Producer-consumer fusion pass that eliminates intermediate memory writes.
// Inspired by XLA's GpuInstructionFusion pass.

import Foundation

// MARK: - Fusion Region

/// A region of operations that can be fused into a single kernel.
public struct FusionRegion: Sendable {
    /// The operations in this fusion region, in topological order.
    public let operations: [HLOOperation]

    /// Indices of the operations in the original function.
    public let indices: Set<Int>

    /// The root operation (output of the fusion).
    public let rootOperation: HLOOperation

    /// External inputs to the fusion region.
    public let inputs: [String]

    /// Whether this region should be emitted as a custom fused operation.
    public let shouldEmitAsCustomCall: Bool

    public init(
        operations: [HLOOperation],
        indices: Set<Int>,
        rootOperation: HLOOperation,
        inputs: [String],
        shouldEmitAsCustomCall: Bool = false
    ) {
        self.operations = operations
        self.indices = indices
        self.rootOperation = rootOperation
        self.inputs = inputs
        self.shouldEmitAsCustomCall = shouldEmitAsCustomCall
    }

    /// The number of operations in this region.
    public var size: Int { operations.count }

    /// Whether this region contains more than one operation.
    public var isFused: Bool { operations.count > 1 }
}

// MARK: - Producer-Consumer Fusion Pass

/// Producer-consumer fusion pass that groups operations to eliminate
/// intermediate memory writes.
///
/// The pass works by:
/// 1. Processing operations in reverse order (consumers before producers)
/// 2. For each consumer, trying to fuse its producers into it
/// 3. Building fusion regions that can be executed as single kernels
///
/// Fusion is allowed when:
/// - The producer has a single use (the consumer)
/// - The producer is a "fusible" operation (elementwise, shape ops)
/// - The fusion won't exceed the maximum region size
/// - The operation isn't on the "unfusible" list (convolution, fft, etc.)
public final class ProducerConsumerFusion: @unchecked Sendable {

    /// Maximum number of operations in a fusion region.
    private let maxFusionSize: Int

    /// Whether to emit fused regions as custom calls.
    private let emitCustomCalls: Bool

    /// Creates a producer-consumer fusion pass.
    ///
    /// - Parameters:
    ///   - maxFusionSize: Maximum operations in a fusion region (default: 50).
    ///   - emitCustomCalls: Whether to emit fused regions as custom_call ops (default: false).
    public init(maxFusionSize: Int = 50, emitCustomCalls: Bool = false) {
        self.maxFusionSize = maxFusionSize
        self.emitCustomCalls = emitCustomCalls
    }

    /// Performs producer-consumer fusion on a function.
    ///
    /// - Parameter function: The function to optimize.
    /// - Returns: The optimized function with fused operations.
    public func fuse(_ function: HLOFunction) -> HLOFunction {
        // Build use/def information
        let useDefInfo = UseDefInfo(function: function)

        // Track which operations have been assigned to a region
        var assigned: Set<Int> = []

        // Build fusion regions in reverse order
        var regions: [FusionRegion] = []

        for (index, _) in function.operations.enumerated().reversed() {
            // Skip if already assigned to a region
            if assigned.contains(index) { continue }

            // Build a fusion region starting from this operation
            let region = buildFusionRegion(
                rootIndex: index,
                function: function,
                useDefInfo: useDefInfo,
                assigned: &assigned
            )

            regions.append(region)
        }

        // Reverse to get topological order
        regions.reverse()

        // Emit the optimized function
        return emitFusedFunction(function: function, regions: regions)
    }

    // MARK: - Region Building

    /// Builds a fusion region starting from a root operation.
    private func buildFusionRegion(
        rootIndex: Int,
        function: HLOFunction,
        useDefInfo: UseDefInfo,
        assigned: inout Set<Int>
    ) -> FusionRegion {
        let rootOp = function.operations[rootIndex]

        // Start with just the root operation
        var regionOps: [(Int, HLOOperation)] = [(rootIndex, rootOp)]
        var regionIndices: Set<Int> = [rootIndex]
        var externalInputs: Set<String> = []

        // Worklist of operations to consider fusing
        var worklist: [(Int, HLOOperation)] = []

        // Add producers of root to worklist
        for operand in rootOp.operands {
            if let (producerOp, producerIndex) = useDefInfo.definingOp(for: operand) {
                worklist.append((producerIndex, producerOp))
            } else {
                // External input (function argument or from outside region)
                externalInputs.insert(operand)
            }
        }

        // Greedily fuse producers
        while !worklist.isEmpty && regionOps.count < maxFusionSize {
            let (producerIndex, producerOp) = worklist.removeFirst()

            // Skip if already in region or assigned elsewhere
            if regionIndices.contains(producerIndex) { continue }
            if assigned.contains(producerIndex) {
                // Producer is in another region - this is an external input
                externalInputs.insert(producerOp.result)
                continue
            }

            // Check if producer can be fused
            if canFuse(producer: producerOp, producerIndex: producerIndex,
                       intoRegion: regionIndices, useDefInfo: useDefInfo) {

                // Add producer to region
                regionOps.append((producerIndex, producerOp))
                regionIndices.insert(producerIndex)

                // For transpose / broadcastInDim, freeze the operand as
                // external — the chain codegen reads these via a computed
                // index (permuted for transpose, broadcast-aware for
                // broadcast) that's a function of `tid` and the source's
                // shape. If the operand were itself a chain op, the prior
                // op's value-per-thread was computed for the *chain's*
                // output coord, not the transpose/broadcast input coord,
                // so reading it at a different coord would require either
                // re-evaluating the prior op or staging in threadgroup
                // memory. Neither is supported here.
                if producerOp.kind == .transpose || producerOp.kind == .broadcastInDim {
                    for operand in producerOp.operands {
                        externalInputs.insert(operand)
                    }
                    continue
                }

                // Add producer's operands to worklist
                for operand in producerOp.operands {
                    if let (grandProducer, grandIndex) = useDefInfo.definingOp(for: operand) {
                        if !regionIndices.contains(grandIndex) {
                            worklist.append((grandIndex, grandProducer))
                        }
                    } else {
                        externalInputs.insert(operand)
                    }
                }
            } else {
                // Can't fuse - mark as external input
                externalInputs.insert(producerOp.result)
            }
        }

        // Sort operations by index to get topological order
        regionOps.sort { $0.0 < $1.0 }
        let orderedOps = regionOps.map { $0.1 }

        // Determine external inputs in a consistent order
        let sortedInputs = computeOrderedInputs(
            operations: orderedOps,
            regionIndices: regionIndices,
            function: function
        )

        // Only emit as custom_call if all ops are supported by the
        // elementwise-chain kernel emitter (reshape counts as a no-op,
        // broadcastInDim and transpose use computed indices).
        //
        // The old `hasLinearDataFlow` constraint is gone — the new chain
        // serialization records per-op operand sources (external slot vs
        // prior chain-op result), so the codegen can express arbitrary
        // elementwise DAGs, not just strict left-to-right linear chains.
        let allElementwise = orderedOps.allSatisfy { isElementwiseOp($0.kind) }
        // Safety for fused select: its predicate (operands[0]) must be produced
        // IN this chain (a prior compare → float register), never an external
        // bool buffer (which the float-typed chain would misread). If a select's
        // pred is external, don't emit this region as a fused chain.
        let regionResultSet = Set(orderedOps.map { $0.result })
        let unsafeSelect = orderedOps.contains { op in
            op.kind == .select && (op.operands.isEmpty || !regionResultSet.contains(op.operands[0]))
        }
        // Safety for fused transpose / non-scalar broadcastInDim: the chain
        // codegen reads these via a per-thread COMPUTED INDEX into the source,
        // which only works when the source is an EXTERNAL buffer (a fresh
        // input the kernel can re-index freely). When the operand is a prior
        // chain result the codegen has no way to re-index it — that value was
        // computed for the chain's OUTPUT coordinate, not the transposed /
        // broadcast input coordinate — so it silently passes the prior through
        // unchanged (identity), which is wrong for any non-trivial permutation
        // or fan-out (e.g. `transpose(exp(x))`, `broadcast(exp(x))` → garbage).
        // Reshape is exempt: it's the identity in linear-index space, so the
        // passthrough is correct. If a transpose/broadcast has an in-region
        // operand, don't emit this region as a fused chain.
        let unsafeShapeOp = orderedOps.contains { op in
            (op.kind == .transpose || op.kind == .broadcastInDim)
                && !op.operands.isEmpty && regionResultSet.contains(op.operands[0])
        }
        let canEmitAsCustomCall = emitCustomCalls && orderedOps.count > 1
            && allElementwise && !unsafeSelect && !unsafeShapeOp

        // Mark this region's ops as assigned so the outer loop doesn't
        // revisit them. There's ONE exception: when the region will emit
        // as a customCall and a duplicatable producer (e.g. a scalar
        // broadcast) has consumers outside this region, leave the
        // duplicatable producer un-assigned. That way the outer loop will
        // still visit it as a root (emitting the standalone form for the
        // non-chain consumers) and other regions can also inline their
        // own copy. If every consumer ends up inlining the op, the
        // standalone becomes dead code and `final-dce` removes it.
        var assignedThisRegion = regionIndices
        if canEmitAsCustomCall {
            for (idx, op) in regionOps where idx != rootIndex {
                let isBroadcast = op.kind == .broadcastInDim
                guard isCheapToDuplicate(op.kind) || isBroadcast else { continue }
                let uses = useDefInfo.uses(of: op.result)
                let hasOutsideUse = uses.contains(where: { !regionIndices.contains($0) })
                if hasOutsideUse {
                    assignedThisRegion.remove(idx)
                }
            }
        }
        assigned.formUnion(assignedThisRegion)

        return FusionRegion(
            operations: orderedOps,
            indices: regionIndices,
            rootOperation: rootOp,
            inputs: sortedInputs,
            shouldEmitAsCustomCall: canEmitAsCustomCall
        )
    }

    /// Determines if a producer can be fused into the current region.
    private func canFuse(
        producer: HLOOperation,
        producerIndex: Int,
        intoRegion regionIndices: Set<Int>,
        useDefInfo: UseDefInfo
    ) -> Bool {
        // Check if operation type is fusible
        guard isFusibleOp(producer.kind) else { return false }

        // Only pull in producers the chain codegen can actually EMIT. The
        // region is emitted as a fused chain solely when every op satisfies
        // `isElementwiseOp` (see `allElementwise` in buildFusionRegion). A
        // producer that is "fusible" but not chain-emittable (select, compare,
        // clamp, convert, bitwise, …) used to get pulled in and then poison the
        // whole region — `allElementwise` went false and EVERY op fell back to
        // standalone. Stopping growth at that boundary lets the elementwise
        // portion fuse instead. Opt out with METALHLO_FUSE_STRICT=0.
        let strictFusion = ProcessInfo.processInfo.environment["METALHLO_FUSE_STRICT"] != "0"
        if strictFusion {
            guard isChainEmittable(producer.kind) else { return false }
        }

        // Check if producer has single use (in the region)
        // Multi-use producers would require code duplication
        let uses = useDefInfo.uses(of: producer.result)
        let usesInRegion = uses.filter { regionIndices.contains($0) }
        let usesOutsideRegion = uses.filter { !regionIndices.contains($0) }

        // Allow fusion only if:
        // 1. All uses are in the region, OR
        // 2. It's a "cheap" operation that can be duplicated, OR
        // 3. It's a broadcastInDim — duplicating just costs one extra
        //    Metal load per chain (the underlying input stays shared). For
        //    scalar broadcasts the load is `input[0]`; for higher-rank
        //    broadcasts it's a computed index. Either way the data isn't
        //    materialised as a full tensor inside the kernel.
        if !usesOutsideRegion.isEmpty {
            let isBroadcast = producer.kind == .broadcastInDim
            guard isCheapToDuplicate(producer.kind) || isBroadcast else { return false }
        }

        // Don't fuse if producer has no uses in region (dead code)
        guard !usesInRegion.isEmpty else { return false }

        return true
    }

    /// Computes the ordered list of external inputs for a fusion region.
    private func computeOrderedInputs(
        operations: [HLOOperation],
        regionIndices: Set<Int>,
        function: HLOFunction
    ) -> [String] {
        var inputs: [String] = []
        var seen: Set<String> = []

        // Results produced within the region
        let regionResults = Set(operations.map { $0.result })

        // Go through operations in order, collecting external inputs
        for op in operations {
            for operand in op.operands {
                if !regionResults.contains(operand) && !seen.contains(operand) {
                    inputs.append(operand)
                    seen.insert(operand)
                }
            }
        }

        return inputs
    }

    // MARK: - Fusibility Checks

    /// Operations that are fusible (can be inlined into a kernel).
    private func isFusibleOp(_ kind: HLOOpKind) -> Bool {
        switch kind {
        // Elementwise arithmetic
        case .add, .subtract, .multiply, .divide, .maximum, .minimum, .power:
            return true

        // Unary operations
        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt,
             .tanh, .logistic, .sine, .cosine, .tan,
             .floor, .ceil, .sign, .expm1, .log1p, .cbrt,
             .roundNearestAfz, .roundNearestEven:
            return true

        // Shape operations (zero cost in fused context)
        // Note: broadcastInDim is NOT fusible because the consumer kernel
        // needs to be broadcast-aware, which our kernels don't support yet
        case .reshape, .transpose:
            return true

        // Type conversion
        case .convert, .bitcastConvert:
            return true

        // Comparison and selection
        case .compare, .select, .clamp:
            return true

        // Bitwise operations
        case .not, .and, .or, .xor, .shiftLeft, .shiftRightArithmetic, .shiftRightLogical:
            return true

        // Constants should NOT be fused - they need separate MTLBuffers
        // and must be preserved for constant buffer extraction
        case .constant:
            return false

        // Broadcast is fusible *only* when the source is rank-0 (a scalar).
        // The chain kernel reads each input slot as `inputN[tid]` or
        // `inputN[0]` based on its element count, so a scalar broadcast
        // turns into a single load reused across all output elements
        // without any broadcast-aware indexing. `canFuse()` checks the
        // attribute and rejects non-scalar broadcasts; the duplicated-op
        // tracking in `buildFusionRegion()` keeps the standalone form
        // alive when there are non-chain consumers, and `final-dce`
        // prunes the standalone when every consumer inlined its own copy.
        case .broadcastInDim:
            return true

        // Operations that should NOT be fused (expensive, library calls, complex memory access)
        case .dot, .dotGeneral, .convolution:
            return false  // Use MPS library calls

        case .reduce, .reduceWindow:
            return false  // Complex control flow

        case .fft, .sort:
            return false  // Specialized algorithms

        case .gather, .scatter, .dynamicGather:
            return false  // Non-uniform memory access

        case .slice, .dynamicSlice, .dynamicUpdateSlice:
            return false  // Complex indexing

        case .pad, .dynamicPad:
            return false  // Boundary handling

        case .concatenate:
            return false  // Memory layout changes

        case .triangularSolve, .cholesky:
            return false  // Linear algebra library

        case .batchNormInference, .batchNormTraining, .batchNormGrad:
            return false  // Use fused implementation

        case .customCall:
            return false  // Already fused

        default:
            return false
        }
    }

    /// Operations that are cheap enough to duplicate if they have multiple uses.
    /// NOTE: tried adding binary arithmetic (multiply / add / subtract /
    /// divide / compare / select) here, but it triggered the
    /// `assignedThisRegion.remove(idx)` re-visit path for nearly every
    /// op that any chain absorbed, breaking the regions into tiny
    /// pieces — 302 chains collapsed to 68 and the total op count went
    /// 1339 → 1647. The re-visit path is structurally designed for
    /// genuinely free duplications (reshape / transpose / negate); binary
    /// arithmetic needs a different fusion entry point.
    private func isCheapToDuplicate(_ kind: HLOOpKind) -> Bool {
        switch kind {
        // Zero-cost operations
        // Note: broadcastInDim is NOT cheap to duplicate since fusing it requires
        // broadcast-aware kernels which we don't generate yet
        case .reshape, .transpose:
            return true

        // Constants should not be duplicated - they should be shared via constant buffers
        case .constant:
            return false

        // Cheap unary ops
        case .negate, .abs, .not:
            return true

        default:
            return false
        }
    }

    /// Operations that the fused-elementwise codegen can express. With the
    /// SSA-DAG kernel emitter, this is a superset of the old "binary +
    /// unary arithmetic" list: reshape is a no-op in linearized indexing,
    /// and broadcast_in_dim of a scalar input compiles to a `[0]` read.
    /// Non-scalar broadcasts are still gated out at `canFuse` and at the
    /// region-level `onlyScalarBroadcasts` check.
    /// Ops the FusedElementwiseChain codegen can actually emit into a chain.
    /// = the `isElementwiseOp` set (which gates region emission) plus the few
    /// binary ops `elementwiseChainExpr` handles but `isElementwiseOp` omits.
    private func isChainEmittable(_ kind: HLOOpKind) -> Bool {
        switch kind {
        case .power, .remainder:
            return true
        default:
            return isElementwiseOp(kind)
        }
    }

    private func isElementwiseOp(_ kind: HLOOpKind) -> Bool {
        switch kind {
        // Binary elementwise operations
        case .add, .subtract, .multiply, .divide, .maximum, .minimum:
            return true

        // Unary elementwise operations
        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt,
             .tanh, .logistic, .sine, .cosine, .floor, .ceil:
            return true

        // Reshape: no-op in linear-index space (chain kernel walks tid).
        // BroadcastInDim (scalar only — gated in canFuse): the chain kernel
        // detects 1-element inputs and reads `input[0]`.
        // Transpose: the chain kernel computes a per-thread permuted index
        // from `tid` and the transpose's permutation attribute (carried in
        // `FusedElementwiseChain.Op.permutation`). Only supported when the
        // transpose's input is a fresh external — gated in canFuse so a
        // transpose-of-prior-result doesn't slip through.
        case .reshape, .broadcastInDim, .transpose:
            return true

        // Compare/select/clamp (opt-in METALHLO_FUSE_SELECT): the chain emits
        // compare as a float predicate (1/0) and select reads it in-register,
        // so the softmax-mask `broadcast→compare→select` collapses into one
        // chain instead of stranding the producers. canFuse guards that a
        // select's predicate is an in-chain compare (no bool-buffer reads).
        case .compare, .select, .clamp:
            return fuseSelect

        // Everything else is not supported by FusedElementwiseHandler
        default:
            return false
        }
    }

    /// Fuse compare/select/clamp into elementwise chains. On by default —
    /// collapses softmax-mask (broadcast→compare→select) and erf/gelu
    /// polynomial guards into single kernels. Opt out with METALHLO_FUSE_SELECT=0.
    private var fuseSelect: Bool {
        ProcessInfo.processInfo.environment["METALHLO_FUSE_SELECT"] != "0"
    }

    // MARK: - Code Emission

    /// Emits the optimized function with fusion regions.
    private func emitFusedFunction(
        function: HLOFunction,
        regions: [FusionRegion]
    ) -> HLOFunction {
        var newOperations: [HLOOperation] = []

        for region in regions {
            if region.shouldEmitAsCustomCall && region.isFused {
                // Emit as a single custom call
                let fusedOp = emitFusedCustomCall(region: region)
                newOperations.append(fusedOp)
            } else {
                // Emit operations as-is (MPSGraph will stitch them)
                // But we need to update operands to reflect any changes
                newOperations.append(contentsOf: region.operations)
            }
        }

        // Repair dangling references. When a shared sub-expression (e.g. the
        // causal mask: iota → compare → select) is partially absorbed across
        // multiple consumer chains, one chain can inline a producer as an
        // internal `prior` value, dropping its standalone definition while a
        // *different* chain still references that SSA name as an external
        // input. The result is an operand that is read but never produced.
        // Detect those and re-materialise their full producer cone from the
        // pre-fusion ops, inserted in original (topological) order.
        newOperations = repairDanglingRefs(newOperations, original: function)

        return HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: newOperations,
            returnValues: function.returnValues
        )
    }

    /// Re-materialises any operand that is referenced by the fused op list but
    /// is neither a function input nor produced by some op in that list. The
    /// missing definitions are pulled from `original.operations` (which still
    /// holds the un-fused producer cone) and inserted ahead of their uses.
    private func repairDanglingRefs(
        _ ops: [HLOOperation],
        original: HLOFunction
    ) -> [HLOOperation] {
        let inputNames = Set(original.inputs.map { $0.name })
        var defined = Set(ops.map { $0.result })

        // Collect operands that are read but undefined.
        var missing: [String] = []
        var missingSet = Set<String>()
        for op in ops {
            for operand in op.operands where !defined.contains(operand)
                && !inputNames.contains(operand) && !missingSet.contains(operand) {
                missing.append(operand)
                missingSet.insert(operand)
            }
        }
        guard !missing.isEmpty else { return ops }

        // Index the original (pre-fusion) ops by result and by position.
        var originalByResult: [String: (Int, HLOOperation)] = [:]
        for (idx, op) in original.operations.enumerated() {
            originalByResult[op.result] = (idx, op)
        }

        // BFS back through the original producer cone for each missing name.
        var needed: [Int: HLOOperation] = [:]   // original index → op
        var worklist = missing
        while let name = worklist.popLast() {
            if defined.contains(name) || inputNames.contains(name) { continue }
            guard let (idx, op) = originalByResult[name] else { continue }
            if needed[idx] != nil { continue }
            needed[idx] = op
            defined.insert(name)
            for operand in op.operands { worklist.append(operand) }
        }
        guard !needed.isEmpty else { return ops }

        // Merge the re-materialised cone back in WITHOUT disturbing the
        // existing schedule. The existing `ops` are already a valid order, and
        // reordering them relative to each other is unsafe (operand edges do
        // not capture write-after-write / write-after-read aliasing on shared
        // buffers). So emit `ops` in their exact original sequence; just before
        // the first op that references a restored value, splice that value's
        // restored producer cone in (restored-internal deps first). A restored
        // op's non-restored deps (iota / constants) are early existing ops and
        // are therefore already emitted by the time their user is reached.
        let restoredMap = Dictionary(needed.values.map { ($0.result, $0) },
                                     uniquingKeysWith: { a, _ in a })
        var emitted = Set<String>()
        var result: [HLOOperation] = []
        result.reserveCapacity(restoredMap.count + ops.count)
        func emitRestored(_ name: String) {
            guard let rop = restoredMap[name], !emitted.contains(name) else { return }
            emitted.insert(name)
            for operand in rop.operands { emitRestored(operand) }
            result.append(rop)
        }
        for op in ops {
            for operand in op.operands { emitRestored(operand) }
            if !emitted.contains(op.result) {
                emitted.insert(op.result)
                result.append(op)
            }
        }
        // Safety: any restored op never referenced (shouldn't happen) appended.
        for rop in needed.keys.sorted().map({ needed[$0]! }) where !emitted.contains(rop.result) {
            result.append(rop)
        }
        return result
    }

    /// Emits a fused region as a custom call operation.
    ///
    /// Encodes the chain as a list of `{k, o}` entries — `k` is the op kind
    /// raw value, `o` is the list of operand sources (`["e", idx]` for an
    /// external input, `["p", idx]` for a prior chain-op result). This is
    /// the new SSA-style format that lets the codegen emit one temporary
    /// per op and support arbitrary DAGs (operands from arbitrary earlier
    /// results), not just strict linear chains.
    private func emitFusedCustomCall(region: FusionRegion) -> HLOOperation {
        var attributes = HLOAttributes()
        attributes.callTargetName = "fused_elementwise"

        // Map: SSA name → chain-op index (for `.prior(k)` references).
        // External inputs map: SSA name → external-input index.
        var priorIndex: [String: Int] = [:]
        var externalIndex: [String: Int] = [:]
        for (i, name) in region.inputs.enumerated() {
            externalIndex[name] = i
        }

        var chainArr: [[String: Any]] = []
        chainArr.reserveCapacity(region.operations.count)
        for (i, op) in region.operations.enumerated() {
            var operands: [[Any]] = []
            operands.reserveCapacity(op.operands.count)
            for operand in op.operands {
                if let p = priorIndex[operand] {
                    operands.append(["p", p])
                } else if let e = externalIndex[operand] {
                    operands.append(["e", e])
                } else {
                    // Operand isn't in either map — shouldn't happen for a
                    // well-formed region, but fall back to treating it as a
                    // fresh external slot appended to the inputs list. This
                    // would only fire if the region builder forgot to record
                    // an external; defensive code, not load-bearing.
                    let newIdx = externalIndex.count
                    externalIndex[operand] = newIdx
                    operands.append(["e", newIdx])
                }
            }
            var entry: [String: Any] = [
                "k": op.kind.rawValue,
                "o": operands
            ]
            // Carry the dimension attribute (perm for transpose,
            // broadcast_dimensions for broadcastInDim).
            if (op.kind == .transpose || op.kind == .broadcastInDim),
               let dims = op.attributes.dimensions {
                entry["t"] = dims
            }
            // Carry the broadcast's output shape — it's encoded in the
            // result type and not derivable from input + dims (size-1 dims
            // can fan out to any output size).
            if op.kind == .broadcastInDim {
                entry["s"] = op.resultType.shape
            }
            // Carry the comparison direction so the chain emits the right op.
            if op.kind == .compare, let dir = op.attributes.comparisonDirection {
                entry["c"] = dir.rawValue
            }
            chainArr.append(entry)
            priorIndex[op.result] = i
        }

        let configDict: [String: Any] = ["chain": chainArr]
        if let data = try? JSONSerialization.data(withJSONObject: configDict, options: []),
           let str = String(data: data, encoding: .utf8) {
            attributes.backendConfig = str
        }

        return HLOOperation(
            result: region.rootOperation.result,
            kind: .customCall,
            operands: region.inputs,
            resultType: region.rootOperation.resultType,
            attributes: attributes
        )
    }
}

// MARK: - Reduce Fusion

/// Fuses a single-use pointwise producer into its consuming `reduce`, emitting
/// a `fused_reduce` custom_call (single-pass layernorm/softmax pattern). The
/// producer may be a single elementwise op or an existing `fused_elementwise`
/// chain — both must be purely pointwise (no transpose/broadcast). Opt-in via
/// METALHLO_FUSE_REDUCE=1; the reduce's init is baked from the reduction kind,
/// so only standard-identity reduces are fused.
public struct ReduceFusion: Sendable {
    public init() {}

    private static func reductionKindString(_ k: ReductionKind) -> String? {
        switch k {
        case .sum: return "sum"
        case .max: return "max"
        case .min: return "min"
        case .mean: return "mean"
        case .product: return "product"
        case .and, .or: return nil  // not supported by the fused kernel
        case .logAddExp: return nil // window-only (cumlogsumexp); not a fusable reduce
        }
    }

    /// Pointwise ops the fused-reduce kernel can emit per element.
    private static func isPointwise(_ kind: HLOOpKind) -> Bool {
        switch kind {
        case .add, .subtract, .multiply, .divide, .remainder, .maximum, .minimum, .power,
             .negate, .abs, .exponential, .log, .sqrt, .rsqrt, .tanh, .logistic,
             .sine, .cosine, .floor, .ceil:
            return true
        default:
            return false
        }
    }

    public func fuse(_ function: HLOFunction) -> HLOFunction {
        // Default-on (validated: nanoGPT loss-neutral, ~3% faster). Disable with
        // METALHLO_FUSE_REDUCE=0. NOTE: the kernel bakes the reduction identity
        // as the init value, so this is only correct for standard-identity
        // reduces (init = 0/1/±inf). Transformer reduces always use the
        // identity; harden with an init-constant check before relying on it
        // for arbitrary reduces.
        guard ProcessInfo.processInfo.environment["METALHLO_FUSE_REDUCE"] != "0" else {
            return function
        }
        let udi = UseDefInfo(function: function)
        let debug = ProcessInfo.processInfo.environment["METALHLO_DEBUG_REDUCE_FUSION"] == "1"
        var dbgTotal = 0, dbgNoKind = 0, dbgNoProducer = 0, dbgMultiUse = 0, dbgNotChain = 0, dbgOK = 0

        // Decide, per reduce op, whether to fuse and what producer to absorb.
        var fusedProducerResults: Set<String> = []   // producers folded away
        var replacement: [Int: HLOOperation] = [:]    // reduce index → fused_reduce op

        for (idx, op) in function.operations.enumerated() {
            guard op.kind == .reduce else { continue }
            dbgTotal += 1
            guard let rk = op.attributes.reductionKind,
                  let rkStr = ReduceFusion.reductionKindString(rk) else { dbgNoKind += 1; continue }
            guard let dataOperand = op.operands.first,
                  let (producer, _) = udi.definingOp(for: dataOperand) else { dbgNoProducer += 1; continue }
            guard udi.uses(of: producer.result).count == 1 else { dbgMultiUse += 1; continue }

            // Build a pointwise chain + external operand list from the producer.
            guard let (chain, externals) = Self.chainFromProducer(producer) else { dbgNotChain += 1; continue }
            dbgOK += 1

            let inShape = producer.resultType.shape       // pre-reduction shape
            let outShape = op.resultType.shape
            let reduceDims = op.attributes.dimensions ?? []
            guard !reduceDims.isEmpty, !inShape.isEmpty else { continue }

            // Serialize config: chain (same {k,o,t,s} format) + reduce metadata.
            let chainArr = Self.serializeChain(chain)
            let configDict: [String: Any] = [
                "chain": chainArr, "rk": rkStr,
                "rd": reduceDims, "is": inShape, "os": outShape
            ]
            guard let data = try? JSONSerialization.data(withJSONObject: configDict),
                  let str = String(data: data, encoding: .utf8) else { continue }

            var attrs = HLOAttributes()
            attrs.callTargetName = "fused_reduce"
            attrs.backendConfig = str
            let fused = HLOOperation(
                result: op.result, kind: .customCall,
                operands: externals, resultType: op.resultType, attributes: attrs)
            replacement[idx] = fused
            fusedProducerResults.insert(producer.result)
        }

        if debug {
            FileHandle.standardError.write("[reduce-fusion] reduces=\(dbgTotal) noKind=\(dbgNoKind) noProducer=\(dbgNoProducer) multiUse=\(dbgMultiUse) notPointwiseChain=\(dbgNotChain) fused=\(replacement.count) (passedGates=\(dbgOK))\n".data(using: .utf8)!)
        }
        guard !replacement.isEmpty else { return function }

        // Emit: drop folded producers (single-use into the reduce), swap each
        // reduce for its fused_reduce.
        var newOps: [HLOOperation] = []
        newOps.reserveCapacity(function.operations.count)
        for (idx, op) in function.operations.enumerated() {
            if fusedProducerResults.contains(op.result) { continue }
            newOps.append(replacement[idx] ?? op)
        }

        return HLOFunction(
            name: function.name, inputs: function.inputs,
            outputTypes: function.outputTypes, operations: newOps,
            returnValues: function.returnValues)
    }

    /// Builds a pointwise chain + ordered external operands from a producer
    /// that is either a single elementwise op or a fused_elementwise chain.
    private static func chainFromProducer(_ producer: HLOOperation) -> (FusedElementwiseChain, [String])? {
        if producer.kind == .customCall,
           producer.attributes.callTargetName == "fused_elementwise",
           let cfg = producer.attributes.backendConfig,
           let chain = parseChainJSON(cfg) {
            // Must be purely pointwise (no transpose/broadcast/reshape).
            guard chain.ops.allSatisfy({ isPointwise($0.kind) }) else { return nil }
            return (chain, producer.operands)
        }
        if isPointwise(producer.kind) {
            let ops = (0..<producer.operands.count).map { FusedElementwiseChain.OperandSource.external($0) }
            let chain = FusedElementwiseChain(ops: [
                FusedElementwiseChain.Op(kind: producer.kind, operands: ops)
            ])
            return (chain, producer.operands)
        }
        return nil
    }

    private static func parseChainJSON(_ cfg: String) -> FusedElementwiseChain? {
        guard let data = cfg.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let arr = json["chain"] as? [[String: Any]] else { return nil }
        var ops: [FusedElementwiseChain.Op] = []
        for d in arr {
            guard let kindStr = d["k"] as? String, let kind = HLOOpKind(rawValue: kindStr),
                  let oArr = d["o"] as? [[Any]] else { return nil }
            var operands: [FusedElementwiseChain.OperandSource] = []
            for s in oArr {
                guard s.count == 2, let tag = s[0] as? String,
                      let i = (s[1] as? Int) ?? (s[1] as? NSNumber).map({ $0.intValue }) else { return nil }
                operands.append(tag == "e" ? .external(i) : .prior(i))
            }
            ops.append(FusedElementwiseChain.Op(kind: kind, operands: operands))
        }
        return ops.isEmpty ? nil : FusedElementwiseChain(ops: ops)
    }

    private static func serializeChain(_ chain: FusedElementwiseChain) -> [[String: Any]] {
        chain.ops.map { op in
            let o: [[Any]] = op.operands.map {
                switch $0 {
                case .external(let i): return ["e", i]
                case .prior(let i):    return ["p", i]
                }
            }
            return ["k": op.kind.rawValue, "o": o]
        }
    }
}

// MARK: - Use-Def Information

/// Tracks use-def relationships for operations in a function.
public struct UseDefInfo: Sendable {
    /// Map from value name to defining operation and index.
    private let defMap: [String: (op: HLOOperation, index: Int)]

    /// Map from value name to indices of operations that use it.
    private let useMap: [String: [Int]]

    /// Creates use-def information for a function.
    public init(function: HLOFunction) {
        var defs: [String: (op: HLOOperation, index: Int)] = [:]
        var uses: [String: [Int]] = [:]

        for (index, op) in function.operations.enumerated() {
            // Record definition
            defs[op.result] = (op, index)

            // Record uses
            for operand in op.operands {
                uses[operand, default: []].append(index)
            }
        }

        // Also record uses in return values (treated as special "use")
        for (index, retVal) in function.returnValues.enumerated() {
            // Use a special index to indicate return value use
            uses[retVal, default: []].append(Int.max - index)
        }

        self.defMap = defs
        self.useMap = uses
    }

    /// Gets the operation that defines a value.
    public func definingOp(for value: String) -> (op: HLOOperation, index: Int)? {
        return defMap[value]
    }

    /// Gets the indices of operations that use a value.
    public func uses(of value: String) -> [Int] {
        return useMap[value] ?? []
    }

    /// Checks if a value has a single use (excluding return values).
    public func hasSingleUse(_ value: String) -> Bool {
        let allUses = uses(of: value)
        // Filter out return value uses (marked with Int.max - index)
        let nonReturnUses = allUses.filter { $0 < Int.max - 1000 }
        return nonReturnUses.count == 1
    }

    /// Checks if a value is used in a return statement.
    public func isReturnValue(_ value: String) -> Bool {
        let allUses = uses(of: value)
        return allUses.contains { $0 >= Int.max - 1000 }
    }
}

// MARK: - Fusion Statistics

/// Statistics about fusion pass results.
public struct FusionStatistics: Sendable {
    /// Number of fusion regions created.
    public let numRegions: Int

    /// Number of fused regions (with more than one operation).
    public let numFusedRegions: Int

    /// Total operations before fusion.
    public let totalOpsBefore: Int

    /// Total operations after fusion.
    public let totalOpsAfter: Int

    /// Average region size.
    public var averageRegionSize: Double {
        guard numRegions > 0 else { return 0 }
        return Double(totalOpsBefore) / Double(numRegions)
    }

    /// Fusion rate (percentage of operations fused).
    public var fusionRate: Double {
        guard totalOpsBefore > 0 else { return 0 }
        let fusedOps = totalOpsBefore - totalOpsAfter
        return Double(fusedOps) / Double(totalOpsBefore) * 100
    }
}

extension ProducerConsumerFusion {
    /// Computes statistics for fusion results.
    public func computeStatistics(
        before: HLOFunction,
        after: HLOFunction,
        regions: [FusionRegion]
    ) -> FusionStatistics {
        return FusionStatistics(
            numRegions: regions.count,
            numFusedRegions: regions.filter { $0.isFused }.count,
            totalOpsBefore: before.operations.count,
            totalOpsAfter: after.operations.count
        )
    }
}
