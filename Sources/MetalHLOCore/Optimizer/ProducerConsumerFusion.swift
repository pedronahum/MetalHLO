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

        // Mark all region operations as assigned
        assigned.formUnion(regionIndices)

        // Sort operations by index to get topological order
        regionOps.sort { $0.0 < $1.0 }
        let orderedOps = regionOps.map { $0.1 }

        // Determine external inputs in a consistent order
        let sortedInputs = computeOrderedInputs(
            operations: orderedOps,
            regionIndices: regionIndices,
            function: function
        )

        // Only emit as custom_call if all operations are supported by FusedElementwiseHandler
        // Shape ops like reshape, transpose, broadcastInDim are "fusible" for producer-consumer
        // analysis but are NOT supported as fused_elementwise custom_calls
        let canEmitAsCustomCall = emitCustomCalls && orderedOps.count > 1 &&
            orderedOps.allSatisfy { isElementwiseOp($0.kind) }

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

        // Check if producer has single use (in the region)
        // Multi-use producers would require code duplication
        let uses = useDefInfo.uses(of: producer.result)
        let usesInRegion = uses.filter { regionIndices.contains($0) }
        let usesOutsideRegion = uses.filter { !regionIndices.contains($0) }

        // Allow fusion only if:
        // 1. All uses are in the region, OR
        // 2. It's a "cheap" operation that can be duplicated
        if !usesOutsideRegion.isEmpty {
            guard isCheapToDuplicate(producer.kind) else { return false }
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

        // Broadcast operations should NOT be fused because consumer kernels
        // would need to be broadcast-aware, which our kernels don't support yet
        case .broadcastInDim:
            return false

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

    /// Operations that are true elementwise operations supported by FusedElementwiseHandler.
    /// This is more restrictive than isFusibleOp - it excludes shape ops like reshape,
    /// transpose, and broadcastInDim which are fusible for analysis but not supported
    /// as fused_elementwise custom_calls.
    private func isElementwiseOp(_ kind: HLOOpKind) -> Bool {
        switch kind {
        // Binary elementwise operations
        case .add, .subtract, .multiply, .divide, .maximum, .minimum:
            return true

        // Unary elementwise operations
        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt,
             .tanh, .logistic, .sine, .cosine, .floor, .ceil:
            return true

        // Shape operations are NOT elementwise - they should not be in fused_elementwise
        case .reshape, .transpose, .broadcastInDim:
            return false

        // Everything else is not supported by FusedElementwiseHandler
        default:
            return false
        }
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

        return HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: newOperations,
            returnValues: function.returnValues
        )
    }

    /// Emits a fused region as a custom call operation.
    private func emitFusedCustomCall(region: FusionRegion) -> HLOOperation {
        var attributes = HLOAttributes()
        attributes.callTargetName = "fused_elementwise"

        // Encode the fused operations in backend config
        let opKinds = region.operations.map { $0.kind.rawValue }
        let configDict: [String: Any] = [
            "operations": opKinds,
            "num_ops": region.operations.count
        ]

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
