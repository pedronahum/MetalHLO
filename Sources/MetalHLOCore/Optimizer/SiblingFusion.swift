// SiblingFusion.swift
// MetalHLOCore
//
// Sibling fusion pass that fuses operations sharing the same producer.
// Inspired by XLA's GpuMultiOutputFusion pass.

import Foundation

// MARK: - Multi-Output Fusion Region

/// A fusion region with multiple outputs (from sibling operations).
public struct MultiOutputFusionRegion: Sendable {
    /// The shared producer operation (input to all siblings).
    public let sharedProducer: HLOOperation

    /// Index of the shared producer in the original function.
    public let producerIndex: Int

    /// The sibling operations that consume the shared producer.
    public let siblings: [HLOOperation]

    /// Indices of the sibling operations in the original function.
    public let siblingIndices: [Int]

    /// All external inputs (including the shared producer's inputs).
    public let inputs: [String]

    /// All outputs from the siblings.
    public var outputs: [String] {
        siblings.map { $0.result }
    }

    /// Number of siblings in this region.
    public var siblingCount: Int { siblings.count }

    /// Total operations (producer + siblings).
    public var totalOperations: Int { 1 + siblings.count }

    public init(
        sharedProducer: HLOOperation,
        producerIndex: Int,
        siblings: [HLOOperation],
        siblingIndices: [Int],
        inputs: [String]
    ) {
        self.sharedProducer = sharedProducer
        self.producerIndex = producerIndex
        self.siblings = siblings
        self.siblingIndices = siblingIndices
        self.inputs = inputs
    }
}

// MARK: - Sibling Fusion Pass

/// Sibling fusion pass that groups operations sharing the same input.
///
/// This pass finds opportunities where multiple operations read from the same
/// producer. By fusing them into a multi-output kernel:
/// - The shared input is read once from memory
/// - Multiple outputs are computed in parallel
/// - Memory bandwidth is significantly reduced
///
/// Example:
/// ```
/// Before:                    After:
///     ┌───┐                     ┌──────────────┐
///     │ A │                     │ MultiOutput  │
///     └─┬─┘                     │   Fusion     │
///    ┌──┴──┐                    │  (A → B,C)   │
///    ▼     ▼                    └──────────────┘
/// ┌───┐ ┌───┐         ═══>          │
/// │ B │ │ C │                   ┌───┴───┐
/// └───┘ └───┘                   ▼       ▼
///                            B_out   C_out
/// ```
public final class SiblingFusion: @unchecked Sendable {

    /// Maximum number of siblings to fuse together.
    private let maxSiblings: Int

    /// Maximum combined register pressure for fused siblings.
    private let maxRegisterPressure: Int

    /// Whether to emit fused regions as custom calls.
    private let emitCustomCalls: Bool

    /// Cost model for fusion decisions (optional).
    private let costModel: FusionHeuristics?

    /// Creates a sibling fusion pass.
    ///
    /// - Parameters:
    ///   - maxSiblings: Maximum siblings per fusion (default: 8).
    ///   - maxRegisterPressure: Maximum register pressure (default: 128).
    ///   - emitCustomCalls: Whether to emit as custom_call (default: false).
    ///   - costModel: Optional cost model for intelligent decisions.
    public init(
        maxSiblings: Int = 8,
        maxRegisterPressure: Int = 128,
        emitCustomCalls: Bool = false,
        costModel: FusionHeuristics? = nil
    ) {
        self.maxSiblings = maxSiblings
        self.maxRegisterPressure = maxRegisterPressure
        self.emitCustomCalls = emitCustomCalls
        self.costModel = costModel
    }

    /// Performs sibling fusion on a function.
    ///
    /// - Parameter function: The function to optimize.
    /// - Returns: The optimized function with sibling fusions.
    public func fuse(_ function: HLOFunction) -> HLOFunction {
        // Build producer → consumers map
        let consumerMap = buildConsumerMap(function: function)

        // Build use-def info
        let useDefInfo = UseDefInfo(function: function)

        // Find sibling groups
        let siblingGroups = findSiblingGroups(
            consumerMap: consumerMap,
            function: function,
            useDefInfo: useDefInfo
        )

        // If no sibling groups, return unchanged
        guard !siblingGroups.isEmpty else {
            return function
        }

        // Create multi-output fusion regions
        let regions = createFusionRegions(
            siblingGroups: siblingGroups,
            function: function,
            useDefInfo: useDefInfo
        )

        // Emit the optimized function
        return emitFusedFunction(function: function, regions: regions)
    }

    // MARK: - Consumer Map Building

    /// Builds a map from producer result to consumer operations.
    private func buildConsumerMap(function: HLOFunction) -> [String: [(op: HLOOperation, index: Int)]] {
        var consumers: [String: [(op: HLOOperation, index: Int)]] = [:]

        for (index, op) in function.operations.enumerated() {
            for operand in op.operands {
                consumers[operand, default: []].append((op, index))
            }
        }

        return consumers
    }

    // MARK: - Sibling Group Finding

    /// A group of sibling operations that share a producer.
    private struct SiblingGroup {
        let producerValue: String
        let producerOp: HLOOperation?
        let producerIndex: Int?
        let siblings: [(op: HLOOperation, index: Int)]
    }

    /// Finds groups of siblings that can potentially be fused.
    private func findSiblingGroups(
        consumerMap: [String: [(op: HLOOperation, index: Int)]],
        function: HLOFunction,
        useDefInfo: UseDefInfo
    ) -> [SiblingGroup] {
        var groups: [SiblingGroup] = []

        for (producerValue, consumers) in consumerMap {
            // Need at least 2 consumers to form a sibling group
            guard consumers.count >= 2 else { continue }

            // Get the producer operation (if it exists in this function)
            let producerInfo = useDefInfo.definingOp(for: producerValue)

            // Filter to fusible siblings
            let fusibleSiblings = consumers.filter { consumer in
                isFusibleAsSibling(consumer.op)
            }

            // Need at least 2 fusible siblings
            guard fusibleSiblings.count >= 2 else { continue }

            groups.append(SiblingGroup(
                producerValue: producerValue,
                producerOp: producerInfo?.op,
                producerIndex: producerInfo?.index,
                siblings: fusibleSiblings
            ))
        }

        // Sort by sibling count (prefer larger groups)
        return groups.sorted { $0.siblings.count > $1.siblings.count }
    }

    // MARK: - Fusion Region Creation

    /// Creates multi-output fusion regions from sibling groups.
    private func createFusionRegions(
        siblingGroups: [SiblingGroup],
        function: HLOFunction,
        useDefInfo: UseDefInfo
    ) -> [MultiOutputFusionRegion] {
        var regions: [MultiOutputFusionRegion] = []
        var processedSiblings: Set<Int> = []

        for group in siblingGroups {
            // Skip if producer is not in this function (external input)
            guard let producerOp = group.producerOp,
                  let producerIndex = group.producerIndex else {
                continue
            }

            // Check if producer is fusible
            guard canFuseProducer(producerOp) else { continue }

            // Select siblings that can be fused together
            var selectedSiblings: [(op: HLOOperation, index: Int)] = []

            for sibling in group.siblings {
                // Skip if already processed
                if processedSiblings.contains(sibling.index) { continue }

                // Check if this sibling can join the group
                if canAddSibling(sibling.op, to: selectedSiblings, sharedProducer: producerOp) {
                    selectedSiblings.append(sibling)
                    processedSiblings.insert(sibling.index)

                    // Limit the number of siblings
                    if selectedSiblings.count >= maxSiblings { break }
                }
            }

            // Create region if we have at least 2 siblings
            if selectedSiblings.count >= 2 {
                // Compute external inputs
                let externalInputs = computeExternalInputs(
                    producer: producerOp,
                    siblings: selectedSiblings.map { $0.op },
                    producerValue: group.producerValue,
                    useDefInfo: useDefInfo
                )

                let region = MultiOutputFusionRegion(
                    sharedProducer: producerOp,
                    producerIndex: producerIndex,
                    siblings: selectedSiblings.map { $0.op },
                    siblingIndices: selectedSiblings.map { $0.index },
                    inputs: externalInputs
                )

                regions.append(region)
            }
        }

        return regions
    }

    /// Checks if a producer operation can be the root of a sibling fusion.
    private func canFuseProducer(_ op: HLOOperation) -> Bool {
        // Producer should be a meaningful computation (not just reshape)
        switch op.kind {
        // Good fusion roots: operations that produce interesting intermediate results
        case .add, .subtract, .multiply, .divide, .maximum, .minimum, .power:
            return true

        case .exponential, .log, .sqrt, .rsqrt, .tanh, .logistic:
            return true

        case .dotGeneral, .dot:
            return true  // MatMul followed by multiple elementwise ops

        case .reduce:
            return true  // Reduction followed by normalization ops

        // Shape operations are not good roots (zero cost anyway)
        case .reshape, .transpose, .broadcastInDim:
            return false

        case .constant:
            return false  // Constants are always duplicated

        default:
            return false
        }
    }

    /// Checks if an operation is fusible as a sibling.
    private func isFusibleAsSibling(_ op: HLOOperation) -> Bool {
        switch op.kind {
        // Elementwise operations are excellent siblings
        case .add, .subtract, .multiply, .divide, .maximum, .minimum, .power:
            return true

        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt,
             .tanh, .logistic, .sine, .cosine, .tan,
             .floor, .ceil, .sign:
            return true

        // Type conversion
        case .convert, .bitcastConvert:
            return true

        // Comparison and selection
        case .compare, .select, .clamp:
            return true

        // Shape ops can be siblings (they're free)
        case .reshape, .broadcastInDim:
            return true

        // Reductions can be siblings if they reduce the same dimensions
        case .reduce:
            return true

        default:
            return false
        }
    }

    /// Checks if a sibling can be added to an existing group.
    private func canAddSibling(
        _ sibling: HLOOperation,
        to group: [(op: HLOOperation, index: Int)],
        sharedProducer: HLOOperation
    ) -> Bool {
        // Empty group: always accept
        if group.isEmpty { return true }

        // Check shape compatibility
        // Siblings should have compatible output shapes for efficient fusion
        let firstSibling = group[0].op
        if !areShapesCompatible(sibling.resultType, firstSibling.resultType) {
            return false
        }

        // Check register pressure
        let allOps = group.map { $0.op } + [sibling]
        let pressure = estimateRegisterPressure(allOps)
        if pressure > maxRegisterPressure {
            return false
        }

        // Use cost model if available
        if let costModel = costModel {
            // Check if fusion is beneficial
            let decision = costModel.shouldFuse(producer: sharedProducer, consumer: sibling)
            if !decision.shouldFuse {
                return false
            }
        }

        return true
    }

    /// Checks if two tensor types have compatible shapes for sibling fusion.
    private func areShapesCompatible(_ a: TensorType, _ b: TensorType) -> Bool {
        // Same shape is always compatible
        if a.shape == b.shape { return true }

        // Same rank with broadcastable dimensions is compatible
        if a.shape.count == b.shape.count {
            // Check if shapes are broadcastable to each other
            for (dimA, dimB) in zip(a.shape, b.shape) {
                if dimA != dimB && dimA != 1 && dimB != 1 {
                    return false
                }
            }
            return true
        }

        // Different ranks: check if one broadcasts to the other
        let smaller = a.shape.count < b.shape.count ? a.shape : b.shape
        let larger = a.shape.count < b.shape.count ? b.shape : a.shape

        // The smaller shape should be a suffix of the larger
        let offset = larger.count - smaller.count
        for i in 0..<smaller.count {
            if smaller[i] != larger[offset + i] && smaller[i] != 1 {
                return false
            }
        }

        return true
    }

    /// Estimates register pressure for a set of operations.
    private func estimateRegisterPressure(_ ops: [HLOOperation]) -> Int {
        var totalRegisters = 0

        for op in ops {
            // Estimate registers based on output type
            let elements = min(op.resultType.count, 16)  // Elements per thread
            let bytesPerElement = op.resultType.elementType.byteSize
            let registersPerElement = max(1, bytesPerElement / 4)
            totalRegisters += elements * registersPerElement
        }

        return totalRegisters
    }

    /// Computes external inputs for a fusion region.
    private func computeExternalInputs(
        producer: HLOOperation,
        siblings: [HLOOperation],
        producerValue: String,
        useDefInfo: UseDefInfo
    ) -> [String] {
        var inputs: [String] = []
        var seen: Set<String> = []

        // Add producer's operands first
        for operand in producer.operands {
            if !seen.contains(operand) {
                inputs.append(operand)
                seen.insert(operand)
            }
        }

        // Add siblings' operands (excluding the shared producer value)
        for sibling in siblings {
            for operand in sibling.operands {
                if operand != producerValue && !seen.contains(operand) {
                    // Check if this operand is produced by another sibling
                    let producedBySibling = siblings.contains { $0.result == operand }
                    if !producedBySibling {
                        inputs.append(operand)
                        seen.insert(operand)
                    }
                }
            }
        }

        return inputs
    }

    // MARK: - Code Emission

    /// Emits the optimized function with sibling fusions.
    private func emitFusedFunction(
        function: HLOFunction,
        regions: [MultiOutputFusionRegion]
    ) -> HLOFunction {
        // Build a set of indices that are part of fusion regions
        var fusedIndices: Set<Int> = []
        for region in regions {
            fusedIndices.insert(region.producerIndex)
            fusedIndices.formUnion(region.siblingIndices)
        }

        var newOperations: [HLOOperation] = []

        // Track which regions we've emitted
        var emittedRegions: Set<Int> = []

        for (index, op) in function.operations.enumerated() {
            // Check if this op is part of a fusion region
            if fusedIndices.contains(index) {
                // Find the region this belongs to
                for (regionIdx, region) in regions.enumerated() {
                    if emittedRegions.contains(regionIdx) { continue }

                    // Emit the region at the producer's position
                    if index == region.producerIndex {
                        if emitCustomCalls {
                            let fusedOps = emitAsCustomCalls(region: region)
                            newOperations.append(contentsOf: fusedOps)
                        } else {
                            // Emit producer and siblings in order
                            // (MPSGraph will handle stitching)
                            newOperations.append(region.sharedProducer)
                            newOperations.append(contentsOf: region.siblings)
                        }
                        emittedRegions.insert(regionIdx)
                        break
                    }
                }
            } else {
                // Not part of any fusion - emit as-is
                newOperations.append(op)
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

    /// Emits a sibling fusion region as custom call operations.
    private func emitAsCustomCalls(region: MultiOutputFusionRegion) -> [HLOOperation] {
        var operations: [HLOOperation] = []

        // Create a multi-output custom call
        var attributes = HLOAttributes()
        attributes.callTargetName = "sibling_fusion"

        // Encode fusion information in backend config
        let siblingKinds = region.siblings.map { $0.kind.rawValue }
        let configDict: [String: Any] = [
            "producer_kind": region.sharedProducer.kind.rawValue,
            "sibling_kinds": siblingKinds,
            "num_outputs": region.siblings.count
        ]

        if let data = try? JSONSerialization.data(withJSONObject: configDict, options: []),
           let str = String(data: data, encoding: .utf8) {
            attributes.backendConfig = str
        }

        // For multi-output, we need to emit one custom_call that produces a tuple,
        // then extract elements. However, HLO typically handles this differently.
        // For now, emit the producer and mark siblings as fused with the producer.

        // Emit producer with fusion annotation
        var producerAttrs = region.sharedProducer.attributes
        producerAttrs.backendConfig = "{\"sibling_fusion_root\": true, \"num_siblings\": \(region.siblings.count)}"

        let annotatedProducer = HLOOperation(
            result: region.sharedProducer.result,
            kind: region.sharedProducer.kind,
            operands: region.sharedProducer.operands,
            resultType: region.sharedProducer.resultType,
            attributes: producerAttrs
        )
        operations.append(annotatedProducer)

        // Emit siblings with fusion annotation
        for (idx, sibling) in region.siblings.enumerated() {
            var siblingAttrs = sibling.attributes
            siblingAttrs.backendConfig = "{\"sibling_fusion_member\": true, \"sibling_index\": \(idx)}"

            let annotatedSibling = HLOOperation(
                result: sibling.result,
                kind: sibling.kind,
                operands: sibling.operands,
                resultType: sibling.resultType,
                attributes: siblingAttrs
            )
            operations.append(annotatedSibling)
        }

        return operations
    }
}

// MARK: - Sibling Fusion Statistics

/// Statistics about sibling fusion results.
public struct SiblingFusionStatistics: Sendable {
    /// Number of sibling groups found.
    public let numSiblingGroups: Int

    /// Total siblings fused.
    public let totalSiblingsFused: Int

    /// Average siblings per group.
    public var averageSiblingsPerGroup: Double {
        guard numSiblingGroups > 0 else { return 0 }
        return Double(totalSiblingsFused) / Double(numSiblingGroups)
    }

    /// Estimated memory reads saved (one per additional sibling).
    public let estimatedReadsSaved: Int

    public init(
        numSiblingGroups: Int,
        totalSiblingsFused: Int,
        estimatedReadsSaved: Int
    ) {
        self.numSiblingGroups = numSiblingGroups
        self.totalSiblingsFused = totalSiblingsFused
        self.estimatedReadsSaved = estimatedReadsSaved
    }
}

extension SiblingFusion {
    /// Analyzes a function for sibling fusion opportunities.
    public func analyzeOpportunities(_ function: HLOFunction) -> SiblingFusionStatistics {
        let consumerMap = buildConsumerMap(function: function)
        let useDefInfo = UseDefInfo(function: function)
        let groups = findSiblingGroups(
            consumerMap: consumerMap,
            function: function,
            useDefInfo: useDefInfo
        )

        var totalSiblings = 0
        var estimatedSavings = 0

        for group in groups {
            let fusibleCount = group.siblings.count
            if fusibleCount >= 2 {
                totalSiblings += fusibleCount
                // Each additional sibling saves one read of the producer
                estimatedSavings += fusibleCount - 1
            }
        }

        let fusibleGroups = groups.filter { $0.siblings.count >= 2 }

        return SiblingFusionStatistics(
            numSiblingGroups: fusibleGroups.count,
            totalSiblingsFused: totalSiblings,
            estimatedReadsSaved: estimatedSavings
        )
    }
}
