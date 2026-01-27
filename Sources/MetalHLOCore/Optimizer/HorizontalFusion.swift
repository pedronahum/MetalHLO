// HorizontalFusion.swift
// MetalHLOCore
//
// Horizontal fusion pass that batches independent small operations.
// Inspired by XLA's HorizontalFusion pass.

import Foundation

// MARK: - Operation Signature

/// Signature for grouping operations by type and shape.
public struct OperationSignature: Hashable, Sendable {
    /// The operation kind.
    public let kind: HLOOpKind

    /// The output element type.
    public let elementType: ElementType

    /// The output shape.
    public let shape: [Int]

    /// Number of operands.
    public let numOperands: Int

    public init(kind: HLOOpKind, elementType: ElementType, shape: [Int], numOperands: Int) {
        self.kind = kind
        self.elementType = elementType
        self.shape = shape
        self.numOperands = numOperands
    }

    /// Creates a signature from an operation.
    public static func from(_ op: HLOOperation) -> OperationSignature {
        return OperationSignature(
            kind: op.kind,
            elementType: op.resultType.elementType,
            shape: op.resultType.shape,
            numOperands: op.operands.count
        )
    }
}

// MARK: - Horizontal Fusion Group

/// A group of operations that can be horizontally fused.
public struct HorizontalFusionGroup: Sendable {
    /// The signature shared by all operations in this group.
    public let signature: OperationSignature

    /// The operations in this group.
    public let operations: [HLOOperation]

    /// Indices of the operations in the original function.
    public let indices: [Int]

    /// Total elements across all operations.
    public var totalElements: Int {
        signature.shape.reduce(1, *) * operations.count
    }

    /// Number of operations in this group.
    public var count: Int { operations.count }

    public init(signature: OperationSignature, operations: [HLOOperation], indices: [Int]) {
        self.signature = signature
        self.operations = operations
        self.indices = indices
    }
}

// MARK: - Horizontal Fusion Pass

/// Horizontal fusion pass that batches independent small operations.
///
/// This pass finds small, independent operations of the same type and shape,
/// then batches them into a single vectorized operation. This amortizes
/// kernel launch overhead across multiple operations.
///
/// Example (optimizer updates):
/// ```
/// Before:                         After:
/// %0 = add %p0, %g0              %batch_in0 = concat [%p0, %p1, %p2, %p3]
/// %1 = add %p1, %g1              %batch_in1 = concat [%g0, %g1, %g2, %g3]
/// %2 = add %p2, %g2     ═══>     %batch_out = add %batch_in0, %batch_in1
/// %3 = add %p3, %g3              %0, %1, %2, %3 = split %batch_out
/// ```
///
/// Common use cases:
/// - Optimizer parameter updates (Adam, SGD on many small tensors)
/// - Elementwise operations on many small buffers
/// - Gradient accumulation across parameters
public final class HorizontalFusion: @unchecked Sendable {

    /// Minimum number of operations to batch together.
    private let minBatchSize: Int

    /// Maximum total elements for a horizontal fusion.
    private let maxCombinedElements: Int

    /// Maximum elements per individual operation to be considered "small".
    private let maxElementsPerOp: Int

    /// Whether to emit fused operations as custom calls.
    private let emitCustomCalls: Bool

    /// Creates a horizontal fusion pass.
    ///
    /// - Parameters:
    ///   - minBatchSize: Minimum operations to batch (default: 4).
    ///   - maxCombinedElements: Maximum total elements (default: 1M).
    ///   - maxElementsPerOp: Maximum elements per op to be "small" (default: 10000).
    ///   - emitCustomCalls: Whether to emit as custom_call (default: false).
    public init(
        minBatchSize: Int = 4,
        maxCombinedElements: Int = 1024 * 1024,
        maxElementsPerOp: Int = 10000,
        emitCustomCalls: Bool = false
    ) {
        self.minBatchSize = minBatchSize
        self.maxCombinedElements = maxCombinedElements
        self.maxElementsPerOp = maxElementsPerOp
        self.emitCustomCalls = emitCustomCalls
    }

    /// Performs horizontal fusion on a function.
    ///
    /// - Parameter function: The function to optimize.
    /// - Returns: The optimized function with horizontally fused operations.
    public func fuse(_ function: HLOFunction) -> HLOFunction {
        // Build use-def information
        let useDefInfo = UseDefInfo(function: function)

        // Find candidate operations
        let candidates = findCandidates(function: function, useDefInfo: useDefInfo)

        // Group by signature
        let groups = groupBySignature(candidates: candidates)

        // Filter groups that meet minimum batch size
        let fusibleGroups = groups.filter { $0.count >= minBatchSize }

        // If no fusible groups, return unchanged
        guard !fusibleGroups.isEmpty else {
            return function
        }

        // Apply horizontal fusion
        return applyFusion(function: function, groups: fusibleGroups)
    }

    // MARK: - Candidate Finding

    /// Finds operations that are candidates for horizontal fusion.
    private func findCandidates(
        function: HLOFunction,
        useDefInfo: UseDefInfo
    ) -> [(op: HLOOperation, index: Int)] {
        var candidates: [(op: HLOOperation, index: Int)] = []

        for (index, op) in function.operations.enumerated() {
            if isCandidate(op, index: index, function: function, useDefInfo: useDefInfo) {
                candidates.append((op, index))
            }
        }

        return candidates
    }

    /// Checks if an operation is a candidate for horizontal fusion.
    private func isCandidate(
        _ op: HLOOperation,
        index: Int,
        function: HLOFunction,
        useDefInfo: UseDefInfo
    ) -> Bool {
        // Must be a small elementwise operation
        guard isSmallElementwiseOp(op) else { return false }

        // Must be independent (no dependencies on other candidates)
        // For now, we check that operands are either function inputs or
        // from operations that are not candidates themselves
        guard isIndependent(op, index: index, function: function, useDefInfo: useDefInfo) else {
            return false
        }

        return true
    }

    /// Checks if an operation is a small elementwise operation.
    private func isSmallElementwiseOp(_ op: HLOOperation) -> Bool {
        // Check element count
        let elements = op.resultType.count
        guard elements < maxElementsPerOp else { return false }

        // Check if elementwise
        switch op.kind {
        // Binary arithmetic
        case .add, .subtract, .multiply, .divide, .maximum, .minimum, .power:
            return true

        // Unary operations
        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt,
             .tanh, .logistic, .sine, .cosine, .tan,
             .floor, .ceil, .sign:
            return true

        // Type conversion
        case .convert:
            return true

        // Comparison
        case .compare:
            return true

        default:
            return false
        }
    }

    /// Checks if an operation is independent (can be reordered).
    private func isIndependent(
        _ op: HLOOperation,
        index: Int,
        function: HLOFunction,
        useDefInfo: UseDefInfo
    ) -> Bool {
        // Check that none of the operands are produced by operations
        // that could also be candidates (to avoid complex dependency analysis)
        for operand in op.operands {
            if let (defOp, defIndex) = useDefInfo.definingOp(for: operand) {
                // If the defining operation is also a small elementwise op,
                // there might be a dependency chain - be conservative
                if isSmallElementwiseOp(defOp) && defIndex < index {
                    // Allow if this is the only use (producer-consumer chain)
                    // but don't batch the consumer in this case
                    return false
                }
            }
        }

        return true
    }

    // MARK: - Grouping

    /// Groups candidates by their signature.
    private func groupBySignature(
        candidates: [(op: HLOOperation, index: Int)]
    ) -> [HorizontalFusionGroup] {
        var groups: [OperationSignature: [(op: HLOOperation, index: Int)]] = [:]

        for (op, index) in candidates {
            let signature = OperationSignature.from(op)
            groups[signature, default: []].append((op, index))
        }

        // Convert to HorizontalFusionGroup and filter by size constraints
        return groups.compactMap { signature, ops -> HorizontalFusionGroup? in
            // Check combined size limit
            let elementsPerOp = signature.shape.reduce(1, *)
            let maxOps = maxCombinedElements / max(1, elementsPerOp)
            let limitedOps = Array(ops.prefix(maxOps))

            guard limitedOps.count >= minBatchSize else { return nil }

            return HorizontalFusionGroup(
                signature: signature,
                operations: limitedOps.map { $0.op },
                indices: limitedOps.map { $0.index }
            )
        }
    }

    // MARK: - Fusion Application

    /// Applies horizontal fusion to the function.
    private func applyFusion(
        function: HLOFunction,
        groups: [HorizontalFusionGroup]
    ) -> HLOFunction {
        // Build set of indices that are being fused
        var fusedIndices: Set<Int> = []
        for group in groups {
            fusedIndices.formUnion(group.indices)
        }

        // Build new operations list
        var newOperations: [HLOOperation] = []
        var emittedGroups: Set<Int> = []

        // Map from original result names to new result names (after slicing)
        var resultMapping: [String: String] = [:]

        for (index, op) in function.operations.enumerated() {
            if fusedIndices.contains(index) {
                // Find which group this belongs to
                for (groupIdx, group) in groups.enumerated() {
                    if emittedGroups.contains(groupIdx) { continue }

                    // Emit at the position of the first operation in the group
                    if index == group.indices.min() {
                        let (batchedOps, mapping) = emitBatchedOperations(group: group)
                        newOperations.append(contentsOf: batchedOps)
                        resultMapping.merge(mapping) { _, new in new }
                        emittedGroups.insert(groupIdx)
                        break
                    }
                }
            } else {
                // Update operands if they were remapped
                let updatedOp = updateOperands(op, mapping: resultMapping)
                newOperations.append(updatedOp)
            }
        }

        // Update return values if needed
        let newReturnValues = function.returnValues.map { retVal in
            resultMapping[retVal] ?? retVal
        }

        return HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: newOperations,
            returnValues: newReturnValues
        )
    }

    /// Updates operands using the result mapping.
    private func updateOperands(_ op: HLOOperation, mapping: [String: String]) -> HLOOperation {
        let newOperands = op.operands.map { operand in
            mapping[operand] ?? operand
        }

        if newOperands == op.operands {
            return op
        }

        return HLOOperation(
            result: op.result,
            kind: op.kind,
            operands: newOperands,
            resultType: op.resultType,
            attributes: op.attributes
        )
    }

    /// Emits batched operations for a group.
    private func emitBatchedOperations(
        group: HorizontalFusionGroup
    ) -> (operations: [HLOOperation], mapping: [String: String]) {
        var operations: [HLOOperation] = []
        let mapping: [String: String] = [:]

        let signature = group.signature
        let batchSize = group.count

        // Create batched shape: [batchSize, ...originalShape]
        let batchedShape = [batchSize] + signature.shape
        let batchedType = TensorType(shape: batchedShape, elementType: signature.elementType)

        // For each operand position, concatenate all operands
        var batchedOperands: [String] = []

        for operandIdx in 0..<signature.numOperands {
            // Collect operands from all operations
            let operands = group.operations.map { $0.operands[operandIdx] }

            // Create concatenate operation
            let concatResult = "%_hfusion_concat_\(operandIdx)_\(group.indices[0])"
            var concatAttrs = HLOAttributes()
            concatAttrs.axis = 0  // Concatenate along new batch dimension

            // First, reshape each operand to add batch dimension
            var reshapedOperands: [String] = []
            for (idx, operand) in operands.enumerated() {
                let reshapeResult = "%_hfusion_reshape_\(operandIdx)_\(idx)_\(group.indices[0])"
                let reshapedShape = [1] + signature.shape
                let reshapedType = TensorType(shape: reshapedShape, elementType: signature.elementType)

                let reshapeOp = HLOOperation(
                    result: reshapeResult,
                    kind: .reshape,
                    operands: [operand],
                    resultType: reshapedType
                )
                operations.append(reshapeOp)
                reshapedOperands.append(reshapeResult)
            }

            // Then concatenate
            let concatOp = HLOOperation(
                result: concatResult,
                kind: .concatenate,
                operands: reshapedOperands,
                resultType: batchedType,
                attributes: concatAttrs
            )
            operations.append(concatOp)
            batchedOperands.append(concatResult)
        }

        // Create the batched elementwise operation
        let batchedResult = "%_hfusion_batched_\(group.indices[0])"
        let batchedOp = HLOOperation(
            result: batchedResult,
            kind: signature.kind,
            operands: batchedOperands,
            resultType: batchedType,
            attributes: group.operations[0].attributes  // Copy attributes from first op
        )
        operations.append(batchedOp)

        // Slice the result back into individual outputs
        for (idx, originalOp) in group.operations.enumerated() {
            let sliceResult = originalOp.result  // Keep original result name

            // Create slice operation to extract this batch element
            var sliceAttrs = HLOAttributes()
            sliceAttrs.sliceStarts = [idx] + Array(repeating: 0, count: signature.shape.count)
            sliceAttrs.sliceLimits = [idx + 1] + signature.shape
            sliceAttrs.sliceStrides = Array(repeating: 1, count: batchedShape.count)

            let slicedShape = [1] + signature.shape
            let slicedType = TensorType(shape: slicedShape, elementType: signature.elementType)

            let sliceOp = HLOOperation(
                result: "%_hfusion_slice_\(idx)_\(group.indices[0])",
                kind: .slice,
                operands: [batchedResult],
                resultType: slicedType,
                attributes: sliceAttrs
            )
            operations.append(sliceOp)

            // Reshape back to original shape (remove batch dimension)
            let originalType = TensorType(shape: signature.shape, elementType: signature.elementType)
            let reshapeOp = HLOOperation(
                result: sliceResult,
                kind: .reshape,
                operands: [sliceOp.result],
                resultType: originalType
            )
            operations.append(reshapeOp)

            // No mapping needed since we keep the original result name
        }

        return (operations, mapping)
    }
}

// MARK: - Horizontal Fusion Statistics

/// Statistics about horizontal fusion results.
public struct HorizontalFusionStatistics: Sendable {
    /// Number of fusion groups created.
    public let numGroups: Int

    /// Total operations batched.
    public let totalOperationsBatched: Int

    /// Estimated kernel launches saved.
    public let kernelLaunchesSaved: Int

    /// Average batch size.
    public var averageBatchSize: Double {
        guard numGroups > 0 else { return 0 }
        return Double(totalOperationsBatched) / Double(numGroups)
    }

    public init(
        numGroups: Int,
        totalOperationsBatched: Int,
        kernelLaunchesSaved: Int
    ) {
        self.numGroups = numGroups
        self.totalOperationsBatched = totalOperationsBatched
        self.kernelLaunchesSaved = kernelLaunchesSaved
    }
}

extension HorizontalFusion {
    /// Analyzes a function for horizontal fusion opportunities.
    public func analyzeOpportunities(_ function: HLOFunction) -> HorizontalFusionStatistics {
        let useDefInfo = UseDefInfo(function: function)
        let candidates = findCandidates(function: function, useDefInfo: useDefInfo)
        let groups = groupBySignature(candidates: candidates)
        let fusibleGroups = groups.filter { $0.count >= minBatchSize }

        var totalBatched = 0
        var launchesSaved = 0

        for group in fusibleGroups {
            totalBatched += group.count
            // Each batched group saves (count - 1) kernel launches
            launchesSaved += group.count - 1
        }

        return HorizontalFusionStatistics(
            numGroups: fusibleGroups.count,
            totalOperationsBatched: totalBatched,
            kernelLaunchesSaved: launchesSaved
        )
    }
}

// MARK: - Batched Operation Emission (Custom Call)

extension HorizontalFusion {
    /// Emits a horizontally fused group as a custom call.
    func emitAsCustomCall(group: HorizontalFusionGroup) -> HLOOperation {
        var attributes = HLOAttributes()
        attributes.callTargetName = "horizontal_fusion"

        // Encode fusion information
        let configDict: [String: Any] = [
            "op_kind": group.signature.kind.rawValue,
            "batch_size": group.count,
            "shape": group.signature.shape,
            "element_type": group.signature.elementType.rawValue
        ]

        if let data = try? JSONSerialization.data(withJSONObject: configDict, options: []),
           let str = String(data: data, encoding: .utf8) {
            attributes.backendConfig = str
        }

        // Collect all operands from all operations
        let allOperands = group.operations.flatMap { $0.operands }

        // Result type would be a tuple of all outputs
        // For now, use the first operation's result type (simplified)
        return HLOOperation(
            result: "%_hfusion_\(group.indices[0])",
            kind: .customCall,
            operands: allOperands,
            resultType: group.operations[0].resultType,
            attributes: attributes
        )
    }
}
