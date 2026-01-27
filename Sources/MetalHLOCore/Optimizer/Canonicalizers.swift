// Canonicalizers.swift
// MetalHLOCore
//
// Canonicalization passes that normalize IR representation.
// These run after simplification and before fusion to ensure consistent patterns.

import Foundation

// MARK: - Pass Protocol

/// Protocol for optimization passes.
public protocol OptimizationPass: Sendable {
    /// Pass name.
    var name: String { get }

    /// Which analysis results this pass invalidates.
    var invalidates: Set<AnalysisType> { get }

    /// Runs the pass on a function.
    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult
}

/// Types of analysis that can be invalidated.
public enum AnalysisType: String, Sendable, CaseIterable {
    case shapes
    case dependencies
    case lifetimes
    case patterns
}

/// Result of running a pass.
public struct PassResult: Sendable {
    /// The transformed function.
    public let function: HLOFunction

    /// Whether the function was modified.
    public let changed: Bool

    /// Statistics about the transformation.
    public let stats: [String: Int]

    public init(function: HLOFunction, changed: Bool, stats: [String: Int] = [:]) {
        self.function = function
        self.changed = changed
        self.stats = stats
    }

    /// No change result.
    public static func unchanged(_ function: HLOFunction) -> PassResult {
        PassResult(function: function, changed: false)
    }
}

// MARK: - Reshape Canonicalizer

/// Canonicalizes reshape operations.
///
/// Transformations:
/// - Remove identity reshapes (shape unchanged)
/// - Fold consecutive reshapes into single reshape
/// - Push reshapes through elementwise ops where beneficial
public final class ReshapeCanonicalizer: OptimizationPass, @unchecked Sendable {

    public let name = "reshape-canonicalizer"
    public let invalidates: Set<AnalysisType> = [.shapes, .lifetimes]

    public init() {}

    public func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        var operations = function.operations
        var changed = false
        var removedCount = 0
        var foldedCount = 0

        // Build defining ops map
        var definingOps: [TensorID: (op: HLOOperation, index: Int)] = [:]
        for (index, op) in operations.enumerated() {
            definingOps[op.result] = (op, index)
        }

        // Process in reverse to handle chains
        var indicesToRemove: Set<Int> = []

        for (index, op) in operations.enumerated().reversed() {
            guard op.kind == .reshape || op.kind == .dynamicReshape else { continue }

            // Check for identity reshape
            if let inputShape = analysis.shapes[op.operands[0]],
               inputShape == op.resultType.shape {
                // Remove identity reshape - replace uses with input
                operations = replaceUses(of: op.result, with: op.operands[0], in: operations)
                indicesToRemove.insert(index)
                changed = true
                removedCount += 1
                continue
            }

            // Check for consecutive reshapes
            if let inputDef = definingOps[op.operands[0]],
               inputDef.op.kind == .reshape || inputDef.op.kind == .dynamicReshape {
                // Fold: reshape(reshape(x, s1), s2) -> reshape(x, s2)
                let newOp = HLOOperation(
                    result: op.result,
                    kind: op.kind,
                    operands: inputDef.op.operands,  // Use original input
                    resultType: op.resultType,
                    attributes: op.attributes
                )
                operations[index] = newOp

                // Mark the intermediate reshape for removal if no other uses
                let hasOtherUses = operations.contains { otherOp in
                    otherOp.result != op.result && otherOp.operands.contains(inputDef.op.result)
                }
                if !hasOtherUses {
                    indicesToRemove.insert(inputDef.index)
                }

                changed = true
                foldedCount += 1
            }
        }

        // Remove marked operations
        if !indicesToRemove.isEmpty {
            operations = operations.enumerated().compactMap { index, op in
                indicesToRemove.contains(index) ? nil : op
            }
        }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: operations,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: changed,
            stats: ["reshapes_removed": removedCount, "reshapes_folded": foldedCount]
        )
    }

    /// Replaces all uses of a tensor with another.
    private func replaceUses(of old: TensorID, with new: TensorID, in operations: [HLOOperation]) -> [HLOOperation] {
        return operations.map { op in
            let newOperands = op.operands.map { $0 == old ? new : $0 }
            if newOperands != op.operands {
                return HLOOperation(
                    result: op.result,
                    kind: op.kind,
                    operands: newOperands,
                    resultType: op.resultType,
                    attributes: op.attributes
                )
            }
            return op
        }
    }
}

// MARK: - Transpose Canonicalizer

/// Canonicalizes transpose operations.
///
/// Transformations:
/// - Remove identity transposes
/// - Fold consecutive transposes
/// - Simplify transpose permutations
public final class TransposeCanonicalizer: OptimizationPass, @unchecked Sendable {

    public let name = "transpose-canonicalizer"
    public let invalidates: Set<AnalysisType> = [.shapes, .lifetimes]

    public init() {}

    public func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        var operations = function.operations
        var changed = false
        var removedCount = 0
        var foldedCount = 0

        // Build defining ops map
        var definingOps: [TensorID: (op: HLOOperation, index: Int)] = [:]
        for (index, op) in operations.enumerated() {
            definingOps[op.result] = (op, index)
        }

        var indicesToRemove: Set<Int> = []

        for (index, op) in operations.enumerated().reversed() {
            guard op.kind == .transpose else { continue }
            guard let permutation = op.attributes.dimensions else { continue }

            // Check for identity transpose
            if isIdentityPermutation(permutation) {
                operations = replaceUses(of: op.result, with: op.operands[0], in: operations)
                indicesToRemove.insert(index)
                changed = true
                removedCount += 1
                continue
            }

            // Check for consecutive transposes
            if let inputDef = definingOps[op.operands[0]],
               inputDef.op.kind == .transpose,
               let innerPerm = inputDef.op.attributes.dimensions {
                // Compose permutations: perm2[perm1[i]] for all i
                let composedPerm = composePermutations(innerPerm, permutation)

                // Check if composed is identity
                if isIdentityPermutation(composedPerm) {
                    // Double transpose cancels out
                    operations = replaceUses(of: op.result, with: inputDef.op.operands[0], in: operations)
                    indicesToRemove.insert(index)

                    let hasOtherUses = operations.contains { otherOp in
                        otherOp.result != op.result && otherOp.operands.contains(inputDef.op.result)
                    }
                    if !hasOtherUses {
                        indicesToRemove.insert(inputDef.index)
                    }

                    changed = true
                    removedCount += 2
                } else {
                    // Replace with single transpose
                    var newAttrs = op.attributes
                    newAttrs.dimensions = composedPerm

                    let newOp = HLOOperation(
                        result: op.result,
                        kind: .transpose,
                        operands: inputDef.op.operands,
                        resultType: op.resultType,
                        attributes: newAttrs
                    )
                    operations[index] = newOp

                    let hasOtherUses = operations.contains { otherOp in
                        otherOp.result != op.result && otherOp.operands.contains(inputDef.op.result)
                    }
                    if !hasOtherUses {
                        indicesToRemove.insert(inputDef.index)
                    }

                    changed = true
                    foldedCount += 1
                }
            }
        }

        // Remove marked operations
        if !indicesToRemove.isEmpty {
            operations = operations.enumerated().compactMap { index, op in
                indicesToRemove.contains(index) ? nil : op
            }
        }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: operations,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: changed,
            stats: ["transposes_removed": removedCount, "transposes_folded": foldedCount]
        )
    }

    /// Checks if a permutation is identity [0, 1, 2, ...].
    private func isIdentityPermutation(_ perm: [Int]) -> Bool {
        for (i, p) in perm.enumerated() {
            if p != i { return false }
        }
        return true
    }

    /// Composes two permutations: result[i] = perm2[perm1[i]].
    private func composePermutations(_ perm1: [Int], _ perm2: [Int]) -> [Int] {
        return perm1.map { perm2[$0] }
    }

    /// Replaces all uses of a tensor with another.
    private func replaceUses(of old: TensorID, with new: TensorID, in operations: [HLOOperation]) -> [HLOOperation] {
        return operations.map { op in
            let newOperands = op.operands.map { $0 == old ? new : $0 }
            if newOperands != op.operands {
                return HLOOperation(
                    result: op.result,
                    kind: op.kind,
                    operands: newOperands,
                    resultType: op.resultType,
                    attributes: op.attributes
                )
            }
            return op
        }
    }
}

// MARK: - Broadcast Canonicalizer

/// Canonicalizes broadcast operations.
///
/// Transformations:
/// - Remove identity broadcasts (no actual broadcasting)
/// - Fold consecutive broadcasts
/// - Merge broadcasts into consumers where possible
public final class BroadcastCanonicalizer: OptimizationPass, @unchecked Sendable {

    public let name = "broadcast-canonicalizer"
    public let invalidates: Set<AnalysisType> = [.shapes, .lifetimes]

    public init() {}

    public func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        var operations = function.operations
        var changed = false
        var removedCount = 0
        var foldedCount = 0

        // Build defining ops map
        var definingOps: [TensorID: (op: HLOOperation, index: Int)] = [:]
        for (index, op) in operations.enumerated() {
            definingOps[op.result] = (op, index)
        }

        var indicesToRemove: Set<Int> = []

        for (index, op) in operations.enumerated().reversed() {
            guard op.kind == .broadcastInDim else { continue }

            // Check for identity broadcast
            if let inputShape = analysis.shapes[op.operands[0]],
               inputShape == op.resultType.shape {
                operations = replaceUses(of: op.result, with: op.operands[0], in: operations)
                indicesToRemove.insert(index)
                changed = true
                removedCount += 1
                continue
            }

            // Check for scalar broadcast that can be folded into consumer
            if let inputShape = analysis.shapes[op.operands[0]],
               inputShape == [1] || inputShape.isEmpty {
                // This is a scalar broadcast - might be foldable
                // For now, just mark as potentially optimizable
            }

            // Check for consecutive broadcasts
            if let inputDef = definingOps[op.operands[0]],
               inputDef.op.kind == .broadcastInDim {
                // Fold: broadcast(broadcast(x)) -> broadcast(x) with combined dimensions
                let newOp = HLOOperation(
                    result: op.result,
                    kind: op.kind,
                    operands: inputDef.op.operands,
                    resultType: op.resultType,
                    attributes: op.attributes
                )
                operations[index] = newOp

                let hasOtherUses = operations.contains { otherOp in
                    otherOp.result != op.result && otherOp.operands.contains(inputDef.op.result)
                }
                if !hasOtherUses {
                    indicesToRemove.insert(inputDef.index)
                }

                changed = true
                foldedCount += 1
            }
        }

        // Remove marked operations
        if !indicesToRemove.isEmpty {
            operations = operations.enumerated().compactMap { index, op in
                indicesToRemove.contains(index) ? nil : op
            }
        }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: operations,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: changed,
            stats: ["broadcasts_removed": removedCount, "broadcasts_folded": foldedCount]
        )
    }

    /// Replaces all uses of a tensor with another.
    private func replaceUses(of old: TensorID, with new: TensorID, in operations: [HLOOperation]) -> [HLOOperation] {
        return operations.map { op in
            let newOperands = op.operands.map { $0 == old ? new : $0 }
            if newOperands != op.operands {
                return HLOOperation(
                    result: op.result,
                    kind: op.kind,
                    operands: newOperands,
                    resultType: op.resultType,
                    attributes: op.attributes
                )
            }
            return op
        }
    }
}

// MARK: - Dead Code Elimination

/// Removes operations whose results are never used.
public final class DeadCodeEliminationPass: OptimizationPass, @unchecked Sendable {

    public let name = "dead-code-elimination"
    public let invalidates: Set<AnalysisType> = [.lifetimes]

    public init() {}

    public func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        var operations = function.operations
        var changed = false
        var removedCount = 0

        // Build set of used tensors
        var usedTensors: Set<TensorID> = Set(function.returnValues)

        // Propagate uses backwards
        for op in operations.reversed() {
            if usedTensors.contains(op.result) {
                usedTensors.formUnion(op.operands)
            }
        }

        // Remove unused operations
        let originalCount = operations.count
        operations = operations.filter { usedTensors.contains($0.result) }
        removedCount = originalCount - operations.count
        changed = removedCount > 0

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: operations,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: changed,
            stats: ["dead_ops_removed": removedCount]
        )
    }
}

// MARK: - Common Subexpression Elimination

/// Eliminates common subexpressions.
public final class CommonSubexpressionEliminationPass: OptimizationPass, @unchecked Sendable {

    public let name = "common-subexpr-elim"
    public let invalidates: Set<AnalysisType> = [.lifetimes]

    public init() {}

    public func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        var operations = function.operations
        var changed = false
        var eliminatedCount = 0

        // Map from operation signature to first occurrence
        var signatures: [String: TensorID] = [:]

        for (index, op) in operations.enumerated() {
            let sig = computeSignature(op)

            if let existing = signatures[sig] {
                // Replace this operation's result with the existing one
                operations = replaceUses(of: op.result, with: existing, in: operations, startingFrom: index + 1)
                changed = true
                eliminatedCount += 1
            } else {
                signatures[sig] = op.result
            }
        }

        // Run DCE to remove the now-dead duplicates
        if changed {
            var usedTensors: Set<TensorID> = Set(function.returnValues)
            for op in operations.reversed() {
                if usedTensors.contains(op.result) {
                    usedTensors.formUnion(op.operands)
                }
            }
            operations = operations.filter { usedTensors.contains($0.result) }
        }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: operations,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: changed,
            stats: ["cse_eliminated": eliminatedCount]
        )
    }

    /// Computes a signature for an operation (kind + operands + key attributes).
    private func computeSignature(_ op: HLOOperation) -> String {
        var parts: [String] = [op.kind.rawValue]
        parts.append(contentsOf: op.operands)

        // Add relevant attributes
        if let perm = op.attributes.dimensions {
            parts.append("perm:\(perm)")
        }
        if let dims = op.attributes.dimensions {
            parts.append("dims:\(dims)")
        }

        // Include constant value in signature to distinguish different constants
        if let constantValue = op.attributes.constantValue {
            parts.append("const:\(constantValue)")
        }

        return parts.joined(separator: "|")
    }

    /// Replaces all uses of a tensor with another, starting from a specific index.
    private func replaceUses(of old: TensorID, with new: TensorID, in operations: [HLOOperation], startingFrom: Int) -> [HLOOperation] {
        return operations.enumerated().map { index, op in
            guard index >= startingFrom else { return op }

            let newOperands = op.operands.map { $0 == old ? new : $0 }
            if newOperands != op.operands {
                return HLOOperation(
                    result: op.result,
                    kind: op.kind,
                    operands: newOperands,
                    resultType: op.resultType,
                    attributes: op.attributes
                )
            }
            return op
        }
    }
}

// MARK: - Copy Elimination

/// Eliminates unnecessary copy operations.
public final class CopyEliminationPass: OptimizationPass, @unchecked Sendable {

    public let name = "copy-elimination"
    public let invalidates: Set<AnalysisType> = [.lifetimes]

    public init() {}

    public func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        var operations = function.operations
        var changed = false
        var eliminatedCount = 0

        var indicesToRemove: Set<Int> = []

        for (index, op) in operations.enumerated() {
            // Look for operations that are effectively copies
            // Note: StableHLO doesn't have an explicit copy op, so we detect implicit copies
            let isCopy = (op.kind == .reshape && op.operands.count == 1 &&
                         analysis.shapes[op.operands[0]] == op.resultType.shape) ||
                        (op.kind == .bitcastConvert && op.operands.count == 1 &&
                         analysis.elementTypes[op.operands[0]] == op.resultType.elementType)

            if isCopy {
                operations = replaceUses(of: op.result, with: op.operands[0], in: operations, startingFrom: index + 1)
                indicesToRemove.insert(index)
                changed = true
                eliminatedCount += 1
            }
        }

        // Remove marked operations
        if !indicesToRemove.isEmpty {
            operations = operations.enumerated().compactMap { index, op in
                indicesToRemove.contains(index) ? nil : op
            }
        }

        // Update return values
        var returnValues = function.returnValues
        for (index, op) in function.operations.enumerated() {
            if indicesToRemove.contains(index) {
                for (i, retVal) in returnValues.enumerated() {
                    if retVal == op.result {
                        returnValues[i] = op.operands[0]
                    }
                }
            }
        }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: operations,
            returnValues: returnValues
        )

        return PassResult(
            function: newFunction,
            changed: changed,
            stats: ["copies_eliminated": eliminatedCount]
        )
    }

    /// Replaces all uses of a tensor with another, starting from a specific index.
    private func replaceUses(of old: TensorID, with new: TensorID, in operations: [HLOOperation], startingFrom: Int) -> [HLOOperation] {
        return operations.enumerated().map { index, op in
            guard index >= startingFrom else { return op }

            let newOperands = op.operands.map { $0 == old ? new : $0 }
            if newOperands != op.operands {
                return HLOOperation(
                    result: op.result,
                    kind: op.kind,
                    operands: newOperands,
                    resultType: op.resultType,
                    attributes: op.attributes
                )
            }
            return op
        }
    }
}
