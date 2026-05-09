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
                // Compose dimensions: for input dim i, the combined dimension is outerDims[innerDims[i]]
                let innerDims = inputDef.op.attributes.dimensions ?? []
                let outerDims = op.attributes.dimensions ?? []

                // Compose the dimensions correctly
                var combinedDims: [Int] = []
                for innerDim in innerDims {
                    if innerDim >= 0 && innerDim < outerDims.count {
                        combinedDims.append(outerDims[innerDim])
                    } else {
                        // Invalid dimension mapping - skip folding this pair
                        continue
                    }
                }

                // Only fold if we successfully composed all dimensions
                guard combinedDims.count == innerDims.count else {
                    continue
                }

                // Create new attributes with combined dimensions
                var newAttributes = op.attributes
                newAttributes.dimensions = combinedDims

                let newOp = HLOOperation(
                    result: op.result,
                    kind: op.kind,
                    operands: inputDef.op.operands,
                    resultType: op.resultType,
                    attributes: newAttributes
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
    /// All attributes that affect computation must be included — otherwise CSE
    /// will incorrectly merge operations that produce different results.
    private func computeSignature(_ op: HLOOperation) -> String {
        var parts: [String] = [op.kind.rawValue]
        parts.append(contentsOf: op.operands)
        // Include result type — operations with the same kind/operands but
        // different result types (e.g., broadcast to uint32 vs float32) must
        // not be merged.
        parts.append("type:\(op.resultType)")

        // Add relevant attributes
        if let dims = op.attributes.dimensions {
            parts.append("dims:\(dims)")
        }

        // Include constant value in signature to distinguish different constants
        if let constantValue = op.attributes.constantValue {
            parts.append("const:\(constantValue)")
        }

        // Iota dimension — different dims produce different results
        if let iotaDim = op.attributes.iotaDimension {
            parts.append("iotaDim:\(iotaDim)")
        }

        // Slice attributes — different start/limit/stride = different result
        if let starts = op.attributes.sliceStarts {
            parts.append("sliceStarts:\(starts)")
        }
        if let limits = op.attributes.sliceLimits {
            parts.append("sliceLimits:\(limits)")
        }
        if let strides = op.attributes.sliceStrides {
            parts.append("sliceStrides:\(strides)")
        }

        // Pad attributes
        if let padLow = op.attributes.padLow {
            parts.append("padLow:\(padLow)")
        }
        if let padHigh = op.attributes.padHigh {
            parts.append("padHigh:\(padHigh)")
        }
        if let padInterior = op.attributes.padInterior {
            parts.append("padInterior:\(padInterior)")
        }

        // Gather/scatter dimension numbers
        if let gatherDims = op.attributes.gatherDimensionNumbers {
            parts.append("gatherDims:\(gatherDims)")
        }
        if let scatterDims = op.attributes.scatterDimensionNumbers {
            parts.append("scatterDims:\(scatterDims)")
        }

        // Dot dimension numbers
        if let dotDims = op.attributes.dotDimensionNumbers {
            parts.append("dotDims:\(dotDims)")
        }

        // Convolution attributes
        if let windowStrides = op.attributes.windowStrides {
            parts.append("windowStrides:\(windowStrides)")
        }
        if let padding = op.attributes.convPadding {
            parts.append("padding:\(padding)")
        }

        // Comparison direction
        if let dir = op.attributes.comparisonDirection {
            parts.append("cmpDir:\(dir)")
        }

        // Reduce/scatter computation
        if let kind = op.attributes.scatterComputationKind {
            parts.append("compKind:\(kind)")
        }

        // Result type shape — operations with same kind/operands but different
        // output shapes (e.g. broadcast_in_dim) are not equivalent
        parts.append("shape:\(op.resultType.shape)")

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

// MARK: - Transpose-Matmul Folding

/// Folds transpose operations into their consuming matmul operations.
///
/// This pass detects patterns like:
/// - transpose(A) @ B -> matmul(A, B, transA=true)
/// - A @ transpose(B) -> matmul(A, B, transB=true)
/// - transpose(A) @ transpose(B) -> matmul(A, B, transA=true, transB=true)
///
/// Instead of performing a physical transpose (expensive memory operation),
/// the matmul operation can handle the transpose internally by adjusting
/// how it reads the matrix. This is a major optimization for attention
/// patterns where Q @ K^T is common.
public final class TransposeMatmulFoldingPass: OptimizationPass, @unchecked Sendable {

    public let name = "transpose-matmul-folding"
    public let invalidates: Set<AnalysisType> = [.shapes, .lifetimes, .dependencies]

    public init() {}

    public func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        var operations = function.operations
        var changed = false
        var foldedCount = 0

        // Build defining ops map
        var definingOps: [TensorID: (op: HLOOperation, index: Int)] = [:]
        for (index, op) in operations.enumerated() {
            definingOps[op.result] = (op, index)
        }

        // Build use count map to check if transpose has single use
        var useCount: [TensorID: Int] = [:]
        for op in operations {
            for operand in op.operands {
                useCount[operand, default: 0] += 1
            }
        }
        // Also count return values as uses
        for retVal in function.returnValues {
            useCount[retVal, default: 0] += 1
        }

        var indicesToRemove: Set<Int> = []

        for (index, op) in operations.enumerated() {
            // Look for dot or dotGeneral operations
            guard op.kind == .dot || op.kind == .dotGeneral else { continue }

            // Check if LHS or RHS comes from a transpose
            var newOperands = op.operands
            var newAttributes = op.attributes
            var didFold = false

            // Check LHS (operand 0)
            if let lhsDef = definingOps[op.operands[0]],
               lhsDef.op.kind == .transpose,
               useCount[lhsDef.op.result] == 1 {
                // Check if this is a simple 2D transpose (swap last two dims)
                if let perm = lhsDef.op.attributes.dimensions,
                   isMatrixTranspose(perm) {
                    // Fold: use original input and set transA flag
                    newOperands[0] = lhsDef.op.operands[0]
                    newAttributes.lhsTranspose = true
                    indicesToRemove.insert(lhsDef.index)
                    didFold = true
                }
            }

            // Check RHS (operand 1)
            if let rhsDef = definingOps[op.operands[1]],
               rhsDef.op.kind == .transpose,
               useCount[rhsDef.op.result] == 1 {
                // Check if this is a simple 2D transpose (swap last two dims)
                if let perm = rhsDef.op.attributes.dimensions,
                   isMatrixTranspose(perm) {
                    // Fold: use original input and set transB flag
                    newOperands[1] = rhsDef.op.operands[0]
                    newAttributes.rhsTranspose = true
                    indicesToRemove.insert(rhsDef.index)
                    didFold = true
                }
            }

            if didFold {
                // Create new operation with folded transposes
                let newOp = HLOOperation(
                    result: op.result,
                    kind: op.kind,
                    operands: newOperands,
                    resultType: op.resultType,
                    attributes: newAttributes
                )
                operations[index] = newOp
                changed = true
                foldedCount += 1
            }
        }

        // Remove folded transpose operations
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
            stats: ["transposes_folded_into_matmul": foldedCount]
        )
    }

    /// Checks if a permutation represents a matrix transpose (swapping last two dimensions).
    /// For 2D: [1, 0]
    /// For 3D: [0, 2, 1] (batch preserved, matrix transposed)
    /// For 4D: [0, 1, 3, 2] (batch dims preserved, matrix transposed)
    private func isMatrixTranspose(_ perm: [Int]) -> Bool {
        guard perm.count >= 2 else { return false }

        let rank = perm.count

        // Check that all dimensions except the last two are in order
        for i in 0..<(rank - 2) {
            if perm[i] != i { return false }
        }

        // Check that last two dimensions are swapped
        return perm[rank - 2] == rank - 1 && perm[rank - 1] == rank - 2
    }
}

// MARK: - Dot General Layout Canonicalize

/// Inserts transposes before `dot_general` operands so the matmul kernel
/// (which assumes `[batch_flat, M, K]` / `[batch_flat, K, N]` row-major
/// memory) sees correctly laid-out inputs.
///
/// The kernel computes `M`, `K`, `N`, and `batchSize` by reducing over the
/// batching/contracting/remaining dim sets, but reads the operand memory as
/// if it were already in `[batch..., remaining..., contracting...]` (LHS) or
/// `[batch..., contracting..., remaining...]` (RHS) order. When batch dims
/// are interleaved with remaining/contracting dims (e.g. multi-head
/// attention's `batching_dims = [0, 2]` for shape `(B, S, H, D)`), the
/// flatten produces wrong values.
///
/// Fix: for any `dot_general` whose operand layout is not already canonical,
/// insert a `transpose` and rewrite the dim numbers to reference the
/// transposed shape.
///
/// Runs late (cleanup phase) so attention / fusion patterns can match on
/// the un-canonicalized form first.
public final class DotGeneralLayoutCanonicalize: OptimizationPass, @unchecked Sendable {

    public let name = "dot-general-layout-canonicalize"
    public let invalidates: Set<AnalysisType> = [.shapes, .lifetimes, .patterns, .dependencies]

    public init() {}

    public func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        var newOps: [HLOOperation] = []
        newOps.reserveCapacity(function.operations.count)
        var changed = false
        var injected = 0

        // Track existing SSA names to avoid collisions with synthesized ones.
        var usedNames = Set(function.operations.map { $0.result })
        for input in function.inputs { usedNames.insert(input.name) }

        func freshName(prefix: String) -> String {
            var i = 0
            while true {
                let n = "\(prefix)\(i)"
                if !usedNames.contains(n) {
                    usedNames.insert(n)
                    return n
                }
                i += 1
            }
        }

        for op in function.operations {
            guard op.kind == .dotGeneral,
                  let dims = op.attributes.dotDimensionNumbers else {
                newOps.append(op)
                continue
            }

            let lhsShape = analysis.shapes[op.operands[0]] ?? []
            let rhsShape = analysis.shapes[op.operands[1]] ?? []
            guard lhsShape.count >= 2, rhsShape.count >= 2 else {
                newOps.append(op)
                continue
            }

            let lhsBatch = dims.lhsBatchingDimensions
            let rhsBatch = dims.rhsBatchingDimensions
            let lhsContract = dims.lhsContractingDimensions
            let rhsContract = dims.rhsContractingDimensions

            // Compute remaining (non-batch, non-contracting) dims, in original order.
            let lhsRemaining = (0..<lhsShape.count).filter {
                !lhsBatch.contains($0) && !lhsContract.contains($0)
            }
            let rhsRemaining = (0..<rhsShape.count).filter {
                !rhsBatch.contains($0) && !rhsContract.contains($0)
            }

            // Canonical operand layouts:
            //   LHS: [batch..., remaining..., contracting...]
            //   RHS: [batch..., contracting..., remaining...]
            let lhsCanonical = lhsBatch + lhsRemaining + lhsContract
            let rhsCanonical = rhsBatch + rhsContract + rhsRemaining

            let lhsNeedsTranspose = lhsCanonical != Array(0..<lhsShape.count)
            let rhsNeedsTranspose = rhsCanonical != Array(0..<rhsShape.count)

            // Even when the layout is already canonical, transpose-folding may
            // have stamped lhsTranspose/rhsTranspose on the dot_general from a
            // matrix-transpose it absorbed earlier. Drop those flags here: from
            // this point on the dim numbers (post-canonicalization) are the
            // single source of truth, and the codegen path's M/K/N derivation
            // will re-derive any needed swap from them.
            if !lhsNeedsTranspose && !rhsNeedsTranspose {
                if op.attributes.lhsTranspose != nil || op.attributes.rhsTranspose != nil {
                    var newAttrs = op.attributes
                    newAttrs.lhsTranspose = nil
                    newAttrs.rhsTranspose = nil
                    let cleared = HLOOperation(
                        result: op.result,
                        kind: .dotGeneral,
                        operands: op.operands,
                        resultType: op.resultType,
                        attributes: newAttrs
                    )
                    newOps.append(cleared)
                    changed = true
                } else {
                    newOps.append(op)
                }
                continue
            }

            var newOperands = op.operands

            if lhsNeedsTranspose {
                let newName = freshName(prefix: "%dotcan_lhs_")
                let newShape = lhsCanonical.map { lhsShape[$0] }
                let lhsType = TensorType(
                    shape: newShape,
                    elementType: op.resultType.elementType
                )
                var attrs = HLOAttributes()
                attrs.dimensions = lhsCanonical
                let tOp = HLOOperation(
                    result: newName,
                    kind: .transpose,
                    operands: [op.operands[0]],
                    resultType: lhsType,
                    attributes: attrs
                )
                newOps.append(tOp)
                newOperands[0] = newName
                injected += 1
            }

            if rhsNeedsTranspose {
                let newName = freshName(prefix: "%dotcan_rhs_")
                let newShape = rhsCanonical.map { rhsShape[$0] }
                let rhsType = TensorType(
                    shape: newShape,
                    elementType: op.resultType.elementType
                )
                var attrs = HLOAttributes()
                attrs.dimensions = rhsCanonical
                let tOp = HLOOperation(
                    result: newName,
                    kind: .transpose,
                    operands: [op.operands[1]],
                    resultType: rhsType,
                    attributes: attrs
                )
                newOps.append(tOp)
                newOperands[1] = newName
                injected += 1
            }

            // After canonicalization, the dim numbers are positional:
            //   LHS batching = [0, ..., bC-1]
            //   LHS contracting = [bC + rC, ..., bC + rC + cC - 1]
            //   RHS batching = [0, ..., bC-1]
            //   RHS contracting = [bC, ..., bC + cC - 1]
            let bC = lhsBatch.count
            let lhsRemCount = lhsRemaining.count
            let cC = lhsContract.count
            let rhsRemCount = rhsRemaining.count

            let newBatchLHS = Array(0..<bC)
            let newContractLHS = Array((bC + lhsRemCount)..<(bC + lhsRemCount + cC))
            let newBatchRHS = Array(0..<bC)
            let newContractRHS = Array(bC..<(bC + cC))

            var newAttrs = op.attributes
            newAttrs.dotDimensionNumbers = DotDimensionNumbers(
                lhsBatchingDimensions: newBatchLHS,
                rhsBatchingDimensions: newBatchRHS,
                lhsContractingDimensions: newContractLHS,
                rhsContractingDimensions: newContractRHS
            )
            // The transpose-folding pass may have set lhsTranspose/rhsTranspose
            // based on the original layout — clear those so they don't get
            // applied a second time on top of our explicit transposes.
            newAttrs.lhsTranspose = nil
            newAttrs.rhsTranspose = nil

            let newDot = HLOOperation(
                result: op.result,
                kind: .dotGeneral,
                operands: newOperands,
                resultType: op.resultType,
                attributes: newAttrs
            )
            newOps.append(newDot)
            changed = true
            _ = rhsRemCount  // silence unused warning when no RHS transpose
        }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: newOps,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: changed,
            stats: ["dot_transposes_inserted": injected]
        )
    }
}
