// HLOOptimizer.swift
// MetalHLOCore
//
// Optimization pass manager for HLO modules.

import Foundation

/// Configuration for the HLO optimizer.
public struct HLOOptimizerConfig: Sendable {
    /// Whether to enable pattern-based fusion.
    public var enableFusion: Bool

    /// Whether to enable constant folding.
    public var enableConstantFolding: Bool

    /// Maximum number of optimization iterations.
    public var maxIterations: Int

    /// Patterns to enable (nil = all patterns).
    public var enabledPatterns: Set<String>?

    /// Creates default configuration.
    public init(
        enableFusion: Bool = true,
        enableConstantFolding: Bool = true,
        maxIterations: Int = 10,
        enabledPatterns: Set<String>? = nil
    ) {
        self.enableFusion = enableFusion
        self.enableConstantFolding = enableConstantFolding
        self.maxIterations = maxIterations
        self.enabledPatterns = enabledPatterns
    }

    /// Default configuration with all optimizations enabled.
    public static let `default` = HLOOptimizerConfig()

    /// Configuration with no optimizations (for debugging).
    public static let none = HLOOptimizerConfig(
        enableFusion: false,
        enableConstantFolding: false
    )
}

/// Optimizer for HLO modules.
///
/// `HLOOptimizer` runs pattern-based optimizations on HLO IR before
/// compilation to MPSGraph. It can detect and replace common patterns
/// like softmax, GELU, and layer normalization with optimized implementations.
public final class HLOOptimizer: @unchecked Sendable {

    private let config: HLOOptimizerConfig
    private let registry: HLOPatternRegistry

    /// Creates an optimizer with the given configuration.
    public init(config: HLOOptimizerConfig = .default, registry: HLOPatternRegistry = .shared) {
        self.config = config
        self.registry = registry
    }

    /// Optimizes a function.
    ///
    /// - Parameter function: The function to optimize.
    /// - Returns: The optimized function.
    public func optimize(_ function: HLOFunction) -> HLOFunction {
        guard config.enableFusion else {
            return function
        }

        var current = function
        var changed = true
        var iteration = 0

        while changed && iteration < config.maxIterations {
            changed = false
            iteration += 1

            if let optimized = runFusionPass(on: current) {
                current = optimized
                changed = true
            }
        }

        return current
    }

    /// Runs a single fusion pass.
    private func runFusionPass(on function: HLOFunction) -> HLOFunction? {
        let definingOpsMap = DefiningOpMap(function: function)
        let patterns = registry.registeredPatterns.filter { pattern in
            config.enabledPatterns == nil || config.enabledPatterns!.contains(pattern.name)
        }

        // Find matches (process from end to preserve indices)
        var matches: [(pattern: any HLOPattern, match: PatternMatch)] = []

        for (index, op) in function.operations.enumerated().reversed() {
            for pattern in patterns {
                if let match = pattern.match(at: op, index: index, in: function, definingOps: definingOpsMap.map) {
                    matches.append((pattern, match))
                    break  // Only one pattern per root operation
                }
            }
        }

        guard !matches.isEmpty else {
            return nil
        }

        // Apply the first non-overlapping match
        // (in a real implementation, we'd handle overlapping matches more carefully)
        let (pattern, match) = matches.first!

        return applyReplacement(pattern: pattern, match: match, to: function)
    }

    /// Applies a pattern replacement to the function.
    private func applyReplacement(
        pattern: any HLOPattern,
        match: PatternMatch,
        to function: HLOFunction
    ) -> HLOFunction {
        let replacements = pattern.replacement(for: match, in: function)
        let matchedIndices = Set(match.indices)

        // Build new operations list
        var newOperations: [HLOOperation] = []
        var insertedReplacement = false

        for (index, op) in function.operations.enumerated() {
            if matchedIndices.contains(index) {
                // Insert replacement at the position of the root operation
                if index == match.indices.last && !insertedReplacement {
                    newOperations.append(contentsOf: replacements)
                    insertedReplacement = true
                }
                // Skip matched operations
            } else {
                newOperations.append(op)
            }
        }

        // Update return values if the root operation is returned
        var newReturnValues = function.returnValues
        if let replacementOp = replacements.last {
            for (i, retVal) in newReturnValues.enumerated() {
                if retVal == match.rootOperation.result {
                    newReturnValues[i] = replacementOp.result
                }
            }
        }

        return HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: newOperations,
            returnValues: newReturnValues
        )
    }
}

