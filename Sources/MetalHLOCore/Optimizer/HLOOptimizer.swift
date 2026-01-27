// HLOOptimizer.swift
// MetalHLOCore
//
// Optimization pass manager for HLO modules.

import Foundation

/// Configuration for the HLO optimizer.
public struct HLOOptimizerConfig: Sendable {
    /// Whether to enable pattern-based fusion.
    public var enableFusion: Bool

    /// Whether to enable algebraic simplification (includes constant folding).
    public var enableAlgebraicSimplification: Bool

    /// Whether to enable constant folding (deprecated, use enableAlgebraicSimplification).
    public var enableConstantFolding: Bool {
        get { enableAlgebraicSimplification }
        set { enableAlgebraicSimplification = newValue }
    }

    /// Whether to enable producer-consumer fusion.
    public var enableProducerConsumerFusion: Bool

    /// Whether to enable sibling fusion (multi-output fusion).
    public var enableSiblingFusion: Bool

    /// Whether to enable horizontal fusion (batching small ops).
    public var enableHorizontalFusion: Bool

    /// Whether to enable layout assignment optimization.
    public var enableLayoutAssignment: Bool

    /// Maximum number of optimization iterations.
    public var maxIterations: Int

    /// Patterns to enable (nil = all patterns).
    public var enabledPatterns: Set<String>?

    /// Maximum operations in a producer-consumer fusion region.
    public var maxFusionRegionSize: Int

    /// Creates default configuration.
    public init(
        enableFusion: Bool = true,
        enableConstantFolding: Bool = true,
        enableProducerConsumerFusion: Bool = true,
        enableSiblingFusion: Bool = true,
        enableHorizontalFusion: Bool = true,
        enableLayoutAssignment: Bool = true,
        maxIterations: Int = 10,
        enabledPatterns: Set<String>? = nil,
        maxFusionRegionSize: Int = 50
    ) {
        self.enableFusion = enableFusion
        self.enableAlgebraicSimplification = enableConstantFolding
        self.enableProducerConsumerFusion = enableProducerConsumerFusion
        self.enableSiblingFusion = enableSiblingFusion
        self.enableHorizontalFusion = enableHorizontalFusion
        self.enableLayoutAssignment = enableLayoutAssignment
        self.maxIterations = maxIterations
        self.enabledPatterns = enabledPatterns
        self.maxFusionRegionSize = maxFusionRegionSize
    }

    /// Default configuration with all optimizations enabled.
    public static let `default` = HLOOptimizerConfig()

    /// Configuration with no optimizations (for debugging).
    public static let none = HLOOptimizerConfig(
        enableFusion: false,
        enableConstantFolding: false,
        enableProducerConsumerFusion: false,
        enableSiblingFusion: false,
        enableHorizontalFusion: false,
        enableLayoutAssignment: false
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
    private let algebraicSimplifier: AlgebraicSimplifier
    private let producerConsumerFusion: ProducerConsumerFusion
    private let siblingFusion: SiblingFusion
    private let horizontalFusion: HorizontalFusion
    private let layoutAssignment: LayoutAssignment

    /// Creates an optimizer with the given configuration.
    public init(config: HLOOptimizerConfig = .default, registry: HLOPatternRegistry = .shared) {
        self.config = config
        self.registry = registry
        self.algebraicSimplifier = AlgebraicSimplifier(maxIterations: config.maxIterations)
        self.producerConsumerFusion = ProducerConsumerFusion(
            maxFusionSize: config.maxFusionRegionSize,
            emitCustomCalls: false  // MPSGraph handles stitching
        )
        self.siblingFusion = SiblingFusion(
            maxSiblings: 8,
            emitCustomCalls: false  // MPSGraph handles multi-output
        )
        self.horizontalFusion = HorizontalFusion(
            minBatchSize: 4,
            emitCustomCalls: false  // MPSGraph handles batching
        )
        self.layoutAssignment = LayoutAssignment(insertTransposes: true)
    }

    /// Optimizes a function.
    ///
    /// - Parameter function: The function to optimize.
    /// - Returns: The optimized function.
    public func optimize(_ function: HLOFunction) -> HLOFunction {
        var current = function

        // Phase 1: Algebraic simplification (runs to convergence)
        // This eliminates redundant operations, folds constants, and simplifies shapes
        if config.enableAlgebraicSimplification {
            current = algebraicSimplifier.simplify(current)
        }

        // Phase 2: Pattern-based fusion
        // This detects and replaces common patterns (attention, softmax, etc.)
        if config.enableFusion {
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
        }

        // Phase 3: Producer-consumer fusion
        // This groups elementwise operations into fusion regions to eliminate
        // intermediate memory writes. MPSGraph will then stitch these together.
        if config.enableProducerConsumerFusion {
            current = producerConsumerFusion.fuse(current)
        }

        // Phase 4: Sibling fusion
        // This fuses operations that share the same input (siblings), allowing
        // the shared input to be read once and multiple outputs computed together.
        if config.enableSiblingFusion {
            current = siblingFusion.fuse(current)
        }

        // Phase 5: Horizontal fusion
        // This batches small independent operations of the same type to reduce
        // kernel launch overhead. Common in optimizer updates (Adam, SGD).
        if config.enableHorizontalFusion {
            current = horizontalFusion.fuse(current)
        }

        // Phase 6: Layout assignment
        // This assigns optimal memory layouts per operation (NHWC for convolution,
        // row/column major for matmul) and inserts transposes when needed.
        if config.enableLayoutAssignment {
            current = layoutAssignment.optimize(current)
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

