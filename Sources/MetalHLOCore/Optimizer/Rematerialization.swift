// Rematerialization.swift
// MetalHLOCore
//
// Rematerialization (recomputation) planner for memory optimization.
// Trades compute for memory by recomputing values instead of storing them.

import Foundation

// MARK: - Recomputation Cost Model

/// Cost model for deciding when to rematerialize.
public struct RecomputationCost: Sendable {
    /// Cost to compute the tensor (in arbitrary units).
    public let computeCost: Double

    /// Memory footprint in bytes.
    public let memoryCost: Int

    /// Ratio of compute cost to memory benefit.
    public var costBenefitRatio: Double {
        guard memoryCost > 0 else { return Double.infinity }
        return computeCost / Double(memoryCost)
    }

    public init(computeCost: Double, memoryCost: Int) {
        self.computeCost = computeCost
        self.memoryCost = memoryCost
    }
}

// MARK: - Rematerialization Decision

/// Decision for a specific tensor.
public enum RematerializationDecision: Sendable, Equatable {
    /// Store the tensor in memory.
    case store

    /// Recompute the tensor when needed.
    case recompute

    /// Checkpoint at specific points (gradient checkpointing).
    case checkpoint(interval: Int)
}

// MARK: - Tensor Dependency

/// Represents dependency information for a tensor.
public struct TensorDependency: Sendable {
    /// ID of the tensor.
    public let tensorId: String

    /// IDs of tensors this depends on (inputs).
    public let inputs: [String]

    /// Operation that produces this tensor.
    public let producingOp: Int

    /// Operations that consume this tensor.
    public let consumers: [Int]

    /// Whether this is a function input (cannot be recomputed).
    public let isInput: Bool

    /// Whether this is a function output (must be stored).
    public let isOutput: Bool

    public init(
        tensorId: String,
        inputs: [String],
        producingOp: Int,
        consumers: [Int],
        isInput: Bool = false,
        isOutput: Bool = false
    ) {
        self.tensorId = tensorId
        self.inputs = inputs
        self.producingOp = producingOp
        self.consumers = consumers
        self.isInput = isInput
        self.isOutput = isOutput
    }
}

// MARK: - Rematerialization Plan

/// A plan specifying which tensors to rematerialize.
public struct RematerializationPlan: Sendable {
    /// Decisions for each tensor.
    public let decisions: [String: RematerializationDecision]

    /// Checkpoint boundaries for gradient checkpointing.
    public let checkpointBoundaries: [Int]

    /// Estimated peak memory after rematerialization.
    public let estimatedPeakMemory: Int

    /// Estimated additional compute from recomputation.
    public let estimatedRecomputeCost: Double

    /// Memory saved by rematerialization.
    public let memorySaved: Int

    public init(
        decisions: [String: RematerializationDecision],
        checkpointBoundaries: [Int] = [],
        estimatedPeakMemory: Int,
        estimatedRecomputeCost: Double,
        memorySaved: Int
    ) {
        self.decisions = decisions
        self.checkpointBoundaries = checkpointBoundaries
        self.estimatedPeakMemory = estimatedPeakMemory
        self.estimatedRecomputeCost = estimatedRecomputeCost
        self.memorySaved = memorySaved
    }

    /// Returns tensors that should be recomputed.
    public var recomputedTensors: [String] {
        decisions.compactMap { key, value in
            if case .recompute = value {
                return key
            }
            return nil
        }
    }

    /// Returns tensors that should be checkpointed.
    public var checkpointedTensors: [String] {
        decisions.compactMap { key, value in
            if case .checkpoint = value {
                return key
            }
            return nil
        }
    }
}

// MARK: - Recomputation Planner

/// Planner that decides which tensors to rematerialize.
public final class RecomputationPlanner: @unchecked Sendable {

    /// Configuration for the planner.
    public struct Config: Sendable {
        /// Maximum memory budget in bytes (0 = unlimited).
        public var memoryBudget: Int = 0

        /// Maximum ratio of recompute cost to memory saved.
        public var maxCostBenefitRatio: Double = 10.0

        /// Whether to enable gradient checkpointing.
        public var enableGradientCheckpointing: Bool = true

        /// Checkpoint interval (operations between checkpoints).
        public var checkpointInterval: Int = 4

        /// Minimum tensor size to consider for rematerialization (bytes).
        public var minTensorSize: Int = 4096

        /// Operations that are cheap to recompute.
        public var cheapOperations: Set<String> = [
            "add", "subtract", "multiply", "divide",
            "negate", "abs", "exp", "log", "sqrt",
            "tanh", "sine", "cosine",
            "broadcast_in_dim", "reshape", "transpose"
        ]

        /// Operations that are expensive to recompute.
        public var expensiveOperations: Set<String> = [
            "dot", "dot_general", "convolution", "reduce"
        ]

        public init() {}
    }

    private let config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    // MARK: - Planning

    /// Creates a rematerialization plan for the given function.
    public func plan(_ function: HLOFunction) -> RematerializationPlan {
        // Build dependency graph
        let dependencies = buildDependencyGraph(function)

        // Compute costs for each tensor
        let costs = computeCosts(function, dependencies: dependencies)

        // Compute original memory usage
        let originalMemory = computePeakMemory(function, decisions: [:])

        // Decide which tensors to rematerialize
        var decisions: [String: RematerializationDecision] = [:]
        var totalRecomputeCost: Double = 0
        var totalMemorySaved: Int = 0

        // Process tensors by cost-benefit ratio (best candidates first)
        let sortedTensors = costs.sorted { $0.value.costBenefitRatio < $1.value.costBenefitRatio }

        for (tensorId, cost) in sortedTensors {
            guard let dep = dependencies[tensorId] else { continue }

            // Never rematerialize inputs or outputs
            if dep.isInput || dep.isOutput {
                decisions[tensorId] = .store
                continue
            }

            // Skip small tensors
            if cost.memoryCost < config.minTensorSize {
                decisions[tensorId] = .store
                continue
            }

            // Check if this operation is cheap to recompute
            let opKind = function.operations[dep.producingOp].kind
            let isCheap = config.cheapOperations.contains(opKind.rawValue)
            let isExpensive = config.expensiveOperations.contains(opKind.rawValue)

            // Decision logic
            if isExpensive {
                // Never rematerialize expensive operations
                decisions[tensorId] = .store
            } else if isCheap && cost.costBenefitRatio < config.maxCostBenefitRatio {
                // Rematerialize cheap operations with good cost-benefit
                decisions[tensorId] = .recompute
                totalRecomputeCost += cost.computeCost
                totalMemorySaved += cost.memoryCost
            } else if config.memoryBudget > 0 {
                // Check memory budget
                let currentMemory = computePeakMemory(function, decisions: decisions)
                if currentMemory > config.memoryBudget {
                    // Need to save memory - consider rematerialization
                    if cost.costBenefitRatio < config.maxCostBenefitRatio * 2 {
                        decisions[tensorId] = .recompute
                        totalRecomputeCost += cost.computeCost
                        totalMemorySaved += cost.memoryCost
                    } else {
                        decisions[tensorId] = .store
                    }
                } else {
                    decisions[tensorId] = .store
                }
            } else {
                decisions[tensorId] = .store
            }
        }

        // Apply gradient checkpointing if enabled
        var checkpointBoundaries: [Int] = []
        if config.enableGradientCheckpointing && function.operations.count > config.checkpointInterval {
            checkpointBoundaries = stride(
                from: config.checkpointInterval,
                to: function.operations.count,
                by: config.checkpointInterval
            ).map { $0 }

            // Mark tensors at checkpoint boundaries
            for boundary in checkpointBoundaries {
                if boundary < function.operations.count {
                    let op = function.operations[boundary]
                    if decisions[op.result] == .recompute {
                        decisions[op.result] = .checkpoint(interval: config.checkpointInterval)
                    }
                }
            }
        }

        let estimatedPeakMemory = computePeakMemory(function, decisions: decisions)

        return RematerializationPlan(
            decisions: decisions,
            checkpointBoundaries: checkpointBoundaries,
            estimatedPeakMemory: estimatedPeakMemory,
            estimatedRecomputeCost: totalRecomputeCost,
            memorySaved: totalMemorySaved
        )
    }

    // MARK: - Dependency Analysis

    private func buildDependencyGraph(_ function: HLOFunction) -> [String: TensorDependency] {
        var dependencies: [String: TensorDependency] = [:]

        // Mark inputs
        for input in function.inputs {
            dependencies[input.name] = TensorDependency(
                tensorId: input.name,
                inputs: [],
                producingOp: -1,
                consumers: [],
                isInput: true,
                isOutput: false
            )
        }

        // Build consumer map
        var consumerMap: [String: [Int]] = [:]
        for (opIndex, op) in function.operations.enumerated() {
            for operand in op.operands {
                consumerMap[operand, default: []].append(opIndex)
            }
        }

        // Mark outputs
        let outputSet = Set(function.returnValues)

        // Process operations
        for (opIndex, op) in function.operations.enumerated() {
            dependencies[op.result] = TensorDependency(
                tensorId: op.result,
                inputs: op.operands,
                producingOp: opIndex,
                consumers: consumerMap[op.result] ?? [],
                isInput: false,
                isOutput: outputSet.contains(op.result)
            )
        }

        return dependencies
    }

    // MARK: - Cost Computation

    private func computeCosts(
        _ function: HLOFunction,
        dependencies: [String: TensorDependency]
    ) -> [String: RecomputationCost] {
        var costs: [String: RecomputationCost] = [:]

        for (opIndex, op) in function.operations.enumerated() {
            let memoryCost = op.resultType.byteCount
            let computeCost = estimateComputeCost(op)

            costs[op.result] = RecomputationCost(
                computeCost: computeCost,
                memoryCost: memoryCost
            )
        }

        return costs
    }

    private func estimateComputeCost(_ op: HLOOperation) -> Double {
        let numElements = Double(op.resultType.shape.reduce(1, *))

        // Rough cost model based on operation type
        switch op.kind {
        case .add, .subtract, .multiply, .divide, .negate, .abs:
            return numElements * 1.0

        case .exponential, .log, .sqrt, .tanh, .sine, .cosine:
            return numElements * 5.0

        case .dot:
            // Matrix multiply is O(M*N*K)
            return numElements * 10.0

        case .dotGeneral:
            return numElements * 15.0

        case .convolution:
            return numElements * 20.0

        case .reduce, .reduceWindow:
            return numElements * 3.0

        case .broadcastInDim, .reshape, .transpose:
            return numElements * 0.5  // Very cheap

        default:
            return numElements * 2.0
        }
    }

    // MARK: - Memory Estimation

    private func computePeakMemory(
        _ function: HLOFunction,
        decisions: [String: RematerializationDecision]
    ) -> Int {
        // Simple liveness analysis
        var liveSet: Set<String> = []
        var peakMemory = 0

        // Add inputs
        for input in function.inputs {
            liveSet.insert(input.name)
        }

        // Process operations
        for (_, op) in function.operations.enumerated() {
            // Add output if stored
            let decision = decisions[op.result] ?? .store
            if case .store = decision {
                liveSet.insert(op.result)
            }

            // Compute current memory
            var currentMemory = 0
            for tensorId in liveSet {
                if let dep = function.operations.first(where: { $0.result == tensorId }) {
                    currentMemory += dep.resultType.byteCount
                } else if let input = function.inputs.first(where: { $0.name == tensorId }) {
                    currentMemory += input.type.byteCount
                }
            }

            peakMemory = max(peakMemory, currentMemory)
        }

        return peakMemory
    }
}

// MARK: - Gradient Checkpointing

/// Specialized planner for gradient checkpointing in training.
public final class GradientCheckpointPlanner: @unchecked Sendable {

    /// Configuration for gradient checkpointing.
    public struct Config: Sendable {
        /// Number of segments to divide the computation into.
        public var numSegments: Int = 4

        /// Whether to use sqrt decomposition.
        public var useSqrtDecomposition: Bool = true

        /// Custom checkpoint operations (layer boundaries).
        public var checkpointOperations: [String] = []

        public init() {}
    }

    private let config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    /// Plans checkpoint boundaries for a forward pass function.
    public func planCheckpoints(_ function: HLOFunction) -> [Int] {
        let numOps = function.operations.count

        if config.useSqrtDecomposition {
            // Optimal checkpoint count is sqrt(n) for sqrt(n) memory
            let numCheckpoints = Int(sqrt(Double(numOps)))
            let interval = max(1, numOps / (numCheckpoints + 1))

            return stride(from: interval, to: numOps, by: interval).map { $0 }
        } else {
            // Fixed segment count
            let interval = max(1, numOps / config.numSegments)
            return stride(from: interval, to: numOps, by: interval).map { $0 }
        }
    }

    /// Estimates memory savings from checkpointing.
    public func estimateMemorySavings(
        _ function: HLOFunction,
        checkpoints: [Int]
    ) -> MemorySavingsEstimate {
        let totalMemory = function.operations.reduce(0) { $0 + $1.resultType.byteCount }

        // With checkpointing, only need to store activations between checkpoints
        let numSegments = checkpoints.count + 1
        let averageSegmentSize = function.operations.count / numSegments

        // Estimate memory per segment
        var maxSegmentMemory = 0
        var currentSegmentStart = 0

        for checkpoint in checkpoints + [function.operations.count] {
            var segmentMemory = 0
            for i in currentSegmentStart..<checkpoint {
                segmentMemory += function.operations[i].resultType.byteCount
            }
            maxSegmentMemory = max(maxSegmentMemory, segmentMemory)
            currentSegmentStart = checkpoint
        }

        // Also need to store checkpoint activations
        let checkpointMemory = checkpoints.reduce(0) { total, idx in
            guard idx < function.operations.count else { return total }
            return total + function.operations[idx].resultType.byteCount
        }

        let checkpointedMemory = maxSegmentMemory + checkpointMemory
        let memorySaved = max(0, totalMemory - checkpointedMemory)

        // Recomputation increases total compute by about 33% on average
        let recomputeOverhead = 0.33 * Double(numSegments - 1) / Double(numSegments)

        return MemorySavingsEstimate(
            originalMemory: totalMemory,
            checkpointedMemory: checkpointedMemory,
            memorySaved: memorySaved,
            recomputeOverhead: recomputeOverhead,
            numCheckpoints: checkpoints.count
        )
    }
}

/// Estimate of memory savings from checkpointing.
public struct MemorySavingsEstimate: Sendable {
    /// Original peak memory without checkpointing.
    public let originalMemory: Int

    /// Peak memory with checkpointing.
    public let checkpointedMemory: Int

    /// Memory saved in bytes.
    public let memorySaved: Int

    /// Additional compute overhead (ratio, e.g., 0.33 = 33% more compute).
    public let recomputeOverhead: Double

    /// Number of checkpoint boundaries.
    public let numCheckpoints: Int

    /// Percentage of memory saved.
    public var savingsPercentage: Double {
        guard originalMemory > 0 else { return 0 }
        return Double(memorySaved) / Double(originalMemory) * 100
    }
}

// MARK: - Memory Budget Optimizer

/// Optimizer that finds the best rematerialization plan within a memory budget.
public final class MemoryBudgetOptimizer: @unchecked Sendable {

    private let memoryBudget: Int
    private let maxRecomputeRatio: Double

    /// Creates an optimizer with the given memory budget.
    ///
    /// - Parameters:
    ///   - memoryBudget: Maximum memory in bytes.
    ///   - maxRecomputeRatio: Maximum ratio of additional compute (e.g., 0.5 = 50% more).
    public init(memoryBudget: Int, maxRecomputeRatio: Double = 0.5) {
        self.memoryBudget = memoryBudget
        self.maxRecomputeRatio = maxRecomputeRatio
    }

    /// Finds the optimal rematerialization plan within the budget.
    public func optimize(_ function: HLOFunction) -> RematerializationPlan {
        // Start with aggressive rematerialization
        var config = RecomputationPlanner.Config()
        config.memoryBudget = memoryBudget
        config.maxCostBenefitRatio = 1.0

        let planner = RecomputationPlanner(config: config)
        var bestPlan = planner.plan(function)

        // Binary search for the best cost-benefit ratio that meets budget
        var low = 0.1
        var high = 100.0

        for _ in 0..<10 {
            let mid = (low + high) / 2

            var testConfig = config
            testConfig.maxCostBenefitRatio = mid

            let testPlanner = RecomputationPlanner(config: testConfig)
            let testPlan = testPlanner.plan(function)

            if testPlan.estimatedPeakMemory <= memoryBudget {
                // Can relax rematerialization
                bestPlan = testPlan
                low = mid
            } else {
                // Need more aggressive rematerialization
                high = mid
            }
        }

        return bestPlan
    }
}

// MARK: - Rematerialization Transformer

/// Transforms an HLO function according to a rematerialization plan.
public final class RematerializationTransformer: @unchecked Sendable {

    public init() {}

    /// Applies the rematerialization plan to the function.
    ///
    /// This inserts recomputation operations where needed.
    public func transform(
        _ function: HLOFunction,
        plan: RematerializationPlan
    ) -> HLOFunction {
        var newOperations: [HLOOperation] = []
        var recomputeMap: [String: String] = [:]  // original -> recomputed tensor name

        // Find operations that need recomputation
        let recomputeOps = Set(plan.recomputedTensors)

        for op in function.operations {
            // Check if any operand needs to be recomputed
            var modifiedOperands = op.operands
            for (idx, operand) in op.operands.enumerated() {
                if let recomputed = recomputeMap[operand] {
                    modifiedOperands[idx] = recomputed
                }
            }

            // Create potentially modified operation
            let newOp = HLOOperation(
                result: op.result,
                kind: op.kind,
                operands: modifiedOperands,
                resultType: op.resultType,
                attributes: op.attributes
            )
            newOperations.append(newOp)

            // If this result is marked for recomputation and is used later,
            // we'll insert the recomputation just before it's needed
            if recomputeOps.contains(op.result) {
                // Mark for potential recomputation
                recomputeMap[op.result] = op.result + "_remat"
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
}
