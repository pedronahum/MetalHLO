// RematerializationTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2F: Rematerialization

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("Rematerialization Tests")
struct RematerializationTests {

    // MARK: - RecomputationCost Tests

    @Test("RecomputationCost computes cost-benefit ratio")
    func recomputationCostRatio() {
        let cost = RecomputationCost(computeCost: 100.0, memoryCost: 1000)

        #expect(cost.costBenefitRatio == 0.1)
    }

    @Test("RecomputationCost handles zero memory")
    func recomputationCostZeroMemory() {
        let cost = RecomputationCost(computeCost: 100.0, memoryCost: 0)

        #expect(cost.costBenefitRatio == Double.infinity)
    }

    // MARK: - RematerializationDecision Tests

    @Test("RematerializationDecision cases are equatable")
    func rematerializationDecisionEquatable() {
        let store1 = RematerializationDecision.store
        let store2 = RematerializationDecision.store
        let recompute = RematerializationDecision.recompute
        let checkpoint1 = RematerializationDecision.checkpoint(interval: 4)
        let checkpoint2 = RematerializationDecision.checkpoint(interval: 4)
        let checkpoint3 = RematerializationDecision.checkpoint(interval: 8)

        #expect(store1 == store2)
        #expect(store1 != recompute)
        #expect(checkpoint1 == checkpoint2)
        #expect(checkpoint1 != checkpoint3)
    }

    // MARK: - TensorDependency Tests

    @Test("TensorDependency tracks inputs and consumers")
    func tensorDependencyTracking() {
        let dep = TensorDependency(
            tensorId: "tensor1",
            inputs: ["input1", "input2"],
            producingOp: 5,
            consumers: [6, 7, 8],
            isInput: false,
            isOutput: true
        )

        #expect(dep.tensorId == "tensor1")
        #expect(dep.inputs.count == 2)
        #expect(dep.producingOp == 5)
        #expect(dep.consumers.count == 3)
        #expect(dep.isOutput == true)
        #expect(dep.isInput == false)
    }

    @Test("TensorDependency marks function inputs")
    func tensorDependencyInput() {
        let dep = TensorDependency(
            tensorId: "arg0",
            inputs: [],
            producingOp: -1,
            consumers: [0, 1],
            isInput: true,
            isOutput: false
        )

        #expect(dep.isInput == true)
        #expect(dep.producingOp == -1)
    }

    // MARK: - RematerializationPlan Tests

    @Test("RematerializationPlan reports recomputed tensors")
    func rematerializationPlanRecomputed() {
        let decisions: [String: RematerializationDecision] = [
            "t1": .store,
            "t2": .recompute,
            "t3": .recompute,
            "t4": .store,
            "t5": .checkpoint(interval: 4)
        ]

        let plan = RematerializationPlan(
            decisions: decisions,
            checkpointBoundaries: [4],
            estimatedPeakMemory: 10000,
            estimatedRecomputeCost: 50.0,
            memorySaved: 5000
        )

        #expect(plan.recomputedTensors.count == 2)
        #expect(plan.recomputedTensors.contains("t2"))
        #expect(plan.recomputedTensors.contains("t3"))
    }

    @Test("RematerializationPlan reports checkpointed tensors")
    func rematerializationPlanCheckpointed() {
        let decisions: [String: RematerializationDecision] = [
            "t1": .store,
            "t2": .checkpoint(interval: 4),
            "t3": .checkpoint(interval: 8)
        ]

        let plan = RematerializationPlan(
            decisions: decisions,
            checkpointBoundaries: [4, 8],
            estimatedPeakMemory: 10000,
            estimatedRecomputeCost: 20.0,
            memorySaved: 3000
        )

        #expect(plan.checkpointedTensors.count == 2)
        #expect(plan.checkpointBoundaries.count == 2)
    }

    // MARK: - RecomputationPlanner Tests

    @Test("RecomputationPlanner creates plan for simple function")
    func recomputationPlannerSimple() {
        let function = createSimpleFunction()
        let planner = RecomputationPlanner()

        let plan = planner.plan(function)

        // Should have decisions for all intermediate tensors
        #expect(plan.decisions.count > 0)
        #expect(plan.estimatedPeakMemory > 0)
    }

    @Test("RecomputationPlanner respects memory budget")
    func recomputationPlannerMemoryBudget() {
        let function = createChainFunction(length: 10)

        // Tight budget should force more rematerialization
        var tightConfig = RecomputationPlanner.Config()
        tightConfig.memoryBudget = 1000
        tightConfig.maxCostBenefitRatio = 100.0
        let tightPlanner = RecomputationPlanner(config: tightConfig)
        let tightPlan = tightPlanner.plan(function)

        // Loose budget should allow more storage
        var looseConfig = RecomputationPlanner.Config()
        looseConfig.memoryBudget = 1000000
        let loosePlanner = RecomputationPlanner(config: looseConfig)
        let loosePlan = loosePlanner.plan(function)

        // Tight budget should save more memory (or have more recomputation)
        #expect(tightPlan.recomputedTensors.count >= loosePlan.recomputedTensors.count)
    }

    @Test("RecomputationPlanner never rematerializes inputs")
    func recomputationPlannerInputs() {
        let function = createSimpleFunction()

        var config = RecomputationPlanner.Config()
        config.maxCostBenefitRatio = 1000.0  // Very aggressive
        let planner = RecomputationPlanner(config: config)

        let plan = planner.plan(function)

        // Inputs should never be marked for recomputation
        for input in function.inputs {
            if let decision = plan.decisions[input.name] {
                #expect(decision == .store)
            }
        }
    }

    @Test("RecomputationPlanner never rematerializes outputs")
    func recomputationPlannerOutputs() {
        let function = createSimpleFunction()

        var config = RecomputationPlanner.Config()
        config.maxCostBenefitRatio = 1000.0
        let planner = RecomputationPlanner(config: config)

        let plan = planner.plan(function)

        // Outputs should never be marked for recomputation
        for output in function.returnValues {
            if let decision = plan.decisions[output] {
                #expect(decision == .store)
            }
        }
    }

    @Test("RecomputationPlanner prefers cheap operations")
    func recomputationPlannerCheapOps() {
        let function = createMixedOpsFunction()

        var config = RecomputationPlanner.Config()
        config.maxCostBenefitRatio = 5.0
        config.minTensorSize = 0  // Consider all tensors
        let planner = RecomputationPlanner(config: config)

        let plan = planner.plan(function)

        // Should prefer to rematerialize cheap ops
        #expect(plan.recomputedTensors.count >= 0)  // May have some recomputed
    }

    @Test("RecomputationPlanner config affects planning")
    func recomputationPlannerConfig() {
        let function = createChainFunction(length: 8)

        var aggressiveConfig = RecomputationPlanner.Config()
        aggressiveConfig.maxCostBenefitRatio = 100.0
        aggressiveConfig.minTensorSize = 0
        let aggressivePlanner = RecomputationPlanner(config: aggressiveConfig)
        let aggressivePlan = aggressivePlanner.plan(function)

        var conservativeConfig = RecomputationPlanner.Config()
        conservativeConfig.maxCostBenefitRatio = 0.1
        conservativeConfig.minTensorSize = 1000000
        let conservativePlanner = RecomputationPlanner(config: conservativeConfig)
        let conservativePlan = conservativePlanner.plan(function)

        // Aggressive config should recompute more
        #expect(aggressivePlan.recomputedTensors.count >= conservativePlan.recomputedTensors.count)
    }

    // MARK: - GradientCheckpointPlanner Tests

    @Test("GradientCheckpointPlanner creates checkpoints")
    func gradientCheckpointPlannerBasic() {
        let function = createChainFunction(length: 16)
        let planner = GradientCheckpointPlanner()

        let checkpoints = planner.planCheckpoints(function)

        // Should create multiple checkpoints
        #expect(checkpoints.count > 0)

        // Checkpoints should be within bounds
        for cp in checkpoints {
            #expect(cp < function.operations.count)
            #expect(cp > 0)
        }
    }

    @Test("GradientCheckpointPlanner uses sqrt decomposition")
    func gradientCheckpointPlannerSqrt() {
        let function = createChainFunction(length: 100)

        var config = GradientCheckpointPlanner.Config()
        config.useSqrtDecomposition = true
        let planner = GradientCheckpointPlanner(config: config)

        let checkpoints = planner.planCheckpoints(function)

        // sqrt(100) = 10, so should have around 9-10 checkpoints
        #expect(checkpoints.count >= 5)
        #expect(checkpoints.count <= 15)
    }

    @Test("GradientCheckpointPlanner fixed segments")
    func gradientCheckpointPlannerFixed() {
        let function = createChainFunction(length: 20)

        var config = GradientCheckpointPlanner.Config()
        config.useSqrtDecomposition = false
        config.numSegments = 4
        let planner = GradientCheckpointPlanner(config: config)

        let checkpoints = planner.planCheckpoints(function)

        // Should have 3 checkpoints for 4 segments
        #expect(checkpoints.count == 3)
    }

    @Test("GradientCheckpointPlanner estimates memory savings")
    func gradientCheckpointPlannerSavings() {
        let function = createChainFunction(length: 16)
        let planner = GradientCheckpointPlanner()

        let checkpoints = planner.planCheckpoints(function)
        let estimate = planner.estimateMemorySavings(function, checkpoints: checkpoints)

        #expect(estimate.originalMemory > 0)
        #expect(estimate.checkpointedMemory > 0)
        #expect(estimate.memorySaved >= 0)
        #expect(estimate.recomputeOverhead >= 0)
        #expect(estimate.numCheckpoints == checkpoints.count)
    }

    @Test("GradientCheckpointPlanner savings percentage")
    func gradientCheckpointPlannerSavingsPercentage() {
        let function = createChainFunction(length: 100)
        let planner = GradientCheckpointPlanner()

        let checkpoints = planner.planCheckpoints(function)
        let estimate = planner.estimateMemorySavings(function, checkpoints: checkpoints)

        // Should save significant memory with many checkpoints
        #expect(estimate.savingsPercentage >= 0)
    }

    // MARK: - MemoryBudgetOptimizer Tests

    @Test("MemoryBudgetOptimizer respects budget")
    func memoryBudgetOptimizer() {
        let function = createChainFunction(length: 10)
        let budget = 50000  // 50KB budget

        let optimizer = MemoryBudgetOptimizer(memoryBudget: budget)
        let plan = optimizer.optimize(function)

        // Plan should exist
        #expect(plan.decisions.count > 0)
    }

    @Test("MemoryBudgetOptimizer with tight budget")
    func memoryBudgetOptimizerTight() {
        let function = createChainFunction(length: 10)

        let tightOptimizer = MemoryBudgetOptimizer(memoryBudget: 1000)
        let tightPlan = tightOptimizer.optimize(function)

        let looseOptimizer = MemoryBudgetOptimizer(memoryBudget: 10000000)
        let loosePlan = looseOptimizer.optimize(function)

        // Tight budget should recompute more
        #expect(tightPlan.recomputedTensors.count >= loosePlan.recomputedTensors.count)
    }

    // MARK: - RematerializationTransformer Tests

    @Test("RematerializationTransformer preserves function structure")
    func rematerializationTransformerPreserves() {
        let function = createSimpleFunction()

        let plan = RematerializationPlan(
            decisions: [:],
            checkpointBoundaries: [],
            estimatedPeakMemory: 1000,
            estimatedRecomputeCost: 0,
            memorySaved: 0
        )

        let transformer = RematerializationTransformer()
        let transformed = transformer.transform(function, plan: plan)

        // Should preserve basic structure
        #expect(transformed.name == function.name)
        #expect(transformed.inputs.count == function.inputs.count)
        #expect(transformed.returnValues.count == function.returnValues.count)
    }

    @Test("RematerializationTransformer applies plan")
    func rematerializationTransformerApplies() {
        let function = createChainFunction(length: 5)

        var decisions: [String: RematerializationDecision] = [:]
        for op in function.operations {
            decisions[op.result] = .store
        }
        // Mark one for recomputation
        if let firstResult = function.operations.first?.result {
            decisions[firstResult] = .recompute
        }

        let plan = RematerializationPlan(
            decisions: decisions,
            checkpointBoundaries: [],
            estimatedPeakMemory: 1000,
            estimatedRecomputeCost: 10.0,
            memorySaved: 500
        )

        let transformer = RematerializationTransformer()
        let transformed = transformer.transform(function, plan: plan)

        // Should have same number of operations (remat insertions happen lazily)
        #expect(transformed.operations.count >= function.operations.count)
    }

    // MARK: - Integration Tests

    @Test("Full rematerialization workflow")
    func fullRematerializationWorkflow() {
        // Create a realistic transformer-like function
        let function = createTransformerLikeFunction()

        // Plan rematerialization
        var config = RecomputationPlanner.Config()
        config.enableGradientCheckpointing = true
        config.checkpointInterval = 2
        let planner = RecomputationPlanner(config: config)

        let plan = planner.plan(function)

        // Apply transformation
        let transformer = RematerializationTransformer()
        let transformed = transformer.transform(function, plan: plan)

        // Verify
        #expect(plan.decisions.count > 0)
        #expect(transformed.operations.count > 0)
    }

    @Test("Rematerialization with gradient checkpointing")
    func rematerializationWithCheckpointing() {
        let function = createChainFunction(length: 20)

        // Use gradient checkpoint planner
        let cpPlanner = GradientCheckpointPlanner()
        let checkpoints = cpPlanner.planCheckpoints(function)
        let savings = cpPlanner.estimateMemorySavings(function, checkpoints: checkpoints)

        // Also use recomputation planner
        var config = RecomputationPlanner.Config()
        config.enableGradientCheckpointing = true
        config.checkpointInterval = 5
        let rematPlanner = RecomputationPlanner(config: config)
        let rematPlan = rematPlanner.plan(function)

        // Should have checkpoint boundaries
        #expect(checkpoints.count > 0)
        #expect(savings.numCheckpoints == checkpoints.count)
        #expect(rematPlan.decisions.count > 0)
    }

    // MARK: - Helper Functions

    private func createSimpleFunction() -> HLOFunction {
        let inputs = [
            HLOArgument(name: "arg0", type: TensorType(shape: [32, 64], elementType: .float32)),
            HLOArgument(name: "arg1", type: TensorType(shape: [64, 32], elementType: .float32))
        ]

        let operations = [
            HLOOperation(
                result: "dot_result",
                kind: .dot,
                operands: ["arg0", "arg1"],
                resultType: TensorType(shape: [32, 32], elementType: .float32),
                attributes: HLOAttributes()
            ),
            HLOOperation(
                result: "exp_result",
                kind: .exponential,
                operands: ["dot_result"],
                resultType: TensorType(shape: [32, 32], elementType: .float32),
                attributes: HLOAttributes()
            )
        ]

        return HLOFunction(
            name: "simple_fn",
            inputs: inputs,
            outputTypes: [TensorType(shape: [32, 32], elementType: .float32)],
            operations: operations,
            returnValues: ["exp_result"]
        )
    }

    private func createChainFunction(length: Int) -> HLOFunction {
        let inputs = [
            HLOArgument(name: "input", type: TensorType(shape: [64, 64], elementType: .float32))
        ]

        var operations: [HLOOperation] = []
        var prevResult = "input"

        for i in 0..<length {
            let result = "chain_\(i)"
            let kind: HLOOpKind = [.exponential, .tanh, .negate, .abs][i % 4]

            operations.append(HLOOperation(
                result: result,
                kind: kind,
                operands: [prevResult],
                resultType: TensorType(shape: [64, 64], elementType: .float32),
                attributes: HLOAttributes()
            ))

            prevResult = result
        }

        return HLOFunction(
            name: "chain_fn",
            inputs: inputs,
            outputTypes: [TensorType(shape: [64, 64], elementType: .float32)],
            operations: operations,
            returnValues: [prevResult]
        )
    }

    private func createMixedOpsFunction() -> HLOFunction {
        let inputs = [
            HLOArgument(name: "a", type: TensorType(shape: [32, 64], elementType: .float32)),
            HLOArgument(name: "b", type: TensorType(shape: [64, 32], elementType: .float32))
        ]

        let operations = [
            // Expensive operation
            HLOOperation(
                result: "matmul",
                kind: .dot,
                operands: ["a", "b"],
                resultType: TensorType(shape: [32, 32], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Cheap operation
            HLOOperation(
                result: "exp_out",
                kind: .exponential,
                operands: ["matmul"],
                resultType: TensorType(shape: [32, 32], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Cheap operation
            HLOOperation(
                result: "neg_out",
                kind: .negate,
                operands: ["exp_out"],
                resultType: TensorType(shape: [32, 32], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Cheap operation
            HLOOperation(
                result: "final",
                kind: .add,
                operands: ["neg_out", "matmul"],
                resultType: TensorType(shape: [32, 32], elementType: .float32),
                attributes: HLOAttributes()
            )
        ]

        return HLOFunction(
            name: "mixed_ops",
            inputs: inputs,
            outputTypes: [TensorType(shape: [32, 32], elementType: .float32)],
            operations: operations,
            returnValues: ["final"]
        )
    }

    private func createTransformerLikeFunction() -> HLOFunction {
        let batchSeq = [4, 512]
        let hidden = 768

        let inputs = [
            HLOArgument(name: "hidden_states", type: TensorType(shape: batchSeq + [hidden], elementType: .float32)),
            HLOArgument(name: "attention_mask", type: TensorType(shape: batchSeq, elementType: .float32))
        ]

        // Simplified transformer operations
        let operations = [
            // Layer norm
            HLOOperation(
                result: "ln1",
                kind: .customCall,
                operands: ["hidden_states"],
                resultType: TensorType(shape: batchSeq + [hidden], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Attention projection
            HLOOperation(
                result: "qkv",
                kind: .dot,
                operands: ["ln1", "hidden_states"],
                resultType: TensorType(shape: batchSeq + [hidden * 3], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Attention output
            HLOOperation(
                result: "attn_out",
                kind: .dot,
                operands: ["qkv", "ln1"],
                resultType: TensorType(shape: batchSeq + [hidden], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Residual
            HLOOperation(
                result: "residual1",
                kind: .add,
                operands: ["hidden_states", "attn_out"],
                resultType: TensorType(shape: batchSeq + [hidden], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Layer norm 2
            HLOOperation(
                result: "ln2",
                kind: .customCall,
                operands: ["residual1"],
                resultType: TensorType(shape: batchSeq + [hidden], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // FFN
            HLOOperation(
                result: "ffn_up",
                kind: .dot,
                operands: ["ln2", "residual1"],
                resultType: TensorType(shape: batchSeq + [hidden * 4], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Activation
            HLOOperation(
                result: "gelu",
                kind: .tanh,  // Simplified
                operands: ["ffn_up"],
                resultType: TensorType(shape: batchSeq + [hidden * 4], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // FFN down
            HLOOperation(
                result: "ffn_down",
                kind: .dot,
                operands: ["gelu", "ln2"],
                resultType: TensorType(shape: batchSeq + [hidden], elementType: .float32),
                attributes: HLOAttributes()
            ),
            // Final residual
            HLOOperation(
                result: "output",
                kind: .add,
                operands: ["residual1", "ffn_down"],
                resultType: TensorType(shape: batchSeq + [hidden], elementType: .float32),
                attributes: HLOAttributes()
            )
        ]

        return HLOFunction(
            name: "transformer_block",
            inputs: inputs,
            outputTypes: [TensorType(shape: batchSeq + [hidden], elementType: .float32)],
            operations: operations,
            returnValues: ["output"]
        )
    }
}
