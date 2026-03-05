// PartitionerTests.swift
// MetalHLOCoreTests
//
// Tests for the graph partitioner: SubFunctionExtractor, TensorTransferManager,
// ANEAnalyzer improvements, and GPUANEPartitioner.

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("Graph Partitioner", .serialized)
struct PartitionerTests {

    // MARK: - Test Helpers

    /// Creates a simple HLO operation for testing.
    private func makeOp(
        result: String,
        kind: HLOOpKind,
        operands: [String],
        shape: [Int] = [4],
        elementType: ElementType = .float32
    ) -> HLOOperation {
        HLOOperation(
            result: result,
            kind: kind,
            operands: operands,
            resultType: TensorType(shape: shape, elementType: elementType)
        )
    }

    /// Creates a simple HLO function for testing.
    private func makeFunction(
        inputs: [(String, [Int])],
        ops: [(String, HLOOpKind, [String], [Int])],
        returnValues: [String]
    ) -> HLOFunction {
        HLOFunction(
            name: "main",
            inputs: inputs.map { name, shape in
                HLOArgument(name: name, type: TensorType(shape: shape, elementType: .float32))
            },
            outputTypes: [TensorType(shape: ops.last!.3, elementType: .float32)],
            operations: ops.map { result, kind, operands, shape in
                HLOOperation(
                    result: result,
                    kind: kind,
                    operands: operands,
                    resultType: TensorType(shape: shape, elementType: .float32)
                )
            },
            returnValues: returnValues
        )
    }

    // MARK: - SubFunctionExtractor Tests

    @Test("Extract sub-function with single op")
    func extractSingleOp() {
        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [
                ("%0", .add, ["%x", "%y"], [4]),
                ("%1", .multiply, ["%0", "%x"], [4]),
            ],
            returnValues: ["%1"]
        )

        let opIds: Set<String> = ["%0"]
        let consumers = SubFunctionExtractor.findExternalConsumers(of: opIds, in: function)
        let sub = SubFunctionExtractor.extract(from: function, operationIds: opIds, outputConsumers: consumers)

        #expect(sub.inputs.count == 2) // %x and %y
        #expect(sub.operations.count == 1)
        #expect(sub.operations[0].result == "%0")
        #expect(sub.returnValues.contains("%0")) // consumed by %1 outside partition
    }

    @Test("Extract sub-function preserves topological order")
    func extractPreservesOrder() {
        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [
                ("%0", .add, ["%x", "%y"], [4]),
                ("%1", .multiply, ["%0", "%y"], [4]),
                ("%2", .subtract, ["%1", "%x"], [4]),
            ],
            returnValues: ["%2"]
        )

        let opIds: Set<String> = ["%0", "%1"]
        let consumers = SubFunctionExtractor.findExternalConsumers(of: opIds, in: function)
        let sub = SubFunctionExtractor.extract(from: function, operationIds: opIds, outputConsumers: consumers)

        #expect(sub.operations.count == 2)
        #expect(sub.operations[0].result == "%0")
        #expect(sub.operations[1].result == "%1")
        #expect(sub.returnValues.contains("%1")) // consumed by %2
    }

    @Test("Extract sub-function creates synthetic inputs for cross-partition deps")
    func extractCrossPartitionInputs() {
        let function = makeFunction(
            inputs: [("%x", [4])],
            ops: [
                ("%0", .add, ["%x", "%x"], [4]),     // partition A
                ("%1", .multiply, ["%0", "%x"], [4]), // partition B - needs %0 from A
            ],
            returnValues: ["%1"]
        )

        let opIds: Set<String> = ["%1"]
        let consumers = SubFunctionExtractor.findExternalConsumers(of: opIds, in: function)
        let sub = SubFunctionExtractor.extract(from: function, operationIds: opIds, outputConsumers: consumers)

        // Should have %x (original input) and %0 (cross-partition dependency)
        #expect(sub.inputs.count == 2)
        let inputNames = Set(sub.inputs.map { $0.name })
        #expect(inputNames.contains("%x"))
        #expect(inputNames.contains("%0"))
    }

    @Test("Find external consumers correctly")
    func findExternalConsumers() {
        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [
                ("%0", .add, ["%x", "%y"], [4]),
                ("%1", .multiply, ["%0", "%x"], [4]),
                ("%2", .subtract, ["%1", "%0"], [4]),
            ],
            returnValues: ["%2"]
        )

        // Check consumers of partition {%0}
        let consumers = SubFunctionExtractor.findExternalConsumers(
            of: ["%0"],
            in: function
        )
        // %0 is consumed by %1 and %2
        #expect(consumers.contains("%0"))
    }

    // MARK: - ANEAnalyzer Tests

    @Test("Analyzer classifies ops correctly")
    func analyzerClassifiesOps() {
        let analyzer = ANEAnalyzer()

        // Optimal ops
        let dotOp = makeOp(result: "%0", kind: .dot, operands: ["%x", "%y"], shape: [64, 64])
        let dotAnalysis = analyzer.analyze(dotOp)
        #expect(dotAnalysis.compatibility == .optimal)

        let convOp = makeOp(result: "%1", kind: .convolution, operands: ["%x", "%w"], shape: [1, 64, 32, 32])
        let convAnalysis = analyzer.analyze(convOp)
        #expect(convAnalysis.compatibility == .optimal)

        // Compatible ops
        let addOp = makeOp(result: "%2", kind: .add, operands: ["%x", "%y"], shape: [64, 64])
        let addAnalysis = analyzer.analyze(addOp)
        #expect(addAnalysis.compatibility == .compatible)

        let tanhOp = makeOp(result: "%3", kind: .tanh, operands: ["%x"], shape: [64, 64])
        let tanhAnalysis = analyzer.analyze(tanhOp)
        #expect(tanhAnalysis.compatibility == .compatible)

        // Incompatible ops
        let gatherOp = makeOp(result: "%4", kind: .gather, operands: ["%x", "%y"], shape: [64, 64])
        let gatherAnalysis = analyzer.analyze(gatherOp)
        #expect(gatherAnalysis.compatibility == .incompatible)

        let sortOp = makeOp(result: "%5", kind: .sort, operands: ["%x"], shape: [64, 64])
        let sortAnalysis = analyzer.analyze(sortOp)
        #expect(sortAnalysis.compatibility == .incompatible)
    }

    @Test("Analyzer classifies newly added compatible ops")
    func analyzerNewCompatibleOps() {
        let analyzer = ANEAnalyzer()

        // These were added to aneCompatibleOps in this change
        let logisticOp = makeOp(result: "%0", kind: .logistic, operands: ["%x"], shape: [64, 64])
        #expect(analyzer.analyze(logisticOp).compatibility == .compatible)

        let absOp = makeOp(result: "%1", kind: .abs, operands: ["%x"], shape: [64, 64])
        #expect(analyzer.analyze(absOp).compatibility == .compatible)

        let selectOp = makeOp(result: "%2", kind: .select, operands: ["%c", "%x", "%y"], shape: [64, 64])
        #expect(analyzer.analyze(selectOp).compatibility == .compatible)

        let sqrtOp = makeOp(result: "%3", kind: .sqrt, operands: ["%x"], shape: [64, 64])
        #expect(analyzer.analyze(sqrtOp).compatibility == .compatible)
    }

    @Test("Analyzer applies chaining bonus for consecutive ANE ops")
    func analyzerChainingBonus() {
        let analyzer = ANEAnalyzer()

        // Single op (no chaining)
        let singleFunc = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [("%0", .add, ["%x", "%y"], [64, 64])],
            returnValues: ["%0"]
        )
        let singleAnalysis = analyzer.analyzeFunction(singleFunc)
        let singleTime = singleAnalysis.operationAnalyses[0].estimatedANETime

        // Chain of 4 ops
        let chainFunc = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [
                ("%0", .add, ["%x", "%y"], [64, 64]),
                ("%1", .multiply, ["%0", "%y"], [64, 64]),
                ("%2", .subtract, ["%1", "%x"], [64, 64]),
                ("%3", .add, ["%2", "%y"], [64, 64]),
            ],
            returnValues: ["%3"]
        )
        let chainAnalysis = analyzer.analyzeFunction(chainFunc)
        let chainTime = chainAnalysis.operationAnalyses[0].estimatedANETime

        // Chained ops should have lower per-op ANE time due to amortized dispatch
        #expect(chainTime < singleTime, "Chaining bonus should reduce per-op ANE time")
    }

    @Test("dotGeneral is classified as optimal")
    func dotGeneralOptimal() {
        let analyzer = ANEAnalyzer()
        let op = makeOp(result: "%0", kind: .dotGeneral, operands: ["%x", "%y"], shape: [64, 64])
        let analysis = analyzer.analyze(op)
        #expect(analysis.compatibility == .optimal)
    }

    // MARK: - GPUANEPartitioner Tests

    @Test("Partitioner groups consecutive same-device ops")
    func partitionerGroupsConsecutive() {
        let function = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [
                ("%0", .add, ["%x", "%y"], [64, 64]),        // ANE compatible
                ("%1", .multiply, ["%0", "%y"], [64, 64]),    // ANE compatible
                ("%2", .gather, ["%1", "%x"], [64, 64]),      // GPU only (incompatible)
                ("%3", .add, ["%2", "%y"], [64, 64]),          // ANE compatible
            ],
            returnValues: ["%3"]
        )

        let partitioner = GPUANEPartitioner(
            config: .init(
                minOpsForANEPartition: 1,
                maxTransferOverheadMs: 100.0,
                enablePipelining: false,
                balanceLoad: false
            )
        )
        let plan = partitioner.partition(function)

        // The gather op must be in a GPU partition
        let gatherPartition = plan.partitions.first { partition in
            partition.operationIds.contains("%2")
        }
        #expect(gatherPartition?.device == .gpu)

        // All ops should be covered
        let allOps = plan.partitions.flatMap { $0.operationIds }
        #expect(allOps.count == 4)
    }

    @Test("Small ANE partitions get merged to GPU")
    func smallPartitionsMergedToGPU() {
        // A single ANE op between GPU ops should be absorbed by GPU
        let partitioner = GPUANEPartitioner(
            config: .init(
                minOpsForANEPartition: 3,  // Require at least 3 ops for ANE
                maxTransferOverheadMs: 1.0,
                enablePipelining: false,
                balanceLoad: false
            )
        )

        let function = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [
                ("%0", .gather, ["%x", "%y"], [64, 64]),  // GPU
                ("%1", .add, ["%0", "%y"], [64, 64]),      // Single ANE op - too small
                ("%2", .gather, ["%1", "%x"], [64, 64]),   // GPU
            ],
            returnValues: ["%2"]
        )

        let plan = partitioner.partition(function)

        // The single add op should be merged into GPU
        let anePartitions = plan.partitions.filter { $0.device == .ane }
        #expect(anePartitions.isEmpty, "Single ANE op should be merged into GPU partition")
    }

    @Test("Cross-partition transfer tensors are identified")
    func transferTensorsIdentified() {
        let function = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [
                ("%0", .add, ["%x", "%y"], [64, 64]),    // ANE
                ("%1", .gather, ["%0", "%x"], [64, 64]),  // GPU - needs %0 from ANE
                ("%2", .add, ["%1", "%y"], [64, 64]),     // ANE - needs %1 from GPU
            ],
            returnValues: ["%2"]
        )

        let partitioner = GPUANEPartitioner(
            config: .init(
                minOpsForANEPartition: 1,
                maxTransferOverheadMs: 100.0,
                enablePipelining: false,
                balanceLoad: false
            )
        )
        let plan = partitioner.partition(function)

        // Check that some partitions have cross-device tensors
        let hasTransfers = plan.partitions.contains { partition in
            !partition.inputTensors.isEmpty || !partition.outputTensors.isEmpty
        }
        // If partitions span different devices, there should be transfers
        let devices = Set(plan.partitions.map { $0.device })
        if devices.count > 1 {
            #expect(hasTransfers, "Multi-device plan should have transfer tensors")
        }
    }

    @Test("Partition plan has valid execution order")
    func validExecutionOrder() {
        let function = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [
                ("%0", .add, ["%x", "%y"], [64, 64]),
                ("%1", .multiply, ["%0", "%y"], [64, 64]),
                ("%2", .subtract, ["%1", "%x"], [64, 64]),
            ],
            returnValues: ["%2"]
        )

        let partitioner = GPUANEPartitioner()
        let plan = partitioner.partition(function)

        // Execution order should cover all partitions
        #expect(plan.executionOrder.count == plan.partitions.count)

        // Each partition index should appear exactly once
        let orderSet = Set(plan.executionOrder)
        #expect(orderSet.count == plan.executionOrder.count)

        // All indices should be valid
        for idx in plan.executionOrder {
            #expect(idx >= 0 && idx < plan.partitions.count)
        }
    }

    @Test("Transfer time uses actual tensor sizes")
    func transferTimeUsesTensorSizes() {
        // Large tensors should result in higher transfer time
        let smallFunc = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [
                ("%0", .add, ["%x", "%y"], [4]),
                ("%1", .gather, ["%0", "%x"], [4]),
            ],
            returnValues: ["%1"]
        )

        let largeFunc = makeFunction(
            inputs: [("%x", [1024, 1024]), ("%y", [1024, 1024])],
            ops: [
                ("%0", .add, ["%x", "%y"], [1024, 1024]),
                ("%1", .gather, ["%0", "%x"], [1024, 1024]),
            ],
            returnValues: ["%1"]
        )

        let partitioner = GPUANEPartitioner(
            config: .init(
                minOpsForANEPartition: 1,
                maxTransferOverheadMs: 100.0,
                enablePipelining: false,
                balanceLoad: false
            )
        )
        let smallPlan = partitioner.partition(smallFunc)
        let largePlan = partitioner.partition(largeFunc)

        // If both have cross-device transfers, larger should have higher transfer time
        if smallPlan.estimatedTransferTimeMs > 0 && largePlan.estimatedTransferTimeMs > 0 {
            #expect(
                largePlan.estimatedTransferTimeMs > smallPlan.estimatedTransferTimeMs,
                "Larger tensors should have higher transfer time"
            )
        }
    }

    // MARK: - Function Analysis Tests

    @Test("All ANE ops function")
    func allANEFunction() {
        let function = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [
                ("%0", .add, ["%x", "%y"], [64, 64]),
                ("%1", .multiply, ["%0", "%y"], [64, 64]),
                ("%2", .tanh, ["%1"], [64, 64]),
            ],
            returnValues: ["%2"]
        )

        let analyzer = ANEAnalyzer()
        let analysis = analyzer.analyzeFunction(function)

        // All ops should be compatible or optimal
        #expect(analysis.incompatibleOpsCount == 0)
        #expect(analysis.compatibleOpsCount + analysis.optimalOpsCount == 3)
    }

    @Test("Mixed function has correct counts")
    func mixedFunctionCounts() {
        let function = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [
                ("%0", .add, ["%x", "%y"], [64, 64]),       // compatible
                ("%1", .gather, ["%0", "%x"], [64, 64]),     // incompatible
                ("%2", .dot, ["%0", "%y"], [64, 64]),        // optimal
            ],
            returnValues: ["%2"]
        )

        let analyzer = ANEAnalyzer()
        let analysis = analyzer.analyzeFunction(function)

        #expect(analysis.incompatibleOpsCount == 1)
        #expect(analysis.optimalOpsCount >= 1)
    }
}
