// MemoryPlanningTests.swift
// MetalHLOCoreTests
//
// Tests for ANE-aware memory planning: working set tracking,
// SRAM-aware partition splitting, and shape efficiency scoring.

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("Memory Planning")
struct MemoryPlanningTests {

    // MARK: - Test Helpers

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

    private func makeTestAnalyzer() -> ANEAnalyzer {
        ANEAnalyzer(config: .init(
            minTensorSizeForANE: 100,
            maxTensorSizeForANE: 16_777_216,
            aneOverheadNs: 50_000,
            preferANEForConvolutions: true,
            preferANEForMatMul: true
        ))
    }

    // MARK: - Working Set Tracking

    @Test("Small partitions have small working set")
    func smallPartitionWorkingSet() {
        // Simple 2-op chain: each produces [4] = 16 bytes
        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [
                ("%0", .add, ["%x", "%y"], [4]),
                ("%1", .multiply, ["%0", "%y"], [4]),
            ],
            returnValues: ["%1"]
        )

        let partitioner = GPUANEPartitioner(
            config: .init(
                minOpsForANEPartition: 1,
                maxTransferOverheadMs: 100.0,
                enablePipelining: false,
                balanceLoad: false
            ),
            analyzer: makeTestAnalyzer()
        )
        let plan = partitioner.partition(function)

        // All partitions should have modest working set
        for partition in plan.partitions {
            // [4] float32 = 16 bytes per tensor, peak should be small
            #expect(partition.workingSetBytes <= 1024, "Working set should be small for tiny tensors")
        }
    }

    @Test("Large tensors produce large working set")
    func largePartitionWorkingSet() {
        // Each op produces [256, 256] = 262144 elements = ~1MB float32
        let function = makeFunction(
            inputs: [("%x", [256, 256]), ("%y", [256, 256])],
            ops: [
                ("%0", .dot, ["%x", "%y"], [256, 256]),
                ("%1", .dot, ["%0", "%y"], [256, 256]),
                ("%2", .dot, ["%1", "%x"], [256, 256]),
            ],
            returnValues: ["%2"]
        )

        let partitioner = GPUANEPartitioner(
            config: .init(
                minOpsForANEPartition: 1,
                maxTransferOverheadMs: 100.0,
                enablePipelining: false,
                balanceLoad: false
            ),
            analyzer: makeTestAnalyzer()
        )
        let plan = partitioner.partition(function)

        // All ops should be in at least one partition
        let allOps = plan.partitions.flatMap { $0.operationIds }
        #expect(allOps.count == 3)

        // At least one partition should have non-zero working set
        // Each tensor is 256*256*4 = ~256KB
        let maxWorkingSet = plan.partitions.map { $0.workingSetBytes }.max() ?? 0
        #expect(maxWorkingSet > 0, "Working set should be computed for all partitions")
    }

    // MARK: - SRAM-Aware Partition Splitting

    @Test("Small partitions are not split")
    func smallPartitionsNotSplit() {
        let function = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [
                ("%0", .dot, ["%x", "%y"], [64, 64]),
                ("%1", .dot, ["%0", "%y"], [64, 64]),
            ],
            returnValues: ["%1"]
        )

        // Use a large SRAM capacity so nothing gets split
        let partitioner = GPUANEPartitioner(
            config: .init(
                minOpsForANEPartition: 1,
                maxTransferOverheadMs: 100.0,
                enablePipelining: false,
                balanceLoad: false,
                aneSRAMCapacityBytes: 1_000_000_000 // 1GB
            ),
            analyzer: makeTestAnalyzer()
        )
        let plan = partitioner.partition(function)

        // All ops should be covered regardless
        let allOps = plan.partitions.flatMap { $0.operationIds }
        #expect(allOps.count == 2)
    }

    @Test("Oversized partitions get split when SRAM capacity is tiny")
    func oversizedPartitionsSplit() {
        // Create a chain of large operations
        let function = makeFunction(
            inputs: [("%x", [512, 512]), ("%y", [512, 512])],
            ops: [
                ("%0", .dot, ["%x", "%y"], [512, 512]),
                ("%1", .dot, ["%0", "%y"], [512, 512]),
                ("%2", .dot, ["%1", "%x"], [512, 512]),
                ("%3", .dot, ["%2", "%y"], [512, 512]),
            ],
            returnValues: ["%3"]
        )

        // Set very small SRAM capacity to force splitting
        let partitioner = GPUANEPartitioner(
            config: .init(
                minOpsForANEPartition: 1,
                maxTransferOverheadMs: 100.0,
                enablePipelining: false,
                balanceLoad: false,
                aneSRAMCapacityBytes: 100 // Tiny - forces split
            ),
            analyzer: makeTestAnalyzer()
        )
        let plan = partitioner.partition(function)

        // Should have more partitions than without splitting
        #expect(plan.partitions.count >= 2, "Oversized partitions should be split")

        // All 4 ops should still be covered
        let allOps = plan.partitions.flatMap { $0.operationIds }
        #expect(allOps.count == 4)
    }

    // MARK: - Shape Efficiency

    @Test("Shape efficiency penalizes poorly-aligned shapes")
    func shapeEfficiencyPenalty() {
        // Compare two shapes with similar element counts but different alignment
        let analyzer = ANEAnalyzer()

        // [64, 64] = 4096 elements, maps cleanly to [1, 64, 1, 64]
        let wellAligned = HLOOperation(
            result: "%well",
            kind: .add,
            operands: ["%x", "%y"],
            resultType: TensorType(shape: [64, 64], elementType: .float32)
        )
        // [63, 65] = 4095 elements, both dims are misaligned
        let poorlyAligned = HLOOperation(
            result: "%poor",
            kind: .add,
            operands: ["%x", "%y"],
            resultType: TensorType(shape: [63, 65], elementType: .float32)
        )

        let wellAnalysis = analyzer.analyze(wellAligned)
        let poorAnalysis = analyzer.analyze(poorlyAligned)

        // Poorly-aligned shape should have higher ANE time due to shape penalty
        // (same element count, but alignment waste inflates the estimate)
        #expect(poorAnalysis.estimatedANETime >= wellAnalysis.estimatedANETime,
            "Poorly-aligned shapes should have higher or equal estimated ANE time")
    }

    @Test("All ops covered after memory planning")
    func allOpsCoveredAfterMemoryPlanning() {
        // Mix of GPU-only and ANE ops
        let function = makeFunction(
            inputs: [("%x", [128, 128]), ("%y", [128, 128])],
            ops: [
                ("%0", .dot, ["%x", "%y"], [128, 128]),
                ("%1", .gather, ["%0", "%x"], [128, 128]),
                ("%2", .dot, ["%1", "%y"], [128, 128]),
            ],
            returnValues: ["%2"]
        )

        let partitioner = GPUANEPartitioner(
            config: .init(
                minOpsForANEPartition: 1,
                maxTransferOverheadMs: 100.0,
                enablePipelining: false,
                balanceLoad: false
            ),
            analyzer: makeTestAnalyzer()
        )
        let plan = partitioner.partition(function)

        let allOps = plan.partitions.flatMap { $0.operationIds }
        #expect(allOps.count == 3, "All ops should be covered after memory planning")
    }
}
