// HeterogeneousIntegrationTests.swift
// MetalHLOCoreTests
//
// End-to-end integration tests for heterogeneous GPU+ANE execution.
// Validates the full pipeline: partition → extract → execute → transfer → collect.

import Testing
import Foundation
import Metal
@testable import MetalHLOCore

@Suite("Heterogeneous Integration", .serialized)
struct HeterogeneousIntegrationTests {

    // MARK: - Test Helpers

    /// Creates a simple HLO operation for testing.
    private func makeOp(
        result: String,
        kind: HLOOpKind,
        operands: [String],
        shape: [Int] = [64, 64],
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

    /// Creates BufferStorage filled with the given float data.
    private func makeStorage(_ data: [Float], shape: [Int], device: MTLDevice) -> BufferStorage {
        BufferStorage(floatData: data, shape: shape, device: device)
    }

    // MARK: - Partitioning Tests

    /// Creates an ANEAnalyzer with a low tensor size threshold for testing.
    private func makeTestAnalyzer() -> ANEAnalyzer {
        ANEAnalyzer(config: .init(
            minTensorSizeForANE: 100,
            maxTensorSizeForANE: 16_777_216,
            aneOverheadNs: 50_000,
            preferANEForConvolutions: true,
            preferANEForMatMul: true
        ))
    }

    @Test("Simple heterogeneous: ANE-optimal ops + GPU-only gather forces multi-partition")
    func simpleHeterogeneousExecution() throws {
        // dot (ANE optimal) → gather (GPU only) → dot (ANE optimal)
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

        // Gather must be on GPU
        let gatherPartition = plan.partitions.first { $0.operationIds.contains("%1") }
        #expect(gatherPartition != nil)
        #expect(gatherPartition?.device == .gpu)

        // Should have at least 2 partitions (ANE dot + GPU gather)
        #expect(plan.partitions.count >= 2)

        // All 3 ops should be covered
        let allOps = plan.partitions.flatMap { $0.operationIds }
        #expect(allOps.count == 3)
    }

    @Test("Multi-op ANE chain with GPU break produces correct partition structure")
    func multiOpANEChainWithGPUBreak() {
        // Mix of ANE-optimal (dot) and compatible ops → gather (GPU-only) → more ANE ops
        let function = makeFunction(
            inputs: [("%x", [128, 128]), ("%y", [128, 128])],
            ops: [
                ("%0", .dot, ["%x", "%y"], [128, 128]),
                ("%1", .dot, ["%0", "%y"], [128, 128]),
                ("%2", .dot, ["%1", "%x"], [128, 128]),
                ("%3", .gather, ["%2", "%x"], [128, 128]),
                ("%4", .dot, ["%3", "%y"], [128, 128]),
                ("%5", .dot, ["%4", "%x"], [128, 128]),
            ],
            returnValues: ["%5"]
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

        // Should have at least 2 partitions (gather forces GPU split)
        #expect(plan.partitions.count >= 2)

        // Gather must be in a GPU partition
        let gatherPartition = plan.partitions.first { $0.operationIds.contains("%3") }
        #expect(gatherPartition?.device == .gpu)

        // All 6 ops should be covered
        let allOps = plan.partitions.flatMap { $0.operationIds }
        #expect(allOps.count == 6)
    }

    @Test("All-GPU fallback when no ANE-compatible ops")
    func allGPUFallbackWhenNoANEOps() {
        // All gather ops → everything runs on GPU
        let function = makeFunction(
            inputs: [("%x", [64, 64]), ("%y", [64, 64])],
            ops: [
                ("%0", .gather, ["%x", "%y"], [64, 64]),
                ("%1", .gather, ["%0", "%y"], [64, 64]),
            ],
            returnValues: ["%1"]
        )

        let partitioner = GPUANEPartitioner()
        let plan = partitioner.partition(function)

        // All partitions should be GPU
        for partition in plan.partitions {
            #expect(partition.device == .gpu)
        }
    }

    // MARK: - Executor Integration Tests

    @Test("HeterogeneousExecutor executes all-ANE-compatible function")
    func executorRunsANECompatibleFunction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return // Skip on systems without Metal
        }

        let metalExec = try MetalExecutor()

        // Simple: add → multiply (all ANE-compatible)
        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [
                ("%0", .add, ["%x", "%y"], [4]),
                ("%1", .multiply, ["%0", "%y"], [4]),
            ],
            returnValues: ["%1"]
        )

        let xData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let yData: [Float] = [0.5, 0.5, 0.5, 0.5]
        let xStorage = makeStorage(xData, shape: [4], device: device)
        let yStorage = makeStorage(yData, shape: [4], device: device)

        let executor = HeterogeneousExecutor(metalExecutor: metalExec)
        let result = try executor.execute(function, inputs: ["%x": xStorage, "%y": yStorage])

        #expect(result.success)
        #expect(result.partitionResults.count >= 1)
        #expect(result.outputStorages["%1"] != nil)

        // Verify result: (x + y) * y = [0.75, 1.25, 1.75, 2.25]
        if let output = result.outputStorages["%1"] {
            let outputData = output.data
            let floats: [Float] = outputData.withUnsafeBytes { bytes in
                Array(bytes.bindMemory(to: Float.self))
            }
            let expected: [Float] = [0.75, 1.25, 1.75, 2.25]
            for (i, (got, exp)) in zip(floats, expected).enumerated() {
                let diff = abs(got - exp)
                #expect(diff < 1e-3, "Mismatch at \(i): got \(got), expected \(exp)")
            }
        }
    }

    @Test("Numerical correctness: heterogeneous vs GPU-only")
    func numericalCorrectnessVsGPUOnly() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let metalExec = try MetalExecutor()

        // Function: (x + y) * y
        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [
                ("%0", .add, ["%x", "%y"], [4]),
                ("%1", .multiply, ["%0", "%y"], [4]),
            ],
            returnValues: ["%1"]
        )

        let xData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let yData: [Float] = [0.1, 0.2, 0.3, 0.4]
        let xStorage = makeStorage(xData, shape: [4], device: device)
        let yStorage = makeStorage(yData, shape: [4], device: device)

        // GPU-only reference via MetalExecutor (MPSGraph)
        let module = HLOModule(name: "ref", function: function)
        let compiled = try metalExec.compile(module: module)
        let gpuOutputs = try metalExec.execute(compiled: compiled, inputs: [xStorage, yStorage])
        let gpuData = gpuOutputs[0].data
        let gpuFloats: [Float] = gpuData.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }

        // Heterogeneous execution
        let hetExecutor = HeterogeneousExecutor(metalExecutor: metalExec)
        let hetResult = try hetExecutor.execute(function, inputs: ["%x": xStorage, "%y": yStorage])
        guard let hetOutput = hetResult.outputStorages["%1"] else {
            #expect(Bool(false), "Missing heterogeneous output")
            return
        }
        let hetData = hetOutput.data
        let hetFloats: [Float] = hetData.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }

        // Compare element-by-element
        #expect(gpuFloats.count == hetFloats.count)
        for i in 0..<gpuFloats.count {
            let diff = abs(gpuFloats[i] - hetFloats[i])
            let maxVal = max(abs(gpuFloats[i]), 1e-6)
            #expect(diff / maxVal < 1e-3, "Mismatch at \(i): GPU=\(gpuFloats[i]) het=\(hetFloats[i])")
        }
    }

    @Test("Execution stats are populated")
    func executionStatsPopulated() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return
        }

        let metalExec = try MetalExecutor()

        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [
                ("%0", .add, ["%x", "%y"], [4]),
            ],
            returnValues: ["%0"]
        )

        let device = metalExec.device
        let xStorage = makeStorage([1.0, 2.0, 3.0, 4.0], shape: [4], device: device)
        let yStorage = makeStorage([5.0, 6.0, 7.0, 8.0], shape: [4], device: device)

        let executor = HeterogeneousExecutor(metalExecutor: metalExec)
        let result = try executor.execute(function, inputs: ["%x": xStorage, "%y": yStorage])

        #expect(result.success)
        #expect(result.stats.partitionsExecuted > 0)
        #expect(result.stats.operationsExecuted > 0)
        #expect(result.stats.totalTimeMs > 0)

        // Also check cumulative stats
        let cumStats = executor.getStats()
        #expect(cumStats.partitionsExecuted > 0)
        #expect(cumStats.totalTimeMs > 0)
    }

    // MARK: - Pipeline Mode Tests

    @Test("Pipelined execution produces same results as sequential")
    func pipelinedMatchesSequential() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let metalExec = try MetalExecutor()

        let function = makeFunction(
            inputs: [("%x", [4]), ("%y", [4])],
            ops: [
                ("%0", .add, ["%x", "%y"], [4]),
                ("%1", .multiply, ["%0", "%y"], [4]),
            ],
            returnValues: ["%1"]
        )

        let xData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let yData: [Float] = [0.5, 0.5, 0.5, 0.5]
        let xStorage = makeStorage(xData, shape: [4], device: device)
        let yStorage = makeStorage(yData, shape: [4], device: device)

        // Sequential
        let seqExecutor = HeterogeneousExecutor(
            metalExecutor: metalExec,
            config: .init(enableANE: true, enableGPU: true, enableCPUFallback: true, enablePipelining: false)
        )
        let seqResult = try seqExecutor.execute(function, inputs: ["%x": xStorage, "%y": yStorage])

        // Pipelined
        let pipExecutor = HeterogeneousExecutor(
            metalExecutor: metalExec,
            config: .init(enableANE: true, enableGPU: true, enableCPUFallback: true, enablePipelining: true)
        )
        let pipResult = try pipExecutor.execute(function, inputs: ["%x": xStorage, "%y": yStorage])

        // Both should succeed
        #expect(seqResult.success)
        #expect(pipResult.success)

        // Compare outputs
        guard let seqOutput = seqResult.outputStorages["%1"],
              let pipOutput = pipResult.outputStorages["%1"] else {
            #expect(Bool(false), "Missing output from one of the executors")
            return
        }

        let seqFloats: [Float] = seqOutput.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let pipFloats: [Float] = pipOutput.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

        #expect(seqFloats.count == pipFloats.count)
        for i in 0..<seqFloats.count {
            let diff = abs(seqFloats[i] - pipFloats[i])
            let maxVal = max(abs(seqFloats[i]), 1e-6)
            #expect(diff / maxVal < 1e-3, "Mismatch at \(i): seq=\(seqFloats[i]) pip=\(pipFloats[i])")
        }
    }
}
