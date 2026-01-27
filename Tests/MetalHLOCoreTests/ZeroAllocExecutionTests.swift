// ZeroAllocExecutionTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2E: Zero-Allocation Execution

import Testing
import Metal
@testable import MetalHLOCore

@Suite("Zero-Allocation Execution Tests")
struct ZeroAllocExecutionTests {

    // MARK: - ScheduledTensorLifetime Tests

    @Test("ScheduledTensorLifetime tracks creation and last use")
    func tensorLifetimeTracking() {
        let lifetime = ScheduledTensorLifetime(
            createdAt: 2,
            lastUsedAt: 5,
            byteSize: 1024
        )

        #expect(lifetime.createdAt == 2)
        #expect(lifetime.lastUsedAt == 5)
        #expect(lifetime.byteSize == 1024)
        #expect(lifetime.isInput == false)
        #expect(lifetime.isOutput == false)
    }

    @Test("ScheduledTensorLifetime isLiveAt returns correct values")
    func tensorLifetimeIsLiveAt() {
        let lifetime = ScheduledTensorLifetime(
            createdAt: 2,
            lastUsedAt: 5,
            byteSize: 1024
        )

        #expect(lifetime.isLiveAt(1) == false)
        #expect(lifetime.isLiveAt(2) == true)
        #expect(lifetime.isLiveAt(3) == true)
        #expect(lifetime.isLiveAt(5) == true)
        #expect(lifetime.isLiveAt(6) == false)
    }

    @Test("ScheduledTensorLifetime tracks input tensors")
    func tensorLifetimeInput() {
        let lifetime = ScheduledTensorLifetime(
            createdAt: -1,
            lastUsedAt: 3,
            byteSize: 4096,
            isInput: true
        )

        #expect(lifetime.isInput == true)
        #expect(lifetime.createdAt == -1)
        #expect(lifetime.isLiveAt(-1) == true)
        #expect(lifetime.isLiveAt(0) == true)
    }

    @Test("ScheduledTensorLifetime tracks output tensors")
    func tensorLifetimeOutput() {
        let lifetime = ScheduledTensorLifetime(
            createdAt: 3,
            lastUsedAt: 10,
            byteSize: 2048,
            isOutput: true
        )

        #expect(lifetime.isOutput == true)
    }

    // MARK: - MemoryInterferenceGraph Tests

    @Test("MemoryInterferenceGraph tracks conflicts")
    func interferenceGraphConflicts() {
        var graph = MemoryInterferenceGraph()

        graph.addEdge("a", "b")
        graph.addEdge("b", "c")

        #expect(graph.conflicts("a", "b") == true)
        #expect(graph.conflicts("b", "a") == true)
        #expect(graph.conflicts("b", "c") == true)
        #expect(graph.conflicts("a", "c") == false)
    }

    @Test("MemoryInterferenceGraph returns neighbors")
    func interferenceGraphNeighbors() {
        var graph = MemoryInterferenceGraph()

        graph.addEdge("center", "a")
        graph.addEdge("center", "b")
        graph.addEdge("center", "c")

        let neighbors = graph.neighbors("center")

        #expect(neighbors.count == 3)
        #expect(neighbors.contains("a"))
        #expect(neighbors.contains("b"))
        #expect(neighbors.contains("c"))
    }

    @Test("MemoryInterferenceGraph handles missing nodes")
    func interferenceGraphMissing() {
        let graph = MemoryInterferenceGraph()

        #expect(graph.conflicts("x", "y") == false)
        #expect(graph.neighbors("x").isEmpty)
    }

    // MARK: - MemoryPlan Tests

    @Test("MemoryPlan stores plan information")
    func memoryPlanStoresInfo() {
        let plan = MemoryPlan(
            totalBytes: 4096,
            tensorOffsets: ["a": 0, "b": 1024, "c": 2048],
            sharingGroups: [["a", "c"]],
            executionOrder: [0, 1, 2],
            peakMemory: 3072
        )

        #expect(plan.totalBytes == 4096)
        #expect(plan.tensorOffsets.count == 3)
        #expect(plan.tensorOffsets["a"] == 0)
        #expect(plan.tensorOffsets["b"] == 1024)
        #expect(plan.sharingGroups.count == 1)
        #expect(plan.executionOrder == [0, 1, 2])
        #expect(plan.peakMemory == 3072)
    }

    @Test("MemoryPlan statistics are computed correctly")
    func memoryPlanStatistics() {
        let plan = MemoryPlan(
            totalBytes: 8192,
            tensorOffsets: ["a": 0, "b": 256, "c": 512, "d": 768],
            sharingGroups: [["a", "c"], ["b", "d"]],
            executionOrder: [0, 1, 2, 3],
            peakMemory: 4096
        )

        let stats = plan.statistics

        #expect(stats.totalBytes == 8192)
        #expect(stats.peakMemory == 4096)
        #expect(stats.numTensors == 4)
        #expect(stats.numSharingGroups == 2)
        #expect(stats.memoryReuse == 0.5)
    }

    // MARK: - StaticMemoryPlanner Tests

    @Test("StaticMemoryPlanner plans simple function")
    func plannerSimpleFunction() {
        let planner = StaticMemoryPlanner()

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "input", type: TensorType(shape: [1, 256], elementType: .float32))],
            outputTypes: [TensorType(shape: [1, 256], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "output",
                    kind: .add,
                    operands: ["input", "input"],
                    resultType: TensorType(shape: [1, 256], elementType: .float32)
                )
            ],
            returnValues: ["output"]
        )

        let plan = planner.plan(function)

        #expect(plan.totalBytes > 0)
        #expect(plan.tensorOffsets.count == 1)  // output only (inputs come from external buffers)
        #expect(plan.executionOrder.count == 1)
    }

    @Test("StaticMemoryPlanner handles multiple operations")
    func plannerMultipleOps() {
        let planner = StaticMemoryPlanner()

        let tensorType = TensorType(shape: [1, 128], elementType: .float32)

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "input", type: tensorType)],
            outputTypes: [tensorType],
            operations: [
                HLOOperation(
                    result: "t1",
                    kind: .add,
                    operands: ["input", "input"],
                    resultType: tensorType
                ),
                HLOOperation(
                    result: "t2",
                    kind: .multiply,
                    operands: ["t1", "input"],
                    resultType: tensorType
                ),
                HLOOperation(
                    result: "output",
                    kind: .add,
                    operands: ["t1", "t2"],
                    resultType: tensorType
                )
            ],
            returnValues: ["output"]
        )

        let plan = planner.plan(function)

        #expect(plan.executionOrder.count == 3)
        #expect(plan.tensorOffsets.count >= 3)
    }

    @Test("StaticMemoryPlanner respects tensor lifetimes")
    func plannerRespectsLifetimes() {
        let planner = StaticMemoryPlanner()

        let tensorType = TensorType(shape: [1, 64], elementType: .float32)  // 256 bytes

        // Create a chain where t1 dies after t2 is created
        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "input", type: tensorType)],
            outputTypes: [tensorType],
            operations: [
                HLOOperation(
                    result: "t1",
                    kind: .add,
                    operands: ["input", "input"],
                    resultType: tensorType
                ),
                HLOOperation(
                    result: "t2",
                    kind: .multiply,
                    operands: ["t1", "t1"],
                    resultType: tensorType
                ),
                HLOOperation(
                    result: "output",
                    kind: .add,
                    operands: ["t2", "input"],
                    resultType: tensorType
                )
            ],
            returnValues: ["output"]
        )

        let plan = planner.plan(function)

        // t1 and output could potentially share memory since t1 dies before output
        #expect(plan.totalBytes > 0)
    }

    @Test("StaticMemoryPlanner config affects planning")
    func plannerConfigAffectsPlanning() {
        var config = StaticMemoryPlanner.Config()
        config.bufferAlignment = 512
        config.optimizeExecutionOrder = false

        let planner = StaticMemoryPlanner(config: config)

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "input", type: TensorType(shape: [1, 100], elementType: .float32))],
            outputTypes: [TensorType(shape: [1, 100], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "output",
                    kind: .add,
                    operands: ["input", "input"],
                    resultType: TensorType(shape: [1, 100], elementType: .float32)
                )
            ],
            returnValues: ["output"]
        )

        let plan = planner.plan(function)

        // Offsets should be aligned to 512
        for (_, offset) in plan.tensorOffsets {
            #expect(offset % 512 == 0 || offset == 0)
        }
    }

    // MARK: - TensorType byteCount Tests

    @Test("TensorType computes byte count correctly")
    func tensorTypeByteCount() {
        let float32Tensor = TensorType(shape: [2, 3, 4], elementType: .float32)
        #expect(float32Tensor.byteCount == 2 * 3 * 4 * 4)  // 96 bytes

        let float16Tensor = TensorType(shape: [10, 10], elementType: .float16)
        #expect(float16Tensor.byteCount == 10 * 10 * 2)  // 200 bytes

        let int64Tensor = TensorType(shape: [5], elementType: .int64)
        #expect(int64Tensor.byteCount == 5 * 8)  // 40 bytes
    }

    @Test("ElementType byteSize returns correct values")
    func elementTypeByteSize() {
        #expect(ElementType.float32.byteSize == 4)
        #expect(ElementType.float16.byteSize == 2)
        #expect(ElementType.bfloat16.byteSize == 2)
        #expect(ElementType.int32.byteSize == 4)
        #expect(ElementType.int64.byteSize == 8)
        #expect(ElementType.int8.byteSize == 1)
        #expect(ElementType.uint8.byteSize == 1)
    }

    // MARK: - BufferPool Tests

    @Test("BufferPool creates requested number of buffers")
    func bufferPoolCreation() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return  // Skip if no Metal device
        }

        let pool = BufferPool(device: device, bufferSize: 1024, count: 3)

        #expect(pool.count == 3)
    }

    @Test("BufferPool returns buffers in round-robin")
    func bufferPoolRoundRobin() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let pool = BufferPool(device: device, bufferSize: 1024, count: 3)

        let b0 = pool.next()
        let b1 = pool.next()
        let b2 = pool.next()
        let b3 = pool.next()  // Should wrap around to first buffer

        // b3 should be the same buffer as b0
        #expect(b0 === b3)
        #expect(b0 !== b1)
        #expect(b1 !== b2)
    }

    // MARK: - TripleBufferManager Tests

    @Test("TripleBufferManager creates three buffers")
    func tripleBufferManagerCreation() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let manager = try TripleBufferManager(device: device, size: 1024)

        #expect(manager.bufferCount == 3)
    }

    @Test("TripleBufferManager advances correctly")
    func tripleBufferManagerAdvance() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let manager = try TripleBufferManager(device: device, size: 1024)

        let initial = manager.getWriteBuffer()
        manager.advanceWrite()
        let afterAdvance = manager.getWriteBuffer()

        #expect(initial !== afterAdvance)
    }

    @Test("TripleBufferManager reset works")
    func tripleBufferManagerReset() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let manager = try TripleBufferManager(device: device, size: 1024)

        let initial = manager.getWriteBuffer()
        manager.advanceWrite()
        manager.advanceWrite()
        manager.reset()
        let afterReset = manager.getWriteBuffer()

        #expect(initial === afterReset)
    }

    // MARK: - ExecutionPlanBuilder Tests

    @Test("ExecutionPlanBuilder creates valid plan")
    func executionPlanBuilderCreation() {
        let builder = ExecutionPlanBuilder()

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "input", type: TensorType(shape: [1, 64], elementType: .float32))],
            outputTypes: [TensorType(shape: [1, 64], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "output",
                    kind: .add,
                    operands: ["input", "input"],
                    resultType: TensorType(shape: [1, 64], elementType: .float32)
                )
            ],
            returnValues: ["output"]
        )

        let plan = builder.build(for: function)

        #expect(plan.function.name == "test")
        #expect(plan.operationCount == 1)
        #expect(plan.totalMemoryRequired > 0)
    }

    // MARK: - ZeroAllocExecutor Tests

    @Test("ZeroAllocExecutor initializes with memory plan")
    func executorInitialization() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let plan = MemoryPlan(
            totalBytes: 4096,
            tensorOffsets: ["t1": 0, "t2": 1024],
            sharingGroups: [],
            executionOrder: [0],
            peakMemory: 2048
        )

        let executor = try ZeroAllocExecutor(device: device, memoryPlan: plan)

        #expect(executor.memoryStatistics.totalAllocatedBytes > 0)
    }

    @Test("ZeroAllocExecutor memory statistics are correct")
    func executorMemoryStatistics() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let plan = MemoryPlan(
            totalBytes: 4096,
            tensorOffsets: ["t1": 0, "t2": 1024, "t3": 2048],
            sharingGroups: [["t1", "t3"]],
            executionOrder: [0, 1],
            peakMemory: 2048
        )

        var config = ZeroAllocExecutor.Config()
        config.inflightBufferCount = 3
        config.useTripleBuffering = true

        let executor = try ZeroAllocExecutor(device: device, memoryPlan: plan, config: config)
        let stats = executor.memoryStatistics

        #expect(stats.totalAllocatedBytes == 4096 * 3)  // Triple buffered
        #expect(stats.peakMemoryBytes == 2048)
        #expect(stats.numTensors == 3)
        #expect(stats.inflightBufferCount == 3)
    }

    // MARK: - Integration Tests

    @Test("Full zero-allocation planning workflow")
    func fullPlanningWorkflow() {
        let planner = StaticMemoryPlanner()

        // Create a realistic transformer-like function
        let hiddenSize = 768
        let seqLen = 512
        let tensorType = TensorType(shape: [1, seqLen, hiddenSize], elementType: .float32)

        let function = HLOFunction(
            name: "transformer_layer",
            inputs: [
                HLOArgument(name: "input", type: tensorType),
                HLOArgument(name: "weight", type: TensorType(shape: [hiddenSize, hiddenSize], elementType: .float32))
            ],
            outputTypes: [tensorType],
            operations: [
                // Pre-norm
                HLOOperation(
                    result: "normed",
                    kind: .customCall,
                    operands: ["input"],
                    resultType: tensorType
                ),
                // Attention projection
                HLOOperation(
                    result: "projected",
                    kind: .dot,
                    operands: ["normed", "weight"],
                    resultType: tensorType
                ),
                // Residual
                HLOOperation(
                    result: "output",
                    kind: .add,
                    operands: ["input", "projected"],
                    resultType: tensorType
                )
            ],
            returnValues: ["output"]
        )

        let plan = planner.plan(function)

        // Verify plan is valid
        #expect(plan.totalBytes > 0)
        #expect(plan.executionOrder.count == 3)
        #expect(plan.tensorOffsets.count >= 3)  // normed, projected, output (inputs come from external buffers)

        // Peak memory should be less than sum of all tensors (memory reuse)
        let inputSize = 1 * seqLen * hiddenSize * 4
        let weightSize = hiddenSize * hiddenSize * 4
        let totalNaive = inputSize * 4 + weightSize  // All intermediates same size as input
        #expect(plan.peakMemory <= totalNaive)
    }

    @Test("Memory plan enables tensor reuse")
    func memoryPlanEnablesReuse() {
        let planner = StaticMemoryPlanner()

        let tensorType = TensorType(shape: [1, 256], elementType: .float32)  // 1024 bytes

        // Create a chain where intermediate tensors can be reused
        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "input", type: tensorType)],
            outputTypes: [tensorType],
            operations: [
                HLOOperation(result: "t1", kind: .add, operands: ["input", "input"], resultType: tensorType),
                HLOOperation(result: "t2", kind: .multiply, operands: ["t1", "t1"], resultType: tensorType),
                HLOOperation(result: "t3", kind: .add, operands: ["t2", "t2"], resultType: tensorType),
                HLOOperation(result: "output", kind: .multiply, operands: ["t3", "input"], resultType: tensorType)
            ],
            returnValues: ["output"]
        )

        let plan = planner.plan(function)

        // Without reuse: input + t1 + t2 + t3 + output = 5 * 1024 = 5120 bytes
        // With reuse: should be less since t1 can be freed after t2, etc.
        let naiveBytes = 5 * 1024
        #expect(plan.peakMemory < naiveBytes)
    }
}
