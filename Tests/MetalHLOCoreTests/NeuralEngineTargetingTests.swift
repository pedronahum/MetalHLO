// NeuralEngineTargetingTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2J: Neural Engine Targeting

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("Neural Engine Targeting Tests")
struct NeuralEngineTargetingTests {

    // MARK: - ANE Compatibility Tests

    @Test("ANECompatibility enum values")
    func aneCompatibilityValues() {
        #expect(ANECompatibility.optimal.rawValue == "optimal")
        #expect(ANECompatibility.compatible.rawValue == "compatible")
        #expect(ANECompatibility.incompatible.rawValue == "incompatible")
        #expect(ANECompatibility.unknown.rawValue == "unknown")
    }

    @Test("ANEOperationAnalysis preferred device")
    func aneOperationAnalysisPreferredDevice() {
        let analysis1 = ANEOperationAnalysis(
            operationId: "op1",
            operationType: .dot,
            compatibility: .optimal,
            estimatedANETime: 1.0,
            estimatedGPUTime: 2.0,
            reason: "Test"
        )
        #expect(analysis1.preferredDevice == .ane)

        let analysis2 = ANEOperationAnalysis(
            operationId: "op2",
            operationType: .add,
            compatibility: .compatible,
            estimatedANETime: 2.0,
            estimatedGPUTime: 1.0,
            reason: "Test"
        )
        #expect(analysis2.preferredDevice == .gpu)

        let analysis3 = ANEOperationAnalysis(
            operationId: "op3",
            operationType: .customCall,
            compatibility: .incompatible,
            estimatedANETime: .infinity,
            estimatedGPUTime: 1.0,
            reason: "Test"
        )
        #expect(analysis3.preferredDevice == .gpu)
    }

    @Test("ANEOperationAnalysis speedup ratio")
    func aneOperationAnalysisSpeedupRatio() {
        let analysis = ANEOperationAnalysis(
            operationId: "op1",
            operationType: .dot,
            compatibility: .optimal,
            estimatedANETime: 1.0,
            estimatedGPUTime: 2.0,
            reason: "Test"
        )
        #expect(analysis.speedupRatio == 2.0)
    }

    // MARK: - ANE Constraint Tests

    @Test("ANEConstraint enum cases")
    func aneConstraintCases() {
        let allCases = ANEConstraint.allCases
        #expect(allCases.contains(.tensorSizeTooSmall))
        #expect(allCases.contains(.tensorSizeTooLarge))
        #expect(allCases.contains(.unsupportedDataType))
        #expect(allCases.contains(.unsupportedOperation))
        #expect(allCases.contains(.nonContiguousMemory))
        #expect(allCases.contains(.dynamicShape))
    }

    // MARK: - ANE Analyzer Tests

    @Test("ANEAnalyzer default config")
    func aneAnalyzerDefaultConfig() {
        let config = ANEAnalyzer.Config.default
        #expect(config.minTensorSizeForANE == 4096)
        #expect(config.maxTensorSizeForANE == 16_777_216)
        #expect(config.preferANEForConvolutions)
        #expect(config.preferANEForMatMul)
    }

    @Test("ANEAnalyzer analyzes dot operation as optimal")
    func aneAnalyzerDotOperation() {
        let analyzer = ANEAnalyzer()

        let dotOp = HLOOperation(
            result: "dot_result",
            kind: .dot,
            operands: ["a", "b"],
            resultType: TensorType(shape: [512, 512], elementType: .float32),
            attributes: HLOAttributes()
        )

        let analysis = analyzer.analyze(dotOp)

        #expect(analysis.operationId == "dot_result")
        #expect(analysis.operationType == .dot)
        #expect(analysis.compatibility == .optimal)
    }

    @Test("ANEAnalyzer marks custom_call as incompatible")
    func aneAnalyzerCustomCallIncompatible() {
        let analyzer = ANEAnalyzer()

        let customOp = HLOOperation(
            result: "custom_result",
            kind: .customCall,
            operands: ["input"],
            resultType: TensorType(shape: [128, 128], elementType: .float32),
            attributes: HLOAttributes()
        )

        let analysis = analyzer.analyze(customOp)

        #expect(analysis.compatibility == .incompatible)
        #expect(analysis.constraints.contains(.unsupportedOperation))
    }

    @Test("ANEAnalyzer detects tensor size too small")
    func aneAnalyzerTensorTooSmall() {
        let analyzer = ANEAnalyzer(config: ANEAnalyzer.Config(
            minTensorSizeForANE: 1000,
            maxTensorSizeForANE: 1000000,
            aneOverheadNs: 50000,
            preferANEForConvolutions: true,
            preferANEForMatMul: true
        ))

        let smallOp = HLOOperation(
            result: "small_result",
            kind: .add,
            operands: ["a", "b"],
            resultType: TensorType(shape: [10, 10], elementType: .float32), // 100 elements
            attributes: HLOAttributes()
        )

        let analysis = analyzer.analyze(smallOp)
        #expect(analysis.constraints.contains(.tensorSizeTooSmall))
    }

    @Test("ANEAnalyzer detects tensor size too large")
    func aneAnalyzerTensorTooLarge() {
        let analyzer = ANEAnalyzer(config: ANEAnalyzer.Config(
            minTensorSizeForANE: 100,
            maxTensorSizeForANE: 1000, // Very small limit for testing
            aneOverheadNs: 50000,
            preferANEForConvolutions: true,
            preferANEForMatMul: true
        ))

        let largeOp = HLOOperation(
            result: "large_result",
            kind: .add,
            operands: ["a", "b"],
            resultType: TensorType(shape: [100, 100], elementType: .float32), // 10000 elements
            attributes: HLOAttributes()
        )

        let analysis = analyzer.analyze(largeOp)
        #expect(analysis.compatibility == .incompatible)
        #expect(analysis.constraints.contains(.tensorSizeTooLarge))
    }

    @Test("ANEAnalyzer analyzes entire function")
    func aneAnalyzerFunction() {
        let analyzer = ANEAnalyzer()

        let function = createTestFunction()
        let analysis = analyzer.analyzeFunction(function)

        #expect(analysis.functionName == "test_func")
        #expect(analysis.operationAnalyses.count == 3)
        #expect(analysis.optimalOpsCount >= 0)
    }

    @Test("FunctionANEAnalysis statistics")
    func functionANEAnalysisStats() {
        let analyses = [
            ANEOperationAnalysis(
                operationId: "op1",
                operationType: .dot,
                compatibility: .optimal,
                estimatedANETime: 1.0,
                estimatedGPUTime: 2.0,
                reason: "Test"
            ),
            ANEOperationAnalysis(
                operationId: "op2",
                operationType: .add,
                compatibility: .compatible,
                estimatedANETime: 0.5,
                estimatedGPUTime: 0.3,
                reason: "Test"
            ),
            ANEOperationAnalysis(
                operationId: "op3",
                operationType: .customCall,
                compatibility: .incompatible,
                estimatedANETime: .infinity,
                estimatedGPUTime: 1.0,
                reason: "Test"
            )
        ]

        let funcAnalysis = FunctionANEAnalysis(
            functionName: "test",
            operationAnalyses: analyses
        )

        #expect(funcAnalysis.optimalOpsCount == 1)
        #expect(funcAnalysis.compatibleOpsCount == 1)
        #expect(funcAnalysis.incompatibleOpsCount == 1)
        #expect(funcAnalysis.aneRecommendedOps.count == 1) // Only op1 prefers ANE
    }

    // MARK: - GPU-ANE Partitioner Tests

    @Test("GPUANEPartitioner default config")
    func partitionerDefaultConfig() {
        let config = GPUANEPartitioner.Config.default
        #expect(config.minOpsForANEPartition == 3)
        #expect(config.maxTransferOverheadMs == 1.0)
        #expect(config.enablePipelining)
        #expect(config.balanceLoad)
    }

    @Test("GPUANEPartitioner creates partitions")
    func partitionerCreatesPartitions() {
        let partitioner = GPUANEPartitioner()
        let function = createTestFunction()

        let plan = partitioner.partition(function)

        #expect(!plan.partitions.isEmpty)
        #expect(!plan.executionOrder.isEmpty)
        #expect(plan.estimatedTotalTimeMs >= 0)
    }

    @Test("DevicePartition properties")
    func devicePartitionProperties() {
        let partition = GPUANEPartitioner.DevicePartition(
            device: .ane,
            operationIds: ["op1", "op2", "op3"],
            estimatedTimeMs: 2.5,
            inputTensors: Set(["input1"]),
            outputTensors: Set(["output1"])
        )

        #expect(partition.device == .ane)
        #expect(partition.operationIds.count == 3)
        #expect(partition.estimatedTimeMs == 2.5)
        #expect(partition.inputTensors.contains("input1"))
        #expect(partition.outputTensors.contains("output1"))
    }

    @Test("PartitionPlan GPU and ANE partitions")
    func partitionPlanDevicePartitions() {
        let gpuPartition = GPUANEPartitioner.DevicePartition(
            device: .gpu,
            operationIds: ["op1"],
            estimatedTimeMs: 1.0,
            inputTensors: Set(),
            outputTensors: Set()
        )

        let anePartition = GPUANEPartitioner.DevicePartition(
            device: .ane,
            operationIds: ["op2", "op3"],
            estimatedTimeMs: 1.5,
            inputTensors: Set(),
            outputTensors: Set()
        )

        let plan = GPUANEPartitioner.PartitionPlan(
            partitions: [gpuPartition, anePartition],
            executionOrder: [0, 1],
            estimatedTotalTimeMs: 2.5,
            estimatedTransferTimeMs: 0.1,
            canPipeline: true
        )

        #expect(plan.gpuPartitions.count == 1)
        #expect(plan.anePartitions.count == 1)
        #expect(plan.canPipeline)
    }

    @Test("GPUANEPartitioner handles small partitions")
    func partitionerSmallPartitions() {
        let partitioner = GPUANEPartitioner(config: GPUANEPartitioner.Config(
            minOpsForANEPartition: 5, // High threshold
            maxTransferOverheadMs: 1.0,
            enablePipelining: true,
            balanceLoad: true
        ))

        // Create function with operations that would be assigned to ANE
        // but are too few to justify an ANE partition
        let function = createSmallFunction()

        let plan = partitioner.partition(function)

        // Small ANE partitions should be merged to GPU
        // Result should have all ops on GPU
        for partition in plan.partitions {
            if partition.device == .ane {
                #expect(partition.operationIds.count >= 5)
            }
        }
    }

    // MARK: - Heterogeneous Executor Tests

    @Test("HeterogeneousExecutor default config")
    func executorDefaultConfig() {
        let config = HeterogeneousExecutor.Config.default
        #expect(config.enableANE)
        #expect(config.enableGPU)
        #expect(config.enableCPUFallback)
        #expect(config.maxConcurrentPartitions == 2)
        #expect(config.syncStrategy == .event)
    }

    @Test("ExecutionStats initial values")
    func executionStatsInitial() {
        let stats = HeterogeneousExecutor.ExecutionStats()
        #expect(stats.gpuTimeMs == 0)
        #expect(stats.aneTimeMs == 0)
        #expect(stats.transferTimeMs == 0)
        #expect(stats.totalTimeMs == 0)
        #expect(stats.operationsExecuted == 0)
        #expect(stats.partitionsExecuted == 0)
    }

    @Test("HeterogeneousExecutor stats tracking")
    func executorStatsTracking() {
        let executor = HeterogeneousExecutor()

        let stats = executor.getStats()
        #expect(stats.totalTimeMs == 0)

        executor.resetStats()
        let resetStats = executor.getStats()
        #expect(resetStats.operationsExecuted == 0)
    }

    @Test("PartitionResult creation")
    func partitionResultCreation() {
        let result = PartitionResult(
            device: .ane,
            operationIds: ["op1", "op2"],
            executionTimeMs: 1.5,
            success: true
        )

        #expect(result.device == .ane)
        #expect(result.operationIds.count == 2)
        #expect(result.executionTimeMs == 1.5)
        #expect(result.success)
    }

    @Test("HeterogeneousExecutorError cases")
    func heterogeneousExecutorErrorCases() {
        let error1 = HeterogeneousExecutorError.deviceNotAvailable("GPU unavailable")
        let error2 = HeterogeneousExecutorError.deviceInitializationFailed("Failed to init")
        let error3 = HeterogeneousExecutorError.executionFailed("Execution error")
        let error4 = HeterogeneousExecutorError.partitioningFailed("Partitioning error")

        #expect("\(error1)".contains("deviceNotAvailable"))
        #expect("\(error2)".contains("deviceInitializationFailed"))
        #expect("\(error3)".contains("executionFailed"))
        #expect("\(error4)".contains("partitioningFailed"))
    }

    // MARK: - ANE Capability Detector Tests

    @Test("ANECapabilityDetector detect")
    func aneCapabilityDetectorDetect() {
        let detector = ANECapabilityDetector()
        let capabilities = detector.detect()

        // On macOS/iOS, ANE should be available
        #if os(macOS) || os(iOS)
        #expect(capabilities.isAvailable)
        #expect(capabilities.maxMemoryBytes > 0)
        #expect(capabilities.supportsFloat16)
        #expect(capabilities.estimatedTOPS > 0)
        #else
        #expect(!capabilities.isAvailable)
        #endif
    }

    @Test("ANECapabilities unavailable constant")
    func aneCapabilitiesUnavailable() {
        let caps = ANECapabilityDetector.ANECapabilities.unavailable
        #expect(!caps.isAvailable)
        #expect(caps.maxMemoryBytes == 0)
        #expect(!caps.supportsFloat16)
        #expect(!caps.supportsInt8)
        #expect(caps.estimatedTOPS == 0)
        #expect(caps.generationName == "None")
    }

    @Test("ANECapabilityDetector canRunOnANE")
    func aneCapabilityDetectorCanRun() {
        let detector = ANECapabilityDetector()
        let capabilities = ANECapabilityDetector.ANECapabilities(
            isAvailable: true,
            maxMemoryBytes: 1_000_000,
            supportsFloat16: true,
            supportsInt8: true,
            estimatedTOPS: 15.0,
            generationName: "Test"
        )

        let smallOp = HLOOperation(
            result: "result",
            kind: .add,
            operands: ["a", "b"],
            resultType: TensorType(shape: [100, 100], elementType: .float16),
            attributes: HLOAttributes()
        )

        #expect(detector.canRunOnANE(smallOp, capabilities: capabilities))

        // Large operation exceeds memory
        let largeOp = HLOOperation(
            result: "result",
            kind: .add,
            operands: ["a", "b"],
            resultType: TensorType(shape: [10000, 10000], elementType: .float32),
            attributes: HLOAttributes()
        )

        #expect(!detector.canRunOnANE(largeOp, capabilities: capabilities))
    }

    @Test("ANECapabilityDetector unavailable returns false")
    func aneCapabilityDetectorUnavailable() {
        let detector = ANECapabilityDetector()

        let op = HLOOperation(
            result: "result",
            kind: .add,
            operands: ["a", "b"],
            resultType: TensorType(shape: [100, 100], elementType: .float32),
            attributes: HLOAttributes()
        )

        #expect(!detector.canRunOnANE(op, capabilities: .unavailable))
    }

    // MARK: - ExecutionDevice Tests

    @Test("ExecutionDevice values")
    func executionDeviceValues() {
        #expect(ExecutionDevice.gpu.rawValue == "gpu")
        #expect(ExecutionDevice.ane.rawValue == "ane")
        #expect(ExecutionDevice.cpu.rawValue == "cpu")
    }

    // MARK: - Integration Tests

    @Test("End-to-end partition and analyze")
    func endToEndPartitionAnalyze() {
        let analyzer = ANEAnalyzer()
        let partitioner = GPUANEPartitioner()

        let function = createLargeFunction()

        // Analyze
        let analysis = analyzer.analyzeFunction(function)
        #expect(analysis.operationAnalyses.count == function.operations.count)

        // Partition
        let plan = partitioner.partition(function)
        #expect(!plan.partitions.isEmpty)

        // Verify all operations are accounted for
        let totalOps = plan.partitions.reduce(0) { $0 + $1.operationIds.count }
        #expect(totalOps == function.operations.count)
    }

    @Test("Partition plan execution order is valid")
    func partitionPlanExecutionOrder() {
        let partitioner = GPUANEPartitioner()
        let function = createLargeFunction()

        let plan = partitioner.partition(function)

        // Execution order should contain valid indices
        for index in plan.executionOrder {
            #expect(index >= 0 && index < plan.partitions.count)
        }

        // All partitions should be in execution order
        #expect(Set(plan.executionOrder).count == plan.partitions.count)
    }

    // MARK: - Helper Functions

    private func createTestFunction() -> HLOFunction {
        let inputs = [
            HLOArgument(name: "a", type: TensorType(shape: [128, 256], elementType: .float32)),
            HLOArgument(name: "b", type: TensorType(shape: [256, 128], elementType: .float32))
        ]

        let operations = [
            HLOOperation(
                result: "dot_result",
                kind: .dot,
                operands: ["a", "b"],
                resultType: TensorType(shape: [128, 128], elementType: .float32),
                attributes: HLOAttributes()
            ),
            HLOOperation(
                result: "exp_result",
                kind: .exponential,
                operands: ["dot_result"],
                resultType: TensorType(shape: [128, 128], elementType: .float32),
                attributes: HLOAttributes()
            ),
            HLOOperation(
                result: "add_result",
                kind: .add,
                operands: ["exp_result", "dot_result"],
                resultType: TensorType(shape: [128, 128], elementType: .float32),
                attributes: HLOAttributes()
            )
        ]

        return HLOFunction(
            name: "test_func",
            inputs: inputs,
            outputTypes: [TensorType(shape: [128, 128], elementType: .float32)],
            operations: operations,
            returnValues: ["add_result"]
        )
    }

    private func createSmallFunction() -> HLOFunction {
        let inputs = [
            HLOArgument(name: "x", type: TensorType(shape: [64, 64], elementType: .float32))
        ]

        let operations = [
            HLOOperation(
                result: "exp_result",
                kind: .exponential,
                operands: ["x"],
                resultType: TensorType(shape: [64, 64], elementType: .float32),
                attributes: HLOAttributes()
            ),
            HLOOperation(
                result: "neg_result",
                kind: .negate,
                operands: ["exp_result"],
                resultType: TensorType(shape: [64, 64], elementType: .float32),
                attributes: HLOAttributes()
            )
        ]

        return HLOFunction(
            name: "small_func",
            inputs: inputs,
            outputTypes: [TensorType(shape: [64, 64], elementType: .float32)],
            operations: operations,
            returnValues: ["neg_result"]
        )
    }

    private func createLargeFunction() -> HLOFunction {
        let inputs = [
            HLOArgument(name: "input", type: TensorType(shape: [512, 512], elementType: .float32))
        ]

        var operations: [HLOOperation] = []
        var prevResult = "input"

        for i in 0..<10 {
            let result = "layer_\(i)"
            let kind: HLOOpKind = [.exponential, .tanh, .negate, .add, .multiply][i % 5]

            let operands: [String]
            if kind == .add || kind == .multiply {
                operands = [prevResult, prevResult]
            } else {
                operands = [prevResult]
            }

            operations.append(HLOOperation(
                result: result,
                kind: kind,
                operands: operands,
                resultType: TensorType(shape: [512, 512], elementType: .float32),
                attributes: HLOAttributes()
            ))

            prevResult = result
        }

        return HLOFunction(
            name: "large_func",
            inputs: inputs,
            outputTypes: [TensorType(shape: [512, 512], elementType: .float32)],
            operations: operations,
            returnValues: [prevResult]
        )
    }
}
