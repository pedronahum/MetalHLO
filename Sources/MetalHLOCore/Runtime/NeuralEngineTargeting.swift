// NeuralEngineTargeting.swift
// Neural Engine Targeting for MetalHLO
// Enables heterogeneous execution across GPU and Apple Neural Engine (ANE)

import Foundation
import Metal
import ANERuntime

// MARK: - ANE Compatibility Analysis

/// Represents the suitability of an operation for Neural Engine execution
public enum ANECompatibility: String, Sendable {
    case optimal       // Best performance on ANE (e.g., convolutions, matmuls with specific sizes)
    case compatible    // Can run on ANE but GPU might be better
    case incompatible  // Cannot run on ANE
    case unknown       // Not analyzed yet
}

/// Detailed analysis of an operation's ANE suitability
public struct ANEOperationAnalysis: Sendable {
    public let operationId: String
    public let operationType: HLOOpKind
    public let compatibility: ANECompatibility
    public let estimatedANETime: Double  // In milliseconds
    public let estimatedGPUTime: Double  // In milliseconds
    public let reason: String
    public let constraints: [ANEConstraint]

    public var preferredDevice: ExecutionDevice {
        if compatibility == .incompatible {
            return .gpu
        }
        return estimatedANETime < estimatedGPUTime ? .ane : .gpu
    }

    public var speedupRatio: Double {
        guard estimatedANETime > 0 else { return 1.0 }
        return estimatedGPUTime / estimatedANETime
    }

    public init(
        operationId: String,
        operationType: HLOOpKind,
        compatibility: ANECompatibility,
        estimatedANETime: Double,
        estimatedGPUTime: Double,
        reason: String,
        constraints: [ANEConstraint] = []
    ) {
        self.operationId = operationId
        self.operationType = operationType
        self.compatibility = compatibility
        self.estimatedANETime = estimatedANETime
        self.estimatedGPUTime = estimatedGPUTime
        self.reason = reason
        self.constraints = constraints
    }
}

/// Constraints that limit ANE execution
public enum ANEConstraint: String, Sendable, CaseIterable {
    case tensorSizeTooSmall     // Tensor too small to benefit from ANE
    case tensorSizeTooLarge     // Tensor exceeds ANE memory limits
    case unsupportedDataType    // Data type not supported on ANE
    case unsupportedOperation   // Operation type not supported
    case nonContiguousMemory    // Memory layout not contiguous
    case dynamicShape          // Dynamic shapes not supported
    case unsupportedAttribute   // Operation attribute not supported
}

/// Target execution device
public enum ExecutionDevice: String, Sendable {
    case gpu
    case ane
    case cpu
}

// MARK: - ANE Analyzer

/// Analyzes operations to determine their suitability for Neural Engine execution
public final class ANEAnalyzer: @unchecked Sendable {

    /// Configuration for ANE analysis
    public struct Config: Sendable {
        public var minTensorSizeForANE: Int      // Minimum elements to consider ANE
        public var maxTensorSizeForANE: Int      // Maximum elements ANE can handle
        public var aneOverheadNs: Double         // Fixed overhead for ANE dispatch
        public var preferANEForConvolutions: Bool
        public var preferANEForMatMul: Bool

        public static let `default` = Config(
            minTensorSizeForANE: 4096,          // 64x64 minimum
            maxTensorSizeForANE: 16_777_216,    // ~16M elements
            aneOverheadNs: 50_000,              // 50 microseconds dispatch overhead
            preferANEForConvolutions: true,
            preferANEForMatMul: true
        )

        public init(
            minTensorSizeForANE: Int,
            maxTensorSizeForANE: Int,
            aneOverheadNs: Double,
            preferANEForConvolutions: Bool,
            preferANEForMatMul: Bool
        ) {
            self.minTensorSizeForANE = minTensorSizeForANE
            self.maxTensorSizeForANE = maxTensorSizeForANE
            self.aneOverheadNs = aneOverheadNs
            self.preferANEForConvolutions = preferANEForConvolutions
            self.preferANEForMatMul = preferANEForMatMul
        }
    }

    /// Operations that are typically optimal on ANE
    /// Aligned with MILOpTranslator.isSupported()
    private static let aneOptimalOps: Set<HLOOpKind> = [
        .convolution,
        .dot,
        .dotGeneral,
    ]

    /// Operations that can run on ANE but may not be optimal.
    /// Aligned with MILOpTranslator.isSupported()
    private static let aneCompatibleOps: Set<HLOOpKind> = [
        // P0: Binary arithmetic
        .add, .multiply, .subtract, .divide,
        .maximum, .minimum,
        // P0: Shape manipulation
        .reshape, .transpose, .broadcastInDim,
        // P1: Unary
        .negate, .abs, .exponential, .tanh, .logistic,
        // P1: Aggregate/indexing
        .reduce, .concatenate, .slice,
        // P1: Comparison/selection
        .compare, .select,
        // P2: Additional math
        .power, .log, .sqrt, .rsqrt,
        // Constants
        .constant,
    ]

    /// Operations that cannot run on ANE
    private static let aneIncompatibleOps: Set<HLOOpKind> = [
        .customCall,
        .scatter, .gather,
        .sort,
        .rng, .rngBitGenerator,
        .fft,
        .complex, .real, .imag,
        .whileOp, .ifOp,
    ]

    private let config: Config

    public init(config: Config = .default) {
        self.config = config
    }

    /// Analyzes a single operation for ANE compatibility
    public func analyze(_ operation: HLOOperation) -> ANEOperationAnalysis {
        let opKind = operation.kind

        // Check basic compatibility
        var constraints: [ANEConstraint] = []

        // Check operation type compatibility
        if Self.aneIncompatibleOps.contains(opKind) {
            return ANEOperationAnalysis(
                operationId: operation.result,
                operationType: opKind,
                compatibility: .incompatible,
                estimatedANETime: .infinity,
                estimatedGPUTime: estimateGPUTime(operation),
                reason: "Operation type not supported on ANE",
                constraints: [.unsupportedOperation]
            )
        }

        // Check tensor size constraints
        let tensorSize = operation.resultType.count
        if tensorSize < config.minTensorSizeForANE {
            constraints.append(.tensorSizeTooSmall)
        }
        if tensorSize > config.maxTensorSizeForANE {
            constraints.append(.tensorSizeTooLarge)
        }

        // Check data type support
        if !isDataTypeSupportedOnANE(operation.resultType.elementType) {
            constraints.append(.unsupportedDataType)
        }

        // If we have blocking constraints, mark as incompatible
        let blockingConstraints: Set<ANEConstraint> = [
            .tensorSizeTooLarge,
            .unsupportedDataType,
            .unsupportedOperation
        ]
        let hasBlockingConstraint = constraints.contains { blockingConstraints.contains($0) }

        if hasBlockingConstraint {
            return ANEOperationAnalysis(
                operationId: operation.result,
                operationType: opKind,
                compatibility: .incompatible,
                estimatedANETime: .infinity,
                estimatedGPUTime: estimateGPUTime(operation),
                reason: "Has blocking constraints: \(constraints.map { $0.rawValue }.joined(separator: ", "))",
                constraints: constraints
            )
        }

        // Estimate execution times
        let aneTime = estimateANETime(operation)
        let gpuTime = estimateGPUTime(operation)

        // Determine compatibility level
        let compatibility: ANECompatibility
        let reason: String

        if Self.aneOptimalOps.contains(opKind) && constraints.isEmpty {
            compatibility = .optimal
            reason = "Operation is optimal for ANE"
        } else if Self.aneCompatibleOps.contains(opKind) {
            if constraints.contains(.tensorSizeTooSmall) {
                compatibility = .compatible
                reason = "Tensor size below optimal threshold for ANE"
            } else {
                compatibility = .compatible
                reason = "Operation can run on ANE"
            }
        } else {
            compatibility = .unknown
            reason = "Operation compatibility not determined"
        }

        return ANEOperationAnalysis(
            operationId: operation.result,
            operationType: opKind,
            compatibility: compatibility,
            estimatedANETime: aneTime,
            estimatedGPUTime: gpuTime,
            reason: reason,
            constraints: constraints
        )
    }

    /// Analyzes an entire function for ANE compatibility.
    ///
    /// Applies a chaining bonus: when consecutive ops are all ANE-compatible,
    /// the per-op dispatch overhead is amortized across the chain (CoreML
    /// batches chains into a single ANE dispatch).
    public func analyzeFunction(_ function: HLOFunction) -> FunctionANEAnalysis {
        // First pass: analyze each op individually
        var analyses: [ANEOperationAnalysis] = []
        for operation in function.operations {
            analyses.append(analyze(operation))
        }

        // Second pass: apply chaining bonus for consecutive ANE-compatible ops
        var chainStart = 0
        while chainStart < analyses.count {
            guard analyses[chainStart].compatibility != .incompatible else {
                chainStart += 1
                continue
            }

            // Find chain end
            var chainEnd = chainStart + 1
            while chainEnd < analyses.count && analyses[chainEnd].compatibility != .incompatible {
                chainEnd += 1
            }

            let chainLength = chainEnd - chainStart
            if chainLength >= 2 {
                // Amortize dispatch overhead across chain.
                // Single dispatch overhead instead of per-op overhead.
                let overheadMs = config.aneOverheadNs / 1_000_000.0
                let perOpOverhead = overheadMs / Double(chainLength)

                for i in chainStart..<chainEnd {
                    let original = analyses[i]
                    // Subtract the per-op overhead and add amortized share
                    let adjustedANETime = max(
                        original.estimatedANETime - overheadMs + perOpOverhead,
                        0.001
                    )
                    analyses[i] = ANEOperationAnalysis(
                        operationId: original.operationId,
                        operationType: original.operationType,
                        compatibility: original.compatibility,
                        estimatedANETime: adjustedANETime,
                        estimatedGPUTime: original.estimatedGPUTime,
                        reason: original.reason + " (chain bonus: \(chainLength) ops)",
                        constraints: original.constraints
                    )
                }
            }

            chainStart = chainEnd
        }

        return FunctionANEAnalysis(
            functionName: function.name,
            operationAnalyses: analyses
        )
    }

    /// Checks if a data type is supported on ANE
    private func isDataTypeSupportedOnANE(_ elementType: ElementType) -> Bool {
        switch elementType {
        case .float16, .float32, .bfloat16:
            return true
        case .int8, .uint8:
            return true // For quantized models
        default:
            return false
        }
    }

    /// Estimates ANE execution time for an operation
    private func estimateANETime(_ operation: HLOOperation) -> Double {
        let baseTime: Double

        switch operation.kind {
        case .dot:
            // ANE is highly optimized for matrix multiply
            let elements = Double(operation.resultType.count)
            baseTime = elements * 0.00001 // ~10ns per element
        case .convolution:
            // ANE excels at convolutions
            let elements = Double(operation.resultType.count)
            baseTime = elements * 0.000005 // ~5ns per element
        default:
            // Generic elementwise
            let elements = Double(operation.resultType.count)
            baseTime = elements * 0.00002 // ~20ns per element
        }

        // Add dispatch overhead
        return baseTime + config.aneOverheadNs / 1_000_000.0
    }

    /// Estimates GPU execution time for an operation
    private func estimateGPUTime(_ operation: HLOOperation) -> Double {
        let elements = Double(operation.resultType.count)

        switch operation.kind {
        case .dot:
            // GPU is also fast at matmul but has different characteristics
            return elements * 0.00002 // ~20ns per element
        case .convolution:
            return elements * 0.000015 // ~15ns per element
        default:
            // GPU is fast at elementwise
            return elements * 0.00001 // ~10ns per element
        }
    }
}

/// Analysis result for an entire function
public struct FunctionANEAnalysis: Sendable {
    public let functionName: String
    public let operationAnalyses: [ANEOperationAnalysis]

    public var optimalOpsCount: Int {
        operationAnalyses.filter { $0.compatibility == .optimal }.count
    }

    public var compatibleOpsCount: Int {
        operationAnalyses.filter { $0.compatibility == .compatible }.count
    }

    public var incompatibleOpsCount: Int {
        operationAnalyses.filter { $0.compatibility == .incompatible }.count
    }

    public var aneRecommendedOps: [ANEOperationAnalysis] {
        operationAnalyses.filter { $0.preferredDevice == .ane }
    }

    public var estimatedSpeedupRatio: Double {
        let aneOps = aneRecommendedOps
        guard !aneOps.isEmpty else { return 1.0 }

        let totalGPUTime = aneOps.reduce(0.0) { $0 + $1.estimatedGPUTime }
        let totalANETime = aneOps.reduce(0.0) { $0 + $1.estimatedANETime }

        guard totalANETime > 0 else { return 1.0 }
        return totalGPUTime / totalANETime
    }
}

// MARK: - GPU-ANE Partitioner

/// Partitions operations between GPU and ANE for optimal execution
public final class GPUANEPartitioner: @unchecked Sendable {

    /// Configuration for partitioning
    public struct Config: Sendable {
        public var minOpsForANEPartition: Int     // Minimum ops to justify ANE context switch
        public var maxTransferOverheadMs: Double  // Maximum acceptable transfer overhead
        public var enablePipelining: Bool         // Enable GPU-ANE pipelining
        public var balanceLoad: Bool              // Try to balance load between devices

        public static let `default` = Config(
            minOpsForANEPartition: 3,
            maxTransferOverheadMs: 1.0,
            enablePipelining: true,
            balanceLoad: true
        )

        public init(
            minOpsForANEPartition: Int,
            maxTransferOverheadMs: Double,
            enablePipelining: Bool,
            balanceLoad: Bool
        ) {
            self.minOpsForANEPartition = minOpsForANEPartition
            self.maxTransferOverheadMs = maxTransferOverheadMs
            self.enablePipelining = enablePipelining
            self.balanceLoad = balanceLoad
        }
    }

    /// A partition of operations assigned to a specific device
    public struct DevicePartition: Sendable {
        public let device: ExecutionDevice
        public let operationIds: [String]
        public let estimatedTimeMs: Double
        public let inputTensors: Set<String>   // Tensors needed from other device
        public let outputTensors: Set<String>  // Tensors produced for other device

        public init(
            device: ExecutionDevice,
            operationIds: [String],
            estimatedTimeMs: Double,
            inputTensors: Set<String>,
            outputTensors: Set<String>
        ) {
            self.device = device
            self.operationIds = operationIds
            self.estimatedTimeMs = estimatedTimeMs
            self.inputTensors = inputTensors
            self.outputTensors = outputTensors
        }
    }

    /// Result of partitioning
    public struct PartitionPlan: Sendable {
        public let partitions: [DevicePartition]
        public let executionOrder: [Int]        // Indices into partitions array
        public let estimatedTotalTimeMs: Double
        public let estimatedTransferTimeMs: Double
        public let canPipeline: Bool

        public var gpuPartitions: [DevicePartition] {
            partitions.filter { $0.device == .gpu }
        }

        public var anePartitions: [DevicePartition] {
            partitions.filter { $0.device == .ane }
        }

        public init(
            partitions: [DevicePartition],
            executionOrder: [Int],
            estimatedTotalTimeMs: Double,
            estimatedTransferTimeMs: Double,
            canPipeline: Bool
        ) {
            self.partitions = partitions
            self.executionOrder = executionOrder
            self.estimatedTotalTimeMs = estimatedTotalTimeMs
            self.estimatedTransferTimeMs = estimatedTransferTimeMs
            self.canPipeline = canPipeline
        }
    }

    private let config: Config
    private let analyzer: ANEAnalyzer

    public init(config: Config = .default, analyzer: ANEAnalyzer = ANEAnalyzer()) {
        self.config = config
        self.analyzer = analyzer
    }

    /// Creates a partition plan for a function
    public func partition(_ function: HLOFunction) -> PartitionPlan {
        let analysis = analyzer.analyzeFunction(function)

        // Build dependency graph
        let dependencies = buildDependencyGraph(function)

        // Build op map for tensor size lookups
        let opMap = Dictionary(
            uniqueKeysWithValues: function.operations.map { ($0.result, $0) }
        )

        // Assign operations to devices
        var assignments: [String: ExecutionDevice] = [:]
        for opAnalysis in analysis.operationAnalyses {
            assignments[opAnalysis.operationId] = opAnalysis.preferredDevice
        }

        // Group consecutive operations on same device into partitions
        let partitions = createPartitions(function, assignments: assignments, analysis: analysis)

        // Optimize partition boundaries to minimize transfers
        let optimizedPartitions = optimizePartitions(partitions, dependencies: dependencies)

        // Determine execution order
        let executionOrder = determineExecutionOrder(optimizedPartitions, dependencies: dependencies)

        // Calculate timing estimates using actual tensor sizes
        let totalTime = optimizedPartitions.reduce(0.0) { $0 + $1.estimatedTimeMs }
        let transferTime = calculateTransferTime(optimizedPartitions, opMap: opMap, function: function)
        let canPipeline = config.enablePipelining && checkPipeliningPossible(optimizedPartitions)

        return PartitionPlan(
            partitions: optimizedPartitions,
            executionOrder: executionOrder,
            estimatedTotalTimeMs: totalTime + transferTime,
            estimatedTransferTimeMs: transferTime,
            canPipeline: canPipeline
        )
    }

    /// Builds dependency graph from operation operands
    private func buildDependencyGraph(_ function: HLOFunction) -> [String: Set<String>] {
        var dependencies: [String: Set<String>] = [:]

        for op in function.operations {
            dependencies[op.result] = Set(op.operands)
        }

        return dependencies
    }

    /// Creates partitions from device assignments
    private func createPartitions(
        _ function: HLOFunction,
        assignments: [String: ExecutionDevice],
        analysis: FunctionANEAnalysis
    ) -> [DevicePartition] {
        var partitions: [DevicePartition] = []
        var currentDevice: ExecutionDevice?
        var currentOps: [String] = []
        var currentTime: Double = 0

        let analysisMap = Dictionary(
            uniqueKeysWithValues: analysis.operationAnalyses.map { ($0.operationId, $0) }
        )

        for op in function.operations {
            let device = assignments[op.result] ?? .gpu

            if device != currentDevice && !currentOps.isEmpty {
                // Finish current partition
                let partition = DevicePartition(
                    device: currentDevice!,
                    operationIds: currentOps,
                    estimatedTimeMs: currentTime,
                    inputTensors: Set(),
                    outputTensors: Set()
                )
                partitions.append(partition)
                currentOps = []
                currentTime = 0
            }

            currentDevice = device
            currentOps.append(op.result)

            if let opAnalysis = analysisMap[op.result] {
                currentTime += device == .ane ? opAnalysis.estimatedANETime : opAnalysis.estimatedGPUTime
            }
        }

        // Add final partition
        if !currentOps.isEmpty {
            let partition = DevicePartition(
                device: currentDevice!,
                operationIds: currentOps,
                estimatedTimeMs: currentTime,
                inputTensors: Set(),
                outputTensors: Set()
            )
            partitions.append(partition)
        }

        return partitions
    }

    /// Optimizes partitions to reduce device transfers
    private func optimizePartitions(
        _ partitions: [DevicePartition],
        dependencies: [String: Set<String>]
    ) -> [DevicePartition] {
        // For now, merge small adjacent partitions on the same device
        var optimized: [DevicePartition] = []

        for partition in partitions {
            if let last = optimized.last,
               last.device == partition.device {
                // Merge with previous partition
                let merged = DevicePartition(
                    device: last.device,
                    operationIds: last.operationIds + partition.operationIds,
                    estimatedTimeMs: last.estimatedTimeMs + partition.estimatedTimeMs,
                    inputTensors: last.inputTensors,
                    outputTensors: partition.outputTensors
                )
                optimized[optimized.count - 1] = merged
            } else if partition.operationIds.count < config.minOpsForANEPartition &&
                        partition.device == .ane {
                // Too small for ANE, move to GPU
                let gpuPartition = DevicePartition(
                    device: .gpu,
                    operationIds: partition.operationIds,
                    estimatedTimeMs: partition.estimatedTimeMs * 1.5, // Estimate GPU time
                    inputTensors: partition.inputTensors,
                    outputTensors: partition.outputTensors
                )

                // Try to merge with adjacent GPU partition
                if let last = optimized.last, last.device == .gpu {
                    let merged = DevicePartition(
                        device: .gpu,
                        operationIds: last.operationIds + gpuPartition.operationIds,
                        estimatedTimeMs: last.estimatedTimeMs + gpuPartition.estimatedTimeMs,
                        inputTensors: last.inputTensors,
                        outputTensors: gpuPartition.outputTensors
                    )
                    optimized[optimized.count - 1] = merged
                } else {
                    optimized.append(gpuPartition)
                }
            } else {
                optimized.append(partition)
            }
        }

        // Calculate cross-device tensors
        return calculateCrossDeviceTensors(optimized, dependencies: dependencies)
    }

    /// Calculates which tensors need to be transferred between devices
    private func calculateCrossDeviceTensors(
        _ partitions: [DevicePartition],
        dependencies: [String: Set<String>]
    ) -> [DevicePartition] {
        // Build mapping of operation to partition index
        var opToPartition: [String: Int] = [:]
        for (i, partition) in partitions.enumerated() {
            for opId in partition.operationIds {
                opToPartition[opId] = i
            }
        }

        // Calculate cross-device dependencies for each partition
        return partitions.enumerated().map { (i, partition) in
            var inputTensors: Set<String> = []
            var outputTensors: Set<String> = []

            for opId in partition.operationIds {
                // Check inputs
                if let deps = dependencies[opId] {
                    for dep in deps {
                        if let depPartition = opToPartition[dep], depPartition != i {
                            // Dependency from different partition
                            if partitions[depPartition].device != partition.device {
                                inputTensors.insert(dep)
                            }
                        }
                    }
                }

                // Check outputs (operations in later partitions that depend on this)
                for (j, otherPartition) in partitions.enumerated() where j > i {
                    for otherOpId in otherPartition.operationIds {
                        if let deps = dependencies[otherOpId], deps.contains(opId) {
                            if otherPartition.device != partition.device {
                                outputTensors.insert(opId)
                            }
                        }
                    }
                }
            }

            return DevicePartition(
                device: partition.device,
                operationIds: partition.operationIds,
                estimatedTimeMs: partition.estimatedTimeMs,
                inputTensors: inputTensors,
                outputTensors: outputTensors
            )
        }
    }

    /// Determines execution order respecting dependencies
    private func determineExecutionOrder(
        _ partitions: [DevicePartition],
        dependencies: [String: Set<String>]
    ) -> [Int] {
        // For now, just execute in order
        // A more sophisticated implementation would enable parallelism
        return Array(0..<partitions.count)
    }

    /// Calculates estimated transfer time between devices using actual tensor sizes.
    private func calculateTransferTime(
        _ partitions: [DevicePartition],
        opMap: [String: HLOOperation],
        function: HLOFunction
    ) -> Double {
        let inputMap = Dictionary(uniqueKeysWithValues: function.inputs.map { ($0.name, $0.type) })
        var totalBytes = 0

        for partition in partitions {
            for tensorName in partition.inputTensors {
                if let op = opMap[tensorName] {
                    totalBytes += op.resultType.byteCount
                } else if let inputType = inputMap[tensorName] {
                    totalBytes += inputType.byteCount
                }
            }
            for tensorName in partition.outputTensors {
                if let op = opMap[tensorName] {
                    totalBytes += op.resultType.byteCount
                }
            }
        }

        // On Apple Silicon unified memory, "transfer" is mostly API conversion overhead.
        // Estimate ~10 GB/s effective bandwidth for the conversion.
        let bytesPerMs = 10_000_000.0
        return Double(totalBytes) / bytesPerMs
    }

    /// Checks if GPU-ANE pipelining is possible
    private func checkPipeliningPossible(_ partitions: [DevicePartition]) -> Bool {
        // Pipelining is possible if we have both GPU and ANE partitions
        // and they don't have circular dependencies
        let hasGPU = partitions.contains { $0.device == .gpu }
        let hasANE = partitions.contains { $0.device == .ane }
        return hasGPU && hasANE && partitions.count >= 2
    }
}

// MARK: - Heterogeneous Executor

/// Executes computation across multiple devices (GPU, ANE, CPU)
public final class HeterogeneousExecutor: @unchecked Sendable {

    /// Execution statistics
    public struct ExecutionStats: Sendable {
        public var gpuTimeMs: Double
        public var aneTimeMs: Double
        public var transferTimeMs: Double
        public var totalTimeMs: Double
        public var operationsExecuted: Int
        public var partitionsExecuted: Int

        public init() {
            self.gpuTimeMs = 0
            self.aneTimeMs = 0
            self.transferTimeMs = 0
            self.totalTimeMs = 0
            self.operationsExecuted = 0
            self.partitionsExecuted = 0
        }
    }

    /// Configuration for heterogeneous execution
    public struct Config: Sendable {
        public var enableANE: Bool
        public var enableGPU: Bool
        public var enableCPUFallback: Bool
        public var enablePipelining: Bool

        public static let `default` = Config(
            enableANE: true,
            enableGPU: true,
            enableCPUFallback: true,
            enablePipelining: false
        )

        public init(
            enableANE: Bool,
            enableGPU: Bool,
            enableCPUFallback: Bool,
            enablePipelining: Bool = false
        ) {
            self.enableANE = enableANE
            self.enableGPU = enableGPU
            self.enableCPUFallback = enableCPUFallback
            self.enablePipelining = enablePipelining
        }
    }

    private let config: Config
    private let partitioner: GPUANEPartitioner
    private let metalExecutor: MetalExecutor
    private let coreMLBridge: ANERuntime.CoreMLBridge
    private var transferManager: TensorTransferManager
    private var stats: ExecutionStats
    private let lock = NSLock()

    /// Dedicated queue for ANE (CoreML) execution in pipelined mode.
    private let aneQueue = DispatchQueue(label: "com.metalHLO.ane", qos: .userInitiated)

    /// Cache of compiled CoreML programs keyed by partition op hash.
    private var aneCache: [String: ANERuntime.CoreMLProgram] = [:]

    public init(
        metalExecutor: MetalExecutor,
        config: Config = .default,
        partitioner: GPUANEPartitioner? = nil
    ) {
        self.config = config
        self.partitioner = partitioner ?? GPUANEPartitioner()
        self.metalExecutor = metalExecutor
        self.coreMLBridge = ANERuntime.CoreMLBridge()
        self.transferManager = TensorTransferManager(device: metalExecutor.device)
        self.stats = ExecutionStats()
    }

    /// Executes a function using heterogeneous devices.
    ///
    /// Partitions the function between GPU and ANE, executes each partition
    /// on its assigned device, and transfers tensors between devices at
    /// partition boundaries.
    ///
    /// - Parameters:
    ///   - function: The HLO function to execute.
    ///   - inputs: Named input BufferStorages.
    /// - Returns: Execution result with per-partition timing and output storages.
    public func execute(
        _ function: HLOFunction,
        inputs: [String: BufferStorage]
    ) throws -> HeterogeneousExecutionResult {
        let startTime = DispatchTime.now()

        // Partition the function
        let plan = partitioner.partition(function)

        // Reset transfer manager for this execution
        transferManager.clear()

        // Store initial inputs in transfer manager
        for (name, storage) in inputs {
            transferManager.storeGPUResult(name: name, storage: storage)
        }

        let partitionResults: [PartitionResult]
        var localStats = ExecutionStats()

        if config.enablePipelining && plan.canPipeline {
            partitionResults = try executePipelined(plan: plan, function: function, stats: &localStats)
        } else {
            partitionResults = try executeSequential(plan: plan, function: function, stats: &localStats)
        }

        let endTime = DispatchTime.now()
        let totalTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0

        localStats.totalTimeMs = totalTime
        localStats.transferTimeMs = plan.estimatedTransferTimeMs

        lock.lock()
        stats.gpuTimeMs += localStats.gpuTimeMs
        stats.aneTimeMs += localStats.aneTimeMs
        stats.transferTimeMs += localStats.transferTimeMs
        stats.totalTimeMs += localStats.totalTimeMs
        stats.operationsExecuted += localStats.operationsExecuted
        stats.partitionsExecuted += localStats.partitionsExecuted
        lock.unlock()

        // Collect outputs for the original function's return values
        var outputStorages: [String: BufferStorage] = [:]
        for retVal in function.returnValues {
            if let storage = try? transferManager.getBufferStorage(name: retVal) {
                outputStorages[retVal] = storage
            }
        }

        return HeterogeneousExecutionResult(
            success: true,
            partitionResults: partitionResults,
            stats: localStats,
            plan: plan,
            outputStorages: outputStorages
        )
    }

    /// Executes partitions sequentially in dependency order.
    private func executeSequential(
        plan: GPUANEPartitioner.PartitionPlan,
        function: HLOFunction,
        stats localStats: inout ExecutionStats
    ) throws -> [PartitionResult] {
        var partitionResults: [PartitionResult] = []

        for partitionIndex in plan.executionOrder {
            let partition = plan.partitions[partitionIndex]

            let result: PartitionResult
            do {
                result = try executePartition(partition, function: function)
            } catch {
                // ANE failure: fall back to GPU if enabled
                if partition.device == .ane && config.enableCPUFallback {
                    let gpuPartition = GPUANEPartitioner.DevicePartition(
                        device: .gpu,
                        operationIds: partition.operationIds,
                        estimatedTimeMs: partition.estimatedTimeMs,
                        inputTensors: partition.inputTensors,
                        outputTensors: partition.outputTensors
                    )
                    result = try executePartition(gpuPartition, function: function)
                } else {
                    throw error
                }
            }

            partitionResults.append(result)
            localStats.partitionsExecuted += 1
            localStats.operationsExecuted += partition.operationIds.count

            switch result.device {
            case .gpu:
                localStats.gpuTimeMs += result.executionTimeMs
            case .ane:
                localStats.aneTimeMs += result.executionTimeMs
            case .cpu:
                break
            }
        }

        return partitionResults
    }

    /// Executes partitions with GPU/ANE pipelining.
    ///
    /// When an ANE partition has no data dependency on a concurrent GPU partition,
    /// they can overlap: CoreML prediction runs on `aneQueue` while Metal commands
    /// execute on the GPU. At dependency boundaries, we synchronize.
    private func executePipelined(
        plan: GPUANEPartitioner.PartitionPlan,
        function: HLOFunction,
        stats localStats: inout ExecutionStats
    ) throws -> [PartitionResult] {
        // Use a class to hold mutable state shared across the async closure,
        // avoiding sendable capture warnings on mutable vars.
        final class PipelineState: @unchecked Sendable {
            let lock = NSLock()
            var results: [PartitionResult?]
            var error: Error?

            init(count: Int) {
                self.results = Array(repeating: nil, count: count)
            }
        }
        let state = PipelineState(count: plan.partitions.count)

        // Track which partitions have completed (for dependency checking)
        var completedPartitions: Set<Int> = []

        for (orderIdx, partitionIndex) in plan.executionOrder.enumerated() {
            let partition = plan.partitions[partitionIndex]

            // Determine if this partition depends on the immediately previous partition
            // by checking if any of its input tensors come from previous partitions
            let previousPartitionOps: Set<String>
            if orderIdx > 0 {
                let prevIdx = plan.executionOrder[orderIdx - 1]
                previousPartitionOps = Set(plan.partitions[prevIdx].operationIds)
            } else {
                previousPartitionOps = []
            }

            let dependsOnPrevious = !partition.inputTensors.intersection(previousPartitionOps).isEmpty

            // Can we overlap this ANE partition with previous GPU work?
            let canOverlap = partition.device == .ane
                && !dependsOnPrevious
                && orderIdx > 0
                && completedPartitions.contains(plan.executionOrder[orderIdx - 1]) == false

            if canOverlap {
                let semaphore = DispatchSemaphore(value: 0)

                aneQueue.async { [self] in
                    do {
                        let result = try self.executePartitionWithFallback(partition, function: function)
                        state.lock.lock()
                        state.results[partitionIndex] = result
                        state.lock.unlock()
                    } catch {
                        state.lock.lock()
                        state.error = error
                        state.lock.unlock()
                    }
                    semaphore.signal()
                }

                // Check if the next partition needs this one's output
                let nextIdx = orderIdx + 1
                let mustWait: Bool
                if nextIdx < plan.executionOrder.count {
                    let nextPartition = plan.partitions[plan.executionOrder[nextIdx]]
                    let partitionOutputs = Set(partition.operationIds)
                    mustWait = !nextPartition.inputTensors.intersection(partitionOutputs).isEmpty
                } else {
                    mustWait = true // Last partition, must wait
                }

                if mustWait {
                    semaphore.wait()
                    completedPartitions.insert(partitionIndex)

                    // Check for async errors
                    if let error = state.error { throw error }
                }
            } else {
                // Execute synchronously (GPU partition or ANE with dependency)
                let result = try executePartitionWithFallback(partition, function: function)

                state.lock.lock()
                state.results[partitionIndex] = result
                state.lock.unlock()
                completedPartitions.insert(partitionIndex)
            }
        }

        // Wait for any remaining async work
        aneQueue.sync {}
        if let error = state.error { throw error }

        // Collect results in execution order
        var orderedResults: [PartitionResult] = []
        for partitionIndex in plan.executionOrder {
            if let result = state.results[partitionIndex] {
                orderedResults.append(result)
                localStats.partitionsExecuted += 1
                localStats.operationsExecuted += plan.partitions[partitionIndex].operationIds.count

                switch result.device {
                case .gpu:
                    localStats.gpuTimeMs += result.executionTimeMs
                case .ane:
                    localStats.aneTimeMs += result.executionTimeMs
                case .cpu:
                    break
                }
            }
        }

        return orderedResults
    }

    /// Executes a partition with ANE-to-GPU fallback on failure.
    private func executePartitionWithFallback(
        _ partition: GPUANEPartitioner.DevicePartition,
        function: HLOFunction
    ) throws -> PartitionResult {
        do {
            return try executePartition(partition, function: function)
        } catch {
            if partition.device == .ane && config.enableCPUFallback {
                let gpuPartition = GPUANEPartitioner.DevicePartition(
                    device: .gpu,
                    operationIds: partition.operationIds,
                    estimatedTimeMs: partition.estimatedTimeMs,
                    inputTensors: partition.inputTensors,
                    outputTensors: partition.outputTensors
                )
                return try executePartition(gpuPartition, function: function)
            }
            throw error
        }
    }

    /// Executes a single partition on its assigned device.
    private func executePartition(
        _ partition: GPUANEPartitioner.DevicePartition,
        function: HLOFunction
    ) throws -> PartitionResult {
        let startTime = DispatchTime.now()

        let opIds = Set(partition.operationIds)
        let externalConsumers = SubFunctionExtractor.findExternalConsumers(of: opIds, in: function)
        let subFunction = SubFunctionExtractor.extract(
            from: function,
            operationIds: opIds,
            outputConsumers: externalConsumers
        )

        switch partition.device {
        case .gpu:
            try executeOnGPU(subFunction)
        case .ane:
            try executeOnANE(subFunction)
        case .cpu:
            throw HeterogeneousExecutorError.deviceNotAvailable("CPU execution not implemented")
        }

        let endTime = DispatchTime.now()
        let executionTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0

        return PartitionResult(
            device: partition.device,
            operationIds: partition.operationIds,
            executionTimeMs: executionTime,
            success: true
        )
    }

    /// Executes a sub-function on GPU via MetalExecutor (MPSGraph path).
    private func executeOnGPU(_ subFunction: HLOFunction) throws {
        // Build an HLOModule from the sub-function
        let module = HLOModule(name: subFunction.name, function: subFunction)

        // Compile via MPSGraph
        let compiled = try metalExecutor.compile(module: module)

        // Gather inputs from transfer manager as BufferStorage
        var inputStorages: [BufferStorage] = []
        for arg in subFunction.inputs {
            let storage = try transferManager.getBufferStorage(name: arg.name)
            inputStorages.append(storage)
        }

        // Execute
        let outputs = try metalExecutor.execute(compiled: compiled, inputs: inputStorages)

        // Store outputs in transfer manager
        for (i, retVal) in subFunction.returnValues.enumerated() where i < outputs.count {
            transferManager.storeGPUResult(name: retVal, storage: outputs[i])
        }
    }

    /// Executes a sub-function on ANE via CoreMLBridge.
    private func executeOnANE(_ subFunction: HLOFunction) throws {
        let builder = CoreMLOpBuilder()
        let (coremlInputs, ops, returnVar) = try builder.build(function: subFunction)

        // Check cache
        let cacheKey = subFunction.operations.map { $0.result }.joined(separator: "_")
        let program: ANERuntime.CoreMLProgram
        if let cached = aneCache[cacheKey] {
            program = cached
        } else {
            program = try coreMLBridge.compile(
                inputs: coremlInputs,
                operations: ops,
                returnVar: returnVar
            )
            aneCache[cacheKey] = program
        }

        // Gather inputs from transfer manager as CPU arrays
        var aneInputs: [(name: String, data: [Float], shape: [Int])] = []
        for (coremlInput, arg) in zip(coremlInputs, subFunction.inputs) {
            let cpuData = try transferManager.getCPUArray(name: arg.name)
            aneInputs.append((name: coremlInput.name, data: cpuData.data, shape: cpuData.shape))
        }

        // Execute via CoreML
        let results = try coreMLBridge.execute(program, inputs: aneInputs)

        // Store outputs — ANE returns a single flat Float array for the return value
        for retVal in subFunction.returnValues {
            let opMap = Dictionary(
                uniqueKeysWithValues: subFunction.operations.map { ($0.result, $0) }
            )
            if let op = opMap[retVal] {
                transferManager.storeCPUResult(
                    name: retVal,
                    data: results,
                    shape: op.resultType.shape,
                    elementType: op.resultType.elementType
                )
            }
        }
    }

    /// Returns current execution statistics.
    public func getStats() -> ExecutionStats {
        lock.lock()
        defer { lock.unlock() }
        return stats
    }

    /// Resets execution statistics.
    public func resetStats() {
        lock.lock()
        defer { lock.unlock() }
        stats = ExecutionStats()
    }

    /// Clears cached ANE compilations.
    public func clearCache() {
        aneCache.removeAll()
        coreMLBridge.clearCache()
    }
}

/// Result of executing a single partition
public struct PartitionResult: Sendable {
    public let device: ExecutionDevice
    public let operationIds: [String]
    public let executionTimeMs: Double
    public let success: Bool
}

/// Result of heterogeneous execution
/// Result of heterogeneous execution
public struct HeterogeneousExecutionResult: @unchecked Sendable {
    public let success: Bool
    public let partitionResults: [PartitionResult]
    public let stats: HeterogeneousExecutor.ExecutionStats
    public let plan: GPUANEPartitioner.PartitionPlan
    /// Output buffers keyed by return value SSA name.
    public let outputStorages: [String: BufferStorage]
}

/// Errors that can occur during heterogeneous execution
public enum HeterogeneousExecutorError: Error, Sendable {
    case deviceNotAvailable(String)
    case deviceInitializationFailed(String)
    case executionFailed(String)
    case partitioningFailed(String)
}

// MARK: - ANE Capability Detection

/// Detects ANE capabilities on the current device
public final class ANECapabilityDetector: @unchecked Sendable {

    /// ANE capabilities
    public struct ANECapabilities: Sendable {
        public let isAvailable: Bool
        public let maxMemoryBytes: Int
        public let supportsFloat16: Bool
        public let supportsInt8: Bool
        public let estimatedTOPS: Double // Tera operations per second
        public let generationName: String

        public static let unavailable = ANECapabilities(
            isAvailable: false,
            maxMemoryBytes: 0,
            supportsFloat16: false,
            supportsInt8: false,
            estimatedTOPS: 0,
            generationName: "None"
        )
    }

    /// Detects ANE capabilities
    public func detect() -> ANECapabilities {
        // In a real implementation, this would query the system
        // For now, we return estimated values for modern Apple Silicon

        #if os(macOS) || os(iOS)
        // Assume M1/M2/M3 class Neural Engine
        return ANECapabilities(
            isAvailable: true,
            maxMemoryBytes: 16_000_000_000, // 16GB unified memory access
            supportsFloat16: true,
            supportsInt8: true,
            estimatedTOPS: 15.8, // M1 Neural Engine ~15.8 TOPS
            generationName: "Apple Neural Engine"
        )
        #else
        return ANECapabilities.unavailable
        #endif
    }

    /// Checks if an operation can run on ANE given capabilities
    public func canRunOnANE(_ operation: HLOOperation, capabilities: ANECapabilities) -> Bool {
        guard capabilities.isAvailable else { return false }

        // Check memory requirements
        let memoryNeeded = operation.resultType.byteCount
        if memoryNeeded > capabilities.maxMemoryBytes {
            return false
        }

        // Check data type support
        switch operation.resultType.elementType {
        case .float16, .bfloat16:
            return capabilities.supportsFloat16
        case .int8, .uint8:
            return capabilities.supportsInt8
        case .float32:
            return true // Converted to float16 on ANE
        default:
            return false
        }
    }
}
