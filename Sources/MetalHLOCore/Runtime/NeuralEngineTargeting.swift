// NeuralEngineTargeting.swift
// Neural Engine Targeting for MetalHLO
// Enables heterogeneous execution across GPU and Apple Neural Engine (ANE)

import Foundation
import Metal
import CoreML
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

        // Apply shape efficiency penalty.
        // ANE's [1,C,1,S] layout wastes memory for poorly-shaped tensors
        // (e.g., very small spatial dims or odd channel counts).
        let shapeEfficiency = aneShapeEfficiency(operation.resultType.shape)
        let shapePenalty = 1.0 / max(shapeEfficiency, 0.1) // Up to 10x penalty

        // Add dispatch overhead
        return baseTime * shapePenalty + config.aneOverheadNs / 1_000_000.0
    }

    /// Estimates how efficiently a tensor shape maps to ANE's [1,C,1,S] layout.
    ///
    /// Returns a value in (0, 1]. Shapes with high alignment waste get lower
    /// scores, biasing the cost model toward GPU.
    private func aneShapeEfficiency(_ shape: [Int]) -> Double {
        guard !shape.isEmpty else { return 1.0 }

        // Map logical shape to ANE physical layout [1, C, 1, S]
        let c: Int
        let s: Int
        switch shape.count {
        case 1:
            c = 1
            s = shape[0]
        case 2:
            c = shape[0]
            s = shape[1]
        case 3:
            c = shape[1]
            s = shape[0] * shape[2]
        default:
            c = shape.count >= 2 ? shape[1] : 1
            s = shape.reduce(1, *) / max(c, 1)
        }

        // ANE aligns channels to multiples of 8 and spatial to multiples of 64 bytes
        let alignedC = ((c + 7) / 8) * 8
        let alignedS = ((s * 2 + 63) / 64) * 32  // FP16 elements per 64-byte row

        let usefulElements = c * s
        let allocatedElements = alignedC * max(alignedS, s)
        guard allocatedElements > 0 else { return 1.0 }

        return Double(usefulElements) / Double(allocatedElements)
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
        public var aneSRAMCapacityBytes: Int      // ANE SRAM capacity for partition splitting

        /// When true, biases forward-pass ops (conv, matmul, activations) toward ANE
        /// and backward-pass ops (transpose→dot patterns, reductions feeding gradients)
        /// toward GPU for FP32 precision.
        public var trainingMode: Bool

        public static let `default` = Config(
            minOpsForANEPartition: 3,
            maxTransferOverheadMs: 1.0,
            enablePipelining: true,
            balanceLoad: true,
            aneSRAMCapacityBytes: 32 * 1024 * 1024,
            trainingMode: false
        )

        public init(
            minOpsForANEPartition: Int,
            maxTransferOverheadMs: Double,
            enablePipelining: Bool,
            balanceLoad: Bool,
            aneSRAMCapacityBytes: Int = 32 * 1024 * 1024,
            trainingMode: Bool = false
        ) {
            self.minOpsForANEPartition = minOpsForANEPartition
            self.maxTransferOverheadMs = maxTransferOverheadMs
            self.enablePipelining = enablePipelining
            self.balanceLoad = balanceLoad
            self.aneSRAMCapacityBytes = aneSRAMCapacityBytes
            self.trainingMode = trainingMode
        }
    }

    /// A partition of operations assigned to a specific device
    public struct DevicePartition: Sendable {
        public let device: ExecutionDevice
        public let operationIds: [String]
        public let estimatedTimeMs: Double
        public let inputTensors: Set<String>   // Tensors needed from other device
        public let outputTensors: Set<String>  // Tensors produced for other device
        public let workingSetBytes: Int        // Peak memory usage during partition execution

        public init(
            device: ExecutionDevice,
            operationIds: [String],
            estimatedTimeMs: Double,
            inputTensors: Set<String>,
            outputTensors: Set<String>,
            workingSetBytes: Int = 0
        ) {
            self.device = device
            self.operationIds = operationIds
            self.estimatedTimeMs = estimatedTimeMs
            self.inputTensors = inputTensors
            self.outputTensors = outputTensors
            self.workingSetBytes = workingSetBytes
        }
    }

    /// Result of partitioning
    public struct PartitionPlan: Sendable {
        public let partitions: [DevicePartition]
        public let executionOrder: [Int]        // Indices into partitions array
        public let concurrencyLevels: [[Int]]   // Groups of partition indices runnable in parallel
        public let estimatedTotalTimeMs: Double
        public let estimatedTransferTimeMs: Double
        public let canPipeline: Bool

        public var gpuPartitions: [DevicePartition] {
            partitions.filter { $0.device == .gpu }
        }

        public var anePartitions: [DevicePartition] {
            partitions.filter { $0.device == .ane }
        }

        /// Whether any concurrency level has multiple partitions (true parallelism possible).
        public var hasConcurrentLevels: Bool {
            concurrencyLevels.contains { $0.count > 1 }
        }

        public init(
            partitions: [DevicePartition],
            executionOrder: [Int],
            concurrencyLevels: [[Int]] = [],
            estimatedTotalTimeMs: Double,
            estimatedTransferTimeMs: Double,
            canPipeline: Bool
        ) {
            self.partitions = partitions
            self.executionOrder = executionOrder
            self.concurrencyLevels = concurrencyLevels.isEmpty
                ? [Array(0..<partitions.count)]  // Default: all in one level (sequential)
                : concurrencyLevels
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
            if config.trainingMode {
                assignments[opAnalysis.operationId] = trainingBiasedDevice(
                    opAnalysis, opMap: opMap
                )
            } else {
                assignments[opAnalysis.operationId] = opAnalysis.preferredDevice
            }
        }

        // Group consecutive operations on same device into partitions
        let partitions = createPartitions(function, assignments: assignments, analysis: analysis)

        // Optimize partition boundaries to minimize transfers
        let optimizedPartitions = optimizePartitions(partitions, dependencies: dependencies)

        // Split ANE partitions that exceed SRAM capacity
        let memoryAwarePartitions = splitOversizedPartitions(optimizedPartitions, function: function)

        // Recalculate cross-device tensors after splitting
        let finalPartitions = calculateCrossDeviceTensors(memoryAwarePartitions, dependencies: dependencies)

        // Build concurrency levels (also produces execution order)
        let (executionOrder, concurrencyLevels) = buildConcurrencyLevels(
            finalPartitions, dependencies: dependencies
        )

        // Calculate timing estimates using actual tensor sizes
        let totalTime = finalPartitions.reduce(0.0) { $0 + $1.estimatedTimeMs }
        let transferTime = calculateTransferTime(finalPartitions, opMap: opMap, function: function)
        let canPipeline = config.enablePipelining && checkPipeliningPossible(finalPartitions)

        return PartitionPlan(
            partitions: finalPartitions,
            executionOrder: executionOrder,
            concurrencyLevels: concurrencyLevels,
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

    /// Applies training-mode bias to device assignment.
    ///
    /// Forward-pass ops (conv, matmul, activations) are biased toward ANE.
    /// Backward-pass heuristic: transpose→dot patterns (weight gradients) and
    /// reductions (gradient accumulation) are biased toward GPU for FP32 precision.
    private func trainingBiasedDevice(
        _ opAnalysis: ANEOperationAnalysis,
        opMap: [String: HLOOperation]
    ) -> ExecutionDevice {
        if opAnalysis.compatibility == .incompatible {
            return .gpu
        }

        // Forward-pass ops: bias toward ANE (0.7x ANE time = prefer ANE)
        let forwardOps: Set<HLOOpKind> = [
            .convolution, .dot, .dotGeneral,
            .tanh, .logistic, .exponential, .sqrt, .rsqrt
        ]

        // Backward-pass indicators: bias toward GPU (1.5x ANE time = prefer GPU)
        let backwardOps: Set<HLOOpKind> = [.reduce]

        let aneTime = opAnalysis.estimatedANETime
        let gpuTime = opAnalysis.estimatedGPUTime

        if forwardOps.contains(opAnalysis.operationType) {
            // Lower ANE cost for forward ops
            return (aneTime * 0.7) < gpuTime ? .ane : .gpu
        } else if backwardOps.contains(opAnalysis.operationType) {
            // Raise ANE cost for backward-indicator ops
            return (aneTime * 1.5) < gpuTime ? .ane : .gpu
        }

        return opAnalysis.preferredDevice
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
                outputTensors: outputTensors,
                workingSetBytes: partition.workingSetBytes
            )
        }
    }

    /// Builds a partition-level dependency DAG and groups partitions into
    /// concurrency levels using Kahn's algorithm.
    ///
    /// Partitions within the same level have no mutual dependencies and
    /// can execute in parallel (e.g., a GPU partition and an ANE partition
    /// that don't share any tensors).
    private func buildConcurrencyLevels(
        _ partitions: [DevicePartition],
        dependencies: [String: Set<String>]
    ) -> (executionOrder: [Int], levels: [[Int]]) {
        let n = partitions.count
        guard n > 0 else { return ([], []) }

        // Build op → partition index mapping
        var opToPartition: [String: Int] = [:]
        for (i, partition) in partitions.enumerated() {
            for opId in partition.operationIds {
                opToPartition[opId] = i
            }
        }

        // Build partition-level predecessor sets
        var predecessors: [Set<Int>] = Array(repeating: Set(), count: n)
        for (i, partition) in partitions.enumerated() {
            for opId in partition.operationIds {
                if let deps = dependencies[opId] {
                    for dep in deps {
                        if let depPartition = opToPartition[dep], depPartition != i {
                            predecessors[i].insert(depPartition)
                        }
                    }
                }
            }
        }

        // Kahn's algorithm: group partitions by concurrency level
        var inDegree = predecessors.map { $0.count }
        var resolved = Set<Int>()
        var levels: [[Int]] = []

        while resolved.count < n {
            // Find all partitions with zero unresolved predecessors
            var level: [Int] = []
            for i in 0..<n {
                if !resolved.contains(i) && inDegree[i] == 0 {
                    level.append(i)
                }
            }

            if level.isEmpty {
                // Cycle detected (shouldn't happen in valid HLO) — fall back to sequential
                let remaining = (0..<n).filter { !resolved.contains($0) }
                levels.append(remaining)
                break
            }

            levels.append(level)
            for i in level {
                resolved.insert(i)
                // Decrement in-degree of successors
                for j in 0..<n {
                    if predecessors[j].contains(i) {
                        inDegree[j] -= 1
                    }
                }
            }
        }

        let executionOrder = levels.flatMap { $0 }
        return (executionOrder, levels)
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

    // MARK: - Memory Planning

    /// Calculates the peak working set (in bytes) for a partition.
    ///
    /// Simulates execution: tracks live tensors (add output on produce,
    /// remove after last use), records the peak.
    private func calculateWorkingSet(
        partition: DevicePartition,
        function: HLOFunction
    ) -> Int {
        let opMap = Dictionary(uniqueKeysWithValues: function.operations.map { ($0.result, $0) })
        let partitionOpSet = Set(partition.operationIds)
        let partitionOps = function.operations.filter { partitionOpSet.contains($0.result) }

        guard !partitionOps.isEmpty else { return 0 }

        // Build last-use map: for each tensor, the index of the last op in this partition that uses it
        var lastUse: [String: Int] = [:]
        for (idx, op) in partitionOps.enumerated() {
            for operand in op.operands {
                lastUse[operand] = idx
            }
        }

        var liveBytes = 0
        var peakBytes = 0
        var liveTensors: [String: Int] = [:]  // tensorId -> byteSize

        for (idx, op) in partitionOps.enumerated() {
            // Add output tensor
            let outputBytes = op.resultType.byteCount
            liveTensors[op.result] = outputBytes
            liveBytes += outputBytes
            peakBytes = max(peakBytes, liveBytes)

            // Remove dead tensors (those whose last use was at or before this index)
            for (tensorId, bytes) in liveTensors {
                if let lastIdx = lastUse[tensorId], lastIdx <= idx, tensorId != op.result {
                    liveBytes -= bytes
                    liveTensors.removeValue(forKey: tensorId)
                }
            }
        }

        return peakBytes
    }

    /// Splits ANE partitions that exceed the SRAM capacity threshold.
    ///
    /// For each oversized ANE partition, finds a split point where the
    /// working set is manageable and creates two sub-partitions.
    /// Sub-partitions below `minOpsForANEPartition` are moved to GPU.
    private func splitOversizedPartitions(
        _ partitions: [DevicePartition],
        function: HLOFunction
    ) -> [DevicePartition] {
        var result: [DevicePartition] = []

        for partition in partitions {
            let workingSet = calculateWorkingSet(partition: partition, function: function)

            guard partition.device == .ane else {
                // Non-ANE partitions: just annotate with working set
                result.append(DevicePartition(
                    device: partition.device,
                    operationIds: partition.operationIds,
                    estimatedTimeMs: partition.estimatedTimeMs,
                    inputTensors: partition.inputTensors,
                    outputTensors: partition.outputTensors,
                    workingSetBytes: workingSet
                ))
                continue
            }

            if workingSet <= config.aneSRAMCapacityBytes || partition.operationIds.count <= 1 {
                // Within SRAM budget or can't split further
                result.append(DevicePartition(
                    device: partition.device,
                    operationIds: partition.operationIds,
                    estimatedTimeMs: partition.estimatedTimeMs,
                    inputTensors: partition.inputTensors,
                    outputTensors: partition.outputTensors,
                    workingSetBytes: workingSet
                ))
                continue
            }

            // Split at midpoint
            let mid = partition.operationIds.count / 2
            let firstOps = Array(partition.operationIds.prefix(mid))
            let secondOps = Array(partition.operationIds.suffix(from: mid))
            let halfTime = partition.estimatedTimeMs / 2.0

            let firstPartition = DevicePartition(
                device: firstOps.count >= config.minOpsForANEPartition ? .ane : .gpu,
                operationIds: firstOps,
                estimatedTimeMs: halfTime,
                inputTensors: partition.inputTensors,
                outputTensors: Set()
            )
            let secondPartition = DevicePartition(
                device: secondOps.count >= config.minOpsForANEPartition ? .ane : .gpu,
                operationIds: secondOps,
                estimatedTimeMs: halfTime,
                inputTensors: Set(),
                outputTensors: partition.outputTensors
            )

            // Recursively split if still too large
            let splitFirst = splitOversizedPartitions([firstPartition], function: function)
            let splitSecond = splitOversizedPartitions([secondPartition], function: function)

            result.append(contentsOf: splitFirst)
            result.append(contentsOf: splitSecond)
        }

        return result
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

        /// When true, biases the cost model to route forward-pass ops (conv, matmul,
        /// activations) toward ANE and backward-pass ops (gradient accumulation,
        /// high-precision reductions) toward GPU. Also enables weight template
        /// caching to avoid hitting the ~115 ANE compilation limit.
        public var trainingMode: Bool

        public static let `default` = Config(
            enableANE: true,
            enableGPU: true,
            enableCPUFallback: true,
            enablePipelining: false,
            trainingMode: false
        )

        public init(
            enableANE: Bool,
            enableGPU: Bool,
            enableCPUFallback: Bool,
            enablePipelining: Bool = false,
            trainingMode: Bool = false
        ) {
            self.enableANE = enableANE
            self.enableGPU = enableGPU
            self.enableCPUFallback = enableCPUFallback
            self.enablePipelining = enablePipelining
            self.trainingMode = trainingMode
        }
    }

    private let config: Config
    private let partitioner: GPUANEPartitioner
    private let metalExecutor: MetalExecutor
    private let coreMLBridge: ANERuntime.CoreMLBridge
    private var transferManager: TensorTransferManager
    private var stats: ExecutionStats
    private let lock = NSLock()

    /// Dedicated queue for ANE (CoreML) execution in pipelined/concurrent mode.
    private let aneQueue = DispatchQueue(label: "com.metalHLO.ane", qos: .userInitiated)

    /// Dedicated queue for GPU (MPSGraph) execution in concurrent mode.
    private let gpuQueue = DispatchQueue(label: "com.metalHLO.gpu", qos: .userInitiated)

    /// Cache of compiled CoreML programs keyed by partition op hash.
    private var aneCache: [String: ANERuntime.CoreMLProgram] = [:]

    /// Cache of weight templates for training weight swapping without recompilation.
    private var aneTemplateCache: [String: ANERuntime.MILWeightTemplate] = [:]

    /// Number of consecutive ANE failures (reset on success).
    private var consecutiveANEFailures: Int = 0

    /// Set to true after too many consecutive ANE failures.
    private var aneDisabled: Bool = false

    /// Maximum consecutive ANE failures before disabling ANE for this executor.
    private let maxConsecutiveANEFailures: Int = 3

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

        if config.enablePipelining && plan.hasConcurrentLevels {
            partitionResults = try executeConcurrent(plan: plan, function: function, stats: &localStats)
        } else if config.enablePipelining && plan.canPipeline {
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

    /// Executes partitions concurrently using dependency-aware scheduling.
    ///
    /// Uses the partition plan's concurrency levels: partitions within the same
    /// level have no mutual data dependencies and are dispatched to their
    /// respective device queues (GPU or ANE) simultaneously. Between levels,
    /// a `DispatchGroup.wait()` barrier ensures all outputs are ready.
    private func executeConcurrent(
        plan: GPUANEPartitioner.PartitionPlan,
        function: HLOFunction,
        stats localStats: inout ExecutionStats
    ) throws -> [PartitionResult] {
        final class ConcurrentState: @unchecked Sendable {
            let lock = NSLock()
            var results: [PartitionResult?]
            var error: Error?

            init(count: Int) {
                self.results = Array(repeating: nil, count: count)
            }
        }
        let state = ConcurrentState(count: plan.partitions.count)

        for level in plan.concurrencyLevels {
            if level.count == 1 {
                // Single partition — execute inline, no dispatch overhead
                let partitionIndex = level[0]
                let partition = plan.partitions[partitionIndex]
                let result = try executePartitionWithFallback(partition, function: function)
                state.results[partitionIndex] = result
            } else {
                // Multiple independent partitions — dispatch concurrently
                let group = DispatchGroup()

                for partitionIndex in level {
                    let partition = plan.partitions[partitionIndex]
                    let queue = (partition.device == .ane) ? aneQueue : gpuQueue

                    group.enter()
                    queue.async { [self] in
                        defer { group.leave() }
                        do {
                            let result = try self.executePartitionWithFallback(
                                partition, function: function
                            )
                            state.lock.lock()
                            state.results[partitionIndex] = result
                            state.lock.unlock()
                        } catch {
                            state.lock.lock()
                            if state.error == nil { state.error = error }
                            state.lock.unlock()
                        }
                    }
                }

                group.wait()
                if let error = state.error { throw error }
            }
        }

        // Collect results in execution order and update stats
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
    ///
    /// Tracks consecutive ANE failures. After `maxConsecutiveANEFailures`
    /// consecutive failures, ANE is disabled for the lifetime of this executor
    /// and all ANE partitions are automatically routed to GPU.
    private func executePartitionWithFallback(
        _ partition: GPUANEPartitioner.DevicePartition,
        function: HLOFunction
    ) throws -> PartitionResult {
        // If ANE is disabled due to repeated failures, route directly to GPU
        if partition.device == .ane && aneDisabled {
            let gpuPartition = GPUANEPartitioner.DevicePartition(
                device: .gpu,
                operationIds: partition.operationIds,
                estimatedTimeMs: partition.estimatedTimeMs,
                inputTensors: partition.inputTensors,
                outputTensors: partition.outputTensors
            )
            return try executePartition(gpuPartition, function: function)
        }

        do {
            let result = try executePartition(partition, function: function)
            // Reset failure counter on ANE success (thread-safe for concurrent execution)
            if partition.device == .ane {
                lock.lock()
                consecutiveANEFailures = 0
                lock.unlock()
            }
            return result
        } catch {
            if partition.device == .ane && config.enableCPUFallback {
                lock.lock()
                consecutiveANEFailures += 1
                if consecutiveANEFailures >= maxConsecutiveANEFailures {
                    aneDisabled = true
                }
                lock.unlock()
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

        // Detect patterns for fusion
        let analyzer = Analyzer()
        var shapes: [TensorID: [Int]] = [:]
        for arg in subFunction.inputs {
            shapes[arg.name] = arg.type.shape
        }
        for op in subFunction.operations {
            shapes[op.result] = op.resultType.shape
        }
        let patterns = analyzer.detectPatterns(subFunction, shapes: shapes)

        // Use fusion-aware build if patterns found, otherwise standard build
        let coremlInputs: [(name: String, shape: [Int])]
        let ops: [CoreMLOp]
        let returnVar: String
        if !patterns.isEmpty {
            (coremlInputs, ops, returnVar) = try builder.buildWithFusion(
                function: subFunction, patterns: patterns
            )
        } else {
            (coremlInputs, ops, returnVar) = try builder.build(function: subFunction)
        }

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

        // Try zero-copy path via shared MLMultiArray inputs
        var sharedInputs: [(name: String, array: CoreML.MLMultiArray)] = []
        var useSharedPath = true

        for (coremlInput, arg) in zip(coremlInputs, subFunction.inputs) {
            if let shared = try transferManager.getSharedMLMultiArray(name: arg.name) {
                sharedInputs.append((name: coremlInput.name, array: shared.array))
            } else {
                useSharedPath = false
                break
            }
        }

        // Execute via CoreML — zero-copy or fallback to array copy
        let results: [Float]
        if useSharedPath {
            results = try coreMLBridge.execute(program, multiArrayInputs: sharedInputs)
        } else {
            var aneInputs: [(name: String, data: [Float], shape: [Int])] = []
            for (coremlInput, arg) in zip(coremlInputs, subFunction.inputs) {
                let cpuData = try transferManager.getCPUArray(name: arg.name)
                aneInputs.append((name: coremlInput.name, data: cpuData.data, shape: cpuData.shape))
            }
            results = try coreMLBridge.execute(program, inputs: aneInputs)
        }

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

    /// Clears cached ANE compilations and weight templates.
    public func clearCache() {
        aneCache.removeAll()
        aneTemplateCache.removeAll()
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
