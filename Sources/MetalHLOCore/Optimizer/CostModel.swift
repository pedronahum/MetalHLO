// CostModel.swift
// MetalHLOCore
//
// Cost model for intelligent fusion decisions.
// Inspired by XLA's GPU performance model.

import Foundation
import Metal

// MARK: - Device Capabilities

/// Capabilities of a Metal GPU device.
///
/// These values affect performance estimation for different operations.
/// Values are approximations for Apple Silicon GPUs.
public struct DeviceCapabilities: Sendable {
    /// Number of GPU compute units.
    public let computeUnits: Int

    /// Maximum threads per threadgroup.
    public let maxThreadsPerThreadgroup: Int

    /// SIMD width (threads per SIMD group).
    public let simdWidth: Int

    /// Memory bandwidth in GB/s.
    public let memoryBandwidthGBps: Double

    /// Peak compute throughput in TFLOPs (FP32).
    public let peakTFlops: Double

    /// Shared memory per threadgroup in bytes.
    public let sharedMemoryPerThreadgroup: Int

    /// Maximum registers per thread (approximate).
    public let maxRegistersPerThread: Int

    /// Kernel launch overhead in microseconds.
    public let kernelLaunchOverheadUs: Double

    public init(
        computeUnits: Int,
        maxThreadsPerThreadgroup: Int,
        simdWidth: Int,
        memoryBandwidthGBps: Double,
        peakTFlops: Double,
        sharedMemoryPerThreadgroup: Int,
        maxRegistersPerThread: Int = 128,
        kernelLaunchOverheadUs: Double = 5.0
    ) {
        self.computeUnits = computeUnits
        self.maxThreadsPerThreadgroup = maxThreadsPerThreadgroup
        self.simdWidth = simdWidth
        self.memoryBandwidthGBps = memoryBandwidthGBps
        self.peakTFlops = peakTFlops
        self.sharedMemoryPerThreadgroup = sharedMemoryPerThreadgroup
        self.maxRegistersPerThread = maxRegistersPerThread
        self.kernelLaunchOverheadUs = kernelLaunchOverheadUs
    }

    /// Default capabilities for Apple Silicon M1.
    public static let m1 = DeviceCapabilities(
        computeUnits: 8,
        maxThreadsPerThreadgroup: 1024,
        simdWidth: 32,
        memoryBandwidthGBps: 68.25,
        peakTFlops: 2.6,
        sharedMemoryPerThreadgroup: 32768
    )

    /// Default capabilities for Apple Silicon M1 Pro.
    public static let m1Pro = DeviceCapabilities(
        computeUnits: 16,
        maxThreadsPerThreadgroup: 1024,
        simdWidth: 32,
        memoryBandwidthGBps: 200.0,
        peakTFlops: 5.2,
        sharedMemoryPerThreadgroup: 32768
    )

    /// Default capabilities for Apple Silicon M1 Max.
    public static let m1Max = DeviceCapabilities(
        computeUnits: 32,
        maxThreadsPerThreadgroup: 1024,
        simdWidth: 32,
        memoryBandwidthGBps: 400.0,
        peakTFlops: 10.4,
        sharedMemoryPerThreadgroup: 32768
    )

    /// Default capabilities for Apple Silicon M2.
    public static let m2 = DeviceCapabilities(
        computeUnits: 10,
        maxThreadsPerThreadgroup: 1024,
        simdWidth: 32,
        memoryBandwidthGBps: 100.0,
        peakTFlops: 3.6,
        sharedMemoryPerThreadgroup: 32768
    )

    /// Default capabilities for Apple Silicon M3.
    public static let m3 = DeviceCapabilities(
        computeUnits: 10,
        maxThreadsPerThreadgroup: 1024,
        simdWidth: 32,
        memoryBandwidthGBps: 100.0,
        peakTFlops: 4.1,
        sharedMemoryPerThreadgroup: 32768
    )

    /// Default capabilities (conservative estimates).
    public static let `default` = m1

    /// Creates capabilities by querying a Metal device.
    public static func fromDevice(_ device: MTLDevice) -> DeviceCapabilities {
        // Query actual device capabilities where possible
        let maxThreads = device.maxThreadsPerThreadgroup.width

        // Estimate based on device name
        let name = device.name.lowercased()

        if name.contains("m3 max") || name.contains("m3max") {
            return DeviceCapabilities(
                computeUnits: 40,
                maxThreadsPerThreadgroup: maxThreads,
                simdWidth: 32,
                memoryBandwidthGBps: 400.0,
                peakTFlops: 14.2,
                sharedMemoryPerThreadgroup: 32768
            )
        } else if name.contains("m3 pro") || name.contains("m3pro") {
            return DeviceCapabilities(
                computeUnits: 18,
                maxThreadsPerThreadgroup: maxThreads,
                simdWidth: 32,
                memoryBandwidthGBps: 150.0,
                peakTFlops: 7.4,
                sharedMemoryPerThreadgroup: 32768
            )
        } else if name.contains("m3") {
            return m3
        } else if name.contains("m2 max") || name.contains("m2max") {
            return DeviceCapabilities(
                computeUnits: 38,
                maxThreadsPerThreadgroup: maxThreads,
                simdWidth: 32,
                memoryBandwidthGBps: 400.0,
                peakTFlops: 13.6,
                sharedMemoryPerThreadgroup: 32768
            )
        } else if name.contains("m2 pro") || name.contains("m2pro") {
            return DeviceCapabilities(
                computeUnits: 19,
                maxThreadsPerThreadgroup: maxThreads,
                simdWidth: 32,
                memoryBandwidthGBps: 200.0,
                peakTFlops: 6.8,
                sharedMemoryPerThreadgroup: 32768
            )
        } else if name.contains("m2") {
            return m2
        } else if name.contains("m1 max") || name.contains("m1max") {
            return m1Max
        } else if name.contains("m1 pro") || name.contains("m1pro") {
            return m1Pro
        } else if name.contains("m1") {
            return m1
        }

        // Default conservative estimate
        return .default
    }
}

// MARK: - Operation Cost

/// Estimated cost of executing an operation.
public struct OperationCost: Sendable {
    /// Estimated compute time in microseconds.
    public let computeTimeUs: Double

    /// Estimated memory transfer time in microseconds.
    public let memoryTimeUs: Double

    /// Total estimated time in microseconds.
    public var totalTimeUs: Double {
        max(computeTimeUs, memoryTimeUs)
    }

    /// Number of floating-point operations.
    public let flops: Int

    /// Number of bytes read.
    public let bytesRead: Int

    /// Number of bytes written.
    public let bytesWritten: Int

    /// Arithmetic intensity (FLOPs per byte).
    public var arithmeticIntensity: Double {
        let totalBytes = bytesRead + bytesWritten
        guard totalBytes > 0 else { return 0 }
        return Double(flops) / Double(totalBytes)
    }

    /// Whether this operation is compute-bound (vs memory-bound).
    public var isComputeBound: Bool {
        computeTimeUs > memoryTimeUs
    }

    public init(
        computeTimeUs: Double,
        memoryTimeUs: Double,
        flops: Int,
        bytesRead: Int,
        bytesWritten: Int
    ) {
        self.computeTimeUs = computeTimeUs
        self.memoryTimeUs = memoryTimeUs
        self.flops = flops
        self.bytesRead = bytesRead
        self.bytesWritten = bytesWritten
    }

    /// Combines two costs (for sequential execution).
    public func combined(with other: OperationCost) -> OperationCost {
        return OperationCost(
            computeTimeUs: computeTimeUs + other.computeTimeUs,
            memoryTimeUs: memoryTimeUs + other.memoryTimeUs,
            flops: flops + other.flops,
            bytesRead: bytesRead + other.bytesRead,
            bytesWritten: bytesWritten + other.bytesWritten
        )
    }
}

// MARK: - GPU Performance Model

/// Performance model for estimating operation execution time on Metal GPUs.
///
/// The model estimates execution time based on:
/// - Compute time: FLOPs / peak throughput
/// - Memory time: Bytes transferred / bandwidth
/// - Actual time is max(compute, memory) for memory-bound vs compute-bound ops
public final class MetalPerformanceModel: @unchecked Sendable {

    /// Device capabilities used for estimation.
    public let device: DeviceCapabilities

    /// Creates a performance model for the given device.
    public init(device: DeviceCapabilities = .default) {
        self.device = device
    }

    /// Estimates the cost of executing an operation.
    ///
    /// - Parameter op: The operation to estimate.
    /// - Returns: The estimated cost.
    public func estimateCost(_ op: HLOOperation) -> OperationCost {
        switch op.kind {
        case .dot, .dotGeneral:
            return estimateMatMulCost(op)

        case .convolution:
            return estimateConvolutionCost(op)

        case .reduce, .reduceWindow:
            return estimateReductionCost(op)

        case .gather, .scatter, .dynamicGather:
            return estimateGatherScatterCost(op)

        default:
            // Elementwise operations
            if op.kind.isUnary || op.kind.isBinaryArithmetic {
                return estimateElementwiseCost(op)
            }
            // Default: assume elementwise-like cost
            return estimateElementwiseCost(op)
        }
    }

    /// Estimates the time to transfer a tensor to/from memory.
    ///
    /// - Parameters:
    ///   - byteCount: Size of the tensor in bytes.
    ///   - isWrite: Whether this is a write (vs read).
    /// - Returns: Estimated time in microseconds.
    public func estimateMemoryTransferTime(byteCount: Int, isWrite: Bool = false) -> Double {
        // Memory bandwidth in bytes per microsecond
        let bandwidthBytesPerUs = device.memoryBandwidthGBps * 1e9 / 1e6
        return Double(byteCount) / bandwidthBytesPerUs
    }

    // MARK: - Matrix Multiplication

    private func estimateMatMulCost(_ op: HLOOperation) -> OperationCost {
        let resultShape = op.resultType.shape

        // For dot_general, we need to figure out M, N, K from dimension numbers
        var m = 1, n = 1, k = 1

        if let dimNums = op.attributes.dotDimensionNumbers {
            // Simplified: assume 2D or batched 2D matmul
            // LHS: [..., M, K], RHS: [..., K, N]
            // Note: Contracting dimensions tell us which dims are summed over (K)
            // but we estimate K separately since we don't have operand shapes
            _ = dimNums.lhsContractingDimensions  // Used for dimension analysis
            _ = dimNums.rhsContractingDimensions  // Used for dimension analysis

            // Get shapes from operands if we had them, but we only have result
            // Estimate based on result shape
            if resultShape.count >= 2 {
                m = resultShape[resultShape.count - 2]
                n = resultShape[resultShape.count - 1]
            } else if resultShape.count == 1 {
                n = resultShape[0]
                m = 1
            }

            // Estimate K from a reasonable assumption
            k = 256  // Default assumption if we can't determine K
        } else {
            // Simple dot product
            if resultShape.count >= 2 {
                m = resultShape[0]
                n = resultShape[1]
                k = 256  // Assume
            }
        }

        // Batch size
        let batchSize = resultShape.dropLast(2).reduce(1, *)

        // FLOPs for matmul: 2 * M * N * K * batch
        let flops = 2 * m * n * k * batchSize

        // Bytes: read LHS (M*K), RHS (K*N), write output (M*N), times batch and element size
        let elementSize = op.resultType.elementType.byteSize
        let bytesRead = (m * k + k * n) * batchSize * elementSize
        let bytesWritten = m * n * batchSize * elementSize

        // Compute time
        let computeTimeUs = Double(flops) / (device.peakTFlops * 1e12 / 1e6)

        // Memory time
        let memoryTimeUs = Double(bytesRead + bytesWritten) / (device.memoryBandwidthGBps * 1e9 / 1e6)

        return OperationCost(
            computeTimeUs: computeTimeUs,
            memoryTimeUs: memoryTimeUs,
            flops: flops,
            bytesRead: bytesRead,
            bytesWritten: bytesWritten
        )
    }

    // MARK: - Convolution

    private func estimateConvolutionCost(_ op: HLOOperation) -> OperationCost {
        let outputShape = op.resultType.shape

        // Assume standard 2D convolution: [N, H, W, C_out] or [N, C_out, H, W]
        // FLOPs = 2 * N * H_out * W_out * C_out * C_in * K_h * K_w

        var n = 1, hOut = 1, wOut = 1, cOut = 1
        var cIn = 1, kH = 3, kW = 3  // Default kernel size

        if let dimNums = op.attributes.convolutionDimensionNumbers {
            n = outputShape[dimNums.outputBatchDimension]
            cOut = outputShape[dimNums.outputFeatureDimension]

            if dimNums.outputSpatialDimensions.count >= 2 {
                hOut = outputShape[dimNums.outputSpatialDimensions[0]]
                wOut = outputShape[dimNums.outputSpatialDimensions[1]]
            }

            // Estimate input channels and kernel size
            cIn = cOut / 2  // Rough estimate
            kH = 3
            kW = 3
        } else if outputShape.count == 4 {
            n = outputShape[0]
            hOut = outputShape[1]
            wOut = outputShape[2]
            cOut = outputShape[3]
        }

        let flops = 2 * n * hOut * wOut * cOut * cIn * kH * kW

        let elementSize = op.resultType.elementType.byteSize
        let inputBytes = n * (hOut + kH - 1) * (wOut + kW - 1) * cIn * elementSize
        let kernelBytes = kH * kW * cIn * cOut * elementSize
        let outputBytes = n * hOut * wOut * cOut * elementSize

        let bytesRead = inputBytes + kernelBytes
        let bytesWritten = outputBytes

        let computeTimeUs = Double(flops) / (device.peakTFlops * 1e12 / 1e6)
        let memoryTimeUs = Double(bytesRead + bytesWritten) / (device.memoryBandwidthGBps * 1e9 / 1e6)

        return OperationCost(
            computeTimeUs: computeTimeUs,
            memoryTimeUs: memoryTimeUs,
            flops: flops,
            bytesRead: bytesRead,
            bytesWritten: bytesWritten
        )
    }

    // MARK: - Reduction

    private func estimateReductionCost(_ op: HLOOperation) -> OperationCost {
        let outputElements = op.resultType.count
        let reduceDims = op.attributes.dimensions ?? []

        // Estimate input size from output and reduced dimensions
        var inputElements = outputElements
        for _ in reduceDims {
            inputElements *= 64  // Assume each reduced dimension was size 64
        }

        // FLOPs: one op per input element
        let flops = inputElements

        let elementSize = op.resultType.elementType.byteSize
        let bytesRead = inputElements * elementSize
        let bytesWritten = outputElements * elementSize

        let computeTimeUs = Double(flops) / (device.peakTFlops * 1e12 / 1e6)
        let memoryTimeUs = Double(bytesRead + bytesWritten) / (device.memoryBandwidthGBps * 1e9 / 1e6)

        return OperationCost(
            computeTimeUs: computeTimeUs,
            memoryTimeUs: memoryTimeUs,
            flops: flops,
            bytesRead: bytesRead,
            bytesWritten: bytesWritten
        )
    }

    // MARK: - Gather/Scatter

    private func estimateGatherScatterCost(_ op: HLOOperation) -> OperationCost {
        let outputElements = op.resultType.count
        let elementSize = op.resultType.elementType.byteSize

        // Gather/scatter have poor memory access patterns
        // Assume each element requires a separate memory transaction
        let bytesRead = outputElements * elementSize * 2  // Data + indices
        let bytesWritten = outputElements * elementSize

        // Very memory bound, minimal compute
        let flops = outputElements

        let computeTimeUs = Double(flops) / (device.peakTFlops * 1e12 / 1e6)
        // Worse memory efficiency due to random access
        let memoryTimeUs = Double(bytesRead + bytesWritten) / (device.memoryBandwidthGBps * 1e9 / 1e6) * 4.0

        return OperationCost(
            computeTimeUs: computeTimeUs,
            memoryTimeUs: memoryTimeUs,
            flops: flops,
            bytesRead: bytesRead,
            bytesWritten: bytesWritten
        )
    }

    // MARK: - Elementwise

    private func estimateElementwiseCost(_ op: HLOOperation) -> OperationCost {
        let elements = op.resultType.count
        let elementSize = op.resultType.elementType.byteSize

        // FLOPs depends on operation type
        let flopsPerElement: Int
        switch op.kind {
        case .add, .subtract, .multiply, .divide, .maximum, .minimum:
            flopsPerElement = 1
        case .exponential, .log, .tanh, .logistic:
            flopsPerElement = 10  // Transcendentals are more expensive
        case .sqrt, .rsqrt:
            flopsPerElement = 5
        case .power:
            flopsPerElement = 15
        default:
            flopsPerElement = 1
        }

        let flops = elements * flopsPerElement

        // Bytes: read inputs + write output
        let numInputs = op.operands.count
        let bytesRead = elements * elementSize * max(1, numInputs)
        let bytesWritten = elements * elementSize

        let computeTimeUs = Double(flops) / (device.peakTFlops * 1e12 / 1e6)
        let memoryTimeUs = Double(bytesRead + bytesWritten) / (device.memoryBandwidthGBps * 1e9 / 1e6)

        return OperationCost(
            computeTimeUs: computeTimeUs,
            memoryTimeUs: memoryTimeUs,
            flops: flops,
            bytesRead: bytesRead,
            bytesWritten: bytesWritten
        )
    }
}

// MARK: - Fusion Decision

/// Result of a fusion decision.
public struct FusionDecision: Sendable {
    /// Whether fusion is recommended.
    public let shouldFuse: Bool

    /// Reason for the decision.
    public let reason: String

    /// Estimated speedup from fusion (1.0 = no change).
    public let estimatedSpeedup: Double

    /// Estimated time without fusion (microseconds).
    public let unfusedTimeUs: Double

    /// Estimated time with fusion (microseconds).
    public let fusedTimeUs: Double

    public init(
        shouldFuse: Bool,
        reason: String,
        estimatedSpeedup: Double,
        unfusedTimeUs: Double,
        fusedTimeUs: Double
    ) {
        self.shouldFuse = shouldFuse
        self.reason = reason
        self.estimatedSpeedup = estimatedSpeedup
        self.unfusedTimeUs = unfusedTimeUs
        self.fusedTimeUs = fusedTimeUs
    }
}

// MARK: - Fusion Heuristics

/// Heuristics for fusion decisions.
///
/// Determines when fusing operations is beneficial based on:
/// - Estimated execution time (fused vs unfused)
/// - Memory transfer elimination
/// - Register pressure constraints
/// - Code growth limits
public final class FusionHeuristics: @unchecked Sendable {

    /// The performance model for cost estimation.
    private let performanceModel: MetalPerformanceModel

    /// Maximum number of operations in a fusion.
    public let maxFusionSize: Int

    /// Maximum estimated register pressure.
    public let maxRegisterPressure: Int

    /// Operations that should NOT be fused.
    public let unfusibleOps: Set<HLOOpKind>

    /// Creates fusion heuristics with the given parameters.
    public init(
        performanceModel: MetalPerformanceModel = MetalPerformanceModel(),
        maxFusionSize: Int = 50,
        maxRegisterPressure: Int = 128
    ) {
        self.performanceModel = performanceModel
        self.maxFusionSize = maxFusionSize
        self.maxRegisterPressure = maxRegisterPressure

        // Operations that should NOT be fused (expensive, use libraries)
        self.unfusibleOps = [
            .convolution,      // Use MPS CNN kernels
            .fft,              // Use vDSP
            .sort,             // Complex control flow
            .scatter,          // Non-uniform memory access
            .gather,           // Non-uniform memory access (large)
            .dynamicGather,    // Non-uniform memory access
            .triangularSolve,  // Library call
            .cholesky,         // Library call
        ]
    }

    /// Decides whether to fuse a producer into a consumer.
    ///
    /// - Parameters:
    ///   - producer: The producer operation.
    ///   - consumer: The consumer operation.
    /// - Returns: The fusion decision.
    public func shouldFuse(producer: HLOOperation, consumer: HLOOperation) -> FusionDecision {
        // Check if operations are fusible
        if unfusibleOps.contains(producer.kind) {
            return FusionDecision(
                shouldFuse: false,
                reason: "Producer \(producer.kind) is unfusible (library call)",
                estimatedSpeedup: 1.0,
                unfusedTimeUs: 0,
                fusedTimeUs: 0
            )
        }

        if unfusibleOps.contains(consumer.kind) {
            return FusionDecision(
                shouldFuse: false,
                reason: "Consumer \(consumer.kind) is unfusible (library call)",
                estimatedSpeedup: 1.0,
                unfusedTimeUs: 0,
                fusedTimeUs: 0
            )
        }

        // Check register pressure
        let combinedPressure = estimateRegisterPressure([producer, consumer])
        if combinedPressure > maxRegisterPressure {
            return FusionDecision(
                shouldFuse: false,
                reason: "Would exceed register limit (\(combinedPressure) > \(maxRegisterPressure))",
                estimatedSpeedup: 1.0,
                unfusedTimeUs: 0,
                fusedTimeUs: 0
            )
        }

        // Estimate costs
        let producerCost = performanceModel.estimateCost(producer)
        let consumerCost = performanceModel.estimateCost(consumer)

        // Time if NOT fused: producer + consumer + memory transfer between them
        let intermediateBytes = producer.resultType.byteCount
        let transferTime = performanceModel.estimateMemoryTransferTime(byteCount: intermediateBytes) * 2  // Write + read
        let unfusedTime = producerCost.totalTimeUs + consumerCost.totalTimeUs + transferTime

        // Time if fused: no intermediate memory transfer
        // Compute is same, but memory is reduced
        let fusedComputeTime = producerCost.computeTimeUs + consumerCost.computeTimeUs
        let fusedMemoryTime = max(0, producerCost.memoryTimeUs + consumerCost.memoryTimeUs - transferTime)
        let fusedTime = max(fusedComputeTime, fusedMemoryTime)

        let speedup = unfusedTime / max(fusedTime, 0.001)

        // Fuse if there's a speedup
        if fusedTime < unfusedTime {
            return FusionDecision(
                shouldFuse: true,
                reason: "Fusion saves \(String(format: "%.1f", transferTime))µs memory transfer",
                estimatedSpeedup: speedup,
                unfusedTimeUs: unfusedTime,
                fusedTimeUs: fusedTime
            )
        } else {
            return FusionDecision(
                shouldFuse: false,
                reason: "No benefit from fusion",
                estimatedSpeedup: speedup,
                unfusedTimeUs: unfusedTime,
                fusedTimeUs: fusedTime
            )
        }
    }

    /// Estimates the register pressure for a set of operations.
    ///
    /// - Parameter ops: The operations to analyze.
    /// - Returns: Estimated number of registers needed.
    public func estimateRegisterPressure(_ ops: [HLOOperation]) -> Int {
        var maxLive = 0
        var currentLive = 0

        for op in ops {
            // Each output needs registers
            let outputRegs = registersForType(op.resultType)
            currentLive += outputRegs
            maxLive = max(maxLive, currentLive)

            // Assume inputs that are last-used free their registers
            // (Simplified: just track output growth)
        }

        return maxLive
    }

    /// Estimates registers needed for a tensor type.
    private func registersForType(_ type: TensorType) -> Int {
        // Each element in flight needs a register
        // For vectorized operations, we process multiple elements
        let elementsPerThread = 4  // Typical vectorization
        let elementSize = type.elementType.byteSize
        let registersPerElement = max(1, elementSize / 4)  // 4 bytes per register
        return elementsPerThread * registersPerElement
    }

    /// Checks if an operation is cheap enough to duplicate.
    ///
    /// Cheap operations can be duplicated when they have multiple uses,
    /// allowing their consumers to be fused separately.
    public func isCheapToDuplicate(_ op: HLOOperation) -> Bool {
        switch op.kind {
        // Zero-cost shape operations
        case .reshape, .broadcastInDim, .transpose:
            return true

        // Constants
        case .constant:
            return true

        // Cheap unary ops (1 FLOP per element)
        case .negate, .abs, .not:
            return true

        default:
            return false
        }
    }

    /// Checks if an operation is fusible (can be inlined into a kernel).
    public func isFusibleOp(_ kind: HLOOpKind) -> Bool {
        !unfusibleOps.contains(kind)
    }
}

// MARK: - Cost Model Statistics

/// Statistics about cost model predictions.
public struct CostModelStatistics: Sendable {
    /// Number of operations analyzed.
    public let numOperations: Int

    /// Total estimated compute time (µs).
    public let totalComputeTimeUs: Double

    /// Total estimated memory time (µs).
    public let totalMemoryTimeUs: Double

    /// Number of compute-bound operations.
    public let numComputeBound: Int

    /// Number of memory-bound operations.
    public let numMemoryBound: Int

    /// Total FLOPs.
    public let totalFlops: Int

    /// Total bytes transferred.
    public let totalBytes: Int

    /// Average arithmetic intensity.
    public var averageArithmeticIntensity: Double {
        guard totalBytes > 0 else { return 0 }
        return Double(totalFlops) / Double(totalBytes)
    }

    /// Fraction of operations that are memory-bound.
    public var memoryBoundFraction: Double {
        guard numOperations > 0 else { return 0 }
        return Double(numMemoryBound) / Double(numOperations)
    }

    public init(
        numOperations: Int,
        totalComputeTimeUs: Double,
        totalMemoryTimeUs: Double,
        numComputeBound: Int,
        numMemoryBound: Int,
        totalFlops: Int,
        totalBytes: Int
    ) {
        self.numOperations = numOperations
        self.totalComputeTimeUs = totalComputeTimeUs
        self.totalMemoryTimeUs = totalMemoryTimeUs
        self.numComputeBound = numComputeBound
        self.numMemoryBound = numMemoryBound
        self.totalFlops = totalFlops
        self.totalBytes = totalBytes
    }
}

extension MetalPerformanceModel {
    /// Analyzes a function and returns cost statistics.
    public func analyzeFunction(_ function: HLOFunction) -> CostModelStatistics {
        var totalCompute = 0.0
        var totalMemory = 0.0
        var computeBound = 0
        var memoryBound = 0
        var totalFlops = 0
        var totalBytes = 0

        for op in function.operations {
            let cost = estimateCost(op)

            totalCompute += cost.computeTimeUs
            totalMemory += cost.memoryTimeUs
            totalFlops += cost.flops
            totalBytes += cost.bytesRead + cost.bytesWritten

            if cost.isComputeBound {
                computeBound += 1
            } else {
                memoryBound += 1
            }
        }

        return CostModelStatistics(
            numOperations: function.operations.count,
            totalComputeTimeUs: totalCompute,
            totalMemoryTimeUs: totalMemory,
            numComputeBound: computeBound,
            numMemoryBound: memoryBound,
            totalFlops: totalFlops,
            totalBytes: totalBytes
        )
    }
}
