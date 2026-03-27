// GPUMetrics.swift
// MetalHLO Benchmarks
//
// GPU utilization and efficiency metrics for Apple Silicon.

import Foundation
import Metal

// MARK: - Apple Silicon Hardware Specifications

/// Theoretical peak performance for Apple Silicon chips.
public struct AppleSiliconSpecs: Sendable {
    public let name: String
    public let gpuCores: Int
    public let fp32TFLOPS: Double
    public let fp16TFLOPS: Double
    public let memoryBandwidthGBps: Double

    public init(name: String, gpuCores: Int, fp32TFLOPS: Double, fp16TFLOPS: Double, memoryBandwidthGBps: Double) {
        self.name = name
        self.gpuCores = gpuCores
        self.fp32TFLOPS = fp32TFLOPS
        self.fp16TFLOPS = fp16TFLOPS
        self.memoryBandwidthGBps = memoryBandwidthGBps
    }

    /// Known Apple Silicon specifications (from benchmark proposal Appendix A.1).
    public static let knownChips: [String: AppleSiliconSpecs] = [
        "M1": AppleSiliconSpecs(name: "M1", gpuCores: 8, fp32TFLOPS: 2.6, fp16TFLOPS: 5.2, memoryBandwidthGBps: 68.25),
        "M1 Pro": AppleSiliconSpecs(name: "M1 Pro", gpuCores: 16, fp32TFLOPS: 5.2, fp16TFLOPS: 10.4, memoryBandwidthGBps: 200),
        "M1 Max": AppleSiliconSpecs(name: "M1 Max", gpuCores: 32, fp32TFLOPS: 10.4, fp16TFLOPS: 20.8, memoryBandwidthGBps: 400),
        "M2": AppleSiliconSpecs(name: "M2", gpuCores: 10, fp32TFLOPS: 3.6, fp16TFLOPS: 7.2, memoryBandwidthGBps: 100),
        "M2 Pro": AppleSiliconSpecs(name: "M2 Pro", gpuCores: 19, fp32TFLOPS: 6.8, fp16TFLOPS: 13.6, memoryBandwidthGBps: 200),
        "M2 Max": AppleSiliconSpecs(name: "M2 Max", gpuCores: 38, fp32TFLOPS: 13.6, fp16TFLOPS: 27.2, memoryBandwidthGBps: 400),
        "M3": AppleSiliconSpecs(name: "M3", gpuCores: 10, fp32TFLOPS: 4.1, fp16TFLOPS: 8.2, memoryBandwidthGBps: 100),
        "M3 Pro": AppleSiliconSpecs(name: "M3 Pro", gpuCores: 18, fp32TFLOPS: 7.4, fp16TFLOPS: 14.8, memoryBandwidthGBps: 150),
        "M3 Max": AppleSiliconSpecs(name: "M3 Max", gpuCores: 40, fp32TFLOPS: 16.4, fp16TFLOPS: 32.8, memoryBandwidthGBps: 400),
        "M4": AppleSiliconSpecs(name: "M4", gpuCores: 10, fp32TFLOPS: 4.3, fp16TFLOPS: 8.6, memoryBandwidthGBps: 120),
        "M4 Pro": AppleSiliconSpecs(name: "M4 Pro", gpuCores: 20, fp32TFLOPS: 8.6, fp16TFLOPS: 17.2, memoryBandwidthGBps: 273),
        "M4 Max": AppleSiliconSpecs(name: "M4 Max", gpuCores: 40, fp32TFLOPS: 17.2, fp16TFLOPS: 34.4, memoryBandwidthGBps: 546),
    ]

    /// Detect current hardware specs based on Metal device name.
    public static func detect() -> AppleSiliconSpecs? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        let name = device.name

        // Try to match known chip
        for (chipName, specs) in knownChips {
            if name.contains(chipName) {
                return specs
            }
        }

        // Fallback: estimate based on device properties
        // This is approximate for unknown chips
        return AppleSiliconSpecs(
            name: name,
            gpuCores: 10,  // Conservative estimate
            fp32TFLOPS: 4.0,
            fp16TFLOPS: 8.0,
            memoryBandwidthGBps: 100
        )
    }
}

// MARK: - GPU Utilization Metrics

/// GPU utilization metrics result.
public struct GPUUtilizationMetrics: Sendable {
    /// Achieved FLOPS (floating-point operations per second).
    public let achievedFLOPS: Double
    /// Theoretical peak FLOPS for this hardware.
    public let peakFLOPS: Double
    /// Compute utilization percentage (achieved / peak * 100).
    public let computeUtilization: Double
    /// Achieved memory bandwidth in GB/s.
    public let achievedBandwidthGBps: Double
    /// Peak memory bandwidth for this hardware.
    public let peakBandwidthGBps: Double
    /// Memory bandwidth utilization percentage.
    public let bandwidthUtilization: Double
    /// Hardware specs used for calculation.
    public let hardwareSpecs: AppleSiliconSpecs

    public init(
        achievedFLOPS: Double,
        peakFLOPS: Double,
        computeUtilization: Double,
        achievedBandwidthGBps: Double,
        peakBandwidthGBps: Double,
        bandwidthUtilization: Double,
        hardwareSpecs: AppleSiliconSpecs
    ) {
        self.achievedFLOPS = achievedFLOPS
        self.peakFLOPS = peakFLOPS
        self.computeUtilization = computeUtilization
        self.achievedBandwidthGBps = achievedBandwidthGBps
        self.peakBandwidthGBps = peakBandwidthGBps
        self.bandwidthUtilization = bandwidthUtilization
        self.hardwareSpecs = hardwareSpecs
    }

    /// Format as human-readable string.
    public func formatted() -> String {
        let achievedTFLOPS = achievedFLOPS / 1e12
        let peakTFLOPS = peakFLOPS / 1e12
        return """
        Hardware: \(hardwareSpecs.name)
        Compute: \(String(format: "%.2f", achievedTFLOPS)) / \(String(format: "%.2f", peakTFLOPS)) TFLOPS (\(String(format: "%.1f", computeUtilization))% utilization)
        Memory BW: \(String(format: "%.1f", achievedBandwidthGBps)) / \(String(format: "%.1f", peakBandwidthGBps)) GB/s (\(String(format: "%.1f", bandwidthUtilization))% utilization)
        """
    }
}

// MARK: - GPU Metrics Calculator

/// Calculator for GPU utilization metrics.
public struct GPUMetricsCalculator: Sendable {

    public let hardwareSpecs: AppleSiliconSpecs

    public init(hardwareSpecs: AppleSiliconSpecs? = nil) {
        self.hardwareSpecs = hardwareSpecs ?? AppleSiliconSpecs.detect() ?? AppleSiliconSpecs(
            name: "Unknown",
            gpuCores: 10,
            fp32TFLOPS: 4.0,
            fp16TFLOPS: 8.0,
            memoryBandwidthGBps: 100
        )
    }

    /// Calculate GPU utilization metrics for a matrix multiplication operation.
    /// - Parameters:
    ///   - m: Number of rows in output matrix
    ///   - n: Number of columns in output matrix
    ///   - k: Shared dimension (contraction dimension)
    ///   - executionTimeSeconds: Measured execution time in seconds
    ///   - isFloat16: Whether the operation uses float16 (default: false for float32)
    /// - Returns: GPU utilization metrics
    public func calculateMatMulMetrics(
        m: Int,
        n: Int,
        k: Int,
        executionTimeSeconds: Double,
        isFloat16: Bool = false
    ) -> GPUUtilizationMetrics {
        // FLOPS for matrix multiplication: 2 * M * N * K (multiply-add)
        let flops = Double(2 * m * n * k)
        let achievedFLOPS = flops / executionTimeSeconds

        let peakFLOPS = (isFloat16 ? hardwareSpecs.fp16TFLOPS : hardwareSpecs.fp32TFLOPS) * 1e12
        let computeUtil = min(100.0, (achievedFLOPS / peakFLOPS) * 100.0)

        // Memory bandwidth: read A, B and write C
        let bytesPerElement = isFloat16 ? 2 : 4
        let bytesRead = (m * k + k * n) * bytesPerElement
        let bytesWritten = m * n * bytesPerElement
        let totalBytes = bytesRead + bytesWritten
        let achievedBW = Double(totalBytes) / executionTimeSeconds / 1e9

        let bandwidthUtil = min(100.0, (achievedBW / hardwareSpecs.memoryBandwidthGBps) * 100.0)

        return GPUUtilizationMetrics(
            achievedFLOPS: achievedFLOPS,
            peakFLOPS: peakFLOPS,
            computeUtilization: computeUtil,
            achievedBandwidthGBps: achievedBW,
            peakBandwidthGBps: hardwareSpecs.memoryBandwidthGBps,
            bandwidthUtilization: bandwidthUtil,
            hardwareSpecs: hardwareSpecs
        )
    }

    /// Calculate GPU utilization metrics for an elementwise operation.
    /// - Parameters:
    ///   - elements: Total number of elements processed
    ///   - opsPerElement: Number of FLOPs per element (e.g., 1 for add, ~10 for exp)
    ///   - executionTimeSeconds: Measured execution time in seconds
    ///   - isFloat16: Whether the operation uses float16
    /// - Returns: GPU utilization metrics
    public func calculateElementwiseMetrics(
        elements: Int,
        opsPerElement: Int,
        executionTimeSeconds: Double,
        isFloat16: Bool = false
    ) -> GPUUtilizationMetrics {
        let flops = Double(elements * opsPerElement)
        let achievedFLOPS = flops / executionTimeSeconds

        let peakFLOPS = (isFloat16 ? hardwareSpecs.fp16TFLOPS : hardwareSpecs.fp32TFLOPS) * 1e12
        let computeUtil = min(100.0, (achievedFLOPS / peakFLOPS) * 100.0)

        // Memory: read input, write output
        let bytesPerElement = isFloat16 ? 2 : 4
        let totalBytes = elements * bytesPerElement * 2  // Read + write
        let achievedBW = Double(totalBytes) / executionTimeSeconds / 1e9

        let bandwidthUtil = min(100.0, (achievedBW / hardwareSpecs.memoryBandwidthGBps) * 100.0)

        return GPUUtilizationMetrics(
            achievedFLOPS: achievedFLOPS,
            peakFLOPS: peakFLOPS,
            computeUtilization: computeUtil,
            achievedBandwidthGBps: achievedBW,
            peakBandwidthGBps: hardwareSpecs.memoryBandwidthGBps,
            bandwidthUtilization: bandwidthUtil,
            hardwareSpecs: hardwareSpecs
        )
    }

    /// Calculate GPU utilization metrics for a reduction operation.
    /// - Parameters:
    ///   - inputElements: Number of input elements
    ///   - outputElements: Number of output elements
    ///   - executionTimeSeconds: Measured execution time in seconds
    ///   - isFloat16: Whether the operation uses float16
    /// - Returns: GPU utilization metrics
    public func calculateReductionMetrics(
        inputElements: Int,
        outputElements: Int,
        executionTimeSeconds: Double,
        isFloat16: Bool = false
    ) -> GPUUtilizationMetrics {
        // Reduction has ~1 FLOP per input element
        let flops = Double(inputElements)
        let achievedFLOPS = flops / executionTimeSeconds

        let peakFLOPS = (isFloat16 ? hardwareSpecs.fp16TFLOPS : hardwareSpecs.fp32TFLOPS) * 1e12
        let computeUtil = min(100.0, (achievedFLOPS / peakFLOPS) * 100.0)

        // Memory: read all input, write output
        let bytesPerElement = isFloat16 ? 2 : 4
        let totalBytes = (inputElements + outputElements) * bytesPerElement
        let achievedBW = Double(totalBytes) / executionTimeSeconds / 1e9

        let bandwidthUtil = min(100.0, (achievedBW / hardwareSpecs.memoryBandwidthGBps) * 100.0)

        return GPUUtilizationMetrics(
            achievedFLOPS: achievedFLOPS,
            peakFLOPS: peakFLOPS,
            computeUtilization: computeUtil,
            achievedBandwidthGBps: achievedBW,
            peakBandwidthGBps: hardwareSpecs.memoryBandwidthGBps,
            bandwidthUtilization: bandwidthUtil,
            hardwareSpecs: hardwareSpecs
        )
    }

    /// Calculate GPU utilization metrics for convolution.
    /// - Parameters:
    ///   - batchSize: Batch size
    ///   - inputHeight: Input height
    ///   - inputWidth: Input width
    ///   - inputChannels: Number of input channels
    ///   - outputChannels: Number of output channels
    ///   - kernelHeight: Kernel height
    ///   - kernelWidth: Kernel width
    ///   - outputHeight: Output height
    ///   - outputWidth: Output width
    ///   - executionTimeSeconds: Measured execution time
    ///   - isFloat16: Whether using float16
    /// - Returns: GPU utilization metrics
    public func calculateConvolutionMetrics(
        batchSize: Int,
        inputHeight: Int,
        inputWidth: Int,
        inputChannels: Int,
        outputChannels: Int,
        kernelHeight: Int,
        kernelWidth: Int,
        outputHeight: Int,
        outputWidth: Int,
        executionTimeSeconds: Double,
        isFloat16: Bool = false
    ) -> GPUUtilizationMetrics {
        // FLOPS for convolution: 2 * B * OH * OW * OC * IC * KH * KW
        let flops = Double(2 * batchSize * outputHeight * outputWidth * outputChannels * inputChannels * kernelHeight * kernelWidth)
        let achievedFLOPS = flops / executionTimeSeconds

        let peakFLOPS = (isFloat16 ? hardwareSpecs.fp16TFLOPS : hardwareSpecs.fp32TFLOPS) * 1e12
        let computeUtil = min(100.0, (achievedFLOPS / peakFLOPS) * 100.0)

        // Memory estimation
        let bytesPerElement = isFloat16 ? 2 : 4
        let inputBytes = batchSize * inputHeight * inputWidth * inputChannels * bytesPerElement
        let kernelBytes = kernelHeight * kernelWidth * inputChannels * outputChannels * bytesPerElement
        let outputBytes = batchSize * outputHeight * outputWidth * outputChannels * bytesPerElement
        let totalBytes = inputBytes + kernelBytes + outputBytes
        let achievedBW = Double(totalBytes) / executionTimeSeconds / 1e9

        let bandwidthUtil = min(100.0, (achievedBW / hardwareSpecs.memoryBandwidthGBps) * 100.0)

        return GPUUtilizationMetrics(
            achievedFLOPS: achievedFLOPS,
            peakFLOPS: peakFLOPS,
            computeUtilization: computeUtil,
            achievedBandwidthGBps: achievedBW,
            peakBandwidthGBps: hardwareSpecs.memoryBandwidthGBps,
            bandwidthUtilization: bandwidthUtil,
            hardwareSpecs: hardwareSpecs
        )
    }
}

// MARK: - GPU Memory Tracker

/// Tracks GPU memory usage using Metal APIs.
public struct GPUMemoryTracker: Sendable {

    /// Current GPU memory statistics.
    public struct MemoryStats: Sendable {
        /// Current allocated size in bytes (from MTLDevice).
        public let currentAllocatedBytes: Int
        /// Recommended maximum working set size.
        public let recommendedMaxWorkingSetBytes: Int
        /// Whether the device has unified memory.
        public let hasUnifiedMemory: Bool
        /// Device name.
        public let deviceName: String

        /// Format as human-readable string.
        public func formatted() -> String {
            let allocMB = Double(currentAllocatedBytes) / (1024 * 1024)
            let maxMB = Double(recommendedMaxWorkingSetBytes) / (1024 * 1024 * 1024)
            return """
            Device: \(deviceName)
            Unified Memory: \(hasUnifiedMemory)
            Current Allocated: \(String(format: "%.2f", allocMB)) MB
            Recommended Max Working Set: \(String(format: "%.2f", maxMB)) GB
            """
        }
    }

    public init() {}

    /// Get current GPU memory statistics.
    public func getMemoryStats() -> MemoryStats? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }

        return MemoryStats(
            currentAllocatedBytes: device.currentAllocatedSize,
            recommendedMaxWorkingSetBytes: Int(device.recommendedMaxWorkingSetSize),
            hasUnifiedMemory: device.hasUnifiedMemory,
            deviceName: device.name
        )
    }

    /// Measure memory change during a closure execution.
    public func measureMemoryDelta<T>(_ operation: () throws -> T) rethrows -> (result: T, deltaBytes: Int) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            let result = try operation()
            return (result, 0)
        }

        let beforeBytes = device.currentAllocatedSize
        let result = try operation()
        let afterBytes = device.currentAllocatedSize

        return (result, afterBytes - beforeBytes)
    }
}
