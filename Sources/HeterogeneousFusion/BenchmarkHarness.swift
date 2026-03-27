// BenchmarkHarness.swift
// HeterogeneousFusion
//
// Timing infrastructure for Phase 1 heterogeneous fusion benchmarks.
// Runs 20 iterations, discards top and bottom 10%, reports trimmed mean ± std.

import Foundation
import Metal
import QuartzCore

/// Configuration for benchmark runs.
public struct HeterogeneousBenchmarkConfig: Sendable {
    /// Total iterations to run.
    public let iterations: Int
    /// Fraction of outliers to trim from each end.
    public let trimFraction: Double
    /// Number of warmup iterations (discarded entirely).
    public let warmupIterations: Int

    public static let standard = HeterogeneousBenchmarkConfig(
        iterations: 20, trimFraction: 0.10, warmupIterations: 5
    )

    public static let quick = HeterogeneousBenchmarkConfig(
        iterations: 10, trimFraction: 0.10, warmupIterations: 3
    )

    public init(iterations: Int, trimFraction: Double, warmupIterations: Int) {
        self.iterations = iterations
        self.trimFraction = trimFraction
        self.warmupIterations = warmupIterations
    }

    /// Number of samples to trim from each end.
    public var trimCount: Int {
        max(1, Int(Double(iterations) * trimFraction))
    }
}

/// Statistical summary of a set of timing measurements.
public struct TimingSummary: Sendable, CustomStringConvertible {
    public let rawSamples: [Double]
    public let trimmedSamples: [Double]
    public let mean: Double
    public let stdDev: Double
    public let min: Double
    public let max: Double
    public let median: Double

    public init(samples: [Double], trimCount: Int) {
        self.rawSamples = samples
        let sorted = samples.sorted()
        let lo = Swift.min(trimCount, sorted.count / 2)
        let hi = Swift.max(lo, sorted.count - trimCount)
        self.trimmedSamples = Array(sorted[lo..<hi])

        guard !trimmedSamples.isEmpty else {
            self.mean = 0; self.stdDev = 0; self.min = 0; self.max = 0; self.median = 0
            return
        }

        let sum = trimmedSamples.reduce(0, +)
        let calculatedMean = sum / Double(trimmedSamples.count)
        self.mean = calculatedMean

        let variance = trimmedSamples.map { ($0 - calculatedMean) * ($0 - calculatedMean) }.reduce(0, +) / Double(trimmedSamples.count)
        self.stdDev = sqrt(variance)

        self.min = trimmedSamples.first!
        self.max = trimmedSamples.last!

        let midIdx = trimmedSamples.count / 2
        self.median = trimmedSamples.count % 2 == 0
            ? (trimmedSamples[midIdx - 1] + trimmedSamples[midIdx]) / 2.0
            : trimmedSamples[midIdx]
    }

    public var description: String {
        String(format: "%.3f ± %.3f ms (min=%.3f, max=%.3f, median=%.3f)",
               mean, stdDev, min, max, median)
    }
}

/// Result of benchmarking a single shape across all execution modes.
public struct ShapeBenchmarkResult: Sendable {
    public let shape: MatrixShape
    public let gpuOnly: TimingSummary
    public let mpsOnly: TimingSummary
    public let cpuOnly: TimingSummary
    public let fusedTwoWay: TimingSummary
    public let fusedThreeWay: TimingSummary?
    public let fusedProfile: FusedProfile  // last fused run's detailed profile

    /// Best single-unit time in ms.
    public var bestSingleMs: Double {
        Swift.min(gpuOnly.mean, Swift.min(mpsOnly.mean, cpuOnly.mean))
    }

    /// Optimal-ratio two-way fused result (GPU fraction tuned to measured throughput).
    public let fusedTwoWayOptimal: TimingSummary?
    /// Optimal-ratio three-way fused result.
    public let fusedThreeWayOptimal: TimingSummary?
    /// The GPU fraction used for the optimal two-way split.
    public let optimalGpuFraction2: Double?
    /// The fractions used for the optimal three-way split.
    public let optimalFractions3: (gpu: Double, mps: Double, cpu: Double)?

    /// Speedup of two-way fused vs best single.
    public var twoWaySpeedup: Double {
        guard fusedTwoWay.mean > 0 else { return 0 }
        return bestSingleMs / fusedTwoWay.mean
    }

    /// Speedup of three-way fused vs best single.
    public var threeWaySpeedup: Double? {
        guard let three = fusedThreeWay, three.mean > 0 else { return nil }
        return bestSingleMs / three.mean
    }

    /// Speedup of optimal two-way fused vs best single.
    public var optimalTwoWaySpeedup: Double? {
        guard let opt = fusedTwoWayOptimal, opt.mean > 0 else { return nil }
        return bestSingleMs / opt.mean
    }

    /// Speedup of optimal three-way fused vs best single.
    public var optimalThreeWaySpeedup: Double? {
        guard let opt = fusedThreeWayOptimal, opt.mean > 0 else { return nil }
        return bestSingleMs / opt.mean
    }

    /// Best speedup across all fused modes.
    public var bestSpeedup: Double {
        let candidates = [
            twoWaySpeedup,
            threeWaySpeedup ?? 0,
            optimalTwoWaySpeedup ?? 0,
            optimalThreeWaySpeedup ?? 0,
        ]
        return candidates.max() ?? 0
    }

    /// Which single unit was fastest?
    public var bestSingleUnit: ComputeUnit {
        let gpu = gpuOnly.mean
        let mps = mpsOnly.mean
        let cpu = cpuOnly.mean
        if gpu <= mps && gpu <= cpu { return .gpu }
        if mps <= gpu && mps <= cpu { return .mps }
        return .cpu
    }
}

/// Result of Phase 3 solver validation.
public struct SolverValidationResult: Sendable {
    public let shape: MatrixShape
    /// Empirical optimal fractions (from 1/time ratios).
    public let empiricalFractions: (gpu: Double, mps: Double, cpu: Double)
    /// Solver-predicted optimal fractions.
    public let solverFractions: (gpu: Double, mps: Double, cpu: Double)
    /// Empirical optimal fused time (ms).
    public let empiricalFusedMs: Double
    /// Solver-predicted fused time (ms).
    public let solverFusedMs: Double
    /// Relative deviation: |solver - empirical| / empirical.
    public let fractionDeviation: Double
    /// Relative deviation of fused times (positive means solver is slower).
    public let timeDeviation: Double
    /// Whether solver fused time is within 5% of empirical optimum.
    public var pass: Bool { timeDeviation < 0.05 }
}

/// Allocates shared-memory MTLBuffers and runs all benchmark configurations.
public final class BenchmarkHarness: @unchecked Sendable {

    private let device: MTLDevice
    private let config: HeterogeneousBenchmarkConfig
    private let gpu: MatmulGPU
    private let mps: MatmulANE
    private let cpu: MatmulCPU
    private let fused: FusedMatmul
    public let profileDB: ProfileDatabase
    public let splitter: OptimalSplitter
    public let monitor: HardwareMonitor

    public init(device: MTLDevice, config: HeterogeneousBenchmarkConfig = .standard) throws {
        self.device = device
        self.config = config
        self.gpu = try MatmulGPU(device: device)
        self.mps = try MatmulANE(device: device)
        self.cpu = MatmulCPU()
        self.fused = try FusedMatmul(device: device)
        self.profileDB = ProfileDatabase()
        self.splitter = OptimalSplitter(profileDB: profileDB)
        self.monitor = HardwareMonitor()
    }

    /// Run all benchmarks for a given shape.
    public func benchmark(shape: MatrixShape) throws -> ShapeBenchmarkResult {
        let M = shape.M, K = shape.K, N = shape.N

        // Allocate shared-memory buffers (zero-copy on Apple Silicon unified memory)
        let aBytes = M * K * MemoryLayout<Float>.size
        let bBytes = K * N * MemoryLayout<Float>.size
        let cBytes = M * N * MemoryLayout<Float>.size

        guard let bufA = device.makeBuffer(length: aBytes, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bBytes, options: .storageModeShared),
              let bufC = device.makeBuffer(length: cBytes, options: .storageModeShared) else {
            throw HeterogeneousError.bufferAllocationFailed("Failed to allocate \(shape) buffers")
        }

        // Fill with random data
        fillRandom(bufA, count: M * K)
        fillRandom(bufB, count: K * N)

        // 1. GPU-only baseline
        let gpuTimes = measure(label: "GPU") {
            _ = gpu.execute(A: bufA, B: bufB, C: bufC, M: M, K: K, N: N)
        }
        let gpuSummary = TimingSummary(samples: gpuTimes, trimCount: config.trimCount)

        // 2. MPS-only baseline
        let mpsTimes = measure(label: "MPS") {
            _ = mps.execute(A: bufA, B: bufB, C: bufC, M: M, K: K, N: N)
        }
        let mpsSummary = TimingSummary(samples: mpsTimes, trimCount: config.trimCount)

        // 3. CPU-only baseline
        let cpuTimes = measure(label: "CPU") {
            _ = cpu.execute(A: bufA, B: bufB, C: bufC, sliceM: M, K: K, N: N)
        }
        let cpuSummary = TimingSummary(samples: cpuTimes, trimCount: config.trimCount)

        // 4. Fused two-way (GPU + MPS, 50/50)
        var lastFusedProfile: FusedProfile!
        let fusedTwoTimes = measure(label: "Fused2-50/50") {
            lastFusedProfile = fused.executeTwoWay(
                A: bufA, B: bufB, C: bufC,
                shape: shape, gpuFraction: 0.5
            )
        }
        let fusedTwoSummary = TimingSummary(samples: fusedTwoTimes, trimCount: config.trimCount)

        // 5. Fused three-way (GPU + MPS + CPU, ~33/33/34)
        var fusedThreeSummary: TimingSummary? = nil
        if M >= 64 {
            let fusedThreeTimes = measure(label: "Fused3-equal") {
                lastFusedProfile = fused.executeThreeWay(
                    A: bufA, B: bufB, C: bufC,
                    shape: shape, gpuFraction: 0.34, mpsFraction: 0.33
                )
            }
            fusedThreeSummary = TimingSummary(samples: fusedThreeTimes, trimCount: config.trimCount)
        }

        // 6. Compute optimal split ratios from measured throughputs
        //    Proportional split: each unit gets rows proportional to 1/time
        //    Goal: all units finish at the same time
        let gpuRate = 1.0 / gpuSummary.mean   // "rows per ms" proxy
        let mpsRate = 1.0 / mpsSummary.mean
        let cpuRate = 1.0 / cpuSummary.mean

        // Optimal two-way (GPU + MPS)
        let total2 = gpuRate + mpsRate
        let optGpuFrac2 = gpuRate / total2
        var fusedTwoOptSummary: TimingSummary? = nil
        let fusedTwoOptTimes = measure(label: "Fused2-opt") {
            lastFusedProfile = fused.executeTwoWay(
                A: bufA, B: bufB, C: bufC,
                shape: shape, gpuFraction: optGpuFrac2
            )
        }
        fusedTwoOptSummary = TimingSummary(samples: fusedTwoOptTimes, trimCount: config.trimCount)

        // Optimal three-way (GPU + MPS + CPU)
        var fusedThreeOptSummary: TimingSummary? = nil
        let total3 = gpuRate + mpsRate + cpuRate
        let optGpuFrac3 = gpuRate / total3
        let optMpsFrac3 = mpsRate / total3
        let optCpuFrac3 = cpuRate / total3
        if M >= 64 {
            let fusedThreeOptTimes = measure(label: "Fused3-opt") {
                lastFusedProfile = fused.executeThreeWay(
                    A: bufA, B: bufB, C: bufC,
                    shape: shape, gpuFraction: optGpuFrac3, mpsFraction: optMpsFrac3
                )
            }
            fusedThreeOptSummary = TimingSummary(samples: fusedThreeOptTimes, trimCount: config.trimCount)
        }

        return ShapeBenchmarkResult(
            shape: shape,
            gpuOnly: gpuSummary,
            mpsOnly: mpsSummary,
            cpuOnly: cpuSummary,
            fusedTwoWay: fusedTwoSummary,
            fusedThreeWay: fusedThreeSummary,
            fusedProfile: lastFusedProfile,
            fusedTwoWayOptimal: fusedTwoOptSummary,
            fusedThreeWayOptimal: fusedThreeOptSummary,
            optimalGpuFraction2: optGpuFrac2,
            optimalFractions3: (gpu: optGpuFrac3, mps: optMpsFrac3, cpu: optCpuFrac3)
        )
    }

    /// Verify that the descriptor-based executor produces numerically correct results
    /// by comparing against a single-unit MPS baseline.
    /// Returns max absolute error across all output elements.
    public func validateCorrectness(shape: MatrixShape) throws -> Double {
        let M = shape.M, K = shape.K, N = shape.N
        let aBytes = M * K * MemoryLayout<Float>.size
        let bBytes = K * N * MemoryLayout<Float>.size
        let cBytes = M * N * MemoryLayout<Float>.size

        guard let bufA = device.makeBuffer(length: aBytes, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bBytes, options: .storageModeShared),
              let cRef = device.makeBuffer(length: cBytes, options: .storageModeShared),
              let cTest = device.makeBuffer(length: cBytes, options: .storageModeShared) else {
            throw HeterogeneousError.bufferAllocationFailed("Validation buffer allocation failed")
        }

        fillRandom(bufA, count: M * K)
        fillRandom(bufB, count: K * N)

        // Reference: MPS single-unit
        _ = mps.execute(A: bufA, B: bufB, C: cRef, M: M, K: K, N: N)

        // Test: descriptor-based three-way split
        let descriptor = PartitionDescriptor.matmul(
            shape: shape,
            fractions: [(.gpu, 0.25), (.mps, 0.50), (.cpu, 0.25)]
        )
        _ = fused.execute(descriptor: descriptor, A: bufA, B: bufB, C: cTest)

        // Compare
        let refPtr = cRef.contents().assumingMemoryBound(to: Float.self)
        let testPtr = cTest.contents().assumingMemoryBound(to: Float.self)
        var maxError: Double = 0
        for i in 0..<(M * N) {
            let err = abs(Double(refPtr[i]) - Double(testPtr[i]))
            maxError = Swift.max(maxError, err)
        }
        return maxError
    }

    // MARK: - Internal

    /// Run a closure `config.iterations` times (after warmup), return wall-clock times in ms.
    private func measure(label: String, _ body: () -> Void) -> [Double] {
        // Warmup
        for _ in 0..<config.warmupIterations {
            body()
        }

        // Measure
        var times: [Double] = []
        times.reserveCapacity(config.iterations)
        for _ in 0..<config.iterations {
            let start = CACurrentMediaTime()
            body()
            let elapsed = (CACurrentMediaTime() - start) * 1000.0
            times.append(elapsed)
        }
        return times
    }

    /// Fill a buffer with random float32 values in [0, 1).
    private func fillRandom(_ buffer: MTLBuffer, count: Int) {
        let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            ptr[i] = Float.random(in: 0..<1)
        }
    }

    // MARK: - Phase 3: Calibration & Solver Validation

    /// Run calibration to build throughput curves for all three units.
    /// Uses the same measurement infrastructure as the benchmark.
    public func calibrate(
        K: Int = 2048, N: Int = 2048,
        rowCounts: [Int] = [64, 256, 512, 1024, 2048, 4096]
    ) throws {
        try profileDB.calibrateMatmul(
            device: device, K: K, N: N,
            rowCounts: rowCounts,
            warmup: config.warmupIterations,
            iterations: config.iterations
        )
    }

    /// Validate the solver against empirical optimal splits for a given shape.
    ///
    /// 1. Calibrates throughput curves at the target shape's K,N for accuracy.
    /// 2. Measures single-unit baselines to get empirical optimal fractions (1/time ratios).
    /// 3. Asks the solver for its predicted optimal partition.
    /// 4. Runs both the empirical and solver splits, measuring actual fused time.
    /// 5. Returns deviation metrics.
    public func validateSolver(shape: MatrixShape) throws -> SolverValidationResult {
        let M = shape.M, K = shape.K, N = shape.N

        // Calibrate at this shape's K,N with dense sampling for accurate curves.
        // Include points around M to capture cache transition effects.
        var rowSet: Set<Int> = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
        // Add points near the target M for local accuracy
        if M >= 128 {
            rowSet.insert(max(64, M / 4))
            rowSet.insert(max(64, M / 2))
            rowSet.insert(max(64, M * 3 / 4))
        }
        let rowCounts = rowSet.sorted().filter { $0 <= M }
        if !rowCounts.isEmpty {
            try profileDB.calibrateMatmul(
                device: device, K: K, N: N,
                rowCounts: rowCounts,
                warmup: config.warmupIterations,
                iterations: config.iterations
            )
        }

        let aBytes = M * K * MemoryLayout<Float>.size
        let bBytes = K * N * MemoryLayout<Float>.size
        let cBytes = M * N * MemoryLayout<Float>.size

        guard let bufA = device.makeBuffer(length: aBytes, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bBytes, options: .storageModeShared),
              let bufC = device.makeBuffer(length: cBytes, options: .storageModeShared) else {
            throw HeterogeneousError.bufferAllocationFailed("Solver validation buffer allocation failed")
        }
        fillRandom(bufA, count: M * K)
        fillRandom(bufB, count: K * N)

        // Measure single-unit baselines
        let gpuTimes = measure(label: "SV-GPU") {
            _ = gpu.execute(A: bufA, B: bufB, C: bufC, M: M, K: K, N: N)
        }
        let mpsTimes = measure(label: "SV-MPS") {
            _ = mps.execute(A: bufA, B: bufB, C: bufC, M: M, K: K, N: N)
        }
        let cpuTimes = measure(label: "SV-CPU") {
            _ = cpu.execute(A: bufA, B: bufB, C: bufC, sliceM: M, K: K, N: N)
        }

        let gpuMs = TimingSummary(samples: gpuTimes, trimCount: config.trimCount).mean
        let mpsMs = TimingSummary(samples: mpsTimes, trimCount: config.trimCount).mean
        let cpuMs = TimingSummary(samples: cpuTimes, trimCount: config.trimCount).mean

        // Empirical optimal: proportional to 1/time
        let gpuRate = 1.0 / gpuMs
        let mpsRate = 1.0 / mpsMs
        let cpuRate = 1.0 / cpuMs
        let totalRate = gpuRate + mpsRate + cpuRate
        let empGpu = gpuRate / totalRate
        let empMps = mpsRate / totalRate
        let empCpu = cpuRate / totalRate

        // Solver optimal
        guard let solverDesc = splitter.optimalMatmulPartition(shape: shape) else {
            throw HeterogeneousError.executionFailed("Solver returned nil — no throughput curves available.")
        }
        var solGpu = 0.0, solMps = 0.0, solCpu = 0.0
        for a in solverDesc.assignments {
            switch a.unit {
            case .gpu: solGpu = a.workFraction
            case .mps: solMps = a.workFraction
            case .cpu: solCpu = a.workFraction
            }
        }

        // Run empirical optimal fused
        let empDesc = PartitionDescriptor.matmul(
            shape: shape,
            fractions: [(.gpu, empGpu), (.mps, empMps), (.cpu, empCpu)]
        )
        let empTimes = measure(label: "SV-Emp") {
            _ = fused.execute(descriptor: empDesc, A: bufA, B: bufB, C: bufC)
        }
        let empFusedMs = TimingSummary(samples: empTimes, trimCount: config.trimCount).mean

        // Run solver optimal fused
        let solTimes = measure(label: "SV-Sol") {
            _ = fused.execute(descriptor: solverDesc, A: bufA, B: bufB, C: bufC)
        }
        let solFusedMs = TimingSummary(samples: solTimes, trimCount: config.trimCount).mean

        // Track with hardware monitor
        if let predicted = splitter.predictFusedTime(descriptor: solverDesc) {
            monitor.record(predicted: predicted, actual: solFusedMs)
        }

        // Compute deviations
        let fracDev = abs(solGpu - empGpu) + abs(solMps - empMps) + abs(solCpu - empCpu)
        // Positive = solver is slower, negative = solver is faster
        let timeDev = empFusedMs > 0 ? max(0, (solFusedMs - empFusedMs) / empFusedMs) : 0

        return SolverValidationResult(
            shape: shape,
            empiricalFractions: (gpu: empGpu, mps: empMps, cpu: empCpu),
            solverFractions: (gpu: solGpu, mps: solMps, cpu: solCpu),
            empiricalFusedMs: empFusedMs,
            solverFusedMs: solFusedMs,
            fractionDeviation: fracDev,
            timeDeviation: timeDev
        )
    }

    /// Run solver-predicted optimal splits in the main benchmark flow.
    /// Returns a PartitionDescriptor using the solver's fractions.
    public func solverOptimalDescriptor(shape: MatrixShape) -> PartitionDescriptor? {
        splitter.optimalMatmulPartition(shape: shape)
    }

    // MARK: - Column-Split Benchmarks

    /// Validate correctness of column-split execution against MPS baseline.
    public func validateColumnSplitCorrectness(shape: MatrixShape) throws -> Double {
        let M = shape.M, K = shape.K, N = shape.N
        let aBytes = M * K * MemoryLayout<Float>.size
        let bBytes = K * N * MemoryLayout<Float>.size
        let cBytes = M * N * MemoryLayout<Float>.size

        guard let bufA = device.makeBuffer(length: aBytes, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bBytes, options: .storageModeShared),
              let cRef = device.makeBuffer(length: cBytes, options: .storageModeShared),
              let cTest = device.makeBuffer(length: cBytes, options: .storageModeShared) else {
            throw HeterogeneousError.bufferAllocationFailed("Column-split validation buffer allocation failed")
        }

        fillRandom(bufA, count: M * K)
        fillRandom(bufB, count: K * N)

        // Reference: MPS single-unit
        _ = mps.execute(A: bufA, B: bufB, C: cRef, M: M, K: K, N: N)

        // Test: column-split three-way
        let descriptor = PartitionDescriptor.matmulColumnSplit(
            shape: shape,
            fractions: [(.gpu, 0.25), (.mps, 0.50), (.cpu, 0.25)]
        )
        _ = fused.execute(descriptor: descriptor, A: bufA, B: bufB, C: cTest)

        let refPtr = cRef.contents().assumingMemoryBound(to: Float.self)
        let testPtr = cTest.contents().assumingMemoryBound(to: Float.self)
        var maxError: Double = 0
        for i in 0..<(M * N) {
            let err = abs(Double(refPtr[i]) - Double(testPtr[i]))
            maxError = Swift.max(maxError, err)
        }
        return maxError
    }

    /// Benchmark row-split vs column-split for a given shape.
    /// Returns (rowSplitMs, colSplitMs) trimmed mean times.
    public func benchmarkSplitComparison(
        shape: MatrixShape,
        fractions: (gpu: Double, mps: Double, cpu: Double)
    ) throws -> (rowSplitMs: Double, colSplitMs: Double) {
        let M = shape.M, K = shape.K, N = shape.N
        let aBytes = M * K * MemoryLayout<Float>.size
        let bBytes = K * N * MemoryLayout<Float>.size
        let cBytes = M * N * MemoryLayout<Float>.size

        guard let bufA = device.makeBuffer(length: aBytes, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bBytes, options: .storageModeShared),
              let bufC = device.makeBuffer(length: cBytes, options: .storageModeShared) else {
            throw HeterogeneousError.bufferAllocationFailed("Split comparison buffer allocation failed")
        }
        fillRandom(bufA, count: M * K)
        fillRandom(bufB, count: K * N)

        let unitFracs: [(ComputeUnit, Double)] = [
            (.gpu, fractions.gpu), (.mps, fractions.mps), (.cpu, fractions.cpu)
        ]

        // Row-split
        let rowDesc = PartitionDescriptor.matmul(shape: shape, fractions: unitFracs)
        let rowTimes = measure(label: "RowSplit") {
            _ = fused.execute(descriptor: rowDesc, A: bufA, B: bufB, C: bufC)
        }
        let rowMs = TimingSummary(samples: rowTimes, trimCount: config.trimCount).mean

        // Column-split
        let colDesc = PartitionDescriptor.matmulColumnSplit(shape: shape, fractions: unitFracs)
        let colTimes = measure(label: "ColSplit") {
            _ = fused.execute(descriptor: colDesc, A: bufA, B: bufB, C: bufC)
        }
        let colMs = TimingSummary(samples: colTimes, trimCount: config.trimCount).mean

        return (rowSplitMs: rowMs, colSplitMs: colMs)
    }

    // MARK: - Bandwidth Contention Measurement

    /// Result of bandwidth contention measurement.
    public struct ContentionResult: Sendable {
        /// Time for a single fused matmul (ms).
        public let singleFusedMs: Double
        /// Time for two concurrent fused matmuls (ms, wall-clock of both finishing).
        public let dualFusedMs: Double
        /// Slowdown factor: dualFusedMs / singleFusedMs.
        /// 1.0 = no contention, 2.0 = fully serialized.
        public let slowdownFactor: Double
        /// Recommended max concurrent fused ops before bandwidth saturation.
        public let recommendedConcurrencyBudget: Int
    }

    /// Measure bandwidth contention by running two fused matmuls simultaneously.
    ///
    /// Allocates two independent buffer sets (different MTLHeaps), both committed
    /// before either waits. Compares wall-clock of dual-concurrent vs single.
    ///
    /// This answers: "In a dense graph with multiple fused ops, does each op's
    /// speedup degrade due to shared memory bus contention?"
    public func measureBandwidthContention(shape: MatrixShape) throws -> ContentionResult {
        let M = shape.M, K = shape.K, N = shape.N
        let aBytes = M * K * MemoryLayout<Float>.size
        let bBytes = K * N * MemoryLayout<Float>.size
        let cBytes = M * N * MemoryLayout<Float>.size

        // Allocate two independent buffer sets
        guard let a1 = device.makeBuffer(length: aBytes, options: .storageModeShared),
              let b1 = device.makeBuffer(length: bBytes, options: .storageModeShared),
              let c1 = device.makeBuffer(length: cBytes, options: .storageModeShared),
              let a2 = device.makeBuffer(length: aBytes, options: .storageModeShared),
              let b2 = device.makeBuffer(length: bBytes, options: .storageModeShared),
              let c2 = device.makeBuffer(length: cBytes, options: .storageModeShared) else {
            throw HeterogeneousError.bufferAllocationFailed("Contention measurement buffer allocation failed")
        }
        fillRandom(a1, count: M * K)
        fillRandom(b1, count: K * N)
        fillRandom(a2, count: M * K)
        fillRandom(b2, count: K * N)

        let desc = PartitionDescriptor.matmul(
            shape: shape,
            fractions: [(.gpu, 0.34), (.mps, 0.33), (.cpu, 0.33)]
        )

        // Measure single fused matmul
        let singleTimes = measure(label: "Contention-single") {
            _ = fused.execute(descriptor: desc, A: a1, B: b1, C: c1)
        }
        let singleMs = TimingSummary(samples: singleTimes, trimCount: config.trimCount).mean

        // Measure two concurrent fused matmuls
        let dualTimes = measure(label: "Contention-dual") {
            // Launch both fused executions overlapping in time.
            // Each fused.execute() internally does commit-before-wait for its 3 units.
            // Use UnsafeMutablePointer to avoid Swift 6 sendable capture warnings.
            nonisolated(unsafe) let times = UnsafeMutablePointer<Double>.allocate(capacity: 2)
            times[0] = 0; times[1] = 0
            let group = DispatchGroup()

            nonisolated(unsafe) let a1_ = a1
            nonisolated(unsafe) let b1_ = b1
            nonisolated(unsafe) let c1_ = c1
            nonisolated(unsafe) let a2_ = a2
            nonisolated(unsafe) let b2_ = b2
            nonisolated(unsafe) let c2_ = c2

            group.enter()
            DispatchQueue.global(qos: .userInteractive).async {
                let start = CACurrentMediaTime()
                _ = self.fused.execute(descriptor: desc, A: a1_, B: b1_, C: c1_)
                times[0] = (CACurrentMediaTime() - start) * 1000.0
                group.leave()
            }

            group.enter()
            DispatchQueue.global(qos: .userInteractive).async {
                let start = CACurrentMediaTime()
                _ = self.fused.execute(descriptor: desc, A: a2_, B: b2_, C: c2_)
                times[1] = (CACurrentMediaTime() - start) * 1000.0
                group.leave()
            }

            group.wait()
            _ = Swift.max(times[0], times[1])
            times.deallocate()
        }
        let dualMs = TimingSummary(samples: dualTimes, trimCount: config.trimCount).mean

        let slowdown = singleMs > 0 ? dualMs / singleMs : 1.0

        // Recommended budget: if slowdown < 1.3, allow 2 concurrent; else limit to 1
        let budget: Int
        if slowdown < 1.15 {
            budget = 3  // Minimal contention — allow 3 concurrent fused ops
        } else if slowdown < 1.30 {
            budget = 2  // Moderate contention — allow 2
        } else {
            budget = 1  // Severe contention — serialize fused ops
        }

        return ContentionResult(
            singleFusedMs: singleMs,
            dualFusedMs: dualMs,
            slowdownFactor: slowdown,
            recommendedConcurrencyBudget: budget
        )
    }

    // MARK: - Systematic Correctness Suite

    /// A single correctness test case.
    public struct CorrectnessTestCase: Sendable {
        public let shape: MatrixShape
        public let splitStrategy: String  // "row", "column", "fallback"
        public let tolerance: Double
        public let maxError: Double
        public let pass: Bool
    }

    /// Run the full correctness suite against all (shape, split) combinations
    /// from the benchmark table.
    ///
    /// Each test: fused result vs single-unit MPS reference. The 512×512 case
    /// at 0.53x (worst performance) is the most dangerous for recombine edge cases.
    public func runCorrectnessSuite() throws -> [CorrectnessTestCase] {
        let tolerance: Double = 1e-2  // float32 matmul accumulation tolerance
        var results: [CorrectnessTestCase] = []

        let testCases: [(MatrixShape, String)] = [
            // Small (regression regime — correctness still required)
            (MatrixShape.square(256),                          "row"),
            (MatrixShape.square(512),                          "row"),
            // Medium (sweet spot)
            (MatrixShape.square(1024),                         "row"),
            (MatrixShape.square(2048),                         "row"),
            // Large
            (MatrixShape.square(4096),                         "row"),
            // Rectangular
            (MatrixShape(M: 4096, K: 4096, N: 512),           "row"),
            (MatrixShape(M: 512, K: 512, N: 4096),            "row"),
            (MatrixShape(M: 256, K: 256, N: 2048),            "row"),
            // Column-split variants
            (MatrixShape(M: 256, K: 256, N: 512),             "column"),
            (MatrixShape(M: 512, K: 512, N: 4096),            "column"),
            (MatrixShape.square(1024),                         "column"),
            // Fallback (single unit — should match exactly)
            (MatrixShape.square(64),                           "fallback"),
            (MatrixShape.square(128),                          "fallback"),
        ]

        for (shape, strategy) in testCases {
            let maxError: Double
            switch strategy {
            case "row":
                maxError = try validateRowSplitCorrectness(shape: shape)
            case "column":
                maxError = try validateColumnSplitCorrectness(shape: shape)
            case "fallback":
                maxError = try validateFallbackCorrectness(shape: shape)
            default:
                continue
            }

            results.append(CorrectnessTestCase(
                shape: shape,
                splitStrategy: strategy,
                tolerance: tolerance,
                maxError: maxError,
                pass: maxError < tolerance
            ))
        }

        return results
    }

    /// Validate row-split correctness against MPS reference.
    private func validateRowSplitCorrectness(shape: MatrixShape) throws -> Double {
        try validateCorrectness(shape: shape)
    }

    /// Validate fallback (single-unit GPU) correctness against MPS reference.
    private func validateFallbackCorrectness(shape: MatrixShape) throws -> Double {
        let M = shape.M, K = shape.K, N = shape.N
        let aBytes = M * K * MemoryLayout<Float>.size
        let bBytes = K * N * MemoryLayout<Float>.size
        let cBytes = M * N * MemoryLayout<Float>.size

        guard let bufA = device.makeBuffer(length: aBytes, options: .storageModeShared),
              let bufB = device.makeBuffer(length: bBytes, options: .storageModeShared),
              let cRef = device.makeBuffer(length: cBytes, options: .storageModeShared),
              let cTest = device.makeBuffer(length: cBytes, options: .storageModeShared) else {
            throw HeterogeneousError.bufferAllocationFailed("Fallback validation buffer allocation failed")
        }

        fillRandom(bufA, count: M * K)
        fillRandom(bufB, count: K * N)

        // Reference: MPS
        _ = mps.execute(A: bufA, B: bufB, C: cRef, M: M, K: K, N: N)
        // Test: GPU only
        _ = gpu.execute(A: bufA, B: bufB, C: cTest, M: M, K: K, N: N)

        let refPtr = cRef.contents().assumingMemoryBound(to: Float.self)
        let testPtr = cTest.contents().assumingMemoryBound(to: Float.self)
        var maxError: Double = 0
        for i in 0..<(M * N) {
            let err = abs(Double(refPtr[i]) - Double(testPtr[i]))
            maxError = Swift.max(maxError, err)
        }
        return maxError
    }

    // MARK: - Profitability Guard

    /// Create a ProfitabilityGuard with this harness's solver and profiles.
    public func makeProfitabilityGuard() -> ProfitabilityGuard {
        ProfitabilityGuard(splitter: splitter, profileDB: profileDB)
    }
}
