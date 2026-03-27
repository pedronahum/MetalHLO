// HardwareCalibration.swift
// HeterogeneousFusion
//
// Derives all five hardware constants for the current device.
// Runtime: 30-90 seconds depending on chip speed.
// Called once on first launch; result cached to disk.
//
// Uses the real pipeline (ProfileDatabase → OptimalSplitter → FusedExecutor)
// to measure fused speedup. This is critical — approximate fractions from
// isolated unit profiling don't match the solver's optimized splits, which
// causes calibration to underestimate achievable speedup.

import Foundation
import Metal
import MetalPerformanceShaders
import QuartzCore

// MARK: - Calibration Context

/// Holds reusable objects for the duration of calibration.
private final class CalibrationContext {
    let device: MTLDevice
    let profileDB: ProfileDatabase
    let splitter: OptimalSplitter
    let executor: FusedExecutor
    let mps: MatmulANE
    let queue: MTLCommandQueue

    init(device: MTLDevice) throws {
        self.device = device
        self.mps = try MatmulANE(device: device)
        guard let q = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create command queue")
        }
        self.queue = q
        self.executor = try FusedExecutor(device: device)

        // Quick calibration: 6 row counts at K=768, N=50257 (the shape class that matters)
        // Plus square calibration for general curves.
        self.profileDB = ProfileDatabase()
        try profileDB.calibrateMatmul(
            device: device, K: 2048, N: 2048,
            rowCounts: [64, 256, 512, 1024, 2048, 4096],
            warmup: 2, iterations: 3
        )
        // Shape-specific curves for the logit projection shape class
        try profileDB.calibrateMatmulShapes(
            device: device,
            shapes: [
                (M: 128, K: 768, N: 50257),
                (M: 256, K: 768, N: 50257),
                (M: 512, K: 768, N: 50257),
                (M: 1024, K: 768, N: 50257),
                (M: 2048, K: 768, N: 50257),
            ],
            warmup: 2, iterations: 3
        )

        let contentionModel = ContentionModel()
        self.splitter = OptimalSplitter(profileDB: profileDB, contentionModel: contentionModel)
    }
}

// MARK: - Entry Point

extension HardwareProfile {

    /// Derives all five hardware constants for the current device.
    public static func calibrate(device: MTLDevice) -> HardwareProfile {
        print("[MetalHLO] First launch: calibrating hardware profile for \(device.name)...")
        print("[MetalHLO] This runs once and takes ~60 seconds.")

        guard let ctx = try? CalibrationContext(device: device) else {
            print("[MetalHLO] Metal setup failed -- using conservative defaults.")
            return defaultProfile(device: device)
        }
        print("[MetalHLO] Throughput curves calibrated.")

        let maxOutBytes = measureMaxOutputBytes(ctx: ctx)
        let budget = measureConcurrencyBudget(ctx: ctx)
        let ceiling = measureContentionCeiling(ctx: ctx)
        let (minElem, minN) = measureProfitabilityBoundary(ctx: ctx)

        let profile = HardwareProfile(
            deviceName: device.name,
            calibrationDate: Date(),
            minOutputElements: minElem,
            minOutputColumns: minN,
            maxOutputBytes: maxOutBytes,
            contentionCeiling: ceiling,
            concurrencyBudget: budget
        )

        let ramGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        print("[MetalHLO] Calibration complete.")
        print("[MetalHLO]   minOutputElements: \(minElem / 1_000_000)M")
        print("[MetalHLO]   minOutputColumns:  \(minN)")
        print("[MetalHLO]   maxOutputBytes:    \(maxOutBytes / (1024 * 1024))MB (\(Int(Double(maxOutBytes) / Double(ProcessInfo.processInfo.physicalMemory) * 100))% of \(String(format: "%.0f", ramGB))GB)")
        print("[MetalHLO]   contentionCeiling: \(String(format: "%.2f", ceiling))x")
        print("[MetalHLO]   concurrencyBudget: \(budget)")

        return profile
    }

    private static func defaultProfile(device: MTLDevice) -> HardwareProfile {
        HardwareProfile(
            deviceName: device.name,
            calibrationDate: Date(),
            minOutputElements: 10_000_000,
            minOutputColumns: 32_768,
            maxOutputBytes: Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.25),
            contentionCeiling: 1.5,
            concurrencyBudget: 1
        )
    }
}

// MARK: - Constant 1: maxOutputBytes

extension HardwareProfile {

    /// Safe output buffer size — fraction of RAM before thrashing.
    fileprivate static func measureMaxOutputBytes(ctx: CalibrationContext) -> Int {
        let physicalRAM = ProcessInfo.processInfo.physicalMemory

        // Step through fractions of RAM. If per-element time spikes 3x, cliff found.
        let testN = 4096
        let testK = 4096
        let fractions: [Double] = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        var lastSafeBytes = Int(Double(physicalRAM) * 0.10)
        var baselineTimePerElement: Double? = nil

        for fraction in fractions {
            let testBytes = Int(Double(physicalRAM) * fraction)
            let M = testBytes / (testN * MemoryLayout<Float>.size)
            guard M >= 64 else { continue }

            let aBytes = M * testK * MemoryLayout<Float>.size
            let bBytes = testK * testN * MemoryLayout<Float>.size
            let cBytes = M * testN * MemoryLayout<Float>.size

            guard let bufA = ctx.device.makeBuffer(length: aBytes, options: .storageModeShared),
                  let bufB = ctx.device.makeBuffer(length: bBytes, options: .storageModeShared),
                  let bufC = ctx.device.makeBuffer(length: cBytes, options: .storageModeShared) else {
                break
            }

            fillRandom(bufA, count: M * testK)
            fillRandom(bufB, count: testK * testN)

            _ = executeMPSOnly(ctx: ctx, A: bufA, B: bufB, C: bufC, M: M, K: testK, N: testN)
            var times = [Double]()
            for _ in 0..<3 {
                times.append(executeMPSOnly(ctx: ctx, A: bufA, B: bufB, C: bufC,
                                            M: M, K: testK, N: testN))
            }
            let meanTime = times.reduce(0, +) / Double(times.count)
            let timePerElement = meanTime / Double(M * testN)

            if let baseline = baselineTimePerElement {
                if timePerElement > baseline * 3.0 { break }
            } else {
                baselineTimePerElement = timePerElement
            }

            lastSafeBytes = testBytes
        }

        return Int(Double(lastSafeBytes) * 0.80)
    }
}

// MARK: - Constant 2: concurrencyBudget

extension HardwareProfile {

    /// 1 vs 2 concurrent MPS matmuls. Slowdown > 1.7x → budget = 1.
    fileprivate static func measureConcurrencyBudget(ctx: CalibrationContext) -> Int {
        let M = 1024, K = 2048, N = 2048
        let runs = 6

        let aBytes = M * K * MemoryLayout<Float>.size
        let bBytes = K * N * MemoryLayout<Float>.size
        let cBytes = M * N * MemoryLayout<Float>.size

        guard let bufA = ctx.device.makeBuffer(length: aBytes, options: .storageModeShared),
              let bufB = ctx.device.makeBuffer(length: bBytes, options: .storageModeShared),
              let bufC1 = ctx.device.makeBuffer(length: cBytes, options: .storageModeShared),
              let bufC2 = ctx.device.makeBuffer(length: cBytes, options: .storageModeShared) else {
            return 1
        }

        fillRandom(bufA, count: M * K)
        fillRandom(bufB, count: K * N)

        for _ in 0..<2 {
            _ = executeMPSOnly(ctx: ctx, A: bufA, B: bufB, C: bufC1, M: M, K: K, N: N)
        }

        var singleTimes = [Double]()
        for _ in 0..<runs {
            singleTimes.append(executeMPSOnly(ctx: ctx, A: bufA, B: bufB, C: bufC1,
                                              M: M, K: K, N: N))
        }
        let t1 = trimmedMean(singleTimes, fraction: 0.10)

        var dualTimes = [Double]()
        for _ in 0..<runs {
            let start = CACurrentMediaTime()
            let cb1 = ctx.queue.makeCommandBuffer()!
            let cb2 = ctx.queue.makeCommandBuffer()!
            ctx.mps.encode(A: bufA, B: bufB, C: bufC1,
                           sliceM: M, K: K, N: N, commandBuffer: cb1)
            ctx.mps.encode(A: bufA, B: bufB, C: bufC2,
                           sliceM: M, K: K, N: N, commandBuffer: cb2)
            cb1.commit()
            cb2.commit()
            cb1.waitUntilCompleted()
            cb2.waitUntilCompleted()
            dualTimes.append((CACurrentMediaTime() - start) * 1000.0)
        }
        let t2 = trimmedMean(dualTimes, fraction: 0.10)

        return (t2 / t1) > 1.70 ? 1 : 2
    }
}

// MARK: - Constant 3: contentionCeiling

extension HardwareProfile {

    /// Max fused speedup at shapes where the solver produces good partitions.
    fileprivate static func measureContentionCeiling(ctx: CalibrationContext) -> Double {
        let physicalRAM = ProcessInfo.processInfo.physicalMemory

        let testShapes: [(M: Int, N: Int, K: Int)] = [
            (512,  50257, 768),
            (1024, 50257, 768),
        ]

        var maxSpeedup = 1.0

        for shape in testShapes {
            let outputBytes = shape.M * shape.N * MemoryLayout<Float>.size
            guard outputBytes < Int(physicalRAM) / 4 else { continue }

            let speedup = measureFusedSpeedup(
                ctx: ctx, M: shape.M, N: shape.N, K: shape.K,
                runs: 5, warmup: 2
            )
            maxSpeedup = max(maxSpeedup, speedup)
        }

        return maxSpeedup * 0.90
    }
}

// MARK: - Constants 4 & 5: minOutputElements and minOutputColumns

extension HardwareProfile {

    fileprivate static func measureProfitabilityBoundary(
        ctx: CalibrationContext
    ) -> (minOutputElements: Int, minOutputColumns: Int) {
        let minN = measureNThreshold(ctx: ctx)
        let minElements = measureElementThreshold(ctx: ctx, confirmedN: minN)
        return (minElements, minN)
    }

    /// N sweep: M=1024, K=768, vary N. Crossover at >= 1.05x.
    private static func measureNThreshold(ctx: CalibrationContext) -> Int {
        let physicalRAM = ProcessInfo.processInfo.physicalMemory
        let crossoverThreshold = 1.05

        let coarseN = [4_096, 8_192, 16_384, 24_576, 32_768, 49_152, 65_536]
        var lastFailN = coarseN[0]
        var crossoverN: Int? = nil

        for N in coarseN {
            let outputBytes = 1024 * N * MemoryLayout<Float>.size
            guard outputBytes < Int(physicalRAM) / 4 else { break }

            let speedup = measureFusedSpeedup(
                ctx: ctx, M: 1024, N: N, K: 768,
                runs: 5, warmup: 2
            )

            if speedup >= crossoverThreshold {
                crossoverN = N
                break
            }
            lastFailN = N
        }

        guard let crossN = crossoverN else {
            return Int(Double(coarseN.last!) * 1.20)
        }

        // Fine sweep: 3 points between lastFailN and crossN
        let step = max((crossN - lastFailN) / 3, 1024)
        var refinedLastFail = lastFailN

        var fineN = lastFailN + step
        while fineN < crossN {
            let outputBytes = 1024 * fineN * MemoryLayout<Float>.size
            guard outputBytes < Int(physicalRAM) / 4 else { break }

            let speedup = measureFusedSpeedup(
                ctx: ctx, M: 1024, N: fineN, K: 768,
                runs: 5, warmup: 2
            )
            if speedup < crossoverThreshold {
                refinedLastFail = fineN
            } else {
                break
            }
            fineN += step
        }

        let midpoint = (refinedLastFail + crossN) / 2
        return Int(Double(midpoint) * 1.10)
    }

    /// Element sweep: fix N at confirmed crossover, vary M.
    private static func measureElementThreshold(
        ctx: CalibrationContext,
        confirmedN: Int
    ) -> Int {
        let physicalRAM = ProcessInfo.processInfo.physicalMemory
        let crossoverThreshold = 1.05

        let targetElements = [
            2_000_000, 5_000_000, 8_000_000, 10_000_000,
            15_000_000, 25_000_000, 50_000_000
        ]
        let mValues = targetElements.map { $0 / max(confirmedN, 1) }
            .filter { $0 >= 32 }

        var lastFailElements = targetElements.first ?? 2_000_000

        for M in mValues {
            let outputBytes = M * confirmedN * MemoryLayout<Float>.size
            guard outputBytes < Int(physicalRAM) / 4 else { break }

            let speedup = measureFusedSpeedup(
                ctx: ctx, M: M, N: confirmedN, K: 768,
                runs: 5, warmup: 2
            )

            if speedup < crossoverThreshold {
                lastFailElements = M * confirmedN
            } else {
                let crossover = (lastFailElements + M * confirmedN) / 2
                return Int(Double(crossover) * 1.10)
            }
        }

        return Int(Double(lastFailElements) * 1.10)
    }
}

// MARK: - Core Measurement

extension HardwareProfile {

    /// Fused speedup using the real pipeline: solver-optimized partition + FusedExecutor.
    fileprivate static func measureFusedSpeedup(
        ctx: CalibrationContext,
        M: Int, N: Int, K: Int,
        runs: Int,
        warmup: Int
    ) -> Double {
        let shape = MatrixShape(M: M, K: K, N: N)

        // Get solver-optimized partition descriptor
        guard let descriptor = ctx.splitter.optimalMatmulPartition(shape: shape) else {
            return 0.0
        }

        let aBytes = M * K * MemoryLayout<Float>.size
        let bBytes = K * N * MemoryLayout<Float>.size
        let cBytes = M * N * MemoryLayout<Float>.size

        guard let bufA = ctx.device.makeBuffer(length: aBytes, options: .storageModeShared),
              let bufB = ctx.device.makeBuffer(length: bBytes, options: .storageModeShared),
              let bufC = ctx.device.makeBuffer(length: cBytes, options: .storageModeShared) else {
            return 0.0
        }

        fillRandom(bufA, count: M * K)
        fillRandom(bufB, count: K * N)

        // Warmup
        for _ in 0..<warmup {
            _ = ctx.executor.execute(descriptor: descriptor,
                                     inputBuffers: [bufA, bufB],
                                     outputBuffer: bufC)
            _ = executeMPSOnly(ctx: ctx, A: bufA, B: bufB, C: bufC, M: M, K: K, N: N)
        }

        // Interleaved measurement
        var fusedTimes = [Double]()
        var mpsTimes = [Double]()

        for _ in 0..<runs {
            let profile = ctx.executor.execute(descriptor: descriptor,
                                               inputBuffers: [bufA, bufB],
                                               outputBuffer: bufC)
            fusedTimes.append(profile.totalWallClockMs)

            mpsTimes.append(
                executeMPSOnly(ctx: ctx, A: bufA, B: bufB, C: bufC, M: M, K: K, N: N))
        }

        let fusedMean = trimmedMean(fusedTimes, fraction: 0.10)
        let mpsMean = trimmedMean(mpsTimes, fraction: 0.10)

        guard fusedMean > 0 else { return 0.0 }
        return mpsMean / fusedMean
    }

    /// MPS-only baseline.
    private static func executeMPSOnly(
        ctx: CalibrationContext,
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        M: Int, K: Int, N: Int
    ) -> Double {
        let start = CACurrentMediaTime()
        let cb = ctx.queue.makeCommandBuffer()!
        ctx.mps.encode(A: A, B: B, C: C, sliceM: M, K: K, N: N, commandBuffer: cb)
        cb.commit()
        cb.waitUntilCompleted()
        return (CACurrentMediaTime() - start) * 1000.0
    }
}

// MARK: - Utilities

extension HardwareProfile {

    static func trimmedMean(_ values: [Double], fraction: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let trimCount = Int(Double(values.count) * fraction)
        let trimmed = Array(sorted.dropFirst(trimCount).dropLast(trimCount))
        guard !trimmed.isEmpty else { return sorted[sorted.count / 2] }
        return trimmed.reduce(0, +) / Double(trimmed.count)
    }

    static func fillRandom(_ buffer: MTLBuffer, count: Int) {
        let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            ptr[i] = Float.random(in: -1..<1)
        }
    }
}
