// ProfileDatabase.swift
// HeterogeneousFusion
//
// Phase 3: Stores fitted ThroughputCurves per (unit, op) pair.
// Built from Phase 1 calibration measurements.
// Can be serialized to JSON for reuse across sessions.

import Foundation
import Metal
import QuartzCore

/// Key for looking up a throughput curve.
///
/// The optional `shapeClass` (K, N) distinguishes curves fitted at different
/// inner/output dimensions. When nil, this is the legacy "square" curve.
/// When set, it's a curve fitted for a specific (K, N) family — e.g., K=768, N=2304
/// for GPT-2's QKV projection shapes.
public struct ProfileKey: Hashable, Sendable, Codable {
    public let unit: ComputeUnit
    public let op: HLOOpCode
    /// Shape class: (K, N) dimensions that the curve was fitted for.
    /// nil = legacy single curve (backwards compatible).
    public let shapeK: Int?
    public let shapeN: Int?

    public init(unit: ComputeUnit, op: HLOOpCode, K: Int? = nil, N: Int? = nil) {
        self.unit = unit
        self.op = op
        self.shapeK = K
        self.shapeN = N
    }
}

/// Stores throughput curves and provides calibration utilities.
public final class ProfileDatabase: @unchecked Sendable {

    private var curves: [ProfileKey: ThroughputCurve] = [:]

    public init() {}

    /// Look up a curve for a given (unit, op) pair.
    /// Falls back to the generic (nil K,N) curve if no shape-specific curve exists.
    public func curve(unit: ComputeUnit, op: HLOOpCode) -> ThroughputCurve? {
        curves[ProfileKey(unit: unit, op: op)]
    }

    /// Look up a shape-specific curve for (unit, op, K, N).
    /// Falls back to the generic curve if no shape-specific curve exists.
    public func curve(unit: ComputeUnit, op: HLOOpCode, K: Int, N: Int) -> ThroughputCurve? {
        let specific = ProfileKey(unit: unit, op: op, K: K, N: N)
        if let curve = curves[specific] { return curve }
        // Fall back to generic curve
        return curves[ProfileKey(unit: unit, op: op)]
    }

    /// Store a fitted curve.
    public func store(_ curve: ThroughputCurve) {
        curves[ProfileKey(unit: curve.unit, op: curve.op)] = curve
    }

    /// Store a shape-specific fitted curve.
    public func store(_ curve: ThroughputCurve, K: Int, N: Int) {
        curves[ProfileKey(unit: curve.unit, op: curve.op, K: K, N: N)] = curve
    }

    /// All stored curves.
    public var allCurves: [ThroughputCurve] {
        Array(curves.values)
    }

    // MARK: - Calibration

    /// Run calibration for matmul on all three units.
    ///
    /// Measures execution time at multiple row counts to build throughput curves.
    /// Uses the provided device for GPU/MPS, CPU is always available.
    ///
    /// - Parameters:
    ///   - device: Metal device.
    ///   - K: Inner dimension for calibration matmuls.
    ///   - N: Output columns for calibration matmuls.
    ///   - rowCounts: Row counts to measure at (e.g., [64, 256, 512, 1024, 2048, 4096]).
    ///   - warmup: Warmup iterations per measurement.
    ///   - iterations: Measurement iterations per data point.
    public func calibrateMatmul(
        device: MTLDevice,
        K: Int = 2048,
        N: Int = 2048,
        rowCounts: [Int] = [64, 256, 512, 1024, 2048, 4096],
        warmup: Int = 3,
        iterations: Int = 5
    ) throws {
        let gpu = try MatmulGPU(device: device)
        let mps = try MatmulANE(device: device)
        let cpu = MatmulCPU()

        var gpuMeasurements: [(rows: Int, timeMs: Double)] = []
        var mpsMeasurements: [(rows: Int, timeMs: Double)] = []
        var cpuMeasurements: [(rows: Int, timeMs: Double)] = []

        for M in rowCounts {
            let aBytes = M * K * MemoryLayout<Float>.size
            let bBytes = K * N * MemoryLayout<Float>.size
            let cBytes = M * N * MemoryLayout<Float>.size

            guard let bufA = device.makeBuffer(length: aBytes, options: .storageModeShared),
                  let bufB = device.makeBuffer(length: bBytes, options: .storageModeShared),
                  let bufC = device.makeBuffer(length: cBytes, options: .storageModeShared) else {
                continue
            }

            // Fill with data
            let aPtr = bufA.contents().assumingMemoryBound(to: Float.self)
            let bPtr = bufB.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<(M * K) { aPtr[i] = Float.random(in: 0..<1) }
            for i in 0..<(K * N) { bPtr[i] = Float.random(in: 0..<1) }

            // GPU
            let gpuTime = measureUnit(warmup: warmup, iterations: iterations) {
                _ = gpu.execute(A: bufA, B: bufB, C: bufC, M: M, K: K, N: N)
            }
            gpuMeasurements.append((rows: M, timeMs: gpuTime))

            // MPS
            let mpsTime = measureUnit(warmup: warmup, iterations: iterations) {
                _ = mps.execute(A: bufA, B: bufB, C: bufC, M: M, K: K, N: N)
            }
            mpsMeasurements.append((rows: M, timeMs: mpsTime))

            // CPU
            let cpuTime = measureUnit(warmup: warmup, iterations: iterations) {
                _ = cpu.execute(A: bufA, B: bufB, C: bufC, sliceM: M, K: K, N: N)
            }
            cpuMeasurements.append((rows: M, timeMs: cpuTime))
        }

        // Fit curves
        store(ThroughputCurve.fit(measurements: gpuMeasurements, unit: .gpu, op: .matmul))
        store(ThroughputCurve.fit(measurements: mpsMeasurements, unit: .mps, op: .matmul))
        store(ThroughputCurve.fit(measurements: cpuMeasurements, unit: .cpu, op: .matmul))
    }

    /// Calibrate matmul at specific (M, K, N) shapes.
    ///
    /// Groups shapes by (K, N) and fits a separate curve per (unit, K, N) family.
    /// This fixes the rectangular shape blind spot: the solver gets curves that
    /// reflect actual time(rows) for each (K, N) class instead of extrapolating
    /// from square calibration data.
    ///
    /// - Parameters:
    ///   - device: Metal device.
    ///   - shapes: List of (M, K, N) triplets to measure.
    ///   - warmup: Warmup iterations per measurement.
    ///   - iterations: Measurement iterations per data point.
    public func calibrateMatmulShapes(
        device: MTLDevice,
        shapes: [(M: Int, K: Int, N: Int)],
        warmup: Int = 3,
        iterations: Int = 5
    ) throws {
        let gpu = try MatmulGPU(device: device)
        let mps = try MatmulANE(device: device)
        let cpu = MatmulCPU()

        // Group shapes by (K, N) family
        var families: [String: [(M: Int, K: Int, N: Int)]] = [:]
        for shape in shapes {
            let key = "\(shape.K)_\(shape.N)"
            families[key, default: []].append(shape)
        }

        for (_, familyShapes) in families {
            let K = familyShapes[0].K
            let N = familyShapes[0].N

            var gpuMeasurements: [(rows: Int, timeMs: Double)] = []
            var mpsMeasurements: [(rows: Int, timeMs: Double)] = []
            var cpuMeasurements: [(rows: Int, timeMs: Double)] = []

            for shape in familyShapes.sorted(by: { $0.M < $1.M }) {
                let M = shape.M
                let aBytes = M * K * MemoryLayout<Float>.size
                let bBytes = K * N * MemoryLayout<Float>.size
                let cBytes = M * N * MemoryLayout<Float>.size

                guard let bufA = device.makeBuffer(length: aBytes, options: .storageModeShared),
                      let bufB = device.makeBuffer(length: bBytes, options: .storageModeShared),
                      let bufC = device.makeBuffer(length: cBytes, options: .storageModeShared) else {
                    continue
                }

                let aPtr = bufA.contents().assumingMemoryBound(to: Float.self)
                let bPtr = bufB.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<(M * K) { aPtr[i] = Float.random(in: 0..<1) }
                for i in 0..<(K * N) { bPtr[i] = Float.random(in: 0..<1) }

                let gpuTime = measureUnit(warmup: warmup, iterations: iterations) {
                    _ = gpu.execute(A: bufA, B: bufB, C: bufC, M: M, K: K, N: N)
                }
                gpuMeasurements.append((rows: M, timeMs: gpuTime))

                let mpsTime = measureUnit(warmup: warmup, iterations: iterations) {
                    _ = mps.execute(A: bufA, B: bufB, C: bufC, M: M, K: K, N: N)
                }
                mpsMeasurements.append((rows: M, timeMs: mpsTime))

                let cpuTime = measureUnit(warmup: warmup, iterations: iterations) {
                    _ = cpu.execute(A: bufA, B: bufB, C: bufC, sliceM: M, K: K, N: N)
                }
                cpuMeasurements.append((rows: M, timeMs: cpuTime))
            }

            // Fit and store shape-specific curves
            store(ThroughputCurve.fit(measurements: gpuMeasurements, unit: .gpu, op: .matmul), K: K, N: N)
            store(ThroughputCurve.fit(measurements: mpsMeasurements, unit: .mps, op: .matmul), K: K, N: N)
            store(ThroughputCurve.fit(measurements: cpuMeasurements, unit: .cpu, op: .matmul), K: K, N: N)
        }
    }

    /// Measure mean execution time of a closure.
    private func measureUnit(warmup: Int, iterations: Int, _ body: () -> Void) -> Double {
        for _ in 0..<warmup { body() }
        var total: Double = 0
        for _ in 0..<iterations {
            let start = CACurrentMediaTime()
            body()
            total += (CACurrentMediaTime() - start) * 1000.0
        }
        return total / Double(iterations)
    }

    // MARK: - Serialization

    /// Export all curves to JSON data.
    public func toJSON() throws -> Data {
        let entries = curves.map { (key, curve) in
            CurveEntry(unit: key.unit.rawValue, op: key.op.rawValue,
                       a: curve.a, b: curve.b, c: curve.c)
        }
        return try JSONEncoder().encode(entries)
    }

    /// Import curves from JSON data.
    public func loadJSON(_ data: Data) throws {
        let entries = try JSONDecoder().decode([CurveEntry].self, from: data)
        for e in entries {
            guard let unit = ComputeUnit(rawValue: e.unit),
                  let op = HLOOpCode(rawValue: e.op) else { continue }
            store(ThroughputCurve(a: e.a, b: e.b, c: e.c, unit: unit, op: op))
        }
    }

    private struct CurveEntry: Codable {
        let unit: String
        let op: String
        let a: Double
        let b: Double
        let c: Double
    }
}
