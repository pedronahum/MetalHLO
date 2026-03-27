// OptimalSplitter.swift
// HeterogeneousFusion
//
// Phase 3: Finds the optimal partition for a given op shape using
// throughput curves from the ProfileDatabase.
//
// Solves the min-max problem:
//   minimize max(time_gpu(f_gpu * rows), time_mps(f_mps * rows), time_cpu(f_cpu * rows))
//   subject to f_gpu + f_mps + f_cpu = 1.0, f_i >= 0
//
// Uses iterative rebalancing: shift rows from slow units to fast units
// until all units finish at approximately the same time.

import Foundation

/// Finds optimal partitions using throughput curves.
public final class OptimalSplitter: Sendable {

    private let profileDB: ProfileDatabase
    private let contentionModel: ContentionModel?

    public init(profileDB: ProfileDatabase, contentionModel: ContentionModel? = nil) {
        self.profileDB = profileDB
        self.contentionModel = contentionModel
    }

    /// Find the optimal partition for a matmul of the given shape.
    ///
    /// - Parameters:
    ///   - shape: Matrix dimensions (M, K, N).
    ///   - units: Which units to include. Default: all three.
    ///   - maxIterations: Solver iterations.
    /// - Returns: A PartitionDescriptor with optimal fractions, or nil if no curves available.
    public func optimalMatmulPartition(
        shape: MatrixShape,
        units: [ComputeUnit] = [.gpu, .mps, .cpu],
        maxIterations: Int = 100
    ) -> PartitionDescriptor? {
        // Get throughput curves for each requested unit.
        // Prefer shape-specific curves (K, N) over generic ones.
        var unitCurves: [(ComputeUnit, ThroughputCurve)] = []
        for unit in units {
            guard let curve = profileDB.curve(unit: unit, op: .matmul, K: shape.K, N: shape.N) else { continue }
            unitCurves.append((unit, curve))
        }

        guard unitCurves.count >= 2 else { return nil }

        let M = shape.M

        // Step 1: Initial proportional split based on throughput at reduced size
        // Using M/sqrt(2) avoids bias from full-size evaluation where CPU's
        // cache-friendly zone can skew the initial estimate.
        let initRows = max(1, Int(Double(M) / sqrt(2.0)))
        var fractions = initialProportionalSplit(unitCurves: unitCurves, totalRows: initRows)

        // Step 2: Iterative rebalancing — converge toward equal finish times
        for _ in 0..<maxIterations {
            let times = unitCurves.enumerated().map { (i, uc) in
                uc.1.estimate(rows: max(1, Int(fractions[i] * Double(M))))
            }

            let maxTime = times.max()!
            let minTime = times.min()!

            // Converged: all units within 0.5% of each other
            if maxTime - minTime < maxTime * 0.005 { break }

            // Rebalance: shift rows from slow units to fast units
            fractions = rebalance(fractions: fractions, times: times, maxTime: maxTime)
        }

        // Clamp and normalize fractions
        fractions = fractions.map { max(0, $0) }
        let sum = fractions.reduce(0, +)
        if sum > 0 {
            fractions = fractions.map { $0 / sum }
        }

        // Step 3: Grid search refinement around converged point.
        // Evaluates ±10% perturbations to find the true min-max.
        if unitCurves.count == 3 {
            fractions = gridRefine(fractions: fractions, unitCurves: unitCurves, M: M)
        }

        // Build descriptor
        let pairs = zip(unitCurves.map(\.0), fractions).map { ($0.0, $0.1) }
        return PartitionDescriptor.matmul(shape: shape, fractions: pairs)
    }

    /// Find optimal two-way split (any two units).
    public func optimalTwoWayPartition(
        shape: MatrixShape,
        units: (ComputeUnit, ComputeUnit) = (.gpu, .mps)
    ) -> PartitionDescriptor? {
        optimalMatmulPartition(shape: shape, units: [units.0, units.1])
    }

    /// Predict the fused execution time for a given partition.
    ///
    /// When a ContentionModel is present, adjusts isolated throughput estimates
    /// to account for concurrent memory bandwidth contention.
    public func predictFusedTime(descriptor: PartitionDescriptor) -> Double? {
        let M = descriptor.fullInputShape[0]
        let K = descriptor.fullInputShape.count >= 2 ? descriptor.fullInputShape[1] : 0
        let N = descriptor.fullOutputShape.count >= 2 ? descriptor.fullOutputShape[1] : 0
        let totalElements = M * N
        var maxTime: Double = 0
        for assignment in descriptor.assignments {
            guard let curve = profileDB.curve(unit: assignment.unit, op: descriptor.op, K: K, N: N) else {
                return nil
            }
            let rows = assignment.inputSlices[0].shape[0]
            var t = curve.estimate(rows: rows)
            // Apply contention penalty: isolated throughput overestimates concurrent performance
            if let cm = contentionModel, descriptor.assignments.count > 1 {
                t = cm.adjustedTime(isolatedTimeMs: t, totalElements: totalElements)
            }
            maxTime = max(maxTime, t)
        }
        return maxTime
    }

    /// Predict the best single-unit time.
    /// Uses shape-specific curves when available.
    public func predictBestSingleTime(
        shape: MatrixShape,
        units: [ComputeUnit] = [.gpu, .mps, .cpu]
    ) -> (ComputeUnit, Double)? {
        var best: (ComputeUnit, Double)?
        for unit in units {
            guard let curve = profileDB.curve(unit: unit, op: .matmul, K: shape.K, N: shape.N) else { continue }
            let t = curve.estimate(rows: shape.M)
            if best == nil || t < best!.1 {
                best = (unit, t)
            }
        }
        return best
    }

    /// Predict speedup of optimal partition vs best single unit.
    public func predictSpeedup(shape: MatrixShape, units: [ComputeUnit] = [.gpu, .mps, .cpu]) -> Double? {
        guard let descriptor = optimalMatmulPartition(shape: shape, units: units),
              let fusedTime = predictFusedTime(descriptor: descriptor),
              let (_, singleTime) = predictBestSingleTime(shape: shape, units: units),
              fusedTime > 0 else {
            return nil
        }
        return singleTime / fusedTime
    }

    // MARK: - Internal

    /// Grid search around converged fractions to find true min-max optimum.
    /// Compensates for model inaccuracies (e.g., concurrent bandwidth contention).
    private func gridRefine(
        fractions: [Double],
        unitCurves: [(ComputeUnit, ThroughputCurve)],
        M: Int
    ) -> [Double] {
        var bestFractions = fractions
        var bestMaxTime = Double.infinity

        // Evaluate converged point
        let baseMax = unitCurves.enumerated().map { (i, uc) in
            uc.1.estimate(rows: max(1, Int(fractions[i] * Double(M))))
        }.max()!
        bestMaxTime = baseMax

        // Search ±10% in 2% steps for first two units (third is determined)
        let step = 0.02
        let range = stride(from: -0.10, through: 0.10, by: step)

        for d0 in range {
            let f0 = fractions[0] + d0
            guard f0 >= 0.01 else { continue }
            for d1 in range {
                let f1 = fractions[1] + d1
                guard f1 >= 0.01 else { continue }
                let f2 = 1.0 - f0 - f1
                guard f2 >= 0.01 else { continue }

                let candidate = [f0, f1, f2]
                let maxTime = unitCurves.enumerated().map { (i, uc) in
                    uc.1.estimate(rows: max(1, Int(candidate[i] * Double(M))))
                }.max()!

                if maxTime < bestMaxTime {
                    bestMaxTime = maxTime
                    bestFractions = candidate
                }
            }
        }

        return bestFractions
    }

    /// Proportional split: each unit gets fraction proportional to throughput at full size.
    private func initialProportionalSplit(
        unitCurves: [(ComputeUnit, ThroughputCurve)],
        totalRows: Int
    ) -> [Double] {
        let throughputs = unitCurves.map { $0.1.throughput(rows: totalRows) }
        let total = throughputs.reduce(0, +)
        guard total > 0 else {
            return Array(repeating: 1.0 / Double(unitCurves.count), count: unitCurves.count)
        }
        return throughputs.map { $0 / total }
    }

    /// Shift rows from slow units to fast units.
    ///
    /// Uses maxTime as the target (min-max objective): all units should
    /// finish at the same time as the current slowest unit, but with
    /// rebalanced work distribution.
    private func rebalance(fractions: [Double], times: [Double], maxTime: Double) -> [Double] {
        var newFractions = fractions

        for i in 0..<fractions.count {
            if times[i] > 0 {
                // Target: the slowest unit's time. Scale each unit's fraction
                // so it would finish at maxTime given its current rate.
                let ratio = maxTime / times[i]
                // Adaptive damping: aggressive when far from target, gentle when close
                let distance = abs(times[i] - maxTime) / maxTime
                let damping = 0.5 + 0.5 * (1.0 - distance)  // [0.5, 1.0]
                newFractions[i] = fractions[i] * (damping + (1.0 - damping) * ratio)
            }
        }

        // Normalize
        let sum = newFractions.reduce(0, +)
        if sum > 0 {
            newFractions = newFractions.map { $0 / sum }
        }

        return newFractions
    }
}
