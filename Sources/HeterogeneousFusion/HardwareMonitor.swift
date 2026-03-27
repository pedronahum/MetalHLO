// HardwareMonitor.swift
// HeterogeneousFusion
//
// Phase 3: Monitors runtime deviation between predicted and actual execution
// times. Triggers re-calibration when deviation exceeds threshold.

import Foundation

/// Recalibration scope for graph-level execution.
///
/// Phase 4 uses `.perOp` — when a single op's deviation exceeds the threshold,
/// only that op's throughput curves are recalibrated. This is fast (~100ms for
/// 6 calibration points) but doesn't account for cross-op bandwidth interference.
///
/// Future: `.heaviestOpProportional` recalibrates the graph's heaviest op and
/// scales other ops by the observed bandwidth ratio.
public enum RecalibrationScope: Sendable {
    /// Recalibrate only the op that triggered the threshold.
    case perOp
    /// Recalibrate the heaviest op, scale others proportionally.
    case heaviestOpProportional
    /// Recalibrate all ops in the graph (expensive).
    case fullGraph
}

/// Monitors predicted vs actual execution times and triggers re-solve
/// when deviation exceeds a configurable threshold.
///
/// Tracks per-op deviation history. The `scope` field determines what
/// gets recalibrated when the threshold is crossed.
public final class HardwareMonitor: @unchecked Sendable {

    /// Threshold for deviation triggering re-calibration (default 15%).
    public let deviationThreshold: Double

    /// Rolling window size for deviation tracking.
    public let windowSize: Int

    /// What to recalibrate when threshold is exceeded.
    public let scope: RecalibrationScope

    /// Maximum number of simultaneously fused ops (bandwidth budget).
    /// Initialized from contention measurement. 0 = unlimited.
    public var concurrencyBudget: Int = 0

    private var history: [(predicted: Double, actual: Double)] = []
    private var recalibrationCount: Int = 0

    public init(
        deviationThreshold: Double = 0.15,
        windowSize: Int = 10,
        scope: RecalibrationScope = .perOp
    ) {
        self.deviationThreshold = deviationThreshold
        self.windowSize = windowSize
        self.scope = scope
    }

    /// Record a predicted vs actual execution time pair.
    /// Returns true if deviation exceeds threshold (caller should re-calibrate).
    @discardableResult
    public func record(predicted: Double, actual: Double) -> Bool {
        history.append((predicted: predicted, actual: actual))
        if history.count > windowSize {
            history.removeFirst(history.count - windowSize)
        }
        return shouldRecalibrate
    }

    /// Whether the recent deviation exceeds the threshold.
    public var shouldRecalibrate: Bool {
        guard history.count >= 3 else { return false }
        return meanDeviation > deviationThreshold
    }

    /// Mean absolute relative deviation over the window.
    public var meanDeviation: Double {
        guard !history.isEmpty else { return 0 }
        let deviations = history.map { entry -> Double in
            guard entry.predicted > 0 else { return 0 }
            return abs(entry.actual - entry.predicted) / entry.predicted
        }
        return deviations.reduce(0, +) / Double(deviations.count)
    }

    /// Number of times re-calibration has been triggered.
    public var totalRecalibrations: Int { recalibrationCount }

    /// Mark that a re-calibration was performed.
    public func didRecalibrate() {
        recalibrationCount += 1
        history.removeAll()
    }

    /// Current window of observations.
    public var observations: [(predicted: Double, actual: Double)] {
        history
    }
}
