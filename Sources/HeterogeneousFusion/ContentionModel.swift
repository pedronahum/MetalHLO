// ContentionModel.swift
// HeterogeneousFusion
//
// Models the throughput penalty from concurrent execution on unified memory.
//
// The core problem: throughput curves are calibrated with each unit running
// in isolation. When GPU, MPS, and CPU run concurrently, they contend for
// unified memory bandwidth. The solver predicts 1.77x speedup but actual
// is 0.6x because it doesn't account for this contention.
//
// Empirical measurements:
//   - Two concurrent fused matmuls: 2.02x slowdown (full serialization)
//   - At large sizes (>50M elements): ~55% of isolated throughput available
//   - At small sizes (<10M elements): ~30% (overhead dominates)
//
// The model applies a contention discount to isolated throughput predictions
// before the solver computes optimal splits.

import Foundation

/// Adjusts isolated throughput predictions for concurrent execution contention.
public struct ContentionModel: Sendable {

    /// Throughput fraction available at large element counts (>50M).
    /// Derived from the 2.02x concurrent slowdown measurement.
    public let largeSizeContentionFactor: Double

    /// Throughput fraction at small element counts (<10M).
    /// Overhead (command buffers, fences, slicing) dominates at small sizes.
    public let smallSizeContentionFactor: Double

    /// Element count threshold below which small-size factor applies.
    public let smallSizeThreshold: Int

    /// Element count threshold above which large-size factor applies.
    public let largeSizeThreshold: Int

    public init(
        largeSizeContentionFactor: Double = 0.55,
        smallSizeContentionFactor: Double = 0.30,
        smallSizeThreshold: Int = 10_000_000,
        largeSizeThreshold: Int = 50_000_000
    ) {
        self.largeSizeContentionFactor = largeSizeContentionFactor
        self.smallSizeContentionFactor = smallSizeContentionFactor
        self.smallSizeThreshold = smallSizeThreshold
        self.largeSizeThreshold = largeSizeThreshold
    }

    /// Returns the fraction of isolated throughput available during concurrent execution.
    ///
    /// Linearly interpolates between small and large factors for element counts
    /// between the two thresholds.
    public func contentionFactor(totalElements: Int) -> Double {
        if totalElements <= smallSizeThreshold {
            return smallSizeContentionFactor
        }
        if totalElements >= largeSizeThreshold {
            return largeSizeContentionFactor
        }
        // Linear interpolation between thresholds
        let t = Double(totalElements - smallSizeThreshold) /
                Double(largeSizeThreshold - smallSizeThreshold)
        return smallSizeContentionFactor + t * (largeSizeContentionFactor - smallSizeContentionFactor)
    }

    /// Adjust an isolated time estimate for contention.
    ///
    /// Contention means each unit runs slower → time increases.
    /// adjustedTime = isolatedTime / contentionFactor
    public func adjustedTime(isolatedTimeMs: Double, totalElements: Int) -> Double {
        let factor = contentionFactor(totalElements: totalElements)
        guard factor > 0 else { return isolatedTimeMs }
        return isolatedTimeMs / factor
    }
}
