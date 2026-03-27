// HardwareProfile.swift
// HeterogeneousFusion
//
// Self-characterizing hardware constants for the compound profitability gate.
// Derived once per device via calibrate(), cached to disk for 30 days.
//
// The five constants govern all gate decisions in ProfitabilityGuard:
//   1. minOutputElements  — minimum M*N for partitioning (10M on M1)
//   2. minOutputColumns   — minimum N for profitability (32768 on M1)
//   3. maxOutputBytes     — memory pressure cliff (25% RAM on M1)
//   4. contentionCeiling  — max achievable fused speedup (1.5-1.9x on M1)
//   5. concurrencyBudget  — max simultaneous fused ops (1 on M1)

import Foundation
import Metal

/// Hardware-specific constants derived by on-device calibration.
///
/// These constants are properties of the chip's memory bandwidth, dispatch
/// overhead, and thermal characteristics — not of any particular model.
/// They transfer across architectures (GPT-2, ViT, etc.) because they
/// describe when the hardware can profitably split work, not what work to split.
public struct HardwareProfile: Codable, Sendable {

    /// MTLDevice.name — used as cache key to detect hardware changes.
    public let deviceName: String

    /// When this profile was calibrated.
    public let calibrationDate: Date

    /// Minimum total output elements (M*N) for partitioning to be profitable.
    /// Below this, dispatch overhead and contention dominate parallelism.
    public let minOutputElements: Int

    /// Minimum output columns (N) for partitioning to amortize split/sync overhead.
    /// Large N means more independent work per row; small N means threads finish
    /// fast and contention dominates regardless of total element count.
    public let minOutputColumns: Int

    /// Maximum output buffer bytes before unified memory pressure causes thrashing.
    /// Concurrent 3-unit execution triples working set; exceeding this fraction of
    /// physical RAM causes catastrophic slowdown (0.05x on M1 8GB).
    public let maxOutputBytes: Int

    /// Maximum observed fused speedup across all profitable shapes.
    /// Used by the solver's contention model to cap predictions.
    /// Not a gate constant — an expectation ceiling.
    public let contentionCeiling: Double

    /// Maximum simultaneous fused ops before cross-op bandwidth contention
    /// causes net regression. 1 = fully serialize between clusters.
    public let concurrencyBudget: Int

    public init(
        deviceName: String,
        calibrationDate: Date,
        minOutputElements: Int,
        minOutputColumns: Int,
        maxOutputBytes: Int,
        contentionCeiling: Double,
        concurrencyBudget: Int
    ) {
        self.deviceName = deviceName
        self.calibrationDate = calibrationDate
        self.minOutputElements = minOutputElements
        self.minOutputColumns = minOutputColumns
        self.maxOutputBytes = maxOutputBytes
        self.contentionCeiling = contentionCeiling
        self.concurrencyBudget = concurrencyBudget
    }
}

// MARK: - Disk Cache

extension HardwareProfile {

    private static var cacheURL: URL {
        let support = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return support
            .appendingPathComponent("MetalHLO")
            .appendingPathComponent("hardware_profile.json")
    }

    /// Load cached profile for the given device, or calibrate if missing/expired.
    public static func load(device: MTLDevice) -> HardwareProfile {
        if let cached = loadFromDisk(device: device) {
            return cached
        }
        let profile = calibrate(device: device)
        saveToDisk(profile)
        return profile
    }

    private static func loadFromDisk(device: MTLDevice) -> HardwareProfile? {
        guard let data = try? Data(contentsOf: cacheURL),
              let profile = try? JSONDecoder().decode(HardwareProfile.self, from: data),
              profile.deviceName == device.name else {
            return nil
        }

        // Re-calibrate if cache is older than 30 days
        let age = Date().timeIntervalSince(profile.calibrationDate)
        guard age < 30 * 24 * 3600 else {
            print("[MetalHLO] Hardware profile expired -- recalibrating.")
            return nil
        }

        return profile
    }

    private static func saveToDisk(_ profile: HardwareProfile) {
        let dir = cacheURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(profile) {
            try? data.write(to: cacheURL)
        }
    }

    /// Force recalibration on next load.
    public static func invalidateCache() {
        try? FileManager.default.removeItem(at: cacheURL)
    }
}
