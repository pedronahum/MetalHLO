// BenchmarkResult.swift
// MetalHLO Benchmarks
//
// Core types for storing and analyzing benchmark results.

import Foundation

/// Configuration for running a benchmark.
public struct BenchmarkConfig: Sendable {
    /// Number of warmup iterations (discarded).
    public let warmupIterations: Int

    /// Number of measurement iterations.
    public let measurementIterations: Int

    /// Whether to clear caches between runs.
    public let clearCachesBetweenRuns: Bool

    /// Whether to synchronize GPU before timing.
    public let gpuSync: Bool

    public init(
        warmupIterations: Int = 10,
        measurementIterations: Int = 100,
        clearCachesBetweenRuns: Bool = false,
        gpuSync: Bool = true
    ) {
        self.warmupIterations = warmupIterations
        self.measurementIterations = measurementIterations
        self.clearCachesBetweenRuns = clearCachesBetweenRuns
        self.gpuSync = gpuSync
    }

    /// Quick benchmark config for development.
    public static let quick = BenchmarkConfig(
        warmupIterations: 3,
        measurementIterations: 10
    )

    /// Standard benchmark config.
    public static let standard = BenchmarkConfig(
        warmupIterations: 10,
        measurementIterations: 100
    )

    /// Thorough benchmark config for final measurements.
    public static let thorough = BenchmarkConfig(
        warmupIterations: 20,
        measurementIterations: 1000
    )
}

/// Statistical summary of timing measurements.
public struct TimingStatistics: Sendable, Codable {
    /// All raw measurements in seconds.
    public let samples: [Double]

    /// Number of samples.
    public var count: Int { samples.count }

    /// Minimum value.
    public let min: Double

    /// Maximum value.
    public let max: Double

    /// Arithmetic mean.
    public let mean: Double

    /// Standard deviation.
    public let stdDev: Double

    /// Median (p50).
    public let median: Double

    /// 95th percentile.
    public let p95: Double

    /// 99th percentile.
    public let p99: Double

    /// Coefficient of variation (stdDev / mean).
    public var coefficientOfVariation: Double {
        mean > 0 ? stdDev / mean : 0
    }

    public init(samples: [Double]) {
        self.samples = samples

        guard !samples.isEmpty else {
            self.min = 0
            self.max = 0
            self.mean = 0
            self.stdDev = 0
            self.median = 0
            self.p95 = 0
            self.p99 = 0
            return
        }

        let sorted = samples.sorted()
        self.min = sorted.first!
        self.max = sorted.last!

        let sum = samples.reduce(0, +)
        let calculatedMean = sum / Double(samples.count)
        self.mean = calculatedMean

        let squaredDiffs = samples.map { ($0 - calculatedMean) * ($0 - calculatedMean) }
        let variance = squaredDiffs.reduce(0, +) / Double(samples.count)
        self.stdDev = sqrt(variance)

        self.median = Self.percentile(sorted, 0.50)
        self.p95 = Self.percentile(sorted, 0.95)
        self.p99 = Self.percentile(sorted, 0.99)
    }

    private static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let index = p * Double(sorted.count - 1)
        let lower = Int(floor(index))
        let upper = Int(ceil(index))
        if lower == upper {
            return sorted[lower]
        }
        let fraction = index - Double(lower)
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction
    }
}

/// Result of a single benchmark run.
public struct BenchmarkResult: Sendable, Codable {
    /// Unique identifier for this benchmark.
    public let id: String

    /// Human-readable name.
    public let name: String

    /// Category (e.g., "arithmetic", "matrix", "reduction").
    public let category: String

    /// Operation being benchmarked.
    public let operation: String

    /// Configuration parameters (shapes, data types, etc.).
    public let configuration: [String: String]

    /// GPU execution time statistics.
    public let gpuTime: TimingStatistics

    /// Total wall-clock time statistics.
    public let totalTime: TimingStatistics

    /// Encode time statistics (if available).
    public let encodeTime: TimingStatistics?

    /// Peak memory usage in bytes (if measured).
    public let peakMemoryBytes: Int64?

    /// Throughput metrics.
    public let throughput: ThroughputMetrics?

    /// Timestamp when benchmark was run.
    public let timestamp: Date

    /// Hardware info.
    public let hardwareInfo: HardwareInfo

    public init(
        id: String,
        name: String,
        category: String,
        operation: String,
        configuration: [String: String],
        gpuTime: TimingStatistics,
        totalTime: TimingStatistics,
        encodeTime: TimingStatistics? = nil,
        peakMemoryBytes: Int64? = nil,
        throughput: ThroughputMetrics? = nil,
        timestamp: Date = Date(),
        hardwareInfo: HardwareInfo = .current()
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.operation = operation
        self.configuration = configuration
        self.gpuTime = gpuTime
        self.totalTime = totalTime
        self.encodeTime = encodeTime
        self.peakMemoryBytes = peakMemoryBytes
        self.throughput = throughput
        self.timestamp = timestamp
        self.hardwareInfo = hardwareInfo
    }
}

/// Throughput metrics for a benchmark.
public struct ThroughputMetrics: Sendable, Codable {
    /// Operations per second.
    public let opsPerSecond: Double

    /// FLOPS achieved (if applicable).
    public let flops: Double?

    /// Memory bandwidth achieved in GB/s (if applicable).
    public let memoryBandwidthGBps: Double?

    /// Elements processed per second.
    public let elementsPerSecond: Double?

    public init(
        opsPerSecond: Double,
        flops: Double? = nil,
        memoryBandwidthGBps: Double? = nil,
        elementsPerSecond: Double? = nil
    ) {
        self.opsPerSecond = opsPerSecond
        self.flops = flops
        self.memoryBandwidthGBps = memoryBandwidthGBps
        self.elementsPerSecond = elementsPerSecond
    }
}

/// Hardware information for benchmark context.
public struct HardwareInfo: Sendable, Codable {
    /// Chip name (e.g., "Apple M3 Max").
    public let chipName: String

    /// Number of GPU cores.
    public let gpuCores: Int?

    /// Total memory in bytes.
    public let totalMemoryBytes: Int64

    /// macOS version.
    public let osVersion: String

    public init(
        chipName: String,
        gpuCores: Int? = nil,
        totalMemoryBytes: Int64,
        osVersion: String
    ) {
        self.chipName = chipName
        self.gpuCores = gpuCores
        self.totalMemoryBytes = totalMemoryBytes
        self.osVersion = osVersion
    }

    /// Get current hardware info.
    public static func current() -> HardwareInfo {
        let processInfo = ProcessInfo.processInfo

        // Get chip name via sysctl
        var chipName = "Unknown"
        var size: Int = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        if size > 0 {
            var buffer = [CChar](repeating: 0, count: size)
            sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
            chipName = String(cString: buffer)
        }

        // Fallback to hw.model for Apple Silicon
        if chipName == "Unknown" || chipName.isEmpty {
            size = 0
            sysctlbyname("hw.model", nil, &size, nil, 0)
            if size > 0 {
                var buffer = [CChar](repeating: 0, count: size)
                sysctlbyname("hw.model", &buffer, &size, nil, 0)
                chipName = String(cString: buffer)
            }
        }

        let totalMemory = Int64(processInfo.physicalMemory)
        let osVersion = processInfo.operatingSystemVersionString

        return HardwareInfo(
            chipName: chipName,
            gpuCores: nil,  // Would need Metal device query
            totalMemoryBytes: totalMemory,
            osVersion: osVersion
        )
    }
}

/// A collection of benchmark results forming a report.
public struct BenchmarkReport: Sendable, Codable {
    /// Report title.
    public let title: String

    /// When the benchmark suite was run.
    public let date: Date

    /// Hardware context.
    public let hardwareInfo: HardwareInfo

    /// All benchmark results.
    public let results: [BenchmarkResult]

    /// MetalHLO version (if available).
    public let metalHLOVersion: String?

    public init(
        title: String,
        date: Date = Date(),
        hardwareInfo: HardwareInfo = .current(),
        results: [BenchmarkResult],
        metalHLOVersion: String? = nil
    ) {
        self.title = title
        self.date = date
        self.hardwareInfo = hardwareInfo
        self.results = results
        self.metalHLOVersion = metalHLOVersion
    }

    /// Generate JSON representation.
    public func toJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }

    /// Generate Markdown report.
    public func toMarkdown() -> String {
        var md = """
        # \(title)

        **Date:** \(ISO8601DateFormatter().string(from: date))
        **Hardware:** \(hardwareInfo.chipName)
        **Memory:** \(hardwareInfo.totalMemoryBytes / (1024 * 1024 * 1024)) GB
        **OS:** \(hardwareInfo.osVersion)

        ## Summary

        - Total benchmarks: \(results.count)

        ## Results by Category

        """

        // Group by category
        let byCategory = Dictionary(grouping: results) { $0.category }
        for (category, categoryResults) in byCategory.sorted(by: { $0.key < $1.key }) {
            md += "\n### \(category.capitalized)\n\n"
            md += "| Benchmark | Operation | Mean (ms) | Std Dev | P95 (ms) | P99 (ms) |\n"
            md += "|-----------|-----------|-----------|---------|----------|----------|\n"

            for result in categoryResults.sorted(by: { $0.id < $1.id }) {
                let meanMs = result.gpuTime.mean * 1000
                let stdMs = result.gpuTime.stdDev * 1000
                let p95Ms = result.gpuTime.p95 * 1000
                let p99Ms = result.gpuTime.p99 * 1000
                md += "| \(result.id) | \(result.operation) | \(String(format: "%.3f", meanMs)) | \(String(format: "%.3f", stdMs)) | \(String(format: "%.3f", p95Ms)) | \(String(format: "%.3f", p99Ms)) |\n"
            }
        }

        return md
    }

    /// Compare with a baseline report.
    public func compare(with baseline: BenchmarkReport) -> ComparisonReport {
        var comparisons: [BenchmarkComparison] = []

        let baselineByID = Dictionary(uniqueKeysWithValues: baseline.results.map { ($0.id, $0) })

        for result in results {
            if let baselineResult = baselineByID[result.id] {
                let speedup = baselineResult.gpuTime.mean / result.gpuTime.mean
                comparisons.append(BenchmarkComparison(
                    benchmarkID: result.id,
                    currentMean: result.gpuTime.mean,
                    baselineMean: baselineResult.gpuTime.mean,
                    speedup: speedup,
                    regression: speedup < 0.95  // 5% slower = regression
                ))
            }
        }

        return ComparisonReport(
            currentReport: self,
            baselineReport: baseline,
            comparisons: comparisons
        )
    }
}

/// Comparison between two benchmark results.
public struct BenchmarkComparison: Sendable, Codable {
    public let benchmarkID: String
    public let currentMean: Double
    public let baselineMean: Double
    public let speedup: Double
    public let regression: Bool
}

/// Report comparing current results against a baseline.
public struct ComparisonReport: Sendable, Codable {
    public let currentReport: BenchmarkReport
    public let baselineReport: BenchmarkReport
    public let comparisons: [BenchmarkComparison]

    public var improvements: [BenchmarkComparison] {
        comparisons.filter { $0.speedup > 1.05 }
    }

    public var regressions: [BenchmarkComparison] {
        comparisons.filter { $0.regression }
    }

    public func toMarkdown() -> String {
        var md = """
        # Benchmark Comparison Report

        **Current:** \(currentReport.date)
        **Baseline:** \(baselineReport.date)

        ## Summary

        - Benchmarks compared: \(comparisons.count)
        - Improvements (>5% faster): \(improvements.count)
        - Regressions (>5% slower): \(regressions.count)

        """

        if !improvements.isEmpty {
            md += "\n## Improvements\n\n"
            md += "| Benchmark | Speedup | Current (ms) | Baseline (ms) |\n"
            md += "|-----------|---------|--------------|---------------|\n"
            for comp in improvements.sorted(by: { $0.speedup > $1.speedup }) {
                md += "| \(comp.benchmarkID) | \(String(format: "%.2fx", comp.speedup)) | \(String(format: "%.3f", comp.currentMean * 1000)) | \(String(format: "%.3f", comp.baselineMean * 1000)) |\n"
            }
        }

        if !regressions.isEmpty {
            md += "\n## Regressions\n\n"
            md += "| Benchmark | Slowdown | Current (ms) | Baseline (ms) |\n"
            md += "|-----------|----------|--------------|---------------|\n"
            for comp in regressions.sorted(by: { $0.speedup < $1.speedup }) {
                md += "| \(comp.benchmarkID) | \(String(format: "%.2fx", 1.0 / comp.speedup)) | \(String(format: "%.3f", comp.currentMean * 1000)) | \(String(format: "%.3f", comp.baselineMean * 1000)) |\n"
            }
        }

        return md
    }
}
