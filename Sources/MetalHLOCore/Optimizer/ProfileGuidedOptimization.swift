// ProfileGuidedOptimization.swift
// Profile-Guided Optimization for MetalHLO
// Collects runtime profiling data to guide kernel optimization decisions

import Foundation
import Metal

// MARK: - Profile Data Types

/// Represents a single kernel execution sample
public struct KernelExecutionSample: Sendable, Codable {
    public let kernelId: String
    public let executionTimeNs: UInt64
    public let inputSizes: [Int]
    public let timestamp: Date
    public let threadgroupSize: (Int, Int, Int)
    public let gridSize: (Int, Int, Int)

    public init(
        kernelId: String,
        executionTimeNs: UInt64,
        inputSizes: [Int],
        timestamp: Date = Date(),
        threadgroupSize: (Int, Int, Int) = (1, 1, 1),
        gridSize: (Int, Int, Int) = (1, 1, 1)
    ) {
        self.kernelId = kernelId
        self.executionTimeNs = executionTimeNs
        self.inputSizes = inputSizes
        self.timestamp = timestamp
        self.threadgroupSize = threadgroupSize
        self.gridSize = gridSize
    }

    // Custom Codable for tuples
    enum CodingKeys: String, CodingKey {
        case kernelId, executionTimeNs, inputSizes, timestamp
        case threadgroupSizeX, threadgroupSizeY, threadgroupSizeZ
        case gridSizeX, gridSizeY, gridSizeZ
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        kernelId = try container.decode(String.self, forKey: .kernelId)
        executionTimeNs = try container.decode(UInt64.self, forKey: .executionTimeNs)
        inputSizes = try container.decode([Int].self, forKey: .inputSizes)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        threadgroupSize = (
            try container.decode(Int.self, forKey: .threadgroupSizeX),
            try container.decode(Int.self, forKey: .threadgroupSizeY),
            try container.decode(Int.self, forKey: .threadgroupSizeZ)
        )
        gridSize = (
            try container.decode(Int.self, forKey: .gridSizeX),
            try container.decode(Int.self, forKey: .gridSizeY),
            try container.decode(Int.self, forKey: .gridSizeZ)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(kernelId, forKey: .kernelId)
        try container.encode(executionTimeNs, forKey: .executionTimeNs)
        try container.encode(inputSizes, forKey: .inputSizes)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(threadgroupSize.0, forKey: .threadgroupSizeX)
        try container.encode(threadgroupSize.1, forKey: .threadgroupSizeY)
        try container.encode(threadgroupSize.2, forKey: .threadgroupSizeZ)
        try container.encode(gridSize.0, forKey: .gridSizeX)
        try container.encode(gridSize.1, forKey: .gridSizeY)
        try container.encode(gridSize.2, forKey: .gridSizeZ)
    }
}

/// Aggregated statistics for a kernel
public struct KernelProfile: Sendable {
    public let kernelId: String
    public var executionCount: Int
    public var totalTimeNs: UInt64
    public var minTimeNs: UInt64
    public var maxTimeNs: UInt64
    public var samples: [KernelExecutionSample]

    public var averageTimeNs: Double {
        guard executionCount > 0 else { return 0 }
        return Double(totalTimeNs) / Double(executionCount)
    }

    public var standardDeviationNs: Double {
        guard samples.count > 1 else { return 0 }
        let avg = averageTimeNs
        let variance = samples.map { pow(Double($0.executionTimeNs) - avg, 2) }.reduce(0, +) / Double(samples.count - 1)
        return sqrt(variance)
    }

    public init(kernelId: String) {
        self.kernelId = kernelId
        self.executionCount = 0
        self.totalTimeNs = 0
        self.minTimeNs = .max
        self.maxTimeNs = 0
        self.samples = []
    }

    public mutating func addSample(_ sample: KernelExecutionSample) {
        executionCount += 1
        totalTimeNs += sample.executionTimeNs
        minTimeNs = min(minTimeNs, sample.executionTimeNs)
        maxTimeNs = max(maxTimeNs, sample.executionTimeNs)
        samples.append(sample)
    }
}

/// Complete profile data for a compilation unit
public struct ProfileData: Sendable {
    public var kernelProfiles: [String: KernelProfile]
    public var totalExecutions: Int
    public var profilingStartTime: Date
    public var profilingEndTime: Date?

    public init() {
        self.kernelProfiles = [:]
        self.totalExecutions = 0
        self.profilingStartTime = Date()
        self.profilingEndTime = nil
    }

    public var totalTimeNs: UInt64 {
        kernelProfiles.values.reduce(0) { $0 + $1.totalTimeNs }
    }

    /// Returns kernels sorted by total execution time (descending)
    public var hotKernels: [KernelProfile] {
        kernelProfiles.values.sorted { $0.totalTimeNs > $1.totalTimeNs }
    }
}

// MARK: - Profile Collector

/// Collects runtime profiling data for kernel executions
public final class ProfileCollector: @unchecked Sendable {
    private var profileData: ProfileData
    private let lock = NSLock()
    private let maxSamplesPerKernel: Int
    private let samplingRate: Double // 0.0 to 1.0
    private var isEnabled: Bool

    public init(maxSamplesPerKernel: Int = 1000, samplingRate: Double = 1.0) {
        self.profileData = ProfileData()
        self.maxSamplesPerKernel = maxSamplesPerKernel
        self.samplingRate = min(1.0, max(0.0, samplingRate))
        self.isEnabled = true
    }

    /// Records a kernel execution sample
    public func recordExecution(_ sample: KernelExecutionSample) {
        guard isEnabled else { return }

        // Apply sampling rate
        if samplingRate < 1.0 && Double.random(in: 0...1) > samplingRate {
            return
        }

        lock.lock()
        defer { lock.unlock() }

        profileData.totalExecutions += 1

        if var profile = profileData.kernelProfiles[sample.kernelId] {
            // Limit samples to prevent memory bloat
            if profile.samples.count >= maxSamplesPerKernel {
                // Remove oldest sample
                profile.samples.removeFirst()
            }
            profile.addSample(sample)
            profileData.kernelProfiles[sample.kernelId] = profile
        } else {
            var profile = KernelProfile(kernelId: sample.kernelId)
            profile.addSample(sample)
            profileData.kernelProfiles[sample.kernelId] = profile
        }
    }

    /// Records execution with individual parameters
    public func recordExecution(
        kernelId: String,
        executionTimeNs: UInt64,
        inputSizes: [Int],
        threadgroupSize: (Int, Int, Int) = (1, 1, 1),
        gridSize: (Int, Int, Int) = (1, 1, 1)
    ) {
        let sample = KernelExecutionSample(
            kernelId: kernelId,
            executionTimeNs: executionTimeNs,
            inputSizes: inputSizes,
            threadgroupSize: threadgroupSize,
            gridSize: gridSize
        )
        recordExecution(sample)
    }

    /// Gets the current profile data snapshot
    public func getProfileData() -> ProfileData {
        lock.lock()
        defer { lock.unlock() }
        var data = profileData
        data.profilingEndTime = Date()
        return data
    }

    /// Resets all collected profile data
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        profileData = ProfileData()
    }

    /// Enables or disables profiling
    public func setEnabled(_ enabled: Bool) {
        lock.lock()
        defer { lock.unlock() }
        isEnabled = enabled
    }

    /// Exports profile data to a file
    public func exportProfile(to url: URL) throws {
        let data = getProfileData()
        let samples = data.kernelProfiles.values.flatMap { $0.samples }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        let jsonData = try encoder.encode(samples)
        try jsonData.write(to: url)
    }

    /// Imports profile data from a file
    public func importProfile(from url: URL) throws {
        let jsonData = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let samples = try decoder.decode([KernelExecutionSample].self, from: jsonData)

        lock.lock()
        defer { lock.unlock() }

        for sample in samples {
            if var profile = profileData.kernelProfiles[sample.kernelId] {
                profile.addSample(sample)
                profileData.kernelProfiles[sample.kernelId] = profile
            } else {
                var profile = KernelProfile(kernelId: sample.kernelId)
                profile.addSample(sample)
                profileData.kernelProfiles[sample.kernelId] = profile
            }
        }
    }
}

// MARK: - Hot Kernel Analyzer

/// Identifies hot kernels that should be prioritized for optimization
public final class HotKernelAnalyzer: @unchecked Sendable {

    /// Threshold for considering a kernel "hot"
    public struct HotKernelThresholds: Sendable {
        public var minExecutionCount: Int
        public var minTotalTimePercent: Double
        public var minAverageTimeNs: UInt64

        public static let `default` = HotKernelThresholds(
            minExecutionCount: 10,
            minTotalTimePercent: 1.0, // 1% of total time
            minAverageTimeNs: 1000 // 1 microsecond
        )

        public init(minExecutionCount: Int, minTotalTimePercent: Double, minAverageTimeNs: UInt64) {
            self.minExecutionCount = minExecutionCount
            self.minTotalTimePercent = minTotalTimePercent
            self.minAverageTimeNs = minAverageTimeNs
        }
    }

    /// Result of hot kernel analysis
    public struct HotKernelAnalysis: Sendable {
        public let hotKernelIds: [String]
        public let kernelRankings: [(kernelId: String, score: Double)]
        public let totalHotTimePercent: Double
        public let optimizationPotential: Double

        public init(
            hotKernelIds: [String],
            kernelRankings: [(kernelId: String, score: Double)],
            totalHotTimePercent: Double,
            optimizationPotential: Double
        ) {
            self.hotKernelIds = hotKernelIds
            self.kernelRankings = kernelRankings
            self.totalHotTimePercent = totalHotTimePercent
            self.optimizationPotential = optimizationPotential
        }
    }

    private let thresholds: HotKernelThresholds

    public init(thresholds: HotKernelThresholds = .default) {
        self.thresholds = thresholds
    }

    /// Analyzes profile data to identify hot kernels
    public func analyze(_ profileData: ProfileData) -> HotKernelAnalysis {
        let totalTime = profileData.totalTimeNs
        guard totalTime > 0 else {
            return HotKernelAnalysis(
                hotKernelIds: [],
                kernelRankings: [],
                totalHotTimePercent: 0,
                optimizationPotential: 0
            )
        }

        // Calculate scores for each kernel
        var rankings: [(kernelId: String, score: Double)] = []

        for (kernelId, profile) in profileData.kernelProfiles {
            let timePercent = Double(profile.totalTimeNs) / Double(totalTime) * 100

            // Score combines execution count, time percentage, and variability
            // High variability suggests optimization potential
            let variabilityFactor = profile.standardDeviationNs / max(1, profile.averageTimeNs)
            let score = timePercent * (1 + variabilityFactor * 0.1) * log(Double(profile.executionCount + 1))

            rankings.append((kernelId, score))
        }

        // Sort by score descending
        rankings.sort { $0.score > $1.score }

        // Identify hot kernels based on thresholds
        var hotKernelIds: [String] = []
        var hotTimeTotal: UInt64 = 0

        for (kernelId, _) in rankings {
            guard let profile = profileData.kernelProfiles[kernelId] else { continue }

            let timePercent = Double(profile.totalTimeNs) / Double(totalTime) * 100

            let isHot = profile.executionCount >= thresholds.minExecutionCount &&
                        timePercent >= thresholds.minTotalTimePercent &&
                        UInt64(profile.averageTimeNs) >= thresholds.minAverageTimeNs

            if isHot {
                hotKernelIds.append(kernelId)
                hotTimeTotal += profile.totalTimeNs
            }
        }

        let hotTimePercent = Double(hotTimeTotal) / Double(totalTime) * 100

        // Estimate optimization potential based on kernel variability
        // High variability suggests room for improvement
        var potentialScore = 0.0
        for kernelId in hotKernelIds {
            if let profile = profileData.kernelProfiles[kernelId] {
                let variability = profile.standardDeviationNs / max(1, profile.averageTimeNs)
                let timeWeight = Double(profile.totalTimeNs) / Double(totalTime)
                potentialScore += variability * timeWeight
            }
        }

        return HotKernelAnalysis(
            hotKernelIds: hotKernelIds,
            kernelRankings: rankings,
            totalHotTimePercent: hotTimePercent,
            optimizationPotential: min(1.0, potentialScore)
        )
    }

    /// Returns the top N kernels by execution time
    public func topKernels(_ profileData: ProfileData, count: Int) -> [KernelProfile] {
        return Array(profileData.hotKernels.prefix(count))
    }
}

// MARK: - PGO Recompilation

/// Optimization hints derived from profile data
public struct OptimizationHints: Sendable {
    public let kernelId: String
    public var preferredThreadgroupSize: (Int, Int, Int)?
    public var preferredTileSize: Int?
    public var shouldUnroll: Bool
    public var shouldVectorize: Bool
    public var expectedInputSizes: [Int]?
    public var memoryAccessPattern: MemoryAccessPattern
    public var recommendedOptimizations: [RecommendedOptimization]

    public enum MemoryAccessPattern: String, Sendable {
        case sequential
        case strided
        case random
        case unknown
    }

    public enum RecommendedOptimization: String, Sendable {
        case specialization
        case tiling
        case vectorization
        case unrolling
        case fusion
        case caching
        case prefetching
    }

    public init(kernelId: String) {
        self.kernelId = kernelId
        self.preferredThreadgroupSize = nil
        self.preferredTileSize = nil
        self.shouldUnroll = false
        self.shouldVectorize = false
        self.expectedInputSizes = nil
        self.memoryAccessPattern = .unknown
        self.recommendedOptimizations = []
    }
}

/// Profile-Guided Optimizer that uses collected profile data to improve code generation
public final class ProfileGuidedOptimizer: @unchecked Sendable {

    public struct Config: Sendable {
        public var enableSpecialization: Bool
        public var enableTileOptimization: Bool
        public var enableUnrolling: Bool
        public var minSamplesForOptimization: Int
        public var confidenceThreshold: Double // 0.0 to 1.0

        public static let `default` = Config(
            enableSpecialization: true,
            enableTileOptimization: true,
            enableUnrolling: true,
            minSamplesForOptimization: 5,
            confidenceThreshold: 0.8
        )

        public init(
            enableSpecialization: Bool,
            enableTileOptimization: Bool,
            enableUnrolling: Bool,
            minSamplesForOptimization: Int,
            confidenceThreshold: Double
        ) {
            self.enableSpecialization = enableSpecialization
            self.enableTileOptimization = enableTileOptimization
            self.enableUnrolling = enableUnrolling
            self.minSamplesForOptimization = minSamplesForOptimization
            self.confidenceThreshold = confidenceThreshold
        }
    }

    private let config: Config
    private let analyzer: HotKernelAnalyzer

    public init(config: Config = .default) {
        self.config = config
        self.analyzer = HotKernelAnalyzer()
    }

    /// Generates optimization hints from profile data
    public func generateHints(for profileData: ProfileData) -> [String: OptimizationHints] {
        var hints: [String: OptimizationHints] = [:]

        let analysis = analyzer.analyze(profileData)

        for kernelId in analysis.hotKernelIds {
            guard let profile = profileData.kernelProfiles[kernelId] else { continue }
            guard profile.samples.count >= config.minSamplesForOptimization else { continue }

            var kernelHints = OptimizationHints(kernelId: kernelId)

            // Analyze threadgroup sizes
            if config.enableTileOptimization {
                kernelHints.preferredThreadgroupSize = analyzeOptimalThreadgroupSize(profile)
                kernelHints.preferredTileSize = analyzeOptimalTileSize(profile)
            }

            // Analyze input sizes for specialization
            if config.enableSpecialization {
                kernelHints.expectedInputSizes = analyzeCommonInputSizes(profile)
                if kernelHints.expectedInputSizes != nil {
                    kernelHints.recommendedOptimizations.append(.specialization)
                }
            }

            // Determine if unrolling is beneficial
            if config.enableUnrolling && shouldEnableUnrolling(profile) {
                kernelHints.shouldUnroll = true
                kernelHints.recommendedOptimizations.append(.unrolling)
            }

            // Analyze memory access patterns
            kernelHints.memoryAccessPattern = analyzeMemoryPattern(profile)

            // Add vectorization recommendation for sequential access
            if kernelHints.memoryAccessPattern == .sequential {
                kernelHints.shouldVectorize = true
                kernelHints.recommendedOptimizations.append(.vectorization)
            }

            hints[kernelId] = kernelHints
        }

        return hints
    }

    /// Analyzes samples to find optimal threadgroup size
    private func analyzeOptimalThreadgroupSize(_ profile: KernelProfile) -> (Int, Int, Int)? {
        // Group samples by threadgroup size
        var sizeToTime: [String: (count: Int, totalTime: UInt64)] = [:]

        for sample in profile.samples {
            let key = "\(sample.threadgroupSize.0)x\(sample.threadgroupSize.1)x\(sample.threadgroupSize.2)"
            var entry = sizeToTime[key] ?? (0, 0)
            entry.count += 1
            entry.totalTime += sample.executionTimeNs
            sizeToTime[key] = entry
        }

        // Find configuration with lowest average time
        var bestKey: String?
        var bestAvg: Double = .infinity

        for (key, entry) in sizeToTime {
            let avg = Double(entry.totalTime) / Double(entry.count)
            if avg < bestAvg && entry.count >= 3 { // Require minimum samples
                bestAvg = avg
                bestKey = key
            }
        }

        if let key = bestKey {
            let parts = key.split(separator: "x").compactMap { Int($0) }
            if parts.count == 3 {
                return (parts[0], parts[1], parts[2])
            }
        }

        return nil
    }

    /// Analyzes samples to find optimal tile size
    private func analyzeOptimalTileSize(_ profile: KernelProfile) -> Int? {
        // For now, derive from threadgroup size
        if let tg = analyzeOptimalThreadgroupSize(profile) {
            return max(tg.0, tg.1, tg.2)
        }
        return nil
    }

    /// Analyzes common input sizes for specialization
    private func analyzeCommonInputSizes(_ profile: KernelProfile) -> [Int]? {
        guard !profile.samples.isEmpty else { return nil }

        // Count frequency of each input size configuration
        var sizeCounts: [[Int]: Int] = [:]

        for sample in profile.samples {
            sizeCounts[sample.inputSizes, default: 0] += 1
        }

        // Find the most common configuration
        let sorted = sizeCounts.sorted { $0.value > $1.value }

        if let (commonSizes, count) = sorted.first {
            let frequency = Double(count) / Double(profile.samples.count)
            if frequency >= config.confidenceThreshold {
                return commonSizes
            }
        }

        return nil
    }

    /// Determines if unrolling would be beneficial
    private func shouldEnableUnrolling(_ profile: KernelProfile) -> Bool {
        // Enable unrolling for:
        // 1. Small, consistent input sizes
        // 2. Low execution time variability (already well-optimized)

        let variability = profile.standardDeviationNs / max(1, profile.averageTimeNs)

        // High variability suggests unrolling might help
        return variability > 0.3
    }

    /// Analyzes memory access patterns from execution data
    private func analyzeMemoryPattern(_ profile: KernelProfile) -> OptimizationHints.MemoryAccessPattern {
        // In a real implementation, this would analyze actual memory traces
        // For now, use heuristics based on kernel ID patterns
        let kernelId = profile.kernelId.lowercased()

        if kernelId.contains("matmul") || kernelId.contains("gemm") {
            return .strided
        } else if kernelId.contains("elementwise") || kernelId.contains("add") || kernelId.contains("mul") {
            return .sequential
        } else if kernelId.contains("gather") || kernelId.contains("scatter") {
            return .random
        }

        return .unknown
    }

    /// Applies PGO hints to optimize a function
    public func optimize(_ function: HLOFunction, hints: [String: OptimizationHints]) -> OptimizedFunction {
        var optimizedOps: [OptimizedOperation] = []

        for op in function.operations {
            var optimizedOp = OptimizedOperation(operation: op)

            // Check if we have hints for this operation
            let opId = "\(function.name).\(op.kind)"
            if let opHints = hints[opId] {
                optimizedOp.hints = opHints
                optimizedOp.wasOptimized = true
            }

            optimizedOps.append(optimizedOp)
        }

        return OptimizedFunction(
            originalFunction: function,
            optimizedOperations: optimizedOps,
            appliedHints: hints
        )
    }
}

/// Wrapper for an optimized operation
public struct OptimizedOperation: Sendable {
    public let operation: HLOOperation
    public var hints: OptimizationHints?
    public var wasOptimized: Bool

    public init(operation: HLOOperation) {
        self.operation = operation
        self.hints = nil
        self.wasOptimized = false
    }
}

/// Result of PGO optimization
public struct OptimizedFunction: Sendable {
    public let originalFunction: HLOFunction
    public let optimizedOperations: [OptimizedOperation]
    public let appliedHints: [String: OptimizationHints]

    public var optimizationCount: Int {
        optimizedOperations.filter { $0.wasOptimized }.count
    }
}

// MARK: - Instrumentation Support

/// Provides instrumentation for kernel execution timing
public final class KernelInstrumentor: @unchecked Sendable {
    private let collector: ProfileCollector
    private var activeTimers: [String: UInt64]
    private let lock = NSLock()

    public init(collector: ProfileCollector) {
        self.collector = collector
        self.activeTimers = [:]
    }

    /// Starts timing a kernel execution
    public func startTiming(kernelId: String) {
        lock.lock()
        defer { lock.unlock() }
        activeTimers[kernelId] = DispatchTime.now().uptimeNanoseconds
    }

    /// Ends timing and records the sample
    public func endTiming(
        kernelId: String,
        inputSizes: [Int],
        threadgroupSize: (Int, Int, Int) = (1, 1, 1),
        gridSize: (Int, Int, Int) = (1, 1, 1)
    ) {
        let endTime = DispatchTime.now().uptimeNanoseconds

        lock.lock()
        guard let startTime = activeTimers.removeValue(forKey: kernelId) else {
            lock.unlock()
            return
        }
        lock.unlock()

        let duration = endTime - startTime
        collector.recordExecution(
            kernelId: kernelId,
            executionTimeNs: duration,
            inputSizes: inputSizes,
            threadgroupSize: threadgroupSize,
            gridSize: gridSize
        )
    }

    /// Executes a closure with timing
    public func timed<T>(
        kernelId: String,
        inputSizes: [Int],
        threadgroupSize: (Int, Int, Int) = (1, 1, 1),
        gridSize: (Int, Int, Int) = (1, 1, 1),
        _ body: () throws -> T
    ) rethrows -> T {
        startTiming(kernelId: kernelId)
        defer {
            endTiming(
                kernelId: kernelId,
                inputSizes: inputSizes,
                threadgroupSize: threadgroupSize,
                gridSize: gridSize
            )
        }
        return try body()
    }
}

// MARK: - Profile-Based Recompilation Controller

/// Controls when and how to trigger recompilation based on profile data
public final class RecompilationController: @unchecked Sendable {

    public struct Config: Sendable {
        public var recompilationThreshold: Int // Minimum executions before considering recompilation
        public var performanceGainThreshold: Double // Minimum expected speedup (e.g., 1.1 = 10%)
        public var maxRecompilations: Int // Maximum times to recompile same kernel
        public var cooldownExecutions: Int // Executions to wait after recompilation

        public static let `default` = Config(
            recompilationThreshold: 100,
            performanceGainThreshold: 1.1,
            maxRecompilations: 3,
            cooldownExecutions: 50
        )

        public init(
            recompilationThreshold: Int,
            performanceGainThreshold: Double,
            maxRecompilations: Int,
            cooldownExecutions: Int
        ) {
            self.recompilationThreshold = recompilationThreshold
            self.performanceGainThreshold = performanceGainThreshold
            self.maxRecompilations = maxRecompilations
            self.cooldownExecutions = cooldownExecutions
        }
    }

    public struct RecompilationDecision: Sendable {
        public let kernelId: String
        public let shouldRecompile: Bool
        public let reason: String
        public let expectedSpeedup: Double
        public let hints: OptimizationHints?
    }

    private let config: Config
    private var recompilationCounts: [String: Int]
    private var lastRecompilationExecution: [String: Int]
    private let lock = NSLock()

    public init(config: Config = .default) {
        self.config = config
        self.recompilationCounts = [:]
        self.lastRecompilationExecution = [:]
    }

    /// Determines if a kernel should be recompiled based on profile data
    public func shouldRecompile(
        kernelId: String,
        profile: KernelProfile,
        currentExecutionCount: Int
    ) -> RecompilationDecision {
        lock.lock()
        defer { lock.unlock() }

        let recompCount = recompilationCounts[kernelId] ?? 0
        let lastRecompExec = lastRecompilationExecution[kernelId] ?? 0

        // Check cooldown period
        if currentExecutionCount - lastRecompExec < config.cooldownExecutions {
            return RecompilationDecision(
                kernelId: kernelId,
                shouldRecompile: false,
                reason: "In cooldown period",
                expectedSpeedup: 1.0,
                hints: nil
            )
        }

        // Check max recompilations
        if recompCount >= config.maxRecompilations {
            return RecompilationDecision(
                kernelId: kernelId,
                shouldRecompile: false,
                reason: "Maximum recompilations reached",
                expectedSpeedup: 1.0,
                hints: nil
            )
        }

        // Check execution threshold
        if profile.executionCount < config.recompilationThreshold {
            return RecompilationDecision(
                kernelId: kernelId,
                shouldRecompile: false,
                reason: "Not enough executions for reliable profiling",
                expectedSpeedup: 1.0,
                hints: nil
            )
        }

        // Estimate potential speedup based on variability
        let variability = profile.standardDeviationNs / max(1, profile.averageTimeNs)
        let estimatedSpeedup = 1.0 + variability * 0.5 // Conservative estimate

        if estimatedSpeedup >= config.performanceGainThreshold {
            var hints = OptimizationHints(kernelId: kernelId)
            hints.recommendedOptimizations = [.specialization, .tiling]

            return RecompilationDecision(
                kernelId: kernelId,
                shouldRecompile: true,
                reason: "High performance variability detected",
                expectedSpeedup: estimatedSpeedup,
                hints: hints
            )
        }

        return RecompilationDecision(
            kernelId: kernelId,
            shouldRecompile: false,
            reason: "Expected speedup below threshold",
            expectedSpeedup: estimatedSpeedup,
            hints: nil
        )
    }

    /// Records that a recompilation was performed
    public func recordRecompilation(kernelId: String, executionCount: Int) {
        lock.lock()
        defer { lock.unlock() }

        recompilationCounts[kernelId] = (recompilationCounts[kernelId] ?? 0) + 1
        lastRecompilationExecution[kernelId] = executionCount
    }

    /// Resets recompilation tracking for a kernel
    public func resetKernel(_ kernelId: String) {
        lock.lock()
        defer { lock.unlock() }

        recompilationCounts.removeValue(forKey: kernelId)
        lastRecompilationExecution.removeValue(forKey: kernelId)
    }
}
