// BenchmarkRunner.swift
// MetalHLO Benchmarks
//
// Core harness for running benchmarks with proper warmup, measurement, and statistics.

import Foundation
import MetalHLO

/// Protocol for defining a benchmark.
public protocol Benchmark: Sendable {
    /// Unique identifier for this benchmark.
    var id: String { get }

    /// Human-readable name.
    var name: String { get }

    /// Category (e.g., "arithmetic", "matrix", "reduction").
    var category: String { get }

    /// Operation being benchmarked.
    var operation: String { get }

    /// Configuration parameters as key-value pairs.
    var configuration: [String: String] { get }

    /// The MLIR program to compile and run.
    var mlirProgram: String { get }

    /// Generate input buffers for this benchmark.
    func createInputs(client: Client) throws -> [Buffer]

    /// Calculate throughput metrics given timing (optional).
    func calculateThroughput(timing: ExecutionTiming) -> ThroughputMetrics?
}

/// Default implementation for throughput calculation.
extension Benchmark {
    public func calculateThroughput(timing: ExecutionTiming) -> ThroughputMetrics? {
        return nil
    }
}

/// The main benchmark runner that executes benchmarks and collects results.
public final class BenchmarkRunner: @unchecked Sendable {
    private let client: Client
    private let config: BenchmarkConfig

    /// Optimization level for compilation.
    public var optimizationLevel: OptimizationLevel = .O2

    /// Progress callback: (currentIndex, totalCount, benchmarkID)
    public var onProgress: ((Int, Int, String) -> Void)?

    /// Verbose output.
    public var verbose: Bool = false

    public init(config: BenchmarkConfig = .standard) throws {
        self.client = try Client.create()
        self.config = config
    }

    /// Run a single benchmark and return the result.
    public func run(_ benchmark: Benchmark) throws -> BenchmarkResult {
        if verbose {
            print("Running benchmark: \(benchmark.id)")
        }

        // Compile the program with the specified optimization level
        let compileConfig = CompilationConfig(optimizationLevel: optimizationLevel)
        let executable = try client.compile(benchmark.mlirProgram, config: compileConfig)

        // Create inputs
        let inputs = try benchmark.createInputs(client: client)

        // Warmup phase
        if verbose {
            print("  Warmup: \(config.warmupIterations) iterations")
        }
        for _ in 0..<config.warmupIterations {
            _ = try executable.execute(inputs)
        }

        // Measurement phase
        if verbose {
            print("  Measuring: \(config.measurementIterations) iterations")
        }
        var gpuTimes: [Double] = []
        var totalTimes: [Double] = []
        var encodeTimes: [Double] = []

        for _ in 0..<config.measurementIterations {
            let (_, timing) = try executable.executeWithTiming(inputs)
            gpuTimes.append(timing.gpuTime)
            totalTimes.append(timing.totalTime)
            encodeTimes.append(timing.encodeTime)
        }

        // Calculate statistics
        let gpuStats = TimingStatistics(samples: gpuTimes)
        let totalStats = TimingStatistics(samples: totalTimes)
        let encodeStats = TimingStatistics(samples: encodeTimes)

        // Calculate throughput if available
        let sampleTiming = ExecutionTiming(
            encodeTime: encodeStats.mean,
            gpuTime: gpuStats.mean,
            totalTime: totalStats.mean
        )
        let throughput = benchmark.calculateThroughput(timing: sampleTiming)

        if verbose {
            print("  Mean GPU time: \(String(format: "%.3f", gpuStats.mean * 1000)) ms")
            print("  Std dev: \(String(format: "%.3f", gpuStats.stdDev * 1000)) ms")
        }

        return BenchmarkResult(
            id: benchmark.id,
            name: benchmark.name,
            category: benchmark.category,
            operation: benchmark.operation,
            configuration: benchmark.configuration,
            gpuTime: gpuStats,
            totalTime: totalStats,
            encodeTime: encodeStats,
            peakMemoryBytes: nil,  // TODO: Add memory tracking
            throughput: throughput
        )
    }

    /// Run multiple benchmarks and return a report.
    public func runAll(_ benchmarks: [Benchmark]) throws -> BenchmarkReport {
        var results: [BenchmarkResult] = []

        for (index, benchmark) in benchmarks.enumerated() {
            onProgress?(index, benchmarks.count, benchmark.id)

            do {
                let result = try run(benchmark)
                results.append(result)
            } catch {
                print("Warning: Benchmark \(benchmark.id) failed: \(error)")
            }
        }

        return BenchmarkReport(
            title: "MetalHLO Benchmark Report",
            results: results
        )
    }

    /// Run benchmarks filtered by category.
    public func runCategory(_ category: String, from benchmarks: [Benchmark]) throws -> BenchmarkReport {
        let filtered = benchmarks.filter { $0.category == category }
        return try runAll(filtered)
    }

    /// Run benchmarks matching a pattern.
    public func runMatching(_ pattern: String, from benchmarks: [Benchmark]) throws -> BenchmarkReport {
        let filtered = benchmarks.filter {
            $0.id.contains(pattern) || $0.name.contains(pattern) || $0.operation.contains(pattern)
        }
        return try runAll(filtered)
    }
}

/// A simple benchmark definition using closures.
public struct SimpleBenchmark: Benchmark {
    public let id: String
    public let name: String
    public let category: String
    public let operation: String
    public let configuration: [String: String]
    public let mlirProgram: String

    private let inputGenerator: @Sendable (Client) throws -> [Buffer]
    private let throughputCalculator: (@Sendable (ExecutionTiming) -> ThroughputMetrics?)?

    public init(
        id: String,
        name: String,
        category: String,
        operation: String,
        configuration: [String: String],
        mlirProgram: String,
        inputGenerator: @escaping @Sendable (Client) throws -> [Buffer],
        throughputCalculator: (@Sendable (ExecutionTiming) -> ThroughputMetrics?)? = nil
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.operation = operation
        self.configuration = configuration
        self.mlirProgram = mlirProgram
        self.inputGenerator = inputGenerator
        self.throughputCalculator = throughputCalculator
    }

    public func createInputs(client: Client) throws -> [Buffer] {
        try inputGenerator(client)
    }

    public func calculateThroughput(timing: ExecutionTiming) -> ThroughputMetrics? {
        throughputCalculator?(timing)
    }
}

/// Builder for creating benchmarks with a fluent API.
public struct BenchmarkBuilder {
    private var id: String = ""
    private var name: String = ""
    private var category: String = ""
    private var operation: String = ""
    private var configuration: [String: String] = [:]
    private var mlirProgram: String = ""
    private var inputGenerator: (@Sendable (Client) throws -> [Buffer])?
    private var throughputCalculator: (@Sendable (ExecutionTiming) -> ThroughputMetrics?)?

    public init() {}

    public func withID(_ id: String) -> BenchmarkBuilder {
        var copy = self
        copy.id = id
        return copy
    }

    public func withName(_ name: String) -> BenchmarkBuilder {
        var copy = self
        copy.name = name
        return copy
    }

    public func withCategory(_ category: String) -> BenchmarkBuilder {
        var copy = self
        copy.category = category
        return copy
    }

    public func withOperation(_ operation: String) -> BenchmarkBuilder {
        var copy = self
        copy.operation = operation
        return copy
    }

    public func withConfiguration(_ configuration: [String: String]) -> BenchmarkBuilder {
        var copy = self
        copy.configuration = configuration
        return copy
    }

    public func withMLIR(_ mlir: String) -> BenchmarkBuilder {
        var copy = self
        copy.mlirProgram = mlir
        return copy
    }

    public func withInputs(_ generator: @escaping @Sendable (Client) throws -> [Buffer]) -> BenchmarkBuilder {
        var copy = self
        copy.inputGenerator = generator
        return copy
    }

    public func withThroughput(_ calculator: @escaping @Sendable (ExecutionTiming) -> ThroughputMetrics?) -> BenchmarkBuilder {
        var copy = self
        copy.throughputCalculator = calculator
        return copy
    }

    public func build() -> SimpleBenchmark {
        SimpleBenchmark(
            id: id,
            name: name,
            category: category,
            operation: operation,
            configuration: configuration,
            mlirProgram: mlirProgram,
            inputGenerator: inputGenerator ?? { _ in [] },
            throughputCalculator: throughputCalculator
        )
    }
}
