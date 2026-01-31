// main.swift
// MLX Comparison Runner
//
// Command-line tool for comparing MetalHLO and MLX performance.

import Foundation
import MetalHLO
import MetalHLOBenchmarks
import MLXBenchmarks
import MLX

/// Note: MLX requires its Metal library to be available. When running from
/// SwiftPM command line, this may require building and running from Xcode,
/// or setting up the Metal library path manually.

// MARK: - CLI Argument Parsing

struct ComparisonCLI {
    enum Command {
        case compare(filter: String?, category: String?, quick: Bool, output: String?)
        case list
        case help
    }

    static func parseArguments() -> Command {
        let args = CommandLine.arguments.dropFirst()

        if args.isEmpty {
            return .compare(filter: nil, category: nil, quick: false, output: nil)
        }

        var filter: String?
        var category: String?
        var quick = false
        var output: String?

        var iterator = args.makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "-h", "--help":
                return .help
            case "-l", "--list":
                return .list
            case "-f", "--filter":
                filter = iterator.next()
            case "-c", "--category":
                category = iterator.next()
            case "-q", "--quick":
                quick = true
            case "-o", "--output":
                output = iterator.next()
            default:
                if !arg.hasPrefix("-") {
                    filter = arg
                }
            }
        }

        return .compare(filter: filter, category: category, quick: quick, output: output)
    }

    static func printHelp() {
        print("""
        MetalHLO vs MLX Comparison Runner

        USAGE:
            mlx-comparison [OPTIONS] [FILTER]

        OPTIONS:
            -h, --help          Show this help message
            -l, --list          List all comparable benchmarks
            -c, --category CAT  Run only benchmarks in category
            -f, --filter PAT    Run only benchmarks matching pattern
            -q, --quick         Quick mode (fewer iterations)
            -o, --output PATH   Write results to JSON file

        EXAMPLES:
            mlx-comparison                  Compare all matching benchmarks
            mlx-comparison -q               Quick comparison
            mlx-comparison -c matrix        Compare only matrix benchmarks
            mlx-comparison -f GEMM          Compare benchmarks matching "GEMM"
            mlx-comparison -o results.json  Save comparison to JSON

        CATEGORIES:
            matrix           Matrix operations (dot, dot_general)
            arithmetic       Element-wise operations (add, multiply, exp)
            reduction        Reduction operations (sum, max)
            convolution      Convolution operations (conv2d)
            normalization    Normalization operations (layer_norm)
            model_mlp        MLP inference benchmarks
            model_transformer Transformer component benchmarks

        NOTE:
            MLX requires its Metal library to be available at runtime.
            If you see "Failed to load the default metallib" errors, try:
            1. Open MetalHLO.xcodeproj in Xcode
            2. Select the mlx-comparison scheme
            3. Build and run from Xcode (Cmd+R)
        """)
    }

    static func listBenchmarks() {
        let mlxBenchmarks = MLXBenchmarks.matchingBenchmarks()

        print("Comparable Benchmarks (\(mlxBenchmarks.count) total):\n")

        let grouped = Dictionary(grouping: mlxBenchmarks.values) { $0.category }
        for (category, benchmarks) in grouped.sorted(by: { $0.key < $1.key }) {
            print("\(category.uppercased()) (\(benchmarks.count) benchmarks)")
            print(String(repeating: "-", count: 60))
            for benchmark in benchmarks.sorted(by: { $0.id < $1.id }) {
                print("  \(benchmark.id.padding(toLength: 16, withPad: " ", startingAt: 0)) \(benchmark.name)")
            }
            print()
        }
    }
}

// MARK: - Comparison Result

struct ComparisonResult: Codable {
    let id: String
    let name: String
    let category: String
    let operation: String
    let metalHLOMeanMs: Double
    let mlxMeanMs: Double
    let metalHLOStdDevMs: Double
    let mlxStdDevMs: Double
    let speedup: Double  // > 1 means MetalHLO is faster
    let iterations: Int
}

struct ComparisonReport: Codable {
    let timestamp: String
    let hardware: HardwareDescription
    let results: [ComparisonResult]
    let summary: ComparisonSummary

    struct HardwareDescription: Codable {
        let chipName: String
        let totalMemoryGB: Int
        let osVersion: String
    }

    struct ComparisonSummary: Codable {
        let totalBenchmarks: Int
        let metalHLOFaster: Int
        let mlxFaster: Int
        let averageSpeedup: Double
        let geometricMeanSpeedup: Double
    }
}

// MARK: - Main

func main() async {
    let command = ComparisonCLI.parseArguments()

    switch command {
    case .help:
        ComparisonCLI.printHelp()

    case .list:
        ComparisonCLI.listBenchmarks()

    case .compare(let filter, let category, let quick, let output):
        await runComparison(filter: filter, category: category, quick: quick, output: output)
    }
}

func runComparison(filter: String?, category: String?, quick: Bool, output: String?) async {
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║           MetalHLO vs MLX Comparison Suite                         ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    // Get hardware info
    let hw = HardwareInfo.current()
    print("Hardware: \(hw.chipName)")
    print("Memory: \(hw.totalMemoryBytes / (1024 * 1024 * 1024)) GB")
    print("OS: \(hw.osVersion)")
    print()

    // Configuration
    let warmupIterations = quick ? 3 : 10
    let measurementIterations = quick ? 10 : 50
    print("Mode: \(quick ? "Quick" : "Standard") (\(warmupIterations) warmup, \(measurementIterations) measurements)")
    print()

    // Get MLX benchmarks
    var mlxBenchmarks = MLXBenchmarks.all()

    // Apply category filter
    if let category = category {
        mlxBenchmarks = mlxBenchmarks.filter { $0.category == category }
        print("Filtered to category: \(category)")
    }

    // Apply pattern filter
    if let filter = filter {
        mlxBenchmarks = mlxBenchmarks.filter {
            $0.id.localizedCaseInsensitiveContains(filter) ||
            $0.name.localizedCaseInsensitiveContains(filter) ||
            $0.operation.localizedCaseInsensitiveContains(filter)
        }
        print("Filtered by pattern: \(filter)")
    }

    if mlxBenchmarks.isEmpty {
        print("No benchmarks match the specified criteria.")
        return
    }

    print("Running \(mlxBenchmarks.count) comparisons...\n")
    print(String(repeating: "=", count: 80))

    // Setup MetalHLO
    let config = BenchmarkConfig(
        warmupIterations: warmupIterations,
        measurementIterations: measurementIterations
    )

    let runner: BenchmarkRunner
    do {
        runner = try BenchmarkRunner(config: config)
        runner.verbose = false
    } catch {
        print("Error: Failed to create MetalHLO benchmark runner: \(error)")
        return
    }

    // Get matching MetalHLO benchmarks
    let metalHLOBenchmarks = OperationBenchmarks.allWithModels()
    let metalHLOByID = Dictionary(uniqueKeysWithValues: metalHLOBenchmarks.map { ($0.id, $0) })

    // Run comparisons
    var results: [ComparisonResult] = []
    let startTime = Date()

    for (index, mlxBenchmark) in mlxBenchmarks.enumerated() {
        let progress = Double(index + 1) / Double(mlxBenchmarks.count) * 100
        print("[\(String(format: "%3.0f", progress))%] Comparing: \(mlxBenchmark.id)")

        // Check if matching MetalHLO benchmark exists
        guard let metalHLOBenchmark = metalHLOByID[mlxBenchmark.id] else {
            print("       ⚠️  No matching MetalHLO benchmark found")
            continue
        }

        // Run MLX benchmark
        print("       Running MLX...")
        let mlxResult = mlxBenchmark.run(iterations: measurementIterations, warmup: warmupIterations)

        // Run MetalHLO benchmark
        print("       Running MetalHLO...")
        let metalHLOResult: BenchmarkResult
        do {
            metalHLOResult = try runner.run(metalHLOBenchmark)
        } catch {
            print("       ❌ MetalHLO error: \(error)")
            continue
        }

        // Calculate speedup
        let metalHLOMeanMs = metalHLOResult.gpuTime.mean * 1000
        let mlxMeanMs = mlxResult.meanTimeSeconds * 1000
        let speedup = mlxMeanMs / metalHLOMeanMs  // > 1 means MetalHLO faster

        let comparison = ComparisonResult(
            id: mlxBenchmark.id,
            name: mlxBenchmark.name,
            category: mlxBenchmark.category,
            operation: mlxBenchmark.operation,
            metalHLOMeanMs: metalHLOMeanMs,
            mlxMeanMs: mlxMeanMs,
            metalHLOStdDevMs: metalHLOResult.gpuTime.stdDev * 1000,
            mlxStdDevMs: mlxResult.stdDevSeconds * 1000,
            speedup: speedup,
            iterations: measurementIterations
        )
        results.append(comparison)

        // Print inline result
        let winner = speedup > 1.0 ? "MetalHLO" : "MLX"
        let speedupStr = speedup > 1.0 ? String(format: "%.2fx faster", speedup) : String(format: "%.2fx slower", 1.0 / speedup)
        print("       ✅ MetalHLO: \(String(format: "%.3f", metalHLOMeanMs))ms | MLX: \(String(format: "%.3f", mlxMeanMs))ms | \(winner) \(speedupStr)")
    }

    let elapsed = Date().timeIntervalSince(startTime)

    // Print results summary
    print(String(repeating: "=", count: 80))
    print("\nResults Summary:\n")

    // Group by category
    let grouped = Dictionary(grouping: results) { $0.category }
    for (cat, catResults) in grouped.sorted(by: { $0.key < $1.key }) {
        print("\(cat.uppercased())")
        print(String(repeating: "-", count: 80))

        let header = "ID".padding(toLength: 16, withPad: " ", startingAt: 0) +
                     "MetalHLO (ms)".padding(toLength: 15, withPad: " ", startingAt: 0) +
                     "MLX (ms)".padding(toLength: 15, withPad: " ", startingAt: 0) +
                     "Speedup".padding(toLength: 12, withPad: " ", startingAt: 0) +
                     "Winner"
        print(header)
        print(String(repeating: "-", count: 80))

        for result in catResults.sorted(by: { $0.id < $1.id }) {
            let idStr = result.id.padding(toLength: 16, withPad: " ", startingAt: 0)
            let metalStr = String(format: "%12.3f", result.metalHLOMeanMs).padding(toLength: 15, withPad: " ", startingAt: 0)
            let mlxStr = String(format: "%12.3f", result.mlxMeanMs).padding(toLength: 15, withPad: " ", startingAt: 0)
            let speedupStr = String(format: "%10.2fx", result.speedup).padding(toLength: 12, withPad: " ", startingAt: 0)
            let winner = result.speedup > 1.0 ? "MetalHLO ✓" : "MLX ✓"
            print("\(idStr)\(metalStr)\(mlxStr)\(speedupStr)\(winner)")
        }
        print()
    }

    // Overall summary
    let metalHLOFaster = results.filter { $0.speedup > 1.0 }.count
    let mlxFaster = results.filter { $0.speedup <= 1.0 }.count
    let avgSpeedup = results.map { $0.speedup }.reduce(0, +) / Double(results.count)
    let geoMeanSpeedup = pow(results.map { $0.speedup }.reduce(1, *), 1.0 / Double(results.count))

    print(String(repeating: "=", count: 80))
    print("Overall Summary:")
    print("  Total benchmarks compared: \(results.count)")
    let metalPct = String(format: "%.1f", Double(metalHLOFaster) / Double(results.count) * 100)
    let mlxPct = String(format: "%.1f", Double(mlxFaster) / Double(results.count) * 100)
    print("  MetalHLO faster: \(metalHLOFaster) (\(metalPct)%)")
    print("  MLX faster: \(mlxFaster) (\(mlxPct)%)")
    print("  Average speedup: \(String(format: "%.2fx", avgSpeedup))")
    print("  Geometric mean speedup: \(String(format: "%.2fx", geoMeanSpeedup))")
    print("  Total comparison time: \(String(format: "%.1f", elapsed)) seconds")
    print()

    // Save to file if requested
    if let outputPath = output {
        let report = ComparisonReport(
            timestamp: ISO8601DateFormatter().string(from: Date()),
            hardware: ComparisonReport.HardwareDescription(
                chipName: hw.chipName,
                totalMemoryGB: Int(hw.totalMemoryBytes / (1024 * 1024 * 1024)),
                osVersion: hw.osVersion
            ),
            results: results,
            summary: ComparisonReport.ComparisonSummary(
                totalBenchmarks: results.count,
                metalHLOFaster: metalHLOFaster,
                mlxFaster: mlxFaster,
                averageSpeedup: avgSpeedup,
                geometricMeanSpeedup: geoMeanSpeedup
            )
        )

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(report)
            try data.write(to: URL(fileURLWithPath: outputPath))
            print("Results saved to: \(outputPath)")
        } catch {
            print("Error saving results: \(error)")
        }

        // Also save markdown report
        let mdPath = outputPath.replacingOccurrences(of: ".json", with: ".md")
        let markdown = generateMarkdownReport(report: report)
        do {
            try markdown.write(toFile: mdPath, atomically: true, encoding: .utf8)
            print("Markdown report saved to: \(mdPath)")
        } catch {
            print("Error saving markdown: \(error)")
        }
    }
}

func generateMarkdownReport(report: ComparisonReport) -> String {
    var md = """
    # MetalHLO vs MLX Comparison Report

    **Date:** \(report.timestamp)

    ## Hardware
    - **Chip:** \(report.hardware.chipName)
    - **Memory:** \(report.hardware.totalMemoryGB) GB
    - **OS:** \(report.hardware.osVersion)

    ## Summary

    | Metric | Value |
    |--------|-------|
    | Total Benchmarks | \(report.summary.totalBenchmarks) |
    | MetalHLO Faster | \(report.summary.metalHLOFaster) (\(String(format: "%.1f", Double(report.summary.metalHLOFaster) / Double(report.summary.totalBenchmarks) * 100))%) |
    | MLX Faster | \(report.summary.mlxFaster) (\(String(format: "%.1f", Double(report.summary.mlxFaster) / Double(report.summary.totalBenchmarks) * 100))%) |
    | Average Speedup | \(String(format: "%.2fx", report.summary.averageSpeedup)) |
    | Geometric Mean | \(String(format: "%.2fx", report.summary.geometricMeanSpeedup)) |

    ## Detailed Results

    """

    // Group by category
    let grouped = Dictionary(grouping: report.results) { $0.category }
    for (category, results) in grouped.sorted(by: { $0.key < $1.key }) {
        md += "### \(category.uppercased())\n\n"
        md += "| ID | MetalHLO (ms) | MLX (ms) | Speedup | Winner |\n"
        md += "|----|---------------|----------|---------|--------|\n"

        for result in results.sorted(by: { $0.id < $1.id }) {
            let winner = result.speedup > 1.0 ? "MetalHLO" : "MLX"
            md += "| \(result.id) | \(String(format: "%.3f", result.metalHLOMeanMs)) | \(String(format: "%.3f", result.mlxMeanMs)) | \(String(format: "%.2fx", result.speedup)) | \(winner) |\n"
        }
        md += "\n"
    }

    md += """

    ---
    *Generated by MetalHLO Benchmark Suite*
    """

    return md
}

// Run main
await main()
