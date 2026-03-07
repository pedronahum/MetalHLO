// main.swift
// MetalHLO Benchmark Runner
//
// Command-line tool for running MetalHLO benchmarks.

import Foundation
import MetalHLOBenchmarks
import MetalHLO

// MARK: - CLI Argument Parsing

struct BenchmarkCLI {
    enum Command {
        case run(filter: String?, category: String?, quick: Bool, output: String?, optLevel: Int?, allOptLevels: Bool)
        case compare(filter: String?, category: String?, quick: Bool, output: String?)
        case list
        case help
    }

    static func parseArguments() -> Command {
        let args = CommandLine.arguments.dropFirst()

        if args.isEmpty {
            return .run(filter: nil, category: nil, quick: false, output: nil, optLevel: nil, allOptLevels: false)
        }

        var filter: String?
        var category: String?
        var quick = false
        var output: String?
        var optLevel: Int?
        var allOptLevels = false
        var compare = false

        var iterator = args.makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "-h", "--help":
                return .help
            case "-l", "--list":
                return .list
            case "--compare":
                compare = true
            case "-f", "--filter":
                filter = iterator.next()
            case "-c", "--category":
                category = iterator.next()
            case "-q", "--quick":
                quick = true
            case "-o", "--output":
                output = iterator.next()
            case "-O0":
                optLevel = 0
            case "-O1":
                optLevel = 1
            case "-O2":
                optLevel = 2
            case "-O3":
                optLevel = 3
            case "-O", "--opt-level":
                if let levelStr = iterator.next(), let level = Int(levelStr) {
                    optLevel = level
                }
            case "--all-opt-levels":
                allOptLevels = true
            default:
                // Treat as filter if no flag
                if !arg.hasPrefix("-") {
                    filter = arg
                }
            }
        }

        if compare {
            return .compare(filter: filter, category: category, quick: quick, output: output)
        }

        return .run(filter: filter, category: category, quick: quick, output: output, optLevel: optLevel, allOptLevels: allOptLevels)
    }

    static func printHelp() {
        print("""
        MetalHLO Benchmark Runner

        USAGE:
            benchmark-runner [OPTIONS] [FILTER]

        OPTIONS:
            -h, --help          Show this help message
            -l, --list          List all available benchmarks
            -c, --category CAT  Run only benchmarks in category
            -f, --filter PAT    Run only benchmarks matching pattern
            -q, --quick         Quick mode (fewer iterations)
            -o, --output PATH   Write results to JSON file
            -O0                 No optimization (debug mode)
            -O1                 Basic optimization
            -O2                 Standard optimization (default)
            -O3                 Aggressive optimization
            --all-opt-levels    Run benchmarks with all optimization levels (O0-O3)
            --compare           Compare backends: MPSGraph vs O2 vs O3 vs GPU+ANE

        EXAMPLES:
            benchmark-runner                    Run all benchmarks at O2
            benchmark-runner -q                 Quick run of all benchmarks
            benchmark-runner -O3                Run with aggressive optimization
            benchmark-runner --all-opt-levels   Compare all optimization levels
            benchmark-runner --compare -q       Compare all backends (quick mode)
            benchmark-runner --compare -c matrix Compare backends for matrix ops
            benchmark-runner -c matrix          Run only matrix benchmarks
            benchmark-runner -f GEMM            Run benchmarks matching "GEMM"
            benchmark-runner -o results.json    Save results to JSON file

        CATEGORIES:
            matrix           Matrix operations (dot, dot_general, transpose)
            reduction        Reduction operations (reduce_sum, reduce_max, reduce_window)
            arithmetic       Element-wise operations (add, multiply, exp, etc.)
            convolution      Convolution operations (conv2d)
            normalization    Normalization operations (batch_norm, layer_norm)
            control_flow     Control flow operations (while, if)
            indexing         Indexing operations (slice, gather, scatter, pad, concatenate)
            model_mlp        MLP inference benchmarks
            model_cnn        CNN inference benchmarks
            model_transformer Transformer component benchmarks
            model_e2e        End-to-end pipeline benchmarks

        NOTE: FFT benchmarks are planned but disabled pending complex type support.
        """)
    }

    static func listBenchmarks() {
        let benchmarks = OperationBenchmarks.allWithModels()

        print("Available Benchmarks (\(benchmarks.count) total):\n")

        let grouped = Dictionary(grouping: benchmarks) { $0.category }
        for (category, categoryBenchmarks) in grouped.sorted(by: { $0.key < $1.key }) {
            print("\(category.uppercased()) (\(categoryBenchmarks.count) benchmarks)")
            print(String(repeating: "-", count: 50))
            for benchmark in categoryBenchmarks.sorted(by: { $0.id < $1.id }) {
                print("  \(benchmark.id.padding(toLength: 16, withPad: " ", startingAt: 0)) \(benchmark.name)")
            }
            print()
        }
    }
}

// MARK: - Backend Configuration

struct BackendConfig {
    let name: String
    let shortName: String
    let optimizationLevel: OptimizationLevel
    let devicePolicy: DevicePolicy
    let useMPSGraph: Bool  // true = use default client.compile(mlir) without config

    static let allBackends: [BackendConfig] = [
        BackendConfig(name: "MPSGraph (default)", shortName: "MPSGraph", optimizationLevel: .O2, devicePolicy: .gpuOnly, useMPSGraph: true),
        BackendConfig(name: "Metal O2", shortName: "O2", optimizationLevel: .O2, devicePolicy: .gpuOnly, useMPSGraph: false),
        BackendConfig(name: "Metal O3", shortName: "O3", optimizationLevel: .O3, devicePolicy: .gpuOnly, useMPSGraph: false),
        BackendConfig(name: "GPU+ANE auto", shortName: "ANE", optimizationLevel: .O2, devicePolicy: .auto, useMPSGraph: false),
    ]
}

/// Result for a single benchmark across one backend.
struct BackendResult {
    let backendName: String
    let meanMs: Double
    let stdDevMs: Double
    let p95Ms: Double
    let failed: Bool
    let error: String?

    static func failure(backend: String, error: String) -> BackendResult {
        BackendResult(backendName: backend, meanMs: 0, stdDevMs: 0, p95Ms: 0, failed: true, error: error)
    }
}

/// Comparison result for one benchmark across all backends.
struct ComparisonEntry {
    let benchmarkID: String
    let benchmarkName: String
    let category: String
    let operation: String
    let results: [BackendResult]

    var baselineMeanMs: Double? {
        results.first(where: { !$0.failed })?.meanMs
    }

    func speedup(for result: BackendResult) -> Double? {
        guard let baseline = baselineMeanMs, baseline > 0, !result.failed else { return nil }
        return baseline / result.meanMs
    }

    var bestResult: BackendResult? {
        results.filter { !$0.failed }.min(by: { $0.meanMs < $1.meanMs })
    }
}

// MARK: - Compare Mode

func runComparison(filter: String?, category: String?, quick: Bool, output: String?) async {
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║      MetalHLO Multi-Backend Comparison                     ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    let hw = HardwareInfo.current()
    print("Hardware: \(hw.chipName)")
    print("Memory: \(hw.totalMemoryBytes / (1024 * 1024 * 1024)) GB")
    print("OS: \(hw.osVersion)")
    print()

    let config: BenchmarkConfig = quick ? .quick : .standard
    print("Mode: \(quick ? "Quick" : "Standard") (\(config.warmupIterations) warmup, \(config.measurementIterations) measurements)")
    print()

    // Get benchmarks
    var benchmarks = OperationBenchmarks.allWithModels()

    if let category = category {
        benchmarks = benchmarks.filter { $0.category == category }
        print("Filtered to category: \(category)")
    }

    if let filter = filter {
        benchmarks = benchmarks.filter {
            $0.id.localizedCaseInsensitiveContains(filter) ||
            $0.name.localizedCaseInsensitiveContains(filter) ||
            $0.operation.localizedCaseInsensitiveContains(filter)
        }
        print("Filtered by pattern: \(filter)")
    }

    if benchmarks.isEmpty {
        print("No benchmarks match the specified criteria.")
        return
    }

    let backends = BackendConfig.allBackends
    print("Backends: \(backends.map { $0.name }.joined(separator: ", "))")
    print("Benchmarks: \(benchmarks.count)")
    print()

    let startTime = Date()
    var comparisons: [ComparisonEntry] = []

    // Create a single runner and reuse it for all backends
    let runner: BenchmarkRunner
    do {
        runner = try BenchmarkRunner(config: config)
        runner.verbose = false
    } catch {
        print("Error: Failed to create benchmark runner: \(error)")
        return
    }

    for (benchIdx, benchmark) in benchmarks.enumerated() {
        let progress = Double(benchIdx) / Double(benchmarks.count) * 100
        print("[\(String(format: "%3.0f", progress))%] \(benchmark.id)", terminator: "")
        fflush(stdout)

        var backendResults: [BackendResult] = []

        for backend in backends {
            do {
                runner.useMPSGraph = backend.useMPSGraph
                runner.optimizationLevel = backend.optimizationLevel
                runner.devicePolicy = backend.devicePolicy

                let result = try runner.run(benchmark)
                let meanMs = result.gpuTime.mean * 1000
                let stdMs = result.gpuTime.stdDev * 1000
                let p95Ms = result.gpuTime.p95 * 1000

                backendResults.append(BackendResult(
                    backendName: backend.shortName,
                    meanMs: meanMs,
                    stdDevMs: stdMs,
                    p95Ms: p95Ms,
                    failed: false,
                    error: nil
                ))
                print(" [\(backend.shortName):OK]", terminator: "")
                fflush(stdout)
            } catch {
                backendResults.append(.failure(backend: backend.shortName, error: "\(error)"))
                print(" [\(backend.shortName):FAIL]", terminator: "")
                fflush(stdout)
            }
        }
        print()  // newline after all backends

        comparisons.append(ComparisonEntry(
            benchmarkID: benchmark.id,
            benchmarkName: benchmark.name,
            category: benchmark.category,
            operation: benchmark.operation,
            results: backendResults
        ))
    }

    let elapsed = Date().timeIntervalSince(startTime)

    // Print comparison table
    print()
    print(String(repeating: "=", count: 110))
    print()

    let backendNames = backends.map { $0.shortName }
    printComparisonTable(comparisons: comparisons, backendNames: backendNames)

    // Print summary
    print(String(repeating: "=", count: 110))
    print("Summary:")
    print("  Benchmarks: \(comparisons.count)")
    print("  Total time: \(String(format: "%.1f", elapsed)) seconds")
    print()

    // Count wins per backend
    var wins: [String: Int] = [:]
    for comp in comparisons {
        if let best = comp.bestResult {
            wins[best.backendName, default: 0] += 1
        }
    }
    print("  Wins by backend:")
    for (name, count) in wins.sorted(by: { $0.value > $1.value }) {
        print("    \(name): \(count)")
    }
    print()

    // Save results
    let outputBase = output ?? "/tmp/benchmark_comparison"
    saveComparisonResults(comparisons: comparisons, backendNames: backendNames, basePath: outputBase, hw: hw)
}

func printComparisonTable(comparisons: [ComparisonEntry], backendNames: [String]) {
    let grouped = Dictionary(grouping: comparisons) { $0.category }

    for (cat, entries) in grouped.sorted(by: { $0.key < $1.key }) {
        print("\(cat.uppercased())")
        print(String(repeating: "-", count: 110))

        // Header
        var header = "Benchmark".padding(toLength: 18, withPad: " ", startingAt: 0)
        for name in backendNames {
            header += "\(name) (ms)".padding(toLength: 16, withPad: " ", startingAt: 0)
        }
        header += "Best Speedup"
        print(header)
        print(String(repeating: "-", count: 110))

        for entry in entries.sorted(by: { $0.benchmarkID < $1.benchmarkID }) {
            var line = entry.benchmarkID.padding(toLength: 18, withPad: " ", startingAt: 0)

            for result in entry.results {
                if result.failed {
                    line += "FAIL".padding(toLength: 16, withPad: " ", startingAt: 0)
                } else {
                    let val = String(format: "%.3f", result.meanMs)
                    let std = String(format: "%.3f", result.stdDevMs)
                    line += "\(val)±\(std)".padding(toLength: 16, withPad: " ", startingAt: 0)
                }
            }

            // Best speedup vs baseline (first backend)
            if let baseline = entry.baselineMeanMs, baseline > 0, let best = entry.bestResult {
                let speedup = baseline / best.meanMs
                if speedup > 1.01 {
                    line += String(format: "%.2fx (%@)", speedup, best.backendName)
                } else if speedup < 0.99 {
                    line += String(format: "%.2fx (%@)", speedup, best.backendName)
                } else {
                    line += "~1.00x"
                }
            }

            print(line)
        }
        print()
    }
}

func saveComparisonResults(comparisons: [ComparisonEntry], backendNames: [String], basePath: String, hw: HardwareInfo) {
    // Save markdown
    var md = "# MetalHLO Multi-Backend Comparison\n\n"
    md += "**Date:** \(ISO8601DateFormatter().string(from: Date()))\n"
    md += "**Hardware:** \(hw.chipName)\n"
    md += "**Memory:** \(hw.totalMemoryBytes / (1024 * 1024 * 1024)) GB\n"
    md += "**OS:** \(hw.osVersion)\n\n"
    md += "## Results\n\n"

    let grouped = Dictionary(grouping: comparisons) { $0.category }

    for (cat, entries) in grouped.sorted(by: { $0.key < $1.key }) {
        md += "### \(cat.capitalized)\n\n"
        md += "| Benchmark |"
        for name in backendNames {
            md += " \(name) (ms) |"
        }
        md += " Best Speedup |\n"

        md += "|-----------|"
        for _ in backendNames {
            md += "------------|"
        }
        md += "--------------|\n"

        for entry in entries.sorted(by: { $0.benchmarkID < $1.benchmarkID }) {
            md += "| \(entry.benchmarkID) |"

            for result in entry.results {
                if result.failed {
                    md += " FAIL |"
                } else {
                    md += " \(String(format: "%.3f", result.meanMs)) |"
                }
            }

            if let baseline = entry.baselineMeanMs, baseline > 0, let best = entry.bestResult {
                let speedup = baseline / best.meanMs
                md += " \(String(format: "%.2fx", speedup)) (\(best.backendName)) |"
            } else {
                md += " N/A |"
            }
            md += "\n"
        }
        md += "\n"
    }

    // Wins summary
    md += "## Backend Wins\n\n"
    var wins: [String: Int] = [:]
    for comp in comparisons {
        if let best = comp.bestResult {
            wins[best.backendName, default: 0] += 1
        }
    }
    md += "| Backend | Wins |\n|---------|------|\n"
    for (name, count) in wins.sorted(by: { $0.value > $1.value }) {
        md += "| \(name) | \(count) |\n"
    }

    let mdPath = basePath.hasSuffix(".md") ? basePath : "\(basePath).md"
    do {
        try md.write(toFile: mdPath, atomically: true, encoding: .utf8)
        print("Comparison saved to: \(mdPath)")
    } catch {
        print("Error saving comparison: \(error)")
    }

    // Save JSON
    let jsonPath = basePath.hasSuffix(".json") ? basePath : "\(basePath).json"
    var jsonEntries: [[String: Any]] = []
    for entry in comparisons {
        var jsonEntry: [String: Any] = [
            "id": entry.benchmarkID,
            "name": entry.benchmarkName,
            "category": entry.category,
            "operation": entry.operation,
        ]
        var backendData: [String: Any] = [:]
        for result in entry.results {
            if result.failed {
                backendData[result.backendName] = ["failed": true, "error": result.error ?? "unknown"]
            } else {
                backendData[result.backendName] = [
                    "mean_ms": result.meanMs,
                    "std_dev_ms": result.stdDevMs,
                    "p95_ms": result.p95Ms,
                ]
            }
        }
        jsonEntry["backends"] = backendData
        jsonEntries.append(jsonEntry)
    }

    let jsonRoot: [String: Any] = [
        "date": ISO8601DateFormatter().string(from: Date()),
        "hardware": hw.chipName,
        "results": jsonEntries,
    ]

    do {
        let data = try JSONSerialization.data(withJSONObject: jsonRoot, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: jsonPath))
        print("JSON saved to: \(jsonPath)")
    } catch {
        print("Error saving JSON: \(error)")
    }
}

// MARK: - Main

func main() async {
    let command = BenchmarkCLI.parseArguments()

    switch command {
    case .help:
        BenchmarkCLI.printHelp()

    case .list:
        BenchmarkCLI.listBenchmarks()

    case .compare(let filter, let category, let quick, let output):
        await runComparison(filter: filter, category: category, quick: quick, output: output)

    case .run(let filter, let category, let quick, let output, let optLevel, let allOptLevels):
        if allOptLevels {
            // Run with all optimization levels
            for level in 0...3 {
                print("\n" + String(repeating: "▓", count: 70))
                print("▓  OPTIMIZATION LEVEL: O\(level)")
                print(String(repeating: "▓", count: 70) + "\n")
                await runBenchmarks(filter: filter, category: category, quick: quick, output: output, optLevel: level)
            }
        } else {
            await runBenchmarks(filter: filter, category: category, quick: quick, output: output, optLevel: optLevel)
        }
    }
}

func runBenchmarks(filter: String?, category: String?, quick: Bool, output: String?, optLevel: Int? = nil) async {
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           MetalHLO Benchmark Suite                         ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    // Get hardware info
    let hw = HardwareInfo.current()
    print("Hardware: \(hw.chipName)")
    print("Memory: \(hw.totalMemoryBytes / (1024 * 1024 * 1024)) GB")
    print("OS: \(hw.osVersion)")
    print()

    // Select config
    let config: BenchmarkConfig = quick ? .quick : .standard
    print("Mode: \(quick ? "Quick" : "Standard") (\(config.warmupIterations) warmup, \(config.measurementIterations) measurements)")

    // Determine optimization level
    let optimizationLevel: OptimizationLevel
    if let level = optLevel {
        switch level {
        case 0: optimizationLevel = .O0
        case 1: optimizationLevel = .O1
        case 3: optimizationLevel = .O3
        default: optimizationLevel = .O2
        }
    } else {
        optimizationLevel = .O2
    }
    print("Optimization: O\(optimizationLevel.rawValue)")
    print()

    // Create runner
    let runner: BenchmarkRunner
    do {
        runner = try BenchmarkRunner(config: config)
        runner.optimizationLevel = optimizationLevel
        runner.verbose = false
    } catch {
        print("Error: Failed to create benchmark runner: \(error)")
        return
    }

    // Get benchmarks
    var benchmarks = OperationBenchmarks.allWithModels()

    // Apply filters
    if let category = category {
        benchmarks = benchmarks.filter { $0.category == category }
        print("Filtered to category: \(category)")
    }

    if let filter = filter {
        benchmarks = benchmarks.filter {
            $0.id.localizedCaseInsensitiveContains(filter) ||
            $0.name.localizedCaseInsensitiveContains(filter) ||
            $0.operation.localizedCaseInsensitiveContains(filter)
        }
        print("Filtered by pattern: \(filter)")
    }

    if benchmarks.isEmpty {
        print("No benchmarks match the specified criteria.")
        return
    }

    print("Running \(benchmarks.count) benchmarks...\n")
    print(String(repeating: "=", count: 70))

    // Progress tracking
    let startTime = Date()

    runner.onProgress = { index, count, id in
        let progress = Double(index) / Double(count) * 100
        print("[\(String(format: "%3.0f", progress))%] Running: \(id)")
    }

    // Run benchmarks
    let report: BenchmarkReport
    do {
        report = try runner.runAll(benchmarks)
    } catch {
        print("Error running benchmarks: \(error)")
        return
    }

    let elapsed = Date().timeIntervalSince(startTime)

    // Print results
    print(String(repeating: "=", count: 70))
    print("\nResults:\n")

    // Group by category
    let grouped = Dictionary(grouping: report.results) { $0.category }
    for (cat, results) in grouped.sorted(by: { $0.key < $1.key }) {
        print("\(cat.uppercased())")
        print(String(repeating: "-", count: 70))
        let header = "ID".padding(toLength: 16, withPad: " ", startingAt: 0) +
                     "Operation".padding(toLength: 20, withPad: " ", startingAt: 0) +
                     "Mean (ms)".padding(toLength: 12, withPad: " ", startingAt: 0) +
                     "Std Dev".padding(toLength: 12, withPad: " ", startingAt: 0) +
                     "P95 (ms)"
        print(header)
        print(String(repeating: "-", count: 70))

        for result in results.sorted(by: { $0.id < $1.id }) {
            let meanMs = result.gpuTime.mean * 1000
            let stdMs = result.gpuTime.stdDev * 1000
            let p95Ms = result.gpuTime.p95 * 1000

            let idStr = result.id.padding(toLength: 16, withPad: " ", startingAt: 0)
            let opStr = String(result.operation.prefix(20)).padding(toLength: 20, withPad: " ", startingAt: 0)
            let meanStr = String(format: "%12.3f", meanMs)
            let stdStr = String(format: "%12.3f", stdMs)
            let p95Str = String(format: "%12.3f", p95Ms)
            print("\(idStr)\(opStr)\(meanStr)\(stdStr)\(p95Str)")

            // Print GFLOPS if available
            if let throughput = result.throughput, let flops = throughput.flops {
                let gflops = flops / 1e9
                if gflops > 1 {
                    print(String(repeating: " ", count: 50) + String(format: "GFLOPS: %.1f", gflops))
                }
            }
        }
        print()
    }

    // Summary
    print(String(repeating: "=", count: 70))
    print("Summary:")
    print("  Benchmarks completed: \(report.results.count)")
    print("  Total time: \(String(format: "%.1f", elapsed)) seconds")
    print()

    // Save to file if requested
    if let outputPath = output {
        // Include optimization level in filename if specified
        let finalPath: String
        if optLevel != nil {
            let ext = (outputPath as NSString).pathExtension
            let base = (outputPath as NSString).deletingPathExtension
            finalPath = "\(base)_O\(optimizationLevel.rawValue).\(ext)"
        } else {
            finalPath = outputPath
        }

        do {
            let json = try report.toJSON()
            try json.write(to: URL(fileURLWithPath: finalPath))
            print("Results saved to: \(finalPath)")
        } catch {
            print("Error saving results: \(error)")
        }

        // Also save markdown
        let mdPath = finalPath.replacingOccurrences(of: ".json", with: ".md")
        let markdown = report.toMarkdown()
        do {
            try markdown.write(toFile: mdPath, atomically: true, encoding: .utf8)
            print("Markdown report saved to: \(mdPath)")
        } catch {
            print("Error saving markdown: \(error)")
        }
    }
}

// Run main
await main()
