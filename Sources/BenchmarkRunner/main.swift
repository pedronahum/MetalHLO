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
            -c, --category CAT  Run only benchmarks in category (matrix, reduction, arithmetic, convolution, normalization)
            -f, --filter PAT    Run only benchmarks matching pattern
            -q, --quick         Quick mode (fewer iterations)
            -o, --output PATH   Write results to JSON file
            -O0                 No optimization (debug mode)
            -O1                 Basic optimization
            -O2                 Standard optimization (default)
            -O3                 Aggressive optimization
            --all-opt-levels    Run benchmarks with all optimization levels (O0-O3)

        EXAMPLES:
            benchmark-runner                    Run all benchmarks at O2
            benchmark-runner -q                 Quick run of all benchmarks
            benchmark-runner -O3                Run with aggressive optimization
            benchmark-runner --all-opt-levels   Compare all optimization levels
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

// MARK: - Main

func main() async {
    let command = BenchmarkCLI.parseArguments()

    switch command {
    case .help:
        BenchmarkCLI.printHelp()

    case .list:
        BenchmarkCLI.listBenchmarks()

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
    var completed = 0
    let total = benchmarks.count
    let startTime = Date()

    runner.onProgress = { index, count, id in
        completed = index
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
        // Use Swift string formatting instead of C-style %s
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
