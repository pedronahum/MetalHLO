// ConformanceTestCLI.swift
// MetalHLOTests
//
// Command-line interface helpers for running conformance tests.
// This can be called from Swift REPL or a simple main.swift.

import Foundation
@testable import MetalHLO

/// Simple CLI for running conformance tests locally
public struct ConformanceTestCLI {

    /// Run a quick sanity check with core operations
    public static func runQuickCheck() async {
        print("MetalHLO StableHLO Conformance - Quick Check")
        print("=============================================\n")

        let testNames = [
            "abs_float32_20_20",
            "neg_float32_20_20",
            "add_float32_20_20_float32_20_20",
            "mul_float32_20_20_float32_20_20",
        ]

        await runTests(testNames)
    }

    /// Run the full priority test suite
    public static func runPrioritySuite() async {
        print("MetalHLO StableHLO Conformance - Priority Suite")
        print("================================================\n")

        await runTests(StableHLOTestSuite.priorityTests)
    }

    /// Run arithmetic operations suite
    public static func runArithmeticSuite() async {
        print("MetalHLO StableHLO Conformance - Arithmetic Suite")
        print("==================================================\n")

        await runTests(StableHLOTestSuite.arithmetic)
    }

    /// Run unary math operations suite
    public static func runUnaryMathSuite() async {
        print("MetalHLO StableHLO Conformance - Unary Math Suite")
        print("==================================================\n")

        await runTests(StableHLOTestSuite.unaryMath)
    }

    /// Run a specific test by name
    public static func runSingleTest(_ testName: String, verbose: Bool = true) async {
        guard let runner = try? ConformanceTestRunner() else {
            print("ERROR: Failed to create test runner")
            return
        }

        print("Running test: \(testName)")

        do {
            let result = try await runner.runTest(testName, tolerance: 1e-4)
            print(result.summary)

            if verbose && !result.passed {
                print("  Error: \(result.errorMessage ?? "unknown")")
            }
        } catch {
            print("FAIL: \(testName) - \(error)")
        }
    }

    /// Run multiple tests
    public static func runTests(_ testNames: [String], tolerance: Float = 1e-4) async {
        guard let runner = try? ConformanceTestRunner() else {
            print("ERROR: Failed to create test runner")
            return
        }

        var passed = 0
        var failed = 0

        for testName in testNames {
            do {
                let result = try await runner.runTest(testName, tolerance: tolerance)
                if result.passed {
                    passed += 1
                    print("  PASS: \(testName) (max diff: \(String(format: "%.2e", result.maxDifference)))")
                } else {
                    failed += 1
                    print("  FAIL: \(testName)")
                    print("        \(result.errorMessage ?? "unknown error")")
                }
            } catch {
                failed += 1
                print("  FAIL: \(testName) - \(error)")
            }
        }

        print("\n-----------------------------------")
        print("Total: \(testNames.count), Passed: \(passed), Failed: \(failed)")
        print("Pass Rate: \(String(format: "%.1f", Double(passed) / Double(testNames.count) * 100))%")
    }

    /// List available test suites
    public static func listSuites() {
        print("""
        Available Test Suites:
        ----------------------
        - quickCheck:   4 basic operations (sanity check)
        - priority:     Core LLM operations (\(StableHLOTestSuite.priorityTests.count) tests)
        - arithmetic:   Binary arithmetic ops (\(StableHLOTestSuite.arithmetic.count) tests)
        - unaryMath:    Unary math functions (\(StableHLOTestSuite.unaryMath.count) tests)
        - trigonometric: Trig functions (\(StableHLOTestSuite.trigonometric.count) tests)
        - comparison:   Comparison ops (\(StableHLOTestSuite.comparison.count) tests)
        - comprehensive: All above (\(StableHLOTestSuite.comprehensive.count) tests)
        """)
    }

    /// List all cached test files
    public static func listCachedTests() {
        let manager = StableHLOTestManager.shared
        let cached = manager.listCachedTests()

        print("Cached Test Files (\(cached.count) total):")
        print("-------------------------------------------")
        for test in cached.prefix(20) {
            print("  \(test)")
        }
        if cached.count > 20 {
            print("  ... and \(cached.count - 20) more")
        }

        let sizeKB = Double(manager.cacheSize()) / 1024.0
        print("\nCache size: \(String(format: "%.1f", sizeKB)) KB")
    }

    /// Download tests for offline use
    public static func downloadSuite(_ suite: [String]) async {
        let manager = StableHLOTestManager.shared

        print("Downloading \(suite.count) test files...")

        do {
            let urls = try await manager.ensureTestFiles(suite)
            print("Successfully downloaded \(urls.count) test files")
        } catch {
            print("Download failed: \(error)")
        }
    }

    /// Clear the test cache
    public static func clearCache() {
        let manager = StableHLOTestManager.shared

        do {
            try manager.clearCache()
            print("Cache cleared successfully")
        } catch {
            print("Failed to clear cache: \(error)")
        }
    }
}

// MARK: - Interactive Mode

extension ConformanceTestCLI {

    /// Print help message
    public static func printHelp() {
        print("""
        MetalHLO StableHLO Conformance Test CLI
        =======================================

        Usage (from Swift REPL or test code):

            // Run quick sanity check
            await ConformanceTestCLI.runQuickCheck()

            // Run priority suite
            await ConformanceTestCLI.runPrioritySuite()

            // Run a specific test
            await ConformanceTestCLI.runSingleTest("abs_float32_20_20")

            // List available suites
            ConformanceTestCLI.listSuites()

            // List cached tests
            ConformanceTestCLI.listCachedTests()

            // Download tests for offline use
            await ConformanceTestCLI.downloadSuite(StableHLOTestSuite.priorityTests)

            // Clear cache
            ConformanceTestCLI.clearCache()

        Test Naming Convention:
            operation_dtype_shape[_dtype_shape].mlir

        Examples:
            abs_float32_20_20           - abs on 20x20 float32 tensor
            add_float32_20_20_float32_20_20 - add two 20x20 float32 tensors
            tanh_float16_20_20          - tanh on 20x20 float16 tensor
        """)
    }
}
