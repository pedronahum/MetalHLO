// StableHLOConformanceTests.swift
// MetalHLOTests
//
// Conformance tests that verify MetalHLO produces correct results
// by comparing against StableHLO reference test data.

import Testing
import Foundation
@testable import MetalHLO

// MARK: - Core Unary Operations

@Suite("StableHLO Conformance: Unary Operations")
struct UnaryConformanceTests {

    @Test("abs float32")
    func absFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("abs_float32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("neg float32")
    func negFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("neg_float32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("exp float32")
    func expFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("exp_float32_20_20", tolerance: ConformanceTestRunner.relaxedTolerance)
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("log float32")
    func logFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("log_float32_20_20", tolerance: ConformanceTestRunner.relaxedTolerance)
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("sqrt float32")
    func sqrtFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("sqrt_float32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("rsqrt float32")
    func rsqrtFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("rsqrt_float32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("ceil float32")
    func ceilFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("ceil_float32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("floor float32")
    func floorFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("floor_float32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("tanh float32")
    func tanhFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("tanh_float32_20_20", tolerance: ConformanceTestRunner.relaxedTolerance)
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("sin float32")
    func sinFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("sin_float32_20_20", tolerance: ConformanceTestRunner.trigTolerance)
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("cos float32")
    func cosFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("cos_float32_20_20", tolerance: ConformanceTestRunner.trigTolerance)
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }
}

// MARK: - Phase 3: Math Operations

@Suite("StableHLO Conformance: Math Operations")
struct MathConformanceTests {

    @Test("acos float32 (via atan2)")
    func acosFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("acos_float32_20_20", tolerance: ConformanceTestRunner.trigTolerance)
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("logistic float32 (sigmoid)")
    func logisticFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("logistic_float32_20_20", tolerance: ConformanceTestRunner.relaxedTolerance)
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }
}

// MARK: - Binary Arithmetic Operations

@Suite("StableHLO Conformance: Binary Operations")
struct BinaryConformanceTests {

    @Test("add float32")
    func addFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("add_float32_20_20_float32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("multiply float32")
    func mulFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("mul_float32_20_20_float32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    // Note: atan2 and divide tests require either:
    // 1. Implementation of those ops in MetalHLO
    // 2. Support for literal constants in the parser (for small tensor tests)

    @Test("maximum float32")
    func maxFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("max_float32_20_20_float32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("minimum float32")
    func minFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("min_float32_20_20_float32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("power float32")
    func powFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("pow_float32_20_30_float32_20_30", tolerance: ConformanceTestRunner.relaxedTolerance)
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("atan2 float32")
    func atan2Float32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("atan2_float32_20_20_float32_20_20", tolerance: ConformanceTestRunner.trigTolerance)
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }
}

// MARK: - Comparison Operations
// Note: Comparison operations use different naming conventions in StableHLO testdata
// and often use literal constants instead of hex data. These tests are disabled
// until we add support for parsing literal constants.

// @Suite("StableHLO Conformance: Comparison Operations")
// struct ComparisonConformanceTests { ... }

// MARK: - Float16 Operations

@Suite("StableHLO Conformance: Float16 Operations")
struct Float16ConformanceTests {

    @Test("abs float16")
    func absFloat16() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("abs_float16_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("add float16")
    func addFloat16() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("add_float16_20_20_float16_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("exp float16")
    func expFloat16() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("exp_float16_20_20", tolerance: 1e-2) // Float16 has lower precision
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("tanh float16")
    func tanhFloat16() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("tanh_float16_20_20", tolerance: 1e-2)
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }
}

// MARK: - BFloat16 Operations
// Note: BFloat16 tests are disabled - MPS converts bfloat16 to float16 internally,
// causing precision mismatches. Enable when native bfloat16 support is verified.

// @Suite("StableHLO Conformance: BFloat16 Operations")
// struct BFloat16ConformanceTests { ... }

// MARK: - Matrix Operations (dot_general)

@Suite("StableHLO Conformance: Matrix Operations")
struct MatrixConformanceTests {

    @Test("dot_general float16 -> float32")
    func dotGeneralFloat16ToFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("dot_general_float16_4_3_float32_3_6")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("dot_general int32 -> float32")
    func dotGeneralInt32ToFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("dot_general_int32_4_3_float32_3_6")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("dot_general int64 -> float32")
    func dotGeneralInt64ToFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("dot_general_int64_4_3_float32_3_6")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    // Note: float64 output tests are skipped - not commonly used in LLM inference
    // and require special output type handling
}

// MARK: - Control Flow Operations (Phase 2)

@Suite("StableHLO Conformance: Control Flow")
struct ControlFlowConformanceTests {

    @Test("clamp float32")
    func clampFloat32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("clamp_float32_float32_2_3_float32")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("clamp int32")
    func clampInt32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("clamp_int32_int32_2_3_int32")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    // Note: select tests require predicate tensor (i1 type) which needs special
    // buffer creation. See Debug suite for manual select test.
}

// MARK: - Integer Operations

@Suite("StableHLO Conformance: Integer Operations")
struct IntegerConformanceTests {

    @Test("abs int32")
    func absInt32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("abs_int32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("add int32")
    func addInt32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("add_int32_20_20_int32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }

    @Test("neg int32")
    func negInt32() async throws {
        let runner = try ConformanceTestRunner()
        let result = try await runner.runTest("neg_int32_20_20")
        #expect(result.passed, "\(result.errorMessage ?? "Unknown error")")
    }
}

// MARK: - Test Suite Runner

@Suite("StableHLO Conformance Suite")
struct ConformanceSuiteTests {

    @Test("Run priority test suite")
    func prioritySuite() async throws {
        let runner = try ConformanceTestRunner()
        let results = await runner.runSuite(StableHLOTestSuite.priorityTests, tolerance: ConformanceTestRunner.relaxedTolerance)

        print(results.summary)

        // Allow some failures during development, but track pass rate
        #expect(results.passRate >= 0.5, "Pass rate below 50%: \(results.passRate * 100)%")
    }
}

// MARK: - Multi-Optimization Level Tests

@Suite("StableHLO Conformance: Optimization Level Consistency")
struct OptimizationLevelConsistencyTests {

    @Test("Unary ops consistent across optimization levels",
          arguments: ["abs_float32_20_20", "neg_float32_20_20", "exp_float32_20_20", "sqrt_float32_20_20"])
    func unaryOpsConsistent(testName: String) async throws {
        let results = try await ConformanceTestRunner.runTestAcrossOptLevels(
            testName,
            tolerance: ConformanceTestRunner.relaxedTolerance
        )

        // All optimization levels should produce the same pass/fail result
        let allPassed = results.allSatisfy { $0.passed }
        let allFailed = results.allSatisfy { !$0.passed }

        #expect(allPassed || allFailed, "Test \(testName) has inconsistent results across optimization levels")

        if !allPassed && !allFailed {
            let passedLevels = results.filter { $0.passed }.map { "O\($0.optimizationLevel.rawValue)" }
            let failedLevels = results.filter { !$0.passed }.map { "O\($0.optimizationLevel.rawValue)" }
            print("INCONSISTENT: \(testName)")
            print("  Passed: \(passedLevels.joined(separator: ", "))")
            print("  Failed: \(failedLevels.joined(separator: ", "))")
        }
    }

    @Test("Binary ops consistent across optimization levels",
          arguments: ["add_float32_20_20_float32_20_20", "mul_float32_20_20_float32_20_20", "max_float32_20_20_float32_20_20"])
    func binaryOpsConsistent(testName: String) async throws {
        let results = try await ConformanceTestRunner.runTestAcrossOptLevels(
            testName,
            tolerance: ConformanceTestRunner.relaxedTolerance
        )

        let allPassed = results.allSatisfy { $0.passed }
        let allFailed = results.allSatisfy { !$0.passed }

        #expect(allPassed || allFailed, "Test \(testName) has inconsistent results across optimization levels")
    }

    @Test("Priority suite consistent across optimization levels")
    func prioritySuiteConsistent() async throws {
        let results = try await ConformanceTestRunner.runSuiteAcrossOptLevels(
            StableHLOTestSuite.priorityTests,
            tolerance: ConformanceTestRunner.relaxedTolerance
        )

        print(results.summary)

        #expect(results.isConsistent, "Results should be consistent across all optimization levels. Inconsistent tests: \(results.inconsistentTests)")
    }
}

// MARK: - Comprehensive Test Suite (All Available Tests)

@Suite("StableHLO Conformance: Comprehensive")
struct ComprehensiveConformanceTests {

    @Test("Cache statistics")
    func cacheStatistics() async throws {
        let manager = StableHLOTestManager.shared
        let stats = try await manager.getCacheStatistics()
        print(stats.summary)

        // Just informational, no assertion
        #expect(stats.totalAvailable > 0, "Should have some available tests")
    }

    @Test("Download all test files")
    func downloadAllTests() async throws {
        let manager = StableHLOTestManager.shared

        print("Fetching list of all available tests...")
        let allTests = try await manager.fetchAllAvailableTests()
        print("Found \(allTests.count) test files")

        print("Downloading all test files (this may take a while)...")
        let (downloaded, failed, skipped) = try await manager.downloadAllTests(
            concurrency: 20,
            progressHandler: { current, total in
                if current % 100 == 0 || current == total {
                    print("Progress: \(current)/\(total)")
                }
            }
        )

        print("Download complete:")
        print("  Downloaded: \(downloaded)")
        print("  Failed: \(failed)")
        print("  Skipped (already cached): \(skipped)")

        // Allow some failures due to network issues
        let successRate = Double(downloaded + skipped) / Double(downloaded + failed + skipped)
        #expect(successRate >= 0.95, "Download success rate should be at least 95%")
    }

    @Test("Run all cached tests at O2")
    func runAllCachedTests() async throws {
        let manager = StableHLOTestManager.shared
        let cachedTests = manager.listCachedTests()

        guard !cachedTests.isEmpty else {
            print("No cached tests found. Run 'Download all test files' test first.")
            return
        }

        print("Running \(cachedTests.count) cached tests at O2...")

        let config = CompilationConfig(optimizationLevel: .O2)
        let runner = try ConformanceTestRunner(config: config)

        var passed = 0
        var failed = 0
        var errors: [String] = []

        for (index, testName) in cachedTests.enumerated() {
            let result = try await runner.runTest(testName, tolerance: ConformanceTestRunner.relaxedTolerance)
            if result.passed {
                passed += 1
            } else {
                failed += 1
                if errors.count < 20 {
                    errors.append("\(testName): \(result.errorMessage ?? "unknown")")
                }
            }

            if (index + 1) % 100 == 0 {
                print("Progress: \(index + 1)/\(cachedTests.count) (passed: \(passed), failed: \(failed))")
            }
        }

        print("\nFinal Results:")
        print("  Passed: \(passed)")
        print("  Failed: \(failed)")
        print("  Pass Rate: \(String(format: "%.1f", Double(passed) / Double(passed + failed) * 100))%")

        if !errors.isEmpty {
            print("\nFirst \(min(20, errors.count)) errors:")
            for error in errors.prefix(20) {
                print("  - \(error)")
            }
        }
    }

    @Test("Generate conformance analysis report")
    func generateAnalysisReport() async throws {
        let manager = StableHLOTestManager.shared
        let cachedTests = manager.listCachedTests()

        guard !cachedTests.isEmpty else {
            print("No cached tests found. Run 'Download all test files' test first.")
            return
        }

        // Analyze all tests
        let analyzer = try ConformanceAnalyzer()
        print("Analyzing \(cachedTests.count) tests...")

        let analysis = await analyzer.analyze(tests: cachedTests) { current, total in
            if current % 500 == 0 {
                print("Progress: \(current)/\(total)")
            }
        }

        // Generate markdown report
        let report = analyzer.generateMarkdownReport(analysis: analysis)

        // Write to file
        let outputPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("programming/MetalHLO/metalhlo-conformance-analysis.md")
        try report.write(to: outputPath, atomically: true, encoding: .utf8)

        print("\nReport written to: \(outputPath.path)")
        print("\nSummary:")
        print("  Total Tests: \(analysis.totalTests)")
        print("  Passed: \(analysis.passedTests)")
        print("  Failed: \(analysis.failedTests)")
        print("  Pass Rate: \(String(format: "%.1f", analysis.passRate))%")
        print("\n  Supported Operations: \(analysis.passedOperations.count)")
        print("  Unsupported Operations: \(analysis.unsupportedOperations.count)")
    }
}

// MARK: - Debug Tests

@Suite("StableHLO Conformance: Debug")
struct DebugConformanceTests {

    @Test("Debug expm1 parsing")
    func debugExpm1Parsing() async throws {
        let parser = StableHLOTestParser()
        let manager = StableHLOTestManager.shared

        // Get the test files
        let expm1URL = try await manager.ensureTestFile("expm1_float32_20_20")
        let expURL = try await manager.ensureTestFile("exp_float32_20_20")

        // Parse both
        let expm1Case = try parser.parse(contentsOf: expm1URL)
        let expCase = try parser.parse(contentsOf: expURL)

        print("=== exp test ===")
        print("Operation: \(expCase.operation)")
        print("Inputs count: \(expCase.inputs.count)")
        print("Expected count: \(expCase.expected.count)")
        if let firstInput = expCase.inputs.first {
            print("Input shape: \(firstInput.shape)")
            print("Input type: \(firstInput.elementType.rawValue)")
            print("Input data bytes: \(firstInput.data.count)")
            let floats = firstInput.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            print("First 5 input values: \(floats.prefix(5))")
        }

        print("\n=== expm1 test ===")
        print("Operation: \(expm1Case.operation)")
        print("Inputs count: \(expm1Case.inputs.count)")
        print("Expected count: \(expm1Case.expected.count)")
        if let firstInput = expm1Case.inputs.first {
            print("Input shape: \(firstInput.shape)")
            print("Input type: \(firstInput.elementType.rawValue)")
            print("Input data bytes: \(firstInput.data.count)")
            let floats = firstInput.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            print("First 5 input values: \(floats.prefix(5))")
        }

        // Check mainMLIR extraction
        print("\n=== mainMLIR comparison ===")
        print("exp mainMLIR:\n\(expCase.mainMLIR)")
        print("\nexpm1 mainMLIR:\n\(expm1Case.mainMLIR)")

        #expect(expCase.inputs.count == expm1Case.inputs.count, "Both should have same number of inputs")
        #expect(expCase.expected.count == expm1Case.expected.count, "Both should have same number of expected outputs")
    }

    @Test("Debug expm1 MLIR generation")
    func debugExpm1MLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("expm1_float32_20_20", tolerance: 0.01)
    }

    @Test("Debug exp MLIR generation (should pass)")
    func debugExpMLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("exp_float32_20_20", tolerance: 0.01)
    }

    @Test("Debug reduce_sum MLIR generation")
    func debugReduceSumMLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("reduce_sum_float32_2_3", tolerance: 0.01)
    }

    @Test("Debug broadcast_in_dim MLIR generation")
    func debugBroadcastInDimMLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("broadcast_in_dim_float32_2", tolerance: 0.01)
    }

    @Test("Debug compare eq MLIR generation")
    func debugCompareEqMLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("eq_float32_float32", tolerance: 0.01)
    }

    @Test("Debug reshape MLIR generation")
    func debugReshapeMLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("reshape_float32_2_3", tolerance: 0.01)
    }

    @Test("Debug concatenate MLIR generation")
    func debugConcatenateMLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("concatenate_float32_2_3_float32_2_3", tolerance: 0.01)
    }

    @Test("Debug dot_general MLIR generation")
    func debugDotGeneralMLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        // Using float16 -> float32 test since pure float32 -> float32 not available
        try await runner.runTestDebug("dot_general_float16_4_3_float32_3_6", tolerance: 0.01)
    }

    @Test("Debug clamp MLIR generation")
    func debugClampMLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("clamp_float32_float32_2_3_float32", tolerance: 0.01)
    }

    @Test("Debug acos MLIR generation")
    func debugAcosMLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("acos_float32_20_20", tolerance: 0.01)
    }

    @Test("Debug and int32 MLIR generation")
    func debugAndInt32MLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("and_int32_20_20_int32_20_20", tolerance: 0.0)
    }

    @Test("Debug or int32 MLIR generation")
    func debugOrInt32MLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("or_int32_20_20_int32_20_20", tolerance: 0.0)
    }

    @Test("Debug xor int32 MLIR generation")
    func debugXorInt32MLIRGeneration() async throws {
        let runner = try ConformanceTestRunner()
        try await runner.runTestDebug("xor_int32_20_20_int32_20_20", tolerance: 0.0)
    }

    @Test("Run batch float32 tests")
    func runBatchFloat32Tests() async throws {
        let runner = try ConformanceTestRunner()
        let testNames = [
            // Phase 3: Math operations
            "acos_float32_20_20",       // via atan2
            "logistic_float32_20_20",   // sigmoid via negate, exp, divide
            // Unary operations
            "abs_float32_20_20",
            "ceil_float32_20_20",
            "cos_float32_20_20",
            "exp_float32_20_20",
            "expm1_float32_20_20",
            "floor_float32_20_20",
            "log_float32_20_20",
            "log1p_float32_20_20",
            "neg_float32_20_20",
            "rsqrt_float32_20_20",
            "sin_float32_20_20",
            "sqrt_float32_20_20",
            "tanh_float32_20_20",
            "cbrt_float32_20_20",
            // Binary operations
            "add_float32_20_20_float32_20_20",
            "atan2_float32_20_20_float32_20_20",
            "div_float32_2_float32_2",
            "max_float32_20_20_float32_20_20",
            "min_float32_20_20_float32_20_20",
            "mul_float32_20_20_float32_20_20",
            "pow_float32_20_30_float32_20_30",
            // Broadcasting operations
            "add_float32_1_20_float32_20_20",
            "mul_float32_1_20_float32_20_20",
            "broadcast_in_dim_float32",
            "broadcast_in_dim_float32_2",
            "broadcast_in_dim_float32_1_2",
            // Reduction operations
            "reduce_max_float32_2_3",
            "reduce_min_float32_2_3",
            "reduce_sum_float32_2_3",
            // Shape operations
            "reshape_float32_2_3",
            "concatenate_float32_2_3_float32_2_3",
            // Comparison operations
            "eq_float32_float32",
            // Float16 tests
            "abs_float16_20_20",
            "add_float16_20_20_float16_20_20",
            "exp_float16_20_20",
            // Int32 tests
            "abs_int32_20_20",
            "add_int32_20_20_int32_20_20",
            "neg_int32_20_20",
            // Bitwise operations
            "and_int32_20_20_int32_20_20",
            "or_int32_20_20_int32_20_20",
            // Unsigned int tests
            "add_uint8_20_20_uint8_20_20",
            "add_uint32_20_20_uint32_20_20",
            // Matrix operations (dot_general)
            "dot_general_float16_4_3_float32_3_6",
            "dot_general_int32_4_3_float32_3_6",
            // Control flow operations (Phase 2)
            "clamp_float32_float32_2_3_float32",
            "clamp_int32_int32_2_3_int32",
        ]

        var passed = 0
        var failed = 0
        var failures: [(String, String)] = []

        for testName in testNames {
            let result = try await runner.runTest(testName, tolerance: ConformanceTestRunner.relaxedTolerance)
            if result.passed {
                passed += 1
                print("PASS: \(testName)")
            } else {
                failed += 1
                failures.append((testName, result.errorMessage ?? "unknown"))
                print("FAIL: \(testName) - \(result.errorMessage ?? "unknown")")
            }
        }

        print("\n=== Summary ===")
        print("Passed: \(passed)/\(testNames.count)")
        print("Failed: \(failed)/\(testNames.count)")
        print("Pass Rate: \(String(format: "%.1f", Double(passed) / Double(testNames.count) * 100))%")

        if !failures.isEmpty {
            print("\nFailures:")
            for (name, error) in failures {
                print("  - \(name): \(error)")
            }
        }

        #expect(passed >= 40, "Should pass at least 40 tests")
    }

    @Test("Select operation test")
    func testSelectOperation() async throws {
        // Test select: selects elements based on predicate
        // select(pred, true_value, false_value)
        let mlir = """
        module @select_test {
          func.func @main(%pred: tensor<2x3xi1>, %on_true: tensor<2x3xf32>, %on_false: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            %result = stablehlo.select %pred, %on_true, %on_false : tensor<2x3xi1>, tensor<2x3xf32>
            return %result : tensor<2x3xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Create predicate as bytes: [[true, false, true], [false, true, false]]
        let pred_bytes = Data([1, 0, 1, 0, 1, 0])
        let on_true: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let on_false: [Float] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

        let pred_buffer = try client.createBuffer(bytes: pred_bytes, shape: [2, 3], elementType: .int1)
        let true_buffer = client.createBuffer(on_true, shape: [2, 3])
        let false_buffer = client.createBuffer(on_false, shape: [2, 3])

        let outputs = try executable.execute([pred_buffer, true_buffer, false_buffer])
        let output = try outputs[0].toFloatArray()

        print("Select output: \(output)")

        // Expected: [1.0, 20.0, 3.0, 40.0, 5.0, 60.0]
        let expected: [Float] = [1.0, 20.0, 3.0, 40.0, 5.0, 60.0]
        for (i, (got, exp)) in zip(output, expected).enumerated() {
            #expect(abs(got - exp) < 0.001, "Mismatch at index \(i): got \(got), expected \(exp)")
        }
    }

    @Test("Clamp operation test")
    func testClampOperation() async throws {
        // Test clamp: clamps values to [min, max] range
        let mlir = """
        module @clamp_test {
          func.func @main(%min: tensor<f32>, %operand: tensor<2x3xf32>, %max: tensor<f32>) -> (tensor<2x3xf32>) {
            %min_bc = stablehlo.broadcast_in_dim %min, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
            %max_bc = stablehlo.broadcast_in_dim %max, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
            %result = stablehlo.clamp %min_bc, %operand, %max_bc : tensor<2x3xf32>
            return %result : tensor<2x3xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Values: some below min (1.0), some in range, some above max (5.0)
        let operand: [Float] = [-2.0, 0.5, 3.0, 7.0, 2.5, 10.0]
        let min_val: [Float] = [1.0]
        let max_val: [Float] = [5.0]

        let min_buffer = client.createBuffer(min_val, shape: [])
        let operand_buffer = client.createBuffer(operand, shape: [2, 3])
        let max_buffer = client.createBuffer(max_val, shape: [])

        let outputs = try executable.execute([min_buffer, operand_buffer, max_buffer])
        let output = try outputs[0].toFloatArray()

        print("Clamp output: \(output)")

        // Expected: [1.0, 1.0, 3.0, 5.0, 2.5, 5.0]
        let expected: [Float] = [1.0, 1.0, 3.0, 5.0, 2.5, 5.0]
        for (i, (got, exp)) in zip(output, expected).enumerated() {
            #expect(abs(got - exp) < 0.001, "Mismatch at index \(i): got \(got), expected \(exp)")
        }
    }

    @Test("Debug iota literal parsing")
    func debugIotaLiteralParsing() async throws {
        // Test parsing of literal constants like [[0, 0, 0], [1, 1, 1]]
        let parser = StableHLOTestParser()
        let manager = StableHLOTestManager.shared

        // Get the iota test file
        let iotaURL = try await manager.ensureTestFile("iota_")

        // Parse the test
        let iotaCase = try parser.parse(contentsOf: iotaURL)

        print("=== iota test ===")
        print("Operation: \(iotaCase.operation)")
        print("Inputs count: \(iotaCase.inputs.count)")  // iota has no inputs
        print("Expected count: \(iotaCase.expected.count)")

        // Expected should parse the literal [[0, 0, 0], [1, 1, 1]] as ui8
        if let expected = iotaCase.expected.first {
            print("Expected shape: \(expected.shape)")
            print("Expected type: \(expected.elementType.rawValue)")
            print("Expected data bytes: \(expected.data.count)")

            // Parse as UInt8
            let values = expected.data.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) }
            print("Expected values: \(values)")

            // Should be [0, 0, 0, 1, 1, 1]
            #expect(expected.shape == [2, 3], "Shape should be [2, 3]")
            #expect(expected.elementType.rawValue == "ui8", "Type should be ui8")
            #expect(values == [0, 0, 0, 1, 1, 1], "Values should be [0, 0, 0, 1, 1, 1]")
        }
    }

    @Test("Debug rem literal parsing with hex NaN")
    func debugRemLiteralParsing() async throws {
        // Test parsing of hex bit pattern 0x7FC00000 (NaN)
        let parser = StableHLOTestParser()
        let manager = StableHLOTestManager.shared

        // Get the rem test file
        let remURL = try await manager.ensureTestFile("rem_float32_2_float32_2")

        // Parse the test
        let remCase = try parser.parse(contentsOf: remURL)

        print("=== rem test ===")
        print("Operation: \(remCase.operation)")
        print("Inputs count: \(remCase.inputs.count)")
        print("Expected count: \(remCase.expected.count)")

        for (i, input) in remCase.inputs.enumerated() {
            print("Input \(i): shape=\(input.shape), type=\(input.elementType.rawValue)")
            let values = input.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            print("  Values: \(values)")
        }

        if let expected = remCase.expected.first {
            print("Expected shape: \(expected.shape)")
            print("Expected type: \(expected.elementType.rawValue)")

            // Parse as Float
            let values = expected.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            print("Expected values: \(values)")

            // Should be NaN (0x7FC00000)
            #expect(expected.shape == [2], "Shape should be [2]")
            #expect(values.allSatisfy { $0.isNaN }, "Values should all be NaN")
        }
    }

    @Test("Attention pattern: Q @ K^T @ V")
    func testAttentionPattern() async throws {
        // Test the core attention pattern used in transformers:
        // scores = Q @ K^T, then output = scores @ V
        //
        // Q: [batch, seq, heads, head_dim] -> [4, 3] after flattening
        // K: [batch, seq, heads, head_dim] -> [4, 3]
        // V: [batch, seq, heads, head_dim] -> [4, 3]
        //
        // For simplicity, we test [4, 3] @ [3, 4] = [4, 4] (QK^T)
        // then [4, 4] @ [4, 3] = [4, 3] (scores @ V)

        let mlir = """
        module @attention_pattern {
          func.func @main(%Q: tensor<4x3xf32>, %K: tensor<4x3xf32>, %V: tensor<4x3xf32>) -> (tensor<4x3xf32>) {
            // Transpose K: [4, 3] -> [3, 4]
            %K_T = stablehlo.transpose %K, dims = [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
            // Q @ K^T: [4, 3] @ [3, 4] -> [4, 4]
            %scores = stablehlo.dot_general %Q, %K_T, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x4xf32>) -> tensor<4x4xf32>
            // scores @ V: [4, 4] @ [4, 3] -> [4, 3]
            %output = stablehlo.dot_general %scores, %V, contracting_dims = [1] x [0] : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
            return %output : tensor<4x3xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir)

        // Create test inputs: Q, K, V each [4, 3] float32
        let Q_data: [Float] = [
            1.0, 0.5, 0.3,
            0.2, 1.0, 0.4,
            0.3, 0.2, 1.0,
            0.4, 0.6, 0.5
        ]
        let K_data: [Float] = [
            0.8, 0.4, 0.2,
            0.3, 0.9, 0.3,
            0.2, 0.3, 0.8,
            0.5, 0.5, 0.6
        ]
        let V_data: [Float] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        ]

        let Q_buffer = client.createBuffer(Q_data, shape: [4, 3])
        let K_buffer = client.createBuffer(K_data, shape: [4, 3])
        let V_buffer = client.createBuffer(V_data, shape: [4, 3])

        let outputs = try executable.execute([Q_buffer, K_buffer, V_buffer])
        #expect(outputs.count == 1, "Should have one output")

        let output = try outputs[0].toFloatArray()
        print("Attention output shape: \(outputs[0].shape)")
        print("First 6 values: \(output.prefix(6))")

        // Verify output is reasonable (non-zero, finite)
        #expect(output.allSatisfy { $0.isFinite }, "All outputs should be finite")
        #expect(output.contains { $0 != 0 }, "Output should not be all zeros")

        // The output should be [4, 3] tensor with attention-weighted values
        #expect(outputs[0].shape == [4, 3], "Output shape should be [4, 3]")
    }
}
