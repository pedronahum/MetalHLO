// IndexingConformanceTests.swift
// MetalHLOTests
//
// Conformance tests for slice, gather, scatter using official test suite.

import Testing
import Foundation
@testable import MetalHLO

@Suite("Indexing Operations Conformance")
struct IndexingConformanceTests {

    @Test("slice.mlir - Official conformance")
    func testSliceConformance() async throws {
        let runner = try InterpretTestRunner()
        let results = try await runner.runTestFile("slice.mlir", category: .interpret)

        printResults(results, filename: "slice.mlir")

        // Check if any tests failed (not just skipped)
        let failed = results.filter { !$0.passed && $0.errorMessage?.contains("SKIPPED") != true }
        if !failed.isEmpty {
            for fail in failed {
                print("FAILED: \(fail.testName) - \(fail.errorMessage ?? "unknown")")
            }
        }
    }

    @Test("gather.mlir - Official conformance")
    func testGatherConformance() async throws {
        let runner = try InterpretTestRunner()
        let results = try await runner.runTestFile("gather.mlir", category: .interpret)

        printResults(results, filename: "gather.mlir")

        let failed = results.filter { !$0.passed && $0.errorMessage?.contains("SKIPPED") != true }
        if !failed.isEmpty {
            for fail in failed {
                print("FAILED: \(fail.testName) - \(fail.errorMessage ?? "unknown")")
            }
        }
    }

    @Test("scatter.mlir - Official conformance")
    func testScatterConformance() async throws {
        let runner = try InterpretTestRunner()
        let results = try await runner.runTestFile("scatter.mlir", category: .interpret)

        printResults(results, filename: "scatter.mlir")

        let failed = results.filter { !$0.passed && $0.errorMessage?.contains("SKIPPED") != true }
        if !failed.isEmpty {
            for fail in failed {
                print("FAILED: \(fail.testName) - \(fail.errorMessage ?? "unknown")")
            }
        }
    }

    @Test("dynamic_slice.mlir - Official conformance")
    func testDynamicSliceConformance() async throws {
        let runner = try InterpretTestRunner()
        let results = try await runner.runTestFile("dynamic_slice.mlir", category: .interpret)

        printResults(results, filename: "dynamic_slice.mlir")

        let failed = results.filter { !$0.passed && $0.errorMessage?.contains("SKIPPED") != true }
        if !failed.isEmpty {
            for fail in failed {
                print("FAILED: \(fail.testName) - \(fail.errorMessage ?? "unknown")")
            }
        }
    }

    // MARK: - Helpers

    private func printResults(_ results: [InterpretTestResult], filename: String) {
        let passed = results.filter { $0.passed }.count
        let skipped = results.filter { $0.errorMessage?.contains("SKIPPED") == true }.count
        let failed = results.filter { !$0.passed && $0.errorMessage?.contains("SKIPPED") != true }.count
        print("\(filename): Passed=\(passed), Skipped=\(skipped), Failed=\(failed)")
    }
}
