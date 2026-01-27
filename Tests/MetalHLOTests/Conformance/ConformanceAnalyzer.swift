// ConformanceAnalyzer.swift
// MetalHLOTests
//
// Analyzes conformance test results to categorize failures and identify gaps.

import Foundation
import MetalHLO

/// Categorizes test failure reasons
public enum FailureCategory: String, CaseIterable, Sendable {
    case unsupportedOperation = "Unsupported Operation"
    case unsupportedElementType = "Unsupported Element Type"
    case parseError = "Parse Error"
    case precisionMismatch = "Precision Mismatch"
    case shapeMismatch = "Shape Mismatch"
    case compilationError = "Compilation Error"
    case executionError = "Execution Error"
    case chloOperation = "CHLO Dialect Operation"
    case complexType = "Complex Type"
    case unknown = "Unknown"
}

/// Detailed failure information
public struct FailureInfo: Sendable {
    public let testName: String
    public let category: FailureCategory
    public let operation: String
    public let elementType: String
    public let errorMessage: String
}

/// Analysis results
public struct ConformanceAnalysis: Sendable {
    public let totalTests: Int
    public let passedTests: Int
    public let failedTests: Int
    public let failuresByCategory: [FailureCategory: [FailureInfo]]
    public let unsupportedOperations: Set<String>
    public let unsupportedElementTypes: Set<String>
    public let passedOperations: Set<String>

    public var passRate: Double {
        totalTests > 0 ? Double(passedTests) / Double(totalTests) * 100 : 0
    }
}

/// Analyzes conformance test results
public final class ConformanceAnalyzer: @unchecked Sendable {

    private let runner: ConformanceTestRunner
    private let manager: StableHLOTestManager

    public init() throws {
        self.runner = try ConformanceTestRunner()
        self.manager = StableHLOTestManager.shared
    }

    /// Analyze a batch of tests
    public func analyze(tests: [String], progressHandler: ((Int, Int) -> Void)? = nil) async -> ConformanceAnalysis {
        var passed = 0
        var failed = 0
        var failures: [FailureInfo] = []
        var passedOps: Set<String> = []

        for (index, testName) in tests.enumerated() {
            let result = try? await runner.runTest(testName, tolerance: ConformanceTestRunner.relaxedTolerance)

            if let result = result {
                if result.passed {
                    passed += 1
                    passedOps.insert(result.operation)
                } else {
                    failed += 1
                    let info = categorizeFailure(testName: testName, result: result)
                    failures.append(info)
                }
            } else {
                failed += 1
                failures.append(FailureInfo(
                    testName: testName,
                    category: .unknown,
                    operation: extractOperation(from: testName),
                    elementType: extractElementType(from: testName),
                    errorMessage: "Test execution failed"
                ))
            }

            if (index + 1) % 100 == 0 {
                progressHandler?(index + 1, tests.count)
            }
        }

        // Group failures by category
        var failuresByCategory: [FailureCategory: [FailureInfo]] = [:]
        for failure in failures {
            failuresByCategory[failure.category, default: []].append(failure)
        }

        // Extract unique unsupported operations and element types
        let unsupportedOps = Set(failures.filter {
            $0.category == .unsupportedOperation || $0.category == .chloOperation
        }.map { $0.operation })

        let unsupportedTypes = Set(failures.filter {
            $0.category == .unsupportedElementType || $0.category == .complexType
        }.map { $0.elementType })

        return ConformanceAnalysis(
            totalTests: tests.count,
            passedTests: passed,
            failedTests: failed,
            failuresByCategory: failuresByCategory,
            unsupportedOperations: unsupportedOps,
            unsupportedElementTypes: unsupportedTypes,
            passedOperations: passedOps
        )
    }

    /// Categorize a test failure
    private func categorizeFailure(testName: String, result: ConformanceTestResult) -> FailureInfo {
        let errorMsg = result.errorMessage ?? ""
        let operation = extractOperation(from: testName)
        let elementType = extractElementType(from: testName)

        // Check for CHLO operations (separate dialect)
        if testName.contains("_chlo") {
            return FailureInfo(
                testName: testName,
                category: .chloOperation,
                operation: operation,
                elementType: elementType,
                errorMessage: errorMsg
            )
        }

        // Check for complex types
        if testName.contains("complex") {
            return FailureInfo(
                testName: testName,
                category: .complexType,
                operation: operation,
                elementType: elementType,
                errorMessage: errorMsg
            )
        }

        // Categorize by error message
        let category: FailureCategory
        if errorMsg.contains("Unsupported element type") {
            category = .unsupportedElementType
        } else if errorMsg.contains("Cannot extract operation") || errorMsg.contains("error 0") {
            category = .parseError
        } else if errorMsg.contains("error 1") {
            // MetalHLO error 1 is typically unsupported operation
            category = .unsupportedOperation
        } else if errorMsg.contains("error 7") {
            // MetalHLO error 7 is typically unsupported operation
            category = .unsupportedOperation
        } else if errorMsg.contains("Value mismatch") || errorMsg.contains("diff") {
            category = .precisionMismatch
        } else if errorMsg.contains("Element count mismatch") || errorMsg.contains("Shape mismatch") {
            category = .shapeMismatch
        } else if errorMsg.contains("Compilation") || errorMsg.contains("compile") {
            category = .compilationError
        } else if errorMsg.contains("Execution") || errorMsg.contains("execute") {
            category = .executionError
        } else {
            category = .unknown
        }

        return FailureInfo(
            testName: testName,
            category: category,
            operation: operation,
            elementType: elementType,
            errorMessage: errorMsg
        )
    }

    /// Extract operation name from test name
    private func extractOperation(from testName: String) -> String {
        // Test names follow pattern: operation_dtype_shape or operation_dtype_shape_dtype_shape
        let parts = testName.split(separator: "_")
        guard !parts.isEmpty else { return "unknown" }

        // Handle multi-word operations like "dot_general"
        var opParts: [String] = []
        for part in parts {
            let s = String(part)
            // Stop at dtype indicators
            if ["float16", "float32", "float64", "bfloat16", "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64", "complex64", "complex128", "bool"].contains(s) {
                break
            }
            // Stop at shape numbers
            if Int(s) != nil {
                break
            }
            opParts.append(s)
        }

        return opParts.isEmpty ? String(parts[0]) : opParts.joined(separator: "_")
    }

    /// Extract element type from test name
    private func extractElementType(from testName: String) -> String {
        let dtypes = ["float16", "float32", "float64", "bfloat16",
                      "int8", "int16", "int32", "int64",
                      "uint8", "uint16", "uint32", "uint64",
                      "complex64", "complex128", "bool"]

        for dtype in dtypes {
            if testName.contains(dtype) {
                return dtype
            }
        }
        return "unknown"
    }

    /// Generate markdown report
    public func generateMarkdownReport(analysis: ConformanceAnalysis) -> String {
        var lines: [String] = []

        lines.append("# MetalHLO StableHLO Conformance Analysis")
        lines.append("")
        lines.append("Generated: \(ISO8601DateFormatter().string(from: Date()))")
        lines.append("")

        // Summary
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append("| Total Tests | \(analysis.totalTests) |")
        lines.append("| Passed | \(analysis.passedTests) |")
        lines.append("| Failed | \(analysis.failedTests) |")
        lines.append("| Pass Rate | \(String(format: "%.1f", analysis.passRate))% |")
        lines.append("")

        // Failure breakdown
        lines.append("## Failure Breakdown by Category")
        lines.append("")
        lines.append("| Category | Count | Percentage |")
        lines.append("|----------|-------|------------|")
        for category in FailureCategory.allCases {
            let count = analysis.failuresByCategory[category]?.count ?? 0
            if count > 0 {
                let pct = Double(count) / Double(analysis.failedTests) * 100
                lines.append("| \(category.rawValue) | \(count) | \(String(format: "%.1f", pct))% |")
            }
        }
        lines.append("")

        // Passed operations
        lines.append("## Supported Operations (Passing Tests)")
        lines.append("")
        let sortedPassedOps = analysis.passedOperations.sorted()
        lines.append("Total: \(sortedPassedOps.count) operations")
        lines.append("")
        lines.append("```")
        for op in sortedPassedOps {
            lines.append(op)
        }
        lines.append("```")
        lines.append("")

        // Unsupported operations
        lines.append("## Unsupported Operations")
        lines.append("")
        let sortedUnsupportedOps = analysis.unsupportedOperations.sorted()
        lines.append("Total: \(sortedUnsupportedOps.count) operations")
        lines.append("")
        lines.append("```")
        for op in sortedUnsupportedOps {
            lines.append(op)
        }
        lines.append("```")
        lines.append("")

        // Unsupported element types
        lines.append("## Unsupported Element Types")
        lines.append("")
        let sortedTypes = analysis.unsupportedElementTypes.sorted()
        for type in sortedTypes {
            lines.append("- `\(type)`")
        }
        lines.append("")

        // CHLO operations (separate dialect)
        if let chloFailures = analysis.failuresByCategory[.chloOperation], !chloFailures.isEmpty {
            lines.append("## CHLO Dialect Operations (Not StableHLO)")
            lines.append("")
            lines.append("These tests use the CHLO (\"CheckerBoard HLO\") dialect, which is a separate dialect from StableHLO.")
            lines.append("CHLO operations are typically decomposed into StableHLO before execution.")
            lines.append("")
            let chloOps = Set(chloFailures.map { $0.operation }).sorted()
            lines.append("Operations: \(chloOps.count)")
            lines.append("")
            lines.append("```")
            for op in chloOps {
                lines.append(op)
            }
            lines.append("```")
            lines.append("")
        }

        // Complex type failures
        if let complexFailures = analysis.failuresByCategory[.complexType], !complexFailures.isEmpty {
            lines.append("## Complex Number Types")
            lines.append("")
            lines.append("Complex number operations (complex64, complex128) are not currently supported by MPS/Metal.")
            lines.append("")
            lines.append("Affected tests: \(complexFailures.count)")
            lines.append("")
        }

        // Precision mismatches
        if let precisionFailures = analysis.failuresByCategory[.precisionMismatch], !precisionFailures.isEmpty {
            lines.append("## Precision Mismatches")
            lines.append("")
            lines.append("These tests produce results but with precision differences exceeding tolerance.")
            lines.append("")
            lines.append("| Test | Operation | Element Type | Error |")
            lines.append("|------|-----------|--------------|-------|")
            for failure in precisionFailures.prefix(50) {
                let shortError = String(failure.errorMessage.prefix(60))
                lines.append("| \(failure.testName) | \(failure.operation) | \(failure.elementType) | \(shortError)... |")
            }
            if precisionFailures.count > 50 {
                lines.append("| ... | ... | ... | (\(precisionFailures.count - 50) more) |")
            }
            lines.append("")
        }

        // Parse errors
        if let parseFailures = analysis.failuresByCategory[.parseError], !parseFailures.isEmpty {
            lines.append("## Parse Errors")
            lines.append("")
            lines.append("These tests have MLIR that couldn't be parsed or transformed.")
            lines.append("Common causes: literal constants (not hex data), unsupported MLIR syntax.")
            lines.append("")
            lines.append("Affected tests: \(parseFailures.count)")
            lines.append("")
        }

        // Shape mismatches
        if let shapeFailures = analysis.failuresByCategory[.shapeMismatch], !shapeFailures.isEmpty {
            lines.append("## Shape Mismatches")
            lines.append("")
            lines.append("These tests produce outputs with different shapes than expected.")
            lines.append("Often caused by dtype conversion (e.g., float64→float32 doubles element count).")
            lines.append("")
            lines.append("| Test | Operation | Element Type |")
            lines.append("|------|-----------|--------------|")
            for failure in shapeFailures.prefix(20) {
                lines.append("| \(failure.testName) | \(failure.operation) | \(failure.elementType) |")
            }
            if shapeFailures.count > 20 {
                lines.append("| ... | ... | (\(shapeFailures.count - 20) more) |")
            }
            lines.append("")
        }

        return lines.joined(separator: "\n")
    }
}
