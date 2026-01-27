// InterpretTestParser.swift
// MetalHLOTests
//
// Parser for official StableHLO interpret tests from:
// https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret
//
// These tests use a cleaner format with inline constants and check.expect_* verification.

import Foundation

/// A single test case from an interpret test file
public struct InterpretTestCase: Sendable {
    /// Name of the test function (e.g., "abs_op_test_si64")
    public let name: String

    /// The primary StableHLO operation being tested
    public let operation: String

    /// The full MLIR function body
    public let mlir: String

    /// Input constants extracted from the test
    public let inputs: [InterpretConstant]

    /// Expected output from check.expect_* call
    public let expected: InterpretConstant?

    /// Whether this test uses approximate comparison (expect_almost_eq)
    public let usesApproximateComparison: Bool
}

/// A constant value from an interpret test
public struct InterpretConstant: Sendable {
    /// Variable name (e.g., "%operand", "%lhs")
    public let name: String

    /// The dense literal value (e.g., "[-2, 0, 2]")
    public let literal: String

    /// The tensor type (e.g., "tensor<3xi64>")
    public let tensorType: String

    /// Parsed shape
    public let shape: [Int]

    /// Parsed element type
    public let elementType: String
}

/// Parser for StableHLO interpret test files
public struct InterpretTestParser {

    public init() {}

    /// Parse an interpret test file, returning all test cases
    public func parse(contentsOf url: URL) throws -> [InterpretTestCase] {
        let content = try String(contentsOf: url, encoding: .utf8)
        return try parse(content: content, filename: url.lastPathComponent)
    }

    /// Parse interpret test content
    public func parse(content: String, filename: String) throws -> [InterpretTestCase] {
        // Split by "// -----" separator
        let sections = content.components(separatedBy: "// -----")

        var testCases: [InterpretTestCase] = []

        for section in sections {
            let trimmed = section.trimmingCharacters(in: .whitespacesAndNewlines)

            // Skip empty sections and RUN comment sections
            if trimmed.isEmpty || !trimmed.contains("func.func") {
                continue
            }

            if let testCase = try? parseTestCase(from: trimmed, filename: filename) {
                testCases.append(testCase)
            }
        }

        return testCases
    }

    /// Parse a single test case from a function definition
    private func parseTestCase(from mlir: String, filename: String) throws -> InterpretTestCase {
        // Extract function name: func.func @name()
        let funcPattern = #"func\.func\s+@(\w+)\s*\("#
        guard let funcRegex = try? NSRegularExpression(pattern: funcPattern),
              let funcMatch = funcRegex.firstMatch(in: mlir, range: NSRange(mlir.startIndex..., in: mlir)),
              let nameRange = Range(funcMatch.range(at: 1), in: mlir) else {
            throw InterpretParseError.noFunctionFound
        }
        let funcName = String(mlir[nameRange])

        // Extract the primary operation being tested
        let operation = extractOperation(from: mlir, funcName: funcName, filename: filename)

        // Extract all constants
        let inputs = extractConstants(from: mlir)

        // Extract expected value from check.expect_*
        let (expected, usesApprox) = extractExpected(from: mlir)

        return InterpretTestCase(
            name: funcName,
            operation: operation,
            mlir: mlir,
            inputs: inputs,
            expected: expected,
            usesApproximateComparison: usesApprox
        )
    }

    /// Extract the primary operation being tested
    private func extractOperation(from mlir: String, funcName: String, filename: String) -> String {
        // Try to get operation from filename first (e.g., "abs.mlir" -> "abs")
        let filenameOp = filename.replacingOccurrences(of: ".mlir", with: "")

        // Also look for stablehlo.* operations in the MLIR
        let opPattern = #"stablehlo\.(\w+)"#
        if let regex = try? NSRegularExpression(pattern: opPattern) {
            let matches = regex.matches(in: mlir, range: NSRange(mlir.startIndex..., in: mlir))

            // Find operations that aren't "constant"
            for match in matches {
                if let range = Range(match.range(at: 1), in: mlir) {
                    let op = String(mlir[range])
                    if op != "constant" {
                        return op
                    }
                }
            }
        }

        return filenameOp
    }

    /// Extract all stablehlo.constant definitions
    private func extractConstants(from mlir: String) -> [InterpretConstant] {
        var constants: [InterpretConstant] = []

        // Pattern: %name = stablehlo.constant dense<...> : tensor<...>
        // Need to handle multi-line constants with nested brackets
        let lines = mlir.components(separatedBy: "\n")
        var currentConstant: String? = nil
        var bracketCount = 0

        for line in lines {
            if currentConstant != nil {
                // Continue accumulating multi-line constant
                currentConstant! += "\n" + line
                bracketCount += line.filter { $0 == "<" || $0 == "[" }.count
                bracketCount -= line.filter { $0 == ">" || $0 == "]" }.count

                if bracketCount <= 0 && line.contains(": tensor<") {
                    // End of constant
                    if let constant = parseConstantLine(currentConstant!) {
                        constants.append(constant)
                    }
                    currentConstant = nil
                    bracketCount = 0
                }
            } else if line.contains("stablehlo.constant dense<") {
                // Start of a constant
                bracketCount = line.filter { $0 == "<" || $0 == "[" }.count
                bracketCount -= line.filter { $0 == ">" || $0 == "]" }.count

                if bracketCount <= 0 && line.contains(": tensor<") {
                    // Single-line constant
                    if let constant = parseConstantLine(line) {
                        constants.append(constant)
                    }
                } else {
                    // Multi-line constant
                    currentConstant = line
                }
            }
        }

        return constants
    }

    /// Parse a single constant line/block
    private func parseConstantLine(_ text: String) -> InterpretConstant? {
        // Pattern: %name = stablehlo.constant dense<VALUE> : tensor<TYPE>
        let pattern = #"(%\w+)\s*=\s*stablehlo\.constant\s+dense<([^>]+(?:>[^>]*)*)>\s*:\s*(tensor<[^>]+>)"#

        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]),
              let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              let nameRange = Range(match.range(at: 1), in: text),
              let literalRange = Range(match.range(at: 2), in: text),
              let typeRange = Range(match.range(at: 3), in: text) else {
            return nil
        }

        let name = String(text[nameRange])
        let literal = String(text[literalRange]).trimmingCharacters(in: .whitespacesAndNewlines)
        let tensorType = String(text[typeRange])

        // Parse shape and element type from tensor type
        let (shape, elementType) = parseTensorType(tensorType)

        return InterpretConstant(
            name: name,
            literal: literal,
            tensorType: tensorType,
            shape: shape,
            elementType: elementType
        )
    }

    /// Extract expected value from check.expect_* call
    private func extractExpected(from mlir: String) -> (InterpretConstant?, Bool) {
        // Pattern: check.expect_eq_const %result, dense<VALUE> : tensor<TYPE>
        // Or: check.expect_almost_eq_const %result, dense<VALUE> : tensor<TYPE>

        let usesApprox = mlir.contains("expect_almost_eq")

        let pattern = #"check\.expect_(?:almost_)?eq_const\s+%\w+,\s*dense<([^>]+(?:>[^>]*)*)>\s*:\s*(tensor<[^>]+>)"#

        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]),
              let match = regex.firstMatch(in: mlir, range: NSRange(mlir.startIndex..., in: mlir)),
              let literalRange = Range(match.range(at: 1), in: mlir),
              let typeRange = Range(match.range(at: 2), in: mlir) else {
            return (nil, usesApprox)
        }

        let literal = String(mlir[literalRange]).trimmingCharacters(in: .whitespacesAndNewlines)
        let tensorType = String(mlir[typeRange])
        let (shape, elementType) = parseTensorType(tensorType)

        let expected = InterpretConstant(
            name: "%expected",
            literal: literal,
            tensorType: tensorType,
            shape: shape,
            elementType: elementType
        )

        return (expected, usesApprox)
    }

    /// Parse tensor type string like "tensor<3xi64>" or "tensor<2x2x2xf32>"
    private func parseTensorType(_ typeString: String) -> ([Int], String) {
        // Remove "tensor<" and ">"
        var inner = typeString
            .replacingOccurrences(of: "tensor<", with: "")
            .replacingOccurrences(of: ">", with: "")
            .trimmingCharacters(in: .whitespaces)

        // Handle complex types
        if inner.contains("complex") {
            if inner.contains("f64") {
                let shape = extractShapeFromType(inner, elementType: "complex<f64>")
                return (shape, "complex<f64>")
            } else {
                let shape = extractShapeFromType(inner, elementType: "complex<f32>")
                return (shape, "complex<f32>")
            }
        }

        // List of element types (order matters - longer first)
        let elementTypes = [
            "bf16", "f16", "f32", "f64",
            "f8E3M4", "f8E4M3", "f8E4M3FN", "f8E5M2",
            "ui64", "ui32", "ui16", "ui8", "ui4", "ui2",
            "i64", "i32", "i16", "i8", "i4", "i2", "i1"
        ]

        for elemType in elementTypes {
            if inner.hasSuffix(elemType) {
                let shape = extractShapeFromType(inner, elementType: elemType)
                return (shape, elemType)
            }
        }

        // Scalar or unknown
        return ([], inner)
    }

    /// Extract shape dimensions from type string
    private func extractShapeFromType(_ typeString: String, elementType: String) -> [Int] {
        let shapeString = typeString
            .replacingOccurrences(of: elementType, with: "")
            .trimmingCharacters(in: .whitespaces)

        if shapeString.isEmpty {
            return [] // Scalar
        }

        // Split by 'x' and parse dimensions
        return shapeString.split(separator: "x").compactMap { Int($0) }
    }
}

/// Errors during interpret test parsing
public enum InterpretParseError: Error, CustomStringConvertible {
    case noFunctionFound
    case invalidConstant(String)
    case unsupportedType(String)

    public var description: String {
        switch self {
        case .noFunctionFound:
            return "No function definition found in test section"
        case .invalidConstant(let msg):
            return "Invalid constant: \(msg)"
        case .unsupportedType(let type):
            return "Unsupported type: \(type)"
        }
    }
}
