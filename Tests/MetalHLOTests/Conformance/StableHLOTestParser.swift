// StableHLOTestParser.swift
// MetalHLOTests
//
// Parser for StableHLO test files that extracts inputs, expected outputs,
// and the operation under test from .mlir test files.

import Foundation

/// Represents a parsed StableHLO test case
public struct StableHLOTestCase: Sendable {
    public let name: String
    public let operation: String
    public let inputsMLIR: String
    public let expectedMLIR: String
    public let mainMLIR: String
    public let fullMLIR: String

    /// Input tensor data extracted from the test
    public let inputs: [TensorData]

    /// Expected output tensor data
    public let expected: [TensorData]
}

/// Represents tensor data extracted from a test file
public struct TensorData: Sendable {
    public let shape: [Int]
    public let elementType: ElementType
    public let data: Data

    public enum ElementType: String, Sendable {
        case float16 = "f16"
        case float32 = "f32"
        case float64 = "f64"
        case bfloat16 = "bf16"
        case int8 = "i8"
        case int16 = "i16"
        case int32 = "i32"
        case int64 = "i64"
        case uint8 = "ui8"
        case uint16 = "ui16"
        case uint32 = "ui32"
        case uint64 = "ui64"
        case bool = "i1"
        case complex64 = "complex<f32>"
        case complex128 = "complex<f64>"

        public var byteSize: Int {
            switch self {
            case .bool, .int8, .uint8: return 1
            case .float16, .bfloat16, .int16, .uint16: return 2
            case .float32, .int32, .uint32: return 4
            case .float64, .int64, .uint64, .complex64: return 8
            case .complex128: return 16
            }
        }
    }

    /// Convert to Float array (for float32 data)
    public func toFloatArray() -> [Float]? {
        guard elementType == .float32 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
    }

    /// Convert to Double array (for float64 data)
    public func toDoubleArray() -> [Double]? {
        guard elementType == .float64 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Double.self))
        }
    }

    /// Convert to Int32 array
    public func toInt32Array() -> [Int32]? {
        guard elementType == .int32 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int32.self))
        }
    }

    /// Convert to Int64 array
    public func toInt64Array() -> [Int64]? {
        guard elementType == .int64 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int64.self))
        }
    }

    /// Convert to Int8 array
    public func toInt8Array() -> [Int8]? {
        guard elementType == .int8 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int8.self))
        }
    }

    /// Convert to Int16 array
    public func toInt16Array() -> [Int16]? {
        guard elementType == .int16 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int16.self))
        }
    }

    /// Convert to UInt8 array
    public func toUInt8Array() -> [UInt8]? {
        guard elementType == .uint8 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: UInt8.self))
        }
    }

    /// Convert to UInt16 array
    public func toUInt16Array() -> [UInt16]? {
        guard elementType == .uint16 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: UInt16.self))
        }
    }

    /// Convert to UInt32 array
    public func toUInt32Array() -> [UInt32]? {
        guard elementType == .uint32 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: UInt32.self))
        }
    }

    /// Convert to UInt64 array
    public func toUInt64Array() -> [UInt64]? {
        guard elementType == .uint64 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: UInt64.self))
        }
    }

    /// Total element count
    public var elementCount: Int {
        shape.isEmpty ? 1 : shape.reduce(1, *)
    }
}

/// Parser for StableHLO .mlir test files
public struct StableHLOTestParser {

    public init() {}

    /// Parse a StableHLO test file
    public func parse(contentsOf url: URL) throws -> StableHLOTestCase {
        let content = try String(contentsOf: url, encoding: .utf8)
        return try parse(content: content, name: url.deletingPathExtension().lastPathComponent)
    }

    /// Parse StableHLO test content
    public func parse(content: String, name: String) throws -> StableHLOTestCase {
        // Extract operation name from the test
        let operation = extractOperation(from: content)

        // Extract function bodies
        let inputsMLIR = extractFunction(named: "inputs", from: content) ?? ""
        let expectedMLIR = extractFunction(named: "expected", from: content) ?? ""
        let mainMLIR = extractFunction(named: "main", from: content) ?? ""

        // Parse tensor data from inputs function
        let inputs = try parseTensorConstants(from: inputsMLIR)

        // Parse tensor data from expected function
        let expected = try parseTensorConstants(from: expectedMLIR)

        return StableHLOTestCase(
            name: name,
            operation: operation,
            inputsMLIR: inputsMLIR,
            expectedMLIR: expectedMLIR,
            mainMLIR: mainMLIR,
            fullMLIR: content,
            inputs: inputs,
            expected: expected
        )
    }

    /// Extract the primary operation being tested
    private func extractOperation(from content: String) -> String {
        // Look for stablehlo.XXX pattern in main function
        let pattern = #"stablehlo\.(\w+)"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: content, range: NSRange(content.startIndex..., in: content)),
              let range = Range(match.range(at: 1), in: content) else {
            return "unknown"
        }
        return String(content[range])
    }

    /// Extract a function body by name
    private func extractFunction(named name: String, from content: String) -> String? {
        // Find the function start: func.func [visibility] @name
        let startPattern = #"func\.func\s+(?:public\s+|private\s+)?@"# + name
        guard let startRegex = try? NSRegularExpression(pattern: startPattern),
              let startMatch = startRegex.firstMatch(in: content, range: NSRange(content.startIndex..., in: content)) else {
            return nil
        }

        let funcStartIdx = content.index(content.startIndex, offsetBy: startMatch.range.location)
        let afterMatch = content.index(content.startIndex, offsetBy: startMatch.range.location + startMatch.range.length)

        // Now find the function body by looking for the pattern:
        // @name(...) [-> (...)] { body }
        // We need to find the opening brace of the body, handling nested parens in return type

        // Skip past the parameter list and return type to find the body brace
        // Track nested parens: () for params, () for return type may contain {} for attributes
        var parenCount = 0
        var inParams = false
        var foundBodyBrace: String.Index? = nil

        for idx in content.indices[afterMatch...] {
            let char = content[idx]

            if char == "(" {
                parenCount += 1
                inParams = true
            } else if char == ")" {
                parenCount -= 1
            } else if char == "{" && parenCount == 0 {
                // This is the body brace (not inside parens)
                foundBodyBrace = idx
                break
            }
        }

        guard let bodyStart = foundBodyBrace else {
            return nil
        }

        // Now count braces to find the matching close
        var braceCount = 0
        var bodyEnd: String.Index? = nil

        for idx in content.indices[bodyStart...] {
            let char = content[idx]
            if char == "{" {
                braceCount += 1
            } else if char == "}" {
                braceCount -= 1
                if braceCount == 0 {
                    bodyEnd = content.index(after: idx)
                    break
                }
            }
        }

        guard let end = bodyEnd else {
            return nil
        }

        // Return from function start to the closing brace
        return String(content[funcStartIdx..<end])
    }

    /// Parse tensor constants from function body
    private func parseTensorConstants(from mlir: String) throws -> [TensorData] {
        var tensors: [TensorData] = []

        // Pattern for dense constants with hex data: dense<"0x..."> : tensor<shape x type>
        let hexPattern = #"dense<\"(0x[0-9A-Fa-f]+)\"\s*>\s*:\s*tensor<([^>]+)>"#
        if let regex = try? NSRegularExpression(pattern: hexPattern) {
            let matches = regex.matches(in: mlir, range: NSRange(mlir.startIndex..., in: mlir))
            for match in matches {
                if let hexRange = Range(match.range(at: 1), in: mlir),
                   let typeRange = Range(match.range(at: 2), in: mlir) {
                    let hexString = String(mlir[hexRange])
                    let typeString = String(mlir[typeRange])

                    let (shape, elementType) = try parseTypeString(typeString)
                    let data = try parseHexData(hexString)

                    tensors.append(TensorData(shape: shape, elementType: elementType, data: data))
                }
            }
        }

        // Parse literal constants using bracket matching (handles nested arrays)
        // Find "dense<" followed by content and "> : tensor<type>"
        let densePrefix = "dense<"
        var searchStart = mlir.startIndex

        while let prefixRange = mlir.range(of: densePrefix, range: searchStart..<mlir.endIndex) {
            let contentStart = prefixRange.upperBound

            // Check if it's hex data (already handled above)
            if mlir[contentStart] == "\"" {
                searchStart = contentStart
                continue
            }

            // Find the matching '>' by counting brackets
            var bracketCount = 0
            var inString = false
            var contentEnd: String.Index? = nil

            for idx in mlir.indices[contentStart...] {
                let char = mlir[idx]

                if char == "\"" {
                    inString = !inString
                } else if !inString {
                    if char == "[" {
                        bracketCount += 1
                    } else if char == "]" {
                        bracketCount -= 1
                    } else if char == ">" && bracketCount == 0 {
                        contentEnd = idx
                        break
                    }
                }
            }

            guard let endIdx = contentEnd else {
                searchStart = contentStart
                continue
            }

            let literalContent = String(mlir[contentStart..<endIdx])

            // Now find the type annotation: "> : tensor<type>"
            let afterContent = mlir.index(after: endIdx)
            let typePattern = #"\s*:\s*tensor<([^>]+)>"#
            let remaining = String(mlir[afterContent...])

            if let typeRegex = try? NSRegularExpression(pattern: typePattern),
               let typeMatch = typeRegex.firstMatch(in: remaining, range: NSRange(remaining.startIndex..., in: remaining)),
               let typeRange = Range(typeMatch.range(at: 1), in: remaining) {

                let typeString = String(remaining[typeRange])
                let (shape, elementType) = try parseTypeString(typeString)
                let data = try parseLiteralData(literalContent, elementType: elementType, shape: shape)

                tensors.append(TensorData(shape: shape, elementType: elementType, data: data))
            }

            searchStart = endIdx
        }

        return tensors
    }

    /// Parse tensor type string like "20x20xf32" or "f32" (scalar)
    private func parseTypeString(_ typeString: String) throws -> ([Int], TensorData.ElementType) {
        let trimmed = typeString.trimmingCharacters(in: .whitespaces)

        // Handle complex types
        if trimmed.contains("complex") {
            if trimmed.contains("f64") {
                let shape = extractShape(from: trimmed, removing: "complex<f64>")
                return (shape, .complex128)
            } else {
                let shape = extractShape(from: trimmed, removing: "complex<f32>")
                return (shape, .complex64)
            }
        }

        // Map of element type suffixes
        // IMPORTANT: Order matters for hasSuffix matching!
        // ui8 ends with "i8", so unsigned types must come BEFORE signed types
        let typeMap: [(String, TensorData.ElementType)] = [
            ("bf16", .bfloat16),
            ("f16", .float16),
            ("f32", .float32),
            ("f64", .float64),
            ("i1", .bool),
            // Unsigned types first (ui8 contains "i8" as suffix)
            ("ui8", .uint8),
            ("ui16", .uint16),
            ("ui32", .uint32),
            ("ui64", .uint64),
            // Signed types after
            ("i8", .int8),
            ("i16", .int16),
            ("i32", .int32),
            ("i64", .int64),
        ]

        for (suffix, elementType) in typeMap {
            if trimmed.hasSuffix(suffix) {
                let shapeString = String(trimmed.dropLast(suffix.count))
                let shape = parseShapeString(shapeString)
                return (shape, elementType)
            }
        }

        throw StableHLOParseError.unknownElementType(typeString)
    }

    /// Extract shape from type string, removing the element type
    private func extractShape(from typeString: String, removing elementType: String) -> [Int] {
        let withoutType = typeString.replacingOccurrences(of: elementType, with: "")
        return parseShapeString(withoutType)
    }

    /// Parse shape string like "20x20x" or "" (scalar)
    private func parseShapeString(_ shapeString: String) -> [Int] {
        let trimmed = shapeString.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty {
            return [] // scalar
        }

        // Split by 'x' and parse dimensions
        let parts = trimmed.split(separator: "x").compactMap { Int($0) }
        return parts
    }

    /// Parse hex string to Data
    private func parseHexData(_ hexString: String) throws -> Data {
        var hex = hexString
        if hex.hasPrefix("0x") || hex.hasPrefix("0X") {
            hex = String(hex.dropFirst(2))
        }

        var data = Data()
        var index = hex.startIndex

        while index < hex.endIndex {
            let nextIndex = hex.index(index, offsetBy: 2, limitedBy: hex.endIndex) ?? hex.endIndex
            let byteString = String(hex[index..<nextIndex])

            guard let byte = UInt8(byteString, radix: 16) else {
                throw StableHLOParseError.invalidHexData(hexString)
            }
            data.append(byte)
            index = nextIndex
        }

        return data
    }

    /// Parse literal data like "[[1.0, 2.0], [3.0, 4.0]]" or "0.0"
    private func parseLiteralData(_ literal: String, elementType: TensorData.ElementType, shape: [Int]) throws -> Data {
        // Remove all brackets and split by comma to get flat list of values
        var cleaned = literal
            .replacingOccurrences(of: "[", with: "")
            .replacingOccurrences(of: "]", with: "")
            .trimmingCharacters(in: .whitespaces)

        // Handle empty tensor
        if cleaned.isEmpty {
            return Data()
        }

        // Split by comma, handling potential whitespace
        let values = cleaned.split(separator: ",").map {
            String($0).trimmingCharacters(in: .whitespaces)
        }.filter { !$0.isEmpty }

        // Calculate expected number of elements from shape
        let expectedCount = shape.isEmpty ? 1 : shape.reduce(1, *)

        // Check if this is a splat constant (single value to be replicated)
        let isSplat = values.count == 1 && expectedCount > 1
        let repeatCount = isSplat ? expectedCount : 1

        var data = Data()

        for value in values {
            // For splat constants, repeat the value to fill the tensor
            for _ in 0..<repeatCount {
                // Check if value is a hex bit pattern (e.g., "0x7F800000" for infinity)
                let isHexBitPattern = value.lowercased().hasPrefix("0x")

                switch elementType {
                case .float16:
                    if isHexBitPattern {
                        // Parse as hex bit pattern
                        guard let bits = UInt16(value.dropFirst(2), radix: 16) else {
                            throw StableHLOParseError.invalidLiteralValue(value)
                        }
                        withUnsafeBytes(of: bits) { data.append(contentsOf: $0) }
                    } else {
                        // Parse as Float first, then convert to Float16
                        guard let f = Float(value) else {
                            throw StableHLOParseError.invalidLiteralValue(value)
                        }
                        let f16 = Float16(f)
                        withUnsafeBytes(of: f16) { data.append(contentsOf: $0) }
                    }

                case .bfloat16:
                    if isHexBitPattern {
                        // Parse as hex bit pattern
                        guard let bits = UInt16(value.dropFirst(2), radix: 16) else {
                            throw StableHLOParseError.invalidLiteralValue(value)
                        }
                        withUnsafeBytes(of: bits) { data.append(contentsOf: $0) }
                    } else {
                        // Parse as Float first, then convert to bfloat16 (truncate to upper 16 bits)
                        guard let f = Float(value) else {
                            throw StableHLOParseError.invalidLiteralValue(value)
                        }
                        let bits = f.bitPattern
                        let bf16bits = UInt16(bits >> 16)
                        withUnsafeBytes(of: bf16bits) { data.append(contentsOf: $0) }
                    }

                case .float32:
                    if isHexBitPattern {
                        // Parse as hex bit pattern (e.g., 0x7F800000 for infinity)
                        guard let bits = UInt32(value.dropFirst(2), radix: 16) else {
                            throw StableHLOParseError.invalidLiteralValue(value)
                        }
                        let f = Float(bitPattern: bits)
                        withUnsafeBytes(of: f) { data.append(contentsOf: $0) }
                    } else {
                        guard let f = Float(value) else {
                            throw StableHLOParseError.invalidLiteralValue(value)
                        }
                        withUnsafeBytes(of: f) { data.append(contentsOf: $0) }
                    }

            case .float64:
                if isHexBitPattern {
                    // Parse as hex bit pattern
                    guard let bits = UInt64(value.dropFirst(2), radix: 16) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    let d = Double(bitPattern: bits)
                    withUnsafeBytes(of: d) { data.append(contentsOf: $0) }
                } else {
                    guard let d = Double(value) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: d) { data.append(contentsOf: $0) }
                }

            case .int8:
                if isHexBitPattern {
                    guard let bits = UInt8(value.dropFirst(2), radix: 16) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    let i = Int8(bitPattern: bits)
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                } else {
                    guard let i = Int8(value) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                }

            case .int16:
                if isHexBitPattern {
                    guard let bits = UInt16(value.dropFirst(2), radix: 16) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    let i = Int16(bitPattern: bits)
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                } else {
                    guard let i = Int16(value) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                }

            case .int32:
                if isHexBitPattern {
                    guard let bits = UInt32(value.dropFirst(2), radix: 16) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    let i = Int32(bitPattern: bits)
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                } else {
                    guard let i = Int32(value) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                }

            case .int64:
                if isHexBitPattern {
                    guard let bits = UInt64(value.dropFirst(2), radix: 16) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    let i = Int64(bitPattern: bits)
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                } else {
                    guard let i = Int64(value) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                }

            case .uint8:
                if isHexBitPattern {
                    guard let i = UInt8(value.dropFirst(2), radix: 16) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                } else {
                    guard let i = UInt8(value) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                }

            case .uint16:
                if isHexBitPattern {
                    guard let i = UInt16(value.dropFirst(2), radix: 16) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                } else {
                    guard let i = UInt16(value) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                }

            case .uint32:
                if isHexBitPattern {
                    guard let i = UInt32(value.dropFirst(2), radix: 16) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                } else {
                    guard let i = UInt32(value) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                }

            case .uint64:
                if isHexBitPattern {
                    guard let i = UInt64(value.dropFirst(2), radix: 16) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                } else {
                    guard let i = UInt64(value) else {
                        throw StableHLOParseError.invalidLiteralValue(value)
                    }
                    withUnsafeBytes(of: i) { data.append(contentsOf: $0) }
                }

            case .bool:
                let b: UInt8 = (value == "true" || value == "1") ? 1 : 0
                data.append(b)

            case .complex64, .complex128:
                throw StableHLOParseError.unsupportedElementType("complex types require special parsing")
                }
            }
        }

        return data
    }
}

/// Errors during StableHLO test parsing
public enum StableHLOParseError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case invalidHexData(String)
    case invalidLiteralValue(String)
    case unknownElementType(String)
    case unsupportedElementType(String)
    case malformedTestFile(String)

    public var description: String {
        switch self {
        case .fileNotFound(let path):
            return "Test file not found: \(path)"
        case .invalidHexData(let data):
            return "Invalid hex data: \(data.prefix(50))..."
        case .invalidLiteralValue(let value):
            return "Invalid literal value: \(value)"
        case .unknownElementType(let type):
            return "Unknown element type: \(type)"
        case .unsupportedElementType(let type):
            return "Unsupported element type for parsing: \(type)"
        case .malformedTestFile(let reason):
            return "Malformed test file: \(reason)"
        }
    }
}
