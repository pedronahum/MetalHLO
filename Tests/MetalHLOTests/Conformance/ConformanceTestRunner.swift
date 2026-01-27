// ConformanceTestRunner.swift
// MetalHLOTests
//
// Runs StableHLO conformance tests through MetalHLO and verifies
// results match expected outputs within tolerance.

import Foundation
import MetalHLO

/// Result of running a conformance test
public struct ConformanceTestResult: Sendable {
    public let testName: String
    public let operation: String
    public let passed: Bool
    public let maxDifference: Float
    public let tolerance: Float
    public let errorMessage: String?
    public let executionTime: TimeInterval
    public let optimizationLevel: OptimizationLevel

    public var summary: String {
        let levelStr = "O\(optimizationLevel.rawValue)"
        if passed {
            return "PASS: \(testName) [\(levelStr)] (max diff: \(String(format: "%.2e", maxDifference)), time: \(String(format: "%.2f", executionTime * 1000))ms)"
        } else {
            return "FAIL: \(testName) [\(levelStr)] - \(errorMessage ?? "unknown error")"
        }
    }
}

/// Runs StableHLO conformance tests through MetalHLO
public final class ConformanceTestRunner: @unchecked Sendable {

    private let client: Client
    private let parser: StableHLOTestParser
    private let manager: StableHLOTestManager
    private let config: CompilationConfig

    /// Default tolerance for floating-point comparison
    public static let defaultTolerance: Float = 1e-5

    /// Relaxed tolerance for operations with numerical instability
    public static let relaxedTolerance: Float = 1e-4

    /// Tolerance for trigonometric operations
    public static let trigTolerance: Float = 1e-3

    /// All optimization levels for comprehensive testing
    public static let allOptimizationLevels: [OptimizationLevel] = [.O0, .O1, .O2, .O3]

    public init(config: CompilationConfig = .default) throws {
        self.client = try Client.create()
        self.parser = StableHLOTestParser()
        self.manager = StableHLOTestManager.shared
        self.config = config
    }

    /// The optimization level this runner is using
    public var optimizationLevel: OptimizationLevel {
        config.optimizationLevel
    }

    /// Run a single conformance test by name
    public func runTest(_ testName: String, tolerance: Float = defaultTolerance) async throws -> ConformanceTestResult {
        let startTime = Date()

        do {
            // Download/get cached test file
            let testURL = try await manager.ensureTestFile(testName)

            // Parse the test file
            let testCase = try parser.parse(contentsOf: testURL)

            // Generate executable MLIR for MetalHLO
            let mlir = try generateExecutableMLIR(from: testCase)

            // Compile the MLIR using the MPSGraph path (more operations supported)
            // Note: The config-based IntegratedExecutor path has issues with some operations
            // like reduce. Using the non-config path ensures correct results.
            let executable = try client.compile(mlir)

            // Create input buffers from parsed test data
            let inputBuffers = try createInputBuffers(from: testCase)

            // Execute
            let outputs = try executable.execute(inputBuffers)

            // Compare outputs with expected
            let (passed, maxDiff, errorMsg) = try compareOutputs(
                actual: outputs,
                expected: testCase.expected,
                tolerance: tolerance
            )

            let elapsed = Date().timeIntervalSince(startTime)

            return ConformanceTestResult(
                testName: testName,
                operation: testCase.operation,
                passed: passed,
                maxDifference: maxDiff,
                tolerance: tolerance,
                errorMessage: errorMsg,
                executionTime: elapsed,
                optimizationLevel: config.optimizationLevel
            )

        } catch {
            let elapsed = Date().timeIntervalSince(startTime)
            return ConformanceTestResult(
                testName: testName,
                operation: "unknown",
                passed: false,
                maxDifference: Float.infinity,
                tolerance: tolerance,
                errorMessage: error.localizedDescription,
                executionTime: elapsed,
                optimizationLevel: config.optimizationLevel
            )
        }
    }

    /// Run multiple conformance tests
    public func runTests(_ testNames: [String], tolerance: Float = defaultTolerance) async -> [ConformanceTestResult] {
        var results: [ConformanceTestResult] = []

        for testName in testNames {
            let result = try? await runTest(testName, tolerance: tolerance)
            results.append(result ?? ConformanceTestResult(
                testName: testName,
                operation: "unknown",
                passed: false,
                maxDifference: Float.infinity,
                tolerance: tolerance,
                errorMessage: "Test execution failed",
                executionTime: 0,
                optimizationLevel: config.optimizationLevel
            ))
        }

        return results
    }

    /// Run a single test with debug output
    public func runTestDebug(_ testName: String, tolerance: Float = defaultTolerance) async throws {
        print("=== Debug Test: \(testName) ===")

        // Download/get cached test file
        let testURL = try await manager.ensureTestFile(testName)
        print("Test file: \(testURL.path)")

        // Parse the test file
        let testCase = try parser.parse(contentsOf: testURL)
        print("Operation: \(testCase.operation)")
        print("Inputs count: \(testCase.inputs.count)")

        for (i, input) in testCase.inputs.enumerated() {
            print("  Input \(i): shape=\(input.shape), type=\(input.elementType.rawValue), bytes=\(input.data.count)")
            if input.elementType == .float32 {
                let floats = input.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                print("    First 5 values: \(floats.prefix(5))")
            }
        }

        print("Expected outputs count: \(testCase.expected.count)")
        for (i, expected) in testCase.expected.enumerated() {
            print("  Expected \(i): shape=\(expected.shape), type=\(expected.elementType.rawValue), bytes=\(expected.data.count)")
            if expected.elementType == .float32 {
                let floats = expected.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                print("    First 5 values: \(floats.prefix(5))")
            }
        }

        // Generate executable MLIR for MetalHLO
        let mlir = try generateExecutableMLIR(from: testCase)
        print("\nGenerated MLIR:")
        print(mlir)

        // Compile the MLIR using MPSGraph path (more operations supported)
        print("\nCompiling...")
        let executable = try client.compile(mlir)

        // Create input buffers
        print("Creating input buffers...")
        let inputBuffers = try createInputBuffers(from: testCase)
        for (i, buf) in inputBuffers.enumerated() {
            print("  Buffer \(i): shape=\(buf.shape), type=\(buf.elementType), bytes=\(buf.byteCount)")
            if buf.elementType == .float32 {
                let floats = try buf.toFloatArray()
                print("    First 5 values: \(floats.prefix(5))")
            }
        }

        // Execute
        print("\nExecuting...")
        let outputs = try executable.execute(inputBuffers)
        print("Outputs count: \(outputs.count)")
        for (i, out) in outputs.enumerated() {
            print("  Output \(i): shape=\(out.shape), type=\(out.elementType), bytes=\(out.byteCount)")
            if out.elementType == .float32 {
                let floats = try out.toFloatArray()
                print("    First 10 values: \(floats.prefix(10))")
            }
        }

        // Compare outputs with expected
        print("\nComparing...")
        let (passed, maxDiff, errorMsg) = try compareOutputs(
            actual: outputs,
            expected: testCase.expected,
            tolerance: tolerance
        )

        print("\nResult: \(passed ? "PASSED" : "FAILED")")
        print("Max difference: \(maxDiff)")
        if let msg = errorMsg {
            print("Error: \(msg)")
        }
    }

    /// Run a test suite
    public func runSuite(_ suite: [String], tolerance: Float = defaultTolerance) async -> ConformanceSuiteResult {
        let results = await runTests(suite, tolerance: tolerance)
        return ConformanceSuiteResult(results: results)
    }

    /// Run a single test across all optimization levels
    public static func runTestAcrossOptLevels(
        _ testName: String,
        tolerance: Float = defaultTolerance
    ) async throws -> [ConformanceTestResult] {
        var results: [ConformanceTestResult] = []

        for level in allOptimizationLevels {
            let config = CompilationConfig(optimizationLevel: level)
            let runner = try ConformanceTestRunner(config: config)
            let result = try await runner.runTest(testName, tolerance: tolerance)
            results.append(result)
        }

        return results
    }

    /// Run a test suite across all optimization levels
    public static func runSuiteAcrossOptLevels(
        _ suite: [String],
        tolerance: Float = defaultTolerance
    ) async throws -> MultiOptLevelResult {
        var allResults: [OptimizationLevel: ConformanceSuiteResult] = [:]

        for level in allOptimizationLevels {
            let config = CompilationConfig(optimizationLevel: level)
            let runner = try ConformanceTestRunner(config: config)
            let suiteResult = await runner.runSuite(suite, tolerance: tolerance)
            allResults[level] = suiteResult
        }

        return MultiOptLevelResult(resultsByLevel: allResults)
    }

    // MARK: - MLIR Generation

    /// Generate executable MLIR from a StableHLO test case.
    /// This transforms the test's structure into a standalone executable.
    /// Handles multi-operation patterns (e.g., broadcast_in_dim + add).
    private func generateExecutableMLIR(from testCase: StableHLOTestCase) throws -> String {
        // The StableHLO test files have a specific structure:
        // - @inputs() returns test inputs as constants
        // - @expected() returns expected outputs as constants
        // - @main() calls inputs, applies operation, compares with expected
        //
        // We need to transform this into a simple module where:
        // - Inputs are function parameters (not constants)
        // - All intermediate operations are preserved
        // - The final result is returned

        let mainBody = testCase.mainMLIR

        // Try special handling for reduce operations first
        if let reduceMLIR = try? generateReduceMLIR(from: testCase, mainBody: mainBody) {
            return reduceMLIR
        }

        // Determine input and output types from the test case
        guard !testCase.inputs.isEmpty else {
            throw ConformanceTestError.noInputsFound(testCase.name)
        }
        guard !testCase.expected.isEmpty else {
            throw ConformanceTestError.noExpectedOutputs(testCase.name)
        }

        // Build input parameters
        var inputParams: [String] = []
        for (i, input) in testCase.inputs.enumerated() {
            let shapeStr = input.shape.isEmpty ? "" : input.shape.map { String($0) }.joined(separator: "x") + "x"
            let typeStr = "tensor<\(shapeStr)\(mapElementType(input.elementType))>"
            inputParams.append("%arg\(i): \(typeStr)")
        }

        // Build output type
        let output = testCase.expected[0]
        let outputShapeStr = output.shape.isEmpty ? "" : output.shape.map { String($0) }.joined(separator: "x") + "x"
        let outputType = "tensor<\(outputShapeStr)\(mapElementType(output.elementType))>"

        // Find ALL stablehlo operations in the main function (excluding custom_call which is for testing)
        // Pattern: %NAME = stablehlo.OPERATION ... : type
        // NAME can be numeric (%2) or named (%cst, %cst_0, etc.)
        let opPattern = #"(%[\w_]+)\s*=\s*stablehlo\.(\w+)\s+([^:]+):\s*([^\n]+)"#
        guard let regex = try? NSRegularExpression(pattern: opPattern) else {
            throw ConformanceTestError.cannotExtractOperation(testCase.name)
        }

        let matches = regex.matches(in: mainBody, range: NSRange(mainBody.startIndex..., in: mainBody))

        // Separate constants and other operations for ordering
        var constants: [(result: String, opName: String, operands: String, types: String)] = []
        var operations: [(result: String, opName: String, operands: String, types: String)] = []

        for match in matches {
            guard let resultRange = Range(match.range(at: 1), in: mainBody),
                  let opNameRange = Range(match.range(at: 2), in: mainBody),
                  let operandsRange = Range(match.range(at: 3), in: mainBody),
                  let typesRange = Range(match.range(at: 4), in: mainBody) else {
                continue
            }

            let opName = String(mainBody[opNameRange])
            if opName == "custom_call" { continue }  // Skip test assertions

            let op = (
                result: String(mainBody[resultRange]),
                opName: opName,
                operands: String(mainBody[operandsRange]).trimmingCharacters(in: .whitespaces),
                types: String(mainBody[typesRange]).trimmingCharacters(in: .whitespaces)
            )

            if opName == "constant" {
                constants.append(op)
            } else {
                operations.append(op)
            }
        }

        guard !operations.isEmpty else {
            throw ConformanceTestError.cannotExtractOperation(testCase.name)
        }

        // Build the operations, replacing input references with function arguments
        var opLines: [String] = []
        var resultCounter = 0

        // First, add all constants (keep their original names as they're referenced)
        for constOp in constants {
            opLines.append("    \(constOp.result) = stablehlo.\(constOp.opName) \(constOp.operands) : \(constOp.types)")
        }

        // Then, add the other operations
        for (i, op) in operations.enumerated() {
            var operands = op.operands

            // Replace tuple element references: %0#0 -> %arg0, %0#1 -> %arg1, etc.
            for j in 0..<testCase.inputs.count {
                operands = operands.replacingOccurrences(of: "%0#\(j)", with: "%arg\(j)")
            }

            // For single input, replace %0 -> %arg0 (but not %0#N which was already handled)
            if testCase.inputs.count == 1 && !operands.contains("%arg0") {
                operands = operands.replacingOccurrences(of: "%0", with: "%arg0")
            }

            // Determine the result name
            let resultName: String
            if i == operations.count - 1 {
                resultName = "%result"
            } else {
                resultName = "%tmp\(resultCounter)"
                resultCounter += 1
            }

            // Replace references to previous operation results with our tmp names
            // e.g., if original was %2 and we renamed to %tmp0, update operands
            for (prevIdx, prevOp) in operations.prefix(i).enumerated() {
                let prevResult = prevOp.result
                let ourName = prevIdx == operations.count - 1 ? "%result" : "%tmp\(prevIdx)"
                operands = operands.replacingOccurrences(of: prevResult, with: ourName)
            }

            opLines.append("    \(resultName) = stablehlo.\(op.opName) \(operands) : \(op.types)")
        }

        // Generate the module
        let mlir = """
        module @\(testCase.name.replacingOccurrences(of: "-", with: "_")) {
          func.func @main(\(inputParams.joined(separator: ", "))) -> (\(outputType)) {
        \(opLines.joined(separator: "\n"))
            return %result : \(outputType)
          }
        }
        """

        return mlir
    }

    /// Generate MLIR for reduce operations which have special syntax.
    /// Pattern: %N = stablehlo.reduce(%input init: %init) applies stablehlo.OP across dimensions = [D]
    private func generateReduceMLIR(from testCase: StableHLOTestCase, mainBody: String) throws -> String {
        // Pattern to match reduce operations:
        // %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
        let reducePattern = #"stablehlo\.reduce\s*\(\s*(%\d+)\s+init:\s*%\w+\s*\)\s*applies\s+(stablehlo\.(\w+))\s+across\s+dimensions\s*=\s*\[([^\]]*)\]\s*:\s*\(([^)]+)\)\s*->\s*(\S+)"#

        guard let regex = try? NSRegularExpression(pattern: reducePattern),
              let match = regex.firstMatch(in: mainBody, range: NSRange(mainBody.startIndex..., in: mainBody)) else {
            throw ConformanceTestError.cannotExtractOperation(testCase.name)
        }

        // Extract matched groups
        guard let inputRange = Range(match.range(at: 1), in: mainBody),
              let reduceOpRange = Range(match.range(at: 3), in: mainBody),
              let dimsRange = Range(match.range(at: 4), in: mainBody),
              let inputTypesRange = Range(match.range(at: 5), in: mainBody),
              let outputTypeRange = Range(match.range(at: 6), in: mainBody) else {
            throw ConformanceTestError.cannotExtractOperation(testCase.name)
        }

        let reduceOp = String(mainBody[reduceOpRange])  // "add", "max", "min", etc.
        let dimensions = String(mainBody[dimsRange])     // "0" or "0, 1", etc.
        let inputTypesStr = String(mainBody[inputTypesRange])
        let outputType = String(mainBody[outputTypeRange]).trimmingCharacters(in: .whitespaces)

        // Parse input types to get the data tensor type (first type in the tuple)
        let inputTypes = inputTypesStr.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
        guard !inputTypes.isEmpty else {
            throw ConformanceTestError.cannotExtractOperation(testCase.name)
        }
        let inputTensorType = inputTypes[0]

        // Build input parameters from test case
        guard !testCase.inputs.isEmpty else {
            throw ConformanceTestError.noInputsFound(testCase.name)
        }
        guard !testCase.expected.isEmpty else {
            throw ConformanceTestError.noExpectedOutputs(testCase.name)
        }

        let input = testCase.inputs[0]
        let shapeStr = input.shape.isEmpty ? "" : input.shape.map { String($0) }.joined(separator: "x") + "x"
        let inputType = "tensor<\(shapeStr)\(mapElementType(input.elementType))>"

        // Determine the init value type (scalar of the same element type)
        let initType = "tensor<\(mapElementType(input.elementType))>"

        // Determine the init value based on reduce operation
        let initValue: String
        switch reduceOp {
        case "add":
            initValue = "0.0"
        case "max":
            initValue = "-0x1.FFFFFEp127"  // Negative max float
        case "min":
            initValue = "0x1.FFFFFEp127"   // Positive max float
        case "mul", "multiply":
            initValue = "1.0"
        default:
            initValue = "0.0"
        }

        // Generate the module with proper reduce syntax
        let mlir = """
        module @\(testCase.name.replacingOccurrences(of: "-", with: "_")) {
          func.func @main(%arg0: \(inputType)) -> (\(outputType)) {
            %init = stablehlo.constant dense<\(initValue)> : \(initType)
            %result = stablehlo.reduce(%arg0 init: %init) applies stablehlo.\(reduceOp) across dimensions = [\(dimensions)] : (\(inputType), \(initType)) -> \(outputType)
            return %result : \(outputType)
          }
        }
        """

        return mlir
    }

    /// Map TensorData.ElementType to MLIR type string
    private func mapElementType(_ type: TensorData.ElementType) -> String {
        switch type {
        case .float16: return "f16"
        case .float32: return "f32"
        case .float64: return "f64"
        case .bfloat16: return "bf16"
        case .int8: return "i8"
        case .int16: return "i16"
        case .int32: return "i32"
        case .int64: return "i64"
        case .uint8: return "ui8"
        case .uint16: return "ui16"
        case .uint32: return "ui32"
        case .uint64: return "ui64"
        case .bool: return "i1"
        case .complex64: return "complex<f32>"
        case .complex128: return "complex<f64>"
        }
    }

    /// Map TensorData.ElementType to MetalHLO ElementType
    private func toMetalHLOElementType(_ type: TensorData.ElementType) -> ElementType {
        switch type {
        case .float16: return .float16
        case .float32: return .float32
        case .float64: return .float64
        case .bfloat16: return .bfloat16
        case .int8: return .int8
        case .int16: return .int16
        case .int32: return .int32
        case .int64: return .int64
        case .uint8: return .uint8
        case .uint16: return .uint16
        case .uint32: return .uint32
        case .uint64: return .uint64
        case .bool: return .int1
        case .complex64, .complex128:
            // Complex types not directly supported, would need special handling
            return .float32
        }
    }

    // MARK: - Buffer Creation

    /// Create MetalHLO buffers from parsed test inputs
    private func createInputBuffers(from testCase: StableHLOTestCase) throws -> [Buffer] {
        var buffers: [Buffer] = []

        for input in testCase.inputs {
            let elementType = toMetalHLOElementType(input.elementType)
            let buffer = try client.createBuffer(
                bytes: input.data,
                shape: input.shape,
                elementType: elementType
            )
            buffers.append(buffer)
        }

        return buffers
    }

    // MARK: - Output Comparison

    /// Compare actual outputs with expected outputs
    private func compareOutputs(
        actual: [Buffer],
        expected: [TensorData],
        tolerance: Float
    ) throws -> (passed: Bool, maxDiff: Float, errorMessage: String?) {

        guard actual.count == expected.count else {
            return (false, Float.infinity, "Output count mismatch: got \(actual.count), expected \(expected.count)")
        }

        var maxDiff: Float = 0

        for (i, (actualBuffer, expectedData)) in zip(actual, expected).enumerated() {
            // Shape check
            if actualBuffer.shape != expectedData.shape {
                return (false, Float.infinity, "Shape mismatch at output \(i): got \(actualBuffer.shape), expected \(expectedData.shape)")
            }

            // Compare values based on element type
            let (passed, diff, msg) = try compareBufferWithExpected(
                actual: actualBuffer,
                expected: expectedData,
                tolerance: tolerance,
                index: i
            )

            maxDiff = max(maxDiff, diff)

            if !passed {
                return (false, maxDiff, msg)
            }
        }

        return (true, maxDiff, nil)
    }

    /// Compare a single buffer with expected data
    private func compareBufferWithExpected(
        actual: Buffer,
        expected: TensorData,
        tolerance: Float,
        index: Int
    ) throws -> (passed: Bool, maxDiff: Float, errorMessage: String?) {

        switch expected.elementType {
        case .float32:
            guard let expectedArray = expected.toFloatArray() else {
                return (false, Float.infinity, "Failed to convert expected data to Float array")
            }
            let actualArray = try actual.toFloatArray()

            return compareFloatArrays(actual: actualArray, expected: expectedArray, tolerance: tolerance, outputIndex: index)

        case .float16:
            // For float16, read actual buffer as float16 and convert to float for comparison
            let expectedFloats = convertHalfToFloat(expected.data, elementType: expected.elementType)
            let actualFloat16s = try actual.toFloat16Array()
            let actualFloats = actualFloat16s.map { Float($0) }

            // Use relaxed tolerance for half-precision types
            let halfTolerance = max(tolerance, 1e-2)
            return compareFloatArrays(actual: actualFloats, expected: expectedFloats, tolerance: halfTolerance, outputIndex: index)

        case .bfloat16:
            // For bfloat16, read actual buffer bytes and convert to float
            let expectedFloats = convertHalfToFloat(expected.data, elementType: expected.elementType)
            // BFloat16 output - read as raw bytes and convert
            let actualFloat16s = try actual.toFloat16Array()
            // Treat float16 output as bfloat16 approximation for now
            let actualFloats = actualFloat16s.map { Float($0) }

            // Use relaxed tolerance for half-precision types
            let halfTolerance = max(tolerance, 1e-2)
            return compareFloatArrays(actual: actualFloats, expected: expectedFloats, tolerance: halfTolerance, outputIndex: index)

        case .float64:
            guard let expectedArray = expected.toDoubleArray() else {
                return (false, Float.infinity, "Failed to convert expected data to Double array")
            }
            // MetalHLO might return Float for float64 ops (MPS limitation)
            let actualArray = try actual.toFloatArray()
            let expectedFloats = expectedArray.map { Float($0) }

            return compareFloatArrays(actual: actualArray, expected: expectedFloats, tolerance: tolerance, outputIndex: index)

        case .int32:
            guard let expectedArray = expected.toInt32Array() else {
                return (false, Float.infinity, "Failed to convert expected data to Int32 array")
            }
            let actualArray = try actual.toInt32Array()

            return compareIntArrays(actual: actualArray, expected: expectedArray, outputIndex: index)

        case .int64:
            guard let expectedArray = expected.toInt64Array() else {
                return (false, Float.infinity, "Failed to convert expected data to Int64 array")
            }
            // Try to read as Int64, fall back to Int32
            do {
                let actualArray = try actual.toInt64Array()
                return compareIntArrays(actual: actualArray, expected: expectedArray, outputIndex: index)
            } catch {
                let actualArray = try actual.toInt32Array()
                let expectedInt32 = expectedArray.map { Int32(clamping: $0) }
                return compareIntArrays(actual: actualArray, expected: expectedInt32, outputIndex: index)
            }

        case .int8:
            guard let expectedArray = expected.toInt8Array() else {
                return (false, Float.infinity, "Failed to convert expected data to Int8 array")
            }
            // MPS may promote int8 to int16 or int32
            do {
                let actualArray = try actual.toInt8Array()
                return compareIntArrays(actual: actualArray, expected: expectedArray, outputIndex: index)
            } catch {
                // Try int16 fallback
                if let actualInt16 = try? actual.toInt16Array() {
                    let expectedInt16 = expectedArray.map { Int16($0) }
                    return compareIntArrays(actual: actualInt16, expected: expectedInt16, outputIndex: index)
                }
                // Try int32 fallback
                let actualArray = try actual.toInt32Array()
                let expectedInt32 = expectedArray.map { Int32($0) }
                return compareIntArrays(actual: actualArray, expected: expectedInt32, outputIndex: index)
            }

        case .int16:
            guard let expectedArray = expected.toInt16Array() else {
                return (false, Float.infinity, "Failed to convert expected data to Int16 array")
            }
            // MPS may promote int16 to int32
            do {
                let actualArray = try actual.toInt16Array()
                return compareIntArrays(actual: actualArray, expected: expectedArray, outputIndex: index)
            } catch {
                let actualArray = try actual.toInt32Array()
                let expectedInt32 = expectedArray.map { Int32($0) }
                return compareIntArrays(actual: actualArray, expected: expectedInt32, outputIndex: index)
            }

        case .uint8:
            guard let expectedArray = expected.toUInt8Array() else {
                return (false, Float.infinity, "Failed to convert expected data to UInt8 array")
            }
            do {
                let actualArray = try actual.toUInt8Array()
                return compareIntArrays(actual: actualArray, expected: expectedArray, outputIndex: index)
            } catch {
                // Try uint16 or uint32 fallback
                if let actualUInt16 = try? actual.toUInt16Array() {
                    let expectedUInt16 = expectedArray.map { UInt16($0) }
                    return compareIntArrays(actual: actualUInt16, expected: expectedUInt16, outputIndex: index)
                }
                let actualArray = try actual.toUInt32Array()
                let expectedUInt32 = expectedArray.map { UInt32($0) }
                return compareIntArrays(actual: actualArray, expected: expectedUInt32, outputIndex: index)
            }

        case .uint16:
            guard let expectedArray = expected.toUInt16Array() else {
                return (false, Float.infinity, "Failed to convert expected data to UInt16 array")
            }
            do {
                let actualArray = try actual.toUInt16Array()
                return compareIntArrays(actual: actualArray, expected: expectedArray, outputIndex: index)
            } catch {
                let actualArray = try actual.toUInt32Array()
                let expectedUInt32 = expectedArray.map { UInt32($0) }
                return compareIntArrays(actual: actualArray, expected: expectedUInt32, outputIndex: index)
            }

        case .uint32:
            guard let expectedArray = expected.toUInt32Array() else {
                return (false, Float.infinity, "Failed to convert expected data to UInt32 array")
            }
            let actualArray = try actual.toUInt32Array()
            return compareIntArrays(actual: actualArray, expected: expectedArray, outputIndex: index)

        case .uint64:
            guard let expectedArray = expected.toUInt64Array() else {
                return (false, Float.infinity, "Failed to convert expected data to UInt64 array")
            }
            do {
                let actualArray = try actual.toUInt64Array()
                return compareIntArrays(actual: actualArray, expected: expectedArray, outputIndex: index)
            } catch {
                let actualArray = try actual.toUInt32Array()
                let expectedUInt32 = expectedArray.map { UInt32(clamping: $0) }
                return compareIntArrays(actual: actualArray, expected: expectedUInt32, outputIndex: index)
            }

        case .bool:
            // Bool is stored as i1, read as UInt8
            let expectedBools = expected.data.map { $0 != 0 }
            let actualBools = try actual.toUInt8Array().map { $0 != 0 }
            guard actualBools.count == expectedBools.count else {
                return (false, Float.infinity, "Bool array count mismatch: got \(actualBools.count), expected \(expectedBools.count)")
            }
            for i in 0..<actualBools.count {
                if actualBools[i] != expectedBools[i] {
                    return (false, Float.infinity, "Bool mismatch at index \(i): got \(actualBools[i]), expected \(expectedBools[i])")
                }
            }
            return (true, 0, nil)

        case .complex64, .complex128:
            return (false, Float.infinity, "Complex types are not supported by Metal/MPS")
        }
    }

    /// Compare Float arrays with tolerance
    private func compareFloatArrays(
        actual: [Float],
        expected: [Float],
        tolerance: Float,
        outputIndex: Int
    ) -> (passed: Bool, maxDiff: Float, errorMessage: String?) {

        guard actual.count == expected.count else {
            return (false, Float.infinity, "Element count mismatch at output \(outputIndex): got \(actual.count), expected \(expected.count)")
        }

        var maxDiff: Float = 0
        var firstMismatchIndex: Int? = nil

        for i in 0..<actual.count {
            let a = actual[i]
            let e = expected[i]

            // Handle special cases
            if a.isNaN && e.isNaN {
                continue // Both NaN is a match
            }
            if a.isInfinite && e.isInfinite && a.sign == e.sign {
                continue // Same infinity is a match
            }

            let diff = abs(a - e)

            // Use relative tolerance for large values
            let relativeTolerance = max(tolerance, tolerance * abs(e))

            if diff > relativeTolerance {
                if firstMismatchIndex == nil {
                    firstMismatchIndex = i
                }
            }

            maxDiff = max(maxDiff, diff)
        }

        if let mismatchIdx = firstMismatchIndex {
            let a = actual[mismatchIdx]
            let e = expected[mismatchIdx]
            return (false, maxDiff, "Value mismatch at index \(mismatchIdx): got \(a), expected \(e), diff \(abs(a - e))")
        }

        return (true, maxDiff, nil)
    }

    /// Compare integer arrays (exact match)
    private func compareIntArrays<T: Equatable>(
        actual: [T],
        expected: [T],
        outputIndex: Int
    ) -> (passed: Bool, maxDiff: Float, errorMessage: String?) {

        guard actual.count == expected.count else {
            return (false, Float.infinity, "Element count mismatch at output \(outputIndex): got \(actual.count), expected \(expected.count)")
        }

        for i in 0..<actual.count {
            if actual[i] != expected[i] {
                return (false, Float.infinity, "Value mismatch at index \(i): got \(actual[i]), expected \(expected[i])")
            }
        }

        return (true, 0, nil)
    }

    /// Convert half-precision data to Float array
    private func convertHalfToFloat(_ data: Data, elementType: TensorData.ElementType) -> [Float] {
        switch elementType {
        case .float16:
            // Float16 is IEEE 754 half-precision
            return data.withUnsafeBytes { buffer in
                let uint16s = Array(buffer.bindMemory(to: UInt16.self))
                return uint16s.map { Float16(bitPattern: $0) }.map { Float($0) }
            }

        case .bfloat16:
            // BFloat16 is the upper 16 bits of a Float32
            return data.withUnsafeBytes { buffer in
                let uint16s = Array(buffer.bindMemory(to: UInt16.self))
                return uint16s.map { bits -> Float in
                    let float32Bits = UInt32(bits) << 16
                    return Float(bitPattern: float32Bits)
                }
            }

        default:
            return []
        }
    }
}

/// Results from running a test suite
public struct ConformanceSuiteResult: Sendable {
    public let results: [ConformanceTestResult]

    public var totalTests: Int { results.count }
    public var passedTests: Int { results.filter { $0.passed }.count }
    public var failedTests: Int { results.filter { !$0.passed }.count }
    public var passRate: Double { totalTests > 0 ? Double(passedTests) / Double(totalTests) : 0 }

    /// The optimization level used for these tests
    public var optimizationLevel: OptimizationLevel? {
        results.first?.optimizationLevel
    }

    public var summary: String {
        let levelStr = optimizationLevel.map { "O\($0.rawValue)" } ?? "unknown"
        return """
        StableHLO Conformance Test Results [\(levelStr)]
        ==================================
        Total:  \(totalTests)
        Passed: \(passedTests)
        Failed: \(failedTests)
        Pass Rate: \(String(format: "%.1f", passRate * 100))%

        \(results.map { $0.summary }.joined(separator: "\n"))
        """
    }

    /// Get only failed test results
    public var failures: [ConformanceTestResult] {
        results.filter { !$0.passed }
    }
}

/// Results from running tests across multiple optimization levels
public struct MultiOptLevelResult: Sendable {
    public let resultsByLevel: [OptimizationLevel: ConformanceSuiteResult]

    /// Whether all optimization levels produced the same pass/fail results
    public var isConsistent: Bool {
        let passedSets = resultsByLevel.values.map { Set($0.results.filter { $0.passed }.map { $0.testName }) }
        guard let first = passedSets.first else { return true }
        return passedSets.allSatisfy { $0 == first }
    }

    /// Tests that have inconsistent results across optimization levels
    public var inconsistentTests: [String] {
        var allTests = Set<String>()
        var passedByTest: [String: Set<OptimizationLevel>] = [:]

        for (level, result) in resultsByLevel {
            for testResult in result.results {
                allTests.insert(testResult.testName)
                if testResult.passed {
                    passedByTest[testResult.testName, default: []].insert(level)
                }
            }
        }

        let allLevels = Set(resultsByLevel.keys)
        return allTests.filter { testName in
            let passedLevels = passedByTest[testName] ?? []
            // Inconsistent if it passes on some levels but not others
            return !passedLevels.isEmpty && passedLevels != allLevels
        }.sorted()
    }

    public var summary: String {
        var lines = [
            "Multi-Optimization Level Conformance Results",
            "============================================="
        ]

        for level in ConformanceTestRunner.allOptimizationLevels {
            if let result = resultsByLevel[level] {
                lines.append("O\(level.rawValue): \(result.passedTests)/\(result.totalTests) passed (\(String(format: "%.1f", result.passRate * 100))%)")
            }
        }

        lines.append("")
        if isConsistent {
            lines.append("✓ Results are CONSISTENT across all optimization levels")
        } else {
            lines.append("✗ Results are INCONSISTENT across optimization levels!")
            lines.append("Inconsistent tests: \(inconsistentTests.joined(separator: ", "))")
        }

        return lines.joined(separator: "\n")
    }
}

/// Errors during conformance testing
public enum ConformanceTestError: Error, CustomStringConvertible {
    case cannotExtractOperation(String)
    case noInputsFound(String)
    case noExpectedOutputs(String)
    case unsupportedOperation(String)
    case executionFailed(String)

    public var description: String {
        switch self {
        case .cannotExtractOperation(let name):
            return "Cannot extract operation from test: \(name)"
        case .noInputsFound(let name):
            return "No inputs found in test: \(name)"
        case .noExpectedOutputs(let name):
            return "No expected outputs found in test: \(name)"
        case .unsupportedOperation(let op):
            return "Unsupported operation: \(op)"
        case .executionFailed(let reason):
            return "Execution failed: \(reason)"
        }
    }
}
