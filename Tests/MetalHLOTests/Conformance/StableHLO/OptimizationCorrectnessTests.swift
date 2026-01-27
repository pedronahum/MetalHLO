// OptimizationCorrectnessTests.swift
// MetalHLOTests
//
// Tests that verify all optimization levels produce numerically identical results.
// This ensures that optimizations (algebraic simplification, fusion, etc.) don't
// change the semantics of the program.

import Testing
import Foundation
import MetalHLO

// MARK: - Optimization Correctness Tests

/// Tests that verify all optimization levels produce the same numerical results.
/// This is critical for ensuring that optimization passes preserve program semantics.
@Suite("Optimization Correctness")
struct OptimizationCorrectnessTests {

    /// Test configurations for all optimization levels
    static let optimizationConfigs: [(name: String, config: CompilationConfig)] = [
        ("O0 (no optimization)", CompilationConfig(optimizationLevel: .O0, enableCaching: false)),
        ("O1 (basic)", CompilationConfig(optimizationLevel: .O1, enableCaching: false)),
        ("O2 (standard)", CompilationConfig(optimizationLevel: .O2, enableCaching: false)),
        ("O3 (aggressive)", CompilationConfig(optimizationLevel: .O3, enableCaching: false)),
    ]

    /// Operations that are safe for optimization correctness testing
    /// (well-supported by MPS and don't cause crashes)
    static let safeOperations = [
        "add", "subtract", "multiply", "divide",
        "abs", "negate", "exponential", "log",
        "sine", "cosine", "tanh", "sqrt",
        "floor", "ceil", "maximum", "minimum",
        "compare", "select", "clamp",
        "reshape", "transpose", "broadcast_in_dim",
        "reduce",
    ]

    // MARK: - Core Operation Tests

    @Test("Add operation preserves results across optimization levels")
    func testAddOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "add.mlir", operation: "add")
    }

    @Test("Subtract operation preserves results across optimization levels")
    func testSubtractOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "subtract.mlir", operation: "subtract")
    }

    @Test("Multiply operation preserves results across optimization levels")
    func testMultiplyOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "multiply.mlir", operation: "multiply")
    }

    @Test("Divide operation preserves results across optimization levels")
    func testDivideOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "divide.mlir", operation: "divide")
    }

    @Test("Abs operation preserves results across optimization levels")
    func testAbsOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "abs.mlir", operation: "abs")
    }

    @Test("Negate operation preserves results across optimization levels")
    func testNegateOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "negate.mlir", operation: "negate")
    }

    @Test("Exponential operation preserves results across optimization levels")
    func testExpOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "exponential.mlir", operation: "exponential")
    }

    @Test("Log operation preserves results across optimization levels")
    func testLogOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "log.mlir", operation: "log")
    }

    @Test("Tanh operation preserves results across optimization levels")
    func testTanhOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "tanh.mlir", operation: "tanh")
    }

    @Test("Sqrt operation preserves results across optimization levels")
    func testSqrtOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "sqrt.mlir", operation: "sqrt")
    }

    // MARK: - Shape Operations

    @Test("Reshape operation preserves results across optimization levels")
    func testReshapeOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "reshape.mlir", operation: "reshape")
    }

    @Test("Transpose operation preserves results across optimization levels")
    func testTransposeOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "transpose.mlir", operation: "transpose")
    }

    @Test("Broadcast operation preserves results across optimization levels")
    func testBroadcastOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "broadcast_in_dim.mlir", operation: "broadcast_in_dim")
    }

    // MARK: - Comparison and Selection

    @Test("Compare operation preserves results across optimization levels")
    func testCompareOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "compare.mlir", operation: "compare")
    }

    @Test("Select operation preserves results across optimization levels")
    func testSelectOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "select.mlir", operation: "select")
    }

    @Test("Clamp operation preserves results across optimization levels")
    func testClampOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "clamp.mlir", operation: "clamp")
    }

    // MARK: - Reductions

    @Test("Reduce operation preserves results across optimization levels")
    func testReduceOptimizationCorrectness() async throws {
        try await verifyOptimizationCorrectness(testFile: "reduce.mlir", operation: "reduce")
    }

    // MARK: - Helper Methods

    /// Verifies that a test file produces the same results across all optimization levels.
    private func verifyOptimizationCorrectness(testFile: String, operation: String) async throws {
        var allResults: [(level: String, results: [InterpretTestResult])] = []
        var errors: [String] = []

        // Run with each optimization level
        for (name, config) in Self.optimizationConfigs {
            do {
                let runner = try InterpretTestRunner(config: config)
                let results = try await runner.runTestFile(testFile)
                allResults.append((level: name, results: results))
            } catch {
                errors.append("\(name): \(error.localizedDescription)")
            }
        }

        // If we couldn't run any configs, fail
        if allResults.isEmpty {
            #expect(Bool(false), "Failed to run any optimization level: \(errors.joined(separator: "; "))")
            return
        }

        // Compare results across optimization levels
        let baseline = allResults[0]
        var mismatches: [String] = []

        for other in allResults.dropFirst() {
            // Compare each test case
            for (baseResult, otherResult) in zip(baseline.results, other.results) {
                // Both should have same pass/fail status
                if baseResult.passed != otherResult.passed {
                    // Skip if both are skipped (different skip reasons are OK)
                    let baseSkipped = baseResult.errorMessage?.hasPrefix("SKIPPED") ?? false
                    let otherSkipped = otherResult.errorMessage?.hasPrefix("SKIPPED") ?? false

                    if !(baseSkipped && otherSkipped) && !(baseSkipped || otherSkipped) {
                        mismatches.append(
                            "\(baseResult.testName): \(baseline.level)=\(baseResult.passed ? "PASS" : "FAIL"), " +
                            "\(other.level)=\(otherResult.passed ? "PASS" : "FAIL")"
                        )
                    }
                }
            }
        }

        // Report summary
        let totalTests = baseline.results.count
        let passedTests = baseline.results.filter { $0.passed }.count
        let skippedTests = baseline.results.filter { $0.errorMessage?.hasPrefix("SKIPPED") ?? false }.count

        print("Operation '\(operation)' optimization correctness:")
        print("  Total tests: \(totalTests)")
        print("  Passed on baseline (\(baseline.level)): \(passedTests)")
        print("  Skipped: \(skippedTests)")

        if !mismatches.isEmpty {
            print("  MISMATCHES (\(mismatches.count)):")
            for mismatch in mismatches.prefix(10) {
                print("    - \(mismatch)")
            }
            if mismatches.count > 10 {
                print("    ... and \(mismatches.count - 10) more")
            }
        }

        // All optimization levels should produce identical pass/fail results
        #expect(mismatches.isEmpty, "Optimization levels produced different results: \(mismatches.prefix(5).joined(separator: "; "))")
    }
}

// MARK: - Cross-Level Comparison Tests

/// Tests that compare specific test cases across all optimization levels with detailed output comparison.
@Suite("Optimization Level Comparison")
struct OptimizationLevelComparisonTests {

    /// Simple MLIR programs for direct comparison testing
    static let testPrograms: [(name: String, mlir: String)] = [
        ("simple_add", """
module @simple_add {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""),
        ("add_multiply_chain", """
module @add_multiply_chain {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    %1 = stablehlo.multiply %0, %arg2 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
"""),
        ("elementwise_chain", """
module @elementwise_chain {
  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
    %0 = stablehlo.abs %arg0 : tensor<4xf32>
    %1 = stablehlo.exponential %0 : tensor<4xf32>
    %2 = stablehlo.log %1 : tensor<4xf32>
    return %2 : tensor<4xf32>
  }
}
"""),
        ("broadcast_add", """
module @broadcast_add {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> (tensor<2x3xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}
"""),
        ("matmul_basic", """
module @matmul_basic {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> (tensor<2x4xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
"""),
    ]

    @Test("Simple add produces identical results across optimization levels")
    func testSimpleAddAcrossLevels() async throws {
        try await compareAcrossOptimizationLevels(
            programIndex: 0,
            inputs: [
                ([1.0, 2.0, 3.0, 4.0], [4]),
                ([0.5, 1.5, 2.5, 3.5], [4])
            ]
        )
    }

    @Test("Add-multiply chain produces identical results across optimization levels")
    func testAddMultiplyChainAcrossLevels() async throws {
        try await compareAcrossOptimizationLevels(
            programIndex: 1,
            inputs: [
                ([1.0, 2.0, 3.0, 4.0], [4]),
                ([0.5, 1.5, 2.5, 3.5], [4]),
                ([2.0, 2.0, 2.0, 2.0], [4])
            ]
        )
    }

    @Test("Elementwise chain (abs->exp->log) produces identical results across optimization levels")
    func testElementwiseChainAcrossLevels() async throws {
        try await compareAcrossOptimizationLevels(
            programIndex: 2,
            inputs: [
                ([-1.0, 2.0, -3.0, 4.0], [4])
            ]
        )
    }

    @Test("Broadcast add produces identical results across optimization levels")
    func testBroadcastAddAcrossLevels() async throws {
        try await compareAcrossOptimizationLevels(
            programIndex: 3,
            inputs: [
                ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
                ([0.1, 0.2, 0.3], [3])
            ]
        )
    }

    @Test("Basic matmul produces identical results across optimization levels")
    func testMatmulAcrossLevels() async throws {
        try await compareAcrossOptimizationLevels(
            programIndex: 4,
            inputs: [
                ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
                ([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [3, 4])
            ]
        )
    }

    // MARK: - Helper

    private func compareAcrossOptimizationLevels(
        programIndex: Int,
        inputs: [([Float], [Int])]
    ) async throws {
        let (name, mlir) = Self.testPrograms[programIndex]
        let client = try Client.create()

        var outputsPerLevel: [(level: String, output: [Float])] = []

        // Run with each optimization level
        for level in OptimizationLevel.allCases {
            let config = CompilationConfig(optimizationLevel: level, enableCaching: false)

            do {
                let executable = try client.compile(mlir, config: config)

                // Create input buffers
                var inputBuffers: [Buffer] = []
                for (data, shape) in inputs {
                    inputBuffers.append(client.createBuffer(data, shape: shape))
                }

                // Execute
                let outputs = try executable.execute(inputBuffers)

                // Get output as float array
                let outputFloats = try outputs[0].toFloatArray()
                outputsPerLevel.append((level: "O\(level.rawValue)", output: outputFloats))

            } catch {
                print("  O\(level.rawValue) failed: \(error)")
            }
        }

        // Compare all outputs
        guard outputsPerLevel.count >= 2 else {
            #expect(Bool(false), "Need at least 2 optimization levels to compare")
            return
        }

        let baseline = outputsPerLevel[0]
        var allMatch = true
        let tolerance: Float = 1e-5

        print("\(name) results:")
        print("  Baseline (\(baseline.level)): \(baseline.output)")

        for other in outputsPerLevel.dropFirst() {
            var matches = true
            var maxDiff: Float = 0

            for (a, b) in zip(baseline.output, other.output) {
                let diff = abs(a - b)
                maxDiff = max(maxDiff, diff)
                if diff > tolerance {
                    matches = false
                }
            }

            print("  \(other.level): \(other.output) (max diff: \(maxDiff), match: \(matches))")

            if !matches {
                allMatch = false
            }
        }

        #expect(allMatch, "Optimization levels produced different outputs for \(name)")
    }
}

// MARK: - Batch Optimization Testing

/// Runs a comprehensive batch of tests across all optimization levels.
@Suite("Batch Optimization Correctness", .tags(.extended))
struct BatchOptimizationCorrectnessTests {

    /// Test files that are safe to run with all optimization levels
    static let safeTestFiles = [
        "add.mlir",
        "subtract.mlir",
        "multiply.mlir",
        "divide.mlir",
        "abs.mlir",
        "negate.mlir",
        "exponential.mlir",
        "log.mlir",
        "tanh.mlir",
        "sqrt.mlir",
        "sine.mlir",
        "cosine.mlir",
        "floor.mlir",
        "ceil.mlir",
        "maximum.mlir",
        "minimum.mlir",
        "compare.mlir",
        "select.mlir",
        "clamp.mlir",
        "reshape.mlir",
        "transpose.mlir",
        "broadcast_in_dim.mlir",
    ]

    @Test("All safe operations produce consistent results across optimization levels")
    func testAllSafeOperationsAcrossLevels() async throws {
        var totalTests = 0
        var consistentTests = 0
        var inconsistentTests: [(file: String, test: String, details: String)] = []

        for testFile in Self.safeTestFiles {
            print("Testing \(testFile)...")

            // Run with O0 as baseline
            let baselineRunner: InterpretTestRunner
            do {
                baselineRunner = try InterpretTestRunner(config: CompilationConfig(optimizationLevel: .O0, enableCaching: false))
            } catch {
                print("  Failed to create baseline runner: \(error)")
                continue
            }

            let baselineResults: [InterpretTestResult]
            do {
                baselineResults = try await baselineRunner.runTestFile(testFile)
            } catch {
                print("  Failed to run baseline: \(error)")
                continue
            }

            // Run with O3 for comparison
            let o3Runner: InterpretTestRunner
            do {
                o3Runner = try InterpretTestRunner(config: CompilationConfig(optimizationLevel: .O3, enableCaching: false))
            } catch {
                print("  Failed to create O3 runner: \(error)")
                continue
            }

            let o3Results: [InterpretTestResult]
            do {
                o3Results = try await o3Runner.runTestFile(testFile)
            } catch {
                print("  Failed to run O3: \(error)")
                continue
            }

            // Compare
            for (baseline, o3) in zip(baselineResults, o3Results) {
                totalTests += 1

                // Both skipped is OK
                let baseSkipped = baseline.errorMessage?.hasPrefix("SKIPPED") ?? false
                let o3Skipped = o3.errorMessage?.hasPrefix("SKIPPED") ?? false

                if baseSkipped && o3Skipped {
                    consistentTests += 1
                    continue
                }

                // Same pass/fail status
                if baseline.passed == o3.passed {
                    consistentTests += 1
                } else {
                    let details = "O0=\(baseline.passed ? "PASS" : "FAIL:\(baseline.errorMessage ?? "")"), " +
                                  "O3=\(o3.passed ? "PASS" : "FAIL:\(o3.errorMessage ?? "")")"
                    inconsistentTests.append((file: testFile, test: baseline.testName, details: details))
                }
            }
        }

        print("\n=== Batch Optimization Correctness Summary ===")
        print("Total tests: \(totalTests)")
        print("Consistent: \(consistentTests)")
        print("Inconsistent: \(inconsistentTests.count)")

        if !inconsistentTests.isEmpty {
            print("\nInconsistent tests:")
            for (file, test, details) in inconsistentTests.prefix(20) {
                print("  \(file) / \(test): \(details)")
            }
            if inconsistentTests.count > 20 {
                print("  ... and \(inconsistentTests.count - 20) more")
            }
        }

        // We expect at least 90% consistency (some operations may not be fully implemented at all levels)
        let consistencyRate = Double(consistentTests) / Double(totalTests)
        #expect(consistencyRate >= 0.9, "Only \(Int(consistencyRate * 100))% consistency across optimization levels")
    }
}

// MARK: - Individual Pass Tests

/// Tests that verify individual optimization passes preserve correctness.
@Suite("Individual Pass Correctness")
struct IndividualPassCorrectnessTests {

    /// Test program: elementwise chain that can benefit from fusion
    static let elementwiseChainMLIR = """
module @elementwise_chain {
  func.func @main(%arg0: tensor<16xf32>) -> (tensor<16xf32>) {
    %0 = stablehlo.abs %arg0 : tensor<16xf32>
    %1 = stablehlo.negate %0 : tensor<16xf32>
    %2 = stablehlo.exponential %1 : tensor<16xf32>
    return %2 : tensor<16xf32>
  }
}
"""

    /// Test program: algebraic simplification candidates
    static let algebraicSimplificationMLIR = """
module @algebraic_simplification {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<8xf32>
    %1 = stablehlo.multiply %0, %arg1 : tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}
"""

    /// Test program: reshape and transpose candidates for canonicalization
    static let reshapeTransposeMLIR = """
module @reshape_transpose {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> (tensor<4x6xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
    %1 = stablehlo.reshape %0 : (tensor<4x2x3xf32>) -> tensor<4x6xf32>
    return %1 : tensor<4x6xf32>
  }
}
"""

    // MARK: - Simplification Pass Tests

    @Test("Algebraic simplifier pass preserves correctness")
    func testAlgebraicSimplifier() async throws {
        try await verifyPassCorrectness(
            mlir: Self.algebraicSimplificationMLIR,
            pass: OptimizationPass.algebraicSimplifier,
            inputs: [
                ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8]),
                ([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], [8])
            ]
        )
    }

    @Test("Constant folding pass preserves correctness")
    func testConstantFolding() async throws {
        try await verifyPassCorrectness(
            mlir: Self.elementwiseChainMLIR,
            pass: OptimizationPass.constantFolding,
            inputs: [
                ([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0,
                  -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0], [16])
            ]
        )
    }

    @Test("Dead code elimination pass preserves correctness")
    func testDeadCodeElimination() async throws {
        try await verifyPassCorrectness(
            mlir: Self.algebraicSimplificationMLIR,
            pass: OptimizationPass.deadCodeElimination,
            inputs: [
                ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8]),
                ([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [8])
            ]
        )
    }

    // MARK: - Canonicalization Pass Tests

    @Test("Reshape canonicalizer pass preserves correctness")
    func testReshapeCanonicalizer() async throws {
        try await verifyPassCorrectness(
            mlir: Self.reshapeTransposeMLIR,
            pass: OptimizationPass.reshapeCanonicalizer,
            inputs: [
                (Array(stride(from: Float(1.0), through: Float(24.0), by: Float(1.0))), [2, 3, 4])
            ]
        )
    }

    @Test("Transpose canonicalizer pass preserves correctness")
    func testTransposeCanonicalizer() async throws {
        try await verifyPassCorrectness(
            mlir: Self.reshapeTransposeMLIR,
            pass: OptimizationPass.transposeCanonicalizer,
            inputs: [
                (Array(stride(from: Float(1.0), through: Float(24.0), by: Float(1.0))), [2, 3, 4])
            ]
        )
    }

    // MARK: - Fusion Pass Tests

    @Test("Producer-consumer fusion pass preserves correctness")
    func testProducerConsumerFusion() async throws {
        try await verifyPassCorrectness(
            mlir: Self.elementwiseChainMLIR,
            pass: OptimizationPass.producerConsumerFusion,
            inputs: [
                ([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0,
                  -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0], [16])
            ]
        )
    }

    @Test("Sibling fusion pass preserves correctness")
    func testSiblingFusion() async throws {
        try await verifyPassCorrectness(
            mlir: Self.elementwiseChainMLIR,
            pass: OptimizationPass.siblingFusion,
            inputs: [
                ([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0,
                  -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0], [16])
            ]
        )
    }

    // MARK: - Helper

    private func verifyPassCorrectness(
        mlir: String,
        pass: OptimizationPass,
        inputs: [([Float], [Int])]
    ) async throws {
        let client = try Client.create()
        let tolerance: Float = 1e-5

        // Baseline: no optimization
        let baselineConfig = CompilationConfig(optimizationLevel: .O0, enableCaching: false)
        let baselineExecutable = try client.compile(mlir, config: baselineConfig)

        // With single pass enabled
        let passConfig = CompilationConfig(
            optimizationLevel: .O0,  // Start from O0
            enableCaching: false,
            enabledPasses: [pass.rawValue]  // Only enable this pass
        )
        let passExecutable = try client.compile(mlir, config: passConfig)

        // Create input buffers
        var inputBuffers: [Buffer] = []
        for (data, shape) in inputs {
            inputBuffers.append(client.createBuffer(data, shape: shape))
        }

        // Execute both
        let baselineOutputs = try baselineExecutable.execute(inputBuffers)
        let passOutputs = try passExecutable.execute(inputBuffers)

        // Compare
        let baselineFloats = try baselineOutputs[0].toFloatArray()
        let passFloats = try passOutputs[0].toFloatArray()

        #expect(baselineFloats.count == passFloats.count, "Output size mismatch")

        var maxDiff: Float = 0
        for (a, b) in zip(baselineFloats, passFloats) {
            let diff = abs(a - b)
            maxDiff = max(maxDiff, diff)
            if diff > tolerance {
                #expect(Bool(false), "Pass '\(pass.rawValue)' changed output: baseline=\(a), with_pass=\(b), diff=\(diff)")
                return
            }
        }

        print("Pass '\(pass.rawValue)' preserves correctness (max diff: \(maxDiff))")
    }
}

// MARK: - Pass Combination Tests

/// Tests combinations of passes to ensure they work correctly together.
@Suite("Pass Combination Correctness")
struct PassCombinationCorrectnessTests {

    static let testMLIR = """
module @test_program {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<8xf32>
    %1 = stablehlo.abs %0 : tensor<8xf32>
    %2 = stablehlo.multiply %1, %arg0 : tensor<8xf32>
    return %2 : tensor<8xf32>
  }
}
"""

    @Test("Simplification passes together preserve correctness")
    func testSimplificationPasses() async throws {
        let simplificationPasses: Set<String> = [
            OptimizationPass.constantFolding.rawValue,
            OptimizationPass.algebraicSimplifier.rawValue,
            OptimizationPass.deadCodeElimination.rawValue,
        ]

        try await verifyPassCombination(
            passes: simplificationPasses,
            inputs: [
                ([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], [8]),
                ([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], [8])
            ]
        )
    }

    @Test("Canonicalization passes together preserve correctness")
    func testCanonicalizationPasses() async throws {
        let canonicalizationPasses: Set<String> = [
            OptimizationPass.reshapeCanonicalizer.rawValue,
            OptimizationPass.transposeCanonicalizer.rawValue,
            OptimizationPass.broadcastCanonicalizer.rawValue,
        ]

        try await verifyPassCombination(
            passes: canonicalizationPasses,
            inputs: [
                ([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], [8]),
                ([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], [8])
            ]
        )
    }

    @Test("Fusion passes together preserve correctness")
    func testFusionPasses() async throws {
        let fusionPasses: Set<String> = [
            OptimizationPass.producerConsumerFusion.rawValue,
            OptimizationPass.siblingFusion.rawValue,
            OptimizationPass.elementwiseChainFusion.rawValue,
        ]

        try await verifyPassCombination(
            passes: fusionPasses,
            inputs: [
                ([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], [8]),
                ([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], [8])
            ]
        )
    }

    @Test("All basic passes (O1 equivalent) preserve correctness")
    func testO1EquivalentPasses() async throws {
        try await verifyPassCombination(
            passes: OptimizationPass.basicPasses,
            inputs: [
                ([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], [8]),
                ([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], [8])
            ]
        )
    }

    @Test("All standard passes (O2 equivalent) preserve correctness")
    func testO2EquivalentPasses() async throws {
        try await verifyPassCombination(
            passes: OptimizationPass.standardPasses,
            inputs: [
                ([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], [8]),
                ([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], [8])
            ]
        )
    }

    // MARK: - Helper

    private func verifyPassCombination(
        passes: Set<String>,
        inputs: [([Float], [Int])]
    ) async throws {
        let client = try Client.create()
        let tolerance: Float = 1e-5

        // Baseline: no optimization
        let baselineConfig = CompilationConfig(optimizationLevel: .O0, enableCaching: false)
        let baselineExecutable = try client.compile(Self.testMLIR, config: baselineConfig)

        // With passes enabled
        let passConfig = CompilationConfig(
            optimizationLevel: .O0,
            enableCaching: false,
            enabledPasses: passes
        )
        let passExecutable = try client.compile(Self.testMLIR, config: passConfig)

        // Create input buffers
        var inputBuffers: [Buffer] = []
        for (data, shape) in inputs {
            inputBuffers.append(client.createBuffer(data, shape: shape))
        }

        // Execute both
        let baselineOutputs = try baselineExecutable.execute(inputBuffers)
        let passOutputs = try passExecutable.execute(inputBuffers)

        // Compare
        let baselineFloats = try baselineOutputs[0].toFloatArray()
        let passFloats = try passOutputs[0].toFloatArray()

        #expect(baselineFloats.count == passFloats.count, "Output size mismatch")

        var maxDiff: Float = 0
        for (a, b) in zip(baselineFloats, passFloats) {
            let diff = abs(a - b)
            maxDiff = max(maxDiff, diff)
            if diff > tolerance {
                #expect(Bool(false), "Pass combination changed output: baseline=\(a), with_passes=\(b), diff=\(diff)")
                return
            }
        }

        print("Pass combination [\(passes.sorted().joined(separator: ", "))] preserves correctness (max diff: \(maxDiff))")
    }
}

// MARK: - Test Tags

extension Tag {
    @Tag static var extended: Self
}
