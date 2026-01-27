// OptimizationBugDiagnostic.swift
// MetalHLOTests
//
// Diagnostic test to identify which optimization pass causes the add-multiply chain bug.

import Testing
import Foundation
import MetalHLO

@Suite("Optimization Bug Diagnostic")
struct OptimizationBugDiagnosticTests {

    static let buggyMLIR = """
module @add_multiply_chain {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    %1 = stablehlo.multiply %0, %arg2 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
"""

    static let inputs: [([Float], [Int])] = [
        ([1.0, 2.0, 3.0, 4.0], [4]),
        ([0.5, 1.5, 2.5, 3.5], [4]),
        ([2.0, 2.0, 2.0, 2.0], [4])
    ]

    static let expectedOutput: [Float] = [3.0, 7.0, 11.0, 15.0]

    // List of all passes in the order they run at O2
    static let allPasses: [String] = [
        "constant-folding",
        "algebraic-simplifier",
        "dead-code-elimination",
        "common-subexpr-elim",
        "reshape-canonicalizer",
        "transpose-canonicalizer",
        "broadcast-canonicalizer",
        "transformer-block-fusion",
        "attention-fusion",
        "ffn-fusion",
        "norm-fusion",
        "gelu-fusion",
        "matmul-bias-act-fusion",
        "producer-consumer-fusion",
        "sibling-fusion",
        "elementwise-chain-fusion",
        "cross-layer-fusion",
        "residual-chain-fusion",
        "layout-assignment",
        "transpose-folding",
        "copy-elimination",
        "memory-aware-scheduler",
        "final-dce",
    ]

    @Test("Find the pass that breaks add-multiply chain")
    func findBrokenPass() async throws {
        let client = try Client.create()
        let tolerance: Float = 1e-5

        print("\n=== Optimization Bug Diagnostic ===")
        print("Testing add-multiply chain: (arg0 + arg1) * arg2")
        print("Inputs: arg0=[1,2,3,4], arg1=[0.5,1.5,2.5,3.5], arg2=[2,2,2,2]")
        print("Expected: [3, 7, 11, 15]")
        print("")

        // First verify baseline (O0)
        let baselineConfig = CompilationConfig(optimizationLevel: .O0, enableCaching: false)
        let baselineResult = try runWithConfig(client: client, config: baselineConfig)
        print("O0 (baseline): \(baselineResult)")

        // Test each pass individually
        print("\n--- Testing each pass individually ---")
        for passName in Self.allPasses {
            let config = CompilationConfig(
                optimizationLevel: .O0,
                enableCaching: false,
                enabledPasses: [passName]
            )
            let result = try runWithConfig(client: client, config: config)
            let isCorrect = checkCorrectness(result, expected: Self.expectedOutput, tolerance: tolerance)
            let status = isCorrect ? "✓" : "✗ BROKEN"
            print("  \(passName): \(result) \(status)")
        }

        // Test cumulative passes (adding one at a time)
        print("\n--- Testing cumulative passes ---")
        var enabledPasses: Set<String> = []
        for passName in Self.allPasses {
            enabledPasses.insert(passName)
            let config = CompilationConfig(
                optimizationLevel: .O0,
                enableCaching: false,
                enabledPasses: enabledPasses
            )
            let result = try runWithConfig(client: client, config: config)
            let isCorrect = checkCorrectness(result, expected: Self.expectedOutput, tolerance: tolerance)
            let status = isCorrect ? "✓" : "✗ BROKEN"
            print("  +\(passName): \(result) \(status)")

            if !isCorrect {
                print("\n  >>> BUG INTRODUCED BY: \(passName)")
                print("  >>> Passes enabled so far: \(enabledPasses.sorted())")
                break
            }
        }

        // Also test the full optimization levels for comparison
        print("\n--- Full optimization levels ---")
        for level in OptimizationLevel.allCases {
            let config = CompilationConfig(optimizationLevel: level, enableCaching: false)
            let result = try runWithConfig(client: client, config: config)
            let isCorrect = checkCorrectness(result, expected: Self.expectedOutput, tolerance: tolerance)
            let status = isCorrect ? "✓" : "✗ BROKEN"
            print("  O\(level.rawValue): \(result) \(status)")
        }
    }

    @Test("Detailed pass interaction test - disable passes from O2")
    func detailedPassInteraction() async throws {
        let client = try Client.create()
        let tolerance: Float = 1e-5

        print("\n=== Detailed Pass Interaction Test (Disabling from O2) ===")
        print("NOTE: enabledPasses only works when optimization level > O0")
        print("So we'll use O2 and DISABLE passes to find the culprit\n")

        // First confirm O2 is broken
        let o2Config = CompilationConfig(optimizationLevel: .O2, enableCaching: false)
        let o2Result = try runWithConfig(client: client, config: o2Config)
        print("O2 baseline (broken): \(o2Result)")

        // Test disabling each pass one at a time from O2
        print("\n--- O2 with each pass disabled ---")
        for passName in Self.allPasses {
            let config = CompilationConfig(
                optimizationLevel: .O2,
                enableCaching: false,
                disabledPasses: [passName]
            )
            let result = try runWithConfig(client: client, config: config)
            let isCorrect = checkCorrectness(result, expected: Self.expectedOutput, tolerance: tolerance)
            let status = isCorrect ? "✓ FIXED!" : "✗ still broken"
            print("  O2 - \(passName): \(result) \(status)")

            if isCorrect {
                print("\n  >>> DISABLING '\(passName)' FIXES THE BUG!")
            }
        }

        // Test disabling multiple passes
        print("\n--- Testing pass combinations ---")

        // Try disabling all fusion passes
        let fusionPasses: Set<String> = [
            "producer-consumer-fusion", "sibling-fusion", "elementwise-chain-fusion",
            "cross-layer-fusion", "residual-chain-fusion", "transformer-block-fusion",
            "attention-fusion", "ffn-fusion", "norm-fusion", "gelu-fusion", "matmul-bias-act-fusion"
        ]
        let noFusionConfig = CompilationConfig(
            optimizationLevel: .O2,
            enableCaching: false,
            disabledPasses: fusionPasses
        )
        let noFusionResult = try runWithConfig(client: client, config: noFusionConfig)
        let noFusionCorrect = checkCorrectness(noFusionResult, expected: Self.expectedOutput, tolerance: tolerance)
        print("O2 - all fusion passes: \(noFusionResult) \(noFusionCorrect ? "✓" : "✗")")

        // Try disabling layout passes
        let layoutPasses: Set<String> = ["layout-assignment", "transpose-folding", "copy-elimination"]
        let noLayoutConfig = CompilationConfig(
            optimizationLevel: .O2,
            enableCaching: false,
            disabledPasses: layoutPasses
        )
        let noLayoutResult = try runWithConfig(client: client, config: noLayoutConfig)
        let noLayoutCorrect = checkCorrectness(noLayoutResult, expected: Self.expectedOutput, tolerance: tolerance)
        print("O2 - layout passes: \(noLayoutResult) \(noLayoutCorrect ? "✓" : "✗")")

        // Try disabling canonicalizers
        let canonPasses: Set<String> = ["reshape-canonicalizer", "transpose-canonicalizer", "broadcast-canonicalizer"]
        let noCanonConfig = CompilationConfig(
            optimizationLevel: .O2,
            enableCaching: false,
            disabledPasses: canonPasses
        )
        let noCanonResult = try runWithConfig(client: client, config: noCanonConfig)
        let noCanonCorrect = checkCorrectness(noCanonResult, expected: Self.expectedOutput, tolerance: tolerance)
        print("O2 - canonicalizer passes: \(noCanonResult) \(noCanonCorrect ? "✓" : "✗")")
    }

    // MARK: - Helpers

    private func runWithConfig(client: Client, config: CompilationConfig) throws -> [Float] {
        let executable = try client.compile(Self.buggyMLIR, config: config)

        var inputBuffers: [Buffer] = []
        for (data, shape) in Self.inputs {
            inputBuffers.append(client.createBuffer(data, shape: shape))
        }

        let outputs = try executable.execute(inputBuffers)
        return try outputs[0].toFloatArray()
    }

    private func checkCorrectness(_ actual: [Float], expected: [Float], tolerance: Float) -> Bool {
        guard actual.count == expected.count else { return false }
        for (a, e) in zip(actual, expected) {
            if abs(a - e) > tolerance {
                return false
            }
        }
        return true
    }
}
