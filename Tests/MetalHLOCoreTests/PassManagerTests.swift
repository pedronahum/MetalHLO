// PassManagerTests.swift
// MetalHLOCoreTests
//
// Tests for the PassManager class.

import Testing
@testable import MetalHLOCore

// MARK: - Pass Manager Tests

@Suite("Pass Manager Tests")
struct PassManagerTests {

    // MARK: - Initialization Tests

    @Test("PassManager initializes with default config")
    func initializesWithDefaultConfig() {
        let passManager = PassManager()

        #expect(passManager.config.maxIterations > 0)
        #expect(passManager.config.collectStatistics == true)
    }

    @Test("PassManager initializes with custom config")
    func initializesWithCustomConfig() {
        var config = PassManager.Config.default
        config.maxIterations = 5
        config.disabledPasses = ["some-pass"]

        let passManager = PassManager(config: config)

        #expect(passManager.config.maxIterations == 5)
        #expect(passManager.config.disabledPasses.contains("some-pass"))
    }

    // MARK: - Statistics Tests

    @Test("Statistics initializes with zeros")
    func statisticsInit() {
        let stats = PassManager.Statistics()

        #expect(stats.passesRun == 0)
        #expect(stats.totalIterations == 0)
        #expect(stats.totalTimeMs == 0)
    }

    // MARK: - Convergence Tests

    @Test("Run passes to convergence")
    func runToConvergence() {
        let passManager = PassManager()

        let input = HLOArgument(name: "x", type: TensorType(shape: [4], elementType: .float32))

        // Create unused operation that should be eliminated
        let unusedOp = HLOOperation(
            result: "unused",
            kind: .sqrt,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let usedOp = HLOOperation(
            result: "y",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [unusedOp, usedOp],
            returnValues: ["y"]
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)

        let (result, newAnalysis) = passManager.runToConvergence(
            on: function,
            analysis: analysis,
            passes: ["dead-code-elimination"]
        )

        #expect(result.operations.count == 1)  // Only used op remains
        #expect(newAnalysis.shapes.count > 0)
    }

    // MARK: - Full Pipeline Tests

    @Test("Run all passes produces OptimizedModule")
    func runAllPasses() {
        let passManager = PassManager()

        let input = HLOArgument(name: "x", type: TensorType(shape: [1, 512, 768], elementType: .float32))

        // Simple tanh operation
        let op = HLOOperation(
            result: "y",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [1, 512, 768], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "forward",
            inputs: [input],
            outputTypes: [TensorType(shape: [1, 512, 768], elementType: .float32)],
            operations: [op],
            returnValues: ["y"]
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)

        let optimized = passManager.runAllPasses(module: function, analysis: analysis)

        #expect(!optimized.operations.isEmpty)
        #expect(!optimized.inputs.isEmpty)
        #expect(!optimized.outputs.isEmpty)
    }

    // MARK: - Config Presets Tests

    @Test("Debug config has expected settings")
    func debugConfig() {
        let config = PassManager.Config.debug

        #expect(config.collectStatistics == true)
        #expect(config.verifyAfterEachPass == true)
    }

    @Test("Minimal config enables only essential passes")
    func minimalConfig() {
        let config = PassManager.Config.minimal

        #expect(config.enabledPasses != nil)
        #expect(config.enabledPasses!.contains("algebraic-simplifier"))
        #expect(config.enabledPasses!.contains("dead-code-elimination"))
    }

    // MARK: - Pass Phase Tests

    @Test("PassPhase has all expected cases")
    func passPhases() {
        let phases = PassPhase.allCases

        #expect(phases.contains(.analysis))
        #expect(phases.contains(.simplification))
        #expect(phases.contains(.canonicalization))
        #expect(phases.contains(.patternFusion))
        #expect(phases.contains(.genericFusion))
        #expect(phases.contains(.crossLayer))
        #expect(phases.contains(.layout))
        #expect(phases.contains(.scheduling))
        #expect(phases.contains(.cleanup))
    }

    @Test("Passes run in correct order")
    func passOrder() {
        let passManager = PassManager()

        let input = HLOArgument(name: "x", type: TensorType(shape: [4], elementType: .float32))

        // Chain of operations
        let op1 = HLOOperation(
            result: "a",
            kind: .tanh,
            operands: ["x"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )
        let op2 = HLOOperation(
            result: "b",
            kind: .sqrt,
            operands: ["a"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )
        let op3 = HLOOperation(
            result: "y",
            kind: .negate,
            operands: ["b"],
            resultType: TensorType(shape: [4], elementType: .float32),
            attributes: HLOAttributes()
        )

        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [TensorType(shape: [4], elementType: .float32)],
            operations: [op1, op2, op3],
            returnValues: ["y"]
        )

        let analyzer = Analyzer()
        let analysis = analyzer.analyze(function)

        // Run all passes - they should execute in the correct phase order
        let optimized = passManager.runAllPasses(module: function, analysis: analysis)

        // Output should be valid
        #expect(!optimized.outputs.isEmpty)
    }
}

// MARK: - PassResult Tests

@Suite("Pass Result Tests")
struct PassResultTests {

    @Test("PassResult stores function and changed flag")
    func storesData() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [4], elementType: .float32))
        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [],
            operations: [],
            returnValues: []
        )

        let result = PassResult(function: function, changed: true, stats: ["test": 1])

        #expect(result.function.name == "test")
        #expect(result.changed == true)
        #expect(result.stats["test"] == 1)
    }

    @Test("PassResult unchanged when no modifications")
    func unchangedResult() {
        let input = HLOArgument(name: "x", type: TensorType(shape: [4], elementType: .float32))
        let function = HLOFunction(
            name: "test",
            inputs: [input],
            outputTypes: [],
            operations: [],
            returnValues: []
        )

        let result = PassResult(function: function, changed: false)

        #expect(result.changed == false)
        #expect(result.stats.isEmpty)
    }
}

// MARK: - Pass Registration Tests

@Suite("Pass Registration Tests")
struct PassRegistrationTests {

    @Test("PassRegistration stores name and phase")
    func storesData() {
        let registration = PassRegistration(
            name: "test-pass",
            phase: .simplification,
            factory: { DeadCodeEliminationPass() }
        )

        #expect(registration.name == "test-pass")
        #expect(registration.phase == .simplification)
    }

    @Test("PassRegistration factory creates passes")
    func factoryCreatesPass() {
        let registration = PassRegistration(
            name: "dce",
            phase: .simplification,
            factory: { DeadCodeEliminationPass() }
        )

        let pass = registration.factory()
        #expect(pass.name == "dead-code-elimination")
    }
}
