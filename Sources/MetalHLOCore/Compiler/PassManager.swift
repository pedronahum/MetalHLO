// PassManager.swift
// MetalHLOCore
//
// Unified pass manager that orchestrates all optimization passes in correct order.

import Foundation

// MARK: - Pass Phase

/// Phases of the optimization pipeline.
public enum PassPhase: String, Sendable, CaseIterable {
    case analysis
    case simplification
    case canonicalization
    case patternFusion
    case genericFusion
    case crossLayer
    case layout
    case scheduling
    case cleanup
}

// MARK: - Pass Registration

/// Registration entry for a pass.
public struct PassRegistration: Sendable {
    public let name: String
    public let phase: PassPhase
    public let factory: @Sendable () -> any OptimizationPass

    public init(name: String, phase: PassPhase, factory: @escaping @Sendable () -> any OptimizationPass) {
        self.name = name
        self.phase = phase
        self.factory = factory
    }
}

// MARK: - Pass Manager

/// Manages and runs optimization passes in the correct order.
public final class PassManager: @unchecked Sendable {

    // MARK: - Properties

    /// Registered passes in execution order.
    private var registeredPasses: [PassRegistration] = []

    /// Pass instances (created lazily).
    private var passInstances: [String: any OptimizationPass] = [:]

    /// Configuration.
    public var config: Config

    /// Statistics from the last run.
    public private(set) var statistics: Statistics = Statistics()

    // MARK: - Configuration

    public struct Config: Sendable {
        /// Which passes to enable (nil = all).
        public var enabledPasses: Set<String>?

        /// Which passes to disable.
        public var disabledPasses: Set<String>

        /// Maximum iterations for convergent passes.
        public var maxIterations: Int

        /// Whether to collect statistics.
        public var collectStatistics: Bool

        /// Whether to verify IR after each pass.
        public var verifyAfterEachPass: Bool

        public init(
            enabledPasses: Set<String>? = nil,
            disabledPasses: Set<String> = [],
            maxIterations: Int = 10,
            collectStatistics: Bool = true,
            verifyAfterEachPass: Bool = false
        ) {
            self.enabledPasses = enabledPasses
            self.disabledPasses = disabledPasses
            self.maxIterations = maxIterations
            self.collectStatistics = collectStatistics
            self.verifyAfterEachPass = verifyAfterEachPass
        }

        public static let `default` = Config()

        public static let debug = Config(
            collectStatistics: true,
            verifyAfterEachPass: true
        )

        public static let minimal = Config(
            enabledPasses: ["algebraic-simplifier", "dead-code-elimination"]
        )
    }

    // MARK: - Statistics

    public struct Statistics: Sendable {
        public var passesRun: Int = 0
        public var totalIterations: Int = 0
        public var passStats: [String: [String: Int]] = [:]
        public var timePerPass: [String: Double] = [:]
        public var totalTimeMs: Double = 0
    }

    // MARK: - Initialization

    public init(config: Config = .default) {
        self.config = config
        registerAllPasses()
    }

    // MARK: - Pass Registration

    /// Registers all built-in passes in the correct order.
    private func registerAllPasses() {
        // ═══════════════════════════════════════════════════════════════
        // PHASE 1: SIMPLIFICATION (reduce complexity)
        // ═══════════════════════════════════════════════════════════════
        register(name: "constant-folding", phase: .simplification) {
            ConstantFoldingPass()
        }
        register(name: "algebraic-simplifier", phase: .simplification) {
            AlgebraicSimplifierPass()
        }
        register(name: "dead-code-elimination", phase: .simplification) {
            DeadCodeEliminationPass()
        }
        register(name: "common-subexpr-elim", phase: .simplification) {
            CommonSubexpressionEliminationPass()
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 2: CANONICALIZATION (normalize representation)
        // ═══════════════════════════════════════════════════════════════
        register(name: "reshape-canonicalizer", phase: .canonicalization) {
            ReshapeCanonicalizer()
        }
        register(name: "transpose-canonicalizer", phase: .canonicalization) {
            TransposeCanonicalizer()
        }
        register(name: "broadcast-canonicalizer", phase: .canonicalization) {
            BroadcastCanonicalizer()
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 3: PATTERN FUSION (largest patterns first!)
        // ═══════════════════════════════════════════════════════════════
        register(name: "transformer-block-fusion", phase: .patternFusion) {
            TransformerBlockFusionPass()
        }
        register(name: "attention-fusion", phase: .patternFusion) {
            AttentionFusionPass()
        }
        register(name: "ffn-fusion", phase: .patternFusion) {
            FFNFusionPass()
        }
        register(name: "norm-fusion", phase: .patternFusion) {
            NormFusionPass()
        }
        register(name: "gelu-fusion", phase: .patternFusion) {
            GELUFusionPass()
        }
        register(name: "matmul-bias-act-fusion", phase: .patternFusion) {
            MatMulBiasActFusionPass()
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 4: GENERIC FUSION (catch remaining opportunities)
        // ═══════════════════════════════════════════════════════════════
        register(name: "producer-consumer-fusion", phase: .genericFusion) {
            ProducerConsumerFusionPass()
        }
        register(name: "sibling-fusion", phase: .genericFusion) {
            SiblingFusionPassWrapper()
        }
        register(name: "elementwise-chain-fusion", phase: .genericFusion) {
            ElementwiseChainFusionPass()
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 5: CROSS-LAYER OPTIMIZATION
        // ═══════════════════════════════════════════════════════════════
        register(name: "cross-layer-fusion", phase: .crossLayer) {
            CrossLayerFusionPassWrapper()
        }
        register(name: "residual-chain-fusion", phase: .crossLayer) {
            ResidualChainFusionPass()
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 6: LAYOUT OPTIMIZATION
        // ═══════════════════════════════════════════════════════════════
        register(name: "layout-assignment", phase: .layout) {
            LayoutAssignmentPassWrapper()
        }
        register(name: "transpose-folding", phase: .layout) {
            TransposeFoldingPass()
        }
        register(name: "copy-elimination", phase: .layout) {
            CopyEliminationPass()
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 7: SCHEDULING
        // ═══════════════════════════════════════════════════════════════
        register(name: "memory-aware-scheduler", phase: .scheduling) {
            MemoryAwareSchedulerPass()
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 8: FINAL CLEANUP
        // ═══════════════════════════════════════════════════════════════
        register(name: "final-dce", phase: .cleanup) {
            DeadCodeEliminationPass()
        }
    }

    /// Registers a pass.
    public func register(name: String, phase: PassPhase, factory: @escaping @Sendable () -> any OptimizationPass) {
        registeredPasses.append(PassRegistration(name: name, phase: phase, factory: factory))
    }

    // MARK: - Running Passes

    /// Runs all enabled passes on a function.
    public func runAllPasses(module: HLOFunction, analysis: AnalysisResults) -> OptimizedModule {
        let startTime = DispatchTime.now()
        statistics = Statistics()

        var currentFunction = module
        var currentAnalysis = analysis

        // Run passes in order
        for registration in registeredPasses {
            guard isPassEnabled(registration.name) else { continue }

            let passStartTime = DispatchTime.now()

            // Get or create pass instance
            let pass = passInstances[registration.name] ?? registration.factory()
            passInstances[registration.name] = pass

            // Run the pass
            let result = pass.run(on: currentFunction, analysis: currentAnalysis)

            // Update statistics
            statistics.passesRun += 1
            if config.collectStatistics {
                statistics.passStats[registration.name] = result.stats
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - passStartTime.uptimeNanoseconds) / 1_000_000
                statistics.timePerPass[registration.name] = elapsed
            }

            // Update function and analysis if changed
            if result.changed {
                currentFunction = result.function

                // Recompute invalidated analysis
                if !pass.invalidates.isEmpty {
                    currentAnalysis = recomputeAnalysis(
                        currentFunction,
                        previous: currentAnalysis,
                        invalidated: pass.invalidates
                    )
                }
            }
        }

        statistics.totalTimeMs = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000

        // Build optimized module
        return buildOptimizedModule(from: currentFunction, analysis: currentAnalysis)
    }

    /// Runs a specific phase of passes.
    public func runPhase(_ phase: PassPhase, on function: HLOFunction, analysis: AnalysisResults) -> (HLOFunction, AnalysisResults) {
        var currentFunction = function
        var currentAnalysis = analysis

        for registration in registeredPasses where registration.phase == phase {
            guard isPassEnabled(registration.name) else { continue }

            let pass = passInstances[registration.name] ?? registration.factory()
            passInstances[registration.name] = pass

            let result = pass.run(on: currentFunction, analysis: currentAnalysis)

            if result.changed {
                currentFunction = result.function
                if !pass.invalidates.isEmpty {
                    currentAnalysis = recomputeAnalysis(
                        currentFunction,
                        previous: currentAnalysis,
                        invalidated: pass.invalidates
                    )
                }
            }
        }

        return (currentFunction, currentAnalysis)
    }

    /// Runs passes to convergence (for simplification phase).
    public func runToConvergence(on function: HLOFunction, analysis: AnalysisResults, passes: [String]) -> (HLOFunction, AnalysisResults) {
        var currentFunction = function
        var currentAnalysis = analysis
        var iteration = 0

        while iteration < config.maxIterations {
            var anyChanged = false
            iteration += 1

            for passName in passes {
                guard isPassEnabled(passName) else { continue }
                guard let registration = registeredPasses.first(where: { $0.name == passName }) else { continue }

                let pass = passInstances[passName] ?? registration.factory()
                passInstances[passName] = pass

                let result = pass.run(on: currentFunction, analysis: currentAnalysis)

                if result.changed {
                    anyChanged = true
                    currentFunction = result.function
                    if !pass.invalidates.isEmpty {
                        currentAnalysis = recomputeAnalysis(
                            currentFunction,
                            previous: currentAnalysis,
                            invalidated: pass.invalidates
                        )
                    }
                }
            }

            if !anyChanged {
                break
            }
        }

        statistics.totalIterations += iteration
        return (currentFunction, currentAnalysis)
    }

    // MARK: - Helpers

    /// Checks if a pass is enabled.
    private func isPassEnabled(_ name: String) -> Bool {
        if config.disabledPasses.contains(name) {
            return false
        }
        if let enabled = config.enabledPasses {
            return enabled.contains(name)
        }
        return true
    }

    /// Recomputes invalidated analysis results.
    private func recomputeAnalysis(_ function: HLOFunction, previous: AnalysisResults, invalidated: Set<AnalysisType>) -> AnalysisResults {
        let analyzer = Analyzer()

        var shapes = previous.shapes
        var dependencies = previous.dependencies
        var users = previous.users
        var lifetimes = previous.lifetimes
        var patterns = previous.patterns
        var elementTypes = previous.elementTypes

        if invalidated.contains(.shapes) {
            shapes = analyzer.inferShapes(function)
            elementTypes = analyzer.inferElementTypes(function)
        }

        if invalidated.contains(.dependencies) {
            let result = analyzer.analyzeDependencies(function)
            dependencies = result.dependencies
            users = result.users
        }

        if invalidated.contains(.lifetimes) {
            lifetimes = analyzer.analyzeLifetimes(function, dependencies: dependencies, users: users)
        }

        if invalidated.contains(.patterns) {
            patterns = analyzer.detectPatterns(function, shapes: shapes)
        }

        return AnalysisResults(
            shapes: shapes,
            dependencies: dependencies,
            users: users,
            lifetimes: lifetimes,
            patterns: patterns,
            elementTypes: elementTypes
        )
    }

    /// Builds an OptimizedModule from the final function and analysis.
    private func buildOptimizedModule(from function: HLOFunction, analysis: AnalysisResults) -> OptimizedModule {
        let fusedOps = function.operations.map { FusedOp(from: $0) }
        let inputs = function.inputs.map { TensorParam(from: $0) }

        var tensors: [TensorID: TensorInfo] = [:]
        for input in function.inputs {
            tensors[input.name] = TensorInfo(id: input.name, type: input.type)
        }
        for op in function.operations {
            tensors[op.result] = TensorInfo(id: op.result, type: op.resultType)
        }

        // Extract constants from the function
        var constants: [TensorID: ConstantValue] = [:]
        for op in function.operations {
            if case .constant = op.kind,
               let constantValue = op.attributes.constantValue {
                constants[op.result] = constantValue
            }
        }

        return OptimizedModule(
            operations: fusedOps,
            inputs: inputs,
            outputs: function.returnValues,
            tensors: tensors,
            layouts: [:],
            schedule: function.operations.map { $0.result },
            analysis: analysis,
            constants: constants
        )
    }
}

// MARK: - Wrapper Passes

/// Wrapper for constant folding (uses AlgebraicSimplifier).
final class ConstantFoldingPass: OptimizationPass, @unchecked Sendable {
    let name = "constant-folding"
    let invalidates: Set<AnalysisType> = [.shapes, .lifetimes]

    private let simplifier = AlgebraicSimplifier(maxIterations: 1)

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let result = simplifier.simplify(function)
        let changed = result.operations.count != function.operations.count
        return PassResult(function: result, changed: changed)
    }
}

/// Wrapper for algebraic simplifier.
final class AlgebraicSimplifierPass: OptimizationPass, @unchecked Sendable {
    let name = "algebraic-simplifier"
    let invalidates: Set<AnalysisType> = [.shapes, .lifetimes]

    private let simplifier = AlgebraicSimplifier(maxIterations: 10)

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let result = simplifier.simplify(function)
        let changed = result.operations.count != function.operations.count
        return PassResult(function: result, changed: changed)
    }
}

/// Wrapper for producer-consumer fusion.
final class ProducerConsumerFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "producer-consumer-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    // NOTE: emitCustomCalls disabled because CodeGenerator doesn't handle fused_elementwise
    // custom calls - it falls back to a copy kernel which produces incorrect results.
    // The pass still reorders operations and MPSGraph handles actual kernel fusion.
    private let fusion = ProducerConsumerFusion(maxFusionSize: 50, emitCustomCalls: false)

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let result = fusion.fuse(function)
        let changed = result.operations.count != function.operations.count
        return PassResult(function: result, changed: changed)
    }
}

/// Wrapper for sibling fusion.
final class SiblingFusionPassWrapper: OptimizationPass, @unchecked Sendable {
    let name = "sibling-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    // NOTE: emitCustomCalls disabled for same reason as producer-consumer-fusion above.
    private let fusion = SiblingFusion(maxSiblings: 8, emitCustomCalls: false)

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let result = fusion.fuse(function)
        let changed = result.operations.count != function.operations.count
        return PassResult(function: result, changed: changed)
    }
}

/// Wrapper for cross-layer fusion.
final class CrossLayerFusionPassWrapper: OptimizationPass, @unchecked Sendable {
    let name = "cross-layer-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    private let fusion = CrossLayerFusionPass()

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let result = fusion.run(on: function)
        let changed = result.operations.count != function.operations.count
        return PassResult(function: result, changed: changed)
    }
}

/// Wrapper for layout assignment.
final class LayoutAssignmentPassWrapper: OptimizationPass, @unchecked Sendable {
    let name = "layout-assignment"
    let invalidates: Set<AnalysisType> = [.shapes]

    private let layoutAssignment = LayoutAssignment(insertTransposes: true)

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let result = layoutAssignment.optimize(function)
        let changed = result.operations.count != function.operations.count
        return PassResult(function: result, changed: changed)
    }
}

// MARK: - Pattern Fusion Passes

/// Transformer block fusion pass.
final class TransformerBlockFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "transformer-block-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        // Look for transformer block patterns from analysis
        let transformerPatterns = analysis.patterns.filter { $0.type == .transformerBlock }
        if transformerPatterns.isEmpty {
            return .unchanged(function)
        }
        // For now, pass through - actual fusion would create fused ops
        return .unchanged(function)
    }
}

/// Attention fusion pass.
final class AttentionFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "attention-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let attentionPatterns = analysis.patterns.filter { $0.type == .attention || $0.type == .multiHeadAttention }
        if attentionPatterns.isEmpty {
            return .unchanged(function)
        }
        return .unchanged(function)
    }
}

/// FFN fusion pass.
final class FFNFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "ffn-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let ffnPatterns = analysis.patterns.filter { $0.type == .ffn || $0.type == .gatedFFN }
        if ffnPatterns.isEmpty {
            return .unchanged(function)
        }
        return .unchanged(function)
    }
}

/// Norm fusion pass.
final class NormFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "norm-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let normPatterns = analysis.patterns.filter { $0.type == .layerNorm || $0.type == .rmsNorm }
        if normPatterns.isEmpty {
            return .unchanged(function)
        }
        return .unchanged(function)
    }
}

/// GELU fusion pass.
final class GELUFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "gelu-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let geluPatterns = analysis.patterns.filter { $0.type == .gelu }
        if geluPatterns.isEmpty {
            return .unchanged(function)
        }
        return .unchanged(function)
    }
}

/// MatMul + Bias + Activation fusion pass.
final class MatMulBiasActFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "matmul-bias-act-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let patterns = analysis.patterns.filter { $0.type == .matmulBiasActivation }
        if patterns.isEmpty {
            return .unchanged(function)
        }
        return .unchanged(function)
    }
}

/// Elementwise chain fusion pass.
final class ElementwiseChainFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "elementwise-chain-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        // Fuse chains of elementwise operations
        return .unchanged(function)
    }
}

/// Residual chain fusion pass.
final class ResidualChainFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "residual-chain-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let patterns = analysis.patterns.filter { $0.type == .residualAdd }
        if patterns.isEmpty {
            return .unchanged(function)
        }
        return .unchanged(function)
    }
}

/// Transpose folding pass.
final class TransposeFoldingPass: OptimizationPass, @unchecked Sendable {
    let name = "transpose-folding"
    let invalidates: Set<AnalysisType> = [.shapes]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        // Fold transposes into matmuls where possible
        return .unchanged(function)
    }
}

/// Memory-aware scheduler pass.
final class MemoryAwareSchedulerPass: OptimizationPass, @unchecked Sendable {
    let name = "memory-aware-scheduler"
    let invalidates: Set<AnalysisType> = []

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        // Reorder operations to minimize peak memory
        return .unchanged(function)
    }
}
