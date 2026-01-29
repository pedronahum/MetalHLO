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

    // emitCustomCalls enabled since FusedOp.init(from:) maps custom_call to FusedOpType
    // and CodeGenerator can generate proper kernels for all fused operation types.
    private let fusion = ProducerConsumerFusion(maxFusionSize: 50, emitCustomCalls: true)

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

    // emitCustomCalls enabled since FusedOp.init(from:) maps custom_call to FusedOpType.
    private let fusion = SiblingFusion(maxSiblings: 8, emitCustomCalls: true)

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
///
/// Detects Q @ K^T -> softmax -> @ V patterns and replaces them with
/// a fused_scaled_dot_product_attention custom_call.
final class AttentionFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "attention-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let attentionPatterns = analysis.patterns.filter { $0.type == .attention || $0.type == .multiHeadAttention }
        if attentionPatterns.isEmpty {
            return .unchanged(function)
        }

        var newOperations = function.operations
        var opsToRemove = Set<Int>()
        var changed = false

        for pattern in attentionPatterns {
            // Pattern has [qk_matmul_idx, softmax_idx, root_idx]
            guard pattern.operationIndices.count >= 3 else { continue }

            let qkMatmulIdx = pattern.operationIndices[0]
            let softmaxIdx = pattern.operationIndices[1]
            let rootIdx = pattern.rootIndex  // final matmul (attention @ V)

            guard qkMatmulIdx < function.operations.count,
                  softmaxIdx < function.operations.count,
                  rootIdx < function.operations.count else { continue }

            let qkMatmul = function.operations[qkMatmulIdx]
            let rootOp = function.operations[rootIdx]

            // Get Q and K from the first matmul
            guard qkMatmul.operands.count >= 2 else { continue }
            let q = qkMatmul.operands[0]
            let k = qkMatmul.operands[1]

            // Get V from the root matmul (second operand)
            guard rootOp.operands.count >= 2 else { continue }
            let v = rootOp.operands[1]

            // Compute scale from head dimension if available
            let scale: Float
            if let headDim = pattern.metadata.headDim {
                scale = 1.0 / Float(headDim).squareRoot()
            } else {
                scale = 1.0
            }

            // Build backend config JSON
            let configDict: [String: Any] = [
                "scale": scale,
                "is_causal": pattern.metadata.causalMask ?? false,
                "has_mask": false
            ]
            let backendConfig = (try? JSONSerialization.data(withJSONObject: configDict))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

            // Create custom_call attributes
            var attrs = HLOAttributes()
            attrs.callTargetName = FusedScaledDotProductAttentionHandler.targetName
            attrs.backendConfig = backendConfig

            // Create the fused operation
            let fusedOp = HLOOperation(
                result: rootOp.result,
                kind: .customCall,
                operands: [q, k, v],
                resultType: rootOp.resultType,
                attributes: attrs
            )

            // Replace the root operation with the fused one
            newOperations[rootIdx] = fusedOp

            // Mark intermediate operations for removal
            opsToRemove.insert(qkMatmulIdx)
            opsToRemove.insert(softmaxIdx)
            changed = true
        }

        if !changed {
            return .unchanged(function)
        }

        // Filter out removed operations
        let filteredOps = newOperations.enumerated()
            .filter { !opsToRemove.contains($0.offset) }
            .map { $0.element }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: filteredOps,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: true,
            stats: [
                "operationsRemoved": opsToRemove.count,
                "operationsFused": attentionPatterns.count
            ]
        )
    }
}

/// FFN fusion pass.
///
/// Detects Feed-Forward Network patterns (up_proj → activation → down_proj)
/// and replaces them with a fused_ffn custom_call.
final class FFNFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "ffn-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let ffnPatterns = analysis.patterns.filter { $0.type == .ffn || $0.type == .gatedFFN }
        if ffnPatterns.isEmpty {
            return .unchanged(function)
        }

        var newOperations = function.operations
        var opsToRemove = Set<Int>()
        var changed = false

        // Build op index map for looking up defining operations
        var opByResult: [TensorID: (op: HLOOperation, index: Int)] = [:]
        for (index, op) in function.operations.enumerated() {
            opByResult[op.result] = (op, index)
        }

        for pattern in ffnPatterns {
            // FFN pattern has: [upMatmulIdx, activationIdx, downMatmulIdx (root)]
            guard pattern.operationIndices.count >= 3 else { continue }

            let upMatmulIdx = pattern.operationIndices[0]
            let activationIdx = pattern.operationIndices[1]
            let rootIdx = pattern.rootIndex  // down projection matmul

            guard upMatmulIdx < function.operations.count,
                  activationIdx < function.operations.count,
                  rootIdx < function.operations.count else { continue }

            let upMatmul = function.operations[upMatmulIdx]
            let activationOp = function.operations[activationIdx]
            let downMatmul = function.operations[rootIdx]

            // Extract operands
            guard upMatmul.operands.count >= 2,
                  downMatmul.operands.count >= 2 else { continue }

            let input = upMatmul.operands[0]      // Input to FFN
            let upWeight = upMatmul.operands[1]   // Up projection weight
            let downWeight = downMatmul.operands[1]  // Down projection weight

            // Determine activation type from the activation operation
            let activationType: String
            switch activationOp.kind {
            case .tanh:
                activationType = "tanh"
            case .logistic:
                activationType = "sigmoid"
            case .customCall:
                if let targetName = activationOp.attributes.callTargetName {
                    if targetName.contains("gelu") {
                        activationType = "gelu"
                    } else if targetName.contains("silu") {
                        activationType = "silu"
                    } else if targetName.contains("relu") {
                        activationType = "relu"
                    } else {
                        activationType = "gelu"  // Default
                    }
                } else {
                    activationType = "gelu"
                }
            default:
                activationType = "gelu"  // Default activation
            }

            // Get hidden dimension from metadata if available
            let hiddenDim = pattern.metadata.hiddenDim ?? 0

            // Build backend config JSON
            let configDict: [String: Any] = [
                "activation": activationType,
                "hidden_dim": hiddenDim,
                "is_gated": pattern.type == .gatedFFN
            ]
            let backendConfig = (try? JSONSerialization.data(withJSONObject: configDict))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

            // Create custom_call attributes
            var attrs = HLOAttributes()
            attrs.callTargetName = "fused_ffn"
            attrs.backendConfig = backendConfig

            // Create the fused operation
            // Operands: [input, up_weight, down_weight]
            let fusedOp = HLOOperation(
                result: downMatmul.result,
                kind: .customCall,
                operands: [input, upWeight, downWeight],
                resultType: downMatmul.resultType,
                attributes: attrs
            )

            // Replace the root operation with the fused one
            newOperations[rootIdx] = fusedOp

            // Mark intermediate operations for removal
            opsToRemove.insert(upMatmulIdx)
            opsToRemove.insert(activationIdx)
            changed = true
        }

        if !changed {
            return .unchanged(function)
        }

        // Filter out removed operations
        let filteredOps = newOperations.enumerated()
            .filter { !opsToRemove.contains($0.offset) }
            .map { $0.element }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: filteredOps,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: true,
            stats: [
                "operationsRemoved": opsToRemove.count,
                "operationsFused": ffnPatterns.count
            ]
        )
    }
}

/// Norm fusion pass.
///
/// Detects layer normalization and RMS normalization patterns and replaces
/// them with fused custom_call operations.
final class NormFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "norm-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let normPatterns = analysis.patterns.filter { $0.type == .layerNorm || $0.type == .rmsNorm }
        if normPatterns.isEmpty {
            return .unchanged(function)
        }

        var newOperations = function.operations
        var opsToRemove = Set<Int>()
        var changed = false

        // Build op index map
        var opByResult: [TensorID: (op: HLOOperation, index: Int)] = [:]
        for (index, op) in function.operations.enumerated() {
            opByResult[op.result] = (op, index)
        }

        for pattern in normPatterns {
            guard !pattern.operationIndices.isEmpty else { continue }

            let rootIdx = pattern.rootIndex
            guard rootIdx < function.operations.count else { continue }

            let rootOp = function.operations[rootIdx]

            // Determine the target name based on pattern type
            let targetName: String
            if pattern.type == .layerNorm {
                targetName = FusedLayerNormHandler.targetName
            } else {
                targetName = FusedRMSNormHandler.targetName
            }

            // Find the input tensor (first operation's input in the pattern)
            let firstOpIdx = pattern.operationIndices.first!
            guard firstOpIdx < function.operations.count else { continue }
            let firstOp = function.operations[firstOpIdx]
            guard !firstOp.operands.isEmpty else { continue }
            let input = firstOp.operands[0]

            // Try to find scale/gamma and bias/beta from pattern operations
            // These are typically the last multiplication and addition
            var operands: [TensorID] = [input]

            // For layer norm we need gamma (scale) and beta (offset)
            // For RMS norm we just need weight (scale)
            // Look for constant tensors used in multiplications and additions
            for idx in pattern.operationIndices.reversed() {
                let op = function.operations[idx]
                if op.kind == .multiply && op.operands.count >= 2 {
                    // Second operand is likely gamma/weight
                    if op.operands[1] != input && !operands.contains(op.operands[1]) {
                        operands.append(op.operands[1])
                        break
                    }
                }
            }

            // For layer norm, also look for bias
            if pattern.type == .layerNorm {
                for idx in pattern.operationIndices.reversed() {
                    let op = function.operations[idx]
                    if op.kind == .add && op.operands.count >= 2 {
                        if op.operands[1] != input && !operands.contains(op.operands[1]) {
                            operands.append(op.operands[1])
                            break
                        }
                    }
                }
            }

            // Get epsilon from metadata or use default
            let eps = pattern.metadata.epsilon ?? 1e-5

            // Build backend config
            let configDict: [String: Any] = [
                "eps": eps,
                "axes": [-1]  // Default to last axis
            ]
            let backendConfig = (try? JSONSerialization.data(withJSONObject: configDict))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

            // Create custom_call attributes
            var attrs = HLOAttributes()
            attrs.callTargetName = targetName
            attrs.backendConfig = backendConfig

            // Create the fused operation
            let fusedOp = HLOOperation(
                result: rootOp.result,
                kind: .customCall,
                operands: operands,
                resultType: rootOp.resultType,
                attributes: attrs
            )

            // Replace the root operation
            newOperations[rootIdx] = fusedOp

            // Mark intermediate operations for removal
            for idx in pattern.operationIndices where idx != rootIdx {
                opsToRemove.insert(idx)
            }
            changed = true
        }

        if !changed {
            return .unchanged(function)
        }

        // Filter out removed operations
        let filteredOps = newOperations.enumerated()
            .filter { !opsToRemove.contains($0.offset) }
            .map { $0.element }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: filteredOps,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: true,
            stats: [
                "operationsRemoved": opsToRemove.count,
                "operationsFused": normPatterns.count
            ]
        )
    }
}

/// GELU fusion pass.
///
/// Detects GELU activation patterns (x * 0.5 * (1 + tanh(...))) and replaces
/// them with a fused_gelu custom_call.
final class GELUFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "gelu-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let geluPatterns = analysis.patterns.filter { $0.type == .gelu }
        if geluPatterns.isEmpty {
            return .unchanged(function)
        }

        var newOperations = function.operations
        var opsToRemove = Set<Int>()
        var changed = false

        for pattern in geluPatterns {
            guard !pattern.operationIndices.isEmpty else { continue }

            let rootIdx = pattern.rootIndex
            guard rootIdx < function.operations.count else { continue }

            let rootOp = function.operations[rootIdx]

            // Find the input to the GELU pattern (first operation's input)
            let firstOpIdx = pattern.operationIndices.first!
            guard firstOpIdx < function.operations.count else { continue }
            let firstOp = function.operations[firstOpIdx]
            guard !firstOp.operands.isEmpty else { continue }
            let input = firstOp.operands[0]

            // Determine if approximate GELU
            let isApproximate = pattern.metadata.activation?.contains("approximate") ?? true

            // Build backend config
            let configDict: [String: Any] = ["approximate": isApproximate]
            let backendConfig = (try? JSONSerialization.data(withJSONObject: configDict))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

            // Create custom_call attributes
            var attrs = HLOAttributes()
            attrs.callTargetName = FusedGeluHandler.targetName
            attrs.backendConfig = backendConfig

            // Create the fused operation
            let fusedOp = HLOOperation(
                result: rootOp.result,
                kind: .customCall,
                operands: [input],
                resultType: rootOp.resultType,
                attributes: attrs
            )

            // Replace the root operation with the fused one
            newOperations[rootIdx] = fusedOp

            // Mark all intermediate operations for removal (except root which is replaced)
            for idx in pattern.operationIndices where idx != rootIdx {
                opsToRemove.insert(idx)
            }
            changed = true
        }

        if !changed {
            return .unchanged(function)
        }

        // Filter out removed operations
        let filteredOps = newOperations.enumerated()
            .filter { !opsToRemove.contains($0.offset) }
            .map { $0.element }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: filteredOps,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: true,
            stats: [
                "operationsRemoved": opsToRemove.count,
                "operationsFused": geluPatterns.count
            ]
        )
    }
}

/// MatMul + Bias + Activation fusion pass.
///
/// Detects patterns of matmul followed by bias add and optional activation,
/// and replaces them with a single fused custom_call.
final class MatMulBiasActFusionPass: OptimizationPass, @unchecked Sendable {
    let name = "matmul-bias-act-fusion"
    let invalidates: Set<AnalysisType> = [.lifetimes, .patterns]

    func run(on function: HLOFunction, analysis: AnalysisResults) -> PassResult {
        let patterns = analysis.patterns.filter { $0.type == .matmulBiasActivation }
        if patterns.isEmpty {
            return .unchanged(function)
        }

        var newOperations = function.operations
        var opsToRemove = Set<Int>()
        var changed = false

        // Build op index map
        var opByResult: [TensorID: (op: HLOOperation, index: Int)] = [:]
        for (index, op) in function.operations.enumerated() {
            opByResult[op.result] = (op, index)
        }

        for pattern in patterns {
            guard !pattern.operationIndices.isEmpty else { continue }

            let rootIdx = pattern.rootIndex
            guard rootIdx < function.operations.count else { continue }

            let rootOp = function.operations[rootIdx]

            // Find the matmul operation (first in pattern typically)
            var matmulOp: HLOOperation?
            var matmulIdx: Int?
            for idx in pattern.operationIndices {
                let op = function.operations[idx]
                if op.kind == .dot || op.kind == .dotGeneral {
                    matmulOp = op
                    matmulIdx = idx
                    break
                }
            }

            guard let matmul = matmulOp, matmul.operands.count >= 2 else { continue }

            let x = matmul.operands[0]  // input
            let w = matmul.operands[1]  // weight

            // Find bias from add operation
            var bias: TensorID?
            for idx in pattern.operationIndices {
                let op = function.operations[idx]
                if op.kind == .add && op.operands.count >= 2 {
                    // The non-matmul operand is likely the bias
                    if let defOp = opByResult[op.operands[0]], defOp.op.kind == .dot || defOp.op.kind == .dotGeneral {
                        bias = op.operands[1]
                    } else if let defOp = opByResult[op.operands[1]], defOp.op.kind == .dot || defOp.op.kind == .dotGeneral {
                        bias = op.operands[0]
                    }
                    break
                }
            }

            guard let biasId = bias else { continue }

            // Get activation type from metadata
            let activation = pattern.metadata.activation ?? "none"

            // Build backend config
            let configDict: [String: Any] = [
                "activation": activation,
                "trans_a": false,
                "trans_b": false
            ]
            let backendConfig = (try? JSONSerialization.data(withJSONObject: configDict))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

            // Create custom_call attributes
            var attrs = HLOAttributes()
            attrs.callTargetName = FusedMatMulBiasActivationHandler.targetName
            attrs.backendConfig = backendConfig

            // Create the fused operation
            let fusedOp = HLOOperation(
                result: rootOp.result,
                kind: .customCall,
                operands: [x, w, biasId],
                resultType: rootOp.resultType,
                attributes: attrs
            )

            // Replace the root operation
            newOperations[rootIdx] = fusedOp

            // Mark intermediate operations for removal
            for idx in pattern.operationIndices where idx != rootIdx {
                opsToRemove.insert(idx)
            }
            changed = true
        }

        if !changed {
            return .unchanged(function)
        }

        // Filter out removed operations
        let filteredOps = newOperations.enumerated()
            .filter { !opsToRemove.contains($0.offset) }
            .map { $0.element }

        let newFunction = HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: filteredOps,
            returnValues: function.returnValues
        )

        return PassResult(
            function: newFunction,
            changed: true,
            stats: [
                "operationsRemoved": opsToRemove.count,
                "operationsFused": patterns.count
            ]
        )
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
