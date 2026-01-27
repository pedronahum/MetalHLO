// CompilationConfig.swift
// MetalHLO
//
// Public configuration for StableHLO compilation.

import Foundation
import MetalHLOCore

/// Optimization level for compilation.
///
/// Higher optimization levels apply more aggressive transformations
/// that can improve runtime performance at the cost of longer compilation times.
public enum OptimizationLevel: Int, Sendable, CaseIterable {
    /// No optimization - fastest compilation, useful for debugging.
    case O0 = 0

    /// Basic optimization - algebraic simplification and dead code elimination.
    case O1 = 1

    /// Standard optimization (default) - includes canonicalization and fusion passes.
    case O2 = 2

    /// Aggressive optimization - multiple fusion iterations for maximum performance.
    case O3 = 3
}

/// Configuration options for StableHLO compilation.
///
/// Use this to control optimization levels, caching behavior, and debug output.
///
/// ## Example
/// ```swift
/// // Use aggressive optimization
/// let config = CompilationConfig(optimizationLevel: .O3)
/// let executable = try client.compile(mlir, config: config)
///
/// // Debug build with no optimization
/// let debugConfig = CompilationConfig.debug
/// let debugExe = try client.compile(mlir, config: debugConfig)
/// ```
public struct CompilationConfig: Sendable {

    // MARK: - Properties

    /// The optimization level to use during compilation.
    public var optimizationLevel: OptimizationLevel

    /// Whether to cache compiled executables for reuse.
    ///
    /// When enabled, compiling the same MLIR text will return a cached result.
    public var enableCaching: Bool

    /// Whether to generate debug information in the compiled output.
    ///
    /// Debug info helps with profiling and debugging but increases compilation time.
    public var generateDebugInfo: Bool

    /// Specific passes to enable (nil = use optimization level defaults).
    ///
    /// When specified, only these passes will run regardless of optimization level.
    /// This is useful for testing specific passes in isolation.
    public var enabledPasses: Set<String>?

    /// Passes to disable even if they would normally run at the selected optimization level.
    public var disabledPasses: Set<String>

    // MARK: - Initialization

    /// Creates a compilation configuration with the specified options.
    ///
    /// - Parameters:
    ///   - optimizationLevel: The optimization level (default: `.O2`).
    ///   - enableCaching: Whether to cache compiled results (default: `true`).
    ///   - generateDebugInfo: Whether to generate debug info (default: `false`).
    ///   - enabledPasses: Specific passes to enable (nil = use level defaults).
    ///   - disabledPasses: Passes to disable (default: empty).
    public init(
        optimizationLevel: OptimizationLevel = .O2,
        enableCaching: Bool = true,
        generateDebugInfo: Bool = false,
        enabledPasses: Set<String>? = nil,
        disabledPasses: Set<String> = []
    ) {
        self.optimizationLevel = optimizationLevel
        self.enableCaching = enableCaching
        self.generateDebugInfo = generateDebugInfo
        self.enabledPasses = enabledPasses
        self.disabledPasses = disabledPasses
    }

    // MARK: - Presets

    /// Default configuration with standard optimization (O2).
    public static let `default` = CompilationConfig()

    /// Debug configuration with no optimization.
    public static let debug = CompilationConfig(
        optimizationLevel: .O0,
        enableCaching: false,
        generateDebugInfo: true
    )

    /// Release configuration with aggressive optimization (O3).
    public static let release = CompilationConfig(
        optimizationLevel: .O3,
        enableCaching: true,
        generateDebugInfo: false
    )

    /// Fast compilation with minimal optimization (O1).
    public static let fast = CompilationConfig(
        optimizationLevel: .O1,
        enableCaching: true,
        generateDebugInfo: false
    )
}

/// Statistics from compilation.
public struct CompilationStatistics: Sendable {
    /// Number of optimization passes run.
    public let passesRun: Int

    /// Total optimization iterations.
    public let totalIterations: Int

    /// Compilation time in milliseconds.
    public let compilationTimeMs: Double

    /// Number of cache hits during compilation.
    public let cacheHits: Int

    /// Number of cache misses during compilation.
    public let cacheMisses: Int

    /// Cache hit rate (0.0 to 1.0).
    public var cacheHitRate: Double {
        let total = cacheHits + cacheMisses
        return total > 0 ? Double(cacheHits) / Double(total) : 0
    }

    public init(
        passesRun: Int = 0,
        totalIterations: Int = 0,
        compilationTimeMs: Double = 0,
        cacheHits: Int = 0,
        cacheMisses: Int = 0
    ) {
        self.passesRun = passesRun
        self.totalIterations = totalIterations
        self.compilationTimeMs = compilationTimeMs
        self.cacheHits = cacheHits
        self.cacheMisses = cacheMisses
    }
}

// MARK: - Available Optimization Passes

/// List of all available optimization passes.
public enum OptimizationPass: String, CaseIterable, Sendable {
    // Simplification phase
    case constantFolding = "constant-folding"
    case algebraicSimplifier = "algebraic-simplifier"
    case deadCodeElimination = "dead-code-elimination"
    case commonSubexprElim = "common-subexpr-elim"

    // Canonicalization phase
    case reshapeCanonicalizer = "reshape-canonicalizer"
    case transposeCanonicalizer = "transpose-canonicalizer"
    case broadcastCanonicalizer = "broadcast-canonicalizer"

    // Pattern fusion phase
    case transformerBlockFusion = "transformer-block-fusion"
    case attentionFusion = "attention-fusion"
    case ffnFusion = "ffn-fusion"
    case normFusion = "norm-fusion"
    case geluFusion = "gelu-fusion"
    case matmulBiasActFusion = "matmul-bias-act-fusion"

    // Generic fusion phase
    case producerConsumerFusion = "producer-consumer-fusion"
    case siblingFusion = "sibling-fusion"
    case elementwiseChainFusion = "elementwise-chain-fusion"

    // Cross-layer phase
    case crossLayerFusion = "cross-layer-fusion"
    case residualChainFusion = "residual-chain-fusion"

    // Layout phase
    case layoutAssignment = "layout-assignment"
    case transposeFolding = "transpose-folding"
    case copyElimination = "copy-elimination"

    // Scheduling phase
    case memoryAwareScheduler = "memory-aware-scheduler"

    // Cleanup phase
    case finalDCE = "final-dce"

    /// All pass names as strings.
    public static var allPassNames: [String] {
        allCases.map { $0.rawValue }
    }

    /// Passes that run at O1 (basic optimization).
    public static var basicPasses: Set<String> {
        Set([
            constantFolding.rawValue,
            algebraicSimplifier.rawValue,
            deadCodeElimination.rawValue,
        ])
    }

    /// Passes that run at O2 (standard optimization).
    public static var standardPasses: Set<String> {
        var passes = basicPasses
        passes.formUnion([
            commonSubexprElim.rawValue,
            reshapeCanonicalizer.rawValue,
            transposeCanonicalizer.rawValue,
            broadcastCanonicalizer.rawValue,
            producerConsumerFusion.rawValue,
            finalDCE.rawValue,
        ])
        return passes
    }

    /// Passes that run at O3 (aggressive optimization) - all passes.
    public static var aggressivePasses: Set<String> {
        Set(allPassNames)
    }
}

// MARK: - Internal Conversions

extension OptimizationLevel {
    /// Converts to the internal compiler optimization level.
    func toCompilerLevel() -> MetalHLOCompiler.Config.OptimizationLevel {
        switch self {
        case .O0: return .O0
        case .O1: return .O1
        case .O2: return .O2
        case .O3: return .O3
        }
    }
}
