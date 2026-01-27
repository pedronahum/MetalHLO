// HLOPattern.swift
// MetalHLOCore
//
// Pattern matching infrastructure for HLO optimization.

import Foundation

/// Metadata value for pattern matches.
public enum PatternMetadataValue: Sendable, Equatable {
    case int(Int)
    case float(Float)
    case string(String)
    case bool(Bool)
    case intArray([Int])
}

/// A match result from pattern detection.
public struct PatternMatch: Sendable {
    /// The operations that were matched (in order).
    public let operations: [HLOOperation]

    /// Indices of matched operations in the original function.
    public let indices: [Int]

    /// The root operation (usually the last/output operation).
    public let rootOperation: HLOOperation

    /// Extracted metadata from the pattern (e.g., axis, epsilon).
    public let metadata: [String: PatternMetadataValue]

    /// The inputs to the fused operation (operand names from outside the pattern).
    public let inputs: [String]

    public init(
        operations: [HLOOperation],
        indices: [Int],
        rootOperation: HLOOperation,
        metadata: [String: PatternMetadataValue] = [:],
        inputs: [String] = []
    ) {
        self.operations = operations
        self.indices = indices
        self.rootOperation = rootOperation
        self.metadata = metadata
        self.inputs = inputs
    }
}

/// Protocol for HLO pattern matchers.
///
/// Implementations detect specific patterns in HLO IR and can
/// replace them with optimized operations.
public protocol HLOPattern: Sendable {
    /// Unique name for this pattern.
    var name: String { get }

    /// Priority for pattern matching (higher = checked first).
    var priority: Int { get }

    /// Attempts to match this pattern starting from a given operation.
    ///
    /// - Parameters:
    ///   - op: The operation to start matching from.
    ///   - index: Index of the operation in the function.
    ///   - function: The containing function.
    ///   - definingOps: Map from result name to defining operation.
    /// - Returns: A match if the pattern is found, nil otherwise.
    func match(
        at op: HLOOperation,
        index: Int,
        in function: HLOFunction,
        definingOps: [String: (op: HLOOperation, index: Int)]
    ) -> PatternMatch?

    /// Creates a replacement operation for the matched pattern.
    ///
    /// - Parameters:
    ///   - match: The pattern match.
    ///   - function: The containing function.
    /// - Returns: The replacement operation(s).
    func replacement(for match: PatternMatch, in function: HLOFunction) -> [HLOOperation]
}

extension HLOPattern {
    /// Default priority.
    public var priority: Int { 0 }
}

/// Helper for building defining operation maps.
public struct DefiningOpMap: Sendable {
    public let map: [String: (op: HLOOperation, index: Int)]

    public init(function: HLOFunction) {
        var m: [String: (op: HLOOperation, index: Int)] = [:]
        for (index, op) in function.operations.enumerated() {
            m[op.result] = (op, index)
        }
        self.map = m
    }

    public subscript(_ name: String) -> (op: HLOOperation, index: Int)? {
        map[name]
    }

    /// Gets the operation that defines a value, if it exists.
    public func definingOp(for name: String) -> HLOOperation? {
        map[name]?.op
    }

    /// Gets the index of the operation that defines a value.
    public func definingIndex(for name: String) -> Int? {
        map[name]?.index
    }
}

/// Registry of HLO patterns.
public final class HLOPatternRegistry: @unchecked Sendable {

    /// Shared registry instance.
    public static let shared = HLOPatternRegistry()

    private var patterns: [any HLOPattern] = []
    private let lock = NSLock()

    private init() {
        registerDefaultPatterns()
    }

    /// Creates a registry with custom patterns (for testing).
    public init(patterns: [any HLOPattern]) {
        self.patterns = patterns.sorted { $0.priority > $1.priority }
    }

    private func registerDefaultPatterns() {
        // Register built-in patterns in priority order (higher priority first)
        register(AttentionPattern())  // Priority 110 - check before softmax
        register(SoftmaxPattern())    // Priority 100
        register(GELUPattern())       // Priority 90
        register(LayerNormPattern())  // Priority 95
        register(RMSNormPattern())    // Priority 94
    }

    /// Registers a pattern.
    public func register(_ pattern: any HLOPattern) {
        lock.lock()
        defer { lock.unlock() }
        patterns.append(pattern)
        patterns.sort { $0.priority > $1.priority }
    }

    /// Returns all registered patterns.
    public var registeredPatterns: [any HLOPattern] {
        lock.lock()
        defer { lock.unlock() }
        return patterns
    }
}
