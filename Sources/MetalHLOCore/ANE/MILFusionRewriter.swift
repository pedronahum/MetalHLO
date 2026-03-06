// MILFusionRewriter.swift
// MetalHLOCore
//
// Rewrites HLO operation sequences by replacing detected patterns
// with fused operations. The rewritten sequence is consumed by
// CoreMLOpBuilder to emit efficient CoreML MIL programs that
// leverage ANE hardware fusion (e.g., layer_norm, gelu, silu).

import Foundation
import ANERuntime

/// Either an original HLO operation or a fused pattern to emit.
public enum FusedOrOriginalOp {
    /// Pass-through: emit normally via `translateOp`.
    case original(HLOOperation)

    /// Fused: the root operation should be emitted as a fused CoreML op.
    /// Contains the pattern info and all consumed operations.
    case fused(DetectedPattern, rootOp: HLOOperation, consumedOps: [HLOOperation])
}

/// Rewrites an HLO function's operation list by merging detected patterns
/// into fused operations. Non-root ops that belong to a pattern are skipped;
/// the root op is replaced with a `.fused` entry.
public struct MILFusionRewriter {

    /// Supported pattern types for CoreML MIL fusion.
    private static let supportedPatterns: Set<PatternType> = [
        .layerNorm, .rmsNorm, .gelu, .silu, .matmulBiasActivation, .ffn
    ]

    /// Rewrites the operation sequence, folding detected patterns.
    ///
    /// - Parameters:
    ///   - function: The HLO function to rewrite.
    ///   - patterns: Detected patterns from `Analyzer.detectPatterns()`.
    /// - Returns: A sequence of fused-or-original ops in topological order.
    public static func rewrite(
        function: HLOFunction,
        patterns: [DetectedPattern]
    ) -> [FusedOrOriginalOp] {
        // Filter to supported patterns only
        let fusible = patterns.filter { supportedPatterns.contains($0.type) }

        guard !fusible.isEmpty else {
            return function.operations.map { .original($0) }
        }

        // Build index → pattern mapping for root and consumed indices
        var rootPatternMap: [Int: DetectedPattern] = [:]
        var consumedIndices: Set<Int> = []

        for pattern in fusible {
            // Only fuse if we haven't already claimed these indices
            let patternIndices = Set(pattern.operationIndices)
            if patternIndices.isDisjoint(with: consumedIndices) ||
               patternIndices.subtracting(consumedIndices) == [pattern.rootIndex] {
                rootPatternMap[pattern.rootIndex] = pattern
                // Mark non-root indices as consumed (they'll be skipped)
                for idx in pattern.operationIndices where idx != pattern.rootIndex {
                    consumedIndices.insert(idx)
                }
            }
        }

        // Build operation lookup
        let ops = function.operations

        var result: [FusedOrOriginalOp] = []
        for (index, op) in ops.enumerated() {
            if consumedIndices.contains(index) {
                // This op is absorbed into a fused pattern — skip it
                continue
            }

            if let pattern = rootPatternMap[index] {
                // Gather all consumed ops for context
                let consumed = pattern.operationIndices
                    .filter { $0 != pattern.rootIndex && $0 < ops.count }
                    .map { ops[$0] }
                result.append(.fused(pattern, rootOp: op, consumedOps: consumed))
            } else {
                result.append(.original(op))
            }
        }

        return result
    }
}
