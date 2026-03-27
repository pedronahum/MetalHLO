// EligibilityAnnotationPass.swift
// HeterogeneousFusion
//
// Phase 4, Pass 1: Annotates each node in the graph with a PartitionDecision.
// No mutations to graph structure — only writes annotations.
//
// The ProfitabilityGuard evaluates each node's shape and op type to determine
// whether heterogeneous partitioning would be profitable. Nodes that fail the
// guard get `.fallback` with a recommended single unit.

import Foundation

/// Pass 1: Annotate each node with a partition decision.
///
/// This pass is pure annotation — it does not modify graph structure.
/// Downstream passes read `node.partitionDecision` to decide fusion and scheduling.
public struct EligibilityAnnotationPass: Sendable {

    private let guard_: ProfitabilityGuard

    public init(guard_: ProfitabilityGuard) {
        self.guard_ = guard_
    }

    /// Run the pass over the graph, annotating every node.
    ///
    /// After this pass, every node has a non-nil `partitionDecision`:
    /// - `.partition(descriptor)` for ops that should be split across units
    /// - `.fallback(reason, unit)` for ops that should run on a single unit
    public func run(_ graph: inout HeteroGraph) {
        for node in graph.nodes {
            let decision = guard_.evaluate(op: node.opcode, shape: node.outputShape)
            node.partitionDecision = decision
        }
    }

    /// Summary of pass results for diagnostics.
    public func summary(_ graph: HeteroGraph) -> (eligible: Int, fallback: Int) {
        var eligible = 0, fallback = 0
        for node in graph.nodes {
            if node.isEligible { eligible += 1 } else { fallback += 1 }
        }
        return (eligible, fallback)
    }
}
