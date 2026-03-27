// OpFusionPass.swift
// HeterogeneousFusion
//
// Phase 4, Pass 2: Merges adjacent eligible ops into FusedOpClusters.
//
// Fusion rule: only fuse linear chains. A branch (output feeds two consumers)
// breaks the cluster. This keeps synchronization simple — each cluster has
// one entry and one exit point.
//
// The canonical case: attention pipeline (QK^T → softmax → AV) becomes one
// cluster with one solve, one set of command buffers, one sync point.

import Foundation

/// Pass 2: Group adjacent eligible nodes into fused clusters.
///
/// Walks the graph in topological order, greedily extending clusters while:
/// 1. Next node is eligible (`.partition` decision from Pass 1)
/// 2. Next node has exactly one input from the current cluster (no branching)
/// 3. Current node has exactly one output (no fan-out)
///
/// Nodes not assigned to any cluster become single-op clusters or passthroughs.
public struct OpFusionPass: Sendable {

    public init() {}

    /// Run the pass, populating `graph.clusters` and setting `node.clusterID`.
    public func run(_ graph: inout HeteroGraph) {
        let sorted = graph.topologicalSort()
        var visited = Set<Int>()
        var clusters: [FusedOpCluster] = []
        var nextClusterID = 0

        for node in sorted {
            guard !visited.contains(node.id) else { continue }

            if node.isEligible {
                // Start a new cluster and greedily extend it
                var chain: [HeteroNode] = [node]
                visited.insert(node.id)

                var current = node
                while let next = singleFusibleSuccessor(of: current, visited: visited) {
                    chain.append(next)
                    visited.insert(next.id)
                    current = next
                }

                let cluster = buildCluster(id: nextClusterID, chain: chain)
                for n in chain {
                    n.clusterID = nextClusterID
                }
                clusters.append(cluster)
                nextClusterID += 1
            } else {
                // Passthrough node — not part of any cluster
                visited.insert(node.id)
            }
        }

        graph.clusters = clusters
    }

    // MARK: - Internal

    /// Find the single fusible successor of a node.
    ///
    /// A successor is fusible if:
    /// 1. The current node has exactly one consumer (no fan-out)
    /// 2. The next node has exactly one producer (no fan-in)
    /// 3. The next node is either:
    ///    a. Eligible for partitioning (normal chain extension), OR
    ///    b. An absorbable non-matmul op between two eligible matmuls
    ///       (attention pattern: matmul → softmax → matmul)
    private func singleFusibleSuccessor(
        of node: HeteroNode,
        visited: Set<Int>
    ) -> HeteroNode? {
        // Node must have exactly one consumer
        guard node.outputs.count == 1 else { return nil }
        let next = node.outputs[0]

        // Next must not already be visited
        guard !visited.contains(next.id) else { return nil }

        // Next must have exactly one producer (no merge points)
        guard next.inputs.count == 1 else { return nil }

        // Case 1: Next is eligible — normal chain extension
        if next.isEligible {
            return next
        }

        // Case 2: Absorbable non-matmul op (attention pattern).
        // A non-eligible op can be absorbed if:
        // - It's not a matmul (it's a reduce/elementwise that runs within each unit's slice)
        // - Its single successor is an eligible matmul (completing the attention pattern)
        // - It has exactly one consumer
        if next.opcode != .matmul,
           next.outputs.count == 1,
           next.outputs[0].inputs.count == 1,
           next.outputs[0].isEligible,
           !visited.contains(next.outputs[0].id) {
            return next
        }

        return nil
    }

    /// Build a FusedOpCluster from a chain of nodes.
    private func buildCluster(id: Int, chain: [HeteroNode]) -> FusedOpCluster {
        let nodeIDs = chain.map(\.id)

        // Primary shape: use the shape of the largest matmul in the chain,
        // or the last node's shape if no matmuls.
        let matmulNodes = chain.filter { $0.opcode == .matmul }
        let primaryShape: MatrixShape
        if let largest = matmulNodes.max(by: { $0.outputShape.flops < $1.outputShape.flops }) {
            primaryShape = largest.outputShape
        } else {
            primaryShape = chain.last!.outputShape
        }

        // Primary op: the dominant op type
        let primaryOp: HLOOpCode
        if !matmulNodes.isEmpty {
            primaryOp = .matmul
        } else {
            primaryOp = chain[0].opcode
        }

        // Classify fusion kind
        let fusionKind: FusionKind
        if chain.count == 1 {
            fusionKind = .single
        } else if isAttentionPattern(chain) {
            fusionKind = .attentionBlock
        } else if matmulNodes.count == chain.count {
            fusionKind = .matmulChain
        } else {
            fusionKind = .elementwiseChain
        }

        return FusedOpCluster(
            id: id,
            nodeIDs: nodeIDs,
            primaryShape: primaryShape,
            fusionKind: fusionKind,
            primaryOp: primaryOp
        )
    }

    /// Detect the attention pattern: matmul → non-matmul → matmul.
    private func isAttentionPattern(_ chain: [HeteroNode]) -> Bool {
        guard chain.count == 3 else { return false }
        return chain[0].opcode == .matmul
            && chain[1].opcode != .matmul
            && chain[2].opcode == .matmul
    }
}
