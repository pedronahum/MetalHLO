// HeteroGraph.swift
// HeterogeneousFusion
//
// Phase 4: Graph representation for the heterogeneous scheduling pipeline.
// Nodes represent individual HLO operations; edges represent data dependencies.
// The 4-pass pipeline annotates, fuses, partitions, and schedules this graph.

import Foundation

// MARK: - FusionKind

/// Classifies the kind of fused operation cluster.
public enum FusionKind: String, Sendable {
    /// A chain of matmuls (e.g., projection layers).
    case matmulChain
    /// QK^T → softmax → AV attention block.
    case attentionBlock
    /// Chain of elementwise ops following a matmul.
    case elementwiseChain
    /// Single op that doesn't fuse with neighbors.
    case single
}

// MARK: - HeteroNode

/// A node in the heterogeneous execution graph.
///
/// Each node wraps a single HLO operation with its shape, data dependencies,
/// and annotations written by the 4-pass pipeline.
public final class HeteroNode: @unchecked Sendable {

    /// Unique identifier within the graph.
    public let id: Int

    /// The operation this node represents.
    public let opcode: HLOOpCode

    /// Output shape of this operation (as a MatrixShape for matmul-like ops).
    /// For non-matmul ops, M and N represent the output dimensions.
    public let outputShape: MatrixShape

    /// Human-readable label (e.g., "qk_matmul", "softmax_0").
    public let name: String

    /// Nodes whose outputs feed into this node's inputs.
    public internal(set) var inputs: [HeteroNode] = []

    /// Nodes that consume this node's output.
    public internal(set) var outputs: [HeteroNode] = []

    // MARK: - Pass Annotations

    /// Set by EligibilityAnnotationPass: whether this op should be partitioned.
    public var partitionDecision: PartitionDecision?

    /// Set by OpFusionPass: which cluster this node belongs to (nil = passthrough).
    public internal(set) var clusterID: Int?

    public init(id: Int, opcode: HLOOpCode, outputShape: MatrixShape, name: String) {
        self.id = id
        self.opcode = opcode
        self.outputShape = outputShape
        self.name = name
    }

    /// Whether this node was marked eligible for partitioning.
    public var isEligible: Bool {
        if case .partition = partitionDecision { return true }
        return false
    }
}

extension HeteroNode: Hashable {
    public static func == (lhs: HeteroNode, rhs: HeteroNode) -> Bool { lhs.id == rhs.id }
    public func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

extension HeteroNode: CustomStringConvertible {
    public var description: String { "\(name)(\(opcode.rawValue), \(outputShape))" }
}

// MARK: - FusedOpCluster

/// A cluster of adjacent, fusible operations that share a single partition solve
/// and execute within one set of command buffers (one sync point).
///
/// The canonical example is an attention block: QK^T → softmax → AV fused as
/// one cluster with one solve, one set of command buffers, one sync point.
public struct FusedOpCluster: Sendable {

    /// Unique cluster identifier.
    public let id: Int

    /// Node IDs in topological order within this cluster.
    public let nodeIDs: [Int]

    /// The primary shape used for partitioning (typically the cluster's
    /// output shape, or the largest matmul shape for attention blocks).
    public let primaryShape: MatrixShape

    /// Classification of the fusion pattern.
    public let fusionKind: FusionKind

    /// Set by HeterogeneousPartitioningPass: the partition descriptor for this cluster.
    public var partitionDescriptor: PartitionDescriptor?

    /// The primary op code (for solver dispatch).
    public let primaryOp: HLOOpCode

    public init(
        id: Int,
        nodeIDs: [Int],
        primaryShape: MatrixShape,
        fusionKind: FusionKind,
        primaryOp: HLOOpCode
    ) {
        self.id = id
        self.nodeIDs = nodeIDs
        self.primaryShape = primaryShape
        self.fusionKind = fusionKind
        self.primaryOp = primaryOp
    }
}

// MARK: - HeteroGraph

/// The heterogeneous execution graph.
///
/// Built from a sequence of HLO operations with explicit data dependencies.
/// The 4-pass pipeline transforms this graph:
///   1. EligibilityAnnotationPass → annotates nodes with partition decisions
///   2. OpFusionPass → groups nodes into FusedOpClusters
///   3. HeterogeneousPartitioningPass → solves partition per cluster
///   4. SequentialScheduler → produces a serial ExecutionPlan
public struct HeteroGraph: Sendable {

    /// All nodes in the graph, indexed by ID.
    public private(set) var nodes: [HeteroNode]

    /// Fused clusters (populated by OpFusionPass).
    public var clusters: [FusedOpCluster] = []

    private var nextNodeID: Int = 0

    public init() {
        self.nodes = []
    }

    /// Add a node to the graph. Returns the node for chaining.
    @discardableResult
    public mutating func addNode(
        opcode: HLOOpCode,
        outputShape: MatrixShape,
        name: String
    ) -> HeteroNode {
        let node = HeteroNode(id: nextNodeID, opcode: opcode, outputShape: outputShape, name: name)
        nextNodeID += 1
        nodes.append(node)
        return node
    }

    /// Add a directed edge: `from` produces data consumed by `to`.
    public func addEdge(from: HeteroNode, to: HeteroNode) {
        from.outputs.append(to)
        to.inputs.append(from)
    }

    /// Look up a node by ID.
    public func node(id: Int) -> HeteroNode? {
        nodes.first { $0.id == id }
    }

    /// Topological sort of all nodes (Kahn's algorithm).
    /// Returns nodes in valid execution order.
    public func topologicalSort() -> [HeteroNode] {
        var inDegree: [Int: Int] = [:]
        for node in nodes {
            inDegree[node.id] = node.inputs.count
        }

        var queue: [HeteroNode] = nodes.filter { inDegree[$0.id] == 0 }
        var sorted: [HeteroNode] = []

        while !queue.isEmpty {
            let node = queue.removeFirst()
            sorted.append(node)
            for output in node.outputs {
                inDegree[output.id]! -= 1
                if inDegree[output.id] == 0 {
                    queue.append(output)
                }
            }
        }

        return sorted
    }
}

// MARK: - ExecutionStep

/// A single step in the execution plan.
public enum ExecutionStep: Sendable {
    /// Execute a fused cluster (parallel within, using FusedExecutor).
    case fused(clusterID: Int)
    /// Execute a passthrough op on a single unit.
    case passthrough(nodeID: Int, unit: ComputeUnit)
}

// MARK: - ExecutionPlan

/// The output of the SequentialScheduler: an ordered list of execution steps.
///
/// Steps are serialized (concurrencyBudget = 1). Within a fused step,
/// the FusedExecutor runs GPU + MPS + CPU in parallel.
public struct ExecutionPlan: Sendable {
    /// Steps in execution order.
    public let steps: [ExecutionStep]
    /// Total number of fused clusters.
    public let fusedClusterCount: Int
    /// Total number of passthrough ops.
    public let passthroughCount: Int

    public init(steps: [ExecutionStep]) {
        self.steps = steps
        self.fusedClusterCount = steps.filter {
            if case .fused = $0 { return true }; return false
        }.count
        self.passthroughCount = steps.filter {
            if case .passthrough = $0 { return true }; return false
        }.count
    }
}

// MARK: - Graph Builders

extension HeteroGraph {

    /// Build an attention block graph: QK^T → softmax → AV.
    ///
    /// This is the canonical Phase 4 test case — three ops fused into one cluster.
    public static func attentionBlock(
        seqLen: Int, dK: Int, dV: Int
    ) -> HeteroGraph {
        var graph = HeteroGraph()

        // QK^T: [seqLen, dK] @ [dK, seqLen] → [seqLen, seqLen]
        let qk = graph.addNode(
            opcode: .matmul,
            outputShape: MatrixShape(M: seqLen, K: dK, N: seqLen),
            name: "qk_matmul"
        )

        // softmax: [seqLen, seqLen] → [seqLen, seqLen]
        let softmax = graph.addNode(
            opcode: .reduceMax,  // softmax uses reduceMax internally
            outputShape: MatrixShape(M: seqLen, K: seqLen, N: seqLen),
            name: "softmax"
        )

        // AV: [seqLen, seqLen] @ [seqLen, dV] → [seqLen, dV]
        let av = graph.addNode(
            opcode: .matmul,
            outputShape: MatrixShape(M: seqLen, K: seqLen, N: dV),
            name: "av_matmul"
        )

        graph.addEdge(from: qk, to: softmax)
        graph.addEdge(from: softmax, to: av)

        return graph
    }

    /// Build a simple linear chain of matmuls (e.g., MLP: x @ W1 → relu → @ W2).
    public static func matmulChain(shapes: [(MatrixShape, String)]) -> HeteroGraph {
        var graph = HeteroGraph()
        var prev: HeteroNode?

        for (shape, name) in shapes {
            let node = graph.addNode(opcode: .matmul, outputShape: shape, name: name)
            if let p = prev {
                graph.addEdge(from: p, to: node)
            }
            prev = node
        }

        return graph
    }
}
