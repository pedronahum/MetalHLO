// SequentialScheduler.swift
// HeterogeneousFusion
//
// Phase 4, Pass 4: Produces a serial ExecutionPlan from the annotated graph.
//
// Key insight from bandwidth contention measurement (2.02x slowdown):
// - Between clusters: SERIAL (concurrencyBudget = 1)
// - Within a cluster: PARALLEL (3-unit FusedExecutor)
//
// The win is not cross-op concurrency. It's op selection and ordering —
// choosing which ops deserve the 3-unit treatment and fusing adjacent
// eligible ops into larger units before partitioning.

import Foundation

/// Pass 4: Schedule clusters and passthrough ops in serial order.
///
/// Topologically sorts all execution units (clusters + passthrough nodes),
/// then serializes them into an ExecutionPlan. Within each fused step,
/// the FusedExecutor still runs GPU + MPS + CPU in parallel.
public struct SequentialScheduler: Sendable {

    /// Maximum concurrent fused ops (from bandwidth contention measurement).
    /// Default 1 means full serialization between clusters.
    public let concurrencyBudget: Int

    public init(concurrencyBudget: Int = 1) {
        self.concurrencyBudget = concurrencyBudget
    }

    /// Produce an execution plan from the annotated graph.
    ///
    /// The plan respects data dependencies via topological ordering.
    /// Clusters with partition descriptors become `.fused` steps;
    /// everything else becomes `.passthrough` on the recommended unit.
    public func schedule(_ graph: HeteroGraph) -> ExecutionPlan {
        let sorted = graph.topologicalSort()
        var steps: [ExecutionStep] = []
        var emittedClusters = Set<Int>()

        for node in sorted {
            if let clusterID = node.clusterID,
               !emittedClusters.contains(clusterID) {
                // Emit the cluster as a fused step (first time we see any node in it)
                let cluster = graph.clusters.first { $0.id == clusterID }
                if cluster?.partitionDescriptor != nil {
                    steps.append(.fused(clusterID: clusterID))
                } else {
                    // Cluster without partition → run as passthrough
                    for nid in cluster?.nodeIDs ?? [] {
                        let unit = recommendedUnit(for: graph.node(id: nid))
                        steps.append(.passthrough(nodeID: nid, unit: unit))
                    }
                }
                emittedClusters.insert(clusterID)
            } else if node.clusterID == nil {
                // Passthrough node
                let unit = recommendedUnit(for: node)
                steps.append(.passthrough(nodeID: node.id, unit: unit))
            }
            // else: node is part of an already-emitted cluster, skip
        }

        return ExecutionPlan(steps: steps)
    }

    /// Determine the recommended unit for a passthrough node.
    private func recommendedUnit(for node: HeteroNode?) -> ComputeUnit {
        guard let node = node else { return .gpu }
        if case .fallback(_, let unit) = node.partitionDecision {
            return unit
        }
        return .gpu  // default fallback
    }
}

// MARK: - GraphPipeline

/// The complete Phase 4 pipeline: runs all 4 passes in sequence.
///
/// Usage:
/// ```swift
/// let pipeline = GraphPipeline(guard_: guard, splitter: splitter, concurrencyBudget: 1)
/// var graph = HeteroGraph.attentionBlock(seqLen: 512, dK: 64, dV: 64)
/// let plan = pipeline.run(&graph)
/// ```
public struct GraphPipeline: Sendable {

    public let eligibilityPass: EligibilityAnnotationPass
    public let fusionPass: OpFusionPass
    public let partitioningPass: HeterogeneousPartitioningPass
    public let scheduler: SequentialScheduler

    public init(
        guard_: ProfitabilityGuard,
        splitter: OptimalSplitter,
        concurrencyBudget: Int = 1
    ) {
        self.eligibilityPass = EligibilityAnnotationPass(guard_: guard_)
        self.fusionPass = OpFusionPass()
        self.partitioningPass = HeterogeneousPartitioningPass(splitter: splitter)
        self.scheduler = SequentialScheduler(concurrencyBudget: concurrencyBudget)
    }

    /// Initialize from a calibrated hardware profile.
    /// Constructs ProfitabilityGuard with profile-derived constants.
    public init(
        profile: HardwareProfile,
        splitter: OptimalSplitter,
        profileDB: ProfileDatabase
    ) {
        let guard_ = ProfitabilityGuard(
            profile: profile,
            splitter: splitter,
            profileDB: profileDB
        )
        self.eligibilityPass = EligibilityAnnotationPass(guard_: guard_)
        self.fusionPass = OpFusionPass()
        self.partitioningPass = HeterogeneousPartitioningPass(splitter: splitter)
        self.scheduler = SequentialScheduler(concurrencyBudget: profile.concurrencyBudget)
    }

    /// Run the full 4-pass pipeline and return the execution plan.
    ///
    /// Mutates the graph in place (annotations, clusters, partition descriptors).
    @discardableResult
    public func run(_ graph: inout HeteroGraph) -> ExecutionPlan {
        // Pass 1: Annotate eligibility
        eligibilityPass.run(&graph)

        // Pass 2: Fuse adjacent eligible ops into clusters
        fusionPass.run(&graph)

        // Pass 3: Solve partition for each cluster
        partitioningPass.run(&graph)

        // Pass 4: Schedule
        return scheduler.schedule(graph)
    }

    /// Diagnostic summary of pipeline results.
    public func diagnostics(_ graph: HeteroGraph) -> PipelineDiagnostics {
        let (eligible, fallback) = eligibilityPass.summary(graph)
        let (partitioned, unpartitioned) = partitioningPass.summary(graph)
        let clusterSizes = graph.clusters.map(\.nodeIDs.count)
        let attentionClusters = graph.clusters.filter { $0.fusionKind == .attentionBlock }.count

        return PipelineDiagnostics(
            totalNodes: graph.nodes.count,
            eligibleNodes: eligible,
            fallbackNodes: fallback,
            totalClusters: graph.clusters.count,
            partitionedClusters: partitioned,
            unpartitionedClusters: unpartitioned,
            attentionClusters: attentionClusters,
            clusterSizes: clusterSizes
        )
    }
}

/// Diagnostic output from the pipeline.
public struct PipelineDiagnostics: Sendable, CustomStringConvertible {
    public let totalNodes: Int
    public let eligibleNodes: Int
    public let fallbackNodes: Int
    public let totalClusters: Int
    public let partitionedClusters: Int
    public let unpartitionedClusters: Int
    public let attentionClusters: Int
    public let clusterSizes: [Int]

    public var description: String {
        var lines: [String] = []
        lines.append("  Nodes: \(totalNodes) total, \(eligibleNodes) eligible, \(fallbackNodes) fallback")
        lines.append("  Clusters: \(totalClusters) total, \(partitionedClusters) partitioned, \(unpartitionedClusters) unpartitioned")
        if attentionClusters > 0 {
            lines.append("  Attention blocks: \(attentionClusters)")
        }
        if !clusterSizes.isEmpty {
            lines.append("  Cluster sizes: \(clusterSizes)")
        }
        return lines.joined(separator: "\n")
    }
}
