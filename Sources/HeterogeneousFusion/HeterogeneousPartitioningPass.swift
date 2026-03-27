// HeterogeneousPartitioningPass.swift
// HeterogeneousFusion
//
// Phase 4, Pass 3: Solves the optimal partition for each FusedOpCluster
// using the existing OptimalSplitter + ProfitabilityGuard stack.
//
// This is the existing Phase 2/3 stack, called once per cluster.
// The cluster's primaryShape determines the partition dimensions.

import Foundation

/// Pass 3: Solve partition descriptors for each fused cluster.
///
/// Uses the OptimalSplitter to find the optimal work split for each cluster's
/// primary shape. Column-split vs row-split is selected automatically via
/// `PartitionDescriptor.matmulAutoSplit()`.
public struct HeterogeneousPartitioningPass: Sendable {

    private let splitter: OptimalSplitter

    public init(splitter: OptimalSplitter) {
        self.splitter = splitter
    }

    /// Run the pass, setting `partitionDescriptor` on each cluster.
    public func run(_ graph: inout HeteroGraph) {
        for i in graph.clusters.indices {
            let cluster = graph.clusters[i]

            guard cluster.primaryOp == .matmul else {
                // Only matmul is supported for heterogeneous partitioning.
                // Other cluster types pass through without a partition descriptor.
                continue
            }

            // Solve optimal partition for this cluster's primary shape
            guard let baseDescriptor = splitter.optimalMatmulPartition(
                shape: cluster.primaryShape
            ) else {
                continue
            }

            // Re-build with auto-split selection (row vs column)
            let fractions = baseDescriptor.assignments.map {
                ($0.unit, $0.workFraction)
            }
            let descriptor = PartitionDescriptor.matmulAutoSplit(
                shape: cluster.primaryShape,
                fractions: fractions
            )

            graph.clusters[i].partitionDescriptor = descriptor
        }
    }

    /// Summary: how many clusters got a partition descriptor.
    public func summary(_ graph: HeteroGraph) -> (partitioned: Int, unpartitioned: Int) {
        var p = 0, u = 0
        for cluster in graph.clusters {
            if cluster.partitionDescriptor != nil { p += 1 } else { u += 1 }
        }
        return (p, u)
    }
}
