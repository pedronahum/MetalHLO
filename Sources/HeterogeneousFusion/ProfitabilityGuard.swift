// ProfitabilityGuard.swift
// HeterogeneousFusion
//
// First-class gate deciding whether an op should be partitioned across
// multiple compute units or fall back to single-unit execution.
//
// The compiler pass calls guard.evaluate() before the solver. If it
// returns .fallback, the op passes through unchanged. This keeps the
// pass correct-by-construction rather than relying on the solver to
// naturally produce a 100/0/0 split as a proxy for "don't partition."

import Foundation

// MARK: - PartitionDecision

/// The outcome of evaluating whether an op should be partitioned.
public enum PartitionDecision: Sendable {
    /// Partition the op across units using the given descriptor.
    case partition(PartitionDescriptor)
    /// Do not partition — run on a single unit.
    case fallback(reason: String, recommendedUnit: ComputeUnit)
}

// MARK: - PartitionPolicy

/// Documents the deterministic policy for shapes where partitioning regresses.
///
/// ## Wide shapes (M << N, e.g., 512×4096)
///
/// These appear constantly in transformer inference (projection layers with
/// small batch sizes). Policy: **do not partition** when M < 384 regardless
/// of N, because:
/// - Row-split gives each of 3 units ~128 rows — below efficient tiling threshold
/// - Column-split adds extract/scatter overhead that exceeds the benefit
/// - MPS single-unit is typically fastest for these shapes
///
/// When batch dimension fusion is available (Phase 4), the graph scheduler
/// should fuse B consecutive invocations into (B*M)×N before evaluating
/// profitability. With B≥4 and M=512, the fused 2048×4096 crosses the
/// profitability threshold.
///
/// ## Small square shapes (≤512×512)
///
/// Dispatch overhead (~0.3ms for 3-unit commit-before-wait) exceeds the
/// total compute time. Policy: **always fallback** to best single unit.
///
/// ## Recalibration scope
///
/// The HardwareMonitor tracks deviation per-op. At graph level:
/// - **Per-op recalibration** (current): Fast, cheap. Used during Phase 4
///   initial integration. Recalibrate the op that triggered the threshold.
/// - **Heaviest-op proportional scaling** (future): When the graph's heaviest
///   op (by FLOP count) is recalibrated, scale other ops' predictions by the
///   observed bandwidth ratio. This approximates cross-op interference without
///   full graph recalibration.
/// - **Concurrency budget** (future): Limit simultaneous fused ops based on
///   measured bandwidth saturation from the contention measurement.
///
/// Phase 4 uses per-op recalibration. The graph scheduler tracks a global
/// `concurrencyBudget` (max simultaneous fused ops) initialized from the
/// contention measurement.
public enum PartitionPolicy {
    /// Shapes that are known to regress under partitioning.
    /// The guard returns `.fallback` for these without consulting the solver.
    public static func isKnownRegression(shape: MatrixShape) -> (Bool, String?) {
        // Primary gate: contention-adjusted element threshold (empirically derived, March 2026)
        if shape.M * shape.N < 10_000_000 {
            return (true, "output < 10M elements — contention and overhead dominate")
        }
        // Wide shapes with small M: neither split dimension works well
        if shape.M < 384 && shape.N > shape.M {
            return (true, "M=\(shape.M) too small for effective splitting with wide N=\(shape.N)")
        }
        return (false, nil)
    }
}

// MARK: - ProfitabilityGuard

/// Evaluates whether partitioning an op is profitable based on shape,
/// calibration data, and empirical thresholds derived from benchmarks.
///
/// The primary gate is an empirical **output element count** threshold,
/// not a predicted speedup. Predicted speedups from isolated calibration
/// don't account for concurrent memory bandwidth contention, making them
/// structurally unreliable for gating decisions.
///
/// Empirical evidence (GPT-2 end-to-end, March 2026):
/// - Shapes < 10M output elements: fused is SLOWER (0.4-0.7x)
/// - Shapes > 50M elements: fused wins (2.4x for 1024×50257)
/// - Crossover: ~10-50M elements (aspect-ratio dependent)
///
/// Secondary gates:
/// - Row-split requires M ≥ 128 per unit (3 units → M ≥ 384 minimum)
/// - Aspect ratio ≤ 100 (very narrow shapes lose even at large sizes)
/// - Wide shapes (N >> M) use column-split only when M*K ≥ 256K elements
public final class ProfitabilityGuard: Sendable {

    private let splitter: OptimalSplitter
    private let profileDB: ProfileDatabase

    /// Minimum total output elements (M×N) for partitioning to be considered.
    /// This is the primary gate — empirically derived from crossover measurement.
    /// Below this threshold, contention and overhead dominate parallelism benefit.
    public let minOutputElements: Int

    /// Minimum output columns (N) for partitioning to be profitable.
    /// Large N means each thread does more independent work per row, which
    /// amortizes split/sync overhead. Small N means threads finish fast and
    /// contention dominates regardless of total element count.
    ///
    /// Empirical: QKV (N=2304) never profitable even at 18.9M elements.
    /// Logit proj (N=50257) crosses over at 12.9M elements.
    /// N sweep (M=1024, K=768): N=16384 → 0.98x, N=32768 → 1.14x.
    /// Boundary is between 16384 and 32768. Conservative threshold: 32768.
    public let minOutputColumns: Int

    /// Maximum output bytes before memory pressure causes thrashing.
    ///
    /// Concurrent execution from 3 units requires ~3x the working set of
    /// single-unit execution (each unit holds its input slice + output slice
    /// simultaneously). On unified memory, this competes with system memory.
    ///
    /// Empirical (8GB M1, March 2026):
    /// - batch=8 logit (1.55GB output): 3.2x speedup — healthy
    /// - batch=16 logit (3.1GB output): 0.05x — catastrophic thrashing
    ///
    /// Default: 25% of physical RAM. Conservative — a system that falls back
    /// at large batch is safe; a system that thrashes is not.
    public let maxOutputBytes: Int

    /// Minimum rows (M) for row-split to be viable.
    /// With 3 units, each gets M/3 rows — needs enough for efficient tiling.
    public let minRowsForRowSplit: Int

    /// Minimum columns (N) for column-split to be viable.
    public let minColsForColumnSplit: Int

    public init(
        splitter: OptimalSplitter,
        profileDB: ProfileDatabase,
        minOutputElements: Int = 10_000_000,
        minOutputColumns: Int = 32_768,
        maxOutputBytes: Int? = nil,
        minRowsForRowSplit: Int = 384,
        minColsForColumnSplit: Int = 1024
    ) {
        self.splitter = splitter
        self.profileDB = profileDB
        self.minOutputElements = minOutputElements
        self.minOutputColumns = minOutputColumns
        // Default: 25% of physical RAM
        self.maxOutputBytes = maxOutputBytes
            ?? Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.25)
        self.minRowsForRowSplit = minRowsForRowSplit
        self.minColsForColumnSplit = minColsForColumnSplit
    }

    /// Initialize from a calibrated hardware profile.
    /// Gate constants come from the profile; solver and profileDB are provided separately.
    public convenience init(
        profile: HardwareProfile,
        splitter: OptimalSplitter,
        profileDB: ProfileDatabase
    ) {
        self.init(
            splitter: splitter,
            profileDB: profileDB,
            minOutputElements: profile.minOutputElements,
            minOutputColumns: profile.minOutputColumns,
            maxOutputBytes: profile.maxOutputBytes
        )
    }

    /// Evaluate whether a matmul of the given shape should be partitioned.
    ///
    /// Uses empirical element-count gates instead of predicted speedup.
    /// Returns `.partition(descriptor)` if profitable, or
    /// `.fallback(reason, unit)` if the op should run on a single unit.
    public func evaluateMatmul(shape: MatrixShape) -> PartitionDecision {
        let M = shape.M, K = shape.K, N = shape.N

        // Gate 1: Total output elements — empirically derived threshold.
        // Below this, contention and overhead dominate parallelism benefit.
        let outputElements = M * N
        if outputElements < minOutputElements {
            let best = bestSingleUnit(shape: shape)
            return .fallback(
                reason: "below element threshold (\(formatElements(outputElements)) < \(formatElements(minOutputElements)))",
                recommendedUnit: best
            )
        }

        // Gate 2: Minimum output columns — large N required to amortize overhead.
        // QKV (N=2304) never profitable even at 18.9M elements.
        // Logit proj (N=50257) crosses at 12.9M. N is the discriminating variable.
        if N < minOutputColumns {
            let best = bestSingleUnit(shape: shape)
            return .fallback(
                reason: "N too small for profitable fusion (N=\(N) < \(minOutputColumns))",
                recommendedUnit: best
            )
        }

        // Gate 3: Memory pressure — prevent thrashing at large batch sizes.
        // Concurrent 3-unit execution triples working set. On unified memory,
        // exceeding ~25% of RAM for the output alone causes catastrophic slowdown.
        let outputBytes = outputElements * MemoryLayout<Float>.size
        if outputBytes > maxOutputBytes {
            let best = bestSingleUnit(shape: shape)
            let outputMB = outputBytes / (1024 * 1024)
            let maxMB = maxOutputBytes / (1024 * 1024)
            return .fallback(
                reason: "output exceeds memory budget (\(outputMB)MB > \(maxMB)MB, \(Int(Double(maxOutputBytes) / Double(ProcessInfo.processInfo.physicalMemory) * 100))% of RAM)",
                recommendedUnit: best
            )
        }

        // Gate 4: Split dimension viability
        let canRowSplit = M >= minRowsForRowSplit
        let canColumnSplit = N >= minColsForColumnSplit && N > 2 * M && M * K >= 256 * 1024

        if !canRowSplit && !canColumnSplit {
            let best = bestSingleUnit(shape: shape)
            return .fallback(
                reason: "M too small for row-split (\(M) < \(minRowsForRowSplit)) and column-split not viable",
                recommendedUnit: best
            )
        }

        // Gate 5: Solver must produce a partition
        let descriptor: PartitionDescriptor?
        if canColumnSplit && !canRowSplit {
            descriptor = solverColumnSplitPartition(shape: shape)
        } else if canRowSplit && canColumnSplit {
            let rowDesc = splitter.optimalMatmulPartition(shape: shape)
            let colDesc = solverColumnSplitPartition(shape: shape)
            let rowTime = rowDesc.flatMap { splitter.predictFusedTime(descriptor: $0) }
            let colTime = colDesc.flatMap { predictFusedTimeColumnSplit(descriptor: $0, shape: shape) }
            switch (rowTime, colTime) {
            case let (.some(rt), .some(ct)):
                descriptor = ct < rt ? colDesc : rowDesc
            case (.some, .none):
                descriptor = rowDesc
            case (.none, .some):
                descriptor = colDesc
            case (.none, .none):
                descriptor = nil
            }
        } else {
            descriptor = splitter.optimalMatmulPartition(shape: shape)
        }

        guard let desc = descriptor else {
            let best = bestSingleUnit(shape: shape)
            return .fallback(
                reason: "solver returned nil (no throughput curves calibrated)",
                recommendedUnit: best
            )
        }

        return .partition(desc)
    }

    /// Evaluate a general op (dispatches to shape-specific logic).
    public func evaluate(op: HLOOpCode, shape: MatrixShape) -> PartitionDecision {
        switch op {
        case .matmul:
            return evaluateMatmul(shape: shape)
        case .convolution:
            return evaluateConvolution(shape: shape)
        case .elementwiseAdd, .elementwiseMul, .elementwiseRelu, .gelu:
            return evaluateElementwise(op: op, shape: shape)
        case .layerNorm:
            return evaluateLayerNorm(shape: shape)
        case .embeddingLookup:
            return evaluateEmbedding(shape: shape)
        case .depthAttention:
            return evaluateDepthAttention(shape: shape)
        default:
            return .fallback(
                reason: "op \(op.rawValue) not yet supported for heterogeneous partitioning",
                recommendedUnit: .gpu
            )
        }
    }

    /// Evaluate with ConvShape (preferred for convolution).
    public func evaluateConvolution(convShape: ConvShape) -> PartitionDecision {
        return evaluateConvolution(shape: convShape.asMatrixShape, convShape: convShape)
    }

    /// Evaluate whether an elementwise op should be partitioned.
    ///
    /// Elementwise ops are bandwidth-bound. GPU alone saturates the memory bus
    /// for typical shapes, so standalone partitioning rarely helps. The primary
    /// value of marking elementwise ops as eligible is **cluster absorption** —
    /// when an elementwise sits between two matmuls, absorbing it into the
    /// matmul cluster eliminates a sync point (3→1). This saves ~0.3ms per
    /// eliminated sync, which dominates the per-element compute cost.
    ///
    /// Policy: mark as eligible when the tensor is large enough that the
    /// cluster-absorbed case is meaningful (M ≥ 512, total ≥ 512K elements).
    /// Standalone execution will use GPU-only anyway since the solver's
    /// fractions heavily favor GPU for bandwidth-bound ops.
    public func evaluateElementwise(op: HLOOpCode, shape: MatrixShape) -> PartitionDecision {
        let totalElements = shape.M * shape.N

        // Gate 1: Total elements must be meaningful for cluster absorption value
        let minElementsForElementwise = 512 * 1024  // 512K elements
        if totalElements < minElementsForElementwise {
            return .fallback(
                reason: "elementwise too small (\(totalElements) < \(minElementsForElementwise) elements)",
                recommendedUnit: .gpu
            )
        }

        // Gate 2: Need enough rows for row-split
        if shape.M < 512 {
            return .fallback(
                reason: "too few rows for elementwise split (M=\(shape.M) < 512)",
                recommendedUnit: .gpu
            )
        }

        // 2-unit: GPU + CPU only (MPS has no native elementwise path).
        // Bandwidth-bound ops don't benefit from a third unit.
        let fractions: [(ComputeUnit, Double)] = [(.gpu, 0.65), (.cpu, 0.35)]

        let descriptor = PartitionDescriptor.elementwise(
            op: op, rows: shape.M, cols: shape.N, fractions: fractions
        )
        return .partition(descriptor)
    }

    /// Evaluate whether a LayerNorm op should be partitioned.
    ///
    /// LayerNorm is per-row independent — each row's mean/variance/normalize
    /// is self-contained. The partition is exact (no cross-unit reduction).
    /// The 2-pass memory access pattern makes this bandwidth-bound, similar
    /// to elementwise but with 3x the memory traffic per element.
    ///
    /// Policy: partition when there are enough rows to split meaningfully
    /// (M ≥ 256) and the hidden dimension is large enough that per-row
    /// work amortizes dispatch overhead (N ≥ 512).
    public func evaluateLayerNorm(shape: MatrixShape) -> PartitionDecision {
        let M = shape.M  // rows
        let N = shape.N  // hidden dimension

        // Gate 1: Need enough rows for meaningful split across 3 units
        if M < 256 {
            return .fallback(
                reason: "too few rows for layerNorm split (M=\(M) < 256)",
                recommendedUnit: .gpu
            )
        }

        // Gate 2: Hidden dimension must be large enough for per-row work
        if N < 512 {
            return .fallback(
                reason: "hidden dim too small for layerNorm (N=\(N) < 512) — 2-pass overhead dominates",
                recommendedUnit: .gpu
            )
        }

        // Gate 3: Total elements check
        let totalElements = M * N
        if totalElements < 256 * 1024 {
            return .fallback(
                reason: "layerNorm too small (\(totalElements) < 256K elements)",
                recommendedUnit: .gpu
            )
        }

        // 2-unit: GPU + CPU only (MPS has no native LayerNorm path).
        // Bandwidth-bound — value is cluster absorption, not standalone speedup.
        let fractions: [(ComputeUnit, Double)] = [(.gpu, 0.65), (.cpu, 0.35)]

        let descriptor = PartitionDescriptor.layerNorm(
            M: M, N: N, fractions: fractions
        )
        return .partition(descriptor)
    }

    /// Evaluate whether an embedding lookup op should be partitioned.
    ///
    /// Embedding lookup uses vocab-range split: each unit owns a shard of the
    /// embedding table by vocab index. All units read all token IDs and write
    /// output only for tokens in their vocab range (scatter-free).
    ///
    /// Shape convention: M=vocabSize, K=embeddingDim, N=seqLen.
    ///
    /// Policy: partition when:
    /// - vocabSize ≥ 8192 (enough rows per shard with 3 units)
    /// - seqLen ≥ 64 (enough work to amortize dispatch overhead)
    /// - embeddingDim ≥ 256 (enough bytes per row for memcpy to be efficient)
    public func evaluateEmbedding(shape: MatrixShape) -> PartitionDecision {
        let vocabSize = shape.M
        let embeddingDim = shape.K
        let seqLen = shape.N

        // Gate 1: Vocab must be large enough to shard across 3 units
        if vocabSize < 8192 {
            return .fallback(
                reason: "vocab too small for sharding (V=\(vocabSize) < 8192)",
                recommendedUnit: .gpu
            )
        }

        // Gate 2: Sequence must be long enough to amortize dispatch
        if seqLen < 64 {
            return .fallback(
                reason: "sequence too short (seqLen=\(seqLen) < 64)",
                recommendedUnit: .gpu
            )
        }

        // Gate 3: Embedding dimension must be wide enough for efficient memcpy
        if embeddingDim < 256 {
            return .fallback(
                reason: "embedding dim too narrow (D=\(embeddingDim) < 256)",
                recommendedUnit: .gpu
            )
        }

        // 2-unit: GPU + CPU only (ANE has no gather hardware).
        // TODO(Phase6): frequency-weighted split based on token distribution
        let fractions: [(ComputeUnit, Double)] = [(.gpu, 0.60), (.cpu, 0.40)]

        let descriptor = PartitionDescriptor.embeddingLookup(
            vocabSize: vocabSize,
            seqLen: seqLen,
            embeddingDim: embeddingDim,
            fractions: fractions
        )
        return .partition(descriptor)
    }

    /// Evaluate whether a convolution should be partitioned.
    ///
    /// Convolution is compute-bound — this is the one op class where all 3 units
    /// (GPU, MPS/ANE, CPU) participate meaningfully. MPSCNNConvolution routes to
    /// the Neural Engine for eligible shapes, making this ANE's home territory.
    ///
    /// Split strategy: output-channel split. Each unit computes a contiguous range
    /// of output channels using the full input (all Cin needed for each Cout).
    ///
    /// Shape convention: M = B*outH*outW, K = Cin*kH*kW, N = Cout.
    ///
    /// Policy:
    /// - outChannels ≥ 96 (min 32 per unit with 3 units)
    /// - total FLOPs ≥ threshold (enough compute to amortize dispatch)
    /// - MPS-biased fractions: MPS ≥ 0.50 (ANE dominates conv)
    public func evaluateConvolution(shape: MatrixShape) -> PartitionDecision {
        // Reconstruct a default ConvShape from MatrixShape for the simple path
        // (used when called from evaluate(op:shape:) without ConvShape)
        return evaluateConvolution(shape: shape, convShape: nil)
    }

    private func evaluateConvolution(shape: MatrixShape, convShape: ConvShape?) -> PartitionDecision {
        let outChannels = shape.N

        // Gate 1: Enough output channels for 3-way split
        let minOutChannels = 96  // ~32 per unit minimum
        if outChannels < minOutChannels {
            return .fallback(
                reason: "too few output channels for conv split (Cout=\(outChannels) < \(minOutChannels))",
                recommendedUnit: .mps  // MPS/ANE is best single unit for conv
            )
        }

        // Gate 2: Total FLOPs must be significant
        // For conv, FLOPs = 2 * M * K * N where M=B*outH*outW, K=Cin*kH*kW, N=Cout
        let totalFlops = shape.flops
        let minFlops: Double = 50_000_000  // 50M FLOPs minimum
        if totalFlops < minFlops {
            return .fallback(
                reason: "conv too small (\(String(format: "%.1fM", totalFlops / 1e6)) < 50M FLOPs)",
                recommendedUnit: .mps
            )
        }

        // Gate 3: Total output elements must justify dispatch overhead
        let outputElements = shape.M * shape.N
        if outputElements < 64 * 1024 {
            return .fallback(
                reason: "conv output too small (\(outputElements) < 64K elements)",
                recommendedUnit: .mps
            )
        }

        // 3-unit: GPU + MPS + CPU — MPS-biased (ANE dominates conv)
        // The fractions reflect ANE's fixed-function conv advantage:
        // MPS ≥ 0.50 (ANE), GPU ~0.30 (custom Metal), CPU ~0.20 (Accelerate GEMM)
        let fractions: [(ComputeUnit, Double)] = [(.gpu, 0.30), (.mps, 0.50), (.cpu, 0.20)]

        if let cs = convShape {
            let descriptor = PartitionDescriptor.convolution(shape: cs, fractions: fractions)
            return .partition(descriptor)
        } else {
            // Fallback: can't build convolution descriptor without ConvShape
            // Return partition decision with a matmul-equivalent descriptor
            let descriptor = PartitionDescriptor.matmul(shape: shape, fractions: fractions)
            return .partition(descriptor)
        }
    }

    /// Evaluate whether a depth attention op should be partitioned.
    ///
    /// Depth attention (Attention Residuals) operates over a tiny depth dimension
    /// (4-16 blocks, max 32). The total element count is always small — at most
    /// batch × hidden × 32 — making partitioning overhead far larger than the
    /// compute. Always fall back to single-unit GPU execution.
    public func evaluateDepthAttention(shape: MatrixShape) -> PartitionDecision {
        return .fallback(
            reason: "depth attention is always tiny (depth ≤ 32) — partitioning overhead exceeds compute",
            recommendedUnit: .gpu
        )
    }

    // MARK: - Helpers

    /// Best single unit from calibration data, or .gpu as default.
    private func bestSingleUnit(shape: MatrixShape) -> ComputeUnit {
        if let (unit, _) = splitter.predictBestSingleTime(shape: shape) {
            return unit
        }
        return .gpu
    }

    /// Format element count for human-readable display.
    private func formatElements(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1e6) }
        if n >= 1_000 { return String(format: "%.0fK", Double(n) / 1e3) }
        return "\(n)"
    }

    /// Build a column-split partition using the solver's fractions.
    /// The solver works on rows (M dimension), but we apply its fraction
    /// logic to columns (N dimension) for the column-split case.
    private func solverColumnSplitPartition(shape: MatrixShape) -> PartitionDescriptor? {
        // Use the solver to get optimal fractions (it works generically on fractions)
        guard let rowDesc = splitter.optimalMatmulPartition(shape: shape) else { return nil }
        // Re-apply the same fractions as a column split
        let fractions = rowDesc.assignments.map { ($0.unit, $0.workFraction) }
        return PartitionDescriptor.matmulColumnSplit(shape: shape, fractions: fractions)
    }

    /// Predict fused time for a column-split descriptor.
    /// Column-split adds extract/scatter overhead (~0.1ms per MB copied).
    private func predictFusedTimeColumnSplit(descriptor: PartitionDescriptor, shape: MatrixShape) -> Double? {
        // Base prediction from the row-split model (same compute, different data movement)
        guard let baseFused = splitter.predictFusedTime(descriptor: descriptor) else { return nil }
        // Estimate extract+scatter overhead: ~0.1ms per MB for GPU/CPU (MPS is zero-copy)
        let elemSize = descriptor.dtype.byteSize
        var overheadMs: Double = 0
        for assignment in descriptor.assignments where assignment.unit != .mps {
            let sliceN = assignment.outputSlice.shape.last ?? 0
            let bBytes = Double(shape.K * sliceN * elemSize)
            let cBytes = Double(shape.M * sliceN * elemSize)
            overheadMs += (bBytes + cBytes) / (1024.0 * 1024.0) * 0.1
        }
        return baseFused + overheadMs
    }
}
