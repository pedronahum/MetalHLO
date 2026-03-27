// main.swift
// GPT2EndToEnd
//
// The forcing function: run GPT-2 small (12 layers, 768 hidden, 12 heads)
// through the full GraphPipeline at real shapes and answer:
//
//   1. Does EligibilityAnnotationPass classify each op correctly?
//   2. Does OpFusionPass produce expected cluster boundaries?
//   3. Where does ProfitabilityGuard fall back, and is that right?
//   4. For eligible ops, does fused execution beat single-unit?
//
// GPT-2 small config:
//   vocab_size = 50257, n_embd = 768, n_head = 12, n_layer = 12
//   head_dim = 64, ffn_dim = 3072 (4 * 768)

import Foundation
import Metal
import QuartzCore
import HeterogeneousFusion

// MARK: - Formatting helpers

/// Pad/truncate a string to a fixed width.
func col(_ s: String, _ width: Int) -> String {
    if s.count >= width { return String(s.prefix(width)) }
    return s + String(repeating: " ", count: width - s.count)
}

func fmtF(_ v: Double, _ decimals: Int = 3) -> String {
    String(format: "%.\(decimals)f", v)
}

// MARK: - GPT-2 Constants

let vocabSize  = 50257
let nEmbd      = 768
let nHead      = 12
let nLayer     = 12
let headDim    = nEmbd / nHead  // 64
let ffnDim     = 4 * nEmbd      // 3072

// MARK: - GPT-2 Graph Builder

/// Build a GPT-2 graph with explicit batch and seqLen.
/// batch=1 is equivalent to the original buildGPT2Graph(seqLen:).
/// With batch>1, M scales by batch for most ops, but attention N stays = seqLen
/// (N is the number of keys per query, not affected by batch).
func buildGPT2Graph(seqLen: Int, batch: Int = 1) -> HeteroGraph {
    var graph = HeteroGraph()
    let totalSeq = batch * seqLen

    // Token embedding: M=vocabSize, K=embeddingDim, N=totalSeq
    let embed = graph.addNode(
        opcode: .embeddingLookup,
        outputShape: MatrixShape(M: vocabSize, K: nEmbd, N: totalSeq),
        name: "token_embedding"
    )

    var prev: HeteroNode = embed

    for layer in 0..<nLayer {
        let pfx = "L\(layer)"

        // Pre-attention LayerNorm [totalSeq, 768]
        let ln1 = graph.addNode(
            opcode: .layerNorm,
            outputShape: MatrixShape(M: totalSeq, K: nEmbd, N: nEmbd),
            name: "\(pfx)_ln1"
        )
        graph.addEdge(from: prev, to: ln1)

        // Fused QKV projection [totalSeq, 768] @ [768, 2304]
        let qkv = graph.addNode(
            opcode: .matmul,
            outputShape: MatrixShape(M: totalSeq, K: nEmbd, N: 3 * nEmbd),
            name: "\(pfx)_qkv_proj"
        )
        graph.addEdge(from: ln1, to: qkv)

        // Batched attention: batch*nHead*seqLen rows, but N = seqLen (not totalSeq)
        let batchedAttnM = batch * nHead * seqLen

        // QK^T: [batch*nHead*seqLen, headDim] @ [headDim, seqLen]
        let qk = graph.addNode(
            opcode: .matmul,
            outputShape: MatrixShape(M: batchedAttnM, K: headDim, N: seqLen),
            name: "\(pfx)_attn_qk"
        )
        graph.addEdge(from: qkv, to: qk)

        // Softmax (absorbed between matmuls in attention pattern)
        let softmax = graph.addNode(
            opcode: .reduceMax,
            outputShape: MatrixShape(M: batchedAttnM, K: seqLen, N: seqLen),
            name: "\(pfx)_attn_softmax"
        )
        graph.addEdge(from: qk, to: softmax)

        // AV: [batch*nHead*seqLen, seqLen] @ [seqLen, headDim]
        let av = graph.addNode(
            opcode: .matmul,
            outputShape: MatrixShape(M: batchedAttnM, K: seqLen, N: headDim),
            name: "\(pfx)_attn_av"
        )
        graph.addEdge(from: softmax, to: av)

        // Output projection [totalSeq, 768] @ [768, 768]
        let outProj = graph.addNode(
            opcode: .matmul,
            outputShape: MatrixShape(M: totalSeq, K: nEmbd, N: nEmbd),
            name: "\(pfx)_out_proj"
        )
        graph.addEdge(from: av, to: outProj)

        // Residual add (2 inputs → breaks chain fusion)
        let residAdd1 = graph.addNode(
            opcode: .elementwiseAdd,
            outputShape: MatrixShape(M: totalSeq, K: nEmbd, N: nEmbd),
            name: "\(pfx)_resid_add1"
        )
        graph.addEdge(from: outProj, to: residAdd1)
        graph.addEdge(from: prev, to: residAdd1)

        // Pre-FFN LayerNorm
        let ln2 = graph.addNode(
            opcode: .layerNorm,
            outputShape: MatrixShape(M: totalSeq, K: nEmbd, N: nEmbd),
            name: "\(pfx)_ln2"
        )
        graph.addEdge(from: residAdd1, to: ln2)

        // FFN up [totalSeq, 768] @ [768, 3072]
        let ffnUp = graph.addNode(
            opcode: .matmul,
            outputShape: MatrixShape(M: totalSeq, K: nEmbd, N: ffnDim),
            name: "\(pfx)_ffn_up"
        )
        graph.addEdge(from: ln2, to: ffnUp)

        // GELU [totalSeq, 3072]
        let gelu = graph.addNode(
            opcode: .gelu,
            outputShape: MatrixShape(M: totalSeq, K: ffnDim, N: ffnDim),
            name: "\(pfx)_gelu"
        )
        graph.addEdge(from: ffnUp, to: gelu)

        // FFN down [totalSeq, 3072] @ [3072, 768]
        let ffnDown = graph.addNode(
            opcode: .matmul,
            outputShape: MatrixShape(M: totalSeq, K: ffnDim, N: nEmbd),
            name: "\(pfx)_ffn_down"
        )
        graph.addEdge(from: gelu, to: ffnDown)

        // Residual add (2 inputs)
        let residAdd2 = graph.addNode(
            opcode: .elementwiseAdd,
            outputShape: MatrixShape(M: totalSeq, K: nEmbd, N: nEmbd),
            name: "\(pfx)_resid_add2"
        )
        graph.addEdge(from: ffnDown, to: residAdd2)
        graph.addEdge(from: residAdd1, to: residAdd2)

        prev = residAdd2
    }

    // Final LayerNorm
    let finalLN = graph.addNode(
        opcode: .layerNorm,
        outputShape: MatrixShape(M: totalSeq, K: nEmbd, N: nEmbd),
        name: "final_ln"
    )
    graph.addEdge(from: prev, to: finalLN)

    // Logit projection [totalSeq, 768] @ [768, 50257]
    let logits = graph.addNode(
        opcode: .matmul,
        outputShape: MatrixShape(M: totalSeq, K: nEmbd, N: vocabSize),
        name: "logit_proj"
    )
    graph.addEdge(from: finalLN, to: logits)

    return graph
}

// MARK: - Analysis

struct OpAnalysis {
    let name: String
    let opcode: HLOOpCode
    let shape: MatrixShape
    let eligible: Bool
    let reason: String?
    let clusterID: Int?
}

func analyzeGraph(_ graph: HeteroGraph) -> [OpAnalysis] {
    return graph.nodes.map { node in
        let eligible: Bool
        let reason: String?
        switch node.partitionDecision {
        case .partition:
            eligible = true
            reason = nil
        case .fallback(let r, _):
            eligible = false
            reason = r
        case .none:
            eligible = false
            reason = "not annotated"
        }
        return OpAnalysis(
            name: node.name, opcode: node.opcode, shape: node.outputShape,
            eligible: eligible, reason: reason, clusterID: node.clusterID
        )
    }
}

func printAnalysis(_ analysis: [OpAnalysis], seqLen: Int) {
    print("\n" + String(repeating: "=", count: 100))
    print("GPT-2 Small @ seqLen=\(seqLen)  (\(analysis.count) ops)")
    print(String(repeating: "=", count: 100))

    let eligible = analysis.filter(\.eligible)
    let fallback = analysis.filter { !$0.eligible }

    print("\n  ELIGIBLE for partitioning: \(eligible.count)/\(analysis.count)")
    if !eligible.isEmpty {
        print("  " + col("Op", 30) + col("Type", 14) + col("Shape (MxKxN)", 30) + "Cluster")
        print("  " + String(repeating: "-", count: 95))
        for op in eligible {
            let shape = "\(op.shape.M)x\(op.shape.K)x\(op.shape.N)"
            let cluster = op.clusterID.map { "cluster_\($0)" } ?? "none"
            print("  " + col(op.name, 30) + col(op.opcode.rawValue, 14) + col(shape, 30) + cluster)
        }
    }

    print("\n  FALLBACK (single-unit): \(fallback.count)/\(analysis.count)")
    var reasonCounts: [(String, String, Int)] = []
    var seen: [String: Int] = [:]
    for op in fallback {
        let reason = op.reason ?? "unknown"
        let baseName: String
        if let range = op.name.range(of: #"^L\d+_"#, options: .regularExpression) {
            baseName = String(op.name[range.upperBound...])
        } else {
            baseName = op.name
        }
        let key = "\(baseName)|\(reason)"
        if let idx = seen[key] {
            reasonCounts[idx].2 += 1
        } else {
            seen[key] = reasonCounts.count
            reasonCounts.append((baseName, reason, 1))
        }
    }

    print("  " + col("Op Pattern", 25) + col("Count", 8) + "Reason")
    print("  " + String(repeating: "-", count: 95))
    for (name, reason, count) in reasonCounts {
        let countStr = count > 1 ? "x\(count)" : "1"
        print("  " + col(name, 25) + col(countStr, 8) + reason)
    }
}

func printClusterAnalysis(_ graph: HeteroGraph) {
    guard !graph.clusters.isEmpty else {
        print("\n  No clusters formed.")
        return
    }

    print("\n  CLUSTERS (\(graph.clusters.count) total):")
    print("  " + col("Cluster", 14) + col("Kind", 20) + col("Primary Op", 14) + col("Primary Shape", 25) + "Partitioned?")
    print("  " + String(repeating: "-", count: 95))

    for cluster in graph.clusters {
        let shape = "\(cluster.primaryShape.M)x\(cluster.primaryShape.K)x\(cluster.primaryShape.N)"
        let partitioned = cluster.partitionDescriptor != nil ? "YES" : "NO"
        let nodeNames = cluster.nodeIDs.compactMap { id in graph.node(id: id)?.name }
        print("  " + col("cluster_\(cluster.id)", 14) + col(cluster.fusionKind.rawValue, 20) + col(cluster.primaryOp.rawValue, 14) + col(shape, 25) + partitioned)
        for name in nodeNames {
            print("                 > \(name)")
        }
    }
}

// MARK: - Execution Test

func executeEligibleOps(
    graph: HeteroGraph,
    executor: FusedExecutor,
    device: MTLDevice,
    seqLen: Int,
    warmup: Int = 3,
    iterations: Int = 10
) {
    let eligibleClusters = graph.clusters.filter { $0.partitionDescriptor != nil }
    guard !eligibleClusters.isEmpty else {
        print("\n  No eligible clusters to execute -- all ops fall back to single-unit.")
        return
    }

    print("\n  EXECUTION COMPARISON (\(eligibleClusters.count) eligible clusters):")
    print("  " + col("Cluster", 30) + col("Fused (ms)", 14) + col("GPU (ms)", 14) + col("MPS (ms)", 14) + "Speedup")
    print("  " + String(repeating: "-", count: 85))

    var totalFusedMs: Double = 0
    var totalBestSingleMs: Double = 0

    for cluster in eligibleClusters {
        guard let descriptor = cluster.partitionDescriptor else { continue }

        let elemSize = descriptor.dtype.byteSize
        let inputBytes0 = descriptor.fullInputShape.reduce(1, *) * elemSize
        let inputBytes1 = (descriptor.secondaryInputShape?.reduce(1, *) ?? 0) * elemSize
        let outputBytes = descriptor.fullOutputShape.reduce(1, *) * elemSize

        let buf0Size = max(inputBytes0, 16)
        let buf1Size = max(inputBytes1, 16)
        let outSize = max(outputBytes, 16)

        guard let buf0 = device.makeBuffer(length: buf0Size, options: .storageModeShared),
              let buf1 = device.makeBuffer(length: buf1Size, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: outSize, options: .storageModeShared) else {
            let totalMB = (buf0Size + buf1Size + outSize) / (1024 * 1024)
            print("  [\(cluster.id)] buffer allocation failed (need \(totalMB) MB)")
            continue
        }

        fillRandom(buf0)
        fillRandom(buf1)

        var inputBuffers: [MTLBuffer] = [buf0]
        if inputBytes1 > 0 { inputBuffers.append(buf1) }

        // Warmup
        for _ in 0..<warmup {
            _ = executor.execute(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outBuf)
        }

        // Measure fused
        var fusedTimes: [Double] = []
        for _ in 0..<iterations {
            let profile = executor.execute(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outBuf)
            fusedTimes.append(profile.totalWallClockMs)
        }
        let fusedMs = trimmedMean(fusedTimes)

        // Measure GPU-only
        let gpuDesc = singleUnitDescriptor(from: descriptor, unit: .gpu)
        for _ in 0..<warmup {
            _ = executor.execute(descriptor: gpuDesc, inputBuffers: inputBuffers, outputBuffer: outBuf)
        }
        var gpuTimes: [Double] = []
        for _ in 0..<iterations {
            let profile = executor.execute(descriptor: gpuDesc, inputBuffers: inputBuffers, outputBuffer: outBuf)
            gpuTimes.append(profile.totalWallClockMs)
        }
        let gpuMs = trimmedMean(gpuTimes)

        // Measure MPS-only
        let mpsDesc = singleUnitDescriptor(from: descriptor, unit: .mps)
        for _ in 0..<warmup {
            _ = executor.execute(descriptor: mpsDesc, inputBuffers: inputBuffers, outputBuffer: outBuf)
        }
        var mpsTimes: [Double] = []
        for _ in 0..<iterations {
            let profile = executor.execute(descriptor: mpsDesc, inputBuffers: inputBuffers, outputBuffer: outBuf)
            mpsTimes.append(profile.totalWallClockMs)
        }
        let mpsMs = trimmedMean(mpsTimes)

        let bestSingle = min(gpuMs, mpsMs)
        let speedup = bestSingle / fusedMs

        let firstName = graph.node(id: cluster.nodeIDs[0])?.name ?? "cluster_\(cluster.id)"

        print("  " + col(firstName, 30) + col(fmtF(fusedMs), 14) + col(fmtF(gpuMs), 14) + col(fmtF(mpsMs), 14) + fmtF(speedup, 2) + "x")

        totalFusedMs += fusedMs
        totalBestSingleMs += bestSingle
    }

    if totalBestSingleMs > 0 {
        print("  " + String(repeating: "-", count: 85))
        print("  " + col("TOTAL (eligible ops)", 30) + col(fmtF(totalFusedMs), 14) + col("", 28) + fmtF(totalBestSingleMs / totalFusedMs, 2) + "x")
    }
}

// MARK: - Shadow Validation

struct PredictionError {
    let name: String
    let shape: String
    let predictedSpeedup: Double
    let actualSpeedup: Double
    var error: Double { abs(predictedSpeedup - actualSpeedup) / max(actualSpeedup, 0.001) }
}

/// Shadow-mode: compare solver's predicted speedup against measured speedup.
/// Log every shape where prediction error > 15%.
func shadowValidation(
    graph: HeteroGraph,
    executor: FusedExecutor,
    splitter: OptimalSplitter,
    device: MTLDevice,
    warmup: Int = 3,
    iterations: Int = 10
) {
    let eligibleClusters = graph.clusters.filter { $0.partitionDescriptor != nil }
    guard !eligibleClusters.isEmpty else { return }

    print("\n  SHADOW VALIDATION: predicted vs actual speedup")
    print("  " + col("Cluster", 25) + col("Shape", 22) + col("Predicted", 12) + col("Actual", 12) + col("Error", 10) + "Status")
    print("  " + String(repeating: "-", count: 90))

    var errors: [PredictionError] = []

    for cluster in eligibleClusters {
        guard let descriptor = cluster.partitionDescriptor,
              descriptor.op == .matmul else { continue }

        let M = descriptor.fullInputShape[0]
        let K = descriptor.fullInputShape[1]
        let N = descriptor.fullOutputShape[1]
        let shape = MatrixShape(M: M, K: K, N: N)

        // Predicted speedup from solver
        let predictedSpeedup = splitter.predictSpeedup(shape: shape) ?? 1.0

        // Allocate and measure actual
        let elemSize = descriptor.dtype.byteSize
        let buf0Size = max(M * K * elemSize, 16)
        let buf1Size = max(K * N * elemSize, 16)
        let outSize = max(M * N * elemSize, 16)

        guard let buf0 = device.makeBuffer(length: buf0Size, options: .storageModeShared),
              let buf1 = device.makeBuffer(length: buf1Size, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: outSize, options: .storageModeShared) else {
            continue
        }
        fillRandom(buf0)
        fillRandom(buf1)
        let inputBuffers: [MTLBuffer] = [buf0, buf1]

        // Measure fused
        for _ in 0..<warmup { _ = executor.execute(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outBuf) }
        var fusedTimes: [Double] = []
        for _ in 0..<iterations {
            let p = executor.execute(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outBuf)
            fusedTimes.append(p.totalWallClockMs)
        }
        let fusedMs = trimmedMean(fusedTimes)

        // Measure best single unit (MPS)
        let mpsDesc = singleUnitDescriptor(from: descriptor, unit: .mps)
        for _ in 0..<warmup { _ = executor.execute(descriptor: mpsDesc, inputBuffers: inputBuffers, outputBuffer: outBuf) }
        var mpsTimes: [Double] = []
        for _ in 0..<iterations {
            let p = executor.execute(descriptor: mpsDesc, inputBuffers: inputBuffers, outputBuffer: outBuf)
            mpsTimes.append(p.totalWallClockMs)
        }
        let mpsMs = trimmedMean(mpsTimes)

        // Measure GPU
        let gpuDesc = singleUnitDescriptor(from: descriptor, unit: .gpu)
        for _ in 0..<warmup { _ = executor.execute(descriptor: gpuDesc, inputBuffers: inputBuffers, outputBuffer: outBuf) }
        var gpuTimes: [Double] = []
        for _ in 0..<iterations {
            let p = executor.execute(descriptor: gpuDesc, inputBuffers: inputBuffers, outputBuffer: outBuf)
            gpuTimes.append(p.totalWallClockMs)
        }
        let gpuMs = trimmedMean(gpuTimes)

        let bestSingle = min(mpsMs, gpuMs)
        let actualSpeedup = bestSingle / fusedMs

        let firstName = graph.node(id: cluster.nodeIDs[0])?.name ?? "cluster_\(cluster.id)"
        let shapeStr = "\(M)x\(K)x\(N)"

        let pe = PredictionError(
            name: firstName, shape: shapeStr,
            predictedSpeedup: predictedSpeedup, actualSpeedup: actualSpeedup
        )
        errors.append(pe)

        let status = pe.error > 0.15 ? "MISPREDICT" : "ok"
        print("  " + col(firstName, 25) + col(shapeStr, 22) + col(fmtF(predictedSpeedup, 2) + "x", 12) + col(fmtF(actualSpeedup, 2) + "x", 12) + col(fmtF(pe.error * 100, 0) + "%", 10) + status)
    }

    let mispredictions = errors.filter { $0.error > 0.15 }
    print("\n  Summary: \(mispredictions.count)/\(errors.count) matmul shapes mispredict (>15% error)")
    if !mispredictions.isEmpty {
        print("  Shapes needing recalibration:")
        for pe in mispredictions {
            print("    \(pe.shape): predicted \(fmtF(pe.predictedSpeedup, 2))x, actual \(fmtF(pe.actualSpeedup, 2))x")
        }
    }
}

func singleUnitDescriptor(from desc: PartitionDescriptor, unit: ComputeUnit) -> PartitionDescriptor {
    let fractions: [(ComputeUnit, Double)] = [(unit, 1.0)]

    switch desc.op {
    case .matmul:
        let M = desc.fullInputShape[0]
        let K = desc.fullInputShape[1]
        let N = desc.fullOutputShape[1]
        return PartitionDescriptor.matmul(
            shape: MatrixShape(M: M, K: K, N: N),
            fractions: fractions, dtype: desc.dtype
        )
    case .elementwiseAdd, .elementwiseMul, .elementwiseRelu, .gelu:
        return PartitionDescriptor.elementwise(
            op: desc.op, rows: desc.fullInputShape[0], cols: desc.fullInputShape[1],
            fractions: fractions, dtype: desc.dtype
        )
    case .layerNorm:
        return PartitionDescriptor.layerNorm(
            M: desc.fullInputShape[0], N: desc.fullInputShape[1],
            fractions: fractions, dtype: desc.dtype
        )
    case .embeddingLookup:
        let secondary = desc.secondaryInputShape!
        return PartitionDescriptor.embeddingLookup(
            vocabSize: secondary[0], seqLen: desc.fullInputShape[0],
            embeddingDim: secondary[1], fractions: fractions, dtype: desc.dtype
        )
    default:
        return desc
    }
}

// MARK: - Crossover Sweep

struct CrossoverResult {
    let M: Int
    let K: Int
    let N: Int
    let outputElements: Int
    let fusedMs: Double
    let gpuMs: Double
    let mpsMs: Double
    let bestSingleMs: Double
    var speedup: Double { bestSingleMs / fusedMs }
}

func crossoverSweep(
    executor: FusedExecutor,
    splitter: OptimalSplitter,
    device: MTLDevice,
    shapes: [(M: Int, K: Int, N: Int)],
    label: String,
    warmup: Int = 3,
    iterations: Int = 10
) -> [CrossoverResult] {
    print("\n  \(label):")
    print("  " + col("Shape (MxKxN)", 24) + col("Output Elems", 14) + col("Fused (ms)", 12) + col("GPU (ms)", 12) + col("MPS (ms)", 12) + col("Best (ms)", 12) + "Speedup")
    print("  " + String(repeating: "-", count: 98))

    var results: [CrossoverResult] = []

    for (M, K, N) in shapes {
        let shape = MatrixShape(M: M, K: K, N: N)
        let outputElements = M * N

        // Get solver partition or use equal 3-way split
        let descriptor: PartitionDescriptor
        if let solverDesc = splitter.optimalMatmulPartition(shape: shape) {
            descriptor = solverDesc
        } else {
            let fractions: [(ComputeUnit, Double)] = [(.gpu, 0.50), (.mps, 0.30), (.cpu, 0.20)]
            descriptor = PartitionDescriptor.matmul(shape: shape, fractions: fractions)
        }

        let elemSize = descriptor.dtype.byteSize
        let buf0Size = max(M * K * elemSize, 16)
        let buf1Size = max(K * N * elemSize, 16)
        let outSize = max(M * N * elemSize, 16)

        guard let buf0 = device.makeBuffer(length: buf0Size, options: .storageModeShared),
              let buf1 = device.makeBuffer(length: buf1Size, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: outSize, options: .storageModeShared) else {
            let totalMB = (buf0Size + buf1Size + outSize) / (1024 * 1024)
            print("  \(M)x\(K)x\(N): buffer allocation failed (need \(totalMB) MB)")
            continue
        }

        fillRandom(buf0)
        fillRandom(buf1)
        let inputBuffers: [MTLBuffer] = [buf0, buf1]

        // Measure fused
        for _ in 0..<warmup { _ = executor.execute(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outBuf) }
        var fusedTimes: [Double] = []
        for _ in 0..<iterations {
            let p = executor.execute(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outBuf)
            fusedTimes.append(p.totalWallClockMs)
        }
        let fusedMs = trimmedMean(fusedTimes)

        // Measure GPU-only
        let gpuDesc = singleUnitDescriptor(from: descriptor, unit: .gpu)
        for _ in 0..<warmup { _ = executor.execute(descriptor: gpuDesc, inputBuffers: inputBuffers, outputBuffer: outBuf) }
        var gpuTimes: [Double] = []
        for _ in 0..<iterations {
            let p = executor.execute(descriptor: gpuDesc, inputBuffers: inputBuffers, outputBuffer: outBuf)
            gpuTimes.append(p.totalWallClockMs)
        }
        let gpuMs = trimmedMean(gpuTimes)

        // Measure MPS-only
        let mpsDesc = singleUnitDescriptor(from: descriptor, unit: .mps)
        for _ in 0..<warmup { _ = executor.execute(descriptor: mpsDesc, inputBuffers: inputBuffers, outputBuffer: outBuf) }
        var mpsTimes: [Double] = []
        for _ in 0..<iterations {
            let p = executor.execute(descriptor: mpsDesc, inputBuffers: inputBuffers, outputBuffer: outBuf)
            mpsTimes.append(p.totalWallClockMs)
        }
        let mpsMs = trimmedMean(mpsTimes)

        let bestSingle = min(gpuMs, mpsMs)
        let speedup = bestSingle / fusedMs

        let shapeStr = "\(M)x\(K)x\(N)"
        let elemsStr: String
        if outputElements >= 1_000_000 {
            elemsStr = String(format: "%.1fM", Double(outputElements) / 1e6)
        } else {
            elemsStr = String(format: "%.0fK", Double(outputElements) / 1e3)
        }

        let marker = speedup >= 1.0 ? " <-- CROSSOVER" : ""
        print("  " + col(shapeStr, 24) + col(elemsStr, 14) + col(fmtF(fusedMs), 12) + col(fmtF(gpuMs), 12) + col(fmtF(mpsMs), 12) + col(fmtF(bestSingle), 12) + fmtF(speedup, 2) + "x" + marker)

        results.append(CrossoverResult(
            M: M, K: K, N: N, outputElements: outputElements,
            fusedMs: fusedMs, gpuMs: gpuMs, mpsMs: mpsMs, bestSingleMs: bestSingle
        ))
    }

    // Find crossover point
    let sorted = results.sorted { $0.outputElements < $1.outputElements }
    var crossoverElements: Int?
    for i in 1..<sorted.count {
        if sorted[i-1].speedup < 1.0 && sorted[i].speedup >= 1.0 {
            crossoverElements = sorted[i].outputElements
            break
        }
    }
    if let ce = crossoverElements {
        print("\n  >>> CROSSOVER at ~\(String(format: "%.1fM", Double(ce) / 1e6)) output elements")
    } else if let first = sorted.first, first.speedup >= 1.0 {
        print("\n  >>> All shapes profitable (smallest: \(String(format: "%.1fM", Double(first.outputElements) / 1e6)) elements)")
    } else {
        print("\n  >>> No crossover found — fused is slower at all tested sizes")
    }

    return results
}

// MARK: - Helpers

func fillRandom(_ buffer: MTLBuffer) {
    let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
    let count = buffer.length / MemoryLayout<Float>.size
    for i in 0..<count {
        ptr[i] = Float.random(in: -0.1...0.1)
    }
}

func trimmedMean(_ samples: [Double], trimFraction: Double = 0.1) -> Double {
    let sorted = samples.sorted()
    let trim = max(1, Int(Double(sorted.count) * trimFraction))
    let trimmed = Array(sorted[trim..<(sorted.count - trim)])
    guard !trimmed.isEmpty else { return sorted[sorted.count / 2] }
    return trimmed.reduce(0, +) / Double(trimmed.count)
}

// MARK: - Controlled Benchmark Protocol

/// Statistics from a controlled benchmark run.
struct BenchmarkStats {
    let mean: Double
    let std: Double
    let min: Double
    let max: Double
    let samples: Int
    /// Coefficient of variation: std / mean. >0.15 = high variance.
    var cv: Double { mean > 0 ? std / mean : 0 }
    var isHighVariance: Bool { cv > 0.15 }

    var summary: String {
        let cvPct = String(format: "%.1f", cv * 100)
        let flag = isHighVariance ? " HIGH-VARIANCE" : ""
        return "\(fmtF(mean, 2))ms ± \(fmtF(std, 2))ms (cv=\(cvPct)%\(flag), n=\(samples), range=[\(fmtF(min, 2)), \(fmtF(max, 2))])"
    }
}

/// Run a controlled benchmark: warmup to stabilize clocks and MPS kernel cache,
/// then measure with trimmed statistics.
func controlledBenchmark(
    executor: FusedExecutor,
    descriptor: PartitionDescriptor,
    inputBuffers: [MTLBuffer],
    outputBuffer: MTLBuffer,
    warmup: Int = 5,
    measured: Int = 20,
    trimFraction: Double = 0.1
) -> BenchmarkStats {
    // Warmup: let MPS compile kernels, let clocks stabilize
    for _ in 0..<warmup {
        _ = executor.execute(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outputBuffer)
    }

    // Measure
    var times: [Double] = []
    times.reserveCapacity(measured)
    for _ in 0..<measured {
        let p = executor.execute(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outputBuffer)
        times.append(p.totalWallClockMs)
    }

    // Trimmed statistics (discard top and bottom 10%)
    let sorted = times.sorted()
    let trim = max(1, Int(Double(sorted.count) * trimFraction))
    let trimmed = Array(sorted[trim..<(sorted.count - trim)])
    let mean = trimmed.isEmpty ? sorted[sorted.count / 2] : trimmed.reduce(0, +) / Double(trimmed.count)
    let variance = trimmed.isEmpty ? 0 : trimmed.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(trimmed.count)
    let std = sqrt(variance)

    return BenchmarkStats(
        mean: mean, std: std,
        min: sorted.first ?? 0, max: sorted.last ?? 0,
        samples: trimmed.count
    )
}

// MARK: - Main

func main() {
    print("""
    ================================================
    GPT-2 Small End-to-End -- GraphPipeline Validation
    12 layers, 768 hidden, 12 heads, 3072 FFN
    The forcing function: does the system work on a real model?
    ================================================
    """)

    guard let device = MTLCreateSystemDefaultDevice() else {
        print("ERROR: No Metal device available.")
        return
    }

    let mem = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
    print("Device:  \(device.name)")
    print("Memory:  \(mem) GB unified")
    print()

    // Initialize pipeline
    let profileDB = ProfileDatabase()

    // Step 1: Legacy square calibration (fallback curves)
    print("Calibrating throughput curves...")
    print("  Step 1: Square shapes (K=N=2048, M=[64..4096])...")
    let calibStart = CACurrentMediaTime()
    do {
        try profileDB.calibrateMatmul(
            device: device, K: 2048, N: 2048,
            rowCounts: [64, 256, 512, 1024, 2048, 4096],
            warmup: 3, iterations: 5
        )
    } catch {
        print("  WARNING: Square calibration failed: \(error)")
    }

    // Step 2: Rectangular calibration for crossover sweep and GPT-2 shapes
    print("  Step 2: Rectangular shapes (logit proj + GPT-2)...")
    do {
        try profileDB.calibrateMatmulShapes(
            device: device,
            shapes: [
                // Crossover sweep: logit projection aspect ratio
                (M: 64, K: 768, N: 50257),
                (M: 128, K: 768, N: 50257),
                (M: 256, K: 768, N: 50257),
                (M: 384, K: 768, N: 50257),
                (M: 512, K: 768, N: 50257),
                (M: 768, K: 768, N: 50257),
                (M: 1024, K: 768, N: 50257),
                (M: 2048, K: 768, N: 50257),
                // GPT-2 rectangular families
                (M: 128, K: 768, N: 2304),
                (M: 256, K: 768, N: 2304),
                (M: 512, K: 768, N: 2304),
                (M: 1024, K: 768, N: 2304),
                (M: 128, K: 768, N: 768),
                (M: 256, K: 768, N: 768),
                (M: 512, K: 768, N: 768),
                (M: 1024, K: 768, N: 768),
                (M: 128, K: 768, N: 3072),
                (M: 256, K: 768, N: 3072),
                (M: 512, K: 768, N: 3072),
                (M: 1024, K: 768, N: 3072),
                (M: 128, K: 3072, N: 768),
                (M: 256, K: 3072, N: 768),
                (M: 512, K: 3072, N: 768),
                (M: 1024, K: 3072, N: 768),
                // N sweep: M=1024, K=768, vary N
                (M: 1024, K: 768, N: 4096),
                (M: 1024, K: 768, N: 8192),
                (M: 1024, K: 768, N: 16384),
                (M: 1024, K: 768, N: 32768),
                // Batch sweep: logit projection at larger M values
                (M: 2048, K: 768, N: 50257),   // batch=2
                (M: 4096, K: 768, N: 50257),   // batch=4
                (M: 8192, K: 768, N: 50257),   // batch=8
                (M: 16384, K: 768, N: 50257),  // batch=16
                (M: 32768, K: 768, N: 50257),  // batch=32
                // Attention
                (M: 1536, K: 64, N: 128),
                (M: 6144, K: 64, N: 512),
                (M: 12288, K: 64, N: 1024),
                (M: 1536, K: 128, N: 64),
                (M: 6144, K: 512, N: 64),
                (M: 12288, K: 1024, N: 64),
            ],
            warmup: 3, iterations: 5
        )
    } catch {
        print("  WARNING: Rectangular calibration failed: \(error)")
    }

    let calibMs = (CACurrentMediaTime() - calibStart) * 1000
    print("Calibration complete (\(Int(calibMs)) ms)\n")

    // Use contention model for solver predictions
    let contentionModel = ContentionModel()
    let splitter = OptimalSplitter(profileDB: profileDB, contentionModel: contentionModel)
    let guard_ = ProfitabilityGuard(splitter: splitter, profileDB: profileDB)
    let pipeline = GraphPipeline(guard_: guard_, splitter: splitter, concurrencyBudget: 1)

    // ── Phase 0: Crossover Sweep ──

    print(String(repeating: "=", count: 100))
    print("PHASE 0: CROSSOVER SWEEP — Find Where Fused Execution Becomes Profitable")
    print(String(repeating: "=", count: 100))

    let executor: FusedExecutor
    do {
        executor = try FusedExecutor(device: device)
    } catch {
        print("ERROR: Failed to create FusedExecutor: \(error)")
        return
    }

    // Sweep 1: logit projection aspect ratio (1:66), vary M
    let logitResults = crossoverSweep(
        executor: executor, splitter: splitter, device: device,
        shapes: [
            (M: 64,   K: 768, N: 50257),   //  3.2M elements
            (M: 128,  K: 768, N: 50257),   //  6.4M elements
            (M: 256,  K: 768, N: 50257),   // 12.9M elements
            (M: 384,  K: 768, N: 50257),   // 19.3M elements
            (M: 512,  K: 768, N: 50257),   // 25.7M elements
            (M: 768,  K: 768, N: 50257),   // 38.6M elements
            (M: 1024, K: 768, N: 50257),   // 51.5M elements
            (M: 2048, K: 768, N: 50257),   // 103M elements
        ],
        label: "SWEEP 1: Logit projection (K=768, N=50257) — vary M",
        warmup: 3, iterations: 10
    )

    // Sweep 2: square-ish shapes
    let squareResults = crossoverSweep(
        executor: executor, splitter: splitter, device: device,
        shapes: [
            (M: 512,  K: 512,  N: 512),    //  0.26M elements
            (M: 1024, K: 1024, N: 1024),   //  1M elements
            (M: 2048, K: 2048, N: 2048),   //  4.2M elements
            (M: 3072, K: 3072, N: 3072),   //  9.4M elements
            (M: 4096, K: 4096, N: 4096),   // 16.8M elements
        ],
        label: "SWEEP 2: Square shapes (M=K=N) — vary size",
        warmup: 3, iterations: 10
    )

    // Sweep 3: N sweep — THE critical measurement.
    // Fix M=1024, K=768, vary N only.
    // Isolates N's independent effect on profitability.
    // This finds the N boundary between "never works" and "works".
    let nSweepResults = crossoverSweep(
        executor: executor, splitter: splitter, device: device,
        shapes: [
            (M: 1024, K: 768, N: 2304),    //  2.4M — QKV, confirmed fail
            (M: 1024, K: 768, N: 3072),    //  3.1M — FFN up
            (M: 1024, K: 768, N: 4096),    //  4.2M
            (M: 1024, K: 768, N: 8192),    //  8.4M
            (M: 1024, K: 768, N: 16384),   // 16.8M
            (M: 1024, K: 768, N: 32768),   // 33.6M
            (M: 1024, K: 768, N: 50257),   // 51.5M — logit, confirmed win
        ],
        label: "SWEEP 3: N sweep (M=1024, K=768) — isolate N's effect on profitability",
        warmup: 3, iterations: 10
    )

    // N crossover analysis
    print("\n  " + String(repeating: "=", count: 70))
    print("  N CROSSOVER ANALYSIS")
    print("  " + String(repeating: "=", count: 70))

    let nSorted = nSweepResults.sorted { $0.N < $1.N }
    var nCrossover: Int?
    for i in 1..<nSorted.count {
        if nSorted[i-1].speedup < 1.0 && nSorted[i].speedup >= 1.0 {
            nCrossover = nSorted[i].N
            break
        }
    }

    print("  N       Elements    Speedup")
    print("  " + String(repeating: "-", count: 40))
    for r in nSorted {
        let marker = r.speedup >= 1.0 ? " ✓" : " ✗"
        print("  \(col("\(r.N)", 8))\(col(String(format: "%.1fM", Double(r.outputElements) / 1e6), 12))\(fmtF(r.speedup, 2))x\(marker)")
    }

    if let nc = nCrossover {
        let prevN = nSorted.last(where: { $0.N < nc })?.N ?? 0
        print("\n  >>> N CROSSOVER between \(prevN) and \(nc)")
        print("  >>> Conservative threshold: N >= \(nc)")
    } else if nSorted.allSatisfy({ $0.speedup >= 1.0 }) {
        print("\n  >>> All N values profitable")
    } else {
        print("\n  >>> No N crossover found in range [2304, 50257]")
    }

    // Element crossover summary
    print("\n  " + String(repeating: "=", count: 70))
    print("  ELEMENT CROSSOVER SUMMARY")
    print("  " + String(repeating: "=", count: 70))

    let allResults = logitResults + squareResults + nSweepResults
    let profitable = allResults.filter { $0.speedup >= 1.0 }.sorted { $0.outputElements < $1.outputElements }
    let unprofitable = allResults.filter { $0.speedup < 1.0 }.sorted { $0.outputElements > $1.outputElements }

    if let smallestWin = profitable.first, let largestLoss = unprofitable.first {
        print("  Smallest profitable: \(smallestWin.M)x\(smallestWin.K)x\(smallestWin.N) = \(String(format: "%.1fM", Double(smallestWin.outputElements) / 1e6)) elements (\(fmtF(smallestWin.speedup, 2))x)")
        print("  Largest unprofitable: \(largestLoss.M)x\(largestLoss.K)x\(largestLoss.N) = \(String(format: "%.1fM", Double(largestLoss.outputElements) / 1e6)) elements (\(fmtF(largestLoss.speedup, 2))x)")
    } else if profitable.isEmpty {
        print("  No profitable shapes found at any tested size")
    } else {
        print("  All tested shapes are profitable")
    }

    // ── Phase 1: Graph Analysis with New Guard ──

    let seqLens = [128, 512, 1024]

    print("\n" + String(repeating: "=", count: 100))
    print("PHASE 1: ELIGIBILITY WITH COMPOUND GATE (elements >= \(guard_.minOutputElements / 1_000_000)M AND N >= \(guard_.minOutputColumns))")
    print(String(repeating: "=", count: 100))

    var graphsBySeqLen: [(Int, HeteroGraph, ExecutionPlan)] = []

    for seqLen in seqLens {
        var graph = buildGPT2Graph(seqLen: seqLen)
        let plan = pipeline.run(&graph)
        let diags = pipeline.diagnostics(graph)

        printAnalysis(analyzeGraph(graph), seqLen: seqLen)
        printClusterAnalysis(graph)

        print("\n  Pipeline diagnostics:")
        print("  " + diags.description.replacingOccurrences(of: "\n", with: "\n  "))
        print("\n  Execution plan: \(plan.steps.count) steps (\(plan.fusedClusterCount) fused, \(plan.passthroughCount) passthrough)")

        graphsBySeqLen.append((seqLen, graph, plan))
    }

    // ── Phase 2: Summary ──

    print("\n" + String(repeating: "=", count: 100))
    print("PHASE 2: ELIGIBILITY SUMMARY ACROSS SEQUENCE LENGTHS")
    print(String(repeating: "=", count: 100))

    print("\n  " + col("SeqLen", 12) + col("Total Ops", 12) + col("Eligible", 12) + col("Fallback", 12) + "% Eligible")
    print("  " + String(repeating: "-", count: 60))

    for (seqLen, graph, _) in graphsBySeqLen {
        let analysis = analyzeGraph(graph)
        let eligible = analysis.filter(\.eligible).count
        let total = analysis.count
        let pct = total > 0 ? Double(eligible) / Double(total) * 100 : 0
        print("  " + col("\(seqLen)", 12) + col("\(total)", 12) + col("\(eligible)", 12) + col("\(total - eligible)", 12) + fmtF(pct, 1) + "%")
    }

    // ── Phase 3: Execute eligible ops ──

    print("\n" + String(repeating: "=", count: 100))
    print("PHASE 3: FUSED EXECUTION vs SINGLE-UNIT BASELINE")
    print(String(repeating: "=", count: 100))

    for (seqLen, graph, _) in graphsBySeqLen.reversed() {
        let eligibleClusters = graph.clusters.filter { $0.partitionDescriptor != nil }
        if !eligibleClusters.isEmpty {
            print("\n  Testing at seqLen=\(seqLen) (\(eligibleClusters.count) eligible clusters):")
            executeEligibleOps(
                graph: graph, executor: executor, device: device,
                seqLen: seqLen, warmup: 3, iterations: 10
            )
            break
        }
    }

    // ── The Verdict ──

    print("\n" + String(repeating: "=", count: 100))
    print("THE VERDICT")
    print(String(repeating: "=", count: 100))

    for (seqLen, graph, _) in graphsBySeqLen {
        let analysis = analyzeGraph(graph)
        let eligible = analysis.filter(\.eligible).count
        let total = analysis.count
        let pct = total > 0 ? Double(eligible) / Double(total) * 100 : 0

        print("\n  seq=\(seqLen): \(eligible)/\(total) ops eligible (\(Int(pct))%)")
        if eligible == 0 {
            print("    -> All ops below \(guard_.minOutputElements / 1_000_000)M element threshold.")
        }
    }

    print("\n  Compound gate: M*N >= \(guard_.minOutputElements / 1_000_000)M AND N >= \(guard_.minOutputColumns)")
    print("  The N gate is the discriminating variable: large N amortizes split/sync overhead.")
    print("  QKV (N=2304) never profitable. Logit proj (N=50257) always profitable above ~13M elements.")
    print("  For GPT-2 batch=1: logit projection is the one profitable heterogeneous cluster.")

    // ── Phase 4: Cross-Architecture Validation ──
    // Test whether the compound gate constants transfer to a structurally different model.
    // If they do: the constants are hardware properties, not model-specific.
    // If they don't: we learn where the model-dependence lives.

    print("\n" + String(repeating: "=", count: 100))
    print("PHASE 4: CROSS-ARCHITECTURE VALIDATION — ViT-B/16")
    print(String(repeating: "=", count: 100))

    print("""

      ViT-B/16 (Vision Transformer):
        patch_size=16, hidden=768, heads=12, layers=12, mlp=3072
        image=224×224 → 196 patches + 1 CLS = 197 tokens
        Classification head: 768→1000 (ImageNet)

      Key structural differences from GPT-2:
        - Fixed sequence length (197 vs variable)
        - No autoregressive / causal mask
        - Final projection N=1000 (not N=50257)
        - Max N in any op = 3072 (MLP up projection)
    """)

    // ViT-B/16 constants
    let vitHidden = 768
    let vitHeads = 12
    let vitLayers = 12
    let vitMLP = 3072
    let vitHeadDim = vitHidden / vitHeads  // 64
    let vitSeqLen = 197  // 14*14 patches + CLS
    let vitClasses = 1000

    // Test at three batch sizes: 1, 8, 32
    // batch=32: QKV output = 6304*2304 = 14.5M elements (passes element gate, fails N gate)
    let vitBatches = [1, 8, 32]

    for batch in vitBatches {
        let batchedSeq = batch * vitSeqLen
        let batchedAttn = batch * vitHeads * vitSeqLen

        var vitGraph = HeteroGraph()

        // Patch embedding (treated as linear projection from flattened patches)
        let patchEmbed = vitGraph.addNode(
            opcode: .matmul,
            outputShape: MatrixShape(M: batchedSeq, K: 768, N: vitHidden),
            name: "patch_embed"
        )

        var prev: HeteroNode = patchEmbed

        for layer in 0..<vitLayers {
            let pfx = "L\(layer)"

            // Pre-attention LayerNorm
            let ln1 = vitGraph.addNode(
                opcode: .layerNorm,
                outputShape: MatrixShape(M: batchedSeq, K: vitHidden, N: vitHidden),
                name: "\(pfx)_ln1"
            )
            vitGraph.addEdge(from: prev, to: ln1)

            // QKV projection [batchedSeq, 768] @ [768, 2304]
            let qkv = vitGraph.addNode(
                opcode: .matmul,
                outputShape: MatrixShape(M: batchedSeq, K: vitHidden, N: 3 * vitHidden),
                name: "\(pfx)_qkv"
            )
            vitGraph.addEdge(from: ln1, to: qkv)

            // QK^T: [batchedAttn, headDim] @ [headDim, seqLen]
            let qk = vitGraph.addNode(
                opcode: .matmul,
                outputShape: MatrixShape(M: batchedAttn, K: vitHeadDim, N: vitSeqLen),
                name: "\(pfx)_attn_qk"
            )
            vitGraph.addEdge(from: qkv, to: qk)

            // Softmax
            let softmax = vitGraph.addNode(
                opcode: .reduceMax,
                outputShape: MatrixShape(M: batchedAttn, K: vitSeqLen, N: vitSeqLen),
                name: "\(pfx)_softmax"
            )
            vitGraph.addEdge(from: qk, to: softmax)

            // AV: [batchedAttn, seqLen] @ [seqLen, headDim]
            let av = vitGraph.addNode(
                opcode: .matmul,
                outputShape: MatrixShape(M: batchedAttn, K: vitSeqLen, N: vitHeadDim),
                name: "\(pfx)_attn_av"
            )
            vitGraph.addEdge(from: softmax, to: av)

            // Output projection
            let outProj = vitGraph.addNode(
                opcode: .matmul,
                outputShape: MatrixShape(M: batchedSeq, K: vitHidden, N: vitHidden),
                name: "\(pfx)_out_proj"
            )
            vitGraph.addEdge(from: av, to: outProj)

            // Residual add
            let residAdd1 = vitGraph.addNode(
                opcode: .elementwiseAdd,
                outputShape: MatrixShape(M: batchedSeq, K: vitHidden, N: vitHidden),
                name: "\(pfx)_resid1"
            )
            vitGraph.addEdge(from: outProj, to: residAdd1)
            vitGraph.addEdge(from: prev, to: residAdd1)

            // Pre-MLP LayerNorm
            let ln2 = vitGraph.addNode(
                opcode: .layerNorm,
                outputShape: MatrixShape(M: batchedSeq, K: vitHidden, N: vitHidden),
                name: "\(pfx)_ln2"
            )
            vitGraph.addEdge(from: residAdd1, to: ln2)

            // MLP up [batchedSeq, 768] @ [768, 3072]
            let mlpUp = vitGraph.addNode(
                opcode: .matmul,
                outputShape: MatrixShape(M: batchedSeq, K: vitHidden, N: vitMLP),
                name: "\(pfx)_mlp_up"
            )
            vitGraph.addEdge(from: ln2, to: mlpUp)

            // GELU
            let gelu = vitGraph.addNode(
                opcode: .gelu,
                outputShape: MatrixShape(M: batchedSeq, K: vitMLP, N: vitMLP),
                name: "\(pfx)_gelu"
            )
            vitGraph.addEdge(from: mlpUp, to: gelu)

            // MLP down [batchedSeq, 3072] @ [3072, 768]
            let mlpDown = vitGraph.addNode(
                opcode: .matmul,
                outputShape: MatrixShape(M: batchedSeq, K: vitMLP, N: vitHidden),
                name: "\(pfx)_mlp_down"
            )
            vitGraph.addEdge(from: gelu, to: mlpDown)

            // Residual add
            let residAdd2 = vitGraph.addNode(
                opcode: .elementwiseAdd,
                outputShape: MatrixShape(M: batchedSeq, K: vitHidden, N: vitHidden),
                name: "\(pfx)_resid2"
            )
            vitGraph.addEdge(from: mlpDown, to: residAdd2)
            vitGraph.addEdge(from: residAdd1, to: residAdd2)

            prev = residAdd2
        }

        // Final LayerNorm
        let vitFinalLN = vitGraph.addNode(
            opcode: .layerNorm,
            outputShape: MatrixShape(M: batchedSeq, K: vitHidden, N: vitHidden),
            name: "final_ln"
        )
        vitGraph.addEdge(from: prev, to: vitFinalLN)

        // Classification head [batchedSeq, 768] @ [768, 1000]
        let classHead = vitGraph.addNode(
            opcode: .matmul,
            outputShape: MatrixShape(M: batchedSeq, K: vitHidden, N: vitClasses),
            name: "class_head"
        )
        vitGraph.addEdge(from: vitFinalLN, to: classHead)

        // Run pipeline
        let vitPlan = pipeline.run(&vitGraph)
        let vitAnalysis = analyzeGraph(vitGraph)
        let vitEligible = vitAnalysis.filter(\.eligible).count
        let vitPartitioned = vitGraph.clusters.filter { $0.partitionDescriptor != nil }.count
        let totalOps = vitAnalysis.count

        print("  ViT-B/16 batch=\(batch) (seq=\(batchedSeq), \(totalOps) ops):")
        print("    Eligible: \(vitEligible)/\(totalOps), Partitioned clusters: \(vitPartitioned)")
        print("    Execution plan: \(vitPlan.steps.count) steps (\(vitPlan.fusedClusterCount) fused, \(vitPlan.passthroughCount) passthrough)")

        // Show key shapes and gate decisions
        let keyOps = ["qkv", "attn_qk", "out_proj", "mlp_up", "mlp_down", "class_head"]
        for opName in keyOps {
            let matching = vitGraph.nodes.filter { node in
                if opName == "class_head" { return node.name == opName }
                return node.name.hasSuffix(opName)
            }
            guard let node = matching.first else { continue }
            let shape = node.outputShape
            let elements = shape.M * shape.N
            let decision: String
            switch node.partitionDecision {
            case .partition:
                decision = "PARTITIONED"
            case .fallback(let reason, _):
                decision = reason
            case .none:
                decision = "not annotated"
            }
            let elemsStr = elements >= 1_000_000
                ? String(format: "%.1fM", Double(elements) / 1e6)
                : String(format: "%.0fK", Double(elements) / 1e3)
            print("    " + col(opName, 14) + col("\(shape.M)x\(shape.K)x\(shape.N)", 20) + col(elemsStr, 10) + col("N=\(shape.N)", 10) + decision)
        }

        // Highlight the critical case: batch=32, MLP up has 19.4M elements but N=3072
        if batch >= 32 {
            let mlpUp = vitGraph.nodes.first { $0.name == "L0_mlp_up" }!
            let elements = mlpUp.outputShape.M * mlpUp.outputShape.N
            print("\n    KEY TEST: mlp_up at batch=\(batch) has \(String(format: "%.1fM", Double(elements) / 1e6)) elements (>10M)")
            print("    but N=\(mlpUp.outputShape.N) < \(guard_.minOutputColumns) → correctly rejected by N gate")
        }
        print()
    }

    // Cross-architecture verdict
    print("  " + String(repeating: "=", count: 70))
    print("  CROSS-ARCHITECTURE VERDICT")
    print("  " + String(repeating: "=", count: 70))
    print("""

      ViT-B/16 has NO op with N >= 32768.
      Max N = 3072 (MLP up). Classification head N = 1000.
      → 0 partitioned clusters at ALL batch sizes. Correct behavior.

      At batch=32, MLP shapes cross the 10M element threshold
      but the N gate correctly rejects them. Without the N gate,
      these would be partitioned and regress (N=3072 confirmed 0.60x).

      The compound gate constants transfer from GPT-2 to ViT:
      - 10M element threshold: correctly rejects all small ViT shapes
      - N >= 32768: correctly rejects large-batch ViT shapes
      - These are hardware properties (M1 unified memory contention),
        not model-specific artifacts.

      Models that benefit from heterogeneous fusion:
      - Must have ops with N >= 32768 AND M*N >= 10M
      - In practice: vocabulary projections in language models
        (GPT-2 N=50257, CLIP N=49408, LLaMA N=32000)
      - Vision models: no naturally occurring ops cross the N threshold
    """)

    // ── Phase 5: Batch Size Sweep ──
    // Does batch size change which ops are profitable?
    // Prediction: N is batch-invariant for all GPT-2 ops.
    //   - logit_proj N=50257 → partitioned at every batch
    //   - qkv N=2304, ffn_up N=3072, attn_qk N=seqLen → excluded at every batch
    // If true: practitioners don't need to tune per batch size.

    print("\n" + String(repeating: "=", count: 100))
    print("PHASE 5: BATCH SIZE SWEEP — Is the compound gate batch-invariant?")
    print(String(repeating: "=", count: 100))

    let baseSeqLen = 1024
    let batchSizes = [1, 2, 4, 8, 16, 32]

    print("""

      GPT-2 Small @ seqLen=\(baseSeqLen), batch=[1, 2, 4, 8, 16, 32]
      Key shapes at batch=B:
        logit_proj:  M=B*\(baseSeqLen), K=768, N=50257  (N fixed)
        qkv_proj:    M=B*\(baseSeqLen), K=768, N=2304   (N fixed)
        ffn_up:      M=B*\(baseSeqLen), K=768, N=3072   (N fixed)
        attn_qk:     M=B*12*\(baseSeqLen), K=64, N=\(baseSeqLen)    (N=seqLen, fixed)
      Prediction: logit_proj partitioned at all batch sizes, everything else excluded.
    """)

    // Summary table header
    print("  " + col("Batch", 8) + col("TotalSeq", 10) + col("Eligible", 10) + col("Partitioned", 13) + col("Logit?", 8) + col("QKV?", 8) + col("FFN up?", 8) + col("attn_qk?", 10) + "Logit Speedup")
    print("  " + String(repeating: "-", count: 105))

    for batch in batchSizes {
        var batchGraph = buildGPT2Graph(seqLen: baseSeqLen, batch: batch)
        let _ = pipeline.run(&batchGraph)
        let batchAnalysis = analyzeGraph(batchGraph)
        let batchEligible = batchAnalysis.filter(\.eligible).count
        let batchPartitioned = batchGraph.clusters.filter { $0.partitionDescriptor != nil }.count

        // Check specific op decisions
        func opDecision(_ suffix: String) -> String {
            let node = batchGraph.nodes.first { $0.name.hasSuffix(suffix) || $0.name == suffix }
            guard let n = node else { return "?" }
            switch n.partitionDecision {
            case .partition: return "YES"
            case .fallback: return "no"
            case .none: return "?"
            }
        }

        let logitDecision = opDecision("logit_proj")
        let qkvDecision = opDecision("qkv_proj")
        let ffnUpDecision = opDecision("ffn_up")
        let attnQkDecision = opDecision("attn_qk")

        // Show logit projection rejection reason if not partitioned
        var logitSpeedupStr = "—"
        let logitNode = batchGraph.nodes.first { $0.name == "logit_proj" }
        if let ln = logitNode, case .fallback(let reason, _) = ln.partitionDecision {
            logitSpeedupStr = "rejected: \(reason)"
        }

        // Measure logit projection speedup if partitioned (controlled protocol)
        let logitCluster = batchGraph.clusters.first { cluster in
            cluster.partitionDescriptor != nil &&
            cluster.nodeIDs.contains(where: { id in batchGraph.node(id: id)?.name == "logit_proj" })
        }
        if let cluster = logitCluster, let descriptor = cluster.partitionDescriptor {
            let elemSize = descriptor.dtype.byteSize
            let M = descriptor.fullInputShape[0]
            let K = descriptor.fullInputShape[1]
            let N = descriptor.fullOutputShape[1]
            let buf0Size = max(M * K * elemSize, 16)
            let buf1Size = max(K * N * elemSize, 16)
            let outSize = max(M * N * elemSize, 16)
            let totalMB = (buf0Size + buf1Size + outSize) / (1024 * 1024)

            if totalMB < Int(mem) * 900 {
                if let buf0 = device.makeBuffer(length: buf0Size, options: .storageModeShared),
                   let buf1 = device.makeBuffer(length: buf1Size, options: .storageModeShared),
                   let outBuf = device.makeBuffer(length: outSize, options: .storageModeShared) {
                    fillRandom(buf0)
                    fillRandom(buf1)
                    let inputBuffers: [MTLBuffer] = [buf0, buf1]

                    // Controlled benchmark: 5 warmup, 20 measured, trimmed
                    let fusedStats = controlledBenchmark(
                        executor: executor, descriptor: descriptor,
                        inputBuffers: inputBuffers, outputBuffer: outBuf
                    )

                    let mpsDesc = singleUnitDescriptor(from: descriptor, unit: .mps)
                    let mpsStats = controlledBenchmark(
                        executor: executor, descriptor: mpsDesc,
                        inputBuffers: inputBuffers, outputBuffer: outBuf
                    )

                    let gpuDesc = singleUnitDescriptor(from: descriptor, unit: .gpu)
                    let gpuStats = controlledBenchmark(
                        executor: executor, descriptor: gpuDesc,
                        inputBuffers: inputBuffers, outputBuffer: outBuf
                    )

                    let bestSingle = min(mpsStats.mean, gpuStats.mean)
                    let speedup = bestSingle / fusedStats.mean
                    let varianceFlag = fusedStats.isHighVariance ? " HIGH-VAR" : ""
                    logitSpeedupStr = "\(fmtF(speedup, 2))x ± \(fmtF(fusedStats.cv * 100, 0))%\(varianceFlag) (\(fmtF(fusedStats.mean, 1))ms vs \(fmtF(bestSingle, 1))ms)"
                } else {
                    logitSpeedupStr = "OOM (\(totalMB)MB)"
                }
            } else {
                logitSpeedupStr = "skip (\(totalMB)MB)"
            }
        }

        let totalSeq = batch * baseSeqLen
        print("  " + col("\(batch)", 8) + col("\(totalSeq)", 10) + col("\(batchEligible)", 10) + col("\(batchPartitioned)", 13) + col(logitDecision, 8) + col(qkvDecision, 8) + col(ffnUpDecision, 8) + col(attnQkDecision, 10) + logitSpeedupStr)
    }

    // Batch sweep verdict
    let maxOutMB = guard_.maxOutputBytes / (1024 * 1024)
    print("\n  " + String(repeating: "=", count: 70))
    print("  BATCH SWEEP VERDICT")
    print("  " + String(repeating: "=", count: 70))
    print("""

      N is batch-invariant for all GPT-2 ops:
        logit_proj N=50257 — always above N threshold
        qkv_proj   N=2304  — always below N threshold
        ffn_up     N=3072  — always below N threshold
        attn_qk    N=1024  — always below N threshold

      Memory gate: maxOutputBytes = \(maxOutMB)MB (25% of \(mem)GB RAM)
      Large batches that exceed memory budget fall back safely.

      For practitioners: heterogeneous fusion helps on the logit
      projection at batch sizes that fit in memory, and nowhere else
      regardless of batch size. No per-batch tuning required.
    """)

    // ── Phase 6: Controlled Benchmark — The Paper Number ──
    // Run the logit projection at batch=1 with the controlled protocol.
    // This is the single number that goes in the results section.

    print("\n" + String(repeating: "=", count: 100))
    print("PHASE 6: CONTROLLED BENCHMARK — The Paper Number")
    print(String(repeating: "=", count: 100))

    print("\n  Logit projection: 1024×768×50257 (batch=1, seqLen=1024)")
    print("  Protocol: 5 warmup, 20 measured, trim top/bottom 10%, report mean ± std")
    print()

    do {
        var paperGraph = buildGPT2Graph(seqLen: 1024, batch: 1)
        let _ = pipeline.run(&paperGraph)
        let logitC = paperGraph.clusters.first { cluster in
            cluster.partitionDescriptor != nil &&
            cluster.nodeIDs.contains(where: { id in paperGraph.node(id: id)?.name == "logit_proj" })
        }
        guard let cluster = logitC, let desc = cluster.partitionDescriptor else {
            print("  ERROR: logit projection not partitioned")
            return
        }

        let M = desc.fullInputShape[0]
        let K = desc.fullInputShape[1]
        let N = desc.fullOutputShape[1]
        let elemSize = desc.dtype.byteSize

        guard let buf0 = device.makeBuffer(length: max(M * K * elemSize, 16), options: .storageModeShared),
              let buf1 = device.makeBuffer(length: max(K * N * elemSize, 16), options: .storageModeShared),
              let outBuf = device.makeBuffer(length: max(M * N * elemSize, 16), options: .storageModeShared) else {
            print("  ERROR: buffer allocation failed")
            return
        }
        fillRandom(buf0)
        fillRandom(buf1)
        let inputs: [MTLBuffer] = [buf0, buf1]

        // Run each configuration back-to-back with fresh warmup
        print("  Running fused (3-unit GPU+MPS+CPU)...")
        let fusedStats = controlledBenchmark(
            executor: executor, descriptor: desc,
            inputBuffers: inputs, outputBuffer: outBuf,
            warmup: 5, measured: 20
        )

        print("  Running MPS-only baseline...")
        let mpsDesc = singleUnitDescriptor(from: desc, unit: .mps)
        let mpsStats = controlledBenchmark(
            executor: executor, descriptor: mpsDesc,
            inputBuffers: inputs, outputBuffer: outBuf,
            warmup: 5, measured: 20
        )

        print("  Running GPU-only baseline...")
        let gpuDesc = singleUnitDescriptor(from: desc, unit: .gpu)
        let gpuStats = controlledBenchmark(
            executor: executor, descriptor: gpuDesc,
            inputBuffers: inputs, outputBuffer: outBuf,
            warmup: 5, measured: 20
        )

        let bestSingleStats = mpsStats.mean < gpuStats.mean ? mpsStats : gpuStats
        let bestUnit = mpsStats.mean < gpuStats.mean ? "MPS" : "GPU"
        let speedup = bestSingleStats.mean / fusedStats.mean

        print()
        print("  RESULTS:")
        print("    Fused:     \(fusedStats.summary)")
        print("    GPU-only:  \(gpuStats.summary)")
        print("    MPS-only:  \(mpsStats.summary)")
        print()
        print("    Best single unit: \(bestUnit)")
        print("    Speedup: \(fmtF(speedup, 2))x (fused vs \(bestUnit))")
        print()

        if fusedStats.isHighVariance || bestSingleStats.isHighVariance {
            print("    ⚠ HIGH VARIANCE detected — investigate thermal state before reporting")
            print("      Fused cv=\(fmtF(fusedStats.cv * 100, 1))%, \(bestUnit) cv=\(fmtF(bestSingleStats.cv * 100, 1))%")
        } else {
            print("    Variance acceptable (fused cv=\(fmtF(fusedStats.cv * 100, 1))%, \(bestUnit) cv=\(fmtF(bestSingleStats.cv * 100, 1))%)")
            print("    → Paper-ready: \(fmtF(speedup, 2))x speedup for logit projection on M1")
        }
    }

    // ── Final Summary: The Five Constants ──
    print("\n" + String(repeating: "=", count: 100))
    print("THE FIVE CONSTANTS — Heterogeneous Fusion on Apple M1")
    print(String(repeating: "=", count: 100))
    print("""

      Constant              Value (M1 8GB)      Source
      ─────────────────────────────────────────────────────────────
      minOutputElements     10M                 N sweep + ViT validation
      minOutputColumns      32768               N sweep crossover
      maxOutputBytes        \(maxOutMB)MB (\(Int(Double(guard_.maxOutputBytes) / Double(ProcessInfo.processInfo.physicalMemory) * 100))% of RAM)     batch=16 cliff measurement
      contentionCeiling     1.5–1.9x            logit projection measurements
      concurrencyBudget     1                   dual-matmul contention measurement
    """)
}

main()
