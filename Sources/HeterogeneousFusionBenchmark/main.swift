// main.swift
// HeterogeneousFusionBenchmark
//
// Phase 1: Prove that simultaneous cross-unit execution of a single matmul
// beats any single-unit execution.

import Foundation
import Metal
import QuartzCore
import HeterogeneousFusion

// MARK: - CLI

func printHeader() {
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Heterogeneous Fusion — Phase 1 Benchmark                   ║
    ║  Cross-unit matmul: GPU (Metal) + ANE (MPS) + CPU (BLAS)    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
}

func printHardwareInfo(device: MTLDevice) {
    let mem = ProcessInfo.processInfo.physicalMemory
    let memGB = mem / (1024 * 1024 * 1024)
    print("Device:  \(device.name)")
    print("Memory:  \(memGB) GB unified")
    print("OS:      \(ProcessInfo.processInfo.operatingSystemVersionString)")
    print()
}

// MARK: - Shapes from spec

let specShapes: [MatrixShape] = [
    .square(512),
    .square(2048),
    .square(4096),
    .square(8192),
    MatrixShape(M: 4096, K: 4096, N: 512),   // skinny
    MatrixShape(M: 512,  K: 512,  N: 4096),  // transposed skinny
]

// MARK: - Main

func main() {
    printHeader()

    guard let device = MTLCreateSystemDefaultDevice() else {
        print("ERROR: No Metal device available.")
        return
    }
    printHardwareInfo(device: device)

    // Parse args
    let args = CommandLine.arguments
    let quick = args.contains("-q") || args.contains("--quick")

    let config: HeterogeneousBenchmarkConfig = quick ? .quick : .standard
    print("Mode: \(quick ? "Quick" : "Standard") (\(config.warmupIterations) warmup, \(config.iterations) measurements, trim \(Int(config.trimFraction * 100))%)")
    print()

    let harness: BenchmarkHarness
    do {
        harness = try BenchmarkHarness(device: device, config: config)
    } catch {
        print("ERROR: Failed to initialize harness: \(error)")
        return
    }

    // Choose shapes
    let shapes: [MatrixShape]
    if let sizeArg = args.first(where: { $0.hasPrefix("--size=") }) {
        let size = Int(sizeArg.dropFirst("--size=".count)) ?? 2048
        shapes = [.square(size)]
    } else {
        shapes = specShapes
    }

    // ── Systematic Correctness Suite ──
    print("Correctness Suite: all (shape, split) combinations")
    print(String(repeating: "-", count: 70))
    do {
        let correctnessResults = try harness.runCorrectnessSuite()
        var correctnessPasses = 0
        for tc in correctnessResults {
            let status = tc.pass ? "PASS" : "FAIL"
            correctnessPasses += tc.pass ? 1 : 0
            print(String(format: "  %-28s %-8s %@ (maxErr=%.6f)",
                         ("\(tc.shape) [\(tc.splitStrategy)]" as NSString).utf8String!,
                         (status as NSString).utf8String!,
                         tc.pass ? "" : "TOLERANCE EXCEEDED",
                         tc.maxError))
        }
        print()
        print("  Correctness: \(correctnessPasses)/\(correctnessResults.count) tests pass")
        if correctnessPasses < correctnessResults.count {
            print("  WARNING: Correctness failures must be resolved before Phase 4")
        }
    } catch {
        print("  Correctness suite ERROR: \(error)")
    }
    print()

    // Phase 3: Solver validation (calibrates per-shape for accuracy)
    print("Phase 3 Validation: Solver vs empirical optimal splits")
    print(String(repeating: "-", count: 70))
    let validationShapes: [MatrixShape] = [.square(512), .square(1024), .square(2048), .square(4096)]
    var solverPasses = 0
    for vs in validationShapes {
        do {
            let result = try harness.validateSolver(shape: vs)
            let status = result.pass ? "PASS" : "FAIL"
            solverPasses += result.pass ? 1 : 0
            let sign = result.solverFusedMs <= result.empiricalFusedMs ? "faster" : "slower"
            print(String(format: "  %@: %@ (time_dev=%.1f%% %@, frac_dev=%.3f)",
                         "\(vs)", status, result.timeDeviation * 100, sign, result.fractionDeviation))
            print(String(format: "    Empirical: gpu=%.0f%% mps=%.0f%% cpu=%.0f%% → %.3f ms",
                         result.empiricalFractions.gpu * 100, result.empiricalFractions.mps * 100,
                         result.empiricalFractions.cpu * 100, result.empiricalFusedMs))
            print(String(format: "    Solver:    gpu=%.0f%% mps=%.0f%% cpu=%.0f%% → %.3f ms",
                         result.solverFractions.gpu * 100, result.solverFractions.mps * 100,
                         result.solverFractions.cpu * 100, result.solverFusedMs))
        } catch {
            print("  \(vs): ERROR (\(error))")
        }
    }
    print()
    print("  Phase 3 gate: \(solverPasses)/\(validationShapes.count) shapes — solver within 5% of empirical")
    if solverPasses >= validationShapes.count / 2 {
        print("  RESULT: PASS → Solver is effective")
    } else {
        print("  RESULT: FAIL → Solver needs tuning")
    }
    if harness.monitor.meanDeviation > 0 {
        print(String(format: "  HardwareMonitor: mean deviation = %.1f%%, recalibrate = %@",
                     harness.monitor.meanDeviation * 100,
                     harness.monitor.shouldRecalibrate ? "YES" : "no"))
    }
    print()

    // ── Bandwidth Contention Measurement ──
    print("Bandwidth Contention: two concurrent fused matmuls")
    print(String(repeating: "-", count: 70))
    do {
        let contentionShape = MatrixShape.square(2048)
        let contention = try harness.measureBandwidthContention(shape: contentionShape)
        print(String(format: "  Shape:    %@", "\(contentionShape)"))
        print(String(format: "  Single:   %.3f ms", contention.singleFusedMs))
        print(String(format: "  Dual:     %.3f ms (wall-clock of both)", contention.dualFusedMs))
        print(String(format: "  Slowdown: %.2fx", contention.slowdownFactor))
        print(String(format: "  Budget:   %d concurrent fused ops recommended", contention.recommendedConcurrencyBudget))
        harness.monitor.concurrencyBudget = contention.recommendedConcurrencyBudget
    } catch {
        print("  Contention measurement ERROR: \(error)")
    }
    print()

    // ── Profitability Guard Validation ──
    print("Profitability Guard: shape → partition/fallback decisions")
    print(String(repeating: "-", count: 70))
    let guardShapes: [MatrixShape] = [
        .square(256),
        .square(512),
        .square(1024),
        .square(2048),
        .square(4096),
        MatrixShape(M: 512, K: 512, N: 4096),
        MatrixShape(M: 4096, K: 4096, N: 512),
        MatrixShape(M: 256, K: 256, N: 2048),
    ]
    let guard_ = harness.makeProfitabilityGuard()
    for gs in guardShapes {
        let decision = guard_.evaluate(op: .matmul, shape: gs)
        switch decision {
        case .partition(let desc):
            let splitType = desc.isColumnSplit ? "col" : "row"
            var fracs = ""
            for a in desc.assignments {
                fracs += "\(a.unit.rawValue)=\(Int(a.workFraction * 100))% "
            }
            print("  \(gs): PARTITION [\(splitType)] \(fracs)")
        case .fallback(let reason, let unit):
            print("  \(gs): FALLBACK → \(unit.rawValue) (\(reason))")
        }
    }
    print()

    // Run benchmarks
    var results: [ShapeBenchmarkResult] = []
    let totalStart = CACurrentMediaTime()

    for (idx, shape) in shapes.enumerated() {
        print("[\(idx + 1)/\(shapes.count)] Benchmarking \(shape)...")

        do {
            let result = try harness.benchmark(shape: shape)
            results.append(result)
            printShapeResult(result)
        } catch {
            print("  FAILED: \(error)")
        }

        print()
    }

    // Summary table
    printSummaryTable(results: results)

    // Decision gate
    printDecisionGate(results: results)

    // ── Column-Split Benchmark ──
    print()
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Column-Split vs Row-Split Comparison                        ║
    ║  Split dimension selection for non-square shapes             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    // Correctness check
    let colValShape = MatrixShape(M: 256, K: 256, N: 512)
    do {
        let colErr = try harness.validateColumnSplitCorrectness(shape: colValShape)
        let tolerance: Double = 1e-2
        if colErr < tolerance {
            print("  Column-split correctness: PASS (max error = \(String(format: "%.6f", colErr)))")
        } else {
            print("  Column-split correctness: FAIL (max error = \(String(format: "%.6f", colErr)))")
        }
    } catch {
        print("  Column-split correctness: ERROR (\(error))")
    }
    print()

    // Compare row-split vs column-split on various shapes
    let splitComparisonShapes: [(MatrixShape, String)] = [
        (MatrixShape(M: 512, K: 512, N: 4096),  "wide N>>M"),
        (MatrixShape(M: 4096, K: 4096, N: 512), "tall M>>N"),
        (.square(2048),                           "square"),
        (MatrixShape(M: 256, K: 256, N: 2048),  "very wide"),
    ]

    let splitFracs = (gpu: 0.34, mps: 0.33, cpu: 0.33)
    print("  Split comparison (gpu=\(Int(splitFracs.gpu*100))% mps=\(Int(splitFracs.mps*100))% cpu=\(Int(splitFracs.cpu*100))%):")
    print(String(repeating: "-", count: 80))
    print("  Shape                       Row (ms)      Col (ms)      Winner      Auto")
    print(String(repeating: "-", count: 80))

    for (splitShape, label) in splitComparisonShapes {
        do {
            let (rowMs, colMs) = try harness.benchmarkSplitComparison(
                shape: splitShape, fractions: splitFracs)
            let winner = colMs < rowMs ? "col" : "row"
            let autoChoice = splitShape.N > splitShape.M ? "col" : "row"
            let correct = winner == autoChoice ? "correct" : "MISMATCH"
            let shapeStr = "\(splitShape) (\(label))"
            print(String(format: "  %-28s%-14.3f%-14.3f%-12s%@",
                         (shapeStr as NSString).utf8String!,
                         rowMs, colMs,
                         (winner as NSString).utf8String!,
                         "\(autoChoice) (\(correct))"))
        } catch {
            print("  \(splitShape): ERROR (\(error))")
        }
    }
    print()

    // ── Attention Benchmark ──
    if !args.contains("--no-attention") {
        print()
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║  Heterogeneous Attention Benchmark                          ║
        ║  Q @ K^T → softmax → weights @ V  (per-head)               ║
        ╚══════════════════════════════════════════════════════════════╝
        """)

        let attentionShapes: [AttentionShape] = [
            AttentionShape(seqLen: 512, dK: 64),
            AttentionShape(seqLen: 1024, dK: 64),
            AttentionShape(seqLen: 2048, dK: 64),
            AttentionShape(seqLen: 4096, dK: 64),
            AttentionShape(seqLen: 2048, dK: 256),
            AttentionShape(seqLen: 4096, dK: 256),
        ]

        do {
            let attnExec = try AttentionExecutor(device: device)
            var attnResults: [AttentionBenchmarkResult] = []

            for (idx, attnShape) in attentionShapes.enumerated() {
                print("[\(idx + 1)/\(attentionShapes.count)] Attention \(attnShape)...")
                do {
                    let result = try attnExec.benchmark(shape: attnShape, config: config)
                    attnResults.append(result)
                    print(String(format: "  GPU:    %.3f ms", result.gpuOnlyMs))
                    print(String(format: "  MPS:    %.3f ms", result.mpsOnlyMs))
                    print(String(format: "  CPU:    %.3f ms", result.cpuOnlyMs))
                    print(String(format: "  Fused:  %.3f ms [gpu=%.0f%% mps=%.0f%% cpu=%.0f%%]",
                                 result.fusedMs,
                                 result.fusedFractions.gpu * 100,
                                 result.fusedFractions.mps * 100,
                                 result.fusedFractions.cpu * 100))
                    print(String(format: "  Best:   %.3f ms (%@) → speedup: %.3fx %@",
                                 result.bestSingleMs, result.bestSingleUnit.rawValue,
                                 result.speedup, result.speedup > 1.0 ? "✓" : "✗"))
                } catch {
                    print("  FAILED: \(error)")
                }
                print()
            }

            // Attention summary
            printAttentionSummary(results: attnResults)
        } catch {
            print("Attention setup failed: \(error)")
        }
    }

    // ── Phase 4: Graph Pipeline ──
    if !args.contains("--no-phase4") {
        print()
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║  Phase 4: Graph Pipeline                                     ║
        ║  EligibilityAnnotation → OpFusion → Partitioning → Schedule  ║
        ╚══════════════════════════════════════════════════════════════╝
        """)

        let pipeline = GraphPipeline(
            guard_: guard_,
            splitter: harness.splitter,
            concurrencyBudget: harness.monitor.concurrencyBudget
        )

        // Test 1: Attention block (QK^T → softmax → AV) as single cluster
        // Using dK=dV=512 (full model dim) so AV output (2048×512=1M) crosses profitability threshold
        print("End-to-end graph test: attention block (seqLen=2048, dModel=512)")
        print(String(repeating: "-", count: 70))
        do {
            var attnGraph = HeteroGraph.attentionBlock(seqLen: 2048, dK: 512, dV: 512)
            let plan = pipeline.run(&attnGraph)
            let diag = pipeline.diagnostics(attnGraph)
            print(diag)
            print("  Execution plan: \(plan.steps.count) steps (\(plan.fusedClusterCount) fused, \(plan.passthroughCount) passthrough)")

            // Verify attention fuses into 1 cluster
            let attnClusters = attnGraph.clusters.filter { $0.fusionKind == .attentionBlock }
            if attnClusters.count == 1 && attnClusters[0].nodeIDs.count == 3 {
                print("  Attention fusion: PASS (3 ops → 1 cluster)")
            } else {
                print("  Attention fusion: FAIL (expected 1 cluster with 3 ops, got \(attnClusters.count) clusters)")
            }

            // Verify partition descriptor exists for the cluster
            if let desc = attnClusters.first?.partitionDescriptor {
                let splitType = desc.isColumnSplit ? "col" : "row"
                var fracs = ""
                for a in desc.assignments {
                    fracs += "\(a.unit.rawValue)=\(Int(a.workFraction * 100))% "
                }
                print("  Partition: [\(splitType)] \(fracs)")
            } else {
                print("  Partition: NONE (no throughput curves — calibrate first)")
            }
        }
        print()

        // Test 2: Matmul chain (MLP-like: x@W1 → @W2 → @W3)
        print("Matmul chain test: 3-layer MLP (2048×2048 → 2048×4096 → 4096×2048)")
        print(String(repeating: "-", count: 70))
        do {
            var mlpGraph = HeteroGraph.matmulChain(shapes: [
                (MatrixShape(M: 2048, K: 2048, N: 2048), "fc1"),
                (MatrixShape(M: 2048, K: 2048, N: 4096), "fc2"),
                (MatrixShape(M: 2048, K: 4096, N: 2048), "fc3"),
            ])
            let plan = pipeline.run(&mlpGraph)
            let diag = pipeline.diagnostics(mlpGraph)
            print(diag)
            print("  Execution plan: \(plan.steps.count) steps (\(plan.fusedClusterCount) fused, \(plan.passthroughCount) passthrough)")

            // All matmuls should be eligible and fused into one chain
            let matmulClusters = mlpGraph.clusters.filter { $0.fusionKind == .matmulChain }
            if matmulClusters.count == 1 && matmulClusters[0].nodeIDs.count == 3 {
                print("  MLP fusion: PASS (3 matmuls → 1 chain cluster)")
            } else {
                print("  MLP fusion: got \(matmulClusters.count) chain clusters")
            }
        }
        print()

        // Test 3: Mixed graph (small fallback + large eligible)
        print("Mixed graph test: small fallback + large eligible ops")
        print(String(repeating: "-", count: 70))
        do {
            var mixedGraph = HeteroGraph()
            // Small op → falls below profitability threshold
            let small = mixedGraph.addNode(
                opcode: .matmul,
                outputShape: MatrixShape.square(256),
                name: "small_matmul"
            )
            // Large op → eligible for partitioning
            let large = mixedGraph.addNode(
                opcode: .matmul,
                outputShape: MatrixShape.square(2048),
                name: "large_matmul"
            )
            mixedGraph.addEdge(from: small, to: large)

            let plan = pipeline.run(&mixedGraph)
            let diag = pipeline.diagnostics(mixedGraph)
            print(diag)
            print("  Execution plan: \(plan.steps.count) steps (\(plan.fusedClusterCount) fused, \(plan.passthroughCount) passthrough)")

            // small should be fallback, large should be in a fused cluster
            let smallIsPassthrough = !small.isEligible
            let largeIsFused = large.clusterID != nil
            if smallIsPassthrough && largeIsFused {
                print("  Mixed classification: PASS (small=fallback, large=fused)")
            } else {
                print("  Mixed classification: small eligible=\(small.isEligible), large clusterID=\(String(describing: large.clusterID))")
            }
        }
        print()

        // Test 4: Serial cluster chain measurement
        // Validates that recalibration overhead is < 2% of cluster time
        print("Serial cluster chain: validate recalibration overhead")
        print(String(repeating: "-", count: 70))
        do {
            let chainShape = MatrixShape.square(2048)
            let clusterCount = 5

            // Measure single fused matmul time (baseline)
            let singleDesc = harness.splitter.optimalMatmulPartition(shape: chainShape)
            if let desc = singleDesc {
                let aBytes = chainShape.M * chainShape.K * MemoryLayout<Float>.size
                let bBytes = chainShape.K * chainShape.N * MemoryLayout<Float>.size
                let cBytes = chainShape.M * chainShape.N * MemoryLayout<Float>.size

                guard let bufA = device.makeBuffer(length: aBytes, options: .storageModeShared),
                      let bufB = device.makeBuffer(length: bBytes, options: .storageModeShared),
                      let bufC = device.makeBuffer(length: cBytes, options: .storageModeShared) else {
                    throw HeterogeneousError.bufferAllocationFailed("Serial chain buffer allocation failed")
                }

                let fused_ = try FusedMatmul(device: device)

                // Measure single fused
                var singleTimes: [Double] = []
                for _ in 0..<3 { _ = fused_.execute(descriptor: desc, A: bufA, B: bufB, C: bufC) }
                for _ in 0..<10 {
                    let start = CACurrentMediaTime()
                    _ = fused_.execute(descriptor: desc, A: bufA, B: bufB, C: bufC)
                    singleTimes.append((CACurrentMediaTime() - start) * 1000.0)
                }
                let singleMs = TimingSummary(samples: singleTimes, trimCount: 1).mean

                // Measure serial chain of N fused matmuls
                var chainTimes: [Double] = []
                for _ in 0..<3 {
                    for _ in 0..<clusterCount {
                        _ = fused_.execute(descriptor: desc, A: bufA, B: bufB, C: bufC)
                    }
                }
                for _ in 0..<10 {
                    let start = CACurrentMediaTime()
                    for _ in 0..<clusterCount {
                        _ = fused_.execute(descriptor: desc, A: bufA, B: bufB, C: bufC)
                    }
                    chainTimes.append((CACurrentMediaTime() - start) * 1000.0)
                }
                let chainMs = TimingSummary(samples: chainTimes, trimCount: 1).mean

                let expectedMs = singleMs * Double(clusterCount)
                let overhead = (chainMs - expectedMs) / expectedMs
                let overheadPct = overhead * 100.0

                print(String(format: "  Single fused:    %.3f ms", singleMs))
                print(String(format: "  Chain of %d:      %.3f ms (expected: %.3f ms)", clusterCount, chainMs, expectedMs))
                print(String(format: "  Serial overhead: %.2f%%", overheadPct))

                if abs(overheadPct) < 2.0 {
                    print("  Serial chain: PASS (overhead < 2%)")
                } else if overheadPct < 0 {
                    print("  Serial chain: PASS (chain is faster than sum — caching)")
                } else {
                    print(String(format: "  Serial chain: WARN (%.1f%% overhead — monitor recal triggers)", overheadPct))
                }
            } else {
                print("  Skipped: no throughput curves available (run calibration first)")
            }
        } catch {
            print("  Serial chain measurement ERROR: \(error)")
        }
        print()

        // Test 5: Regression — correctness through full pass pipeline
        print("Regression: correctness suite through full pipeline")
        print(String(repeating: "-", count: 70))
        do {
            let regressionShapes: [(MatrixShape, String)] = [
                (MatrixShape.square(256), "small"),
                (MatrixShape.square(1024), "medium"),
                (MatrixShape.square(2048), "large"),
                (MatrixShape(M: 512, K: 512, N: 4096), "wide"),
            ]

            var regressionPasses = 0
            for (shape, label) in regressionShapes {
                var graph = HeteroGraph()
                let node = graph.addNode(opcode: .matmul, outputShape: shape, name: "test_\(label)")
                _ = node // silence warning
                let plan = pipeline.run(&graph)

                let status: String
                if plan.fusedClusterCount > 0 {
                    status = "FUSED (\(graph.clusters[0].nodeIDs.count) ops)"
                } else {
                    status = "PASSTHROUGH"
                }

                // Check that pipeline decision matches profitability guard
                let directDecision = guard_.evaluate(op: .matmul, shape: shape)
                let pipelineDecision = node.partitionDecision

                var consistent = false
                switch (directDecision, pipelineDecision) {
                case (.partition, .partition): consistent = true
                case (.fallback, .fallback): consistent = true
                default: consistent = false
                }

                if consistent {
                    regressionPasses += 1
                    print("  \(shape) (\(label)): \(status) — consistent")
                } else {
                    print("  \(shape) (\(label)): \(status) — INCONSISTENT")
                }
            }

            print()
            if regressionPasses == regressionShapes.count {
                print("  Regression: PASS (\(regressionPasses)/\(regressionShapes.count) consistent)")
            } else {
                print("  Regression: FAIL (\(regressionPasses)/\(regressionShapes.count) consistent)")
            }
        }
        print()
    }

    // ── Phase 5: Elementwise Op Coverage ──
    if !args.contains("--no-elementwise") {
        print()
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║  Phase 5: Elementwise Op Coverage                            ║
        ║  add, mul, relu, gelu — row-split, cluster absorption        ║
        ╚══════════════════════════════════════════════════════════════╝
        """)

        do {
            let ewGPU = try ElementwiseGPU(device: device)
            let ewCPU = ElementwiseCPU()
            let executor = try FusedExecutor(device: device)

            // Correctness: compare fused vs GPU-only for each op
            print("Elementwise correctness: fused vs GPU-only")
            print(String(repeating: "-", count: 70))

            let ewTestShape = (rows: 1024, cols: 1024)
            let ewCount = ewTestShape.rows * ewTestShape.cols
            let ewBytes = ewCount * MemoryLayout<Float>.size

            guard let ewA = device.makeBuffer(length: ewBytes, options: .storageModeShared),
                  let ewB = device.makeBuffer(length: ewBytes, options: .storageModeShared),
                  let ewRef = device.makeBuffer(length: ewBytes, options: .storageModeShared),
                  let ewTest = device.makeBuffer(length: ewBytes, options: .storageModeShared) else {
                throw HeterogeneousError.bufferAllocationFailed("Elementwise buffers")
            }

            // Fill with data
            let aPtr = ewA.contents().assumingMemoryBound(to: Float.self)
            let bPtr = ewB.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<ewCount {
                aPtr[i] = Float.random(in: -1.0..<1.0)
                bPtr[i] = Float.random(in: -1.0..<1.0)
            }

            let ewOps: [(ElementwiseOp, HLOOpCode)] = [
                (.add, .elementwiseAdd),
                (.mul, .elementwiseMul),
                (.relu, .elementwiseRelu),
                (.gelu, .gelu),
            ]

            var ewCorrectnessPass = 0
            for (ewOp, opcode) in ewOps {
                // Reference: GPU-only
                _ = ewGPU.execute(op: ewOp, A: ewA, B: ewOp.isBinary ? ewB : nil, C: ewRef, count: ewCount)

                // Clear test buffer
                memset(ewTest.contents(), 0, ewBytes)

                // Test: fused 3-way
                let shape = MatrixShape(M: ewTestShape.rows, K: ewTestShape.cols, N: ewTestShape.cols)
                let desc = PartitionDescriptor.elementwise(
                    op: opcode, rows: ewTestShape.rows, cols: ewTestShape.cols,
                    fractions: [(.gpu, 0.65), (.cpu, 0.35)]
                )
                var inputs = [ewA]
                if ewOp.isBinary { inputs.append(ewB) }
                _ = executor.execute(descriptor: desc, inputBuffers: inputs, outputBuffer: ewTest)

                // Compare
                let refP = ewRef.contents().assumingMemoryBound(to: Float.self)
                let testP = ewTest.contents().assumingMemoryBound(to: Float.self)
                var maxErr: Double = 0
                for i in 0..<ewCount {
                    maxErr = Swift.max(maxErr, abs(Double(refP[i]) - Double(testP[i])))
                }

                let tolerance: Double = 1e-4
                let pass = maxErr < tolerance
                if pass { ewCorrectnessPass += 1 }
                print("  \(ewOp.rawValue): \(pass ? "PASS" : "FAIL") (maxErr=\(String(format: "%.6f", maxErr)))")
                _ = shape  // use shape variable
            }
            print("  Elementwise correctness: \(ewCorrectnessPass)/\(ewOps.count)")
            print()

            // Benchmark: single-unit vs fused for each op
            print("Elementwise benchmark: single-unit vs fused")
            print(String(repeating: "-", count: 70))

            let benchShapes: [(Int, Int, String)] = [
                (1024, 1024, "1024x1024"),
                (1024, 4096, "1024x4096 (FFN)"),
                (2048, 2048, "2048x2048"),
            ]

            for (rows, cols, label) in benchShapes {
                let count = rows * cols
                let bytes = count * MemoryLayout<Float>.size

                guard let bA = device.makeBuffer(length: bytes, options: .storageModeShared),
                      let bB = device.makeBuffer(length: bytes, options: .storageModeShared),
                      let bC = device.makeBuffer(length: bytes, options: .storageModeShared) else { continue }

                let p = bA.contents().assumingMemoryBound(to: Float.self)
                let q = bB.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<count { p[i] = Float.random(in: -1..<1); q[i] = Float.random(in: -1..<1) }

                // Test with GELU (most compute-heavy elementwise)
                // GPU baseline
                for _ in 0..<3 { _ = ewGPU.execute(op: .gelu, A: bA, B: nil, C: bC, count: count) }
                var gpuTimes: [Double] = []
                for _ in 0..<10 {
                    let t = ewGPU.execute(op: .gelu, A: bA, B: nil, C: bC, count: count)
                    gpuTimes.append(t * 1000.0)
                }
                let gpuMs = TimingSummary(samples: gpuTimes, trimCount: 1).mean

                // CPU baseline
                let cpuA = bA.contents().assumingMemoryBound(to: Float.self)
                let cpuC = bC.contents().assumingMemoryBound(to: Float.self)
                for _ in 0..<3 { ewCPU.execute(op: .gelu, A: cpuA, B: nil, C: cpuC, count: count) }
                var cpuTimes: [Double] = []
                for _ in 0..<10 {
                    let start = CACurrentMediaTime()
                    ewCPU.execute(op: .gelu, A: cpuA, B: nil, C: cpuC, count: count)
                    cpuTimes.append((CACurrentMediaTime() - start) * 1000.0)
                }
                let cpuMs = TimingSummary(samples: cpuTimes, trimCount: 1).mean

                // Fused
                let desc = PartitionDescriptor.elementwise(
                    op: .gelu, rows: rows, cols: cols,
                    fractions: [(.gpu, 0.65), (.cpu, 0.35)]
                )
                for _ in 0..<3 { _ = executor.execute(descriptor: desc, inputBuffers: [bA], outputBuffer: bC) }
                var fusedTimes: [Double] = []
                for _ in 0..<10 {
                    let start = CACurrentMediaTime()
                    _ = executor.execute(descriptor: desc, inputBuffers: [bA], outputBuffer: bC)
                    fusedTimes.append((CACurrentMediaTime() - start) * 1000.0)
                }
                let fusedMs = TimingSummary(samples: fusedTimes, trimCount: 1).mean

                let bestSingle = Swift.min(gpuMs, cpuMs)
                let speedup = bestSingle / fusedMs

                print(String(format: "  %@ GELU: GPU=%.3fms CPU=%.3fms Fused=%.3fms → %.2fx %@",
                             label, gpuMs, cpuMs, fusedMs, speedup, speedup > 1.0 ? "win" : "loss"))
            }
            print()

            // Cluster absorption test: matmul → add → matmul
            // Should fuse into 1 cluster (3 ops, 1 sync point) instead of 3 separate steps
            print("Cluster absorption: matmul → add → matmul")
            print(String(repeating: "-", count: 70))
            do {
                var graph = HeteroGraph()
                let mm1 = graph.addNode(
                    opcode: .matmul,
                    outputShape: MatrixShape.square(2048),
                    name: "fc1"
                )
                let residual = graph.addNode(
                    opcode: .elementwiseAdd,
                    outputShape: MatrixShape(M: 2048, K: 2048, N: 2048),
                    name: "residual_add"
                )
                let mm2 = graph.addNode(
                    opcode: .matmul,
                    outputShape: MatrixShape.square(2048),
                    name: "fc2"
                )
                graph.addEdge(from: mm1, to: residual)
                graph.addEdge(from: residual, to: mm2)

                let pipeline = GraphPipeline(
                    guard_: guard_,
                    splitter: harness.splitter,
                    concurrencyBudget: harness.monitor.concurrencyBudget
                )
                let plan = pipeline.run(&graph)
                let diag = pipeline.diagnostics(graph)
                print(diag)
                print("  Execution plan: \(plan.steps.count) steps (\(plan.fusedClusterCount) fused, \(plan.passthroughCount) passthrough)")

                // All 3 ops should be in 1 cluster
                if graph.clusters.count == 1 && graph.clusters[0].nodeIDs.count == 3 {
                    print("  Cluster absorption: PASS (matmul → add → matmul = 1 cluster, 1 sync point)")
                } else {
                    print("  Cluster absorption: \(graph.clusters.count) clusters (expected 1)")
                    for c in graph.clusters {
                        print("    cluster \(c.id): \(c.nodeIDs.count) ops, kind=\(c.fusionKind.rawValue)")
                    }
                }
            }
            print()

        } catch {
            print("  Elementwise setup failed: \(error)")
        }
    }

    // ── Phase 5: LayerNorm Op Coverage ──
    if !args.contains("--no-layernorm") {
        print()
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║  Phase 5: LayerNorm Op Coverage                              ║
        ║  Per-row normalization — row-split, no recombine             ║
        ╚══════════════════════════════════════════════════════════════╝
        """)

        do {
            let lnGPU = try LayerNormGPU(device: device)
            let lnCPU = LayerNormCPU()
            let executor = try FusedExecutor(device: device)

            // ── Correctness: fused vs GPU-only for each shape ──
            print("LayerNorm correctness: fused vs GPU-only")
            print(String(repeating: "-", count: 70))

            let correctnessShapes: [(M: Int, N: Int, String)] = [
                (512, 768, "BERT-base"),
                (1024, 1024, "GPT-2 small"),
                (2048, 2048, "large square"),
                (256, 512, "boundary"),
                (128, 768, "small batch"),
                (1024, 2048, "wide — Option B kernel"),
            ]

            var lnCorrectnessPass = 0
            for (M, N, label) in correctnessShapes {
                let count = M * N
                let inputBytes = count * MemoryLayout<Float>.size
                let paramBytes = N * MemoryLayout<Float>.size

                guard let bufIn = device.makeBuffer(length: inputBytes, options: .storageModeShared),
                      let bufGamma = device.makeBuffer(length: paramBytes, options: .storageModeShared),
                      let bufBeta = device.makeBuffer(length: paramBytes, options: .storageModeShared),
                      let bufRef = device.makeBuffer(length: inputBytes, options: .storageModeShared),
                      let bufTest = device.makeBuffer(length: inputBytes, options: .storageModeShared) else { continue }

                // Fill with data
                let inPtr = bufIn.contents().assumingMemoryBound(to: Float.self)
                let gammaPtr = bufGamma.contents().assumingMemoryBound(to: Float.self)
                let betaPtr = bufBeta.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<count { inPtr[i] = Float.random(in: -2.0..<2.0) }
                for i in 0..<N { gammaPtr[i] = Float.random(in: 0.5..<1.5) }
                for i in 0..<N { betaPtr[i] = Float.random(in: -0.5..<0.5) }

                // Reference: GPU-only
                _ = lnGPU.execute(input: bufIn, gamma: bufGamma, beta: bufBeta, output: bufRef,
                                  rowCount: M, N: N)

                // Clear test buffer
                memset(bufTest.contents(), 0, inputBytes)

                // Check if this shape should use fused or fallback
                let shape = MatrixShape(M: M, K: N, N: N)
                let decision = guard_.evaluate(op: .layerNorm, shape: shape)

                switch decision {
                case .partition:
                    // Fused 3-way
                    let desc = PartitionDescriptor.layerNorm(
                        M: M, N: N,
                        fractions: [(.gpu, 0.65), (.cpu, 0.35)]
                    )
                    _ = executor.execute(
                        descriptor: desc,
                        inputBuffers: [bufIn, bufGamma, bufBeta],
                        outputBuffer: bufTest
                    )
                case .fallback:
                    // Fallback: GPU-only
                    _ = lnGPU.execute(input: bufIn, gamma: bufGamma, beta: bufBeta, output: bufTest,
                                      rowCount: M, N: N)
                }

                // Compare
                let refP = bufRef.contents().assumingMemoryBound(to: Float.self)
                let testP = bufTest.contents().assumingMemoryBound(to: Float.self)
                var maxErr: Double = 0
                for i in 0..<count {
                    maxErr = Swift.max(maxErr, abs(Double(refP[i]) - Double(testP[i])))
                }

                let tolerance: Double = 1e-4
                let pass = maxErr < tolerance
                if pass { lnCorrectnessPass += 1 }

                let mode: String
                if case .partition = decision { mode = "FUSED" } else { mode = "FALLBACK" }
                print("  \(M)×\(N) (\(label)): \(pass ? "PASS" : "FAIL") (maxErr=\(String(format: "%.6f", maxErr)), \(mode))")
            }
            print("  LayerNorm correctness: \(lnCorrectnessPass)/\(correctnessShapes.count)")
            print()

            // ── Profitability gate verification ──
            print("LayerNorm profitability gate")
            print(String(repeating: "-", count: 70))

            let gateTests: [(M: Int, N: Int, String, Bool)] = [
                (256, 512, "boundary", false),    // M=256, on boundary — may or may not pass
                (128, 768, "small batch", false),  // M < 256 → fallback
                (512, 768, "BERT-base", true),
                (1024, 1024, "GPT-2 small", true),
                (2048, 2048, "large", true),
                (512, 256, "narrow hidden", false),  // N < 512 → fallback
            ]

            for (M, N, label, expectedPartition) in gateTests {
                let shape = MatrixShape(M: M, K: N, N: N)
                let decision = guard_.evaluate(op: .layerNorm, shape: shape)
                let isPartition: Bool
                if case .partition = decision { isPartition = true } else { isPartition = false }
                let match = isPartition == expectedPartition
                let reason: String
                if case .fallback(let r, _) = decision { reason = r } else { reason = "partition" }
                print("  \(M)×\(N) (\(label)): \(match ? "PASS" : "WARN") — \(reason)")
            }
            print()

            // ── Benchmark: single-unit vs fused ──
            print("LayerNorm benchmark: single-unit vs fused")
            print(String(repeating: "-", count: 70))

            let benchShapes: [(M: Int, N: Int, String)] = [
                (512, 768, "512×768 (BERT)"),
                (512, 1024, "512×1024 (GPT-2)"),
                (1024, 1024, "1024×1024"),
                (1024, 2048, "1024×2048 (wide)"),
                (2048, 2048, "2048×2048"),
                (2048, 4096, "2048×4096 (large)"),
            ]

            for (M, N, label) in benchShapes {
                let count = M * N
                let inputBytes = count * MemoryLayout<Float>.size
                let paramBytes = N * MemoryLayout<Float>.size

                guard let bIn = device.makeBuffer(length: inputBytes, options: .storageModeShared),
                      let bGamma = device.makeBuffer(length: paramBytes, options: .storageModeShared),
                      let bBeta = device.makeBuffer(length: paramBytes, options: .storageModeShared),
                      let bOut = device.makeBuffer(length: inputBytes, options: .storageModeShared) else { continue }

                let p = bIn.contents().assumingMemoryBound(to: Float.self)
                let g = bGamma.contents().assumingMemoryBound(to: Float.self)
                let b = bBeta.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<count { p[i] = Float.random(in: -2..<2) }
                for i in 0..<N { g[i] = Float.random(in: 0.5..<1.5) }
                for i in 0..<N { b[i] = Float.random(in: -0.5..<0.5) }

                // GPU baseline
                for _ in 0..<3 { _ = lnGPU.execute(input: bIn, gamma: bGamma, beta: bBeta, output: bOut, rowCount: M, N: N) }
                var gpuTimes: [Double] = []
                for _ in 0..<20 {
                    let t = lnGPU.execute(input: bIn, gamma: bGamma, beta: bBeta, output: bOut, rowCount: M, N: N)
                    gpuTimes.append(t * 1000.0)
                }
                let gpuMs = TimingSummary(samples: gpuTimes, trimCount: 2).mean

                // CPU baseline
                let cpuIn = bIn.contents().assumingMemoryBound(to: Float.self)
                let cpuGamma = bGamma.contents().assumingMemoryBound(to: Float.self)
                let cpuBeta = bBeta.contents().assumingMemoryBound(to: Float.self)
                let cpuOut = bOut.contents().assumingMemoryBound(to: Float.self)
                for _ in 0..<3 { lnCPU.execute(input: cpuIn, gamma: cpuGamma, beta: cpuBeta, output: cpuOut, M: M, N: N, epsilon: 1e-5) }
                var cpuTimes: [Double] = []
                for _ in 0..<20 {
                    let start = CACurrentMediaTime()
                    lnCPU.execute(input: cpuIn, gamma: cpuGamma, beta: cpuBeta, output: cpuOut, M: M, N: N, epsilon: 1e-5)
                    cpuTimes.append((CACurrentMediaTime() - start) * 1000.0)
                }
                let cpuMs = TimingSummary(samples: cpuTimes, trimCount: 2).mean

                // Fused 3-way
                let desc = PartitionDescriptor.layerNorm(
                    M: M, N: N,
                    fractions: [(.gpu, 0.65), (.cpu, 0.35)]
                )
                for _ in 0..<3 { _ = executor.execute(descriptor: desc, inputBuffers: [bIn, bGamma, bBeta], outputBuffer: bOut) }
                var fusedTimes: [Double] = []
                for _ in 0..<20 {
                    let start = CACurrentMediaTime()
                    _ = executor.execute(descriptor: desc, inputBuffers: [bIn, bGamma, bBeta], outputBuffer: bOut)
                    fusedTimes.append((CACurrentMediaTime() - start) * 1000.0)
                }
                let fusedMs = TimingSummary(samples: fusedTimes, trimCount: 2).mean

                let bestSingle = Swift.min(gpuMs, cpuMs)
                let speedup = bestSingle / fusedMs

                print(String(format: "  %@ : GPU=%.3fms CPU=%.3fms Fused=%.3fms → %.2fx %@",
                             label, gpuMs, cpuMs, fusedMs, speedup, speedup > 1.0 ? "win" : "loss"))
            }
            print()

            // ── Cluster absorption: matmul → layerNorm → matmul ──
            print("Cluster absorption: matmul → layerNorm → matmul")
            print(String(repeating: "-", count: 70))
            do {
                var graph = HeteroGraph()
                let mm1 = graph.addNode(
                    opcode: .matmul,
                    outputShape: MatrixShape.square(2048),
                    name: "fc1"
                )
                let ln = graph.addNode(
                    opcode: .layerNorm,
                    outputShape: MatrixShape(M: 2048, K: 2048, N: 2048),
                    name: "layer_norm"
                )
                let mm2 = graph.addNode(
                    opcode: .matmul,
                    outputShape: MatrixShape.square(2048),
                    name: "fc2"
                )
                graph.addEdge(from: mm1, to: ln)
                graph.addEdge(from: ln, to: mm2)

                let pipeline = GraphPipeline(
                    guard_: guard_,
                    splitter: harness.splitter,
                    concurrencyBudget: harness.monitor.concurrencyBudget
                )
                let plan = pipeline.run(&graph)
                let diag = pipeline.diagnostics(graph)
                print(diag)
                print("  Execution plan: \(plan.steps.count) steps (\(plan.fusedClusterCount) fused, \(plan.passthroughCount) passthrough)")

                if graph.clusters.count == 1 && graph.clusters[0].nodeIDs.count == 3 {
                    print("  Cluster absorption: PASS (matmul → layerNorm → matmul = 1 cluster, 1 sync point)")
                } else {
                    print("  Cluster absorption: \(graph.clusters.count) clusters (expected 1)")
                    for c in graph.clusters {
                        print("    cluster \(c.id): \(c.nodeIDs.count) ops, kind=\(c.fusionKind.rawValue)")
                    }
                }
            }
            print()

        } catch {
            print("  LayerNorm setup failed: \(error)")
        }
    }

    // ── Phase 5: Embedding Lookup Op Coverage ──
    if !args.contains("--no-embedding") {
        print()
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║  Phase 5: Embedding Lookup Op Coverage                       ║
        ║  Vocab-range split — scatter-free output, pre-zeroed buffer  ║
        ╚══════════════════════════════════════════════════════════════╝
        """)

        do {
            let embGPU = try EmbeddingGPU(device: device)
            let embCPU = EmbeddingCPU()
            let executor = try FusedExecutor(device: device)

            // ── Correctness: exact match (tolerance = 0.0) ──
            print("Embedding correctness: fused vs GPU-only (exact match)")
            print(String(repeating: "-", count: 70))

            let correctnessShapes: [(V: Int, S: Int, D: Int, String)] = [
                (32000, 128, 768,  "GPT-2"),         // standard transformer
                (50257, 256, 1024, "GPT-2 large"),    // real GPT-2 vocab
                (8192,  64,  256,  "small boundary"), // boundary case
                (16384, 512, 512,  "medium"),
                (4096,  128, 256,  "fallback small"), // vocabSize < 8192 → fallback
                (32000, 32,  768,  "fallback short"), // seqLen < 64 → fallback
            ]

            var embCorrectnessPass = 0
            for (V, S, D, label) in correctnessShapes {
                let tableBytes = V * D * MemoryLayout<Float>.size
                let tokenBytes = S * MemoryLayout<Int32>.size
                let outputBytes = S * D * MemoryLayout<Float>.size

                guard let bufTokens = device.makeBuffer(length: tokenBytes, options: .storageModeShared),
                      let bufTable = device.makeBuffer(length: tableBytes, options: .storageModeShared),
                      let bufRef = device.makeBuffer(length: outputBytes, options: .storageModeShared),
                      let bufTest = device.makeBuffer(length: outputBytes, options: .storageModeShared) else { continue }

                // Fill table with random data
                let tableP = bufTable.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<(V * D) { tableP[i] = Float.random(in: -1..<1) }

                // Fill token IDs with random values in [0, V)
                let tokenP = bufTokens.contents().assumingMemoryBound(to: Int32.self)
                for i in 0..<S { tokenP[i] = Int32.random(in: 0..<Int32(V)) }

                // Reference: GPU-only (single unit, full vocab)
                memset(bufRef.contents(), 0, outputBytes)
                _ = embGPU.execute(tokenIds: bufTokens, table: bufTable, output: bufRef,
                                   seqLen: S, embeddingDim: D, vocabSize: V)

                // Clear test buffer
                memset(bufTest.contents(), 0, outputBytes)

                // Check profitability gate
                let shape = MatrixShape(M: V, K: D, N: S)
                let decision = guard_.evaluate(op: .embeddingLookup, shape: shape)

                switch decision {
                case .partition:
                    let desc = PartitionDescriptor.embeddingLookup(
                        vocabSize: V, seqLen: S, embeddingDim: D,
                        fractions: [(.gpu, 0.60), (.cpu, 0.40)]
                    )
                    _ = executor.execute(
                        descriptor: desc,
                        inputBuffers: [bufTokens, bufTable],
                        outputBuffer: bufTest
                    )
                case .fallback:
                    memset(bufTest.contents(), 0, outputBytes)
                    _ = embGPU.execute(tokenIds: bufTokens, table: bufTable, output: bufTest,
                                       seqLen: S, embeddingDim: D, vocabSize: V)
                }

                // Compare (exact match — no floating point arithmetic)
                let refP = bufRef.contents().assumingMemoryBound(to: Float.self)
                let testP = bufTest.contents().assumingMemoryBound(to: Float.self)
                var maxErr: Double = 0
                for i in 0..<(S * D) {
                    maxErr = Swift.max(maxErr, abs(Double(refP[i]) - Double(testP[i])))
                }

                let tolerance: Double = 0.0  // exact match — integer-indexed memcpy
                let pass = maxErr <= tolerance
                if pass { embCorrectnessPass += 1 }

                let mode: String
                if case .partition = decision { mode = "FUSED" } else { mode = "FALLBACK" }
                print("  V=\(V) S=\(S) D=\(D) (\(label)): \(pass ? "PASS" : "FAIL") (maxErr=\(String(format: "%.6f", maxErr)), \(mode))")
            }
            print("  Embedding correctness: \(embCorrectnessPass)/\(correctnessShapes.count)")
            print()

            // ── Profitability gate verification ──
            print("Embedding profitability gate")
            print(String(repeating: "-", count: 70))

            let gateTests: [(V: Int, S: Int, D: Int, String, Bool)] = [
                (8192,  64,  256,  "boundary",       true),   // minimum sizes
                (4096,  128, 256,  "small vocab",     false),  // V < 8192
                (8192,  32,  256,  "short seq",       false),  // S < 64
                (8192,  64,  128,  "narrow dim",      false),  // D < 256
                (32000, 128, 768,  "GPT-2",           true),
                (50257, 512, 1024, "GPT-2 large",     true),
            ]

            for (V, S, D, label, expectedPartition) in gateTests {
                let shape = MatrixShape(M: V, K: D, N: S)
                let decision = guard_.evaluate(op: .embeddingLookup, shape: shape)
                let isPartition: Bool
                if case .partition = decision { isPartition = true } else { isPartition = false }
                let match = isPartition == expectedPartition
                let reason: String
                if case .fallback(let r, _) = decision { reason = r } else { reason = "partition" }
                print("  V=\(V) S=\(S) D=\(D) (\(label)): \(match ? "PASS" : "WARN") — \(reason)")
            }
            print()

            // ── GPU/CPU crossover measurement ──
            // Measures the seqLen threshold where GPU overtakes cblas_scopy.
            // This should be baked into ProfitabilityGuard as a measured constant.
            print("Embedding GPU/CPU crossover: find seqLen where GPU < CPU")
            print(String(repeating: "-", count: 70))
            do {
                let crossoverV = 32000
                let crossoverD = 768
                let seqLens = [16, 32, 64, 128, 256, 512, 1024, 2048]

                var crossoverSeqLen: Int? = nil

                for S in seqLens {
                    let tableBytes = crossoverV * crossoverD * MemoryLayout<Float>.size
                    let tokenBytes = S * MemoryLayout<Int32>.size
                    let outputBytes = S * crossoverD * MemoryLayout<Float>.size

                    guard let bTok = device.makeBuffer(length: tokenBytes, options: .storageModeShared),
                          let bTab = device.makeBuffer(length: tableBytes, options: .storageModeShared),
                          let bOut = device.makeBuffer(length: outputBytes, options: .storageModeShared) else { continue }

                    let tP = bTab.contents().assumingMemoryBound(to: Float.self)
                    for i in 0..<(crossoverV * crossoverD) { tP[i] = Float.random(in: -1..<1) }
                    let tokP = bTok.contents().assumingMemoryBound(to: Int32.self)
                    for i in 0..<S { tokP[i] = Int32.random(in: 0..<Int32(crossoverV)) }

                    // GPU
                    for _ in 0..<3 {
                        memset(bOut.contents(), 0, outputBytes)
                        _ = embGPU.execute(tokenIds: bTok, table: bTab, output: bOut,
                                           seqLen: S, embeddingDim: crossoverD, vocabSize: crossoverV)
                    }
                    var gpuTimes: [Double] = []
                    for _ in 0..<20 {
                        memset(bOut.contents(), 0, outputBytes)
                        let t = embGPU.execute(tokenIds: bTok, table: bTab, output: bOut,
                                               seqLen: S, embeddingDim: crossoverD, vocabSize: crossoverV)
                        gpuTimes.append(t * 1000.0)
                    }
                    let gpuMs = TimingSummary(samples: gpuTimes, trimCount: 2).mean

                    // CPU
                    let cpuTok = bTok.contents().assumingMemoryBound(to: Int32.self)
                    let cpuTab = bTab.contents().assumingMemoryBound(to: Float.self)
                    let cpuOut = bOut.contents().assumingMemoryBound(to: Float.self)
                    for _ in 0..<3 {
                        memset(bOut.contents(), 0, outputBytes)
                        embCPU.execute(tokenIds: cpuTok, tableShard: cpuTab, output: cpuOut,
                                       seqLen: S, embeddingDim: crossoverD, vocabOffset: 0, vocabSize: crossoverV)
                    }
                    var cpuTimes: [Double] = []
                    for _ in 0..<20 {
                        memset(bOut.contents(), 0, outputBytes)
                        let start = CACurrentMediaTime()
                        embCPU.execute(tokenIds: cpuTok, tableShard: cpuTab, output: cpuOut,
                                       seqLen: S, embeddingDim: crossoverD, vocabOffset: 0, vocabSize: crossoverV)
                        cpuTimes.append((CACurrentMediaTime() - start) * 1000.0)
                    }
                    let cpuMs = TimingSummary(samples: cpuTimes, trimCount: 2).mean

                    let winner = gpuMs < cpuMs ? "GPU" : "CPU"
                    print(String(format: "  seqLen=%4d : GPU=%.3fms CPU=%.3fms → %@ wins (%.2fx)",
                                 S, gpuMs, cpuMs, winner, cpuMs / gpuMs))

                    if gpuMs < cpuMs && crossoverSeqLen == nil {
                        crossoverSeqLen = S
                    }
                }

                if let cross = crossoverSeqLen {
                    print("  Crossover: GPU wins at seqLen ≥ \(cross) (V=\(crossoverV), D=\(crossoverD))")
                } else {
                    print("  Crossover: CPU dominates at all measured seqLens")
                }
            }
            print()

            // ── Benchmark: single-unit vs fused ──
            print("Embedding benchmark: single-unit vs fused")
            print(String(repeating: "-", count: 70))

            let benchShapes: [(V: Int, S: Int, D: Int, String)] = [
                (8192,  128, 256,  "8K×128×256 (small)"),
                (32000, 128, 768,  "32K×128×768 (GPT-2)"),
                (32000, 256, 768,  "32K×256×768 (GPT-2 long)"),
                (50257, 256, 1024, "50K×256×1024 (large)"),
                (50257, 512, 1024, "50K×512×1024 (large long)"),
                (65536, 512, 1024, "64K×512×1024 (xlarge)"),
            ]

            for (V, S, D, label) in benchShapes {
                let tableBytes = V * D * MemoryLayout<Float>.size
                let tokenBytes = S * MemoryLayout<Int32>.size
                let outputBytes = S * D * MemoryLayout<Float>.size

                guard let bTokens = device.makeBuffer(length: tokenBytes, options: .storageModeShared),
                      let bTable = device.makeBuffer(length: tableBytes, options: .storageModeShared),
                      let bOut = device.makeBuffer(length: outputBytes, options: .storageModeShared) else { continue }

                let tP = bTable.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<(V * D) { tP[i] = Float.random(in: -1..<1) }
                let tokP = bTokens.contents().assumingMemoryBound(to: Int32.self)
                for i in 0..<S { tokP[i] = Int32.random(in: 0..<Int32(V)) }

                // GPU baseline
                for _ in 0..<3 {
                    memset(bOut.contents(), 0, outputBytes)
                    _ = embGPU.execute(tokenIds: bTokens, table: bTable, output: bOut,
                                       seqLen: S, embeddingDim: D, vocabSize: V)
                }
                var gpuTimes: [Double] = []
                for _ in 0..<20 {
                    memset(bOut.contents(), 0, outputBytes)
                    let t = embGPU.execute(tokenIds: bTokens, table: bTable, output: bOut,
                                           seqLen: S, embeddingDim: D, vocabSize: V)
                    gpuTimes.append(t * 1000.0)
                }
                let gpuMs = TimingSummary(samples: gpuTimes, trimCount: 2).mean

                // CPU baseline
                let cpuTokens = bTokens.contents().assumingMemoryBound(to: Int32.self)
                let cpuTable = bTable.contents().assumingMemoryBound(to: Float.self)
                let cpuOut = bOut.contents().assumingMemoryBound(to: Float.self)
                for _ in 0..<3 {
                    memset(bOut.contents(), 0, outputBytes)
                    embCPU.execute(tokenIds: cpuTokens, tableShard: cpuTable, output: cpuOut,
                                   seqLen: S, embeddingDim: D, vocabOffset: 0, vocabSize: V)
                }
                var cpuTimes: [Double] = []
                for _ in 0..<20 {
                    memset(bOut.contents(), 0, outputBytes)
                    let start = CACurrentMediaTime()
                    embCPU.execute(tokenIds: cpuTokens, tableShard: cpuTable, output: cpuOut,
                                   seqLen: S, embeddingDim: D, vocabOffset: 0, vocabSize: V)
                    cpuTimes.append((CACurrentMediaTime() - start) * 1000.0)
                }
                let cpuMs = TimingSummary(samples: cpuTimes, trimCount: 2).mean

                // Fused 3-way
                let desc = PartitionDescriptor.embeddingLookup(
                    vocabSize: V, seqLen: S, embeddingDim: D,
                    fractions: [(.gpu, 0.60), (.cpu, 0.40)]
                )
                for _ in 0..<3 {
                    _ = executor.execute(descriptor: desc, inputBuffers: [bTokens, bTable], outputBuffer: bOut)
                }
                var fusedTimes: [Double] = []
                for _ in 0..<20 {
                    let start = CACurrentMediaTime()
                    _ = executor.execute(descriptor: desc, inputBuffers: [bTokens, bTable], outputBuffer: bOut)
                    fusedTimes.append((CACurrentMediaTime() - start) * 1000.0)
                }
                let fusedMs = TimingSummary(samples: fusedTimes, trimCount: 2).mean

                let bestSingle = Swift.min(gpuMs, cpuMs)
                let speedup = bestSingle / fusedMs

                print(String(format: "  %@ : GPU=%.3fms CPU=%.3fms Fused=%.3fms → %.2fx %@",
                             label, gpuMs, cpuMs, fusedMs, speedup, speedup > 1.0 ? "win" : "loss"))
            }
            print()

        } catch {
            print("  Embedding setup failed: \(error)")
        }
    }

    // ── Phase 5: Convolution Op Coverage ──
    if !args.contains("--no-convolution") {
        print()
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║  Phase 5: Convolution Op Coverage                            ║
        ║  3-unit compute-bound — ANE's home territory                 ║
        ║  Output-channel split: GPU + MPS/ANE + CPU                   ║
        ╚══════════════════════════════════════════════════════════════╝
        """)

        do {
            let convGPU = try ConvolutionGPU(device: device)
            let convMPS = try ConvolutionMPS(device: device)
            let convCPU = ConvolutionCPU()
            let executor = try FusedExecutor(device: device)

            // ── Correctness: fused vs GPU-only ──
            print("Convolution correctness: fused vs GPU-only")
            print(String(repeating: "-", count: 70))

            let correctnessShapes: [(ConvShape, String)] = [
                (ConvShape(batch: 1, inChannels: 64, outChannels: 128,
                           height: 28, width: 28, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "64→128 28×28 3×3 pad1"),
                (ConvShape(batch: 1, inChannels: 128, outChannels: 256,
                           height: 14, width: 14, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "128→256 14×14 3×3 pad1"),
                (ConvShape(batch: 1, inChannels: 256, outChannels: 256,
                           height: 14, width: 14, kernelH: 1, kernelW: 1),
                 "256→256 14×14 1×1 (pointwise)"),
                (ConvShape(batch: 1, inChannels: 3, outChannels: 64,
                           height: 56, width: 56, kernelH: 7, kernelW: 7, padH: 3, padW: 3),
                 "3→64 56×56 7×7 pad3 (ResNet stem)"),
                (ConvShape(batch: 1, inChannels: 128, outChannels: 128,
                           height: 28, width: 28, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "128→128 28×28 3×3 pad1"),
                (ConvShape(batch: 1, inChannels: 256, outChannels: 512,
                           height: 7, width: 7, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "256→512 7×7 3×3 pad1 (ResNet late)"),
            ]

            var convCorrectnessPass = 0
            for (shape, label) in correctnessShapes {
                guard let bufInput = device.makeBuffer(length: shape.inputBytes, options: .storageModeShared),
                      let bufWeights = device.makeBuffer(length: shape.weightBytes, options: .storageModeShared),
                      let bufRef = device.makeBuffer(length: shape.outputBytes, options: .storageModeShared),
                      let bufTest = device.makeBuffer(length: shape.outputBytes, options: .storageModeShared) else { continue }

                // Fill with small random values
                let inP = bufInput.contents().assumingMemoryBound(to: Float.self)
                let wP = bufWeights.contents().assumingMemoryBound(to: Float.self)
                let inCount = shape.inputBytes / MemoryLayout<Float>.size
                let wCount = shape.weightBytes / MemoryLayout<Float>.size
                for i in 0..<inCount { inP[i] = Float.random(in: -0.5..<0.5) }
                for i in 0..<wCount { wP[i] = Float.random(in: -0.1..<0.1) }

                // Reference: GPU-only
                _ = convGPU.execute(input: bufInput, weights: bufWeights, output: bufRef, shape: shape)

                // Clear test buffer
                memset(bufTest.contents(), 0, shape.outputBytes)

                // Test: fused 3-way
                let desc = PartitionDescriptor.convolution(
                    shape: shape,
                    fractions: [(.gpu, 0.30), (.mps, 0.50), (.cpu, 0.20)]
                )
                _ = executor.execute(
                    descriptor: desc,
                    inputBuffers: [bufInput, bufWeights],
                    outputBuffer: bufTest
                )

                // Compare
                let refP = bufRef.contents().assumingMemoryBound(to: Float.self)
                let testP = bufTest.contents().assumingMemoryBound(to: Float.self)
                let outCount = shape.outputBytes / MemoryLayout<Float>.size
                var maxErr: Double = 0
                for i in 0..<outCount {
                    maxErr = Swift.max(maxErr, abs(Double(refP[i]) - Double(testP[i])))
                }

                // Conv through MPS involves texture conversion, allow larger tolerance
                let tolerance: Double = 1e-2
                let pass = maxErr < tolerance
                if pass { convCorrectnessPass += 1 }
                print("  \(label): \(pass ? "PASS" : "FAIL") (maxErr=\(String(format: "%.6f", maxErr)))")
            }
            print("  Convolution correctness: \(convCorrectnessPass)/\(correctnessShapes.count)")
            print()

            // ── Profitability gate ──
            print("Convolution profitability gate")
            print(String(repeating: "-", count: 70))

            let gateTests: [(ConvShape, String, Bool)] = [
                // Should partition: enough output channels and FLOPs
                (ConvShape(batch: 1, inChannels: 64, outChannels: 128,
                           height: 28, width: 28, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "64→128 28×28 3×3", true),
                (ConvShape(batch: 1, inChannels: 256, outChannels: 512,
                           height: 14, width: 14, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "256→512 14×14 3×3", true),
                // Should fallback: too few output channels
                (ConvShape(batch: 1, inChannels: 3, outChannels: 32,
                           height: 56, width: 56, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "3→32 56×56 3×3 (few Cout)", false),
                // Should fallback: too small
                (ConvShape(batch: 1, inChannels: 16, outChannels: 16,
                           height: 4, width: 4, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "16→16 4×4 3×3 (tiny)", false),
                // Should partition: pointwise with many channels
                (ConvShape(batch: 1, inChannels: 256, outChannels: 256,
                           height: 14, width: 14, kernelH: 1, kernelW: 1),
                 "256→256 14×14 1×1 (pointwise)", true),
                // Should partition: large ResNet stem
                (ConvShape(batch: 1, inChannels: 3, outChannels: 64,
                           height: 224, width: 224, kernelH: 7, kernelW: 7, strideH: 2, strideW: 2, padH: 3, padW: 3),
                 "3→64 224×224 7×7 s2 (ImageNet)", false),  // outC=64 < 96
            ]

            for (cs, label, expectedPartition) in gateTests {
                let decision = guard_.evaluateConvolution(convShape: cs)
                let isPartition: Bool
                if case .partition = decision { isPartition = true } else { isPartition = false }
                let match = isPartition == expectedPartition
                let reason: String
                if case .fallback(let r, _) = decision { reason = r } else { reason = "partition" }
                print("  \(label): \(match ? "PASS" : "WARN") — \(reason)")
            }
            print()

            // ── Benchmark: single-unit vs fused ──
            print("Convolution benchmark: single-unit vs fused")
            print(String(repeating: "-", count: 70))

            let benchShapes: [(ConvShape, String)] = [
                (ConvShape(batch: 1, inChannels: 64, outChannels: 128,
                           height: 28, width: 28, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "64→128 28×28 3×3"),
                (ConvShape(batch: 1, inChannels: 128, outChannels: 256,
                           height: 14, width: 14, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "128→256 14×14 3×3"),
                (ConvShape(batch: 1, inChannels: 256, outChannels: 512,
                           height: 14, width: 14, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "256→512 14×14 3×3"),
                (ConvShape(batch: 1, inChannels: 256, outChannels: 256,
                           height: 14, width: 14, kernelH: 1, kernelW: 1),
                 "256→256 14×14 1×1"),
                (ConvShape(batch: 1, inChannels: 128, outChannels: 128,
                           height: 56, width: 56, kernelH: 3, kernelW: 3, padH: 1, padW: 1),
                 "128→128 56×56 3×3 (large spatial)"),
            ]

            for (shape, label) in benchShapes {
                guard let bIn = device.makeBuffer(length: shape.inputBytes, options: .storageModeShared),
                      let bW = device.makeBuffer(length: shape.weightBytes, options: .storageModeShared),
                      let bOut = device.makeBuffer(length: shape.outputBytes, options: .storageModeShared) else { continue }

                let inP = bIn.contents().assumingMemoryBound(to: Float.self)
                let wP = bW.contents().assumingMemoryBound(to: Float.self)
                let inCount = shape.inputBytes / MemoryLayout<Float>.size
                let wCount = shape.weightBytes / MemoryLayout<Float>.size
                for i in 0..<inCount { inP[i] = Float.random(in: -0.5..<0.5) }
                for i in 0..<wCount { wP[i] = Float.random(in: -0.1..<0.1) }

                // GPU baseline
                for _ in 0..<3 { _ = convGPU.execute(input: bIn, weights: bW, output: bOut, shape: shape) }
                var gpuTimes: [Double] = []
                for _ in 0..<10 {
                    let t = convGPU.execute(input: bIn, weights: bW, output: bOut, shape: shape)
                    gpuTimes.append(t * 1000.0)
                }
                let gpuMs = TimingSummary(samples: gpuTimes, trimCount: 1).mean

                // MPS baseline (includes buffer↔texture overhead)
                for _ in 0..<3 {
                    _ = convMPS.execute(input: bIn, weights: bW, output: bOut,
                                        weightsOffset: 0, outputOffset: 0,
                                        shape: shape, sliceOutC: shape.outChannels)
                }
                var mpsTimes: [Double] = []
                for _ in 0..<10 {
                    let t = convMPS.execute(input: bIn, weights: bW, output: bOut,
                                             weightsOffset: 0, outputOffset: 0,
                                             shape: shape, sliceOutC: shape.outChannels)
                    mpsTimes.append(t * 1000.0)
                }
                let mpsMs = TimingSummary(samples: mpsTimes, trimCount: 1).mean

                // CPU baseline
                for _ in 0..<3 {
                    _ = convCPU.execute(input: bIn, weights: bW, output: bOut,
                                         weightsOffset: 0, outputOffset: 0,
                                         shape: shape, sliceOutC: shape.outChannels)
                }
                var cpuTimes: [Double] = []
                for _ in 0..<10 {
                    let t = convCPU.execute(input: bIn, weights: bW, output: bOut,
                                             weightsOffset: 0, outputOffset: 0,
                                             shape: shape, sliceOutC: shape.outChannels)
                    cpuTimes.append(t * 1000.0)
                }
                let cpuMs = TimingSummary(samples: cpuTimes, trimCount: 1).mean

                // Fused 3-way
                let desc = PartitionDescriptor.convolution(
                    shape: shape,
                    fractions: [(.gpu, 0.30), (.mps, 0.50), (.cpu, 0.20)]
                )
                for _ in 0..<3 {
                    _ = executor.execute(descriptor: desc, inputBuffers: [bIn, bW], outputBuffer: bOut)
                }
                var fusedTimes: [Double] = []
                for _ in 0..<10 {
                    let start = CACurrentMediaTime()
                    _ = executor.execute(descriptor: desc, inputBuffers: [bIn, bW], outputBuffer: bOut)
                    fusedTimes.append((CACurrentMediaTime() - start) * 1000.0)
                }
                let fusedMs = TimingSummary(samples: fusedTimes, trimCount: 1).mean

                let bestSingle = Swift.min(gpuMs, Swift.min(mpsMs, cpuMs))
                let bestUnit = gpuMs <= mpsMs && gpuMs <= cpuMs ? "GPU" :
                               mpsMs <= cpuMs ? "MPS" : "CPU"
                let speedup = bestSingle / fusedMs

                print(String(format: "  %@ : GPU=%.3fms MPS=%.3fms CPU=%.3fms Fused=%.3fms → %.2fx vs %@ %@",
                             label, gpuMs, mpsMs, cpuMs, fusedMs, speedup, bestUnit,
                             speedup > 1.0 ? "win" : "loss"))

                let gflops = shape.flops / 1e9
                print(String(format: "    FLOPs: %.1fG  GPU: %.1f GFLOPS  MPS: %.1f GFLOPS  CPU: %.1f GFLOPS",
                             gflops,
                             gpuMs > 0 ? gflops / (gpuMs / 1000.0) : 0,
                             mpsMs > 0 ? gflops / (mpsMs / 1000.0) : 0,
                             cpuMs > 0 ? gflops / (cpuMs / 1000.0) : 0))
            }
            print()

        } catch {
            print("  Convolution setup failed: \(error)")
        }
    }

    print(String(format: "\nTotal benchmark time: %.1f seconds", CACurrentMediaTime() - totalStart))
}

// MARK: - Attention Output

func printAttentionSummary(results: [AttentionBenchmarkResult]) {
    print(String(repeating: "=", count: 90))
    print("ATTENTION SUMMARY")
    print(String(repeating: "=", count: 90))

    let header = "Shape".padding(toLength: 22, withPad: " ", startingAt: 0)
        + "GPU (ms)".padding(toLength: 12, withPad: " ", startingAt: 0)
        + "MPS (ms)".padding(toLength: 12, withPad: " ", startingAt: 0)
        + "CPU (ms)".padding(toLength: 12, withPad: " ", startingAt: 0)
        + "Fused (ms)".padding(toLength: 14, withPad: " ", startingAt: 0)
        + "Speedup"
    print(header)
    print(String(repeating: "-", count: 90))

    var wins = 0
    for r in results {
        var line = "\(r.shape)".padding(toLength: 22, withPad: " ", startingAt: 0)
        line += String(format: "%-12.3f", r.gpuOnlyMs)
        line += String(format: "%-12.3f", r.mpsOnlyMs)
        line += String(format: "%-12.3f", r.cpuOnlyMs)
        line += String(format: "%-14.3f", r.fusedMs)
        if r.speedup > 1.0 {
            line += String(format: "%.2fx ✓", r.speedup)
            wins += 1
        } else {
            line += String(format: "%.2fx ✗", r.speedup)
        }
        print(line)
    }
    print()
    print("  Attention fusion: \(wins)/\(results.count) shapes show speedup")
}

// MARK: - Output

func printShapeResult(_ r: ShapeBenchmarkResult) {
    print("  GPU only:      \(r.gpuOnly)")
    print("  MPS only:      \(r.mpsOnly)")
    print("  CPU only:      \(r.cpuOnly)")
    print("  Fused2 (50/50):\(r.fusedTwoWay)")
    if let three = r.fusedThreeWay {
        print("  Fused3 (equal):\(three)")
    }
    if let opt2 = r.fusedTwoWayOptimal, let frac = r.optimalGpuFraction2 {
        print(String(format: "  Fused2 (opt):  \(opt2)  [gpu=%.0f%% mps=%.0f%%]",
                     frac * 100, (1 - frac) * 100))
    }
    if let opt3 = r.fusedThreeWayOptimal, let f = r.optimalFractions3 {
        print(String(format: "  Fused3 (opt):  \(opt3)  [gpu=%.0f%% mps=%.0f%% cpu=%.0f%%]",
                     f.gpu * 100, f.mps * 100, f.cpu * 100))
    }
    print(String(format: "  Best single:   %.3f ms (%@)", r.bestSingleMs, r.bestSingleUnit.rawValue))
    print(String(format: "  Best speedup:  %.3fx", r.bestSpeedup))
}

func printSummaryTable(results: [ShapeBenchmarkResult]) {
    print(String(repeating: "=", count: 100))
    print("SUMMARY")
    print(String(repeating: "=", count: 100))

    let header = "Shape".padding(toLength: 20, withPad: " ", startingAt: 0)
        + "GPU (ms)".padding(toLength: 12, withPad: " ", startingAt: 0)
        + "MPS (ms)".padding(toLength: 12, withPad: " ", startingAt: 0)
        + "CPU (ms)".padding(toLength: 12, withPad: " ", startingAt: 0)
        + "F2-opt (ms)".padding(toLength: 14, withPad: " ", startingAt: 0)
        + "F3-opt (ms)".padding(toLength: 14, withPad: " ", startingAt: 0)
        + "Best"
    print(header)
    print(String(repeating: "-", count: 100))

    for r in results {
        var line = "\(r.shape)".padding(toLength: 20, withPad: " ", startingAt: 0)
        line += String(format: "%-12.3f", r.gpuOnly.mean)
        line += String(format: "%-12.3f", r.mpsOnly.mean)
        line += String(format: "%-12.3f", r.cpuOnly.mean)

        if let opt2 = r.fusedTwoWayOptimal {
            line += String(format: "%-14.3f", opt2.mean)
        } else {
            line += "N/A".padding(toLength: 14, withPad: " ", startingAt: 0)
        }

        if let opt3 = r.fusedThreeWayOptimal {
            line += String(format: "%-14.3f", opt3.mean)
        } else {
            line += "N/A".padding(toLength: 14, withPad: " ", startingAt: 0)
        }

        let s = r.bestSpeedup
        if s > 1.0 {
            line += String(format: "%.2fx ✓", s)
        } else {
            line += String(format: "%.2fx ✗", s)
        }
        print(line)
    }
    print()
}

func printDecisionGate(results: [ShapeBenchmarkResult]) {
    print(String(repeating: "=", count: 100))
    print("DECISION GATE: T_fused < min(T_gpu, T_ane, T_cpu) on ≥ 3 shapes?")
    print(String(repeating: "=", count: 100))

    var wins = 0
    for r in results {
        let s = r.bestSpeedup
        if s > 1.0 {
            wins += 1
            print("  \(r.shape): PASS (speedup = \(String(format: "%.2fx", s)))")
        } else {
            print("  \(r.shape): FAIL (speedup = \(String(format: "%.2fx", s)))")
        }
    }

    print()
    if wins >= 3 {
        print("  RESULT: PASS (\(wins)/\(results.count) shapes show speedup)")
        print("  → Proceed to Phase 2: Partition Descriptor")
    } else {
        print("  RESULT: FAIL (\(wins)/\(results.count) shapes show speedup)")
        print("  → Investigate: sync overhead, split ratios, kernel efficiency")
    }
}

main()
