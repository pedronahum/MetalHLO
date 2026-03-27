// FusionAnalysisBenchmarks.swift
// MetalHLO Benchmarks
//
// Fusion effectiveness analysis benchmarks comparing fused vs naive execution.

import Foundation
import MetalHLO

/// Factory for creating fusion analysis benchmarks.
/// These benchmarks measure the effectiveness of operation fusion by comparing
/// fused execution (operations combined in one program) vs naive execution
/// (operations executed separately).
public enum FusionAnalysisBenchmarks {

    /// Fusion benchmark configuration.
    public struct FusionBenchmark: Sendable {
        public let id: String
        public let name: String
        public let category: String = "fusion"
        public let pattern: String
        public let expectedSpeedup: String
        public let fusedProgram: String
        public let naivePrograms: [String]
        public let inputShapes: [[Int]]
        public let inputGenerator: @Sendable (Client) throws -> [Buffer]

        public init(
            id: String,
            name: String,
            pattern: String,
            expectedSpeedup: String,
            fusedProgram: String,
            naivePrograms: [String],
            inputShapes: [[Int]],
            inputGenerator: @escaping @Sendable (Client) throws -> [Buffer]
        ) {
            self.id = id
            self.name = name
            self.pattern = pattern
            self.expectedSpeedup = expectedSpeedup
            self.fusedProgram = fusedProgram
            self.naivePrograms = naivePrograms
            self.inputShapes = inputShapes
            self.inputGenerator = inputGenerator
        }
    }

    // MARK: - Fusion Analysis Benchmarks

    /// Get all fusion analysis benchmarks.
    public static func all() -> [FusionBenchmark] {
        var benchmarks: [FusionBenchmark] = []

        // FUSION-001: Element-wise chain (add → mul → exp)
        benchmarks.append(FusionBenchmark(
            id: "FUSION-001",
            name: "Elementwise Chain (add→mul→exp)",
            pattern: "elementwise_chain",
            expectedSpeedup: "2-3x",
            fusedProgram: """
            module @fused_elementwise_chain {
              func.func @main(%a: tensor<4096x4096xf32>, %b: tensor<4096x4096xf32>, %c: tensor<4096x4096xf32>) -> (tensor<4096x4096xf32>) {
                %add = stablehlo.add %a, %b : tensor<4096x4096xf32>
                %mul = stablehlo.multiply %add, %c : tensor<4096x4096xf32>
                %exp = stablehlo.exponential %mul : tensor<4096x4096xf32>
                return %exp : tensor<4096x4096xf32>
              }
            }
            """,
            naivePrograms: [
                """
                module @naive_add {
                  func.func @main(%a: tensor<4096x4096xf32>, %b: tensor<4096x4096xf32>) -> (tensor<4096x4096xf32>) {
                    %r = stablehlo.add %a, %b : tensor<4096x4096xf32>
                    return %r : tensor<4096x4096xf32>
                  }
                }
                """,
                """
                module @naive_mul {
                  func.func @main(%a: tensor<4096x4096xf32>, %b: tensor<4096x4096xf32>) -> (tensor<4096x4096xf32>) {
                    %r = stablehlo.multiply %a, %b : tensor<4096x4096xf32>
                    return %r : tensor<4096x4096xf32>
                  }
                }
                """,
                """
                module @naive_exp {
                  func.func @main(%a: tensor<4096x4096xf32>) -> (tensor<4096x4096xf32>) {
                    %r = stablehlo.exponential %a : tensor<4096x4096xf32>
                    return %r : tensor<4096x4096xf32>
                  }
                }
                """
            ],
            inputShapes: [[4096, 4096], [4096, 4096], [4096, 4096]],
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let a = try gen.createUniformFloat32Buffer(client: client, shape: [4096, 4096], min: -0.5, max: 0.5)
                let b = try gen.createUniformFloat32Buffer(client: client, shape: [4096, 4096], min: -0.5, max: 0.5)
                let c = try gen.createUniformFloat32Buffer(client: client, shape: [4096, 4096], min: 0.1, max: 0.5)
                return [a, b, c]
            }
        ))

        // FUSION-002: Reduction fusion (mul → reduce_sum)
        benchmarks.append(FusionBenchmark(
            id: "FUSION-002",
            name: "Reduction Fusion (mul→reduce_sum)",
            pattern: "reduction_fusion",
            expectedSpeedup: "1.5-2x",
            fusedProgram: """
            module @fused_reduction {
              func.func @main(%a: tensor<4096x4096xf32>, %b: tensor<4096x4096xf32>) -> (tensor<4096xf32>) {
                %mul = stablehlo.multiply %a, %b : tensor<4096x4096xf32>
                %init = stablehlo.constant dense<0.0> : tensor<f32>
                %sum = stablehlo.reduce %mul, %init applies stablehlo.add across dimensions = [1] : (tensor<4096x4096xf32>, tensor<f32>) -> tensor<4096xf32>
                return %sum : tensor<4096xf32>
              }
            }
            """,
            naivePrograms: [
                """
                module @naive_mul {
                  func.func @main(%a: tensor<4096x4096xf32>, %b: tensor<4096x4096xf32>) -> (tensor<4096x4096xf32>) {
                    %r = stablehlo.multiply %a, %b : tensor<4096x4096xf32>
                    return %r : tensor<4096x4096xf32>
                  }
                }
                """,
                """
                module @naive_reduce {
                  func.func @main(%a: tensor<4096x4096xf32>) -> (tensor<4096xf32>) {
                    %init = stablehlo.constant dense<0.0> : tensor<f32>
                    %sum = stablehlo.reduce %a, %init applies stablehlo.add across dimensions = [1] : (tensor<4096x4096xf32>, tensor<f32>) -> tensor<4096xf32>
                    return %sum : tensor<4096xf32>
                  }
                }
                """
            ],
            inputShapes: [[4096, 4096], [4096, 4096]],
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let a = try gen.createUniformFloat32Buffer(client: client, shape: [4096, 4096])
                let b = try gen.createUniformFloat32Buffer(client: client, shape: [4096, 4096])
                return [a, b]
            }
        ))

        // FUSION-003: Broadcast fusion (broadcast → add)
        benchmarks.append(FusionBenchmark(
            id: "FUSION-003",
            name: "Broadcast Fusion (broadcast→add)",
            pattern: "broadcast_fusion",
            expectedSpeedup: "1.5-2x",
            fusedProgram: """
            module @fused_broadcast {
              func.func @main(%a: tensor<4096x4096xf32>, %b: tensor<4096xf32>) -> (tensor<4096x4096xf32>) {
                %bc = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<4096xf32>) -> tensor<4096x4096xf32>
                %add = stablehlo.add %a, %bc : tensor<4096x4096xf32>
                return %add : tensor<4096x4096xf32>
              }
            }
            """,
            naivePrograms: [
                """
                module @naive_broadcast {
                  func.func @main(%b: tensor<4096xf32>) -> (tensor<4096x4096xf32>) {
                    %bc = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<4096xf32>) -> tensor<4096x4096xf32>
                    return %bc : tensor<4096x4096xf32>
                  }
                }
                """,
                """
                module @naive_add {
                  func.func @main(%a: tensor<4096x4096xf32>, %b: tensor<4096x4096xf32>) -> (tensor<4096x4096xf32>) {
                    %r = stablehlo.add %a, %b : tensor<4096x4096xf32>
                    return %r : tensor<4096x4096xf32>
                  }
                }
                """
            ],
            inputShapes: [[4096, 4096], [4096]],
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let a = try gen.createUniformFloat32Buffer(client: client, shape: [4096, 4096])
                let b = try gen.createUniformFloat32Buffer(client: client, shape: [4096])
                return [a, b]
            }
        ))

        // FUSION-004: Activation fusion (matmul → bias → relu)
        benchmarks.append(FusionBenchmark(
            id: "FUSION-004",
            name: "Activation Fusion (matmul→bias→relu)",
            pattern: "activation_fusion",
            expectedSpeedup: "1.2-1.5x",
            fusedProgram: """
            module @fused_activation {
              func.func @main(%x: tensor<128x1024xf32>, %w: tensor<1024x1024xf32>, %b: tensor<1024xf32>) -> (tensor<128x1024xf32>) {
                %mm = stablehlo.dot_general %x, %w, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x1024xf32>, tensor<1024x1024xf32>) -> tensor<128x1024xf32>
                %bc = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024xf32>
                %bias = stablehlo.add %mm, %bc : tensor<128x1024xf32>
                %zero = stablehlo.constant dense<0.0> : tensor<128x1024xf32>
                %relu = stablehlo.maximum %bias, %zero : tensor<128x1024xf32>
                return %relu : tensor<128x1024xf32>
              }
            }
            """,
            naivePrograms: [
                """
                module @naive_matmul {
                  func.func @main(%x: tensor<128x1024xf32>, %w: tensor<1024x1024xf32>) -> (tensor<128x1024xf32>) {
                    %mm = stablehlo.dot_general %x, %w, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x1024xf32>, tensor<1024x1024xf32>) -> tensor<128x1024xf32>
                    return %mm : tensor<128x1024xf32>
                  }
                }
                """,
                """
                module @naive_bias {
                  func.func @main(%a: tensor<128x1024xf32>, %b: tensor<1024xf32>) -> (tensor<128x1024xf32>) {
                    %bc = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<1024xf32>) -> tensor<128x1024xf32>
                    %bias = stablehlo.add %a, %bc : tensor<128x1024xf32>
                    return %bias : tensor<128x1024xf32>
                  }
                }
                """,
                """
                module @naive_relu {
                  func.func @main(%a: tensor<128x1024xf32>) -> (tensor<128x1024xf32>) {
                    %zero = stablehlo.constant dense<0.0> : tensor<128x1024xf32>
                    %relu = stablehlo.maximum %a, %zero : tensor<128x1024xf32>
                    return %relu : tensor<128x1024xf32>
                  }
                }
                """
            ],
            inputShapes: [[128, 1024], [1024, 1024], [1024]],
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [128, 1024])
                let w = try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])
                let b = try gen.createUniformFloat32Buffer(client: client, shape: [1024])
                return [x, w, b]
            }
        ))

        return benchmarks
    }
}

// MARK: - Fusion Analysis Result

/// Result of a fusion analysis benchmark.
public struct FusionAnalysisResult: Sendable {
    public let id: String
    public let name: String
    public let pattern: String
    public let expectedSpeedup: String
    public let fusedTime: TimingStatistics
    public let naiveTime: TimingStatistics
    public let actualSpeedup: Double
    public let iterations: Int

    public init(
        id: String,
        name: String,
        pattern: String,
        expectedSpeedup: String,
        fusedTime: TimingStatistics,
        naiveTime: TimingStatistics,
        actualSpeedup: Double,
        iterations: Int
    ) {
        self.id = id
        self.name = name
        self.pattern = pattern
        self.expectedSpeedup = expectedSpeedup
        self.fusedTime = fusedTime
        self.naiveTime = naiveTime
        self.actualSpeedup = actualSpeedup
        self.iterations = iterations
    }

    /// Format result as a string.
    public func formatted() -> String {
        let fusedMs = fusedTime.mean * 1000
        let naiveMs = naiveTime.mean * 1000
        return "\(id): Fused=\(String(format: "%.3f", fusedMs))ms, Naive=\(String(format: "%.3f", naiveMs))ms, Speedup=\(String(format: "%.2f", actualSpeedup))x (expected: \(expectedSpeedup))"
    }
}

// MARK: - Fusion Analysis Runner

/// Runner for fusion analysis benchmarks.
public struct FusionAnalysisRunner: Sendable {

    public let warmupIterations: Int
    public let measurementIterations: Int

    public init(warmupIterations: Int = 5, measurementIterations: Int = 20) {
        self.warmupIterations = warmupIterations
        self.measurementIterations = measurementIterations
    }

    /// Run a fusion analysis benchmark.
    public func run(_ benchmark: FusionAnalysisBenchmarks.FusionBenchmark) throws -> FusionAnalysisResult {
        let client = try Client.create()

        // Create inputs
        let inputs = try benchmark.inputGenerator(client)

        // Compile fused program
        let fusedExe = try client.compile(benchmark.fusedProgram)

        // Measure fused execution
        var fusedTimes: [Double] = []

        // Warmup fused
        for _ in 0..<warmupIterations {
            let _ = try fusedExe.execute(inputs)
        }

        // Measure fused
        for _ in 0..<measurementIterations {
            let start = CFAbsoluteTimeGetCurrent()
            let _ = try fusedExe.execute(inputs)
            let end = CFAbsoluteTimeGetCurrent()
            fusedTimes.append(end - start)
        }

        // Measure naive execution (run each operation separately)
        var naiveTimes: [Double] = []

        // Compile naive programs
        var naiveExes: [Executable] = []
        for program in benchmark.naivePrograms {
            naiveExes.append(try client.compile(program))
        }

        // For naive execution, we need to chain the outputs
        // This is a simplified measurement - we measure total time for all separate ops
        // Warmup naive
        for _ in 0..<warmupIterations {
            var currentInputs = inputs
            for (idx, exe) in naiveExes.enumerated() {
                // Build inputs for this operation
                let opInputs: [Buffer]
                if idx == 0 {
                    // First op uses original inputs (subset based on operation)
                    opInputs = selectInputsForOp(benchmark: benchmark, opIndex: idx, inputs: currentInputs)
                } else {
                    // Subsequent ops use previous output + remaining inputs
                    opInputs = selectInputsForOp(benchmark: benchmark, opIndex: idx, inputs: currentInputs)
                }
                let outputs = try exe.execute(opInputs)
                // Update currentInputs with outputs for next operation
                currentInputs = updateInputsWithOutputs(currentInputs: currentInputs, outputs: outputs, opIndex: idx, benchmark: benchmark)
            }
        }

        // Measure naive
        for _ in 0..<measurementIterations {
            let start = CFAbsoluteTimeGetCurrent()
            var currentInputs = inputs
            for (idx, exe) in naiveExes.enumerated() {
                let opInputs = selectInputsForOp(benchmark: benchmark, opIndex: idx, inputs: currentInputs)
                let outputs = try exe.execute(opInputs)
                currentInputs = updateInputsWithOutputs(currentInputs: currentInputs, outputs: outputs, opIndex: idx, benchmark: benchmark)
            }
            let end = CFAbsoluteTimeGetCurrent()
            naiveTimes.append(end - start)
        }

        let fusedStats = TimingStatistics(samples: fusedTimes)
        let naiveStats = TimingStatistics(samples: naiveTimes)
        let speedup = naiveStats.mean / fusedStats.mean

        return FusionAnalysisResult(
            id: benchmark.id,
            name: benchmark.name,
            pattern: benchmark.pattern,
            expectedSpeedup: benchmark.expectedSpeedup,
            fusedTime: fusedStats,
            naiveTime: naiveStats,
            actualSpeedup: speedup,
            iterations: measurementIterations
        )
    }

    /// Run all fusion analysis benchmarks.
    public func runAll() throws -> [FusionAnalysisResult] {
        var results: [FusionAnalysisResult] = []
        for benchmark in FusionAnalysisBenchmarks.all() {
            let result = try run(benchmark)
            results.append(result)
        }
        return results
    }

    // Helper to select inputs for each operation in the naive chain
    private func selectInputsForOp(benchmark: FusionAnalysisBenchmarks.FusionBenchmark, opIndex: Int, inputs: [Buffer]) -> [Buffer] {
        switch benchmark.pattern {
        case "elementwise_chain":
            // Op 0: add(a, b), Op 1: mul(prev, c), Op 2: exp(prev)
            switch opIndex {
            case 0: return [inputs[0], inputs[1]]  // a, b
            case 1: return [inputs[0], inputs[2]]  // prev output, c (inputs[0] is replaced with prev output)
            case 2: return [inputs[0]]  // prev output
            default: return inputs
            }
        case "reduction_fusion":
            // Op 0: mul(a, b), Op 1: reduce(prev)
            switch opIndex {
            case 0: return [inputs[0], inputs[1]]  // a, b
            case 1: return [inputs[0]]  // prev output
            default: return inputs
            }
        case "broadcast_fusion":
            // Op 0: broadcast(b), Op 1: add(a, prev)
            switch opIndex {
            case 0: return [inputs[1]]  // b
            case 1: return [inputs[0], inputs[1]]  // a, prev output (inputs[1] replaced)
            default: return inputs
            }
        case "activation_fusion":
            // Op 0: matmul(x, w), Op 1: bias(prev, b), Op 2: relu(prev)
            switch opIndex {
            case 0: return [inputs[0], inputs[1]]  // x, w
            case 1: return [inputs[0], inputs[2]]  // prev output, b
            case 2: return [inputs[0]]  // prev output
            default: return inputs
            }
        default:
            return inputs
        }
    }

    // Helper to update inputs with operation outputs for chaining
    private func updateInputsWithOutputs(currentInputs: [Buffer], outputs: [Buffer], opIndex: Int, benchmark: FusionAnalysisBenchmarks.FusionBenchmark) -> [Buffer] {
        guard !outputs.isEmpty else { return currentInputs }

        var updated = currentInputs
        // Replace first input with the output for next operation
        if !updated.isEmpty {
            updated[0] = outputs[0]
        }

        // For broadcast fusion, the broadcast output goes to position 1
        if benchmark.pattern == "broadcast_fusion" && opIndex == 0 && updated.count > 1 {
            updated[1] = outputs[0]
        }

        return updated
    }
}
