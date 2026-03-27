// CompilerBenchmarks.swift
// MetalHLO Benchmarks
//
// Compiler analysis benchmarks measuring compilation time for various program sizes.

import Foundation
import MetalHLO

/// Factory for creating compiler analysis benchmarks.
/// These benchmarks measure compilation time rather than execution time.
public enum CompilerBenchmarks {

    /// Compiler benchmark that measures compilation time.
    /// Note: This does NOT conform to the Benchmark protocol because it measures
    /// compilation time, not execution time, and doesn't require input buffers.
    public struct CompilerBenchmark: Sendable {
        public let id: String
        public let name: String
        public let category: String = "compiler"
        public let operation: String
        public let configuration: [String: String]
        public let mlirProgram: String
        public let expectedOps: Int

        public init(
            id: String,
            name: String,
            operation: String,
            configuration: [String: String],
            mlirProgram: String,
            expectedOps: Int
        ) {
            self.id = id
            self.name = name
            self.operation = operation
            self.configuration = configuration
            self.mlirProgram = mlirProgram
            self.expectedOps = expectedOps
        }
    }

    // MARK: - Compiler Analysis Benchmarks

    /// Get all compiler benchmarks.
    public static func all() -> [CompilerBenchmark] {
        var benchmarks: [CompilerBenchmark] = []

        // COMP-001: Simple add operation (~5 ops)
        benchmarks.append(CompilerBenchmark(
            id: "COMP-001",
            name: "Compile Simple Add (~5 ops)",
            operation: "compile",
            configuration: ["ops": "~5", "target": "<100ms"],
            mlirProgram: """
            module @simple_add {
              func.func @main(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>) {
                %c = stablehlo.add %a, %b : tensor<1024x1024xf32>
                return %c : tensor<1024x1024xf32>
              }
            }
            """,
            expectedOps: 5
        ))

        // COMP-002: MLP Forward (~50 ops)
        benchmarks.append(CompilerBenchmark(
            id: "COMP-002",
            name: "Compile MLP Forward (~50 ops)",
            operation: "compile",
            configuration: ["ops": "~50", "target": "<500ms"],
            mlirProgram: """
            module @mlp_forward {
              func.func @main(%x: tensor<32x784xf32>, %w1: tensor<784x512xf32>, %b1: tensor<512xf32>, %w2: tensor<512x256xf32>, %b2: tensor<256xf32>, %w3: tensor<256x128xf32>, %b3: tensor<128xf32>, %w4: tensor<128x10xf32>, %b4: tensor<10xf32>) -> (tensor<32x10xf32>) {
                // Layer 1
                %h1_mm = stablehlo.dot_general %x, %w1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x784xf32>, tensor<784x512xf32>) -> tensor<32x512xf32>
                %b1_bc = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512xf32>
                %h1_bias = stablehlo.add %h1_mm, %b1_bc : tensor<32x512xf32>
                %zero1 = stablehlo.constant dense<0.0> : tensor<32x512xf32>
                %h1 = stablehlo.maximum %h1_bias, %zero1 : tensor<32x512xf32>

                // Layer 2
                %h2_mm = stablehlo.dot_general %h1, %w2, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x512xf32>, tensor<512x256xf32>) -> tensor<32x256xf32>
                %b2_bc = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256xf32>
                %h2_bias = stablehlo.add %h2_mm, %b2_bc : tensor<32x256xf32>
                %zero2 = stablehlo.constant dense<0.0> : tensor<32x256xf32>
                %h2 = stablehlo.maximum %h2_bias, %zero2 : tensor<32x256xf32>

                // Layer 3
                %h3_mm = stablehlo.dot_general %h2, %w3, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x256xf32>, tensor<256x128xf32>) -> tensor<32x128xf32>
                %b3_bc = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<128xf32>) -> tensor<32x128xf32>
                %h3_bias = stablehlo.add %h3_mm, %b3_bc : tensor<32x128xf32>
                %zero3 = stablehlo.constant dense<0.0> : tensor<32x128xf32>
                %h3 = stablehlo.maximum %h3_bias, %zero3 : tensor<32x128xf32>

                // Layer 4
                %out_mm = stablehlo.dot_general %h3, %w4, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x128xf32>, tensor<128x10xf32>) -> tensor<32x10xf32>
                %b4_bc = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
                %out = stablehlo.add %out_mm, %b4_bc : tensor<32x10xf32>

                return %out : tensor<32x10xf32>
              }
            }
            """,
            expectedOps: 50
        ))

        // COMP-003: CNN Block (~100 ops)
        benchmarks.append(CompilerBenchmark(
            id: "COMP-003",
            name: "Compile CNN Block (~100 ops)",
            operation: "compile",
            configuration: ["ops": "~100", "target": "<1s"],
            mlirProgram: """
            module @cnn_block {
              func.func @main(%input: tensor<32x56x56x64xf32>, %conv1_w: tensor<3x3x64x64xf32>, %bn1_scale: tensor<64xf32>, %bn1_offset: tensor<64xf32>, %bn1_mean: tensor<64xf32>, %bn1_var: tensor<64xf32>, %conv2_w: tensor<3x3x64x64xf32>, %bn2_scale: tensor<64xf32>, %bn2_offset: tensor<64xf32>, %bn2_mean: tensor<64xf32>, %bn2_var: tensor<64xf32>) -> (tensor<32x56x56x64xf32>) {
                // Conv1
                %conv1 = stablehlo.convolution %input, %conv1_w window_strides = [1, 1], padding = [[1, 1], [1, 1]], feature_group_count = 1 : (tensor<32x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<32x56x56x64xf32>

                // BatchNorm1
                %bn1 = stablehlo.batch_norm_inference %conv1, %bn1_scale, %bn1_offset, %bn1_mean, %bn1_var, epsilon = 1.0e-05, feature_index = 3 : (tensor<32x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<32x56x56x64xf32>

                // ReLU1
                %zero1 = stablehlo.constant dense<0.0> : tensor<32x56x56x64xf32>
                %relu1 = stablehlo.maximum %bn1, %zero1 : tensor<32x56x56x64xf32>

                // Conv2
                %conv2 = stablehlo.convolution %relu1, %conv2_w window_strides = [1, 1], padding = [[1, 1], [1, 1]], feature_group_count = 1 : (tensor<32x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<32x56x56x64xf32>

                // BatchNorm2
                %bn2 = stablehlo.batch_norm_inference %conv2, %bn2_scale, %bn2_offset, %bn2_mean, %bn2_var, epsilon = 1.0e-05, feature_index = 3 : (tensor<32x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<32x56x56x64xf32>

                // Residual add
                %residual = stablehlo.add %bn2, %input : tensor<32x56x56x64xf32>

                // ReLU2
                %zero2 = stablehlo.constant dense<0.0> : tensor<32x56x56x64xf32>
                %out = stablehlo.maximum %residual, %zero2 : tensor<32x56x56x64xf32>

                return %out : tensor<32x56x56x64xf32>
              }
            }
            """,
            expectedOps: 100
        ))

        // COMP-004: Transformer Block (~200 ops)
        benchmarks.append(CompilerBenchmark(
            id: "COMP-004",
            name: "Compile Transformer Block (~200 ops)",
            operation: "compile",
            configuration: ["ops": "~200", "target": "<2s"],
            mlirProgram: """
            module @transformer_block {
              func.func @main(%x: tensor<8x128x768xf32>, %wq: tensor<768x768xf32>, %wk: tensor<768x768xf32>, %wv: tensor<768x768xf32>, %wo: tensor<768x768xf32>, %ln1_gamma: tensor<768xf32>, %ln1_beta: tensor<768xf32>, %w1: tensor<768x3072xf32>, %b1: tensor<3072xf32>, %w2: tensor<3072x768xf32>, %b2: tensor<768xf32>, %ln2_gamma: tensor<768xf32>, %ln2_beta: tensor<768xf32>) -> (tensor<8x128x768xf32>) {
                %eps = stablehlo.constant dense<1.0e-05> : tensor<f32>
                %hidden_size = stablehlo.constant dense<768.0> : tensor<f32>
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %scale = stablehlo.constant dense<0.125> : tensor<f32>

                // === Self-Attention ===
                // Reshape input for projections
                %x_flat = stablehlo.reshape %x : (tensor<8x128x768xf32>) -> tensor<1024x768xf32>

                // Q, K, V projections
                %q_proj = stablehlo.dot_general %x_flat, %wq, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x768xf32>, tensor<768x768xf32>) -> tensor<1024x768xf32>
                %k_proj = stablehlo.dot_general %x_flat, %wk, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x768xf32>, tensor<768x768xf32>) -> tensor<1024x768xf32>
                %v_proj = stablehlo.dot_general %x_flat, %wv, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x768xf32>, tensor<768x768xf32>) -> tensor<1024x768xf32>

                // Reshape to multi-head: [8, 128, 12, 64]
                %q_reshape = stablehlo.reshape %q_proj : (tensor<1024x768xf32>) -> tensor<8x128x12x64xf32>
                %k_reshape = stablehlo.reshape %k_proj : (tensor<1024x768xf32>) -> tensor<8x128x12x64xf32>
                %v_reshape = stablehlo.reshape %v_proj : (tensor<1024x768xf32>) -> tensor<8x128x12x64xf32>

                // Transpose to [8, 12, 128, 64]
                %q = stablehlo.transpose %q_reshape, dims = [0, 2, 1, 3] : (tensor<8x128x12x64xf32>) -> tensor<8x12x128x64xf32>
                %k = stablehlo.transpose %k_reshape, dims = [0, 2, 1, 3] : (tensor<8x128x12x64xf32>) -> tensor<8x12x128x64xf32>
                %v = stablehlo.transpose %v_reshape, dims = [0, 2, 1, 3] : (tensor<8x128x12x64xf32>) -> tensor<8x12x128x64xf32>

                // Attention scores: Q @ K^T
                %k_t = stablehlo.transpose %k, dims = [0, 1, 3, 2] : (tensor<8x12x128x64xf32>) -> tensor<8x12x64x128xf32>
                %scores = stablehlo.dot_general %q, %k_t, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<8x12x128x64xf32>, tensor<8x12x64x128xf32>) -> tensor<8x12x128x128xf32>
                %scale_bc = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<8x12x128x128xf32>
                %scores_scaled = stablehlo.multiply %scores, %scale_bc : tensor<8x12x128x128xf32>

                // Softmax
                %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
                %max = stablehlo.reduce %scores_scaled, %neg_inf applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x128x128xf32>, tensor<f32>) -> tensor<8x12x128xf32>
                %max_bc = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2] : (tensor<8x12x128xf32>) -> tensor<8x12x128x128xf32>
                %shifted = stablehlo.subtract %scores_scaled, %max_bc : tensor<8x12x128x128xf32>
                %exp = stablehlo.exponential %shifted : tensor<8x12x128x128xf32>
                %sum = stablehlo.reduce %exp, %zero applies stablehlo.add across dimensions = [3] : (tensor<8x12x128x128xf32>, tensor<f32>) -> tensor<8x12x128xf32>
                %sum_bc = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<8x12x128xf32>) -> tensor<8x12x128x128xf32>
                %attn = stablehlo.divide %exp, %sum_bc : tensor<8x12x128x128xf32>

                // Attention output
                %attn_out = stablehlo.dot_general %attn, %v, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<8x12x128x128xf32>, tensor<8x12x128x64xf32>) -> tensor<8x12x128x64xf32>

                // Reshape back
                %attn_transpose = stablehlo.transpose %attn_out, dims = [0, 2, 1, 3] : (tensor<8x12x128x64xf32>) -> tensor<8x128x12x64xf32>
                %attn_reshape = stablehlo.reshape %attn_transpose : (tensor<8x128x12x64xf32>) -> tensor<1024x768xf32>

                // Output projection
                %out_proj = stablehlo.dot_general %attn_reshape, %wo, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x768xf32>, tensor<768x768xf32>) -> tensor<1024x768xf32>
                %out_proj_3d = stablehlo.reshape %out_proj : (tensor<1024x768xf32>) -> tensor<8x128x768xf32>

                // Residual + LayerNorm1
                %residual1 = stablehlo.add %x, %out_proj_3d : tensor<8x128x768xf32>
                %sum1 = stablehlo.reduce %residual1, %zero applies stablehlo.add across dimensions = [2] : (tensor<8x128x768xf32>, tensor<f32>) -> tensor<8x128xf32>
                %hidden_bc1 = stablehlo.broadcast_in_dim %hidden_size, dims = [] : (tensor<f32>) -> tensor<8x128xf32>
                %mean1 = stablehlo.divide %sum1, %hidden_bc1 : tensor<8x128xf32>
                %mean1_bc = stablehlo.broadcast_in_dim %mean1, dims = [0, 1] : (tensor<8x128xf32>) -> tensor<8x128x768xf32>
                %centered1 = stablehlo.subtract %residual1, %mean1_bc : tensor<8x128x768xf32>
                %sq1 = stablehlo.multiply %centered1, %centered1 : tensor<8x128x768xf32>
                %var_sum1 = stablehlo.reduce %sq1, %zero applies stablehlo.add across dimensions = [2] : (tensor<8x128x768xf32>, tensor<f32>) -> tensor<8x128xf32>
                %var1 = stablehlo.divide %var_sum1, %hidden_bc1 : tensor<8x128xf32>
                %eps_bc1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<8x128xf32>
                %var_eps1 = stablehlo.add %var1, %eps_bc1 : tensor<8x128xf32>
                %rstd1 = stablehlo.rsqrt %var_eps1 : tensor<8x128xf32>
                %rstd1_bc = stablehlo.broadcast_in_dim %rstd1, dims = [0, 1] : (tensor<8x128xf32>) -> tensor<8x128x768xf32>
                %normalized1 = stablehlo.multiply %centered1, %rstd1_bc : tensor<8x128x768xf32>
                %gamma1_bc = stablehlo.broadcast_in_dim %ln1_gamma, dims = [2] : (tensor<768xf32>) -> tensor<8x128x768xf32>
                %beta1_bc = stablehlo.broadcast_in_dim %ln1_beta, dims = [2] : (tensor<768xf32>) -> tensor<8x128x768xf32>
                %scaled1 = stablehlo.multiply %normalized1, %gamma1_bc : tensor<8x128x768xf32>
                %ln1_out = stablehlo.add %scaled1, %beta1_bc : tensor<8x128x768xf32>

                // === FFN ===
                %ln1_flat = stablehlo.reshape %ln1_out : (tensor<8x128x768xf32>) -> tensor<1024x768xf32>
                %ffn1 = stablehlo.dot_general %ln1_flat, %w1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x768xf32>, tensor<768x3072xf32>) -> tensor<1024x3072xf32>
                %b1_bc_ffn = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<3072xf32>) -> tensor<1024x3072xf32>
                %ffn1_bias = stablehlo.add %ffn1, %b1_bc_ffn : tensor<1024x3072xf32>
                %gelu = stablehlo.logistic %ffn1_bias : tensor<1024x3072xf32>
                %gelu_out = stablehlo.multiply %ffn1_bias, %gelu : tensor<1024x3072xf32>
                %ffn2 = stablehlo.dot_general %gelu_out, %w2, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x3072xf32>, tensor<3072x768xf32>) -> tensor<1024x768xf32>
                %b2_bc_ffn = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<768xf32>) -> tensor<1024x768xf32>
                %ffn2_bias = stablehlo.add %ffn2, %b2_bc_ffn : tensor<1024x768xf32>
                %ffn_out = stablehlo.reshape %ffn2_bias : (tensor<1024x768xf32>) -> tensor<8x128x768xf32>

                // Residual + LayerNorm2
                %residual2 = stablehlo.add %ln1_out, %ffn_out : tensor<8x128x768xf32>
                %sum2 = stablehlo.reduce %residual2, %zero applies stablehlo.add across dimensions = [2] : (tensor<8x128x768xf32>, tensor<f32>) -> tensor<8x128xf32>
                %mean2 = stablehlo.divide %sum2, %hidden_bc1 : tensor<8x128xf32>
                %mean2_bc = stablehlo.broadcast_in_dim %mean2, dims = [0, 1] : (tensor<8x128xf32>) -> tensor<8x128x768xf32>
                %centered2 = stablehlo.subtract %residual2, %mean2_bc : tensor<8x128x768xf32>
                %sq2 = stablehlo.multiply %centered2, %centered2 : tensor<8x128x768xf32>
                %var_sum2 = stablehlo.reduce %sq2, %zero applies stablehlo.add across dimensions = [2] : (tensor<8x128x768xf32>, tensor<f32>) -> tensor<8x128xf32>
                %var2 = stablehlo.divide %var_sum2, %hidden_bc1 : tensor<8x128xf32>
                %var_eps2 = stablehlo.add %var2, %eps_bc1 : tensor<8x128xf32>
                %rstd2 = stablehlo.rsqrt %var_eps2 : tensor<8x128xf32>
                %rstd2_bc = stablehlo.broadcast_in_dim %rstd2, dims = [0, 1] : (tensor<8x128xf32>) -> tensor<8x128x768xf32>
                %normalized2 = stablehlo.multiply %centered2, %rstd2_bc : tensor<8x128x768xf32>
                %gamma2_bc = stablehlo.broadcast_in_dim %ln2_gamma, dims = [2] : (tensor<768xf32>) -> tensor<8x128x768xf32>
                %beta2_bc = stablehlo.broadcast_in_dim %ln2_beta, dims = [2] : (tensor<768xf32>) -> tensor<8x128x768xf32>
                %scaled2 = stablehlo.multiply %normalized2, %gamma2_bc : tensor<8x128x768xf32>
                %out = stablehlo.add %scaled2, %beta2_bc : tensor<8x128x768xf32>

                return %out : tensor<8x128x768xf32>
              }
            }
            """,
            expectedOps: 200
        ))

        // COMP-005: Multiple Transformer Blocks (~500 ops)
        // For this benchmark, we'll chain multiple simpler operations to simulate a larger program
        benchmarks.append(CompilerBenchmark(
            id: "COMP-005",
            name: "Compile Large Model (~500 ops)",
            operation: "compile",
            configuration: ["ops": "~500", "target": "<5s"],
            mlirProgram: generateLargeModelMLIR(),
            expectedOps: 500
        ))

        return benchmarks
    }

    /// Generate MLIR for a large model with ~500 operations.
    private static func generateLargeModelMLIR() -> String {
        // Generate a program with many chained operations to test compilation scaling
        var mlir = """
        module @large_model {
          func.func @main(%x: tensor<8x1024xf32>
        """

        // Add weight parameters for multiple layers
        for i in 0..<10 {
            mlir += ", %w\(i): tensor<1024x1024xf32>, %b\(i): tensor<1024xf32>"
        }

        mlir += ") -> (tensor<8x1024xf32>) {\n"
        mlir += "    %zero = stablehlo.constant dense<0.0> : tensor<8x1024xf32>\n"

        // Generate a chain of operations
        var currentVar = "%x"
        for i in 0..<10 {
            let mm = "%mm\(i)"
            let bias = "%bias\(i)"
            let relu = "%relu\(i)"

            mlir += """
                // Layer \(i)
                \(mm) = stablehlo.dot_general \(currentVar), %w\(i), #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<8x1024xf32>, tensor<1024x1024xf32>) -> tensor<8x1024xf32>
                %b\(i)_bc = stablehlo.broadcast_in_dim %b\(i), dims = [1] : (tensor<1024xf32>) -> tensor<8x1024xf32>
                \(bias) = stablehlo.add \(mm), %b\(i)_bc : tensor<8x1024xf32>
                \(relu) = stablehlo.maximum \(bias), %zero : tensor<8x1024xf32>

            """

            // Add some extra elementwise ops to increase op count
            let exp = "%exp\(i)"
            let tanh_op = "%tanh\(i)"
            let mul = "%mul\(i)"
            let add = "%add\(i)"

            mlir += """
                \(exp) = stablehlo.exponential \(relu) : tensor<8x1024xf32>
                \(tanh_op) = stablehlo.tanh \(exp) : tensor<8x1024xf32>
                \(mul) = stablehlo.multiply \(tanh_op), \(relu) : tensor<8x1024xf32>
                \(add) = stablehlo.add \(mul), \(relu) : tensor<8x1024xf32>

            """
            currentVar = add
        }

        mlir += "    return \(currentVar) : tensor<8x1024xf32>\n"
        mlir += "  }\n"
        mlir += "}\n"

        return mlir
    }
}

// MARK: - Compiler Benchmark Runner

/// Standalone runner for compiler benchmarks.
public struct CompilerBenchmarkRunner: Sendable {

    /// Number of iterations for measuring compilation time.
    public let iterations: Int

    public init(iterations: Int = 10) {
        self.iterations = iterations
    }

    /// Run a compiler benchmark, measuring compilation time rather than execution time.
    public func run(_ benchmark: CompilerBenchmarks.CompilerBenchmark) throws -> CompilerBenchmarkResult {
        let client = try Client.create()

        var compileTimes: [Double] = []

        // Measure compilation time multiple times
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let _ = try client.compile(benchmark.mlirProgram)
            let end = CFAbsoluteTimeGetCurrent()
            compileTimes.append(end - start)
        }

        // Use TimingStatistics for consistency with other benchmarks
        let stats = TimingStatistics(samples: compileTimes)

        return CompilerBenchmarkResult(
            id: benchmark.id,
            name: benchmark.name,
            category: benchmark.category,
            operation: benchmark.operation,
            expectedOps: benchmark.expectedOps,
            compileTime: stats,
            iterations: iterations
        )
    }

    /// Run all compiler benchmarks.
    public func runAll() throws -> [CompilerBenchmarkResult] {
        var results: [CompilerBenchmarkResult] = []
        for benchmark in CompilerBenchmarks.all() {
            let result = try run(benchmark)
            results.append(result)
        }
        return results
    }
}

/// Result of a compiler benchmark.
public struct CompilerBenchmarkResult: Sendable {
    public let id: String
    public let name: String
    public let category: String
    public let operation: String
    public let expectedOps: Int
    public let compileTime: TimingStatistics
    public let iterations: Int

    public init(
        id: String,
        name: String,
        category: String,
        operation: String,
        expectedOps: Int,
        compileTime: TimingStatistics,
        iterations: Int
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.operation = operation
        self.expectedOps = expectedOps
        self.compileTime = compileTime
        self.iterations = iterations
    }

    /// Format result as a string.
    public func formatted() -> String {
        let meanMs = compileTime.mean * 1000
        let stdMs = compileTime.stdDev * 1000
        return "\(id): \(String(format: "%.2f", meanMs))ms ± \(String(format: "%.2f", stdMs))ms (~\(expectedOps) ops)"
    }
}
