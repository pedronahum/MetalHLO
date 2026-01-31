// ModelBenchmarks.swift
// MetalHLO Benchmarks
//
// Model-level benchmark definitions based on the benchmark proposal.
// These benchmarks test complete model components and end-to-end pipelines.

import Foundation
import MetalHLO

/// Factory for creating model-level benchmarks.
public enum ModelBenchmarks {

    // MARK: - MLP Inference Benchmarks

    /// Multi-Layer Perceptron inference benchmarks.
    public static func mlpInferenceBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // MLP-INF-001: Simple 2-layer MLP, batch 1
        // Architecture: 784 -> 256 -> 10 with ReLU
        benchmarks.append(SimpleBenchmark(
            id: "MLP-INF-001",
            name: "MLP 784->256->10 (BS=1)",
            category: "model_mlp",
            operation: "mlp_inference",
            configuration: [
                "batch_size": "1",
                "layers": "2",
                "architecture": "784->256->10",
                "activation": "relu"
            ],
            mlirProgram: """
            module @mlp_2layer_bs1 {
              func.func @main(%x: tensor<1x784xf32>, %w1: tensor<784x256xf32>, %b1: tensor<256xf32>, %w2: tensor<256x10xf32>, %b2: tensor<10xf32>) -> (tensor<1x10xf32>) {
                // Layer 1: Linear + ReLU
                %h1_mm = stablehlo.dot_general %x, %w1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1x784xf32>, tensor<784x256xf32>) -> tensor<1x256xf32>
                %b1_bc = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
                %h1_bias = stablehlo.add %h1_mm, %b1_bc : tensor<1x256xf32>
                %zero = stablehlo.constant dense<0.0> : tensor<1x256xf32>
                %h1 = stablehlo.maximum %h1_bias, %zero : tensor<1x256xf32>

                // Layer 2: Linear (no activation for logits)
                %out_mm = stablehlo.dot_general %h1, %w2, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1x256xf32>, tensor<256x10xf32>) -> tensor<1x10xf32>
                %b2_bc = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<10xf32>) -> tensor<1x10xf32>
                %out = stablehlo.add %out_mm, %b2_bc : tensor<1x10xf32>

                return %out : tensor<1x10xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [1, 784])
                let w1 = try gen.createNormalFloat32Buffer(client: client, shape: [784, 256], mean: 0, stdDev: 0.05)
                let b1 = try gen.createZerosBuffer(client: client, shape: [256], elementType: .float32)
                let w2 = try gen.createNormalFloat32Buffer(client: client, shape: [256, 10], mean: 0, stdDev: 0.05)
                let b2 = try gen.createZerosBuffer(client: client, shape: [10], elementType: .float32)
                return [x, w1, b1, w2, b2]
            },
            throughputCalculator: { timing in
                // FLOPs: 2*1*784*256 + 2*1*256*10 = 406,528
                let flops = 2 * 1 * 784 * 256 + 2 * 1 * 256 * 10
                let gflops = FLOPSCalculator.gflops(flops: Double(flops), timeSeconds: timing.gpuTime)
                return ThroughputMetrics(
                    opsPerSecond: 1.0 / timing.gpuTime,
                    flops: gflops * 1e9,
                    elementsPerSecond: 1.0 / timing.gpuTime
                )
            }
        ))

        // MLP-INF-002: Simple 2-layer MLP, batch 32
        benchmarks.append(SimpleBenchmark(
            id: "MLP-INF-002",
            name: "MLP 784->256->10 (BS=32)",
            category: "model_mlp",
            operation: "mlp_inference",
            configuration: [
                "batch_size": "32",
                "layers": "2",
                "architecture": "784->256->10",
                "activation": "relu"
            ],
            mlirProgram: """
            module @mlp_2layer_bs32 {
              func.func @main(%x: tensor<32x784xf32>, %w1: tensor<784x256xf32>, %b1: tensor<256xf32>, %w2: tensor<256x10xf32>, %b2: tensor<10xf32>) -> (tensor<32x10xf32>) {
                // Layer 1: Linear + ReLU
                %h1_mm = stablehlo.dot_general %x, %w1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x784xf32>, tensor<784x256xf32>) -> tensor<32x256xf32>
                %b1_bc = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256xf32>
                %h1_bias = stablehlo.add %h1_mm, %b1_bc : tensor<32x256xf32>
                %zero = stablehlo.constant dense<0.0> : tensor<32x256xf32>
                %h1 = stablehlo.maximum %h1_bias, %zero : tensor<32x256xf32>

                // Layer 2: Linear
                %out_mm = stablehlo.dot_general %h1, %w2, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x256xf32>, tensor<256x10xf32>) -> tensor<32x10xf32>
                %b2_bc = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
                %out = stablehlo.add %out_mm, %b2_bc : tensor<32x10xf32>

                return %out : tensor<32x10xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [32, 784])
                let w1 = try gen.createNormalFloat32Buffer(client: client, shape: [784, 256], mean: 0, stdDev: 0.05)
                let b1 = try gen.createZerosBuffer(client: client, shape: [256], elementType: .float32)
                let w2 = try gen.createNormalFloat32Buffer(client: client, shape: [256, 10], mean: 0, stdDev: 0.05)
                let b2 = try gen.createZerosBuffer(client: client, shape: [10], elementType: .float32)
                return [x, w1, b1, w2, b2]
            },
            throughputCalculator: { timing in
                let flops = 2 * 32 * 784 * 256 + 2 * 32 * 256 * 10
                let gflops = FLOPSCalculator.gflops(flops: Double(flops), timeSeconds: timing.gpuTime)
                return ThroughputMetrics(
                    opsPerSecond: 32.0 / timing.gpuTime,
                    flops: gflops * 1e9,
                    elementsPerSecond: 32.0 / timing.gpuTime
                )
            }
        ))

        // MLP-INF-003: Simple 2-layer MLP, batch 128
        benchmarks.append(SimpleBenchmark(
            id: "MLP-INF-003",
            name: "MLP 784->256->10 (BS=128)",
            category: "model_mlp",
            operation: "mlp_inference",
            configuration: [
                "batch_size": "128",
                "layers": "2",
                "architecture": "784->256->10",
                "activation": "relu"
            ],
            mlirProgram: """
            module @mlp_2layer_bs128 {
              func.func @main(%x: tensor<128x784xf32>, %w1: tensor<784x256xf32>, %b1: tensor<256xf32>, %w2: tensor<256x10xf32>, %b2: tensor<10xf32>) -> (tensor<128x10xf32>) {
                // Layer 1: Linear + ReLU
                %h1_mm = stablehlo.dot_general %x, %w1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x784xf32>, tensor<784x256xf32>) -> tensor<128x256xf32>
                %b1_bc = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<256xf32>) -> tensor<128x256xf32>
                %h1_bias = stablehlo.add %h1_mm, %b1_bc : tensor<128x256xf32>
                %zero = stablehlo.constant dense<0.0> : tensor<128x256xf32>
                %h1 = stablehlo.maximum %h1_bias, %zero : tensor<128x256xf32>

                // Layer 2: Linear
                %out_mm = stablehlo.dot_general %h1, %w2, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x256xf32>, tensor<256x10xf32>) -> tensor<128x10xf32>
                %b2_bc = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<10xf32>) -> tensor<128x10xf32>
                %out = stablehlo.add %out_mm, %b2_bc : tensor<128x10xf32>

                return %out : tensor<128x10xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [128, 784])
                let w1 = try gen.createNormalFloat32Buffer(client: client, shape: [784, 256], mean: 0, stdDev: 0.05)
                let b1 = try gen.createZerosBuffer(client: client, shape: [256], elementType: .float32)
                let w2 = try gen.createNormalFloat32Buffer(client: client, shape: [256, 10], mean: 0, stdDev: 0.05)
                let b2 = try gen.createZerosBuffer(client: client, shape: [10], elementType: .float32)
                return [x, w1, b1, w2, b2]
            },
            throughputCalculator: { timing in
                let flops = 2 * 128 * 784 * 256 + 2 * 128 * 256 * 10
                let gflops = FLOPSCalculator.gflops(flops: Double(flops), timeSeconds: timing.gpuTime)
                return ThroughputMetrics(
                    opsPerSecond: 128.0 / timing.gpuTime,
                    flops: gflops * 1e9,
                    elementsPerSecond: 128.0 / timing.gpuTime
                )
            }
        ))

        // MLP-INF-004: 4-layer MLP, batch 32
        // Architecture: 784 -> 512 -> 256 -> 128 -> 10
        benchmarks.append(SimpleBenchmark(
            id: "MLP-INF-004",
            name: "MLP 784->512->256->128->10 (BS=32)",
            category: "model_mlp",
            operation: "mlp_inference",
            configuration: [
                "batch_size": "32",
                "layers": "4",
                "architecture": "784->512->256->128->10",
                "activation": "relu"
            ],
            mlirProgram: """
            module @mlp_4layer_bs32 {
              func.func @main(%x: tensor<32x784xf32>, %w1: tensor<784x512xf32>, %b1: tensor<512xf32>, %w2: tensor<512x256xf32>, %b2: tensor<256xf32>, %w3: tensor<256x128xf32>, %b3: tensor<128xf32>, %w4: tensor<128x10xf32>, %b4: tensor<10xf32>) -> (tensor<32x10xf32>) {
                %zero1 = stablehlo.constant dense<0.0> : tensor<32x512xf32>
                %zero2 = stablehlo.constant dense<0.0> : tensor<32x256xf32>
                %zero3 = stablehlo.constant dense<0.0> : tensor<32x128xf32>

                // Layer 1
                %h1_mm = stablehlo.dot_general %x, %w1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x784xf32>, tensor<784x512xf32>) -> tensor<32x512xf32>
                %b1_bc = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512xf32>
                %h1_bias = stablehlo.add %h1_mm, %b1_bc : tensor<32x512xf32>
                %h1 = stablehlo.maximum %h1_bias, %zero1 : tensor<32x512xf32>

                // Layer 2
                %h2_mm = stablehlo.dot_general %h1, %w2, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x512xf32>, tensor<512x256xf32>) -> tensor<32x256xf32>
                %b2_bc = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256xf32>
                %h2_bias = stablehlo.add %h2_mm, %b2_bc : tensor<32x256xf32>
                %h2 = stablehlo.maximum %h2_bias, %zero2 : tensor<32x256xf32>

                // Layer 3
                %h3_mm = stablehlo.dot_general %h2, %w3, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x256xf32>, tensor<256x128xf32>) -> tensor<32x128xf32>
                %b3_bc = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<128xf32>) -> tensor<32x128xf32>
                %h3_bias = stablehlo.add %h3_mm, %b3_bc : tensor<32x128xf32>
                %h3 = stablehlo.maximum %h3_bias, %zero3 : tensor<32x128xf32>

                // Layer 4
                %out_mm = stablehlo.dot_general %h3, %w4, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x128xf32>, tensor<128x10xf32>) -> tensor<32x10xf32>
                %b4_bc = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
                %out = stablehlo.add %out_mm, %b4_bc : tensor<32x10xf32>

                return %out : tensor<32x10xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [32, 784])
                let w1 = try gen.createNormalFloat32Buffer(client: client, shape: [784, 512], mean: 0, stdDev: 0.05)
                let b1 = try gen.createZerosBuffer(client: client, shape: [512], elementType: .float32)
                let w2 = try gen.createNormalFloat32Buffer(client: client, shape: [512, 256], mean: 0, stdDev: 0.05)
                let b2 = try gen.createZerosBuffer(client: client, shape: [256], elementType: .float32)
                let w3 = try gen.createNormalFloat32Buffer(client: client, shape: [256, 128], mean: 0, stdDev: 0.05)
                let b3 = try gen.createZerosBuffer(client: client, shape: [128], elementType: .float32)
                let w4 = try gen.createNormalFloat32Buffer(client: client, shape: [128, 10], mean: 0, stdDev: 0.05)
                let b4 = try gen.createZerosBuffer(client: client, shape: [10], elementType: .float32)
                return [x, w1, b1, w2, b2, w3, b3, w4, b4]
            },
            throughputCalculator: { timing in
                let flops = 2 * 32 * (784 * 512 + 512 * 256 + 256 * 128 + 128 * 10)
                let gflops = FLOPSCalculator.gflops(flops: Double(flops), timeSeconds: timing.gpuTime)
                return ThroughputMetrics(
                    opsPerSecond: 32.0 / timing.gpuTime,
                    flops: gflops * 1e9,
                    elementsPerSecond: 32.0 / timing.gpuTime
                )
            }
        ))

        // MLP-INF-005: Large MLP with GELU, batch 32
        // Architecture: 768 -> 3072 -> 768 (Transformer FFN style)
        benchmarks.append(SimpleBenchmark(
            id: "MLP-INF-005",
            name: "MLP FFN 768->3072->768 GELU (BS=32)",
            category: "model_mlp",
            operation: "mlp_inference",
            configuration: [
                "batch_size": "32",
                "layers": "2",
                "architecture": "768->3072->768",
                "activation": "gelu"
            ],
            mlirProgram: """
            module @mlp_ffn_gelu {
              func.func @main(%x: tensor<32x768xf32>, %w1: tensor<768x3072xf32>, %b1: tensor<3072xf32>, %w2: tensor<3072x768xf32>, %b2: tensor<768xf32>) -> (tensor<32x768xf32>) {
                // Layer 1: Linear
                %h1_mm = stablehlo.dot_general %x, %w1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x768xf32>, tensor<768x3072xf32>) -> tensor<32x3072xf32>
                %b1_bc = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072xf32>
                %h1_bias = stablehlo.add %h1_mm, %b1_bc : tensor<32x3072xf32>

                // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                // Simplified: using logistic (sigmoid) approximation for speed
                %h1 = stablehlo.logistic %h1_bias : tensor<32x3072xf32>
                %h1_gelu = stablehlo.multiply %h1_bias, %h1 : tensor<32x3072xf32>

                // Layer 2: Linear
                %out_mm = stablehlo.dot_general %h1_gelu, %w2, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x3072xf32>, tensor<3072x768xf32>) -> tensor<32x768xf32>
                %b2_bc = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<768xf32>) -> tensor<32x768xf32>
                %out = stablehlo.add %out_mm, %b2_bc : tensor<32x768xf32>

                return %out : tensor<32x768xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [32, 768])
                let w1 = try gen.createNormalFloat32Buffer(client: client, shape: [768, 3072], mean: 0, stdDev: 0.02)
                let b1 = try gen.createZerosBuffer(client: client, shape: [3072], elementType: .float32)
                let w2 = try gen.createNormalFloat32Buffer(client: client, shape: [3072, 768], mean: 0, stdDev: 0.02)
                let b2 = try gen.createZerosBuffer(client: client, shape: [768], elementType: .float32)
                return [x, w1, b1, w2, b2]
            },
            throughputCalculator: { timing in
                let flops = 2 * 32 * (768 * 3072 + 3072 * 768)
                let gflops = FLOPSCalculator.gflops(flops: Double(flops), timeSeconds: timing.gpuTime)
                return ThroughputMetrics(
                    opsPerSecond: 32.0 / timing.gpuTime,
                    flops: gflops * 1e9,
                    elementsPerSecond: 32.0 / timing.gpuTime
                )
            }
        ))

        return benchmarks
    }

    // MARK: - CNN Inference Benchmarks

    /// Convolutional Neural Network inference benchmarks.
    public static func cnnInferenceBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // CNN-INF-001: LeNet-like, batch 1
        // Simple: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC
        benchmarks.append(SimpleBenchmark(
            id: "CNN-INF-001",
            name: "CNN LeNet-like 28x28 (BS=1)",
            category: "model_cnn",
            operation: "cnn_inference",
            configuration: [
                "batch_size": "1",
                "input": "28x28x1",
                "architecture": "LeNet-like"
            ],
            mlirProgram: """
            module @lenet_bs1 {
              func.func @main(%input: tensor<1x28x28x1xf32>, %conv1_w: tensor<5x5x1x6xf32>, %conv1_b: tensor<6xf32>, %conv2_w: tensor<5x5x6x16xf32>, %conv2_b: tensor<16xf32>, %fc_w: tensor<256x10xf32>, %fc_b: tensor<10xf32>) -> (tensor<1x10xf32>) {
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>

                // Conv1: 28x28x1 -> 24x24x6 (valid padding)
                %conv1 = stablehlo.convolution %input, %conv1_w window_strides = [1, 1], feature_group_count = 1 : (tensor<1x28x28x1xf32>, tensor<5x5x1x6xf32>) -> tensor<1x24x24x6xf32>
                %conv1_b_bc = stablehlo.broadcast_in_dim %conv1_b, dims = [3] : (tensor<6xf32>) -> tensor<1x24x24x6xf32>
                %conv1_bias = stablehlo.add %conv1, %conv1_b_bc : tensor<1x24x24x6xf32>
                %relu1_zero = stablehlo.constant dense<0.0> : tensor<1x24x24x6xf32>
                %relu1 = stablehlo.maximum %conv1_bias, %relu1_zero : tensor<1x24x24x6xf32>

                // MaxPool1: 24x24x6 -> 12x12x6
                %pool1 = stablehlo.reduce_window %relu1, %neg_inf applies stablehlo.maximum window_dimensions = [1, 2, 2, 1], window_strides = [1, 2, 2, 1], padding = [[0, 0], [0, 0], [0, 0], [0, 0]] : (tensor<1x24x24x6xf32>, tensor<f32>) -> tensor<1x12x12x6xf32>

                // Conv2: 12x12x6 -> 8x8x16
                %conv2 = stablehlo.convolution %pool1, %conv2_w window_strides = [1, 1], feature_group_count = 1 : (tensor<1x12x12x6xf32>, tensor<5x5x6x16xf32>) -> tensor<1x8x8x16xf32>
                %conv2_b_bc = stablehlo.broadcast_in_dim %conv2_b, dims = [3] : (tensor<16xf32>) -> tensor<1x8x8x16xf32>
                %conv2_bias = stablehlo.add %conv2, %conv2_b_bc : tensor<1x8x8x16xf32>
                %relu2_zero = stablehlo.constant dense<0.0> : tensor<1x8x8x16xf32>
                %relu2 = stablehlo.maximum %conv2_bias, %relu2_zero : tensor<1x8x8x16xf32>

                // MaxPool2: 8x8x16 -> 4x4x16
                %pool2 = stablehlo.reduce_window %relu2, %neg_inf applies stablehlo.maximum window_dimensions = [1, 2, 2, 1], window_strides = [1, 2, 2, 1], padding = [[0, 0], [0, 0], [0, 0], [0, 0]] : (tensor<1x8x8x16xf32>, tensor<f32>) -> tensor<1x4x4x16xf32>

                // Flatten: 1x4x4x16 -> 1x256
                %flat = stablehlo.reshape %pool2 : (tensor<1x4x4x16xf32>) -> tensor<1x256xf32>

                // FC: 256 -> 10
                %fc = stablehlo.dot_general %flat, %fc_w, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1x256xf32>, tensor<256x10xf32>) -> tensor<1x10xf32>
                %fc_b_bc = stablehlo.broadcast_in_dim %fc_b, dims = [1] : (tensor<10xf32>) -> tensor<1x10xf32>
                %out = stablehlo.add %fc, %fc_b_bc : tensor<1x10xf32>

                return %out : tensor<1x10xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let input = try gen.createUniformFloat32Buffer(client: client, shape: [1, 28, 28, 1])
                let conv1_w = try gen.createNormalFloat32Buffer(client: client, shape: [5, 5, 1, 6], mean: 0, stdDev: 0.1)
                let conv1_b = try gen.createZerosBuffer(client: client, shape: [6], elementType: .float32)
                let conv2_w = try gen.createNormalFloat32Buffer(client: client, shape: [5, 5, 6, 16], mean: 0, stdDev: 0.1)
                let conv2_b = try gen.createZerosBuffer(client: client, shape: [16], elementType: .float32)
                let fc_w = try gen.createNormalFloat32Buffer(client: client, shape: [256, 10], mean: 0, stdDev: 0.1)
                let fc_b = try gen.createZerosBuffer(client: client, shape: [10], elementType: .float32)
                return [input, conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b]
            }
        ))

        // CNN-INF-002: LeNet-like, batch 32
        benchmarks.append(SimpleBenchmark(
            id: "CNN-INF-002",
            name: "CNN LeNet-like 28x28 (BS=32)",
            category: "model_cnn",
            operation: "cnn_inference",
            configuration: [
                "batch_size": "32",
                "input": "28x28x1",
                "architecture": "LeNet-like"
            ],
            mlirProgram: """
            module @lenet_bs32 {
              func.func @main(%input: tensor<32x28x28x1xf32>, %conv1_w: tensor<5x5x1x6xf32>, %conv1_b: tensor<6xf32>, %conv2_w: tensor<5x5x6x16xf32>, %conv2_b: tensor<16xf32>, %fc_w: tensor<256x10xf32>, %fc_b: tensor<10xf32>) -> (tensor<32x10xf32>) {
                %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>

                // Conv1: 32x28x28x1 -> 32x24x24x6
                %conv1 = stablehlo.convolution %input, %conv1_w window_strides = [1, 1], feature_group_count = 1 : (tensor<32x28x28x1xf32>, tensor<5x5x1x6xf32>) -> tensor<32x24x24x6xf32>
                %conv1_b_bc = stablehlo.broadcast_in_dim %conv1_b, dims = [3] : (tensor<6xf32>) -> tensor<32x24x24x6xf32>
                %conv1_bias = stablehlo.add %conv1, %conv1_b_bc : tensor<32x24x24x6xf32>
                %relu1_zero = stablehlo.constant dense<0.0> : tensor<32x24x24x6xf32>
                %relu1 = stablehlo.maximum %conv1_bias, %relu1_zero : tensor<32x24x24x6xf32>

                // MaxPool1: 32x24x24x6 -> 32x12x12x6
                %pool1 = stablehlo.reduce_window %relu1, %neg_inf applies stablehlo.maximum window_dimensions = [1, 2, 2, 1], window_strides = [1, 2, 2, 1], padding = [[0, 0], [0, 0], [0, 0], [0, 0]] : (tensor<32x24x24x6xf32>, tensor<f32>) -> tensor<32x12x12x6xf32>

                // Conv2: 32x12x12x6 -> 32x8x8x16
                %conv2 = stablehlo.convolution %pool1, %conv2_w window_strides = [1, 1], feature_group_count = 1 : (tensor<32x12x12x6xf32>, tensor<5x5x6x16xf32>) -> tensor<32x8x8x16xf32>
                %conv2_b_bc = stablehlo.broadcast_in_dim %conv2_b, dims = [3] : (tensor<16xf32>) -> tensor<32x8x8x16xf32>
                %conv2_bias = stablehlo.add %conv2, %conv2_b_bc : tensor<32x8x8x16xf32>
                %relu2_zero = stablehlo.constant dense<0.0> : tensor<32x8x8x16xf32>
                %relu2 = stablehlo.maximum %conv2_bias, %relu2_zero : tensor<32x8x8x16xf32>

                // MaxPool2: 32x8x8x16 -> 32x4x4x16
                %pool2 = stablehlo.reduce_window %relu2, %neg_inf applies stablehlo.maximum window_dimensions = [1, 2, 2, 1], window_strides = [1, 2, 2, 1], padding = [[0, 0], [0, 0], [0, 0], [0, 0]] : (tensor<32x8x8x16xf32>, tensor<f32>) -> tensor<32x4x4x16xf32>

                // Flatten and FC
                %flat = stablehlo.reshape %pool2 : (tensor<32x4x4x16xf32>) -> tensor<32x256xf32>
                %fc = stablehlo.dot_general %flat, %fc_w, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<32x256xf32>, tensor<256x10xf32>) -> tensor<32x10xf32>
                %fc_b_bc = stablehlo.broadcast_in_dim %fc_b, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
                %out = stablehlo.add %fc, %fc_b_bc : tensor<32x10xf32>

                return %out : tensor<32x10xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let input = try gen.createUniformFloat32Buffer(client: client, shape: [32, 28, 28, 1])
                let conv1_w = try gen.createNormalFloat32Buffer(client: client, shape: [5, 5, 1, 6], mean: 0, stdDev: 0.1)
                let conv1_b = try gen.createZerosBuffer(client: client, shape: [6], elementType: .float32)
                let conv2_w = try gen.createNormalFloat32Buffer(client: client, shape: [5, 5, 6, 16], mean: 0, stdDev: 0.1)
                let conv2_b = try gen.createZerosBuffer(client: client, shape: [16], elementType: .float32)
                let fc_w = try gen.createNormalFloat32Buffer(client: client, shape: [256, 10], mean: 0, stdDev: 0.1)
                let fc_b = try gen.createZerosBuffer(client: client, shape: [10], elementType: .float32)
                return [input, conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b]
            }
        ))

        // CNN-INF-003: ResNet-like block, batch 1
        // Single residual block: Conv-BN-ReLU-Conv-BN + skip connection
        benchmarks.append(SimpleBenchmark(
            id: "CNN-INF-003",
            name: "CNN ResNet Block 56x56x64 (BS=1)",
            category: "model_cnn",
            operation: "cnn_inference",
            configuration: [
                "batch_size": "1",
                "input": "56x56x64",
                "architecture": "ResNet-block"
            ],
            mlirProgram: """
            module @resnet_block_bs1 {
              func.func @main(%input: tensor<1x56x56x64xf32>, %conv1_w: tensor<3x3x64x64xf32>, %bn1_scale: tensor<64xf32>, %bn1_offset: tensor<64xf32>, %bn1_mean: tensor<64xf32>, %bn1_var: tensor<64xf32>, %conv2_w: tensor<3x3x64x64xf32>, %bn2_scale: tensor<64xf32>, %bn2_offset: tensor<64xf32>, %bn2_mean: tensor<64xf32>, %bn2_var: tensor<64xf32>) -> (tensor<1x56x56x64xf32>) {

                // Conv1 3x3 same padding (pad input to get same output size)
                %padded1 = stablehlo.pad %input, %cst, low = [0, 1, 1, 0], high = [0, 1, 1, 0], interior = [0, 0, 0, 0] : (tensor<1x56x56x64xf32>, tensor<f32>) -> tensor<1x58x58x64xf32> {
                  %cst = stablehlo.constant dense<0.0> : tensor<f32>
                }
                %conv1 = stablehlo.convolution %input, %conv1_w padding = [[1, 1], [1, 1]], window_strides = [1, 1], feature_group_count = 1 : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>

                // BatchNorm1
                %bn1 = stablehlo.batch_norm_inference %conv1, %bn1_scale, %bn1_offset, %bn1_mean, %bn1_var, epsilon = 1.0e-05, feature_index = 3 : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>

                // ReLU
                %zero1 = stablehlo.constant dense<0.0> : tensor<1x56x56x64xf32>
                %relu1 = stablehlo.maximum %bn1, %zero1 : tensor<1x56x56x64xf32>

                // Conv2 3x3 same padding
                %conv2 = stablehlo.convolution %relu1, %conv2_w padding = [[1, 1], [1, 1]], window_strides = [1, 1], feature_group_count = 1 : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>

                // BatchNorm2
                %bn2 = stablehlo.batch_norm_inference %conv2, %bn2_scale, %bn2_offset, %bn2_mean, %bn2_var, epsilon = 1.0e-05, feature_index = 3 : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>

                // Skip connection + ReLU
                %residual = stablehlo.add %bn2, %input : tensor<1x56x56x64xf32>
                %zero2 = stablehlo.constant dense<0.0> : tensor<1x56x56x64xf32>
                %out = stablehlo.maximum %residual, %zero2 : tensor<1x56x56x64xf32>

                return %out : tensor<1x56x56x64xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let input = try gen.createUniformFloat32Buffer(client: client, shape: [1, 56, 56, 64])
                let conv1_w = try gen.createNormalFloat32Buffer(client: client, shape: [3, 3, 64, 64], mean: 0, stdDev: 0.05)
                let bn1_scale = try client.createBuffer([Float](repeating: 1.0, count: 64), shape: [64])
                let bn1_offset = try gen.createZerosBuffer(client: client, shape: [64], elementType: .float32)
                let bn1_mean = try gen.createZerosBuffer(client: client, shape: [64], elementType: .float32)
                let bn1_var = try client.createBuffer([Float](repeating: 1.0, count: 64), shape: [64])
                let conv2_w = try gen.createNormalFloat32Buffer(client: client, shape: [3, 3, 64, 64], mean: 0, stdDev: 0.05)
                let bn2_scale = try client.createBuffer([Float](repeating: 1.0, count: 64), shape: [64])
                let bn2_offset = try gen.createZerosBuffer(client: client, shape: [64], elementType: .float32)
                let bn2_mean = try gen.createZerosBuffer(client: client, shape: [64], elementType: .float32)
                let bn2_var = try client.createBuffer([Float](repeating: 1.0, count: 64), shape: [64])
                return [input, conv1_w, bn1_scale, bn1_offset, bn1_mean, bn1_var, conv2_w, bn2_scale, bn2_offset, bn2_mean, bn2_var]
            }
        ))

        // CNN-INF-004: ResNet-like block, batch 8
        benchmarks.append(SimpleBenchmark(
            id: "CNN-INF-004",
            name: "CNN ResNet Block 56x56x64 (BS=8)",
            category: "model_cnn",
            operation: "cnn_inference",
            configuration: [
                "batch_size": "8",
                "input": "56x56x64",
                "architecture": "ResNet-block"
            ],
            mlirProgram: """
            module @resnet_block_bs8 {
              func.func @main(%input: tensor<8x56x56x64xf32>, %conv1_w: tensor<3x3x64x64xf32>, %bn1_scale: tensor<64xf32>, %bn1_offset: tensor<64xf32>, %bn1_mean: tensor<64xf32>, %bn1_var: tensor<64xf32>, %conv2_w: tensor<3x3x64x64xf32>, %bn2_scale: tensor<64xf32>, %bn2_offset: tensor<64xf32>, %bn2_mean: tensor<64xf32>, %bn2_var: tensor<64xf32>) -> (tensor<8x56x56x64xf32>) {

                // Conv1 3x3 same padding
                %conv1 = stablehlo.convolution %input, %conv1_w padding = [[1, 1], [1, 1]], window_strides = [1, 1], feature_group_count = 1 : (tensor<8x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<8x56x56x64xf32>
                %bn1 = stablehlo.batch_norm_inference %conv1, %bn1_scale, %bn1_offset, %bn1_mean, %bn1_var, epsilon = 1.0e-05, feature_index = 3 : (tensor<8x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<8x56x56x64xf32>
                %zero1 = stablehlo.constant dense<0.0> : tensor<8x56x56x64xf32>
                %relu1 = stablehlo.maximum %bn1, %zero1 : tensor<8x56x56x64xf32>

                // Conv2 3x3 same padding
                %conv2 = stablehlo.convolution %relu1, %conv2_w padding = [[1, 1], [1, 1]], window_strides = [1, 1], feature_group_count = 1 : (tensor<8x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<8x56x56x64xf32>
                %bn2 = stablehlo.batch_norm_inference %conv2, %bn2_scale, %bn2_offset, %bn2_mean, %bn2_var, epsilon = 1.0e-05, feature_index = 3 : (tensor<8x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<8x56x56x64xf32>

                // Skip + ReLU
                %residual = stablehlo.add %bn2, %input : tensor<8x56x56x64xf32>
                %zero2 = stablehlo.constant dense<0.0> : tensor<8x56x56x64xf32>
                %out = stablehlo.maximum %residual, %zero2 : tensor<8x56x56x64xf32>

                return %out : tensor<8x56x56x64xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let input = try gen.createUniformFloat32Buffer(client: client, shape: [8, 56, 56, 64])
                let conv1_w = try gen.createNormalFloat32Buffer(client: client, shape: [3, 3, 64, 64], mean: 0, stdDev: 0.05)
                let bn1_scale = try client.createBuffer([Float](repeating: 1.0, count: 64), shape: [64])
                let bn1_offset = try gen.createZerosBuffer(client: client, shape: [64], elementType: .float32)
                let bn1_mean = try gen.createZerosBuffer(client: client, shape: [64], elementType: .float32)
                let bn1_var = try client.createBuffer([Float](repeating: 1.0, count: 64), shape: [64])
                let conv2_w = try gen.createNormalFloat32Buffer(client: client, shape: [3, 3, 64, 64], mean: 0, stdDev: 0.05)
                let bn2_scale = try client.createBuffer([Float](repeating: 1.0, count: 64), shape: [64])
                let bn2_offset = try gen.createZerosBuffer(client: client, shape: [64], elementType: .float32)
                let bn2_mean = try gen.createZerosBuffer(client: client, shape: [64], elementType: .float32)
                let bn2_var = try client.createBuffer([Float](repeating: 1.0, count: 64), shape: [64])
                return [input, conv1_w, bn1_scale, bn1_offset, bn1_mean, bn1_var, conv2_w, bn2_scale, bn2_offset, bn2_mean, bn2_var]
            }
        ))

        // CNN-INF-005: VGG-like block (stacked convolutions)
        benchmarks.append(SimpleBenchmark(
            id: "CNN-INF-005",
            name: "CNN VGG Block 56x56x64 (BS=1)",
            category: "model_cnn",
            operation: "cnn_inference",
            configuration: [
                "batch_size": "1",
                "input": "56x56x64",
                "architecture": "VGG-block"
            ],
            mlirProgram: """
            module @vgg_block {
              func.func @main(%input: tensor<1x56x56x64xf32>, %conv1_w: tensor<3x3x64x128xf32>, %conv1_b: tensor<128xf32>, %conv2_w: tensor<3x3x128x128xf32>, %conv2_b: tensor<128xf32>) -> (tensor<1x28x28x128xf32>) {
                %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>

                // Conv1: 56x56x64 -> 56x56x128
                %conv1 = stablehlo.convolution %input, %conv1_w padding = [[1, 1], [1, 1]], window_strides = [1, 1], feature_group_count = 1 : (tensor<1x56x56x64xf32>, tensor<3x3x64x128xf32>) -> tensor<1x56x56x128xf32>
                %conv1_b_bc = stablehlo.broadcast_in_dim %conv1_b, dims = [3] : (tensor<128xf32>) -> tensor<1x56x56x128xf32>
                %conv1_bias = stablehlo.add %conv1, %conv1_b_bc : tensor<1x56x56x128xf32>
                %zero1 = stablehlo.constant dense<0.0> : tensor<1x56x56x128xf32>
                %relu1 = stablehlo.maximum %conv1_bias, %zero1 : tensor<1x56x56x128xf32>

                // Conv2: 56x56x128 -> 56x56x128
                %conv2 = stablehlo.convolution %relu1, %conv2_w padding = [[1, 1], [1, 1]], window_strides = [1, 1], feature_group_count = 1 : (tensor<1x56x56x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x56x56x128xf32>
                %conv2_b_bc = stablehlo.broadcast_in_dim %conv2_b, dims = [3] : (tensor<128xf32>) -> tensor<1x56x56x128xf32>
                %conv2_bias = stablehlo.add %conv2, %conv2_b_bc : tensor<1x56x56x128xf32>
                %zero2 = stablehlo.constant dense<0.0> : tensor<1x56x56x128xf32>
                %relu2 = stablehlo.maximum %conv2_bias, %zero2 : tensor<1x56x56x128xf32>

                // MaxPool: 56x56x128 -> 28x28x128
                %out = stablehlo.reduce_window %relu2, %neg_inf applies stablehlo.maximum window_dimensions = [1, 2, 2, 1], window_strides = [1, 2, 2, 1], padding = [[0, 0], [0, 0], [0, 0], [0, 0]] : (tensor<1x56x56x128xf32>, tensor<f32>) -> tensor<1x28x28x128xf32>

                return %out : tensor<1x28x28x128xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let input = try gen.createUniformFloat32Buffer(client: client, shape: [1, 56, 56, 64])
                let conv1_w = try gen.createNormalFloat32Buffer(client: client, shape: [3, 3, 64, 128], mean: 0, stdDev: 0.05)
                let conv1_b = try gen.createZerosBuffer(client: client, shape: [128], elementType: .float32)
                let conv2_w = try gen.createNormalFloat32Buffer(client: client, shape: [3, 3, 128, 128], mean: 0, stdDev: 0.05)
                let conv2_b = try gen.createZerosBuffer(client: client, shape: [128], elementType: .float32)
                return [input, conv1_w, conv1_b, conv2_w, conv2_b]
            }
        ))

        return benchmarks
    }

    // MARK: - Transformer Component Benchmarks

    /// Transformer component benchmarks (attention, FFN, encoder blocks).
    public static func transformerBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // XFMR-INF-001: Self-Attention, Seq 128, Hidden 768, Heads 12, Batch 1
        benchmarks.append(SimpleBenchmark(
            id: "XFMR-INF-001",
            name: "Self-Attention Seq128 H768 (BS=1)",
            category: "model_transformer",
            operation: "self_attention",
            configuration: [
                "batch_size": "1",
                "seq_len": "128",
                "hidden": "768",
                "heads": "12",
                "head_dim": "64"
            ],
            mlirProgram: """
            module @self_attention_bs1_seq128 {
              func.func @main(%x: tensor<1x128x768xf32>, %wq: tensor<768x768xf32>, %wk: tensor<768x768xf32>, %wv: tensor<768x768xf32>, %wo: tensor<768x768xf32>) -> (tensor<1x128x768xf32>) {
                %scale = stablehlo.constant dense<0.125> : tensor<f32>

                // Project Q, K, V: [1, 128, 768]
                %q_flat = stablehlo.reshape %x : (tensor<1x128x768xf32>) -> tensor<128x768xf32>
                %k_flat = stablehlo.reshape %x : (tensor<1x128x768xf32>) -> tensor<128x768xf32>
                %v_flat = stablehlo.reshape %x : (tensor<1x128x768xf32>) -> tensor<128x768xf32>

                %q_proj = stablehlo.dot_general %q_flat, %wq, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x768xf32>, tensor<768x768xf32>) -> tensor<128x768xf32>
                %k_proj = stablehlo.dot_general %k_flat, %wk, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x768xf32>, tensor<768x768xf32>) -> tensor<128x768xf32>
                %v_proj = stablehlo.dot_general %v_flat, %wv, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x768xf32>, tensor<768x768xf32>) -> tensor<128x768xf32>

                // Reshape to multi-head: [1, 12, 128, 64]
                %q = stablehlo.reshape %q_proj : (tensor<128x768xf32>) -> tensor<1x12x128x64xf32>
                %k = stablehlo.reshape %k_proj : (tensor<128x768xf32>) -> tensor<1x12x128x64xf32>
                %v = stablehlo.reshape %v_proj : (tensor<128x768xf32>) -> tensor<1x12x128x64xf32>

                // Attention: Q @ K^T
                %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] : (tensor<1x12x128x64xf32>) -> tensor<1x12x64x128xf32>
                %scores = stablehlo.dot_general %q, %kt, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>

                // Scale
                %scale_bc = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<1x12x128x128xf32>
                %scaled = stablehlo.multiply %scores, %scale_bc : tensor<1x12x128x128xf32>

                // Softmax (simplified: exp and normalize)
                %exp_scores = stablehlo.exponential %scaled : tensor<1x12x128x128xf32>
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %sum = stablehlo.reduce %exp_scores, %zero applies stablehlo.add across dimensions = [3] : (tensor<1x12x128x128xf32>, tensor<f32>) -> tensor<1x12x128xf32>
                %sum_bc = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<1x12x128xf32>) -> tensor<1x12x128x128xf32>
                %attn_weights = stablehlo.divide %exp_scores, %sum_bc : tensor<1x12x128x128xf32>

                // Attention @ V
                %attn_out = stablehlo.dot_general %attn_weights, %v, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<1x12x128x128xf32>, tensor<1x12x128x64xf32>) -> tensor<1x12x128x64xf32>

                // Reshape back: [1, 128, 768]
                %concat = stablehlo.reshape %attn_out : (tensor<1x12x128x64xf32>) -> tensor<128x768xf32>

                // Output projection
                %out_proj = stablehlo.dot_general %concat, %wo, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x768xf32>, tensor<768x768xf32>) -> tensor<128x768xf32>
                %out = stablehlo.reshape %out_proj : (tensor<128x768xf32>) -> tensor<1x128x768xf32>

                return %out : tensor<1x128x768xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [1, 128, 768])
                let wq = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wk = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wv = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wo = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                return [x, wq, wk, wv, wo]
            },
            throughputCalculator: { timing in
                // QKV projections: 3 * 2 * 128 * 768 * 768
                // Attention scores: 2 * 12 * 128 * 128 * 64
                // Attention @ V: 2 * 12 * 128 * 64 * 128
                // Output proj: 2 * 128 * 768 * 768
                let qkv_flops = 3 * 2 * 128 * 768 * 768
                let attn_flops = 2 * 12 * 128 * 128 * 64 + 2 * 12 * 128 * 64 * 128
                let out_flops = 2 * 128 * 768 * 768
                let total_flops = qkv_flops + attn_flops + out_flops
                let gflops = FLOPSCalculator.gflops(flops: Double(total_flops), timeSeconds: timing.gpuTime)
                return ThroughputMetrics(
                    opsPerSecond: 1.0 / timing.gpuTime,
                    flops: gflops * 1e9,
                    elementsPerSecond: Double(1 * 128 * 768) / timing.gpuTime
                )
            }
        ))

        // XFMR-INF-002: Self-Attention, Seq 512, Hidden 768, Heads 12, Batch 1
        benchmarks.append(SimpleBenchmark(
            id: "XFMR-INF-002",
            name: "Self-Attention Seq512 H768 (BS=1)",
            category: "model_transformer",
            operation: "self_attention",
            configuration: [
                "batch_size": "1",
                "seq_len": "512",
                "hidden": "768",
                "heads": "12",
                "head_dim": "64"
            ],
            mlirProgram: """
            module @self_attention_bs1_seq512 {
              func.func @main(%x: tensor<1x512x768xf32>, %wq: tensor<768x768xf32>, %wk: tensor<768x768xf32>, %wv: tensor<768x768xf32>, %wo: tensor<768x768xf32>) -> (tensor<1x512x768xf32>) {
                %scale = stablehlo.constant dense<0.125> : tensor<f32>

                // Project Q, K, V
                %q_flat = stablehlo.reshape %x : (tensor<1x512x768xf32>) -> tensor<512x768xf32>
                %q_proj = stablehlo.dot_general %q_flat, %wq, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<512x768xf32>, tensor<768x768xf32>) -> tensor<512x768xf32>
                %k_proj = stablehlo.dot_general %q_flat, %wk, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<512x768xf32>, tensor<768x768xf32>) -> tensor<512x768xf32>
                %v_proj = stablehlo.dot_general %q_flat, %wv, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<512x768xf32>, tensor<768x768xf32>) -> tensor<512x768xf32>

                // Reshape to multi-head
                %q = stablehlo.reshape %q_proj : (tensor<512x768xf32>) -> tensor<1x12x512x64xf32>
                %k = stablehlo.reshape %k_proj : (tensor<512x768xf32>) -> tensor<1x12x512x64xf32>
                %v = stablehlo.reshape %v_proj : (tensor<512x768xf32>) -> tensor<1x12x512x64xf32>

                // Attention
                %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
                %scores = stablehlo.dot_general %q, %kt, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<1x12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<1x12x512x512xf32>

                %scale_bc = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<1x12x512x512xf32>
                %scaled = stablehlo.multiply %scores, %scale_bc : tensor<1x12x512x512xf32>

                %exp_scores = stablehlo.exponential %scaled : tensor<1x12x512x512xf32>
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %sum = stablehlo.reduce %exp_scores, %zero applies stablehlo.add across dimensions = [3] : (tensor<1x12x512x512xf32>, tensor<f32>) -> tensor<1x12x512xf32>
                %sum_bc = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<1x12x512xf32>) -> tensor<1x12x512x512xf32>
                %attn_weights = stablehlo.divide %exp_scores, %sum_bc : tensor<1x12x512x512xf32>

                %attn_out = stablehlo.dot_general %attn_weights, %v, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<1x12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<1x12x512x64xf32>

                %concat = stablehlo.reshape %attn_out : (tensor<1x12x512x64xf32>) -> tensor<512x768xf32>
                %out_proj = stablehlo.dot_general %concat, %wo, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<512x768xf32>, tensor<768x768xf32>) -> tensor<512x768xf32>
                %out = stablehlo.reshape %out_proj : (tensor<512x768xf32>) -> tensor<1x512x768xf32>

                return %out : tensor<1x512x768xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [1, 512, 768])
                let wq = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wk = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wv = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wo = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                return [x, wq, wk, wv, wo]
            }
        ))

        // XFMR-INF-003: Self-Attention, Seq 128, Hidden 768, Heads 12, Batch 8
        benchmarks.append(SimpleBenchmark(
            id: "XFMR-INF-003",
            name: "Self-Attention Seq128 H768 (BS=8)",
            category: "model_transformer",
            operation: "self_attention",
            configuration: [
                "batch_size": "8",
                "seq_len": "128",
                "hidden": "768",
                "heads": "12",
                "head_dim": "64"
            ],
            mlirProgram: """
            module @self_attention_bs8_seq128 {
              func.func @main(%x: tensor<8x128x768xf32>, %wq: tensor<768x768xf32>, %wk: tensor<768x768xf32>, %wv: tensor<768x768xf32>, %wo: tensor<768x768xf32>) -> (tensor<8x128x768xf32>) {
                %scale = stablehlo.constant dense<0.125> : tensor<f32>

                // Flatten batch*seq for projection
                %x_flat = stablehlo.reshape %x : (tensor<8x128x768xf32>) -> tensor<1024x768xf32>
                %q_proj = stablehlo.dot_general %x_flat, %wq, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x768xf32>, tensor<768x768xf32>) -> tensor<1024x768xf32>
                %k_proj = stablehlo.dot_general %x_flat, %wk, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x768xf32>, tensor<768x768xf32>) -> tensor<1024x768xf32>
                %v_proj = stablehlo.dot_general %x_flat, %wv, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x768xf32>, tensor<768x768xf32>) -> tensor<1024x768xf32>

                // Reshape to [batch, heads, seq, head_dim]
                %q = stablehlo.reshape %q_proj : (tensor<1024x768xf32>) -> tensor<8x12x128x64xf32>
                %k = stablehlo.reshape %k_proj : (tensor<1024x768xf32>) -> tensor<8x12x128x64xf32>
                %v = stablehlo.reshape %v_proj : (tensor<1024x768xf32>) -> tensor<8x12x128x64xf32>

                // Attention
                %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] : (tensor<8x12x128x64xf32>) -> tensor<8x12x64x128xf32>
                %scores = stablehlo.dot_general %q, %kt, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<8x12x128x64xf32>, tensor<8x12x64x128xf32>) -> tensor<8x12x128x128xf32>

                %scale_bc = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<8x12x128x128xf32>
                %scaled = stablehlo.multiply %scores, %scale_bc : tensor<8x12x128x128xf32>

                %exp_scores = stablehlo.exponential %scaled : tensor<8x12x128x128xf32>
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %sum = stablehlo.reduce %exp_scores, %zero applies stablehlo.add across dimensions = [3] : (tensor<8x12x128x128xf32>, tensor<f32>) -> tensor<8x12x128xf32>
                %sum_bc = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<8x12x128xf32>) -> tensor<8x12x128x128xf32>
                %attn_weights = stablehlo.divide %exp_scores, %sum_bc : tensor<8x12x128x128xf32>

                %attn_out = stablehlo.dot_general %attn_weights, %v, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<8x12x128x128xf32>, tensor<8x12x128x64xf32>) -> tensor<8x12x128x64xf32>

                %concat = stablehlo.reshape %attn_out : (tensor<8x12x128x64xf32>) -> tensor<1024x768xf32>
                %out_proj = stablehlo.dot_general %concat, %wo, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x768xf32>, tensor<768x768xf32>) -> tensor<1024x768xf32>
                %out = stablehlo.reshape %out_proj : (tensor<1024x768xf32>) -> tensor<8x128x768xf32>

                return %out : tensor<8x128x768xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [8, 128, 768])
                let wq = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wk = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wv = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wo = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                return [x, wq, wk, wv, wo]
            }
        ))

        // XFMR-INF-004: Transformer FFN, Seq 128, 768->3072->768, Batch 8
        benchmarks.append(SimpleBenchmark(
            id: "XFMR-INF-004",
            name: "Transformer FFN 768->3072->768 (BS=8, Seq=128)",
            category: "model_transformer",
            operation: "ffn",
            configuration: [
                "batch_size": "8",
                "seq_len": "128",
                "hidden": "768",
                "intermediate": "3072"
            ],
            mlirProgram: """
            module @transformer_ffn {
              func.func @main(%x: tensor<8x128x768xf32>, %w1: tensor<768x3072xf32>, %b1: tensor<3072xf32>, %w2: tensor<3072x768xf32>, %b2: tensor<768xf32>) -> (tensor<8x128x768xf32>) {
                // Flatten for matmul
                %x_flat = stablehlo.reshape %x : (tensor<8x128x768xf32>) -> tensor<1024x768xf32>

                // FC1: 768 -> 3072
                %h1_mm = stablehlo.dot_general %x_flat, %w1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x768xf32>, tensor<768x3072xf32>) -> tensor<1024x3072xf32>
                %b1_bc = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<3072xf32>) -> tensor<1024x3072xf32>
                %h1_bias = stablehlo.add %h1_mm, %b1_bc : tensor<1024x3072xf32>

                // GELU (using sigmoid approximation: x * sigmoid(x))
                %sig = stablehlo.logistic %h1_bias : tensor<1024x3072xf32>
                %h1 = stablehlo.multiply %h1_bias, %sig : tensor<1024x3072xf32>

                // FC2: 3072 -> 768
                %h2_mm = stablehlo.dot_general %h1, %w2, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x3072xf32>, tensor<3072x768xf32>) -> tensor<1024x768xf32>
                %b2_bc = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<768xf32>) -> tensor<1024x768xf32>
                %h2 = stablehlo.add %h2_mm, %b2_bc : tensor<1024x768xf32>

                // Reshape back
                %out = stablehlo.reshape %h2 : (tensor<1024x768xf32>) -> tensor<8x128x768xf32>

                return %out : tensor<8x128x768xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [8, 128, 768])
                let w1 = try gen.createNormalFloat32Buffer(client: client, shape: [768, 3072], mean: 0, stdDev: 0.02)
                let b1 = try gen.createZerosBuffer(client: client, shape: [3072], elementType: .float32)
                let w2 = try gen.createNormalFloat32Buffer(client: client, shape: [3072, 768], mean: 0, stdDev: 0.02)
                let b2 = try gen.createZerosBuffer(client: client, shape: [768], elementType: .float32)
                return [x, w1, b1, w2, b2]
            },
            throughputCalculator: { timing in
                let flops = 2 * 8 * 128 * (768 * 3072 + 3072 * 768)
                let gflops = FLOPSCalculator.gflops(flops: Double(flops), timeSeconds: timing.gpuTime)
                return ThroughputMetrics(
                    opsPerSecond: Double(8 * 128) / timing.gpuTime,
                    flops: gflops * 1e9,
                    elementsPerSecond: Double(8 * 128 * 768) / timing.gpuTime
                )
            }
        ))

        // XFMR-INF-005: Softmax benchmark (common operation in transformers)
        benchmarks.append(SimpleBenchmark(
            id: "XFMR-INF-005",
            name: "Softmax 8x12x128x128",
            category: "model_transformer",
            operation: "softmax",
            configuration: [
                "batch_size": "8",
                "heads": "12",
                "seq_len": "128"
            ],
            mlirProgram: """
            module @softmax_attention {
              func.func @main(%x: tensor<8x12x128x128xf32>) -> (tensor<8x12x128x128xf32>) {
                // Softmax over last dimension
                // max for numerical stability
                %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
                %max = stablehlo.reduce %x, %neg_inf applies stablehlo.maximum across dimensions = [3] : (tensor<8x12x128x128xf32>, tensor<f32>) -> tensor<8x12x128xf32>
                %max_bc = stablehlo.broadcast_in_dim %max, dims = [0, 1, 2] : (tensor<8x12x128xf32>) -> tensor<8x12x128x128xf32>
                %shifted = stablehlo.subtract %x, %max_bc : tensor<8x12x128x128xf32>

                // exp
                %exp = stablehlo.exponential %shifted : tensor<8x12x128x128xf32>

                // sum
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %sum = stablehlo.reduce %exp, %zero applies stablehlo.add across dimensions = [3] : (tensor<8x12x128x128xf32>, tensor<f32>) -> tensor<8x12x128xf32>
                %sum_bc = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<8x12x128xf32>) -> tensor<8x12x128x128xf32>

                // normalize
                %out = stablehlo.divide %exp, %sum_bc : tensor<8x12x128x128xf32>

                return %out : tensor<8x12x128x128xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [8, 12, 128, 128], min: -5.0, max: 5.0)]
            }
        ))

        // XFMR-INF-006: Full Encoder Block (Attention + FFN + LayerNorms + Residuals)
        benchmarks.append(SimpleBenchmark(
            id: "XFMR-INF-006",
            name: "Transformer Encoder Block (BS=1, Seq=128)",
            category: "model_transformer",
            operation: "encoder_block",
            configuration: [
                "batch_size": "1",
                "seq_len": "128",
                "hidden": "768",
                "heads": "12",
                "intermediate": "3072"
            ],
            mlirProgram: """
            module @encoder_block {
              func.func @main(%x: tensor<1x128x768xf32>, %ln1_g: tensor<768xf32>, %ln1_b: tensor<768xf32>, %wq: tensor<768x768xf32>, %wk: tensor<768x768xf32>, %wv: tensor<768x768xf32>, %wo: tensor<768x768xf32>, %ln2_g: tensor<768xf32>, %ln2_b: tensor<768xf32>, %ff_w1: tensor<768x3072xf32>, %ff_b1: tensor<3072xf32>, %ff_w2: tensor<3072x768xf32>, %ff_b2: tensor<768xf32>) -> (tensor<1x128x768xf32>) {
                %eps = stablehlo.constant dense<1.0e-05> : tensor<f32>
                %hidden_size = stablehlo.constant dense<768.0> : tensor<f32>
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %scale = stablehlo.constant dense<0.125> : tensor<f32>

                // LayerNorm 1 (simplified inline)
                %sum1 = stablehlo.reduce %x, %zero applies stablehlo.add across dimensions = [2] : (tensor<1x128x768xf32>, tensor<f32>) -> tensor<1x128xf32>
                %hidden_bc1 = stablehlo.broadcast_in_dim %hidden_size, dims = [] : (tensor<f32>) -> tensor<1x128xf32>
                %mean1 = stablehlo.divide %sum1, %hidden_bc1 : tensor<1x128xf32>
                %mean1_bc = stablehlo.broadcast_in_dim %mean1, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x768xf32>
                %centered1 = stablehlo.subtract %x, %mean1_bc : tensor<1x128x768xf32>
                %sq1 = stablehlo.multiply %centered1, %centered1 : tensor<1x128x768xf32>
                %sq_sum1 = stablehlo.reduce %sq1, %zero applies stablehlo.add across dimensions = [2] : (tensor<1x128x768xf32>, tensor<f32>) -> tensor<1x128xf32>
                %var1 = stablehlo.divide %sq_sum1, %hidden_bc1 : tensor<1x128xf32>
                %eps_bc1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1x128xf32>
                %var_eps1 = stablehlo.add %var1, %eps_bc1 : tensor<1x128xf32>
                %rstd1 = stablehlo.rsqrt %var_eps1 : tensor<1x128xf32>
                %rstd1_bc = stablehlo.broadcast_in_dim %rstd1, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x768xf32>
                %norm1 = stablehlo.multiply %centered1, %rstd1_bc : tensor<1x128x768xf32>
                %ln1_g_bc = stablehlo.broadcast_in_dim %ln1_g, dims = [2] : (tensor<768xf32>) -> tensor<1x128x768xf32>
                %ln1_b_bc = stablehlo.broadcast_in_dim %ln1_b, dims = [2] : (tensor<768xf32>) -> tensor<1x128x768xf32>
                %ln1_scaled = stablehlo.multiply %norm1, %ln1_g_bc : tensor<1x128x768xf32>
                %ln1_out = stablehlo.add %ln1_scaled, %ln1_b_bc : tensor<1x128x768xf32>

                // Self-Attention (simplified)
                %x_flat = stablehlo.reshape %ln1_out : (tensor<1x128x768xf32>) -> tensor<128x768xf32>
                %q_proj = stablehlo.dot_general %x_flat, %wq, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x768xf32>, tensor<768x768xf32>) -> tensor<128x768xf32>
                %k_proj = stablehlo.dot_general %x_flat, %wk, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x768xf32>, tensor<768x768xf32>) -> tensor<128x768xf32>
                %v_proj = stablehlo.dot_general %x_flat, %wv, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x768xf32>, tensor<768x768xf32>) -> tensor<128x768xf32>

                %q = stablehlo.reshape %q_proj : (tensor<128x768xf32>) -> tensor<1x12x128x64xf32>
                %k = stablehlo.reshape %k_proj : (tensor<128x768xf32>) -> tensor<1x12x128x64xf32>
                %v = stablehlo.reshape %v_proj : (tensor<128x768xf32>) -> tensor<1x12x128x64xf32>

                %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] : (tensor<1x12x128x64xf32>) -> tensor<1x12x64x128xf32>
                %scores = stablehlo.dot_general %q, %kt, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<1x12x128x64xf32>, tensor<1x12x64x128xf32>) -> tensor<1x12x128x128xf32>
                %scale_bc = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<1x12x128x128xf32>
                %scaled = stablehlo.multiply %scores, %scale_bc : tensor<1x12x128x128xf32>
                %exp_scores = stablehlo.exponential %scaled : tensor<1x12x128x128xf32>
                %sum_attn = stablehlo.reduce %exp_scores, %zero applies stablehlo.add across dimensions = [3] : (tensor<1x12x128x128xf32>, tensor<f32>) -> tensor<1x12x128xf32>
                %sum_attn_bc = stablehlo.broadcast_in_dim %sum_attn, dims = [0, 1, 2] : (tensor<1x12x128xf32>) -> tensor<1x12x128x128xf32>
                %attn_weights = stablehlo.divide %exp_scores, %sum_attn_bc : tensor<1x12x128x128xf32>
                %attn_out = stablehlo.dot_general %attn_weights, %v, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<1x12x128x128xf32>, tensor<1x12x128x64xf32>) -> tensor<1x12x128x64xf32>
                %concat = stablehlo.reshape %attn_out : (tensor<1x12x128x64xf32>) -> tensor<128x768xf32>
                %out_proj = stablehlo.dot_general %concat, %wo, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x768xf32>, tensor<768x768xf32>) -> tensor<128x768xf32>
                %attn_result = stablehlo.reshape %out_proj : (tensor<128x768xf32>) -> tensor<1x128x768xf32>

                // Residual 1
                %res1 = stablehlo.add %x, %attn_result : tensor<1x128x768xf32>

                // LayerNorm 2 (reuse computation pattern)
                %sum2 = stablehlo.reduce %res1, %zero applies stablehlo.add across dimensions = [2] : (tensor<1x128x768xf32>, tensor<f32>) -> tensor<1x128xf32>
                %mean2 = stablehlo.divide %sum2, %hidden_bc1 : tensor<1x128xf32>
                %mean2_bc = stablehlo.broadcast_in_dim %mean2, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x768xf32>
                %centered2 = stablehlo.subtract %res1, %mean2_bc : tensor<1x128x768xf32>
                %sq2 = stablehlo.multiply %centered2, %centered2 : tensor<1x128x768xf32>
                %sq_sum2 = stablehlo.reduce %sq2, %zero applies stablehlo.add across dimensions = [2] : (tensor<1x128x768xf32>, tensor<f32>) -> tensor<1x128xf32>
                %var2 = stablehlo.divide %sq_sum2, %hidden_bc1 : tensor<1x128xf32>
                %var_eps2 = stablehlo.add %var2, %eps_bc1 : tensor<1x128xf32>
                %rstd2 = stablehlo.rsqrt %var_eps2 : tensor<1x128xf32>
                %rstd2_bc = stablehlo.broadcast_in_dim %rstd2, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x768xf32>
                %norm2 = stablehlo.multiply %centered2, %rstd2_bc : tensor<1x128x768xf32>
                %ln2_g_bc = stablehlo.broadcast_in_dim %ln2_g, dims = [2] : (tensor<768xf32>) -> tensor<1x128x768xf32>
                %ln2_b_bc = stablehlo.broadcast_in_dim %ln2_b, dims = [2] : (tensor<768xf32>) -> tensor<1x128x768xf32>
                %ln2_scaled = stablehlo.multiply %norm2, %ln2_g_bc : tensor<1x128x768xf32>
                %ln2_out = stablehlo.add %ln2_scaled, %ln2_b_bc : tensor<1x128x768xf32>

                // FFN
                %ff_flat = stablehlo.reshape %ln2_out : (tensor<1x128x768xf32>) -> tensor<128x768xf32>
                %ff1 = stablehlo.dot_general %ff_flat, %ff_w1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x768xf32>, tensor<768x3072xf32>) -> tensor<128x3072xf32>
                %ff_b1_bc = stablehlo.broadcast_in_dim %ff_b1, dims = [1] : (tensor<3072xf32>) -> tensor<128x3072xf32>
                %ff1_bias = stablehlo.add %ff1, %ff_b1_bc : tensor<128x3072xf32>
                %ff1_sig = stablehlo.logistic %ff1_bias : tensor<128x3072xf32>
                %ff1_gelu = stablehlo.multiply %ff1_bias, %ff1_sig : tensor<128x3072xf32>
                %ff2 = stablehlo.dot_general %ff1_gelu, %ff_w2, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x3072xf32>, tensor<3072x768xf32>) -> tensor<128x768xf32>
                %ff_b2_bc = stablehlo.broadcast_in_dim %ff_b2, dims = [1] : (tensor<768xf32>) -> tensor<128x768xf32>
                %ff2_bias = stablehlo.add %ff2, %ff_b2_bc : tensor<128x768xf32>
                %ff_out = stablehlo.reshape %ff2_bias : (tensor<128x768xf32>) -> tensor<1x128x768xf32>

                // Residual 2
                %out = stablehlo.add %res1, %ff_out : tensor<1x128x768xf32>

                return %out : tensor<1x128x768xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let x = try gen.createUniformFloat32Buffer(client: client, shape: [1, 128, 768])
                let ln1_g = try client.createBuffer([Float](repeating: 1.0, count: 768), shape: [768])
                let ln1_b = try gen.createZerosBuffer(client: client, shape: [768], elementType: .float32)
                let wq = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wk = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wv = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let wo = try gen.createNormalFloat32Buffer(client: client, shape: [768, 768], mean: 0, stdDev: 0.02)
                let ln2_g = try client.createBuffer([Float](repeating: 1.0, count: 768), shape: [768])
                let ln2_b = try gen.createZerosBuffer(client: client, shape: [768], elementType: .float32)
                let ff_w1 = try gen.createNormalFloat32Buffer(client: client, shape: [768, 3072], mean: 0, stdDev: 0.02)
                let ff_b1 = try gen.createZerosBuffer(client: client, shape: [3072], elementType: .float32)
                let ff_w2 = try gen.createNormalFloat32Buffer(client: client, shape: [3072, 768], mean: 0, stdDev: 0.02)
                let ff_b2 = try gen.createZerosBuffer(client: client, shape: [768], elementType: .float32)
                return [x, ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, ff_w1, ff_b1, ff_w2, ff_b2]
            }
        ))

        return benchmarks
    }

    // MARK: - End-to-End Pipeline Benchmarks

    /// End-to-end pipeline benchmarks.
    public static func endToEndBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // E2E-001: Text embedding lookup (common in NLP)
        benchmarks.append(SimpleBenchmark(
            id: "E2E-001",
            name: "Text Embedding Lookup (Vocab=30000, Dim=768, Seq=128)",
            category: "model_e2e",
            operation: "embedding_lookup",
            configuration: [
                "vocab_size": "30000",
                "embedding_dim": "768",
                "seq_len": "128",
                "batch_size": "8"
            ],
            mlirProgram: """
            module @text_embedding {
              func.func @main(%embeddings: tensor<30000x768xf32>, %token_ids: tensor<8x128xi32>) -> (tensor<8x128x768xf32>) {
                // Gather embeddings for each token
                %flat_ids = stablehlo.reshape %token_ids : (tensor<8x128xi32>) -> tensor<1024xi32>
                %gathered = stablehlo.gather %embeddings, %flat_ids,
                  offset_dims = [1],
                  collapsed_slice_dims = [0],
                  start_index_map = [0],
                  index_vector_dim = 1,
                  slice_sizes = [1, 768] : (tensor<30000x768xf32>, tensor<1024xi32>) -> tensor<1024x768xf32>
                %out = stablehlo.reshape %gathered : (tensor<1024x768xf32>) -> tensor<8x128x768xf32>
                return %out : tensor<8x128x768xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let embeddings = try gen.createNormalFloat32Buffer(client: client, shape: [30000, 768], mean: 0, stdDev: 0.02)
                // Generate random token IDs in range [0, 30000)
                var tokenIds: [Int32] = []
                for _ in 0..<(8 * 128) {
                    tokenIds.append(Int32(gen.uniformFloat32(count: 1)[0] * 29999))
                }
                let tokenIdsBuffer = try client.createBuffer(tokenIds, shape: [8, 128])
                return [embeddings, tokenIdsBuffer]
            },
            throughputCalculator: { timing in
                let elements = 8 * 128 * 768
                let bytesRead = Double(elements * 4)
                return ThroughputMetrics(
                    opsPerSecond: Double(8 * 128) / timing.gpuTime,
                    memoryBandwidthGBps: bytesRead / timing.gpuTime / 1e9,
                    elementsPerSecond: Double(elements) / timing.gpuTime
                )
            }
        ))

        // E2E-002: Image normalization pipeline
        benchmarks.append(SimpleBenchmark(
            id: "E2E-002",
            name: "Image Normalize (BS=32, 224x224x3)",
            category: "model_e2e",
            operation: "image_normalize",
            configuration: [
                "batch_size": "32",
                "height": "224",
                "width": "224",
                "channels": "3"
            ],
            mlirProgram: """
            module @image_normalize {
              func.func @main(%images: tensor<32x224x224x3xf32>, %mean: tensor<3xf32>, %std: tensor<3xf32>) -> (tensor<32x224x224x3xf32>) {
                // Normalize: (x - mean) / std
                %mean_bc = stablehlo.broadcast_in_dim %mean, dims = [3] : (tensor<3xf32>) -> tensor<32x224x224x3xf32>
                %centered = stablehlo.subtract %images, %mean_bc : tensor<32x224x224x3xf32>
                %std_bc = stablehlo.broadcast_in_dim %std, dims = [3] : (tensor<3xf32>) -> tensor<32x224x224x3xf32>
                %normalized = stablehlo.divide %centered, %std_bc : tensor<32x224x224x3xf32>
                return %normalized : tensor<32x224x224x3xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let images = try gen.createUniformFloat32Buffer(client: client, shape: [32, 224, 224, 3], min: 0, max: 255)
                // ImageNet mean and std
                let mean = try client.createBuffer([Float(123.675), Float(116.28), Float(103.53)], shape: [3])
                let std = try client.createBuffer([Float(58.395), Float(57.12), Float(57.375)], shape: [3])
                return [images, mean, std]
            }
        ))

        // E2E-003: Softmax + CrossEntropy loss (classification head)
        benchmarks.append(SimpleBenchmark(
            id: "E2E-003",
            name: "Classification Head (BS=32, Classes=1000)",
            category: "model_e2e",
            operation: "classification_head",
            configuration: [
                "batch_size": "32",
                "num_classes": "1000"
            ],
            mlirProgram: """
            module @classification_head {
              func.func @main(%logits: tensor<32x1000xf32>, %labels: tensor<32xi32>) -> (tensor<f32>) {
                // Softmax
                %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
                %max = stablehlo.reduce %logits, %neg_inf applies stablehlo.maximum across dimensions = [1] : (tensor<32x1000xf32>, tensor<f32>) -> tensor<32xf32>
                %max_bc = stablehlo.broadcast_in_dim %max, dims = [0] : (tensor<32xf32>) -> tensor<32x1000xf32>
                %shifted = stablehlo.subtract %logits, %max_bc : tensor<32x1000xf32>
                %exp = stablehlo.exponential %shifted : tensor<32x1000xf32>
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %sum = stablehlo.reduce %exp, %zero applies stablehlo.add across dimensions = [1] : (tensor<32x1000xf32>, tensor<f32>) -> tensor<32xf32>
                %log_sum = stablehlo.log %sum : tensor<32xf32>

                // Log softmax = shifted - log(sum(exp(shifted)))
                %log_sum_bc = stablehlo.broadcast_in_dim %log_sum, dims = [0] : (tensor<32xf32>) -> tensor<32x1000xf32>
                %log_probs = stablehlo.subtract %shifted, %log_sum_bc : tensor<32x1000xf32>

                // Gather the log probability at the correct label index
                %labels_2d = stablehlo.reshape %labels : (tensor<32xi32>) -> tensor<32x1xi32>
                %target_log_probs = stablehlo.gather %log_probs, %labels_2d,
                  offset_dims = [],
                  collapsed_slice_dims = [0, 1],
                  start_index_map = [0, 1],
                  index_vector_dim = 1,
                  slice_sizes = [1, 1] : (tensor<32x1000xf32>, tensor<32x1xi32>) -> tensor<32xf32>

                // Negative mean for cross entropy loss
                %neg_log_probs = stablehlo.negate %target_log_probs : tensor<32xf32>
                %loss_sum = stablehlo.reduce %neg_log_probs, %zero applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
                %batch_size = stablehlo.constant dense<32.0> : tensor<f32>
                %loss = stablehlo.divide %loss_sum, %batch_size : tensor<f32>

                return %loss : tensor<f32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let logits = try gen.createUniformFloat32Buffer(client: client, shape: [32, 1000], min: -5.0, max: 5.0)
                // Generate random labels in range [0, 1000)
                var labels: [Int32] = []
                for _ in 0..<32 {
                    labels.append(Int32(gen.uniformFloat32(count: 1)[0] * 999))
                }
                let labelsBuffer = try client.createBuffer(labels, shape: [32])
                return [logits, labelsBuffer]
            }
        ))

        return benchmarks
    }

    // MARK: - All Model Benchmarks

    /// Get all model benchmarks.
    public static func all() -> [Benchmark] {
        var benchmarks: [Benchmark] = []
        benchmarks.append(contentsOf: mlpInferenceBenchmarks())
        benchmarks.append(contentsOf: cnnInferenceBenchmarks())
        benchmarks.append(contentsOf: transformerBenchmarks())
        benchmarks.append(contentsOf: endToEndBenchmarks())
        return benchmarks
    }
}
