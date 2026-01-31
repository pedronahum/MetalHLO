// OperationBenchmarks.swift
// MetalHLO Benchmarks
//
// Operation-level benchmark definitions based on the benchmark proposal.

import Foundation
import MetalHLO

/// Factory for creating operation-level benchmarks.
public enum OperationBenchmarks {

    // MARK: - Matrix Operations (Critical Priority)

    /// Matrix multiplication benchmarks.
    public static func matrixBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // Standard GEMM benchmarks
        let gemmConfigs: [(id: String, m: Int, n: Int, k: Int, notes: String)] = [
            ("MAT-DOT-001", 128, 128, 128, "Small square"),
            ("MAT-DOT-002", 512, 512, 512, "Medium square"),
            ("MAT-DOT-003", 1024, 1024, 1024, "Large square"),
            ("MAT-DOT-004", 2048, 2048, 2048, "Very large"),
            ("MAT-DOT-005", 4096, 4096, 4096, "Extreme"),
            ("MAT-DOT-006", 32, 4096, 768, "Transformer-like"),
            ("MAT-DOT-007", 128, 768, 3072, "MLP layer"),
            ("MAT-DOT-008", 1, 4096, 4096, "Vector-matrix"),
        ]

        for config in gemmConfigs {
            let mlir = """
            module @gemm_\(config.m)_\(config.n)_\(config.k) {
              func.func @main(%arg0: tensor<\(config.m)x\(config.k)xf32>, %arg1: tensor<\(config.k)x\(config.n)xf32>) -> (tensor<\(config.m)x\(config.n)xf32>) {
                %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<\(config.m)x\(config.k)xf32>, tensor<\(config.k)x\(config.n)xf32>) -> tensor<\(config.m)x\(config.n)xf32>
                return %0 : tensor<\(config.m)x\(config.n)xf32>
              }
            }
            """

            let flops = FLOPSCalculator.matmul(m: config.m, n: config.n, k: config.k)

            benchmarks.append(SimpleBenchmark(
                id: config.id,
                name: "GEMM \(config.m)x\(config.k) @ \(config.k)x\(config.n)",
                category: "matrix",
                operation: "dot_general",
                configuration: [
                    "M": "\(config.m)",
                    "N": "\(config.n)",
                    "K": "\(config.k)",
                    "dtype": "f32",
                    "notes": config.notes
                ],
                mlirProgram: mlir,
                inputGenerator: { client in
                    let gen = TestDataGenerator(seed: 42)
                    let a = try gen.createUniformFloat32Buffer(client: client, shape: [config.m, config.k])
                    let b = try gen.createUniformFloat32Buffer(client: client, shape: [config.k, config.n])
                    return [a, b]
                },
                throughputCalculator: { timing in
                    let gflops = FLOPSCalculator.gflops(flops: flops, timeSeconds: timing.gpuTime)
                    return ThroughputMetrics(
                        opsPerSecond: 1.0 / timing.gpuTime,
                        flops: gflops * 1e9,
                        elementsPerSecond: Double(config.m * config.n) / timing.gpuTime
                    )
                }
            ))
        }

        // Batched GEMM benchmarks
        let batchedConfigs: [(id: String, batch: Int, m: Int, n: Int, k: Int, notes: String)] = [
            ("MAT-BATCH-001", 8, 512, 512, 512, "Small batch"),
            ("MAT-BATCH-002", 32, 256, 256, 256, "Medium batch"),
            ("MAT-BATCH-003", 64, 128, 128, 64, "Attention heads"),
            ("MAT-BATCH-004", 12, 64, 512, 64, "Multi-head attention"),
        ]

        for config in batchedConfigs {
            let mlir = """
            module @batched_gemm_\(config.batch)_\(config.m)_\(config.n)_\(config.k) {
              func.func @main(%arg0: tensor<\(config.batch)x\(config.m)x\(config.k)xf32>, %arg1: tensor<\(config.batch)x\(config.k)x\(config.n)xf32>) -> (tensor<\(config.batch)x\(config.m)x\(config.n)xf32>) {
                %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]> : (tensor<\(config.batch)x\(config.m)x\(config.k)xf32>, tensor<\(config.batch)x\(config.k)x\(config.n)xf32>) -> tensor<\(config.batch)x\(config.m)x\(config.n)xf32>
                return %0 : tensor<\(config.batch)x\(config.m)x\(config.n)xf32>
              }
            }
            """

            let flops = FLOPSCalculator.batchedMatmul(batch: config.batch, m: config.m, n: config.n, k: config.k)

            benchmarks.append(SimpleBenchmark(
                id: config.id,
                name: "Batched GEMM [\(config.batch)] \(config.m)x\(config.k) @ \(config.k)x\(config.n)",
                category: "matrix",
                operation: "dot_general_batched",
                configuration: [
                    "batch": "\(config.batch)",
                    "M": "\(config.m)",
                    "N": "\(config.n)",
                    "K": "\(config.k)",
                    "dtype": "f32",
                    "notes": config.notes
                ],
                mlirProgram: mlir,
                inputGenerator: { client in
                    let gen = TestDataGenerator(seed: 42)
                    let a = try gen.createUniformFloat32Buffer(client: client, shape: [config.batch, config.m, config.k])
                    let b = try gen.createUniformFloat32Buffer(client: client, shape: [config.batch, config.k, config.n])
                    return [a, b]
                },
                throughputCalculator: { timing in
                    let gflops = FLOPSCalculator.gflops(flops: flops, timeSeconds: timing.gpuTime)
                    return ThroughputMetrics(
                        opsPerSecond: 1.0 / timing.gpuTime,
                        flops: gflops * 1e9,
                        elementsPerSecond: Double(config.batch * config.m * config.n) / timing.gpuTime
                    )
                }
            ))
        }

        // Transpose benchmarks
        benchmarks.append(SimpleBenchmark(
            id: "MAT-TR-001",
            name: "Transpose 1024x1024",
            category: "matrix",
            operation: "transpose",
            configuration: ["shape": "1024x1024", "dtype": "f32"],
            mlirProgram: """
            module @transpose_2d {
              func.func @main(%arg0: tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>) {
                %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
                return %0 : tensor<1024x1024xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])]
            }
        ))

        benchmarks.append(SimpleBenchmark(
            id: "MAT-TR-002",
            name: "Transpose 3D 32x128x64",
            category: "matrix",
            operation: "transpose",
            configuration: ["shape": "32x128x64", "permutation": "[0,2,1]", "dtype": "f32"],
            mlirProgram: """
            module @transpose_3d {
              func.func @main(%arg0: tensor<32x128x64xf32>) -> (tensor<32x64x128xf32>) {
                %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<32x128x64xf32>) -> tensor<32x64x128xf32>
                return %0 : tensor<32x64x128xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [32, 128, 64])]
            }
        ))

        return benchmarks
    }

    // MARK: - Reduction Operations (Critical Priority)

    /// Reduction benchmarks.
    public static func reductionBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // Global reductions
        benchmarks.append(SimpleBenchmark(
            id: "RED-001",
            name: "Global Sum 1024x1024",
            category: "reduction",
            operation: "reduce_sum",
            configuration: ["shape": "1024x1024", "dims": "all", "dtype": "f32"],
            mlirProgram: """
            module @global_sum {
              func.func @main(%arg0: tensor<1024x1024xf32>) -> (tensor<f32>) {
                %init = stablehlo.constant dense<0.0> : tensor<f32>
                %0 = stablehlo.reduce %arg0, %init applies stablehlo.add across dimensions = [0, 1] : (tensor<1024x1024xf32>, tensor<f32>) -> tensor<f32>
                return %0 : tensor<f32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])]
            }
        ))

        // Row-wise reduction
        benchmarks.append(SimpleBenchmark(
            id: "RED-002",
            name: "Row-wise Sum 1024x1024",
            category: "reduction",
            operation: "reduce_sum",
            configuration: ["shape": "1024x1024", "dims": "[1]", "dtype": "f32"],
            mlirProgram: """
            module @row_sum {
              func.func @main(%arg0: tensor<1024x1024xf32>) -> (tensor<1024xf32>) {
                %init = stablehlo.constant dense<0.0> : tensor<f32>
                %0 = stablehlo.reduce %arg0, %init applies stablehlo.add across dimensions = [1] : (tensor<1024x1024xf32>, tensor<f32>) -> tensor<1024xf32>
                return %0 : tensor<1024xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])]
            }
        ))

        // Column-wise reduction
        benchmarks.append(SimpleBenchmark(
            id: "RED-003",
            name: "Column-wise Sum 1024x1024",
            category: "reduction",
            operation: "reduce_sum",
            configuration: ["shape": "1024x1024", "dims": "[0]", "dtype": "f32"],
            mlirProgram: """
            module @col_sum {
              func.func @main(%arg0: tensor<1024x1024xf32>) -> (tensor<1024xf32>) {
                %init = stablehlo.constant dense<0.0> : tensor<f32>
                %0 = stablehlo.reduce %arg0, %init applies stablehlo.add across dimensions = [0] : (tensor<1024x1024xf32>, tensor<f32>) -> tensor<1024xf32>
                return %0 : tensor<1024xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])]
            }
        ))

        // Max reduction (softmax prep)
        benchmarks.append(SimpleBenchmark(
            id: "RED-004",
            name: "Row-wise Max 4096x4096",
            category: "reduction",
            operation: "reduce_max",
            configuration: ["shape": "4096x4096", "dims": "[1]", "dtype": "f32"],
            mlirProgram: """
            module @row_max {
              func.func @main(%arg0: tensor<4096x4096xf32>) -> (tensor<4096xf32>) {
                %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
                %0 = stablehlo.reduce %arg0, %init applies stablehlo.maximum across dimensions = [1] : (tensor<4096x4096xf32>, tensor<f32>) -> tensor<4096xf32>
                return %0 : tensor<4096xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [4096, 4096])]
            }
        ))

        // LayerNorm prep reduction
        benchmarks.append(SimpleBenchmark(
            id: "RED-005",
            name: "LayerNorm Reduction 32x128x768",
            category: "reduction",
            operation: "reduce_sum",
            configuration: ["shape": "32x128x768", "dims": "[2]", "dtype": "f32", "notes": "LayerNorm prep"],
            mlirProgram: """
            module @layernorm_reduce {
              func.func @main(%arg0: tensor<32x128x768xf32>) -> (tensor<32x128xf32>) {
                %init = stablehlo.constant dense<0.0> : tensor<f32>
                %0 = stablehlo.reduce %arg0, %init applies stablehlo.add across dimensions = [2] : (tensor<32x128x768xf32>, tensor<f32>) -> tensor<32x128xf32>
                return %0 : tensor<32x128xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [32, 128, 768])]
            }
        ))

        // Attention scores reduction
        benchmarks.append(SimpleBenchmark(
            id: "RED-006",
            name: "Attention Reduction 32x12x512x512",
            category: "reduction",
            operation: "reduce_sum",
            configuration: ["shape": "32x12x512x512", "dims": "[3]", "dtype": "f32", "notes": "Attention scores"],
            mlirProgram: """
            module @attention_reduce {
              func.func @main(%arg0: tensor<32x12x512x512xf32>) -> (tensor<32x12x512xf32>) {
                %init = stablehlo.constant dense<0.0> : tensor<f32>
                %0 = stablehlo.reduce %arg0, %init applies stablehlo.add across dimensions = [3] : (tensor<32x12x512x512xf32>, tensor<f32>) -> tensor<32x12x512xf32>
                return %0 : tensor<32x12x512xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [32, 12, 512, 512])]
            }
        ))

        return benchmarks
    }

    // MARK: - Arithmetic Operations (High Priority)

    /// Arithmetic benchmarks.
    public static func arithmeticBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // Binary operations
        let binaryConfigs: [(id: String, op: String, stablehloOp: String, shape: [Int])] = [
            ("ARITH-B-001", "add", "add", [1024, 1024]),
            ("ARITH-B-002", "add", "add", [4096, 4096]),
            ("ARITH-B-003", "multiply", "multiply", [1024, 1024]),
            ("ARITH-B-004", "multiply", "multiply", [4096, 4096]),
            ("ARITH-B-005", "divide", "divide", [1024, 1024]),
            ("ARITH-B-006", "maximum", "maximum", [4096, 4096]),
        ]

        for config in binaryConfigs {
            let shapeStr = config.shape.map(String.init).joined(separator: "x")
            let tensorType = "tensor<\(shapeStr)xf32>"

            let mlir = """
            module @\(config.op)_\(shapeStr) {
              func.func @main(%arg0: \(tensorType), %arg1: \(tensorType)) -> (\(tensorType)) {
                %0 = stablehlo.\(config.stablehloOp) %arg0, %arg1 : \(tensorType)
                return %0 : \(tensorType)
              }
            }
            """

            benchmarks.append(SimpleBenchmark(
                id: config.id,
                name: "\(config.op.capitalized) \(shapeStr)",
                category: "arithmetic",
                operation: config.op,
                configuration: ["shape": shapeStr, "dtype": "f32"],
                mlirProgram: mlir,
                inputGenerator: { client in
                    let gen = TestDataGenerator(seed: 42)
                    let a = try gen.createUniformFloat32Buffer(client: client, shape: config.shape)
                    let gen2 = TestDataGenerator(seed: 123)
                    let b = try gen2.createUniformFloat32Buffer(client: client, shape: config.shape, min: 0.1, max: 1.0)
                    return [a, b]
                },
                throughputCalculator: { timing in
                    let elements = Double(config.shape.reduce(1, *))
                    return ThroughputMetrics(
                        opsPerSecond: 1.0 / timing.gpuTime,
                        flops: elements / timing.gpuTime,
                        elementsPerSecond: elements / timing.gpuTime
                    )
                }
            ))
        }

        // Unary operations
        let unaryConfigs: [(id: String, op: String, stablehloOp: String, shape: [Int])] = [
            ("ARITH-U-001", "exp", "exponential", [1024, 1024]),
            ("ARITH-U-002", "log", "log", [4096, 4096]),
            ("ARITH-U-003", "tanh", "tanh", [1024, 1024]),
            ("ARITH-U-004", "sqrt", "sqrt", [4096, 4096]),
            ("ARITH-U-005", "rsqrt", "rsqrt", [4096, 4096]),
            ("ARITH-U-006", "sigmoid", "logistic", [1024, 1024]),
        ]

        for config in unaryConfigs {
            let shapeStr = config.shape.map(String.init).joined(separator: "x")
            let tensorType = "tensor<\(shapeStr)xf32>"

            let mlir = """
            module @\(config.op)_\(shapeStr) {
              func.func @main(%arg0: \(tensorType)) -> (\(tensorType)) {
                %0 = stablehlo.\(config.stablehloOp) %arg0 : \(tensorType)
                return %0 : \(tensorType)
              }
            }
            """

            benchmarks.append(SimpleBenchmark(
                id: config.id,
                name: "\(config.op.capitalized) \(shapeStr)",
                category: "arithmetic",
                operation: config.op,
                configuration: ["shape": shapeStr, "dtype": "f32"],
                mlirProgram: mlir,
                inputGenerator: { client in
                    let gen = TestDataGenerator(seed: 42)
                    // Use positive values for log/sqrt/rsqrt
                    return [try gen.createUniformFloat32Buffer(client: client, shape: config.shape, min: 0.1, max: 10.0)]
                },
                throughputCalculator: { timing in
                    let elements = Double(config.shape.reduce(1, *))
                    return ThroughputMetrics(
                        opsPerSecond: 1.0 / timing.gpuTime,
                        flops: elements / timing.gpuTime,
                        elementsPerSecond: elements / timing.gpuTime
                    )
                }
            ))
        }

        // Broadcast operations
        benchmarks.append(SimpleBenchmark(
            id: "ARITH-BC-001",
            name: "Add with Row Broadcast",
            category: "arithmetic",
            operation: "add_broadcast",
            configuration: ["shapes": "1024x1024 + 1024", "dtype": "f32"],
            mlirProgram: """
            module @add_row_broadcast {
              func.func @main(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024xf32>) -> (tensor<1024x1024xf32>) {
                %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<1024xf32>) -> tensor<1024x1024xf32>
                %1 = stablehlo.add %arg0, %0 : tensor<1024x1024xf32>
                return %1 : tensor<1024x1024xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let a = try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])
                let b = try gen.createUniformFloat32Buffer(client: client, shape: [1024])
                return [a, b]
            }
        ))

        benchmarks.append(SimpleBenchmark(
            id: "ARITH-BC-002",
            name: "Add with Scalar Broadcast",
            category: "arithmetic",
            operation: "add_broadcast",
            configuration: ["shapes": "1024x1024 + scalar", "dtype": "f32"],
            mlirProgram: """
            module @add_scalar_broadcast {
              func.func @main(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1xf32>) -> (tensor<1024x1024xf32>) {
                %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xf32>) -> tensor<1024x1024xf32>
                %1 = stablehlo.add %arg0, %0 : tensor<1024x1024xf32>
                return %1 : tensor<1024x1024xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let a = try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])
                let b = try client.createBuffer([1.5], shape: [1])
                return [a, b]
            }
        ))

        return benchmarks
    }

    // MARK: - Convolution Operations

    /// Convolution benchmarks.
    public static func convolutionBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // Conv2D configurations: (id, batch, height, width, inChannels, outChannels, kernel, stride, notes)
        let conv2dConfigs: [(id: String, batch: Int, h: Int, w: Int, cin: Int, cout: Int, k: Int, stride: Int, notes: String)] = [
            ("CONV-001", 1, 224, 224, 3, 64, 7, 2, "ResNet first layer"),
            ("CONV-002", 1, 56, 56, 64, 128, 3, 1, "ResNet stage2 3x3"),
            ("CONV-003", 1, 28, 28, 128, 256, 3, 1, "ResNet stage3 3x3"),
            ("CONV-004", 1, 14, 14, 256, 512, 3, 1, "ResNet stage4 3x3"),
            ("CONV-005", 32, 56, 56, 64, 64, 3, 1, "Batched conv"),
            ("CONV-006", 1, 112, 112, 32, 64, 1, 1, "1x1 pointwise"),
            ("CONV-007", 1, 56, 56, 128, 128, 3, 1, "Depthwise-like"),
        ]

        for config in conv2dConfigs {
            // Calculate output dimensions (no padding, so output shrinks)
            let outH = (config.h - config.k) / config.stride + 1
            let outW = (config.w - config.k) / config.stride + 1

            // NHWC format for input and output - simple stride-only convolution
            let moduleId = config.id.replacingOccurrences(of: "-", with: "_")
            let mlir = """
            module @conv2d_\(moduleId) {
              func.func @main(%input: tensor<\(config.batch)x\(config.h)x\(config.w)x\(config.cin)xf32>, %kernel: tensor<\(config.k)x\(config.k)x\(config.cin)x\(config.cout)xf32>) -> (tensor<\(config.batch)x\(outH)x\(outW)x\(config.cout)xf32>) {
                %0 = stablehlo.convolution %input, %kernel window_strides = [\(config.stride), \(config.stride)], feature_group_count = 1 : (tensor<\(config.batch)x\(config.h)x\(config.w)x\(config.cin)xf32>, tensor<\(config.k)x\(config.k)x\(config.cin)x\(config.cout)xf32>) -> tensor<\(config.batch)x\(outH)x\(outW)x\(config.cout)xf32>
                return %0 : tensor<\(config.batch)x\(outH)x\(outW)x\(config.cout)xf32>
              }
            }
            """

            let flops = FLOPSCalculator.conv2d(
                batchSize: config.batch,
                inputHeight: config.h,
                inputWidth: config.w,
                inputChannels: config.cin,
                outputChannels: config.cout,
                kernelHeight: config.k,
                kernelWidth: config.k,
                stride: config.stride
            )

            benchmarks.append(SimpleBenchmark(
                id: config.id,
                name: "Conv2D \(config.batch)x\(config.h)x\(config.w)x\(config.cin) k\(config.k)s\(config.stride)",
                category: "convolution",
                operation: "convolution",
                configuration: [
                    "batch": "\(config.batch)",
                    "input_shape": "\(config.h)x\(config.w)x\(config.cin)",
                    "output_channels": "\(config.cout)",
                    "kernel": "\(config.k)x\(config.k)",
                    "stride": "\(config.stride)",
                    "dtype": "f32",
                    "notes": config.notes
                ],
                mlirProgram: mlir,
                inputGenerator: { client in
                    let gen = TestDataGenerator(seed: 42)
                    let input = try gen.createUniformFloat32Buffer(client: client, shape: [config.batch, config.h, config.w, config.cin])
                    let kernel = try gen.createNormalFloat32Buffer(client: client, shape: [config.k, config.k, config.cin, config.cout], mean: 0, stdDev: 0.1)
                    return [input, kernel]
                },
                throughputCalculator: { timing in
                    let gflops = FLOPSCalculator.gflops(flops: flops, timeSeconds: timing.gpuTime)
                    return ThroughputMetrics(
                        opsPerSecond: 1.0 / timing.gpuTime,
                        flops: gflops * 1e9,
                        elementsPerSecond: Double(config.batch * outH * outW * config.cout) / timing.gpuTime
                    )
                }
            ))
        }

        return benchmarks
    }

    // MARK: - Normalization Operations

    /// Normalization benchmarks.
    public static func normalizationBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // Batch norm inference configs (NHWC format)
        let batchNormConfigs: [(id: String, batch: Int, h: Int, w: Int, c: Int, notes: String)] = [
            ("NORM-BN-001", 1, 56, 56, 64, "ResNet BN"),
            ("NORM-BN-002", 32, 56, 56, 64, "Batched ResNet BN"),
            ("NORM-BN-003", 1, 28, 28, 128, "Mid-layer BN"),
            ("NORM-BN-004", 1, 14, 14, 256, "Late-layer BN"),
        ]

        for config in batchNormConfigs {
            let moduleId = config.id.replacingOccurrences(of: "-", with: "_")
            let mlir = """
            module @batch_norm_\(moduleId) {
              func.func @main(%input: tensor<\(config.batch)x\(config.h)x\(config.w)x\(config.c)xf32>, %scale: tensor<\(config.c)xf32>, %offset: tensor<\(config.c)xf32>, %mean: tensor<\(config.c)xf32>, %variance: tensor<\(config.c)xf32>) -> (tensor<\(config.batch)x\(config.h)x\(config.w)x\(config.c)xf32>) {
                %0 = stablehlo.batch_norm_inference %input, %scale, %offset, %mean, %variance, epsilon = 1.0e-05, feature_index = 3 : (tensor<\(config.batch)x\(config.h)x\(config.w)x\(config.c)xf32>, tensor<\(config.c)xf32>, tensor<\(config.c)xf32>, tensor<\(config.c)xf32>, tensor<\(config.c)xf32>) -> tensor<\(config.batch)x\(config.h)x\(config.w)x\(config.c)xf32>
                return %0 : tensor<\(config.batch)x\(config.h)x\(config.w)x\(config.c)xf32>
              }
            }
            """

            let elements = config.batch * config.h * config.w * config.c

            benchmarks.append(SimpleBenchmark(
                id: config.id,
                name: "BatchNorm \(config.batch)x\(config.h)x\(config.w)x\(config.c)",
                category: "normalization",
                operation: "batch_norm_inference",
                configuration: [
                    "batch": "\(config.batch)",
                    "shape": "\(config.h)x\(config.w)x\(config.c)",
                    "dtype": "f32",
                    "notes": config.notes
                ],
                mlirProgram: mlir,
                inputGenerator: { client in
                    let gen = TestDataGenerator(seed: 42)
                    let input = try gen.createUniformFloat32Buffer(client: client, shape: [config.batch, config.h, config.w, config.c])
                    let scale = try gen.createUniformFloat32Buffer(client: client, shape: [config.c], min: 0.8, max: 1.2)
                    let offset = try gen.createUniformFloat32Buffer(client: client, shape: [config.c], min: -0.1, max: 0.1)
                    let mean = try gen.createUniformFloat32Buffer(client: client, shape: [config.c], min: -0.5, max: 0.5)
                    let variance = try gen.createUniformFloat32Buffer(client: client, shape: [config.c], min: 0.5, max: 1.5)
                    return [input, scale, offset, mean, variance]
                },
                throughputCalculator: { timing in
                    // BatchNorm: ~5 ops per element (subtract mean, multiply by scale, divide by sqrt(var), add offset)
                    let flops = Double(elements) * 5
                    return ThroughputMetrics(
                        opsPerSecond: 1.0 / timing.gpuTime,
                        flops: flops / timing.gpuTime,
                        elementsPerSecond: Double(elements) / timing.gpuTime
                    )
                }
            ))
        }

        // Layer norm benchmarks - hand-rolled since there's no native stablehlo.layer_norm op
        // Uses reduce + broadcast + elementwise ops to implement: (x - mean) / sqrt(var + eps) * gamma + beta
        // IMPORTANT: Use %arg0, %arg1, %arg2 naming to avoid MLIR alphabetical argument sorting
        let layerNormConfigs: [(id: String, batch: Int, seq: Int, hidden: Int, notes: String)] = [
            ("NORM-LN-001", 1, 128, 768, "BERT-base single"),
            ("NORM-LN-002", 32, 128, 768, "BERT-base batched"),
            ("NORM-LN-003", 1, 512, 1024, "BERT-large single"),
            ("NORM-LN-004", 8, 2048, 768, "Long sequence"),
        ]

        for config in layerNormConfigs {
            let moduleId = config.id.replacingOccurrences(of: "-", with: "_")
            let hiddenFloat = Float(config.hidden)

            // Layer norm MLIR: normalize over the last dimension (hidden)
            // Formula: (x - mean) / sqrt(var + eps) * gamma + beta
            let mlir = """
            module @layer_norm_\(moduleId) {
              func.func @main(%arg0: tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>, %arg1: tensor<\(config.hidden)xf32>, %arg2: tensor<\(config.hidden)xf32>) -> (tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>) {
                // Constants
                %eps = stablehlo.constant dense<1.0e-05> : tensor<f32>
                %hidden_size = stablehlo.constant dense<\(hiddenFloat)> : tensor<f32>
                %zero = stablehlo.constant dense<0.0> : tensor<f32>

                // Compute mean: sum over hidden dim, then divide by hidden_size
                %sum = stablehlo.reduce %arg0, %zero applies stablehlo.add across dimensions = [2] : (tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>, tensor<f32>) -> tensor<\(config.batch)x\(config.seq)xf32>
                %hidden_bc = stablehlo.broadcast_in_dim %hidden_size, dims = [] : (tensor<f32>) -> tensor<\(config.batch)x\(config.seq)xf32>
                %mean = stablehlo.divide %sum, %hidden_bc : tensor<\(config.batch)x\(config.seq)xf32>

                // Broadcast mean back to input shape
                %mean_bc = stablehlo.broadcast_in_dim %mean, dims = [0, 1] : (tensor<\(config.batch)x\(config.seq)xf32>) -> tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>

                // Compute x - mean
                %centered = stablehlo.subtract %arg0, %mean_bc : tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>

                // Compute variance: mean of squared centered values
                %sq = stablehlo.multiply %centered, %centered : tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>
                %sq_sum = stablehlo.reduce %sq, %zero applies stablehlo.add across dimensions = [2] : (tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>, tensor<f32>) -> tensor<\(config.batch)x\(config.seq)xf32>
                %var = stablehlo.divide %sq_sum, %hidden_bc : tensor<\(config.batch)x\(config.seq)xf32>

                // Add epsilon and compute rsqrt
                %eps_bc = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<\(config.batch)x\(config.seq)xf32>
                %var_eps = stablehlo.add %var, %eps_bc : tensor<\(config.batch)x\(config.seq)xf32>
                %rstd = stablehlo.rsqrt %var_eps : tensor<\(config.batch)x\(config.seq)xf32>

                // Broadcast rstd and apply normalization
                %rstd_bc = stablehlo.broadcast_in_dim %rstd, dims = [0, 1] : (tensor<\(config.batch)x\(config.seq)xf32>) -> tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>
                %normalized = stablehlo.multiply %centered, %rstd_bc : tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>

                // Apply gamma (%arg1) and beta (%arg2) (affine transformation)
                %gamma_bc = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<\(config.hidden)xf32>) -> tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>
                %beta_bc = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<\(config.hidden)xf32>) -> tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>
                %scaled = stablehlo.multiply %normalized, %gamma_bc : tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>
                %result = stablehlo.add %scaled, %beta_bc : tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>

                return %result : tensor<\(config.batch)x\(config.seq)x\(config.hidden)xf32>
              }
            }
            """

            let elements = config.batch * config.seq * config.hidden
            // Capture config values individually for the closure
            let batch = config.batch
            let seq = config.seq
            let hidden = config.hidden

            benchmarks.append(SimpleBenchmark(
                id: config.id,
                name: "LayerNorm \(config.batch)x\(config.seq)x\(config.hidden)",
                category: "normalization",
                operation: "layer_norm",
                configuration: [
                    "batch": "\(config.batch)",
                    "seq": "\(config.seq)",
                    "hidden": "\(config.hidden)",
                    "dtype": "f32",
                    "notes": config.notes
                ],
                mlirProgram: mlir,
                inputGenerator: { client in
                    let gen = TestDataGenerator(seed: 42)
                    // Order: input (%arg0), gamma (%arg1), beta (%arg2)
                    let input = try gen.createUniformFloat32Buffer(client: client, shape: [batch, seq, hidden])
                    let gamma = try gen.createUniformFloat32Buffer(client: client, shape: [hidden], min: 0.8, max: 1.2)
                    let beta = try gen.createUniformFloat32Buffer(client: client, shape: [hidden], min: -0.1, max: 0.1)
                    return [input, gamma, beta]
                },
                throughputCalculator: { timing in
                    // LayerNorm: ~10 ops per element (subtract, square, reduce, divide, rsqrt, multiply, add)
                    let flops = Double(elements) * 10
                    return ThroughputMetrics(
                        opsPerSecond: 1.0 / timing.gpuTime,
                        flops: flops / timing.gpuTime,
                        elementsPerSecond: Double(elements) / timing.gpuTime
                    )
                }
            ))
        }

        return benchmarks
    }

    // MARK: - Control Flow Operations (Medium Priority)

    /// Control flow benchmarks (while, if operations).
    public static func controlFlowBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // CF-001: Simple counter while loop
        benchmarks.append(SimpleBenchmark(
            id: "CF-001",
            name: "While Loop Simple Counter (100 iters)",
            category: "control_flow",
            operation: "while",
            configuration: ["iterations": "100", "notes": "Simple counter loop"],
            mlirProgram: """
            module @while_simple_counter {
              func.func @main(%arg0: tensor<i32>) -> (tensor<i32>) {
                %limit = stablehlo.constant dense<100> : tensor<i32>
                %one = stablehlo.constant dense<1> : tensor<i32>
                %result = stablehlo.while(%iter = %arg0) : tensor<i32>
                  cond {
                    %cond = stablehlo.compare LT, %iter, %limit : (tensor<i32>, tensor<i32>) -> tensor<i1>
                    stablehlo.return %cond : tensor<i1>
                  } do {
                    %next = stablehlo.add %iter, %one : tensor<i32>
                    stablehlo.return %next : tensor<i32>
                  }
                return %result : tensor<i32>
              }
            }
            """,
            inputGenerator: { client in
                return [try client.createBuffer([Int32(0)], shape: [1])]
            }
        ))

        // CF-002: While loop with accumulation
        benchmarks.append(SimpleBenchmark(
            id: "CF-002",
            name: "While Loop Accumulate (100 iters)",
            category: "control_flow",
            operation: "while",
            configuration: ["iterations": "100", "shape": "1024", "notes": "Sum accumulation loop"],
            mlirProgram: """
            module @while_accumulate {
              func.func @main(%arg0: tensor<1024xf32>) -> (tensor<1024xf32>) {
                %zero_counter = stablehlo.constant dense<0> : tensor<i32>
                %limit = stablehlo.constant dense<100> : tensor<i32>
                %one = stablehlo.constant dense<1> : tensor<i32>
                %zero_acc = stablehlo.constant dense<0.0> : tensor<1024xf32>

                %result:2 = stablehlo.while(%counter = %zero_counter, %acc = %zero_acc) : tensor<i32>, tensor<1024xf32>
                  cond {
                    %cond = stablehlo.compare LT, %counter, %limit : (tensor<i32>, tensor<i32>) -> tensor<i1>
                    stablehlo.return %cond : tensor<i1>
                  } do {
                    %next_counter = stablehlo.add %counter, %one : tensor<i32>
                    %next_acc = stablehlo.add %acc, %arg0 : tensor<1024xf32>
                    stablehlo.return %next_counter, %next_acc : tensor<i32>, tensor<1024xf32>
                  }
                return %result#1 : tensor<1024xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [1024])]
            }
        ))

        // CF-003: While loop with matrix multiplication
        benchmarks.append(SimpleBenchmark(
            id: "CF-003",
            name: "While Loop MatMul (10 iters)",
            category: "control_flow",
            operation: "while",
            configuration: ["iterations": "10", "shape": "128x128", "notes": "Matrix multiplication in loop"],
            mlirProgram: """
            module @while_matmul {
              func.func @main(%arg0: tensor<128x128xf32>) -> (tensor<128x128xf32>) {
                %zero_counter = stablehlo.constant dense<0> : tensor<i32>
                %limit = stablehlo.constant dense<10> : tensor<i32>
                %one = stablehlo.constant dense<1> : tensor<i32>

                %result:2 = stablehlo.while(%counter = %zero_counter, %matrix = %arg0) : tensor<i32>, tensor<128x128xf32>
                  cond {
                    %cond = stablehlo.compare LT, %counter, %limit : (tensor<i32>, tensor<i32>) -> tensor<i1>
                    stablehlo.return %cond : tensor<i1>
                  } do {
                    %next_counter = stablehlo.add %counter, %one : tensor<i32>
                    %next_matrix = stablehlo.dot_general %matrix, %arg0, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
                    stablehlo.return %next_counter, %next_matrix : tensor<i32>, tensor<128x128xf32>
                  }
                return %result#1 : tensor<128x128xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [128, 128], min: -0.1, max: 0.1)]
            },
            throughputCalculator: { timing in
                // 10 iterations of 128x128 matmul
                let flops = FLOPSCalculator.matmul(m: 128, n: 128, k: 128) * 10
                let gflops = FLOPSCalculator.gflops(flops: flops, timeSeconds: timing.gpuTime)
                return ThroughputMetrics(
                    opsPerSecond: 1.0 / timing.gpuTime,
                    flops: gflops * 1e9,
                    elementsPerSecond: Double(128 * 128 * 10) / timing.gpuTime
                )
            }
        ))

        // CF-004: Simple if/conditional
        benchmarks.append(SimpleBenchmark(
            id: "CF-004",
            name: "If Conditional Simple",
            category: "control_flow",
            operation: "if",
            configuration: ["shape": "1024x1024", "notes": "Simple branch selection"],
            mlirProgram: """
            module @if_simple {
              func.func @main(%pred: tensor<i1>, %arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>) {
                %result = stablehlo.if %pred -> tensor<1024x1024xf32> {
                  stablehlo.return %arg0 : tensor<1024x1024xf32>
                } else {
                  stablehlo.return %arg1 : tensor<1024x1024xf32>
                }
                return %result : tensor<1024x1024xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                // Create boolean tensor as int1 (using Int8 array with value 1 for true)
                let pred = try client.createBuffer([Int8(1)], shape: [], elementType: .int1)
                let a = try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])
                let b = try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])
                return [pred, a, b]
            }
        ))

        // CF-005: If with heavy compute in branches
        benchmarks.append(SimpleBenchmark(
            id: "CF-005",
            name: "If Conditional with Compute",
            category: "control_flow",
            operation: "if",
            configuration: ["shape": "512x512", "notes": "Conditional with heavy compute in branches"],
            mlirProgram: """
            module @if_compute {
              func.func @main(%pred: tensor<i1>, %arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>) -> (tensor<512x512xf32>) {
                %result = stablehlo.if %pred -> tensor<512x512xf32> {
                  // True branch: matmul
                  %mm = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
                  stablehlo.return %mm : tensor<512x512xf32>
                } else {
                  // False branch: elementwise ops
                  %add = stablehlo.add %arg0, %arg1 : tensor<512x512xf32>
                  %exp = stablehlo.exponential %add : tensor<512x512xf32>
                  stablehlo.return %exp : tensor<512x512xf32>
                }
                return %result : tensor<512x512xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                // Create boolean tensor as int1 (using Int8 array with value 1 for true)
                let pred = try client.createBuffer([Int8(1)], shape: [], elementType: .int1)
                let a = try gen.createUniformFloat32Buffer(client: client, shape: [512, 512], min: -0.1, max: 0.1)
                let b = try gen.createUniformFloat32Buffer(client: client, shape: [512, 512], min: -0.1, max: 0.1)
                return [pred, a, b]
            }
        ))

        return benchmarks
    }

    // MARK: - FFT Operations (Medium Priority)

    /// FFT benchmarks.
    /// Note: FFT operations require complex number types (complex<f32>) which need
    /// special handling. These benchmarks are currently disabled pending complex
    /// type support in the public API. The MLIR programs are preserved for reference.
    public static func fftBenchmarks() -> [Benchmark] {
        // FFT benchmarks are disabled until complex64 type is exposed in the public API.
        // The operations are supported internally but need buffer creation support.
        //
        // Planned benchmarks:
        // - FFT-001: FFT 1D (1024 points)
        // - FFT-002: FFT 1D (4096 points)
        // - FFT-003: IFFT 1D (1024 points)
        // - FFT-004: RFFT 1D (1024 points)
        // - FFT-005: IRFFT 1D (513 -> 1024)
        // - FFT-006: FFT 2D (64x64)
        return []
    }

    // MARK: - Indexing and Slicing Operations (Medium Priority)

    /// Indexing and slicing benchmarks.
    public static func indexingBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // IDX-001: Static slice extraction
        benchmarks.append(SimpleBenchmark(
            id: "IDX-001",
            name: "Slice Static 1024x1024 -> 512x512",
            category: "indexing",
            operation: "slice",
            configuration: ["input_shape": "1024x1024", "output_shape": "512x512", "notes": "Static slice extraction"],
            mlirProgram: """
            module @slice_static {
              func.func @main(%arg0: tensor<1024x1024xf32>) -> (tensor<512x512xf32>) {
                %0 = stablehlo.slice %arg0 [256:768, 256:768] : (tensor<1024x1024xf32>) -> tensor<512x512xf32>
                return %0 : tensor<512x512xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])]
            }
        ))

        // IDX-002: Dynamic slice
        benchmarks.append(SimpleBenchmark(
            id: "IDX-002",
            name: "Dynamic Slice 1024x1024 -> 256x256",
            category: "indexing",
            operation: "dynamic_slice",
            configuration: ["input_shape": "1024x1024", "slice_shape": "256x256", "notes": "Runtime-determined slice"],
            mlirProgram: """
            module @dynamic_slice {
              func.func @main(%arg0: tensor<1024x1024xf32>, %start0: tensor<i32>, %start1: tensor<i32>) -> (tensor<256x256xf32>) {
                %0 = stablehlo.dynamic_slice %arg0, %start0, %start1, sizes = [256, 256] : (tensor<1024x1024xf32>, tensor<i32>, tensor<i32>) -> tensor<256x256xf32>
                return %0 : tensor<256x256xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let input = try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])
                let start0 = try client.createBuffer([Int32(128)], shape: [1])
                let start1 = try client.createBuffer([Int32(128)], shape: [1])
                return [input, start0, start1]
            }
        ))

        // IDX-003: Dynamic update slice
        benchmarks.append(SimpleBenchmark(
            id: "IDX-003",
            name: "Dynamic Update Slice 1024x1024",
            category: "indexing",
            operation: "dynamic_update_slice",
            configuration: ["input_shape": "1024x1024", "update_shape": "256x256", "notes": "In-place update"],
            mlirProgram: """
            module @dynamic_update_slice {
              func.func @main(%arg0: tensor<1024x1024xf32>, %update: tensor<256x256xf32>, %start0: tensor<i32>, %start1: tensor<i32>) -> (tensor<1024x1024xf32>) {
                %0 = stablehlo.dynamic_update_slice %arg0, %update, %start0, %start1 : (tensor<1024x1024xf32>, tensor<256x256xf32>, tensor<i32>, tensor<i32>) -> tensor<1024x1024xf32>
                return %0 : tensor<1024x1024xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let input = try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])
                let update = try gen.createUniformFloat32Buffer(client: client, shape: [256, 256])
                let start0 = try client.createBuffer([Int32(128)], shape: [1])
                let start1 = try client.createBuffer([Int32(128)], shape: [1])
                return [input, update, start0, start1]
            }
        ))

        // IDX-004: Gather operation
        benchmarks.append(SimpleBenchmark(
            id: "IDX-004",
            name: "Gather Embedding Lookup 10000x256",
            category: "indexing",
            operation: "gather",
            configuration: ["table_shape": "10000x256", "indices": "1024", "notes": "Embedding lookup pattern"],
            mlirProgram: """
            module @gather_embedding {
              func.func @main(%arg0: tensor<10000x256xf32>, %indices: tensor<1024xi32>) -> (tensor<1024x256xf32>) {
                %0 = stablehlo.gather %arg0, %indices,
                  offset_dims = [1],
                  collapsed_slice_dims = [0],
                  start_index_map = [0],
                  index_vector_dim = 1,
                  slice_sizes = [1, 256] : (tensor<10000x256xf32>, tensor<1024xi32>) -> tensor<1024x256xf32>
                return %0 : tensor<1024x256xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let table = try gen.createUniformFloat32Buffer(client: client, shape: [10000, 256])
                // Generate random indices in range [0, 10000)
                var indices: [Int32] = []
                for _ in 0..<1024 {
                    indices.append(Int32(gen.uniformFloat32(count: 1)[0] * 9999))
                }
                let indicesBuffer = try client.createBuffer(indices, shape: [1024])
                return [table, indicesBuffer]
            },
            throughputCalculator: { timing in
                // 1024 lookups, each fetching 256 floats
                let elements = 1024 * 256
                let bytesRead = Double(elements * 4)
                return ThroughputMetrics(
                    opsPerSecond: 1.0 / timing.gpuTime,
                    memoryBandwidthGBps: bytesRead / timing.gpuTime / 1e9,
                    elementsPerSecond: Double(elements) / timing.gpuTime
                )
            }
        ))

        // IDX-005: Scatter operation
        benchmarks.append(SimpleBenchmark(
            id: "IDX-005",
            name: "Scatter Add 10000x256",
            category: "indexing",
            operation: "scatter",
            configuration: ["table_shape": "10000x256", "updates": "1024x256", "notes": "Scatter add pattern"],
            mlirProgram: """
            module @scatter_add {
              func.func @main(%arg0: tensor<10000x256xf32>, %indices: tensor<1024x1xi32>, %updates: tensor<1024x256xf32>) -> (tensor<10000x256xf32>) {
                %0 = stablehlo.scatter %arg0, %indices, %updates,
                  update_window_dims = [1],
                  inserted_window_dims = [0],
                  scatter_dims_to_operand_dims = [0],
                  index_vector_dim = 1,
                  unique_indices = false
                  ((update_computation)) { ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
                    %sum = stablehlo.add %lhs, %rhs : tensor<f32>
                    stablehlo.return %sum : tensor<f32>
                  } : (tensor<10000x256xf32>, tensor<1024x1xi32>, tensor<1024x256xf32>) -> tensor<10000x256xf32>
                return %0 : tensor<10000x256xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let table = try gen.createUniformFloat32Buffer(client: client, shape: [10000, 256])
                // Generate random indices in range [0, 10000)
                var indices: [Int32] = []
                for _ in 0..<1024 {
                    indices.append(Int32(gen.uniformFloat32(count: 1)[0] * 9999))
                }
                let indicesBuffer = try client.createBuffer(indices, shape: [1024, 1])
                let updates = try gen.createUniformFloat32Buffer(client: client, shape: [1024, 256])
                return [table, indicesBuffer, updates]
            }
        ))

        // IDX-006: Concatenate operation
        benchmarks.append(SimpleBenchmark(
            id: "IDX-006",
            name: "Concatenate 4x 512x512",
            category: "indexing",
            operation: "concatenate",
            configuration: ["input_shapes": "4x 512x512", "output_shape": "2048x512", "notes": "Tensor concatenation"],
            mlirProgram: """
            module @concatenate {
              func.func @main(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> (tensor<2048x512xf32>) {
                %0 = stablehlo.concatenate %arg0, %arg1, %arg2, %arg3, dim = 0 : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<2048x512xf32>
                return %0 : tensor<2048x512xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let a = try gen.createUniformFloat32Buffer(client: client, shape: [512, 512])
                let b = try gen.createUniformFloat32Buffer(client: client, shape: [512, 512])
                let c = try gen.createUniformFloat32Buffer(client: client, shape: [512, 512])
                let d = try gen.createUniformFloat32Buffer(client: client, shape: [512, 512])
                return [a, b, c, d]
            }
        ))

        // IDX-007: Pad operation
        benchmarks.append(SimpleBenchmark(
            id: "IDX-007",
            name: "Pad 512x512 -> 640x640",
            category: "indexing",
            operation: "pad",
            configuration: ["input_shape": "512x512", "output_shape": "640x640", "padding": "64 on each side", "notes": "Constant padding"],
            mlirProgram: """
            module @pad {
              func.func @main(%arg0: tensor<512x512xf32>) -> (tensor<640x640xf32>) {
                %padding_value = stablehlo.constant dense<0.0> : tensor<f32>
                %0 = stablehlo.pad %arg0, %padding_value, low = [64, 64], high = [64, 64], interior = [0, 0] : (tensor<512x512xf32>, tensor<f32>) -> tensor<640x640xf32>
                return %0 : tensor<640x640xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [512, 512])]
            }
        ))

        return benchmarks
    }

    // MARK: - Reduce Window Operations (Pooling)

    /// Reduce window benchmarks (pooling operations).
    public static func reduceWindowBenchmarks() -> [Benchmark] {
        var benchmarks: [Benchmark] = []

        // RED-007: Max pooling 2x2
        benchmarks.append(SimpleBenchmark(
            id: "RED-007",
            name: "Max Pool 2x2 (224x224x64)",
            category: "reduction",
            operation: "reduce_window_max",
            configuration: ["input_shape": "1x224x224x64", "window": "2x2", "stride": "2", "notes": "Max pooling"],
            mlirProgram: """
            module @max_pool_2x2 {
              func.func @main(%arg0: tensor<1x224x224x64xf32>) -> (tensor<1x112x112x64xf32>) {
                %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
                %0 = stablehlo.reduce_window %arg0, %init applies stablehlo.maximum
                  window_dimensions = [1, 2, 2, 1],
                  window_strides = [1, 2, 2, 1],
                  padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
                  : (tensor<1x224x224x64xf32>, tensor<f32>) -> tensor<1x112x112x64xf32>
                return %0 : tensor<1x112x112x64xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [1, 224, 224, 64])]
            },
            throughputCalculator: { timing in
                // Each output element compares 4 input elements
                let outputElements = 1 * 112 * 112 * 64
                let comparisons = outputElements * 4
                return ThroughputMetrics(
                    opsPerSecond: 1.0 / timing.gpuTime,
                    flops: Double(comparisons) / timing.gpuTime,
                    elementsPerSecond: Double(outputElements) / timing.gpuTime
                )
            }
        ))

        // RED-008: Average pooling 2x2
        benchmarks.append(SimpleBenchmark(
            id: "RED-008",
            name: "Avg Pool 2x2 (56x56x256)",
            category: "reduction",
            operation: "reduce_window_avg",
            configuration: ["input_shape": "1x56x56x256", "window": "2x2", "stride": "2", "notes": "Average pooling"],
            mlirProgram: """
            module @avg_pool_2x2 {
              func.func @main(%arg0: tensor<1x56x56x256xf32>) -> (tensor<1x28x28x256xf32>) {
                %init = stablehlo.constant dense<0.0> : tensor<f32>
                %sum = stablehlo.reduce_window %arg0, %init applies stablehlo.add
                  window_dimensions = [1, 2, 2, 1],
                  window_strides = [1, 2, 2, 1],
                  padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
                  : (tensor<1x56x56x256xf32>, tensor<f32>) -> tensor<1x28x28x256xf32>
                %divisor = stablehlo.constant dense<4.0> : tensor<1x28x28x256xf32>
                %avg = stablehlo.divide %sum, %divisor : tensor<1x28x28x256xf32>
                return %avg : tensor<1x28x28x256xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [1, 56, 56, 256])]
            },
            throughputCalculator: { timing in
                // Sum 4 elements + 1 divide per output
                let outputElements = 1 * 28 * 28 * 256
                let ops = outputElements * 5 // 4 adds + 1 divide
                return ThroughputMetrics(
                    opsPerSecond: 1.0 / timing.gpuTime,
                    flops: Double(ops) / timing.gpuTime,
                    elementsPerSecond: Double(outputElements) / timing.gpuTime
                )
            }
        ))

        // Global average pooling (common in CNNs)
        benchmarks.append(SimpleBenchmark(
            id: "RED-009",
            name: "Global Avg Pool (7x7x512)",
            category: "reduction",
            operation: "reduce_window_avg",
            configuration: ["input_shape": "1x7x7x512", "window": "7x7", "notes": "Global average pooling"],
            mlirProgram: """
            module @global_avg_pool {
              func.func @main(%arg0: tensor<1x7x7x512xf32>) -> (tensor<1x1x1x512xf32>) {
                %init = stablehlo.constant dense<0.0> : tensor<f32>
                %sum = stablehlo.reduce_window %arg0, %init applies stablehlo.add
                  window_dimensions = [1, 7, 7, 1],
                  window_strides = [1, 1, 1, 1],
                  padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
                  : (tensor<1x7x7x512xf32>, tensor<f32>) -> tensor<1x1x1x512xf32>
                %divisor = stablehlo.constant dense<49.0> : tensor<1x1x1x512xf32>
                %avg = stablehlo.divide %sum, %divisor : tensor<1x1x1x512xf32>
                return %avg : tensor<1x1x1x512xf32>
              }
            }
            """,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                return [try gen.createUniformFloat32Buffer(client: client, shape: [1, 7, 7, 512])]
            }
        ))

        return benchmarks
    }

    // MARK: - All Benchmarks

    /// Get all operation benchmarks (excluding model-level benchmarks).
    public static func all() -> [Benchmark] {
        var benchmarks: [Benchmark] = []
        benchmarks.append(contentsOf: matrixBenchmarks())
        benchmarks.append(contentsOf: reductionBenchmarks())
        benchmarks.append(contentsOf: arithmeticBenchmarks())
        benchmarks.append(contentsOf: convolutionBenchmarks())
        benchmarks.append(contentsOf: normalizationBenchmarks())
        benchmarks.append(contentsOf: controlFlowBenchmarks())
        benchmarks.append(contentsOf: fftBenchmarks())
        benchmarks.append(contentsOf: indexingBenchmarks())
        benchmarks.append(contentsOf: reduceWindowBenchmarks())
        return benchmarks
    }

    /// Get all benchmarks including model-level benchmarks.
    public static func allWithModels() -> [Benchmark] {
        var benchmarks = all()
        benchmarks.append(contentsOf: ModelBenchmarks.all())
        return benchmarks
    }

    /// Get benchmarks by priority.
    public static func critical() -> [Benchmark] {
        var benchmarks: [Benchmark] = []
        benchmarks.append(contentsOf: matrixBenchmarks())
        benchmarks.append(contentsOf: reductionBenchmarks())
        return benchmarks
    }
}
