// PowerEfficiencyBenchmarks.swift
// MetalHLO Benchmarks
//
// Power efficiency benchmarks for measuring throughput per watt.
// Note: Actual power measurement requires IOKit or external tools.
// This provides estimates based on published TDP values.

import Foundation
import MetalHLO

// MARK: - Apple Silicon Power Specifications

/// Power specifications for Apple Silicon chips.
public struct AppleSiliconPowerSpecs: Sendable {
    public let name: String
    /// CPU TDP in watts.
    public let cpuTDP: Double
    /// GPU TDP in watts (estimated).
    public let gpuTDP: Double
    /// Total chip TDP in watts.
    public let totalTDP: Double

    public init(name: String, cpuTDP: Double, gpuTDP: Double, totalTDP: Double) {
        self.name = name
        self.cpuTDP = cpuTDP
        self.gpuTDP = gpuTDP
        self.totalTDP = totalTDP
    }

    /// Known Apple Silicon power specifications (estimated from reviews/documentation).
    /// Note: Apple doesn't publish official TDP numbers; these are estimates.
    public static let knownChips: [String: AppleSiliconPowerSpecs] = [
        "M1": AppleSiliconPowerSpecs(name: "M1", cpuTDP: 10, gpuTDP: 10, totalTDP: 20),
        "M1 Pro": AppleSiliconPowerSpecs(name: "M1 Pro", cpuTDP: 20, gpuTDP: 20, totalTDP: 40),
        "M1 Max": AppleSiliconPowerSpecs(name: "M1 Max", cpuTDP: 30, gpuTDP: 30, totalTDP: 60),
        "M2": AppleSiliconPowerSpecs(name: "M2", cpuTDP: 12, gpuTDP: 12, totalTDP: 22),
        "M2 Pro": AppleSiliconPowerSpecs(name: "M2 Pro", cpuTDP: 25, gpuTDP: 25, totalTDP: 45),
        "M2 Max": AppleSiliconPowerSpecs(name: "M2 Max", cpuTDP: 35, gpuTDP: 35, totalTDP: 67),
        "M3": AppleSiliconPowerSpecs(name: "M3", cpuTDP: 12, gpuTDP: 12, totalTDP: 22),
        "M3 Pro": AppleSiliconPowerSpecs(name: "M3 Pro", cpuTDP: 25, gpuTDP: 25, totalTDP: 45),
        "M3 Max": AppleSiliconPowerSpecs(name: "M3 Max", cpuTDP: 40, gpuTDP: 40, totalTDP: 75),
        "M4": AppleSiliconPowerSpecs(name: "M4", cpuTDP: 12, gpuTDP: 12, totalTDP: 22),
        "M4 Pro": AppleSiliconPowerSpecs(name: "M4 Pro", cpuTDP: 28, gpuTDP: 28, totalTDP: 52),
        "M4 Max": AppleSiliconPowerSpecs(name: "M4 Max", cpuTDP: 45, gpuTDP: 45, totalTDP: 85),
    ]

    /// Detect power specs for current hardware.
    public static func detect(deviceName: String) -> AppleSiliconPowerSpecs {
        for (chipName, specs) in knownChips {
            if deviceName.contains(chipName) {
                return specs
            }
        }
        // Default conservative estimate
        return AppleSiliconPowerSpecs(name: deviceName, cpuTDP: 15, gpuTDP: 15, totalTDP: 30)
    }
}

// MARK: - Power Efficiency Result

/// Result of a power efficiency benchmark.
public struct PowerEfficiencyResult: Sendable {
    public let id: String
    public let name: String
    /// Total operations performed.
    public let totalOperations: Int64
    /// Total execution time in seconds.
    public let totalTimeSeconds: Double
    /// Throughput in operations per second.
    public let throughputOpsPerSecond: Double
    /// Estimated GPU power in watts.
    public let estimatedGPUPowerWatts: Double
    /// Estimated throughput per watt (ops/s/W).
    public let throughputPerWatt: Double
    /// GFLOPS achieved.
    public let gflops: Double
    /// GFLOPS per watt.
    public let gflopsPerWatt: Double
    /// Hardware specs used.
    public let hardwareSpecs: AppleSiliconPowerSpecs

    public init(
        id: String,
        name: String,
        totalOperations: Int64,
        totalTimeSeconds: Double,
        throughputOpsPerSecond: Double,
        estimatedGPUPowerWatts: Double,
        throughputPerWatt: Double,
        gflops: Double,
        gflopsPerWatt: Double,
        hardwareSpecs: AppleSiliconPowerSpecs
    ) {
        self.id = id
        self.name = name
        self.totalOperations = totalOperations
        self.totalTimeSeconds = totalTimeSeconds
        self.throughputOpsPerSecond = throughputOpsPerSecond
        self.estimatedGPUPowerWatts = estimatedGPUPowerWatts
        self.throughputPerWatt = throughputPerWatt
        self.gflops = gflops
        self.gflopsPerWatt = gflopsPerWatt
        self.hardwareSpecs = hardwareSpecs
    }

    /// Format as human-readable string.
    public func formatted() -> String {
        return """
        \(id): \(name)
          Hardware: \(hardwareSpecs.name) (est. GPU TDP: \(String(format: "%.0f", estimatedGPUPowerWatts))W)
          Duration: \(String(format: "%.2f", totalTimeSeconds))s
          Throughput: \(String(format: "%.2f", throughputOpsPerSecond / 1e6))M ops/s
          GFLOPS: \(String(format: "%.2f", gflops))
          Efficiency: \(String(format: "%.2f", gflopsPerWatt)) GFLOPS/W
        """
    }
}

// MARK: - Power Efficiency Benchmarks

/// Factory for power efficiency benchmarks.
public enum PowerEfficiencyBenchmarks {

    /// Power efficiency benchmark configuration.
    public struct PowerBenchmarkConfig: Sendable {
        public let id: String
        public let name: String
        public let description: String
        public let mlirProgram: String
        public let flopsPerExecution: Int64
        public let inputGenerator: @Sendable (Client) throws -> [Buffer]

        public init(
            id: String,
            name: String,
            description: String,
            mlirProgram: String,
            flopsPerExecution: Int64,
            inputGenerator: @escaping @Sendable (Client) throws -> [Buffer]
        ) {
            self.id = id
            self.name = name
            self.description = description
            self.mlirProgram = mlirProgram
            self.flopsPerExecution = flopsPerExecution
            self.inputGenerator = inputGenerator
        }
    }

    /// Get all power efficiency benchmark configurations.
    public static func all() -> [PowerBenchmarkConfig] {
        var configs: [PowerBenchmarkConfig] = []

        // PWR-001: Sustained inference throughput
        // Matrix multiplication 1024x1024 repeated
        configs.append(PowerBenchmarkConfig(
            id: "PWR-001",
            name: "Sustained Inference (1024x1024 MatMul)",
            description: "Measures throughput per watt during sustained matrix multiplication",
            mlirProgram: """
            module @sustained_matmul {
              func.func @main(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>) {
                %c = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
                return %c : tensor<1024x1024xf32>
              }
            }
            """,
            flopsPerExecution: 2 * 1024 * 1024 * 1024,  // 2 * M * N * K
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let a = try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])
                let b = try gen.createUniformFloat32Buffer(client: client, shape: [1024, 1024])
                return [a, b]
            }
        ))

        // PWR-002: Burst compute (larger matrices)
        configs.append(PowerBenchmarkConfig(
            id: "PWR-002",
            name: "Burst Compute (2048x2048 MatMul)",
            description: "Measures peak power draw during intensive computation",
            mlirProgram: """
            module @burst_matmul {
              func.func @main(%a: tensor<2048x2048xf32>, %b: tensor<2048x2048xf32>) -> (tensor<2048x2048xf32>) {
                %c = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
                return %c : tensor<2048x2048xf32>
              }
            }
            """,
            flopsPerExecution: 2 * 2048 * 2048 * 2048,
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let a = try gen.createUniformFloat32Buffer(client: client, shape: [2048, 2048])
                let b = try gen.createUniformFloat32Buffer(client: client, shape: [2048, 2048])
                return [a, b]
            }
        ))

        // PWR-003: Transformer attention efficiency
        configs.append(PowerBenchmarkConfig(
            id: "PWR-003",
            name: "Transformer Attention Efficiency",
            description: "Measures efficiency of attention computation",
            mlirProgram: """
            module @attention_efficiency {
              func.func @main(%q: tensor<8x12x128x64xf32>, %k: tensor<8x12x128x64xf32>, %v: tensor<8x12x128x64xf32>) -> (tensor<8x12x128x64xf32>) {
                // Q @ K^T
                %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] : (tensor<8x12x128x64xf32>) -> tensor<8x12x64x128xf32>
                %scores = stablehlo.dot_general %q, %kt, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<8x12x128x64xf32>, tensor<8x12x64x128xf32>) -> tensor<8x12x128x128xf32>

                // Scale
                %scale = stablehlo.constant dense<0.125> : tensor<f32>
                %scale_bc = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<8x12x128x128xf32>
                %scaled = stablehlo.multiply %scores, %scale_bc : tensor<8x12x128x128xf32>

                // Softmax (simplified - just exp and normalize)
                %exp = stablehlo.exponential %scaled : tensor<8x12x128x128xf32>
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %sum = stablehlo.reduce %exp, %zero applies stablehlo.add across dimensions = [3] : (tensor<8x12x128x128xf32>, tensor<f32>) -> tensor<8x12x128xf32>
                %sum_bc = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<8x12x128xf32>) -> tensor<8x12x128x128xf32>
                %attn = stablehlo.divide %exp, %sum_bc : tensor<8x12x128x128xf32>

                // Attn @ V
                %out = stablehlo.dot_general %attn, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<8x12x128x128xf32>, tensor<8x12x128x64xf32>) -> tensor<8x12x128x64xf32>
                return %out : tensor<8x12x128x64xf32>
              }
            }
            """,
            // Q@K^T: 8*12*128*128*64*2 + Attn@V: 8*12*128*64*128*2 + softmax ops
            flopsPerExecution: 2 * (8 * 12 * 128 * 128 * 64) + 2 * (8 * 12 * 128 * 64 * 128) + (8 * 12 * 128 * 128 * 3),
            inputGenerator: { client in
                let gen = TestDataGenerator(seed: 42)
                let q = try gen.createUniformFloat32Buffer(client: client, shape: [8, 12, 128, 64])
                let k = try gen.createUniformFloat32Buffer(client: client, shape: [8, 12, 128, 64])
                let v = try gen.createUniformFloat32Buffer(client: client, shape: [8, 12, 128, 64])
                return [q, k, v]
            }
        ))

        return configs
    }
}

// MARK: - Power Efficiency Runner

/// Runner for power efficiency benchmarks.
public struct PowerEfficiencyRunner: Sendable {

    /// Duration for sustained benchmarks in seconds.
    public let sustainedDurationSeconds: Double
    /// Warmup iterations before measurement.
    public let warmupIterations: Int

    public init(sustainedDurationSeconds: Double = 5.0, warmupIterations: Int = 10) {
        self.sustainedDurationSeconds = sustainedDurationSeconds
        self.warmupIterations = warmupIterations
    }

    /// Run a power efficiency benchmark.
    public func run(_ config: PowerEfficiencyBenchmarks.PowerBenchmarkConfig) throws -> PowerEfficiencyResult {
        let client = try Client.create()

        // Detect hardware
        let gpuCalc = GPUMetricsCalculator()
        let powerSpecs = AppleSiliconPowerSpecs.detect(deviceName: gpuCalc.hardwareSpecs.name)

        // Compile program
        let exe = try client.compile(config.mlirProgram)

        // Create inputs
        let inputs = try config.inputGenerator(client)

        // Warmup
        for _ in 0..<warmupIterations {
            let _ = try exe.execute(inputs)
        }

        // Sustained execution for the specified duration
        var totalIterations: Int64 = 0
        let startTime = CFAbsoluteTimeGetCurrent()
        let endTime = startTime + sustainedDurationSeconds

        while CFAbsoluteTimeGetCurrent() < endTime {
            let _ = try exe.execute(inputs)
            totalIterations += 1
        }

        let actualDuration = CFAbsoluteTimeGetCurrent() - startTime
        let totalFlops = totalIterations * config.flopsPerExecution
        let throughputOps = Double(totalIterations) / actualDuration
        let gflops = Double(totalFlops) / actualDuration / 1e9

        // Estimate power (assuming GPU runs at ~TDP during sustained compute)
        let estimatedPower = powerSpecs.gpuTDP
        let gflopsPerWatt = gflops / estimatedPower
        let throughputPerWatt = throughputOps / estimatedPower

        return PowerEfficiencyResult(
            id: config.id,
            name: config.name,
            totalOperations: totalIterations,
            totalTimeSeconds: actualDuration,
            throughputOpsPerSecond: throughputOps,
            estimatedGPUPowerWatts: estimatedPower,
            throughputPerWatt: throughputPerWatt,
            gflops: gflops,
            gflopsPerWatt: gflopsPerWatt,
            hardwareSpecs: powerSpecs
        )
    }

    /// Run all power efficiency benchmarks.
    public func runAll() throws -> [PowerEfficiencyResult] {
        var results: [PowerEfficiencyResult] = []
        for config in PowerEfficiencyBenchmarks.all() {
            let result = try run(config)
            results.append(result)
        }
        return results
    }
}
