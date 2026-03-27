// MemoryBenchmarks.swift
// MetalHLO Benchmarks
//
// Memory and resource benchmarks for measuring GPU memory usage.

import Foundation
import MetalHLO

/// Factory for creating memory and resource benchmarks.
public enum MemoryBenchmarks {

    /// Memory benchmark result.
    public struct MemoryBenchmarkResult: Sendable {
        public let id: String
        public let name: String
        public let category: String
        public let operation: String
        public let peakMemoryBytes: Int64
        public let allocatedBufferBytes: Int64
        public let intermediateBufferBytes: Int64
        public let timestamp: Date

        public init(
            id: String,
            name: String,
            category: String,
            operation: String,
            peakMemoryBytes: Int64,
            allocatedBufferBytes: Int64,
            intermediateBufferBytes: Int64,
            timestamp: Date = Date()
        ) {
            self.id = id
            self.name = name
            self.category = category
            self.operation = operation
            self.peakMemoryBytes = peakMemoryBytes
            self.allocatedBufferBytes = allocatedBufferBytes
            self.intermediateBufferBytes = intermediateBufferBytes
            self.timestamp = timestamp
        }

        /// Format the result as a human-readable string.
        public func formatted() -> String {
            let peakMB = Double(peakMemoryBytes) / (1024 * 1024)
            let allocMB = Double(allocatedBufferBytes) / (1024 * 1024)
            let intermediateMB = Double(intermediateBufferBytes) / (1024 * 1024)
            return "\(id): Peak=\(String(format: "%.2f", peakMB))MB, Alloc=\(String(format: "%.2f", allocMB))MB, Intermediate=\(String(format: "%.2f", intermediateMB))MB"
        }
    }

    /// Configuration for a memory benchmark.
    public struct MemoryBenchmarkConfig: Sendable {
        public let id: String
        public let name: String
        public let operation: String
        public let configuration: [String: String]
        public let mlirProgram: String
        public let inputShapes: [[Int]]
        public let outputShapes: [[Int]]

        public init(
            id: String,
            name: String,
            operation: String,
            configuration: [String: String],
            mlirProgram: String,
            inputShapes: [[Int]],
            outputShapes: [[Int]]
        ) {
            self.id = id
            self.name = name
            self.operation = operation
            self.configuration = configuration
            self.mlirProgram = mlirProgram
            self.inputShapes = inputShapes
            self.outputShapes = outputShapes
        }
    }

    // MARK: - Memory Benchmarks

    /// Get all memory benchmark configurations.
    public static func all() -> [MemoryBenchmarkConfig] {
        var configs: [MemoryBenchmarkConfig] = []

        // MEM-001: Peak allocation for large tensor operation
        configs.append(MemoryBenchmarkConfig(
            id: "MEM-001",
            name: "Peak Allocation (2048x2048 MatMul)",
            operation: "memory_peak",
            configuration: ["shape": "2048x2048", "dtype": "f32"],
            mlirProgram: """
            module @peak_allocation {
              func.func @main(%a: tensor<2048x2048xf32>, %b: tensor<2048x2048xf32>) -> (tensor<2048x2048xf32>) {
                %c = stablehlo.dot_general %a, %b, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
                return %c : tensor<2048x2048xf32>
              }
            }
            """,
            inputShapes: [[2048, 2048], [2048, 2048]],
            outputShapes: [[2048, 2048]]
        ))

        // MEM-002: Buffer reuse test (chain of operations)
        configs.append(MemoryBenchmarkConfig(
            id: "MEM-002",
            name: "Buffer Reuse (Elementwise Chain)",
            operation: "memory_reuse",
            configuration: ["shape": "4096x4096", "ops": "5"],
            mlirProgram: """
            module @buffer_reuse {
              func.func @main(%a: tensor<4096x4096xf32>) -> (tensor<4096x4096xf32>) {
                %b = stablehlo.exponential %a : tensor<4096x4096xf32>
                %c = stablehlo.tanh %b : tensor<4096x4096xf32>
                %d = stablehlo.log %c : tensor<4096x4096xf32>
                %e = stablehlo.sqrt %d : tensor<4096x4096xf32>
                %f = stablehlo.negate %e : tensor<4096x4096xf32>
                return %f : tensor<4096x4096xf32>
              }
            }
            """,
            inputShapes: [[4096, 4096]],
            outputShapes: [[4096, 4096]]
        ))

        // MEM-003: Intermediate tensor memory (fusion boundary test)
        configs.append(MemoryBenchmarkConfig(
            id: "MEM-003",
            name: "Intermediate Tensors (Reduction + Broadcast)",
            operation: "memory_intermediate",
            configuration: ["input_shape": "32x1024x1024", "notes": "Tests intermediate buffer allocation"],
            mlirProgram: """
            module @intermediate_tensors {
              func.func @main(%x: tensor<32x1024x1024xf32>) -> (tensor<32x1024x1024xf32>) {
                %zero = stablehlo.constant dense<0.0> : tensor<f32>

                // Reduce to intermediate shape
                %sum = stablehlo.reduce %x, %zero applies stablehlo.add across dimensions = [2] : (tensor<32x1024x1024xf32>, tensor<f32>) -> tensor<32x1024xf32>

                // Broadcast back up
                %sum_bc = stablehlo.broadcast_in_dim %sum, dims = [0, 1] : (tensor<32x1024xf32>) -> tensor<32x1024x1024xf32>

                // Normalize
                %normalized = stablehlo.divide %x, %sum_bc : tensor<32x1024x1024xf32>

                return %normalized : tensor<32x1024x1024xf32>
              }
            }
            """,
            inputShapes: [[32, 1024, 1024]],
            outputShapes: [[32, 1024, 1024]]
        ))

        return configs
    }

    /// Estimate memory usage for a given benchmark without running it.
    /// This provides a theoretical estimate based on tensor shapes.
    public static func estimateMemory(for config: MemoryBenchmarkConfig) -> (inputBytes: Int64, outputBytes: Int64, totalBytes: Int64) {
        // Calculate input memory
        var inputBytes: Int64 = 0
        for shape in config.inputShapes {
            let elements = shape.reduce(1, *)
            inputBytes += Int64(elements * 4)  // Assuming f32
        }

        // Calculate output memory
        var outputBytes: Int64 = 0
        for shape in config.outputShapes {
            let elements = shape.reduce(1, *)
            outputBytes += Int64(elements * 4)
        }

        return (inputBytes, outputBytes, inputBytes + outputBytes)
    }
}

// MARK: - FFT Benchmarks (Pending Complex Type Support)

/// FFT benchmarks are currently disabled pending complex type support in the public API.
/// When complex64 type is exposed, these benchmarks can be enabled.
public enum FFTBenchmarks {

    /// FFT benchmark configuration (for future use).
    public struct FFTBenchmarkConfig: Sendable {
        public let id: String
        public let name: String
        public let operation: String
        public let inputShape: [Int]
        public let fftType: FFTType

        public enum FFTType: String, Sendable {
            case fft = "fft"
            case ifft = "ifft"
            case rfft = "rfft"
            case irfft = "irfft"
            case fft2d = "fft2d"
        }
    }

    /// Get all FFT benchmark configurations (currently empty, pending complex type support).
    public static func all() -> [FFTBenchmarkConfig] {
        // FFT benchmarks require complex64 type support which is not yet exposed in the public API.
        // The following configurations are preserved for when support is added:
        //
        // FFT-001: FFT 1D (1024 points) - Complex input/output
        // FFT-002: FFT 1D (4096 points) - Larger 1D FFT
        // FFT-003: IFFT 1D (1024 points) - Inverse FFT
        // FFT-004: RFFT 1D (1024 points) - Real-to-complex FFT
        // FFT-005: IRFFT 1D (513 -> 1024) - Complex-to-real inverse
        // FFT-006: FFT 2D (64x64) - 2D FFT for image processing
        //
        // Example MLIR (not executable yet):
        // module @fft_1d {
        //   func.func @main(%input: tensor<1024xcomplex<f32>>) -> tensor<1024xcomplex<f32>> {
        //     %result = stablehlo.fft %input, type = FFT, length = [1024] : tensor<1024xcomplex<f32>>
        //     return %result : tensor<1024xcomplex<f32>>
        //   }
        // }

        return []  // Empty until complex type support is added
    }

    /// Check if FFT operations are currently supported.
    public static var isSupported: Bool {
        // Complex type is supported internally but not exposed in the public Buffer API
        return false
    }

    /// Status message explaining FFT benchmark availability.
    public static var statusMessage: String {
        """
        FFT benchmarks are currently disabled.

        Reason: The complex64 type (used for FFT input/output) is supported internally
        in MetalHLO but not yet exposed in the public Buffer creation API.

        Status: Pending implementation of public complex buffer support.

        Planned benchmarks when enabled:
        - FFT-001: 1D FFT (1024 points)
        - FFT-002: 1D FFT (4096 points)
        - FFT-003: 1D IFFT (1024 points)
        - FFT-004: 1D RFFT (real-to-complex)
        - FFT-005: 1D IRFFT (complex-to-real)
        - FFT-006: 2D FFT (64x64)
        """
    }
}

// MARK: - Memory Benchmark Runner

/// Runner for memory benchmarks.
public struct MemoryBenchmarkRunner: Sendable {

    public init() {}

    /// Run a memory benchmark and measure actual memory usage.
    /// Note: Actual GPU memory tracking requires Metal Performance HUD or Instruments.
    /// This provides estimates based on buffer allocations.
    public func run(_ config: MemoryBenchmarks.MemoryBenchmarkConfig) throws -> MemoryBenchmarks.MemoryBenchmarkResult {
        let client = try Client.create()

        // Compile the program
        let _ = try client.compile(config.mlirProgram)

        // Estimate memory usage
        let (inputBytes, outputBytes, _) = MemoryBenchmarks.estimateMemory(for: config)

        // For now, we estimate intermediate memory as output size (conservative estimate)
        // A more accurate measurement would require Metal memory tracking APIs
        let intermediateBytes = outputBytes

        return MemoryBenchmarks.MemoryBenchmarkResult(
            id: config.id,
            name: config.name,
            category: "memory",
            operation: config.operation,
            peakMemoryBytes: inputBytes + outputBytes + intermediateBytes,
            allocatedBufferBytes: inputBytes + outputBytes,
            intermediateBufferBytes: intermediateBytes
        )
    }

    /// Run all memory benchmarks.
    public func runAll() throws -> [MemoryBenchmarks.MemoryBenchmarkResult] {
        var results: [MemoryBenchmarks.MemoryBenchmarkResult] = []
        for config in MemoryBenchmarks.all() {
            let result = try run(config)
            results.append(result)
        }
        return results
    }
}

// MARK: - Graph Optimization Metrics (Section 5.3)

/// Metrics for analyzing graph optimization effectiveness.
public struct GraphOptimizationMetrics: Sendable {
    /// Estimated number of operations in the source program.
    public let sourceOpCount: Int
    /// Estimated number of intermediate tensors (fusion boundaries).
    public let intermediateTensorCount: Int
    /// Estimated total memory footprint in bytes.
    public let estimatedMemoryBytes: Int64
    /// Breakdown of operation types.
    public let operationBreakdown: [String: Int]

    public init(
        sourceOpCount: Int,
        intermediateTensorCount: Int,
        estimatedMemoryBytes: Int64,
        operationBreakdown: [String: Int]
    ) {
        self.sourceOpCount = sourceOpCount
        self.intermediateTensorCount = intermediateTensorCount
        self.estimatedMemoryBytes = estimatedMemoryBytes
        self.operationBreakdown = operationBreakdown
    }

    /// Format as human-readable string.
    public func formatted() -> String {
        let memMB = Double(estimatedMemoryBytes) / (1024 * 1024)
        var result = "Ops: \(sourceOpCount), Intermediates: \(intermediateTensorCount), Memory: \(String(format: "%.2f", memMB))MB"
        if !operationBreakdown.isEmpty {
            let breakdown = operationBreakdown.map { "\($0.key):\($0.value)" }.sorted().joined(separator: ", ")
            result += "\n  Breakdown: \(breakdown)"
        }
        return result
    }
}

/// Analyzer for extracting graph optimization metrics from MLIR programs.
public struct GraphOptimizationAnalyzer: Sendable {

    public init() {}

    /// Analyze an MLIR program and extract optimization metrics.
    public func analyze(mlirProgram: String, inputShapes: [[Int]], outputShapes: [[Int]]) -> GraphOptimizationMetrics {
        // Count operations by parsing MLIR text
        let opCounts = countOperations(in: mlirProgram)
        let totalOps = opCounts.values.reduce(0, +)

        // Estimate intermediate tensors (each SSA value that's not input/output)
        let intermediates = estimateIntermediateTensors(in: mlirProgram, inputCount: inputShapes.count)

        // Estimate memory
        var memoryBytes: Int64 = 0

        // Input memory
        for shape in inputShapes {
            let elements = shape.reduce(1, *)
            memoryBytes += Int64(elements * 4)  // Assuming f32
        }

        // Output memory
        for shape in outputShapes {
            let elements = shape.reduce(1, *)
            memoryBytes += Int64(elements * 4)
        }

        // Intermediate memory (estimate based on largest tensor shape in program)
        let maxIntermediateSize = estimateMaxIntermediateSize(in: mlirProgram)
        memoryBytes += Int64(maxIntermediateSize * 4 * intermediates)

        return GraphOptimizationMetrics(
            sourceOpCount: totalOps,
            intermediateTensorCount: intermediates,
            estimatedMemoryBytes: memoryBytes,
            operationBreakdown: opCounts
        )
    }

    /// Count StableHLO operations in the MLIR program.
    private func countOperations(in mlir: String) -> [String: Int] {
        var counts: [String: Int] = [:]

        // Common StableHLO operations to look for
        let operations = [
            "stablehlo.add", "stablehlo.subtract", "stablehlo.multiply", "stablehlo.divide",
            "stablehlo.dot_general", "stablehlo.dot",
            "stablehlo.reduce", "stablehlo.broadcast_in_dim",
            "stablehlo.transpose", "stablehlo.reshape",
            "stablehlo.exponential", "stablehlo.log", "stablehlo.tanh", "stablehlo.logistic",
            "stablehlo.sqrt", "stablehlo.rsqrt", "stablehlo.negate", "stablehlo.abs",
            "stablehlo.maximum", "stablehlo.minimum", "stablehlo.compare", "stablehlo.select",
            "stablehlo.convolution", "stablehlo.concatenate", "stablehlo.slice",
            "stablehlo.gather", "stablehlo.scatter", "stablehlo.pad",
            "stablehlo.constant", "stablehlo.iota",
            "stablehlo.power", "stablehlo.clamp"
        ]

        for op in operations {
            let pattern = op + " "
            var searchRange = mlir.startIndex..<mlir.endIndex
            var count = 0

            while let range = mlir.range(of: pattern, range: searchRange) {
                count += 1
                searchRange = range.upperBound..<mlir.endIndex
            }

            if count > 0 {
                let shortName = op.replacingOccurrences(of: "stablehlo.", with: "")
                counts[shortName] = count
            }
        }

        return counts
    }

    /// Estimate number of intermediate tensors (fusion boundary candidates).
    private func estimateIntermediateTensors(in mlir: String, inputCount: Int) -> Int {
        // Count SSA assignments (% followed by identifier and =)
        var count = 0
        var searchRange = mlir.startIndex..<mlir.endIndex

        while let range = mlir.range(of: "%", range: searchRange) {
            // Check if this is an assignment (has = after the identifier)
            let afterPercent = range.upperBound
            if afterPercent < mlir.endIndex {
                let remaining = mlir[afterPercent...]
                if let eqRange = remaining.range(of: " = ") {
                    // Check this is on the same line
                    let beforeEq = mlir[afterPercent..<eqRange.lowerBound]
                    if !beforeEq.contains("\n") {
                        count += 1
                    }
                }
            }
            searchRange = range.upperBound..<mlir.endIndex
        }

        // Subtract input arguments (they're not intermediates)
        return max(0, count - inputCount)
    }

    /// Estimate maximum intermediate tensor size from tensor type annotations.
    private func estimateMaxIntermediateSize(in mlir: String) -> Int {
        var maxSize = 0

        // Look for tensor type patterns like tensor<1024x1024xf32>
        let pattern = try? NSRegularExpression(pattern: "tensor<([0-9x]+)x[a-z0-9]+>", options: [])
        guard let regex = pattern else { return 1024 * 1024 }  // Default 1M elements

        let nsString = mlir as NSString
        let results = regex.matches(in: mlir, options: [], range: NSRange(location: 0, length: nsString.length))

        for match in results {
            if match.numberOfRanges >= 2 {
                let shapeRange = match.range(at: 1)
                let shapeStr = nsString.substring(with: shapeRange)
                let dims = shapeStr.split(separator: "x").compactMap { Int($0) }
                let size = dims.reduce(1, *)
                maxSize = max(maxSize, size)
            }
        }

        return maxSize > 0 ? maxSize : 1024 * 1024
    }
}

/// Graph optimization benchmark configuration.
public struct GraphOptimizationBenchmarkConfig: Sendable {
    public let id: String
    public let name: String
    public let description: String
    public let mlirProgram: String
    public let inputShapes: [[Int]]
    public let outputShapes: [[Int]]

    public init(
        id: String,
        name: String,
        description: String,
        mlirProgram: String,
        inputShapes: [[Int]],
        outputShapes: [[Int]]
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.mlirProgram = mlirProgram
        self.inputShapes = inputShapes
        self.outputShapes = outputShapes
    }
}

/// Factory for graph optimization benchmarks.
public enum GraphOptimizationBenchmarks {

    /// Get all graph optimization benchmark configurations.
    public static func all() -> [GraphOptimizationBenchmarkConfig] {
        var configs: [GraphOptimizationBenchmarkConfig] = []

        // GRAPH-001: Simple elementwise chain (should fuse completely)
        configs.append(GraphOptimizationBenchmarkConfig(
            id: "GRAPH-001",
            name: "Elementwise Chain Optimization",
            description: "Chain of elementwise ops that should fuse into single kernel",
            mlirProgram: """
            module @elementwise_chain {
              func.func @main(%a: tensor<2048x2048xf32>) -> (tensor<2048x2048xf32>) {
                %b = stablehlo.exponential %a : tensor<2048x2048xf32>
                %c = stablehlo.tanh %b : tensor<2048x2048xf32>
                %d = stablehlo.log %c : tensor<2048x2048xf32>
                %e = stablehlo.sqrt %d : tensor<2048x2048xf32>
                %f = stablehlo.negate %e : tensor<2048x2048xf32>
                return %f : tensor<2048x2048xf32>
              }
            }
            """,
            inputShapes: [[2048, 2048]],
            outputShapes: [[2048, 2048]]
        ))

        // GRAPH-002: Matmul + bias + activation (partial fusion expected)
        configs.append(GraphOptimizationBenchmarkConfig(
            id: "GRAPH-002",
            name: "MatMul-Bias-ReLU Optimization",
            description: "MatMul followed by bias and ReLU - tests activation fusion",
            mlirProgram: """
            module @matmul_bias_relu {
              func.func @main(%x: tensor<128x512xf32>, %w: tensor<512x512xf32>, %b: tensor<512xf32>) -> (tensor<128x512xf32>) {
                %mm = stablehlo.dot_general %x, %w, #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
                %bc = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
                %bias = stablehlo.add %mm, %bc : tensor<128x512xf32>
                %zero = stablehlo.constant dense<0.0> : tensor<128x512xf32>
                %relu = stablehlo.maximum %bias, %zero : tensor<128x512xf32>
                return %relu : tensor<128x512xf32>
              }
            }
            """,
            inputShapes: [[128, 512], [512, 512], [512]],
            outputShapes: [[128, 512]]
        ))

        // GRAPH-003: Softmax pattern (reduction + broadcast + elementwise)
        configs.append(GraphOptimizationBenchmarkConfig(
            id: "GRAPH-003",
            name: "Softmax Optimization",
            description: "Softmax pattern with max, subtract, exp, sum, divide",
            mlirProgram: """
            module @softmax {
              func.func @main(%x: tensor<32x128x512xf32>) -> (tensor<32x128x512xf32>) {
                %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
                %zero = stablehlo.constant dense<0.0> : tensor<f32>

                %max = stablehlo.reduce %x, %neg_inf applies stablehlo.maximum across dimensions = [2] : (tensor<32x128x512xf32>, tensor<f32>) -> tensor<32x128xf32>
                %max_bc = stablehlo.broadcast_in_dim %max, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x512xf32>
                %shifted = stablehlo.subtract %x, %max_bc : tensor<32x128x512xf32>
                %exp = stablehlo.exponential %shifted : tensor<32x128x512xf32>
                %sum = stablehlo.reduce %exp, %zero applies stablehlo.add across dimensions = [2] : (tensor<32x128x512xf32>, tensor<f32>) -> tensor<32x128xf32>
                %sum_bc = stablehlo.broadcast_in_dim %sum, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x512xf32>
                %result = stablehlo.divide %exp, %sum_bc : tensor<32x128x512xf32>

                return %result : tensor<32x128x512xf32>
              }
            }
            """,
            inputShapes: [[32, 128, 512]],
            outputShapes: [[32, 128, 512]]
        ))

        // GRAPH-004: LayerNorm pattern
        configs.append(GraphOptimizationBenchmarkConfig(
            id: "GRAPH-004",
            name: "LayerNorm Optimization",
            description: "LayerNorm with mean, variance, normalize, scale, shift",
            mlirProgram: """
            module @layernorm {
              func.func @main(%x: tensor<32x128x768xf32>, %gamma: tensor<768xf32>, %beta: tensor<768xf32>) -> (tensor<32x128x768xf32>) {
                %zero = stablehlo.constant dense<0.0> : tensor<f32>
                %eps = stablehlo.constant dense<1.0e-5> : tensor<f32>
                %n = stablehlo.constant dense<768.0> : tensor<f32>

                // Mean
                %sum = stablehlo.reduce %x, %zero applies stablehlo.add across dimensions = [2] : (tensor<32x128x768xf32>, tensor<f32>) -> tensor<32x128xf32>
                %n_bc1 = stablehlo.broadcast_in_dim %n, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
                %mean = stablehlo.divide %sum, %n_bc1 : tensor<32x128xf32>
                %mean_bc = stablehlo.broadcast_in_dim %mean, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x768xf32>

                // Variance
                %centered = stablehlo.subtract %x, %mean_bc : tensor<32x128x768xf32>
                %sq = stablehlo.multiply %centered, %centered : tensor<32x128x768xf32>
                %var_sum = stablehlo.reduce %sq, %zero applies stablehlo.add across dimensions = [2] : (tensor<32x128x768xf32>, tensor<f32>) -> tensor<32x128xf32>
                %var = stablehlo.divide %var_sum, %n_bc1 : tensor<32x128xf32>
                %eps_bc = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
                %var_eps = stablehlo.add %var, %eps_bc : tensor<32x128xf32>
                %std = stablehlo.sqrt %var_eps : tensor<32x128xf32>
                %std_bc = stablehlo.broadcast_in_dim %std, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x768xf32>

                // Normalize
                %norm = stablehlo.divide %centered, %std_bc : tensor<32x128x768xf32>

                // Scale and shift
                %gamma_bc = stablehlo.broadcast_in_dim %gamma, dims = [2] : (tensor<768xf32>) -> tensor<32x128x768xf32>
                %beta_bc = stablehlo.broadcast_in_dim %beta, dims = [2] : (tensor<768xf32>) -> tensor<32x128x768xf32>
                %scaled = stablehlo.multiply %norm, %gamma_bc : tensor<32x128x768xf32>
                %result = stablehlo.add %scaled, %beta_bc : tensor<32x128x768xf32>

                return %result : tensor<32x128x768xf32>
              }
            }
            """,
            inputShapes: [[32, 128, 768], [768], [768]],
            outputShapes: [[32, 128, 768]]
        ))

        return configs
    }

    /// Analyze all graph optimization benchmarks.
    public static func analyzeAll() -> [(config: GraphOptimizationBenchmarkConfig, metrics: GraphOptimizationMetrics)] {
        let analyzer = GraphOptimizationAnalyzer()
        return all().map { config in
            let metrics = analyzer.analyze(
                mlirProgram: config.mlirProgram,
                inputShapes: config.inputShapes,
                outputShapes: config.outputShapes
            )
            return (config, metrics)
        }
    }
}
