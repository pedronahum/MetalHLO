// MLXBenchmarks.swift
// MetalHLO
//
// MLX implementations of benchmarks for comparison with MetalHLO.

import Foundation
import MLX
import MLXRandom
import MetalHLOBenchmarks

// MARK: - MLX Benchmark Protocol

/// Protocol for MLX benchmarks that can be compared with MetalHLO.
public protocol MLXBenchmark: Sendable {
    /// Unique identifier matching the MetalHLO benchmark ID.
    var id: String { get }

    /// Human-readable name.
    var name: String { get }

    /// Category for grouping.
    var category: String { get }

    /// The operation being benchmarked.
    var operation: String { get }

    /// Run the benchmark and return timing results.
    func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult
}

/// Result of an MLX benchmark run.
public struct MLXBenchmarkResult: Sendable {
    public let id: String
    public let name: String
    public let category: String
    public let operation: String
    public let meanTimeSeconds: Double
    public let stdDevSeconds: Double
    public let minTimeSeconds: Double
    public let maxTimeSeconds: Double
    public let iterations: Int

    public init(
        id: String,
        name: String,
        category: String,
        operation: String,
        meanTimeSeconds: Double,
        stdDevSeconds: Double,
        minTimeSeconds: Double,
        maxTimeSeconds: Double,
        iterations: Int
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.operation = operation
        self.meanTimeSeconds = meanTimeSeconds
        self.stdDevSeconds = stdDevSeconds
        self.minTimeSeconds = minTimeSeconds
        self.maxTimeSeconds = maxTimeSeconds
        self.iterations = iterations
    }
}

// MARK: - Matmul Benchmark

/// Matrix multiplication benchmark using MLX.
public struct MatmulBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "matrix"
    public let operation: String = "dot"
    public let m: Int
    public let n: Int
    public let k: Int

    public init(id: String, name: String, m: Int, n: Int, k: Int) {
        self.id = id
        self.name = name
        self.m = m
        self.n = n
        self.k = k
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        // Setup
        let a = MLXRandom.normal([m, k])
        let b = MLXRandom.normal([k, n])
        eval(a, b)

        // Warmup
        for _ in 0..<warmup {
            let c = matmul(a, b)
            eval(c)
        }

        // Measure
        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = matmul(a, b)
            eval(c)
            // Synchronize to ensure GPU work is complete
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

/// Batched matrix multiplication benchmark.
public struct BatchedMatmulBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "matrix"
    public let operation: String = "batched_dot"
    public let batch: Int
    public let m: Int
    public let n: Int
    public let k: Int

    public init(id: String, name: String, batch: Int, m: Int, n: Int, k: Int) {
        self.id = id
        self.name = name
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let a = MLXRandom.normal([batch, m, k])
        let b = MLXRandom.normal([batch, k, n])
        eval(a, b)

        for _ in 0..<warmup {
            let c = matmul(a, b)
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = matmul(a, b)
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - Transpose Benchmark

/// Transpose operation benchmark.
public struct TransposeBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "matrix"
    public let operation: String = "transpose"
    public let shape: [Int]
    public let axes: [Int]?  // nil for simple 2D transpose

    public init(id: String, name: String, shape: [Int], axes: [Int]? = nil) {
        self.id = id
        self.name = name
        self.shape = shape
        self.axes = axes
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let a = MLXRandom.normal(shape)
        eval(a)

        func doTranspose() -> MLXArray {
            if let axes = axes {
                // Use swappedAxes for 3D transpose (swap axes 1 and 2)
                if axes == [0, 2, 1] {
                    return a.swappedAxes(1, 2)
                }
                // For other cases, use general transpose
                return a.T
            } else {
                return a.T
            }
        }

        for _ in 0..<warmup {
            let c = doTranspose()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doTranspose()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - Arithmetic Benchmarks

/// Broadcast binary operation benchmark.
public struct BroadcastBinaryOpBenchmark: MLXBenchmark {
    public enum BroadcastType: String, Sendable {
        case row      // [M, N] + [1, N]
        case scalar   // [M, N] + scalar
    }

    public let id: String
    public let name: String
    public let category: String = "arithmetic"
    public let operation: String = "broadcast_add"
    public let shape: [Int]
    public let broadcastType: BroadcastType

    public init(id: String, name: String, shape: [Int], broadcastType: BroadcastType) {
        self.id = id
        self.name = name
        self.shape = shape
        self.broadcastType = broadcastType
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let a = MLXRandom.normal(shape)
        let b: MLXArray
        switch broadcastType {
        case .row:
            b = MLXRandom.normal([1, shape[1]])
        case .scalar:
            b = MLXArray(2.0)
        }
        eval(a, b)

        for _ in 0..<warmup {
            let c = a + b
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = a + b
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

/// Binary elementwise operation benchmark.
public struct BinaryOpBenchmark: MLXBenchmark {
    public enum OpType: String, Sendable {
        case add, multiply, divide, maximum
    }

    public let id: String
    public let name: String
    public let category: String = "arithmetic"
    public let operation: String = "binary"
    public let shape: [Int]
    public let opType: OpType

    public init(id: String, name: String, shape: [Int], opType: OpType) {
        self.id = id
        self.name = name
        self.shape = shape
        self.opType = opType
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let a = MLXRandom.normal(shape)
        let b = MLXRandom.normal(shape)
        eval(a, b)

        func doOp() -> MLXArray {
            switch opType {
            case .add: return a + b
            case .multiply: return a * b
            case .divide: return a / b
            case .maximum: return maximum(a, b)
            }
        }

        for _ in 0..<warmup {
            let c = doOp()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doOp()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

/// Unary elementwise operation benchmark.
public struct UnaryOpBenchmark: MLXBenchmark {
    public enum OpType: String, Sendable {
        case exp, log, tanh, sqrt, rsqrt, sigmoid
    }

    public let id: String
    public let name: String
    public let category: String = "arithmetic"
    public let operation: String = "unary"
    public let shape: [Int]
    public let opType: OpType

    public init(id: String, name: String, shape: [Int], opType: OpType) {
        self.id = id
        self.name = name
        self.shape = shape
        self.opType = opType
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let a = MLXRandom.uniform(low: 0.1, high: 10.0, shape)
        eval(a)

        func doOp() -> MLXArray {
            switch opType {
            case .exp: return MLX.exp(a)
            case .log: return MLX.log(a)
            case .tanh: return MLX.tanh(a)
            case .sqrt: return MLX.sqrt(a)
            case .rsqrt: return MLX.rsqrt(a)
            case .sigmoid: return MLX.sigmoid(a)
            }
        }

        for _ in 0..<warmup {
            let c = doOp()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doOp()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - Reduction Benchmark

/// Reduction operation benchmark.
public struct ReductionBenchmark: MLXBenchmark {
    public enum ReductionType: String, Sendable {
        case sumAll, sumAxis, maxAxis
    }

    public let id: String
    public let name: String
    public let category: String = "reduction"
    public let operation: String = "reduce"
    public let shape: [Int]
    public let reductionType: ReductionType
    public let axis: Int?

    public init(id: String, name: String, shape: [Int], reductionType: ReductionType, axis: Int? = nil) {
        self.id = id
        self.name = name
        self.shape = shape
        self.reductionType = reductionType
        self.axis = axis
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let a = MLXRandom.normal(shape)
        eval(a)

        func doOp() -> MLXArray {
            switch reductionType {
            case .sumAll:
                return MLX.sum(a, keepDims: false)
            case .sumAxis:
                return MLX.sum(a, axis: axis!, keepDims: false)
            case .maxAxis:
                return MLX.max(a, axis: axis!, keepDims: false)
            }
        }

        for _ in 0..<warmup {
            let c = doOp()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doOp()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - Convolution Benchmark (disabled)

// Conv2D benchmarks are temporarily disabled due to IntOrPair type compatibility issues
// with the MLX-Swift conv2d API. The MetalHLO convolution benchmarks can still be compared
// manually by running them separately.
// TODO: Fix conv2d parameter types for MLX-Swift when API is better understood

// MARK: - Normalization Benchmark

/// Layer normalization benchmark.
public struct LayerNormBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "normalization"
    public let operation: String = "layer_norm"
    public let shape: [Int]
    public let normalizeAxis: Int

    public init(id: String, name: String, shape: [Int], normalizeAxis: Int) {
        self.id = id
        self.name = name
        self.shape = shape
        self.normalizeAxis = normalizeAxis
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let input = MLXRandom.normal(shape)
        let normSize = shape[normalizeAxis]
        let gamma = MLX.ones([normSize])
        let beta = MLX.zeros([normSize])
        eval(input, gamma, beta)

        func doLayerNorm() -> MLXArray {
            let meanVal = MLX.mean(input, axis: normalizeAxis, keepDims: true)
            let varianceVal = MLX.variance(input, axis: normalizeAxis, keepDims: true)
            let normalized = (input - meanVal) / MLX.sqrt(varianceVal + 1e-5)
            return normalized * gamma + beta
        }

        for _ in 0..<warmup {
            let c = doLayerNorm()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doLayerNorm()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - Softmax Benchmark

/// Softmax benchmark.
public struct SoftmaxBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "model_transformer"
    public let operation: String = "softmax"
    public let shape: [Int]
    public let axis: Int

    public init(id: String, name: String, shape: [Int], axis: Int) {
        self.id = id
        self.name = name
        self.shape = shape
        self.axis = axis
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let input = MLXRandom.normal(shape)
        eval(input)

        for _ in 0..<warmup {
            let c = MLX.softmax(input, axis: axis)
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = MLX.softmax(input, axis: axis)
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - MLP Benchmark

/// MLP inference benchmark.
public struct MLPBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "model_mlp"
    public let operation: String = "mlp_inference"
    public let batchSize: Int
    public let layers: [Int]

    public init(id: String, name: String, batchSize: Int, layers: [Int]) {
        self.id = id
        self.name = name
        self.batchSize = batchSize
        self.layers = layers
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let input = MLXRandom.normal([batchSize, layers[0]])
        var weights: [MLXArray] = []
        var biases: [MLXArray] = []

        for i in 0..<(layers.count - 1) {
            weights.append(MLXRandom.normal([layers[i], layers[i + 1]]) * 0.01)
            biases.append(MLX.zeros([layers[i + 1]]))
        }

        eval(input)
        for w in weights { eval(w) }
        for b in biases { eval(b) }

        func doMLP() -> MLXArray {
            var x = input
            for i in 0..<weights.count {
                x = matmul(x, weights[i]) + biases[i]
                if i < weights.count - 1 {
                    x = maximum(x, 0)  // ReLU
                }
            }
            return x
        }

        for _ in 0..<warmup {
            let c = doMLP()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doMLP()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - Transformer FFN Benchmark

/// Transformer FFN with GELU benchmark.
public struct TransformerFFNBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "model_mlp"
    public let operation: String = "ffn_gelu"
    public let batchSize: Int
    public let seqLen: Int
    public let hiddenSize: Int
    public let ffnSize: Int

    public init(id: String, name: String, batchSize: Int, seqLen: Int, hiddenSize: Int, ffnSize: Int) {
        self.id = id
        self.name = name
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.hiddenSize = hiddenSize
        self.ffnSize = ffnSize
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let input = MLXRandom.normal([batchSize, seqLen, hiddenSize])
        let w1 = MLXRandom.normal([hiddenSize, ffnSize]) * 0.02
        let b1 = MLX.zeros([ffnSize])
        let w2 = MLXRandom.normal([ffnSize, hiddenSize]) * 0.02
        let b2 = MLX.zeros([hiddenSize])
        eval(input, w1, b1, w2, b2)

        func doFFN() -> MLXArray {
            let flat = input.reshaped([batchSize * seqLen, hiddenSize])
            var h = matmul(flat, w1) + b1
            h = h * MLX.sigmoid(1.702 * h)  // GELU approximation
            h = matmul(h, w2) + b2
            return h.reshaped([batchSize, seqLen, hiddenSize])
        }

        for _ in 0..<warmup {
            let c = doFFN()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doFFN()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - Attention Benchmark

/// Self-attention benchmark.
public struct AttentionBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "model_transformer"
    public let operation: String = "attention"
    public let batchSize: Int
    public let numHeads: Int
    public let seqLen: Int
    public let headDim: Int

    public init(id: String, name: String, batchSize: Int, numHeads: Int, seqLen: Int, headDim: Int) {
        self.id = id
        self.name = name
        self.batchSize = batchSize
        self.numHeads = numHeads
        self.seqLen = seqLen
        self.headDim = headDim
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        // Q, K, V in [batch, heads, seq, head_dim] format
        let q = MLXRandom.normal([batchSize, numHeads, seqLen, headDim])
        let k = MLXRandom.normal([batchSize, numHeads, seqLen, headDim])
        let v = MLXRandom.normal([batchSize, numHeads, seqLen, headDim])
        let scale = MLXArray(1.0 / Float(headDim).squareRoot())
        eval(q, k, v, scale)

        func doAttention() -> MLXArray {
            // Attention: softmax(QK^T / sqrt(d)) * V
            let kT = k.transposed(0, 1, 3, 2)  // [batch, heads, head_dim, seq]
            let scores = matmul(q, kT) * scale  // [batch, heads, seq, seq]
            let attnWeights = MLX.softmax(scores, axis: -1)
            return matmul(attnWeights, v)  // [batch, heads, seq, head_dim]
        }

        for _ in 0..<warmup {
            let c = doAttention()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doAttention()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - Indexing Benchmarks

/// Slice operation benchmark.
public struct SliceBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "indexing"
    public let operation: String = "slice"
    public let inputShape: [Int]
    public let sliceStart: [Int]
    public let sliceEnd: [Int]

    public init(id: String, name: String, inputShape: [Int], sliceStart: [Int], sliceEnd: [Int]) {
        self.id = id
        self.name = name
        self.inputShape = inputShape
        self.sliceStart = sliceStart
        self.sliceEnd = sliceEnd
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let a = MLXRandom.normal(inputShape)
        eval(a)

        func doSlice() -> MLXArray {
            // Use subscript for slicing
            return a[sliceStart[0]..<sliceEnd[0], sliceStart[1]..<sliceEnd[1]]
        }

        for _ in 0..<warmup {
            let c = doSlice()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doSlice()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

/// Gather (embedding lookup) benchmark.
public struct GatherBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "indexing"
    public let operation: String = "gather"
    public let vocabSize: Int
    public let embeddingDim: Int
    public let numIndices: Int

    public init(id: String, name: String, vocabSize: Int, embeddingDim: Int, numIndices: Int) {
        self.id = id
        self.name = name
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.numIndices = numIndices
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let table = MLXRandom.normal([vocabSize, embeddingDim])
        let indices = MLXRandom.randInt(low: 0, high: vocabSize, [numIndices])
        eval(table, indices)

        func doGather() -> MLXArray {
            return table[indices]
        }

        for _ in 0..<warmup {
            let c = doGather()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doGather()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

/// Concatenate operation benchmark.
public struct ConcatenateBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "indexing"
    public let operation: String = "concatenate"
    public let shape: [Int]
    public let numTensors: Int
    public let axis: Int

    public init(id: String, name: String, shape: [Int], numTensors: Int, axis: Int) {
        self.id = id
        self.name = name
        self.shape = shape
        self.numTensors = numTensors
        self.axis = axis
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        var tensors: [MLXArray] = []
        for _ in 0..<numTensors {
            tensors.append(MLXRandom.normal(shape))
        }
        for t in tensors { eval(t) }

        func doConcat() -> MLXArray {
            return MLX.concatenated(tensors, axis: axis)
        }

        for _ in 0..<warmup {
            let c = doConcat()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doConcat()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

/// Pad operation benchmark.
public struct PadBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "indexing"
    public let operation: String = "pad"
    public let inputShape: [Int]
    public let padSize: Int  // Symmetric padding on all sides

    public init(id: String, name: String, inputShape: [Int], padSize: Int) {
        self.id = id
        self.name = name
        self.inputShape = inputShape
        self.padSize = padSize
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        let a = MLXRandom.normal(inputShape)
        eval(a)

        func doPad() -> MLXArray {
            // Use IntOrPair for padding - same padding on all dimensions
            return MLX.padded(a, width: IntOrPair(padSize))
        }

        for _ in 0..<warmup {
            let c = doPad()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doPad()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - Conv2D Benchmark

/// Conv2D benchmark.
public struct Conv2DBenchmark: MLXBenchmark {
    public let id: String
    public let name: String
    public let category: String = "convolution"
    public let operation: String = "conv2d"
    public let batchSize: Int
    public let height: Int
    public let width: Int
    public let inChannels: Int
    public let outChannels: Int
    public let kernelSize: Int
    public let stride: Int
    public let padding: Int

    public init(id: String, name: String, batchSize: Int, height: Int, width: Int,
                inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self.id = id
        self.name = name
        self.batchSize = batchSize
        self.height = height
        self.width = width
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
    }

    public func run(iterations: Int, warmup: Int) -> MLXBenchmarkResult {
        // MLX uses NHWC format
        let input = MLXRandom.normal([batchSize, height, width, inChannels])
        // Weight shape for conv2d: [outChannels, kernelH, kernelW, inChannels]
        let weight = MLXRandom.normal([outChannels, kernelSize, kernelSize, inChannels]) * 0.01
        eval(input, weight)

        func doConv() -> MLXArray {
            return MLX.conv2d(input, weight, stride: IntOrPair(stride), padding: IntOrPair(padding))
        }

        for _ in 0..<warmup {
            let c = doConv()
            eval(c)
        }

        var times: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let c = doConv()
            eval(c)
            Stream.gpu.synchronize()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }

        return computeResult(id: id, name: name, category: category, operation: operation, times: times, iterations: iterations)
    }
}

// MARK: - Helper Function

private func computeResult(id: String, name: String, category: String, operation: String, times: [Double], iterations: Int) -> MLXBenchmarkResult {
    let mean = times.reduce(0, +) / Double(times.count)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(times.count)
    let stdDev = sqrt(variance)
    let minTime = times.min() ?? 0
    let maxTime = times.max() ?? 0

    return MLXBenchmarkResult(
        id: id,
        name: name,
        category: category,
        operation: operation,
        meanTimeSeconds: mean,
        stdDevSeconds: stdDev,
        minTimeSeconds: minTime,
        maxTimeSeconds: maxTime,
        iterations: iterations
    )
}

// MARK: - MLX Benchmark Collection

/// Collection of MLX benchmarks for comparison.
public enum MLXBenchmarks {

    /// Get all MLX benchmarks.
    public static func all() -> [MLXBenchmark] {
        var benchmarks: [MLXBenchmark] = []

        // Matrix benchmarks
        benchmarks.append(MatmulBenchmark(id: "MAT-DOT-001", name: "MLX GEMM 128x128", m: 128, n: 128, k: 128))
        benchmarks.append(MatmulBenchmark(id: "MAT-DOT-002", name: "MLX GEMM 512x512", m: 512, n: 512, k: 512))
        benchmarks.append(MatmulBenchmark(id: "MAT-DOT-003", name: "MLX GEMM 1024x1024", m: 1024, n: 1024, k: 1024))
        benchmarks.append(MatmulBenchmark(id: "MAT-DOT-004", name: "MLX GEMM 2048x2048", m: 2048, n: 2048, k: 2048))
        benchmarks.append(MatmulBenchmark(id: "MAT-DOT-005", name: "MLX GEMM 4096x4096", m: 4096, n: 4096, k: 4096))
        benchmarks.append(MatmulBenchmark(id: "MAT-DOT-006", name: "MLX GEMM 32x768 @ 768x4096", m: 32, n: 4096, k: 768))
        benchmarks.append(MatmulBenchmark(id: "MAT-DOT-007", name: "MLX GEMM 128x3072 @ 3072x768", m: 128, n: 768, k: 3072))
        benchmarks.append(MatmulBenchmark(id: "MAT-DOT-008", name: "MLX GEMM 1x4096 @ 4096x4096", m: 1, n: 4096, k: 4096))

        // Batched matrix benchmarks
        benchmarks.append(BatchedMatmulBenchmark(id: "MAT-BATCH-001", name: "MLX Batched GEMM [8] 512x512", batch: 8, m: 512, n: 512, k: 512))
        benchmarks.append(BatchedMatmulBenchmark(id: "MAT-BATCH-002", name: "MLX Batched GEMM [32] 256x256", batch: 32, m: 256, n: 256, k: 256))
        benchmarks.append(BatchedMatmulBenchmark(id: "MAT-BATCH-003", name: "MLX Batched GEMM [64] 128x64 @ 64x128", batch: 64, m: 128, n: 128, k: 64))
        benchmarks.append(BatchedMatmulBenchmark(id: "MAT-BATCH-004", name: "MLX Batched GEMM [12] 64x64 @ 64x512", batch: 12, m: 64, n: 512, k: 64))

        // Transpose benchmarks
        benchmarks.append(TransposeBenchmark(id: "MAT-TR-001", name: "MLX Transpose 1024x1024", shape: [1024, 1024]))
        benchmarks.append(TransposeBenchmark(id: "MAT-TR-002", name: "MLX Transpose 3D 32x128x64", shape: [32, 128, 64], axes: [0, 2, 1]))

        // Arithmetic benchmarks - binary
        benchmarks.append(BinaryOpBenchmark(id: "ARITH-B-001", name: "MLX Add 1024x1024", shape: [1024, 1024], opType: .add))
        benchmarks.append(BinaryOpBenchmark(id: "ARITH-B-002", name: "MLX Add 4096x4096", shape: [4096, 4096], opType: .add))
        benchmarks.append(BinaryOpBenchmark(id: "ARITH-B-003", name: "MLX Multiply 1024x1024", shape: [1024, 1024], opType: .multiply))
        benchmarks.append(BinaryOpBenchmark(id: "ARITH-B-004", name: "MLX Multiply 4096x4096", shape: [4096, 4096], opType: .multiply))
        benchmarks.append(BinaryOpBenchmark(id: "ARITH-B-005", name: "MLX Divide 1024x1024", shape: [1024, 1024], opType: .divide))
        benchmarks.append(BinaryOpBenchmark(id: "ARITH-B-006", name: "MLX Maximum 4096x4096", shape: [4096, 4096], opType: .maximum))

        // Arithmetic benchmarks - broadcast
        benchmarks.append(BroadcastBinaryOpBenchmark(id: "ARITH-BC-001", name: "MLX Add Row Broadcast 4096x4096", shape: [4096, 4096], broadcastType: .row))
        benchmarks.append(BroadcastBinaryOpBenchmark(id: "ARITH-BC-002", name: "MLX Add Scalar Broadcast 4096x4096", shape: [4096, 4096], broadcastType: .scalar))

        // Arithmetic benchmarks - unary
        benchmarks.append(UnaryOpBenchmark(id: "ARITH-U-001", name: "MLX Exp 1024x1024", shape: [1024, 1024], opType: .exp))
        benchmarks.append(UnaryOpBenchmark(id: "ARITH-U-002", name: "MLX Log 4096x4096", shape: [4096, 4096], opType: .log))
        benchmarks.append(UnaryOpBenchmark(id: "ARITH-U-003", name: "MLX Tanh 1024x1024", shape: [1024, 1024], opType: .tanh))
        benchmarks.append(UnaryOpBenchmark(id: "ARITH-U-004", name: "MLX Sqrt 4096x4096", shape: [4096, 4096], opType: .sqrt))
        benchmarks.append(UnaryOpBenchmark(id: "ARITH-U-005", name: "MLX Rsqrt 4096x4096", shape: [4096, 4096], opType: .rsqrt))
        benchmarks.append(UnaryOpBenchmark(id: "ARITH-U-006", name: "MLX Sigmoid 1024x1024", shape: [1024, 1024], opType: .sigmoid))

        // Reduction benchmarks
        benchmarks.append(ReductionBenchmark(id: "RED-001", name: "MLX Global Sum 1024x1024", shape: [1024, 1024], reductionType: .sumAll))
        benchmarks.append(ReductionBenchmark(id: "RED-002", name: "MLX Row-wise Sum 1024x1024", shape: [1024, 1024], reductionType: .sumAxis, axis: 1))
        benchmarks.append(ReductionBenchmark(id: "RED-003", name: "MLX Column-wise Sum 1024x1024", shape: [1024, 1024], reductionType: .sumAxis, axis: 0))
        benchmarks.append(ReductionBenchmark(id: "RED-004", name: "MLX Row-wise Max 4096x4096", shape: [4096, 4096], reductionType: .maxAxis, axis: 1))
        benchmarks.append(ReductionBenchmark(id: "RED-005", name: "MLX LayerNorm Reduction 32x128x768", shape: [32, 128, 768], reductionType: .sumAxis, axis: 2))
        benchmarks.append(ReductionBenchmark(id: "RED-006", name: "MLX Attention Reduction 32x12x512x512", shape: [32, 12, 512, 512], reductionType: .sumAxis, axis: 3))

        // Convolution benchmarks
        benchmarks.append(Conv2DBenchmark(id: "CONV-001", name: "MLX Conv2D 1x224x224x3 k7s2", batchSize: 1, height: 224, width: 224, inChannels: 3, outChannels: 64, kernelSize: 7, stride: 2, padding: 3))
        benchmarks.append(Conv2DBenchmark(id: "CONV-002", name: "MLX Conv2D 1x56x56x64 k3s1", batchSize: 1, height: 56, width: 56, inChannels: 64, outChannels: 64, kernelSize: 3, stride: 1, padding: 1))
        benchmarks.append(Conv2DBenchmark(id: "CONV-003", name: "MLX Conv2D 1x28x28x128 k3s1", batchSize: 1, height: 28, width: 28, inChannels: 128, outChannels: 128, kernelSize: 3, stride: 1, padding: 1))
        benchmarks.append(Conv2DBenchmark(id: "CONV-004", name: "MLX Conv2D 1x14x14x256 k3s1", batchSize: 1, height: 14, width: 14, inChannels: 256, outChannels: 256, kernelSize: 3, stride: 1, padding: 1))
        benchmarks.append(Conv2DBenchmark(id: "CONV-005", name: "MLX Conv2D 32x56x56x64 k3s1", batchSize: 32, height: 56, width: 56, inChannels: 64, outChannels: 64, kernelSize: 3, stride: 1, padding: 1))
        benchmarks.append(Conv2DBenchmark(id: "CONV-006", name: "MLX Conv2D 1x112x112x32 k1s1", batchSize: 1, height: 112, width: 112, inChannels: 32, outChannels: 32, kernelSize: 1, stride: 1, padding: 0))
        benchmarks.append(Conv2DBenchmark(id: "CONV-007", name: "MLX Conv2D 1x56x56x128 k3s1", batchSize: 1, height: 56, width: 56, inChannels: 128, outChannels: 128, kernelSize: 3, stride: 1, padding: 1))

        // Normalization benchmarks
        benchmarks.append(LayerNormBenchmark(id: "NORM-LN-001", name: "MLX LayerNorm 1x128x768", shape: [1, 128, 768], normalizeAxis: 2))
        benchmarks.append(LayerNormBenchmark(id: "NORM-LN-002", name: "MLX LayerNorm 32x128x768", shape: [32, 128, 768], normalizeAxis: 2))
        benchmarks.append(LayerNormBenchmark(id: "NORM-LN-003", name: "MLX LayerNorm 1x512x1024", shape: [1, 512, 1024], normalizeAxis: 2))
        benchmarks.append(LayerNormBenchmark(id: "NORM-LN-004", name: "MLX LayerNorm 8x2048x768", shape: [8, 2048, 768], normalizeAxis: 2))

        // Softmax benchmark
        benchmarks.append(SoftmaxBenchmark(id: "XFMR-INF-005", name: "MLX Softmax 8x12x128x128", shape: [8, 12, 128, 128], axis: -1))

        // Attention benchmarks (standalone attention without QKV projections)
        // Note: XFMR-INF-001 to 004 in MetalHLO include QKV projections, so we use different IDs
        benchmarks.append(AttentionBenchmark(id: "ATTN-001", name: "MLX Attention 1x12x128x64", batchSize: 1, numHeads: 12, seqLen: 128, headDim: 64))
        benchmarks.append(AttentionBenchmark(id: "ATTN-002", name: "MLX Attention 8x12x128x64", batchSize: 8, numHeads: 12, seqLen: 128, headDim: 64))
        benchmarks.append(AttentionBenchmark(id: "ATTN-003", name: "MLX Attention 1x12x512x64", batchSize: 1, numHeads: 12, seqLen: 512, headDim: 64))
        benchmarks.append(AttentionBenchmark(id: "ATTN-004", name: "MLX Attention 32x8x128x64", batchSize: 32, numHeads: 8, seqLen: 128, headDim: 64))

        // Indexing benchmarks
        benchmarks.append(SliceBenchmark(id: "IDX-001", name: "MLX Slice 1024x1024 -> 512x512", inputShape: [1024, 1024], sliceStart: [256, 256], sliceEnd: [768, 768]))
        benchmarks.append(GatherBenchmark(id: "IDX-004", name: "MLX Gather Embedding 10000x256", vocabSize: 10000, embeddingDim: 256, numIndices: 1024))
        benchmarks.append(ConcatenateBenchmark(id: "IDX-006", name: "MLX Concatenate 4x 512x512", shape: [512, 512], numTensors: 4, axis: 0))
        benchmarks.append(PadBenchmark(id: "IDX-007", name: "MLX Pad 512x512 -> 640x640", inputShape: [512, 512], padSize: 64))

        // MLP benchmarks
        benchmarks.append(MLPBenchmark(id: "MLP-INF-001", name: "MLX MLP 784->256->10 (BS=1)", batchSize: 1, layers: [784, 256, 10]))
        benchmarks.append(MLPBenchmark(id: "MLP-INF-002", name: "MLX MLP 784->256->10 (BS=32)", batchSize: 32, layers: [784, 256, 10]))
        benchmarks.append(MLPBenchmark(id: "MLP-INF-003", name: "MLX MLP 784->256->10 (BS=128)", batchSize: 128, layers: [784, 256, 10]))
        benchmarks.append(MLPBenchmark(id: "MLP-INF-004", name: "MLX MLP 784->512->256->128->10 (BS=32)", batchSize: 32, layers: [784, 512, 256, 128, 10]))
        benchmarks.append(TransformerFFNBenchmark(id: "MLP-INF-005", name: "MLX FFN 768->3072->768 GELU (BS=32)", batchSize: 32, seqLen: 128, hiddenSize: 768, ffnSize: 3072))

        return benchmarks
    }

    /// Get benchmarks by ID for comparison lookup.
    public static func matchingBenchmarks() -> [String: MLXBenchmark] {
        var result: [String: MLXBenchmark] = [:]
        for benchmark in all() {
            result[benchmark.id] = benchmark
        }
        return result
    }
}
