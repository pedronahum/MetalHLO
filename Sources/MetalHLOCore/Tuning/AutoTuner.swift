// AutoTuner.swift
// MetalHLOCore
//
// Auto-tuning infrastructure for finding optimal kernel parameters.

import Metal
import Foundation

// MARK: - Search Space

/// Defines the parameter search space for tuning.
public struct SearchSpace: Sendable {
    /// Available tile sizes for M dimension.
    public let tileMOptions: [Int]

    /// Available tile sizes for N dimension.
    public let tileNOptions: [Int]

    /// Available tile sizes for K dimension.
    public let tileKOptions: [Int]

    /// Available numbers of SIMD groups.
    public let numSimdGroupOptions: [Int]

    /// Available vector widths.
    public let vectorWidthOptions: [Int]

    /// Whether to try loop unrolling.
    public let unrollOptions: [Bool]

    public init(
        tileMOptions: [Int] = [32, 64, 128],
        tileNOptions: [Int] = [32, 64, 128],
        tileKOptions: [Int] = [8, 16, 32],
        numSimdGroupOptions: [Int] = [2, 4, 8],
        vectorWidthOptions: [Int] = [1, 2, 4],
        unrollOptions: [Bool] = [true, false]
    ) {
        self.tileMOptions = tileMOptions
        self.tileNOptions = tileNOptions
        self.tileKOptions = tileKOptions
        self.numSimdGroupOptions = numSimdGroupOptions
        self.vectorWidthOptions = vectorWidthOptions
        self.unrollOptions = unrollOptions
    }

    /// Total number of configurations in this search space.
    public var totalConfigurations: Int {
        tileMOptions.count *
        tileNOptions.count *
        tileKOptions.count *
        numSimdGroupOptions.count *
        vectorWidthOptions.count *
        unrollOptions.count
    }

    /// Search space for MatMul operations.
    public static func forMatMul(M: Int, N: Int, K: Int) -> SearchSpace {
        // Filter options based on problem size
        let maxTileM = min(128, M)
        let maxTileN = min(128, N)
        let maxTileK = min(32, K)

        return SearchSpace(
            tileMOptions: [32, 64, 128].filter { $0 <= maxTileM },
            tileNOptions: [32, 64, 128].filter { $0 <= maxTileN },
            tileKOptions: [8, 16, 32].filter { $0 <= maxTileK },
            numSimdGroupOptions: [2, 4, 8],
            vectorWidthOptions: [1, 2, 4],
            unrollOptions: [true, false]
        )
    }

    /// Search space for Attention operations.
    public static func forAttention(seqLen: Int, headDim: Int) -> SearchSpace {
        let maxBlockQ = min(128, seqLen)
        let maxBlockKV = min(128, seqLen)

        return SearchSpace(
            tileMOptions: [32, 64, 128].filter { $0 <= maxBlockQ },
            tileNOptions: [32, 64, 128].filter { $0 <= maxBlockKV },
            tileKOptions: [headDim], // Fixed to head dimension
            numSimdGroupOptions: [2, 4, 8],
            vectorWidthOptions: [1, 2, 4],
            unrollOptions: [true]
        )
    }

    /// Search space for reduction operations.
    public static func forReduction(numElements: Int) -> SearchSpace {
        return SearchSpace(
            tileMOptions: [64, 128, 256, 512, 1024].filter { $0 <= numElements },
            tileNOptions: [1],
            tileKOptions: [1],
            numSimdGroupOptions: [4, 8, 16, 32],
            vectorWidthOptions: [1, 2, 4, 8],
            unrollOptions: [true, false]
        )
    }

    /// Default search space.
    public static let `default` = SearchSpace()
}

// MARK: - Tuning Configuration

/// A specific configuration to benchmark.
public struct TuningConfiguration: Hashable, Codable, Sendable {
    public let tileM: Int
    public let tileN: Int
    public let tileK: Int
    public let numSimdGroups: Int
    public let vectorWidth: Int
    public let useUnroll: Bool

    public init(
        tileM: Int,
        tileN: Int,
        tileK: Int,
        numSimdGroups: Int,
        vectorWidth: Int,
        useUnroll: Bool
    ) {
        self.tileM = tileM
        self.tileN = tileN
        self.tileK = tileK
        self.numSimdGroups = numSimdGroups
        self.vectorWidth = vectorWidth
        self.useUnroll = useUnroll
    }

    /// Configuration description for logging.
    public var description: String {
        "tile[\(tileM)x\(tileN)x\(tileK)]_simd\(numSimdGroups)_vec\(vectorWidth)_unroll\(useUnroll)"
    }
}

// MARK: - Tuning Key

/// Unique key for caching tuning results.
public struct TuningKey: Hashable, Codable, Sendable {
    /// Operation type identifier.
    public let opType: String

    /// Tensor shapes (dimensions).
    public let shapes: [[Int]]

    /// Data type.
    public let dtype: String

    /// Device identifier.
    public let deviceID: String

    public init(opType: HLOOpKind, shapes: [TensorType], device: MTLDevice) {
        self.opType = String(describing: opType)
        self.shapes = shapes.map { $0.shape }
        self.dtype = shapes.first.map { String(describing: $0.elementType) } ?? "float32"
        self.deviceID = device.name
    }

    public init(opType: String, shapes: [[Int]], dtype: String, deviceID: String) {
        self.opType = opType
        self.shapes = shapes
        self.dtype = dtype
        self.deviceID = deviceID
    }
}

// MARK: - Tuning Result

/// Result of auto-tuning a kernel.
public struct TuningResult: Codable, Sendable {
    /// Best configuration found.
    public let bestConfig: TuningConfiguration

    /// Measured execution time (median, in seconds).
    public let measuredTime: Double

    /// GFLOPS achieved.
    public let gflops: Double

    /// Number of configurations tried.
    public let configsTried: Int

    /// Timestamp when tuning was performed.
    public let timestamp: Date

    public init(
        bestConfig: TuningConfiguration,
        measuredTime: Double,
        gflops: Double,
        configsTried: Int
    ) {
        self.bestConfig = bestConfig
        self.measuredTime = measuredTime
        self.gflops = gflops
        self.configsTried = configsTried
        self.timestamp = Date()
    }
}

// MARK: - Tuning Strategy

/// Strategy for when to tune kernels.
public enum TuningStrategy: Sendable {
    /// Tune at compile time (slower compilation, optimal runtime).
    case offline(maxTrials: Int)

    /// Use heuristics at compile time, tune at first run.
    case online(maxTrials: Int)

    /// Use only heuristics (fastest compilation).
    case heuristicsOnly

    /// Load pre-computed tuning database.
    case database
}

// MARK: - Auto Tuner

/// Auto-tuning engine for finding optimal kernel parameters.
///
/// The auto-tuner empirically finds the best configuration for each
/// kernel by benchmarking different parameter combinations.
public final class AutoTuner: @unchecked Sendable {

    // MARK: - Properties

    /// Metal device.
    private let device: MTLDevice

    /// Command queue for benchmarking.
    private let commandQueue: MTLCommandQueue

    /// Tuning database for caching results.
    private let database: TuningDatabase

    /// Tuning strategy.
    public var strategy: TuningStrategy

    /// Number of warmup runs before benchmarking.
    public var warmupRuns: Int = 5

    /// Number of benchmark runs.
    public var benchmarkRuns: Int = 20

    /// Maximum shared memory (32KB for Apple Silicon).
    private let maxSharedMemory: Int = 32768

    /// Lock for thread safety.
    private let lock = NSLock()

    // MARK: - Statistics

    /// Number of tuning operations performed.
    private(set) var tuningCount: Int = 0

    /// Number of cache hits.
    private(set) var cacheHits: Int = 0

    // MARK: - Initialization

    /// Creates an auto-tuner for the given device.
    ///
    /// - Parameters:
    ///   - device: Metal device.
    ///   - strategy: Tuning strategy.
    ///   - databasePath: Path to the tuning database file.
    public init(
        device: MTLDevice,
        strategy: TuningStrategy = .heuristicsOnly,
        databasePath: String? = nil
    ) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.strategy = strategy

        let path = databasePath ?? AutoTuner.defaultDatabasePath()
        self.database = TuningDatabase(path: path)
    }

    /// Default path for the tuning database.
    private static func defaultDatabasePath() -> String {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir.appendingPathComponent("metalhlo_tuning.json").path
    }

    // MARK: - Tuning

    /// Gets optimal configuration for an operation.
    ///
    /// Depending on the strategy, this may:
    /// - Return cached results from the database
    /// - Run auto-tuning to find optimal parameters
    /// - Return heuristic-based defaults
    ///
    /// - Parameters:
    ///   - op: Operation kind.
    ///   - shapes: Input tensor shapes.
    /// - Returns: Optimal configuration.
    public func getOptimalConfig(
        for op: HLOOpKind,
        shapes: [TensorType]
    ) -> TuningConfiguration {
        let key = TuningKey(opType: op, shapes: shapes, device: device)

        // Check database cache first
        if let cached = database.get(key) {
            lock.lock()
            cacheHits += 1
            lock.unlock()
            return cached.bestConfig
        }

        // Apply strategy
        switch strategy {
        case .offline(let maxTrials):
            let result = tune(key: key, op: op, shapes: shapes, maxTrials: maxTrials)
            return result.bestConfig

        case .online(let maxTrials):
            // Schedule background tuning, return heuristic for now
            scheduleBackgroundTuning(key: key, op: op, shapes: shapes, maxTrials: maxTrials)
            return getHeuristicConfig(for: op, shapes: shapes)

        case .heuristicsOnly:
            return getHeuristicConfig(for: op, shapes: shapes)

        case .database:
            // Database miss - fall back to heuristics
            return getHeuristicConfig(for: op, shapes: shapes)
        }
    }

    /// Tunes a kernel to find the optimal configuration.
    ///
    /// - Parameters:
    ///   - key: Tuning key.
    ///   - op: Operation kind.
    ///   - shapes: Input tensor shapes.
    ///   - maxTrials: Maximum number of configurations to try.
    /// - Returns: Tuning result with best configuration.
    public func tune(
        key: TuningKey,
        op: HLOOpKind,
        shapes: [TensorType],
        maxTrials: Int = 50
    ) -> TuningResult {
        lock.lock()
        tuningCount += 1
        lock.unlock()

        // Generate search space
        let searchSpace = generateSearchSpace(for: op, shapes: shapes)

        // Generate all valid configurations
        var configs = generateConfigurations(searchSpace: searchSpace, shapes: shapes)

        // Shuffle and limit to maxTrials
        configs.shuffle()
        if configs.count > maxTrials {
            configs = Array(configs.prefix(maxTrials))
        }

        // Benchmark each configuration
        var bestConfig: TuningConfiguration?
        var bestTime: Double = .infinity

        for config in configs {
            if let time = benchmark(config: config, op: op, shapes: shapes) {
                if time < bestTime {
                    bestTime = time
                    bestConfig = config
                }
            }
        }

        // Calculate GFLOPS
        let flops = estimateFLOPs(for: op, shapes: shapes)
        let gflops = bestTime > 0 ? Double(flops) / bestTime / 1e9 : 0

        // Create result
        let result = TuningResult(
            bestConfig: bestConfig ?? getHeuristicConfig(for: op, shapes: shapes),
            measuredTime: bestTime,
            gflops: gflops,
            configsTried: configs.count
        )

        // Cache result
        database.store(key, result: result)

        return result
    }

    // MARK: - Search Space Generation

    private func generateSearchSpace(for op: HLOOpKind, shapes: [TensorType]) -> SearchSpace {
        switch op {
        case .dot, .dotGeneral:
            guard shapes.count >= 2 else { return .default }
            let M = shapes[0].shape.count >= 2 ? shapes[0].shape[0] : 1
            let K = shapes[0].shape.count >= 2 ? shapes[0].shape[1] : shapes[0].shape[0]
            let N = shapes[1].shape.count >= 2 ? shapes[1].shape[1] : shapes[1].shape[0]
            return .forMatMul(M: M, N: N, K: K)

        case .customCall:
            // Assume attention if shapes match pattern
            if shapes.count >= 3 && shapes[0].shape.count == 4 {
                let seqLen = shapes[0].shape[2]
                let headDim = shapes[0].shape[3]
                return .forAttention(seqLen: seqLen, headDim: headDim)
            }
            return .default

        case .reduce:
            let numElements = shapes[0].shape.reduce(1, *)
            return .forReduction(numElements: numElements)

        default:
            return .default
        }
    }

    private func generateConfigurations(searchSpace: SearchSpace, shapes: [TensorType]) -> [TuningConfiguration] {
        var configs: [TuningConfiguration] = []
        let bytesPerElement = max(shapes.first?.elementType.byteSize ?? 4, 2)

        for tileM in searchSpace.tileMOptions {
            for tileN in searchSpace.tileNOptions {
                for tileK in searchSpace.tileKOptions {
                    for numSimdGroups in searchSpace.numSimdGroupOptions {
                        for vectorWidth in searchSpace.vectorWidthOptions {
                            for useUnroll in searchSpace.unrollOptions {
                                // Validate configuration
                                let sharedMem = (tileM * tileK + tileK * tileN) * bytesPerElement
                                guard sharedMem <= maxSharedMemory else { continue }

                                let threads = numSimdGroups * 32
                                guard threads <= device.maxThreadsPerThreadgroup.width else { continue }

                                configs.append(TuningConfiguration(
                                    tileM: tileM,
                                    tileN: tileN,
                                    tileK: tileK,
                                    numSimdGroups: numSimdGroups,
                                    vectorWidth: vectorWidth,
                                    useUnroll: useUnroll
                                ))
                            }
                        }
                    }
                }
            }
        }

        return configs
    }

    // MARK: - Benchmarking

    private func benchmark(
        config: TuningConfiguration,
        op: HLOOpKind,
        shapes: [TensorType]
    ) -> Double? {
        // Create test buffers
        guard let buffers = createTestBuffers(shapes: shapes) else {
            return nil
        }

        // Create a simple benchmark kernel based on operation
        // This is a simplified version - real implementation would generate
        // specialized kernels for each configuration
        guard let pipeline = createBenchmarkPipeline(config: config, op: op, shapes: shapes) else {
            return nil
        }

        // Warmup
        for _ in 0..<warmupRuns {
            executeBenchmark(pipeline: pipeline, buffers: buffers, config: config, shapes: shapes)
        }

        // Benchmark
        var times: [Double] = []

        for _ in 0..<benchmarkRuns {
            let commandBuffer = commandQueue.makeCommandBuffer()!

            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)

            for (index, buffer) in buffers.enumerated() {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }

            let gridSize = computeGridSize(config: config, shapes: shapes)
            let threadgroupSize = MTLSize(width: config.numSimdGroups * 32, height: 1, depth: 1)

            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()

            let start = CFAbsoluteTimeGetCurrent()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            times.append(elapsed)
        }

        // Return median time
        times.sort()
        return times[times.count / 2]
    }

    private func createTestBuffers(shapes: [TensorType]) -> [MTLBuffer]? {
        var buffers: [MTLBuffer] = []

        for shape in shapes {
            let byteSize = shape.shape.reduce(1, *) * shape.elementType.byteSize
            guard let buffer = device.makeBuffer(length: max(byteSize, 4), options: .storageModeShared) else {
                return nil
            }
            buffers.append(buffer)
        }

        // Add output buffer
        if let firstShape = shapes.first {
            let outputSize = firstShape.shape.reduce(1, *) * firstShape.elementType.byteSize
            if let outputBuffer = device.makeBuffer(length: max(outputSize, 4), options: .storageModeShared) {
                buffers.append(outputBuffer)
            }
        }

        return buffers
    }

    private func createBenchmarkPipeline(
        config: TuningConfiguration,
        op: HLOOpKind,
        shapes: [TensorType]
    ) -> MTLComputePipelineState? {
        // Generate simple benchmark kernel
        let source = generateBenchmarkKernel(config: config, op: op, shapes: shapes)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        guard let library = try? device.makeLibrary(source: source, options: options),
              let function = library.makeFunction(name: "benchmark_kernel"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            return nil
        }

        return pipeline
    }

    private func generateBenchmarkKernel(
        config: TuningConfiguration,
        op: HLOOpKind,
        shapes: [TensorType]
    ) -> String {
        // Simplified benchmark kernel - just measures overhead and basic compute
        let dtype = shapes.first?.elementType == .float16 ? "half" : "float"

        return """
        #include <metal_stdlib>
        using namespace metal;

        kernel void benchmark_kernel(
            device const \(dtype)* input [[buffer(0)]],
            device \(dtype)* output [[buffer(1)]],
            uint gid [[thread_position_in_grid]]
        ) {
            // Simple computation to measure configuration overhead
            \(dtype) val = input[gid];
            \(dtype) acc = 0;

            // Simulate tiled computation
            for (int i = 0; i < \(config.tileK); i++) {
                acc += val * val;
            }

            output[gid] = acc;
        }
        """
    }

    private func executeBenchmark(
        pipeline: MTLComputePipelineState,
        buffers: [MTLBuffer],
        config: TuningConfiguration,
        shapes: [TensorType]
    ) {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipeline)

        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }

        let gridSize = computeGridSize(config: config, shapes: shapes)
        let threadgroupSize = MTLSize(width: min(config.numSimdGroups * 32, 256), height: 1, depth: 1)

        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    private func computeGridSize(config: TuningConfiguration, shapes: [TensorType]) -> MTLSize {
        guard let firstShape = shapes.first else {
            return MTLSize(width: 1, height: 1, depth: 1)
        }

        let totalElements = firstShape.shape.reduce(1, *)
        let threadsPerGroup = config.numSimdGroups * 32
        let numGroups = (totalElements + threadsPerGroup - 1) / threadsPerGroup

        return MTLSize(width: numGroups, height: 1, depth: 1)
    }

    // MARK: - Heuristics

    /// Gets a heuristic-based configuration without benchmarking.
    public func getHeuristicConfig(for op: HLOOpKind, shapes: [TensorType]) -> TuningConfiguration {
        switch op {
        case .dot, .dotGeneral:
            return getMatMulHeuristic(shapes: shapes)
        case .customCall:
            if shapes.count >= 3 && shapes[0].shape.count == 4 {
                return getAttentionHeuristic(shapes: shapes)
            }
            return defaultConfig()
        case .reduce:
            return getReductionHeuristic(shapes: shapes)
        default:
            return defaultConfig()
        }
    }

    private func getMatMulHeuristic(shapes: [TensorType]) -> TuningConfiguration {
        guard shapes.count >= 2 else { return defaultConfig() }

        let M = shapes[0].shape.count >= 2 ? shapes[0].shape[0] : 1
        let K = shapes[0].shape.count >= 2 ? shapes[0].shape[1] : shapes[0].shape[0]
        let N = shapes[1].shape.count >= 2 ? shapes[1].shape[1] : shapes[1].shape[0]

        // Choose tile sizes based on problem dimensions
        let tileM: Int
        let tileN: Int
        let tileK: Int

        if M >= 1024 && N >= 1024 {
            // Large problem: maximize throughput
            tileM = 128
            tileN = 128
            tileK = 32
        } else if M >= 256 && N >= 256 {
            // Medium problem
            tileM = 64
            tileN = 64
            tileK = 32
        } else {
            // Small problem: minimize overhead
            tileM = min(32, M)
            tileN = min(32, N)
            tileK = min(16, K)
        }

        return TuningConfiguration(
            tileM: tileM,
            tileN: tileN,
            tileK: tileK,
            numSimdGroups: 4,
            vectorWidth: 4,
            useUnroll: K <= 64
        )
    }

    private func getAttentionHeuristic(shapes: [TensorType]) -> TuningConfiguration {
        let seqLen = shapes[0].shape.count >= 3 ? shapes[0].shape[2] : 512

        let blockSize = seqLen > 1024 ? 128 : 64

        return TuningConfiguration(
            tileM: blockSize,
            tileN: blockSize,
            tileK: 64, // Typical head dimension
            numSimdGroups: 4,
            vectorWidth: 4,
            useUnroll: true
        )
    }

    private func getReductionHeuristic(shapes: [TensorType]) -> TuningConfiguration {
        let numElements = shapes.first?.shape.reduce(1, *) ?? 1024

        let threadgroupSize = numElements > 10000 ? 256 : 128

        return TuningConfiguration(
            tileM: threadgroupSize,
            tileN: 1,
            tileK: 1,
            numSimdGroups: threadgroupSize / 32,
            vectorWidth: 4,
            useUnroll: true
        )
    }

    private func defaultConfig() -> TuningConfiguration {
        TuningConfiguration(
            tileM: 64,
            tileN: 64,
            tileK: 16,
            numSimdGroups: 4,
            vectorWidth: 4,
            useUnroll: true
        )
    }

    // MARK: - Helper Methods

    private func estimateFLOPs(for op: HLOOpKind, shapes: [TensorType]) -> Int {
        switch op {
        case .dot, .dotGeneral:
            guard shapes.count >= 2 else { return 0 }
            let M = shapes[0].shape.count >= 2 ? shapes[0].shape[0] : 1
            let K = shapes[0].shape.count >= 2 ? shapes[0].shape[1] : shapes[0].shape[0]
            let N = shapes[1].shape.count >= 2 ? shapes[1].shape[1] : shapes[1].shape[0]
            return 2 * M * N * K
        default:
            return shapes.first?.shape.reduce(1, *) ?? 0
        }
    }

    private func scheduleBackgroundTuning(
        key: TuningKey,
        op: HLOOpKind,
        shapes: [TensorType],
        maxTrials: Int
    ) {
        DispatchQueue.global(qos: .background).async { [weak self] in
            _ = self?.tune(key: key, op: op, shapes: shapes, maxTrials: maxTrials)
        }
    }

    // MARK: - Public Statistics

    /// Returns tuning statistics.
    public var statistics: (tuningCount: Int, cacheHits: Int, cacheSize: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (tuningCount, cacheHits, database.count)
    }

    /// Clears the tuning cache.
    public func clearCache() {
        database.clear()
    }
}
