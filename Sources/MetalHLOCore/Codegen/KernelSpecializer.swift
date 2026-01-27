// KernelSpecializer.swift
// MetalHLOCore
//
// Engine for generating shape-specialized Metal kernels at compile time.

import Metal
import Foundation

/// Key for caching specialized kernels.
public struct SpecializationKey: Hashable, Sendable {
    /// Operation type.
    public let opType: HLOOpKind

    /// Input and output shapes.
    public let shapes: [TensorType]

    /// Additional configuration (transpose, causal, etc.).
    public let config: [String: Int]

    public init(
        opType: HLOOpKind,
        shapes: [TensorType],
        config: [String: Int] = [:]
    ) {
        self.opType = opType
        self.shapes = shapes
        self.config = config
    }

    // Custom hash to group similar shapes for cache efficiency
    public func hash(into hasher: inout Hasher) {
        hasher.combine(opType)
        for shape in shapes {
            // Round dimensions to cache-friendly boundaries
            for dim in shape.shape {
                hasher.combine(roundToCacheBoundary(dim))
            }
            hasher.combine(shape.elementType)
        }
        for (key, value) in config.sorted(by: { $0.key < $1.key }) {
            hasher.combine(key)
            hasher.combine(value)
        }
    }

    private func roundToCacheBoundary(_ dim: Int) -> Int {
        // Round small dimensions exactly, larger ones to powers of 2
        if dim <= 64 {
            return dim
        } else if dim <= 128 {
            return 128
        } else if dim <= 256 {
            return 256
        } else if dim <= 512 {
            return 512
        } else if dim <= 1024 {
            return 1024
        } else if dim <= 2048 {
            return 2048
        } else if dim <= 4096 {
            return 4096
        } else {
            // Round to nearest 1024 for very large dimensions
            return ((dim + 1023) / 1024) * 1024
        }
    }
}

/// Type-erased tile configuration wrapper for Sendable conformance.
public struct TileConfigWrapper: Sendable {
    /// The tile configuration type name.
    public let typeName: String

    /// Encoded configuration values.
    public let values: [String: Int]

    public init(matmul config: MatMulTileConfig) {
        self.typeName = "MatMulTileConfig"
        self.values = [
            "tileM": config.tileM,
            "tileN": config.tileN,
            "tileK": config.tileK,
            "numSimdGroups": config.numSimdGroups,
            "sharedMemorySize": config.sharedMemorySize,
            "unrollFactor": config.unrollFactor,
            "useVectorLoads": config.useVectorLoads ? 1 : 0
        ]
    }

    public init(attention config: AttentionTileConfig) {
        self.typeName = "AttentionTileConfig"
        self.values = [
            "blockQ": config.blockQ,
            "blockKV": config.blockKV,
            "numSimdGroups": config.numSimdGroups,
            "useCausalSkip": config.useCausalSkip ? 1 : 0,
            "sharedMemorySize": config.sharedMemorySize,
            "useOnlineSoftmax": config.useOnlineSoftmax ? 1 : 0
        ]
    }

    public init(reduction config: ReductionTileConfig) {
        self.typeName = "ReductionTileConfig"
        self.values = [
            "elementsPerThread": config.elementsPerThread,
            "threadgroupSize": config.threadgroupSize,
            "useSimdReduction": config.useSimdReduction ? 1 : 0,
            "numStages": config.numStages
        ]
    }

    public init(generic value: Int) {
        self.typeName = "Generic"
        self.values = ["value": value]
    }
}

/// Result of kernel specialization.
public struct SpecializedKernel: Sendable {
    /// Generated Metal shader source code.
    public let source: String

    /// Entry point function name.
    public let functionName: String

    /// Tile configuration used.
    public let tileConfig: TileConfigWrapper

    /// Estimated performance metrics.
    public let estimatedMetrics: PerformanceEstimate

    public init(
        source: String,
        functionName: String,
        tileConfig: TileConfigWrapper,
        estimatedMetrics: PerformanceEstimate
    ) {
        self.source = source
        self.functionName = functionName
        self.tileConfig = tileConfig
        self.estimatedMetrics = estimatedMetrics
    }
}

/// Estimated performance characteristics of a kernel.
public struct PerformanceEstimate: Sendable {
    /// Estimated FLOPs.
    public let flops: Int

    /// Estimated memory bandwidth (bytes).
    public let memoryBytes: Int

    /// Whether kernel is compute-bound or memory-bound.
    public let isComputeBound: Bool

    /// Arithmetic intensity (FLOPs / byte).
    public var arithmeticIntensity: Double {
        guard memoryBytes > 0 else { return 0 }
        return Double(flops) / Double(memoryBytes)
    }

    public init(flops: Int, memoryBytes: Int, isComputeBound: Bool) {
        self.flops = flops
        self.memoryBytes = memoryBytes
        self.isComputeBound = isComputeBound
    }
}

/// Protocol for kernel code generators.
public protocol KernelCodeGenerator: Sendable {
    /// The operation type this generator handles.
    var supportedOps: Set<HLOOpKind> { get }

    /// Whether this generator can handle the given shapes.
    func canSpecialize(shapes: [TensorType], config: [String: Int]) -> Bool

    /// Generates specialized kernel code.
    func generate(
        shapes: [TensorType],
        config: [String: Int],
        tileCalculator: TileCalculator
    ) -> SpecializedKernel
}

/// Engine for generating and caching specialized kernels.
///
/// `KernelSpecializer` decides whether to use a specialized or generic kernel
/// based on operation type and shapes, then generates optimized Metal code
/// for the specific dimensions.
///
/// Benefits of specialization:
/// - No runtime branching on dimensions
/// - Optimal tile sizes computed at compile time
/// - Loops can be fully unrolled
/// - Shared memory sized exactly
/// - Better register allocation
public final class KernelSpecializer: @unchecked Sendable {

    // MARK: - Properties

    /// Tile calculator for optimal configurations.
    private let tileCalculator: TileCalculator

    /// Registered code generators.
    private var generators: [HLOOpKind: KernelCodeGenerator] = [:]

    /// Cache of generated kernel sources.
    private var sourceCache: [SpecializationKey: SpecializedKernel] = [:]

    /// Cache of compiled pipelines.
    private var pipelineCache: [String: MTLComputePipelineState] = [:]

    /// Lock for thread-safe access.
    private let lock = NSLock()

    /// Statistics about specialization.
    private var stats = SpecializationStats()

    // MARK: - Initialization

    /// Creates a kernel specializer with a tile calculator.
    public init(tileCalculator: TileCalculator = TileCalculator()) {
        self.tileCalculator = tileCalculator
        registerBuiltInGenerators()
    }

    /// Creates a kernel specializer for a specific Metal device.
    public init(device: MTLDevice) {
        self.tileCalculator = TileCalculator(device: device)
        registerBuiltInGenerators()
    }

    private func registerBuiltInGenerators() {
        register(MatMulCodeGenerator())
        register(AttentionCodeGenerator())
        register(ElementwiseCodeGenerator())
        register(ReductionCodeGenerator())
    }

    // MARK: - Registration

    /// Registers a code generator.
    public func register(_ generator: KernelCodeGenerator) {
        lock.lock()
        defer { lock.unlock() }
        for op in generator.supportedOps {
            generators[op] = generator
        }
    }

    // MARK: - Specialization Decision

    /// Determines whether to use a specialized kernel for the given operation.
    ///
    /// Specialization criteria:
    /// - Compute-heavy operations (matmul, attention, reduction)
    /// - Shapes are common enough to benefit from caching
    /// - Expected reuse justifies compilation cost
    public func shouldSpecialize(
        op: HLOOpKind,
        shapes: [TensorType]
    ) -> Bool {
        // Always specialize compute-intensive operations
        switch op {
        case .dot, .dotGeneral:
            return true

        case .customCall:
            // Specialize attention, norms, etc.
            return true

        case .reduce:
            // Specialize reductions over large tensors
            let elements = shapes.first?.count ?? 0
            return elements > 10000

        case .convolution:
            return true

        default:
            // Specialize elementwise only for very large tensors
            if op.isBinaryArithmetic || op.isUnary {
                let elements = shapes.first?.count ?? 0
                return elements > 1_000_000
            }
            return false
        }
    }

    // MARK: - Kernel Generation

    /// Gets or generates a specialized kernel for the given operation.
    ///
    /// - Parameters:
    ///   - op: The HLO operation kind.
    ///   - shapes: Input and output tensor shapes.
    ///   - config: Additional configuration parameters.
    /// - Returns: The specialized kernel, or nil if not supported.
    public func getSpecializedKernel(
        op: HLOOpKind,
        shapes: [TensorType],
        config: [String: Int] = [:]
    ) -> SpecializedKernel? {
        let key = SpecializationKey(opType: op, shapes: shapes, config: config)

        // Check cache first
        lock.lock()
        if let cached = sourceCache[key] {
            lock.unlock()
            stats.cacheHits += 1
            return cached
        }
        lock.unlock()

        // Get generator for this operation
        guard let generator = generators[op] else {
            stats.misses += 1
            return nil
        }

        // Check if generator can handle these shapes
        guard generator.canSpecialize(shapes: shapes, config: config) else {
            stats.misses += 1
            return nil
        }

        // Generate the kernel
        let kernel = generator.generate(
            shapes: shapes,
            config: config,
            tileCalculator: tileCalculator
        )

        // Cache the result
        lock.lock()
        sourceCache[key] = kernel
        stats.generated += 1
        lock.unlock()

        return kernel
    }

    /// Compiles a specialized kernel for a Metal device.
    ///
    /// - Parameters:
    ///   - kernel: The specialized kernel to compile.
    ///   - device: The Metal device.
    /// - Returns: The compiled pipeline state.
    /// - Throws: If compilation fails.
    public func compile(
        _ kernel: SpecializedKernel,
        device: MTLDevice
    ) throws -> MTLComputePipelineState {
        let cacheKey = "\(kernel.functionName)_\(device.name)"

        // Check pipeline cache
        lock.lock()
        if let cached = pipelineCache[cacheKey] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        // Compile the shader
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version3_1

        let library = try device.makeLibrary(source: kernel.source, options: options)

        guard let function = library.makeFunction(name: kernel.functionName) else {
            throw SpecializationError.functionNotFound(kernel.functionName)
        }

        let pipeline = try device.makeComputePipelineState(function: function)

        // Cache the pipeline
        lock.lock()
        pipelineCache[cacheKey] = pipeline
        lock.unlock()

        return pipeline
    }

    // MARK: - Statistics

    /// Returns specialization statistics.
    public var statistics: SpecializationStats {
        lock.lock()
        defer { lock.unlock() }
        return stats
    }

    /// Clears all caches.
    public func clearCache() {
        lock.lock()
        defer { lock.unlock() }
        sourceCache.removeAll()
        pipelineCache.removeAll()
    }
}

/// Statistics about kernel specialization.
public struct SpecializationStats: Sendable {
    /// Number of cache hits.
    public var cacheHits: Int = 0

    /// Number of kernels generated.
    public var generated: Int = 0

    /// Number of operations not specialized.
    public var misses: Int = 0

    /// Total kernels in cache.
    public var cacheSize: Int { cacheHits + generated }

    /// Cache hit rate.
    public var hitRate: Double {
        let total = cacheHits + generated + misses
        guard total > 0 else { return 0 }
        return Double(cacheHits) / Double(total)
    }
}

/// Errors during kernel specialization.
public enum SpecializationError: Error, CustomStringConvertible {
    case unsupportedOperation(HLOOpKind)
    case unsupportedShapes([TensorType])
    case functionNotFound(String)
    case compilationFailed(String)

    public var description: String {
        switch self {
        case .unsupportedOperation(let op):
            return "Cannot specialize operation: \(op)"
        case .unsupportedShapes(let shapes):
            return "Cannot specialize for shapes: \(shapes)"
        case .functionNotFound(let name):
            return "Function not found: \(name)"
        case .compilationFailed(let reason):
            return "Compilation failed: \(reason)"
        }
    }
}

// MARK: - Code Generators

/// Code generator for elementwise operations.
public struct ElementwiseCodeGenerator: KernelCodeGenerator, Sendable {

    public let supportedOps: Set<HLOOpKind> = [
        .add, .subtract, .multiply, .divide,
        .exponential, .log, .sqrt, .rsqrt,
        .tanh, .logistic, .negate, .abs
    ]

    public init() {}

    public func canSpecialize(shapes: [TensorType], config: [String: Int]) -> Bool {
        guard let first = shapes.first else { return false }
        return first.count > 1_000_000
    }

    public func generate(
        shapes: [TensorType],
        config: [String: Int],
        tileCalculator: TileCalculator
    ) -> SpecializedKernel {
        let inputShape = shapes[0]
        let numElements = inputShape.count
        let dtype = MetalType.from(inputShape.elementType)

        let functionName = "elementwise_specialized_\(numElements)_\(dtype.rawValue)"

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void \(functionName)(
            device const \(dtype.rawValue)* input [[buffer(0)]],
            device \(dtype.rawValue)* output [[buffer(1)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= \(numElements)) return;
            \(dtype.rawValue) x = input[gid];
            output[gid] = x;  // Operation applied at call site
        }
        """

        let estimate = PerformanceEstimate(
            flops: numElements,
            memoryBytes: numElements * dtype.byteSize * 2,
            isComputeBound: false
        )

        return SpecializedKernel(
            source: source,
            functionName: functionName,
            tileConfig: TileConfigWrapper(generic: numElements),
            estimatedMetrics: estimate
        )
    }
}

/// Code generator for reduction operations.
public struct ReductionCodeGenerator: KernelCodeGenerator, Sendable {

    public let supportedOps: Set<HLOOpKind> = [.reduce]

    public init() {}

    public func canSpecialize(shapes: [TensorType], config: [String: Int]) -> Bool {
        guard let first = shapes.first else { return false }
        return first.count > 10000
    }

    public func generate(
        shapes: [TensorType],
        config: [String: Int],
        tileCalculator: TileCalculator
    ) -> SpecializedKernel {
        let inputShape = shapes[0]
        let numElements = inputShape.count
        let dtype = MetalType.from(inputShape.elementType)
        let reductionConfig = tileCalculator.calculateReductionConfig(
            numElements: numElements,
            elementType: inputShape.elementType
        )

        let functionName = "reduce_sum_\(numElements)_\(dtype.rawValue)"
        let tgSize = reductionConfig.threadgroupSize

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void \(functionName)(
            device const \(dtype.rawValue)* input [[buffer(0)]],
            device \(dtype.rawValue)* output [[buffer(1)]],
            threadgroup \(dtype.rawValue)* shared_data [[threadgroup(0)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tg_size [[threads_per_threadgroup]],
            uint gid [[thread_position_in_grid]]
        ) {
            // Each thread loads and accumulates multiple elements
            \(dtype.rawValue) sum = 0;
            for (uint i = gid; i < \(numElements); i += tg_size * \(tgSize)) {
                sum += input[i];
            }

            // Store to shared memory
            shared_data[tid] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Tree reduction in shared memory
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_data[tid] += shared_data[tid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Write final result
            if (tid == 0) {
                output[0] = shared_data[0];
            }
        }
        """

        let estimate = PerformanceEstimate(
            flops: numElements,
            memoryBytes: numElements * dtype.byteSize + dtype.byteSize,
            isComputeBound: false
        )

        return SpecializedKernel(
            source: source,
            functionName: functionName,
            tileConfig: TileConfigWrapper(reduction: reductionConfig),
            estimatedMetrics: estimate
        )
    }
}

/// Code generator for matrix multiplication.
public struct MatMulCodeGenerator: KernelCodeGenerator, Sendable {

    public let supportedOps: Set<HLOOpKind> = [.dot, .dotGeneral]

    public init() {}

    public func canSpecialize(shapes: [TensorType], config: [String: Int]) -> Bool {
        guard shapes.count >= 2 else { return false }
        guard shapes[0].rank >= 2 && shapes[1].rank >= 2 else { return false }
        return true
    }

    public func generate(
        shapes: [TensorType],
        config: [String: Int],
        tileCalculator: TileCalculator
    ) -> SpecializedKernel {
        let shapeA = shapes[0]
        let shapeB = shapes[1]

        // Extract dimensions for matmul
        let M = shapeA.shape[shapeA.rank - 2]
        let K = shapeA.shape[shapeA.rank - 1]
        let N = shapeB.shape[shapeB.rank - 1]

        let dtype = MetalType.from(shapeA.elementType)
        let tileConfig = tileCalculator.calculateMatMulTiles(
            M: M, N: N, K: K,
            elementType: shapeA.elementType
        )

        let functionName = "matmul_\(M)x\(K)x\(N)_\(dtype.rawValue)"

        let source = generateMatMulSource(
            functionName: functionName,
            M: M, N: N, K: K,
            dtype: dtype,
            tileConfig: tileConfig
        )

        let flops = 2 * M * N * K
        let memoryBytes = (M * K + K * N + M * N) * dtype.byteSize

        let estimate = PerformanceEstimate(
            flops: flops,
            memoryBytes: memoryBytes,
            isComputeBound: flops > memoryBytes * 10
        )

        return SpecializedKernel(
            source: source,
            functionName: functionName,
            tileConfig: TileConfigWrapper(matmul: tileConfig),
            estimatedMetrics: estimate
        )
    }

    private func generateMatMulSource(
        functionName: String,
        M: Int, N: Int, K: Int,
        dtype: MetalType,
        tileConfig: MatMulTileConfig
    ) -> String {
        let tileM = tileConfig.tileM
        let tileN = tileConfig.tileN
        let tileK = tileConfig.tileK

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Specialized matmul for M=\(M), N=\(N), K=\(K)
        // Tile: \(tileM)x\(tileN)x\(tileK)
        kernel void \(functionName)(
            device const \(dtype.rawValue)* A [[buffer(0)]],
            device const \(dtype.rawValue)* B [[buffer(1)]],
            device \(dtype.rawValue)* C [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            // Constants baked in at compile time
            constexpr uint M = \(M);
            constexpr uint N = \(N);
            constexpr uint K = \(K);
            constexpr uint TILE_M = \(tileM);
            constexpr uint TILE_N = \(tileN);
            constexpr uint TILE_K = \(tileK);

            // Compute output position
            uint row = tgid.y * TILE_M + tid.y;
            uint col = tgid.x * TILE_N + tid.x;

            if (row >= M || col >= N) return;

            // Accumulator
            \(dtype.rawValue) acc = 0;

            // Tiled multiplication
            for (uint k_block = 0; k_block < K; k_block += TILE_K) {
                // Unrolled inner loop for TILE_K
                #pragma unroll
                for (uint k = 0; k < TILE_K && (k_block + k) < K; k++) {
                    uint k_idx = k_block + k;
                    acc += A[row * K + k_idx] * B[k_idx * N + col];
                }
            }

            C[row * N + col] = acc;
        }
        """
    }
}

/// Code generator for attention operations.
public struct AttentionCodeGenerator: KernelCodeGenerator, Sendable {

    public let supportedOps: Set<HLOOpKind> = [.customCall]

    public init() {}

    public func canSpecialize(shapes: [TensorType], config: [String: Int]) -> Bool {
        // Check if this is an attention operation
        guard config["is_attention"] == 1 else { return false }
        guard shapes.count >= 3 else { return false }  // Q, K, V
        return true
    }

    public func generate(
        shapes: [TensorType],
        config: [String: Int],
        tileCalculator: TileCalculator
    ) -> SpecializedKernel {
        let qShape = shapes[0]
        let dtype = MetalType.from(qShape.elementType)

        // Extract attention dimensions
        // Assuming shape is [batch, heads, seq, head_dim]
        let batchSize = qShape.shape.count > 3 ? qShape.shape[0] : 1
        let numHeads = qShape.shape.count > 3 ? qShape.shape[1] : qShape.shape[0]
        let seqLen = qShape.shape.count > 3 ? qShape.shape[2] : qShape.shape[1]
        let headDim = qShape.shape.last ?? 64

        let isCausal = config["is_causal"] == 1

        let tileConfig = tileCalculator.calculateAttentionTiles(
            batchSize: batchSize,
            seqLen: seqLen,
            numHeads: numHeads,
            headDim: headDim,
            isCausal: isCausal,
            elementType: qShape.elementType
        )

        let functionName = "attention_b\(batchSize)_h\(numHeads)_s\(seqLen)_d\(headDim)_\(isCausal ? "causal" : "full")"

        let source = generateAttentionSource(
            functionName: functionName,
            batchSize: batchSize,
            numHeads: numHeads,
            seqLen: seqLen,
            headDim: headDim,
            isCausal: isCausal,
            dtype: dtype,
            tileConfig: tileConfig
        )

        // FLOPs: QK^T (2*seq*seq*dim) + softmax (~5*seq*seq) + AV (2*seq*seq*dim) per head
        let flopsPerHead = 2 * seqLen * seqLen * headDim + 5 * seqLen * seqLen + 2 * seqLen * seqLen * headDim
        let totalFlops = batchSize * numHeads * flopsPerHead

        let memoryBytes = batchSize * numHeads * seqLen * headDim * dtype.byteSize * 4  // Q, K, V, O

        let estimate = PerformanceEstimate(
            flops: totalFlops,
            memoryBytes: memoryBytes,
            isComputeBound: true
        )

        return SpecializedKernel(
            source: source,
            functionName: functionName,
            tileConfig: TileConfigWrapper(attention: tileConfig),
            estimatedMetrics: estimate
        )
    }

    private func generateAttentionSource(
        functionName: String,
        batchSize: Int,
        numHeads: Int,
        seqLen: Int,
        headDim: Int,
        isCausal: Bool,
        dtype: MetalType,
        tileConfig: AttentionTileConfig
    ) -> String {
        let scale = 1.0 / sqrt(Float(headDim))

        return """
        #include <metal_stdlib>
        using namespace metal;

        // Specialized attention for batch=\(batchSize), heads=\(numHeads), seq=\(seqLen), dim=\(headDim)
        // Causal: \(isCausal), Block: \(tileConfig.blockQ)x\(tileConfig.blockKV)
        kernel void \(functionName)(
            device const \(dtype.rawValue)* Q [[buffer(0)]],
            device const \(dtype.rawValue)* K [[buffer(1)]],
            device const \(dtype.rawValue)* V [[buffer(2)]],
            device \(dtype.rawValue)* O [[buffer(3)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            // Constants baked in
            constexpr uint BATCH = \(batchSize);
            constexpr uint HEADS = \(numHeads);
            constexpr uint SEQ = \(seqLen);
            constexpr uint DIM = \(headDim);
            constexpr float SCALE = \(scale)f;
            constexpr bool IS_CAUSAL = \(isCausal);

            uint batch = gid.z;
            uint head = gid.y;
            uint q_pos = gid.x;

            if (batch >= BATCH || head >= HEADS || q_pos >= SEQ) return;

            // Base offsets
            uint q_base = ((batch * HEADS + head) * SEQ + q_pos) * DIM;
            uint kv_head_base = (batch * HEADS + head) * SEQ;

            // Online softmax state
            float m_i = -INFINITY;
            float l_i = 0.0f;

            // First pass: compute softmax normalization
            for (uint k_pos = 0; k_pos < SEQ; k_pos++) {
                if (IS_CAUSAL && k_pos > q_pos) break;

                uint k_base = (kv_head_base + k_pos) * DIM;
                float score = 0.0f;

                for (uint d = 0; d < DIM; d++) {
                    score += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                score *= SCALE;

                float m_new = max(m_i, score);
                float exp_old = exp(m_i - m_new);
                float exp_new = exp(score - m_new);
                l_i = l_i * exp_old + exp_new;
                m_i = m_new;
            }

            float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            uint o_base = q_base;

            // Initialize output
            for (uint d = 0; d < DIM; d++) {
                O[o_base + d] = \(dtype.rawValue)(0);
            }

            // Second pass: weighted sum
            for (uint k_pos = 0; k_pos < SEQ; k_pos++) {
                if (IS_CAUSAL && k_pos > q_pos) break;

                uint k_base = (kv_head_base + k_pos) * DIM;
                float score = 0.0f;

                for (uint d = 0; d < DIM; d++) {
                    score += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                score *= SCALE;

                float weight = exp(score - m_i) * l_inv;
                uint v_base = k_base;

                for (uint d = 0; d < DIM; d++) {
                    O[o_base + d] = \(dtype.rawValue)(float(O[o_base + d]) + weight * float(V[v_base + d]));
                }
            }
        }
        """
    }
}
