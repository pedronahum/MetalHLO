// TileCalculator.swift
// MetalHLOCore
//
// Calculates optimal tile configurations for specialized kernels.

import Metal
import Foundation

/// Configuration for tiled matrix multiplication.
public struct MatMulTileConfig: Hashable, Sendable {
    /// Tile size in M dimension (output rows).
    public let tileM: Int

    /// Tile size in N dimension (output columns).
    public let tileN: Int

    /// Tile size in K dimension (contraction).
    public let tileK: Int

    /// Number of SIMD groups (warps) to use.
    public let numSimdGroups: Int

    /// Shared memory size in bytes.
    public let sharedMemorySize: Int

    /// Loop unroll factor for K dimension.
    public let unrollFactor: Int

    /// Whether to use vectorized loads (float4/half4).
    public let useVectorLoads: Bool

    public init(
        tileM: Int,
        tileN: Int,
        tileK: Int,
        numSimdGroups: Int,
        sharedMemorySize: Int,
        unrollFactor: Int,
        useVectorLoads: Bool
    ) {
        self.tileM = tileM
        self.tileN = tileN
        self.tileK = tileK
        self.numSimdGroups = numSimdGroups
        self.sharedMemorySize = sharedMemorySize
        self.unrollFactor = unrollFactor
        self.useVectorLoads = useVectorLoads
    }
}

/// Configuration for tiled attention computation.
public struct AttentionTileConfig: Hashable, Sendable {
    /// Block size for query dimension.
    public let blockQ: Int

    /// Block size for key/value dimension.
    public let blockKV: Int

    /// Number of SIMD groups to use.
    public let numSimdGroups: Int

    /// Whether to use causal block skipping optimization.
    public let useCausalSkip: Bool

    /// Shared memory size in bytes.
    public let sharedMemorySize: Int

    /// Whether to use online softmax (memory efficient).
    public let useOnlineSoftmax: Bool

    public init(
        blockQ: Int,
        blockKV: Int,
        numSimdGroups: Int,
        useCausalSkip: Bool,
        sharedMemorySize: Int,
        useOnlineSoftmax: Bool
    ) {
        self.blockQ = blockQ
        self.blockKV = blockKV
        self.numSimdGroups = numSimdGroups
        self.useCausalSkip = useCausalSkip
        self.sharedMemorySize = sharedMemorySize
        self.useOnlineSoftmax = useOnlineSoftmax
    }
}

/// Configuration for reduction operations.
public struct ReductionTileConfig: Hashable, Sendable {
    /// Number of elements per thread.
    public let elementsPerThread: Int

    /// Threadgroup size.
    public let threadgroupSize: Int

    /// Whether to use simdgroup reduction.
    public let useSimdReduction: Bool

    /// Number of reduction stages.
    public let numStages: Int

    public init(
        elementsPerThread: Int,
        threadgroupSize: Int,
        useSimdReduction: Bool,
        numStages: Int
    ) {
        self.elementsPerThread = elementsPerThread
        self.threadgroupSize = threadgroupSize
        self.useSimdReduction = useSimdReduction
        self.numStages = numStages
    }
}

/// Calculates optimal tile configurations for various kernel types.
///
/// Takes into account:
/// - Device capabilities (max threads, shared memory)
/// - Problem dimensions (M, N, K, etc.)
/// - Data types (float vs half precision)
/// - Memory access patterns
public final class TileCalculator: Sendable {

    // MARK: - Device Capabilities

    /// Maximum threads per threadgroup.
    public let maxThreadsPerThreadgroup: Int

    /// Maximum shared memory in bytes.
    public let maxSharedMemory: Int

    /// SIMD width (32 for Apple Silicon).
    public let simdWidth: Int

    /// Whether the device supports simdgroup matrix operations.
    public let hasSimdgroupMatrixOps: Bool

    // MARK: - Initialization

    /// Creates a tile calculator with device-specific parameters.
    public init(
        maxThreadsPerThreadgroup: Int = 1024,
        maxSharedMemory: Int = 32768,
        simdWidth: Int = 32,
        hasSimdgroupMatrixOps: Bool = true
    ) {
        self.maxThreadsPerThreadgroup = maxThreadsPerThreadgroup
        self.maxSharedMemory = maxSharedMemory
        self.simdWidth = simdWidth
        self.hasSimdgroupMatrixOps = hasSimdgroupMatrixOps
    }

    /// Creates a tile calculator for a specific Metal device.
    public init(device: MTLDevice) {
        self.maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup.width
        self.maxSharedMemory = 32768  // 32KB typical for Apple Silicon
        self.simdWidth = 32
        // Apple Silicon M1+ supports simdgroup matrix ops
        self.hasSimdgroupMatrixOps = device.supportsFamily(.apple7)
    }

    // MARK: - MatMul Configuration

    /// Calculates optimal tile configuration for matrix multiplication.
    ///
    /// - Parameters:
    ///   - M: Rows of output matrix.
    ///   - N: Columns of output matrix.
    ///   - K: Inner dimension (columns of A, rows of B).
    ///   - elementType: Data type (affects bytes per element).
    /// - Returns: Optimal tile configuration.
    public func calculateMatMulTiles(
        M: Int,
        N: Int,
        K: Int,
        elementType: ElementType
    ) -> MatMulTileConfig {
        let bytesPerElement = max(elementType.byteSize, 2)  // At least 2 for f16

        // Start with ideal tiles based on hardware capabilities
        // For Apple Silicon with simdgroup matrix ops, 8x8 is the base tile
        var tileM: Int
        var tileN: Int
        var tileK: Int

        if hasSimdgroupMatrixOps {
            // Simdgroup matrix ops work best with 8x8 base tiles
            // Scale up to 32x32 or 64x64 for larger problems
            tileM = min(64, roundUpToPowerOf2(M, base: 8))
            tileN = min(64, roundUpToPowerOf2(N, base: 8))
            tileK = min(32, roundUpToPowerOf2(K, base: 8))
        } else {
            // Without simdgroup ops, use larger tiles with more threads
            tileM = min(128, roundUpToMultiple(M, of: simdWidth))
            tileN = min(128, roundUpToMultiple(N, of: simdWidth))
            tileK = min(32, max(8, K))
        }

        // Ensure tiles don't exceed problem size
        tileM = min(tileM, M)
        tileN = min(tileN, N)
        tileK = min(tileK, K)

        // Adjust for shared memory constraint
        // Shared memory needs: A tile (tileM x tileK) + B tile (tileK x tileN)
        while sharedMemoryForMatMul(tileM, tileN, tileK, bytesPerElement) > maxSharedMemory {
            if tileK > 8 {
                tileK = max(8, tileK - 8)
            } else if tileM > 16 && tileM >= tileN {
                tileM = max(16, tileM / 2)
            } else if tileN > 16 {
                tileN = max(16, tileN / 2)
            } else {
                break
            }
        }

        // Calculate number of SIMD groups
        // Each simdgroup handles a portion of the output tile
        let outputTileElements = tileM * tileN
        let elementsPerSimdGroup = simdWidth * 4  // Each thread computes 4 elements
        let numSimdGroups = min(8, max(1, outputTileElements / elementsPerSimdGroup))

        // Calculate unroll factor
        let numKIterations = (K + tileK - 1) / tileK
        let unrollFactor = min(8, max(1, numKIterations))

        // Check if we can use vectorized loads (float4/half4)
        let useVectorLoads = (K % 4 == 0) && (tileK % 4 == 0)

        return MatMulTileConfig(
            tileM: tileM,
            tileN: tileN,
            tileK: tileK,
            numSimdGroups: numSimdGroups,
            sharedMemorySize: sharedMemoryForMatMul(tileM, tileN, tileK, bytesPerElement),
            unrollFactor: unrollFactor,
            useVectorLoads: useVectorLoads
        )
    }

    // MARK: - Attention Configuration

    /// Calculates optimal tile configuration for attention computation.
    ///
    /// - Parameters:
    ///   - batchSize: Batch size.
    ///   - seqLen: Sequence length.
    ///   - numHeads: Number of attention heads.
    ///   - headDim: Dimension of each head.
    ///   - isCausal: Whether causal masking is used.
    ///   - elementType: Data type.
    /// - Returns: Optimal attention tile configuration.
    public func calculateAttentionTiles(
        batchSize: Int,
        seqLen: Int,
        numHeads: Int,
        headDim: Int,
        isCausal: Bool,
        elementType: ElementType
    ) -> AttentionTileConfig {
        let bytesPerElement = max(elementType.byteSize, 2)

        // Start with reasonable block sizes
        var blockQ = min(64, seqLen)
        var blockKV = min(64, seqLen)

        // Align to SIMD width
        blockQ = roundUpToMultiple(min(blockQ, seqLen), of: 8)
        blockKV = roundUpToMultiple(min(blockKV, seqLen), of: 8)

        // Adjust for shared memory constraint
        // Need: Q block, K block, V block, scores, output accumulator
        while attentionSharedMemory(blockQ, blockKV, headDim, bytesPerElement) > maxSharedMemory {
            if blockKV > 16 {
                blockKV = max(16, blockKV - 16)
            } else if blockQ > 16 {
                blockQ = max(16, blockQ - 16)
            } else {
                break
            }
        }

        // For long sequences, always use online softmax
        let useOnlineSoftmax = seqLen > 512

        let numSimdGroups = min(4, max(1, blockQ / 16))

        return AttentionTileConfig(
            blockQ: blockQ,
            blockKV: blockKV,
            numSimdGroups: numSimdGroups,
            useCausalSkip: isCausal,
            sharedMemorySize: attentionSharedMemory(blockQ, blockKV, headDim, bytesPerElement),
            useOnlineSoftmax: useOnlineSoftmax
        )
    }

    // MARK: - Reduction Configuration

    /// Calculates optimal configuration for reduction operations.
    ///
    /// - Parameters:
    ///   - numElements: Total number of elements to reduce.
    ///   - elementType: Data type.
    /// - Returns: Optimal reduction configuration.
    public func calculateReductionConfig(
        numElements: Int,
        elementType: ElementType
    ) -> ReductionTileConfig {
        // Use simdgroup reduction when available and beneficial
        let useSimdReduction = hasSimdgroupMatrixOps && numElements >= simdWidth

        // Calculate threadgroup size
        let threadgroupSize: Int
        if numElements <= 256 {
            threadgroupSize = min(numElements, 256)
        } else if numElements <= 1024 {
            threadgroupSize = 256
        } else {
            threadgroupSize = min(1024, maxThreadsPerThreadgroup)
        }

        // Elements per thread for large reductions
        let elementsPerThread = max(1, (numElements + threadgroupSize - 1) / threadgroupSize)

        // Number of reduction stages
        let numStages: Int
        if useSimdReduction {
            // With SIMD reduction: log2(simdWidth) + log2(numSimdgroups)
            let numSimdGroups = threadgroupSize / simdWidth
            numStages = Int(log2(Double(simdWidth))) + Int(log2(Double(max(1, numSimdGroups))))
        } else {
            // Tree reduction: log2(threadgroupSize)
            numStages = Int(log2(Double(threadgroupSize)))
        }

        return ReductionTileConfig(
            elementsPerThread: elementsPerThread,
            threadgroupSize: threadgroupSize,
            useSimdReduction: useSimdReduction,
            numStages: numStages
        )
    }

    // MARK: - Helper Methods

    private func sharedMemoryForMatMul(_ tileM: Int, _ tileN: Int, _ tileK: Int, _ bytes: Int) -> Int {
        // A tile: tileM x tileK
        // B tile: tileK x tileN
        return (tileM * tileK + tileK * tileN) * bytes
    }

    private func attentionSharedMemory(_ blockQ: Int, _ blockKV: Int, _ headDim: Int, _ bytes: Int) -> Int {
        // Q block: blockQ x headDim
        // K block: blockKV x headDim
        // V block: blockKV x headDim
        // Scores: blockQ x blockKV
        // Output accumulator: blockQ x headDim
        return (blockQ * headDim + 2 * blockKV * headDim + blockQ * blockKV + blockQ * headDim) * bytes
    }

    private func roundUpToMultiple(_ value: Int, of multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }

    private func roundUpToPowerOf2(_ value: Int, base: Int) -> Int {
        var result = base
        while result < value && result < 256 {
            result *= 2
        }
        return result
    }
}
