// EmbeddingGPU.swift
// HeterogeneousFusion
//
// Phase 5: Vocab-shard embedding lookup via Metal compute kernels.
// Each unit owns a range of the embedding table (vocab split).
// Each thread scans one sequence position — writes output only if
// the token ID falls in this unit's vocab range.
//
// Scatter-free output: each output row is written by exactly one unit.
// Output buffer must be pre-zeroed before dispatch.

import Metal
import QuartzCore

/// Performs vocab-shard embedding lookup on GPU via Metal compute kernels.
public final class EmbeddingGPU: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create embedding command queue")
        }
        self.commandQueue = queue

        let source = Self.generateKernel()
        let library = try device.makeLibrary(source: source, options: nil)
        guard let fn = library.makeFunction(name: "embedding_lookup_vocab_shard") else {
            throw HeterogeneousError.metalSetupFailed("Missing kernel: embedding_lookup_vocab_shard")
        }
        self.pipeline = try device.makeComputePipelineState(function: fn)
    }

    /// Expose command queue for shared use.
    public var queue: MTLCommandQueue { commandQueue }

    /// Encode a vocab-shard embedding lookup into the given command buffer.
    ///
    /// - Parameters:
    ///   - tokenIds: Buffer of int32 token IDs [seqLen].
    ///   - tableShard: Buffer of float32 embedding rows for this unit's vocab range.
    ///   - output: Shared output buffer [seqLen, embeddingDim]. Pre-zeroed.
    ///   - seqLen: Number of tokens in the sequence.
    ///   - embeddingDim: Width of each embedding row.
    ///   - vocabOffset: Global vocab index where this shard starts.
    ///   - vocabSize: Number of vocab rows in this shard.
    ///   - commandBuffer: Metal command buffer to encode into.
    public func encode(
        tokenIds: MTLBuffer, tableShard: MTLBuffer, output: MTLBuffer,
        tableShardOffset: Int,
        seqLen: Int, embeddingDim: Int,
        vocabOffset: Int, vocabSize: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tokenIds, offset: 0, index: 0)
        encoder.setBuffer(tableShard, offset: tableShardOffset, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var sl = UInt32(seqLen)
        var ed = UInt32(embeddingDim)
        var vo = UInt32(vocabOffset)
        var vs = UInt32(vocabSize)
        encoder.setBytes(&sl, length: 4, index: 3)
        encoder.setBytes(&ed, length: 4, index: 4)
        encoder.setBytes(&vo, length: 4, index: 5)
        encoder.setBytes(&vs, length: 4, index: 6)

        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let groups = (seqLen + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Execute synchronously on the full table (single-unit). Returns wall-clock time in seconds.
    public func execute(
        tokenIds: MTLBuffer, table: MTLBuffer, output: MTLBuffer,
        seqLen: Int, embeddingDim: Int, vocabSize: Int
    ) -> Double {
        guard let cb = commandQueue.makeCommandBuffer() else { return 0 }
        encode(
            tokenIds: tokenIds, tableShard: table, output: output,
            tableShardOffset: 0,
            seqLen: seqLen, embeddingDim: embeddingDim,
            vocabOffset: 0, vocabSize: vocabSize,
            commandBuffer: cb
        )
        let start = CACurrentMediaTime()
        cb.commit()
        cb.waitUntilCompleted()
        return CACurrentMediaTime() - start
    }

    // MARK: - Metal Kernel Source

    private static func generateKernel() -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        // Each thread handles one sequence position.
        // Checks if the token ID falls in this unit's vocab range.
        // If yes: copies the embedding row to the output.
        // If no: skips (output was pre-zeroed).
        kernel void embedding_lookup_vocab_shard(
            device const int*   token_ids     [[buffer(0)]],
            device const float* table_shard   [[buffer(1)]],
            device float*       output        [[buffer(2)]],
            constant uint&      seq_len       [[buffer(3)]],
            constant uint&      embedding_dim [[buffer(4)]],
            constant uint&      vocab_offset  [[buffer(5)]],
            constant uint&      vocab_size    [[buffer(6)]],
            uint pos [[thread_position_in_grid]])
        {
            if (pos >= seq_len) return;

            int token = token_ids[pos];

            // Skip tokens not owned by this shard
            if (token < int(vocab_offset)) return;
            uint local_idx = uint(token) - vocab_offset;
            if (local_idx >= vocab_size) return;

            // Copy embedding row to output position
            device const float* src = table_shard + local_idx * embedding_dim;
            device float*       dst = output      + pos       * embedding_dim;

            for (uint d = 0; d < embedding_dim; d++) {
                dst[d] = src[d];
            }
        }
        """
    }
}
