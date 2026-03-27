// EmbeddingCPU.swift
// HeterogeneousFusion
//
// Phase 5: Vocab-shard embedding lookup via Accelerate (cblas_scopy).
// Each unit owns a range of the embedding table (vocab split).
// Scans all token IDs — copies the embedding row only if the token
// falls in this unit's vocab range.
//
// Scatter-free output: each output row is written by exactly one unit.
// Output buffer must be pre-zeroed before dispatch.

import Accelerate
import Foundation

/// Performs vocab-shard embedding lookup on CPU via Accelerate.
public final class EmbeddingCPU: Sendable {

    public init() {}

    /// Execute a vocab-shard embedding lookup on CPU.
    ///
    /// - Parameters:
    ///   - tokenIds: Pointer to int32 token IDs [seqLen].
    ///   - tableShard: Pointer to float32 embedding rows for this unit's vocab range.
    ///   - output: Output pointer [seqLen, embeddingDim]. Pre-zeroed.
    ///   - seqLen: Number of tokens in the sequence.
    ///   - embeddingDim: Width of each embedding row.
    ///   - vocabOffset: Global vocab index where this shard starts.
    ///   - vocabSize: Number of vocab rows in this shard.
    public func execute(
        tokenIds: UnsafePointer<Int32>,
        tableShard: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        seqLen: Int, embeddingDim: Int,
        vocabOffset: Int, vocabSize: Int
    ) {
        let ed = Int32(embeddingDim)

        for pos in 0..<seqLen {
            let token = Int(tokenIds[pos])

            // Skip tokens not owned by this shard
            if token < vocabOffset { continue }
            let localIdx = token - vocabOffset
            if localIdx >= vocabSize { continue }

            // Copy embedding row: cblas_scopy(N, src, 1, dst, 1)
            let src = tableShard + localIdx * embeddingDim
            let dst = output + pos * embeddingDim
            cblas_scopy(ed, src, 1, dst, 1)
        }
    }
}
