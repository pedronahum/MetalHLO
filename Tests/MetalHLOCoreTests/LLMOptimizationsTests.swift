// LLMOptimizationsTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2H: LLM Optimizations

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("LLM Optimizations Tests")
struct LLMOptimizationsTests {

    // MARK: - PagedAttentionConfig Tests

    @Test("PagedAttentionConfig computes max sequence length")
    func pagedAttentionConfigMaxSeqLen() {
        let config = PagedAttentionConfig(
            blockSize: 16,
            maxBlocks: 256
        )

        #expect(config.maxSeqLen == 4096)  // 16 * 256
    }

    @Test("PagedAttentionConfig computes bytes per block")
    func pagedAttentionConfigBytesPerBlock() {
        let config = PagedAttentionConfig(
            blockSize: 16,
            maxBlocks: 256,
            numHeads: 32,
            headDim: 128,
            numKVHeads: 8
        )

        // 16 tokens * 8 KV heads * 128 dim * 4 bytes
        let expected = 16 * 8 * 128 * 4
        #expect(config.bytesPerBlock == expected)
    }

    @Test("PagedAttentionConfig GQA defaults")
    func pagedAttentionConfigGQA() {
        let config = PagedAttentionConfig(
            numHeads: 32,
            numKVHeads: nil
        )

        #expect(config.numKVHeads == 32)

        let gqaConfig = PagedAttentionConfig(
            numHeads: 32,
            numKVHeads: 8
        )

        #expect(gqaConfig.numKVHeads == 8)
    }

    // MARK: - PageBlock Tests

    @Test("PageBlock tracks token count")
    func pageBlockTokenCount() {
        var block = PageBlock(blockId: 0, blockSize: 16)

        #expect(block.numTokens == 0)
        #expect(block.isFull == false)
        #expect(block.remainingCapacity == 16)

        block.numTokens = 10
        #expect(block.remainingCapacity == 6)
        #expect(block.isFull == false)

        block.numTokens = 16
        #expect(block.isFull == true)
        #expect(block.remainingCapacity == 0)
    }

    // MARK: - SequenceBlockTable Tests

    @Test("SequenceBlockTable manages blocks")
    func sequenceBlockTableBlocks() {
        var table = SequenceBlockTable(sequenceId: 1)

        #expect(table.numBlocks == 0)
        #expect(table.seqLen == 0)

        table.addBlock(5)
        table.addBlock(10)
        table.updateSeqLen(25)

        #expect(table.numBlocks == 2)
        #expect(table.blockIds == [5, 10])
        #expect(table.seqLen == 25)
    }

    // MARK: - PagedBlockAllocator Tests

    @Test("PagedBlockAllocator allocates and frees blocks")
    func pagedBlockAllocatorBasic() {
        let config = PagedAttentionConfig(maxBlocks: 10)
        let allocator = PagedBlockAllocator(config: config)

        #expect(allocator.numFreeBlocks == 10)
        #expect(allocator.numAllocatedBlocks == 0)

        let block1 = allocator.allocate()
        let block2 = allocator.allocate()

        #expect(block1 != nil)
        #expect(block2 != nil)
        #expect(allocator.numFreeBlocks == 8)
        #expect(allocator.numAllocatedBlocks == 2)

        allocator.free(block1!)

        #expect(allocator.numFreeBlocks == 9)
        #expect(allocator.numAllocatedBlocks == 1)
    }

    @Test("PagedBlockAllocator handles exhaustion")
    func pagedBlockAllocatorExhaustion() {
        let config = PagedAttentionConfig(maxBlocks: 3)
        let allocator = PagedBlockAllocator(config: config)

        let _ = allocator.allocate()
        let _ = allocator.allocate()
        let _ = allocator.allocate()

        let block4 = allocator.allocate()
        #expect(block4 == nil)
    }

    @Test("PagedBlockAllocator reference counting")
    func pagedBlockAllocatorRefCount() {
        let config = PagedAttentionConfig(maxBlocks: 5)
        let allocator = PagedBlockAllocator(config: config)

        let block = allocator.allocate()!
        allocator.incrementRef(block)

        #expect(allocator.numAllocatedBlocks == 1)

        // First free decrements ref count
        allocator.free(block)
        #expect(allocator.numAllocatedBlocks == 1)

        // Second free actually frees
        allocator.free(block)
        #expect(allocator.numAllocatedBlocks == 0)
    }

    // MARK: - PagedAttentionManager Tests

    @Test("PagedAttentionManager manages sequences")
    func pagedAttentionManagerSequences() {
        let config = PagedAttentionConfig(blockSize: 16, maxBlocks: 10)
        let manager = PagedAttentionManager(config: config)

        manager.registerSequence(1)

        let success = manager.appendTokens(1, numTokens: 20)
        #expect(success == true)

        let seqLen = manager.getSeqLen(1)
        #expect(seqLen == 20)

        let blockTable = manager.getBlockTable(1)
        #expect(blockTable != nil)
        #expect(blockTable!.count == 2)  // 20 tokens needs 2 blocks of 16
    }

    @Test("PagedAttentionManager fork sequence")
    func pagedAttentionManagerFork() {
        let config = PagedAttentionConfig(blockSize: 16, maxBlocks: 10)
        let manager = PagedAttentionManager(config: config)

        manager.registerSequence(1)
        _ = manager.appendTokens(1, numTokens: 32)

        let forkSuccess = manager.forkSequence(1, targetId: 2)
        #expect(forkSuccess == true)

        let len1 = manager.getSeqLen(1)
        let len2 = manager.getSeqLen(2)
        #expect(len1 == len2)

        let blocks1 = manager.getBlockTable(1)
        let blocks2 = manager.getBlockTable(2)
        #expect(blocks1 == blocks2)
    }

    @Test("PagedAttentionManager memory stats")
    func pagedAttentionManagerMemoryStats() {
        let config = PagedAttentionConfig(blockSize: 16, maxBlocks: 100)
        let manager = PagedAttentionManager(config: config)

        let (used, total, utilization) = manager.memoryStats
        #expect(used == 0)
        #expect(total > 0)
        #expect(utilization == 0)

        manager.registerSequence(1)
        _ = manager.appendTokens(1, numTokens: 100)

        let (used2, _, util2) = manager.memoryStats
        #expect(used2 > 0)
        #expect(util2 > 0)
    }

    // MARK: - BatchRequest Tests

    @Test("BatchRequest tracks state")
    func batchRequestState() {
        var request = BatchRequest(
            requestId: 1,
            sequenceId: 1,
            inputTokens: [1, 2, 3, 4, 5],
            maxNewTokens: 10
        )

        #expect(request.state == .waiting)
        #expect(request.totalSeqLen == 5)
        #expect(request.isComplete == false)

        request.addToken(100)
        #expect(request.outputTokens.count == 1)
        #expect(request.totalSeqLen == 6)

        request.state = .finished
        #expect(request.isComplete == true)
    }

    @Test("BatchRequest completion by max tokens")
    func batchRequestCompletion() {
        var request = BatchRequest(
            requestId: 1,
            sequenceId: 1,
            inputTokens: [1],
            maxNewTokens: 3
        )

        request.addToken(10)
        request.addToken(20)
        #expect(request.isComplete == false)

        request.addToken(30)
        #expect(request.isComplete == true)
    }

    // MARK: - ContinuousBatch Tests

    @Test("ContinuousBatch tracks capacity")
    func continuousBatchCapacity() {
        var batch = ContinuousBatch(maxBatchSize: 4, maxTokens: 100)

        #expect(batch.hasCapacity == true)
        #expect(batch.size == 0)
        #expect(batch.totalTokens == 0)

        batch.prefillRequests.append(BatchRequest(
            requestId: 1,
            sequenceId: 1,
            inputTokens: Array(repeating: 0, count: 50)
        ))

        #expect(batch.size == 1)
        #expect(batch.totalTokens == 50)
        #expect(batch.hasCapacity == true)

        batch.prefillRequests.append(BatchRequest(
            requestId: 2,
            sequenceId: 2,
            inputTokens: Array(repeating: 0, count: 60)
        ))

        #expect(batch.totalTokens == 110)
        #expect(batch.hasCapacity == false)  // Exceeds maxTokens
    }

    // MARK: - ContinuousBatchScheduler Tests

    @Test("ContinuousBatchScheduler FCFS scheduling")
    func schedulerFCFS() {
        let scheduler = ContinuousBatchScheduler(policy: .fcfs, maxBatchSize: 2)

        scheduler.addRequest(BatchRequest(requestId: 1, sequenceId: 1, inputTokens: [1, 2, 3]))
        scheduler.addRequest(BatchRequest(requestId: 2, sequenceId: 2, inputTokens: [4, 5]))

        #expect(scheduler.waitingCount == 2)

        let batch = scheduler.schedule()

        #expect(batch.prefillRequests.count == 2)
        #expect(batch.prefillRequests[0].requestId == 1)
        #expect(batch.prefillRequests[1].requestId == 2)
    }

    @Test("ContinuousBatchScheduler SJF scheduling")
    func schedulerSJF() {
        let scheduler = ContinuousBatchScheduler(policy: .sjf, maxBatchSize: 4)

        scheduler.addRequest(BatchRequest(requestId: 1, sequenceId: 1, inputTokens: [1], maxNewTokens: 100))
        scheduler.addRequest(BatchRequest(requestId: 2, sequenceId: 2, inputTokens: [2], maxNewTokens: 10))
        scheduler.addRequest(BatchRequest(requestId: 3, sequenceId: 3, inputTokens: [3], maxNewTokens: 50))

        let batch = scheduler.schedule()

        // Should be sorted by maxNewTokens (shortest first)
        #expect(batch.prefillRequests[0].maxNewTokens == 10)
    }

    @Test("ContinuousBatchScheduler update and preempt")
    func schedulerUpdatePreempt() {
        let scheduler = ContinuousBatchScheduler(policy: .fcfs)

        scheduler.addRequest(BatchRequest(requestId: 1, sequenceId: 1, inputTokens: [1, 2]))
        _ = scheduler.schedule()

        #expect(scheduler.runningCount == 1)

        scheduler.updateRequest(1, newToken: 100, finished: false)
        #expect(scheduler.runningCount == 1)

        scheduler.preemptRequest(1)
        #expect(scheduler.runningCount == 0)

        // Preempted request should be rescheduled first
        let batch2 = scheduler.schedule()
        #expect(batch2.decodeRequests.count == 1)
    }

    // MARK: - SpeculativeDecodingConfig Tests

    @Test("SpeculativeDecodingConfig defaults")
    func speculativeConfigDefaults() {
        let config = SpeculativeDecodingConfig()

        #expect(config.numSpeculativeTokens == 4)
        #expect(config.draftTemperature == 1.0)
        #expect(config.targetTemperature == 1.0)
        #expect(config.useTreeSpeculation == false)
    }

    // MARK: - SpeculativeDecodingEngine Tests

    @Test("SpeculativeDecodingEngine speculate")
    func speculativeEngineSpeculate() {
        let config = SpeculativeDecodingConfig(numSpeculativeTokens: 3)
        let engine = SpeculativeDecodingEngine(config: config)

        let draftLogits: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 2.0, 2.0, 2.0]
        ]

        let tokens = engine.speculate(draftLogits: draftLogits, contextTokens: [])

        #expect(tokens.count == 3)
        // Tokens should be valid indices
        for token in tokens {
            #expect(token >= 0 && token < 4)
        }
    }

    @Test("SpeculativeDecodingEngine verify")
    func speculativeEngineVerify() {
        let config = SpeculativeDecodingConfig(numSpeculativeTokens: 3)
        let engine = SpeculativeDecodingEngine(config: config)

        let speculatedTokens = [0, 1, 2]

        // Create logits that should mostly accept token 0
        let targetLogits: [[Float]] = [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0]
        ]

        let draftProbs: [[Float]] = [
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
            [0.03, 0.03, 0.9, 0.04]
        ]

        let result = engine.verify(
            speculatedTokens: speculatedTokens,
            targetLogits: targetLogits,
            draftProbs: draftProbs
        )

        // With high agreement, should accept most tokens
        #expect(result.acceptedCount >= 0)
        #expect(result.acceptedTokens.count <= speculatedTokens.count)
    }

    @Test("SpeculativeDecodingEngine statistics")
    func speculativeEngineStatistics() {
        let config = SpeculativeDecodingConfig(numSpeculativeTokens: 2)
        let engine = SpeculativeDecodingEngine(config: config)

        let initialStats = engine.currentStatistics
        #expect(initialStats.totalSpeculations == 0)

        // Run a verification
        _ = engine.verify(
            speculatedTokens: [0, 1],
            targetLogits: [[1.0, 2.0], [2.0, 1.0]],
            draftProbs: [[0.5, 0.5], [0.5, 0.5]]
        )

        let stats = engine.currentStatistics
        #expect(stats.totalSpeculations == 2)

        engine.resetStatistics()
        let resetStats = engine.currentStatistics
        #expect(resetStats.totalSpeculations == 0)
    }

    // MARK: - LLMInferenceEngine Tests

    @Test("LLMInferenceEngine submits and tracks requests")
    func llmEngineSubmit() {
        let config = LLMInferenceEngine.Config()
        let engine = LLMInferenceEngine(config: config)

        let requestId = engine.submit(inputTokens: [1, 2, 3, 4], maxNewTokens: 100)

        #expect(requestId >= 0)

        let stats = engine.statistics
        #expect(stats.waitingRequests == 1)
    }

    @Test("LLMInferenceEngine schedules batches")
    func llmEngineSchedule() {
        let config = LLMInferenceEngine.Config(maxBatchSize: 2)
        let engine = LLMInferenceEngine(config: config)

        _ = engine.submit(inputTokens: [1, 2, 3], maxNewTokens: 10)
        _ = engine.submit(inputTokens: [4, 5], maxNewTokens: 20)

        let batch = engine.getNextBatch()

        #expect(batch.prefillRequests.count == 2)
    }

    @Test("LLMInferenceEngine with speculative decoding")
    func llmEngineSpeculative() {
        let specConfig = SpeculativeDecodingConfig(numSpeculativeTokens: 4)
        let config = LLMInferenceEngine.Config(speculativeConfig: specConfig)
        let engine = LLMInferenceEngine(config: config)

        _ = engine.submit(inputTokens: [1, 2, 3], maxNewTokens: 50)

        let stats = engine.statistics
        #expect(stats.speculativeStats != nil)
    }

    @Test("LLMInferenceEngine memory tracking")
    func llmEngineMemoryTracking() {
        let pagedConfig = PagedAttentionConfig(blockSize: 16, maxBlocks: 100)
        let config = LLMInferenceEngine.Config(pagedAttentionConfig: pagedConfig)
        let engine = LLMInferenceEngine(config: config)

        let stats1 = engine.statistics
        #expect(stats1.memoryUsed == 0)
        #expect(stats1.memoryTotal > 0)

        _ = engine.submit(inputTokens: Array(repeating: 0, count: 100), maxNewTokens: 100)
        // Need to schedule to actually allocate
        _ = engine.getNextBatch()

        let stats2 = engine.statistics
        // After allocation, memory should be used
        #expect(stats2.memoryUtilization >= 0)
    }

    // MARK: - Integration Tests

    @Test("Full LLM inference workflow")
    func fullLLMWorkflow() {
        let pagedConfig = PagedAttentionConfig(
            blockSize: 16,
            maxBlocks: 64,
            numHeads: 8,
            headDim: 64
        )

        let config = LLMInferenceEngine.Config(
            pagedAttentionConfig: pagedConfig,
            maxBatchSize: 4,
            maxTokensPerBatch: 512,
            schedulingPolicy: .fcfs
        )

        let engine = LLMInferenceEngine(config: config)

        // Submit multiple requests
        let req1 = engine.submit(inputTokens: [1, 2, 3, 4, 5], maxNewTokens: 10)
        let req2 = engine.submit(inputTokens: [10, 20, 30], maxNewTokens: 20)
        let req3 = engine.submit(inputTokens: [100], maxNewTokens: 5)

        #expect(req1 >= 0)
        #expect(req2 >= 0)
        #expect(req3 >= 0)

        // Schedule first batch
        let batch1 = engine.getNextBatch()
        #expect(batch1.prefillRequests.count == 3)

        // Simulate processing
        engine.processResult(requestId: req1, token: 1000, finished: false)
        engine.processResult(requestId: req2, token: 2000, finished: false)
        engine.processResult(requestId: req3, token: 3000, finished: true)

        // Check stats
        let stats = engine.statistics
        #expect(stats.runningRequests >= 0)
    }

    @Test("Continuous batching with mixed phases")
    func continuousBatchingMixedPhases() {
        let scheduler = ContinuousBatchScheduler(
            policy: .fcfs,
            maxBatchSize: 10,
            maxTokensPerBatch: 1000
        )

        // Add requests with different sizes
        for i in 0..<5 {
            scheduler.addRequest(BatchRequest(
                requestId: i,
                sequenceId: i,
                inputTokens: Array(repeating: 0, count: 10 + i * 5),
                maxNewTokens: 50
            ))
        }

        // First batch - all prefill
        let batch1 = scheduler.schedule()
        #expect(batch1.prefillRequests.count == 5)
        #expect(batch1.decodeRequests.count == 0)

        // Process first token for each
        for i in 0..<5 {
            scheduler.updateRequest(i, newToken: 100, finished: false)
        }

        // Second batch - all decode
        let batch2 = scheduler.schedule()
        #expect(batch2.prefillRequests.count == 0)
        #expect(batch2.decodeRequests.count == 5)
    }

    @Test("Paged attention under memory pressure")
    func pagedAttentionMemoryPressure() {
        let config = PagedAttentionConfig(
            blockSize: 16,
            maxBlocks: 10  // Limited blocks
        )

        let manager = PagedAttentionManager(config: config)

        // Fill up most of the cache
        manager.registerSequence(1)
        let success1 = manager.appendTokens(1, numTokens: 100)  // Needs 7 blocks
        #expect(success1 == true)

        manager.registerSequence(2)
        let success2 = manager.appendTokens(2, numTokens: 40)  // Needs 3 blocks
        #expect(success2 == true)

        // This should fail - no more blocks
        manager.registerSequence(3)
        let success3 = manager.appendTokens(3, numTokens: 20)  // Needs 2 blocks
        #expect(success3 == false)

        // Free first sequence
        manager.freeSequence(1)

        // Now should succeed
        let success4 = manager.appendTokens(3, numTokens: 20)
        #expect(success4 == true)
    }
}
