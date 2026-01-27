// LLMOptimizations.swift
// MetalHLOCore
//
// LLM-specific optimizations: Paged Attention, Continuous Batching, Speculative Decoding.
// These optimizations are critical for efficient LLM inference serving.

import Foundation

// MARK: - Paged Attention

/// Block size for paged attention (number of tokens per block).
public let kPagedAttentionBlockSize = 16

/// Configuration for paged attention.
public struct PagedAttentionConfig: Sendable, Equatable {
    /// Number of tokens per block.
    public let blockSize: Int

    /// Maximum number of blocks in the cache.
    public let maxBlocks: Int

    /// Number of attention heads.
    public let numHeads: Int

    /// Dimension per head.
    public let headDim: Int

    /// Number of KV heads (for GQA).
    public let numKVHeads: Int

    /// Maximum sequence length supported.
    public var maxSeqLen: Int {
        blockSize * maxBlocks
    }

    public init(
        blockSize: Int = kPagedAttentionBlockSize,
        maxBlocks: Int = 256,
        numHeads: Int = 32,
        headDim: Int = 128,
        numKVHeads: Int? = nil
    ) {
        self.blockSize = blockSize
        self.maxBlocks = maxBlocks
        self.numHeads = numHeads
        self.headDim = headDim
        self.numKVHeads = numKVHeads ?? numHeads
    }

    /// Bytes per block for K or V.
    public var bytesPerBlock: Int {
        blockSize * numKVHeads * headDim * MemoryLayout<Float>.size
    }

    /// Total cache memory in bytes.
    public var totalCacheMemory: Int {
        maxBlocks * bytesPerBlock * 2  // K and V
    }
}

/// A block in the paged attention cache.
public struct PageBlock: Sendable, Equatable {
    /// Unique block ID.
    public let blockId: Int

    /// Number of tokens currently stored in this block.
    public var numTokens: Int

    /// Whether this block is full.
    public var isFull: Bool {
        numTokens >= blockSize
    }

    /// Block size (tokens per block).
    public let blockSize: Int

    /// Reference count for copy-on-write.
    public var refCount: Int

    public init(blockId: Int, blockSize: Int = kPagedAttentionBlockSize) {
        self.blockId = blockId
        self.numTokens = 0
        self.blockSize = blockSize
        self.refCount = 1
    }

    /// Remaining capacity in tokens.
    public var remainingCapacity: Int {
        blockSize - numTokens
    }
}

/// Manages the paged KV cache for a sequence.
public struct SequenceBlockTable: Sendable {
    /// Sequence ID.
    public let sequenceId: Int

    /// Block IDs for this sequence (in order).
    public private(set) var blockIds: [Int]

    /// Current sequence length in tokens.
    public var seqLen: Int

    /// Block size.
    public let blockSize: Int

    public init(sequenceId: Int, blockSize: Int = kPagedAttentionBlockSize) {
        self.sequenceId = sequenceId
        self.blockIds = []
        self.seqLen = 0
        self.blockSize = blockSize
    }

    /// Number of blocks used.
    public var numBlocks: Int {
        blockIds.count
    }

    /// Adds a new block to this sequence.
    public mutating func addBlock(_ blockId: Int) {
        blockIds.append(blockId)
    }

    /// Updates sequence length.
    public mutating func updateSeqLen(_ newLen: Int) {
        seqLen = newLen
    }
}

/// Block allocator for paged attention.
public final class PagedBlockAllocator: @unchecked Sendable {
    private let config: PagedAttentionConfig
    private var freeBlocks: [Int]
    private var allocatedBlocks: Set<Int>
    private var blockRefCounts: [Int: Int]
    private let lock = NSLock()

    public init(config: PagedAttentionConfig) {
        self.config = config
        self.freeBlocks = Array(0..<config.maxBlocks)
        self.allocatedBlocks = []
        self.blockRefCounts = [:]
    }

    /// Allocates a new block.
    public func allocate() -> Int? {
        lock.lock()
        defer { lock.unlock() }

        guard let blockId = freeBlocks.popLast() else {
            return nil
        }

        allocatedBlocks.insert(blockId)
        blockRefCounts[blockId] = 1
        return blockId
    }

    /// Frees a block.
    public func free(_ blockId: Int) {
        lock.lock()
        defer { lock.unlock() }

        guard allocatedBlocks.contains(blockId) else { return }

        if let refCount = blockRefCounts[blockId], refCount > 1 {
            blockRefCounts[blockId] = refCount - 1
            return
        }

        allocatedBlocks.remove(blockId)
        blockRefCounts.removeValue(forKey: blockId)
        freeBlocks.append(blockId)
    }

    /// Increases reference count for copy-on-write.
    public func incrementRef(_ blockId: Int) {
        lock.lock()
        defer { lock.unlock() }

        if let refCount = blockRefCounts[blockId] {
            blockRefCounts[blockId] = refCount + 1
        }
    }

    /// Number of free blocks.
    public var numFreeBlocks: Int {
        lock.lock()
        defer { lock.unlock() }
        return freeBlocks.count
    }

    /// Number of allocated blocks.
    public var numAllocatedBlocks: Int {
        lock.lock()
        defer { lock.unlock() }
        return allocatedBlocks.count
    }

    /// Memory utilization ratio.
    public var utilizationRatio: Double {
        Double(numAllocatedBlocks) / Double(config.maxBlocks)
    }
}

/// Paged attention manager.
public final class PagedAttentionManager: @unchecked Sendable {
    private let config: PagedAttentionConfig
    private let allocator: PagedBlockAllocator
    private var sequenceTables: [Int: SequenceBlockTable]
    private let lock = NSLock()

    public init(config: PagedAttentionConfig) {
        self.config = config
        self.allocator = PagedBlockAllocator(config: config)
        self.sequenceTables = [:]
    }

    /// Registers a new sequence.
    public func registerSequence(_ sequenceId: Int) {
        lock.lock()
        defer { lock.unlock() }

        sequenceTables[sequenceId] = SequenceBlockTable(
            sequenceId: sequenceId,
            blockSize: config.blockSize
        )
    }

    /// Appends tokens to a sequence, allocating blocks as needed.
    public func appendTokens(_ sequenceId: Int, numTokens: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard var table = sequenceTables[sequenceId] else {
            return false
        }

        var tokensToAdd = numTokens
        let currentTokensInLastBlock = table.seqLen % config.blockSize

        // Fill current block first
        if currentTokensInLastBlock > 0 && !table.blockIds.isEmpty {
            let remaining = config.blockSize - currentTokensInLastBlock
            let toFill = min(remaining, tokensToAdd)
            tokensToAdd -= toFill
            table.updateSeqLen(table.seqLen + toFill)
        }

        // Allocate new blocks as needed
        while tokensToAdd > 0 {
            guard let newBlockId = allocator.allocate() else {
                sequenceTables[sequenceId] = table
                return false  // Out of memory
            }

            table.addBlock(newBlockId)
            let toFill = min(config.blockSize, tokensToAdd)
            tokensToAdd -= toFill
            table.updateSeqLen(table.seqLen + toFill)
        }

        sequenceTables[sequenceId] = table
        return true
    }

    /// Frees a sequence and its blocks.
    public func freeSequence(_ sequenceId: Int) {
        lock.lock()
        defer { lock.unlock() }

        guard let table = sequenceTables.removeValue(forKey: sequenceId) else {
            return
        }

        for blockId in table.blockIds {
            allocator.free(blockId)
        }
    }

    /// Forks a sequence (copy-on-write).
    public func forkSequence(_ sourceId: Int, targetId: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard let sourceTable = sequenceTables[sourceId] else {
            return false
        }

        // Create new table sharing blocks
        var newTable = SequenceBlockTable(
            sequenceId: targetId,
            blockSize: config.blockSize
        )

        for blockId in sourceTable.blockIds {
            allocator.incrementRef(blockId)
            newTable.addBlock(blockId)
        }
        newTable.updateSeqLen(sourceTable.seqLen)

        sequenceTables[targetId] = newTable
        return true
    }

    /// Gets block table for a sequence.
    public func getBlockTable(_ sequenceId: Int) -> [Int]? {
        lock.lock()
        defer { lock.unlock() }
        return sequenceTables[sequenceId]?.blockIds
    }

    /// Gets sequence length.
    public func getSeqLen(_ sequenceId: Int) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        return sequenceTables[sequenceId]?.seqLen
    }

    /// Memory statistics.
    public var memoryStats: (used: Int, total: Int, utilization: Double) {
        let allocated = allocator.numAllocatedBlocks
        let total = config.maxBlocks
        return (
            used: allocated * config.bytesPerBlock * 2,
            total: config.totalCacheMemory,
            utilization: allocator.utilizationRatio
        )
    }
}

// MARK: - Continuous Batching

/// Request state in continuous batching.
public enum RequestState: Sendable, Equatable {
    /// Request is waiting to be scheduled.
    case waiting

    /// Request is currently being processed (prefill phase).
    case prefilling

    /// Request is in decode phase (generating tokens).
    case decoding

    /// Request is complete.
    case finished

    /// Request was preempted and needs to resume.
    case preempted
}

/// A single request in the batch.
public struct BatchRequest: Sendable, Equatable {
    /// Unique request ID.
    public let requestId: Int

    /// Sequence ID for KV cache.
    public let sequenceId: Int

    /// Input token IDs.
    public let inputTokens: [Int]

    /// Generated token IDs so far.
    public private(set) var outputTokens: [Int]

    /// Current state.
    public var state: RequestState

    /// Maximum tokens to generate.
    public let maxNewTokens: Int

    /// Current position in generation.
    public var position: Int

    /// Priority (lower = higher priority).
    public let priority: Int

    public init(
        requestId: Int,
        sequenceId: Int,
        inputTokens: [Int],
        maxNewTokens: Int = 256,
        priority: Int = 0
    ) {
        self.requestId = requestId
        self.sequenceId = sequenceId
        self.inputTokens = inputTokens
        self.outputTokens = []
        self.state = .waiting
        self.maxNewTokens = maxNewTokens
        self.position = 0
        self.priority = priority
    }

    /// Total sequence length.
    public var totalSeqLen: Int {
        inputTokens.count + outputTokens.count
    }

    /// Whether generation is complete.
    public var isComplete: Bool {
        outputTokens.count >= maxNewTokens || state == .finished
    }

    /// Adds a generated token.
    public mutating func addToken(_ tokenId: Int) {
        outputTokens.append(tokenId)
        position += 1
    }
}

/// Batch for continuous batching.
public struct ContinuousBatch: Sendable {
    /// Requests in prefill phase.
    public var prefillRequests: [BatchRequest]

    /// Requests in decode phase.
    public var decodeRequests: [BatchRequest]

    /// Maximum batch size.
    public let maxBatchSize: Int

    /// Maximum total tokens in batch.
    public let maxTokens: Int

    public init(maxBatchSize: Int = 64, maxTokens: Int = 4096) {
        self.prefillRequests = []
        self.decodeRequests = []
        self.maxBatchSize = maxBatchSize
        self.maxTokens = maxTokens
    }

    /// Total number of requests.
    public var size: Int {
        prefillRequests.count + decodeRequests.count
    }

    /// Whether batch has room for more requests.
    public var hasCapacity: Bool {
        size < maxBatchSize && totalTokens < maxTokens
    }

    /// Total tokens in current batch.
    public var totalTokens: Int {
        let prefillTokens = prefillRequests.reduce(0) { $0 + $1.inputTokens.count }
        let decodeTokens = decodeRequests.count  // 1 token per decode request
        return prefillTokens + decodeTokens
    }
}

/// Scheduler for continuous batching.
public final class ContinuousBatchScheduler: @unchecked Sendable {

    /// Scheduling policy.
    public enum Policy: Sendable {
        /// First come, first served.
        case fcfs

        /// Shortest job first (by max tokens).
        case sjf

        /// Priority-based scheduling.
        case priority
    }

    private let policy: Policy
    private let maxBatchSize: Int
    private let maxTokensPerBatch: Int
    private var waitingQueue: [BatchRequest]
    private var runningRequests: [Int: BatchRequest]
    private var preemptedRequests: [BatchRequest]
    private let lock = NSLock()

    public init(
        policy: Policy = .fcfs,
        maxBatchSize: Int = 64,
        maxTokensPerBatch: Int = 4096
    ) {
        self.policy = policy
        self.maxBatchSize = maxBatchSize
        self.maxTokensPerBatch = maxTokensPerBatch
        self.waitingQueue = []
        self.runningRequests = [:]
        self.preemptedRequests = []
    }

    /// Adds a new request.
    public func addRequest(_ request: BatchRequest) {
        lock.lock()
        defer { lock.unlock() }

        waitingQueue.append(request)
        sortWaitingQueue()
    }

    /// Schedules the next batch.
    public func schedule() -> ContinuousBatch {
        lock.lock()
        defer { lock.unlock() }

        var batch = ContinuousBatch(
            maxBatchSize: maxBatchSize,
            maxTokens: maxTokensPerBatch
        )

        // Track which request IDs have been added to avoid duplicates
        var addedRequestIds = Set<Int>()

        // First, add preempted requests (they have priority)
        while !preemptedRequests.isEmpty && batch.hasCapacity {
            var request = preemptedRequests.removeFirst()
            request.state = .decoding
            batch.decodeRequests.append(request)
            runningRequests[request.requestId] = request
            addedRequestIds.insert(request.requestId)
        }

        // Add decode requests for running sequences (skip already added)
        for (requestId, request) in runningRequests {
            if !addedRequestIds.contains(requestId) &&
               request.state == .decoding && !request.isComplete {
                batch.decodeRequests.append(request)
                addedRequestIds.insert(requestId)
            }
        }

        // Add new prefill requests
        while !waitingQueue.isEmpty && batch.hasCapacity {
            let request = waitingQueue.first!
            let prefillTokens = request.inputTokens.count

            if batch.totalTokens + prefillTokens <= maxTokensPerBatch {
                var req = waitingQueue.removeFirst()
                req.state = .prefilling
                batch.prefillRequests.append(req)
                runningRequests[req.requestId] = req
            } else {
                break
            }
        }

        return batch
    }

    /// Updates request after token generation.
    public func updateRequest(_ requestId: Int, newToken: Int, finished: Bool) {
        lock.lock()
        defer { lock.unlock() }

        guard var request = runningRequests[requestId] else { return }

        request.addToken(newToken)

        if finished || request.isComplete {
            request.state = .finished
            runningRequests.removeValue(forKey: requestId)
        } else {
            request.state = .decoding
            runningRequests[requestId] = request
        }
    }

    /// Preempts a request (for memory pressure).
    public func preemptRequest(_ requestId: Int) {
        lock.lock()
        defer { lock.unlock() }

        guard var request = runningRequests.removeValue(forKey: requestId) else {
            return
        }

        request.state = .preempted
        preemptedRequests.append(request)
    }

    /// Number of waiting requests.
    public var waitingCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return waitingQueue.count
    }

    /// Number of running requests.
    public var runningCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return runningRequests.count
    }

    private func sortWaitingQueue() {
        switch policy {
        case .fcfs:
            break  // Already in order
        case .sjf:
            waitingQueue.sort { $0.maxNewTokens < $1.maxNewTokens }
        case .priority:
            waitingQueue.sort { $0.priority < $1.priority }
        }
    }
}

// MARK: - Speculative Decoding

/// Configuration for speculative decoding.
public struct SpeculativeDecodingConfig: Sendable, Equatable {
    /// Number of tokens to speculate.
    public let numSpeculativeTokens: Int

    /// Temperature for draft model sampling.
    public let draftTemperature: Float

    /// Temperature for target model verification.
    public let targetTemperature: Float

    /// Whether to use tree-based speculation.
    public let useTreeSpeculation: Bool

    /// Maximum tree width.
    public let maxTreeWidth: Int

    public init(
        numSpeculativeTokens: Int = 4,
        draftTemperature: Float = 1.0,
        targetTemperature: Float = 1.0,
        useTreeSpeculation: Bool = false,
        maxTreeWidth: Int = 4
    ) {
        self.numSpeculativeTokens = numSpeculativeTokens
        self.draftTemperature = draftTemperature
        self.targetTemperature = targetTemperature
        self.useTreeSpeculation = useTreeSpeculation
        self.maxTreeWidth = maxTreeWidth
    }
}

/// Result of speculative decoding verification.
public struct SpeculativeResult: Sendable {
    /// Number of accepted tokens.
    public let acceptedCount: Int

    /// The accepted token IDs.
    public let acceptedTokens: [Int]

    /// The bonus token (from target model).
    public let bonusToken: Int?

    /// Acceptance rate.
    public var acceptanceRate: Double {
        guard acceptedTokens.count > 0 else { return 0 }
        return Double(acceptedCount) / Double(acceptedTokens.count)
    }

    public init(acceptedCount: Int, acceptedTokens: [Int], bonusToken: Int?) {
        self.acceptedCount = acceptedCount
        self.acceptedTokens = acceptedTokens
        self.bonusToken = bonusToken
    }
}

/// Speculative decoding engine.
public final class SpeculativeDecodingEngine: @unchecked Sendable {

    /// Statistics for speculative decoding.
    public struct Statistics: Sendable {
        public var totalSpeculations: Int = 0
        public var totalAccepted: Int = 0
        public var totalGenerated: Int = 0

        public var averageAcceptanceRate: Double {
            guard totalSpeculations > 0 else { return 0 }
            return Double(totalAccepted) / Double(totalSpeculations)
        }

        public var speedupFactor: Double {
            guard totalGenerated > 0 else { return 1.0 }
            // Speedup = tokens generated / model calls
            return Double(totalAccepted + totalSpeculations) / Double(totalSpeculations)
        }
    }

    private let config: SpeculativeDecodingConfig
    private var statistics: Statistics
    private let lock = NSLock()

    public init(config: SpeculativeDecodingConfig) {
        self.config = config
        self.statistics = Statistics()
    }

    /// Generates speculative tokens using draft model.
    ///
    /// - Parameters:
    ///   - draftLogits: Logits from draft model for each position.
    ///   - contextTokens: Current context tokens.
    /// - Returns: Speculated token IDs.
    public func speculate(
        draftLogits: [[Float]],
        contextTokens: [Int]
    ) -> [Int] {
        var speculatedTokens: [Int] = []

        for logits in draftLogits.prefix(config.numSpeculativeTokens) {
            let tokenId = sampleToken(logits, temperature: config.draftTemperature)
            speculatedTokens.append(tokenId)
        }

        return speculatedTokens
    }

    /// Verifies speculated tokens against target model.
    ///
    /// - Parameters:
    ///   - speculatedTokens: Tokens from draft model.
    ///   - targetLogits: Logits from target model for each position.
    ///   - draftProbs: Probabilities from draft model.
    /// - Returns: Verification result.
    public func verify(
        speculatedTokens: [Int],
        targetLogits: [[Float]],
        draftProbs: [[Float]]
    ) -> SpeculativeResult {
        lock.lock()
        defer { lock.unlock() }

        var acceptedTokens: [Int] = []
        var acceptedCount = 0

        for (i, token) in speculatedTokens.enumerated() {
            guard i < targetLogits.count && i < draftProbs.count else { break }

            let targetProbs = softmax(targetLogits[i], temperature: config.targetTemperature)
            let draftProbForToken = draftProbs[i][token]
            let targetProbForToken = targetProbs[token]

            // Acceptance probability: min(1, p_target / p_draft)
            let acceptProb = min(1.0, targetProbForToken / max(draftProbForToken, 1e-10))

            if Float.random(in: 0...1) < acceptProb {
                acceptedTokens.append(token)
                acceptedCount += 1
            } else {
                // Rejection - sample from adjusted distribution
                break
            }
        }

        // Sample bonus token from target model at accepted position
        let bonusToken: Int?
        if acceptedCount < targetLogits.count {
            bonusToken = sampleToken(
                targetLogits[acceptedCount],
                temperature: config.targetTemperature
            )
        } else {
            bonusToken = nil
        }

        // Update statistics
        statistics.totalSpeculations += speculatedTokens.count
        statistics.totalAccepted += acceptedCount
        statistics.totalGenerated += acceptedCount + (bonusToken != nil ? 1 : 0)

        return SpeculativeResult(
            acceptedCount: acceptedCount,
            acceptedTokens: acceptedTokens,
            bonusToken: bonusToken
        )
    }

    /// Current statistics.
    public var currentStatistics: Statistics {
        lock.lock()
        defer { lock.unlock() }
        return statistics
    }

    /// Resets statistics.
    public func resetStatistics() {
        lock.lock()
        defer { lock.unlock() }
        statistics = Statistics()
    }

    // MARK: - Private Helpers

    private func sampleToken(_ logits: [Float], temperature: Float) -> Int {
        guard !logits.isEmpty else { return 0 }

        let probs = softmax(logits, temperature: temperature)
        let rand = Float.random(in: 0...1)

        var cumulative: Float = 0
        for (i, prob) in probs.enumerated() {
            cumulative += prob
            if rand <= cumulative {
                return i
            }
        }

        return probs.count - 1
    }

    private func softmax(_ logits: [Float], temperature: Float) -> [Float] {
        guard !logits.isEmpty else { return [] }

        let scaledLogits = logits.map { $0 / max(temperature, 1e-10) }
        let maxLogit = scaledLogits.max() ?? 0

        let expLogits = scaledLogits.map { exp($0 - maxLogit) }
        let sumExp = expLogits.reduce(0, +)

        return expLogits.map { $0 / sumExp }
    }
}

// MARK: - LLM Inference Engine

/// Unified LLM inference engine combining all optimizations.
public final class LLMInferenceEngine: @unchecked Sendable {

    /// Configuration for the inference engine.
    public struct Config: Sendable {
        /// Paged attention config.
        public var pagedAttentionConfig: PagedAttentionConfig

        /// Maximum batch size.
        public var maxBatchSize: Int

        /// Maximum tokens per batch.
        public var maxTokensPerBatch: Int

        /// Scheduling policy.
        public var schedulingPolicy: ContinuousBatchScheduler.Policy

        /// Speculative decoding config (nil to disable).
        public var speculativeConfig: SpeculativeDecodingConfig?

        public init(
            pagedAttentionConfig: PagedAttentionConfig = PagedAttentionConfig(),
            maxBatchSize: Int = 64,
            maxTokensPerBatch: Int = 4096,
            schedulingPolicy: ContinuousBatchScheduler.Policy = .fcfs,
            speculativeConfig: SpeculativeDecodingConfig? = nil
        ) {
            self.pagedAttentionConfig = pagedAttentionConfig
            self.maxBatchSize = maxBatchSize
            self.maxTokensPerBatch = maxTokensPerBatch
            self.schedulingPolicy = schedulingPolicy
            self.speculativeConfig = speculativeConfig
        }
    }

    private let config: Config
    private let pagedAttention: PagedAttentionManager
    private let scheduler: ContinuousBatchScheduler
    private let speculativeEngine: SpeculativeDecodingEngine?
    private var nextRequestId: Int = 0
    private var nextSequenceId: Int = 0
    private let lock = NSLock()

    public init(config: Config) {
        self.config = config
        self.pagedAttention = PagedAttentionManager(config: config.pagedAttentionConfig)
        self.scheduler = ContinuousBatchScheduler(
            policy: config.schedulingPolicy,
            maxBatchSize: config.maxBatchSize,
            maxTokensPerBatch: config.maxTokensPerBatch
        )

        if let specConfig = config.speculativeConfig {
            self.speculativeEngine = SpeculativeDecodingEngine(config: specConfig)
        } else {
            self.speculativeEngine = nil
        }
    }

    /// Submits a new generation request.
    public func submit(inputTokens: [Int], maxNewTokens: Int = 256) -> Int {
        lock.lock()
        let requestId = nextRequestId
        nextRequestId += 1
        let sequenceId = nextSequenceId
        nextSequenceId += 1
        lock.unlock()

        // Register sequence for paged attention
        pagedAttention.registerSequence(sequenceId)
        _ = pagedAttention.appendTokens(sequenceId, numTokens: inputTokens.count)

        // Create and schedule request
        let request = BatchRequest(
            requestId: requestId,
            sequenceId: sequenceId,
            inputTokens: inputTokens,
            maxNewTokens: maxNewTokens
        )
        scheduler.addRequest(request)

        return requestId
    }

    /// Gets the next batch to process.
    public func getNextBatch() -> ContinuousBatch {
        scheduler.schedule()
    }

    /// Processes token generation result.
    public func processResult(requestId: Int, token: Int, finished: Bool) {
        lock.lock()
        defer { lock.unlock() }

        scheduler.updateRequest(requestId, newToken: token, finished: finished)
    }

    /// Frees resources for a completed request.
    public func freeRequest(sequenceId: Int) {
        pagedAttention.freeSequence(sequenceId)
    }

    /// Engine statistics.
    public var statistics: EngineStatistics {
        let memStats = pagedAttention.memoryStats
        return EngineStatistics(
            waitingRequests: scheduler.waitingCount,
            runningRequests: scheduler.runningCount,
            memoryUsed: memStats.used,
            memoryTotal: memStats.total,
            memoryUtilization: memStats.utilization,
            speculativeStats: speculativeEngine?.currentStatistics
        )
    }

    /// Engine statistics.
    public struct EngineStatistics: Sendable {
        public let waitingRequests: Int
        public let runningRequests: Int
        public let memoryUsed: Int
        public let memoryTotal: Int
        public let memoryUtilization: Double
        public let speculativeStats: SpeculativeDecodingEngine.Statistics?
    }
}
