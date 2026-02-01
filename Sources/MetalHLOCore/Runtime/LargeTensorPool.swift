// LargeTensorPool.swift
// MetalHLOCore
//
// Pool for reusing large tensor buffers across executions.

import Foundation
import Metal

/// Pool of pre-allocated large tensor buffers for reuse.
///
/// The pool organizes buffers by size buckets (powers of 2) to enable efficient
/// reuse of similarly-sized allocations. This eliminates the allocation overhead
/// for repeated inference with large tensors.
///
/// Example:
/// ```swift
/// let pool = LargeTensorPool(device: device)
///
/// // Acquire a buffer
/// let buffer = try pool.acquire(size: 1024 * 1024 * 256)
///
/// // Use buffer for computation...
///
/// // Release back to pool
/// pool.release(buffer)
/// ```
public final class LargeTensorPool: @unchecked Sendable {

    // MARK: - Configuration

    /// Configuration for the large tensor pool.
    public struct Config: Sendable {
        /// Minimum buffer size to pool (smaller buffers aren't worth pooling).
        public var minPoolSize: Int

        /// Maximum total memory the pool should hold.
        public var maxPoolMemory: Int

        /// Maximum number of buffers per size bucket.
        public var maxBuffersPerBucket: Int

        public init(
            minPoolSize: Int = 64 * 1024 * 1024,  // 64MB
            maxPoolMemory: Int = 2 * 1024 * 1024 * 1024,  // 2GB
            maxBuffersPerBucket: Int = 4
        ) {
            self.minPoolSize = minPoolSize
            self.maxPoolMemory = maxPoolMemory
            self.maxBuffersPerBucket = maxBuffersPerBucket
        }

        public static let `default` = Config()

        /// Configuration for memory-constrained environments.
        public static let conservative = Config(
            minPoolSize: 128 * 1024 * 1024,
            maxPoolMemory: 512 * 1024 * 1024,
            maxBuffersPerBucket: 2
        )

        /// Configuration for high-throughput inference.
        public static let aggressive = Config(
            minPoolSize: 32 * 1024 * 1024,
            maxPoolMemory: 4 * 1024 * 1024 * 1024,
            maxBuffersPerBucket: 8
        )
    }

    // MARK: - Properties

    private let device: MTLDevice
    private let config: Config
    private var buckets: [Int: [MTLBuffer]]  // Size bucket -> available buffers
    private var inUse: Set<ObjectIdentifier>
    private let lock = NSLock()

    /// Current total memory used by pooled buffers.
    private var currentPoolMemory: Int = 0

    /// Statistics about pool usage.
    public private(set) var stats: Stats

    // MARK: - Statistics

    public struct Stats: Sendable {
        public var hits: Int = 0
        public var misses: Int = 0
        public var allocations: Int = 0
        public var releases: Int = 0
        public var evictions: Int = 0

        public var hitRate: Double {
            let total = hits + misses
            return total > 0 ? Double(hits) / Double(total) : 0
        }
    }

    // MARK: - Initialization

    /// Creates a new large tensor pool.
    /// - Parameters:
    ///   - device: Metal device for buffer allocation.
    ///   - config: Pool configuration.
    public init(device: MTLDevice, config: Config = .default) {
        self.device = device
        self.config = config
        self.buckets = [:]
        self.inUse = []
        self.stats = Stats()
    }

    // MARK: - Public Methods

    /// Acquires a buffer of at least the specified size.
    ///
    /// The pool will:
    /// 1. Look for an available buffer in the appropriate size bucket
    /// 2. If no buffer available, allocate a new one
    /// 3. Return nil if allocation fails
    ///
    /// - Parameter size: Minimum buffer size in bytes.
    /// - Returns: A Metal buffer, or nil if allocation fails.
    public func acquire(size: Int) -> MTLBuffer? {
        let bucket = roundUpToPowerOf2(size)

        lock.lock()
        defer { lock.unlock() }

        // Try to get from pool
        if var available = buckets[bucket], !available.isEmpty {
            let buffer = available.removeLast()
            buckets[bucket] = available
            inUse.insert(ObjectIdentifier(buffer))
            currentPoolMemory -= buffer.length
            stats.hits += 1
            return buffer
        }

        stats.misses += 1

        // Need to allocate - check if we need to evict first
        if currentPoolMemory + bucket > config.maxPoolMemory {
            evictBuffers(toFree: bucket)
        }

        // Allocate new buffer
        guard let buffer = device.makeBuffer(length: bucket, options: .storageModeShared) else {
            return nil
        }

        buffer.label = "pooled_large_tensor_\(bucket)"
        inUse.insert(ObjectIdentifier(buffer))
        stats.allocations += 1

        return buffer
    }

    /// Acquires a buffer and wraps it in LargeTensorStorage.
    ///
    /// - Parameters:
    ///   - shape: Tensor shape.
    ///   - elementType: Element type.
    /// - Returns: LargeTensorStorage wrapping a pooled buffer.
    /// - Throws: `LargeTensorError.poolExhausted` if allocation fails.
    public func acquire(shape: [Int], elementType: ElementType) throws -> LargeTensorStorage {
        let count = shape.isEmpty ? 1 : shape.reduce(1, *)
        let byteSize = elementType == .int1 ? count : count * elementType.byteSize

        guard let buffer = acquire(size: byteSize) else {
            throw LargeTensorError.poolExhausted
        }

        return LargeTensorStorage(buffer: buffer, shape: shape, elementType: elementType)
    }

    /// Releases a buffer back to the pool for reuse.
    ///
    /// The buffer will be returned to its size bucket. If the pool is at capacity,
    /// older buffers may be evicted.
    ///
    /// - Parameter buffer: The buffer to release.
    public func release(_ buffer: MTLBuffer) {
        lock.lock()
        defer { lock.unlock() }

        let id = ObjectIdentifier(buffer)
        guard inUse.contains(id) else { return }

        inUse.remove(id)
        stats.releases += 1

        let bucket = roundUpToPowerOf2(buffer.length)

        // Don't pool if buffer is too small
        if buffer.length < config.minPoolSize {
            return
        }

        // Don't pool if we'd exceed memory limit
        if currentPoolMemory + buffer.length > config.maxPoolMemory {
            return
        }

        // Don't pool if bucket is full
        var available = buckets[bucket] ?? []
        if available.count >= config.maxBuffersPerBucket {
            return
        }

        available.append(buffer)
        buckets[bucket] = available
        currentPoolMemory += buffer.length
    }

    /// Releases a LargeTensorStorage back to the pool.
    public func release(_ storage: LargeTensorStorage) {
        release(storage.buffer)
    }

    /// Clears all pooled buffers.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }

        buckets.removeAll()
        currentPoolMemory = 0
    }

    /// Prewarms the pool by allocating buffers for expected sizes.
    ///
    /// - Parameter sizes: Array of expected buffer sizes.
    public func prewarm(sizes: [Int]) {
        for size in sizes {
            if let buffer = acquire(size: size) {
                release(buffer)
            }
        }
    }

    // MARK: - Statistics

    /// Returns current pool memory usage in bytes.
    public var memoryUsage: Int {
        lock.lock()
        defer { lock.unlock() }
        return currentPoolMemory
    }

    /// Returns total number of pooled buffers.
    public var bufferCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return buckets.values.reduce(0) { $0 + $1.count }
    }

    /// Returns number of buffers currently in use.
    public var inUseCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return inUse.count
    }

    // MARK: - Private Methods

    /// Rounds a size up to the nearest power of 2.
    private func roundUpToPowerOf2(_ size: Int) -> Int {
        var n = size - 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        return n + 1
    }

    /// Evicts buffers to free up memory.
    private func evictBuffers(toFree: Int) {
        // Evict oldest buffers from largest buckets first
        let sortedBuckets = buckets.keys.sorted(by: >)

        var freed = 0
        for bucket in sortedBuckets {
            guard var available = buckets[bucket], !available.isEmpty else { continue }

            while !available.isEmpty && freed < toFree {
                let buffer = available.removeFirst()
                freed += buffer.length
                currentPoolMemory -= buffer.length
                stats.evictions += 1
            }

            buckets[bucket] = available
            if freed >= toFree { break }
        }
    }
}

// MARK: - Global Pool

/// Global shared pool for large tensors.
///
/// Use this for convenience when you don't need multiple isolated pools.
public enum GlobalLargeTensorPool {
    private static nonisolated(unsafe) var _shared: LargeTensorPool?
    private static let lock = NSLock()

    /// Gets or creates the shared pool for a device.
    public static func shared(for device: MTLDevice) -> LargeTensorPool {
        lock.lock()
        defer { lock.unlock() }

        if let pool = _shared {
            return pool
        }

        let pool = LargeTensorPool(device: device)
        _shared = pool
        return pool
    }

    /// Clears the shared pool.
    public static func clear() {
        lock.lock()
        defer { lock.unlock() }
        _shared?.clear()
    }
}
