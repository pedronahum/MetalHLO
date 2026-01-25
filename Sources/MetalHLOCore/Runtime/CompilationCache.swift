// CompilationCache.swift
// MetalHLOCore
//
// Caches compiled graphs for reuse.

import Foundation

/// Thread-safe cache for compiled graphs.
///
/// `CompilationCache` stores compiled graphs by a key (typically the
/// module name or hash) to avoid recompiling identical programs.
public final class CompilationCache: @unchecked Sendable {

    // MARK: - Properties

    private var cache: [String: CompiledGraph] = [:]
    private let lock = NSLock()

    /// Maximum number of entries to cache.
    public var maxEntries: Int = 100

    // MARK: - Initialization

    public init() {}

    // MARK: - Cache Operations

    /// Gets a cached compiled graph.
    ///
    /// - Parameter key: The cache key.
    /// - Returns: The cached graph, or nil if not found.
    public func get(key: String) -> CompiledGraph? {
        lock.lock()
        defer { lock.unlock() }
        return cache[key]
    }

    /// Stores a compiled graph in the cache.
    ///
    /// - Parameters:
    ///   - key: The cache key.
    ///   - value: The compiled graph to cache.
    public func set(key: String, value: CompiledGraph) {
        lock.lock()
        defer { lock.unlock() }

        // Evict oldest entries if at capacity
        if cache.count >= maxEntries {
            // Simple eviction: remove first entry
            if let firstKey = cache.keys.first {
                cache.removeValue(forKey: firstKey)
            }
        }

        cache[key] = value
    }

    /// Removes a cached entry.
    ///
    /// - Parameter key: The cache key.
    public func remove(key: String) {
        lock.lock()
        defer { lock.unlock() }
        cache.removeValue(forKey: key)
    }

    /// Clears all cached entries.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }

    /// The number of cached entries.
    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }
}
