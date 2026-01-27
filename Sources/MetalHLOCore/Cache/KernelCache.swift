// KernelCache.swift
// Kernel Cache Infrastructure for MetalHLO
// Provides persistent kernel caching and incremental compilation

import Foundation
import Metal
import CryptoKit

// MARK: - Cache Key Generation

/// Generates unique cache keys for compiled kernels
public struct KernelCacheKey: Hashable, Sendable, Codable {
    public let sourceHash: String      // Hash of kernel source code
    public let deviceId: String        // GPU device identifier
    public let compilerVersion: String // Metal compiler version
    public let optimizationLevel: Int  // Optimization level used
    public let featureFlags: Set<String> // Feature flags enabled

    public init(
        sourceHash: String,
        deviceId: String,
        compilerVersion: String = "1.0",
        optimizationLevel: Int = 2,
        featureFlags: Set<String> = []
    ) {
        self.sourceHash = sourceHash
        self.deviceId = deviceId
        self.compilerVersion = compilerVersion
        self.optimizationLevel = optimizationLevel
        self.featureFlags = featureFlags
    }

    /// Creates a cache key from source code and device
    public static func from(
        source: String,
        device: MTLDevice,
        optimizationLevel: Int = 2,
        featureFlags: Set<String> = []
    ) -> KernelCacheKey {
        let sourceHash = hashSource(source)
        let deviceId = extractDeviceId(device)

        return KernelCacheKey(
            sourceHash: sourceHash,
            deviceId: deviceId,
            compilerVersion: "1.0",
            optimizationLevel: optimizationLevel,
            featureFlags: featureFlags
        )
    }

    /// Hashes source code using SHA256
    private static func hashSource(_ source: String) -> String {
        let data = Data(source.utf8)
        let hash = SHA256.hash(data: data)
        return hash.map { String(format: "%02x", $0) }.joined()
    }

    /// Extracts device identifier
    private static func extractDeviceId(_ device: MTLDevice) -> String {
        // Use device name and registry ID for uniqueness
        return "\(device.name)_\(device.registryID)"
    }

    /// String representation for file naming
    public var fileName: String {
        let prefix = String(sourceHash.prefix(16))
        let devicePrefix = String(deviceId.prefix(8).replacingOccurrences(of: " ", with: "_"))
        return "\(prefix)_\(devicePrefix)_opt\(optimizationLevel).metallib"
    }
}

// MARK: - Cached Kernel Entry

/// Metadata for a cached kernel
public struct CachedKernelEntry: Sendable, Codable {
    public let key: KernelCacheKey
    public let createdAt: Date
    public let lastAccessedAt: Date
    public let accessCount: Int
    public let sizeBytes: Int
    public let compilationTimeMs: Double
    public let kernelFunctions: [String]

    public init(
        key: KernelCacheKey,
        createdAt: Date = Date(),
        lastAccessedAt: Date = Date(),
        accessCount: Int = 1,
        sizeBytes: Int = 0,
        compilationTimeMs: Double = 0,
        kernelFunctions: [String] = []
    ) {
        self.key = key
        self.createdAt = createdAt
        self.lastAccessedAt = lastAccessedAt
        self.accessCount = accessCount
        self.sizeBytes = sizeBytes
        self.compilationTimeMs = compilationTimeMs
        self.kernelFunctions = kernelFunctions
    }

    /// Creates updated entry with new access time
    public func accessed() -> CachedKernelEntry {
        CachedKernelEntry(
            key: key,
            createdAt: createdAt,
            lastAccessedAt: Date(),
            accessCount: accessCount + 1,
            sizeBytes: sizeBytes,
            compilationTimeMs: compilationTimeMs,
            kernelFunctions: kernelFunctions
        )
    }
}

// MARK: - Cache Statistics

/// Statistics about cache usage
public struct CacheStatistics: Sendable {
    public var hitCount: Int
    public var missCount: Int
    public var totalEntries: Int
    public var totalSizeBytes: Int
    public var compilationTimeSavedMs: Double

    public var hitRate: Double {
        let total = hitCount + missCount
        guard total > 0 else { return 0 }
        return Double(hitCount) / Double(total)
    }

    public init() {
        self.hitCount = 0
        self.missCount = 0
        self.totalEntries = 0
        self.totalSizeBytes = 0
        self.compilationTimeSavedMs = 0
    }
}

// MARK: - Persistent Kernel Cache

/// Persistent cache for compiled Metal kernels
public final class PersistentKernelCache: @unchecked Sendable {

    /// Cache configuration
    public struct Config: Sendable {
        public var cacheDirectory: URL
        public var maxCacheSizeBytes: Int
        public var maxEntries: Int
        public var evictionPolicy: EvictionPolicy
        public var enableCompression: Bool

        public enum EvictionPolicy: String, Sendable {
            case lru    // Least recently used
            case lfu    // Least frequently used
            case fifo   // First in first out
            case size   // Largest first
        }

        public static let `default` = Config(
            cacheDirectory: FileManager.default.urls(
                for: .cachesDirectory,
                in: .userDomainMask
            ).first!.appendingPathComponent("MetalHLO/KernelCache"),
            maxCacheSizeBytes: 500_000_000, // 500 MB
            maxEntries: 1000,
            evictionPolicy: .lru,
            enableCompression: true
        )

        public init(
            cacheDirectory: URL,
            maxCacheSizeBytes: Int,
            maxEntries: Int,
            evictionPolicy: EvictionPolicy,
            enableCompression: Bool
        ) {
            self.cacheDirectory = cacheDirectory
            self.maxCacheSizeBytes = maxCacheSizeBytes
            self.maxEntries = maxEntries
            self.evictionPolicy = evictionPolicy
            self.enableCompression = enableCompression
        }
    }

    private let config: Config
    private var entries: [KernelCacheKey: CachedKernelEntry]
    private var memoryCache: [KernelCacheKey: Data] // In-memory cache for hot kernels
    private var stats: CacheStatistics
    private let lock = NSLock()
    private let fileManager = FileManager.default

    public init(config: Config = .default) {
        self.config = config
        self.entries = [:]
        self.memoryCache = [:]
        self.stats = CacheStatistics()

        // Create cache directory if needed
        try? fileManager.createDirectory(
            at: config.cacheDirectory,
            withIntermediateDirectories: true
        )

        // Load existing cache index
        loadIndex()
    }

    /// Gets a cached kernel library
    public func get(_ key: KernelCacheKey, device: MTLDevice) -> MTLLibrary? {
        lock.lock()
        defer { lock.unlock() }

        // Check if entry exists
        guard var entry = entries[key] else {
            stats.missCount += 1
            return nil
        }

        // Try memory cache first
        if let data = memoryCache[key] {
            stats.hitCount += 1
            stats.compilationTimeSavedMs += entry.compilationTimeMs

            // Update access info
            entry = entry.accessed()
            entries[key] = entry

            return createLibrary(from: data, device: device)
        }

        // Load from disk
        let filePath = config.cacheDirectory.appendingPathComponent(key.fileName)
        guard let data = try? Data(contentsOf: filePath) else {
            // File missing, remove entry
            entries.removeValue(forKey: key)
            stats.missCount += 1
            return nil
        }

        stats.hitCount += 1
        stats.compilationTimeSavedMs += entry.compilationTimeMs

        // Update access info
        entry = entry.accessed()
        entries[key] = entry

        // Add to memory cache for hot kernels
        if entry.accessCount > 3 {
            memoryCache[key] = data
        }

        return createLibrary(from: data, device: device)
    }

    /// Stores a compiled kernel library
    public func put(
        _ key: KernelCacheKey,
        library: MTLLibrary,
        compilationTimeMs: Double,
        functionNames: [String]
    ) throws {
        // Serialize library to data
        guard let data = serializeLibrary(library) else {
            throw CacheError.serializationFailed
        }

        lock.lock()
        defer { lock.unlock() }

        // Check if we need to evict
        while stats.totalEntries >= config.maxEntries ||
              stats.totalSizeBytes + data.count > config.maxCacheSizeBytes {
            evictOne()
        }

        // Write to disk
        let filePath = config.cacheDirectory.appendingPathComponent(key.fileName)
        try data.write(to: filePath)

        // Create entry
        let entry = CachedKernelEntry(
            key: key,
            sizeBytes: data.count,
            compilationTimeMs: compilationTimeMs,
            kernelFunctions: functionNames
        )

        entries[key] = entry
        stats.totalEntries += 1
        stats.totalSizeBytes += data.count

        // Save index
        saveIndex()
    }

    /// Checks if a kernel is cached
    public func contains(_ key: KernelCacheKey) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return entries[key] != nil
    }

    /// Removes a cached kernel
    public func remove(_ key: KernelCacheKey) {
        lock.lock()
        defer { lock.unlock() }

        guard let entry = entries.removeValue(forKey: key) else { return }

        memoryCache.removeValue(forKey: key)

        let filePath = config.cacheDirectory.appendingPathComponent(key.fileName)
        try? fileManager.removeItem(at: filePath)

        stats.totalEntries -= 1
        stats.totalSizeBytes -= entry.sizeBytes
    }

    /// Clears the entire cache
    public func clear() {
        lock.lock()
        defer { lock.unlock() }

        entries.removeAll()
        memoryCache.removeAll()
        stats = CacheStatistics()

        // Remove all files
        try? fileManager.removeItem(at: config.cacheDirectory)
        try? fileManager.createDirectory(
            at: config.cacheDirectory,
            withIntermediateDirectories: true
        )
    }

    /// Returns cache statistics
    public func getStatistics() -> CacheStatistics {
        lock.lock()
        defer { lock.unlock() }
        return stats
    }

    /// Prewarms the cache by loading entries into memory
    public func prewarm(keys: [KernelCacheKey]) {
        lock.lock()
        defer { lock.unlock() }

        for key in keys {
            guard entries[key] != nil else { continue }
            guard memoryCache[key] == nil else { continue }

            let filePath = config.cacheDirectory.appendingPathComponent(key.fileName)
            if let data = try? Data(contentsOf: filePath) {
                memoryCache[key] = data
            }
        }
    }

    // MARK: - Private Methods

    private func createLibrary(from data: Data, device: MTLDevice) -> MTLLibrary? {
        // Create dispatch data from Data
        return data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) -> MTLLibrary? in
            guard let baseAddress = bytes.baseAddress else { return nil }
            let dispatchData = DispatchData(
                bytes: UnsafeBufferPointer(
                    start: baseAddress.assumingMemoryBound(to: UInt8.self),
                    count: data.count
                )
            )
            return try? device.makeLibrary(data: dispatchData as __DispatchData)
        }
    }

    private func serializeLibrary(_ library: MTLLibrary) -> Data? {
        // Note: In production, this would use Metal binary archives
        // For now, we'll need the source code path
        // Metal libraries compiled at runtime need to be stored via source + options
        return nil
    }

    private func evictOne() {
        guard !entries.isEmpty else { return }

        let keyToEvict: KernelCacheKey

        switch config.evictionPolicy {
        case .lru:
            keyToEvict = entries.min { $0.value.lastAccessedAt < $1.value.lastAccessedAt }!.key
        case .lfu:
            keyToEvict = entries.min { $0.value.accessCount < $1.value.accessCount }!.key
        case .fifo:
            keyToEvict = entries.min { $0.value.createdAt < $1.value.createdAt }!.key
        case .size:
            keyToEvict = entries.max { $0.value.sizeBytes < $1.value.sizeBytes }!.key
        }

        if let entry = entries.removeValue(forKey: keyToEvict) {
            memoryCache.removeValue(forKey: keyToEvict)

            let filePath = config.cacheDirectory.appendingPathComponent(keyToEvict.fileName)
            try? fileManager.removeItem(at: filePath)

            stats.totalEntries -= 1
            stats.totalSizeBytes -= entry.sizeBytes
        }
    }

    private func loadIndex() {
        let indexPath = config.cacheDirectory.appendingPathComponent("index.json")
        guard let data = try? Data(contentsOf: indexPath),
              let loadedEntries = try? JSONDecoder().decode(
                [KernelCacheKey: CachedKernelEntry].self,
                from: data
              ) else {
            return
        }

        entries = loadedEntries
        stats.totalEntries = entries.count
        stats.totalSizeBytes = entries.values.reduce(0) { $0 + $1.sizeBytes }
    }

    private func saveIndex() {
        let indexPath = config.cacheDirectory.appendingPathComponent("index.json")
        if let data = try? JSONEncoder().encode(entries) {
            try? data.write(to: indexPath)
        }
    }
}

// MARK: - Incremental Compiler

/// Supports incremental compilation by tracking source dependencies
public final class IncrementalCompiler: @unchecked Sendable {

    /// Compilation unit tracking
    public struct CompilationUnit: Sendable, Codable {
        public let id: String
        public let sourceHash: String
        public let dependencies: Set<String>
        public let lastCompiled: Date
        public var isValid: Bool

        public init(
            id: String,
            sourceHash: String,
            dependencies: Set<String> = [],
            lastCompiled: Date = Date(),
            isValid: Bool = true
        ) {
            self.id = id
            self.sourceHash = sourceHash
            self.dependencies = dependencies
            self.lastCompiled = lastCompiled
            self.isValid = isValid
        }
    }

    /// Compilation result
    public struct CompilationResult: Sendable {
        public let unit: CompilationUnit
        public let wasRecompiled: Bool
        public let compilationTimeMs: Double
        public let library: MTLLibrary?
        public let error: String?
    }

    private var units: [String: CompilationUnit]
    private let cache: PersistentKernelCache
    private let lock = NSLock()

    public init(cache: PersistentKernelCache) {
        self.units = [:]
        self.cache = cache
    }

    /// Compiles a unit incrementally
    public func compile(
        id: String,
        source: String,
        dependencies: Set<String>,
        device: MTLDevice,
        options: MTLCompileOptions? = nil
    ) -> CompilationResult {
        let sourceHash = hashSource(source)

        lock.lock()
        let existingUnit = units[id]
        lock.unlock()

        // Check if recompilation is needed
        let needsRecompilation = existingUnit == nil ||
                                  existingUnit!.sourceHash != sourceHash ||
                                  !existingUnit!.isValid ||
                                  hasDependencyChanged(dependencies, existingUnit: existingUnit)

        if !needsRecompilation, let unit = existingUnit {
            // Try to get from cache
            let cacheKey = KernelCacheKey(
                sourceHash: sourceHash,
                deviceId: "\(device.name)_\(device.registryID)"
            )

            if let library = cache.get(cacheKey, device: device) {
                return CompilationResult(
                    unit: unit,
                    wasRecompiled: false,
                    compilationTimeMs: 0,
                    library: library,
                    error: nil
                )
            }
        }

        // Compile
        let startTime = DispatchTime.now()
        let compileOptions = options ?? MTLCompileOptions()

        do {
            let library = try device.makeLibrary(source: source, options: compileOptions)
            let endTime = DispatchTime.now()
            let compilationTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0

            let newUnit = CompilationUnit(
                id: id,
                sourceHash: sourceHash,
                dependencies: dependencies
            )

            lock.lock()
            units[id] = newUnit
            lock.unlock()

            // Cache the result
            let cacheKey = KernelCacheKey(
                sourceHash: sourceHash,
                deviceId: "\(device.name)_\(device.registryID)"
            )

            let functionNames = library.functionNames
            try? cache.put(
                cacheKey,
                library: library,
                compilationTimeMs: compilationTime,
                functionNames: functionNames
            )

            return CompilationResult(
                unit: newUnit,
                wasRecompiled: true,
                compilationTimeMs: compilationTime,
                library: library,
                error: nil
            )

        } catch {
            let endTime = DispatchTime.now()
            let compilationTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0

            // Mark unit as invalid
            if var unit = existingUnit {
                unit = CompilationUnit(
                    id: unit.id,
                    sourceHash: unit.sourceHash,
                    dependencies: unit.dependencies,
                    lastCompiled: unit.lastCompiled,
                    isValid: false
                )
                lock.lock()
                units[id] = unit
                lock.unlock()
            }

            return CompilationResult(
                unit: existingUnit ?? CompilationUnit(
                    id: id,
                    sourceHash: sourceHash,
                    dependencies: dependencies,
                    isValid: false
                ),
                wasRecompiled: true,
                compilationTimeMs: compilationTime,
                library: nil,
                error: error.localizedDescription
            )
        }
    }

    /// Invalidates a compilation unit and its dependents
    public func invalidate(_ id: String) {
        lock.lock()
        defer { lock.unlock() }

        guard var unit = units[id] else { return }

        unit = CompilationUnit(
            id: unit.id,
            sourceHash: unit.sourceHash,
            dependencies: unit.dependencies,
            lastCompiled: unit.lastCompiled,
            isValid: false
        )
        units[id] = unit

        // Invalidate dependents
        for (otherId, otherUnit) in units {
            if otherUnit.dependencies.contains(id) {
                invalidateWithoutLock(otherId)
            }
        }
    }

    private func invalidateWithoutLock(_ id: String) {
        guard var unit = units[id] else { return }

        unit = CompilationUnit(
            id: unit.id,
            sourceHash: unit.sourceHash,
            dependencies: unit.dependencies,
            lastCompiled: unit.lastCompiled,
            isValid: false
        )
        units[id] = unit

        // Invalidate dependents recursively
        for (otherId, otherUnit) in units {
            if otherUnit.dependencies.contains(id) && otherUnit.isValid {
                invalidateWithoutLock(otherId)
            }
        }
    }

    /// Gets compilation units that need recompilation
    public func getInvalidUnits() -> [CompilationUnit] {
        lock.lock()
        defer { lock.unlock() }
        return units.values.filter { !$0.isValid }
    }

    /// Checks if any dependency has changed
    private func hasDependencyChanged(_ dependencies: Set<String>, existingUnit: CompilationUnit?) -> Bool {
        guard let unit = existingUnit else { return true }

        // Check if dependency set changed
        if unit.dependencies != dependencies {
            return true
        }

        // Check if any dependency is invalid
        lock.lock()
        defer { lock.unlock() }

        for depId in dependencies {
            if let depUnit = units[depId], !depUnit.isValid {
                return true
            }
        }

        return false
    }

    private func hashSource(_ source: String) -> String {
        let data = Data(source.utf8)
        let hash = SHA256.hash(data: data)
        return hash.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Disk Cache Manager

/// Manages disk caching for compiled Metal libraries
public final class DiskCacheManager: @unchecked Sendable {

    /// Configuration
    public struct Config: Sendable {
        public var basePath: URL
        public var maxDiskUsageBytes: Int
        public var cleanupThreshold: Double // 0.0 to 1.0, cleanup when usage exceeds this
        public var enableBackgroundCleanup: Bool

        public static let `default` = Config(
            basePath: FileManager.default.urls(
                for: .cachesDirectory,
                in: .userDomainMask
            ).first!.appendingPathComponent("MetalHLO"),
            maxDiskUsageBytes: 1_000_000_000, // 1 GB
            cleanupThreshold: 0.9,
            enableBackgroundCleanup: true
        )

        public init(
            basePath: URL,
            maxDiskUsageBytes: Int,
            cleanupThreshold: Double,
            enableBackgroundCleanup: Bool
        ) {
            self.basePath = basePath
            self.maxDiskUsageBytes = maxDiskUsageBytes
            self.cleanupThreshold = cleanupThreshold
            self.enableBackgroundCleanup = enableBackgroundCleanup
        }
    }

    /// Cache namespace for organizing cached items
    public struct CacheNamespace: Sendable {
        public let name: String
        public let version: Int

        public var path: String { "\(name)/v\(version)" }

        public init(name: String, version: Int = 1) {
            self.name = name
            self.version = version
        }
    }

    private let config: Config
    private let fileManager = FileManager.default
    private let lock = NSLock()
    private var namespaces: Set<String>

    public init(config: Config = .default) {
        self.config = config
        self.namespaces = []

        // Create base directory
        try? fileManager.createDirectory(
            at: config.basePath,
            withIntermediateDirectories: true
        )

        // Start background cleanup if enabled
        if config.enableBackgroundCleanup {
            scheduleBackgroundCleanup()
        }
    }

    /// Gets the path for a cached item
    public func cachePath(for key: String, in namespace: CacheNamespace) -> URL {
        let namespacePath = config.basePath.appendingPathComponent(namespace.path)

        lock.lock()
        if !namespaces.contains(namespace.path) {
            namespaces.insert(namespace.path)
            try? fileManager.createDirectory(
                at: namespacePath,
                withIntermediateDirectories: true
            )
        }
        lock.unlock()

        return namespacePath.appendingPathComponent(key)
    }

    /// Writes data to cache
    public func write(_ data: Data, for key: String, in namespace: CacheNamespace) throws {
        let path = cachePath(for: key, in: namespace)

        // Check disk usage
        let currentUsage = calculateDiskUsage()
        if currentUsage + data.count > config.maxDiskUsageBytes {
            performCleanup(targetBytes: data.count)
        }

        try data.write(to: path)

        // Set extended attributes for metadata
        try? setCreationDate(path, date: Date())
    }

    /// Reads data from cache
    public func read(for key: String, in namespace: CacheNamespace) -> Data? {
        let path = cachePath(for: key, in: namespace)

        guard let data = try? Data(contentsOf: path) else {
            return nil
        }

        // Update access time
        try? updateAccessTime(path)

        return data
    }

    /// Checks if item exists in cache
    public func exists(key: String, in namespace: CacheNamespace) -> Bool {
        let path = cachePath(for: key, in: namespace)
        return fileManager.fileExists(atPath: path.path)
    }

    /// Removes an item from cache
    public func remove(key: String, in namespace: CacheNamespace) {
        let path = cachePath(for: key, in: namespace)
        try? fileManager.removeItem(at: path)
    }

    /// Clears a namespace
    public func clearNamespace(_ namespace: CacheNamespace) {
        let path = config.basePath.appendingPathComponent(namespace.path)
        try? fileManager.removeItem(at: path)

        lock.lock()
        namespaces.remove(namespace.path)
        lock.unlock()
    }

    /// Clears all cached data
    public func clearAll() {
        try? fileManager.removeItem(at: config.basePath)
        try? fileManager.createDirectory(
            at: config.basePath,
            withIntermediateDirectories: true
        )

        lock.lock()
        namespaces.removeAll()
        lock.unlock()
    }

    /// Returns current disk usage in bytes
    public func calculateDiskUsage() -> Int {
        guard let enumerator = fileManager.enumerator(
            at: config.basePath,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var totalSize = 0

        while let url = enumerator.nextObject() as? URL {
            if let resourceValues = try? url.resourceValues(forKeys: [.fileSizeKey]),
               let fileSize = resourceValues.fileSize {
                totalSize += fileSize
            }
        }

        return totalSize
    }

    /// Returns disk usage statistics
    public func getDiskStats() -> DiskStats {
        let usage = calculateDiskUsage()
        return DiskStats(
            usedBytes: usage,
            maxBytes: config.maxDiskUsageBytes,
            usagePercentage: Double(usage) / Double(config.maxDiskUsageBytes)
        )
    }

    // MARK: - Private Methods

    private func performCleanup(targetBytes: Int) {
        guard let enumerator = fileManager.enumerator(
            at: config.basePath,
            includingPropertiesForKeys: [.fileSizeKey, .contentAccessDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return
        }

        // Collect files with metadata
        var files: [(url: URL, size: Int, accessed: Date)] = []

        while let url = enumerator.nextObject() as? URL {
            if let resourceValues = try? url.resourceValues(
                forKeys: [.fileSizeKey, .contentAccessDateKey]
            ),
               let fileSize = resourceValues.fileSize,
               let accessDate = resourceValues.contentAccessDate {
                files.append((url, fileSize, accessDate))
            }
        }

        // Sort by access date (oldest first)
        files.sort { $0.accessed < $1.accessed }

        // Remove files until we have enough space
        var freedBytes = 0
        for file in files {
            if freedBytes >= targetBytes {
                break
            }

            try? fileManager.removeItem(at: file.url)
            freedBytes += file.size
        }
    }

    private func scheduleBackgroundCleanup() {
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 60) { [weak self] in
            self?.backgroundCleanupTask()
        }
    }

    private func backgroundCleanupTask() {
        let stats = getDiskStats()

        if stats.usagePercentage > config.cleanupThreshold {
            let targetBytes = Int(Double(stats.usedBytes) * 0.2) // Free 20%
            performCleanup(targetBytes: targetBytes)
        }

        // Reschedule
        if config.enableBackgroundCleanup {
            scheduleBackgroundCleanup()
        }
    }

    private func setCreationDate(_ url: URL, date: Date) throws {
        try fileManager.setAttributes(
            [.creationDate: date],
            ofItemAtPath: url.path
        )
    }

    private func updateAccessTime(_ url: URL) throws {
        try fileManager.setAttributes(
            [.modificationDate: Date()],
            ofItemAtPath: url.path
        )
    }
}

/// Disk usage statistics
public struct DiskStats: Sendable {
    public let usedBytes: Int
    public let maxBytes: Int
    public let usagePercentage: Double
}

// MARK: - Errors

/// Cache-related errors
public enum CacheError: Error, Sendable {
    case serializationFailed
    case deserializationFailed
    case diskFull
    case ioError(String)
    case invalidKey
    case entryNotFound
}

// MARK: - Source Cache

/// Caches source code transformations
public final class SourceCache: @unchecked Sendable {

    private var cache: [String: (source: String, timestamp: Date)]
    private let maxEntries: Int
    private let lock = NSLock()

    public init(maxEntries: Int = 100) {
        self.cache = [:]
        self.maxEntries = maxEntries
    }

    /// Gets cached source for a key
    public func get(_ key: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return cache[key]?.source
    }

    /// Caches source with a key
    public func put(_ key: String, source: String) {
        lock.lock()
        defer { lock.unlock() }

        // Evict if needed
        if cache.count >= maxEntries {
            let oldest = cache.min { $0.value.timestamp < $1.value.timestamp }
            if let oldestKey = oldest?.key {
                cache.removeValue(forKey: oldestKey)
            }
        }

        cache[key] = (source, Date())
    }

    /// Checks if source is cached
    public func contains(_ key: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cache[key] != nil
    }

    /// Clears the source cache
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }
}

// MARK: - Pipeline State Cache

/// Caches Metal pipeline states for fast access
public final class PipelineStateCache: @unchecked Sendable {

    private var computePipelines: [String: MTLComputePipelineState]
    private var renderPipelines: [String: MTLRenderPipelineState]
    private let lock = NSLock()

    public init() {
        self.computePipelines = [:]
        self.renderPipelines = [:]
    }

    /// Gets a cached compute pipeline state
    public func getComputePipeline(_ key: String) -> MTLComputePipelineState? {
        lock.lock()
        defer { lock.unlock() }
        return computePipelines[key]
    }

    /// Caches a compute pipeline state
    public func putComputePipeline(_ key: String, state: MTLComputePipelineState) {
        lock.lock()
        defer { lock.unlock() }
        computePipelines[key] = state
    }

    /// Gets a cached render pipeline state
    public func getRenderPipeline(_ key: String) -> MTLRenderPipelineState? {
        lock.lock()
        defer { lock.unlock() }
        return renderPipelines[key]
    }

    /// Caches a render pipeline state
    public func putRenderPipeline(_ key: String, state: MTLRenderPipelineState) {
        lock.lock()
        defer { lock.unlock() }
        renderPipelines[key] = state
    }

    /// Removes a pipeline state
    public func remove(_ key: String) {
        lock.lock()
        defer { lock.unlock() }
        computePipelines.removeValue(forKey: key)
        renderPipelines.removeValue(forKey: key)
    }

    /// Clears all cached pipeline states
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        computePipelines.removeAll()
        renderPipelines.removeAll()
    }

    /// Returns statistics about cached pipelines
    public func getStats() -> (compute: Int, render: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (computePipelines.count, renderPipelines.count)
    }
}
