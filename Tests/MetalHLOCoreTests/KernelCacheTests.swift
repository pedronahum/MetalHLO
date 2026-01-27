// KernelCacheTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2K: Kernel Cache

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("Kernel Cache Tests")
struct KernelCacheTests {

    // MARK: - KernelCacheKey Tests

    @Test("KernelCacheKey creation")
    func kernelCacheKeyCreation() {
        let key = KernelCacheKey(
            sourceHash: "abc123",
            deviceId: "M1_12345",
            compilerVersion: "1.0",
            optimizationLevel: 2,
            featureFlags: ["fast_math"]
        )

        #expect(key.sourceHash == "abc123")
        #expect(key.deviceId == "M1_12345")
        #expect(key.compilerVersion == "1.0")
        #expect(key.optimizationLevel == 2)
        #expect(key.featureFlags.contains("fast_math"))
    }

    @Test("KernelCacheKey fileName generation")
    func kernelCacheKeyFileName() {
        let key = KernelCacheKey(
            sourceHash: "abcdef1234567890abcdef",
            deviceId: "TestDevice123",
            optimizationLevel: 2
        )

        let fileName = key.fileName
        #expect(fileName.contains("abcdef1234567890"))
        #expect(fileName.contains("opt2"))
        #expect(fileName.hasSuffix(".metallib"))
    }

    @Test("KernelCacheKey hashability")
    func kernelCacheKeyHashable() {
        let key1 = KernelCacheKey(
            sourceHash: "hash1",
            deviceId: "device1"
        )
        let key2 = KernelCacheKey(
            sourceHash: "hash1",
            deviceId: "device1"
        )
        let key3 = KernelCacheKey(
            sourceHash: "hash2",
            deviceId: "device1"
        )

        #expect(key1 == key2)
        #expect(key1 != key3)

        var dict: [KernelCacheKey: Int] = [:]
        dict[key1] = 1
        dict[key2] = 2
        #expect(dict[key1] == 2) // key2 overwrites key1
    }

    @Test("KernelCacheKey Codable")
    func kernelCacheKeyCodable() throws {
        let key = KernelCacheKey(
            sourceHash: "testhash",
            deviceId: "testdevice",
            compilerVersion: "2.0",
            optimizationLevel: 3,
            featureFlags: ["flag1", "flag2"]
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(key)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(KernelCacheKey.self, from: data)

        #expect(decoded.sourceHash == key.sourceHash)
        #expect(decoded.deviceId == key.deviceId)
        #expect(decoded.compilerVersion == key.compilerVersion)
        #expect(decoded.optimizationLevel == key.optimizationLevel)
        #expect(decoded.featureFlags == key.featureFlags)
    }

    // MARK: - CachedKernelEntry Tests

    @Test("CachedKernelEntry creation")
    func cachedKernelEntryCreation() {
        let key = KernelCacheKey(sourceHash: "hash", deviceId: "device")
        let entry = CachedKernelEntry(
            key: key,
            sizeBytes: 1024,
            compilationTimeMs: 50.5,
            kernelFunctions: ["func1", "func2"]
        )

        #expect(entry.key == key)
        #expect(entry.sizeBytes == 1024)
        #expect(entry.compilationTimeMs == 50.5)
        #expect(entry.kernelFunctions.count == 2)
        #expect(entry.accessCount == 1)
    }

    @Test("CachedKernelEntry accessed updates")
    func cachedKernelEntryAccessed() {
        let key = KernelCacheKey(sourceHash: "hash", deviceId: "device")
        let entry = CachedKernelEntry(key: key, accessCount: 5)

        let updated = entry.accessed()

        #expect(updated.accessCount == 6)
        #expect(updated.lastAccessedAt >= entry.lastAccessedAt)
        #expect(updated.createdAt == entry.createdAt)
    }

    // MARK: - CacheStatistics Tests

    @Test("CacheStatistics initial values")
    func cacheStatisticsInitial() {
        let stats = CacheStatistics()

        #expect(stats.hitCount == 0)
        #expect(stats.missCount == 0)
        #expect(stats.totalEntries == 0)
        #expect(stats.totalSizeBytes == 0)
        #expect(stats.compilationTimeSavedMs == 0)
        #expect(stats.hitRate == 0)
    }

    @Test("CacheStatistics hit rate calculation")
    func cacheStatisticsHitRate() {
        var stats = CacheStatistics()
        stats.hitCount = 80
        stats.missCount = 20

        #expect(stats.hitRate == 0.8)
    }

    // MARK: - PersistentKernelCache Config Tests

    @Test("PersistentKernelCache default config")
    func persistentKernelCacheDefaultConfig() {
        let config = PersistentKernelCache.Config.default

        #expect(config.maxCacheSizeBytes == 500_000_000)
        #expect(config.maxEntries == 1000)
        #expect(config.evictionPolicy == .lru)
        #expect(config.enableCompression)
    }

    @Test("PersistentKernelCache eviction policies")
    func evictionPolicies() {
        #expect(PersistentKernelCache.Config.EvictionPolicy.lru.rawValue == "lru")
        #expect(PersistentKernelCache.Config.EvictionPolicy.lfu.rawValue == "lfu")
        #expect(PersistentKernelCache.Config.EvictionPolicy.fifo.rawValue == "fifo")
        #expect(PersistentKernelCache.Config.EvictionPolicy.size.rawValue == "size")
    }

    @Test("PersistentKernelCache contains")
    func persistentKernelCacheContains() {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = PersistentKernelCache.Config(
            cacheDirectory: tempDir,
            maxCacheSizeBytes: 10_000_000,
            maxEntries: 100,
            evictionPolicy: .lru,
            enableCompression: false
        )

        let cache = PersistentKernelCache(config: config)
        let key = KernelCacheKey(sourceHash: "test", deviceId: "device")

        #expect(!cache.contains(key))

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("PersistentKernelCache statistics")
    func persistentKernelCacheStatistics() {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = PersistentKernelCache.Config(
            cacheDirectory: tempDir,
            maxCacheSizeBytes: 10_000_000,
            maxEntries: 100,
            evictionPolicy: .lru,
            enableCompression: false
        )

        let cache = PersistentKernelCache(config: config)
        let stats = cache.getStatistics()

        #expect(stats.hitCount == 0)
        #expect(stats.missCount == 0)

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("PersistentKernelCache clear")
    func persistentKernelCacheClear() {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = PersistentKernelCache.Config(
            cacheDirectory: tempDir,
            maxCacheSizeBytes: 10_000_000,
            maxEntries: 100,
            evictionPolicy: .lru,
            enableCompression: false
        )

        let cache = PersistentKernelCache(config: config)
        cache.clear()

        let stats = cache.getStatistics()
        #expect(stats.totalEntries == 0)

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - IncrementalCompiler Tests

    @Test("IncrementalCompiler CompilationUnit creation")
    func compilationUnitCreation() {
        let unit = IncrementalCompiler.CompilationUnit(
            id: "kernel1",
            sourceHash: "hash123",
            dependencies: ["dep1", "dep2"],
            isValid: true
        )

        #expect(unit.id == "kernel1")
        #expect(unit.sourceHash == "hash123")
        #expect(unit.dependencies.count == 2)
        #expect(unit.isValid)
    }

    @Test("IncrementalCompiler getInvalidUnits empty")
    func incrementalCompilerInvalidUnitsEmpty() {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let cacheConfig = PersistentKernelCache.Config(
            cacheDirectory: tempDir,
            maxCacheSizeBytes: 10_000_000,
            maxEntries: 100,
            evictionPolicy: .lru,
            enableCompression: false
        )
        let cache = PersistentKernelCache(config: cacheConfig)
        let compiler = IncrementalCompiler(cache: cache)

        let invalidUnits = compiler.getInvalidUnits()
        #expect(invalidUnits.isEmpty)

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - DiskCacheManager Tests

    @Test("DiskCacheManager default config")
    func diskCacheManagerDefaultConfig() {
        let config = DiskCacheManager.Config.default

        #expect(config.maxDiskUsageBytes == 1_000_000_000)
        #expect(config.cleanupThreshold == 0.9)
        #expect(config.enableBackgroundCleanup)
    }

    @Test("CacheNamespace creation")
    func cacheNamespaceCreation() {
        let namespace = DiskCacheManager.CacheNamespace(name: "kernels", version: 2)

        #expect(namespace.name == "kernels")
        #expect(namespace.version == 2)
        #expect(namespace.path == "kernels/v2")
    }

    @Test("DiskCacheManager write and read")
    func diskCacheManagerWriteRead() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = DiskCacheManager.Config(
            basePath: tempDir,
            maxDiskUsageBytes: 10_000_000,
            cleanupThreshold: 0.9,
            enableBackgroundCleanup: false
        )

        let manager = DiskCacheManager(config: config)
        let namespace = DiskCacheManager.CacheNamespace(name: "test", version: 1)

        let testData = "Hello, cache!".data(using: .utf8)!
        try manager.write(testData, for: "testkey", in: namespace)

        let readData = manager.read(for: "testkey", in: namespace)
        #expect(readData == testData)

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("DiskCacheManager exists")
    func diskCacheManagerExists() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = DiskCacheManager.Config(
            basePath: tempDir,
            maxDiskUsageBytes: 10_000_000,
            cleanupThreshold: 0.9,
            enableBackgroundCleanup: false
        )

        let manager = DiskCacheManager(config: config)
        let namespace = DiskCacheManager.CacheNamespace(name: "test", version: 1)

        #expect(!manager.exists(key: "testkey", in: namespace))

        try manager.write("data".data(using: .utf8)!, for: "testkey", in: namespace)

        #expect(manager.exists(key: "testkey", in: namespace))

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("DiskCacheManager remove")
    func diskCacheManagerRemove() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = DiskCacheManager.Config(
            basePath: tempDir,
            maxDiskUsageBytes: 10_000_000,
            cleanupThreshold: 0.9,
            enableBackgroundCleanup: false
        )

        let manager = DiskCacheManager(config: config)
        let namespace = DiskCacheManager.CacheNamespace(name: "test", version: 1)

        try manager.write("data".data(using: .utf8)!, for: "testkey", in: namespace)
        #expect(manager.exists(key: "testkey", in: namespace))

        manager.remove(key: "testkey", in: namespace)
        #expect(!manager.exists(key: "testkey", in: namespace))

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("DiskCacheManager clearNamespace")
    func diskCacheManagerClearNamespace() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = DiskCacheManager.Config(
            basePath: tempDir,
            maxDiskUsageBytes: 10_000_000,
            cleanupThreshold: 0.9,
            enableBackgroundCleanup: false
        )

        let manager = DiskCacheManager(config: config)
        let namespace = DiskCacheManager.CacheNamespace(name: "test", version: 1)

        try manager.write("data1".data(using: .utf8)!, for: "key1", in: namespace)
        try manager.write("data2".data(using: .utf8)!, for: "key2", in: namespace)

        manager.clearNamespace(namespace)

        #expect(!manager.exists(key: "key1", in: namespace))
        #expect(!manager.exists(key: "key2", in: namespace))

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("DiskCacheManager calculateDiskUsage")
    func diskCacheManagerDiskUsage() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = DiskCacheManager.Config(
            basePath: tempDir,
            maxDiskUsageBytes: 10_000_000,
            cleanupThreshold: 0.9,
            enableBackgroundCleanup: false
        )

        let manager = DiskCacheManager(config: config)
        let namespace = DiskCacheManager.CacheNamespace(name: "test", version: 1)

        let data = String(repeating: "x", count: 1000).data(using: .utf8)!
        try manager.write(data, for: "key1", in: namespace)

        let usage = manager.calculateDiskUsage()
        #expect(usage >= 1000)

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("DiskCacheManager getDiskStats")
    func diskCacheManagerDiskStats() {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = DiskCacheManager.Config(
            basePath: tempDir,
            maxDiskUsageBytes: 10_000_000,
            cleanupThreshold: 0.9,
            enableBackgroundCleanup: false
        )

        let manager = DiskCacheManager(config: config)
        let stats = manager.getDiskStats()

        #expect(stats.maxBytes == 10_000_000)
        #expect(stats.usedBytes >= 0)
        #expect(stats.usagePercentage >= 0)

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - SourceCache Tests

    @Test("SourceCache put and get")
    func sourceCachePutGet() {
        let cache = SourceCache(maxEntries: 10)

        cache.put("key1", source: "source code 1")
        cache.put("key2", source: "source code 2")

        #expect(cache.get("key1") == "source code 1")
        #expect(cache.get("key2") == "source code 2")
        #expect(cache.get("key3") == nil)
    }

    @Test("SourceCache contains")
    func sourceCacheContains() {
        let cache = SourceCache()

        #expect(!cache.contains("key1"))

        cache.put("key1", source: "source")

        #expect(cache.contains("key1"))
    }

    @Test("SourceCache eviction")
    func sourceCacheEviction() {
        let cache = SourceCache(maxEntries: 3)

        cache.put("key1", source: "source1")
        Thread.sleep(forTimeInterval: 0.01)
        cache.put("key2", source: "source2")
        Thread.sleep(forTimeInterval: 0.01)
        cache.put("key3", source: "source3")
        Thread.sleep(forTimeInterval: 0.01)

        // This should evict key1 (oldest)
        cache.put("key4", source: "source4")

        #expect(cache.get("key1") == nil) // Evicted
        #expect(cache.get("key2") != nil)
        #expect(cache.get("key3") != nil)
        #expect(cache.get("key4") != nil)
    }

    @Test("SourceCache clear")
    func sourceCacheClear() {
        let cache = SourceCache()

        cache.put("key1", source: "source1")
        cache.put("key2", source: "source2")

        cache.clear()

        #expect(cache.get("key1") == nil)
        #expect(cache.get("key2") == nil)
    }

    // MARK: - PipelineStateCache Tests

    @Test("PipelineStateCache initial stats")
    func pipelineStateCacheInitialStats() {
        let cache = PipelineStateCache()
        let stats = cache.getStats()

        #expect(stats.compute == 0)
        #expect(stats.render == 0)
    }

    @Test("PipelineStateCache compute pipeline operations")
    func pipelineStateCacheComputeOps() {
        let cache = PipelineStateCache()

        // Get non-existent pipeline
        #expect(cache.getComputePipeline("test") == nil)

        // Stats after attempted get
        let stats = cache.getStats()
        #expect(stats.compute == 0)
    }

    @Test("PipelineStateCache render pipeline operations")
    func pipelineStateCacheRenderOps() {
        let cache = PipelineStateCache()

        // Get non-existent pipeline
        #expect(cache.getRenderPipeline("test") == nil)

        // Stats after attempted get
        let stats = cache.getStats()
        #expect(stats.render == 0)
    }

    @Test("PipelineStateCache remove")
    func pipelineStateCacheRemove() {
        let cache = PipelineStateCache()

        // Remove non-existent should not crash
        cache.remove("test")

        let stats = cache.getStats()
        #expect(stats.compute == 0)
        #expect(stats.render == 0)
    }

    @Test("PipelineStateCache clear")
    func pipelineStateCacheClear() {
        let cache = PipelineStateCache()

        cache.clear()

        let stats = cache.getStats()
        #expect(stats.compute == 0)
        #expect(stats.render == 0)
    }

    // MARK: - CacheError Tests

    @Test("CacheError cases")
    func cacheErrorCases() {
        let error1 = CacheError.serializationFailed
        let error2 = CacheError.deserializationFailed
        let error3 = CacheError.diskFull
        let error4 = CacheError.ioError("Test IO error")
        let error5 = CacheError.invalidKey
        let error6 = CacheError.entryNotFound

        #expect("\(error1)".contains("serializationFailed"))
        #expect("\(error2)".contains("deserializationFailed"))
        #expect("\(error3)".contains("diskFull"))
        #expect("\(error4)".contains("ioError"))
        #expect("\(error5)".contains("invalidKey"))
        #expect("\(error6)".contains("entryNotFound"))
    }

    // MARK: - DiskStats Tests

    @Test("DiskStats creation")
    func diskStatsCreation() {
        let stats = DiskStats(
            usedBytes: 500_000_000,
            maxBytes: 1_000_000_000,
            usagePercentage: 0.5
        )

        #expect(stats.usedBytes == 500_000_000)
        #expect(stats.maxBytes == 1_000_000_000)
        #expect(stats.usagePercentage == 0.5)
    }

    // MARK: - Integration Tests

    @Test("Cache key consistency")
    func cacheKeyConsistency() {
        let source1 = "kernel void test() {}"
        let source2 = "kernel void test() {}"
        let source3 = "kernel void other() {}"

        // Same source should produce same hash
        let key1 = KernelCacheKey(
            sourceHash: String(source1.hashValue),
            deviceId: "device"
        )
        let key2 = KernelCacheKey(
            sourceHash: String(source2.hashValue),
            deviceId: "device"
        )
        let key3 = KernelCacheKey(
            sourceHash: String(source3.hashValue),
            deviceId: "device"
        )

        #expect(key1 == key2)
        #expect(key1 != key3)
    }

    @Test("End-to-end disk cache workflow")
    func endToEndDiskCacheWorkflow() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = DiskCacheManager.Config(
            basePath: tempDir,
            maxDiskUsageBytes: 10_000_000,
            cleanupThreshold: 0.9,
            enableBackgroundCleanup: false
        )

        let manager = DiskCacheManager(config: config)

        // Write multiple items
        let namespace = DiskCacheManager.CacheNamespace(name: "kernels", version: 1)

        for i in 0..<5 {
            let data = "Kernel source \(i)".data(using: .utf8)!
            try manager.write(data, for: "kernel_\(i)", in: namespace)
        }

        // Verify all items exist
        for i in 0..<5 {
            #expect(manager.exists(key: "kernel_\(i)", in: namespace))
        }

        // Read and verify content
        for i in 0..<5 {
            let data = manager.read(for: "kernel_\(i)", in: namespace)
            let content = String(data: data!, encoding: .utf8)
            #expect(content == "Kernel source \(i)")
        }

        // Clear and verify
        manager.clearAll()
        for i in 0..<5 {
            #expect(!manager.exists(key: "kernel_\(i)", in: namespace))
        }

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    @Test("Source cache thread safety")
    func sourceCacheThreadSafety() async {
        let cache = SourceCache(maxEntries: 1000)

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    for j in 0..<50 {
                        cache.put("key_\(i)_\(j)", source: "source_\(i)_\(j)")
                        _ = cache.get("key_\(i)_\(j)")
                    }
                }
            }
        }

        // Test passes if it doesn't crash - thread safety verified
        // The cache should have retained some entries (500 total inserts, 1000 capacity)
        var foundCount = 0
        for i in 0..<10 {
            for j in 0..<50 {
                if cache.contains("key_\(i)_\(j)") {
                    foundCount += 1
                }
            }
        }
        // With 1000 capacity and 500 inserts, we should find most entries
        #expect(foundCount > 0)
    }
}
