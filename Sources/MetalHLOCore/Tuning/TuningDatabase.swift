// TuningDatabase.swift
// MetalHLOCore
//
// Persistent storage for auto-tuning results.

import Foundation
import Metal

/// Persistent database for storing auto-tuning results.
///
/// The tuning database caches optimal kernel configurations, enabling:
/// - Reuse of tuning results across sessions
/// - Building of device-specific tuning profiles
/// - Sharing of tuning data between deployments
public final class TuningDatabase: @unchecked Sendable {

    // MARK: - Properties

    /// Path to the database file.
    private let path: String

    /// In-memory cache of tuning results.
    private var cache: [TuningKey: TuningResult] = [:]

    /// Lock for thread-safe access.
    private let lock = NSLock()

    /// Whether the database has been modified since last save.
    private var isDirty: Bool = false

    /// Auto-save timer.
    private var saveTimer: Timer?

    // MARK: - Initialization

    /// Creates a tuning database at the given path.
    ///
    /// - Parameter path: Path to the database file.
    public init(path: String) {
        self.path = path
        loadFromDisk()
        setupAutoSave()
    }

    deinit {
        saveTimer?.invalidate()
        saveToDisk()
    }

    // MARK: - Access

    /// Gets a cached tuning result.
    ///
    /// - Parameter key: The tuning key.
    /// - Returns: Cached result, or nil if not found.
    public func get(_ key: TuningKey) -> TuningResult? {
        lock.lock()
        defer { lock.unlock() }
        return cache[key]
    }

    /// Stores a tuning result.
    ///
    /// - Parameters:
    ///   - key: The tuning key.
    ///   - result: The tuning result.
    public func store(_ key: TuningKey, result: TuningResult) {
        lock.lock()
        cache[key] = result
        isDirty = true
        lock.unlock()
    }

    /// Removes a tuning result.
    ///
    /// - Parameter key: The tuning key.
    public func remove(_ key: TuningKey) {
        lock.lock()
        cache.removeValue(forKey: key)
        isDirty = true
        lock.unlock()
    }

    /// Clears all cached results.
    public func clear() {
        lock.lock()
        cache.removeAll()
        isDirty = true
        lock.unlock()
    }

    /// Number of cached entries.
    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }

    /// All cached keys.
    public var keys: [TuningKey] {
        lock.lock()
        defer { lock.unlock() }
        return Array(cache.keys)
    }

    // MARK: - Persistence

    /// Loads the database from disk.
    private func loadFromDisk() {
        guard FileManager.default.fileExists(atPath: path),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            return
        }

        do {
            let decoded = try JSONDecoder().decode(DatabaseFormat.self, from: data)
            lock.lock()
            cache = decoded.entries
            lock.unlock()
        } catch {
            // Failed to decode - start fresh
            print("Warning: Failed to load tuning database: \(error)")
        }
    }

    /// Saves the database to disk.
    public func saveToDisk() {
        lock.lock()
        guard isDirty else {
            lock.unlock()
            return
        }
        let snapshot = cache
        isDirty = false
        lock.unlock()

        let format = DatabaseFormat(
            version: 1,
            entries: snapshot,
            savedAt: Date()
        )

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(format)
            try data.write(to: URL(fileURLWithPath: path))
        } catch {
            print("Warning: Failed to save tuning database: \(error)")
        }
    }

    /// Sets up periodic auto-save.
    private func setupAutoSave() {
        // Auto-save every 60 seconds if dirty
        saveTimer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
            self?.saveToDisk()
        }
    }

    // MARK: - Bulk Operations

    /// Imports tuning results from another database.
    ///
    /// - Parameter other: Path to the database to import.
    /// - Parameter overwrite: Whether to overwrite existing entries.
    /// - Returns: Number of entries imported.
    @discardableResult
    public func importFrom(_ other: String, overwrite: Bool = false) -> Int {
        guard FileManager.default.fileExists(atPath: other),
              let data = try? Data(contentsOf: URL(fileURLWithPath: other)),
              let decoded = try? JSONDecoder().decode(DatabaseFormat.self, from: data) else {
            return 0
        }

        lock.lock()
        var imported = 0

        for (key, result) in decoded.entries {
            if overwrite || cache[key] == nil {
                cache[key] = result
                imported += 1
            }
        }

        if imported > 0 {
            isDirty = true
        }

        lock.unlock()
        return imported
    }

    /// Exports tuning results for a specific device.
    ///
    /// - Parameter deviceID: Device identifier to filter by.
    /// - Returns: Filtered database entries.
    public func exportForDevice(_ deviceID: String) -> [TuningKey: TuningResult] {
        lock.lock()
        defer { lock.unlock() }

        return cache.filter { $0.key.deviceID == deviceID }
    }

    /// Prunes old entries from the database.
    ///
    /// - Parameter maxAge: Maximum age of entries to keep (in days).
    /// - Returns: Number of entries removed.
    @discardableResult
    public func pruneOldEntries(maxAge: Int = 30) -> Int {
        let cutoff = Calendar.current.date(byAdding: .day, value: -maxAge, to: Date())!

        lock.lock()
        let before = cache.count
        cache = cache.filter { $0.value.timestamp > cutoff }
        let removed = before - cache.count

        if removed > 0 {
            isDirty = true
        }

        lock.unlock()
        return removed
    }

    // MARK: - Statistics

    /// Database statistics.
    public struct Statistics {
        /// Total number of entries.
        public let totalEntries: Int

        /// Entries by operation type.
        public let entriesByOp: [String: Int]

        /// Entries by device.
        public let entriesByDevice: [String: Int]

        /// Average GFLOPS across all entries.
        public let averageGFLOPS: Double

        /// Best GFLOPS achieved.
        public let bestGFLOPS: Double

        /// Total disk size in bytes.
        public let diskSizeBytes: Int
    }

    /// Gets database statistics.
    public func getStatistics() -> Statistics {
        lock.lock()
        defer { lock.unlock() }

        var entriesByOp: [String: Int] = [:]
        var entriesByDevice: [String: Int] = [:]
        var totalGFLOPS: Double = 0
        var bestGFLOPS: Double = 0

        for (key, result) in cache {
            entriesByOp[key.opType, default: 0] += 1
            entriesByDevice[key.deviceID, default: 0] += 1
            totalGFLOPS += result.gflops
            bestGFLOPS = max(bestGFLOPS, result.gflops)
        }

        let diskSize: Int
        if let attrs = try? FileManager.default.attributesOfItem(atPath: path),
           let size = attrs[.size] as? Int {
            diskSize = size
        } else {
            diskSize = 0
        }

        return Statistics(
            totalEntries: cache.count,
            entriesByOp: entriesByOp,
            entriesByDevice: entriesByDevice,
            averageGFLOPS: cache.isEmpty ? 0 : totalGFLOPS / Double(cache.count),
            bestGFLOPS: bestGFLOPS,
            diskSizeBytes: diskSize
        )
    }
}

// MARK: - Database Format

/// On-disk format for the tuning database.
private struct DatabaseFormat: Codable {
    let version: Int
    let entries: [TuningKey: TuningResult]
    let savedAt: Date
}

// MARK: - Pre-built Tuning Profiles

extension TuningDatabase {

    /// Generates a pre-tuned database for common transformer configurations.
    ///
    /// This creates optimal configurations for common model shapes,
    /// allowing fast startup without online tuning.
    ///
    /// - Parameters:
    ///   - device: Metal device.
    ///   - autoTuner: AutoTuner for benchmarking.
    /// - Returns: Number of configurations tuned.
    @discardableResult
    public func generateTransformerProfiles(
        device: MTLDevice,
        autoTuner: AutoTuner
    ) -> Int {
        var count = 0

        // Common hidden dimensions
        let hiddenSizes = [768, 1024, 2048, 4096, 5120, 8192]

        // Common sequence lengths
        let seqLens = [128, 256, 512, 1024, 2048]

        // Common batch sizes
        let batchSizes = [1, 2, 4, 8]

        // Common head configurations
        let headConfigs = [
            (heads: 12, dim: 64),   // BERT-base
            (heads: 16, dim: 64),   // BERT-large, GPT-2
            (heads: 32, dim: 128),  // LLaMA-7B
            (heads: 40, dim: 128),  // LLaMA-13B
            (heads: 64, dim: 128),  // LLaMA-70B
        ]

        for hidden in hiddenSizes {
            for seq in seqLens {
                for batch in batchSizes {
                    // QKV projection: [batch*seq, hidden] @ [hidden, 3*hidden]
                    let qkvShapes = [
                        TensorType(shape: [batch * seq, hidden], elementType: .float32),
                        TensorType(shape: [hidden, 3 * hidden], elementType: .float32)
                    ]
                    let qkvKey = TuningKey(opType: .dot, shapes: qkvShapes, device: device)
                    if cache[qkvKey] == nil {
                        let result = autoTuner.tune(key: qkvKey, op: .dot, shapes: qkvShapes, maxTrials: 20)
                        store(qkvKey, result: result)
                        count += 1
                    }

                    // Output projection: [batch*seq, hidden] @ [hidden, hidden]
                    let outShapes = [
                        TensorType(shape: [batch * seq, hidden], elementType: .float32),
                        TensorType(shape: [hidden, hidden], elementType: .float32)
                    ]
                    let outKey = TuningKey(opType: .dot, shapes: outShapes, device: device)
                    if cache[outKey] == nil {
                        let result = autoTuner.tune(key: outKey, op: .dot, shapes: outShapes, maxTrials: 20)
                        store(outKey, result: result)
                        count += 1
                    }

                    // FFN up: [batch*seq, hidden] @ [hidden, 4*hidden]
                    let ffnUpShapes = [
                        TensorType(shape: [batch * seq, hidden], elementType: .float32),
                        TensorType(shape: [hidden, 4 * hidden], elementType: .float32)
                    ]
                    let ffnUpKey = TuningKey(opType: .dot, shapes: ffnUpShapes, device: device)
                    if cache[ffnUpKey] == nil {
                        let result = autoTuner.tune(key: ffnUpKey, op: .dot, shapes: ffnUpShapes, maxTrials: 20)
                        store(ffnUpKey, result: result)
                        count += 1
                    }

                    // FFN down: [batch*seq, 4*hidden] @ [4*hidden, hidden]
                    let ffnDownShapes = [
                        TensorType(shape: [batch * seq, 4 * hidden], elementType: .float32),
                        TensorType(shape: [4 * hidden, hidden], elementType: .float32)
                    ]
                    let ffnDownKey = TuningKey(opType: .dot, shapes: ffnDownShapes, device: device)
                    if cache[ffnDownKey] == nil {
                        let result = autoTuner.tune(key: ffnDownKey, op: .dot, shapes: ffnDownShapes, maxTrials: 20)
                        store(ffnDownKey, result: result)
                        count += 1
                    }
                }
            }
        }

        // Attention configurations
        for (heads, headDim) in headConfigs {
            for batch in batchSizes {
                for seq in seqLens {
                    // Q @ K^T: [batch*heads, seq, dim] @ [batch*heads, dim, seq]
                    let qkShapes = [
                        TensorType(shape: [batch * heads, seq, headDim], elementType: .float32),
                        TensorType(shape: [batch * heads, headDim, seq], elementType: .float32)
                    ]
                    let qkKey = TuningKey(opType: .dot, shapes: qkShapes, device: device)
                    if cache[qkKey] == nil {
                        let result = autoTuner.tune(key: qkKey, op: .dot, shapes: qkShapes, maxTrials: 20)
                        store(qkKey, result: result)
                        count += 1
                    }

                    // Attn @ V: [batch*heads, seq, seq] @ [batch*heads, seq, dim]
                    let avShapes = [
                        TensorType(shape: [batch * heads, seq, seq], elementType: .float32),
                        TensorType(shape: [batch * heads, seq, headDim], elementType: .float32)
                    ]
                    let avKey = TuningKey(opType: .dot, shapes: avShapes, device: device)
                    if cache[avKey] == nil {
                        let result = autoTuner.tune(key: avKey, op: .dot, shapes: avShapes, maxTrials: 20)
                        store(avKey, result: result)
                        count += 1
                    }
                }
            }
        }

        // Save after generating profiles
        saveToDisk()

        return count
    }
}
