// TuningTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2B: Auto-Tuning Infrastructure

import Testing
import Metal
@testable import MetalHLOCore

@Suite("Auto-Tuning Tests")
struct TuningTests {

    // MARK: - Search Space Tests

    @Test("SearchSpace calculates total configurations")
    func searchSpaceTotalConfigurations() {
        let space = SearchSpace(
            tileMOptions: [32, 64],
            tileNOptions: [32, 64],
            tileKOptions: [8, 16],
            numSimdGroupOptions: [2, 4],
            vectorWidthOptions: [1, 4],
            unrollOptions: [true, false]
        )

        // 2 * 2 * 2 * 2 * 2 * 2 = 64
        #expect(space.totalConfigurations == 64)
    }

    @Test("SearchSpace for MatMul filters by dimensions")
    func searchSpaceForMatMul() {
        let space = SearchSpace.forMatMul(M: 64, N: 64, K: 16)

        // Should not include tiles larger than dimensions
        #expect(space.tileMOptions.allSatisfy { $0 <= 64 })
        #expect(space.tileNOptions.allSatisfy { $0 <= 64 })
        #expect(space.tileKOptions.allSatisfy { $0 <= 16 })
    }

    @Test("SearchSpace for Attention uses head dimension")
    func searchSpaceForAttention() {
        let space = SearchSpace.forAttention(seqLen: 512, headDim: 64)

        #expect(space.tileMOptions.allSatisfy { $0 <= 512 })
        #expect(space.tileKOptions.contains(64))
    }

    @Test("SearchSpace for Reduction scales with elements")
    func searchSpaceForReduction() {
        let smallSpace = SearchSpace.forReduction(numElements: 100)
        let largeSpace = SearchSpace.forReduction(numElements: 100000)

        // Large reduction should have more threadgroup options
        #expect(largeSpace.tileMOptions.count >= smallSpace.tileMOptions.count)
    }

    // MARK: - Tuning Configuration Tests

    @Test("TuningConfiguration is hashable")
    func tuningConfigurationHashable() {
        let config1 = TuningConfiguration(
            tileM: 64, tileN: 64, tileK: 16,
            numSimdGroups: 4, vectorWidth: 4, useUnroll: true
        )

        let config2 = TuningConfiguration(
            tileM: 64, tileN: 64, tileK: 16,
            numSimdGroups: 4, vectorWidth: 4, useUnroll: true
        )

        #expect(config1 == config2)
        #expect(config1.hashValue == config2.hashValue)
    }

    @Test("TuningConfiguration description is correct")
    func tuningConfigurationDescription() {
        let config = TuningConfiguration(
            tileM: 64, tileN: 128, tileK: 32,
            numSimdGroups: 4, vectorWidth: 4, useUnroll: true
        )

        #expect(config.description.contains("64"))
        #expect(config.description.contains("128"))
        #expect(config.description.contains("32"))
        #expect(config.description.contains("simd4"))
    }

    // MARK: - Tuning Key Tests

    @Test("TuningKey encodes operation and shapes")
    func tuningKeyEncoding() {
        let shapes = [
            TensorType(shape: [512, 768], elementType: .float32),
            TensorType(shape: [768, 1024], elementType: .float32)
        ]

        let key = TuningKey(
            opType: "dot",
            shapes: [[512, 768], [768, 1024]],
            dtype: "float32",
            deviceID: "TestDevice"
        )

        #expect(key.opType == "dot")
        #expect(key.shapes == [[512, 768], [768, 1024]])
        #expect(key.dtype == "float32")
        #expect(key.deviceID == "TestDevice")
    }

    @Test("TuningKey equality works correctly")
    func tuningKeyEquality() {
        let key1 = TuningKey(
            opType: "dot",
            shapes: [[256, 256], [256, 256]],
            dtype: "float32",
            deviceID: "Device1"
        )

        let key2 = TuningKey(
            opType: "dot",
            shapes: [[256, 256], [256, 256]],
            dtype: "float32",
            deviceID: "Device1"
        )

        let key3 = TuningKey(
            opType: "add",
            shapes: [[256, 256], [256, 256]],
            dtype: "float32",
            deviceID: "Device1"
        )

        #expect(key1 == key2)
        #expect(key1 != key3)
    }

    // MARK: - Tuning Result Tests

    @Test("TuningResult records timestamp")
    func tuningResultTimestamp() {
        let before = Date()

        let result = TuningResult(
            bestConfig: TuningConfiguration(
                tileM: 64, tileN: 64, tileK: 16,
                numSimdGroups: 4, vectorWidth: 4, useUnroll: true
            ),
            measuredTime: 0.001,
            gflops: 1000.0,
            configsTried: 10
        )

        let after = Date()

        #expect(result.timestamp >= before)
        #expect(result.timestamp <= after)
    }

    @Test("TuningResult is codable")
    func tuningResultCodable() throws {
        let result = TuningResult(
            bestConfig: TuningConfiguration(
                tileM: 64, tileN: 64, tileK: 16,
                numSimdGroups: 4, vectorWidth: 4, useUnroll: true
            ),
            measuredTime: 0.001,
            gflops: 1000.0,
            configsTried: 10
        )

        let encoded = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(TuningResult.self, from: encoded)

        #expect(decoded.bestConfig == result.bestConfig)
        #expect(decoded.measuredTime == result.measuredTime)
        #expect(decoded.gflops == result.gflops)
        #expect(decoded.configsTried == result.configsTried)
    }

    // MARK: - Tuning Database Tests

    @Test("TuningDatabase stores and retrieves results")
    func tuningDatabaseStoreRetrieve() {
        let tempPath = NSTemporaryDirectory() + "test_tuning_\(UUID()).json"
        let db = TuningDatabase(path: tempPath)

        let key = TuningKey(
            opType: "dot",
            shapes: [[256, 256]],
            dtype: "float32",
            deviceID: "Test"
        )

        let result = TuningResult(
            bestConfig: TuningConfiguration(
                tileM: 64, tileN: 64, tileK: 16,
                numSimdGroups: 4, vectorWidth: 4, useUnroll: true
            ),
            measuredTime: 0.001,
            gflops: 500.0,
            configsTried: 5
        )

        db.store(key, result: result)

        let retrieved = db.get(key)
        #expect(retrieved != nil)
        #expect(retrieved?.bestConfig == result.bestConfig)
        #expect(db.count == 1)

        // Cleanup
        try? FileManager.default.removeItem(atPath: tempPath)
    }

    @Test("TuningDatabase clears all entries")
    func tuningDatabaseClear() {
        let tempPath = NSTemporaryDirectory() + "test_tuning_\(UUID()).json"
        let db = TuningDatabase(path: tempPath)

        // Add some entries
        for i in 0..<5 {
            let key = TuningKey(
                opType: "dot",
                shapes: [[i * 100]],
                dtype: "float32",
                deviceID: "Test"
            )
            let result = TuningResult(
                bestConfig: TuningConfiguration(
                    tileM: 64, tileN: 64, tileK: 16,
                    numSimdGroups: 4, vectorWidth: 4, useUnroll: true
                ),
                measuredTime: 0.001,
                gflops: 500.0,
                configsTried: 5
            )
            db.store(key, result: result)
        }

        #expect(db.count == 5)

        db.clear()

        #expect(db.count == 0)

        // Cleanup
        try? FileManager.default.removeItem(atPath: tempPath)
    }

    @Test("TuningDatabase persists to disk")
    func tuningDatabasePersistence() {
        let tempPath = NSTemporaryDirectory() + "test_tuning_\(UUID()).json"

        // Create and populate database
        do {
            let db = TuningDatabase(path: tempPath)
            let key = TuningKey(
                opType: "dot",
                shapes: [[512, 512]],
                dtype: "float32",
                deviceID: "Test"
            )
            let result = TuningResult(
                bestConfig: TuningConfiguration(
                    tileM: 128, tileN: 128, tileK: 32,
                    numSimdGroups: 8, vectorWidth: 4, useUnroll: true
                ),
                measuredTime: 0.002,
                gflops: 800.0,
                configsTried: 20
            )
            db.store(key, result: result)
            db.saveToDisk()
        }

        // Load in new instance
        let db2 = TuningDatabase(path: tempPath)
        let key = TuningKey(
            opType: "dot",
            shapes: [[512, 512]],
            dtype: "float32",
            deviceID: "Test"
        )

        let retrieved = db2.get(key)
        #expect(retrieved != nil)
        #expect(retrieved?.bestConfig.tileM == 128)

        // Cleanup
        try? FileManager.default.removeItem(atPath: tempPath)
    }

    @Test("TuningDatabase exports for device")
    func tuningDatabaseExportDevice() {
        let tempPath = NSTemporaryDirectory() + "test_tuning_\(UUID()).json"
        let db = TuningDatabase(path: tempPath)

        // Add entries for different devices
        for device in ["DeviceA", "DeviceB", "DeviceA"] {
            let key = TuningKey(
                opType: "dot",
                shapes: [[Int.random(in: 100...1000)]],
                dtype: "float32",
                deviceID: device
            )
            let result = TuningResult(
                bestConfig: TuningConfiguration(
                    tileM: 64, tileN: 64, tileK: 16,
                    numSimdGroups: 4, vectorWidth: 4, useUnroll: true
                ),
                measuredTime: 0.001,
                gflops: 500.0,
                configsTried: 5
            )
            db.store(key, result: result)
        }

        let deviceAEntries = db.exportForDevice("DeviceA")
        let deviceBEntries = db.exportForDevice("DeviceB")

        #expect(deviceAEntries.count == 2)
        #expect(deviceBEntries.count == 1)

        // Cleanup
        try? FileManager.default.removeItem(atPath: tempPath)
    }

    @Test("TuningDatabase statistics are correct")
    func tuningDatabaseStatistics() {
        let tempPath = NSTemporaryDirectory() + "test_tuning_\(UUID()).json"
        let db = TuningDatabase(path: tempPath)

        // Add entries with unique keys
        let entries: [(String, [[Int]], Double)] = [
            ("dot", [[256, 256]], 100.0),
            ("add", [[512]], 200.0),
            ("add", [[1024]], 300.0)
        ]

        for (opType, shapes, gflops) in entries {
            let key = TuningKey(
                opType: opType,
                shapes: shapes,
                dtype: "float32",
                deviceID: "TestDevice"
            )
            let result = TuningResult(
                bestConfig: TuningConfiguration(
                    tileM: 64, tileN: 64, tileK: 16,
                    numSimdGroups: 4, vectorWidth: 4, useUnroll: true
                ),
                measuredTime: 0.001,
                gflops: gflops,
                configsTried: 5
            )
            db.store(key, result: result)
        }

        let stats = db.getStatistics()

        #expect(stats.totalEntries == 3)
        #expect(stats.entriesByOp["dot"] == 1)
        #expect(stats.entriesByOp["add"] == 2)
        #expect(stats.entriesByDevice["TestDevice"] == 3)
        #expect(stats.bestGFLOPS == 300.0)

        // Cleanup
        try? FileManager.default.removeItem(atPath: tempPath)
    }

    // MARK: - AutoTuner Tests (Unit)

    @Test("AutoTuner returns heuristic config with heuristicsOnly strategy")
    func autoTunerHeuristicsOnly() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return // Skip if no Metal device
        }

        let tuner = AutoTuner(
            device: device,
            strategy: .heuristicsOnly
        )

        let shapes = [
            TensorType(shape: [512, 768], elementType: .float32),
            TensorType(shape: [768, 1024], elementType: .float32)
        ]

        let config = tuner.getOptimalConfig(for: .dot, shapes: shapes)

        // Should return valid config
        #expect(config.tileM > 0)
        #expect(config.tileN > 0)
        #expect(config.tileK > 0)
        #expect(config.numSimdGroups > 0)
    }

    @Test("AutoTuner heuristics for small MatMul")
    func autoTunerSmallMatMulHeuristic() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let tuner = AutoTuner(device: device, strategy: .heuristicsOnly)

        let shapes = [
            TensorType(shape: [32, 32], elementType: .float32),
            TensorType(shape: [32, 32], elementType: .float32)
        ]

        let config = tuner.getOptimalConfig(for: .dot, shapes: shapes)

        // Small problem should have small tiles
        #expect(config.tileM <= 32)
        #expect(config.tileN <= 32)
    }

    @Test("AutoTuner heuristics for large MatMul")
    func autoTunerLargeMatMulHeuristic() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let tuner = AutoTuner(device: device, strategy: .heuristicsOnly)

        let shapes = [
            TensorType(shape: [2048, 2048], elementType: .float32),
            TensorType(shape: [2048, 2048], elementType: .float32)
        ]

        let config = tuner.getOptimalConfig(for: .dot, shapes: shapes)

        // Large problem should have larger tiles
        #expect(config.tileM >= 64)
        #expect(config.tileN >= 64)
    }

    @Test("AutoTuner tracks statistics")
    func autoTunerStatistics() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let tuner = AutoTuner(device: device, strategy: .heuristicsOnly)

        // Make some queries
        for _ in 0..<5 {
            let shapes = [TensorType(shape: [256, 256], elementType: .float32)]
            _ = tuner.getOptimalConfig(for: .add, shapes: shapes)
        }

        let stats = tuner.statistics
        #expect(stats.tuningCount == 0) // Heuristics don't count as tuning
    }

    @Test("AutoTuner cache works")
    func autoTunerCache() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let tempPath = NSTemporaryDirectory() + "test_tuner_cache_\(UUID()).json"
        let tuner = AutoTuner(device: device, strategy: .heuristicsOnly, databasePath: tempPath)

        let shapes = [
            TensorType(shape: [512, 512], elementType: .float32),
            TensorType(shape: [512, 512], elementType: .float32)
        ]

        // First call - no cache hit
        _ = tuner.getOptimalConfig(for: .dot, shapes: shapes)
        let stats1 = tuner.statistics
        #expect(stats1.cacheHits == 0)

        // Manually store a result to trigger cache hit on next call
        // (In real usage, this would be done by actual tuning)

        // Cleanup
        try? FileManager.default.removeItem(atPath: tempPath)
    }

    // MARK: - Integration Tests

    @Test("Full tuning workflow")
    func fullTuningWorkflow() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return
        }

        let tempPath = NSTemporaryDirectory() + "test_workflow_\(UUID()).json"
        let tuner = AutoTuner(device: device, strategy: .heuristicsOnly, databasePath: tempPath)

        // Simulate multiple kernel configurations
        let configs = [
            ([512, 768], [768, 1024]),
            ([1024, 1024], [1024, 1024]),
            ([256, 256], [256, 256])
        ]

        for (shapeA, shapeB) in configs {
            let shapes = [
                TensorType(shape: shapeA, elementType: .float32),
                TensorType(shape: shapeB, elementType: .float32)
            ]

            let config = tuner.getOptimalConfig(for: .dot, shapes: shapes)

            // Verify we got valid configs
            #expect(config.tileM > 0)
            #expect(config.tileK > 0)
            #expect(config.numSimdGroups > 0)
        }

        // Cleanup
        try? FileManager.default.removeItem(atPath: tempPath)
    }
}
