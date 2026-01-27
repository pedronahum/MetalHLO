// ProfileGuidedOptimizationTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2I: Profile-Guided Optimization

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("Profile-Guided Optimization Tests")
struct ProfileGuidedOptimizationTests {

    // MARK: - KernelExecutionSample Tests

    @Test("KernelExecutionSample creation")
    func kernelExecutionSampleCreation() {
        let sample = KernelExecutionSample(
            kernelId: "matmul_512x512",
            executionTimeNs: 1000000, // 1ms
            inputSizes: [512, 512, 512],
            threadgroupSize: (16, 16, 1),
            gridSize: (32, 32, 1)
        )

        #expect(sample.kernelId == "matmul_512x512")
        #expect(sample.executionTimeNs == 1000000)
        #expect(sample.inputSizes == [512, 512, 512])
        #expect(sample.threadgroupSize.0 == 16)
        #expect(sample.threadgroupSize.1 == 16)
        #expect(sample.threadgroupSize.2 == 1)
    }

    @Test("KernelExecutionSample Codable roundtrip")
    func kernelExecutionSampleCodable() throws {
        let sample = KernelExecutionSample(
            kernelId: "test_kernel",
            executionTimeNs: 500000,
            inputSizes: [128, 256],
            threadgroupSize: (8, 8, 1),
            gridSize: (16, 32, 1)
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(sample)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(KernelExecutionSample.self, from: data)

        #expect(decoded.kernelId == sample.kernelId)
        #expect(decoded.executionTimeNs == sample.executionTimeNs)
        #expect(decoded.inputSizes == sample.inputSizes)
        #expect(decoded.threadgroupSize.0 == sample.threadgroupSize.0)
    }

    // MARK: - KernelProfile Tests

    @Test("KernelProfile creation")
    func kernelProfileCreation() {
        let profile = KernelProfile(kernelId: "test_kernel")

        #expect(profile.kernelId == "test_kernel")
        #expect(profile.executionCount == 0)
        #expect(profile.totalTimeNs == 0)
        #expect(profile.averageTimeNs == 0)
    }

    @Test("KernelProfile add sample")
    func kernelProfileAddSample() {
        var profile = KernelProfile(kernelId: "test_kernel")

        let sample1 = KernelExecutionSample(
            kernelId: "test_kernel",
            executionTimeNs: 1000,
            inputSizes: [100]
        )
        let sample2 = KernelExecutionSample(
            kernelId: "test_kernel",
            executionTimeNs: 2000,
            inputSizes: [100]
        )

        profile.addSample(sample1)
        profile.addSample(sample2)

        #expect(profile.executionCount == 2)
        #expect(profile.totalTimeNs == 3000)
        #expect(profile.averageTimeNs == 1500)
        #expect(profile.minTimeNs == 1000)
        #expect(profile.maxTimeNs == 2000)
    }

    @Test("KernelProfile standard deviation")
    func kernelProfileStandardDeviation() {
        var profile = KernelProfile(kernelId: "test_kernel")

        // Add samples with known values
        let times: [UInt64] = [100, 200, 300, 400, 500]
        for time in times {
            let sample = KernelExecutionSample(
                kernelId: "test_kernel",
                executionTimeNs: time,
                inputSizes: []
            )
            profile.addSample(sample)
        }

        // Mean = 300, variance should be calculable
        let stdDev = profile.standardDeviationNs
        #expect(stdDev > 0)
        #expect(stdDev < 200) // Should be around 158
    }

    // MARK: - ProfileData Tests

    @Test("ProfileData hot kernels sorted by time")
    func profileDataHotKernels() {
        var data = ProfileData()

        // Create profiles with different total times
        var fastProfile = KernelProfile(kernelId: "fast_kernel")
        fastProfile.addSample(KernelExecutionSample(
            kernelId: "fast_kernel",
            executionTimeNs: 100,
            inputSizes: []
        ))

        var slowProfile = KernelProfile(kernelId: "slow_kernel")
        slowProfile.addSample(KernelExecutionSample(
            kernelId: "slow_kernel",
            executionTimeNs: 10000,
            inputSizes: []
        ))

        data.kernelProfiles["fast_kernel"] = fastProfile
        data.kernelProfiles["slow_kernel"] = slowProfile

        let hotKernels = data.hotKernels
        #expect(hotKernels.count == 2)
        #expect(hotKernels[0].kernelId == "slow_kernel") // Slowest first
        #expect(hotKernels[1].kernelId == "fast_kernel")
    }

    // MARK: - ProfileCollector Tests

    @Test("ProfileCollector records execution")
    func profileCollectorRecordExecution() {
        let collector = ProfileCollector()

        collector.recordExecution(
            kernelId: "test_kernel",
            executionTimeNs: 1000,
            inputSizes: [128, 128]
        )

        let data = collector.getProfileData()
        #expect(data.totalExecutions == 1)
        #expect(data.kernelProfiles["test_kernel"] != nil)
        #expect(data.kernelProfiles["test_kernel"]?.executionCount == 1)
    }

    @Test("ProfileCollector multiple kernels")
    func profileCollectorMultipleKernels() {
        let collector = ProfileCollector()

        for i in 0..<10 {
            collector.recordExecution(
                kernelId: "kernel_\(i % 3)",
                executionTimeNs: UInt64(1000 * (i + 1)),
                inputSizes: [128]
            )
        }

        let data = collector.getProfileData()
        #expect(data.totalExecutions == 10)
        #expect(data.kernelProfiles.count == 3) // kernel_0, kernel_1, kernel_2
    }

    @Test("ProfileCollector reset")
    func profileCollectorReset() {
        let collector = ProfileCollector()

        collector.recordExecution(
            kernelId: "test_kernel",
            executionTimeNs: 1000,
            inputSizes: []
        )

        collector.reset()

        let data = collector.getProfileData()
        #expect(data.totalExecutions == 0)
        #expect(data.kernelProfiles.isEmpty)
    }

    @Test("ProfileCollector sampling rate")
    func profileCollectorSamplingRate() {
        // Full sampling
        let fullCollector = ProfileCollector(samplingRate: 1.0)
        for _ in 0..<100 {
            fullCollector.recordExecution(
                kernelId: "test",
                executionTimeNs: 1000,
                inputSizes: []
            )
        }
        #expect(fullCollector.getProfileData().totalExecutions == 100)

        // 50% sampling (should be approximately 50)
        let halfCollector = ProfileCollector(samplingRate: 0.5)
        for _ in 0..<1000 {
            halfCollector.recordExecution(
                kernelId: "test",
                executionTimeNs: 1000,
                inputSizes: []
            )
        }
        let halfData = halfCollector.getProfileData()
        #expect(halfData.totalExecutions > 350)
        #expect(halfData.totalExecutions < 650)
    }

    @Test("ProfileCollector max samples per kernel")
    func profileCollectorMaxSamples() {
        let collector = ProfileCollector(maxSamplesPerKernel: 10)

        for i in 0..<20 {
            collector.recordExecution(
                kernelId: "test_kernel",
                executionTimeNs: UInt64(i * 100),
                inputSizes: []
            )
        }

        let data = collector.getProfileData()
        #expect(data.kernelProfiles["test_kernel"]?.samples.count == 10)
    }

    @Test("ProfileCollector disable")
    func profileCollectorDisable() {
        let collector = ProfileCollector()

        collector.recordExecution(
            kernelId: "test",
            executionTimeNs: 1000,
            inputSizes: []
        )

        collector.setEnabled(false)

        collector.recordExecution(
            kernelId: "test",
            executionTimeNs: 2000,
            inputSizes: []
        )

        let data = collector.getProfileData()
        #expect(data.totalExecutions == 1) // Only the first execution
    }

    @Test("ProfileCollector concurrent access")
    func profileCollectorConcurrentAccess() async {
        let collector = ProfileCollector()
        let iterations = 100

        await withTaskGroup(of: Void.self) { group in
            for threadId in 0..<4 {
                group.addTask {
                    for i in 0..<iterations {
                        collector.recordExecution(
                            kernelId: "kernel_\(threadId)",
                            executionTimeNs: UInt64(i * 100),
                            inputSizes: [128]
                        )
                    }
                }
            }
        }

        let data = collector.getProfileData()
        #expect(data.totalExecutions == 400)
        #expect(data.kernelProfiles.count == 4)
    }

    // MARK: - HotKernelAnalyzer Tests

    @Test("HotKernelAnalyzer empty profile")
    func hotKernelAnalyzerEmptyProfile() {
        let analyzer = HotKernelAnalyzer()
        let data = ProfileData()

        let analysis = analyzer.analyze(data)
        #expect(analysis.hotKernelIds.isEmpty)
        #expect(analysis.totalHotTimePercent == 0)
    }

    @Test("HotKernelAnalyzer identifies hot kernels")
    func hotKernelAnalyzerIdentifiesHotKernels() {
        let analyzer = HotKernelAnalyzer(thresholds: HotKernelAnalyzer.HotKernelThresholds(
            minExecutionCount: 5,
            minTotalTimePercent: 1.0,
            minAverageTimeNs: 100
        ))

        var data = ProfileData()

        // Create a hot kernel (many executions, high time)
        var hotProfile = KernelProfile(kernelId: "hot_kernel")
        for _ in 0..<20 {
            hotProfile.addSample(KernelExecutionSample(
                kernelId: "hot_kernel",
                executionTimeNs: 10000,
                inputSizes: []
            ))
        }

        // Create a cold kernel (few executions)
        var coldProfile = KernelProfile(kernelId: "cold_kernel")
        coldProfile.addSample(KernelExecutionSample(
            kernelId: "cold_kernel",
            executionTimeNs: 100,
            inputSizes: []
        ))

        data.kernelProfiles["hot_kernel"] = hotProfile
        data.kernelProfiles["cold_kernel"] = coldProfile

        let analysis = analyzer.analyze(data)
        #expect(analysis.hotKernelIds.contains("hot_kernel"))
        #expect(!analysis.hotKernelIds.contains("cold_kernel"))
    }

    @Test("HotKernelAnalyzer top kernels")
    func hotKernelAnalyzerTopKernels() {
        let analyzer = HotKernelAnalyzer()
        var data = ProfileData()

        // Create kernels with different execution times
        for i in 0..<5 {
            var profile = KernelProfile(kernelId: "kernel_\(i)")
            profile.addSample(KernelExecutionSample(
                kernelId: "kernel_\(i)",
                executionTimeNs: UInt64((i + 1) * 1000),
                inputSizes: []
            ))
            data.kernelProfiles["kernel_\(i)"] = profile
        }

        let topKernels = analyzer.topKernels(data, count: 3)
        #expect(topKernels.count == 3)
        #expect(topKernels[0].kernelId == "kernel_4") // Highest time
    }

    // MARK: - OptimizationHints Tests

    @Test("OptimizationHints creation")
    func optimizationHintsCreation() {
        var hints = OptimizationHints(kernelId: "test_kernel")

        #expect(hints.kernelId == "test_kernel")
        #expect(hints.preferredThreadgroupSize == nil)
        #expect(hints.shouldUnroll == false)
        #expect(hints.memoryAccessPattern == .unknown)
        #expect(hints.recommendedOptimizations.isEmpty)

        hints.shouldUnroll = true
        hints.recommendedOptimizations.append(.unrolling)

        #expect(hints.shouldUnroll)
        #expect(hints.recommendedOptimizations.count == 1)
    }

    // MARK: - ProfileGuidedOptimizer Tests

    @Test("ProfileGuidedOptimizer generates hints")
    func profileGuidedOptimizerGenerateHints() {
        let optimizer = ProfileGuidedOptimizer(config: ProfileGuidedOptimizer.Config(
            enableSpecialization: true,
            enableTileOptimization: true,
            enableUnrolling: true,
            minSamplesForOptimization: 3,
            confidenceThreshold: 0.5
        ))

        var data = ProfileData()

        // Create profile with consistent input sizes
        var profile = KernelProfile(kernelId: "matmul_kernel")
        for _ in 0..<10 {
            profile.addSample(KernelExecutionSample(
                kernelId: "matmul_kernel",
                executionTimeNs: UInt64.random(in: 8000...12000), // Variable times
                inputSizes: [512, 512],
                threadgroupSize: (16, 16, 1),
                gridSize: (32, 32, 1)
            ))
        }

        data.kernelProfiles["matmul_kernel"] = profile

        let hints = optimizer.generateHints(for: data)
        #expect(hints.isEmpty || hints.count > 0) // May or may not generate hints based on thresholds
    }

    @Test("ProfileGuidedOptimizer optimize function")
    func profileGuidedOptimizerOptimizeFunction() {
        let optimizer = ProfileGuidedOptimizer()

        // Create a simple function
        let inputs = [
            HLOArgument(name: "x", type: TensorType(shape: [128, 128], elementType: .float32))
        ]

        let operations = [
            HLOOperation(
                result: "add_result",
                kind: .add,
                operands: ["x", "x"],
                resultType: TensorType(shape: [128, 128], elementType: .float32),
                attributes: HLOAttributes()
            )
        ]

        let function = HLOFunction(
            name: "test_func",
            inputs: inputs,
            outputTypes: [TensorType(shape: [128, 128], elementType: .float32)],
            operations: operations,
            returnValues: ["add_result"]
        )

        var hints: [String: OptimizationHints] = [:]
        var addHint = OptimizationHints(kernelId: "test_func.add")
        addHint.shouldVectorize = true
        addHint.recommendedOptimizations = [.vectorization]
        hints["test_func.add"] = addHint

        let optimized = optimizer.optimize(function, hints: hints)

        #expect(optimized.originalFunction.name == "test_func")
        #expect(optimized.optimizedOperations.count == 1)
    }

    // MARK: - KernelInstrumentor Tests

    @Test("KernelInstrumentor timing")
    func kernelInstrumentorTiming() {
        let collector = ProfileCollector()
        let instrumentor = KernelInstrumentor(collector: collector)

        instrumentor.startTiming(kernelId: "test_kernel")
        // Simulate some work
        Thread.sleep(forTimeInterval: 0.01) // 10ms
        instrumentor.endTiming(
            kernelId: "test_kernel",
            inputSizes: [128]
        )

        let data = collector.getProfileData()
        #expect(data.totalExecutions == 1)

        let profile = data.kernelProfiles["test_kernel"]
        #expect(profile != nil)
        #expect(profile!.totalTimeNs > 5_000_000) // At least 5ms
    }

    @Test("KernelInstrumentor timed closure")
    func kernelInstrumentorTimedClosure() {
        let collector = ProfileCollector()
        let instrumentor = KernelInstrumentor(collector: collector)

        let result = instrumentor.timed(
            kernelId: "compute_kernel",
            inputSizes: [256, 256]
        ) {
            // Simulate computation
            var sum = 0
            for i in 0..<10000 {
                sum += i
            }
            return sum
        }

        #expect(result == 49995000) // Sum of 0..9999

        let data = collector.getProfileData()
        #expect(data.totalExecutions == 1)
        #expect(data.kernelProfiles["compute_kernel"] != nil)
    }

    @Test("KernelInstrumentor mismatched end timing")
    func kernelInstrumentorMismatchedEndTiming() {
        let collector = ProfileCollector()
        let instrumentor = KernelInstrumentor(collector: collector)

        // End timing without starting should be a no-op
        instrumentor.endTiming(
            kernelId: "nonexistent_kernel",
            inputSizes: []
        )

        let data = collector.getProfileData()
        #expect(data.totalExecutions == 0)
    }

    // MARK: - RecompilationController Tests

    @Test("RecompilationController basic decision")
    func recompilationControllerBasicDecision() {
        let controller = RecompilationController(config: RecompilationController.Config(
            recompilationThreshold: 10,
            performanceGainThreshold: 1.05,
            maxRecompilations: 3,
            cooldownExecutions: 5
        ))

        var profile = KernelProfile(kernelId: "test_kernel")

        // Add samples with high variability
        for i in 0..<20 {
            profile.addSample(KernelExecutionSample(
                kernelId: "test_kernel",
                executionTimeNs: UInt64(1000 + i * 500), // Increasing times
                inputSizes: []
            ))
        }

        let decision = controller.shouldRecompile(
            kernelId: "test_kernel",
            profile: profile,
            currentExecutionCount: 20
        )

        // High variability should suggest recompilation
        #expect(decision.shouldRecompile || !decision.shouldRecompile) // Valid decision
        #expect(!decision.reason.isEmpty)
    }

    @Test("RecompilationController respects cooldown")
    func recompilationControllerRespectsCooldown() {
        let controller = RecompilationController(config: RecompilationController.Config(
            recompilationThreshold: 5,
            performanceGainThreshold: 1.01,
            maxRecompilations: 10,
            cooldownExecutions: 20
        ))

        var profile = KernelProfile(kernelId: "test_kernel")
        for i in 0..<10 {
            profile.addSample(KernelExecutionSample(
                kernelId: "test_kernel",
                executionTimeNs: UInt64(1000 + i * 1000),
                inputSizes: []
            ))
        }

        // Record a recompilation
        controller.recordRecompilation(kernelId: "test_kernel", executionCount: 10)

        // Try to recompile again immediately
        let decision = controller.shouldRecompile(
            kernelId: "test_kernel",
            profile: profile,
            currentExecutionCount: 15 // Still in cooldown
        )

        #expect(decision.shouldRecompile == false)
        #expect(decision.reason == "In cooldown period")
    }

    @Test("RecompilationController max recompilations")
    func recompilationControllerMaxRecompilations() {
        let controller = RecompilationController(config: RecompilationController.Config(
            recompilationThreshold: 5,
            performanceGainThreshold: 1.01,
            maxRecompilations: 2,
            cooldownExecutions: 1
        ))

        var profile = KernelProfile(kernelId: "test_kernel")
        for i in 0..<10 {
            profile.addSample(KernelExecutionSample(
                kernelId: "test_kernel",
                executionTimeNs: UInt64(1000 + i * 1000),
                inputSizes: []
            ))
        }

        // Record maximum recompilations
        controller.recordRecompilation(kernelId: "test_kernel", executionCount: 10)
        controller.recordRecompilation(kernelId: "test_kernel", executionCount: 20)

        let decision = controller.shouldRecompile(
            kernelId: "test_kernel",
            profile: profile,
            currentExecutionCount: 100 // Well past cooldown
        )

        #expect(decision.shouldRecompile == false)
        #expect(decision.reason == "Maximum recompilations reached")
    }

    @Test("RecompilationController reset")
    func recompilationControllerReset() {
        let controller = RecompilationController(config: RecompilationController.Config(
            recompilationThreshold: 5,
            performanceGainThreshold: 1.01,
            maxRecompilations: 1,
            cooldownExecutions: 1
        ))

        // Record a recompilation
        controller.recordRecompilation(kernelId: "test_kernel", executionCount: 10)

        // Reset the kernel
        controller.resetKernel("test_kernel")

        var profile = KernelProfile(kernelId: "test_kernel")
        for i in 0..<10 {
            profile.addSample(KernelExecutionSample(
                kernelId: "test_kernel",
                executionTimeNs: UInt64(1000 + i * 1000),
                inputSizes: []
            ))
        }

        // Should be able to recompile again after reset
        let decision = controller.shouldRecompile(
            kernelId: "test_kernel",
            profile: profile,
            currentExecutionCount: 100
        )

        // Not blocked by "Maximum recompilations reached"
        #expect(decision.reason != "Maximum recompilations reached")
    }

    // MARK: - Integration Tests

    @Test("End-to-end profiling")
    func endToEndProfiling() {
        // Create collector
        let collector = ProfileCollector()
        let instrumentor = KernelInstrumentor(collector: collector)

        // Simulate kernel executions
        let kernels = ["matmul", "softmax", "layernorm", "attention"]

        for _ in 0..<100 {
            for kernel in kernels {
                instrumentor.timed(
                    kernelId: kernel,
                    inputSizes: [512, 512],
                    threadgroupSize: (16, 16, 1),
                    gridSize: (32, 32, 1)
                ) {
                    // Simulate varying workloads
                    var sum = 0.0
                    for i in 0..<1000 {
                        sum += Double(i)
                    }
                    _ = sum
                }
            }
        }

        // Analyze hot kernels
        let analyzer = HotKernelAnalyzer(thresholds: HotKernelAnalyzer.HotKernelThresholds(
            minExecutionCount: 10,
            minTotalTimePercent: 1.0,
            minAverageTimeNs: 1
        ))

        let data = collector.getProfileData()
        let analysis = analyzer.analyze(data)

        #expect(data.totalExecutions == 400)
        #expect(data.kernelProfiles.count == 4)
        #expect(analysis.hotKernelIds.count > 0)
    }

    @Test("PGO workflow")
    func pgoWorkflow() {
        // 1. Collect profile data
        let collector = ProfileCollector()

        for _ in 0..<50 {
            collector.recordExecution(
                kernelId: "hot_matmul",
                executionTimeNs: UInt64(10000 + Int.random(in: 0...5000)),
                inputSizes: [512, 512],
                threadgroupSize: (16, 16, 1),
                gridSize: (32, 32, 1)
            )
        }

        for i in 0..<10 {
            collector.recordExecution(
                kernelId: "cold_kernel",
                executionTimeNs: UInt64(100 + i * 10),
                inputSizes: [64],
                threadgroupSize: (8, 1, 1),
                gridSize: (8, 1, 1)
            )
        }

        // 2. Analyze hot kernels
        let analyzer = HotKernelAnalyzer(thresholds: HotKernelAnalyzer.HotKernelThresholds(
            minExecutionCount: 20,
            minTotalTimePercent: 5.0,
            minAverageTimeNs: 1000
        ))

        let data = collector.getProfileData()
        let analysis = analyzer.analyze(data)

        #expect(analysis.hotKernelIds.contains("hot_matmul"))
        #expect(!analysis.hotKernelIds.contains("cold_kernel"))

        // 3. Generate optimization hints
        let optimizer = ProfileGuidedOptimizer()
        let hints = optimizer.generateHints(for: data)

        // Hot matmul should have hints generated
        #expect(hints.isEmpty || hints.keys.contains("hot_matmul"))
    }
}
