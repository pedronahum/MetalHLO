// CodegenTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2A: Shape-Specialized Kernel Generation

import Testing
import Metal
@testable import MetalHLOCore

@Suite("Codegen Tests")
struct CodegenTests {

    // MARK: - TileCalculator Tests

    @Test("TileCalculator initializes with default values")
    func tileCalculatorInitialization() {
        let calculator = TileCalculator()

        #expect(calculator.maxThreadsPerThreadgroup == 1024)
        #expect(calculator.maxSharedMemory == 32768)
        #expect(calculator.simdWidth == 32)
    }

    @Test("TileCalculator computes valid MatMul tiles")
    func matMulTileCalculation() {
        let calculator = TileCalculator()

        let config = calculator.calculateMatMulTiles(
            M: 512, N: 768, K: 768,
            elementType: .float32
        )

        // Verify reasonable tile sizes
        #expect(config.tileM > 0)
        #expect(config.tileN > 0)
        #expect(config.tileK > 0)
        #expect(config.tileM <= 512)
        #expect(config.tileN <= 768)
        #expect(config.tileK <= 768)

        // Verify shared memory fits
        #expect(config.sharedMemorySize <= calculator.maxSharedMemory)
    }

    @Test("TileCalculator respects small dimensions")
    func matMulSmallDimensions() {
        let calculator = TileCalculator()

        let config = calculator.calculateMatMulTiles(
            M: 16, N: 16, K: 8,
            elementType: .float32
        )

        // Small dimensions should not exceed problem size
        #expect(config.tileM <= 16)
        #expect(config.tileN <= 16)
        #expect(config.tileK <= 8)
    }

    @Test("TileCalculator computes valid attention tiles")
    func attentionTileCalculation() {
        let calculator = TileCalculator()

        let config = calculator.calculateAttentionTiles(
            batchSize: 4,
            seqLen: 512,
            numHeads: 8,
            headDim: 64,
            isCausal: true,
            elementType: .float32
        )

        #expect(config.blockQ > 0)
        #expect(config.blockKV > 0)
        #expect(config.sharedMemorySize <= calculator.maxSharedMemory)
    }

    @Test("TileCalculator computes valid reduction config")
    func reductionConfig() {
        let calculator = TileCalculator()

        let config = calculator.calculateReductionConfig(
            numElements: 1_000_000,
            elementType: .float32
        )

        #expect(config.elementsPerThread > 0)
        #expect(config.threadgroupSize > 0)
        #expect(config.threadgroupSize <= calculator.maxThreadsPerThreadgroup)
    }

    // MARK: - MetalIR Tests

    @Test("MetalIR emits constant values correctly")
    func metalIRConstantEmission() {
        let intConst = MetalIR.intConstant(42)
        #expect(intConst.emit() == "42")

        let floatConst = MetalIR.floatConstant(3.14)
        #expect(floatConst.emit().contains("3.14"))

        let boolConst = MetalIR.boolConstant(true)
        #expect(boolConst.emit() == "true")
    }

    @Test("MetalIR emits arithmetic operations correctly")
    func metalIRArithmetic() {
        let a = MetalIR.variable(name: "a")
        let b = MetalIR.variable(name: "b")

        let add = MetalIR.add(a, b)
        #expect(add.emit() == "(a + b)")

        let mul = MetalIR.mul(a, b)
        #expect(mul.emit() == "(a * b)")

        let fma = MetalIR.fma(a, b, MetalIR.variable(name: "c"))
        #expect(fma.emit() == "fma(a, b, c)")
    }

    @Test("MetalIR emits math functions correctly")
    func metalIRMathFunctions() {
        let x = MetalIR.variable(name: "x")

        #expect(MetalIR.exp(x).emit() == "exp(x)")
        #expect(MetalIR.log(x).emit() == "log(x)")
        #expect(MetalIR.sqrt(x).emit() == "sqrt(x)")
        #expect(MetalIR.tanh(x).emit() == "tanh(x)")
    }

    @Test("MetalIR emits control flow correctly")
    func metalIRControlFlow() {
        let cond = MetalIR.variable(name: "cond")
        let t = MetalIR.intConstant(1)
        let f = MetalIR.intConstant(0)

        let select = MetalIR.select(condition: cond, trueExpr: t, falseExpr: f)
        #expect(select.emit() == "(cond ? 1 : 0)")
    }

    @Test("MetalIR emits thread indexing correctly")
    func metalIRThreadIndexing() {
        let gidX = MetalIR.threadPositionInGrid(.x)
        #expect(gidX.emit() == "gid.x")

        let tidY = MetalIR.threadPositionInThreadgroup(.y)
        #expect(tidY.emit() == "tid.y")

        let simdLane = MetalIR.simdLaneID
        #expect(simdLane.emit() == "simd_lane_id")
    }

    @Test("MetalIR emits for loop correctly")
    func metalIRForLoop() {
        let loop = MetalIR.forLoop(
            iterVar: "i",
            start: .intConstant(0),
            end: .intConstant(10),
            step: .intConstant(1),
            body: .assign(name: "sum", value: .add(.variable(name: "sum"), .variable(name: "i")))
        )

        let emitted = loop.emit()
        #expect(emitted.contains("for (int i = 0; i < 10; i += 1)"))
    }

    // MARK: - SpecializationKey Tests

    @Test("SpecializationKey equality works correctly")
    func specializationKeyEquality() {
        let key1 = SpecializationKey(
            opType: .dot,
            shapes: [
                TensorType(shape: [512, 768], elementType: .float32),
                TensorType(shape: [768, 1024], elementType: .float32)
            ]
        )

        let key2 = SpecializationKey(
            opType: .dot,
            shapes: [
                TensorType(shape: [512, 768], elementType: .float32),
                TensorType(shape: [768, 1024], elementType: .float32)
            ]
        )

        #expect(key1 == key2)
        #expect(key1.hashValue == key2.hashValue)
    }

    @Test("SpecializationKey differs by operation")
    func specializationKeyDifferentOps() {
        let key1 = SpecializationKey(opType: .dot, shapes: [])
        let key2 = SpecializationKey(opType: .add, shapes: [])

        #expect(key1 != key2)
    }

    // MARK: - KernelSpecializer Tests

    @Test("KernelSpecializer should specialize matmul")
    func shouldSpecializeMatMul() {
        let specializer = KernelSpecializer()

        let shapes = [
            TensorType(shape: [512, 768], elementType: .float32),
            TensorType(shape: [768, 1024], elementType: .float32)
        ]

        #expect(specializer.shouldSpecialize(op: .dot, shapes: shapes) == true)
        #expect(specializer.shouldSpecialize(op: .dotGeneral, shapes: shapes) == true)
    }

    @Test("KernelSpecializer should not specialize small elementwise")
    func shouldNotSpecializeSmallElementwise() {
        let specializer = KernelSpecializer()

        // Small tensor - should not specialize
        let shapes = [TensorType(shape: [100], elementType: .float32)]
        #expect(specializer.shouldSpecialize(op: .add, shapes: shapes) == false)
    }

    @Test("KernelSpecializer should specialize large elementwise")
    func shouldSpecializeLargeElementwise() {
        let specializer = KernelSpecializer()

        // Large tensor - should specialize
        let shapes = [TensorType(shape: [10_000_000], elementType: .float32)]
        #expect(specializer.shouldSpecialize(op: .add, shapes: shapes) == true)
    }

    @Test("KernelSpecializer generates matmul kernel")
    func matMulKernelGeneration() {
        let specializer = KernelSpecializer()

        let shapes = [
            TensorType(shape: [512, 768], elementType: .float32),
            TensorType(shape: [768, 1024], elementType: .float32)
        ]

        let kernel = specializer.getSpecializedKernel(op: .dot, shapes: shapes)
        #expect(kernel != nil)

        if let kernel = kernel {
            #expect(kernel.source.isEmpty == false)
            #expect(kernel.source.contains("kernel void"))
            #expect(kernel.source.contains("512"))  // M dimension
            #expect(kernel.source.contains("768"))  // K dimension
            #expect(kernel.source.contains("1024")) // N dimension
        }
    }

    @Test("KernelSpecializer tracks statistics")
    func specializationStatistics() {
        let specializer = KernelSpecializer()

        // Generate a kernel
        let shapes = [
            TensorType(shape: [256, 256], elementType: .float32),
            TensorType(shape: [256, 256], elementType: .float32)
        ]

        _ = specializer.getSpecializedKernel(op: .dot, shapes: shapes)

        let stats = specializer.statistics
        #expect(stats.generated == 1)
        #expect(stats.cacheHits == 0)

        // Generate same kernel again - should be cached
        _ = specializer.getSpecializedKernel(op: .dot, shapes: shapes)

        let stats2 = specializer.statistics
        #expect(stats2.cacheHits == 1)
    }

    // MARK: - MatMul Specialization Tests

    @Test("MatMulSpecialization identifier is correct")
    func matMulSpecializationIdentifier() {
        let spec = MatMulSpecialization(M: 512, N: 768, K: 256)
        #expect(spec.identifier.contains("matmul"))
        #expect(spec.identifier.contains("512x256x768"))
    }

    @Test("MatMulSpecialization batched identifier includes batch")
    func matMulBatchedSpecialization() {
        let spec = MatMulSpecialization(
            M: 64, N: 64, K: 64,
            batchSize: 16
        )
        #expect(spec.identifier.contains("b16"))
    }

    @Test("MatMulSpecialization transposed identifier includes flags")
    func matMulTransposedSpecialization() {
        let spec = MatMulSpecialization(
            M: 128, N: 128, K: 128,
            transA: true, transB: true
        )
        #expect(spec.identifier.contains("tA"))
        #expect(spec.identifier.contains("tB"))
    }

    @Test("SpecializedMatMulGenerator generates kernel")
    func specializedMatMulGenerator() {
        let generator = SpecializedMatMulGenerator()

        let spec = MatMulSpecialization(M: 256, N: 256, K: 256)
        let kernel = generator.generate(spec: spec)

        #expect(kernel.source.isEmpty == false)
        #expect(kernel.functionName == spec.identifier)
        #expect(kernel.estimatedMetrics.flops > 0)
    }

    @Test("Transformer MatMul presets are correct")
    func transformerMatMulPresets() {
        let qk = MatMulSpecialization.transformerQK(batchHeads: 32, seqLen: 512, headDim: 64)
        #expect(qk.M == 512)
        #expect(qk.N == 512)
        #expect(qk.K == 64)
        #expect(qk.transB == true)

        let ffn1 = MatMulSpecialization.transformerFFN1(batchSeq: 2048, hiddenDim: 768, ffnDim: 3072)
        #expect(ffn1.M == 2048)
        #expect(ffn1.N == 3072)
        #expect(ffn1.K == 768)
    }

    // MARK: - Attention Specialization Tests

    @Test("AttentionSpecialization identifier is correct")
    func attentionSpecializationIdentifier() {
        let spec = AttentionSpecialization(
            batchSize: 4,
            numHeads: 8,
            seqLenQ: 512,
            headDim: 64,
            isCausal: true
        )

        #expect(spec.identifier.contains("attention"))
        #expect(spec.identifier.contains("b4"))
        #expect(spec.identifier.contains("h8"))
        #expect(spec.identifier.contains("sq512"))
        #expect(spec.identifier.contains("d64"))
        #expect(spec.identifier.contains("causal"))
    }

    @Test("GQA specialization is detected correctly")
    func gqaSpecialization() {
        let spec = AttentionSpecialization(
            batchSize: 1,
            numHeads: 32,
            numKVHeads: 8,
            seqLenQ: 1024,
            headDim: 128,
            isCausal: true
        )

        #expect(spec.isGQA == true)
        #expect(spec.headsPerKVHead == 4)
        #expect(spec.identifier.contains("kv8"))
    }

    @Test("SpecializedAttentionGenerator generates kernel")
    func specializedAttentionGenerator() {
        let generator = SpecializedAttentionGenerator()

        let spec = AttentionSpecialization(
            batchSize: 2,
            numHeads: 8,
            seqLenQ: 256,
            headDim: 64,
            isCausal: true
        )

        let kernel = generator.generate(spec: spec)

        #expect(kernel.source.isEmpty == false)
        #expect(kernel.source.contains("kernel void"))
        #expect(kernel.estimatedMetrics.flops > 0)
        #expect(kernel.estimatedMetrics.isComputeBound == true)
    }

    @Test("Llama presets are correct")
    func llamaPresets() {
        let llama7b = AttentionSpecialization.llama7B(batchSize: 1, seqLen: 2048)
        #expect(llama7b.numHeads == 32)
        #expect(llama7b.headDim == 128)
        #expect(llama7b.isCausal == true)

        let llama70b = AttentionSpecialization.llama70B(batchSize: 1, seqLen: 2048)
        #expect(llama70b.numHeads == 64)
        #expect(llama70b.numKVHeads == 8)
        #expect(llama70b.isGQA == true)
    }

    // MARK: - Performance Estimate Tests

    @Test("PerformanceEstimate computes arithmetic intensity")
    func performanceEstimateArithmeticIntensity() {
        let estimate = PerformanceEstimate(
            flops: 1_000_000,
            memoryBytes: 100_000,
            isComputeBound: true
        )

        #expect(estimate.arithmeticIntensity == 10.0)
    }

    // MARK: - TileConfigWrapper Tests

    @Test("TileConfigWrapper wraps MatMul config")
    func tileConfigWrapperMatMul() {
        let config = MatMulTileConfig(
            tileM: 64, tileN: 64, tileK: 32,
            numSimdGroups: 4,
            sharedMemorySize: 8192,
            unrollFactor: 4,
            useVectorLoads: true
        )

        let wrapper = TileConfigWrapper(matmul: config)
        #expect(wrapper.typeName == "MatMulTileConfig")
        #expect(wrapper.values["tileM"] == 64)
        #expect(wrapper.values["tileN"] == 64)
        #expect(wrapper.values["tileK"] == 32)
    }

    @Test("TileConfigWrapper wraps Attention config")
    func tileConfigWrapperAttention() {
        let config = AttentionTileConfig(
            blockQ: 64, blockKV: 64,
            numSimdGroups: 4,
            useCausalSkip: true,
            sharedMemorySize: 16384,
            useOnlineSoftmax: true
        )

        let wrapper = TileConfigWrapper(attention: config)
        #expect(wrapper.typeName == "AttentionTileConfig")
        #expect(wrapper.values["blockQ"] == 64)
        #expect(wrapper.values["useCausalSkip"] == 1)
    }

    // MARK: - MetalType Tests

    @Test("MetalType converts from ElementType")
    func metalTypeFromElementType() {
        #expect(MetalType.from(.float32) == .float)
        #expect(MetalType.from(.float16) == .half)
        #expect(MetalType.from(.int32) == .int)
    }

    @Test("MetalType byte sizes are correct")
    func metalTypeByteSize() {
        #expect(MetalType.float.byteSize == 4)
        #expect(MetalType.half.byteSize == 2)
        #expect(MetalType.float4.byteSize == 16)
    }
}
