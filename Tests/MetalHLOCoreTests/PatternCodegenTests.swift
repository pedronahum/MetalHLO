// PatternCodegenTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2C: Custom Metal Codegen

import Testing
import Foundation
@testable import MetalHLOCore

@Suite("Pattern Codegen Tests")
struct PatternCodegenTests {

    // MARK: - ComputePattern Tests

    @Test("AttentionConfig computes effective scale")
    func attentionConfigScale() {
        let config = ComputePattern.AttentionConfig(
            batchSize: 1,
            numHeads: 8,
            seqLenQ: 512,
            headDim: 64
        )

        let expected = 1.0 / sqrt(64.0)
        #expect(abs(config.effectiveScale - Float(expected)) < 0.001)
    }

    @Test("AttentionConfig detects GQA")
    func attentionConfigGQA() {
        let standard = ComputePattern.AttentionConfig(
            batchSize: 1,
            numHeads: 32,
            seqLenQ: 1024,
            headDim: 128
        )
        #expect(standard.isGQA == false)

        let gqa = ComputePattern.AttentionConfig(
            batchSize: 1,
            numHeads: 32,
            numKVHeads: 8,
            seqLenQ: 1024,
            headDim: 128
        )
        #expect(gqa.isGQA == true)
        #expect(gqa.headsPerKVHead == 4)
    }

    @Test("TransformerConfig computes head dimension")
    func transformerConfigHeadDim() {
        let config = ComputePattern.TransformerConfig(
            batchSize: 1,
            seqLen: 512,
            hiddenDim: 768,
            numHeads: 12
        )

        #expect(config.headDim == 64)
    }

    @Test("TransformerConfig has default FFN dimension")
    func transformerConfigFFNDim() {
        let config = ComputePattern.TransformerConfig(
            batchSize: 1,
            seqLen: 512,
            hiddenDim: 768,
            numHeads: 12
        )

        #expect(config.ffnDim == 768 * 4)
    }

    // MARK: - PatternCodegen Tests

    @Test("PatternCodegen generates MatMul kernel")
    func patternCodegenMatMul() {
        let codegen = PatternCodegen()

        let pattern = ComputePattern.matmul(M: 256, N: 256, K: 256, transA: false, transB: false)
        let source = codegen.generate(pattern)

        #expect(source.contains("#include <metal_stdlib>"))
        #expect(source.contains("kernel void"))
        #expect(source.contains("matmul_256x256x256"))
        #expect(source.contains("constant uint M = 256"))
        #expect(source.contains("constant uint N = 256"))
        #expect(source.contains("constant uint K = 256"))
    }

    @Test("PatternCodegen generates batched MatMul kernel")
    func patternCodegenBatchedMatMul() {
        let codegen = PatternCodegen()

        let pattern = ComputePattern.batchedMatmul(batchSize: 8, M: 128, N: 128, K: 64)
        let source = codegen.generate(pattern)

        #expect(source.contains("batched_matmul"))
        #expect(source.contains("constant uint BATCH = 8"))
        #expect(source.contains("uint batch = gid.z"))
    }

    @Test("PatternCodegen generates attention kernel")
    func patternCodegenAttention() {
        let codegen = PatternCodegen()

        let config = ComputePattern.AttentionConfig(
            batchSize: 2,
            numHeads: 8,
            seqLenQ: 512,
            headDim: 64,
            isCausal: true
        )
        let pattern = ComputePattern.attention(config)
        let source = codegen.generate(pattern)

        #expect(source.contains("FlashAttention"))
        #expect(source.contains("constant uint BATCH = 2"))
        #expect(source.contains("constant uint NUM_HEADS = 8"))
        #expect(source.contains("constant uint SEQ_LEN = 512"))
        #expect(source.contains("constant uint HEAD_DIM = 64"))
        #expect(source.contains("IS_CAUSAL"))
    }

    @Test("PatternCodegen generates attention with RoPE")
    func patternCodegenAttentionRoPE() {
        let codegen = PatternCodegen()

        let config = ComputePattern.AttentionConfig(
            batchSize: 1,
            numHeads: 32,
            seqLenQ: 1024,
            headDim: 128
        )
        let pattern = ComputePattern.attentionWithRoPE(config, ropeTheta: 10000.0)
        let source = codegen.generate(pattern)

        #expect(source.contains("RoPE"))
        #expect(source.contains("ROPE_THETA"))
        #expect(source.contains("apply_rope"))
    }

    @Test("PatternCodegen generates RMSNorm kernel")
    func patternCodegenRMSNorm() {
        let codegen = PatternCodegen()

        let config = ComputePattern.NormConfig(
            batchSize: 4,
            seqLen: 512,
            hiddenDim: 768,
            isRMSNorm: true
        )
        let pattern = ComputePattern.normalization(config)
        let source = codegen.generate(pattern)

        #expect(source.contains("rmsnorm"))
        #expect(source.contains("rsqrt"))
        #expect(source.contains("simd_sum"))
    }

    @Test("PatternCodegen generates LayerNorm kernel")
    func patternCodegenLayerNorm() {
        let codegen = PatternCodegen()

        let config = ComputePattern.NormConfig(
            batchSize: 4,
            seqLen: 512,
            hiddenDim: 768,
            isRMSNorm: false
        )
        let pattern = ComputePattern.normalization(config)
        let source = codegen.generate(pattern)

        #expect(source.contains("layernorm"))
        #expect(source.contains("mean"))
        #expect(source.contains("bias"))
    }

    @Test("PatternCodegen generates gated FFN kernel")
    func patternCodegenGatedFFN() {
        let codegen = PatternCodegen()

        let config = ComputePattern.FFNConfig(
            batchSize: 2,
            seqLen: 256,
            hiddenDim: 768,
            ffnDim: 3072,
            activation: .silu,
            gated: true
        )
        let pattern = ComputePattern.ffn(config)
        let source = codegen.generate(pattern)

        #expect(source.contains("gated_ffn"))
        #expect(source.contains("gate_weight"))
        #expect(source.contains("up_weight"))
        #expect(source.contains("1.0 / (1.0 + exp"))  // SiLU activation
    }

    @Test("PatternCodegen generates non-gated FFN kernel")
    func patternCodegenFFN() {
        let codegen = PatternCodegen()

        let config = ComputePattern.FFNConfig(
            batchSize: 2,
            seqLen: 256,
            hiddenDim: 768,
            ffnDim: 3072,
            activation: .gelu,
            gated: false
        )
        let pattern = ComputePattern.ffn(config)
        let source = codegen.generate(pattern)

        #expect(source.contains("ffn_"))
        #expect(source.contains("tanh"))  // GELU uses tanh
    }

    @Test("PatternCodegen generates elementwise chain")
    func patternCodegenElementwiseChain() {
        let codegen = PatternCodegen()

        let ops: [ComputePattern.ElementwiseOp] = [
            .mul,
            .exp,
            .scale(2.0)
        ]
        let pattern = ComputePattern.elementwiseChain(ops: ops)
        let source = codegen.generate(pattern)

        #expect(source.contains("fused_elementwise"))
        #expect(source.contains("exp("))
        #expect(source.contains("* 2.0"))
    }

    @Test("PatternCodegen generates transformer block")
    func patternCodegenTransformerBlock() {
        let codegen = PatternCodegen()

        let config = ComputePattern.TransformerConfig(
            batchSize: 1,
            seqLen: 512,
            hiddenDim: 768,
            numHeads: 12,
            ffnDim: 3072
        )
        let pattern = ComputePattern.transformerBlock(config)
        let source = codegen.generate(pattern)

        #expect(source.contains("MegaKernel"))
        #expect(source.contains("transformer_block"))
        #expect(source.contains("rms_norm_inline"))
        #expect(source.contains("silu"))
        #expect(source.contains("ln1_weight"))
        #expect(source.contains("qkv_weight"))
        #expect(source.contains("ffn_gate"))
    }

    @Test("PatternCodegen generates reduction kernel")
    func patternCodegenReduction() {
        let codegen = PatternCodegen()

        let config = ComputePattern.ReductionConfig(
            dimensions: [1024, 768],
            axis: 1,
            operation: .sum
        )
        let pattern = ComputePattern.reduction(config)
        let source = codegen.generate(pattern)

        #expect(source.contains("reduce_sum"))
        #expect(source.contains("simd_sum"))
    }

    // MARK: - MetalEmitter Tests

    @Test("MetalEmitter emits kernel signature")
    func metalEmitterSignature() {
        let emitter = MetalEmitter()

        let kernel = MetalEmitter.KernelDefinition(
            name: "test_kernel",
            parameters: [
                MetalEmitter.KernelParameter(name: "input", type: .float, accessMode: .readonly, bufferIndex: 0),
                MetalEmitter.KernelParameter(name: "output", type: .float, accessMode: .writeonly, bufferIndex: 1)
            ],
            threadIndexing: .minimal,
            body: []
        )

        let source = emitter.emit(kernel)

        #expect(source.contains("kernel void test_kernel"))
        #expect(source.contains("device const float* input"))
        #expect(source.contains("device  float* output"))
        #expect(source.contains("[[buffer(0)]]"))
        #expect(source.contains("[[buffer(1)]]"))
    }

    @Test("MetalEmitter emits shared memory")
    func metalEmitterSharedMemory() {
        let emitter = MetalEmitter()

        let kernel = MetalEmitter.KernelDefinition(
            name: "test_kernel",
            parameters: [],
            threadIndexing: .minimal,
            sharedMemory: [
                MetalEmitter.SharedMemoryDecl(name: "shared_data", type: .float, dimensions: [64, 64])
            ],
            body: []
        )

        let source = emitter.emit(kernel)

        #expect(source.contains("threadgroup float shared_data[64][64]"))
    }

    @Test("MetalEmitter emits constants")
    func metalEmitterConstants() {
        let emitter = MetalEmitter()

        let kernel = MetalEmitter.KernelDefinition(
            name: "test_kernel",
            parameters: [],
            threadIndexing: .minimal,
            constants: [
                MetalEmitter.ConstantDecl(name: "TILE_SIZE", type: .uint, value: "64"),
                MetalEmitter.ConstantDecl(name: "EPSILON", type: .float, value: "1e-5")
            ],
            body: []
        )

        let source = emitter.emit(kernel)

        #expect(source.contains("constant uint TILE_SIZE = 64"))
        #expect(source.contains("constant float EPSILON = 1e-5"))
    }

    @Test("MetalEmitter emits thread indexing")
    func metalEmitterThreadIndexing() {
        let emitter = MetalEmitter()

        let kernel = MetalEmitter.KernelDefinition(
            name: "test_kernel",
            parameters: [],
            threadIndexing: .simdAware,
            body: []
        )

        let source = emitter.emit(kernel)

        #expect(source.contains("uint3 gid [[thread_position_in_grid]]"))
        #expect(source.contains("uint3 tid [[thread_position_in_threadgroup]]"))
        #expect(source.contains("uint simd_lane [[thread_index_in_simdgroup]]"))
    }

    @Test("MetalEmitter convenience factories work")
    func metalEmitterFactories() {
        let kernel = MetalEmitter.elementwiseKernel(
            name: "add_kernel",
            inputCount: 2,
            outputCount: 1,
            dtype: .float,
            body: []
        )

        #expect(kernel.name == "add_kernel")
        #expect(kernel.parameters.count == 3)  // 2 inputs + 1 output
    }

    // MARK: - MetalType Tests

    @Test("MetalType returns correct Metal type names")
    func metalTypeNames() {
        #expect(MetalType.float.metalTypeName == "float")
        #expect(MetalType.half.metalTypeName == "half")
        #expect(MetalType.int.metalTypeName == "int")
        #expect(MetalType.uint.metalTypeName == "uint")
        #expect(MetalType.float4.metalTypeName == "float4")
        #expect(MetalType.half4.metalTypeName == "half4")
    }

    // MARK: - Activation Type Tests

    @Test("Activation types are equatable")
    func activationTypesEquatable() {
        let relu1 = ComputePattern.ActivationType.relu
        let relu2 = ComputePattern.ActivationType.relu
        let gelu = ComputePattern.ActivationType.gelu

        #expect(relu1 == relu2)
        #expect(relu1 != gelu)
    }

    // MARK: - Integration Tests

    @Test("Full codegen pipeline produces valid Metal")
    func fullCodegenPipeline() {
        let codegen = PatternCodegen()

        // Test multiple patterns
        let patterns: [ComputePattern] = [
            .matmul(M: 512, N: 512, K: 512, transA: false, transB: false),
            .attention(ComputePattern.AttentionConfig(
                batchSize: 1, numHeads: 8, seqLenQ: 256, headDim: 64, isCausal: true
            )),
            .normalization(ComputePattern.NormConfig(
                batchSize: 2, seqLen: 128, hiddenDim: 512, isRMSNorm: true
            ))
        ]

        for pattern in patterns {
            let source = codegen.generate(pattern)

            // Basic validation
            #expect(source.contains("#include <metal_stdlib>"))
            #expect(source.contains("using namespace metal"))
            #expect(source.contains("kernel void"))

            // Should have balanced braces
            let openBraces = source.filter { $0 == "{" }.count
            let closeBraces = source.filter { $0 == "}" }.count
            #expect(openBraces == closeBraces)
        }
    }
}
