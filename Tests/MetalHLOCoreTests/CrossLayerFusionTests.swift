// CrossLayerFusionTests.swift
// MetalHLOCoreTests
//
// Tests for Phase 2D: Cross-Layer Fusion

import Testing
@testable import MetalHLOCore

@Suite("Cross-Layer Fusion Tests")
struct CrossLayerFusionTests {

    // MARK: - Model Structure Tests

    @Test("TransformerLayer stores layer information")
    func transformerLayerStoresInfo() {
        let layer = TransformerLayer(
            index: 0,
            preAttentionNorm: 0,
            attention: nil,
            postAttentionResidual: 5,
            preFFNNorm: 6,
            ffn: nil,
            postFFNResidual: 12,
            operationIndices: Array(0...12)
        )

        #expect(layer.index == 0)
        #expect(layer.preAttentionNorm == 0)
        #expect(layer.postFFNResidual == 12)
        #expect(layer.operationIndices.count == 13)
    }

    @Test("AttentionBlock stores block information")
    func attentionBlockStoresInfo() {
        let block = AttentionBlock(
            startIndex: 1,
            endIndex: 4,
            qProjection: 1,
            kProjection: 2,
            vProjection: 3,
            sdpa: nil,
            outputProjection: 4,
            hasKVCache: true
        )

        #expect(block.startIndex == 1)
        #expect(block.endIndex == 4)
        #expect(block.qProjection == 1)
        #expect(block.hasKVCache == true)
    }

    @Test("FFNBlock stores block information")
    func ffnBlockStoresInfo() {
        let block = FFNBlock(
            startIndex: 7,
            endIndex: 11,
            upProjection: 7,
            gateProjection: 8,
            activation: 9,
            downProjection: 11,
            isGated: true
        )

        #expect(block.startIndex == 7)
        #expect(block.isGated == true)
        #expect(block.gateProjection == 8)
    }

    @Test("NormBlock stores normalization info")
    func normBlockStoresInfo() {
        let rmsNorm = NormBlock(
            operationIndex: 0,
            isRMSNorm: true,
            hiddenDim: 768
        )

        let layerNorm = NormBlock(
            operationIndex: 5,
            isRMSNorm: false,
            hiddenDim: 1024
        )

        #expect(rmsNorm.isRMSNorm == true)
        #expect(layerNorm.isRMSNorm == false)
        #expect(rmsNorm.hiddenDim == 768)
    }

    @Test("ResidualConnection stores connection info")
    func residualConnectionStoresInfo() {
        let connection = ResidualConnection(
            addOpIndex: 5,
            skipFromIndex: 0,
            skipToIndex: 5
        )

        #expect(connection.addOpIndex == 5)
        #expect(connection.skipFromIndex == 0)
        #expect(connection.skipToIndex == 5)
    }

    @Test("ModelStructure aggregates all blocks")
    func modelStructureAggregatesBlocks() {
        var structure = ModelStructure()

        structure.attentionBlocks.append(AttentionBlock(
            startIndex: 1, endIndex: 4, qProjection: 1, kProjection: 2,
            vProjection: 3, sdpa: nil, outputProjection: 4, hasKVCache: false
        ))

        structure.ffnBlocks.append(FFNBlock(
            startIndex: 7, endIndex: 10, upProjection: 7, gateProjection: nil,
            activation: 8, downProjection: 10, isGated: false
        ))

        structure.normBlocks.append(NormBlock(operationIndex: 0, isRMSNorm: true, hiddenDim: 768))
        structure.normBlocks.append(NormBlock(operationIndex: 6, isRMSNorm: true, hiddenDim: 768))

        #expect(structure.attentionBlocks.count == 1)
        #expect(structure.ffnBlocks.count == 1)
        #expect(structure.normBlocks.count == 2)
    }

    // MARK: - CrossLayerMatch Tests

    @Test("CrossLayerMatch stores pattern information")
    func crossLayerMatchStoresInfo() {
        let match = CrossLayerMatch(
            operationIndices: [0, 1, 2, 3, 4, 5],
            patternType: .transformerBlock,
            metadata: ["layer_index": 0]
        )

        #expect(match.operationIndices.count == 6)
        #expect(match.patternType == .transformerBlock)
        #expect(match.metadata["layer_index"] == 0)
    }

    @Test("CrossLayerPatternType has expected cases")
    func crossLayerPatternTypeHasCases() {
        let types: [CrossLayerPatternType] = [
            .transformerBlock,
            .attentionFFN,
            .residualChain,
            .consecutiveNorms,
            .embeddingPlusFirstLayer,
            .lastLayerPlusOutput,
            .kvCacheUpdate,
            .multiLayerPipeline
        ]

        #expect(types.count == 8)
        #expect(CrossLayerPatternType.transformerBlock.rawValue == "transformerBlock")
    }

    // MARK: - CrossLayerFusionPass Config Tests

    @Test("CrossLayerFusionPass.Config has default values")
    func fusionPassConfigDefaults() {
        let config = CrossLayerFusionPass.Config()

        #expect(config.enableBlockFusion == true)
        #expect(config.enableResidualChainFusion == true)
        #expect(config.enableKVCacheOptimization == true)
        #expect(config.enableMultiLayerPipelining == false)
        #expect(config.pipelineDepth == 2)
        #expect(config.minOpsForFusion == 5)
    }

    @Test("CrossLayerFusionPass initializes with config")
    func fusionPassInitialization() {
        var config = CrossLayerFusionPass.Config()
        config.enableBlockFusion = false
        config.pipelineDepth = 4

        let pass = CrossLayerFusionPass(config: config)

        #expect(pass.name == "cross-layer-fusion")
    }

    // MARK: - CrossLayerFusionPass Tests

    @Test("CrossLayerFusionPass runs on simple function")
    func fusionPassRunsOnSimpleFunction() {
        let pass = CrossLayerFusionPass()

        // Create a simple function
        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "input", type: TensorType(shape: [1, 512, 768], elementType: .float32))],
            outputTypes: [TensorType(shape: [1, 512, 768], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "output",
                    kind: .add,
                    operands: ["input", "input"],
                    resultType: TensorType(shape: [1, 512, 768], elementType: .float32)
                )
            ],
            returnValues: ["output"]
        )

        let result = pass.run(on: function)

        // Should not crash and return a valid function
        #expect(result.name == "test")
        #expect(result.operations.count >= 1)
    }

    @Test("CrossLayerFusionPass detects add operations")
    func fusionPassDetectsAddOps() {
        let pass = CrossLayerFusionPass()

        // Create a function with multiple adds (potential residual chain)
        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "input", type: TensorType(shape: [1, 512, 768], elementType: .float32))],
            outputTypes: [TensorType(shape: [1, 512, 768], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "add1",
                    kind: .add,
                    operands: ["input", "input"],
                    resultType: TensorType(shape: [1, 512, 768], elementType: .float32)
                ),
                HLOOperation(
                    result: "output",
                    kind: .add,
                    operands: ["add1", "input"],
                    resultType: TensorType(shape: [1, 512, 768], elementType: .float32)
                )
            ],
            returnValues: ["output"]
        )

        let result = pass.run(on: function)

        // Should not crash
        #expect(result.name == "test")
    }

    // MARK: - MultiLayerPipeliningPass Tests

    @Test("MultiLayerPipeliningPass initializes with depth")
    func pipeliningPassInitialization() {
        let pass2 = MultiLayerPipeliningPass(pipelineDepth: 2)
        let pass4 = MultiLayerPipeliningPass(pipelineDepth: 4)

        #expect(pass2.pipelineDepth == 2)
        #expect(pass4.pipelineDepth == 4)
        #expect(pass2.name == "multi-layer-pipelining")
    }

    @Test("MultiLayerPipeliningPass creates pipelined groups")
    func pipeliningPassCreatesGroups() {
        let pass = MultiLayerPipeliningPass(pipelineDepth: 2)

        let layers = [
            TransformerLayer(index: 0, preAttentionNorm: nil, attention: nil,
                           postAttentionResidual: nil, preFFNNorm: nil,
                           ffn: nil, postFFNResidual: nil, operationIndices: [0, 1, 2]),
            TransformerLayer(index: 1, preAttentionNorm: nil, attention: nil,
                           postAttentionResidual: nil, preFFNNorm: nil,
                           ffn: nil, postFFNResidual: nil, operationIndices: [3, 4, 5]),
            TransformerLayer(index: 2, preAttentionNorm: nil, attention: nil,
                           postAttentionResidual: nil, preFFNNorm: nil,
                           ffn: nil, postFFNResidual: nil, operationIndices: [6, 7, 8]),
            TransformerLayer(index: 3, preAttentionNorm: nil, attention: nil,
                           postAttentionResidual: nil, preFFNNorm: nil,
                           ffn: nil, postFFNResidual: nil, operationIndices: [9, 10, 11])
        ]

        let function = HLOFunction(
            name: "test",
            inputs: [],
            outputTypes: [],
            operations: [],
            returnValues: []
        )

        let result = pass.createPipelinedLayers(layers, function: function)

        #expect(result != nil)
        #expect(result?.stages.count == 2)  // 4 layers / 2 depth = 2 stages
        #expect(result?.totalLayers == 4)
    }

    @Test("MultiLayerPipeliningPass returns nil for insufficient layers")
    func pipeliningPassReturnsNilForFewLayers() {
        let pass = MultiLayerPipeliningPass(pipelineDepth: 4)

        let layers = [
            TransformerLayer(index: 0, preAttentionNorm: nil, attention: nil,
                           postAttentionResidual: nil, preFFNNorm: nil,
                           ffn: nil, postFFNResidual: nil, operationIndices: [0])
        ]

        let function = HLOFunction(name: "test", inputs: [], outputTypes: [], operations: [], returnValues: [])
        let result = pass.createPipelinedLayers(layers, function: function)

        #expect(result == nil)
    }

    // MARK: - PipelinedLayerGroup Tests

    @Test("PipelinedLayerGroup calculates total layers")
    func pipelinedLayerGroupTotalLayers() {
        let stages: [[TransformerLayer]] = [
            [
                TransformerLayer(index: 0, preAttentionNorm: nil, attention: nil,
                               postAttentionResidual: nil, preFFNNorm: nil,
                               ffn: nil, postFFNResidual: nil, operationIndices: []),
                TransformerLayer(index: 1, preAttentionNorm: nil, attention: nil,
                               postAttentionResidual: nil, preFFNNorm: nil,
                               ffn: nil, postFFNResidual: nil, operationIndices: [])
            ],
            [
                TransformerLayer(index: 2, preAttentionNorm: nil, attention: nil,
                               postAttentionResidual: nil, preFFNNorm: nil,
                               ffn: nil, postFFNResidual: nil, operationIndices: [])
            ]
        ]

        let group = PipelinedLayerGroup(stages: stages, pipelineDepth: 2)

        #expect(group.totalLayers == 3)
        #expect(group.stages.count == 2)
        #expect(group.pipelineDepth == 2)
    }

    // MARK: - KVCacheOptimizer Tests

    @Test("KVCacheOptimizer initializes correctly")
    func kvCacheOptimizerInitialization() {
        let optimizer = KVCacheOptimizer()

        #expect(optimizer.name == "kv-cache-optimizer")
    }

    @Test("KVCacheOptimizer runs on simple function")
    func kvCacheOptimizerRunsOnFunction() {
        let optimizer = KVCacheOptimizer()

        let function = HLOFunction(
            name: "test",
            inputs: [HLOArgument(name: "input", type: TensorType(shape: [1, 512, 768], elementType: .float32))],
            outputTypes: [TensorType(shape: [1, 512, 768], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "output",
                    kind: .add,
                    operands: ["input", "input"],
                    resultType: TensorType(shape: [1, 512, 768], elementType: .float32)
                )
            ],
            returnValues: ["output"]
        )

        let result = optimizer.optimize(function)

        #expect(result.name == "test")
    }

    @Test("KVCacheOptimizer detects dynamic update slice")
    func kvCacheOptimizerDetectsDynamicUpdateSlice() {
        let optimizer = KVCacheOptimizer()

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "cache", type: TensorType(shape: [1, 32, 2048, 64], elementType: .float32)),
                HLOArgument(name: "update", type: TensorType(shape: [1, 32, 1, 64], elementType: .float32))
            ],
            outputTypes: [TensorType(shape: [1, 32, 2048, 64], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "key_cache_updated",
                    kind: .dynamicUpdateSlice,
                    operands: ["cache", "update"],
                    resultType: TensorType(shape: [1, 32, 2048, 64], elementType: .float32)
                )
            ],
            returnValues: ["key_cache_updated"]
        )

        let result = optimizer.optimize(function)

        // Should not crash and return a valid function
        #expect(result.operations.count >= 1)
    }

    // MARK: - Integration Tests

    @Test("Full cross-layer optimization pipeline")
    func fullCrossLayerPipeline() {
        // Create a mini transformer-like function
        let input = TensorType(shape: [1, 512, 768], elementType: .float32)

        // Helper to create attributes with callTargetName
        func makeCustomCallAttrs(_ name: String) -> HLOAttributes {
            var attrs = HLOAttributes()
            attrs.callTargetName = name
            return attrs
        }

        let function = HLOFunction(
            name: "mini_transformer",
            inputs: [HLOArgument(name: "input", type: input)],
            outputTypes: [input],
            operations: [
                // Pre-attention norm (simplified)
                HLOOperation(
                    result: "normed",
                    kind: .customCall,
                    operands: ["input"],
                    resultType: input,
                    attributes: makeCustomCallAttrs("rms_norm")
                ),
                // Q projection (dot)
                HLOOperation(
                    result: "q",
                    kind: .dot,
                    operands: ["normed", "qkv_weight"],
                    resultType: input
                ),
                // Add (residual)
                HLOOperation(
                    result: "residual1",
                    kind: .add,
                    operands: ["input", "q"],
                    resultType: input
                ),
                // Pre-FFN norm
                HLOOperation(
                    result: "normed2",
                    kind: .customCall,
                    operands: ["residual1"],
                    resultType: input,
                    attributes: makeCustomCallAttrs("rms_norm")
                ),
                // FFN up
                HLOOperation(
                    result: "ffn_hidden",
                    kind: .dot,
                    operands: ["normed2", "ffn_up"],
                    resultType: input
                ),
                // Activation
                HLOOperation(
                    result: "activated",
                    kind: .customCall,
                    operands: ["ffn_hidden"],
                    resultType: input,
                    attributes: makeCustomCallAttrs("silu")
                ),
                // FFN down
                HLOOperation(
                    result: "ffn_out",
                    kind: .dot,
                    operands: ["activated", "ffn_down"],
                    resultType: input
                ),
                // Final residual
                HLOOperation(
                    result: "output",
                    kind: .add,
                    operands: ["residual1", "ffn_out"],
                    resultType: input
                )
            ],
            returnValues: ["output"]
        )

        let pass = CrossLayerFusionPass()
        let result = pass.run(on: function)

        // Should process without errors
        #expect(result.name == "mini_transformer")
    }
}
