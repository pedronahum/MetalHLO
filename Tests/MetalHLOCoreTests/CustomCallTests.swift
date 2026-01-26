// CustomCallTests.swift
// MetalHLOCoreTests
//
// Tests for custom call handlers that implement fused operations.

import Testing
@testable import MetalHLOCore

// MARK: - Custom Call Registry Tests

@Suite("CustomCallRegistry Tests")
struct CustomCallRegistryTests {

    @Test("Registry registers default handlers")
    func registersDefaultHandlers() {
        let registry = CustomCallRegistry.shared

        #expect(registry.isSupported("fused_scaled_dot_product_attention"))
        #expect(registry.isSupported("fused_layer_norm"))
        #expect(registry.isSupported("fused_rms_norm"))
        #expect(registry.isSupported("fused_matmul_bias_activation"))
        #expect(registry.isSupported("fused_softmax"))
        #expect(registry.isSupported("fused_gelu"))
        #expect(registry.isSupported("fused_rope"))
    }

    @Test("Registry returns handlers")
    func returnsHandlers() {
        let registry = CustomCallRegistry.shared

        let attentionHandler = registry.handler(for: "fused_scaled_dot_product_attention")
        #expect(attentionHandler != nil)

        let layerNormHandler = registry.handler(for: "fused_layer_norm")
        #expect(layerNormHandler != nil)
    }

    @Test("Registry returns nil for unknown target")
    func returnsNilForUnknown() {
        let registry = CustomCallRegistry.shared

        let handler = registry.handler(for: "nonexistent_operation")
        #expect(handler == nil)
        #expect(!registry.isSupported("nonexistent_operation"))
    }

    @Test("Registry lists supported targets")
    func listsSupportedTargets() {
        let registry = CustomCallRegistry.shared
        let targets = registry.supportedTargets

        #expect(targets.count >= 7)  // At least the default handlers
        #expect(targets.contains("fused_layer_norm"))
        #expect(targets.contains("fused_softmax"))
    }
}

// MARK: - Backend Config Parser Tests

@Suite("BackendConfigParser Tests")
struct BackendConfigParserTests {

    @Test("Parse valid JSON config")
    func parseValidJson() {
        let json = """
        {"eps": 1e-5, "axis": -1, "is_causal": true}
        """

        let config = BackendConfigParser.parse(json)

        #expect(config.count == 3)
    }

    @Test("Parse returns empty dict for invalid JSON")
    func parseInvalidJson() {
        let config = BackendConfigParser.parse("not valid json")
        #expect(config.isEmpty)
    }

    @Test("Parse returns empty dict for empty string")
    func parseEmptyString() {
        let config = BackendConfigParser.parse("")
        #expect(config.isEmpty)
    }

    @Test("getFloat extracts float values")
    func getFloatValue() {
        let config: [String: Any] = [
            "eps": 0.00001,
            "scale": 0.125,
            "intValue": 5
        ]

        #expect(BackendConfigParser.getFloat(config, key: "eps", default: 1.0) == Float(0.00001))
        #expect(BackendConfigParser.getFloat(config, key: "scale", default: 1.0) == Float(0.125))
        #expect(BackendConfigParser.getFloat(config, key: "intValue", default: 1.0) == Float(5.0))
        #expect(BackendConfigParser.getFloat(config, key: "missing", default: 99.0) == Float(99.0))
    }

    @Test("getInt extracts int values")
    func getIntValue() {
        let config: [String: Any] = [
            "axis": -1,
            "size": 512,
            "floatValue": 3.0
        ]

        #expect(BackendConfigParser.getInt(config, key: "axis", default: 0) == -1)
        #expect(BackendConfigParser.getInt(config, key: "size", default: 0) == 512)
        #expect(BackendConfigParser.getInt(config, key: "floatValue", default: 0) == 3)
        #expect(BackendConfigParser.getInt(config, key: "missing", default: 42) == 42)
    }

    @Test("getBool extracts bool values")
    func getBoolValue() {
        let config: [String: Any] = [
            "is_causal": true,
            "has_mask": false,
            "intTrue": 1,
            "intFalse": 0
        ]

        #expect(BackendConfigParser.getBool(config, key: "is_causal", default: false) == true)
        #expect(BackendConfigParser.getBool(config, key: "has_mask", default: true) == false)
        #expect(BackendConfigParser.getBool(config, key: "intTrue", default: false) == true)
        #expect(BackendConfigParser.getBool(config, key: "intFalse", default: true) == false)
        #expect(BackendConfigParser.getBool(config, key: "missing", default: true) == true)
    }

    @Test("getString extracts string values")
    func getStringValue() {
        let config: [String: Any] = [
            "activation": "relu",
            "name": "test"
        ]

        #expect(BackendConfigParser.getString(config, key: "activation", default: "none") == "relu")
        #expect(BackendConfigParser.getString(config, key: "name", default: "") == "test")
        #expect(BackendConfigParser.getString(config, key: "missing", default: "default") == "default")
    }

    @Test("getIntArray extracts int arrays")
    func getIntArrayValue() {
        let config: [String: Any] = [
            "axes": [-1],
            "dims": [0, 1, 2],
            "floatDims": [1.0, 2.0, 3.0]
        ]

        #expect(BackendConfigParser.getIntArray(config, key: "axes", default: []) == [-1])
        #expect(BackendConfigParser.getIntArray(config, key: "dims", default: []) == [0, 1, 2])
        #expect(BackendConfigParser.getIntArray(config, key: "floatDims", default: []) == [1, 2, 3])
        #expect(BackendConfigParser.getIntArray(config, key: "missing", default: [99]) == [99])
    }
}

// MARK: - Custom Call Error Tests

@Suite("CustomCallError Tests")
struct CustomCallErrorTests {

    @Test("Error descriptions are informative")
    func errorDescriptions() {
        let inputCountError = CustomCallError.invalidInputCount(expected: 3, got: 2)
        #expect(inputCountError.description.contains("expected 3"))
        #expect(inputCountError.description.contains("got 2"))

        let unsupportedError = CustomCallError.unsupportedTarget("unknown_op")
        #expect(unsupportedError.description.contains("unknown_op"))

        let configError = CustomCallError.invalidConfig("missing eps")
        #expect(configError.description.contains("missing eps"))

        let emissionError = CustomCallError.emissionFailed("GPU error")
        #expect(emissionError.description.contains("GPU error"))
    }
}

// MARK: - Handler Target Name Tests

@Suite("Handler Target Names Tests")
struct HandlerTargetNamesTests {

    @Test("FusedScaledDotProductAttentionHandler target name")
    func attentionTargetName() {
        #expect(FusedScaledDotProductAttentionHandler.targetName == "fused_scaled_dot_product_attention")
    }

    @Test("FusedLayerNormHandler target name")
    func layerNormTargetName() {
        #expect(FusedLayerNormHandler.targetName == "fused_layer_norm")
    }

    @Test("FusedRMSNormHandler target name")
    func rmsNormTargetName() {
        #expect(FusedRMSNormHandler.targetName == "fused_rms_norm")
    }

    @Test("FusedMatMulBiasActivationHandler target name")
    func matmulBiasActivationTargetName() {
        #expect(FusedMatMulBiasActivationHandler.targetName == "fused_matmul_bias_activation")
    }

    @Test("FusedSoftmaxHandler target name")
    func softmaxTargetName() {
        #expect(FusedSoftmaxHandler.targetName == "fused_softmax")
    }

    @Test("FusedGeluHandler target name")
    func geluTargetName() {
        #expect(FusedGeluHandler.targetName == "fused_gelu")
    }

    @Test("FusedRoPEHandler target name")
    func ropeTargetName() {
        #expect(FusedRoPEHandler.targetName == "fused_rope")
    }
}

// MARK: - Handler Initialization Tests

@Suite("Handler Initialization Tests")
struct HandlerInitializationTests {

    @Test("All handlers can be initialized")
    func handlersInitialize() {
        // Verify all handlers initialize without crashing
        let _ = FusedScaledDotProductAttentionHandler()
        let _ = FusedLayerNormHandler()
        let _ = FusedRMSNormHandler()
        let _ = FusedMatMulBiasActivationHandler()
        let _ = FusedSoftmaxHandler()
        let _ = FusedGeluHandler()
        let _ = FusedRoPEHandler()

        // If we get here, all initializations succeeded
        #expect(true)
    }
}
