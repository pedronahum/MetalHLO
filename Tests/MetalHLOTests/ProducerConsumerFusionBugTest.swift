// ProducerConsumerFusionBugTest.swift
// Tests for the fix to the producer-consumer fusion bug that caused
// softmax 4D tensors to produce Inf values.

import Testing
@testable import MetalHLOCore

@Suite("Producer-Consumer Fusion Bug Fix Tests")
struct ProducerConsumerFusionBugFixTests {

    /// Test that reshape-broadcast regions are NOT emitted as fused_elementwise
    @Test("Reshape-broadcast regions should not be fused_elementwise")
    func reshapeBroadcastNotFusedElementwise() throws {
        // Create a function with reshape -> broadcast pattern
        // This pattern was causing the bug when emitted as fused_elementwise
        var broadcastAttrs = HLOAttributes()
        broadcastAttrs.dimensions = [0, 1, 2, 3]

        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "arg0", type: TensorType(shape: [1, 1, 4], elementType: .float32))
            ],
            outputTypes: [TensorType(shape: [1, 1, 4, 4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .reshape,
                    operands: ["arg0"],
                    resultType: TensorType(shape: [1, 1, 4, 1], elementType: .float32)
                ),
                HLOOperation(
                    result: "%1",
                    kind: .broadcastInDim,
                    operands: ["%0"],
                    resultType: TensorType(shape: [1, 1, 4, 4], elementType: .float32),
                    attributes: broadcastAttrs
                )
            ],
            returnValues: ["%1"]
        )

        // Run fusion with emitCustomCalls enabled
        let fusion = ProducerConsumerFusion(maxFusionSize: 50, emitCustomCalls: true)
        let result = fusion.fuse(function)

        // The result should NOT contain a custom_call operation
        // because reshape+broadcast is not a valid fused_elementwise pattern
        let hasCustomCall = result.operations.contains { $0.kind == HLOOpKind.customCall }
        #expect(!hasCustomCall, "Reshape+broadcast should NOT be emitted as custom_call")

        // The operations should be preserved as-is
        #expect(result.operations.count == 2, "Operations should be preserved")
        #expect(result.operations[0].kind == HLOOpKind.reshape)
        #expect(result.operations[1].kind == HLOOpKind.broadcastInDim)
    }

    /// Test that true elementwise chains ARE still fused as custom_call
    @Test("Elementwise chains should still be fused")
    func elementwiseChainsStillFused() throws {
        // Create a function with subtract -> exp -> divide pattern
        // This is a valid fused_elementwise pattern
        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "arg0", type: TensorType(shape: [4, 4], elementType: .float32)),
                HLOArgument(name: "arg1", type: TensorType(shape: [4, 4], elementType: .float32)),
                HLOArgument(name: "arg2", type: TensorType(shape: [4, 4], elementType: .float32))
            ],
            outputTypes: [TensorType(shape: [4, 4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .subtract,
                    operands: ["arg0", "arg1"],
                    resultType: TensorType(shape: [4, 4], elementType: .float32)
                ),
                HLOOperation(
                    result: "%1",
                    kind: .exponential,
                    operands: ["%0"],
                    resultType: TensorType(shape: [4, 4], elementType: .float32)
                ),
                HLOOperation(
                    result: "%2",
                    kind: .divide,
                    operands: ["%1", "arg2"],
                    resultType: TensorType(shape: [4, 4], elementType: .float32)
                )
            ],
            returnValues: ["%2"]
        )

        // Run fusion with emitCustomCalls enabled
        let fusion = ProducerConsumerFusion(maxFusionSize: 50, emitCustomCalls: true)
        let result = fusion.fuse(function)

        // The result SHOULD contain a custom_call operation for the elementwise chain
        let hasCustomCall = result.operations.contains { $0.kind == HLOOpKind.customCall }
        #expect(hasCustomCall, "Elementwise chain should be emitted as custom_call")

        // Should be reduced to a single operation
        #expect(result.operations.count == 1, "Should be fused into single custom_call")
    }

    /// Test that mixed regions (elementwise + reshape) are NOT fused as custom_call
    @Test("Mixed regions with reshape should not be fused")
    func mixedRegionsNotFused() throws {
        // Create a function that mixes reshape with elementwise ops
        let function = HLOFunction(
            name: "test",
            inputs: [
                HLOArgument(name: "arg0", type: TensorType(shape: [16], elementType: .float32)),
                HLOArgument(name: "arg1", type: TensorType(shape: [4, 4], elementType: .float32))
            ],
            outputTypes: [TensorType(shape: [4, 4], elementType: .float32)],
            operations: [
                HLOOperation(
                    result: "%0",
                    kind: .reshape,
                    operands: ["arg0"],
                    resultType: TensorType(shape: [4, 4], elementType: .float32)
                ),
                HLOOperation(
                    result: "%1",
                    kind: .add,
                    operands: ["%0", "arg1"],
                    resultType: TensorType(shape: [4, 4], elementType: .float32)
                )
            ],
            returnValues: ["%1"]
        )

        // Run fusion with emitCustomCalls enabled
        let fusion = ProducerConsumerFusion(maxFusionSize: 50, emitCustomCalls: true)
        let result = fusion.fuse(function)

        // The result should NOT contain a custom_call operation
        // because the region includes reshape which is not elementwise
        let hasCustomCall = result.operations.contains { $0.kind == HLOOpKind.customCall }
        #expect(!hasCustomCall, "Mixed region with reshape should NOT be emitted as custom_call")
    }
}
