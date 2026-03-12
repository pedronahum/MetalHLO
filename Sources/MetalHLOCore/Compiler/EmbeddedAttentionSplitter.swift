// EmbeddedAttentionSplitter.swift
// MetalHLOCore
//
// Detects fused_scaled_dot_product_attention inside multi-op HLO functions and
// splits execution into pre/post MPSGraph sub-graphs with a direct Metal kernel
// for the attention op.

import Metal
@preconcurrency import MetalPerformanceShadersGraph

// MARK: - QKVSource

/// Identifies where a Q, K, or V tensor comes from after the pre-graph executes.
public struct QKVSource: Sendable {
    /// True when this tensor is a direct original function input (no pre-graph needed).
    public let isOriginalInput: Bool
    /// When `isOriginalInput == true`: index into the original executor input array.
    /// When `isOriginalInput == false`: index into the pre-graph's output array.
    public let index: Int
}

// MARK: - EmbeddedAttentionSplit

/// Metadata for splitting a multi-op function around an embedded SDPA attention op.
///
/// Execution proceeds in three stages:
///   1. `preGraph` (MPSGraph, optional): compute Q, K, V tensors
///   2. Flash Attention Metal kernel: Q/K/V → attention output
///   3. `postGraph` (MPSGraph, optional): produce final output from attention + other inputs
/// Declared as a final class to break the recursive struct cycle:
/// `CompiledGraph` (struct) ↔ `EmbeddedAttentionSplit`.
public final class EmbeddedAttentionSplit: @unchecked Sendable {

    // Pre-graph -----------------------------------------------------------

    /// Pre-attention graph. Nil when all Q/K/V are direct function inputs.
    public let preGraph: CompiledGraph?

    /// Maps original function-input indices → preGraph input indices.
    public let originalToPreInputMap: [Int: Int]

    // QKV sources ---------------------------------------------------------

    public let qSource: QKVSource
    public let kSource: QKVSource
    public let vSource: QKVSource

    // Attention shape -----------------------------------------------------

    public let batchSize: Int
    public let numHeads: Int
    public let seqLenQ: Int
    public let seqLenKV: Int
    public let headDim: Int
    public let scale: Float
    public let isCausal: Bool
    public let isFloat16: Bool

    // Post-graph ----------------------------------------------------------

    /// Post-attention graph. Nil when SDPA output is the direct function return.
    public let postGraph: CompiledGraph?

    /// Index of the attention output tensor in `postGraph`'s input list.
    public let postAttentionInputIndex: Int

    /// Maps original function-input indices → postGraph input indices.
    public let originalToPostInputMap: [Int: Int]

    /// Final output types of the whole split computation.
    public let outputTypes: [TensorType]

    init(
        preGraph: CompiledGraph?,
        originalToPreInputMap: [Int: Int],
        qSource: QKVSource,
        kSource: QKVSource,
        vSource: QKVSource,
        batchSize: Int,
        numHeads: Int,
        seqLenQ: Int,
        seqLenKV: Int,
        headDim: Int,
        scale: Float,
        isCausal: Bool,
        isFloat16: Bool,
        postGraph: CompiledGraph?,
        postAttentionInputIndex: Int,
        originalToPostInputMap: [Int: Int],
        outputTypes: [TensorType]
    ) {
        self.preGraph = preGraph
        self.originalToPreInputMap = originalToPreInputMap
        self.qSource = qSource
        self.kSource = kSource
        self.vSource = vSource
        self.batchSize = batchSize
        self.numHeads = numHeads
        self.seqLenQ = seqLenQ
        self.seqLenKV = seqLenKV
        self.headDim = headDim
        self.scale = scale
        self.isCausal = isCausal
        self.isFloat16 = isFloat16
        self.postGraph = postGraph
        self.postAttentionInputIndex = postAttentionInputIndex
        self.originalToPostInputMap = originalToPostInputMap
        self.outputTypes = outputTypes
    }
}

// MARK: - EmbeddedAttentionSplitter

enum EmbeddedAttentionSplitter {

    // MARK: Entry Point

    /// Attempts to detect and split an embedded SDPA op within `function`.
    ///
    /// Returns `nil` when the function has ≤1 op, contains ≠1 SDPA op, or when
    /// the partition is ambiguous (ops shared between pre and post stages).
    static func detectSplit(
        function: HLOFunction,
        device: MTLDevice
    ) -> EmbeddedAttentionSplit? {
        guard function.operations.count > 1 else { return nil }

        // Find exactly one SDPA custom_call
        let sdpaOps = function.operations.filter {
            $0.kind == .customCall &&
            $0.attributes.callTargetName == "fused_scaled_dot_product_attention"
        }
        guard sdpaOps.count == 1, let sdpaOp = sdpaOps.first,
              sdpaOp.operands.count >= 3
        else { return nil }

        let qName = sdpaOp.operands[0]
        let kName = sdpaOp.operands[1]
        let vName = sdpaOp.operands[2]
        let sdpaResult = sdpaOp.result

        let inputNames = function.inputs.map { $0.name }
        let inputNameSet = Set(inputNames)
        let opResultNames = Set(function.operations.map { $0.result })

        // Validate that Q/K/V names exist (either function inputs or op results)
        for name in [qName, kName, vName] {
            guard inputNameSet.contains(name) || opResultNames.contains(name) else { return nil }
        }

        // If all Q/K/V are direct inputs, the single-SDPA shortcut handles it with FlashAttentionShortcut
        // (but that requires 1 op total, which we already filtered above). Skip to avoid confusion.
        let qDirect = inputNameSet.contains(qName)
        let kDirect = inputNameSet.contains(kName)
        let vDirect = inputNameSet.contains(vName)
        if qDirect && kDirect && vDirect { return nil }

        // Build result → op map for the whole function
        let opMap = Dictionary(uniqueKeysWithValues: function.operations.map { ($0.result, $0) })

        // Backward BFS from non-direct Q/K/V sources to find pre-op set
        var preOpSet = Set<String>()
        var bfsQueue: [String] = []
        for name in [qName, kName, vName] where !inputNameSet.contains(name) {
            guard opResultNames.contains(name) else { return nil }
            if preOpSet.insert(name).inserted { bfsQueue.append(name) }
        }
        while !bfsQueue.isEmpty {
            let curr = bfsQueue.removeFirst()
            guard let op = opMap[curr] else { continue }
            for operand in op.operands where !inputNameSet.contains(operand) {
                if preOpSet.insert(operand).inserted { bfsQueue.append(operand) }
            }
        }

        // Forward BFS from SDPA result to find post-op set
        var postOpSet = Set<String>()
        var fwdQueue: [String] = [sdpaResult]
        while !fwdQueue.isEmpty {
            let curr = fwdQueue.removeFirst()
            for op in function.operations where op.operands.contains(curr) {
                if postOpSet.insert(op.result).inserted { fwdQueue.append(op.result) }
            }
        }

        // Reject if any op appears in both sets (would require duplication)
        guard preOpSet.isDisjoint(with: postOpSet) else { return nil }

        // Extract attention metadata
        guard let meta = extractAttentionMetadata(sdpaOp: sdpaOp, function: function) else { return nil }

        // ---- Pre-graph --------------------------------------------------

        let preGraph: CompiledGraph?
        var originalToPreInputMap: [Int: Int] = [:]
        var preReturnValues: [String] = []  // so we can find Q/K/V indices

        if preOpSet.isEmpty {
            preGraph = nil
        } else {
            let preConsumers = SubFunctionExtractor.findExternalConsumers(of: preOpSet, in: function)
            let preFunction = SubFunctionExtractor.extract(
                from: function,
                operationIds: preOpSet,
                outputConsumers: preConsumers
            )
            preReturnValues = preFunction.returnValues

            for (preIdx, arg) in preFunction.inputs.enumerated() {
                if let origIdx = inputNames.firstIndex(of: arg.name) {
                    originalToPreInputMap[origIdx] = preIdx
                }
            }

            let preModule = HLOModule(name: "split_pre", function: preFunction)
            guard let compiled = try? MPSGraphCompiler(device: device).compile(module: preModule)
            else { return nil }
            preGraph = compiled
        }

        // Build QKV sources using preReturnValues index
        func qkvSource(name: String, isDirect: Bool) -> QKVSource {
            if isDirect {
                return QKVSource(isOriginalInput: true, index: inputNames.firstIndex(of: name) ?? 0)
            }
            let idx = preReturnValues.firstIndex(of: name) ?? 0
            return QKVSource(isOriginalInput: false, index: idx)
        }
        let qSource = qkvSource(name: qName, isDirect: qDirect)
        let kSource = qkvSource(name: kName, isDirect: kDirect)
        let vSource = qkvSource(name: vName, isDirect: vDirect)

        // ---- Post-graph -------------------------------------------------

        let postGraph: CompiledGraph?
        var originalToPostInputMap: [Int: Int] = [:]
        var postAttentionInputIndex = 0
        let outputTypes: [TensorType]

        if postOpSet.isEmpty {
            postGraph = nil
            outputTypes = [meta.outputType]
        } else {
            let postConsumers = SubFunctionExtractor.findExternalConsumers(of: postOpSet, in: function)
            let postFunction = SubFunctionExtractor.extract(
                from: function,
                operationIds: postOpSet,
                outputConsumers: postConsumers
            )

            postAttentionInputIndex = postFunction.inputs.firstIndex(where: { $0.name == sdpaResult }) ?? 0

            for (postIdx, arg) in postFunction.inputs.enumerated() {
                if arg.name == sdpaResult { continue }
                if let origIdx = inputNames.firstIndex(of: arg.name) {
                    originalToPostInputMap[origIdx] = postIdx
                }
            }

            let postModule = HLOModule(name: "split_post", function: postFunction)
            guard let compiled = try? MPSGraphCompiler(device: device).compile(module: postModule)
            else { return nil }
            postGraph = compiled
            outputTypes = postFunction.outputTypes
        }

        return EmbeddedAttentionSplit(
            preGraph: preGraph,
            originalToPreInputMap: originalToPreInputMap,
            qSource: qSource,
            kSource: kSource,
            vSource: vSource,
            batchSize: meta.batchSize,
            numHeads: meta.numHeads,
            seqLenQ: meta.seqLenQ,
            seqLenKV: meta.seqLenKV,
            headDim: meta.headDim,
            scale: meta.scale,
            isCausal: meta.isCausal,
            isFloat16: meta.isFloat16,
            postGraph: postGraph,
            postAttentionInputIndex: postAttentionInputIndex,
            originalToPostInputMap: originalToPostInputMap,
            outputTypes: outputTypes
        )
    }

    // MARK: Private helpers

    private struct AttentionMetadata {
        let batchSize, numHeads, seqLenQ, seqLenKV, headDim: Int
        let scale: Float
        let isCausal, isFloat16: Bool
        let outputType: TensorType
    }

    private static func extractAttentionMetadata(
        sdpaOp: HLOOperation,
        function: HLOFunction
    ) -> AttentionMetadata? {
        let shape = sdpaOp.resultType.shape
        guard shape.count == 4 else { return nil }

        // seqLenKV from K operand type
        let kName = sdpaOp.operands[1]
        let kType: TensorType
        if let kArg = function.inputs.first(where: { $0.name == kName }) {
            kType = kArg.type
        } else if let kOp = function.operations.first(where: { $0.result == kName }) {
            kType = kOp.resultType
        } else {
            return nil
        }
        guard kType.shape.count == 4 else { return nil }
        let seqLenKV = kType.shape[2]

        var scale: Float = 1.0 / sqrt(Float(shape[3]))
        var isCausal = false
        if let cfg = sdpaOp.attributes.backendConfig,
           let data = cfg.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let s = json["scale"] as? Double { scale = Float(s) }
            if let c = json["is_causal"] as? Bool { isCausal = c }
        }

        return AttentionMetadata(
            batchSize: shape[0], numHeads: shape[1], seqLenQ: shape[2], seqLenKV: seqLenKV,
            headDim: shape[3], scale: scale, isCausal: isCausal,
            isFloat16: sdpaOp.resultType.elementType == .float16,
            outputType: sdpaOp.resultType
        )
    }
}
