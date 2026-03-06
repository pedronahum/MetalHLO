// CoreMLOpBuilder.swift
// MetalHLOCore
//
// Converts HLO functions to CoreMLOp arrays for the CoreML compilation path.
// This bridges MetalHLOCore's IR to ANERuntime's CoreMLBridge.

import Foundation
import ANERuntime

/// Converts an HLO function into CoreML operation descriptors.
///
/// Produces the same translations as `MILOpTranslator` but outputs
/// structured `CoreMLOp` arrays instead of MIL text. These can be
/// passed directly to `CoreMLBridge.compile()`.
public struct CoreMLOpBuilder {

    public init() {}

    /// Ensures a name is a valid CoreML MIL identifier.
    /// CoreML rejects names that start with a digit.
    private func coreMLName(_ hloName: String) -> String {
        let name = MILTypeMapper.sanitizeName(hloName)
        if let first = name.first, first.isNumber {
            return "v\(name)"
        }
        return name
    }

    /// Converts an HLO function to CoreML operations.
    ///
    /// - Parameter function: The HLO function to convert.
    /// - Returns: Inputs, operations, and return variable name.
    public func build(function: HLOFunction) throws -> (
        inputs: [(name: String, shape: [Int])],
        operations: [CoreMLOp],
        returnVar: String
    ) {
        var ops: [CoreMLOp] = []
        var operandShapes: [String: [Int]] = [:]

        // Map inputs
        let inputs = function.inputs.map { arg in
            let name = coreMLName(arg.name)
            operandShapes[arg.name] = arg.type.shape
            return (name: name, shape: arg.type.shape)
        }

        // Translate operations
        for op in function.operations {
            let translated = try translateOp(op, operandShapes: &operandShapes)
            ops.append(contentsOf: translated)
        }

        // Return variable
        guard let returnValue = function.returnValues.first else {
            throw MILEmitterError.internalError("Function has no return values")
        }
        let returnVar = coreMLName(returnValue)

        return (inputs, ops, returnVar)
    }

    // MARK: - Fusion-Aware Build

    /// Converts an HLO function to CoreML operations with pattern fusion.
    ///
    /// Detected patterns (LayerNorm, GELU, SiLU, etc.) are emitted as single
    /// fused CoreML ops instead of decomposed element-wise chains. This lets
    /// the ANE execute longer fused graphs in its internal SRAM.
    ///
    /// - Parameters:
    ///   - function: The HLO function to convert.
    ///   - patterns: Detected patterns from `Analyzer.detectPatterns()`.
    /// - Returns: Inputs, operations, and return variable name.
    public func buildWithFusion(
        function: HLOFunction,
        patterns: [DetectedPattern]
    ) throws -> (
        inputs: [(name: String, shape: [Int])],
        operations: [CoreMLOp],
        returnVar: String
    ) {
        let rewritten = MILFusionRewriter.rewrite(function: function, patterns: patterns)

        var ops: [CoreMLOp] = []
        var operandShapes: [String: [Int]] = [:]

        // Map inputs
        let inputs = function.inputs.map { arg in
            let name = coreMLName(arg.name)
            operandShapes[arg.name] = arg.type.shape
            return (name: name, shape: arg.type.shape)
        }

        // Translate operations (fused or original)
        for entry in rewritten {
            switch entry {
            case .original(let op):
                let translated = try translateOp(op, operandShapes: &operandShapes)
                ops.append(contentsOf: translated)

            case .fused(let pattern, let rootOp, let consumedOps):
                let translated = try translateFusedPattern(
                    pattern: pattern,
                    rootOp: rootOp,
                    consumedOps: consumedOps,
                    operandShapes: &operandShapes
                )
                ops.append(contentsOf: translated)
            }
        }

        // Return variable
        guard let returnValue = function.returnValues.first else {
            throw MILEmitterError.internalError("Function has no return values")
        }
        let returnVar = coreMLName(returnValue)

        return (inputs, ops, returnVar)
    }

    // MARK: - Fused Pattern Translation

    private func translateFusedPattern(
        pattern: DetectedPattern,
        rootOp: HLOOperation,
        consumedOps: [HLOOperation],
        operandShapes: inout [String: [Int]]
    ) throws -> [CoreMLOp] {
        let resultName = coreMLName(rootOp.result)
        let outputShape = rootOp.resultType.shape
        operandShapes[rootOp.result] = outputShape

        // Also register consumed op shapes so downstream ops can reference them
        for consumed in consumedOps {
            operandShapes[consumed.result] = consumed.resultType.shape
        }

        switch pattern.type {
        case .layerNorm:
            return translateFusedLayerNorm(
                resultName: resultName, outputShape: outputShape,
                rootOp: rootOp, consumedOps: consumedOps,
                metadata: pattern.metadata, operandShapes: &operandShapes
            )

        case .rmsNorm:
            return translateFusedRMSNorm(
                resultName: resultName, outputShape: outputShape,
                rootOp: rootOp, consumedOps: consumedOps,
                metadata: pattern.metadata, operandShapes: &operandShapes
            )

        case .gelu:
            return translateFusedGELU(
                resultName: resultName, outputShape: outputShape,
                rootOp: rootOp, operandShapes: &operandShapes
            )

        case .silu:
            return translateFusedSiLU(
                resultName: resultName, outputShape: outputShape,
                rootOp: rootOp, consumedOps: consumedOps,
                operandShapes: &operandShapes
            )

        case .matmulBiasActivation:
            return translateFusedMatMulBiasAct(
                resultName: resultName, outputShape: outputShape,
                rootOp: rootOp, consumedOps: consumedOps,
                metadata: pattern.metadata, operandShapes: &operandShapes
            )

        case .ffn:
            return translateFusedFFN(
                resultName: resultName, outputShape: outputShape,
                rootOp: rootOp, consumedOps: consumedOps,
                metadata: pattern.metadata, operandShapes: &operandShapes
            )

        default:
            // Unsupported fusion — fall back to original translation
            return try translateOp(rootOp, operandShapes: &operandShapes)
        }
    }

    // MARK: - LayerNorm Fusion

    private func translateFusedLayerNorm(
        resultName: String, outputShape: [Int],
        rootOp: HLOOperation, consumedOps: [HLOOperation],
        metadata: PatternMetadata,
        operandShapes: inout [String: [Int]]
    ) -> [CoreMLOp] {
        // Find the input: first operand of the earliest consumed op, or root's first operand
        let inputName: String
        if let earliest = consumedOps.sorted(by: { $0.result < $1.result }).first {
            inputName = coreMLName(earliest.operands[0])
        } else {
            inputName = coreMLName(rootOp.operands[0])
        }

        let epsilon = metadata.epsilon ?? 1e-5
        let axes = [outputShape.count - 1]  // Normalize over last axis

        return [CoreMLOp(
            result: resultName, op: "layer_norm", shape: outputShape,
            params: [
                ("x", .variable(inputName)),
                ("axes", .intArray(axes)),
                ("epsilon", .floatValue(Double(epsilon))),
            ]
        )]
    }

    // MARK: - RMSNorm Fusion

    private func translateFusedRMSNorm(
        resultName: String, outputShape: [Int],
        rootOp: HLOOperation, consumedOps: [HLOOperation],
        metadata: PatternMetadata,
        operandShapes: inout [String: [Int]]
    ) -> [CoreMLOp] {
        // RMSNorm pattern: reduce → rsqrt → mul
        // The original input is the first operand of the reduce op
        let inputName: String
        if let reduceOp = consumedOps.first(where: { $0.kind == .reduce }) {
            inputName = coreMLName(reduceOp.operands[0])
        } else {
            inputName = coreMLName(rootOp.operands[0])
        }

        let epsilon = metadata.epsilon ?? 1e-5
        let axes = [outputShape.count - 1]

        // CoreML doesn't have native rms_norm, so emit: x * rsqrt(reduce_mean(x^2) + eps)
        // But we can use layer_norm with no bias/gamma as a close approximation,
        // or emit the decomposed but still fused sequence
        let sqName = resultName + "_sq"
        let meanName = resultName + "_mean"
        let epsName = resultName + "_eps"
        let addEpsName = resultName + "_add_eps"
        let rsqrtName = resultName + "_rsqrt"

        return [
            // x^2
            CoreMLOp(result: sqName, op: "mul", shape: outputShape,
                     params: [("x", .variable(inputName)), ("y", .variable(inputName))]),
            // mean(x^2, axis=-1, keepdims=true)
            CoreMLOp(result: meanName, op: "reduce_mean", shape: outputShape,
                     params: [("x", .variable(sqName)), ("axes", .intArray(axes)), ("keep_dims", .boolValue(true))]),
            // epsilon constant
            CoreMLOp(result: epsName, op: "const", shape: [],
                     params: [("val", .floatValue(Double(epsilon)))]),
            // mean + epsilon
            CoreMLOp(result: addEpsName, op: "add", shape: outputShape,
                     params: [("x", .variable(meanName)), ("y", .variable(epsName))]),
            // rsqrt(mean + epsilon)
            CoreMLOp(result: rsqrtName, op: "rsqrt", shape: outputShape,
                     params: [("x", .variable(addEpsName))]),
            // x * rsqrt(...)
            CoreMLOp(result: resultName, op: "mul", shape: outputShape,
                     params: [("x", .variable(inputName)), ("y", .variable(rsqrtName))]),
        ]
    }

    // MARK: - GELU Fusion

    private func translateFusedGELU(
        resultName: String, outputShape: [Int],
        rootOp: HLOOperation,
        operandShapes: inout [String: [Int]]
    ) -> [CoreMLOp] {
        let inputName = coreMLName(rootOp.operands[0])

        return [CoreMLOp(
            result: resultName, op: "gelu", shape: outputShape,
            params: [("x", .variable(inputName))]
        )]
    }

    // MARK: - SiLU Fusion

    private func translateFusedSiLU(
        resultName: String, outputShape: [Int],
        rootOp: HLOOperation, consumedOps: [HLOOperation],
        operandShapes: inout [String: [Int]]
    ) -> [CoreMLOp] {
        // SiLU = x * sigmoid(x)
        // The root is the multiply, consumed is the sigmoid
        let inputName: String
        if let sigmoidOp = consumedOps.first(where: { $0.kind == .logistic }) {
            inputName = coreMLName(sigmoidOp.operands[0])
        } else {
            inputName = coreMLName(rootOp.operands[0])
        }

        let sigmoidName = resultName + "_sigmoid"
        return [
            CoreMLOp(result: sigmoidName, op: "sigmoid", shape: outputShape,
                     params: [("x", .variable(inputName))]),
            CoreMLOp(result: resultName, op: "mul", shape: outputShape,
                     params: [("x", .variable(inputName)), ("y", .variable(sigmoidName))]),
        ]
    }

    // MARK: - MatMul+Bias+Activation Fusion

    private func translateFusedMatMulBiasAct(
        resultName: String, outputShape: [Int],
        rootOp: HLOOperation, consumedOps: [HLOOperation],
        metadata: PatternMetadata,
        operandShapes: inout [String: [Int]]
    ) -> [CoreMLOp] {
        // Find the matmul in consumed ops or root
        let matmulOp = consumedOps.first(where: { $0.kind == .dot || $0.kind == .dotGeneral }) ?? rootOp
        let lhsName = coreMLName(matmulOp.operands[0])
        let rhsName = coreMLName(matmulOp.operands[1])

        var ops: [CoreMLOp] = []

        // Emit matmul
        let matmulResultName = (rootOp.kind == .dot || rootOp.kind == .dotGeneral)
            ? resultName : resultName + "_matmul"
        let matmulShape = matmulOp.resultType.shape

        ops.append(CoreMLOp(
            result: matmulResultName, op: "matmul", shape: matmulShape,
            params: [
                ("x", .variable(lhsName)),
                ("y", .variable(rhsName)),
                ("transpose_x", .boolValue(false)),
                ("transpose_y", .boolValue(false)),
            ]
        ))

        // If root is not the matmul, check for bias add
        if rootOp.kind == .add {
            let biasOperand = rootOp.operands.first(where: { $0 != matmulOp.result })
            let biasName = coreMLName(biasOperand ?? rootOp.operands[1])
            ops.append(CoreMLOp(
                result: resultName, op: "add", shape: outputShape,
                params: [("x", .variable(matmulResultName)), ("y", .variable(biasName))]
            ))
        }

        return ops
    }

    // MARK: - FFN Fusion

    private func translateFusedFFN(
        resultName: String, outputShape: [Int],
        rootOp: HLOOperation, consumedOps: [HLOOperation],
        metadata: PatternMetadata,
        operandShapes: inout [String: [Int]]
    ) -> [CoreMLOp] {
        // FFN pattern: matmul → activation → matmul
        // Emit two linear ops with activation between them.
        // This keeps them in a single CoreML program for deep-chain ANE benefit.
        let allOps = (consumedOps + [rootOp]).sorted { a, b in
            // Sort by result name to get topological order
            a.result < b.result
        }

        // Find matmul ops and activation
        let matmuls = allOps.filter { $0.kind == .dot || $0.kind == .dotGeneral }

        if matmuls.count >= 2 {
            let firstMatmul = matmuls[0]
            let secondMatmul = matmuls[1]

            var ops: [CoreMLOp] = []

            // First matmul
            let firstResultName = resultName + "_ffn_fc1"
            ops.append(CoreMLOp(
                result: firstResultName, op: "matmul",
                shape: firstMatmul.resultType.shape,
                params: [
                    ("x", .variable(coreMLName(firstMatmul.operands[0]))),
                    ("y", .variable(coreMLName(firstMatmul.operands[1]))),
                    ("transpose_x", .boolValue(false)),
                    ("transpose_y", .boolValue(false)),
                ]
            ))

            // Activation (if any)
            let activationName = resultName + "_ffn_act"
            let activationType = metadata.activation ?? "gelu"
            let milActivation: String
            switch activationType {
            case "relu": milActivation = "relu"
            case "silu", "swish": milActivation = "silu"
            default: milActivation = "gelu"
            }

            // For silu, we need sigmoid + mul
            if milActivation == "silu" {
                let sigName = activationName + "_sig"
                ops.append(CoreMLOp(
                    result: sigName, op: "sigmoid",
                    shape: firstMatmul.resultType.shape,
                    params: [("x", .variable(firstResultName))]
                ))
                ops.append(CoreMLOp(
                    result: activationName, op: "mul",
                    shape: firstMatmul.resultType.shape,
                    params: [("x", .variable(firstResultName)), ("y", .variable(sigName))]
                ))
            } else {
                ops.append(CoreMLOp(
                    result: activationName, op: milActivation,
                    shape: firstMatmul.resultType.shape,
                    params: [("x", .variable(firstResultName))]
                ))
            }

            // Second matmul
            ops.append(CoreMLOp(
                result: resultName, op: "matmul", shape: outputShape,
                params: [
                    ("x", .variable(activationName)),
                    ("y", .variable(coreMLName(secondMatmul.operands[1]))),
                    ("transpose_x", .boolValue(false)),
                    ("transpose_y", .boolValue(false)),
                ]
            ))

            return ops
        }

        // Fallback: emit ops individually
        var ops: [CoreMLOp] = []
        for op in allOps {
            let translated = try? translateOp(op, operandShapes: &operandShapes)
            if let translated = translated {
                ops.append(contentsOf: translated)
            }
        }
        return ops
    }

    // MARK: - Operation Translation

    private func translateOp(
        _ op: HLOOperation,
        operandShapes: inout [String: [Int]]
    ) throws -> [CoreMLOp] {
        let resultName = coreMLName(op.result)
        let outputShape = op.resultType.shape

        switch op.kind {
        // Constants
        case .constant:
            let ops = try translateConstant(op: op, operandShapes: &operandShapes)
            return ops

        // Binary arithmetic
        case .add:
            return [makeBinary(op, milOp: "add", operandShapes: &operandShapes)]
        case .subtract:
            return [makeBinary(op, milOp: "sub", operandShapes: &operandShapes)]
        case .multiply:
            return [makeBinary(op, milOp: "mul", operandShapes: &operandShapes)]
        case .divide:
            return [makeBinary(op, milOp: "real_div", operandShapes: &operandShapes)]
        case .maximum:
            return [makeBinary(op, milOp: "maximum", operandShapes: &operandShapes)]
        case .minimum:
            return [makeBinary(op, milOp: "minimum", operandShapes: &operandShapes)]
        case .power:
            return [makeBinary(op, milOp: "pow", operandShapes: &operandShapes)]

        // Unary ops
        case .abs:
            return [makeUnary(op, milOp: "abs", operandShapes: &operandShapes)]
        case .exponential:
            return [makeUnary(op, milOp: "exp", operandShapes: &operandShapes)]
        case .tanh:
            return [makeUnary(op, milOp: "tanh", operandShapes: &operandShapes)]
        case .logistic:
            return [makeUnary(op, milOp: "sigmoid", operandShapes: &operandShapes)]
        case .log:
            return [makeUnary(op, milOp: "log", operandShapes: &operandShapes)]
        case .sqrt:
            return [makeUnary(op, milOp: "sqrt", operandShapes: &operandShapes)]
        case .rsqrt:
            return [makeUnary(op, milOp: "rsqrt", operandShapes: &operandShapes)]

        // Negate: MIL has no neg, use mul(x, -1)
        case .negate:
            let inputName = coreMLName(op.operands[0])
            let constName = resultName + "_neg_one"
            operandShapes[op.result] = outputShape
            return [
                CoreMLOp(result: constName, op: "const", shape: [],
                         params: [("val", .floatValue(-1.0))]),
                CoreMLOp(result: resultName, op: "mul", shape: outputShape,
                         params: [("x", .variable(inputName)), ("y", .variable(constName))]),
            ]

        // Reshape
        case .reshape:
            let inputName = coreMLName(op.operands[0])
            operandShapes[op.result] = outputShape
            return [CoreMLOp(result: resultName, op: "reshape", shape: outputShape,
                             params: [("x", .variable(inputName)), ("shape", .intArray(outputShape))])]

        // Transpose
        case .transpose:
            let inputName = coreMLName(op.operands[0])
            guard let perm = op.attributes.dimensions else {
                throw MILEmitterError.invalidAttributes("transpose missing dimensions")
            }
            operandShapes[op.result] = outputShape
            return [CoreMLOp(result: resultName, op: "transpose", shape: outputShape,
                             params: [("x", .variable(inputName)), ("perm", .intArray(perm))])]

        // Broadcast
        case .broadcastInDim:
            return try translateBroadcast(op: op, operandShapes: &operandShapes)

        // Dot / DotGeneral
        case .dot, .dotGeneral:
            return try translateDot(op: op, operandShapes: &operandShapes)

        // Convolution
        case .convolution:
            return try translateConvolution(op: op, operandShapes: &operandShapes)

        // Reduce
        case .reduce:
            return try translateReduce(op: op, operandShapes: &operandShapes)

        // Concatenate
        case .concatenate:
            let inputNames = op.operands.map { coreMLName($0) }
            guard let axis = op.attributes.axis else {
                throw MILEmitterError.invalidAttributes("concatenate missing axis")
            }
            operandShapes[op.result] = outputShape
            return [CoreMLOp(result: resultName, op: "concat", shape: outputShape,
                             params: [("values", .variableTuple(inputNames)), ("axis", .intValue(axis))])]

        // Slice
        case .slice:
            return try translateSlice(op: op, operandShapes: &operandShapes)

        // Compare
        case .compare:
            return try translateCompare(op: op, operandShapes: &operandShapes)

        // Select
        case .select:
            let condName = coreMLName(op.operands[0])
            let trueName = coreMLName(op.operands[1])
            let falseName = coreMLName(op.operands[2])
            operandShapes[op.result] = outputShape
            return [CoreMLOp(result: resultName, op: "select", shape: outputShape,
                             params: [("cond", .variable(condName)), ("a", .variable(trueName)), ("b", .variable(falseName))])]

        default:
            throw MILEmitterError.unsupportedOperation(op.kind, MILOpTranslator.unsupportedReason(op.kind))
        }
    }

    // MARK: - Helpers

    private func makeUnary(_ op: HLOOperation, milOp: String, operandShapes: inout [String: [Int]]) -> CoreMLOp {
        let inputName = coreMLName(op.operands[0])
        let resultName = coreMLName(op.result)
        operandShapes[op.result] = op.resultType.shape
        return CoreMLOp(result: resultName, op: milOp, shape: op.resultType.shape,
                        params: [("x", .variable(inputName))])
    }

    private func makeBinary(_ op: HLOOperation, milOp: String, operandShapes: inout [String: [Int]]) -> CoreMLOp {
        let lhsName = coreMLName(op.operands[0])
        let rhsName = coreMLName(op.operands[1])
        let resultName = coreMLName(op.result)
        operandShapes[op.result] = op.resultType.shape
        return CoreMLOp(result: resultName, op: milOp, shape: op.resultType.shape,
                        params: [("x", .variable(lhsName)), ("y", .variable(rhsName))])
    }

    // MARK: - Constants

    private func translateConstant(op: HLOOperation, operandShapes: inout [String: [Int]]) throws -> [CoreMLOp] {
        let resultName = coreMLName(op.result)
        let shape = op.resultType.shape

        guard let value = op.attributes.constantValue else {
            throw MILEmitterError.invalidAttributes("constant op missing constantValue")
        }

        operandShapes[op.result] = shape

        switch value {
        case .scalar(let v):
            return [CoreMLOp(result: resultName, op: "const", shape: shape,
                             params: [("val", .floatValue(v))])]
        case .splat(let v, _):
            let count = shape.reduce(1, *)
            let arr = [Double](repeating: v, count: count)
            return [CoreMLOp(result: resultName, op: "const", shape: shape,
                             params: [("val", .floatArray(arr))])]
        case .dense(let values, _):
            return [CoreMLOp(result: resultName, op: "const", shape: shape,
                             params: [("val", .floatArray(values))])]
        }
    }

    // MARK: - Broadcast

    private func translateBroadcast(op: HLOOperation, operandShapes: inout [String: [Int]]) throws -> [CoreMLOp] {
        let inputName = coreMLName(op.operands[0])
        let resultName = coreMLName(op.result)
        let outputShape = op.resultType.shape
        let inputShape = operandShapes[op.operands[0]] ?? []
        let broadcastDims = op.attributes.dimensions ?? []

        var intermediateShape = Array(repeating: 1, count: outputShape.count)
        for (inputDimIdx, outputDimIdx) in broadcastDims.enumerated() {
            if inputDimIdx < inputShape.count {
                intermediateShape[outputDimIdx] = inputShape[inputDimIdx]
            }
        }

        var ops: [CoreMLOp] = []
        operandShapes[op.result] = outputShape

        if intermediateShape != inputShape {
            let reshapeName = resultName + "_bc_reshape"
            ops.append(CoreMLOp(result: reshapeName, op: "reshape", shape: intermediateShape,
                                params: [("x", .variable(inputName)), ("shape", .intArray(intermediateShape))]))

            var reps = Array(repeating: 1, count: outputShape.count)
            var needsTile = false
            for i in 0..<outputShape.count {
                if intermediateShape[i] == 1 && outputShape[i] > 1 {
                    reps[i] = outputShape[i]
                    needsTile = true
                }
            }

            if needsTile {
                ops.append(CoreMLOp(result: resultName, op: "tile", shape: outputShape,
                                    params: [("x", .variable(reshapeName)), ("reps", .intArray(reps))]))
            } else {
                ops.append(CoreMLOp(result: resultName, op: "reshape", shape: outputShape,
                                    params: [("x", .variable(reshapeName)), ("shape", .intArray(outputShape))]))
            }
        } else {
            var reps = Array(repeating: 1, count: outputShape.count)
            var needsTile = false
            for i in 0..<outputShape.count {
                if intermediateShape[i] == 1 && outputShape[i] > 1 {
                    reps[i] = outputShape[i]
                    needsTile = true
                }
            }

            if needsTile {
                ops.append(CoreMLOp(result: resultName, op: "tile", shape: outputShape,
                                    params: [("x", .variable(inputName)), ("reps", .intArray(reps))]))
            } else {
                ops.append(CoreMLOp(result: resultName, op: "identity", shape: outputShape,
                                    params: [("x", .variable(inputName))]))
            }
        }

        return ops
    }

    // MARK: - Dot / DotGeneral

    private func translateDot(op: HLOOperation, operandShapes: inout [String: [Int]]) throws -> [CoreMLOp] {
        let lhsName = coreMLName(op.operands[0])
        let rhsName = coreMLName(op.operands[1])
        let resultName = coreMLName(op.result)
        let outputShape = op.resultType.shape

        var transposeX = false
        var transposeY = false

        if op.kind == .dotGeneral, let dimNums = op.attributes.dotDimensionNumbers {
            let lhsShape = operandShapes[op.operands[0]] ?? []
            let rhsShape = operandShapes[op.operands[1]] ?? []

            if dimNums.lhsBatchingDimensions.isEmpty {
                if let lhsContract = dimNums.lhsContractingDimensions.first {
                    if lhsShape.count >= 2 && lhsContract == lhsShape.count - 2 {
                        transposeX = true
                    }
                }
                if let rhsContract = dimNums.rhsContractingDimensions.first {
                    if rhsShape.count >= 2 && rhsContract == rhsShape.count - 1 {
                        transposeY = true
                    }
                }
            }
        }

        operandShapes[op.result] = outputShape
        return [CoreMLOp(result: resultName, op: "matmul", shape: outputShape,
                         params: [
                            ("x", .variable(lhsName)),
                            ("y", .variable(rhsName)),
                            ("transpose_x", .boolValue(transposeX)),
                            ("transpose_y", .boolValue(transposeY)),
                         ])]
    }

    // MARK: - Convolution

    private func translateConvolution(op: HLOOperation, operandShapes: inout [String: [Int]]) throws -> [CoreMLOp] {
        let resultName = coreMLName(op.result)
        let outputShape = op.resultType.shape
        let attrs = op.attributes

        guard let dimNums = attrs.convolutionDimensionNumbers else {
            throw MILEmitterError.invalidAttributes("convolution missing dimension_numbers")
        }

        let strides = attrs.windowStrides ?? Array(repeating: 1, count: dimNums.inputSpatialDimensions.count)
        let dilations = attrs.rhsDilation ?? Array(repeating: 1, count: dimNums.inputSpatialDimensions.count)
        let groups = attrs.featureGroupCount ?? 1

        let padding: [Int]
        if let convPad = attrs.convPadding {
            padding = convPad.flatMap { $0 }
        } else {
            padding = Array(repeating: 0, count: dimNums.inputSpatialDimensions.count * 2)
        }

        var ops: [CoreMLOp] = []
        var milInputName = coreMLName(op.operands[0])
        let inputShape = operandShapes[op.operands[0]] ?? []

        let isInputNCHW = dimNums.inputBatchDimension == 0 && dimNums.inputFeatureDimension == 1
        if !isInputNCHW && !inputShape.isEmpty {
            var desiredOrder = [dimNums.inputBatchDimension, dimNums.inputFeatureDimension]
            desiredOrder.append(contentsOf: dimNums.inputSpatialDimensions)
            let transposedName = resultName + "_input_nchw"
            let transposedShape = desiredOrder.map { inputShape[$0] }
            ops.append(CoreMLOp(result: transposedName, op: "transpose", shape: transposedShape,
                                params: [("x", .variable(milInputName)), ("perm", .intArray(desiredOrder))]))
            milInputName = transposedName
        }

        var milKernelName = coreMLName(op.operands[1])
        let kernelShape = operandShapes[op.operands[1]] ?? []
        let isKernelOIHW = dimNums.kernelOutputFeatureDimension == 0 && dimNums.kernelInputFeatureDimension == 1
        if !isKernelOIHW && !kernelShape.isEmpty {
            var desiredOrder = [dimNums.kernelOutputFeatureDimension, dimNums.kernelInputFeatureDimension]
            desiredOrder.append(contentsOf: dimNums.kernelSpatialDimensions)
            let transposedName = resultName + "_kernel_oihw"
            let transposedShape = desiredOrder.map { kernelShape[$0] }
            ops.append(CoreMLOp(result: transposedName, op: "transpose", shape: transposedShape,
                                params: [("x", .variable(milKernelName)), ("perm", .intArray(desiredOrder))]))
            milKernelName = transposedName
        }

        let isOutputNCHW = dimNums.outputBatchDimension == 0 && dimNums.outputFeatureDimension == 1
        let convOutputName = isOutputNCHW ? resultName : resultName + "_conv_nchw"

        let nchwOutputShape: [Int]
        if isOutputNCHW {
            nchwOutputShape = outputShape
        } else {
            var shape = Array(repeating: 0, count: outputShape.count)
            shape[0] = outputShape[dimNums.outputBatchDimension]
            shape[1] = outputShape[dimNums.outputFeatureDimension]
            for (i, spatialDim) in dimNums.outputSpatialDimensions.enumerated() {
                shape[2 + i] = outputShape[spatialDim]
            }
            nchwOutputShape = shape
        }

        ops.append(CoreMLOp(result: convOutputName, op: "conv", shape: nchwOutputShape,
                            params: [
                                ("x", .variable(milInputName)),
                                ("weight", .variable(milKernelName)),
                                ("strides", .intArray(strides)),
                                ("pad_type", .stringValue("custom")),
                                ("pad", .intArray(padding)),
                                ("dilations", .intArray(dilations)),
                                ("groups", .intValue(groups)),
                            ]))

        if !isOutputNCHW {
            var inversePerm = Array(repeating: 0, count: outputShape.count)
            inversePerm[dimNums.outputBatchDimension] = 0
            inversePerm[dimNums.outputFeatureDimension] = 1
            for (i, spatialDim) in dimNums.outputSpatialDimensions.enumerated() {
                inversePerm[spatialDim] = 2 + i
            }
            ops.append(CoreMLOp(result: resultName, op: "transpose", shape: outputShape,
                                params: [("x", .variable(convOutputName)), ("perm", .intArray(inversePerm))]))
        }

        operandShapes[op.result] = outputShape
        return ops
    }

    // MARK: - Reduce

    private func translateReduce(op: HLOOperation, operandShapes: inout [String: [Int]]) throws -> [CoreMLOp] {
        let inputName = coreMLName(op.operands[0])
        let resultName = coreMLName(op.result)
        let outputShape = op.resultType.shape

        guard let reductionKind = op.attributes.reductionKind else {
            throw MILEmitterError.invalidAttributes("reduce missing reductionKind")
        }
        guard let axes = op.attributes.dimensions else {
            throw MILEmitterError.invalidAttributes("reduce missing dimensions")
        }

        let milOp: String
        switch reductionKind {
        case .sum: milOp = "reduce_sum"
        case .max: milOp = "reduce_max"
        case .min: milOp = "reduce_min"
        case .mean: milOp = "reduce_mean"
        case .product: milOp = "reduce_prod"
        case .and, .or:
            throw MILEmitterError.unsupportedOperation(.reduce, "boolean reduction not supported on ANE")
        }

        let inputShape = operandShapes[op.operands[0]] ?? []
        let keepDims = outputShape.count == inputShape.count

        operandShapes[op.result] = outputShape
        return [CoreMLOp(result: resultName, op: milOp, shape: outputShape,
                         params: [
                            ("x", .variable(inputName)),
                            ("axes", .intArray(axes)),
                            ("keep_dims", .boolValue(keepDims)),
                         ])]
    }

    // MARK: - Slice

    private func translateSlice(op: HLOOperation, operandShapes: inout [String: [Int]]) throws -> [CoreMLOp] {
        let inputName = coreMLName(op.operands[0])
        let resultName = coreMLName(op.result)
        let outputShape = op.resultType.shape

        guard let starts = op.attributes.sliceStarts,
              let limits = op.attributes.sliceLimits else {
            throw MILEmitterError.invalidAttributes("slice missing start/limit indices")
        }

        let strides = op.attributes.sliceStrides ?? Array(repeating: 1, count: starts.count)

        operandShapes[op.result] = outputShape
        return [CoreMLOp(result: resultName, op: "slice_by_index", shape: outputShape,
                         params: [
                            ("x", .variable(inputName)),
                            ("begin", .intArray(starts)),
                            ("end", .intArray(limits)),
                            ("strides", .intArray(strides)),
                         ])]
    }

    // MARK: - Compare

    private func translateCompare(op: HLOOperation, operandShapes: inout [String: [Int]]) throws -> [CoreMLOp] {
        let lhsName = coreMLName(op.operands[0])
        let rhsName = coreMLName(op.operands[1])
        let resultName = coreMLName(op.result)
        let outputShape = op.resultType.shape

        guard let direction = op.attributes.comparisonDirection else {
            throw MILEmitterError.invalidAttributes("compare missing comparisonDirection")
        }

        let milOp: String
        switch direction {
        case .eq: milOp = "equal"
        case .ne: milOp = "not_equal"
        case .lt: milOp = "less"
        case .le: milOp = "less_equal"
        case .gt: milOp = "greater"
        case .ge: milOp = "greater_equal"
        }

        operandShapes[op.result] = outputShape
        return [CoreMLOp(result: resultName, op: milOp, shape: outputShape,
                         params: [("x", .variable(lhsName)), ("y", .variable(rhsName))])]
    }
}
