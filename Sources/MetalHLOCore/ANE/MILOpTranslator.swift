// MILOpTranslator.swift
// MetalHLOCore
//
// Translates individual HLO operations to MIL operations.
// This is the core mapping from StableHLO semantics to Apple's
// Machine Learning Intermediate Language.

import Foundation

/// Translates individual HLO operations to MIL operations.
///
/// Each HLO operation maps to one or more MIL statements emitted via the
/// `MILTextBuilder`. The translator handles shape-dependent logic like
/// broadcast decomposition and convolution layout transposition.
internal struct MILOpTranslator {

    /// Returns whether an HLOOpKind is supported for MIL emission.
    static func isSupported(_ kind: HLOOpKind) -> Bool {
        switch kind {
        // P0: Must-have
        case .constant, .add, .subtract, .multiply, .divide,
             .maximum, .minimum, .reshape, .transpose,
             .broadcastInDim, .dot, .dotGeneral, .convolution:
            return true
        // P1: Should-have
        case .negate, .abs, .exponential, .tanh, .logistic,
             .reduce, .concatenate, .slice, .compare, .select:
            return true
        // P2: Nice-to-have
        case .power, .log, .sqrt, .rsqrt:
            return true
        default:
            return false
        }
    }

    /// Returns a human-readable reason why an op is unsupported.
    static func unsupportedReason(_ kind: HLOOpKind) -> String {
        switch kind {
        case .fft: return "FFT not available on ANE"
        case .sort: return "Sort not available on ANE"
        case .scatter: return "Scatter not available on ANE"
        case .complex, .real, .imag: return "Complex numbers not available on ANE"
        case .whileOp, .ifOp: return "Control flow not yet supported in MIL emitter"
        case .gather: return "Gather not yet supported in MIL emitter"
        default:
            if kind.isDynamic { return "Dynamic operations not supported on ANE (requires static shapes)" }
            return "\(kind) not yet implemented in MIL emitter"
        }
    }

    /// Translates an HLO operation into MIL text via the builder.
    ///
    /// May emit multiple MIL statements for one HLO op (e.g., broadcast
    /// decomposes into reshape + tile).
    ///
    /// - Parameters:
    ///   - op: The HLO operation.
    ///   - builder: The MIL text builder.
    ///   - weightPacker: The weight packer (for constants).
    ///   - operandShapes: Map of SSA name → shape (for broadcasts etc.).
    /// - Throws: `MILEmitterError` if the op is unsupported.
    static func translate(
        op: HLOOperation,
        builder: MILTextBuilder,
        weightPacker: MILWeightPacker,
        operandShapes: inout [String: [Int]]
    ) throws {
        let resultName = MILTypeMapper.sanitizeName(op.result)
        let outputShape = op.resultType.shape

        switch op.kind {
        // MARK: - P0: Constants
        case .constant:
            guard let value = op.attributes.constantValue else {
                throw MILEmitterError.invalidAttributes("constant op missing constantValue")
            }
            weightPacker.packConstant(
                name: resultName,
                value: value,
                resultType: op.resultType,
                builder: builder
            )
            operandShapes[op.result] = outputShape

        // MARK: - P0: Binary Arithmetic
        case .add:
            try emitBinary(op: op, milOp: "add", builder: builder, operandShapes: &operandShapes)
        case .subtract:
            try emitBinary(op: op, milOp: "sub", builder: builder, operandShapes: &operandShapes)
        case .multiply:
            try emitBinary(op: op, milOp: "mul", builder: builder, operandShapes: &operandShapes)
        case .divide:
            try emitBinary(op: op, milOp: "real_div", builder: builder, operandShapes: &operandShapes)
        case .maximum:
            try emitBinary(op: op, milOp: "maximum", builder: builder, operandShapes: &operandShapes)
        case .minimum:
            try emitBinary(op: op, milOp: "minimum", builder: builder, operandShapes: &operandShapes)

        // MARK: - P0: Reshape
        case .reshape:
            let inputName = MILTypeMapper.sanitizeName(op.operands[0])
            builder.emitOp(
                name: resultName,
                shape: outputShape,
                op: "reshape",
                params: [
                    ("x", .variable(inputName)),
                    ("shape", .intArray(outputShape)),
                ]
            )
            operandShapes[op.result] = outputShape

        // MARK: - P0: Transpose
        case .transpose:
            let inputName = MILTypeMapper.sanitizeName(op.operands[0])
            guard let perm = op.attributes.dimensions else {
                throw MILEmitterError.invalidAttributes("transpose missing dimensions")
            }
            builder.emitOp(
                name: resultName,
                shape: outputShape,
                op: "transpose",
                params: [
                    ("x", .variable(inputName)),
                    ("perm", .intArray(perm)),
                ]
            )
            operandShapes[op.result] = outputShape

        // MARK: - P0: Broadcast
        case .broadcastInDim:
            try translateBroadcast(op: op, builder: builder, operandShapes: &operandShapes)

        // MARK: - P0: Dot / DotGeneral
        case .dot, .dotGeneral:
            try translateDot(op: op, builder: builder, operandShapes: &operandShapes)

        // MARK: - P0: Convolution
        case .convolution:
            try translateConvolution(op: op, builder: builder, operandShapes: &operandShapes)

        // MARK: - P1: Unary ops
        case .negate:
            try translateNegate(op: op, builder: builder, operandShapes: &operandShapes)
        case .abs:
            try emitUnary(op: op, milOp: "abs", builder: builder, operandShapes: &operandShapes)
        case .exponential:
            try emitUnary(op: op, milOp: "exp", builder: builder, operandShapes: &operandShapes)
        case .tanh:
            try emitUnary(op: op, milOp: "tanh", builder: builder, operandShapes: &operandShapes)
        case .logistic:
            try emitUnary(op: op, milOp: "sigmoid", builder: builder, operandShapes: &operandShapes)

        // MARK: - P1: Reduce
        case .reduce:
            try translateReduce(op: op, builder: builder, operandShapes: &operandShapes)

        // MARK: - P1: Concatenate
        case .concatenate:
            try translateConcatenate(op: op, builder: builder, operandShapes: &operandShapes)

        // MARK: - P1: Slice
        case .slice:
            try translateSlice(op: op, builder: builder, operandShapes: &operandShapes)

        // MARK: - P1: Compare
        case .compare:
            try translateCompare(op: op, builder: builder, operandShapes: &operandShapes)

        // MARK: - P1: Select
        case .select:
            try translateSelect(op: op, builder: builder, operandShapes: &operandShapes)

        // MARK: - P2: Additional unary ops
        case .power:
            try emitBinary(op: op, milOp: "pow", builder: builder, operandShapes: &operandShapes)
        case .log:
            try emitUnary(op: op, milOp: "log", builder: builder, operandShapes: &operandShapes)
        case .sqrt:
            try emitUnary(op: op, milOp: "sqrt", builder: builder, operandShapes: &operandShapes)
        case .rsqrt:
            try emitUnary(op: op, milOp: "rsqrt", builder: builder, operandShapes: &operandShapes)

        default:
            throw MILEmitterError.unsupportedOperation(op.kind, unsupportedReason(op.kind))
        }
    }

    // MARK: - Helpers

    /// Emits a unary MIL operation: `%result = op(x=%input)`
    private static func emitUnary(
        op: HLOOperation,
        milOp: String,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let inputName = MILTypeMapper.sanitizeName(op.operands[0])
        let resultName = MILTypeMapper.sanitizeName(op.result)
        builder.emitOp(
            name: resultName,
            shape: op.resultType.shape,
            op: milOp,
            params: [("x", .variable(inputName))]
        )
        operandShapes[op.result] = op.resultType.shape
    }

    /// Emits a binary MIL operation: `%result = op(x=%lhs, y=%rhs)`
    private static func emitBinary(
        op: HLOOperation,
        milOp: String,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let lhsName = MILTypeMapper.sanitizeName(op.operands[0])
        let rhsName = MILTypeMapper.sanitizeName(op.operands[1])
        let resultName = MILTypeMapper.sanitizeName(op.result)
        builder.emitOp(
            name: resultName,
            shape: op.resultType.shape,
            op: milOp,
            params: [
                ("x", .variable(lhsName)),
                ("y", .variable(rhsName)),
            ]
        )
        operandShapes[op.result] = op.resultType.shape
    }

    // MARK: - Negate

    /// MIL has no `neg` op. Emit `mul(x, -1)`.
    private static func translateNegate(
        op: HLOOperation,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let inputName = MILTypeMapper.sanitizeName(op.operands[0])
        let resultName = MILTypeMapper.sanitizeName(op.result)

        // Emit const -1
        let negOneName = resultName + "_neg_one"
        builder.emitInlineConst(name: negOneName, shape: [], value: "-1.0")

        builder.emitOp(
            name: resultName,
            shape: op.resultType.shape,
            op: "mul",
            params: [
                ("x", .variable(inputName)),
                ("y", .variable(negOneName)),
            ]
        )
        operandShapes[op.result] = op.resultType.shape
    }

    // MARK: - Broadcast

    /// Translates `broadcast_in_dim` to `reshape` + `tile`.
    ///
    /// HLO `broadcast_in_dim` maps input dimensions to output dimensions
    /// via `attributes.dimensions`. We decompose this into:
    /// 1. Reshape input to match output rank (inserting size-1 dims)
    /// 2. Tile to replicate along broadcast dims
    private static func translateBroadcast(
        op: HLOOperation,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let inputName = MILTypeMapper.sanitizeName(op.operands[0])
        let resultName = MILTypeMapper.sanitizeName(op.result)
        let outputShape = op.resultType.shape
        let inputShape = operandShapes[op.operands[0]] ?? []
        let broadcastDims = op.attributes.dimensions ?? []

        // Build intermediate shape: same rank as output, with 1s where broadcast happens
        var intermediateShape = Array(repeating: 1, count: outputShape.count)
        for (inputDimIdx, outputDimIdx) in broadcastDims.enumerated() {
            if inputDimIdx < inputShape.count {
                intermediateShape[outputDimIdx] = inputShape[inputDimIdx]
            }
        }

        // Step 1: Reshape to insert size-1 dimensions
        if intermediateShape != inputShape {
            let reshapeName = resultName + "_bc_reshape"
            builder.emitOp(
                name: reshapeName,
                shape: intermediateShape,
                op: "reshape",
                params: [
                    ("x", .variable(inputName)),
                    ("shape", .intArray(intermediateShape)),
                ]
            )

            // Step 2: Check if tiling is needed
            var reps = Array(repeating: 1, count: outputShape.count)
            var needsTile = false
            for i in 0..<outputShape.count {
                if intermediateShape[i] == 1 && outputShape[i] > 1 {
                    reps[i] = outputShape[i]
                    needsTile = true
                }
            }

            if needsTile {
                builder.emitOp(
                    name: resultName,
                    shape: outputShape,
                    op: "tile",
                    params: [
                        ("x", .variable(reshapeName)),
                        ("reps", .intArray(reps)),
                    ]
                )
            } else {
                // Just alias reshape result
                builder.emitOp(
                    name: resultName,
                    shape: outputShape,
                    op: "reshape",
                    params: [
                        ("x", .variable(reshapeName)),
                        ("shape", .intArray(outputShape)),
                    ]
                )
            }
        } else {
            // Input shape already matches intermediate — just tile if needed
            var reps = Array(repeating: 1, count: outputShape.count)
            var needsTile = false
            for i in 0..<outputShape.count {
                if intermediateShape[i] == 1 && outputShape[i] > 1 {
                    reps[i] = outputShape[i]
                    needsTile = true
                }
            }

            if needsTile {
                builder.emitOp(
                    name: resultName,
                    shape: outputShape,
                    op: "tile",
                    params: [
                        ("x", .variable(inputName)),
                        ("reps", .intArray(reps)),
                    ]
                )
            } else {
                // Identity — emit identity reshape
                builder.emitOp(
                    name: resultName,
                    shape: outputShape,
                    op: "identity",
                    params: [("x", .variable(inputName))]
                )
            }
        }

        operandShapes[op.result] = outputShape
    }

    // MARK: - Dot / DotGeneral

    /// Translates `dot` and `dot_general` to MIL `matmul`.
    private static func translateDot(
        op: HLOOperation,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let lhsName = MILTypeMapper.sanitizeName(op.operands[0])
        let rhsName = MILTypeMapper.sanitizeName(op.operands[1])
        let resultName = MILTypeMapper.sanitizeName(op.result)
        let outputShape = op.resultType.shape

        var transposeX = false
        var transposeY = false

        if op.kind == .dotGeneral, let dimNums = op.attributes.dotDimensionNumbers {
            // Standard matmul: lhs_contracting=[K_dim], rhs_contracting=[K_dim]
            // MIL matmul expects: (M,K) @ (K,N) -> (M,N)
            // transpose_x=True if contracting is first dim of lhs
            // transpose_y=True if contracting is last dim of rhs
            let lhsShape = operandShapes[op.operands[0]] ?? []
            let rhsShape = operandShapes[op.operands[1]] ?? []

            if dimNums.lhsBatchingDimensions.isEmpty {
                // Non-batched matmul
                if let lhsContract = dimNums.lhsContractingDimensions.first {
                    let lhsRank = lhsShape.count
                    if lhsRank >= 2 && lhsContract == lhsRank - 2 {
                        transposeX = true
                    }
                }
                if let rhsContract = dimNums.rhsContractingDimensions.first {
                    let rhsRank = rhsShape.count
                    if rhsRank >= 2 && rhsContract == rhsRank - 1 {
                        transposeY = true
                    }
                }
            }
            // For batched matmul, MIL matmul handles leading batch dims automatically
        }

        builder.emitOp(
            name: resultName,
            shape: outputShape,
            op: "matmul",
            params: [
                ("x", .variable(lhsName)),
                ("y", .variable(rhsName)),
                ("transpose_x", .boolLiteral(transposeX)),
                ("transpose_y", .boolLiteral(transposeY)),
            ]
        )
        operandShapes[op.result] = outputShape
    }

    // MARK: - Convolution

    /// Translates HLO `convolution` to MIL `conv`.
    ///
    /// HLO convolutions use flexible layout via `ConvolutionDimensionNumbers`.
    /// MIL `conv` expects NCHW input and OIHW kernel. We insert transposes
    /// as needed.
    private static func translateConvolution(
        op: HLOOperation,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let resultName = MILTypeMapper.sanitizeName(op.result)
        let outputShape = op.resultType.shape
        let attrs = op.attributes

        guard let dimNums = attrs.convolutionDimensionNumbers else {
            throw MILEmitterError.invalidAttributes("convolution missing dimension_numbers")
        }

        let strides = attrs.windowStrides ?? Array(repeating: 1, count: dimNums.inputSpatialDimensions.count)
        let dilations = attrs.rhsDilation ?? Array(repeating: 1, count: dimNums.inputSpatialDimensions.count)
        let groups = attrs.featureGroupCount ?? 1

        // Build padding array: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        let padding: [Int]
        if let convPad = attrs.convPadding {
            padding = convPad.flatMap { $0 }
        } else {
            padding = Array(repeating: 0, count: dimNums.inputSpatialDimensions.count * 2)
        }

        // Check if input is already NCHW
        let isInputNCHW = dimNums.inputBatchDimension == 0
            && dimNums.inputFeatureDimension == 1

        var milInputName = MILTypeMapper.sanitizeName(op.operands[0])
        let inputShape = operandShapes[op.operands[0]] ?? []

        if !isInputNCHW && !inputShape.isEmpty {
            // Need to transpose input to NCHW
            // Build permutation: [batchDim, featureDim, spatial0, spatial1, ...]
            var desiredOrder = [dimNums.inputBatchDimension, dimNums.inputFeatureDimension]
            desiredOrder.append(contentsOf: dimNums.inputSpatialDimensions)
            // desiredOrder[i] = which source dim goes to position i

            let transposedName = resultName + "_input_nchw"
            let transposedShape = desiredOrder.map { inputShape[$0] }

            builder.emitOp(
                name: transposedName,
                shape: transposedShape,
                op: "transpose",
                params: [
                    ("x", .variable(milInputName)),
                    ("perm", .intArray(desiredOrder)),
                ]
            )
            milInputName = transposedName
        }

        // Check if kernel is already OIHW
        let isKernelOIHW = dimNums.kernelOutputFeatureDimension == 0
            && dimNums.kernelInputFeatureDimension == 1

        var milKernelName = MILTypeMapper.sanitizeName(op.operands[1])
        let kernelShape = operandShapes[op.operands[1]] ?? []

        if !isKernelOIHW && !kernelShape.isEmpty {
            var desiredOrder = [dimNums.kernelOutputFeatureDimension, dimNums.kernelInputFeatureDimension]
            desiredOrder.append(contentsOf: dimNums.kernelSpatialDimensions)

            let transposedName = resultName + "_kernel_oihw"
            let transposedShape = desiredOrder.map { kernelShape[$0] }

            builder.emitOp(
                name: transposedName,
                shape: transposedShape,
                op: "transpose",
                params: [
                    ("x", .variable(milKernelName)),
                    ("perm", .intArray(desiredOrder)),
                ]
            )
            milKernelName = transposedName
        }

        // Emit conv — output is in NCHW format
        let isOutputNCHW = dimNums.outputBatchDimension == 0
            && dimNums.outputFeatureDimension == 1

        let convOutputName: String
        if !isOutputNCHW {
            convOutputName = resultName + "_conv_nchw"
        } else {
            convOutputName = resultName
        }

        // Compute NCHW output shape
        let nchwOutputShape: [Int]
        if isOutputNCHW {
            nchwOutputShape = outputShape
        } else {
            // Build NCHW shape from output dim numbers
            var shape = Array(repeating: 0, count: outputShape.count)
            shape[0] = outputShape[dimNums.outputBatchDimension]
            shape[1] = outputShape[dimNums.outputFeatureDimension]
            for (i, spatialDim) in dimNums.outputSpatialDimensions.enumerated() {
                shape[2 + i] = outputShape[spatialDim]
            }
            nchwOutputShape = shape
        }

        builder.emitOp(
            name: convOutputName,
            shape: nchwOutputShape,
            op: "conv",
            params: [
                ("x", .variable(milInputName)),
                ("weight", .variable(milKernelName)),
                ("strides", .intArray(strides)),
                ("pad_type", .stringLiteral("custom")),
                ("pad", .intArray(padding)),
                ("dilations", .intArray(dilations)),
                ("groups", .intLiteral(groups)),
            ]
        )

        // Transpose output back if needed
        if !isOutputNCHW {
            // Build inverse permutation: from NCHW back to original layout
            var inversePerm = Array(repeating: 0, count: outputShape.count)
            inversePerm[dimNums.outputBatchDimension] = 0
            inversePerm[dimNums.outputFeatureDimension] = 1
            for (i, spatialDim) in dimNums.outputSpatialDimensions.enumerated() {
                inversePerm[spatialDim] = 2 + i
            }
            // inversePerm[target] = source, but we need perm[source] = target
            // Actually MIL transpose perm: output[i] = input[perm[i]]
            // We want: output[target] = nchwOutput[source]
            // So perm[target] = source, which is inversePerm
            builder.emitOp(
                name: resultName,
                shape: outputShape,
                op: "transpose",
                params: [
                    ("x", .variable(convOutputName)),
                    ("perm", .intArray(inversePerm)),
                ]
            )
        }

        operandShapes[op.result] = outputShape
    }

    // MARK: - Reduce

    /// Translates `reduce` to MIL `reduce_sum` / `reduce_max` / etc.
    private static func translateReduce(
        op: HLOOperation,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let inputName = MILTypeMapper.sanitizeName(op.operands[0])
        let resultName = MILTypeMapper.sanitizeName(op.result)
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

        // Determine if we keep dims based on output shape
        let inputShape = operandShapes[op.operands[0]] ?? []
        let keepDims = outputShape.count == inputShape.count

        builder.emitOp(
            name: resultName,
            shape: outputShape,
            op: milOp,
            params: [
                ("x", .variable(inputName)),
                ("axes", .intArray(axes)),
                ("keep_dims", .boolLiteral(keepDims)),
            ]
        )
        operandShapes[op.result] = outputShape
    }

    // MARK: - Concatenate

    /// Translates `concatenate` to MIL `concat`.
    private static func translateConcatenate(
        op: HLOOperation,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let resultName = MILTypeMapper.sanitizeName(op.result)
        let outputShape = op.resultType.shape

        guard let axis = op.attributes.axis else {
            throw MILEmitterError.invalidAttributes("concatenate missing axis")
        }

        let inputNames = op.operands.map { MILTypeMapper.sanitizeName($0) }

        builder.emitOp(
            name: resultName,
            shape: outputShape,
            op: "concat",
            params: [
                ("values", .variableTuple(inputNames)),
                ("axis", .intLiteral(axis)),
            ]
        )
        operandShapes[op.result] = outputShape
    }

    // MARK: - Slice

    /// Translates `slice` to MIL `slice_by_index`.
    private static func translateSlice(
        op: HLOOperation,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let inputName = MILTypeMapper.sanitizeName(op.operands[0])
        let resultName = MILTypeMapper.sanitizeName(op.result)
        let outputShape = op.resultType.shape

        guard let starts = op.attributes.sliceStarts,
              let limits = op.attributes.sliceLimits else {
            throw MILEmitterError.invalidAttributes("slice missing start/limit indices")
        }

        let strides = op.attributes.sliceStrides ?? Array(repeating: 1, count: starts.count)

        builder.emitOp(
            name: resultName,
            shape: outputShape,
            op: "slice_by_index",
            params: [
                ("x", .variable(inputName)),
                ("begin", .intArray(starts)),
                ("end", .intArray(limits)),
                ("strides", .intArray(strides)),
            ]
        )
        operandShapes[op.result] = outputShape
    }

    // MARK: - Compare

    /// Translates `compare` to MIL comparison ops.
    private static func translateCompare(
        op: HLOOperation,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let lhsName = MILTypeMapper.sanitizeName(op.operands[0])
        let rhsName = MILTypeMapper.sanitizeName(op.operands[1])
        let resultName = MILTypeMapper.sanitizeName(op.result)
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

        builder.emitOp(
            name: resultName,
            shape: outputShape,
            op: milOp,
            params: [
                ("x", .variable(lhsName)),
                ("y", .variable(rhsName)),
            ]
        )
        operandShapes[op.result] = outputShape
    }

    // MARK: - Select

    /// Translates `select` to MIL `select`.
    private static func translateSelect(
        op: HLOOperation,
        builder: MILTextBuilder,
        operandShapes: inout [String: [Int]]
    ) throws {
        let condName = MILTypeMapper.sanitizeName(op.operands[0])
        let trueName = MILTypeMapper.sanitizeName(op.operands[1])
        let falseName = MILTypeMapper.sanitizeName(op.operands[2])
        let resultName = MILTypeMapper.sanitizeName(op.result)
        let outputShape = op.resultType.shape

        builder.emitOp(
            name: resultName,
            shape: outputShape,
            op: "select",
            params: [
                ("cond", .variable(condName)),
                ("a", .variable(trueName)),
                ("b", .variable(falseName)),
            ]
        )
        operandShapes[op.result] = outputShape
    }
}
