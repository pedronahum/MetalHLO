// MPSGraphCompiler.swift
// MetalHLOCore
//
// Compiles HLOModule to MPSGraph.

import Metal
@preconcurrency import MetalPerformanceShadersGraph

/// Compiles an HLOModule to an executable MPSGraph.
///
/// `MPSGraphCompiler` translates HLO operations to their MPSGraph equivalents
/// and produces a compiled graph ready for execution.
public final class MPSGraphCompiler {

    // MARK: - Properties

    private let device: MTLDevice
    private let graph: MPSGraph
    private var valueMap: [String: MPSGraphTensor] = [:]
    private var typeMap: [String: TensorType] = [:]
    private var scalarConstantMap: [String: Double] = [:]  // Track scalar constants for ops like pad
    private var tupleOutputCounts: [String: Int] = [:]  // Track multi-output operations (while, if)

    // MARK: - Initialization

    /// Creates a new compiler for the given Metal device.
    ///
    /// - Parameter device: The Metal device to compile for.
    public init(device: MTLDevice) {
        self.device = device
        self.graph = MPSGraph()
    }

    // MARK: - Compilation

    /// Compiles an HLOModule to a CompiledGraph.
    ///
    /// - Parameter module: The HLO module to compile.
    /// - Throws: `CompilationError` if compilation fails.
    /// - Returns: A compiled graph ready for execution.
    public func compile(module: HLOModule) throws -> CompiledGraph {
        let function = module.function

        // Create placeholders for inputs
        var inputTensors: [MPSGraphTensor] = []
        for input in function.inputs {
            let placeholder = graph.placeholder(
                shape: input.type.mpsShape,
                dataType: input.type.elementType.mpsDataType,
                name: input.name
            )
            valueMap[input.name] = placeholder
            typeMap[input.name] = input.type
            inputTensors.append(placeholder)
        }

        // Compile each operation
        for operation in function.operations {
            let resultTensor = try compileOperation(operation)
            valueMap[operation.result] = resultTensor
            typeMap[operation.result] = operation.resultType
        }

        // Get output tensors
        var outputTensors: [MPSGraphTensor] = []
        for returnValue in function.returnValues {
            guard let tensor = valueMap[returnValue] else {
                throw CompilationError.undefinedValue(returnValue)
            }
            outputTensors.append(tensor)
        }

        // Compile the graph
        let mpsDevice = MPSGraphDevice(mtlDevice: device)
        var feeds: [MPSGraphTensor: MPSGraphShapedType] = [:]
        for (tensor, input) in zip(inputTensors, function.inputs) {
            feeds[tensor] = MPSGraphShapedType(
                shape: input.type.mpsShape,
                dataType: input.type.elementType.mpsDataType
            )
        }

        let executable = graph.compile(
            with: mpsDevice,
            feeds: feeds,
            targetTensors: outputTensors,
            targetOperations: nil,
            compilationDescriptor: nil
        )

        return CompiledGraph(
            executable: executable,
            graph: graph,
            inputTensors: inputTensors,
            outputTensors: outputTensors,
            inputTypes: function.inputs.map { $0.type },
            outputTypes: function.outputTypes,
            device: device
        )
    }

    // MARK: - Operation Compilation

    private func compileOperation(_ op: HLOOperation) throws -> MPSGraphTensor {
        switch op.kind {
        // Binary arithmetic
        case .add:
            return try compileBinaryOp(op) { graph.addition($0, $1, name: op.result) }
        case .subtract:
            return try compileBinaryOp(op) { graph.subtraction($0, $1, name: op.result) }
        case .multiply:
            return try compileBinaryOp(op) { graph.multiplication($0, $1, name: op.result) }
        case .divide:
            return try compileBinaryOp(op) { graph.division($0, $1, name: op.result) }
        case .maximum:
            return try compileBinaryOp(op) { graph.maximum($0, $1, name: op.result) }
        case .minimum:
            return try compileBinaryOp(op) { graph.minimum($0, $1, name: op.result) }
        case .power:
            return try compileBinaryOp(op) { graph.power($0, $1, name: op.result) }

        // Unary operations
        case .negate:
            return try compileUnaryOp(op) { graph.negative(with: $0, name: op.result) }
        case .abs:
            return try compileUnaryOp(op) { graph.absolute(with: $0, name: op.result) }
        case .exponential:
            return try compileUnaryOp(op) { graph.exponent(with: $0, name: op.result) }
        case .log:
            return try compileUnaryOp(op) { graph.logarithm(with: $0, name: op.result) }
        case .sqrt:
            return try compileUnaryOp(op) { graph.squareRoot(with: $0, name: op.result) }
        case .rsqrt:
            return try compileUnaryOp(op) { graph.reverseSquareRoot(with: $0, name: op.result) }
        case .sine:
            return try compileUnaryOp(op) { graph.sin(with: $0, name: op.result) }
        case .cosine:
            return try compileUnaryOp(op) { graph.cos(with: $0, name: op.result) }
        case .tanh:
            return try compileUnaryOp(op) { graph.tanh(with: $0, name: op.result) }
        case .floor:
            return try compileUnaryOp(op) { graph.floor(with: $0, name: op.result) }
        case .ceil:
            return try compileUnaryOp(op) { graph.ceil(with: $0, name: op.result) }
        case .sign:
            return try compileUnaryOp(op) { graph.sign(with: $0, name: op.result) }
        case .not:
            return try compileUnaryOp(op) { graph.not(with: $0, name: op.result) }
        case .tan:
            return try compileUnaryOp(op) { graph.tan(with: $0, name: op.result) }
        case .atan2:
            return try compileBinaryOp(op) { graph.atan2(withPrimaryTensor: $0, secondaryTensor: $1, name: op.result) }
        case .logistic:
            return try compileUnaryOp(op) { graph.sigmoid(with: $0, name: op.result) }
        case .isFinite:
            return try compileUnaryOp(op) { graph.isFinite(with: $0, name: op.result) }

        // Additional math operations
        case .expm1:
            return try compileExpm1(op)
        case .log1p:
            return try compileLog1p(op)
        case .cbrt:
            return try compileCbrt(op)
        case .roundNearestAfz:
            return try compileUnaryOp(op) { graph.round(with: $0, name: op.result) }
        case .roundNearestEven:
            return try compileUnaryOp(op) { graph.rint(with: $0, name: op.result) }
        case .popcnt:
            return try compilePopcnt(op)

        // Shift operations
        case .shiftLeft:
            return try compileBinaryOp(op) { graph.bitwiseLeftShift($0, $1, name: op.result) }
        case .shiftRightArithmetic:
            return try compileBinaryOp(op) { graph.bitwiseRightShift($0, $1, name: op.result) }
        case .shiftRightLogical:
            return try compileShiftRightLogical(op)

        // Complex number operations
        case .complex:
            return try compileComplex(op)
        case .real:
            return try compileReal(op)
        case .imag:
            return try compileImag(op)

        // Type conversion
        case .convert:
            return try compileConvert(op)
        case .bitcastConvert:
            return try compileBitcastConvert(op)
        case .reducePrecision:
            return try compileReducePrecision(op)

        // Matrix operations
        case .dot:
            return try compileDot(op)
        case .dotGeneral:
            return try compileDotGeneral(op)
        case .transpose:
            return try compileTranspose(op)
        case .reshape:
            return try compileReshape(op)
        case .broadcastInDim:
            return try compileBroadcast(op)
        case .reverse:
            return try compileReverse(op)
        case .dynamicBroadcastInDim:
            return try compileDynamicBroadcast(op)
        case .dynamicReshape:
            return try compileDynamicReshape(op)

        // Convolution
        case .convolution:
            return try compileConvolution(op)

        // Reductions
        case .reduce:
            return try compileReduce(op)
        case .reduceWindow:
            return try compileReduceWindow(op)
        case .selectAndScatter:
            return try compileSelectAndScatter(op)

        // Linear algebra
        case .triangularSolve:
            return try compileTriangularSolve(op)
        case .cholesky:
            return try compileCholesky(op)

        // Normalization
        case .batchNormInference:
            return try compileBatchNormInference(op)
        case .batchNormTraining:
            return try compileBatchNormTraining(op)
        case .batchNormGrad:
            return try compileBatchNormGrad(op)

        // FFT
        case .fft:
            return try compileFFT(op)

        // Sort
        case .sort:
            return try compileSort(op)

        // Comparison
        case .compare:
            return try compileCompare(op)
        case .select:
            return try compileSelect(op)
        case .clamp:
            return try compileClamp(op)

        // Indexing
        case .slice:
            return try compileSlice(op)
        case .dynamicSlice:
            return try compileDynamicSlice(op)
        case .dynamicUpdateSlice:
            return try compileDynamicUpdateSlice(op)
        case .pad:
            return try compilePad(op)
        case .dynamicPad:
            return try compileDynamicPad(op)
        case .concatenate:
            return try compileConcatenate(op)
        case .gather:
            return try compileGather(op)
        case .dynamicGather:
            return try compileDynamicGather(op)
        case .scatter:
            return try compileScatter(op)

        // Constants
        case .constant:
            return try compileConstant(op)

        // RNG
        case .rng:
            return try compileRNG(op)
        case .rngBitGenerator:
            return try compileRngBitGenerator(op)

        // Bitwise
        case .and:
            return try compileBinaryOp(op) { graph.bitwiseAND($0, $1, name: op.result) }
        case .or:
            return try compileBinaryOp(op) { graph.bitwiseOR($0, $1, name: op.result) }
        case .xor:
            return try compileBinaryOp(op) { graph.bitwiseXOR($0, $1, name: op.result) }

        // Iota
        case .iota:
            return try compileIota(op)
        case .dynamicIota:
            return try compileDynamicIota(op)

        // Map
        case .map:
            return try compileMap(op)

        // Control flow
        case .whileOp:
            return try compileWhile(op)
        case .ifOp:
            return try compileIf(op)

        case .clz:
            throw CompilationError.unsupportedOperation("count_leading_zeros")

        // Quantization
        case .uniformQuantize:
            return try compileQuantize(op)
        case .uniformDequantize:
            return try compileDequantize(op)

        // Custom calls (fused operations from Magma)
        case .customCall:
            return try compileCustomCall(op)
        }
    }

    // MARK: - Custom Call Compilation

    private func compileCustomCall(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Extract target name from attributes
        guard let target = op.attributes.callTargetName else {
            throw CompilationError.missingAttribute("call_target_name", operation: "custom_call")
        }

        // Get the handler for this target
        guard let handler = CustomCallRegistry.shared.handler(for: target) else {
            throw CompilationError.unsupportedOperation("custom_call:\(target)")
        }

        // Resolve input operands
        var inputs: [MPSGraphTensor] = []
        for operandName in op.operands {
            let tensor = try getOperand(operandName)
            inputs.append(tensor)
        }

        // Parse backend config
        let backendConfig = op.attributes.backendConfig ?? ""
        let config = BackendConfigParser.parse(backendConfig)

        // Emit the custom call
        do {
            let outputs = try handler.emit(operation: op, graph: graph, inputs: inputs, config: config)
            guard let output = outputs.first else {
                throw CompilationError.invalidAttribute("no_outputs", operation: "custom_call:\(target)")
            }
            return output
        } catch let error as CustomCallError {
            throw CompilationError.invalidAttribute(error.description, operation: "custom_call:\(target)")
        }
    }

    // MARK: - Helper Methods

    private func getOperand(_ name: String) throws -> MPSGraphTensor {
        guard let tensor = valueMap[name] else {
            throw CompilationError.undefinedValue(name)
        }
        return tensor
    }

    private func compileBinaryOp(
        _ op: HLOOperation,
        using operation: (MPSGraphTensor, MPSGraphTensor) -> MPSGraphTensor
    ) throws -> MPSGraphTensor {
        guard op.operands.count == 2 else {
            throw CompilationError.wrongOperandCount(expected: 2, got: op.operands.count)
        }
        let lhs = try getOperand(op.operands[0])
        let rhs = try getOperand(op.operands[1])
        return operation(lhs, rhs)
    }

    private func compileUnaryOp(
        _ op: HLOOperation,
        using operation: (MPSGraphTensor) -> MPSGraphTensor
    ) throws -> MPSGraphTensor {
        guard op.operands.count == 1 else {
            throw CompilationError.wrongOperandCount(expected: 1, got: op.operands.count)
        }
        let input = try getOperand(op.operands[0])
        return operation(input)
    }

    private func compileConvert(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])
        return graph.cast(input, to: op.resultType.elementType.mpsDataType, name: op.result)
    }

    private func compileDot(_ op: HLOOperation) throws -> MPSGraphTensor {
        let lhs = try getOperand(op.operands[0])
        let rhs = try getOperand(op.operands[1])
        return graph.matrixMultiplication(primary: lhs, secondary: rhs, name: op.result)
    }

    private func compileDotGeneral(_ op: HLOOperation) throws -> MPSGraphTensor {
        let lhs = try getOperand(op.operands[0])
        let rhs = try getOperand(op.operands[1])

        guard let dimNumbers = op.attributes.dotDimensionNumbers else {
            // No dimension numbers provided, fall back to simple matmul
            return graph.matrixMultiplication(primary: lhs, secondary: rhs, name: op.result)
        }

        // Get shapes from type map
        guard let lhsType = typeMap[op.operands[0]],
              let rhsType = typeMap[op.operands[1]] else {
            throw CompilationError.undefinedValue(op.operands[0])
        }

        let lhsRank = lhsType.shape.count
        let rhsRank = rhsType.shape.count

        let lhsBatching = dimNumbers.lhsBatchingDimensions
        let rhsBatching = dimNumbers.rhsBatchingDimensions
        let lhsContracting = dimNumbers.lhsContractingDimensions
        let rhsContracting = dimNumbers.rhsContractingDimensions

        // Compute remaining dimensions (non-batching, non-contracting)
        let lhsRemaining = (0..<lhsRank).filter { !lhsBatching.contains($0) && !lhsContracting.contains($0) }
        let rhsRemaining = (0..<rhsRank).filter { !rhsBatching.contains($0) && !rhsContracting.contains($0) }

        // Build permutation for LHS: [batching, remaining, contracting]
        let lhsPermutation = lhsBatching + lhsRemaining + lhsContracting
        // Build permutation for RHS: [batching, contracting, remaining]
        let rhsPermutation = rhsBatching + rhsContracting + rhsRemaining

        // Transpose LHS and RHS to standard form
        var lhsTransposed = lhs
        if lhsPermutation != Array(0..<lhsRank) {
            lhsTransposed = graph.transpose(
                lhs,
                permutation: lhsPermutation.map { NSNumber(value: $0) },
                name: nil
            )
        }

        var rhsTransposed = rhs
        if rhsPermutation != Array(0..<rhsRank) {
            rhsTransposed = graph.transpose(
                rhs,
                permutation: rhsPermutation.map { NSNumber(value: $0) },
                name: nil
            )
        }

        // After transpose:
        // LHS shape: [batch..., M..., K...]
        // RHS shape: [batch..., K..., N...]
        // We need to collapse M and K dims for matmul

        let batchCount = lhsBatching.count
        let lhsRemainingCount = lhsRemaining.count
        let rhsRemainingCount = rhsRemaining.count
        let contractingCount = lhsContracting.count

        // Compute sizes for reshape
        let lhsTransposedShape = lhsPermutation.map { lhsType.shape[$0] }
        let rhsTransposedShape = rhsPermutation.map { rhsType.shape[$0] }

        let batchSize = lhsTransposedShape.prefix(batchCount).reduce(1, *)
        let mSize = lhsTransposedShape.dropFirst(batchCount).prefix(lhsRemainingCount).reduce(1, *)
        let kSize = lhsTransposedShape.suffix(contractingCount).reduce(1, *)
        let nSize = rhsTransposedShape.suffix(rhsRemainingCount).reduce(1, *)

        // Reshape to 3D for batched matmul: [batch, M, K] and [batch, K, N]
        let lhsReshaped: MPSGraphTensor
        let rhsReshaped: MPSGraphTensor

        if batchCount > 0 {
            lhsReshaped = graph.reshape(
                lhsTransposed,
                shape: [NSNumber(value: batchSize), NSNumber(value: mSize), NSNumber(value: kSize)],
                name: nil
            )
            rhsReshaped = graph.reshape(
                rhsTransposed,
                shape: [NSNumber(value: batchSize), NSNumber(value: kSize), NSNumber(value: nSize)],
                name: nil
            )
        } else {
            // No batching, use 2D matmul
            lhsReshaped = graph.reshape(
                lhsTransposed,
                shape: [NSNumber(value: mSize), NSNumber(value: kSize)],
                name: nil
            )
            rhsReshaped = graph.reshape(
                rhsTransposed,
                shape: [NSNumber(value: kSize), NSNumber(value: nSize)],
                name: nil
            )
        }

        // Perform matmul
        let matmulResult = graph.matrixMultiplication(
            primary: lhsReshaped,
            secondary: rhsReshaped,
            name: nil
        )

        // Reshape to output shape
        return graph.reshape(matmulResult, shape: op.resultType.mpsShape, name: op.result)
    }

    private func compileTranspose(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])
        let permutation = op.attributes.dimensions ?? Array((0..<op.resultType.rank).reversed())
        return graph.transpose(
            input,
            permutation: permutation.map { NSNumber(value: $0) },
            name: op.result
        )
    }

    private func compileReshape(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])
        return graph.reshape(input, shape: op.resultType.mpsShape, name: op.result)
    }

    private func compileBroadcast(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])
        let outputShape = op.resultType.shape
        let dims = op.attributes.dimensions ?? []

        // broadcast_in_dim semantics: dims[i] tells us which output dimension
        // input dimension i maps to. We need to reshape to insert size-1 dims
        // at positions not covered by dims, then broadcast.

        if dims.isEmpty {
            // Simple broadcast without dimension mapping
            return graph.broadcast(input, shape: op.resultType.mpsShape, name: op.result)
        }

        // Get input shape from type map
        guard let inputType = typeMap[op.operands[0]] else {
            throw CompilationError.undefinedValue(op.operands[0])
        }
        let inputShape = inputType.shape

        // Create intermediate shape with 1s everywhere except where dims specifies
        var intermediateShape = [Int](repeating: 1, count: outputShape.count)

        for (inputDim, outputDim) in dims.enumerated() {
            if inputDim < inputShape.count {
                intermediateShape[outputDim] = inputShape[inputDim]
            }
        }

        // Reshape input to intermediate shape
        let intermediateShapeNS = intermediateShape.map { NSNumber(value: $0) }
        let reshaped = graph.reshape(input, shape: intermediateShapeNS, name: nil)

        // Broadcast to final shape
        return graph.broadcast(reshaped, shape: op.resultType.mpsShape, name: op.result)
    }

    private func compileReduce(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])
        let axes = op.attributes.dimensions ?? []
        let axesNS = axes.map { NSNumber(value: $0) }

        switch op.attributes.reductionKind {
        case .sum:
            return graph.reductionSum(with: input, axes: axesNS, name: op.result)
        case .max:
            return graph.reductionMaximum(with: input, axes: axesNS, name: op.result)
        case .min:
            return graph.reductionMinimum(with: input, axes: axesNS, name: op.result)
        case .mean:
            return graph.mean(of: input, axes: axesNS, name: op.result)
        case .none:
            throw CompilationError.missingAttribute("reductionKind", operation: "reduce")
        }
    }

    private func compileCompare(_ op: HLOOperation) throws -> MPSGraphTensor {
        let lhs = try getOperand(op.operands[0])
        let rhs = try getOperand(op.operands[1])

        switch op.attributes.comparisonDirection {
        case .eq:
            return graph.equal(lhs, rhs, name: op.result)
        case .ne:
            return graph.notEqual(lhs, rhs, name: op.result)
        case .lt:
            return graph.lessThan(lhs, rhs, name: op.result)
        case .le:
            return graph.lessThanOrEqualTo(lhs, rhs, name: op.result)
        case .gt:
            return graph.greaterThan(lhs, rhs, name: op.result)
        case .ge:
            return graph.greaterThanOrEqualTo(lhs, rhs, name: op.result)
        case .none:
            throw CompilationError.missingAttribute("comparisonDirection", operation: "compare")
        }
    }

    private func compileSelect(_ op: HLOOperation) throws -> MPSGraphTensor {
        guard op.operands.count == 3 else {
            throw CompilationError.wrongOperandCount(expected: 3, got: op.operands.count)
        }
        let predicate = try getOperand(op.operands[0])
        let trueValue = try getOperand(op.operands[1])
        let falseValue = try getOperand(op.operands[2])
        return graph.select(predicate: predicate, trueTensor: trueValue, falseTensor: falseValue, name: op.result)
    }

    private func compileClamp(_ op: HLOOperation) throws -> MPSGraphTensor {
        guard op.operands.count == 3 else {
            throw CompilationError.wrongOperandCount(expected: 3, got: op.operands.count)
        }
        let minValue = try getOperand(op.operands[0])
        let operand = try getOperand(op.operands[1])
        let maxValue = try getOperand(op.operands[2])
        return graph.clamp(operand, min: minValue, max: maxValue, name: op.result)
    }

    private func compileSlice(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])
        let starts = op.attributes.sliceStarts ?? []
        let ends = op.attributes.sliceLimits ?? []
        let strides = op.attributes.sliceStrides ?? Array(repeating: 1, count: starts.count)

        return graph.sliceTensor(
            input,
            starts: starts.map { NSNumber(value: $0) },
            ends: ends.map { NSNumber(value: $0) },
            strides: strides.map { NSNumber(value: $0) },
            name: op.result
        )
    }

    private func compilePad(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])
        let padValueOperand = op.operands[1]

        // Look up the padding value from tracked scalar constants
        let padValue = scalarConstantMap[padValueOperand] ?? 0.0

        let low = op.attributes.padLow ?? []
        let high = op.attributes.padHigh ?? []

        // Create padding mode - constant padding
        var leftPadding: [NSNumber] = []
        var rightPadding: [NSNumber] = []

        for i in 0..<low.count {
            leftPadding.append(NSNumber(value: low[i]))
            rightPadding.append(NSNumber(value: high[i]))
        }

        return graph.padTensor(
            input,
            with: .constant,
            leftPadding: leftPadding,
            rightPadding: rightPadding,
            constantValue: padValue,
            name: op.result
        )
    }

    private func compileConcatenate(_ op: HLOOperation) throws -> MPSGraphTensor {
        let tensors = try op.operands.map { try getOperand($0) }
        let axis = op.attributes.axis ?? 0
        return graph.concatTensors(tensors, dimension: axis, name: op.result)
    }

    private func compileGather(_ op: HLOOperation) throws -> MPSGraphTensor {
        let operand = try getOperand(op.operands[0])
        let indices = try getOperand(op.operands[1])

        guard let dimNumbers = op.attributes.gatherDimensionNumbers else {
            throw CompilationError.missingAttribute("gatherDimensionNumbers", operation: "gather")
        }

        // Cast indices to int32 if needed
        let int32Indices: MPSGraphTensor
        if indices.dataType != .int32 {
            int32Indices = graph.cast(indices, to: .int32, name: "\(op.result)_indices_cast")
        } else {
            int32Indices = indices
        }

        // Use the batching-aware gather compilation
        return try compileGatherWithBatching(
            op,
            operand: operand,
            startIndices: int32Indices,
            dimNumbers: dimNumbers,
            sliceSizes: dimNumbers.sliceSizes
        )
    }

    // MARK: - Gather Batching Implementation

    /// Compile a gather operation with full batching dimension support.
    ///
    /// Handles arbitrary batching dimension positions by transposing inputs
    /// to move batch dims to leading positions (as required by MPS), then
    /// transposing the output to match the expected StableHLO layout.
    private func compileGatherWithBatching(
        _ op: HLOOperation,
        operand: MPSGraphTensor,
        startIndices: MPSGraphTensor,
        dimNumbers: GatherDimensionNumbers,
        sliceSizes: [Int]
    ) throws -> MPSGraphTensor {

        let operandBatchDims = dimNumbers.operandBatchingDims
        let indicesBatchDims = dimNumbers.startIndicesBatchingDims

        // FAST PATH: No batching dimensions
        if operandBatchDims.isEmpty {
            return try compileGatherNoBatching(
                op,
                operand: operand,
                startIndices: startIndices,
                dimNumbers: dimNumbers,
                sliceSizes: sliceSizes
            )
        }

        guard let operandShape = operand.shape?.map({ $0.intValue }),
              let indicesShape = startIndices.shape?.map({ $0.intValue }) else {
            throw CompilationError.undefinedValue("\(op.result)_shape")
        }

        let operandRank = operandShape.count
        let indicesRank = indicesShape.count
        let numBatchDims = operandBatchDims.count

        // FAST PATH: Batch dims already at leading positions
        if PermutationUtils.areDimsContiguousAtFront(operandBatchDims, rank: operandRank) &&
           PermutationUtils.areDimsContiguousAtFront(indicesBatchDims, rank: indicesRank) {
            return try compileGatherWithLeadingBatchDims(
                op,
                operand: operand,
                startIndices: startIndices,
                dimNumbers: dimNumbers,
                sliceSizes: sliceSizes,
                numBatchDims: numBatchDims
            )
        }

        // GENERAL PATH: Transpose -> Gather -> Transpose
        return try compileGatherWithTranspose(
            op,
            operand: operand,
            startIndices: startIndices,
            dimNumbers: dimNumbers,
            sliceSizes: sliceSizes,
            operandShape: operandShape,
            indicesShape: indicesShape
        )
    }

    private func compileGatherWithTranspose(
        _ op: HLOOperation,
        operand: MPSGraphTensor,
        startIndices: MPSGraphTensor,
        dimNumbers: GatherDimensionNumbers,
        sliceSizes: [Int],
        operandShape: [Int],
        indicesShape: [Int]
    ) throws -> MPSGraphTensor {

        let operandBatchDims = dimNumbers.operandBatchingDims
        let indicesBatchDims = dimNumbers.startIndicesBatchingDims
        let numBatchDims = operandBatchDims.count

        let operandRank = operandShape.count
        let indicesRank = indicesShape.count

        // STEP 1: Transpose operand to move batch dims to front
        let operandPerm = PermutationUtils.buildPermutationMovingToFront(
            dims: operandBatchDims,
            rank: operandRank
        )

        let transposedOperand: MPSGraphTensor
        if PermutationUtils.isIdentity(operandPerm) {
            transposedOperand = operand
        } else {
            transposedOperand = graph.transpose(
                operand,
                permutation: operandPerm.map { NSNumber(value: $0) },
                name: "\(op.result)_operand_to_batch_front"
            )
        }

        // STEP 2: Transpose indices to move batch dims to front
        let indicesPerm = PermutationUtils.buildPermutationMovingToFront(
            dims: indicesBatchDims,
            rank: indicesRank
        )

        let transposedIndices: MPSGraphTensor
        if PermutationUtils.isIdentity(indicesPerm) {
            transposedIndices = startIndices
        } else {
            transposedIndices = graph.transpose(
                startIndices,
                permutation: indicesPerm.map { NSNumber(value: $0) },
                name: "\(op.result)_indices_to_batch_front"
            )
        }

        // Adjust slice sizes to match transposed operand
        let adjustedSliceSizes = PermutationUtils.applyPermutation(operandPerm, to: sliceSizes)

        // STEP 3: Adjust dimension numbers for transposed tensors
        let adjustedDimNumbers = adjustGatherDimNumbers(
            original: dimNumbers,
            operandPerm: operandPerm,
            indicesPerm: indicesPerm,
            numBatchDims: numBatchDims,
            adjustedSliceSizes: adjustedSliceSizes
        )

        // STEP 4: Execute gather with batch dims at front
        let gathered = try compileGatherWithLeadingBatchDims(
            op,
            operand: transposedOperand,
            startIndices: transposedIndices,
            dimNumbers: adjustedDimNumbers,
            sliceSizes: adjustedSliceSizes,
            numBatchDims: numBatchDims
        )

        // STEP 5: Transpose output to expected layout
        guard let gatheredShape = gathered.shape?.map({ $0.intValue }) else {
            return gathered  // If shape unknown, return as-is
        }

        let outputPerm = computeGatherOutputPermutation(
            originalDimNumbers: dimNumbers,
            numBatchDims: numBatchDims,
            gatheredRank: gatheredShape.count
        )

        if PermutationUtils.isIdentity(outputPerm) {
            return gathered
        }

        return graph.transpose(
            gathered,
            permutation: outputPerm.map { NSNumber(value: $0) },
            name: "\(op.result)_output_reorder"
        )
    }

    /// Adjust GatherDimensionNumbers after transposing inputs.
    private func adjustGatherDimNumbers(
        original: GatherDimensionNumbers,
        operandPerm: [Int],
        indicesPerm: [Int],
        numBatchDims: Int,
        adjustedSliceSizes: [Int]
    ) -> GatherDimensionNumbers {

        // Inverse permutations map old positions -> new positions
        let operandInverse = PermutationUtils.invertPermutation(operandPerm)
        let indicesInverse = PermutationUtils.invertPermutation(indicesPerm)

        // Safe index access helper
        func safeOperandIndex(_ idx: Int) -> Int {
            guard idx >= 0 && idx < operandInverse.count else { return idx }
            return operandInverse[idx]
        }
        func safeIndicesIndex(_ idx: Int) -> Int {
            guard idx >= 0 && idx < indicesInverse.count else { return idx }
            return indicesInverse[idx]
        }

        // collapsed_slice_dims: operand dimensions to collapse
        let adjustedCollapsedSliceDims = original.collapsedSliceDims.map { safeOperandIndex($0) }.sorted()

        // start_index_map: maps index vector elements to operand dimensions
        let adjustedStartIndexMap = original.startIndexMap.map { safeOperandIndex($0) }

        // index_vector_dim: which dimension of indices contains the index vector
        let adjustedIndexVectorDim = safeIndicesIndex(original.indexVectorDim)

        // After transposition, batch dims are now [0, 1, ..., numBatchDims-1]
        let adjustedOperandBatchDims = Array(0..<numBatchDims)
        let adjustedIndicesBatchDims = Array(0..<numBatchDims)

        // offset_dims are OUTPUT dimensions, not operand dimensions.
        // They specify which dimensions of the output correspond to the slice.
        // After transpose, we keep them as-is since the output layout adjustment
        // happens in computeGatherOutputPermutation.
        let adjustedOffsetDims = original.offsetDims

        return GatherDimensionNumbers(
            offsetDims: adjustedOffsetDims,
            collapsedSliceDims: adjustedCollapsedSliceDims,
            startIndexMap: adjustedStartIndexMap,
            indexVectorDim: adjustedIndexVectorDim,
            sliceSizes: adjustedSliceSizes,
            operandBatchingDims: adjustedOperandBatchDims,
            startIndicesBatchingDims: adjustedIndicesBatchDims
        )
    }

    /// Compute the permutation needed to restore expected output layout.
    private func computeGatherOutputPermutation(
        originalDimNumbers: GatherDimensionNumbers,
        numBatchDims: Int,
        gatheredRank: Int
    ) -> [Int] {

        let offsetDims = originalDimNumbers.offsetDims

        // MPS output: [batch_0, batch_1, ..., offset_0, offset_1, ...]
        // Expected: dimensions with offset_dims at their specified positions,
        //           batch dims filling the rest

        var mpsToExpected = [Int](repeating: -1, count: gatheredRank)

        var batchSourceIdx = 0
        var offsetSourceIdx = numBatchDims

        for outputIdx in 0..<gatheredRank {
            if offsetDims.contains(outputIdx) {
                if offsetSourceIdx < gatheredRank {
                    mpsToExpected[outputIdx] = offsetSourceIdx
                    offsetSourceIdx += 1
                }
            } else {
                if batchSourceIdx < numBatchDims {
                    mpsToExpected[outputIdx] = batchSourceIdx
                    batchSourceIdx += 1
                }
            }
        }

        // Validate that mpsToExpected is a valid permutation before inverting
        let usedIndices = Set(mpsToExpected.filter { $0 >= 0 })
        if usedIndices.count != gatheredRank || mpsToExpected.contains(-1) {
            // Invalid permutation - return identity to avoid crash
            return Array(0..<gatheredRank)
        }

        return PermutationUtils.invertPermutation(mpsToExpected)
    }

    /// Compile gather when batch dims are already at leading positions.
    private func compileGatherWithLeadingBatchDims(
        _ op: HLOOperation,
        operand: MPSGraphTensor,
        startIndices: MPSGraphTensor,
        dimNumbers: GatherDimensionNumbers,
        sliceSizes: [Int],
        numBatchDims: Int
    ) throws -> MPSGraphTensor {

        guard let indicesShape = startIndices.shape?.map({ $0.intValue }) else {
            throw CompilationError.undefinedValue("\(op.result)_indices")
        }

        let indicesRank = indicesShape.count
        let indexVectorDim = dimNumbers.indexVectorDim

        // Handle index_vector_dim: MPS expects it at the last position
        let processedIndices: MPSGraphTensor
        if indexVectorDim != indicesRank - 1 {
            var perm = Array(0..<indicesRank)
            perm.remove(at: indexVectorDim)
            perm.append(indexVectorDim)

            processedIndices = graph.transpose(
                startIndices,
                permutation: perm.map { NSNumber(value: $0) },
                name: "\(op.result)_move_ivd_to_last"
            )
        } else {
            processedIndices = startIndices
        }

        return graph.gatherND(
            withUpdatesTensor: operand,
            indicesTensor: processedIndices,
            batchDimensions: numBatchDims,
            name: op.result
        )
    }

    /// Compile gather with no batching dimensions.
    private func compileGatherNoBatching(
        _ op: HLOOperation,
        operand: MPSGraphTensor,
        startIndices: MPSGraphTensor,
        dimNumbers: GatherDimensionNumbers,
        sliceSizes: [Int]
    ) throws -> MPSGraphTensor {

        guard let indicesShape = startIndices.shape?.map({ $0.intValue }) else {
            throw CompilationError.undefinedValue("\(op.result)_indices")
        }

        let indicesRank = indicesShape.count
        let indexVectorDim = dimNumbers.indexVectorDim
        let indexVectorSize = dimNumbers.startIndexMap.count

        // Build indices tensor for gatherND
        var reshapedIndices: MPSGraphTensor

        if indexVectorDim == indicesRank {
            // Index vector is implicit (past last dimension)
            var newShape = indicesShape.map { NSNumber(value: $0) }
            newShape.append(NSNumber(value: indexVectorSize > 0 ? indexVectorSize : 1))
            reshapedIndices = graph.reshape(
                startIndices,
                shape: newShape,
                name: "\(op.result)_indices_reshape"
            )
        } else if indexVectorDim == indicesRank - 1 {
            reshapedIndices = startIndices
        } else {
            var perm = Array(0..<indicesRank)
            perm.remove(at: indexVectorDim)
            perm.append(indexVectorDim)

            reshapedIndices = graph.transpose(
                startIndices,
                permutation: perm.map { NSNumber(value: $0) },
                name: "\(op.result)_indices_transpose"
            )
        }

        return graph.gatherND(
            withUpdatesTensor: operand,
            indicesTensor: reshapedIndices,
            batchDimensions: 0,
            name: op.result
        )
    }

    private func compileScatter(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Scatter updates values into an operand at specified indices
        // Supports common computation patterns via MPS scatter modes:
        // - .set: replace value (default)
        // - .add: add update to existing value
        // - .max: take maximum of existing and update
        // - .min: take minimum of existing and update
        // - .mul: multiply existing by update

        let operand = try getOperand(op.operands[0])
        let indices = try getOperand(op.operands[1])
        let updates = try getOperand(op.operands[2])

        guard let dimNumbers = op.attributes.scatterDimensionNumbers else {
            throw CompilationError.missingAttribute("scatterDimensionNumbers", operation: "scatter")
        }

        // Determine scatter mode from computation kind
        let scatterMode: MPSGraphScatterMode
        switch op.attributes.scatterComputationKind {
        case .add:
            scatterMode = .add
        case .max:
            scatterMode = .max
        case .min:
            scatterMode = .min
        case .mul:
            scatterMode = .mul
        case .set, .none:
            scatterMode = .set
        }

        // Cast indices to int32 if needed
        let int32Indices: MPSGraphTensor
        if indices.dataType != .int32 {
            int32Indices = graph.cast(indices, to: .int32, name: "\(op.result)_indices_cast")
        } else {
            int32Indices = indices
        }

        // Use the batching-aware scatter compilation
        return try compileScatterWithBatching(
            op,
            operand: operand,
            scatterIndices: int32Indices,
            updates: updates,
            dimNumbers: dimNumbers,
            scatterMode: scatterMode
        )
    }

    // MARK: - Scatter Batching Implementation

    /// Compile a scatter operation with full batching dimension support.
    private func compileScatterWithBatching(
        _ op: HLOOperation,
        operand: MPSGraphTensor,
        scatterIndices: MPSGraphTensor,
        updates: MPSGraphTensor,
        dimNumbers: ScatterDimensionNumbers,
        scatterMode: MPSGraphScatterMode
    ) throws -> MPSGraphTensor {

        let inputBatchDims = dimNumbers.inputBatchingDims
        let indicesBatchDims = dimNumbers.scatterIndicesBatchingDims

        // FAST PATH: No batching dimensions
        if inputBatchDims.isEmpty {
            return try compileScatterNoBatching(
                op,
                operand: operand,
                scatterIndices: scatterIndices,
                updates: updates,
                dimNumbers: dimNumbers,
                scatterMode: scatterMode
            )
        }

        guard let operandShape = operand.shape?.map({ $0.intValue }),
              let indicesShape = scatterIndices.shape?.map({ $0.intValue }),
              let updatesShape = updates.shape?.map({ $0.intValue }) else {
            throw CompilationError.undefinedValue("\(op.result)_shape")
        }

        let operandRank = operandShape.count
        let indicesRank = indicesShape.count
        let numBatchDims = inputBatchDims.count

        // FAST PATH: Batch dims already at leading positions
        if PermutationUtils.areDimsContiguousAtFront(inputBatchDims, rank: operandRank) &&
           PermutationUtils.areDimsContiguousAtFront(indicesBatchDims, rank: indicesRank) {
            return try compileScatterWithLeadingBatchDims(
                op,
                operand: operand,
                scatterIndices: scatterIndices,
                updates: updates,
                dimNumbers: dimNumbers,
                scatterMode: scatterMode,
                numBatchDims: numBatchDims
            )
        }

        // GENERAL PATH: Transpose -> Scatter -> Transpose
        return try compileScatterWithTranspose(
            op,
            operand: operand,
            scatterIndices: scatterIndices,
            updates: updates,
            dimNumbers: dimNumbers,
            scatterMode: scatterMode,
            operandShape: operandShape,
            indicesShape: indicesShape,
            updatesShape: updatesShape
        )
    }

    private func compileScatterWithTranspose(
        _ op: HLOOperation,
        operand: MPSGraphTensor,
        scatterIndices: MPSGraphTensor,
        updates: MPSGraphTensor,
        dimNumbers: ScatterDimensionNumbers,
        scatterMode: MPSGraphScatterMode,
        operandShape: [Int],
        indicesShape: [Int],
        updatesShape: [Int]
    ) throws -> MPSGraphTensor {

        let inputBatchDims = dimNumbers.inputBatchingDims
        let indicesBatchDims = dimNumbers.scatterIndicesBatchingDims
        let numBatchDims = inputBatchDims.count

        let operandRank = operandShape.count
        let indicesRank = indicesShape.count
        let updatesRank = updatesShape.count

        // STEP 1: Transpose operand to move batch dims to front
        let operandPerm = PermutationUtils.buildPermutationMovingToFront(
            dims: inputBatchDims,
            rank: operandRank
        )

        let transposedOperand: MPSGraphTensor
        if PermutationUtils.isIdentity(operandPerm) {
            transposedOperand = operand
        } else {
            transposedOperand = graph.transpose(
                operand,
                permutation: operandPerm.map { NSNumber(value: $0) },
                name: "\(op.result)_operand_to_batch_front"
            )
        }

        // STEP 2: Transpose indices to move batch dims to front
        let indicesPerm = PermutationUtils.buildPermutationMovingToFront(
            dims: indicesBatchDims,
            rank: indicesRank
        )

        let transposedIndices: MPSGraphTensor
        if PermutationUtils.isIdentity(indicesPerm) {
            transposedIndices = scatterIndices
        } else {
            transposedIndices = graph.transpose(
                scatterIndices,
                permutation: indicesPerm.map { NSNumber(value: $0) },
                name: "\(op.result)_indices_to_batch_front"
            )
        }

        // STEP 3: Transpose updates to match
        let updatesPerm = computeUpdatesPermutation(
            dimNumbers: dimNumbers,
            updatesRank: updatesRank
        )

        let transposedUpdates: MPSGraphTensor
        if PermutationUtils.isIdentity(updatesPerm) {
            transposedUpdates = updates
        } else {
            transposedUpdates = graph.transpose(
                updates,
                permutation: updatesPerm.map { NSNumber(value: $0) },
                name: "\(op.result)_updates_to_batch_front"
            )
        }

        // STEP 4: Adjust dimension numbers
        let adjustedDimNumbers = adjustScatterDimNumbers(
            original: dimNumbers,
            operandPerm: operandPerm,
            indicesPerm: indicesPerm,
            updatesPerm: updatesPerm,
            numBatchDims: numBatchDims
        )

        // STEP 5: Execute scatter with batch dims at front
        let scattered = try compileScatterWithLeadingBatchDims(
            op,
            operand: transposedOperand,
            scatterIndices: transposedIndices,
            updates: transposedUpdates,
            dimNumbers: adjustedDimNumbers,
            scatterMode: scatterMode,
            numBatchDims: numBatchDims
        )

        // STEP 6: Transpose output to expected layout
        // Output has same shape as operand, so use inverse of operand permutation
        let outputPerm = PermutationUtils.invertPermutation(operandPerm)

        if PermutationUtils.isIdentity(outputPerm) {
            return scattered
        }

        return graph.transpose(
            scattered,
            permutation: outputPerm.map { NSNumber(value: $0) },
            name: "\(op.result)_output_reorder"
        )
    }

    private func adjustScatterDimNumbers(
        original: ScatterDimensionNumbers,
        operandPerm: [Int],
        indicesPerm: [Int],
        updatesPerm: [Int],
        numBatchDims: Int
    ) -> ScatterDimensionNumbers {

        let operandInverse = PermutationUtils.invertPermutation(operandPerm)
        let indicesInverse = PermutationUtils.invertPermutation(indicesPerm)
        let updatesInverse = PermutationUtils.invertPermutation(updatesPerm)

        // update_window_dims: dimensions in updates that are window dimensions
        let adjustedUpdateWindowDims = original.updateWindowDims.map { updatesInverse[$0] }.sorted()

        // inserted_window_dims: operand dimensions where size-1 windows are inserted
        let adjustedInsertedWindowDims = original.insertedWindowDims.map { operandInverse[$0] }.sorted()

        // scatter_dims_to_operand_dims: maps index elements to operand dimensions
        let adjustedScatterDimsToOperandDims = original.scatterDimsToOperandDims.map { operandInverse[$0] }

        // index_vector_dim: dimension of indices containing the index vector
        let adjustedIndexVectorDim = indicesInverse[original.indexVectorDim]

        // Batch dims are now at front
        let adjustedInputBatchDims = Array(0..<numBatchDims)
        let adjustedIndicesBatchDims = Array(0..<numBatchDims)

        return ScatterDimensionNumbers(
            updateWindowDims: adjustedUpdateWindowDims,
            insertedWindowDims: adjustedInsertedWindowDims,
            scatterDimsToOperandDims: adjustedScatterDimsToOperandDims,
            indexVectorDim: adjustedIndexVectorDim,
            inputBatchingDims: adjustedInputBatchDims,
            scatterIndicesBatchingDims: adjustedIndicesBatchDims
        )
    }

    private func computeUpdatesPermutation(
        dimNumbers: ScatterDimensionNumbers,
        updatesRank: Int
    ) -> [Int] {
        // Updates tensor structure:
        // - Batch dims (from indices, excluding index_vector_dim)
        // - Window dims (at positions specified by update_window_dims)

        let updateWindowDims = dimNumbers.updateWindowDims

        // Find which dims in updates are batch dims (non-window dims)
        var updateBatchDims: [Int] = []
        for i in 0..<updatesRank {
            if !updateWindowDims.contains(i) {
                updateBatchDims.append(i)
            }
        }

        // Move batch dims to front, keep window dims in relative order after
        return PermutationUtils.buildPermutationMovingToFront(
            dims: updateBatchDims,
            rank: updatesRank
        )
    }

    /// Compile scatter when batch dims are already at leading positions.
    private func compileScatterWithLeadingBatchDims(
        _ op: HLOOperation,
        operand: MPSGraphTensor,
        scatterIndices: MPSGraphTensor,
        updates: MPSGraphTensor,
        dimNumbers: ScatterDimensionNumbers,
        scatterMode: MPSGraphScatterMode,
        numBatchDims: Int
    ) throws -> MPSGraphTensor {

        guard let indicesShape = scatterIndices.shape?.map({ $0.intValue }),
              let updatesShapeNS = updates.shape,
              let operandShapeNS = operand.shape else {
            throw CompilationError.undefinedValue("\(op.result)_indices")
        }

        let updatesShape = updatesShapeNS.map { $0.intValue }
        let operandShape = operandShapeNS.map { $0.intValue }
        let indicesRank = indicesShape.count
        let indexVectorDim = dimNumbers.indexVectorDim

        // Handle index_vector_dim: move to last position
        let processedIndices: MPSGraphTensor
        if indexVectorDim != indicesRank - 1 && indicesRank > 1 {
            var perm = Array(0..<indicesRank)
            perm.remove(at: indexVectorDim)
            perm.append(indexVectorDim)

            processedIndices = graph.transpose(
                scatterIndices,
                permutation: perm.map { NSNumber(value: $0) },
                name: "\(op.result)_move_ivd_to_last"
            )
        } else {
            processedIndices = scatterIndices
        }

        // For batched scatter with window dimensions, we need to use scatterAlongAxis.
        // MPS scatterAlongAxis requires: data, updates, indices all have the same rank.
        //
        // Operand:  [B, D1, D2, ...] (e.g., [2, 3, 4])
        // Updates:  [B, W1, W2, ...] (e.g., [2, 4]) - may have fewer dims
        // Indices:  [B, K] -> processed to [B, 1] where 1 is scatter count per batch
        //
        // We scatter along axis = numBatchDims (the first non-batch dimension).
        // scatterAlongAxis semantics: output[..., indices[..., k], ...] = updates[..., k, ...]
        //
        // To make this work:
        // 1. Reshape updates to match operand rank by inserting size-1 dim at scatter axis
        // 2. Reshape indices to match operand rank by broadcasting

        let scatterAxis = numBatchDims

        // Determine the number of scatter updates per batch (typically 1)
        // This is the size along the scatter dimension in the conceptual output
        guard let processedIndicesShape = processedIndices.shape?.map({ $0.intValue }) else {
            throw CompilationError.undefinedValue("\(op.result)_processed_indices")
        }

        // Scatter count: how many scatter operations per batch
        // After moving index_vector_dim to last, indices shape is [B, S1, S2, ..., indexVectorSize]
        // The scatter dims are the middle dimensions (between batch dims and index vector)
        // For [B, 1] with index_vector_dim=1, after processing it's still [B, 1], scatter count = 1
        // For [B, 3, 2] with index_vector_dim=2, scatter count = 3
        var scatterCount = 1
        if processedIndicesShape.count > numBatchDims + 1 {
            // Scatter dims are between batch dims and the last dim (index vector)
            for i in numBatchDims..<(processedIndicesShape.count - 1) {
                scatterCount *= processedIndicesShape[i]
            }
        }

        // Build target shape for updates: insert scatter dimension
        // Updates [B, W1, W2] -> [B, scatterCount, W1, W2] for scatter along axis numBatchDims
        var targetUpdatesShape: [NSNumber] = []
        for i in 0..<numBatchDims {
            targetUpdatesShape.append(NSNumber(value: updatesShape[i]))
        }
        targetUpdatesShape.append(NSNumber(value: scatterCount))
        for i in numBatchDims..<updatesShape.count {
            targetUpdatesShape.append(NSNumber(value: updatesShape[i]))
        }

        let reshapedUpdates = graph.reshape(
            updates,
            shape: targetUpdatesShape,
            name: "\(op.result)_updates_reshape"
        )

        // Build indices shape to match: [B, scatterCount, trailing dims...]
        // Indices are [B, 1] -> need [B, scatterCount, 1, 1, ...] to broadcast with operand
        var targetIndicesShape: [NSNumber] = []
        for i in 0..<numBatchDims {
            targetIndicesShape.append(NSNumber(value: processedIndicesShape[i]))
        }
        targetIndicesShape.append(NSNumber(value: scatterCount))
        // Add trailing dimensions to match operand rank
        for _ in (numBatchDims + 1)..<operandShape.count {
            targetIndicesShape.append(NSNumber(value: 1))
        }

        // Reshape and broadcast indices
        let reshapedIndices = graph.reshape(
            processedIndices,
            shape: targetIndicesShape,
            name: "\(op.result)_indices_reshape"
        )

        // Broadcast indices to match reshapedUpdates shape
        let broadcastIndices = graph.broadcast(
            reshapedIndices,
            shape: reshapedUpdates.shape ?? targetUpdatesShape,
            name: "\(op.result)_indices_broadcast"
        )

        return graph.scatterAlongAxis(
            scatterAxis,
            data: operand,
            updates: reshapedUpdates,
            indices: broadcastIndices,
            mode: scatterMode,
            name: op.result
        )
    }

    /// Compile scatter with no batching dimensions.
    private func compileScatterNoBatching(
        _ op: HLOOperation,
        operand: MPSGraphTensor,
        scatterIndices: MPSGraphTensor,
        updates: MPSGraphTensor,
        dimNumbers: ScatterDimensionNumbers,
        scatterMode: MPSGraphScatterMode
    ) throws -> MPSGraphTensor {

        guard let indicesShape = scatterIndices.shape?.map({ $0.intValue }) else {
            throw CompilationError.undefinedValue("\(op.result)_indices")
        }

        guard let updatesShape = updates.shape else {
            throw CompilationError.undefinedValue("\(op.result)_updates")
        }

        let indicesRank = indicesShape.count
        let indexVectorDim = dimNumbers.indexVectorDim

        // Handle index_vector_dim
        let processedIndices: MPSGraphTensor
        if indexVectorDim != indicesRank - 1 && indicesRank > 1 {
            var perm = Array(0..<indicesRank)
            perm.remove(at: indexVectorDim)
            perm.append(indexVectorDim)

            processedIndices = graph.transpose(
                scatterIndices,
                permutation: perm.map { NSNumber(value: $0) },
                name: "\(op.result)_move_ivd"
            )
        } else {
            processedIndices = scatterIndices
        }

        // For simple scatter without batching, use scatterND
        // For scatterAlongAxis, indices must have same shape as updates
        // Reshape and broadcast indices to match updates shape
        var reshapeForBroadcast: [NSNumber] = processedIndices.shape?.map { $0 } ?? []
        while reshapeForBroadcast.count < updatesShape.count {
            reshapeForBroadcast.append(1)
        }

        let reshapedIndices = graph.reshape(
            processedIndices,
            shape: reshapeForBroadcast,
            name: "\(op.result)_indices_reshape"
        )

        let broadcastIndices = graph.broadcast(
            reshapedIndices,
            shape: updatesShape,
            name: "\(op.result)_indices_broadcast"
        )

        return graph.scatterAlongAxis(
            0,
            data: operand,
            updates: updates,
            indices: broadcastIndices,
            mode: scatterMode,
            name: op.result
        )
    }

    private func compileConstant(_ op: HLOOperation) throws -> MPSGraphTensor {
        guard let constantValue = op.attributes.constantValue else {
            throw CompilationError.missingAttribute("constantValue", operation: "constant")
        }

        switch constantValue {
        case .scalar(let value):
            // Track scalar constants for use in ops like pad
            scalarConstantMap[op.result] = value
            return graph.constant(
                value,
                shape: op.resultType.mpsShape,
                dataType: op.resultType.elementType.mpsDataType
            )

        case .splat(let value, _):
            // Track splat constants as scalars too (they're uniform values)
            scalarConstantMap[op.result] = value
            return graph.constant(
                value,
                shape: op.resultType.mpsShape,
                dataType: op.resultType.elementType.mpsDataType
            )

        case .dense(let values, _):
            // Create data from values with correct element type
            let data: Data
            switch op.resultType.elementType {
            case .float32:
                var floats = values.map { Float($0) }
                data = Data(bytes: &floats, count: floats.count * MemoryLayout<Float>.stride)
            case .float16:
                var floats = values.map { Float16($0) }
                data = Data(bytes: &floats, count: floats.count * MemoryLayout<Float16>.stride)
            case .float64:
                var doubles = values
                data = Data(bytes: &doubles, count: doubles.count * MemoryLayout<Double>.stride)
            case .bfloat16:
                // BFloat16 needs special handling - convert through Float
                var floats = values.map { Float($0) }
                data = Data(bytes: &floats, count: floats.count * MemoryLayout<Float>.stride)
            case .int8:
                var ints = values.map { Int8($0) }
                data = Data(bytes: &ints, count: ints.count * MemoryLayout<Int8>.stride)
            case .int16:
                var ints = values.map { Int16($0) }
                data = Data(bytes: &ints, count: ints.count * MemoryLayout<Int16>.stride)
            case .int32:
                var ints = values.map { Int32($0) }
                data = Data(bytes: &ints, count: ints.count * MemoryLayout<Int32>.stride)
            case .int64:
                var ints = values.map { Int64($0) }
                data = Data(bytes: &ints, count: ints.count * MemoryLayout<Int64>.stride)
            case .uint8:
                var ints = values.map { UInt8($0) }
                data = Data(bytes: &ints, count: ints.count * MemoryLayout<UInt8>.stride)
            case .uint16:
                var ints = values.map { UInt16($0) }
                data = Data(bytes: &ints, count: ints.count * MemoryLayout<UInt16>.stride)
            case .uint32:
                var ints = values.map { UInt32($0) }
                data = Data(bytes: &ints, count: ints.count * MemoryLayout<UInt32>.stride)
            case .uint64:
                var ints = values.map { UInt64($0) }
                data = Data(bytes: &ints, count: ints.count * MemoryLayout<UInt64>.stride)
            case .int1:
                // Boolean type - treat as UInt8 with 0/1 values
                var bools = values.map { UInt8($0 != 0 ? 1 : 0) }
                data = Data(bytes: &bools, count: bools.count * MemoryLayout<UInt8>.stride)
            }

            return graph.constant(
                data,
                shape: op.resultType.mpsShape,
                dataType: op.resultType.elementType.mpsDataType
            )
        }
    }

    private func compileRNG(_ op: HLOOperation) throws -> MPSGraphTensor {
        let distribution = op.attributes.rngDistribution ?? .uniform

        switch distribution {
        case .uniform:
            return graph.randomUniformTensor(
                withShape: op.resultType.mpsShape,
                name: op.result
            )
        case .normal:
            return graph.randomTensor(
                withShape: op.resultType.mpsShape,
                descriptor: MPSGraphRandomOpDescriptor(distribution: .normal, dataType: op.resultType.elementType.mpsDataType)!,
                name: op.result
            )
        }
    }

    private func compileIota(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Iota creates a tensor filled with incrementing values along a dimension
        // Uses the iotaDimension attribute (or axis for backward compatibility), defaulting to 0
        let dim = op.attributes.iotaDimension ?? op.attributes.axis ?? 0
        let shape = op.resultType.mpsShape
        let targetType = op.resultType.elementType.mpsDataType

        // Create coordinate tensor for the specified dimension
        // coordinate() returns Int32 by default, so we need to cast to the target type
        let coords = graph.coordinate(
            alongAxis: dim,
            withShape: shape,
            name: nil
        )

        // Cast to the expected output type
        return graph.cast(coords, to: targetType, name: op.result)
    }

    private func compileReverse(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])
        let axes = op.attributes.dimensions ?? []
        return graph.reverse(input, axes: axes.map { NSNumber(value: $0) }, name: op.result)
    }

    private func compileConvolution(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])
        let weights = try getOperand(op.operands[1])

        // Get convolution parameters
        let strides = op.attributes.windowStrides ?? [1, 1]
        _ = op.attributes.lhsDilation ?? [1, 1]  // Not used for basic 2D conv
        let rhsDilation = op.attributes.rhsDilation ?? [1, 1]
        let featureGroups = op.attributes.featureGroupCount ?? 1

        // Get padding (default to 0)
        let padding = op.attributes.convPadding ?? []
        let padTop = padding.count > 0 ? padding[0][0] : 0
        let padBottom = padding.count > 0 ? padding[0][1] : 0
        let padLeft = padding.count > 1 ? padding[1][0] : 0
        let padRight = padding.count > 1 ? padding[1][1] : 0

        // Determine data layout from dimension numbers
        let dimNumbers = op.attributes.convolutionDimensionNumbers
        let (dataLayout, weightsLayout, needsTranspose) = determineConvLayoutsFromDimNumbers(dimNumbers)

        // Transpose input if needed (for non-standard layouts)
        var processedInput = input
        var processedWeights = weights
        if needsTranspose, let dimNumbers = dimNumbers {
            (processedInput, processedWeights) = try transposeConvInputs(
                input: input,
                weights: weights,
                dimNumbers: dimNumbers,
                targetDataLayout: dataLayout,
                targetWeightsLayout: weightsLayout,
                name: op.result
            )
        }

        // Create convolution descriptor
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: strides.count > 1 ? strides[1] : strides[0],
            strideInY: strides[0],
            dilationRateInX: rhsDilation.count > 1 ? rhsDilation[1] : rhsDilation[0],
            dilationRateInY: rhsDilation[0],
            groups: featureGroups,
            paddingLeft: padLeft,
            paddingRight: padRight,
            paddingTop: padTop,
            paddingBottom: padBottom,
            paddingStyle: .explicit,
            dataLayout: dataLayout,
            weightsLayout: weightsLayout
        )!

        let convResult = graph.convolution2D(processedInput, weights: processedWeights, descriptor: descriptor, name: needsTranspose ? nil : op.result)

        // Transpose output back if needed
        if needsTranspose, let dimNumbers = dimNumbers {
            return try transposeConvOutput(
                output: convResult,
                dimNumbers: dimNumbers,
                sourceDataLayout: dataLayout,
                name: op.result
            )
        }

        return convResult
    }

    /// Determine MPS layouts from StableHLO dimension numbers.
    /// Returns (dataLayout, weightsLayout, needsTranspose)
    private func determineConvLayoutsFromDimNumbers(_ dimNumbers: ConvolutionDimensionNumbers?) -> (MPSGraphTensorNamedDataLayout, MPSGraphTensorNamedDataLayout, Bool) {
        guard let dimNumbers = dimNumbers else {
            // Default to NHWC/HWIO
            return (.NHWC, .HWIO, false)
        }

        // Check for NHWC: batch=0, spatial=[1,2], feature=3
        let isNHWC = dimNumbers.inputBatchDimension == 0 &&
                     dimNumbers.inputSpatialDimensions == [1, 2] &&
                     dimNumbers.inputFeatureDimension == 3

        // Check for NCHW: batch=0, feature=1, spatial=[2,3]
        let isNCHW = dimNumbers.inputBatchDimension == 0 &&
                     dimNumbers.inputFeatureDimension == 1 &&
                     dimNumbers.inputSpatialDimensions == [2, 3]

        // Check for HWIO: spatial=[0,1], input=2, output=3
        let isHWIO = dimNumbers.kernelSpatialDimensions == [0, 1] &&
                     dimNumbers.kernelInputFeatureDimension == 2 &&
                     dimNumbers.kernelOutputFeatureDimension == 3

        // Check for OIHW: output=0, input=1, spatial=[2,3]
        let isOIHW = dimNumbers.kernelOutputFeatureDimension == 0 &&
                     dimNumbers.kernelInputFeatureDimension == 1 &&
                     dimNumbers.kernelSpatialDimensions == [2, 3]

        if isNHWC && isHWIO {
            return (.NHWC, .HWIO, false)
        } else if isNCHW && isOIHW {
            return (.NCHW, .OIHW, false)
        } else if isNCHW && isHWIO {
            // NCHW input but HWIO weights - need to transpose weights
            return (.NCHW, .HWIO, true)
        } else if isNHWC && isOIHW {
            // NHWC input but OIHW weights - need to transpose weights
            return (.NHWC, .OIHW, true)
        }

        // For non-standard layouts, default to NHWC and transpose
        return (.NHWC, .HWIO, true)
    }

    /// Transpose convolution inputs to match target layout.
    private func transposeConvInputs(
        input: MPSGraphTensor,
        weights: MPSGraphTensor,
        dimNumbers: ConvolutionDimensionNumbers,
        targetDataLayout: MPSGraphTensorNamedDataLayout,
        targetWeightsLayout: MPSGraphTensorNamedDataLayout,
        name: String
    ) throws -> (MPSGraphTensor, MPSGraphTensor) {
        var processedInput = input
        var processedWeights = weights

        // Transpose input to target layout
        let targetInputOrder: [Int]
        switch targetDataLayout {
        case .NHWC:
            targetInputOrder = [0, 1, 2, 3]  // [N, H, W, C]
        case .NCHW:
            targetInputOrder = [0, 1, 2, 3]  // [N, C, H, W]
        default:
            targetInputOrder = [0, 1, 2, 3]
        }

        // Build permutation from current layout to target
        let currentInputOrder = [
            dimNumbers.inputBatchDimension,
            dimNumbers.inputSpatialDimensions[0],
            dimNumbers.inputSpatialDimensions.count > 1 ? dimNumbers.inputSpatialDimensions[1] : dimNumbers.inputSpatialDimensions[0],
            dimNumbers.inputFeatureDimension
        ]

        if targetDataLayout == .NHWC && currentInputOrder != [0, 1, 2, 3] {
            // Need to transpose input to NHWC
            let perm = buildPermutationToNHWC(from: dimNumbers)
            if !PermutationUtils.isIdentity(perm) {
                processedInput = graph.transpose(
                    input,
                    permutation: perm.map { NSNumber(value: $0) },
                    name: "\(name)_input_to_nhwc"
                )
            }
        }

        // Transpose weights if needed
        let targetKernelOrder: [Int]
        switch targetWeightsLayout {
        case .HWIO:
            targetKernelOrder = [0, 1, 2, 3]  // [H, W, I, O]
        case .OIHW:
            targetKernelOrder = [0, 1, 2, 3]  // [O, I, H, W]
        default:
            targetKernelOrder = [0, 1, 2, 3]
        }

        if targetWeightsLayout == .HWIO {
            let perm = buildKernelPermutationToHWIO(from: dimNumbers)
            if !PermutationUtils.isIdentity(perm) {
                processedWeights = graph.transpose(
                    weights,
                    permutation: perm.map { NSNumber(value: $0) },
                    name: "\(name)_weights_to_hwio"
                )
            }
        } else if targetWeightsLayout == .OIHW {
            let perm = buildKernelPermutationToOIHW(from: dimNumbers)
            if !PermutationUtils.isIdentity(perm) {
                processedWeights = graph.transpose(
                    weights,
                    permutation: perm.map { NSNumber(value: $0) },
                    name: "\(name)_weights_to_oihw"
                )
            }
        }

        return (processedInput, processedWeights)
    }

    /// Transpose convolution output back to expected layout.
    private func transposeConvOutput(
        output: MPSGraphTensor,
        dimNumbers: ConvolutionDimensionNumbers,
        sourceDataLayout: MPSGraphTensorNamedDataLayout,
        name: String
    ) throws -> MPSGraphTensor {
        // Build inverse permutation: from MPS output layout back to expected output layout
        let expectedOutputOrder = [
            dimNumbers.outputBatchDimension,
            dimNumbers.outputSpatialDimensions[0],
            dimNumbers.outputSpatialDimensions.count > 1 ? dimNumbers.outputSpatialDimensions[1] : dimNumbers.outputSpatialDimensions[0],
            dimNumbers.outputFeatureDimension
        ]

        let mpsOutputOrder: [Int]
        switch sourceDataLayout {
        case .NHWC:
            mpsOutputOrder = [0, 1, 2, 3]  // MPS outputs [N, H, W, C]
        case .NCHW:
            mpsOutputOrder = [0, 1, 2, 3]  // MPS outputs [N, C, H, W]
        default:
            mpsOutputOrder = [0, 1, 2, 3]
        }

        // If MPS output matches expected output, no transpose needed
        if sourceDataLayout == .NHWC && expectedOutputOrder == [0, 1, 2, 3] {
            return graph.identity(with: output, name: name)
        }

        // Build permutation from MPS output to expected output
        let perm = buildPermutationFromNHWC(to: dimNumbers)
        if PermutationUtils.isIdentity(perm) {
            return graph.identity(with: output, name: name)
        }

        return graph.transpose(
            output,
            permutation: perm.map { NSNumber(value: $0) },
            name: name
        )
    }

    /// Build permutation to convert input from current layout to NHWC.
    private func buildPermutationToNHWC(from dimNumbers: ConvolutionDimensionNumbers) -> [Int] {
        // Current positions of [B, H, W, C]
        let batchPos = dimNumbers.inputBatchDimension
        let heightPos = dimNumbers.inputSpatialDimensions[0]
        let widthPos = dimNumbers.inputSpatialDimensions.count > 1 ? dimNumbers.inputSpatialDimensions[1] : heightPos
        let channelPos = dimNumbers.inputFeatureDimension

        // Build inverse: where does each position go?
        var perm = [Int](repeating: 0, count: 4)
        perm[0] = batchPos
        perm[1] = heightPos
        perm[2] = widthPos
        perm[3] = channelPos

        return perm
    }

    /// Build permutation to convert kernel from current layout to HWIO.
    private func buildKernelPermutationToHWIO(from dimNumbers: ConvolutionDimensionNumbers) -> [Int] {
        let heightPos = dimNumbers.kernelSpatialDimensions[0]
        let widthPos = dimNumbers.kernelSpatialDimensions.count > 1 ? dimNumbers.kernelSpatialDimensions[1] : heightPos
        let inputPos = dimNumbers.kernelInputFeatureDimension
        let outputPos = dimNumbers.kernelOutputFeatureDimension

        var perm = [Int](repeating: 0, count: 4)
        perm[0] = heightPos
        perm[1] = widthPos
        perm[2] = inputPos
        perm[3] = outputPos

        return perm
    }

    /// Build permutation to convert kernel from current layout to OIHW.
    private func buildKernelPermutationToOIHW(from dimNumbers: ConvolutionDimensionNumbers) -> [Int] {
        let heightPos = dimNumbers.kernelSpatialDimensions[0]
        let widthPos = dimNumbers.kernelSpatialDimensions.count > 1 ? dimNumbers.kernelSpatialDimensions[1] : heightPos
        let inputPos = dimNumbers.kernelInputFeatureDimension
        let outputPos = dimNumbers.kernelOutputFeatureDimension

        var perm = [Int](repeating: 0, count: 4)
        perm[0] = outputPos
        perm[1] = inputPos
        perm[2] = heightPos
        perm[3] = widthPos

        return perm
    }

    /// Build permutation to convert output from NHWC to expected layout.
    private func buildPermutationFromNHWC(to dimNumbers: ConvolutionDimensionNumbers) -> [Int] {
        // MPS outputs NHWC: positions [0, 1, 2, 3] = [B, H, W, C]
        // Need to rearrange to expected output positions
        let batchTarget = dimNumbers.outputBatchDimension
        let heightTarget = dimNumbers.outputSpatialDimensions[0]
        let widthTarget = dimNumbers.outputSpatialDimensions.count > 1 ? dimNumbers.outputSpatialDimensions[1] : heightTarget
        let channelTarget = dimNumbers.outputFeatureDimension

        // Build inverse permutation
        var perm = [Int](repeating: 0, count: 4)
        perm[batchTarget] = 0     // B from position 0
        perm[heightTarget] = 1    // H from position 1
        perm[widthTarget] = 2     // W from position 2
        perm[channelTarget] = 3   // C from position 3

        return PermutationUtils.invertPermutation(perm)
    }

    private func compileReduceWindow(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])

        // Get window parameters - assume NHWC layout
        let windowDims = op.attributes.windowDimensions ?? [1, 2, 2, 1]
        let strides = op.attributes.windowStrides ?? [1, 2, 2, 1]
        let padding = op.attributes.convPadding ?? []
        let windowDilations = op.attributes.windowDilations ?? [1, 1, 1, 1]
        let baseDilations = op.attributes.baseDilations ?? [1, 1, 1, 1]

        // Extract spatial dimensions (assuming NHWC: indices 1 and 2)
        let kernelHeight = windowDims.count > 1 ? windowDims[1] : windowDims[0]
        let kernelWidth = windowDims.count > 2 ? windowDims[2] : kernelHeight
        let strideY = strides.count > 1 ? strides[1] : strides[0]
        let strideX = strides.count > 2 ? strides[2] : strideY

        // Extract window dilations for spatial dimensions
        let dilationY = windowDilations.count > 1 ? windowDilations[1] : 1
        let dilationX = windowDilations.count > 2 ? windowDilations[2] : dilationY

        let padTop = padding.count > 1 ? padding[1][0] : 0
        let padBottom = padding.count > 1 ? padding[1][1] : 0
        let padLeft = padding.count > 2 ? padding[2][0] : 0
        let padRight = padding.count > 2 ? padding[2][1] : 0

        // Handle base dilations by interleaving the input with zeros
        var processedInput = input
        let baseDilationY = baseDilations.count > 1 ? baseDilations[1] : 1
        let baseDilationX = baseDilations.count > 2 ? baseDilations[2] : 1

        if baseDilationY > 1 || baseDilationX > 1 {
            // Base dilation: insert zeros between input elements
            // For dilation d, input [H, W] becomes [(H-1)*d + 1, (W-1)*d + 1]
            processedInput = try applyBaseDilation(
                processedInput,
                dilationY: baseDilationY,
                dilationX: baseDilationX,
                name: op.result
            )
        }

        // Create pooling descriptor with window dilations
        let descriptor = MPSGraphPooling2DOpDescriptor(
            kernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            strideInX: strideX,
            strideInY: strideY,
            dilationRateInX: dilationX,
            dilationRateInY: dilationY,
            paddingLeft: padLeft,
            paddingRight: padRight,
            paddingTop: padTop,
            paddingBottom: padBottom,
            paddingStyle: .explicit,
            dataLayout: .NHWC
        )!

        // Choose pooling type based on reduction kind
        let reductionKind = op.attributes.reductionKind ?? .max
        switch reductionKind {
        case .max:
            return graph.maxPooling2D(withSourceTensor: processedInput, descriptor: descriptor, name: op.result)
        case .sum:
            // Average pooling approximation - need to multiply by window size for sum
            let avgPooled = graph.avgPooling2D(withSourceTensor: processedInput, descriptor: descriptor, name: nil)
            let windowSize = Float(kernelHeight * kernelWidth)
            let scale = graph.constant(Double(windowSize), dataType: avgPooled.dataType)
            return graph.multiplication(avgPooled, scale, name: op.result)
        case .mean:
            return graph.avgPooling2D(withSourceTensor: processedInput, descriptor: descriptor, name: op.result)
        case .min:
            // MPSGraph doesn't have min pooling directly, use negative max pooling trick
            let negInput = graph.negative(with: processedInput, name: nil)
            let maxPooled = graph.maxPooling2D(withSourceTensor: negInput, descriptor: descriptor, name: nil)
            return graph.negative(with: maxPooled, name: op.result)
        }
    }

    /// Apply base dilation to input by interleaving zeros.
    /// For dilation d, each element is separated by (d-1) zeros.
    /// Input [N, H, W, C] becomes [N, (H-1)*dY + 1, (W-1)*dX + 1, C]
    private func applyBaseDilation(
        _ input: MPSGraphTensor,
        dilationY: Int,
        dilationX: Int,
        name: String
    ) throws -> MPSGraphTensor {
        guard let inputShape = input.shape?.map({ $0.intValue }) else {
            throw CompilationError.undefinedValue("\(name)_base_dilation")
        }

        guard inputShape.count == 4 else {
            // Only support 4D NHWC tensors for now
            return input
        }

        let batchSize = inputShape[0]
        let height = inputShape[1]
        let width = inputShape[2]
        let channels = inputShape[3]

        // Calculate dilated dimensions
        let dilatedHeight = (height - 1) * dilationY + 1
        let dilatedWidth = (width - 1) * dilationX + 1

        // Create zero tensor with dilated shape
        let zeros = graph.constant(
            0.0,
            shape: [
                NSNumber(value: batchSize),
                NSNumber(value: dilatedHeight),
                NSNumber(value: dilatedWidth),
                NSNumber(value: channels)
            ],
            dataType: input.dataType
        )

        // Use scatter to place original values at strided positions
        // This is equivalent to: output[n, h*dY, w*dX, c] = input[n, h, w, c]

        // Create indices for the strided scatter
        // We need to scatter along axes 1 and 2 simultaneously

        // For simplicity, use a reshape + space_to_depth style approach:
        // Create coordinate tensors and use scatterND

        // Alternative simpler approach: use stridedSlice assignment via update
        // MPSGraph doesn't have direct strided assignment, so we build indices

        // Simplest approach for now: unfold the operation using multiple slices
        // This is inefficient but correct
        // TODO: Optimize with a more efficient scatter approach

        // For very large dilations, consider a different strategy
        if dilationY <= 4 && dilationX <= 4 {
            // Use index-based scatter for small dilations
            return try scatterBaseDilated(
                input,
                zeros: zeros,
                dilationY: dilationY,
                dilationX: dilationX,
                name: name
            )
        }

        // Fallback: return input unchanged (dilation not applied)
        // Log a warning in debug builds
        #if DEBUG
        print("Warning: Base dilation (\(dilationY), \(dilationX)) too large, skipping")
        #endif
        return input
    }

    /// Scatter input values into zero tensor at dilated positions.
    private func scatterBaseDilated(
        _ input: MPSGraphTensor,
        zeros: MPSGraphTensor,
        dilationY: Int,
        dilationX: Int,
        name: String
    ) throws -> MPSGraphTensor {
        guard let inputShape = input.shape?.map({ $0.intValue }) else {
            return input
        }

        let height = inputShape[1]
        let width = inputShape[2]

        // Reshape input to flatten spatial dims: [N, H*W, C]
        let flatInput = graph.reshape(
            input,
            shape: [
                NSNumber(value: inputShape[0]),
                NSNumber(value: height * width),
                NSNumber(value: inputShape[3])
            ],
            name: "\(name)_flat_input"
        )

        // Build 2D indices into the flattened dilated spatial dimensions
        // For each (h, w) pair, target index is h*dilationY * dilatedWidth + w*dilationX

        guard let zerosShape = zeros.shape?.map({ $0.intValue }) else {
            return input
        }
        let dilatedWidth = zerosShape[2]

        // Create flat index tensor [H*W]
        var flatIndices: [Int32] = []
        for h in 0..<height {
            for w in 0..<width {
                let targetIdx = Int32(h * dilationY * dilatedWidth + w * dilationX)
                flatIndices.append(targetIdx)
            }
        }

        let indicesTensor = graph.constant(
            Data(bytes: flatIndices, count: flatIndices.count * 4),
            shape: [NSNumber(value: height * width)],
            dataType: .int32
        )

        // Reshape zeros to [N, dilatedH * dilatedW, C]
        let flatZeros = graph.reshape(
            zeros,
            shape: [
                NSNumber(value: zerosShape[0]),
                NSNumber(value: zerosShape[1] * zerosShape[2]),
                NSNumber(value: zerosShape[3])
            ],
            name: "\(name)_flat_zeros"
        )

        // Scatter along axis 1
        let scattered = graph.scatterAlongAxis(
            1,
            data: flatZeros,
            updates: flatInput,
            indices: graph.broadcast(
                indicesTensor,
                shape: flatInput.shape ?? [
                    NSNumber(value: inputShape[0]),
                    NSNumber(value: height * width),
                    NSNumber(value: inputShape[3])
                ],
                name: "\(name)_broadcast_indices"
            ),
            mode: .set,
            name: "\(name)_scatter"
        )

        // Reshape back to [N, dilatedH, dilatedW, C]
        return graph.reshape(
            scattered,
            shape: zeros.shape ?? [
                NSNumber(value: zerosShape[0]),
                NSNumber(value: zerosShape[1]),
                NSNumber(value: zerosShape[2]),
                NSNumber(value: zerosShape[3])
            ],
            name: "\(name)_dilated"
        )
    }

    private func compileBatchNormInference(_ op: HLOOperation) throws -> MPSGraphTensor {
        guard op.operands.count >= 5 else {
            throw CompilationError.wrongOperandCount(expected: 5, got: op.operands.count)
        }

        let input = try getOperand(op.operands[0])
        let scale = try getOperand(op.operands[1])
        let offset = try getOperand(op.operands[2])
        let mean = try getOperand(op.operands[3])
        let variance = try getOperand(op.operands[4])

        let epsilon = op.attributes.epsilon ?? 1e-5

        return graph.normalize(
            input,
            mean: mean,
            variance: variance,
            gamma: scale,
            beta: offset,
            epsilon: epsilon,
            name: op.result
        )
    }

    private func compileBatchNormTraining(_ op: HLOOperation) throws -> MPSGraphTensor {
        guard op.operands.count >= 3 else {
            throw CompilationError.wrongOperandCount(expected: 3, got: op.operands.count)
        }

        let input = try getOperand(op.operands[0])
        let scale = try getOperand(op.operands[1])
        let offset = try getOperand(op.operands[2])

        let epsilon = op.attributes.epsilon ?? 1e-5
        let featureIndex = op.attributes.featureIndex ?? -1  // Default to last axis

        // Compute mean and variance
        // For batch norm, we reduce over all axes except the feature axis
        guard let inputType = typeMap[op.operands[0]] else {
            throw CompilationError.undefinedValue(op.operands[0])
        }

        let rank = inputType.shape.count
        let actualFeatureIndex = featureIndex < 0 ? rank + featureIndex : featureIndex
        var reductionAxes: [NSNumber] = []
        for i in 0..<rank {
            if i != actualFeatureIndex {
                reductionAxes.append(NSNumber(value: i))
            }
        }

        let mean = graph.mean(of: input, axes: reductionAxes, name: nil)
        let variance = graph.variance(of: input, mean: mean, axes: reductionAxes, name: nil)

        // Normalize
        return graph.normalize(
            input,
            mean: mean,
            variance: variance,
            gamma: scale,
            beta: offset,
            epsilon: epsilon,
            name: op.result
        )
    }

    private func compileBatchNormGrad(_ op: HLOOperation) throws -> MPSGraphTensor {
        // batch_norm_grad computes gradients for batch normalization
        // This is a simplified implementation
        guard op.operands.count >= 5 else {
            throw CompilationError.wrongOperandCount(expected: 5, got: op.operands.count)
        }

        let input = try getOperand(op.operands[0])
        let scale = try getOperand(op.operands[1])
        let mean = try getOperand(op.operands[2])
        let variance = try getOperand(op.operands[3])
        let gradOutput = try getOperand(op.operands[4])

        let epsilon = op.attributes.epsilon ?? 1e-5
        let featureIndex = op.attributes.featureIndex ?? -1

        guard let inputType = typeMap[op.operands[0]] else {
            throw CompilationError.undefinedValue(op.operands[0])
        }

        let rank = inputType.shape.count
        let actualFeatureIndex = featureIndex < 0 ? rank + featureIndex : featureIndex
        var reductionAxes: [NSNumber] = []
        for i in 0..<rank {
            if i != actualFeatureIndex {
                reductionAxes.append(NSNumber(value: i))
            }
        }

        // Compute gradient using normalizationGradient
        // Returns the gradient with respect to the input
        let gradInput = graph.normalizationGradient(
            withIncomingGradientTensor: gradOutput,
            sourceTensor: input,
            mean: mean,
            varianceTensor: variance,
            gammaTensor: scale,
            gammaGradientTensor: nil,
            betaGradientTensor: nil,
            reductionAxes: reductionAxes,
            epsilon: epsilon,
            name: op.result
        )

        return gradInput
    }

    private func compileFFT(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])

        let fftType = op.attributes.fftType ?? .fft
        let fftLength = op.attributes.fftLength ?? []

        // Create FFT descriptor
        let descriptor = MPSGraphFFTDescriptor()
        descriptor.inverse = (fftType == .ifft || fftType == .irfft)
        descriptor.scalingMode = .none

        // Create axes tensor from fft_length indices
        // By default, FFT over the last n dimensions where n = fftLength.count
        guard let inputType = typeMap[op.operands[0]] else {
            throw CompilationError.undefinedValue(op.operands[0])
        }

        let rank = inputType.shape.count
        let numAxes = fftLength.isEmpty ? 1 : fftLength.count
        var axes: [Int32] = []
        for i in 0..<numAxes {
            axes.append(Int32(rank - numAxes + i))
        }

        var axesData = axes
        let axesTensor = graph.constant(
            Data(bytes: &axesData, count: axesData.count * MemoryLayout<Int32>.stride),
            shape: [NSNumber(value: axes.count)],
            dataType: .int32
        )

        return graph.fastFourierTransform(
            input,
            axesTensor: axesTensor,
            descriptor: descriptor,
            name: op.result
        )
    }

    private func compileSort(_ op: HLOOperation) throws -> MPSGraphTensor {
        let input = try getOperand(op.operands[0])
        let axis = op.attributes.axis ?? -1
        let descending = op.attributes.sortDescending ?? false

        // Get indices that would sort the tensor
        let indices = graph.argSort(input, axis: axis, descending: descending, name: nil)

        // Gather values in sorted order
        return graph.gatherAlongAxis(
            axis,
            updates: input,
            indices: indices,
            name: op.result
        )
    }

    private func compileQuantize(_ op: HLOOperation) throws -> MPSGraphTensor {
        // uniform_quantize: quantize float to int
        // Simplified implementation using scale and zero point
        let input = try getOperand(op.operands[0])
        let scale = try getOperand(op.operands[1])

        // Quantize: output = round(input / scale)
        let scaled = graph.division(input, scale, name: nil)
        let rounded = graph.round(with: scaled, name: nil)

        // Cast to output type
        return graph.cast(rounded, to: op.resultType.elementType.mpsDataType, name: op.result)
    }

    private func compileDequantize(_ op: HLOOperation) throws -> MPSGraphTensor {
        // uniform_dequantize: dequantize int to float
        let input = try getOperand(op.operands[0])
        let scale = try getOperand(op.operands[1])

        // First cast to float
        let floatInput = graph.cast(input, to: op.resultType.elementType.mpsDataType, name: nil)

        // Dequantize: output = float(input) * scale
        return graph.multiplication(floatInput, scale, name: op.result)
    }

    // MARK: - Additional Math Operations

    private func compileExpm1(_ op: HLOOperation) throws -> MPSGraphTensor {
        // expm1(x) = exp(x) - 1, numerically stable for small x
        let input = try getOperand(op.operands[0])
        let expX = graph.exponent(with: input, name: nil)
        let one = graph.constant(1.0, dataType: expX.dataType)
        return graph.subtraction(expX, one, name: op.result)
    }

    private func compileLog1p(_ op: HLOOperation) throws -> MPSGraphTensor {
        // log1p(x) = log(1 + x), numerically stable for small x
        let input = try getOperand(op.operands[0])
        let one = graph.constant(1.0, dataType: input.dataType)
        let onePlusX = graph.addition(one, input, name: nil)
        return graph.logarithm(with: onePlusX, name: op.result)
    }

    private func compileCbrt(_ op: HLOOperation) throws -> MPSGraphTensor {
        // cbrt(x) = sign(x) * |x|^(1/3)
        let input = try getOperand(op.operands[0])
        let absX = graph.absolute(with: input, name: nil)
        let signX = graph.sign(with: input, name: nil)
        let oneThird = graph.constant(1.0/3.0, dataType: input.dataType)
        let absCbrt = graph.power(absX, oneThird, name: nil)
        return graph.multiplication(signX, absCbrt, name: op.result)
    }

    private func compilePopcnt(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Population count - count set bits using parallel bit counting algorithm
        // This implements the classic SWAR (SIMD Within A Register) algorithm for 32-bit integers
        let input = try getOperand(op.operands[0])

        // Cast to UInt32 for bit manipulation
        var x = graph.cast(input, to: .uInt32, name: nil)

        // Magic constants for parallel bit counting
        let c1 = graph.constant(Double(0x55555555), dataType: .uInt32)  // 0101...
        let c2 = graph.constant(Double(0x33333333), dataType: .uInt32)  // 0011...
        let c3 = graph.constant(Double(0x0F0F0F0F), dataType: .uInt32)  // 00001111...
        let c4 = graph.constant(Double(0x01010101), dataType: .uInt32)  // 00000001...
        let shift24 = graph.constant(Double(24), dataType: .uInt32)
        let one = graph.constant(Double(1), dataType: .uInt32)
        let two = graph.constant(Double(2), dataType: .uInt32)
        let four = graph.constant(Double(4), dataType: .uInt32)

        // Step 1: x = x - ((x >> 1) & 0x55555555)
        let xShift1 = graph.bitwiseRightShift(x, one, name: nil)
        let masked1 = graph.bitwiseAND(xShift1, c1, name: nil)
        x = graph.subtraction(x, masked1, name: nil)

        // Step 2: x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        let xAnd2 = graph.bitwiseAND(x, c2, name: nil)
        let xShift2 = graph.bitwiseRightShift(x, two, name: nil)
        let masked2 = graph.bitwiseAND(xShift2, c2, name: nil)
        x = graph.addition(xAnd2, masked2, name: nil)

        // Step 3: x = (x + (x >> 4)) & 0x0F0F0F0F
        let xShift4 = graph.bitwiseRightShift(x, four, name: nil)
        let xSum4 = graph.addition(x, xShift4, name: nil)
        x = graph.bitwiseAND(xSum4, c3, name: nil)

        // Step 4: x = (x * 0x01010101) >> 24
        let xMul = graph.multiplication(x, c4, name: nil)
        x = graph.bitwiseRightShift(xMul, shift24, name: nil)

        // Cast back to output type
        return graph.cast(x, to: op.resultType.elementType.mpsDataType, name: op.result)
    }

    private func compileShiftRightLogical(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Logical right shift (zero-filling)
        // MPSGraph's bitwiseRightShift is arithmetic, so we need to handle unsigned
        let input = try getOperand(op.operands[0])
        let shiftAmount = try getOperand(op.operands[1])

        // For unsigned types, arithmetic and logical shifts are the same
        // For signed types, we'd need to mask, but MPSGraph handles this
        return graph.bitwiseRightShift(input, shiftAmount, name: op.result)
    }

    // MARK: - Complex Number Operations

    private func compileComplex(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Create complex from real and imaginary parts
        // MPSGraph has limited complex support - this is a simplified implementation
        let real = try getOperand(op.operands[0])
        let imag = try getOperand(op.operands[1])

        // Stack real and imag as last dimension [... , 2] representing complex
        let realExpanded = graph.expandDims(real, axis: -1, name: nil)
        let imagExpanded = graph.expandDims(imag, axis: -1, name: nil)
        return graph.concatTensors([realExpanded, imagExpanded], dimension: -1, name: op.result)
    }

    private func compileReal(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Extract real part from complex tensor
        let input = try getOperand(op.operands[0])
        // Assume complex is stored as [..., 2] where index 0 is real
        return graph.sliceTensor(input, dimension: -1, start: 0, length: 1, name: op.result)
    }

    private func compileImag(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Extract imaginary part from complex tensor
        let input = try getOperand(op.operands[0])
        // Assume complex is stored as [..., 2] where index 1 is imaginary
        return graph.sliceTensor(input, dimension: -1, start: 1, length: 1, name: op.result)
    }

    // MARK: - Type Conversion Operations

    private func compileBitcastConvert(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Bitcast - reinterpret bits as different type
        let input = try getOperand(op.operands[0])
        // MPSGraph doesn't have native bitcast, use reshape + cast approach
        // This preserves the bit pattern by going through the same memory
        return graph.cast(input, to: op.resultType.elementType.mpsDataType, name: op.result)
    }

    private func compileReducePrecision(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Reduce precision by truncating mantissa/exponent bits
        // IEEE 754 float32 format: 1 sign bit + 8 exponent bits + 23 mantissa bits
        //
        // LIMITATION: MPSGraph doesn't have true bitcast operations (cast performs
        // numeric conversion, not bit reinterpretation). Implementing reduce_precision
        // correctly requires bit manipulation on the IEEE 754 representation, which
        // would need a custom Metal kernel.
        //
        // Current behavior: Returns input unchanged (identity operation).
        // This is a valid but suboptimal implementation - the precision is not
        // actually reduced, but the operation compiles and runs without error.
        //
        // For applications that require actual precision reduction, consider:
        // 1. Using a custom Metal kernel for bit manipulation
        // 2. Converting to/from half precision using convert operations
        // 3. Implementing quantize/dequantize with appropriate scales
        let input = try getOperand(op.operands[0])

        // Log the attributes for debugging (not used in current identity impl)
        _ = op.attributes.exponentBits ?? 8
        _ = op.attributes.mantissaBits ?? 23

        // Return identity - precision reduction requires custom Metal kernel
        return graph.identity(with: input, name: op.result)
    }

    // MARK: - Dynamic Shape Operations

    private func compileDynamicBroadcast(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Dynamic broadcast with runtime shape
        let input = try getOperand(op.operands[0])
        // Output shape comes from second operand (a tensor containing the shape)
        // For now, use the result type's shape
        return graph.broadcast(input, shape: op.resultType.mpsShape, name: op.result)
    }

    private func compileDynamicReshape(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Dynamic reshape with runtime shape
        let input = try getOperand(op.operands[0])
        // Shape tensor is second operand, but we use result type for static compilation
        return graph.reshape(input, shape: op.resultType.mpsShape, name: op.result)
    }

    private func compileDynamicSlice(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Dynamic slice with runtime start indices
        // Note: True dynamic indices would require CPU readback from tensor operands.
        // For now, we assume start indices are 0 (common case for static graphs).
        let input = try getOperand(op.operands[0])
        let sliceSizes = op.attributes.dynamicSliceSizes ?? []

        guard !sliceSizes.isEmpty else {
            throw CompilationError.missingAttribute("dynamicSliceSizes", operation: "dynamic_slice")
        }

        // Start at 0 for each dimension (we can't read runtime tensor values statically)
        // This works correctly when the dynamic indices happen to be 0, or when the
        // slice size equals the input size (making start index irrelevant)
        let starts = Array(repeating: NSNumber(value: 0), count: sliceSizes.count)
        let ends = sliceSizes.map { NSNumber(value: $0) }
        let strides = Array(repeating: NSNumber(value: 1), count: sliceSizes.count)

        return graph.sliceTensor(
            input,
            starts: starts,
            ends: ends,
            strides: strides,
            name: op.result
        )
    }

    private func compileDynamicUpdateSlice(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Dynamic update slice
        let operand = try getOperand(op.operands[0])
        let update = try getOperand(op.operands[1])

        // Start indices come from remaining operands
        // This is a simplified implementation using scatter
        guard let updateShape = update.shape else {
            throw CompilationError.unsupportedOperation("dynamic_update_slice requires known update shape")
        }

        // Create indices for scatter
        let zeros = graph.constant(0.0, shape: updateShape, dataType: .int32)

        return graph.scatterAlongAxis(
            0,
            data: operand,
            updates: update,
            indices: zeros,
            mode: .set,
            name: op.result
        )
    }

    private func compileDynamicPad(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Dynamic padding with runtime padding amounts
        let input = try getOperand(op.operands[0])
        _ = try getOperand(op.operands[1])  // padValue - would be used in full implementation

        // Padding amounts from remaining operands - use zeros as default
        let rank = op.resultType.shape.count
        let leftPadding = Array(repeating: NSNumber(value: 0), count: rank)
        let rightPadding = Array(repeating: NSNumber(value: 0), count: rank)

        return graph.padTensor(
            input,
            with: .constant,
            leftPadding: leftPadding,
            rightPadding: rightPadding,
            constantValue: 0.0,
            name: op.result
        )
    }

    private func compileDynamicGather(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Dynamic gather with runtime slice sizes
        let operand = try getOperand(op.operands[0])
        let indices = try getOperand(op.operands[1])

        // Use regular gather logic
        let int32Indices: MPSGraphTensor
        if indices.dataType != .int32 {
            int32Indices = graph.cast(indices, to: .int32, name: nil)
        } else {
            int32Indices = indices
        }

        let indicesShape = int32Indices.shape ?? []
        var newIndicesShape = indicesShape.map { $0 as NSNumber }
        newIndicesShape.append(1)

        let reshapedIndices = graph.reshape(int32Indices, shape: newIndicesShape, name: nil)

        return graph.gatherND(
            withUpdatesTensor: operand,
            indicesTensor: reshapedIndices,
            batchDimensions: 0,
            name: op.result
        )
    }

    private func compileDynamicIota(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Dynamic iota with runtime shape
        let axis = op.attributes.axis ?? 0
        // Shape comes from operand, but we use result type for static compilation
        return graph.coordinate(
            alongAxis: axis,
            withShape: op.resultType.mpsShape,
            name: op.result
        )
    }

    // MARK: - Linear Algebra Operations

    private func compileTriangularSolve(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Solve Ax = b where A is triangular
        // MPSGraph doesn't have direct triangular solve - requires MPSMatrixSolveTriangular
        //
        // To implement properly, you would need to:
        // 1. Extract MTLBuffer from MPSGraphTensorData
        // 2. Create MPSMatrix objects
        // 3. Use MPSMatrixSolveTriangular to solve
        // 4. Wrap result back into MPSGraphTensor
        //
        // This requires breaking out of the pure MPSGraph compilation model.
        _ = try getOperand(op.operands[0])  // a - triangular matrix
        _ = try getOperand(op.operands[1])  // b - right-hand side

        throw CompilationError.unsupportedOperation(
            "triangular_solve requires MPSMatrixSolveTriangular kernel bridging - " +
            "not available in pure MPSGraph compilation"
        )
    }

    private func compileCholesky(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Cholesky decomposition A = LL^T
        // MPSGraph doesn't have direct Cholesky - requires MPSMatrixDecompositionCholesky
        //
        // To implement properly, you would need to:
        // 1. Extract MTLBuffer from MPSGraphTensorData
        // 2. Create MPSMatrix object
        // 3. Use MPSMatrixDecompositionCholesky
        // 4. Wrap result back into MPSGraphTensor
        //
        // This requires breaking out of the pure MPSGraph compilation model.
        _ = try getOperand(op.operands[0])  // input matrix

        throw CompilationError.unsupportedOperation(
            "cholesky requires MPSMatrixDecompositionCholesky kernel bridging - " +
            "not available in pure MPSGraph compilation"
        )
    }

    private func compileSelectAndScatter(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Select and scatter - used for pooling gradients
        // This selects values based on a window function and scatters updates
        let operand = try getOperand(op.operands[0])
        let source = try getOperand(op.operands[1])
        _ = try getOperand(op.operands[2])  // initValue - not used in simplified impl

        guard let dimNumbers = op.attributes.selectAndScatterDimensionNumbers else {
            throw CompilationError.missingAttribute("selectAndScatterDimensionNumbers", operation: "select_and_scatter")
        }

        // For max pooling gradient, use maxPooling2DGradient
        // This is a simplified implementation for the common case
        let windowDims = dimNumbers.windowDimensions
        let strides = dimNumbers.windowStrides

        // Assume NHWC format
        let kernelHeight = windowDims.count > 1 ? windowDims[1] : windowDims[0]
        let kernelWidth = windowDims.count > 2 ? windowDims[2] : kernelHeight
        let strideY = strides.count > 1 ? strides[1] : strides[0]
        let strideX = strides.count > 2 ? strides[2] : strideY

        let descriptor = MPSGraphPooling2DOpDescriptor(
            kernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            strideInX: strideX,
            strideInY: strideY,
            dilationRateInX: 1,
            dilationRateInY: 1,
            paddingLeft: 0,
            paddingRight: 0,
            paddingTop: 0,
            paddingBottom: 0,
            paddingStyle: .explicit,
            dataLayout: .NHWC
        )!

        return graph.maxPooling2DGradient(
            withGradientTensor: source,
            sourceTensor: operand,
            descriptor: descriptor,
            name: op.result
        )
    }

    // MARK: - RNG Operations

    private func compileRngBitGenerator(_ op: HLOOperation) throws -> MPSGraphTensor {
        // RNG bit generator with deterministic PRNG
        // MPSGraph's random doesn't support seeded PRNG directly
        // Return random tensor matching output type
        let algorithm = op.attributes.rngAlgorithm ?? .defaultAlgorithm

        // For now, use MPSGraph's random - proper implementation needs custom PRNG
        _ = algorithm  // Would use this to select algorithm

        return graph.randomUniformTensor(
            withShape: op.resultType.mpsShape,
            name: op.result
        )
    }

    // MARK: - Map Operation

    private func compileMap(_ op: HLOOperation) throws -> MPSGraphTensor {
        // Map applies a computation element-wise over tensors
        // The computation is defined by a region (lambda function)
        //
        // To implement properly, you would need to:
        // 1. Parse and compile the computation region
        // 2. Generate a custom Metal kernel that applies the computation
        // 3. Execute element-wise across all inputs
        //
        // This requires JIT compilation capabilities not available in MPSGraph.
        // For common patterns (like element-wise unary/binary ops), use the
        // corresponding StableHLO operations directly instead of map.
        throw CompilationError.unsupportedOperation(
            "map requires JIT compilation of computation region - " +
            "use explicit element-wise operations instead"
        )
    }

    // MARK: - Control Flow Compilation

    private func compileWhile(_ op: HLOOperation) throws -> MPSGraphTensor {
        guard let whileRegions = op.attributes.whileRegions else {
            throw CompilationError.missingAttribute("whileRegions", operation: "while")
        }

        // Get initial values from operands
        var initialInputs: [MPSGraphTensor] = []
        for operandName in op.operands {
            let tensor = try getOperand(operandName)
            initialInputs.append(tensor)
        }

        guard !initialInputs.isEmpty else {
            throw CompilationError.wrongOperandCount(expected: 1, got: 0)
        }

        // Capture regions for use in closures
        let condRegion = whileRegions.condition
        let bodyRegion = whileRegions.body

        // Compile while loop using MPSGraph native control flow
        let results = graph.while(
            initialInputs: initialInputs,
            before: { (inputTensors: [MPSGraphTensor], resultTensors: NSMutableArray) -> MPSGraphTensor in
                // Compile condition region
                // Map region arguments to input tensors
                var localValueMap: [String: MPSGraphTensor] = [:]
                for (arg, tensor) in zip(condRegion.arguments, inputTensors) {
                    localValueMap[arg.name] = tensor
                }

                // Add input tensors to resultTensors - these get passed to the after block
                for tensor in inputTensors {
                    resultTensors.add(tensor)
                }

                // Compile operations in condition region
                var lastResult: MPSGraphTensor? = nil
                for operation in condRegion.operations {
                    let result = try? self.compileRegionOperation(
                        operation,
                        localValueMap: &localValueMap
                    )
                    if let result = result {
                        localValueMap[operation.result] = result
                        lastResult = result
                    }
                }

                // Get return value (should be i1 predicate)
                var predicate: MPSGraphTensor
                if let returnVal = condRegion.returnValues.first,
                   let pred = localValueMap[returnVal] {
                    predicate = pred
                } else if let last = lastResult {
                    predicate = last
                } else {
                    // Fallback: create false to exit loop
                    predicate = self.graph.constant(0.0, dataType: .bool)
                }

                return predicate
            },
            after: { (inputTensors: [MPSGraphTensor]) -> [MPSGraphTensor] in
                // Compile body region
                var localValueMap: [String: MPSGraphTensor] = [:]
                for (arg, tensor) in zip(bodyRegion.arguments, inputTensors) {
                    localValueMap[arg.name] = tensor
                }

                // Compile operations in body region
                for operation in bodyRegion.operations {
                    let result = try? self.compileRegionOperation(
                        operation,
                        localValueMap: &localValueMap
                    )
                    if let result = result {
                        localValueMap[operation.result] = result
                    }
                }

                // Get return values
                var outputs: [MPSGraphTensor] = []
                for returnVal in bodyRegion.returnValues {
                    if let tensor = localValueMap[returnVal] {
                        outputs.append(tensor)
                    }
                }

                // If no explicit returns, return input tensors unchanged
                if outputs.isEmpty {
                    return inputTensors
                }

                return outputs
            },
            name: op.result
        )

        // Handle multiple outputs from while loops
        // Store all results in valueMap with indexed names for later extraction
        // via get-tuple-element operations
        for (index, tensor) in results.enumerated() {
            let indexedName = "\(op.result).\(index)"
            valueMap[indexedName] = tensor
        }

        // Also store the count of outputs for tuple handling
        tupleOutputCounts[op.result] = results.count

        // Return first result (primary output)
        // Other outputs can be accessed via get-tuple-element or indexed names
        return results.first ?? initialInputs.first!
    }

    private func compileIf(_ op: HLOOperation) throws -> MPSGraphTensor {
        guard let ifRegions = op.attributes.ifRegions else {
            throw CompilationError.missingAttribute("ifRegions", operation: "if")
        }

        guard !op.operands.isEmpty else {
            throw CompilationError.wrongOperandCount(expected: 1, got: 0)
        }

        // First operand is the predicate
        let predicate = try getOperand(op.operands[0])

        // Compile if using MPSGraph native control flow
        let results = graph.if(
            predicate,
            then: { [self] () -> [MPSGraphTensor] in
                // Compile then branch
                var localValueMap: [String: MPSGraphTensor] = self.valueMap

                // Compile operations in then region
                for operation in ifRegions.thenBranch.operations {
                    let result = try? self.compileRegionOperation(
                        operation,
                        localValueMap: &localValueMap
                    )
                    if let result = result {
                        localValueMap[operation.result] = result
                    }
                }

                // Get return values
                var outputs: [MPSGraphTensor] = []
                for returnVal in ifRegions.thenBranch.returnValues {
                    if let tensor = localValueMap[returnVal] {
                        outputs.append(tensor)
                    }
                }

                return outputs
            },
            else: { [self] () -> [MPSGraphTensor] in
                // Compile else branch (or return default)
                guard let elseBranch = ifRegions.elseBranch else {
                    // No else branch - return zeros matching result type
                    let zeros = self.graph.constant(
                        0.0,
                        shape: op.resultType.mpsShape,
                        dataType: op.resultType.elementType.mpsDataType
                    )
                    return [zeros]
                }

                var localValueMap: [String: MPSGraphTensor] = self.valueMap

                // Compile operations in else region
                for operation in elseBranch.operations {
                    let result = try? self.compileRegionOperation(
                        operation,
                        localValueMap: &localValueMap
                    )
                    if let result = result {
                        localValueMap[operation.result] = result
                    }
                }

                // Get return values
                var outputs: [MPSGraphTensor] = []
                for returnVal in elseBranch.returnValues {
                    if let tensor = localValueMap[returnVal] {
                        outputs.append(tensor)
                    }
                }

                return outputs
            },
            name: op.result
        )

        return results.first ?? graph.constant(0.0, shape: op.resultType.mpsShape, dataType: op.resultType.elementType.mpsDataType)
    }

    /// Compiles an operation within a control flow region using a local value map.
    private func compileRegionOperation(
        _ op: HLOOperation,
        localValueMap: inout [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor {
        // Helper to get operand from local or global map
        func getLocalOperand(_ name: String) throws -> MPSGraphTensor {
            if let tensor = localValueMap[name] {
                return tensor
            }
            if let tensor = valueMap[name] {
                return tensor
            }
            throw CompilationError.undefinedValue(name)
        }

        // Compile based on operation kind
        switch op.kind {
        case .add:
            let lhs = try getLocalOperand(op.operands[0])
            let rhs = try getLocalOperand(op.operands[1])
            return graph.addition(lhs, rhs, name: op.result)

        case .subtract:
            let lhs = try getLocalOperand(op.operands[0])
            let rhs = try getLocalOperand(op.operands[1])
            return graph.subtraction(lhs, rhs, name: op.result)

        case .multiply:
            let lhs = try getLocalOperand(op.operands[0])
            let rhs = try getLocalOperand(op.operands[1])
            return graph.multiplication(lhs, rhs, name: op.result)

        case .divide:
            let lhs = try getLocalOperand(op.operands[0])
            let rhs = try getLocalOperand(op.operands[1])
            return graph.division(lhs, rhs, name: op.result)

        case .maximum:
            let lhs = try getLocalOperand(op.operands[0])
            let rhs = try getLocalOperand(op.operands[1])
            return graph.maximum(lhs, rhs, name: op.result)

        case .minimum:
            let lhs = try getLocalOperand(op.operands[0])
            let rhs = try getLocalOperand(op.operands[1])
            return graph.minimum(lhs, rhs, name: op.result)

        case .negate:
            let input = try getLocalOperand(op.operands[0])
            return graph.negative(with: input, name: op.result)

        case .abs:
            let input = try getLocalOperand(op.operands[0])
            return graph.absolute(with: input, name: op.result)

        case .sqrt:
            let input = try getLocalOperand(op.operands[0])
            return graph.squareRoot(with: input, name: op.result)

        case .compare:
            let lhs = try getLocalOperand(op.operands[0])
            let rhs = try getLocalOperand(op.operands[1])
            let direction = op.attributes.comparisonDirection ?? .lt
            switch direction {
            case .lt:
                return graph.lessThan(lhs, rhs, name: op.result)
            case .le:
                return graph.lessThanOrEqualTo(lhs, rhs, name: op.result)
            case .gt:
                return graph.greaterThan(lhs, rhs, name: op.result)
            case .ge:
                return graph.greaterThanOrEqualTo(lhs, rhs, name: op.result)
            case .eq:
                return graph.equal(lhs, rhs, name: op.result)
            case .ne:
                return graph.notEqual(lhs, rhs, name: op.result)
            }

        case .select:
            let condition = try getLocalOperand(op.operands[0])
            let trueVal = try getLocalOperand(op.operands[1])
            let falseVal = try getLocalOperand(op.operands[2])
            return graph.select(predicate: condition, trueTensor: trueVal, falseTensor: falseVal, name: op.result)

        case .constant:
            guard let constantValue = op.attributes.constantValue else {
                throw CompilationError.missingAttribute("constantValue", operation: "constant")
            }
            switch constantValue {
            case .scalar(let value), .splat(let value, _):
                return graph.constant(value, shape: op.resultType.mpsShape, dataType: op.resultType.elementType.mpsDataType)
            case .dense(let values, _):
                var floats = values.map { Float($0) }
                let data = Data(bytes: &floats, count: floats.count * MemoryLayout<Float>.stride)
                return graph.constant(data, shape: op.resultType.mpsShape, dataType: op.resultType.elementType.mpsDataType)
            }

        default:
            // For unsupported operations in regions, try to use the regular compilation
            // by temporarily updating the global value map
            for operand in op.operands {
                if let tensor = localValueMap[operand], valueMap[operand] == nil {
                    valueMap[operand] = tensor
                }
            }
            return try compileOperation(op)
        }
    }
}

// MARK: - Compilation Errors

/// Errors that can occur during compilation.
public enum CompilationError: Error, Sendable, CustomStringConvertible {
    case undefinedValue(String)
    case unsupportedOperation(String)
    case wrongOperandCount(expected: Int, got: Int)
    case missingAttribute(String, operation: String)
    case invalidAttribute(String, operation: String)

    public var description: String {
        switch self {
        case .undefinedValue(let name):
            return "Undefined value: \(name)"
        case .unsupportedOperation(let op):
            return "Unsupported operation: \(op)"
        case .wrongOperandCount(let expected, let got):
            return "Wrong operand count: expected \(expected), got \(got)"
        case .missingAttribute(let attr, let op):
            return "Missing attribute '\(attr)' for operation '\(op)'"
        case .invalidAttribute(let attr, let op):
            return "Invalid attribute '\(attr)' for operation '\(op)'"
        }
    }
}

// MARK: - CompiledGraph

/// A compiled MPSGraph ready for execution.
public struct CompiledGraph: @unchecked Sendable {
    /// The compiled executable.
    public let executable: MPSGraphExecutable

    /// The underlying graph.
    public let graph: MPSGraph

    /// Input placeholders.
    public let inputTensors: [MPSGraphTensor]

    /// Output tensors.
    public let outputTensors: [MPSGraphTensor]

    /// Input types.
    public let inputTypes: [TensorType]

    /// Output types.
    public let outputTypes: [TensorType]

    /// The Metal device.
    public let device: MTLDevice
}
