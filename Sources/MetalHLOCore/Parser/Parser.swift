// Parser.swift
// MetalHLOCore
//
// MLIR parser for StableHLO modules.

/// Parses MLIR text into an HLOModule.
public final class Parser {

    // MARK: - Properties

    private let lexer: Lexer
    private var currentToken: Token
    private var peekedToken: Token?

    // MARK: - Initialization

    /// Creates a new parser for the given source text.
    ///
    /// - Parameter source: The MLIR source text.
    public init(source: String) {
        self.lexer = Lexer(source: source)
        self.currentToken = lexer.nextToken()
    }

    // MARK: - Public API

    /// Parses the source and returns an HLOModule.
    ///
    /// - Throws: `ParseError` if parsing fails.
    /// - Returns: The parsed HLOModule.
    public func parse() throws -> HLOModule {
        try parseModule()
    }

    // MARK: - Module Parsing

    private func parseModule() throws -> HLOModule {
        // Skip any leading newlines
        skipNewlines()

        // Expect: module @name {
        try expect(.keyword(.module))
        let moduleName = try parseAtIdentifier()
        try expect(.leftBrace)
        skipNewlines()

        // Parse function
        let function = try parseFunction()
        skipNewlines()

        // Expect: }
        try expect(.rightBrace)

        return HLOModule(name: moduleName, function: function)
    }

    // MARK: - Function Parsing

    private func parseFunction() throws -> HLOFunction {
        // Expect: func.func @main(args) -> (return_types) {
        try expectIdentifier("func.func")
        let funcName = try parseAtIdentifier()
        let inputs = try parseFunctionArguments()
        try expect(.arrow)
        let outputTypes = try parseReturnTypes()
        try expect(.leftBrace)
        skipNewlines()

        // Parse operations
        var operations: [HLOOperation] = []
        var returnValues: [String] = []

        // Tuple elimination: track tuple compositions and value aliases
        var tupleOperands: [String: [String]] = [:]
        var valueAliases: [String: String] = [:]

        while !check(.rightBrace) && !check(.eof) {
            if checkKeyword(.return) {
                returnValues = try parseReturn()
                // Resolve aliases in return values
                returnValues = returnValues.map { resolveAlias($0, aliases: valueAliases) }
                break
            }
            let op = try parseOperation()

            if op.kind == .tuple {
                // Record tuple composition — resolve any aliases in operands
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                tupleOperands[op.result] = resolvedOperands
            } else if op.kind == .getTupleElement {
                // Resolve get_tuple_element to the actual tensor
                let tupleRef = resolveAlias(op.operands.first ?? "", aliases: valueAliases)
                let index = op.attributes.tupleIndex ?? 0
                if let members = tupleOperands[tupleRef], index < members.count {
                    valueAliases[op.result] = members[index]
                }
            } else {
                // Substitute aliases in operands for regular operations
                let resolvedOperands = op.operands.map { resolveAlias($0, aliases: valueAliases) }
                if resolvedOperands != op.operands {
                    operations.append(HLOOperation(
                        result: op.result,
                        kind: op.kind,
                        operands: resolvedOperands,
                        resultType: op.resultType,
                        attributes: op.attributes
                    ))
                } else {
                    operations.append(op)
                }
            }
            skipNewlines()
        }

        skipNewlines()
        try expect(.rightBrace)

        return HLOFunction(
            name: funcName,
            inputs: inputs,
            outputTypes: outputTypes,
            operations: operations,
            returnValues: returnValues
        )
    }

    private func parseFunctionArguments() throws -> [HLOArgument] {
        try expect(.leftParen)
        var args: [HLOArgument] = []

        if !check(.rightParen) {
            repeat {
                let name = try parsePercentIdentifier()
                try expect(.colon)
                let type = try parseTensorType()
                args.append(HLOArgument(name: name, type: type))
            } while match(.comma)
        }

        try expect(.rightParen)
        return args
    }

    private func parseReturnTypes() throws -> [TensorType] {
        try expect(.leftParen)
        var types: [TensorType] = []

        if !check(.rightParen) {
            repeat {
                let type = try parseTensorType()
                types.append(type)
            } while match(.comma)
        }

        try expect(.rightParen)
        return types
    }

    private func parseReturn() throws -> [String] {
        try expectKeyword(.return)
        var values: [String] = []

        // Check for empty return (void function) - no values and no type annotation
        if check(.rightBrace) || check(.newline) || check(.eof) {
            return values
        }

        // Check for return with values
        if check(.percentIdentifier) {
            repeat {
                let value = try parsePercentIdentifier()
                values.append(value)
            } while match(.comma)
        }

        // Parse type annotation if present
        if match(.colon) {
            // Skip the return type(s)
            _ = try parseTensorType()
            while match(.comma) {
                _ = try parseTensorType()
            }
        }

        return values
    }

    // MARK: - Operation Parsing

    private func parseOperation() throws -> HLOOperation {
        // Format: %result = stablehlo.op operands attributes : type
        let result = try parsePercentIdentifier()
        try expect(.equal)

        // Parse operation name (e.g., stablehlo.add)
        let opName = try parseOperationName()
        let kind = try parseOpKind(from: opName)

        // Parse operands and attributes based on operation
        let (operands, attributes, resultType) = try parseOperationBody(kind: kind)

        return HLOOperation(
            result: result,
            kind: kind,
            operands: operands,
            resultType: resultType,
            attributes: attributes
        )
    }

    /// Tracks whether the last parsed operation name was in generic form (quoted)
    private var isGenericForm = false

    private func parseOperationName() throws -> String {
        var name = ""
        isGenericForm = false

        // Handle quoted operation names (generic form): "stablehlo.rng"
        if case .string(let stringValue) = currentToken.kind {
            name = stringValue
            isGenericForm = true
            advance()
            return name
        }

        // Handle identifiers that may contain dots (e.g., stablehlo.add, func.func)
        if case .identifier = currentToken.kind {
            name = currentToken.text
            advance()

            // Handle dotted names - only continue if name ends with dot and next is identifier
            while name.hasSuffix(".") && currentToken.kind == .identifier {
                name += currentToken.text
                advance()
            }
        } else if case .keyword(.stablehlo) = currentToken.kind {
            name = "stablehlo"
            advance()
            // Continue parsing if there's more to the dotted name
            while name.hasSuffix(".") && currentToken.kind == .identifier {
                name += currentToken.text
                advance()
            }
        } else {
            throw ParseError.unexpectedToken(expected: "operation name", got: currentToken)
        }

        return name
    }

    private func parseOpKind(from name: String) throws -> HLOOpKind {
        // Extract the operation name after "stablehlo."
        let opName: String
        if name.hasPrefix("stablehlo.") {
            opName = String(name.dropFirst("stablehlo.".count))
        } else {
            opName = name
        }

        guard let kind = HLOOpKind(rawValue: opName) else {
            throw ParseError.invalidOperation(
                "Unknown operation '\(name)'",
                location: currentToken.location
            )
        }

        return kind
    }

    private func parseOperationBody(kind: HLOOpKind) throws -> ([String], HLOAttributes, TensorType) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Handle generic form: "stablehlo.op"(%op1, %op2) {attributes} : type
        if isGenericForm {
            (operands, attributes) = try parseGenericFormOperandsAndAttributes(kind: kind)
        } else if kind == .customCall {
            // Special handling for custom_call which has @target(operands) format
            (operands, attributes) = try parseCustomCallOperandsAndAttributes()
        } else if kind == .reduce {
            // Special handling for reduce: (%input init: %init) applies stablehlo.add across dimensions = [0]
            (operands, attributes) = try parseReduceOperandsAndAttributes()
        } else if kind == .compare {
            // Special handling for compare: EQ, %arg0, %arg1, FLOAT : types
            (operands, attributes) = try parseCompareOperandsAndAttributes()
        } else if kind == .select {
            // Special handling for select: %pred, %on_true, %on_false : pred_type, result_type
            (operands, attributes) = try parseSelectOperandsAndAttributes()
        } else if kind == .iota {
            // Special handling for iota: dim = N : tensor<...>
            (operands, attributes) = try parseIotaOperandsAndAttributes()
        } else if kind == .getTupleElement {
            // Special handling for get_tuple_element: %tuple[index]
            (operands, attributes) = try parseGetTupleElementOperandsAndAttributes()
        } else {
            // Parse operands (% identifiers before attributes/type)
            while check(.percentIdentifier) {
                let operand = try parsePercentIdentifier()
                operands.append(operand)
                _ = match(.comma)
            }

            // Parse attributes based on operation kind
            attributes = try parseAttributes(for: kind)
        }

        // Parse type signature
        try expect(.colon)
        let resultType = try parseTypeSignature(for: kind)

        return (operands, attributes, resultType)
    }

    /// Parse custom_call operands and attributes in the format: @target(operands) { backend_config = "..." }
    private func parseCustomCallOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Parse target name: @target_name
        if check(.atIdentifier) {
            attributes.callTargetName = String(currentToken.text.dropFirst())  // Remove @
            advance()
        }

        // Parse operands: (%op1, %op2, ...)
        if match(.leftParen) {
            if !check(.rightParen) {
                repeat {
                    let operand = try parsePercentIdentifier()
                    operands.append(operand)
                } while match(.comma)
            }
            try expect(.rightParen)
        }

        // Parse attribute block: { backend_config = "..." }
        if match(.leftBrace) {
            skipNewlines()
            while !check(.rightBrace) && !check(.eof) {
                if checkIdentifier("backend_config") {
                    try expectIdentifier("backend_config")
                    try expect(.equal)
                    if case .string(let value) = currentToken.kind {
                        attributes.backendConfig = value
                        advance()
                    }
                } else {
                    // Skip unknown attributes
                    advance()
                }
                _ = match(.comma)
                skipNewlines()
            }
            try expect(.rightBrace)
        }

        return (operands, attributes)
    }

    /// Parse generic form operands and attributes
    /// Format: (%op1, %op2, ...) ({region}) {attr = value, ...}
    /// The region is optional and used by operations like scatter and reduce.
    private func parseGenericFormOperandsAndAttributes(kind: HLOOpKind) throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Parse operands: (%op1, %op2, ...)
        if match(.leftParen) {
            if !check(.rightParen) {
                repeat {
                    let operand = try parsePercentIdentifier()
                    operands.append(operand)
                } while match(.comma)
            }
            try expect(.rightParen)
        }

        // Parse region if present: ({^bb0(%arg0: type, %arg1: type): ops... })
        // This appears between operands and attributes for ops like scatter and reduce.
        // The region body determines the computation kind (e.g., identity, add, max).
        if check(.leftParen) {
            let computationKind = try parseOperationRegion()
            if kind == .scatter {
                attributes.scatterComputationKind = computationKind
            }
        }

        // Parse attribute block: {attr = value, ...}
        if match(.leftBrace) {
            skipNewlines()

            // For gather operations, use specialized parsing
            if kind == .gather {
                attributes.gatherDimensionNumbers = try parseGatherDimensionNumbers()
                // Skip to end of brace
                while !check(.rightBrace) && !check(.eof) {
                    advance()
                }
            } else if kind == .scatter {
                let (dimNumbers, attrComputationKind) = try parseScatterAttributes()
                attributes.scatterDimensionNumbers = dimNumbers
                // Only overwrite computation kind if the attribute block had one;
                // otherwise preserve what the region parsing already set.
                if attrComputationKind != nil {
                    attributes.scatterComputationKind = attrComputationKind
                }
                while !check(.rightBrace) && !check(.eof) {
                    advance()
                }
            } else {
                // Generic attribute parsing
                while !check(.rightBrace) && !check(.eof) {
                    // Parse attribute name
                    if checkIdentifier("index") && kind == .getTupleElement {
                        // Parse index attribute: index = 0 : i32
                        try expectIdentifier("index")
                        try expect(.equal)
                        if case .integer(let indexValue) = currentToken.kind {
                            attributes.tupleIndex = Int(indexValue)
                            advance()
                        }
                        // Skip optional type annotation ": i32"
                        if match(.colon) {
                            while !check(.comma) && !check(.rightBrace) && !check(.eof) {
                                advance()
                            }
                        }
                    } else if checkIdentifier("rng_distribution") {
                        try expectIdentifier("rng_distribution")
                        try expect(.equal)
                        // Parse #stablehlo<rng_distribution UNIFORM> or #stablehlo<rng_distribution NORMAL>
                        if check(.hashIdentifier) {
                            // Skip #stablehlo
                            advance()
                            // Parse <rng_distribution UNIFORM>
                            if match(.leftAngle) {
                                // Skip "rng_distribution" identifier
                                if check(.identifier) {
                                    advance()
                                }
                                // Parse UNIFORM or NORMAL
                                if checkIdentifier("UNIFORM") {
                                    attributes.rngDistribution = .uniform
                                    advance()
                                } else if checkIdentifier("NORMAL") {
                                    attributes.rngDistribution = .normal
                                    advance()
                                }
                                try expect(.rightAngle)
                            }
                        }
                    } else {
                        // Skip unknown attributes
                        advance()
                    }
                    _ = match(.comma)
                    skipNewlines()
                }
            }
            try expect(.rightBrace)
        }

        return (operands, attributes)
    }

    /// Parse reduce operands and attributes
    /// Supports two formats:
    ///   1. (%input init: %init) applies stablehlo.add across dimensions = [0]
    ///   2. %input, %init applies stablehlo.add across dimensions = [0]
    private func parseReduceOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Check which format we're parsing
        if check(.leftParen) {
            // Format 1: (%input init: %init)
            try expect(.leftParen)

            // Parse input operand
            let inputOperand = try parsePercentIdentifier()
            operands.append(inputOperand)

            // Parse "init:" and init operand
            try expectIdentifier("init")
            try expect(.colon)
            let initOperand = try parsePercentIdentifier()
            operands.append(initOperand)

            try expect(.rightParen)
        } else {
            // Format 2: %input, %init
            // Parse input operand
            let inputOperand = try parsePercentIdentifier()
            operands.append(inputOperand)

            _ = match(.comma)

            // Parse init operand
            let initOperand = try parsePercentIdentifier()
            operands.append(initOperand)
        }

        // Parse: applies stablehlo.add across dimensions = [0]
        if checkIdentifier("applies") {
            try expectIdentifier("applies")
            let reductionOp = try parseOperationName()

            if reductionOp.contains("add") {
                attributes.reductionKind = .sum
            } else if reductionOp.contains("max") {
                attributes.reductionKind = .max
            } else if reductionOp.contains("min") {
                attributes.reductionKind = .min
            }
            if reductionOp.contains("multiply") {
                attributes.reductionKind = .product
            } else if reductionOp.contains("and") {
                attributes.reductionKind = .and
            } else if reductionOp.contains("or") {
                attributes.reductionKind = .or
            }

            if checkIdentifier("across") {
                try expectIdentifier("across")
                try expectIdentifier("dimensions")
                try expect(.equal)
                attributes.dimensions = try parseDimensionList()
            }
        }

        return (operands, attributes)
    }

    /// Parse compare operands and attributes
    /// Supports two formats:
    ///   1. Direction first: EQ, %arg0, %arg1, FLOAT : types (official StableHLO test format)
    ///   2. Operands first: %arg0, %arg1, EQ, FLOAT : types (common MLIR format)
    private func parseCompareOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        let directions = ["EQ", "NE", "LT", "LE", "GT", "GE"]

        // Check if direction comes first (official StableHLO format)
        var directionFirst = false
        for dir in directions {
            if checkIdentifier(dir) {
                directionFirst = true
                attributes.comparisonDirection = ComparisonDirection(rawValue: dir)
                advance()
                _ = match(.comma)
                break
            }
        }

        // Parse operands
        while check(.percentIdentifier) {
            let operand = try parsePercentIdentifier()
            operands.append(operand)
            _ = match(.comma)
        }

        // If direction wasn't first, parse it after operands
        if !directionFirst {
            var foundDirection = false
            for dir in directions {
                if checkIdentifier(dir) {
                    attributes.comparisonDirection = ComparisonDirection(rawValue: dir)
                    advance()
                    _ = match(.comma)
                    foundDirection = true
                    break
                }
            }

            if !foundDirection {
                throw ParseError.unexpectedToken(expected: "comparison direction (EQ, NE, LT, LE, GT, GE)", got: currentToken)
            }
        }

        // Skip optional comparison type (FLOAT, SIGNED, UNSIGNED, TOTALORDER)
        let compTypes = ["FLOAT", "SIGNED", "UNSIGNED", "TOTALORDER"]
        for compType in compTypes {
            if checkIdentifier(compType) {
                advance()
                break
            }
        }

        return (operands, attributes)
    }

    /// Parse select operands in the format: %pred, %on_true, %on_false
    private func parseSelectOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        let attributes = HLOAttributes()

        // Parse the three operands: predicate, true value, false value
        while check(.percentIdentifier) {
            let operand = try parsePercentIdentifier()
            operands.append(operand)
            _ = match(.comma)
        }

        return (operands, attributes)
    }

    /// Parse iota operands and attributes in the format: dim = N
    /// iota has no operands, just a dimension attribute
    private func parseIotaOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        let operands: [String] = []
        var attributes = HLOAttributes()

        // Parse: dim = N
        if checkIdentifier("dim") {
            try expectIdentifier("dim")
            try expect(.equal)
            if case .integer(let dimValue) = currentToken.kind {
                attributes.iotaDimension = Int(dimValue)
                advance()
            } else {
                throw ParseError.unexpectedToken(expected: "integer dimension", got: currentToken)
            }
        }

        return (operands, attributes)
    }

    /// Parse get_tuple_element operands and attributes.
    /// Pretty-printed form: %tuple[index]
    private func parseGetTupleElementOperandsAndAttributes() throws -> ([String], HLOAttributes) {
        var operands: [String] = []
        var attributes = HLOAttributes()

        // Parse the tuple operand
        let operand = try parsePercentIdentifier()
        operands.append(operand)

        // Parse [index] if present (pretty-printed form)
        if match(.leftBracket) {
            if case .integer(let indexValue) = currentToken.kind {
                attributes.tupleIndex = Int(indexValue)
                advance()
            }
            try expect(.rightBracket)
        }

        return (operands, attributes)
    }

    private func parseAttributes(for kind: HLOOpKind) throws -> HLOAttributes {
        var attributes = HLOAttributes()

        // Handle operation-specific attribute parsing
        switch kind {
        case .constant:
            attributes.constantValue = try parseConstantValue()

        case .transpose:
            if match(.comma) || checkIdentifier("dims") {
                if checkIdentifier("dims") {
                    try expectIdentifier("dims")
                    try expect(.equal)
                }
                attributes.dimensions = try parseDimensionList()
            }

        case .broadcastInDim:
            if match(.comma) || checkIdentifier("dims") {
                if checkIdentifier("dims") {
                    try expectIdentifier("dims")
                    try expect(.equal)
                }
                attributes.dimensions = try parseDimensionList()
            }

        case .reduce:
            attributes = try parseReduceAttributes()

        case .dotGeneral:
            attributes.dotDimensionNumbers = try parseDotDimensionNumbers()

        case .gather:
            attributes.gatherDimensionNumbers = try parseGatherDimensionNumbers()

        case .scatter:
            (attributes.scatterDimensionNumbers, attributes.scatterComputationKind) = try parseScatterAttributes()

        case .compare:
            attributes.comparisonDirection = try parseComparisonDirection()

        case .slice:
            attributes = try parseSliceAttributes()

        case .pad:
            attributes = try parsePadAttributes()

        case .concatenate:
            if match(.comma) || checkIdentifier("dim") || checkIdentifier("dimension") {
                if checkIdentifier("dim") {
                    try expectIdentifier("dim")
                } else if checkIdentifier("dimension") {
                    try expectIdentifier("dimension")
                }
                try expect(.equal)
                attributes.axis = try parseInteger()
            }

        case .rng:
            // RNG has format: shape = [...], distribution = UNIFORM/NORMAL
            // Skip shape attribute (output shape comes from result type)
            while !check(.colon) && !check(.eof) {
                if checkIdentifier("shape") {
                    try expectIdentifier("shape")
                    try expect(.equal)
                    // Skip the shape array [...]
                    if match(.leftBracket) {
                        while !check(.rightBracket) && !check(.eof) {
                            advance()
                        }
                        try expect(.rightBracket)
                    }
                    _ = match(.comma)
                } else if checkIdentifier("distribution") {
                    try expectIdentifier("distribution")
                    try expect(.equal)
                    attributes.rngDistribution = try parseRNGDistribution()
                    _ = match(.comma)
                } else if checkIdentifier("UNIFORM") || checkIdentifier("NORMAL") {
                    // Direct distribution without "distribution =" prefix
                    attributes.rngDistribution = try parseRNGDistribution()
                    _ = match(.comma)
                } else {
                    break
                }
            }

        case .whileOp:
            attributes.whileRegions = try parseWhileRegions()

        case .ifOp:
            attributes.ifRegions = try parseIfRegions()

        case .reverse:
            if match(.comma) || checkIdentifier("dims") || checkIdentifier("dimensions") {
                if checkIdentifier("dims") {
                    try expectIdentifier("dims")
                } else if checkIdentifier("dimensions") {
                    try expectIdentifier("dimensions")
                }
                try expect(.equal)
                attributes.dimensions = try parseDimensionList()
            }

        case .convolution:
            attributes = try parseConvolutionAttributes()

        case .reduceWindow:
            attributes = try parseReduceWindowAttributes()

        case .batchNormInference, .batchNormTraining, .batchNormGrad:
            attributes = try parseBatchNormAttributes()

        case .fft:
            attributes = try parseFFTAttributes()

        case .sort:
            attributes = try parseSortAttributes()

        case .tan, .logistic, .isFinite, .expm1, .log1p, .cbrt,
             .roundNearestAfz, .roundNearestEven, .popcnt, .real, .imag:
            // No special attributes needed - simple unary ops
            break

        case .shiftLeft, .shiftRightArithmetic, .shiftRightLogical, .complex:
            // Binary ops with no special attributes
            break

        case .uniformQuantize, .uniformDequantize, .bitcastConvert:
            // Type conversion operations
            break

        case .dynamicSlice:
            attributes = try parseDynamicSliceAttributes()

        case .dynamicUpdateSlice, .dynamicPad, .dynamicGather:
            // Dynamic indexing operations - operands contain dynamic info
            break

        case .dynamicReshape, .dynamicBroadcastInDim, .dynamicIota:
            // Dynamic shape operations - output shape from operand
            break

        case .triangularSolve:
            attributes = try parseTriangularSolveAttributes()

        case .cholesky:
            attributes = try parseCholeskyAttributes()

        case .reducePrecision:
            attributes = try parseReducePrecisionAttributes()

        case .rngBitGenerator:
            attributes = try parseRngBitGeneratorAttributes()

        case .selectAndScatter:
            attributes = try parseSelectAndScatterAttributes()

        case .map:
            // Map has a computation region - simplified parsing
            break

        case .customCall:
            // custom_call is handled specially in parseOperationBody via parseCustomCallOperandsAndAttributes()
            break

        default:
            // No special attributes needed
            break
        }

        return attributes
    }

    // MARK: - Convolution Attribute Parsing

    private func parseConvolutionAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("window_strides") {
                try expectIdentifier("window_strides")
                try expect(.equal)
                attributes.windowStrides = try parseArrayAttribute()
            } else if checkIdentifier("padding") {
                try expectIdentifier("padding")
                try expect(.equal)
                attributes.convPadding = try parsePaddingArray()
            } else if checkIdentifier("lhs_dilation") {
                try expectIdentifier("lhs_dilation")
                try expect(.equal)
                attributes.lhsDilation = try parseArrayAttribute()
            } else if checkIdentifier("rhs_dilation") {
                try expectIdentifier("rhs_dilation")
                try expect(.equal)
                attributes.rhsDilation = try parseArrayAttribute()
            } else if checkIdentifier("feature_group_count") {
                try expectIdentifier("feature_group_count")
                try expect(.equal)
                attributes.featureGroupCount = try parseInteger()
            } else if checkIdentifier("batch_group_count") {
                try expectIdentifier("batch_group_count")
                try expect(.equal)
                attributes.batchGroupCount = try parseInteger()
            } else if checkIdentifier("dimension_numbers") || check(.hashIdentifier) {
                attributes.convolutionDimensionNumbers = try parseConvolutionDimensionNumbers()
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    private func parseConvolutionDimensionNumbers() throws -> ConvolutionDimensionNumbers? {
        // Skip the #stablehlo.conv< prefix if present
        if check(.hashIdentifier) {
            advance()
        } else if checkIdentifier("dimension_numbers") {
            try expectIdentifier("dimension_numbers")
            try expect(.equal)
            if check(.hashIdentifier) {
                advance()
            }
        }

        // Skip angle bracket if present
        if check(.leftAngle) {
            advance()
        }

        // Collect the format string: [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
        var formatString = ""
        while !check(.rightAngle) && !check(.colon) && !check(.eof) {
            formatString += currentToken.text
            advance()
        }

        if check(.rightAngle) {
            advance()
        }

        // Parse the dimension format
        // Format: input_layout x kernel_layout -> output_layout
        // Example: [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
        return parseDimensionFormat(formatString)
    }

    /// Parse dimension format string like [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
    private func parseDimensionFormat(_ format: String) -> ConvolutionDimensionNumbers {
        // Default NHWC layout
        var inputBatch = 0
        var inputFeature = 3
        var inputSpatial = [1, 2]
        var kernelInput = 2
        var kernelOutput = 3
        var kernelSpatial = [0, 1]
        var outputBatch = 0
        var outputFeature = 3
        var outputSpatial = [1, 2]

        // Clean up the format string
        let cleaned = format.replacingOccurrences(of: " ", with: "")
                           .replacingOccurrences(of: ",", with: "")

        // Split into parts: input x kernel -> output
        // Try to find the 'x' separator between input and kernel
        let xParts = cleaned.components(separatedBy: "x")
        guard xParts.count >= 2 else {
            // Malformed, return defaults
            return ConvolutionDimensionNumbers(
                inputBatchDimension: inputBatch,
                inputFeatureDimension: inputFeature,
                inputSpatialDimensions: inputSpatial,
                kernelInputFeatureDimension: kernelInput,
                kernelOutputFeatureDimension: kernelOutput,
                kernelSpatialDimensions: kernelSpatial,
                outputBatchDimension: outputBatch,
                outputFeatureDimension: outputFeature,
                outputSpatialDimensions: outputSpatial
            )
        }

        let inputPart = xParts[0]
        let restPart = xParts[1]

        // Split kernel and output by ->
        let arrowParts = restPart.components(separatedBy: "->")
        let kernelPart = arrowParts[0]
        let outputPart = arrowParts.count > 1 ? arrowParts[1] : inputPart

        // Parse each layout part
        (inputBatch, inputFeature, inputSpatial) = parseLayoutPart(inputPart, isKernel: false)
        (kernelInput, kernelOutput, kernelSpatial) = parseKernelLayoutPart(kernelPart)
        (outputBatch, outputFeature, outputSpatial) = parseLayoutPart(outputPart, isKernel: false)

        return ConvolutionDimensionNumbers(
            inputBatchDimension: inputBatch,
            inputFeatureDimension: inputFeature,
            inputSpatialDimensions: inputSpatial,
            kernelInputFeatureDimension: kernelInput,
            kernelOutputFeatureDimension: kernelOutput,
            kernelSpatialDimensions: kernelSpatial,
            outputBatchDimension: outputBatch,
            outputFeatureDimension: outputFeature,
            outputSpatialDimensions: outputSpatial
        )
    }

    /// Parse a layout part like [b, 0, 1, f] for input/output tensors
    /// Returns (batchDim, featureDim, spatialDims)
    private func parseLayoutPart(_ part: String, isKernel: Bool) -> (Int, Int, [Int]) {
        // Remove brackets
        let cleaned = part.replacingOccurrences(of: "[", with: "")
                         .replacingOccurrences(of: "]", with: "")

        var batchDim = 0
        var featureDim = 3
        var spatialDims: [Int] = []

        // Parse each character/token
        var position = 0
        for char in cleaned {
            switch char {
            case "b", "B":
                batchDim = position
            case "f", "F":
                featureDim = position
            case "0"..."9":
                // Spatial dimension numbered 0, 1, 2, etc.
                spatialDims.append(position)
            default:
                continue
            }
            position += 1
        }

        // Ensure spatialDims is sorted
        spatialDims.sort()

        return (batchDim, featureDim, spatialDims)
    }

    /// Parse a kernel layout part like [0, 1, i, o]
    /// Returns (inputFeatureDim, outputFeatureDim, spatialDims)
    private func parseKernelLayoutPart(_ part: String) -> (Int, Int, [Int]) {
        // Remove brackets
        let cleaned = part.replacingOccurrences(of: "[", with: "")
                         .replacingOccurrences(of: "]", with: "")

        var inputDim = 2
        var outputDim = 3
        var spatialDims: [Int] = []

        // Parse each character/token
        var position = 0
        for char in cleaned {
            switch char {
            case "i", "I":
                inputDim = position
            case "o", "O":
                outputDim = position
            case "0"..."9":
                // Spatial dimension
                spatialDims.append(position)
            default:
                continue
            }
            position += 1
        }

        // Ensure spatialDims is sorted by their numeric label, not position
        spatialDims.sort()

        return (inputDim, outputDim, spatialDims)
    }

    private func parseArrayAttribute() throws -> [Int] {
        if checkIdentifier("array") {
            try expectIdentifier("array")
            if check(.leftAngle) {
                advance()
                // Skip type like i64:
                while !check(.colon) && !check(.eof) {
                    advance()
                }
                if check(.colon) {
                    advance()
                }
            }
        }
        return try parseDimensionList()
    }

    private func parsePaddingArray() throws -> [[Int]] {
        var result: [[Int]] = []

        if checkIdentifier("dense") {
            // dense<0> or dense<[[0,0],[0,0]]> format
            try expectIdentifier("dense")
            try expect(.leftAngle)

            if case .integer(_) = currentToken.kind {
                // Simple dense<0> - all zeros
                advance()
                try expect(.rightAngle)
                // Return empty, will use default
                return result
            }

            if check(.leftBracket) {
                // Nested array format
                try expect(.leftBracket)
                repeat {
                    try expect(.leftBracket)
                    var pair: [Int] = []
                    repeat {
                        pair.append(try parseInteger())
                    } while match(.comma)
                    try expect(.rightBracket)
                    result.append(pair)
                } while match(.comma)
                try expect(.rightBracket)
            }

            try expect(.rightAngle)

            // Skip tensor type if present
            if check(.colon) {
                advance()
                while !check(.comma) && !check(.eof) && !check(.rightAngle) {
                    advance()
                }
            }
        }

        return result
    }

    // MARK: - Reduce Window Attribute Parsing

    private func parseReduceWindowAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("window_dimensions") {
                try expectIdentifier("window_dimensions")
                try expect(.equal)
                attributes.windowDimensions = try parseArrayAttribute()
            } else if checkIdentifier("window_strides") {
                try expectIdentifier("window_strides")
                try expect(.equal)
                attributes.windowStrides = try parseArrayAttribute()
            } else if checkIdentifier("padding") {
                try expectIdentifier("padding")
                try expect(.equal)
                attributes.convPadding = try parsePaddingArray()
            } else if checkIdentifier("base_dilations") {
                try expectIdentifier("base_dilations")
                try expect(.equal)
                attributes.baseDilations = try parseArrayAttribute()
            } else if checkIdentifier("window_dilations") {
                try expectIdentifier("window_dilations")
                try expect(.equal)
                attributes.windowDilations = try parseArrayAttribute()
            } else if checkIdentifier("applies") {
                // Same as regular reduce
                try expectIdentifier("applies")
                let reductionOp = try parseOperationName()
                if reductionOp.contains("add") {
                    attributes.reductionKind = .sum
                } else if reductionOp.contains("max") {
                    attributes.reductionKind = .max
                } else if reductionOp.contains("min") {
                    attributes.reductionKind = .min
                } else if reductionOp.contains("multiply") {
                    attributes.reductionKind = .product
                } else if reductionOp.contains("and") {
                    attributes.reductionKind = .and
                } else if reductionOp.contains("or") {
                    attributes.reductionKind = .or
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Batch Norm Attribute Parsing

    private func parseBatchNormAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("epsilon") {
                try expectIdentifier("epsilon")
                try expect(.equal)
                if case .float(let val) = currentToken.kind {
                    attributes.epsilon = Float(val)
                    advance()
                } else if case .integer(let val) = currentToken.kind {
                    attributes.epsilon = Float(val)
                    advance()
                }
            } else if checkIdentifier("feature_index") {
                try expectIdentifier("feature_index")
                try expect(.equal)
                attributes.featureIndex = try parseInteger()
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - FFT Attribute Parsing

    private func parseFFTAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("fft_type") {
                try expectIdentifier("fft_type")
                try expect(.equal)
                if check(.hashIdentifier) {
                    advance()  // Skip #stablehlo<fft_type ...>
                }
                if checkIdentifier("FFT") {
                    attributes.fftType = .fft
                    advance()
                } else if checkIdentifier("IFFT") {
                    attributes.fftType = .ifft
                    advance()
                } else if checkIdentifier("RFFT") {
                    attributes.fftType = .rfft
                    advance()
                } else if checkIdentifier("IRFFT") {
                    attributes.fftType = .irfft
                    advance()
                }
                if check(.rightAngle) {
                    advance()
                }
            } else if checkIdentifier("fft_length") {
                try expectIdentifier("fft_length")
                try expect(.equal)
                attributes.fftLength = try parseArrayAttribute()
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Sort Attribute Parsing

    private func parseSortAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("dimension") {
                try expectIdentifier("dimension")
                try expect(.equal)
                attributes.axis = try parseInteger()
            } else if checkIdentifier("is_stable") {
                try expectIdentifier("is_stable")
                try expect(.equal)
                if checkIdentifier("true") {
                    attributes.isStable = true
                    advance()
                } else if checkIdentifier("false") {
                    attributes.isStable = false
                    advance()
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Dynamic Slice Attribute Parsing

    private func parseDynamicSliceAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("slice_sizes") {
                try expectIdentifier("slice_sizes")
                try expect(.equal)
                attributes.dynamicSliceSizes = try parseArrayAttribute()
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Triangular Solve Attribute Parsing

    private func parseTriangularSolveAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("left_side") {
                try expectIdentifier("left_side")
                try expect(.equal)
                if checkIdentifier("true") {
                    attributes.leftSide = true
                    advance()
                } else if checkIdentifier("false") {
                    attributes.leftSide = false
                    advance()
                }
            } else if checkIdentifier("lower") {
                try expectIdentifier("lower")
                try expect(.equal)
                if checkIdentifier("true") {
                    attributes.lower = true
                    advance()
                } else if checkIdentifier("false") {
                    attributes.lower = false
                    advance()
                }
            } else if checkIdentifier("unit_diagonal") {
                try expectIdentifier("unit_diagonal")
                try expect(.equal)
                if checkIdentifier("true") {
                    attributes.unitDiagonal = true
                    advance()
                } else if checkIdentifier("false") {
                    attributes.unitDiagonal = false
                    advance()
                }
            } else if checkIdentifier("transpose_a") {
                try expectIdentifier("transpose_a")
                try expect(.equal)
                if checkIdentifier("NO_TRANSPOSE") {
                    attributes.transposeA = .noTranspose
                    advance()
                } else if checkIdentifier("TRANSPOSE") {
                    attributes.transposeA = .transpose
                    advance()
                } else if checkIdentifier("ADJOINT") {
                    attributes.transposeA = .adjoint
                    advance()
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Cholesky Attribute Parsing

    private func parseCholeskyAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("lower") {
                try expectIdentifier("lower")
                try expect(.equal)
                if checkIdentifier("true") {
                    attributes.lower = true
                    advance()
                } else if checkIdentifier("false") {
                    attributes.lower = false
                    advance()
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Reduce Precision Attribute Parsing

    private func parseReducePrecisionAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("exponent_bits") {
                try expectIdentifier("exponent_bits")
                try expect(.equal)
                attributes.exponentBits = try parseInteger()
            } else if checkIdentifier("mantissa_bits") {
                try expectIdentifier("mantissa_bits")
                try expect(.equal)
                attributes.mantissaBits = try parseInteger()
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - RNG Bit Generator Attribute Parsing

    private func parseRngBitGeneratorAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("rng_algorithm") {
                try expectIdentifier("rng_algorithm")
                try expect(.equal)
                if check(.hashIdentifier) {
                    advance()  // Skip #rng_algorithm<...>
                }
                if checkIdentifier("DEFAULT") {
                    attributes.rngAlgorithm = .defaultAlgorithm
                    advance()
                } else if checkIdentifier("THREE_FRY") {
                    attributes.rngAlgorithm = .threeFry
                    advance()
                } else if checkIdentifier("PHILOX") {
                    attributes.rngAlgorithm = .philox
                    advance()
                }
                if check(.rightAngle) {
                    advance()
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        return attributes
    }

    // MARK: - Select and Scatter Attribute Parsing

    private func parseSelectAndScatterAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        var windowDims: [Int] = []
        var windowStrides: [Int] = []
        var padding: [[Int]]? = nil

        while !check(.colon) && !check(.eof) {
            if checkIdentifier("window_dimensions") {
                try expectIdentifier("window_dimensions")
                try expect(.equal)
                windowDims = try parseArrayAttribute()
            } else if checkIdentifier("window_strides") {
                try expectIdentifier("window_strides")
                try expect(.equal)
                windowStrides = try parseArrayAttribute()
            } else if checkIdentifier("padding") {
                try expectIdentifier("padding")
                try expect(.equal)
                padding = try parsePaddingArray()
            } else if checkIdentifier("select") || checkIdentifier("scatter") {
                // Skip select and scatter regions for now
                while !check(.comma) && !check(.colon) && !check(.eof) {
                    if check(.leftBrace) {
                        var depth = 1
                        advance()
                        while depth > 0 && !check(.eof) {
                            if check(.leftBrace) { depth += 1 }
                            if check(.rightBrace) { depth -= 1 }
                            advance()
                        }
                    } else {
                        advance()
                    }
                }
            } else {
                break
            }
            _ = match(.comma)
        }

        if !windowDims.isEmpty {
            attributes.selectAndScatterDimensionNumbers = SelectAndScatterDimensionNumbers(
                windowDimensions: windowDims,
                windowStrides: windowStrides,
                padding: padding
            )
        }

        return attributes
    }

    // MARK: - Control Flow Region Parsing

    /// Parses while loop regions.
    /// Format: (%init) cond { ... stablehlo.return %pred } do { ... stablehlo.return %val }
    private func parseWhileRegions() throws -> WhileRegions {
        // Parse condition region
        // Format: cond { ^bb(%args): ops... stablehlo.return %pred }
        try expectIdentifier("cond")
        let condRegion = try parseRegion()

        // Parse body region
        // Format: do { ^bb(%args): ops... stablehlo.return %vals }
        skipNewlines()
        try expectIdentifier("do")
        let bodyRegion = try parseRegion()

        return WhileRegions(condition: condRegion, body: bodyRegion)
    }

    /// Parses if conditional regions.
    /// Format: then { ... } else { ... }
    private func parseIfRegions() throws -> IfRegions {
        // Parse then branch
        try expectIdentifier("then")
        let thenRegion = try parseRegion()

        // Parse optional else branch
        skipNewlines()
        var elseRegion: Region? = nil
        if checkIdentifier("else") {
            try expectIdentifier("else")
            elseRegion = try parseRegion()
        }

        return IfRegions(thenBranch: thenRegion, elseBranch: elseRegion)
    }

    /// Parses an inline operation region wrapped in parentheses.
    /// Format: ({ ^bb0(%arg0: type, %arg1: type): ops... stablehlo.return %val })
    ///
    /// Used by generic-form operations like scatter and reduce that embed
    /// a computation region between the operand list and the attribute dict.
    /// Returns the computation kind determined by the operation in the region body:
    /// - `stablehlo.return %arg1` (identity) → `.set`
    /// - `stablehlo.add` → `.add`
    /// - `stablehlo.maximum` → `.max`
    /// - `stablehlo.minimum` → `.min`
    /// - `stablehlo.multiply` → `.mul`
    private func parseOperationRegion() throws -> ScatterComputationKind? {
        try expect(.leftParen)
        try expect(.leftBrace)
        skipNewlines()

        var computationKind: ScatterComputationKind? = nil

        // Track brace depth to handle nested braces
        var braceDepth = 1

        // Scan through the region body looking for known operations
        while braceDepth > 0 && !check(.eof) {
            if check(.leftBrace) {
                braceDepth += 1
                advance()
            } else if check(.rightBrace) {
                braceDepth -= 1
                if braceDepth == 0 {
                    break
                }
                advance()
            } else if check(.identifier) || check(.percentIdentifier) {
                let text = currentToken.text

                // Check for stablehlo operations that determine computation kind
                if text == "stablehlo.add" || text == "add" {
                    computationKind = .add
                } else if text == "stablehlo.maximum" || text == "maximum" {
                    computationKind = .max
                } else if text == "stablehlo.minimum" || text == "minimum" {
                    computationKind = .min
                } else if text == "stablehlo.multiply" || text == "multiply" {
                    computationKind = .mul
                }
                // If we see stablehlo.return without having found an operation,
                // the region is identity (just returns the update argument)
                advance()
            } else {
                advance()
            }
        }

        // Consume closing } and )
        try expect(.rightBrace)
        try expect(.rightParen)
        skipNewlines()

        // If no computation operation was found, it's an identity/set operation
        // (region just does: stablehlo.return %arg1)
        if computationKind == nil {
            computationKind = .set
        }

        return computationKind
    }

    /// Parses a region block.
    /// Format: { ^bb(%arg0: type, ...): ops... stablehlo.return %vals }
    private func parseRegion() throws -> Region {
        try expect(.leftBrace)
        skipNewlines()

        var arguments: [RegionArgument] = []
        var operations: [HLOOperation] = []
        var returnValues: [String] = []

        // Check for block label with arguments: ^bb(%arg0: type, ...)
        // The lexer tokenizes this as: ^ (unknown), bb (identifier), ( , args..., ), :
        if currentToken.text == "^" {
            // Skip the ^ token
            advance()
            // Skip the block label identifier (e.g., "bb")
            if check(.identifier) {
                advance()
            }
            // Parse block arguments if present
            if check(.leftParen) {
                try expect(.leftParen)
                if !check(.rightParen) {
                    repeat {
                        let name = try parsePercentIdentifier()
                        try expect(.colon)
                        let type = try parseTensorType()
                        arguments.append(RegionArgument(name: name, type: type))
                    } while match(.comma)
                }
                try expect(.rightParen)
                _ = match(.colon)  // Optional colon after block args
            }
        }

        skipNewlines()

        // Parse operations until stablehlo.return or }
        while !check(.rightBrace) && !check(.eof) {
            // Check for stablehlo.return
            if checkIdentifier("stablehlo.return") {
                returnValues = try parseStablehloReturn()
                break
            }
            let op = try parseOperation()
            operations.append(op)
            skipNewlines()
        }

        skipNewlines()
        try expect(.rightBrace)

        return Region(
            arguments: arguments,
            operations: operations,
            returnValues: returnValues
        )
    }

    /// Parses stablehlo.return statement.
    /// Format: stablehlo.return %val1, %val2, ... : types
    private func parseStablehloReturn() throws -> [String] {
        try expectIdentifier("stablehlo.return")

        var values: [String] = []

        // Parse return values
        if check(.percentIdentifier) {
            repeat {
                let value = try parsePercentIdentifier()
                values.append(value)
            } while match(.comma)
        }

        // Skip optional type annotation
        if match(.colon) {
            _ = try parseTensorType()
            while match(.comma) {
                _ = try parseTensorType()
            }
        }

        return values
    }

    // MARK: - Type Parsing

    private func parseTensorType() throws -> TensorType {
        // Handle tuple type (skip and return placeholder — tuples are eliminated during parsing)
        if checkIdentifier("tuple") {
            return try skipTupleType()
        }
        try expectKeyword(.tensor)
        try expect(.leftAngle)

        var shape: [Int] = []
        var elementType: ElementType?

        // Parse shape dimensions and element type
        while !check(.rightAngle) {
            if let et = tryParseElementType() {
                elementType = et
                break
            }

            if case .integer(let dim) = currentToken.kind {
                shape.append(Int(dim))
                advance()
                if check(.identifier) && currentToken.text == "x" {
                    advance()
                }
            } else if currentToken.text == "x" {
                advance()
            } else {
                break
            }
        }

        try expect(.rightAngle)

        guard let et = elementType else {
            throw ParseError.invalidTensorType(
                "Missing element type",
                location: currentToken.location
            )
        }

        return TensorType(shape: shape, elementType: et)
    }

    private func tryParseElementType() -> ElementType? {
        let text = currentToken.text
        let elementType: ElementType?

        switch text {
        case "f16": elementType = .float16
        case "f32": elementType = .float32
        case "f64": elementType = .float64
        case "bf16": elementType = .bfloat16
        case "i1": elementType = .int1
        case "i8": elementType = .int8
        case "i16": elementType = .int16
        case "i32": elementType = .int32
        case "i64": elementType = .int64
        case "ui8": elementType = .uint8
        case "ui16": elementType = .uint16
        case "ui32": elementType = .uint32
        case "ui64": elementType = .uint64
        default: elementType = nil
        }

        if elementType != nil {
            advance()
        }

        return elementType
    }

    private func parseTypeSignature(for kind: HLOOpKind = .add) throws -> TensorType {
        // Handle both simple types: tensor<2x3xf32>
        // and function-like types: (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>

        if check(.leftParen) {
            // Function-like signature, skip to return type
            try expect(.leftParen)
            var depth = 1
            while depth > 0 && !check(.eof) {
                if check(.leftParen) { depth += 1 }
                if check(.rightParen) { depth -= 1 }
                advance()
            }
            try expect(.arrow)
            return try parseTensorType()
        }

        // For select: pred_type, result_type - skip predicate type and get result type
        if kind == .select {
            // Skip the predicate type (e.g., tensor<2x3xi1>)
            _ = try parseTensorType()
            _ = match(.comma)
            // Parse the result type
            return try parseTensorType()
        }

        return try parseTensorType()
    }

    /// Skips a `tuple<...>` type and returns a placeholder TensorType.
    /// Tuple operations are eliminated during parsing, so the type is not used.
    private func skipTupleType() throws -> TensorType {
        // Skip "tuple" identifier
        if checkIdentifier("tuple") {
            advance()
        }
        // Skip <...> with nested angle brackets
        if check(.leftAngle) {
            advance()
            var depth = 1
            while depth > 0 && !check(.eof) {
                if check(.leftAngle) { depth += 1 }
                if check(.rightAngle) { depth -= 1 }
                advance()
            }
        }
        return TensorType(shape: [], elementType: .int32)
    }

    /// Resolves a value through the alias chain (for tuple elimination).
    private func resolveAlias(_ value: String, aliases: [String: String]) -> String {
        var current = value
        var seen = Set<String>()
        while let alias = aliases[current], !seen.contains(current) {
            seen.insert(current)
            current = alias
        }
        return current
    }

    // MARK: - Attribute Parsing Helpers

    private func parseConstantValue() throws -> ConstantValue {
        try expectKeyword(.dense)
        try expect(.leftAngle)

        // Parse the constant data
        let value: ConstantValue

        if check(.leftBracket) {
            // Dense array
            let values = try parseDenseArray()
            // We'll get the type from the type signature later
            value = .dense(values, TensorType(shape: [], elementType: .float32))
        } else if case .float(let f) = currentToken.kind {
            advance()
            value = .scalar(f)
        } else if case .integer(let i) = currentToken.kind {
            // In MLIR, hex integers in dense<> represent float bit patterns
            // (e.g., 0xFF800000 is -inf as f32, not the integer 4286578688)
            let doubleValue = integerToConstantDouble(i, text: currentToken.text)
            advance()
            value = .scalar(doubleValue)
        } else {
            throw ParseError.invalidConstant(
                "Expected constant value",
                location: currentToken.location
            )
        }

        try expect(.rightAngle)
        return value
    }

    /// Converts an integer token to a Double for use in a constant value.
    ///
    /// In MLIR text format, hex integers in `dense<>` represent IEEE 754 float
    /// bit patterns: `0xFF800000` is `-inf` (f32), not the integer 4286578688.
    /// Decimal integers are plain numeric values (e.g., `dense<0>` is 0.0).
    private func integerToConstantDouble(_ value: Int64, text: String) -> Double {
        let isHex = text.hasPrefix("0x") || text.hasPrefix("0X")
            || text.hasPrefix("-0x") || text.hasPrefix("-0X")
        guard isHex else {
            return Double(value)
        }
        // Reinterpret hex value as a float bit pattern.
        // f32 bit patterns fit in 32 bits; f64 patterns use 64 bits.
        let bits = UInt64(bitPattern: value)
        if bits <= UInt64(UInt32.max) {
            return Double(Float(bitPattern: UInt32(bits)))
        } else {
            return Double(bitPattern: bits)
        }
    }

    private func parseDenseArray() throws -> [Double] {
        try expect(.leftBracket)
        var values: [Double] = []

        if !check(.rightBracket) {
            repeat {
                if check(.leftBracket) {
                    // Nested array
                    let nested = try parseDenseArray()
                    values.append(contentsOf: nested)
                } else if case .float(let f) = currentToken.kind {
                    values.append(f)
                    advance()
                } else if case .integer(let i) = currentToken.kind {
                    values.append(integerToConstantDouble(i, text: currentToken.text))
                    advance()
                } else if check(.minus) {
                    advance()
                    if case .float(let f) = currentToken.kind {
                        values.append(-f)
                        advance()
                    } else if case .integer(let i) = currentToken.kind {
                        values.append(Double(-i))
                        advance()
                    }
                }
            } while match(.comma)
        }

        try expect(.rightBracket)
        return values
    }

    private func parseDimensionList() throws -> [Int] {
        try expect(.leftBracket)
        var dims: [Int] = []

        if !check(.rightBracket) {
            repeat {
                dims.append(try parseInteger())
            } while match(.comma)
        }

        try expect(.rightBracket)
        return dims
    }

    private func parseInteger() throws -> Int {
        if case .integer(let value) = currentToken.kind {
            advance()
            return Int(value)
        }
        throw ParseError.unexpectedToken(expected: "integer", got: currentToken)
    }

    private func parseReduceAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        // Parse: applies stablehlo.add across dimensions = [1]
        // or other reduce patterns

        if checkIdentifier("applies") {
            try expectIdentifier("applies")
            let reductionOp = try parseOperationName()

            if reductionOp.contains("add") {
                attributes.reductionKind = .sum
            } else if reductionOp.contains("max") {
                attributes.reductionKind = .max
            } else if reductionOp.contains("min") {
                attributes.reductionKind = .min
            } else if reductionOp.contains("multiply") {
                attributes.reductionKind = .product
            } else if reductionOp.contains("and") {
                attributes.reductionKind = .and
            } else if reductionOp.contains("or") {
                attributes.reductionKind = .or
            }

            if checkIdentifier("across") {
                try expectIdentifier("across")
                try expectIdentifier("dimensions")
                try expect(.equal)
                attributes.dimensions = try parseDimensionList()
            }
        }

        return attributes
    }

    private func parseDotDimensionNumbers() throws -> DotDimensionNumbers? {
        var lhsBatching: [Int] = []
        var rhsBatching: [Int] = []
        var lhsContracting: [Int] = []
        var rhsContracting: [Int] = []

        // Handle verbose format: #stablehlo.dot<lhs_batching_dimensions = [], ...>
        if check(.hashIdentifier) && currentToken.text == "#stablehlo.dot" {
            advance()
            try expect(.leftAngle)

            while !check(.rightAngle) {
                if checkIdentifier("lhs_batching_dimensions") {
                    try expectIdentifier("lhs_batching_dimensions")
                    try expect(.equal)
                    lhsBatching = try parseDimensionList()
                } else if checkIdentifier("rhs_batching_dimensions") {
                    try expectIdentifier("rhs_batching_dimensions")
                    try expect(.equal)
                    rhsBatching = try parseDimensionList()
                } else if checkIdentifier("lhs_contracting_dimensions") {
                    try expectIdentifier("lhs_contracting_dimensions")
                    try expect(.equal)
                    lhsContracting = try parseDimensionList()
                } else if checkIdentifier("rhs_contracting_dimensions") {
                    try expectIdentifier("rhs_contracting_dimensions")
                    try expect(.equal)
                    rhsContracting = try parseDimensionList()
                }
                _ = match(.comma)
            }

            try expect(.rightAngle)
        }
        // Handle simple format: contracting_dims = [1] x [0]
        // and optionally: batching_dims = [0] x [0]
        else if checkIdentifier("contracting_dims") || checkIdentifier("batching_dims") {
            // Parse optional batching_dims = [lhs] x [rhs]
            if checkIdentifier("batching_dims") {
                try expectIdentifier("batching_dims")
                try expect(.equal)
                lhsBatching = try parseDimensionList()
                try expectIdentifier("x")
                rhsBatching = try parseDimensionList()
                _ = match(.comma)
            }

            // Parse contracting_dims = [lhs] x [rhs]
            if checkIdentifier("contracting_dims") {
                try expectIdentifier("contracting_dims")
                try expect(.equal)
                lhsContracting = try parseDimensionList()
                try expectIdentifier("x")
                rhsContracting = try parseDimensionList()
            }
        } else {
            // No dimension numbers provided
            return nil
        }

        return DotDimensionNumbers(
            lhsBatchingDimensions: lhsBatching,
            rhsBatchingDimensions: rhsBatching,
            lhsContractingDimensions: lhsContracting,
            rhsContractingDimensions: rhsContracting
        )
    }

    private func parseGatherDimensionNumbers() throws -> GatherDimensionNumbers? {
        // Parse gather dimension numbers in StableHLO MLIR format:
        // dimension_numbers = #stablehlo.gather<
        //   offset_dims = [...],
        //   collapsed_slice_dims = [...],
        //   start_index_map = [...],
        //   index_vector_dim = N
        // >,
        // slice_sizes = array<i64: ...>,
        // indices_are_sorted = false

        var offsetDims: [Int] = []
        var collapsedSliceDims: [Int] = []
        var startIndexMap: [Int] = []
        var indexVectorDim: Int = 0
        var sliceSizes: [Int] = []
        var operandBatchingDims: [Int] = []
        var startIndicesBatchingDims: [Int] = []

        while !check(.colon) && !check(.eof) {
            skipNewlines()

            if checkIdentifier("dimension_numbers") {
                // Parse: dimension_numbers = #stablehlo.gather<...>
                try expectIdentifier("dimension_numbers")
                try expect(.equal)

                // Skip #stablehlo.gather or similar prefix
                if check(.hashIdentifier) {
                    advance()
                }

                // Parse the inner content between < and >
                if match(.leftAngle) {
                    while !check(.rightAngle) && !check(.eof) {
                        skipNewlines()

                        if checkIdentifier("offset_dims") {
                            try expectIdentifier("offset_dims")
                            try expect(.equal)
                            offsetDims = try parseDimensionList()
                        } else if checkIdentifier("collapsed_slice_dims") {
                            try expectIdentifier("collapsed_slice_dims")
                            try expect(.equal)
                            collapsedSliceDims = try parseDimensionList()
                        } else if checkIdentifier("start_index_map") {
                            try expectIdentifier("start_index_map")
                            try expect(.equal)
                            startIndexMap = try parseDimensionList()
                        } else if checkIdentifier("index_vector_dim") {
                            try expectIdentifier("index_vector_dim")
                            try expect(.equal)
                            indexVectorDim = try parseInteger()
                        } else if checkIdentifier("operand_batching_dims") {
                            try expectIdentifier("operand_batching_dims")
                            try expect(.equal)
                            operandBatchingDims = try parseDimensionList()
                        } else if checkIdentifier("start_indices_batching_dims") {
                            try expectIdentifier("start_indices_batching_dims")
                            try expect(.equal)
                            startIndicesBatchingDims = try parseDimensionList()
                        } else {
                            // Skip unknown tokens
                            advance()
                        }
                        _ = match(.comma)
                        skipNewlines()
                    }
                    _ = match(.rightAngle)
                }
            } else if checkIdentifier("slice_sizes") {
                // Parse: slice_sizes = array<i64: 1, 3>
                try expectIdentifier("slice_sizes")
                try expect(.equal)

                // Handle array<i64: ...> format
                if checkIdentifier("array") {
                    try expectIdentifier("array")
                    if match(.leftAngle) {
                        // Skip type like "i64:"
                        while !check(.colon) && !check(.rightAngle) && !check(.eof) {
                            advance()
                        }
                        _ = match(.colon)

                        // Parse the integers
                        var sizes: [Int] = []
                        while !check(.rightAngle) && !check(.eof) {
                            if case .integer(let val) = currentToken.kind {
                                sizes.append(Int(val))
                                advance()
                            } else if check(.minus) {
                                advance()
                                if case .integer(let val) = currentToken.kind {
                                    sizes.append(-Int(val))
                                    advance()
                                }
                            } else {
                                _ = match(.comma)
                            }
                        }
                        _ = match(.rightAngle)
                        sliceSizes = sizes
                    }
                } else {
                    sliceSizes = try parseDimensionList()
                }
            } else if checkIdentifier("offset_dims") {
                // Fallback: flat format without dimension_numbers wrapper
                try expectIdentifier("offset_dims")
                try expect(.equal)
                offsetDims = try parseDimensionList()
            } else if checkIdentifier("collapsed_slice_dims") {
                try expectIdentifier("collapsed_slice_dims")
                try expect(.equal)
                collapsedSliceDims = try parseDimensionList()
            } else if checkIdentifier("start_index_map") {
                try expectIdentifier("start_index_map")
                try expect(.equal)
                startIndexMap = try parseDimensionList()
            } else if checkIdentifier("index_vector_dim") {
                try expectIdentifier("index_vector_dim")
                try expect(.equal)
                indexVectorDim = try parseInteger()
            } else if checkIdentifier("operand_batching_dims") {
                try expectIdentifier("operand_batching_dims")
                try expect(.equal)
                operandBatchingDims = try parseDimensionList()
            } else if checkIdentifier("start_indices_batching_dims") {
                try expectIdentifier("start_indices_batching_dims")
                try expect(.equal)
                startIndicesBatchingDims = try parseDimensionList()
            } else if checkIdentifier("slice_sizes") {
                try expectIdentifier("slice_sizes")
                try expect(.equal)
                sliceSizes = try parseDimensionList()
            } else if checkIdentifier("indices_are_sorted") {
                // Skip this attribute
                try expectIdentifier("indices_are_sorted")
                try expect(.equal)
                // Skip true/false
                if checkIdentifier("true") || checkIdentifier("false") {
                    advance()
                }
            } else {
                // Skip unknown tokens
                break
            }
            _ = match(.comma)
            skipNewlines()
        }

        // Only return if we parsed the essential fields
        guard !sliceSizes.isEmpty else {
            return nil
        }

        return GatherDimensionNumbers(
            offsetDims: offsetDims,
            collapsedSliceDims: collapsedSliceDims,
            startIndexMap: startIndexMap,
            indexVectorDim: indexVectorDim,
            sliceSizes: sliceSizes,
            operandBatchingDims: operandBatchingDims,
            startIndicesBatchingDims: startIndicesBatchingDims
        )
    }

    private func parseScatterAttributes() throws -> (ScatterDimensionNumbers?, ScatterComputationKind?) {
        // Parse scatter attributes in inline format:
        // update_window_dims = [...], inserted_window_dims = [...],
        // scatter_dims_to_operand_dims = [...], index_vector_dim = N,
        // input_batching_dims = [...], scatter_indices_batching_dims = [...],
        // computation = add/max/min/mul

        var updateWindowDims: [Int] = []
        var insertedWindowDims: [Int] = []
        var scatterDimsToOperandDims: [Int] = []
        var indexVectorDim: Int = 0
        var inputBatchingDims: [Int] = []
        var scatterIndicesBatchingDims: [Int] = []
        var computationKind: ScatterComputationKind? = nil

        while !check(.colon) && !check(.eof) {
            // Handle #stablehlo.scatter<...> wrapper format:
            // scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = ..., ...>
            if checkIdentifier("scatter_dimension_numbers") {
                try expectIdentifier("scatter_dimension_numbers")
                try expect(.equal)
                // Skip #stablehlo.scatter or #stablehlo<scatter<...>>
                if check(.hashIdentifier) {
                    advance() // skip #stablehlo.scatter
                }
                if check(.leftAngle) {
                    advance() // skip <
                    skipNewlines()
                }
                continue
            } else if checkIdentifier("indices_are_sorted") || checkIdentifier("unique_indices") {
                // Skip boolean attributes not needed for compilation
                advance() // name
                if match(.equal) {
                    advance() // value (true/false)
                }
                _ = match(.comma)
                skipNewlines()
                continue
            } else if check(.rightAngle) {
                // End of #stablehlo.scatter<...> block
                advance()
                _ = match(.comma)
                skipNewlines()
                continue
            } else if checkIdentifier("update_window_dims") {
                try expectIdentifier("update_window_dims")
                try expect(.equal)
                updateWindowDims = try parseDimensionList()
            } else if checkIdentifier("inserted_window_dims") {
                try expectIdentifier("inserted_window_dims")
                try expect(.equal)
                insertedWindowDims = try parseDimensionList()
            } else if checkIdentifier("scatter_dims_to_operand_dims") {
                try expectIdentifier("scatter_dims_to_operand_dims")
                try expect(.equal)
                scatterDimsToOperandDims = try parseDimensionList()
            } else if checkIdentifier("index_vector_dim") {
                try expectIdentifier("index_vector_dim")
                try expect(.equal)
                indexVectorDim = try parseInteger()
            } else if checkIdentifier("input_batching_dims") {
                try expectIdentifier("input_batching_dims")
                try expect(.equal)
                inputBatchingDims = try parseDimensionList()
            } else if checkIdentifier("scatter_indices_batching_dims") {
                try expectIdentifier("scatter_indices_batching_dims")
                try expect(.equal)
                scatterIndicesBatchingDims = try parseDimensionList()
            } else if checkIdentifier("computation") {
                try expectIdentifier("computation")
                try expect(.equal)
                // Parse computation kind: add, max, min, mul
                if checkIdentifier("add") {
                    try expectIdentifier("add")
                    computationKind = .add
                } else if checkIdentifier("max") {
                    try expectIdentifier("max")
                    computationKind = .max
                } else if checkIdentifier("min") {
                    try expectIdentifier("min")
                    computationKind = .min
                } else if checkIdentifier("mul") {
                    try expectIdentifier("mul")
                    computationKind = .mul
                } else if checkIdentifier("set") {
                    try expectIdentifier("set")
                    computationKind = .set
                }
            } else {
                // Skip unknown tokens
                break
            }
            _ = match(.comma)
        }

        // Return even if some fields are empty (they have defaults)
        let dimNumbers = ScatterDimensionNumbers(
            updateWindowDims: updateWindowDims,
            insertedWindowDims: insertedWindowDims,
            scatterDimsToOperandDims: scatterDimsToOperandDims,
            indexVectorDim: indexVectorDim,
            inputBatchingDims: inputBatchingDims,
            scatterIndicesBatchingDims: scatterIndicesBatchingDims
        )
        return (dimNumbers, computationKind)
    }

    private func parseComparisonDirection() throws -> ComparisonDirection? {
        let directions = ["EQ", "NE", "LT", "LE", "GT", "GE"]
        for dir in directions {
            if checkIdentifier(dir) {
                advance()
                _ = match(.comma)
                return ComparisonDirection(rawValue: dir)
            }
        }
        return nil
    }

    private func parseSliceAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        // Parse slice notation: [start:end:stride, ...]
        if check(.leftBracket) {
            advance()
            var starts: [Int] = []
            var limits: [Int] = []
            var strides: [Int] = []

            repeat {
                let start = try parseInteger()
                starts.append(start)
                try expect(.colon)
                let limit = try parseInteger()
                limits.append(limit)
                if match(.colon) {
                    let stride = try parseInteger()
                    strides.append(stride)
                } else {
                    strides.append(1)
                }
            } while match(.comma)

            try expect(.rightBracket)

            attributes.sliceStarts = starts
            attributes.sliceLimits = limits
            attributes.sliceStrides = strides
        }

        return attributes
    }

    private func parsePadAttributes() throws -> HLOAttributes {
        var attributes = HLOAttributes()

        // Parse: low = [...], high = [...], interior = [...]
        while !check(.colon) && !check(.eof) {
            if checkIdentifier("low") {
                try expectIdentifier("low")
                try expect(.equal)
                attributes.padLow = try parseDimensionList()
            } else if checkIdentifier("high") {
                try expectIdentifier("high")
                try expect(.equal)
                attributes.padHigh = try parseDimensionList()
            } else if checkIdentifier("interior") {
                try expectIdentifier("interior")
                try expect(.equal)
                attributes.padInterior = try parseDimensionList()
            }
            _ = match(.comma)
        }

        return attributes
    }

    private func parseRNGDistribution() throws -> RNGDistribution? {
        if checkIdentifier("UNIFORM") {
            advance()
            return .uniform
        } else if checkIdentifier("NORMAL") {
            advance()
            return .normal
        }
        return nil
    }

    // MARK: - Token Helpers

    private func advance() {
        if let peeked = peekedToken {
            currentToken = peeked
            peekedToken = nil
        } else {
            currentToken = lexer.nextToken()
        }
        // Skip newlines automatically
        while currentToken.kind == .newline {
            currentToken = lexer.nextToken()
        }
    }

    private func skipNewlines() {
        while currentToken.kind == .newline {
            currentToken = lexer.nextToken()
        }
    }

    private func check(_ kind: TokenKind) -> Bool {
        currentToken.kind == kind
    }

    private func checkKeyword(_ keyword: Keyword) -> Bool {
        if case .keyword(let kw) = currentToken.kind {
            return kw == keyword
        }
        return false
    }

    private func checkIdentifier(_ name: String) -> Bool {
        currentToken.kind == .identifier && currentToken.text == name
    }

    private func match(_ kind: TokenKind) -> Bool {
        if check(kind) {
            advance()
            return true
        }
        return false
    }

    private func expect(_ kind: TokenKind) throws {
        if !match(kind) {
            throw ParseError.unexpectedToken(expected: "\(kind)", got: currentToken)
        }
    }

    private func expectKeyword(_ keyword: Keyword) throws {
        if !checkKeyword(keyword) {
            throw ParseError.unexpectedToken(expected: keyword.rawValue, got: currentToken)
        }
        advance()
    }

    private func expectIdentifier(_ name: String) throws {
        if !checkIdentifier(name) {
            throw ParseError.unexpectedToken(expected: name, got: currentToken)
        }
        advance()
    }

    private func parseAtIdentifier() throws -> String {
        guard case .atIdentifier = currentToken.kind else {
            throw ParseError.unexpectedToken(expected: "@identifier", got: currentToken)
        }
        let name = String(currentToken.text.dropFirst())  // Remove @
        advance()
        return name
    }

    private func parsePercentIdentifier() throws -> String {
        guard case .percentIdentifier = currentToken.kind else {
            throw ParseError.unexpectedToken(expected: "%identifier", got: currentToken)
        }
        let name = currentToken.text
        advance()
        return name
    }
}
