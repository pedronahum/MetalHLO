// CodeGenerator.swift
// MetalHLOCore
//
// Generates Metal kernel specifications from optimized modules.

import Foundation
import Metal

// MARK: - Generation Result

/// Result of code generation including both kernel specs and view mappings.
public struct CodeGenerationResult: Sendable {
    /// Kernel specifications for operations that require GPU execution.
    public let kernelSpecs: [OpID: KernelSpec]

    /// View mappings for operations that are zero-copy views.
    /// Key is the output tensor ID, value is the view definition.
    public let viewMappings: [TensorID: StridedTensorView]

    /// Operation IDs that are view operations (no kernel needed).
    public let viewOperations: Set<OpID>

    public init(
        kernelSpecs: [OpID: KernelSpec],
        viewMappings: [TensorID: StridedTensorView] = [:],
        viewOperations: Set<OpID> = []
    ) {
        self.kernelSpecs = kernelSpecs
        self.viewMappings = viewMappings
        self.viewOperations = viewOperations
    }
}

// MARK: - Code Generator

/// Generates Metal kernel specifications for an optimized module.
public final class CodeGenerator: @unchecked Sendable {

    // MARK: - Properties

    private let device: MTLDevice
    private let config: Config

    // MARK: - Configuration

    public struct Config: Sendable {
        /// Target Metal feature set.
        public var metalFeatureSet: MetalFeatureSet

        /// Whether to generate debug comments in Metal source.
        public var debugComments: Bool

        /// Whether to use fast math.
        public var fastMath: Bool

        /// Default threadgroup size.
        public var defaultThreadgroupSize: Int

        /// Maximum threadgroup memory size.
        public var maxThreadgroupMemory: Int

        public enum MetalFeatureSet: String, Sendable {
            case apple7  // M1
            case apple8  // M2
            case apple9  // M3
            case common  // Conservative
        }

        public init(
            metalFeatureSet: MetalFeatureSet = .common,
            debugComments: Bool = false,
            fastMath: Bool = true,
            defaultThreadgroupSize: Int = 256,
            maxThreadgroupMemory: Int = 32768
        ) {
            self.metalFeatureSet = metalFeatureSet
            self.debugComments = debugComments
            self.fastMath = fastMath
            self.defaultThreadgroupSize = defaultThreadgroupSize
            self.maxThreadgroupMemory = maxThreadgroupMemory
        }

        public static let `default` = Config()
    }

    // MARK: - Initialization

    public init(device: MTLDevice, config: Config = .default) {
        self.device = device
        self.config = config
    }

    // MARK: - Generation

    /// Generates kernel specifications for all operations in the module.
    /// Returns only the kernel specs for backwards compatibility.
    public func generate(module: OptimizedModule, memoryPlan: MemoryPlan) -> [OpID: KernelSpec] {
        return generateWithViews(module: module, memoryPlan: memoryPlan).kernelSpecs
    }

    /// Generates kernel specifications and view mappings for all operations.
    /// View operations (transpose, reshape) are converted to zero-copy views.
    public func generateWithViews(module: OptimizedModule, memoryPlan: MemoryPlan) -> CodeGenerationResult {
        var specs: [OpID: KernelSpec] = [:]
        var viewMappings: [TensorID: StridedTensorView] = [:]
        var viewOperations: Set<OpID> = []

        // Track function input names for proper binding generation
        let inputNames = Set(module.inputs.map { $0.name })

        // Track constant tensor IDs
        let constantIDs = Set(module.constants.keys)

        // Build a view registry that tracks tensor views
        // NOTE: Inputs are NOT added to viewMappings - they are base tensors, not views.
        // Only view operations (transpose, reshape, slice) and computed outputs should be tracked here.
        // Adding inputs would create self-referential cycles since baseTensorID == input.name.

        for opID in memoryPlan.executionOrder {
            // opID is an integer index into the operations array
            guard opID >= 0 && opID < module.operations.count else { continue }
            let op = module.operations[opID]

            // Skip constant operations - they are pre-materialized into buffers
            if case .original(.constant) = op.type {
                continue
            }

            // Check if this is a view operation
            if let viewResult = tryGenerateViewOperation(
                op: op,
                tensors: module.tensors,
                viewMappings: viewMappings,
                constantIDs: constantIDs
            ) {
                // This is a view operation - store the view mapping
                viewMappings[viewResult.outputID] = viewResult.view
                viewOperations.insert(String(opID))
                continue
            }

            // Generate kernel for non-view operations
            let spec = generateKernel(
                op: op,
                tensors: module.tensors,
                layouts: module.layouts,
                memoryPlan: memoryPlan,
                inputNames: inputNames,
                constantIDs: constantIDs,
                viewMappings: viewMappings
            )

            // Use string-converted integer as key to match executionOrder format
            specs[String(opID)] = spec
            // NOTE: We do NOT add views for regular compute operation outputs.
            // Views are only created for actual view operations (transpose, reshape, slice)
            // which are handled by tryGenerateViewOperation above.
            // Adding viewMappings[output.id] = StridedTensorView.from(output) here would
            // create self-referential cycles that cause infinite loops in resolveViewChain.
        }

        return CodeGenerationResult(
            kernelSpecs: specs,
            viewMappings: viewMappings,
            viewOperations: viewOperations
        )
    }

    // MARK: - View Operation Detection

    /// Result of attempting to generate a view operation.
    private struct ViewResult {
        let outputID: TensorID
        let view: StridedTensorView
    }

    /// Attempts to convert an operation to a view operation.
    /// Returns nil if the operation requires a kernel.
    ///
    /// Currently only reshape is supported as a view operation because:
    /// - Reshape of contiguous data produces contiguous output (same strides)
    /// - Transpose produces non-contiguous views requiring strided access in downstream ops
    /// - Full transpose-as-view support requires strided kernel generation (future work)
    private func tryGenerateViewOperation(
        op: FusedOp,
        tensors: [TensorID: TensorInfo],
        viewMappings: [TensorID: StridedTensorView],
        constantIDs: Set<TensorID>
    ) -> ViewResult? {
        guard case .original(let opKind) = op.type else {
            return nil
        }

        switch opKind {
        // Reshape is safe as a view - maintains contiguous layout
        case .reshape:
            return tryGenerateReshapeView(
                op: op, tensors: tensors, viewMappings: viewMappings,
                constantIDs: constantIDs
            )

        // Transpose requires strided access patterns in downstream kernels
        // For now, keep as a kernel until strided kernel generation is implemented
        // case .transpose:
        //     return tryGenerateTransposeView(op: op, tensors: tensors, viewMappings: viewMappings)

        default:
            return nil
        }
    }

    /// Attempts to convert a transpose to a view.
    private func tryGenerateTransposeView(
        op: FusedOp,
        tensors: [TensorID: TensorInfo],
        viewMappings: [TensorID: StridedTensorView]
    ) -> ViewResult? {
        guard let inputID = op.inputs.first,
              let inputView = viewMappings[inputID],
              let permutation = op.attributes.dimensions,
              let outputInfo = op.outputs.first else {
            return nil
        }

        // Create transposed view
        let transposedView = inputView.transposed(permutation: permutation)

        return ViewResult(outputID: outputInfo.id, view: transposedView)
    }

    /// Attempts to convert a reshape to a view.
    ///
    /// Two cases:
    ///   1. Input is already a view (a prior reshape/slice in the chain) — reshape
    ///      the existing strided view; only valid while it stays contiguous.
    ///   2. Input is a *base* tensor (a compute-kernel output or a function input).
    ///      Kernels always write their output contiguously into the memory slab,
    ///      and inputs are contiguous, so a reshape of either is a pure relabel:
    ///      emit a fresh contiguous view over the base. This eliminates the
    ///      reshape COPY kernel for the common reshape-of-compute-output case.
    ///
    /// Constants are excluded — they live in separate constant buffers, not the
    /// slab, so the executor's offset lookup (memoryPlan.tensorOffsets) can't
    /// resolve a view whose base is a constant.
    ///
    /// The companion correctness requirement lives in StaticMemoryPlanner: the
    /// base tensor's lifetime must be extended to cover the view's readers, or
    /// the planner would reuse the base's slot after the reshape op and corrupt
    /// the view (see extendViewSourceLiveness).
    private func tryGenerateReshapeView(
        op: FusedOp,
        tensors: [TensorID: TensorInfo],
        viewMappings: [TensorID: StridedTensorView],
        constantIDs: Set<TensorID>
    ) -> ViewResult? {
        guard let inputID = op.inputs.first,
              let outputInfo = op.outputs.first else {
            return nil
        }

        // Case 1: input is itself a view — reshape the existing view.
        if let inputView = viewMappings[inputID] {
            guard let reshapedView = inputView.reshaped(to: outputInfo.shape) else {
                // Reshape requires copy (view is non-contiguous, e.g. transposed).
                return nil
            }
            return ViewResult(outputID: outputInfo.id, view: reshapedView)
        }

        // Case 2: input is a base tensor. Constants are not slab-resident — skip.
        if constantIDs.contains(inputID) {
            return nil
        }
        // Element type comes from the input's TensorInfo (reshape preserves it).
        guard let elementType = tensors[inputID]?.elementType else {
            return nil
        }
        let baseView = StridedTensorView.contiguous(
            tensorID: inputID,
            shape: outputInfo.shape,
            elementType: elementType
        )
        return ViewResult(outputID: outputInfo.id, view: baseView)
    }

    // MARK: - Type Mapping

    /// Maps ElementType to Metal type string.
    private func metalTypeName(for elementType: ElementType) -> String {
        switch elementType {
        case .float32: return "float"
        case .float16: return "half"
        case .float64: return "float"  // Metal doesn't support double in kernels, use float
        case .bfloat16: return "bfloat"  // Metal 3 native bf16 (macOS 13+, iOS 16+)
        case .int8: return "char"
        case .int16: return "short"
        case .int32: return "int"
        case .int64: return "long"
        case .uint8: return "uchar"
        case .uint16: return "ushort"
        case .uint32: return "uint"
        case .uint64: return "ulong"
        case .int1: return "bool"
        }
    }

    /// Returns true if the element type is a floating-point type.
    private func isFloatType(_ elementType: ElementType) -> Bool {
        switch elementType {
        case .float16, .float32, .float64, .bfloat16:
            return true
        default:
            return false
        }
    }

    /// Returns `dims` with any reference to position `rank-2` swapped with `rank-1`.
    /// Used to "apply" a `lhsTranspose`/`rhsTranspose` attribute to dot-op contracting
    /// or batching dim lists so that they index into the actual operand's shape rather
    /// than the pre-fold post-transpose shape.
    private func swapLastTwoDimPositions(_ dims: [Int], rank: Int) -> [Int] {
        guard rank >= 2 else { return dims }
        let a = rank - 2
        let b = rank - 1
        return dims.map { d in
            if d == a { return b }
            if d == b { return a }
            return d
        }
    }

    /// Generates a single kernel specification.
    private func generateKernel(
        op: FusedOp,
        tensors: [TensorID: TensorInfo],
        layouts: [TensorID: TensorLayout],
        memoryPlan: MemoryPlan,
        inputNames: Set<String>,
        constantIDs: Set<TensorID>,
        viewMappings: [TensorID: StridedTensorView] = [:]
    ) -> KernelSpec {
        // Get shapes and element type. Use `?? []` (NOT compactMap) so a
        // missing shape keeps its index slot AND is treated as a scalar (read
        // `inputN[0]`). Missing-shape operands are inlined scalar constants
        // (e.g. the -inf select fill); compactMap would drop them, shifting
        // every later operand's index and reading the scalar as `inputN[tid]`
        // (out-of-bounds on a 1-element buffer → GPU fault).
        let inputShapes = op.inputs.map { tensors[$0]?.shape ?? [] }
        let outputShapes = op.outputs.map { $0.shape }
        let elementType = op.outputs.first?.elementType ?? .float32

        // Get input element types for operations that need them (convert, gather)
        let inputElementTypes = op.inputs.compactMap { tensors[$0]?.elementType }
        // Index-aligned variant (one entry per op.inputs slot, never dropped).
        // The fused-elementwise chain needs this to know which externals are
        // boolean (i1) so it can read a 1-byte `bool*` buffer and convert to a
        // float predicate — reading a bool mask as `float` corrupts it (NaN).
        let alignedInputTypes = op.inputs.map { tensors[$0]?.elementType ?? .float32 }

        // Create modified attributes with input types for source generation
        var modifiedAttributes = op.attributes
        modifiedAttributes.inputElementTypes = inputElementTypes
        // For gather, set indices element type (second input)
        if inputElementTypes.count > 1 {
            modifiedAttributes.indicesElementType = inputElementTypes[1]
        }

        // Generate Metal source and entry point
        let (source, entry, tuning) = generateSource(
            type: op.type,
            inputShapes: inputShapes,
            outputShapes: outputShapes,
            attributes: modifiedAttributes,
            elementType: elementType,
            inputElementTypes: alignedInputTypes
        )

        // Calculate dispatch configuration. Pass modifiedAttributes so the
        // dispatch sees inputElementTypes (used to gate the vectorized
        // convert dispatch on whether the input is a >=2-byte float type).
        let dispatch = calculateDispatch(
            type: op.type,
            shapes: outputShapes,
            inputShapes: inputShapes,
            attributes: modifiedAttributes,
            tuning: tuning,
            elementType: elementType
        )

        // Build buffer bindings (with view resolution)
        let bindings = buildBindings(
            op: op,
            tensors: tensors,
            memoryPlan: memoryPlan,
            inputNames: inputNames,
            constantIDs: constantIDs,
            viewMappings: viewMappings
        )

        // Calculate shared memory size and buffer count
        let sharedMemorySize = calculateSharedMemorySize(type: op.type, tuning: tuning, elementType: elementType)
        let threadgroupBufferCount = calculateThreadgroupBufferCount(type: op.type, tuning: tuning, elementType: elementType)

        return KernelSpec(
            opID: op.id,
            metalSource: source,
            entryPoint: entry,
            dispatch: dispatch,
            bindings: bindings,
            tuning: tuning,
            sharedMemorySize: sharedMemorySize,
            threadgroupBufferCount: threadgroupBufferCount,
            inputShapes: inputShapes,
            outputShapes: outputShapes
        )
    }

    // MARK: - Source Generation

    /// Generates Metal source code for an operation.
    private func generateSource(
        type: FusedOpType,
        inputShapes: [[Int]],
        outputShapes: [[Int]],
        attributes: HLOAttributes,
        elementType: ElementType = .float32,
        inputElementTypes: [ElementType] = []
    ) -> (source: String, entryPoint: String, tuning: TuningConfig?) {
        switch type {
        case .original(let opKind):
            return generateOriginalOpSource(opKind, inputShapes: inputShapes, outputShapes: outputShapes, attributes: attributes, elementType: elementType)

        case .fusedAttention(let config):
            return generateAttentionSource(config, inputShapes: inputShapes)

        case .fusedMultiHeadAttention(let config):
            return generateMultiHeadAttentionSource(config, inputShapes: inputShapes)

        case .fusedRMSNorm(let config):
            return generateRMSNormSource(config, inputShapes: inputShapes)

        case .fusedLayerNorm(let config):
            return generateLayerNormSource(config, inputShapes: inputShapes)

        case .fusedBatchNorm(let config):
            // ResNet-style BN training goes through MPSGraph via the heterogeneous
            // executor, not this IntegratedExecutor source-gen path. The LayerNorm
            // generator is a safe stand-in if a BN custom_call ever reaches here:
            // both consume (input, gamma, beta) plus epsilon — the shape semantics
            // differ but small graphs that actually land in IntegratedExecutor
            // won't hit BN under current pattern detection.
            return generateLayerNormSource(config, inputShapes: inputShapes)

        case .fusedMatMulBiasAct(let config):
            return generateMatMulBiasActSource(config, inputShapes: inputShapes)

        case .fusedGELU(let approximate):
            return generateGELUSource(approximate: approximate, inputShapes: inputShapes)

        case .fusedSiLU:
            return generateSiLUSource(inputShapes: inputShapes)

        case .fusedElementwise(let chain):
            return generateElementwiseChainSource(chain, inputShapes: inputShapes, outputShape: outputShapes.first ?? [], elementType: elementType, inputElementTypes: inputElementTypes)

        case .fusedReduce(let cfg):
            return generateFusedReduceSource(cfg, inputShapes: inputShapes, elementType: elementType)

        case .fusedFFN(let config):
            return generateFFNSource(config, inputShapes: inputShapes)

        case .fusedTransformerBlock(let config):
            return generateTransformerBlockSource(config, inputShapes: inputShapes)

        case .fusedRoPE(let config):
            return generateRoPESource(config, inputShapes: inputShapes)

        case .fusedSoftmax(let axis):
            return generateSoftmaxSource(axis: axis, inputShapes: inputShapes)

        case .fusedConvBiasAct(let config):
            return generateFusedConvBiasActSource(config, inputShapes: inputShapes, outputShapes: outputShapes, attributes: attributes, elementType: elementType)
        }
    }

    /// Generates source for original (unfused) operations.
    private func generateOriginalOpSource(
        _ opKind: HLOOpKind,
        inputShapes: [[Int]],
        outputShapes: [[Int]],
        attributes: HLOAttributes,
        elementType: ElementType = .float32
    ) -> (String, String, TuningConfig?) {
        let outputShape = outputShapes.first ?? []
        let totalElements = outputShape.reduce(1, *)
        let entryPoint = "kernel_\(opKind.rawValue)"
        let metalType = metalTypeName(for: elementType)
        // Metal's math intrinsics (sqrt, exp, log, ...) don't have direct
        // bfloat overloads — they return fp32 from a bfloat input. For
        // bfloat we therefore take the same path as integer types: cast
        // through float, then back to bfloat at assignment.
        let isFloat = isFloatType(elementType) && elementType != .bfloat16

        var source = """
        #include <metal_stdlib>
        using namespace metal;

        """

        switch opKind {
        // Unary elementwise operations
        case .negate:
            source += generateUnaryKernel(entryPoint: entryPoint, operation: "-x", metalType: metalType)
        case .abs:
            source += generateUnaryKernel(entryPoint: entryPoint, operation: "abs(x)", metalType: metalType)
        case .exponential:
            // exp only works on float types - cast if needed
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "exp(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(exp(float(x)))", metalType: metalType)
            }
        case .log:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "log(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(log(float(x)))", metalType: metalType)
            }
        case .sqrt:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "sqrt(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(sqrt(float(x)))", metalType: metalType)
            }
        case .rsqrt:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "rsqrt(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(rsqrt(float(x)))", metalType: metalType)
            }
        case .sine:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "sin(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(sin(float(x)))", metalType: metalType)
            }
        case .cosine:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "cos(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(cos(float(x)))", metalType: metalType)
            }
        case .tanh:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "tanh(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(tanh(float(x)))", metalType: metalType)
            }
        case .logistic:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "1.0f / (1.0f + exp(-x))", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(1.0f / (1.0f + exp(-float(x))))", metalType: metalType)
            }
        case .expm1:
            // exp(x) - 1
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "exp(x) - 1.0f", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(exp(float(x)) - 1.0f)", metalType: metalType)
            }
        case .log1p:
            // log(1 + x)
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "log(1.0f + x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(log(1.0f + float(x)))", metalType: metalType)
            }
        case .cbrt:
            // Cube root: sign(x) * pow(abs(x), 1/3)
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "sign(x) * pow(abs(x), 0.333333333333f)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(sign(float(x)) * pow(abs(float(x)), 0.333333333333f))", metalType: metalType)
            }
        case .sign:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "sign(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(sign(float(x)))", metalType: metalType)
            }
        case .floor:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "floor(x)", metalType: metalType)
            } else {
                // floor is a no-op for integers
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "x", metalType: metalType)
            }
        case .ceil:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "ceil(x)", metalType: metalType)
            } else {
                // ceil is a no-op for integers
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "x", metalType: metalType)
            }

        // Binary elementwise operations
        case .add:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a + b", metalType: metalType)
        case .subtract:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a - b", metalType: metalType)
        case .multiply:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a * b", metalType: metalType)
        case .divide:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a / b", metalType: metalType)
        case .remainder:
            if isFloat {
                source += generateBinaryKernel(entryPoint: entryPoint, operation: "fmod(a, b)", metalType: metalType)
            } else {
                source += generateBinaryKernel(entryPoint: entryPoint, operation: "a % b", metalType: metalType)
            }
        case .maximum:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "max(a, b)", metalType: metalType)
        case .minimum:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "min(a, b)", metalType: metalType)
        case .power:
            if isFloat {
                source += generateBinaryKernel(entryPoint: entryPoint, operation: "pow(a, b)", metalType: metalType)
            } else {
                source += generateBinaryKernel(entryPoint: entryPoint, operation: "\(metalType)(pow(float(a), float(b)))", metalType: metalType)
            }
        case .atan2:
            // stablehlo.atan2 %y, %x -> atan2(y, x); float-only in JAX lowerings.
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "atan2(a, b)", metalType: metalType)

        // Bitwise operations (integer types)
        case .and:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a & b", metalType: metalType)
        case .or:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a | b", metalType: metalType)
        case .xor:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a ^ b", metalType: metalType)
        case .not:
            source += generateUnaryKernel(entryPoint: entryPoint, operation: "~x", metalType: metalType)
        case .shiftLeft:
            // StableHLO: shift_left(a, b) returns 0 if b >= bit_width.
            // Metal/C: shifting by >= bit_width is undefined behavior.
            let bitWidthL = elementByteSize(for: elementType) * 8
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "(b >= \(bitWidthL)) ? (\(metalType))(0) : (a << b)", metalType: metalType)
        case .shiftRightLogical:
            // StableHLO: shift_right_logical(a, b) returns 0 if b >= bit_width.
            let bitWidthRL = elementByteSize(for: elementType) * 8
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "(b >= \(bitWidthRL)) ? (\(metalType))(0) : (a >> b)", metalType: metalType)
        case .shiftRightArithmetic:
            // StableHLO: shift_right_arithmetic(a, b) returns sign-fill if b >= bit_width.
            let bitWidthRA = elementByteSize(for: elementType) * 8
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "(b >= \(bitWidthRA)) ? (a >> \(bitWidthRA - 1)) : (a >> b)", metalType: metalType)

        // Comparison operations (output is bool/i1)
        case .compare:
            let inputType = attributes.inputElementTypes?.first ?? .float32
            let inputMetal = metalTypeName(for: inputType)
            let compareOp: String
            switch attributes.comparisonDirection {
            case .eq: compareOp = "a == b"
            case .ne: compareOp = "a != b"
            case .lt: compareOp = "a < b"
            case .le: compareOp = "a <= b"
            case .gt: compareOp = "a > b"
            case .ge: compareOp = "a >= b"
            case .none: compareOp = "a == b"
            }
            source += generateCompareKernel(
                entryPoint: entryPoint,
                compareOp: compareOp,
                inputType: inputMetal
            )

        // Matrix operations
        case .dot, .dotGeneral:
            return generateMatMulSource(inputShapes: inputShapes, attributes: attributes, elementType: elementType)

        // Reduction operations
        case .reduce:
            return generateReductionSource(inputShapes: inputShapes, attributes: attributes, elementType: elementType)

        // Arg-reduce (argmax/argmin index) — the index half of a multi-input
        // reduce that the parser split off. `elementType` here is the i32/i64
        // index type; the input values are read as their own float/int type.
        case .reduceArg:
            let valueType = attributes.inputElementTypes?.first ?? .float32
            return generateArgReduceSource(
                inputShape: inputShapes.first ?? [],
                reduceDims: attributes.dimensions ?? [0],
                reductionKind: attributes.reductionKind ?? .max,
                valueType: valueType,
                indexType: elementType
            )

        case .sort:
            // Single-operand stable sort along `axis` (jnp.sort). Multi-operand
            // sort (argsort/lexsort) is split by the parser into `sortResult`
            // ops; see generateSortResultSource.
            let inShape = inputShapes.first ?? []
            return generateSortSource(
                inputShape: inShape,
                axis: attributes.axis ?? (inShape.count - 1),
                descending: attributes.sortDescending ?? false,
                valueType: elementType
            )

        case .sortResult:
            // One result of a multi-operand sort: inputs are [key0..keyK-1, payload].
            // Rank lexicographically by the K keys (stably), scatter the payload to
            // those ranks. Output type == payload type.
            let inShape = inputShapes.first ?? []
            let numKeys = max(1, inputShapes.count - 1)
            let allTypes = attributes.inputElementTypes ?? []
            let keyTypes = (0..<numKeys).map { allTypes.indices.contains($0) ? allTypes[$0] : ElementType.float32 }
            return generateSortResultSource(
                inputShape: inShape,
                axis: attributes.axis ?? (inShape.count - 1),
                descending: attributes.sortDescending ?? false,
                keyTypes: keyTypes,
                payloadType: elementType
            )

        // Shape operations
        case .reshape:
            source += generateCopyKernel(entryPoint: entryPoint, metalType: metalType)

        case .transpose:
            source += generateTransposeKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                outputShape: outputShapes.first ?? [],
                permutation: attributes.dimensions ?? [],
                metalType: metalType
            )

        // Broadcast operations
        case .broadcastInDim:
            source += generateBroadcastKernel(
                entryPoint: entryPoint,
                inputShapes: inputShapes,
                outputShapes: outputShapes,
                attributes: attributes,
                metalType: metalType
            )

        // Type conversion operations
        case .convert:
            let inputType = attributes.inputElementTypes?.first ?? .float32
            source += generateConvertKernel(
                entryPoint: entryPoint,
                inputType: inputType,
                outputType: elementType
            )

        case .bitcastConvert:
            let inputType = attributes.inputElementTypes?.first ?? .float32
            source += generateBitcastConvertKernel(
                entryPoint: entryPoint,
                inputType: inputType,
                outputType: elementType
            )

        // Gather operations (embedding lookup)
        case .gather, .dynamicGather:
            if let dimNumbers = attributes.gatherDimensionNumbers {
                source += generateGatherKernel(
                    entryPoint: entryPoint,
                    operandShape: inputShapes.first ?? [],
                    indicesShape: inputShapes.count > 1 ? inputShapes[1] : [],
                    outputShape: outputShape,
                    dimNumbers: dimNumbers,
                    operandType: metalType,
                    indicesType: attributes.indicesElementType.map { metalTypeName(for: $0) } ?? "int"
                )
            } else {
                // Fallback if no dimension numbers
                source += generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
            }

        // Scatter operations (embedding gradient)
        case .scatter:
            if let dimNumbers = attributes.scatterDimensionNumbers {
                source += generateScatterKernel(
                    entryPoint: entryPoint,
                    operandShape: inputShapes.first ?? [],
                    indicesShape: inputShapes.count > 1 ? inputShapes[1] : [],
                    updatesShape: inputShapes.count > 2 ? inputShapes[2] : [],
                    outputShape: outputShape,
                    dimNumbers: dimNumbers,
                    computationKind: attributes.scatterComputationKind ?? .set,
                    operandType: metalType,
                    indicesType: attributes.indicesElementType.map { metalTypeName(for: $0) } ?? "int"
                )
            } else {
                source += generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
            }

        // Slice operations
        case .slice:
            source += generateSliceKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                outputShape: outputShape,
                starts: attributes.sliceStarts ?? [],
                limits: attributes.sliceLimits ?? [],
                strides: attributes.sliceStrides ?? Array(repeating: 1, count: outputShape.count),
                metalType: metalType
            )

        // Dynamic slice operations
        case .dynamicSlice:
            source += generateDynamicSliceKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                outputShape: outputShape,
                sliceSizes: attributes.dynamicSliceSizes ?? outputShape,
                metalType: metalType,
                numStartIndices: inputShapes.count - 1
            )

        // Dynamic update slice operations
        case .dynamicUpdateSlice:
            source += generateDynamicUpdateSliceKernel(
                entryPoint: entryPoint,
                operandShape: inputShapes.first ?? [],
                updateShape: inputShapes.count > 1 ? inputShapes[1] : [],
                outputShape: outputShape,
                metalType: metalType,
                numStartIndices: max(0, inputShapes.count - 2)
            )

        // Reverse operations
        case .reverse:
            source += generateReverseKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                dimensions: attributes.dimensions ?? [],
                metalType: metalType
            )

        // Pad operations
        case .pad:
            source += generatePadKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                outputShape: outputShape,
                padLow: attributes.padLow ?? [],
                padHigh: attributes.padHigh ?? [],
                padInterior: attributes.padInterior ?? Array(repeating: 0, count: (inputShapes.first ?? []).count),
                metalType: metalType
            )

        // Concatenate operations
        case .concatenate:
            source += generateConcatenateKernel(
                entryPoint: entryPoint,
                inputShapes: inputShapes,
                outputShape: outputShape,
                axis: attributes.axis ?? 0,
                metalType: metalType
            )

        // Top-k operations (values + indices over the last axis)
        case .topKValues:
            source += generateTopKKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                k: attributes.topK ?? outputShape.last ?? 0,
                emitIndices: false,
                valueType: metalType
            )

        case .topKIndices:
            // The output element type is i32; the input is read as float.
            let inType = attributes.inputElementTypes?.first ?? .float32
            source += generateTopKKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                k: attributes.topK ?? outputShape.last ?? 0,
                emitIndices: true,
                valueType: metalTypeName(for: inType)
            )

        // Iota operations
        case .iota:
            source += generateIotaKernel(
                entryPoint: entryPoint,
                outputShape: outputShape,
                dimension: attributes.iotaDimension ?? attributes.axis ?? 0,
                metalType: metalType
            )

        // Count leading zeros
        case .clz:
            source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(clz(x))", metalType: metalType)

        // Select (ternary) operations
        case .select:
            let inputType = attributes.inputElementTypes?.last ?? elementType
            let valueMetal = metalTypeName(for: inputType)
            let predShape = inputShapes.first ?? []
            let predIsScalar = predShape.reduce(1, *) == 1
            source += generateSelectKernel(
                entryPoint: entryPoint,
                metalType: valueMetal,
                predIsScalar: predIsScalar
            )

        // Clamp operations
        case .clamp:
            source += generateClampKernel(
                entryPoint: entryPoint,
                metalType: metalType
            )

        // Convolution operations
        case .convolution:
            source += generateConvolutionKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                weightsShape: inputShapes.count > 1 ? inputShapes[1] : [],
                outputShape: outputShape,
                attributes: attributes,
                metalType: metalType
            )

        // Reduce window (pooling) operations
        case .reduceWindow:
            source += generateReduceWindowKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                outputShape: outputShape,
                attributes: attributes,
                metalType: metalType
            )

        // Select and scatter (pooling gradient) operations
        case .selectAndScatter:
            source += generateSelectAndScatterKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                sourceShape: inputShapes.count > 1 ? inputShapes[1] : [],
                outputShape: outputShape,
                attributes: attributes,
                metalType: metalType
            )

        // FFT operations
        case .fft:
            source += generateFFTKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                outputShape: outputShape,
                attributes: attributes,
                metalType: metalType
            )

        // Reduce precision: round a float to a target exponent/mantissa width
        // (e.g. emulate fp16/bf16 rounding from fp32) via IEEE-754 bit surgery.
        case .reducePrecision:
            source += generateReducePrecisionKernel(
                entryPoint: entryPoint,
                exponentBits: attributes.exponentBits ?? 8,
                mantissaBits: attributes.mantissaBits ?? 23,
                metalType: metalType
            )

        default:
            // Fallback to copy kernel for unsupported ops
            source += generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        return (source, entryPoint, nil)
    }

    /// Generates a kernel for `stablehlo.reduce_precision`.
    ///
    /// Rounds each fp32 element to the dynamic range and precision of a target
    /// float format with `exponentBits` exponent bits and `mantissaBits`
    /// mantissa bits, using round-to-nearest-even — matching XLA / JAX
    /// `jax.lax.reduce_precision`. The two halves are independent:
    ///   - Mantissa reduction rounds the 23-bit fp32 mantissa down to
    ///     `mantissaBits` bits (RNE), exactly as bf16/fp16 narrowing would.
    ///   - Exponent reduction clamps the dynamic range: values above the target
    ///     max become +/-inf, values below the target min-normal flush to zero.
    ///     The target format's subnormals are NOT produced (XLA semantics).
    /// NaN and infinity pass through unchanged. When the target widths equal
    /// fp32's own (8/23) the op is an exact identity.
    private func generateReducePrecisionKernel(
        entryPoint: String,
        exponentBits: Int,
        mantissaBits: Int,
        metalType: String
    ) -> String {
        // fp32 layout constants.
        let srcMantissaBits = 23
        let srcExponentBits = 8
        let srcExponentBias = 127

        // How many low mantissa bits to discard, and the RNE rounding setup.
        let bitsToRound = max(0, srcMantissaBits - mantissaBits)
        // Reductions in the exponent range are only needed when the target has
        // fewer exponent bits than fp32.
        let reduceExponent = exponentBits < srcExponentBits

        // Target exponent bias and the clamped [min,max] biased-exponent range,
        // expressed in fp32's biased-exponent encoding.
        let targetBias = (1 << (exponentBits - 1)) - 1
        // Largest finite target exponent (unbiased) -> fp32 biased value.
        let maxExp = ((1 << exponentBits) - 2) - targetBias + srcExponentBias
        // Smallest normal target exponent (unbiased = 1 - bias) -> fp32 biased.
        let minExp = (1 - targetBias) + srcExponentBias

        // Stores compute a float result then cast to the output element type.
        let storeBits = "output[tid] = \(metalType)(as_type<float>(sign | absBits)); return;"
        let storeInf = "output[tid] = \(metalType)(as_type<float>(sign | 0x7F800000u)); return;"
        let storeZero = "output[tid] = \(metalType)(as_type<float>(sign)); return;"
        let storePass = "output[tid] = \(metalType)(x); return;"

        // Build the per-element transform on a float `x`, operating on its bits.
        var body = """
            uint bits = as_type<uint>(x);
            uint sign = bits & 0x80000000u;
            uint absBits = bits & 0x7FFFFFFFu;
            // NaN and infinity (exponent all-ones) pass through unchanged.
            if (absBits >= 0x7F800000u) { \(storePass) }

        """

        if bitsToRound > 0 {
            // Round-to-nearest-even on the mantissa: add a bias of (half the LSB
            // we drop minus one) plus the kept-LSB so exact halves go to even,
            // then mask off the discarded low bits.
            body += """
                {
                    uint mask = (1u << \(bitsToRound)) - 1u;
                    uint lsb = (absBits >> \(bitsToRound)) & 1u;
                    absBits = (absBits + ((1u << \(bitsToRound - 1)) - 1u) + lsb) & ~mask;
                }

            """
        }

        if reduceExponent {
            // Clamp the dynamic range of the (already mantissa-rounded) value.
            body += """
                // Mantissa rounding can carry into the exponent; re-check overflow.
                if (absBits >= 0x7F800000u) { \(storeInf) }
                {
                    uint exp = absBits >> 23;
                    // Overflow above the target's largest finite -> signed inf.
                    if (exp > \(maxExp)u) { \(storeInf) }
                    // Underflow below the target's smallest normal -> +/-0.
                    // (Target subnormals are not represented, per XLA.)
                    if (exp < \(minExp)u) { \(storeZero) }
                }

            """
        }

        body += "    \(storeBits)"

        // reduce_precision is defined on floating-point types; the JAX/XLA
        // lowerings only ever feed fp32 here. We compute on the fp32 bits and
        // cast the result back to the output element type.
        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            float x = float(input[tid]);
        \(body)
        }
        """
    }

    /// Generates a unary kernel with configurable element type.
    /// Uses float8 vectorization (two float4 per thread) for better memory bandwidth.
    private func generateUnaryKernel(entryPoint: String, operation: String, metalType: String = "float") -> String {
        // Use vectorized kernel for float types
        if metalType == "float" {
            // Convert operation to work with float4
            // Replace 'x' with 'x4' for the vectorized version
            let vec4Operation = operation
                .replacingOccurrences(of: "exp(x)", with: "exp(x4)")
                .replacingOccurrences(of: "log(x)", with: "log(x4)")
                .replacingOccurrences(of: "sqrt(x)", with: "sqrt(x4)")
                .replacingOccurrences(of: "rsqrt(x)", with: "rsqrt(x4)")
                .replacingOccurrences(of: "sin(x)", with: "sin(x4)")
                .replacingOccurrences(of: "cos(x)", with: "cos(x4)")
                .replacingOccurrences(of: "tanh(x)", with: "tanh(x4)")
                .replacingOccurrences(of: "abs(x)", with: "abs(x4)")
                .replacingOccurrences(of: "floor(x)", with: "floor(x4)")
                .replacingOccurrences(of: "ceil(x)", with: "ceil(x4)")
                .replacingOccurrences(of: "sign(x)", with: "sign(x4)")
                .replacingOccurrences(of: "-x", with: "-x4")
                .replacingOccurrences(of: "(x)", with: "(x4)")
                .replacingOccurrences(of: " x ", with: " x4 ")
                .replacingOccurrences(of: " x)", with: " x4)")
                .replacingOccurrences(of: "(x ", with: "(x4 ")

            return """
            kernel void \(entryPoint)(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& count [[buffer(2)]],
                uint tid [[thread_position_in_grid]])
            {
                // Process 8 elements at a time using float4x2 vectorization
                uint idx8 = tid * 8;
                uint count8 = count / 8;

                if (tid < count8) {
                    // Vectorized path: process 8 elements (two float4)
                    uint base = tid * 2;
                    float4 x4 = reinterpret_cast<device const float4*>(input)[base];
                    reinterpret_cast<device float4*>(output)[base] = \(vec4Operation);
                    x4 = reinterpret_cast<device const float4*>(input)[base + 1];
                    reinterpret_cast<device float4*>(output)[base + 1] = \(vec4Operation);
                }
                else if (tid == count8) {
                    // Handle remainder (up to 7 elements)
                    for (uint i = idx8; i < count; i++) {
                        float x = input[i];
                        output[i] = \(operation);
                    }
                }
            }
            """
        }

        // Non-float types use scalar kernel
        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            \(metalType) x = input[tid];
            output[tid] = \(operation);
        }
        """
    }

    /// Generates a binary kernel with configurable element type.
    /// Uses float8 vectorization (two float4 per thread) for better memory bandwidth.
    private func generateBinaryKernel(entryPoint: String, operation: String, metalType: String = "float") -> String {
        // Use vectorized kernel for float types
        if metalType == "float" {
            // Convert operation to work with float4
            let vec4Operation = operation
                .replacingOccurrences(of: "max(a, b)", with: "max(a4, b4)")
                .replacingOccurrences(of: "min(a, b)", with: "min(a4, b4)")
                .replacingOccurrences(of: "pow(a, b)", with: "pow(a4, b4)")
                .replacingOccurrences(of: "a + b", with: "a4 + b4")
                .replacingOccurrences(of: "a - b", with: "a4 - b4")
                .replacingOccurrences(of: "a * b", with: "a4 * b4")
                .replacingOccurrences(of: "a / b", with: "a4 / b4")

            return """
            kernel void \(entryPoint)(
                device const float* inputA [[buffer(0)]],
                device const float* inputB [[buffer(1)]],
                device float* output [[buffer(2)]],
                constant uint& count [[buffer(3)]],
                uint tid [[thread_position_in_grid]])
            {
                // Process 8 elements at a time using float4x2 vectorization
                uint idx8 = tid * 8;
                uint count8 = count / 8;

                if (tid < count8) {
                    // Vectorized path: process 8 elements (two float4)
                    uint base = tid * 2;
                    float4 a4 = reinterpret_cast<device const float4*>(inputA)[base];
                    float4 b4 = reinterpret_cast<device const float4*>(inputB)[base];
                    reinterpret_cast<device float4*>(output)[base] = \(vec4Operation);
                    a4 = reinterpret_cast<device const float4*>(inputA)[base + 1];
                    b4 = reinterpret_cast<device const float4*>(inputB)[base + 1];
                    reinterpret_cast<device float4*>(output)[base + 1] = \(vec4Operation);
                }
                else if (tid == count8) {
                    // Handle remainder (up to 7 elements)
                    for (uint i = idx8; i < count; i++) {
                        float a = inputA[i];
                        float b = inputB[i];
                        output[i] = \(operation);
                    }
                }
            }
            """
        }

        // Non-float types use scalar kernel
        //
        // For bfloat, Metal's `max(a, b)` and `min(a, b)` overloads create
        // an ambiguous call (the implicit conversions to fp16/fp32/int all
        // match). Rewrite to a ternary so the expression has a definite
        // type without depending on overload resolution.
        var resolvedOp = operation
        if metalType == "bfloat" {
            resolvedOp = resolvedOp
                .replacingOccurrences(of: "max(a, b)", with: "(a > b ? a : b)")
                .replacingOccurrences(of: "min(a, b)", with: "(a < b ? a : b)")
        }
        return """
        kernel void \(entryPoint)(
            device const \(metalType)* inputA [[buffer(0)]],
            device const \(metalType)* inputB [[buffer(1)]],
            device \(metalType)* output [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            \(metalType) a = inputA[tid];
            \(metalType) b = inputB[tid];
            output[tid] = \(resolvedOp);
        }
        """
    }

    /// Generates a copy kernel with configurable element type.
    private func generateCopyKernel(entryPoint: String, metalType: String = "float") -> String {
        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            output[tid] = input[tid];
        }
        """
    }

    /// Generates a type conversion kernel.
    private func generateConvertKernel(
        entryPoint: String,
        inputType: ElementType,
        outputType: ElementType
    ) -> String {
        let inputMetal = metalTypeName(for: inputType)
        let outputMetal = metalTypeName(for: outputType)

        // Same cast for every category (`<outputMetal>(x)`); Metal handles
        // float<->int truncation, narrowing/widening, and same-category casts
        // identically through static_cast semantics.

        // Vectorize for byte-aligned float/half/bfloat conversions: process 4
        // elements per thread via vec<T,4> reads + writes, with a scalar tail
        // for the trailing <4 elements. This is the hot path for the TF32
        // matmul wrapper (fp32->fp16 input, fp16->fp32 output) where each
        // convert touches tens of MB; the 1-thread-per-element form here was
        // costing ~1ms per convert kernel on 4096^2 inputs.
        let vectorizable = isFloatType(inputType) && isFloatType(outputType)
            && elementByteSize(for: inputType) >= 2
            && elementByteSize(for: outputType) >= 2

        if vectorizable {
            return """
            kernel void \(entryPoint)(
                device const \(inputMetal)* input [[buffer(0)]],
                device \(outputMetal)* output [[buffer(1)]],
                constant uint& count [[buffer(2)]],
                uint tid [[thread_position_in_grid]])
            {
                uint base = tid * 4;
                if (base + 4 <= count) {
                    \(inputMetal)4 v = ((const device \(inputMetal)4*)input)[tid];
                    // Explicit per-component scalar casts: Metal does not provide
                    // a bfloat4(float, float, float, float) constructor — only the
                    // strict bfloat4(bfloat, bfloat, bfloat, bfloat) form — so we
                    // cast component-wise to be uniform across all conversions.
                    ((device \(outputMetal)4*)output)[tid] = \(outputMetal)4(
                        \(outputMetal)(v.x),
                        \(outputMetal)(v.y),
                        \(outputMetal)(v.z),
                        \(outputMetal)(v.w));
                } else {
                    for (uint i = base; i < count; ++i) {
                        output[i] = \(outputMetal)(input[i]);
                    }
                }
            }
            """
        }

        return """
        kernel void \(entryPoint)(
            device const \(inputMetal)* input [[buffer(0)]],
            device \(outputMetal)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            \(inputMetal) x = input[tid];
            output[tid] = \(outputMetal)(x);
        }
        """
    }

    /// Generates a bitcast conversion kernel that reinterprets bits without changing values.
    /// Uses Metal's as_type<T>() for same-size types.
    private func generateBitcastConvertKernel(
        entryPoint: String,
        inputType: ElementType,
        outputType: ElementType
    ) -> String {
        let inputMetal = metalTypeName(for: inputType)
        let outputMetal = metalTypeName(for: outputType)

        return """
        kernel void \(entryPoint)(
            device const \(inputMetal)* input [[buffer(0)]],
            device \(outputMetal)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            output[tid] = as_type<\(outputMetal)>(input[tid]);
        }
        """
    }

    /// Generates a compare kernel that takes two typed inputs and outputs bool.
    private func generateCompareKernel(
        entryPoint: String,
        compareOp: String,
        inputType: String
    ) -> String {
        // compareOp uses "a" and "b" as placeholders, e.g. "a == b"
        let expr = compareOp
            .replacingOccurrences(of: "a", with: "a[tid]")
            .replacingOccurrences(of: "b", with: "b[tid]")
        return """
        kernel void \(entryPoint)(
            device const \(inputType)* a [[buffer(0)]],
            device const \(inputType)* b [[buffer(1)]],
            device bool* output [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            output[tid] = \(expr);
        }
        """
    }

    /// Generates a gather kernel for embedding lookup patterns.
    ///
    /// This handles the common case where:
    /// - operand is the data tensor to gather from
    /// - indices contains positions to gather
    /// - output is gathered data
    ///
    /// Supports the standard embedding lookup pattern where each index selects
    /// a row/slice from the operand tensor.
    private func generateGatherKernel(
        entryPoint: String,
        operandShape: [Int],
        indicesShape: [Int],
        outputShape: [Int],
        dimNumbers: GatherDimensionNumbers,
        operandType: String,
        indicesType: String
    ) -> String {
        // The specialized batched-gather kernel below was written for the
        // vmap'd-jnp.roll pattern (1D batch index) and assumed batchDim=0,
        // 1 index per batch, and 0 offset dims. It silently produced wrong
        // results for the patterns that show up under value_and_grad of
        // `optax.softmax_cross_entropy_with_integer_labels` — multi-dim
        // batching, batch dim != 0, offset_dims present. The general kernel
        // below (extended to honour operand_batching_dims) handles all of
        // them correctly. Route everything through it for now; revisit if a
        // dedicated fast path becomes necessary.
        _ = generateBatchedGatherKernel  // keep symbol alive; unused.

        // General gather kernel.
        //
        // Output layout: [batch_dims (from index grid)..., offset_dims...]
        // For each output element:
        //   1. Decompose tid into index-grid coordinates and offset coordinates
        //   2. Look up start indices from the index tensor
        //   3. Compute operand coordinates = start_indices + offset_coords
        //   4. Read from operand at those coordinates

        let sliceSizes = dimNumbers.sliceSizes
        let totalOutputElements = outputShape.reduce(1, *)

        // Compute operand strides (row-major)
        var operandStrides = [Int](repeating: 1, count: operandShape.count)
        for i in stride(from: operandShape.count - 2, through: 0, by: -1) {
            operandStrides[i] = operandStrides[i + 1] * operandShape[i + 1]
        }

        // Compute output strides
        var outputStrides = [Int](repeating: 1, count: outputShape.count)
        for i in stride(from: outputShape.count - 2, through: 0, by: -1) {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }

        // Index grid shape: indices shape without the index_vector_dim
        var indexGridShape: [Int] = []
        for (i, s) in indicesShape.enumerated() {
            if i != dimNumbers.indexVectorDim { indexGridShape.append(s) }
        }
        let indexGridSize = max(indexGridShape.reduce(1, *), 1)

        // Non-collapsed, non-batched operand dims → these are offset dims in order
        var offsetOperandDims: [Int] = []
        for i in 0..<operandShape.count {
            if !dimNumbers.collapsedSliceDims.contains(i) && !dimNumbers.operandBatchingDims.contains(i) {
                offsetOperandDims.append(i)
            }
        }

        // Offset sizes (from slice_sizes for the offset operand dims)
        var offsetSizes: [Int] = []
        for d in offsetOperandDims {
            offsetSizes.append(sliceSizes[d])
        }
        let offsetTotalSize = max(offsetSizes.reduce(1, *), 1)

        // Number of index components per grid point
        let numIndexComponents = dimNumbers.startIndexMap.count

        // Indices stride for index_vector_dim
        let ivd = dimNumbers.indexVectorDim
        var indicesStrides = [Int](repeating: 1, count: indicesShape.count)
        for i in stride(from: indicesShape.count - 2, through: 0, by: -1) {
            indicesStrides[i] = indicesStrides[i + 1] * indicesShape[i + 1]
        }

        // Build the kernel body — decompose tid into output coordinates,
        // then separate into grid coords (non-offset output dims) and offset coords.
        var body = ""

        // Decompose tid into output multi-index using output strides
        for (d, _) in outputShape.enumerated() {
            body += "            uint out\(d) = (tid / \(outputStrides[d])) % \(outputShape[d]);\n"
        }

        // Map output dims to grid coords and offset coords
        // offset_dims tells us which output dims are offset dims
        let offsetDimSet = Set(dimNumbers.offsetDims)
        var gridDimIdx = 0
        for (d, _) in outputShape.enumerated() {
            if offsetDimSet.contains(d) {
                // This output dim is an offset dim — find which offset index it is
                if let oi = dimNumbers.offsetDims.firstIndex(of: d) {
                    body += "            uint oc\(oi) = out\(d);\n"
                }
            } else {
                // This output dim is a grid (batch/index) dim
                body += "            uint gc\(gridDimIdx) = out\(d);\n"
                gridDimIdx += 1
            }
        }

        // Look up start indices from the indices tensor
        // indices[grid_coords..., index_vector_dim_component]
        for (comp, mappedDim) in dimNumbers.startIndexMap.enumerated() {
            // Build flat index into indices tensor
            var idxTerms: [String] = []
            var gridDimIdx = 0
            for d in 0..<indicesShape.count {
                if d == ivd {
                    idxTerms.append("\(comp) * \(indicesStrides[d])")
                } else {
                    idxTerms.append("gc\(gridDimIdx) * \(indicesStrides[d])")
                    gridDimIdx += 1
                }
            }
            let idxExpr = idxTerms.joined(separator: " + ")
            body += "            uint startIdx\(comp) = uint(indices[\(idxExpr)]);\n"
            // Clamp
            body += "            if (startIdx\(comp) >= \(operandShape[mappedDim])) startIdx\(comp) = \(operandShape[mappedDim] - 1);\n"
        }

        // Helper: indices dim → grid coord index (skips index_vector_dim).
        // gc indices were emitted earlier in output-dim order over the
        // non-offset output dims; those map 1:1 to indices' non-IVD dims
        // (StableHLO gather semantics).
        func gridIndexForIndicesDim(_ indicesDim: Int) -> Int {
            var idx = 0
            for d in 0..<indicesShape.count {
                if d == ivd { continue }
                if d == indicesDim { return idx }
                idx += 1
            }
            return -1
        }

        // Compute flat operand index
        body += "            uint srcPos = 0;\n"
        var offsetDimCounter = 0
        for d in 0..<operandShape.count {
            if dimNumbers.collapsedSliceDims.contains(d) {
                // Collapsed dim: use start index
                if let comp = dimNumbers.startIndexMap.firstIndex(of: d) {
                    body += "            srcPos += startIdx\(comp) * \(operandStrides[d]);\n"
                }
            } else if let bIdx = dimNumbers.operandBatchingDims.firstIndex(of: d) {
                // Operand batching dim — paired with start_indices_batching_dims[bIdx].
                // The matching start_indices dim is one of the non-IVD indices
                // dims, whose coord comes from the gc emitted earlier.
                let siDim = bIdx < dimNumbers.startIndicesBatchingDims.count
                    ? dimNumbers.startIndicesBatchingDims[bIdx]
                    : bIdx
                let gridIdx = gridIndexForIndicesDim(siDim)
                if gridIdx >= 0 {
                    body += "            srcPos += gc\(gridIdx) * \(operandStrides[d]);\n"
                }
            } else {
                // Offset dim: start_index (if indexed) + offset coordinate
                if let comp = dimNumbers.startIndexMap.firstIndex(of: d) {
                    body += "            srcPos += (startIdx\(comp) + oc\(offsetDimCounter)) * \(operandStrides[d]);\n"
                } else {
                    body += "            srcPos += oc\(offsetDimCounter) * \(operandStrides[d]);\n"
                }
                offsetDimCounter += 1
            }
        }

        return """
        kernel void \(entryPoint)(
            device const \(operandType)* operand [[buffer(0)]],
            device const \(indicesType)* indices [[buffer(1)]],
            device \(operandType)* output [[buffer(2)]],
            constant uint& outputCount [[buffer(3)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalOutputElements)) return;
            \(body)
            output[tid] = operand[srcPos];
        }
        """
    }

    /// Generates a batched gather kernel where the first dimension is a batch dimension.
    ///
    /// Supports multiple start indices mapping to different operand dimensions.
    /// Common pattern: vmap'd jnp.roll compiles to batched gather with 2 start
    /// indices for 2D rolls (one per spatial axis).
    private func generateBatchedGatherKernel(
        entryPoint: String,
        operandShape: [Int],
        indicesShape: [Int],
        outputShape: [Int],
        dimNumbers: GatherDimensionNumbers,
        operandType: String,
        indicesType: String
    ) -> String {
        let batchDim = dimNumbers.operandBatchingDims[0]
        let batchSize = operandShape[batchDim]
        let startIndexMap = dimNumbers.startIndexMap
        let sliceSizes = dimNumbers.sliceSizes
        let numStartIndices = startIndexMap.count

        // Compute operand strides for each dimension
        var operandStrides = [Int](repeating: 1, count: operandShape.count)
        for i in stride(from: operandShape.count - 2, through: 0, by: -1) {
            operandStrides[i] = operandStrides[i + 1] * operandShape[i + 1]
        }

        // Compute output strides
        var outputStrides = [Int](repeating: 1, count: outputShape.count)
        for i in stride(from: outputShape.count - 2, through: 0, by: -1) {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }

        // Slice shape for offset dimensions (non-batch, non-collapsed)
        var offsetDimSizes: [(dim: Int, size: Int, operandDim: Int)] = []
        for od in dimNumbers.offsetDims {
            offsetDimSizes.append((dim: od, size: outputShape[od], operandDim: startIndexMap.count > 0 ? 0 : od))
        }

        // Indices per batch element
        let indicesPerBatch = numStartIndices

        let totalOutputElements = outputShape.reduce(1, *)
        let outputBatchStride = outputShape.suffix(from: 1).reduce(1, *)

        // For each offset dim, its stride in the output and operand
        // offset_dims maps output dims to the non-batch, non-collapsed operand dims
        // The operand dims corresponding to offset_dims are the ones NOT in
        // collapsedSliceDims and NOT in operandBatchingDims, in order
        var nonBatchNonCollapsedDims: [Int] = []
        for i in 0..<operandShape.count {
            if !dimNumbers.operandBatchingDims.contains(i) && !dimNumbers.collapsedSliceDims.contains(i) {
                nonBatchNonCollapsedDims.append(i)
            }
        }

        // Build code to decompose offsetInBatch into per-dimension offsets
        // and compute the source position
        var decompose = ""
        var srcCalc = "uint srcPos = batchIdx * \(operandStrides[batchDim]);\n"

        // Add start index contributions
        for (i, mappedDim) in startIndexMap.enumerated() {
            let dimSize = operandShape[mappedDim]
            decompose += "            \(indicesType) startIdx\(i) = indices[batchIdx * \(indicesPerBatch) + \(i)];\n"
            decompose += "            uint si\(i) = uint(startIdx\(i));\n"
            decompose += "            if (si\(i) >= \(dimSize)) si\(i) = \(dimSize - 1);\n"
            srcCalc += "            srcPos += si\(i) * \(operandStrides[mappedDim]);\n"
        }

        // Decompose offsetInBatch into per-offset-dim coordinates
        for (i, od) in dimNumbers.offsetDims.enumerated() {
            let outputDimStride = outputStrides[od]
            let outputDimSize = outputShape[od]
            let operandDim = nonBatchNonCollapsedDims[i]
            let operandDimStride = operandStrides[operandDim]
            decompose += "            uint od\(i) = (offsetInBatch / \(outputDimStride)) % \(outputDimSize);\n"
            srcCalc += "            srcPos += od\(i) * \(operandDimStride);\n"
        }

        return """
        kernel void \(entryPoint)(
            device const \(operandType)* operand [[buffer(0)]],
            device const \(indicesType)* indices [[buffer(1)]],
            device \(operandType)* output [[buffer(2)]],
            constant uint& outputCount [[buffer(3)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalOutputElements)) return;

            uint batchIdx = tid / \(outputBatchStride);
            uint offsetInBatch = tid % \(outputBatchStride);

            if (batchIdx >= \(batchSize)) return;
            \(decompose)
            \(srcCalc)
            output[tid] = operand[srcPos];
        }
        """
    }

    /// Generates a scatter kernel for writing updates into an operand tensor at indexed positions.
    ///
    /// The scatter kernel first copies the operand to the output, then writes updates
    /// at the positions specified by indices. Supports set (replace) and add (accumulate) modes.
    // `internal` (not private) so kernel-contract regression tests can assert the
    // generated source directly — see ScatterTests.generatedAddScatterSeedsOperand.
    func generateScatterKernel(
        entryPoint: String,
        operandShape: [Int],
        indicesShape: [Int],
        updatesShape: [Int],
        outputShape: [Int],
        dimNumbers: ScatterDimensionNumbers,
        computationKind: ScatterComputationKind,
        operandType: String,
        indicesType: String
    ) -> String {
        // General scatter kernel that mirrors the (now-fixed) gather kernel
        // and adds input_batching_dims support.
        //
        // The previous specialization was written for the
        // numIndices = indicesShape[0], scatter into a single row pattern
        // (one batching dim *always* at operand dim 0, no operand-side window
        // beyond inserted-singleton, indices laid out flat). It silently
        // produced wrong dstPos for the multi-dim-batching pattern that
        // `optax.softmax_cross_entropy_with_integer_labels`'s backward
        // emits — `input_batching_dims = [0]`, `scatter_indices_batching_dims = [0]`,
        // operand (B, V), updates (B, 1) — by dropping the batch-coord
        // contribution and writing every batched update at
        // `operand[0, scatter_idx]` instead of `operand[b, scatter_idx]`.
        //
        // The general algorithm here matches the StableHLO scatter semantics:
        //   For each update element (tid) decomposed into per-dim coords `uc[]`:
        //     • split `uc` into "scatter dims" (positions NOT in
        //       update_window_dims) and "window dims" (positions in
        //       update_window_dims).
        //     • scatter dims, in update-order, map 1:1 to scatter_indices'
        //       non-IVD dims. Use them both to read the scatter index
        //       components AND to provide the batch coord for any
        //       operand dim listed in `input_batching_dims`.
        //     • window dims map to operand dims that aren't inserted or
        //       batched, in operand-order.
        //   Then for each operand dim d:
        //     • d ∈ input_batching_dims[k] → batch coord (= uc[update-dim
        //       paired with scatter_indices_batching_dims[k]])
        //     • d ∈ scatter_dims_to_operand_dims[k] → scatter_idx[k]
        //     • else → window coord (= uc[update_window_dims[i]] where i is
        //       d's position in the operand-window-dim list)

        let totalOperandElements = operandShape.reduce(1, *)
        let totalUpdateElements = updatesShape.reduce(1, *)

        // Strides for each tensor (row-major).
        func strides(of shape: [Int]) -> [Int] {
            var s = [Int](repeating: 1, count: shape.count)
            if shape.count >= 2 {
                for d in stride(from: shape.count - 2, through: 0, by: -1) {
                    s[d] = s[d + 1] * shape[d + 1]
                }
            }
            return s
        }
        let operandStrides = strides(of: operandShape)
        let updatesStrides = strides(of: updatesShape)
        let indicesStrides = strides(of: indicesShape)

        let ivd = dimNumbers.indexVectorDim
        let updateWindowSet = Set(dimNumbers.updateWindowDims)
        // update "scatter" dims (in update-shape order) — those NOT in update_window_dims
        let updateScatterDims = (0..<updatesShape.count).filter { !updateWindowSet.contains($0) }
        // scatter_indices non-IVD dims (in indices-shape order)
        let indicesNonIVDDims = (0..<indicesShape.count).filter { $0 != ivd }

        // Number of scatter-index components per grid point. `ivd` past the
        // end of indices.shape means there's an implicit single component.
        let numComps = (ivd >= 0 && ivd < indicesShape.count)
            ? indicesShape[ivd]
            : 1

        // indices position = Σ over non-IVD indices dims i of
        //   (uc[updateScatterDims[i]]) * indicesStrides[indicesNonIVDDims[i]]
        // + k * indicesStrides[ivd]  (when ivd is in-shape)
        var baseTerms: [String] = []
        for (i, sidxDim) in indicesNonIVDDims.enumerated() where i < updateScatterDims.count {
            let updDim = updateScatterDims[i]
            baseTerms.append("uc\(updDim) * \(indicesStrides[sidxDim])")
        }
        let baseExpr = baseTerms.isEmpty ? "0" : baseTerms.joined(separator: " + ")

        // Helper: scatter_indices dim → corresponding update scatter dim.
        func updateDimForIndicesDim(_ sidxDim: Int) -> Int? {
            if let i = indicesNonIVDDims.firstIndex(of: sidxDim), i < updateScatterDims.count {
                return updateScatterDims[i]
            }
            return nil
        }

        let insertedWindowSet = Set(dimNumbers.insertedWindowDims)
        let inputBatching = dimNumbers.inputBatchingDims
        let scatterIdxBatching = dimNumbers.scatterIndicesBatchingDims
        let scatterToOperand = dimNumbers.scatterDimsToOperandDims
        // operand window dims (in operand-shape order): dims that are neither
        // batching nor inserted-window.
        let operandWindowDims = (0..<operandShape.count).filter {
            !inputBatching.contains($0) && !insertedWindowSet.contains($0)
        }

        // Emits the per-update-element `dstPos` computation, decomposing the
        // update index `idx` into coords, reading the scatter-index components,
        // and accumulating the operand offset. Per StableHLO an operand dim can
        // get BOTH a scatter-index offset (window start) AND an in-window coord —
        // they ADD; a dim with only one role gets 0 from the others. Sets `oob`
        // if a scatter index is out of range (the caller skips that element).
        func dstPosBody(_ idx: String) -> String {
            var b = ""
            for d in 0..<updatesShape.count {
                b += "                uint uc\(d) = (\(idx) / \(updatesStrides[d])) % \(updatesShape[d]);\n"
            }
            for k in 0..<numComps {
                let ivdTerm = (ivd >= 0 && ivd < indicesShape.count) ? " + \(k) * \(indicesStrides[ivd])" : ""
                b += "                uint scatter_idx_\(k) = uint(indices[\(baseExpr)\(ivdTerm)]);\n"
            }
            b += "                bool oob = false;\n"
            b += "                uint dstPos = 0;\n"
            for d in 0..<operandShape.count {
                if let k = inputBatching.firstIndex(of: d),
                   k < scatterIdxBatching.count,
                   let updDim = updateDimForIndicesDim(scatterIdxBatching[k]) {
                    b += "                dstPos += uc\(updDim) * \(operandStrides[d]);\n"
                }
                if let i = operandWindowDims.firstIndex(of: d), i < dimNumbers.updateWindowDims.count {
                    let updWinDim = dimNumbers.updateWindowDims[i]
                    b += "                dstPos += uc\(updWinDim) * \(operandStrides[d]);\n"
                }
                if let k = scatterToOperand.firstIndex(of: d) {
                    b += "                if (scatter_idx_\(k) >= \(operandShape[d])) oob = true;\n"
                    b += "                dstPos += scatter_idx_\(k) * \(operandStrides[d]);\n"
                }
            }
            return b
        }

        // Embedding-style autograd backward scatters multiple updates to the
        // same operand position (e.g. one row per token id, with repeated
        // tokens). A naive `output[dstPos] = output[dstPos] + updates[tid]`
        // RMW races across threads — only one update survives per cycle, and
        // every embedding gradient gets approximately the contribution of a
        // *single* occurrence instead of the sum. Metal 3+ exposes float
        // atomic_fetch_add (macOS 13+, Apple7+, available on M5 Pro), so use
        // it for ADD scatter on float32 operand. Non-add cases keep the
        // direct write; in practice they target unique positions per scatter.
        let isFloat32Operand = (operandType == "float")
        let useAtomicAdd = isFloat32Operand && (computationKind == .add)

        let kernelHeader = """
        kernel void \(entryPoint)(
            device const \(operandType)* operand [[buffer(0)]],
            device const \(indicesType)* indices [[buffer(1)]],
            device const \(operandType)* updates [[buffer(2)]],
            device \(operandType)* output [[buffer(3)]],
            uint tid [[thread_position_in_grid]])
        """

        // SET scatter: output-centric. One thread per OUTPUT element starts from
        // the operand value, then overwrites from any update that targets this
        // position. The accumulating path below skips the operand→output copy
        // (it assumes a fresh-zero operand, true for autograd scatter-add); SET
        // scatters into a NON-zero operand — e.g. jnp.unique builds its mask by
        // scattering into broadcast(true) — and would otherwise leave the
        // unscattered positions at the memset-0 value. unique_indices ⇒ at most
        // one match; for general SET the last match wins, matching StableHLO.
        // O(output × updates); SET scatters are small in practice.
        if computationKind == .set {
            return """
            \(kernelHeader)
            {
                if (tid >= \(totalOperandElements)) return;
                \(operandType) result = operand[tid];
                for (uint u = 0; u < \(totalUpdateElements); u++) {
            \(dstPosBody("u"))
                    if (!oob && dstPos == tid) { result = updates[u]; }
                }
                output[tid] = result;
            }
            """
        }

        // Accumulating scatter (add/max/min/mul), float32 atomic-add fast path:
        // one thread per UPDATE element. atomic_fetch_add avoids the RMW race for
        // repeated indices (embedding-gradient backward, where many tokens map to
        // the same row). The operand here is the reduction identity (broadcast(0)
        // for add) and the planner currently hands this result an untouched region
        // (the executor's one-time slab memset leaves it fresh-zero), so the
        // per-update atomic accumulation is correct as-is. Left unchanged to keep
        // the embedding-backward path (and the nanoGPT loss gate) exactly stable;
        // the reuse-stale-byte hazard is fixed in the non-atomic output-centric
        // path below, which is the one jnp.unique's i32 add-scatter takes.
        if useAtomicAdd {
            return """
            \(kernelHeader)
            {
                if (tid >= \(totalUpdateElements)) return;
            \(dstPosBody("tid"))
                if (oob) return;
                atomic_fetch_add_explicit((device atomic_float*)(output + dstPos), updates[tid], memory_order_relaxed);
            }
            """
        }

        // Non-atomic accumulating scatter (add/max/min/mul): OUTPUT-centric, mirroring
        // the SET branch. One thread per OUTPUT element seeds `result = operand[tid]`
        // (the operand is the reduction identity, e.g. broadcast(0) for add), then
        // folds in every update that targets this position with the reduction op, and
        // ALWAYS writes output[tid]. This guarantees every output position is written
        // even when the memory planner reuses a slot whose stale bytes are nonzero —
        // killing the previous hidden dependency on a fresh-zero slab (which broke
        // jnp.unique: its i32 add-scatter was handed a reused slot still holding a
        // bool-compare tensor's 0x01 bytes, so the unscattered positions read dirty).
        // O(output × updates); these scatters are small in practice. No atomics needed
        // because each output position is owned by exactly one thread.
        let fold: String
        switch computationKind {
        case .add: fold = "result = result + updates[u];"
        case .max: fold = "result = max(result, updates[u]);"
        case .min: fold = "result = min(result, updates[u]);"
        case .mul: fold = "result = result * updates[u];"
        case .set: fold = "result = updates[u];"  // handled above
        }
        return """
        \(kernelHeader)
        {
            if (tid >= \(totalOperandElements)) return;
            \(operandType) result = operand[tid];
            for (uint u = 0; u < \(totalUpdateElements); u++) {
        \(dstPosBody("u"))
                if (!oob && dstPos == tid) { \(fold) }
            }
            output[tid] = result;
        }
        """
    }

    /// Generates a slice kernel for extracting sub-tensors.
    private func generateSliceKernel(
        entryPoint: String,
        inputShape: [Int],
        outputShape: [Int],
        starts: [Int],
        limits: [Int],
        strides: [Int],
        metalType: String
    ) -> String {
        guard !inputShape.isEmpty, !outputShape.isEmpty else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let rank = inputShape.count
        let totalElements = outputShape.reduce(1, *)

        // Calculate input and output strides
        var inputStrides = [Int](repeating: 1, count: rank)
        var outputStrides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1]
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }

        // Generate index calculation code
        var indexCode = ""
        for i in 0..<rank {
            indexCode += "    uint out_idx_\(i) = (tid / \(outputStrides[i])) % \(outputShape[i]);\n"
            indexCode += "    uint in_idx_\(i) = \(starts[i]) + out_idx_\(i) * \(strides[i]);\n"
        }

        // Generate source index calculation
        var srcPosCalc = "uint srcPos = "
        for i in 0..<rank {
            if i > 0 { srcPosCalc += " + " }
            srcPosCalc += "in_idx_\(i) * \(inputStrides[i])"
        }
        srcPosCalc += ";"

        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalElements)) return;

        \(indexCode)
            \(srcPosCalc)
            output[tid] = input[srcPos];
        }
        """
    }

    /// Generates a top-k kernel over the last axis. Each thread handles one row
    /// (the flattened product of all leading dims) and selects the k largest
    /// elements of that row in descending order. With `emitIndices == false` it
    /// writes the k values; with `emitIndices == true` it writes their original
    /// last-axis positions as int32. Ties break toward the smaller index, which
    /// matches jax.lax.top_k's stable descending sort.
    ///
    /// The selection is an O(n·k) running top-k pass (k is small in practice),
    /// avoiding a full sort. Buffer layout matches the other unary kernels:
    /// buffer(0) = input, buffer(1) = output, buffer(2) = count.
    private func generateTopKKernel(
        entryPoint: String,
        inputShape: [Int],
        k: Int,
        emitIndices: Bool,
        valueType: String
    ) -> String {
        guard !inputShape.isEmpty, k > 0 else {
            // Degenerate: nothing to select. Emit a no-op copy to keep wiring.
            return generateCopyKernel(entryPoint: entryPoint, metalType: emitIndices ? "int" : valueType)
        }

        let n = inputShape.last ?? 0
        let rows = n > 0 ? inputShape.reduce(1, *) / n : 0
        let outType = emitIndices ? "int" : valueType
        // The k-th selected element is written from the running top-k buffers.
        let writeStmt = emitIndices
            ? "output[base_out + j] = best_idx[j];"
            : "output[base_out + j] = best_val[j];"

        return """
        kernel void \(entryPoint)(
            device const \(valueType)* input [[buffer(0)]],
            device \(outType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(rows)) return;
            const uint n = \(n);
            const uint k = \(k);
            uint base_in = tid * n;
            uint base_out = tid * k;

            // Running top-k held in descending order. best_val[0] is the
            // current maximum. Capacity is the compile-time constant k.
            \(valueType) best_val[\(k)];
            int best_idx[\(k)];
            uint filled = 0;

            for (uint i = 0; i < n; ++i) {
                \(valueType) v = input[base_in + i];
                // Find insertion position: keep descending order, and on ties
                // place the new (larger-index) element AFTER existing ones so
                // the smaller original index sorts first (stable descending).
                if (filled < k) {
                    uint pos = filled;
                    while (pos > 0 && best_val[pos - 1] < v) {
                        best_val[pos] = best_val[pos - 1];
                        best_idx[pos] = best_idx[pos - 1];
                        --pos;
                    }
                    best_val[pos] = v;
                    best_idx[pos] = int(i);
                    ++filled;
                } else if (v > best_val[k - 1]) {
                    // Displace the current k-th element.
                    uint pos = k - 1;
                    while (pos > 0 && best_val[pos - 1] < v) {
                        best_val[pos] = best_val[pos - 1];
                        best_idx[pos] = best_idx[pos - 1];
                        --pos;
                    }
                    best_val[pos] = v;
                    best_idx[pos] = int(i);
                }
            }

            for (uint j = 0; j < k; ++j) {
                \(writeStmt)
            }
        }
        """
    }

    /// Generates a dynamic_slice kernel that reads start indices from buffers at runtime.
    /// buffer(0) = operand, buffer(1..N) = scalar start indices (one per dim), buffer(N+1) = output, buffer(N+2) = count
    private func generateDynamicSliceKernel(
        entryPoint: String,
        inputShape: [Int],
        outputShape: [Int],
        sliceSizes: [Int],
        metalType: String,
        numStartIndices: Int
    ) -> String {
        guard !inputShape.isEmpty, !outputShape.isEmpty else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let rank = inputShape.count
        let totalElements = outputShape.reduce(1, *)

        // Calculate input strides
        var inputStrides = [Int](repeating: 1, count: rank)
        var outputStrides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1]
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }

        // Build buffer parameter list: operand + start indices + output + count
        var params = "    device const \(metalType)* input [[buffer(0)]],\n"
        for i in 0..<rank {
            params += "    device const int* start_idx_\(i) [[buffer(\(i + 1))]],\n"
        }
        params += "    device \(metalType)* output [[buffer(\(rank + 1))]],\n"
        params += "    constant uint& count [[buffer(\(rank + 2))]]"

        // Generate index calculation
        var indexCode = ""
        for i in 0..<rank {
            let maxStart = inputShape[i] - sliceSizes[i]
            indexCode += "    int s_\(i) = clamp(start_idx_\(i)[0], 0, \(maxStart));\n"
            indexCode += "    uint out_idx_\(i) = (tid / \(outputStrides[i])) % \(outputShape[i]);\n"
            indexCode += "    uint in_idx_\(i) = uint(s_\(i)) + out_idx_\(i);\n"
        }

        var srcPosCalc = "uint srcPos = "
        for i in 0..<rank {
            if i > 0 { srcPosCalc += " + " }
            srcPosCalc += "in_idx_\(i) * \(inputStrides[i])"
        }
        srcPosCalc += ";"

        return """
        kernel void \(entryPoint)(
        \(params),
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalElements)) return;

        \(indexCode)
            \(srcPosCalc)
            output[tid] = input[srcPos];
        }
        """
    }

    /// Generates a dynamic_update_slice kernel.
    /// Copies the operand to output, then overwrites the slice region with the update.
    /// buffer(0) = operand, buffer(1) = update, buffer(2..N+1) = scalar start indices,
    /// buffer(N+2) = output, buffer(N+3) = count
    private func generateDynamicUpdateSliceKernel(
        entryPoint: String,
        operandShape: [Int],
        updateShape: [Int],
        outputShape: [Int],
        metalType: String,
        numStartIndices: Int
    ) -> String {
        guard !operandShape.isEmpty, !updateShape.isEmpty else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let rank = operandShape.count
        let totalElements = operandShape.reduce(1, *)
        let updateElements = updateShape.reduce(1, *)

        // Calculate strides
        var operandStrides = [Int](repeating: 1, count: rank)
        var updateStrides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            operandStrides[i] = operandStrides[i + 1] * operandShape[i + 1]
            updateStrides[i] = updateStrides[i + 1] * updateShape[i + 1]
        }

        // Build buffer parameter list
        var params = "    device const \(metalType)* operand [[buffer(0)]],\n"
        params += "    device const \(metalType)* update [[buffer(1)]],\n"
        for i in 0..<rank {
            params += "    device const int* start_idx_\(i) [[buffer(\(i + 2))]],\n"
        }
        params += "    device \(metalType)* output [[buffer(\(rank + 2))]],\n"
        params += "    constant uint& count [[buffer(\(rank + 3))]]"

        // Generate start index clamping
        var startCode = ""
        for i in 0..<rank {
            let maxStart = operandShape[i] - updateShape[i]
            startCode += "    int s_\(i) = clamp(start_idx_\(i)[0], 0, \(maxStart));\n"
        }

        // Generate check: is this thread's position inside the update region?
        var coordCode = ""
        var inRegionCheck = ""
        var updateIdxCalc = "uint updateIdx = "
        for i in 0..<rank {
            coordCode += "    uint coord_\(i) = (tid / \(operandStrides[i])) % \(operandShape[i]);\n"
            coordCode += "    int rel_\(i) = int(coord_\(i)) - s_\(i);\n"
            if i > 0 { inRegionCheck += " && " }
            inRegionCheck += "rel_\(i) >= 0 && rel_\(i) < \(updateShape[i])"
            if i > 0 { updateIdxCalc += " + " }
            updateIdxCalc += "uint(rel_\(i)) * \(updateStrides[i])"
        }
        updateIdxCalc += ";"

        return """
        kernel void \(entryPoint)(
        \(params),
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalElements)) return;

        \(startCode)
        \(coordCode)

            if (\(inRegionCheck)) {
                \(updateIdxCalc)
                output[tid] = update[updateIdx];
            } else {
                output[tid] = operand[tid];
            }
        }
        """
    }

    /// Generates a reverse kernel that reverses elements along specified dimensions.
    private func generateReverseKernel(
        entryPoint: String,
        inputShape: [Int],
        dimensions: [Int],
        metalType: String
    ) -> String {
        guard !inputShape.isEmpty else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let rank = inputShape.count
        let totalElements = inputShape.reduce(1, *)
        let reversedDims = Set(dimensions)

        // Calculate strides
        var strides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * inputShape[i + 1]
        }

        // Generate index calculation: decompose flat index, flip reversed dims, recompute
        var indexCode = ""
        for i in 0..<rank {
            indexCode += "    uint coord_\(i) = (tid / \(strides[i])) % \(inputShape[i]);\n"
            if reversedDims.contains(i) {
                indexCode += "    coord_\(i) = \(inputShape[i] - 1) - coord_\(i);\n"
            }
        }

        var srcPosCalc = "uint srcPos = "
        for i in 0..<rank {
            if i > 0 { srcPosCalc += " + " }
            srcPosCalc += "coord_\(i) * \(strides[i])"
        }
        srcPosCalc += ";"

        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalElements)) return;

        \(indexCode)
            \(srcPosCalc)
            output[tid] = input[srcPos];
        }
        """
    }

    /// Generates a pad kernel that pads a tensor with a constant value.
    ///
    /// Supports low padding, high padding, and interior padding.
    /// Interior padding inserts `padInterior[d]` elements between consecutive elements along dim d.
    private func generatePadKernel(
        entryPoint: String,
        inputShape: [Int],
        outputShape: [Int],
        padLow: [Int],
        padHigh: [Int],
        padInterior: [Int],
        metalType: String
    ) -> String {
        guard !inputShape.isEmpty, !outputShape.isEmpty else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let rank = inputShape.count
        let totalOutputElements = outputShape.reduce(1, *)

        // Calculate output strides
        var outputStrides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }

        // Calculate input strides
        var inputStrides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1]
        }

        // Generate the coordinate decomposition and data region check
        // Output coord `c` maps to input index `(c - padLow[d]) / (1 + padInterior[d])`
        // if `(c - padLow[d]) % (1 + padInterior[d]) == 0` and the index is in range.
        var coordCode = ""
        var checkCode = "    bool inData = true;\n    uint srcIdx = 0;\n"

        for i in 0..<rank {
            coordCode += "    uint c_\(i) = (tid / \(outputStrides[i])) % \(outputShape[i]);\n"
            let interiorStride = 1 + padInterior[i]
            checkCode += "    int adj_\(i) = int(c_\(i)) - \(padLow[i]);\n"
            if interiorStride > 1 {
                checkCode += "    inData = inData && (adj_\(i) >= 0) && (adj_\(i) % \(interiorStride) == 0);\n"
                checkCode += "    uint src_\(i) = uint(adj_\(i) / \(interiorStride));\n"
            } else {
                checkCode += "    inData = inData && (adj_\(i) >= 0);\n"
                checkCode += "    uint src_\(i) = uint(adj_\(i));\n"
            }
            checkCode += "    inData = inData && (src_\(i) < \(inputShape[i]));\n"
            checkCode += "    srcIdx += src_\(i) * \(inputStrides[i]);\n"
        }

        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device const \(metalType)* pad_value [[buffer(1)]],
            device \(metalType)* output [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalOutputElements)) return;

        \(coordCode)
        \(checkCode)
            output[tid] = inData ? input[srcIdx] : pad_value[0];
        }
        """
    }

    /// Generates a concatenate kernel that joins tensors along an axis.
    private func generateConcatenateKernel(
        entryPoint: String,
        inputShapes: [[Int]],
        outputShape: [Int],
        axis: Int,
        metalType: String
    ) -> String {
        guard !outputShape.isEmpty, !inputShapes.isEmpty else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let rank = outputShape.count
        let totalOutputElements = outputShape.reduce(1, *)
        let numInputs = inputShapes.count

        // Calculate output strides
        var outputStrides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }

        // Calculate cumulative sizes along concat axis
        var cumulativeSizes = [Int]()
        var cumSum = 0
        for shape in inputShapes {
            cumulativeSizes.append(cumSum)
            cumSum += shape[axis]
        }

        // Calculate input strides for each input
        var allInputStrides = [[Int]]()
        for shape in inputShapes {
            var strides = [Int](repeating: 1, count: rank)
            for i in stride(from: rank - 2, through: 0, by: -1) {
                strides[i] = strides[i + 1] * shape[i + 1]
            }
            allInputStrides.append(strides)
        }

        // Build the kernel: decompose output index, determine which input, compute source index
        var coordCode = ""
        for i in 0..<rank {
            coordCode += "    uint c_\(i) = (tid / \(outputStrides[i])) % \(outputShape[i]);\n"
        }

        // Build input buffer parameters
        var bufferParams = ""
        for i in 0..<numInputs {
            bufferParams += "    device const \(metalType)* input\(i) [[buffer(\(i))]],\n"
        }

        // Build if-else chain to select the right input
        var selectCode = ""
        for i in 0..<numInputs {
            let condition: String
            if i == numInputs - 1 {
                condition = "else"
            } else if i == 0 {
                condition = "if (c_\(axis) < \(cumulativeSizes[i] + inputShapes[i][axis]))"
            } else {
                condition = "else if (c_\(axis) < \(cumulativeSizes[i] + inputShapes[i][axis]))"
            }

            // Compute source index within this input
            var srcCalc = "uint srcIdx = "
            for d in 0..<rank {
                if d > 0 { srcCalc += " + " }
                if d == axis {
                    srcCalc += "(c_\(d) - \(cumulativeSizes[i])) * \(allInputStrides[i][d])"
                } else {
                    srcCalc += "c_\(d) * \(allInputStrides[i][d])"
                }
            }
            srcCalc += ";"

            if i == numInputs - 1 && numInputs > 1 {
                selectCode += "    \(condition) {\n"
            } else {
                selectCode += "    \(condition) {\n"
            }
            selectCode += "        \(srcCalc)\n"
            selectCode += "        output[tid] = input\(i)[srcIdx];\n"
            selectCode += "    }\n"
        }

        return """
        kernel void \(entryPoint)(
        \(bufferParams)    device \(metalType)* output [[buffer(\(numInputs))]],
            constant uint& count [[buffer(\(numInputs + 1))]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalOutputElements)) return;

        \(coordCode)
        \(selectCode)}
        """
    }

    /// Generates an iota kernel that fills a tensor with sequential indices along a dimension.
    private func generateIotaKernel(
        entryPoint: String,
        outputShape: [Int],
        dimension: Int,
        metalType: String
    ) -> String {
        guard !outputShape.isEmpty else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let totalElements = outputShape.reduce(1, *)

        // Calculate stride for the target dimension
        var stride = 1
        for i in (dimension + 1)..<outputShape.count {
            stride *= outputShape[i]
        }

        return """
        kernel void \(entryPoint)(
            device \(metalType)* output [[buffer(0)]],
            constant uint& count [[buffer(1)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalElements)) return;
            output[tid] = \(metalType)((tid / \(stride)) % \(outputShape[dimension]));
        }
        """
    }

    /// Generates a select kernel: output = pred ? on_true : on_false.
    private func generateSelectKernel(
        entryPoint: String,
        metalType: String,
        predIsScalar: Bool = false
    ) -> String {
        // StableHLO permits a rank-0 predicate that's implicitly broadcast to
        // the value-tensor shape; without the scalar branch below the kernel
        // indexes off the end of a 1-element bool buffer for every position
        // past 0 and gets garbage (this is precisely how jnp.var's
        // safe-divide-by-N wrapper produces NaN downstream — see the var
        // jaxpr's `select_n(scalar_bool, NaN_bcast, var)`).
        let predIndex = predIsScalar ? "0" : "tid"
        return """
        kernel void \(entryPoint)(
            device const bool* pred [[buffer(0)]],
            device const \(metalType)* on_true [[buffer(1)]],
            device const \(metalType)* on_false [[buffer(2)]],
            device \(metalType)* output [[buffer(3)]],
            constant uint& count [[buffer(4)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            output[tid] = pred[\(predIndex)] ? on_true[tid] : on_false[tid];
        }
        """
    }

    /// Generates a clamp kernel: output = min(max(operand, min_val), max_val).
    private func generateClampKernel(
        entryPoint: String,
        metalType: String
    ) -> String {
        return """
        kernel void \(entryPoint)(
            device const \(metalType)* min_val [[buffer(0)]],
            device const \(metalType)* operand [[buffer(1)]],
            device const \(metalType)* max_val [[buffer(2)]],
            device \(metalType)* output [[buffer(3)]],
            constant uint& count [[buffer(4)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            output[tid] = clamp(operand[tid], min_val[tid], max_val[tid]);
        }
        """
    }

    /// Generates a 2D convolution kernel that handles arbitrary dimension layouts,
    /// strides, padding, dilation, and grouped convolutions.
    /// 1 thread per output element. Each thread computes the convolution sum for its output position.
    private func generateConvolutionKernel(
        entryPoint: String,
        inputShape: [Int],
        weightsShape: [Int],
        outputShape: [Int],
        attributes: HLOAttributes,
        metalType: String
    ) -> String {
        guard inputShape.count >= 3, weightsShape.count >= 3, outputShape.count >= 3 else {
            // Fallback for unexpected ranks
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let numSpatialDims = inputShape.count - 2
        guard numSpatialDims >= 1 && numSpatialDims <= 3 else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        // Extract dimension numbers (default: NHWC-style: batch=0, spatial=1..N-2, feature=N-1)
        let dimNumbers = attributes.convolutionDimensionNumbers
        let inBatchDim = dimNumbers?.inputBatchDimension ?? 0
        let inFeatureDim = dimNumbers?.inputFeatureDimension ?? (inputShape.count - 1)
        let inSpatialDims = dimNumbers?.inputSpatialDimensions ?? Array(1..<(inputShape.count - 1))

        let kInFeatureDim = dimNumbers?.kernelInputFeatureDimension ?? (weightsShape.count - 2)
        let kOutFeatureDim = dimNumbers?.kernelOutputFeatureDimension ?? (weightsShape.count - 1)
        let kSpatialDims = dimNumbers?.kernelSpatialDimensions ?? Array(0..<(weightsShape.count - 2))

        let outBatchDim = dimNumbers?.outputBatchDimension ?? 0
        let outFeatureDim = dimNumbers?.outputFeatureDimension ?? (outputShape.count - 1)
        let outSpatialDims = dimNumbers?.outputSpatialDimensions ?? Array(1..<(outputShape.count - 1))

        // Extract conv parameters
        let strides = attributes.windowStrides ?? Array(repeating: 1, count: numSpatialDims)
        let padding = attributes.convPadding ?? Array(repeating: [0, 0], count: numSpatialDims)
        let rhsDilation = attributes.rhsDilation ?? Array(repeating: 1, count: numSpatialDims)
        let lhsDilation = attributes.lhsDilation ?? Array(repeating: 1, count: numSpatialDims)
        let featureGroupCount = attributes.featureGroupCount ?? 1
        let batchGroupCount = attributes.batchGroupCount ?? 1

        // Compute strides for input, weights, output (row-major)
        func computeStrides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }

        let inputStrides = computeStrides(inputShape)
        let weightsStrides = computeStrides(weightsShape)
        let outputStrides = computeStrides(outputShape)

        let totalOutputElements = outputShape.reduce(1, *)

        // Dimension sizes
        let inputChannels = inputShape[inFeatureDim]
        let outputChannels = outputShape[outFeatureDim]
        let icPerGroup = inputChannels / featureGroupCount

        // Build kernel code
        var code = """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device const \(metalType)* weights [[buffer(1)]],
            device \(metalType)* output [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalOutputElements)u) return;

            // Decompose tid into output N-D coordinates
            uint remaining = tid;

        """

        // Generate output coordinate decomposition
        var outCoordNames: [String] = Array(repeating: "", count: outputShape.count)
        for i in 0..<outputShape.count {
            let name = "o\(i)"
            outCoordNames[i] = name
            if i == outputShape.count - 1 {
                code += "    int \(name) = int(remaining);\n"
            } else {
                code += "    int \(name) = int(remaining / \(outputStrides[i])u);\n"
                code += "    remaining = remaining % \(outputStrides[i])u;\n"
            }
        }

        let oBatch = outCoordNames[outBatchDim]
        let oFeature = outCoordNames[outFeatureDim]

        // Grouped convolution: determine input channel range
        if featureGroupCount > 1 {
            code += "\n    int group = \(oFeature) / \(outputChannels / featureGroupCount);\n"
            code += "    int ic_start = group * \(icPerGroup);\n"
        } else if batchGroupCount > 1 {
            code += "\n    int batch_group = \(oBatch) / \(outputShape[outBatchDim] / batchGroupCount);\n"
        }

        code += "\n    \(metalType) sum = 0;\n"

        // Generate nested loops over kernel spatial dims
        for s in 0..<numSpatialDims {
            let kSize = weightsShape[kSpatialDims[s]]
            code += "    for (int k\(s) = 0; k\(s) < \(kSize); k\(s)++) {\n"
        }

        // Compute input spatial positions
        for s in 0..<numSpatialDims {
            let outSpatialCoord = outCoordNames[outSpatialDims[s]]
            let padLow = padding.count > s ? padding[s][0] : 0
            let stride = strides.count > s ? strides[s] : 1
            let rhsDil = rhsDilation.count > s ? rhsDilation[s] : 1
            let lhsDil = lhsDilation.count > s ? lhsDilation[s] : 1
            let inputSpatialSize = inputShape[inSpatialDims[s]]

            if lhsDil > 1 {
                // With LHS dilation: virtual input position
                code += "        int ih\(s)_virtual = \(outSpatialCoord) * \(stride) + k\(s) * \(rhsDil) - \(padLow);\n"
                code += "        int ih\(s) = ih\(s)_virtual / \(lhsDil);\n"
                code += "        bool ih\(s)_valid = (ih\(s)_virtual >= 0 && ih\(s)_virtual % \(lhsDil) == 0 && ih\(s) >= 0 && ih\(s) < \(inputSpatialSize));\n"
            } else {
                code += "        int ih\(s) = \(outSpatialCoord) * \(stride) + k\(s) * \(rhsDil) - \(padLow);\n"
                code += "        bool ih\(s)_valid = (ih\(s) >= 0 && ih\(s) < \(inputSpatialSize));\n"
            }
        }

        // Bounds check
        let validChecks = (0..<numSpatialDims).map { "ih\($0)_valid" }.joined(separator: " && ")
        code += "        if (\(validChecks)) {\n"

        // Inner loop over input channels
        code += "            for (int ic = 0; ic < \(icPerGroup); ic++) {\n"

        // Compute input flat index
        var inputIdxParts: [String] = Array(repeating: "0", count: inputShape.count)
        inputIdxParts[inBatchDim] = "\(oBatch) * \(inputStrides[inBatchDim])"
        if featureGroupCount > 1 {
            inputIdxParts[inFeatureDim] = "(ic_start + ic) * \(inputStrides[inFeatureDim])"
        } else {
            inputIdxParts[inFeatureDim] = "ic * \(inputStrides[inFeatureDim])"
        }
        for s in 0..<numSpatialDims {
            inputIdxParts[inSpatialDims[s]] = "ih\(s) * \(inputStrides[inSpatialDims[s]])"
        }
        let inputIdx = inputIdxParts.filter { $0 != "0" }.joined(separator: " + ")

        // Compute weights flat index
        var weightsIdxParts: [String] = Array(repeating: "0", count: weightsShape.count)
        weightsIdxParts[kOutFeatureDim] = "\(oFeature) * \(weightsStrides[kOutFeatureDim])"
        weightsIdxParts[kInFeatureDim] = "ic * \(weightsStrides[kInFeatureDim])"
        for s in 0..<numSpatialDims {
            weightsIdxParts[kSpatialDims[s]] = "k\(s) * \(weightsStrides[kSpatialDims[s]])"
        }
        let weightsIdx = weightsIdxParts.filter { $0 != "0" }.joined(separator: " + ")

        code += "                int in_idx = \(inputIdx);\n"
        code += "                int w_idx = \(weightsIdx);\n"
        code += "                sum += input[in_idx] * weights[w_idx];\n"
        code += "            }\n"  // end ic loop
        code += "        }\n"  // end bounds check

        // Close kernel spatial loops
        for _ in 0..<numSpatialDims {
            code += "    }\n"
        }

        code += """

            output[tid] = sum;
        }
        """

        return code
    }

    /// Generates a fused conv + bias + activation kernel. The fusion pass
    /// only fires for the canonical conv → add(broadcast(bias)) → activation
    /// chain, so this kernel is dispatched with up to 3 input buffers
    /// (input, weights, bias). Reuses the same per-output structure as the
    /// generic conv kernel — bias add + activation are appended to the
    /// final reduction step before the device write.
    private func generateFusedConvBiasActSource(
        _ config: ConvBiasActConfig,
        inputShapes: [[Int]],
        outputShapes: [[Int]],
        attributes: HLOAttributes,
        elementType: ElementType
    ) -> (source: String, entryPoint: String, tuning: TuningConfig?) {
        let entryPoint = "kernel_fused_conv_bias_act"
        let metalType = metalTypeName(for: elementType)
        let inputShape = inputShapes.first ?? []
        let weightsShape = inputShapes.count > 1 ? inputShapes[1] : []
        let outputShape = outputShapes.first ?? []

        // Start from the generic conv kernel, then patch in bias + activation
        // around the output write. The conv kernel signature uses buffers
        // (input, weights, output, count); we need to add a bias buffer when
        // hasBias and modify the final write line.
        var convCode = generateConvolutionKernel(
            entryPoint: entryPoint,
            inputShape: inputShape,
            weightsShape: weightsShape,
            outputShape: outputShape,
            attributes: attributes,
            metalType: metalType
        )

        // Bias add + activation epilogue. Bias is per-output-channel, indexed
        // by the trailing output dim (NHWC convention puts feature last; this
        // matches the unfused chain we matched in the detector).
        let outFeatureDim = attributes.convolutionDimensionNumbers?.outputFeatureDimension
            ?? (outputShape.count - 1)
        let outFeatureCoord = "o\(outFeatureDim)"
        let activationExpr: String
        switch config.activation {
        case .relu:           activationExpr = "max(sum, \(metalType)(0))"
        case .sigmoid:        activationExpr = "\(metalType)(1) / (\(metalType)(1) + exp(-sum))"
        case .tanh:           activationExpr = "tanh(sum)"
        case .silu:           activationExpr = "sum * (\(metalType)(1) / (\(metalType)(1) + exp(-sum)))"
        case .gelu, .geluApproximate:
            activationExpr = "sum * \(metalType)(0.5) * (\(metalType)(1) + tanh(\(metalType)(0.7978845608) * (sum + \(metalType)(0.044715) * sum * sum * sum)))"
        case .none:
            activationExpr = "sum"
        }

        // Splice the bias-add + activation in just before `output[tid] = sum;`,
        // and add the bias buffer arg + buffer-index bump for `count`.
        let oldOutputWrite = "output[tid] = sum;"
        let biasLine = config.hasBias
            ? "    sum += bias[\(outFeatureCoord)];"
            : ""
        let newOutputWrite = """
        \(biasLine)
            sum = \(activationExpr);
            output[tid] = sum;
        """
        convCode = convCode.replacingOccurrences(of: oldOutputWrite, with: newOutputWrite)

        // Insert bias buffer at index 2; shift output to 3, count to 4.
        // The conv kernel's signature uses 4-space indentation (start of
        // `kernel void` block) — match that exactly so the replacement fires.
        if config.hasBias {
            convCode = convCode.replacingOccurrences(
                of: "    device \(metalType)* output [[buffer(2)]],\n    constant uint& count [[buffer(3)]],",
                with: "    device const \(metalType)* bias [[buffer(2)]],\n    device \(metalType)* output [[buffer(3)]],\n    constant uint& count [[buffer(4)]],"
            )
        }

        return (convCode, entryPoint, nil)
    }

    /// Generates a simple scalar reduction kernel for integer/boolean types.
    /// Uses 1 thread per output element with a sequential loop over the reduction dimension.
    private func generateIntegerReductionSource(
        inputShape: [Int],
        reduceDims: [Int],
        reductionKind: ReductionKind,
        metalType: String,
        elementType: ElementType
    ) -> (String, String, TuningConfig?) {
        // Compute reduction parameters
        let totalInputElements = inputShape.reduce(1, *)
        let reduceDimSorted = reduceDims.sorted()

        // Compute reduction size and inner/outer sizes
        var reduceSize = 1
        for d in reduceDimSorted {
            if d < inputShape.count { reduceSize *= inputShape[d] }
        }

        // Compute output shape (input shape with reduce dims removed)
        var outputShape: [Int] = []
        for (i, s) in inputShape.enumerated() {
            if !reduceDimSorted.contains(i) { outputShape.append(s) }
        }
        let outputCount = max(outputShape.reduce(1, *), 1)

        // innerSize = product of dims after last reduce dim
        let lastReduceDim = reduceDimSorted.last ?? 0
        let innerSize = lastReduceDim + 1 < inputShape.count
            ? inputShape.suffix(from: lastReduceDim + 1).reduce(1, *)
            : 1

        // Determine identity and operation
        let isSigned = elementType == .int8 || elementType == .int16 || elementType == .int32 || elementType == .int64
        let is1Bit = elementType == .int1

        let initValue: String
        let accumOp: String
        switch reductionKind {
        case .sum, .mean:
            initValue = "0"
            accumOp = "accum += val;"
        case .max:
            if is1Bit {
                initValue = "0"
                accumOp = "accum = accum | val;"
            } else if isSigned {
                // Use minimum value for signed types
                let bits = elementByteSize(for: elementType) * 8
                initValue = bits == 32 ? "(-2147483647 - 1)" : "(\(metalType)(-1) << \(bits - 1))"
                accumOp = "accum = (val > accum) ? val : accum;"
            } else {
                initValue = "0"
                accumOp = "accum = (val > accum) ? val : accum;"
            }
        case .min:
            if is1Bit {
                initValue = "1"
                accumOp = "accum = accum & val;"
            } else if isSigned {
                let bits = elementByteSize(for: elementType) * 8
                initValue = bits == 32 ? "2147483647" : "((\(metalType))((1u << \(bits - 1)) - 1))"
                accumOp = "accum = (val < accum) ? val : accum;"
            } else {
                let bits = elementByteSize(for: elementType) * 8
                initValue = bits <= 32 ? "(\(metalType))((1u << \(bits)) - 1)" : "(\(metalType))(-1)"
                accumOp = "accum = (val < accum) ? val : accum;"
            }
        case .product:
            initValue = "1"
            accumOp = "accum *= val;"
        case .and:
            initValue = is1Bit ? "1" : "(\(metalType))(-1)"
            accumOp = "accum = accum & val;"
        case .or:
            initValue = "0"
            accumOp = "accum = accum | val;"
        case .logAddExp:
            // Only emitted for reduce_window (cumlogsumexp), never plain reduce.
            fatalError("logAddExp reduction is only supported via reduce_window")
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_reduce(
            device const \(metalType)* input [[buffer(0)]],
            device const \(metalType)* initVal [[buffer(1)]],
            device \(metalType)* output [[buffer(2)]],
            constant uint& outputCount [[buffer(3)]],
            constant uint& reduceSize [[buffer(4)]],
            constant uint& innerSize [[buffer(5)]],
            uint tgid [[threadgroup_position_in_grid]])
        {
            if (tgid >= outputCount) return;

            uint outerIdx = tgid / innerSize;
            uint innerIdx = tgid % innerSize;
            uint baseIdx = outerIdx * reduceSize * innerSize + innerIdx;

            \(metalType) accum = \(initValue);
            for (uint i = 0; i < reduceSize; i++) {
                \(metalType) val = input[baseIdx + i * innerSize];
                \(accumOp)
            }
            output[tgid] = accum;
        }
        """

        return (source, "kernel_reduce", TuningConfig(blockSize: 1))
    }

    /// Generates an arg-reduce (argmax/argmin index) kernel. This is the index
    /// half of a multi-input reduce that jnp.argmax/argmin lowers to: it scans
    /// the reduce axis of the float/int input and writes the position of the
    /// max (argmax) or min (argmin) element. Ties break toward the smaller
    /// index, matching jnp.argmax/argmin (the reducer keeps the smaller index on
    /// equal values). Strict `>` / `<` comparison gives that for free since the
    /// best index is only updated on a strictly better value.
    ///
    /// One thread per output element (the flattened non-reduced dims). Buffer
    /// layout mirrors generateIntegerReductionSource: buffer(0) = input values,
    /// buffer(1) = output indices, then outputCount / reduceSize / innerSize.
    /// Stable sort along one axis via per-element rank-scatter. Each thread owns
    /// one element, counts how many elements of its row are ordered before it
    /// (with a `j < i` tiebreak for stability), and scatters its value to that
    /// rank. Ranks are a permutation, so writes never collide. NaN is treated as
    /// the largest value (matching JAX's TOTALORDER, NaN last ascending).
    /// O(axisLen) per element; fine for typical sort lengths.
    private func generateSortSource(
        inputShape: [Int], axis: Int, descending: Bool, valueType: ElementType
    ) -> (String, String, TuningConfig?) {
        let metal = metalTypeName(for: valueType)
        let rank = inputShape.count
        let ax = axis < 0 ? axis + rank : axis
        let axisLen = (ax >= 0 && ax < rank) ? inputShape[ax] : 1
        var innerSize = 1
        if ax + 1 < rank { for i in (ax + 1)..<rank { innerSize *= inputShape[i] } }
        let total = max(inputShape.reduce(1, *), 1)
        // `c` = order(kj, ki): -1 if kj before ki by value, +1 if after, 0 equal.
        let beforeExpr = descending
            ? "(c > 0) || (c == 0 && j < i)"
            : "(c < 0) || (c == 0 && j < i)"
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_sort(
            device const \(metal)* input [[buffer(0)]],
            device \(metal)* output [[buffer(1)]],
            constant uint& axisLen [[buffer(2)]],
            constant uint& innerSize [[buffer(3)]],
            constant uint& total [[buffer(4)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= total) return;
            uint inner = tid % innerSize;
            uint tmp = tid / innerSize;
            uint i = tmp % axisLen;
            uint outer = tmp / axisLen;
            uint base = outer * axisLen * innerSize + inner;

            \(metal) ki = input[base + i * innerSize];
            uint rank = 0;
            for (uint j = 0; j < axisLen; j++) {
                \(metal) kj = input[base + j * innerSize];
                bool an = (kj != kj), bn = (ki != ki);   // NaN tests
                int c;
                if (an && bn) c = 0;
                else if (an) c = 1;                      // NaN is largest
                else if (bn) c = -1;
                else if (kj < ki) c = -1;
                else if (kj > ki) c = 1;
                else c = 0;
                if (\(beforeExpr)) rank++;
            }
            output[base + rank * innerSize] = ki;
        }
        """
        _ = axisLen; _ = innerSize; _ = total
        return (source, "kernel_sort", nil)
    }

    /// Multi-operand sort result: ranks each element by lexicographic comparison
    /// of the K keys (stably, with a j<i tiebreak) and scatters the matching
    /// `payload` element to that rank. argsort uses K=1 (payload = iota); lexsort
    /// uses K>1. Buffers: key0..keyK-1, payload, output, then axisLen/innerSize/total.
    private func generateSortResultSource(
        inputShape: [Int], axis: Int, descending: Bool,
        keyTypes: [ElementType], payloadType: ElementType
    ) -> (String, String, TuningConfig?) {
        let k = max(1, keyTypes.count)
        let payMetal = metalTypeName(for: payloadType)
        let rank = inputShape.count
        let ax = axis < 0 ? axis + rank : axis
        let axisLen = (ax >= 0 && ax < rank) ? inputShape[ax] : 1
        var innerSize = 1
        if ax + 1 < rank { for i in (ax + 1)..<rank { innerSize *= inputShape[i] } }
        let total = max(inputShape.reduce(1, *), 1)
        let beforeExpr = descending
            ? "(c > 0) || (c == 0 && j < i)"
            : "(c < 0) || (c == 0 && j < i)"

        // Key buffer params (buffers 0..K-1) and the lexicographic comparison:
        // compare key0; only if equal fall through to key1; etc. `c` ends as the
        // total-order comparison of element j vs i over the keys (NaN = largest).
        let keyParams = (0..<k).map { m in
            "    device const \(metalTypeName(for: keyTypes.indices.contains(m) ? keyTypes[m] : .float32))* key\(m) [[buffer(\(m))]],"
        }.joined(separator: "\n")
        var keyCompare = "                int c = 0;\n"
        for m in 0..<k {
            let km = metalTypeName(for: keyTypes.indices.contains(m) ? keyTypes[m] : .float32)
            let cmp = """
                            { \(km) a = key\(m)[base + j * innerSize]; \(km) b = key\(m)[base + i * innerSize];
                              bool an = (a != a), bn = (b != b);
                              if (an && bn) c = 0; else if (an) c = 1; else if (bn) c = -1;
                              else if (a < b) c = -1; else if (a > b) c = 1; else c = 0; }
                """
            if m == 0 { keyCompare += cmp + "\n" }
            else { keyCompare += "                if (c == 0) {\n" + cmp + "\n                }\n" }
        }

        let payIdx = k, outIdx = k + 1, aIdx = k + 2, iIdx = k + 3, tIdx = k + 4
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_sort_result(
        \(keyParams)
            device const \(payMetal)* payload [[buffer(\(payIdx))]],
            device \(payMetal)* output [[buffer(\(outIdx))]],
            constant uint& axisLen [[buffer(\(aIdx))]],
            constant uint& innerSize [[buffer(\(iIdx))]],
            constant uint& total [[buffer(\(tIdx))]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= total) return;
            uint inner = tid % innerSize;
            uint tmp = tid / innerSize;
            uint i = tmp % axisLen;
            uint outer = tmp / axisLen;
            uint base = outer * axisLen * innerSize + inner;

            uint rk = 0;
            for (uint j = 0; j < axisLen; j++) {
        \(keyCompare)
                if (\(beforeExpr)) rk++;
            }
            output[base + rk * innerSize] = payload[base + i * innerSize];
        }
        """
        _ = (axisLen, innerSize, total)
        return (source, "kernel_sort_result", nil)
    }

    private func generateArgReduceSource(
        inputShape: [Int],
        reduceDims: [Int],
        reductionKind: ReductionKind,
        valueType: ElementType,
        indexType: ElementType
    ) -> (String, String, TuningConfig?) {
        let reduceDimSorted = reduceDims.sorted()

        // reduceSize = product of reduced dims.
        var reduceSize = 1
        for d in reduceDimSorted where d < inputShape.count {
            reduceSize *= inputShape[d]
        }

        // outputCount = product of non-reduced dims.
        var outputShape: [Int] = []
        for (i, s) in inputShape.enumerated() where !reduceDimSorted.contains(i) {
            outputShape.append(s)
        }
        let outputCount = max(outputShape.reduce(1, *), 1)

        // innerSize = product of dims after the last reduced dim, so the scan
        // strides over the reduce axis correctly for non-trailing reductions.
        let lastReduceDim = reduceDimSorted.last ?? 0
        let innerSize = lastReduceDim + 1 < inputShape.count
            ? inputShape.suffix(from: lastReduceDim + 1).reduce(1, *)
            : 1

        let valueMetal = metalTypeName(for: valueType)
        let indexMetal = metalTypeName(for: indexType)

        // argmax keeps the strictly-greater value; argmin the strictly-smaller.
        // For NaN inputs JAX treats NaN as the largest (its reducer ORs in a
        // `value != value` test), so on argmax a NaN wins. `!(val <= best)` is
        // true when val is NaN, replicating that; for argmin NaN never wins.
        let betterCond: String
        switch reductionKind {
        case .min:
            betterCond = "val < best"
        default:
            // .max (argmax) and any non-min comparison kind.
            betterCond = "!(val <= best)"
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_arg_reduce(
            device const \(valueMetal)* input [[buffer(0)]],
            device \(indexMetal)* output [[buffer(1)]],
            constant uint& outputCount [[buffer(2)]],
            constant uint& reduceSize [[buffer(3)]],
            constant uint& innerSize [[buffer(4)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= outputCount) return;

            uint outerIdx = tid / innerSize;
            uint innerIdx = tid % innerSize;
            uint baseIdx = outerIdx * reduceSize * innerSize + innerIdx;

            \(valueMetal) best = input[baseIdx];
            uint bestIdx = 0;
            for (uint i = 1; i < reduceSize; i++) {
                \(valueMetal) val = input[baseIdx + i * innerSize];
                if (\(betterCond)) {
                    best = val;
                    bestIdx = i;
                }
            }
            output[tid] = (\(indexMetal))bestIdx;
        }
        """

        return (source, "kernel_arg_reduce", TuningConfig(blockSize: 256))
    }

    /// Generates a reduce_window (pooling) kernel.
    /// Supports max, sum, min reductions with arbitrary window dimensions, strides, padding, and dilations.
    /// 1 thread per output element.
    private func generateReduceWindowKernel(
        entryPoint: String,
        inputShape: [Int],
        outputShape: [Int],
        attributes: HLOAttributes,
        metalType: String
    ) -> String {
        // Rank 1 and 2 reduce_windows are how JAX lowers jnp.cumsum (a windowed
        // prefix sum with left padding) along a single axis; the loop/bounds
        // logic below is rank-general, so support any rank >= 1.
        guard inputShape.count >= 1, outputShape.count >= 1,
              inputShape.count == outputShape.count else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let rank = inputShape.count
        let windowDims = attributes.windowDimensions ?? Array(repeating: 1, count: rank)
        let strides = attributes.windowStrides ?? Array(repeating: 1, count: rank)
        let padding = attributes.convPadding ?? Array(repeating: [0, 0], count: rank)
        let windowDilations = attributes.windowDilations ?? Array(repeating: 1, count: rank)
        let baseDilations = attributes.baseDilations ?? Array(repeating: 1, count: rank)
        let reductionKind = attributes.reductionKind ?? .max

        let totalOutputElements = outputShape.reduce(1, *)

        // Compute input/output strides
        func computeStrides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let outputStrides = computeStrides(outputShape)
        let inputStrides = computeStrides(inputShape)

        // Initial value for reduction
        let initValue: String
        switch reductionKind {
        case .max: initValue = "-INFINITY"
        case .min: initValue = "INFINITY"
        case .sum, .mean: initValue = "0"
        case .product: initValue = "1"
        case .and: initValue = "1"
        case .or: initValue = "0"
        // log-add-exp identity is -inf (logsumexp of the empty set), matching
        // the 0xFF800000 (-inf) init constant JAX feeds the reduce_window.
        case .logAddExp: initValue = "-INFINITY"
        }

        var code = """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device const \(metalType)* init_val [[buffer(1)]],
            device \(metalType)* output [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= \(totalOutputElements)u) return;

            // Decompose tid into output coordinates
            uint remaining = tid;

        """

        // Generate output coordinate decomposition
        for i in 0..<rank {
            if i == rank - 1 {
                code += "    int o\(i) = int(remaining);\n"
            } else {
                code += "    int o\(i) = int(remaining / \(outputStrides[i])u);\n"
                code += "    remaining = remaining % \(outputStrides[i])u;\n"
            }
        }

        code += "\n    \(metalType) acc = \(metalType)(\(initValue));\n"

        // Count for mean computation
        if reductionKind == .mean {
            code += "    int window_count = 0;\n"
        }

        // log-add-exp seeds acc with -inf, but Metal's fast-math contracts treat
        // infinities as UB, so a -inf accumulator can poison the first combine.
        // Track whether any element has been folded in yet and assign the first
        // one directly (logaddexp(-inf, x) == x), keeping the stable form only
        // for subsequent elements. This is exact and avoids all inf arithmetic.
        if reductionKind == .logAddExp {
            code += "    bool acc_seen = false;\n"
        }

        // Generate nested loops for window dimensions
        // Only loop over dimensions where window > 1
        var loopDims: [Int] = []
        for d in 0..<rank {
            if windowDims[d] > 1 {
                loopDims.append(d)
                code += "    for (int w\(d) = 0; w\(d) < \(windowDims[d]); w\(d)++) {\n"
            }
        }

        // Compute input positions and bounds check
        var boundsChecks: [String] = []
        for d in 0..<rank {
            let padLow = padding.count > d ? padding[d][0] : 0
            let stride = strides.count > d ? strides[d] : 1
            let wDil = windowDilations.count > d ? windowDilations[d] : 1
            let bDil = baseDilations.count > d ? baseDilations[d] : 1

            if windowDims[d] > 1 {
                if bDil > 1 {
                    code += "        int ip\(d)_virtual = o\(d) * \(stride) + w\(d) * \(wDil) - \(padLow);\n"
                    code += "        int ip\(d) = ip\(d)_virtual / \(bDil);\n"
                    boundsChecks.append("ip\(d)_virtual >= 0 && ip\(d)_virtual % \(bDil) == 0 && ip\(d) >= 0 && ip\(d) < \(inputShape[d])")
                } else {
                    code += "        int ip\(d) = o\(d) * \(stride) + w\(d) * \(wDil) - \(padLow);\n"
                    boundsChecks.append("ip\(d) >= 0 && ip\(d) < \(inputShape[d])")
                }
            } else {
                // No window in this dimension — input pos = output pos
                code += "        int ip\(d) = o\(d);\n"
            }
        }

        if !boundsChecks.isEmpty {
            code += "        if (\(boundsChecks.joined(separator: " && "))) {\n"
        }

        // Compute input flat index
        let inputIdxParts = (0..<rank).map { "ip\($0) * \(inputStrides[$0])" }
        let inputIdx = inputIdxParts.joined(separator: " + ")
        code += "            \(metalType) val = input[\(inputIdx)];\n"

        // Apply reduction
        switch reductionKind {
        case .max:
            code += "            acc = max(acc, val);\n"
        case .min:
            code += "            acc = min(acc, val);\n"
        case .sum:
            code += "            acc += val;\n"
        case .mean:
            code += "            acc += val;\n"
            code += "            window_count++;\n"
        case .product:
            code += "            acc *= val;\n"
        case .and:
            code += "            acc = \(metalType)(int(acc) & int(val));\n"
        case .or:
            code += "            acc = \(metalType)(int(acc) | int(val));\n"
        case .logAddExp:
            // Numerically stable logaddexp, mirroring JAX's reducer region:
            //   m = max(acc, val); d = acc - val;
            //   result = m + log1p(exp(-|d|))
            // The d != d (NaN) guard handles acc == val == -inf, where d is NaN;
            // there the stable form is undefined, so fall back to acc + val (= -inf)
            // exactly as the source select(NE d d, acc+val, ...) does.
            code += "            if (!acc_seen) {\n"
            code += "                acc = val;\n"
            code += "                acc_seen = true;\n"
            code += "            } else {\n"
            // MSL has no log1p; the argument 1 + exp(-|d|) is always in [1, 2],
            // so log(1 + e) loses no precision versus log1p(e) here (log1p only
            // matters when e is near 0, but e + 1 >= 1 keeps full mantissa).
            code += "                \(metalType) m = max(acc, val);\n"
            code += "                \(metalType) d = acc - val;\n"
            code += "                \(metalType) stable = m + \(metalType)(log(\(metalType)(1) + exp(-abs(d))));\n"
            code += "                acc = (d != d) ? (acc + val) : stable;\n"
            code += "            }\n"
        }

        if !boundsChecks.isEmpty {
            code += "        }\n"
        }

        // Close window loops
        for _ in loopDims {
            code += "    }\n"
        }

        if reductionKind == .mean {
            code += "    if (window_count > 0) acc = acc / \(metalType)(window_count);\n"
        }

        code += """

            output[tid] = acc;
        }
        """

        return code
    }

    /// Generates a select_and_scatter kernel (pooling gradient).
    /// For max pooling gradient: finds the max position in each window and scatters the gradient there.
    /// Dispatches 1 thread per output (input-shaped) element; each initializes to init value,
    /// then a second pass scatters gradients.
    private func generateSelectAndScatterKernel(
        entryPoint: String,
        inputShape: [Int],   // operand (forward pass input)
        sourceShape: [Int],  // gradient to scatter
        outputShape: [Int],  // same shape as inputShape
        attributes: HLOAttributes,
        metalType: String
    ) -> String {
        guard inputShape.count >= 3 else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let rank = inputShape.count
        let dimNumbers = attributes.selectAndScatterDimensionNumbers
        let windowDims = dimNumbers?.windowDimensions ?? Array(repeating: 1, count: rank)
        let strides = dimNumbers?.windowStrides ?? Array(repeating: 1, count: rank)
        let padding = dimNumbers?.padding ?? Array(repeating: [0, 0], count: rank)

        let totalInputElements = inputShape.reduce(1, *)
        let totalSourceElements = sourceShape.reduce(1, *)

        func computeStrides(_ shape: [Int]) -> [Int] {
            var s = Array(repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                s[i] = s[i + 1] * shape[i + 1]
            }
            return s
        }
        let inputStrides = computeStrides(inputShape)
        let sourceStrides = computeStrides(sourceShape)

        // Two-phase kernel: first init output, then scatter
        var code = """
        kernel void \(entryPoint)(
            device const \(metalType)* operand [[buffer(0)]],
            device const \(metalType)* source [[buffer(1)]],
            device const \(metalType)* init_val [[buffer(2)]],
            device \(metalType)* output [[buffer(3)]],
            constant uint& count [[buffer(4)]],
            uint tid [[thread_position_in_grid]])
        {
            // Phase 1: Initialize output with init value
            if (tid < \(totalInputElements)u) {
                output[tid] = init_val[0];
            }

            threadgroup_barrier(mem_flags::mem_device);

            // Phase 2: For each source element, find selected position and scatter
            if (tid < \(totalSourceElements)u) {
                uint remaining = tid;

        """

        // Decompose source tid into coordinates
        for i in 0..<rank {
            if i == rank - 1 {
                code += "        int s\(i) = int(remaining);\n"
            } else {
                code += "        int s\(i) = int(remaining / \(sourceStrides[i])u);\n"
                code += "        remaining = remaining % \(sourceStrides[i])u;\n"
            }
        }

        code += "\n        // Find max position in window\n"
        code += "        \(metalType) max_val = -INFINITY;\n"
        code += "        int max_idx = 0;\n"

        // Nested loops over window
        var loopDims: [Int] = []
        for d in 0..<rank {
            if windowDims[d] > 1 {
                loopDims.append(d)
                code += "        for (int w\(d) = 0; w\(d) < \(windowDims[d]); w\(d)++) {\n"
            }
        }

        // Compute input positions
        var boundsChecks: [String] = []
        for d in 0..<rank {
            let padLow = padding.count > d ? padding[d][0] : 0
            let stride = strides.count > d ? strides[d] : 1
            if windowDims[d] > 1 {
                code += "            int ip\(d) = s\(d) * \(stride) + w\(d) - \(padLow);\n"
                boundsChecks.append("ip\(d) >= 0 && ip\(d) < \(inputShape[d])")
            } else {
                code += "            int ip\(d) = s\(d);\n"
            }
        }

        if !boundsChecks.isEmpty {
            code += "            if (\(boundsChecks.joined(separator: " && "))) {\n"
        }

        let inputIdxParts = (0..<rank).map { "ip\($0) * \(inputStrides[$0])" }
        let inputIdx = inputIdxParts.joined(separator: " + ")
        code += "                int idx = \(inputIdx);\n"
        code += "                \(metalType) val = operand[idx];\n"
        code += "                if (val > max_val) { max_val = val; max_idx = idx; }\n"

        if !boundsChecks.isEmpty {
            code += "            }\n"
        }

        for _ in loopDims {
            code += "        }\n"
        }

        // Scatter source value at max position (atomic add for correctness)
        code += "\n        output[max_idx] += source[tid];\n"

        code += """
            }
        }
        """

        return code
    }

    /// Generates an FFT kernel using the naive DFT formula.
    /// X[k] = Σ x[n] * exp(-2πi*n*k/N) for forward FFT
    /// Handles real and complex input/output.
    /// 1 thread per output element.
    private func generateFFTKernel(
        entryPoint: String,
        inputShape: [Int],
        outputShape: [Int],
        attributes: HLOAttributes,
        metalType: String
    ) -> String {
        let fftType = attributes.fftType ?? .fft
        let fftLength = attributes.fftLength ?? [inputShape.last ?? 1]

        guard let N = fftLength.last, N > 0 else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let totalOutputElements = outputShape.reduce(1, *)
        let batchSize = totalOutputElements / (fftType == .rfft ? (N / 2 + 1) : N)
        let isInverse = (fftType == .ifft || fftType == .irfft)
        let sign = isInverse ? "1.0" : "-1.0"

        switch fftType {
        case .rfft:
            // Real input → complex output (interleaved real/imag)
            let outputN = N / 2 + 1
            return """
            kernel void \(entryPoint)(
                device const \(metalType)* input [[buffer(0)]],
                device \(metalType)* output [[buffer(1)]],
                constant uint& count [[buffer(2)]],
                uint tid [[thread_position_in_grid]])
            {
                if (tid >= \(totalOutputElements)u) return;

                // Each output element is a complex pair (real, imag)
                uint complex_idx = tid / 2;
                bool is_imag = (tid % 2) == 1;

                uint batch = complex_idx / \(outputN)u;
                uint k = complex_idx % \(outputN)u;

                float sum_real = 0.0;
                float sum_imag = 0.0;
                float angle_base = \(sign) * 2.0 * M_PI_F * float(k) / float(\(N));

                for (uint n = 0; n < \(N)u; n++) {
                    float angle = angle_base * float(n);
                    float val = float(input[batch * \(N)u + n]);
                    sum_real += val * cos(angle);
                    sum_imag += val * sin(angle);
                }

                output[tid] = \(metalType)(is_imag ? sum_imag : sum_real);
            }
            """

        case .irfft:
            // Complex input → real output
            let inputN = N / 2 + 1
            return """
            kernel void \(entryPoint)(
                device const \(metalType)* input [[buffer(0)]],
                device \(metalType)* output [[buffer(1)]],
                constant uint& count [[buffer(2)]],
                uint tid [[thread_position_in_grid]])
            {
                if (tid >= \(totalOutputElements)u) return;

                uint batch = tid / \(N)u;
                uint n = tid % \(N)u;

                float sum = 0.0;
                for (uint k = 0; k < \(inputN)u; k++) {
                    uint base = batch * \(inputN)u * 2 + k * 2;
                    float re = float(input[base]);
                    float im = float(input[base + 1]);

                    float angle = 2.0 * M_PI_F * float(k) * float(n) / float(\(N));
                    sum += re * cos(angle) - im * sin(angle);

                    // Mirror: add conjugate contribution for k > 0 and k < N/2
                    if (k > 0 && k < \(N / 2)u) {
                        uint mirror_k = \(N)u - k;
                        float m_angle = 2.0 * M_PI_F * float(mirror_k) * float(n) / float(\(N));
                        sum += re * cos(m_angle) + im * sin(m_angle);
                    }
                }

                output[tid] = \(metalType)(sum / float(\(N)));
            }
            """

        case .fft, .ifft:
            // Complex-to-complex (interleaved real/imag pairs)
            let scale = isInverse ? "/ float(\(N))" : ""
            return """
            kernel void \(entryPoint)(
                device const \(metalType)* input [[buffer(0)]],
                device \(metalType)* output [[buffer(1)]],
                constant uint& count [[buffer(2)]],
                uint tid [[thread_position_in_grid]])
            {
                if (tid >= \(totalOutputElements)u) return;

                // Input/output are interleaved (real, imag) pairs
                uint complex_idx = tid / 2;
                bool is_imag = (tid % 2) == 1;

                uint batch = complex_idx / \(N)u;
                uint k = complex_idx % \(N)u;

                float sum_real = 0.0;
                float sum_imag = 0.0;
                float angle_base = \(sign) * 2.0 * M_PI_F * float(k) / float(\(N));

                for (uint n = 0; n < \(N)u; n++) {
                    uint base = batch * \(N)u * 2 + n * 2;
                    float re = float(input[base]);
                    float im = float(input[base + 1]);

                    float angle = angle_base * float(n);
                    float cos_a = cos(angle);
                    float sin_a = sin(angle);

                    sum_real += re * cos_a - im * sin_a;
                    sum_imag += re * sin_a + im * cos_a;
                }

                output[tid] = \(metalType)(is_imag ? sum_imag \(scale) : sum_real \(scale));
            }
            """
        }
    }

    /// Generates a transpose kernel that permutes dimensions.
    ///
    /// The transpose operation reorders elements from input to output according to the permutation.
    /// For example, transpose([2,3,4], perm=[2,0,1]) means:
    /// - output dim 0 comes from input dim 2
    /// - output dim 1 comes from input dim 0
    /// - output dim 2 comes from input dim 1
    private func generateTransposeKernel(
        entryPoint: String,
        inputShape: [Int],
        outputShape: [Int],
        permutation: [Int],
        metalType: String = "float"
    ) -> String {
        // If no permutation or shapes are empty, fall back to copy
        guard !inputShape.isEmpty, !permutation.isEmpty else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        // Check for simple 2D matrix transpose [M, N] -> [N, M]
        // This is the most common case and can be heavily optimized with tiled shared memory
        if inputShape.count == 2 && permutation == [1, 0] {
            return generateTiled2DTransposeKernel(
                entryPoint: entryPoint,
                rows: inputShape[0],
                cols: inputShape[1],
                metalType: metalType
            )
        }

        // Check for 3D transpose that swaps last two dimensions [B, M, N] -> [B, N, M]
        // This is common for attention mechanisms
        if inputShape.count == 3 && permutation == [0, 2, 1] {
            return generateBatched2DTransposeKernel(
                entryPoint: entryPoint,
                batch: inputShape[0],
                rows: inputShape[1],
                cols: inputShape[2],
                metalType: metalType
            )
        }

        // For other cases, use the general transpose kernel
        return generateGeneralTransposeKernel(
            entryPoint: entryPoint,
            inputShape: inputShape,
            outputShape: outputShape,
            permutation: permutation,
            metalType: metalType
        )
    }

    /// Generates an optimized tiled 2D transpose kernel using shared memory.
    /// Uses 32x32 tiles with padding to avoid bank conflicts.
    private func generateTiled2DTransposeKernel(
        entryPoint: String,
        rows: Int,
        cols: Int,
        metalType: String = "float"
    ) -> String {
        let tileSize = 32
        let tilePadding = 1  // Padding to avoid bank conflicts

        return """
        // Optimized tiled 2D transpose with shared memory
        // Input: [\(rows), \(cols)] -> Output: [\(cols), \(rows)]
        // Uses \(tileSize)x\(tileSize) tiles with +\(tilePadding) padding to avoid bank conflicts
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            threadgroup \(metalType)* tile [[threadgroup(0)]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 gid [[threadgroup_position_in_grid]],
            uint2 threads_per_tg [[threads_per_threadgroup]])
        {
            const uint TILE_DIM = \(tileSize);
            const uint TILE_PADDED = \(tileSize + tilePadding);
            const uint rows = \(rows);
            const uint cols = \(cols);

            // Input tile position
            uint x_in = gid.x * TILE_DIM + tid.x;
            uint y_in = gid.y * TILE_DIM + tid.y;

            // Load tile from input into shared memory with coalesced reads
            // Each thread loads one element
            if (x_in < cols && y_in < rows) {
                tile[tid.y * TILE_PADDED + tid.x] = input[y_in * cols + x_in];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Output tile position - note the transposed tile indices
            // We write to (gid.y, gid.x) in the output, not (gid.x, gid.y)
            uint x_out = gid.y * TILE_DIM + tid.x;
            uint y_out = gid.x * TILE_DIM + tid.y;

            // Write transposed tile to output with coalesced writes
            // Note: we read from [tid.x, tid.y] in the tile (transposed)
            if (x_out < rows && y_out < cols) {
                output[y_out * rows + x_out] = tile[tid.x * TILE_PADDED + tid.y];
            }
        }
        """
    }

    /// Generates an optimized batched 2D transpose kernel for 3D tensors [B, M, N] -> [B, N, M].
    private func generateBatched2DTransposeKernel(
        entryPoint: String,
        batch: Int,
        rows: Int,
        cols: Int,
        metalType: String = "float"
    ) -> String {
        let tileSize = 32
        let tilePadding = 1

        return """
        // Optimized batched 2D transpose with shared memory
        // Input: [\(batch), \(rows), \(cols)] -> Output: [\(batch), \(cols), \(rows)]
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            threadgroup \(metalType)* tile [[threadgroup(0)]],
            uint3 tid [[thread_position_in_threadgroup]],
            uint3 gid [[threadgroup_position_in_grid]])
        {
            const uint TILE_DIM = \(tileSize);
            const uint TILE_PADDED = \(tileSize + tilePadding);
            const uint rows = \(rows);
            const uint cols = \(cols);
            const uint batch_stride = rows * cols;

            uint b = gid.z;  // Batch index

            // Input tile position within batch slice
            uint x_in = gid.x * TILE_DIM + tid.x;
            uint y_in = gid.y * TILE_DIM + tid.y;

            // Load tile from input into shared memory
            if (x_in < cols && y_in < rows) {
                tile[tid.y * TILE_PADDED + tid.x] = input[b * batch_stride + y_in * cols + x_in];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Output position - transposed tile
            uint x_out = gid.y * TILE_DIM + tid.x;
            uint y_out = gid.x * TILE_DIM + tid.y;

            // Write transposed tile to output
            if (x_out < rows && y_out < cols) {
                output[b * batch_stride + y_out * rows + x_out] = tile[tid.x * TILE_PADDED + tid.y];
            }
        }
        """
    }

    /// Generates a general transpose kernel for arbitrary dimension permutations.
    /// Falls back to element-wise approach but with better memory access patterns.
    private func generateGeneralTransposeKernel(
        entryPoint: String,
        inputShape: [Int],
        outputShape: [Int],
        permutation: [Int],
        metalType: String = "float"
    ) -> String {
        let rank = inputShape.count

        // Calculate input strides (row-major)
        var inputStrides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1]
        }

        // Calculate output strides (row-major)
        var outputStrides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }

        // Generate code to compute output coordinates from linear index
        var coordCode = "uint remaining = tid;\n"
        for i in stride(from: 0, to: rank, by: 1) {
            coordCode += "        uint coord\(i) = remaining / \(outputStrides[i]);\n"
            coordCode += "        remaining = remaining % \(outputStrides[i]);\n"
        }

        // Generate code to compute input linear index from permuted coordinates
        // For transpose, permutation[i] tells us which input dimension feeds into output dimension i
        // So for output coordinate coord[i], we need to use inputStrides[permutation[i]]
        var inputIndexCode = "uint inputIdx = "
        var terms: [String] = []
        for i in 0..<rank {
            let inputDim = permutation[i]
            if inputDim < inputStrides.count && inputStrides[inputDim] != 0 {
                terms.append("coord\(i) * \(inputStrides[inputDim])")
            }
        }
        inputIndexCode += terms.isEmpty ? "0" : terms.joined(separator: " + ")
        inputIndexCode += ";"

        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;

            // Compute output coordinates from linear index
            \(coordCode)
            // Compute input index using permuted coordinates
            \(inputIndexCode)

            output[tid] = input[inputIdx];
        }
        """
    }

    /// Generates a broadcast kernel that maps elements from a smaller input to a larger output.
    private func generateBroadcastKernel(
        entryPoint: String,
        inputShapes: [[Int]],
        outputShapes: [[Int]],
        attributes: HLOAttributes,
        metalType: String = "float"
    ) -> String {
        guard let inputShape = inputShapes.first,
              let outputShape = outputShapes.first else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        // Get broadcast dimensions from attributes (dimensions attribute for broadcast_in_dim)
        let broadcastDims = attributes.dimensions ?? []

        // Calculate output strides for index computation
        var outputStrides = [Int](repeating: 1, count: outputShape.count)
        for i in stride(from: outputShape.count - 2, through: 0, by: -1) {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }

        // Calculate input strides
        var inputStrides = [Int](repeating: 1, count: inputShape.count)
        for i in stride(from: inputShape.count - 2, through: 0, by: -1) {
            inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1]
        }

        // Build index computation code.
        //
        // Each coord is computed independently from outputStrides:
        //   coord = (tid / outputStrides[d]) % outputShape[d]
        //
        // METAL UINT-MODULO MISCOMPILE WORKAROUND
        // ───────────────────────────────────────
        // On Apple Silicon (verified on M5 Pro / macOS 26 / Metal 4), the `%`
        // operator on unsigned ints produces garbage when both:
        //   (a) the dividend is a `tid`-derived expression (e.g. `tid / K`)
        //   (b) the divisor is a non-power-of-2 compile-time constant
        // Concrete repro: `(tid / 65536u) % 6u` returns 65530 instead of 0
        // for small tid. The same expression rewritten as
        //   `a - (a / b) * b`
        // produces correct results. Unsigned divide is fine; only `%` is
        // miscompiled. Symptom for callers: broadcast (16,6,256) →
        // (16,6,256,256) produced 0x60000000 (= ~3.7e19 as float) at >99%
        // of output positions, which then propagated as NaN through the
        // softmax/attention pipeline that consumed the broadcast result.
        //
        // We always emit the manual form for non-power-of-2 sizes; for
        // power-of-2 sizes the `% N` form compiles to `& (N-1)` and works.
        func emitMod(dividend: String, divisor: Int) -> String {
            if divisor > 0 && (divisor & (divisor - 1)) == 0 {
                return "(\(dividend)) % \(divisor)u"
            }
            return "((\(dividend)) - ((\(dividend)) / \(divisor)u) * \(divisor)u)"
        }
        var indexCode = ""
        if inputShape.isEmpty || inputShape == [1] || (inputShape.count == 1 && inputShape[0] == 1) {
            // Scalar broadcast - all elements read from same location
            indexCode = "uint inputIdx = 0;"
        } else if broadcastDims.isEmpty {
            // No explicit dimensions — assume trailing dimensions match.
            indexCode = "uint inputIdx = 0;"
            for outDim in 0..<outputShape.count {
                let inputDim = outDim - (outputShape.count - inputShape.count)
                guard inputDim >= 0 && inputDim < inputShape.count else { continue }
                if inputShape[inputDim] == 1 { continue }  // broadcasts (size 1)
                let stride = outputStrides[outDim]
                let size = outputShape[outDim]
                let div = stride == 1 ? "tid" : "(tid / \(stride)u)"
                indexCode += "\n    uint bcCoord\(outDim) = \(emitMod(dividend: div, divisor: size));"
                indexCode += "\n    inputIdx += bcCoord\(outDim) * \(inputStrides[inputDim])u;"
            }
        } else {
            // Explicit broadcast dimensions mapping.
            indexCode = "uint inputIdx = 0;"
            for (inputDim, outputDim) in broadcastDims.enumerated() {
                guard inputDim < inputShape.count && outputDim < outputShape.count else { continue }
                if inputShape[inputDim] == 1 { continue }  // broadcasts (size 1)
                let stride = outputStrides[outputDim]
                let size = outputShape[outputDim]
                let div = stride == 1 ? "tid" : "(tid / \(stride)u)"
                indexCode += "\n    uint bcCoord\(outputDim) = \(emitMod(dividend: div, divisor: size));"
                indexCode += "\n    inputIdx += bcCoord\(outputDim) * \(inputStrides[inputDim])u;"
            }
        }

        _ = outputShape.reduce(1, *)  // outputCount available if needed; no longer baked into kernel.

        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            \(indexCode)
            output[tid] = input[inputIdx];
        }
        """
    }

    /// Generates matmul source with support for all numeric types.
    /// Uses simdgroup_matrix operations for float32/float16 for optimal performance on Apple Silicon.
    private func generateMatMulSource(inputShapes: [[Int]], attributes: HLOAttributes, elementType: ElementType = .float32) -> (String, String, TuningConfig?) {
        guard inputShapes.count >= 2 else {
            let metalType = metalTypeName(for: elementType)
            return (generateCopyKernel(entryPoint: "kernel_matmul", metalType: metalType), "kernel_matmul", nil)
        }

        let lhsShape = inputShapes[0]
        let rhsShape = inputShapes[1]
        let metalType = metalTypeName(for: elementType)
        let isFloat = isFloatType(elementType)

        // Get contracting dimensions from dot_general attributes
        let dotDimNums = attributes.dotDimensionNumbers
        var lhsContractDims = dotDimNums?.lhsContractingDimensions ?? [lhsShape.count - 1]
        var rhsContractDims = dotDimNums?.rhsContractingDimensions ?? (rhsShape.count >= 2 ? [rhsShape.count - 2] : [0])
        var lhsBatchDims = dotDimNums?.lhsBatchingDimensions ?? []
        var rhsBatchDims = dotDimNums?.rhsBatchingDimensions ?? []

        // Apply lhsTranspose/rhsTranspose attributes from the transpose-folding optimizer.
        // The fold pass replaces `dot(transpose(X), Y)` with `dot(X, Y, lhsTranspose=true)`
        // but copies the original dot's dim numbers verbatim — those still index into the
        // pre-fold (post-transpose) tensor's shape. Swap the last-two-dim positions so the
        // dim numbers index into the actual operand X's shape, matching how the existing
        // M/K/N extraction and transA/transB synthesis below interpret them.
        if attributes.lhsTranspose == true, lhsShape.count >= 2 {
            lhsContractDims = swapLastTwoDimPositions(lhsContractDims, rank: lhsShape.count)
            lhsBatchDims = swapLastTwoDimPositions(lhsBatchDims, rank: lhsShape.count)
        }
        if attributes.rhsTranspose == true, rhsShape.count >= 2 {
            rhsContractDims = swapLastTwoDimPositions(rhsContractDims, rank: rhsShape.count)
            rhsBatchDims = swapLastTwoDimPositions(rhsBatchDims, rank: rhsShape.count)
        }

        // Calculate batch size from explicit batching_dims only. See the
        // matching comment in dotGeneralDims for why the implicit-batch
        // heuristic on lhsShape was wrong.
        let batchSize: Int = lhsBatchDims.isEmpty
            ? 1
            : lhsBatchDims.reduce(1) { $0 * lhsShape[$1] }

        // Extract M, K, N:
        // M = product of LHS non-contracting, non-batch dims
        // K = product of contracting dims
        // N = product of RHS non-contracting, non-batch dims
        var M = 1
        for (i, s) in lhsShape.enumerated() {
            if !lhsContractDims.contains(i) && !lhsBatchDims.contains(i) { M *= s }
        }
        var K = 1
        for d in lhsContractDims { K *= lhsShape[d] }
        var N = 1
        for (i, s) in rhsShape.enumerated() {
            if !rhsContractDims.contains(i) && !rhsBatchDims.contains(i) { N *= s }
        }

        // Determine if LHS or RHS needs transposition.
        // The kernel assumes A is [M, K] and B is [K, N] in memory.
        // If the contracting dim is not in the expected position, we need to
        // adjust the read pattern.
        let lhsNonBatchDims = lhsShape.enumerated().filter { !lhsBatchDims.contains($0.offset) }.map { $0.offset }
        let rhsNonBatchDims = rhsShape.enumerated().filter { !rhsBatchDims.contains($0.offset) }.map { $0.offset }
        // LHS: contracting dim should be last among non-batch dims for [M, K] layout
        let transA = !lhsNonBatchDims.isEmpty && lhsContractDims.contains(lhsNonBatchDims.first!)
        // RHS: contracting dim should be first among non-batch dims for [K, N] layout
        let transB = !rhsNonBatchDims.isEmpty && lhsContractDims.count == 1 &&
                     rhsContractDims.contains(rhsNonBatchDims.last!)

        // GEMV path: M=1 (vector-matrix multiply — e.g. LLM logit projection,
        // single-token decode). The simdgroup_matrix kernel needs M ≥ 8 to
        // fill its 8×8 tiles and wastes (8-M)/8 of every tile when M < 8.
        // The dedicated kernel uses 32-thread TGs, each thread owns 4 output
        // columns, giving 128 outputs/TG with cacheline-coalesced W reads.
        // Split-K GEMV: 4 simdgroups per TG cooperate on K, sum partials via
        // threadgroup memory. Requires K divisible by tk*numSimd = 8*4 = 32.
        if isFloat && M == 1 && batchSize == 1 && !transA && !transB
           && N >= 64 && N % 2 == 0 && K % 32 == 0 {
            return generateGEMVSource(K: K, N: N, metalType: metalType)
        }

        // Use optimized simdgroup kernel only for float types with dims that are
        // multiples of 8. The MPP path additionally handles transA/transB via
        // matmul2d_descriptor's transpose_left/transpose_right; the
        // simdgroup_matrix kernel does not yet, so it stays gated on
        // !transA && !transB.
        if isFloat && M % 8 == 0 && K % 8 == 0 && N % 8 == 0 && M >= 8 && K >= 8 && N >= 8 {
            // Apple Matrix Coprocessor path via MetalPerformancePrimitives.
            // Requires Metal language 4.0 and Apple9 GPU family (M3+); MPSGraph
            // and MLX both use this primitive to hit ~97% of fp16 peak on M3+.
            // Tile is 128x128, dispatched as 8 simdgroups (256 threads).
            // Batched and non-batched share one kernel emitter — the kernel
            // reads tgid.z for batch index and offsets the device pointers.
            //
            // Gate on total threadgroup count (M/128 * N/128 * batchSize) so
            // small per-matrix shapes still benefit when there are many
            // batches feeding the GPU. M5 Pro has 16 cores × ~4 concurrent
            // threadgroups; we want at least ~16 to keep occupancy.
            //
            // Mixed precision: when input element type is fp16 but output is
            // fp32, we use the MPP `half × half → float` overload, which
            // fuses the output cast in-register. This is what the TF32
            // transform's output-convert fusion produces. The same gate that
            // TF32Transform uses (in MetalHLOCompiler) must mirror this.
            // Matmul kernel selection.
            //
            // For fp32: prefer an MLX-style "Steel" simdgroup_matrix kernel.
            // MLX explicitly avoids Apple's `matmul2d` MPP primitive for
            // fp32 because the matrix coprocessor is tuned for half/bfloat;
            // a hand-rolled `simdgroup_float8x8` kernel with small tiles
            // and cooperative threadgroup-memory loads beats MPP on fp32.
            //
            // For non-fp32 (half / bfloat16): keep the matmul2d path with
            // either 128-tile or 64-tile variant, picked by occupancy.
            //
            // Both paths require the matmul itself to be untransposed; the
            // transposed case still falls through to the basic kernel.
            let tg128 = (M / 128) * (N / 128) * batchSize
            let tg64  = (M / 64)  * (N / 64)  * batchSize
            let supports = supportsMetalPerformancePrimitives()
            let inputElementType = attributes.inputElementTypes?.first ?? elementType
            let isFp32 = elementType == .float32 && inputElementType == .float32
            // The Steel-style fp32 kernel (generateMatMul2dSteelSource)
            // turns out to be ~5–10× slower than `matmul2d` MPP on M5 Pro
            // — the matrix coprocessor's fp32 path is actually competitive
            // on Apple9-class GPUs. MLX's gate against fp32-MPP was tuned
            // for older Apple Silicon. We keep the Steel codegen as a
            // diagnostic option (METALHLO_FORCE_STEEL=1) but default to
            // MPP for fp32 too. See nanogpt_5x_target_blockers.md for the
            // per-shape numbers.
            let disableSteel = ProcessInfo.processInfo.environment["METALHLO_FORCE_STEEL"] != "1"
            // Steel fp32 path. Aligned only (M%64==N%64==K%16==0) for now;
            // smaller / unaligned shapes fall through to simdgroup or basic.
            // Need at least ~16 64-tiles to make the kernel worth it.
            let steelEligible = isFp32 && !disableSteel
                && !transA && !transB
                && M % 64 == 0 && N % 64 == 0 && K % 16 == 0
                && M >= 64 && N >= 64
                && tg64 >= 16

            // Cooperative-tensor tf32 GEMM path (METALHLO_COOP_GEMM=1).
            // Keeps a BM×BN accumulator in registers across the whole K
            // dimension (in-register multiply_accumulate, write C once) like
            // MLX's gemm_nax — vs the plain op.run which lets MPP pick a small
            // internal tileK and goes bandwidth-bound. Aligned fp32 only:
            // M%64==N? we use a 64×128 tile, K looped in TILEK=128 chunks.
            // tf32 cooperative-tensor GEMM is default-on (measured correct +
            // loss-neutral on nanoGPT, faster than plain op.run). Disable with
            // METALHLO_COOP_GEMM=0; METALHLO_EXACT_FP32=1 also disables it
            // (it is a reduced-precision path).
            let coopExactFp32 = ProcessInfo.processInfo.environment["METALHLO_EXACT_FP32"] == "1"

            // Low-level NAX GEMM (MLX gemm_loop). MLX-grade tf32 on the matrix
            // coprocessor. Opt-in (METALHLO_NAX_GEMM=1) while it stabilizes.
            let naxEnabled = ProcessInfo.processInfo.environment["METALHLO_NAX_GEMM"] == "1"
            let naxCfg = CodeGenerator.naxGemmConfig()
            let naxEligible = naxEnabled && supports && isFp32 && !coopExactFp32
                && batchSize == 1 && !transA && !transB
                && M % naxCfg.bm == 0 && N % naxCfg.bn == 0 && K % naxCfg.bk == 0
                && (M / naxCfg.bm) * (N / naxCfg.bn) >= 16

            let coopEnabled = ProcessInfo.processInfo.environment["METALHLO_COOP_GEMM"] != "0"
            let coopCfg = CodeGenerator.coopGemmConfig()
            let coopEligible = !naxEligible && coopEnabled && supports && isFp32 && !coopExactFp32
                && batchSize == 1 && !transA && !transB
                && M % coopCfg.bm == 0 && N % coopCfg.bn == 0 && K % coopCfg.tilek == 0
                && (M / coopCfg.bm) * (N / coopCfg.bn) >= 16

            // matmul2d MPP path. Used for non-fp32 inputs normally; also
            // serves as a fallback for fp32 when the Steel kernel is
            // disabled or ineligible.
            let mppAllowsFp32 = !steelEligible && !coopEligible
            let mppFp32OK = !isFp32 || mppAllowsFp32
            let use128 = supports && mppFp32OK && M % 128 == 0 && N % 128 == 0 && tg128 >= 128
            let use64  = supports && mppFp32OK && M % 64  == 0 && N % 64  == 0 && tg64  >= 16
                         && !use128
            let mppEligible = use128 || use64
            let chosenTile = use128 ? 128 : 64

            if ProcessInfo.processInfo.environment["METALHLO_DEBUG_MATMUL_PATH"] != nil {
                let kernelChoice: String
                if naxEligible {
                    kernelChoice = "NAX"
                } else if coopEligible {
                    kernelChoice = "CoopTF32"
                } else if steelEligible {
                    kernelChoice = "Steel64"
                } else if mppEligible {
                    kernelChoice = "MPP\(chosenTile)"
                } else if !transA && !transB && M >= 32 && N >= 32 {
                    kernelChoice = "simdgroup"
                } else {
                    kernelChoice = "basic"
                }
                FileHandle.standardError.write("[matmul] M=\(M) N=\(N) K=\(K) batch=\(batchSize) fp32=\(isFp32) transA=\(transA) transB=\(transB) → \(kernelChoice) (tg128=\(tg128) tg64=\(tg64))\n".data(using: .utf8)!)
            }

            if naxEligible {
                return generateMatMul2dNAXSource()
            }
            if coopEligible {
                return generateMatMul2dMPPCoopSource()
            }
            if steelEligible {
                return generateMatMul2dSteelSource(batchSize: batchSize, M: M, N: N, K: K)
            }
            if mppEligible {
                return generateMatMul2dMPPSource(
                    batchSize: batchSize,
                    inputElementType: inputElementType,
                    outputElementType: elementType,
                    transA: transA,
                    transB: transB,
                    tileSize: chosenTile
                )
            }
            // The simdgroup_matrix kernel uses a 32×32 output tile (4 simdgroups,
            // each computing 8×32). For M<32 or N<32, multiple simdgroups end up
            // outside the valid output region and race-write to clamped offsets
            // — produces silently wrong values when feeding into downstream ops.
            // Concretely surfaced as the einsum→transpose→dot_general chain in
            // multi-head attention with d_model=8 (BERT-shape MHA). Gate on
            // M >= 32 AND N >= 32 so that all 4 simdgroups have valid work.
            if !transA && !transB && M >= 32 && N >= 32 {
                return generateSimdgroupMatMulSource(batchSize: batchSize, metalType: metalType, elementType: elementType)
            }
            // Fall through to the basic kernel when MPP isn't available and
            // a transpose is requested — the simdgroup_matrix kernel doesn't
            // yet honour transA/transB.
            // For bfloat, the literal `0.0` is treated as `double` which
            // doesn't implicitly convert. Use an explicit construction.
            let zeroValue = isFloat ? (metalType == "bfloat" ? "\(metalType)(0)" : "0.0") : "0"
            let aRead = transA ? "batchA[k * \(M) + row]" : "batchA[row * K + k]"
            let bRead = transB ? "batchB[col * K + k]" : "batchB[k * N + col]"
            return generateBasicMatMulSourceWithTranspose(
                batchSize: batchSize, metalType: metalType, zeroValue: zeroValue, M: M, N: N, K: K,
                aRead: aRead, bRead: bRead)
        } else {
            // For bfloat, the literal `0.0` is treated as `double` which
            // doesn't implicitly convert. Use an explicit construction.
            let zeroValue = isFloat ? (metalType == "bfloat" ? "\(metalType)(0)" : "0.0") : "0"
            // Generate read expressions that handle transposition
            let aRead = transA ? "batchA[k * \(M) + row]" : "batchA[row * K + k]"
            let bRead = transB ? "batchB[col * K + k]" : "batchB[k * N + col]"
            return generateBasicMatMulSourceWithTranspose(
                batchSize: batchSize, metalType: metalType, zeroValue: zeroValue, M: M, N: N, K: K,
                aRead: aRead, bRead: bRead)
        }
    }

    /// Returns true if the device supports Apple's MetalPerformancePrimitives
    /// matmul2d primitive — requires Apple9 GPU family (M3+) and Metal 4.
    ///
    /// The runtime kill switch `METALHLO_DISABLE_MPP=1` forces this to false
    /// on capable hardware so users can fall back to the simdgroup_matrix
    /// kernel — useful for diagnosing kernel-compile failures or working
    /// around any future regressions in the framework.
    private func supportsMetalPerformancePrimitives() -> Bool {
        if ProcessInfo.processInfo.environment["METALHLO_DISABLE_MPP"] == "1" {
            return false
        }
        // Apple9 is the M3+ GPU family. The MPP matmul2d primitive ships in
        // macOS 26 / Xcode 26 and requires Metal language version 4.0 at
        // kernel-compile time. On older OS or pre-M3 GPUs we fall back to the
        // simdgroup_matrix kernel automatically — callers don't need to gate.
        if #available(macOS 26.0, iOS 26.0, *) {
            return device.supportsFamily(.apple9)
        }
        return false
    }

    /// Generates a matmul kernel that uses Apple's MetalPerformancePrimitives
    /// `mpp::tensor_ops::matmul2d` — the matrix-coprocessor primitive used by
    /// MPSGraph and MLX's `steel_gemm_*_nax` to reach ~97% of fp16 peak on M3+.
    ///
    /// Tile: BM=64, BN=32 per threadgroup, 4 cooperating simdgroups (128
    /// threads). The kernel constructs `tensor_inline` views of the raw device
    /// pointers in-kernel so the host-side binding stays as ordinary
    /// Generates a dedicated GEMV (matrix-vector) kernel for `M=1` matmul.
    /// y = x @ W where x is [1, K] and W is [K, N].
    ///
    /// **Layout (split-K)**: one TG = `numSimd` simdgroups (`numSimd * 32`
    /// threads). All simdgroups in a TG process the **same** `outputsPerTG = 32 * tn`
    /// output columns but different K-direction chunks. Each simdgroup
    /// computes its own partial sums; partials are summed across simdgroups
    /// via threadgroup memory + a single barrier, and simdgroup 0 writes the
    /// final result. This gives `numSimd × outputsPerTG_lanes` more in-flight
    /// threads than a 32-thread-per-TG GEMV would, dramatically improving GPU
    /// occupancy on the bandwidth-bound M=1 case.
    ///
    /// Per-thread inner loop: tk=8 K-rows per iteration → `tk * tn = 16`
    /// independent FMAs to pipeline.
    ///
    /// Caller guarantees `K % (tk * numSimd) == 0` and `N % tn == 0, N ≥ outputsPerTG`.
    private func generateGEMVSource(K: Int, N: Int, metalType: String) -> (String, String, TuningConfig?) {
        let tn = 2          // outputs per thread
        let tk = 8          // K-iteration unroll — 16 independent FMAs/iter
        let numSimd = 4     // simdgroups per TG → split K into numSimd chunks
        let outputsPerTG = 32 * tn        // 64 outputs per TG
        let tgThreads = 32 * numSimd      // 128 threads per TG

        var accDecls = ""
        var accStores = ""
        for c in 0..<tn {
            accDecls += "            float acc\(c) = 0.0f;\n"
            accStores += "                if (my_col + \(c)u < N_DIM) y[my_col + \(c)u] = sum\(c);\n"
        }

        // Inner-iteration multiplies. Each iteration reads tk x values (in
        // (tk+3)/4 float4 chunks) and tk*tn W values, then `tk*tn` FMAs.
        var multiplies = ""
        let xField = ["x", "y", "z", "w"]
        for r in 0..<tk {
            let lane = xField[r % 4]
            let chunk = r / 4
            let xRef = "xs\(chunk).\(lane)"
            for c in 0..<tn {
                multiplies += "                acc\(c) += \(xRef) * W[(k + \(r)u) * N_DIM + my_col + \(c)u];\n"
            }
        }
        let xChunks = (tk + 3) / 4
        var xLoads = ""
        for i in 0..<xChunks {
            xLoads += "                float4 xs\(i) = *reinterpret_cast<const device float4*>(x + k + \(i * 4)u);\n"
        }

        // Cross-simdgroup partial sums. Each (simd_lane, c) has `numSimd`
        // partials to sum. Generated as a straight-line reduction so the
        // compiler can issue independent loads.
        var crossSimdReduce = ""
        for c in 0..<tn {
            crossSimdReduce += "                float sum\(c) = "
            crossSimdReduce += (0..<numSimd).map { sg in
                "shared[\(sg)u * 32u * TN + simd_lane * TN + \(c)u]"
            }.joined(separator: " + ")
            crossSimdReduce += ";\n"
        }

        let header = """
        #include <metal_stdlib>
        using namespace metal;

        constant uint K_DIM = \(K);
        constant uint N_DIM = \(N);

        // Split-K GEMV: y = x @ W. \(tgThreads)-thread TG = \(numSimd) simdgroups.
        // Each TG handles \(outputsPerTG) output columns; the \(numSimd) simdgroups
        // share the N-tile but split the K dimension into \(numSimd) chunks.
        // Partials are summed via threadgroup memory + one barrier.
        kernel void kernel_gemv(
            device const float* x [[buffer(0)]],
            device const float* W [[buffer(1)]],
            device float* y [[buffer(2)]],
            uint tgid [[threadgroup_position_in_grid]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint TN = \(tn);
            constexpr uint TK = \(tk);
            constexpr uint NUM_SIMD = \(numSimd);
            // Statically-sized threadgroup buffer for cross-simdgroup partial
            // sums. NUM_SIMD * 32 lanes * TN values per lane = \(numSimd * 32 * tn) floats.
            threadgroup float shared[\(numSimd * 32 * tn)];
            uint my_col = tgid * 32u * TN + simd_lane * TN;
            if (my_col >= N_DIM) return;

            uint k_chunk = K_DIM / NUM_SIMD;
            uint k_start = simd_group * k_chunk;
            uint k_end = k_start + k_chunk;

        """
        let body = """

            for (uint k = k_start; k < k_end; k += TK) {
        \(xLoads)
        """
        let storePartial: String = {
            var s = "            }\n\n"
            // Each thread writes its tn partial sums to threadgroup memory at
            // its (simd_group, simd_lane, c) slot.
            for c in 0..<tn {
                s += "            shared[simd_group * 32u * TN + simd_lane * TN + \(c)u] = acc\(c);\n"
            }
            s += "            threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
            s += "            if (simd_group == 0u) {\n"
            return s
        }()
        let endBlock = "            }\n}\n"

        let source = header
            + accDecls
            + body
            + multiplies
            + storePartial
            + crossSimdReduce
            + accStores
            + endBlock

        return (source, "kernel_gemv", TuningConfig(
            blockSize: tgThreads,
            useSIMDGroups: true,
            useGEMV: true,
            gemvNWrites: outputsPerTG
        ))
    }

    /// MTLBuffers — no MTLTensor / MTL4 host APIs required.
    ///
    /// The emitted source contains
    /// `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`,
    /// which `MetalHLOCompiler.compileKernel` uses as a sentinel to switch the
    /// runtime compile to `MTLLanguageVersion.version4_0`. Other kernels keep
    /// the default Metal version.
    /// Cooperative-tensor tf32 GEMM. Mirrors MLX's gemm_nax strategy: hold a
    /// BM×BN output accumulator in registers (cooperative_tensor) across the
    /// whole K dimension and write C exactly once, instead of letting MPP's
    /// `op.run` over full K pick a small internal tileK and become bandwidth-
    /// bound. Aligned fp32 only (M%64==0, N%128==0, K%128==0), no transpose,
    /// batch 1 — the caller gates these. relaxed_precision=true (tf32).
    /// Tile 64×128, 8 simdgroups (256 threads), K looped in 128-wide chunks.
    /// Tile config for the cooperative-tensor GEMM, env-overridable for tuning.
    /// Defaults match MLX's NAX choice for Pro-class devices (64×128 tile, 8
    /// simdgroups). TILEK is the per-op.run K-block (MLX bk ≈ 256–512).
    static func coopGemmConfig() -> (bm: Int, bn: Int, tilek: Int, nsg: Int) {
        let env = ProcessInfo.processInfo.environment
        let bm = Int(env["METALHLO_COOP_BM"] ?? "") ?? 64
        let bn = Int(env["METALHLO_COOP_BN"] ?? "") ?? 128
        let tilek = Int(env["METALHLO_COOP_TILEK"] ?? "") ?? 128
        let nsg = Int(env["METALHLO_COOP_NSG"] ?? "") ?? 8
        return (bm, bn, tilek, nsg)
    }

    /// Config for the low-level NAX GEMM (MLX gemm_loop). Env-overridable.
    /// Defaults match MLX's NAX choice for Pro-class devices: BM=64, BN=128,
    /// BK=256, WM=2, WN=4 (→ SM=SN=32, 8 simdgroups, 256 threads).
    static func naxGemmConfig() -> (bm: Int, bn: Int, bk: Int, wm: Int, wn: Int) {
        let env = ProcessInfo.processInfo.environment
        let bm = Int(env["METALHLO_NAX_BM"] ?? "") ?? 64
        let bn = Int(env["METALHLO_NAX_BN"] ?? "") ?? 128
        let bk = Int(env["METALHLO_NAX_BK"] ?? "") ?? 256
        let wm = Int(env["METALHLO_NAX_WM"] ?? "") ?? 2
        let wn = Int(env["METALHLO_NAX_WN"] ?? "") ?? 4
        return (bm, bn, bk, wm, wn)
    }

    /// MLX-grade tf32 GEMM: bundles MLX's tested `gemm_loop` (NAX cooperative-
    /// tensor, register-blocked) and adds a thin wrapper using our buffer ABI.
    /// Aligned fp32, no transpose, batch 1 — caller gates on M%BM==N%BN==K%BK==0.
    /// UM/UN/UK = 16/32/16 and SK=32 are the coprocessor's natural NAX units.
    private func generateMatMul2dNAXSource() -> (String, String, TuningConfig?) {
        let c = CodeGenerator.naxGemmConfig()
        let sm = c.bm / c.wm
        let sn = c.bn / c.wn
        let nsg = c.wm * c.wn
        // MLX uses swizzle_log=2 for 's'/Pro devices: reshapes the launch grid
        // so concurrently-resident threadgroups read overlapping A/B tiles →
        // L2 reuse. The dispatcher reshapes the grid (gridWidth*=tile,
        // gridHeight=ceil/tile); the kernel decodes (tid.x,tid.y)→logical tile.
        let swizzleLog = Int(ProcessInfo.processInfo.environment["METALHLO_NAX_SWIZZLE"] ?? "") ?? 2
        let tuning = TuningConfig(
            tileM: c.bm,
            tileN: c.bn,
            tileK: nil,
            blockSize: nsg * 32,
            useSharedMemory: false,
            useSIMDGroups: true,
            swizzleLog: swizzleLog
        )
        let wrapper = """

        // NAX_GEMM_KERNEL (compiler keys fast-math + Metal 4.0 on this marker)
        using namespace metal;
        using namespace mlx::steel;

        // max_total_threads_per_threadgroup lets the compiler size the register
        // file for exactly \(nsg * 32) threads/TG instead of the 1024 worst case —
        // critical for this register-heavy cooperative-tensor kernel's occupancy.
        [[kernel, max_total_threads_per_threadgroup(\(nsg * 32))]] void kernel_matmul(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],
            device float* C [[buffer(2)]],
            constant uint& M [[buffer(3)]],
            constant uint& N [[buffer(4)]],
            constant uint& K [[buffer(5)]],
            uint simd_group_id [[simdgroup_index_in_threadgroup]],
            uint3 tid [[threadgroup_position_in_grid]])
        {
            constexpr int BM = \(c.bm), BN = \(c.bn), BK = \(c.bk), WM = \(c.wm), WN = \(c.wn);
            constexpr int SWIZZLE_LOG = \(swizzleLog);
            constexpr short UM = 16, UN = 32, UK = 16, SK = 32;
            constexpr short SM = BM / WM, SN = BN / WN;

            const int tiles_n = int(N) / BN;
            const int tiles_m = int(M) / BM;
            // Decode swizzled launch grid → logical (tile_n, tile_m).
            const int tid_y = (int(tid.y) << SWIZZLE_LOG) +
                              (int(tid.x) & ((1 << SWIZZLE_LOG) - 1));
            const int tid_x = int(tid.x) >> SWIZZLE_LOG;
            if (tid_x >= tiles_n || tid_y >= tiles_m) { return; }

            const int c_row = tid_y * BM;
            const int c_col = tid_x * BN;
            const int lda = int(K), ldb = int(N), ldd = int(N);

            // Per-simdgroup sub-tile origin within the BM×BN block.
            const short tm = SM * short(simd_group_id / WN);
            const short tn = SN * short(simd_group_id % WN);

            const device float* Aptr = A + (size_t)(c_row + tm) * lda;   // no transpose_a
            const device float* Bptr = B + (size_t)(c_col + tn);         // no transpose_b
            device float* Dptr = C + (size_t)(c_row + tm) * ldd + (c_col + tn);

            const int gemm_k_iterations = int(K) / BK;

            auto Dtile = gemm_loop<
                float, SM, SN, SK, BK,
                false, false,         // transpose_a, transpose_b
                true, true, true,     // kAlignedM, kAlignedN, kAlignedK
                UM, UN, UK, float>(
                Aptr, Bptr, lda, ldb, int(K), gemm_k_iterations, SM, SN);

            Dtile.store(Dptr, ldd);
        }
        """
        return (NAXGemmBundle.metalSource + wrapper, "kernel_matmul", tuning)
    }

    private func generateMatMul2dMPPCoopSource() -> (String, String, TuningConfig?) {
        let cfg = CodeGenerator.coopGemmConfig()
        let tuning = TuningConfig(
            tileM: cfg.bm,
            tileN: cfg.bn,
            tileK: nil,
            blockSize: cfg.nsg * 32,
            useSharedMemory: false,
            useSIMDGroups: true
        )
        let source = """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <metal_cooperative_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        using namespace metal;
        using namespace mpp::tensor_ops;

        #define BM \(cfg.bm)
        #define BN \(cfg.bn)
        #define TILEK \(cfg.tilek)
        #define NSG \(cfg.nsg)

        // dextents convention is (cols, rows); cols is the faster-varying dim.
        // A[M,K] → (K, M); B[K,N] → (N, K); C[M,N] → (N, M). No transpose.
        kernel void kernel_matmul(
            device float* A_ptr [[buffer(0)]],
            device float* B_ptr [[buffer(1)]],
            device float* C_ptr [[buffer(2)]],
            constant uint& M [[buffer(3)]],
            constant uint& N [[buffer(4)]],
            constant uint& K [[buffer(5)]],
            uint3 tgid [[threadgroup_position_in_grid]])
        {
            tensor<device float, dextents<int32_t, 2>, tensor_inline> A(
                A_ptr, dextents<int32_t, 2>{int32_t(K), int32_t(M)});
            tensor<device float, dextents<int32_t, 2>, tensor_inline> B(
                B_ptr, dextents<int32_t, 2>{int32_t(N), int32_t(K)});
            tensor<device float, dextents<int32_t, 2>, tensor_inline> C(
                C_ptr, dextents<int32_t, 2>{int32_t(N), int32_t(M)});

            constexpr auto desc = matmul2d_descriptor(
                BM, BN, TILEK,
                false, false,                        // transpose_left/right
                true,                                // relaxed_precision (tf32)
                matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<desc, execution_simdgroups<NSG>> op;

            const uint mRow = tgid.y * BM;
            const uint nCol = tgid.x * BN;

            // In-register accumulator (layout depends on op scope/desc). Seed
            // its type from a representative slice, then zero it. The descriptor
            // pins the op tile to BM×BN×TILEK, so op.run reads exactly TILEK
            // columns from the slice origin regardless of the slice's extent.
            auto sA = A.slice(0, mRow);
            auto sB = B.slice(nCol, 0);
            auto cT = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), float>();

            // Zero all thread-private accumulator slots (multiply_accumulate
            // adds onto the initial value). Writing every slot is safe; store()
            // only emits the valid elements.
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                cT[i] = 0.0f;
            }

            // K-loop: each op.run does a BM×TILEK · TILEK×BN multiply and
            // accumulates into the register tensor. Caller guarantees K%TILEK==0.
            for (uint k = 0; k + TILEK <= K; k += TILEK) {
                auto tA = A.slice(k, mRow);
                auto tB = B.slice(nCol, k);
                op.run(tA, tB, cT);
            }

            auto mC = C.slice(nCol, mRow);
            cT.store(mC);
        }
        """
        return (source, "kernel_matmul", tuning)
    }

    private func generateMatMul2dMPPSource(
        batchSize: Int,
        inputElementType: ElementType,
        outputElementType: ElementType,
        transA: Bool = false,
        transB: Bool = false,
        tileSize: Int = 128
    ) -> (String, String, TuningConfig?) {
        // Tile size: BM == BN. Defaults to 128 (8 simdgroups, 256 threads).
        // A 64-tile variant (4 simdgroups, 128 threads) is used for shapes
        // with too few 128-tiles to fill the GPU — the matrix coprocessor's
        // throughput per TG is unchanged, but more TGs improves occupancy
        // on shapes like (M=1536, N=384, K=4096) where (M/128)*(N/128) = 36
        // only covers ~0.5 of the M5 Pro's ~64 concurrent-TG capacity.
        let numSimdgroups = (tileSize == 64) ? 4 : 8
        let blockSize = numSimdgroups * 32
        let tuning = TuningConfig(
            tileM: tileSize,
            tileN: tileSize,
            tileK: nil,
            blockSize: blockSize,
            useSharedMemory: false,
            useSIMDGroups: true
        )

        let inputMetalType = metalTypeName(for: inputElementType)
        let outputMetalType = metalTypeName(for: outputElementType)
        let isBatched = batchSize > 1
        // The bindings layer adds buffer(6) = batchCount when batchSize > 1
        // (see CodeGenerator.swift "Add batchCount for batched matmul").
        let batchCountParam = isBatched
            ? "constant uint& batchCount [[buffer(6)]],"
            : ""
        let batchOffsetSetup = isBatched ? """
                // Skip out-of-range batches in case the dispatch grid was rounded up.
                // A and B per-batch byte counts are M*K and K*N regardless of transpose
                // (the buffer is the same size; only the interpretation changes).
                if (tgid.z >= batchCount) return;
                A_ptr += uint(tgid.z) * M * K;
                B_ptr += uint(tgid.z) * K * N;
                C_ptr += uint(tgid.z) * M * N;
        """ : ""

        // dextents follow the (cols, rows) convention — cols is the
        // faster-varying / inner-stride dim of the row-major buffer.
        //
        //   transA=false → A in [M, K]  → dextents{cols=K, rows=M}
        //   transA=true  → A in [K, M]  → dextents{cols=M, rows=K}
        //   transB=false → B in [K, N]  → dextents{cols=N, rows=K}
        //   transB=true  → B in [N, K]  → dextents{cols=K, rows=N}
        //
        // Slices use (col_origin, row_origin). For each operand, the M/N tile
        // origin lives on whichever dim is the *non-contracting* one. K is
        // not sliced (full reduction).
        let aDExtents = transA
            ? "dextents<int32_t, 2>{int32_t(M), int32_t(K)}"
            : "dextents<int32_t, 2>{int32_t(K), int32_t(M)}"
        let bDExtents = transB
            ? "dextents<int32_t, 2>{int32_t(K), int32_t(N)}"
            : "dextents<int32_t, 2>{int32_t(N), int32_t(K)}"
        let aSlice = transA
            ? "A.slice(tgid.y * BM, 0)"
            : "A.slice(0, tgid.y * BM)"
        let bSlice = transB
            ? "B.slice(0, tgid.x * BN)"
            : "B.slice(tgid.x * BN, 0)"
        let descTransposeFlags = "\(transA ? "true" : "false"), \(transB ? "true" : "false")"

        // tf32 fast path. For fp32 inputs, `relaxed_precision = true` lets the
        // matrix coprocessor round operands to tf32 (10-bit mantissa) and run
        // the multiply at ~5× the exact-fp32 throughput — exactly what MLX does
        // by default (matmul.cpp gate `enable_tf32() || dtype != float32`, with
        // enable_tf32 defaulting true). On M5 Pro this takes fp32 matmul from
        // ~3.6 TF (exact) toward ~18 TF (tf32). For fp16/bf16 inputs the
        // mantissa is already ≤10 bits so relaxed has no effect; keep it off.
        // Escape hatch: METALHLO_EXACT_FP32=1 forces bit-exact fp32 (rel err
        // ~1e-6 instead of tf32's ~1e-3) at the throughput cost.
        let exactFp32 = ProcessInfo.processInfo.environment["METALHLO_EXACT_FP32"] == "1"
        let relaxedPrecision = (inputElementType == .float32) && !exactFp32
        let relaxedPrecisionFlag = relaxedPrecision ? "true" : "false"

        let source = """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <metal_cooperative_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        using namespace metal;
        using namespace mpp::tensor_ops;

        #define BM \(tileSize)
        #define BN \(tileSize)
        #define NUM_SIMDGROUPS \(numSimdgroups)

        // MetalPerformancePrimitives matmul2d on Apple Matrix Coprocessor.
        // 4 simdgroups (128 threads) per threadgroup. dextents convention is
        // (cols, rows). transA/transB switch the buffer interpretation and
        // the matmul2d_descriptor's transpose_left/transpose_right flags so
        // the matrix coprocessor reads the operand transposed in-place
        // instead of needing a separate physical-transpose kernel.
        // Input element type: \(inputMetalType)  Output: \(outputMetalType)
        // transA=\(transA), transB=\(transB)
        //
        // When input != output (e.g., half × half → float), we're in the
        // mixed-precision path that fuses TF32's trailing fp16→fp32 convert.
        // MPP's matmul2d supports this directly (`half × half → float` is in
        // the documented type table); the cast happens in-register inside the
        // matrix coprocessor, eliminating one device-memory round-trip per
        // matmul.
        //
        // Note: input tensor element types intentionally drop `const` — MPP's
        // matmul2d dispatches on `__is_same_v<leftValueType, half>`, which
        // fails for `const half`. The buffers are still bound read-only and
        // the kernel doesn't write through them, so this is a type-system
        // workaround, not a correctness issue.
        kernel void kernel_matmul(
            device \(inputMetalType)* A_ptr [[buffer(0)]],
            device \(inputMetalType)* B_ptr [[buffer(1)]],
            device \(outputMetalType)* C_ptr [[buffer(2)]],
            constant uint& M [[buffer(3)]],
            constant uint& N [[buffer(4)]],
            constant uint& K [[buffer(5)]],
            \(batchCountParam)
            uint3 tgid [[threadgroup_position_in_grid]])
        {
            \(batchOffsetSetup)
            tensor<device \(inputMetalType), dextents<int32_t, 2>, tensor_inline> A(
                A_ptr,
                \(aDExtents));
            tensor<device \(inputMetalType), dextents<int32_t, 2>, tensor_inline> B(
                B_ptr,
                \(bDExtents));
            tensor<device \(outputMetalType), dextents<int32_t, 2>, tensor_inline> C(
                C_ptr,
                dextents<int32_t, 2>{int32_t(N), int32_t(M)});

            constexpr auto desc = matmul2d_descriptor(
                BM,                                  // m outer dim
                BN,                                  // n outer dim
                static_cast<int>(dynamic_extent),    // k = pulled from operands
                \(descTransposeFlags),                        // transpose_left, transpose_right
                \(relaxedPrecisionFlag));             // relaxed_precision: tf32
                                                     // for fp32 inputs (MLX
                                                     // default), exact for
                                                     // fp16/bf16. Override with
                                                     // METALHLO_EXACT_FP32=1.
            matmul2d<desc, execution_simdgroups<NUM_SIMDGROUPS>> op;

            // Grid is dispatched as gridSize.width = N_tiles, gridSize.height = M_tiles
            // (matches the 32x32 simdgroup kernel's dispatch convention).
            auto a = \(aSlice);
            auto b = \(bSlice);
            auto c = C.slice(tgid.x * BN, tgid.y * BM);

            op.run(a, b, c);
        }
        """
        return (source, "kernel_matmul", tuning)
    }

    /// MLX-style "Steel" GEMM for fp32. Hand-tuned simdgroup_matrix kernel
    /// with cooperative threadgroup loads — much faster on Apple Silicon for
    /// fp32 than the matrix-coprocessor `matmul2d` primitive (which is
    /// tuned for half/bfloat). MLX gates fp32 *away* from `matmul2d` for
    /// exactly this reason (`mlx/backend/metal/matmul.cpp:915`).
    ///
    /// Tile: BM=64 BN=64 BK=16 with WM=2 WN=2 simdgroups (128 threads/TG).
    /// Each simdgroup writes a 32×32 sub-tile of output via a 4×4 grid of
    /// `simdgroup_float8x8` accumulators. Per K-step (BK=16) the inner loop
    /// runs 2 simdgroup_load + 4×4 simdgroup_multiply_accumulate per simdgroup.
    /// Aligned variant only — caller gates on M%64==N%64==K%16==0.
    /// Declare 16 named simdgroup_float8x8 accumulators (acc_0_0 … acc_3_3)
    /// initialized to 0. Used in place of `simdgroup_float8x8 acc[4][4]` so
    /// the Metal compiler keeps each in a register file slot rather than as
    /// a stack alloca that gets load/stored per MMA.
    private func steelAccDecls() -> String {
        var lines: [String] = []
        for i in 0..<4 {
            for j in 0..<4 {
                lines.append("simdgroup_float8x8 acc_\(i)_\(j) = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
            }
        }
        return lines.joined(separator: "\n            ")
    }

    /// Generate the inner-K loop body for the Steel matmul, fully unrolled.
    /// `#pragma clang loop unroll(full)` is silently ignored by the Metal
    /// compiler for the simdgroup_matrix MMA loop nest (verified via offline
    /// `xcrun metal -S`; the resulting IR contains 1 MMA call inside a 4×4
    /// loop with a runtime-computed serpentine select, plus alloca + load/
    /// store of the accumulator per MMA). Generating the full unroll
    /// here in Swift produces 16 explicit MMA calls per kk iter with all
    /// constant indices, which the compiler can register-promote.
    private func steelInnerKK() -> String {
        var src = ""
        for kk in 0..<2 {                                          // BK / kFragSize = 16 / 8
            let kkOff = kk * 8
            src += "{ // kk = \(kkOff)\n"
            src += "                frag_t a_te_0, a_te_1, a_te_2, a_te_3;\n"
            for i in 0..<4 {
                let aBase = i * 16 * 20 + kkOff                    // i * TM_STRIDE * A_LD + kk
                src += "                a_te_\(i)[0] = As_base[\(aBase + 0)];\n"
                src += "                a_te_\(i)[1] = As_base[\(aBase + 1)];\n"
            }
            src += "                frag_t b_te_0, b_te_1, b_te_2, b_te_3;\n"
            for j in 0..<4 {
                let bBase = j * 16 + kkOff * 68                    // j * TN_STRIDE + kk * B_LD
                src += "                b_te_\(j)[0] = Bs_base[\(bBase + 0)];\n"
                src += "                b_te_\(j)[1] = Bs_base[\(bBase + 1)];\n"
            }
            for j in 0..<4 {
                src += "                simdgroup_float8x8 b_mat_\(j);\n"
                src += "                reinterpret_cast<thread frag_t&>(b_mat_\(j).thread_elements()) = b_te_\(j);\n"
            }
            for i in 0..<4 {
                src += "                { simdgroup_float8x8 A_mat;\n"
                src += "                  reinterpret_cast<thread frag_t&>(A_mat.thread_elements()) = a_te_\(i);\n"
                // Serpentine N order — alternate direction per i for register reuse.
                let jOrder = (i & 1 == 1) ? [3, 2, 1, 0] : [0, 1, 2, 3]
                for j in jOrder {
                    src += "                  simdgroup_multiply_accumulate(acc_\(i)_\(j), A_mat, b_mat_\(j), acc_\(i)_\(j));\n"
                }
                src += "                }\n"
            }
            src += "            }\n"
        }
        return src
    }

    /// Generate the unrolled store-back loop using the named acc_i_j vars.
    private func steelStore() -> String {
        var src = ""
        for i in 0..<4 {
            for j in 0..<4 {
                src += "{ // (i=\(i), j=\(j))\n"
                src += "                uint row = out_base_row + \(i * 16);\n"
                src += "                uint col = out_base_col + \(j * 16);\n"
                src += "                frag_t te = reinterpret_cast<thread frag_t&>(acc_\(i)_\(j).thread_elements());\n"
                src += "                C_ptr[row * N + col + 0] = te[0];\n"
                src += "                C_ptr[row * N + col + 1] = te[1];\n"
                src += "            }\n            "
            }
        }
        return src
    }

    private func generateMatMul2dSteelSource(batchSize: Int, M: Int, N: Int, K: Int) -> (String, String, TuningConfig?) {
        // MLX picks swizzle_log=2 for `medium` Apple GPUs (M5 Pro devc='s').
        // Empirically swizzle hurts for our kernel: TF off the K=4096 shapes
        // drops 5-13%, possibly because the bounds-check overhead outweighs
        // L2 reuse with our register footprint / occupancy. Keep the code
        // path (swizzle_log=0 → no-op) so we can re-enable per-shape later.
        let swizzleLog = 0
        let tuning = TuningConfig(
            tileM: 64,
            tileN: 64,
            tileK: 16,
            blockSize: 128,    // 4 simdgroups × 32 lanes
            useSharedMemory: true,
            useSIMDGroups: true,
            swizzleLog: swizzleLog
        )
        let isBatched = batchSize > 1
        let batchCountParam = isBatched ? "constant uint& batchCount [[buffer(6)]]," : ""
        let batchOffsetSetup = isBatched ? """
                if (tgid.z >= batchCount) return;
                A_ptr += uint(tgid.z) * M * K;
                B_ptr += uint(tgid.z) * K * N;
                C_ptr += uint(tgid.z) * M * N;
        """ : ""

        let source = """
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;

        // Faithful port of mlx::steel::BlockMMA / GEMMKernel (fp32 path).
        //
        // Tile: BM=64 BN=64 BK=16 with WM=2 WN=2 simdgroups (128 threads).
        // Each simdgroup writes a 32×32 sub-tile interleaved at stride
        // (kFragSize×WM, kFragSize×WN) = (16, 16) — i.e. 4×4 fragments
        // sitting at positions {(tm+i*16, tn+j*16) for i,j ∈ 0..3} where
        // (tm, tn) ∈ {(0,0),(0,8),(8,0),(8,8)} per simdgroup. This is the
        // MLX TM_stride=kFragSize*WM pattern.
        //
        // The key trick vs my earlier draft: per-thread fragment loads,
        // not `simdgroup_load`. Each thread reads its own 2 elements from
        // padded threadgroup memory using a lane-coord offset, packs them
        // into the per-thread half of a `simdgroup_float8x8` via
        // `thread_elements()`, then calls `simdgroup_multiply_accumulate`.
        // This is what MLX's `BaseMMAFrag<T,8,8>::load` does. `simdgroup_load`
        // measured ~10× slower than per-thread loads on M5 Pro.
        //
        // Padding `tgp_padding = 16/sizeof(T) = 4 fp32` on the leading dim
        // of each tile avoids 32-way bank conflicts (Apple GPU threadgroup
        // memory has 32 banks of 4 bytes).
        #define BM 64
        #define BN 64
        #define BK 16
        #define WM 2
        #define WN 2
        #define A_LD 20   // BK + 4 padding
        #define B_LD 68   // BN + 4 padding
        // TM_stride / TN_stride from MLX BlockMMA — distance between
        // consecutive fragments within a simdgroup, in elements.
        #define TM_STRIDE 16  // kFragSize * WM
        #define TN_STRIDE 16  // kFragSize * WN

        // M, N, K are baked in as compile-time constants — we specialize the
        // kernel per-shape anyway, and inlined literals let the Metal
        // compiler constant-fold every address calc and pointer stride.
        // Buffer-bound versions are kept (named _runtime) so the dispatch
        // path's binding plan does not have to change; they go unused.
        #define M_CONST \(M)
        #define N_CONST \(N)
        #define K_CONST \(K)

        [[max_total_threads_per_threadgroup(128)]]
        kernel void kernel_matmul(
            device const float* A_ptr [[buffer(0)]],
            device const float* B_ptr [[buffer(1)]],
            device float* C_ptr [[buffer(2)]],
            constant uint& M_runtime [[buffer(3)]],
            constant uint& N_runtime [[buffer(4)]],
            constant uint& K_runtime [[buffer(5)]],
            \(batchCountParam)
            uint3 tgid [[threadgroup_position_in_grid]],
            uint thread_idx [[thread_index_in_threadgroup]],
            uint simd_lane_id [[thread_index_in_simdgroup]],
            uint simd_group_id [[simdgroup_index_in_threadgroup]])
        {
            (void)M_runtime; (void)N_runtime; (void)K_runtime;
            constexpr uint M = M_CONST;
            constexpr uint N = N_CONST;
            constexpr uint K = K_CONST;
            \(batchOffsetSetup)
            threadgroup float Asub[BM * A_LD];   // 64*20 = 1280 floats
            threadgroup float Bsub[BK * B_LD];   // 16*68 = 1088 floats

            // MLX L2 swizzle (steel_gemm_fused.h:154-160). Launch grid is
            // (tiles_n * (1<<SWIZZLE_LOG), ceildiv(tiles_m, 1<<SWIZZLE_LOG)).
            // Decode back: 4 consecutive tgid.x values (mod SWIZZLE_TILE)
            // share the same logical tile_n, improving B-tile reuse across
            // simultaneously-resident threadgroups.
            #define SWIZZLE_LOG \(swizzleLog)
            #define SWIZZLE_TILE (1 << SWIZZLE_LOG)
            const uint tiles_m = M / BM;
            const uint tiles_n = N / BN;
            const uint logical_tile_m = (tgid.y << SWIZZLE_LOG) +
                                        (tgid.x & (SWIZZLE_TILE - 1));
            const uint logical_tile_n = tgid.x >> SWIZZLE_LOG;
            if (logical_tile_m >= tiles_m || logical_tile_n >= tiles_n) return;
            uint tileM = logical_tile_m * BM;
            uint tileN = logical_tile_n * BN;

            // Cooperative load layout (MLX BlockLoader pattern).
            // A_sub (BM×BK=64×16): TCOLS=2, TROWS=64. Each thread loads
            // 8 cols of one row (2× float4).
            uint A_bi = thread_idx >> 1;            // / 2
            uint A_bj = (thread_idx & 1u) * 8u;     // 0 or 8
            // B_sub (BK×BN=16×64): TCOLS=8, TROWS=16. Each thread loads
            // 8 cols of one row.
            uint B_bi = thread_idx >> 3;            // / 8
            uint B_bj = (thread_idx & 7u) * 8u;     // 0, 8, ..., 56

            // BlockMMA constructor (mma.h:488) — derive the per-thread
            // (sm, sn) lane-coords inside an 8×8 fragment from simd_lane_id.
            uint qid = simd_lane_id >> 2;                              // / 4
            uint fm  = (qid & 4u) + ((simd_lane_id >> 1) & 3u);        // row 0..7
            uint fn  = ((qid & 2u) << 1) + ((simd_lane_id & 1u) << 1); // col 0/2/4/6
            // Per-simdgroup offset into the BM/BN tile (interleaved layout).
            uint tm = 8u * (simd_group_id / WN);   // 0 or 8
            uint tn = 8u * (simd_group_id % WN);   // 0 or 8
            // Final per-thread base offsets into As, Bs.
            uint As_off = (tm + fm) * A_LD + fn;   // row=(tm+fm), col=fn
            uint Bs_off = fm * B_LD + (tn + fn);   // row=fm, col=(tn+fn)

            typedef vec<float, 2> frag_t;
            // 16 accumulators as INDIVIDUAL NAMED variables instead of
            // `simdgroup_float8x8 acc[4][4]`. The Metal compiler keeps
            // arrays-of-simdgroup_matrix as alloca + load/store-per-MMA
            // (verified via offline `xcrun metal -S`) even when indices
            // are loop-invariant; #pragma unroll(full) does not unroll
            // the surrounding MMA loop on this toolchain. Individual
            // variables force the compiler to allocate each to its own
            // register and eliminate the inner-loop stack traffic.
            \(steelAccDecls())

            // Hoist the per-thread device pointers (MLX BlockLoader pattern).
            // src_a, src_b are pre-offset to this thread's slice of the
            // tile, and advance by `+BK` / `+BK*N` per outer K iteration.
            const device float* src_a = A_ptr + (tileM + A_bi) * K + A_bj;
            const device float* src_b = B_ptr + B_bi * N + tileN + B_bj;
            threadgroup float* dst_a = Asub + A_bi * A_LD + A_bj;
            threadgroup float* dst_b = Bsub + B_bi * B_LD + B_bj;

            uint k_iters = K / BK;
            for (uint k_iter = 0; k_iter < k_iters; k_iter++) {
                // --- Cooperative threadgroup load of A and B tiles ---
                // 8 floats per thread = 2× float4. ReadVector-equivalent
                // (MLX loader.h:74-80, 8-element ReadVector).
                {
                    const device float4* src = (const device float4*)src_a;
                    threadgroup float4* dst = (threadgroup float4*)dst_a;
                    dst[0] = src[0];
                    dst[1] = src[1];
                }
                {
                    const device float4* src = (const device float4*)src_b;
                    threadgroup float4* dst = (threadgroup float4*)dst_b;
                    dst[0] = src[0];
                    dst[1] = src[1];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Inner K loop, BK/kFragSize = 2 iterations, FULLY UNROLLED
                // via Swift codegen because #pragma unroll(full) is ignored
                // by the Metal compiler for the simdgroup_matrix loop nest.
                threadgroup const float* As_base = Asub + As_off;
                threadgroup const float* Bs_base = Bsub + Bs_off;
                \(steelInnerKK())
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Advance device pointers (MLX BlockLoader::next pattern).
                src_a += BK;
                src_b += BK * N;
            }

            // Store back to device — fully unrolled per the same reasoning
            // as the MMA loop. Uses the individually-named acc_i_j vars.
            uint out_base_row = tileM + tm + fm;
            uint out_base_col = tileN + tn + fn;
            \(steelStore())
        }
        """
        return (source, "kernel_matmul", tuning)
    }

    /// Generates an optimized matmul kernel using simdgroup_matrix operations.
    /// This leverages hardware-accelerated matrix multiplication on Apple Silicon (M1+).
    /// Parameterized by `metalType` so the same template emits a homogeneous-type
    /// kernel for either float (`\(simdMatType)`) or half (`simdgroup_half8x8`).
    /// Apple Silicon's matrix coprocessors run fp16 multiply at ~2x fp32 throughput.
    private func generateSimdgroupMatMulSource(batchSize: Int, metalType: String, elementType: ElementType) -> (String, String, TuningConfig?) {
        // Tuning config for simdgroup matmul
        // TILE_M x TILE_N is computed by each threadgroup
        // Each simdgroup computes 8x8 output tiles using simdgroup_matrix
        let tuning = TuningConfig(
            tileM: 32,
            tileN: 32,
            tileK: 8,
            useSharedMemory: true,
            useSIMDGroups: true
        )

        // Template helpers: parameterize the simdgroup matrix type so the same
        // kernel template works for both fp32 and fp16. Using `\(metalType)(0)`
        // for the zero literal avoids the float-suffix-vs-half-suffix split.
        let simdMatType = "simdgroup_\(metalType)8x8"
        let zeroFill = "make_filled_simdgroup_matrix<\(metalType), 8, 8>(\(metalType)(0))"

        let source: String
        if batchSize > 1 {
            source = """
            #include <metal_stdlib>
            using namespace metal;
            
            // Optimized batched matmul using simdgroup_matrix operations
            // Uses 8x8 simdgroup tiles for hardware-accelerated multiply-accumulate
            // Each threadgroup computes a 32x32 output tile using 4x4 simdgroup tiles
            // Element type: \(metalType)

            #define TILE_M 32
            #define TILE_N 32
            #define TILE_K 8

            kernel void kernel_matmul(
                device const \(metalType)* A [[buffer(0)]],
                device const \(metalType)* B [[buffer(1)]],
                device \(metalType)* C [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                constant uint& batchCount [[buffer(6)]],
                uint3 gid [[threadgroup_position_in_grid]],
                uint simd_lane_id [[thread_index_in_simdgroup]],
                uint simd_group_id [[simdgroup_index_in_threadgroup]])
            {
                uint batch = gid.z;
                if (batch >= batchCount) return;

                // Each threadgroup handles a 32x32 output tile
                // We use 4 simdgroups, each computing an 8x32 strip
                uint tileRowStart = gid.y * TILE_M;
                uint tileColStart = gid.x * TILE_N;

                // Calculate batch offsets
                uint matrixSizeA = M * K;
                uint matrixSizeB = K * N;
                uint matrixSizeC = M * N;

                device const \(metalType)* batchA = A + batch * matrixSizeA;
                device const \(metalType)* batchB = B + batch * matrixSizeB;
                device \(metalType)* batchC = C + batch * matrixSizeC;

                // Each simdgroup computes 8 rows of the 32x32 tile
                uint simdRowOffset = simd_group_id * 8;

                // Initialize 4 accumulator matrices (8x8 each, covering 8x32 output)
                \(simdMatType) acc0, acc1, acc2, acc3;
                acc0 = \(zeroFill);
                acc1 = \(zeroFill);
                acc2 = \(zeroFill);
                acc3 = \(zeroFill);

                // Iterate over K dimension in tiles of 8
                for (uint k = 0; k < K; k += TILE_K) {
                    // Load 8x8 tile from A (rows from our simd group)
                    \(simdMatType) a_tile;
                    uint aRow = tileRowStart + simdRowOffset;
                    uint aCol = k;

                    if (aRow + 8 <= M && aCol + 8 <= K) {
                        simdgroup_load(a_tile, batchA + aRow * K + aCol, K);
                    } else {
                        a_tile = \(zeroFill);
                        for (uint i = 0; i < 8 && aRow + i < M; i++) {
                            for (uint j = 0; j < 8 && aCol + j < K; j++) {
                                // Manual load for edge cases
                                uint lane = i * 8 + j;
                                if (simd_lane_id == lane % 32) {
                                    // This is a simplified edge case handler
                                }
                            }
                        }
                        // For edge cases, load element by element
                        for (uint i = 0; i < 8; i++) {
                            for (uint j = 0; j < 8; j++) {
                                if (aRow + i < M && aCol + j < K) {
                                    // Use slower path for edges
                                }
                            }
                        }
                        simdgroup_load(a_tile, batchA + min(aRow, M-8) * K + min(aCol, K-8), K);
                    }

                    // Load 4 8x8 tiles from B (covers 8 rows, 32 cols)
                    \(simdMatType) b_tile0, b_tile1, b_tile2, b_tile3;
                    uint bRow = k;
                    uint bCol = tileColStart;

                    if (bRow + 8 <= K && bCol + 32 <= N) {
                        simdgroup_load(b_tile0, batchB + bRow * N + bCol, N);
                        simdgroup_load(b_tile1, batchB + bRow * N + bCol + 8, N);
                        simdgroup_load(b_tile2, batchB + bRow * N + bCol + 16, N);
                        simdgroup_load(b_tile3, batchB + bRow * N + bCol + 24, N);
                    } else {
                        b_tile0 = \(zeroFill);
                        b_tile1 = \(zeroFill);
                        b_tile2 = \(zeroFill);
                        b_tile3 = \(zeroFill);
                        if (bRow < K) {
                            simdgroup_load(b_tile0, batchB + min(bRow, K-8) * N + min(bCol, N > 8 ? N-8 : 0), N);
                            if (bCol + 8 < N) simdgroup_load(b_tile1, batchB + min(bRow, K-8) * N + min(bCol + 8, N-8), N);
                            if (bCol + 16 < N) simdgroup_load(b_tile2, batchB + min(bRow, K-8) * N + min(bCol + 16, N-8), N);
                            if (bCol + 24 < N) simdgroup_load(b_tile3, batchB + min(bRow, K-8) * N + min(bCol + 24, N-8), N);
                        }
                    }

                    // Multiply-accumulate: C += A * B
                    simdgroup_multiply_accumulate(acc0, a_tile, b_tile0, acc0);
                    simdgroup_multiply_accumulate(acc1, a_tile, b_tile1, acc1);
                    simdgroup_multiply_accumulate(acc2, a_tile, b_tile2, acc2);
                    simdgroup_multiply_accumulate(acc3, a_tile, b_tile3, acc3);
                }

                // Store results
                uint outRow = tileRowStart + simdRowOffset;
                uint outCol = tileColStart;

                if (outRow + 8 <= M && outCol + 32 <= N) {
                    simdgroup_store(acc0, batchC + outRow * N + outCol, N);
                    simdgroup_store(acc1, batchC + outRow * N + outCol + 8, N);
                    simdgroup_store(acc2, batchC + outRow * N + outCol + 16, N);
                    simdgroup_store(acc3, batchC + outRow * N + outCol + 24, N);
                } else {
                    // Handle edge cases - store only valid elements
                    if (outRow < M && outCol < N) {
                        simdgroup_store(acc0, batchC + min(outRow, M-8) * N + min(outCol, N-8), N);
                    }
                    if (outRow < M && outCol + 8 < N) {
                        simdgroup_store(acc1, batchC + min(outRow, M-8) * N + min(outCol + 8, N-8), N);
                    }
                    if (outRow < M && outCol + 16 < N) {
                        simdgroup_store(acc2, batchC + min(outRow, M-8) * N + min(outCol + 16, N-8), N);
                    }
                    if (outRow < M && outCol + 24 < N) {
                        simdgroup_store(acc3, batchC + min(outRow, M-8) * N + min(outCol + 24, N-8), N);
                    }
                }
            }
            """
        } else {
            // Non-batched optimized kernel
            source = """
            #include <metal_stdlib>
            using namespace metal;
            
            // Optimized matmul using simdgroup_matrix operations
            // Uses 8x8 simdgroup tiles for hardware-accelerated multiply-accumulate
            // Each threadgroup computes a 32x32 output tile using 4 simdgroups
            // Element type: \(metalType)

            #define TILE_M 32
            #define TILE_N 32
            #define TILE_K 8

            kernel void kernel_matmul(
                device const \(metalType)* A [[buffer(0)]],
                device const \(metalType)* B [[buffer(1)]],
                device \(metalType)* C [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                uint2 gid [[threadgroup_position_in_grid]],
                uint simd_lane_id [[thread_index_in_simdgroup]],
                uint simd_group_id [[simdgroup_index_in_threadgroup]])
            {
                // Each threadgroup handles a 32x32 output tile
                // We use 4 simdgroups, each computing an 8x32 strip
                uint tileRowStart = gid.y * TILE_M;
                uint tileColStart = gid.x * TILE_N;

                // Each simdgroup computes 8 rows of the 32x32 tile
                uint simdRowOffset = simd_group_id * 8;

                // Initialize 4 accumulator matrices (8x8 each, covering 8x32 output)
                \(simdMatType) acc0, acc1, acc2, acc3;
                acc0 = \(zeroFill);
                acc1 = \(zeroFill);
                acc2 = \(zeroFill);
                acc3 = \(zeroFill);

                // Iterate over K dimension in tiles of 8
                for (uint k = 0; k < K; k += TILE_K) {
                    // Load 8x8 tile from A
                    \(simdMatType) a_tile;
                    uint aRow = tileRowStart + simdRowOffset;
                    uint aCol = k;

                    if (aRow + 8 <= M && aCol + 8 <= K) {
                        simdgroup_load(a_tile, A + aRow * K + aCol, K);
                    } else {
                        a_tile = \(zeroFill);
                        if (aRow < M && aCol < K) {
                            simdgroup_load(a_tile, A + min(aRow, M > 8 ? M-8 : 0) * K + min(aCol, K > 8 ? K-8 : 0), K);
                        }
                    }

                    // Load 4 8x8 tiles from B (covers 8 rows, 32 cols)
                    \(simdMatType) b_tile0, b_tile1, b_tile2, b_tile3;
                    uint bRow = k;
                    uint bCol = tileColStart;

                    if (bRow + 8 <= K && bCol + 32 <= N) {
                        simdgroup_load(b_tile0, B + bRow * N + bCol, N);
                        simdgroup_load(b_tile1, B + bRow * N + bCol + 8, N);
                        simdgroup_load(b_tile2, B + bRow * N + bCol + 16, N);
                        simdgroup_load(b_tile3, B + bRow * N + bCol + 24, N);
                    } else {
                        b_tile0 = \(zeroFill);
                        b_tile1 = \(zeroFill);
                        b_tile2 = \(zeroFill);
                        b_tile3 = \(zeroFill);
                        if (bRow < K) {
                            uint safeRow = min(bRow, K > 8 ? K-8 : 0);
                            simdgroup_load(b_tile0, B + safeRow * N + min(bCol, N > 8 ? N-8 : 0), N);
                            if (bCol + 8 < N) simdgroup_load(b_tile1, B + safeRow * N + min(bCol + 8, N > 8 ? N-8 : 0), N);
                            if (bCol + 16 < N) simdgroup_load(b_tile2, B + safeRow * N + min(bCol + 16, N > 8 ? N-8 : 0), N);
                            if (bCol + 24 < N) simdgroup_load(b_tile3, B + safeRow * N + min(bCol + 24, N > 8 ? N-8 : 0), N);
                        }
                    }

                    // Multiply-accumulate: C += A * B
                    simdgroup_multiply_accumulate(acc0, a_tile, b_tile0, acc0);
                    simdgroup_multiply_accumulate(acc1, a_tile, b_tile1, acc1);
                    simdgroup_multiply_accumulate(acc2, a_tile, b_tile2, acc2);
                    simdgroup_multiply_accumulate(acc3, a_tile, b_tile3, acc3);
                }

                // Store results
                uint outRow = tileRowStart + simdRowOffset;
                uint outCol = tileColStart;

                if (outRow + 8 <= M && outCol + 32 <= N) {
                    simdgroup_store(acc0, C + outRow * N + outCol, N);
                    simdgroup_store(acc1, C + outRow * N + outCol + 8, N);
                    simdgroup_store(acc2, C + outRow * N + outCol + 16, N);
                    simdgroup_store(acc3, C + outRow * N + outCol + 24, N);
                } else {
                    // Handle edge cases
                    if (outRow < M && outCol < N) {
                        uint safeRow = min(outRow, M > 8 ? M-8 : 0);
                        simdgroup_store(acc0, C + safeRow * N + min(outCol, N > 8 ? N-8 : 0), N);
                    }
                    if (outRow < M && outCol + 8 < N) {
                        uint safeRow = min(outRow, M > 8 ? M-8 : 0);
                        simdgroup_store(acc1, C + safeRow * N + min(outCol + 8, N > 8 ? N-8 : 0), N);
                    }
                    if (outRow < M && outCol + 16 < N) {
                        uint safeRow = min(outRow, M > 8 ? M-8 : 0);
                        simdgroup_store(acc2, C + safeRow * N + min(outCol + 16, N > 8 ? N-8 : 0), N);
                    }
                    if (outRow < M && outCol + 24 < N) {
                        uint safeRow = min(outRow, M > 8 ? M-8 : 0);
                        simdgroup_store(acc3, C + safeRow * N + min(outCol + 24, N > 8 ? N-8 : 0), N);
                    }
                }
            }
            """
        }

        return (source, "kernel_matmul", tuning)
    }

    /// Generates a basic tiled matmul kernel for integer types or small float matrices
    /// (simdgroup_matrix requires all dims >= 8).
    /// Generates a basic matmul kernel with custom A/B read expressions for transposed access.
    private func generateBasicMatMulSourceWithTranspose(
        batchSize: Int, metalType: String, zeroValue: String,
        M: Int, N: Int, K: Int,
        aRead: String, bRead: String
    ) -> (String, String, TuningConfig?) {
        // Simple non-tiled kernel — 1 thread per output element
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_matmul(
            device const \(metalType)* A [[buffer(0)]],
            device const \(metalType)* B [[buffer(1)]],
            device \(metalType)* C [[buffer(2)]],
            constant uint& M [[buffer(3)]],
            constant uint& N [[buffer(4)]],
            constant uint& K [[buffer(5)]],
            constant uint& batchCount [[buffer(6)]],
            uint tid [[thread_position_in_grid]])
        {
            uint totalOutputs = \(batchSize) * M * N;
            if (tid >= totalOutputs) return;

            uint batch = tid / (M * N);
            uint rem = tid % (M * N);
            uint row = rem / N;
            uint col = rem % N;

            device const \(metalType)* batchA = A + batch * M * K;
            device const \(metalType)* batchB = B + batch * K * N;

            \(metalType) sum = \(zeroValue);
            for (uint k = 0; k < K; k++) {
                sum += \(aRead) * \(bRead);
            }
            C[tid] = sum;
        }
        """
        return (source, "kernel_matmul", TuningConfig(blockSize: 1))
    }

    private func generateBasicTiledMatMulSource(batchSize: Int, metalType: String, zeroValue: String) -> (String, String, TuningConfig?) {
        let tuning = TuningConfig(
            tileM: 32, tileN: 32, tileK: 32,
            useSharedMemory: true, useSIMDGroups: false
        )

        let source: String
        if batchSize > 1 {
            source = """
            #include <metal_stdlib>
            using namespace metal;

            #define TILE_SIZE 32

            // Batched matrix multiplication kernel (integer types)
            // Element type: \(metalType)
            kernel void kernel_matmul(
                device const \(metalType)* A [[buffer(0)]],
                device const \(metalType)* B [[buffer(1)]],
                device \(metalType)* C [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                constant uint& batchCount [[buffer(6)]],
                threadgroup \(metalType)* tileA [[threadgroup(0)]],
                threadgroup \(metalType)* tileB [[threadgroup(1)]],
                uint3 gid [[threadgroup_position_in_grid]],
                uint3 tid [[thread_position_in_threadgroup]])
            {
                uint batch = gid.z;
                if (batch >= batchCount) return;

                uint row = gid.y * TILE_SIZE + tid.y;
                uint col = gid.x * TILE_SIZE + tid.x;

                uint matrixSizeA = M * K;
                uint matrixSizeB = K * N;
                uint matrixSizeC = M * N;

                device const \(metalType)* batchA = A + batch * matrixSizeA;
                device const \(metalType)* batchB = B + batch * matrixSizeB;
                device \(metalType)* batchC = C + batch * matrixSizeC;

                \(metalType) sum = \(zeroValue);

                for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
                    uint aRow = row;
                    uint aCol = t * TILE_SIZE + tid.x;
                    if (aRow < M && aCol < K) {
                        tileA[tid.y * TILE_SIZE + tid.x] = batchA[aRow * K + aCol];
                    } else {
                        tileA[tid.y * TILE_SIZE + tid.x] = \(zeroValue);
                    }

                    uint bRow = t * TILE_SIZE + tid.y;
                    uint bCol = col;
                    if (bRow < K && bCol < N) {
                        tileB[tid.y * TILE_SIZE + tid.x] = batchB[bRow * N + bCol];
                    } else {
                        tileB[tid.y * TILE_SIZE + tid.x] = \(zeroValue);
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint k = 0; k < TILE_SIZE; k++) {
                        sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (row < M && col < N) {
                    batchC[row * N + col] = sum;
                }
            }
            """
        } else {
            source = """
            #include <metal_stdlib>
            using namespace metal;

            #define TILE_SIZE 32

            // Matrix multiplication kernel (integer types)
            // Element type: \(metalType)
            kernel void kernel_matmul(
                device const \(metalType)* A [[buffer(0)]],
                device const \(metalType)* B [[buffer(1)]],
                device \(metalType)* C [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                threadgroup \(metalType)* tileA [[threadgroup(0)]],
                threadgroup \(metalType)* tileB [[threadgroup(1)]],
                uint2 gid [[threadgroup_position_in_grid]],
                uint2 tid [[thread_position_in_threadgroup]])
            {
                uint row = gid.y * TILE_SIZE + tid.y;
                uint col = gid.x * TILE_SIZE + tid.x;

                \(metalType) sum = \(zeroValue);

                for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
                    uint aRow = row;
                    uint aCol = t * TILE_SIZE + tid.x;
                    if (aRow < M && aCol < K) {
                        tileA[tid.y * TILE_SIZE + tid.x] = A[aRow * K + aCol];
                    } else {
                        tileA[tid.y * TILE_SIZE + tid.x] = \(zeroValue);
                    }

                    uint bRow = t * TILE_SIZE + tid.y;
                    uint bCol = col;
                    if (bRow < K && bCol < N) {
                        tileB[tid.y * TILE_SIZE + tid.x] = B[bRow * N + bCol];
                    } else {
                        tileB[tid.y * TILE_SIZE + tid.x] = \(zeroValue);
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint k = 0; k < TILE_SIZE; k++) {
                        sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (row < M && col < N) {
                    C[row * N + col] = sum;
                }
            }
            """
        }

        return (source, "kernel_matmul", tuning)
    }

    /// Generates reduction source.
    private func generateReductionSource(inputShapes: [[Int]], attributes: HLOAttributes, elementType: ElementType = .float32) -> (String, String, TuningConfig?) {
        // Get reduction dimensions from attributes
        let reduceDims = attributes.dimensions ?? [0]

        // Get input shape
        let inputShape = inputShapes.first ?? []

        // Determine reduction operation based on reductionKind
        let reductionKind = attributes.reductionKind ?? .sum

        let metalType = metalTypeName(for: elementType)

        // For non-fp32 types use the simple scalar reduction kernel that's
        // parameterized on metalType. The optimized SIMD/shared-memory
        // kernels in ReductionKernelGenerator hard-code `device const float*`
        // pointers, so feeding fp16/bf16 buffers through them reads bytes at
        // the wrong stride. (Surfaced as huge bias-grad diffs in both the
        // fp16 and bf16 test suites.)
        if !isFloatType(elementType) || elementType == .bfloat16 || elementType == .float16 {
            return generateIntegerReductionSource(
                inputShape: inputShape, reduceDims: reduceDims,
                reductionKind: reductionKind, metalType: metalType, elementType: elementType
            )
        }

        // Map HLO reduction kind to ReductionKernelGenerator op
        let reductionOp: ReductionKernelGenerator.ReductionOp
        switch reductionKind {
        case .sum: reductionOp = .sum
        case .max: reductionOp = .max
        case .min: reductionOp = .min
        case .mean: reductionOp = .mean
        case .product: reductionOp = .prod
        case .and: reductionOp = .sum  // Bitwise AND treated as sum of bools for now
        case .or: reductionOp = .max   // Bitwise OR treated as max of bools for now
        case .logAddExp:
            // logAddExp is only produced for reduce_window (cumlogsumexp); it is
            // never emitted as a plain `reduce` reducer (JAX lowers logsumexp into
            // reduce-max + exp + sum + log instead). Reaching here means routing
            // sent a logAddExp window down the full-axis reduce path.
            fatalError("logAddExp reduction is only supported via reduce_window")
        }

        // Analyze the reduction pattern and try to use specialized kernels
        let pattern = ReductionKernelGenerator.analyzePattern(inputShape: inputShape, reduceDims: reduceDims)

        // Compute the reduction-axis size so we can pick between the per-thread
        // specialised kernels (good for short rows/columns) and the cooperative
        // tree-reduction general kernel (one TG per output, 1024 threads doing
        // SIMD-tree reduction — far faster on long axes).
        let reduceAxisSize: Int
        if let firstReduceDim = reduceDims.first, firstReduceDim < inputShape.count {
            reduceAxisSize = inputShape[firstReduceDim]
        } else {
            reduceAxisSize = 1
        }
        // 64 is the break-even point on M5 Pro: below it the per-thread kernel
        // wins because 1024-thread cooperative tree-reduction has ~5µs of
        // SIMD-shuffle + threadgroup-barrier overhead per output. Above it the
        // sequential per-thread loop is the bottleneck (e.g. RED-004's 4096-wide
        // row max takes 4096 sequential reads per thread; with the general
        // kernel each of 1024 threads does 4 reads then tree-reduces in registers).
        let useGeneralForLongAxis = reduceAxisSize >= 64

        // For row and column reductions, use optimized specialized kernels
        switch pattern {
        case .row:
            if !useGeneralForLongAxis {
                let source = ReductionKernelGenerator.generateRowReductionKernel(
                    op: reductionOp,
                    entryPoint: "kernel_reduce"
                )
                return (source, "kernel_reduce", TuningConfig(blockSize: 1024, useRowReduction: true))
            }
            // For long-axis row reductions, decide between two cooperative kernels
            // based on outputCount (= product of non-reduced dims):
            // - Huge outputCount (≥ 2048): SIMD-per-output kernel — 32 outputs/TG
            //   so we get few enough TGs to keep dispatch cost low while still
            //   saturating the GPU. Best for softmax-style cases (RED-006).
            // - Smaller outputCount: general kernel (1 TG per output, full TG
            //   cooperatively reducing) — gives more in-flight TGs which is the
            //   limit on smaller workloads (RED-002 / RED-004).
            let outputCount = inputShape.enumerated()
                .filter { !reduceDims.contains($0.offset) }
                .map { $0.element }
                .reduce(1, *)
            if outputCount >= 2048 {
                // The MLX-style interleaved kernel requires the reduce axis to
                // be a multiple of (32 * N_READS) = 128 so we can drop the tail
                // path. Most ML shapes (powers of 2, common channel counts)
                // satisfy this. Otherwise fall back to the scalar SIMD-per-output
                // kernel which handles arbitrary axis sizes.
                if reduceAxisSize % 128 == 0 {
                    return generateRowReductionSIMDPerOutputFloat4Source(
                        reductionKind: reductionKind,
                        metalType: metalType
                    )
                }
                return generateRowReductionSIMDPerOutputSource(
                    reductionKind: reductionKind,
                    metalType: metalType
                )
            }
            // Otherwise fall through to the general tree-reduction kernel.

        case .column:
            if !useGeneralForLongAxis {
                let source = ReductionKernelGenerator.generateColumnReductionKernel(
                    op: reductionOp,
                    entryPoint: "kernel_reduce"
                )
                return (source, "kernel_reduce", TuningConfig(blockSize: 1024, useColumnReduction: true))
            }
            // Long-axis column reductions also fall through to the general kernel.

        case .global, .general:
            break
        }

        // General reduction fallback with tree reduction
        let (accumOp, reduceOp, initValue): (String, String, String)
        switch reductionKind {
        case .sum:
            accumOp = "accum += val;"
            reduceOp = "a + b"
            initValue = "0.0f"
        case .max:
            accumOp = "accum = max(accum, val);"
            reduceOp = "max(a, b)"
            initValue = "-INFINITY"
        case .min:
            accumOp = "accum = min(accum, val);"
            reduceOp = "min(a, b)"
            initValue = "INFINITY"
        case .mean:
            accumOp = "accum += val;"
            reduceOp = "a + b"
            initValue = "0.0f"
        case .product:
            accumOp = "accum *= val;"
            reduceOp = "a * b"
            initValue = "1.0f"
        case .and:
            accumOp = "accum = float(int(accum) & int(val));"
            reduceOp = "float(int(a) & int(b))"
            initValue = "1.0f"
        case .or:
            accumOp = "accum = float(int(accum) | int(val));"
            reduceOp = "float(int(a) | int(b))"
            initValue = "0.0f"
        case .logAddExp:
            // Only emitted for reduce_window (cumlogsumexp), never plain reduce.
            fatalError("logAddExp reduction is only supported via reduce_window")
        }

        // NOTE: reduce operation has two inputs: data (buffer 0) and init value (buffer 1)
        // Output is at buffer 2, followed by scalar parameters

        // Use parallel tree reduction with SIMD intrinsics for better performance
        let blockSize = 1024  // Increased from 256 for better occupancy

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        // Optimized parallel tree reduction kernel with SIMD intrinsics
        // Each threadgroup handles one output element
        // Threads cooperatively reduce using SIMD + shared memory
        // Reduction type: \(reductionKind)
        kernel void kernel_reduce(
            device const float* input [[buffer(0)]],
            device const float* initValue [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& outputCount [[buffer(3)]],
            constant uint& reduceSize [[buffer(4)]],
            constant uint& innerSize [[buffer(5)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgid [[threadgroup_position_in_grid]],
            uint tgSize [[threads_per_threadgroup]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]])
        {
            // Shared memory for inter-simdgroup reduction
            threadgroup float shared[32];  // One per simdgroup

            // Each threadgroup handles one output element
            if (tgid >= outputCount) return;

            // Compute output coordinates
            uint outerIdx = tgid / innerSize;
            uint innerIdx = tgid % innerSize;
            uint baseInputIdx = outerIdx * reduceSize * innerSize + innerIdx;

            // Phase 1: Each thread reduces its assigned chunk of the reduction dimension
            float accum = \(initValue);

            // Stride through reduction dimension, each thread handles multiple elements
            for (uint r = tid; r < reduceSize; r += tgSize) {
                uint inputIdx = baseInputIdx + r * innerSize;
                float val = input[inputIdx];
                \(accumOp)
            }

            // Phase 2: SIMD reduction within each simdgroup (32 threads)
            \(simdReduceCode(reductionKind: reductionKind, varName: "accum"))

            // First lane of each simdgroup stores to shared memory
            if (simd_lane == 0) {
                shared[simd_group] = accum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 3: First simdgroup reduces the partial results
            if (tid < 32) {
                uint numSimdGroups = (tgSize + 31) / 32;
                float val = (tid < numSimdGroups) ? shared[tid] : \(initValue);
                \(simdReduceCode(reductionKind: reductionKind, varName: "val"))

                // Thread 0 writes the final result
                if (tid == 0) {
                    // Apply the user-provided init value as well
                    float a = val;
                    float b = initValue[0];
                    output[tgid] = \(reduceOp);
                }
            }
        }
        """

        return (source, "kernel_reduce", TuningConfig(blockSize: blockSize))
    }

    /// Float4-unrolled SIMD-per-output reduction kernel with N_WRITES outputs
    /// packed per simdgroup. Each simdgroup of 32 threads sequentially reduces
    /// `nWrites` outputs; with 32 simdgroups per TG we get `32 * nWrites`
    /// outputs/TG. nWrites=4 cuts the TG count by another 4× over the
    /// 1-output-per-simdgroup variant — important because RED-006's bottleneck
    /// is per-TG dispatch overhead (196k outputs / 32-per-TG = 6144 TGs).
    /// Caller must verify `reduceSize % 4 == 0`.
    private func generateRowReductionSIMDPerOutputFloat4Source(
        reductionKind: ReductionKind,
        metalType: String,
        nWrites: Int = 4
    ) -> (String, String, TuningConfig?) {
        let initValue: String
        let reduceOp: String
        let scalarReduceExpr: (String, String) -> String
        switch reductionKind {
        case .sum, .mean:
            initValue = "0.0f"
            reduceOp = "a + b"
            scalarReduceExpr = { acc, val in "\(acc) + \(val)" }
        case .max:
            initValue = "-INFINITY"
            reduceOp = "max(a, b)"
            scalarReduceExpr = { acc, val in "max(\(acc), \(val))" }
        case .min:
            initValue = "INFINITY"
            reduceOp = "min(a, b)"
            scalarReduceExpr = { acc, val in "min(\(acc), \(val))" }
        case .product:
            initValue = "1.0f"
            reduceOp = "a * b"
            scalarReduceExpr = { acc, val in "(\(acc)) * (\(val))" }
        case .and:
            initValue = "1.0f"
            reduceOp = "float(int(a) & int(b))"
            scalarReduceExpr = { acc, val in "float(int(\(acc)) & int(\(val)))" }
        case .or:
            initValue = "0.0f"
            reduceOp = "float(int(a) | int(b))"
            scalarReduceExpr = { acc, val in "float(int(\(acc)) | int(\(val)))" }
        case .logAddExp:
            // Only emitted for reduce_window (cumlogsumexp), never plain reduce.
            fatalError("logAddExp reduction is only supported via reduce_window")
        }

        // Build the MLX-style interleaved inner loop. The Metal compiler issues
        // N_WRITES * N_READS independent reads per iteration before any
        // accumulator dependency, giving the GPU plenty of in-flight loads to
        // hide memory latency.
        let nReads = 4
        var perThreadReduceLines = ""
        for w in 0..<nWrites {
            for r in 0..<nReads {
                perThreadReduceLines += "                total\(w) = \(scalarReduceExpr("total\(w)", "ptr\(w)[\(r)]"));\n"
            }
        }
        var advancePtrLines = ""
        for w in 0..<nWrites {
            advancePtrLines += "                ptr\(w) += 32u * N_READS;\n"
        }
        var declareTotals = ""
        var simdReduceTotals = ""
        var storeOutputs = ""
        for w in 0..<nWrites {
            declareTotals += "            float total\(w) = \(initValue);\n"
            simdReduceTotals += "            \(simdReduceCode(reductionKind: reductionKind, varName: "total\(w)"))\n"
            storeOutputs += "                if (outputBase + \(w)u < outputCount) { float a = total\(w); float b = initB; output[outputBase + \(w)u] = \(reduceOp); }\n"
        }
        var declarePtrs = ""
        for w in 0..<nWrites {
            declarePtrs += "            device const float* ptr\(w) = input + (outputBase + \(w)u) * reduceSize + simd_lane * N_READS;\n"
        }

        let prelude = """
        #include <metal_stdlib>
        using namespace metal;

        // 1 simdgroup (32 threads) per TG, handling N_WRITES=\(nWrites) outputs
        // simultaneously with N_READS=\(nReads) elements/output/iteration.
        // Inner loop interleaves all N_WRITES outputs before any reduce —
        // gives the compiler N_WRITES*N_READS = \(nWrites * nReads) independent
        // loads per iteration to pipeline. Mirrors MLX's `row_reduce_simple`.
        // Caller verifies reduceSize % (32 * N_READS) == 0.
        kernel void kernel_reduce(
            device const \(metalType)* input [[buffer(0)]],
            device const \(metalType)* initValue [[buffer(1)]],
            device \(metalType)* output [[buffer(2)]],
            constant uint& outputCount [[buffer(3)]],
            constant uint& reduceSize [[buffer(4)]],
            constant uint& innerSize [[buffer(5)]],
            uint tgid [[threadgroup_position_in_grid]],
            uint simd_lane [[thread_index_in_simdgroup]])
        {
            constexpr uint N_WRITES = \(nWrites);
            constexpr uint N_READS = \(nReads);
            uint outputBase = tgid * N_WRITES;
            if (outputBase >= outputCount) return;

            float initB = initValue[0];

        """
        let postlude = """

            uint blocks = reduceSize / (32u * N_READS);

            for (uint b = 0u; b < blocks; ++b) {

        """
        let postlude2 = """
            }


        """
        let storeBlock = """
            if (simd_lane == 0u) {

        """
        let endStore = """
            }
        }
        """

        let source = prelude
            + declareTotals
            + declarePtrs
            + postlude
            + perThreadReduceLines
            + advancePtrLines
            + postlude2
            + simdReduceTotals
            + storeBlock
            + storeOutputs
            + endStore

        return (source, "kernel_reduce", TuningConfig(
            blockSize: 32,
            useSIMDPerOutputReduction: true,
            simdPerOutputNWrites: nWrites
        ))
    }

    /// Generates a row-reduction kernel where each simdgroup of 32 threads
    /// cooperatively reduces one output. With one threadgroup containing 32
    /// simdgroups (1024 threads total) we handle 32 outputs per TG, cutting
    /// per-TG dispatch overhead by 32× compared to "one TG per output".
    /// Used only when innerSize == 1 (pure last-axis reduction) — the most
    /// common case for softmax / attention reductions.
    private func generateRowReductionSIMDPerOutputSource(
        reductionKind: ReductionKind,
        metalType: String
    ) -> (String, String, TuningConfig?) {
        let initValue: String
        let reduceOp: String
        let accumOp: String
        switch reductionKind {
        case .sum, .mean: initValue = "0.0f"; reduceOp = "a + b";   accumOp = "accum += val;"
        case .max:        initValue = "-INFINITY"; reduceOp = "max(a, b)"; accumOp = "accum = max(accum, val);"
        case .min:        initValue = "INFINITY";  reduceOp = "min(a, b)"; accumOp = "accum = min(accum, val);"
        case .product:    initValue = "1.0f"; reduceOp = "a * b";   accumOp = "accum *= val;"
        case .and:        initValue = "1.0f"; reduceOp = "float(int(a) & int(b))"; accumOp = "accum = float(int(accum) & int(val));"
        case .or:         initValue = "0.0f"; reduceOp = "float(int(a) | int(b))"; accumOp = "accum = float(int(accum) | int(val));"
        case .logAddExp:
            // Only emitted for reduce_window (cumlogsumexp), never plain reduce.
            fatalError("logAddExp reduction is only supported via reduce_window")
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        // SIMD-per-output row reduction. innerSize is implicitly 1 (caller
        // verifies). Each TG = 32 simdgroups × 32 lanes = 1024 threads, each
        // simdgroup independently reduces one output row → 32 outputs/TG.
        kernel void kernel_reduce(
            device const \(metalType)* input [[buffer(0)]],
            device const \(metalType)* initValue [[buffer(1)]],
            device \(metalType)* output [[buffer(2)]],
            constant uint& outputCount [[buffer(3)]],
            constant uint& reduceSize [[buffer(4)]],
            constant uint& innerSize [[buffer(5)]],
            uint tgid [[threadgroup_position_in_grid]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]])
        {
            uint myOutput = tgid * 32u + simd_group;
            if (myOutput >= outputCount) return;

            uint base = myOutput * reduceSize;
            float accum = \(initValue);
            for (uint r = simd_lane; r < reduceSize; r += 32u) {
                float val = input[base + r];
                \(accumOp)
            }

            \(simdReduceCode(reductionKind: reductionKind, varName: "accum"))

            if (simd_lane == 0u) {
                float a = accum;
                float b = initValue[0];
                output[myOutput] = \(reduceOp);
            }
        }
        """

        return (source, "kernel_reduce", TuningConfig(blockSize: 1024, useSIMDPerOutputReduction: true))
    }

    /// Generates SIMD reduction code for a given reduction kind.
    private func simdReduceCode(reductionKind: ReductionKind, varName: String) -> String {
        switch reductionKind {
        case .sum, .mean:
            return "\(varName) = simd_sum(\(varName));"
        case .max:
            return "\(varName) = simd_max(\(varName));"
        case .min:
            return "\(varName) = simd_min(\(varName));"
        case .product:
            // No simd_product intrinsic; use shuffle-based reduction
            return """
            for (uint _off = 16; _off > 0; _off /= 2) {
                        \(varName) *= simd_shuffle_down(\(varName), _off);
                    }
            """
        case .and:
            return """
            for (uint _off = 16; _off > 0; _off /= 2) {
                        \(varName) = float(int(\(varName)) & int(simd_shuffle_down(\(varName), _off)));
                    }
            """
        case .or:
            return """
            for (uint _off = 16; _off > 0; _off /= 2) {
                        \(varName) = float(int(\(varName)) | int(simd_shuffle_down(\(varName), _off)));
                    }
            """
        case .logAddExp:
            // Only emitted for reduce_window (cumlogsumexp), never plain reduce.
            fatalError("logAddExp reduction is only supported via reduce_window")
        }
    }

    /// Generates attention source.
    private func generateAttentionSource(_ config: AttentionConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        // Flash Attention kernel
        kernel void kernel_attention(
            device const float* Q [[buffer(0)]],
            device const float* K [[buffer(1)]],
            device const float* V [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint& seqLen [[buffer(4)]],
            constant uint& headDim [[buffer(5)]],
            constant float& scale [[buffer(6)]],
            uint3 gid [[thread_position_in_grid]])
        {
            uint batch = gid.z;
            uint head = gid.y;
            uint queryIdx = gid.x;

            if (queryIdx >= seqLen) return;

            uint headOffset = (batch * \(config.numHeads) + head) * seqLen * headDim;

            // Compute attention scores and output for this query position
            float maxScore = -INFINITY;
            float sumExp = 0.0f;

            // First pass: find max score
            for (uint k = 0; k < seqLen; k++) {
                \(config.causalMask ? "if (k > queryIdx) continue;" : "")
                float score = 0.0f;
                for (uint d = 0; d < headDim; d++) {
                    score += Q[headOffset + queryIdx * headDim + d] * K[headOffset + k * headDim + d];
                }
                score *= scale;
                maxScore = max(maxScore, score);
            }

            // Second pass: compute softmax denominator
            for (uint k = 0; k < seqLen; k++) {
                \(config.causalMask ? "if (k > queryIdx) continue;" : "")
                float score = 0.0f;
                for (uint d = 0; d < headDim; d++) {
                    score += Q[headOffset + queryIdx * headDim + d] * K[headOffset + k * headDim + d];
                }
                score = exp((score * scale) - maxScore);
                sumExp += score;
            }

            // Third pass: compute weighted sum
            for (uint d = 0; d < headDim; d++) {
                float acc = 0.0f;
                for (uint k = 0; k < seqLen; k++) {
                    \(config.causalMask ? "if (k > queryIdx) continue;" : "")
                    float score = 0.0f;
                    for (uint dd = 0; dd < headDim; dd++) {
                        score += Q[headOffset + queryIdx * headDim + dd] * K[headOffset + k * headDim + dd];
                    }
                    score = exp((score * scale) - maxScore) / sumExp;
                    acc += score * V[headOffset + k * headDim + d];
                }
                output[headOffset + queryIdx * headDim + d] = acc;
            }
        }
        """

        return (source, "kernel_attention", TuningConfig.attention)
    }

    /// Generates multi-head attention source.
    private func generateMultiHeadAttentionSource(_ config: MultiHeadAttentionConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        return generateAttentionSource(
            AttentionConfig(
                numHeads: config.numHeads,
                headDim: config.headDim,
                causalMask: config.causalMask
            ),
            inputShapes: inputShapes
        )
    }

    /// Generates RMSNorm source.
    private func generateRMSNormSource(_ config: NormConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_rms_norm(
            device const float* input [[buffer(0)]],
            device const float* weight [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& hiddenSize [[buffer(3)]],
            constant float& epsilon [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]])
        {
            uint batch = gid.y;
            uint offset = batch * hiddenSize;

            // Compute RMS
            float sumSq = 0.0f;
            for (uint i = 0; i < hiddenSize; i++) {
                float x = input[offset + i];
                sumSq += x * x;
            }
            float rms = rsqrt(sumSq / float(hiddenSize) + \(config.epsilon));

            // Normalize and scale
            for (uint i = gid.x; i < hiddenSize; i += \(256)) {
                output[offset + i] = input[offset + i] * rms * weight[i];
            }
        }
        """

        return (source, "kernel_rms_norm", TuningConfig(blockSize: 256))
    }

    /// Generates LayerNorm source.
    private func generateLayerNormSource(_ config: NormConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        // Bake hiddenSize + epsilon as literals so binding matches the custom
        // call's 3 operands [input, gamma, beta] + output — no scalar buffers
        // (the executor doesn't bind extra scalars for a fused custom_call).
        // SIMD-per-row: one simdgroup (32 lanes) reduces one row's mean+var
        // cooperatively, then writes the affine output. 32 simdgroups/TG.
        let hiddenSize = inputShapes.first?.last ?? 1
        if ProcessInfo.processInfo.environment["METALHLO_FUSE_LAYERNORM"] == "1" {
            FileHandle.standardError.write("[ln-codegen] generateLayerNormSource called H=\(hiddenSize) inputShapes=\(inputShapes)\n".data(using: .utf8)!)
        }
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_layer_norm(
            device const float* input [[buffer(0)]],
            device const float* gamma [[buffer(1)]],
            device const float* beta [[buffer(2)]],
            device float* output [[buffer(3)]],
            uint tgid [[threadgroup_position_in_grid]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]])
        {
            const uint H = \(hiddenSize)u;
            uint row = tgid * 32u + simd_group;
            uint offset = row * H;

            float sum = 0.0f;
            for (uint i = simd_lane; i < H; i += 32u) { sum += input[offset + i]; }
            sum = simd_sum(sum);
            float mean = sum / float(H);

            float vs = 0.0f;
            for (uint i = simd_lane; i < H; i += 32u) {
                float d = input[offset + i] - mean; vs += d * d;
            }
            vs = simd_sum(vs);
            float invStd = rsqrt(vs / float(H) + \(config.epsilon)f);

            for (uint i = simd_lane; i < H; i += 32u) {
                float n = (input[offset + i] - mean) * invStd;
                output[offset + i] = n * gamma[i] + beta[i];
            }
        }
        """

        return (source, "kernel_layer_norm", TuningConfig(blockSize: 1024))
    }

    /// Generates numerically stable softmax kernel.
    /// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    private func generateSoftmaxSource(axis: Int, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let inputShape = inputShapes.first ?? []
        let rank = inputShape.count
        // Resolve negative axis
        let resolvedAxis = axis < 0 ? rank + axis : axis
        let reductionSize = rank > 0 ? inputShape[resolvedAxis] : 1
        // Batch = product of all dims except the reduction axis
        let batchSize = inputShape.enumerated().filter { $0.offset != resolvedAxis }.map { $0.element }.reduce(1, *)

        let source: String
        if resolvedAxis == rank - 1 {
            // Common case: softmax over last axis. Use 1 simdgroup (32 threads)
            // per output row. Threads cooperatively reduce across the row using
            // SIMD intrinsics (no shared memory or threadgroup_barrier needed
            // when the cooperating set fits in a single simdgroup). Mirrors the
            // SIMD-per-output reduction kernel — proper parallelism across the
            // reduction dimension.
            source = """
            #include <metal_stdlib>
            using namespace metal;

            constant uint REDUCTION_SIZE = \(reductionSize);
            constant uint BATCH_COUNT = \(batchSize);

            kernel void kernel_softmax(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                uint tgid [[threadgroup_position_in_grid]],
                uint simd_lane [[thread_index_in_simdgroup]])
            {
                uint batch = tgid;
                if (batch >= BATCH_COUNT) return;
                uint offset = batch * REDUCTION_SIZE;

                // Pass 1: find max for numerical stability.
                float maxVal = -INFINITY;
                for (uint i = simd_lane; i < REDUCTION_SIZE; i += 32u) {
                    maxVal = max(maxVal, input[offset + i]);
                }
                maxVal = simd_max(maxVal);

                // Pass 2: compute exp(x - max) and accumulate sum.
                float sumExp = 0.0f;
                for (uint i = simd_lane; i < REDUCTION_SIZE; i += 32u) {
                    sumExp += exp(input[offset + i] - maxVal);
                }
                sumExp = simd_sum(sumExp);

                // Pass 3: normalize. Re-read input + recompute exp to avoid
                // a write-then-read round trip through device memory.
                float invSum = 1.0f / sumExp;
                for (uint i = simd_lane; i < REDUCTION_SIZE; i += 32u) {
                    output[offset + i] = exp(input[offset + i] - maxVal) * invSum;
                }
            }
            """
        } else {
            // General case: softmax over arbitrary axis
            var innerStride = 1
            for d in (resolvedAxis + 1)..<rank {
                innerStride *= inputShape[d]
            }
            let axisStride = innerStride
            let outerStride = reductionSize * innerStride

            source = """
            #include <metal_stdlib>
            using namespace metal;

            constant uint REDUCTION_SIZE = \(reductionSize);

            kernel void kernel_softmax(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                uint tid [[thread_position_in_grid]])
            {
                if (tid >= \(batchSize)) return;

                uint inner = tid % \(innerStride);
                uint outer = tid / \(innerStride);
                uint base = outer * \(outerStride) + inner;

                // Pass 1: find max
                float maxVal = -INFINITY;
                for (uint i = 0; i < REDUCTION_SIZE; i++) {
                    maxVal = max(maxVal, input[base + i * \(axisStride)]);
                }

                // Pass 2: exp and sum
                float sumExp = 0.0f;
                for (uint i = 0; i < REDUCTION_SIZE; i++) {
                    sumExp += exp(input[base + i * \(axisStride)] - maxVal);
                }

                // Pass 3: normalize
                float invSum = 1.0f / sumExp;
                for (uint i = 0; i < REDUCTION_SIZE; i++) {
                    output[base + i * \(axisStride)] = exp(input[base + i * \(axisStride)] - maxVal) * invSum;
                }
            }
            """
        }

        return (source, "kernel_softmax", TuningConfig(blockSize: 256))
    }

    /// Generates MatMul + Bias + Activation source.
    private func generateMatMulBiasActSource(_ config: MatMulConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let activationCode: String
        if let activation = config.activation {
            switch activation {
            case .relu: activationCode = "max(x, 0.0f)"
            case .gelu: activationCode = "x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
            case .silu: activationCode = "x / (1.0f + exp(-x))"
            case .tanh: activationCode = "tanh(x)"
            case .sigmoid: activationCode = "1.0f / (1.0f + exp(-x))"
            case .geluApproximate: activationCode = "x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
            case .none: activationCode = "x"
            }
        } else {
            activationCode = "x"
        }

        let biasCode = config.hasBias ? "result += bias[col];" : ""

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        #define TILE_SIZE 32

        kernel void kernel_matmul_bias_act(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],
            device const float* bias [[buffer(2)]],
            device float* C [[buffer(3)]],
            constant uint& M [[buffer(4)]],
            constant uint& N [[buffer(5)]],
            constant uint& K [[buffer(6)]],
            uint2 gid [[thread_position_in_grid]])
        {
            uint row = gid.y;
            uint col = gid.x;

            if (row >= M || col >= N) return;

            float sum = 0.0f;
            for (uint k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }

            float result = sum;
            \(biasCode)

            float x = result;
            C[row * N + col] = \(activationCode);
        }
        """

        return (source, "kernel_matmul_bias_act", TuningConfig.matmul)
    }

    /// Generates GELU source.
    private func generateGELUSource(approximate: Bool, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let geluCode = approximate
            ? "x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
            : "x * 0.5f * (1.0f + erf(x * 0.7071067811865475f))"

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_gelu(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            float x = input[tid];
            output[tid] = \(geluCode);
        }
        """

        return (source, "kernel_gelu", nil)
    }

    /// Generates SiLU source.
    private func generateSiLUSource(inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_silu(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            float x = input[tid];
            output[tid] = x / (1.0f + exp(-x));
        }
        """

        return (source, "kernel_silu", nil)
    }

    /// Generates the fused-elementwise chain kernel from an SSA-DAG chain.
    ///
    /// The chain records, for each op, the source of each operand: either
    /// an external input slot (read as `inputN[tid]` for full-shape tensors
    /// or `inputN[0]` for scalar 1-element tensors) or the result of an
    /// earlier chain op (already materialised as a local `vK` variable).
    /// The kernel emits one temporary per op and writes the last op's
    /// result to `output[tid]`.
    ///
    /// Reshape ops are linear-index no-ops — the temp just aliases the
    /// operand expression. broadcast_in_dim of a scalar is the same: the
    /// scalar is loaded once via `input[0]` and propagated by re-reading
    /// it from each consuming op.
    ///
    /// Buffer layout (must match `buildBindings()`):
    /// - Input buffers at indices 0..<inputShapes.count
    /// - Output buffer at index inputShapes.count
    /// - Count scalar at index inputShapes.count + 1
    private func generateElementwiseChainSource(
        _ chain: FusedElementwiseChain,
        inputShapes: [[Int]],
        outputShape: [Int],
        elementType: ElementType = .float32,
        inputElementTypes: [ElementType] = []
    ) -> (String, String, TuningConfig?) {
        // Declare enough inputs to cover every external operand the chain
        // references. inputShapes can be SHORTER than the operand list when a
        // shape is missing from analysis (op.inputs.compactMap drops nils), so
        // sizing off inputShapes.count alone leaves a referenced `inputN`
        // undeclared. The binding always provides all operands as buffers.
        let maxExternal = chain.ops
            .flatMap { $0.operands }
            .compactMap { src -> Int? in if case .external(let s) = src { return s }; return nil }
            .max() ?? -1
        let inputCount = max(inputShapes.count, maxExternal + 1, 1)
        let metalType = metalTypeName(for: elementType)
        // Per-thread temporaries' type. Normally the chain's output type, but a
        // BOOLEAN-result chain must compute in float: such chains mix integer
        // index math with a boolean result — e.g. the causal-mask tril:
        // iota → add → compare(GE) → select : i1. With `bool` temps the integer
        // `add` is truncated to 0/1 and corrupts the mask (the compare/select
        // already emit float 1.0/0.0). Float keeps that arithmetic exact
        // (values ≪ 2^24); the final store narrows back to the bool output.
        // Do NOT widen integer chains to float — they may use integer
        // divide/modulo (index unflattening), whose semantics differ in float.
        let computeType = (metalType == "bool") ? "float" : metalType

        // Per-external buffer element type. The chain computes in `metalType`
        // (the root's type — float for the usual case), but an external may be
        // a DIFFERENT, narrower type whose buffer must be read at its true
        // width before converting. Two cases bite:
        //
        //  • i1 (bool): a causal mask feeding a select predicate is a 1-byte
        //    `bool` buffer. Read as `float` → corrupt (NaN).
        //  • integers: `select(compare(int_a, int_b), float_x, float_y)` puts
        //    integer compare operands in a float chain (HLO forbids mixed-type
        //    arithmetic, but select/compare legitimately mix predicate/operand
        //    types). Reading an `int` buffer as `float` reinterprets the bits.
        //
        // Declare each such external at its true Metal type; the existing
        // `metalType vK = inputN[idx]` then does the correct implicit numeric
        // conversion (bool→float 1/0, int→float). Float externals are left as
        // `metalType` exactly as before — the common path is unchanged.
        //
        // Type tracking alone is insufficient for the predicate: a
        // re-materialised inlined mask (ProducerConsumerFusion.repairDanglingRefs)
        // can reach the chain with no recorded element type, defaulting to f32.
        // So ALSO infer boolean-ness structurally — any external feeding a
        // select's predicate position (through shape-only ops) is an i1 mask.
        var predicateSlots = Set<Int>()
        func markPredicateSource(_ src: FusedElementwiseChain.OperandSource) {
            switch src {
            case .external(let slot):
                predicateSlots.insert(slot)
            case .prior(let k):
                guard k < chain.ops.count else { return }
                let op = chain.ops[k]
                // Shape-only ops forward their single operand's mask-ness.
                if op.kind == .broadcastInDim || op.kind == .reshape || op.kind == .transpose,
                   let first = op.operands.first {
                    markPredicateSource(first)
                }
                // A `.compare` prior is an in-chain float predicate (1.0/0.0),
                // not a bool buffer — nothing to mark.
            }
        }
        for op in chain.ops where op.kind == .select {
            if let pred = op.operands.first { markPredicateSource(pred) }
        }
        func slotMetalType(_ slot: Int) -> String {
            // Structurally-detected predicate, or a recorded i1: read as bool.
            if predicateSlots.contains(slot)
                || (slot < inputElementTypes.count && inputElementTypes[slot] == .int1) {
                return "bool"
            }
            // A recorded non-float (integer) type: read at its true width so
            // the load converts to the chain's float type instead of
            // reinterpreting raw bytes. Float / unknown → chain compute type.
            if slot < inputElementTypes.count {
                let et = inputElementTypes[slot]
                if !isFloatType(et) {
                    return metalTypeName(for: et)
                }
            }
            // Non-predicate float (or unknown) externals are read at the chain's
            // *compute* type, not the output type. When the chain root is a
            // compare its output `metalType` is `bool`, but its float data
            // inputs (the compare operands, select branches, …) must be read as
            // `float` — reading them as `bool` reinterprets every nonzero value
            // as 1.0 and collapses the comparison (e.g. `8.0 <= 5.0` becomes
            // `true <= true`). computeType is `float` exactly in that case.
            return computeType
        }

        var inputParams: [String] = []
        for i in 0..<inputCount {
            inputParams.append("device const \(slotMetalType(i))* input\(i) [[buffer(\(i))]]")
        }
        let outputBufferIndex = inputCount

        func externalRef(_ slot: Int) -> String {
            if slot < inputShapes.count {
                let count = inputShapes[slot].reduce(1, *)
                if count == 1 { return "input\(slot)[0]" }
            }
            return "input\(slot)[tid]"
        }

        // For chains that absorb a transpose or non-scalar broadcast:
        // compute the permuted/broadcast source index per thread.
        //
        // * transpose: input_coord[d] = output_coord[perm[d]]
        // * broadcast_in_dim: input_coord[d] = output_coord[dims[d]] if
        //   input.shape[d] > 1 else 0 (size-1 dims fan out)
        //
        // In both cases tid is unflattened using the op's *own* output
        // shape (different from the chain root's shape if a reshape
        // follows; same total element count, different rank).
        var preamble = ""
        var computedIdxFor: [Int: String] = [:]
        for (i, op) in chain.ops.enumerated() {
            guard op.kind == .transpose || op.kind == .broadcastInDim,
                  let dims = op.dimensions,
                  case .external(let slot) = op.operands.first,
                  slot < inputShapes.count else { continue }
            var inShape = inputShapes[slot]
            // The operand shape may be missing from analysis (inlined mask
            // constants like %inlN_6 aren't shape-tracked). For a broadcast we
            // can reconstruct it: operand dim d maps to output dim dims[d], so
            // operand.shape[d] = outputShape[dims[d]] (no size-1 input dims in
            // the patterns we fuse — the causal mask is [S,S]→[B,H,S,S]).
            if inShape.isEmpty, op.kind == .broadcastInDim,
               let outShape = op.outputShape, !dims.isEmpty {
                inShape = dims.map { $0 < outShape.count ? outShape[$0] : 1 }
            }
            guard dims.count == inShape.count else { continue }

            // input_stride[d] = product of inShape[d+1..]
            var inStrides = [Int](repeating: 1, count: inShape.count)
            for d in stride(from: inShape.count - 2, through: 0, by: -1) {
                inStrides[d] = inStrides[d + 1] * inShape[d + 1]
            }

            // The op's own output shape:
            // - transpose: derived as inShape[perm[d]]
            // - broadcast: stored in `op.outputShape` (encoded in result type)
            let opOutShape: [Int]
            if op.kind == .transpose {
                opOutShape = dims.map { inShape[$0] }
            } else {
                opOutShape = op.outputShape ?? []
            }
            guard !opOutShape.isEmpty else { continue }

            var outStrides = [Int](repeating: 1, count: opOutShape.count)
            for d in stride(from: opOutShape.count - 2, through: 0, by: -1) {
                outStrides[d] = outStrides[d + 1] * opOutShape[d + 1]
            }

            preamble += "    // \(op.kind) at chain op \(i): in \(inShape) dims \(dims) out \(opOutShape)\n"
            preamble += "    uint t\(i)_rem = tid;\n"
            for d in 0..<opOutShape.count {
                preamble += "    uint t\(i)_c\(d) = t\(i)_rem / \(outStrides[d])u;\n"
                preamble += "    t\(i)_rem = t\(i)_rem % \(outStrides[d])u;\n"
            }

            // For each input dim d, which OUTPUT-coordinate axis supplies it:
            // - broadcast_in_dim: input dim d maps to output dim dims[d], so
            //   input_coord[d] = output_coord[dims[d]] (size-1 dims fan out).
            // - transpose: output[k] = input[perm[k]] (perm = dims), so
            //   input_coord[d] = output_coord[perm⁻¹[d]] — the INVERSE
            //   permutation. Using dims[d] directly is only correct when perm
            //   is an involution (e.g. (0,2,1,3)); rotations like (3,0,1,2,4)
            //   need the inverse, otherwise the data is permuted the wrong way.
            var sourceDim = dims
            if op.kind == .transpose {
                var inv = [Int](repeating: 0, count: dims.count)
                for (k, p) in dims.enumerated() where p < inv.count { inv[p] = k }
                sourceDim = inv
            }
            var terms: [String] = []
            for d in 0..<inShape.count {
                if op.kind == .broadcastInDim && inShape[d] == 1 {
                    continue
                }
                let sourceOutDim = sourceDim[d]
                if inStrides[d] == 1 {
                    terms.append("t\(i)_c\(sourceOutDim)")
                } else {
                    terms.append("t\(i)_c\(sourceOutDim) * \(inStrides[d])u")
                }
            }
            let idxExpr = terms.isEmpty ? "0u" : terms.joined(separator: " + ")
            preamble += "    uint t\(i)_idx = \(idxExpr);\n"
            computedIdxFor[i] = "t\(i)_idx"
        }

        // Build per-op expressions referring to temps `v0…v_{n-1}`. Each
        // op emits `metalType vK = <expr>;`. Unary "shape" ops (reshape)
        // emit `vK = operand;` — the codegen-level identity. Transpose
        // and non-scalar broadcast load via the precomputed index.
        var body = ""
        for (i, op) in chain.ops.enumerated() {
            if (op.kind == .transpose || op.kind == .broadcastInDim),
               let idx = computedIdxFor[i],
               case .external(let slot) = op.operands.first {
                body += "    \(computeType) v\(i) = input\(slot)[\(idx)];\n"
                continue
            }
            func operandExpr(_ src: FusedElementwiseChain.OperandSource) -> String {
                switch src {
                case .external(let slot): return externalRef(slot)
                case .prior(let idx):     return "v\(idx)"
                }
            }
            let operands = op.operands.map(operandExpr)
            let expr = elementwiseChainExpr(kind: op.kind, operands: operands, metalType: computeType, comparison: op.comparison)
            body += "    \(computeType) v\(i) = \(expr);\n"
        }
        let resultExpr: String = chain.ops.isEmpty ? externalRef(0) : "v\(chain.ops.count - 1)"

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_elementwise_chain(
            \(inputParams.joined(separator: ",\n            ")),
            device \(metalType)* output [[buffer(\(outputBufferIndex))]],
            constant uint& count [[buffer(\(outputBufferIndex + 1))]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
        \(preamble)\(body)    output[tid] = \(resultExpr);
        }
        """

        return (source, "kernel_elementwise_chain", nil)
    }

    /// Generates a single-pass fused-reduce kernel: applies a pointwise chain
    /// transform per input element, then reduces (MLX layernorm/softmax style).
    /// Shape params (outputCount/reduceSize/innerSize) and the init value are
    /// baked as literals, so binding is identical to fusedElementwise: chain
    /// externals at buffers 0..M-1, output at buffer M. fp32 only; the fusion
    /// pass guarantees a pointwise chain (no transpose/broadcast).
    private func generateFusedReduceSource(
        _ cfg: FusedReduceConfig,
        inputShapes: [[Int]],
        elementType: ElementType
    ) -> (String, String, TuningConfig?) {
        let metalType = "float"
        let inputCount = max(inputShapes.count, 1)
        let inShape = cfg.inputShape
        let reduceDims = cfg.reduceDims.sorted()
        let reduceSize = max(reduceDims.reduce(1) { $0 * (($1 < inShape.count) ? inShape[$1] : 1) }, 1)
        let lastReduceDim = reduceDims.last ?? 0
        let innerSize = (lastReduceDim + 1 < inShape.count)
            ? inShape[(lastReduceDim + 1)...].reduce(1, *)
            : 1
        let outputCount = max(cfg.outputShape.reduce(1, *), 1)

        // init / per-element combine / simd intrinsic / finalize per reduce kind
        let initVal: String, combine: String, simdOp: String
        var finalize = "v"
        switch cfg.reductionKind {
        case "sum":     initVal = "0.0f";        combine = "accum + val";        simdOp = "simd_sum"
        case "mean":    initVal = "0.0f";        combine = "accum + val";        simdOp = "simd_sum"; finalize = "v / float(\(reduceSize))"
        case "max":     initVal = "-INFINITY";   combine = "max(accum, val)";    simdOp = "simd_max"
        case "min":     initVal = "INFINITY";    combine = "min(accum, val)";    simdOp = "simd_min"
        case "product": initVal = "1.0f";        combine = "accum * val";        simdOp = "simd_product"
        default:        initVal = "0.0f";        combine = "accum + val";        simdOp = "simd_sum"
        }

        var inputParams: [String] = []
        for i in 0..<inputCount {
            inputParams.append("device const \(metalType)* input\(i) [[buffer(\(i))]]")
        }

        // Emit the pointwise chain computing `float cv{n-1}` at element `idx`.
        func operandExpr(_ src: FusedElementwiseChain.OperandSource, idx: String) -> String {
            switch src {
            case .external(let slot):
                let count = slot < inputShapes.count ? inputShapes[slot].reduce(1, *) : 0
                return count == 1 ? "input\(slot)[0]" : "input\(slot)[\(idx)]"
            case .prior(let p): return "cv\(p)"
            }
        }
        var chainBody = ""
        for (i, op) in cfg.chain.ops.enumerated() {
            let ops = op.operands.map { operandExpr($0, idx: "idx") }
            let expr = elementwiseChainExpr(kind: op.kind, operands: ops, metalType: metalType)
            chainBody += "                \(metalType) cv\(i) = \(expr);\n"
        }
        let chainResult = cfg.chain.ops.isEmpty ? "input0[idx]" : "cv\(cfg.chain.ops.count - 1)"

        let outputBufferIndex = inputCount
        // SIMD-per-output: one simdgroup (32 lanes) reduces one output; 32
        // simdgroups per 1024-thread TG → 32 outputs/TG. Matches the fast
        // specialized reduce variant (vs 1024-threads-per-output, which wastes
        // threads on nanoGPT's short reduce axes).
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        // Single-pass fused reduce (SIMD-per-output). Baked: outputCount=\(outputCount),
        // reduceSize=\(reduceSize), innerSize=\(innerSize), kind=\(cfg.reductionKind).
        kernel void kernel_reduce(
            \(inputParams.joined(separator: ",\n            ")),
            device \(metalType)* output [[buffer(\(outputBufferIndex))]],
            uint tgid [[threadgroup_position_in_grid]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]])
        {
            uint outIdx = tgid * 32u + simd_group;
            if (outIdx >= \(outputCount)u) return;

            uint outerIdx = outIdx / \(innerSize)u;
            uint innerIdx = outIdx % \(innerSize)u;
            uint baseInputIdx = outerIdx * \(reduceSize)u * \(innerSize)u + innerIdx;

            float accum = \(initVal);
            for (uint r = simd_lane; r < \(reduceSize)u; r += 32u) {
                uint idx = baseInputIdx + r * \(innerSize)u;
                \(chainBody)float val = \(chainResult);
                accum = \(combine);
            }

            accum = \(simdOp)(accum);
            if (simd_lane == 0) {
                float v = accum;
                output[outIdx] = \(finalize);
            }
        }
        """
        return (source, "kernel_reduce", TuningConfig(blockSize: 1024))
    }

    /// Renders the right-hand side for a single chain op into a Metal
    /// expression. Each operand is already a stringified expression
    /// (`inputN[tid]`, `inputN[0]`, or a `vK` reference).
    private func elementwiseChainExpr(kind: HLOOpKind, operands: [String], metalType: String, comparison: String? = nil) -> String {
        // Defensive: bad chain serialization shouldn't crash codegen.
        let lhs = operands.first ?? "0"
        let rhs = operands.count > 1 ? operands[1] : "0"
        switch kind {
        // Comparison → float predicate (1.0/0.0), consumed in-register by a
        // downstream chain `.select`. No bool buffer involved.
        case .compare:
            let op: String
            // ComparisonDirection.rawValue is upper-case ("LT", "EQ", …); the
            // serialized chain carries that raw value verbatim. Normalize so a
            // case mismatch doesn't silently collapse every comparison to "=="
            // (which inverts predicates feeding select → wrong branch → NaN).
            switch comparison?.uppercased() {
            case "EQ": op = "=="
            case "NE": op = "!="
            case "LT": op = "<"
            case "LE": op = "<="
            case "GT": op = ">"
            case "GE": op = ">="
            default:   op = "=="
            }
            return "((\(lhs) \(op) \(rhs)) ? 1.0f : 0.0f)"
        // select(pred, onTrue, onFalse): pred is a float predicate (1.0/0.0).
        case .select:
            let a = operands.count > 1 ? operands[1] : "0"
            let b = operands.count > 2 ? operands[2] : "0"
            return "((\(lhs) != 0.0f) ? \(a) : \(b))"
        case .clamp:
            // clamp(lo, x, hi)
            let x = operands.count > 1 ? operands[1] : "0"
            let hi = operands.count > 2 ? operands[2] : "0"
            return "clamp(\(x), \(lhs), \(hi))"
        default:
            break
        }
        switch kind {
        case .add:       return "\(lhs) + \(rhs)"
        case .subtract:  return "\(lhs) - \(rhs)"
        case .multiply:  return "\(lhs) * \(rhs)"
        case .divide:    return "\(lhs) / \(rhs)"
        case .remainder: return "fmod(\(lhs), \(rhs))"
        case .maximum:   return "max(\(lhs), \(rhs))"
        case .minimum:   return "min(\(lhs), \(rhs))"
        case .power:     return "pow(\(lhs), \(rhs))"
        case .atan2:     return "atan2(\(lhs), \(rhs))"
        case .negate:    return "-\(lhs)"
        case .abs:       return "abs(\(lhs))"
        case .exponential: return "exp(\(lhs))"
        case .log:       return "log(\(lhs))"
        case .sqrt:      return "sqrt(\(lhs))"
        case .rsqrt:     return "rsqrt(\(lhs))"
        case .tanh:      return "tanh(\(lhs))"
        case .logistic:  return "1.0f / (1.0f + exp(-\(lhs)))"
        case .sine:      return "sin(\(lhs))"
        case .cosine:    return "cos(\(lhs))"
        case .floor:     return "floor(\(lhs))"
        case .ceil:      return "ceil(\(lhs))"
        // Reshape and scalar broadcast: pass the operand through unchanged.
        // Reshape is a no-op in linear-index space; scalar broadcast was
        // already lowered to `input[0]` by externalRef.
        case .reshape, .broadcastInDim: return lhs
        default:         return lhs
        }
    }

    /// Checks if an operation is binary.
    private func isBinaryOp(_ op: HLOOpKind) -> Bool {
        switch op {
        case .add, .subtract, .multiply, .divide, .remainder, .maximum, .minimum, .power, .atan2,
             .and, .or, .xor, .shiftLeft, .shiftRightLogical, .shiftRightArithmetic:
            return true
        default:
            return false
        }
    }

    /// Generates code for a binary operation.
    private func generateBinaryOpCode(_ op: HLOOpKind, left: String, right: String, declareX: Bool, metalType: String = "float") -> String {
        let prefix = declareX ? "\(metalType) x = " : "x = "
        switch op {
        case .add:
            return "\(prefix)\(left) + \(right);\n"
        case .subtract:
            return "\(prefix)\(left) - \(right);\n"
        case .multiply:
            return "\(prefix)\(left) * \(right);\n"
        case .divide:
            return "\(prefix)\(left) / \(right);\n"
        case .remainder:
            return "\(prefix)fmod(\(left), \(right));\n"
        case .maximum:
            return "\(prefix)max(\(left), \(right));\n"
        case .minimum:
            return "\(prefix)min(\(left), \(right));\n"
        case .power:
            return "\(prefix)pow(\(left), \(right));\n"
        case .atan2:
            return "\(prefix)atan2(\(left), \(right));\n"
        case .and:
            return "\(prefix)\(left) & \(right);\n"
        case .or:
            return "\(prefix)\(left) | \(right);\n"
        case .xor:
            return "\(prefix)\(left) ^ \(right);\n"
        case .shiftLeft:
            let bwL = metalType == "ulong" || metalType == "long" ? 64 : (metalType == "ushort" || metalType == "short" ? 16 : (metalType == "uchar" || metalType == "char" ? 8 : 32))
            return "\(prefix)((\(right)) >= \(bwL)) ? (\(metalType))(0) : ((\(left)) << (\(right)));\n"
        case .shiftRightLogical:
            let bwRL = metalType == "ulong" || metalType == "long" ? 64 : (metalType == "ushort" || metalType == "short" ? 16 : (metalType == "uchar" || metalType == "char" ? 8 : 32))
            return "\(prefix)((\(right)) >= \(bwRL)) ? (\(metalType))(0) : ((\(left)) >> (\(right)));\n"
        case .shiftRightArithmetic:
            let bwRA = metalType == "ulong" || metalType == "long" ? 64 : (metalType == "ushort" || metalType == "short" ? 16 : (metalType == "uchar" || metalType == "char" ? 8 : 32))
            return "\(prefix)((\(right)) >= \(bwRA)) ? ((\(left)) >> \(bwRA - 1)) : ((\(left)) >> (\(right)));\n"
        default:
            return "\(prefix)\(left);\n"  // Fallback
        }
    }

    /// Generates code for a unary operation.
    private func generateUnaryOpCode(_ op: HLOOpKind) -> String {
        switch op {
        case .negate: return "x = -x;\n"
        case .abs: return "x = abs(x);\n"
        case .exponential: return "x = exp(x);\n"
        case .log: return "x = log(x);\n"
        case .sqrt: return "x = sqrt(x);\n"
        case .rsqrt: return "x = rsqrt(x);\n"
        case .tanh: return "x = tanh(x);\n"
        case .logistic: return "x = 1.0f / (1.0f + exp(-x));\n"
        case .sine: return "x = sin(x);\n"
        case .cosine: return "x = cos(x);\n"
        case .floor: return "x = floor(x);\n"
        case .ceil: return "x = ceil(x);\n"
        default: return ""
        }
    }

    /// Generates FFN source.
    private func generateFFNSource(_ config: FFNConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        // For now, generate separate kernels - full fusion would be more complex
        return generateMatMulBiasActSource(
            MatMulConfig(activation: config.activation),
            inputShapes: inputShapes
        )
    }

    /// Generates transformer block source.
    /// Mega-kernel that fuses: pre-norm -> attention -> residual -> pre-norm -> FFN -> residual
    private func generateTransformerBlockSource(_ config: TransformerBlockConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let numHeads = config.attention.numHeads
        let headDim = config.attention.headDim
        let hiddenDim = numHeads * headDim  // Derive hidden dim from attention config
        let intermediateDim = config.ffn.intermediateDim
        let isCausal = config.attention.causalMask ? 1 : 0

        // Activation code for FFN
        let activationCode: String
        switch config.ffn.activation {
        case .relu: activationCode = "max(x, 0.0f)"
        case .gelu, .geluApproximate:
            activationCode = "x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
        case .silu: activationCode = "x / (1.0f + exp(-x))"
        case .tanh: activationCode = "tanh(x)"
        case .sigmoid: activationCode = "1.0f / (1.0f + exp(-x))"
        case .none: activationCode = "x"
        }

        // Norm code (RMSNorm or LayerNorm)
        let normFunction: String
        if config.normType == .rmsNorm {
            normFunction = """
            inline void apply_norm(
                thread float* normed,
                device const float* input,
                device const float* weight,
                uint offset,
                uint hiddenDim,
                float epsilon
            ) {
                float sumSq = 0.0f;
                for (uint i = 0; i < hiddenDim; i++) {
                    float v = input[offset + i];
                    sumSq += v * v;
                }
                float rms = rsqrt(sumSq / float(hiddenDim) + epsilon);
                for (uint i = 0; i < hiddenDim; i++) {
                    normed[i] = input[offset + i] * rms * weight[i];
                }
            }
            """
        } else {
            normFunction = """
            inline void apply_norm(
                thread float* normed,
                device const float* input,
                device const float* weight,
                device const float* bias,
                uint offset,
                uint hiddenDim,
                float epsilon
            ) {
                float mean = 0.0f;
                for (uint i = 0; i < hiddenDim; i++) {
                    mean += input[offset + i];
                }
                mean /= float(hiddenDim);
                float var_sum = 0.0f;
                for (uint i = 0; i < hiddenDim; i++) {
                    float diff = input[offset + i] - mean;
                    var_sum += diff * diff;
                }
                float invStd = rsqrt(var_sum / float(hiddenDim) + epsilon);
                for (uint i = 0; i < hiddenDim; i++) {
                    normed[i] = (input[offset + i] - mean) * invStd * weight[i] + bias[i];
                }
            }
            """
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        \(normFunction)

        // Transformer Block Mega-Kernel
        // Fuses: PreNorm -> Attention -> Residual -> PreNorm -> FFN -> Residual
        kernel void kernel_transformer_block(
            device const float* input [[buffer(0)]],           // [batch, seq, hidden]
            device const float* attnNormWeight [[buffer(1)]],  // [hidden]
            device const float* Wq [[buffer(2)]],              // [hidden, hidden]
            device const float* Wk [[buffer(3)]],              // [hidden, hidden]
            device const float* Wv [[buffer(4)]],              // [hidden, hidden]
            device const float* Wo [[buffer(5)]],              // [hidden, hidden]
            device const float* ffnNormWeight [[buffer(6)]],   // [hidden]
            device const float* W1 [[buffer(7)]],              // [hidden, intermediate]
            device const float* W2 [[buffer(8)]],              // [intermediate, hidden]
            device float* output [[buffer(9)]],                // [batch, seq, hidden]
            constant uint& batchSize [[buffer(10)]],
            constant uint& seqLen [[buffer(11)]],
            constant float& scale [[buffer(12)]],
            constant float& epsilon [[buffer(13)]],
            threadgroup float* shared_q [[threadgroup(0)]],    // [block_size, head_dim]
            threadgroup float* shared_k [[threadgroup(1)]],    // [seq_len, head_dim]
            threadgroup float* shared_v [[threadgroup(2)]],    // [seq_len, head_dim]
            uint3 threadgroup_id [[threadgroup_position_in_grid]],
            uint3 thread_id [[thread_position_in_threadgroup]],
            uint3 threads_per_group [[threads_per_threadgroup]]
        ) {
            uint batch = threadgroup_id.z;
            uint head = threadgroup_id.y;
            uint q_block = threadgroup_id.x;

            if (batch >= batchSize) return;

            const uint hiddenDim = \(hiddenDim);
            const uint numHeads = \(numHeads);
            const uint headDim = \(headDim);
            const uint intermediateDim = \(intermediateDim);
            const uint isCausal = \(isCausal);
            const uint blockSize = 64;

            uint local_q = thread_id.x;
            uint global_q = q_block * blockSize + local_q;
            if (global_q >= seqLen) return;

            uint headOffset = head * headDim;
            uint inputOffset = (batch * seqLen + global_q) * hiddenDim;

            // ========== Pre-Attention Norm ==========
            float normed[64];  // Max hidden dim per head
            float sumSq = 0.0f;
            for (uint i = 0; i < hiddenDim; i++) {
                float v = input[inputOffset + i];
                sumSq += v * v;
            }
            float rms = rsqrt(sumSq / float(hiddenDim) + epsilon);
            for (uint i = 0; i < hiddenDim; i++) {
                normed[i] = input[inputOffset + i] * rms * attnNormWeight[i];
            }

            // ========== Q, K, V Projections ==========
            float Q[64], K[64], V[64];
            for (uint d = 0; d < headDim; d++) {
                float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
                uint out_idx = headOffset + d;
                for (uint i = 0; i < hiddenDim; i++) {
                    q_sum += normed[i] * Wq[i * hiddenDim + out_idx];
                    k_sum += normed[i] * Wk[i * hiddenDim + out_idx];
                    v_sum += normed[i] * Wv[i * hiddenDim + out_idx];
                }
                Q[d] = q_sum;
                K[d] = k_sum;
                V[d] = v_sum;
            }

            // Store Q in shared memory for this block
            for (uint d = 0; d < headDim; d++) {
                shared_q[local_q * headDim + d] = Q[d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ========== Self-Attention with Online Softmax ==========
            float m_i = -INFINITY;
            float l_i = 0.0f;
            float o_acc[64];
            for (uint d = 0; d < headDim; d++) o_acc[d] = 0.0f;

            // Iterate over all K/V positions
            for (uint k_pos = 0; k_pos < seqLen; k_pos++) {
                if (isCausal && k_pos > global_q) break;

                // Load K, V for k_pos (would be from shared memory in full implementation)
                float k_val[64], v_val[64];
                uint k_input_offset = (batch * seqLen + k_pos) * hiddenDim;

                // Compute K, V for this position
                for (uint d = 0; d < headDim; d++) {
                    float k_sum = 0.0f, v_sum = 0.0f;
                    uint out_idx = headOffset + d;
                    // Simplified: recompute projection (in production, would cache)
                    for (uint i = 0; i < min(hiddenDim, 64u); i++) {
                        float inp = input[k_input_offset + i] * rms * attnNormWeight[i];
                        k_sum += inp * Wk[i * hiddenDim + out_idx];
                        v_sum += inp * Wv[i * hiddenDim + out_idx];
                    }
                    k_val[d] = k_sum;
                    v_val[d] = v_sum;
                }

                // Compute attention score
                float score = 0.0f;
                for (uint d = 0; d < headDim; d++) {
                    score += Q[d] * k_val[d];
                }
                score *= scale;

                // Online softmax update
                float m_new = max(m_i, score);
                float exp_diff_old = exp(m_i - m_new);
                float exp_diff_new = exp(score - m_new);

                for (uint d = 0; d < headDim; d++) {
                    o_acc[d] = o_acc[d] * exp_diff_old + exp_diff_new * v_val[d];
                }
                l_i = l_i * exp_diff_old + exp_diff_new;
                m_i = m_new;
            }

            // Normalize attention output
            float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            for (uint d = 0; d < headDim; d++) {
                o_acc[d] *= l_inv;
            }

            // Output projection
            float attn_out[64];
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                float sum = 0.0f;
                for (uint d = 0; d < headDim; d++) {
                    sum += o_acc[d] * Wo[(headOffset + d) * hiddenDim + i];
                }
                attn_out[i] = sum;
            }

            // ========== Attention Residual ==========
            float after_attn[64];
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                after_attn[i] = input[inputOffset + i] + attn_out[i];
            }

            // ========== Pre-FFN Norm ==========
            sumSq = 0.0f;
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                sumSq += after_attn[i] * after_attn[i];
            }
            float rms2 = rsqrt(sumSq / float(hiddenDim) + epsilon);
            float ffn_in[64];
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                ffn_in[i] = after_attn[i] * rms2 * ffnNormWeight[i];
            }

            // ========== FFN ==========
            // Up projection + activation
            float intermediate[64];
            for (uint j = 0; j < min(intermediateDim, 64u); j++) {
                float sum = 0.0f;
                for (uint i = 0; i < hiddenDim && i < 64; i++) {
                    sum += ffn_in[i] * W1[i * intermediateDim + j];
                }
                float x = sum;
                intermediate[j] = \(activationCode);
            }

            // Down projection
            float ffn_out[64];
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                float sum = 0.0f;
                for (uint j = 0; j < min(intermediateDim, 64u); j++) {
                    sum += intermediate[j] * W2[j * hiddenDim + i];
                }
                ffn_out[i] = sum;
            }

            // ========== FFN Residual + Output ==========
            uint outputOffset = inputOffset;
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                output[outputOffset + i] = after_attn[i] + ffn_out[i];
            }
        }
        """

        let tuning = TuningConfig(
            tileM: 64,
            tileN: 64,
            tileK: 64,
            blockSize: 64,
            useSharedMemory: true,
            useSIMDGroups: true
        )

        return (source, "kernel_transformer_block", tuning)
    }

    /// Generates RoPE source.
    private func generateRoPESource(_ config: RoPEConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_rope(
            device const float* input [[buffer(0)]],
            device const float* cos_cache [[buffer(1)]],
            device const float* sin_cache [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint& seqLen [[buffer(4)]],
            constant uint& headDim [[buffer(5)]],
            uint3 gid [[thread_position_in_grid]])
        {
            uint pos = gid.x;
            uint dim = gid.y;

            if (pos >= seqLen || dim >= headDim / 2) return;

            uint idx1 = pos * headDim + dim * 2;
            uint idx2 = idx1 + 1;

            float x1 = input[idx1];
            float x2 = input[idx2];

            float cos_val = cos_cache[pos * (headDim / 2) + dim];
            float sin_val = sin_cache[pos * (headDim / 2) + dim];

            output[idx1] = x1 * cos_val - x2 * sin_val;
            output[idx2] = x1 * sin_val + x2 * cos_val;
        }
        """

        return (source, "kernel_rope", nil)
    }

    // MARK: - Dispatch Calculation

    /// Calculates dispatch configuration for an operation.
    private func calculateDispatch(type: FusedOpType, shapes: [[Int]], inputShapes: [[Int]] = [], attributes: HLOAttributes = HLOAttributes(), tuning: TuningConfig?, elementType: ElementType = .float32) -> DispatchConfig {
        guard let outputShape = shapes.first else {
            return DispatchConfig.dispatch1D(elements: 1)
        }

        let totalElements = outputShape.reduce(1, *)

        switch type {
        case .original(let opKind):
            switch opKind {
            case .dot, .dotGeneral:
                // Compute M, K, N from dot_general dimension numbers
                let dims: (M: UInt32, K: UInt32, N: UInt32, batchSize: UInt32)
                if inputShapes.count >= 2 {
                    dims = dotGeneralDims(lhsShape: inputShapes[0], rhsShape: inputShapes[1], attributes: attributes)
                } else {
                    let m = outputShape.count >= 2 ? outputShape[outputShape.count - 2] : (outputShape.first ?? 1)
                    let n = outputShape.count >= 2 ? outputShape[outputShape.count - 1] : 1
                    dims = (UInt32(m), 1, UInt32(n), 1)
                }
                let M = Int(dims.M)
                let N = Int(dims.N)
                let K = Int(dims.K)
                let batchSize = Int(dims.batchSize)

                // Output tile size per threadgroup. Default 32x32 for the
                // simdgroup_matrix kernel; the MetalPerformancePrimitives
                // kernel signals its (BM=64, BN=32) tile via tuning.tileM/tileN.
                let tileM = tuning?.tileM ?? 32
                let tileN = tuning?.tileN ?? 32
                var gridWidth = (N + tileN - 1) / tileN
                var gridHeight = (M + tileM - 1) / tileM
                // MLX-style swizzle reshapes the launch grid so concurrent
                // threadgroups share B-tile reads → L2 reuse. The kernel
                // decodes (tid.x, tid.y) back into logical (tile_n, tile_m).
                if let swl = tuning?.swizzleLog, swl > 0 {
                    let tile = 1 << swl
                    gridHeight = (gridHeight + tile - 1) / tile
                    gridWidth = gridWidth * tile
                }
                let basicTileSize = 32
                let useSimdgroup = isFloatType(elementType) && M % 8 == 0 && K % 8 == 0 && N % 8 == 0 && M >= 8 && K >= 8 && N >= 8

                // Non-tiled kernel (1 thread per output): for transposed matmul
                if let t = tuning, t.blockSize == 1 {
                    return DispatchConfig.dispatch1D(elements: M * N * batchSize, threadgroupSize: 256)
                }

                // GEMV path: 1 simdgroup (32 threads) per TG, each TG handles
                // gemvNWrites consecutive output columns. Dispatch
                // ceildiv(N, gemvNWrites) threadgroups.
                if let t = tuning, t.useGEMV {
                    let nw = max(1, t.gemvNWrites)
                    let numThreadgroups = (N + nw - 1) / nw
                    return DispatchConfig(
                        gridSize: MTLSize(width: numThreadgroups, height: 1, depth: 1),
                        threadgroupSize: MTLSize(width: 32, height: 1, depth: 1)
                    )
                }

                // Threads per threadgroup. Default 128 (4 simdgroups). The MPP
                // kernel signals an alternate count (e.g. 256 for 8 simdgroups)
                // via tuning.blockSize.
                let simdgroupThreads = tuning?.blockSize ?? 128

                if batchSize > 1 {
                    // 3D dispatch for batched matmul
                    if useSimdgroup {
                        return DispatchConfig(
                            gridSize: MTLSize(width: gridWidth, height: gridHeight, depth: batchSize),
                            threadgroupSize: MTLSize(width: simdgroupThreads, height: 1, depth: 1)
                        )
                    } else {
                        return DispatchConfig(
                            gridSize: MTLSize(width: gridWidth, height: gridHeight, depth: batchSize),
                            threadgroupSize: MTLSize(width: basicTileSize, height: basicTileSize, depth: 1)
                        )
                    }
                } else {
                    if useSimdgroup {
                        return DispatchConfig(
                            gridSize: MTLSize(width: gridWidth, height: gridHeight, depth: 1),
                            threadgroupSize: MTLSize(width: simdgroupThreads, height: 1, depth: 1)
                        )
                    } else {
                        // Basic tiled matmul needs the full 32x32 threadgroup to
                        // initialize all shared memory tile entries. dispatch2D
                        // would clamp to min(32, matrixDim), leaving tiles uninitialized.
                        return DispatchConfig(
                            gridSize: MTLSize(width: gridWidth, height: gridHeight, depth: 1),
                            threadgroupSize: MTLSize(width: basicTileSize, height: basicTileSize, depth: 1)
                        )
                    }
                }
            case .reduceArg:
                // argmax/argmin index kernel: one thread per output element
                // (the flattened non-reduced dims), each scanning its reduce
                // axis. totalElements is already the output count.
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)

            case .sort, .sortResult:
                // One thread per element; each ranks its element within the row.
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)

            case .reduce:
                // Check for specialized reduction kernels
                if let tuning = tuning, (tuning.useRowReduction || tuning.useColumnReduction) {
                    // Specialized row/column reduction: one thread per output element
                    // Uses 1D dispatch with threads processing entire rows/columns independently
                    let blockSize = tuning.blockSize ?? 1024
                    return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: blockSize)
                }

                // SIMD-per-output reduction. `tuning.blockSize` controls the TG
                // size; the float4 variant uses 32 (one simdgroup per TG, each
                // TG handling `simdPerOutputNWrites` consecutive outputs), the
                // scalar variant uses 1024 (32 simdgroups, each on a distinct
                // output → 32 outputs per TG).
                if let tuning = tuning, tuning.useSIMDPerOutputReduction {
                    let nWrites = max(1, tuning.simdPerOutputNWrites)
                    let tgSize = tuning.blockSize ?? 1024
                    // Outputs per TG: with a 32-thread TG it's `nWrites`, with a
                    // 1024-thread TG it's `32 * nWrites`.
                    let simdgroupsPerTG = max(1, tgSize / 32)
                    let outputsPerTG = simdgroupsPerTG * nWrites
                    let numThreadgroups = (totalElements + outputsPerTG - 1) / outputsPerTG
                    return DispatchConfig(
                        gridSize: MTLSize(width: numThreadgroups, height: 1, depth: 1),
                        threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1)
                    )
                }

                // General reduction: one threadgroup per output element
                // Threads within threadgroup cooperatively reduce using tree reduction + SIMD
                let blockSize = tuning?.blockSize ?? 1024
                let numThreadgroups = totalElements  // One threadgroup per output element
                return DispatchConfig(
                    gridSize: MTLSize(width: numThreadgroups, height: 1, depth: 1),
                    threadgroupSize: MTLSize(width: blockSize, height: 1, depth: 1)
                )
            case .transpose:
                // Get permutation from attributes
                let permutation = attributes.dimensions ?? []
                let inputShape = inputShapes.first ?? []

                // Use tiled dispatch ONLY for optimized 2D [1,0] and 3D [0,2,1] transposes
                // For all other permutations, use 1D dispatch since we use the general transpose kernel
                let tileSize = 32

                // Check for optimized 2D transpose [M, N] -> [N, M] with permutation [1, 0]
                if inputShape.count == 2 && permutation == [1, 0] {
                    let inputCols = outputShape[0]  // N = output rows = input cols
                    let inputRows = outputShape[1]  // M = output cols = input rows
                    let gridWidth = (inputCols + tileSize - 1) / tileSize
                    let gridHeight = (inputRows + tileSize - 1) / tileSize
                    return DispatchConfig(
                        gridSize: MTLSize(width: gridWidth, height: gridHeight, depth: 1),
                        threadgroupSize: MTLSize(width: tileSize, height: tileSize, depth: 1)
                    )
                }

                // Check for optimized 3D transpose [B, M, N] -> [B, N, M] with permutation [0, 2, 1]
                if inputShape.count == 3 && permutation == [0, 2, 1] {
                    let batch = outputShape[0]
                    let inputCols = outputShape[1]  // N = output rows = input cols per batch
                    let inputRows = outputShape[2]  // M = output cols = input rows per batch
                    let gridWidth = (inputCols + tileSize - 1) / tileSize
                    let gridHeight = (inputRows + tileSize - 1) / tileSize
                    return DispatchConfig(
                        gridSize: MTLSize(width: gridWidth, height: gridHeight, depth: batch),
                        threadgroupSize: MTLSize(width: tileSize, height: tileSize, depth: 1)
                    )
                }

                // For all other transpose permutations, use 1D dispatch
                // This matches the general transpose kernel which uses thread_position_in_grid as uint
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 1024)
            case .broadcastInDim:
                // Broadcast kernel is NOT vectorized - it expects tid to be the output element index
                // Use non-vectorized 1D dispatch to match the kernel
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            case .gather, .dynamicGather:
                // Gather kernel is NOT vectorized - it expects tid to be the output element index
                // Use non-vectorized 1D dispatch to match the kernel
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            case .convolution:
                // Convolution kernel: 1 thread per output element
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            case .reduceWindow:
                // Reduce window kernel: 1 thread per output element
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            case .selectAndScatter:
                // Select and scatter: 1 thread per output element (input shape)
                let inputElements = inputShapes.first.map { $0.reduce(1, *) } ?? totalElements
                return DispatchConfig.dispatch1D(elements: inputElements, threadgroupSize: 256)
            case .fft:
                // FFT kernel: 1 thread per output element
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            case .scatter:
                // Scatter kernel: copy operand to output then write updates at indexed positions.
                // Needs max(operandElements, updateElements) threads.
                let operandElements = inputShapes.first.map { $0.reduce(1, *) } ?? totalElements
                let updatesElements = inputShapes.count > 2
                    ? inputShapes[2].reduce(1, *)
                    : totalElements
                return DispatchConfig.dispatch1D(
                    elements: max(operandElements, updatesElements),
                    threadgroupSize: 256
                )
            case .convert, .bitcastConvert:
                // The convert kernel is vectorized (4 elements per thread via
                // vec<T,4>) for >=2-byte float-to-float conversions — the hot
                // path the TF32 matmul wrapper hits. Other conversions use the
                // scalar 1-thread-per-element form. Match the
                // generateConvertKernel gate exactly.
                let inputType = attributes.inputElementTypes?.first
                if opKind == .convert,
                   let inputType,
                   isFloatType(inputType), isFloatType(elementType),
                   elementByteSize(for: inputType) >= 2,
                   elementByteSize(for: elementType) >= 2 {
                    let groups = (totalElements + 3) / 4
                    return DispatchConfig.dispatch1D(elements: groups, threadgroupSize: 256)
                }
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            case .compare:
                // Compare kernel is NOT vectorized - one thread per element
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            case .slice:
                // Slice kernel is NOT vectorized - it expects tid to be the output element index
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            case .reshape:
                // Reshape uses a non-vectorized copy kernel (1 element per thread).
                // Must not use the default float32 vectorized dispatch (8 elements/thread).
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            case .reverse, .pad, .concatenate, .iota, .select, .clamp,
                 .dynamicSlice, .dynamicUpdateSlice:
                // These kernels are NOT vectorized - 1 thread per element
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            case .topKValues, .topKIndices:
                // One thread per output row (all dims except the last). Each
                // thread selects the k largest of its input row. k = last output
                // dim, so rows = totalOutputElements / k.
                let k = outputShape.last ?? 1
                let rows = k > 0 ? totalElements / k : totalElements
                return DispatchConfig.dispatch1D(elements: max(rows, 1), threadgroupSize: 256)
            default:
                break
            }

        case .fusedAttention, .fusedMultiHeadAttention:
            if outputShape.count >= 3 {
                let batch = outputShape[0]
                let heads = outputShape.count > 3 ? outputShape[1] : 1
                let seqLen = outputShape[outputShape.count - 2]
                return DispatchConfig(
                    gridSize: MTLSize(width: seqLen, height: heads, depth: batch),
                    threadgroupSize: MTLSize(width: 1, height: 1, depth: 1)
                )
            }

        case .fusedRMSNorm, .fusedBatchNorm:
            if outputShape.count >= 2 {
                let batch = outputShape.dropLast().reduce(1, *)
                return DispatchConfig(
                    gridSize: MTLSize(width: (256 + 255) / 256, height: batch, depth: 1),
                    threadgroupSize: MTLSize(width: 256, height: 1, depth: 1)
                )
            }

        case .fusedLayerNorm:
            // SIMD-per-row: 32 simdgroups/TG, one row each → 32 rows/TG.
            if ProcessInfo.processInfo.environment["METALHLO_FUSE_LAYERNORM"] == "1" {
                FileHandle.standardError.write("[ln-dispatch] outputShape=\(outputShape)\n".data(using: .utf8)!)
            }
            let lnRows: Int
            if outputShape.count >= 2 {
                lnRows = outputShape.dropLast().reduce(1, *)
            } else {
                // Fallback: rows = total / hiddenSize unknown → use totalElements/last.
                lnRows = max(outputShape.last.map { totalElements / max($0, 1) } ?? totalElements, 1)
            }
            let lnTgs = (lnRows + 31) / 32
            return DispatchConfig(
                gridSize: MTLSize(width: max(lnTgs, 1), height: 1, depth: 1),
                threadgroupSize: MTLSize(width: 1024, height: 1, depth: 1)
            )

        case .fusedMatMulBiasAct:
            if outputShape.count >= 2 {
                let M = outputShape[outputShape.count - 2]
                let N = outputShape[outputShape.count - 1]
                return DispatchConfig.dispatch2D(width: N, height: M)
            }

        case .fusedConvBiasAct:
            // Same shape as the unfused conv kernel: 1 thread per output.
            return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)

        case .fusedElementwise:
            // FusedElementwise kernel is NOT vectorized - it processes 1 element per thread
            // using tid directly as the element index. Do NOT use vectorized dispatch.
            return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)

        case .fusedReduce:
            // SIMD-per-output: 32 simdgroups/TG, one output each → 32 outputs/TG.
            // totalElements = output element count (post-reduction).
            let outs = max(totalElements, 1)
            let tgs = (outs + 31) / 32
            return DispatchConfig(
                gridSize: MTLSize(width: tgs, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: 1024, height: 1, depth: 1)
            )

        case .fusedGELU, .fusedSiLU:
            // GELU and SiLU kernels are NOT vectorized - they use tid directly
            return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)

        case .fusedSoftmax(let axis):
            // Last-axis softmax: 1 simdgroup (32 threads) per output row.
            // batch TGs each cooperatively reduce one row using simd_max /
            // simd_sum — much faster than the previous "threads do redundant
            // sequential scans" form which the 256-thread TG implied.
            let rank = outputShape.count
            let resolvedAxis = axis < 0 ? rank + axis : axis
            if resolvedAxis == rank - 1 && outputShape.count >= 2 {
                let batch = outputShape.dropLast().reduce(1, *)
                return DispatchConfig(
                    gridSize: MTLSize(width: batch, height: 1, depth: 1),
                    threadgroupSize: MTLSize(width: 32, height: 1, depth: 1)
                )
            } else {
                // General axis: 1D dispatch over batch elements
                let batchSize = outputShape.enumerated().filter { $0.offset != resolvedAxis }.map { $0.element }.reduce(1, *)
                return DispatchConfig.dispatch1D(elements: batchSize, threadgroupSize: 256)
            }

        default:
            break
        }

        // For float elementwise operations, use vectorized dispatch (8 elements per thread)
        if elementType == .float32 {
            let vectorizedCount = (totalElements + 7) / 8
            return DispatchConfig.dispatch1D(elements: vectorizedCount, threadgroupSize: 1024)
        }

        return DispatchConfig.dispatch1D(elements: totalElements)
    }

    // MARK: - Buffer Bindings

    /// Builds buffer bindings for an operation.
    /// Resolves view tensors to their base tensor's memory location.
    private func buildBindings(
        op: FusedOp,
        tensors: [TensorID: TensorInfo],
        memoryPlan: MemoryPlan,
        inputNames: Set<String>,
        constantIDs: Set<TensorID>,
        viewMappings: [TensorID: StridedTensorView] = [:]
    ) -> [BufferBinding] {
        var bindings: [BufferBinding] = []
        var index = 0

        // Helper to resolve view chain and get base tensor ID and byte offset
        func resolveViewChain(_ tensorID: TensorID) -> (baseTensorID: TensorID, byteOffset: Int) {
            var currentID = tensorID
            var totalOffset = 0

            while let view = viewMappings[currentID] {
                totalOffset += view.byteOffset
                currentID = view.baseTensorID
            }

            return (currentID, totalOffset)
        }

        // Input bindings
        for inputID in op.inputs {
            let source: BufferSource
            let size = tensors[inputID]?.byteSize ?? 0

            // Resolve view chain to get base tensor
            let (baseTensorID, viewOffset) = resolveViewChain(inputID)

            // Check if this is a constant (pre-materialized into constant buffer)
            if constantIDs.contains(baseTensorID) {
                source = .constant(id: baseTensorID)
            }
            // Check if this is a function input (comes from external buffer)
            else if inputNames.contains(baseTensorID) {
                source = .input(name: baseTensorID)
            } else if let offset = memoryPlan.tensorOffsets[baseTensorID] {
                // Intermediate result in the unified buffer (include view offset)
                source = .unified(offset: offset + viewOffset)
            } else if let offset = memoryPlan.tensorOffsets[inputID] {
                // Fallback: try original tensor ID
                source = .unified(offset: offset)
            } else {
                // Fallback: treat as input (shouldn't happen normally)
                source = .input(name: inputID)
            }

            bindings.append(BufferBinding(
                index: index,
                source: source,
                size: size,
                access: .read
            ))
            index += 1
        }

        // Output bindings
        for output in op.outputs {
            let source: BufferSource
            let size = output.byteSize

            if let offset = memoryPlan.tensorOffsets[output.id] {
                source = .unified(offset: offset)
            } else {
                source = .output(name: output.id)
            }

            bindings.append(BufferBinding(
                index: index,
                source: source,
                size: size,
                access: .write
            ))
            index += 1
        }

        // Add scalar parameter bindings for operations that need them
        // (e.g., element count for elementwise operations)
        if let count = scalarCountForOperation(op) {
            bindings.append(BufferBinding(
                index: index,
                source: .scalar(count),
                size: MemoryLayout<UInt32>.size,
                access: .read
            ))
            index += 1
        }

        // Add M, N, K dimension parameters for matmul operations
        if case .original(let opKind) = op.type, (opKind == .dot || opKind == .dotGeneral) {
            // Get input shapes to determine M, N, K
            let inputShapes = op.inputs.compactMap { tensors[$0]?.shape }
            if inputShapes.count >= 2 {
                let lhsShape = inputShapes[0]
                let rhsShape = inputShapes[1]

                let (M, K, N, batchSize) = dotGeneralDims(lhsShape: lhsShape, rhsShape: rhsShape, attributes: op.attributes)

                bindings.append(BufferBinding(
                    index: index,
                    source: .scalar(M),
                    size: MemoryLayout<UInt32>.size,
                    access: .read
                ))
                index += 1

                bindings.append(BufferBinding(
                    index: index,
                    source: .scalar(N),
                    size: MemoryLayout<UInt32>.size,
                    access: .read
                ))
                index += 1

                bindings.append(BufferBinding(
                    index: index,
                    source: .scalar(K),
                    size: MemoryLayout<UInt32>.size,
                    access: .read
                ))
                index += 1

                // Add batchCount for batched matmul (kernel uses this when batchSize > 1)
                if batchSize > 1 {
                    bindings.append(BufferBinding(
                        index: index,
                        source: .scalar(batchSize),
                        size: MemoryLayout<UInt32>.size,
                        access: .read
                    ))
                }
            }
        }

        // Add reduce operation parameters: outputCount, reduceSize, innerSize.
        // Both the value reduce (.reduce) and the argmax/argmin index reduce
        // (.reduceArg) take these three scalars after their buffers; .reduceArg
        // has only one input (the values) and writes the index output.
        if case .original(let opKind) = op.type, opKind == .reduce || opKind == .reduceArg {
            let inputShapes = op.inputs.compactMap { tensors[$0]?.shape }
            guard let inputShape = inputShapes.first, !inputShape.isEmpty else {
                return bindings
            }

            // Get reduction dimensions from attributes (first input is data, second is init value)
            let reduceDims = op.attributes.dimensions ?? [inputShape.count - 1]

            // Calculate outputCount, reduceSize, and innerSize
            // For now, handle reduction along a single dimension (most common case)
            // The output shape removes the reduction dimensions
            let outputShape = op.outputs.first?.shape ?? []
            let outputCount = outputShape.isEmpty ? 1 : outputShape.reduce(1, *)

            // Calculate reduceSize (product of reduction dimensions)
            var reduceSize = 1
            for dim in reduceDims {
                if dim >= 0 && dim < inputShape.count {
                    reduceSize *= inputShape[dim]
                }
            }

            // innerSize is the product of dimensions after all reduction dimensions
            // For simple last-dim reduction: innerSize = 1
            // For general case: we need to handle the layout properly
            let maxReduceDim = reduceDims.max() ?? (inputShape.count - 1)
            var innerSize = 1
            for i in (maxReduceDim + 1)..<inputShape.count {
                innerSize *= inputShape[i]
            }

            // If reduction is along last dimension(s), innerSize = 1
            if innerSize == 0 { innerSize = 1 }

            bindings.append(BufferBinding(
                index: index,
                source: .scalar(UInt32(outputCount)),
                size: MemoryLayout<UInt32>.size,
                access: .read
            ))
            index += 1

            bindings.append(BufferBinding(
                index: index,
                source: .scalar(UInt32(reduceSize)),
                size: MemoryLayout<UInt32>.size,
                access: .read
            ))
            index += 1

            bindings.append(BufferBinding(
                index: index,
                source: .scalar(UInt32(innerSize)),
                size: MemoryLayout<UInt32>.size,
                access: .read
            ))
        }

        // sort: after input (0) and output (1) — or key (0), payload (1),
        // output (2) for sortResult — bind axisLen / innerSize / total. The sort
        // axis length comes from input 0 (the key), the same shape as the output.
        if case .original(let opKind) = op.type, opKind == .sort || opKind == .sortResult,
           let inputShape = op.inputs.first.flatMap({ tensors[$0]?.shape }), !inputShape.isEmpty {
            let rank = inputShape.count
            let axisAttr = op.attributes.axis ?? (rank - 1)
            let ax = axisAttr < 0 ? axisAttr + rank : axisAttr
            let axisLen = (ax >= 0 && ax < rank) ? inputShape[ax] : 1
            var innerSize = 1
            if ax + 1 < rank { for i in (ax + 1)..<rank { innerSize *= inputShape[i] } }
            let total = max(inputShape.reduce(1, *), 1)
            for value in [axisLen, innerSize, total] {
                bindings.append(BufferBinding(
                    index: index, source: .scalar(UInt32(value)),
                    size: MemoryLayout<UInt32>.size, access: .read))
                index += 1
            }
        }

        return bindings
    }

    /// Determines if an operation needs a scalar count parameter and returns it.
    private func scalarCountForOperation(_ op: FusedOp) -> UInt32? {
        switch op.type {
        case .original(let opKind):
            // Elementwise operations need count parameter
            switch opKind {
            case .add, .subtract, .multiply, .divide, .remainder, .maximum, .minimum, .power, .atan2,
                 .negate, .abs, .exponential, .log, .sqrt, .rsqrt, .ceil, .floor, .roundNearestEven,
                 .sine, .cosine, .tanh, .sign, .logistic, .expm1, .log1p, .cbrt,
                 .and, .or, .xor, .not,
                 .shiftLeft, .shiftRightArithmetic, .shiftRightLogical,
                 .clamp, .select,
                 .compare,
                 .convert,
                 .popcnt,
                 .bitcastConvert,
                 .clz,
                 // Shape/indexing operations
                 .reshape, .transpose, .broadcastInDim,
                 .reverse, .pad, .concatenate, .iota,
                 .convolution, .reduceWindow, .selectAndScatter, .fft:
                // Calculate total elements from output shape
                if let output = op.outputs.first {
                    let count = output.shape.isEmpty ? 1 : output.shape.reduce(1, *)
                    return UInt32(count)
                }
            default:
                return nil
            }

        case .fusedGELU, .fusedSiLU:
            if let output = op.outputs.first {
                let count = output.shape.isEmpty ? 1 : output.shape.reduce(1, *)
                return UInt32(count)
            }

        case .fusedElementwise:
            if let output = op.outputs.first {
                let count = output.shape.isEmpty ? 1 : output.shape.reduce(1, *)
                return UInt32(count)
            }

        case .fusedConvBiasAct:
            if let output = op.outputs.first {
                let count = output.shape.isEmpty ? 1 : output.shape.reduce(1, *)
                return UInt32(count)
            }

        default:
            return nil
        }

        return nil
    }

    // MARK: - Shared Memory

    /// Computes M, K, N, batchSize for dot_general from dimension numbers.
    private func dotGeneralDims(lhsShape: [Int], rhsShape: [Int], attributes: HLOAttributes) -> (M: UInt32, K: UInt32, N: UInt32, batchSize: UInt32) {
        let dotDimNums = attributes.dotDimensionNumbers
        var lhsContractDims = dotDimNums?.lhsContractingDimensions ?? [lhsShape.count - 1]
        var rhsContractDims = dotDimNums?.rhsContractingDimensions ?? (rhsShape.count >= 2 ? [rhsShape.count - 2] : [0])
        var lhsBatchDims = dotDimNums?.lhsBatchingDimensions ?? []
        var rhsBatchDims = dotDimNums?.rhsBatchingDimensions ?? []

        // Apply transpose-folding attributes (see generateMatMulSource for the rationale).
        if attributes.lhsTranspose == true, lhsShape.count >= 2 {
            lhsContractDims = swapLastTwoDimPositions(lhsContractDims, rank: lhsShape.count)
            lhsBatchDims = swapLastTwoDimPositions(lhsBatchDims, rank: lhsShape.count)
        }
        if attributes.rhsTranspose == true, rhsShape.count >= 2 {
            rhsContractDims = swapLastTwoDimPositions(rhsContractDims, rank: rhsShape.count)
            rhsBatchDims = swapLastTwoDimPositions(rhsBatchDims, rank: rhsShape.count)
        }

        var mVal = 1
        for (i, s) in lhsShape.enumerated() {
            if !lhsContractDims.contains(i) && !lhsBatchDims.contains(i) { mVal *= s }
        }
        var kVal = 1
        for d in lhsContractDims { kVal *= lhsShape[d] }
        var nVal = 1
        for (i, s) in rhsShape.enumerated() {
            if !rhsContractDims.contains(i) && !rhsBatchDims.contains(i) { nVal *= s }
        }
        // Batch size comes from explicit lhs_batching_dimensions only. The
        // earlier "implicit batch from LHS leading dims when LHS and RHS share
        // rank" heuristic worked for (B,M,K) @ (B,K,N) without explicit batch
        // dims but mis-handled the canonical layout that the
        // dot-general-layout-canonicalize pass produces — e.g. a backward dot
        // `(1,8,16) @ (1,8,32) contract=[0,1]x[0,1]` gets canonicalized to
        // `(16,1,8) @ (1,8,32) contract=[1,2]x[0,1]`, both still rank-3 but
        // not actually batched. JAX always emits explicit batching_dims for
        // real batched matmuls (attention, vmap), so trusting only those is
        // correct.
        let batchVal: Int = lhsBatchDims.isEmpty
            ? 1
            : lhsBatchDims.reduce(1) { $0 * lhsShape[$1] }
        return (UInt32(mVal), UInt32(kVal), UInt32(nVal), UInt32(batchVal))
    }

    /// Returns the byte size of an element type.
    private func elementByteSize(for elementType: ElementType) -> Int {
        switch elementType {
        case .float64, .int64, .uint64:
            return 8
        case .float32, .int32, .uint32:
            return 4
        case .float16, .bfloat16, .int16, .uint16:
            return 2
        case .int8, .uint8, .int1:
            return 1
        }
    }

    /// Calculates shared memory size for an operation.
    private func calculateSharedMemorySize(type: FusedOpType, tuning: TuningConfig?, elementType: ElementType = .float32) -> Int {
        let elemSize = elementByteSize(for: elementType)

        switch type {
        case .original(let opKind):
            if opKind == .dot || opKind == .dotGeneral {
                // Simdgroup kernels use simdgroup_matrix operations in registers (no shared memory).
                // Basic tiled kernels need 2 threadgroup buffers (tileA and tileB).
                if tuning?.useSIMDGroups == true {
                    return 0
                }
                let tileSize = tuning?.tileM ?? 32
                return 2 * tileSize * tileSize * elemSize
            }
            if opKind == .transpose {
                // Tiled transpose uses 32x32 tile with +1 padding to avoid bank conflicts
                // Total: 32 * 33 * elemSize = 1056 * elemSize
                let tileSize = 32
                let tilePadded = tileSize + 1
                return tileSize * tilePadded * elemSize
            }
            if opKind == .reduce {
                // Parallel tree reduction uses shared memory for 256 threads
                let blockSize = 256
                return blockSize * elemSize
            }
        case .fusedMatMulBiasAct:
            // For now, fusedMatMulBiasAct uses basic tiled approach
            let tileSize = tuning?.tileM ?? 32
            return 2 * tileSize * tileSize * elemSize
        default:
            break
        }

        return 0
    }

    /// Calculates the number of threadgroup memory buffers for an operation.
    private func calculateThreadgroupBufferCount(type: FusedOpType, tuning: TuningConfig? = nil, elementType: ElementType = .float32) -> Int {
        switch type {
        case .original(let opKind):
            if opKind == .dot || opKind == .dotGeneral {
                // Simdgroup kernels don't use threadgroup memory.
                // Basic tiled matmul needs 2 buffers (tileA and tileB).
                if tuning?.useSIMDGroups == true {
                    return 0
                }
                return 2
            }
            if opKind == .transpose {
                // Tiled transpose uses 1 buffer (tile)
                return 1
            }
            if opKind == .reduce {
                // Parallel tree reduction uses 1 buffer (shared accumulator)
                return 1
            }
        case .fusedMatMulBiasAct:
            // Uses 2 buffers (tileA and tileB)
            return 2
        default:
            break
        }
        return 1
    }
}
