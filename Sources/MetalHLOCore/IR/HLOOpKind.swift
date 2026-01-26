// HLOOpKind.swift
// MetalHLOCore
//
// Enumeration of supported StableHLO operation kinds.

/// Supported StableHLO operation kinds.
///
/// This enum covers the 47 operations supported by MetalHLO.
public enum HLOOpKind: String, CaseIterable, Sendable {

    // MARK: - Binary Arithmetic (7 ops)

    /// Element-wise addition.
    case add

    /// Element-wise subtraction.
    case subtract

    /// Element-wise multiplication.
    case multiply

    /// Element-wise division.
    case divide

    /// Element-wise maximum.
    case maximum

    /// Element-wise minimum.
    case minimum

    /// Element-wise power.
    case power

    // MARK: - Unary Operations (17 ops)

    /// Negation.
    case negate

    /// Absolute value.
    case abs

    /// Exponential (e^x).
    case exponential

    /// Natural logarithm.
    case log

    /// Square root.
    case sqrt

    /// Reciprocal square root (1/sqrt(x)).
    case rsqrt

    /// Sine.
    case sine

    /// Cosine.
    case cosine

    /// Hyperbolic tangent.
    case tanh

    /// Floor.
    case floor

    /// Ceiling.
    case ceil

    /// Sign function.
    case sign

    /// Logical not (for boolean tensors).
    case not

    /// Bitwise and.
    case and

    /// Bitwise or.
    case or

    /// Bitwise xor.
    case xor

    /// Count leading zeros.
    case clz = "count_leading_zeros"

    /// Tangent.
    case tan

    /// Logistic (sigmoid): 1/(1+e^-x).
    case logistic

    /// Check if value is finite.
    case isFinite = "is_finite"

    /// Exponential minus one (e^x - 1).
    case expm1 = "exponential_minus_one"

    /// Log plus one (log(1 + x)).
    case log1p = "log_plus_one"

    /// Cube root.
    case cbrt

    /// Round to nearest, ties away from zero.
    case roundNearestAfz = "round_nearest_afz"

    /// Round to nearest, ties to even (banker's rounding).
    case roundNearestEven = "round_nearest_even"

    /// Population count (count set bits).
    case popcnt = "popcnt"

    /// Shift left.
    case shiftLeft = "shift_left"

    /// Shift right arithmetic (sign-extending).
    case shiftRightArithmetic = "shift_right_arithmetic"

    /// Shift right logical (zero-filling).
    case shiftRightLogical = "shift_right_logical"

    /// Create complex number from real and imaginary parts.
    case complex

    /// Extract real part from complex number.
    case real

    /// Extract imaginary part from complex number.
    case imag

    // MARK: - Type Conversion (1 op)

    /// Type conversion.
    case convert

    /// Bitcast convert (reinterpret bits as different type).
    case bitcastConvert = "bitcast_convert"

    /// Reduce precision (truncate mantissa/exponent).
    case reducePrecision = "reduce_precision"

    // MARK: - Matrix Operations (5 ops)

    /// 2D matrix multiplication.
    case dot

    /// Generalized dot product (batched matmul).
    case dotGeneral = "dot_general"

    /// Dimension permutation (transpose).
    case transpose

    /// Shape change (reshape).
    case reshape

    /// Broadcast to shape.
    case broadcastInDim = "broadcast_in_dim"

    /// Dynamic broadcast (runtime output shape).
    case dynamicBroadcastInDim = "dynamic_broadcast_in_dim"

    /// Dynamic reshape (runtime output shape).
    case dynamicReshape = "dynamic_reshape"

    /// Reverse tensor along dimensions.
    case reverse

    // MARK: - Convolution Operations (1 op)

    /// N-dimensional convolution.
    case convolution

    // MARK: - Reduction Operations (1 op that covers multiple patterns)

    /// Reduction operation (sum, max, min, etc.).
    case reduce

    /// Reduction over sliding windows (pooling).
    case reduceWindow = "reduce_window"

    /// Select and scatter (used for pooling gradients).
    case selectAndScatter = "select_and_scatter"

    // MARK: - Linear Algebra Operations (2 ops)

    /// Triangular solve (solve Ax = b where A is triangular).
    case triangularSolve = "triangular_solve"

    /// Cholesky decomposition.
    case cholesky

    // MARK: - Normalization Operations (3 ops)

    /// Batch normalization for inference.
    case batchNormInference = "batch_norm_inference"

    /// Batch normalization for training.
    case batchNormTraining = "batch_norm_training"

    /// Batch normalization gradient.
    case batchNormGrad = "batch_norm_grad"

    // MARK: - FFT Operations (1 op)

    /// Fast Fourier Transform.
    case fft

    // MARK: - Sorting Operations (1 op)

    /// Sort along dimension.
    case sort

    // MARK: - Comparison Operations (2 ops)

    /// Comparison (EQ, NE, LT, LE, GT, GE).
    case compare

    /// Conditional selection.
    case select

    // MARK: - Clamp (1 op)

    /// Clamp to range.
    case clamp

    // MARK: - Indexing and Slicing (5 ops)

    /// Extract slice.
    case slice

    /// Dynamic slice (runtime start indices).
    case dynamicSlice = "dynamic_slice"

    /// Dynamic update slice.
    case dynamicUpdateSlice = "dynamic_update_slice"

    /// Pad tensor.
    case pad

    /// Dynamic pad (runtime padding amounts).
    case dynamicPad = "dynamic_pad"

    /// Gather elements.
    case gather

    /// Dynamic gather (runtime slice sizes).
    case dynamicGather = "dynamic_gather"

    /// Scatter elements.
    case scatter

    /// Concatenate tensors.
    case concatenate

    // MARK: - Random Number Generation (2 ops)

    /// Random number generation.
    case rng

    /// RNG bit generator (deterministic PRNG).
    case rngBitGenerator = "rng_bit_generator"

    // MARK: - Constants (1 op)

    /// Constant tensor.
    case constant

    // MARK: - Control Flow (2 ops - Deferred)

    /// While loop (deferred).
    case whileOp = "while"

    /// Conditional (deferred).
    case ifOp = "if"

    // MARK: - Iota (2 ops)

    /// Iota - fills tensor with increasing values.
    case iota

    /// Dynamic iota (runtime output shape).
    case dynamicIota = "dynamic_iota"

    // MARK: - Map Operation (1 op)

    /// Apply computation element-wise.
    case map

    // MARK: - Quantization Operations (2 ops)

    /// Quantize float to integer.
    case uniformQuantize = "uniform_quantize"

    /// Dequantize integer to float.
    case uniformDequantize = "uniform_dequantize"

    // MARK: - Custom Calls (1 op)

    /// Custom call for fused operations.
    /// These are handled by specialized Metal kernels or MPSGraph implementations.
    /// Supported targets:
    /// - fused_scaled_dot_product_attention
    /// - fused_layer_norm
    /// - fused_rms_norm
    /// - fused_matmul_bias_activation
    /// - fused_softmax
    /// - fused_gelu
    /// - fused_rope
    case customCall = "custom_call"
}

// MARK: - Operation Categories

extension HLOOpKind {

    /// Whether this is a binary arithmetic operation.
    public var isBinaryArithmetic: Bool {
        switch self {
        case .add, .subtract, .multiply, .divide, .maximum, .minimum, .power:
            return true
        default:
            return false
        }
    }

    /// Whether this is a unary operation.
    public var isUnary: Bool {
        switch self {
        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt,
             .sine, .cosine, .tanh, .floor, .ceil, .sign,
             .not, .and, .or, .xor, .clz, .tan, .logistic, .isFinite,
             .expm1, .log1p, .cbrt, .roundNearestAfz, .roundNearestEven,
             .popcnt, .real, .imag, .cholesky:
            return true
        default:
            return false
        }
    }

    /// Whether this is a matrix operation.
    public var isMatrixOp: Bool {
        switch self {
        case .dot, .dotGeneral, .transpose, .reshape, .broadcastInDim,
             .dynamicBroadcastInDim, .dynamicReshape, .reverse, .convolution,
             .triangularSolve, .cholesky:
            return true
        default:
            return false
        }
    }

    /// Whether this is a linear algebra operation.
    public var isLinearAlgebra: Bool {
        switch self {
        case .triangularSolve, .cholesky:
            return true
        default:
            return false
        }
    }

    /// Whether this is a dynamic operation.
    public var isDynamic: Bool {
        switch self {
        case .dynamicSlice, .dynamicUpdateSlice, .dynamicReshape,
             .dynamicBroadcastInDim, .dynamicPad, .dynamicIota, .dynamicGather:
            return true
        default:
            return false
        }
    }

    /// Whether this is a normalization operation.
    public var isNormalization: Bool {
        switch self {
        case .batchNormInference, .batchNormTraining, .batchNormGrad:
            return true
        default:
            return false
        }
    }

    /// Whether this is a comparison operation.
    public var isComparison: Bool {
        switch self {
        case .compare, .select:
            return true
        default:
            return false
        }
    }

    /// Whether this is an indexing operation.
    public var isIndexing: Bool {
        switch self {
        case .slice, .dynamicSlice, .dynamicUpdateSlice, .pad, .dynamicPad,
             .gather, .dynamicGather, .scatter, .concatenate:
            return true
        default:
            return false
        }
    }

    /// Whether this is a control flow operation.
    public var isControlFlow: Bool {
        switch self {
        case .whileOp, .ifOp:
            return true
        default:
            return false
        }
    }

    /// The number of operands expected.
    public var operandCount: OperandCount {
        switch self {
        // Unary
        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt,
             .sine, .cosine, .tanh, .floor, .ceil, .sign,
             .not, .clz, .convert, .reshape, .transpose, .broadcastInDim,
             .iota, .tan, .logistic, .isFinite, .reverse, .fft, .sort,
             .expm1, .log1p, .cbrt, .roundNearestAfz, .roundNearestEven,
             .popcnt, .real, .imag, .cholesky, .bitcastConvert:
            return .exactly(1)

        // Binary
        case .add, .subtract, .multiply, .divide, .maximum, .minimum,
             .power, .compare, .and, .or, .xor, .dot, .dotGeneral,
             .convolution, .uniformQuantize, .uniformDequantize,
             .shiftLeft, .shiftRightArithmetic, .shiftRightLogical,
             .complex, .triangularSolve, .dynamicReshape, .dynamicBroadcastInDim,
             .dynamicIota:
            return .exactly(2)

        // Ternary
        case .select, .clamp, .reducePrecision:
            return .exactly(3)

        // Variable
        case .reduce, .gather, .scatter, .pad, .reduceWindow,
             .dynamicSlice, .dynamicUpdateSlice, .dynamicPad, .dynamicGather,
             .selectAndScatter, .map:
            return .atLeast(2)

        case .concatenate:
            return .atLeast(1)

        // Batch norm operations
        case .batchNormInference:
            return .exactly(5)  // input, scale, offset, mean, variance

        case .batchNormTraining:
            return .exactly(3)  // input, scale, offset

        case .batchNormGrad:
            return .exactly(5)  // input, scale, mean, variance, grad_output

        // Special
        case .constant, .rng:
            return .exactly(0)

        case .slice:
            return .exactly(1)

        case .whileOp, .ifOp:
            return .atLeast(1)

        case .rngBitGenerator:
            return .exactly(1)  // initial_state

        case .customCall:
            return .atLeast(1)  // variable inputs depending on target
        }
    }
}

/// Describes the expected operand count for an operation.
public enum OperandCount: Sendable {
    case exactly(Int)
    case atLeast(Int)
    case range(min: Int, max: Int)
}
