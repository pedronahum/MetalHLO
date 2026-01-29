// CustomCallHandlers.swift
// MetalHLOCore
//
// Handlers for fused operations emitted as custom_call by Magma.
// These handlers convert fused operations to efficient MPSGraph implementations
// or custom Metal kernels.

import Foundation
import MetalPerformanceShadersGraph

// MARK: - Custom Call Handler Protocol

/// Protocol for handling specific custom call targets.
///
/// Custom call handlers translate fused operations from Magma into
/// efficient MPSGraph implementations or custom Metal kernels.
public protocol CustomCallHandler {
    /// The target name this handler supports (e.g., "fused_attention")
    static var targetName: String { get }

    /// Emit the custom call as MPSGraph operations.
    ///
    /// - Parameters:
    ///   - operation: The HLO operation (custom_call)
    ///   - graph: The MPSGraph to emit into
    ///   - inputs: Input tensors (already resolved)
    ///   - config: Parsed backend configuration
    /// - Returns: The output tensor(s)
    func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        inputs: [MPSGraphTensor],
        config: [String: Any]
    ) throws -> [MPSGraphTensor]
}

// MARK: - Custom Call Registry

/// Registry for custom call handlers.
///
/// Manages the mapping from custom call target names to their handlers.
public final class CustomCallRegistry: @unchecked Sendable {

    /// Shared registry instance
    public static let shared = CustomCallRegistry()

    private var handlers: [String: any CustomCallHandler] = [:]
    private let lock = NSLock()

    private init() {
        registerDefaultHandlers()
    }

    /// Register a custom call handler
    public func register(_ handler: any CustomCallHandler) {
        lock.lock()
        defer { lock.unlock() }
        handlers[type(of: handler).targetName] = handler
    }

    /// Get handler for a target name
    public func handler(for target: String) -> (any CustomCallHandler)? {
        lock.lock()
        defer { lock.unlock() }
        return handlers[target]
    }

    /// Check if a target is supported
    public func isSupported(_ target: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return handlers[target] != nil
    }

    private func registerDefaultHandlers() {
        // Register all fused operation handlers
        register(FusedScaledDotProductAttentionHandler())
        register(FusedLayerNormHandler())
        register(FusedRMSNormHandler())
        register(FusedMatMulBiasActivationHandler())
        register(FusedSoftmaxHandler())
        register(FusedGeluHandler())
        register(FusedRoPEHandler())
        register(FusedFFNHandler())
        register(FusedElementwiseHandler())
    }

    /// List of supported targets
    public var supportedTargets: [String] {
        lock.lock()
        defer { lock.unlock() }
        return Array(handlers.keys).sorted()
    }
}

// MARK: - Backend Config Parser

/// Parses the JSON backend_config from custom_call operations
public enum BackendConfigParser {

    /// Parse a JSON config string into a dictionary
    public static func parse(_ config: String) -> [String: Any] {
        guard !config.isEmpty,
              let data = config.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        return json
    }

    /// Get a float value from config
    public static func getFloat(_ config: [String: Any], key: String, default defaultValue: Float) -> Float {
        if let value = config[key] as? Double {
            return Float(value)
        }
        if let value = config[key] as? Float {
            return value
        }
        if let value = config[key] as? Int {
            return Float(value)
        }
        return defaultValue
    }

    /// Get an int value from config
    public static func getInt(_ config: [String: Any], key: String, default defaultValue: Int) -> Int {
        if let value = config[key] as? Int {
            return value
        }
        if let value = config[key] as? Double {
            return Int(value)
        }
        return defaultValue
    }

    /// Get a bool value from config
    public static func getBool(_ config: [String: Any], key: String, default defaultValue: Bool) -> Bool {
        if let value = config[key] as? Bool {
            return value
        }
        if let value = config[key] as? Int {
            return value != 0
        }
        return defaultValue
    }

    /// Get a string value from config
    public static func getString(_ config: [String: Any], key: String, default defaultValue: String) -> String {
        if let value = config[key] as? String {
            return value
        }
        return defaultValue
    }

    /// Get an int array from config
    public static func getIntArray(_ config: [String: Any], key: String, default defaultValue: [Int]) -> [Int] {
        if let value = config[key] as? [Int] {
            return value
        }
        if let value = config[key] as? [Double] {
            return value.map { Int($0) }
        }
        return defaultValue
    }
}

// MARK: - Fused Scaled Dot-Product Attention Handler

/// Handles fused scaled dot-product attention
///
/// Implements: softmax(Q @ K^T * scale) @ V
/// With optional causal masking
///
/// On macOS 15+ / iOS 18+, uses the native MPSGraph SDPA operation for optimal performance.
/// Falls back to manual implementation on older systems.
public final class FusedScaledDotProductAttentionHandler: CustomCallHandler {

    public static let targetName = "fused_scaled_dot_product_attention"

    public init() {}

    public func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        inputs: [MPSGraphTensor],
        config: [String: Any]
    ) throws -> [MPSGraphTensor] {
        guard inputs.count >= 3 else {
            throw CustomCallError.invalidInputCount(expected: 3, got: inputs.count)
        }

        let q = inputs[0]  // [batch, heads, seq_q, head_dim]
        let k = inputs[1]  // [batch, heads, seq_k, head_dim]
        let v = inputs[2]  // [batch, heads, seq_k, head_dim]

        let scale = BackendConfigParser.getFloat(config, key: "scale", default: 1.0)
        let hasMask = BackendConfigParser.getBool(config, key: "has_mask", default: false)

        // Get optional mask tensor
        let mask: MPSGraphTensor? = (hasMask && inputs.count > 3) ? inputs[3] : nil

        // Try to use native MPSGraph SDPA (available on macOS 15+ / iOS 18+)
        if #available(macOS 15.0, iOS 18.0, *) {
            return try emitNativeSDPA(graph: graph, q: q, k: k, v: v, mask: mask, scale: scale)
        } else {
            return try emitManualSDPA(graph: graph, q: q, k: k, v: v, mask: mask, scale: scale)
        }
    }

    /// Uses native MPSGraph scaledDotProductAttention (macOS 15+ / iOS 18+)
    @available(macOS 15.0, iOS 18.0, *)
    private func emitNativeSDPA(
        graph: MPSGraph,
        q: MPSGraphTensor,
        k: MPSGraphTensor,
        v: MPSGraphTensor,
        mask: MPSGraphTensor?,
        scale: Float
    ) throws -> [MPSGraphTensor] {
        // Use native SDPA - much more efficient as it runs as a fused kernel
        let output = graph.scaledDotProductAttention(
            query: q,
            key: k,
            value: v,
            mask: mask,
            scale: scale,
            name: "native_sdpa"
        )

        return [output]
    }

    /// Manual SDPA implementation for older systems
    private func emitManualSDPA(
        graph: MPSGraph,
        q: MPSGraphTensor,
        k: MPSGraphTensor,
        v: MPSGraphTensor,
        mask: MPSGraphTensor?,
        scale: Float
    ) throws -> [MPSGraphTensor] {
        // Transpose K for matmul: [batch, heads, head_dim, seq_k]
        let kT = graph.transposeTensor(k, dimension: 2, withDimension: 3, name: nil)

        // Q @ K^T -> [batch, heads, seq_q, seq_k]
        var scores = graph.matrixMultiplication(
            primary: q,
            secondary: kT,
            name: "attention_scores"
        )

        // Scale
        if scale != 1.0 {
            let scaleConst = graph.constant(Double(scale), dataType: scores.dataType)
            scores = graph.multiplication(scores, scaleConst, name: "scaled_scores")
        }

        // Apply mask if provided
        if let mask = mask {
            scores = graph.addition(scores, mask, name: "masked_scores")
        }

        // Softmax along last dimension (seq_k)
        let weights = graph.softMax(with: scores, axis: -1, name: "attention_weights")

        // Weights @ V -> [batch, heads, seq_q, head_dim]
        let output = graph.matrixMultiplication(
            primary: weights,
            secondary: v,
            name: "attention_output"
        )

        return [output]
    }
}

// MARK: - Fused Layer Norm Handler

/// Handles fused layer normalization
///
/// Implements: (x - mean) / sqrt(var + eps) * gamma + beta
public final class FusedLayerNormHandler: CustomCallHandler {

    public static let targetName = "fused_layer_norm"

    public init() {}

    public func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        inputs: [MPSGraphTensor],
        config: [String: Any]
    ) throws -> [MPSGraphTensor] {
        guard inputs.count >= 3 else {
            throw CustomCallError.invalidInputCount(expected: 3, got: inputs.count)
        }

        let x = inputs[0]      // input
        let gamma = inputs[1]  // scale
        let beta = inputs[2]   // offset

        let eps = BackendConfigParser.getFloat(config, key: "eps", default: 1e-5)
        let axes = BackendConfigParser.getIntArray(config, key: "axes", default: [-1])

        // Get input shape
        guard let inputShape = x.shape else {
            throw CustomCallError.invalidConfig("Input tensor has no shape")
        }
        let rank = inputShape.count

        // Normalize axis (handle negative indices)
        let normalizedAxes = axes.map { $0 < 0 ? rank + $0 : $0 }

        // Compute shape for keepDims semantics: replace reduced dims with 1
        var keepDimsShape = inputShape.map { $0.intValue }
        for axis in normalizedAxes {
            keepDimsShape[axis] = 1
        }
        let keepDimsShapeNS = keepDimsShape.map { NSNumber(value: $0) }

        // Compute mean and reshape to keep dimensions
        let meanReduced = graph.mean(of: x, axes: normalizedAxes.map { NSNumber(value: $0) }, name: "ln_mean_reduced")
        let mean = graph.reshape(meanReduced, shape: keepDimsShapeNS, name: "ln_mean")

        // Compute variance: mean((x - mean)^2)
        let centered = graph.subtraction(x, mean, name: "ln_centered")
        let squared = graph.square(with: centered, name: "ln_squared")
        let varianceReduced = graph.mean(of: squared, axes: normalizedAxes.map { NSNumber(value: $0) }, name: "ln_variance_reduced")
        let variance = graph.reshape(varianceReduced, shape: keepDimsShapeNS, name: "ln_variance")

        // Normalize: (x - mean) / sqrt(variance + eps)
        let epsConst = graph.constant(Double(eps), dataType: x.dataType)
        let varPlusEps = graph.addition(variance, epsConst, name: "ln_var_eps")
        let stddev = graph.squareRoot(with: varPlusEps, name: "ln_stddev")
        let normalized = graph.division(centered, stddev, name: "ln_normalized")

        // Scale and shift: normalized * gamma + beta
        let scaled = graph.multiplication(normalized, gamma, name: "ln_scaled")
        let output = graph.addition(scaled, beta, name: "ln_output")

        return [output]
    }
}

// MARK: - Fused RMS Norm Handler

/// Handles fused RMS normalization
///
/// Implements: x / sqrt(mean(x^2) + eps) * weight
public final class FusedRMSNormHandler: CustomCallHandler {

    public static let targetName = "fused_rms_norm"

    public init() {}

    public func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        inputs: [MPSGraphTensor],
        config: [String: Any]
    ) throws -> [MPSGraphTensor] {
        guard inputs.count >= 2 else {
            throw CustomCallError.invalidInputCount(expected: 2, got: inputs.count)
        }

        let x = inputs[0]       // input
        let weight = inputs[1]  // scale weight

        let eps = BackendConfigParser.getFloat(config, key: "eps", default: 1e-5)
        let axes = BackendConfigParser.getIntArray(config, key: "axes", default: [-1])

        // Normalize axis
        let rank = x.shape?.count ?? 1
        let normalizedAxes = axes.map { $0 < 0 ? rank + $0 : $0 }

        // Compute RMS: sqrt(mean(x^2) + eps)
        let squared = graph.square(with: x, name: "rms_squared")
        let meanSquared = graph.mean(of: squared, axes: normalizedAxes.map { NSNumber(value: $0) }, name: "rms_mean")
        let epsConst = graph.constant(Double(eps), dataType: x.dataType)
        let meanPlusEps = graph.addition(meanSquared, epsConst, name: "rms_eps")
        let rms = graph.squareRoot(with: meanPlusEps, name: "rms")

        // Normalize: x / rms
        let normalized = graph.division(x, rms, name: "rms_normalized")

        // Scale: normalized * weight
        let output = graph.multiplication(normalized, weight, name: "rms_output")

        return [output]
    }
}

// MARK: - Fused MatMul + Bias + Activation Handler

/// Handles fused matrix multiply with bias and activation
///
/// Implements: activation(x @ w + bias)
public final class FusedMatMulBiasActivationHandler: CustomCallHandler {

    public static let targetName = "fused_matmul_bias_activation"

    public init() {}

    public func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        inputs: [MPSGraphTensor],
        config: [String: Any]
    ) throws -> [MPSGraphTensor] {
        guard inputs.count >= 3 else {
            throw CustomCallError.invalidInputCount(expected: 3, got: inputs.count)
        }

        let x = inputs[0]     // input
        let w = inputs[1]     // weight
        let bias = inputs[2]  // bias

        let activation = BackendConfigParser.getString(config, key: "activation", default: "none")
        let transA = BackendConfigParser.getBool(config, key: "trans_a", default: false)
        let transB = BackendConfigParser.getBool(config, key: "trans_b", default: false)

        // Transpose if needed
        var lhs = x
        var rhs = w

        if transA {
            let rank = lhs.shape?.count ?? 2
            lhs = graph.transposeTensor(lhs, dimension: rank - 2, withDimension: rank - 1, name: nil)
        }

        if transB {
            let rank = rhs.shape?.count ?? 2
            rhs = graph.transposeTensor(rhs, dimension: rank - 2, withDimension: rank - 1, name: nil)
        }

        // MatMul
        var result = graph.matrixMultiplication(primary: lhs, secondary: rhs, name: "fused_matmul")

        // Add bias
        result = graph.addition(result, bias, name: "fused_bias")

        // Apply activation
        result = applyActivation(result, activation: activation, graph: graph)

        return [result]
    }

    private func applyActivation(_ tensor: MPSGraphTensor, activation: String, graph: MPSGraph) -> MPSGraphTensor {
        switch activation {
        case "relu":
            return graph.reLU(with: tensor, name: "fused_relu")
        case "gelu":
            return applyGelu(tensor, approximate: false, graph: graph)
        case "gelu_approximate", "geluApproximate":
            return applyGelu(tensor, approximate: true, graph: graph)
        case "sigmoid":
            return graph.sigmoid(with: tensor, name: "fused_sigmoid")
        case "tanh":
            return graph.tanh(with: tensor, name: "fused_tanh")
        case "silu":
            // SiLU: x * sigmoid(x)
            let sig = graph.sigmoid(with: tensor, name: "fused_silu_sig")
            return graph.multiplication(tensor, sig, name: "fused_silu")
        case "leaky_relu", "leakyRelu":
            // LeakyReLU with default alpha=0.01
            return graph.leakyReLU(with: tensor, alpha: 0.01, name: "fused_leaky_relu")
        case "elu":
            // ELU: x if x > 0, else alpha * (exp(x) - 1)
            let alpha = graph.constant(1.0, dataType: tensor.dataType)
            let zero = graph.constant(0.0, dataType: tensor.dataType)
            let one = graph.constant(1.0, dataType: tensor.dataType)
            let expX = graph.exponent(with: tensor, name: nil)
            let expMinus1 = graph.subtraction(expX, one, name: nil)
            let negPart = graph.multiplication(alpha, expMinus1, name: nil)
            let condition = graph.greaterThan(tensor, zero, name: nil)
            return graph.select(predicate: condition, trueTensor: tensor, falseTensor: negPart, name: "fused_elu")
        default:
            return tensor  // No activation
        }
    }

    private func applyGelu(_ tensor: MPSGraphTensor, approximate: Bool, graph: MPSGraph) -> MPSGraphTensor {
        if approximate {
            // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let half = graph.constant(0.5, dataType: tensor.dataType)
            let one = graph.constant(1.0, dataType: tensor.dataType)
            let sqrtTwoPi = graph.constant(0.7978845608, dataType: tensor.dataType)
            let coeff = graph.constant(0.044715, dataType: tensor.dataType)

            let x3 = graph.multiplication(tensor, graph.multiplication(tensor, tensor, name: nil), name: nil)
            let inner = graph.addition(tensor, graph.multiplication(coeff, x3, name: nil), name: nil)
            let tanhArg = graph.multiplication(sqrtTwoPi, inner, name: nil)
            let tanhVal = graph.tanh(with: tanhArg, name: nil)
            let onePlusTanh = graph.addition(one, tanhVal, name: nil)
            return graph.multiplication(half, graph.multiplication(tensor, onePlusTanh, name: nil), name: "fused_gelu")
        } else {
            // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            // MPSGraph doesn't have erf, so use approximate for now
            return applyGelu(tensor, approximate: true, graph: graph)
        }
    }
}

// MARK: - Fused Softmax Handler

/// Handles fused softmax (numerically stable)
public final class FusedSoftmaxHandler: CustomCallHandler {

    public static let targetName = "fused_softmax"

    public init() {}

    public func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        inputs: [MPSGraphTensor],
        config: [String: Any]
    ) throws -> [MPSGraphTensor] {
        guard inputs.count >= 1 else {
            throw CustomCallError.invalidInputCount(expected: 1, got: inputs.count)
        }

        let x = inputs[0]
        let axis = BackendConfigParser.getInt(config, key: "axis", default: -1)

        // Normalize axis
        let rank = x.shape?.count ?? 1
        let normalizedAxis = axis < 0 ? rank + axis : axis

        // Use built-in softmax (already numerically stable)
        let output = graph.softMax(with: x, axis: normalizedAxis, name: "fused_softmax")

        return [output]
    }
}

// MARK: - Fused GELU Handler

/// Handles fused GELU activation
public final class FusedGeluHandler: CustomCallHandler {

    public static let targetName = "fused_gelu"

    public init() {}

    public func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        inputs: [MPSGraphTensor],
        config: [String: Any]
    ) throws -> [MPSGraphTensor] {
        guard inputs.count >= 1 else {
            throw CustomCallError.invalidInputCount(expected: 1, got: inputs.count)
        }

        let x = inputs[0]
        let approximate = BackendConfigParser.getBool(config, key: "approximate", default: false)

        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let half = graph.constant(0.5, dataType: x.dataType)
        let one = graph.constant(1.0, dataType: x.dataType)
        let sqrtTwoPi = graph.constant(0.7978845608, dataType: x.dataType)
        let coeff = graph.constant(0.044715, dataType: x.dataType)

        let x3 = graph.multiplication(x, graph.multiplication(x, x, name: nil), name: nil)
        let inner = graph.addition(x, graph.multiplication(coeff, x3, name: nil), name: nil)
        let tanhArg = graph.multiplication(sqrtTwoPi, inner, name: nil)
        let tanhVal = graph.tanh(with: tanhArg, name: nil)
        let onePlusTanh = graph.addition(one, tanhVal, name: nil)
        let output = graph.multiplication(half, graph.multiplication(x, onePlusTanh, name: nil), name: "fused_gelu")

        return [output]
    }
}

// MARK: - Fused RoPE Handler

/// Handles fused Rotary Position Embedding
///
/// RoPE applies rotation to pairs of elements in the head dimension:
/// For each position i with frequency theta:
///   (x[2i], x[2i+1]) -> (x[2i]*cos - x[2i+1]*sin, x[2i]*sin + x[2i+1]*cos)
public final class FusedRoPEHandler: CustomCallHandler {

    public static let targetName = "fused_rope"

    public init() {}

    public func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        inputs: [MPSGraphTensor],
        config: [String: Any]
    ) throws -> [MPSGraphTensor] {
        guard inputs.count >= 2 else {
            throw CustomCallError.invalidInputCount(expected: 2, got: inputs.count)
        }

        let x = inputs[0]       // input: [batch, seq, heads, head_dim] or [batch, heads, seq, head_dim]
        let freqs = inputs[1]   // frequencies: [seq, head_dim] contains interleaved cos/sin

        guard let xShape = x.shape, xShape.count >= 2 else {
            throw CustomCallError.invalidConfig("RoPE input must have at least 2 dimensions")
        }

        let rank = xShape.count
        let headDim = xShape[rank - 1].intValue
        let halfDim = headDim / 2

        // Split x into first half and second half along last dimension
        // x1 = x[..., :halfDim], x2 = x[..., halfDim:]
        var startIndices1 = [Int](repeating: 0, count: rank)
        var endIndices1 = xShape.map { $0.intValue }
        endIndices1[rank - 1] = halfDim

        var startIndices2 = [Int](repeating: 0, count: rank)
        startIndices2[rank - 1] = halfDim
        let endIndices2 = xShape.map { $0.intValue }

        let x1 = graph.sliceTensor(
            x,
            starts: startIndices1.map { NSNumber(value: $0) },
            ends: endIndices1.map { NSNumber(value: $0) },
            strides: [Int](repeating: 1, count: rank).map { NSNumber(value: $0) },
            name: "rope_x1"
        )

        let x2 = graph.sliceTensor(
            x,
            starts: startIndices2.map { NSNumber(value: $0) },
            ends: endIndices2.map { NSNumber(value: $0) },
            strides: [Int](repeating: 1, count: rank).map { NSNumber(value: $0) },
            name: "rope_x2"
        )

        // Extract cos and sin from freqs
        // freqs is typically [seq, head_dim] with cos in first half, sin in second half
        // Or it might be [seq, head_dim/2] for cos and a separate tensor for sin
        guard let freqsShape = freqs.shape, freqsShape.count >= 1 else {
            throw CustomCallError.invalidConfig("RoPE frequencies must have at least 1 dimension")
        }

        let freqsRank = freqsShape.count
        let freqsDim = freqsShape[freqsRank - 1].intValue

        let cosFreqs: MPSGraphTensor
        let sinFreqs: MPSGraphTensor

        if freqsDim == headDim {
            // freqs contains both cos and sin interleaved or concatenated
            // Assume first half is cos, second half is sin
            var cosStart = [Int](repeating: 0, count: freqsRank)
            var cosEnd = freqsShape.map { $0.intValue }
            cosEnd[freqsRank - 1] = halfDim

            var sinStart = [Int](repeating: 0, count: freqsRank)
            sinStart[freqsRank - 1] = halfDim
            let sinEnd = freqsShape.map { $0.intValue }

            cosFreqs = graph.sliceTensor(
                freqs,
                starts: cosStart.map { NSNumber(value: $0) },
                ends: cosEnd.map { NSNumber(value: $0) },
                strides: [Int](repeating: 1, count: freqsRank).map { NSNumber(value: $0) },
                name: "rope_cos"
            )

            sinFreqs = graph.sliceTensor(
                freqs,
                starts: sinStart.map { NSNumber(value: $0) },
                ends: sinEnd.map { NSNumber(value: $0) },
                strides: [Int](repeating: 1, count: freqsRank).map { NSNumber(value: $0) },
                name: "rope_sin"
            )
        } else if freqsDim == halfDim {
            // freqs is just the angle - compute cos/sin
            cosFreqs = graph.cos(with: freqs, name: "rope_cos")
            sinFreqs = graph.sin(with: freqs, name: "rope_sin")
        } else {
            // Treat freqs as-is for cos, assume third input is sin if available
            cosFreqs = freqs
            if inputs.count >= 3 {
                sinFreqs = inputs[2]
            } else {
                // Compute sin from cos: sin = sqrt(1 - cos^2) - but this loses sign
                // Better to just use freqs as angles
                sinFreqs = graph.sin(with: freqs, name: "rope_sin")
            }
        }

        // Apply rotation:
        // out1 = x1 * cos - x2 * sin
        // out2 = x1 * sin + x2 * cos
        let x1Cos = graph.multiplication(x1, cosFreqs, name: "rope_x1_cos")
        let x2Sin = graph.multiplication(x2, sinFreqs, name: "rope_x2_sin")
        let x1Sin = graph.multiplication(x1, sinFreqs, name: "rope_x1_sin")
        let x2Cos = graph.multiplication(x2, cosFreqs, name: "rope_x2_cos")

        let out1 = graph.subtraction(x1Cos, x2Sin, name: "rope_out1")
        let out2 = graph.addition(x1Sin, x2Cos, name: "rope_out2")

        // Concatenate back together along last dimension
        let output = graph.concatTensors([out1, out2], dimension: rank - 1, name: "rope_output")

        return [output]
    }
}

// MARK: - Fused FFN Handler

/// Handles fused Feed-Forward Network (up_proj → activation → down_proj)
public final class FusedFFNHandler: CustomCallHandler {

    public static let targetName = "fused_ffn"

    public init() {}

    public func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        inputs: [MPSGraphTensor],
        config: [String: Any]
    ) throws -> [MPSGraphTensor] {
        // Inputs: [input, up_weight, down_weight]
        guard inputs.count >= 3 else {
            throw CustomCallError.invalidInputCount(expected: 3, got: inputs.count)
        }

        let x = inputs[0]           // input tensor
        let upWeight = inputs[1]    // up projection weight
        let downWeight = inputs[2]  // down projection weight

        let activation = BackendConfigParser.getString(config, key: "activation", default: "gelu")
        let _ = BackendConfigParser.getBool(config, key: "is_gated", default: false)  // Reserved for gated FFN

        // Up projection: x @ up_weight
        var hidden = graph.matrixMultiplication(primary: x, secondary: upWeight, name: "ffn_up_proj")

        // Apply activation
        hidden = applyActivation(hidden, activation: activation, graph: graph)

        // Down projection: hidden @ down_weight
        let output = graph.matrixMultiplication(primary: hidden, secondary: downWeight, name: "ffn_down_proj")

        return [output]
    }

    private func applyActivation(_ tensor: MPSGraphTensor, activation: String, graph: MPSGraph) -> MPSGraphTensor {
        switch activation {
        case "relu":
            return graph.reLU(with: tensor, name: "ffn_relu")
        case "gelu":
            return applyGelu(tensor, approximate: true, graph: graph)
        case "gelu_approximate", "geluApproximate":
            return applyGelu(tensor, approximate: true, graph: graph)
        case "gelu_exact":
            return applyGelu(tensor, approximate: false, graph: graph)
        case "sigmoid":
            return graph.sigmoid(with: tensor, name: "ffn_sigmoid")
        case "tanh":
            return graph.tanh(with: tensor, name: "ffn_tanh")
        case "silu", "swish":
            // SiLU/Swish: x * sigmoid(x)
            let sig = graph.sigmoid(with: tensor, name: "ffn_silu_sig")
            return graph.multiplication(tensor, sig, name: "ffn_silu")
        default:
            return tensor  // No activation
        }
    }

    private func applyGelu(_ tensor: MPSGraphTensor, approximate: Bool, graph: MPSGraph) -> MPSGraphTensor {
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let half = graph.constant(0.5, dataType: tensor.dataType)
        let one = graph.constant(1.0, dataType: tensor.dataType)
        let sqrtTwoPi = graph.constant(0.7978845608, dataType: tensor.dataType)
        let coeff = graph.constant(0.044715, dataType: tensor.dataType)

        let x3 = graph.multiplication(tensor, graph.multiplication(tensor, tensor, name: nil), name: nil)
        let inner = graph.addition(tensor, graph.multiplication(coeff, x3, name: nil), name: nil)
        let tanhArg = graph.multiplication(sqrtTwoPi, inner, name: nil)
        let tanhVal = graph.tanh(with: tanhArg, name: nil)
        let onePlusTanh = graph.addition(one, tanhVal, name: nil)
        return graph.multiplication(half, graph.multiplication(tensor, onePlusTanh, name: nil), name: "ffn_gelu")
    }
}

// MARK: - Fused Elementwise Handler

/// Handles fused elementwise operation chains (e.g., add -> multiply)
public final class FusedElementwiseHandler: CustomCallHandler {

    public static let targetName = "fused_elementwise"

    public init() {}

    public func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        inputs: [MPSGraphTensor],
        config: [String: Any]
    ) throws -> [MPSGraphTensor] {
        // Parse the operations from config
        // Support both "ops" and "operations" keys for compatibility
        guard let opsArray = (config["ops"] as? [String]) ?? (config["operations"] as? [String]) else {
            throw CustomCallError.invalidConfig("Missing 'ops' or 'operations' in fused_elementwise config")
        }

        guard !inputs.isEmpty else {
            throw CustomCallError.invalidInputCount(expected: 1, got: 0)
        }

        // For the add-multiply chain with 3 inputs (arg0, arg1, arg2):
        // ops = ["add", "multiply"]
        // %0 = add(arg0, arg1)
        // %1 = multiply(%0, arg2)
        //
        // We need to apply each operation in sequence, consuming inputs as needed

        var currentResult: MPSGraphTensor? = nil
        var inputIndex = 0

        for (opIndex, opName) in opsArray.enumerated() {
            // Determine left operand: either previous result or next input
            let left: MPSGraphTensor
            if let prev = currentResult {
                left = prev
            } else if inputIndex < inputs.count {
                left = inputs[inputIndex]
                inputIndex += 1
            } else {
                throw CustomCallError.invalidConfig("Not enough inputs for operation \(opIndex)")
            }

            // Determine right operand (for binary ops)
            let result: MPSGraphTensor
            switch opName {
            // Binary operations
            case "add":
                guard inputIndex < inputs.count else {
                    throw CustomCallError.invalidConfig("Not enough inputs for add operation")
                }
                let right = inputs[inputIndex]
                inputIndex += 1
                result = graph.addition(left, right, name: "fused_add_\(opIndex)")

            case "subtract":
                guard inputIndex < inputs.count else {
                    throw CustomCallError.invalidConfig("Not enough inputs for subtract operation")
                }
                let right = inputs[inputIndex]
                inputIndex += 1
                result = graph.subtraction(left, right, name: "fused_sub_\(opIndex)")

            case "multiply":
                guard inputIndex < inputs.count else {
                    throw CustomCallError.invalidConfig("Not enough inputs for multiply operation")
                }
                let right = inputs[inputIndex]
                inputIndex += 1
                result = graph.multiplication(left, right, name: "fused_mul_\(opIndex)")

            case "divide":
                guard inputIndex < inputs.count else {
                    throw CustomCallError.invalidConfig("Not enough inputs for divide operation")
                }
                let right = inputs[inputIndex]
                inputIndex += 1
                result = graph.division(left, right, name: "fused_div_\(opIndex)")

            case "maximum":
                guard inputIndex < inputs.count else {
                    throw CustomCallError.invalidConfig("Not enough inputs for maximum operation")
                }
                let right = inputs[inputIndex]
                inputIndex += 1
                result = graph.maximum(left, right, name: "fused_max_\(opIndex)")

            case "minimum":
                guard inputIndex < inputs.count else {
                    throw CustomCallError.invalidConfig("Not enough inputs for minimum operation")
                }
                let right = inputs[inputIndex]
                inputIndex += 1
                result = graph.minimum(left, right, name: "fused_min_\(opIndex)")

            // Unary operations
            case "negate":
                result = graph.negative(with: left, name: "fused_neg_\(opIndex)")

            case "abs":
                result = graph.absolute(with: left, name: "fused_abs_\(opIndex)")

            case "exponential":
                result = graph.exponent(with: left, name: "fused_exp_\(opIndex)")

            case "log":
                result = graph.logarithm(with: left, name: "fused_log_\(opIndex)")

            case "sqrt":
                result = graph.squareRoot(with: left, name: "fused_sqrt_\(opIndex)")

            case "rsqrt":
                result = graph.reverseSquareRoot(with: left, name: "fused_rsqrt_\(opIndex)")

            case "tanh":
                result = graph.tanh(with: left, name: "fused_tanh_\(opIndex)")

            case "logistic":
                result = graph.sigmoid(with: left, name: "fused_sigmoid_\(opIndex)")

            case "sine":
                result = graph.sin(with: left, name: "fused_sin_\(opIndex)")

            case "cosine":
                result = graph.cos(with: left, name: "fused_cos_\(opIndex)")

            case "floor":
                result = graph.floor(with: left, name: "fused_floor_\(opIndex)")

            case "ceil":
                result = graph.ceil(with: left, name: "fused_ceil_\(opIndex)")

            default:
                throw CustomCallError.invalidConfig("Unsupported operation in fused chain: \(opName)")
            }

            currentResult = result
        }

        guard let finalResult = currentResult else {
            throw CustomCallError.invalidConfig("No operations to fuse")
        }

        return [finalResult]
    }
}

// MARK: - Custom Call Error

/// Errors that can occur during custom call handling
public enum CustomCallError: Error, CustomStringConvertible {
    case invalidInputCount(expected: Int, got: Int)
    case unsupportedTarget(String)
    case invalidConfig(String)
    case emissionFailed(String)

    public var description: String {
        switch self {
        case .invalidInputCount(let expected, let got):
            return "Invalid input count: expected \(expected), got \(got)"
        case .unsupportedTarget(let target):
            return "Unsupported custom call target: \(target)"
        case .invalidConfig(let reason):
            return "Invalid backend config: \(reason)"
        case .emissionFailed(let reason):
            return "Custom call emission failed: \(reason)"
        }
    }
}
