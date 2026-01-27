// AlgebraicSimplifier.swift
// MetalHLOCore
//
// Algebraic simplification pass that eliminates redundant operations.
// Inspired by XLA's AlgebraicSimplifier - runs to convergence.

import Foundation

// MARK: - Simplification Result

/// Result of attempting to simplify an operation.
public enum SimplificationResult: Sendable {
    /// Replace the operation with a different operation.
    case replaceWith(HLOOperation)

    /// Replace the operation's uses with an existing value (forward the operand).
    case replaceWithOperand(String)

    /// Replace with a new constant operation.
    case replaceWithConstant(ConstantValue, TensorType)
}

// MARK: - Simplification Rule Protocol

/// Protocol for algebraic simplification rules.
public protocol SimplificationRule: Sendable {
    /// Unique name for this rule.
    var name: String { get }

    /// Attempt to simplify an operation.
    ///
    /// - Parameters:
    ///   - op: The operation to simplify.
    ///   - definingOps: Map from value names to their defining operations.
    ///   - function: The containing function.
    /// - Returns: A simplification result if the rule applies, nil otherwise.
    func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult?
}

// MARK: - Algebraic Simplifier

/// Algebraic simplifier that runs simplification rules to convergence.
///
/// The simplifier applies a set of algebraic rules to eliminate redundant
/// operations, fold constants, and simplify shape operations.
public final class AlgebraicSimplifier: @unchecked Sendable {

    private let rules: [any SimplificationRule]
    private let maxIterations: Int

    /// Creates an algebraic simplifier with default rules.
    public init(maxIterations: Int = 100) {
        self.maxIterations = maxIterations
        self.rules = Self.defaultRules()
    }

    /// Creates an algebraic simplifier with custom rules.
    public init(rules: [any SimplificationRule], maxIterations: Int = 100) {
        self.rules = rules
        self.maxIterations = maxIterations
    }

    /// Returns the default set of simplification rules.
    public static func defaultRules() -> [any SimplificationRule] {
        return [
            // Identity rules (highest priority - most common)
            AddZeroRule(),
            MultiplyOneRule(),
            MultiplyZeroRule(),
            SubtractZeroRule(),
            DivideOneRule(),
            SubtractSelfRule(),
            DivideSelfRule(),

            // Inverse rules
            NegNegRule(),
            ExpLogRule(),
            LogExpRule(),

            // Constant folding
            BinaryConstantFoldingRule(),
            UnaryConstantFoldingRule(),

            // Shape simplifications
            ReshapeReshapeRule(),
            ReshapeToSameRule(),
            TransposeIdentityRule(),
            TransposeTransposeRule(),
        ]
    }

    /// Simplifies a function by applying rules to convergence.
    ///
    /// - Parameter function: The function to simplify.
    /// - Returns: The simplified function.
    public func simplify(_ function: HLOFunction) -> HLOFunction {
        var current = function
        var iteration = 0

        while iteration < maxIterations {
            iteration += 1

            if let simplified = runSimplificationPass(on: current) {
                current = simplified
            } else {
                // No changes - convergence reached
                break
            }
        }

        return current
    }

    /// Runs a single simplification pass over the function.
    private func runSimplificationPass(on function: HLOFunction) -> HLOFunction? {
        let definingOps = buildDefiningOpsMap(function)
        var changed = false
        var newOperations: [HLOOperation] = []
        var replacements: [String: String] = [:]  // old result -> new result

        for op in function.operations {
            // Apply any pending replacements to operands
            let updatedOp = applyReplacements(to: op, replacements: replacements)

            // Try each rule
            var simplified = false
            for rule in rules {
                if let result = rule.simplify(updatedOp, definingOps: definingOps, function: function) {
                    switch result {
                    case .replaceWith(let newOp):
                        newOperations.append(newOp)
                        if newOp.result != updatedOp.result {
                            replacements[updatedOp.result] = newOp.result
                        }
                        changed = true
                        simplified = true

                    case .replaceWithOperand(let operandName):
                        // Don't emit this op, just forward its uses to the operand
                        replacements[updatedOp.result] = operandName
                        changed = true
                        simplified = true

                    case .replaceWithConstant(let value, let type):
                        // Create a new constant operation
                        var attrs = HLOAttributes()
                        attrs.constantValue = value
                        let constOp = HLOOperation(
                            result: updatedOp.result,
                            kind: .constant,
                            operands: [],
                            resultType: type,
                            attributes: attrs
                        )
                        newOperations.append(constOp)
                        changed = true
                        simplified = true
                    }
                    break  // Only apply first matching rule
                }
            }

            if !simplified {
                newOperations.append(updatedOp)
            }
        }

        guard changed else { return nil }

        // Apply replacements to return values
        let newReturnValues = function.returnValues.map { replacements[$0] ?? $0 }

        return HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: newOperations,
            returnValues: newReturnValues
        )
    }

    private func buildDefiningOpsMap(_ function: HLOFunction) -> [String: (op: HLOOperation, index: Int)] {
        var map: [String: (op: HLOOperation, index: Int)] = [:]
        for (index, op) in function.operations.enumerated() {
            map[op.result] = (op, index)
        }
        return map
    }

    private func applyReplacements(to op: HLOOperation, replacements: [String: String]) -> HLOOperation {
        guard !replacements.isEmpty else { return op }

        let newOperands = op.operands.map { replacements[$0] ?? $0 }

        if newOperands == op.operands {
            return op
        }

        return HLOOperation(
            result: op.result,
            kind: op.kind,
            operands: newOperands,
            resultType: op.resultType,
            attributes: op.attributes
        )
    }
}

// MARK: - Helper Functions

/// Checks if a constant value is zero.
private func isZero(_ value: ConstantValue) -> Bool {
    switch value {
    case .scalar(let v):
        return v == 0.0
    case .splat(let v, _):
        return v == 0.0
    case .dense(let values, _):
        return values.allSatisfy { $0 == 0.0 }
    }
}

/// Checks if a constant value is one.
private func isOne(_ value: ConstantValue) -> Bool {
    switch value {
    case .scalar(let v):
        return v == 1.0
    case .splat(let v, _):
        return v == 1.0
    case .dense(let values, _):
        return values.allSatisfy { $0 == 1.0 }
    }
}

/// Gets the scalar value from a constant if it's a scalar or splat.
private func getScalarValue(_ value: ConstantValue) -> Double? {
    switch value {
    case .scalar(let v):
        return v
    case .splat(let v, _):
        return v
    case .dense:
        return nil
    }
}

// MARK: - Identity Rules

/// Rule: x + 0 = x, 0 + x = x
public struct AddZeroRule: SimplificationRule {
    public let name = "add_zero"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .add, op.operands.count == 2 else { return nil }

        // Check if first operand is zero constant
        if let def0 = definingOps[op.operands[0]]?.op,
           def0.kind == .constant,
           let value = def0.attributes.constantValue,
           isZero(value) {
            return .replaceWithOperand(op.operands[1])
        }

        // Check if second operand is zero constant
        if let def1 = definingOps[op.operands[1]]?.op,
           def1.kind == .constant,
           let value = def1.attributes.constantValue,
           isZero(value) {
            return .replaceWithOperand(op.operands[0])
        }

        return nil
    }
}

/// Rule: x * 1 = x, 1 * x = x
public struct MultiplyOneRule: SimplificationRule {
    public let name = "multiply_one"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .multiply, op.operands.count == 2 else { return nil }

        // Check if first operand is one constant
        if let def0 = definingOps[op.operands[0]]?.op,
           def0.kind == .constant,
           let value = def0.attributes.constantValue,
           isOne(value) {
            return .replaceWithOperand(op.operands[1])
        }

        // Check if second operand is one constant
        if let def1 = definingOps[op.operands[1]]?.op,
           def1.kind == .constant,
           let value = def1.attributes.constantValue,
           isOne(value) {
            return .replaceWithOperand(op.operands[0])
        }

        return nil
    }
}

/// Rule: x * 0 = 0, 0 * x = 0
public struct MultiplyZeroRule: SimplificationRule {
    public let name = "multiply_zero"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .multiply, op.operands.count == 2 else { return nil }

        // Check if first operand is zero constant
        if let def0 = definingOps[op.operands[0]]?.op,
           def0.kind == .constant,
           let value = def0.attributes.constantValue,
           isZero(value) {
            return .replaceWithConstant(.splat(0.0, op.resultType), op.resultType)
        }

        // Check if second operand is zero constant
        if let def1 = definingOps[op.operands[1]]?.op,
           def1.kind == .constant,
           let value = def1.attributes.constantValue,
           isZero(value) {
            return .replaceWithConstant(.splat(0.0, op.resultType), op.resultType)
        }

        return nil
    }
}

/// Rule: x - 0 = x
public struct SubtractZeroRule: SimplificationRule {
    public let name = "subtract_zero"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .subtract, op.operands.count == 2 else { return nil }

        // Check if second operand is zero constant (x - 0 = x)
        if let def1 = definingOps[op.operands[1]]?.op,
           def1.kind == .constant,
           let value = def1.attributes.constantValue,
           isZero(value) {
            return .replaceWithOperand(op.operands[0])
        }

        return nil
    }
}

/// Rule: x / 1 = x
public struct DivideOneRule: SimplificationRule {
    public let name = "divide_one"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .divide, op.operands.count == 2 else { return nil }

        // Check if second operand is one constant (x / 1 = x)
        if let def1 = definingOps[op.operands[1]]?.op,
           def1.kind == .constant,
           let value = def1.attributes.constantValue,
           isOne(value) {
            return .replaceWithOperand(op.operands[0])
        }

        return nil
    }
}

/// Rule: x - x = 0
public struct SubtractSelfRule: SimplificationRule {
    public let name = "subtract_self"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .subtract, op.operands.count == 2 else { return nil }

        // Check if both operands are the same
        if op.operands[0] == op.operands[1] {
            return .replaceWithConstant(.splat(0.0, op.resultType), op.resultType)
        }

        return nil
    }
}

/// Rule: x / x = 1
public struct DivideSelfRule: SimplificationRule {
    public let name = "divide_self"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .divide, op.operands.count == 2 else { return nil }

        // Check if both operands are the same
        if op.operands[0] == op.operands[1] {
            return .replaceWithConstant(.splat(1.0, op.resultType), op.resultType)
        }

        return nil
    }
}

// MARK: - Inverse Rules

/// Rule: -(-x) = x
public struct NegNegRule: SimplificationRule {
    public let name = "neg_neg"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .negate, op.operands.count == 1 else { return nil }

        // Check if operand is also a negate
        if let innerOp = definingOps[op.operands[0]]?.op,
           innerOp.kind == .negate,
           innerOp.operands.count == 1 {
            return .replaceWithOperand(innerOp.operands[0])
        }

        return nil
    }
}

/// Rule: exp(log(x)) = x
public struct ExpLogRule: SimplificationRule {
    public let name = "exp_log"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .exponential, op.operands.count == 1 else { return nil }

        // Check if operand is log
        if let innerOp = definingOps[op.operands[0]]?.op,
           innerOp.kind == .log,
           innerOp.operands.count == 1 {
            return .replaceWithOperand(innerOp.operands[0])
        }

        return nil
    }
}

/// Rule: log(exp(x)) = x
public struct LogExpRule: SimplificationRule {
    public let name = "log_exp"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .log, op.operands.count == 1 else { return nil }

        // Check if operand is exp
        if let innerOp = definingOps[op.operands[0]]?.op,
           innerOp.kind == .exponential,
           innerOp.operands.count == 1 {
            return .replaceWithOperand(innerOp.operands[0])
        }

        return nil
    }
}

// MARK: - Constant Folding Rules

/// Rule: const op const = result_const (for binary operations)
public struct BinaryConstantFoldingRule: SimplificationRule {
    public let name = "binary_constant_folding"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        // Only fold binary arithmetic operations
        guard op.kind.isBinaryArithmetic, op.operands.count == 2 else { return nil }

        // Both operands must be constants
        guard let def0 = definingOps[op.operands[0]]?.op,
              def0.kind == .constant,
              let val0 = def0.attributes.constantValue,
              let scalar0 = getScalarValue(val0) else { return nil }

        guard let def1 = definingOps[op.operands[1]]?.op,
              def1.kind == .constant,
              let val1 = def1.attributes.constantValue,
              let scalar1 = getScalarValue(val1) else { return nil }

        // Compute the result
        let result: Double
        switch op.kind {
        case .add:
            result = scalar0 + scalar1
        case .subtract:
            result = scalar0 - scalar1
        case .multiply:
            result = scalar0 * scalar1
        case .divide:
            guard scalar1 != 0 else { return nil }  // Don't fold division by zero
            result = scalar0 / scalar1
        case .maximum:
            result = max(scalar0, scalar1)
        case .minimum:
            result = min(scalar0, scalar1)
        case .power:
            result = pow(scalar0, scalar1)
        default:
            return nil
        }

        return .replaceWithConstant(.splat(result, op.resultType), op.resultType)
    }
}

/// Rule: unary(const) = result_const
public struct UnaryConstantFoldingRule: SimplificationRule {
    public let name = "unary_constant_folding"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.operands.count == 1 else { return nil }

        // Operand must be a constant
        guard let def0 = definingOps[op.operands[0]]?.op,
              def0.kind == .constant,
              let val0 = def0.attributes.constantValue,
              let scalar0 = getScalarValue(val0) else { return nil }

        // Compute the result based on operation
        let result: Double?
        switch op.kind {
        case .negate:
            result = -scalar0
        case .abs:
            result = Swift.abs(scalar0)
        case .exponential:
            result = exp(scalar0)
        case .log:
            guard scalar0 > 0 else { return nil }
            result = Foundation.log(scalar0)
        case .sqrt:
            guard scalar0 >= 0 else { return nil }
            result = Foundation.sqrt(scalar0)
        case .rsqrt:
            guard scalar0 > 0 else { return nil }
            result = 1.0 / Foundation.sqrt(scalar0)
        case .tanh:
            result = Foundation.tanh(scalar0)
        case .sine:
            result = sin(scalar0)
        case .cosine:
            result = cos(scalar0)
        case .floor:
            result = Foundation.floor(scalar0)
        case .ceil:
            result = Foundation.ceil(scalar0)
        case .sign:
            result = scalar0 > 0 ? 1.0 : (scalar0 < 0 ? -1.0 : 0.0)
        case .logistic:
            result = 1.0 / (1.0 + exp(-scalar0))
        default:
            result = nil
        }

        guard let r = result else { return nil }

        return .replaceWithConstant(.splat(r, op.resultType), op.resultType)
    }
}

// MARK: - Shape Simplification Rules

/// Rule: reshape(reshape(x)) = reshape(x) (with combined shape)
public struct ReshapeReshapeRule: SimplificationRule {
    public let name = "reshape_reshape"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .reshape, op.operands.count == 1 else { return nil }

        // Check if operand is also a reshape
        if let innerOp = definingOps[op.operands[0]]?.op,
           innerOp.kind == .reshape,
           innerOp.operands.count == 1 {
            // Combine: just use the original input with the final shape
            let newOp = HLOOperation(
                result: op.result,
                kind: .reshape,
                operands: [innerOp.operands[0]],
                resultType: op.resultType,
                attributes: op.attributes
            )
            return .replaceWith(newOp)
        }

        return nil
    }
}

/// Rule: reshape(x) where input shape == output shape = x
public struct ReshapeToSameRule: SimplificationRule {
    public let name = "reshape_to_same"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .reshape, op.operands.count == 1 else { return nil }

        // Get input type
        let inputType: TensorType?
        if let def = definingOps[op.operands[0]]?.op {
            inputType = def.resultType
        } else {
            // Input is a function argument
            inputType = function.inputs.first { $0.name == op.operands[0] }?.type
        }

        guard let inType = inputType else { return nil }

        // Check if shapes are the same
        if inType.shape == op.resultType.shape {
            return .replaceWithOperand(op.operands[0])
        }

        return nil
    }
}

/// Rule: transpose with identity permutation = x
public struct TransposeIdentityRule: SimplificationRule {
    public let name = "transpose_identity"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .transpose, op.operands.count == 1 else { return nil }

        // Check if dimensions form identity permutation [0, 1, 2, ...]
        guard let dims = op.attributes.dimensions else { return nil }

        let isIdentity = dims.enumerated().allSatisfy { $0.offset == $0.element }

        if isIdentity {
            return .replaceWithOperand(op.operands[0])
        }

        return nil
    }
}

/// Rule: transpose(transpose(x, p1), p2) = transpose(x, compose(p1, p2)) or x if identity
public struct TransposeTransposeRule: SimplificationRule {
    public let name = "transpose_transpose"

    public init() {}

    public func simplify(
        _ op: HLOOperation,
        definingOps: [String: (op: HLOOperation, index: Int)],
        function: HLOFunction
    ) -> SimplificationResult? {
        guard op.kind == .transpose,
              op.operands.count == 1,
              let outerDims = op.attributes.dimensions else { return nil }

        // Check if operand is also a transpose
        guard let innerOp = definingOps[op.operands[0]]?.op,
              innerOp.kind == .transpose,
              innerOp.operands.count == 1,
              let innerDims = innerOp.attributes.dimensions else { return nil }

        // Compose permutations: composedDims[i] = innerDims[outerDims[i]]
        let composedDims = outerDims.map { innerDims[$0] }

        // Check if result is identity
        let isIdentity = composedDims.enumerated().allSatisfy { $0.offset == $0.element }

        if isIdentity {
            return .replaceWithOperand(innerOp.operands[0])
        }

        // Otherwise, create a single transpose with composed permutation
        var newAttrs = HLOAttributes()
        newAttrs.dimensions = composedDims

        let newOp = HLOOperation(
            result: op.result,
            kind: .transpose,
            operands: [innerOp.operands[0]],
            resultType: op.resultType,
            attributes: newAttrs
        )

        return .replaceWith(newOp)
    }
}
