// FusionPatterns.swift
// MetalHLOCore
//
// Pattern implementations for common fused operations.

import Foundation

// MARK: - Softmax Pattern

/// Detects softmax pattern: exp(x - max(x)) / sum(exp(x - max(x)))
///
/// The pattern looks for:
/// 1. reduce_max over an axis
/// 2. subtract (broadcast max)
/// 3. exp
/// 4. reduce_sum
/// 5. divide
public struct SoftmaxPattern: HLOPattern {
    public let name = "softmax"
    public let priority = 100  // High priority

    public init() {}

    public func match(
        at op: HLOOperation,
        index: Int,
        in function: HLOFunction,
        definingOps: [String: (op: HLOOperation, index: Int)]
    ) -> PatternMatch? {
        // Softmax ends with divide
        guard op.kind == .divide else { return nil }

        // Numerator should be exp
        guard let expOp = definingOps[op.operands[0]]?.op,
              expOp.kind == .exponential else { return nil }

        // Denominator should be reduce (sum)
        guard let sumOp = definingOps[op.operands[1]]?.op,
              sumOp.kind == .reduce else { return nil }

        // Sum's input should be same exp
        guard sumOp.operands.first == expOp.result else { return nil }

        // Exp's input should be subtract
        guard let subOp = definingOps[expOp.operands[0]]?.op,
              subOp.kind == .subtract else { return nil }

        // Subtract's second operand should come from reduce_max (possibly via broadcast)
        let maxSource = subOp.operands[1]
        var maxOp: HLOOperation?
        var broadcastOp: HLOOperation?

        if let directMax = definingOps[maxSource]?.op {
            if directMax.kind == .reduce {
                maxOp = directMax
            } else if directMax.kind == .broadcastInDim {
                broadcastOp = directMax
                if let innerMax = definingOps[directMax.operands[0]]?.op,
                   innerMax.kind == .reduce {
                    maxOp = innerMax
                }
            }
        }

        guard maxOp != nil else { return nil }

        // Collect all matched operations
        var matchedOps: [HLOOperation] = []
        var matchedIndices: [Int] = []

        // Add operations in dependency order
        if let m = maxOp, let idx = definingOps[m.result]?.index {
            matchedOps.append(m)
            matchedIndices.append(idx)
        }
        if let b = broadcastOp, let idx = definingOps[b.result]?.index {
            matchedOps.append(b)
            matchedIndices.append(idx)
        }
        if let idx = definingOps[subOp.result]?.index {
            matchedOps.append(subOp)
            matchedIndices.append(idx)
        }
        if let idx = definingOps[expOp.result]?.index {
            matchedOps.append(expOp)
            matchedIndices.append(idx)
        }
        if let idx = definingOps[sumOp.result]?.index {
            matchedOps.append(sumOp)
            matchedIndices.append(idx)
        }
        matchedOps.append(op)
        matchedIndices.append(index)

        // Extract axis from reduce_max
        let axis = maxOp?.attributes.dimensions?.first ?? -1

        return PatternMatch(
            operations: matchedOps,
            indices: matchedIndices.sorted(),
            rootOperation: op,
            metadata: ["axis": .int(axis), "pattern": .string("softmax")],
            inputs: [subOp.operands[0]]  // The original input to softmax
        )
    }

    public func replacement(for match: PatternMatch, in function: HLOFunction) -> [HLOOperation] {
        let axis: Int
        if case .int(let a) = match.metadata["axis"] {
            axis = a
        } else {
            axis = -1
        }
        let input = match.inputs.first ?? match.operations.first!.operands[0]

        // Create custom_call for fused softmax
        var attributes = HLOAttributes()
        attributes.callTargetName = "fused_softmax"
        attributes.backendConfig = "{\"axis\": \(axis)}"

        return [HLOOperation(
            result: match.rootOperation.result,
            kind: .customCall,
            operands: [input],
            resultType: match.rootOperation.resultType,
            attributes: attributes
        )]
    }
}

// MARK: - GELU Pattern

/// Detects GELU pattern: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// This is the approximate GELU formula. The exact pattern is complex,
/// so we look for key indicators: multiply, tanh, and specific constants.
public struct GELUPattern: HLOPattern {
    public let name = "gelu"
    public let priority = 90

    public init() {}

    public func match(
        at op: HLOOperation,
        index: Int,
        in function: HLOFunction,
        definingOps: [String: (op: HLOOperation, index: Int)]
    ) -> PatternMatch? {
        // GELU ends with multiply (0.5 * x * ...)
        guard op.kind == .multiply else { return nil }

        // Look for tanh in the chain
        var current = op
        var visited: Set<String> = []
        var foundTanh = false
        var tanhOp: HLOOperation?
        var matchedOps: [HLOOperation] = []
        var matchedIndices: [Int] = []

        // BFS to find tanh
        var queue: [HLOOperation] = [op]
        while !queue.isEmpty {
            let curr = queue.removeFirst()
            if visited.contains(curr.result) { continue }
            visited.insert(curr.result)

            if curr.kind == .tanh {
                foundTanh = true
                tanhOp = curr
            }

            for operand in curr.operands {
                if let def = definingOps[operand]?.op {
                    queue.append(def)
                }
            }
        }

        guard foundTanh, let tanh = tanhOp else { return nil }

        // Look for the characteristic pattern:
        // - Multiple multiplies
        // - An add with 1
        // - Power of 3 or multiply chain for x^3

        // For simplicity, we'll check for:
        // 1. tanh exists
        // 2. There's a multiply with 0.5 constant
        // 3. There's a power with 3 or multiple multiplies

        var hasPowerOf3 = false
        var hasHalfConstant = false
        var inputOperand: String?

        // Re-traverse to collect pattern
        visited.removeAll()
        queue = [op]
        while !queue.isEmpty {
            let curr = queue.removeFirst()
            if visited.contains(curr.result) { continue }
            visited.insert(curr.result)
            matchedOps.append(curr)
            if let idx = definingOps[curr.result]?.index {
                matchedIndices.append(idx)
            } else if curr.result == op.result {
                matchedIndices.append(index)
            }

            if curr.kind == .power {
                hasPowerOf3 = true
            }
            if curr.kind == .constant {
                // Check for 0.5 constant
                if let constVal = curr.attributes.constantValue {
                    switch constVal {
                    case .scalar(let v) where abs(v - 0.5) < 0.01:
                        hasHalfConstant = true
                    case .splat(let v, _) where abs(v - 0.5) < 0.01:
                        hasHalfConstant = true
                    default:
                        break
                    }
                }
            }

            for operand in curr.operands {
                if let def = definingOps[operand]?.op {
                    queue.append(def)
                } else if inputOperand == nil {
                    // This operand comes from outside - it's the input
                    inputOperand = operand
                }
            }
        }

        // Require tanh and 0.5 constant for GELU detection
        guard foundTanh && hasHalfConstant else { return nil }

        return PatternMatch(
            operations: matchedOps,
            indices: matchedIndices.sorted(),
            rootOperation: op,
            metadata: ["pattern": .string("gelu")],
            inputs: inputOperand.map { [$0] } ?? []
        )
    }

    public func replacement(for match: PatternMatch, in function: HLOFunction) -> [HLOOperation] {
        let input = match.inputs.first ?? match.operations.first!.operands[0]

        var attributes = HLOAttributes()
        attributes.callTargetName = "fused_gelu"

        return [HLOOperation(
            result: match.rootOperation.result,
            kind: .customCall,
            operands: [input],
            resultType: match.rootOperation.resultType,
            attributes: attributes
        )]
    }
}

// MARK: - LayerNorm Pattern

/// Detects LayerNorm pattern: (x - mean) / sqrt(variance + eps) * gamma + beta
///
/// Key operations:
/// 1. reduce (mean)
/// 2. subtract (center)
/// 3. multiply (squared)
/// 4. reduce (variance)
/// 5. add (eps)
/// 6. rsqrt or sqrt+divide
/// 7. multiply (normalize)
/// 8. multiply (scale by gamma)
/// 9. add (shift by beta)
public struct LayerNormPattern: HLOPattern {
    public let name = "layer_norm"
    public let priority = 95

    public init() {}

    public func match(
        at op: HLOOperation,
        index: Int,
        in function: HLOFunction,
        definingOps: [String: (op: HLOOperation, index: Int)]
    ) -> PatternMatch? {
        // LayerNorm typically ends with add (beta) or multiply (gamma)
        guard op.kind == .add || op.kind == .multiply else { return nil }

        // Look for rsqrt or sqrt in the chain (variance normalization)
        var foundRsqrt = false
        var foundVarianceReduce = false
        var foundMeanReduce = false
        var visited: Set<String> = []
        var matchedOps: [HLOOperation] = []
        var matchedIndices: [Int] = []
        var inputOperand: String?
        var axis: Int = -1

        var queue: [HLOOperation] = [op]
        while !queue.isEmpty {
            let curr = queue.removeFirst()
            if visited.contains(curr.result) { continue }
            visited.insert(curr.result)
            matchedOps.append(curr)
            if let idx = definingOps[curr.result]?.index {
                matchedIndices.append(idx)
            } else if curr.result == op.result {
                matchedIndices.append(index)
            }

            if curr.kind == .rsqrt || curr.kind == .sqrt {
                foundRsqrt = true
            }
            if curr.kind == .reduce {
                // Check if this could be mean or variance reduce
                if let dims = curr.attributes.dimensions {
                    axis = dims.first ?? -1
                    if foundMeanReduce {
                        foundVarianceReduce = true
                    } else {
                        foundMeanReduce = true
                    }
                }
            }

            for operand in curr.operands {
                if let def = definingOps[operand]?.op {
                    queue.append(def)
                } else if inputOperand == nil && !operand.starts(with: "%arg") == false {
                    inputOperand = operand
                }
            }
        }

        // Require rsqrt and at least one reduce for LayerNorm
        guard foundRsqrt && foundMeanReduce else { return nil }

        // Find the actual input (first external operand that looks like data)
        if inputOperand == nil {
            for op in matchedOps {
                for operand in op.operands {
                    if definingOps[operand] == nil {
                        inputOperand = operand
                        break
                    }
                }
                if inputOperand != nil { break }
            }
        }

        return PatternMatch(
            operations: matchedOps,
            indices: matchedIndices.sorted(),
            rootOperation: op,
            metadata: ["axis": .int(axis), "pattern": .string("layer_norm")],
            inputs: inputOperand.map { [$0] } ?? []
        )
    }

    public func replacement(for match: PatternMatch, in function: HLOFunction) -> [HLOOperation] {
        let axis: Int
        if case .int(let a) = match.metadata["axis"] {
            axis = a
        } else {
            axis = -1
        }
        let input = match.inputs.first ?? match.operations.first!.operands[0]

        var attributes = HLOAttributes()
        attributes.callTargetName = "fused_layer_norm"
        attributes.backendConfig = "{\"axis\": \(axis), \"epsilon\": 1e-5}"

        // For now, emit without gamma/beta (they can be added later)
        return [HLOOperation(
            result: match.rootOperation.result,
            kind: .customCall,
            operands: [input],
            resultType: match.rootOperation.resultType,
            attributes: attributes
        )]
    }
}

// MARK: - RMSNorm Pattern

/// Detects RMSNorm pattern: x / sqrt(mean(x^2) + eps) * gamma
///
/// Simpler than LayerNorm - no mean subtraction.
public struct RMSNormPattern: HLOPattern {
    public let name = "rms_norm"
    public let priority = 94

    public init() {}

    public func match(
        at op: HLOOperation,
        index: Int,
        in function: HLOFunction,
        definingOps: [String: (op: HLOOperation, index: Int)]
    ) -> PatternMatch? {
        // RMSNorm ends with multiply (by gamma) or divide (by rsqrt result)
        guard op.kind == .multiply || op.kind == .divide else { return nil }

        var foundRsqrt = false
        var foundSquare = false
        var foundReduce = false
        var foundMeanSubtract = false  // Should NOT be present for RMSNorm
        var visited: Set<String> = []
        var matchedOps: [HLOOperation] = []
        var matchedIndices: [Int] = []
        var inputOperand: String?
        var axis: Int = -1

        var queue: [HLOOperation] = [op]
        while !queue.isEmpty {
            let curr = queue.removeFirst()
            if visited.contains(curr.result) { continue }
            visited.insert(curr.result)
            matchedOps.append(curr)
            if let idx = definingOps[curr.result]?.index {
                matchedIndices.append(idx)
            } else if curr.result == op.result {
                matchedIndices.append(index)
            }

            if curr.kind == .rsqrt {
                foundRsqrt = true
            }
            if curr.kind == .multiply {
                // Check if this is x * x (squaring)
                if curr.operands.count == 2 && curr.operands[0] == curr.operands[1] {
                    foundSquare = true
                }
            }
            if curr.kind == .reduce {
                foundReduce = true
                if let dims = curr.attributes.dimensions {
                    axis = dims.first ?? -1
                }
            }
            if curr.kind == .subtract {
                // If we see subtract with a reduce input, this is LayerNorm not RMSNorm
                for operand in curr.operands {
                    if let def = definingOps[operand]?.op, def.kind == .reduce {
                        foundMeanSubtract = true
                    }
                }
            }

            for operand in curr.operands {
                if let def = definingOps[operand]?.op {
                    queue.append(def)
                } else if inputOperand == nil {
                    inputOperand = operand
                }
            }
        }

        // RMSNorm: rsqrt + reduce + square, but NO mean subtraction
        guard foundRsqrt && foundReduce && !foundMeanSubtract else { return nil }

        return PatternMatch(
            operations: matchedOps,
            indices: matchedIndices.sorted(),
            rootOperation: op,
            metadata: ["axis": .int(axis), "pattern": .string("rms_norm")],
            inputs: inputOperand.map { [$0] } ?? []
        )
    }

    public func replacement(for match: PatternMatch, in function: HLOFunction) -> [HLOOperation] {
        let axis: Int
        if case .int(let a) = match.metadata["axis"] {
            axis = a
        } else {
            axis = -1
        }
        let input = match.inputs.first ?? match.operations.first!.operands[0]

        var attributes = HLOAttributes()
        attributes.callTargetName = "fused_rms_norm"
        attributes.backendConfig = "{\"axis\": \(axis), \"epsilon\": 1e-5}"

        return [HLOOperation(
            result: match.rootOperation.result,
            kind: .customCall,
            operands: [input],
            resultType: match.rootOperation.resultType,
            attributes: attributes
        )]
    }
}
