// MetalIR.swift
// MetalHLOCore
//
// Intermediate representation for Metal shader code generation.

import Foundation

/// Metal data types for code generation.
public enum MetalType: String, Sendable, Hashable {
    case float = "float"
    case half = "half"
    case float2 = "float2"
    case float4 = "float4"
    case half2 = "half2"
    case half4 = "half4"
    case int = "int"
    case uint = "uint"
    case uint2 = "uint2"
    case uint3 = "uint3"
    case bool = "bool"
    case void = "void"

    /// Returns the scalar type for vector types.
    public var scalarType: MetalType {
        switch self {
        case .float2, .float4: return .float
        case .half2, .half4: return .half
        case .uint2, .uint3: return .uint
        default: return self
        }
    }

    /// Returns the size in bytes.
    public var byteSize: Int {
        switch self {
        case .half: return 2
        case .float, .int, .uint: return 4
        case .half2: return 4
        case .half4, .float2: return 8
        case .float4: return 16
        case .uint2: return 8
        case .uint3: return 12
        case .bool: return 1
        case .void: return 0
        }
    }

    /// Creates a MetalType from an ElementType.
    public static func from(_ elementType: ElementType) -> MetalType {
        switch elementType {
        case .float16: return .half
        case .float32: return .float
        case .int32: return .int
        case .uint32: return .uint
        default: return .float
        }
    }
}

/// Dimension for thread indexing.
public enum ThreadDimension: String, Sendable {
    case x = "x"
    case y = "y"
    case z = "z"
}

/// Intermediate representation for Metal expressions.
///
/// This IR is designed to be:
/// - Easy to generate from HLO operations
/// - Easy to emit as Metal source code
/// - Optimizable (constant folding, common subexpressions)
public indirect enum MetalIR: Sendable, Hashable {

    // MARK: - Constants and Values

    /// Integer constant.
    case intConstant(Int)

    /// Float constant.
    case floatConstant(Float)

    /// Boolean constant.
    case boolConstant(Bool)

    /// Named variable reference.
    case variable(name: String)

    /// Parameter reference (kernel argument).
    case parameter(name: String, type: MetalType, index: Int)

    // MARK: - Thread Indexing

    /// Thread position in grid.
    case threadPositionInGrid(ThreadDimension)

    /// Thread position in threadgroup.
    case threadPositionInThreadgroup(ThreadDimension)

    /// Threadgroup position in grid.
    case threadgroupPositionInGrid(ThreadDimension)

    /// Threads per threadgroup.
    case threadsPerThreadgroup(ThreadDimension)

    /// SIMD lane ID.
    case simdLaneID

    /// SIMD group ID.
    case simdGroupID

    /// SIMD group index in threadgroup.
    case simdGroupIndexInThreadgroup

    // MARK: - Memory Operations

    /// Load from device memory.
    case load(address: MetalIR, type: MetalType)

    /// Store to device memory.
    case store(value: MetalIR, address: MetalIR)

    /// Load from threadgroup memory.
    case loadShared(offset: MetalIR, type: MetalType)

    /// Store to threadgroup memory.
    case storeShared(value: MetalIR, offset: MetalIR)

    /// Pointer arithmetic (base + offset).
    case pointerAdd(base: MetalIR, offset: MetalIR)

    // MARK: - Arithmetic Operations

    /// Addition.
    case add(MetalIR, MetalIR)

    /// Subtraction.
    case sub(MetalIR, MetalIR)

    /// Multiplication.
    case mul(MetalIR, MetalIR)

    /// Division.
    case div(MetalIR, MetalIR)

    /// Fused multiply-add: a * b + c.
    case fma(MetalIR, MetalIR, MetalIR)

    /// Maximum.
    case max(MetalIR, MetalIR)

    /// Minimum.
    case min(MetalIR, MetalIR)

    /// Negation.
    case neg(MetalIR)

    // MARK: - Math Functions

    /// Exponential.
    case exp(MetalIR)

    /// Natural logarithm.
    case log(MetalIR)

    /// Square root.
    case sqrt(MetalIR)

    /// Reciprocal square root.
    case rsqrt(MetalIR)

    /// Sine.
    case sin(MetalIR)

    /// Cosine.
    case cos(MetalIR)

    /// Hyperbolic tangent.
    case tanh(MetalIR)

    /// Absolute value.
    case abs(MetalIR)

    /// Floor.
    case floor(MetalIR)

    /// Ceiling.
    case ceil(MetalIR)

    // MARK: - Comparison Operations

    /// Equal.
    case eq(MetalIR, MetalIR)

    /// Not equal.
    case ne(MetalIR, MetalIR)

    /// Less than.
    case lt(MetalIR, MetalIR)

    /// Less than or equal.
    case le(MetalIR, MetalIR)

    /// Greater than.
    case gt(MetalIR, MetalIR)

    /// Greater than or equal.
    case ge(MetalIR, MetalIR)

    // MARK: - Logical Operations

    /// Logical and.
    case and(MetalIR, MetalIR)

    /// Logical or.
    case or(MetalIR, MetalIR)

    /// Logical not.
    case not(MetalIR)

    // MARK: - Bitwise Operations

    /// Bitwise and.
    case bitwiseAnd(MetalIR, MetalIR)

    /// Bitwise or.
    case bitwiseOr(MetalIR, MetalIR)

    /// Left shift.
    case shiftLeft(MetalIR, MetalIR)

    /// Right shift.
    case shiftRight(MetalIR, MetalIR)

    // MARK: - Type Operations

    /// Type cast.
    case cast(MetalIR, to: MetalType)

    /// Reinterpret cast.
    case bitcast(MetalIR, to: MetalType)

    // MARK: - Control Flow

    /// Conditional expression: condition ? trueExpr : falseExpr.
    case select(condition: MetalIR, trueExpr: MetalIR, falseExpr: MetalIR)

    // MARK: - SIMD Operations

    /// Broadcast scalar to all SIMD lanes.
    case simdBroadcast(MetalIR, lane: Int)

    /// SIMD shuffle.
    case simdShuffle(value: MetalIR, lane: MetalIR)

    /// SIMD reduction (sum, max, etc.).
    case simdReduce(MetalIR, op: ReductionOp)

    // MARK: - Barrier Operations

    /// Threadgroup memory barrier.
    case threadgroupBarrier

    /// SIMD barrier.
    case simdBarrier

    // MARK: - Compound Expressions

    /// Sequence of expressions (for statements).
    case sequence([MetalIR])

    /// Variable declaration and assignment.
    case declare(name: String, type: MetalType, value: MetalIR?)

    /// Assignment to existing variable.
    case assign(name: String, value: MetalIR)

    /// For loop.
    case forLoop(
        iterVar: String,
        start: MetalIR,
        end: MetalIR,
        step: MetalIR,
        body: MetalIR
    )

    /// If statement.
    case ifStatement(condition: MetalIR, thenBody: MetalIR, elseBody: MetalIR?)

    /// Return statement.
    case returnStatement(MetalIR?)

    /// Early return/break.
    case earlyReturn
}

/// Reduction operation types.
public enum ReductionOp: String, Sendable {
    case sum = "simd_sum"
    case max = "simd_max"
    case min = "simd_min"
    case product = "simd_product"
    case and = "simd_and"
    case or = "simd_or"
}

// MARK: - Code Generation

extension MetalIR {

    /// Generates Metal source code from the IR.
    public func emit(indent: Int = 0) -> String {
        let padding = String(repeating: "    ", count: indent)

        switch self {
        // Constants
        case .intConstant(let v):
            return "\(v)"
        case .floatConstant(let v):
            if v.isInfinite {
                return v > 0 ? "INFINITY" : "-INFINITY"
            }
            return "\(v)f"
        case .boolConstant(let v):
            return v ? "true" : "false"

        // Variables
        case .variable(let name):
            return name
        case .parameter(let name, _, _):
            return name

        // Thread indexing
        case .threadPositionInGrid(let dim):
            return "gid.\(dim.rawValue)"
        case .threadPositionInThreadgroup(let dim):
            return "tid.\(dim.rawValue)"
        case .threadgroupPositionInGrid(let dim):
            return "tgid.\(dim.rawValue)"
        case .threadsPerThreadgroup(let dim):
            return "tg_size.\(dim.rawValue)"
        case .simdLaneID:
            return "simd_lane_id"
        case .simdGroupID:
            return "simdgroup_index_in_threadgroup"
        case .simdGroupIndexInThreadgroup:
            return "simdgroup_index_in_threadgroup"

        // Memory operations
        case .load(let addr, _):
            return "\(addr.emit())"
        case .store(let value, let addr):
            return "\(padding)\(addr.emit()) = \(value.emit());"
        case .loadShared(let offset, _):
            return "shared_mem[\(offset.emit())]"
        case .storeShared(let value, let offset):
            return "\(padding)shared_mem[\(offset.emit())] = \(value.emit());"
        case .pointerAdd(let base, let offset):
            return "(\(base.emit()) + \(offset.emit()))"

        // Arithmetic
        case .add(let a, let b):
            return "(\(a.emit()) + \(b.emit()))"
        case .sub(let a, let b):
            return "(\(a.emit()) - \(b.emit()))"
        case .mul(let a, let b):
            return "(\(a.emit()) * \(b.emit()))"
        case .div(let a, let b):
            return "(\(a.emit()) / \(b.emit()))"
        case .fma(let a, let b, let c):
            return "fma(\(a.emit()), \(b.emit()), \(c.emit()))"
        case .max(let a, let b):
            return "max(\(a.emit()), \(b.emit()))"
        case .min(let a, let b):
            return "min(\(a.emit()), \(b.emit()))"
        case .neg(let a):
            return "(-\(a.emit()))"

        // Math functions
        case .exp(let a):
            return "exp(\(a.emit()))"
        case .log(let a):
            return "log(\(a.emit()))"
        case .sqrt(let a):
            return "sqrt(\(a.emit()))"
        case .rsqrt(let a):
            return "rsqrt(\(a.emit()))"
        case .sin(let a):
            return "sin(\(a.emit()))"
        case .cos(let a):
            return "cos(\(a.emit()))"
        case .tanh(let a):
            return "tanh(\(a.emit()))"
        case .abs(let a):
            return "abs(\(a.emit()))"
        case .floor(let a):
            return "floor(\(a.emit()))"
        case .ceil(let a):
            return "ceil(\(a.emit()))"

        // Comparisons
        case .eq(let a, let b):
            return "(\(a.emit()) == \(b.emit()))"
        case .ne(let a, let b):
            return "(\(a.emit()) != \(b.emit()))"
        case .lt(let a, let b):
            return "(\(a.emit()) < \(b.emit()))"
        case .le(let a, let b):
            return "(\(a.emit()) <= \(b.emit()))"
        case .gt(let a, let b):
            return "(\(a.emit()) > \(b.emit()))"
        case .ge(let a, let b):
            return "(\(a.emit()) >= \(b.emit()))"

        // Logical
        case .and(let a, let b):
            return "(\(a.emit()) && \(b.emit()))"
        case .or(let a, let b):
            return "(\(a.emit()) || \(b.emit()))"
        case .not(let a):
            return "(!\(a.emit()))"

        // Bitwise
        case .bitwiseAnd(let a, let b):
            return "(\(a.emit()) & \(b.emit()))"
        case .bitwiseOr(let a, let b):
            return "(\(a.emit()) | \(b.emit()))"
        case .shiftLeft(let a, let b):
            return "(\(a.emit()) << \(b.emit()))"
        case .shiftRight(let a, let b):
            return "(\(a.emit()) >> \(b.emit()))"

        // Type operations
        case .cast(let a, let type):
            return "\(type.rawValue)(\(a.emit()))"
        case .bitcast(let a, let type):
            return "as_type<\(type.rawValue)>(\(a.emit()))"

        // Control flow
        case .select(let cond, let t, let f):
            return "(\(cond.emit()) ? \(t.emit()) : \(f.emit()))"

        // SIMD operations
        case .simdBroadcast(let v, let lane):
            return "simd_broadcast(\(v.emit()), \(lane))"
        case .simdShuffle(let v, let lane):
            return "simd_shuffle(\(v.emit()), \(lane.emit()))"
        case .simdReduce(let v, let op):
            return "\(op.rawValue)(\(v.emit()))"

        // Barriers
        case .threadgroupBarrier:
            return "\(padding)threadgroup_barrier(mem_flags::mem_threadgroup);"
        case .simdBarrier:
            return "\(padding)simdgroup_barrier(mem_flags::mem_none);"

        // Compound expressions
        case .sequence(let exprs):
            return exprs.map { $0.emit(indent: indent) }.joined(separator: "\n")

        case .declare(let name, let type, let value):
            if let v = value {
                return "\(padding)\(type.rawValue) \(name) = \(v.emit());"
            } else {
                return "\(padding)\(type.rawValue) \(name);"
            }

        case .assign(let name, let value):
            return "\(padding)\(name) = \(value.emit());"

        case .forLoop(let iterVar, let start, let end, let step, let body):
            let header = "for (int \(iterVar) = \(start.emit()); \(iterVar) < \(end.emit()); \(iterVar) += \(step.emit()))"
            return """
            \(padding)\(header) {
            \(body.emit(indent: indent + 1))
            \(padding)}
            """

        case .ifStatement(let cond, let thenBody, let elseBody):
            var result = """
            \(padding)if (\(cond.emit())) {
            \(thenBody.emit(indent: indent + 1))
            \(padding)}
            """
            if let elseBody = elseBody {
                result += """
                 else {
                \(elseBody.emit(indent: indent + 1))
                \(padding)}
                """
            }
            return result

        case .returnStatement(let value):
            if let v = value {
                return "\(padding)return \(v.emit());"
            } else {
                return "\(padding)return;"
            }

        case .earlyReturn:
            return "\(padding)return;"
        }
    }
}

// MARK: - IR Builder Helpers

extension MetalIR {
    /// Creates a loop index variable.
    public static func loopIndex(_ name: String) -> MetalIR {
        .variable(name: name)
    }

    /// Creates an array access expression.
    public static func arrayAccess(base: MetalIR, index: MetalIR) -> MetalIR {
        .load(address: .pointerAdd(base: base, offset: index), type: .float)
    }

    /// Creates a compound assignment (+=, *=, etc.).
    public static func addAssign(name: String, value: MetalIR) -> MetalIR {
        .assign(name: name, value: .add(.variable(name: name), value))
    }

    /// Creates a multiply-accumulate pattern.
    public static func mac(acc: String, a: MetalIR, b: MetalIR) -> MetalIR {
        .assign(name: acc, value: .fma(a, b, .variable(name: acc)))
    }
}
