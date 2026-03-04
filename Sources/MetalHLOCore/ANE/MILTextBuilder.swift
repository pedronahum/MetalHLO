// MILTextBuilder.swift
// MetalHLOCore
//
// Builds syntactically correct MIL (Machine Learning Intermediate Language)
// text through structured API calls. This is the critical safety layer:
// _ANEClient.compileModel crashes the process on invalid MIL, so all text
// must be well-formed by construction.
//
// The correct MIL text format is:
//   program(1.0)
//   [buildInfo = {}]
//   {
//     func main(tensor<fp16, [4]> x, tensor<fp16, [4]> y) {
//       tensor<fp16, [4]> out = add(x=x, y=y);
//     } -> (out);
//   }

import Foundation

/// A parameter value in a MIL operation call.
internal enum MILParamValue {
    /// A reference to a MIL variable: name (no % prefix in this format)
    case variable(String)
    /// An integer literal: 42
    case intLiteral(Int)
    /// An integer array literal: [1, 2, 3]
    case intArray([Int])
    /// A floating-point literal: 3.14
    case floatLiteral(Double)
    /// A float array literal: [1.0, 2.0]
    case floatArray([Double])
    /// A boolean literal: True / False
    case boolLiteral(Bool)
    /// A string literal: "fp16"
    case stringLiteral(String)
    /// A tuple of variable references: (a, b, c)
    case variableTuple([String])
}

/// Builds syntactically correct MIL text through structured API calls.
///
/// Every method produces well-formed MIL fragments. The final `build()`
/// method assembles them into a complete MIL program. This prevents
/// malformed MIL that would crash the ANE compiler.
internal final class MILTextBuilder {

    private var functionArgs: [(name: String, typeStr: String)] = []
    private var statements: [String] = []
    private var returnVar: String = ""

    // MARK: - Function Signature

    /// Adds a function argument.
    /// Produces: `tensor<fp16, [shape]> name` in the function signature.
    func addFunctionArg(name: String, shape: [Int]) {
        let typeStr = MILTypeMapper.tensorType(shape: shape)
        functionArgs.append((name: name, typeStr: typeStr))
    }

    // MARK: - Statements

    /// Emits a const statement for BLOBFILE reference.
    /// `tensor<fp16, [shape]> name = const(val=BLOBFILE(offset=X, size=Y));`
    func emitBlobConst(name: String, shape: [Int], offset: Int, size: Int) {
        let typeStr = MILTypeMapper.tensorType(shape: shape)
        let stmt = "      \(typeStr) \(name) = const(val=BLOBFILE(offset=\(offset), size=\(size)));"
        statements.append(stmt)
    }

    /// Emits an inline const statement.
    /// `tensor<fp16, [shape]> name = const(val=VALUE);`
    func emitInlineConst(name: String, shape: [Int], value: String) {
        let typeStr = MILTypeMapper.tensorType(shape: shape)
        let stmt = "      \(typeStr) \(name) = const(val=\(value));"
        statements.append(stmt)
    }

    /// Emits an operation statement with named parameters.
    /// `tensor<fp16, [shape]> name = op(param1=a, param2=b, ...);`
    func emitOp(
        name: String,
        shape: [Int],
        op: String,
        params: [(key: String, value: MILParamValue)]
    ) {
        let typeStr = MILTypeMapper.tensorType(shape: shape)
        let paramsStr = params.map { "\($0.key)=\(formatParamValue($0.value))" }
            .joined(separator: ", ")
        let stmt = "      \(typeStr) \(name) = \(op)(\(paramsStr));"
        statements.append(stmt)
    }

    /// Sets the return variable name.
    func setReturn(name: String) {
        returnVar = name
    }

    // MARK: - Build

    /// Assembles the complete MIL program text.
    func build() -> String {
        var lines: [String] = []

        // Program header
        lines.append("program(1.0)")
        lines.append("[buildInfo = {}]")
        lines.append("{")

        // Function signature
        let argsStr = functionArgs
            .map { "\($0.typeStr) \($0.name)" }
            .joined(separator: ", ")
        lines.append("    func main(\(argsStr)) {")

        // Statements
        for stmt in statements {
            lines.append(stmt)
        }

        // Close function with return
        lines.append("    } -> (\(returnVar));")

        // Close program
        lines.append("}")

        return lines.joined(separator: "\n")
    }

    // MARK: - Private

    private func formatParamValue(_ value: MILParamValue) -> String {
        switch value {
        case .variable(let name):
            return name
        case .intLiteral(let v):
            return "\(v)"
        case .intArray(let arr):
            return "[\(arr.map(String.init).joined(separator: ", "))]"
        case .floatLiteral(let v):
            return formatFloat(v)
        case .floatArray(let arr):
            return "[\(arr.map { formatFloat($0) }.joined(separator: ", "))]"
        case .boolLiteral(let v):
            return v ? "True" : "False"
        case .stringLiteral(let s):
            return "\"\(s)\""
        case .variableTuple(let names):
            return "(\(names.joined(separator: ", ")))"
        }
    }

    private func formatFloat(_ v: Double) -> String {
        if v == 0.0 && !v.sign.rawValue.nonzeroBitCount.isMultiple(of: 2) {
            return "-0.0"
        }
        if v == 0.0 { return "0.0" }
        if v == 1.0 { return "1.0" }
        if v == -1.0 { return "-1.0" }
        // Use enough precision to roundtrip through FP16
        return String(format: "%.6g", v)
    }
}
