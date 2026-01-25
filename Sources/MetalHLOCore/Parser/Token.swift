// Token.swift
// MetalHLOCore
//
// Token types for the MLIR lexer.

/// A token produced by the MLIR lexer.
public struct Token: Equatable, Sendable {

    /// The type of token.
    public let kind: TokenKind

    /// The raw text of the token.
    public let text: String

    /// The source location of the token.
    public let location: SourceLocation

    public init(kind: TokenKind, text: String, location: SourceLocation) {
        self.kind = kind
        self.text = text
        self.location = location
    }
}

/// The kind of token.
public enum TokenKind: Equatable, Sendable {
    // MARK: - Literals
    case integer(Int64)
    case float(Double)
    case string(String)

    // MARK: - Identifiers
    case identifier          // e.g., module, func, tensor
    case atIdentifier        // e.g., @main, @module_name
    case percentIdentifier   // e.g., %arg0, %0
    case hashIdentifier      // e.g., #stablehlo.dot

    // MARK: - Keywords
    case keyword(Keyword)

    // MARK: - Punctuation
    case leftParen           // (
    case rightParen          // )
    case leftBrace           // {
    case rightBrace          // }
    case leftBracket         // [
    case rightBracket        // ]
    case leftAngle           // <
    case rightAngle          // >
    case comma               // ,
    case colon               // :
    case equal               // =
    case arrow               // ->
    case minus               // -
    case plus                // +
    case star                // *

    // MARK: - Special
    case newline
    case eof
    case unknown
}

/// Keywords in MLIR.
public enum Keyword: String, CaseIterable, Sendable {
    case module
    case `func`
    case `return`
    case tensor
    case dense
    case true_ = "true"
    case false_ = "false"

    // StableHLO namespace
    case stablehlo

    // Element types
    case f16, f32, f64, bf16
    case i1, i8, i16, i32, i64
    case ui8, ui16, ui32, ui64
}

/// A location in the source code.
public struct SourceLocation: Equatable, Sendable, CustomStringConvertible {
    /// The line number (1-based).
    public let line: Int

    /// The column number (1-based).
    public let column: Int

    /// The byte offset in the source.
    public let offset: Int

    public init(line: Int, column: Int, offset: Int) {
        self.line = line
        self.column = column
        self.offset = offset
    }

    public var description: String {
        "line \(line), column \(column)"
    }
}
