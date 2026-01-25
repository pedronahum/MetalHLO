// LexerTests.swift
// MetalHLOCoreTests
//
// Tests for the MLIR Lexer.

import Testing
@testable import MetalHLOCore

@Suite("Lexer Tests")
struct LexerTests {

    // MARK: - Basic Tokens

    @Test("Empty input returns EOF")
    func emptyInput() {
        let lexer = Lexer(source: "")
        let token = lexer.nextToken()
        #expect(token.kind == .eof)
    }

    @Test("Punctuation tokens")
    func punctuation() {
        let lexer = Lexer(source: "(){}<>[],:")
        let expected: [TokenKind] = [
            .leftParen, .rightParen, .leftBrace, .rightBrace,
            .leftAngle, .rightAngle, .leftBracket, .rightBracket,
            .comma, .colon
        ]

        for expectedKind in expected {
            let token = lexer.nextToken()
            #expect(token.kind == expectedKind)
        }
    }

    @Test("Arrow token")
    func arrow() {
        let lexer = Lexer(source: "->")
        let token = lexer.nextToken()
        #expect(token.kind == .arrow)
    }

    // MARK: - Numbers

    @Test("Integer token")
    func integer() {
        let lexer = Lexer(source: "42")
        let token = lexer.nextToken()
        if case .integer(let value) = token.kind {
            #expect(value == 42)
        } else {
            Issue.record("Expected integer token")
        }
    }

    @Test("Negative integer token")
    func negativeInteger() {
        let lexer = Lexer(source: "-123")
        let token = lexer.nextToken()
        if case .integer(let value) = token.kind {
            #expect(value == -123)
        } else {
            Issue.record("Expected integer token")
        }
    }

    @Test("Float token")
    func floatToken() {
        let lexer = Lexer(source: "3.14")
        let token = lexer.nextToken()
        if case .float(let value) = token.kind {
            #expect(Swift.abs(value - 3.14) < 0.001)
        } else {
            Issue.record("Expected float token")
        }
    }

    @Test("Float with exponent")
    func floatWithExponent() {
        let lexer = Lexer(source: "1.5e-10")
        let token = lexer.nextToken()
        if case .float(let value) = token.kind {
            #expect(Swift.abs(value - 1.5e-10) < 1e-15)
        } else {
            Issue.record("Expected float token")
        }
    }

    // MARK: - Identifiers

    @Test("Keyword identifier")
    func identifier() {
        let lexer = Lexer(source: "tensor")
        let token = lexer.nextToken()
        #expect(token.kind == .keyword(.tensor))
    }

    @Test("At-identifier")
    func atIdentifier() {
        let lexer = Lexer(source: "@main")
        let token = lexer.nextToken()
        #expect(token.kind == .atIdentifier)
        #expect(token.text == "@main")
    }

    @Test("Percent-identifier")
    func percentIdentifier() {
        let lexer = Lexer(source: "%arg0")
        let token = lexer.nextToken()
        #expect(token.kind == .percentIdentifier)
        #expect(token.text == "%arg0")
    }

    @Test("Hash-identifier")
    func hashIdentifier() {
        let lexer = Lexer(source: "#stablehlo.dot")
        let token = lexer.nextToken()
        #expect(token.kind == .hashIdentifier)
        #expect(token.text == "#stablehlo.dot")
    }

    // MARK: - Keywords

    @Test("Module keyword")
    func moduleKeyword() {
        let lexer = Lexer(source: "module")
        let token = lexer.nextToken()
        #expect(token.kind == .keyword(.module))
    }

    @Test("Func keyword")
    func funcKeyword() {
        let lexer = Lexer(source: "func")
        let token = lexer.nextToken()
        #expect(token.kind == .keyword(.func))
    }

    @Test("Return keyword")
    func returnKeyword() {
        let lexer = Lexer(source: "return")
        let token = lexer.nextToken()
        #expect(token.kind == .keyword(.return))
    }

    // MARK: - Comments

    @Test("Line comment is skipped")
    func lineComment() {
        let lexer = Lexer(source: "// comment\n42")
        let token = lexer.nextToken()
        // Should skip the comment and newline
        if case .integer(let value) = token.kind {
            #expect(value == 42)
        } else {
            Issue.record("Expected integer after comment")
        }
    }

    // MARK: - Complete Tokenization

    @Test("Simple MLIR tokenization")
    func simpleMLIRTokenization() {
        let mlir = """
        module @test {
          func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            return %arg0 : tensor<2x3xf32>
          }
        }
        """

        let lexer = Lexer(source: mlir)
        let tokens = lexer.tokenize()

        // Should have multiple tokens and end with EOF
        #expect(tokens.count > 10)
        #expect(tokens.last?.kind == .eof)
    }
}
