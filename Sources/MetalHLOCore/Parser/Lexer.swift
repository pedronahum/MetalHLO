// Lexer.swift
// MetalHLOCore
//
// MLIR tokenizer for StableHLO.

/// Tokenizes MLIR text into a stream of tokens.
public final class Lexer {

    // MARK: - Properties

    private let source: String
    private var currentIndex: String.Index
    private var line: Int = 1
    private var column: Int = 1
    private var offset: Int = 0

    // MARK: - Initialization

    /// Creates a new lexer for the given source text.
    ///
    /// - Parameter source: The MLIR source text.
    public init(source: String) {
        self.source = source
        self.currentIndex = source.startIndex
    }

    // MARK: - Public API

    /// Returns the next token from the source.
    ///
    /// - Returns: The next token, or `.eof` if at end of input.
    public func nextToken() -> Token {
        skipWhitespaceAndComments()

        guard !isAtEnd else {
            return makeToken(.eof, text: "")
        }

        let startLocation = currentLocation

        let char = peek()

        // Numbers (including negative)
        if char.isNumber || (char == "-" && peekNext()?.isNumber == true) {
            return scanNumber()
        }

        // Identifiers and keywords
        if char.isLetter || char == "_" {
            return scanIdentifier()
        }

        // @ identifiers
        if char == "@" {
            return scanAtIdentifier()
        }

        // % identifiers
        if char == "%" {
            return scanPercentIdentifier()
        }

        // # identifiers (attributes)
        if char == "#" {
            return scanHashIdentifier()
        }

        // String literals
        if char == "\"" {
            return scanString()
        }

        // Punctuation
        advance()
        switch char {
        case "(": return makeToken(.leftParen, text: "(", location: startLocation)
        case ")": return makeToken(.rightParen, text: ")", location: startLocation)
        case "{": return makeToken(.leftBrace, text: "{", location: startLocation)
        case "}": return makeToken(.rightBrace, text: "}", location: startLocation)
        case "[": return makeToken(.leftBracket, text: "[", location: startLocation)
        case "]": return makeToken(.rightBracket, text: "]", location: startLocation)
        case "<": return makeToken(.leftAngle, text: "<", location: startLocation)
        case ">": return makeToken(.rightAngle, text: ">", location: startLocation)
        case ",": return makeToken(.comma, text: ",", location: startLocation)
        case ":": return makeToken(.colon, text: ":", location: startLocation)
        case "=": return makeToken(.equal, text: "=", location: startLocation)
        case "+": return makeToken(.plus, text: "+", location: startLocation)
        case "*": return makeToken(.star, text: "*", location: startLocation)
        case "-":
            if peek() == ">" {
                advance()
                return makeToken(.arrow, text: "->", location: startLocation)
            }
            return makeToken(.minus, text: "-", location: startLocation)
        case "\n":
            return makeToken(.newline, text: "\n", location: startLocation)
        default:
            return makeToken(.unknown, text: String(char), location: startLocation)
        }
    }

    /// Tokenizes the entire source and returns all tokens.
    ///
    /// - Returns: An array of all tokens.
    public func tokenize() -> [Token] {
        var tokens: [Token] = []
        while true {
            let token = nextToken()
            tokens.append(token)
            if token.kind == .eof {
                break
            }
        }
        return tokens
    }

    // MARK: - Private Helpers

    private var isAtEnd: Bool {
        currentIndex >= source.endIndex
    }

    private var currentLocation: SourceLocation {
        SourceLocation(line: line, column: column, offset: offset)
    }

    private func peek() -> Character {
        guard !isAtEnd else { return "\0" }
        return source[currentIndex]
    }

    private func peekNext() -> Character? {
        let nextIndex = source.index(after: currentIndex)
        guard nextIndex < source.endIndex else { return nil }
        return source[nextIndex]
    }

    @discardableResult
    private func advance() -> Character {
        guard !isAtEnd else { return "\0" }
        let char = source[currentIndex]
        currentIndex = source.index(after: currentIndex)
        offset += 1
        if char == "\n" {
            line += 1
            column = 1
        } else {
            column += 1
        }
        return char
    }

    private func skipWhitespaceAndComments() {
        while !isAtEnd {
            let char = peek()
            switch char {
            case " ", "\t", "\r":
                advance()
            case "/":
                if peekNext() == "/" {
                    // Line comment - skip to end of line
                    while !isAtEnd && peek() != "\n" {
                        advance()
                    }
                    // Also skip the newline after the comment
                    if !isAtEnd && peek() == "\n" {
                        advance()
                    }
                } else {
                    return
                }
            default:
                return
            }
        }
    }

    private func makeToken(_ kind: TokenKind, text: String, location: SourceLocation? = nil) -> Token {
        Token(kind: kind, text: text, location: location ?? currentLocation)
    }

    private func scanNumber() -> Token {
        let startLocation = currentLocation
        var text = ""
        var isFloat = false

        // Handle negative sign
        if peek() == "-" {
            text.append(advance())
        }

        // Check for hexadecimal format: 0x or 0X
        if peek() == "0" && (peekNext() == "x" || peekNext() == "X") {
            text.append(advance())  // consume '0'
            text.append(advance())  // consume 'x' or 'X'

            // Parse hex digits
            while !isAtEnd && isHexDigit(peek()) {
                text.append(advance())
            }

            // Check for hex float: 0x1.FFFFFEp127
            if peek() == "." {
                isFloat = true
                text.append(advance())  // consume '.'
                while !isAtEnd && isHexDigit(peek()) {
                    text.append(advance())
                }
            }

            // Hex exponent: p or P followed by decimal exponent
            if peek() == "p" || peek() == "P" {
                isFloat = true
                text.append(advance())
                if peek() == "+" || peek() == "-" {
                    text.append(advance())
                }
                while !isAtEnd && peek().isNumber {
                    text.append(advance())
                }
            }

            if isFloat {
                // Parse hex float using Swift's built-in support
                let value = Double(text) ?? 0.0
                return makeToken(.float(value), text: text, location: startLocation)
            } else {
                // Parse hex integer
                let hexValue = text.dropFirst(text.hasPrefix("-") ? 3 : 2)  // Remove "-0x" or "0x"
                let value = Int64(hexValue, radix: 16) ?? 0
                let signedValue = text.hasPrefix("-") ? -value : value
                return makeToken(.integer(signedValue), text: text, location: startLocation)
            }
        }

        // Integer part (decimal)
        while !isAtEnd && peek().isNumber {
            text.append(advance())
        }

        // Decimal part
        if peek() == "." && peekNext()?.isNumber == true {
            isFloat = true
            text.append(advance())  // consume '.'
            while !isAtEnd && peek().isNumber {
                text.append(advance())
            }
        }

        // Exponent part
        if peek() == "e" || peek() == "E" {
            isFloat = true
            text.append(advance())
            if peek() == "+" || peek() == "-" {
                text.append(advance())
            }
            while !isAtEnd && peek().isNumber {
                text.append(advance())
            }
        }

        if isFloat {
            let value = Double(text) ?? 0.0
            return makeToken(.float(value), text: text, location: startLocation)
        } else {
            let value = Int64(text) ?? 0
            return makeToken(.integer(value), text: text, location: startLocation)
        }
    }

    private func isHexDigit(_ char: Character) -> Bool {
        return char.isNumber || (char >= "a" && char <= "f") || (char >= "A" && char <= "F")
    }

    private func scanIdentifier() -> Token {
        let startLocation = currentLocation
        var text = ""

        while !isAtEnd && (peek().isLetter || peek().isNumber || peek() == "_" || peek() == ".") {
            // Special case: if we have just "x" and next char is a digit or element type prefix,
            // stop here to handle tensor dimension separators like "2x3xf32"
            // Element types start with: f (f16, f32, f64), i (i1, i8, i16, i32, i64),
            // u (ui8, ui16, ui32, ui64), b (bf16)
            if text == "x" {
                let next = peek()
                if next.isNumber || next == "f" || next == "i" || next == "u" || next == "b" {
                    break
                }
            }
            text.append(advance())
        }

        // Check for keywords
        if let keyword = Keyword(rawValue: text) {
            return makeToken(.keyword(keyword), text: text, location: startLocation)
        }

        return makeToken(.identifier, text: text, location: startLocation)
    }

    private func scanAtIdentifier() -> Token {
        let startLocation = currentLocation
        var text = ""
        text.append(advance())  // consume '@'

        while !isAtEnd && (peek().isLetter || peek().isNumber || peek() == "_") {
            text.append(advance())
        }

        return makeToken(.atIdentifier, text: text, location: startLocation)
    }

    private func scanPercentIdentifier() -> Token {
        let startLocation = currentLocation
        var text = ""
        text.append(advance())  // consume '%'

        while !isAtEnd && (peek().isLetter || peek().isNumber || peek() == "_") {
            text.append(advance())
        }

        return makeToken(.percentIdentifier, text: text, location: startLocation)
    }

    private func scanHashIdentifier() -> Token {
        let startLocation = currentLocation
        var text = ""
        text.append(advance())  // consume '#'

        // Read the full attribute including nested angle brackets
        while !isAtEnd && (peek().isLetter || peek().isNumber || peek() == "_" || peek() == ".") {
            text.append(advance())
        }

        return makeToken(.hashIdentifier, text: text, location: startLocation)
    }

    private func scanString() -> Token {
        let startLocation = currentLocation
        var text = ""
        advance()  // consume opening quote

        while !isAtEnd && peek() != "\"" {
            if peek() == "\\" {
                advance()
                if !isAtEnd {
                    text.append(advance())
                }
            } else {
                text.append(advance())
            }
        }

        if !isAtEnd {
            advance()  // consume closing quote
        }

        return makeToken(.string(text), text: "\"\(text)\"", location: startLocation)
    }
}
