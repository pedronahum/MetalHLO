// ParseError.swift
// MetalHLOCore
//
// Parser error types.

/// Errors that can occur during parsing.
public enum ParseError: Error, Sendable, CustomStringConvertible {

    /// Unexpected token encountered.
    case unexpectedToken(expected: String, got: Token)

    /// Unexpected end of input.
    case unexpectedEOF(expected: String)

    /// Invalid tensor type syntax.
    case invalidTensorType(String, location: SourceLocation)

    /// Invalid element type.
    case invalidElementType(String, location: SourceLocation)

    /// Invalid operation.
    case invalidOperation(String, location: SourceLocation)

    /// Invalid attribute syntax.
    case invalidAttribute(String, location: SourceLocation)

    /// Invalid constant value.
    case invalidConstant(String, location: SourceLocation)

    /// Missing required attribute.
    case missingAttribute(String, operation: String, location: SourceLocation)

    /// Duplicate definition.
    case duplicateDefinition(String, location: SourceLocation)

    /// Undefined value reference.
    case undefinedValue(String, location: SourceLocation)

    // MARK: - CustomStringConvertible

    public var description: String {
        switch self {
        case .unexpectedToken(let expected, let got):
            return "Unexpected token at \(got.location): expected \(expected), got '\(got.text)'"
        case .unexpectedEOF(let expected):
            return "Unexpected end of input: expected \(expected)"
        case .invalidTensorType(let message, let location):
            return "Invalid tensor type at \(location): \(message)"
        case .invalidElementType(let type, let location):
            return "Invalid element type '\(type)' at \(location)"
        case .invalidOperation(let message, let location):
            return "Invalid operation at \(location): \(message)"
        case .invalidAttribute(let message, let location):
            return "Invalid attribute at \(location): \(message)"
        case .invalidConstant(let message, let location):
            return "Invalid constant at \(location): \(message)"
        case .missingAttribute(let attr, let operation, let location):
            return "Missing required attribute '\(attr)' for operation '\(operation)' at \(location)"
        case .duplicateDefinition(let name, let location):
            return "Duplicate definition of '\(name)' at \(location)"
        case .undefinedValue(let name, let location):
            return "Undefined value '\(name)' at \(location)"
        }
    }

    /// The source location of the error, if available.
    public var location: SourceLocation? {
        switch self {
        case .unexpectedToken(_, let token):
            return token.location
        case .unexpectedEOF:
            return nil
        case .invalidTensorType(_, let location),
             .invalidElementType(_, let location),
             .invalidOperation(_, let location),
             .invalidAttribute(_, let location),
             .invalidConstant(_, let location),
             .missingAttribute(_, _, let location),
             .duplicateDefinition(_, let location),
             .undefinedValue(_, let location):
            return location
        }
    }
}
