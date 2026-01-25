// HLOModule.swift
// MetalHLOCore
//
// Internal representation of a parsed StableHLO module.

/// A parsed StableHLO module.
///
/// `HLOModule` is the top-level container for a StableHLO program,
/// containing a single function (the entry point).
public struct HLOModule: Sendable {

    /// The module name (from `module @name`).
    public let name: String

    /// The main function.
    public let function: HLOFunction

    /// Creates a new HLO module.
    ///
    /// - Parameters:
    ///   - name: The module name.
    ///   - function: The main function.
    public init(name: String, function: HLOFunction) {
        self.name = name
        self.function = function
    }
}

extension HLOModule: CustomStringConvertible {
    public var description: String {
        """
        module @\(name) {
        \(function)
        }
        """
    }
}
