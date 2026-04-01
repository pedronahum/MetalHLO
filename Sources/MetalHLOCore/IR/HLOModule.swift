// HLOModule.swift
// MetalHLOCore
//
// Internal representation of a parsed StableHLO module.

/// A parsed StableHLO module.
///
/// `HLOModule` is the top-level container for a StableHLO program,
/// containing one or more functions (a main entry point plus optional private helpers).
public struct HLOModule: Sendable {

    /// The module name (from `module @name`).
    public let name: String

    /// All functions in the module (main + private helpers).
    public let functions: [HLOFunction]

    /// The main (entry point) function — the first public function, or the first function.
    public var function: HLOFunction {
        functions.first(where: { !$0.isPrivate }) ?? functions[0]
    }

    /// Look up a function by name.
    public func getFunction(named name: String) -> HLOFunction? {
        functions.first(where: { $0.name == name })
    }

    /// Creates a new HLO module.
    ///
    /// - Parameters:
    ///   - name: The module name.
    ///   - functions: All functions in the module.
    public init(name: String, functions: [HLOFunction]) {
        self.name = name
        self.functions = functions
    }

    /// Backwards-compatible init for single-function modules.
    public init(name: String, function: HLOFunction) {
        self.name = name
        self.functions = [function]
    }
}

extension HLOModule: CustomStringConvertible {
    public var description: String {
        let funcs = functions.map { String(describing: $0) }.joined(separator: "\n")
        return """
        module @\(name) {
        \(funcs)
        }
        """
    }
}
