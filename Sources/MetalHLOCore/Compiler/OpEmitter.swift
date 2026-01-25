// OpEmitter.swift
// MetalHLOCore
//
// Protocol for operation emission.

import MetalPerformanceShadersGraph

/// Protocol for emitting MPSGraph operations from HLO operations.
///
/// `OpEmitter` defines the interface for translating specific HLO
/// operations to their MPSGraph equivalents. This allows for modular
/// operation implementations.
public protocol OpEmitter {

    /// The operation kinds this emitter handles.
    static var supportedOps: [HLOOpKind] { get }

    /// Emits an MPSGraph tensor for the given HLO operation.
    ///
    /// - Parameters:
    ///   - operation: The HLO operation to emit.
    ///   - graph: The MPSGraph to emit into.
    ///   - valueMap: Map of value names to their tensors.
    /// - Throws: `CompilationError` if emission fails.
    /// - Returns: The resulting MPSGraph tensor.
    func emit(
        operation: HLOOperation,
        graph: MPSGraph,
        valueMap: [String: MPSGraphTensor]
    ) throws -> MPSGraphTensor
}

/// Registry for operation emitters.
///
/// `OpEmitterRegistry` manages the mapping from operation kinds
/// to their emitters, allowing for extensible operation support.
public final class OpEmitterRegistry: @unchecked Sendable {

    /// Shared registry instance.
    public static let shared = OpEmitterRegistry()

    private var emitters: [HLOOpKind: any OpEmitter] = [:]
    private let lock = NSLock()

    private init() {
        // Register default emitters
        registerDefaultEmitters()
    }

    /// Registers an emitter for its supported operations.
    ///
    /// - Parameter emitter: The emitter to register.
    public func register(_ emitter: any OpEmitter) {
        for kind in type(of: emitter).supportedOps {
            emitters[kind] = emitter
        }
    }

    /// Gets the emitter for an operation kind.
    ///
    /// - Parameter kind: The operation kind.
    /// - Returns: The emitter, or nil if none registered.
    public func emitter(for kind: HLOOpKind) -> (any OpEmitter)? {
        emitters[kind]
    }

    private func registerDefaultEmitters() {
        // Emitters will be registered as they are implemented
        // For now, the MPSGraphCompiler handles all operations directly
    }
}
