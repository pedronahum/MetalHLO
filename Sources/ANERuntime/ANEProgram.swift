// ANEProgram.swift
// ANERuntime
//
// Compiled ANE program handle.

import Foundation

/// A compiled ANE program ready for execution.
///
/// This wraps the opaque handle returned by `_ANEClient`'s compilation
/// method. Programs are tied to the `ANEDevice` that created them.
public final class ANEProgram: @unchecked Sendable {

    /// A unique identifier for this program.
    public let id: String

    /// The number of input buffers expected.
    public let inputCount: Int

    /// The number of output buffers produced.
    public let outputCount: Int

    /// Input buffer descriptors (shapes/types).
    public let inputDescriptors: [ANEBufferDescriptor]

    /// Output buffer descriptors (shapes/types).
    public let outputDescriptors: [ANEBufferDescriptor]

    /// The opaque program handle from _ANEClient.
    /// Retained via Unmanaged if non-nil.
    internal let programHandle: AnyObject?

    /// The compilation count at which this program was created.
    internal let compilationNumber: Int

    private let lock = NSLock()
    private var released = false

    internal init(
        id: String,
        programHandle: AnyObject?,
        inputDescriptors: [ANEBufferDescriptor],
        outputDescriptors: [ANEBufferDescriptor],
        compilationNumber: Int
    ) {
        self.id = id
        self.programHandle = programHandle
        self.inputCount = inputDescriptors.count
        self.outputCount = outputDescriptors.count
        self.inputDescriptors = inputDescriptors
        self.outputDescriptors = outputDescriptors
        self.compilationNumber = compilationNumber
    }

    /// Whether this program has been released.
    public var isValid: Bool {
        lock.lock()
        defer { lock.unlock() }
        return !released
    }

    /// Releases the program handle. Called automatically in deinit.
    public func release() {
        lock.lock()
        defer { lock.unlock() }
        released = true
    }

    deinit {
        released = true
    }
}
