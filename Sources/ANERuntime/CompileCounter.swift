// CompileCounter.swift
// ANERuntime
//
// Thread-safe counter tracking ANE compilations to guard against
// the known ~119-compile resource leak.

import Foundation

/// Tracks ANE compilation count to prevent the known ~119-compile leak.
///
/// The Apple Neural Engine leaks resources after approximately 119
/// compilations in a single process lifetime. This counter tracks
/// compilations and refuses further compilation at a conservative limit.
public final class CompileCounter: @unchecked Sendable {

    /// Conservative warning threshold (below the ~119 hard limit).
    public static let defaultWarningLimit: Int = 100

    /// Hard limit at which compilation is refused.
    public static let defaultHardLimit: Int = 115

    private let lock = NSLock()
    private var count: Int = 0
    private let warningLimit: Int
    private let maximumLimit: Int

    public init(
        warningLimit: Int = CompileCounter.defaultWarningLimit,
        maximumLimit: Int = CompileCounter.defaultHardLimit
    ) {
        self.warningLimit = warningLimit
        self.maximumLimit = maximumLimit
    }

    /// The current compilation count.
    public var currentCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    /// Whether further compilation is safe (below hard limit).
    public var canCompile: Bool {
        lock.lock()
        defer { lock.unlock() }
        return count < maximumLimit
    }

    /// Whether the warning threshold has been passed.
    public var isWarning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return count >= warningLimit
    }

    /// Increments the counter. Returns the new count.
    /// Throws `ANEError.compilationLimitReached` if the hard limit is exceeded.
    @discardableResult
    public func increment() throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        count += 1
        if count > maximumLimit {
            throw ANEError.compilationLimitReached(current: count, limit: maximumLimit)
        }
        return count
    }

    /// Resets the counter (only meaningful after process restart in practice).
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        count = 0
    }
}
