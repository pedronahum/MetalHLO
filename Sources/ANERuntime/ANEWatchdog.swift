// ANEWatchdog.swift
// ANERuntime
//
// Watchdog timer for ANE dispatches. Wraps operations with a timeout
// to prevent hung CoreML predictions or compilations from blocking
// the execution pipeline indefinitely.

import Foundation

/// Thread-safe result box for passing values across dispatch queues.
private final class WatchdogResultBox: @unchecked Sendable {
    var value: Any?
    var error: Error?
}

/// Provides timeout-guarded execution for ANE operations.
///
/// CoreML compilations and predictions can occasionally hang due to
/// driver issues or resource exhaustion. The watchdog wraps these
/// operations with a deadline and throws on timeout, allowing the
/// heterogeneous executor to fall back to GPU.
public struct ANEWatchdog {

    /// Default timeout for ANE operations (30 seconds).
    public static let defaultTimeoutMs: UInt64 = 30_000

    /// Executes an operation with a timeout guard.
    ///
    /// The operation runs on a background queue. If it does not complete
    /// within `timeoutMs` milliseconds, the call throws
    /// `ANEError.executionFailed` with a timeout message.
    ///
    /// - Parameters:
    ///   - timeoutMs: Maximum wait time in milliseconds.
    ///   - operation: The operation to execute.
    /// - Returns: The operation's result.
    /// - Throws: `ANEError.executionFailed` on timeout, or the operation's error.
    public static func withTimeout<T>(
        _ timeoutMs: UInt64 = defaultTimeoutMs,
        operation: @escaping () throws -> T
    ) throws -> T {
        let box = WatchdogResultBox()
        let semaphore = DispatchSemaphore(value: 0)
        let queue = DispatchQueue(label: "com.metalHLO.ane.watchdog", qos: .userInitiated)

        queue.async {
            do {
                let result = try operation()
                box.value = result
            } catch {
                box.error = error
            }
            semaphore.signal()
        }

        let waitResult = semaphore.wait(timeout: .now() + .milliseconds(Int(timeoutMs)))
        guard waitResult == .success else {
            throw ANEError.executionFailed("ANE dispatch timed out after \(timeoutMs)ms")
        }

        if let error = box.error {
            throw error
        }

        guard let result = box.value as? T else {
            throw ANEError.internalError("Watchdog: operation completed but no result captured")
        }

        return result
    }
}
