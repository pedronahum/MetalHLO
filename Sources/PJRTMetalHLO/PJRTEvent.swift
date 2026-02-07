// PJRTEvent.swift
// PJRTMetalHLO
//
// PJRT event implementation for async completion tracking.
// Wraps Metal command buffer completion or immediate completion.

import CPJRTApi
import Foundation

/// Concrete backing storage for opaque PJRT_Event pointers.
final class PJRTEventImpl: @unchecked Sendable {

    enum State {
        case pending
        case completed
        case failed(PJRTErrorImpl)
    }

    private let lock = NSLock()
    private var state: State
    private var callbacks: [(OpaquePointer?) -> Void] = []

    /// Create an already-completed event.
    init() {
        self.state = .completed
    }

    /// Create a pending event that will be completed later.
    init(pending: Bool) {
        self.state = pending ? .pending : .completed
    }

    var isReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        switch state {
        case .pending: return false
        case .completed, .failed: return true
        }
    }

    var error: PJRTErrorImpl? {
        lock.lock()
        defer { lock.unlock() }
        if case .failed(let err) = state { return err }
        return nil
    }

    /// Mark the event as completed (success).
    func complete() {
        lock.lock()
        state = .completed
        let cbs = callbacks
        callbacks.removeAll()
        lock.unlock()
        for cb in cbs { cb(nil) }
    }

    /// Mark the event as failed.
    func fail(_ error: PJRTErrorImpl) {
        lock.lock()
        state = .failed(error)
        let cbs = callbacks
        callbacks.removeAll()
        lock.unlock()
        for cb in cbs {
            // Each callback gets its own retained reference
            cb(retainAsOpaque(error))
        }
    }

    /// Register a callback for when the event completes.
    func onReady(_ callback: @escaping (OpaquePointer?) -> Void) {
        lock.lock()
        switch state {
        case .pending:
            callbacks.append(callback)
            lock.unlock()
        case .completed:
            lock.unlock()
            callback(nil)
        case .failed(let err):
            lock.unlock()
            callback(retainAsOpaque(err))
        }
    }

    /// Block until the event completes.
    func await() {
        // Spin-wait with backoff for pending events
        while !isReady {
            Thread.sleep(forTimeInterval: 0.0001) // 100us
        }
    }
}

// MARK: - PJRT Event Functions

func pjrt_event_destroy(
    _ args: UnsafeMutablePointer<PJRT_Event_Destroy_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let eventPtr = args.pointee.event else { return nil }
    releaseOpaque(OpaquePointer(eventPtr), as: PJRTEventImpl.self)
    return nil
}

func pjrt_event_is_ready(
    _ args: UnsafeMutablePointer<PJRT_Event_IsReady_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let eventPtr = args.pointee.event else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args or event")
    }
    let event = fromOpaque(OpaquePointer(eventPtr), as: PJRTEventImpl.self)
    args.pointee.is_ready = event.isReady
    return nil
}

func pjrt_event_error(
    _ args: UnsafeMutablePointer<PJRT_Event_Error_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let eventPtr = args.pointee.event else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args or event")
    }
    let event = fromOpaque(OpaquePointer(eventPtr), as: PJRTEventImpl.self)
    if let err = event.error {
        return UnsafeMutablePointer<PJRT_Error>(retainAsOpaque(err))
    }
    return nil
}

func pjrt_event_await(
    _ args: UnsafeMutablePointer<PJRT_Event_Await_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let eventPtr = args.pointee.event else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args or event")
    }
    let event = fromOpaque(OpaquePointer(eventPtr), as: PJRTEventImpl.self)
    event.await()
    if let err = event.error {
        return UnsafeMutablePointer<PJRT_Error>(retainAsOpaque(err))
    }
    return nil
}

func pjrt_event_on_ready(
    _ args: UnsafeMutablePointer<PJRT_Event_OnReady_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let eventPtr = args.pointee.event else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args or event")
    }
    let event = fromOpaque(OpaquePointer(eventPtr), as: PJRTEventImpl.self)
    let callback = args.pointee.callback
    let userArg = args.pointee.user_arg

    event.onReady { errorPtr in
        // Convert OpaquePointer? to UnsafeMutablePointer<PJRT_Error>?
        let cError: UnsafeMutablePointer<PJRT_Error>?
        if let errorPtr = errorPtr {
            cError = UnsafeMutablePointer<PJRT_Error>(errorPtr)
        } else {
            cError = nil
        }
        callback?(cError, userArg)
    }
    return nil
}
