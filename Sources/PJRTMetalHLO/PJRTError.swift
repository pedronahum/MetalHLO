// PJRTError.swift
// PJRTMetalHLO
//
// PJRT error type implementation. PJRT_Error is an opaque type — we back it
// with a Swift class stored behind an OpaquePointer.

import CPJRTApi
import Darwin

/// Concrete backing storage for opaque PJRT_Error pointers.
final class PJRTErrorImpl: @unchecked Sendable {
    let code: PJRT_Error_Code
    let message: String

    /// Cached C string with the same lifetime as this object.
    let cMessage: UnsafeMutablePointer<CChar>

    init(code: PJRT_Error_Code, message: String) {
        self.code = code
        self.message = message
        self.cMessage = strdup(message)!
    }

    deinit {
        free(cMessage)
    }
}

// MARK: - PJRT_Error_Destroy

func pjrt_error_destroy(
    _ args: UnsafeMutablePointer<PJRT_Error_Destroy_Args>?
) {
    guard let args = args else { return }
    guard let errorPtr = args.pointee.error else { return }
    let opaque = OpaquePointer(errorPtr)
    releaseOpaque(opaque, as: PJRTErrorImpl.self)
}

// MARK: - PJRT_Error_Message

func pjrt_error_message(
    _ args: UnsafeMutablePointer<PJRT_Error_Message_Args>?
) {
    guard let args = args else { return }
    guard let errorPtr = args.pointee.error else { return }
    let opaque = OpaquePointer(errorPtr)
    let impl = fromOpaque(opaque, as: PJRTErrorImpl.self)
    args.pointee.message = UnsafePointer(impl.cMessage)
    args.pointee.message_size = strlen(impl.cMessage)
}

// MARK: - PJRT_Error_GetCode

func pjrt_error_getcode(
    _ args: UnsafeMutablePointer<PJRT_Error_GetCode_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    guard let errorPtr = args.pointee.error else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL error")
    }
    let opaque = OpaquePointer(errorPtr)
    let impl = fromOpaque(opaque, as: PJRTErrorImpl.self)
    args.pointee.code = impl.code
    return nil
}
