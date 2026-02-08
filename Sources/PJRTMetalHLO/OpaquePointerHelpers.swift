// OpaquePointerHelpers.swift
// PJRTMetalHLO
//
// Utilities for bridging between Swift objects and C opaque pointers.

import CPJRTApi

// MARK: - Opaque Pointer Bridging

/// Retains a Swift object and returns it as an OpaquePointer for C interop.
func retainAsOpaque<T: AnyObject>(_ obj: T) -> OpaquePointer {
    OpaquePointer(Unmanaged.passRetained(obj).toOpaque())
}

/// Gets a Swift object from an OpaquePointer without changing the retain count.
func fromOpaque<T: AnyObject>(_ ptr: OpaquePointer, as _: T.Type) -> T {
    Unmanaged<T>.fromOpaque(UnsafeRawPointer(ptr)).takeUnretainedValue()
}

/// Releases a Swift object held behind an OpaquePointer.
func releaseOpaque<T: AnyObject>(_ ptr: OpaquePointer, as _: T.Type) {
    Unmanaged<T>.fromOpaque(UnsafeRawPointer(ptr)).release()
}

// MARK: - PJRT Error Helpers

/// Creates a PJRT_Error pointer from a PJRTErrorImpl.
func makeError(_ code: PJRT_Error_Code, _ message: String) -> UnsafeMutablePointer<PJRT_Error>? {
    let opaque = retainAsOpaque(PJRTErrorImpl(code: code, message: message))
    return UnsafeMutablePointer<PJRT_Error>(opaque)
}

/// Creates an "unimplemented" error for optional PJRT functions.
func unimplementedError(_ functionName: String) -> UnsafeMutablePointer<PJRT_Error>? {
    makeError(PJRT_Error_Code_UNIMPLEMENTED, "\(functionName) is not implemented")
}
