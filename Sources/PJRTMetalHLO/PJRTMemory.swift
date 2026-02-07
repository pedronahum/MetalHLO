// PJRTMemory.swift
// PJRTMetalHLO
//
// PJRT memory space implementation for Metal unified/shared memory.

import CPJRTApi
import Metal

/// Concrete backing storage for opaque PJRT_Memory pointers.
///
/// Metal on Apple Silicon uses unified memory, so we model a single
/// "device" memory space. On systems with discrete GPUs, we could add
/// a "host" memory space as well.
final class PJRTMemoryImpl: @unchecked Sendable {
    let id: Int32
    let kind: String
    let kindId: Int32
    let debugString: String
    let displayString: String

    // Devices that can access this memory
    var devices: [PJRTDeviceImpl] = []

    // Cached C strings
    let cKind: UnsafeMutablePointer<CChar>
    let cDebugString: UnsafeMutablePointer<CChar>
    let cDisplayString: UnsafeMutablePointer<CChar>

    // Stable pointer array for PJRT_Memory_AddressableByDevices
    var devicesPtrBuffer: UnsafeMutableBufferPointer<UnsafeMutablePointer<PJRT_Device>?>?

    init(id: Int32, kind: String, kindId: Int32) {
        self.id = id
        self.kind = kind
        self.kindId = kindId
        self.debugString = "Metal \(kind) memory"
        self.displayString = "Metal \(kind)"
        self.cKind = strdup(kind)!
        self.cDebugString = strdup(debugString)!
        self.cDisplayString = strdup(displayString)!
    }

    deinit {
        free(cKind)
        free(cDebugString)
        free(cDisplayString)
        if let buf = devicesPtrBuffer {
            buf.deallocate()
        }
    }

    func buildDevicesPtr() {
        if let buf = devicesPtrBuffer { buf.deallocate() }
        let buf = UnsafeMutableBufferPointer<UnsafeMutablePointer<PJRT_Device>?>.allocate(capacity: devices.count)
        for (i, device) in devices.enumerated() {
            buf[i] = UnsafeMutablePointer<PJRT_Device>(retainAsOpaque(device))
        }
        devicesPtrBuffer = buf
    }
}

// MARK: - Memory Functions

func pjrt_memory_id(
    _ args: UnsafeMutablePointer<PJRT_Memory_Id_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let memPtr = args.pointee.memory else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let mem = fromOpaque(OpaquePointer(memPtr), as: PJRTMemoryImpl.self)
    args.pointee.id = mem.id
    return nil
}

func pjrt_memory_kind(
    _ args: UnsafeMutablePointer<PJRT_Memory_Kind_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let memPtr = args.pointee.memory else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let mem = fromOpaque(OpaquePointer(memPtr), as: PJRTMemoryImpl.self)
    args.pointee.kind = UnsafePointer(mem.cKind)
    args.pointee.kind_size = strlen(mem.cKind)
    return nil
}

func pjrt_memory_kind_id(
    _ args: UnsafeMutablePointer<PJRT_Memory_Kind_Id_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let memPtr = args.pointee.memory else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let mem = fromOpaque(OpaquePointer(memPtr), as: PJRTMemoryImpl.self)
    args.pointee.kind_id = mem.kindId
    return nil
}

func pjrt_memory_debug_string(
    _ args: UnsafeMutablePointer<PJRT_Memory_DebugString_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let memPtr = args.pointee.memory else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let mem = fromOpaque(OpaquePointer(memPtr), as: PJRTMemoryImpl.self)
    args.pointee.debug_string = UnsafePointer(mem.cDebugString)
    args.pointee.debug_string_size = strlen(mem.cDebugString)
    return nil
}

func pjrt_memory_to_string(
    _ args: UnsafeMutablePointer<PJRT_Memory_ToString_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let memPtr = args.pointee.memory else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let mem = fromOpaque(OpaquePointer(memPtr), as: PJRTMemoryImpl.self)
    args.pointee.to_string = UnsafePointer(mem.cDisplayString)
    args.pointee.to_string_size = strlen(mem.cDisplayString)
    return nil
}

func pjrt_memory_addressable_by_devices(
    _ args: UnsafeMutablePointer<PJRT_Memory_AddressableByDevices_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let memPtr = args.pointee.memory else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let mem = fromOpaque(OpaquePointer(memPtr), as: PJRTMemoryImpl.self)
    if let buf = mem.devicesPtrBuffer {
        args.pointee.devices = UnsafePointer(buf.baseAddress)
        args.pointee.num_devices = buf.count
    } else {
        args.pointee.num_devices = 0
    }
    return nil
}
