// PJRTBuffer.swift
// PJRTMetalHLO
//
// PJRT buffer implementation wrapping MetalHLO buffers with async support.

import CPJRTApi
import Metal
import MetalHLO
import MetalHLOCore
import Foundation

/// Concrete backing storage for opaque PJRT_Buffer pointers.
final class PJRTBufferImpl: @unchecked Sendable {
    let buffer: Buffer
    let device: PJRTDeviceImpl
    let memory: PJRTMemoryImpl
    let readyEvent: PJRTEventImpl
    let elementType: PJRT_Buffer_Type

    /// Stable C array for dimensions — pointer survives scope.
    let dimsBuffer: UnsafeMutableBufferPointer<Int64>
    let numDims: Int

    private let lock = NSLock()
    private var deleted = false

    init(buffer: Buffer, device: PJRTDeviceImpl, memory: PJRTMemoryImpl, event: PJRTEventImpl? = nil) {
        self.buffer = buffer
        self.device = device
        self.memory = memory
        self.readyEvent = event ?? PJRTEventImpl() // immediate completion
        self.elementType = Self.mapElementType(buffer.elementType)

        let shape = buffer.shape
        self.numDims = shape.count
        self.dimsBuffer = .allocate(capacity: max(shape.count, 1))
        for (i, dim) in shape.enumerated() {
            dimsBuffer[i] = Int64(dim)
        }
    }

    deinit {
        dimsBuffer.deallocate()
    }

    var isDeleted: Bool {
        lock.lock()
        defer { lock.unlock() }
        return deleted
    }

    func markDeleted() {
        lock.lock()
        deleted = true
        lock.unlock()
    }

    static func mapElementType(_ et: MetalHLO.ElementType) -> PJRT_Buffer_Type {
        switch et {
        case .float16:  return PJRT_Buffer_Type_F16
        case .float32:  return PJRT_Buffer_Type_F32
        case .float64:  return PJRT_Buffer_Type_F64
        case .bfloat16: return PJRT_Buffer_Type_BF16
        case .int1:     return PJRT_Buffer_Type_PRED
        case .int8:     return PJRT_Buffer_Type_S8
        case .int16:    return PJRT_Buffer_Type_S16
        case .int32:    return PJRT_Buffer_Type_S32
        case .int64:    return PJRT_Buffer_Type_S64
        case .uint8:    return PJRT_Buffer_Type_U8
        case .uint16:   return PJRT_Buffer_Type_U16
        case .uint32:   return PJRT_Buffer_Type_U32
        case .uint64:   return PJRT_Buffer_Type_U64
        }
    }

    static func mapFromPJRTType(_ t: PJRT_Buffer_Type) -> MetalHLO.ElementType? {
        switch t {
        case PJRT_Buffer_Type_F16:  return .float16
        case PJRT_Buffer_Type_F32:  return .float32
        case PJRT_Buffer_Type_F64:  return .float64
        case PJRT_Buffer_Type_BF16: return .bfloat16
        case PJRT_Buffer_Type_PRED: return .int1
        case PJRT_Buffer_Type_S8:   return .int8
        case PJRT_Buffer_Type_S16:  return .int16
        case PJRT_Buffer_Type_S32:  return .int32
        case PJRT_Buffer_Type_S64:  return .int64
        case PJRT_Buffer_Type_U8:   return .uint8
        case PJRT_Buffer_Type_U16:  return .uint16
        case PJRT_Buffer_Type_U32:  return .uint32
        case PJRT_Buffer_Type_U64:  return .uint64
        default: return nil
        }
    }
}

// MARK: - BufferFromHostBuffer

func pjrt_client_buffer_from_host_buffer(
    _ args: UnsafeMutablePointer<PJRT_Client_BufferFromHostBuffer_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args or client")
    }
    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)

    let pjrtType = args.pointee.type
    guard let elementType = PJRTBufferImpl.mapFromPJRTType(pjrtType) else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "Unsupported element type: \(pjrtType)")
    }

    let numDims = Int(args.pointee.num_dims)
    var shape = [Int]()
    if let dims = args.pointee.dims {
        for i in 0..<numDims {
            shape.append(Int(dims[i]))
        }
    }

    // Calculate expected byte count
    let elementCount = shape.reduce(1, *)
    let byteSize = elementType.byteSize
    let byteCount = elementCount * (byteSize > 0 ? byteSize : 1)

    guard let data = args.pointee.data else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL data pointer")
    }

    do {
        let rawData = Data(bytes: data, count: byteCount)
        let buffer = try impl.client.createBuffer(bytes: rawData, shape: shape, elementType: elementType)

        // Create done-with-host-buffer event (immediate — we copied the data)
        let doneEvent = PJRTEventImpl()
        args.pointee.done_with_host_buffer = UnsafeMutablePointer<PJRT_Event>(
            retainAsOpaque(doneEvent)
        )

        let device = impl.devices[0]
        let memory = impl.memories[0]
        let bufferImpl = PJRTBufferImpl(buffer: buffer, device: device, memory: memory)
        args.pointee.buffer = UnsafeMutablePointer<PJRT_Buffer>(
            retainAsOpaque(bufferImpl)
        )
        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "Buffer creation failed: \(error)")
    }
}

// MARK: - Buffer Functions

func pjrt_buffer_destroy(
    _ args: UnsafeMutablePointer<PJRT_Buffer_Destroy_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else { return nil }
    releaseOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    return nil
}

func pjrt_buffer_element_type(
    _ args: UnsafeMutablePointer<PJRT_Buffer_ElementType_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    args.pointee.type = buf.elementType
    return nil
}

func pjrt_buffer_dimensions(
    _ args: UnsafeMutablePointer<PJRT_Buffer_Dimensions_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    args.pointee.dims = UnsafePointer(buf.dimsBuffer.baseAddress!)
    args.pointee.num_dims = buf.numDims
    return nil
}

func pjrt_buffer_unpadded_dimensions(
    _ args: UnsafeMutablePointer<PJRT_Buffer_UnpaddedDimensions_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    args.pointee.unpadded_dims = UnsafePointer(buf.dimsBuffer.baseAddress!)
    args.pointee.num_dims = buf.numDims
    return nil
}

func pjrt_buffer_dynamic_dimension_indices(
    _ args: UnsafeMutablePointer<PJRT_Buffer_DynamicDimensionIndices_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    // No dynamic dimensions supported
    args.pointee.num_dynamic_dims = 0
    args.pointee.dynamic_dim_indices = nil
    return nil
}

func pjrt_buffer_get_memory_layout(
    _ args: UnsafeMutablePointer<PJRT_Buffer_GetMemoryLayout_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    // Deprecated in PJRT v0.54 — return unimplemented
    return unimplementedError("PJRT_Buffer_GetMemoryLayout")
}

func pjrt_buffer_on_device_size_in_bytes(
    _ args: UnsafeMutablePointer<PJRT_Buffer_OnDeviceSizeInBytes_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    args.pointee.on_device_size_in_bytes = buf.buffer.byteCount
    return nil
}

func pjrt_buffer_device(
    _ args: UnsafeMutablePointer<PJRT_Buffer_Device_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    args.pointee.device = UnsafeMutablePointer<PJRT_Device>(
        retainAsOpaque(buf.device)
    )
    return nil
}

func pjrt_buffer_memory(
    _ args: UnsafeMutablePointer<PJRT_Buffer_Memory_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    args.pointee.memory = UnsafeMutablePointer<PJRT_Memory>(
        retainAsOpaque(buf.memory)
    )
    return nil
}

func pjrt_buffer_delete(
    _ args: UnsafeMutablePointer<PJRT_Buffer_Delete_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    buf.markDeleted()
    return nil
}

func pjrt_buffer_is_deleted(
    _ args: UnsafeMutablePointer<PJRT_Buffer_IsDeleted_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    args.pointee.is_deleted = buf.isDeleted
    return nil
}

func pjrt_buffer_to_host_buffer(
    _ args: UnsafeMutablePointer<PJRT_Buffer_ToHostBuffer_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.src else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)

    // If dst is NULL, just report the size
    guard let dst = args.pointee.dst else {
        args.pointee.dst_size = buf.buffer.byteCount
        // Create a ready event
        let event = PJRTEventImpl()
        args.pointee.event = UnsafeMutablePointer<PJRT_Event>(
            retainAsOpaque(event)
        )
        return nil
    }

    // Copy data to host
    do {
        let data = try buf.buffer.toData()
        let copySize = min(data.count, buf.buffer.byteCount)
        data.withUnsafeBytes { src in
            dst.copyMemory(from: src.baseAddress!, byteCount: copySize)
        }
        args.pointee.dst_size = copySize

        let event = PJRTEventImpl()
        args.pointee.event = UnsafeMutablePointer<PJRT_Event>(
            retainAsOpaque(event)
        )
        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "ToHostBuffer failed: \(error)")
    }
}

func pjrt_buffer_copy_raw_to_host(
    _ args: UnsafeMutablePointer<PJRT_Buffer_CopyRawToHost_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    guard let dst = args.pointee.dst else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL destination")
    }

    do {
        let data = try buf.buffer.toData()
        let offset = Int(args.pointee.offset)
        let transferSize = Int(args.pointee.transfer_size)
        guard offset + transferSize <= data.count else {
            return makeError(PJRT_Error_Code_OUT_OF_RANGE, "Copy range exceeds buffer size")
        }
        data.withUnsafeBytes { src in
            let srcAddr = src.baseAddress!.advanced(by: offset)
            dst.copyMemory(from: srcAddr, byteCount: transferSize)
        }
        // Create completion event
        let event = PJRTEventImpl()
        args.pointee.event = UnsafeMutablePointer<PJRT_Event>(
            retainAsOpaque(event)
        )
        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "CopyRawToHost failed: \(error)")
    }
}

func pjrt_buffer_is_on_cpu(
    _ args: UnsafeMutablePointer<PJRT_Buffer_IsOnCpu_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    // Metal uses shared memory on Apple Silicon, but we're a GPU plugin
    args.pointee.is_on_cpu = false
    return nil
}

func pjrt_buffer_ready_event(
    _ args: UnsafeMutablePointer<PJRT_Buffer_ReadyEvent_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    args.pointee.event = UnsafeMutablePointer<PJRT_Event>(
        retainAsOpaque(buf.readyEvent)
    )
    return nil
}

func pjrt_buffer_unsafe_pointer(
    _ args: UnsafeMutablePointer<PJRT_Buffer_UnsafePointer_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    // We don't expose raw device pointers
    return unimplementedError("PJRT_Buffer_UnsafePointer")
}

func pjrt_buffer_increase_external_reference_count(
    _ args: UnsafeMutablePointer<PJRT_Buffer_IncreaseExternalReferenceCount_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    // No-op for now — MetalHLO buffers use ARC
    return nil
}

func pjrt_buffer_decrease_external_reference_count(
    _ args: UnsafeMutablePointer<PJRT_Buffer_DecreaseExternalReferenceCount_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    return nil
}

func pjrt_buffer_opaque_device_memory_data_pointer(
    _ args: UnsafeMutablePointer<PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    return unimplementedError("PJRT_Buffer_OpaqueDeviceMemoryDataPointer")
}

func pjrt_buffer_copy_to_device(
    _ args: UnsafeMutablePointer<PJRT_Buffer_CopyToDevice_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    // Single device — copy to self
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    do {
        let data = try buf.buffer.toData()
        let shape = buf.buffer.shape
        let et = buf.buffer.elementType
        let client = buf.device.client!
        let newBuffer = try client.client.createBuffer(bytes: data, shape: shape, elementType: et)
        let newImpl = PJRTBufferImpl(buffer: newBuffer, device: buf.device, memory: buf.memory)
        args.pointee.dst_buffer = UnsafeMutablePointer<PJRT_Buffer>(
            retainAsOpaque(newImpl)
        )
        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "CopyToDevice failed: \(error)")
    }
}

func pjrt_buffer_copy_to_memory(
    _ args: UnsafeMutablePointer<PJRT_Buffer_CopyToMemory_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    // Treat same as CopyToDevice for single-device
    guard let args = args, let bufPtr = args.pointee.buffer else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let buf = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
    do {
        let data = try buf.buffer.toData()
        let shape = buf.buffer.shape
        let et = buf.buffer.elementType
        let client = buf.device.client!
        let newBuffer = try client.client.createBuffer(bytes: data, shape: shape, elementType: et)
        let newImpl = PJRTBufferImpl(buffer: newBuffer, device: buf.device, memory: buf.memory)
        args.pointee.dst_buffer = UnsafeMutablePointer<PJRT_Buffer>(
            retainAsOpaque(newImpl)
        )
        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "CopyToMemory failed: \(error)")
    }
}
