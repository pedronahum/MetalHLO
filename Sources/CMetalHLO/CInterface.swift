// CInterface.swift
// CMetalHLO
//
// C API implementation for MetalHLO.

import Foundation
import MetalHLO

// MARK: - C-Compatible Type Aliases

// Status codes (matches MHLOStatusCode enum in header)
// MHLO_OK = 0, MHLO_ERROR_NO_DEVICE = 1, etc.
private let MHLO_OK: Int32 = 0
private let MHLO_ERROR_NO_DEVICE: Int32 = 1
private let MHLO_ERROR_UNSUPPORTED_DEVICE: Int32 = 2
private let MHLO_ERROR_PARSE_FAILED: Int32 = 3
private let MHLO_ERROR_UNSUPPORTED_OP: Int32 = 4
private let MHLO_ERROR_COMPILATION_FAILED: Int32 = 5
private let MHLO_ERROR_BUFFER_FAILED: Int32 = 6
private let MHLO_ERROR_INPUT_MISMATCH: Int32 = 7
private let MHLO_ERROR_EXECUTION_FAILED: Int32 = 8
private let MHLO_ERROR_TRANSFER_FAILED: Int32 = 9
private let MHLO_ERROR_INVALID_ARGUMENT: Int32 = 10

// Element type codes (matches MHLOElementType enum in header)
private let MHLO_F16: Int32 = 0
private let MHLO_F32: Int32 = 1
private let MHLO_F64: Int32 = 2
private let MHLO_BF16: Int32 = 3
private let MHLO_I1: Int32 = 4
private let MHLO_I8: Int32 = 5
private let MHLO_I16: Int32 = 6
private let MHLO_I32: Int32 = 7
private let MHLO_I64: Int32 = 8
private let MHLO_UI8: Int32 = 9
private let MHLO_UI16: Int32 = 10
private let MHLO_UI32: Int32 = 11
private let MHLO_UI64: Int32 = 12

// MARK: - Opaque Handle Wrappers

/// Wrapper class to hold a Client reference across the C boundary.
final class ClientHandle {
    let client: Client
    init(_ client: Client) { self.client = client }
}

/// Wrapper class to hold an Executable reference across the C boundary.
final class ExecutableHandle {
    let executable: Executable
    // Store shapes as contiguous arrays that C can reference
    var inputShapes: [[Int64]] = []
    var outputShapes: [[Int64]] = []

    init(_ executable: Executable) {
        self.executable = executable
        // Pre-compute shapes for C API access
        self.inputShapes = executable.inputTypes.map { $0.shape.map { Int64($0) } }
        self.outputShapes = executable.outputTypes.map { $0.shape.map { Int64($0) } }
    }
}

/// Wrapper class to hold a Buffer reference across the C boundary.
final class BufferHandle {
    let buffer: Buffer
    // Store shape as contiguous array that C can reference
    var shapeArray: [Int64]

    init(_ buffer: Buffer) {
        self.buffer = buffer
        self.shapeArray = buffer.shape.map { Int64($0) }
    }
}

// MARK: - Status Storage

/// Thread-local storage for the last error message.
/// This allows us to return status codes while keeping error messages available.
/// Note: nonisolated(unsafe) is used because this is a C API where thread safety
/// is the caller's responsibility, as is typical for C libraries.
nonisolated(unsafe) private var lastErrorMessage: UnsafeMutablePointer<CChar>?

/// Set the last error message and return the status code.
private func setError(_ code: Int32, _ message: String) -> Int32 {
    // Free any previous error message
    if let prev = lastErrorMessage {
        free(prev)
    }
    lastErrorMessage = strdup(message)
    return code
}

/// Clear error state.
private func clearError() {
    if let prev = lastErrorMessage {
        free(prev)
        lastErrorMessage = nil
    }
}

/// Converts a MetalHLOError to an error code, setting the message.
private func statusFromError(_ error: MetalHLOError) -> Int32 {
    switch error {
    case .noMetalDevice:
        return setError(MHLO_ERROR_NO_DEVICE, error.description)
    case .unsupportedDevice:
        return setError(MHLO_ERROR_UNSUPPORTED_DEVICE, error.description)
    case .parseFailed:
        return setError(MHLO_ERROR_PARSE_FAILED, error.description)
    case .unsupportedOperation:
        return setError(MHLO_ERROR_UNSUPPORTED_OP, error.description)
    case .compilationFailed:
        return setError(MHLO_ERROR_COMPILATION_FAILED, error.description)
    case .bufferCreationFailed:
        return setError(MHLO_ERROR_BUFFER_FAILED, error.description)
    case .inputMismatch, .inputTypeMismatch:
        return setError(MHLO_ERROR_INPUT_MISMATCH, error.description)
    case .executionFailed:
        return setError(MHLO_ERROR_EXECUTION_FAILED, error.description)
    case .transferFailed:
        return setError(MHLO_ERROR_TRANSFER_FAILED, error.description)
    }
}

/// Converts element type code to Swift ElementType.
private func elementTypeFromC(_ cType: Int32) -> ElementType {
    switch cType {
    case MHLO_F16: return .float16
    case MHLO_F32: return .float32
    case MHLO_F64: return .float64
    case MHLO_BF16: return .bfloat16
    case MHLO_I1: return .int1
    case MHLO_I8: return .int8
    case MHLO_I16: return .int16
    case MHLO_I32: return .int32
    case MHLO_I64: return .int64
    case MHLO_UI8: return .uint8
    case MHLO_UI16: return .uint16
    case MHLO_UI32: return .uint32
    case MHLO_UI64: return .uint64
    default: return .float32
    }
}

/// Converts Swift ElementType to C element type code.
private func elementTypeToC(_ swiftType: ElementType) -> Int32 {
    switch swiftType {
    case .float16: return MHLO_F16
    case .float32: return MHLO_F32
    case .float64: return MHLO_F64
    case .bfloat16: return MHLO_BF16
    case .int1: return MHLO_I1
    case .int8: return MHLO_I8
    case .int16: return MHLO_I16
    case .int32: return MHLO_I32
    case .int64: return MHLO_I64
    case .uint8: return MHLO_UI8
    case .uint16: return MHLO_UI16
    case .uint32: return MHLO_UI32
    case .uint64: return MHLO_UI64
    }
}

// MARK: - Version

/// Get the library version string.
@_cdecl("mhlo_version")
public func mhlo_version() -> UnsafePointer<CChar>? {
    // Return a static string - this doesn't need to be freed
    return ("0.1.0" as NSString).utf8String
}

// MARK: - Error Message

/// Get the last error message (NULL if no error).
@_cdecl("mhlo_get_last_error")
public func mhlo_get_last_error() -> UnsafePointer<CChar>? {
    return UnsafePointer(lastErrorMessage)
}

/// Clear the last error message.
@_cdecl("mhlo_clear_error")
public func mhlo_clear_error() {
    clearError()
}

// MARK: - Utility

/// Free a string returned by the API.
@_cdecl("mhlo_free_string")
public func mhlo_free_string(_ str: UnsafePointer<CChar>?) {
    guard let str = str else { return }
    // Free strings allocated with strdup
    free(UnsafeMutablePointer(mutating: str))
}

/// Get size in bytes for an element type.
@_cdecl("mhlo_element_type_size")
public func mhlo_element_type_size(_ type: Int32) -> Int {
    let swiftType = elementTypeFromC(type)
    return swiftType.byteSize
}

// MARK: - Client API

/// Create a client for the default Metal device.
/// Returns: MHLO_OK on success, error code otherwise.
@_cdecl("mhlo_client_create")
public func mhlo_client_create(_ outClient: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let outClient = outClient else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "out_client is null")
    }

    do {
        clearError()
        let client = try Client.create()
        let handle = ClientHandle(client)
        let retained = Unmanaged.passRetained(handle).toOpaque()
        outClient.pointee = OpaquePointer(retained)
        return MHLO_OK
    } catch let error as MetalHLOError {
        return statusFromError(error)
    } catch {
        return setError(MHLO_ERROR_NO_DEVICE, error.localizedDescription)
    }
}

/// Destroy a client.
@_cdecl("mhlo_client_destroy")
public func mhlo_client_destroy(_ client: OpaquePointer?) {
    guard let client = client else { return }
    Unmanaged<ClientHandle>.fromOpaque(UnsafeRawPointer(client)).release()
}

/// Get device name.
/// Returns: Device name string (caller must free with mhlo_free_string).
@_cdecl("mhlo_client_device_name")
public func mhlo_client_device_name(_ client: OpaquePointer?) -> UnsafeMutablePointer<CChar>? {
    guard let client = client else { return nil }
    let handle = Unmanaged<ClientHandle>.fromOpaque(UnsafeRawPointer(client)).takeUnretainedValue()
    return strdup(handle.client.deviceName)
}

// MARK: - Compilation API

/// Compile StableHLO MLIR text.
/// Returns: MHLO_OK on success, error code otherwise.
@_cdecl("mhlo_compile")
public func mhlo_compile(
    _ client: OpaquePointer?,
    _ mlirText: UnsafePointer<CChar>?,
    _ outExecutable: UnsafeMutablePointer<OpaquePointer?>?
) -> Int32 {
    guard let client = client else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "client is null")
    }
    guard let mlirText = mlirText else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "mlir_text is null")
    }
    guard let outExecutable = outExecutable else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "out_executable is null")
    }

    let handle = Unmanaged<ClientHandle>.fromOpaque(UnsafeRawPointer(client)).takeUnretainedValue()
    let mlirString = String(cString: mlirText)

    do {
        clearError()
        let executable = try handle.client.compile(mlirString)
        let execHandle = ExecutableHandle(executable)
        let retained = Unmanaged.passRetained(execHandle).toOpaque()
        outExecutable.pointee = OpaquePointer(retained)
        return MHLO_OK
    } catch let error as MetalHLOError {
        return statusFromError(error)
    } catch {
        return setError(MHLO_ERROR_COMPILATION_FAILED, error.localizedDescription)
    }
}

/// Compile StableHLO MLIR with explicit length.
@_cdecl("mhlo_compile_n")
public func mhlo_compile_n(
    _ client: OpaquePointer?,
    _ mlirText: UnsafePointer<CChar>?,
    _ mlirLength: Int,
    _ outExecutable: UnsafeMutablePointer<OpaquePointer?>?
) -> Int32 {
    guard let client = client else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "client is null")
    }
    guard let mlirText = mlirText else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "mlir_text is null")
    }
    guard let outExecutable = outExecutable else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "out_executable is null")
    }

    let handle = Unmanaged<ClientHandle>.fromOpaque(UnsafeRawPointer(client)).takeUnretainedValue()

    // Create string from buffer with explicit length
    let data = Data(bytes: mlirText, count: mlirLength)
    guard let mlirString = String(data: data, encoding: .utf8) else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "Invalid UTF-8 in MLIR text")
    }

    do {
        clearError()
        let executable = try handle.client.compile(mlirString)
        let execHandle = ExecutableHandle(executable)
        let retained = Unmanaged.passRetained(execHandle).toOpaque()
        outExecutable.pointee = OpaquePointer(retained)
        return MHLO_OK
    } catch let error as MetalHLOError {
        return statusFromError(error)
    } catch {
        return setError(MHLO_ERROR_COMPILATION_FAILED, error.localizedDescription)
    }
}

/// Destroy an executable.
@_cdecl("mhlo_executable_destroy")
public func mhlo_executable_destroy(_ executable: OpaquePointer?) {
    guard let executable = executable else { return }
    Unmanaged<ExecutableHandle>.fromOpaque(UnsafeRawPointer(executable)).release()
}

/// Get number of inputs.
@_cdecl("mhlo_executable_input_count")
public func mhlo_executable_input_count(_ executable: OpaquePointer?) -> Int32 {
    guard let executable = executable else { return 0 }
    let handle = Unmanaged<ExecutableHandle>.fromOpaque(UnsafeRawPointer(executable)).takeUnretainedValue()
    return Int32(handle.executable.inputCount)
}

/// Get number of outputs.
@_cdecl("mhlo_executable_output_count")
public func mhlo_executable_output_count(_ executable: OpaquePointer?) -> Int32 {
    guard let executable = executable else { return 0 }
    let handle = Unmanaged<ExecutableHandle>.fromOpaque(UnsafeRawPointer(executable)).takeUnretainedValue()
    return Int32(handle.executable.outputCount)
}

/// Get input tensor shape.
/// Parameters:
///   - executable: The executable
///   - index: Input index
///   - outShape: Pre-allocated array for shape dimensions
///   - outRank: Output for rank
///   - outElementType: Output for element type
/// Returns: MHLO_OK on success.
@_cdecl("mhlo_executable_input_type")
public func mhlo_executable_input_type(
    _ executable: OpaquePointer?,
    _ index: Int32,
    _ outShape: UnsafeMutablePointer<Int64>?,
    _ outRank: UnsafeMutablePointer<Int32>?,
    _ outElementType: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let executable = executable else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "executable is null")
    }

    let handle = Unmanaged<ExecutableHandle>.fromOpaque(UnsafeRawPointer(executable)).takeUnretainedValue()
    let idx = Int(index)

    guard idx >= 0 && idx < handle.executable.inputCount else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "Input index \(index) out of range")
    }

    let inputType = handle.executable.inputTypes[idx]

    if let outRank = outRank {
        outRank.pointee = Int32(inputType.shape.count)
    }

    if let outShape = outShape {
        for (i, dim) in inputType.shape.enumerated() {
            outShape[i] = Int64(dim)
        }
    }

    if let outElementType = outElementType {
        outElementType.pointee = elementTypeToC(inputType.elementType)
    }

    return MHLO_OK
}

/// Get output tensor shape.
@_cdecl("mhlo_executable_output_type")
public func mhlo_executable_output_type(
    _ executable: OpaquePointer?,
    _ index: Int32,
    _ outShape: UnsafeMutablePointer<Int64>?,
    _ outRank: UnsafeMutablePointer<Int32>?,
    _ outElementType: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let executable = executable else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "executable is null")
    }

    let handle = Unmanaged<ExecutableHandle>.fromOpaque(UnsafeRawPointer(executable)).takeUnretainedValue()
    let idx = Int(index)

    guard idx >= 0 && idx < handle.executable.outputCount else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "Output index \(index) out of range")
    }

    let outputType = handle.executable.outputTypes[idx]

    if let outRank = outRank {
        outRank.pointee = Int32(outputType.shape.count)
    }

    if let outShape = outShape {
        for (i, dim) in outputType.shape.enumerated() {
            outShape[i] = Int64(dim)
        }
    }

    if let outElementType = outElementType {
        outElementType.pointee = elementTypeToC(outputType.elementType)
    }

    return MHLO_OK
}

// MARK: - Buffer API

/// Create a buffer from host data.
@_cdecl("mhlo_buffer_create")
public func mhlo_buffer_create(
    _ client: OpaquePointer?,
    _ data: UnsafeRawPointer?,
    _ dataSize: Int,
    _ shape: UnsafePointer<Int64>?,
    _ rank: Int32,
    _ elementType: Int32,
    _ outBuffer: UnsafeMutablePointer<OpaquePointer?>?
) -> Int32 {
    guard let client = client else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "client is null")
    }
    guard let data = data else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "data is null")
    }
    guard let outBuffer = outBuffer else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "out_buffer is null")
    }

    let handle = Unmanaged<ClientHandle>.fromOpaque(UnsafeRawPointer(client)).takeUnretainedValue()

    // Convert shape
    var shapeArray: [Int] = []
    if let shape = shape, rank > 0 {
        for i in 0..<Int(rank) {
            shapeArray.append(Int(shape[i]))
        }
    }

    let swiftElementType = elementTypeFromC(elementType)

    do {
        clearError()
        // Create Data from raw pointer
        let dataObj = Data(bytes: data, count: dataSize)
        let buffer = try handle.client.createBuffer(bytes: dataObj, shape: shapeArray, elementType: swiftElementType)
        let bufHandle = BufferHandle(buffer)
        let retained = Unmanaged.passRetained(bufHandle).toOpaque()
        outBuffer.pointee = OpaquePointer(retained)
        return MHLO_OK
    } catch let error as MetalHLOError {
        return statusFromError(error)
    } catch {
        return setError(MHLO_ERROR_BUFFER_FAILED, error.localizedDescription)
    }
}

/// Create an uninitialized buffer.
@_cdecl("mhlo_buffer_create_uninitialized")
public func mhlo_buffer_create_uninitialized(
    _ client: OpaquePointer?,
    _ shape: UnsafePointer<Int64>?,
    _ rank: Int32,
    _ elementType: Int32,
    _ outBuffer: UnsafeMutablePointer<OpaquePointer?>?
) -> Int32 {
    guard let client = client else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "client is null")
    }
    guard let outBuffer = outBuffer else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "out_buffer is null")
    }

    let handle = Unmanaged<ClientHandle>.fromOpaque(UnsafeRawPointer(client)).takeUnretainedValue()

    // Convert shape
    var shapeArray: [Int] = []
    if let shape = shape, rank > 0 {
        for i in 0..<Int(rank) {
            shapeArray.append(Int(shape[i]))
        }
    }

    let swiftElementType = elementTypeFromC(elementType)

    do {
        clearError()
        let buffer = try handle.client.createBuffer(shape: shapeArray, elementType: swiftElementType)
        let bufHandle = BufferHandle(buffer)
        let retained = Unmanaged.passRetained(bufHandle).toOpaque()
        outBuffer.pointee = OpaquePointer(retained)
        return MHLO_OK
    } catch let error as MetalHLOError {
        return statusFromError(error)
    } catch {
        return setError(MHLO_ERROR_BUFFER_FAILED, error.localizedDescription)
    }
}

/// Destroy a buffer.
@_cdecl("mhlo_buffer_destroy")
public func mhlo_buffer_destroy(_ buffer: OpaquePointer?) {
    guard let buffer = buffer else { return }
    Unmanaged<BufferHandle>.fromOpaque(UnsafeRawPointer(buffer)).release()
}

/// Get buffer shape.
@_cdecl("mhlo_buffer_shape")
public func mhlo_buffer_shape(
    _ buffer: OpaquePointer?,
    _ outShape: UnsafeMutablePointer<Int64>?,
    _ outRank: UnsafeMutablePointer<Int32>?
) {
    guard let buffer = buffer else { return }
    let handle = Unmanaged<BufferHandle>.fromOpaque(UnsafeRawPointer(buffer)).takeUnretainedValue()

    if let outRank = outRank {
        outRank.pointee = Int32(handle.buffer.shape.count)
    }

    if let outShape = outShape {
        for (i, dim) in handle.buffer.shape.enumerated() {
            outShape[i] = Int64(dim)
        }
    }
}

/// Get buffer rank.
@_cdecl("mhlo_buffer_rank")
public func mhlo_buffer_rank(_ buffer: OpaquePointer?) -> Int32 {
    guard let buffer = buffer else { return 0 }
    let handle = Unmanaged<BufferHandle>.fromOpaque(UnsafeRawPointer(buffer)).takeUnretainedValue()
    return Int32(handle.buffer.shape.count)
}

/// Get buffer element type.
@_cdecl("mhlo_buffer_element_type")
public func mhlo_buffer_element_type(_ buffer: OpaquePointer?) -> Int32 {
    guard let buffer = buffer else { return MHLO_F32 }
    let handle = Unmanaged<BufferHandle>.fromOpaque(UnsafeRawPointer(buffer)).takeUnretainedValue()
    return elementTypeToC(handle.buffer.elementType)
}

/// Get buffer size in bytes.
@_cdecl("mhlo_buffer_byte_count")
public func mhlo_buffer_byte_count(_ buffer: OpaquePointer?) -> Int {
    guard let buffer = buffer else { return 0 }
    let handle = Unmanaged<BufferHandle>.fromOpaque(UnsafeRawPointer(buffer)).takeUnretainedValue()
    return handle.buffer.byteCount
}

/// Copy buffer contents to host.
@_cdecl("mhlo_buffer_to_host")
public func mhlo_buffer_to_host(
    _ buffer: OpaquePointer?,
    _ outData: UnsafeMutableRawPointer?,
    _ dataSize: Int
) -> Int32 {
    guard let buffer = buffer else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "buffer is null")
    }
    guard let outData = outData else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "out_data is null")
    }

    let handle = Unmanaged<BufferHandle>.fromOpaque(UnsafeRawPointer(buffer)).takeUnretainedValue()

    do {
        let data = try handle.buffer.toData()

        guard data.count <= dataSize else {
            return setError(MHLO_ERROR_TRANSFER_FAILED, "Output buffer too small: need \(data.count), got \(dataSize)")
        }

        data.copyBytes(to: outData.assumingMemoryBound(to: UInt8.self), count: data.count)
        return MHLO_OK
    } catch let error as MetalHLOError {
        return statusFromError(error)
    } catch {
        return setError(MHLO_ERROR_TRANSFER_FAILED, error.localizedDescription)
    }
}

// MARK: - Execution API

/// Execute a compiled program.
@_cdecl("mhlo_execute")
public func mhlo_execute(
    _ executable: OpaquePointer?,
    _ inputs: UnsafePointer<OpaquePointer?>?,
    _ numInputs: Int32,
    _ outOutputs: UnsafeMutablePointer<OpaquePointer?>?,
    _ outNumOutputs: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let executable = executable else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "executable is null")
    }
    guard let outOutputs = outOutputs else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "out_outputs is null")
    }
    guard let outNumOutputs = outNumOutputs else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "out_num_outputs is null")
    }

    let execHandle = Unmanaged<ExecutableHandle>.fromOpaque(UnsafeRawPointer(executable)).takeUnretainedValue()

    // Convert input buffers
    var inputBuffers: [Buffer] = []
    if numInputs > 0, let inputs = inputs {
        for i in 0..<Int(numInputs) {
            guard let inputPtr = inputs[i] else {
                return setError(MHLO_ERROR_INVALID_ARGUMENT, "Input \(i) is null")
            }
            let bufHandle = Unmanaged<BufferHandle>.fromOpaque(UnsafeRawPointer(inputPtr)).takeUnretainedValue()
            inputBuffers.append(bufHandle.buffer)
        }
    }

    do {
        clearError()
        let outputs = try execHandle.executable.execute(inputBuffers)

        outNumOutputs.pointee = Int32(outputs.count)

        for (i, output) in outputs.enumerated() {
            let bufHandle = BufferHandle(output)
            let retained = Unmanaged.passRetained(bufHandle).toOpaque()
            outOutputs[i] = OpaquePointer(retained)
        }

        return MHLO_OK
    } catch let error as MetalHLOError {
        return statusFromError(error)
    } catch {
        return setError(MHLO_ERROR_EXECUTION_FAILED, error.localizedDescription)
    }
}

/// Execute with timing information.
/// Outputs timing via separate pointers for encode, gpu, and total time.
@_cdecl("mhlo_execute_with_timing")
public func mhlo_execute_with_timing(
    _ executable: OpaquePointer?,
    _ inputs: UnsafePointer<OpaquePointer?>?,
    _ numInputs: Int32,
    _ outOutputs: UnsafeMutablePointer<OpaquePointer?>?,
    _ outNumOutputs: UnsafeMutablePointer<Int32>?,
    _ outEncodeTime: UnsafeMutablePointer<Double>?,
    _ outGpuTime: UnsafeMutablePointer<Double>?,
    _ outTotalTime: UnsafeMutablePointer<Double>?
) -> Int32 {
    guard let executable = executable else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "executable is null")
    }
    guard let outOutputs = outOutputs else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "out_outputs is null")
    }
    guard let outNumOutputs = outNumOutputs else {
        return setError(MHLO_ERROR_INVALID_ARGUMENT, "out_num_outputs is null")
    }

    let execHandle = Unmanaged<ExecutableHandle>.fromOpaque(UnsafeRawPointer(executable)).takeUnretainedValue()

    // Convert input buffers
    var inputBuffers: [Buffer] = []
    if numInputs > 0, let inputs = inputs {
        for i in 0..<Int(numInputs) {
            guard let inputPtr = inputs[i] else {
                return setError(MHLO_ERROR_INVALID_ARGUMENT, "Input \(i) is null")
            }
            let bufHandle = Unmanaged<BufferHandle>.fromOpaque(UnsafeRawPointer(inputPtr)).takeUnretainedValue()
            inputBuffers.append(bufHandle.buffer)
        }
    }

    do {
        clearError()
        let (outputs, timing) = try execHandle.executable.executeWithTiming(inputBuffers)

        outNumOutputs.pointee = Int32(outputs.count)

        for (i, output) in outputs.enumerated() {
            let bufHandle = BufferHandle(output)
            let retained = Unmanaged.passRetained(bufHandle).toOpaque()
            outOutputs[i] = OpaquePointer(retained)
        }

        if let outEncodeTime = outEncodeTime {
            outEncodeTime.pointee = timing.encodeTime
        }
        if let outGpuTime = outGpuTime {
            outGpuTime.pointee = timing.gpuTime
        }
        if let outTotalTime = outTotalTime {
            outTotalTime.pointee = timing.totalTime
        }

        return MHLO_OK
    } catch let error as MetalHLOError {
        return statusFromError(error)
    } catch {
        return setError(MHLO_ERROR_EXECUTION_FAILED, error.localizedDescription)
    }
}
