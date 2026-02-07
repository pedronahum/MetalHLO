// PJRTExecutable.swift
// PJRTMetalHLO
//
// PJRT Executable (serializable) and LoadedExecutable (ready to run).

import CPJRTApi
import Metal
import MetalHLO
import MetalHLOCore
import Foundation

/// Concrete backing storage for opaque PJRT_Executable pointers.
/// Represents a serializable compiled artifact.
final class PJRTExecutableImpl: @unchecked Sendable {
    let name: String
    let cName: UnsafeMutablePointer<CChar>

    /// The original MLIR source, kept for serialization.
    let mlirSource: String

    /// Output element types (stable C array — pointer survives scope).
    let outputTypesBuffer: UnsafeMutableBufferPointer<PJRT_Buffer_Type>

    /// Output dimensions flattened (stable C array).
    let outputDimsBuffer: UnsafeMutableBufferPointer<Int64>

    /// Per-output dimension counts (stable C array).
    let outputDimSizesBuffer: UnsafeMutableBufferPointer<Int>

    /// Per-output memory kind C strings (stable C array).
    let outputMemoryKindsBuffer: UnsafeMutableBufferPointer<UnsafePointer<CChar>?>

    /// Per-output memory kind sizes (stable C array).
    let outputMemoryKindSizesBuffer: UnsafeMutableBufferPointer<Int>

    /// Number of outputs.
    let numOutputs: Int

    /// Serialized representation (cached). Bytes returned from serialize() point into this.
    private var serializedCache: Data?
    private let lock = NSLock()

    // Stable C string for output memory kind
    let cOutputMemoryKind: UnsafeMutablePointer<CChar>

    /// Cached fingerprint C string (lazily computed, freed in deinit).
    private var cachedFingerprint: UnsafeMutablePointer<CChar>?

    init(name: String, mlirSource: String, executable: Executable) {
        self.name = name
        self.cName = strdup(name)!
        self.mlirSource = mlirSource
        self.numOutputs = executable.outputCount
        self.cOutputMemoryKind = strdup("device")!

        // Map output types into stable C arrays
        var types: [PJRT_Buffer_Type] = []
        var dims: [Int64] = []
        var dimSizes: [Int] = []

        for outputType in executable.outputTypes {
            types.append(PJRTBufferImpl.mapElementType(outputType.elementType))
            dimSizes.append(outputType.shape.count)
            for dim in outputType.shape {
                dims.append(Int64(dim))
            }
        }

        // Allocate stable buffers that outlive any withUnsafeBufferPointer scope
        self.outputTypesBuffer = .allocate(capacity: max(types.count, 1))
        for (i, t) in types.enumerated() { outputTypesBuffer[i] = t }

        self.outputDimsBuffer = .allocate(capacity: max(dims.count, 1))
        for (i, d) in dims.enumerated() { outputDimsBuffer[i] = d }

        self.outputDimSizesBuffer = .allocate(capacity: max(dimSizes.count, 1))
        for (i, s) in dimSizes.enumerated() { outputDimSizesBuffer[i] = s }

        // All outputs use "device" memory kind
        self.outputMemoryKindsBuffer = .allocate(capacity: max(numOutputs, 1))
        self.outputMemoryKindSizesBuffer = .allocate(capacity: max(numOutputs, 1))
        let kindLen = strlen(cOutputMemoryKind)
        for i in 0..<numOutputs {
            outputMemoryKindsBuffer[i] = UnsafePointer(cOutputMemoryKind)
            outputMemoryKindSizesBuffer[i] = kindLen
        }
    }

    deinit {
        outputTypesBuffer.deallocate()
        outputDimsBuffer.deallocate()
        outputDimSizesBuffer.deallocate()
        outputMemoryKindsBuffer.deallocate()
        outputMemoryKindSizesBuffer.deallocate()
        free(cName)
        free(cOutputMemoryKind)
        if let fp = cachedFingerprint { free(fp) }
    }

    /// Returns a stable C string for the fingerprint. Owned by this object.
    func getFingerprint() -> UnsafeMutablePointer<CChar> {
        lock.lock()
        defer { lock.unlock() }
        if let existing = cachedFingerprint { return existing }
        var hasher = Hasher()
        hasher.combine(mlirSource)
        let hash = hasher.finalize()
        let fp = strdup(String(hash, radix: 16))!
        cachedFingerprint = fp
        return fp
    }

    func serialize() -> Data {
        lock.lock()
        defer { lock.unlock() }
        if let cached = serializedCache { return cached }

        // Serialize as: [format_version: UInt32][mlir_len: UInt64][mlir_bytes...]
        var data = Data()
        var version: UInt32 = 1
        data.append(Data(bytes: &version, count: 4))

        let mlirData = mlirSource.data(using: .utf8)!
        var mlirLen = UInt64(mlirData.count)
        data.append(Data(bytes: &mlirLen, count: 8))
        data.append(mlirData)

        serializedCache = data
        return data
    }
}

/// Concrete backing storage for opaque PJRT_LoadedExecutable pointers.
/// Holds a loaded executable ready for execution on specific devices.
final class PJRTLoadedExecutableImpl: @unchecked Sendable {
    let executable: Executable
    let executableMeta: PJRTExecutableImpl
    let client: PJRTClientImpl

    private let lock = NSLock()
    private var deleted = false

    // Stable device pointer array
    let devicesPtrBuffer: UnsafeMutableBufferPointer<UnsafeMutablePointer<PJRT_Device>?>

    init(executable: Executable, meta: PJRTExecutableImpl, client: PJRTClientImpl) {
        self.executable = executable
        self.executableMeta = meta
        self.client = client

        self.devicesPtrBuffer = .allocate(capacity: client.devices.count)
        for (i, device) in client.devices.enumerated() {
            devicesPtrBuffer[i] = UnsafeMutablePointer<PJRT_Device>(
                retainAsOpaque(device)
            )
        }
    }

    deinit {
        for i in 0..<devicesPtrBuffer.count {
            if let ptr = devicesPtrBuffer[i] {
                releaseOpaque(OpaquePointer(ptr), as: PJRTDeviceImpl.self)
            }
        }
        devicesPtrBuffer.deallocate()
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
}

// MARK: - Client Compile

func pjrt_client_compile(
    _ args: UnsafeMutablePointer<PJRT_Client_Compile_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args or client")
    }
    guard let program = args.pointee.program else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL program")
    }

    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)

    // Extract MLIR source from the PJRT_Program
    let prog = program.pointee
    guard let codePtr = prog.code else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL program code")
    }
    let codeSize = prog.code_size
    let mlirSource = String(
        decoding: UnsafeRawBufferPointer(start: codePtr, count: codeSize),
        as: UTF8.self
    )

    do {
        // Use O2 optimization by default for the PJRT path
        let config = CompilationConfig(optimizationLevel: .O2)
        let executable = try impl.client.compile(mlirSource, config: config)

        let meta = PJRTExecutableImpl(
            name: "metalhlo_program",
            mlirSource: mlirSource,
            executable: executable
        )
        let loaded = PJRTLoadedExecutableImpl(
            executable: executable,
            meta: meta,
            client: impl
        )
        args.pointee.executable = UnsafeMutablePointer<PJRT_LoadedExecutable>(
            retainAsOpaque(loaded)
        )
        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "Compilation failed: \(error)")
    }
}

// MARK: - Executable Functions

func pjrt_executable_destroy(
    _ args: UnsafeMutablePointer<PJRT_Executable_Destroy_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else { return nil }
    releaseOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    return nil
}

func pjrt_executable_name(
    _ args: UnsafeMutablePointer<PJRT_Executable_Name_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    args.pointee.executable_name = UnsafePointer(exe.cName)
    args.pointee.executable_name_size = strlen(exe.cName)
    return nil
}

func pjrt_executable_num_replicas(
    _ args: UnsafeMutablePointer<PJRT_Executable_NumReplicas_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.num_replicas = 1
    return nil
}

func pjrt_executable_num_partitions(
    _ args: UnsafeMutablePointer<PJRT_Executable_NumPartitions_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.num_partitions = 1
    return nil
}

func pjrt_executable_num_outputs(
    _ args: UnsafeMutablePointer<PJRT_Executable_NumOutputs_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    args.pointee.num_outputs = exe.numOutputs
    return nil
}

func pjrt_executable_size_of_generated_code(
    _ args: UnsafeMutablePointer<PJRT_Executable_SizeOfGeneratedCodeInBytes_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.size_in_bytes = 0 // Unknown
    return nil
}

func pjrt_executable_get_cost_analysis(
    _ args: UnsafeMutablePointer<PJRT_Executable_GetCostAnalysis_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.num_properties = 0
    args.pointee.properties = nil
    return nil
}

func pjrt_executable_output_memory_kinds(
    _ args: UnsafeMutablePointer<PJRT_Executable_OutputMemoryKinds_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    args.pointee.num_outputs = exe.numOutputs
    args.pointee.memory_kinds = UnsafePointer(exe.outputMemoryKindsBuffer.baseAddress!)
    args.pointee.memory_kind_sizes = UnsafePointer(exe.outputMemoryKindSizesBuffer.baseAddress!)
    return nil
}

func pjrt_executable_optimized_program(
    _ args: UnsafeMutablePointer<PJRT_Executable_OptimizedProgram_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    return unimplementedError("PJRT_Executable_OptimizedProgram")
}

func pjrt_executable_serialize(
    _ args: UnsafeMutablePointer<PJRT_Executable_Serialize_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    let data = exe.serialize()

    // Return a pointer into the cached Data owned by exe.
    // The bytes remain valid as long as the PJRTExecutableImpl is alive.
    data.withUnsafeBytes { src in
        args.pointee.serialized_bytes = src.bindMemory(to: CChar.self).baseAddress
        args.pointee.serialized_bytes_size = src.count
    }
    return nil
}

func pjrt_executable_output_element_types(
    _ args: UnsafeMutablePointer<PJRT_Executable_OutputElementTypes_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    args.pointee.output_types = UnsafeMutablePointer(exe.outputTypesBuffer.baseAddress!)
    args.pointee.num_output_types = exe.numOutputs
    return nil
}

func pjrt_executable_output_dimensions(
    _ args: UnsafeMutablePointer<PJRT_Executable_OutputDimensions_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    args.pointee.num_outputs = exe.numOutputs
    args.pointee.dims = UnsafePointer(exe.outputDimsBuffer.baseAddress!)
    args.pointee.dim_sizes = UnsafePointer(exe.outputDimSizesBuffer.baseAddress!)
    return nil
}

func pjrt_executable_fingerprint(
    _ args: UnsafeMutablePointer<PJRT_Executable_Fingerprint_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    let fp = exe.getFingerprint()
    args.pointee.executable_fingerprint = UnsafePointer(fp)
    args.pointee.executable_fingerprint_size = strlen(fp)
    return nil
}

func pjrt_executable_get_compiled_memory_stats(
    _ args: UnsafeMutablePointer<PJRT_Executable_GetCompiledMemoryStats_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    return unimplementedError("PJRT_Executable_GetCompiledMemoryStats")
}

// MARK: - LoadedExecutable Functions

func pjrt_loaded_executable_destroy(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_Destroy_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else { return nil }
    releaseOpaque(OpaquePointer(exePtr), as: PJRTLoadedExecutableImpl.self)
    return nil
}

func pjrt_loaded_executable_get_executable(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_GetExecutable_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.loaded_executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)
    args.pointee.executable = UnsafeMutablePointer<PJRT_Executable>(
        retainAsOpaque(loaded.executableMeta)
    )
    return nil
}

func pjrt_loaded_executable_addressable_devices(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_AddressableDevices_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)
    args.pointee.addressable_devices = UnsafePointer(loaded.devicesPtrBuffer.baseAddress)
    args.pointee.num_addressable_devices = loaded.devicesPtrBuffer.count
    return nil
}

func pjrt_loaded_executable_delete(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_Delete_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)
    loaded.markDeleted()
    return nil
}

func pjrt_loaded_executable_is_deleted(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_IsDeleted_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)
    args.pointee.is_deleted = loaded.isDeleted
    return nil
}

func pjrt_loaded_executable_execute(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_Execute_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)

    let numDevices = Int(args.pointee.num_devices)
    let numArgs = Int(args.pointee.num_args)

    // We only support single-device execution
    guard numDevices <= 1 else {
        return makeError(PJRT_Error_Code_UNIMPLEMENTED, "Multi-device execution not supported")
    }

    // Extract input buffers from argument_lists[0]
    var inputs: [Buffer] = []
    if let argLists = args.pointee.argument_lists, numDevices > 0 {
        guard let deviceArgs = argLists[0] else {
            return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL argument list for device 0")
        }
        for i in 0..<numArgs {
            guard let bufPtr = deviceArgs[i] else {
                return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL buffer at argument index \(i)")
            }
            let bufImpl = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
            inputs.append(bufImpl.buffer)
        }
    }

    // Execute
    do {
        let outputs = try loaded.executable.execute(inputs)

        // Write output buffers to output_lists[0]
        if let outputLists = args.pointee.output_lists, numDevices > 0,
           let deviceOutputs = outputLists[0] {
            for (i, output) in outputs.enumerated() {
                let device = loaded.client.devices[0]
                let memory = loaded.client.memories[0]
                let bufImpl = PJRTBufferImpl(
                    buffer: output,
                    device: device,
                    memory: memory
                )
                deviceOutputs[i] = UnsafeMutablePointer<PJRT_Buffer>(
                    retainAsOpaque(bufImpl)
                )
            }
        }

        // Create completion event
        if let events = args.pointee.device_complete_events, numDevices > 0 {
            let event = PJRTEventImpl() // Already complete (synchronous execution)
            events[0] = UnsafeMutablePointer<PJRT_Event>(
                retainAsOpaque(event)
            )
        }

        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "Execution failed: \(error)")
    }
}

func pjrt_loaded_executable_fingerprint(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_Fingerprint_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)
    let fp = loaded.executableMeta.getFingerprint()
    args.pointee.executable_fingerprint = UnsafePointer(fp)
    args.pointee.executable_fingerprint_size = strlen(fp)
    return nil
}

// MARK: - DeserializeAndLoad

func pjrt_executable_deserialize_and_load(
    _ args: UnsafeMutablePointer<PJRT_Executable_DeserializeAndLoad_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args or client")
    }
    guard let bytes = args.pointee.serialized_executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL serialized bytes")
    }

    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)
    let size = args.pointee.serialized_executable_size

    let data = Data(bytes: bytes, count: size)

    // Deserialize: [version: UInt32][mlir_len: UInt64][mlir_bytes...]
    guard data.count >= 12 else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "Serialized data too short")
    }

    let version = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) }
    guard version == 1 else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "Unknown serialization version: \(version)")
    }

    let mlirLen = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt64.self) }
    guard data.count >= 12 + Int(mlirLen) else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "Serialized data truncated")
    }

    let mlirData = data.subdata(in: 12..<(12 + Int(mlirLen)))
    guard let mlirSource = String(data: mlirData, encoding: .utf8) else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid MLIR UTF-8")
    }

    // Re-compile
    do {
        let config = CompilationConfig(optimizationLevel: .O2)
        let executable = try impl.client.compile(mlirSource, config: config)

        let meta = PJRTExecutableImpl(
            name: "metalhlo_deserialized",
            mlirSource: mlirSource,
            executable: executable
        )
        let loaded = PJRTLoadedExecutableImpl(
            executable: executable,
            meta: meta,
            client: impl
        )
        args.pointee.loaded_executable = UnsafeMutablePointer<PJRT_LoadedExecutable>(
            retainAsOpaque(loaded)
        )
        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "Deserialization re-compilation failed: \(error)")
    }
}
