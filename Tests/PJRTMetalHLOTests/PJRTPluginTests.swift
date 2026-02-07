// PJRTPluginTests.swift
// PJRTMetalHLOTests

import Testing
import CPJRTApi
import Darwin
@testable import PJRTMetalHLO

// MARK: - Test Helpers

private func createClient() -> UnsafeMutablePointer<PJRT_Client>? {
    var args = PJRT_Client_Create_Args()
    args.struct_size = MemoryLayout<PJRT_Client_Create_Args>.size
    let error = pjrt_client_create(&args)
    if let error = error { destroyError(error) }
    return args.client
}

private func destroyClient(_ client: UnsafeMutablePointer<PJRT_Client>) {
    var args = PJRT_Client_Destroy_Args()
    args.struct_size = MemoryLayout<PJRT_Client_Destroy_Args>.size
    args.client = client
    let _ = pjrt_client_destroy(&args)
}

private func createF32Buffer(
    client: UnsafeMutablePointer<PJRT_Client>,
    data: [Float],
    shape: [Int64]
) -> UnsafeMutablePointer<PJRT_Buffer>? {
    var bufArgs = PJRT_Client_BufferFromHostBuffer_Args()
    bufArgs.struct_size = MemoryLayout<PJRT_Client_BufferFromHostBuffer_Args>.size
    bufArgs.client = client
    bufArgs.type = PJRT_Buffer_Type_F32
    bufArgs.num_dims = shape.count
    data.withUnsafeBufferPointer { dataPtr in
        shape.withUnsafeBufferPointer { shapePtr in
            bufArgs.data = UnsafeRawPointer(dataPtr.baseAddress)
            bufArgs.dims = shapePtr.baseAddress
            let error = pjrt_client_buffer_from_host_buffer(&bufArgs)
            if let error = error { destroyError(error) }
        }
    }
    return bufArgs.buffer
}

private func compile(
    client: UnsafeMutablePointer<PJRT_Client>,
    mlir: String
) -> UnsafeMutablePointer<PJRT_LoadedExecutable>? {
    let cString = mlir.utf8CString
    var program = PJRT_Program()
    program.struct_size = MemoryLayout<PJRT_Program>.size
    program.code_size = mlir.utf8.count

    var compileArgs = PJRT_Client_Compile_Args()
    compileArgs.struct_size = MemoryLayout<PJRT_Client_Compile_Args>.size
    compileArgs.client = client

    cString.withUnsafeBufferPointer { buf in
        program.code = UnsafeMutablePointer(mutating: buf.baseAddress)
        withUnsafeMutablePointer(to: &program) { progPtr in
            compileArgs.program = UnsafePointer(progPtr)
            let error = pjrt_client_compile(&compileArgs)
            if let error = error { destroyError(error) }
        }
    }
    return compileArgs.executable
}

private func execute(
    loadedExe: UnsafeMutablePointer<PJRT_LoadedExecutable>,
    inputs: [UnsafeMutablePointer<PJRT_Buffer>],
    numOutputs: Int
) -> [UnsafeMutablePointer<PJRT_Buffer>?] {
    var options = PJRT_ExecuteOptions()
    options.struct_size = MemoryLayout<PJRT_ExecuteOptions>.size

    var argList: [UnsafeMutablePointer<PJRT_Buffer>?] = inputs.map { $0 }
    var outputList = [UnsafeMutablePointer<PJRT_Buffer>?](repeating: nil, count: numOutputs)

    var executeArgs = PJRT_LoadedExecutable_Execute_Args()
    executeArgs.struct_size = MemoryLayout<PJRT_LoadedExecutable_Execute_Args>.size
    executeArgs.executable = loadedExe
    executeArgs.num_devices = 1
    executeArgs.num_args = inputs.count

    argList.withUnsafeMutableBufferPointer { argBuf in
        outputList.withUnsafeMutableBufferPointer { outBuf in
            // argument_lists: PJRT_Buffer* const* const* (3-level pointer)
            var argBase = UnsafePointer<UnsafeMutablePointer<PJRT_Buffer>?>(argBuf.baseAddress)
            // output_lists: PJRT_Buffer** const* (3-level pointer, middle is mutable)
            var outBase = outBuf.baseAddress
            withUnsafeMutablePointer(to: &options) { optPtr in
                executeArgs.options = optPtr
                withUnsafePointer(to: &argBase) { ap in
                    executeArgs.argument_lists = UnsafePointer(ap)
                    withUnsafePointer(to: &outBase) { op in
                        executeArgs.output_lists = UnsafePointer(op)
                        let error = pjrt_loaded_executable_execute(&executeArgs)
                        if let error = error { destroyError(error) }
                    }
                }
            }
        }
    }
    return outputList
}

private func readF32Buffer(
    _ buffer: UnsafeMutablePointer<PJRT_Buffer>,
    count: Int
) -> [Float] {
    var result = [Float](repeating: 0, count: count)
    result.withUnsafeMutableBufferPointer { resultPtr in
        var toHostArgs = PJRT_Buffer_ToHostBuffer_Args()
        toHostArgs.struct_size = MemoryLayout<PJRT_Buffer_ToHostBuffer_Args>.size
        toHostArgs.src = buffer
        toHostArgs.dst = UnsafeMutableRawPointer(resultPtr.baseAddress!)
        toHostArgs.dst_size = count * MemoryLayout<Float>.size
        let _ = pjrt_buffer_to_host_buffer(&toHostArgs)
    }
    return result
}

private func destroyBuffer(_ buf: UnsafeMutablePointer<PJRT_Buffer>) {
    var args = PJRT_Buffer_Destroy_Args()
    args.struct_size = MemoryLayout<PJRT_Buffer_Destroy_Args>.size
    args.buffer = buf
    let _ = pjrt_buffer_destroy(&args)
}

private func destroyLoadedExecutable(_ exe: UnsafeMutablePointer<PJRT_LoadedExecutable>) {
    var args = PJRT_LoadedExecutable_Destroy_Args()
    args.struct_size = MemoryLayout<PJRT_LoadedExecutable_Destroy_Args>.size
    args.executable = exe
    let _ = pjrt_loaded_executable_destroy(&args)
}

private func destroyError(_ error: UnsafeMutablePointer<PJRT_Error>) {
    var msgArgs = PJRT_Error_Message_Args()
    msgArgs.struct_size = MemoryLayout<PJRT_Error_Message_Args>.size
    msgArgs.error = UnsafePointer(error)
    pjrt_error_message(&msgArgs)
    if let msg = msgArgs.message {
        let str = String(cString: msg)
        print("PJRT Error: \(str)")
    }
    var args = PJRT_Error_Destroy_Args()
    args.struct_size = MemoryLayout<PJRT_Error_Destroy_Args>.size
    args.error = error
    pjrt_error_destroy(&args)
}

// MARK: - Tests

@Suite("PJRT Plugin")
struct PJRTPluginTests {

    @Test("GetPjrtApi returns valid pointer with correct version")
    func apiVersion() {
        let api = GetPjrtApi()
        #expect(api.pointee.pjrt_api_version.major_version == Int32(PJRT_API_MAJOR))
        #expect(api.pointee.pjrt_api_version.minor_version == Int32(PJRT_API_MINOR))
    }

    @Test("All required functions are populated")
    func requiredFunctions() {
        let api = GetPjrtApi()
        #expect(api.pointee.PJRT_Error_Destroy != nil)
        #expect(api.pointee.PJRT_Error_Message != nil)
        #expect(api.pointee.PJRT_Error_GetCode != nil)
        #expect(api.pointee.PJRT_Client_Create != nil)
        #expect(api.pointee.PJRT_Client_Destroy != nil)
        #expect(api.pointee.PJRT_Client_Compile != nil)
        #expect(api.pointee.PJRT_Client_BufferFromHostBuffer != nil)
        #expect(api.pointee.PJRT_LoadedExecutable_Execute != nil)
        #expect(api.pointee.PJRT_Buffer_ToHostBuffer != nil)
        #expect(api.pointee.PJRT_Buffer_Destroy != nil)
    }

    @Test("Error lifecycle: create, message, code, destroy")
    func errorLifecycle() {
        let errorPtr = makeError(PJRT_Error_Code_INTERNAL, "test error")!

        // Get message
        var msgArgs = PJRT_Error_Message_Args()
        msgArgs.struct_size = MemoryLayout<PJRT_Error_Message_Args>.size
        msgArgs.error = UnsafePointer(errorPtr)
        pjrt_error_message(&msgArgs)
        let message = String(cString: msgArgs.message!)
        #expect(message == "test error")

        // Get code
        var codeArgs = PJRT_Error_GetCode_Args()
        codeArgs.struct_size = MemoryLayout<PJRT_Error_GetCode_Args>.size
        codeArgs.error = UnsafePointer(errorPtr)
        let result = pjrt_error_getcode(&codeArgs)
        #expect(result == nil)
        #expect(codeArgs.code == PJRT_Error_Code_INTERNAL)

        // Destroy
        var destroyArgs = PJRT_Error_Destroy_Args()
        destroyArgs.struct_size = MemoryLayout<PJRT_Error_Destroy_Args>.size
        destroyArgs.error = errorPtr
        pjrt_error_destroy(&destroyArgs)
    }

    @Test("Event is immediately ready when created without pending flag")
    func eventImmediateCompletion() {
        let event = PJRTEventImpl()
        #expect(event.isReady)
        #expect(event.error == nil)
    }

    @Test("Event transitions from pending to completed")
    func eventPendingThenComplete() {
        let event = PJRTEventImpl(pending: true)
        #expect(!event.isReady)
        event.complete()
        #expect(event.isReady)
        #expect(event.error == nil)
    }

    @Test("Event onReady fires after completion")
    func eventOnReady() {
        let event = PJRTEventImpl(pending: true)
        var callbackFired = false
        event.onReady { errorPtr in
            #expect(errorPtr == nil)
            callbackFired = true
        }
        event.complete()
        #expect(callbackFired)
    }

    @Test("Client creation returns valid client with platform name 'metalhlo'")
    func clientCreateAndPlatformName() {
        let clientPtr = createClient()
        #expect(clientPtr != nil)
        guard let clientPtr = clientPtr else { return }

        // Platform name
        var nameArgs = PJRT_Client_PlatformName_Args()
        nameArgs.struct_size = MemoryLayout<PJRT_Client_PlatformName_Args>.size
        nameArgs.client = clientPtr
        let nameError = pjrt_client_platform_name(&nameArgs)
        #expect(nameError == nil)
        let name = String(cString: nameArgs.platform_name!)
        #expect(name == "metalhlo")

        // Devices
        var devArgs = PJRT_Client_Devices_Args()
        devArgs.struct_size = MemoryLayout<PJRT_Client_Devices_Args>.size
        devArgs.client = clientPtr
        let devError = pjrt_client_devices(&devArgs)
        #expect(devError == nil)
        #expect(devArgs.num_devices == 1)

        destroyClient(clientPtr)
    }

    @Test("Buffer host-to-device round-trip preserves data")
    func bufferHostToDeviceToHost() {
        let clientPtr = createClient()!

        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let bufferPtr = createF32Buffer(client: clientPtr, data: data, shape: [4])
        #expect(bufferPtr != nil)
        guard let bufferPtr = bufferPtr else { destroyClient(clientPtr); return }

        // Check element type
        var typeArgs = PJRT_Buffer_ElementType_Args()
        typeArgs.struct_size = MemoryLayout<PJRT_Buffer_ElementType_Args>.size
        typeArgs.buffer = bufferPtr
        let _ = pjrt_buffer_element_type(&typeArgs)
        #expect(typeArgs.type == PJRT_Buffer_Type_F32)

        // Copy back to host
        let result = readF32Buffer(bufferPtr, count: 4)
        #expect(result == [1.0, 2.0, 3.0, 4.0])

        destroyBuffer(bufferPtr)
        destroyClient(clientPtr)
    }

    @Test("Compile and execute stablehlo.add produces correct results")
    func compileAndExecuteAdd() {
        let clientPtr = createClient()!

        let mlir = """
            module @pjrt_add {
              func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
                %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
                return %0 : tensor<4xf32>
              }
            }
            """

        let loadedExePtr = compile(client: clientPtr, mlir: mlir)
        #expect(loadedExePtr != nil)
        guard let loadedExePtr = loadedExePtr else { destroyClient(clientPtr); return }

        let bufA = createF32Buffer(client: clientPtr, data: [1.0, 2.0, 3.0, 4.0], shape: [4])!
        let bufB = createF32Buffer(client: clientPtr, data: [5.0, 6.0, 7.0, 8.0], shape: [4])!

        let outputs = execute(loadedExe: loadedExePtr, inputs: [bufA, bufB], numOutputs: 1)
        #expect(outputs.count == 1)

        let result = readF32Buffer(outputs[0]!, count: 4)
        #expect(result == [6.0, 8.0, 10.0, 12.0])

        for output in outputs { if let o = output { destroyBuffer(o) } }
        destroyBuffer(bufA)
        destroyBuffer(bufB)
        destroyLoadedExecutable(loadedExePtr)
        destroyClient(clientPtr)
    }

    @Test("Executable serialize and deserialize round-trip")
    func executableSerializeAndDeserialize() {
        let clientPtr = createClient()!

        let mlir = """
            module @pjrt_mul {
              func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>) {
                %0 = stablehlo.multiply %arg0, %arg1 : tensor<2xf32>
                return %0 : tensor<2xf32>
              }
            }
            """

        let loadedExe1 = compile(client: clientPtr, mlir: mlir)!

        // Get the base executable for serialization
        var getExeArgs = PJRT_LoadedExecutable_GetExecutable_Args()
        getExeArgs.struct_size = MemoryLayout<PJRT_LoadedExecutable_GetExecutable_Args>.size
        getExeArgs.loaded_executable = loadedExe1
        let _ = pjrt_loaded_executable_get_executable(&getExeArgs)
        let exePtr = getExeArgs.executable!

        // Serialize
        var serArgs = PJRT_Executable_Serialize_Args()
        serArgs.struct_size = MemoryLayout<PJRT_Executable_Serialize_Args>.size
        serArgs.executable = UnsafePointer(exePtr)
        let serError = pjrt_executable_serialize(&serArgs)
        #expect(serError == nil)
        #expect(serArgs.serialized_bytes_size > 0)

        // Deserialize and load
        var deserArgs = PJRT_Executable_DeserializeAndLoad_Args()
        deserArgs.struct_size = MemoryLayout<PJRT_Executable_DeserializeAndLoad_Args>.size
        deserArgs.client = clientPtr
        deserArgs.serialized_executable = serArgs.serialized_bytes
        deserArgs.serialized_executable_size = serArgs.serialized_bytes_size
        let deserError = pjrt_executable_deserialize_and_load(&deserArgs)
        #expect(deserError == nil)
        let loadedExe2 = deserArgs.loaded_executable!

        // Execute the deserialized version
        let bufA = createF32Buffer(client: clientPtr, data: [2.0, 3.0], shape: [2])!
        let bufB = createF32Buffer(client: clientPtr, data: [4.0, 5.0], shape: [2])!
        let outputs = execute(loadedExe: loadedExe2, inputs: [bufA, bufB], numOutputs: 1)
        let result = readF32Buffer(outputs[0]!, count: 2)
        #expect(result == [8.0, 15.0])

        for output in outputs { if let o = output { destroyBuffer(o) } }
        destroyBuffer(bufA)
        destroyBuffer(bufB)

        // serialized_bytes are owned by the executable — no manual free needed

        var exeDestroyArgs = PJRT_Executable_Destroy_Args()
        exeDestroyArgs.struct_size = MemoryLayout<PJRT_Executable_Destroy_Args>.size
        exeDestroyArgs.executable = exePtr
        let _ = pjrt_executable_destroy(&exeDestroyArgs)

        destroyLoadedExecutable(loadedExe1)
        destroyLoadedExecutable(loadedExe2)
        destroyClient(clientPtr)
    }
}
