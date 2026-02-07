// main.swift
// PJRTExample
//
// Example demonstrating MetalHLO usage through the PJRT plugin interface.
// This mirrors how JAX/XLA interacts with MetalHLO: load the plugin,
// get the API vtable, and call functions through it.

import PJRTMetalHLO
import CPJRTApi
import Darwin

// MARK: - Helpers

@MainActor
func check(_ error: UnsafeMutablePointer<PJRT_Error>?, api: UnsafePointer<PJRT_Api>) {
    guard let error = error else { return }
    var msgArgs = PJRT_Error_Message_Args()
    msgArgs.struct_size = MemoryLayout<PJRT_Error_Message_Args>.size
    msgArgs.error = UnsafePointer(error)
    api.pointee.PJRT_Error_Message!(&msgArgs)
    let message = String(cString: msgArgs.message!)
    print("PJRT Error: \(message)")

    var destroyArgs = PJRT_Error_Destroy_Args()
    destroyArgs.struct_size = MemoryLayout<PJRT_Error_Destroy_Args>.size
    destroyArgs.error = error
    api.pointee.PJRT_Error_Destroy!(&destroyArgs)

    fatalError("PJRT call failed: \(message)")
}

// MARK: - Main

print("MetalHLO PJRT Plugin Example")
print("============================")
print()

// Step 1: Get the PJRT API vtable
// In JAX, this is done via dlsym(handle, "GetPjrtApi") after loading the dylib.
// Here we call it directly since we linked against PJRTMetalHLO.
let api = GetPjrtApi()
print("PJRT API version: \(api.pointee.pjrt_api_version.major_version).\(api.pointee.pjrt_api_version.minor_version)")

// Step 2: Create a client (represents a connection to the Metal GPU)
var createArgs = PJRT_Client_Create_Args()
createArgs.struct_size = MemoryLayout<PJRT_Client_Create_Args>.size
check(api.pointee.PJRT_Client_Create!(&createArgs), api: api)
let client = createArgs.client!

// Query platform name
var nameArgs = PJRT_Client_PlatformName_Args()
nameArgs.struct_size = MemoryLayout<PJRT_Client_PlatformName_Args>.size
nameArgs.client = client
check(api.pointee.PJRT_Client_PlatformName!(&nameArgs), api: api)
print("Platform: \(String(cString: nameArgs.platform_name!))")

// Query devices
var devArgs = PJRT_Client_Devices_Args()
devArgs.struct_size = MemoryLayout<PJRT_Client_Devices_Args>.size
devArgs.client = client
check(api.pointee.PJRT_Client_Devices!(&devArgs), api: api)
print("Devices: \(devArgs.num_devices)")
print()

// Step 3: Compile a StableHLO program
// This is the same MLIR that JAX's XLA compiler would emit.
print("Example 1: Vector Addition")
print("--------------------------")

let addMLIR = """
    module @vector_add {
      func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
        %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
    """

let cString = addMLIR.utf8CString
var program = PJRT_Program()
program.struct_size = MemoryLayout<PJRT_Program>.size
program.code_size = addMLIR.utf8.count

var compileArgs = PJRT_Client_Compile_Args()
compileArgs.struct_size = MemoryLayout<PJRT_Client_Compile_Args>.size
compileArgs.client = client

cString.withUnsafeBufferPointer { buf in
    program.code = UnsafeMutablePointer(mutating: buf.baseAddress)
    withUnsafeMutablePointer(to: &program) { progPtr in
        compileArgs.program = UnsafePointer(progPtr)
        check(api.pointee.PJRT_Client_Compile!(&compileArgs), api: api)
    }
}
let addExe = compileArgs.executable!
print("Compiled 'vector_add' program")

// Step 4: Create input buffers (host → device transfer)
let dataA: [Float] = [1.0, 2.0, 3.0, 4.0]
let dataB: [Float] = [10.0, 20.0, 30.0, 40.0]
let shape: [Int64] = [4]

@MainActor
func createBuffer(_ data: [Float], shape: [Int64]) -> UnsafeMutablePointer<PJRT_Buffer> {
    var bufArgs = PJRT_Client_BufferFromHostBuffer_Args()
    bufArgs.struct_size = MemoryLayout<PJRT_Client_BufferFromHostBuffer_Args>.size
    bufArgs.client = client
    bufArgs.type = PJRT_Buffer_Type_F32
    bufArgs.num_dims = shape.count
    data.withUnsafeBufferPointer { dataPtr in
        shape.withUnsafeBufferPointer { shapePtr in
            bufArgs.data = UnsafeRawPointer(dataPtr.baseAddress)
            bufArgs.dims = shapePtr.baseAddress
            check(api.pointee.PJRT_Client_BufferFromHostBuffer!(&bufArgs), api: api)
        }
    }
    return bufArgs.buffer!
}

let bufA = createBuffer(dataA, shape: shape)
let bufB = createBuffer(dataB, shape: shape)

// Step 5: Execute the compiled program
@MainActor
func execute(
    _ loadedExe: UnsafeMutablePointer<PJRT_LoadedExecutable>,
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
            var argBase = UnsafePointer<UnsafeMutablePointer<PJRT_Buffer>?>(argBuf.baseAddress)
            var outBase = outBuf.baseAddress
            withUnsafeMutablePointer(to: &options) { optPtr in
                executeArgs.options = optPtr
                withUnsafePointer(to: &argBase) { ap in
                    executeArgs.argument_lists = UnsafePointer(ap)
                    withUnsafePointer(to: &outBase) { op in
                        executeArgs.output_lists = UnsafePointer(op)
                        check(api.pointee.PJRT_LoadedExecutable_Execute!(&executeArgs), api: api)
                    }
                }
            }
        }
    }
    return outputList
}

let addOutputs = execute(addExe, inputs: [bufA, bufB], numOutputs: 1)

// Step 6: Read results back to host (device → host transfer)
@MainActor
func readBuffer(_ buffer: UnsafeMutablePointer<PJRT_Buffer>, count: Int) -> [Float] {
    var result = [Float](repeating: 0, count: count)
    result.withUnsafeMutableBufferPointer { resultPtr in
        var toHostArgs = PJRT_Buffer_ToHostBuffer_Args()
        toHostArgs.struct_size = MemoryLayout<PJRT_Buffer_ToHostBuffer_Args>.size
        toHostArgs.src = buffer
        toHostArgs.dst = UnsafeMutableRawPointer(resultPtr.baseAddress!)
        toHostArgs.dst_size = count * MemoryLayout<Float>.size
        check(api.pointee.PJRT_Buffer_ToHostBuffer!(&toHostArgs), api: api)
    }
    return result
}

let addResult = readBuffer(addOutputs[0]!, count: 4)
print("  Input A: \(dataA)")
print("  Input B: \(dataB)")
print("  A + B:   \(addResult)")
print()

// Example 2: Matrix Multiply with Activation (MLP Layer)
print("Example 2: MLP Layer (matmul + bias + relu)")
print("---------------------------------------------")

let mlpMLIR = """
    module @mlp_layer {
      func.func @main(%x: tensor<2x3xf32>, %w: tensor<3x2xf32>, %b: tensor<2xf32>) -> (tensor<2x2xf32>) {
        %0 = stablehlo.dot %x, %w : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
        %1 = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
        %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
        %zero = stablehlo.constant dense<0.0> : tensor<2x2xf32>
        %3 = stablehlo.maximum %2, %zero : tensor<2x2xf32>
        return %3 : tensor<2x2xf32>
      }
    }
    """

let mlpCString = mlpMLIR.utf8CString
var mlpProgram = PJRT_Program()
mlpProgram.struct_size = MemoryLayout<PJRT_Program>.size
mlpProgram.code_size = mlpMLIR.utf8.count

var mlpCompileArgs = PJRT_Client_Compile_Args()
mlpCompileArgs.struct_size = MemoryLayout<PJRT_Client_Compile_Args>.size
mlpCompileArgs.client = client

mlpCString.withUnsafeBufferPointer { buf in
    mlpProgram.code = UnsafeMutablePointer(mutating: buf.baseAddress)
    withUnsafeMutablePointer(to: &mlpProgram) { progPtr in
        mlpCompileArgs.program = UnsafePointer(progPtr)
        check(api.pointee.PJRT_Client_Compile!(&mlpCompileArgs), api: api)
    }
}
let mlpExe = mlpCompileArgs.executable!
print("Compiled 'mlp_layer' program")

// x: 2x3 input, w: 3x2 weights, b: 2 bias
let xData: [Float] = [1, 2, 3, 4, 5, 6]
let wData: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
let bData: [Float] = [-1.0, 0.5]

let bufX = createBuffer(xData, shape: [2, 3])
let bufW = createBuffer(wData, shape: [3, 2])
let bufBias = createBuffer(bData, shape: [2])

let mlpOutputs = execute(mlpExe, inputs: [bufX, bufW, bufBias], numOutputs: 1)
let mlpResult = readBuffer(mlpOutputs[0]!, count: 4)

print("  x (2x3): \(xData)")
print("  w (3x2): \(wData)")
print("  b (2):   \(bData)")
print("  relu(x @ w + b) = \(mlpResult)")
print()

// Example 3: Serialize and Deserialize
print("Example 3: Executable Serialization Round-Trip")
print("-----------------------------------------------")

// Get the base executable for serialization
var getExeArgs = PJRT_LoadedExecutable_GetExecutable_Args()
getExeArgs.struct_size = MemoryLayout<PJRT_LoadedExecutable_GetExecutable_Args>.size
getExeArgs.loaded_executable = addExe
check(api.pointee.PJRT_LoadedExecutable_GetExecutable!(&getExeArgs), api: api)
let baseExe = getExeArgs.executable!

// Serialize
var serArgs = PJRT_Executable_Serialize_Args()
serArgs.struct_size = MemoryLayout<PJRT_Executable_Serialize_Args>.size
serArgs.executable = UnsafePointer(baseExe)
check(api.pointee.PJRT_Executable_Serialize!(&serArgs), api: api)
print("  Serialized executable: \(serArgs.serialized_bytes_size) bytes")

// Deserialize into a new loaded executable
var deserArgs = PJRT_Executable_DeserializeAndLoad_Args()
deserArgs.struct_size = MemoryLayout<PJRT_Executable_DeserializeAndLoad_Args>.size
deserArgs.client = client
deserArgs.serialized_executable = serArgs.serialized_bytes
deserArgs.serialized_executable_size = serArgs.serialized_bytes_size
check(api.pointee.PJRT_Executable_DeserializeAndLoad!(&deserArgs), api: api)
let restoredExe = deserArgs.loaded_executable!

// Execute the deserialized program with new inputs
let dataC: [Float] = [100.0, 200.0, 300.0, 400.0]
let dataD: [Float] = [0.5, 0.5, 0.5, 0.5]
let bufC = createBuffer(dataC, shape: shape)
let bufD = createBuffer(dataD, shape: shape)

let restoredOutputs = execute(restoredExe, inputs: [bufC, bufD], numOutputs: 1)
let restoredResult = readBuffer(restoredOutputs[0]!, count: 4)
print("  Deserialized and executed: \(dataC) + \(dataD) = \(restoredResult)")
print()

// Cleanup
@MainActor
func destroyBuffer(_ buf: UnsafeMutablePointer<PJRT_Buffer>) {
    var args = PJRT_Buffer_Destroy_Args()
    args.struct_size = MemoryLayout<PJRT_Buffer_Destroy_Args>.size
    args.buffer = buf
    let _ = api.pointee.PJRT_Buffer_Destroy!(&args)
}

@MainActor
func destroyLoadedExecutable(_ exe: UnsafeMutablePointer<PJRT_LoadedExecutable>) {
    var args = PJRT_LoadedExecutable_Destroy_Args()
    args.struct_size = MemoryLayout<PJRT_LoadedExecutable_Destroy_Args>.size
    args.executable = exe
    let _ = api.pointee.PJRT_LoadedExecutable_Destroy!(&args)
}

for output in addOutputs { if let o = output { destroyBuffer(o) } }
for output in mlpOutputs { if let o = output { destroyBuffer(o) } }
for output in restoredOutputs { if let o = output { destroyBuffer(o) } }
destroyBuffer(bufA)
destroyBuffer(bufB)
destroyBuffer(bufX)
destroyBuffer(bufW)
destroyBuffer(bufBias)
destroyBuffer(bufC)
destroyBuffer(bufD)

var exeDestroyArgs = PJRT_Executable_Destroy_Args()
exeDestroyArgs.struct_size = MemoryLayout<PJRT_Executable_Destroy_Args>.size
exeDestroyArgs.executable = baseExe
let _ = api.pointee.PJRT_Executable_Destroy!(&exeDestroyArgs)

destroyLoadedExecutable(addExe)
destroyLoadedExecutable(mlpExe)
destroyLoadedExecutable(restoredExe)

var clientDestroyArgs = PJRT_Client_Destroy_Args()
clientDestroyArgs.struct_size = MemoryLayout<PJRT_Client_Destroy_Args>.size
clientDestroyArgs.client = client
let _ = api.pointee.PJRT_Client_Destroy!(&clientDestroyArgs)

print("All examples completed successfully!")
print("All resources cleaned up.")
