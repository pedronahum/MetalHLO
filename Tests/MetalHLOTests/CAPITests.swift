// CAPITests.swift
// MetalHLOTests
//
// Tests for the C API.

import Testing
import Foundation
@testable import CMetalHLO

// MARK: - C API Basic Tests

@Suite("C API")
struct CAPITests {

    @Test("Version returns valid string")
    func testVersion() {
        let version = mhlo_version()
        #expect(version != nil)
        if let version = version {
            let str = String(cString: version)
            #expect(str == "0.1.0")
        }
    }

    @Test("Element type sizes are correct")
    func testElementTypeSizes() {
        #expect(mhlo_element_type_size(0) == 2)  // F16
        #expect(mhlo_element_type_size(1) == 4)  // F32
        #expect(mhlo_element_type_size(2) == 8)  // F64
        #expect(mhlo_element_type_size(3) == 2)  // BF16
        #expect(mhlo_element_type_size(4) == 0)  // I1 (bool)
        #expect(mhlo_element_type_size(5) == 1)  // I8
        #expect(mhlo_element_type_size(6) == 2)  // I16
        #expect(mhlo_element_type_size(7) == 4)  // I32
        #expect(mhlo_element_type_size(8) == 8)  // I64
    }

    @Test("Client creation and destruction")
    func testClientCreateDestroy() {
        var client: OpaquePointer?
        let status = mhlo_client_create(&client)
        #expect(status == 0) // MHLO_OK
        #expect(client != nil)

        if let client = client {
            let deviceName = mhlo_client_device_name(client)
            #expect(deviceName != nil)
            if let deviceName = deviceName {
                let name = String(cString: deviceName)
                #expect(!name.isEmpty)
                mhlo_free_string(deviceName)
            }

            mhlo_client_destroy(client)
        }
    }

    @Test("Compile simple module")
    func testCompile() throws {
        var client: OpaquePointer?
        var status = mhlo_client_create(&client)
        #expect(status == 0)
        #expect(client != nil)

        let mlir = """
        module @simple_add {
          func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.add %a, %b : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        var executable: OpaquePointer?
        status = mhlo_compile(client, mlir, &executable)
        #expect(status == 0)
        #expect(executable != nil)

        if let executable = executable {
            let inputCount = mhlo_executable_input_count(executable)
            let outputCount = mhlo_executable_output_count(executable)
            #expect(inputCount == 2)
            #expect(outputCount == 1)

            mhlo_executable_destroy(executable)
        }

        mhlo_client_destroy(client)
    }

    @Test("Create and read buffer")
    func testBufferCreateRead() throws {
        var client: OpaquePointer?
        var status = mhlo_client_create(&client)
        #expect(status == 0)
        #expect(client != nil)

        // Create buffer with data
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let shape: [Int64] = [4]
        var buffer: OpaquePointer?

        status = data.withUnsafeBytes { dataPtr in
            shape.withUnsafeBufferPointer { shapePtr in
                mhlo_buffer_create(
                    client,
                    dataPtr.baseAddress,
                    dataPtr.count,
                    shapePtr.baseAddress,
                    Int32(shape.count),
                    1, // MHLO_F32
                    &buffer
                )
            }
        }

        #expect(status == 0)
        #expect(buffer != nil)

        if let buffer = buffer {
            // Check buffer properties
            #expect(mhlo_buffer_rank(buffer) == 1)
            #expect(mhlo_buffer_element_type(buffer) == 1) // F32
            #expect(mhlo_buffer_byte_count(buffer) == 16) // 4 floats * 4 bytes

            // Check shape
            var outShape: [Int64] = [0]
            var outRank: Int32 = 0
            mhlo_buffer_shape(buffer, &outShape, &outRank)
            #expect(outRank == 1)
            #expect(outShape[0] == 4)

            // Read data back
            var result = [Float](repeating: 0, count: 4)
            status = result.withUnsafeMutableBytes { resultPtr in
                mhlo_buffer_to_host(buffer, resultPtr.baseAddress, resultPtr.count)
            }
            #expect(status == 0)
            #expect(result == [1.0, 2.0, 3.0, 4.0])

            mhlo_buffer_destroy(buffer)
        }

        mhlo_client_destroy(client)
    }

    @Test("Execute simple add")
    func testExecuteSimpleAdd() throws {
        var client: OpaquePointer?
        var status = mhlo_client_create(&client)
        #expect(status == 0)

        // Compile
        let mlir = """
        module @add {
          func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.add %a, %b : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        var executable: OpaquePointer?
        status = mhlo_compile(client, mlir, &executable)
        #expect(status == 0)

        // Create input buffers
        let aData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let bData: [Float] = [5.0, 6.0, 7.0, 8.0]
        let shape: [Int64] = [4]

        var bufferA: OpaquePointer?
        var bufferB: OpaquePointer?

        status = aData.withUnsafeBytes { dataPtr in
            shape.withUnsafeBufferPointer { shapePtr in
                mhlo_buffer_create(client, dataPtr.baseAddress, dataPtr.count, shapePtr.baseAddress, 1, 1, &bufferA)
            }
        }
        #expect(status == 0)

        status = bData.withUnsafeBytes { dataPtr in
            shape.withUnsafeBufferPointer { shapePtr in
                mhlo_buffer_create(client, dataPtr.baseAddress, dataPtr.count, shapePtr.baseAddress, 1, 1, &bufferB)
            }
        }
        #expect(status == 0)

        // Execute
        var inputs: [OpaquePointer?] = [bufferA, bufferB]
        var outputs: [OpaquePointer?] = [nil]
        var numOutputs: Int32 = 0

        status = inputs.withUnsafeBufferPointer { inputsPtr in
            outputs.withUnsafeMutableBufferPointer { outputsPtr in
                mhlo_execute(executable, inputsPtr.baseAddress, 2, outputsPtr.baseAddress, &numOutputs)
            }
        }
        #expect(status == 0)
        #expect(numOutputs == 1)

        // Read output
        if let output = outputs[0] {
            var result = [Float](repeating: 0, count: 4)
            status = result.withUnsafeMutableBytes { resultPtr in
                mhlo_buffer_to_host(output, resultPtr.baseAddress, resultPtr.count)
            }
            #expect(status == 0)
            #expect(result == [6.0, 8.0, 10.0, 12.0])

            mhlo_buffer_destroy(output)
        }

        // Cleanup
        mhlo_buffer_destroy(bufferA)
        mhlo_buffer_destroy(bufferB)
        mhlo_executable_destroy(executable)
        mhlo_client_destroy(client)
    }

    @Test("Execute with timing")
    func testExecuteWithTiming() throws {
        var client: OpaquePointer?
        var status = mhlo_client_create(&client)
        #expect(status == 0)

        // Compile
        let mlir = """
        module @matmul {
          func.func @main(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> (tensor<32x32xf32>) {
            %0 = stablehlo.dot %a, %b : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
            return %0 : tensor<32x32xf32>
          }
        }
        """

        var executable: OpaquePointer?
        status = mhlo_compile(client, mlir, &executable)
        #expect(status == 0)

        // Create input buffers
        let size = 32 * 32
        let aData = [Float](repeating: 1.0, count: size)
        let bData = [Float](repeating: 1.0, count: size)
        let shape: [Int64] = [32, 32]

        var bufferA: OpaquePointer?
        var bufferB: OpaquePointer?

        status = aData.withUnsafeBytes { dataPtr in
            shape.withUnsafeBufferPointer { shapePtr in
                mhlo_buffer_create(client, dataPtr.baseAddress, dataPtr.count, shapePtr.baseAddress, 2, 1, &bufferA)
            }
        }
        #expect(status == 0)

        status = bData.withUnsafeBytes { dataPtr in
            shape.withUnsafeBufferPointer { shapePtr in
                mhlo_buffer_create(client, dataPtr.baseAddress, dataPtr.count, shapePtr.baseAddress, 2, 1, &bufferB)
            }
        }
        #expect(status == 0)

        // Execute with timing
        var inputs: [OpaquePointer?] = [bufferA, bufferB]
        var outputs: [OpaquePointer?] = [nil]
        var numOutputs: Int32 = 0
        var encodeTime: Double = 0
        var gpuTime: Double = 0
        var totalTime: Double = 0

        status = inputs.withUnsafeBufferPointer { inputsPtr in
            outputs.withUnsafeMutableBufferPointer { outputsPtr in
                mhlo_execute_with_timing(
                    executable,
                    inputsPtr.baseAddress,
                    2,
                    outputsPtr.baseAddress,
                    &numOutputs,
                    &encodeTime,
                    &gpuTime,
                    &totalTime
                )
            }
        }
        #expect(status == 0)
        #expect(numOutputs == 1)

        // Timing values should be non-negative
        #expect(encodeTime >= 0)
        #expect(gpuTime >= 0)
        #expect(totalTime >= 0)

        // Cleanup
        if let output = outputs[0] {
            mhlo_buffer_destroy(output)
        }
        mhlo_buffer_destroy(bufferA)
        mhlo_buffer_destroy(bufferB)
        mhlo_executable_destroy(executable)
        mhlo_client_destroy(client)
    }

    @Test("Error handling")
    func testErrorHandling() {
        // Try to compile invalid MLIR
        var client: OpaquePointer?
        var status = mhlo_client_create(&client)
        #expect(status == 0)

        var executable: OpaquePointer?
        status = mhlo_compile(client, "invalid mlir text", &executable)
        #expect(status != 0) // Should be an error
        #expect(executable == nil)

        // Check error message is available
        let errorMsg = mhlo_get_last_error()
        #expect(errorMsg != nil)
        if let errorMsg = errorMsg {
            let msg = String(cString: errorMsg)
            #expect(!msg.isEmpty)
        }

        mhlo_clear_error()
        #expect(mhlo_get_last_error() == nil)

        mhlo_client_destroy(client)
    }

    @Test("Get tensor types from executable")
    func testGetTensorTypes() throws {
        var client: OpaquePointer?
        var status = mhlo_client_create(&client)
        #expect(status == 0)

        let mlir = """
        module @test {
          func.func @main(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> (tensor<2x4xf32>) {
            %0 = stablehlo.dot %a, %b : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
            return %0 : tensor<2x4xf32>
          }
        }
        """

        var executable: OpaquePointer?
        status = mhlo_compile(client, mlir, &executable)
        #expect(status == 0)

        if let executable = executable {
            // Check input types
            var shape = [Int64](repeating: 0, count: 8)
            var rank: Int32 = 0
            var elementType: Int32 = 0

            // Input 0: tensor<2x3xf32>
            status = mhlo_executable_input_type(executable, 0, &shape, &rank, &elementType)
            #expect(status == 0)
            #expect(rank == 2)
            #expect(shape[0] == 2)
            #expect(shape[1] == 3)
            #expect(elementType == 1) // F32

            // Input 1: tensor<3x4xf32>
            status = mhlo_executable_input_type(executable, 1, &shape, &rank, &elementType)
            #expect(status == 0)
            #expect(rank == 2)
            #expect(shape[0] == 3)
            #expect(shape[1] == 4)
            #expect(elementType == 1) // F32

            // Output 0: tensor<2x4xf32>
            status = mhlo_executable_output_type(executable, 0, &shape, &rank, &elementType)
            #expect(status == 0)
            #expect(rank == 2)
            #expect(shape[0] == 2)
            #expect(shape[1] == 4)
            #expect(elementType == 1) // F32

            mhlo_executable_destroy(executable)
        }

        mhlo_client_destroy(client)
    }
}
