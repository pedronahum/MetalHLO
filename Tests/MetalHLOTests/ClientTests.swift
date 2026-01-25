// ClientTests.swift
// MetalHLOTests
//
// Tests for the MetalHLO Client.

import Testing
@testable import MetalHLO

@Suite("Client Tests")
struct ClientTests {

    // MARK: - Client Creation

    @Test("Client creation succeeds")
    func clientCreation() throws {
        let client = try Client.create()
        #expect(!client.deviceName.isEmpty)
    }

    @Test("Client device name indicates Apple Silicon")
    func clientDeviceName() throws {
        let client = try Client.create()
        // Should contain "Apple" for Apple Silicon
        let isAppleSilicon = client.deviceName.contains("Apple") ||
            client.deviceName.contains("M1") ||
            client.deviceName.contains("M2") ||
            client.deviceName.contains("M3") ||
            client.deviceName.contains("M4")
        #expect(isAppleSilicon, "Device name should indicate Apple Silicon: \(client.deviceName)")
    }

    // MARK: - Buffer Creation

    @Test("Create buffer from Float array")
    func createBufferFromFloatArray() throws {
        let client = try Client.create()
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let buffer = try client.createBuffer(data, shape: [2, 3], elementType: .float32)

        #expect(buffer.shape == [2, 3])
        #expect(buffer.count == 6)
        #expect(buffer.elementType == .float32)
    }

    @Test("Create buffer from Int32 array")
    func createBufferFromIntArray() throws {
        let client = try Client.create()
        let data: [Int32] = [1, 2, 3, 4]
        let buffer = try client.createBuffer(data, shape: [4], elementType: .int32)

        #expect(buffer.shape == [4])
        #expect(buffer.count == 4)
        #expect(buffer.elementType == .int32)
    }

    @Test("Create uninitialized buffer")
    func createUninitializedBuffer() throws {
        let client = try Client.create()
        let buffer = try client.createBuffer(shape: [2, 3], elementType: .float32)

        #expect(buffer.shape == [2, 3])
        #expect(buffer.count == 6)
        #expect(buffer.elementType == .float32)
    }

    // MARK: - Buffer Data Transfer

    @Test("Float buffer round trip")
    func bufferRoundTrip() throws {
        let client = try Client.create()
        let original: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let buffer = try client.createBuffer(original, shape: [2, 3], elementType: .float32)
        let result = try buffer.toFloatArray()

        #expect(result == original)
    }

    @Test("Int32 buffer round trip")
    func bufferInt32RoundTrip() throws {
        let client = try Client.create()
        let original: [Int32] = [1, -2, 3, -4]
        let buffer = try client.createBuffer(original, shape: [4], elementType: .int32)
        let result = try buffer.toInt32Array()

        #expect(result == original)
    }
}
