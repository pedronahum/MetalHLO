// ANEDeviceTests.swift
// ANERuntimeTests

import Testing
@testable import ANERuntime

@Suite("ANE Device", .serialized)
struct ANEDeviceTests {

    @Test("ANEDevice.init returns without crashing on any machine")
    func gracefulInit() {
        let device = ANEDevice()
        if device == nil {
            // Expected on non-ANE machines
        }
    }

    @Test("ANEDevice exposes availability info when available")
    func availabilityInfo() {
        guard let device = ANEDevice() else { return }
        #expect(device.availability.isAvailable)
        #expect(device.availability.chipName != nil)
    }

    @Test("ANEDevice reports hardware info")
    func hardwareInfo() {
        guard let device = ANEDevice() else { return }
        #expect(device.hardwareInfo.hasANE)
        #expect(device.hardwareInfo.coreCount > 0)
    }

    @Test("ANEDevice discovers methods on _ANEClient")
    func methodDiscovery() {
        guard let device = ANEDevice() else { return }
        #expect(!device.availableMethods.isEmpty)
        let allMethods = device.availableMethods.joined(separator: "\n")
        #expect(allMethods.contains("compileModel"))
        #expect(allMethods.contains("loadModel"))
        #expect(allMethods.contains("evaluateWithModel"))
    }

    @Test("ANEDevice can create buffers")
    func createBuffer() throws {
        guard let device = ANEDevice() else { return }
        let desc = ANEBufferDescriptor(shape: [64])
        let buffer = device.createBuffer(descriptor: desc)
        #expect(buffer != nil)
        #expect(buffer?.isValid == true)
    }

    @Test("ANEDevice can create buffers from Float32 data")
    func createBufferFromData() throws {
        guard let device = ANEDevice() else { return }
        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer = try device.createBuffer(data: data, shape: [4])
        #expect(buffer.isValid)
        let readback = try buffer.readFloat32()
        for (a, b) in zip(data, readback) {
            #expect(abs(a - b) < 0.01)
        }
    }

    @Test("Compile counter tracks compilations")
    func compileCounterTracking() throws {
        guard let device = ANEDevice(warningLimit: 2, maximumLimit: 3) else {
            return
        }
        #expect(device.compileCounter.currentCount == 0)
        #expect(device.compileCounter.canCompile)
        #expect(!device.compileCounter.isWarning)
    }

    @Test("Execute rejects released program")
    func executeReleasedProgram() throws {
        guard let device = ANEDevice() else { return }
        let desc = ANEBufferDescriptor(shape: [4])
        let program = ANEProgram(
            id: "test",
            programHandle: nil,
            inputDescriptors: [desc],
            outputDescriptors: [desc],
            compilationNumber: 0
        )
        program.release()
        let buffer = ANEBuffer(descriptor: desc)!
        #expect(throws: ANEError.self) {
            _ = try device.execute(program, inputs: [buffer])
        }
    }

    @Test("Execute rejects wrong input count")
    func executeWrongInputCount() throws {
        guard let device = ANEDevice() else { return }
        let desc = ANEBufferDescriptor(shape: [4])
        let program = ANEProgram(
            id: "test",
            programHandle: "dummy" as AnyObject,
            inputDescriptors: [desc, desc],
            outputDescriptors: [desc],
            compilationNumber: 0
        )
        let buffer = ANEBuffer(descriptor: desc)!
        #expect(throws: ANEError.self) {
            _ = try device.execute(program, inputs: [buffer])
        }
    }

    @Test("Execute rejects nil program handle")
    func executeNilHandle() throws {
        guard let device = ANEDevice() else { return }
        let desc = ANEBufferDescriptor(shape: [4])
        let program = ANEProgram(
            id: "test",
            programHandle: nil,
            inputDescriptors: [desc],
            outputDescriptors: [desc],
            compilationNumber: 0
        )
        let buffer = ANEBuffer(descriptor: desc)!
        #expect(throws: ANEError.self) {
            _ = try device.execute(program, inputs: [buffer])
        }
    }

    @Test("Unload handles nil program handle gracefully")
    func unloadNilHandle() {
        guard let device = ANEDevice() else { return }
        let desc = ANEBufferDescriptor(shape: [4])
        let program = ANEProgram(
            id: "test",
            programHandle: nil,
            inputDescriptors: [desc],
            outputDescriptors: [desc],
            compilationNumber: 0
        )
        device.unload(program)
        #expect(!program.isValid)
    }
}
