// CoreMLBridgeTests.swift
// ANERuntimeTests
//
// Tests for the CoreML compilation bridge.
// Verifies protobuf encoding, model building, compilation, and execution.

import Testing
import Foundation
@testable import ANERuntime

@Suite("CoreML Bridge", .serialized)
struct CoreMLBridgeTests {

    // MARK: - Protobuf Writer

    @Test("Varint encoding produces correct bytes")
    func varintEncoding() {
        let writer = ProtobufWriter()

        // 0 -> 0x00
        writer.writeRawVarint(0)
        #expect(writer.data == Data([0x00]))

        let writer2 = ProtobufWriter()
        // 1 -> 0x01
        writer2.writeRawVarint(1)
        #expect(writer2.data == Data([0x01]))

        let writer3 = ProtobufWriter()
        // 150 -> 0x96 0x01
        writer3.writeRawVarint(150)
        #expect(writer3.data == Data([0x96, 0x01]))

        let writer4 = ProtobufWriter()
        // 300 -> 0xAC 0x02
        writer4.writeRawVarint(300)
        #expect(writer4.data == Data([0xAC, 0x02]))
    }

    @Test("String field encoding")
    func stringFieldEncoding() {
        let writer = ProtobufWriter()
        writer.writeString(1, value: "hello")
        // Field tag: (1 << 3) | 2 = 0x0A
        // Length: 5 = 0x05
        // "hello" = 0x68 0x65 0x6C 0x6C 0x6F
        #expect(writer.data == Data([0x0A, 0x05, 0x68, 0x65, 0x6C, 0x6C, 0x6F]))
    }

    @Test("Nested message encoding")
    func nestedMessageEncoding() {
        let writer = ProtobufWriter()
        writer.writeMessage(2) { sub in
            sub.writeInt32(1, value: 42)
        }
        // Outer tag: (2 << 3) | 2 = 0x12
        // Inner: tag (1 << 3)|0 = 0x08, value 42 = 0x2A
        // Inner length: 2
        #expect(writer.data == Data([0x12, 0x02, 0x08, 0x2A]))
    }

    // MARK: - Model Building

    @Test("Build simple add model protobuf")
    func buildAddModel() {
        let modelData = CoreMLModelBuilder.build(
            inputs: [("x", [4]), ("y", [4])],
            operations: [
                CoreMLOp(
                    result: "out",
                    op: "add",
                    shape: [4],
                    params: [("x", .variable("x")), ("y", .variable("y"))]
                ),
            ],
            returnVar: "out"
        )

        // Should produce non-empty protobuf data
        #expect(modelData.count > 0)
        // Should start with field 1 (specificationVersion) varint
        #expect(modelData[0] == 0x08) // (1 << 3) | 0
    }

    // MARK: - End-to-End Compilation and Execution

    @Test("Compile and execute add operation")
    func compileAndExecuteAdd() throws {
        let bridge = CoreMLBridge()
        defer { bridge.clearCache() }

        let program = try bridge.compile(
            inputs: [("x", [4]), ("y", [4])],
            operations: [
                CoreMLOp(
                    result: "out",
                    op: "add",
                    shape: [4],
                    params: [("x", .variable("x")), ("y", .variable("y"))]
                ),
            ],
            returnVar: "out"
        )

        #expect(program.isValid)

        let result = try bridge.execute(program, inputs: [
            (name: "x", data: [1, 2, 3, 4], shape: [4]),
            (name: "y", data: [5, 6, 7, 8], shape: [4]),
        ])

        #expect(result.count == 4)
        let expected: [Float] = [6, 8, 10, 12]
        for (a, b) in zip(result, expected) {
            #expect(abs(a - b) < 0.1, "Expected \(b), got \(a)")
        }
    }

    @Test("Compile and execute multiply operation")
    func compileAndExecuteMultiply() throws {
        let bridge = CoreMLBridge()
        defer { bridge.clearCache() }

        let program = try bridge.compile(
            inputs: [("x", [4]), ("y", [4])],
            operations: [
                CoreMLOp(
                    result: "out",
                    op: "mul",
                    shape: [4],
                    params: [("x", .variable("x")), ("y", .variable("y"))]
                ),
            ],
            returnVar: "out"
        )

        let result = try bridge.execute(program, inputs: [
            (name: "x", data: [2, 3, 4, 5], shape: [4]),
            (name: "y", data: [10, 20, 30, 40], shape: [4]),
        ])

        let expected: [Float] = [20, 60, 120, 200]
        for (a, b) in zip(result, expected) {
            #expect(abs(a - b) < 1.0, "Expected \(b), got \(a)")
        }
    }

    @Test("Compile and execute chain: multiply then add")
    func compileAndExecuteChain() throws {
        let bridge = CoreMLBridge()
        defer { bridge.clearCache() }

        // y = x0 * x1 + x2
        let program = try bridge.compile(
            inputs: [("x0", [4]), ("x1", [4]), ("x2", [4])],
            operations: [
                CoreMLOp(
                    result: "t",
                    op: "mul",
                    shape: [4],
                    params: [("x", .variable("x0")), ("y", .variable("x1"))]
                ),
                CoreMLOp(
                    result: "out",
                    op: "add",
                    shape: [4],
                    params: [("x", .variable("t")), ("y", .variable("x2"))]
                ),
            ],
            returnVar: "out"
        )

        let result = try bridge.execute(program, inputs: [
            (name: "x0", data: [2, 3, 4, 5], shape: [4]),
            (name: "x1", data: [10, 10, 10, 10], shape: [4]),
            (name: "x2", data: [1, 1, 1, 1], shape: [4]),
        ])

        let expected: [Float] = [21, 31, 41, 51]
        for (a, b) in zip(result, expected) {
            #expect(abs(a - b) < 1.0, "Expected \(b), got \(a)")
        }
    }

    @Test("Compile and execute sigmoid")
    func compileAndExecuteSigmoid() throws {
        let bridge = CoreMLBridge()
        defer { bridge.clearCache() }

        let program = try bridge.compile(
            inputs: [("x", [4])],
            operations: [
                CoreMLOp(
                    result: "out",
                    op: "sigmoid",
                    shape: [4],
                    params: [("x", .variable("x"))]
                ),
            ],
            returnVar: "out"
        )

        let result = try bridge.execute(program, inputs: [
            (name: "x", data: [0, 1, -1, 10], shape: [4]),
        ])

        let expected: [Float] = [0.5, 0.731, 0.269, 1.0]
        for (a, b) in zip(result, expected) {
            #expect(abs(a - b) < 0.05, "Expected \(b), got \(a)")
        }
    }

    @Test("Compile and execute exp")
    func compileAndExecuteExp() throws {
        let bridge = CoreMLBridge()
        defer { bridge.clearCache() }

        let program = try bridge.compile(
            inputs: [("x", [4])],
            operations: [
                CoreMLOp(
                    result: "out",
                    op: "exp",
                    shape: [4],
                    params: [("x", .variable("x"))]
                ),
            ],
            returnVar: "out"
        )

        let result = try bridge.execute(program, inputs: [
            (name: "x", data: [0, 1, -1, 2], shape: [4]),
        ])

        let expected: [Float] = [1.0, 2.718, 0.368, 7.389]
        for (a, b) in zip(result, expected) {
            #expect(abs(a - b) < 0.05, "Expected \(b), got \(a)")
        }
    }

    @Test("Multiple compilations reuse bridge")
    func multipleCompilations() throws {
        let bridge = CoreMLBridge()
        defer { bridge.clearCache() }

        // Compile two different programs
        let prog1 = try bridge.compile(
            inputs: [("x", [2]), ("y", [2])],
            operations: [
                CoreMLOp(result: "out", op: "add", shape: [2],
                         params: [("x", .variable("x")), ("y", .variable("y"))]),
            ],
            returnVar: "out"
        )

        let prog2 = try bridge.compile(
            inputs: [("x", [2]), ("y", [2])],
            operations: [
                CoreMLOp(result: "out", op: "mul", shape: [2],
                         params: [("x", .variable("x")), ("y", .variable("y"))]),
            ],
            returnVar: "out"
        )

        // Execute both
        let r1 = try bridge.execute(prog1, inputs: [
            (name: "x", data: [1, 2], shape: [2]),
            (name: "y", data: [3, 4], shape: [2]),
        ])
        let r2 = try bridge.execute(prog2, inputs: [
            (name: "x", data: [1, 2], shape: [2]),
            (name: "y", data: [3, 4], shape: [2]),
        ])

        #expect(abs(r1[0] - 4) < 0.1) // 1+3
        #expect(abs(r1[1] - 6) < 0.1) // 2+4
        #expect(abs(r2[0] - 3) < 0.1) // 1*3
        #expect(abs(r2[1] - 8) < 0.1) // 2*4
    }

    @Test("Released program throws on execute")
    func releasedProgramThrows() throws {
        let bridge = CoreMLBridge()
        defer { bridge.clearCache() }

        let program = try bridge.compile(
            inputs: [("x", [2])],
            operations: [
                CoreMLOp(result: "out", op: "sigmoid", shape: [2],
                         params: [("x", .variable("x"))]),
            ],
            returnVar: "out"
        )

        program.release()
        #expect(!program.isValid)

        #expect(throws: ANEError.self) {
            _ = try bridge.execute(program, inputs: [
                (name: "x", data: [0, 1], shape: [2]),
            ])
        }
    }
}
