// MetalEmitter.swift
// MetalHLOCore
//
// Converts MetalIR to Metal shader source code.

import Foundation

/// Emits Metal shader source code from MetalIR.
///
/// The MetalEmitter is responsible for converting the high-level MetalIR
/// representation into actual Metal Shading Language (MSL) code.
public final class MetalEmitter: Sendable {

    // MARK: - Types

    /// Kernel definition with parameters and body.
    public struct KernelDefinition: Sendable {
        /// Kernel function name.
        public let name: String

        /// Kernel parameters (buffers, threadgroup memory, etc.).
        public let parameters: [KernelParameter]

        /// Thread indexing parameters to include.
        public let threadIndexing: ThreadIndexing

        /// Threadgroup memory declarations.
        public let sharedMemory: [SharedMemoryDecl]

        /// Constants to define at the top.
        public let constants: [ConstantDecl]

        /// Kernel body IR.
        public let body: [MetalIR]

        public init(
            name: String,
            parameters: [KernelParameter] = [],
            threadIndexing: ThreadIndexing = .standard,
            sharedMemory: [SharedMemoryDecl] = [],
            constants: [ConstantDecl] = [],
            body: [MetalIR]
        ) {
            self.name = name
            self.parameters = parameters
            self.threadIndexing = threadIndexing
            self.sharedMemory = sharedMemory
            self.constants = constants
            self.body = body
        }
    }

    /// Kernel parameter definition.
    public struct KernelParameter: Sendable {
        public let name: String
        public let type: MetalType
        public let accessMode: AccessMode
        public let bufferIndex: Int

        public enum AccessMode: Sendable {
            case readonly
            case writeonly
            case readwrite
        }

        public init(name: String, type: MetalType, accessMode: AccessMode = .readonly, bufferIndex: Int) {
            self.name = name
            self.type = type
            self.accessMode = accessMode
            self.bufferIndex = bufferIndex
        }
    }

    /// Thread indexing mode.
    public struct ThreadIndexing: Sendable {
        public let includeGID: Bool
        public let includeTID: Bool
        public let includeTGID: Bool
        public let includeSimdLane: Bool
        public let includeSimdGroup: Bool

        public static let standard = ThreadIndexing(
            includeGID: true,
            includeTID: true,
            includeTGID: true,
            includeSimdLane: false,
            includeSimdGroup: false
        )

        public static let simdAware = ThreadIndexing(
            includeGID: true,
            includeTID: true,
            includeTGID: true,
            includeSimdLane: true,
            includeSimdGroup: true
        )

        public static let minimal = ThreadIndexing(
            includeGID: true,
            includeTID: false,
            includeTGID: false,
            includeSimdLane: false,
            includeSimdGroup: false
        )

        public init(
            includeGID: Bool = true,
            includeTID: Bool = false,
            includeTGID: Bool = false,
            includeSimdLane: Bool = false,
            includeSimdGroup: Bool = false
        ) {
            self.includeGID = includeGID
            self.includeTID = includeTID
            self.includeTGID = includeTGID
            self.includeSimdLane = includeSimdLane
            self.includeSimdGroup = includeSimdGroup
        }
    }

    /// Shared memory declaration.
    public struct SharedMemoryDecl: Sendable {
        public let name: String
        public let type: MetalType
        public let dimensions: [Int]

        public init(name: String, type: MetalType, dimensions: [Int]) {
            self.name = name
            self.type = type
            self.dimensions = dimensions
        }
    }

    /// Constant declaration.
    public struct ConstantDecl: Sendable {
        public let name: String
        public let type: MetalType
        public let value: String

        public init(name: String, type: MetalType, value: String) {
            self.name = name
            self.type = type
            self.value = value
        }
    }

    // MARK: - Emission

    /// Emits a complete Metal kernel from a kernel definition.
    ///
    /// - Parameter kernel: The kernel definition.
    /// - Returns: Complete Metal source code.
    public func emit(_ kernel: KernelDefinition) -> String {
        var code = emitHeader()
        code += emitConstants(kernel.constants)
        code += emitKernelSignature(kernel)
        code += emitSharedMemory(kernel.sharedMemory)
        code += emitBody(kernel.body)
        code += "}\n"
        return code
    }

    /// Emits a standalone expression.
    ///
    /// - Parameters:
    ///   - ir: The MetalIR expression.
    ///   - indent: Indentation level.
    /// - Returns: Metal source code for the expression.
    public func emit(_ ir: MetalIR, indent: Int = 0) -> String {
        ir.emit(indent: indent)
    }

    // MARK: - Private Emission Methods

    private func emitHeader() -> String {
        return """
        #include <metal_stdlib>
        using namespace metal;


        """
    }

    private func emitConstants(_ constants: [ConstantDecl]) -> String {
        guard !constants.isEmpty else { return "" }

        var code = ""
        for constant in constants {
            code += "constant \(constant.type.metalTypeName) \(constant.name) = \(constant.value);\n"
        }
        code += "\n"
        return code
    }

    private func emitKernelSignature(_ kernel: KernelDefinition) -> String {
        var code = "kernel void \(kernel.name)(\n"

        var params: [String] = []

        // Buffer parameters
        for param in kernel.parameters {
            let accessQualifier: String
            switch param.accessMode {
            case .readonly:
                accessQualifier = "const"
            case .writeonly, .readwrite:
                accessQualifier = ""
            }

            let typeStr = "device \(accessQualifier) \(param.type.metalTypeName)*"
            params.append("    \(typeStr) \(param.name) [[buffer(\(param.bufferIndex))]]")
        }

        // Thread indexing parameters
        if kernel.threadIndexing.includeGID {
            params.append("    uint3 gid [[thread_position_in_grid]]")
        }
        if kernel.threadIndexing.includeTID {
            params.append("    uint3 tid [[thread_position_in_threadgroup]]")
        }
        if kernel.threadIndexing.includeTGID {
            params.append("    uint3 tgid [[threadgroup_position_in_grid]]")
        }
        if kernel.threadIndexing.includeSimdLane {
            params.append("    uint simd_lane [[thread_index_in_simdgroup]]")
        }
        if kernel.threadIndexing.includeSimdGroup {
            params.append("    uint simd_group [[simdgroup_index_in_threadgroup]]")
        }

        code += params.joined(separator: ",\n")
        code += "\n) {\n"

        return code
    }

    private func emitSharedMemory(_ decls: [SharedMemoryDecl]) -> String {
        guard !decls.isEmpty else { return "" }

        var code = ""
        for decl in decls {
            let dims = decl.dimensions.map { "[\($0)]" }.joined()
            code += "    threadgroup \(decl.type.metalTypeName) \(decl.name)\(dims);\n"
        }
        code += "\n"
        return code
    }

    private func emitBody(_ body: [MetalIR]) -> String {
        var code = ""
        for statement in body {
            code += statement.emit(indent: 1)
            if !code.hasSuffix("\n") {
                code += "\n"
            }
        }
        return code
    }
}

// MARK: - MetalType Extension for Emission

extension MetalType {
    /// Metal type name for code generation.
    var metalTypeName: String {
        switch self {
        case .float:
            return "float"
        case .half:
            return "half"
        case .int:
            return "int"
        case .uint:
            return "uint"
        case .bool:
            return "bool"
        case .float4:
            return "float4"
        case .half4:
            return "half4"
        case .float2:
            return "float2"
        case .half2:
            return "half2"
        case .uint2:
            return "uint2"
        case .uint3:
            return "uint3"
        case .void:
            return "void"
        }
    }
}

// MARK: - Convenience Factory Methods

extension MetalEmitter {

    /// Creates a simple elementwise kernel definition.
    ///
    /// - Parameters:
    ///   - name: Kernel name.
    ///   - inputCount: Number of input buffers.
    ///   - outputCount: Number of output buffers.
    ///   - dtype: Data type.
    ///   - body: IR for the kernel body.
    /// - Returns: Kernel definition.
    public static func elementwiseKernel(
        name: String,
        inputCount: Int,
        outputCount: Int,
        dtype: MetalType,
        body: [MetalIR]
    ) -> KernelDefinition {
        var params: [KernelParameter] = []

        for i in 0..<inputCount {
            params.append(KernelParameter(
                name: "input\(i)",
                type: dtype,
                accessMode: .readonly,
                bufferIndex: i
            ))
        }

        for i in 0..<outputCount {
            params.append(KernelParameter(
                name: "output\(i)",
                type: dtype,
                accessMode: .writeonly,
                bufferIndex: inputCount + i
            ))
        }

        return KernelDefinition(
            name: name,
            parameters: params,
            threadIndexing: .minimal,
            body: body
        )
    }

    /// Creates a tiled matmul kernel definition.
    ///
    /// - Parameters:
    ///   - name: Kernel name.
    ///   - M: Output rows.
    ///   - N: Output columns.
    ///   - K: Inner dimension.
    ///   - tileM: Tile size for M.
    ///   - tileN: Tile size for N.
    ///   - tileK: Tile size for K.
    ///   - dtype: Data type.
    ///   - body: IR for the kernel body.
    /// - Returns: Kernel definition.
    public static func tiledMatMulKernel(
        name: String,
        M: Int, N: Int, K: Int,
        tileM: Int, tileN: Int, tileK: Int,
        dtype: MetalType,
        body: [MetalIR]
    ) -> KernelDefinition {
        let params = [
            KernelParameter(name: "A", type: dtype, accessMode: .readonly, bufferIndex: 0),
            KernelParameter(name: "B", type: dtype, accessMode: .readonly, bufferIndex: 1),
            KernelParameter(name: "C", type: dtype, accessMode: .writeonly, bufferIndex: 2)
        ]

        let constants = [
            ConstantDecl(name: "M", type: .uint, value: "\(M)"),
            ConstantDecl(name: "N", type: .uint, value: "\(N)"),
            ConstantDecl(name: "K", type: .uint, value: "\(K)"),
            ConstantDecl(name: "TILE_M", type: .uint, value: "\(tileM)"),
            ConstantDecl(name: "TILE_N", type: .uint, value: "\(tileN)"),
            ConstantDecl(name: "TILE_K", type: .uint, value: "\(tileK)")
        ]

        let sharedMem = [
            SharedMemoryDecl(name: "As", type: dtype, dimensions: [tileM, tileK]),
            SharedMemoryDecl(name: "Bs", type: dtype, dimensions: [tileK, tileN])
        ]

        return KernelDefinition(
            name: name,
            parameters: params,
            threadIndexing: .simdAware,
            sharedMemory: sharedMem,
            constants: constants,
            body: body
        )
    }

    /// Creates an attention kernel definition.
    ///
    /// - Parameters:
    ///   - name: Kernel name.
    ///   - batchSize: Batch size.
    ///   - numHeads: Number of attention heads.
    ///   - seqLen: Sequence length.
    ///   - headDim: Dimension per head.
    ///   - blockQ: Block size for Q.
    ///   - blockKV: Block size for K/V.
    ///   - dtype: Data type.
    ///   - body: IR for the kernel body.
    /// - Returns: Kernel definition.
    public static func attentionKernel(
        name: String,
        batchSize: Int, numHeads: Int, seqLen: Int, headDim: Int,
        blockQ: Int, blockKV: Int,
        dtype: MetalType,
        body: [MetalIR]
    ) -> KernelDefinition {
        let params = [
            KernelParameter(name: "Q", type: dtype, accessMode: .readonly, bufferIndex: 0),
            KernelParameter(name: "K", type: dtype, accessMode: .readonly, bufferIndex: 1),
            KernelParameter(name: "V", type: dtype, accessMode: .readonly, bufferIndex: 2),
            KernelParameter(name: "O", type: dtype, accessMode: .writeonly, bufferIndex: 3)
        ]

        let constants = [
            ConstantDecl(name: "BATCH", type: .uint, value: "\(batchSize)"),
            ConstantDecl(name: "NUM_HEADS", type: .uint, value: "\(numHeads)"),
            ConstantDecl(name: "SEQ_LEN", type: .uint, value: "\(seqLen)"),
            ConstantDecl(name: "HEAD_DIM", type: .uint, value: "\(headDim)"),
            ConstantDecl(name: "BLOCK_Q", type: .uint, value: "\(blockQ)"),
            ConstantDecl(name: "BLOCK_KV", type: .uint, value: "\(blockKV)"),
            ConstantDecl(name: "SCALE", type: .float, value: String(format: "%.6f", 1.0 / sqrt(Double(headDim))))
        ]

        let sharedMem = [
            SharedMemoryDecl(name: "Qs", type: dtype, dimensions: [blockQ, headDim]),
            SharedMemoryDecl(name: "Ks", type: dtype, dimensions: [blockKV, headDim]),
            SharedMemoryDecl(name: "Vs", type: dtype, dimensions: [blockKV, headDim]),
            SharedMemoryDecl(name: "scores", type: dtype, dimensions: [blockQ, blockKV])
        ]

        return KernelDefinition(
            name: name,
            parameters: params,
            threadIndexing: .simdAware,
            sharedMemory: sharedMem,
            constants: constants,
            body: body
        )
    }
}
