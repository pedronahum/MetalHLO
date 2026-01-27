// KernelTypes.swift
// MetalHLOCore
//
// Types for kernel specification, dispatch configuration, and compiled executables.

import Foundation
@preconcurrency import Metal

// MARK: - Dispatch Configuration

/// Configuration for how to dispatch a Metal kernel.
public struct DispatchConfig: Sendable, Hashable {
    /// Grid size (number of threadgroups).
    public let gridSize: MTLSize

    /// Threadgroup size (threads per threadgroup).
    public let threadgroupSize: MTLSize

    /// Whether to use non-uniform threadgroups.
    public let useNonUniform: Bool

    public init(gridSize: MTLSize, threadgroupSize: MTLSize, useNonUniform: Bool = false) {
        self.gridSize = gridSize
        self.threadgroupSize = threadgroupSize
        self.useNonUniform = useNonUniform
    }

    /// Creates a 1D dispatch configuration.
    public static func dispatch1D(elements: Int, threadgroupSize: Int = 256) -> DispatchConfig {
        let tgSize = min(threadgroupSize, elements)
        let gridWidth = (elements + tgSize - 1) / tgSize
        return DispatchConfig(
            gridSize: MTLSize(width: gridWidth, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    /// Creates a 2D dispatch configuration.
    public static func dispatch2D(width: Int, height: Int, tileWidth: Int = 16, tileHeight: Int = 16) -> DispatchConfig {
        let twSize = min(tileWidth, width)
        let thSize = min(tileHeight, height)
        let gridWidth = (width + twSize - 1) / twSize
        let gridHeight = (height + thSize - 1) / thSize
        return DispatchConfig(
            gridSize: MTLSize(width: gridWidth, height: gridHeight, depth: 1),
            threadgroupSize: MTLSize(width: twSize, height: thSize, depth: 1)
        )
    }

    /// Creates a 3D dispatch configuration.
    public static func dispatch3D(width: Int, height: Int, depth: Int, tileSize: Int = 8) -> DispatchConfig {
        let gridWidth = (width + tileSize - 1) / tileSize
        let gridHeight = (height + tileSize - 1) / tileSize
        let gridDepth = (depth + tileSize - 1) / tileSize
        return DispatchConfig(
            gridSize: MTLSize(width: gridWidth, height: gridHeight, depth: gridDepth),
            threadgroupSize: MTLSize(width: tileSize, height: tileSize, depth: tileSize)
        )
    }

    /// Total number of threads.
    public var totalThreads: Int {
        gridSize.width * gridSize.height * gridSize.depth *
        threadgroupSize.width * threadgroupSize.height * threadgroupSize.depth
    }

    /// Total number of threadgroups.
    public var totalThreadgroups: Int {
        gridSize.width * gridSize.height * gridSize.depth
    }
}

// Hashable conformance for MTLSize
extension MTLSize: @retroactive Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(width)
        hasher.combine(height)
        hasher.combine(depth)
    }
}

extension MTLSize: @retroactive Equatable {
    public static func == (lhs: MTLSize, rhs: MTLSize) -> Bool {
        lhs.width == rhs.width && lhs.height == rhs.height && lhs.depth == rhs.depth
    }
}

// MARK: - Buffer Source

/// Where buffer data comes from.
public enum BufferSource: Sendable, Hashable {
    /// External input buffer (user-provided).
    case input(name: String)

    /// External output buffer (user-provided or unified).
    case output(name: String)

    /// Offset within the unified intermediate buffer.
    case unified(offset: Int)

    /// Inline constant data.
    case constant(id: String)

    /// Threadgroup (shared) memory.
    case threadgroup(size: Int)

    /// Scalar uniform value (passed via setBytes).
    case scalar(UInt32)
}

// MARK: - Buffer Binding

/// How to bind a buffer to a kernel argument.
public struct BufferBinding: Sendable, Hashable {
    /// Argument index in the kernel function.
    public let index: Int

    /// Where the buffer data comes from.
    public let source: BufferSource

    /// Offset within the source buffer.
    public let offset: Int

    /// Size in bytes (for validation).
    public let size: Int

    /// Access mode.
    public let access: AccessMode

    public enum AccessMode: String, Sendable, Hashable {
        case read
        case write
        case readWrite
    }

    public init(index: Int, source: BufferSource, offset: Int = 0, size: Int = 0, access: AccessMode = .read) {
        self.index = index
        self.source = source
        self.offset = offset
        self.size = size
        self.access = access
    }
}

// MARK: - Tuning Configuration

/// Tuning parameters for a kernel.
public struct TuningConfig: Sendable, Hashable {
    /// Tile size in M dimension (for matmul).
    public var tileM: Int?

    /// Tile size in N dimension (for matmul).
    public var tileN: Int?

    /// Tile size in K dimension (for matmul).
    public var tileK: Int?

    /// Block size for reductions.
    public var blockSize: Int?

    /// Whether to use shared memory.
    public var useSharedMemory: Bool

    /// Whether to use SIMD groups.
    public var useSIMDGroups: Bool

    /// Unroll factor for inner loops.
    public var unrollFactor: Int?

    /// Number of elements per thread.
    public var elementsPerThread: Int?

    public init(
        tileM: Int? = nil,
        tileN: Int? = nil,
        tileK: Int? = nil,
        blockSize: Int? = nil,
        useSharedMemory: Bool = false,
        useSIMDGroups: Bool = false,
        unrollFactor: Int? = nil,
        elementsPerThread: Int? = nil
    ) {
        self.tileM = tileM
        self.tileN = tileN
        self.tileK = tileK
        self.blockSize = blockSize
        self.useSharedMemory = useSharedMemory
        self.useSIMDGroups = useSIMDGroups
        self.unrollFactor = unrollFactor
        self.elementsPerThread = elementsPerThread
    }

    /// Default tuning for small operations.
    public static let small = TuningConfig(
        blockSize: 256,
        useSharedMemory: false,
        useSIMDGroups: false
    )

    /// Default tuning for matrix operations.
    public static let matmul = TuningConfig(
        tileM: 32,
        tileN: 32,
        tileK: 32,
        useSharedMemory: true,
        useSIMDGroups: true
    )

    /// Default tuning for attention.
    public static let attention = TuningConfig(
        tileM: 64,
        tileN: 64,
        tileK: 64,
        useSharedMemory: true,
        useSIMDGroups: true,
        unrollFactor: 4
    )
}

// MARK: - Kernel Spec

/// Complete specification for a compiled kernel.
public struct KernelSpec: Sendable {
    /// Operation ID this kernel implements.
    public let opID: OpID

    /// Generated Metal source code.
    public let metalSource: String

    /// Entry point function name.
    public let entryPoint: String

    /// Dispatch configuration.
    public let dispatch: DispatchConfig

    /// Buffer bindings.
    public let bindings: [BufferBinding]

    /// Tuning configuration used.
    public let tuning: TuningConfig?

    /// Shared memory size required.
    public let sharedMemorySize: Int

    /// Input tensor shapes (for validation).
    public let inputShapes: [[Int]]

    /// Output tensor shapes.
    public let outputShapes: [[Int]]

    public init(
        opID: OpID,
        metalSource: String,
        entryPoint: String,
        dispatch: DispatchConfig,
        bindings: [BufferBinding],
        tuning: TuningConfig? = nil,
        sharedMemorySize: Int = 0,
        inputShapes: [[Int]] = [],
        outputShapes: [[Int]] = []
    ) {
        self.opID = opID
        self.metalSource = metalSource
        self.entryPoint = entryPoint
        self.dispatch = dispatch
        self.bindings = bindings
        self.tuning = tuning
        self.sharedMemorySize = sharedMemorySize
        self.inputShapes = inputShapes
        self.outputShapes = outputShapes
    }
}

// MARK: - Tensor Spec

/// Specification for an input or output tensor.
public struct TensorSpec: Sendable, Hashable {
    /// Tensor name.
    public let name: String

    /// Shape dimensions.
    public let shape: [Int]

    /// Element type.
    public let elementType: ElementType

    /// Size in bytes.
    public var byteSize: Int {
        shape.reduce(1, *) * elementType.byteSize
    }

    public init(name: String, shape: [Int], elementType: ElementType) {
        self.name = name
        self.shape = shape
        self.elementType = elementType
    }

    public init(from param: TensorParam) {
        self.name = param.name
        self.shape = param.type.shape
        self.elementType = param.type.elementType
    }

    public init(from info: TensorInfo) {
        self.name = info.id
        self.shape = info.shape
        self.elementType = info.elementType
    }
}

// MARK: - Compiled Executable

/// A fully compiled executable ready for execution.
public final class CompiledExecutable: @unchecked Sendable {

    /// Compiled pipeline states for each operation.
    public let pipelines: [OpID: MTLComputePipelineState]

    /// Dispatch configurations for each operation.
    public let dispatches: [OpID: DispatchConfig]

    /// Buffer bindings for each operation.
    public let bindings: [OpID: [BufferBinding]]

    /// Shared (threadgroup) memory sizes for each operation.
    public let sharedMemorySizes: [OpID: Int]

    /// Memory plan for the unified buffer.
    public let memoryPlan: MemoryPlan

    /// Input tensor specifications.
    public let inputSpecs: [String: TensorSpec]

    /// Output tensor specifications.
    public let outputSpecs: [String: TensorSpec]

    /// Constant buffers (pre-created).
    public let constantBuffers: [String: MTLBuffer]

    /// Execution order (excludes constant operations which don't have pipelines).
    public var executionOrder: [OpID] {
        memoryPlan.executionOrder
            .map { String($0) }
            .filter { pipelines[$0] != nil }
    }

    /// Total memory required for intermediates.
    public var totalMemoryBytes: Int {
        memoryPlan.totalBytes
    }

    public init(
        pipelines: [OpID: MTLComputePipelineState],
        dispatches: [OpID: DispatchConfig],
        bindings: [OpID: [BufferBinding]],
        sharedMemorySizes: [OpID: Int] = [:],
        memoryPlan: MemoryPlan,
        inputSpecs: [String: TensorSpec],
        outputSpecs: [String: TensorSpec],
        constantBuffers: [String: MTLBuffer] = [:]
    ) {
        self.pipelines = pipelines
        self.dispatches = dispatches
        self.bindings = bindings
        self.sharedMemorySizes = sharedMemorySizes
        self.memoryPlan = memoryPlan
        self.inputSpecs = inputSpecs
        self.outputSpecs = outputSpecs
        self.constantBuffers = constantBuffers
    }

    /// Validates that the provided inputs match the expected specs.
    public func validateInputs(_ inputs: [String: MTLBuffer]) throws {
        for (name, spec) in inputSpecs {
            guard let buffer = inputs[name] else {
                throw CompiledExecutableError.missingInput(name)
            }
            guard buffer.length >= spec.byteSize else {
                throw CompiledExecutableError.inputTooSmall(
                    name: name,
                    expected: spec.byteSize,
                    actual: buffer.length
                )
            }
        }
    }

    /// Returns statistics about this executable.
    public var statistics: Statistics {
        Statistics(
            numOperations: pipelines.count,
            totalMemoryBytes: totalMemoryBytes,
            peakMemoryBytes: memoryPlan.peakMemory,
            numInputs: inputSpecs.count,
            numOutputs: outputSpecs.count,
            numConstants: constantBuffers.count
        )
    }

    public struct Statistics: Sendable {
        public let numOperations: Int
        public let totalMemoryBytes: Int
        public let peakMemoryBytes: Int
        public let numInputs: Int
        public let numOutputs: Int
        public let numConstants: Int
    }
}

/// Errors that can occur with compiled executables.
public enum CompiledExecutableError: Error, Sendable {
    case missingInput(String)
    case inputTooSmall(name: String, expected: Int, actual: Int)
    case missingPipeline(OpID)
    case executionFailed(String)
}

// MARK: - Execution Result

/// Result of executing a compiled executable.
public struct ExecutionResult: Sendable {
    /// Output buffers.
    public let outputs: [String: MTLBuffer]

    /// Execution time in milliseconds.
    public let executionTimeMs: Double

    /// Per-kernel timing (if profiling enabled).
    public let kernelTimings: [OpID: Double]?

    public init(
        outputs: [String: MTLBuffer],
        executionTimeMs: Double,
        kernelTimings: [OpID: Double]? = nil
    ) {
        self.outputs = outputs
        self.executionTimeMs = executionTimeMs
        self.kernelTimings = kernelTimings
    }
}
