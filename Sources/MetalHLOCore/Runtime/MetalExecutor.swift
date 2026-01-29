// MetalExecutor.swift
// MetalHLOCore
//
// Manages Metal device and execution.

import Foundation
import Metal
@preconcurrency import MetalPerformanceShadersGraph

/// Manages Metal device, compilation, and execution.
///
/// `MetalExecutor` is the central runtime component that handles
/// device management, graph compilation, buffer creation, and execution.
public final class MetalExecutor: @unchecked Sendable {

    // MARK: - Properties

    /// The Metal device.
    public let device: MTLDevice

    /// The command queue for execution.
    private let commandQueue: MTLCommandQueue

    /// The MPSGraph device wrapper.
    private let mpsDevice: MPSGraphDevice

    /// Cache for compiled graphs.
    private let compilationCache: CompilationCache

    // MARK: - Initialization

    /// Creates an executor for the default Metal device.
    ///
    /// - Throws: `ExecutorError.noMetalDevice` if no device is available.
    public convenience init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ExecutorError.noMetalDevice
        }
        try self.init(device: device)
    }

    /// Creates an executor for a specific Metal device.
    ///
    /// - Parameter device: The Metal device to use.
    /// - Throws: `ExecutorError.unsupportedDevice` if the device doesn't support required features.
    public init(device: MTLDevice) throws {
        // Verify device supports required features
        guard device.supportsFamily(.apple7) || device.supportsFamily(.mac2) else {
            throw ExecutorError.unsupportedDevice(
                "Device must support Apple7 (M1+) or Mac2 family"
            )
        }

        guard let queue = device.makeCommandQueue() else {
            throw ExecutorError.commandQueueCreationFailed
        }

        self.device = device
        self.commandQueue = queue
        self.mpsDevice = MPSGraphDevice(mtlDevice: device)
        self.compilationCache = CompilationCache()
    }

    // MARK: - Compilation

    /// Compiles an HLO module to a CompiledGraph.
    ///
    /// - Parameter module: The HLO module to compile.
    /// - Throws: `CompilationError` if compilation fails.
    /// - Returns: A compiled graph.
    public func compile(module: HLOModule) throws -> CompiledGraph {
        // Check cache first - use content hash to avoid returning wrong cached results
        // for different modules with the same name
        let cacheKey = "\(module.name)_\(module.description.hashValue)"
        if let cached = compilationCache.get(key: cacheKey) {
            return cached
        }

        // Compile
        let compiler = MPSGraphCompiler(device: device)
        let compiled = try compiler.compile(module: module)

        // Cache the result
        compilationCache.set(key: cacheKey, value: compiled)

        return compiled
    }

    // MARK: - Buffer Creation

    /// Creates a buffer storage from Float data (optimized fast path).
    public func createBufferStorage(
        _ data: [Float],
        shape: [Int]
    ) -> BufferStorage {
        BufferStorage(
            floatData: data,
            shape: shape,
            device: device
        )
    }

    /// Creates a buffer storage from Int32 data (optimized fast path).
    public func createBufferStorage(
        _ data: [Int32],
        shape: [Int]
    ) -> BufferStorage {
        BufferStorage(
            int32Data: data,
            shape: shape,
            device: device
        )
    }

    /// Creates a buffer storage from Int64 data (optimized fast path).
    public func createBufferStorage(
        _ data: [Int64],
        shape: [Int]
    ) -> BufferStorage {
        BufferStorage(
            int64Data: data,
            shape: shape,
            device: device
        )
    }

    /// Creates a buffer storage from numeric data (generic path).
    public func createBufferStorage<T: Numeric>(
        _ data: [T],
        shape: [Int],
        elementType: ElementType
    ) throws -> BufferStorage {
        try BufferStorage(
            data: data,
            shape: shape,
            elementType: elementType,
            device: device
        )
    }

    /// Creates a buffer storage from raw bytes.
    public func createBufferStorage(
        bytes data: Data,
        shape: [Int],
        elementType: ElementType
    ) throws -> BufferStorage {
        try BufferStorage(
            bytes: data,
            shape: shape,
            elementType: elementType,
            device: device
        )
    }

    /// Creates an uninitialized buffer storage.
    public func createBufferStorage(
        shape: [Int],
        elementType: ElementType
    ) throws -> BufferStorage {
        try BufferStorage(
            shape: shape,
            elementType: elementType,
            device: device
        )
    }

    // MARK: - Execution

    /// Executes a compiled graph with the given inputs.
    ///
    /// - Parameters:
    ///   - compiled: The compiled graph.
    ///   - inputs: The input buffer storages.
    /// - Throws: `ExecutorError` if execution fails.
    /// - Returns: The output buffer storages.
    public func execute(compiled: CompiledGraph, inputs: [BufferStorage]) throws -> [BufferStorage] {
        let (outputs, _) = try executeWithTiming(compiled: compiled, inputs: inputs)
        return outputs
    }

    /// Executes a compiled graph with timing information.
    ///
    /// - Parameters:
    ///   - compiled: The compiled graph.
    ///   - inputs: The input buffer storages.
    /// - Throws: `ExecutorError` if execution fails.
    /// - Returns: The output buffer storages and timing information.
    public func executeWithTiming(
        compiled: CompiledGraph,
        inputs: [BufferStorage]
    ) throws -> ([BufferStorage], ExecutionTiming) {
        // Validate input count
        guard inputs.count == compiled.inputTensors.count else {
            throw ExecutorError.inputMismatch(
                expected: compiled.inputTensors.count,
                got: inputs.count
            )
        }

        // Validate input types
        for (index, (input, expectedType)) in zip(inputs, compiled.inputTypes).enumerated() {
            let inputType = TensorType(shape: input.shape, elementType: input.elementType)
            if inputType != expectedType {
                throw ExecutorError.inputTypeMismatch(
                    index: index,
                    expected: expectedType,
                    got: inputType
                )
            }
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        // Create input tensor data
        var inputDict: [MPSGraphTensor: MPSGraphTensorData] = [:]
        for (tensor, storage) in zip(compiled.inputTensors, inputs) {
            let tensorData = MPSGraphTensorData(
                device: mpsDevice,
                data: storage.data,
                shape: storage.shape.map { NSNumber(value: $0) },
                dataType: storage.elementType.mpsDataType
            )
            inputDict[tensor] = tensorData
        }

        let encodeEndTime = CFAbsoluteTimeGetCurrent()

        // Execute using the executable's runAsync method for better GPU utilization
        // runAsync allows better pipelining than graph.run() which is synchronous
        let inputDataArray = compiled.inputTensors.compactMap { inputDict[$0] }

        let results: [MPSGraphTensor: MPSGraphTensorData]
        do {
            // Use synchronous run() to ensure GPU work completes before reading results.
            // runAsync returns immediately with placeholder data, causing reads to return zeros.
            let outputDataArray = compiled.executable.run(
                with: commandQueue,
                inputs: inputDataArray,
                results: nil,
                executionDescriptor: nil
            )

            // Map outputs back to tensor dictionary
            var resultDict: [MPSGraphTensor: MPSGraphTensorData] = [:]
            for (tensor, data) in zip(compiled.outputTensors, outputDataArray) {
                resultDict[tensor] = data
            }
            results = resultDict
        }

        let gpuEndTime = CFAbsoluteTimeGetCurrent()

        // Convert results to buffer storages
        var outputStorages: [BufferStorage] = []
        for (index, outputTensor) in compiled.outputTensors.enumerated() {
            guard let resultData = results[outputTensor] else {
                throw ExecutorError.executionFailed("Missing output for tensor at index \(index)")
            }
            let outputType = compiled.outputTypes[index]
            let storage = try BufferStorage(
                tensorData: resultData,
                type: outputType,
                device: device
            )
            outputStorages.append(storage)
        }

        let timing = ExecutionTiming(
            encodeTime: encodeEndTime - startTime,
            gpuTime: gpuEndTime - encodeEndTime,
            totalTime: gpuEndTime - startTime
        )

        return (outputStorages, timing)
    }
}

// MARK: - Executor Errors

/// Errors that can occur during execution.
public enum ExecutorError: Error, Sendable, CustomStringConvertible {
    case noMetalDevice
    case unsupportedDevice(String)
    case commandQueueCreationFailed
    case commandBufferCreationFailed
    case inputMismatch(expected: Int, got: Int)
    case inputTypeMismatch(index: Int, expected: TensorType, got: TensorType)
    case executionFailed(String)
    case bufferCreationFailed(String)
    case transferFailed(String)

    public var description: String {
        switch self {
        case .noMetalDevice:
            return "No Metal device available"
        case .unsupportedDevice(let reason):
            return "Unsupported device: \(reason)"
        case .commandQueueCreationFailed:
            return "Failed to create command queue"
        case .commandBufferCreationFailed:
            return "Failed to create command buffer"
        case .inputMismatch(let expected, let got):
            return "Input count mismatch: expected \(expected), got \(got)"
        case .inputTypeMismatch(let index, let expected, let got):
            return "Input type mismatch at index \(index): expected \(expected), got \(got)"
        case .executionFailed(let reason):
            return "Execution failed: \(reason)"
        case .bufferCreationFailed(let reason):
            return "Buffer creation failed: \(reason)"
        case .transferFailed(let reason):
            return "Transfer failed: \(reason)"
        }
    }
}

// MARK: - Execution Timing (Core)

/// Timing information for an execution.
public struct ExecutionTiming: Sendable {
    /// Time to encode commands (seconds).
    public let encodeTime: Double

    /// Time for GPU execution (seconds).
    public let gpuTime: Double

    /// Total wall-clock time (seconds).
    public let totalTime: Double

    public init(encodeTime: Double, gpuTime: Double, totalTime: Double) {
        self.encodeTime = encodeTime
        self.gpuTime = gpuTime
        self.totalTime = totalTime
    }
}
