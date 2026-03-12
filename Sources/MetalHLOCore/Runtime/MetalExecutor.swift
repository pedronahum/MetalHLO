// MetalExecutor.swift
// MetalHLOCore
//
// Manages Metal device and execution.

import Foundation
import Metal
@preconcurrency import MetalPerformanceShadersGraph

/// Global semaphore to serialize Metal shader compilation.
/// Metal shader compilation can crash when multiple instances compile concurrently.
/// This is only needed during compilation, NOT during execution of compiled pipelines.
/// Metal command queues handle concurrent command buffer submission safely.
public let metalExecutionSemaphore = DispatchSemaphore(value: 1)

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

        // Serialize MPSGraph compilation to prevent concurrent access crashes
        metalExecutionSemaphore.wait()
        defer { metalExecutionSemaphore.signal() }

        // Double-check cache after acquiring semaphore (another thread may have compiled)
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
        // Fast path: single elementwise op (softmax, gelu, tanh, sigmoid, exp) — bypass MPSGraph.
        if let shortcut = compiled.elementwiseShortcut {
            if let result = try? executeElementwiseDirect(shortcut: shortcut, inputs: inputs) {
                return result
            }
            // Fall through to MPSGraph path if direct execution fails (e.g. non-MTLBuffer inputs)
        }

        // Fast path: if the function is a single attention op with direct MTLBuffer inputs,
        // bypass MPSGraph and execute FlashAttentionTiledKernel directly.
        if let shortcut = compiled.flashAttentionShortcut {
            if let result = try? executeFlashAttentionDirect(shortcut: shortcut, inputs: inputs) {
                return result
            }
            // Fall through to MPSGraph path if direct execution fails (e.g. non-MTLBuffer inputs)
        }

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

        // Note: Semaphore removed - MPSGraph execution is thread-safe once compiled.
        // The command queue handles concurrent command buffer submission safely.

        // Create input tensor data
        // When an MTLBuffer is available (large tensors), create MPSGraphTensorData
        // directly from the buffer to avoid GPU→CPU→GPU round-trip copies.
        var inputDict: [MPSGraphTensor: MPSGraphTensorData] = [:]
        for (tensor, storage) in zip(compiled.inputTensors, inputs) {
            let tensorData: MPSGraphTensorData
            let shape = storage.shape.map { NSNumber(value: $0) }
            let dataType = storage.elementType.mpsDataType

            if let ndarray = storage.mpsNDArray {
                // Zero-copy: reuse the MPSNDArray from a previous execution directly.
                tensorData = MPSGraphTensorData(ndarray)
            } else if let buffer = storage.metalBuffer {
                // Zero-copy: MPSGraphTensorData aliases the MTLBuffer directly.
                tensorData = MPSGraphTensorData(buffer, shape: shape, dataType: dataType)
            } else {
                // Small tensor path: copy from Data (small enough that overhead is negligible)
                tensorData = MPSGraphTensorData(
                    device: mpsDevice,
                    data: storage.data,
                    shape: shape,
                    dataType: dataType
                )
            }
            inputDict[tensor] = tensorData
        }

        let encodeEndTime = CFAbsoluteTimeGetCurrent()

        let inputDataArray = compiled.inputTensors.compactMap { inputDict[$0] }

        // Execute: synchronous run() ensures GPU work completes before reading results.
        let outputDataArray = compiled.executable.run(
            with: commandQueue,
            inputs: inputDataArray,
            results: nil,
            executionDescriptor: nil
        )

        let gpuEndTime = CFAbsoluteTimeGetCurrent()

        // Convert results to buffer storages.
        // Outputs use readBytes into an MTLBuffer — a single memcpy on unified memory.
        // (Pre-allocating output buffers via `results` is unsafe: MPSGraph may produce
        // shapes that differ from compiled.outputTypes for ops like pad/concat.)
        var outputStorages: [BufferStorage] = []
        for (index, outputType) in compiled.outputTypes.enumerated() {
            let storage = try BufferStorage(
                tensorData: outputDataArray[index],
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

    /// Executes a single elementwise op directly via a Metal kernel, bypassing MPSGraph.
    ///
    /// Called when `compiled.elementwiseShortcut` is non-nil and the input has MTLBuffer backing.
    /// Eliminates MPSGraph dispatch overhead for cheap single-op functions.
    private func executeElementwiseDirect(
        shortcut: ElementwiseShortcut,
        inputs: [BufferStorage]
    ) throws -> ([BufferStorage], ExecutionTiming) {
        let start = CFAbsoluteTimeGetCurrent()

        guard shortcut.inputIndex < inputs.count else {
            throw ExecutorError.inputMismatch(expected: shortcut.inputIndex + 1, got: inputs.count)
        }

        // Float16 not currently supported in Metal kernel shortcut; fall back to MPSGraph.
        guard !shortcut.isFloat16 else {
            throw ExecutorError.generalError("Elementwise shortcut: float16 not supported")
        }

        let inputStorage = inputs[shortcut.inputIndex]

        guard let inputBuffer = inputStorage.metalBuffer else {
            throw ExecutorError.generalError("Elementwise shortcut requires MTLBuffer input")
        }

        let bytesPerElem = shortcut.isFloat16 ? 2 : 4
        let outByteCount = shortcut.totalElements * bytesPerElem
        guard let outBuffer = device.makeBuffer(length: outByteCount, options: .storageModeShared) else {
            throw ExecutorError.generalError("Failed to allocate elementwise output buffer")
        }

        let params: any KernelParams
        switch shortcut.op {
        case .softmax(let batchSize, let seqLen):
            params = SoftmaxParams(batchSize: batchSize, seqLen: seqLen)
        default:
            params = ElementwiseParams(totalElements: shortcut.totalElements, shape: shortcut.shape)
        }

        try MetalKernelRegistry.shared.execute(
            kernelName: shortcut.op.kernelName,
            inputs:  [inputBuffer],
            outputs: [outBuffer],
            params:  params,
            device:  device,
            commandQueue: commandQueue
        )

        let end = CFAbsoluteTimeGetCurrent()

        let elementType: ElementType = shortcut.isFloat16 ? .float16 : .float32
        let largeTensor = LargeTensorStorage(buffer: outBuffer, shape: shortcut.shape, elementType: elementType)
        let outputStorage = BufferStorage(largeTensor: largeTensor, device: device)
        let timing = ExecutionTiming(encodeTime: 0, gpuTime: end - start, totalTime: end - start)
        return ([outputStorage], timing)
    }

    /// Executes a Flash Attention function directly via FlashAttentionTiledKernel,
    /// bypassing MPSGraph. Called when `compiled.flashAttentionShortcut` is non-nil
    /// and all inputs have MTLBuffer backing.
    private func executeFlashAttentionDirect(
        shortcut: FlashAttentionShortcut,
        inputs: [BufferStorage]
    ) throws -> ([BufferStorage], ExecutionTiming) {
        let start = CFAbsoluteTimeGetCurrent()

        guard shortcut.qIndex < inputs.count,
              shortcut.kIndex < inputs.count,
              shortcut.vIndex < inputs.count
        else { throw ExecutorError.inputMismatch(expected: 3, got: inputs.count) }

        let qStorage = inputs[shortcut.qIndex]
        let kStorage = inputs[shortcut.kIndex]
        let vStorage = inputs[shortcut.vIndex]

        guard let qBuffer = qStorage.metalBuffer,
              let kBuffer = kStorage.metalBuffer,
              let vBuffer = vStorage.metalBuffer
        else { throw ExecutorError.generalError("Flash attention direct path requires MTLBuffer inputs") }

        // Allocate output buffer
        let outShape    = [shortcut.batchSize, shortcut.numHeads, shortcut.seqLenQ, shortcut.headDim]
        let outElements = outShape.reduce(1, *)
        let bytesPerElem = shortcut.isFloat16 ? 2 : 4
        let outByteCount = outElements * bytesPerElem
        guard let outBuffer = device.makeBuffer(length: outByteCount, options: .storageModeShared) else {
            throw ExecutorError.generalError("Failed to allocate Flash Attention output buffer")
        }

        // Build kernel params (blockSize=32 is the tiled kernel's Br=Bc)
        let flashParams = FlashAttentionParams(
            batchSize:  shortcut.batchSize,
            seqLenQ:    shortcut.seqLenQ,
            seqLenKV:   shortcut.seqLenKV,
            numHeads:   shortcut.numHeads,
            headDim:    shortcut.headDim,
            scale:      shortcut.scale,
            isCausal:   shortcut.isCausal,
            blockSize:  32
        )

        try MetalKernelRegistry.shared.execute(
            kernelName: "flash_attention_tiled",
            inputs:  [qBuffer, kBuffer, vBuffer],
            outputs: [outBuffer],
            params:  flashParams,
            device:  device,
            commandQueue: commandQueue
        )

        let end = CFAbsoluteTimeGetCurrent()

        // Wrap output in BufferStorage
        let elementType: ElementType = shortcut.isFloat16 ? .float16 : .float32
        let largeTensor = LargeTensorStorage(buffer: outBuffer, shape: outShape, elementType: elementType)
        let outputStorage = BufferStorage(largeTensor: largeTensor, device: device)
        let timing = ExecutionTiming(
            encodeTime: 0,
            gpuTime:    end - start,
            totalTime:  end - start
        )
        return ([outputStorage], timing)
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
    case generalError(String)

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
        case .generalError(let reason):
            return "Error: \(reason)"
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
