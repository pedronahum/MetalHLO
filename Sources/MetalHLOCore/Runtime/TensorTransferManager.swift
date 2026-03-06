// TensorTransferManager.swift
// MetalHLOCore
//
// Manages tensor data flow between GPU and ANE partitions during
// heterogeneous execution. On Apple Silicon, GPU and ANE share
// unified memory, so transfers are API conversions rather than
// physical memory copies.

import Foundation
import Metal
import CoreML

/// Manages intermediate tensor buffers during heterogeneous execution.
///
/// Tracks tensors by their SSA names and converts between GPU (`BufferStorage`)
/// and ANE (`[Float]`) representations as needed at partition boundaries.
///
/// When shared IOSurface buffers are available, the GPU→ANE input path
/// is zero-copy: the Metal buffer and MLMultiArray share the same memory.
///
/// Thread-safe: all public methods are synchronized via `NSLock` to support
/// concurrent GPU and ANE partition execution in pipelined mode.
final class TensorTransferManager: @unchecked Sendable {

    private let device: MTLDevice
    private let lock = NSLock()

    /// GPU-side tensor data (BufferStorage from MetalExecutor).
    private var gpuStorages: [String: BufferStorage] = [:]

    /// ANE-side tensor data (Float arrays from CoreMLBridge).
    private var cpuArrays: [String: (data: [Float], shape: [Int])] = [:]

    /// Shared buffers for zero-copy GPU↔ANE transfer.
    private var sharedBuffers: [String: SharedIOSurfaceBuffer] = [:]

    /// Tensor types for all stored tensors.
    private var tensorTypes: [String: TensorType] = [:]

    /// Tracks which tensors have been fully written and are safe to read.
    /// In the level-based concurrent scheduler, all tensors from level N are
    /// implicitly ready when level N+1 starts. This tracking supports future
    /// finer-grained task-level schedulers that dispatch individual partitions
    /// as soon as all their input tensors become available.
    private var readyTensors: Set<String> = []

    init(device: MTLDevice) {
        self.device = device
    }

    // MARK: - Store Results

    /// Stores a GPU BufferStorage result from a GPU partition.
    func storeGPUResult(name: String, storage: BufferStorage) {
        lock.lock()
        defer { lock.unlock() }
        gpuStorages[name] = storage
        tensorTypes[name] = TensorType(shape: storage.shape, elementType: storage.elementType)
    }

    /// Stores a CPU array result from an ANE partition.
    func storeCPUResult(name: String, data: [Float], shape: [Int], elementType: ElementType = .float32) {
        lock.lock()
        defer { lock.unlock() }
        cpuArrays[name] = (data: data, shape: shape)
        tensorTypes[name] = TensorType(shape: shape, elementType: elementType)
    }

    // MARK: - Retrieve as BufferStorage (for GPU partitions)

    /// Gets a tensor as BufferStorage for GPU partition inputs.
    /// If the tensor is only available as a CPU array, creates a BufferStorage from it.
    func getBufferStorage(name: String) throws -> BufferStorage {
        lock.lock()
        defer { lock.unlock() }

        // Already a GPU storage
        if let storage = gpuStorages[name] {
            return storage
        }

        // Convert from CPU array
        guard let cpuData = cpuArrays[name] else {
            throw TensorTransferError.tensorNotFound(name)
        }

        let storage = BufferStorage(
            floatData: cpuData.data,
            shape: cpuData.shape,
            device: device
        )

        // Cache the GPU storage
        gpuStorages[name] = storage
        return storage
    }

    // MARK: - Retrieve as CPU Array (for ANE partitions)

    /// Gets a tensor as a Float array for ANE partition inputs.
    /// If the tensor is only available as a GPU BufferStorage, extracts data from it.
    func getCPUArray(name: String) throws -> (data: [Float], shape: [Int]) {
        lock.lock()
        defer { lock.unlock() }

        // Already a CPU array
        if let cpuData = cpuArrays[name] {
            return cpuData
        }

        // Convert from GPU storage
        guard let storage = gpuStorages[name] else {
            throw TensorTransferError.tensorNotFound(name)
        }

        let data = storage.data
        let floatCount = storage.count
        let floatArray: [Float] = data.withUnsafeBytes { bytes in
            let floatPtr = bytes.bindMemory(to: Float.self)
            return Array(floatPtr.prefix(floatCount))
        }

        let result = (data: floatArray, shape: storage.shape)
        // Cache the CPU array
        cpuArrays[name] = result
        return result
    }

    // MARK: - Shared Buffer (Zero-Copy)

    /// Gets or creates a shared IOSurface-backed buffer for a tensor.
    ///
    /// If the tensor exists as a GPU storage or CPU array, creates a shared
    /// buffer and copies the data into it (one-time cost). Subsequent calls
    /// return the cached shared buffer.
    ///
    /// - Returns: An MLMultiArray backed by shared memory, or nil if
    ///   the tensor isn't available or shared buffer creation fails.
    func getSharedMLMultiArray(name: String) throws -> (array: MLMultiArray, shape: [Int])? {
        lock.lock()
        defer { lock.unlock() }

        // Return cached shared buffer
        if let shared = sharedBuffers[name] {
            let array = try shared.makeMLMultiArray()
            return (array: array, shape: shared.shape)
        }

        // Try to create from GPU storage
        if let storage = gpuStorages[name] {
            let data = storage.data
            let floatArray: [Float] = data.withUnsafeBytes { bytes in
                let floatPtr = bytes.bindMemory(to: Float.self)
                return Array(floatPtr.prefix(storage.count))
            }
            guard let shared = SharedIOSurfaceBuffer(
                device: device, shape: storage.shape, data: floatArray
            ) else {
                return nil
            }
            sharedBuffers[name] = shared
            let array = try shared.makeMLMultiArray()
            return (array: array, shape: shared.shape)
        }

        // Try to create from CPU array
        if let cpuData = cpuArrays[name] {
            guard let shared = SharedIOSurfaceBuffer(
                device: device, shape: cpuData.shape, data: cpuData.data
            ) else {
                return nil
            }
            sharedBuffers[name] = shared
            let array = try shared.makeMLMultiArray()
            return (array: array, shape: shared.shape)
        }

        return nil
    }

    // MARK: - Readiness Tracking

    /// Marks a tensor as fully written and safe to read from any device.
    func markReady(name: String) {
        lock.lock()
        defer { lock.unlock() }
        readyTensors.insert(name)
    }

    /// Marks multiple tensors as ready atomically.
    func markReady(names: some Sequence<String>) {
        lock.lock()
        defer { lock.unlock() }
        for name in names {
            readyTensors.insert(name)
        }
    }

    /// Checks if a tensor has been marked as ready for consumption.
    func isReady(name: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return readyTensors.contains(name)
    }

    // MARK: - Query

    /// Checks if a tensor is available (in either GPU or CPU form).
    func hasTensor(name: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return gpuStorages[name] != nil || cpuArrays[name] != nil
    }

    /// Returns the tensor type for a stored tensor, if available.
    func tensorType(for name: String) -> TensorType? {
        lock.lock()
        defer { lock.unlock() }
        return tensorTypes[name]
    }

    // MARK: - Cleanup

    /// Clears all stored tensors and shared buffers.
    func clear() {
        lock.lock()
        defer { lock.unlock() }
        gpuStorages.removeAll()
        cpuArrays.removeAll()
        sharedBuffers.removeAll()
        tensorTypes.removeAll()
        readyTensors.removeAll()
    }
}

/// Errors during tensor transfer between devices.
enum TensorTransferError: Error, CustomStringConvertible {
    case tensorNotFound(String)
    case bufferAllocationFailed(String, Int)

    var description: String {
        switch self {
        case .tensorNotFound(let name):
            return "Tensor '\(name)' not found in transfer manager"
        case .bufferAllocationFailed(let name, let bytes):
            return "Failed to allocate \(bytes) byte buffer for tensor '\(name)'"
        }
    }
}
