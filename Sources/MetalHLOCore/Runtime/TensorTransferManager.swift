// TensorTransferManager.swift
// MetalHLOCore
//
// Manages tensor data flow between GPU and ANE partitions during
// heterogeneous execution. On Apple Silicon, GPU and ANE share
// unified memory, so transfers are API conversions rather than
// physical memory copies.

import Metal

/// Manages intermediate tensor buffers during heterogeneous execution.
///
/// Tracks tensors by their SSA names and converts between GPU (`BufferStorage`)
/// and ANE (`[Float]`) representations as needed at partition boundaries.
final class TensorTransferManager: @unchecked Sendable {

    private let device: MTLDevice

    /// GPU-side tensor data (BufferStorage from MetalExecutor).
    private var gpuStorages: [String: BufferStorage] = [:]

    /// ANE-side tensor data (Float arrays from CoreMLBridge).
    private var cpuArrays: [String: (data: [Float], shape: [Int])] = [:]

    /// Tensor types for all stored tensors.
    private var tensorTypes: [String: TensorType] = [:]

    init(device: MTLDevice) {
        self.device = device
    }

    // MARK: - Store Results

    /// Stores a GPU BufferStorage result from a GPU partition.
    func storeGPUResult(name: String, storage: BufferStorage) {
        gpuStorages[name] = storage
        tensorTypes[name] = TensorType(shape: storage.shape, elementType: storage.elementType)
    }

    /// Stores a CPU array result from an ANE partition.
    func storeCPUResult(name: String, data: [Float], shape: [Int], elementType: ElementType = .float32) {
        cpuArrays[name] = (data: data, shape: shape)
        tensorTypes[name] = TensorType(shape: shape, elementType: elementType)
    }

    // MARK: - Retrieve as BufferStorage (for GPU partitions)

    /// Gets a tensor as BufferStorage for GPU partition inputs.
    /// If the tensor is only available as a CPU array, creates a BufferStorage from it.
    func getBufferStorage(name: String) throws -> BufferStorage {
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

    // MARK: - Query

    /// Checks if a tensor is available (in either GPU or CPU form).
    func hasTensor(name: String) -> Bool {
        gpuStorages[name] != nil || cpuArrays[name] != nil
    }

    /// Returns the tensor type for a stored tensor, if available.
    func tensorType(for name: String) -> TensorType? {
        tensorTypes[name]
    }

    // MARK: - Cleanup

    /// Clears all stored tensors.
    func clear() {
        gpuStorages.removeAll()
        cpuArrays.removeAll()
        tensorTypes.removeAll()
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
