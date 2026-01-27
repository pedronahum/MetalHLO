// KernelRegistry.swift
// MetalHLOCore
//
// Registry for custom Metal kernels with compilation caching.

import Metal
import Foundation

/// Registry for managing custom Metal compute kernels.
///
/// `MetalKernelRegistry` handles:
/// - Registration of custom kernels
/// - Runtime compilation of Metal shaders
/// - Caching of compiled pipeline states
/// - Specialization via function constants
/// - Shape-specialized kernel generation (Phase 2A)
public final class MetalKernelRegistry: @unchecked Sendable {

    // MARK: - Singleton

    /// Shared registry instance.
    public static let shared = MetalKernelRegistry()

    // MARK: - Properties

    /// Registered kernels by name.
    private var kernels: [String: any MetalKernel] = [:]

    /// Cached compiled pipeline states.
    private var compiledPipelines: [String: MTLComputePipelineState] = [:]

    /// Compiled Metal libraries by kernel name.
    private var compiledLibraries: [String: MTLLibrary] = [:]

    /// Kernel specializer for shape-specialized codegen.
    private var kernelSpecializer: KernelSpecializer?

    /// Cached specialized pipeline states.
    private var specializedPipelines: [String: MTLComputePipelineState] = [:]

    /// Lock for thread-safe access.
    private let lock = NSLock()

    // MARK: - Initialization

    private init() {
        // Register built-in kernels
        registerBuiltInKernels()
    }

    /// Creates a registry for a specific device (for testing).
    public init(device: MTLDevice) {
        registerBuiltInKernels()
        self.kernelSpecializer = KernelSpecializer(device: device)
    }

    /// Initializes the kernel specializer for a device.
    ///
    /// Call this before using shape-specialized kernels.
    /// - Parameter device: The Metal device.
    public func initializeSpecializer(for device: MTLDevice) {
        lock.lock()
        defer { lock.unlock() }
        if kernelSpecializer == nil {
            kernelSpecializer = KernelSpecializer(device: device)
        }
    }

    private func registerBuiltInKernels() {
        // Register all built-in custom kernels
        registerKernel(SoftmaxKernel())
        registerKernel(GELUKernel())
        registerKernel(LayerNormKernel())
        registerKernel(RMSNormKernel())
        registerKernel(FlashAttentionKernel())
    }

    // MARK: - Registration

    /// Registers a custom kernel.
    ///
    /// - Parameter kernel: The kernel to register.
    public func registerKernel(_ kernel: any MetalKernel) {
        lock.lock()
        defer { lock.unlock() }
        kernels[kernel.name] = kernel
    }

    /// Returns a registered kernel by name.
    ///
    /// - Parameter name: The kernel name.
    /// - Returns: The kernel, or nil if not found.
    public func getKernel(_ name: String) -> (any MetalKernel)? {
        lock.lock()
        defer { lock.unlock() }
        return kernels[name]
    }

    /// Returns all registered kernel names.
    public var registeredKernels: [String] {
        lock.lock()
        defer { lock.unlock() }
        return Array(kernels.keys)
    }

    // MARK: - Pipeline Compilation

    /// Gets or compiles a pipeline state for a kernel.
    ///
    /// - Parameters:
    ///   - kernel: The kernel to compile.
    ///   - device: The Metal device.
    ///   - specialization: Optional specialization constants.
    /// - Returns: The compiled pipeline state.
    /// - Throws: `KernelError` if compilation fails.
    public func getPipeline(
        for kernel: any MetalKernel,
        device: MTLDevice,
        specialization: KernelSpecialization = KernelSpecialization()
    ) throws -> MTLComputePipelineState {
        let cacheKey = "\(kernel.name)_\(device.name)_\(specialization.key)"

        lock.lock()
        if let cached = compiledPipelines[cacheKey] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        // Compile the shader
        let library = try getOrCompileLibrary(for: kernel, device: device)

        // Get the function with specialization
        let function: MTLFunction
        if specialization.options.isEmpty && specialization.dataType == "float" {
            // No specialization needed
            guard let fn = library.makeFunction(name: kernel.metalFunctionName) else {
                throw KernelError.functionNotFound(kernel.metalFunctionName)
            }
            function = fn
        } else {
            // Apply specialization constants
            let constants = specialization.toFunctionConstants()
            function = try library.makeFunction(
                name: kernel.metalFunctionName,
                constantValues: constants
            )
        }

        // Create pipeline state
        let pipeline = try device.makeComputePipelineState(function: function)

        // Cache the result
        lock.lock()
        compiledPipelines[cacheKey] = pipeline
        lock.unlock()

        return pipeline
    }

    /// Gets or compiles a Metal library for a kernel.
    private func getOrCompileLibrary(
        for kernel: any MetalKernel,
        device: MTLDevice
    ) throws -> MTLLibrary {
        let libraryKey = "\(kernel.name)_\(device.name)"

        lock.lock()
        if let cached = compiledLibraries[libraryKey] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        // Compile from source
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version3_1

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: kernel.shaderSource, options: options)
        } catch {
            throw KernelError.compilationFailed(kernel.name, error.localizedDescription)
        }

        lock.lock()
        compiledLibraries[libraryKey] = library
        lock.unlock()

        return library
    }

    // MARK: - Kernel Execution

    /// Executes a kernel with the given parameters.
    ///
    /// - Parameters:
    ///   - kernelName: Name of the registered kernel.
    ///   - inputs: Input Metal buffers.
    ///   - outputs: Output Metal buffers.
    ///   - params: Kernel parameters.
    ///   - device: Metal device.
    ///   - commandQueue: Command queue for execution.
    ///   - waitUntilCompleted: Whether to wait for execution to complete.
    /// - Throws: `KernelError` if kernel not found or execution fails.
    public func execute(
        kernelName: String,
        inputs: [MTLBuffer],
        outputs: [MTLBuffer],
        params: KernelParams,
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        waitUntilCompleted: Bool = true
    ) throws {
        guard let kernel = getKernel(kernelName) else {
            throw KernelError.kernelNotFound(kernelName)
        }

        let pipeline = try getPipeline(for: kernel, device: device)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw KernelError.commandBufferCreationFailed
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw KernelError.encoderCreationFailed
        }

        kernel.encode(
            into: encoder,
            inputs: inputs,
            outputs: outputs,
            params: params,
            pipeline: pipeline
        )

        encoder.endEncoding()
        commandBuffer.commit()

        if waitUntilCompleted {
            commandBuffer.waitUntilCompleted()

            if let error = commandBuffer.error {
                throw KernelError.executionFailed(error.localizedDescription)
            }
        }
    }

    // MARK: - Shape-Specialized Kernels

    /// Gets or generates a shape-specialized pipeline for the given operation.
    ///
    /// Shape-specialized kernels have compile-time constant dimensions,
    /// enabling better optimizations like loop unrolling and optimal tiling.
    ///
    /// - Parameters:
    ///   - op: The HLO operation kind.
    ///   - shapes: Input and output tensor shapes.
    ///   - config: Additional configuration parameters.
    ///   - device: The Metal device.
    /// - Returns: The compiled pipeline state, or nil if specialization not supported.
    /// - Throws: `KernelError` if compilation fails.
    public func getSpecializedPipeline(
        for op: HLOOpKind,
        shapes: [TensorType],
        config: [String: Int] = [:],
        device: MTLDevice
    ) throws -> MTLComputePipelineState? {
        // Initialize specializer if needed
        initializeSpecializer(for: device)

        guard let specializer = kernelSpecializer else {
            return nil
        }

        // Check if we should specialize
        guard specializer.shouldSpecialize(op: op, shapes: shapes) else {
            return nil
        }

        // Get the specialized kernel
        guard let kernel = specializer.getSpecializedKernel(op: op, shapes: shapes, config: config) else {
            return nil
        }

        // Check pipeline cache
        let cacheKey = "\(kernel.functionName)_\(device.name)"

        lock.lock()
        if let cached = specializedPipelines[cacheKey] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        // Compile the specialized kernel
        let pipeline = try specializer.compile(kernel, device: device)

        // Cache the result
        lock.lock()
        specializedPipelines[cacheKey] = pipeline
        lock.unlock()

        return pipeline
    }

    /// Gets a specialized matmul pipeline for exact dimensions.
    ///
    /// - Parameters:
    ///   - M: Rows of output.
    ///   - N: Columns of output.
    ///   - K: Contraction dimension.
    ///   - dtype: Data type.
    ///   - device: The Metal device.
    /// - Returns: The compiled pipeline state.
    /// - Throws: `KernelError` if compilation fails.
    public func getSpecializedMatMulPipeline(
        M: Int, N: Int, K: Int,
        dtype: ElementType = .float32,
        device: MTLDevice
    ) throws -> MTLComputePipelineState? {
        let shapes = [
            TensorType(shape: [M, K], elementType: dtype),
            TensorType(shape: [K, N], elementType: dtype)
        ]
        return try getSpecializedPipeline(for: .dot, shapes: shapes, device: device)
    }

    /// Gets a specialized attention pipeline for exact dimensions.
    ///
    /// - Parameters:
    ///   - batchSize: Batch size.
    ///   - numHeads: Number of attention heads.
    ///   - seqLen: Sequence length.
    ///   - headDim: Dimension per head.
    ///   - isCausal: Whether to use causal masking.
    ///   - dtype: Data type.
    ///   - device: The Metal device.
    /// - Returns: The compiled pipeline state.
    /// - Throws: `KernelError` if compilation fails.
    public func getSpecializedAttentionPipeline(
        batchSize: Int,
        numHeads: Int,
        seqLen: Int,
        headDim: Int,
        isCausal: Bool = true,
        dtype: ElementType = .float32,
        device: MTLDevice
    ) throws -> MTLComputePipelineState? {
        let qkvShape = TensorType(
            shape: [batchSize, numHeads, seqLen, headDim],
            elementType: dtype
        )
        let config: [String: Int] = [
            "is_attention": 1,
            "is_causal": isCausal ? 1 : 0
        ]
        return try getSpecializedPipeline(
            for: .customCall,
            shapes: [qkvShape, qkvShape, qkvShape],
            config: config,
            device: device
        )
    }

    /// Returns specialization statistics.
    public var specializationStatistics: SpecializationStats? {
        lock.lock()
        defer { lock.unlock() }
        return kernelSpecializer?.statistics
    }

    // MARK: - Cache Management

    /// Clears all cached pipelines and libraries.
    public func clearCache() {
        lock.lock()
        defer { lock.unlock() }
        compiledPipelines.removeAll()
        compiledLibraries.removeAll()
        specializedPipelines.removeAll()
        kernelSpecializer?.clearCache()
    }

    /// Precompiles all registered kernels for a device.
    ///
    /// - Parameter device: The Metal device.
    /// - Throws: `KernelError` if any compilation fails.
    public func precompileAll(for device: MTLDevice) throws {
        lock.lock()
        let kernelList = Array(kernels.values)
        lock.unlock()

        for kernel in kernelList {
            _ = try getPipeline(for: kernel, device: device)
        }
    }
}

// MARK: - Errors

/// Errors that can occur during kernel operations.
public enum KernelError: Error, Sendable, CustomStringConvertible {
    case kernelNotFound(String)
    case functionNotFound(String)
    case compilationFailed(String, String)
    case commandBufferCreationFailed
    case encoderCreationFailed
    case executionFailed(String)
    case invalidParameters(String)

    public var description: String {
        switch self {
        case .kernelNotFound(let name):
            return "Kernel not found: \(name)"
        case .functionNotFound(let name):
            return "Metal function not found: \(name)"
        case .compilationFailed(let kernel, let reason):
            return "Failed to compile kernel '\(kernel)': \(reason)"
        case .commandBufferCreationFailed:
            return "Failed to create command buffer"
        case .encoderCreationFailed:
            return "Failed to create compute encoder"
        case .executionFailed(let reason):
            return "Kernel execution failed: \(reason)"
        case .invalidParameters(let reason):
            return "Invalid kernel parameters: \(reason)"
        }
    }
}
