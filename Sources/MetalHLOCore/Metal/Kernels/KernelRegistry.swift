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
    }

    private func registerBuiltInKernels() {
        // Register all built-in custom kernels
        registerKernel(SoftmaxKernel())
        registerKernel(GELUKernel())
        registerKernel(LayerNormKernel())
        registerKernel(RMSNormKernel())
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

    // MARK: - Cache Management

    /// Clears all cached pipelines and libraries.
    public func clearCache() {
        lock.lock()
        defer { lock.unlock() }
        compiledPipelines.removeAll()
        compiledLibraries.removeAll()
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
