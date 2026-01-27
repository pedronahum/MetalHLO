// Client.swift
// MetalHLO
//
// Main entry point for compilation and buffer creation.

import Metal
import MetalHLOCore

/// The main entry point for MetalHLO operations.
///
/// `Client` manages the Metal device and provides methods for compiling
/// StableHLO MLIR programs and creating device buffers.
///
/// ## Example
/// ```swift
/// let client = try Client.create()
/// let executable = try client.compile(mlirString)
/// let buffer = try client.createBuffer([1.0, 2.0], shape: [2], elementType: .float32)
/// ```
public final class Client: @unchecked Sendable {

    // MARK: - Properties

    private let executor: MetalExecutor

    /// The underlying Metal device.
    public var device: MTLDevice {
        executor.device
    }

    /// The device name (e.g., "Apple M1 Pro").
    public var deviceName: String {
        device.name
    }

    // MARK: - Initialization

    private init(executor: MetalExecutor) {
        self.executor = executor
    }

    /// Creates a client for the default Metal device.
    ///
    /// - Throws: `MetalHLOError.noMetalDevice` if no Metal device is available.
    /// - Returns: A new `Client` instance.
    public static func create() throws -> Client {
        let executor = try MetalExecutor()
        return Client(executor: executor)
    }

    /// Creates a client for a specific Metal device.
    ///
    /// - Parameter device: The `MTLDevice` to use.
    /// - Throws: `MetalHLOError.unsupportedDevice` if the device doesn't support required features.
    /// - Returns: A new `Client` instance.
    public static func create(device: MTLDevice) throws -> Client {
        let executor = try MetalExecutor(device: device)
        return Client(executor: executor)
    }

    // MARK: - Compilation

    /// Compiles StableHLO MLIR text to an executable.
    ///
    /// This uses the default compilation path with standard optimization (O2).
    ///
    /// - Parameter mlir: The StableHLO MLIR module text.
    /// - Throws: `MetalHLOError.parseFailed` or `MetalHLOError.compilationFailed`.
    /// - Returns: A compiled `Executable` ready for execution.
    public func compile(_ mlir: String) throws -> Executable {
        // Parse MLIR to HLOModule
        let parser = Parser(source: mlir)
        let module: HLOModule
        do {
            module = try parser.parse()
        } catch let error as ParseError {
            let location = error.location ?? SourceLocation(line: 1, column: 1, offset: 0)
            throw MetalHLOError.parseFailed(
                line: location.line,
                column: location.column,
                message: error.description
            )
        }

        // Compile HLOModule to MPSGraph executable
        let compiled = try executor.compile(module: module)

        return Executable(compiled: compiled, executor: executor)
    }

    /// Compiles StableHLO MLIR text with explicit compilation configuration.
    ///
    /// This method uses the full MetalHLO compiler pipeline with the specified
    /// optimization level. Higher optimization levels enable more aggressive
    /// transformations like operator fusion and algebraic simplification.
    ///
    /// ## Example
    /// ```swift
    /// // Aggressive optimization for production
    /// let config = CompilationConfig(optimizationLevel: .O3)
    /// let exe = try client.compile(mlir, config: config)
    ///
    /// // Debug mode with no optimization
    /// let debugExe = try client.compile(mlir, config: .debug)
    /// ```
    ///
    /// - Parameters:
    ///   - mlir: The StableHLO MLIR module text.
    ///   - config: Compilation configuration specifying optimization level and options.
    /// - Throws: `MetalHLOError.parseFailed` or `MetalHLOError.compilationFailed`.
    /// - Returns: A compiled `Executable` ready for execution.
    public func compile(_ mlir: String, config: CompilationConfig) throws -> Executable {
        // Convert public config to internal compiler config
        let passManagerConfig = PassManager.Config(
            enabledPasses: config.enabledPasses,
            disabledPasses: config.disabledPasses
        )

        let compilerConfig = MetalHLOCompiler.Config(
            optimizationLevel: config.optimizationLevel.toCompilerLevel(),
            enableCaching: config.enableCaching,
            generateDebugInfo: config.generateDebugInfo,
            passManagerConfig: passManagerConfig
        )

        // Create compiler with the configuration
        let compiler = MetalHLOCompiler(device: device, config: compilerConfig)

        // Compile through the full optimization pipeline
        let compiled: CompiledExecutable
        do {
            compiled = try compiler.compile(mlir)
        } catch let error as MetalCompilationError {
            throw Self.convertCompilationError(error)
        }

        // Create integrated executor for the compiled executable
        let integratedExecutor = IntegratedExecutor(device: device, executable: compiled)

        return Executable(compiled: compiled, executor: integratedExecutor)
    }

    // MARK: - Error Conversion

    private static func convertCompilationError(_ error: MetalCompilationError) -> MetalHLOError {
        switch error {
        case .parseError(let message):
            return .parseFailed(line: 1, column: 1, message: message)
        case .metalCompilationFailed(let kernel, let errorMsg):
            return .compilationFailed("Metal compilation failed for kernel '\(kernel)': \(errorMsg)")
        case .kernelNotFound(let name):
            return .compilationFailed("Kernel not found: \(name)")
        case .pipelineCreationFailed(let kernel, let errorMsg):
            return .compilationFailed("Pipeline creation failed for kernel '\(kernel)': \(errorMsg)")
        case .timeout:
            return .compilationFailed("Compilation timeout")
        case .invalidInput(let reason):
            return .compilationFailed("Invalid input: \(reason)")
        }
    }

    // MARK: - Buffer Creation

    /// Creates a buffer from Float data (optimized fast path).
    ///
    /// This method bypasses type conversion and directly copies the bytes,
    /// providing ~100x faster buffer creation for large arrays.
    ///
    /// - Parameters:
    ///   - data: The Float data array.
    ///   - shape: The tensor shape.
    /// - Returns: A device buffer containing the data.
    public func createBuffer(
        _ data: [Float],
        shape: [Int]
    ) -> Buffer {
        let storage = executor.createBufferStorage(data, shape: shape)
        return Buffer(storage: storage)
    }

    /// Creates a buffer from Int32 data (optimized fast path).
    public func createBuffer(
        _ data: [Int32],
        shape: [Int]
    ) -> Buffer {
        let storage = executor.createBufferStorage(data, shape: shape)
        return Buffer(storage: storage)
    }

    /// Creates a buffer from Int64 data (optimized fast path).
    public func createBuffer(
        _ data: [Int64],
        shape: [Int]
    ) -> Buffer {
        let storage = executor.createBufferStorage(data, shape: shape)
        return Buffer(storage: storage)
    }

    /// Creates a buffer from host data (generic path with type conversion).
    ///
    /// - Parameters:
    ///   - data: The host data array.
    ///   - shape: The tensor shape.
    ///   - elementType: The element type.
    /// - Throws: `MetalHLOError.bufferCreationFailed` on failure.
    /// - Returns: A device buffer containing the data.
    public func createBuffer<T: Numeric>(
        _ data: [T],
        shape: [Int],
        elementType: ElementType
    ) throws -> Buffer {
        let coreElementType = elementType.toCoreType()
        let storage = try executor.createBufferStorage(data, shape: shape, elementType: coreElementType)
        return Buffer(storage: storage)
    }

    /// Creates a buffer from raw bytes.
    ///
    /// - Parameters:
    ///   - data: The raw byte data.
    ///   - shape: The tensor shape.
    ///   - elementType: The element type.
    /// - Throws: `MetalHLOError.bufferCreationFailed` on failure.
    /// - Returns: A device buffer containing the data.
    public func createBuffer(
        bytes data: Data,
        shape: [Int],
        elementType: ElementType
    ) throws -> Buffer {
        let coreElementType = elementType.toCoreType()
        let storage = try executor.createBufferStorage(bytes: data, shape: shape, elementType: coreElementType)
        return Buffer(storage: storage)
    }

    /// Creates an uninitialized buffer.
    ///
    /// - Parameters:
    ///   - shape: The tensor shape.
    ///   - elementType: The element type.
    /// - Throws: `MetalHLOError.bufferCreationFailed` on failure.
    /// - Returns: An uninitialized device buffer.
    public func createBuffer(
        shape: [Int],
        elementType: ElementType
    ) throws -> Buffer {
        let coreElementType = elementType.toCoreType()
        let storage = try executor.createBufferStorage(shape: shape, elementType: coreElementType)
        return Buffer(storage: storage)
    }
}
