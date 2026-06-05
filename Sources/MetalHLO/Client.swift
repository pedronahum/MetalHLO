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

    /// Emits the "-O3 under repair, falling back to -O2" warning exactly once
    /// per process (a static `let` initializer is run at most once, thread-safe).
    private static let o3FallbackWarningOnce: Bool = {
        FileHandle.standardError.write(Data(
            "[MetalHLO] warning: -O3 is experimental and under active repair; using -O2 instead. Set METALHLO_ALLOW_O3=1 to force -O3 (may crash or miscompile).\n".utf8))
        return true
    }()

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

        // Run pattern-based optimizer: fuses attention, softmax, GELU, layerNorm
        // before lowering to MPSGraph so the compiler sees fused custom_calls.
        // Restrict to pattern-only passes — structural transforms (TransposeMatmulFolding,
        // LayoutAssignment, SiblingFusion, HorizontalFusion) assume the integrated
        // executor and corrupt the graph for the MPSGraph backend.
        //
        // Skip optimization for large functions (>200 ops) to avoid optimizer bugs
        // that can incorrectly drop operations from complex graphs (e.g., LBM simulations
        // with scatter/gather/while patterns). The pattern fusions are designed for
        // transformer-like models and don't apply to these workloads anyway.
        let optimizedModule: HLOModule
        if module.function.operations.count > 200 {
            optimizedModule = module
        } else {
            let optimizerConfig = HLOOptimizerConfig(
                enableFusion: true,
                enableConstantFolding: false,
                enableProducerConsumerFusion: false,
                enableSiblingFusion: false,
                enableHorizontalFusion: false,
                enableLayoutAssignment: false,
                enableTransposeMatmulFolding: false,
                emitFusedCustomCalls: true
            )
            let optimizer = HLOOptimizer(config: optimizerConfig)
            let optimizedFunction = optimizer.optimize(module.function)
            // Preserve all functions (main + private helpers needed by call ops)
            var allFunctions = module.functions.filter { $0.isPrivate }
            allFunctions.insert(optimizedFunction, at: 0)
            optimizedModule = HLOModule(name: module.name, functions: allFunctions)
        }

        // Compile HLOModule to MPSGraph executable
        let compiled = try executor.compile(module: optimizedModule)

        return Executable(compiled: compiled, executor: executor)
    }

    /// Compiles StableHLO MLIR text with explicit compilation configuration.
    ///
    /// This method uses the full MetalHLO compiler pipeline with the specified
    /// optimization level. Higher optimization levels enable more aggressive
    /// transformations like operator fusion and algebraic simplification.
    ///
    /// When `devicePolicy` is not `.gpuOnly`, the compiler will analyze the
    /// computation graph and partition it between GPU and Apple Neural Engine
    /// for heterogeneous execution.
    ///
    /// ## Example
    /// ```swift
    /// // Aggressive optimization for production
    /// let config = CompilationConfig(optimizationLevel: .O3)
    /// let exe = try client.compile(mlir, config: config)
    ///
    /// // Heterogeneous GPU+ANE execution
    /// let hetConfig = CompilationConfig(devicePolicy: .auto)
    /// let hetExe = try client.compile(mlir, config: hetConfig)
    /// ```
    ///
    /// - Parameters:
    ///   - mlir: The StableHLO MLIR module text.
    ///   - config: Compilation configuration specifying optimization level and options.
    /// - Throws: `MetalHLOError.parseFailed` or `MetalHLOError.compilationFailed`.
    /// - Returns: A compiled `Executable` ready for execution.
    public func compile(_ mlir: String, config: CompilationConfig) throws -> Executable {
        // Check if heterogeneous execution is requested
        if config.devicePolicy != .gpuOnly {
            return try compileHeterogeneous(mlir, config: config)
        }

        // Convolution and scaled-dot-product attention run far faster on Apple's
        // MPSGraph kernels than on the codegen path (the codegen conv is a naive
        // 1-thread-per-output kernel; attention runs as separate matmuls + softmax
        // rather than MPSGraph's native `scaledDotProductAttention`). When the
        // graph contains either, upgrade to the heterogeneous path, which routes
        // those ops to MPSGraph (≈MLX parity, often faster) while keeping the fast
        // codegen kernels for everything else.
        let env = ProcessInfo.processInfo.environment

        // Attention = a (batched) dot_general feeding a softmax. `exponential`
        // identifies the softmax and distinguishes it from FFN activations
        // (ReLU = maximum, SiLU = logistic), so FFN graphs keep their codegen
        // fusion instead of being diverted to MPSGraph. The pattern optimizer in
        // the no-config compile fuses this into MPSGraph's native
        // `scaledDotProductAttention` (≈MLX parity, often faster). We route to
        // that pure-MPSGraph path rather than `.auto` because the heterogeneous
        // partitioner currently faults on some 4-D attention shapes.
        let hasAttention = env["METALHLO_ATTN_MPSGRAPH"] != "0"
            && mlir.contains("stablehlo.dot_general")
            && mlir.contains("stablehlo.exponential")
        if hasAttention {
            return try compile(mlir)
        }

        // The codegen conv kernel is a naive 1-thread-per-output direct conv
        // (~3-4x slower than MPSCNNConvolution). Upgrade conv-containing graphs to
        // the heterogeneous path, which routes conv to MPSGraph while keeping the
        // fast codegen kernels for everything else.
        if env["METALHLO_CONV_MPSGRAPH"] != "0", mlir.contains("stablehlo.convolution") {
            var hetConfig = config
            hetConfig.devicePolicy = .auto
            return try compileHeterogeneous(mlir, config: hetConfig)
        }

        // ─── O3 is UNDER REPAIR and gated off for users ──────────────────
        // -O3 (aggressivePasses) enables the pattern-fusion stack — but those
        // passes currently hit broken/incomplete code: a cross-layer
        // residual fusion that crashed the compiler, a fusedGelu custom_call
        // with a binding failure, and a fused-attention kernel that drops the
        // causal mask (wrong results). Until those are fixed, treat -O3 as -O2
        // so users get a correct (if unfused) compile. Developers working on
        // the fix opt in with METALHLO_ALLOW_O3=1.
        var optLevel = config.optimizationLevel
        if optLevel == .O3,
           ProcessInfo.processInfo.environment["METALHLO_ALLOW_O3"] != "1" {
            _ = Client.o3FallbackWarningOnce
            optLevel = .O2
        }

        // Map optimization level to appropriate pass set if user hasn't specified
        let effectiveEnabledPasses: Set<String>?
        if let userPasses = config.enabledPasses {
            // User explicitly specified passes, use those
            effectiveEnabledPasses = userPasses
        } else {
            // Map optimization level to pass set
            switch optLevel {
            case .O0:
                effectiveEnabledPasses = Set()  // No passes
            case .O1:
                effectiveEnabledPasses = OptimizationPass.basicPasses
            case .O2:
                effectiveEnabledPasses = OptimizationPass.standardPasses
            case .O3:
                effectiveEnabledPasses = OptimizationPass.aggressivePasses
            }
        }

        // Convert public config to internal compiler config
        let passManagerConfig = PassManager.Config(
            enabledPasses: effectiveEnabledPasses,
            disabledPasses: config.disabledPasses
        )

        let compilerConfig = MetalHLOCompiler.Config(
            optimizationLevel: optLevel.toCompilerLevel(),
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
        let integratedExecutor: IntegratedExecutor
        do {
            integratedExecutor = try IntegratedExecutor(device: device, executable: compiled)
        } catch let error as IntegratedExecutorError {
            throw Self.convertExecutorError(error)
        }

        return Executable(compiled: compiled, executor: integratedExecutor)
    }

    // MARK: - Heterogeneous Compilation

    /// Compiles with GPU+ANE heterogeneous execution.
    private func compileHeterogeneous(_ mlir: String, config: CompilationConfig) throws -> Executable {
        // Parse MLIR to get the HLO function
        let parser = Parser(source: mlir)
        let parsedModule: HLOModule
        do {
            parsedModule = try parser.parse()
        } catch let error as ParseError {
            let location = error.location ?? SourceLocation(line: 1, column: 1, offset: 0)
            throw MetalHLOError.parseFailed(
                line: location.line,
                column: location.column,
                message: error.description
            )
        }

        // Flatten any func.call into the entry function. The heterogeneous
        // executor runs a single function and has no cross-function call
        // support, so unresolved calls would silently produce no output (JAX
        // lowers composites like jnp.cumsum through nested private helpers).
        //
        // Some composites (jax.random.bits, jnp.searchsorted) lower through a
        // counted `while` whose body is a `func.call`. Mirror the gpuOnly fast
        // path's inline→unroll→inline sequence: a bare inline would leave the
        // `while` op (and its `%result#k` references) in the entry, which the
        // heterogeneous executor cannot run — every loop-carried result then
        // resolves to a missing input. Unrolling flattens the loop to
        // straight-line ops and the second inline flattens the body calls it
        // exposes. Each step is a no-op when its pattern is absent.
        let inlined1 = FunctionInliner.inline(parsedModule)
        let unrolled = WhileLoopUnroller.unroll(inlined1)
        let module = FunctionInliner.inline(unrolled)

        // If the entry still carries a `while` (a data-dependent loop the
        // unroller can't flatten), the heterogeneous executor has no runtime
        // control flow — route to the gpuOnly path, which compiles such
        // programs through the MPSGraph backend with native control flow.
        if module.function.operations.contains(where: { $0.kind == .whileOp }) {
            var gpuConfig = config
            gpuConfig.devicePolicy = .gpuOnly
            return try compile(mlir, config: gpuConfig)
        }

        // Analyze whether heterogeneous execution is beneficial
        let analyzer = ANEAnalyzer()
        let analysis = analyzer.analyzeFunction(module.function)

        // If no ops benefit from ANE, fall back to GPU-only
        if analysis.aneRecommendedOps.isEmpty && config.devicePolicy != .aneOnly {
            // Fall through to standard GPU compilation
            var gpuConfig = config
            gpuConfig.devicePolicy = .gpuOnly
            return try compile(mlir, config: gpuConfig)
        }

        // Create heterogeneous executor
        let hetExecutor = HeterogeneousExecutor(metalExecutor: executor)
        return Executable(function: module.function, executor: hetExecutor)
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

    private static func convertExecutorError(_ error: IntegratedExecutorError) -> MetalHLOError {
        switch error {
        case .commandQueueCreationFailed:
            return .executionFailed("Failed to create Metal command queue")
        case .bufferAllocationFailed(let size):
            return .bufferCreationFailed("Failed to allocate buffer of size \(size) bytes")
        case .commandBufferCreationFailed:
            return .executionFailed("Failed to create command buffer")
        case .encoderCreationFailed:
            return .executionFailed("Failed to create command encoder")
        case .missingPipeline(let opID):
            return .executionFailed("Missing pipeline for operation: \(opID)")
        case .missingDispatch(let opID):
            return .executionFailed("Missing dispatch for operation: \(opID)")
        case .missingBindings(let opID):
            return .executionFailed("Missing bindings for operation: \(opID)")
        case .missingInput(let name):
            return .executionFailed("Missing input: \(name)")
        case .missingConstant(let id):
            return .executionFailed("Missing constant: \(id)")
        case .invalidBinding(let reason):
            return .executionFailed("Invalid binding: \(reason)")
        case .executionFailed(let reason):
            return .executionFailed(reason)
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

    /// Creates a float32 buffer and fills it in-place via a closure.
    /// This avoids allocating an intermediate Swift array, which is critical
    /// for large tensors where a temporary copy would double memory usage.
    public func createBufferDirect(
        shape: [Int],
        fill: (UnsafeMutableRawBufferPointer) -> Void
    ) throws -> Buffer {
        let largeTensor = try LargeTensorStorage(device: device, shape: shape, elementType: .float32)
        largeTensor.withUnsafeMutableBytes { ptr in
            fill(ptr)
        }
        let storage = BufferStorage(largeTensor: largeTensor, device: device)
        return Buffer(storage: storage)
    }
}
