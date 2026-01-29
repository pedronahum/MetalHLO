// MetalHLOCompiler.swift
// MetalHLOCore
//
// Main compiler class that orchestrates the complete compilation pipeline.

import Foundation
import Metal

// MARK: - MetalHLO Compiler

/// The main compiler for MetalHLO.
///
/// `MetalHLOCompiler` orchestrates the complete compilation pipeline:
/// 1. Parse StableHLO MLIR
/// 2. Analyze the IR (shapes, dependencies, lifetimes, patterns)
/// 3. Optimize (simplification, canonicalization, fusion)
/// 4. Plan memory layout
/// 5. Generate Metal kernels
/// 6. Compile to pipeline states
/// 7. Package into executable
///
/// Example:
/// ```swift
/// let device = MTLCreateSystemDefaultDevice()!
/// let compiler = MetalHLOCompiler(device: device)
///
/// let mlir = """
/// module {
///   func.func @forward(%x: tensor<1x512x768xf32>) -> tensor<1x512x768xf32> {
///     %0 = stablehlo.tanh %x : tensor<1x512x768xf32>
///     return %0
///   }
/// }
/// """
///
/// let executable = try compiler.compile(mlir)
/// let executor = IntegratedExecutor(device: device, executable: executable)
/// let outputs = try executor.execute(inputs: ["x": inputBuffer])
/// ```
public final class MetalHLOCompiler: @unchecked Sendable {

    // MARK: - Properties

    /// Metal device for compilation.
    public let device: MTLDevice

    /// Compilation configuration.
    public let config: Config

    /// Compilation cache.
    private let cache: InternalCompilationCache

    /// Analyzer for computing analysis results.
    private let analyzer: Analyzer

    /// Pass manager for optimization.
    private let passManager: PassManager

    /// Memory planner.
    private let memoryPlanner: StaticMemoryPlanner

    /// Code generator.
    private let codeGenerator: CodeGenerator

    // MARK: - Configuration

    /// Configuration for the compiler.
    public struct Config: Sendable {
        /// Optimization level.
        public var optimizationLevel: OptimizationLevel

        /// Whether to enable caching.
        public var enableCaching: Bool

        /// Cache directory path.
        public var cacheDirectory: String?

        /// Whether to generate debug info.
        public var generateDebugInfo: Bool

        /// Maximum compilation timeout in seconds.
        public var timeoutSeconds: Double

        /// Pass manager configuration.
        public var passManagerConfig: PassManager.Config

        /// Code generator configuration.
        public var codeGeneratorConfig: CodeGenerator.Config

        /// Memory planner configuration.
        public var memoryPlannerConfig: StaticMemoryPlanner.Config

        public enum OptimizationLevel: Int, Sendable {
            case O0 = 0  // No optimization
            case O1 = 1  // Basic optimization
            case O2 = 2  // Standard optimization (default)
            case O3 = 3  // Aggressive optimization
        }

        public init(
            optimizationLevel: OptimizationLevel = .O2,
            enableCaching: Bool = true,
            cacheDirectory: String? = nil,
            generateDebugInfo: Bool = false,
            timeoutSeconds: Double = 60.0,
            passManagerConfig: PassManager.Config = .default,
            codeGeneratorConfig: CodeGenerator.Config = .default,
            memoryPlannerConfig: StaticMemoryPlanner.Config = .init()
        ) {
            self.optimizationLevel = optimizationLevel
            self.enableCaching = enableCaching
            self.cacheDirectory = cacheDirectory
            self.generateDebugInfo = generateDebugInfo
            self.timeoutSeconds = timeoutSeconds
            self.passManagerConfig = passManagerConfig
            self.codeGeneratorConfig = codeGeneratorConfig
            self.memoryPlannerConfig = memoryPlannerConfig
        }

        public static let `default` = Config()

        public static let debug = Config(
            optimizationLevel: .O0,
            generateDebugInfo: true,
            passManagerConfig: .debug
        )

        public static let release = Config(
            optimizationLevel: .O3,
            enableCaching: true
        )
    }

    // MARK: - Initialization

    /// Creates a new compiler.
    public init(device: MTLDevice, config: Config = .default) {
        self.device = device
        self.config = config
        self.cache = InternalCompilationCache()
        self.analyzer = Analyzer()
        self.passManager = PassManager(config: config.passManagerConfig)
        self.memoryPlanner = StaticMemoryPlanner(config: config.memoryPlannerConfig)
        self.codeGenerator = CodeGenerator(device: device, config: config.codeGeneratorConfig)
    }

    // MARK: - Compilation

    /// Compiles StableHLO MLIR to a compiled executable.
    ///
    /// - Parameter mlir: StableHLO MLIR source code.
    /// - Returns: A compiled executable ready for execution.
    /// - Throws: `CompilationError` if compilation fails.
    public func compile(_ mlir: String) throws -> CompiledExecutable {
        // Check cache first
        let hash = mlir.hashValue
        if config.enableCaching, let cached = cache.getCachedExecutable(hash: hash) {
            return cached
        }

        // ═══════════════════════════════════════════════════════════════
        // STAGE 1: PARSE
        // ═══════════════════════════════════════════════════════════════
        let module = try parseModule(mlir)

        // ═══════════════════════════════════════════════════════════════
        // STAGE 2: ANALYZE
        // ═══════════════════════════════════════════════════════════════
        let analysis = analyzer.analyze(module)

        // ═══════════════════════════════════════════════════════════════
        // STAGE 3: OPTIMIZE
        // ═══════════════════════════════════════════════════════════════
        let optimized = optimize(module, analysis: analysis)

        // ═══════════════════════════════════════════════════════════════
        // STAGE 4: PLAN MEMORY
        // ═══════════════════════════════════════════════════════════════
        let memoryPlan = planMemory(optimized)

        // ═══════════════════════════════════════════════════════════════
        // STAGE 5: GENERATE KERNELS
        // ═══════════════════════════════════════════════════════════════
        let kernelSpecs = codeGenerator.generate(module: optimized, memoryPlan: memoryPlan)

        // ═══════════════════════════════════════════════════════════════
        // STAGE 6: COMPILE METAL
        // ═══════════════════════════════════════════════════════════════
        let pipelines = try compileKernels(kernelSpecs)

        // ═══════════════════════════════════════════════════════════════
        // STAGE 7: PACKAGE
        // ═══════════════════════════════════════════════════════════════
        let executable = packageExecutable(
            pipelines: pipelines,
            kernelSpecs: kernelSpecs,
            memoryPlan: memoryPlan,
            optimized: optimized
        )

        // Cache the result
        if config.enableCaching {
            cache.cacheExecutable(hash: hash, executable: executable)
        }

        return executable
    }

    /// Compiles an HLO function directly (for testing).
    public func compile(function: HLOFunction) throws -> CompiledExecutable {
        let analysis = analyzer.analyze(function)
        let optimized = optimize(function, analysis: analysis)
        let memoryPlan = planMemory(optimized)
        let kernelSpecs = codeGenerator.generate(module: optimized, memoryPlan: memoryPlan)
        let pipelines = try compileKernels(kernelSpecs)

        return packageExecutable(
            pipelines: pipelines,
            kernelSpecs: kernelSpecs,
            memoryPlan: memoryPlan,
            optimized: optimized
        )
    }

    // MARK: - Pipeline Stages

    /// Parses MLIR to an HLO function.
    private func parseModule(_ mlir: String) throws -> HLOFunction {
        let parser = Parser(source: mlir)
        do {
            let module = try parser.parse()
            return module.function
        } catch {
            throw MetalCompilationError.parseError(error.localizedDescription)
        }
    }

    /// Optimizes the function based on optimization level.
    ///
    /// Pass selection is controlled by the PassManager configuration set in Client.swift:
    /// - O0: No passes (enabledPasses = empty set)
    /// - O1: Basic passes (constant-folding, algebraic-simplifier, dead-code-elimination)
    /// - O2: Standard passes (basic + CSE, canonicalizers, producer-consumer fusion)
    /// - O3: All passes with multiple iterations
    private func optimize(_ function: HLOFunction, analysis: AnalysisResults) -> OptimizedModule {
        switch config.optimizationLevel {
        case .O0:
            // No optimization - just convert to module
            return OptimizedModule.from(function)

        case .O1, .O2:
            // Run configured passes (controlled by PassManager.Config.enabledPasses)
            return passManager.runAllPasses(module: function, analysis: analysis)

        case .O3:
            // Aggressive optimization: multiple iterations until convergence
            var currentFunction = function
            var currentAnalysis = analysis

            for _ in 0..<3 {
                let module = passManager.runAllPasses(module: currentFunction, analysis: currentAnalysis)
                let newFunction = reconstructFunction(from: module)
                if newFunction.operations.count == currentFunction.operations.count {
                    break
                }
                currentFunction = newFunction
                currentAnalysis = analyzer.analyze(currentFunction)
            }

            return passManager.runAllPasses(module: currentFunction, analysis: currentAnalysis)
        }
    }

    /// Plans memory layout.
    private func planMemory(_ module: OptimizedModule) -> MemoryPlan {
        // Convert OptimizedModule to HLOFunction for memory planning
        let function = reconstructFunction(from: module)
        return memoryPlanner.plan(function)
    }

    /// Compiles kernel specs to Metal pipeline states.
    private func compileKernels(_ specs: [OpID: KernelSpec]) throws -> [OpID: MTLComputePipelineState] {
        var pipelines: [OpID: MTLComputePipelineState] = [:]

        for (opID, spec) in specs {
            let pipeline = try compileKernel(spec)
            pipelines[opID] = pipeline
        }

        return pipelines
    }

    /// Compiles a single kernel to a pipeline state.
    private func compileKernel(_ spec: KernelSpec) throws -> MTLComputePipelineState {
        // Check kernel cache
        let sourceHash = spec.metalSource.hashValue
        if let cached = cache.getCachedPipeline(hash: sourceHash) {
            return cached
        }

        // Compile Metal source
        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = config.codeGeneratorConfig.fastMath

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: spec.metalSource, options: compileOptions)
        } catch {
            throw MetalCompilationError.metalCompilationFailed(
                kernel: spec.entryPoint,
                error: error.localizedDescription
            )
        }

        // Get kernel function
        guard let function = library.makeFunction(name: spec.entryPoint) else {
            throw MetalCompilationError.kernelNotFound(spec.entryPoint)
        }

        // Create pipeline state
        let pipelineDescriptor = MTLComputePipelineDescriptor()
        pipelineDescriptor.computeFunction = function

        if spec.sharedMemorySize > 0 {
            pipelineDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        }

        let pipeline: MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(
                descriptor: pipelineDescriptor,
                options: [],
                reflection: nil
            )
        } catch {
            throw MetalCompilationError.pipelineCreationFailed(
                kernel: spec.entryPoint,
                error: error.localizedDescription
            )
        }

        // Cache the pipeline
        cache.cachePipeline(hash: sourceHash, pipeline: pipeline)

        return pipeline
    }

    /// Packages everything into a compiled executable.
    private func packageExecutable(
        pipelines: [OpID: MTLComputePipelineState],
        kernelSpecs: [OpID: KernelSpec],
        memoryPlan: MemoryPlan,
        optimized: OptimizedModule
    ) -> CompiledExecutable {
        // Build dispatch configs and shared memory sizes
        var dispatches: [OpID: DispatchConfig] = [:]
        var bindings: [OpID: [BufferBinding]] = [:]
        var sharedMemorySizes: [OpID: Int] = [:]

        for (opID, spec) in kernelSpecs {
            dispatches[opID] = spec.dispatch
            bindings[opID] = spec.bindings
            if spec.sharedMemorySize > 0 {
                sharedMemorySizes[opID] = spec.sharedMemorySize
            }
        }

        // Build input/output specs
        var inputSpecs: [String: TensorSpec] = [:]
        for input in optimized.inputs {
            inputSpecs[input.name] = TensorSpec(from: input)
        }

        var outputSpecs: [String: TensorSpec] = [:]
        for outputID in optimized.outputs {
            if let info = optimized.tensors[outputID] {
                outputSpecs[outputID] = TensorSpec(from: info)
            }
        }

        // Create Metal buffers for constants
        let constantBuffers = createConstantBuffers(optimized.constants, tensors: optimized.tensors)

        return CompiledExecutable(
            pipelines: pipelines,
            dispatches: dispatches,
            bindings: bindings,
            sharedMemorySizes: sharedMemorySizes,
            memoryPlan: memoryPlan,
            inputSpecs: inputSpecs,
            outputSpecs: outputSpecs,
            constantBuffers: constantBuffers
        )
    }

    /// Creates Metal buffers for constant values.
    private func createConstantBuffers(
        _ constants: [TensorID: ConstantValue],
        tensors: [TensorID: TensorInfo]
    ) -> [String: MTLBuffer] {
        var buffers: [String: MTLBuffer] = [:]

        for (tensorID, constantValue) in constants {
            guard let tensorInfo = tensors[tensorID] else { continue }

            let byteSize = tensorInfo.byteSize
            guard byteSize > 0 else { continue }

            // Serialize constant value to bytes
            let data = serializeConstant(constantValue, tensorInfo: tensorInfo)

            // Create Metal buffer
            if let buffer = device.makeBuffer(bytes: data, length: data.count, options: .storageModeShared) {
                buffer.label = "constant_\(tensorID)"
                buffers[tensorID] = buffer
            }
        }

        return buffers
    }

    /// Serializes a constant value to raw bytes.
    private func serializeConstant(_ value: ConstantValue, tensorInfo: TensorInfo) -> [UInt8] {
        let elementCount = tensorInfo.shape.isEmpty ? 1 : tensorInfo.shape.reduce(1, *)
        let elementSize = tensorInfo.elementType.byteSize

        switch value {
        case .scalar(let doubleValue):
            // Scalar value - replicate for all elements if tensor is not rank-0
            let scalarBytes = serializeScalar(doubleValue, elementType: tensorInfo.elementType)
            if elementCount == 1 {
                return scalarBytes
            }
            // Non-scalar tensor with single value = splat
            var result = [UInt8]()
            result.reserveCapacity(elementCount * elementSize)
            for _ in 0..<elementCount {
                result.append(contentsOf: scalarBytes)
            }
            return result

        case .splat(let doubleValue, _):
            // Same value repeated for all elements
            let scalarBytes = serializeScalar(doubleValue, elementType: tensorInfo.elementType)
            var result = [UInt8]()
            result.reserveCapacity(elementCount * elementSize)
            for _ in 0..<elementCount {
                result.append(contentsOf: scalarBytes)
            }
            return result

        case .dense(let values, _):
            // Dense array of values
            var result = [UInt8]()
            result.reserveCapacity(elementCount * elementSize)
            for i in 0..<min(values.count, elementCount) {
                result.append(contentsOf: serializeScalar(values[i], elementType: tensorInfo.elementType))
            }
            // Pad with zeros if needed
            while result.count < elementCount * elementSize {
                result.append(0)
            }
            return result
        }
    }

    /// Serializes a single scalar value to bytes.
    private func serializeScalar(_ value: Double, elementType: ElementType) -> [UInt8] {
        switch elementType {
        case .float32:
            var floatValue = Float(value)
            return withUnsafeBytes(of: &floatValue) { Array($0) }

        case .float64:
            var doubleValue = value
            return withUnsafeBytes(of: &doubleValue) { Array($0) }

        case .float16:
            // Convert to float16 (IEEE 754 half precision)
            let floatValue = Float(value)
            var halfValue = floatToHalf(floatValue)
            return withUnsafeBytes(of: &halfValue) { Array($0) }

        case .bfloat16:
            // BFloat16: truncate float32's lower 16 bits
            var floatValue = Float(value)
            var bfValue: UInt16 = 0
            withUnsafeBytes(of: &floatValue) { bytes in
                // Take the upper 16 bits of the float32
                bfValue = UInt16(bytes[2]) | (UInt16(bytes[3]) << 8)
            }
            return withUnsafeBytes(of: &bfValue) { Array($0) }

        case .int8:
            var intValue = Int8(clamping: Int(value))
            return withUnsafeBytes(of: &intValue) { Array($0) }

        case .int16:
            var intValue = Int16(clamping: Int(value))
            return withUnsafeBytes(of: &intValue) { Array($0) }

        case .int32:
            var intValue = Int32(clamping: Int(value))
            return withUnsafeBytes(of: &intValue) { Array($0) }

        case .int64:
            var intValue = Int64(value)
            return withUnsafeBytes(of: &intValue) { Array($0) }

        case .uint8:
            var intValue = UInt8(clamping: UInt(max(0, value)))
            return withUnsafeBytes(of: &intValue) { Array($0) }

        case .uint16:
            var intValue = UInt16(clamping: UInt(max(0, value)))
            return withUnsafeBytes(of: &intValue) { Array($0) }

        case .uint32:
            var intValue = UInt32(clamping: UInt(max(0, value)))
            return withUnsafeBytes(of: &intValue) { Array($0) }

        case .uint64:
            var intValue = UInt64(max(0, value))
            return withUnsafeBytes(of: &intValue) { Array($0) }

        case .int1:
            // int1 is used for booleans - store as single byte
            let boolValue: UInt8 = value != 0 ? 1 : 0
            return [boolValue]
        }
    }

    /// Converts float32 to float16 (IEEE 754).
    private func floatToHalf(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = (bits >> 31) & 0x1
        let exp = Int((bits >> 23) & 0xFF) - 127
        var mantissa = bits & 0x7FFFFF

        if exp == 128 {
            // Infinity or NaN
            return UInt16((sign << 15) | 0x7C00 | (mantissa != 0 ? 0x200 : 0))
        }

        if exp < -24 {
            // Too small, return zero
            return UInt16(sign << 15)
        }

        if exp < -14 {
            // Denormalized number
            mantissa |= 0x800000
            let shift = -14 - exp
            mantissa >>= shift
            return UInt16((sign << 15) | (mantissa >> 13))
        }

        if exp > 15 {
            // Overflow to infinity
            return UInt16((sign << 15) | 0x7C00)
        }

        // Normalized number
        let halfExp = UInt16(exp + 15)
        let halfMantissa = UInt16(mantissa >> 13)
        return UInt16((sign << 15) | (UInt32(halfExp) << 10) | UInt32(halfMantissa))
    }

    /// Reconstructs an HLOFunction from an OptimizedModule.
    private func reconstructFunction(from module: OptimizedModule) -> HLOFunction {
        let operations = module.operations.map { fusedOp -> HLOOperation in
            let resultType: TensorType
            if let output = fusedOp.outputs.first {
                resultType = TensorType(shape: output.shape, elementType: output.elementType)
            } else {
                resultType = TensorType(shape: [], elementType: .float32)
            }

            let kind: HLOOpKind
            switch fusedOp.type {
            case .original(let opKind):
                kind = opKind
            case .fusedAttention, .fusedMultiHeadAttention:
                kind = .customCall
            case .fusedRMSNorm, .fusedLayerNorm:
                kind = .customCall
            case .fusedMatMulBiasAct:
                kind = .dot
            case .fusedGELU, .fusedSiLU:
                kind = .customCall
            case .fusedElementwise:
                kind = .customCall
            case .fusedFFN:
                kind = .customCall
            case .fusedTransformerBlock:
                kind = .customCall
            case .fusedRoPE:
                kind = .customCall
            }

            return HLOOperation(
                result: fusedOp.id,
                kind: kind,
                operands: fusedOp.inputs,
                resultType: resultType,
                attributes: fusedOp.attributes
            )
        }

        let inputs = module.inputs.map { param in
            HLOArgument(name: param.name, type: param.type)
        }

        let outputTypes = module.outputs.compactMap { id -> TensorType? in
            guard let info = module.tensors[id] else { return nil }
            return TensorType(shape: info.shape, elementType: info.elementType)
        }

        return HLOFunction(
            name: "main",
            inputs: inputs,
            outputTypes: outputTypes,
            operations: operations,
            returnValues: module.outputs
        )
    }

    // MARK: - Statistics

    /// Returns compilation statistics from the last compilation.
    public var statistics: CompilationStatistics {
        CompilationStatistics(
            passManagerStats: passManager.statistics,
            cacheHits: cache.hitCount,
            cacheMisses: cache.missCount
        )
    }
}

// MARK: - Internal Compilation Cache

/// Internal cache for compiled pipelines and executables.
/// This is separate from the public CompilationCache in Runtime.
final class InternalCompilationCache: @unchecked Sendable {
    private var pipelineCache: [Int: MTLComputePipelineState] = [:]
    private var executableCache: [Int: CompiledExecutable] = [:]
    private let lock = NSLock()

    var hitCount: Int = 0
    var missCount: Int = 0

    func getCachedPipeline(hash: Int) -> MTLComputePipelineState? {
        lock.lock()
        defer { lock.unlock() }

        if let cached = pipelineCache[hash] {
            hitCount += 1
            return cached
        }
        missCount += 1
        return nil
    }

    func cachePipeline(hash: Int, pipeline: MTLComputePipelineState) {
        lock.lock()
        defer { lock.unlock() }
        pipelineCache[hash] = pipeline
    }

    func getCachedExecutable(hash: Int) -> CompiledExecutable? {
        lock.lock()
        defer { lock.unlock() }

        if let cached = executableCache[hash] {
            hitCount += 1
            return cached
        }
        missCount += 1
        return nil
    }

    func cacheExecutable(hash: Int, executable: CompiledExecutable) {
        lock.lock()
        defer { lock.unlock() }
        executableCache[hash] = executable
    }

    func clear() {
        lock.lock()
        defer { lock.unlock() }
        pipelineCache.removeAll()
        executableCache.removeAll()
        hitCount = 0
        missCount = 0
    }
}

// MARK: - Metal Compilation Errors

/// Errors that can occur during Metal kernel compilation.
public enum MetalCompilationError: Error, Sendable {
    case parseError(String)
    case metalCompilationFailed(kernel: String, error: String)
    case kernelNotFound(String)
    case pipelineCreationFailed(kernel: String, error: String)
    case timeout
    case invalidInput(String)
}

// MARK: - Compilation Statistics

/// Statistics from compilation.
public struct CompilationStatistics: Sendable {
    public let passManagerStats: PassManager.Statistics
    public let cacheHits: Int
    public let cacheMisses: Int

    public var cacheHitRate: Double {
        let total = cacheHits + cacheMisses
        return total > 0 ? Double(cacheHits) / Double(total) : 0
    }
}
