// CodeGenerator.swift
// MetalHLOCore
//
// Generates Metal kernel specifications from optimized modules.

import Foundation
import Metal

// MARK: - Code Generator

/// Generates Metal kernel specifications for an optimized module.
public final class CodeGenerator: @unchecked Sendable {

    // MARK: - Properties

    private let device: MTLDevice
    private let config: Config

    // MARK: - Configuration

    public struct Config: Sendable {
        /// Target Metal feature set.
        public var metalFeatureSet: MetalFeatureSet

        /// Whether to generate debug comments in Metal source.
        public var debugComments: Bool

        /// Whether to use fast math.
        public var fastMath: Bool

        /// Default threadgroup size.
        public var defaultThreadgroupSize: Int

        /// Maximum threadgroup memory size.
        public var maxThreadgroupMemory: Int

        public enum MetalFeatureSet: String, Sendable {
            case apple7  // M1
            case apple8  // M2
            case apple9  // M3
            case common  // Conservative
        }

        public init(
            metalFeatureSet: MetalFeatureSet = .common,
            debugComments: Bool = false,
            fastMath: Bool = true,
            defaultThreadgroupSize: Int = 256,
            maxThreadgroupMemory: Int = 32768
        ) {
            self.metalFeatureSet = metalFeatureSet
            self.debugComments = debugComments
            self.fastMath = fastMath
            self.defaultThreadgroupSize = defaultThreadgroupSize
            self.maxThreadgroupMemory = maxThreadgroupMemory
        }

        public static let `default` = Config()
    }

    // MARK: - Initialization

    public init(device: MTLDevice, config: Config = .default) {
        self.device = device
        self.config = config
    }

    // MARK: - Generation

    /// Generates kernel specifications for all operations in the module.
    public func generate(module: OptimizedModule, memoryPlan: MemoryPlan) -> [OpID: KernelSpec] {
        var specs: [OpID: KernelSpec] = [:]

        // Track function input names for proper binding generation
        let inputNames = Set(module.inputs.map { $0.name })

        // Track constant tensor IDs
        let constantIDs = Set(module.constants.keys)

        for opID in memoryPlan.executionOrder {
            // opID is an integer index into the operations array
            guard opID >= 0 && opID < module.operations.count else { continue }
            let op = module.operations[opID]

            // Skip constant operations - they are pre-materialized into buffers
            if case .original(.constant) = op.type {
                continue
            }

            let spec = generateKernel(
                op: op,
                tensors: module.tensors,
                layouts: module.layouts,
                memoryPlan: memoryPlan,
                inputNames: inputNames,
                constantIDs: constantIDs
            )

            // Use string-converted integer as key to match executionOrder format
            specs[String(opID)] = spec
        }

        return specs
    }

    // MARK: - Type Mapping

    /// Maps ElementType to Metal type string.
    private func metalTypeName(for elementType: ElementType) -> String {
        switch elementType {
        case .float32: return "float"
        case .float16: return "half"
        case .float64: return "float"  // Metal doesn't support double in kernels, use float
        case .bfloat16: return "half"  // Approximate with half
        case .int8: return "char"
        case .int16: return "short"
        case .int32: return "int"
        case .int64: return "long"
        case .uint8: return "uchar"
        case .uint16: return "ushort"
        case .uint32: return "uint"
        case .uint64: return "ulong"
        case .int1: return "bool"
        }
    }

    /// Returns true if the element type is a floating-point type.
    private func isFloatType(_ elementType: ElementType) -> Bool {
        switch elementType {
        case .float16, .float32, .float64, .bfloat16:
            return true
        default:
            return false
        }
    }

    /// Generates a single kernel specification.
    private func generateKernel(
        op: FusedOp,
        tensors: [TensorID: TensorInfo],
        layouts: [TensorID: TensorLayout],
        memoryPlan: MemoryPlan,
        inputNames: Set<String>,
        constantIDs: Set<TensorID>
    ) -> KernelSpec {
        // Get shapes and element type
        let inputShapes = op.inputs.compactMap { tensors[$0]?.shape }
        let outputShapes = op.outputs.map { $0.shape }
        let elementType = op.outputs.first?.elementType ?? .float32

        // Generate Metal source and entry point
        let (source, entry, tuning) = generateSource(
            type: op.type,
            inputShapes: inputShapes,
            outputShapes: outputShapes,
            attributes: op.attributes,
            elementType: elementType
        )

        // Calculate dispatch configuration
        let dispatch = calculateDispatch(
            type: op.type,
            shapes: outputShapes,
            tuning: tuning
        )

        // Build buffer bindings
        let bindings = buildBindings(
            op: op,
            tensors: tensors,
            memoryPlan: memoryPlan,
            inputNames: inputNames,
            constantIDs: constantIDs
        )

        // Calculate shared memory size
        let sharedMemorySize = calculateSharedMemorySize(type: op.type, tuning: tuning, elementType: elementType)

        return KernelSpec(
            opID: op.id,
            metalSource: source,
            entryPoint: entry,
            dispatch: dispatch,
            bindings: bindings,
            tuning: tuning,
            sharedMemorySize: sharedMemorySize,
            inputShapes: inputShapes,
            outputShapes: outputShapes
        )
    }

    // MARK: - Source Generation

    /// Generates Metal source code for an operation.
    private func generateSource(
        type: FusedOpType,
        inputShapes: [[Int]],
        outputShapes: [[Int]],
        attributes: HLOAttributes,
        elementType: ElementType = .float32
    ) -> (source: String, entryPoint: String, tuning: TuningConfig?) {
        switch type {
        case .original(let opKind):
            return generateOriginalOpSource(opKind, inputShapes: inputShapes, outputShapes: outputShapes, attributes: attributes, elementType: elementType)

        case .fusedAttention(let config):
            return generateAttentionSource(config, inputShapes: inputShapes)

        case .fusedMultiHeadAttention(let config):
            return generateMultiHeadAttentionSource(config, inputShapes: inputShapes)

        case .fusedRMSNorm(let config):
            return generateRMSNormSource(config, inputShapes: inputShapes)

        case .fusedLayerNorm(let config):
            return generateLayerNormSource(config, inputShapes: inputShapes)

        case .fusedMatMulBiasAct(let config):
            return generateMatMulBiasActSource(config, inputShapes: inputShapes)

        case .fusedGELU(let approximate):
            return generateGELUSource(approximate: approximate, inputShapes: inputShapes)

        case .fusedSiLU:
            return generateSiLUSource(inputShapes: inputShapes)

        case .fusedElementwise(let ops):
            return generateElementwiseChainSource(ops, inputShapes: inputShapes)

        case .fusedFFN(let config):
            return generateFFNSource(config, inputShapes: inputShapes)

        case .fusedTransformerBlock(let config):
            return generateTransformerBlockSource(config, inputShapes: inputShapes)

        case .fusedRoPE(let config):
            return generateRoPESource(config, inputShapes: inputShapes)
        }
    }

    /// Generates source for original (unfused) operations.
    private func generateOriginalOpSource(
        _ opKind: HLOOpKind,
        inputShapes: [[Int]],
        outputShapes: [[Int]],
        attributes: HLOAttributes,
        elementType: ElementType = .float32
    ) -> (String, String, TuningConfig?) {
        let outputShape = outputShapes.first ?? []
        let totalElements = outputShape.reduce(1, *)
        let entryPoint = "kernel_\(opKind.rawValue)"
        let metalType = metalTypeName(for: elementType)
        let isFloat = isFloatType(elementType)

        var source = """
        #include <metal_stdlib>
        using namespace metal;

        """

        switch opKind {
        // Unary elementwise operations
        case .negate:
            source += generateUnaryKernel(entryPoint: entryPoint, operation: "-x", metalType: metalType)
        case .abs:
            source += generateUnaryKernel(entryPoint: entryPoint, operation: "abs(x)", metalType: metalType)
        case .exponential:
            // exp only works on float types - cast if needed
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "exp(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(exp(float(x)))", metalType: metalType)
            }
        case .log:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "log(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(log(float(x)))", metalType: metalType)
            }
        case .sqrt:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "sqrt(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(sqrt(float(x)))", metalType: metalType)
            }
        case .rsqrt:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "rsqrt(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(rsqrt(float(x)))", metalType: metalType)
            }
        case .sine:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "sin(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(sin(float(x)))", metalType: metalType)
            }
        case .cosine:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "cos(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(cos(float(x)))", metalType: metalType)
            }
        case .tanh:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "tanh(x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(tanh(float(x)))", metalType: metalType)
            }
        case .logistic:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "1.0f / (1.0f + exp(-x))", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(1.0f / (1.0f + exp(-float(x))))", metalType: metalType)
            }
        case .expm1:
            // exp(x) - 1
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "exp(x) - 1.0f", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(exp(float(x)) - 1.0f)", metalType: metalType)
            }
        case .log1p:
            // log(1 + x)
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "log(1.0f + x)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(log(1.0f + float(x)))", metalType: metalType)
            }
        case .cbrt:
            // Cube root: sign(x) * pow(abs(x), 1/3)
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "sign(x) * pow(abs(x), 0.333333333333f)", metalType: metalType)
            } else {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "\(metalType)(sign(float(x)) * pow(abs(float(x)), 0.333333333333f))", metalType: metalType)
            }
        case .floor:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "floor(x)", metalType: metalType)
            } else {
                // floor is a no-op for integers
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "x", metalType: metalType)
            }
        case .ceil:
            if isFloat {
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "ceil(x)", metalType: metalType)
            } else {
                // ceil is a no-op for integers
                source += generateUnaryKernel(entryPoint: entryPoint, operation: "x", metalType: metalType)
            }

        // Binary elementwise operations
        case .add:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a + b", metalType: metalType)
        case .subtract:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a - b", metalType: metalType)
        case .multiply:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a * b", metalType: metalType)
        case .divide:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "a / b", metalType: metalType)
        case .maximum:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "max(a, b)", metalType: metalType)
        case .minimum:
            source += generateBinaryKernel(entryPoint: entryPoint, operation: "min(a, b)", metalType: metalType)
        case .power:
            if isFloat {
                source += generateBinaryKernel(entryPoint: entryPoint, operation: "pow(a, b)", metalType: metalType)
            } else {
                source += generateBinaryKernel(entryPoint: entryPoint, operation: "\(metalType)(pow(float(a), float(b)))", metalType: metalType)
            }

        // Matrix operations
        case .dot, .dotGeneral:
            return generateMatMulSource(inputShapes: inputShapes, attributes: attributes, elementType: elementType)

        // Reduction operations
        case .reduce:
            return generateReductionSource(inputShapes: inputShapes, attributes: attributes)

        // Shape operations
        case .reshape:
            source += generateCopyKernel(entryPoint: entryPoint, metalType: metalType)

        case .transpose:
            source += generateTransposeKernel(
                entryPoint: entryPoint,
                inputShape: inputShapes.first ?? [],
                outputShape: outputShapes.first ?? [],
                permutation: attributes.dimensions ?? [],
                metalType: metalType
            )

        // Broadcast operations
        case .broadcastInDim:
            source += generateBroadcastKernel(
                entryPoint: entryPoint,
                inputShapes: inputShapes,
                outputShapes: outputShapes,
                attributes: attributes,
                metalType: metalType
            )

        default:
            // Fallback to copy kernel for unsupported ops
            source += generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        return (source, entryPoint, nil)
    }

    /// Generates a unary kernel with configurable element type.
    private func generateUnaryKernel(entryPoint: String, operation: String, metalType: String = "float") -> String {
        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            \(metalType) x = input[tid];
            output[tid] = \(operation);
        }
        """
    }

    /// Generates a binary kernel with configurable element type.
    private func generateBinaryKernel(entryPoint: String, operation: String, metalType: String = "float") -> String {
        return """
        kernel void \(entryPoint)(
            device const \(metalType)* inputA [[buffer(0)]],
            device const \(metalType)* inputB [[buffer(1)]],
            device \(metalType)* output [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            \(metalType) a = inputA[tid];
            \(metalType) b = inputB[tid];
            output[tid] = \(operation);
        }
        """
    }

    /// Generates a copy kernel with configurable element type.
    private func generateCopyKernel(entryPoint: String, metalType: String = "float") -> String {
        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            output[tid] = input[tid];
        }
        """
    }

    /// Generates a transpose kernel that permutes dimensions.
    ///
    /// The transpose operation reorders elements from input to output according to the permutation.
    /// For example, transpose([2,3,4], perm=[2,0,1]) means:
    /// - output dim 0 comes from input dim 2
    /// - output dim 1 comes from input dim 0
    /// - output dim 2 comes from input dim 1
    private func generateTransposeKernel(
        entryPoint: String,
        inputShape: [Int],
        outputShape: [Int],
        permutation: [Int],
        metalType: String = "float"
    ) -> String {
        // If no permutation or shapes are empty, fall back to copy
        guard !inputShape.isEmpty, !permutation.isEmpty else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        let rank = inputShape.count

        // Calculate input strides (row-major)
        var inputStrides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1]
        }

        // Calculate output strides (row-major)
        var outputStrides = [Int](repeating: 1, count: rank)
        for i in stride(from: rank - 2, through: 0, by: -1) {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }

        // Generate code to compute output coordinates from linear index
        var coordCode = "uint remaining = tid;\n"
        for i in stride(from: 0, to: rank, by: 1) {
            coordCode += "        uint coord\(i) = remaining / \(outputStrides[i]);\n"
            coordCode += "        remaining = remaining % \(outputStrides[i]);\n"
        }

        // Generate code to compute input linear index from permuted coordinates
        // output[coord0, coord1, ...] = input[coord_perm[0], coord_perm[1], ...]
        // So for input index, we use coord[permutation[i]] for each input dimension i
        var inputIndexCode = "uint inputIdx = "
        var terms: [String] = []
        for i in 0..<rank {
            let outputDim = permutation[i]  // Which output coordinate maps to input dim i
            if inputStrides[i] != 0 {
                terms.append("coord\(outputDim) * \(inputStrides[i])")
            }
        }
        inputIndexCode += terms.isEmpty ? "0" : terms.joined(separator: " + ")
        inputIndexCode += ";"

        let totalElements = outputShape.reduce(1, *)

        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;

            // Compute output coordinates from linear index
            \(coordCode)
            // Compute input index using permuted coordinates
            \(inputIndexCode)

            output[tid] = input[inputIdx];
        }
        """
    }

    /// Generates a broadcast kernel that maps elements from a smaller input to a larger output.
    private func generateBroadcastKernel(
        entryPoint: String,
        inputShapes: [[Int]],
        outputShapes: [[Int]],
        attributes: HLOAttributes,
        metalType: String = "float"
    ) -> String {
        guard let inputShape = inputShapes.first,
              let outputShape = outputShapes.first else {
            return generateCopyKernel(entryPoint: entryPoint, metalType: metalType)
        }

        // Get broadcast dimensions from attributes (dimensions attribute for broadcast_in_dim)
        let broadcastDims = attributes.dimensions ?? []

        // Calculate output strides for index computation
        var outputStrides = [Int](repeating: 1, count: outputShape.count)
        for i in stride(from: outputShape.count - 2, through: 0, by: -1) {
            outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1]
        }

        // Calculate input strides
        var inputStrides = [Int](repeating: 1, count: inputShape.count)
        for i in stride(from: inputShape.count - 2, through: 0, by: -1) {
            inputStrides[i] = inputStrides[i + 1] * inputShape[i + 1]
        }

        // Build index computation code
        var indexCode = ""
        if inputShape.isEmpty || inputShape == [1] || (inputShape.count == 1 && inputShape[0] == 1) {
            // Scalar broadcast - all elements read from same location
            indexCode = "uint inputIdx = 0;"
        } else if broadcastDims.isEmpty {
            // No explicit dimensions - assume trailing dimensions match
            // This is a simple element-by-element broadcast
            indexCode = """
            uint inputIdx = 0;
            uint remaining = tid;
            """
            for (outDim, outSize) in outputShape.enumerated().reversed() {
                let coord = "remaining % \(outSize)"
                indexCode += "\n    uint coord\(outDim) = \(coord);"
                indexCode += "\n    remaining /= \(outSize);"

                // Check if this output dimension maps to an input dimension
                let inputDim = outDim - (outputShape.count - inputShape.count)
                if inputDim >= 0 && inputDim < inputShape.count {
                    if inputShape[inputDim] == 1 {
                        // Broadcast dimension - use 0
                        indexCode += "\n    // dim \(outDim) broadcasts from input dim \(inputDim)"
                    } else {
                        indexCode += "\n    inputIdx += coord\(outDim) * \(inputStrides[inputDim]);"
                    }
                }
            }
        } else {
            // Explicit broadcast dimensions mapping
            indexCode = """
            uint inputIdx = 0;
            uint remaining = tid;
            """
            for (outDim, outSize) in outputShape.enumerated().reversed() {
                let coord = "remaining % \(outSize)"
                indexCode += "\n    uint coord\(outDim) = \(coord);"
                indexCode += "\n    remaining /= \(outSize);"
            }

            // Map output coordinates to input coordinates using broadcast_dimensions
            for (inputDim, outputDim) in broadcastDims.enumerated() {
                if inputDim < inputShape.count && outputDim < outputShape.count {
                    if inputShape[inputDim] == 1 {
                        // Broadcast dimension
                        indexCode += "\n    // input dim \(inputDim) broadcasts (size 1)"
                    } else {
                        indexCode += "\n    inputIdx += coord\(outputDim) * \(inputStrides[inputDim]);"
                    }
                }
            }
        }

        let outputCount = outputShape.reduce(1, *)

        return """
        kernel void \(entryPoint)(
            device const \(metalType)* input [[buffer(0)]],
            device \(metalType)* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            \(indexCode)
            output[tid] = input[inputIdx];
        }
        """
    }

    /// Generates matmul source with support for all numeric types.
    private func generateMatMulSource(inputShapes: [[Int]], attributes: HLOAttributes, elementType: ElementType = .float32) -> (String, String, TuningConfig?) {
        guard inputShapes.count >= 2 else {
            let metalType = metalTypeName(for: elementType)
            return (generateCopyKernel(entryPoint: "kernel_matmul", metalType: metalType), "kernel_matmul", nil)
        }

        let lhsShape = inputShapes[0]
        let metalType = metalTypeName(for: elementType)
        let isFloat = isFloatType(elementType)

        // Calculate batch size (product of all dimensions except last 2)
        let batchSize = lhsShape.count > 2 ? lhsShape.dropLast(2).reduce(1, *) : 1

        let tuning = TuningConfig.matmul

        // Zero value depends on type
        let zeroValue = isFloat ? "0.0f" : "0"

        // Use batched kernel if we have batch dimensions
        let source: String
        if batchSize > 1 {
            source = """
            #include <metal_stdlib>
            using namespace metal;

            #define TILE_SIZE 32

            // Batched matrix multiplication kernel
            // Handles tensors of shape [...batch_dims, M, K] x [...batch_dims, K, N] -> [...batch_dims, M, N]
            // Element type: \(metalType)
            kernel void kernel_matmul(
                device const \(metalType)* A [[buffer(0)]],
                device const \(metalType)* B [[buffer(1)]],
                device \(metalType)* C [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                constant uint& batchCount [[buffer(6)]],
                threadgroup \(metalType)* tileA [[threadgroup(0)]],
                threadgroup \(metalType)* tileB [[threadgroup(1)]],
                uint3 gid [[threadgroup_position_in_grid]],
                uint3 tid [[thread_position_in_threadgroup]])
            {
                uint batch = gid.z;
                if (batch >= batchCount) return;

                uint row = gid.y * TILE_SIZE + tid.y;
                uint col = gid.x * TILE_SIZE + tid.x;

                // Calculate offsets for this batch
                uint matrixSizeA = M * K;
                uint matrixSizeB = K * N;
                uint matrixSizeC = M * N;

                device const \(metalType)* batchA = A + batch * matrixSizeA;
                device const \(metalType)* batchB = B + batch * matrixSizeB;
                device \(metalType)* batchC = C + batch * matrixSizeC;

                \(metalType) sum = \(zeroValue);

                for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
                    // Load tile of A
                    uint aRow = row;
                    uint aCol = t * TILE_SIZE + tid.x;
                    if (aRow < M && aCol < K) {
                        tileA[tid.y * TILE_SIZE + tid.x] = batchA[aRow * K + aCol];
                    } else {
                        tileA[tid.y * TILE_SIZE + tid.x] = \(zeroValue);
                    }

                    // Load tile of B
                    uint bRow = t * TILE_SIZE + tid.y;
                    uint bCol = col;
                    if (bRow < K && bCol < N) {
                        tileB[tid.y * TILE_SIZE + tid.x] = batchB[bRow * N + bCol];
                    } else {
                        tileB[tid.y * TILE_SIZE + tid.x] = \(zeroValue);
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // Compute partial sum
                    for (uint k = 0; k < TILE_SIZE; k++) {
                        sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (row < M && col < N) {
                    batchC[row * N + col] = sum;
                }
            }
            """
        } else {
            source = """
            #include <metal_stdlib>
            using namespace metal;

            #define TILE_SIZE 32

            // Matrix multiplication kernel
            // Element type: \(metalType)
            kernel void kernel_matmul(
                device const \(metalType)* A [[buffer(0)]],
                device const \(metalType)* B [[buffer(1)]],
                device \(metalType)* C [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                threadgroup \(metalType)* tileA [[threadgroup(0)]],
                threadgroup \(metalType)* tileB [[threadgroup(1)]],
                uint2 gid [[threadgroup_position_in_grid]],
                uint2 tid [[thread_position_in_threadgroup]])
            {
                uint row = gid.y * TILE_SIZE + tid.y;
                uint col = gid.x * TILE_SIZE + tid.x;

                \(metalType) sum = \(zeroValue);

                for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
                    // Load tile of A
                    uint aRow = row;
                    uint aCol = t * TILE_SIZE + tid.x;
                    if (aRow < M && aCol < K) {
                        tileA[tid.y * TILE_SIZE + tid.x] = A[aRow * K + aCol];
                    } else {
                        tileA[tid.y * TILE_SIZE + tid.x] = \(zeroValue);
                    }

                    // Load tile of B
                    uint bRow = t * TILE_SIZE + tid.y;
                    uint bCol = col;
                    if (bRow < K && bCol < N) {
                        tileB[tid.y * TILE_SIZE + tid.x] = B[bRow * N + bCol];
                    } else {
                        tileB[tid.y * TILE_SIZE + tid.x] = \(zeroValue);
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // Compute partial sum
                    for (uint k = 0; k < TILE_SIZE; k++) {
                        sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (row < M && col < N) {
                    C[row * N + col] = sum;
                }
            }
            """
        }

        return (source, "kernel_matmul", tuning)
    }

    /// Generates reduction source.
    private func generateReductionSource(inputShapes: [[Int]], attributes: HLOAttributes) -> (String, String, TuningConfig?) {
        // Get reduction dimensions from attributes
        let reduceDims = attributes.dimensions ?? [0]

        // Get input shape
        let inputShape = inputShapes.first ?? []

        // Determine reduction operation based on reductionKind
        let reductionKind = attributes.reductionKind ?? .sum
        let (accumOp, initComment): (String, String)
        switch reductionKind {
        case .sum:
            accumOp = "accum += input[inputIdx];"
            initComment = "// Start with init value (typically 0 for sum)"
        case .max:
            accumOp = "accum = max(accum, input[inputIdx]);"
            initComment = "// Start with init value (typically -FLT_MAX for max)"
        case .min:
            accumOp = "accum = min(accum, input[inputIdx]);"
            initComment = "// Start with init value (typically FLT_MAX for min)"
        case .mean:
            // Mean reduction: sum then divide by count (handled as sum here, divide done separately)
            accumOp = "accum += input[inputIdx];"
            initComment = "// Start with init value (0 for mean accumulation)"
        }

        // NOTE: reduce operation has two inputs: data (buffer 0) and init value (buffer 1)
        // Output is at buffer 2, followed by scalar parameters

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        // Reduction kernel: each thread handles one output element
        // and sequentially reduces over the reduction dimension
        // Reduction type: \(reductionKind)
        kernel void kernel_reduce(
            device const float* input [[buffer(0)]],
            device const float* initValue [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& outputCount [[buffer(3)]],
            constant uint& reduceSize [[buffer(4)]],
            constant uint& innerSize [[buffer(5)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= outputCount) return;

            // Compute which output element this thread handles
            // output layout: [outer..., inner...]
            // input layout: [outer..., reduce, inner...]
            uint outerIdx = tid / innerSize;
            uint innerIdx = tid % innerSize;

            \(initComment)
            float accum = initValue[0];

            // Reduce over the reduction dimension
            for (uint r = 0; r < reduceSize; r++) {
                uint inputIdx = outerIdx * reduceSize * innerSize + r * innerSize + innerIdx;
                \(accumOp)
            }

            output[tid] = accum;
        }
        """

        return (source, "kernel_reduce", TuningConfig(blockSize: 256))
    }

    /// Generates attention source.
    private func generateAttentionSource(_ config: AttentionConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        // Flash Attention kernel
        kernel void kernel_attention(
            device const float* Q [[buffer(0)]],
            device const float* K [[buffer(1)]],
            device const float* V [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint& seqLen [[buffer(4)]],
            constant uint& headDim [[buffer(5)]],
            constant float& scale [[buffer(6)]],
            uint3 gid [[thread_position_in_grid]])
        {
            uint batch = gid.z;
            uint head = gid.y;
            uint queryIdx = gid.x;

            if (queryIdx >= seqLen) return;

            uint headOffset = (batch * \(config.numHeads) + head) * seqLen * headDim;

            // Compute attention scores and output for this query position
            float maxScore = -INFINITY;
            float sumExp = 0.0f;

            // First pass: find max score
            for (uint k = 0; k < seqLen; k++) {
                \(config.causalMask ? "if (k > queryIdx) continue;" : "")
                float score = 0.0f;
                for (uint d = 0; d < headDim; d++) {
                    score += Q[headOffset + queryIdx * headDim + d] * K[headOffset + k * headDim + d];
                }
                score *= scale;
                maxScore = max(maxScore, score);
            }

            // Second pass: compute softmax denominator
            for (uint k = 0; k < seqLen; k++) {
                \(config.causalMask ? "if (k > queryIdx) continue;" : "")
                float score = 0.0f;
                for (uint d = 0; d < headDim; d++) {
                    score += Q[headOffset + queryIdx * headDim + d] * K[headOffset + k * headDim + d];
                }
                score = exp((score * scale) - maxScore);
                sumExp += score;
            }

            // Third pass: compute weighted sum
            for (uint d = 0; d < headDim; d++) {
                float acc = 0.0f;
                for (uint k = 0; k < seqLen; k++) {
                    \(config.causalMask ? "if (k > queryIdx) continue;" : "")
                    float score = 0.0f;
                    for (uint dd = 0; dd < headDim; dd++) {
                        score += Q[headOffset + queryIdx * headDim + dd] * K[headOffset + k * headDim + dd];
                    }
                    score = exp((score * scale) - maxScore) / sumExp;
                    acc += score * V[headOffset + k * headDim + d];
                }
                output[headOffset + queryIdx * headDim + d] = acc;
            }
        }
        """

        return (source, "kernel_attention", TuningConfig.attention)
    }

    /// Generates multi-head attention source.
    private func generateMultiHeadAttentionSource(_ config: MultiHeadAttentionConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        return generateAttentionSource(
            AttentionConfig(
                numHeads: config.numHeads,
                headDim: config.headDim,
                causalMask: config.causalMask
            ),
            inputShapes: inputShapes
        )
    }

    /// Generates RMSNorm source.
    private func generateRMSNormSource(_ config: NormConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_rms_norm(
            device const float* input [[buffer(0)]],
            device const float* weight [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& hiddenSize [[buffer(3)]],
            constant float& epsilon [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]])
        {
            uint batch = gid.y;
            uint offset = batch * hiddenSize;

            // Compute RMS
            float sumSq = 0.0f;
            for (uint i = 0; i < hiddenSize; i++) {
                float x = input[offset + i];
                sumSq += x * x;
            }
            float rms = rsqrt(sumSq / float(hiddenSize) + \(config.epsilon));

            // Normalize and scale
            for (uint i = gid.x; i < hiddenSize; i += \(256)) {
                output[offset + i] = input[offset + i] * rms * weight[i];
            }
        }
        """

        return (source, "kernel_rms_norm", TuningConfig(blockSize: 256))
    }

    /// Generates LayerNorm source.
    private func generateLayerNormSource(_ config: NormConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_layer_norm(
            device const float* input [[buffer(0)]],
            device const float* gamma [[buffer(1)]],
            device const float* beta [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint& hiddenSize [[buffer(4)]],
            constant float& epsilon [[buffer(5)]],
            uint2 gid [[thread_position_in_grid]])
        {
            uint batch = gid.y;
            uint offset = batch * hiddenSize;

            // Compute mean
            float sum = 0.0f;
            for (uint i = 0; i < hiddenSize; i++) {
                sum += input[offset + i];
            }
            float mean = sum / float(hiddenSize);

            // Compute variance
            float varSum = 0.0f;
            for (uint i = 0; i < hiddenSize; i++) {
                float diff = input[offset + i] - mean;
                varSum += diff * diff;
            }
            float invStd = rsqrt(varSum / float(hiddenSize) + \(config.epsilon));

            // Normalize and apply affine transform
            for (uint i = gid.x; i < hiddenSize; i += \(256)) {
                float normalized = (input[offset + i] - mean) * invStd;
                output[offset + i] = normalized * gamma[i] + beta[i];
            }
        }
        """

        return (source, "kernel_layer_norm", TuningConfig(blockSize: 256))
    }

    /// Generates MatMul + Bias + Activation source.
    private func generateMatMulBiasActSource(_ config: MatMulConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let activationCode: String
        if let activation = config.activation {
            switch activation {
            case .relu: activationCode = "max(x, 0.0f)"
            case .gelu: activationCode = "x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
            case .silu: activationCode = "x / (1.0f + exp(-x))"
            case .tanh: activationCode = "tanh(x)"
            case .sigmoid: activationCode = "1.0f / (1.0f + exp(-x))"
            case .geluApproximate: activationCode = "x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
            case .none: activationCode = "x"
            }
        } else {
            activationCode = "x"
        }

        let biasCode = config.hasBias ? "result += bias[col];" : ""

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        #define TILE_SIZE 32

        kernel void kernel_matmul_bias_act(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],
            device const float* bias [[buffer(2)]],
            device float* C [[buffer(3)]],
            constant uint& M [[buffer(4)]],
            constant uint& N [[buffer(5)]],
            constant uint& K [[buffer(6)]],
            uint2 gid [[thread_position_in_grid]])
        {
            uint row = gid.y;
            uint col = gid.x;

            if (row >= M || col >= N) return;

            float sum = 0.0f;
            for (uint k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }

            float result = sum;
            \(biasCode)

            float x = result;
            C[row * N + col] = \(activationCode);
        }
        """

        return (source, "kernel_matmul_bias_act", TuningConfig.matmul)
    }

    /// Generates GELU source.
    private func generateGELUSource(approximate: Bool, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let geluCode = approximate
            ? "x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
            : "x * 0.5f * (1.0f + erf(x * 0.7071067811865475f))"

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_gelu(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            float x = input[tid];
            output[tid] = \(geluCode);
        }
        """

        return (source, "kernel_gelu", nil)
    }

    /// Generates SiLU source.
    private func generateSiLUSource(inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_silu(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            float x = input[tid];
            output[tid] = x / (1.0f + exp(-x));
        }
        """

        return (source, "kernel_silu", nil)
    }

    /// Generates elementwise chain source.
    ///
    /// Handles both unary and binary operations. For binary ops, consumes inputs in order.
    /// Example: ops = [add, multiply] with inputs [a, b, c]
    ///   x = a + b (consumes inputs 0 and 1)
    ///   x = x * c (uses previous result and input 2)
    ///
    /// IMPORTANT: The kernel buffer indices MUST match what buildBindings() provides:
    /// - Input buffers at indices 0..<inputShapes.count
    /// - Output buffer at index inputShapes.count
    /// - Count scalar at index inputShapes.count + 1
    private func generateElementwiseChainSource(_ ops: [HLOOpKind], inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        // Use the actual number of inputs passed to the operation.
        // This MUST match op.inputs.count used by buildBindings() to ensure
        // buffer indices are aligned between kernel signature and buffer bindings.
        let inputCount = max(inputShapes.count, 1)

        // Generate kernel parameters for all inputs
        var inputParams: [String] = []
        for i in 0..<inputCount {
            inputParams.append("device const float* input\(i) [[buffer(\(i))]]")
        }

        // Output buffer comes after inputs (must match buildBindings layout)
        let outputBufferIndex = inputCount

        // Generate the operation code
        var code = ""
        var inputIndex = 0
        var hasX = false

        for op in ops {
            if isBinaryOp(op) {
                let left: String
                let right: String

                if hasX {
                    left = "x"
                    right = "input\(inputIndex)[tid]"
                    inputIndex += 1
                } else {
                    left = "input\(inputIndex)[tid]"
                    inputIndex += 1
                    right = "input\(inputIndex)[tid]"
                    inputIndex += 1
                }

                code += generateBinaryOpCode(op, left: left, right: right, declareX: !hasX)
                hasX = true
            } else {
                // Unary operation
                if !hasX {
                    code += "float x = input\(inputIndex)[tid];\n"
                    inputIndex += 1
                    hasX = true
                }
                code += generateUnaryOpCode(op)
            }
        }

        // If no operations, just copy input
        if !hasX && inputCount > 0 {
            code = "float x = input0[tid];\n"
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_elementwise_chain(
            \(inputParams.joined(separator: ",\n            ")),
            device float* output [[buffer(\(outputBufferIndex))]],
            constant uint& count [[buffer(\(outputBufferIndex + 1))]],
            uint tid [[thread_position_in_grid]])
        {
            if (tid >= count) return;
            \(code)
            output[tid] = x;
        }
        """

        return (source, "kernel_elementwise_chain", nil)
    }

    /// Checks if an operation is binary.
    private func isBinaryOp(_ op: HLOOpKind) -> Bool {
        switch op {
        case .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .and, .or, .xor:
            return true
        default:
            return false
        }
    }

    /// Generates code for a binary operation.
    private func generateBinaryOpCode(_ op: HLOOpKind, left: String, right: String, declareX: Bool) -> String {
        let prefix = declareX ? "float x = " : "x = "
        switch op {
        case .add:
            return "\(prefix)\(left) + \(right);\n"
        case .subtract:
            return "\(prefix)\(left) - \(right);\n"
        case .multiply:
            return "\(prefix)\(left) * \(right);\n"
        case .divide:
            return "\(prefix)\(left) / \(right);\n"
        case .maximum:
            return "\(prefix)max(\(left), \(right));\n"
        case .minimum:
            return "\(prefix)min(\(left), \(right));\n"
        case .power:
            return "\(prefix)pow(\(left), \(right));\n"
        default:
            return "\(prefix)\(left);\n"  // Fallback
        }
    }

    /// Generates code for a unary operation.
    private func generateUnaryOpCode(_ op: HLOOpKind) -> String {
        switch op {
        case .negate: return "x = -x;\n"
        case .abs: return "x = abs(x);\n"
        case .exponential: return "x = exp(x);\n"
        case .log: return "x = log(x);\n"
        case .sqrt: return "x = sqrt(x);\n"
        case .rsqrt: return "x = rsqrt(x);\n"
        case .tanh: return "x = tanh(x);\n"
        case .logistic: return "x = 1.0f / (1.0f + exp(-x));\n"
        case .sine: return "x = sin(x);\n"
        case .cosine: return "x = cos(x);\n"
        case .floor: return "x = floor(x);\n"
        case .ceil: return "x = ceil(x);\n"
        default: return ""
        }
    }

    /// Generates FFN source.
    private func generateFFNSource(_ config: FFNConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        // For now, generate separate kernels - full fusion would be more complex
        return generateMatMulBiasActSource(
            MatMulConfig(activation: config.activation),
            inputShapes: inputShapes
        )
    }

    /// Generates transformer block source.
    /// Mega-kernel that fuses: pre-norm -> attention -> residual -> pre-norm -> FFN -> residual
    private func generateTransformerBlockSource(_ config: TransformerBlockConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let numHeads = config.attention.numHeads
        let headDim = config.attention.headDim
        let hiddenDim = numHeads * headDim  // Derive hidden dim from attention config
        let intermediateDim = config.ffn.intermediateDim
        let isCausal = config.attention.causalMask ? 1 : 0

        // Activation code for FFN
        let activationCode: String
        switch config.ffn.activation {
        case .relu: activationCode = "max(x, 0.0f)"
        case .gelu, .geluApproximate:
            activationCode = "x * 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
        case .silu: activationCode = "x / (1.0f + exp(-x))"
        case .tanh: activationCode = "tanh(x)"
        case .sigmoid: activationCode = "1.0f / (1.0f + exp(-x))"
        case .none: activationCode = "x"
        }

        // Norm code (RMSNorm or LayerNorm)
        let normFunction: String
        if config.normType == .rmsNorm {
            normFunction = """
            inline void apply_norm(
                thread float* normed,
                device const float* input,
                device const float* weight,
                uint offset,
                uint hiddenDim,
                float epsilon
            ) {
                float sumSq = 0.0f;
                for (uint i = 0; i < hiddenDim; i++) {
                    float v = input[offset + i];
                    sumSq += v * v;
                }
                float rms = rsqrt(sumSq / float(hiddenDim) + epsilon);
                for (uint i = 0; i < hiddenDim; i++) {
                    normed[i] = input[offset + i] * rms * weight[i];
                }
            }
            """
        } else {
            normFunction = """
            inline void apply_norm(
                thread float* normed,
                device const float* input,
                device const float* weight,
                device const float* bias,
                uint offset,
                uint hiddenDim,
                float epsilon
            ) {
                float mean = 0.0f;
                for (uint i = 0; i < hiddenDim; i++) {
                    mean += input[offset + i];
                }
                mean /= float(hiddenDim);
                float var_sum = 0.0f;
                for (uint i = 0; i < hiddenDim; i++) {
                    float diff = input[offset + i] - mean;
                    var_sum += diff * diff;
                }
                float invStd = rsqrt(var_sum / float(hiddenDim) + epsilon);
                for (uint i = 0; i < hiddenDim; i++) {
                    normed[i] = (input[offset + i] - mean) * invStd * weight[i] + bias[i];
                }
            }
            """
        }

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        \(normFunction)

        // Transformer Block Mega-Kernel
        // Fuses: PreNorm -> Attention -> Residual -> PreNorm -> FFN -> Residual
        kernel void kernel_transformer_block(
            device const float* input [[buffer(0)]],           // [batch, seq, hidden]
            device const float* attnNormWeight [[buffer(1)]],  // [hidden]
            device const float* Wq [[buffer(2)]],              // [hidden, hidden]
            device const float* Wk [[buffer(3)]],              // [hidden, hidden]
            device const float* Wv [[buffer(4)]],              // [hidden, hidden]
            device const float* Wo [[buffer(5)]],              // [hidden, hidden]
            device const float* ffnNormWeight [[buffer(6)]],   // [hidden]
            device const float* W1 [[buffer(7)]],              // [hidden, intermediate]
            device const float* W2 [[buffer(8)]],              // [intermediate, hidden]
            device float* output [[buffer(9)]],                // [batch, seq, hidden]
            constant uint& batchSize [[buffer(10)]],
            constant uint& seqLen [[buffer(11)]],
            constant float& scale [[buffer(12)]],
            constant float& epsilon [[buffer(13)]],
            threadgroup float* shared_q [[threadgroup(0)]],    // [block_size, head_dim]
            threadgroup float* shared_k [[threadgroup(1)]],    // [seq_len, head_dim]
            threadgroup float* shared_v [[threadgroup(2)]],    // [seq_len, head_dim]
            uint3 threadgroup_id [[threadgroup_position_in_grid]],
            uint3 thread_id [[thread_position_in_threadgroup]],
            uint3 threads_per_group [[threads_per_threadgroup]]
        ) {
            uint batch = threadgroup_id.z;
            uint head = threadgroup_id.y;
            uint q_block = threadgroup_id.x;

            if (batch >= batchSize) return;

            const uint hiddenDim = \(hiddenDim);
            const uint numHeads = \(numHeads);
            const uint headDim = \(headDim);
            const uint intermediateDim = \(intermediateDim);
            const uint isCausal = \(isCausal);
            const uint blockSize = 64;

            uint local_q = thread_id.x;
            uint global_q = q_block * blockSize + local_q;
            if (global_q >= seqLen) return;

            uint headOffset = head * headDim;
            uint inputOffset = (batch * seqLen + global_q) * hiddenDim;

            // ========== Pre-Attention Norm ==========
            float normed[64];  // Max hidden dim per head
            float sumSq = 0.0f;
            for (uint i = 0; i < hiddenDim; i++) {
                float v = input[inputOffset + i];
                sumSq += v * v;
            }
            float rms = rsqrt(sumSq / float(hiddenDim) + epsilon);
            for (uint i = 0; i < hiddenDim; i++) {
                normed[i] = input[inputOffset + i] * rms * attnNormWeight[i];
            }

            // ========== Q, K, V Projections ==========
            float Q[64], K[64], V[64];
            for (uint d = 0; d < headDim; d++) {
                float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
                uint out_idx = headOffset + d;
                for (uint i = 0; i < hiddenDim; i++) {
                    q_sum += normed[i] * Wq[i * hiddenDim + out_idx];
                    k_sum += normed[i] * Wk[i * hiddenDim + out_idx];
                    v_sum += normed[i] * Wv[i * hiddenDim + out_idx];
                }
                Q[d] = q_sum;
                K[d] = k_sum;
                V[d] = v_sum;
            }

            // Store Q in shared memory for this block
            for (uint d = 0; d < headDim; d++) {
                shared_q[local_q * headDim + d] = Q[d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ========== Self-Attention with Online Softmax ==========
            float m_i = -INFINITY;
            float l_i = 0.0f;
            float o_acc[64];
            for (uint d = 0; d < headDim; d++) o_acc[d] = 0.0f;

            // Iterate over all K/V positions
            for (uint k_pos = 0; k_pos < seqLen; k_pos++) {
                if (isCausal && k_pos > global_q) break;

                // Load K, V for k_pos (would be from shared memory in full implementation)
                float k_val[64], v_val[64];
                uint k_input_offset = (batch * seqLen + k_pos) * hiddenDim;

                // Compute K, V for this position
                for (uint d = 0; d < headDim; d++) {
                    float k_sum = 0.0f, v_sum = 0.0f;
                    uint out_idx = headOffset + d;
                    // Simplified: recompute projection (in production, would cache)
                    for (uint i = 0; i < min(hiddenDim, 64u); i++) {
                        float inp = input[k_input_offset + i] * rms * attnNormWeight[i];
                        k_sum += inp * Wk[i * hiddenDim + out_idx];
                        v_sum += inp * Wv[i * hiddenDim + out_idx];
                    }
                    k_val[d] = k_sum;
                    v_val[d] = v_sum;
                }

                // Compute attention score
                float score = 0.0f;
                for (uint d = 0; d < headDim; d++) {
                    score += Q[d] * k_val[d];
                }
                score *= scale;

                // Online softmax update
                float m_new = max(m_i, score);
                float exp_diff_old = exp(m_i - m_new);
                float exp_diff_new = exp(score - m_new);

                for (uint d = 0; d < headDim; d++) {
                    o_acc[d] = o_acc[d] * exp_diff_old + exp_diff_new * v_val[d];
                }
                l_i = l_i * exp_diff_old + exp_diff_new;
                m_i = m_new;
            }

            // Normalize attention output
            float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            for (uint d = 0; d < headDim; d++) {
                o_acc[d] *= l_inv;
            }

            // Output projection
            float attn_out[64];
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                float sum = 0.0f;
                for (uint d = 0; d < headDim; d++) {
                    sum += o_acc[d] * Wo[(headOffset + d) * hiddenDim + i];
                }
                attn_out[i] = sum;
            }

            // ========== Attention Residual ==========
            float after_attn[64];
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                after_attn[i] = input[inputOffset + i] + attn_out[i];
            }

            // ========== Pre-FFN Norm ==========
            sumSq = 0.0f;
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                sumSq += after_attn[i] * after_attn[i];
            }
            float rms2 = rsqrt(sumSq / float(hiddenDim) + epsilon);
            float ffn_in[64];
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                ffn_in[i] = after_attn[i] * rms2 * ffnNormWeight[i];
            }

            // ========== FFN ==========
            // Up projection + activation
            float intermediate[64];
            for (uint j = 0; j < min(intermediateDim, 64u); j++) {
                float sum = 0.0f;
                for (uint i = 0; i < hiddenDim && i < 64; i++) {
                    sum += ffn_in[i] * W1[i * intermediateDim + j];
                }
                float x = sum;
                intermediate[j] = \(activationCode);
            }

            // Down projection
            float ffn_out[64];
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                float sum = 0.0f;
                for (uint j = 0; j < min(intermediateDim, 64u); j++) {
                    sum += intermediate[j] * W2[j * hiddenDim + i];
                }
                ffn_out[i] = sum;
            }

            // ========== FFN Residual + Output ==========
            uint outputOffset = inputOffset;
            for (uint i = 0; i < hiddenDim && i < 64; i++) {
                output[outputOffset + i] = after_attn[i] + ffn_out[i];
            }
        }
        """

        let tuning = TuningConfig(
            tileM: 64,
            tileN: 64,
            tileK: 64,
            blockSize: 64,
            useSharedMemory: true,
            useSIMDGroups: true
        )

        return (source, "kernel_transformer_block", tuning)
    }

    /// Generates RoPE source.
    private func generateRoPESource(_ config: RoPEConfig, inputShapes: [[Int]]) -> (String, String, TuningConfig?) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_rope(
            device const float* input [[buffer(0)]],
            device const float* cos_cache [[buffer(1)]],
            device const float* sin_cache [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint& seqLen [[buffer(4)]],
            constant uint& headDim [[buffer(5)]],
            uint3 gid [[thread_position_in_grid]])
        {
            uint pos = gid.x;
            uint dim = gid.y;

            if (pos >= seqLen || dim >= headDim / 2) return;

            uint idx1 = pos * headDim + dim * 2;
            uint idx2 = idx1 + 1;

            float x1 = input[idx1];
            float x2 = input[idx2];

            float cos_val = cos_cache[pos * (headDim / 2) + dim];
            float sin_val = sin_cache[pos * (headDim / 2) + dim];

            output[idx1] = x1 * cos_val - x2 * sin_val;
            output[idx2] = x1 * sin_val + x2 * cos_val;
        }
        """

        return (source, "kernel_rope", nil)
    }

    // MARK: - Dispatch Calculation

    /// Calculates dispatch configuration for an operation.
    private func calculateDispatch(type: FusedOpType, shapes: [[Int]], tuning: TuningConfig?) -> DispatchConfig {
        guard let outputShape = shapes.first else {
            return DispatchConfig.dispatch1D(elements: 1)
        }

        let totalElements = outputShape.reduce(1, *)

        switch type {
        case .original(let opKind):
            switch opKind {
            case .dot, .dotGeneral:
                if outputShape.count >= 2 {
                    let M = outputShape[outputShape.count - 2]
                    let N = outputShape[outputShape.count - 1]
                    // Calculate batch size for batched matmul
                    let batchSize = outputShape.count > 2 ? outputShape.dropLast(2).reduce(1, *) : 1

                    if batchSize > 1 {
                        // 3D dispatch for batched matmul
                        let tileSize = 32
                        let gridWidth = (N + tileSize - 1) / tileSize
                        let gridHeight = (M + tileSize - 1) / tileSize
                        return DispatchConfig(
                            gridSize: MTLSize(width: gridWidth, height: gridHeight, depth: batchSize),
                            threadgroupSize: MTLSize(width: tileSize, height: tileSize, depth: 1)
                        )
                    } else {
                        return DispatchConfig.dispatch2D(width: N, height: M, tileWidth: 32, tileHeight: 32)
                    }
                }
            case .reduce:
                return DispatchConfig.dispatch1D(elements: totalElements, threadgroupSize: 256)
            default:
                break
            }

        case .fusedAttention, .fusedMultiHeadAttention:
            if outputShape.count >= 3 {
                let batch = outputShape[0]
                let heads = outputShape.count > 3 ? outputShape[1] : 1
                let seqLen = outputShape[outputShape.count - 2]
                return DispatchConfig(
                    gridSize: MTLSize(width: seqLen, height: heads, depth: batch),
                    threadgroupSize: MTLSize(width: 1, height: 1, depth: 1)
                )
            }

        case .fusedRMSNorm, .fusedLayerNorm:
            if outputShape.count >= 2 {
                let batch = outputShape.dropLast().reduce(1, *)
                return DispatchConfig(
                    gridSize: MTLSize(width: (256 + 255) / 256, height: batch, depth: 1),
                    threadgroupSize: MTLSize(width: 256, height: 1, depth: 1)
                )
            }

        case .fusedMatMulBiasAct:
            if outputShape.count >= 2 {
                let M = outputShape[outputShape.count - 2]
                let N = outputShape[outputShape.count - 1]
                return DispatchConfig.dispatch2D(width: N, height: M)
            }

        default:
            break
        }

        return DispatchConfig.dispatch1D(elements: totalElements)
    }

    // MARK: - Buffer Bindings

    /// Builds buffer bindings for an operation.
    private func buildBindings(
        op: FusedOp,
        tensors: [TensorID: TensorInfo],
        memoryPlan: MemoryPlan,
        inputNames: Set<String>,
        constantIDs: Set<TensorID>
    ) -> [BufferBinding] {
        var bindings: [BufferBinding] = []
        var index = 0

        // Input bindings
        for inputID in op.inputs {
            let source: BufferSource
            let size = tensors[inputID]?.byteSize ?? 0

            // Check if this is a constant (pre-materialized into constant buffer)
            if constantIDs.contains(inputID) {
                source = .constant(id: inputID)
            }
            // Check if this is a function input (comes from external buffer)
            else if inputNames.contains(inputID) {
                source = .input(name: inputID)
            } else if let offset = memoryPlan.tensorOffsets[inputID] {
                // Intermediate result in the unified buffer
                source = .unified(offset: offset)
            } else {
                // Fallback: treat as input (shouldn't happen normally)
                source = .input(name: inputID)
            }

            bindings.append(BufferBinding(
                index: index,
                source: source,
                size: size,
                access: .read
            ))
            index += 1
        }

        // Output bindings
        for output in op.outputs {
            let source: BufferSource
            let size = output.byteSize

            if let offset = memoryPlan.tensorOffsets[output.id] {
                source = .unified(offset: offset)
            } else {
                source = .output(name: output.id)
            }

            bindings.append(BufferBinding(
                index: index,
                source: source,
                size: size,
                access: .write
            ))
            index += 1
        }

        // Add scalar parameter bindings for operations that need them
        // (e.g., element count for elementwise operations)
        if let count = scalarCountForOperation(op) {
            bindings.append(BufferBinding(
                index: index,
                source: .scalar(count),
                size: MemoryLayout<UInt32>.size,
                access: .read
            ))
            index += 1
        }

        // Add M, N, K dimension parameters for matmul operations
        if case .original(let opKind) = op.type, (opKind == .dot || opKind == .dotGeneral) {
            // Get input shapes to determine M, N, K
            let inputShapes = op.inputs.compactMap { tensors[$0]?.shape }
            if inputShapes.count >= 2 {
                let lhsShape = inputShapes[0]
                let rhsShape = inputShapes[1]

                // M = rows of A, K = cols of A = rows of B, N = cols of B
                let M = lhsShape.count >= 2 ? UInt32(lhsShape[lhsShape.count - 2]) : 1
                let K = lhsShape.count >= 1 ? UInt32(lhsShape[lhsShape.count - 1]) : 1
                let N = rhsShape.count >= 1 ? UInt32(rhsShape[rhsShape.count - 1]) : 1

                // Calculate batch size for batched matmul
                let batchSize = lhsShape.count > 2 ? UInt32(lhsShape.dropLast(2).reduce(1, *)) : 1

                bindings.append(BufferBinding(
                    index: index,
                    source: .scalar(M),
                    size: MemoryLayout<UInt32>.size,
                    access: .read
                ))
                index += 1

                bindings.append(BufferBinding(
                    index: index,
                    source: .scalar(N),
                    size: MemoryLayout<UInt32>.size,
                    access: .read
                ))
                index += 1

                bindings.append(BufferBinding(
                    index: index,
                    source: .scalar(K),
                    size: MemoryLayout<UInt32>.size,
                    access: .read
                ))
                index += 1

                // Add batchCount for batched matmul (kernel uses this when batchSize > 1)
                if batchSize > 1 {
                    bindings.append(BufferBinding(
                        index: index,
                        source: .scalar(batchSize),
                        size: MemoryLayout<UInt32>.size,
                        access: .read
                    ))
                }
            }
        }

        // Add reduce operation parameters: outputCount, reduceSize, innerSize
        if case .original(let opKind) = op.type, opKind == .reduce {
            let inputShapes = op.inputs.compactMap { tensors[$0]?.shape }
            guard let inputShape = inputShapes.first, !inputShape.isEmpty else {
                return bindings
            }

            // Get reduction dimensions from attributes (first input is data, second is init value)
            let reduceDims = op.attributes.dimensions ?? [inputShape.count - 1]

            // Calculate outputCount, reduceSize, and innerSize
            // For now, handle reduction along a single dimension (most common case)
            // The output shape removes the reduction dimensions
            let outputShape = op.outputs.first?.shape ?? []
            let outputCount = outputShape.isEmpty ? 1 : outputShape.reduce(1, *)

            // Calculate reduceSize (product of reduction dimensions)
            var reduceSize = 1
            for dim in reduceDims {
                if dim >= 0 && dim < inputShape.count {
                    reduceSize *= inputShape[dim]
                }
            }

            // innerSize is the product of dimensions after all reduction dimensions
            // For simple last-dim reduction: innerSize = 1
            // For general case: we need to handle the layout properly
            let maxReduceDim = reduceDims.max() ?? (inputShape.count - 1)
            var innerSize = 1
            for i in (maxReduceDim + 1)..<inputShape.count {
                innerSize *= inputShape[i]
            }

            // If reduction is along last dimension(s), innerSize = 1
            if innerSize == 0 { innerSize = 1 }

            bindings.append(BufferBinding(
                index: index,
                source: .scalar(UInt32(outputCount)),
                size: MemoryLayout<UInt32>.size,
                access: .read
            ))
            index += 1

            bindings.append(BufferBinding(
                index: index,
                source: .scalar(UInt32(reduceSize)),
                size: MemoryLayout<UInt32>.size,
                access: .read
            ))
            index += 1

            bindings.append(BufferBinding(
                index: index,
                source: .scalar(UInt32(innerSize)),
                size: MemoryLayout<UInt32>.size,
                access: .read
            ))
        }

        return bindings
    }

    /// Determines if an operation needs a scalar count parameter and returns it.
    private func scalarCountForOperation(_ op: FusedOp) -> UInt32? {
        switch op.type {
        case .original(let opKind):
            // Elementwise operations need count parameter
            switch opKind {
            case .add, .subtract, .multiply, .divide, .maximum, .minimum, .power,
                 .negate, .abs, .exponential, .log, .sqrt, .rsqrt, .ceil, .floor, .roundNearestEven,
                 .sine, .cosine, .tanh, .sign, .logistic, .expm1, .log1p, .cbrt,
                 .and, .or, .xor, .not,
                 .shiftLeft, .shiftRightArithmetic, .shiftRightLogical,
                 .clamp, .select,
                 .compare,
                 .convert,
                 .popcnt,
                 .bitcastConvert,
                 // Shape operations that use copy/broadcast kernels
                 .reshape, .transpose, .broadcastInDim:
                // Calculate total elements from output shape
                if let output = op.outputs.first {
                    let count = output.shape.isEmpty ? 1 : output.shape.reduce(1, *)
                    return UInt32(count)
                }
            default:
                return nil
            }

        case .fusedGELU, .fusedSiLU:
            if let output = op.outputs.first {
                let count = output.shape.isEmpty ? 1 : output.shape.reduce(1, *)
                return UInt32(count)
            }

        case .fusedElementwise:
            if let output = op.outputs.first {
                let count = output.shape.isEmpty ? 1 : output.shape.reduce(1, *)
                return UInt32(count)
            }

        default:
            return nil
        }

        return nil
    }

    // MARK: - Shared Memory

    /// Returns the byte size of an element type.
    private func elementByteSize(for elementType: ElementType) -> Int {
        switch elementType {
        case .float64, .int64, .uint64:
            return 8
        case .float32, .int32, .uint32:
            return 4
        case .float16, .bfloat16, .int16, .uint16:
            return 2
        case .int8, .uint8, .int1:
            return 1
        }
    }

    /// Calculates shared memory size for an operation.
    private func calculateSharedMemorySize(type: FusedOpType, tuning: TuningConfig?, elementType: ElementType = .float32) -> Int {
        let elemSize = elementByteSize(for: elementType)

        switch type {
        case .original(let opKind):
            if opKind == .dot || opKind == .dotGeneral {
                let tileSize = tuning?.tileM ?? 32
                return 2 * tileSize * tileSize * elemSize
            }
            // Reduce kernel uses simple sequential reduction per thread, no shared memory needed
        case .fusedMatMulBiasAct:
            let tileSize = tuning?.tileM ?? 32
            return 2 * tileSize * tileSize * elemSize
        default:
            break
        }

        return 0
    }
}
