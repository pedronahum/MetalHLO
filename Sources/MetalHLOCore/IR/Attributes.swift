// Attributes.swift
// MetalHLOCore
//
// Operation-specific attributes for HLO operations.

/// Operation-specific attributes.
///
/// `HLOAttributes` contains all possible attributes that operations
/// may need. Only relevant attributes are populated for each operation.
public struct HLOAttributes: Sendable {

    // MARK: - Dimension Attributes

    /// Dimension list (for transpose, broadcast, reduce, etc.).
    public var dimensions: [Int]?

    /// Single axis (for softmax, concatenate, etc.).
    public var axis: Int?

    /// Iota dimension - which dimension to fill with sequential indices.
    public var iotaDimension: Int?

    // MARK: - Dot Product Attributes

    /// Dot product dimension numbers for dot_general.
    public var dotDimensionNumbers: DotDimensionNumbers?

    // MARK: - Gather/Scatter Attributes

    /// Gather dimension numbers.
    public var gatherDimensionNumbers: GatherDimensionNumbers?

    /// Scatter dimension numbers.
    public var scatterDimensionNumbers: ScatterDimensionNumbers?

    /// Scatter computation kind (how to combine existing and update values).
    /// Defaults to .set if not specified (simple replacement).
    public var scatterComputationKind: ScatterComputationKind?

    // MARK: - Comparison Attributes

    /// Comparison direction (EQ, NE, LT, LE, GT, GE).
    public var comparisonDirection: ComparisonDirection?

    // MARK: - Reduction Attributes

    /// Reduction kind (sum, max, min, mean).
    public var reductionKind: ReductionKind?

    // MARK: - RNG Attributes

    /// RNG distribution (uniform, normal).
    public var rngDistribution: RNGDistribution?

    // MARK: - Constant Attributes

    /// Constant value for stablehlo.constant.
    public var constantValue: ConstantValue?

    // MARK: - Slice Attributes

    /// Slice start indices.
    public var sliceStarts: [Int]?

    /// Slice limit indices.
    public var sliceLimits: [Int]?

    /// Slice strides.
    public var sliceStrides: [Int]?

    // MARK: - Pad Attributes

    /// Padding amounts at the low end of each dimension.
    public var padLow: [Int]?

    /// Padding amounts at the high end of each dimension.
    public var padHigh: [Int]?

    /// Interior padding (between elements).
    public var padInterior: [Int]?

    // MARK: - Convolution Attributes

    /// Convolution dimension numbers.
    public var convolutionDimensionNumbers: ConvolutionDimensionNumbers?

    /// Window strides for convolution/pooling.
    public var windowStrides: [Int]?

    /// Padding for convolution (as tensor<Nx2>: [[low, high], ...]).
    public var convPadding: [[Int]]?

    /// LHS dilation for convolution.
    public var lhsDilation: [Int]?

    /// RHS dilation for convolution.
    public var rhsDilation: [Int]?

    /// Feature group count for grouped convolutions.
    public var featureGroupCount: Int?

    /// Batch group count for batch-grouped convolutions.
    public var batchGroupCount: Int?

    // MARK: - Reduce Window Attributes

    /// Window dimensions for reduce_window (pooling).
    public var windowDimensions: [Int]?

    /// Base dilations for reduce_window.
    public var baseDilations: [Int]?

    /// Window dilations for reduce_window.
    public var windowDilations: [Int]?

    // MARK: - Batch Normalization Attributes

    /// Epsilon for numerical stability in batch norm.
    public var epsilon: Float?

    /// Feature index (channel dimension) for batch norm.
    public var featureIndex: Int?

    // MARK: - FFT Attributes

    /// FFT type (FFT, IFFT, RFFT, IRFFT).
    public var fftType: FFTType?

    /// FFT length along each transformed axis.
    public var fftLength: [Int]?

    // MARK: - Sort Attributes

    /// Whether the sort should be stable.
    public var isStable: Bool?

    /// Sort descending instead of ascending.
    public var sortDescending: Bool?

    // MARK: - Triangular Solve Attributes

    /// Whether A is on the left side (Ax = b) or right (xA = b).
    public var leftSide: Bool?

    /// Whether to use the lower triangle.
    public var lower: Bool?

    /// Whether the diagonal is implicitly unit (all ones).
    public var unitDiagonal: Bool?

    /// Transpose operation to apply to A.
    public var transposeA: TransposeKind?

    // MARK: - RNG Bit Generator Attributes

    /// PRNG algorithm to use.
    public var rngAlgorithm: RNGAlgorithm?

    // MARK: - Reduce Precision Attributes

    /// Number of bits for exponent in reduce_precision.
    public var exponentBits: Int?

    /// Number of bits for mantissa in reduce_precision.
    public var mantissaBits: Int?

    // MARK: - Dynamic Slice Attributes

    /// Slice sizes for dynamic_slice.
    public var dynamicSliceSizes: [Int]?

    // MARK: - Select and Scatter Attributes

    /// Select and scatter dimension numbers.
    public var selectAndScatterDimensionNumbers: SelectAndScatterDimensionNumbers?

    // MARK: - Map Attributes

    /// Computation region for map operation.
    public var mapComputation: Region?

    // MARK: - Control Flow Attributes

    /// While loop regions (condition and body).
    public var whileRegions: WhileRegions?

    /// If conditional regions (then and else).
    public var ifRegions: IfRegions?

    // MARK: - Custom Call Attributes

    /// Target name for custom_call operation.
    public var callTargetName: String?

    /// Backend configuration JSON string for custom_call.
    public var backendConfig: String?

    // MARK: - Initialization

    public init() {}
}

// MARK: - Supporting Types

/// Dimension numbers for dot_general operation.
public struct DotDimensionNumbers: Sendable, Equatable {
    /// Batching dimensions on the LHS.
    public let lhsBatchingDimensions: [Int]

    /// Batching dimensions on the RHS.
    public let rhsBatchingDimensions: [Int]

    /// Contracting dimensions on the LHS.
    public let lhsContractingDimensions: [Int]

    /// Contracting dimensions on the RHS.
    public let rhsContractingDimensions: [Int]

    public init(
        lhsBatchingDimensions: [Int],
        rhsBatchingDimensions: [Int],
        lhsContractingDimensions: [Int],
        rhsContractingDimensions: [Int]
    ) {
        self.lhsBatchingDimensions = lhsBatchingDimensions
        self.rhsBatchingDimensions = rhsBatchingDimensions
        self.lhsContractingDimensions = lhsContractingDimensions
        self.rhsContractingDimensions = rhsContractingDimensions
    }
}

/// Dimension numbers for gather operation.
public struct GatherDimensionNumbers: Sendable, Equatable {
    /// Offset dimensions in the output.
    public let offsetDims: [Int]

    /// Collapsed slice dimensions.
    public let collapsedSliceDims: [Int]

    /// Start index map - maps index vector elements to operand dimensions.
    public let startIndexMap: [Int]

    /// Index vector dimension in start_indices.
    public let indexVectorDim: Int

    /// Slice sizes for each dimension of the operand.
    public let sliceSizes: [Int]

    /// Operand batching dimensions - dimensions of operand that are batching dimensions.
    /// These correspond to startIndicesBatchingDims in the indices tensor.
    public let operandBatchingDims: [Int]

    /// Start indices batching dimensions - dimensions of start_indices that are batching dimensions.
    /// These correspond to operandBatchingDims in the operand tensor.
    public let startIndicesBatchingDims: [Int]

    public init(
        offsetDims: [Int],
        collapsedSliceDims: [Int],
        startIndexMap: [Int],
        indexVectorDim: Int,
        sliceSizes: [Int],
        operandBatchingDims: [Int] = [],
        startIndicesBatchingDims: [Int] = []
    ) {
        self.offsetDims = offsetDims
        self.collapsedSliceDims = collapsedSliceDims
        self.startIndexMap = startIndexMap
        self.indexVectorDim = indexVectorDim
        self.sliceSizes = sliceSizes
        self.operandBatchingDims = operandBatchingDims
        self.startIndicesBatchingDims = startIndicesBatchingDims
    }
}

/// Dimension numbers for scatter operation.
public struct ScatterDimensionNumbers: Sendable, Equatable {
    /// Update window dimensions in the updates tensor.
    public let updateWindowDims: [Int]

    /// Inserted window dimensions - dimensions that are inserted (size 1) during scatter.
    public let insertedWindowDims: [Int]

    /// Scatter dimensions to operand dimensions mapping.
    public let scatterDimsToOperandDims: [Int]

    /// Index vector dimension in scatter_indices.
    public let indexVectorDim: Int

    /// Input (operand) batching dimensions - dimensions of inputs that are batching dimensions.
    /// These correspond to scatterIndicesBatchingDims in the indices tensor.
    public let inputBatchingDims: [Int]

    /// Scatter indices batching dimensions - dimensions of scatter_indices that are batching dimensions.
    /// These correspond to inputBatchingDims in the operand tensor.
    public let scatterIndicesBatchingDims: [Int]

    public init(
        updateWindowDims: [Int],
        insertedWindowDims: [Int],
        scatterDimsToOperandDims: [Int],
        indexVectorDim: Int,
        inputBatchingDims: [Int] = [],
        scatterIndicesBatchingDims: [Int] = []
    ) {
        self.updateWindowDims = updateWindowDims
        self.insertedWindowDims = insertedWindowDims
        self.scatterDimsToOperandDims = scatterDimsToOperandDims
        self.indexVectorDim = indexVectorDim
        self.inputBatchingDims = inputBatchingDims
        self.scatterIndicesBatchingDims = scatterIndicesBatchingDims
    }
}

/// Dimension numbers for convolution operation.
public struct ConvolutionDimensionNumbers: Sendable, Equatable {
    /// Input batch dimension.
    public let inputBatchDimension: Int

    /// Input feature (channel) dimension.
    public let inputFeatureDimension: Int

    /// Input spatial dimensions.
    public let inputSpatialDimensions: [Int]

    /// Kernel input feature dimension.
    public let kernelInputFeatureDimension: Int

    /// Kernel output feature dimension.
    public let kernelOutputFeatureDimension: Int

    /// Kernel spatial dimensions.
    public let kernelSpatialDimensions: [Int]

    /// Output batch dimension.
    public let outputBatchDimension: Int

    /// Output feature dimension.
    public let outputFeatureDimension: Int

    /// Output spatial dimensions.
    public let outputSpatialDimensions: [Int]

    public init(
        inputBatchDimension: Int,
        inputFeatureDimension: Int,
        inputSpatialDimensions: [Int],
        kernelInputFeatureDimension: Int,
        kernelOutputFeatureDimension: Int,
        kernelSpatialDimensions: [Int],
        outputBatchDimension: Int,
        outputFeatureDimension: Int,
        outputSpatialDimensions: [Int]
    ) {
        self.inputBatchDimension = inputBatchDimension
        self.inputFeatureDimension = inputFeatureDimension
        self.inputSpatialDimensions = inputSpatialDimensions
        self.kernelInputFeatureDimension = kernelInputFeatureDimension
        self.kernelOutputFeatureDimension = kernelOutputFeatureDimension
        self.kernelSpatialDimensions = kernelSpatialDimensions
        self.outputBatchDimension = outputBatchDimension
        self.outputFeatureDimension = outputFeatureDimension
        self.outputSpatialDimensions = outputSpatialDimensions
    }
}

/// FFT type for fft operation.
public enum FFTType: String, Sendable, CaseIterable {
    /// Forward complex-to-complex FFT.
    case fft = "FFT"

    /// Inverse complex-to-complex FFT.
    case ifft = "IFFT"

    /// Real-to-complex FFT.
    case rfft = "RFFT"

    /// Complex-to-real inverse FFT.
    case irfft = "IRFFT"
}

/// Comparison direction for compare operations.
public enum ComparisonDirection: String, Sendable, CaseIterable {
    case eq = "EQ"
    case ne = "NE"
    case lt = "LT"
    case le = "LE"
    case gt = "GT"
    case ge = "GE"
}

/// Reduction kind for reduce operations.
public enum ReductionKind: Sendable {
    case sum
    case max
    case min
    case mean
}

/// Scatter computation kind for combining values.
/// Maps to MPS scatter modes.
public enum ScatterComputationKind: Sendable {
    /// Replace the existing value with the update (MPS .set mode).
    case set
    /// Add the update to the existing value (MPS .add mode).
    case add
    /// Take the maximum of existing and update (MPS .max mode).
    case max
    /// Take the minimum of existing and update (MPS .min mode).
    case min
    /// Multiply the existing value by the update (MPS .mul mode).
    case mul
}

/// RNG distribution type.
public enum RNGDistribution: Sendable {
    case uniform
    case normal
}

/// Constant value representation.
public enum ConstantValue: Sendable {
    /// A scalar constant.
    case scalar(Double)

    /// A splat constant (same value for all elements).
    case splat(Double, TensorType)

    /// A dense constant with explicit values.
    case dense([Double], TensorType)
}

// MARK: - Control Flow Regions

/// A region containing operations (for control flow bodies).
public struct Region: Sendable {
    /// Block arguments (input parameters to the region).
    public let arguments: [RegionArgument]

    /// Operations in the region.
    public let operations: [HLOOperation]

    /// Return values from the region.
    public let returnValues: [String]

    public init(
        arguments: [RegionArgument],
        operations: [HLOOperation],
        returnValues: [String]
    ) {
        self.arguments = arguments
        self.operations = operations
        self.returnValues = returnValues
    }
}

/// An argument to a region block.
public struct RegionArgument: Sendable {
    /// Argument name (e.g., "%arg0").
    public let name: String

    /// Argument type.
    public let type: TensorType

    public init(name: String, type: TensorType) {
        self.name = name
        self.type = type
    }
}

/// Regions for a while loop operation.
public struct WhileRegions: Sendable {
    /// The condition region (returns i1 predicate).
    public let condition: Region

    /// The body region (returns loop-carried values).
    public let body: Region

    public init(condition: Region, body: Region) {
        self.condition = condition
        self.body = body
    }
}

/// Regions for an if conditional operation.
public struct IfRegions: Sendable {
    /// The then branch region.
    public let thenBranch: Region

    /// The else branch region (optional).
    public let elseBranch: Region?

    public init(thenBranch: Region, elseBranch: Region?) {
        self.thenBranch = thenBranch
        self.elseBranch = elseBranch
    }
}

// MARK: - Triangular Solve Types

/// Transpose operation kind for triangular solve.
public enum TransposeKind: String, Sendable, CaseIterable {
    /// No transpose.
    case noTranspose = "NO_TRANSPOSE"

    /// Transpose.
    case transpose = "TRANSPOSE"

    /// Conjugate transpose (Hermitian).
    case adjoint = "ADJOINT"
}

// MARK: - RNG Types

/// PRNG algorithm for rng_bit_generator.
public enum RNGAlgorithm: String, Sendable, CaseIterable {
    /// Default algorithm (implementation-defined).
    case defaultAlgorithm = "DEFAULT"

    /// Three-fry counter-based PRNG.
    case threeFry = "THREE_FRY"

    /// Philox counter-based PRNG.
    case philox = "PHILOX"
}

// MARK: - Select and Scatter Types

/// Dimension numbers for select_and_scatter operation.
public struct SelectAndScatterDimensionNumbers: Sendable, Equatable {
    /// Window dimensions.
    public let windowDimensions: [Int]

    /// Window strides.
    public let windowStrides: [Int]

    /// Padding (low, high for each dimension).
    public let padding: [[Int]]?

    public init(
        windowDimensions: [Int],
        windowStrides: [Int],
        padding: [[Int]]?
    ) {
        self.windowDimensions = windowDimensions
        self.windowStrides = windowStrides
        self.padding = padding
    }
}
