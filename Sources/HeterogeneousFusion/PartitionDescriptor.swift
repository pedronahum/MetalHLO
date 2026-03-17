// PartitionDescriptor.swift
// HeterogeneousFusion
//
// Phase 2: Declarative data structure describing how a single operation
// is partitioned across compute units. The executor consumes this without
// knowing how the partition was decided.

import Metal

// MARK: - DType

/// Element data type for tensor slices.
public enum DType: String, Sendable {
    case float32
    case float16
    case int32
    case int16
    case int8

    /// Size in bytes of a single element.
    public var byteSize: Int {
        switch self {
        case .float32, .int32: return 4
        case .float16, .int16: return 2
        case .int8: return 1
        }
    }
}

// MARK: - HLOOpCode

/// Operation codes eligible for heterogeneous partitioning.
public enum HLOOpCode: String, Sendable, Codable {
    case matmul
    case convolution
    case elementwiseAdd
    case elementwiseMul
    case elementwiseRelu
    case gelu
    case layerNorm
    case embeddingLookup
    case reduceSum
    case reduceMax
    case gather
    case scatter
    case depthAttention

    /// Whether this op is an elementwise operation.
    public var isElementwise: Bool {
        switch self {
        case .elementwiseAdd, .elementwiseMul, .elementwiseRelu, .gelu:
            return true
        default:
            return false
        }
    }

    /// Whether this elementwise op is binary (takes two inputs).
    public var isBinaryElementwise: Bool {
        switch self {
        case .elementwiseAdd, .elementwiseMul: return true
        default: return false
        }
    }

    /// Whether this op requires auxiliary parameter buffers (γ, β for layerNorm).
    public var hasAuxParameters: Bool {
        switch self {
        case .layerNorm: return true
        default: return false
        }
    }

    /// Whether this op uses vocab-split partitioning (embedding lookup).
    public var isVocabSplit: Bool {
        switch self {
        case .embeddingLookup: return true
        default: return false
        }
    }

    /// Whether this op is a convolution.
    public var isConvolution: Bool {
        switch self {
        case .convolution: return true
        default: return false
        }
    }
}

// MARK: - TensorSlice

/// A view into a shared buffer, describing a region that one compute unit
/// will read from or write to.
///
/// For row-split (dim 0): slices are contiguous, `fullRowBytes` is nil.
/// For column-split (dim 1): slices are strided, `fullRowBytes` gives the
/// parent matrix's row stride in bytes.
public struct TensorSlice: Sendable {
    /// Byte offset into the shared MTLBuffer (first element of this slice).
    public let bufferOffset: Int
    /// Shape of this slice (e.g., [sliceM, K] for row-split, [K, sliceN] for column-split B).
    public let shape: [Int]
    /// Element data type.
    public let dtype: DType
    /// For column-split slices: the full row width in bytes of the parent matrix.
    /// nil means the slice is contiguous (row-split or full matrix).
    public let fullRowBytes: Int?

    /// Total number of elements in this slice.
    public var elementCount: Int {
        shape.reduce(1, *)
    }

    /// Total byte size of this slice (contiguous size, ignoring stride).
    public var byteSize: Int {
        elementCount * dtype.byteSize
    }

    /// Whether this slice is strided (column-split).
    public var isStrided: Bool { fullRowBytes != nil }

    public init(bufferOffset: Int, shape: [Int], dtype: DType, fullRowBytes: Int? = nil) {
        self.bufferOffset = bufferOffset
        self.shape = shape
        self.dtype = dtype
        self.fullRowBytes = fullRowBytes
    }
}

// MARK: - UnitAssignment

/// Assigns a slice of work to a specific compute unit.
/// The executor reads `inputSlices` from the shared input buffers,
/// executes the op on `unit`, and writes to `outputSlice` in the shared output buffer.
public struct UnitAssignment: Sendable {
    /// Which compute unit executes this slice.
    public let unit: ComputeUnit
    /// Input tensor slices (e.g., A-slice and B for matmul).
    /// For matmul: [0] = A-slice (partitioned), [1] = B (full, shared across units).
    public let inputSlices: [TensorSlice]
    /// Output tensor slice (portion of C this unit writes).
    public let outputSlice: TensorSlice
    /// Fraction of total work assigned to this unit (for profiling).
    public let workFraction: Double

    public init(unit: ComputeUnit, inputSlices: [TensorSlice], outputSlice: TensorSlice, workFraction: Double) {
        self.unit = unit
        self.inputSlices = inputSlices
        self.outputSlice = outputSlice
        self.workFraction = workFraction
    }
}

// MARK: - PartitionDescriptor

/// A complete, declarative description of how to partition a single operation
/// across multiple compute units.
///
/// The executor consumes this without knowledge of how the split was decided.
/// The splitter produces this without knowledge of how execution works.
public struct PartitionDescriptor: Sendable {
    /// The operation being partitioned.
    public let op: HLOOpCode
    /// Full shape of the primary input tensor (before partitioning).
    public let fullInputShape: [Int]
    /// Full shape of the secondary input tensor (e.g., B matrix), if any.
    public let secondaryInputShape: [Int]?
    /// Full shape of the output tensor.
    public let fullOutputShape: [Int]
    /// Which axis of the primary input was partitioned.
    public let splitDimension: Int
    /// Per-unit work assignments.
    public let assignments: [UnitAssignment]
    /// Element data type.
    public let dtype: DType

    public init(
        op: HLOOpCode,
        fullInputShape: [Int],
        secondaryInputShape: [Int]? = nil,
        fullOutputShape: [Int],
        splitDimension: Int,
        assignments: [UnitAssignment],
        dtype: DType
    ) {
        self.op = op
        self.fullInputShape = fullInputShape
        self.secondaryInputShape = secondaryInputShape
        self.fullOutputShape = fullOutputShape
        self.splitDimension = splitDimension
        self.assignments = assignments
        self.dtype = dtype
    }

    /// Total number of units participating.
    public var unitCount: Int { assignments.count }

    /// Units involved.
    public var units: [ComputeUnit] { assignments.map(\.unit) }

    /// Validate that all slices are consistent.
    public func validate() -> Bool {
        // Check work fractions sum to ~1.0
        let totalWork = assignments.map(\.workFraction).reduce(0, +)
        guard abs(totalWork - 1.0) < 0.01 else { return false }

        // Check output slices cover the full output without overlap
        let totalOutputElements = assignments.map(\.outputSlice.elementCount).reduce(0, +)
        let expectedOutputElements = fullOutputShape.reduce(1, *)
        guard totalOutputElements == expectedOutputElements else { return false }

        return true
    }

    /// Whether this descriptor uses column splitting (strided output slices).
    public var isColumnSplit: Bool { splitDimension == 1 }

    /// Whether this descriptor uses vocab-split (embedding lookup).
    /// splitDimension == -1 is the sentinel for vocab-split.
    public var isVocabSplit: Bool { splitDimension == -1 }
}

// MARK: - PartitionDescriptor Builder for Matmul

extension PartitionDescriptor {

    /// Create a matmul partition descriptor for C = A @ B,
    /// splitting A along the row dimension (dim 0).
    ///
    /// - Parameters:
    ///   - shape: The matmul dimensions (M, K, N).
    ///   - fractions: Per-unit fraction of rows. Must sum to 1.0.
    ///     E.g., [(.gpu, 0.35), (.mps, 0.45), (.cpu, 0.20)].
    ///   - dtype: Element type (default .float32).
    public static func matmul(
        shape: MatrixShape,
        fractions: [(ComputeUnit, Double)],
        dtype: DType = .float32
    ) -> PartitionDescriptor {
        let M = shape.M, K = shape.K, N = shape.N
        let elemSize = dtype.byteSize

        var assignments: [UnitAssignment] = []
        var rowOffset = 0

        for (idx, (unit, fraction)) in fractions.enumerated() {
            // Last unit gets all remaining rows to avoid rounding errors
            let rows: Int
            if idx == fractions.count - 1 {
                rows = M - rowOffset
            } else {
                rows = max(1, Int(Double(M) * fraction))
            }

            guard rows > 0 else { continue }

            let aSlice = TensorSlice(
                bufferOffset: rowOffset * K * elemSize,
                shape: [rows, K],
                dtype: dtype
            )
            // B is shared across all units — full matrix, no offset
            let bSlice = TensorSlice(
                bufferOffset: 0,
                shape: [K, N],
                dtype: dtype
            )
            let cSlice = TensorSlice(
                bufferOffset: rowOffset * N * elemSize,
                shape: [rows, N],
                dtype: dtype
            )

            assignments.append(UnitAssignment(
                unit: unit,
                inputSlices: [aSlice, bSlice],
                outputSlice: cSlice,
                workFraction: Double(rows) / Double(M)
            ))

            rowOffset += rows
        }

        return PartitionDescriptor(
            op: .matmul,
            fullInputShape: [M, K],
            secondaryInputShape: [K, N],
            fullOutputShape: [M, N],
            splitDimension: 0,
            assignments: assignments,
            dtype: dtype
        )
    }

    /// Create a matmul partition descriptor for C = A @ B,
    /// splitting B and C along the column dimension (dim 1).
    ///
    /// Each unit gets: full A [M, K], a column slice of B [K, sliceN], and
    /// writes a column slice of C [M, sliceN].
    ///
    /// Column slices are strided in row-major memory. The `fullRowBytes` field
    /// on each slice enables the executor to handle extraction/scattering.
    /// MPS can use strided access natively; GPU/CPU need extract+scatter helpers.
    ///
    /// - Parameters:
    ///   - shape: The matmul dimensions (M, K, N).
    ///   - fractions: Per-unit fraction of columns. Must sum to 1.0.
    ///   - dtype: Element type (default .float32).
    public static func matmulColumnSplit(
        shape: MatrixShape,
        fractions: [(ComputeUnit, Double)],
        dtype: DType = .float32
    ) -> PartitionDescriptor {
        let M = shape.M, K = shape.K, N = shape.N
        let elemSize = dtype.byteSize

        var assignments: [UnitAssignment] = []
        var colOffset = 0

        for (idx, (unit, fraction)) in fractions.enumerated() {
            let cols: Int
            if idx == fractions.count - 1 {
                cols = N - colOffset
            } else {
                cols = max(1, Int(Double(N) * fraction))
            }

            guard cols > 0 else { continue }

            // A is shared across all units — full matrix
            let aSlice = TensorSlice(
                bufferOffset: 0,
                shape: [M, K],
                dtype: dtype
            )

            // B column slice: strided, starts at colOffset
            let bSlice = TensorSlice(
                bufferOffset: colOffset * elemSize,
                shape: [K, cols],
                dtype: dtype,
                fullRowBytes: N * elemSize
            )

            // C column slice: strided, starts at colOffset
            let cSlice = TensorSlice(
                bufferOffset: colOffset * elemSize,
                shape: [M, cols],
                dtype: dtype,
                fullRowBytes: N * elemSize
            )

            assignments.append(UnitAssignment(
                unit: unit,
                inputSlices: [aSlice, bSlice],
                outputSlice: cSlice,
                workFraction: Double(cols) / Double(N)
            ))

            colOffset += cols
        }

        return PartitionDescriptor(
            op: .matmul,
            fullInputShape: [M, K],
            secondaryInputShape: [K, N],
            fullOutputShape: [M, N],
            splitDimension: 1,
            assignments: assignments,
            dtype: dtype
        )
    }

    /// Auto-select row-split or column-split based on shape.
    ///
    /// Column-split is preferred when:
    /// 1. N is significantly larger than M (N > 2*M), AND
    /// 2. The matrix is large enough for extract/scatter overhead to pay off (M*K >= 256K elements)
    ///
    /// The 2x threshold avoids column-split overhead for mildly rectangular shapes.
    /// The size floor avoids column-split on small matrices where the copy cost dominates.
    public static func matmulAutoSplit(
        shape: MatrixShape,
        fractions: [(ComputeUnit, Double)],
        dtype: DType = .float32
    ) -> PartitionDescriptor {
        let totalElements = shape.M * shape.K
        if shape.N > 2 * shape.M && totalElements >= 256 * 1024 {
            return matmulColumnSplit(shape: shape, fractions: fractions, dtype: dtype)
        } else {
            return matmul(shape: shape, fractions: fractions, dtype: dtype)
        }
    }
}

// MARK: - PartitionDescriptor Builder for Elementwise

extension PartitionDescriptor {

    /// Create an elementwise partition descriptor, splitting along rows (dim 0).
    ///
    /// Elementwise ops are trivially row-splittable — no cross-element dependencies.
    /// For binary ops (add, mul), both inputs must have the same shape.
    ///
    /// - Parameters:
    ///   - op: The elementwise op code (.elementwiseAdd, .elementwiseMul, .elementwiseRelu, .gelu).
    ///   - rows: Number of rows (M dimension).
    ///   - cols: Number of columns (N dimension).
    ///   - fractions: Per-unit fraction of rows.
    ///   - dtype: Element type (default .float32).
    public static func elementwise(
        op: HLOOpCode,
        rows: Int,
        cols: Int,
        fractions: [(ComputeUnit, Double)],
        dtype: DType = .float32
    ) -> PartitionDescriptor {
        let elemSize = dtype.byteSize

        var assignments: [UnitAssignment] = []
        var rowOffset = 0

        for (idx, (unit, fraction)) in fractions.enumerated() {
            let sliceRows: Int
            if idx == fractions.count - 1 {
                sliceRows = rows - rowOffset
            } else {
                sliceRows = max(1, Int(Double(rows) * fraction))
            }
            guard sliceRows > 0 else { continue }

            let aSlice = TensorSlice(
                bufferOffset: rowOffset * cols * elemSize,
                shape: [sliceRows, cols],
                dtype: dtype
            )

            // Binary ops: B has the same shape and offset as A
            var inputSlices = [aSlice]
            if op.isBinaryElementwise {
                let bSlice = TensorSlice(
                    bufferOffset: rowOffset * cols * elemSize,
                    shape: [sliceRows, cols],
                    dtype: dtype
                )
                inputSlices.append(bSlice)
            }

            let cSlice = TensorSlice(
                bufferOffset: rowOffset * cols * elemSize,
                shape: [sliceRows, cols],
                dtype: dtype
            )

            assignments.append(UnitAssignment(
                unit: unit,
                inputSlices: inputSlices,
                outputSlice: cSlice,
                workFraction: Double(sliceRows) / Double(rows)
            ))

            rowOffset += sliceRows
        }

        return PartitionDescriptor(
            op: op,
            fullInputShape: [rows, cols],
            secondaryInputShape: op.isBinaryElementwise ? [rows, cols] : nil,
            fullOutputShape: [rows, cols],
            splitDimension: 0,
            assignments: assignments,
            dtype: dtype
        )
    }
}

// MARK: - PartitionDescriptor Builder for LayerNorm

extension PartitionDescriptor {

    /// Create a LayerNorm partition descriptor, splitting along rows (dim 0).
    ///
    /// LayerNorm is per-row independent: each row's mean/variance/normalize is
    /// self-contained. γ and β are read-only broadcast parameters — not sliced.
    /// Output slices are contiguous in row-major layout (no recombine needed).
    ///
    /// Input buffers layout: [0]=input [M,N], [1]=gamma [N], [2]=beta [N]
    /// Output buffer: [M,N]
    ///
    /// - Parameters:
    ///   - M: Number of rows.
    ///   - N: Hidden dimension (columns per row).
    ///   - fractions: Per-unit fraction of rows.
    ///   - dtype: Element type (default .float32).
    public static func layerNorm(
        M: Int,
        N: Int,
        fractions: [(ComputeUnit, Double)],
        dtype: DType = .float32
    ) -> PartitionDescriptor {
        let elemSize = dtype.byteSize

        var assignments: [UnitAssignment] = []
        var rowOffset = 0

        for (idx, (unit, fraction)) in fractions.enumerated() {
            let sliceRows: Int
            if idx == fractions.count - 1 {
                sliceRows = M - rowOffset
            } else {
                sliceRows = max(1, Int(Double(M) * fraction))
            }
            guard sliceRows > 0 else { continue }

            // Input slice: contiguous row range of [M, N]
            let inputSlice = TensorSlice(
                bufferOffset: rowOffset * N * elemSize,
                shape: [sliceRows, N],
                dtype: dtype
            )

            // γ and β: full vectors, shared across all units (no offset)
            let gammaSlice = TensorSlice(
                bufferOffset: 0,
                shape: [N],
                dtype: dtype
            )
            let betaSlice = TensorSlice(
                bufferOffset: 0,
                shape: [N],
                dtype: dtype
            )

            // Output slice: same row range as input
            let outputSlice = TensorSlice(
                bufferOffset: rowOffset * N * elemSize,
                shape: [sliceRows, N],
                dtype: dtype
            )

            assignments.append(UnitAssignment(
                unit: unit,
                inputSlices: [inputSlice, gammaSlice, betaSlice],
                outputSlice: outputSlice,
                workFraction: Double(sliceRows) / Double(M)
            ))

            rowOffset += sliceRows
        }

        return PartitionDescriptor(
            op: .layerNorm,
            fullInputShape: [M, N],
            secondaryInputShape: nil,
            fullOutputShape: [M, N],
            splitDimension: 0,
            assignments: assignments,
            dtype: dtype
        )
    }
}

// MARK: - PartitionDescriptor Builder for Embedding Lookup

extension PartitionDescriptor {

    /// Create an embedding lookup partition descriptor using vocab-range split.
    ///
    /// Unlike row-split ops, embedding splits the **vocabulary** (embedding table rows)
    /// instead of the input sequence. All units read ALL token IDs and write output
    /// only for tokens that fall in their vocab range.
    ///
    /// Output buffer must be **pre-zeroed** before dispatch — each output row is
    /// written by exactly one unit (scatter-free), but unwritten rows must be zero.
    ///
    /// Uses splitDimension = -1 as a sentinel for vocab-split partitioning.
    ///
    /// Input buffers layout: [0]=tokenIds [seqLen] (int32, shared),
    ///                        [1]=embeddingTable [vocabSize, embeddingDim] (float32, full table)
    /// Output buffer: [seqLen, embeddingDim]
    ///
    /// - Parameters:
    ///   - vocabSize: Total vocabulary size.
    ///   - seqLen: Sequence length (number of tokens).
    ///   - embeddingDim: Width of each embedding row.
    ///   - fractions: Per-unit fraction of vocab rows.
    ///   - dtype: Element type for embedding table (default .float32).
    public static func embeddingLookup(
        vocabSize: Int,
        seqLen: Int,
        embeddingDim: Int,
        fractions: [(ComputeUnit, Double)],
        dtype: DType = .float32
    ) -> PartitionDescriptor {
        let elemSize = dtype.byteSize

        var assignments: [UnitAssignment] = []
        var vocabOffset = 0

        for (idx, (unit, fraction)) in fractions.enumerated() {
            let shardSize: Int
            if idx == fractions.count - 1 {
                shardSize = vocabSize - vocabOffset
            } else {
                shardSize = max(1, Int(Double(vocabSize) * fraction))
            }
            guard shardSize > 0 else { continue }

            // Token IDs: shared, full sequence, int32
            let tokenSlice = TensorSlice(
                bufferOffset: 0,
                shape: [seqLen],
                dtype: .int32
            )

            // Table shard: offset into the full embedding table
            let tableSlice = TensorSlice(
                bufferOffset: vocabOffset * embeddingDim * elemSize,
                shape: [shardSize, embeddingDim],
                dtype: dtype
            )

            // Output: full output buffer (each unit writes only its rows, others are zero)
            let outputSlice = TensorSlice(
                bufferOffset: 0,
                shape: [seqLen, embeddingDim],
                dtype: dtype
            )

            assignments.append(UnitAssignment(
                unit: unit,
                inputSlices: [tokenSlice, tableSlice],
                outputSlice: outputSlice,
                workFraction: Double(shardSize) / Double(vocabSize)
            ))

            vocabOffset += shardSize
        }

        return PartitionDescriptor(
            op: .embeddingLookup,
            fullInputShape: [seqLen],
            secondaryInputShape: [vocabSize, embeddingDim],
            fullOutputShape: [seqLen, embeddingDim],
            splitDimension: -1,  // sentinel for vocab-split
            assignments: assignments,
            dtype: dtype
        )
    }
}

// MARK: - PartitionDescriptor Builder for Convolution

extension PartitionDescriptor {

    /// Create a convolution partition descriptor, splitting along output channels.
    ///
    /// NCHW layout: input [B, Cin, H, W], weights [Cout, Cin, Kh, Kw],
    /// output [B, Cout, outH, outW].
    ///
    /// Output-channel split: each unit gets a contiguous range of output channels.
    /// In NCHW layout, each unit's output is contiguous in memory (within a batch).
    /// Weights are sliced along the first dimension (outChannels).
    /// Input is shared across all units (all Cin needed for every Cout).
    ///
    /// splitDimension = -2 is the sentinel for output-channel split.
    ///
    /// - Parameters:
    ///   - shape: Convolution dimensions.
    ///   - fractions: Per-unit fraction of output channels.
    ///   - dtype: Element type (default .float32).
    public static func convolution(
        shape: ConvShape,
        fractions: [(ComputeUnit, Double)],
        dtype: DType = .float32
    ) -> PartitionDescriptor {
        let elemSize = dtype.byteSize
        let outH = shape.outHeight
        let outW = shape.outWidth
        let outSpatial = outH * outW
        let filterSize = shape.inChannels * shape.kernelH * shape.kernelW

        var assignments: [UnitAssignment] = []
        var chanOffset = 0

        for (idx, (unit, fraction)) in fractions.enumerated() {
            let sliceC: Int
            if idx == fractions.count - 1 {
                sliceC = shape.outChannels - chanOffset
            } else {
                sliceC = max(1, Int(Double(shape.outChannels) * fraction))
            }
            guard sliceC > 0 else { continue }

            // Input: full [B, Cin, H, W], shared across all units
            let inputSlice = TensorSlice(
                bufferOffset: 0,
                shape: [shape.batch, shape.inChannels, shape.height, shape.width],
                dtype: dtype
            )

            // Weights: slice [sliceC, Cin, Kh, Kw] starting at chanOffset
            let weightsSlice = TensorSlice(
                bufferOffset: chanOffset * filterSize * elemSize,
                shape: [sliceC, shape.inChannels, shape.kernelH, shape.kernelW],
                dtype: dtype
            )

            // Output: slice [B, sliceC, outH, outW] starting at chanOffset
            // In NCHW, each batch element's channels are contiguous:
            // offset for batch b = b * Cout * outSpatial + chanOffset * outSpatial
            // For batch=1, this is simply chanOffset * outSpatial.
            // For batch>1, we store per-batch channel slices contiguously
            // (the executor will handle multi-batch output assembly).
            let outputSlice = TensorSlice(
                bufferOffset: chanOffset * outSpatial * elemSize,
                shape: [shape.batch, sliceC, outH, outW],
                dtype: dtype
            )

            assignments.append(UnitAssignment(
                unit: unit,
                inputSlices: [inputSlice, weightsSlice],
                outputSlice: outputSlice,
                workFraction: Double(sliceC) / Double(shape.outChannels)
            ))

            chanOffset += sliceC
        }

        return PartitionDescriptor(
            op: .convolution,
            fullInputShape: [shape.batch, shape.inChannels, shape.height, shape.width],
            secondaryInputShape: [shape.outChannels, shape.inChannels, shape.kernelH, shape.kernelW],
            fullOutputShape: [shape.batch, shape.outChannels, outH, outW],
            splitDimension: -2,  // sentinel for output-channel split
            assignments: assignments,
            dtype: dtype
        )
    }
}

extension PartitionDescriptor {
    /// Whether this descriptor uses output-channel splitting (convolution).
    public var isChannelSplit: Bool { splitDimension == -2 }
}
