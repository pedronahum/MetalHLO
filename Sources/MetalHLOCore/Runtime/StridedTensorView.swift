// StridedTensorView.swift
// MetalHLOCore
//
// View abstraction for zero-copy transpose/reshape operations.
// Instead of copying data, views track stride/offset metadata.

import Foundation

// MARK: - StridedTensorView

/// A view into a tensor with custom strides and offset.
///
/// This enables zero-copy operations for transpose, reshape, slice, and similar
/// operations that only change how data is interpreted, not the data itself.
///
/// For example, transposing a [M, N] matrix to [N, M] doesn't copy data -
/// it just changes the strides from [N, 1] to [1, M].
public struct StridedTensorView: Sendable, Hashable {

    /// The tensor ID of the base buffer that contains the actual data.
    public let baseTensorID: TensorID

    /// Byte offset into the base buffer where this view starts.
    public let byteOffset: Int

    /// Logical shape of this view.
    public let shape: [Int]

    /// Strides in elements (not bytes) for each dimension.
    /// stride[i] is the number of elements to skip to advance one position in dimension i.
    public let strides: [Int]

    /// Element data type.
    public let elementType: ElementType

    /// Creates a strided tensor view.
    public init(
        baseTensorID: TensorID,
        byteOffset: Int = 0,
        shape: [Int],
        strides: [Int],
        elementType: ElementType
    ) {
        self.baseTensorID = baseTensorID
        self.byteOffset = byteOffset
        self.shape = shape
        self.strides = strides
        self.elementType = elementType
    }

    // MARK: - Computed Properties

    /// Total number of elements in the logical shape.
    public var elementCount: Int {
        shape.reduce(1, *)
    }

    /// Size in bytes of element type.
    public var elementByteSize: Int {
        elementType.byteSize
    }

    /// Whether this view has contiguous row-major memory layout.
    /// Contiguous means elements are stored without gaps in row-major order.
    public var isContiguous: Bool {
        guard !shape.isEmpty else { return true }

        // Calculate expected row-major strides
        let expectedStrides = Self.rowMajorStrides(for: shape)
        return strides == expectedStrides
    }

    /// Whether this is a simple view (no offset, contiguous).
    public var isSimple: Bool {
        byteOffset == 0 && isContiguous
    }

    /// The rank (number of dimensions) of the tensor.
    public var rank: Int {
        shape.count
    }

    // MARK: - Factory Methods

    /// Creates a contiguous view with row-major layout.
    public static func contiguous(
        tensorID: TensorID,
        shape: [Int],
        elementType: ElementType
    ) -> StridedTensorView {
        StridedTensorView(
            baseTensorID: tensorID,
            byteOffset: 0,
            shape: shape,
            strides: rowMajorStrides(for: shape),
            elementType: elementType
        )
    }

    /// Creates a view from a TensorInfo (assumes contiguous).
    public static func from(_ tensor: TensorInfo) -> StridedTensorView {
        contiguous(
            tensorID: tensor.id,
            shape: tensor.shape,
            elementType: tensor.elementType
        )
    }

    /// Calculates row-major strides for a given shape.
    public static func rowMajorStrides(for shape: [Int]) -> [Int] {
        guard !shape.isEmpty else { return [] }

        var strides = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        return strides
    }

    // MARK: - View Transformations

    /// Creates a transposed view with permuted dimensions.
    /// This is a zero-copy operation - only strides are changed.
    public func transposed(permutation: [Int]) -> StridedTensorView {
        precondition(permutation.count == shape.count, "Permutation must match rank")
        precondition(Set(permutation) == Set(0..<shape.count), "Invalid permutation")

        let newShape = permutation.map { shape[$0] }
        let newStrides = permutation.map { strides[$0] }

        return StridedTensorView(
            baseTensorID: baseTensorID,
            byteOffset: byteOffset,
            shape: newShape,
            strides: newStrides,
            elementType: elementType
        )
    }

    /// Creates a reshaped view if the reshape is compatible with current strides.
    /// Returns nil if the reshape requires a copy (non-contiguous data).
    public func reshaped(to newShape: [Int]) -> StridedTensorView? {
        // Check element count matches
        let newCount = newShape.reduce(1, *)
        guard newCount == elementCount else { return nil }

        // Reshape is only valid without copy if the view is contiguous
        // or if the reshape preserves the memory layout.
        //
        // For now, we only support reshape of contiguous views.
        // More sophisticated analysis could handle some non-contiguous cases.
        guard isContiguous else { return nil }

        return StridedTensorView(
            baseTensorID: baseTensorID,
            byteOffset: byteOffset,
            shape: newShape,
            strides: Self.rowMajorStrides(for: newShape),
            elementType: elementType
        )
    }

    /// Creates a sliced view with an offset into the data.
    /// For a slice starting at multi-dimensional index `starts` with shape `sizes`.
    public func sliced(starts: [Int], sizes: [Int]) -> StridedTensorView {
        precondition(starts.count == shape.count, "Starts must match rank")
        precondition(sizes.count == shape.count, "Sizes must match rank")

        // Calculate byte offset for the starting position
        var additionalOffset = 0
        for i in 0..<starts.count {
            additionalOffset += starts[i] * strides[i]
        }
        additionalOffset *= elementByteSize

        return StridedTensorView(
            baseTensorID: baseTensorID,
            byteOffset: byteOffset + additionalOffset,
            shape: sizes,
            strides: strides,  // Strides remain the same
            elementType: elementType
        )
    }

    /// Creates a broadcasted view by setting strides to 0 for broadcast dimensions.
    /// `targetShape` must be compatible with current shape (each dim either matches or is 1).
    public func broadcast(to targetShape: [Int]) -> StridedTensorView? {
        // Handle rank extension
        let rankDiff = targetShape.count - shape.count
        guard rankDiff >= 0 else { return nil }

        // Pad shape and strides with 1s and 0s for extended dimensions
        let paddedShape = [Int](repeating: 1, count: rankDiff) + shape
        let paddedStrides = [Int](repeating: 0, count: rankDiff) + strides

        var newStrides = [Int](repeating: 0, count: targetShape.count)

        for i in 0..<targetShape.count {
            if paddedShape[i] == targetShape[i] {
                // Same size - keep stride
                newStrides[i] = paddedStrides[i]
            } else if paddedShape[i] == 1 {
                // Broadcast dimension - set stride to 0
                newStrides[i] = 0
            } else {
                // Incompatible dimensions
                return nil
            }
        }

        return StridedTensorView(
            baseTensorID: baseTensorID,
            byteOffset: byteOffset,
            shape: targetShape,
            strides: newStrides,
            elementType: elementType
        )
    }

    // MARK: - Index Calculation

    /// Calculates the flat index (in elements) for a given multi-dimensional index.
    public func flatIndex(for indices: [Int]) -> Int {
        precondition(indices.count == shape.count, "Index must match rank")

        var offset = byteOffset / elementByteSize
        for i in 0..<indices.count {
            offset += indices[i] * strides[i]
        }
        return offset
    }

    /// Generates Metal code for calculating the offset from a flat index.
    /// This is used for kernel code generation with strided access.
    public func generateOffsetCalculation(flatIndexVar: String, resultVar: String) -> String {
        guard !isContiguous else {
            // Contiguous - just use flat index directly
            if byteOffset == 0 {
                return "uint \(resultVar) = \(flatIndexVar);"
            } else {
                let elementOffset = byteOffset / elementByteSize
                return "uint \(resultVar) = \(flatIndexVar) + \(elementOffset);"
            }
        }

        // Non-contiguous - need to unflatten and re-calculate with strides
        var code = "uint \(resultVar);\n"
        code += "{\n"
        code += "    uint remaining = \(flatIndexVar);\n"

        let baseOffset = byteOffset / elementByteSize
        if baseOffset != 0 {
            code += "    uint offset = \(baseOffset);\n"
        } else {
            code += "    uint offset = 0;\n"
        }

        // Unflatten using shape (row-major) then apply custom strides
        let rowMajorStrides = Self.rowMajorStrides(for: shape)

        for i in 0..<shape.count {
            if rowMajorStrides[i] > 0 {
                code += "    uint idx\(i) = remaining / \(rowMajorStrides[i]);\n"
                code += "    remaining = remaining % \(rowMajorStrides[i]);\n"
            } else {
                code += "    uint idx\(i) = remaining;\n"
            }
            code += "    offset += idx\(i) * \(strides[i]);\n"
        }

        code += "    \(resultVar) = offset;\n"
        code += "}"

        return code
    }
}

// MARK: - ViewRegistry

/// Registry for tracking which tensors are views of other tensors.
/// Used during compilation to avoid generating kernels for pure view operations.
public final class ViewRegistry: @unchecked Sendable {

    /// Maps tensor IDs to their view definitions.
    private var views: [TensorID: StridedTensorView]

    /// Lock for thread-safe access.
    private let lock = NSLock()

    public init() {
        self.views = [:]
    }

    /// Registers a tensor as a view of a base tensor.
    public func registerView(_ tensorID: TensorID, view: StridedTensorView) {
        lock.lock()
        defer { lock.unlock() }
        views[tensorID] = view
    }

    /// Gets the view definition for a tensor, if it is a view.
    public func getView(_ tensorID: TensorID) -> StridedTensorView? {
        lock.lock()
        defer { lock.unlock() }
        return views[tensorID]
    }

    /// Checks if a tensor is a view.
    public func isView(_ tensorID: TensorID) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return views[tensorID] != nil
    }

    /// Gets the ultimate base tensor for a view (follows view chain).
    public func getBaseTensor(_ tensorID: TensorID) -> TensorID {
        lock.lock()
        defer { lock.unlock() }
        return resolveBaseTensor(tensorID)
    }

    /// Resolves view chain to find ultimate base tensor (called with lock held).
    private func resolveBaseTensor(_ tensorID: TensorID) -> TensorID {
        guard let view = views[tensorID] else {
            return tensorID
        }
        return resolveBaseTensor(view.baseTensorID)
    }

    /// Gets the fully resolved view for a tensor (follows and combines view chain).
    public func getResolvedView(_ tensorID: TensorID) -> StridedTensorView? {
        lock.lock()
        defer { lock.unlock() }

        guard var view = views[tensorID] else {
            return nil
        }

        // Follow view chain and combine transformations
        while let nextView = views[view.baseTensorID] {
            // Combine offset
            let newOffset = nextView.byteOffset + view.byteOffset

            // The strides stay the same since they're relative to the base tensor
            view = StridedTensorView(
                baseTensorID: nextView.baseTensorID,
                byteOffset: newOffset,
                shape: view.shape,
                strides: view.strides,
                elementType: view.elementType
            )
        }

        return view
    }

    /// Returns all registered views.
    public func allViews() -> [TensorID: StridedTensorView] {
        lock.lock()
        defer { lock.unlock() }
        return views
    }

    /// Clears all registered views.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        views.removeAll()
    }
}

// MARK: - View Operation Detection

/// Utility for detecting which operations can be implemented as views.
public struct ViewOperationDetector {

    /// Determines if a transpose operation can be a view.
    /// Transpose is always a view - it just permutes strides.
    public static func canTransposeBeView(inputShape: [Int], permutation: [Int]) -> Bool {
        // Transpose is always a valid view operation
        return true
    }

    /// Determines if a reshape operation can be a view.
    /// Reshape is a view only if the input is contiguous.
    public static func canReshapeBeView(
        inputView: StridedTensorView,
        outputShape: [Int]
    ) -> Bool {
        // Check element counts match
        let inputCount = inputView.elementCount
        let outputCount = outputShape.reduce(1, *)
        guard inputCount == outputCount else { return false }

        // Only contiguous views can be reshaped without copy
        return inputView.isContiguous
    }

    /// Determines if a slice operation can be a view.
    /// Slice is always a view - it adjusts offset and shape but keeps strides.
    public static func canSliceBeView(
        inputShape: [Int],
        starts: [Int],
        sizes: [Int]
    ) -> Bool {
        // Slice is always a valid view operation
        return true
    }

    /// Determines if a broadcast can be a view.
    /// Broadcast is always a view - it sets strides to 0 for broadcast dims.
    public static func canBroadcastBeView(
        inputShape: [Int],
        outputShape: [Int]
    ) -> Bool {
        // Check rank extension is non-negative
        let rankDiff = outputShape.count - inputShape.count
        guard rankDiff >= 0 else { return false }

        // Pad input shape for comparison
        let paddedInput = [Int](repeating: 1, count: rankDiff) + inputShape

        // Check each dimension is compatible
        for i in 0..<outputShape.count {
            if paddedInput[i] != outputShape[i] && paddedInput[i] != 1 {
                return false
            }
        }

        return true
    }
}

// MARK: - Materialization Detection

/// Determines when views must be materialized (copied to contiguous memory).
public struct MaterializationAnalyzer {

    /// Operations that require contiguous input memory.
    public static let requiresContiguousInput: Set<HLOOpKind> = [
        .dot,
        .dotGeneral,
        .convolution,
        .reduce,
        .reduceWindow,
        .fft,
        .sort,
        .triangularSolve,
        .cholesky,
        .batchNormInference,
        .batchNormTraining,
        .batchNormGrad,
    ]

    /// Operations that can work with strided inputs (elementwise).
    public static let supportsStridedInput: Set<HLOOpKind> = [
        .add, .subtract, .multiply, .divide,
        .maximum, .minimum, .power,
        .negate, .abs, .exponential, .log, .sqrt, .rsqrt,
        .sine, .cosine, .tanh, .floor, .ceil, .sign,
        .not, .and, .or, .xor,
        .compare, .select, .clamp,
        .convert,
    ]

    /// Determines if an operation requires its input to be materialized.
    public static func requiresMaterialization(
        opKind: HLOOpKind,
        inputView: StridedTensorView
    ) -> Bool {
        // If already contiguous, no materialization needed
        if inputView.isContiguous {
            return false
        }

        // Check if operation requires contiguous input
        if requiresContiguousInput.contains(opKind) {
            return true
        }

        // Operations that support strided input don't need materialization
        if supportsStridedInput.contains(opKind) {
            return false
        }

        // Default: require materialization for unknown operations
        return true
    }
}
