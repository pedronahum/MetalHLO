// LayoutAssignment.swift
// MetalHLOCore
//
// Layout assignment pass that assigns optimal memory layouts per operation.
// Inspired by XLA's LayoutAssignment pass.

import Foundation

// MARK: - Tensor Layout

/// Represents a tensor's memory layout.
///
/// Layout determines how tensor elements are stored in memory, which affects
/// performance of different operations. Optimal layout depends on the operation:
/// - Convolution prefers NHWC on Apple Silicon
/// - MatMul prefers row-major for LHS, column-major for RHS
/// - Reductions prefer contiguous reduction dimension
public enum TensorLayout: String, Sendable, Equatable, CustomStringConvertible {
    /// Row-major (C-style) layout - last dimension is contiguous.
    /// For 2D: [row][col] where elements in same row are contiguous.
    /// Best for operations that iterate over last dimension.
    case rowMajor = "row_major"

    /// Column-major (Fortran-style) layout - first dimension is contiguous.
    /// For 2D: [col][row] where elements in same column are contiguous.
    /// Best for matrix RHS in matmul (transposed access).
    case columnMajor = "column_major"

    /// NHWC layout: [batch, height, width, channels].
    /// Preferred for convolution on Apple Silicon/MPSGraph.
    case nhwc = "nhwc"

    /// NCHW layout: [batch, channels, height, width].
    /// Common in PyTorch models, may need conversion.
    case nchw = "nchw"

    /// HWIO layout for convolution kernels: [height, width, inputChannels, outputChannels].
    /// Preferred for convolution weights on Apple Silicon.
    case hwio = "hwio"

    /// OIHW layout for convolution kernels: [outputChannels, inputChannels, height, width].
    /// Common in PyTorch models, may need conversion.
    case oihw = "oihw"

    /// Default/unspecified layout - uses whatever the producer provides.
    case unspecified = "unspecified"

    public var description: String { rawValue }

    /// Whether this layout is for convolution operands (4D+).
    public var isConvolutionLayout: Bool {
        switch self {
        case .nhwc, .nchw, .hwio, .oihw:
            return true
        default:
            return false
        }
    }

    /// Whether this layout is for matrix operands (2D).
    public var isMatrixLayout: Bool {
        switch self {
        case .rowMajor, .columnMajor:
            return true
        default:
            return false
        }
    }
}

// MARK: - Layout Preference

/// Layout preference for a tensor value.
public struct LayoutPreference: Sendable {
    /// The preferred layout.
    public let layout: TensorLayout

    /// Priority (higher = stronger preference).
    public let priority: Int

    /// The operation that requested this layout.
    public let source: String

    public init(layout: TensorLayout, priority: Int = 0, source: String = "") {
        self.layout = layout
        self.priority = priority
        self.source = source
    }
}

// MARK: - Layout Plan

/// The result of layout assignment.
public struct LayoutPlan: Sendable {
    /// Assigned layouts for each tensor value.
    public let layouts: [String: TensorLayout]

    /// Copy operations needed for layout conversion.
    public let copies: [LayoutCopyOp]

    /// Whether any layout conversion is needed.
    public var requiresConversion: Bool {
        !copies.isEmpty
    }

    public init(layouts: [String: TensorLayout], copies: [LayoutCopyOp]) {
        self.layouts = layouts
        self.copies = copies
    }
}

/// A copy operation for layout conversion.
public struct LayoutCopyOp: Sendable {
    /// The source tensor value name.
    public let source: String

    /// The destination tensor value name.
    public let destination: String

    /// The source layout.
    public let sourceLayout: TensorLayout

    /// The target layout.
    public let targetLayout: TensorLayout

    /// The tensor type.
    public let type: TensorType

    public init(
        source: String,
        destination: String,
        sourceLayout: TensorLayout,
        targetLayout: TensorLayout,
        type: TensorType
    ) {
        self.source = source
        self.destination = destination
        self.sourceLayout = sourceLayout
        self.targetLayout = targetLayout
        self.type = type
    }
}

// MARK: - Layout Assignment Pass

/// Layout assignment pass that assigns optimal memory layouts to tensors.
///
/// The pass works by:
/// 1. Assigning preferred layouts based on operation type
/// 2. Propagating layouts through the graph
/// 3. Resolving conflicts by choosing compatible layouts
/// 4. Inserting copy operations when layout conflicts can't be avoided
///
/// Expected impact: 1.1-1.3x speedup from better memory access patterns.
public final class LayoutAssignment: @unchecked Sendable {

    /// Whether to insert explicit transpose operations for layout conversion.
    private let insertTransposes: Bool

    /// Creates a layout assignment pass.
    ///
    /// - Parameter insertTransposes: Whether to insert transpose operations
    ///   for layout conversion (default: true).
    public init(insertTransposes: Bool = true) {
        self.insertTransposes = insertTransposes
    }

    /// Assigns layouts to all tensors in a function.
    ///
    /// - Parameter function: The function to optimize.
    /// - Returns: The layout plan with assigned layouts and required copies.
    public func assignLayouts(to function: HLOFunction) -> LayoutPlan {
        var layouts: [String: TensorLayout] = [:]
        var preferences: [String: [LayoutPreference]] = [:]

        // Phase 1: Collect layout preferences from operations
        collectPreferences(from: function, into: &preferences)

        // Phase 2: Resolve preferences to final layouts
        resolveLayouts(preferences: preferences, into: &layouts)

        // Phase 3: Propagate layouts through the graph
        propagateLayouts(function: function, layouts: &layouts)

        // Phase 4: Find layout conflicts and create copy operations
        let copies = findConflicts(function: function, layouts: layouts)

        return LayoutPlan(layouts: layouts, copies: copies)
    }

    /// Optimizes a function by inserting layout conversion operations.
    ///
    /// - Parameter function: The function to optimize.
    /// - Returns: The optimized function with layout conversions.
    public func optimize(_ function: HLOFunction) -> HLOFunction {
        let plan = assignLayouts(to: function)

        guard plan.requiresConversion && insertTransposes else {
            return function
        }

        return insertLayoutConversions(function: function, plan: plan)
    }

    // MARK: - Phase 1: Collect Preferences

    /// Collects layout preferences from all operations.
    private func collectPreferences(
        from function: HLOFunction,
        into preferences: inout [String: [LayoutPreference]]
    ) {
        for op in function.operations {
            let opPreferences = getPreferredLayouts(for: op)

            for (tensorName, pref) in opPreferences {
                preferences[tensorName, default: []].append(pref)
            }
        }

        // Add default preferences for function inputs
        for input in function.inputs {
            if preferences[input.name] == nil {
                preferences[input.name] = [
                    LayoutPreference(layout: .rowMajor, priority: 0, source: "input_default")
                ]
            }
        }
    }

    /// Gets preferred layouts for an operation.
    private func getPreferredLayouts(for op: HLOOperation) -> [(String, LayoutPreference)] {
        var result: [(String, LayoutPreference)] = []

        switch op.kind {
        case .convolution:
            // Convolution prefers NHWC on Apple Silicon
            let inputLayout = getInputLayoutForConvolution(op)
            let kernelLayout = getKernelLayoutForConvolution(op)

            if op.operands.count >= 1 {
                result.append((op.operands[0], LayoutPreference(
                    layout: inputLayout,
                    priority: 10,
                    source: "convolution_input"
                )))
            }
            if op.operands.count >= 2 {
                result.append((op.operands[1], LayoutPreference(
                    layout: kernelLayout,
                    priority: 10,
                    source: "convolution_kernel"
                )))
            }
            result.append((op.result, LayoutPreference(
                layout: inputLayout,  // Output same as input
                priority: 10,
                source: "convolution_output"
            )))

        case .dot, .dotGeneral:
            // MatMul prefers row-major for LHS, column-major for RHS
            // This enables efficient SIMD matrix operations
            if op.operands.count >= 1 {
                let lhsLayout = getLHSLayoutForMatMul(op)
                result.append((op.operands[0], LayoutPreference(
                    layout: lhsLayout,
                    priority: 8,
                    source: "matmul_lhs"
                )))
            }
            if op.operands.count >= 2 {
                let rhsLayout = getRHSLayoutForMatMul(op)
                result.append((op.operands[1], LayoutPreference(
                    layout: rhsLayout,
                    priority: 8,
                    source: "matmul_rhs"
                )))
            }
            result.append((op.result, LayoutPreference(
                layout: .rowMajor,
                priority: 8,
                source: "matmul_output"
            )))

        case .reduce:
            // Reductions prefer contiguous reduction dimension
            let layout = getLayoutForReduction(op)
            if let firstOperand = op.operands.first {
                result.append((firstOperand, LayoutPreference(
                    layout: layout,
                    priority: 6,
                    source: "reduce_input"
                )))
            }

        case .transpose:
            // Transpose operations indicate a layout change
            // Output layout is inverse of input layout
            if let firstOperand = op.operands.first {
                result.append((firstOperand, LayoutPreference(
                    layout: .unspecified,
                    priority: 2,
                    source: "transpose_input"
                )))
            }
            result.append((op.result, LayoutPreference(
                layout: .unspecified,
                priority: 2,
                source: "transpose_output"
            )))

        case .batchNormInference, .batchNormTraining:
            // Batch norm follows input layout (usually NHWC for conv nets)
            if let firstOperand = op.operands.first {
                result.append((firstOperand, LayoutPreference(
                    layout: .nhwc,
                    priority: 5,
                    source: "batchnorm_input"
                )))
            }
            result.append((op.result, LayoutPreference(
                layout: .nhwc,
                priority: 5,
                source: "batchnorm_output"
            )))

        default:
            // Elementwise and most other ops have no layout preference
            // They follow their inputs' layout
            break
        }

        return result
    }

    // MARK: - Convolution Layout Helpers

    private func getInputLayoutForConvolution(_ op: HLOOperation) -> TensorLayout {
        // Check if convolution dimension numbers specify the layout
        if let dimNums = op.attributes.convolutionDimensionNumbers {
            // NHWC: batch=0, feature(channel)=last, spatial in between
            // NCHW: batch=0, feature(channel)=1, spatial after
            if dimNums.inputBatchDimension == 0 {
                if dimNums.inputFeatureDimension == op.resultType.rank - 1 {
                    return .nhwc  // Channels last
                } else if dimNums.inputFeatureDimension == 1 {
                    return .nchw  // Channels first
                }
            }
        }
        // Default to NHWC (preferred on Apple Silicon)
        return .nhwc
    }

    private func getKernelLayoutForConvolution(_ op: HLOOperation) -> TensorLayout {
        // Check if convolution dimension numbers specify kernel layout
        if let dimNums = op.attributes.convolutionDimensionNumbers {
            // HWIO: spatial first, then input feature, then output feature
            // OIHW: output feature first, then input feature, then spatial
            let spatialFirst = dimNums.kernelSpatialDimensions.first ?? 0
            if spatialFirst == 0 {
                return .hwio
            } else if dimNums.kernelOutputFeatureDimension == 0 {
                return .oihw
            }
        }
        // Default to HWIO (preferred on Apple Silicon)
        return .hwio
    }

    // MARK: - MatMul Layout Helpers

    private func getLHSLayoutForMatMul(_ op: HLOOperation) -> TensorLayout {
        // Check for transposed LHS from dot dimension numbers
        if let dimNums = op.attributes.dotDimensionNumbers {
            // If contracting on first non-batch dim, LHS should be column-major
            // If contracting on last dim, LHS should be row-major
            if let contractingDim = dimNums.lhsContractingDimensions.first {
                let effectiveDim = contractingDim - dimNums.lhsBatchingDimensions.count
                if effectiveDim == 0 {
                    return .columnMajor  // A^T @ B
                }
            }
        }
        return .rowMajor  // Default: A @ B with row-major A
    }

    private func getRHSLayoutForMatMul(_ op: HLOOperation) -> TensorLayout {
        // Check for transposed RHS from dot dimension numbers
        // Standard matmul A @ B:
        // - LHS [M, K] contracts K (last dim)
        // - RHS [K, N] contracts K (first dim)
        // For RHS, output reads columns of B, so column-major is optimal
        if let dimNums = op.attributes.dotDimensionNumbers {
            if let contractingDim = dimNums.rhsContractingDimensions.first {
                let effectiveDim = contractingDim - dimNums.rhsBatchingDimensions.count
                // If contracting on first non-batch dim (standard case), column-major is best
                // because we read columns when computing output
                if effectiveDim == 0 {
                    return .columnMajor  // A @ B with contracting first dim of B
                } else {
                    // Contracting on last dim means B is transposed, row-major works
                    return .rowMajor
                }
            }
        }
        return .columnMajor  // Default: column-major for RHS
    }

    // MARK: - Reduction Layout Helpers

    private func getLayoutForReduction(_ op: HLOOperation) -> TensorLayout {
        // Reductions prefer the reduction dimension to be contiguous
        guard let dims = op.attributes.dimensions,
              !dims.isEmpty,
              !op.operands.isEmpty else {
            return .rowMajor
        }

        let rank = op.resultType.rank + dims.count  // Input rank
        let lastDim = rank - 1

        // If reducing last dimension, row-major is best
        if dims.contains(lastDim) {
            return .rowMajor
        }

        // If reducing first dimension, column-major is best
        if dims.contains(0) {
            return .columnMajor
        }

        // Default to row-major
        return .rowMajor
    }

    // MARK: - Phase 2: Resolve Layouts

    /// Resolves layout preferences to final layouts.
    private func resolveLayouts(
        preferences: [String: [LayoutPreference]],
        into layouts: inout [String: TensorLayout]
    ) {
        for (tensorName, prefs) in preferences {
            // Choose the highest priority preference
            if let bestPref = prefs.max(by: { $0.priority < $1.priority }) {
                if bestPref.layout != .unspecified {
                    layouts[tensorName] = bestPref.layout
                }
            }
        }
    }

    // MARK: - Phase 3: Propagate Layouts

    /// Propagates layouts through the graph.
    private func propagateLayouts(
        function: HLOFunction,
        layouts: inout [String: TensorLayout]
    ) {
        // Build use-def information
        var producers: [String: HLOOperation] = [:]
        for op in function.operations {
            producers[op.result] = op
        }

        // Forward propagation: if an op has no layout but its inputs do, inherit
        var changed = true
        var iterations = 0
        let maxIterations = 10

        while changed && iterations < maxIterations {
            changed = false
            iterations += 1

            for op in function.operations {
                // Skip if result already has a layout
                if layouts[op.result] != nil {
                    continue
                }

                // For elementwise ops, inherit layout from first input with a layout
                if isLayoutPreservingOp(op.kind) {
                    for operand in op.operands {
                        if let inputLayout = layouts[operand], inputLayout != .unspecified {
                            layouts[op.result] = inputLayout
                            changed = true
                            break
                        }
                    }
                }
            }
        }

        // Fill in any remaining unassigned layouts with default
        for op in function.operations {
            if layouts[op.result] == nil {
                layouts[op.result] = .rowMajor
            }
        }

        // Also ensure all inputs have layouts
        for input in function.inputs {
            if layouts[input.name] == nil {
                layouts[input.name] = .rowMajor
            }
        }
    }

    /// Checks if an operation preserves its input layout.
    private func isLayoutPreservingOp(_ kind: HLOOpKind) -> Bool {
        switch kind {
        // Elementwise ops preserve layout
        case .add, .subtract, .multiply, .divide, .maximum, .minimum, .power,
             .negate, .abs, .exponential, .log, .sqrt, .rsqrt, .tanh, .logistic,
             .sine, .cosine, .tan, .floor, .ceil, .sign, .not, .and, .or, .xor,
             .compare, .select, .clamp, .convert, .bitcastConvert,
             .expm1, .log1p, .cbrt, .roundNearestAfz, .roundNearestEven:
            return true

        // Broadcast and reshape may preserve layout depending on dimensions
        case .broadcastInDim:
            return true

        default:
            return false
        }
    }

    // MARK: - Phase 4: Find Conflicts

    /// Finds layout conflicts and creates copy operations.
    private func findConflicts(
        function: HLOFunction,
        layouts: [String: TensorLayout]
    ) -> [LayoutCopyOp] {
        var copies: [LayoutCopyOp] = []
        var typeMap: [String: TensorType] = [:]

        // Build type map
        for input in function.inputs {
            typeMap[input.name] = input.type
        }
        for op in function.operations {
            typeMap[op.result] = op.resultType
        }

        // Check each operation for layout conflicts
        for op in function.operations {
            let requiredLayouts = getRequiredInputLayouts(for: op)

            for (index, operand) in op.operands.enumerated() {
                guard let currentLayout = layouts[operand],
                      let requiredLayout = requiredLayouts[index],
                      requiredLayout != .unspecified,
                      currentLayout != requiredLayout,
                      let type = typeMap[operand] else {
                    continue
                }

                // Create a copy operation for layout conversion
                let newName = "\(operand)_layout_\(requiredLayout.rawValue)"
                let copy = LayoutCopyOp(
                    source: operand,
                    destination: newName,
                    sourceLayout: currentLayout,
                    targetLayout: requiredLayout,
                    type: type
                )
                copies.append(copy)
            }
        }

        return copies
    }

    /// Gets required input layouts for an operation.
    private func getRequiredInputLayouts(for op: HLOOperation) -> [Int: TensorLayout] {
        var result: [Int: TensorLayout] = [:]

        switch op.kind {
        case .convolution:
            result[0] = getInputLayoutForConvolution(op)
            result[1] = getKernelLayoutForConvolution(op)

        case .dot, .dotGeneral:
            result[0] = getLHSLayoutForMatMul(op)
            result[1] = getRHSLayoutForMatMul(op)

        default:
            break
        }

        return result
    }

    // MARK: - Insert Layout Conversions

    /// Inserts layout conversion operations into the function.
    private func insertLayoutConversions(
        function: HLOFunction,
        plan: LayoutPlan
    ) -> HLOFunction {
        // Build a map of operand replacements
        var replacements: [String: String] = [:]
        var newOps: [HLOOperation] = []
        var insertedOps: [HLOOperation] = []

        for copy in plan.copies {
            replacements[copy.source] = copy.destination

            // Create a transpose operation for layout conversion
            let transposeOp = createLayoutConversionOp(copy: copy)
            insertedOps.append(transposeOp)
        }

        // Insert conversion ops before the first use
        var insertedBefore: Set<String> = []

        for op in function.operations {
            // Insert any needed conversion ops before this operation
            for operand in op.operands {
                if let newName = replacements[operand],
                   !insertedBefore.contains(newName) {
                    // Find the corresponding conversion op
                    if let convOp = insertedOps.first(where: { $0.result == newName }) {
                        newOps.append(convOp)
                        insertedBefore.insert(newName)
                    }
                }
            }

            // Update operands to use converted versions
            var newOperands = op.operands
            for (index, operand) in op.operands.enumerated() {
                if let newName = replacements[operand] {
                    newOperands[index] = newName
                }
            }

            let newOp = HLOOperation(
                result: op.result,
                kind: op.kind,
                operands: newOperands,
                resultType: op.resultType,
                attributes: op.attributes
            )
            newOps.append(newOp)
        }

        return HLOFunction(
            name: function.name,
            inputs: function.inputs,
            outputTypes: function.outputTypes,
            operations: newOps,
            returnValues: function.returnValues
        )
    }

    /// Creates a layout conversion operation (transpose or reshape).
    private func createLayoutConversionOp(copy: LayoutCopyOp) -> HLOOperation {
        // Determine the transpose permutation based on source and target layouts
        let permutation = getTransposePermutation(
            from: copy.sourceLayout,
            to: copy.targetLayout,
            rank: copy.type.rank
        )

        var attributes = HLOAttributes()
        attributes.dimensions = permutation

        // Compute the transposed shape
        let transposedShape = permutation.map { copy.type.shape[$0] }
        let transposedType = TensorType(shape: transposedShape, elementType: copy.type.elementType)

        return HLOOperation(
            result: copy.destination,
            kind: .transpose,
            operands: [copy.source],
            resultType: transposedType,
            attributes: attributes
        )
    }

    /// Gets the transpose permutation to convert between layouts.
    private func getTransposePermutation(
        from source: TensorLayout,
        to target: TensorLayout,
        rank: Int
    ) -> [Int] {
        // Handle 2D matrix layout conversions
        if rank == 2 {
            switch (source, target) {
            case (.rowMajor, .columnMajor), (.columnMajor, .rowMajor):
                return [1, 0]
            default:
                break
            }
        }

        // Handle 4D convolution layout conversions
        if rank == 4 {
            switch (source, target) {
            // NCHW to NHWC: [0, 1, 2, 3] -> [0, 2, 3, 1]
            case (.nchw, .nhwc):
                return [0, 2, 3, 1]

            // NHWC to NCHW: [0, 1, 2, 3] -> [0, 3, 1, 2]
            case (.nhwc, .nchw):
                return [0, 3, 1, 2]

            // OIHW to HWIO: [0, 1, 2, 3] -> [2, 3, 1, 0]
            case (.oihw, .hwio):
                return [2, 3, 1, 0]

            // HWIO to OIHW: [0, 1, 2, 3] -> [3, 2, 0, 1]
            case (.hwio, .oihw):
                return [3, 2, 0, 1]

            default:
                break
            }
        }

        // Default: identity permutation
        return Array(0..<rank)
    }
}

// MARK: - Layout Statistics

/// Statistics about layout assignment.
public struct LayoutStatistics: Sendable {
    /// Number of tensors with assigned layouts.
    public let numAssignedLayouts: Int

    /// Number of layout conflicts found.
    public let numConflicts: Int

    /// Number of copy operations inserted.
    public let numCopiesInserted: Int

    /// Breakdown by layout type.
    public let layoutCounts: [TensorLayout: Int]

    public init(
        numAssignedLayouts: Int,
        numConflicts: Int,
        numCopiesInserted: Int,
        layoutCounts: [TensorLayout: Int]
    ) {
        self.numAssignedLayouts = numAssignedLayouts
        self.numConflicts = numConflicts
        self.numCopiesInserted = numCopiesInserted
        self.layoutCounts = layoutCounts
    }
}

extension LayoutAssignment {
    /// Computes statistics for layout assignment.
    public func computeStatistics(for plan: LayoutPlan) -> LayoutStatistics {
        var layoutCounts: [TensorLayout: Int] = [:]

        for (_, layout) in plan.layouts {
            layoutCounts[layout, default: 0] += 1
        }

        return LayoutStatistics(
            numAssignedLayouts: plan.layouts.count,
            numConflicts: plan.copies.count,
            numCopiesInserted: plan.copies.count,
            layoutCounts: layoutCounts
        )
    }
}
