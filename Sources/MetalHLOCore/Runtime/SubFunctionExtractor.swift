// SubFunctionExtractor.swift
// MetalHLOCore
//
// Extracts sub-functions from an HLOFunction for device-specific execution.
// Used by the heterogeneous executor to build runnable sub-graphs for
// GPU and ANE partitions.

/// Extracts a subset of operations from an HLOFunction into a standalone sub-function.
///
/// Given a partition (set of operation IDs), this builds a new `HLOFunction` with:
/// - Inputs: original function inputs used by partition ops + cross-partition dependencies
/// - Operations: only the ops in the partition, in topological order
/// - Return values: ops consumed by later partitions or the original function's return values
struct SubFunctionExtractor {

    /// Extracts a sub-function containing only the specified operations.
    ///
    /// - Parameters:
    ///   - function: The original HLO function.
    ///   - operationIds: The set of operation result names to include.
    ///   - outputConsumers: Operations outside this partition that consume outputs from it.
    ///     Used to determine which ops become return values.
    /// - Returns: A standalone `HLOFunction` for the partition.
    static func extract(
        from function: HLOFunction,
        operationIds: Set<String>,
        outputConsumers: Set<String>
    ) -> HLOFunction {
        let originalInputNames = Set(function.inputs.map { $0.name })

        // Build a map of result name → operation for quick lookup
        let opMap = Dictionary(uniqueKeysWithValues: function.operations.map { ($0.result, $0) })

        // Collect operations in this partition (preserving topological order)
        let partitionOps = function.operations.filter { operationIds.contains($0.result) }

        // Determine inputs: original function inputs + cross-partition dependencies
        var inputArgs: [HLOArgument] = []
        var seenInputs = Set<String>()

        // Add original function inputs that are directly used by partition ops
        for arg in function.inputs {
            for op in partitionOps {
                if op.operands.contains(arg.name) && !seenInputs.contains(arg.name) {
                    inputArgs.append(arg)
                    seenInputs.insert(arg.name)
                }
            }
        }

        // Add cross-partition dependencies as synthetic inputs
        for op in partitionOps {
            for operand in op.operands {
                if seenInputs.contains(operand) { continue }
                if operationIds.contains(operand) { continue }
                if originalInputNames.contains(operand) { continue }

                // This operand comes from an op outside this partition
                if let sourceOp = opMap[operand] {
                    let arg = HLOArgument(name: operand, type: sourceOp.resultType)
                    inputArgs.append(arg)
                    seenInputs.insert(operand)
                }
            }
        }

        // Determine return values:
        // 1. Ops whose results are consumed outside this partition
        // 2. Ops that are in the original function's return values
        var returnValues: [String] = []
        var returnTypes: [TensorType] = []
        var seenReturns = Set<String>()

        // Check which partition ops are consumed by outside ops
        for op in partitionOps {
            if outputConsumers.contains(op.result) && !seenReturns.contains(op.result) {
                returnValues.append(op.result)
                returnTypes.append(op.resultType)
                seenReturns.insert(op.result)
            }
        }

        // Check if any partition ops are original return values
        for retVal in function.returnValues {
            if operationIds.contains(retVal) && !seenReturns.contains(retVal) {
                returnValues.append(retVal)
                if let op = opMap[retVal] {
                    returnTypes.append(op.resultType)
                }
                seenReturns.insert(retVal)
            }
        }

        // If no return values determined, use the last op
        if returnValues.isEmpty, let lastOp = partitionOps.last {
            returnValues = [lastOp.result]
            returnTypes = [lastOp.resultType]
        }

        return HLOFunction(
            name: function.name + "_partition",
            inputs: inputArgs,
            outputTypes: returnTypes,
            operations: partitionOps,
            returnValues: returnValues
        )
    }

    /// Determines which operations outside a partition consume outputs from it.
    ///
    /// - Parameters:
    ///   - function: The original HLO function.
    ///   - operationIds: The set of operation IDs in this partition.
    /// - Returns: The set of partition op results consumed by external ops.
    static func findExternalConsumers(
        of operationIds: Set<String>,
        in function: HLOFunction
    ) -> Set<String> {
        var consumed = Set<String>()

        for op in function.operations {
            guard !operationIds.contains(op.result) else { continue }
            for operand in op.operands {
                if operationIds.contains(operand) {
                    consumed.insert(operand)
                }
            }
        }

        // Also include original return values that are in this partition
        for retVal in function.returnValues {
            if operationIds.contains(retVal) {
                consumed.insert(retVal)
            }
        }

        return consumed
    }
}
