// StaticMemoryPlanner.swift
// MetalHLOCore
//
// Static memory planning for zero-allocation execution.
// Computes all memory requirements at compile time.

import Foundation

// MARK: - Type Aliases

/// Unique identifier for a tensor in the computation graph.
public typealias TensorID = String

/// Unique identifier for an operation.
public typealias OperationID = Int

// MARK: - Scheduled Tensor Lifetime

/// Represents the lifetime of a tensor during execution with scheduling info.
/// Named differently to avoid conflict with existing TensorLifetime in BufferAssignment.
public struct ScheduledTensorLifetime: Sendable {
    /// Operation index where the tensor is created.
    public let createdAt: Int

    /// Operation index where the tensor is last used.
    public let lastUsedAt: Int

    /// Size in bytes.
    public let byteSize: Int

    /// Whether this tensor is an input (never freed).
    public let isInput: Bool

    /// Whether this tensor is an output (kept until end).
    public let isOutput: Bool

    /// Whether this tensor is a constant (stored in separate constant buffer).
    public let isConstant: Bool

    public init(createdAt: Int, lastUsedAt: Int, byteSize: Int, isInput: Bool = false, isOutput: Bool = false, isConstant: Bool = false) {
        self.createdAt = createdAt
        self.lastUsedAt = lastUsedAt
        self.byteSize = byteSize
        self.isInput = isInput
        self.isOutput = isOutput
        self.isConstant = isConstant
    }

    /// Returns true if this tensor is live at the given operation index.
    public func isLiveAt(_ opIndex: Int) -> Bool {
        return opIndex >= createdAt && opIndex <= lastUsedAt
    }
}

// MARK: - Memory Interference Graph

/// Graph representing which tensors cannot share memory (for memory planning).
/// Named differently to avoid conflict with existing InterferenceGraph in BufferAssignment.
public struct MemoryInterferenceGraph: Sendable {
    /// Adjacency list: tensor -> set of conflicting tensors
    private var adjacency: [TensorID: Set<TensorID>]

    public init() {
        self.adjacency = [:]
    }

    /// Adds an interference edge between two tensors.
    public mutating func addEdge(_ a: TensorID, _ b: TensorID) {
        adjacency[a, default: []].insert(b)
        adjacency[b, default: []].insert(a)
    }

    /// Returns true if two tensors conflict (cannot share memory).
    public func conflicts(_ a: TensorID, _ b: TensorID) -> Bool {
        return adjacency[a]?.contains(b) ?? false
    }

    /// Returns all tensors that conflict with the given tensor.
    public func neighbors(_ tensor: TensorID) -> Set<TensorID> {
        return adjacency[tensor] ?? []
    }
}

// MARK: - Memory Plan

/// Complete memory plan for zero-allocation execution.
public struct MemoryPlan: Sendable {
    /// Total bytes needed (allocate once).
    public let totalBytes: Int

    /// Offset for each tensor within the unified buffer.
    public let tensorOffsets: [TensorID: Int]

    /// Which tensors share memory (for debugging).
    public let sharingGroups: [[TensorID]]

    /// Execution order that minimizes peak memory.
    public let executionOrder: [OperationID]

    /// Peak memory usage during execution.
    public let peakMemory: Int

    /// Statistics about the memory plan.
    public var statistics: Statistics {
        Statistics(
            totalBytes: totalBytes,
            peakMemory: peakMemory,
            numTensors: tensorOffsets.count,
            numSharingGroups: sharingGroups.count,
            memoryReuse: Double(sharingGroups.filter { $0.count > 1 }.count) / Double(max(1, tensorOffsets.count))
        )
    }

    /// Memory plan statistics.
    public struct Statistics: Sendable {
        public let totalBytes: Int
        public let peakMemory: Int
        public let numTensors: Int
        public let numSharingGroups: Int
        public let memoryReuse: Double
    }

    public init(
        totalBytes: Int,
        tensorOffsets: [TensorID: Int],
        sharingGroups: [[TensorID]],
        executionOrder: [OperationID],
        peakMemory: Int
    ) {
        self.totalBytes = totalBytes
        self.tensorOffsets = tensorOffsets
        self.sharingGroups = sharingGroups
        self.executionOrder = executionOrder
        self.peakMemory = peakMemory
    }
}

// MARK: - Static Memory Planner

/// Computes static memory layouts for zero-allocation execution.
///
/// The planner analyzes tensor lifetimes and finds an optimal memory layout
/// where tensors that don't overlap in time can share memory.
public final class StaticMemoryPlanner: @unchecked Sendable {

    /// Configuration for memory planning.
    public struct Config: Sendable {
        /// Buffer alignment requirement (Metal typically needs 256).
        public var bufferAlignment: Int = 256

        /// Page alignment for total buffer size.
        public var pageAlignment: Int = 4096

        /// Whether to optimize execution order for memory.
        public var optimizeExecutionOrder: Bool = true

        /// Maximum iterations for optimization.
        public var maxOptimizationIterations: Int = 100

        public init() {}
    }

    private let config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    // MARK: - Planning

    /// Creates a memory plan for the given HLO function.
    ///
    /// - Parameter function: The function to plan memory for.
    /// - Returns: A complete memory plan.
    public func plan(_ function: HLOFunction) -> MemoryPlan {
        // 1. Compute tensor lifetimes
        let lifetimes = computeLifetimes(function)

        // 2. Optionally optimize execution order
        let order: [OperationID]
        let orderedLifetimes: [TensorID: ScheduledTensorLifetime]

        if config.optimizeExecutionOrder {
            order = optimizeExecutionOrder(function, lifetimes: lifetimes)
            orderedLifetimes = computeLifetimesWithOrder(function, order: order)
        } else {
            order = Array(0..<function.operations.count)
            orderedLifetimes = lifetimes
        }

        // 3. Build interference graph
        let interference = buildInterferenceGraph(orderedLifetimes)

        // 4. Filter out inputs and constants - they come from external buffers, not the unified buffer
        let intermediateLifetimes = orderedLifetimes.filter { !$0.value.isInput && !$0.value.isConstant }

        // 5. Assign offsets using best-fit decreasing bin packing (only for intermediates)
        let (offsets, totalSize, peakMemory) = assignOffsets(intermediateLifetimes, interference: interference)

        // 5. Identify sharing groups
        let groups = findSharingGroups(offsets, interference: interference, lifetimes: orderedLifetimes)

        return MemoryPlan(
            totalBytes: align(totalSize, to: config.pageAlignment),
            tensorOffsets: offsets,
            sharingGroups: groups,
            executionOrder: order,
            peakMemory: peakMemory
        )
    }

    // MARK: - Lifetime Analysis

    /// Computes tensor lifetimes for a function.
    private func computeLifetimes(_ function: HLOFunction) -> [TensorID: ScheduledTensorLifetime] {
        var lifetimes: [TensorID: ScheduledTensorLifetime] = [:]

        // Inputs are live from -1 (before all ops) to their last use
        for input in function.inputs {
            let lastUse = findLastUse(input.name, in: function)
            let byteSize = input.type.byteCount
            lifetimes[input.name] = ScheduledTensorLifetime(
                createdAt: -1,
                lastUsedAt: lastUse,
                byteSize: byteSize,
                isInput: true
            )
        }

        // Track operation results
        for (index, op) in function.operations.enumerated() {
            let lastUse = findLastUse(op.result, in: function, startingFrom: index)
            let isOutput = function.returnValues.contains(op.result)
            let byteSize = op.resultType.byteCount
            let isConstant = op.kind == .constant

            lifetimes[op.result] = ScheduledTensorLifetime(
                createdAt: index,
                lastUsedAt: isOutput ? function.operations.count : lastUse,
                byteSize: byteSize,
                isOutput: isOutput,
                isConstant: isConstant
            )
        }

        return lifetimes
    }

    /// Computes lifetimes with a specific execution order.
    private func computeLifetimesWithOrder(_ function: HLOFunction, order: [OperationID]) -> [TensorID: ScheduledTensorLifetime] {
        var lifetimes: [TensorID: ScheduledTensorLifetime] = [:]

        // Build index mapping
        var orderIndex: [OperationID: Int] = [:]
        for (i, opID) in order.enumerated() {
            orderIndex[opID] = i
        }

        // Inputs
        for input in function.inputs {
            let lastUse = findLastUseInOrder(input.name, in: function, orderIndex: orderIndex)
            lifetimes[input.name] = ScheduledTensorLifetime(
                createdAt: -1,
                lastUsedAt: lastUse,
                byteSize: input.type.byteCount,
                isInput: true
            )
        }

        // Operations
        for (originalIndex, op) in function.operations.enumerated() {
            let scheduledIndex = orderIndex[originalIndex] ?? originalIndex
            let lastUse = findLastUseInOrder(op.result, in: function, orderIndex: orderIndex, startingFrom: scheduledIndex)
            let isOutput = function.returnValues.contains(op.result)
            let isConstant = op.kind == .constant

            lifetimes[op.result] = ScheduledTensorLifetime(
                createdAt: scheduledIndex,
                lastUsedAt: isOutput ? order.count : lastUse,
                byteSize: op.resultType.byteCount,
                isOutput: isOutput,
                isConstant: isConstant
            )
        }

        return lifetimes
    }

    /// Finds the last use of a tensor.
    private func findLastUse(_ tensorID: TensorID, in function: HLOFunction, startingFrom: Int = 0) -> Int {
        var lastUse = startingFrom

        for (index, op) in function.operations.enumerated() where index >= startingFrom {
            if op.operands.contains(tensorID) {
                lastUse = max(lastUse, index)
            }
        }

        // Check return values
        if function.returnValues.contains(tensorID) {
            lastUse = function.operations.count
        }

        return lastUse
    }

    /// Finds the last use with a custom execution order.
    private func findLastUseInOrder(_ tensorID: TensorID, in function: HLOFunction, orderIndex: [OperationID: Int], startingFrom: Int = 0) -> Int {
        var lastUse = startingFrom

        for (originalIndex, op) in function.operations.enumerated() {
            if op.operands.contains(tensorID) {
                let scheduledIndex = orderIndex[originalIndex] ?? originalIndex
                if scheduledIndex >= startingFrom {
                    lastUse = max(lastUse, scheduledIndex)
                }
            }
        }

        if function.returnValues.contains(tensorID) {
            lastUse = orderIndex.values.max() ?? function.operations.count
        }

        return lastUse
    }

    // MARK: - Execution Order Optimization

    /// Optimizes execution order to minimize peak memory.
    private func optimizeExecutionOrder(_ function: HLOFunction, lifetimes: [TensorID: ScheduledTensorLifetime]) -> [OperationID] {
        // Build dependency graph
        var dependencies: [OperationID: Set<OperationID>] = [:]
        var produced: [TensorID: OperationID] = [:]

        // Map tensors to their producing operations
        for (index, op) in function.operations.enumerated() {
            produced[op.result] = index
            dependencies[index] = []
        }

        // Build dependency edges
        for (index, op) in function.operations.enumerated() {
            for operand in op.operands {
                if let producer = produced[operand] {
                    dependencies[index]?.insert(producer)
                }
            }
        }

        // Greedy scheduling: pick operation that minimizes memory delta
        var scheduled: [OperationID] = []
        var completed: Set<OperationID> = []
        var available: Set<TensorID> = Set(function.inputs.map { $0.name })

        while scheduled.count < function.operations.count {
            // Find ready operations
            var ready: [OperationID] = []
            for (opID, deps) in dependencies {
                if !completed.contains(opID) && deps.isSubset(of: completed) {
                    ready.append(opID)
                }
            }

            if ready.isEmpty { break }

            // Pick operation with best memory delta
            var bestOp = ready[0]
            var bestDelta = Int.max

            for opID in ready {
                let op = function.operations[opID]
                let allocate = op.resultType.byteCount

                // Memory freed = operands whose last use is this operation
                var freed = 0
                for operand in op.operands {
                    if let lifetime = lifetimes[operand],
                       lifetime.lastUsedAt == opID,
                       !lifetime.isOutput {
                        freed += lifetime.byteSize
                    }
                }

                let delta = allocate - freed
                if delta < bestDelta {
                    bestDelta = delta
                    bestOp = opID
                }
            }

            // Schedule the best operation
            scheduled.append(bestOp)
            completed.insert(bestOp)
            available.insert(function.operations[bestOp].result)
        }

        return scheduled
    }

    // MARK: - Interference Graph

    /// Builds an interference graph for tensors.
    private func buildInterferenceGraph(_ lifetimes: [TensorID: ScheduledTensorLifetime]) -> MemoryInterferenceGraph {
        var graph = MemoryInterferenceGraph()
        let tensorIDs = Array(lifetimes.keys)

        for i in 0..<tensorIDs.count {
            for j in (i + 1)..<tensorIDs.count {
                let a = tensorIDs[i]
                let b = tensorIDs[j]

                guard let lifetimeA = lifetimes[a],
                      let lifetimeB = lifetimes[b] else { continue }

                // Check if lifetimes overlap
                if lifetimesOverlap(lifetimeA, lifetimeB) {
                    graph.addEdge(a, b)
                }
            }
        }

        return graph
    }

    /// Checks if two lifetimes overlap.
    private func lifetimesOverlap(_ a: ScheduledTensorLifetime, _ b: ScheduledTensorLifetime) -> Bool {
        return a.createdAt <= b.lastUsedAt && b.createdAt <= a.lastUsedAt
    }

    // MARK: - Offset Assignment

    /// Assigns memory offsets using best-fit decreasing bin packing.
    private func assignOffsets(
        _ lifetimes: [TensorID: ScheduledTensorLifetime],
        interference: MemoryInterferenceGraph
    ) -> (offsets: [TensorID: Int], totalSize: Int, peakMemory: Int) {
        var offsets: [TensorID: Int] = [:]
        var allocatedRegions: [(start: Int, end: Int, tensor: TensorID)] = []

        // Sort tensors by size (largest first) for better packing
        let sortedTensors = lifetimes.sorted { $0.value.byteSize > $1.value.byteSize }

        for (tensorID, lifetime) in sortedTensors {
            let size = lifetime.byteSize
            let alignment = config.bufferAlignment

            // Find best offset that doesn't conflict
            var bestOffset: Int?
            var bestWaste = Int.max

            // Candidate offsets: 0 and end of each existing region
            var candidateOffsets = [0]
            for (_, end, _) in allocatedRegions {
                candidateOffsets.append(align(end, to: alignment))
            }

            for offset in candidateOffsets {
                let alignedOffset = align(offset, to: alignment)
                let regionEnd = alignedOffset + size

                // Check for conflicts
                var hasConflict = false
                for (start, end, existingTensor) in allocatedRegions {
                    // Spatial overlap?
                    if alignedOffset < end && regionEnd > start {
                        // Temporal overlap (interference)?
                        if interference.conflicts(tensorID, existingTensor) {
                            hasConflict = true
                            break
                        }
                    }
                }

                if !hasConflict {
                    let waste = alignedOffset - offset
                    if waste < bestWaste || (waste == bestWaste && alignedOffset < (bestOffset ?? Int.max)) {
                        bestWaste = waste
                        bestOffset = alignedOffset
                    }
                }
            }

            let finalOffset = bestOffset ?? align(allocatedRegions.last?.end ?? 0, to: alignment)
            offsets[tensorID] = finalOffset
            allocatedRegions.append((finalOffset, finalOffset + size, tensorID))
            allocatedRegions.sort { $0.start < $1.start }
        }

        let totalSize = allocatedRegions.map { $0.end }.max() ?? 0

        // Compute peak memory (maximum live memory at any point)
        let peakMemory = computePeakMemory(lifetimes)

        return (offsets, totalSize, peakMemory)
    }

    /// Computes the peak memory usage.
    private func computePeakMemory(_ lifetimes: [TensorID: ScheduledTensorLifetime]) -> Int {
        guard !lifetimes.isEmpty else { return 0 }

        let maxOp = lifetimes.values.map { $0.lastUsedAt }.max() ?? 0

        var peak = 0
        for opIndex in -1...maxOp {
            var liveMemory = 0
            for (_, lifetime) in lifetimes {
                if lifetime.isLiveAt(opIndex) {
                    liveMemory += lifetime.byteSize
                }
            }
            peak = max(peak, liveMemory)
        }

        return peak
    }

    // MARK: - Sharing Groups

    /// Finds groups of tensors that share memory.
    private func findSharingGroups(
        _ offsets: [TensorID: Int],
        interference: MemoryInterferenceGraph,
        lifetimes: [TensorID: ScheduledTensorLifetime]
    ) -> [[TensorID]] {
        // Group tensors by their offset
        var offsetGroups: [Int: [TensorID]] = [:]
        for (tensorID, offset) in offsets {
            offsetGroups[offset, default: []].append(tensorID)
        }

        // Only report groups where tensors actually share memory (same offset, no interference)
        var sharingGroups: [[TensorID]] = []

        for (_, tensors) in offsetGroups {
            if tensors.count > 1 {
                // Verify they don't interfere
                var canShare = true
                for i in 0..<tensors.count {
                    for j in (i + 1)..<tensors.count {
                        if interference.conflicts(tensors[i], tensors[j]) {
                            canShare = false
                            break
                        }
                    }
                    if !canShare { break }
                }

                if canShare {
                    sharingGroups.append(tensors)
                }
            }
        }

        return sharingGroups
    }

    // MARK: - Utilities

    /// Aligns a value to the given alignment.
    private func align(_ value: Int, to alignment: Int) -> Int {
        guard alignment > 0 else { return value }
        return (value + alignment - 1) / alignment * alignment
    }
}
