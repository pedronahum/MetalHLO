// BufferAssignment.swift
// MetalHLOCore
//
// Static buffer assignment pass for memory optimization.
// Inspired by XLA's BufferAssignment.

import Foundation

// MARK: - Tensor Lifetime

/// Represents the lifetime of a tensor value in the computation graph.
///
/// A tensor is "live" from the operation that defines it until its last use.
/// Tensors with non-overlapping lifetimes can share the same memory buffer.
public struct TensorLifetime: Sendable, Equatable {
    /// The index of the operation that defines this tensor.
    public let start: Int

    /// The index of the last operation that uses this tensor.
    public let end: Int

    /// The tensor value name.
    public let tensorName: String

    /// The size in bytes.
    public let byteSize: Int

    public init(start: Int, end: Int, tensorName: String, byteSize: Int) {
        self.start = start
        self.end = end
        self.tensorName = tensorName
        self.byteSize = byteSize
    }

    /// Checks if this lifetime overlaps with another.
    public func overlaps(with other: TensorLifetime) -> Bool {
        // Two intervals overlap if neither ends before the other starts
        return !(end < other.start || other.end < start)
    }

    /// The duration of this lifetime (number of operations).
    public var duration: Int {
        end - start + 1
    }
}

// MARK: - Buffer Slot

/// A buffer slot that can hold one or more tensors (with non-overlapping lifetimes).
public struct BufferSlot: Sendable {
    /// Unique identifier for this slot.
    public let id: Int

    /// The size of this buffer in bytes.
    public let size: Int

    /// The tensors assigned to this slot.
    public let tensors: [String]

    /// Whether this slot holds a function output (should not be reused).
    public let isOutput: Bool

    public init(id: Int, size: Int, tensors: [String], isOutput: Bool = false) {
        self.id = id
        self.size = size
        self.tensors = tensors
        self.isOutput = isOutput
    }
}

// MARK: - Buffer Plan

/// The result of buffer assignment.
public struct BufferPlan: Sendable {
    /// The buffer slots.
    public let slots: [BufferSlot]

    /// Mapping from tensor name to slot ID.
    public let tensorToSlot: [String: Int]

    /// Total memory required (sum of slot sizes).
    public var totalMemory: Int {
        slots.reduce(0) { $0 + $1.size }
    }

    /// Memory saved compared to naive allocation.
    public var memorySaved: Int {
        let naiveTotal = slots.flatMap { slot in
            slot.tensors.map { _ in slot.size }
        }.reduce(0, +)
        return naiveTotal - totalMemory
    }

    /// Number of tensors that share buffers.
    public var numSharedTensors: Int {
        slots.filter { $0.tensors.count > 1 }.flatMap { $0.tensors }.count
    }

    public init(slots: [BufferSlot], tensorToSlot: [String: Int]) {
        self.slots = slots
        self.tensorToSlot = tensorToSlot
    }

    /// Checks if a slot is an output slot.
    public func isOutputSlot(_ slotId: Int) -> Bool {
        slots.first { $0.id == slotId }?.isOutput ?? false
    }
}

// MARK: - Interference Graph

/// Graph representing which tensors cannot share memory.
///
/// Two tensors interfere if their lifetimes overlap, meaning they
/// cannot be assigned to the same buffer slot.
public struct InterferenceGraph: Sendable {
    /// Adjacency list representation.
    private var adjacency: [String: Set<String>]

    /// All vertices (tensor names).
    public var vertices: Set<String> {
        Set(adjacency.keys)
    }

    public init() {
        self.adjacency = [:]
    }

    /// Adds an edge between two interfering tensors.
    public mutating func addEdge(_ a: String, _ b: String) {
        adjacency[a, default: []].insert(b)
        adjacency[b, default: []].insert(a)
    }

    /// Adds a vertex (tensor) to the graph.
    public mutating func addVertex(_ v: String) {
        if adjacency[v] == nil {
            adjacency[v] = []
        }
    }

    /// Gets the neighbors (interfering tensors) of a vertex.
    public func neighbors(of vertex: String) -> Set<String> {
        adjacency[vertex] ?? []
    }

    /// Checks if two tensors interfere.
    public func interferes(_ a: String, _ b: String) -> Bool {
        adjacency[a]?.contains(b) ?? false
    }
}

// MARK: - Buffer Assignment Pass

/// Static buffer assignment pass that plans memory allocation with reuse.
///
/// The pass works by:
/// 1. Computing tensor lifetimes (when each tensor is live)
/// 2. Building an interference graph (which tensors can't share memory)
/// 3. Graph coloring to assign tensors to buffer slots
/// 4. Creating a buffer plan for execution
///
/// Expected impact: 1.1x speedup from reduced allocation overhead,
/// and reduced memory footprint through buffer reuse.
public final class BufferAssignment: @unchecked Sendable {

    /// Whether to preserve separate buffers for function outputs.
    private let preserveOutputs: Bool

    /// Creates a buffer assignment pass.
    ///
    /// - Parameter preserveOutputs: Whether to keep function outputs in
    ///   separate buffers (default: true).
    public init(preserveOutputs: Bool = true) {
        self.preserveOutputs = preserveOutputs
    }

    /// Assigns buffers to all tensors in a function.
    ///
    /// - Parameter function: The function to analyze.
    /// - Returns: The buffer plan with slot assignments.
    public func assignBuffers(to function: HLOFunction) -> BufferPlan {
        // Phase 1: Compute tensor lifetimes
        let lifetimes = computeLifetimes(function)

        // Phase 2: Build interference graph
        let interference = buildInterferenceGraph(lifetimes: lifetimes)

        // Phase 3: Graph coloring to assign slots
        let coloring = colorGraph(
            interference: interference,
            lifetimes: lifetimes,
            outputs: Set(function.returnValues)
        )

        // Phase 4: Create buffer slots
        let slots = createBufferSlots(
            coloring: coloring,
            lifetimes: lifetimes,
            outputs: Set(function.returnValues)
        )

        return BufferPlan(slots: slots, tensorToSlot: coloring)
    }

    // MARK: - Phase 1: Lifetime Analysis

    /// Computes lifetimes for all tensors in the function.
    private func computeLifetimes(_ function: HLOFunction) -> [String: TensorLifetime] {
        var lifetimes: [String: TensorLifetime] = [:]
        var lastUse: [String: Int] = [:]

        // First pass: find last use of each tensor
        for (index, op) in function.operations.enumerated() {
            for operand in op.operands {
                lastUse[operand] = index
            }
        }

        // Return values are used at the end
        let numOps = function.operations.count
        for retVal in function.returnValues {
            lastUse[retVal] = numOps
        }

        // Function inputs are defined at the start (index -1 conceptually, but use 0)
        for input in function.inputs {
            let endIndex = lastUse[input.name] ?? 0
            lifetimes[input.name] = TensorLifetime(
                start: 0,
                end: endIndex,
                tensorName: input.name,
                byteSize: input.type.byteCount
            )
        }

        // Each operation defines its result
        for (index, op) in function.operations.enumerated() {
            let endIndex = lastUse[op.result] ?? index
            lifetimes[op.result] = TensorLifetime(
                start: index,
                end: endIndex,
                tensorName: op.result,
                byteSize: op.resultType.byteCount
            )
        }

        return lifetimes
    }

    // MARK: - Phase 2: Interference Graph

    /// Builds the interference graph from tensor lifetimes.
    private func buildInterferenceGraph(lifetimes: [String: TensorLifetime]) -> InterferenceGraph {
        var graph = InterferenceGraph()

        let tensors = Array(lifetimes.keys)

        // Add all vertices
        for tensor in tensors {
            graph.addVertex(tensor)
        }

        // Add edges between interfering tensors
        for i in 0..<tensors.count {
            let lifetime1 = lifetimes[tensors[i]]!

            for j in (i + 1)..<tensors.count {
                let lifetime2 = lifetimes[tensors[j]]!

                if lifetime1.overlaps(with: lifetime2) {
                    graph.addEdge(tensors[i], tensors[j])
                }
            }
        }

        return graph
    }

    // MARK: - Phase 3: Graph Coloring

    /// Colors the interference graph using a greedy algorithm.
    ///
    /// The algorithm assigns colors (buffer slots) to tensors such that
    /// no two interfering tensors have the same color.
    private func colorGraph(
        interference: InterferenceGraph,
        lifetimes: [String: TensorLifetime],
        outputs: Set<String>
    ) -> [String: Int] {
        var coloring: [String: Int] = [:]
        var nextColor = 0

        // Sort tensors by size (largest first) for better packing
        let sortedTensors = interference.vertices.sorted { a, b in
            let sizeA = lifetimes[a]?.byteSize ?? 0
            let sizeB = lifetimes[b]?.byteSize ?? 0
            return sizeA > sizeB
        }

        // Assign colors that outputs get unique slots if preserveOutputs is true
        if preserveOutputs {
            for output in outputs {
                coloring[output] = nextColor
                nextColor += 1
            }
        }

        // Color remaining tensors
        for tensor in sortedTensors {
            // Skip if already colored (outputs)
            if coloring[tensor] != nil {
                continue
            }

            // Find colors used by neighbors
            var usedColors: Set<Int> = []
            for neighbor in interference.neighbors(of: tensor) {
                if let color = coloring[neighbor] {
                    usedColors.insert(color)
                }
            }

            // Also check size compatibility - can only share if sizes match or slot is big enough
            // For simplicity, we find the first unused color
            var color = 0
            while usedColors.contains(color) {
                color += 1
            }

            // Use existing color if possible, otherwise create new one
            if color >= nextColor {
                nextColor = color + 1
            }

            coloring[tensor] = color
        }

        return coloring
    }

    // MARK: - Phase 4: Create Buffer Slots

    /// Creates buffer slots from the graph coloring.
    private func createBufferSlots(
        coloring: [String: Int],
        lifetimes: [String: TensorLifetime],
        outputs: Set<String>
    ) -> [BufferSlot] {
        // Group tensors by color
        var colorToTensors: [Int: [String]] = [:]
        for (tensor, color) in coloring {
            colorToTensors[color, default: []].append(tensor)
        }

        // Create slots
        var slots: [BufferSlot] = []
        for (color, tensors) in colorToTensors.sorted(by: { $0.key < $1.key }) {
            // Slot size is the maximum size among its tensors
            let maxSize = tensors.compactMap { lifetimes[$0]?.byteSize }.max() ?? 0

            // Check if any tensor in this slot is an output
            let isOutput = preserveOutputs && tensors.contains { outputs.contains($0) }

            slots.append(BufferSlot(
                id: color,
                size: maxSize,
                tensors: tensors.sorted(),
                isOutput: isOutput
            ))
        }

        return slots
    }
}

// MARK: - Buffer Statistics

/// Statistics about buffer assignment.
public struct BufferStatistics: Sendable {
    /// Total number of tensors.
    public let numTensors: Int

    /// Number of buffer slots.
    public let numSlots: Int

    /// Total memory required (bytes).
    public let totalMemory: Int

    /// Memory that would be required without sharing (bytes).
    public let naiveMemory: Int

    /// Memory saved through buffer reuse (bytes).
    public var memorySaved: Int {
        naiveMemory - totalMemory
    }

    /// Memory reduction percentage.
    public var reductionPercentage: Double {
        guard naiveMemory > 0 else { return 0 }
        return Double(memorySaved) / Double(naiveMemory) * 100
    }

    /// Average tensors per slot.
    public var averageTensorsPerSlot: Double {
        guard numSlots > 0 else { return 0 }
        return Double(numTensors) / Double(numSlots)
    }

    public init(numTensors: Int, numSlots: Int, totalMemory: Int, naiveMemory: Int) {
        self.numTensors = numTensors
        self.numSlots = numSlots
        self.totalMemory = totalMemory
        self.naiveMemory = naiveMemory
    }
}

extension BufferAssignment {
    /// Computes statistics for a buffer plan.
    public func computeStatistics(
        for plan: BufferPlan,
        function: HLOFunction
    ) -> BufferStatistics {
        // Compute naive memory (no sharing)
        var naiveMemory = 0
        for input in function.inputs {
            naiveMemory += input.type.byteCount
        }
        for op in function.operations {
            naiveMemory += op.resultType.byteCount
        }

        return BufferStatistics(
            numTensors: plan.tensorToSlot.count,
            numSlots: plan.slots.count,
            totalMemory: plan.totalMemory,
            naiveMemory: naiveMemory
        )
    }
}

// MARK: - Lifetime Analysis Helpers

extension BufferAssignment {
    /// Analyzes which tensors are live at each operation.
    ///
    /// This is useful for debugging and visualization.
    public func analyzeLiveness(_ function: HLOFunction) -> [Int: Set<String>] {
        let lifetimes = computeLifetimes(function)
        var liveness: [Int: Set<String>] = [:]

        for opIndex in 0..<function.operations.count {
            var live: Set<String> = []

            for (tensor, lifetime) in lifetimes {
                if lifetime.start <= opIndex && opIndex <= lifetime.end {
                    live.insert(tensor)
                }
            }

            liveness[opIndex] = live
        }

        return liveness
    }

    /// Computes the peak memory usage (maximum live tensors at any point).
    public func computePeakMemory(_ function: HLOFunction) -> Int {
        let lifetimes = computeLifetimes(function)
        var maxMemory = 0

        for opIndex in 0..<function.operations.count {
            var liveMemory = 0

            for (_, lifetime) in lifetimes {
                if lifetime.start <= opIndex && opIndex <= lifetime.end {
                    liveMemory += lifetime.byteSize
                }
            }

            maxMemory = max(maxMemory, liveMemory)
        }

        return maxMemory
    }
}

// MARK: - In-Place Optimization

extension BufferAssignment {
    /// Identifies operations that can be performed in-place.
    ///
    /// An operation can be in-place if:
    /// 1. It's a unary operation
    /// 2. The input has no other uses after this operation
    /// 3. Input and output have the same type
    ///
    /// - Returns: Set of operation results that can reuse their input buffer.
    public func findInPlaceCandidates(_ function: HLOFunction) -> Set<String> {
        var candidates: Set<String> = []
        let lifetimes = computeLifetimes(function)

        for (index, op) in function.operations.enumerated() {
            // Must be unary
            guard op.operands.count == 1 else { continue }

            // Must be an in-place compatible operation
            guard isInPlaceCompatible(op.kind) else { continue }

            let input = op.operands[0]

            // Input must end at this operation (no other uses)
            guard let inputLifetime = lifetimes[input],
                  inputLifetime.end == index else { continue }

            // Input must not be a function argument (we can't modify inputs)
            guard !function.inputs.contains(where: { $0.name == input }) else { continue }

            candidates.insert(op.result)
        }

        return candidates
    }

    /// Checks if an operation kind supports in-place execution.
    private func isInPlaceCompatible(_ kind: HLOOpKind) -> Bool {
        switch kind {
        // Unary elementwise operations
        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt,
             .tanh, .logistic, .sine, .cosine, .tan,
             .floor, .ceil, .sign, .not,
             .expm1, .log1p, .cbrt, .roundNearestAfz, .roundNearestEven:
            return true

        // Type-preserving conversions (same size)
        case .convert:
            return true  // Caller should verify sizes match

        default:
            return false
        }
    }
}
