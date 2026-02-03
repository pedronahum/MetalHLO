// ReductionKernelGenerator.swift
// MetalHLOCore
//
// Specialized reduction kernel generator with SIMD optimizations.

import Foundation

/// Generates optimized reduction kernels for different reduction patterns.
///
/// This generator creates axis-specific reduction kernels that leverage:
/// - SIMD intrinsics (simd_sum, simd_max, simd_min) for efficient warp-level reduction
/// - Vectorized loads (float4) for better memory bandwidth
/// - Threadgroup-optimal dispatch patterns
///
/// Three main patterns are supported:
/// - Row reduction: reduce over the last axis (most common in neural networks)
/// - Column reduction: reduce over the first axis
/// - Global reduction: reduce all elements to a single value
public struct ReductionKernelGenerator {

    // MARK: - Types

    /// Reduction operation type.
    public enum ReductionOp: String {
        case sum
        case max
        case min
        case mean
        case prod

        var simdIntrinsic: String {
            switch self {
            case .sum, .mean: return "simd_sum"
            case .max: return "simd_max"
            case .min: return "simd_min"
            case .prod: return "simd_product"
            }
        }

        var identity: String {
            switch self {
            case .sum, .mean: return "0.0f"
            case .max: return "-INFINITY"
            case .min: return "INFINITY"
            case .prod: return "1.0f"
            }
        }

        var binaryOp: String {
            switch self {
            case .sum, .mean: return "a + b"
            case .max: return "max(a, b)"
            case .min: return "min(a, b)"
            case .prod: return "a * b"
            }
        }

        var accumOp: String {
            switch self {
            case .sum, .mean: return "accum += val;"
            case .max: return "accum = max(accum, val);"
            case .min: return "accum = min(accum, val);"
            case .prod: return "accum *= val;"
            }
        }
    }

    /// Reduction pattern based on axis configuration.
    public enum ReductionPattern {
        /// Reduce over the last axis (e.g., [M, N] -> [M])
        case row
        /// Reduce over the first axis (e.g., [M, N] -> [N])
        case column
        /// Reduce all elements to scalar (e.g., [M, N] -> [])
        case global
        /// General reduction over arbitrary axes
        case general
    }

    // MARK: - Analysis

    /// Analyzes the reduction pattern from input shape and reduction dimensions.
    public static func analyzePattern(inputShape: [Int], reduceDims: [Int]) -> ReductionPattern {
        guard !inputShape.isEmpty else { return .global }

        let rank = inputShape.count

        // Single element - treat as global
        if inputShape.reduce(1, *) == 1 {
            return .global
        }

        // Global reduction: all dims or single dim tensor
        if reduceDims.count == rank || rank == 1 {
            return .global
        }

        // Row reduction: reducing over last axis only
        if reduceDims.count == 1 && reduceDims[0] == rank - 1 {
            return .row
        }

        // Column reduction: reducing over first axis only
        if reduceDims.count == 1 && reduceDims[0] == 0 {
            return .column
        }

        return .general
    }

    // MARK: - Kernel Generation

    /// Generates an optimized row reduction kernel.
    ///
    /// Row reduction sums/max/min across the last axis.
    /// Each thread handles one row, using scalar loads for correctness
    /// (vectorized loads require 16-byte alignment which isn't guaranteed
    /// when row size isn't a multiple of 4).
    /// Uses same buffer interface as general kernel for compatibility.
    public static func generateRowReductionKernel(
        op: ReductionOp,
        entryPoint: String = "row_reduce"
    ) -> String {
        return """
        #include <metal_stdlib>
        using namespace metal;

        // Optimized row reduction kernel
        // Reduces over the last axis: [M, N] -> [M]
        // Each thread handles one complete row
        kernel void \(entryPoint)(
            device const float* input [[buffer(0)]],
            device const float* initValue [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& outputCount [[buffer(3)]],
            constant uint& reduceSize [[buffer(4)]],
            constant uint& innerSize [[buffer(5)]],
            uint gid [[thread_position_in_grid]])
        {
            // For row reduction: outputCount = rows, reduceSize = cols, innerSize = 1
            if (gid >= outputCount) return;

            uint base = gid * reduceSize;
            float accum = \(op.identity);

            // Use scalar loads to avoid alignment issues when reduceSize is not a multiple of 4
            // Loop unrolling by 4 for performance without requiring alignment
            uint i = 0;
            uint reduceSize4 = reduceSize & ~3u;  // Round down to multiple of 4
            for (; i < reduceSize4; i += 4) {
                float val = input[base + i]; \(op.accumOp)
                val = input[base + i + 1]; \(op.accumOp)
                val = input[base + i + 2]; \(op.accumOp)
                val = input[base + i + 3]; \(op.accumOp)
            }

            // Handle remainder (0-3 elements)
            for (; i < reduceSize; i++) {
                float val = input[base + i];
                \(op.accumOp)
            }

            // Combine with init value
            float a = accum;
            float b = initValue[0];
            output[gid] = \(op.binaryOp);
        }
        """
    }

    /// Generates an optimized column reduction kernel.
    ///
    /// Column reduction sums/max/min across the first axis.
    /// Uses same buffer interface as general kernel for compatibility.
    public static func generateColumnReductionKernel(
        op: ReductionOp,
        entryPoint: String = "col_reduce"
    ) -> String {
        return """
        #include <metal_stdlib>
        using namespace metal;

        // Optimized column reduction kernel
        // Reduces over the first axis: [M, N] -> [N]
        // Each thread handles one column
        kernel void \(entryPoint)(
            device const float* input [[buffer(0)]],
            device const float* initValue [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& outputCount [[buffer(3)]],
            constant uint& reduceSize [[buffer(4)]],
            constant uint& innerSize [[buffer(5)]],
            uint gid [[thread_position_in_grid]])
        {
            // For column reduction: outputCount = cols, reduceSize = rows, innerSize = cols
            // But actually we need to be careful - column reduction is reduce over dim 0
            // Input layout: [rows, cols], reducing over rows -> [cols]
            // So we iterate over rows (reduceSize), with stride = innerSize (= cols)
            if (gid >= outputCount) return;

            float accum = \(op.identity);

            // Each thread sums its column
            // innerSize == outputCount for column reduction
            for (uint r = 0; r < reduceSize; r++) {
                float val = input[r * innerSize + gid];
                \(op.accumOp)
            }

            // Combine with init value
            float a = accum;
            float b = initValue[0];
            output[gid] = \(op.binaryOp);
        }
        """
    }

    /// Generates an optimized global reduction kernel.
    ///
    /// Global reduction reduces all elements to a single scalar.
    /// Uses multi-level reduction: thread -> SIMD -> threadgroup -> global.
    public static func generateGlobalReductionKernel(
        op: ReductionOp,
        entryPoint: String = "global_reduce"
    ) -> String {
        return """
        #include <metal_stdlib>
        using namespace metal;

        // Optimized global reduction kernel using SIMD intrinsics
        // Reduces all elements to a single value
        kernel void \(entryPoint)(
            device const float* input [[buffer(0)]],
            device const float* initValue [[buffer(1)]],
            device atomic_float* output [[buffer(2)]],
            constant uint& count [[buffer(3)]],
            uint gid [[thread_position_in_grid]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgid [[threadgroup_position_in_grid]],
            uint simd_lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]],
            uint tg_size [[threads_per_threadgroup]],
            uint num_tgs [[threadgroups_per_grid]])
        {
            // Shared memory for threadgroup reduction
            threadgroup float shared[32];  // One per SIMD group

            float accum = \(op.identity);

            // Phase 1: Each thread reduces multiple elements
            uint stride = tg_size * num_tgs;
            for (uint i = gid; i < count; i += stride) {
                float val = input[i];
                \(op.accumOp)
            }

            // Phase 2: SIMD reduction within simdgroup
            accum = \(op.simdIntrinsic)(accum);

            // Phase 3: First lane of each simdgroup stores to shared memory
            if (simd_lane == 0) {
                shared[simd_group] = accum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 4: First simdgroup reduces shared memory
            if (tid < 32) {
                float val = (tid < (tg_size / 32)) ? shared[tid] : \(op.identity);
                val = \(op.simdIntrinsic)(val);

                // Thread 0 does atomic update to global result
                if (tid == 0) {
                    \(atomicOp(for: op))
                }
            }
        }
        """
    }

    /// Returns the atomic operation for global reduction.
    private static func atomicOp(for op: ReductionOp) -> String {
        switch op {
        case .sum, .mean:
            return "atomic_fetch_add_explicit(output, val, memory_order_relaxed);"
        case .max:
            return """
            float old = atomic_load_explicit(output, memory_order_relaxed);
                        while (val > old) {
                            if (atomic_compare_exchange_weak_explicit(output, &old, val, memory_order_relaxed, memory_order_relaxed)) break;
                        }
            """
        case .min:
            return """
            float old = atomic_load_explicit(output, memory_order_relaxed);
                        while (val < old) {
                            if (atomic_compare_exchange_weak_explicit(output, &old, val, memory_order_relaxed, memory_order_relaxed)) break;
                        }
            """
        case .prod:
            return """
            float old = atomic_load_explicit(output, memory_order_relaxed);
                        while (!atomic_compare_exchange_weak_explicit(output, &old, old * val, memory_order_relaxed, memory_order_relaxed)) {}
            """
        }
    }

    /// Generates a reduction kernel based on the detected pattern.
    public static func generateKernel(
        inputShape: [Int],
        reduceDims: [Int],
        op: ReductionOp,
        entryPoint: String = "reduce"
    ) -> (source: String, entryPoint: String, pattern: ReductionPattern) {
        let pattern = analyzePattern(inputShape: inputShape, reduceDims: reduceDims)

        let source: String
        let kernelName: String

        switch pattern {
        case .row:
            source = generateRowReductionKernel(op: op, entryPoint: entryPoint + "_row")
            kernelName = entryPoint + "_row"
        case .column:
            source = generateColumnReductionKernel(op: op, entryPoint: entryPoint + "_col")
            kernelName = entryPoint + "_col"
        case .global:
            source = generateGlobalReductionKernel(op: op, entryPoint: entryPoint + "_global")
            kernelName = entryPoint + "_global"
        case .general:
            // Fall back to general reduction (not implemented here - use existing)
            source = ""
            kernelName = ""
        }

        return (source, kernelName, pattern)
    }
}
