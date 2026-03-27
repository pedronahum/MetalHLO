// FusedMatmul.swift
// HeterogeneousFusion
//
// Convenience API for heterogeneous matmul execution.
// Builds a PartitionDescriptor and delegates to FusedExecutor.
//
// Phase 1 API is preserved for backward compatibility.
// Phase 2 adds the descriptor-based path.

import Metal
import QuartzCore

/// Convenience wrapper that builds PartitionDescriptors for matmul
/// and delegates execution to FusedExecutor.
public final class FusedMatmul: @unchecked Sendable {

    private let executor: FusedExecutor

    public init(device: MTLDevice) throws {
        self.executor = try FusedExecutor(device: device)
    }

    // MARK: - Phase 2 API (descriptor-based)

    /// Execute a matmul using an explicit PartitionDescriptor.
    public func execute(
        descriptor: PartitionDescriptor,
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer
    ) -> FusedProfile {
        executor.execute(descriptor: descriptor, inputBuffers: [A, B], outputBuffer: C)
    }

    // MARK: - Phase 1 API (fraction-based, builds descriptor internally)

    /// Execute matmul with a two-way GPU+MPS split.
    public func executeTwoWay(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        shape: MatrixShape,
        gpuFraction: Double = 0.5
    ) -> FusedProfile {
        let descriptor = PartitionDescriptor.matmul(
            shape: shape,
            fractions: [
                (.gpu, gpuFraction),
                (.mps, 1.0 - gpuFraction),
            ]
        )
        return executor.execute(descriptor: descriptor, inputBuffers: [A, B], outputBuffer: C)
    }

    /// Execute matmul with a three-way GPU+MPS+CPU split.
    public func executeThreeWay(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        shape: MatrixShape,
        gpuFraction: Double = 0.34,
        mpsFraction: Double = 0.33
    ) -> FusedProfile {
        let cpuFraction = 1.0 - gpuFraction - mpsFraction
        let descriptor = PartitionDescriptor.matmul(
            shape: shape,
            fractions: [
                (.gpu, gpuFraction),
                (.mps, mpsFraction),
                (.cpu, cpuFraction),
            ]
        )
        return executor.execute(descriptor: descriptor, inputBuffers: [A, B], outputBuffer: C)
    }

    // MARK: - Column-Split API

    /// Execute matmul with column-split three-way GPU+MPS+CPU.
    public func executeThreeWayColumnSplit(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        shape: MatrixShape,
        gpuFraction: Double = 0.34,
        mpsFraction: Double = 0.33
    ) -> FusedProfile {
        let cpuFraction = 1.0 - gpuFraction - mpsFraction
        let descriptor = PartitionDescriptor.matmulColumnSplit(
            shape: shape,
            fractions: [
                (.gpu, gpuFraction),
                (.mps, mpsFraction),
                (.cpu, cpuFraction),
            ]
        )
        return executor.execute(descriptor: descriptor, inputBuffers: [A, B], outputBuffer: C)
    }

    /// Execute matmul with auto-selected split dimension (row or column).
    public func executeAutoSplit(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        shape: MatrixShape,
        gpuFraction: Double = 0.34,
        mpsFraction: Double = 0.33
    ) -> FusedProfile {
        let cpuFraction = 1.0 - gpuFraction - mpsFraction
        let descriptor = PartitionDescriptor.matmulAutoSplit(
            shape: shape,
            fractions: [
                (.gpu, gpuFraction),
                (.mps, mpsFraction),
                (.cpu, cpuFraction),
            ]
        )
        return executor.execute(descriptor: descriptor, inputBuffers: [A, B], outputBuffer: C)
    }
}
