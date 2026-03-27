// MatmulANE.swift
// HeterogeneousFusion
//
// Row-sliceable matrix multiplication via MPS (Metal Performance Shaders).
// On Apple Silicon, MPSMatrixMultiplication may route through the ANE for
// eligible shapes, and at minimum uses Apple's optimized GPU path.

import Metal
import MetalPerformanceShaders
import QuartzCore

/// Performs matmul via MPSMatrixMultiplication.
/// Supports row-slicing: computes C_slice = A_slice @ B.
public final class MatmulANE: @unchecked Sendable {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create command queue for MPS")
        }
        self.commandQueue = queue
    }

    /// Encode a row-sliced matmul into the given command buffer.
    ///
    /// - Parameters:
    ///   - A: Buffer for the A slice [sliceM, K], float32.
    ///   - B: Buffer for B [K, N], float32.
    ///   - C: Buffer for the output slice [sliceM, N], float32.
    ///   - sliceM: Number of rows in this slice.
    ///   - K: Inner dimension.
    ///   - N: Output columns.
    ///   - commandBuffer: The command buffer to encode into.
    public func encode(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        sliceM: Int, K: Int, N: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        let rowBytesA = K * MemoryLayout<Float>.size
        let rowBytesB = N * MemoryLayout<Float>.size
        let rowBytesC = N * MemoryLayout<Float>.size

        let matA = MPSMatrix(
            buffer: A,
            descriptor: MPSMatrixDescriptor(
                rows: sliceM, columns: K,
                rowBytes: rowBytesA,
                dataType: .float32
            )
        )
        let matB = MPSMatrix(
            buffer: B,
            descriptor: MPSMatrixDescriptor(
                rows: K, columns: N,
                rowBytes: rowBytesB,
                dataType: .float32
            )
        )
        let matC = MPSMatrix(
            buffer: C,
            descriptor: MPSMatrixDescriptor(
                rows: sliceM, columns: N,
                rowBytes: rowBytesC,
                dataType: .float32
            )
        )

        let mul = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: sliceM,
            resultColumns: N,
            interiorColumns: K,
            alpha: 1.0,
            beta: 0.0
        )

        mul.encode(commandBuffer: commandBuffer, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
    }

    /// Execute matmul synchronously. Returns wall-clock time in seconds.
    public func execute(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        M: Int, K: Int, N: Int
    ) -> Double {
        guard let cb = commandQueue.makeCommandBuffer() else { return 0 }
        encode(A: A, B: B, C: C, sliceM: M, K: K, N: N, commandBuffer: cb)

        let start = CACurrentMediaTime()
        cb.commit()
        cb.waitUntilCompleted()
        return CACurrentMediaTime() - start
    }

    /// Expose command queue for shared use.
    public var queue: MTLCommandQueue { commandQueue }
}
