// ColumnSplitHelper.swift
// HeterogeneousFusion
//
// Metal kernels for column-split matmul support:
//   - extract_columns: strided B → contiguous B_slice
//   - scatter_columns: contiguous C_slice → strided C
//
// These are lightweight copy kernels that enable column-dimension
// partitioning without modifying the core matmul kernels.

import Metal

/// Provides GPU kernels for extracting and scattering column slices.
public final class ColumnSplitHelper: @unchecked Sendable {

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let extractPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create column split command queue")
        }
        self.queue = q

        let source = ColumnSplitHelper.kernelSource()
        let library = try device.makeLibrary(source: source, options: nil)

        guard let extractFn = library.makeFunction(name: "extract_columns"),
              let scatterFn = library.makeFunction(name: "scatter_columns") else {
            throw HeterogeneousError.metalSetupFailed("Failed to find column split kernels")
        }
        self.extractPipeline = try device.makeComputePipelineState(function: extractFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)
    }

    /// Extract columns [colOffset, colOffset+sliceCols) from a [rows, fullCols] matrix
    /// into a contiguous [rows, sliceCols] output buffer.
    public func extractColumns(
        src: MTLBuffer,
        dst: MTLBuffer, dstOffset: Int = 0,
        rows: Int, fullCols: Int,
        colOffset: Int, sliceCols: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(extractPipeline)
        encoder.setBuffer(src, offset: 0, index: 0)
        encoder.setBuffer(dst, offset: dstOffset, index: 1)
        var r = UInt32(rows), fc = UInt32(fullCols), co = UInt32(colOffset), sc = UInt32(sliceCols)
        encoder.setBytes(&r, length: 4, index: 2)
        encoder.setBytes(&fc, length: 4, index: 3)
        encoder.setBytes(&co, length: 4, index: 4)
        encoder.setBytes(&sc, length: 4, index: 5)

        let total = rows * sliceCols
        let tpg = min(256, total)
        let groups = (total + tpg - 1) / tpg
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Scatter a contiguous [rows, sliceCols] buffer into columns [colOffset, colOffset+sliceCols)
    /// of a [rows, fullCols] destination matrix.
    public func scatterColumns(
        src: MTLBuffer, srcOffset: Int = 0,
        dst: MTLBuffer,
        rows: Int, fullCols: Int,
        colOffset: Int, sliceCols: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(scatterPipeline)
        encoder.setBuffer(src, offset: srcOffset, index: 0)
        encoder.setBuffer(dst, offset: 0, index: 1)
        var r = UInt32(rows), fc = UInt32(fullCols), co = UInt32(colOffset), sc = UInt32(sliceCols)
        encoder.setBytes(&r, length: 4, index: 2)
        encoder.setBytes(&fc, length: 4, index: 3)
        encoder.setBytes(&co, length: 4, index: 4)
        encoder.setBytes(&sc, length: 4, index: 5)

        let total = rows * sliceCols
        let tpg = min(256, total)
        let groups = (total + tpg - 1) / tpg
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// CPU column extraction: strided src → contiguous dst.
    public static func extractColumnsCPU(
        src: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>,
        rows: Int, fullCols: Int,
        colOffset: Int, sliceCols: Int
    ) {
        for r in 0..<rows {
            let srcRow = src + r * fullCols + colOffset
            let dstRow = dst + r * sliceCols
            dstRow.update(from: srcRow, count: sliceCols)
        }
    }

    /// CPU column scatter: contiguous src → strided dst.
    public static func scatterColumnsCPU(
        src: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>,
        rows: Int, fullCols: Int,
        colOffset: Int, sliceCols: Int
    ) {
        for r in 0..<rows {
            let srcRow = src + r * sliceCols
            let dstRow = dst + r * fullCols + colOffset
            dstRow.update(from: srcRow, count: sliceCols)
        }
    }

    /// Create a temporary buffer for column slice operations.
    public func makeTempBuffer(rows: Int, cols: Int) -> MTLBuffer? {
        device.makeBuffer(length: rows * cols * MemoryLayout<Float>.size, options: .storageModeShared)
    }

    private static func kernelSource() -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        // Extract columns [colOffset, colOffset+sliceCols) from [rows, fullCols]
        // into contiguous [rows, sliceCols] output.
        kernel void extract_columns(
            device const float* src [[buffer(0)]],
            device float* dst       [[buffer(1)]],
            constant uint& rows     [[buffer(2)]],
            constant uint& fullCols [[buffer(3)]],
            constant uint& colOffset[[buffer(4)]],
            constant uint& sliceCols[[buffer(5)]],
            uint tid [[thread_position_in_grid]]
        ) {
            uint total = rows * sliceCols;
            if (tid >= total) return;
            uint r = tid / sliceCols;
            uint c = tid % sliceCols;
            dst[r * sliceCols + c] = src[r * fullCols + colOffset + c];
        }

        // Scatter contiguous [rows, sliceCols] into columns [colOffset, colOffset+sliceCols)
        // of [rows, fullCols] destination.
        kernel void scatter_columns(
            device const float* src [[buffer(0)]],
            device float* dst       [[buffer(1)]],
            constant uint& rows     [[buffer(2)]],
            constant uint& fullCols [[buffer(3)]],
            constant uint& colOffset[[buffer(4)]],
            constant uint& sliceCols[[buffer(5)]],
            uint tid [[thread_position_in_grid]]
        ) {
            uint total = rows * sliceCols;
            if (tid >= total) return;
            uint r = tid / sliceCols;
            uint c = tid % sliceCols;
            dst[r * fullCols + colOffset + c] = src[r * sliceCols + c];
        }
        """
    }
}
