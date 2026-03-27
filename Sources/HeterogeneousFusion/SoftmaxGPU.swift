// SoftmaxGPU.swift
// HeterogeneousFusion
//
// Metal compute kernel for row-wise softmax with optional scaling.
// Used in attention: softmax(Q @ K^T / sqrt(d_k))

import Metal

/// GPU softmax using a Metal compute kernel.
/// Computes row-wise softmax in-place on a [rows, cols] matrix.
public final class SoftmaxGPU: @unchecked Sendable {

    let queue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        guard let queue = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create softmax command queue")
        }
        self.queue = queue

        let source = SoftmaxGPU.generateKernel()
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "softmax_rows") else {
            throw HeterogeneousError.metalSetupFailed("Failed to find softmax_rows kernel")
        }
        self.pipelineState = try device.makeComputePipelineState(function: function)
    }

    /// Execute softmax in-place, optionally scaling by `scale` first.
    /// Returns wall-clock time in seconds.
    @discardableResult
    public func execute(
        buffer: MTLBuffer, offset: Int = 0,
        rows: Int, cols: Int, scale: Float = 1.0
    ) -> Double {
        guard let cb = queue.makeCommandBuffer() else { return 0 }
        encode(buffer: buffer, offset: offset, rows: rows, cols: cols, scale: scale, commandBuffer: cb)
        let start = CACurrentMediaTime()
        cb.commit()
        cb.waitUntilCompleted()
        return CACurrentMediaTime() - start
    }

    /// Encode softmax into an existing command buffer.
    public func encode(
        buffer: MTLBuffer, offset: Int = 0,
        rows: Int, cols: Int, scale: Float = 1.0,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(buffer, offset: offset, index: 0)
        var r = UInt32(rows)
        var c = UInt32(cols)
        var s = scale
        encoder.setBytes(&r, length: 4, index: 1)
        encoder.setBytes(&c, length: 4, index: 2)
        encoder.setBytes(&s, length: 4, index: 3)
        // One thread per row
        let threadsPerGroup = min(256, rows)
        let groups = (rows + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    /// Generate Metal kernel source for row-wise softmax.
    /// Each thread processes one row: find max, subtract, exp, sum, normalize.
    private static func generateKernel() -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void softmax_rows(
            device float* data [[buffer(0)]],
            constant uint& rows [[buffer(1)]],
            constant uint& cols [[buffer(2)]],
            constant float& scale [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= rows) return;

            device float* row = data + tid * cols;

            // Scale and find max
            float max_val = -INFINITY;
            for (uint j = 0; j < cols; j++) {
                row[j] *= scale;
                max_val = max(max_val, row[j]);
            }

            // Subtract max and exponentiate, accumulate sum
            float sum = 0.0f;
            for (uint j = 0; j < cols; j++) {
                float e = exp(row[j] - max_val);
                row[j] = e;
                sum += e;
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            for (uint j = 0; j < cols; j++) {
                row[j] *= inv_sum;
            }
        }
        """
    }
}

import QuartzCore
