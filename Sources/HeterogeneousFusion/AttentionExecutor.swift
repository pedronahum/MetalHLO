// AttentionExecutor.swift
// HeterogeneousFusion
//
// Heterogeneous attention: splits along query sequence dimension.
// Each unit independently computes:
//   scores = Q_slice @ K^T → [slice_rows, seq_len]
//   weights = softmax(scores / sqrt(d_k)) → [slice_rows, seq_len]
//   output = weights @ V → [slice_rows, d_v]
//
// K and V are shared across units (like B in matmul).
// No cross-unit communication needed — each query position is independent.

import Metal
import MetalPerformanceShaders
import QuartzCore

/// Shape descriptor for single-head attention.
public struct AttentionShape: Sendable, CustomStringConvertible {
    /// Sequence length (number of queries = number of keys).
    public let seqLen: Int
    /// Key/query head dimension.
    public let dK: Int
    /// Value head dimension (often == dK).
    public let dV: Int

    public init(seqLen: Int, dK: Int, dV: Int = 0) {
        self.seqLen = seqLen
        self.dK = dK
        self.dV = dV == 0 ? dK : dV
    }

    public var description: String {
        "attn(\(seqLen), d=\(dK))"
    }

    /// Scale factor: 1/sqrt(d_k).
    public var scale: Float {
        1.0 / sqrt(Float(dK))
    }

    /// Total FLOPs for attention (2 matmuls + softmax).
    /// QK^T: 2*seq*dK*seq, AV: 2*seq*seq*dV, softmax: ~5*seq*seq
    public var flops: Double {
        let qk = 2.0 * Double(seqLen) * Double(dK) * Double(seqLen)
        let av = 2.0 * Double(seqLen) * Double(seqLen) * Double(dV)
        let sm = 5.0 * Double(seqLen) * Double(seqLen) // exp + sum + div per element
        return qk + av + sm
    }
}

/// Result of attention benchmark.
public struct AttentionBenchmarkResult: Sendable {
    public let shape: AttentionShape
    public let gpuOnlyMs: Double
    public let mpsOnlyMs: Double
    public let cpuOnlyMs: Double
    public let fusedMs: Double
    public let fusedFractions: (gpu: Double, mps: Double, cpu: Double)
    public let bestSingleMs: Double
    public let bestSingleUnit: ComputeUnit
    public let speedup: Double
}

/// Executes single-head attention across GPU, MPS, and CPU.
public final class AttentionExecutor: @unchecked Sendable {

    private let device: MTLDevice
    private let gpuMatmul: MatmulGPU
    private let gpuSoftmax: SoftmaxGPU
    private let mpsQueue: MTLCommandQueue
    private let cpuMatmul: MatmulCPU
    private let cpuSoftmax: SoftmaxCPU

    public init(device: MTLDevice) throws {
        self.device = device
        self.gpuMatmul = try MatmulGPU(device: device)
        self.gpuSoftmax = try SoftmaxGPU(device: device)
        guard let q = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create MPS queue for attention")
        }
        self.mpsQueue = q
        self.cpuMatmul = MatmulCPU()
        self.cpuSoftmax = SoftmaxCPU()
    }

    // MARK: - Single-Unit Execution

    /// Execute full attention on GPU only. Returns wall-clock time in ms.
    public func executeGPU(
        Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer,
        scores: MTLBuffer, output: MTLBuffer,
        shape: AttentionShape
    ) -> Double {
        let seq = shape.seqLen, dK = shape.dK, dV = shape.dV

        // QK^T: [seq, dK] @ [dK, seq] → [seq, seq]
        // Note: K^T means we treat K as [dK, seq] (transposed)
        guard let cb1 = gpuMatmul.queue.makeCommandBuffer() else { return 0 }
        gpuMatmul.encode(A: Q, B: K, C: scores, sliceM: seq, K: dK, N: seq, commandBuffer: cb1)
        let start = CACurrentMediaTime()
        cb1.commit()
        cb1.waitUntilCompleted()

        // Softmax with scale
        guard let cb2 = gpuSoftmax.queue.makeCommandBuffer() else { return 0 }
        gpuSoftmax.encode(buffer: scores, rows: seq, cols: seq, scale: shape.scale, commandBuffer: cb2)
        cb2.commit()
        cb2.waitUntilCompleted()

        // weights @ V: [seq, seq] @ [seq, dV] → [seq, dV]
        guard let cb3 = gpuMatmul.queue.makeCommandBuffer() else { return 0 }
        gpuMatmul.encode(A: scores, B: V, C: output, sliceM: seq, K: seq, N: dV, commandBuffer: cb3)
        cb3.commit()
        cb3.waitUntilCompleted()

        return (CACurrentMediaTime() - start) * 1000.0
    }

    /// Execute full attention on MPS only. Returns wall-clock time in ms.
    public func executeMPS(
        Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer,
        scores: MTLBuffer, output: MTLBuffer,
        shape: AttentionShape
    ) -> Double {
        let seq = shape.seqLen, dK = shape.dK, dV = shape.dV
        let start = CACurrentMediaTime()

        // QK^T
        guard let cb1 = mpsQueue.makeCommandBuffer() else { return 0 }
        encodeMPS(A: Q, B: K, C: scores, aOffset: 0, cOffset: 0,
                  M: seq, K: dK, N: seq, commandBuffer: cb1)
        cb1.commit()
        cb1.waitUntilCompleted()

        // Softmax (use GPU kernel — MPS doesn't have standalone matrix softmax)
        guard let cb2 = gpuSoftmax.queue.makeCommandBuffer() else { return 0 }
        gpuSoftmax.encode(buffer: scores, rows: seq, cols: seq, scale: shape.scale, commandBuffer: cb2)
        cb2.commit()
        cb2.waitUntilCompleted()

        // weights @ V
        guard let cb3 = mpsQueue.makeCommandBuffer() else { return 0 }
        encodeMPS(A: scores, B: V, C: output, aOffset: 0, cOffset: 0,
                  M: seq, K: seq, N: dV, commandBuffer: cb3)
        cb3.commit()
        cb3.waitUntilCompleted()

        return (CACurrentMediaTime() - start) * 1000.0
    }

    /// Execute full attention on CPU only. Returns wall-clock time in ms.
    public func executeCPU(
        Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer,
        scores: MTLBuffer, output: MTLBuffer,
        shape: AttentionShape
    ) -> Double {
        let seq = shape.seqLen, dK = shape.dK, dV = shape.dV
        let start = CACurrentMediaTime()

        let qPtr = Q.contents().assumingMemoryBound(to: Float.self)
        let kPtr = K.contents().assumingMemoryBound(to: Float.self)
        let vPtr = V.contents().assumingMemoryBound(to: Float.self)
        let sPtr = scores.contents().assumingMemoryBound(to: Float.self)
        let oPtr = output.contents().assumingMemoryBound(to: Float.self)

        // QK^T
        _ = cpuMatmul.execute(A: qPtr, B: kPtr, C: sPtr, sliceM: seq, K: dK, N: seq)

        // Softmax with scale
        cpuSoftmax.execute(data: sPtr, rows: seq, cols: seq, scale: shape.scale)

        // weights @ V
        _ = cpuMatmul.execute(A: sPtr, B: vPtr, C: oPtr, sliceM: seq, K: seq, N: dV)

        return (CACurrentMediaTime() - start) * 1000.0
    }

    // MARK: - Heterogeneous Fused Execution

    /// Execute attention with heterogeneous partition across GPU + MPS + CPU.
    /// Splits Q along the sequence dimension; K and V are shared.
    ///
    /// Each unit pipelines the full attention (QK^T → softmax → AV) in a single
    /// command buffer, so there's only 1 sync point instead of 3.
    public func executeFused(
        Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer,
        scores: MTLBuffer, output: MTLBuffer,
        shape: AttentionShape,
        gpuFraction: Double, mpsFraction: Double
    ) -> Double {
        let seq = shape.seqLen, dK = shape.dK, dV = shape.dV

        // Compute row assignments
        let gpuRows = max(1, Int(Double(seq) * gpuFraction))
        let mpsRows = max(1, Int(Double(seq) * mpsFraction))
        let cpuRows = seq - gpuRows - mpsRows
        guard cpuRows >= 0 else { return 0 }

        let floatSize = MemoryLayout<Float>.size

        // Byte offsets into Q (partitioned), scores, and output
        let gpuQOff = 0
        let gpuSOff = 0
        let gpuOOff = 0

        let mpsQOff = gpuRows * dK * floatSize
        let mpsSOff = gpuRows * seq * floatSize
        let mpsOOff = gpuRows * dV * floatSize

        let cpuQOff = (gpuRows + mpsRows) * dK * floatSize
        let cpuSOff = (gpuRows + mpsRows) * seq * floatSize
        let cpuOOff = (gpuRows + mpsRows) * dV * floatSize

        // ── GPU: full pipeline in one command buffer ──
        let gpuCB = gpuMatmul.queue.makeCommandBuffer()!

        // QK^T
        gpuMatmul.encode(
            A: Q, B: K, C: scores,
            aOffset: gpuQOff, cOffset: gpuSOff,
            sliceM: gpuRows, K: dK, N: seq,
            commandBuffer: gpuCB
        )
        // Softmax (encode into same command buffer — GPU ensures ordering)
        gpuSoftmax.encode(
            buffer: scores, offset: gpuSOff,
            rows: gpuRows, cols: seq, scale: shape.scale,
            commandBuffer: gpuCB
        )
        // weights @ V
        gpuMatmul.encode(
            A: scores, B: V, C: output,
            aOffset: gpuSOff, cOffset: gpuOOff,
            sliceM: gpuRows, K: seq, N: dV,
            commandBuffer: gpuCB
        )

        // ── MPS: full pipeline in one command buffer ──
        let mpsCB = mpsQueue.makeCommandBuffer()!

        // QK^T
        encodeMPS(
            A: Q, B: K, C: scores,
            aOffset: mpsQOff, cOffset: mpsSOff,
            M: mpsRows, K: dK, N: seq,
            commandBuffer: mpsCB
        )
        // Softmax (GPU kernel in MPS command buffer — Metal handles scheduling)
        gpuSoftmax.encode(
            buffer: scores, offset: mpsSOff,
            rows: mpsRows, cols: seq, scale: shape.scale,
            commandBuffer: mpsCB
        )
        // weights @ V
        encodeMPS(
            A: scores, B: V, C: output,
            aOffset: mpsSOff, cOffset: mpsOOff,
            M: mpsRows, K: seq, N: dV,
            commandBuffer: mpsCB
        )

        // ── Commit all before waiting (single sync point) ──
        let start = CACurrentMediaTime()
        gpuCB.commit()
        mpsCB.commit()

        // CPU: full pipeline runs concurrently with GPU/MPS
        if cpuRows > 0 {
            let qPtr = (Q.contents() + cpuQOff).assumingMemoryBound(to: Float.self)
            let kPtr = K.contents().assumingMemoryBound(to: Float.self)
            let vPtr = V.contents().assumingMemoryBound(to: Float.self)
            let sPtr = (scores.contents() + cpuSOff).assumingMemoryBound(to: Float.self)
            let oPtr = (output.contents() + cpuOOff).assumingMemoryBound(to: Float.self)

            // QK^T
            _ = cpuMatmul.execute(A: qPtr, B: kPtr, C: sPtr, sliceM: cpuRows, K: dK, N: seq)
            // Softmax
            cpuSoftmax.execute(data: sPtr, rows: cpuRows, cols: seq, scale: shape.scale)
            // weights @ V
            _ = cpuMatmul.execute(A: sPtr, B: vPtr, C: oPtr, sliceM: cpuRows, K: seq, N: dV)
        }

        gpuCB.waitUntilCompleted()
        mpsCB.waitUntilCompleted()

        return (CACurrentMediaTime() - start) * 1000.0
    }

    // MARK: - Benchmark

    /// Run attention benchmark for a given shape.
    public func benchmark(
        shape: AttentionShape,
        config: HeterogeneousBenchmarkConfig = .quick
    ) throws -> AttentionBenchmarkResult {
        let seq = shape.seqLen, dK = shape.dK, dV = shape.dV

        // Allocate buffers
        let qBytes = seq * dK * MemoryLayout<Float>.size
        let kBytes = dK * seq * MemoryLayout<Float>.size  // K^T: [dK, seq]
        let vBytes = seq * dV * MemoryLayout<Float>.size
        let sBytes = seq * seq * MemoryLayout<Float>.size  // scores: [seq, seq]
        let oBytes = seq * dV * MemoryLayout<Float>.size

        guard let bufQ = device.makeBuffer(length: qBytes, options: .storageModeShared),
              let bufK = device.makeBuffer(length: kBytes, options: .storageModeShared),
              let bufV = device.makeBuffer(length: vBytes, options: .storageModeShared),
              let bufS = device.makeBuffer(length: sBytes, options: .storageModeShared),
              let bufO = device.makeBuffer(length: oBytes, options: .storageModeShared) else {
            throw HeterogeneousError.bufferAllocationFailed("Attention buffer allocation failed for \(shape)")
        }

        fillRandom(bufQ, count: seq * dK)
        fillRandom(bufK, count: dK * seq)
        fillRandom(bufV, count: seq * dV)

        // Warmup + measure function
        func measure(_ body: () -> Double) -> Double {
            for _ in 0..<config.warmupIterations { _ = body() }
            var times: [Double] = []
            for _ in 0..<config.iterations { times.append(body()) }
            return TimingSummary(samples: times, trimCount: config.trimCount).mean
        }

        // Single-unit baselines
        let gpuMs = measure {
            executeGPU(Q: bufQ, K: bufK, V: bufV, scores: bufS, output: bufO, shape: shape)
        }
        let mpsMs = measure {
            executeMPS(Q: bufQ, K: bufK, V: bufV, scores: bufS, output: bufO, shape: shape)
        }
        let cpuMs = measure {
            executeCPU(Q: bufQ, K: bufK, V: bufV, scores: bufS, output: bufO, shape: shape)
        }

        // Compute optimal fractions (1/time ratios)
        let gpuRate = 1.0 / gpuMs
        let mpsRate = 1.0 / mpsMs
        let cpuRate = 1.0 / cpuMs
        let totalRate = gpuRate + mpsRate + cpuRate
        let optGpu = gpuRate / totalRate
        let optMps = mpsRate / totalRate

        // Fused with optimal split
        let fusedMs = measure {
            executeFused(
                Q: bufQ, K: bufK, V: bufV, scores: bufS, output: bufO,
                shape: shape, gpuFraction: optGpu, mpsFraction: optMps
            )
        }

        let bestMs = Swift.min(gpuMs, Swift.min(mpsMs, cpuMs))
        let bestUnit: ComputeUnit = gpuMs <= mpsMs && gpuMs <= cpuMs ? .gpu
            : mpsMs <= gpuMs && mpsMs <= cpuMs ? .mps : .cpu

        return AttentionBenchmarkResult(
            shape: shape,
            gpuOnlyMs: gpuMs,
            mpsOnlyMs: mpsMs,
            cpuOnlyMs: cpuMs,
            fusedMs: fusedMs,
            fusedFractions: (gpu: optGpu, mps: optMps, cpu: 1.0 - optGpu - optMps),
            bestSingleMs: bestMs,
            bestSingleUnit: bestUnit,
            speedup: bestMs / fusedMs
        )
    }

    // MARK: - MPS Encoding

    private func encodeMPS(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        aOffset: Int, cOffset: Int,
        M: Int, K: Int, N: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        let rowBytesA = K * MemoryLayout<Float>.size
        let rowBytesB = N * MemoryLayout<Float>.size
        let rowBytesC = N * MemoryLayout<Float>.size

        let matA = MPSMatrix(
            buffer: A, offset: aOffset,
            descriptor: MPSMatrixDescriptor(
                rows: M, columns: K, rowBytes: rowBytesA, dataType: .float32
            )
        )
        let matB = MPSMatrix(
            buffer: B,
            descriptor: MPSMatrixDescriptor(
                rows: K, columns: N, rowBytes: rowBytesB, dataType: .float32
            )
        )
        let matC = MPSMatrix(
            buffer: C, offset: cOffset,
            descriptor: MPSMatrixDescriptor(
                rows: M, columns: N, rowBytes: rowBytesC, dataType: .float32
            )
        )

        let mul = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false, transposeRight: false,
            resultRows: M, resultColumns: N, interiorColumns: K,
            alpha: 1.0, beta: 0.0
        )
        mul.encode(commandBuffer: commandBuffer, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
    }

    // MARK: - Helpers

    private func fillRandom(_ buffer: MTLBuffer, count: Int) {
        let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            ptr[i] = Float.random(in: -0.5..<0.5)
        }
    }
}
