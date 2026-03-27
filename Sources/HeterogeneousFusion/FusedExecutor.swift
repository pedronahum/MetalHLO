// FusedExecutor.swift
// HeterogeneousFusion
//
// Phase 2: Generic executor that consumes a PartitionDescriptor and
// dispatches work to GPU, MPS, and CPU simultaneously.
//
// The executor has NO knowledge of how the partition was decided.
// It only knows how to:
//   1. Encode each assignment to its unit's command buffer
//   2. Commit all command buffers before waiting on any
//   3. Wait for all completions
//   4. Return profiling data

import Metal
import MetalPerformanceShaders
import QuartzCore

/// Executes a partitioned operation across multiple compute units.
///
/// Accepts a `PartitionDescriptor` and shared buffers, dispatches work
/// to each unit in parallel, and returns a `FusedProfile`.
///
/// The executor is op-agnostic for the dispatch protocol (commit-before-wait),
/// but encodes op-specific kernels for each unit via registered encoders.
public final class FusedExecutor: @unchecked Sendable {

    private let device: MTLDevice
    private let gpuKernel: MatmulGPU
    private let mpsQueue: MTLCommandQueue
    private let cpuKernel: MatmulCPU
    private let columnHelper: ColumnSplitHelper
    private let elementwiseGPU: ElementwiseGPU
    private let elementwiseCPU: ElementwiseCPU
    private let layerNormGPU: LayerNormGPU
    private let layerNormCPU: LayerNormCPU
    private let embeddingGPU: EmbeddingGPU
    private let embeddingCPU: EmbeddingCPU
    private let convGPU: ConvolutionGPU
    private let convMPS: ConvolutionMPS
    private let convCPU: ConvolutionCPU

    public init(device: MTLDevice) throws {
        self.device = device
        self.gpuKernel = try MatmulGPU(device: device)
        guard let queue = device.makeCommandQueue() else {
            throw HeterogeneousError.metalSetupFailed("Failed to create MPS command queue")
        }
        self.mpsQueue = queue
        self.cpuKernel = MatmulCPU()
        self.columnHelper = try ColumnSplitHelper(device: device)
        self.elementwiseGPU = try ElementwiseGPU(device: device)
        self.elementwiseCPU = ElementwiseCPU()
        self.layerNormGPU = try LayerNormGPU(device: device)
        self.layerNormCPU = LayerNormCPU()
        self.embeddingGPU = try EmbeddingGPU(device: device)
        self.embeddingCPU = EmbeddingCPU()
        self.convGPU = try ConvolutionGPU(device: device)
        self.convMPS = try ConvolutionMPS(device: device)
        self.convCPU = ConvolutionCPU()
    }

    /// Execute a partitioned operation.
    ///
    /// - Parameters:
    ///   - descriptor: Describes how the op is split across units.
    ///   - inputBuffers: Shared input buffers (A, B for matmul). Slices reference into these.
    ///   - outputBuffer: Shared output buffer (C). Slices write into non-overlapping regions.
    /// - Returns: FusedProfile with per-unit timing breakdown.
    public func execute(
        descriptor: PartitionDescriptor,
        inputBuffers: [MTLBuffer],
        outputBuffer: MTLBuffer
    ) -> FusedProfile {
        switch descriptor.op {
        case .matmul:
            if descriptor.isColumnSplit {
                return executeMatmulColumnSplit(
                    descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outputBuffer)
            }
            return executeMatmul(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outputBuffer)
        case .elementwiseAdd, .elementwiseMul, .elementwiseRelu, .gelu:
            return executeElementwise(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outputBuffer)
        case .layerNorm:
            return executeLayerNorm(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outputBuffer)
        case .embeddingLookup:
            return executeEmbedding(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outputBuffer)
        case .convolution:
            return executeConvolution(descriptor: descriptor, inputBuffers: inputBuffers, outputBuffer: outputBuffer)
        default:
            return FusedProfile(
                totalWallClockMs: 0, perUnit: [], splitRatio: [:],
                recombineMs: 0, speedupVsBestSingle: 0
            )
        }
    }

    // MARK: - Matmul Execution

    private func executeMatmul(
        descriptor: PartitionDescriptor,
        inputBuffers: [MTLBuffer],
        outputBuffer: MTLBuffer
    ) -> FusedProfile {
        guard inputBuffers.count >= 2 else {
            return emptyProfile()
        }

        let bufA = inputBuffers[0]
        let bufB = inputBuffers[1]
        let bufC = outputBuffer

        // Separate assignments by unit type
        let gpuAssignments = descriptor.assignments.filter { $0.unit == .gpu }
        let mpsAssignments = descriptor.assignments.filter { $0.unit == .mps }
        let cpuAssignments = descriptor.assignments.filter { $0.unit == .cpu }

        // Prepare GPU command buffers
        var gpuCommandBuffers: [(MTLCommandBuffer, UnitAssignment)] = []
        for assignment in gpuAssignments {
            guard let cb = gpuKernel.queue.makeCommandBuffer() else { continue }
            let aSlice = assignment.inputSlices[0]
            let bSlice = assignment.inputSlices[1]
            let cSlice = assignment.outputSlice
            let sliceM = aSlice.shape[0]
            let K = bSlice.shape[0]
            let N = bSlice.shape[1]

            gpuKernel.encode(
                A: bufA, B: bufB, C: bufC,
                aOffset: aSlice.bufferOffset,
                cOffset: cSlice.bufferOffset,
                sliceM: sliceM, K: K, N: N,
                commandBuffer: cb
            )
            gpuCommandBuffers.append((cb, assignment))
        }

        // Prepare MPS command buffers
        var mpsCommandBuffers: [(MTLCommandBuffer, UnitAssignment)] = []
        for assignment in mpsAssignments {
            guard let cb = mpsQueue.makeCommandBuffer() else { continue }
            let aSlice = assignment.inputSlices[0]
            let bSlice = assignment.inputSlices[1]
            let cSlice = assignment.outputSlice
            let sliceM = aSlice.shape[0]
            let K = bSlice.shape[0]
            let N = bSlice.shape[1]

            encodeMPS(
                A: bufA, B: bufB, C: bufC,
                aOffset: aSlice.bufferOffset,
                cOffset: cSlice.bufferOffset,
                sliceM: sliceM, K: K, N: N,
                commandBuffer: cb
            )
            mpsCommandBuffers.append((cb, assignment))
        }

        // ── Commit all GPU and MPS command buffers BEFORE waiting on any ──
        let start = CACurrentMediaTime()

        for (cb, _) in gpuCommandBuffers { cb.commit() }
        for (cb, _) in mpsCommandBuffers { cb.commit() }

        // Run CPU slices concurrently (CPU work happens while GPU/MPS execute)
        var cpuProfiles: [UnitProfile] = []
        for assignment in cpuAssignments {
            let aSlice = assignment.inputSlices[0]
            let bSlice = assignment.inputSlices[1]
            let cSlice = assignment.outputSlice
            let sliceM = aSlice.shape[0]
            let K = bSlice.shape[0]
            let N = bSlice.shape[1]

            let cpuStart = CACurrentMediaTime()
            let aPtr = (bufA.contents() + aSlice.bufferOffset).assumingMemoryBound(to: Float.self)
            let bPtr = bufB.contents().assumingMemoryBound(to: Float.self)
            let cPtr = (bufC.contents() + cSlice.bufferOffset).assumingMemoryBound(to: Float.self)
            _ = cpuKernel.execute(A: aPtr, B: bPtr, C: cPtr, sliceM: sliceM, K: K, N: N)
            let cpuMs = (CACurrentMediaTime() - cpuStart) * 1000.0

            cpuProfiles.append(UnitProfile(
                unit: .cpu,
                wallClockMs: cpuMs,
                bandwidthGBs: bandwidth(rows: sliceM, K: K, N: N, timeS: cpuMs / 1000.0),
                tflops: tflops(rows: sliceM, K: K, N: N, timeS: cpuMs / 1000.0),
                syncOverheadMs: 0
            ))
        }

        // Wait for GPU completions
        var gpuProfiles: [UnitProfile] = []
        for (cb, assignment) in gpuCommandBuffers {
            cb.waitUntilCompleted()
            let ms = (CACurrentMediaTime() - start) * 1000.0
            let sliceM = assignment.inputSlices[0].shape[0]
            let K = assignment.inputSlices[1].shape[0]
            let N = assignment.inputSlices[1].shape[1]
            gpuProfiles.append(UnitProfile(
                unit: .gpu, wallClockMs: ms,
                bandwidthGBs: bandwidth(rows: sliceM, K: K, N: N, timeS: ms / 1000.0),
                tflops: tflops(rows: sliceM, K: K, N: N, timeS: ms / 1000.0),
                syncOverheadMs: 0
            ))
        }

        // Wait for MPS completions
        var mpsProfiles: [UnitProfile] = []
        for (cb, assignment) in mpsCommandBuffers {
            cb.waitUntilCompleted()
            let ms = (CACurrentMediaTime() - start) * 1000.0
            let sliceM = assignment.inputSlices[0].shape[0]
            let K = assignment.inputSlices[1].shape[0]
            let N = assignment.inputSlices[1].shape[1]
            mpsProfiles.append(UnitProfile(
                unit: .mps, wallClockMs: ms,
                bandwidthGBs: bandwidth(rows: sliceM, K: K, N: N, timeS: ms / 1000.0),
                tflops: tflops(rows: sliceM, K: K, N: N, timeS: ms / 1000.0),
                syncOverheadMs: 0
            ))
        }

        let totalMs = (CACurrentMediaTime() - start) * 1000.0

        // No recombine needed — slices wrote directly into contiguous output buffer
        let allProfiles = gpuProfiles + mpsProfiles + cpuProfiles
        var splitRatio: [ComputeUnit: Double] = [:]
        for a in descriptor.assignments {
            splitRatio[a.unit, default: 0] += a.workFraction
        }

        return FusedProfile(
            totalWallClockMs: totalMs,
            perUnit: allProfiles,
            splitRatio: splitRatio,
            recombineMs: 0,
            speedupVsBestSingle: 0
        )
    }

    // MARK: - Column-Split Matmul Execution

    /// Execute matmul with column-split partitioning (splitDimension == 1).
    ///
    /// Each unit computes C[:, colSlice] = A @ B[:, colSlice].
    /// - MPS: strided MPSMatrix descriptors (zero-copy read/write)
    /// - GPU: extract B columns → contiguous matmul → scatter C columns
    /// - CPU: extract B columns → vDSP_mmul → scatter C columns
    private func executeMatmulColumnSplit(
        descriptor: PartitionDescriptor,
        inputBuffers: [MTLBuffer],
        outputBuffer: MTLBuffer
    ) -> FusedProfile {
        guard inputBuffers.count >= 2 else { return emptyProfile() }

        let bufA = inputBuffers[0]
        let bufB = inputBuffers[1]
        let bufC = outputBuffer

        let M = descriptor.fullInputShape[0]
        let K = descriptor.fullInputShape[1]
        let N = descriptor.fullOutputShape[1]

        let gpuAssignments = descriptor.assignments.filter { $0.unit == .gpu }
        let mpsAssignments = descriptor.assignments.filter { $0.unit == .mps }
        let cpuAssignments = descriptor.assignments.filter { $0.unit == .cpu }

        // Allocate temp buffers for GPU/CPU column extraction
        struct TempBuffers {
            let bTemp: MTLBuffer
            let cTemp: MTLBuffer
            let sliceN: Int
            let colOffset: Int
        }

        var gpuTemps: [TempBuffers] = []
        var cpuTemps: [(bTemp: UnsafeMutablePointer<Float>, cTemp: UnsafeMutablePointer<Float>,
                         sliceN: Int, colOffset: Int)] = []

        // Prepare GPU temp buffers and extract B columns
        var gpuCommandBuffers: [(MTLCommandBuffer, UnitAssignment, TempBuffers)] = []
        for assignment in gpuAssignments {
            let bSlice = assignment.inputSlices[1]
            let sliceN = bSlice.shape[1]
            let colOffset = bSlice.bufferOffset / bSlice.dtype.byteSize

            guard let bTemp = columnHelper.makeTempBuffer(rows: K, cols: sliceN),
                  let cTemp = columnHelper.makeTempBuffer(rows: M, cols: sliceN) else { continue }

            let temps = TempBuffers(bTemp: bTemp, cTemp: cTemp, sliceN: sliceN, colOffset: colOffset)
            gpuTemps.append(temps)

            // Extract B columns on GPU, then matmul into cTemp
            guard let cb = gpuKernel.queue.makeCommandBuffer() else { continue }
            columnHelper.extractColumns(
                src: bufB, dst: bTemp,
                rows: K, fullCols: N, colOffset: colOffset, sliceCols: sliceN,
                commandBuffer: cb
            )
            gpuKernel.encode(
                A: bufA, B: bTemp, C: cTemp,
                sliceM: M, K: K, N: sliceN,
                commandBuffer: cb
            )
            gpuCommandBuffers.append((cb, assignment, temps))
        }

        // Prepare MPS command buffers — zero-copy strided access
        var mpsCommandBuffers: [(MTLCommandBuffer, UnitAssignment)] = []
        for assignment in mpsAssignments {
            let bSlice = assignment.inputSlices[1]
            let sliceN = bSlice.shape[1]
            let colOffset = bSlice.bufferOffset / bSlice.dtype.byteSize

            guard let cb = mpsQueue.makeCommandBuffer() else { continue }
            encodeMPSColumnSplit(
                A: bufA, B: bufB, C: bufC,
                M: M, K: K, N: N,
                colOffset: colOffset, sliceN: sliceN,
                commandBuffer: cb
            )
            mpsCommandBuffers.append((cb, assignment))
        }

        // Prepare CPU temp buffers and extract B columns on CPU
        for assignment in cpuAssignments {
            let bSlice = assignment.inputSlices[1]
            let sliceN = bSlice.shape[1]
            let colOffset = bSlice.bufferOffset / bSlice.dtype.byteSize

            let bTemp = UnsafeMutablePointer<Float>.allocate(capacity: K * sliceN)
            let cTemp = UnsafeMutablePointer<Float>.allocate(capacity: M * sliceN)

            // Extract B columns on CPU
            let bPtr = bufB.contents().assumingMemoryBound(to: Float.self)
            ColumnSplitHelper.extractColumnsCPU(
                src: bPtr, dst: bTemp,
                rows: K, fullCols: N, colOffset: colOffset, sliceCols: sliceN
            )
            cpuTemps.append((bTemp: bTemp, cTemp: cTemp, sliceN: sliceN, colOffset: colOffset))
        }

        // ── Commit all GPU and MPS command buffers BEFORE waiting on any ──
        let start = CACurrentMediaTime()

        for (cb, _, _) in gpuCommandBuffers { cb.commit() }
        for (cb, _) in mpsCommandBuffers { cb.commit() }

        // Run CPU slices concurrently
        var cpuProfiles: [UnitProfile] = []
        for (idx, _) in cpuAssignments.enumerated() {
            let temps = cpuTemps[idx]
            let aPtr = bufA.contents().assumingMemoryBound(to: Float.self)

            let cpuStart = CACurrentMediaTime()
            _ = cpuKernel.execute(A: aPtr, B: temps.bTemp, C: temps.cTemp,
                                  sliceM: M, K: K, N: temps.sliceN)
            let cpuMs = (CACurrentMediaTime() - cpuStart) * 1000.0

            cpuProfiles.append(UnitProfile(
                unit: .cpu, wallClockMs: cpuMs,
                bandwidthGBs: bandwidth(rows: M, K: K, N: temps.sliceN, timeS: cpuMs / 1000.0),
                tflops: tflops(rows: M, K: K, N: temps.sliceN, timeS: cpuMs / 1000.0),
                syncOverheadMs: 0
            ))
        }

        // Wait for GPU, then scatter C columns back
        var gpuProfiles: [UnitProfile] = []
        for (cb, _, temps) in gpuCommandBuffers {
            cb.waitUntilCompleted()

            // Scatter cTemp columns back into bufC
            let scatterCB = gpuKernel.queue.makeCommandBuffer()!
            columnHelper.scatterColumns(
                src: temps.cTemp, dst: bufC,
                rows: M, fullCols: N, colOffset: temps.colOffset, sliceCols: temps.sliceN,
                commandBuffer: scatterCB
            )
            scatterCB.commit()
            scatterCB.waitUntilCompleted()

            let ms = (CACurrentMediaTime() - start) * 1000.0
            gpuProfiles.append(UnitProfile(
                unit: .gpu, wallClockMs: ms,
                bandwidthGBs: bandwidth(rows: M, K: K, N: temps.sliceN, timeS: ms / 1000.0),
                tflops: tflops(rows: M, K: K, N: temps.sliceN, timeS: ms / 1000.0),
                syncOverheadMs: 0
            ))
        }

        // Wait for MPS (MPS writes directly into strided C — no scatter needed)
        var mpsProfiles: [UnitProfile] = []
        for (cb, assignment) in mpsCommandBuffers {
            cb.waitUntilCompleted()
            let ms = (CACurrentMediaTime() - start) * 1000.0
            let sliceN = assignment.inputSlices[1].shape[1]
            mpsProfiles.append(UnitProfile(
                unit: .mps, wallClockMs: ms,
                bandwidthGBs: bandwidth(rows: M, K: K, N: sliceN, timeS: ms / 1000.0),
                tflops: tflops(rows: M, K: K, N: sliceN, timeS: ms / 1000.0),
                syncOverheadMs: 0
            ))
        }

        // Scatter CPU results back
        let scatterStart = CACurrentMediaTime()
        let cPtr = bufC.contents().assumingMemoryBound(to: Float.self)
        for temps in cpuTemps {
            ColumnSplitHelper.scatterColumnsCPU(
                src: UnsafePointer(temps.cTemp), dst: cPtr,
                rows: M, fullCols: N, colOffset: temps.colOffset, sliceCols: temps.sliceN
            )
        }
        let scatterMs = (CACurrentMediaTime() - scatterStart) * 1000.0

        // Free CPU temp buffers
        for temps in cpuTemps {
            temps.bTemp.deallocate()
            temps.cTemp.deallocate()
        }

        let totalMs = (CACurrentMediaTime() - start) * 1000.0

        let allProfiles = gpuProfiles + mpsProfiles + cpuProfiles
        var splitRatio: [ComputeUnit: Double] = [:]
        for a in descriptor.assignments {
            splitRatio[a.unit, default: 0] += a.workFraction
        }

        return FusedProfile(
            totalWallClockMs: totalMs,
            perUnit: allProfiles,
            splitRatio: splitRatio,
            recombineMs: scatterMs,
            speedupVsBestSingle: 0
        )
    }

    // MARK: - MPS Column-Split Encoding

    /// Encode column-split matmul using strided MPSMatrix descriptors (zero-copy).
    private func encodeMPSColumnSplit(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        M: Int, K: Int, N: Int,
        colOffset: Int, sliceN: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        let elemSize = MemoryLayout<Float>.size

        // A: full matrix [M, K] — no stride tricks needed
        let matA = MPSMatrix(
            buffer: A,
            descriptor: MPSMatrixDescriptor(
                rows: M, columns: K,
                rowBytes: K * elemSize, dataType: .float32
            )
        )

        // B: strided column slice [K, sliceN] from [K, N]
        // offset = colOffset * elemSize, rowBytes = N * elemSize (full row stride)
        let matB = MPSMatrix(
            buffer: B, offset: colOffset * elemSize,
            descriptor: MPSMatrixDescriptor(
                rows: K, columns: sliceN,
                rowBytes: N * elemSize, dataType: .float32
            )
        )

        // C: strided column slice [M, sliceN] from [M, N]
        let matC = MPSMatrix(
            buffer: C, offset: colOffset * elemSize,
            descriptor: MPSMatrixDescriptor(
                rows: M, columns: sliceN,
                rowBytes: N * elemSize, dataType: .float32
            )
        )

        let mul = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: M,
            resultColumns: sliceN,
            interiorColumns: K,
            alpha: 1.0,
            beta: 0.0
        )

        mul.encode(commandBuffer: commandBuffer, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
    }

    // MARK: - MPS Encoding

    private func encodeMPS(
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        aOffset: Int, cOffset: Int,
        sliceM: Int, K: Int, N: Int,
        commandBuffer: MTLCommandBuffer
    ) {
        let rowBytesA = K * MemoryLayout<Float>.size
        let rowBytesB = N * MemoryLayout<Float>.size
        let rowBytesC = N * MemoryLayout<Float>.size

        let matA = MPSMatrix(
            buffer: A, offset: aOffset,
            descriptor: MPSMatrixDescriptor(
                rows: sliceM, columns: K,
                rowBytes: rowBytesA, dataType: .float32
            )
        )
        let matB = MPSMatrix(
            buffer: B,
            descriptor: MPSMatrixDescriptor(
                rows: K, columns: N,
                rowBytes: rowBytesB, dataType: .float32
            )
        )
        let matC = MPSMatrix(
            buffer: C, offset: cOffset,
            descriptor: MPSMatrixDescriptor(
                rows: sliceM, columns: N,
                rowBytes: rowBytesC, dataType: .float32
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

    // MARK: - Elementwise Execution

    /// Execute an elementwise op with row-split partitioning.
    ///
    /// Elementwise ops are bandwidth-bound — MPS has no native elementwise path.
    /// Only GPU (Metal compute) and CPU (Accelerate) participate.
    /// Value is cluster absorption, not standalone speedup.
    private func executeElementwise(
        descriptor: PartitionDescriptor,
        inputBuffers: [MTLBuffer],
        outputBuffer: MTLBuffer
    ) -> FusedProfile {
        guard !inputBuffers.isEmpty else { return emptyProfile() }

        let bufA = inputBuffers[0]
        let bufB = inputBuffers.count > 1 ? inputBuffers[1] : nil
        let bufC = outputBuffer

        guard let ewOp = ElementwiseOp.from(descriptor.op) else { return emptyProfile() }

        // 2-unit: GPU + CPU only (MPS has no standalone elementwise)
        let gpuAssignments = descriptor.assignments.filter { $0.unit == .gpu }
        let cpuAssignments = descriptor.assignments.filter { $0.unit == .cpu }

        // Prepare GPU command buffers
        var gpuCommandBuffers: [(MTLCommandBuffer, UnitAssignment)] = []
        for assignment in gpuAssignments {
            guard let cb = elementwiseGPU.queue.makeCommandBuffer() else { continue }
            let aSlice = assignment.inputSlices[0]
            let count = aSlice.elementCount
            let bOffset = ewOp.isBinary ? assignment.inputSlices[1].bufferOffset : 0

            elementwiseGPU.encode(
                op: ewOp,
                A: bufA, B: bufB, C: bufC,
                aOffset: aSlice.bufferOffset,
                bOffset: bOffset,
                cOffset: assignment.outputSlice.bufferOffset,
                count: count,
                commandBuffer: cb
            )
            gpuCommandBuffers.append((cb, assignment))
        }

        // Commit all before waiting
        let start = CACurrentMediaTime()
        for (cb, _) in gpuCommandBuffers { cb.commit() }

        // CPU runs concurrently
        var cpuProfiles: [UnitProfile] = []
        for assignment in cpuAssignments {
            let aSlice = assignment.inputSlices[0]
            let count = aSlice.elementCount
            let cpuStart = CACurrentMediaTime()

            let aPtr = (bufA.contents() + aSlice.bufferOffset).assumingMemoryBound(to: Float.self)
            let bPtr: UnsafePointer<Float>?
            if ewOp.isBinary, let b = bufB {
                let bOffset = assignment.inputSlices[1].bufferOffset
                let rawBPtr = (b.contents() + bOffset).assumingMemoryBound(to: Float.self)
                bPtr = UnsafePointer(rawBPtr)
            } else {
                bPtr = nil
            }
            let cPtr = (bufC.contents() + assignment.outputSlice.bufferOffset).assumingMemoryBound(to: Float.self)

            elementwiseCPU.execute(op: ewOp, A: aPtr, B: bPtr, C: cPtr, count: count)
            let cpuMs = (CACurrentMediaTime() - cpuStart) * 1000.0

            let bytes = Double(count * 2 * (ewOp.isBinary ? 3 : 2)) * 4.0  // read A [+B] + write C
            cpuProfiles.append(UnitProfile(
                unit: .cpu, wallClockMs: cpuMs,
                bandwidthGBs: cpuMs > 0 ? bytes / (cpuMs / 1000.0) / 1e9 : 0,
                tflops: 0, syncOverheadMs: 0
            ))
        }

        // Wait for GPU
        var gpuProfiles: [UnitProfile] = []
        for (cb, _) in gpuCommandBuffers {
            cb.waitUntilCompleted()
            let ms = (CACurrentMediaTime() - start) * 1000.0
            gpuProfiles.append(UnitProfile(
                unit: .gpu, wallClockMs: ms,
                bandwidthGBs: 0, tflops: 0, syncOverheadMs: 0
            ))
        }

        let totalMs = (CACurrentMediaTime() - start) * 1000.0
        let allProfiles = gpuProfiles + cpuProfiles
        var splitRatio: [ComputeUnit: Double] = [:]
        for a in descriptor.assignments {
            splitRatio[a.unit, default: 0] += a.workFraction
        }

        return FusedProfile(
            totalWallClockMs: totalMs,
            perUnit: allProfiles,
            splitRatio: splitRatio,
            recombineMs: 0,
            speedupVsBestSingle: 0
        )
    }

    // MARK: - LayerNorm Execution

    /// Execute a LayerNorm op with row-split partitioning.
    ///
    /// Each unit processes its row slice independently — mean/variance/normalize
    /// are per-row operations with no cross-unit communication.
    /// γ and β are read-only broadcast parameters shared by all units.
    ///
    /// Input buffers: [0]=input [M,N], [1]=gamma [N], [2]=beta [N]
    /// Output buffer: [M,N]
    private func executeLayerNorm(
        descriptor: PartitionDescriptor,
        inputBuffers: [MTLBuffer],
        outputBuffer: MTLBuffer
    ) -> FusedProfile {
        guard inputBuffers.count >= 3 else { return emptyProfile() }

        let bufInput = inputBuffers[0]
        let bufGamma = inputBuffers[1]
        let bufBeta = inputBuffers[2]
        let bufOutput = outputBuffer

        let epsilon: Float = 1e-5

        // 2-unit: GPU + CPU only (MPS has no native LayerNorm path)
        let gpuAssignments = descriptor.assignments.filter { $0.unit == .gpu }
        let cpuAssignments = descriptor.assignments.filter { $0.unit == .cpu }

        // Prepare GPU command buffers
        var gpuCommandBuffers: [(MTLCommandBuffer, UnitAssignment)] = []
        for assignment in gpuAssignments {
            guard let cb = layerNormGPU.queue.makeCommandBuffer() else { continue }
            let inputSlice = assignment.inputSlices[0]
            let rowCount = inputSlice.shape[0]
            let N = inputSlice.shape[1]

            layerNormGPU.encode(
                input: bufInput, gamma: bufGamma, beta: bufBeta, output: bufOutput,
                inputOffset: inputSlice.bufferOffset,
                outputOffset: assignment.outputSlice.bufferOffset,
                rowCount: rowCount, N: N, epsilon: epsilon,
                commandBuffer: cb
            )
            gpuCommandBuffers.append((cb, assignment))
        }

        // Commit all before waiting
        let start = CACurrentMediaTime()
        for (cb, _) in gpuCommandBuffers { cb.commit() }

        // CPU runs concurrently
        var cpuProfiles: [UnitProfile] = []
        for assignment in cpuAssignments {
            let inputSlice = assignment.inputSlices[0]
            let rowCount = inputSlice.shape[0]
            let N = inputSlice.shape[1]
            let cpuStart = CACurrentMediaTime()

            let inPtr = (bufInput.contents() + inputSlice.bufferOffset).assumingMemoryBound(to: Float.self)
            let gammaPtr = bufGamma.contents().assumingMemoryBound(to: Float.self)
            let betaPtr = bufBeta.contents().assumingMemoryBound(to: Float.self)
            let outPtr = (bufOutput.contents() + assignment.outputSlice.bufferOffset).assumingMemoryBound(to: Float.self)

            layerNormCPU.execute(
                input: inPtr, gamma: gammaPtr, beta: betaPtr, output: outPtr,
                M: rowCount, N: N, epsilon: epsilon
            )
            let cpuMs = (CACurrentMediaTime() - cpuStart) * 1000.0

            // Bandwidth: read input + gamma + beta, write output = 2*M*N + 2*N floats
            let bytes = Double(rowCount * N * 2 + N * 2) * 4.0
            cpuProfiles.append(UnitProfile(
                unit: .cpu, wallClockMs: cpuMs,
                bandwidthGBs: cpuMs > 0 ? bytes / (cpuMs / 1000.0) / 1e9 : 0,
                tflops: 0, syncOverheadMs: 0
            ))
        }

        // Wait for GPU
        var gpuProfiles: [UnitProfile] = []
        for (cb, _) in gpuCommandBuffers {
            cb.waitUntilCompleted()
            let ms = (CACurrentMediaTime() - start) * 1000.0
            gpuProfiles.append(UnitProfile(
                unit: .gpu, wallClockMs: ms,
                bandwidthGBs: 0, tflops: 0, syncOverheadMs: 0
            ))
        }

        let totalMs = (CACurrentMediaTime() - start) * 1000.0
        let allProfiles = gpuProfiles + cpuProfiles
        var splitRatio: [ComputeUnit: Double] = [:]
        for a in descriptor.assignments {
            splitRatio[a.unit, default: 0] += a.workFraction
        }

        return FusedProfile(
            totalWallClockMs: totalMs,
            perUnit: allProfiles,
            splitRatio: splitRatio,
            recombineMs: 0,  // No recombine — output slices are already contiguous
            speedupVsBestSingle: 0
        )
    }

    // MARK: - Embedding Execution

    /// Execute an embedding lookup op with vocab-range split partitioning.
    ///
    /// Each unit owns a shard of the embedding table. All units read all token IDs
    /// and write output only for tokens in their vocab range (scatter-free).
    ///
    /// The output buffer must be pre-zeroed before dispatch (each output row is
    /// written by exactly one unit; unwritten rows must remain zero).
    ///
    /// Input buffers: [0]=tokenIds [seqLen] (int32), [1]=embeddingTable [vocabSize, embeddingDim] (float32)
    /// Output buffer: [seqLen, embeddingDim]
    private func executeEmbedding(
        descriptor: PartitionDescriptor,
        inputBuffers: [MTLBuffer],
        outputBuffer: MTLBuffer
    ) -> FusedProfile {
        guard inputBuffers.count >= 2 else { return emptyProfile() }

        let bufTokenIds = inputBuffers[0]
        let bufTable = inputBuffers[1]
        let bufOutput = outputBuffer

        let seqLen = descriptor.fullOutputShape[0]
        let embeddingDim = descriptor.fullOutputShape[1]

        // Pre-zero the output buffer (precondition for scatter-free writes)
        guard let blitCB = embeddingGPU.queue.makeCommandBuffer(),
              let blit = blitCB.makeBlitCommandEncoder() else { return emptyProfile() }
        blit.fill(buffer: bufOutput, range: 0..<bufOutput.length, value: 0)
        blit.endEncoding()
        blitCB.commit()
        blitCB.waitUntilCompleted()

        // 2-unit: GPU + CPU only (ANE has no gather hardware)
        let gpuAssignments = descriptor.assignments.filter { $0.unit == .gpu }
        let cpuAssignments = descriptor.assignments.filter { $0.unit == .cpu }

        // Prepare GPU command buffers
        var gpuCommandBuffers: [(MTLCommandBuffer, UnitAssignment)] = []
        for assignment in gpuAssignments {
            guard let cb = embeddingGPU.queue.makeCommandBuffer() else { continue }
            let tableSlice = assignment.inputSlices[1]
            let shardVocabSize = tableSlice.shape[0]
            let vocabOffset = tableSlice.bufferOffset / (embeddingDim * descriptor.dtype.byteSize)

            embeddingGPU.encode(
                tokenIds: bufTokenIds, tableShard: bufTable, output: bufOutput,
                tableShardOffset: tableSlice.bufferOffset,
                seqLen: seqLen, embeddingDim: embeddingDim,
                vocabOffset: vocabOffset, vocabSize: shardVocabSize,
                commandBuffer: cb
            )
            gpuCommandBuffers.append((cb, assignment))
        }

        // Commit all before waiting
        let start = CACurrentMediaTime()
        for (cb, _) in gpuCommandBuffers { cb.commit() }

        // CPU runs concurrently
        var cpuProfiles: [UnitProfile] = []
        for assignment in cpuAssignments {
            let tableSlice = assignment.inputSlices[1]
            let shardVocabSize = tableSlice.shape[0]
            let vocabOffset = tableSlice.bufferOffset / (embeddingDim * descriptor.dtype.byteSize)

            let cpuStart = CACurrentMediaTime()
            let tokenPtr = bufTokenIds.contents().assumingMemoryBound(to: Int32.self)
            let tablePtr = (bufTable.contents() + tableSlice.bufferOffset).assumingMemoryBound(to: Float.self)
            let outPtr = bufOutput.contents().assumingMemoryBound(to: Float.self)

            embeddingCPU.execute(
                tokenIds: tokenPtr, tableShard: tablePtr, output: outPtr,
                seqLen: seqLen, embeddingDim: embeddingDim,
                vocabOffset: vocabOffset, vocabSize: shardVocabSize
            )
            let cpuMs = (CACurrentMediaTime() - cpuStart) * 1000.0

            // Bandwidth: read tokenIds + output rows touched
            let bytes = Double(seqLen * 4 + seqLen * embeddingDim * 4)
            cpuProfiles.append(UnitProfile(
                unit: .cpu, wallClockMs: cpuMs,
                bandwidthGBs: cpuMs > 0 ? bytes / (cpuMs / 1000.0) / 1e9 : 0,
                tflops: 0, syncOverheadMs: 0
            ))
        }

        // Wait for GPU
        var gpuProfiles: [UnitProfile] = []
        for (cb, _) in gpuCommandBuffers {
            cb.waitUntilCompleted()
            let ms = (CACurrentMediaTime() - start) * 1000.0
            gpuProfiles.append(UnitProfile(
                unit: .gpu, wallClockMs: ms,
                bandwidthGBs: 0, tflops: 0, syncOverheadMs: 0
            ))
        }

        let totalMs = (CACurrentMediaTime() - start) * 1000.0
        let allProfiles = gpuProfiles + cpuProfiles
        var splitRatio: [ComputeUnit: Double] = [:]
        for a in descriptor.assignments {
            splitRatio[a.unit, default: 0] += a.workFraction
        }

        return FusedProfile(
            totalWallClockMs: totalMs,
            perUnit: allProfiles,
            splitRatio: splitRatio,
            recombineMs: 0,  // No recombine — scatter-free output, pre-zeroed
            speedupVsBestSingle: 0
        )
    }

    // MARK: - Convolution Execution

    /// Execute a convolution with output-channel-split partitioning.
    ///
    /// This is a 3-unit compute-bound op — GPU, MPS/ANE, and CPU all participate.
    /// MPSCNNConvolution routes to the Neural Engine for eligible shapes.
    ///
    /// Input buffers: [0]=input [B, Cin, H, W] (float32 NCHW),
    ///                [1]=weights [Cout, Cin, Kh, Kw] (float32)
    /// Output buffer: [B, Cout, outH, outW]
    ///
    /// Each unit computes a slice of output channels using the full input.
    /// In NCHW layout, each unit's output is contiguous for batch=1.
    private func executeConvolution(
        descriptor: PartitionDescriptor,
        inputBuffers: [MTLBuffer],
        outputBuffer: MTLBuffer
    ) -> FusedProfile {
        guard inputBuffers.count >= 2 else { return emptyProfile() }

        let bufInput = inputBuffers[0]
        let bufWeights = inputBuffers[1]
        let bufOutput = outputBuffer

        // Reconstruct ConvShape from descriptor
        let fullInput = descriptor.fullInputShape       // [B, Cin, H, W]
        let fullWeights = descriptor.secondaryInputShape! // [Cout, Cin, Kh, Kw]
        let fullOutput = descriptor.fullOutputShape     // [B, Cout, outH, outW]

        let batch = fullInput[0]
        let inC = fullInput[1]
        let H = fullInput[2]
        let W = fullInput[3]
        let outC = fullWeights[0]
        let kH = fullWeights[2]
        let kW = fullWeights[3]
        let outH = fullOutput[2]
        let outW = fullOutput[3]

        // Infer stride and padding from dimensions
        // outH = (H + 2*padH - kH) / strideH + 1
        // For now, assume stride=1 and compute padding
        let strideH = 1
        let strideW = 1
        let padH = (outH - 1) * strideH - H + kH
        let padW = (outW - 1) * strideW - W + kW
        let padHHalf = max(0, padH / 2)
        let padWHalf = max(0, padW / 2)

        let shape = ConvShape(
            batch: batch, inChannels: inC, outChannels: outC,
            height: H, width: W,
            kernelH: kH, kernelW: kW,
            strideH: strideH, strideW: strideW,
            padH: padHHalf, padW: padWHalf
        )

        // 3-unit: GPU + MPS + CPU
        let gpuAssignments = descriptor.assignments.filter { $0.unit == .gpu }
        let mpsAssignments = descriptor.assignments.filter { $0.unit == .mps }
        let cpuAssignments = descriptor.assignments.filter { $0.unit == .cpu }

        // Prepare GPU command buffers
        var gpuCommandBuffers: [(MTLCommandBuffer, UnitAssignment)] = []
        for assignment in gpuAssignments {
            guard let cb = convGPU.queue.makeCommandBuffer() else { continue }
            let weightsSlice = assignment.inputSlices[1]
            let sliceOutC = weightsSlice.shape[0]

            let sliceShape = ConvShape(
                batch: batch, inChannels: inC, outChannels: sliceOutC,
                height: H, width: W,
                kernelH: kH, kernelW: kW,
                strideH: strideH, strideW: strideW,
                padH: padHHalf, padW: padWHalf
            )

            convGPU.encode(
                input: bufInput, weights: bufWeights, output: bufOutput,
                weightsOffset: weightsSlice.bufferOffset,
                outputOffset: assignment.outputSlice.bufferOffset,
                shape: sliceShape, sliceOutC: sliceOutC,
                commandBuffer: cb
            )
            gpuCommandBuffers.append((cb, assignment))
        }

        // Prepare MPS command buffers using encode+commit pattern.
        // Buffer→texture conversion happens before commit (CPU side),
        // then MPS conv runs on ANE/GPU in parallel with our custom GPU kernel.
        // Texture→buffer readback happens after waitUntilCompleted.
        struct MPSEncodedWork {
            let commandBuffer: MTLCommandBuffer
            let assignment: UnitAssignment
            let outputImage: MPSImage
            let sliceOutC: Int
            let outputOffset: Int
        }
        var mpsEncodedWork: [MPSEncodedWork] = []
        for assignment in mpsAssignments {
            let weightsSlice = assignment.inputSlices[1]
            let sliceOutC = weightsSlice.shape[0]
            let mpsShape = ConvShape(
                batch: batch, inChannels: inC, outChannels: sliceOutC,
                height: H, width: W,
                kernelH: kH, kernelW: kW,
                strideH: strideH, strideW: strideW,
                padH: padHHalf, padW: padWHalf
            )
            guard let cb = convMPS.queue.makeCommandBuffer() else { continue }
            if let outputImage = convMPS.encodeReturningImage(
                input: bufInput, weights: bufWeights,
                weightsOffset: weightsSlice.bufferOffset,
                shape: mpsShape, sliceOutC: sliceOutC,
                commandBuffer: cb
            ) {
                mpsEncodedWork.append(MPSEncodedWork(
                    commandBuffer: cb,
                    assignment: assignment,
                    outputImage: outputImage,
                    sliceOutC: sliceOutC,
                    outputOffset: assignment.outputSlice.bufferOffset
                ))
            }
        }

        // ── Commit all GPU and MPS command buffers BEFORE waiting on any ──
        let start = CACurrentMediaTime()
        for (cb, _) in gpuCommandBuffers { cb.commit() }
        for work in mpsEncodedWork { work.commandBuffer.commit() }

        // Run CPU slices concurrently with GPU and MPS
        var cpuProfiles: [UnitProfile] = []
        for assignment in cpuAssignments {
            let weightsSlice = assignment.inputSlices[1]
            let sliceOutC = weightsSlice.shape[0]

            let cpuStart = CACurrentMediaTime()
            _ = convCPU.execute(
                input: bufInput, weights: bufWeights, output: bufOutput,
                weightsOffset: weightsSlice.bufferOffset,
                outputOffset: assignment.outputSlice.bufferOffset,
                shape: shape, sliceOutC: sliceOutC
            )
            let cpuMs = (CACurrentMediaTime() - cpuStart) * 1000.0

            cpuProfiles.append(UnitProfile(
                unit: .cpu, wallClockMs: cpuMs,
                bandwidthGBs: 0,
                tflops: convTflops(sliceOutC: sliceOutC, shape: shape, timeS: cpuMs / 1000.0),
                syncOverheadMs: 0
            ))
        }

        // Wait for GPU completions
        var gpuProfiles: [UnitProfile] = []
        for (cb, assignment) in gpuCommandBuffers {
            cb.waitUntilCompleted()
            let ms = (CACurrentMediaTime() - start) * 1000.0
            let sliceOutC = assignment.inputSlices[1].shape[0]
            gpuProfiles.append(UnitProfile(
                unit: .gpu, wallClockMs: ms,
                bandwidthGBs: 0,
                tflops: convTflops(sliceOutC: sliceOutC, shape: shape, timeS: ms / 1000.0),
                syncOverheadMs: 0
            ))
        }

        // Wait for MPS completions, then read back texture→buffer
        var mpsProfiles: [UnitProfile] = []
        for work in mpsEncodedWork {
            work.commandBuffer.waitUntilCompleted()
            let ms = (CACurrentMediaTime() - start) * 1000.0

            convMPS.readBackImage(
                work.outputImage, to: bufOutput, bufferOffset: work.outputOffset,
                batch: batch, channels: work.sliceOutC,
                height: outH, width: outW
            )

            mpsProfiles.append(UnitProfile(
                unit: .mps, wallClockMs: ms,
                bandwidthGBs: 0,
                tflops: convTflops(sliceOutC: work.sliceOutC, shape: shape, timeS: ms / 1000.0),
                syncOverheadMs: 0
            ))
        }

        let totalMs = (CACurrentMediaTime() - start) * 1000.0
        let allProfiles = gpuProfiles + mpsProfiles + cpuProfiles
        var splitRatio: [ComputeUnit: Double] = [:]
        for a in descriptor.assignments {
            splitRatio[a.unit, default: 0] += a.workFraction
        }

        return FusedProfile(
            totalWallClockMs: totalMs,
            perUnit: allProfiles,
            splitRatio: splitRatio,
            recombineMs: 0,  // NCHW output-channel split is contiguous (batch=1)
            speedupVsBestSingle: 0
        )
    }

    private func convTflops(sliceOutC: Int, shape: ConvShape, timeS: Double) -> Double {
        guard timeS > 0 else { return 0 }
        let flops = 2.0 * Double(shape.batch) * Double(shape.outHeight) * Double(shape.outWidth) *
                    Double(sliceOutC) * Double(shape.inChannels) * Double(shape.kernelH) * Double(shape.kernelW)
        return flops / timeS / 1e12
    }

    // MARK: - Helpers

    private func emptyProfile() -> FusedProfile {
        FusedProfile(totalWallClockMs: 0, perUnit: [], splitRatio: [:], recombineMs: 0, speedupVsBestSingle: 0)
    }

    private func bandwidth(rows: Int, K: Int, N: Int, timeS: Double) -> Double {
        guard timeS > 0 else { return 0 }
        return Double(rows * K + K * N + rows * N) * 4.0 / timeS / 1e9
    }

    private func tflops(rows: Int, K: Int, N: Int, timeS: Double) -> Double {
        guard timeS > 0 else { return 0 }
        return 2.0 * Double(rows) * Double(K) * Double(N) / timeS / 1e12
    }
}
