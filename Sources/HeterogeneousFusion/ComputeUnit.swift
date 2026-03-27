// ComputeUnit.swift
// HeterogeneousFusion
//
// Core types for heterogeneous cross-unit execution.

import Foundation
import Metal

/// A compute unit available on Apple Silicon.
public enum ComputeUnit: String, Sendable, CaseIterable, Codable {
    case cpu
    case gpu
    case mps   // MPS framework — may route to ANE for supported ops (matmul, conv)
}

/// Profiling data for a single compute unit's execution of an operation.
public struct UnitProfile: Sendable {
    /// Which unit executed.
    public let unit: ComputeUnit
    /// Wall-clock time in milliseconds.
    public let wallClockMs: Double
    /// Effective bandwidth: bytes moved / time (GB/s).
    public let bandwidthGBs: Double
    /// Throughput in TFLOPS.
    public let tflops: Double
    /// Synchronization overhead in milliseconds (fence/wait cost only).
    public let syncOverheadMs: Double

    public init(unit: ComputeUnit, wallClockMs: Double, bandwidthGBs: Double, tflops: Double, syncOverheadMs: Double) {
        self.unit = unit
        self.wallClockMs = wallClockMs
        self.bandwidthGBs = bandwidthGBs
        self.tflops = tflops
        self.syncOverheadMs = syncOverheadMs
    }
}

/// Profiling data for a fused cross-unit execution.
public struct FusedProfile: Sendable {
    /// Total wall-clock time in milliseconds.
    public let totalWallClockMs: Double
    /// Per-unit breakdown.
    public let perUnit: [UnitProfile]
    /// Split ratio per unit (fractions summing to 1.0).
    public let splitRatio: [ComputeUnit: Double]
    /// Time spent recombining output slices (ms).
    public let recombineMs: Double
    /// Speedup vs the best single-unit execution.
    public let speedupVsBestSingle: Double

    public init(
        totalWallClockMs: Double,
        perUnit: [UnitProfile],
        splitRatio: [ComputeUnit: Double],
        recombineMs: Double,
        speedupVsBestSingle: Double
    ) {
        self.totalWallClockMs = totalWallClockMs
        self.perUnit = perUnit
        self.splitRatio = splitRatio
        self.recombineMs = recombineMs
        self.speedupVsBestSingle = speedupVsBestSingle
    }
}

/// A matrix shape for benchmarking.
public struct MatrixShape: Sendable, CustomStringConvertible {
    public let M: Int
    public let K: Int
    public let N: Int

    /// Square matrix: M×K @ K×N where M=K=N=size.
    public static func square(_ size: Int) -> MatrixShape {
        MatrixShape(M: size, K: size, N: size)
    }

    /// General shape: [M, K] @ [K, N] → [M, N].
    public init(M: Int, K: Int, N: Int) {
        self.M = M
        self.K = K
        self.N = N
    }

    public var description: String {
        if M == K && K == N { return "\(M)×\(M)" }
        return "\(M)×\(K) @ \(K)×\(N)"
    }

    /// Total FLOPs for a matmul of this shape (2*M*N*K).
    public var flops: Double {
        2.0 * Double(M) * Double(N) * Double(K)
    }

    /// Total bytes read + written for float32: A + B + C.
    public var totalBytes: Double {
        Double(M * K + K * N + M * N) * 4.0
    }
}

/// Convolution shape parameters (NCHW layout).
public struct ConvShape: Sendable, CustomStringConvertible {
    public let batch: Int
    public let inChannels: Int
    public let outChannels: Int
    public let height: Int          // input spatial height
    public let width: Int           // input spatial width
    public let kernelH: Int
    public let kernelW: Int
    public let strideH: Int
    public let strideW: Int
    public let padH: Int
    public let padW: Int

    public init(
        batch: Int, inChannels: Int, outChannels: Int,
        height: Int, width: Int,
        kernelH: Int, kernelW: Int,
        strideH: Int = 1, strideW: Int = 1,
        padH: Int = 0, padW: Int = 0
    ) {
        self.batch = batch
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.height = height
        self.width = width
        self.kernelH = kernelH
        self.kernelW = kernelW
        self.strideH = strideH
        self.strideW = strideW
        self.padH = padH
        self.padW = padW
    }

    /// Output spatial height after convolution.
    public var outHeight: Int { (height + 2 * padH - kernelH) / strideH + 1 }
    /// Output spatial width after convolution.
    public var outWidth: Int { (width + 2 * padW - kernelW) / strideW + 1 }
    /// Total output elements.
    public var outputElements: Int { batch * outChannels * outHeight * outWidth }
    /// Total FLOPs: 2 * B * outH * outW * Cout * Cin * kH * kW.
    public var flops: Double {
        2.0 * Double(batch) * Double(outHeight) * Double(outWidth) *
        Double(outChannels) * Double(inChannels) * Double(kernelH) * Double(kernelW)
    }
    /// Total input buffer bytes (float32, NCHW).
    public var inputBytes: Int { batch * inChannels * height * width * MemoryLayout<Float>.size }
    /// Total weight buffer bytes (float32).
    public var weightBytes: Int { outChannels * inChannels * kernelH * kernelW * MemoryLayout<Float>.size }
    /// Total output buffer bytes (float32, NCHW).
    public var outputBytes: Int { batch * outChannels * outHeight * outWidth * MemoryLayout<Float>.size }

    /// Equivalent MatrixShape for solver/profitability evaluation.
    /// Maps conv to GEMM: M = B*outH*outW, K = Cin*kH*kW, N = Cout.
    public var asMatrixShape: MatrixShape {
        MatrixShape(
            M: batch * outHeight * outWidth,
            K: inChannels * kernelH * kernelW,
            N: outChannels
        )
    }

    public var description: String {
        let padStr = (padH > 0 || padW > 0) ? " pad=\(padH)×\(padW)" : ""
        let strStr = (strideH > 1 || strideW > 1) ? " stride=\(strideH)×\(strideW)" : ""
        return "\(batch)×\(inChannels)×\(height)×\(width) → \(outChannels)ch \(kernelH)×\(kernelW)\(strStr)\(padStr)"
    }
}
