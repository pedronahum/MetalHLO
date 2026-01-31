// TestDataGenerator.swift
// MetalHLO Benchmarks
//
// Reproducible test data generation for benchmarks.

import Foundation
import MetalHLO

/// Generator for reproducible test data.
public struct TestDataGenerator: Sendable {
    /// Seed for random number generation.
    public let seed: UInt64

    public init(seed: UInt64 = 42) {
        self.seed = seed
    }

    /// A simple, reproducible random number generator (xorshift64).
    private struct Xorshift64: Sendable {
        var state: UInt64

        mutating func next() -> UInt64 {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return state
        }

        mutating func nextFloat() -> Float {
            return Float(next() & 0xFFFFFF) / Float(0xFFFFFF)
        }

        mutating func nextDouble() -> Double {
            return Double(next()) / Double(UInt64.max)
        }
    }

    // MARK: - Float32 Generation

    /// Generate Float32 array with uniform distribution [0, 1).
    public func uniformFloat32(count: Int) -> [Float] {
        var rng = Xorshift64(state: seed)
        return (0..<count).map { _ in rng.nextFloat() }
    }

    /// Generate Float32 array with uniform distribution [min, max).
    public func uniformFloat32(count: Int, min: Float, max: Float) -> [Float] {
        var rng = Xorshift64(state: seed)
        let range = max - min
        return (0..<count).map { _ in min + rng.nextFloat() * range }
    }

    /// Generate Float32 array with normal distribution (Box-Muller).
    public func normalFloat32(count: Int, mean: Float = 0, stdDev: Float = 1) -> [Float] {
        var rng = Xorshift64(state: seed)
        var result: [Float] = []
        result.reserveCapacity(count)

        while result.count < count {
            let u1 = max(rng.nextFloat(), Float.leastNormalMagnitude)
            let u2 = rng.nextFloat()
            let mag = stdDev * sqrt(-2.0 * log(u1))
            let z0 = mag * cos(2 * .pi * u2) + mean
            let z1 = mag * sin(2 * .pi * u2) + mean
            result.append(z0)
            if result.count < count {
                result.append(z1)
            }
        }

        return result
    }

    /// Generate Float32 zeros.
    public func zerosFloat32(count: Int) -> [Float] {
        return [Float](repeating: 0, count: count)
    }

    /// Generate Float32 ones.
    public func onesFloat32(count: Int) -> [Float] {
        return [Float](repeating: 1, count: count)
    }

    /// Generate Float32 with constant value.
    public func constantFloat32(count: Int, value: Float) -> [Float] {
        return [Float](repeating: value, count: count)
    }

    // MARK: - Float16 Generation

    /// Generate Float16 array with uniform distribution [0, 1).
    @available(macOS 11.0, *)
    public func uniformFloat16(count: Int) -> [Float16] {
        var rng = Xorshift64(state: seed)
        return (0..<count).map { _ in Float16(rng.nextFloat()) }
    }

    /// Generate Float16 array with uniform distribution [min, max).
    @available(macOS 11.0, *)
    public func uniformFloat16(count: Int, min: Float, max: Float) -> [Float16] {
        var rng = Xorshift64(state: seed)
        let range = max - min
        return (0..<count).map { _ in Float16(min + rng.nextFloat() * range) }
    }

    // MARK: - Int32 Generation

    /// Generate Int32 array with uniform distribution [0, max).
    public func uniformInt32(count: Int, max: Int32) -> [Int32] {
        var rng = Xorshift64(state: seed)
        return (0..<count).map { _ in Int32(rng.next() % UInt64(max)) }
    }

    /// Generate Int32 array with uniform distribution [min, max).
    public func uniformInt32(count: Int, min: Int32, max: Int32) -> [Int32] {
        var rng = Xorshift64(state: seed)
        let range = UInt64(max - min)
        return (0..<count).map { _ in min + Int32(rng.next() % range) }
    }

    // MARK: - Buffer Creation Helpers

    /// Create a buffer with uniform Float32 data.
    public func createUniformFloat32Buffer(
        client: Client,
        shape: [Int]
    ) throws -> Buffer {
        let count = shape.reduce(1, *)
        let data = uniformFloat32(count: count)
        return try client.createBuffer(data, shape: shape)
    }

    /// Create a buffer with uniform Float32 data in range.
    public func createUniformFloat32Buffer(
        client: Client,
        shape: [Int],
        min: Float,
        max: Float
    ) throws -> Buffer {
        let count = shape.reduce(1, *)
        let data = uniformFloat32(count: count, min: min, max: max)
        return try client.createBuffer(data, shape: shape)
    }

    /// Create a buffer with normal Float32 data.
    public func createNormalFloat32Buffer(
        client: Client,
        shape: [Int],
        mean: Float = 0,
        stdDev: Float = 1
    ) throws -> Buffer {
        let count = shape.reduce(1, *)
        let data = normalFloat32(count: count, mean: mean, stdDev: stdDev)
        return try client.createBuffer(data, shape: shape)
    }

    /// Create a buffer with uniform Float16 data.
    @available(macOS 11.0, *)
    public func createUniformFloat16Buffer(
        client: Client,
        shape: [Int]
    ) throws -> Buffer {
        let count = shape.reduce(1, *)
        let data = uniformFloat16(count: count)
        return try client.createBuffer(data, shape: shape, elementType: .float16)
    }

    /// Create a zeros buffer.
    public func createZerosBuffer(
        client: Client,
        shape: [Int],
        elementType: ElementType = .float32
    ) throws -> Buffer {
        let count = shape.reduce(1, *)
        switch elementType {
        case .float32:
            return try client.createBuffer(zerosFloat32(count: count), shape: shape)
        case .float16:
            let data = [Float16](repeating: 0, count: count)
            return try client.createBuffer(data, shape: shape, elementType: .float16)
        case .int32:
            let data = [Int32](repeating: 0, count: count)
            return try client.createBuffer(data, shape: shape, elementType: .int32)
        default:
            // Fallback to float32
            return try client.createBuffer(zerosFloat32(count: count), shape: shape)
        }
    }

    /// Create an identity matrix buffer.
    public func createIdentityMatrixBuffer(
        client: Client,
        size: Int
    ) throws -> Buffer {
        var data = [Float](repeating: 0, count: size * size)
        for i in 0..<size {
            data[i * size + i] = 1.0
        }
        return try client.createBuffer(data, shape: [size, size])
    }
}

// MARK: - Common Tensor Shapes

/// Standard shapes used across benchmarks.
public enum StandardShapes {
    /// Small square matrices.
    public static let smallSquare: [[Int]] = [
        [128, 128],
        [256, 256],
        [512, 512]
    ]

    /// Medium square matrices.
    public static let mediumSquare: [[Int]] = [
        [1024, 1024],
        [2048, 2048]
    ]

    /// Large square matrices.
    public static let largeSquare: [[Int]] = [
        [4096, 4096],
        [8192, 8192]
    ]

    /// Transformer-like shapes.
    public static let transformerShapes: [[Int]] = [
        [32, 128, 768],      // batch, seq, hidden
        [32, 512, 768],
        [32, 1024, 768],
        [8, 2048, 768]
    ]

    /// Attention shapes: [batch, heads, seq, head_dim]
    public static let attentionShapes: [[Int]] = [
        [1, 12, 128, 64],
        [1, 12, 512, 64],
        [8, 12, 128, 64],
        [1, 32, 128, 128]   // LLaMA-like
    ]

    /// CNN feature map shapes: [batch, height, width, channels]
    public static let cnnShapes: [[Int]] = [
        [1, 224, 224, 3],    // Input
        [1, 56, 56, 64],     // Early layers
        [1, 28, 28, 128],    // Mid layers
        [1, 14, 14, 256],    // Late layers
        [32, 56, 56, 64]     // Batched
    ]

    /// GEMM shapes: [M, K], [K, N] for various workloads.
    public static let gemmConfigs: [(m: Int, n: Int, k: Int)] = [
        (128, 128, 128),     // Small
        (512, 512, 512),     // Medium
        (1024, 1024, 1024),  // Large
        (2048, 2048, 2048),  // Very large
        (32, 4096, 768),     // Transformer-like
        (128, 768, 3072),    // MLP layer
        (1, 4096, 4096)      // Vector-matrix
    ]
}

// MARK: - FLOPS Calculation Helpers

/// Helper functions for calculating theoretical FLOPS.
public enum FLOPSCalculator {
    /// FLOPS for matrix multiplication: 2 * M * N * K
    public static func matmul(m: Int, n: Int, k: Int) -> Double {
        return Double(2 * m * n * k)
    }

    /// FLOPS for batched matrix multiplication.
    public static func batchedMatmul(batch: Int, m: Int, n: Int, k: Int) -> Double {
        return Double(batch) * matmul(m: m, n: n, k: k)
    }

    /// FLOPS for element-wise operation (1 FLOP per element).
    public static func elementWise(elements: Int) -> Double {
        return Double(elements)
    }

    /// FLOPS for reduction (N-1 operations for N elements).
    public static func reduction(elements: Int) -> Double {
        return Double(max(0, elements - 1))
    }

    /// FLOPS for convolution (approximate).
    public static func conv2d(
        batchSize: Int,
        inputHeight: Int,
        inputWidth: Int,
        inputChannels: Int,
        outputChannels: Int,
        kernelHeight: Int,
        kernelWidth: Int,
        stride: Int = 1
    ) -> Double {
        let outputHeight = inputHeight / stride
        let outputWidth = inputWidth / stride
        let flopsPerOutput = 2 * kernelHeight * kernelWidth * inputChannels
        return Double(batchSize * outputHeight * outputWidth * outputChannels * flopsPerOutput)
    }

    /// Calculate achieved GFLOPS.
    public static func gflops(flops: Double, timeSeconds: Double) -> Double {
        guard timeSeconds > 0 else { return 0 }
        return flops / timeSeconds / 1e9
    }
}

// MARK: - Memory Bandwidth Calculation

/// Helper for calculating memory bandwidth.
public enum MemoryBandwidthCalculator {
    /// Calculate bytes transferred for a tensor.
    public static func tensorBytes(shape: [Int], elementType: ElementType) -> Int {
        let count = shape.reduce(1, *)
        return count * elementType.byteSize
    }

    /// Calculate achieved bandwidth in GB/s.
    public static func bandwidthGBps(bytes: Int, timeSeconds: Double) -> Double {
        guard timeSeconds > 0 else { return 0 }
        return Double(bytes) / timeSeconds / 1e9
    }
}

