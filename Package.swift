// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MetalHLO",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        // Swift Public API
        .library(
            name: "MetalHLO",
            targets: ["MetalHLO"]
        ),
        // C API (dynamic library for C/C++ interop)
        .library(
            name: "CMetalHLO",
            type: .dynamic,
            targets: ["CMetalHLO"]
        ),
        // PJRT plugin (dynamic library for JAX/XLA integration)
        .library(
            name: "PJRTMetalHLO",
            type: .dynamic,
            targets: ["PJRTMetalHLO"]
        ),
        // Benchmark library
        .library(
            name: "MetalHLOBenchmarks",
            targets: ["MetalHLOBenchmarks"]
        ),
        // MLX comparison benchmarks
        .library(
            name: "MLXBenchmarks",
            targets: ["MLXBenchmarks"]
        ),
        // Benchmark runner executable
        .executable(
            name: "benchmark-runner",
            targets: ["BenchmarkRunner"]
        ),
        // MLX comparison runner executable
        .executable(
            name: "mlx-comparison",
            targets: ["MLXComparison"]
        ),
    ],
    dependencies: [
        // MLX Swift for comparison benchmarks
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
    ],
    targets: [
        // MARK: - Public Swift API
        .target(
            name: "MetalHLO",
            dependencies: ["MetalHLOCore"],
            path: "Sources/MetalHLO"
        ),

        // MARK: - Core Implementation
        .target(
            name: "MetalHLOCore",
            dependencies: [],
            path: "Sources/MetalHLOCore",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShadersGraph"),
            ]
        ),

        // MARK: - C API
        .target(
            name: "CMetalHLO",
            dependencies: ["MetalHLO"],
            path: "Sources/CMetalHLO",
            publicHeadersPath: "include"
        ),

        // MARK: - PJRT C API Header (vendored from OpenXLA)
        .target(
            name: "CPJRTApi",
            dependencies: [],
            path: "Sources/CPJRTApi",
            publicHeadersPath: "include"
        ),

        // MARK: - PJRT Plugin Implementation
        .target(
            name: "PJRTMetalHLO",
            dependencies: ["MetalHLO", "MetalHLOCore", "CPJRTApi"],
            path: "Sources/PJRTMetalHLO",
            linkerSettings: [
                .linkedFramework("Metal"),
            ]
        ),

        // MARK: - Benchmarks
        .target(
            name: "MetalHLOBenchmarks",
            dependencies: ["MetalHLO"],
            path: "Sources/MetalHLOBenchmarks"
        ),
        .executableTarget(
            name: "BenchmarkRunner",
            dependencies: ["MetalHLOBenchmarks"],
            path: "Sources/BenchmarkRunner"
        ),

        // MARK: - MLX Comparison Benchmarks
        .target(
            name: "MLXBenchmarks",
            dependencies: [
                "MetalHLOBenchmarks",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/MLXBenchmarks"
        ),
        .executableTarget(
            name: "MLXComparison",
            dependencies: ["MLXBenchmarks", "MetalHLOBenchmarks"],
            path: "Sources/MLXComparison"
        ),

        // MARK: - Tests
        .testTarget(
            name: "MetalHLOTests",
            dependencies: ["MetalHLO", "CMetalHLO"],
            path: "Tests/MetalHLOTests"
        ),
        .testTarget(
            name: "MetalHLOCoreTests",
            dependencies: ["MetalHLOCore"],
            path: "Tests/MetalHLOCoreTests"
        ),
        .testTarget(
            name: "PJRTMetalHLOTests",
            dependencies: ["PJRTMetalHLO", "MetalHLO"],
            path: "Tests/PJRTMetalHLOTests"
        ),
    ],
    swiftLanguageModes: [.v6]
)
