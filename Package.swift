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
    ],
    swiftLanguageModes: [.v6]
)
