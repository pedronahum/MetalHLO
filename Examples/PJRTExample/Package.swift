// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "PJRTExample",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "PJRTExample",
            dependencies: [
                .product(name: "PJRTMetalHLO", package: "MetalHLO"),
            ]
        ),
    ]
)
