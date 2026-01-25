// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SwiftExample",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "SwiftExample",
            dependencies: [
                .product(name: "MetalHLO", package: "MetalHLO"),
            ]
        ),
    ]
)
