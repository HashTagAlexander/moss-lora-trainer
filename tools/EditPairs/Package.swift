// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "EditPairs",
    platforms: [.macOS(.v13)],
    targets: [
        .target(name: "EditPairsCore"),
        .executableTarget(name: "EditPairs", dependencies: ["EditPairsCore"]),
        .testTarget(name: "EditPairsCoreTests", dependencies: ["EditPairsCore"]),
    ]
)
