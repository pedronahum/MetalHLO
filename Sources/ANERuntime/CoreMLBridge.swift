// CoreMLBridge.swift
// ANERuntime
//
// Compiles MIL programs to CoreML models and executes them.
// Pure Swift — no Python, no coremltools dependency.
//
// Flow:
//   1. CoreMLModelBuilder produces protobuf binary from CoreMLOp list
//   2. Written to .mlpackage directory on disk
//   3. MLModel.compileModel() compiles to .mlmodelc
//   4. MLModel(contentsOf:) loads for execution
//   5. MLModel.prediction() runs inference

import Foundation
import CoreML

/// Compiles and executes MIL programs via CoreML.
public final class CoreMLBridge: @unchecked Sendable {

    /// Base directory for temporary .mlpackage files.
    private let cacheDirectory: URL

    private let lock = NSLock()

    public init() {
        self.cacheDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MetalHLO_CoreML/\(UUID().uuidString)", isDirectory: true)
    }

    /// Compiles a MIL program to a CoreML model.
    ///
    /// - Parameters:
    ///   - inputs: Function input names and shapes.
    ///   - operations: MIL operations in topological order.
    ///   - returnVar: Name of the return variable.
    ///   - weightsData: Optional concatenated weight blob.
    /// - Returns: A compiled CoreMLProgram ready for execution.
    public func compile(
        inputs: [(name: String, shape: [Int])],
        operations: [CoreMLOp],
        returnVar: String,
        weightsData: Data? = nil
    ) throws -> CoreMLProgram {
        let programId = UUID().uuidString

        // Build protobuf model spec
        let modelData = CoreMLModelBuilder.build(
            inputs: inputs,
            operations: operations,
            returnVar: returnVar
        )

        // Write .mlpackage directory
        let packagePath = try writeMLPackage(
            id: programId,
            modelData: modelData,
            weightsData: weightsData
        )

        // Compile with CoreML (guarded by watchdog)
        let compiledPath: URL
        do {
            compiledPath = try ANEWatchdog.withTimeout {
                try MLModel.compileModel(at: packagePath)
            }
        } catch let error as ANEError {
            throw error
        } catch {
            throw ANEError.coreMLCompilationFailed(
                "MLModel.compileModel failed: \(error.localizedDescription)"
            )
        }

        // Load compiled model
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model: MLModel
        do {
            model = try MLModel(contentsOf: compiledPath, configuration: config)
        } catch {
            throw ANEError.coreMLCompilationFailed(
                "MLModel load failed: \(error.localizedDescription)"
            )
        }

        // Determine output shape
        let outputShape: [Int]
        if let lastOp = operations.last(where: { $0.result == returnVar }) {
            outputShape = lastOp.shape
        } else {
            outputShape = []
        }

        return CoreMLProgram(
            id: programId,
            model: model,
            inputNames: inputs.map { $0.name },
            inputShapes: inputs.map { $0.shape },
            outputShape: outputShape,
            compiledModelPath: compiledPath,
            sourcePackagePath: packagePath
        )
    }

    /// Executes a compiled CoreML program with Float32 input arrays.
    ///
    /// - Parameters:
    ///   - program: A compiled CoreMLProgram.
    ///   - inputs: Named input arrays with shapes.
    /// - Returns: Output as flat Float32 array.
    public func execute(
        _ program: CoreMLProgram,
        inputs: [(name: String, data: [Float], shape: [Int])]
    ) throws -> [Float] {
        guard program.isValid else {
            throw ANEError.executionFailed("CoreML program has been released")
        }

        // Build feature provider
        var featureDict: [String: MLFeatureValue] = [:]
        for input in inputs {
            let mlArray = try MLMultiArray(
                shape: input.shape.map { NSNumber(value: $0) },
                dataType: .float32
            )
            let ptr = mlArray.dataPointer.bindMemory(
                to: Float.self,
                capacity: input.data.count
            )
            for i in 0..<input.data.count {
                ptr[i] = input.data[i]
            }
            featureDict[input.name] = MLFeatureValue(multiArray: mlArray)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: featureDict)
        let prediction = try ANEWatchdog.withTimeout {
            try program.model.prediction(from: provider)
        }

        // Extract output — use first feature
        guard let outputName = prediction.featureNames.first,
              let outputValue = prediction.featureValue(for: outputName),
              let outputArray = outputValue.multiArrayValue else {
            throw ANEError.executionFailed("No output from CoreML prediction")
        }

        // Convert MLMultiArray to [Float]
        let count = outputArray.count
        var result = [Float](repeating: 0, count: count)
        let outPtr = outputArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            result[i] = outPtr[i]
        }
        return result
    }

    /// Executes a compiled CoreML program with pre-built MLMultiArray inputs.
    ///
    /// This avoids the Float array → MLMultiArray copy on the input path,
    /// enabling zero-copy execution when inputs are backed by shared memory
    /// (e.g., IOSurface-backed Metal buffers).
    ///
    /// - Parameters:
    ///   - program: A compiled CoreMLProgram.
    ///   - multiArrayInputs: Named MLMultiArray inputs.
    /// - Returns: Output as flat Float32 array.
    public func execute(
        _ program: CoreMLProgram,
        multiArrayInputs: [(name: String, array: MLMultiArray)]
    ) throws -> [Float] {
        guard program.isValid else {
            throw ANEError.executionFailed("CoreML program has been released")
        }

        var featureDict: [String: MLFeatureValue] = [:]
        for input in multiArrayInputs {
            featureDict[input.name] = MLFeatureValue(multiArray: input.array)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: featureDict)
        let prediction = try ANEWatchdog.withTimeout {
            try program.model.prediction(from: provider)
        }

        guard let outputName = prediction.featureNames.first,
              let outputValue = prediction.featureValue(for: outputName),
              let outputArray = outputValue.multiArrayValue else {
            throw ANEError.executionFailed("No output from CoreML prediction")
        }

        let count = outputArray.count
        var result = [Float](repeating: 0, count: count)
        let outPtr = outputArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            result[i] = outPtr[i]
        }
        return result
    }

    // MARK: - .mlpackage writing

    private func writeMLPackage(
        id: String,
        modelData: Data,
        weightsData: Data?
    ) throws -> URL {
        let fm = FileManager.default

        // Create cache directory
        try fm.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

        let packagePath = cacheDirectory.appendingPathComponent("\(id).mlpackage")
        let dataPath = packagePath.appendingPathComponent("Data/com.apple.CoreML")

        try fm.createDirectory(at: dataPath, withIntermediateDirectories: true)

        // Write model spec
        try modelData.write(to: dataPath.appendingPathComponent("model.mlmodel"))

        // Write weights if present
        if let weights = weightsData {
            let weightsDir = dataPath.appendingPathComponent("weights")
            try fm.createDirectory(at: weightsDir, withIntermediateDirectories: true)
            try weights.write(to: weightsDir.appendingPathComponent("weight.bin"))
        }

        // Write Manifest.json
        // Each item is keyed by a UUID. rootModelIdentifier points to the model spec.
        let modelUUID = UUID().uuidString
        let weightsUUID = UUID().uuidString
        var itemInfoEntries: [String: Any] = [
            modelUUID: [
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel"
            ] as [String: Any]
        ]
        if weightsData != nil {
            itemInfoEntries[weightsUUID] = [
                "author": "com.apple.CoreML",
                "description": "CoreML Model Weights",
                "name": "weights",
                "path": "com.apple.CoreML/weights"
            ] as [String: Any]
        }
        let manifest: [String: Any] = [
            "fileFormatVersion": "1.0.0",
            "itemInfoEntries": itemInfoEntries,
            "rootModelIdentifier": modelUUID
        ]
        let manifestData = try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys]
        )
        try manifestData.write(to: packagePath.appendingPathComponent("Manifest.json"))

        return packagePath
    }

    /// Removes all cached .mlpackage and .mlmodelc files.
    public func clearCache() {
        try? FileManager.default.removeItem(at: cacheDirectory)
    }
}
