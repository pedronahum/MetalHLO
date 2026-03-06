// MILWeightTemplate.swift
// ANERuntime
//
// A pre-compiled CoreML model where weight data can be swapped without
// recompilation. Avoids hitting the ANE ~115-compilation limit during
// training by compiling once and patching weight.bin on subsequent
// iterations. MLModel(contentsOf:) reloads the model without invoking
// the ANE compiler.

import Foundation
import CoreML

/// A pre-compiled CoreML model template that supports weight swapping.
///
/// During training, weights change every iteration. Without templates,
/// each weight change requires a full `MLModel.compileModel()` call,
/// which counts toward the ~115-compilation-per-process limit. With
/// templates, we compile once, then patch the weight blob in the
/// compiled `.mlmodelc` and reload via `MLModel(contentsOf:)`.
public final class MILWeightTemplate: @unchecked Sendable {

    /// Path to the compiled .mlmodelc directory.
    public let compiledModelPath: URL

    /// Path to the source .mlpackage (for locating weight.bin).
    public let sourcePackagePath: URL

    /// Layout of named weights within the blob: name → (offset, size).
    public let weightLayout: [String: (offset: Int, size: Int)]

    /// Input names in function signature order.
    public let inputNames: [String]

    /// Input shapes in function signature order.
    public let inputShapes: [[Int]]

    /// Output shape.
    public let outputShape: [Int]

    /// The loaded CoreML model, reloaded after each weight swap.
    private var model: MLModel

    /// Path to weight.bin in the source .mlpackage.
    private let sourceWeightBinPath: URL

    /// Path to weight.bin in the compiled .mlmodelc (if present).
    private let compiledWeightBinPath: URL?

    public init(
        compiledModelPath: URL,
        sourcePackagePath: URL,
        model: MLModel,
        weightLayout: [String: (offset: Int, size: Int)],
        inputNames: [String],
        inputShapes: [[Int]],
        outputShape: [Int]
    ) {
        self.compiledModelPath = compiledModelPath
        self.sourcePackagePath = sourcePackagePath
        self.model = model
        self.weightLayout = weightLayout
        self.inputNames = inputNames
        self.inputShapes = inputShapes
        self.outputShape = outputShape

        // Source weight.bin location
        self.sourceWeightBinPath = sourcePackagePath
            .appendingPathComponent("Data")
            .appendingPathComponent("com.apple.CoreML")
            .appendingPathComponent("weights")
            .appendingPathComponent("weight.bin")

        // Compiled weight.bin location (CoreML may copy it here)
        let candidatePath = compiledModelPath.appendingPathComponent("weight.bin")
        self.compiledWeightBinPath = FileManager.default.fileExists(atPath: candidatePath.path)
            ? candidatePath
            : nil
    }

    /// The current loaded MLModel for execution.
    public var currentModel: MLModel { model }

    /// Swaps weight data and reloads the model without recompilation.
    ///
    /// This does NOT call `MLModel.compileModel()` and does NOT
    /// increment the CompileCounter.
    ///
    /// - Parameter newWeights: Map of weight name → new FP16 data blob.
    /// - Throws: If weight patching or model reload fails.
    public func updateWeights(_ newWeights: [String: Data]) throws {
        // Patch weight.bin in the source package
        try patchWeightBin(at: sourceWeightBinPath, with: newWeights)

        // Also patch in compiled model if weight.bin was copied there
        if let compiledPath = compiledWeightBinPath {
            try patchWeightBin(at: compiledPath, with: newWeights)
        }

        // Reload the model from the compiled path (no recompilation)
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try MLModel(contentsOf: compiledModelPath, configuration: config)
    }

    /// Patches specific weight entries in a weight.bin file.
    private func patchWeightBin(at path: URL, with newWeights: [String: Data]) throws {
        guard FileManager.default.fileExists(atPath: path.path) else { return }

        var blobData = try Data(contentsOf: path)

        for (name, newData) in newWeights {
            guard let entry = weightLayout[name] else { continue }
            guard newData.count == entry.size else {
                throw MILWeightTemplateError.sizeMismatch(
                    name: name,
                    expected: entry.size,
                    got: newData.count
                )
            }
            blobData.replaceSubrange(entry.offset..<(entry.offset + entry.size), with: newData)
        }

        try blobData.write(to: path)
    }
}

/// Errors during weight template operations.
public enum MILWeightTemplateError: Error, CustomStringConvertible {
    case sizeMismatch(name: String, expected: Int, got: Int)
    case weightBinNotFound(URL)

    public var description: String {
        switch self {
        case .sizeMismatch(let name, let expected, let got):
            return "Weight '\(name)' size mismatch: expected \(expected) bytes, got \(got)"
        case .weightBinNotFound(let url):
            return "weight.bin not found at \(url.path)"
        }
    }
}
