// CoreMLProgram.swift
// ANERuntime
//
// Compiled CoreML model wrapper, analogous to ANEProgram.

import Foundation
import CoreML

/// A compiled CoreML model ready for execution via MLModel.prediction().
public final class CoreMLProgram: @unchecked Sendable {

    /// Unique identifier for this program.
    public let id: String

    /// The loaded CoreML model.
    internal let model: MLModel

    /// Input names in function signature order.
    public let inputNames: [String]

    /// Input shapes in function signature order.
    public let inputShapes: [[Int]]

    /// Output shape.
    public let outputShape: [Int]

    /// Path to the compiled .mlmodelc on disk.
    public let compiledModelPath: URL

    /// Path to the source .mlpackage (for cleanup).
    public let sourcePackagePath: URL

    private var _isValid = true

    public var isValid: Bool { _isValid }

    internal init(
        id: String,
        model: MLModel,
        inputNames: [String],
        inputShapes: [[Int]],
        outputShape: [Int],
        compiledModelPath: URL,
        sourcePackagePath: URL
    ) {
        self.id = id
        self.model = model
        self.inputNames = inputNames
        self.inputShapes = inputShapes
        self.outputShape = outputShape
        self.compiledModelPath = compiledModelPath
        self.sourcePackagePath = sourcePackagePath
    }

    /// Releases the model and optionally removes compiled files from disk.
    public func release(removeFromDisk: Bool = false) {
        _isValid = false
        if removeFromDisk {
            try? FileManager.default.removeItem(at: compiledModelPath)
            try? FileManager.default.removeItem(at: sourcePackagePath)
        }
    }
}
