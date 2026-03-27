// HeterogeneousError.swift
// HeterogeneousFusion

import Foundation

public enum HeterogeneousError: Error, LocalizedError {
    case metalSetupFailed(String)
    case bufferAllocationFailed(String)
    case executionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .metalSetupFailed(let msg): return "Metal setup failed: \(msg)"
        case .bufferAllocationFailed(let msg): return "Buffer allocation failed: \(msg)"
        case .executionFailed(let msg): return "Execution failed: \(msg)"
        }
    }
}
