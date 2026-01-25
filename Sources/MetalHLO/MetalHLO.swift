// MetalHLO.swift
// MetalHLO
//
// Public module exports for MetalHLO.

// Re-export Foundation.Data for buffer operations
import Foundation
@_exported import struct Foundation.Data

// Version information
public enum MetalHLO {
    /// The MetalHLO library version.
    public static let version = "0.1.0"

    /// The StableHLO specification version supported.
    public static let stableHLOVersion = "1.0"
}
