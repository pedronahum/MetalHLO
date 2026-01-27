// PermutationUtils.swift
// MetalHLOCore
//
// Utilities for building and manipulating dimension permutations.
// Used by gather/scatter batching to transpose inputs/outputs.

import Foundation

/// Utilities for building and manipulating dimension permutations.
public enum PermutationUtils {

    /// Build permutation that moves specified dimensions to the front.
    ///
    /// Example: `moveToFront([2, 4], rank: 5)` -> `[2, 4, 0, 1, 3]`
    ///
    /// - Parameters:
    ///   - dims: Dimensions to move to front (will be sorted)
    ///   - rank: Total number of dimensions
    /// - Returns: Permutation array where result[i] = source dimension
    public static func buildPermutationMovingToFront(dims: [Int], rank: Int) -> [Int] {
        guard !dims.isEmpty else {
            return Array(0..<rank)
        }

        let sortedDims = dims.sorted()
        var perm: [Int] = []

        // First: the dims we want at front (in sorted order)
        perm.append(contentsOf: sortedDims)

        // Then: all other dims in their original order
        for i in 0..<rank {
            if !sortedDims.contains(i) {
                perm.append(i)
            }
        }

        return perm
    }

    /// Compute the inverse of a permutation.
    ///
    /// If `perm[i] = j`, then `inverse[j] = i`
    ///
    /// - Parameter perm: The permutation to invert
    /// - Returns: The inverse permutation
    public static func invertPermutation(_ perm: [Int]) -> [Int] {
        var inverse = [Int](repeating: 0, count: perm.count)
        for (newPos, oldPos) in perm.enumerated() {
            inverse[oldPos] = newPos
        }
        return inverse
    }

    /// Compose two permutations: `result[i] = perm2[perm1[i]]`
    ///
    /// Applying `result` is equivalent to applying `perm1` then `perm2`.
    public static func compose(_ perm1: [Int], _ perm2: [Int]) -> [Int] {
        precondition(perm1.count == perm2.count, "Permutations must have same length")
        return perm1.map { perm2[$0] }
    }

    /// Check if the specified dimensions are contiguous starting at position 0.
    ///
    /// Example: `[0, 1, 2]` with rank 5 -> true
    /// Example: `[1, 2]` with rank 5 -> false (doesn't start at 0)
    /// Example: `[0, 2]` with rank 5 -> false (not contiguous)
    public static func areDimsContiguousAtFront(_ dims: [Int], rank: Int) -> Bool {
        guard !dims.isEmpty else { return true }
        let sorted = dims.sorted()
        return sorted == Array(0..<dims.count)
    }

    /// Check if a permutation is the identity (no-op).
    public static func isIdentity(_ perm: [Int]) -> Bool {
        return perm == Array(0..<perm.count)
    }

    /// Apply a permutation to an array of values.
    ///
    /// `result[i] = values[perm[i]]`
    public static func applyPermutation<T>(_ perm: [Int], to values: [T]) -> [T] {
        precondition(perm.count == values.count, "Permutation and values must have same length")
        return perm.map { values[$0] }
    }

    /// Map dimension indices through a permutation.
    ///
    /// Given dims in the original space, returns their positions after permutation.
    public static func mapDimsThroughPermutation(_ dims: [Int], perm: [Int]) -> [Int] {
        let inverse = invertPermutation(perm)
        return dims.map { inverse[$0] }
    }
}
