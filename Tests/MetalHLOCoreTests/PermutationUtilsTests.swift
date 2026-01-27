// PermutationUtilsTests.swift
// MetalHLOCoreTests
//
// Tests for PermutationUtils dimension permutation utilities.

import Testing
@testable import MetalHLOCore

@Suite("PermutationUtils Tests")
struct PermutationUtilsTests {

    // MARK: - buildPermutationMovingToFront Tests

    @Test("Move single dim to front")
    func testMoveToFrontSingleDim() {
        let perm = PermutationUtils.buildPermutationMovingToFront(dims: [2], rank: 4)
        #expect(perm == [2, 0, 1, 3])
    }

    @Test("Move multiple dims to front")
    func testMoveToFrontMultipleDims() {
        let perm = PermutationUtils.buildPermutationMovingToFront(dims: [1, 3], rank: 5)
        #expect(perm == [1, 3, 0, 2, 4])
    }

    @Test("Move dims already at front")
    func testMoveToFrontAlreadyAtFront() {
        let perm = PermutationUtils.buildPermutationMovingToFront(dims: [0, 1], rank: 4)
        #expect(perm == [0, 1, 2, 3])
    }

    @Test("Move empty dims")
    func testMoveToFrontEmpty() {
        let perm = PermutationUtils.buildPermutationMovingToFront(dims: [], rank: 4)
        #expect(perm == [0, 1, 2, 3])
    }

    @Test("Move last dim to front")
    func testMoveToFrontLastDim() {
        let perm = PermutationUtils.buildPermutationMovingToFront(dims: [3], rank: 4)
        #expect(perm == [3, 0, 1, 2])
    }

    @Test("Move unsorted dims to front")
    func testMoveToFrontUnsortedDims() {
        // [4, 2] should become [2, 4, ...] (sorted)
        let perm = PermutationUtils.buildPermutationMovingToFront(dims: [4, 2], rank: 5)
        #expect(perm == [2, 4, 0, 1, 3])
    }

    // MARK: - invertPermutation Tests

    @Test("Invert permutation")
    func testInvertPermutation() {
        let perm = [2, 0, 1, 3]
        let inverse = PermutationUtils.invertPermutation(perm)
        #expect(inverse == [1, 2, 0, 3])

        // Verify: applying perm then inverse gives identity
        let composed = PermutationUtils.compose(perm, inverse)
        #expect(composed == [0, 1, 2, 3])
    }

    @Test("Invert identity permutation")
    func testInvertIdentity() {
        let perm = [0, 1, 2, 3]
        let inverse = PermutationUtils.invertPermutation(perm)
        #expect(inverse == [0, 1, 2, 3])
    }

    @Test("Invert reverse permutation")
    func testInvertReverse() {
        let perm = [3, 2, 1, 0]
        let inverse = PermutationUtils.invertPermutation(perm)
        #expect(inverse == [3, 2, 1, 0])  // Reverse is self-inverse
    }

    // MARK: - compose Tests

    @Test("Compose permutations")
    func testCompose() {
        let perm1 = [1, 0, 2]  // swap first two
        let perm2 = [0, 2, 1]  // swap last two
        let composed = PermutationUtils.compose(perm1, perm2)
        // First apply perm1: [1, 0, 2], then apply perm2 to result
        // perm1[0]=1, perm2[1]=2, so composed[0]=2
        // perm1[1]=0, perm2[0]=0, so composed[1]=0
        // perm1[2]=2, perm2[2]=1, so composed[2]=1
        #expect(composed == [2, 0, 1])
    }

    // MARK: - areDimsContiguousAtFront Tests

    @Test("Dims contiguous at front")
    func testAreDimsContiguousAtFront() {
        #expect(PermutationUtils.areDimsContiguousAtFront([0, 1, 2], rank: 5) == true)
        #expect(PermutationUtils.areDimsContiguousAtFront([0], rank: 5) == true)
        #expect(PermutationUtils.areDimsContiguousAtFront([], rank: 5) == true)
    }

    @Test("Dims not contiguous at front")
    func testAreDimsNotContiguousAtFront() {
        #expect(PermutationUtils.areDimsContiguousAtFront([1, 2], rank: 5) == false)  // doesn't start at 0
        #expect(PermutationUtils.areDimsContiguousAtFront([0, 2], rank: 5) == false)  // gap
        #expect(PermutationUtils.areDimsContiguousAtFront([2], rank: 5) == false)     // not at front
    }

    @Test("Dims unsorted but contiguous at front")
    func testAreDimsUnsortedContiguous() {
        // [1, 0] should be treated as [0, 1] which is contiguous at front
        #expect(PermutationUtils.areDimsContiguousAtFront([1, 0], rank: 5) == true)
        #expect(PermutationUtils.areDimsContiguousAtFront([2, 0, 1], rank: 5) == true)
    }

    // MARK: - isIdentity Tests

    @Test("Is identity permutation")
    func testIsIdentity() {
        #expect(PermutationUtils.isIdentity([0, 1, 2, 3]) == true)
        #expect(PermutationUtils.isIdentity([0]) == true)
        #expect(PermutationUtils.isIdentity([]) == true)
    }

    @Test("Is not identity permutation")
    func testIsNotIdentity() {
        #expect(PermutationUtils.isIdentity([0, 2, 1, 3]) == false)
        #expect(PermutationUtils.isIdentity([1, 0, 2, 3]) == false)
        #expect(PermutationUtils.isIdentity([3, 2, 1, 0]) == false)
    }

    // MARK: - applyPermutation Tests

    @Test("Apply permutation to array")
    func testApplyPermutation() {
        let perm = [2, 0, 1, 3]
        let values = ["a", "b", "c", "d"]
        let result = PermutationUtils.applyPermutation(perm, to: values)
        #expect(result == ["c", "a", "b", "d"])
    }

    @Test("Apply identity permutation")
    func testApplyIdentityPermutation() {
        let perm = [0, 1, 2]
        let values = [10, 20, 30]
        let result = PermutationUtils.applyPermutation(perm, to: values)
        #expect(result == [10, 20, 30])
    }

    @Test("Apply reverse permutation")
    func testApplyReversePermutation() {
        let perm = [2, 1, 0]
        let values = [1, 2, 3]
        let result = PermutationUtils.applyPermutation(perm, to: values)
        #expect(result == [3, 2, 1])
    }

    // MARK: - mapDimsThroughPermutation Tests

    @Test("Map dims through permutation")
    func testMapDimsThroughPermutation() {
        // If perm = [2, 0, 1, 3], then:
        // - Original dim 0 is now at position 1
        // - Original dim 1 is now at position 2
        // - Original dim 2 is now at position 0
        // - Original dim 3 is now at position 3
        let perm = [2, 0, 1, 3]
        let dims = [0, 2]
        let mapped = PermutationUtils.mapDimsThroughPermutation(dims, perm: perm)
        #expect(mapped == [1, 0])  // dim 0 -> pos 1, dim 2 -> pos 0
    }

    @Test("Map dims through identity")
    func testMapDimsThroughIdentity() {
        let perm = [0, 1, 2, 3]
        let dims = [1, 3]
        let mapped = PermutationUtils.mapDimsThroughPermutation(dims, perm: perm)
        #expect(mapped == [1, 3])
    }

    // MARK: - Round-trip Tests

    @Test("Permutation round-trip")
    func testPermutationRoundTrip() {
        // Moving dims to front and then inverting should give back original
        let originalDims = [2, 4]
        let rank = 5
        let perm = PermutationUtils.buildPermutationMovingToFront(dims: originalDims, rank: rank)
        let inverse = PermutationUtils.invertPermutation(perm)

        // Applying perm then inverse should be identity
        let roundTrip = PermutationUtils.compose(perm, inverse)
        #expect(roundTrip == [0, 1, 2, 3, 4])

        // Also verify compose in reverse order
        let roundTrip2 = PermutationUtils.compose(inverse, perm)
        #expect(roundTrip2 == [0, 1, 2, 3, 4])
    }

    @Test("Apply and map consistency")
    func testApplyAndMapConsistency() {
        // If we apply a permutation to shapes, the new positions should match mapDims
        let perm = [2, 0, 1, 3]
        let shape = [10, 20, 30, 40]
        let permutedShape = PermutationUtils.applyPermutation(perm, to: shape)

        // Original dim 2 (value 30) should now be at front
        #expect(permutedShape[0] == 30)

        // mapDims should tell us where original dims end up
        let mappedPos = PermutationUtils.mapDimsThroughPermutation([2], perm: perm)
        #expect(mappedPos == [0])  // dim 2 is now at position 0
    }
}
