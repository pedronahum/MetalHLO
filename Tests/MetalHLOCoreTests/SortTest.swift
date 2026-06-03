// SortTest.swift
// Exercises the fast-path stable sort (jnp.sort) and the multi-operand
// argsort/lexsort split (sortResult) through the full compile→execute path.
//
// JAX lowers these to `stablehlo.sort` with a comparator region whose final
// ordering compare (LT ascending / GT descending) the parser reads for
// direction; the kernel ranks each element within its row (stable j<i tiebreak)
// and scatters. End-to-end correctness vs JAX CPU is checked separately; these
// guard the parser + kernel against regressions.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Sort", .serialized)
struct SortTests {

    // jnp.sort lowering: single operand, ascending (comparator ends in LT).
    @Test("Single-operand ascending sort")
    func sortAscending() async throws {
        let mlir = """
        module @sort_test {
          func.func @main(%arg0: tensor<6xf32>) -> (tensor<6xf32>) {
            %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
            ^bb0(%a: tensor<f32>, %b: tensor<f32>):
              %c = stablehlo.compare LT, %a, %b, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
              stablehlo.return %c : tensor<i1>
            }) : (tensor<6xf32>) -> tensor<6xf32>
            return %0 : tensor<6xf32>
          }
        }
        """
        let client = try Client.create()
        let exe = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        let x = try client.createBuffer([5, 3, 8, 1, 3, 9] as [Float], shape: [6], elementType: .float32)
        let out = try exe.execute([x])
        #expect(try out[0].toFloatArray() == [1, 3, 3, 5, 8, 9])
    }

    // Descending: comparator ends in GT.
    @Test("Single-operand descending sort")
    func sortDescending() async throws {
        let mlir = """
        module @sort_desc {
          func.func @main(%arg0: tensor<5xf32>) -> (tensor<5xf32>) {
            %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = true}> ({
            ^bb0(%a: tensor<f32>, %b: tensor<f32>):
              %c = stablehlo.compare GT, %a, %b, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
              stablehlo.return %c : tensor<i1>
            }) : (tensor<5xf32>) -> tensor<5xf32>
            return %0 : tensor<5xf32>
          }
        }
        """
        let client = try Client.create()
        let exe = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        let x = try client.createBuffer([5, 3, 8, 1, 9] as [Float], shape: [5], elementType: .float32)
        let out = try exe.execute([x])
        #expect(try out[0].toFloatArray() == [9, 8, 5, 3, 1])
    }

    // 2-result sort (argsort shape): sort (key, iota); result #1 is the
    // permutation. Stable: equal keys keep ascending index order.
    @Test("argsort: 2-operand sort returns a stable permutation")
    func argsort() async throws {
        let mlir = """
        module @argsort_test {
          func.func @main(%arg0: tensor<5xf32>) -> (tensor<5xi32>) {
            %iota = stablehlo.iota dim = 0 : tensor<5xi32>
            %r:2 = "stablehlo.sort"(%arg0, %iota) <{dimension = 0 : i64, is_stable = true}> ({
            ^bb0(%a: tensor<f32>, %b: tensor<f32>, %ia: tensor<i32>, %ib: tensor<i32>):
              %c = stablehlo.compare LT, %a, %b, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
              stablehlo.return %c : tensor<i1>
            }) : (tensor<5xf32>, tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>)
            return %r#1 : tensor<5xi32>
          }
        }
        """
        let client = try Client.create()
        let exe = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
        // values [2,1,2,1,3] -> stable argsort [1,3,0,2,4]
        let x = try client.createBuffer([2, 1, 2, 1, 3] as [Float], shape: [5], elementType: .float32)
        let out = try exe.execute([x])
        let idx = try out[0].toInt32Array()
        #expect(idx == [1, 3, 0, 2, 4], "argsort wrong: \(idx)")
    }
}
