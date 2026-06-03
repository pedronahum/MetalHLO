// SearchSortedTest.swift
// Tests jnp.searchsorted, which JAX lowers as a counted-`while` binary search
// (@_searchsorted_scan_impl) over a nested func.call chain. Exercises three
// things end-to-end: the WhileLoopUnroller flattening the bounded loop, the
// FunctionInliner carrying enclosing-scope renames into the while regions (the
// bound/step constants are hoisted above the loop), and the fused compare/select
// chain reading its float operands at the chain compute type (the final compare
// produces an i1 output, but its data inputs must not be read as `bool`).

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("SearchSorted Tests", .serialized)
struct SearchSortedTests {

    /// jnp.searchsorted([1,3,5,7,9], v) lowering from JAX 0.10. The induction
    /// counter (slot 2, `%iterArg_5`), its LT bound (`dense<3>`) and its `add`
    /// step (`dense<1>`) are hoisted as function-level constants ahead of the
    /// `while` — exactly the shape the PJRT text transforms produce — so the
    /// inliner must move the region references onto the renamed defs for the
    /// unroller to see a static trip count.
    static let searchSortedMLIR = """
    module @jit_F {
      func.func public @main(%arg0: tensor<5xf32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>) {
        %0 = call @searchsorted(%arg0, %arg1) : (tensor<5xf32>, tensor<4xf32>) -> tensor<4xi32>
        return %0 : tensor<4xi32>
      }
      func.func private @searchsorted(%arg0: tensor<5xf32>, %arg1: tensor<4xf32>) -> tensor<4xi32> {
        %0 = call @_searchsorted_scan_impl(%arg0, %arg1) : (tensor<5xf32>, tensor<4xf32>) -> tensor<4xi32>
        return %0 : tensor<4xi32>
      }
      func.func private @_searchsorted_scan_impl(%arg0: tensor<5xf32>, %arg1: tensor<4xf32>) -> tensor<4xi32> {
        %0 = call @_searchsorted_scan_impl_0(%arg0, %arg1) : (tensor<5xf32>, tensor<4xf32>) -> tensor<4xi32>
        return %0 : tensor<4xi32>
      }
      func.func private @_searchsorted_scan_impl_0(%arg0: tensor<5xf32>, %arg1: tensor<4xf32>) -> tensor<4xi32> {
        %c = stablehlo.constant dense<1> : tensor<i32>
        %c_0 = stablehlo.constant dense<3> : tensor<i32>
        %c_1 = stablehlo.constant dense<0> : tensor<i32>
        %c_2 = stablehlo.constant dense<5> : tensor<ui32>
        %c_3 = stablehlo.constant dense<0> : tensor<ui32>
        %0 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<4xui32>
        %1 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<4xui32>
        %2:5 = stablehlo.while(%iterArg = %arg0, %iterArg_4 = %arg1, %iterArg_5 = %c_1, %iterArg_6 = %0, %iterArg_7 = %1) : tensor<5xf32>, tensor<4xf32>, tensor<i32>, tensor<4xui32>, tensor<4xui32>
        cond {
          %4 = stablehlo.compare LT, %iterArg_5, %c_0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
          stablehlo.return %4 : tensor<i1>
        } do {
          %4:2 = func.call @closed_call(%iterArg, %iterArg_4, %iterArg_6, %iterArg_7) : (tensor<5xf32>, tensor<4xf32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<4xui32>, tensor<4xui32>)
          %5 = stablehlo.add %iterArg_5, %c : tensor<i32>
          stablehlo.return %iterArg, %iterArg_4, %5, %4#0, %4#1 : tensor<5xf32>, tensor<4xf32>, tensor<i32>, tensor<4xui32>, tensor<4xui32>
        }
        %3 = stablehlo.convert %2#4 : (tensor<4xui32>) -> tensor<4xi32>
        return %3 : tensor<4xi32>
      }
      func.func private @closed_call(%arg0: tensor<5xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xui32>, %arg3: tensor<4xui32>) -> (tensor<4xui32>, tensor<4xui32>) {
        %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
        %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %c = stablehlo.constant dense<2> : tensor<i32>
        %0 = stablehlo.subtract %arg3, %arg2 : tensor<4xui32>
        %1 = call @floor_divide(%0, %c) : (tensor<4xui32>, tensor<i32>) -> tensor<4xui32>
        %2 = stablehlo.add %arg2, %1 : tensor<4xui32>
        %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<4xui32>) -> tensor<4x1xui32>
        %4 = stablehlo.gather %arg0, %3, dimension_numbers = {offset_dims = [1], start_index_map = [0], index_vector_dim = 1}, slice_sizes = [1] : (tensor<5xf32>, tensor<4x1xui32>) -> tensor<4x1xf32>
        %5 = stablehlo.reshape %4 : (tensor<4x1xf32>) -> tensor<4xf32>
        %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4xf32>
        %7 = stablehlo.compare EQ, %arg1, %6, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4xf32>
        %9 = stablehlo.select %7, %8, %arg1 : tensor<4xi1>, tensor<4xf32>
        %10 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
        %12 = stablehlo.select %10, %11, %9 : tensor<4xi1>, tensor<4xf32>
        %13 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4xf32>
        %14 = stablehlo.compare EQ, %5, %13, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4xf32>
        %16 = stablehlo.select %14, %15, %5 : tensor<4xi1>, tensor<4xf32>
        %17 = stablehlo.compare NE, %5, %5, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %18 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
        %19 = stablehlo.select %17, %18, %16 : tensor<4xi1>, tensor<4xf32>
        %20 = stablehlo.compare LE, %12, %19, TOTALORDER : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %21 = stablehlo.select %20, %arg2, %2 : tensor<4xi1>, tensor<4xui32>
        %22 = stablehlo.select %20, %2, %arg3 : tensor<4xi1>, tensor<4xui32>
        return %21, %22 : tensor<4xui32>, tensor<4xui32>
      }
      func.func private @floor_divide(%arg0: tensor<4xui32>, %arg1: tensor<i32>) -> tensor<4xui32> {
        %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<ui32>
        %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<ui32>) -> tensor<4xui32>
        %2 = stablehlo.divide %arg0, %1 : tensor<4xui32>
        return %2 : tensor<4xui32>
      }
    }
    """

    @Test("searchsorted binary search matches JAX CPU (side='left')")
    func testSearchSorted() async throws {
        let client = try Client.create()
        // .auto exercises the heterogeneous entry, which (like the live PJRT
        // plugin) inlines → unrolls → re-inlines before deciding placement.
        var config = CompilationConfig(optimizationLevel: .O2)
        config.devicePolicy = .auto
        let executable = try client.compile(Self.searchSortedMLIR, config: config)

        let sorted: [Float] = [1.0, 3.0, 5.0, 7.0, 9.0]
        let values: [Float] = [0.0, 2.0, 5.0, 8.0]
        // jnp.searchsorted([1,3,5,7,9], [0,2,5,8]) on JAX CPU:
        //   0 < 1               → 0
        //   1 < 2 ≤ 3           → 1
        //   3 < 5 ≤ 5           → 2
        //   7 < 8 ≤ 9           → 4
        let expected: [Int32] = [0, 1, 2, 4]

        let sortedBuffer = try client.createBuffer(sorted, shape: [5], elementType: .float32)
        let valuesBuffer = try client.createBuffer(values, shape: [4], elementType: .float32)

        let outputs = try executable.execute([sortedBuffer, valuesBuffer])
        let result = try outputs[0].toInt32Array()

        #expect(result == expected,
                "searchsorted = \(result), expected \(expected)")
    }
}
