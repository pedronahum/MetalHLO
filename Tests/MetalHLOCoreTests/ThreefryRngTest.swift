// ThreefryRngTest.swift
// MetalHLOCoreTests
//
// Verifies that jax.random's threefry2x32 lowering — a counted `while` loop
// around a func.call, which the WhileLoopUnroller flattens for the fast path —
// produces bit-identical output to JAX CPU for a fixed key.
//
// The MLIR below is exactly what `jax.jit(lambda k: jax.random.bits(k,
// shape=(8,), dtype=uint32)).lower(...).as_text()` emits on JAX 0.10. The
// reference bits were captured from `jax.random.bits` on CPU for key=[0, 0]:
//   [4070199207, 4202968722, 1427181096, 2012915765,
//     2447653815,  710830403, 1332275837, 2961296638]

import Testing
import Foundation
@testable import MetalHLO

@Suite("Threefry RNG", .serialized)
struct ThreefryRngTests {

    /// jax.random.bits(key, (8,), uint32) lowering from JAX 0.10.
    static let threefryMLIR = """
    module @jit_F {
      func.func public @main(%arg0: tensor<2xui32>) -> (tensor<8xui32>) {
        %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %1 = stablehlo.reshape %0 : (tensor<1xui32>) -> tensor<ui32>
        %2 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %3 = stablehlo.reshape %2 : (tensor<1xui32>) -> tensor<ui32>
        %4 = stablehlo.iota dim = 0 : tensor<8xui64>
        %c = stablehlo.constant dense<1> : tensor<ui64>
        %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<8xui64>
        %6 = stablehlo.multiply %5, %4 : tensor<8xui64>
        %c_0 = stablehlo.constant dense<32> : tensor<ui64>
        %7 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui64>) -> tensor<8xui64>
        %8 = stablehlo.shift_right_logical %6, %7 : tensor<8xui64>
        %9 = stablehlo.convert %6 : (tensor<8xui64>) -> tensor<8xui32>
        %10 = stablehlo.convert %8 : (tensor<8xui64>) -> tensor<8xui32>
        %11:2 = call @threefry2x32(%1, %3, %10, %9) : (tensor<ui32>, tensor<ui32>, tensor<8xui32>, tensor<8xui32>) -> (tensor<8xui32>, tensor<8xui32>)
        %12 = stablehlo.xor %11#0, %11#1 : tensor<8xui32>
        return %12 : tensor<8xui32>
      }
      func.func private @threefry2x32(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<8xui32>, %arg3: tensor<8xui32>) -> (tensor<8xui32>, tensor<8xui32>) {
        %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
        %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
        %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
        %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
        %1 = stablehlo.xor %0, %c_1 : tensor<ui32>
        %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %3 = stablehlo.add %arg2, %2 : tensor<8xui32>
        %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %5 = stablehlo.add %arg3, %4 : tensor<8xui32>
        %c_2 = stablehlo.constant dense<0> : tensor<i32>
        %c_3 = stablehlo.constant dense<0> : tensor<i32>
        %6:9 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %c_2, %iterArg_5 = %3, %iterArg_6 = %5, %iterArg_7 = %arg1, %iterArg_8 = %1, %iterArg_9 = %arg0, %iterArg_10 = %c, %iterArg_11 = %c_0) : tensor<i32>, tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
        cond {
          %c_12 = stablehlo.constant dense<5> : tensor<i32>
          %7 = stablehlo.compare LT, %iterArg, %c_12, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
          stablehlo.return %7 : tensor<i1>
        } do {
          %7:8 = func.call @closed_call(%iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11) : (tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
          %c_12 = stablehlo.constant dense<1> : tensor<i32>
          %8 = stablehlo.add %iterArg, %c_12 : tensor<i32>
          stablehlo.return %8, %7#0, %7#1, %7#2, %7#3, %7#4, %7#5, %7#6, %7#7 : tensor<i32>, tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
        }
        return %6#2, %6#3 : tensor<8xui32>, tensor<8xui32>
      }
      func.func private @closed_call(%arg0: tensor<i32>, %arg1: tensor<8xui32>, %arg2: tensor<8xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
        %c = stablehlo.constant dense<1> : tensor<i32>
        %0 = stablehlo.add %arg0, %c : tensor<i32>
        %1 = stablehlo.slice %arg6 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
        %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
        %3 = stablehlo.add %arg1, %arg2 : tensor<8xui32>
        %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %5 = stablehlo.shift_left %arg2, %4 : tensor<8xui32>
        %c_0 = stablehlo.constant dense<32> : tensor<ui32>
        %6 = stablehlo.subtract %c_0, %2 : tensor<ui32>
        %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %8 = stablehlo.shift_right_logical %arg2, %7 : tensor<8xui32>
        %9 = stablehlo.or %5, %8 : tensor<8xui32>
        %10 = stablehlo.xor %3, %9 : tensor<8xui32>
        %11 = stablehlo.slice %arg6 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
        %12 = stablehlo.reshape %11 : (tensor<1xui32>) -> tensor<ui32>
        %13 = stablehlo.add %3, %10 : tensor<8xui32>
        %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %15 = stablehlo.shift_left %10, %14 : tensor<8xui32>
        %c_1 = stablehlo.constant dense<32> : tensor<ui32>
        %16 = stablehlo.subtract %c_1, %12 : tensor<ui32>
        %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %18 = stablehlo.shift_right_logical %10, %17 : tensor<8xui32>
        %19 = stablehlo.or %15, %18 : tensor<8xui32>
        %20 = stablehlo.xor %13, %19 : tensor<8xui32>
        %21 = stablehlo.slice %arg6 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
        %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
        %23 = stablehlo.add %13, %20 : tensor<8xui32>
        %24 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %25 = stablehlo.shift_left %20, %24 : tensor<8xui32>
        %c_2 = stablehlo.constant dense<32> : tensor<ui32>
        %26 = stablehlo.subtract %c_2, %22 : tensor<ui32>
        %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %28 = stablehlo.shift_right_logical %20, %27 : tensor<8xui32>
        %29 = stablehlo.or %25, %28 : tensor<8xui32>
        %30 = stablehlo.xor %23, %29 : tensor<8xui32>
        %31 = stablehlo.slice %arg6 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
        %32 = stablehlo.reshape %31 : (tensor<1xui32>) -> tensor<ui32>
        %33 = stablehlo.add %23, %30 : tensor<8xui32>
        %34 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %35 = stablehlo.shift_left %30, %34 : tensor<8xui32>
        %c_3 = stablehlo.constant dense<32> : tensor<ui32>
        %36 = stablehlo.subtract %c_3, %32 : tensor<ui32>
        %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %38 = stablehlo.shift_right_logical %30, %37 : tensor<8xui32>
        %39 = stablehlo.or %35, %38 : tensor<8xui32>
        %40 = stablehlo.xor %33, %39 : tensor<8xui32>
        %41 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %42 = stablehlo.add %33, %41 : tensor<8xui32>
        %43 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %44 = stablehlo.add %40, %43 : tensor<8xui32>
        %c_4 = stablehlo.constant dense<1> : tensor<i32>
        %45 = stablehlo.add %arg0, %c_4 : tensor<i32>
        %46 = stablehlo.convert %45 : (tensor<i32>) -> tensor<ui32>
        %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<8xui32>
        %48 = stablehlo.add %44, %47 : tensor<8xui32>
        return %0, %42, %48, %arg4, %arg5, %arg3, %arg7, %arg6 : tensor<i32>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
      }
    }
    """

    @Test("threefry2x32 matches JAX CPU bit-for-bit (key=[0,0])")
    func threefryBitExact() throws {
        let client = try Client.create()
        let executable = try client.compile(Self.threefryMLIR)

        let key = try client.createBuffer([UInt32(0), UInt32(0)], shape: [2], elementType: .uint32)
        let outputs = try executable.execute([key])
        let result = try outputs[0].toUInt32Array()

        // jax.random.bits(jnp.array([0,0], uint32), (8,), uint32) on CPU.
        let expected: [UInt32] = [
            4070199207, 4202968722, 1427181096, 2012915765,
            2447653815, 710830403, 1332275837, 2961296638,
        ]
        #expect(result == expected, "threefry bits \(result) != JAX \(expected)")
    }
}
