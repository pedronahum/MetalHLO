// SpecialFunctionsTest.swift
// Tests chlo.lgamma / chlo.digamma (jax.lax.lgamma / digamma,
// jax.scipy.special.gammaln / digamma) on the fast CodeGenerator path.
//
// JAX never sends chlo.lgamma / chlo.digamma to the plugin verbatim: the
// portable-artifact serialization legalizes the chlo dialect into pure
// stablehlo before the bytecode reaches us. lgamma expands to the standard
// Lanczos approximation (the g=7, n=8 coefficient set: 676.520386,
// -1259.13916, 771.323425, ...) plus the reflection formula for x < 0.5;
// digamma expands to the matching recurrence + asymptotic series. Both
// therefore run entirely on existing elementwise/select ops. These tests pin
// the exact legalized programs JAX 0.10 emits and verify the numerics against
// scipy.special.gammaln / digamma so a regression in any of the underlying
// ops (log_plus_one, sine, select, is_finite, ...) is caught here.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Special Functions Tests", .serialized)
struct SpecialFunctionsTests {

    // Legalized stablehlo for jax.lax.lgamma over tensor<4xf32>, captured from
    // the deserialized portable artifact the plugin actually parses.
    static let lgammaMLIR = """
    module @jit_lgamma {
      func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
        %cst = stablehlo.constant dense<0x7F800000> : tensor<4xf32>
        %cst_0 = stablehlo.constant dense<1.14472985> : tensor<4xf32>
        %cst_1 = stablehlo.constant dense<3.14159274> : tensor<4xf32>
        %cst_2 = stablehlo.constant dense<0.918938517> : tensor<4xf32>
        %cst_3 = stablehlo.constant dense<2.01490307> : tensor<4xf32>
        %cst_4 = stablehlo.constant dense<7.500000e+00> : tensor<4xf32>
        %cst_5 = stablehlo.constant dense<8.000000e+00> : tensor<4xf32>
        %cst_6 = stablehlo.constant dense<1.50563267E-7> : tensor<4xf32>
        %cst_7 = stablehlo.constant dense<7.000000e+00> : tensor<4xf32>
        %cst_8 = stablehlo.constant dense<9.98436917E-6> : tensor<4xf32>
        %cst_9 = stablehlo.constant dense<6.000000e+00> : tensor<4xf32>
        %cst_10 = stablehlo.constant dense<-0.138571098> : tensor<4xf32>
        %cst_11 = stablehlo.constant dense<5.000000e+00> : tensor<4xf32>
        %cst_12 = stablehlo.constant dense<12.5073433> : tensor<4xf32>
        %cst_13 = stablehlo.constant dense<4.000000e+00> : tensor<4xf32>
        %cst_14 = stablehlo.constant dense<-176.615036> : tensor<4xf32>
        %cst_15 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
        %cst_16 = stablehlo.constant dense<771.323425> : tensor<4xf32>
        %cst_17 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
        %cst_18 = stablehlo.constant dense<-1259.13916> : tensor<4xf32>
        %cst_19 = stablehlo.constant dense<676.520386> : tensor<4xf32>
        %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
        %cst_21 = stablehlo.constant dense<5.000000e-01> : tensor<4xf32>
        %0 = stablehlo.compare LT, %arg0, %cst_21 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %1 = stablehlo.negate %arg0 : tensor<4xf32>
        %2 = stablehlo.subtract %arg0, %cst_20 : tensor<4xf32>
        %3 = stablehlo.select %0, %1, %2 : tensor<4xi1>, tensor<4xf32>
        %4 = stablehlo.add %3, %cst_20 : tensor<4xf32>
        %5 = stablehlo.divide %cst_19, %4 : tensor<4xf32>
        %6 = stablehlo.add %cst_20, %5 : tensor<4xf32>
        %7 = stablehlo.add %3, %cst_17 : tensor<4xf32>
        %8 = stablehlo.divide %cst_18, %7 : tensor<4xf32>
        %9 = stablehlo.add %6, %8 : tensor<4xf32>
        %10 = stablehlo.add %3, %cst_15 : tensor<4xf32>
        %11 = stablehlo.divide %cst_16, %10 : tensor<4xf32>
        %12 = stablehlo.add %9, %11 : tensor<4xf32>
        %13 = stablehlo.add %3, %cst_13 : tensor<4xf32>
        %14 = stablehlo.divide %cst_14, %13 : tensor<4xf32>
        %15 = stablehlo.add %12, %14 : tensor<4xf32>
        %16 = stablehlo.add %3, %cst_11 : tensor<4xf32>
        %17 = stablehlo.divide %cst_12, %16 : tensor<4xf32>
        %18 = stablehlo.add %15, %17 : tensor<4xf32>
        %19 = stablehlo.add %3, %cst_9 : tensor<4xf32>
        %20 = stablehlo.divide %cst_10, %19 : tensor<4xf32>
        %21 = stablehlo.add %18, %20 : tensor<4xf32>
        %22 = stablehlo.add %3, %cst_7 : tensor<4xf32>
        %23 = stablehlo.divide %cst_8, %22 : tensor<4xf32>
        %24 = stablehlo.add %21, %23 : tensor<4xf32>
        %25 = stablehlo.add %3, %cst_5 : tensor<4xf32>
        %26 = stablehlo.divide %cst_6, %25 : tensor<4xf32>
        %27 = stablehlo.add %24, %26 : tensor<4xf32>
        %28 = stablehlo.add %cst_4, %3 : tensor<4xf32>
        %29 = stablehlo.divide %3, %cst_4 : tensor<4xf32>
        %30 = stablehlo.log_plus_one %29 : tensor<4xf32>
        %31 = stablehlo.add %cst_3, %30 : tensor<4xf32>
        %32 = stablehlo.divide %28, %31 : tensor<4xf32>
        %33 = stablehlo.add %3, %cst_21 : tensor<4xf32>
        %34 = stablehlo.subtract %33, %32 : tensor<4xf32>
        %35 = stablehlo.multiply %34, %31 : tensor<4xf32>
        %36 = stablehlo.log %27 : tensor<4xf32>
        %37 = stablehlo.add %cst_2, %35 : tensor<4xf32>
        %38 = stablehlo.add %37, %36 : tensor<4xf32>
        %39 = stablehlo.abs %arg0 : tensor<4xf32>
        %40 = stablehlo.floor %39 : tensor<4xf32>
        %41 = stablehlo.subtract %39, %40 : tensor<4xf32>
        %42 = stablehlo.compare LT, %cst_21, %41 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %43 = stablehlo.subtract %cst_20, %41 : tensor<4xf32>
        %44 = stablehlo.select %42, %43, %41 : tensor<4xi1>, tensor<4xf32>
        %45 = stablehlo.multiply %cst_1, %44 : tensor<4xf32>
        %46 = stablehlo.sine %45 : tensor<4xf32>
        %47 = stablehlo.log %46 : tensor<4xf32>
        %48 = stablehlo.subtract %cst_0, %47 : tensor<4xf32>
        %49 = stablehlo.subtract %48, %38 : tensor<4xf32>
        %50 = stablehlo.is_finite %47 : (tensor<4xf32>) -> tensor<4xi1>
        %51 = stablehlo.negate %47 : tensor<4xf32>
        %52 = stablehlo.select %50, %49, %51 : tensor<4xi1>, tensor<4xf32>
        %53 = stablehlo.select %0, %52, %38 : tensor<4xi1>, tensor<4xf32>
        %54 = stablehlo.abs %arg0 : tensor<4xf32>
        %55 = stablehlo.compare EQ, %54, %cst : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %56 = stablehlo.select %55, %cst, %53 : tensor<4xi1>, tensor<4xf32>
        return %56 : tensor<4xf32>
      }
    }
    """

    // Legalized stablehlo for jax.lax.digamma over tensor<4xf32>.
    static let digammaMLIR = """
    module @jit_digamma {
      func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
        %cst = stablehlo.constant dense<0x7FC00000> : tensor<4xf32>
        %cst_0 = stablehlo.constant dense<3.14159274> : tensor<4xf32>
        %cst_1 = stablehlo.constant dense<2.01490307> : tensor<4xf32>
        %cst_2 = stablehlo.constant dense<7.500000e+00> : tensor<4xf32>
        %cst_3 = stablehlo.constant dense<8.000000e+00> : tensor<4xf32>
        %cst_4 = stablehlo.constant dense<1.50563267E-7> : tensor<4xf32>
        %cst_5 = stablehlo.constant dense<7.000000e+00> : tensor<4xf32>
        %cst_6 = stablehlo.constant dense<9.98436917E-6> : tensor<4xf32>
        %cst_7 = stablehlo.constant dense<6.000000e+00> : tensor<4xf32>
        %cst_8 = stablehlo.constant dense<-0.138571098> : tensor<4xf32>
        %cst_9 = stablehlo.constant dense<5.000000e+00> : tensor<4xf32>
        %cst_10 = stablehlo.constant dense<12.5073433> : tensor<4xf32>
        %cst_11 = stablehlo.constant dense<4.000000e+00> : tensor<4xf32>
        %cst_12 = stablehlo.constant dense<-176.615036> : tensor<4xf32>
        %cst_13 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
        %cst_14 = stablehlo.constant dense<771.323425> : tensor<4xf32>
        %cst_15 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
        %cst_16 = stablehlo.constant dense<-1259.13916> : tensor<4xf32>
        %cst_17 = stablehlo.constant dense<676.520386> : tensor<4xf32>
        %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
        %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
        %cst_20 = stablehlo.constant dense<5.000000e-01> : tensor<4xf32>
        %0 = stablehlo.compare LT, %arg0, %cst_20 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %1 = stablehlo.negate %arg0 : tensor<4xf32>
        %2 = stablehlo.subtract %arg0, %cst_19 : tensor<4xf32>
        %3 = stablehlo.select %0, %1, %2 : tensor<4xi1>, tensor<4xf32>
        %4 = stablehlo.add %3, %cst_19 : tensor<4xf32>
        %5 = stablehlo.multiply %4, %4 : tensor<4xf32>
        %6 = stablehlo.divide %cst_17, %5 : tensor<4xf32>
        %7 = stablehlo.subtract %cst_18, %6 : tensor<4xf32>
        %8 = stablehlo.divide %cst_17, %4 : tensor<4xf32>
        %9 = stablehlo.add %cst_19, %8 : tensor<4xf32>
        %10 = stablehlo.add %3, %cst_15 : tensor<4xf32>
        %11 = stablehlo.multiply %10, %10 : tensor<4xf32>
        %12 = stablehlo.divide %cst_16, %11 : tensor<4xf32>
        %13 = stablehlo.subtract %7, %12 : tensor<4xf32>
        %14 = stablehlo.divide %cst_16, %10 : tensor<4xf32>
        %15 = stablehlo.add %9, %14 : tensor<4xf32>
        %16 = stablehlo.add %3, %cst_13 : tensor<4xf32>
        %17 = stablehlo.multiply %16, %16 : tensor<4xf32>
        %18 = stablehlo.divide %cst_14, %17 : tensor<4xf32>
        %19 = stablehlo.subtract %13, %18 : tensor<4xf32>
        %20 = stablehlo.divide %cst_14, %16 : tensor<4xf32>
        %21 = stablehlo.add %15, %20 : tensor<4xf32>
        %22 = stablehlo.add %3, %cst_11 : tensor<4xf32>
        %23 = stablehlo.multiply %22, %22 : tensor<4xf32>
        %24 = stablehlo.divide %cst_12, %23 : tensor<4xf32>
        %25 = stablehlo.subtract %19, %24 : tensor<4xf32>
        %26 = stablehlo.divide %cst_12, %22 : tensor<4xf32>
        %27 = stablehlo.add %21, %26 : tensor<4xf32>
        %28 = stablehlo.add %3, %cst_9 : tensor<4xf32>
        %29 = stablehlo.multiply %28, %28 : tensor<4xf32>
        %30 = stablehlo.divide %cst_10, %29 : tensor<4xf32>
        %31 = stablehlo.subtract %25, %30 : tensor<4xf32>
        %32 = stablehlo.divide %cst_10, %28 : tensor<4xf32>
        %33 = stablehlo.add %27, %32 : tensor<4xf32>
        %34 = stablehlo.add %3, %cst_7 : tensor<4xf32>
        %35 = stablehlo.multiply %34, %34 : tensor<4xf32>
        %36 = stablehlo.divide %cst_8, %35 : tensor<4xf32>
        %37 = stablehlo.subtract %31, %36 : tensor<4xf32>
        %38 = stablehlo.divide %cst_8, %34 : tensor<4xf32>
        %39 = stablehlo.add %33, %38 : tensor<4xf32>
        %40 = stablehlo.add %3, %cst_5 : tensor<4xf32>
        %41 = stablehlo.multiply %40, %40 : tensor<4xf32>
        %42 = stablehlo.divide %cst_6, %41 : tensor<4xf32>
        %43 = stablehlo.subtract %37, %42 : tensor<4xf32>
        %44 = stablehlo.divide %cst_6, %40 : tensor<4xf32>
        %45 = stablehlo.add %39, %44 : tensor<4xf32>
        %46 = stablehlo.add %3, %cst_3 : tensor<4xf32>
        %47 = stablehlo.multiply %46, %46 : tensor<4xf32>
        %48 = stablehlo.divide %cst_4, %47 : tensor<4xf32>
        %49 = stablehlo.subtract %43, %48 : tensor<4xf32>
        %50 = stablehlo.divide %cst_4, %46 : tensor<4xf32>
        %51 = stablehlo.add %45, %50 : tensor<4xf32>
        %52 = stablehlo.add %cst_2, %3 : tensor<4xf32>
        %53 = stablehlo.divide %3, %cst_2 : tensor<4xf32>
        %54 = stablehlo.log_plus_one %53 : tensor<4xf32>
        %55 = stablehlo.add %cst_1, %54 : tensor<4xf32>
        %56 = stablehlo.divide %49, %51 : tensor<4xf32>
        %57 = stablehlo.divide %cst_5, %52 : tensor<4xf32>
        %58 = stablehlo.add %55, %56 : tensor<4xf32>
        %59 = stablehlo.subtract %58, %57 : tensor<4xf32>
        %60 = stablehlo.add %arg0, %cst_20 : tensor<4xf32>
        %61 = stablehlo.floor %60 : tensor<4xf32>
        %62 = stablehlo.abs %61 : tensor<4xf32>
        %63 = stablehlo.add %arg0, %62 : tensor<4xf32>
        %64 = stablehlo.multiply %cst_0, %63 : tensor<4xf32>
        %65 = stablehlo.cosine %64 : tensor<4xf32>
        %66 = stablehlo.sine %64 : tensor<4xf32>
        %67 = stablehlo.multiply %cst_0, %65 : tensor<4xf32>
        %68 = stablehlo.divide %67, %66 : tensor<4xf32>
        %69 = stablehlo.subtract %59, %68 : tensor<4xf32>
        %70 = stablehlo.select %0, %69, %59 : tensor<4xi1>, tensor<4xf32>
        %71 = stablehlo.compare LE, %arg0, %cst_18 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %72 = stablehlo.floor %arg0 : tensor<4xf32>
        %73 = stablehlo.compare EQ, %arg0, %72 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
        %74 = stablehlo.and %71, %73 : tensor<4xi1>
        %75 = stablehlo.select %74, %cst, %70 : tensor<4xi1>, tensor<4xf32>
        return %75 : tensor<4xf32>
      }
    }
    """

    @Test("lgamma matches scipy.special.gammaln")
    func testLgamma() async throws {
        let client = try Client.create()
        let executable = try client.compile(Self.lgammaMLIR)

        let x: [Float] = [0.5, 1.5, 2.5, 4.0]
        // scipy.special.gammaln(x), which jax.lax.lgamma reproduces.
        let expected: [Float] = [0.5723649, -0.1207822, 0.2846829, 1.7917595]

        let xBuffer = try client.createBuffer(x, shape: [4], elementType: .float32)
        let outputs = try executable.execute([xBuffer])
        let result = try outputs[0].toFloatArray()

        print("lgamma result: \(result)")
        #expect(result.count == 4)
        for i in 0..<4 {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "lgamma(\(x[i])) = \(result[i]), expected \(expected[i])")
        }
    }

    @Test("digamma matches scipy.special.digamma")
    func testDigamma() async throws {
        let client = try Client.create()
        let executable = try client.compile(Self.digammaMLIR)

        let x: [Float] = [0.5, 1.5, 2.5, 4.0]
        // scipy.special.digamma(x), which jax.lax.digamma reproduces.
        let expected: [Float] = [-1.9635100, 0.0364900, 0.7031566, 1.2561177]

        let xBuffer = try client.createBuffer(x, shape: [4], elementType: .float32)
        let outputs = try executable.execute([xBuffer])
        let result = try outputs[0].toFloatArray()

        print("digamma result: \(result)")
        #expect(result.count == 4)
        for i in 0..<4 {
            #expect(abs(result[i] - expected[i]) < 1e-4,
                    "digamma(\(x[i])) = \(result[i]), expected \(expected[i])")
        }
    }
}
