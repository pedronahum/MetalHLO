// LapackRoutingTest.swift
// MetalHLOCoreTests
//
// Verifies that JAX-CPU LAPACK FFI custom_calls for cholesky / solve_triangular
// (@lapack_spotrf_ffi / @lapack_strsm_ffi) are recognized by the parser, have
// their ASCII-coded backend_config flags decoded, and are routed to the native
// MPSGraph cholesky / triangular_solve operations.

import Testing
import Metal
@testable import MetalHLOCore

@Suite("LAPACK FFI routing")
struct LapackRoutingTest {

    @Test("spotrf custom_call routes to cholesky + info constant")
    func choleskyRouting() throws {
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %0:2 = stablehlo.custom_call @lapack_spotrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<i32>)
            return %0#0 : tensor<2x2xf32>
          }
        }
        """
        let module = try Parser(source: mlir).parse()
        let ops = module.function.operations
        // One .cholesky (factor) + one .constant (info=0); the original
        // custom_call op must not survive.
        #expect(ops.contains { $0.kind == .cholesky && $0.result == "%0.0" })
        #expect(ops.contains { $0.kind == .constant && $0.result == "%0.1" })
        #expect(!ops.contains { $0.kind == .customCall })
        let chol = ops.first { $0.kind == .cholesky }!
        // uplo = 'L' (76) → lower factor.
        #expect(chol.attributes.lower == true)
    }

    @Test("spotrf upper uplo decodes lower=false")
    func choleskyUpperFlag() throws {
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %0:2 = stablehlo.custom_call @lapack_spotrf_ffi(%arg0) {mhlo.backend_config = {uplo = 85 : ui8}} : (tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<i32>)
            return %0#0 : tensor<2x2xf32>
          }
        }
        """
        let module = try Parser(source: mlir).parse()
        let chol = module.function.operations.first { $0.kind == .cholesky }!
        #expect(chol.attributes.lower == false)
    }

    @Test("strsm custom_call routes to triangular_solve with decoded flags")
    func triangularSolveRouting() throws {
        // diag = 78('N') non-unit, side = 76('L') left, trans_x = 84('T') transpose,
        // uplo = 85('U') upper.
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x1xf32> {
            %0 = stablehlo.custom_call @lapack_strsm_ffi(%arg0, %arg1) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 84 : ui8, uplo = 85 : ui8}} : (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
            return %0 : tensor<2x1xf32>
          }
        }
        """
        let module = try Parser(source: mlir).parse()
        let ops = module.function.operations
        let solve = ops.first { $0.kind == .triangularSolve }
        #expect(solve != nil)
        #expect(!ops.contains { $0.kind == .customCall })
        #expect(solve?.attributes.unitDiagonal == false)  // diag = 'N'
        #expect(solve?.attributes.leftSide == true)        // side = 'L'
        #expect(solve?.attributes.transposeA == .transpose) // trans_x = 'T'
        #expect(solve?.attributes.lower == false)          // uplo = 'U'
    }
}

// End-to-end numeric checks: parse a StableHLO triangular_solve / cholesky and
// run it through the MPSGraph compiler + executor, comparing against a reference
// computed on the CPU. These pin the native triangular_solve flag handling
// (lower / unit_diagonal / transpose) that the LAPACK routing depends on.
@Suite("LAPACK native numerics")
struct LapackNumericsTest {

    /// Reference forward/backward substitution for op(A)·X = B on row-major data.
    private func refSolve(_ a: [[Float]], _ b: [[Float]],
                          lower: Bool, unit: Bool, transpose: Bool) -> [[Float]] {
        let n = a.count
        let m = b[0].count
        // Effective matrix: Aᵀ flips lower<->upper.
        var mat = a
        var lwr = lower
        if transpose {
            var t = Array(repeating: Array(repeating: Float(0), count: n), count: n)
            for i in 0..<n { for j in 0..<n { t[i][j] = a[j][i] } }
            mat = t
            lwr = !lower
        }
        var x = Array(repeating: Array(repeating: Float(0), count: m), count: n)
        let order = lwr ? Array(0..<n) : Array((0..<n).reversed())
        for i in order {
            for c in 0..<m {
                var s = b[i][c]
                let prev = lwr ? Array(0..<i) : Array((i+1)..<n)
                for k in prev { s -= mat[i][k] * x[k][c] }
                x[i][c] = unit ? s : s / mat[i][i]
            }
        }
        return x
    }

    private func runSolve(aFlat: [Float], aShape: [Int], bFlat: [Float], bShape: [Int],
                          lower: Bool, unit: Bool, transpose: String) throws -> [Float]? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<\(aShape[0])x\(aShape[1])xf32>, %arg1: tensor<\(bShape[0])x\(bShape[1])xf32>) -> tensor<\(bShape[0])x\(bShape[1])xf32> {
            %0 = stablehlo.triangular_solve %arg0, %arg1, left_side = true, lower = \(lower), transpose_a = #stablehlo<transpose \(transpose)>, unit_diagonal = \(unit) : (tensor<\(aShape[0])x\(aShape[1])xf32>, tensor<\(bShape[0])x\(bShape[1])xf32>) -> tensor<\(bShape[0])x\(bShape[1])xf32>
            return %0 : tensor<\(bShape[0])x\(bShape[1])xf32>
          }
        }
        """
        let module = try Parser(source: mlir).parse()
        let exec = try MetalExecutor()
        let compiled = try exec.compile(module: module)
        let aStorage = BufferStorage(floatData: aFlat, shape: aShape, device: device)
        let bStorage = BufferStorage(floatData: bFlat, shape: bShape, device: device)
        let outs = try exec.execute(compiled: compiled, inputs: [aStorage, bStorage])
        return outs[0].data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    @Test("triangular_solve honors lower / unit / transpose flags")
    func solveFlags() throws {
        // Lower-triangular A and a general upper-triangular A reused via flags.
        let lowerA: [[Float]] = [[2, 0, 0], [1, 3, 0], [-1, 2, 4]]
        let upperA: [[Float]] = [[2, 1, -1], [0, 3, 2], [0, 0, 4]]
        let b: [[Float]] = [[1, 2], [3, -1], [0, 5]]
        let bFlat = b.flatMap { $0 }

        for (a, lower) in [(lowerA, true), (upperA, false)] {
            let aFlat = a.flatMap { $0 }
            for unit in [false, true] {
                for (transStr, transBool) in [("NO_TRANSPOSE", false), ("TRANSPOSE", true)] {
                    guard let got = try runSolve(
                        aFlat: aFlat, aShape: [3, 3], bFlat: bFlat, bShape: [3, 2],
                        lower: lower, unit: unit, transpose: transStr
                    ) else { return }  // no Metal device (CI)
                    let want = refSolve(a, b, lower: lower, unit: unit, transpose: transBool)
                        .flatMap { $0 }
                    for i in 0..<got.count {
                        let diff = abs(got[i] - want[i])
                        #expect(diff < 1e-3,
                            "lower=\(lower) unit=\(unit) trans=\(transStr) idx \(i): got=\(got[i]) want=\(want[i])")
                    }
                }
            }
        }
    }

    @Test("cholesky factor reconstructs the SPD input")
    func choleskyNumeric() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        // SPD matrix A = LLᵀ.
        let a: [[Float]] = [[4, 2, -2], [2, 10, 2], [-2, 2, 5]]
        let aFlat = a.flatMap { $0 }
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
            %0 = stablehlo.cholesky %arg0, lower = true : (tensor<3x3xf32>) -> tensor<3x3xf32>
            return %0 : tensor<3x3xf32>
          }
        }
        """
        let module = try Parser(source: mlir).parse()
        let device = MTLCreateSystemDefaultDevice()!
        let exec = try MetalExecutor()
        let compiled = try exec.compile(module: module)
        let aStorage = BufferStorage(floatData: aFlat, shape: [3, 3], device: device)
        let outs = try exec.execute(compiled: compiled, inputs: [aStorage])
        let l = outs[0].data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        // Reconstruct L·Lᵀ and compare to A.
        for i in 0..<3 {
            for j in 0..<3 {
                var s: Float = 0
                for k in 0..<3 { s += l[i*3+k] * l[j*3+k] }
                #expect(abs(s - a[i][j]) < 1e-3, "LLᵀ[\(i),\(j)]=\(s) want \(a[i][j])")
            }
        }
    }
}
