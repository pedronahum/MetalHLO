// LinalgMPSTests.swift
// MetalHLOTests
//
// Coverage for the linear-algebra / region ops that lower onto the MPSGraph
// backend: stablehlo.triangular_solve, stablehlo.cholesky, and stablehlo.map.
// Each case is checked numerically against a NumPy/JAX reference.

import Testing
@testable import MetalHLO

@Suite("Linalg + Map (MPSGraph path)", .serialized)
struct LinalgMPSTests {

    // Solve a lower-triangular system L·x = b by forward substitution.
    //   L = [[2,0,0],[6,1,0],[-8,5,3]]   (the Cholesky factor of an SPD matrix)
    //   b = [[1],[2],[3]]   →   x = [0.5, -1, 4]
    @Test("triangular_solve lower-triangular forward substitution")
    func triangularSolveLower() throws {
        let client = try Client.create()
        let mlir = """
        module @trisolve {
          func.func @main(%a: tensor<3x3xf32>, %b: tensor<3x1xf32>) -> (tensor<3x1xf32>) {
            %0 = "stablehlo.triangular_solve"(%a, %b) {
              left_side = true,
              lower = true,
              unit_diagonal = false,
              transpose_a = #stablehlo<transpose NO_TRANSPOSE>
            } : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
            return %0 : tensor<3x1xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer(
            [2, 0, 0, 6, 1, 0, -8, 5, 3] as [Float], shape: [3, 3], elementType: .float32)
        let b = try client.createBuffer([1, 2, 3] as [Float], shape: [3, 1], elementType: .float32)
        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()
        let expected: [Float] = [0.5, -1, 4]
        #expect(result.count == 3)
        for (got, want) in zip(result, expected) {
            #expect(abs(got - want) < 1e-4, "trisolve got \(got), want \(want)")
        }
    }

    // Cholesky of the SPD matrix
    //   A = [[4,12,-16],[12,37,-43],[-16,-43,98]]
    // The lower factor is L = [[2,0,0],[6,1,0],[-8,5,3]], and L·Lᵀ == A.
    @Test("cholesky lower factor of an SPD matrix")
    func choleskyLower() throws {
        let client = try Client.create()
        let mlir = """
        module @chol {
          func.func @main(%a: tensor<3x3xf32>) -> (tensor<3x3xf32>) {
            %0 = "stablehlo.cholesky"(%a) {lower = true} : (tensor<3x3xf32>) -> tensor<3x3xf32>
            return %0 : tensor<3x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let a = try client.createBuffer(
            [4, 12, -16, 12, 37, -43, -16, -43, 98] as [Float],
            shape: [3, 3], elementType: .float32)
        let outputs = try executable.execute([a])
        let result = try outputs[0].toFloatArray()
        // Row-major lower-triangular factor (upper entries are zero).
        let expected: [Float] = [2, 0, 0, 6, 1, 0, -8, 5, 3]
        #expect(result.count == 9)
        for (got, want) in zip(result, expected) {
            #expect(abs(got - want) < 1e-3, "cholesky got \(got), want \(want)")
        }
    }

    // map applies a scalar lambda element-wise. Here: x -> x*x + 1.
    //   [0,1,2,3] -> [1,2,5,10]
    @Test("map inlines an element-wise region")
    func mapElementwise() throws {
        let client = try Client.create()
        let mlir = """
        module @mapfn {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = "stablehlo.map"(%arg0) ({
              ^bb0(%a: tensor<f32>):
                %1 = stablehlo.multiply %a, %a : tensor<f32>
                %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                %2 = stablehlo.add %1, %cst : tensor<f32>
                stablehlo.return %2 : tensor<f32>
            }) {dimensions = array<i64: 0>} : (tensor<4xf32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let x = try client.createBuffer([0, 1, 2, 3] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()
        let expected: [Float] = [1, 2, 5, 10]
        #expect(result.count == 4)
        for (got, want) in zip(result, expected) {
            #expect(abs(got - want) < 1e-4, "map got \(got), want \(want)")
        }
    }

    // SVD of a 3×3 matrix, full_matrices = true. Lowered exactly as JAX-CPU emits
    // it: a single @lapack_sgesdd_ffi custom_call (mode 65 = 'A') wrapped in the
    // info-check select chain. Routed to the native .svd op and executed host-side
    // via Accelerate LAPACK. Verified by reconstruction U·diag(S)·Vh ≈ A and the
    // singular values (sign-unambiguous, descending).
    @Test("svd full_matrices of a 3x3 matrix (host LAPACK)")
    func svdFull3x3() throws {
        let client = try Client.create()
        let mlir = """
        module @jit_svd {
          func.func public @main(%arg0: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<3x3xf32>) {
            %0:5 = stablehlo.custom_call @lapack_sgesdd_ffi(%arg0) {backend_config = "", mhlo.backend_config = {mode = 65 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<i32>)
            %c = stablehlo.constant dense<0> : tensor<i32>
            %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
            %2 = stablehlo.compare EQ, %0#4, %1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
            %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1xi1>
            %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
            %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3xf32>
            %5 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xi1>) -> tensor<3xi1>
            %6 = stablehlo.select %5, %0#1, %4 : tensor<3xi1>, tensor<3xf32>
            %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
            %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
            %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
            %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
            %10 = stablehlo.select %9, %0#2, %8 : tensor<3x3xi1>, tensor<3x3xf32>
            %11 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
            %cst_1 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
            %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
            %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
            %14 = stablehlo.select %13, %0#3, %12 : tensor<3x3xi1>, tensor<3x3xf32>
            return %10, %6, %14 : tensor<3x3xf32>, tensor<3xf32>, tensor<3x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        // Row-major A (3×3).
        let aData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 10]
        let a = try client.createBuffer(aData, shape: [3, 3], elementType: .float32)
        let outputs = try executable.execute([a])
        let U = try outputs[0].toFloatArray()   // 3×3
        let S = try outputs[1].toFloatArray()   // 3
        let Vh = try outputs[2].toFloatArray()  // 3×3
        #expect(U.count == 9 && S.count == 3 && Vh.count == 9)

        // Singular values are positive and sorted descending.
        #expect(S[0] >= S[1] && S[1] >= S[2])
        #expect(S.allSatisfy { $0 >= -1e-5 })

        // Reconstruct A = U · diag(S) · Vh (all 3×3, row-major).
        var recon = [Float](repeating: 0, count: 9)
        for i in 0..<3 {
            for j in 0..<3 {
                var acc: Float = 0
                for k in 0..<3 {
                    acc += U[i * 3 + k] * S[k] * Vh[k * 3 + j]
                }
                recon[i * 3 + j] = acc
            }
        }
        for idx in 0..<9 {
            #expect(abs(recon[idx] - aData[idx]) < 1e-3,
                    "svd 3x3 recon[\(idx)] got \(recon[idx]), want \(aData[idx])")
        }
    }

    // SVD of a non-square 4×3 matrix, full_matrices = false (mode 83 = 'S').
    // U is 4×3 (thin), S is length-3, Vh is 3×3. Reconstruction U·diag(S)·Vh ≈ A.
    @Test("svd thin (full_matrices=false) of a 4x3 matrix (host LAPACK)")
    func svdThin4x3() throws {
        let client = try Client.create()
        let mlir = """
        module @jit_svd {
          func.func public @main(%arg0: tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<3xf32>, tensor<3x3xf32>) {
            %0:5 = stablehlo.custom_call @lapack_sgesdd_ffi(%arg0) {backend_config = "", mhlo.backend_config = {mode = 83 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<3xf32>, tensor<4x3xf32>, tensor<3x3xf32>, tensor<i32>)
            %c = stablehlo.constant dense<0> : tensor<i32>
            %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
            %2 = stablehlo.compare EQ, %0#4, %1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
            %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1xi1>
            %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
            %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3xf32>
            %5 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xi1>) -> tensor<3xi1>
            %6 = stablehlo.select %5, %0#1, %4 : tensor<3xi1>, tensor<3xf32>
            %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
            %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
            %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x3xf32>
            %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x3xi1>
            %10 = stablehlo.select %9, %0#2, %8 : tensor<4x3xi1>, tensor<4x3xf32>
            %11 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
            %cst_1 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
            %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
            %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
            %14 = stablehlo.select %13, %0#3, %12 : tensor<3x3xi1>, tensor<3x3xf32>
            return %10, %6, %14 : tensor<4x3xf32>, tensor<3xf32>, tensor<3x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        // Row-major A (4×3).
        let aData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
        let a = try client.createBuffer(aData, shape: [4, 3], elementType: .float32)
        let outputs = try executable.execute([a])
        let U = try outputs[0].toFloatArray()   // 4×3
        let S = try outputs[1].toFloatArray()   // 3
        let Vh = try outputs[2].toFloatArray()  // 3×3
        #expect(U.count == 12 && S.count == 3 && Vh.count == 9)
        #expect(S[0] >= S[1] && S[1] >= S[2])

        // Reconstruct A (4×3) = U(4×3) · diag(S) · Vh(3×3).
        var recon = [Float](repeating: 0, count: 12)
        for i in 0..<4 {
            for j in 0..<3 {
                var acc: Float = 0
                for k in 0..<3 {
                    acc += U[i * 3 + k] * S[k] * Vh[k * 3 + j]
                }
                recon[i * 3 + j] = acc
            }
        }
        for idx in 0..<12 {
            #expect(abs(recon[idx] - aData[idx]) < 1e-3,
                    "svd 4x3 recon[\(idx)] got \(recon[idx]), want \(aData[idx])")
        }
    }

    // Verifies a QR result: Q·R reconstructs A and Q's columns are orthonormal
    // (QᵀQ ≈ I). Raw Q/R signs are convention-dependent, so we never compare raw
    // factors — only the reconstruction and orthonormality.
    private func checkQR(
        A: [Float], m: Int, n: Int,
        Q: [Float], R: [Float], qCols: Int
    ) {
        #expect(Q.count == m * qCols, "Q has \(Q.count) elems, want \(m * qCols)")
        #expect(R.count == qCols * n, "R has \(R.count) elems, want \(qCols * n)")

        // Q·R ≈ A  (Q is m×qCols, R is qCols×n).
        for i in 0..<m {
            for j in 0..<n {
                var acc: Float = 0
                for k in 0..<qCols { acc += Q[i * qCols + k] * R[k * n + j] }
                #expect(abs(acc - A[i * n + j]) < 1e-3,
                        "QR recon[\(i),\(j)] got \(acc), want \(A[i * n + j])")
            }
        }
        // QᵀQ ≈ I_qCols (columns orthonormal).
        for p in 0..<qCols {
            for q in 0..<qCols {
                var acc: Float = 0
                for i in 0..<m { acc += Q[i * qCols + p] * Q[i * qCols + q] }
                let want: Float = (p == q) ? 1 : 0
                #expect(abs(acc - want) < 1e-3,
                        "QᵀQ[\(p),\(q)] got \(acc), want \(want)")
            }
        }
    }

    // QR of a square 3×3 matrix, reduced mode (Q 3×3, R 3×3). Lowered exactly as
    // JAX-CPU emits it: @lapack_sgeqrf_ffi factorization + @lapack_sorgqr_ffi to
    // materialize Q, with the slice/iota/select wrapper that zeroes R's strict
    // lower triangle. Routed to the native .qr ops and executed host-side.
    @Test("qr reduced of a 3x3 matrix (host LAPACK)")
    func qrReduced3x3() throws {
        let client = try Client.create()
        let mlir = """
        module @jit_qr {
          func.func public @main(%arg0: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>) {
            %0:2 = stablehlo.custom_call @lapack_sgeqrf_ffi(%arg0) {backend_config = "", operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>)
            %1 = stablehlo.custom_call @lapack_sorgqr_ffi(%0#0, %0#1) {backend_config = "", operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32>
            %2 = stablehlo.iota dim = 0 : tensor<3x3xi32>
            %c = stablehlo.constant dense<-1> : tensor<i32>
            %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32>
            %4 = stablehlo.add %2, %3 : tensor<3x3xi32>
            %5 = stablehlo.iota dim = 1 : tensor<3x3xi32>
            %6 = stablehlo.compare GE, %4, %5, SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
            %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
            %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
            %8 = stablehlo.select %6, %7, %0#0 : tensor<3x3xi1>, tensor<3x3xf32>
            return %1, %8 : tensor<3x3xf32>, tensor<3x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let aData: [Float] = [1, 2, 3, 4, 5.5, 6, 7, 8, 10]
        let a = try client.createBuffer(aData, shape: [3, 3], elementType: .float32)
        let outputs = try executable.execute([a])
        let Q = try outputs[0].toFloatArray()
        let R = try outputs[1].toFloatArray()
        checkQR(A: aData, m: 3, n: 3, Q: Q, R: R, qCols: 3)
    }

    // QR of a tall 4×3 matrix, reduced mode (Q 4×3, R 3×3). The geqrf factored
    // matrix is sliced to its top-left 3×3 block before the strict-lower-triangle
    // mask forms R.
    @Test("qr reduced of a tall 4x3 matrix (host LAPACK)")
    func qrReducedTall4x3() throws {
        let client = try Client.create()
        let mlir = """
        module @jit_qr {
          func.func public @main(%arg0: tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<3x3xf32>) {
            %0:2 = stablehlo.custom_call @lapack_sgeqrf_ffi(%arg0) {backend_config = "", operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<3xf32>)
            %1 = stablehlo.custom_call @lapack_sorgqr_ffi(%0#0, %0#1) {backend_config = "", operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x3xf32>, tensor<3xf32>) -> tensor<4x3xf32>
            %2 = stablehlo.slice %0#0 [0:3, 0:3] : (tensor<4x3xf32>) -> tensor<3x3xf32>
            %3 = stablehlo.iota dim = 0 : tensor<3x3xi32>
            %c = stablehlo.constant dense<-1> : tensor<i32>
            %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32>
            %5 = stablehlo.add %3, %4 : tensor<3x3xi32>
            %6 = stablehlo.iota dim = 1 : tensor<3x3xi32>
            %7 = stablehlo.compare GE, %5, %6, SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
            %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
            %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
            %9 = stablehlo.select %7, %8, %2 : tensor<3x3xi1>, tensor<3x3xf32>
            return %1, %9 : tensor<4x3xf32>, tensor<3x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let aData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
        let a = try client.createBuffer(aData, shape: [4, 3], elementType: .float32)
        let outputs = try executable.execute([a])
        let Q = try outputs[0].toFloatArray()
        let R = try outputs[1].toFloatArray()
        checkQR(A: aData, m: 4, n: 3, Q: Q, R: R, qCols: 3)
    }

    // QR of a tall 4×3 matrix, complete mode (Q 4×4, R 4×3). The geqrf factored
    // matrix is padded to 4×4 before sorgqr materializes the full orthonormal Q.
    @Test("qr complete of a tall 4x3 matrix (host LAPACK)")
    func qrCompleteTall4x3() throws {
        let client = try Client.create()
        let mlir = """
        module @jit_qr {
          func.func public @main(%arg0: tensor<4x3xf32>) -> (tensor<4x4xf32>, tensor<4x3xf32>) {
            %0:2 = stablehlo.custom_call @lapack_sgeqrf_ffi(%arg0) {backend_config = "", operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<3xf32>)
            %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
            %1 = stablehlo.pad %0#0, %cst, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x3xf32>, tensor<f32>) -> tensor<4x4xf32>
            %2 = stablehlo.custom_call @lapack_sorgqr_ffi(%1, %0#1) {backend_config = "", operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x4xf32>, tensor<3xf32>) -> tensor<4x4xf32>
            %3 = stablehlo.iota dim = 0 : tensor<4x3xi32>
            %c = stablehlo.constant dense<-1> : tensor<i32>
            %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x3xi32>
            %5 = stablehlo.add %3, %4 : tensor<4x3xi32>
            %6 = stablehlo.iota dim = 1 : tensor<4x3xi32>
            %7 = stablehlo.compare GE, %5, %6, SIGNED : (tensor<4x3xi32>, tensor<4x3xi32>) -> tensor<4x3xi1>
            %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
            %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x3xf32>
            %9 = stablehlo.select %7, %8, %0#0 : tensor<4x3xi1>, tensor<4x3xf32>
            return %2, %9 : tensor<4x4xf32>, tensor<4x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        let aData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
        let a = try client.createBuffer(aData, shape: [4, 3], elementType: .float32)
        let outputs = try executable.execute([a])
        let Q = try outputs[0].toFloatArray()
        let R = try outputs[1].toFloatArray()
        checkQR(A: aData, m: 4, n: 3, Q: Q, R: R, qCols: 4)
    }

    // Verifies an eigh result: eigenvalues w match the reference (ascending) and
    // the reconstruction v·diag(w)·vᵀ ≈ A. Eigenvector signs are convention-
    // dependent, so we never compare raw eigenvectors — only w and the
    // reconstruction. A is symmetric N×N, w is length N, v is N×N (columns are
    // eigenvectors, row-major).
    private func checkEigh(
        A: [Float], n: Int, w: [Float], v: [Float], expectedW: [Float]
    ) {
        #expect(w.count == n, "w has \(w.count) elems, want \(n)")
        #expect(v.count == n * n, "v has \(v.count) elems, want \(n * n)")

        // Eigenvalues ascending and matching the reference.
        for i in 0..<(n - 1) {
            #expect(w[i] <= w[i + 1] + 1e-4, "eigenvalues not ascending at \(i): \(w)")
        }
        for i in 0..<n {
            #expect(abs(w[i] - expectedW[i]) < 1e-3,
                    "eigh w[\(i)] got \(w[i]), want \(expectedW[i])")
        }

        // Reconstruct A = v·diag(w)·vᵀ  (v column = eigenvector, row-major).
        for i in 0..<n {
            for j in 0..<n {
                var acc: Float = 0
                for k in 0..<n { acc += v[i * n + k] * w[k] * v[j * n + k] }
                #expect(abs(acc - A[i * n + j]) < 1e-3,
                        "eigh recon[\(i),\(j)] got \(acc), want \(A[i * n + j])")
            }
        }
    }

    // Symmetric eigendecomposition of a 3×3 matrix, lowered exactly as JAX-CPU
    // emits it: the matrix is symmetrized ((A + Aᵀ)/2) then passed to a single
    // @lapack_ssyevd_ffi custom_call (mode 86 = 'V', uplo 76 = 'L') wrapped in the
    // info-check select chain. Routed to native .eigh ops and executed host-side
    // via Accelerate LAPACK. Verified by w (ascending) + reconstruction.
    @Test("eigh of a symmetric 3x3 matrix (host LAPACK)")
    func eighSym3x3() throws {
        let client = try Client.create()
        let mlir = """
        module @jit_eigh {
          func.func public @main(%arg0: tensor<3x3xf32>) -> (tensor<3xf32>, tensor<3x3xf32>) {
            %t = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
            %s = stablehlo.add %arg0, %t : tensor<3x3xf32>
            %two = stablehlo.constant dense<2.000000e+00> : tensor<f32>
            %twob = stablehlo.broadcast_in_dim %two, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
            %a = stablehlo.divide %s, %twob : tensor<3x3xf32>
            %0:3 = stablehlo.custom_call @lapack_ssyevd_ffi(%a) {backend_config = "", mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<i32>)
            %c = stablehlo.constant dense<0> : tensor<i32>
            %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
            %2 = stablehlo.compare EQ, %0#2, %1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
            %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
            %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
            %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
            %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
            %6 = stablehlo.select %5, %0#0, %4 : tensor<3x3xi1>, tensor<3x3xf32>
            %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1xi1>
            %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
            %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3xf32>
            %9 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<1xi1>) -> tensor<3xi1>
            %10 = stablehlo.select %9, %0#1, %8 : tensor<3xi1>, tensor<3xf32>
            return %10, %6 : tensor<3xf32>, tensor<3x3xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        // Symmetric A (3×3): diag(2,3,4) + small off-diagonal coupling.
        //   A = [[2,1,0],[1,3,1],[0,1,4]]
        let aData: [Float] = [2, 1, 0, 1, 3, 1, 0, 1, 4]
        let a = try client.createBuffer(aData, shape: [3, 3], elementType: .float32)
        let outputs = try executable.execute([a])
        let w = try outputs[0].toFloatArray()   // 3
        let v = try outputs[1].toFloatArray()   // 3×3
        // Reference eigenvalues (NumPy/LAPACK, ascending) for the matrix above.
        let expectedW: [Float] = [1.2679492, 3.0, 4.732051]
        checkEigh(A: aData, n: 3, w: w, v: v, expectedW: expectedW)
    }

    // Symmetric eigendecomposition of a 4×4 matrix, upper triangle (uplo 85 = 'U').
    // Same lowering shape as the 3×3 case at N = 4.
    @Test("eigh of a symmetric 4x4 matrix, upper (host LAPACK)")
    func eighSym4x4Upper() throws {
        let client = try Client.create()
        let mlir = """
        module @jit_eigh {
          func.func public @main(%arg0: tensor<4x4xf32>) -> (tensor<4xf32>, tensor<4x4xf32>) {
            %t = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
            %s = stablehlo.add %arg0, %t : tensor<4x4xf32>
            %two = stablehlo.constant dense<2.000000e+00> : tensor<f32>
            %twob = stablehlo.broadcast_in_dim %two, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
            %a = stablehlo.divide %s, %twob : tensor<4x4xf32>
            %0:3 = stablehlo.custom_call @lapack_ssyevd_ffi(%a) {backend_config = "", mhlo.backend_config = {mode = 86 : ui8, uplo = 85 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>, tensor<i32>)
            %c = stablehlo.constant dense<0> : tensor<i32>
            %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
            %2 = stablehlo.compare EQ, %0#2, %1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
            %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
            %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
            %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
            %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1>
            %6 = stablehlo.select %5, %0#0, %4 : tensor<4x4xi1>, tensor<4x4xf32>
            %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1xi1>
            %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
            %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4xf32>
            %9 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<1xi1>) -> tensor<4xi1>
            %10 = stablehlo.select %9, %0#1, %8 : tensor<4xi1>, tensor<4xf32>
            return %10, %6 : tensor<4xf32>, tensor<4x4xf32>
          }
        }
        """
        let executable = try client.compile(mlir)
        // Symmetric tridiagonal A (4×4): diag 2..5 with unit off-diagonal coupling.
        let aData: [Float] = [
            2, 1, 0, 0,
            1, 3, 1, 0,
            0, 1, 4, 1,
            0, 0, 1, 5,
        ]
        let a = try client.createBuffer(aData, shape: [4, 4], elementType: .float32)
        let outputs = try executable.execute([a])
        let w = try outputs[0].toFloatArray()
        let v = try outputs[1].toFloatArray()
        // Reference eigenvalues (LAPACK, ascending).
        let expectedW: [Float] = [1.2547188, 2.8227172, 4.177283, 5.745281]
        checkEigh(A: aData, n: 4, w: w, v: v, expectedW: expectedW)
    }
}
