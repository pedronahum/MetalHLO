// TlalocCompatibilityTests.swift
// MetalHLOTests
//
// Regression tests for StableHLO that Tlaloc (a Kotlin frontend that emits
// StableHLO text directly, not through JAX) sends to the PJRT plugin: the
// pretty and generic spellings JAX never produces, and the numerics bugs those
// programs exposed. Where a fix touches both backends, each program runs on the
// MPSGraph path (`client.compile`) and the Metal-kernel path (`.O0`).

import Testing
import Foundation
@testable import MetalHLO

private func kernelPath() -> CompilationConfig {
    var config = CompilationConfig()
    config.optimizationLevel = .O0
    return config
}

private func compileBoth(_ client: Client, _ mlir: String) throws -> [Executable] {
    [try client.compile(mlir), try client.compile(mlir, config: kernelPath())]
}

private func expectClose(_ got: [Float], _ want: [Float], tol: Float = 1e-5, _ label: String = "") {
    #expect(got.count == want.count, "\(label) count \(got.count) != \(want.count)")
    for (i, (g, w)) in zip(got, want).enumerated() {
        #expect(abs(g - w) <= tol * max(1, abs(w)), "\(label)[\(i)] \(g) != \(w)")
    }
}

@Suite("Tlaloc compatibility - parser", .serialized)
struct TlalocParserTests {

    @Test("dot_general accepts trailing precision and algorithm attributes")
    func dotGeneralTrailingAttributes() throws {
        let client = try Client.create()
        for attrs in ["precision = [HIGHEST, HIGHEST]",
                      "algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>"] {
            let mlir = """
            module @m {
              func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
                %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0], \(attrs) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
                %1 = stablehlo.constant dense<1.0> : tensor<f32>
                %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
                %3 = stablehlo.add %0, %2 : tensor<2x2xf32>
                return %3 : tensor<2x2xf32>
              }
            }
            """
            let a = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [2, 2])
            let b = try client.createBuffer([5, 6, 7, 8] as [Float], shape: [2, 2])
            let out = try client.compile(mlir).execute([a, b])
            expectClose(try out[0].toFloatArray(), [20, 23, 44, 51])
        }
    }

    @Test("dynamic_slice with the pretty `sizes = [...]` spelling")
    func dynamicSliceSizes() throws {
        let client = try Client.create()
        let mlir = """
        module @m {
          func.func @main(%x: tensor<3x4xf32>, %i: tensor<i32>) -> tensor<2x2xf32> {
            %c1 = stablehlo.constant dense<1> : tensor<i32>
            %0 = stablehlo.dynamic_slice %x, %i, %c1, sizes = [2, 2] : (tensor<3x4xf32>, tensor<i32>, tensor<i32>) -> tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }
        """
        let x = try client.createBuffer((0..<12).map { Float($0) }, shape: [3, 4])
        let i = try client.createBuffer([Int32(1)], shape: [])
        let out = try client.compile(mlir).execute([x, i])
        expectClose(try out[0].toFloatArray(), [5, 6, 9, 10])
    }

    @Test("generic while: regions, generic return, and shaped multi-result uses")
    func genericWhile() throws {
        let client = try Client.create()
        // Doubles the vector 3 times; the counter result #0 and vector result #1
        // feed a broadcast and an add after the loop.
        let mlir = """
        module @m {
          func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
            %z = stablehlo.constant dense<0> : tensor<i32>
            %w:2 = "stablehlo.while"(%z, %x) ({
              ^bb0(%i: tensor<i32>, %v: tensor<4xf32>):
                %n = stablehlo.constant dense<3> : tensor<i32>
                %p = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
                "stablehlo.return"(%p) : (tensor<i1>) -> ()
            }, {
              ^bb0(%i: tensor<i32>, %v: tensor<4xf32>):
                %one = stablehlo.constant dense<1> : tensor<i32>
                %i2 = stablehlo.add %i, %one : tensor<i32>
                %v2 = stablehlo.add %v, %v : tensor<4xf32>
                "stablehlo.return"(%i2, %v2) : (tensor<i32>, tensor<4xf32>) -> ()
            }) : (tensor<i32>, tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
            %f = stablehlo.convert %w#0 : (tensor<i32>) -> tensor<f32>
            %b = stablehlo.broadcast_in_dim %f, dims = [] : (tensor<f32>) -> tensor<4xf32>
            %r = stablehlo.add %w#1, %b : tensor<4xf32>
            return %r : tensor<4xf32>
          }
        }
        """
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4])
        let out = try client.compile(mlir).execute([x])
        expectClose(try out[0].toFloatArray(), [11, 19, 27, 35])
    }

    @Test("pretty-printed convolution")
    func prettyConvolution() throws {
        let client = try Client.create()
        let mlir = """
        module @m {
          func.func @main(%x: tensor<1x1x3x3xf32>, %k: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
            %0 = stablehlo.convolution(%x, %k) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x3x3xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
            return %0 : tensor<1x1x2x2xf32>
          }
        }
        """
        let x = try client.createBuffer((1...9).map { Float($0) }, shape: [1, 1, 3, 3])
        let k = try client.createBuffer([1, 0, 0, 1] as [Float], shape: [1, 1, 2, 2])
        let out = try client.compile(mlir).execute([x, k])
        // Each output is x[i][j] + x[i+1][j+1].
        expectClose(try out[0].toFloatArray(), [6, 8, 12, 14])
    }

    @Test("generic reduce_window whose padding carries a tensor type")
    func genericReduceWindowTypedPadding() throws {
        let client = try Client.create()
        let mlir = """
        module @m {
          func.func @main(%x: tensor<4xf32>) -> tensor<2xf32> {
            %z = stablehlo.constant dense<0.0> : tensor<f32>
            %0 = "stablehlo.reduce_window"(%x, %z) <{window_dimensions = array<i64: 2>, window_strides = array<i64: 2>, padding = dense<[[1, 0]]> : tensor<1x2xi64>}> ({
              ^bb0(%a: tensor<f32>, %b: tensor<f32>):
                %s = stablehlo.add %a, %b : tensor<f32>
                stablehlo.return %s : tensor<f32>
            }) : (tensor<4xf32>, tensor<f32>) -> tensor<2xf32>
            return %0 : tensor<2xf32>
          }
        }
        """
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4])
        // The Metal-kernel path, which the PJRT plugin takes for this program.
        // (The MPSGraph path's 1-D reduce_window computes [4, 8] here.)
        let out = try client.compile(mlir, config: kernelPath()).execute([x])
        expectClose(try out[0].toFloatArray(), [1, 5])
    }

    @Test("two-operand sort: comparator keys, trailing attribute dict, stable ties")
    func sortKeysFromComparator() throws {
        let client = try Client.create()
        // Descending on operand 0 only; the payload keeps input order on ties.
        let mlir = """
        module @m {
          func.func @main(%k: tensor<4xf32>, %v: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
            %s:2 = "stablehlo.sort"(%k, %v) ({
              ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<i32>, %d: tensor<i32>):
                %p = stablehlo.compare GT, %a, %b, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                "stablehlo.return"(%p) : (tensor<i1>) -> ()
            }) {dimension = 0 : i64, is_stable = true} : (tensor<4xf32>, tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
            return %s#0, %s#1 : tensor<4xf32>, tensor<4xi32>
          }
        }
        """
        let k = try client.createBuffer([1, 3, 3, 2] as [Float], shape: [4])
        let v = try client.createBuffer([0, 1, 2, 3] as [Int32], shape: [4])
        for exe in try compileBoth(client, mlir) {
            let out = try exe.execute([k, v])
            expectClose(try out[0].toFloatArray(), [3, 3, 2, 1])
            #expect(try out[1].toInt32Array() == [1, 2, 3, 0])
        }
    }
}

@Suite("Tlaloc compatibility - numerics", .serialized)
struct TlalocNumericsTests {

    @Test("gather of windowed slices at per-row starts")
    func windowedGather() throws {
        let client = try Client.create()
        // Row r takes operand[i0, i1 : i1 + 2] for its index pair (i0, i1).
        let mlir = """
        module @m {
          func.func @main(%x: tensor<3x4xf32>, %i: tensor<2x2xi32>) -> tensor<2x2xf32> {
            %0 = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 2>}> : (tensor<3x4xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }
        """
        let x = try client.createBuffer((0..<12).map { Float($0) }, shape: [3, 4])
        let i = try client.createBuffer([0, 1, 2, 3] as [Int32], shape: [2, 2])
        let out = try client.compile(mlir).execute([x, i])
        // Row 1 starts at column 3, clamped to 2 (the slice must fit).
        expectClose(try out[0].toFloatArray(), [1, 2, 10, 11])
    }

    @Test("scatter-add: repeated indices, a non-zero operand, and a reused output slot")
    func scatterAdd() throws {
        let client = try Client.create()
        // The gather and multiply before the scatter leave non-zero bytes in the
        // slots the kernel path's planner may hand the scatter result.
        let mlir = """
        module @m {
          func.func @main(%t: tensor<3x2xf32>, %u: tensor<4x2xf32>) -> (tensor<3x2xf32>, tensor<4x2xf32>) {
            %idx = stablehlo.constant dense<[0, 2, 0, 1]> : tensor<4xi32>
            %g = "stablehlo.gather"(%t, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 2>}> : (tensor<3x2xf32>, tensor<4xi32>) -> tensor<4x2xf32>
            %two = stablehlo.constant dense<2.0> : tensor<4x2xf32>
            %m = stablehlo.multiply %g, %two : tensor<4x2xf32>
            %base = stablehlo.constant dense<10.0> : tensor<3x2xf32>
            %s = "stablehlo.scatter"(%base, %idx, %u) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
              ^bb0(%a: tensor<f32>, %b: tensor<f32>):
                %r = stablehlo.add %a, %b : tensor<f32>
                stablehlo.return %r : tensor<f32>
            }) : (tensor<3x2xf32>, tensor<4xi32>, tensor<4x2xf32>) -> tensor<3x2xf32>
            return %s, %m : tensor<3x2xf32>, tensor<4x2xf32>
          }
        }
        """
        let t = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [3, 2])
        let u = try client.createBuffer([1, 2, 3, 4, 5, 6, 7, 8] as [Float], shape: [4, 2])
        for exe in try compileBoth(client, mlir) {
            let out = try exe.execute([t, u])
            expectClose(try out[0].toFloatArray(), [16, 18, 17, 18, 13, 14])
            expectClose(try out[1].toFloatArray(), [2, 4, 10, 12, 2, 4, 6, 8])
        }
    }

    @Test("f64 tensors run as f32 but cross the API as doubles")
    func float64Boundary() throws {
        let client = try Client.create()
        let mlir = """
        module @m {
          func.func @main(%x: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
            %two = stablehlo.constant dense<2.0> : tensor<3xf64>
            %one = stablehlo.constant dense<1.0> : tensor<3xf64>
            %m = stablehlo.multiply %x, %two : tensor<3xf64>
            %y = stablehlo.add %m, %one : tensor<3xf64>
            return %y, %x : tensor<3xf64>, tensor<3xf64>
          }
        }
        """
        let x = try client.createBuffer([0.5, -1.25, 3.0] as [Double], shape: [3], elementType: .float64)
        for exe in try compileBoth(client, mlir) {
            let out = try exe.execute([x])
            #expect(out[0].elementType == .float64)
            #expect(try out[0].toDoubleArray() == [2.0, -1.5, 7.0])
            #expect(try out[1].toDoubleArray() == [0.5, -1.25, 3.0])
        }
    }

    @Test("tan and atan2 on the Metal-kernel path")
    func tanAndAtan2() throws {
        let client = try Client.create()
        // Nine elements, so atan2 takes the float4 path and its scalar tail.
        let mlir = """
        module @m {
          func.func @main(%y: tensor<9xf32>, %x: tensor<9xf32>) -> (tensor<9xf32>, tensor<9xf32>) {
            %t = stablehlo.tan %y : tensor<9xf32>
            %a = stablehlo.atan2 %y, %x : tensor<9xf32>
            return %t, %a : tensor<9xf32>, tensor<9xf32>
          }
        }
        """
        let ys: [Float] = (0..<9).map { Float($0) * 0.1 - 0.4 }
        let xs: [Float] = (0..<9).map { 1 + Float($0) * 0.25 }
        let y = try client.createBuffer(ys, shape: [9])
        let x = try client.createBuffer(xs, shape: [9])
        let out = try client.compile(mlir, config: kernelPath()).execute([y, x])
        expectClose(try out[0].toFloatArray(), ys.map { Foundation.tan($0) }, tol: 1e-5, "tan")
        expectClose(try out[1].toFloatArray(), zip(ys, xs).map { Foundation.atan2($0, $1) }, tol: 1e-5, "atan2")
    }

    @Test("logical and left shifts on i32: zero fill, oversized counts")
    func integerShifts() throws {
        let client = try Client.create()
        let mlir = """
        module @m {
          func.func @main(%a: tensor<4xi32>, %n: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
            %r = stablehlo.shift_right_logical %a, %n : tensor<4xi32>
            %l = stablehlo.shift_left %a, %n : tensor<4xi32>
            return %r, %l : tensor<4xi32>, tensor<4xi32>
          }
        }
        """
        let a = try client.createBuffer([-16, -1, 1 << 30, 7] as [Int32], shape: [4])
        let n = try client.createBuffer([2, 31, 32, -1] as [Int32], shape: [4])
        for exe in try compileBoth(client, mlir) {
            let out = try exe.execute([a, n])
            #expect(try out[0].toInt32Array() == [1_073_741_820, 1, 0, 0])
            #expect(try out[1].toInt32Array() == [-64, Int32.min, 0, 0])
        }
    }

    @Test("batched triangular_solve and cholesky")
    func batchedLinalg() throws {
        let client = try Client.create()
        let mlir = """
        module @m {
          func.func @main(%a: tensor<2x2x2xf32>, %b: tensor<2x2x1xf32>, %s: tensor<2x2x2xf32>) -> (tensor<2x2x1xf32>, tensor<2x2x2xf32>) {
            %x = "stablehlo.triangular_solve"(%a, %b) {left_side = true, lower = true, unit_diagonal = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>} : (tensor<2x2x2xf32>, tensor<2x2x1xf32>) -> tensor<2x2x1xf32>
            %l = "stablehlo.cholesky"(%s) {lower = true} : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
            return %x, %l : tensor<2x2x1xf32>, tensor<2x2x2xf32>
          }
        }
        """
        let a = try client.createBuffer([2, 0, 1, 1, 1, 0, 3, 4] as [Float], shape: [2, 2, 2])
        let b = try client.createBuffer([2, 3, 1, 7] as [Float], shape: [2, 2, 1])
        let s = try client.createBuffer([4, 2, 2, 2, 9, 3, 3, 5] as [Float], shape: [2, 2, 2])
        let out = try client.compile(mlir).execute([a, b, s])
        expectClose(try out[0].toFloatArray(), [1, 2, 1, 1])
        expectClose(try out[1].toFloatArray(), [2, 0, 1, 1, 3, 0, 1, 2])
    }

    @Test("an argument returned unchanged is an output on the Metal-kernel path")
    func passthroughOutput() throws {
        let client = try Client.create()
        let mlir = """
        module @m {
          func.func @main(%a: tensor<3xf32>, %b: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) {
            %s = stablehlo.add %a, %b : tensor<3xf32>
            return %s, %b, %a : tensor<3xf32>, tensor<3xf32>, tensor<3xf32>
          }
        }
        """
        let a = try client.createBuffer([1, 2, 3] as [Float], shape: [3])
        let b = try client.createBuffer([10, 20, 30] as [Float], shape: [3])
        let out = try client.compile(mlir, config: kernelPath()).execute([a, b])
        #expect(out.count == 3)
        expectClose(try out[0].toFloatArray(), [11, 22, 33])
        expectClose(try out[1].toFloatArray(), [10, 20, 30])
        expectClose(try out[2].toFloatArray(), [1, 2, 3])
    }

    @Test("concurrent executions of one executable keep their own results")
    func concurrentExecute() throws {
        let client = try Client.create()
        let mlir = """
        module @m {
          func.func @main(%a: tensor<256xf32>) -> tensor<256xf32> {
            %t = stablehlo.add %a, %a : tensor<256xf32>
            %u = stablehlo.multiply %t, %a : tensor<256xf32>
            return %u : tensor<256xf32>
          }
        }
        """
        let exe = try client.compile(mlir, config: kernelPath())
        let inputs = try (0..<8).map { k in try client.createBuffer([Float](repeating: Float(k + 1), count: 256), shape: [256]) }
        var results = [[Float]](repeating: [], count: 8)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: 8) { k in
            let r = (try? exe.execute([inputs[k]]))?.first.flatMap { try? $0.toFloatArray() } ?? []
            lock.lock(); results[k] = r; lock.unlock()
        }
        for k in 0..<8 {
            let v = Float(k + 1)
            #expect(results[k] == [Float](repeating: 2 * v * v, count: 256), "execution \(k)")
        }
    }
}
