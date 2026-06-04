// ReshapeViewLivenessTest.swift
// Guards the reshape-as-view memory-planner liveness fix.
//
// The code generator folds a reshape of a contiguous source (a compute-kernel
// output or a function input) into a zero-copy view that aliases the source's
// slab slot (CodeGenerator.tryGenerateReshapeView). For that to be correct, the
// memory planner must keep the source tensor alive across the view's readers —
// otherwise the offset assigner sees the source "die" at the reshape op, reuses
// its slot for a later tensor, and silently corrupts the view. This is the
// documented prior regression (loss 4.567 → 4.239) that
// StaticMemoryPlanner.extendViewSourceLiveness fixes.

import Testing
import Foundation
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Reshape View Liveness", .serialized)
struct ReshapeViewLivenessTests {

    private func lifetime(_ created: Int, _ lastUsed: Int) -> ScheduledTensorLifetime {
        ScheduledTensorLifetime(createdAt: created, lastUsedAt: lastUsed, byteSize: 32)
    }

    // Mechanism guard (deterministic, no offset-assignment dependence): the
    // reshape SOURCE's lifetime must be stretched to cover the reshape RESULT's
    // last use, transitively through a reshape-of-reshape chain.
    //
    //   op0 %p = add(...)               source, would otherwise die at op1
    //   op1 %r = reshape(%p)            view of %p
    //   op5 %rflat = reshape(%r)        view of %r (→ %p); last used at op6
    //   op6 %out = add(%rflat, ...)
    //
    // Base lifetimes (as computeLifetimes would produce them): %p dies at op1,
    // %r at op5. After the pass both must reach op6 so %p's slot survives.
    @Test("View source lifetime is extended through a reshape chain")
    func sourceLivenessExtendedThroughChain() {
        let function = HLOFunction(
            name: "main",
            inputs: [HLOArgument(name: "%x", type: TensorType(shape: [8], elementType: .float32))],
            outputTypes: [TensorType(shape: [8], elementType: .float32)],
            operations: [
                HLOOperation(result: "%p", kind: .add, operands: ["%x", "%x"],
                             resultType: TensorType(shape: [8], elementType: .float32)),
                HLOOperation(result: "%r", kind: .reshape, operands: ["%p"],
                             resultType: TensorType(shape: [2, 4], elementType: .float32)),
                HLOOperation(result: "%rflat", kind: .reshape, operands: ["%r"],
                             resultType: TensorType(shape: [8], elementType: .float32)),
                HLOOperation(result: "%out", kind: .add, operands: ["%rflat", "%x"],
                             resultType: TensorType(shape: [8], elementType: .float32)),
            ],
            returnValues: ["%out"]
        )

        // Hand-built base lifetimes matching the chain above.
        let base: [TensorID: ScheduledTensorLifetime] = [
            "%p": lifetime(0, 1),       // dies at the reshape without the fix
            "%r": lifetime(1, 2),       // dies at the second reshape without the fix
            "%rflat": lifetime(2, 3),   // read by %out
            "%out": lifetime(3, 4),
        ]

        let planner = StaticMemoryPlanner()
        let extended = planner.extendViewSourceLiveness(function, base)

        // %r feeds %rflat (last used at 3) → %r must reach 3.
        #expect(extended["%r"]?.lastUsedAt == 3, "view %r not extended to its reader: \(String(describing: extended["%r"]))")
        // %p feeds %r (now reaching 3) → %p must reach 3, transitively.
        #expect(extended["%p"]?.lastUsedAt == 3, "source %p not extended through the chain: \(String(describing: extended["%p"]))")
        // createdAt and size must be preserved.
        #expect(extended["%p"]?.createdAt == 0)
        #expect(extended["%p"]?.byteSize == 32)
    }

    // End-to-end correctness smoke: a reshape of a compute output, read back
    // after intervening allocations, produces the right values through the
    // view path (codegen + planner together).
    @Test("Reshape of a compute output is numerically correct end-to-end")
    func reshapeOfComputeOutputCorrect() async throws {
        let mlir = """
        module @reshape_view_e2e {
          func.func @main(%x: tensor<8xf32>) -> (tensor<8xf32>) {
            %p = stablehlo.add %x, %x : tensor<8xf32>
            %r = stablehlo.reshape %p : (tensor<8xf32>) -> tensor<2x4xf32>
            %q1 = stablehlo.multiply %x, %x : tensor<8xf32>
            %q2 = stablehlo.add %q1, %x : tensor<8xf32>
            %q3 = stablehlo.multiply %q2, %x : tensor<8xf32>
            %rflat = stablehlo.reshape %r : (tensor<2x4xf32>) -> tensor<8xf32>
            %out = stablehlo.add %rflat, %q3 : tensor<8xf32>
            return %out : tensor<8xf32>
          }
        }
        """

        let client = try Client.create()
        let executable = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))

        let xin: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let x = try client.createBuffer(xin, shape: [8], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // out = 2x + (x²+x)·x = 2x + x³ + x²
        let expected = xin.map { xv -> Float in 2 * xv + xv * xv * xv + xv * xv }
        #expect(result == expected, "reshape-view produced wrong values: \(result) vs \(expected)")
    }

    // Regression: reshape of a FUNCTION INPUT must not be folded into a slab
    // view. Function inputs are passed zero-copy (not slab-resident), so a view
    // over them resolves to zeroed slab memory and the reshape reads all zeros.
    // This silently broke ResNet18 training: the BatchNorm scale/bias params are
    // 1-D inputs reshaped to [1,1,1,C], and the zeroed reshape made the whole
    // network output 0 (loss flat, no learning). The fix excludes inputs from
    // the reshape-as-view fold (CodeGenerator.tryGenerateReshapeView), falling
    // back to a real reshape kernel. Several target shapes are covered because
    // the bug hit every shape-changing reshape of an input, not just one.
    @Test("Reshape of a function input is numerically correct (not zeroed)")
    func reshapeOfFunctionInputCorrect() async throws {
        let shapes: [(out: String, dims: [Int])] = [
            ("tensor<1x1x1x16xf32>", [1, 1, 1, 16]),
            ("tensor<2x8xf32>", [2, 8]),
            ("tensor<4x4xf32>", [4, 4]),
            ("tensor<1x16xf32>", [1, 16]),
        ]
        let xin: [Float] = (0..<16).map { Float($0) }
        let client = try Client.create()

        for shape in shapes {
            let mlir = """
            module @reshape_input {
              func.func @main(%p: tensor<16xf32>) -> (\(shape.out)) {
                %r = stablehlo.reshape %p : (tensor<16xf32>) -> \(shape.out)
                return %r : \(shape.out)
              }
            }
            """
            let executable = try client.compile(mlir, config: CompilationConfig(optimizationLevel: .O2))
            let p = try client.createBuffer(xin, shape: [16], elementType: .float32)
            let result = try executable.execute([p])[0].toFloatArray()
            // Reshape is a pure relabel — values and order are preserved.
            #expect(result == xin,
                    "reshape(input[16])->\(shape.dims) zeroed/corrupted values: \(result)")
        }
    }
}
