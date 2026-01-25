// ControlFlowTests.swift
// MetalHLOTests
//
// Tests for Phase 8: Control Flow operations.

import Testing
import Foundation
@testable import MetalHLO

// MARK: - While Loop Tests

@Suite("While Loop")
struct WhileLoopTests {

    @Test("Simple counter loop")
    func simpleCounterLoop() throws {
        let client = try Client.create()

        // Count from 0 to 5
        // while (x < limit) { x = x + 1 }
        let mlir = """
        module @counter {
          func.func @main(%init: tensor<f32>, %limit: tensor<f32>, %step: tensor<f32>) -> (tensor<f32>) {
            %result = stablehlo.while %init
              cond {
                ^bb(%x: tensor<f32>):
                %cond = stablehlo.compare %x, %limit, LT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                stablehlo.return %cond : tensor<i1>
              }
              do {
                ^bb(%x: tensor<f32>):
                %next = stablehlo.add %x, %step : tensor<f32>
                stablehlo.return %next : tensor<f32>
              }
              : tensor<f32>
            return %result : tensor<f32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let initVal = try client.createBuffer([Float(0.0)], shape: [], elementType: .float32)
        let limit = try client.createBuffer([Float(5.0)], shape: [], elementType: .float32)
        let step = try client.createBuffer([Float(1.0)], shape: [], elementType: .float32)

        let outputs = try executable.execute([initVal, limit, step])
        let result = try outputs[0].toFloatArray()

        #expect(result[0] == 5.0, "Counter should reach limit of 5")
    }

    @Test("Newton-Raphson square root")
    func newtonRaphsonSqrt() throws {
        let client = try Client.create()

        // Newton-Raphson iteration for sqrt(a):
        // x_{n+1} = 0.5 * (x_n + a / x_n)
        // Continue while |x^2 - a| > epsilon (error is above threshold)
        let mlir = """
        module @newton_sqrt {
          func.func @main(%a: tensor<f32>, %x_init: tensor<f32>, %epsilon: tensor<f32>) -> (tensor<f32>) {
            %result = stablehlo.while %x_init
              cond {
                ^bb(%x: tensor<f32>):
                // Check if |x^2 - a| > epsilon (continue while error is above threshold)
                %x_squared = stablehlo.multiply %x, %x : tensor<f32>
                %diff = stablehlo.subtract %x_squared, %a : tensor<f32>
                %abs_diff = stablehlo.abs %diff : tensor<f32>
                %cond = stablehlo.compare %abs_diff, %epsilon, GT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                stablehlo.return %cond : tensor<i1>
              }
              do {
                ^bb(%x: tensor<f32>):
                // x_next = 0.5 * (x + a/x)
                %div = stablehlo.divide %a, %x : tensor<f32>
                %sum = stablehlo.add %x, %div : tensor<f32>
                %half = stablehlo.constant dense<0.5> : tensor<f32>
                %next = stablehlo.multiply %sum, %half : tensor<f32>
                stablehlo.return %next : tensor<f32>
              }
              : tensor<f32>
            return %result : tensor<f32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Compute sqrt(16) = 4, starting from initial guess of 10
        let a = try client.createBuffer([Float(16.0)], shape: [], elementType: .float32)
        let xInit = try client.createBuffer([Float(10.0)], shape: [], elementType: .float32)  // Initial guess far from answer
        let epsilon = try client.createBuffer([Float(0.0001)], shape: [], elementType: .float32)

        let outputs = try executable.execute([a, xInit, epsilon])
        let result = try outputs[0].toFloatArray()

        #expect(abs(result[0] - 4.0) < 0.01, "Newton-Raphson should converge to sqrt(16) = 4")
    }
}

// MARK: - If Conditional Tests

@Suite("If Conditional")
struct IfConditionalTests {

    @Test("Simple if-then-else")
    func simpleIfThenElse() throws {
        let client = try Client.create()

        // if (pred) then { a } else { b }
        let mlir = """
        module @conditional {
          func.func @main(%pred: tensor<i1>, %a: tensor<f32>, %b: tensor<f32>) -> (tensor<f32>) {
            %result = stablehlo.if %pred
              then {
                stablehlo.return %a : tensor<f32>
              }
              else {
                stablehlo.return %b : tensor<f32>
              }
              : tensor<f32>
            return %result : tensor<f32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Test with true predicate
        let trueData = Data([UInt8(1)])  // Bool true as 1
        let predTrue = try client.createBuffer(bytes: trueData, shape: [], elementType: .int1)
        let a = try client.createBuffer([Float(10.0)], shape: [], elementType: .float32)
        let b = try client.createBuffer([Float(20.0)], shape: [], elementType: .float32)

        let outputsTrue = try executable.execute([predTrue, a, b])
        let resultTrue = try outputsTrue[0].toFloatArray()
        #expect(resultTrue[0] == 10.0, "With true predicate, should return 'a'")

        // Test with false predicate
        let falseData = Data([UInt8(0)])  // Bool false as 0
        let predFalse = try client.createBuffer(bytes: falseData, shape: [], elementType: .int1)
        let outputsFalse = try executable.execute([predFalse, a, b])
        let resultFalse = try outputsFalse[0].toFloatArray()
        #expect(resultFalse[0] == 20.0, "With false predicate, should return 'b'")
    }

    @Test("If with computation in branches")
    func ifWithComputation() throws {
        let client = try Client.create()

        // if (pred) then { x + 1 } else { x * 2 }
        let mlir = """
        module @if_compute {
          func.func @main(%pred: tensor<i1>, %x: tensor<4xf32>) -> (tensor<4xf32>) {
            %one = stablehlo.constant dense<1.0> : tensor<4xf32>
            %two = stablehlo.constant dense<2.0> : tensor<4xf32>
            %result = stablehlo.if %pred
              then {
                %added = stablehlo.add %x, %one : tensor<4xf32>
                stablehlo.return %added : tensor<4xf32>
              }
              else {
                %multiplied = stablehlo.multiply %x, %two : tensor<4xf32>
                stablehlo.return %multiplied : tensor<4xf32>
              }
              : tensor<4xf32>
            return %result : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let x: [Float] = [1.0, 2.0, 3.0, 4.0]
        let xBuffer = try client.createBuffer(x, shape: [4], elementType: .float32)

        // Test true: x + 1
        let trueData = Data([UInt8(1)])
        let predTrue = try client.createBuffer(bytes: trueData, shape: [], elementType: .int1)
        let outputsTrue = try executable.execute([predTrue, xBuffer])
        let resultTrue = try outputsTrue[0].toFloatArray()
        #expect(resultTrue == [2.0, 3.0, 4.0, 5.0], "With true: should add 1")

        // Test false: x * 2
        let falseData = Data([UInt8(0)])
        let predFalse = try client.createBuffer(bytes: falseData, shape: [], elementType: .int1)
        let outputsFalse = try executable.execute([predFalse, xBuffer])
        let resultFalse = try outputsFalse[0].toFloatArray()
        #expect(resultFalse == [2.0, 4.0, 6.0, 8.0], "With false: should multiply by 2")
    }
}

// MARK: - Combined Control Flow Tests

@Suite("Combined Control Flow")
struct CombinedControlFlowTests {

    @Test("ReLU using max (control flow alternative)")
    func reluUsingMax() throws {
        // This tests that control flow doesn't break regular operations
        let client = try Client.create()

        let mlir = """
        module @relu_simple {
          func.func @main(%x: tensor<4xf32>) -> (tensor<4xf32>) {
            %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
            %result = stablehlo.maximum %x, %zero : tensor<4xf32>
            return %result : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x: [Float] = [-2.0, -1.0, 1.0, 2.0]
        let xBuffer = try client.createBuffer(x, shape: [4], elementType: .float32)

        let outputs = try executable.execute([xBuffer])
        let result = try outputs[0].toFloatArray()

        #expect(result == [0.0, 0.0, 1.0, 2.0])
    }
}
