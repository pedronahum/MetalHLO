// ParserTests.swift
// MetalHLOCoreTests
//
// Tests for the MLIR Parser.

import Testing
@testable import MetalHLOCore

@Suite("Parser Tests")
struct ParserTests {

    // MARK: - Basic Module Parsing

    @Test("Parse empty module")
    func parseEmptyModule() throws {
        let mlir = """
        module @empty {
          func.func @main() -> () {
            return
          }
        }
        """

        let parser = Parser(source: mlir)
        let module = try parser.parse()

        #expect(module.name == "empty")
        #expect(module.function.name == "main")
        #expect(module.function.inputs.isEmpty)
        #expect(module.function.operations.isEmpty)
    }

    @Test("Parse module with inputs")
    func parseModuleWithInputs() throws {
        let mlir = """
        module @test {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            return %arg0 : tensor<2x3xf32>
          }
        }
        """

        let parser = Parser(source: mlir)
        let module = try parser.parse()

        #expect(module.function.inputs.count == 2)
        #expect(module.function.inputs[0].name == "%arg0")
        #expect(module.function.inputs[0].type.shape == [2, 3])
        #expect(module.function.inputs[0].type.elementType == .float32)
    }

    // MARK: - Tensor Type Parsing

    @Test("Parse various tensor shapes")
    func parseTensorTypes() throws {
        let mlir = """
        module @types {
          func.func @main(%a: tensor<f32>, %b: tensor<10xf32>, %c: tensor<2x3xf32>, %d: tensor<2x3x4xf32>) -> (tensor<f32>) {
            return %a : tensor<f32>
          }
        }
        """

        let parser = Parser(source: mlir)
        let module = try parser.parse()

        #expect(module.function.inputs[0].type.shape == [])  // scalar
        #expect(module.function.inputs[1].type.shape == [10])  // 1D
        #expect(module.function.inputs[2].type.shape == [2, 3])  // 2D
        #expect(module.function.inputs[3].type.shape == [2, 3, 4])  // 3D
    }

    @Test("Parse various element types")
    func parseElementTypes() throws {
        let mlir = """
        module @types {
          func.func @main(%a: tensor<2xf16>, %b: tensor<2xf32>, %c: tensor<2xi32>, %d: tensor<2xi64>) -> (tensor<2xf32>) {
            return %b : tensor<2xf32>
          }
        }
        """

        let parser = Parser(source: mlir)
        let module = try parser.parse()

        #expect(module.function.inputs[0].type.elementType == .float16)
        #expect(module.function.inputs[1].type.elementType == .float32)
        #expect(module.function.inputs[2].type.elementType == .int32)
        #expect(module.function.inputs[3].type.elementType == .int64)
    }

    // MARK: - Operation Parsing

    @Test("Parse add operation")
    func parseAddOperation() throws {
        let mlir = """
        module @add {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """

        let parser = Parser(source: mlir)
        let module = try parser.parse()

        #expect(module.function.operations.count == 1)

        let op = module.function.operations[0]
        #expect(op.result == "%0")
        #expect(op.kind == .add)
        #expect(op.operands == ["%arg0", "%arg1"])
        #expect(op.resultType.shape == [2, 3])
    }

    @Test("Parse multiple operations")
    func parseMultipleOperations() throws {
        let mlir = """
        module @multi {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
            %1 = stablehlo.multiply %0, %arg0 : tensor<2x3xf32>
            return %1 : tensor<2x3xf32>
          }
        }
        """

        let parser = Parser(source: mlir)
        let module = try parser.parse()

        #expect(module.function.operations.count == 2)
        #expect(module.function.operations[0].kind == .add)
        #expect(module.function.operations[1].kind == .multiply)
    }

    // MARK: - Unary Operations

    @Test("Parse unary operations")
    func parseUnaryOperations() throws {
        let mlir = """
        module @unary {
          func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.negate %arg0 : tensor<2x3xf32>
            %1 = stablehlo.abs %0 : tensor<2x3xf32>
            %2 = stablehlo.exponential %1 : tensor<2x3xf32>
            return %2 : tensor<2x3xf32>
          }
        }
        """

        let parser = Parser(source: mlir)
        let module = try parser.parse()

        #expect(module.function.operations.count == 3)
        #expect(module.function.operations[0].kind == .negate)
        #expect(module.function.operations[1].kind == .abs)
        #expect(module.function.operations[2].kind == .exponential)
    }

    // MARK: - Return Values

    @Test("Parse return values")
    func parseReturnValues() throws {
        let mlir = """
        module @test {
          func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
            %0 = stablehlo.abs %arg0 : tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """

        let parser = Parser(source: mlir)
        let module = try parser.parse()

        #expect(module.function.returnValues == ["%0"])
    }

    // MARK: - Error Handling

    @Test("Parse error on invalid syntax")
    func parseErrorOnInvalidSyntax() {
        let mlir = "this is not valid MLIR"

        let parser = Parser(source: mlir)

        #expect(throws: ParseError.self) {
            try parser.parse()
        }
    }
}
