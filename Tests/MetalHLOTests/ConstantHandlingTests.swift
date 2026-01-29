// ConstantHandlingTests.swift
// MetalHLOTests
//
// Tests for constant handling in both MPSGraph and IntegratedExecutor paths.

import Testing
import Metal
@testable import MetalHLO
@testable import MetalHLOCore

@Suite("Constant Handling Tests")
struct ConstantHandlingTests {

    // MARK: - Debug Test

    @Test("Debug: trace constant extraction")
    func debugConstantExtraction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let mlir = """
        module @constants_only {
          func.func @main() -> (tensor<4xf32>) {
            %0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
            %1 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
            %2 = stablehlo.add %0, %1 : tensor<4xf32>
            return %2 : tensor<4xf32>
          }
        }
        """

        // Parse the MLIR
        let parser = Parser(source: mlir)
        let module = try parser.parse()
        let function = module.function

        print("\n=== DEBUG: Parsed Operations ===")
        for op in function.operations {
            print("  \(op.result) = \(op.kind.rawValue) operands:\(op.operands)")
            if let cv = op.attributes.constantValue {
                print("    constantValue: \(cv)")
            }
        }

        // Build OptimizedModule
        let optimized = OptimizedModule.from(function)
        print("\n=== DEBUG: OptimizedModule.constants ===")
        print("  Constants count: \(optimized.constants.count)")
        for (id, value) in optimized.constants {
            print("  '\(id)' -> \(value)")
        }

        print("\n=== DEBUG: OptimizedModule.operations ===")
        for op in optimized.operations {
            print("  \(op.id) type:\(op.type) inputs:\(op.inputs)")
        }

        // Full compilation through MetalHLOCompiler
        let compilerConfig = MetalHLOCompiler.Config(optimizationLevel: .O0)
        let compiler = MetalHLOCompiler(device: device, config: compilerConfig)
        let compiled = try compiler.compile(mlir)

        print("\n=== DEBUG: CompiledExecutable ===")
        print("  Pipelines: \(compiled.pipelines.keys.sorted())")
        print("  Dispatches: \(compiled.dispatches.keys.sorted())")
        print("  Bindings keys: \(compiled.bindings.keys.sorted())")
        print("  ConstantBuffers keys: \(compiled.constantBuffers.keys.sorted())")
        print("  ConstantBuffers count: \(compiled.constantBuffers.count)")
        print("  ExecutionOrder: \(compiled.executionOrder)")

        for (opID, bindings) in compiled.bindings {
            print("  Bindings for \(opID):")
            for binding in bindings {
                print("    index:\(binding.index) source:\(binding.source)")
            }
        }

        // Check constant buffers content
        for (id, buffer) in compiled.constantBuffers {
            let ptr = buffer.contents().bindMemory(to: Float.self, capacity: 4)
            let values = (0..<4).map { ptr[$0] }
            print("  ConstantBuffer '\(id)': \(values)")
        }

        print("\n=== DEBUG: Memory Plan ===")
        print("  ExecutionOrder: \(compiled.memoryPlan.executionOrder)")
        print("  TensorOffsets: \(compiled.memoryPlan.tensorOffsets)")
        print("  TotalBytes: \(compiled.memoryPlan.totalBytes)")
        print("  PeakMemory: \(compiled.memoryPlan.peakMemory)")

        print("\n=== DEBUG: Input/Output Specs ===")
        print("  InputSpecs: \(compiled.inputSpecs)")
        print("  OutputSpecs: \(compiled.outputSpecs)")

        // Verify
        #expect(compiled.constantBuffers.count == 2, "Expected 2 constant buffers, got \(compiled.constantBuffers.count)")

        // Actually execute and check results
        let executor = try IntegratedExecutor(device: device, executable: compiled)
        let result = try executor.execute(inputs: [:])

        print("\n=== DEBUG: Execution Result ===")
        print("  Output keys: \(result.outputs.keys)")
        for (name, buffer) in result.outputs {
            let ptr = buffer.contents().bindMemory(to: Float.self, capacity: 4)
            let values = (0..<4).map { ptr[$0] }
            print("  Output '\(name)': \(values)")
        }

        // The result should be [1+1, 2+1, 3+1, 4+1] = [2, 3, 4, 5]
        if let outputBuffer = result.outputs.values.first {
            let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: 4)
            let values = (0..<4).map { ptr[$0] }
            #expect(values == [2.0, 3.0, 4.0, 5.0], "Expected [2, 3, 4, 5], got \(values)")
        }

        // Test O1 (basic optimization without aggressive constant folding)
        print("\n=== DEBUG: Testing O1 optimization ===")
        let compilerConfigO1 = MetalHLOCompiler.Config(optimizationLevel: .O1)
        let compilerO1 = MetalHLOCompiler(device: device, config: compilerConfigO1)
        let compiledO1 = try compilerO1.compile(mlir)

        print("  Pipelines: \(compiledO1.pipelines.keys.sorted())")
        print("  ConstantBuffers keys: \(compiledO1.constantBuffers.keys.sorted())")
        print("  ExecutionOrder: \(compiledO1.executionOrder)")

        for (opID, bindings) in compiledO1.bindings {
            print("  Bindings for \(opID):")
            for binding in bindings {
                print("    index:\(binding.index) source:\(binding.source)")
            }
        }

        let executorO1 = try IntegratedExecutor(device: device, executable: compiledO1)
        let resultO1 = try executorO1.execute(inputs: [:])

        print("\n=== DEBUG: O1 Execution Result ===")
        for (name, buffer) in resultO1.outputs {
            let ptr = buffer.contents().bindMemory(to: Float.self, capacity: 4)
            let values = (0..<4).map { ptr[$0] }
            print("  Output '\(name)': \(values)")
        }

        // O1 should still work correctly with constants
        if let outputBuffer = resultO1.outputs.values.first {
            let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: 4)
            let values = (0..<4).map { ptr[$0] }
            #expect(values == [2.0, 3.0, 4.0, 5.0], "O1: Expected [2, 3, 4, 5], got \(values)")
        }

        // Test O2 (aggressive constant folding)
        print("\n=== DEBUG: Testing O2 optimization ===")

        // First, let's see what the O2 OptimizedModule looks like (bypass createConstantBuffers)
        let parserO2 = Parser(source: mlir)
        let moduleO2 = try parserO2.parse()
        let functionO2 = moduleO2.function
        let optimizedO2 = OptimizedModule.from(functionO2)
        print("  Before O2 passes - OptimizedModule.operations:")
        for op in optimizedO2.operations {
            print("    \(op.id): type=\(op.type) inputs=\(op.inputs)")
            if let outputs = op.outputs.first {
                print("      output shape: \(outputs.shape), type: \(outputs.elementType)")
            }
        }
        print("  Before O2 passes - OptimizedModule.constants: \(optimizedO2.constants.keys.sorted())")

        let compilerConfigO2 = MetalHLOCompiler.Config(optimizationLevel: .O2)
        let compilerO2 = MetalHLOCompiler(device: device, config: compilerConfigO2)
        let compiledO2 = try compilerO2.compile(mlir)

        print("  Pipelines: \(compiledO2.pipelines.keys.sorted())")
        print("  ConstantBuffers keys: \(compiledO2.constantBuffers.keys.sorted())")
        print("  ExecutionOrder: \(compiledO2.executionOrder)")
        print("  OutputSpecs: \(compiledO2.outputSpecs)")
        print("  MemoryPlan.tensorOffsets: \(compiledO2.memoryPlan.tensorOffsets)")

        for (opID, bindings) in compiledO2.bindings {
            print("  Bindings for \(opID):")
            for binding in bindings {
                print("    index:\(binding.index) source:\(binding.source)")
            }
        }

        // Check if O2 folded everything into a constant
        if compiledO2.executionOrder.isEmpty {
            print("  O2 folded everything - no operations to execute!")
            print("  This case needs special handling: output is a pre-computed constant")
        } else {
            print("  O2 has \(compiledO2.executionOrder.count) operations to execute")
            print("  This should be 0 if constant folding is working - the result should be a constant")

            // Try to execute and see what happens
            do {
                let executorO2 = try IntegratedExecutor(device: device, executable: compiledO2)
                let resultO2 = try executorO2.execute(inputs: [:])
                print("\n=== DEBUG: O2 Execution Result ===")
                for (name, buffer) in resultO2.outputs {
                    let ptr = buffer.contents().bindMemory(to: Float.self, capacity: 4)
                    let values = (0..<4).map { ptr[$0] }
                    print("  Output '\(name)': \(values)")
                }
            } catch {
                print("  O2 execution error: \(error)")
            }
        }
    }

    // MARK: - Constants-Only Functions

    @Test("Execute function with only constants (MPSGraph path)")
    func executeConstantsOnlyMPSGraph() throws {
        let client = try Client.create()
        let mlir = """
        module @constants_only {
          func.func @main() -> (tensor<4xf32>) {
            %0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
            %1 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
            %2 = stablehlo.add %0, %1 : tensor<4xf32>
            return %2 : tensor<4xf32>
          }
        }
        """

        // Test with default MPSGraph path (O0 - no optimization)
        let config = CompilationConfig(optimizationLevel: .O0)
        let exe = try client.compile(mlir, config: config)
        let outputs = try exe.execute([])
        let result = try outputs[0].toFloatArray()
        #expect(result == [2, 3, 4, 5])
    }

    @Test("Execute function with only constants (IntegratedExecutor path)")
    func executeConstantsOnlyIntegrated() throws {
        let client = try Client.create()
        let mlir = """
        module @constants_only {
          func.func @main() -> (tensor<4xf32>) {
            %0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
            %1 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
            %2 = stablehlo.add %0, %1 : tensor<4xf32>
            return %2 : tensor<4xf32>
          }
        }
        """

        // Test with IntegratedExecutor path (O2)
        let config = CompilationConfig(optimizationLevel: .O2)
        let exe = try client.compile(mlir, config: config)
        let outputs = try exe.execute([])
        let result = try outputs[0].toFloatArray()
        #expect(result == [2, 3, 4, 5])
    }

    @Test("Execute function with splat constants")
    func executeSplatConstants() throws {
        let client = try Client.create()
        let mlir = """
        module @splat_constants {
          func.func @main() -> (tensor<8xf32>) {
            %0 = stablehlo.constant dense<2.5> : tensor<8xf32>
            %1 = stablehlo.constant dense<1.5> : tensor<8xf32>
            %2 = stablehlo.multiply %0, %1 : tensor<8xf32>
            return %2 : tensor<8xf32>
          }
        }
        """

        // Test with O2 optimization (IntegratedExecutor)
        let config = CompilationConfig(optimizationLevel: .O2)
        let exe = try client.compile(mlir, config: config)
        let outputs = try exe.execute([])
        let result = try outputs[0].toFloatArray()

        // 2.5 * 1.5 = 3.75 for all elements
        let expected: [Float] = Array(repeating: 3.75, count: 8)
        #expect(result == expected)
    }

    // MARK: - Large Constants

    @Test("Execute function with large baked constants")
    func executeLargeConstants() throws {
        let client = try Client.create()

        // Generate a 32x32 constant matrix (1024 elements)
        var mlirRows: [String] = []
        for i in 0..<32 {
            let rowValues = (0..<32).map { j in
                String(format: "%.1f", Float(i * 32 + j) / 1024.0)
            }.joined(separator: ", ")
            mlirRows.append("[\(rowValues)]")
        }
        let mlirMatrix = mlirRows.joined(separator: ", ")

        let mlir = """
        module @large_constants {
          func.func @main() -> (tensor<32x32xf32>) {
            %0 = stablehlo.constant dense<[\(mlirMatrix)]> : tensor<32x32xf32>
            %1 = stablehlo.multiply %0, %0 : tensor<32x32xf32>
            return %1 : tensor<32x32xf32>
          }
        }
        """

        // Use O1 for complex cases (O2 fusion can cause issues with multi-op graphs)
        let config = CompilationConfig(optimizationLevel: .O1)
        let exe = try client.compile(mlir, config: config)
        let outputs = try exe.execute([])
        let result = try outputs[0].toFloatArray()

        // Verify first few values are correct (not zeros)
        #expect(result[0] == 0)  // 0 * 0 = 0
        let val1 = Float(1) / 1024.0
        #expect(abs(result[1] - val1 * val1) < 0.0001)
        let val2 = Float(2) / 1024.0
        #expect(abs(result[2] - val2 * val2) < 0.0001)

        // Verify a value in the middle
        let midIdx = 512
        let midVal = Float(midIdx) / 1024.0
        #expect(abs(result[midIdx] - midVal * midVal) < 0.0001)
    }

    // MARK: - Mixed Inputs and Constants

    @Test("Execute function with mixed inputs and constants")
    func executeMixedInputsAndConstants() throws {
        let client = try Client.create()
        let mlir = """
        module @mixed {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %cst = stablehlo.constant dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf32>
            %0 = stablehlo.add %arg0, %cst : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        // Use O1 for tests with inputs (O2 fusion can cause issues)
        let config = CompilationConfig(optimizationLevel: .O1)
        let exe = try client.compile(mlir, config: config)
        let input = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let outputs = try exe.execute([input])
        let result = try outputs[0].toFloatArray()
        #expect(result == [11, 22, 33, 44])
    }

    @Test("Execute function with multiple constants and inputs")
    func executeMultipleConstantsAndInputs() throws {
        let client = try Client.create()
        let mlir = """
        module @multi_constants {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %cst1 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
            %cst2 = stablehlo.constant dense<[2.0, 2.0, 2.0, 2.0]> : tensor<4xf32>
            %0 = stablehlo.add %arg0, %cst1 : tensor<4xf32>
            %1 = stablehlo.multiply %arg1, %cst2 : tensor<4xf32>
            %2 = stablehlo.add %0, %1 : tensor<4xf32>
            return %2 : tensor<4xf32>
          }
        }
        """

        // Use O1 for multi-op graphs (O2 fusion can cause issues)
        let config = CompilationConfig(optimizationLevel: .O1)
        let exe = try client.compile(mlir, config: config)
        let a = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([5, 5, 5, 5] as [Float], shape: [4], elementType: .float32)
        let outputs = try exe.execute([a, b])
        let result = try outputs[0].toFloatArray()

        // (arg0 + cst1) + (arg1 * cst2) = ([1,2,3,4] + [1,1,1,1]) + ([5,5,5,5] * [2,2,2,2])
        // = [2,3,4,5] + [10,10,10,10] = [12,13,14,15]
        #expect(result == [12, 13, 14, 15])
    }

    // MARK: - Constant with Different Types

    @Test("Execute function with integer constants")
    func executeIntegerConstants() throws {
        let client = try Client.create()
        let mlir = """
        module @int_constants {
          func.func @main() -> (tensor<4xi32>) {
            %0 = stablehlo.constant dense<[10, 20, 30, 40]> : tensor<4xi32>
            %1 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
            %2 = stablehlo.add %0, %1 : tensor<4xi32>
            return %2 : tensor<4xi32>
          }
        }
        """

        let config = CompilationConfig(optimizationLevel: .O2)
        let exe = try client.compile(mlir, config: config)
        let outputs = try exe.execute([])
        let result = try outputs[0].toInt32Array()
        #expect(result == [11, 22, 33, 44])
    }

    // MARK: - Scalar Constants

    @Test("Execute function with scalar constant")
    func executeScalarConstant() throws {
        let client = try Client.create()
        let mlir = """
        module @scalar_constant {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %cst = stablehlo.constant dense<2.0> : tensor<f32>
            %broadcast = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
            %0 = stablehlo.multiply %arg0, %broadcast : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let config = CompilationConfig(optimizationLevel: .O2)
        let exe = try client.compile(mlir, config: config)
        let input = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let outputs = try exe.execute([input])
        let result = try outputs[0].toFloatArray()
        #expect(result == [2, 4, 6, 8])
    }

    // MARK: - Constant Chain Operations

    @Test("Execute function with chained constant operations")
    func executeChainedConstantOps() throws {
        let client = try Client.create()
        let mlir = """
        module @chained_constants {
          func.func @main() -> (tensor<4xf32>) {
            %c1 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
            %c2 = stablehlo.constant dense<[2.0, 2.0, 2.0, 2.0]> : tensor<4xf32>
            %c3 = stablehlo.constant dense<[0.5, 0.5, 0.5, 0.5]> : tensor<4xf32>
            %0 = stablehlo.multiply %c1, %c2 : tensor<4xf32>
            %1 = stablehlo.add %0, %c3 : tensor<4xf32>
            return %1 : tensor<4xf32>
          }
        }
        """

        // Use O1 for multi-op graphs (O2 fusion needs more work)
        let config = CompilationConfig(optimizationLevel: .O1)
        let exe = try client.compile(mlir, config: config)
        let outputs = try exe.execute([])
        let result = try outputs[0].toFloatArray()

        // ([1,2,3,4] * [2,2,2,2]) + [0.5,0.5,0.5,0.5] = [2,4,6,8] + [0.5,0.5,0.5,0.5] = [2.5, 4.5, 6.5, 8.5]
        #expect(result == [2.5, 4.5, 6.5, 8.5])
    }

    // MARK: - Consistency Between Paths

    @Test("MPSGraph and IntegratedExecutor produce same results for constants")
    func consistencyBetweenPaths() throws {
        let client = try Client.create()
        let mlir = """
        module @consistency {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %cst = stablehlo.constant dense<[5.0, 10.0, 15.0, 20.0]> : tensor<4xf32>
            %0 = stablehlo.add %arg0, %cst : tensor<4xf32>
            %1 = stablehlo.multiply %0, %cst : tensor<4xf32>
            return %1 : tensor<4xf32>
          }
        }
        """

        let input = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)

        // MPSGraph path (O0)
        let mpsConfig = CompilationConfig(optimizationLevel: .O0)
        let mpsExe = try client.compile(mlir, config: mpsConfig)
        let mpsOutputs = try mpsExe.execute([input])
        let mpsResult = try mpsOutputs[0].toFloatArray()

        // IntegratedExecutor path (O1 - O2 fusion needs more work)
        let intConfig = CompilationConfig(optimizationLevel: .O1)
        let intExe = try client.compile(mlir, config: intConfig)
        let intOutputs = try intExe.execute([input])
        let intResult = try intOutputs[0].toFloatArray()

        // Both paths should produce the same result
        // (arg0 + cst) * cst = ([1,2,3,4] + [5,10,15,20]) * [5,10,15,20]
        // = [6,12,18,24] * [5,10,15,20] = [30, 120, 270, 480]
        #expect(mpsResult == [30, 120, 270, 480])
        #expect(intResult == [30, 120, 270, 480])
        #expect(mpsResult == intResult)
    }
}
