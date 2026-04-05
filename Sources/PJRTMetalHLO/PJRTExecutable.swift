// PJRTExecutable.swift
// PJRTMetalHLO
//
// PJRT Executable (serializable) and LoadedExecutable (ready to run).

import CPJRTApi
import Metal
import MetalHLO
import MetalHLOCore
import Foundation

// MARK: - MLIR Bytecode Conversion

/// Converts MLIR bytecode to text using a Python helper subprocess.
///
/// JAX sends compiled StableHLO programs as MLIR bytecode (binary format
/// starting with `ML\xefR`). Our MLIR parser requires text format, so we
/// use JAX's own MLIR infrastructure to deserialize the bytecode.
private enum BytecodeError: Error {
    case conversionFailed(String)
}

private func convertBytecodeToText(_ bytecode: Data) -> Result<String, BytecodeError> {
    let pythonScript = #"""
        import sys, re
        from jax._src.interpreters.mlir import make_ir_context
        from jaxlib.mlir._mlir_libs import _stablehlo
        ctx = make_ir_context()
        bytecode = sys.stdin.buffer.read()
        module = _stablehlo.deserialize_portable_artifact(ctx, bytecode)
        text = str(module)
        # Strip module attributes (our parser doesn't handle them)
        text = re.sub(r'(module @\S+)\s+attributes\s+\{[^}]*\}', r'\1', text)
        # Strip 'public' visibility on func.func
        text = text.replace('func.func public ', 'func.func ')
        # Strip result annotations like {jax.result_info = "..."}
        text = re.sub(r'\{jax\.\w+\s*=\s*"[^"]*"\}', '', text)
        # Strip SDY sharding mesh declarations and annotations
        text = re.sub(r'^\s*sdy\.mesh\s+@\S+\s*=\s*<[^>]*>\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\{sdy\.sharding\s*=\s*#sdy\.sharding<[^>]*>\}', '', text)
        # Strip sdy.sharding_constraint passthrough ops:
        #   %out = sdy.sharding_constraint %in <@mesh, [...]> : type
        # Replace all uses of %out with %in since the op is identity.
        for m in re.finditer(r'^\s*(%\S+)\s*=\s*sdy\.sharding_constraint\s+(%\S+)\s.*$', text, re.MULTILINE):
            text = text.replace(m.group(0), '')  # remove the constraint line
            # Word-boundary-aware replacement: %out followed by non-identifier char
            text = re.sub(re.escape(m.group(1)) + r'(?=[^a-zA-Z0-9_#]|$)', m.group(2), text)
        # Strip precision attributes
        text = re.sub(r',\s*precision_config\s*=\s*\[[^\]]*\]', '', text)
        text = re.sub(r',\s*precision\s*=\s*\[[^\]]*\]', '', text)
        # Convert generic form inherent attributes <{...}> to {...}
        text = text.replace('<{', '{')
        text = text.replace('}>', '}')
        # Convert #stablehlo.gather<...> and #stablehlo.scatter<...> attributes
        # to inline brace form: #stablehlo.gather<body> → {body}
        text = re.sub(r'#stablehlo\.(?:gather|scatter)<([^>]*)>', r'{\1}', text)
        # Convert generic form operations: "stablehlo.op"(%a, %b) <{attrs}> to stablehlo.op %a, %b, attrs
        text = re.sub(r'"stablehlo\.(\w+)"\(([^)]*)\)\s*<', r'stablehlo.\1 \2, ', text)
        # Handle "stablehlo.op"(%a, %b) {attrs} ({region}) : type
        # The {attrs} block contains key=value pairs. For scatter/gather with regions,
        # we need to unwrap the outer {} and leave attrs inline.
        def convert_generic_op_with_attrs(m):
            op = m.group(1)
            operands = m.group(2)
            attrs = m.group(3)
            return f'stablehlo.{op} {operands}, {attrs}'
        # Match: "stablehlo.op"(operands) {attrs-no-nested-braces}
        # Use [^{}]* for attrs without nested braces, handle nested {} for dim_numbers
        text = re.sub(
            r'"stablehlo\.(\w+)"\(([^)]*)\)\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
            convert_generic_op_with_attrs,
            text
        )
        # Strip 'array<i64: ...>' wrapping on slice_sizes (convert to plain list)
        text = re.sub(r'array<i\d+:\s*([^>]*)>', r'[\1]', text)
        # Convert dynamic_slice 'sizes' to 'slice_sizes'
        text = re.sub(r',\s*sizes\s*=\s*\[', ', slice_sizes = [', text)
        # Convert convolution parenthesized operands to standard form
        text = re.sub(r'stablehlo\.convolution\(([^)]+)\)\s*dim_numbers',
                       r'stablehlo.convolution \1, dim_numbers', text)
        text = re.sub(r'\bdim_numbers\s*=', 'dimension_numbers =', text)
        # Expand convolution window= {...} to individual attributes
        def expand_conv_window(m):
            wb = m.group(1)
            r = ''
            s = re.search(r'stride\s*=\s*(\[[^\]]*\])', wb)
            if s: r += ', window_strides = ' + s.group(1)
            p = re.search(r'pad\s*=\s*(\[\[[^\]]*\]\]|\[[^\]]*\])', wb)
            if p: r += ', padding = ' + p.group(1)
            l = re.search(r'lhs_dilate\s*=\s*(\[[^\]]*\])', wb)
            if l: r += ', lhs_dilation = ' + l.group(1)
            d = re.search(r'rhs_dilate\s*=\s*(\[[^\]]*\])', wb)
            if d: r += ', rhs_dilation = ' + d.group(1)
            return r
        text = re.sub(r',\s*window\s*=\s*\{([^}]*)\}', expand_conv_window, text)
        # Inline trailing conv attribute block (use [^}\n]* to avoid crossing lines)
        def inline_conv_attrs(m):
            a = re.sub(r'\s*:\s*i\d+', '', m.group(1))
            return ', ' + a.strip() + ' :'
        text = re.sub(r' \{([^}\n]*(?:batch_group_count|feature_group_count)[^}\n]*)\} :',
                       inline_conv_attrs, text)
        # Clean up extra whitespace from removals
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        sys.stdout.write(text)
        """#

    let process = Process()
    let stdinPipe = Pipe()
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()

    // Find Python — we're running inside a Python process (JAX called us).
    // Use the same Python executable that's running the current process.
    let pythonPath = ProcessInfo.processInfo.arguments[0]
    process.executableURL = URL(fileURLWithPath: pythonPath)
    process.arguments = ["-c", pythonScript]
    process.standardInput = stdinPipe
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    do {
        try process.run()
    } catch {
        return .failure(.conversionFailed("Failed to launch Python: \(error)"))
    }

    // Write bytecode to stdin on a background thread to avoid pipe deadlock.
    // If the subprocess produces enough output to fill the stdout pipe buffer
    // while we're still writing to stdin, both processes would block forever.
    let writeQueue = DispatchQueue(label: "metalhlo.bytecode-write")
    writeQueue.async {
        stdinPipe.fileHandleForWriting.write(bytecode)
        stdinPipe.fileHandleForWriting.closeFile()
    }

    // Read stdout/stderr before waitUntilExit to prevent buffer-full deadlock
    let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
    let errorData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

    process.waitUntilExit()

    guard process.terminationStatus == 0 else {
        let stderr = String(data: errorData, encoding: .utf8) ?? "unknown error"
        return .failure(.conversionFailed("Python conversion failed (exit \(process.terminationStatus)): \(stderr)"))
    }

    guard let text = String(data: outputData, encoding: .utf8), !text.isEmpty else {
        return .failure(.conversionFailed("Python conversion produced empty output"))
    }

    return .success(text)
}

/// Concrete backing storage for opaque PJRT_Executable pointers.
/// Represents a serializable compiled artifact.
final class PJRTExecutableImpl: @unchecked Sendable {
    let name: String
    let cName: UnsafeMutablePointer<CChar>

    /// The original MLIR source, kept for serialization.
    let mlirSource: String

    /// Output element types (stable C array — pointer survives scope).
    let outputTypesBuffer: UnsafeMutableBufferPointer<PJRT_Buffer_Type>

    /// Output dimensions flattened (stable C array).
    let outputDimsBuffer: UnsafeMutableBufferPointer<Int64>

    /// Per-output dimension counts (stable C array).
    let outputDimSizesBuffer: UnsafeMutableBufferPointer<Int>

    /// Per-output memory kind C strings (stable C array).
    let outputMemoryKindsBuffer: UnsafeMutableBufferPointer<UnsafePointer<CChar>?>

    /// Per-output memory kind sizes (stable C array).
    let outputMemoryKindSizesBuffer: UnsafeMutableBufferPointer<Int>

    /// Number of outputs.
    let numOutputs: Int

    /// Serialized representation (cached). Bytes returned from serialize() point into this.
    private var serializedCache: Data?
    private let lock = NSLock()

    // Stable C string for output memory kind
    let cOutputMemoryKind: UnsafeMutablePointer<CChar>

    /// Cached fingerprint C string (lazily computed, freed in deinit).
    private var cachedFingerprint: UnsafeMutablePointer<CChar>?

    init(name: String, mlirSource: String, executable: Executable) {
        self.name = name
        self.cName = strdup(name)!
        self.mlirSource = mlirSource
        self.numOutputs = executable.outputCount
        self.cOutputMemoryKind = strdup("device")!

        // Map output types into stable C arrays
        var types: [PJRT_Buffer_Type] = []
        var dims: [Int64] = []
        var dimSizes: [Int] = []

        for outputType in executable.outputTypes {
            types.append(PJRTBufferImpl.mapElementType(outputType.elementType))
            dimSizes.append(outputType.shape.count)
            for dim in outputType.shape {
                dims.append(Int64(dim))
            }
        }

        // Allocate stable buffers that outlive any withUnsafeBufferPointer scope
        self.outputTypesBuffer = .allocate(capacity: max(types.count, 1))
        for (i, t) in types.enumerated() { outputTypesBuffer[i] = t }

        self.outputDimsBuffer = .allocate(capacity: max(dims.count, 1))
        for (i, d) in dims.enumerated() { outputDimsBuffer[i] = d }

        self.outputDimSizesBuffer = .allocate(capacity: max(dimSizes.count, 1))
        for (i, s) in dimSizes.enumerated() { outputDimSizesBuffer[i] = s }

        // All outputs use "device" memory kind
        self.outputMemoryKindsBuffer = .allocate(capacity: max(numOutputs, 1))
        self.outputMemoryKindSizesBuffer = .allocate(capacity: max(numOutputs, 1))
        let kindLen = strlen(cOutputMemoryKind)
        for i in 0..<numOutputs {
            outputMemoryKindsBuffer[i] = UnsafePointer(cOutputMemoryKind)
            outputMemoryKindSizesBuffer[i] = kindLen
        }
    }

    deinit {
        outputTypesBuffer.deallocate()
        outputDimsBuffer.deallocate()
        outputDimSizesBuffer.deallocate()
        outputMemoryKindsBuffer.deallocate()
        outputMemoryKindSizesBuffer.deallocate()
        free(cName)
        free(cOutputMemoryKind)
        if let fp = cachedFingerprint { free(fp) }
    }

    /// Returns a stable C string for the fingerprint. Owned by this object.
    func getFingerprint() -> UnsafeMutablePointer<CChar> {
        lock.lock()
        defer { lock.unlock() }
        if let existing = cachedFingerprint { return existing }
        var hasher = Hasher()
        hasher.combine(mlirSource)
        let hash = hasher.finalize()
        let fp = strdup(String(hash, radix: 16))!
        cachedFingerprint = fp
        return fp
    }

    func serialize() -> Data {
        lock.lock()
        defer { lock.unlock() }
        if let cached = serializedCache { return cached }

        // Serialize as: [format_version: UInt32][mlir_len: UInt64][mlir_bytes...]
        var data = Data()
        var version: UInt32 = 1
        data.append(Data(bytes: &version, count: 4))

        let mlirData = mlirSource.data(using: .utf8)!
        var mlirLen = UInt64(mlirData.count)
        data.append(Data(bytes: &mlirLen, count: 8))
        data.append(mlirData)

        serializedCache = data
        return data
    }
}

/// Concrete backing storage for opaque PJRT_LoadedExecutable pointers.
/// Holds a loaded executable ready for execution on specific devices.
final class PJRTLoadedExecutableImpl: @unchecked Sendable {
    let executable: Executable
    let executableMeta: PJRTExecutableImpl
    let client: PJRTClientImpl

    private let lock = NSLock()
    private var deleted = false

    // Stable device pointer array
    let devicesPtrBuffer: UnsafeMutableBufferPointer<UnsafeMutablePointer<PJRT_Device>?>

    init(executable: Executable, meta: PJRTExecutableImpl, client: PJRTClientImpl) {
        self.executable = executable
        self.executableMeta = meta
        self.client = client

        self.devicesPtrBuffer = .allocate(capacity: client.devices.count)
        for (i, device) in client.devices.enumerated() {
            devicesPtrBuffer[i] = UnsafeMutablePointer<PJRT_Device>(
                retainAsOpaque(device)
            )
        }
    }

    deinit {
        for i in 0..<devicesPtrBuffer.count {
            if let ptr = devicesPtrBuffer[i] {
                releaseOpaque(OpaquePointer(ptr), as: PJRTDeviceImpl.self)
            }
        }
        devicesPtrBuffer.deallocate()
    }

    var isDeleted: Bool {
        lock.lock()
        defer { lock.unlock() }
        return deleted
    }

    func markDeleted() {
        lock.lock()
        deleted = true
        lock.unlock()
    }
}

// MARK: - Device Policy Resolution

/// Resolves the device policy for PJRT compilation.
///
/// Checks the `METALHLO_DEVICE_POLICY` environment variable first, then
/// defaults to `.auto` (cost-model decides GPU vs ANE partitioning).
///
/// Environment variable values (case-insensitive):
///   - `gpu_only` / `gpuonly` — GPU only, no ANE
///   - `ane_only` / `aneonly` — Force ANE for compatible ops
///   - `auto` — Cost-model decides (default)
///   - `prefer_ane` / `preferane` — Bias toward ANE
///   - `prefer_gpu` / `prefergpu` — Bias toward GPU
private func resolveDevicePolicy() -> DevicePolicy {
    if let env = ProcessInfo.processInfo.environment["METALHLO_DEVICE_POLICY"] {
        switch env.lowercased() {
        case "gpu_only", "gpuonly": return .gpuOnly
        case "ane_only", "aneonly": return .aneOnly
        case "auto": return .auto
        case "prefer_ane", "preferane": return .preferANE
        case "prefer_gpu", "prefergpu": return .preferGPU
        default: break
        }
    }
    return .auto
}

// MARK: - MLIR Call Inlining

/// Inlines trivial JAX wrapper functions at the MLIR text level.
///
/// JAX typically emits a module like:
/// ```
/// module @jit_func {
///   func.func @main(%arg0: T) -> T {
///     %0 = call @actual_func(%arg0) : (T) -> T
///     return %0 : T
///   }
///   func.func private @actual_func(%arg0: T) -> T {
///     // ... actual computation ...
///   }
/// }
/// ```
///
/// This function detects this pattern and replaces the module with just the
/// private function body, renamed to @main. This avoids MPSGraph's internal
/// reshape optimization bugs that occur when compiling graphs with integer
/// tensor operations through the call inlining path.
private func inlineSimpleCallWrapper(_ mlir: String) -> String {
    // Quick check: must have both call and private function
    guard mlir.contains("call @") && mlir.contains("func.func private") else {
        return mlir
    }

    let lines = mlir.split(separator: "\n", omittingEmptySubsequences: false)

    // Find the private function declaration
    var privateFuncStart: Int? = nil
    var privateFuncName: String? = nil
    for (i, line) in lines.enumerated() {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("func.func private @") {
            // Extract function name: "func.func private @name(..."
            let afterAt = trimmed.dropFirst("func.func private @".count)
            if let parenIdx = afterAt.firstIndex(of: "(") {
                privateFuncName = String(afterAt[afterAt.startIndex..<parenIdx])
                privateFuncStart = i
            }
            break
        }
    }

    guard let funcName = privateFuncName, let funcStart = privateFuncStart else {
        return mlir
    }

    // Verify @main is a simple wrapper: its body must contain ONLY a call to @funcName
    // and a return. If @main has other operations (slice, broadcast, reshape, etc.),
    // it's a non-trivial wrapper with a different signature — skip inlining.
    var mainBodyStart: Int? = nil
    var mainBodyEnd: Int? = nil
    for (i, line) in lines.enumerated() {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("func.func @main(") {
            // Find body boundaries (skip the declaration line itself)
            var depth = 0
            for j in i..<lines.count {
                for char in lines[j] {
                    if char == "{" { depth += 1 }
                    if char == "}" { depth -= 1 }
                }
                if j == i { mainBodyStart = j + 1 }
                if depth == 0 { mainBodyEnd = j; break }
            }
            break
        }
    }

    guard let bodyStart = mainBodyStart, let bodyEnd = mainBodyEnd else {
        return mlir
    }

    // Check that all non-empty body lines are either the call or the return
    var hasCall = false
    // hasReturn tracked but not checked (return always present)
    var hasOtherOps = false
    for i in bodyStart..<bodyEnd {
        let trimmed = lines[i].trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty || trimmed == "{" || trimmed == "}" || trimmed.hasPrefix("^bb") {
            continue
        }
        if trimmed.contains("call @\(funcName)") {
            hasCall = true
        } else if trimmed.hasPrefix("return ") || trimmed == "return" {
            continue
        } else {
            hasOtherOps = true
        }
    }

    let debugCompileEarly = ProcessInfo.processInfo.environment["METALHLO_DEBUG_COMPILE"] != nil
    guard hasCall && !hasOtherOps else {
        if debugCompileEarly && hasOtherOps {
            print("[MetalHLO] Skipping inlining of @main -> @\(funcName): @main has non-trivial body")
        }
        return mlir
    }

    // Only inline single-output functions. Multi-output functions (returning tuples)
    // require O2 multi-output support which isn't fully implemented yet.
    let funcDecl = String(lines[funcStart])
    let returnArrow = funcDecl.range(of: "->")
    if let arrowRange = returnArrow {
        let returnPart = funcDecl[arrowRange.upperBound...]
        // Multi-output: "-> (tensor<...>, tensor<...>, ...)"
        if returnPart.contains(",") {
            return mlir
        }
    }

    // Find the private function body (from its declaration to the matching closing brace)
    var braceDepth = 0
    var privateFuncEnd: Int? = nil
    for i in funcStart..<lines.count {
        for char in lines[i] {
            if char == "{" { braceDepth += 1 }
            if char == "}" { braceDepth -= 1 }
        }
        if braceDepth == 0 {
            privateFuncEnd = i
            break
        }
    }

    guard let funcEnd = privateFuncEnd else {
        return mlir
    }

    // Don't inline if the target function contains nested calls to other functions
    // (e.g., jnp.linalg.solve → _lu_solve). These have different input counts.
    for i in (funcStart + 1)..<funcEnd {
        if lines[i].contains("call @") {
            let debugCompile = ProcessInfo.processInfo.environment["METALHLO_DEBUG_COMPILE"] != nil
            if debugCompile {
                print("[MetalHLO] Skipping inlining of @\(funcName): contains nested calls")
            }
            return mlir
        }
    }

    // Extract the module header (first line)
    let moduleHeader = String(lines[0])

    // Find the @main wrapper function boundaries (to exclude it)
    var mainStart: Int? = nil
    for (i, line) in lines.enumerated() {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("func.func @main(") {
            mainStart = i
            break
        }
    }

    // Rebuild module: keep header, replace @main wrapper with the target function
    // renamed to @main, and keep ALL other private functions.
    var result = moduleHeader + "\n"

    // First: emit the target private function renamed to @main
    for i in funcStart...funcEnd {
        var line = String(lines[i])
        if i == funcStart {
            line = line.replacingOccurrences(
                of: "func.func private @\(funcName)",
                with: "func.func @main"
            )
        }
        result += line + "\n"
    }

    // Then: emit all other functions (private functions that aren't the target
    // or the @main wrapper), preserving them for call references
    var i = 0
    while i < lines.count {
        let trimmed = lines[i].trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("func.func") {
            // Find the end of this function
            var depth = 0
            var funcEndIdx = i
            for j in i..<lines.count {
                for char in lines[j] {
                    if char == "{" { depth += 1 }
                    if char == "}" { depth -= 1 }
                }
                if depth == 0 { funcEndIdx = j; break }
            }

            // Skip the @main wrapper and the already-inlined function
            let isMainWrapper = (mainStart != nil && i == mainStart!)
            let isTargetFunc = (i == funcStart)
            if !isMainWrapper && !isTargetFunc {
                for j in i...funcEndIdx {
                    result += String(lines[j]) + "\n"
                }
            }
            i = funcEndIdx + 1
        } else {
            i += 1
        }
    }

    result += "}\n"

    let debugCompile = ProcessInfo.processInfo.environment["METALHLO_DEBUG_COMPILE"] != nil
    if debugCompile {
        print("[MetalHLO] Inlined wrapper: @main -> @\(funcName) "
            + "(\(funcEnd - funcStart + 1) lines, eliminated call)")
    }

    return result
}

// MARK: - While Loop Unrolling

/// Unrolls static while loops (from JAX's `fori_loop`) at the MLIR text level.
///
/// Detects the pattern:
/// ```
/// %r:N = stablehlo.while(%iterArg = %init0, ...) : types...
///   cond { compare LT, %counter, %bound }
///   do { body ops + increment counter }
/// ```
///
/// If the loop bound is a known constant and the counter starts at 0 with
/// increment 1, unrolls the loop by replicating the body N times with
/// appropriate variable renaming.
private func unrollStaticWhileLoops(_ mlir: String) -> String {
    let debugCompile = ProcessInfo.processInfo.environment["METALHLO_DEBUG_COMPILE"] != nil

    // Quick check
    guard mlir.contains("stablehlo.while") else { return mlir }

    let lines = mlir.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)

    // Find all constant definitions: %name = stablehlo.constant dense<VALUE> : tensor<i32>
    // These are needed to resolve loop bounds.
    var intConstants: [String: Int] = [:]
    for line in lines {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        // Match: %name = stablehlo.constant dense<N> : tensor<i32>
        // Also match: %name = stablehlo.constant dense<N> : tensor<i64>
        if trimmed.contains("stablehlo.constant dense<"),
           trimmed.contains(": tensor<i") {
            // Extract name
            if let eqRange = trimmed.range(of: " = ") {
                let name = String(trimmed[trimmed.startIndex..<eqRange.lowerBound])
                // Extract value from dense<VALUE>
                if let openAngle = trimmed.range(of: "dense<"),
                   let closeAngle = trimmed[openAngle.upperBound...].firstIndex(of: ">") {
                    let valStr = String(trimmed[openAngle.upperBound..<closeAngle])
                    if let val = Int(valStr) {
                        intConstants[name] = val
                    }
                }
            }
        }
    }

    // Find the while loop start
    var whileLineIdx: Int? = nil
    for (i, line) in lines.enumerated() {
        if line.contains("stablehlo.while(") || line.contains("stablehlo.while %") {
            whileLineIdx = i
            break
        }
    }

    guard let whileStart = whileLineIdx else { return mlir }

    // Parse the while loop header to get result name and count
    // Pattern: %name:count = stablehlo.while(...) or %name = stablehlo.while(...)
    let whileLine = lines[whileStart].trimmingCharacters(in: .whitespaces)

    // Extract result variable and count
    guard let eqRange = whileLine.range(of: " = ") else {
        if debugCompile { print("[MetalHLO] While unroll: can't find '=' in while line") }
        return mlir
    }

    let resultPart = String(whileLine[whileLine.startIndex..<eqRange.lowerBound])
        .trimmingCharacters(in: .whitespaces)
    var resultName: String
    var resultCount: Int

    if let colonIdx = resultPart.lastIndex(of: ":"),
       colonIdx > resultPart.startIndex,
       let count = Int(String(resultPart[resultPart.index(after: colonIdx)...])) {
        resultName = String(resultPart[resultPart.startIndex..<colonIdx])
        resultCount = count
    } else {
        resultName = resultPart
        resultCount = 1
    }

    // Parse initial values: %iterArg = %init0, %iterArg_1 = %init1, ...
    // These appear within stablehlo.while(...)
    // The while(...) may span multiple lines
    var iterArgs: [(argName: String, initValue: String)] = []
    var whileHeaderEnd = whileStart

    // Collect all text from "stablehlo.while(" to the matching ")"
    var parenDepth = 0
    var headerText = ""
    var foundWhileOpen = false
    var foundWhileClose = false
    for i in whileStart..<lines.count {
        for char in lines[i] {
            if foundWhileOpen {
                if char == "(" { parenDepth += 1 }
                if char == ")" {
                    if parenDepth == 0 {
                        whileHeaderEnd = i
                        foundWhileClose = true
                        break
                    }
                    parenDepth -= 1
                }
                headerText.append(char)
            }
            if !foundWhileOpen && char == "(" {
                foundWhileOpen = true
            }
        }
        if foundWhileClose { break }
    }

    // Parse iterArg assignments from headerText
    // Format: %iterArg = %init0, %iterArg_1 = %init1, ...
    let argPairs = headerText.components(separatedBy: ",")
    for pair in argPairs {
        let trimmed = pair.trimmingCharacters(in: .whitespaces)
        if let eqR = trimmed.range(of: " = ") {
            let argName = String(trimmed[trimmed.startIndex..<eqR.lowerBound])
                .trimmingCharacters(in: .whitespaces)
            let initVal = String(trimmed[eqR.upperBound...])
                .trimmingCharacters(in: .whitespaces)
            if argName.hasPrefix("%") && initVal.hasPrefix("%") {
                iterArgs.append((argName: argName, initValue: initVal))
            }
        }
    }

    guard !iterArgs.isEmpty else {
        if debugCompile { print("[MetalHLO] While unroll: no iterArgs found") }
        return mlir
    }

    // Find the cond block, do block, and the overall while end
    // Structure: ... cond { ... } do { ... }
    var condStart: Int? = nil
    var condEnd: Int? = nil
    var doStart: Int? = nil
    var doEnd: Int? = nil

    // Scan from after the while header for "cond {" and "} do {"
    var braceDepth = 0
    var inCond = false
    var inDo = false

    for i in (whileHeaderEnd)..<lines.count {
        let trimmed = lines[i].trimmingCharacters(in: .whitespaces)

        if !inCond && !inDo {
            if trimmed == "cond {" || trimmed.hasSuffix("cond {") {
                condStart = i
                inCond = true
                braceDepth = 1
                continue
            }
        }

        if inCond {
            // Special case: "} do {" on one line closes cond and opens do
            if trimmed == "} do {" || trimmed.hasSuffix("} do {") {
                condEnd = i
                inCond = false
                doStart = i
                inDo = true
                braceDepth = 1
                continue
            }
            for char in lines[i] {
                if char == "{" { braceDepth += 1 }
                if char == "}" { braceDepth -= 1 }
            }
            if i != condStart && braceDepth == 0 {
                condEnd = i
                inCond = false
                continue
            }
        }

        if !inCond && !inDo && condEnd != nil {
            if trimmed == "} do {" || trimmed == "do {" || trimmed.hasSuffix("do {") {
                doStart = i
                inDo = true
                braceDepth = 1
                continue
            }
        }

        if inDo {
            for char in lines[i] {
                if char == "{" { braceDepth += 1 }
                if char == "}" { braceDepth -= 1 }
            }
            if i != doStart && braceDepth == 0 {
                doEnd = i
                break
            }
        }
    }

    guard let cStart = condStart, let cEnd = condEnd,
          let dStart = doStart, let dEnd = doEnd else {
        if debugCompile { print("[MetalHLO] While unroll: can't find cond/do blocks") }
        return mlir
    }

    // Parse the condition block to find: compare LT, %counter, %bound
    var counterArgName: String? = nil
    var boundValue: Int? = nil

    for i in (cStart + 1)..<cEnd {
        let trimmed = lines[i].trimmingCharacters(in: .whitespaces)
        // Match: %cond = stablehlo.compare LT, %counterArg, %boundVar, SIGNED : ...
        // or: stablehlo.compare %counterArg, %boundVar, LT : ...
        if trimmed.contains("stablehlo.compare") && trimmed.contains("LT") {
            // Extract the two operands being compared
            // Pattern 1: stablehlo.compare LT, %a, %b, SIGNED
            // Pattern 2: stablehlo.compare %a, %b, LT
            let parts = trimmed.components(separatedBy: ",")
            for (_, part) in parts.enumerated() {
                let p = part.trimmingCharacters(in: .whitespaces)
                if p.hasPrefix("%") {
                    // This is an operand — could be counter or bound
                    let varName = "%" + p.dropFirst().prefix(while: { $0.isLetter || $0.isNumber || $0 == "_" })
                    // Check if it's an iterArg (counter) or a constant (bound)
                    if iterArgs.contains(where: { $0.argName == String(varName) }) {
                        counterArgName = String(varName)
                    } else if let constVal = intConstants[String(varName)] {
                        boundValue = constVal
                    }
                }
            }
        }
    }

    guard let counterArg = counterArgName, let loopBound = boundValue else {
        if debugCompile {
            print("[MetalHLO] While unroll: can't determine counter/bound "
                + "(counter=\(counterArgName ?? "nil"), bound=\(boundValue.map(String.init) ?? "nil"))")
        }
        return mlir
    }

    // Verify the counter starts at 0 (its init value should be a constant 0)
    guard let counterIdx = iterArgs.firstIndex(where: { $0.argName == counterArg }) else {
        if debugCompile { print("[MetalHLO] While unroll: counter not in iterArgs") }
        return mlir
    }
    let counterInitVar = iterArgs[counterIdx].initValue
    if let initVal = intConstants[counterInitVar], initVal != 0 {
        if debugCompile { print("[MetalHLO] While unroll: counter doesn't start at 0 (starts at \(initVal))") }
        return mlir
    }
    // If we can't find the constant, check if it's 0 — if unknown, bail
    if intConstants[counterInitVar] == nil {
        if debugCompile { print("[MetalHLO] While unroll: can't resolve counter init value \(counterInitVar)") }
        return mlir
    }

    // Safety: don't unroll very large loops
    guard loopBound <= 1000 else {
        if debugCompile { print("[MetalHLO] While unroll: loop bound \(loopBound) too large") }
        return mlir
    }

    // Parse the do-body to find the stablehlo.return and the body operations
    var doBodyLines: [String] = []
    var returnLine: String? = nil
    for i in (dStart + 1)..<dEnd {
        let trimmed = lines[i].trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty || trimmed.hasPrefix("^bb") { continue }
        if trimmed.hasPrefix("stablehlo.return ") {
            returnLine = trimmed
        } else {
            doBodyLines.append(trimmed)
        }
    }

    guard let retLine = returnLine else {
        if debugCompile { print("[MetalHLO] While unroll: no stablehlo.return in do body") }
        return mlir
    }

    // Parse the return values: stablehlo.return %a, %b, %c : types...
    let retBodyStr: String
    if let colonRange = retLine.range(of: " : ") {
        let retStart = retLine.index(retLine.startIndex, offsetBy: "stablehlo.return ".count)
        retBodyStr = String(retLine[retStart..<colonRange.lowerBound])
    } else {
        retBodyStr = String(retLine.dropFirst("stablehlo.return ".count))
    }
    let returnValues = retBodyStr.components(separatedBy: ",").map {
        $0.trimmingCharacters(in: .whitespaces)
    }

    guard returnValues.count == iterArgs.count else {
        if debugCompile {
            print("[MetalHLO] While unroll: return count (\(returnValues.count)) != iterArgs count (\(iterArgs.count))")
        }
        return mlir
    }

    // Verify counter is incremented by 1: look for stablehlo.add %counter, %one
    var counterIncrementFound = false
    for bodyLine in doBodyLines {
        if bodyLine.contains("stablehlo.add") && bodyLine.contains(counterArg) {
            counterIncrementFound = true
            break
        }
    }

    if !counterIncrementFound {
        if debugCompile { print("[MetalHLO] While unroll: no counter increment found") }
        return mlir
    }

    if debugCompile {
        print("[MetalHLO] While unroll: unrolling \(loopBound) iterations "
            + "(counter=\(counterArg), \(iterArgs.count) iterArgs, \(doBodyLines.count) body lines)")
    }

    // Now generate the unrolled code.
    // For each iteration, we:
    // 1. Create prefixed versions of all body SSA values
    // 2. Substitute iterArg references with current values
    // 3. Track the return values for the next iteration

    // Build the mapping: iterArg -> current value (starts with init values)
    var currentValues: [String: String] = [:]
    for arg in iterArgs {
        currentValues[arg.argName] = arg.initValue
    }

    // Pre-compute body definitions and their expanded use-references.
    // For multi-result defs like %result:3, we need to rename %result#0, %result#1, etc.
    struct BodyDef {
        let rawDef: String       // e.g., "%result:3" or "%foo"
        let baseName: String     // e.g., "%result" or "%foo"
        let count: Int           // e.g., 3 or 1
    }
    var bodyDefs: [BodyDef] = []
    for bodyLine in doBodyLines {
        if let eqR = bodyLine.range(of: " = ") {
            let defVar = String(bodyLine[bodyLine.startIndex..<eqR.lowerBound])
                .trimmingCharacters(in: .whitespaces)
            if defVar.hasPrefix("%") {
                let (baseName, count) = parseMultiResultDef(defVar)
                bodyDefs.append(BodyDef(rawDef: defVar, baseName: baseName, count: count))
            }
        }
    }

    var unrolledLines: [String] = []

    for iter in 0..<loopBound {
        let prefix = "iter\(iter)"

        // For each body line, rename SSA values and substitute iterArgs
        for bodyLine in doBodyLines {
            var line = bodyLine

            // First: rename the SSA definition on this line with iteration prefix
            if let eqR = line.range(of: " = ") {
                let defVar = String(line[line.startIndex..<eqR.lowerBound])
                    .trimmingCharacters(in: .whitespaces)
                if defVar.hasPrefix("%") {
                    let (baseName, count) = parseMultiResultDef(defVar)
                    let newBaseName = "%\(prefix)_\(baseName.dropFirst())"
                    let newDef = count > 1 ? "\(newBaseName):\(count)" : newBaseName
                    line = line.replacingOccurrences(of: defVar + " = ", with: newDef + " = ")
                }
            }

            // Substitute iterArg references with current values
            for (argName, currentVal) in currentValues {
                line = replaceSSAVariable(in: line, variable: argName, with: currentVal)
            }

            // Replace body-internal SSA references with prefixed versions.
            // For multi-result defs, replace each individual result ref (%name#i).
            for def in bodyDefs {
                if def.count > 1 {
                    // Multi-result: replace %name#0, %name#1, ... with prefixed versions
                    for idx in 0..<def.count {
                        let oldRef = "\(def.baseName)#\(idx)"
                        let newRef = "%\(prefix)_\(def.baseName.dropFirst())#\(idx)"
                        line = replaceSSAVariable(in: line, variable: oldRef, with: newRef)
                    }
                } else {
                    let newRef = "%\(prefix)_\(def.baseName.dropFirst())"
                    line = replaceSSAVariable(in: line, variable: def.baseName, with: newRef)
                }
            }

            unrolledLines.append("    " + line)
        }

        // Update currentValues based on return mapping.
        // Each return value corresponds to an iterArg.
        var newValues: [String: String] = [:]
        for (argIdx, arg) in iterArgs.enumerated() {
            var retVal = returnValues[argIdx]

            // The return value might reference a body-local variable that was prefixed.
            // For multi-result defs, match individual result refs (%name#i).
            for def in bodyDefs {
                if def.count > 1 {
                    for idx in 0..<def.count {
                        let oldRef = "\(def.baseName)#\(idx)"
                        if retVal == oldRef {
                            retVal = "%\(prefix)_\(def.baseName.dropFirst())#\(idx)"
                        }
                    }
                } else if retVal == def.baseName {
                    retVal = "%\(prefix)_\(def.baseName.dropFirst())"
                }
            }

            // It might also be an iterArg reference (loop-invariant values)
            if let curVal = currentValues[retVal] {
                retVal = curVal
            }
            newValues[arg.argName] = retVal
        }
        currentValues = newValues
    }

    // Build the final replacement: map %resultName#i to the final values
    var resultMapping: [String: String] = [:]
    for (i, arg) in iterArgs.enumerated() {
        let resultRef = resultCount > 1 ? "\(resultName)#\(i)" : resultName
        if let finalVal = currentValues[arg.argName] {
            resultMapping[resultRef] = finalVal
        }
    }

    // Rebuild the MLIR: replace the while loop with unrolled code, then
    // substitute result references in subsequent lines
    var newLines: [String] = []

    // Add lines before the while loop
    for i in 0..<whileStart {
        newLines.append(lines[i])
    }

    // Add unrolled body
    for line in unrolledLines {
        newLines.append(line)
    }

    // Find the actual end of the while construct (after the do block closing brace)
    // The while might have a trailing type annotation on the same or next line
    var whileEnd = dEnd
    // Check if there's a trailing ": type" on the closing line or next lines
    if whileEnd + 1 < lines.count {
        let nextTrimmed = lines[whileEnd + 1].trimmingCharacters(in: .whitespaces)
        if nextTrimmed.hasPrefix(": ") || nextTrimmed.hasPrefix(") :") {
            whileEnd += 1
        }
    }

    // Add lines after the while loop, with result substitution
    for i in (whileEnd + 1)..<lines.count {
        var line = lines[i]
        for (resultRef, finalVal) in resultMapping {
            line = replaceSSAVariable(in: line, variable: resultRef, with: finalVal)
        }
        newLines.append(line)
    }

    let result = newLines.joined(separator: "\n")

    if debugCompile {
        print("[MetalHLO] While unroll: successfully unrolled \(loopBound) iterations, "
            + "replaced \(resultMapping.count) result references")
    }

    return result
}

/// Replaces an SSA variable reference in a line, respecting word boundaries.
///
/// SSA variables in MLIR are like `%foo`, `%foo_1`, `%foo#0`. We need to replace
/// `%foo` without accidentally replacing `%foo_1` or `%foobar`.
/// For variables with `#N` suffixes (like `%result#1`), we must not match
/// inside `%result#10` — the character after must not be a digit.
private func replaceSSAVariable(in line: String, variable: String, with replacement: String) -> String {
    var result = ""
    var remaining = line[line.startIndex...]

    while let range = remaining.range(of: variable) {
        let afterIdx = range.upperBound
        if afterIdx < remaining.endIndex {
            let nextChar = remaining[afterIdx]
            // Must not be followed by alphanumeric, underscore, or '#'.
            // - Letters/digits/underscore: prevents %foo matching %foo_1 or %foobar
            // - '#': prevents %foo matching inside %foo#0 (multi-result ref)
            // - Digits after '#': prevents %result#1 matching inside %result#10
            if nextChar.isLetter || nextChar.isNumber || nextChar == "_" || nextChar == "#" {
                // Not a word boundary — skip this occurrence
                result += String(remaining[remaining.startIndex..<remaining.index(after: range.lowerBound)])
                remaining = remaining[remaining.index(after: range.lowerBound)...]
                continue
            }
        }

        result += String(remaining[remaining.startIndex..<range.lowerBound])
        result += replacement
        remaining = remaining[range.upperBound...]
    }

    result += String(remaining)
    return result
}

/// Parses a multi-result SSA definition like `%result:3` into base name and count.
/// Returns `(baseName, count)` where baseName is `%result` and count is 3.
/// For single-result definitions like `%foo`, returns `("%foo", 1)`.
private func parseMultiResultDef(_ def: String) -> (baseName: String, count: Int) {
    // Multi-result: %name:N where N is the result count
    if let colonIdx = def.lastIndex(of: ":"),
       colonIdx > def.startIndex,
       let count = Int(String(def[def.index(after: colonIdx)...])),
       count > 1 {
        return (String(def[def.startIndex..<colonIdx]), count)
    }
    return (def, 1)
}

/// Expands a multi-result definition into individual result references.
/// E.g., `%result:3` → `[("%result#0", "%result"), ("%result#1", "%result"), ("%result#2", "%result")]`
/// For single results, returns `[("%foo", "%foo")]`.
/// Each tuple is `(useRef, defBaseName)` — the `useRef` is what appears in code.
private func expandMultiResultDef(_ def: String) -> [(useRef: String, baseName: String)] {
    let (baseName, count) = parseMultiResultDef(def)
    if count > 1 {
        return (0..<count).map { i in ("\(baseName)#\(i)", baseName) }
    }
    return [(baseName, baseName)]
}

// MARK: - Function Call Inlining in Body

/// Inlines private function calls within a function body at the MLIR text level.
///
/// After while loop unrolling, the @main function body may contain
/// `func.call @some_func(...)` calls. This function finds the called private
/// function definition and inlines its body at each call site.
private func inlineCallsInBody(_ mlir: String) -> String {
    let debugCompile = ProcessInfo.processInfo.environment["METALHLO_DEBUG_COMPILE"] != nil

    // Quick check: need both a call site and a private function definition.
    // Calls may appear as "func.call @name" or just "call @name" depending on MLIR form.
    guard mlir.contains("call @") && mlir.contains("func.func private @") else {
        return mlir
    }

    let lines = mlir.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)

    // Find all private function definitions
    struct PrivateFunc {
        let name: String
        let paramNames: [String]  // %arg0, %arg1, ...
        let startLine: Int
        let endLine: Int
        let bodyLines: [String]   // Lines between { and }, excluding ^bb and return
        let returnValues: [String] // Values in the final return statement
    }

    var privateFuncs: [String: PrivateFunc] = [:]

    var i = 0
    while i < lines.count {
        let trimmed = lines[i].trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("func.func private @") {
            let afterAt = trimmed.dropFirst("func.func private @".count)
            guard let parenOpen = afterAt.firstIndex(of: "(") else { i += 1; continue }
            let funcName = String(afterAt[afterAt.startIndex..<parenOpen])

            // Extract parameter names
            let afterParen = afterAt[afterAt.index(after: parenOpen)...]
            var paramNames: [String] = []
            let paramStr: String
            if let parenClose = afterParen.firstIndex(of: ")") {
                paramStr = String(afterParen[afterParen.startIndex..<parenClose])
            } else {
                paramStr = String(afterParen)
            }
            // Parse: %arg0: tensor<...>, %arg1: tensor<...>
            for param in paramStr.components(separatedBy: ",") {
                let p = param.trimmingCharacters(in: .whitespaces)
                if p.hasPrefix("%") {
                    if let colonIdx = p.firstIndex(of: ":") {
                        paramNames.append(String(p[p.startIndex..<colonIdx]).trimmingCharacters(in: .whitespaces))
                    } else {
                        paramNames.append(p)
                    }
                }
            }

            // Find function body
            var depth = 0
            var funcEndLine = i
            for j in i..<lines.count {
                for char in lines[j] {
                    if char == "{" { depth += 1 }
                    if char == "}" { depth -= 1 }
                }
                if depth == 0 { funcEndLine = j; break }
            }

            // Extract body lines and return values
            var bodyLines: [String] = []
            var returnVals: [String] = []
            for j in (i + 1)..<funcEndLine {
                let btrimmed = lines[j].trimmingCharacters(in: .whitespaces)
                if btrimmed.isEmpty || btrimmed.hasPrefix("^bb") || btrimmed == "{" || btrimmed == "}" { continue }
                if btrimmed.hasPrefix("return ") || btrimmed.hasPrefix("stablehlo.return ") {
                    // Parse return values
                    let retKeyword = btrimmed.hasPrefix("return ") ? "return " : "stablehlo.return "
                    let afterRet = String(btrimmed.dropFirst(retKeyword.count))
                    let retBody: String
                    if let colonR = afterRet.range(of: " : ") {
                        retBody = String(afterRet[afterRet.startIndex..<colonR.lowerBound])
                    } else {
                        retBody = afterRet
                    }
                    returnVals = retBody.components(separatedBy: ",").map {
                        $0.trimmingCharacters(in: .whitespaces)
                    }
                } else {
                    bodyLines.append(btrimmed)
                }
            }

            // Don't inline functions that contain while loops (control flow
            // that can't be flattened). Functions with nested calls ARE allowed —
            // the convergence loop will inline them level by level.
            let hasWhile = bodyLines.contains { $0.contains("stablehlo.while") }
            if !hasWhile {
                privateFuncs[funcName] = PrivateFunc(
                    name: funcName,
                    paramNames: paramNames,
                    startLine: i,
                    endLine: funcEndLine,
                    bodyLines: bodyLines,
                    returnValues: returnVals
                )
            }

            i = funcEndLine + 1
        } else {
            i += 1
        }
    }

    guard !privateFuncs.isEmpty else { return mlir }

    // Find call sites in @main's body ONLY and inline them
    // (Don't inline calls inside other private functions — they may have
    // different structures like while loops that we shouldn't modify.)
    var mainBodyStart: Int? = nil
    var mainBodyEnd: Int? = nil
    for (idx, line) in lines.enumerated() {
        let t = line.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("func.func") && t.contains("@main(") {
            var depth = 0
            for j in idx..<lines.count {
                for char in lines[j] {
                    if char == "{" { depth += 1 }
                    if char == "}" { depth -= 1 }
                }
                if depth == 0 { mainBodyEnd = j; break }
            }
            mainBodyStart = idx + 1
            break
        }
    }

    var newLines: [String] = []
    var callsInlined = 0
    var inlinedFunctions: Set<String> = []

    i = 0
    while i < lines.count {
        let trimmed = lines[i].trimmingCharacters(in: .whitespaces)

        // Only inline calls within @main's body
        let insideMain = (mainBodyStart != nil && mainBodyEnd != nil
            && i >= mainBodyStart! && i < mainBodyEnd!)

        // Check if this line is a call to a private function we can inline
        // Pattern: %result:N = func.call @name(%arg0, %arg1) : (types) -> (types)
        // or: %result = func.call @name(...) : ...
        // or: func.call @name(...) : ... (no result)
        var matched = false
        for (funcName, funcDef) in privateFuncs {
            if insideMain && trimmed.contains("call @\(funcName)(") {
                matched = true

                // Extract result variable(s)
                var resultVar: String? = nil
                var resultCount = 1
                if let eqR = trimmed.range(of: " = ") {
                    let lhs = String(trimmed[trimmed.startIndex..<eqR.lowerBound])
                        .trimmingCharacters(in: .whitespaces)
                    if let colonIdx = lhs.lastIndex(of: ":"),
                       colonIdx > lhs.startIndex,
                       let count = Int(String(lhs[lhs.index(after: colonIdx)...])) {
                        resultVar = String(lhs[lhs.startIndex..<colonIdx])
                        resultCount = count
                    } else {
                        resultVar = lhs
                        resultCount = 1
                    }
                }

                // Extract call arguments
                let callStr = trimmed
                var callArgs: [String] = []
                if let openParen = callStr.range(of: "@\(funcName)(") {
                    let afterOpen = callStr[openParen.upperBound...]
                    // Find matching close paren
                    var depth = 1
                    var argStr = ""
                    for char in afterOpen {
                        if char == "(" { depth += 1 }
                        if char == ")" {
                            depth -= 1
                            if depth == 0 { break }
                        }
                        argStr.append(char)
                    }
                    callArgs = argStr.components(separatedBy: ",").map {
                        $0.trimmingCharacters(in: .whitespaces)
                    }
                }

                // Verify argument count matches
                guard callArgs.count == funcDef.paramNames.count else {
                    if debugCompile {
                        print("[MetalHLO] Inline call: arg count mismatch for @\(funcName) "
                            + "(\(callArgs.count) vs \(funcDef.paramNames.count))")
                    }
                    break
                }

                // Generate unique prefix for this inlining
                let inlinePrefix = "inl\(callsInlined)"

                // Build parameter substitution map
                var subst: [String: String] = [:]
                for (paramName, argValue) in zip(funcDef.paramNames, callArgs) {
                    subst[paramName] = argValue
                }

                // Collect SSA definitions in the body for prefixing.
                // For multi-result defs (%name:N), expand into individual refs.
                struct InlineBodyDef {
                    let rawDef: String
                    let baseName: String
                    let count: Int
                }
                var bodyDefs: [InlineBodyDef] = []
                for bodyLine in funcDef.bodyLines {
                    if let eqR = bodyLine.range(of: " = ") {
                        let def = String(bodyLine[bodyLine.startIndex..<eqR.lowerBound])
                            .trimmingCharacters(in: .whitespaces)
                        if def.hasPrefix("%") {
                            let (baseName, count) = parseMultiResultDef(def)
                            bodyDefs.append(InlineBodyDef(rawDef: def, baseName: baseName, count: count))
                        }
                    }
                }

                // Emit inlined body
                for bodyLine in funcDef.bodyLines {
                    var line = bodyLine

                    // Rename the definition on this line
                    if let eqR = line.range(of: " = ") {
                        let defVar = String(line[line.startIndex..<eqR.lowerBound])
                            .trimmingCharacters(in: .whitespaces)
                        if defVar.hasPrefix("%") {
                            let (baseName, count) = parseMultiResultDef(defVar)
                            let newBaseName = "%\(inlinePrefix)_\(baseName.dropFirst())"
                            let newDef = count > 1 ? "\(newBaseName):\(count)" : newBaseName
                            line = line.replacingOccurrences(of: defVar + " = ", with: newDef + " = ")
                        }
                    }

                    // Prefix all body-local references (uses of defs from other lines)
                    for def in bodyDefs {
                        if def.count > 1 {
                            for idx in 0..<def.count {
                                let oldRef = "\(def.baseName)#\(idx)"
                                let newRef = "%\(inlinePrefix)_\(def.baseName.dropFirst())#\(idx)"
                                line = replaceSSAVariable(in: line, variable: oldRef, with: newRef)
                            }
                        } else {
                            let newName = "%\(inlinePrefix)_\(def.baseName.dropFirst())"
                            line = replaceSSAVariable(in: line, variable: def.baseName, with: newName)
                        }
                    }

                    // Substitute parameters with call arguments
                    for (paramName, argValue) in subst {
                        line = replaceSSAVariable(in: line, variable: paramName, with: argValue)
                    }

                    newLines.append("    " + line)
                }

                // Map result references to the function's return values
                if let rv = resultVar {
                    // Build result mapping for post-processing
                    // We need to replace %result#i references in lines after this call
                    var resultMap: [(String, String)] = []
                    for (retIdx, retVal) in funcDef.returnValues.enumerated() {
                        var mappedVal = retVal
                        // Apply same renaming to return values
                        for def in bodyDefs {
                            if def.count > 1 {
                                for idx in 0..<def.count {
                                    let oldRef = "\(def.baseName)#\(idx)"
                                    let newRef = "%\(inlinePrefix)_\(def.baseName.dropFirst())#\(idx)"
                                    mappedVal = replaceSSAVariable(in: mappedVal, variable: oldRef, with: newRef)
                                }
                            } else {
                                let newName = "%\(inlinePrefix)_\(def.baseName.dropFirst())"
                                mappedVal = replaceSSAVariable(in: mappedVal, variable: def.baseName, with: newName)
                            }
                        }
                        for (paramName, argValue) in subst {
                            mappedVal = replaceSSAVariable(in: mappedVal, variable: paramName, with: argValue)
                        }

                        let refStr = resultCount > 1 ? "\(rv)#\(retIdx)" : rv
                        resultMap.append((refStr, mappedVal))
                    }

                    // Apply result mapping to all previously added lines AND track for future lines
                    // We need to apply to future lines as we add them
                    // Store as a global mapping
                    // Actually, let's just collect all lines first, then do a final pass
                    // Mark these result mappings for the final pass
                    for (refStr, mappedVal) in resultMap {
                        // Tag the newLines array end so we can apply in final pass
                        newLines.append("// __INLINE_MAP__ \(refStr) -> \(mappedVal)")
                    }
                }

                inlinedFunctions.insert(funcName)
                callsInlined += 1
                break
            }
        }

        if !matched {
            newLines.append(lines[i])
        }
        i += 1
    }

    guard callsInlined > 0 else { return mlir }

    // Post-process: extract inline maps and apply substitutions
    var inlineMaps: [(String, String)] = []
    var cleanLines: [String] = []
    for line in newLines {
        if line.trimmingCharacters(in: .whitespaces).hasPrefix("// __INLINE_MAP__") {
            // Parse: // __INLINE_MAP__ %ref -> %val
            let parts = line.components(separatedBy: " -> ")
            if parts.count == 2 {
                let ref = parts[0].components(separatedBy: " ").last ?? ""
                let val = parts[1].trimmingCharacters(in: .whitespaces)
                inlineMaps.append((ref, val))
            }
        } else {
            cleanLines.append(line)
        }
    }

    // Apply inline maps to all lines
    var finalLines: [String] = []
    for line in cleanLines {
        var l = line
        for (ref, val) in inlineMaps {
            l = replaceSSAVariable(in: l, variable: ref, with: val)
        }
        finalLines.append(l)
    }

    // Remove the bodies of inlined functions (they span multiple lines)
    var result: [String] = []
    var skipDepth = 0
    var skipping = false
    for line in finalLines {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if !skipping {
            var shouldSkip = false
            for funcName in inlinedFunctions {
                if trimmed.hasPrefix("func.func private @\(funcName)") {
                    shouldSkip = true
                    break
                }
            }
            if shouldSkip {
                skipping = true
                skipDepth = 0
                for char in line {
                    if char == "{" { skipDepth += 1 }
                    if char == "}" { skipDepth -= 1 }
                }
                if skipDepth == 0 { skipping = false }
                continue
            }
        }
        if skipping {
            for char in line {
                if char == "{" { skipDepth += 1 }
                if char == "}" { skipDepth -= 1 }
            }
            if skipDepth <= 0 { skipping = false }
            continue
        }
        result.append(line)
    }

    if debugCompile {
        print("[MetalHLO] Inlined \(callsInlined) call(s) to "
            + "\(inlinedFunctions.joined(separator: ", ")), removed function definitions")
    }

    return result.joined(separator: "\n")
}

// MARK: - Client Compile

func pjrt_client_compile(
    _ args: UnsafeMutablePointer<PJRT_Client_Compile_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args or client")
    }
    guard let program = args.pointee.program else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL program")
    }

    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)

    // Extract MLIR source from the PJRT_Program
    let prog = program.pointee
    guard let codePtr = prog.code else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL program code")
    }
    let codeSize = prog.code_size

    // For MLIR bytecode, check magic bytes "ML\xefR"
    let rawBytes = UnsafeRawBufferPointer(start: codePtr, count: codeSize)
    let isBytecode = codeSize >= 4
        && rawBytes[0] == 0x4D  // 'M'
        && rawBytes[1] == 0x4C  // 'L'
        && rawBytes[2] == 0xEF
        && rawBytes[3] == 0x52  // 'R'

    var mlirSource: String
    if isBytecode {
        // JAX sends MLIR bytecode; convert to text via Python helper
        switch convertBytecodeToText(Data(rawBytes)) {
        case .success(let text):
            mlirSource = text
        case .failure(let error):
            return makeError(PJRT_Error_Code_INTERNAL, "Bytecode conversion failed: \(error)")
        }
    } else {
        mlirSource = String(decoding: rawBytes, as: UTF8.self)
    }

    // Debug: save MLIR source to file when METALHLO_DUMP_MLIR is set
    if let dumpDir = ProcessInfo.processInfo.environment["METALHLO_DUMP_MLIR"] {
        let lineCount = mlirSource.components(separatedBy: "\n").count
        let filename = "\(dumpDir)/mlir_\(lineCount)_\(ProcessInfo.processInfo.systemUptime).txt"
        try? mlirSource.write(toFile: filename, atomically: true, encoding: .utf8)
    }

    // Step 1: Inline simple call wrappers (@main -> @private_func)
    mlirSource = inlineSimpleCallWrapper(mlirSource)
    // Step 2: Unroll static while loops (from fori_loop)
    mlirSource = unrollStaticWhileLoops(mlirSource)
    // Step 3: Inline remaining private function calls exposed by unrolling
    // Run in a loop until convergence (unrolling may expose new call sites)
    var transformChanged = true
    while transformChanged {
        let before = mlirSource
        mlirSource = inlineCallsInBody(mlirSource)
        mlirSource = unrollStaticWhileLoops(mlirSource)
        transformChanged = (mlirSource != before)
    }

    // Debug: dump MLIR after transformations
    if let dumpDir = ProcessInfo.processInfo.environment["METALHLO_DUMP_MLIR"] {
        let lineCount = mlirSource.components(separatedBy: "\n").count
        let filename = "\(dumpDir)/mlir_POST_\(lineCount)_\(ProcessInfo.processInfo.systemUptime).txt"
        try? mlirSource.write(toFile: filename, atomically: true, encoding: .utf8)
    }

    do {
        // After inlining and unrolling, re-check what's needed.
        // The call inliner removes `call @` / `func.func private`, and the
        // unroller removes `stablehlo.while`. After both, needsMPSGraph should
        // be false for functions that were previously routed to MPSGraph only
        // because of fori_loop control flow.
        let hasWhileLoop = mlirSource.contains("stablehlo.while")
        let hasLinAlg = mlirSource.contains("triangular_solve")
            || mlirSource.contains("cholesky")
        let hasCall = mlirSource.contains("call @")
            || mlirSource.contains("func.func private")

        let needsMPSGraph = hasWhileLoop || hasLinAlg || hasCall

        let compileStart = CFAbsoluteTimeGetCurrent()
        let mlirLines = mlirSource.components(separatedBy: "\n").count
        let debugCompile = ProcessInfo.processInfo.environment["METALHLO_DEBUG_COMPILE"] != nil
        if debugCompile {
            print("[MetalHLO] Compiling \(mlirLines) lines, needsMPSGraph=\(needsMPSGraph)")
        }

        let executable: Executable
        if needsMPSGraph {
            executable = try impl.client.compile(mlirSource)
        } else {
            let optLevel: OptimizationLevel
            if let envOpt = ProcessInfo.processInfo.environment["METALHLO_OPT_LEVEL"] {
                switch envOpt {
                case "O0": optLevel = .O0
                case "O1": optLevel = .O1
                case "O3": optLevel = .O3
                default: optLevel = .O2
                }
            } else {
                optLevel = .O2
            }
            let config = CompilationConfig(
                optimizationLevel: optLevel,
                devicePolicy: resolveDevicePolicy()
            )
            if debugCompile {
                print("[MetalHLO] Using O2 path, devicePolicy=\(resolveDevicePolicy())")
            }
            executable = try impl.client.compile(mlirSource, config: config)
        }
        if debugCompile {
            let elapsed = CFAbsoluteTimeGetCurrent() - compileStart
            print("[MetalHLO] Compilation done in \(String(format: "%.3f", elapsed))s")
        }

        if debugCompile {
            print("[MetalHLO] Executable: \(executable.outputCount) outputs, \(executable.inputTypes.count) inputs")
        }

        let meta = PJRTExecutableImpl(
            name: "metalhlo_program",
            mlirSource: mlirSource,
            executable: executable
        )
        let loaded = PJRTLoadedExecutableImpl(
            executable: executable,
            meta: meta,
            client: impl
        )
        args.pointee.executable = UnsafeMutablePointer<PJRT_LoadedExecutable>(
            retainAsOpaque(loaded)
        )
        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "Compilation failed: \(error)")
    }
}

// MARK: - Executable Functions

func pjrt_executable_destroy(
    _ args: UnsafeMutablePointer<PJRT_Executable_Destroy_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else { return nil }
    releaseOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    return nil
}

func pjrt_executable_name(
    _ args: UnsafeMutablePointer<PJRT_Executable_Name_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    args.pointee.executable_name = UnsafePointer(exe.cName)
    args.pointee.executable_name_size = strlen(exe.cName)
    return nil
}

func pjrt_executable_num_replicas(
    _ args: UnsafeMutablePointer<PJRT_Executable_NumReplicas_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.num_replicas = 1
    return nil
}

func pjrt_executable_num_partitions(
    _ args: UnsafeMutablePointer<PJRT_Executable_NumPartitions_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.num_partitions = 1
    return nil
}

func pjrt_executable_num_outputs(
    _ args: UnsafeMutablePointer<PJRT_Executable_NumOutputs_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    args.pointee.num_outputs = exe.numOutputs
    return nil
}

func pjrt_executable_size_of_generated_code(
    _ args: UnsafeMutablePointer<PJRT_Executable_SizeOfGeneratedCodeInBytes_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.size_in_bytes = 0 // Unknown
    return nil
}

func pjrt_executable_get_cost_analysis(
    _ args: UnsafeMutablePointer<PJRT_Executable_GetCostAnalysis_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.num_properties = 0
    args.pointee.properties = nil
    return nil
}

func pjrt_executable_output_memory_kinds(
    _ args: UnsafeMutablePointer<PJRT_Executable_OutputMemoryKinds_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    args.pointee.num_outputs = exe.numOutputs
    args.pointee.memory_kinds = UnsafePointer(exe.outputMemoryKindsBuffer.baseAddress!)
    args.pointee.memory_kind_sizes = UnsafePointer(exe.outputMemoryKindSizesBuffer.baseAddress!)
    return nil
}

func pjrt_executable_optimized_program(
    _ args: UnsafeMutablePointer<PJRT_Executable_OptimizedProgram_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    return unimplementedError("PJRT_Executable_OptimizedProgram")
}

func pjrt_executable_serialize(
    _ args: UnsafeMutablePointer<PJRT_Executable_Serialize_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    let data = exe.serialize()

    // Return a pointer into the cached Data owned by exe.
    // The bytes remain valid as long as the PJRTExecutableImpl is alive.
    data.withUnsafeBytes { src in
        args.pointee.serialized_bytes = src.bindMemory(to: CChar.self).baseAddress
        args.pointee.serialized_bytes_size = src.count
    }
    return nil
}

func pjrt_executable_output_element_types(
    _ args: UnsafeMutablePointer<PJRT_Executable_OutputElementTypes_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    args.pointee.output_types = UnsafeMutablePointer(exe.outputTypesBuffer.baseAddress!)
    args.pointee.num_output_types = exe.numOutputs
    return nil
}

func pjrt_executable_output_dimensions(
    _ args: UnsafeMutablePointer<PJRT_Executable_OutputDimensions_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    args.pointee.num_outputs = exe.numOutputs
    args.pointee.dims = UnsafePointer(exe.outputDimsBuffer.baseAddress!)
    args.pointee.dim_sizes = UnsafePointer(exe.outputDimSizesBuffer.baseAddress!)
    return nil
}

func pjrt_executable_fingerprint(
    _ args: UnsafeMutablePointer<PJRT_Executable_Fingerprint_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let exe = fromOpaque(OpaquePointer(exePtr), as: PJRTExecutableImpl.self)
    let fp = exe.getFingerprint()
    args.pointee.executable_fingerprint = UnsafePointer(fp)
    args.pointee.executable_fingerprint_size = strlen(fp)
    return nil
}

func pjrt_executable_get_compiled_memory_stats(
    _ args: UnsafeMutablePointer<PJRT_Executable_GetCompiledMemoryStats_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    return unimplementedError("PJRT_Executable_GetCompiledMemoryStats")
}

// MARK: - LoadedExecutable Functions

func pjrt_loaded_executable_destroy(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_Destroy_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let exePtr = args.pointee.executable else { return nil }
    releaseOpaque(OpaquePointer(exePtr), as: PJRTLoadedExecutableImpl.self)
    return nil
}

func pjrt_loaded_executable_get_executable(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_GetExecutable_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.loaded_executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)
    args.pointee.executable = UnsafeMutablePointer<PJRT_Executable>(
        retainAsOpaque(loaded.executableMeta)
    )
    return nil
}

func pjrt_loaded_executable_addressable_devices(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_AddressableDevices_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)
    args.pointee.addressable_devices = UnsafePointer(loaded.devicesPtrBuffer.baseAddress)
    args.pointee.num_addressable_devices = loaded.devicesPtrBuffer.count
    return nil
}

func pjrt_loaded_executable_delete(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_Delete_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)
    loaded.markDeleted()
    return nil
}

func pjrt_loaded_executable_is_deleted(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_IsDeleted_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)
    args.pointee.is_deleted = loaded.isDeleted
    return nil
}

func pjrt_loaded_executable_execute(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_Execute_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)

    let numDevices = Int(args.pointee.num_devices)
    let numArgs = Int(args.pointee.num_args)

    // We only support single-device execution
    guard numDevices <= 1 else {
        return makeError(PJRT_Error_Code_UNIMPLEMENTED, "Multi-device execution not supported")
    }

    // Extract input buffers from argument_lists[0]
    var inputs: [Buffer] = []
    if numDevices > 0 && numArgs > 0 {
        guard let argLists = args.pointee.argument_lists, let deviceArgs = argLists[0] else {
            return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL argument list for device 0")
        }
        for i in 0..<numArgs {
            guard let bufPtr = deviceArgs[i] else {
                return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL buffer at argument index \(i)")
            }
            let bufImpl = fromOpaque(OpaquePointer(bufPtr), as: PJRTBufferImpl.self)
            inputs.append(bufImpl.buffer)
        }
    }

    // Execute
    do {
        let outputs = try loaded.executable.execute(inputs)

        // Write output buffers to output_lists[0]
        if let outputLists = args.pointee.output_lists, numDevices > 0,
           let deviceOutputs = outputLists[0] {
            for (i, output) in outputs.enumerated() {
                let device = loaded.client.devices[0]
                let memory = loaded.client.memories[0]
                let bufImpl = PJRTBufferImpl(
                    buffer: output,
                    device: device,
                    memory: memory
                )
                deviceOutputs[i] = UnsafeMutablePointer<PJRT_Buffer>(
                    retainAsOpaque(bufImpl)
                )
            }
        }

        // Create completion event
        if let events = args.pointee.device_complete_events, numDevices > 0 {
            let event = PJRTEventImpl() // Already complete (synchronous execution)
            events[0] = UnsafeMutablePointer<PJRT_Event>(
                retainAsOpaque(event)
            )
        }

        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "Execution failed: \(error)")
    }
}

func pjrt_loaded_executable_get_device_assignment(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_GetDeviceAssignment_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }

    // Serialize a DeviceAssignmentProto for single-device:
    //   replica_count = 1, computation_count = 1,
    //   computation_devices = [{replica_device_ids: [0]}]
    // Protobuf encoding:
    //   field 1 varint 1, field 2 varint 1, field 3 length-delimited {field 1 varint 0}
    let proto: [UInt8] = [0x08, 0x01, 0x10, 0x01, 0x1A, 0x02, 0x08, 0x00]

    // Allocate a copy of the proto bytes that outlives this call.
    // We use malloc so the deleter can simply call free().
    let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: proto.count)
    proto.withUnsafeBufferPointer { src in
        buf.initialize(from: src.baseAddress!, count: src.count)
    }

    args.pointee.serialized_bytes = UnsafePointer<CChar>(OpaquePointer(buf))
    args.pointee.serialized_bytes_size = proto.count
    // Use the buffer pointer itself as the opaque "serialized_device_assignment" handle.
    args.pointee.serialized_device_assignment = OpaquePointer(buf)
    args.pointee.serialized_device_assignment_deleter = { daPtr in
        guard let daPtr = daPtr else { return }
        UnsafeMutableRawPointer(daPtr).deallocate()
    }

    return nil
}

func pjrt_loaded_executable_fingerprint(
    _ args: UnsafeMutablePointer<PJRT_LoadedExecutable_Fingerprint_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let loadedPtr = args.pointee.executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let loaded = fromOpaque(OpaquePointer(loadedPtr), as: PJRTLoadedExecutableImpl.self)
    let fp = loaded.executableMeta.getFingerprint()
    args.pointee.executable_fingerprint = UnsafePointer(fp)
    args.pointee.executable_fingerprint_size = strlen(fp)
    return nil
}

// MARK: - DeserializeAndLoad

func pjrt_executable_deserialize_and_load(
    _ args: UnsafeMutablePointer<PJRT_Executable_DeserializeAndLoad_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args or client")
    }
    guard let bytes = args.pointee.serialized_executable else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL serialized bytes")
    }

    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)
    let size = args.pointee.serialized_executable_size

    let data = Data(bytes: bytes, count: size)

    // Deserialize: [version: UInt32][mlir_len: UInt64][mlir_bytes...]
    guard data.count >= 12 else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "Serialized data too short")
    }

    let version = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) }
    guard version == 1 else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "Unknown serialization version: \(version)")
    }

    let mlirLen = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt64.self) }
    guard data.count >= 12 + Int(mlirLen) else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "Serialized data truncated")
    }

    let mlirData = data.subdata(in: 12..<(12 + Int(mlirLen)))
    guard let mlirSource = String(data: mlirData, encoding: .utf8) else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid MLIR UTF-8")
    }

    // Re-compile
    do {
        let config = CompilationConfig(
            optimizationLevel: .O2,
            devicePolicy: resolveDevicePolicy()
        )
        let executable = try impl.client.compile(mlirSource, config: config)

        let meta = PJRTExecutableImpl(
            name: "metalhlo_deserialized",
            mlirSource: mlirSource,
            executable: executable
        )
        let loaded = PJRTLoadedExecutableImpl(
            executable: executable,
            meta: meta,
            client: impl
        )
        args.pointee.loaded_executable = UnsafeMutablePointer<PJRT_LoadedExecutable>(
            retainAsOpaque(loaded)
        )
        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "Deserialization re-compilation failed: \(error)")
    }
}
