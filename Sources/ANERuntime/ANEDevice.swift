// ANEDevice.swift
// ANERuntime
//
// Main entry point for ANE operations. Manages the connection to the
// Apple Neural Engine via runtime-resolved private APIs.
//
// Discovered API surface (macOS 15, M1):
//
// Compile flow:
//   _ANEInMemoryModelDescriptor.modelWithMILText:weights:optionsPlist:
//   _ANEClient.compileModel:options:qos:error:
//   _ANEClient.loadModel:options:qos:error:
//
// Execute flow:
//   _ANEIOSurfaceObject.objectWithIOSurface:
//   _ANERequest.requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:
//   _ANEClient.evaluateWithModel:options:request:qos:error:
//
// Cleanup:
//   _ANEClient.unloadModel:options:qos:error:

import Foundation
import IOSurface

// MARK: - ObjC Runtime Bridge

/// Bridges to ANE private API methods via the ObjC runtime.
///
/// All selectors and function pointers are resolved at runtime.
/// No private symbols appear in headers or link-time dependencies.
internal struct ANEBridge: Sendable {
    /// The _ANEClient class.
    let clientClass: AnyClass

    /// The _ANEInMemoryModelDescriptor class.
    let modelDescriptorClass: AnyClass

    /// The _ANEIOSurfaceObject class.
    let ioSurfaceObjectClass: AnyClass

    /// The _ANERequest class.
    let requestClass: AnyClass

    /// The _ANEDeviceInfo class (optional — only for hardware queries).
    let deviceInfoClass: AnyClass?

    /// The resolved objc_msgSend function pointer.
    nonisolated(unsafe) let msgSend: UnsafeMutableRawPointer

    /// Attempts to resolve all required classes and symbols.
    static func resolve(from availability: ANEAvailability) -> ANEBridge? {
        guard let clientClass = availability.clientClass else { return nil }

        guard let msgSend = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "objc_msgSend") else {
            return nil
        }

        // Required classes
        guard let modelDescClass = objc_getClass("_ANEInMemoryModelDescriptor") as? AnyClass else {
            return nil
        }
        guard let ioSurfObjClass = objc_getClass("_ANEIOSurfaceObject") as? AnyClass else {
            return nil
        }
        guard let reqClass = objc_getClass("_ANERequest") as? AnyClass else {
            return nil
        }

        // Optional
        let devInfoClass: AnyClass? = objc_getClass("_ANEDeviceInfo") as? AnyClass

        return ANEBridge(
            clientClass: clientClass,
            modelDescriptorClass: modelDescClass,
            ioSurfaceObjectClass: ioSurfObjClass,
            requestClass: reqClass,
            deviceInfoClass: devInfoClass,
            msgSend: msgSend
        )
    }

    // MARK: - Typed objc_msgSend Wrappers

    // Class method: (AnyClass, Selector) -> AnyObject?
    private var classToObj: (@convention(c) (AnyClass, Selector) -> AnyObject?) {
        unsafeBitCast(msgSend, to: (@convention(c) (AnyClass, Selector) -> AnyObject?).self)
    }

    // Instance method: (AnyObject, Selector) -> AnyObject?
    private var objToObj: (@convention(c) (AnyObject, Selector) -> AnyObject?) {
        unsafeBitCast(msgSend, to: (@convention(c) (AnyObject, Selector) -> AnyObject?).self)
    }

    // MARK: - _ANEClient

    /// +[_ANEClient sharedConnection]
    func createClient() -> AnyObject? {
        let sel = sel_registerName("sharedConnection")
        return classToObj(clientClass, sel)
    }

    /// -[_ANEClient compileModel:options:qos:error:]
    func compileModel(
        client: AnyObject,
        model: AnyObject,
        options: AnyObject?,
        qos: UInt
    ) -> (success: Bool, error: AnyObject?) {
        let sel = sel_registerName("compileModel:options:qos:error:")
        typealias Fn = @convention(c) (
            AnyObject, Selector,
            AnyObject,          // model descriptor
            AnyObject?,         // options (nil)
            UInt,               // qos
            UnsafeMutablePointer<AnyObject?> // error out
        ) -> Bool
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        var error: AnyObject?
        let ok = fn(client, sel, model, options, qos, &error)
        return (ok, error)
    }

    /// -[_ANEClient loadModel:options:qos:error:]
    func loadModel(
        client: AnyObject,
        model: AnyObject,
        options: AnyObject?,
        qos: UInt
    ) -> (success: Bool, error: AnyObject?) {
        let sel = sel_registerName("loadModel:options:qos:error:")
        typealias Fn = @convention(c) (
            AnyObject, Selector,
            AnyObject, AnyObject?, UInt,
            UnsafeMutablePointer<AnyObject?>
        ) -> Bool
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        var error: AnyObject?
        let ok = fn(client, sel, model, options, qos, &error)
        return (ok, error)
    }

    /// -[_ANEClient evaluateWithModel:options:request:qos:error:]
    func evaluate(
        client: AnyObject,
        model: AnyObject,
        options: AnyObject?,
        request: AnyObject,
        qos: UInt
    ) -> (success: Bool, error: AnyObject?) {
        let sel = sel_registerName("evaluateWithModel:options:request:qos:error:")
        typealias Fn = @convention(c) (
            AnyObject, Selector,
            AnyObject, AnyObject?, AnyObject, UInt,
            UnsafeMutablePointer<AnyObject?>
        ) -> Bool
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        var error: AnyObject?
        let ok = fn(client, sel, model, options, request, qos, &error)
        return (ok, error)
    }

    /// -[_ANEClient unloadModel:options:qos:error:]
    func unloadModel(
        client: AnyObject,
        model: AnyObject,
        options: AnyObject?,
        qos: UInt
    ) -> Bool {
        let sel = sel_registerName("unloadModel:options:qos:error:")
        typealias Fn = @convention(c) (
            AnyObject, Selector,
            AnyObject, AnyObject?, UInt,
            UnsafeMutablePointer<AnyObject?>
        ) -> Bool
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        var error: AnyObject?
        return fn(client, sel, model, options, qos, &error)
    }

    // MARK: - _ANEInMemoryModelDescriptor

    /// +[_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:]
    func createModelDescriptor(
        milText: Data,
        weights: NSDictionary,
        optionsPlist: Data?
    ) -> AnyObject? {
        let sel = sel_registerName("modelWithMILText:weights:optionsPlist:")
        typealias Fn = @convention(c) (
            AnyClass, Selector,
            AnyObject,    // milText (NSData)
            AnyObject,    // weights (NSDictionary)
            AnyObject?    // optionsPlist (NSData?)
        ) -> AnyObject?
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        return fn(modelDescriptorClass, sel, milText as NSData, weights, optionsPlist as NSData?)
    }

    // MARK: - _ANEModel property access

    /// Read the programHandle property from an _ANEModel
    func getProgramHandle(model: AnyObject) -> UInt {
        let sel = sel_registerName("programHandle")
        typealias Fn = @convention(c) (AnyObject, Selector) -> UInt
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        return fn(model, sel)
    }

    // MARK: - _ANEIOSurfaceObject

    /// +[_ANEIOSurfaceObject objectWithIOSurface:]
    func createIOSurfaceObject(surface: IOSurfaceRef) -> AnyObject? {
        let sel = sel_registerName("objectWithIOSurface:")
        typealias Fn = @convention(c) (
            AnyClass, Selector, IOSurfaceRef
        ) -> AnyObject?
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        return fn(ioSurfaceObjectClass, sel, surface)
    }

    // MARK: - _ANERequest

    /// +[_ANERequest requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:]
    func createRequest(
        inputs: NSArray,
        inputIndices: NSArray,
        outputs: NSArray,
        outputIndices: NSArray,
        procedureIndex: NSNumber
    ) -> AnyObject? {
        let sel = sel_registerName(
            "requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:")
        typealias Fn = @convention(c) (
            AnyClass, Selector,
            AnyObject, AnyObject, AnyObject, AnyObject, AnyObject
        ) -> AnyObject?
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        return fn(requestClass, sel, inputs, inputIndices, outputs, outputIndices, procedureIndex)
    }

    // MARK: - _ANEDeviceInfo (optional)

    /// +[_ANEDeviceInfo hasANE]
    func hasANE() -> Bool {
        guard let cls = deviceInfoClass else { return false }
        let sel = sel_registerName("hasANE")
        typealias Fn = @convention(c) (AnyClass, Selector) -> Bool
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        return fn(cls, sel)
    }

    /// +[_ANEDeviceInfo numANECores]
    func numANECores() -> Int {
        guard let cls = deviceInfoClass else { return 0 }
        let sel = sel_registerName("numANECores")
        typealias Fn = @convention(c) (AnyClass, Selector) -> Int
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        return fn(cls, sel)
    }

    // MARK: - Method Discovery

    /// Lists available methods on a class (for debugging).
    func discoverMethods(for cls: AnyClass) -> [String] {
        var methods: [String] = []

        var methodCount: UInt32 = 0
        if let methodList = class_copyMethodList(cls, &methodCount) {
            for i in 0..<Int(methodCount) {
                let selector = method_getName(methodList[i])
                methods.append("- " + String(cString: sel_getName(selector)))
            }
            free(methodList)
        }

        if let metaClass = objc_getMetaClass(NSStringFromClass(cls)) as? AnyClass {
            var classMethodCount: UInt32 = 0
            if let methodList = class_copyMethodList(metaClass, &classMethodCount) {
                for i in 0..<Int(classMethodCount) {
                    let selector = method_getName(methodList[i])
                    methods.append("+ " + String(cString: sel_getName(selector)))
                }
                free(methodList)
            }
        }

        return methods
    }
}

// MARK: - ANE Hardware Info

/// Hardware information about the ANE, detected via _ANEDeviceInfo.
public struct ANEHardwareInfo: Sendable {
    /// Whether the hardware reports ANE presence.
    public let hasANE: Bool
    /// Number of ANE cores (16 on M1).
    public let coreCount: Int
}

// MARK: - ANEDevice

/// The main entry point for ANE operations.
///
/// `ANEDevice` manages the connection to the Apple Neural Engine and
/// provides methods for compiling and executing programs. Returns nil
/// from init if ANE is not available.
///
/// ## Example
/// ```swift
/// guard let device = ANEDevice() else {
///     print("ANE not available")
///     return
/// }
/// print("ANE on \(device.availability.chipName ?? "unknown") with \(device.hardwareInfo.coreCount) cores")
/// let program = try device.compile(milProgram: milText,
///                                  inputDescriptors: [inputDesc],
///                                  outputDescriptors: [outputDesc])
/// let outputs = try device.execute(program, inputs: [inputBuffer])
/// ```
public final class ANEDevice: @unchecked Sendable {

    /// Availability information for this device.
    public let availability: ANEAvailabilityInfo

    /// Hardware info from _ANEDeviceInfo.
    public let hardwareInfo: ANEHardwareInfo

    /// The compilation counter.
    public let compileCounter: CompileCounter

    /// Methods discovered on _ANEClient (for development/debugging).
    public let availableMethods: [String]

    /// The ObjC _ANEClient instance.
    private let aneClient: AnyObject

    /// The runtime-resolved bridge.
    private let bridge: ANEBridge

    /// The availability detector (retains dlopen handle).
    private let availabilityDetector: ANEAvailability

    private let lock = NSLock()

    /// Creates an ANEDevice. Returns nil if ANE is not available.
    public convenience init?() {
        self.init(warningLimit: CompileCounter.defaultWarningLimit,
                  maximumLimit: CompileCounter.defaultHardLimit)
    }

    /// Creates an ANEDevice with custom compilation counter limits.
    public init?(warningLimit: Int, maximumLimit: Int) {
        let detector = ANEAvailability()
        let info = detector.probe()

        guard info.isAvailable else { return nil }

        guard let bridge = ANEBridge.resolve(from: detector) else { return nil }

        guard let client = bridge.createClient() else { return nil }

        self.availabilityDetector = detector
        self.availability = info
        self.bridge = bridge
        self.aneClient = client
        self.compileCounter = CompileCounter(
            warningLimit: warningLimit,
            maximumLimit: maximumLimit
        )
        self.hardwareInfo = ANEHardwareInfo(
            hasANE: bridge.hasANE(),
            coreCount: bridge.numANECores()
        )
        self.availableMethods = bridge.discoverMethods(for: bridge.clientClass)
    }

    // MARK: - Compilation

    /// Compiles a MIL program with weight data into an ANE program.
    ///
    /// The compilation pipeline:
    /// 1. Create `_ANEInMemoryModelDescriptor` from MIL text + weights
    /// 2. Call `_ANEClient.compileModel:options:qos:error:` to compile for ANE
    /// 3. Call `_ANEClient.loadModel:options:qos:error:` to load onto hardware
    /// 4. Wrap the loaded model in an `ANEProgram`
    ///
    /// - Parameters:
    ///   - milProgram: The MIL program text (Core ML intermediate representation).
    ///   - weights: Weight name→data mapping as a dictionary. Empty dict if no weights.
    ///   - inputDescriptors: Descriptors for input buffers.
    ///   - outputDescriptors: Descriptors for output buffers.
    /// - Throws: `ANEError.compilationFailed`, `ANEError.compilationLimitReached`
    /// - Returns: A compiled `ANEProgram`.
    public func compile(
        milProgram: String,
        weights: [String: Data] = [:],
        inputDescriptors: [ANEBufferDescriptor],
        outputDescriptors: [ANEBufferDescriptor]
    ) throws -> ANEProgram {
        guard compileCounter.canCompile else {
            throw ANEError.compilationLimitReached(
                current: compileCounter.currentCount,
                limit: CompileCounter.defaultHardLimit
            )
        }

        // 1. Create model descriptor
        guard let milData = milProgram.data(using: .utf8) else {
            throw ANEError.compilationFailed("Failed to encode MIL program as UTF-8")
        }

        // Convert weights dict to NSDictionary of NSData
        let weightsDict = NSMutableDictionary()
        for (key, value) in weights {
            weightsDict[key] = value as NSData
        }

        guard let modelDescriptor = bridge.createModelDescriptor(
            milText: milData,
            weights: weightsDict,
            optionsPlist: nil
        ) else {
            throw ANEError.compilationFailed(
                "Failed to create _ANEInMemoryModelDescriptor from MIL text")
        }

        // 2. Compile the model
        let compileResult = bridge.compileModel(
            client: aneClient,
            model: modelDescriptor,
            options: nil,
            qos: 0x21  // QOS_CLASS_USER_INITIATED
        )

        if !compileResult.success {
            let errorDesc = compileResult.error.map { "\($0)" } ?? "unknown error"
            throw ANEError.compilationFailed(
                "compileModel failed: \(errorDesc)")
        }

        // 3. Load the model onto ANE hardware
        let loadResult = bridge.loadModel(
            client: aneClient,
            model: modelDescriptor,
            options: nil,
            qos: 0x21
        )

        if !loadResult.success {
            let errorDesc = loadResult.error.map { "\($0)" } ?? "unknown error"
            throw ANEError.compilationFailed(
                "loadModel failed: \(errorDesc)")
        }

        // 4. Increment counter and wrap in ANEProgram
        let compileNum = try compileCounter.increment()

        let program = ANEProgram(
            id: UUID().uuidString,
            programHandle: modelDescriptor,
            inputDescriptors: inputDescriptors,
            outputDescriptors: outputDescriptors,
            compilationNumber: compileNum
        )

        return program
    }

    // MARK: - Execution

    /// Executes a compiled program with input buffers.
    ///
    /// The execution pipeline:
    /// 1. Wrap input/output IOSurfaces as `_ANEIOSurfaceObject`
    /// 2. Build an `_ANERequest` with input/output arrays and indices
    /// 3. Call `_ANEClient.evaluateWithModel:options:request:qos:error:`
    /// 4. Results are written directly into output IOSurfaces by the ANE
    ///
    /// - Parameters:
    ///   - program: The compiled ANE program.
    ///   - inputs: Input buffers (must match program's inputDescriptors).
    /// - Throws: `ANEError.executionFailed`
    /// - Returns: Output buffers containing results.
    public func execute(
        _ program: ANEProgram,
        inputs: [ANEBuffer]
    ) throws -> [ANEBuffer] {
        guard program.isValid else {
            throw ANEError.executionFailed("Program has been released")
        }

        guard let modelHandle = program.programHandle else {
            throw ANEError.executionFailed("Program has no model handle")
        }

        guard inputs.count == program.inputCount else {
            throw ANEError.executionFailed(
                "Input count mismatch: expected \(program.inputCount), got \(inputs.count)")
        }

        for (i, input) in inputs.enumerated() {
            guard input.isValid else {
                throw ANEError.executionFailed("Input buffer \(i) has been released")
            }
        }

        // 1. Allocate output buffers
        var outputs: [ANEBuffer] = []
        for desc in program.outputDescriptors {
            guard let buffer = ANEBuffer(descriptor: desc) else {
                throw ANEError.executionFailed(
                    "Failed to create output buffer for shape \(desc.shape)")
            }
            outputs.append(buffer)
        }

        // 2. Wrap IOSurfaces as _ANEIOSurfaceObject
        let inputSurfaceObjects = NSMutableArray()
        let inputIndices = NSMutableArray()
        for (i, input) in inputs.enumerated() {
            guard let surfObj = bridge.createIOSurfaceObject(surface: input.surface) else {
                throw ANEError.executionFailed(
                    "Failed to create _ANEIOSurfaceObject for input \(i)")
            }
            inputSurfaceObjects.add(surfObj)
            inputIndices.add(NSNumber(value: i))
        }

        let outputSurfaceObjects = NSMutableArray()
        let outputIndices = NSMutableArray()
        for (i, output) in outputs.enumerated() {
            guard let surfObj = bridge.createIOSurfaceObject(surface: output.surface) else {
                throw ANEError.executionFailed(
                    "Failed to create _ANEIOSurfaceObject for output \(i)")
            }
            outputSurfaceObjects.add(surfObj)
            outputIndices.add(NSNumber(value: i))
        }

        // 3. Create the request
        guard let request = bridge.createRequest(
            inputs: inputSurfaceObjects,
            inputIndices: inputIndices,
            outputs: outputSurfaceObjects,
            outputIndices: outputIndices,
            procedureIndex: NSNumber(value: 0)
        ) else {
            throw ANEError.executionFailed("Failed to create _ANERequest")
        }

        // 4. Evaluate
        let evalResult = bridge.evaluate(
            client: aneClient,
            model: modelHandle,
            options: nil,
            request: request,
            qos: 0x21  // QOS_CLASS_USER_INITIATED
        )

        if !evalResult.success {
            let errorDesc = evalResult.error.map { "\($0)" } ?? "unknown error"
            throw ANEError.executionFailed(
                "evaluateWithModel failed: \(errorDesc)")
        }

        return outputs
    }

    // MARK: - Model Lifecycle

    /// Unloads a program from ANE hardware, freeing hardware resources.
    /// The program object becomes invalid after this call.
    public func unload(_ program: ANEProgram) {
        if let modelHandle = program.programHandle {
            _ = bridge.unloadModel(
                client: aneClient,
                model: modelHandle,
                options: nil,
                qos: 0x21
            )
        }
        program.release()
    }

    // MARK: - Buffer Creation

    /// Creates an ANE buffer with the given descriptor.
    public func createBuffer(descriptor: ANEBufferDescriptor) -> ANEBuffer? {
        return ANEBuffer(descriptor: descriptor)
    }

    /// Creates an ANE buffer from Float32 data.
    public func createBuffer(
        data: [Float],
        shape: [Int]
    ) throws -> ANEBuffer {
        let descriptor = ANEBufferDescriptor(shape: shape)
        guard let buffer = ANEBuffer(descriptor: descriptor) else {
            throw ANEError.bufferError("Failed to create IOSurface for shape \(shape)")
        }
        try buffer.writeFloat32(data, shape: shape)
        return buffer
    }
}
