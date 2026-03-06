// ANEAvailability.swift
// ANERuntime
//
// Runtime detection of ANE hardware and private framework availability.
// All detection is performed via dlopen/objc_getClass — no private API
// symbols appear at link time.

import Foundation

/// Information about ANE availability on this machine.
public struct ANEAvailabilityInfo: Sendable {
    /// Whether the ANE is physically present and accessible.
    public let isAvailable: Bool
    /// Human-readable reason if unavailable.
    public let reason: String?
    /// The chip name if detected (e.g., "Apple M1").
    public let chipName: String?
    /// Number of ANE cores detected (0 if unavailable).
    public let coreCount: Int
    /// API level detected (0 = unknown, 1 = base selectors present).
    public let apiLevel: Int
}

/// Probes the system for ANE hardware and private framework availability.
///
/// This class dynamically loads `AppleNeuralEngine.framework` and resolves
/// the `_ANEClient` class at runtime. It never references private API symbols
/// at link time, so it compiles on any macOS machine.
public final class ANEAvailability: @unchecked Sendable {

    /// Cached framework handle from dlopen.
    private var frameworkHandle: UnsafeMutableRawPointer?

    /// Cached _ANEClient class reference.
    private var aneClientClass: AnyClass?

    /// Thread safety.
    private let lock = NSLock()
    private var probed = false
    private var cachedInfo: ANEAvailabilityInfo?

    public init() {}

    /// Probes ANE availability. Safe to call multiple times (result is cached).
    public func probe() -> ANEAvailabilityInfo {
        lock.lock()
        defer { lock.unlock() }

        if let cached = cachedInfo {
            return cached
        }

        let info = performProbe()
        cachedInfo = info
        probed = true
        return info
    }

    /// Returns the dlopen handle for the ANE framework. Nil if unavailable.
    internal var handle: UnsafeMutableRawPointer? {
        lock.lock()
        defer { lock.unlock() }
        return frameworkHandle
    }

    /// Returns the _ANEClient class. Nil if unavailable.
    internal var clientClass: AnyClass? {
        lock.lock()
        defer { lock.unlock() }
        return aneClientClass
    }

    private func performProbe() -> ANEAvailabilityInfo {
        #if !arch(arm64)
        return ANEAvailabilityInfo(
            isAvailable: false,
            reason: "ANE requires Apple Silicon (arm64)",
            chipName: nil,
            coreCount: 0,
            apiLevel: 0
        )
        #else

        // Check macOS version
        let version = ProcessInfo.processInfo.operatingSystemVersion
        if version.majorVersion < 14 {
            return ANEAvailabilityInfo(
                isAvailable: false,
                reason: "ANE runtime requires macOS 14.0 or later (running \(version.majorVersion).\(version.minorVersion))",
                chipName: readChipName(),
                coreCount: 0,
                apiLevel: 0
            )
        }

        // Try to load the private framework
        let frameworkPath = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
        guard let handle = dlopen(frameworkPath, RTLD_LAZY) else {
            let error = String(cString: dlerror())
            return ANEAvailabilityInfo(
                isAvailable: false,
                reason: "Failed to load AppleNeuralEngine.framework: \(error)",
                chipName: readChipName(),
                coreCount: 0,
                apiLevel: 0
            )
        }
        self.frameworkHandle = handle

        // Try to find _ANEClient class
        guard let clientClass = objc_getClass("_ANEClient") as? AnyClass else {
            return ANEAvailabilityInfo(
                isAvailable: false,
                reason: "_ANEClient class not found in framework",
                chipName: readChipName(),
                coreCount: 0,
                apiLevel: 0
            )
        }
        self.aneClientClass = clientClass

        // Query core count via _ANEDeviceInfo if available
        let cores = queryCoreCount()

        // Detect API level by checking for required selectors
        let apiLevel = detectAPILevel(clientClass: clientClass)

        // If required base selectors are missing, mark as unavailable
        if apiLevel < 1 {
            return ANEAvailabilityInfo(
                isAvailable: false,
                reason: "_ANEClient is missing required method selectors (API may have changed)",
                chipName: readChipName(),
                coreCount: cores,
                apiLevel: 0
            )
        }

        return ANEAvailabilityInfo(
            isAvailable: true,
            reason: nil,
            chipName: readChipName(),
            coreCount: cores,
            apiLevel: apiLevel
        )
        #endif
    }

    deinit {
        if let handle = frameworkHandle {
            dlclose(handle)
        }
    }
}

// MARK: - Chip Name Detection

extension ANEAvailability {
    /// Queries the ANE core count via +[_ANEDeviceInfo numANECores].
    private func queryCoreCount() -> Int {
        guard let devInfoClass = objc_getClass("_ANEDeviceInfo") as? AnyClass else {
            return 0
        }
        guard let msgSend = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "objc_msgSend") else {
            return 0
        }
        let sel = sel_registerName("numANECores")
        typealias Fn = @convention(c) (AnyClass, Selector) -> Int
        let fn = unsafeBitCast(msgSend, to: Fn.self)
        return fn(devInfoClass, sel)
    }

    /// Detects the ANE API level by checking for known selectors on `_ANEClient`.
    ///
    /// - Level 0: Required selectors missing (API changed or unavailable)
    /// - Level 1: Base compile/load/evaluate selectors present
    /// - Level 2: Async variants also present (macOS 15+)
    private func detectAPILevel(clientClass: AnyClass) -> Int {
        // Required base selectors — all must be present for level 1
        let requiredSelectors = [
            "compileModel:options:qos:error:",
            "loadModel:options:qos:error:",
            "evaluateWithModel:options:request:qos:error:",
        ]

        for selectorName in requiredSelectors {
            let sel = sel_registerName(selectorName)
            if class_getInstanceMethod(clientClass, sel) == nil {
                return 0
            }
        }

        // Optional selectors for higher API levels
        let asyncSelector = sel_registerName("compileModel:options:qos:completionHandler:")
        if class_getInstanceMethod(clientClass, asyncSelector) != nil {
            return 2
        }

        return 1
    }

    /// Reads the chip/CPU brand string via sysctl.
    private func readChipName() -> String? {
        var size: Int = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return nil }

        var buffer = [UInt8](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        let length = buffer.firstIndex(of: 0) ?? buffer.count
        return String(decoding: buffer[..<length], as: UTF8.self)
    }
}
