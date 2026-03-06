// PJRTDevice.swift
// PJRTMetalHLO
//
// PJRT device and device description implementations wrapping MTLDevice.

import CPJRTApi
import Metal
import Foundation
import ANERuntime
import MetalHLO

/// Concrete backing storage for opaque PJRT_DeviceDescription pointers.
final class PJRTDeviceDescriptionImpl: @unchecked Sendable {
    let id: Int32
    let processIndex: Int32
    let kind: String
    let debugString: String
    let displayString: String

    // Cached C strings
    let cKind: UnsafeMutablePointer<CChar>
    let cDebugString: UnsafeMutablePointer<CChar>
    let cDisplayString: UnsafeMutablePointer<CChar>

    // Device attributes (stable C arrays for PJRT_NamedValue)
    let attributes: UnsafeMutableBufferPointer<PJRT_NamedValue>
    let numAttributes: Int
    private let attrNamePtrs: [UnsafeMutablePointer<CChar>]
    private let attrStringPtrs: [UnsafeMutablePointer<CChar>]

    init(device: MTLDevice, id: Int32) {
        self.id = id
        self.processIndex = 0
        self.kind = "Metal"
        self.debugString = "Metal:\(device.name)"
        self.displayString = device.name
        self.cKind = strdup(kind)!
        self.cDebugString = strdup(debugString)!
        self.cDisplayString = strdup(displayString)!

        // Build device attributes
        let aneInfo = ANEAvailability().probe()
        let policy = ProcessInfo.processInfo.environment["METALHLO_DEVICE_POLICY"] ?? "auto"

        var names: [UnsafeMutablePointer<CChar>] = []
        var stringValues: [UnsafeMutablePointer<CChar>] = []
        var namedValues: [PJRT_NamedValue] = []

        // Attribute: ane_available (bool)
        let aneAvailName = strdup("ane_available")!
        names.append(aneAvailName)
        var aneAvailAttr = PJRT_NamedValue()
        aneAvailAttr.struct_size = MemoryLayout<PJRT_NamedValue>.size
        aneAvailAttr.name = UnsafePointer(aneAvailName)
        aneAvailAttr.name_size = strlen(aneAvailName)
        aneAvailAttr.type = PJRT_NamedValue_kBool
        aneAvailAttr.bool_value = aneInfo.isAvailable
        aneAvailAttr.value_size = 1
        namedValues.append(aneAvailAttr)

        // Attribute: device_policy (string)
        let policyName = strdup("device_policy")!
        let policyValue = strdup(policy)!
        names.append(policyName)
        stringValues.append(policyValue)
        var policyAttr = PJRT_NamedValue()
        policyAttr.struct_size = MemoryLayout<PJRT_NamedValue>.size
        policyAttr.name = UnsafePointer(policyName)
        policyAttr.name_size = strlen(policyName)
        policyAttr.type = PJRT_NamedValue_kString
        policyAttr.string_value = UnsafePointer(policyValue)
        policyAttr.value_size = strlen(policyValue)
        namedValues.append(policyAttr)

        // Attribute: ane_core_count (int64)
        if aneInfo.isAvailable {
            let coreName = strdup("ane_core_count")!
            names.append(coreName)
            var coreAttr = PJRT_NamedValue()
            coreAttr.struct_size = MemoryLayout<PJRT_NamedValue>.size
            coreAttr.name = UnsafePointer(coreName)
            coreAttr.name_size = strlen(coreName)
            coreAttr.type = PJRT_NamedValue_kInt64
            coreAttr.int64_value = Int64(aneInfo.coreCount)
            coreAttr.value_size = 1
            namedValues.append(coreAttr)
        }

        self.numAttributes = namedValues.count
        self.attributes = .allocate(capacity: max(namedValues.count, 1))
        for (i, v) in namedValues.enumerated() { attributes[i] = v }
        self.attrNamePtrs = names
        self.attrStringPtrs = stringValues
    }

    deinit {
        free(cKind)
        free(cDebugString)
        free(cDisplayString)
        attributes.deallocate()
        for ptr in attrNamePtrs { free(ptr) }
        for ptr in attrStringPtrs { free(ptr) }
    }
}

/// Concrete backing storage for opaque PJRT_Device pointers.
final class PJRTDeviceImpl: @unchecked Sendable {
    let metalDevice: MTLDevice
    let description: PJRTDeviceDescriptionImpl
    var memories: [PJRTMemoryImpl] = []

    weak var client: PJRTClientImpl?

    init(device: MTLDevice, id: Int32) {
        self.metalDevice = device
        self.description = PJRTDeviceDescriptionImpl(device: device, id: id)
    }
}

// MARK: - DeviceDescription Functions

func pjrt_device_description_id(
    _ args: UnsafeMutablePointer<PJRT_DeviceDescription_Id_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let descPtr = args.pointee.device_description else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let desc = fromOpaque(OpaquePointer(descPtr), as: PJRTDeviceDescriptionImpl.self)
    args.pointee.id = desc.id
    return nil
}

func pjrt_device_description_process_index(
    _ args: UnsafeMutablePointer<PJRT_DeviceDescription_ProcessIndex_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let descPtr = args.pointee.device_description else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let desc = fromOpaque(OpaquePointer(descPtr), as: PJRTDeviceDescriptionImpl.self)
    args.pointee.process_index = desc.processIndex
    return nil
}

func pjrt_device_description_attributes(
    _ args: UnsafeMutablePointer<PJRT_DeviceDescription_Attributes_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let descPtr = args.pointee.device_description else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let desc = fromOpaque(OpaquePointer(descPtr), as: PJRTDeviceDescriptionImpl.self)
    args.pointee.num_attributes = desc.numAttributes
    args.pointee.attributes = UnsafePointer(desc.attributes.baseAddress)
    return nil
}

func pjrt_device_description_kind(
    _ args: UnsafeMutablePointer<PJRT_DeviceDescription_Kind_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let descPtr = args.pointee.device_description else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let desc = fromOpaque(OpaquePointer(descPtr), as: PJRTDeviceDescriptionImpl.self)
    args.pointee.device_kind = UnsafePointer(desc.cKind)
    args.pointee.device_kind_size = strlen(desc.cKind)
    return nil
}

func pjrt_device_description_debug_string(
    _ args: UnsafeMutablePointer<PJRT_DeviceDescription_DebugString_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let descPtr = args.pointee.device_description else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let desc = fromOpaque(OpaquePointer(descPtr), as: PJRTDeviceDescriptionImpl.self)
    args.pointee.debug_string = UnsafePointer(desc.cDebugString)
    args.pointee.debug_string_size = strlen(desc.cDebugString)
    return nil
}

func pjrt_device_description_to_string(
    _ args: UnsafeMutablePointer<PJRT_DeviceDescription_ToString_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let descPtr = args.pointee.device_description else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let desc = fromOpaque(OpaquePointer(descPtr), as: PJRTDeviceDescriptionImpl.self)
    args.pointee.to_string = UnsafePointer(desc.cDisplayString)
    args.pointee.to_string_size = strlen(desc.cDisplayString)
    return nil
}

// MARK: - Device Functions

func pjrt_device_get_description(
    _ args: UnsafeMutablePointer<PJRT_Device_GetDescription_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let devicePtr = args.pointee.device else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let device = fromOpaque(OpaquePointer(devicePtr), as: PJRTDeviceImpl.self)
    let descOpaque = retainAsOpaque(device.description)
    args.pointee.device_description = UnsafeMutablePointer<PJRT_DeviceDescription>(descOpaque)
    return nil
}

func pjrt_device_is_addressable(
    _ args: UnsafeMutablePointer<PJRT_Device_IsAddressable_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    // All Metal devices on this host are addressable
    args.pointee.is_addressable = true
    return nil
}

func pjrt_device_local_hardware_id(
    _ args: UnsafeMutablePointer<PJRT_Device_LocalHardwareId_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let devicePtr = args.pointee.device else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let device = fromOpaque(OpaquePointer(devicePtr), as: PJRTDeviceImpl.self)
    args.pointee.local_hardware_id = Int32(device.description.id)
    return nil
}

func pjrt_device_addressable_memories(
    _ args: UnsafeMutablePointer<PJRT_Device_AddressableMemories_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let devicePtr = args.pointee.device else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let device = fromOpaque(OpaquePointer(devicePtr), as: PJRTDeviceImpl.self)
    if let client = device.client {
        args.pointee.memories = UnsafePointer(client.memoriesPtr)
        args.pointee.num_memories = client.memoriesPtrCount
    } else {
        args.pointee.num_memories = 0
    }
    return nil
}

func pjrt_device_default_memory(
    _ args: UnsafeMutablePointer<PJRT_Device_DefaultMemory_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let devicePtr = args.pointee.device else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let device = fromOpaque(OpaquePointer(devicePtr), as: PJRTDeviceImpl.self)
    if let client = device.client, let firstMemory = client.memories.first {
        args.pointee.memory = UnsafeMutablePointer<PJRT_Memory>(
            retainAsOpaque(firstMemory)
        )
    }
    return nil
}

func pjrt_device_memory_stats(
    _ args: UnsafeMutablePointer<PJRT_Device_MemoryStats_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let devicePtr = args.pointee.device else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let device = fromOpaque(OpaquePointer(devicePtr), as: PJRTDeviceImpl.self)
    let mtl = device.metalDevice

    // Report recommended max working set size as total memory
    let totalBytes = Int64(mtl.recommendedMaxWorkingSetSize)
    args.pointee.bytes_in_use = 0 // Not easily tracked per-device in Metal
    return nil
}
