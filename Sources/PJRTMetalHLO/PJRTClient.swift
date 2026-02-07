// PJRTClient.swift
// PJRTMetalHLO
//
// PJRT client implementation wrapping MetalHLO Client.

import CPJRTApi
import Metal
import MetalHLO
import MetalHLOCore
import Foundation

// Platform strings are defined in PJRTTopology.swift as platformNameStatic / platformVersionStatic

/// Concrete backing storage for opaque PJRT_Client pointers.
final class PJRTClientImpl: @unchecked Sendable {
    let client: Client
    let metalDevice: MTLDevice
    let devices: [PJRTDeviceImpl]
    let memories: [PJRTMemoryImpl]
    let topology: PJRTTopologyDescriptionImpl
    let topologyOpaquePtr: OpaquePointer

    // Stable pointer arrays for C API
    let devicesPtrBuffer: UnsafeMutableBufferPointer<UnsafeMutablePointer<PJRT_Device>?>
    let memoriesPtrBuffer: UnsafeMutableBufferPointer<UnsafeMutablePointer<PJRT_Memory>?>

    var devicesPtr: UnsafeMutablePointer<UnsafeMutablePointer<PJRT_Device>?>? {
        devicesPtrBuffer.baseAddress
    }
    var devicesPtrCount: Int { devicesPtrBuffer.count }

    var memoriesPtr: UnsafeMutablePointer<UnsafeMutablePointer<PJRT_Memory>?>? {
        memoriesPtrBuffer.baseAddress
    }
    var memoriesPtrCount: Int { memoriesPtrBuffer.count }

    init(client: Client) {
        self.client = client
        self.metalDevice = client.device

        // Create device wrapper
        let device = PJRTDeviceImpl(device: client.device, id: 0)
        self.devices = [device]

        // Create unified memory space (Metal uses shared memory on Apple Silicon)
        let memory = PJRTMemoryImpl(id: 0, kind: "device", kindId: 0)
        memory.devices = [device]
        memory.buildDevicesPtr()
        device.memories = [memory]
        self.memories = [memory]

        // Build stable C pointer arrays
        self.devicesPtrBuffer = .allocate(capacity: 1)
        devicesPtrBuffer[0] = UnsafeMutablePointer<PJRT_Device>(
            retainAsOpaque(device)
        )

        self.memoriesPtrBuffer = .allocate(capacity: 1)
        memoriesPtrBuffer[0] = UnsafeMutablePointer<PJRT_Memory>(
            retainAsOpaque(memory)
        )

        // Build topology (owned by client, not separately destroyed)
        self.topology = PJRTTopologyDescriptionImpl(deviceDescriptions: [device.description])
        self.topologyOpaquePtr = retainAsOpaque(self.topology)

        // Set back-reference
        device.client = self
    }

    deinit {
        // Release topology opaque pointer
        releaseOpaque(topologyOpaquePtr, as: PJRTTopologyDescriptionImpl.self)

        // Release device and memory opaque pointers
        for i in 0..<devicesPtrBuffer.count {
            if let ptr = devicesPtrBuffer[i] {
                releaseOpaque(OpaquePointer(ptr), as: PJRTDeviceImpl.self)
            }
        }
        devicesPtrBuffer.deallocate()

        for i in 0..<memoriesPtrBuffer.count {
            if let ptr = memoriesPtrBuffer[i] {
                releaseOpaque(OpaquePointer(ptr), as: PJRTMemoryImpl.self)
            }
        }
        memoriesPtrBuffer.deallocate()
    }
}

// MARK: - PJRT_Plugin Functions

func pjrt_plugin_initialize(
    _ args: UnsafeMutablePointer<PJRT_Plugin_Initialize_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    // No special initialization needed
    return nil
}

func pjrt_plugin_attributes(
    _ args: UnsafeMutablePointer<PJRT_Plugin_Attributes_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.num_attributes = 0
    args.pointee.attributes = nil
    return nil
}

// MARK: - PJRT_Client Functions

func pjrt_client_create(
    _ args: UnsafeMutablePointer<PJRT_Client_Create_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    do {
        let client = try Client.create()
        let impl = PJRTClientImpl(client: client)
        args.pointee.client = UnsafeMutablePointer<PJRT_Client>(
            retainAsOpaque(impl)
        )
        return nil
    } catch {
        return makeError(PJRT_Error_Code_INTERNAL, "Failed to create Metal client: \(error)")
    }
}

func pjrt_client_destroy(
    _ args: UnsafeMutablePointer<PJRT_Client_Destroy_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else { return nil }
    releaseOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)
    return nil
}

func pjrt_client_platform_name(
    _ args: UnsafeMutablePointer<PJRT_Client_PlatformName_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.platform_name = UnsafePointer(platformNameStatic)
    args.pointee.platform_name_size = strlen(platformNameStatic)
    return nil
}

func pjrt_client_process_index(
    _ args: UnsafeMutablePointer<PJRT_Client_ProcessIndex_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.process_index = 0 // Single-process
    return nil
}

func pjrt_client_platform_version(
    _ args: UnsafeMutablePointer<PJRT_Client_PlatformVersion_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.platform_version = UnsafePointer(platformVersionStatic)
    args.pointee.platform_version_size = strlen(platformVersionStatic)
    return nil
}

func pjrt_client_devices(
    _ args: UnsafeMutablePointer<PJRT_Client_Devices_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)
    args.pointee.devices = UnsafePointer(impl.devicesPtr)
    args.pointee.num_devices = impl.devicesPtrCount
    return nil
}

func pjrt_client_addressable_devices(
    _ args: UnsafeMutablePointer<PJRT_Client_AddressableDevices_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)
    // All devices are addressable in single-host
    args.pointee.addressable_devices = UnsafePointer(impl.devicesPtr)
    args.pointee.num_addressable_devices = impl.devicesPtrCount
    return nil
}

func pjrt_client_addressable_memories(
    _ args: UnsafeMutablePointer<PJRT_Client_AddressableMemories_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)
    args.pointee.addressable_memories = UnsafePointer(impl.memoriesPtr)
    args.pointee.num_addressable_memories = impl.memoriesPtrCount
    return nil
}

func pjrt_client_lookup_device(
    _ args: UnsafeMutablePointer<PJRT_Client_LookupDevice_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)
    let targetId = args.pointee.id
    guard let device = impl.devices.first(where: { Int32($0.description.id) == targetId }) else {
        return makeError(PJRT_Error_Code_NOT_FOUND, "Device \(targetId) not found")
    }
    args.pointee.device = UnsafeMutablePointer<PJRT_Device>(
        retainAsOpaque(device)
    )
    return nil
}

func pjrt_client_lookup_addressable_device(
    _ args: UnsafeMutablePointer<PJRT_Client_LookupAddressableDevice_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)
    let localId = args.pointee.local_hardware_id
    guard Int(localId) < impl.devices.count else {
        return makeError(PJRT_Error_Code_NOT_FOUND, "Local device \(localId) not found")
    }
    let device = impl.devices[Int(localId)]
    args.pointee.addressable_device = UnsafeMutablePointer<PJRT_Device>(
        retainAsOpaque(device)
    )
    return nil
}

func pjrt_client_default_device_assignment(
    _ args: UnsafeMutablePointer<PJRT_Client_DefaultDeviceAssignment_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    // Single device: assign device 0 to all replicas/partitions
    let numReplicas = Int(args.pointee.num_replicas)
    let numPartitions = Int(args.pointee.num_partitions)
    if let output = args.pointee.default_assignment {
        for i in 0..<(numReplicas * numPartitions) {
            output[i] = 0
        }
    }
    return nil
}
