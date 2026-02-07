// PJRTTopology.swift
// PJRTMetalHLO
//
// PJRT topology description for single-device Metal setup.

import CPJRTApi
import Foundation

/// Concrete backing storage for opaque PJRT_TopologyDescription pointers.
final class PJRTTopologyDescriptionImpl: @unchecked Sendable {
    let deviceDescriptions: [PJRTDeviceDescriptionImpl]

    // Stable pointer array for C API
    let descriptionsPtrBuffer: UnsafeMutableBufferPointer<UnsafeMutablePointer<PJRT_DeviceDescription>?>

    init(deviceDescriptions: [PJRTDeviceDescriptionImpl]) {
        self.deviceDescriptions = deviceDescriptions
        self.descriptionsPtrBuffer = .allocate(capacity: deviceDescriptions.count)
        for (i, desc) in deviceDescriptions.enumerated() {
            descriptionsPtrBuffer[i] = UnsafeMutablePointer<PJRT_DeviceDescription>(
                retainAsOpaque(desc)
            )
        }
    }

    deinit {
        for i in 0..<descriptionsPtrBuffer.count {
            if let ptr = descriptionsPtrBuffer[i] {
                releaseOpaque(OpaquePointer(ptr), as: PJRTDeviceDescriptionImpl.self)
            }
        }
        descriptionsPtrBuffer.deallocate()
    }
}

// MARK: - PJRT_Client_TopologyDescription

func pjrt_client_topology_description(
    _ args: UnsafeMutablePointer<PJRT_Client_TopologyDescription_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let clientPtr = args.pointee.client else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let impl = fromOpaque(OpaquePointer(clientPtr), as: PJRTClientImpl.self)
    // Topology is owned by the client — use the existing opaque pointer without retaining
    args.pointee.topology = UnsafeMutablePointer<PJRT_TopologyDescription>(impl.topologyOpaquePtr)
    return nil
}

// MARK: - TopologyDescription Functions

func pjrt_topology_description_platform_name(
    _ args: UnsafeMutablePointer<PJRT_TopologyDescription_PlatformName_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    nonisolated(unsafe) let name = platformNameStatic
    args.pointee.platform_name = UnsafePointer(name)
    args.pointee.platform_name_size = strlen(name)
    return nil
}

func pjrt_topology_description_platform_version(
    _ args: UnsafeMutablePointer<PJRT_TopologyDescription_PlatformVersion_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    nonisolated(unsafe) let version = platformVersionStatic
    args.pointee.platform_version = UnsafePointer(version)
    args.pointee.platform_version_size = strlen(version)
    return nil
}

func pjrt_topology_description_get_device_descriptions(
    _ args: UnsafeMutablePointer<PJRT_TopologyDescription_GetDeviceDescriptions_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args, let topoPtr = args.pointee.topology else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    let topo = fromOpaque(OpaquePointer(topoPtr), as: PJRTTopologyDescriptionImpl.self)
    args.pointee.descriptions = UnsafePointer(topo.descriptionsPtrBuffer.baseAddress)
    args.pointee.num_descriptions = topo.descriptionsPtrBuffer.count
    return nil
}

func pjrt_topology_description_serialize(
    _ args: UnsafeMutablePointer<PJRT_TopologyDescription_Serialize_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    return makeError(PJRT_Error_Code_UNIMPLEMENTED, "TopologyDescription_Serialize not implemented")
}

func pjrt_topology_description_attributes(
    _ args: UnsafeMutablePointer<PJRT_TopologyDescription_Attributes_Args>?
) -> UnsafeMutablePointer<PJRT_Error>? {
    guard let args = args else {
        return makeError(PJRT_Error_Code_INVALID_ARGUMENT, "NULL args")
    }
    args.pointee.num_attributes = 0
    args.pointee.attributes = nil
    return nil
}

// Shared platform strings accessible from both PJRTClient and PJRTTopology
nonisolated(unsafe) let platformNameStatic = strdup("metalhlo")!
nonisolated(unsafe) let platformVersionStatic = strdup("0.1.0")!
