// PJRTApi.swift
// PJRTMetalHLO
//
// Constructs the PJRT_Api vtable and exports the GetPjrtApi entry point.

import CPJRTApi
import Foundation

// MARK: - Stub functions for optional/unimplemented APIs

private func stub_void(_ args: UnsafeMutableRawPointer?) {}

private func stub_error(_ name: String) -> (UnsafeMutableRawPointer?) -> UnsafeMutablePointer<PJRT_Error>? {
    { _ in unimplementedError(name) }
}

// MARK: - PJRT_Api Construction

/// Build and return the static PJRT_Api instance.
///
/// This populates all ~150 function pointers. Required functions get real
/// implementations; optional functions return UNIMPLEMENTED errors.
private func buildPjrtApi() -> PJRT_Api {
    var api = PJRT_Api()

    api.struct_size = MemoryLayout<PJRT_Api>.size

    // Version
    api.pjrt_api_version.struct_size = MemoryLayout<PJRT_Api_Version>.size
    api.pjrt_api_version.major_version = Int32(PJRT_API_MAJOR)
    api.pjrt_api_version.minor_version = Int32(PJRT_API_MINOR)

    // ---- Error ----
    api.PJRT_Error_Destroy = pjrt_error_destroy
    api.PJRT_Error_Message = pjrt_error_message
    api.PJRT_Error_GetCode = pjrt_error_getcode

    // ---- Plugin ----
    api.PJRT_Plugin_Initialize = pjrt_plugin_initialize
    api.PJRT_Plugin_Attributes = pjrt_plugin_attributes

    // ---- Event ----
    api.PJRT_Event_Destroy = pjrt_event_destroy
    api.PJRT_Event_IsReady = pjrt_event_is_ready
    api.PJRT_Event_Error = pjrt_event_error
    api.PJRT_Event_Await = pjrt_event_await
    api.PJRT_Event_OnReady = pjrt_event_on_ready

    // ---- Client ----
    api.PJRT_Client_Create = pjrt_client_create
    api.PJRT_Client_Destroy = pjrt_client_destroy
    api.PJRT_Client_PlatformName = pjrt_client_platform_name
    api.PJRT_Client_ProcessIndex = pjrt_client_process_index
    api.PJRT_Client_PlatformVersion = pjrt_client_platform_version
    api.PJRT_Client_Devices = pjrt_client_devices
    api.PJRT_Client_AddressableDevices = pjrt_client_addressable_devices
    api.PJRT_Client_LookupDevice = pjrt_client_lookup_device
    api.PJRT_Client_LookupAddressableDevice = pjrt_client_lookup_addressable_device
    api.PJRT_Client_AddressableMemories = pjrt_client_addressable_memories
    api.PJRT_Client_Compile = pjrt_client_compile
    api.PJRT_Client_DefaultDeviceAssignment = pjrt_client_default_device_assignment
    api.PJRT_Client_BufferFromHostBuffer = pjrt_client_buffer_from_host_buffer

    // ---- DeviceDescription ----
    api.PJRT_DeviceDescription_Id = pjrt_device_description_id
    api.PJRT_DeviceDescription_ProcessIndex = pjrt_device_description_process_index
    api.PJRT_DeviceDescription_Attributes = pjrt_device_description_attributes
    api.PJRT_DeviceDescription_Kind = pjrt_device_description_kind
    api.PJRT_DeviceDescription_DebugString = pjrt_device_description_debug_string
    api.PJRT_DeviceDescription_ToString = pjrt_device_description_to_string

    // ---- Device ----
    api.PJRT_Device_GetDescription = pjrt_device_get_description
    api.PJRT_Device_IsAddressable = pjrt_device_is_addressable
    api.PJRT_Device_LocalHardwareId = pjrt_device_local_hardware_id
    api.PJRT_Device_AddressableMemories = pjrt_device_addressable_memories
    api.PJRT_Device_DefaultMemory = pjrt_device_default_memory
    api.PJRT_Device_MemoryStats = pjrt_device_memory_stats

    // ---- Memory ----
    api.PJRT_Memory_Id = pjrt_memory_id
    api.PJRT_Memory_Kind = pjrt_memory_kind
    api.PJRT_Memory_DebugString = pjrt_memory_debug_string
    api.PJRT_Memory_ToString = pjrt_memory_to_string
    api.PJRT_Memory_AddressableByDevices = pjrt_memory_addressable_by_devices
    api.PJRT_Memory_Kind_Id = pjrt_memory_kind_id

    // ---- Executable ----
    api.PJRT_Executable_Destroy = pjrt_executable_destroy
    api.PJRT_Executable_Name = pjrt_executable_name
    api.PJRT_Executable_NumReplicas = pjrt_executable_num_replicas
    api.PJRT_Executable_NumPartitions = pjrt_executable_num_partitions
    api.PJRT_Executable_NumOutputs = pjrt_executable_num_outputs
    api.PJRT_Executable_SizeOfGeneratedCodeInBytes = pjrt_executable_size_of_generated_code
    api.PJRT_Executable_GetCostAnalysis = pjrt_executable_get_cost_analysis
    api.PJRT_Executable_OutputMemoryKinds = pjrt_executable_output_memory_kinds
    api.PJRT_Executable_OptimizedProgram = pjrt_executable_optimized_program
    api.PJRT_Executable_Serialize = pjrt_executable_serialize
    api.PJRT_Executable_OutputElementTypes = pjrt_executable_output_element_types
    api.PJRT_Executable_OutputDimensions = pjrt_executable_output_dimensions
    api.PJRT_Executable_Fingerprint = pjrt_executable_fingerprint
    api.PJRT_Executable_GetCompiledMemoryStats = pjrt_executable_get_compiled_memory_stats

    // ---- LoadedExecutable ----
    api.PJRT_LoadedExecutable_Destroy = pjrt_loaded_executable_destroy
    api.PJRT_LoadedExecutable_GetExecutable = pjrt_loaded_executable_get_executable
    api.PJRT_LoadedExecutable_AddressableDevices = pjrt_loaded_executable_addressable_devices
    api.PJRT_LoadedExecutable_Delete = pjrt_loaded_executable_delete
    api.PJRT_LoadedExecutable_IsDeleted = pjrt_loaded_executable_is_deleted
    api.PJRT_LoadedExecutable_Execute = pjrt_loaded_executable_execute
    api.PJRT_Executable_DeserializeAndLoad = pjrt_executable_deserialize_and_load
    api.PJRT_LoadedExecutable_Fingerprint = pjrt_loaded_executable_fingerprint

    // ---- Buffer ----
    api.PJRT_Buffer_Destroy = pjrt_buffer_destroy
    api.PJRT_Buffer_ElementType = pjrt_buffer_element_type
    api.PJRT_Buffer_Dimensions = pjrt_buffer_dimensions
    api.PJRT_Buffer_UnpaddedDimensions = pjrt_buffer_unpadded_dimensions
    api.PJRT_Buffer_DynamicDimensionIndices = pjrt_buffer_dynamic_dimension_indices
    api.PJRT_Buffer_GetMemoryLayout = pjrt_buffer_get_memory_layout
    api.PJRT_Buffer_OnDeviceSizeInBytes = pjrt_buffer_on_device_size_in_bytes
    api.PJRT_Buffer_Device = pjrt_buffer_device
    api.PJRT_Buffer_Memory = pjrt_buffer_memory
    api.PJRT_Buffer_Delete = pjrt_buffer_delete
    api.PJRT_Buffer_IsDeleted = pjrt_buffer_is_deleted
    api.PJRT_Buffer_CopyToDevice = pjrt_buffer_copy_to_device
    api.PJRT_Buffer_ToHostBuffer = pjrt_buffer_to_host_buffer
    api.PJRT_Buffer_IsOnCpu = pjrt_buffer_is_on_cpu
    api.PJRT_Buffer_ReadyEvent = pjrt_buffer_ready_event
    api.PJRT_Buffer_UnsafePointer = pjrt_buffer_unsafe_pointer
    api.PJRT_Buffer_IncreaseExternalReferenceCount = pjrt_buffer_increase_external_reference_count
    api.PJRT_Buffer_DecreaseExternalReferenceCount = pjrt_buffer_decrease_external_reference_count
    api.PJRT_Buffer_OpaqueDeviceMemoryDataPointer = pjrt_buffer_opaque_device_memory_data_pointer
    api.PJRT_Buffer_CopyToMemory = pjrt_buffer_copy_to_memory
    api.PJRT_Buffer_CopyRawToHost = pjrt_buffer_copy_raw_to_host

    // ---- Remaining optional functions are left as nil (NULL) ----
    // The framework checks for NULL before calling optional functions.
    // These include: TopologyDescription, CopyToDeviceStream, Compile (standalone),
    // ExecuteContext, AsyncHostToDeviceTransfer, DMA, etc.

    return api
}

// MARK: - Static API Instance

/// The singleton PJRT_Api instance, allocated on the heap and never freed.
nonisolated(unsafe) private var pjrtApiStorage: UnsafeMutablePointer<PJRT_Api> = {
    let ptr = UnsafeMutablePointer<PJRT_Api>.allocate(capacity: 1)
    ptr.initialize(to: buildPjrtApi())
    return ptr
}()

// MARK: - GetPjrtApi Entry Point

/// The C-callable entry point that JAX/XLA uses to load the plugin.
///
/// JAX discovers this symbol via `dlsym(handle, "GetPjrtApi")` after
/// loading the shared library.
@_cdecl("GetPjrtApi")
public func GetPjrtApi() -> UnsafePointer<PJRT_Api> {
    UnsafePointer(pjrtApiStorage)
}
