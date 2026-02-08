// pjrt_metal_types.h
// CPJRTApi
//
// Defines opaque type bodies for PJRT types. The PJRT C API only
// forward-declares these as incomplete types; plugins must provide
// concrete definitions. We define minimal structs so Swift can import
// them as proper pointer types (UnsafeMutablePointer<PJRT_Client>)
// rather than OpaquePointer.

#ifndef PJRT_METAL_TYPES_H
#define PJRT_METAL_TYPES_H

#include "pjrt_c_api.h"

// Each opaque PJRT type gets a single `impl` field that the Swift
// layer will use via Unmanaged<SwiftClass>.toOpaque().
// These are never accessed from C directly — they exist solely
// to make the types complete for Swift's C importer.

struct PJRT_Error { void* impl; };
struct PJRT_Client { void* impl; };
struct PJRT_Device { void* impl; };
struct PJRT_DeviceDescription { void* impl; };
struct PJRT_Memory { void* impl; };
struct PJRT_Buffer { void* impl; };
struct PJRT_Executable { void* impl; };
struct PJRT_LoadedExecutable { void* impl; };
struct PJRT_Event { void* impl; };
struct PJRT_TopologyDescription { void* impl; };
struct PJRT_ExecuteContext { void* impl; };
struct PJRT_CopyToDeviceStream { void* impl; };
struct PJRT_AsyncHostToDeviceTransferManager { void* impl; };
struct PJRT_AsyncTrackingEvent { void* impl; };

#endif // PJRT_METAL_TYPES_H
