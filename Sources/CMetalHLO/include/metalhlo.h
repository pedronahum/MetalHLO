// metalhlo.h
// CMetalHLO
//
// C API for MetalHLO - StableHLO execution on Apple Metal.

#ifndef METALHLO_H
#define METALHLO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Version
// ============================================================================

/// Get the library version string.
/// Returns: A static string, do not free.
const char* mhlo_version(void);

// ============================================================================
// Opaque Handles
// ============================================================================

typedef void* MHLOClientRef;
typedef void* MHLOExecutableRef;
typedef void* MHLOBufferRef;

// ============================================================================
// Status Codes
// ============================================================================

typedef enum {
    MHLO_OK = 0,
    MHLO_ERROR_NO_DEVICE = 1,
    MHLO_ERROR_UNSUPPORTED_DEVICE = 2,
    MHLO_ERROR_PARSE_FAILED = 3,
    MHLO_ERROR_UNSUPPORTED_OP = 4,
    MHLO_ERROR_COMPILATION_FAILED = 5,
    MHLO_ERROR_BUFFER_FAILED = 6,
    MHLO_ERROR_INPUT_MISMATCH = 7,
    MHLO_ERROR_EXECUTION_FAILED = 8,
    MHLO_ERROR_TRANSFER_FAILED = 9,
    MHLO_ERROR_INVALID_ARGUMENT = 10,
} MHLOStatusCode;

/// Get the last error message (NULL if no error).
/// The returned string is managed by the library; do not free it.
const char* mhlo_get_last_error(void);

/// Clear the last error message.
void mhlo_clear_error(void);

// ============================================================================
// Element Types
// ============================================================================

typedef enum {
    MHLO_F16 = 0,
    MHLO_F32 = 1,
    MHLO_F64 = 2,
    MHLO_BF16 = 3,
    MHLO_I1 = 4,
    MHLO_I8 = 5,
    MHLO_I16 = 6,
    MHLO_I32 = 7,
    MHLO_I64 = 8,
    MHLO_UI8 = 9,
    MHLO_UI16 = 10,
    MHLO_UI32 = 11,
    MHLO_UI64 = 12,
} MHLOElementType;

/// Get size in bytes for an element type (0 for I1).
size_t mhlo_element_type_size(MHLOElementType type);

// ============================================================================
// Client API
// ============================================================================

/// Create a client for the default Metal device.
/// - Parameter out_client: Output client handle.
/// - Returns: MHLO_OK on success, error code otherwise.
MHLOStatusCode mhlo_client_create(MHLOClientRef* out_client);

/// Destroy a client.
void mhlo_client_destroy(MHLOClientRef client);

/// Get device name.
/// - Returns: Device name string (caller must free with mhlo_free_string).
char* mhlo_client_device_name(MHLOClientRef client);

// ============================================================================
// Compilation API
// ============================================================================

/// Compile StableHLO MLIR text.
/// - Parameters:
///   - client: The client to compile with.
///   - mlir_text: StableHLO MLIR module text (null-terminated).
///   - out_executable: Output executable handle.
/// - Returns: MHLO_OK on success, error code otherwise.
MHLOStatusCode mhlo_compile(
    MHLOClientRef client,
    const char* mlir_text,
    MHLOExecutableRef* out_executable
);

/// Compile StableHLO MLIR with explicit length.
MHLOStatusCode mhlo_compile_n(
    MHLOClientRef client,
    const char* mlir_text,
    size_t mlir_length,
    MHLOExecutableRef* out_executable
);

/// Destroy an executable.
void mhlo_executable_destroy(MHLOExecutableRef executable);

/// Get number of inputs.
int32_t mhlo_executable_input_count(MHLOExecutableRef executable);

/// Get number of outputs.
int32_t mhlo_executable_output_count(MHLOExecutableRef executable);

/// Get input tensor type.
/// - Parameters:
///   - executable: The executable.
///   - index: Input index.
///   - out_shape: Pre-allocated array for shape dimensions (can be NULL).
///   - out_rank: Output for rank (can be NULL).
///   - out_element_type: Output for element type (can be NULL).
/// - Returns: MHLO_OK on success, error code otherwise.
MHLOStatusCode mhlo_executable_input_type(
    MHLOExecutableRef executable,
    int32_t index,
    int64_t* out_shape,
    int32_t* out_rank,
    MHLOElementType* out_element_type
);

/// Get output tensor type.
MHLOStatusCode mhlo_executable_output_type(
    MHLOExecutableRef executable,
    int32_t index,
    int64_t* out_shape,
    int32_t* out_rank,
    MHLOElementType* out_element_type
);

// ============================================================================
// Buffer API
// ============================================================================

/// Create a buffer from host data.
/// - Parameters:
///   - client: The client.
///   - data: Host data pointer.
///   - data_size: Size of data in bytes.
///   - shape: Tensor shape array.
///   - rank: Number of dimensions.
///   - element_type: Element type.
///   - out_buffer: Output buffer handle.
/// - Returns: MHLO_OK on success, error code otherwise.
MHLOStatusCode mhlo_buffer_create(
    MHLOClientRef client,
    const void* data,
    size_t data_size,
    const int64_t* shape,
    int32_t rank,
    MHLOElementType element_type,
    MHLOBufferRef* out_buffer
);

/// Create an uninitialized buffer.
MHLOStatusCode mhlo_buffer_create_uninitialized(
    MHLOClientRef client,
    const int64_t* shape,
    int32_t rank,
    MHLOElementType element_type,
    MHLOBufferRef* out_buffer
);

/// Destroy a buffer.
void mhlo_buffer_destroy(MHLOBufferRef buffer);

/// Get buffer shape.
/// - Parameters:
///   - buffer: The buffer.
///   - out_shape: Output shape array (must be pre-allocated with at least rank elements).
///   - out_rank: Output rank.
void mhlo_buffer_shape(
    MHLOBufferRef buffer,
    int64_t* out_shape,
    int32_t* out_rank
);

/// Get buffer rank (number of dimensions).
int32_t mhlo_buffer_rank(MHLOBufferRef buffer);

/// Get buffer element type.
MHLOElementType mhlo_buffer_element_type(MHLOBufferRef buffer);

/// Get buffer size in bytes.
size_t mhlo_buffer_byte_count(MHLOBufferRef buffer);

/// Copy buffer contents to host.
/// - Parameters:
///   - buffer: The buffer.
///   - out_data: Output data pointer (must be pre-allocated).
///   - data_size: Size of output buffer in bytes.
/// - Returns: MHLO_OK on success, error code otherwise.
MHLOStatusCode mhlo_buffer_to_host(
    MHLOBufferRef buffer,
    void* out_data,
    size_t data_size
);

// ============================================================================
// Execution API
// ============================================================================

/// Execute a compiled program.
/// - Parameters:
///   - executable: The executable.
///   - inputs: Array of input buffer handles.
///   - num_inputs: Number of inputs.
///   - out_outputs: Output array for output buffer handles (must be pre-allocated).
///   - out_num_outputs: Output number of outputs.
/// - Returns: MHLO_OK on success, error code otherwise.
MHLOStatusCode mhlo_execute(
    MHLOExecutableRef executable,
    const MHLOBufferRef* inputs,
    int32_t num_inputs,
    MHLOBufferRef* out_outputs,
    int32_t* out_num_outputs
);

/// Execute with timing information.
/// - Parameters:
///   - executable: The executable.
///   - inputs: Array of input buffer handles.
///   - num_inputs: Number of inputs.
///   - out_outputs: Output array for output buffer handles.
///   - out_num_outputs: Output number of outputs.
///   - out_encode_time: Encode time in seconds (can be NULL).
///   - out_gpu_time: GPU execution time in seconds (can be NULL).
///   - out_total_time: Total wall-clock time in seconds (can be NULL).
/// - Returns: MHLO_OK on success, error code otherwise.
MHLOStatusCode mhlo_execute_with_timing(
    MHLOExecutableRef executable,
    const MHLOBufferRef* inputs,
    int32_t num_inputs,
    MHLOBufferRef* out_outputs,
    int32_t* out_num_outputs,
    double* out_encode_time,
    double* out_gpu_time,
    double* out_total_time
);

// ============================================================================
// Utility
// ============================================================================

/// Free a string returned by the API.
void mhlo_free_string(const char* str);

#ifdef __cplusplus
}
#endif

#endif // METALHLO_H
