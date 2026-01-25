// main.c
// MetalHLO C Example
//
// Demonstrates using the MetalHLO C API to compile and execute StableHLO programs.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "metalhlo.h"

// Helper macro for error checking
#define CHECK_STATUS(status, msg) do { \
    if ((status) != MHLO_OK) { \
        const char* err = mhlo_get_last_error(); \
        fprintf(stderr, "Error: %s - %s\n", (msg), err ? err : "Unknown error"); \
        return 1; \
    } \
} while(0)

int main(int argc, char* argv[]) {
    printf("MetalHLO C Example\n");
    printf("==================\n\n");

    // Print version
    const char* version = mhlo_version();
    printf("MetalHLO version: %s\n\n", version);

    // Create client
    MHLOClientRef client = NULL;
    MHLOStatusCode status = mhlo_client_create(&client);
    CHECK_STATUS(status, "Failed to create client");

    // Get device name
    char* device_name = mhlo_client_device_name(client);
    printf("Device: %s\n\n", device_name);
    mhlo_free_string(device_name);

    // ========================================================================
    // Example 1: Simple addition
    // ========================================================================
    printf("Example 1: Simple Addition\n");
    printf("--------------------------\n");

    const char* add_mlir = 
        "module @simple_add {\n"
        "  func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>) {\n"
        "    %0 = stablehlo.add %a, %b : tensor<4xf32>\n"
        "    return %0 : tensor<4xf32>\n"
        "  }\n"
        "}\n";

    MHLOExecutableRef add_exe = NULL;
    status = mhlo_compile(client, add_mlir, &add_exe);
    CHECK_STATUS(status, "Failed to compile addition module");

    printf("Compiled addition module:\n");
    printf("  Inputs: %d\n", mhlo_executable_input_count(add_exe));
    printf("  Outputs: %d\n", mhlo_executable_output_count(add_exe));

    // Create input buffers
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
    int64_t shape[] = {4};

    MHLOBufferRef buffer_a = NULL;
    MHLOBufferRef buffer_b = NULL;
    
    status = mhlo_buffer_create(client, a_data, sizeof(a_data), shape, 1, MHLO_F32, &buffer_a);
    CHECK_STATUS(status, "Failed to create buffer A");
    
    status = mhlo_buffer_create(client, b_data, sizeof(b_data), shape, 1, MHLO_F32, &buffer_b);
    CHECK_STATUS(status, "Failed to create buffer B");

    // Execute
    MHLOBufferRef inputs[] = {buffer_a, buffer_b};
    MHLOBufferRef outputs[1] = {NULL};
    int32_t num_outputs = 0;

    status = mhlo_execute(add_exe, inputs, 2, outputs, &num_outputs);
    CHECK_STATUS(status, "Failed to execute addition");

    // Read result
    float result[4];
    status = mhlo_buffer_to_host(outputs[0], result, sizeof(result));
    CHECK_STATUS(status, "Failed to read result");

    printf("  a = [%.1f, %.1f, %.1f, %.1f]\n", a_data[0], a_data[1], a_data[2], a_data[3]);
    printf("  b = [%.1f, %.1f, %.1f, %.1f]\n", b_data[0], b_data[1], b_data[2], b_data[3]);
    printf("  a + b = [%.1f, %.1f, %.1f, %.1f]\n\n", result[0], result[1], result[2], result[3]);

    // Cleanup example 1
    mhlo_buffer_destroy(buffer_a);
    mhlo_buffer_destroy(buffer_b);
    mhlo_buffer_destroy(outputs[0]);
    mhlo_executable_destroy(add_exe);

    // ========================================================================
    // Example 2: Matrix multiplication
    // ========================================================================
    printf("Example 2: Matrix Multiplication\n");
    printf("--------------------------------\n");

    const char* matmul_mlir = 
        "module @matmul {\n"
        "  func.func @main(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> (tensor<2x2xf32>) {\n"
        "    %0 = stablehlo.dot %a, %b : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>\n"
        "    return %0 : tensor<2x2xf32>\n"
        "  }\n"
        "}\n";

    MHLOExecutableRef matmul_exe = NULL;
    status = mhlo_compile(client, matmul_mlir, &matmul_exe);
    CHECK_STATUS(status, "Failed to compile matmul module");

    // Create 2x3 and 3x2 matrices
    float mat_a[] = {1, 2, 3, 4, 5, 6};  // 2x3
    float mat_b[] = {1, 2, 3, 4, 5, 6};  // 3x2
    int64_t shape_a[] = {2, 3};
    int64_t shape_b[] = {3, 2};

    MHLOBufferRef buf_a = NULL, buf_b = NULL;
    status = mhlo_buffer_create(client, mat_a, sizeof(mat_a), shape_a, 2, MHLO_F32, &buf_a);
    CHECK_STATUS(status, "Failed to create matrix A");
    
    status = mhlo_buffer_create(client, mat_b, sizeof(mat_b), shape_b, 2, MHLO_F32, &buf_b);
    CHECK_STATUS(status, "Failed to create matrix B");

    // Execute with timing
    MHLOBufferRef mm_inputs[] = {buf_a, buf_b};
    MHLOBufferRef mm_outputs[1] = {NULL};
    int32_t mm_num_outputs = 0;
    double encode_time, gpu_time, total_time;

    status = mhlo_execute_with_timing(matmul_exe, mm_inputs, 2, mm_outputs, &mm_num_outputs,
                                       &encode_time, &gpu_time, &total_time);
    CHECK_STATUS(status, "Failed to execute matmul");

    // Read result
    float mm_result[4];
    status = mhlo_buffer_to_host(mm_outputs[0], mm_result, sizeof(mm_result));
    CHECK_STATUS(status, "Failed to read matmul result");

    printf("  A (2x3) = [[1, 2, 3], [4, 5, 6]]\n");
    printf("  B (3x2) = [[1, 2], [3, 4], [5, 6]]\n");
    printf("  A @ B (2x2) = [[%.0f, %.0f], [%.0f, %.0f]]\n", 
           mm_result[0], mm_result[1], mm_result[2], mm_result[3]);
    printf("\n  Timing:\n");
    printf("    Encode: %.3f ms\n", encode_time * 1000);
    printf("    GPU:    %.3f ms\n", gpu_time * 1000);
    printf("    Total:  %.3f ms\n\n", total_time * 1000);

    // Cleanup example 2
    mhlo_buffer_destroy(buf_a);
    mhlo_buffer_destroy(buf_b);
    mhlo_buffer_destroy(mm_outputs[0]);
    mhlo_executable_destroy(matmul_exe);

    // ========================================================================
    // Example 3: Neural network layer (linear + ReLU)
    // ========================================================================
    printf("Example 3: Linear Layer with ReLU\n");
    printf("---------------------------------\n");

    const char* linear_mlir = 
        "module @linear_relu {\n"
        "  func.func @main(%x: tensor<1x4xf32>, %w: tensor<4x2xf32>, %b: tensor<1x2xf32>) -> (tensor<1x2xf32>) {\n"
        "    // Linear: y = x @ w + b\n"
        "    %mm = stablehlo.dot %x, %w : (tensor<1x4xf32>, tensor<4x2xf32>) -> tensor<1x2xf32>\n"
        "    %linear = stablehlo.add %mm, %b : tensor<1x2xf32>\n"
        "    // ReLU: max(0, y)\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<1x2xf32>\n"
        "    %relu = stablehlo.maximum %linear, %zero : tensor<1x2xf32>\n"
        "    return %relu : tensor<1x2xf32>\n"
        "  }\n"
        "}\n";

    MHLOExecutableRef linear_exe = NULL;
    status = mhlo_compile(client, linear_mlir, &linear_exe);
    CHECK_STATUS(status, "Failed to compile linear+relu module");

    // Input, weights, bias
    float x_data[] = {1.0f, -2.0f, 3.0f, -4.0f};  // 1x4
    float w_data[] = {0.5f, -0.5f, 0.3f, 0.7f, -0.2f, 0.8f, 0.1f, -0.3f};  // 4x2
    float bias_data[] = {0.1f, -0.2f};  // 1x2

    int64_t x_shape[] = {1, 4};
    int64_t w_shape[] = {4, 2};
    int64_t b_shape[] = {1, 2};

    MHLOBufferRef x_buf = NULL, w_buf = NULL, b_buf = NULL;
    status = mhlo_buffer_create(client, x_data, sizeof(x_data), x_shape, 2, MHLO_F32, &x_buf);
    CHECK_STATUS(status, "Failed to create x buffer");
    status = mhlo_buffer_create(client, w_data, sizeof(w_data), w_shape, 2, MHLO_F32, &w_buf);
    CHECK_STATUS(status, "Failed to create w buffer");
    status = mhlo_buffer_create(client, bias_data, sizeof(bias_data), b_shape, 2, MHLO_F32, &b_buf);
    CHECK_STATUS(status, "Failed to create bias buffer");

    MHLOBufferRef linear_inputs[] = {x_buf, w_buf, b_buf};
    MHLOBufferRef linear_outputs[1] = {NULL};
    int32_t linear_num_outputs = 0;

    status = mhlo_execute(linear_exe, linear_inputs, 3, linear_outputs, &linear_num_outputs);
    CHECK_STATUS(status, "Failed to execute linear+relu");

    float linear_result[2];
    status = mhlo_buffer_to_host(linear_outputs[0], linear_result, sizeof(linear_result));
    CHECK_STATUS(status, "Failed to read linear+relu result");

    printf("  Input x = [%.1f, %.1f, %.1f, %.1f]\n", x_data[0], x_data[1], x_data[2], x_data[3]);
    printf("  Output = [%.4f, %.4f] (after linear + ReLU)\n\n", linear_result[0], linear_result[1]);

    // Cleanup example 3
    mhlo_buffer_destroy(x_buf);
    mhlo_buffer_destroy(w_buf);
    mhlo_buffer_destroy(b_buf);
    mhlo_buffer_destroy(linear_outputs[0]);
    mhlo_executable_destroy(linear_exe);

    // Cleanup
    mhlo_client_destroy(client);

    printf("All examples completed successfully!\n");
    return 0;
}
