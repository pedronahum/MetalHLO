cmake_minimum_required(VERSION 3.20)
project(MetalHLOExample C)

set(CMAKE_C_STANDARD 11)

# Path to the MetalHLO library
# Build with: swift build -c release
set(METALHLO_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.build/release")

# Add the MetalHLO include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../Sources/CMetalHLO/include)

# Find the dynamic library
find_library(CMETALHLO_LIB
    NAMES CMetalHLO libCMetalHLO
    PATHS ${METALHLO_BUILD_DIR}
    NO_DEFAULT_PATH
)

if(NOT CMETALHLO_LIB)
    message(FATAL_ERROR "CMetalHLO library not found. Please build MetalHLO first with: swift build -c release")
endif()

# Create executable
add_executable(metalhlo_example main.c)

# Link against CMetalHLO
target_link_libraries(metalhlo_example ${CMETALHLO_LIB})

# Set rpath to find the dynamic library at runtime
set_target_properties(metalhlo_example PROPERTIES
    BUILD_RPATH ${METALHLO_BUILD_DIR}
    INSTALL_RPATH ${METALHLO_BUILD_DIR}
)

# Print build instructions
message(STATUS "")
message(STATUS "Build Instructions:")
message(STATUS "  1. First build MetalHLO: swift build -c release")
message(STATUS "  2. Then build this example:")
message(STATUS "     mkdir build && cd build")
message(STATUS "     cmake ..")
message(STATUS "     make")
message(STATUS "  3. Run: ./metalhlo_example")
message(STATUS "")
