cmake_minimum_required(VERSION 3.20...3.29)
project(duckdb_ai_extension)

# ============================================================================
# Extension Configuration
# ============================================================================
set(EXTENSION_NAME "ai_extension")
option(DUCKDB_WASM_EXTENSION "Whether compiling for Wasm target" OFF)
option(BUILD_TESTING "Build tests" OFF)  # Default to OFF until test files exist

# DuckDB API version configuration
if (DEFINED TARGET_DUCKDB_VERSION_MAJOR)
    add_definitions(-DDUCKDB_EXTENSION_API_VERSION_MAJOR=${TARGET_DUCKDB_VERSION_MAJOR})
endif()
if (DEFINED TARGET_DUCKDB_VERSION_MINOR)
    add_definitions(-DDUCKDB_EXTENSION_API_VERSION_MINOR=${TARGET_DUCKDB_VERSION_MINOR})
endif()
if (DEFINED TARGET_DUCKDB_VERSION_PATCH)
    add_definitions(-DDUCKDB_EXTENSION_API_VERSION_PATCH=${TARGET_DUCKDB_VERSION_PATCH})
endif()

add_definitions(-DDUCKDB_EXTENSION_NAME=${EXTENSION_NAME})

if (DEFINED DUCKDB_EXTENSION_API_VERSION_UNSTABLE)
    add_definitions(-DDUCKDB_EXTENSION_API_VERSION_UNSTABLE=${DUCKDB_EXTENSION_API_VERSION_UNSTABLE})
endif()

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# C++ Standard Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Hide symbols (from your DuckDB template)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

# ============================================================================
# Find Required Packages
# ============================================================================

# Threads (required for parallel operations)
find_package(Threads REQUIRED)

# OpenMP (for LightGBM parallelization) - Fixed for clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # For Apple Clang or regular Clang with OpenMP
    if(APPLE)
        # macOS with Homebrew libomp
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/lib/libomp.dylib" CACHE PATH "OpenMP library")
        set(OpenMP_CXX_LIBRARIES ${OpenMP_omp_LIBRARY})
        set(OpenMP_C_LIBRARIES ${OpenMP_omp_LIBRARY})
    else()
        # Linux with clang and libomp
        set(OpenMP_CXX_FLAGS "-fopenmp=libomp")
        set(OpenMP_C_FLAGS "-fopenmp=libomp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "omp" CACHE STRING "OpenMP library")
    endif()
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    message(STATUS "OpenMP CXX Flags: ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP is required for LightGBM optimization. Please install OpenMP.\n"
                        "On Ubuntu/Debian: sudo apt-get install libomp-dev\n"
                        "On macOS: brew install libomp\n"
                        "On Fedora: sudo dnf install libomp-devel")
endif()

# nlohmann/json (for JSON serialization)
find_package(nlohmann_json 3.11.2 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "Downloading nlohmann/json...")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ============================================================================
# Find LightGBM (with multiple fallback options)
# ============================================================================

# Try to find LightGBM in standard locations
find_library(LIGHTGBM_LIBRARY
    NAMES lightgbm liblightgbm.so liblightgbm.dylib lightgbm.lib lib_lightgbm.a lib_lightgbm.so
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        /opt/homebrew/lib
        ${CMAKE_SOURCE_DIR}/libs
        ${CMAKE_SOURCE_DIR}/LightGBM/build
        ${CMAKE_SOURCE_DIR}/deps/LightGBM/build
        ${CMAKE_SOURCE_DIR}/third_party/LightGBM/build
        $ENV{LIGHTGBM_ROOT}/lib
)

find_path(LIGHTGBM_INCLUDE_DIR
    NAMES LightGBM/c_api.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/local/include
        /opt/homebrew/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/LightGBM/include
        ${CMAKE_SOURCE_DIR}/deps/LightGBM/include
        ${CMAKE_SOURCE_DIR}/third_party/LightGBM/include
        $ENV{LIGHTGBM_ROOT}/include
)

# If LightGBM not found, try to download and build it
if(NOT LIGHTGBM_LIBRARY OR NOT LIGHTGBM_INCLUDE_DIR)
    message(STATUS "LightGBM not found in system, attempting to build from source...")

    include(FetchContent)
    FetchContent_Declare(
        lightgbm
        GIT_REPOSITORY https://github.com/microsoft/LightGBM.git
        GIT_TAG v4.5.0
        GIT_PROGRESS TRUE
    )

    # Only build if we're actually going to use it
    if(NOT lightgbm_POPULATED)
        message(STATUS "Downloading and building LightGBM...")
        FetchContent_Populate(lightgbm)

        # Add LightGBM as an external project
        set(LIGHTGBM_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/lightgbm-build)

        # Pass OpenMP flags to LightGBM build
        set(LIGHTGBM_CMAKE_ARGS
            -DUSE_OPENMP=ON
            -DBUILD_STATIC_LIB=OFF
            -DBUILD_SHARED_LIB=ON
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        )

        if(OpenMP_CXX_FOUND)
            list(APPEND LIGHTGBM_CMAKE_ARGS
                -DOpenMP_CXX_FLAGS=${OpenMP_CXX_FLAGS}
                -DOpenMP_CXX_LIB_NAMES=${OpenMP_CXX_LIB_NAMES}
            )
            if(DEFINED OpenMP_omp_LIBRARY)
                list(APPEND LIGHTGBM_CMAKE_ARGS -DOpenMP_omp_LIBRARY=${OpenMP_omp_LIBRARY})
            endif()
        endif()

        execute_process(
            COMMAND ${CMAKE_COMMAND}
                -S ${lightgbm_SOURCE_DIR}
                -B ${LIGHTGBM_BUILD_DIR}
                ${LIGHTGBM_CMAKE_ARGS}
            RESULT_VARIABLE LIGHTGBM_CMAKE_RESULT
        )

        if(NOT LIGHTGBM_CMAKE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to configure LightGBM")
        endif()

        execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${LIGHTGBM_BUILD_DIR} -- -j4
            RESULT_VARIABLE LIGHTGBM_BUILD_RESULT
        )

        if(NOT LIGHTGBM_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to build LightGBM")
        endif()

        # Set the paths to the built library
        if(EXISTS "${LIGHTGBM_BUILD_DIR}/lib_lightgbm.so")
            set(LIGHTGBM_LIBRARY "${LIGHTGBM_BUILD_DIR}/lib_lightgbm.so")
        elseif(EXISTS "${LIGHTGBM_BUILD_DIR}/Release/lib_lightgbm.so")
            set(LIGHTGBM_LIBRARY "${LIGHTGBM_BUILD_DIR}/Release/lib_lightgbm.so")
        elseif(EXISTS "${LIGHTGBM_BUILD_DIR}/lib_lightgbm.dylib")
            set(LIGHTGBM_LIBRARY "${LIGHTGBM_BUILD_DIR}/lib_lightgbm.dylib")
        elseif(EXISTS "${LIGHTGBM_BUILD_DIR}/lib_lightgbm.a")
            set(LIGHTGBM_LIBRARY "${LIGHTGBM_BUILD_DIR}/lib_lightgbm.a")
        endif()

        set(LIGHTGBM_INCLUDE_DIR ${lightgbm_SOURCE_DIR}/include)

        message(STATUS "LightGBM built successfully")
    endif()
endif()

# Check for LightGBM availability
if(LIGHTGBM_LIBRARY AND LIGHTGBM_INCLUDE_DIR)
    message(STATUS "Found LightGBM library: ${LIGHTGBM_LIBRARY}")
    message(STATUS "Found LightGBM headers: ${LIGHTGBM_INCLUDE_DIR}")
    add_definitions(-DHAVE_LIGHTGBM)
else()
    message(WARNING "LightGBM not found. AI/ML features will be disabled.")
endif()

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    src/include
    duckdb_capi
)

if(LIGHTGBM_INCLUDE_DIR)
    include_directories(${LIGHTGBM_INCLUDE_DIR})
endif()

# ============================================================================
# Source Files
# ============================================================================

# Core extension sources (matching your DuckDB pattern)
set(EXTENSION_SOURCES
    src/ai_extension.cpp
    src/parser_extension.cpp
    src/parser/create_model_parser.cpp
    src/train_model_function.cpp
    src/algorithm_registry.cpp
    src/model_registry.cpp
    src/lightgbm_train_splits.cpp
    src/lightgbm_detailed_metrics.cpp
    src/lightgbm_training_metrics.cpp
)

# ============================================================================
# Create Extension Library
# ============================================================================

if (DUCKDB_WASM_EXTENSION)
    add_library(${EXTENSION_NAME} STATIC ${EXTENSION_SOURCES})
else()
    add_library(${EXTENSION_NAME} SHARED ${EXTENSION_SOURCES})
endif()

# ============================================================================
# Link Libraries
# ============================================================================

# DuckDB C API headers are in duckdb_capi
target_include_directories(${EXTENSION_NAME} PRIVATE duckdb_capi)

# Link required libraries
target_link_libraries(${EXTENSION_NAME}
    Threads::Threads  # Correct way to link Threads
)

# Add LightGBM if found
if(LIGHTGBM_LIBRARY)
    target_link_libraries(${EXTENSION_NAME} ${LIGHTGBM_LIBRARY})
endif()

# Add OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(${EXTENSION_NAME} OpenMP::OpenMP_CXX)
    target_compile_options(${EXTENSION_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Add nlohmann/json
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(${EXTENSION_NAME} nlohmann_json::nlohmann_json)
elseif(nlohmann_json_FOUND)
    target_include_directories(${EXTENSION_NAME} PRIVATE ${nlohmann_json_INCLUDE_DIR})
endif()

# Add pthread on Unix-like systems
if(UNIX)
    target_link_libraries(${EXTENSION_NAME} pthread)
endif()

# Add math library
target_link_libraries(${EXTENSION_NAME} m)

# ============================================================================
# Compiler Options
# ============================================================================

# Architecture-specific optimizations (from your ESQL project)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${EXTENSION_NAME} PRIVATE -O3 -march=native -flto)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${EXTENSION_NAME} PRIVATE -g -O0 -DDEBUG)
endif()

# Add warning flags
target_compile_options(${EXTENSION_NAME} PRIVATE -Wall -Wextra)

# ============================================================================
# Installation (from your DuckDB template)
# ============================================================================

# Set output directory
set_target_properties(${EXTENSION_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extension"
    PREFIX ""
)

# Install the extension
install(TARGETS ${EXTENSION_NAME}
    LIBRARY DESTINATION lib/duckdb/extensions
)

# ============================================================================
# Testing (optional) - Fixed to check for file existence
# ============================================================================
if(BUILD_TESTING)
    enable_testing()

    # Check if test file exists before adding executable
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_lightgbm.cpp")
        add_executable(test_lightgbm test/test_lightgbm.cpp)
        target_link_libraries(test_lightgbm ${EXTENSION_NAME} Threads::Threads)
        target_include_directories(test_lightgbm PRIVATE src/include)

        if(OpenMP_CXX_FOUND)
            target_link_libraries(test_lightgbm OpenMP::OpenMP_CXX)
            target_compile_options(test_lightgbm PRIVATE ${OpenMP_CXX_FLAGS})
        endif()

        add_test(NAME LightGBMTest COMMAND test_lightgbm)
    else()
        message(STATUS "Test file test/test_lightgbm.cpp not found - skipping test build")
    endif()
endif()

# ============================================================================
# Print Summary
# ============================================================================
message(STATUS "=========================================")
message(STATUS "DuckDB AI Extension Build Configuration:")
message(STATUS "  Extension Name: ${EXTENSION_NAME}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: 17")
message(STATUS "  LightGBM: ${LIGHTGBM_LIBRARY}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "  OpenMP Flags: ${OpenMP_CXX_FLAGS}")
message(STATUS "  Threads: ${Threads_FOUND}")
message(STATUS "=========================================")
