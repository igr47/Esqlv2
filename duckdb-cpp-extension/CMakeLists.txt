cmake_minimum_required(VERSION 3.20...3.29)
project(duckdb_ai_extension)

# Include required CMake modules
include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)

# ============================================================================
# Find DuckDB - Enhanced with project-specific paths
# ============================================================================
# Set DuckDB_DIR if provided via command line
if(DEFINED DuckDB_DIR AND NOT DuckDB_DIR STREQUAL "")
    set(CMAKE_PREFIX_PATH ${DuckDB_DIR} ${CMAKE_PREFIX_PATH})
endif()

# Try to find DuckDB via config mode first
find_package(DuckDB CONFIG QUIET)

if(DuckDB_FOUND)
    message(STATUS "✅ Found DuckDB via config: ${DUCKDB_LIBRARIES}")
    get_target_property(DUCKDB_INCLUDE_DIR duckdb INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT DUCKDB_INCLUDE_DIR)
        find_path(DUCKDB_INCLUDE_DIR
            NAMES duckdb.hpp duckdb.h
            PATHS
                ${DuckDB_DIR}/include
                ${DuckDB_DIR}/src/include
                ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/src/include
                /usr/local/include
                /usr/include
        )
    endif()

    find_library(DUCKDB_LIBRARY_PATH
        NAMES duckdb libduckdb.so libduckdb.dylib duckdb.lib
        PATHS
            ${DuckDB_DIR}
            ${DuckDB_DIR}/build
            ${DuckDB_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/duckdb
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build
    )
else()
    # Manual fallback with project-specific paths
    message(STATUS "DuckDB config not found, attempting manual search in project directories...")

    # Common locations including your project structure
    find_path(DUCKDB_INCLUDE_DIR
        NAMES duckdb.hpp duckdb.h
        PATHS
            ${DuckDB_DIR}/include
            ${DuckDB_DIR}/src/include
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/src/include          # <-- Your duckdb source include
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/debug/include  # <-- Built includes (if installed)
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/release/include
            /usr/local/include
            /usr/include
    )

    find_library(DUCKDB_LIBRARY_PATH
        NAMES duckdb libduckdb.so libduckdb.dylib duckdb.lib
        PATHS
            ${DuckDB_DIR}
            ${DuckDB_DIR}/build
            ${DuckDB_DIR}/build/debug
            ${DuckDB_DIR}/build/release
            ${DuckDB_DIR}/build/debug/src        # <-- ADDED: inside src
            ${DuckDB_DIR}/build/release/src       # <-- ADDED
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/debug
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/release
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/debug/src   # <-- ADDED
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/release/src  # <-- ADDED
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build
            ${CMAKE_CURRENT_BINARY_DIR}/duckdb
            /usr/local/lib
            /usr/lib

    )
endif()

# Validate findings
if(NOT DUCKDB_INCLUDE_DIR)
    message(FATAL_ERROR "DuckDB headers (duckdb.hpp) not found. Searched in:\n"
                        "  - ${DuckDB_DIR}/include\n"
                        "  - ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/src/include\n"
                        "  - ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/*/include\n"
                        "Please ensure DuckDB is built and headers are available.")
endif()

if(NOT DUCKDB_LIBRARY_PATH)
    message(FATAL_ERROR "DuckDB library not found. Searched in:\n"
                        "  - ${DuckDB_DIR}/build/debug\n"
                        "  - ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/debug\n"
                        "  - ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/release\n"
                        "Please set -DDuckDB_DIR=/path/to/duckdb/build if not in standard location.")
endif()

message(STATUS "✅ Found DuckDB headers: ${DUCKDB_INCLUDE_DIR}")
message(STATUS "✅ Found DuckDB library: ${DUCKDB_LIBRARY_PATH}")

# Create an imported target for DuckDB
if(NOT TARGET duckdb)
    add_library(duckdb UNKNOWN IMPORTED)
    set_target_properties(duckdb PROPERTIES
        IMPORTED_LOCATION "${DUCKDB_LIBRARY_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${DUCKDB_INCLUDE_DIR}"
    )
endif()

# ============================================================================
# Extension Configuration
# ============================================================================
set(EXTENSION_NAME "ai_extension")
option(DUCKDB_WASM_EXTENSION "Whether compiling for Wasm target" OFF)
option(BUILD_TESTING "Build tests" OFF)

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

# Hide symbols
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(VISIBILITY_INLINES_HIDDEN ON)
# Remove or comment out the -s flag to avoid the unused warning
# set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

# ============================================================================
# Find Required Packages
# ============================================================================

# Threads
find_package(Threads REQUIRED)

# OpenMP detection - Enhanced for better compatibility
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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
    message(STATUS "✅ Found OpenMP")
    message(STATUS "   OpenMP CXX Flags: ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP is required for LightGBM optimization. Please install OpenMP.\n"
                        "On Ubuntu/Debian: sudo apt-get install libomp-dev\n"
                        "On macOS: brew install libomp\n"
                        "On Fedora: sudo dnf install libomp-devel")
endif()

# ============================================================================
# nlohmann/json
# ============================================================================
find_package(nlohmann_json 3.11.2 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "Downloading nlohmann/json...")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ============================================================================
# Find LightGBM
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

# Build LightGBM if not found
if(NOT LIGHTGBM_LIBRARY OR NOT LIGHTGBM_INCLUDE_DIR)
    message(STATUS "LightGBM not found in system, attempting to build from source...")

    include(FetchContent)
    FetchContent_Declare(
        lightgbm
        GIT_REPOSITORY https://github.com/microsoft/LightGBM.git
        GIT_TAG v4.5.0
        GIT_PROGRESS TRUE
    )

    if(NOT lightgbm_POPULATED)
        message(STATUS "Downloading and building LightGBM...")
        FetchContent_Populate(lightgbm)

        set(LIGHTGBM_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/lightgbm-build)

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
            OUTPUT_QUIET
            ERROR_QUIET
        )

        if(NOT LIGHTGBM_CMAKE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to configure LightGBM")
        endif()

        execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${LIGHTGBM_BUILD_DIR} -- -j4
            RESULT_VARIABLE LIGHTGBM_BUILD_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )

        if(NOT LIGHTGBM_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to build LightGBM")
        endif()

        # Find the built library
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
        message(STATUS "✅ LightGBM built successfully")
    endif()
endif()

# Final LightGBM check
if(LIGHTGBM_LIBRARY AND LIGHTGBM_INCLUDE_DIR)
    message(STATUS "✅ Found LightGBM library: ${LIGHTGBM_LIBRARY}")
    message(STATUS "✅ Found LightGBM headers: ${LIGHTGBM_INCLUDE_DIR}")
    add_definitions(-DHAVE_LIGHTGBM)
else()
    message(WARNING "⚠️ LightGBM not found. AI/ML features will be disabled.")
endif()

# ============================================================================
# Source Files
# ============================================================================
set(EXTENSION_SOURCES
    src/ai_extension.cpp
    src/parser_extension.cpp
    src/create_model_parser.cpp
    src/train_model_function.cpp
    src/algorithm_registry.cpp
    src/model_registry.cpp
    src/lightgbm_train_splits.cpp
    src/lightgbm_detailed_metrics.cpp
    src/lightgbm_metric_extraction.cpp
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

# Set output directory and properties
set_target_properties(${EXTENSION_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extension"
    PREFIX ""
    SUFFIX ".duckdb_extension"
)

# ============================================================================
# CRITICAL FIX: Define LIGHTGBM_ARROW_H_ to prevent Arrow redefinition
# ============================================================================
target_compile_definitions(${EXTENSION_NAME} PRIVATE
    LIGHTGBM_ARROW_H_  # This prevents LightGBM from including its own arrow.h
)

# Also add these to ensure no conflicts
target_compile_definitions(${EXTENSION_NAME} PRIVATE
    ARROW_SCHEMA_ALREADY_DEFINED
    ARROW_ARRAY_ALREADY_DEFINED
)

# ============================================================================
# Include Directories
# ============================================================================
target_include_directories(${EXTENSION_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/duckdb_capi
    ${DUCKDB_INCLUDE_DIR}
    ${LIGHTGBM_INCLUDE_DIR}
)

# ============================================================================
# Link Libraries
# ============================================================================
target_link_libraries(${EXTENSION_NAME}
    PRIVATE
    duckdb  # Use the imported target
    Threads::Threads
)

# Add LightGBM if found
if(LIGHTGBM_LIBRARY)
    target_link_libraries(${EXTENSION_NAME} PRIVATE ${LIGHTGBM_LIBRARY})
endif()

# Add OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(${EXTENSION_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${EXTENSION_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Add nlohmann/json
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(${EXTENSION_NAME} PRIVATE nlohmann_json::nlohmann_json)
elseif(nlohmann_json_FOUND)
    target_include_directories(${EXTENSION_NAME} PRIVATE ${nlohmann_json_INCLUDE_DIR})
endif()

# Add pthread on Unix-like systems
if(UNIX)
    find_library(PTHREAD_LIBRARY pthread)
    if(PTHREAD_LIBRARY)
        target_link_libraries(${EXTENSION_NAME} PRIVATE ${PTHREAD_LIBRARY})
    endif()
endif()

# Add math library
target_link_libraries(${EXTENSION_NAME} PRIVATE m)

# ============================================================================
# Compiler Options
# ============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    #target_compile_options(${EXTENSION_NAME} PRIVATE -O3 -march=native -flto)
    target_compile_options(${EXTENSION_NAME} PRIVATE -O3 -march=native)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${EXTENSION_NAME} PRIVATE -g -O0 -DDEBUG)
endif()

target_compile_options(${EXTENSION_NAME} PRIVATE -Wall -Wextra)

# Suppress specific warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${EXTENSION_NAME} PRIVATE
        -Wno-unused-parameter  # Suppress unused parameter warnings
        -Wno-unused-but-set-variable  # Suppress unused but set variable warnings
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(${EXTENSION_NAME} PRIVATE
        -Wno-unused-parameter
        -Wno-unused-but-set-variable
    )
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS ${EXTENSION_NAME}
    LIBRARY DESTINATION lib/duckdb/extensions
)

# ============================================================================
# Testing
# ============================================================================
if(BUILD_TESTING)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_lightgbm.cpp")
        add_executable(test_lightgbm test/test_lightgbm.cpp)
        target_link_libraries(test_lightgbm PRIVATE ${EXTENSION_NAME} Threads::Threads)
        target_include_directories(test_lightgbm PRIVATE
            src/include
            ${DUCKDB_INCLUDE_DIR}
        )

        if(OpenMP_CXX_FOUND)
            target_link_libraries(test_lightgbm PRIVATE OpenMP::OpenMP_CXX)
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
message(STATUS "")
message(STATUS "=========================================")
message(STATUS "DuckDB AI Extension Build Configuration:")
message(STATUS "  Extension Name: ${EXTENSION_NAME}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: 17")
message(STATUS "  DuckDB: ${DUCKDB_INCLUDE_DIR}")
message(STATUS "  LightGBM: ${LIGHTGBM_LIBRARY}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP Flags: ${OpenMP_CXX_FLAGS}")
endif()
message(STATUS "  Threads: ${Threads_FOUND}")
message(STATUS "=========================================")
message(STATUS "")
