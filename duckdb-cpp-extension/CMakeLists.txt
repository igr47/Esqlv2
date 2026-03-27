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
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/src/include
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/debug/include
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
            ${DuckDB_DIR}/build/debug/src
            ${DuckDB_DIR}/build/release/src
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/debug
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/release
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/debug/src
            ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/build/release/src
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

# ============================================================================
# Extension Configuration
# ============================================================================
set(EXTENSION_NAME "ai_extension")
option(DUCKDB_WASM_EXTENSION "Whether compiling for Wasm target" OFF)
option(BUILD_TESTING "Build tests" OFF)

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

# ============================================================================
# Find Required Packages
# ============================================================================

# Threads
find_package(Threads REQUIRED)

# OpenMP detection
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(APPLE)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/lib/libomp.dylib" CACHE PATH "OpenMP library")
        set(OpenMP_CXX_LIBRARIES ${OpenMP_omp_LIBRARY})
        set(OpenMP_C_LIBRARIES ${OpenMP_omp_LIBRARY})
    else()
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
    message(FATAL_ERROR "OpenMP is required for LightGBM optimization.")
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
    src/lightgbm_model_impl.cpp
    src/show_models_function.cpp
    src/predict_function.cpp
    src/lightgbm_model_schema_impl.cpp
    src/train_model_function.cpp
    src/algorithm_registry.cpp
    src/model_registry.cpp
    src/lightgbm_train_splits.cpp
    src/lightgbm_detailed_metrics.cpp
    src/lightgbm_metric_extraction.cpp
    src/lightgbm_training_metrics.cpp
)

# ============================================================================
# Build Extension - Direct approach without helper
# ============================================================================

# Create shared library
add_library(${EXTENSION_NAME} SHARED ${EXTENSION_SOURCES})

# Set output directory and properties
set_target_properties(${EXTENSION_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extension"
    PREFIX ""
    SUFFIX ".duckdb_extension"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Add compile definitions to prevent Arrow conflicts
target_compile_definitions(${EXTENSION_NAME} PRIVATE
    LIGHTGBM_ARROW_H_
    ARROW_SCHEMA_ALREADY_DEFINED
    ARROW_ARRAY_ALREADY_DEFINED
)

# Include directories
target_include_directories(${EXTENSION_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${DUCKDB_INCLUDE_DIR}
    ${LIGHTGBM_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(${EXTENSION_NAME} PRIVATE
    ${DUCKDB_LIBRARY_PATH}
    ${LIGHTGBM_LIBRARY}
    OpenMP::OpenMP_CXX
    Threads::Threads
    m
)

# Add nlohmann/json
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(${EXTENSION_NAME} PRIVATE nlohmann_json::nlohmann_json)
elseif(nlohmann_json_FOUND)
    target_include_directories(${EXTENSION_NAME} PRIVATE ${nlohmann_json_INCLUDE_DIR})
endif()

# On Linux, ensure symbols are exported
if(UNIX AND NOT APPLE)
    target_link_options(${EXTENSION_NAME} PRIVATE -Wl,--export-dynamic)
endif()

# ============================================================================
# Compiler Options
# ============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${EXTENSION_NAME} PRIVATE -O3 -march=native)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${EXTENSION_NAME} PRIVATE -g -O0 -DDEBUG)
endif()

target_compile_options(${EXTENSION_NAME} PRIVATE -Wall -Wextra)

# Suppress specific warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${EXTENSION_NAME} PRIVATE
        -Wno-unused-parameter
        -Wno-unused-but-set-variable
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
# Print Summary
# ============================================================================
message(STATUS "")
message(STATUS "=========================================")
message(STATUS "DuckDB AI Extension Build Configuration:")
message(STATUS "  Extension Name: ${EXTENSION_NAME}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: 17")
message(STATUS "  DuckDB Headers: ${DUCKDB_INCLUDE_DIR}")
message(STATUS "  DuckDB Library: ${DUCKDB_LIBRARY_PATH}")
message(STATUS "  LightGBM: ${LIGHTGBM_LIBRARY}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "=========================================")
message(STATUS "")
