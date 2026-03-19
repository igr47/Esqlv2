#pragma once
#ifndef LIGHTGBM_ISOLATED_H
#define LIGHTGBM_ISOLATED_H

// Step 1: Include DuckDB normally (this is C++ code)
#include "duckdb.hpp"

// Step 2: Create a namespace for DuckDB's Arrow structures
namespace duckdb_arrows {
    // These are just forward declarations - we don't need the full definitions
    struct ArrowSchema;
    struct ArrowArray;
}

// Step 3: Temporarily rename Arrow symbols to avoid conflicts
// This must be done BEFORE including LightGBM
#define ArrowSchema LightGBM_ArrowSchema
#define ArrowArray LightGBM_ArrowArray

// Step 4: Include LightGBM - WITHOUT extern "C" because it contains C++ templates
// LightGBM's c_api.h already handles extern "C" internally for C functions
#include "LightGBM/c_api.h"

// Step 5: Undefine the macros
#undef ArrowSchema
#undef ArrowArray

// Step 6: Put LightGBM's Arrow structures in their own namespace
namespace lightgbm_arrow {
    using ArrowSchema = struct ::LightGBM_ArrowSchema;
    using ArrowArray = struct ::LightGBM_ArrowArray;
}

#endif // LIGHTGBM_ISOLATED_H
