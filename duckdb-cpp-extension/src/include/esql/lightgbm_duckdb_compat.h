#pragma once
#ifndef LIGHTGBM_DUCKDB_COMPAT_H
#define LIGHTGBM_DUCKDB_COMPAT_H

// Force define before ANY includes
#ifndef LGB_NO_ARROW
#define LGB_NO_ARROW 1
#endif

// Include DuckDB first (which defines Arrow structures)
#include "duckdb.hpp"

// Now include LightGBM which should respect LGB_NO_ARROW
#ifdef __cplusplus
extern "C" {
#endif

// We need to undefine any potential conflicts before including LightGBM
#ifdef ArrowSchema
#undef ArrowSchema
#endif

#ifdef ArrowArray
#undef ArrowArray
#endif

// Include LightGBM with Arrow disabled
#include "LightGBM/c_api.h"

#ifdef __cplusplus
}
#endif

#endif // LIGHTGBM_DUCKDB_COMPAT_H
