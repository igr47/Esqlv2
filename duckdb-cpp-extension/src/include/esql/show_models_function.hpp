#pragma once

#include "duckdb.hpp"

namespace duckdb {

// SHOW MODELS functions
unique_ptr<FunctionData> ShowModelsBind(ClientContext &context,
                                        TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types,
                                        vector<string> &names);

void ShowModelsFunction(ClientContext &context,
                        TableFunctionInput &data_p,
                        DataChunk &output);

// SHOW MODEL functions (detailed view)
unique_ptr<FunctionData> ShowModelBind(ClientContext &context,
                                       TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names);

void ShowModelFunction(ClientContext &context,
                       TableFunctionInput &data_p,
                       DataChunk &output);

} // namespace duckdb
