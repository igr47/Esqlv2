#include "duckdb.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "include/esql/algorithm_registry.hpp"
#include "include/ai_extension_extension.hpp"
#include "include/esql/model_registry.hpp"
#include <filesystem>

namespace duckdb {

// Forward declarations for the table function implementations (defined in train_model_function.cpp)
void TrainModelFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output);
unique_ptr<FunctionData> TrainModelBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<string> &names);

void AiExtensionExtension::Load(ExtensionLoader &loader) {
    // ========================================================================
    // 1. Register table functions
    // ========================================================================
    // train_model
    loader.RegisterFunction(
        TableFunction("train_model",
            {
                LogicalType::VARCHAR,                     // model_name
                LogicalType::VARCHAR,                     // algorithm
                LogicalType::LIST(LogicalType::VARCHAR),  // features
                LogicalType::VARCHAR,                     // target_column
                LogicalType::VARCHAR,                     // target_type (optional)
                LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR), // parameters
                LogicalType::VARCHAR                      // source_table
            },
            TrainModelFunction, TrainModelBind)
    );

    // predict table function
    loader.RegisterFunction(
        TableFunction("predict",
            {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
            [](ClientContext &context, TableFunctionInput &data, DataChunk &output) {
                output.SetCardinality(1);
                output.SetValue(0, 0, Value(0.0));
            },
            [](ClientContext &context, TableFunctionBindInput &input,
               vector<LogicalType> &return_types, vector<string> &names) -> unique_ptr<FunctionData> {
                return_types.push_back(LogicalType::DOUBLE);
                names.push_back("prediction");
                return nullptr;
            })
    );

    // show_models table function
    loader.RegisterFunction(
        TableFunction("show_models", {},
            [](ClientContext &context, TableFunctionInput &data, DataChunk &output) {
                // TODO: implement
                output.SetCardinality(0);
            })
    );

    // ========================================================================
    // 2. Register scalar functions
    // ========================================================================
    loader.RegisterFunction(
        ScalarFunction("predict",
            {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
            LogicalType::DOUBLE,
            [](DataChunk &args, ExpressionState &state, Vector &result) {
                result.SetValue(0, Value(0.0));
            })
    );

    loader.RegisterFunction(
        ScalarFunction("predict_class",
            {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
            LogicalType::VARCHAR,
            [](DataChunk &args, ExpressionState &state, Vector &result) {
                result.SetValue(0, Value("0"));
            })
    );

    loader.RegisterFunction(
        ScalarFunction("predict_proba",
            {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
            LogicalType::DOUBLE,
            [](DataChunk &args, ExpressionState &state, Vector &result) {
                result.SetValue(0, Value(0.0));
            })
    );

    // ========================================================================
    // 3. Initialize algorithm registry (no database access)
    // ========================================================================
    AlgorithmRegistry::Instance();

    // ========================================================================
    // 4. Defer table creation and directory setup to first use.
    //    This is done inside TrainModelBind via a lazy initializer.
    // ========================================================================
}

std::string AiExtensionExtension::Name() {
    return "ai_extension";
}

std::string AiExtensionExtension::Version() const {
    return "0.1.0";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void ai_extension_init(duckdb::ExtensionLoader &loader) {
    duckdb::AiExtensionExtension extension;
    extension.Load(loader);
}

DUCKDB_EXTENSION_API const char *ai_extension_version() {
    return "0.1.0";
}

} // extern "C"
