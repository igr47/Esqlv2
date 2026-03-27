#pragma once

#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
//#include "duckdb/common/optional.hpp"
#include "lightgbm_model.h"
#include "model_registry.hpp"
#include <optional>

namespace duckdb {

struct ModelMetadata;
struct ModelSchema;

// ============================================================================
// PREDICT Table Function (for SELECT ... FROM predict(...))
// ============================================================================

// Table function that returns predictions for a table
struct PredictTableFunctionData : public TableFunctionData {
    string model_name;
    string input_table;
    string where_clause;
    vector<string> output_columns;
    bool include_probabilities = false;
    bool include_confidence = false;
	std::optional<idx_t> limit;
    string output_table;

    // Model info
    ModelMetadata model_metadata;
	std::optional<ModelSchema> model_schema;
    string problem_type;
    size_t num_classes = 0;
    vector<string> class_labels;

    // State
    idx_t current_row = 0;
    bool completed = false;
    vector<Value> row_cache;
    vector<vector<Value>> prediction_cache;

    // Performance
    unique_ptr<esql::AdaptiveLightGBMModel> model;

	vector<string> input_column_names;
};

unique_ptr<FunctionData> PredictTableBind(ClientContext &context,
                                          TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types,
                                          vector<string> &names);

void PredictTableFunction(ClientContext &context,
                          TableFunctionInput &data_p,
                          DataChunk &output);

// ============================================================================
// PREDICT Scalar Functions (for SELECT ai_predict(...) FROM ...)
// ============================================================================

// Basic predict - returns prediction value
void PredictScalarFunction(DataChunk &args, ExpressionState &state, Vector &result);

// Predict class - returns class label
void PredictClassFunction(DataChunk &args, ExpressionState &state, Vector &result);

// Predict probability - returns probability for binary/classification
void PredictProbaFunction(DataChunk &args, ExpressionState &state, Vector &result);

// Predict with probabilities for multiclass - returns JSON with all probabilities
void PredictProbasFunction(DataChunk &args, ExpressionState &state, Vector &result);

// ============================================================================
// Parser Extension for PREDICT Statement
// ============================================================================

class PredictStatement : public ParserExtensionParseData {
public:
    string model_name;
    string input_table;
    string where_clause;
    vector<string> output_columns;
    bool include_probabilities = false;
    bool include_confidence = false;
	std::optional<idx_t> limit;
    string output_table;

    string ToString() const override {
        return "PREDICT USING " + model_name + " ON " + input_table;
    }

    unique_ptr<ParserExtensionParseData> Copy() const override {
        return make_uniq<PredictStatement>(*this);
    }
};

bool IsPredictStatement(const string &query);
unique_ptr<PredictStatement> ParsePredictStatement(const string &query);

} // namespace duckdb
