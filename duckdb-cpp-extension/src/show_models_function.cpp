#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "esql/model_registry.hpp"
#include "esql/lightgbm_model.h"
#include <nlohmann/json.hpp>
#include <iomanip>
#include <sstream>

namespace duckdb {

// ============================================
// SHOW MODELS - List all models
// ============================================

struct ShowModelsBindData : public TableFunctionData {
    vector<ModelMetadata> models;
    idx_t current_idx = 0;
    bool completed = false;
};

unique_ptr<FunctionData> ShowModelsBind(ClientContext &context,
                                        TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types,
                                        vector<string> &names) {
    auto result = make_uniq<ShowModelsBindData>();

    // Fetch all models from registry
    result->models = ModelRegistry::ListModelsDetailed(context);

    // Define output columns
    return_types.push_back(LogicalType::VARCHAR);  // model_name
    return_types.push_back(LogicalType::VARCHAR);  // algorithm
    return_types.push_back(LogicalType::VARCHAR);  // problem_type
    return_types.push_back(LogicalType::VARCHAR);  // target_column
    return_types.push_back(LogicalType::UBIGINT);  // training_samples
    return_types.push_back(LogicalType::DOUBLE);   // accuracy
    return_types.push_back(LogicalType::DOUBLE);   // primary_metric (F1 for classification, R2 for regression)
    return_types.push_back(LogicalType::TIMESTAMP); // created_at
    return_types.push_back(LogicalType::VARCHAR);  // status (active/inactive)
    return_types.push_back(LogicalType::VARCHAR);  // feature_count

    names.push_back("model_name");
    names.push_back("algorithm");
    names.push_back("problem_type");
    names.push_back("target_column");
    names.push_back("training_samples");
    names.push_back("accuracy");
    names.push_back("primary_metric");
    names.push_back("created_at");
    names.push_back("status");
    names.push_back("feature_count");

    return std::move(result);
}

void ShowModelsFunction(ClientContext &context,
                        TableFunctionInput &data_p,
                        DataChunk &output) {
    auto &data = (ShowModelsBindData &)*data_p.bind_data;

    if (data.completed) {
        return;
    }

    if (data.current_idx >= data.models.size()) {
        data.completed = true;
        output.SetCardinality(0);
        return;
    }

    // Calculate how many rows to output (max 1024 per chunk)
    idx_t rows_to_output = std::min((idx_t)1024, data.models.size() - data.current_idx);
    output.SetCardinality(rows_to_output);

    for (idx_t i = 0; i < rows_to_output; i++) {
        const auto &model = data.models[data.current_idx + i];

        // Calculate primary metric
        double primary_metric = model.accuracy;
        if (model.problem_type == "binary_classification" || model.problem_type == "multiclass") {
            primary_metric = model.f1_score > 0 ? model.f1_score : model.accuracy;
        } else if (model.problem_type == "regression") {
            primary_metric = model.r2_score > 0 ? model.r2_score : model.accuracy;
        }

        output.SetValue(0, i, Value(model.name));
        output.SetValue(1, i, Value(model.algorithm));
        output.SetValue(2, i, Value(model.problem_type));
        output.SetValue(3, i, Value(model.target_column));
        output.SetValue(4, i, Value::UBIGINT(model.training_samples));
        output.SetValue(5, i, Value(model.accuracy));
        output.SetValue(6, i, Value(primary_metric));

        // Convert timestamp
        auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
            model.created_at.time_since_epoch()).count();
        output.SetValue(7, i, Value::TIMESTAMP(Timestamp::FromEpochMicroSeconds(micros)));

        output.SetValue(8, i, Value(model.is_active ? "active" : "inactive"));
        output.SetValue(9, i, Value(std::to_string(model.feature_names.size())));
    }

    data.current_idx += rows_to_output;
    if (data.current_idx >= data.models.size()) {
        data.completed = true;
    }
}

// ============================================
// SHOW MODEL - Show detailed model information
// ============================================

struct ShowModelBindData : public TableFunctionData {
    ModelMetadata metadata;
    std::optional<ModelSchema> schema;  // This is duckdb::ModelSchema
    bool loaded = false;
    bool completed = false;
    string model_name;
};

unique_ptr<FunctionData> ShowModelBind(ClientContext &context,
                                       TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
    auto result = make_uniq<ShowModelBindData>();

    result->model_name = input.inputs[0].GetValue<string>();

    // Fetch model metadata
    auto metadata_opt = ModelRegistry::GetModel(context, result->model_name);
    if (!metadata_opt.has_value()) {
        throw std::runtime_error("Model not found: " + result->model_name);
    }
    result->metadata = metadata_opt.value();

    // Load model schema for detailed info - this returns duckdb::ModelSchema
    result->schema = ModelRegistry::LoadModelSchema(context, result->model_name);

    // Define output columns
    return_types.push_back(LogicalType::VARCHAR);  // model_name
    return_types.push_back(LogicalType::VARCHAR);  // algorithm
    return_types.push_back(LogicalType::VARCHAR);  // problem_type
    return_types.push_back(LogicalType::VARCHAR);  // target_column
    return_types.push_back(LogicalType::UBIGINT);  // training_samples
    return_types.push_back(LogicalType::DOUBLE);   // accuracy
    return_types.push_back(LogicalType::DOUBLE);   // precision
    return_types.push_back(LogicalType::DOUBLE);   // recall
    return_types.push_back(LogicalType::DOUBLE);   // f1_score
    return_types.push_back(LogicalType::DOUBLE);   // auc_score (if classification)
    return_types.push_back(LogicalType::DOUBLE);   // r2_score (if regression)
    return_types.push_back(LogicalType::DOUBLE);   // rmse (if regression)
    return_types.push_back(LogicalType::DOUBLE);   // mae (if regression)
    return_types.push_back(LogicalType::TIMESTAMP); // created_at
    return_types.push_back(LogicalType::TIMESTAMP); // last_updated
    return_types.push_back(LogicalType::VARCHAR);  // features (JSON string)
    return_types.push_back(LogicalType::VARCHAR);  // parameters (JSON string)
    return_types.push_back(LogicalType::VARCHAR);  // metadata (JSON string)
    return_types.push_back(LogicalType::VARCHAR);  // model_path
    return_types.push_back(LogicalType::VARCHAR);  // status
    return_types.push_back(LogicalType::UBIGINT);  // prediction_count
    return_types.push_back(LogicalType::DOUBLE);   // drift_score

    names.push_back("model_name");
    names.push_back("algorithm");
    names.push_back("problem_type");
    names.push_back("target_column");
    names.push_back("training_samples");
    names.push_back("accuracy");
    names.push_back("precision");
    names.push_back("recall");
    names.push_back("f1_score");
    names.push_back("auc_score");
    names.push_back("r2_score");
    names.push_back("rmse");
    names.push_back("mae");
    names.push_back("created_at");
    names.push_back("last_updated");
    names.push_back("features");
    names.push_back("parameters");
    names.push_back("metadata");
    names.push_back("model_path");
    names.push_back("status");
    names.push_back("prediction_count");
    names.push_back("drift_score");

    result->loaded = true;
    return std::move(result);
}

// Helper function to format duckdb::FeatureDescriptor (not esql::FeatureDescriptor)
std::string FormatFeaturesAsJSON(const std::vector<FeatureDescriptor>& features) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& f : features) {
        nlohmann::json feature_json;
        feature_json["name"] = f.name;
        feature_json["db_column"] = f.db_column;
        feature_json["data_type"] = f.data_type;
        feature_json["transformation"] = f.transformation;
        feature_json["required"] = f.required;
        feature_json["is_categorical"] = f.is_categorical;
        if (f.is_categorical && !f.categories.empty()) {
            feature_json["categories"] = f.categories;
        }
        if (f.transformation != "direct") {
            feature_json["min_value"] = f.min_value;
            feature_json["max_value"] = f.max_value;
            feature_json["mean_value"] = f.mean_value;
            feature_json["std_value"] = f.std_value;
        }
        j.push_back(feature_json);
    }
    return j.dump(2);
}

std::string FormatParametersAsJSON(const std::unordered_map<std::string, std::string>& params) {
    nlohmann::json j;
    for (const auto& [key, value] : params) {
        j[key] = value;
    }
    return j.dump(2);
}

std::string FormatMetadataAsJSON(const std::unordered_map<std::string, std::string>& metadata) {
    nlohmann::json j;
    for (const auto& [key, value] : metadata) {
        j[key] = value;
    }
    return j.dump(2);
}

void ShowModelFunction(ClientContext &context,
                       TableFunctionInput &data_p,
                       DataChunk &output) {
    auto &data = (ShowModelBindData &)*data_p.bind_data;

    if (data.completed) {
        return;
    }

    output.SetCardinality(1);

    const auto &model = data.metadata;

    output.SetValue(0, 0, Value(model.name));
    output.SetValue(1, 0, Value(model.algorithm));
    output.SetValue(2, 0, Value(model.problem_type));
    output.SetValue(3, 0, Value(model.target_column));
    output.SetValue(4, 0, Value::UBIGINT(model.training_samples));
    output.SetValue(5, 0, Value(model.accuracy));
    output.SetValue(6, 0, Value(model.precision));
    output.SetValue(7, 0, Value(model.recall));
    output.SetValue(8, 0, Value(model.f1_score));
    output.SetValue(9, 0, Value(model.auc_score));
    output.SetValue(10, 0, Value(model.r2_score));
    output.SetValue(11, 0, Value(model.rmse));
    output.SetValue(12, 0, Value(model.mae));

    auto created_micros = std::chrono::duration_cast<std::chrono::microseconds>(
        model.created_at.time_since_epoch()).count();
    output.SetValue(13, 0, Value::TIMESTAMP(Timestamp::FromEpochMicroSeconds(created_micros)));

    // Last updated from schema if available
    if (data.schema.has_value()) {
        auto updated_micros = std::chrono::duration_cast<std::chrono::microseconds>(
            data.schema->last_updated.time_since_epoch()).count();
        output.SetValue(14, 0, Value::TIMESTAMP(Timestamp::FromEpochMicroSeconds(updated_micros)));

        // Format features as JSON - using duckdb::FeatureDescriptor
        output.SetValue(15, 0, Value(FormatFeaturesAsJSON(data.schema->features)));

        // Format metadata as JSON
        output.SetValue(17, 0, Value(FormatMetadataAsJSON(data.schema->metadata)));
    } else {
        output.SetValue(14, 0, Value(LogicalType::TIMESTAMP));
        output.SetValue(15, 0, Value("{}"));
        output.SetValue(17, 0, Value("{}"));
    }

    // Format parameters as JSON
    output.SetValue(16, 0, Value(FormatParametersAsJSON(model.parameters)));

    output.SetValue(18, 0, Value(model.model_path));
    output.SetValue(19, 0, Value(model.is_active ? "active" : "inactive"));
    output.SetValue(20, 0, Value::UBIGINT(model.prediction_count));
    output.SetValue(21, 0, Value(model.drift_score));

    data.completed = true;
}

} // namespace duckdb
