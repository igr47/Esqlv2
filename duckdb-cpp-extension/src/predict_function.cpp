#include "include/esql/predict_function.hpp"
#include "include/esql/lightgbm_model.h"
#include "include/esql/model_registry.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"

#include <mutex>
#include <regex>
#include <optional>
#include <chrono>
#include <nlohmann/json.hpp>

namespace duckdb {

// ============================================================================
// Helper Functions
// ============================================================================

static string ToUpper(const string &s) {
    string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

static float ValueToFloat(const Value &val) {
    if (val.IsNull()) return 0.0f;

    switch (val.type().id()) {
        case LogicalTypeId::INTEGER:
        case LogicalTypeId::BIGINT:
        case LogicalTypeId::HUGEINT:
            return static_cast<float>(val.GetValue<int64_t>());
        case LogicalTypeId::DOUBLE:
        case LogicalTypeId::FLOAT:
        case LogicalTypeId::DECIMAL:
            return static_cast<float>(val.GetValue<double>());
        case LogicalTypeId::BOOLEAN:
            return val.GetValue<bool>() ? 1.0f : 0.0f;
        case LogicalTypeId::VARCHAR:
            try {
                return std::stof(val.ToString());
            } catch (...) {
                size_t hash = std::hash<string>{}(val.ToString());
                return static_cast<float>(hash % 1000) / 1000.0f;
            }
        default:
            return 0.0f;
    }
}

static unordered_map<string, Value> RowToMap(DataChunk &chunk, idx_t row) {
    unordered_map<string, Value> row_map;
    /*for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
        row_map[chunk.GetName(i)] = chunk.GetValue(i, row);
    }*/
    return row_map;
}

static string FormatProbabilityOutput(const vector<float>& probabilities,
                                       const vector<string>& class_labels) {
    nlohmann::json j;
    if (class_labels.empty()) {
        for (size_t i = 0; i < probabilities.size(); i++) {
            j[std::to_string(i)] = probabilities[i];
        }
    } else {
        for (size_t i = 0; i < probabilities.size() && i < class_labels.size(); i++) {
            j[class_labels[i]] = probabilities[i];
        }
    }
    return j.dump(2);
}

// ============================================================================
// Model Inference Cache (per session)
// ============================================================================

struct ModelCache {
    struct Entry {
        unique_ptr<esql::AdaptiveLightGBMModel> model;
        ModelMetadata metadata;
		std::optional<ModelSchema> schema;
        string problem_type;
        size_t num_classes;
        vector<string> class_labels;
		std::chrono::system_clock::time_point loaded_at;
    };

    unordered_map<string, Entry> models;
	std::mutex mutex;

    Entry* GetOrLoad(ClientContext &context, const string &model_name) {
        lock_guard<std::mutex> lock(mutex);

        auto it = models.find(model_name);
        if (it != models.end()) {
            return &it->second;
        }

        // Load model
        auto metadata_opt = ModelRegistry::GetModel(context, model_name);
        if (!metadata_opt.has_value()) {
            throw std::runtime_error("Model not found: " + model_name);
        }

        Entry entry;
        entry.metadata = metadata_opt.value();
        entry.schema = ModelRegistry::LoadModelSchema(context, model_name);

        if (!entry.schema.has_value()) {
            throw std::runtime_error("Failed to load model schema for: " + model_name);
        }

        entry.problem_type = entry.schema->problem_type;

        // Extract class labels if multiclass
        if (entry.problem_type == "multiclass") {
            auto it = entry.schema->metadata.find("num_classes");
            if (it != entry.schema->metadata.end()) {
                entry.num_classes = std::stoi(it->second);
            }

            auto labels_it = entry.schema->metadata.find("class_labels");
            if (labels_it != entry.schema->metadata.end()) {
                // Parse class labels from JSON string
                try {
                    nlohmann::json j = nlohmann::json::parse(labels_it->second);
                    entry.class_labels = j.get<vector<string>>();
                } catch (...) {}
            }

            // Generate default labels if none
            if (entry.class_labels.empty() && entry.num_classes > 0) {
                for (size_t i = 0; i < entry.num_classes; i++) {
                    entry.class_labels.push_back("class_" + std::to_string(i));
                }
            }
        }

		// Create and load model - use make_uniq with proper conversion
        // Need to convert duckdb::ModelSchema to esql::ModelSchema
        esql::ModelSchema esql_schema;
        esql_schema.model_id = entry.schema->model_id;
        esql_schema.description = entry.schema->description;
        esql_schema.target_column = entry.schema->target_column;
        esql_schema.algorithm = entry.schema->algorithm;
        esql_schema.problem_type = entry.schema->problem_type;
        esql_schema.created_at = entry.schema->created_at;
        esql_schema.last_updated = entry.schema->last_updated;
        esql_schema.training_samples = entry.schema->training_samples;
        esql_schema.accuracy = entry.schema->accuracy;
        esql_schema.drift_score = entry.schema->drift_score;
        esql_schema.metadata = entry.schema->metadata;

        // Convert features
        for (const auto& f : entry.schema->features) {
            esql::FeatureDescriptor ef;
            ef.name = f.name;
            ef.db_column = f.db_column;
            ef.data_type = f.data_type;
            ef.transformation = f.transformation;
            ef.default_value = f.default_value;
            ef.required = f.required;
            ef.is_categorical = f.is_categorical;
            ef.categories = f.categories;
            ef.min_value = f.min_value;
            ef.max_value = f.max_value;
            ef.mean_value = f.mean_value;
            ef.std_value = f.std_value;
            esql_schema.features.push_back(ef);
        }

        esql_schema.stats.total_predictions = entry.schema->stats.total_predictions;
        esql_schema.stats.failed_predictions = entry.schema->stats.failed_predictions;
        esql_schema.stats.avg_confidence = entry.schema->stats.avg_confidence;
        esql_schema.stats.avg_inference_time = entry.schema->stats.avg_inference_time;

        // Create and load model
		entry.model = make_uniq<esql::AdaptiveLightGBMModel>(esql_schema);
        //entry.model = make_uniq<esql::AdaptiveLightGBMModel>(entry.schema.value());
        if (!entry.model->Load(entry.metadata.model_path)) {
            throw std::runtime_error("Failed to load model file: " + entry.metadata.model_path);
        }

        entry.loaded_at = std::chrono::system_clock::now();

        models[model_name] = std::move(entry);
        return &models[model_name];
    }

    void Clear() {
        lock_guard<std::mutex> lock(mutex);
        models.clear();
    }
};

static ModelCache& GetModelCache() {
    static ModelCache cache;
    return cache;
}

// ============================================================================
// Table Row Cache for Table‑Name Mode
// ============================================================================

struct TableRowsCache {
    struct CachedTable {
        vector<vector<Value>> rows;
        vector<string> column_names;
        size_t row_count;
    };

    unordered_map<string, CachedTable> tables;
    std::mutex mutex;

    const CachedTable& GetTable(ClientContext &context, const string &table_name) {
        lock_guard<std::mutex> lock(mutex);
        auto it = tables.find(table_name);
        if (it != tables.end()) {
            return it->second;
        }

        // Fetch entire table
        Connection conn(DatabaseInstance::GetDatabase(context));
        auto result = conn.Query("SELECT * FROM " + table_name);
        if (result->HasError()) {
            throw std::runtime_error("Failed to fetch table " + table_name + ": " + result->GetError());
        }

        auto& materialized = dynamic_cast<MaterializedQueryResult&>(*result);
        CachedTable cached;
        cached.row_count = materialized.RowCount();
        cached.column_names = materialized.names;

        for (idx_t row = 0; row < cached.row_count; ++row) {
            vector<Value> row_values;
            for (idx_t col = 0; col < materialized.ColumnCount(); ++col) {
                row_values.push_back(materialized.GetValue(col, row));
            }
            cached.rows.push_back(std::move(row_values));
        }

        tables[table_name] = std::move(cached);
        return tables[table_name];
    }

    void Clear() {
        lock_guard<std::mutex> lock(mutex);
        tables.clear();
    }
};

static TableRowsCache& GetTableRowsCache() {
    static TableRowsCache cache;
    return cache;
}

// ============================================================================
// Prediction Helper Functions
// ============================================================================

struct PredictionResult {
    float prediction;
    vector<float> probabilities;
    vector<string> class_labels;
    string problem_type;

    Value ToValue() const {
        if (problem_type == "binary_classification") {
            // Return the class (0 or 1)
            return Value::INTEGER(prediction > 0.5f ? 1 : 0);
        } else if (problem_type == "multiclass") {
            // Return the predicted class label
            size_t class_idx = static_cast<size_t>(std::round(prediction));
            if (class_idx < class_labels.size()) {
                return Value(class_labels[class_idx]);
            }
            return Value::INTEGER(static_cast<int64_t>(class_idx));
        } else {
            // Regression - return the numeric value
            return Value::DOUBLE(static_cast<double>(prediction));
        }
    }

    Value ToProbabilityValue() const {
        if (problem_type == "binary_classification") {
            return Value::DOUBLE(static_cast<double>(prediction));
        } else if (problem_type == "multiclass" && !probabilities.empty()) {
            size_t class_idx = static_cast<size_t>(std::round(prediction));
            if (class_idx < probabilities.size()) {
                return Value::DOUBLE(static_cast<double>(probabilities[class_idx]));
            }
        }
        return Value::DOUBLE(0.0);
    }

    Value ToProbabilitiesValue() const {
        if (problem_type == "binary_classification") {
            nlohmann::json j;
            j["0"] = 1.0 - prediction;
            j["1"] = prediction;
            return Value(j.dump());
        } else if (problem_type == "multiclass" && !probabilities.empty()) {
            return Value(FormatProbabilityOutput(probabilities, class_labels));
        }
        return Value("{}");
    }
};

static PredictionResult PredictRow(esql::AdaptiveLightGBMModel& model,
                                    const unordered_map<string, Value>& row,
                                    const ModelCache::Entry& cache) {
    PredictionResult result;
    result.problem_type = cache.problem_type;
    result.class_labels = cache.class_labels;

	auto& schema = model.GetSchema();

	vector<float> features;
    features.reserve(schema.features.size());

    for (const auto& feature : schema.features) {
        auto it = row.find(feature.db_column);
        if (it != row.end()) {
            features.push_back(feature.transform(it->second));
        } else {
            features.push_back(feature.default_value);
        }
    }

    //auto features = cache.schema->ExtractFeatures(row);
	esql::Tensor input(features, {features.size()});
    auto output = model.Predict(input);

    if (output.data.empty()) {
        throw std::runtime_error("Prediction failed - no output");
    }

    result.prediction = output.data[0];

    // For multiclass, we need probabilities
    if (cache.problem_type == "multiclass" && output.data.size() > 1) {
        result.probabilities.assign(output.data.begin(), output.data.end());
    }

    return result;
}

// ============================================================================
// Predict Table Function - Implementation
// ============================================================================

unique_ptr<FunctionData> PredictTableBind(ClientContext &context,
                                          TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types,
                                          vector<string> &names) {
    auto result = make_uniq<PredictTableFunctionData>();

    // Parse arguments
    result->model_name = input.inputs[0].GetValue<string>();
    result->input_table = input.inputs[1].GetValue<string>();

    // Parse optional arguments
    if (input.inputs.size() > 2 && !input.inputs[2].IsNull()) {
        result->where_clause = input.inputs[2].GetValue<string>();
    }

    // Load model
    auto& cache = GetModelCache();
    auto* cached = cache.GetOrLoad(context, result->model_name);
    result->model_metadata = cached->metadata;
    result->model_schema = cached->schema;
    result->problem_type = cached->problem_type;
    result->num_classes = cached->num_classes;
    result->class_labels = cached->class_labels;

	// Create model copy - need to convert schema properly
    esql::ModelSchema esql_schema;
    if (cached->schema.has_value()) {
        esql_schema.model_id = cached->schema->model_id;
        esql_schema.description = cached->schema->description;
        esql_schema.target_column = cached->schema->target_column;
        esql_schema.algorithm = cached->schema->algorithm;
        esql_schema.problem_type = cached->schema->problem_type;
        esql_schema.created_at = cached->schema->created_at;
        esql_schema.last_updated = cached->schema->last_updated;
        esql_schema.training_samples = cached->schema->training_samples;
        esql_schema.accuracy = cached->schema->accuracy;
        esql_schema.drift_score = cached->schema->drift_score;
        esql_schema.metadata = cached->schema->metadata;

        for (const auto& f : cached->schema->features) {
            esql::FeatureDescriptor ef;
            ef.name = f.name;
            ef.db_column = f.db_column;
            ef.data_type = f.data_type;
            ef.transformation = f.transformation;
            ef.default_value = f.default_value;
            ef.required = f.required;
            ef.is_categorical = f.is_categorical;
            ef.categories = f.categories;
            ef.min_value = f.min_value;
            ef.max_value = f.max_value;
            ef.mean_value = f.mean_value;
            ef.std_value = f.std_value;
            esql_schema.features.push_back(ef);
        }

        esql_schema.stats.total_predictions = cached->schema->stats.total_predictions;
        esql_schema.stats.failed_predictions = cached->schema->stats.failed_predictions;
        esql_schema.stats.avg_confidence = cached->schema->stats.avg_confidence;
        esql_schema.stats.avg_inference_time = cached->schema->stats.avg_inference_time;
    }

    result->model = make_uniq<esql::AdaptiveLightGBMModel>(esql_schema);
    // Note: We need to load the model file, not copy the cached model
    if (!result->model->Load(cached->metadata.model_path)) {
        throw std::runtime_error("Failed to load model file: " + cached->metadata.model_path);
    }
    //result->model = make_uniq<esql::AdaptiveLightGBMModel>(*cached->model);

    // Define output columns based on problem type and requested output
    // First, add all input columns
    auto input_columns = ModelRegistry::GetAllColumns(context, result->input_table);
    for (const auto& col : input_columns) {
        return_types.push_back(LogicalType::VARCHAR);
        names.push_back(col);
    }

    // Add prediction columns
    if (result->problem_type == "binary_classification") {
        return_types.push_back(LogicalType::VARCHAR);  // predicted_class
        names.push_back("predicted_class");

        if (result->include_probabilities) {
            return_types.push_back(LogicalType::DOUBLE);  // probability
            names.push_back("probability");
        }
        if (result->include_confidence) {
            return_types.push_back(LogicalType::DOUBLE);  // confidence
            names.push_back("confidence");
        }
    } else if (result->problem_type == "multiclass") {
        return_types.push_back(LogicalType::VARCHAR);  // predicted_class
        names.push_back("predicted_class");

        if (result->include_probabilities) {
            return_types.push_back(LogicalType::VARCHAR);  // probabilities (JSON)
            names.push_back("probabilities");
        }
    } else {
        // Regression
        return_types.push_back(LogicalType::DOUBLE);  // prediction
        names.push_back("prediction");

        if (result->include_confidence) {
            return_types.push_back(LogicalType::DOUBLE);  // confidence interval
            names.push_back("confidence");
        }
    }

    return std::move(result);
}

void PredictTableFunction(ClientContext &context,
                          TableFunctionInput &data_p,
                          DataChunk &output) {
    auto &data = (PredictTableFunctionData &)*data_p.bind_data;

    if (data.completed) {
        return;
    }

    // Build query to fetch data
    string query = "SELECT * FROM " + data.input_table;
    if (!data.where_clause.empty()) {
        query += " WHERE " + data.where_clause;
    }
    if (data.limit.has_value()) {
        query += " LIMIT " + std::to_string(data.limit.value());
    }

    //Connection conn(*DatabaseInstance::GetDatabase(context).get());
    auto &db = DatabaseInstance::GetDatabase(context);
    Connection conn(db);
    auto result = conn.Query(query);

    if (result->HasError()) {
        throw std::runtime_error("Failed to query input table: " + result->GetError());
    }

    auto& materialized = dynamic_cast<MaterializedQueryResult&>(*result);
    idx_t total_rows = materialized.RowCount();
    idx_t start_idx = data.current_row;
    idx_t rows_to_output = std::min((idx_t)1024, total_rows - start_idx);

    if (rows_to_output == 0) {
        data.completed = true;
        output.SetCardinality(0);
        return;
    }

    // Process rows in batches
    output.SetCardinality(rows_to_output);
    idx_t output_col_offset = materialized.ColumnCount();

    for (idx_t i = 0; i < rows_to_output; i++) {
        idx_t row_idx = start_idx + i;

        // Copy input columns
        for (idx_t col = 0; col < materialized.ColumnCount(); col++) {
            output.SetValue(col, i, materialized.GetValue(col, row_idx));
        }

        // Build row map for prediction
        unordered_map<string, Value> row_map;
        for (idx_t col = 0; col < materialized.ColumnCount(); col++) {
            row_map[materialized.names[col]] = materialized.GetValue(col, row_idx);
        }

        // Make prediction
        //auto pred_result = PredictRow(*data.model, row_map,//ModelCache::GetOrLoad(context, data.model_name));
		auto& cache = GetModelCache();
        auto* cached = cache.GetOrLoad(context, data.model_name);
        auto pred_result = PredictRow(*data.model, row_map, *cached);

        // Fill prediction columns
        if (data.problem_type == "binary_classification") {
            output.SetValue(output_col_offset, i,
                pred_result.prediction > 0.5f ? Value("1") : Value("0"));

            if (data.include_probabilities) {
                output.SetValue(output_col_offset + 1, i,
                    Value::DOUBLE(pred_result.prediction));
            }
            if (data.include_confidence) {
                float confidence = std::max(pred_result.prediction, 1.0f - pred_result.prediction);
                output.SetValue(output_col_offset + (data.include_probabilities ? 2 : 1), i,
                    Value::DOUBLE(confidence));
            }
        } else if (data.problem_type == "multiclass") {
            size_t class_idx = static_cast<size_t>(std::round(pred_result.prediction));
            string class_label = class_idx < data.class_labels.size() ?
                                 data.class_labels[class_idx] : std::to_string(class_idx);
            output.SetValue(output_col_offset, i, Value(class_label));

            if (data.include_probabilities && !pred_result.probabilities.empty()) {
                output.SetValue(output_col_offset + 1, i,
                    Value(FormatProbabilityOutput(pred_result.probabilities, data.class_labels)));
            }
        } else {
            // Regression
output.SetValue(output_col_offset, i, Value::DOUBLE(pred_result.prediction));
        }
    }

    data.current_row += rows_to_output;
    if (data.current_row >= total_rows) {
        data.completed = true;
    }
}

// ============================================================================
// Predict Scalar Functions
// ============================================================================

// Global model cache for scalar functions (per query)
static mutex scalar_model_mutex;
static unordered_map<string, pair<unique_ptr<esql::AdaptiveLightGBMModel>, ModelMetadata>> scalar_model_cache;

static esql::AdaptiveLightGBMModel* GetScalarModel(ClientContext& context, const string& model_name) {
    lock_guard<std::mutex> lock(scalar_model_mutex);

    auto it = scalar_model_cache.find(model_name);
    if (it != scalar_model_cache.end()) {
        return it->second.first.get();
    }

    auto metadata_opt = ModelRegistry::GetModel(context, model_name);
    if (!metadata_opt.has_value()) {
        throw std::runtime_error("Model not found: " + model_name);
    }

    auto schema_opt = ModelRegistry::LoadModelSchema(context, model_name);
    if (!schema_opt.has_value()) {
        throw std::runtime_error("Failed to load model schema for: " + model_name);
    }

    esql::ModelSchema esql_schema;
    esql_schema.model_id = schema_opt->model_id;
    esql_schema.description = schema_opt->description;
    esql_schema.target_column = schema_opt->target_column;
    esql_schema.algorithm = schema_opt->algorithm;
    esql_schema.problem_type = schema_opt->problem_type;
    esql_schema.created_at = schema_opt->created_at;
    esql_schema.last_updated = schema_opt->last_updated;
    esql_schema.training_samples = schema_opt->training_samples;
    esql_schema.accuracy = schema_opt->accuracy;
    esql_schema.drift_score = schema_opt->drift_score;
    esql_schema.metadata = schema_opt->metadata;

    for (const auto& f : schema_opt->features) {
        esql::FeatureDescriptor ef;
        ef.name = f.name;
        ef.db_column = f.db_column;
        ef.data_type = f.data_type;
        ef.transformation = f.transformation;
        ef.default_value = f.default_value;
        ef.required = f.required;
        ef.is_categorical = f.is_categorical;
        ef.categories = f.categories;
        ef.min_value = f.min_value;
        ef.max_value = f.max_value;
        ef.mean_value = f.mean_value;
        ef.std_value = f.std_value;
        esql_schema.features.push_back(ef);
    }

    esql_schema.stats.total_predictions = schema_opt->stats.total_predictions;
    esql_schema.stats.failed_predictions = schema_opt->stats.failed_predictions;
    esql_schema.stats.avg_confidence = schema_opt->stats.avg_confidence;
    esql_schema.stats.avg_inference_time = schema_opt->stats.avg_inference_time;

    auto model = make_uniq<esql::AdaptiveLightGBMModel>(esql_schema);
    //auto model = make_unique<esql::AdaptiveLightGBMModel>(schema_opt.value());
    if (!model->Load(metadata_opt->model_path)) {
        throw std::runtime_error("Failed to load model: " + model_name);
    }

    auto& cached = scalar_model_cache[model_name];
    cached.first = std::move(model);
    cached.second = metadata_opt.value();

    return cached.first.get();
}

// Main dispatcher for ai_predict
void PredictScalarFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    if (args.ColumnCount() < 1) {
        throw std::runtime_error("ai_predict requires at least model_name");
    }

    auto& model_name_vector = args.data[0];
    auto& context = state.GetContext();

    // Determine mode: if second argument exists and is a constant string, treat as table name
    bool use_table_name_mode = false;
    string table_name;
    if (args.ColumnCount() >= 2) {
        // Check if second argument is a constant string literal
        auto& second_arg = args.data[1];
        if (second_arg.GetType().id() == LogicalTypeId::VARCHAR) {
            // We need to check if it's a constant (i.e., all values are the same and it's a literal)
            // In DuckDB, we can't easily get that here. We'll assume that if the second argument is a
            // VARCHAR and the number of arguments is exactly 2, then it's a table name.
            // This is a heuristic; the user must not pass a column name as a string literal.
            if (args.ColumnCount() == 2) {
                use_table_name_mode = true;
                table_name = second_arg.GetValue(0).GetValue<string>();
            }
        }
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
    idx_t row_count = args.size();

    if (use_table_name_mode) {
        // Mode 2: predict from specified table
        string model_name = model_name_vector.GetValue(0).GetValue<string>();
        auto* model = GetScalarModel(context, model_name);
        auto& schema = model->GetSchema();

        // Fetch cached table rows
        auto& table_cache = GetTableRowsCache();
        const auto& cached_table = table_cache.GetTable(context, table_name);

        // For each row, use the same row index from the cached table
        for (idx_t i = 0; i < row_count; ++i) {
            if (i >= cached_table.row_count) {
                // No corresponding row in target table
                result.SetValue(i, Value());
                continue;
            }

            // Build row map from cached table row
            unordered_map<string, Value> row_map;
            for (size_t col = 0; col < cached_table.column_names.size(); ++col) {
                row_map[cached_table.column_names[col]] = cached_table.rows[i][col];
            }

            // Extract features
            vector<float> features;
            for (const auto& fd : schema.features) {
                auto it = row_map.find(fd.db_column);
                if (it != row_map.end()) {
                    features.push_back(fd.transform(it->second));
                } else if (fd.required) {
                    features.push_back(fd.default_value);
                } else {
                    features.push_back(fd.default_value);
                }
            }

            esql::Tensor input(features, {features.size()});
            auto output = model->Predict(input);
            if (output.data.empty()) {
                result.SetValue(i, Value());
            } else {
                result.SetValue(i, Value::DOUBLE(output.data[0]));
            }
        }
    } else {
        // Mode 1: predict from provided features
        // We expect model_name + N features, where N equals the number of features in the model.
        string model_name = model_name_vector.GetValue(0).GetValue<string>();
        auto* model = GetScalarModel(context, model_name);
        auto& schema = model->GetSchema();

        idx_t feature_count = args.ColumnCount() - 1;
        if (feature_count != schema.features.size()) {
            throw std::runtime_error("Expected " + std::to_string(schema.features.size()) +
                                     " features, got " + std::to_string(feature_count));
        }

        for (idx_t i = 0; i < row_count; ++i) {
            unordered_map<string, Value> row_map;
            for (size_t f = 0; f < schema.features.size(); ++f) {
                row_map[schema.features[f].db_column] = args.data[f + 1].GetValue(i);
            }

            auto features = schema.ExtractFeatures(row_map);
            esql::Tensor input(features, {features.size()});
            auto output = model->Predict(input);
            if (output.data.empty()) {
                result.SetValue(i, Value());
            } else {
                result.SetValue(i, Value::DOUBLE(output.data[0]));
            }
        }
    }
}

// ai_predict_class
void PredictClassFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    if (args.ColumnCount() < 1) {
        throw std::runtime_error("ai_predict_class requires at least model_name");
    }

    auto& model_name_vector = args.data[0];
    auto& context = state.GetContext();

    bool use_table_name_mode = false;
    string table_name;
    if (args.ColumnCount() >= 2) {
        auto& second_arg = args.data[1];
        if (second_arg.GetType().id() == LogicalTypeId::VARCHAR && args.ColumnCount() == 2) {
            use_table_name_mode = true;
            table_name = second_arg.GetValue(0).GetValue<string>();
        }
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
    idx_t row_count = args.size();

    if (use_table_name_mode) {
        string model_name = model_name_vector.GetValue(0).GetValue<string>();
        auto* model = GetScalarModel(context, model_name);
        auto& schema = model->GetSchema();

        auto& table_cache = GetTableRowsCache();
        const auto& cached_table = table_cache.GetTable(context, table_name);

        for (idx_t i = 0; i < row_count; ++i) {
            if (i >= cached_table.row_count) {
                result.SetValue(i, Value());
                continue;
            }

            unordered_map<string, Value> row_map;
            for (size_t col = 0; col < cached_table.column_names.size(); ++col) {
                row_map[cached_table.column_names[col]] = cached_table.rows[i][col];
            }

            auto features = schema.ExtractFeatures(row_map);
            esql::Tensor input(features, {features.size()});
            auto output = model->Predict(input);

            if (output.data.empty()) {
                result.SetValue(i, Value());
            } else if (schema.problem_type == "binary_classification") {
                result.SetValue(i, Value(output.data[0] > 0.5f ? "1" : "0"));
            } else if (schema.problem_type == "multiclass") {
                size_t class_idx = static_cast<size_t>(std::round(output.data[0]));
                auto it = schema.metadata.find("class_labels");
                if (it != schema.metadata.end()) {
                    try {
                        nlohmann::json j = nlohmann::json::parse(it->second);
                        if (class_idx < j.size()) {
                            result.SetValue(i, Value(j[class_idx].get<string>()));
                            continue;
                        }
                    } catch (...) {}
                }
                result.SetValue(i, Value::INTEGER(static_cast<int64_t>(class_idx)));
            } else {
                result.SetValue(i, Value::DOUBLE(output.data[0]));
            }
        }
    } else {
        // Mode: features provided
        string model_name = model_name_vector.GetValue(0).GetValue<string>();
        auto* model = GetScalarModel(context, model_name);
        auto& schema = model->GetSchema();

        idx_t feature_count = args.ColumnCount() - 1;
        if (feature_count != schema.features.size()) {
            throw std::runtime_error("Expected " + std::to_string(schema.features.size()) +
                                     " features, got " + std::to_string(feature_count));
        }

        for (idx_t i = 0; i < row_count; ++i) {
            unordered_map<string, Value> row_map;
            for (size_t f = 0; f < schema.features.size(); ++f) {
                row_map[schema.features[f].db_column] = args.data[f + 1].GetValue(i);
            }

            auto features = schema.ExtractFeatures(row_map);
            esql::Tensor input(features, {features.size()});
            auto output = model->Predict(input);

            if (output.data.empty()) {
                result.SetValue(i, Value());
            } else if (schema.problem_type == "binary_classification") {
                result.SetValue(i, Value(output.data[0] > 0.5f ? "1" : "0"));
            } else if (schema.problem_type == "multiclass") {
                size_t class_idx = static_cast<size_t>(std::round(output.data[0]));
                auto it = schema.metadata.find("class_labels");
                if (it != schema.metadata.end()) {
                    try {
                        nlohmann::json j = nlohmann::json::parse(it->second);
                        if (class_idx < j.size()) {
                            result.SetValue(i, Value(j[class_idx].get<string>()));
                            continue;
                        }
                    } catch (...) {}
                }
                result.SetValue(i, Value::INTEGER(static_cast<int64_t>(class_idx)));
            } else {
                result.SetValue(i, Value::DOUBLE(output.data[0]));
            }
        }
    }
}

// ai_predict_proba
void PredictProbaFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    if (args.ColumnCount() < 1) {
        throw std::runtime_error("ai_predict_proba requires at least model_name");
    }

    auto& model_name_vector = args.data[0];
    auto& context = state.GetContext();

    bool use_table_name_mode = false;
    string table_name;
    if (args.ColumnCount() >= 2) {
        auto& second_arg = args.data[1];
        if (second_arg.GetType().id() == LogicalTypeId::VARCHAR && args.ColumnCount() == 2) {
            use_table_name_mode = true;
            table_name = second_arg.GetValue(0).GetValue<string>();
        }
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
    idx_t row_count = args.size();

    if (use_table_name_mode) {
        string model_name = model_name_vector.GetValue(0).GetValue<string>();
        auto* model = GetScalarModel(context, model_name);
        auto& schema = model->GetSchema();

        auto& table_cache = GetTableRowsCache();
        const auto& cached_table = table_cache.GetTable(context, table_name);

        for (idx_t i = 0; i < row_count; ++i) {
            if (i >= cached_table.row_count) {
                result.SetValue(i, Value());
                continue;
            }

            unordered_map<string, Value> row_map;
            for (size_t col = 0; col < cached_table.column_names.size(); ++col) {
                row_map[cached_table.column_names[col]] = cached_table.rows[i][col];
            }

            auto features = schema.ExtractFeatures(row_map);
            esql::Tensor input(features, {features.size()});
            auto output = model->Predict(input);

            if (output.data.empty()) {
                result.SetValue(i, Value());
            } else if (schema.problem_type == "binary_classification") {
                result.SetValue(i, Value::DOUBLE(output.data[0]));
            } else if (schema.problem_type == "multiclass" && output.data.size() > 1) {
                size_t class_idx = static_cast<size_t>(std::round(output.data[0]));
                if (class_idx < output.data.size()) {
                    result.SetValue(i, Value::DOUBLE(output.data[class_idx]));
                } else {
                    result.SetValue(i, Value::DOUBLE(0.0));
                }
            } else {
                result.SetValue(i, Value::DOUBLE(output.data[0]));
            }
        }
    } else {
        string model_name = model_name_vector.GetValue(0).GetValue<string>();
        auto* model = GetScalarModel(context, model_name);
        auto& schema = model->GetSchema();

        idx_t feature_count = args.ColumnCount() - 1;
        if (feature_count != schema.features.size()) {
            throw std::runtime_error("Expected " + std::to_string(schema.features.size()) +
                                     " features, got " + std::to_string(feature_count));
        }

        for (idx_t i = 0; i < row_count; ++i) {
            unordered_map<string, Value> row_map;
            for (size_t f = 0; f < schema.features.size(); ++f) {
                row_map[schema.features[f].db_column] = args.data[f + 1].GetValue(i);
            }

            auto features = schema.ExtractFeatures(row_map);
            esql::Tensor input(features, {features.size()});
            auto output = model->Predict(input);

            if (output.data.empty()) {
                result.SetValue(i, Value());
            } else if (schema.problem_type == "binary_classification") {
                result.SetValue(i, Value::DOUBLE(output.data[0]));
            } else if (schema.problem_type == "multiclass" && output.data.size() > 1) {
                size_t class_idx = static_cast<size_t>(std::round(output.data[0]));
                if (class_idx < output.data.size()) {
                    result.SetValue(i, Value::DOUBLE(output.data[class_idx]));
                } else {
                    result.SetValue(i, Value::DOUBLE(0.0));
                }
            } else {
                result.SetValue(i, Value::DOUBLE(output.data[0]));
            }
        }
    }
}

// ai_predict_probas
void PredictProbasFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    if (args.ColumnCount() < 1) {
        throw std::runtime_error("ai_predict_probas requires at least model_name");
    }

    auto& model_name_vector = args.data[0];
    auto& context = state.GetContext();

    bool use_table_name_mode = false;
    string table_name;
    if (args.ColumnCount() >= 2) {
        auto& second_arg = args.data[1];
        if (second_arg.GetType().id() == LogicalTypeId::VARCHAR && args.ColumnCount() == 2) {
            use_table_name_mode = true;
            table_name = second_arg.GetValue(0).GetValue<string>();
        }
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
    idx_t row_count = args.size();

    if (use_table_name_mode) {
        string model_name = model_name_vector.GetValue(0).GetValue<string>();
        auto* model = GetScalarModel(context, model_name);
        auto& schema = model->GetSchema();

        auto& table_cache = GetTableRowsCache();
        const auto& cached_table = table_cache.GetTable(context, table_name);

        for (idx_t i = 0; i < row_count; ++i) {
            if (i >= cached_table.row_count) {
                result.SetValue(i, Value());
                continue;
            }

            unordered_map<string, Value> row_map;
            for (size_t col = 0; col < cached_table.column_names.size(); ++col) {
                row_map[cached_table.column_names[col]] = cached_table.rows[i][col];
            }

            auto features = schema.ExtractFeatures(row_map);
            esql::Tensor input(features, {features.size()});
            auto output = model->Predict(input);

            if (output.data.empty()) {
                result.SetValue(i, Value());
            } else {
                vector<string> class_labels;
                auto it = schema.metadata.find("class_labels");
                if (it != schema.metadata.end()) {
                    try {
                        nlohmann::json j = nlohmann::json::parse(it->second);
                        class_labels = j.get<vector<string>>();
                    } catch (...) {}
                }

                string json_output;
                if (schema.problem_type == "binary_classification") {
                    nlohmann::json j;
                    j["0"] = 1.0 - output.data[0];
                    j["1"] = output.data[0];
                    json_output = j.dump();
                } else if (output.data.size() > 1) {
                    json_output = FormatProbabilityOutput(
                        vector<float>(output.data.begin(), output.data.end()),
                        class_labels);
                } else {
                    json_output = "{}";
                }

                result.SetValue(i, Value(json_output));
            }
        }
    } else {
        string model_name = model_name_vector.GetValue(0).GetValue<string>();
        auto* model = GetScalarModel(context, model_name);
        auto& schema = model->GetSchema();

        idx_t feature_count = args.ColumnCount() - 1;
        if (feature_count != schema.features.size()) {
            throw std::runtime_error("Expected " + std::to_string(schema.features.size()) +
                                     " features, got " + std::to_string(feature_count));
        }

        for (idx_t i = 0; i < row_count; ++i) {
            unordered_map<string, Value> row_map;
            for (size_t f = 0; f < schema.features.size(); ++f) {
                row_map[schema.features[f].db_column] = args.data[f + 1].GetValue(i);
            }

            auto features = schema.ExtractFeatures(row_map);
            esql::Tensor input(features, {features.size()});
            auto output = model->Predict(input);

            if (output.data.empty()) {
                result.SetValue(i, Value());
            } else {
                vector<string> class_labels;
                auto it = schema.metadata.find("class_labels");
                if (it != schema.metadata.end()) {
                    try {
                        nlohmann::json j = nlohmann::json::parse(it->second);
                        class_labels = j.get<vector<string>>();
                    } catch (...) {}
                }

                string json_output;
                if (schema.problem_type == "binary_classification") {
                    nlohmann::json j;
                    j["0"] = 1.0 - output.data[0];
                    j["1"] = output.data[0];
                    json_output = j.dump();
                } else if (output.data.size() > 1) {
                    json_output = FormatProbabilityOutput(
                        vector<float>(output.data.begin(), output.data.end()),
                        class_labels);
                } else {
                    json_output = "{}";
                }

                result.SetValue(i, Value(json_output));
            }
        }
    }
}

// ============================================================================
// Parser Extension for PREDICT Statement
// ============================================================================

bool IsPredictStatement(const string &query) {
    std::regex pattern(R"(^\s*(PREDICT|INFER)\s+USING\s+[a-zA-Z_][a-zA-Z0-9_]*\s+ON\s+[a-zA-Z_][a-zA-Z0-9_]*)",
                       std::regex::icase | std::regex::optimize);
    return std::regex_search(query, pattern);
}

unique_ptr<PredictStatement> ParsePredictStatement(const string &query) {
    auto stmt = make_uniq<PredictStatement>();
    string s = query;

    // Regex patterns
    std::regex type_regex(R"(^\s*(PREDICT|INFER)\s+USING\s+([a-zA-Z_][a-zA-Z0-9_]*))",
                          std::regex::icase);
    std::regex table_regex(R"(ON\s+([a-zA-Z_][a-zA-Z0-9_]*))", std::regex::icase);
    std::regex where_regex(R"(WHERE\s+(.+?)(?=\s+OUTPUT|\s+WITH|\s+INTO|\s+LIMIT|\s*$))",
                           std::regex::icase);
    std::regex output_regex(R"(OUTPUT\s*\(\s*([^)]+)\s*\))", std::regex::icase);
    std::regex with_regex(R"(WITH\s+(PROBABILITIES|CONFIDENCE))", std::regex::icase);
    std::regex into_regex(R"(INTO\s+([a-zA-Z_][a-zA-Z0-9_]*))", std::regex::icase);
    std::regex limit_regex(R"(LIMIT\s+(\d+))", std::regex::icase);

    std::smatch match;

    // Parse model name
    if (std::regex_search(s, match, type_regex)) {
        stmt->model_name = match[2];
    } else {
        throw std::runtime_error("Invalid PREDICT statement: model name required");
    }

    // Parse input table
    if (std::regex_search(s, match, table_regex)) {
        stmt->input_table = match[1];
    } else {
        throw std::runtime_error("Invalid PREDICT statement: ON clause required");
    }

    // Parse WHERE clause
    if (std::regex_search(s, match, where_regex)) {
        stmt->where_clause = match[1];
    }

    // Parse OUTPUT columns
    if (std::regex_search(s, match, output_regex)) {
        string cols = match[1];
        std::regex col_regex(R"([a-zA-Z_][a-zA-Z0-9_]*(?:\s+AS\s+[a-zA-Z_][a-zA-Z0-9_]*)?)");
        auto begin = std::sregex_iterator(cols.begin(), cols.end(), col_regex);
        auto end = std::sregex_iterator();

        for (auto it = begin; it != end; ++it) {
            string col = it->str();
            size_t as_pos = ToUpper(col).find(" AS ");
            if (as_pos != string::npos) {
                string alias = col.substr(as_pos + 4);
                stmt->output_columns.push_back(alias);
            } else {
                stmt->output_columns.push_back(col);
            }
        }
    }

    // Parse WITH clause
    if (std::regex_search(s, match, with_regex)) {
        string opt = ToUpper(match[1]);
        if (opt == "PROBABILITIES") {
            stmt->include_probabilities = true;
        } else if (opt == "CONFIDENCE") {
            stmt->include_confidence = true;
        }
    }

    // Parse INTO clause
    if (std::regex_search(s, match, into_regex)) {
        stmt->output_table = match[1];
    }

    // Parse LIMIT
    if (std::regex_search(s, match, limit_regex)) {
        try {
            stmt->limit = std::stoul(match[1].str());
        } catch (...) {}
    }

    return stmt;
}

} // namespace duckdb
