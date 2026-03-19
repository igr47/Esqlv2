#include "include/esql/model_registry.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/common/types/value.hpp"
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace duckdb {

// ============================================================================
// JSON conversion helpers (unchanged from original)
// ============================================================================
namespace {
    nlohmann::json FeatureToJson(const FeatureDescriptor &fd) {
        nlohmann::json j;
        j["name"] = fd.name;
        j["db_column"] = fd.db_column;
        j["data_type"] = fd.data_type;
        j["transformation"] = fd.transformation;
        j["default_value"] = fd.default_value;
        j["required"] = fd.required;
        j["is_categorical"] = fd.is_categorical;
        j["min_value"] = fd.min_value;
        j["max_value"] = fd.max_value;
        j["mean_value"] = fd.mean_value;
        j["std_value"] = fd.std_value;

        if (fd.is_categorical) {
            j["categories"] = fd.categories;
        }
        return j;
    }

    FeatureDescriptor FeatureFromJson(const nlohmann::json &j) {
        FeatureDescriptor fd;
        fd.name = j["name"];
        fd.db_column = j["db_column"];
        fd.data_type = j["data_type"];
        fd.transformation = j["transformation"];
        fd.default_value = j["default_value"];
        fd.required = j["required"];
        fd.is_categorical = j["is_categorical"];
        fd.min_value = j["min_value"];
        fd.max_value = j["max_value"];
        fd.mean_value = j["mean_value"];
        fd.std_value = j["std_value"];

        if (fd.is_categorical && j.contains("categories")) {
            fd.categories = j["categories"].get<std::vector<std::string>>();
        }
        return fd;
    }

    nlohmann::json SchemaToJson(const ModelSchema &schema) {
        nlohmann::json j;
        j["model_id"] = schema.model_id;
        j["description"] = schema.description;
        j["target_column"] = schema.target_column;
        j["algorithm"] = schema.algorithm;
        j["problem_type"] = schema.problem_type;
        j["created_at"] = std::chrono::system_clock::to_time_t(schema.created_at);
        j["last_updated"] = std::chrono::system_clock::to_time_t(schema.last_updated);
        j["training_samples"] = schema.training_samples;
        j["accuracy"] = schema.accuracy;
        j["drift_score"] = schema.drift_score;

        j["features"] = nlohmann::json::array();
        for (const auto &f : schema.features) {
            j["features"].push_back(FeatureToJson(f));
        }

        j["metadata"] = schema.metadata;

        j["stats"] = {
            {"total_predictions", schema.stats.total_predictions},
            {"failed_predictions", schema.stats.failed_predictions},
            {"avg_confidence", schema.stats.avg_confidence},
            {"avg_inference_time_us", schema.stats.avg_inference_time.count()}
        };

        return j;
    }

    ModelSchema SchemaFromJson(const nlohmann::json &j) {
        ModelSchema schema;
        schema.model_id = j["model_id"];
        schema.description = j["description"];
        schema.target_column = j["target_column"];
        schema.algorithm = j["algorithm"];
        schema.problem_type = j["problem_type"];
        schema.created_at = std::chrono::system_clock::from_time_t(j["created_at"]);
        schema.last_updated = std::chrono::system_clock::from_time_t(j["last_updated"]);
        schema.training_samples = j["training_samples"];
        schema.accuracy = j["accuracy"];
        schema.drift_score = j["drift_score"];

        for (const auto &f : j["features"]) {
            schema.features.push_back(FeatureFromJson(f));
        }

        if (j.contains("metadata")) {
            schema.metadata = j["metadata"].get<std::unordered_map<std::string, std::string>>();
        }

        if (j.contains("stats")) {
            schema.stats.total_predictions = j["stats"]["total_predictions"];
            schema.stats.failed_predictions = j["stats"]["failed_predictions"];
            schema.stats.avg_confidence = j["stats"]["avg_confidence"];
            schema.stats.avg_inference_time = std::chrono::microseconds(
                j["stats"]["avg_inference_time_us"]);
        }

        return schema;
    }
} // namespace

// ============================================================================
// Helper functions to convert C++ types to DuckDB Values
// ============================================================================
static Value ToValue(const std::vector<std::string> &vec) {
    vector<Value> list_values;
    list_values.reserve(vec.size());
    for (const auto &s : vec) {
        list_values.emplace_back(s);
    }
    return Value::LIST(list_values);
}

static Value ToValue(const std::unordered_map<std::string, std::string> &map) {
    vector<Value> keys, values;
    keys.reserve(map.size());
    values.reserve(map.size());
    for (const auto &kv : map) {
        keys.emplace_back(kv.first);
        values.emplace_back(kv.second);
    }
    //return Value::MAP(keys, values);
	return Value::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR, keys, values);
}

static Value ToValue(const std::chrono::system_clock::time_point &tp) {
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()).count();
    auto timestamp = Timestamp::FromEpochMicroSeconds(micros);
    return Value::TIMESTAMP(timestamp);
}

// ============================================================================
// ModelRegistry implementation (all methods now require ClientContext)
// ============================================================================

bool ModelRegistry::RegisterModel(ClientContext &context, const string &name,
                                   const ModelMetadata &metadata) {
    auto &db = DatabaseInstance::GetDatabase(context);
    Connection conn(db);

    auto name_val      = Value(name);
    auto algo_val      = Value(metadata.algorithm);
    auto problem_val   = Value(metadata.problem_type);
    auto features_val  = ToValue(metadata.feature_names);
    auto target_val    = Value(metadata.target_column);
    auto params_val    = ToValue(metadata.parameters);
    auto created_val   = ToValue(metadata.created_at);
    auto samples_val   = Value::UBIGINT(metadata.training_samples);
    auto acc_val       = Value(metadata.accuracy);
    auto path_val      = Value(metadata.model_path);
    auto active_val    = Value::BOOLEAN(metadata.is_active);

    // Prepare the statement
    auto prepared = conn.Prepare(
        "INSERT OR REPLACE INTO __model_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");

    if (prepared->HasError()) {
        return false;
    }

    // Execute with vector of values
    vector<Value> values = {
        name_val, algo_val, problem_val, features_val, target_val,
        params_val, created_val, samples_val, acc_val, path_val, active_val
    };

    auto result = prepared->Execute(values);
    return !result->HasError();
}


bool ModelRegistry::UpdateModel(ClientContext &context, const string &name,
                                 const ModelMetadata &metadata) {
    // Same as RegisterModel – INSERT OR REPLACE does the job
    return RegisterModel(context, name, metadata);
}

/*std::optional<ModelMetadata> ModelRegistry::GetModel(ClientContext &context, const string &name) {
    auto &db = DatabaseInstance::GetDatabase(context);
    Connection conn(db);

    auto prepared = conn.Prepare("SELECT * FROM __model_registry WHERE model_name = ?");
    if (prepared->HasError()) {
        return std::nullopt;
    }

    vector<Value> values = {Value(name)};
    auto result = prepared->Execute(values);

    if (result->HasError() || result->RowCount() == 0) {
        return std::nullopt;
    }

    // Exactly one row expected
    auto &row = *result;
    ModelMetadata metadata;
    metadata.name           = row.GetValue(0, 0).GetValue<string>();
    metadata.algorithm      = row.GetValue(1, 0).GetValue<string>();
    metadata.problem_type   = row.GetValue(2, 0).GetValue<string>();

    // feature_names (LIST)
    auto features_list = ListValue::GetChildren(row.GetValue(3, 0));
    for (auto &v : features_list) {
        metadata.feature_names.push_back(v.GetValue<string>());
    }

    metadata.target_column  = row.GetValue(4, 0).GetValue<string>();

    // parameters (MAP)
    auto map_val = row.GetValue(5, 0);
    auto &map_children = StructValue::GetChildren(map_val);
    auto &keys = ListValue::GetChildren(map_children[0]);
    auto &values_vec = ListValue::GetChildren(map_children[1]);
    for (size_t i = 0; i < keys.size(); ++i) {
        metadata.parameters[keys[i].GetValue<string>()] = values_vec[i].GetValue<string>();
    }

    // created_at (TIMESTAMP)
    auto timestamp = row.GetValue(6, 0).GetValue<timestamp_t>();
    metadata.created_at = std::chrono::system_clock::time_point(
        std::chrono::microseconds(Timestamp::GetEpochMicroSeconds(timestamp)));

    metadata.training_samples = row.GetValue(7, 0).GetValue<uint64_t>();
    metadata.accuracy         = row.GetValue(8, 0).GetValue<float>();
    metadata.model_path       = row.GetValue(9, 0).GetValue<string>();
    metadata.is_active        = row.GetValue(10, 0).GetValue<bool>();

    // Default values for fields not in the table
    metadata.precision = 0.0f;
    metadata.recall = 0.0f;
    metadata.f1_score = 0.0f;
    metadata.auc_score = 0.0f;
    metadata.r2_score = 0.0f;
    metadata.rmse = 0.0f;
    metadata.mae = 0.0f;
    metadata.prediction_count = 0;
    metadata.drift_score = 0.0f;

    return metadata;
}*/

std::optional<ModelMetadata> ModelRegistry::GetModel(ClientContext &context, const string &name) {
    auto &db = DatabaseInstance::GetDatabase(context);
    Connection conn(db);

    auto prepared = conn.Prepare("SELECT * FROM __model_registry WHERE model_name = ?");
    if (prepared->HasError()) {
        return std::nullopt;
    }

    vector<Value> values = {Value(name)};
    auto result = prepared->Execute(values);

    if (result->HasError()) {
        return std::nullopt;
    }

    // Cast to MaterializedQueryResult to access row-based methods
    auto &materialized = dynamic_cast<MaterializedQueryResult &>(*result);

    if (materialized.RowCount() == 0) {
        return std::nullopt;
    }

    ModelMetadata metadata;
    metadata.name           = materialized.GetValue(0, 0).GetValue<string>();
    metadata.algorithm      = materialized.GetValue(1, 0).GetValue<string>();
    metadata.problem_type   = materialized.GetValue(2, 0).GetValue<string>();

    // feature_names (LIST)
    auto features_list = ListValue::GetChildren(materialized.GetValue(3, 0));
    for (auto &v : features_list) {
        metadata.feature_names.push_back(v.GetValue<string>());
    }

    metadata.target_column  = materialized.GetValue(4, 0).GetValue<string>();

    // parameters (MAP)
    auto map_val = materialized.GetValue(5, 0);
    auto &map_children = StructValue::GetChildren(map_val);
    auto &keys = ListValue::GetChildren(map_children[0]);
    auto &values_vec = ListValue::GetChildren(map_children[1]);
    for (size_t i = 0; i < keys.size(); ++i) {
        metadata.parameters[keys[i].GetValue<string>()] = values_vec[i].GetValue<string>();
    }

    // created_at (TIMESTAMP)
    auto timestamp = materialized.GetValue(6, 0).GetValue<timestamp_t>();
    metadata.created_at = std::chrono::system_clock::time_point(
        std::chrono::microseconds(Timestamp::GetEpochMicroSeconds(timestamp)));

    metadata.training_samples = materialized.GetValue(7, 0).GetValue<uint64_t>();
    metadata.accuracy         = materialized.GetValue(8, 0).GetValue<float>();
    metadata.model_path       = materialized.GetValue(9, 0).GetValue<string>();
    metadata.is_active        = materialized.GetValue(10, 0).GetValue<bool>();

    // Default values for fields not in the table
    metadata.precision = 0.0f;
    metadata.recall = 0.0f;
    metadata.f1_score = 0.0f;
    metadata.auc_score = 0.0f;
    metadata.r2_score = 0.0f;
    metadata.rmse = 0.0f;
    metadata.mae = 0.0f;
    metadata.prediction_count = 0;
    metadata.drift_score = 0.0f;

    return metadata;
}

bool ModelRegistry::ModelExists(ClientContext &context, const string &name) {
    auto &db = DatabaseInstance::GetDatabase(context);
    Connection conn(db);

    auto prepared = conn.Prepare("SELECT COUNT(*) FROM __model_registry WHERE model_name = ?");
    if (prepared->HasError()) return false;

    vector<Value> values = {Value(name)};
    auto result = prepared->Execute(values);

    if (result->HasError()) return false;

    auto &materialized = dynamic_cast<MaterializedQueryResult &>(*result);
    return materialized.GetValue(0, 0).GetValue<uint64_t>() > 0;
}

bool ModelRegistry::DeleteModel(ClientContext &context, const string &name) {
    auto &db = DatabaseInstance::GetDatabase(context);
    Connection conn(db);

    auto prepared = conn.Prepare("DELETE FROM __model_registry WHERE model_name = ?");
    if (prepared->HasError()) return false;

    vector<Value> values = {Value(name)};
    auto result = prepared->Execute(values);

    if (result->HasError()) return false;

    // Also delete the model blob file and schema file
    auto &instance = GetInstance();
    std::unique_lock<std::shared_mutex> lock(instance.dir_mutex_);
    std::filesystem::remove(instance.GetModelPath(name));
    std::filesystem::remove(instance.GetSchemaPath(name));

    return true;
}

/*bool ModelRegistry::DeleteModel(ClientContext &context, const string &name) {
    //Connection conn(context);
    auto result = context.Query("DELETE FROM __model_registry WHERE model_name = ?", {Value(name)});
    if (result->HasError()) return false;

    // Also delete the model blob file and schema file
    auto &instance = GetInstance();
    std::unique_lock<std::shared_mutex> lock(instance.dir_mutex_);
    std::filesystem::remove(instance.GetModelPath(name));
    std::filesystem::remove(instance.GetSchemaPath(name));

    return true;
}*/

vector<string> ModelRegistry::ListModels(ClientContext &context) {
    auto &db = DatabaseInstance::GetDatabase(context);
    Connection conn(db);

    auto result = conn.Query("SELECT model_name FROM __model_registry ORDER BY model_name");

    if (result->HasError()) return {};

    vector<string> names;
    for (idx_t i = 0; i < result->RowCount(); ++i) {
        names.push_back(result->GetValue(0, i).GetValue<string>());
    }
    return names;
}

/*vector<string> ModelRegistry::ListModels(ClientContext &context) {
    //Connection conn(context);
    auto result = context.Query("SELECT model_name FROM __model_registry ORDER BY model_name");
    if (result->HasError()) return {};

    vector<string> names;
    for (idx_t i = 0; i < result->RowCount(); ++i) {
        names.push_back(result->GetValue(0, i).GetValue<string>());
    }
    return names;
}*/

vector<ModelMetadata> ModelRegistry::ListModelsDetailed(ClientContext &context) {
    auto &db = DatabaseInstance::GetDatabase(context);
    Connection conn(db);

    auto result = conn.Query("SELECT * FROM __model_registry ORDER BY model_name");

    if (result->HasError()) return {};

    auto &materialized = dynamic_cast<MaterializedQueryResult &>(*result);

    vector<ModelMetadata> models;
    for (idx_t i = 0; i < materialized.RowCount(); ++i) {
        ModelMetadata metadata;

        metadata.name           = materialized.GetValue(0, i).GetValue<string>();
        metadata.algorithm      = materialized.GetValue(1, i).GetValue<string>();
        metadata.problem_type   = materialized.GetValue(2, i).GetValue<string>();

        auto features_list = ListValue::GetChildren(materialized.GetValue(3, i));
        for (auto &v : features_list) {
            metadata.feature_names.push_back(v.GetValue<string>());
        }

        metadata.target_column  = materialized.GetValue(4, i).GetValue<string>();

 auto map_val = materialized.GetValue(5, i);
        auto &map_children = StructValue::GetChildren(map_val);
        auto &keys = ListValue::GetChildren(map_children[0]);
        auto &values_vec = ListValue::GetChildren(map_children[1]);
        for (size_t j = 0; j < keys.size(); ++j) {
            metadata.parameters[keys[j].GetValue<string>()] = values_vec[j].GetValue<string>();
        }

        auto timestamp = materialized.GetValue(6, i).GetValue<timestamp_t>();
        metadata.created_at = std::chrono::system_clock::time_point(
            std::chrono::microseconds(Timestamp::GetEpochMicroSeconds(timestamp)));

        metadata.training_samples = materialized.GetValue(7, i).GetValue<uint64_t>();
        metadata.accuracy         = materialized.GetValue(8, i).GetValue<float>();
        metadata.model_path       = materialized.GetValue(9, i).GetValue<string>();
        metadata.is_active        = materialized.GetValue(10, i).GetValue<bool>();

        // Default values
        metadata.precision = 0.0f;
        metadata.recall = 0.0f;
        metadata.f1_score = 0.0f;
        metadata.auc_score = 0.0f;
        metadata.r2_score = 0.0f;
        metadata.rmse = 0.0f;
        metadata.mae = 0.0f;
        metadata.prediction_count = 0;
        metadata.drift_score = 0.0f;

        models.push_back(metadata);
    }
    return models;
}

/*vector<ModelMetadata> ModelRegistry::ListModelsDetailed(ClientContext &context) {
    //Connection conn(context);
    auto result = context.Query("SELECT * FROM __model_registry ORDER BY model_name");
    if (result->HasError()) return {};

    vector<ModelMetadata> models;
    for (idx_t i = 0; i < result->RowCount(); ++i) {
        auto &row = *result;
        ModelMetadata metadata;

        metadata.name           = row.GetValue(0, i).GetValue<string>();
        metadata.algorithm      = row.GetValue(1, i).GetValue<string>();
        metadata.problem_type   = row.GetValue(2, i).GetValue<string>();

        auto features_list = ListValue::GetChildren(row.GetValue(3, i));
        for (auto &v : features_list) {
            metadata.feature_names.push_back(v.GetValue<string>());
        }

        metadata.target_column  = row.GetValue(4, i).GetValue<string>();

        auto map_val = row.GetValue(5, i);
        auto &map_children = StructValue::GetChildren(map_val);
        auto &keys = ListValue::GetChildren(map_children[0]);
        auto &values = ListValue::GetChildren(map_children[1]);
        for (size_t j = 0; j < keys.size(); ++j) {
            metadata.parameters[keys[j].GetValue<string>()] = values[j].GetValue<string>();
        }

        auto timestamp = row.GetValue(6, i).GetValue<timestamp_t>();
        metadata.created_at = std::chrono::system_clock::time_point(
            std::chrono::microseconds(Timestamp::GetEpochMicroSeconds(timestamp)));

        metadata.training_samples = row.GetValue(7, i).GetValue<uint64_t>();
        metadata.accuracy         = row.GetValue(8, i).GetValue<float>();
        metadata.model_path       = row.GetValue(9, i).GetValue<string>();
        metadata.is_active        = row.GetValue(10, i).GetValue<bool>();

        // Remaining fields default
        metadata.precision = 0.0f;
        metadata.recall = 0.0f;
        metadata.f1_score = 0.0f;
        metadata.auc_score = 0.0f;
        metadata.r2_score = 0.0f;
        metadata.rmse = 0.0f;
        metadata.mae = 0.0f;
        metadata.prediction_count = 0;
        metadata.drift_score = 0.0f;

        models.push_back(metadata);
    }
    return models;
}*/

bool ModelRegistry::StoreModelBlob(ClientContext &context, const string &name, const string &model_blob) {
    (void)context; // unused, but required for signature consistency
    auto &instance = GetInstance();

    std::unique_lock<std::shared_mutex> lock(instance.dir_mutex_);
    std::filesystem::create_directories(instance.models_directory_);

    string path = instance.GetModelPath(name);
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    file.write(model_blob.data(), model_blob.size());
    return file.good();
}

std::string ModelRegistry::LoadModelBlob(ClientContext &context, const string &name) {
    (void)context;
    auto &instance = GetInstance();
    std::shared_lock<std::shared_mutex> lock(instance.dir_mutex_);

    string path = instance.GetModelPath(name);
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return "";

    return string((std::istreambuf_iterator<char>(file)),
                  std::istreambuf_iterator<char>());
}

bool ModelRegistry::SaveModelSchema(ClientContext &context, const string &name, const ModelSchema &schema) {
    (void)context;
    auto &instance = GetInstance();
    std::unique_lock<std::shared_mutex> lock(instance.dir_mutex_);

    string path = instance.GetSchemaPath(name);
    std::ofstream file(path);
    if (!file.is_open()) return false;

    nlohmann::json j = SchemaToJson(schema);
    file << j.dump(2);
    return true;
}

std::optional<ModelSchema> ModelRegistry::LoadModelSchema(ClientContext &context, const string &name) {
    (void)context;
    auto &instance = GetInstance();
    std::shared_lock<std::shared_mutex> lock(instance.dir_mutex_);

    string path = instance.GetSchemaPath(name);
    std::ifstream file(path);
    if (!file.is_open()) return std::nullopt;

    nlohmann::json j;
    file >> j;
    return SchemaFromJson(j);
}

void ModelRegistry::SetModelsDirectory(const string &dir) {
    auto &instance = GetInstance();
    std::unique_lock<std::shared_mutex> lock(instance.dir_mutex_);
    instance.models_directory_ = dir;
}

void ModelRegistry::UpdateModelMetrics(ClientContext &context, const string &name,
                                       const unordered_map<string, float> &metrics) {
    // This method would need to update the relevant columns in the registry table.
    // Since the table currently does not store all metrics, this is left as a placeholder.
    // You could extend the table and implement UPDATE statements here.
    (void)context; (void)name; (void)metrics;
    // Example: UPDATE __model_registry SET accuracy = ?, precision = ? WHERE model_name = ?
}

string ModelRegistry::GetModelPath(const string &name) const {
    return models_directory_ + "/" + name + ".lgbm";
}

string ModelRegistry::GetSchemaPath(const string &name) const {
    return models_directory_ + "/" + name + ".schema.json";
}

} // namespace duckdb
