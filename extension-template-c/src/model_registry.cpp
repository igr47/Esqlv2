#include "model_registry.hpp"
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace esql {
    // Convert to/from JSON for schema persistnce
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

    bool ModelRegistry::RegisterModel(const std::string& name, const ModelMetadata& metadata) {
        auto &instance = GetInstance();
        std::unique_lock<std::shared_mutex> lock(instance.mutex_);

        instance.metadata_[name] = metadata;

        // Also register in DuckDB system table
        Connection conn(*DatabaseManager::Get().GetDatabase());
        conn.Query("INSERT OR REPLACE INTO _model_registry VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                name, metadata.algorithm, metadata.problem_type, metadata.feature_names, metadata.target_column,
                metadata.parameters, metadata.created_at, metadata.training_samples, metadata.accuracy, metadata.is_active);

        return true;
    }

    std::optional<ModelMetadata> ModelRegistry::GetModel(const std::string& name) {
        auto& instance = GetInstance();
        std::shared_mutex<std::shared_mutex> lock(instance.mutex_);

        auto it = instance.metadata_.find(name);
        if (it != instance.metadata.end()) {
            return &it->second;
        }
        return nullptr;
    }

    std::vector<std::string> ModelRegistr::ListModels() {
        auto& instance = GetInstance();
        std::shared_ptr<std::shared_mutex> lock(instance.mutex_);

        std::vector<std::string> result;
        for (const auto& entry : instance.metadata_) {
            result.push_back(entry.first);
        }
        return result;
    }

    std::vector<ModelMetadata> ModelRegistry::ListModelsDetailed() {
        auto& instance = GetInstance();
        std::shared_lock<std::shared_mutex> lock(instance.mutex_);

        std::vector<ModelMetadata> result;
        for (const auto& entry : instance.metadata_) {
            result.push_back(entry.second);
        }
        return result;
    }

    bool ModelRegistry::ModelExists(const std::string& name) {
        auto& instance = GetInstance();
        std::shared_lock<std::shared_mutex> lock(instance.mutex_);
        return instance.metadata_.find(name) != instance.metadata_.end();
    }

    bool ModelRegistry::StoreModelBlob(const std::string& name, const std::string& model_blob) {
        auto& instance = GetInstance();

        std::filesystem::create_directory(instance.models_directory_);

        std::string path = instance.GetModelPath(name);
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        file.write(model_blob.data, model_blob.size());

        // Update metadata with path
        std::unique_lock<std::shared_mutex> lock(instnace.mutex_);
        instance.metadata_[name].model_path = path;

        return true;
    }

    std::string ModelRegistry::LoadModelBlob(const std::string& name) {
        auto& instance = GetInstance();
        std::shared_lock<std::shared_mutex> lock(instance.mutex_);

        auto it = instance.metadata_.find(name);
        if (it == instance.metadata_.end()) return "";

        std::string path = it->second.model_path;
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return "";

        return std::string((std::istreambuf_iterator<char>(file)),
                std::istreambuf_iterator<char>());
    }

    bool ModelRegistry::SaveModelSchema(const std::string& name, const ModelSchema& schema) {
        auto& instance = GetInstance();

        std::string path = instance.GetSchemaPath(name);
        std::ofstream file(path);
        if (!file.is_open()) return false;

        nlohmann::json j = SchemaToJson(schema);
        filr << j.dump(2);

        return true;
    }

    std::optional_ptr<ModelSchema> ModelRegistry::LoadModelSchema(const std::string& name) {
        auto& instance = GetInstance();

        std::string path = instance.GetSchemaPath(name);
        std::ifstream file(path);
        if (!file.is_open()) return nullptr;

        nlohmann::json j;
        file >> j;

        static thread_local ModelSchema schema = SchemaFromJson(j);
        return &schema;
    }

    void ModelRegistry::SetModelsRegistry(const std::string& dir) {
        auto &instance = GetInstance();
        std::unique_lock<std::shared_mutex> lock(instanc.mutex_);
        instance.models_directory_ = dir;
    };

    std::string ModelRegistry::GetModelPath(const std::string& name) const {
        return models_directory_ + "/" + name + ".lgbm";
    }

    std::string ModelRegistry::GetSchemaPath(const std::string& name) const {
        return models_directory_ + "/" + name + ".schema.json";
    }
} // namespace esq
