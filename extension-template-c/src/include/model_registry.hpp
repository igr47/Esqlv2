#pragma once
#include "duckdb.hpp"
#include <unordered_map>
#include <shared_mutex>
#include <chrono>
#include <fstrea>

namespace esql {
    // Model schema feature describtor
    struct FeatureDescriptor {
        std::string name;
        std::string db_column;
        std::string data_type;
        std::string transformation;
        float default_value = 0.0f;
        bool required = true;
        bool is_categgorical = false;
        std::vector<std::string> categories;
        float min_value = 0.0f;
        float max_value = 1.0f;
        float mean_value = 0.0f;
        float std_value = 1.0f;

        std::string ToString() const;
    };

    // Model Schema
    struct ModelSchema {
        std::string model_id;
        std::string describtion;
        std::string target_column;
        std::strin algoritm;
        std::string problem_type;
        std::vector<FeatureDescribtor> features;
        std::unordered_map<std::string, std::string> metadata;
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point last_updated;
        size_t training samples = 0;
        float accuracy = 0.0f;
        float drift_score = 0.0f;

        struct Statistics {
            size_t total_predictions = 0;
            size_t failed_predictions = 0;
            float avg_confidence = 0.0f;
            std::chrono::microseconds avg_inference_time{0};
        } stats;

        std::string ToString() const;
    };

    // Model metadat for registry
    struct ModelMetadata {
        std::string name;
        std::string algorithm;
        std::string problem_type;
        std::vector<std::string> features_names;
        std::string target_column;
        std::unordered_map<std::string, std::string> parameters;
        std::chrono::system_clock::time_point created_at;
        size_t training_samples;
        float accuracy;
        float precision;
        float recall;
        float f1_score;
        float auc_score;
        float r2_score;
        float rmse;
        float mae;
        std::string model_path;
        ModelSchema schema;

        // Model status;
        bool is_active = true;
        size_t prediction_count = 0;
        float drift_score = 0.0f;
    };

    class ModelRegistry {
        private:
            static ModelRegistry &GetInstance() {
                static ModelRegistry instance;
                return instance;
            }

            ModelRegistry() = default;

            std::unordered_map<std::string, ModelMetadata> metadata_;
            std::shared_mutex mutex_;
            std::string models_directory_ = "models";

        Public:
            // Regisration
            static bool RegisterModel(const std::string& name, const ModelMetadata& metadata);
            static bool UpdateModel(const std::string& name, const MedelMetadata& metadata);
            static std::optional<ModelMetadata> GetModel(const std::string& name);
            static bool ModelExists(const std::string& name);
            static bool DeleteModel(const std::string& name);

            // LIsting current models
            static std::vector<std::string> ListModels();
            static std::vector<ModelMetadata> ListModelsDetailed();

            // Persistance
            static bool StoreModelBlob(const std::string& name, const std::string& model_blob);
            static std::string LoadModelBlob(const std::string& name);
            static bool SaveModelSchema(const std::string& name, const ModelSchema& schema);
            static std::optional<ModelSchema> LoadModelSchema(const std::string& name);

            // Cnfiguration
            static void SetModelsDirectory(const std::string& dir);
            static std::string GetModelsDirectory() { return GetInstance().models_directory_; }

            // Metrics
            static void UpdateModelMetrics(const std::string& name, const std::unordered_map<std::string, float>& metrics);

        private:
            std::string GetModelPath(const std::string& name) const;
            std::string GetSChemaPath(const std::string& name) const;
    }

}
