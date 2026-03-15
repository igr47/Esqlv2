#pragma once
#ifndef ADAPTIVE_LIGHTGBM_H
#define ADAPTIVE_LIGHTGBM_H

#include "duckdb.hpp"
#include "LightGBM/c_api.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <nlohmann/json.hpp>

namespace esql {

// Feature descriptor for mapping database columns to model features
struct FeatureDescriptor {
    std::string name;
    std::string db_column;          // Original database column name
    std::string data_type;          // int, float, string, date, bool
    std::string transformation;      // normalize, log, onehot, etc.
    float default_value = 0.0f;      // Default if column missing
    bool required = true;             // Required feature
    bool is_categorical = false;      // Categorical feature
    std::vector<std::string> categories;  // For categorical features
    float min_value = 0.0f;           // For normalization
    float max_value = 1.0f;           // For normalization
    float mean_value = 0.0f;           // For imputation
    float std_value = 1.0f;            // For standardization

    FeatureDescriptor() = default;

    // Convert database value to model feature
    float transform(const duckdb::Value& value) const;

    nlohmann::json ToJson() const;
    static FeatureDescriptor FromJson(const nlohmann::json& j);

    std::string ToString() const;
};

// Complete model schema for adaptive handling
struct ModelSchema {
    std::string model_id;
    std::string description;
    std::string target_column;
    std::string algorithm;
    std::string problem_type;  // binary_classification, multiclass, regression
    std::vector<FeatureDescriptor> features;
    std::unordered_map<std::string, std::string> metadata;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_updated;
    size_t training_samples = 0;
    float accuracy = 0.0f;
    float drift_score = 0.0f;

    struct Statistics {
        size_t total_predictions = 0;
        size_t failed_predictions = 0;
        float avg_confidence = 0.0f;
        std::chrono::microseconds avg_inference_time{0};
    } stats;

    nlohmann::json ToJson() const;
    static ModelSchema FromJson(const nlohmann::json& j);

    float GetMetadataFloat(const std::string& key, float default_value) const;
    bool MatchesRow(const std::unordered_map<std::string, duckdb::Value>& row) const;
    std::vector<float> ExtractFeatures(const std::unordered_map<std::string, duckdb::Value>& row) const;
    std::vector<std::string> GetMissingFeatures(const std::unordered_map<std::string, duckdb::Value>& row) const;

    std::string ToString() const;
};

// Training data structure (matching your existing format)
struct TrainingData {
    std::vector<std::vector<float>> features;
    std::vector<float> labels;
    std::vector<std::string> feature_names;
    std::string label_name;
    size_t total_samples = 0;
    size_t valid_samples = 0;

    // Split datasets
    struct SplitData {
        std::vector<std::vector<float>> features;
        std::vector<float> labels;
        size_t size = 0;

        bool Empty() const { return features.empty(); }
        void Clear() {
            features.clear();
            labels.clear();
            size = 0;
        }
    };

    SplitData train;
    SplitData test;
    SplitData validation;

    void Split(float train_ratio = 0.8f, float test_ratio = 0.1f,
               float validation_ratio = 0.1f, int seed = 42);

    std::string ToString() const;
};

// Lightweight Tensor type for fast inference
struct Tensor {
    std::vector<float> data;
    std::vector<size_t> shape;

    Tensor() = default;
    Tensor(const std::vector<float>& d, const std::vector<size_t>& s) : data(d), shape(s) {}
    Tensor(std::vector<float>&& d, std::vector<size_t>&& s) : data(std::move(d)), shape(std::move(s)) {}

    size_t TotalSize() const {
        size_t size = 1;
        for (auto& s : shape) size *= s;
        return size;
    }

    float* Ptr() { return data.data(); }
    const float* Ptr() const { return data.data(); }
};

// Model metadata for registry
struct ModelMetadata {
    std::string name;
    std::string algorithm;
    std::string problem_type;
    size_t input_size = 0;
    size_t output_size = 0;
    float accuracy = 0.0f;
    float precision = 0.0f;
    float recall = 0.0f;
    float f1_score = 0.0f;
    float auc_score = 0.0f;
    float r2_score = 0.0f;
    float rmse = 0.0f;
    float mae = 0.0f;
    float within_10_percent = 0.0f;
    float within_1_std = 0.0f;
    float coverage_95 = 0.0f;
    size_t model_size = 0;
    std::chrono::milliseconds avg_inference_time{0};
    std::unordered_map<std::string, std::string> parameters;

    ModelMetadata() = default;
};

// Main LightGBM model class
class AdaptiveLightGBMModel {
public:
    AdaptiveLightGBMModel();
    explicit AdaptiveLightGBMModel(const ModelSchema& schema);
    ~AdaptiveLightGBMModel();

    // Core methods
    bool Load(const std::string& path);
    bool Save(const std::string& path);

    Tensor Predict(const Tensor& input);
    Tensor PredictRow(const std::unordered_map<std::string, duckdb::Value>& row);
    std::vector<Tensor> PredictBatch(const std::vector<Tensor>& inputs);

    ModelMetadata GetMetadata() const;

    // Training methods
    bool Train(const std::vector<std::vector<float>>& features,
               const std::vector<float>& labels,
               const std::unordered_map<std::string, std::string>& params = {});

    bool TrainWithSplits(
        const TrainingData::SplitData& train_data,
        const TrainingData::SplitData& validation_data,
        const std::unordered_map<std::string, std::string>& params = {},
        int early_stopping_rounds = 10);

    // Schema management
    const ModelSchema& GetSchema() const { return schema_; }
    void UpdateSchema(const ModelSchema& new_schema);
    void UpdateFeature(const FeatureDescriptor& feature);

    // Drift detection
    bool NeedsRetraining() const;
    float GetDriftScore() const { return schema_.drift_score; }
    void ResetDriftDetector();

    // Performance
    void SetBatchSize(size_t batch_size);
    void Warmup(size_t iterations = 10);
    size_t GetMemoryUsage() const;
    void ReleaseUnusedMemory();
    size_t GetPredictionCount() const { return prediction_count_; }
    float GetAvgInferenceTimeMs() const;

    // Validation
    bool CanHandleRow(const std::unordered_map<std::string, duckdb::Value>& row) const;

    // Metric extraction
    void ExtractNativeMetrics(BoosterHandle booster);
    void CalculateDetailedValidationMetrics(const TrainingData::SplitData& validation_data);

private:
    BoosterHandle booster_ = nullptr;
    ModelSchema schema_;
    mutable std::mutex model_mutex_;
    std::atomic<bool> is_loaded_{false};
    std::atomic<size_t> prediction_count_{0};

    // Performance optimization
    std::vector<float> input_buffer_;
    std::vector<double> output_buffer_;
    size_t batch_size_ = 1;

    // Drift detection
    struct DriftDetector {
        std::vector<std::vector<float>> recent_features;
        std::vector<float> recent_predictions;
        std::chrono::system_clock::time_point last_drift_check;
        float current_drift_score = 0.0f;

        void AddSample(const std::vector<float>& features, float prediction);
        float CalculateDriftScore();
    } drift_detector_;

    // Private helper methods
    void CreateMinimalSchema(const std::string& model_path);
    void AdjustSchemaToModel(size_t expected_features);
    std::string GenerateParameters(const std::unordered_map<std::string, std::string>& params);
    size_t GetOutputSize() const;
    size_t GetModelSize() const;

    // Metric processing
    void ProcessMetricsByType(const std::vector<std::string>& eval_names,
                              const std::vector<double>& eval_values);

    void ProcessBinaryMetrics(const std::vector<std::string>& eval_names,
                             const std::vector<double>& eval_values);

    void ProcessMulticlassMetrics(const std::vector<std::string>& eval_names,
                                 const std::vector<double>& eval_values);

    void ProcessRegressionMetrics(const std::vector<std::string>& eval_names,
                                 const std::vector<double>& eval_values);

    // Detailed metric calculators
    void CalculateDetailedBinaryMetrics(const TrainingData::SplitData& validation_data);
    void CalculateDetailedMulticlassMetrics(const TrainingData::SplitData& validation_data);
    void CalculateDetailedRegressionMetrics(const TrainingData::SplitData& validation_data);

    // Training metric calculators (by problem type)
    void CalculateBinaryTrainingMetrics(const std::vector<std::vector<float>>& features,
                                       const std::vector<float>& labels);

    void CalculateMulticlassTrainingMetrics(const std::vector<std::vector<float>>& features,
                                           const std::vector<float>& labels);

    void CalculateRegressionTrainingMetrics(const std::vector<std::vector<float>>& features,
                                           const std::vector<float>& labels);

    void CalculatePoissonTrainingMetrics(const std::vector<std::vector<float>>& features,
                                        const std::vector<float>& labels);

    void CalculateQuantileTrainingMetrics(const std::vector<std::vector<float>>& features,
                                         const std::vector<float>& labels);

    void CalculateHuberTrainingMetrics(const std::vector<std::vector<float>>& features,
                                      const std::vector<float>& labels);

    void CalculateTweedieTrainingMetrics(const std::vector<std::vector<float>>& features,
                                        const std::vector<float>& labels);

    void CalculateGammaTrainingMetrics(const std::vector<std::vector<float>>& features,
                                      const std::vector<float>& labels);

    void CalculateFallbackMetrics(const std::vector<std::vector<float>>& features,
                                 const std::vector<float>& labels);

    // Utility
    float CalculateAUC(const std::vector<double>& predictions,
                       const std::vector<float>& labels);

    float GetMetricFromMetadata(const std::string& key, float default_value) const;
};

} // namespace esql

#endif // ADAPTIVE_LIGHTGBM_H
