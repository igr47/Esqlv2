#include "include/esql/lightgbm_model.h"
#include "LightGBM/c_api.h"
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>

namespace esql {

// Constructor
AdaptiveLightGBMModel::AdaptiveLightGBMModel()
    : booster_(nullptr), is_loaded_(false), prediction_count_(0), batch_size_(1) {
}

AdaptiveLightGBMModel::AdaptiveLightGBMModel(const ModelSchema& schema)
    : schema_(schema), booster_(nullptr), is_loaded_(false), prediction_count_(0), batch_size_(1) {
}

// Destructor
AdaptiveLightGBMModel::~AdaptiveLightGBMModel() {
    if (booster_) {
        LGBM_BoosterFree(booster_);
        booster_ = nullptr;
    }
}

// Save model to file
bool AdaptiveLightGBMModel::Save(const std::string& path) {
    if (!booster_) {
        std::cerr << "[LightGBM] Cannot save model - booster is null" << std::endl;
        return false;
    }

    int result = LGBM_BoosterSaveModel(booster_, 0, -1, 0, path.c_str());
    if (result != 0) {
        std::cerr << "[LightGBM] Failed to save model: " << LGBM_GetLastError() << std::endl;
        return false;
    }

    return true;
}

// Load model from file
bool AdaptiveLightGBMModel::Load(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (booster_) {
        LGBM_BoosterFree(booster_);
        booster_ = nullptr;
    }

    int out_iterations = 0;
    int result = LGBM_BoosterCreateFromModelfile(path.c_str(), &out_iterations, &booster_);

    if (result != 0 || !booster_) {
        std::cerr << "[LightGBM] Failed to load model: " << LGBM_GetLastError() << std::endl;
        return false;
    }

    is_loaded_ = true;
    CreateMinimalSchema(path);

    return true;
}

// Predict using tensor input
Tensor AdaptiveLightGBMModel::Predict(const Tensor& input) {
    if (!booster_) {
        std::cerr << "[LightGBM] Cannot predict - model not loaded" << std::endl;
        return Tensor();
    }

    size_t num_samples = input.shape.size() >= 2 ? input.shape[0] : 1;
    size_t num_features = input.shape.size() >= 2 ? input.shape[1] : input.shape[0];

    // Prepare output
    std::vector<double> predictions;
    size_t output_size = GetOutputSize();
    predictions.resize(num_samples * output_size);

    int64_t out_len = 0;

    int result = LGBM_BoosterPredictForMat(
        booster_,
        input.Ptr(),
        C_API_DTYPE_FLOAT32,
        static_cast<int32_t>(num_samples),
        static_cast<int32_t>(num_features),
        1,  // row major
        0,  // normal prediction
        0,  // start iteration
        -1, // num iterations
        "",
        &out_len,
        predictions.data()
    );

    if (result != 0) {
        std::cerr << "[LightGBM] Prediction failed: " << LGBM_GetLastError() << std::endl;
        return Tensor();
    }

    std::vector<float> output_data;
    for (double val : predictions) {
        output_data.push_back(static_cast<float>(val));
    }

    std::vector<size_t> output_shape = {num_samples, output_size};
    return Tensor(std::move(output_data), std::move(output_shape));
}

// Predict from row data
Tensor AdaptiveLightGBMModel::PredictRow(const std::unordered_map<std::string, duckdb::Value>& row) {
    auto features = schema_.ExtractFeatures(row);
    if (features.empty()) {
        return Tensor();
    }

    Tensor input(features, {features.size()});
    return Predict(input);
}

// Batch prediction
std::vector<Tensor> AdaptiveLightGBMModel::PredictBatch(const std::vector<Tensor>& inputs) {
    std::vector<Tensor> results;
    results.reserve(inputs.size());

    for (const auto& input : inputs) {
        results.push_back(Predict(input));
    }

    return results;
}

// Get model metadata
ModelMetadata AdaptiveLightGBMModel::GetMetadata() const {
    ModelMetadata metadata;
    metadata.name = schema_.model_id;
    metadata.algorithm = schema_.algorithm;
    metadata.problem_type = schema_.problem_type;

    for (const auto& feature : schema_.features) {
        metadata.feature_names.push_back(feature.name);
    }

    metadata.target_column = schema_.target_column;
    metadata.parameters = schema_.metadata;
    metadata.created_at = schema_.created_at;
    metadata.training_samples = schema_.training_samples;
    metadata.accuracy = schema_.accuracy;
    metadata.model_path = ""; // Not stored here

    metadata.precision = GetMetricFromMetadata("precision", 0.0f);
    metadata.recall = GetMetricFromMetadata("recall", 0.0f);
    metadata.f1_score = GetMetricFromMetadata("f1_score", 0.0f);
    metadata.auc_score = GetMetricFromMetadata("auc_score", 0.0f);
    metadata.r2_score = GetMetricFromMetadata("r2_score", 0.0f);
    metadata.rmse = GetMetricFromMetadata("rmse", 0.0f);
    metadata.mae = GetMetricFromMetadata("mae", 0.0f);

    metadata.is_active = true;
    metadata.prediction_count = prediction_count_.load();
    metadata.drift_score = schema_.drift_score;

    return metadata;
}

// Train without validation splits
bool AdaptiveLightGBMModel::Train(const std::vector<std::vector<float>>& features,
                                   const std::vector<float>& labels,
                                   const std::unordered_map<std::string, std::string>& params) {
    TrainingData::SplitData train_data;
    train_data.features = features;
    train_data.labels = labels;
    train_data.size = features.size();

    TrainingData::SplitData empty_val;

    return TrainWithSplits(train_data, empty_val, params, 0);
}

// Generate parameters string for LightGBM
std::string AdaptiveLightGBMModel::GenerateParameters(
    const std::unordered_map<std::string, std::string>& params) {

    std::stringstream ss;
    bool first = true;

    for (const auto& [key, value] : params) {
        if (!first) ss << " ";
        first = false;
        ss << key << "=" << value;
    }

    // Add default parameters if not specified
    if (params.find("verbose") == params.end()) {
        if (!first) ss << " ";
        ss << "verbose=-1";
    }

    return ss.str();
}

// Get output size (number of predictions per sample)
size_t AdaptiveLightGBMModel::GetOutputSize() const {
    if (schema_.problem_type == "multiclass") {
        auto it = schema_.metadata.find("num_classes");
        if (it != schema_.metadata.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return 1;
            }
        }
        return 1;
    }
    return 1;
}

// Get model size in bytes
size_t AdaptiveLightGBMModel::GetModelSize() const {
    if (!booster_) return 0;

    // Try to get model size from file if saved
    // For now, return a placeholder
    return 1024 * 1024; // 1 MB placeholder
}

// Reset drift detector
void AdaptiveLightGBMModel::ResetDriftDetector() {
    drift_detector_.recent_features.clear();
    drift_detector_.recent_predictions.clear();
    drift_detector_.current_drift_score = 0.0f;
    drift_detector_.last_drift_check = std::chrono::system_clock::now();
}

// Check if model needs retraining
bool AdaptiveLightGBMModel::NeedsRetraining() const {
    return schema_.drift_score > 0.2f; // 20% drift threshold
}

// Set batch size
void AdaptiveLightGBMModel::SetBatchSize(size_t batch_size) {
    batch_size_ = batch_size;
}

// Warmup the model
void AdaptiveLightGBMModel::Warmup(size_t iterations) {
    if (!booster_ || iterations == 0) return;

    // Create dummy input
    if (schema_.features.empty()) return;

    std::vector<float> dummy_input(schema_.features.size(), 0.0f);
    Tensor dummy(dummy_input, {dummy_input.size()});

    for (size_t i = 0; i < iterations; ++i) {
        Predict(dummy);
    }
}

// Get memory usage
size_t AdaptiveLightGBMModel::GetMemoryUsage() const {
    return GetModelSize();
}

// Release unused memory
void AdaptiveLightGBMModel::ReleaseUnusedMemory() {
    // LightGBM doesn't have explicit memory release
    // This is a placeholder
}

// Get average inference time
float AdaptiveLightGBMModel::GetAvgInferenceTimeMs() const {
    auto duration_us = schema_.stats.avg_inference_time.count();
    return static_cast<float>(duration_us) / 1000.0f;
}

// Check if model can handle a row
bool AdaptiveLightGBMModel::CanHandleRow(
    const std::unordered_map<std::string, duckdb::Value>& row) const {

    auto missing = schema_.GetMissingFeatures(row);
    return missing.empty();
}

// Update schema
void AdaptiveLightGBMModel::UpdateSchema(const ModelSchema& new_schema) {
    schema_ = new_schema;
    schema_.last_updated = std::chrono::system_clock::now();
}

// Update feature
void AdaptiveLightGBMModel::UpdateFeature(const FeatureDescriptor& feature) {
    for (auto& f : schema_.features) {
        if (f.name == feature.name) {
            f = feature;
            break;
        }
    }
    schema_.last_updated = std::chrono::system_clock::now();
}

// Create minimal schema from model file
void AdaptiveLightGBMModel::CreateMinimalSchema(const std::string& model_path) {
    // Try to extract feature count from model
    if (booster_) {
        int num_features = 0;
        int result = LGBM_BoosterGetNumFeature(booster_, &num_features);
        if (result == 0 && num_features > 0) {
            AdjustSchemaToModel(num_features);
        }
    }

    schema_.model_id = model_path;
    schema_.created_at = std::chrono::system_clock::now();
    schema_.last_updated = schema_.created_at;
}

// Adjust schema to match model
void AdaptiveLightGBMModel::AdjustSchemaToModel(size_t expected_features) {
    // If schema has no features, create placeholders
    if (schema_.features.empty()) {
        for (size_t i = 0; i < expected_features; ++i) {
            FeatureDescriptor fd;
            fd.name = "feature_" + std::to_string(i);
            fd.db_column = fd.name;
            fd.data_type = "float";
            fd.transformation = "direct";
            fd.required = true;
            fd.is_categorical = false;
            fd.mean_value = 0.0f;
            fd.std_value = 1.0f;
            fd.min_value = 0.0f;
            fd.max_value = 1.0f;
            fd.default_value = 0.0f;
            schema_.features.push_back(fd);
        }
    }

    // Ensure we have the right number of features
    while (schema_.features.size() < expected_features) {
        FeatureDescriptor fd;
        fd.name = "feature_" + std::to_string(schema_.features.size());
        fd.db_column = fd.name;
        fd.data_type = "float";
        fd.transformation = "direct";
        fd.required = false;
        fd.default_value = 0.0f;
        schema_.features.push_back(fd);
    }
}

// Get metric from metadata
float AdaptiveLightGBMModel::GetMetricFromMetadata(const std::string& key, float default_value) const {
    auto it = schema_.metadata.find(key);
    if (it != schema_.metadata.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

// Drift detector implementation
void AdaptiveLightGBMModel::DriftDetector::AddSample(const std::vector<float>& features, float prediction) {
    recent_features.push_back(features);
    recent_predictions.push_back(prediction);

    // Keep only last 1000 samples
    if (recent_features.size() > 1000) {
        recent_features.erase(recent_features.begin());
        recent_predictions.erase(recent_predictions.begin());
    }
}

float AdaptiveLightGBMModel::DriftDetector::CalculateDriftScore() {
    // Simple drift detection based on feature distribution changes
    // This is a placeholder - in production, use statistical tests

    if (recent_features.size() < 100) {
        return 0.0f;
    }

    // Simple score based on mean shift
    std::vector<float> means(recent_features[0].size(), 0.0f);
    for (const auto& sample : recent_features) {
        for (size_t i = 0; i < sample.size(); ++i) {
            means[i] += sample[i];
        }
    }

    for (auto& mean : means) {
        mean /= recent_features.size();
    }

    // Placeholder drift score
    current_drift_score = 0.05f;
    return current_drift_score;
}

// Update schema with drift score
/*float AdaptiveLightGBMModel::GetDriftScore() const {
    return schema_.drift_score;
}*/

} // namespace esql
