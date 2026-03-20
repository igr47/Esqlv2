// lightgbm_model_schema_impl.cpp
#include "include/esql/lightgbm_model.h"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include <algorithm>
#include <sstream>
#include <random>
#include <numeric>

namespace esql {

// Helper function to convert Value to float
static float ValueToFloat(const duckdb::Value& val) {
    if (val.IsNull()) return 0.0f;

    switch (val.type().id()) {
        case duckdb::LogicalTypeId::INTEGER:
        case duckdb::LogicalTypeId::BIGINT:
        case duckdb::LogicalTypeId::HUGEINT:
            return static_cast<float>(val.GetValue<int64_t>());
        case duckdb::LogicalTypeId::DOUBLE:
        case duckdb::LogicalTypeId::FLOAT:
        case duckdb::LogicalTypeId::DECIMAL:
            return static_cast<float>(val.GetValue<double>());
        case duckdb::LogicalTypeId::BOOLEAN:
            return val.GetValue<bool>() ? 1.0f : 0.0f;
        case duckdb::LogicalTypeId::VARCHAR:
            try {
                return std::stof(val.ToString());
            } catch (...) {
                // Hash string for categorical encoding
                size_t hash = std::hash<std::string>{}(val.ToString());
                return static_cast<float>(hash % 1000) / 1000.0f;
            }
        default:
            return 0.0f;
    }
}

// FeatureDescriptor::transform implementation
float FeatureDescriptor::transform(const duckdb::Value& value) const {
    float raw_value = ValueToFloat(value);

    // Apply transformation
    if (transformation == "normalize" || transformation == "standard") {
        if (std_value > 0.0f) {
            return (raw_value - mean_value) / std_value;
        }
        return raw_value - mean_value;
    }
    else if (transformation == "minmax") {
        float range = max_value - min_value;
        if (range > 1e-10f) {
            return (raw_value - min_value) / range;
        }
        return 0.5f;
    }
    else if (transformation == "log") {
        return std::log(std::max(raw_value, 1e-10f));
    }
    else if (transformation == "sqrt") {
        return std::sqrt(std::max(raw_value, 0.0f));
    }

    return raw_value;
}

// FeatureDescriptor::ToString
std::string FeatureDescriptor::ToString() const {
    std::stringstream ss;
    ss << name << " (" << db_column << ", " << data_type;
    if (transformation != "direct") {
        ss << ", " << transformation;
    }
    if (is_categorical) {
        ss << ", categorical";
    }
    ss << ")";
    return ss.str();
}

// FeatureDescriptor::ToJson
nlohmann::json FeatureDescriptor::ToJson() const {
    nlohmann::json j;
    j["name"] = name;
    j["db_column"] = db_column;
    j["data_type"] = data_type;
    j["transformation"] = transformation;
    j["default_value"] = default_value;
    j["required"] = required;
    j["is_categorical"] = is_categorical;
    j["min_value"] = min_value;
    j["max_value"] = max_value;
    j["mean_value"] = mean_value;
    j["std_value"] = std_value;

    if (is_categorical && !categories.empty()) {
        j["categories"] = categories;
    }

    return j;
}

// FeatureDescriptor::FromJson
FeatureDescriptor FeatureDescriptor::FromJson(const nlohmann::json& j) {
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

// ModelSchema::ExtractFeatures implementation
std::vector<float> ModelSchema::ExtractFeatures(
    const std::unordered_map<std::string, duckdb::Value>& row) const {

    std::vector<float> features;
    features.reserve(this->features.size());

    for (const auto& feature : this->features) {
        auto it = row.find(feature.db_column);
        if (it != row.end()) {
            features.push_back(feature.transform(it->second));
        } else if (feature.required) {
            // Missing required feature, use default
            features.push_back(feature.default_value);
        } else {
            // Optional feature missing, use default
            features.push_back(feature.default_value);
        }
    }

    return features;
}

// ModelSchema::GetMissingFeatures implementation
std::vector<std::string> ModelSchema::GetMissingFeatures(
    const std::unordered_map<std::string, duckdb::Value>& row) const {

    std::vector<std::string> missing;

    for (const auto& feature : this->features) {
        if (feature.required) {
            auto it = row.find(feature.db_column);
            if (it == row.end() || it->second.IsNull()) {
                missing.push_back(feature.name);
            }
        }
    }

    return missing;
}

// ModelSchema::MatchesRow implementation
bool ModelSchema::MatchesRow(
    const std::unordered_map<std::string, duckdb::Value>& row) const {

    auto missing = GetMissingFeatures(row);
    return missing.empty();
}

// ModelSchema::ToJson implementation
nlohmann::json ModelSchema::ToJson() const {
    nlohmann::json j;
    j["model_id"] = model_id;
    j["description"] = description;
    j["target_column"] = target_column;
    j["algorithm"] = algorithm;
    j["problem_type"] = problem_type;
    j["created_at"] = std::chrono::system_clock::to_time_t(created_at);
    j["last_updated"] = std::chrono::system_clock::to_time_t(last_updated);
    j["training_samples"] = training_samples;
    j["accuracy"] = accuracy;
    j["drift_score"] = drift_score;
    j["metadata"] = metadata;

    nlohmann::json features_json = nlohmann::json::array();
    for (const auto& f : features) {
        features_json.push_back(f.ToJson());
    }
    j["features"] = features_json;

    nlohmann::json stats_json;
    stats_json["total_predictions"] = stats.total_predictions;
    stats_json["failed_predictions"] = stats.failed_predictions;
    stats_json["avg_confidence"] = stats.avg_confidence;
    stats_json["avg_inference_time"] = stats.avg_inference_time.count();
    j["stats"] = stats_json;

    return j;
}

// ModelSchema::FromJson implementation
ModelSchema ModelSchema::FromJson(const nlohmann::json& j) {
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

    if (j.contains("metadata")) {
        schema.metadata = j["metadata"].get<std::unordered_map<std::string, std::string>>();
    }

    if (j.contains("features")) {
        for (const auto& f : j["features"]) {
            schema.features.push_back(FeatureDescriptor::FromJson(f));
        }
    }

    if (j.contains("stats")) {
        schema.stats.total_predictions = j["stats"]["total_predictions"];
        schema.stats.failed_predictions = j["stats"]["failed_predictions"];
        schema.stats.avg_confidence = j["stats"]["avg_confidence"];
        schema.stats.avg_inference_time = std::chrono::microseconds(
            j["stats"]["avg_inference_time"]);
    }

    return schema;
}

// ModelSchema::GetMetadataFloat implementation
float ModelSchema::GetMetadataFloat(const std::string& key, float default_value) const {
    auto it = metadata.find(key);
    if (it != metadata.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

// ModelSchema::ToString implementation
std::string ModelSchema::ToString() const {
    std::stringstream ss;
    ss << "Model " << model_id << " (" << algorithm << ")\n";
    ss << "  Problem Type: " << problem_type << "\n";
    ss << "  Target: " << target_column << "\n";
    ss << "  Features: " << features.size() << "\n";
    ss << "  Training Samples: " << training_samples << "\n";
    ss << "  Accuracy: " << accuracy << "\n";
    ss << "  Created: " << std::chrono::system_clock::to_time_t(created_at) << "\n";

    for (const auto& f : features) {
        ss << "    - " << f.ToString() << "\n";
    }

    return ss.str();
}

// TrainingData::Split implementation
void TrainingData::Split(float train_ratio, float test_ratio,
                         float validation_ratio, int seed) {
    if (features.empty()) return;

    // Create random number generator
    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::vector<size_t> indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t train_count = static_cast<size_t>(features.size() * train_ratio);
    size_t test_count = static_cast<size_t>(features.size() * test_ratio);

    // Clear existing splits
    train.Clear();
    test.Clear();
    validation.Clear();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (i < train_count) {
            train.features.push_back(features[idx]);
            train.labels.push_back(labels[idx]);
            train.size++;
        } else if (i < train_count + test_count) {
            test.features.push_back(features[idx]);
            test.labels.push_back(labels[idx]);
            test.size++;
        } else {
            validation.features.push_back(features[idx]);
            validation.labels.push_back(labels[idx]);
            validation.size++;
        }
    }
}

// TrainingData::ToString implementation
std::string TrainingData::ToString() const {
    std::stringstream ss;
    ss << "TrainingData:\n";
    ss << "  Total Samples: " << total_samples << "\n";
    ss << "  Valid Samples: " << valid_samples << "\n";
    ss << "  Features: " << features.size() << " samples x "
       << (features.empty() ? 0 : features[0].size()) << " features\n";
    ss << "  Train Split: " << train.size << " samples\n";
    ss << "  Test Split: " << test.size << " samples\n";
    ss << "  Validation Split: " << validation.size << " samples\n";
    return ss.str();
}

} // namespace esql
