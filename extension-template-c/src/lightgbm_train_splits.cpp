#include "lightgbm_model.h"
#include "algorithm_registry.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>

namespace esql {
namespace ai {

bool AdaptiveLightGBMModel::train_with_splits(
    const DataExtractor::TrainingData::SplitData& train_data,
    const DataExtractor::TrainingData::SplitData& validation_data,
    const std::unordered_map<std::string, std::string>& params,
    int early_stopping_rounds) {

    if (train_data.features.empty() || train_data.labels.empty()) {
        std::cerr << "[LightGBM] ERROR: Invalid training data" << std::endl;
        return false;
    }

    size_t num_samples = train_data.features.size();
    size_t num_features = train_data.features[0].size();

    // Validate all samples have same number of features
    for (const auto& sample : train_data.features) {
        if (sample.size() != num_features) {
            std::cerr << "[LightGBM] ERROR: Inconsistent feature sizes in training data" << std::endl;
            return false;
        }
    }

    // Log training configuration
    std::cout << "\n========== LightGBM Training with Validation ==========" << std::endl;
    std::cout << "Training samples:   " << num_samples << std::endl;
    std::cout << "Validation samples: " << validation_data.size << std::endl;
    std::cout << "Features:           " << num_features << std::endl;
    std::cout << "Problem type:       " << schema_.problem_type << std::endl;
    std::cout << "Algorithm:          " << schema_.algorithm << std::endl;
    std::cout << "Early stopping:     " << early_stopping_rounds << " rounds" << std::endl;
    std::cout << "========================================================\n" << std::endl;

    // Debug: Print first few samples
    std::cout << "[LightGBM] DEBUG: First 3 training samples:" << std::endl;
    for (size_t i = 0; i < std::min((size_t)3, num_samples); ++i) {
        std::cout << "  Sample " << i << ": Features [";
        for (size_t j = 0; j < std::min((size_t)5, num_features); ++j) {
            std::cout << train_data.features[i][j] << " ";
        }
        std::cout << "], Label: " << train_data.labels[i] << std::endl;
    }

    // Flatten features for LightGBM
    std::vector<float> flat_features;
    flat_features.reserve(num_samples * num_features);
    for (const auto& sample : train_data.features) {
        flat_features.insert(flat_features.end(), sample.begin(), sample.end());
    }

    // Create training dataset
    DatasetHandle train_dataset = nullptr;
    std::string param_str = generate_parameters(params);

    int result = LGBM_DatasetCreateFromMat(
        flat_features.data(),
        C_API_DTYPE_FLOAT32,
        static_cast<int32_t>(num_samples),
        static_cast<int32_t>(num_features),
        1,  // row major
        "", // Empty parameters for dataset creation
        nullptr,
        &train_dataset
    );

    if (result != 0 || !train_dataset) {
        std::cerr << "[LightGBM] Failed to create training dataset: "
                  << LGBM_GetLastError() << std::endl;
        return false;
    }

    // Set labels for training dataset
    std::vector<float> labels_float(train_data.labels.begin(), train_data.labels.end());
    result = LGBM_DatasetSetField(
        train_dataset,
        "label",
        labels_float.data(),
        static_cast<int>(labels_float.size()),
        C_API_DTYPE_FLOAT32
    );

    if (result != 0) {
        std::cerr << "[LightGBM] Failed to set training labels: "
                  << LGBM_GetLastError() << std::endl;
        LGBM_DatasetFree(train_dataset);
        return false;
    }

    // Create validation dataset if provided
    DatasetHandle valid_dataset = nullptr;
    if (!validation_data.empty()) {
        size_t valid_samples = validation_data.features.size();

        std::cout << "[LightGBM] Creating validation dataset with " << valid_samples << " samples" << std::endl;

        // Flatten validation features
        std::vector<float> flat_valid_features;
        flat_valid_features.reserve(valid_samples * num_features);
        for (const auto& sample : validation_data.features) {
            flat_valid_features.insert(flat_valid_features.end(),
                                      sample.begin(), sample.end());
        }

        // Create validation dataset
        result = LGBM_DatasetCreateFromMat(
            flat_valid_features.data(),
            C_API_DTYPE_FLOAT32,
            static_cast<int32_t>(valid_samples),
            static_cast<int32_t>(num_features),
            1,
            "",
            train_dataset,  // Reference dataset
            &valid_dataset
        );

        if (result != 0 || !valid_dataset) {
            std::cerr << "[LightGBM] WARNING: Failed to create validation dataset: "
                      << LGBM_GetLastError() << std::endl;
            std::cerr << "[LightGBM] Continuing without validation data" << std::endl;
            valid_dataset = nullptr;
        } else {
            // Set validation labels
            std::vector<float> valid_labels_float(validation_data.labels.begin(),
                                                  validation_data.labels.end());
            result = LGBM_DatasetSetField(
                valid_dataset,
                "label",
                valid_labels_float.data(),
                static_cast<int>(valid_labels_float.size()),
                C_API_DTYPE_FLOAT32
            );

            if (result != 0) {
                std::cerr << "[LightGBM] WARNING: Failed to set validation labels" << std::endl;
                LGBM_DatasetFree(valid_dataset);
                valid_dataset = nullptr;
            } else {
                // Verify labels were set
                int32_t label_len = 0;
                int32_t out_type = 0;
                const void* label_ptr = nullptr;
                LGBM_DatasetGetField(valid_dataset, "label", &label_len, &label_ptr, &out_type);
                std::cout << "[LightGBM] Validation labels verified: " << label_len << " labels" << std::endl;
            }
        }
    }

    // Create booster
    BoosterHandle booster = nullptr;
    std::cout << "[LightGBM] Creating booster with parameters: " << param_str << std::endl;
    result = LGBM_BoosterCreate(train_dataset, param_str.c_str(), &booster);

    if (result != 0 || !booster) {
        std::cerr << "[LightGBM] Failed to create booster: "
                  << LGBM_GetLastError() << std::endl;
        LGBM_DatasetFree(train_dataset);
        if (valid_dataset) LGBM_DatasetFree(valid_dataset);
        return false;
    }

    // Add validation dataset to booster
    if (valid_dataset) {
        std::cout << "[LightGBM] Adding validation dataset to booster..." << std::endl;
        result = LGBM_BoosterAddValidData(booster, valid_dataset);
        if (result != 0) {
            std::cerr << "[LightGBM] WARNING: Failed to add validation data: "
                      << LGBM_GetLastError() << std::endl;
        } else {
            // Verify validation was added
            int out_len = 0;
            LGBM_BoosterGetEvalCounts(booster, &out_len);
            std::cout << "[LightGBM] Number of datasets in booster: " << out_len << std::endl;
        }
    }

    // Get number of iterations
    int num_iterations = 100;
    if (params.find("num_iterations") != params.end()) {
        try {
            num_iterations = std::stoi(params.at("num_iterations"));
        } catch (...) {
            num_iterations = 100;
        }
    }

    // Training loop with early stopping
    std::cout << "[LightGBM] Training for up to " << num_iterations
              << " iterations with early stopping..." << std::endl;

    int best_iteration = 0;
    float best_score = std::numeric_limits<float>::max();
    int no_improve_count = 0;
    bool training_success = true;
    std::vector<float> validation_scores;
    std::vector<int> iteration_numbers;

    // For tracking metrics
    std::vector<std::vector<double>> all_eval_results;
    std::vector<int> iteration_list;

    for (int i = 0; i < num_iterations; ++i) {
        int is_finished = 0;
        result = LGBM_BoosterUpdateOneIter(booster, &is_finished);

        if (result != 0) {
            std::cerr << "[LightGBM] Training iteration " << i + 1
                      << " failed: " << LGBM_GetLastError() << std::endl;
            training_success = false;
            break;
        }

        if (is_finished) {
            std::cout << "[LightGBM] Early stopping at iteration " << i + 1 << std::endl;
            break;
        }

        // Check validation score if we have validation data
        if (valid_dataset && early_stopping_rounds > 0) {
            int eval_count = 0;
            result = LGBM_BoosterGetEvalCounts(booster, &eval_count);

            if (result == 0 && eval_count > 0) {
                std::vector<double> results(eval_count);
                int out_len = 0;

                // Get evaluation results for validation data (index 1)
                result = LGBM_BoosterGetEval(booster, 1, &out_len, results.data());

                if (result == 0 && out_len > 0) {
                    float current_score = static_cast<float>(results[0]);

                    // Store for learning curves
                    if (i % 10 == 0) {
                        all_eval_results.push_back(results);
                        iteration_list.push_back(i + 1);
                    }

                    // For early stopping, we want to minimize the metric
                    // (lower is better for logloss, error, rmse, etc.)
                    if (current_score < best_score - 0.0001f) {
                        best_score = current_score;
                        best_iteration = i;
                        no_improve_count = 0;

                        // Save best model state
                        LGBM_BoosterSaveModel(booster, 0, -1, 0, "best_model_temp.txt");

                        if (i % 10 == 0) {
                            std::cout << "[LightGBM] Iteration " << i + 1
                                      << ", validation score improved to " << current_score << std::endl;
                        }
                    } else {
                        no_improve_count++;
                        if (no_improve_count >= early_stopping_rounds) {
                            std::cout << "[LightGBM] Early stopping triggered at iteration "
                                      << i + 1 << " (no improvement for "
                                      << early_stopping_rounds << " rounds)" << std::endl;

                            // Load best model
                            BoosterHandle best_booster = nullptr;
                            int best_iter = 0;
                            result = LGBM_BoosterCreateFromModelfile(
                                "best_model_temp.txt",
                                &best_iter,
                                &best_booster
                            );

                            if (result == 0 && best_booster) {
                                std::lock_guard<std::mutex> lock(model_mutex_);
                                if (booster_) {
                                    LGBM_BoosterFree(booster_);
                                }
                                booster_ = best_booster;
                            }

                            std::remove("best_model_temp.txt");
                            break;
                        }
                    }

                    validation_scores.push_back(current_score);
                    iteration_numbers.push_back(i + 1);
                }
            }
        }

        if ((i + 1) % 10 == 0) {
            std::cout << "[LightGBM] Completed iteration " << (i + 1) << std::endl;
        }
    }

    // Clean up datasets
    LGBM_DatasetFree(train_dataset);
    if (valid_dataset) {
        LGBM_DatasetFree(valid_dataset);
    }

    if (!training_success) {
        LGBM_BoosterFree(booster);
        return false;
    }

    // Set final booster (if not already set by early stopping)
    if (!valid_dataset || no_improve_count < early_stopping_rounds) {
        std::lock_guard<std::mutex> lock(model_mutex_);
        if (booster_) {
            LGBM_BoosterFree(booster_);
        }
        booster_ = booster;
    }

    // Update schema statistics
    schema_.training_samples = train_data.size;
    schema_.last_updated = std::chrono::system_clock::now();

    // Calculate comprehensive metrics
    if (!validation_data.empty()) {
        std::cout << "\n[LightGBM] Calculating validation metrics using native evaluator..." << std::endl;
        extract_native_metrics(booster_, schema_);

        // Also calculate detailed metrics on validation set
        calculate_detailed_validation_metrics(validation_data);
    } else {
        std::cout << "[LightGBM] WARNING: No validation data provided. Using training data for metrics." << std::endl;
        calculate_training_metrics(train_data.features, train_data.labels);
    }

    // Store training history in metadata
    if (!validation_scores.empty()) {
        std::string history;
        for (size_t i = 0; i < validation_scores.size(); i += std::max(1, (int)validation_scores.size() / 10)) {
            if (!history.empty()) history += ";";
            history += std::to_string(iteration_numbers[i]) + ":" + std::to_string(validation_scores[i]);
        }
        schema_.metadata["validation_history"] = history;
        schema_.metadata["best_iteration"] = std::to_string(best_iteration + 1);
        schema_.metadata["best_validation_score"] = std::to_string(best_score);
    }

    // Reset drift detector
    reset_drift_detector();

    std::cout << "\n========== Training Complete ==========" << std::endl;
    std::cout << "Best iteration: " << best_iteration + 1 << std::endl;
    std::cout << "Best validation score: " << best_score << std::endl;
    std::cout << "Final accuracy: " << schema_.accuracy << std::endl;
    std::cout << "=======================================\n" << std::endl;

    return true;
}

} // namespace ai
} // namespace esql
