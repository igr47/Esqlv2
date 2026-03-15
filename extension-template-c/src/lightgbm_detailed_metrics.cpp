#include "lightgbm_model.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace esql {
namespace ai {

void AdaptiveLightGBMModel::calculate_detailed_validation_metrics(
    const DataExtractor::TrainingData::SplitData& validation_data) {

    if (validation_data.features.empty() || !booster_) {
        std::cerr << "[LightGBM] Cannot calculate detailed metrics - no validation data" << std::endl;
        return;
    }

    std::cout << "\n========== Detailed Validation Metrics ==========" << std::endl;

    if (schema_.problem_type == "binary_classification") {
        calculate_detailed_binary_metrics(validation_data);
    }
    else if (schema_.problem_type == "multiclass") {
        calculate_detailed_multiclass_metrics(validation_data);
    }
    else {
        calculate_detailed_regression_metrics(validation_data);
    }

    std::cout << "=================================================\n" << std::endl;
}

void AdaptiveLightGBMModel::calculate_detailed_binary_metrics(
    const DataExtractor::TrainingData::SplitData& validation_data) {

    size_t n = validation_data.features.size();
    size_t num_features = validation_data.features[0].size();

    // Prepare features for prediction
    std::vector<float> flat_features;
    flat_features.reserve(n * num_features);
    for (const auto& sample : validation_data.features) {
        flat_features.insert(flat_features.end(), sample.begin(), sample.end());
    }

    // Allocate output buffer
    std::vector<double> predictions(n);
    int64_t out_len = 0;

    // Make predictions
    int result = LGBM_BoosterPredictForMat(
        booster_,
        flat_features.data(),
        C_API_DTYPE_FLOAT32,
        static_cast<int32_t>(n),
        static_cast<int32_t>(num_features),
        1,
        0,  // normal prediction
        0,
        -1,
        "",
        &out_len,
        predictions.data()
    );

    if (result != 0 || static_cast<size_t>(out_len) != n) {
        std::cerr << "[LightGBM] Failed to make predictions for metrics" << std::endl;
        return;
    }

    // Calculate confusion matrix
    int64_t tp = 0, tn = 0, fp = 0, fn = 0;

    for (size_t i = 0; i < n; ++i) {
        bool pred_class = predictions[i] > 0.5;
        bool true_class = validation_data.labels[i] > 0.5;

        if (pred_class && true_class) tp++;
        else if (!pred_class && !true_class) tn++;
        else if (pred_class && !true_class) fp++;
        else if (!pred_class && true_class) fn++;
    }

    // Calculate metrics
    float accuracy = static_cast<float>(tp + tn) / n;
    float precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
    float recall = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
    float specificity = (tn + fp > 0) ? static_cast<float>(tn) / (tn + fp) : 0.0f;
    float f1_score = (precision + recall > 0) ? 2.0f * (precision * recall) / (precision + recall) : 0.0f;

    // Calculate AUC (simplified)
    float auc = calculate_auc(predictions, validation_data.labels);

    // Store in metadata
    schema_.metadata["validation_accuracy"] = std::to_string(accuracy);
    schema_.metadata["validation_precision"] = std::to_string(precision);
    schema_.metadata["validation_recall"] = std::to_string(recall);
    schema_.metadata["validation_specificity"] = std::to_string(specificity);
    schema_.metadata["validation_f1"] = std::to_string(f1_score);
    schema_.metadata["validation_auc"] = std::to_string(auc);
    schema_.metadata["true_positives"] = std::to_string(tp);
    schema_.metadata["true_negatives"] = std::to_string(tn);
    schema_.metadata["false_positives"] = std::to_string(fp);
    schema_.metadata["false_negatives"] = std::to_string(fn);

    // Update accuracy if better than existing
    if (accuracy > schema_.accuracy) {
        schema_.accuracy = accuracy;
    }

    // Print metrics
    std::cout << "Binary Classification Metrics:" << std::endl;
    std::cout << "  Accuracy:     " << std::fixed << std::setprecision(4) << accuracy << std::endl;
    std::cout << "  Precision:    " << precision << std::endl;
    std::cout << "  Recall:       " << recall << std::endl;
    std::cout << "  Specificity:  " << specificity << std::endl;
    std::cout << "  F1 Score:     " << f1_score << std::endl;
    std::cout << "  AUC:          " << auc << std::endl;
    std::cout << "\n  Confusion Matrix:" << std::endl;
    std::cout << "    TP: " << tp << "  FP: " << fp << std::endl;
    std::cout << "    FN: " << fn << "  TN: " << tn << std::endl;
}

void AdaptiveLightGBMModel::calculate_detailed_multiclass_metrics(
    const DataExtractor::TrainingData::SplitData& validation_data) {

    size_t n = validation_data.features.size();
    size_t num_features = validation_data.features[0].size();

    // Get number of classes
    size_t num_classes = 1;
    auto it = schema_.metadata.find("num_classes");
    if (it != schema_.metadata.end()) {
        try {
            num_classes = std::stoi(it->second);
        } catch (...) {
            num_classes = 1;
        }
    }

    if (num_classes <= 1) {
        // Try to infer from data
        std::unordered_set<float> unique_labels;
        for (float label : validation_data.labels) {
            unique_labels.insert(label);
        }
        num_classes = unique_labels.size();
        schema_.metadata["num_classes"] = std::to_string(num_classes);
    }

    // Prepare features for prediction
    std::vector<float> flat_features;
    flat_features.reserve(n * num_features);
    for (const auto& sample : validation_data.features) {
        flat_features.insert(flat_features.end(), sample.begin(), sample.end());
    }

    // Allocate output buffer for probabilities
    std::vector<double> predictions(n * num_classes);
    int64_t out_len = 0;

    // Make predictions (probabilities)
    int result = LGBM_BoosterPredictForMat(
        booster_,
        flat_features.data(),
        C_API_DTYPE_FLOAT32,
        static_cast<int32_t>(n),
        static_cast<int32_t>(num_features),
        1,
        1,  // predict raw score (returns probabilities)
        0,
        -1,
        "",
        &out_len,
        predictions.data()
    );

    if (result != 0 || static_cast<size_t>(out_len) != n * num_classes) {
        std::cerr << "[LightGBM] Failed to make predictions for multiclass metrics" << std::endl;
        return;
    }

    // Initialize confusion matrix
    std::vector<std::vector<int64_t>> confusion_matrix(num_classes,
                                                       std::vector<int64_t>(num_classes, 0));

    // Calculate predicted classes
    std::vector<size_t> predicted_classes(n);
    std::vector<std::vector<float>> probabilities(n, std::vector<float>(num_classes));

    for (size_t i = 0; i < n; ++i) {
        size_t pred_class = 0;
        double max_prob = predictions[i * num_classes];

        for (size_t c = 0; c < num_classes; ++c) {
            float prob = static_cast<float>(predictions[i * num_classes + c]);
            probabilities[i][c] = prob;
            if (prob > max_prob) {
                max_prob = prob;
                pred_class = c;
            }
        }

        predicted_classes[i] = pred_class;
    }

    // Build confusion matrix
    for (size_t i = 0; i < n; ++i) {
        size_t true_class = static_cast<size_t>(validation_data.labels[i]);
        if (true_class < num_classes && predicted_classes[i] < num_classes) {
            confusion_matrix[true_class][predicted_classes[i]]++;
        }
    }

    // Calculate per-class metrics
    std::vector<float> per_class_precision(num_classes, 0.0f);
    std::vector<float> per_class_recall(num_classes, 0.0f);
    std::vector<float> per_class_f1(num_classes, 0.0f);

    int64_t total_correct = 0;

    for (size_t c = 0; c < num_classes; ++c) {
        int64_t tp = confusion_matrix[c][c];
        total_correct += tp;

        // Calculate false positives (sum of column c except diagonal)
        int64_t fp = 0;
        for (size_t i = 0; i < num_classes; ++i) {
            if (i != c) fp += confusion_matrix[i][c];
        }

        // Calculate false negatives (sum of row c except diagonal)
        int64_t fn = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            if (j != c) fn += confusion_matrix[c][j];
        }

        per_class_precision[c] = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
        per_class_recall[c] = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
        per_class_f1[c] = (per_class_precision[c] + per_class_recall[c] > 0) ?
            2.0f * (per_class_precision[c] * per_class_recall[c]) /
            (per_class_precision[c] + per_class_recall[c]) : 0.0f;
    }

    // Calculate macro-averaged metrics
    float macro_precision = 0.0f;
    float macro_recall = 0.0f;
    float macro_f1 = 0.0f;

    for (size_t c = 0; c < num_classes; ++c) {
        macro_precision += per_class_precision[c];
        macro_recall += per_class_recall[c];
        macro_f1 += per_class_f1[c];
    }

    macro_precision /= num_classes;
    macro_recall /= num_classes;
    macro_f1 /= num_classes;

    // Calculate micro-averaged metrics (same as accuracy)
    float micro_precision = static_cast<float>(total_correct) / n;

    // Store in metadata
    schema_.metadata["validation_accuracy"] = std::to_string(micro_precision);
    schema_.metadata["validation_macro_precision"] = std::to_string(macro_precision);
    schema_.metadata["validation_macro_recall"] = std::to_string(macro_recall);
    schema_.metadata["validation_macro_f1"] = std::to_string(macro_f1);
    schema_.metadata["validation_micro_precision"] = std::to_string(micro_precision);

    // Update accuracy
    schema_.accuracy = micro_precision;

    // Print metrics
    std::cout << "Multiclass Classification Metrics (" << num_classes << " classes):" << std::endl;
    std::cout << "  Accuracy:           " << std::fixed << std::setprecision(4) << micro_precision << std::endl;
    std::cout << "  Macro Precision:    " << macro_precision << std::endl;
    std::cout << "  Macro Recall:       " << macro_recall << std::endl;
    std::cout << "  Macro F1:           " << macro_f1 << std::endl;

    // Print per-class metrics
    std::cout << "\n  Per-Class Metrics:" << std::endl;
    for (size_t c = 0; c < num_classes; ++c) {
        std::cout << "    Class " << c << ": Precision=" << per_class_precision[c]
                  << ", Recall=" << per_class_recall[c]
                  << ", F1=" << per_class_f1[c] << std::endl;
    }
}

void AdaptiveLightGBMModel::calculate_detailed_regression_metrics(
    const DataExtractor::TrainingData::SplitData& validation_data) {

    size_t n = validation_data.features.size();
    size_t num_features = validation_data.features[0].size();

    // Prepare features for prediction
    std::vector<float> flat_features;
    flat_features.reserve(n * num_features);
    for (const auto& sample : validation_data.features) {
        flat_features.insert(flat_features.end(), sample.begin(), sample.end());
    }

    // Allocate output buffer
    std::vector<double> predictions(n);
    int64_t out_len = 0;

    // Make predictions
    int result = LGBM_BoosterPredictForMat(
        booster_,
        flat_features.data(),
        C_API_DTYPE_FLOAT32,
        static_cast<int32_t>(n),
        static_cast<int32_t>(num_features),
        1,
        0,
        0,
        -1,
        "",
        &out_len,
        predictions.data()
    );

    if (result != 0 || static_cast<size_t>(out_len) != n) {
        std::cerr << "[LightGBM] Failed to make predictions for regression metrics" << std::endl;
        return;
    }

    // Calculate residuals and errors
    std::vector<float> residuals(n);
    std::vector<float> absolute_errors(n);
    std::vector<float> squared_errors(n);
    std::vector<float> relative_errors(n);

    float sum_true = 0.0f;
    float sum_pred = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float true_val = validation_data.labels[i];
        float pred_val = static_cast<float>(predictions[i]);

        sum_true += true_val;
        sum_pred += pred_val;

        float error = pred_val - true_val;
        residuals[i] = error;
        absolute_errors[i] = std::abs(error);
        squared_errors[i] = error * error;

        if (true_val != 0.0f) {
            relative_errors[i] = std::abs(error) / std::abs(true_val);
        } else {
            relative_errors[i] = (std::abs(error) < 1e-6) ? 0.0f : 1.0f;
        }
    }

    float mean_true = sum_true / n;
    float mean_pred = sum_pred / n;

    // Calculate R²
    float ss_total = 0.0f;
    float ss_residual = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float true_val = validation_data.labels[i];
        float pred_val = static_cast<float>(predictions[i]);

        ss_total += (true_val - mean_true) * (true_val - mean_true);
        ss_residual += (true_val - pred_val) * (true_val - pred_val);
    }

    float r2 = (ss_total > 0) ? 1.0f - (ss_residual / ss_total) : 0.0f;

    // Calculate RMSE, MAE, MAPE
    float mse = ss_residual / n;
    float rmse = std::sqrt(mse);

    float mae = 0.0f;
    for (float ae : absolute_errors) {
        mae += ae;
    }
    mae /= n;

    float mape = 0.0f;
    for (float re : relative_errors) {
        mape += re;
    }
    mape = (mape / n) * 100.0f;

    // Calculate Median Absolute Error
    std::vector<float> sorted_abs_errors = absolute_errors;
    std::sort(sorted_abs_errors.begin(), sorted_abs_errors.end());
    float medae = sorted_abs_errors[n / 2];

    // Calculate error distribution statistics
    float mean_error = 0.0f;
    for (float err : residuals) {
        mean_error += err;
    }
    mean_error /= n;

    float error_variance = 0.0f;
    for (float err : residuals) {
        float diff = err - mean_error;
        error_variance += diff * diff;
    }
    error_variance /= n;
    float error_std = std::sqrt(error_variance);

    // Calculate within-tolerance metrics
    float within_5_percent = 0.0f;
    float within_10_percent = 0.0f;
    float within_20_percent = 0.0f;
    float within_1_std = 0.0f;
    float within_2_std = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float true_val = validation_data.labels[i];
        float pred_val = static_cast<float>(predictions[i]);
        float abs_error = absolute_errors[i];

        if (true_val != 0.0f) {
            float rel_error = abs_error / std::abs(true_val);
            if (rel_error <= 0.05f) within_5_percent += 1.0f;
            if (rel_error <= 0.10f) within_10_percent += 1.0f;
            if (rel_error <= 0.20f) within_20_percent += 1.0f;
        }

        if (std::abs(pred_val - true_val) <= error_std) within_1_std += 1.0f;
        if (std::abs(pred_val - true_val) <= 2.0f * error_std) within_2_std += 1.0f;
    }

    within_5_percent = (within_5_percent / n) * 100.0f;
    within_10_percent = (within_10_percent / n) * 100.0f;
    within_20_percent = (within_20_percent / n) * 100.0f;
    within_1_std = (within_1_std / n) * 100.0f;
    within_2_std = (within_2_std / n) * 100.0f;

    // Calculate prediction interval coverage
    float coverage_90 = 0.0f;
    float coverage_95 = 0.0f;
    float coverage_99 = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float pred_val = static_cast<float>(predictions[i]);
        float true_val = validation_data.labels[i];

        float lower_90 = pred_val - 1.645f * error_std;
        float upper_90 = pred_val + 1.645f * error_std;
        float lower_95 = pred_val - 1.960f * error_std;
        float upper_95 = pred_val + 1.960f * error_std;
        float lower_99 = pred_val - 2.576f * error_std;
        float upper_99 = pred_val + 2.576f * error_std;

        if (true_val >= lower_90 && true_val <= upper_90) coverage_90 += 1.0f;
        if (true_val >= lower_95 && true_val <= upper_95) coverage_95 += 1.0f;
        if (true_val >= lower_99 && true_val <= upper_99) coverage_99 += 1.0f;
    }

    coverage_90 = (coverage_90 / n) * 100.0f;
    coverage_95 = (coverage_95 / n) * 100.0f;
    coverage_99 = (coverage_99 / n) * 100.0f;

    // Store in metadata
    schema_.metadata["validation_r2"] = std::to_string(r2);
    schema_.metadata["validation_rmse"] = std::to_string(rmse);
    schema_.metadata["validation_mae"] = std::to_string(mae);
    schema_.metadata["validation_mape"] = std::to_string(mape);
    schema_.metadata["validation_medae"] = std::to_string(medae);
    schema_.metadata["validation_mse"] = std::to_string(mse);
    schema_.metadata["validation_error_mean"] = std::to_string(mean_error);
    schema_.metadata["validation_error_std"] = std::to_string(error_std);
    schema_.metadata["validation_within_5_percent"] = std::to_string(within_5_percent);
    schema_.metadata["validation_within_10_percent"] = std::to_string(within_10_percent);
    schema_.metadata["validation_within_20_percent"] = std::to_string(within_20_percent);
    schema_.metadata["validation_within_1_std"] = std::to_string(within_1_std);
    schema_.metadata["validation_within_2_std"] = std::to_string(within_2_std);
    schema_.metadata["validation_coverage_90"] = std::to_string(coverage_90);
    schema_.metadata["validation_coverage_95"] = std::to_string(coverage_95);
    schema_.metadata["validation_coverage_99"] = std::to_string(coverage_99);

    // Update accuracy
    if (r2 > 0.0f && r2 < 1.0f) {
        schema_.accuracy = r2;
    } else if (r2 >= 1.0f) {
        schema_.accuracy = 0.99f;
    } else if (r2 < 0.0f) {
        schema_.accuracy = std::max(0.0f, 1.0f - rmse / 100.0f);
    }

    // Print metrics
    std::cout << "Regression Metrics:" << std::endl;
    std::cout << "  R²:                 " << std::fixed << std::setprecision(4) << r2 << std::endl;
    std::cout << "  RMSE:               " << rmse << std::endl;
    std::cout << "  MAE:                " << mae << std::endl;
    std::cout << "  MAPE:               " << mape << "%" << std::endl;
    std::cout << "  MedAE:              " << medae << std::endl;
    std::cout << "  Error Mean:         " << mean_error << std::endl;
    std::cout << "  Error Std Dev:      " << error_std << std::endl;
    std::cout << "\n  Precision-like metrics:" << std::endl;
    std::cout << "    Within 5%:         " << within_5_percent << "%" << std::endl;
    std::cout << "    Within 10%:        " << within_10_percent << "%" << std::endl;
    std::cout << "    Within 20%:        " << within_20_percent << "%" << std::endl;
    std::cout << "    Within 1σ:          " << within_1_std << "%" << std::endl;
    std::cout << "    Within 2σ:          " << within_2_std << "%" << std::endl;
    std::cout << "\n  Prediction Interval Coverage:" << std::endl;
    std::cout << "    90% CI:            " << coverage_90 << "%" << std::endl;
    std::cout << "    95% CI:            " << coverage_95 << "%" << std::endl;
    std::cout << "    99% CI:            " << coverage_99 << "%" << std::endl;
}

float AdaptiveLightGBMModel::calculate_auc(
    const std::vector<double>& predictions,
    const std::vector<float>& labels) {

    // Simplified AUC calculation
    std::vector<std::pair<float, bool>> pairs;
    for (size_t i = 0; i < predictions.size(); ++i) {
        pairs.emplace_back(static_cast<float>(predictions[i]), labels[i] > 0.5f);
    }

    // Sort by prediction score descending
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int64_t positives = 0;
    int64_t negatives = 0;
    for (const auto& p : pairs) {
        if (p.second) positives++;
        else negatives++;
    }

    if (positives == 0 || negatives == 0) return 0.5f;

    int64_t rank_sum = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i].second) {
            rank_sum += static_cast<int64_t>(i + 1);
        }
    }

    float auc = (static_cast<float>(rank_sum) - (positives * (positives + 1) / 2.0f)) /
                (positives * negatives);

    return auc;
}

} // namespace ai
} // namespace esql
