#include "algorithm_registry.cpp"
#include <iostram>
#include <algorithm>

namespace esql {
    AlgorithRegistry::AlgorithmRegistry() {
        InitializeDefaultAlgorithms();
    }

    AlgorithmRegistry &AlgorithmRegistry::Instance() {
        static AlgorithmRegistry instance;
        return instance;
    }

    void AlgorithmRegistry::InitializeDefaultAlgorithms() {
        // ===== REGRESSION ALGORITHMS =====

        RegisterAlgorithm({
            "REGRESSION", "regression", AlgorithmCategory::REGRESSION,
            "Standard L2 loss regression (mean squared error)",
            {"REGRESSION", "CONTINUOUS", "NUMERIC"},
            {{"metric", "rmse"}, {"boosting", "gbdt"}}
        });

        RegisterAlgorithm({
            "REGRESSION_L1", "regression_l1", AlgorithmCategory::REGRESSION,
            "L1 loss regression (mean absolute error), robust to outliers",
            {"REGRESSION", "CONTINUOUS", "NUMERIC"},
            {{"metric", "mae"}, {"boosting", "gbdt"}}
        });

        RegisterAlgorithm({
            "HUBER", "huber", AlgorithmCategory::REGRESSION,
            "Huber loss regression, less sensitive to outliers",
            {"REGRESSION", "CONTINUOUS", "NUMERIC"},
            {{"metric", "huber"}, {"boosting", "gbdt"}}
        });

        RegisterAlgorithm({
            "POISSON", "poisson", AlgorithmCategory::REGRESSION,
            "Poisson regression for count data (non-negative integers)",
            {"COUNT", "POISSON", "NON_NEGATIVE"},
            {{"metric", "poisson"}, {"boosting", "gbdt"}}
        });

        RegisterAlgorithm({
            "QUANTILE", "quantile", AlgorithmCategory::REGRESSION,
            "Quantile regression for predicting specific percentiles",
            {"QUANTILE", "PERCENTILE", "REGRESSION"},
            {{"metric", "quantile"}, {"alpha", "0.5"}, {"boosting", "gbdt"}}
        });

        RegisterAlgorithm({
            "GAMMA", "gamma", AlgorithmCategory::REGRESSION,
            "Gamma regression for positive continuous data",
            {"GAMMA", "POSITIVE", "CONTINUOUS"},
            {{"metric", "gamma"}, {"boosting", "gbdt"}}
        });

        RegisterAlgorithm({
            "TWEEDIE", "tweedie", AlgorithmCategory::REGRESSION,
            "Tweedie regression for zero-inflated data",
            {"TWEEDIE", "ZERO_INFLATED", "POSITIVE"},
            {{"metric", "tweedie"}, {"tweedie_variance_power", "1.5"}, {"boosting", "gbdt"}}
        });

        RegisterAlgorithm({
            "FAIR", "fair", AlgorithmCategory::REGRESSION,
            "Fair loss regression with balanced sensitivity",
            {"REGRESSION", "CONTINUOUS", "OUTLIERS"},
            {{"metric", "fair"}, {"fair_c", "1.0"}, {"boosting", "gbdt"}}
        });

        RegisterAlgorithm({
            "MAPE", "mape", AlgorithmCategory::REGRESSION,
            "Mean Absolute Percentage Error regression",
            {"PERCENTAGE", "SCALE_INVARIANT", "REGRESSION"},
            {{"metric", "mape"}, {"boosting", "gbdt"}}
        });

        // ===== CLASSIFICATION ALGORITHMS =====

        RegisterAlgorithm({
            "BINARY", "binary", AlgorithmCategory::CLASSIFICATION,
            "Binary classification with log loss",
            {"BINARY", "CLASSIFICATION", "TWO_CLASS"},
            {{"metric", "binary_logloss"}, {"boosting", "gbdt"}},
            false, true
        });

        RegisterAlgorithm({
            "MULTICLASS", "multiclass", AlgorithmCategory::CLASSIFICATION,
            "Multi-class classification with softmax",
            {"MULTICLASS", "CLASSIFICATION", "CATEGORICAL"},
            {{"metric", "multi_logloss"}, {"boosting", "gbdt"}},
            true, true
        });

        RegisterAlgorithm({
            "MULTICLASS_OVA", "multiclassova", AlgorithmCategory::CLASSIFICATION,
            "Multi-class One-vs-All classification",
            {"MULTICLASS", "CLASSIFICATION", "CATEGORICAL"},
            {{"metric", "multi_logloss"}, {"boosting", "gbdt"}},
            true, true
        });

        RegisterAlgorithm({
            "CROSS_ENTROPY", "cross_entropy", AlgorithmCategory::CLASSIFICATION,
            "Cross-entropy loss for probability estimation",
            {"CLASSIFICATION", "PROBABILITY"},
            {{"metric", "cross_entropy"}, {"boosting", "gbdt"}},
            false, true
        });

        // ===== RANKING ALGORITHMS =====

        RegisterAlgorithm({
            "LAMBDARANK", "lambdarank", AlgorithmCategory::RANKING,
            "LambdaRank for learning-to-rank problems",
            {"RANKING", "ORDER", "RECOMMENDATION"},
            {{"metric", "ndcg"}, {"boosting", "gbdt"}}
        });

        RegisterAlgorithm({
            "RANK_XENDCG", "rank_xendcg", AlgorithmCategory::RANKING,
            "RankXENDCG for normalized DCG optimization",
            {"RANKING", "NDCG", "RECOMMENDATION"},
            {{"metric", "ndcg"}, {"boosting", "gbdt"}}
        });

        // ===== ANOMALY DETECTION ALGORITHMS =====
        RegisterAlgorithm({
            "ANOMALY_ISO_FOREST", "regression", AlgorithmCategory::REGRESSION,
            "Isolation Forest for anomaly detection",
            {"ANOMALY", "OUTLIER", "FRAUD"},
            {{"metric", "rmse"}, {"boosting", "gbdt"}, {"objective", "anomaly"}}
        });

        // ===== CLUSTERING ALGORITHMS =====
        RegisterAlgorithm({
            "CLUSTERING_KMEANS", "multiclass", AlgorithmCategory::CLASSIFICATION,
            "K-Means clustering (using multiclass classification)",
            {"CLUSTERING", "SEGMENTATION", "UNSUPERVISED"},
            {{"metric", "multi_logloss"}, {"boosting", "gbdt"}, {"num_class", "8"}},
            true, false
        });

        // ===== FORECASTING ALGORITHMS =====
        RegisterAlgorithm({
            "FORECAST_ARIMA", "regression", AlgorithmCategory::REGRESSION,
            "ARIMA-like time series forecasting",
            {"FORECAST", "TIMESERIES", "ARIMA"},
            {{"metric", "mae"}, {"boosting", "gbdt"}, {"max_lag", "7"}}
        });

        RegisterAlgorithm({
            "FORECAST_PROPHET", "regression", AlgorithmCategory::REGRESSION,
            "Facebook Prophet-like forecasting",
            {"FORECAST", "TIMESERIES", "SEASONAL"},
            {{"metric", "mape"}, {"boosting", "gbdt"}, {"seasonality", "auto"}}
        });
    }

    bool AlgorithmRegistry::RegisterAlgorithm(const AlgorithmInfo& info) {
        std::string upper_name = info.name;
        std::transform(upper_name.begin(), upper_name.end(), upper_name.beggin(), ::toupper);

        if (algorithms_.find(upper_name) != algorithms_.end()) {
            return false;
        }
        algorithms_[upper_name] = info;
        by_category_[info.category].push_back(upper_name);
        return true;
    }

    const AlgoritmInfo *AlgorithmRegistry::GetAlgorithm(const std::string& name) const {
        std::string upper_name = name;
        std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);
        auto it = algorithms_.find(upper_name);
        return it != algorithms_.end() ? &it->second : nullptr;
    }

    std::vector<std::string> AlgorithmRegistry::GetSupportedAlgorithms() const {
        std::vector<std::string> result;
        for (const auto &[name, _] : algorithms_) {
            result.push_back(name);
        }
        return result;
    }

    std::vector<std::string> AlgorithmRegistry::GetAlgorithmsByCategory(AlgorithmCategory categor) const {
        auto it = by_category_.find(category);
        return it != by_category_.end() ? it->second : std::vector<std::string>();
    }

    bool AlgorithmRegistry:IsAlgorithmSupported(const std::string& name) const {
        std::string upper_name = name;
        std::transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);
        return algorithms_.find(upper_name) != algorithms_.end();
    }

    bool AlgorithmInfo::IsSuitableFor(const std::string& target_type, size_t num_classes) const {
        std::string upper_target = target_type;
        std::transform(upper_target.begin(), upper_target.end(), upper_target.begin(), ::toupper);

        for (const auto& problem : suitable_problems) {
            std::string upper_problem = problem;
            std::transform(upper_problem.begin(), upper_problem.end(), upper_problem.begin(), ::toupper);
            if (upper_target.find(upper_problem) != std::string::npos) {
                if (requires_num_classes) {
                    return num_classes > 0;
                }
                return true;
            }
        }
        return false;
    }

    std::string AlgorithmRegistry::SuggestAlgorithm(const std::string& problem_type,const std::vector<float>& sample_labels) const {
        std::string upper_problem = problem_type;
        std::transform(upper_problem.begin(), upper_problem.end(),upper_problem.begin(), ::toupper);

        // Analyze sample labels
        std::unordered_set<float> unique_values;
        bool all_integer = true;
        bool all_non_negative = true;
        bool has_zeros = false;

        for (float label : sample_labels) {
            unique_values.insert(label);
            if (std::abs(label - std::round(label)) > 1e-6) all_integer = false;
            if (label < 0) all_non_negative = false;
            if (label == 0) has_zeros = true;
        }

        size_t unique_labels = unique_values.size();

        // Auto-detect problem type
        if (upper_problem.empty() || upper_problem == "AUTO") {
            if (unique_labels == 2) {
                upper_problem = "BINARY";
            } else if (unique_labels > 2 && unique_labels < 20) {
                upper_problem = "MULTICLASS";
            } else if (all_integer && all_non_negative && unique_labels > 20) {
                upper_problem = "COUNT";
            } else {
                upper_problem = "REGRESSION";
            }
        }

        // Suggest based on problem type
        if (upper_problem.find("BINARY") != string::npos) {
            return "BINARY";
        } else if (upper_problem.find("MULTICLASS") != string::npos) {
            return "MULTICLASS";
        } else if (upper_problem.find("RANK") != string::npos) {
            return "LAMBDARANK";
        } else if (upper_problem.find("COUNT") != string::npos) {
            return has_zeros ? "TWEEDIE" : "POISSON";
        } else if (upper_problem.find("PERCENT") != string::npos) {
            return "GAMMA";
        }

        return "REGRESSION";
    }

}
