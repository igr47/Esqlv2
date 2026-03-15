#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "create_model_statement.h"
#include "algorithm_registr.hpp"
#include "model_registry.hpp"
#include "lightgbm_model.h"
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace esql {
    using namespace esql::ai;

    // Training data structure
    struct TrainingData {
        std::vector<std::vector<float>> features;
        std::vector<float> labels;
        std::vector<std::string> feature_names;
        std::string lable_name;
        size_t total_samples = 0;
        size_t valid_samples = 0;

        // Split datasets
        struct SplitData {
            std::vector<std::vector<float>> features;
            std::vector<float> labels;
            size_t size = 0;

            bool Empty() const { return features.empty(); }
        };

        SplitData train;
        SplitData test;
        SplitData validation;
    };

    // Function state
    struct TrainModelBindData : public TableFunctionData {
        std::string model_name;
        std::string algorithm;
        std::vector<std::pair<std::string, std::string>> features;
        std::vector<std::string> exclude_features;
        bool use_all_features = false;
        std::string target_column;
        std::string target_type;
        std::unordered_map<std::string, std::string> parameters;
        TrainingOptions training_options;
        TuningOptions tuning_options;
        std::string source_table;
        std::string where_clause;
        std::string data_sampling;
        float sampling_ratio;
        bool feature_selection;
        std::string feature_selection_method;
        int max_features_to_select;
        bool feature_scaling;
        std::string scaling_method;
        std::string output_table;

        bool completed = false;
        ModelMetadata result_metadata;
    };

    // ====================================================================
    // Data Extraction Helpers
    // To extract the features and the labels and then convert them to floats
    // ======================================================================

    // Extract all columns from the table
    std::vector<std::string> GetAllColumns(ClientContext& context, const std::string& table_name) {
        auto result = context.db->Query("SELECT * FROM " + table_name + " LIMIT 0");
        std::vector<std::string> columns;
        for (idx i = 0; i < result->ColumnCount(); i++) {
            columns.push_back(result->GetName(i));
        }

        return columns;
    }

    // Convert to float since LIGHTGBM uses floats
    // For string we will use one hot encoding to convert them into usable format
    float ValueToFloat(const Value& value) {
        if (val.IsNull()) return 0.0f;

        switch (val.type().id()) {
            case LogicalTypeId::INTEGER:
            case LogicalTypeId::BIGINT:
            case LogicalTypeId::HUGEINT:
                return val.GetValue<int64_t>();
            case LogicalTypeId::DOUBLE:
            case LogicalTypeId::FLOAT:
            case LogicalTypeId::DECIMAL:
                return val.GetValue<double>();
            case LogicalTypeId::BOOLEAN:
                return val.GetValue<bool>() ? 1.0f : 0.0f;
            case LogicalTypeId::VARCHAR:
                try {
                    return std::stof(val.ToString());
                } catch (...) {
                    // Hash strings for categorical encoding
                    size_t hash = std::hash<std::string>{}(val.ToString());
                    return static_cast<float>(hash % 1000) / 1000.0f;
                }
            default:
                return 0.0f;
        }
    }

    // DEtect probem type from labels
    std::string DetectProblemType(const std::vector<float>& labels) {
        std::unordered_set<float> unique_labels;
        bool all_integer = true;
        bool all_non_negative = true;

        for (float label : labels) {
            unique_labels.insert(label);
            if (std::abs(label - std::round(label)) > 1e-6) all_integer = false;
            if (label < 0) all_non_negative = false;
        }

        size_t num_unique = unique_labels.size();

        if (num_unique == 2) {
            return "binary_classification";
        } else if (num_unique > 2 && num_unique < 20 && all_integers) {
            return "multiclass";
        } else {
            return "regression";
        }
    }

    // Extract training data from source table
    TrainingData ExtractTrainingData(ClientContext& context, const std::string& table_name, const std::vector<std::string>& feature_cols,
            const std::string& target_col, const std::string& where_clause) {
        TrainingData result;
        result.feature_names = feature_cols;
        result.label_name = target_col;

        // Build query
        std::string query = "SELECT";
        for (size_t i = 0; i < feature_cols.size(); i++) {
            if (i > 0) query += ", ";
            query += feature_cols[i];
        }
        query += ", " + target_col + " FROM " + table_name;

        if (!where_clause.empty()) {
            query += " WHERE " + where_clause;
        }

        // Execute query
        auto query_result = context.db->Query(query);
        result.total_samples = query_result->RowCount();

        // Extract data
        while (query_result->NextRow()) {
            std::vector<float> feature_row;
            bool row_valid = true;

            for (const auto& col : feature_cols) {
                auto val = query_result->GetValue(col);
                if (val.IsNull()) {
                    // Use 0.0 for missing values (could use meand imputation but later)
                    feature_row.push_back(0.0f);
                } else {
                    feature_row.push_back(ValueToFloat(val));
                }
            }

            auto label_val = query_result->GetValue(target_col);
            if (!label_val.IsNull()) {
                float label = ValueToFloat(label_val);
                result.labels.push_back(label);
                result.features.push_back(feature_row);
                result.valid_samples++;
            }
        }

        return results;
    }

    // =================================================================
    // DATA PROCESSING
    // Applying sampling but simplified
    // ================================================================

    TrainingData ApplySampling(const TrainingData& data, const std::string& method, float ratio, int seed) {
        if (method == "none" || ratio >= 1.0f) return data;

        TrainngData sampled_data = data;

        if (method == "random") {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            std::vector<size_t> selected_indices;
            for (size_t i = 0; i < data.features.size(); ++i) {
                if (dist(rng) <= ratio) {
                    selected_indices.push_back(i)
                }
            }

            if (selected_indices.empty()) return data;

            sampled_data.features.clear();
            sampled_data.labels.clear();
            sampled_data.valid_samples = 0;

            for (size_t idx : selected_indices) {
                if (idx < data.features.size() && idx < data.labels.size()) {
                    sampled_data.features.push_back(data.features[idx]);
                    sampled_data.labels.push_back(data.labels[idx]);
                    sampled_data.valid_samples++;
                }
            }

            sampled_data.total_samples = sampled_data.valid_samples;
        }
        return sampled_data;
    }

    TrainingData ApplyScaling(const TrainingData& data, const std::string& method) {
        if (data.features.empty() || method == "none") return data;

        TrainingData scaled_data = data;
        size_t num_features = data.features[0].size();

        // Calculate statistics
        std::vector<float> means(num_features, 0.0f);
        std::vector<float> stds(num_features, 0.0f);
        std::vector<float> mins(num_features, std::numeric_limits<float>::max());
        std::vector<float> maxs(num_features, std::numeric_limits<float>::lowest());

        // First pass: means and min/max
        for (const auto& sample : data.features) {
            for (size_t i = 0; i < num_features; i++) {
                means[i] += sample[i];
                mins[i] = std::min(mins[i], sample[i]);
                maxs[i] = std::max(maxs[i], sample[i]);
            }
        }

        for (size_t i = 0; i < num_features; ++i) {
            means[i] /= data.features.size();
        }

        // SEcond pass: standard deviations
        for (const auto& sample : data.features) {
            for (size_t = 0; i < num_features; ++i) {
                float diff = sample[i] - means[i];
                stds[i] += diff * diff;
            }
        }

        for (size_t i = 0; i < num_features; i++) {
            stds[i] = std::sqrt(stds[i] / data.features.size());
            if (stds[i] < 1e-10f) stds[i] = 1.0f;
        }

        // Apply scaling
        if (method == "STANDARD") {
            for (auto& sample : scaled_data.features) {
                for (size_t i = 0; i < num_features; i++) {
                    sample[i] = (sample[i] - means[i]) / stds[i];
                }
            }
        } else if (method == "MINMAX") {
            for (auto& sample : scaled_data.features) {
                for (size_t i = 0; i < num_features; ++i) {
                    float range = maxs[i] - mins[i];
                    if (range < 1e-10f) range = 1.0f;
                    sample[i] = (sample[i] - mins[i]) / range;
                }
            }
        }
        return scaled_data;
    }

    //===============================================================
    // FEATURE SLECTION / Simplified version will come later
    // =============================================================

    std::vector<size_t> SelectedFeatures(const TrainingData& data, std::string& method, int max_features) {
        if (data.features.empty() || max_features <= 0) {
            return {};
        }

        size_t num_features = data.features[0].size();
        if (num_fetures <= static_cast<size_t>(max_features)) {
            std::vector<size_t> indices(num_features);
            std::iota(indices.begin, indices.end(), 0);
            return indices;
        }

        std::vector<float> importance(num_features, 0.0f);

        // Simplified version uses variance
        if (method == "VARIANCE" || method == "AUTO") {
            for (size_t i = 0; i < num_features; i++) {
                float mean = 0.0f;
                for (const auto& sample : data.features) {
                    mean += sample[i];
                }
                mean /= data.features.size();

                float variance = 0.0f;
                for (const auto& sample : data.features) {
                    float diff = sample[i] - mean;
                    variance += diff * diff;
                }
                variance /= data.features.size();
                importance[i] = variance;
            }
        }

        // Select top features
        std::vector<size_t> indices(num_features);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort (indices.begin(), indices.end(), [&](size_t a, size_t b) { return importance[a] > importance [b]; });
        size_t num_to_select = std::min(static_cast<size_t>(max_features), num_features);
        std::vector<size_t> selected(indices.begin(), indices.begin() + num_to_select);
        std::sort(selected.begin(), selected.end());

        return selected;
    }

    // ==================================================================
    // Hyperparameter Tuning
    // ==================================================================
    struct TrialResult {
        std::unordered_map<std::string, std::string> parameters;
        float score;
        std::chrono::milliseconds duration;

        bool operator<(const TrialResult& other) const { return score < other.score; }
    };

    float EvaluateModel(AdaptiveLightGBMModel& model, const TrainigData::SplitData& test_data, const std::string& problem_type) {
        if (test_data.Empty()) return 0.0f;

        size_t correct = 0;

        if (problem_type.find("classification") != std::string::npos) {
            for (size_t i = 0; i < test_data.features.size(); i++) {
                Tensor input_tensor(test_data.features[i], {test_data.features[i].size()});
                auto pred = model.predict(input_tensor);

                if (problem_type == "binary_classication") {
                    bool pred_class = pred.data[0] > 0.05f;
                    bool true_class = test_data.labels[i] > 0.5f;
                    if (pred_class == true_class) correct++;
                } else {
                    // Multiclass
                    if (std::abs(pred.data[0] - test_data,labels[i]) < 0.5f) correct++;
                }
            }
            return static_cast<float>(correct) / test_data.features.size();
        } else {
            // REgression we use R2
            float mean = std::accumulate(test_data.labels.begin(), test_data.lables.end(),0.0f) / test_data.labels.size();
            float ss_total = 0.0f, ss_residual = 0.0f;
            for (size_t i = 0; i < test_data.features.size(); ++i) {
                Tensor input_tensor(test_data.features[i], { test_daa.features[i].size()});
                auto pred = model.predict(input_tensor);

                ss_total += (test_data.labels[i] - mean) * (test_data.labels[i] - mean);

                ss_residual += (test_data.labels[i] - pred.data[0]) *
                    (test_data.labels[i] - pred.data[0]);
            }

            return ss_total > 0 ? 1.0f - (ss_residual / ss_total) : 0.0f;
        }
    }

    std::unordered_map<std::string, std::string> TuneHyperparameters(const TrainingData& data, const std::string& algirithm,
            const std::string& problem_type, const TuningOptions& tuning_options, const std::vector<FeatureDescribtor>& feature_describtors,
            int seed) {
        if (!tuning_options.tune_hyperparameters) {
            return {};
        }

        // Simple grid search implementation
        std::vector<TrailResult> trials;
        std::mt19937 rng(seed);

        std::unordered_map<std::string, std::vector<std::string>> search_space;

        if (tuning_options.param_grid.empty()) {
            // DEfault search space
            if (proble_type.find("classification") != std::string::npos) {
                search_space = {
                    {"learning_rate", {"0.01", "0.05", "0.1"}},
                    {"num_leaves", {"31", "63", "127"}},
                    {"min_data_in_leaf", {"10", "20", "50"}},
                    {"feature_fraction", {"0.7", "0.8", "0.9"}},
                    {"bagging_fraction", {"0.7", "0.8", "0.9"}}
                };
            } else {
                serach_space = {
                    {"learning_rate", {"0.01", "0.05", "0.1"}},
                    {"num_leaves", {"31", "63", "127"}},
                    {"min_data_in_leaf", {"5", "10", "20"}},
                    {"feature_fraction", {"0.8", "0.9", "1.0"}},
                    {"bagging_fraction", {"0.8", "0.9", "1.0"}}
                };
            }
        } else {
            search_space = tuning_options.param_grid;
        }

        // Generate all combinations
        std::vector<std::unordered_map<std::string, std::string>> param_combinations;

        std::function<void(size_t, std::unordered_map<std::string, std::string>)> generate =
            [&](size_t idx, unordered_map<std::string, std::string> current) {
                if (idx == search_space.size()) {
                    param_combinations.push_back(current);
                    return
                }

                auto it = search_space.begin();
                std::advance(it, idx);

                for (const auto& value : it->second) {
                    current[it->first] = value;
                    generate(idx + 1, current);
                }
            };
        generate(0, {});

        // Limit number of combinations
        if (param_combinations.size() > static_cast<size_t>(tuning_options.tuning_iterations)) {
            // Rnadom sample
            std::shuffle(param_combinations.begin(), param_combinations.end(), rng);
            param_combinations.resize(tuning_options.tuning_itterations);
        }

        // Evaluate each combinations
        for (const auto& params : param_combinations) {
            try {
                // Create model schema
                ModelSchema schema;
                schema.algorithm = algorithm;
                schema.problem_type = problem_type;
                schema.features = feature_descriptors;

                auto model = std::make_unique<AdaptiveLightGBMModel>(schema);

                // TRain on training data
                if (!model->train(data.features, data.labels, params)) {
                    continue;
                }

                // Evaluate on validation split ( 80/20 )
                size_t split_idx = data.features.size() * 4 / 5;
                TrainingData::SplitData val_data;

                for (size_t i = split_idx; i < data.features.size(); ++i) {
                    val_data.features.push_back(data.features[i]);
                    val_data.labels.push_back(data.labels[i]);
                }
                val_data.size = val_data.features.size();

                float score = EvaluateModel (*model, val_data, problem_type);

                trials.push_back({params, score, std::chrono::milliseconds(0)});
            } catch (...) {
                // Skip failed trials
            }
        }

        if (trails.empty()) return {}

        // Find best trail
        auto best_it = std::max_element(trials.begin(), trails.end(), [](const TrailResult& a, const TrialResult& b) {
                return a.score < b.score;
                });
        return best_it->parameters;
     }

    // Main Traing functions

    static std::unique_ptr<FunctionData> TrainModelBind(ClientContext& context, TableFunctionBindInput& input,
            std::vector<LogicalType>& return_types, std::vector<std::string>& names) {
        auto result = std::make_unique<TranModelBindData>();

        // Parse inputs in order from the function call
        result->model_name = input.inputs[0].GetValue<std::string>();
        result->algorithm = input.inputs[1].GetValue<string>();

        // Parse feature list (list of strings with format "name:type")
        auto &feature_list = input.inputs[2].GetValue<List>();
        for (idx_t i = 0; i < feature_list.size(); i++) {
            std::string feature_str = feature_list[i].GetValue<string>();
            size_t colon_pos = feature_str.find(':');
            if (colon_pos != std::string::npos) {
                std::string name = feature_str.substr(0, colon_pos);
                std::string type = feature_str.substr(colon_pos + 1);
                result->features.emplace_back(name, type);
            } else {
                result->features.smplace_back(feature_str, "AUTO");
            }
        }

        // Parse target column
        result->target_column = input.inputs[3].GetValue<string>();

        // Parse target types if provided (may be empty)
        if (input.inputs.size() > 4 && input.inputs[4].type().id() != LogicalType::SQLNULL) {
            result.target_type = input.inputs[4].GetValue<string();
        }

        // Parse parameters
        if (input.inputs.size() > 5 && input.inputs[5].type().id() != LogicalTypeId::SQLNULL) {
            auto &param_map = input.inputs[5].GetValue<Map>();
            auto &keys = param_map.GetValue<List>();
            auto &values = param_map.GetValue<List>();

            for (idx_t i = 0; i < keys.size(); i++) {
                string key = keys[i].GetValue<string>();
                string value = values[i].GetValue<string>();
                result->parameters[key] = value;
            }
        }

        // Parse source table
        if (input.inputs.size() > 6) {
            result->source_table = input.inputs[6].GetValue<string>();
        }

        // Parse options from parameters
        auto get_bool = [&](const string &key, bool default_val) {
            auto it = result->parameters.find(ToUpper(key));
            if (it == result->parameters.end()) return default_val;
            string val = ToUpper(it->second);
            return val == "TRUE" || val == "1" || val == "YES";
        };

        auto get_int = [&](const string &key, int default_val) {
            auto it = result->parameters.find(ToUpper(key));
            if (it == result->parameters.end()) return default_val;
            try { return std::stoi(it->second); } catch (...) { return default_val; }
        };

        auto get_float = [&](const string &key, float default_val) {
            auto it = result->parameters.find(ToUpper(key));
            if (it == result->parameters.end()) return default_val;
            try { return std::stof(it->second); } catch (...) { return default_val; }
        };

        auto get_string = [&](const string &key, const string &default_val) {
            auto it = result->parameters.find(ToUpper(key));
            return it == result->parameters.end() ? default_val : it->second;
        };

        // Parse training options
        result->training_options.cross_validation = get_bool("CROSS_VALIDATION", false);
        result->training_options.cv_folds = get_int("CV_FOLDS", 5);
        result->training_options.early_stopping = get_bool("EARLY_STOPPING", true);
        result->training_options.early_stopping_rounds = get_int("EARLY_STOPPING_ROUNDS", 10);
        result->training_options.validation_table = get_string("VALIDATION_TABLE", "");
        result->training_options.validation_split = get_float("VALIDATION_SPLIT", 0.2f);
        result->training_options.use_gpu = get_bool("USE_GPU", false);
        result->training_options.device_type = get_bool("USE_GPU", false) ? "gpu" : "cpu";
        result->training_options.num_threads = get_int("NUM_THREADS", -1);
        result->training_options.metric = get_string("METRIC", "auto");
        result->training_options.boosting_type = get_string("BOOSTING", "gbdt");
        result->training_options.seed = get_int("SEED", 42);
        result->training_options.deterministic = get_bool("DETERMINISTIC", true);

        // Parse tuning options
        result->tuning_options.tune_hyperparameters = get_bool("TUNE_HYPERPARAMETERS", false);
        result->tuning_options.tuning_method = get_string("TUNING_METHOD", "grid");
        result->tuning_options.tuning_iterations = get_int("TUNING_ITERATIONS", 10);
        result->tuning_options.tuning_folds = get_int("TUNING_FOLDS", 3);
        result->tuning_options.scoring_metric = get_string("SCORING_METRIC", "auto");
        result->tuning_options.parallel_tuning = get_bool("PARALLEL_TUNING", true);
        result->tuning_options.tuning_jobs = get_int("TUNING_JOBS", -1);

        // Parse data preprocessing options
        result->data_sampling = get_string("DATA_SAMPLING", "none");
        result->sampling_ratio = get_float("SAMPLING_RATIO", 1.0f);
        result->feature_selection = get_bool("FEATURE_SELECTION", false);
        result->feature_selection_method = get_string("FEATURE_SELECTION_METHOD", "auto");
        result->max_features_to_select = get_int("MAX_FEATURES_TO_SELECT", -1);
        result->feature_scaling = !get_bool("NO_FEATURE_SCALING", false);
        result->scaling_method = get_string("SCALING_METHOD", "standard");

        // Parse output table
        result->output_table = get_string("OUTPUT_TABLE", "");

        // Validate algorithm
        auto &registry = AlgorithmRegistry::Instance();
        if (!registry.IsAlgorithmSupported(result->algorithm)) {
            throw std::runtime_error("Unsupported algorithm: " + result->algorithm +
                                    ". Supported: " + StringUtil::Join(registry.GetSupportedAlgorithms(), ", "));
        }

        // Define output schema
        return_types.push_back(LogicalType::VARCHAR);   // model_name
        return_types.push_back(LogicalType::VARCHAR);   // algorithm
        return_types.push_back(LogicalType::VARCHAR);   // status
        return_types.push_back(LogicalType::DOUBLE);    // accuracy
        return_types.push_back(LogicalType::UBIGINT);   // training_samples
        return_types.push_back(LogicalType::TIMESTAMP); // created_at
        return_types.push_back(LogicalType::VARCHAR);   // model_path
        return_types.push_back(LogicalType::VARCHAR);   // problem_type
        return_types.push_back(LogicalType::DOUBLE);    // precision/recall/etc
        return_types.push_back(LogicalType::VARCHAR);   // feature_list

        names.push_back("model_name");
        names.push_back("algorithm");
        names.push_back("status");
        names.push_back("accuracy");
        names.push_back("training_samples");
        names.push_back("created_at");
        names.push_back("model_path");
        names.push_back("problem_type");
        names.push_back("primary_metric");
        names.push_back("features");

        return std::move(result);
    }

    static void TrainModelFunction(ClientContext& context, TableFunctionInput& data_p, DataChunck& output) {
        auto& data = (TrainModelBindData & )*data_p.bind_data;

        if (data.completed) {
            return;
        }

        // Step 1: Get all columns from the source table if needed
        std::vector<std::string> feature_columns;

        if (data.use_all_features) {
            auto all_columns = GetAllColumns(context, data.source_table);
            for (const auto& col : all_columns) {
                if (col != data.target_column && std::find(data.exclude_features.begin(), data.exclude_features.end(), col) ==
                        data.exclude_features.end()) {
                    feature_columns.push_back(col);
                }
            }
        } else {
            for (const auto& feature : data.features) {
                feature_columns.push_back(feature.first);
            }
        }

        // Step 2: Extract training data
        auto training_data = ExtractTrainingData(context, data.source_table, feature_columns, data.target_column, data.where_clause);

        if (training_data.valid_samples < 10) {
            throw std::runtime_error("Isufficinet training data: " + std::to_string(training_data.valid_samples) + " samples (Minimun 10 required)");
        }

        // Step 3: Detect problem type if not specified
        std::string problem_type = data.target_type;
        if (problem_type.empty()){
            problem_type = DetectProblemType(training_data.labels);
        }

        // Convert to lowe case for internal use
        std::transform(problem_type.begin(), proble_type.end(), ::tolower);

        // Step 4: create feaure describtors
        std::vector<FeatureDescribtor> feature_describtors;
        for (size_t i = 0; i < feature_columns.size(); ++i) {
            FeatureDescribtor fd;
            fd.name = feature_columns[i];
            fd.db_column = feature_columns[i];
            fd_data_type = "float";
            fd.transformation = "direct";
            fd.required = true;
            fd.is_categorical = false;

            // Calculate statistics
            float sum = 0.0f;
            float min_val = std::numeric_limits<float>::max();
            float max_value = std::numeric_linits<float>::lowest();

            for (const auto& features : training_data.features) {
                float val = features[i];
                sum += val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }

            fd.mean_value = sum / training_data.features.size();
            fd.min_value = min_val;
            fd.max_value = max_val;

            // Calculate standard deviation
            float variance = 0.0f;
            for (const auto& features : training_data.features) {
                float diff = features[i] - fd.mean_value;
                variance += diff * diff;
            }

            fd.std_value = std::sqrt(variance / training_data.features.size());
            fd.default_value = fd.mean_value;

            // Check if cateorical based on feature type or cardinality
            if (i < data.features.size() && data.features[i].second == "CATEGORICAL") {
                fd.is_categorical = true;
            }
            feature_describtors.psu_back(fd);
        }

        // Step 5: Apply data processing
        TrainingData processed_data = training_data;

        // Samplng
        if (data.data_sampling != "none") {
            processed_data = ApplySampling(processed_data, data.data_sampling,data.sampling_ratio, data.training_options.seed);
        }

        // Feature selection
        if (data.feature_selection && data.max_features_to_select > 0) {
            auto selected_indices = SelectFeatures(processed_data, data.feature_selection_method, data.max_features_to_select);

            if(!selected_indices.empty()) {
                TrainingData selected_data;
                selected_data.labels = processed_data.labels;
                selected_data.feature_names.clear();

                for (size_t idx : selected_indices) {
                    if (idx < feature_columns.size()) {
                        selected_data.feature_names.push_back(feature_columns[idx]);
                    }
                }

                for (const auto& sample : precessed_data.features) {
                    std::vector<float> selected_sample;
                    for (size_t idx : sleceted_indices) {
                        if (idx < sample.size()) {
                            selected_sample.push_back(sample[idx]);
                        }
                    }
                    selected_data.features.push_back(selected_sample);
                }

                selected_data.total_samples = processed_data.total_samples;
                selected_data.valid_samples = processed_data.valid_samples;
                processed_data = selected_data;

                // Update feature describtors
                std::vector<FeatureDescribtor> selected_describtors;
                for (size_t idx : selected_indices) {
                    if (idx < feature_describtors.size()) {
                        selected_describtors.push_back(feature_describtors[idx]);
                    }
                }
                feature_describtors = selscted_describtors;
            }
        }

        // Feature scaling
        if (data.feature_scaling) {
            processed_data = ApplyScaling(processed_data, data.scaling_method);
        }

        // Step 6: Split data into train validation
        TrainingData::SplitData train_data, val_data;

        size_t split_idx = processed_data.features.size() * 0.8;
        for (size_t i = 0; i < processed_data.features.size(); ++i) {
            if (i < split_idx) {
                train_data.features.push_back(processed_data.features[i]);
                train_data.lables.push_back(processed_data.labels[i]);
            } else {
                val_data.features.push_back(processed_data.features[i]);
                val_data.labels.push_back(processed_data.labels[i]);
            }
        }
        train_data.size = train_data.features.size();
        val_data.size = val_data.features.size();

        //Step 7: prepare hyperparameters
        std::unordered_map<std::string, std::string> train_params = data.parameters;

        // Set algorithm-specific defaults from registry
        auto &registry = AlgorithmRegistry::Instance();
        const auto *algo_info = registry.GetAlgorithm(data.algorithm);

        if (algo_info) {
            for (const auto &[key, value] : algo_info->deault_params) {
                if (train_params.find(key) == train_params.end()) {
                    train_params[key] = value;
                }
            }

            // Set objectie based on the problem type
            if (problem_type == "binary_classification") {
                train_params["objective"] = "binary";
                if (train_params.find("metric") == train_params.end()) {
                    train_params["metric"] = "binary_logloss,auc";
                }
            } else if (problem_type == "multiclass") {
                train_params["objective"] = "multiclass";
                // Count unique classes
                std::unordered_set<float> classes(train_data.labels.begin(),train_data.labeld.end());
                train_params["num_class"] = std::to_string(classes.size());
                if (train_params.find("metric") == train_params.end()) {
                    train_params["metric"] = "multi_logloss,auc_mu";
                }
            } else {
                train_params["objective"] = algo_info->lightgbm_objective;
                if (train_params.find("metric") == train_params.end()) {
                    train_params["metric"] = "rmse,mae";
                }
            }
        }

        // Apply training options
        if (data.training_options.early_stopping) {
            train_params["early_stopping_round"] = std::to_string(data.traing_options.early_stopping_rounds);
        }

        if (data.training_options.use_gpu) {
            train_params["device"] = "gpu";
        }

        if (data.training_options.num_threads > 0) {
            train_params["num_threads"] = std::to_string(data.training_options.num_threads);
        }

        if (data.training_options.deterministic) {
            train_params["seed"] = std::to_string(data.training_options.seed);
            train_params["deterministic"] = "true";
        }

        // Step 8: Apply hyperparameter tuning if requested
        if (data.tuning_options.tune_hyperparameters) {
            auto tuned_params = TuneHyperparameters(train_data, data.algorithm, problem_type, data.tuning_options,
                    feature_describtors, data.training_options.seed);
            //Merge tune parameters, but preserve user-specified ones
            for (const auto& [key, value] : tuned_params) {
                if (train_params.find(key) == train_params.end()) {
                    train_params[key] = value;
                }
            }
        }

        // Step 9: Create schema and train model
        ModelSchema schema;
        schema.model_id = data.model_name;
        sceham.describtion = "Model created from table: " + data.source_table;
        schema.target_column = data.target_column;
        schema.algorithm = data.algorithm;
        schema.problem_type = probelm_type;
        schema.created_at = std::chrono::system_clock::now();
        schema.last_updated = schema.created_at;
        schema.training_samples = train_data.size;
        schema.features = feature_describtors;

        // Store metadata
        schema.metadata["source_table"] = data.source_table;
        schema.metadata["target_column"] = data.target_column;
        schema.metadata["algorithm"] = data.algorithm;
        schema.metadata["problem_type"] = probelm_type;

        if (problem_tpe == "multiclass" && train_params.find("num_class") != train_params.end()) {
            schema.metadata["num_classes"] = train_params["num_class"];
        }

        // Create model
        auto model = make_unique<AdaptiveLightGBMModel>(schema);

        // Train with validation if availble
        bool training_success;
        if (!val_data.Empty()) {
            // Convert to split data for training
            TrainingData::SplitData train_split, val_split;
            train_split.features = train_data.features;
            train_split.labels = train_data.labels;
            train_split.size = train_data.size;

            val_split.features = val_data.features;
            val_split.labels = val_data.labels;
            val_split.size = val_data.size;

            training_success = model->train_with_splits(train_split, val_split, trainng_params, data.training_options.ealy_stopping_rounds);
        } else {
            training_success = model->train(train_data.features, train_data.labels, train_params);
        }

        if (!training_success) {
            throw std::runtime_error("Model training failed");
        }

        // Step 10: Save model to disk
        std::string model_path = ModelRegistry::GetModelsDirectory() + "/" + data.model_name + ".lgbm";
        model->save(model_path);

        // Save schema
        ModelRegistry::SaveModelSchema(data.model_name, schema);

        // Step 11: Create metadata for registry
        ModelMetadata metadata;
        metadata.name = data.model_name;
        metadata.algorithm = data.algorithm;
        metadata.problem_type = problem_type;
        metadata.feature_names = feature_columns;
        metadata.target_column = data.target_column;
        metadata.parameters = train_params;
        metadata.created_at = schema.created_at;
        metadata.training_samples = train_data.size;
        metadata.accuracy = schema.accuracy;
        metadata.model_path = model_path;
        metadata.schema = schema;

        // Get metrics from schema
        auto get_float = [&](const string &key, float default_val) {
            auto it = schema.metadata.find(key);
            if (it != schema.metadata.end()) {
                try { return std::stof(it->second); } catch (...) { return default_val; }
            }
            return default_val;
        };

        if (problem_type == "binary_classification") {
            metadata.precision = get_float("precision", 0.0f);
            metadata.recall = get_float("recall", 0.0f);
            metadata.f1_score = get_float("f1_score", 0.0f);
            metadata.auc_score = get_float("auc_score", 0.0f);
        } else if (problem_type == "multiclass") {
            metadata.precision = get_float("macro_precision", 0.0f);
            metadata.recall = get_float("macro_recall", 0.0f);
            metadata.f1_score = get_float("macro_f1", 0.0f);
        } else {
            metadata.r2_score = get_float("r2_score", 0.0f);
            metadata.rmse = get_float("rmse", 0.0f);
            metadata.mae = get_float("mae", 0.0f);
        }

        // Register model
        ModelRegistry::RegisterModel(data.model_name, metadata);

        // Step 12: Create output table if specified
        if (!data.output_table.empty()) {
            string create_sql = "CREATE TABLE " + data.output_table + " AS SELECT * FROM __model_registry WHERE model_name = '" + data.model_name + "'";
            context.db->Query(create_sql);
        }

        // Step 13: Fill output
        output.SetCardinality(1);

        output.SetValue(0, 0, Value(data.model_name));                     // model_name
        output.SetValue(1, 0, Value(data.algorithm));                      // algorithm
        output.SetValue(2, 0, Value("trained"));                           // status
        output.SetValue(3, 0, Value(schema.accuracy));                     // accuracy

        output.SetValue(4, 0, Value::UBIGINT(metadata.training_samples));  // training_samples

        output.SetValue(5, 0, Value::TIMESTAMP(Timestamp::FromEpochMicroSeconds(
            std::chrono::duration_cast<std::chrono::microseconds>(
                metadata.created_at.time_since_epoch()).count())));        // created_at

        output.SetValue(6, 0, Value(model_path));                           // model_path
        output.SetValue(7, 0, Value(problem_type));                         // problem_type

        // Primary metric
        float primary_metric = schema.accuracy;
        if (problem_type == "binary_classification") {
            primary_metric = metadata.f1_score > 0 ? metadata.f1_score : metadata.accuracy;
        } else if (problem_type == "multiclass") {
            primary_metric = metadata.f1_score > 0 ? metadata.f1_score : metadata.accuracy;
        } else {
            primary_metric = metadata.r2_score > 0 ? metadata.r2_score : metadata.accuracy;
        }
        output.SetValue(8, 0, Value(primary_metric));                       // primary_metric

        // Features list
        output.SetValue(9, 0, Value(StringUtil::Join(feature_columns, ", "))); // features

        data.completed = true;
    }

    void RegisterTrainModelFunction(DatabaseInstance& db) {
        TableFunction train_model("train_model",
            {
                LogicalType::VARCHAR, // Model_name
                LogicalType::VARCHAR, // algorithm
                LogicalType::LIST(LogicalType::VARCHAR), // features
                LogicalType::VARCHAR, // tARGET COLUMN
                LogicalType::VARCHAR, // target_type(optional)
                LogicalType::Map(LogicalType::VARCHAR, LOgicalType::VARCHAR), // params
                LogicalType::VARCHAR // source_table
            },
            TrainModelFunction,
            TrainModelBind
        );

        // Make some parameters optonal
        train_model.named_parameters["target_type"] = LogicalType::VARCHAR;

        ExtensionUtil::RgisterFunction(db, train_model);
    }

} // namespace esql
