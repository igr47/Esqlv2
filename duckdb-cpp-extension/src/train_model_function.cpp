#include "include/esql/lightgbm_model.h"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "include/esql/create_model_statement.h"
#include "include/esql/algorithm_registry.hpp"
#include "include/esql/model_registry.hpp"
//#include "include/esql/lightgbm_model.h"
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_set>

namespace duckdb {

//using esql::ModelSchema;

inline string ToUpper(const string &s) {
    string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

// ============================================
// Data Extraction Helpers
// ============================================

// Extract all columns from a table
vector<string> GetAllColumns(ClientContext &context, const string &table_name) {
    auto result = context.Query("SELECT * FROM " + table_name + " LIMIT 0", QueryParameters());
    vector<string> columns;
    for (idx_t i = 0; i < result->types.size(); i++) {
        columns.push_back(result->names[i]);
    }
    return columns;
}

// Convert a Value to float
float ValueToFloat(const Value &val) {
    if (val.IsNull()) return 0.0f;

    switch (val.type().id()) {
        case LogicalTypeId::INTEGER:
        case LogicalTypeId::BIGINT:
        case LogicalTypeId::HUGEINT:
            return static_cast<float>(val.GetValue<int64_t>());
        case LogicalTypeId::DOUBLE:
        case LogicalTypeId::FLOAT:
        case LogicalTypeId::DECIMAL:
            return static_cast<float>(val.GetValue<double>());
        case LogicalTypeId::BOOLEAN:
            return val.GetValue<bool>() ? 1.0f : 0.0f;
        case LogicalTypeId::VARCHAR:
            try {
                return std::stof(val.ToString());
            } catch (...) {
                // Hash string for categorical encoding
                size_t hash = std::hash<string>{}(val.ToString());
                return static_cast<float>(hash % 1000) / 1000.0f;
            }
        default:
            return 0.0f;
    }
}

// Detect problem type from labels
std::string DetectProblemType(const std::vector<float> &labels) {
    std::unordered_set<float> unique_labels;
    bool all_integer = true;

    for (float label : labels) {
        unique_labels.insert(label);
        if (std::abs(label - std::round(label)) > 1e-6) all_integer = false;
    }

    size_t num_unique = unique_labels.size();

    if (num_unique == 2) {
        return "binary_classification";
    } else if (num_unique > 2 && num_unique < 20) {
        return "multiclass";
    } else if (num_unique <= 20) {
        return "multiclass";
    } else {
        return "regression";
    }
}

// Extract training data from source table
esql::TrainingData ExtractTrainingData(ClientContext &context,
                                       const string &table_name,
                                       const vector<string> &feature_cols,
                                       const string &target_col,
                                       const string &where_clause) {

    esql::TrainingData result;
    result.feature_names = feature_cols;
    result.label_name = target_col;

    // Build query
    string query = "SELECT ";
    for (size_t i = 0; i < feature_cols.size(); i++) {
        if (i > 0) query += ", ";
        query += feature_cols[i];
    }
    query += ", " + target_col + " FROM " + table_name;

    if (!where_clause.empty()) {
        query += " WHERE " + where_clause;
    }

    // Execute query
    auto query_result = context.Query(query, QueryParameters());
    if (!query_result || query_result->HasError()) {
        throw std::runtime_error("Failed to execute query: " + (query_result ? query_result->GetError() : "unknown error"));
    }

    // For DuckDB's MaterializedQueryResult, we need to use different methods
    auto &materialized_result = dynamic_cast<MaterializedQueryResult&>(*query_result);
    result.total_samples = materialized_result.RowCount();

	    // Get column indices
    std::unordered_map<string, idx_t> column_indices;
    for (idx_t i = 0; i < materialized_result.ColumnCount(); i++) {
        column_indices[materialized_result.names[i]] = i;
    }

    // Get target column index
    idx_t target_idx = column_indices[target_col];

    // Get feature column indices
    vector<idx_t> feature_indices;
    for (const auto &col : feature_cols) {
        feature_indices.push_back(column_indices[col]);
    }

    // Extract data - using MaterializedQueryResult methods
	for (idx_t row_idx = 0; row_idx < materialized_result.RowCount(); row_idx++) {
        vector<float> feature_row;

        for (idx_t feature_idx : feature_indices) {
            auto val = materialized_result.GetValue(row_idx, feature_idx);
            if (val.IsNull()) {
                feature_row.push_back(0.0f);
            } else {
                feature_row.push_back(ValueToFloat(val));
            }
        }

        auto label_val = materialized_result.GetValue(row_idx, target_idx);
        if (!label_val.IsNull()) {
            float label = ValueToFloat(label_val);
            result.labels.push_back(label);
            result.features.push_back(feature_row);
            result.valid_samples++;
        }
    }
    /*for (idx_t row_idx = 0; row_idx < materialized_result.RowCount(); row_idx++) {
        vector<float> feature_row;
        bool row_valid = true;

        for (const auto &col : feature_cols) {
            auto val = materialized_result.GetValue(row_idx, col);
            if (val.IsNull()) {
                feature_row.push_back(0.0f);
            } else {
                feature_row.push_back(ValueToFloat(val));
            }
        }

        auto label_val = materialized_result.GetValue(row_idx, target_col);
        if (!label_val.IsNull()) {
            float label = ValueToFloat(label_val);
            result.labels.push_back(label);
            result.features.push_back(feature_row);
            result.valid_samples++;
        }
    }*/

    return result;
}

// ============================================
// Data Preprocessing
// ============================================

esql::TrainingData ApplySampling(const esql::TrainingData &data,
                                 const string &method,
                                 float ratio,
                                 int seed) {
    if (method == "none" || ratio >= 1.0f) return data;

    esql::TrainingData sampled_data = data;

    if (method == "random") {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        vector<size_t> selected_indices;
        for (size_t i = 0; i < data.features.size(); ++i) {
            if (dist(rng) <= ratio) {
                selected_indices.push_back(i);
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

esql::TrainingData ApplyScaling(const esql::TrainingData &data,
                                const string &method) {
    if (data.features.empty() || method == "none") return data;

    esql::TrainingData scaled_data = data;
    size_t num_features = data.features[0].size();

    vector<float> means(num_features, 0.0f);
    vector<float> stds(num_features, 0.0f);
    vector<float> mins(num_features, std::numeric_limits<float>::max());
    vector<float> maxs(num_features, std::numeric_limits<float>::lowest());

    for (const auto &sample : data.features) {
        for (size_t i = 0; i < num_features; ++i) {
            means[i] += sample[i];
            mins[i] = std::min(mins[i], sample[i]);
            maxs[i] = std::max(maxs[i], sample[i]);
        }
    }

    for (size_t i = 0; i < num_features; ++i) {
        means[i] /= data.features.size();
    }

    for (const auto &sample : data.features) {
        for (size_t i = 0; i < num_features; ++i) {
            float diff = sample[i] - means[i];
            stds[i] += diff * diff;
        }
    }

    for (size_t i = 0; i < num_features; ++i) {
        stds[i] = std::sqrt(stds[i] / data.features.size());
        if (stds[i] < 1e-10f) stds[i] = 1.0f;
    }

    if (method == "STANDARD" || method == "standard") {
        for (auto &sample : scaled_data.features) {
            for (size_t i = 0; i < num_features; ++i) {
                sample[i] = (sample[i] - means[i]) / stds[i];
            }
        }
    } else if (method == "MINMAX" || method == "minmax") {
        for (auto &sample : scaled_data.features) {
            for (size_t i = 0; i < num_features; ++i) {
                float range = maxs[i] - mins[i];
                if (range < 1e-10f) range = 1.0f;
                sample[i] = (sample[i] - mins[i]) / range;
            }
        }
    }

    return scaled_data;
}

// ============================================
// Feature Selection
// ============================================

vector<size_t> SelectFeatures(const esql::TrainingData &data,
                              const string &method,
                              int max_features) {
    if (data.features.empty() || max_features <= 0) {
        return {};
    }

    size_t num_features = data.features[0].size();
    if (num_features <= static_cast<size_t>(max_features)) {
        vector<size_t> indices(num_features);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }

    vector<float> importance(num_features, 0.0f);

    if (method == "VARIANCE" || method == "AUTO" || method == "auto") {
        for (size_t i = 0; i < num_features; ++i) {
            float mean = 0.0f;
            for (const auto &sample : data.features) {
                mean += sample[i];
            }
            mean /= data.features.size();

            float variance = 0.0f;
            for (const auto &sample : data.features) {
                float diff = sample[i] - mean;
                variance += diff * diff;
            }
            variance /= data.features.size();
            importance[i] = variance;
        }
    }

    vector<size_t> indices(num_features);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return importance[a] > importance[b]; });

    size_t num_to_select = std::min(static_cast<size_t>(max_features), num_features);
    vector<size_t> selected(indices.begin(), indices.begin() + num_to_select);
    std::sort(selected.begin(), selected.end());

    return selected;
}

// ============================================
// Hyperparameter Tuning
// ============================================

struct TrialResult {
    unordered_map<string, string> parameters;
    float score;
    std::chrono::milliseconds duration;

    bool operator<(const TrialResult &other) const { return score < other.score; }
};

float EvaluateModel(esql::AdaptiveLightGBMModel &model,
                    const esql::TrainingData::SplitData &test_data,
                    const string &problem_type) {
    if (test_data.Empty()) return 0.0f;

    size_t correct = 0;

    if (problem_type.find("classification") != string::npos) {
        for (size_t i = 0; i < test_data.features.size(); ++i) {
            esql::Tensor input_tensor(test_data.features[i], {test_data.features[i].size()});
            auto pred = model.Predict(input_tensor);

            if (problem_type == "binary_classification") {
                bool pred_class = pred.data[0] > 0.5f;
                bool true_class = test_data.labels[i] > 0.5f;
                if (pred_class == true_class) correct++;
            } else {
                if (std::abs(pred.data[0] - test_data.labels[i]) < 0.5f) correct++;
            }
        }
        return static_cast<float>(correct) / test_data.features.size();
    } else {
        float mean = std::accumulate(test_data.labels.begin(), test_data.labels.end(), 0.0f) /
                     test_data.labels.size();

        float ss_total = 0.0f, ss_residual = 0.0f;
        for (size_t i = 0; i < test_data.features.size(); ++i) {
            esql::Tensor input_tensor(test_data.features[i], {test_data.features[i].size()});
            auto pred = model.Predict(input_tensor);

            ss_total += (test_data.labels[i] - mean) * (test_data.labels[i] - mean);
            ss_residual += (test_data.labels[i] - pred.data[0]) *
                           (test_data.labels[i] - pred.data[0]);
        }

        return ss_total > 0 ? 1.0f - (ss_residual / ss_total) : 0.0f;
    }
}

unordered_map<string, string> TuneHyperparameters(
    const esql::TrainingData &data,
    const string &algorithm,
    const string &problem_type,
    const TuningOptions &tuning_options,
    const std::vector<esql::FeatureDescriptor> &feature_descriptors,
    int seed) {

    if (!tuning_options.tune_hyperparameters) {
        return {};
    }

    vector<TrialResult> trials;
    std::mt19937 rng(seed);

    std::unordered_map<std::string, std::vector<std::string>> search_space;

    if (tuning_options.param_grid.empty()) {
        if (problem_type.find("classification") != string::npos) {
            search_space = {
                {"learning_rate", {"0.01", "0.05", "0.1"}},
                {"num_leaves", {"31", "63", "127"}},
                {"min_data_in_leaf", {"10", "20", "50"}},
                {"feature_fraction", {"0.7", "0.8", "0.9"}},
                {"bagging_fraction", {"0.7", "0.8", "0.9"}}
            };
        } else {
            search_space = {
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

    std::vector<std::unordered_map<std::string, std::string>> param_combinations;

    std::function<void(size_t, unordered_map<string, string>)> generate;
    generate = [&](size_t idx, unordered_map<string, string> current) {
        if (idx == search_space.size()) {
            param_combinations.push_back(current);
            return;
        }

        auto it = search_space.begin();
        std::advance(it, idx);

        for (const auto &value : it->second) {
            current[it->first] = value;
            generate(idx + 1, current);
        }
    };

    generate(0, {});

    if (param_combinations.size() > static_cast<size_t>(tuning_options.tuning_iterations)) {
        std::shuffle(param_combinations.begin(), param_combinations.end(), rng);
        param_combinations.resize(tuning_options.tuning_iterations);
    }

    // Create train/validation split from data
    size_t split_idx = data.features.size() * 4 / 5;
    esql::TrainingData::SplitData train_split, val_split;

    for (size_t i = 0; i < split_idx && i < data.features.size(); ++i) {
        train_split.features.push_back(data.features[i]);
        train_split.labels.push_back(data.labels[i]);
    }
    train_split.size = train_split.features.size();

    for (size_t i = split_idx; i < data.features.size(); ++i) {
        val_split.features.push_back(data.features[i]);
        val_split.labels.push_back(data.labels[i]);
    }
    val_split.size = val_split.features.size();

    for (const auto &params : param_combinations) {
        try {
            esql::ModelSchema model_schema;
            model_schema.algorithm = algorithm;
            model_schema.problem_type = problem_type;
            model_schema.features = feature_descriptors;

            auto model = std::make_unique<esql::AdaptiveLightGBMModel>(model_schema);

            if (!model->Train(train_split.features, train_split.labels, params)) {
                continue;
            }

            float score = EvaluateModel(*model, val_split, problem_type);
            trials.push_back({params, score, std::chrono::milliseconds(0)});

        } catch (...) {
            // Skip failed trials
        }
    }

    if (trials.empty()) return {};

    auto best_it = std::max_element(trials.begin(), trials.end(),
                                     [](const TrialResult &a, const TrialResult &b) {
                                         return a.score < b.score;
                                     });

    return best_it->parameters;
}

// ============================================
// Function State
// ============================================
struct TrainModelBindData : public TableFunctionData {
    string model_name;
    string algorithm;
    vector<pair<string, string>> features;
    vector<string> exclude_features;
    bool use_all_features = false;
    string target_column;
    string target_type;
    unordered_map<string, string> parameters;
    TrainingOptions training_options;
    TuningOptions tuning_options;
    string source_table;
    string where_clause;
    string data_sampling;
    float sampling_ratio;
    bool feature_selection;
    string feature_selection_method;
    int max_features_to_select;
    bool feature_scaling;
    string scaling_method;
    string output_table;

    bool completed = false;
    esql::ModelMetadata result_metadata;
};

// ============================================
// Main Training Function
// ============================================

static unique_ptr<FunctionData> TrainModelBind(ClientContext &context,
                                                TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types,
                                                vector<string> &names) {

    auto result = make_uniq<TrainModelBindData>();

    result->model_name = input.inputs[0].GetValue<string>();
    result->algorithm = input.inputs[1].GetValue<string>();

    // Parse features list
    auto &feature_list = ListValue::GetChildren(input.inputs[2]);
    for (idx_t i = 0; i < feature_list.size(); i++) {
        string feature_str = feature_list[i].GetValue<string>();
        size_t colon_pos = feature_str.find(':');
        if (colon_pos != string::npos) {
            string name = feature_str.substr(0, colon_pos);
            string type = feature_str.substr(colon_pos + 1);
            result->features.emplace_back(name, type);
        } else {
            result->features.emplace_back(feature_str, "AUTO");
        }
    }

    result->target_column = input.inputs[3].GetValue<string>();

    if (input.inputs.size() > 4 && input.inputs[4].type().id() != LogicalTypeId::SQLNULL) {
        result->target_type = input.inputs[4].GetValue<string>();
    }

    // Parse parameters map
    if (input.inputs.size() > 5 && !input.inputs[5].IsNull()) {
        auto &param_map = input.inputs[5];
        auto &struct_children = StructValue::GetChildren(param_map);
        if (struct_children.size() == 2) {
            auto &keys_list = struct_children[0];
            auto &values_list = struct_children[1];

            auto &keys = ListValue::GetChildren(keys_list);
            auto &values = ListValue::GetChildren(values_list);

            for (idx_t i = 0; i < keys.size(); i++) {
                string key = keys[i].ToString();
                string value = values[i].ToString();
                result->parameters[key] = value;
            }
        }
    }

    if (input.inputs.size() > 6) {
        result->source_table = input.inputs[6].GetValue<string>();
    }

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

    result->tuning_options.tune_hyperparameters = get_bool("TUNE_HYPERPARAMETERS", false);
    result->tuning_options.tuning_method = get_string("TUNING_METHOD", "grid");
    result->tuning_options.tuning_iterations = get_int("TUNING_ITERATIONS", 10);
    result->tuning_options.tuning_folds = get_int("TUNING_FOLDS", 3);
    result->tuning_options.scoring_metric = get_string("SCORING_METRIC", "auto");
    result->tuning_options.parallel_tuning = get_bool("PARALLEL_TUNING", true);
    result->tuning_options.tuning_jobs = get_int("TUNING_JOBS", -1);

    result->data_sampling = get_string("DATA_SAMPLING", "none");
    result->sampling_ratio = get_float("SAMPLING_RATIO", 1.0f);
    result->feature_selection = get_bool("FEATURE_SELECTION", false);
    result->feature_selection_method = get_string("FEATURE_SELECTION_METHOD", "auto");
    result->max_features_to_select = get_int("MAX_FEATURES_TO_SELECT", -1);
    result->feature_scaling = !get_bool("NO_FEATURE_SCALING", false);
    result->scaling_method = get_string("SCALING_METHOD", "standard");

    result->output_table = get_string("OUTPUT_TABLE", "");

    auto &registry = AlgorithmRegistry::Instance();
    if (!registry.IsAlgorithmSupported(result->algorithm)) {
        throw std::runtime_error("Unsupported algorithm: " + result->algorithm +
                                 ". Supported: " + StringUtil::Join(registry.GetSupportedAlgorithms(), ", "));
    }

    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::DOUBLE);
    return_types.push_back(LogicalType::UBIGINT);
    return_types.push_back(LogicalType::TIMESTAMP);
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::DOUBLE);
    return_types.push_back(LogicalType::VARCHAR);

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

static void TrainModelFunction(ClientContext &context,
                                TableFunctionInput &data_p,
                                DataChunk &output) {
    auto &data = (TrainModelBindData &)*data_p.bind_data;

    if (data.completed) {
        return;
    }

    // Step 1: Get all columns from source table if needed
    vector<string> feature_columns;

    if (data.use_all_features) {
        auto all_columns = GetAllColumns(context, data.source_table);
        for (const auto &col : all_columns) {
            if (col != data.target_column &&
                std::find(data.exclude_features.begin(), data.exclude_features.end(), col) == data.exclude_features.end()) {
                feature_columns.push_back(col);
            }
        }
    } else {
        for (const auto &feature : data.features) {
            feature_columns.push_back(feature.first);
        }
    }

    // Step 2: Extract training data
    auto training_data = ExtractTrainingData(context, data.source_table,
                                             feature_columns, data.target_column,
                                             data.where_clause);

    if (training_data.valid_samples < 10) {
        throw std::runtime_error("Insufficient training data: " +
                                 std::to_string(training_data.valid_samples) +
                                 " samples (minimum 10 required)");
    }

    // Step 3: Detect problem type if not specified
    string problem_type = data.target_type;
    if (problem_type.empty()) {
        problem_type = DetectProblemType(training_data.labels);
    }

    std::transform(problem_type.begin(), problem_type.end(),
                   problem_type.begin(), ::tolower);

    // Step 4: Create feature descriptors
    std::vector<esql::FeatureDescriptor> feature_descriptors;
    for (size_t i = 0; i < feature_columns.size(); ++i) {
        esql::FeatureDescriptor fd;
        fd.name = feature_columns[i];
        fd.db_column = feature_columns[i];
        fd.data_type = "float";
        fd.transformation = "direct";
        fd.required = true;
        fd.is_categorical = false;

        float sum = 0.0f;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();

        for (const auto &features : training_data.features) {
            if (i < features.size()) {
                float val = features[i];
                sum += val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }

        fd.mean_value = sum / training_data.features.size();
        fd.min_value = min_val;
        fd.max_value = max_val;

        float variance = 0.0f;
        for (const auto &features : training_data.features) {
            if (i < features.size()) {
                float diff = features[i] - fd.mean_value;
                variance += diff * diff;
            }
        }
        fd.std_value = std::sqrt(variance / training_data.features.size());
        fd.default_value = fd.mean_value;

        if (i < data.features.size() && data.features[i].second == "CATEGORICAL") {
            fd.is_categorical = true;
        }

        feature_descriptors.push_back(fd);
    }

    // Step 5: Apply data preprocessing
    esql::TrainingData processed_data = training_data;

    if (data.data_sampling != "none") {
        processed_data = ApplySampling(processed_data, data.data_sampling,
                                       data.sampling_ratio, data.training_options.seed);
    }

    if (data.feature_selection && data.max_features_to_select > 0) {
        auto selected_indices = SelectFeatures(processed_data,
                                                data.feature_selection_method,
                                                data.max_features_to_select);

        if (!selected_indices.empty()) {
            esql::TrainingData selected_data;
            selected_data.labels = processed_data.labels;
            selected_data.feature_names.clear();

            for (size_t idx : selected_indices) {
                if (idx < feature_columns.size()) {
                    selected_data.feature_names.push_back(feature_columns[idx]);
                }
            }

            for (const auto &sample : processed_data.features) {
                vector<float> selected_sample;
                for (size_t idx : selected_indices) {
                    if (idx < sample.size()) {
                        selected_sample.push_back(sample[idx]);
                    }
                }
                selected_data.features.push_back(selected_sample);
            }

            selected_data.total_samples = processed_data.total_samples;
            selected_data.valid_samples = processed_data.valid_samples;
            processed_data = selected_data;

            std::vector<esql::FeatureDescriptor> selected_descriptors;
            for (size_t idx : selected_indices) {
                if (idx < feature_descriptors.size()) {
                    selected_descriptors.push_back(feature_descriptors[idx]);
                }
            }
            feature_descriptors = selected_descriptors;
			feature_columns.clear();
			for (const auto& name : selected_data.feature_names) {
				feature_columns.push_back(name);
			}
            //feature_columns = selected_data.feature_names;
        }
    }

    if (data.feature_scaling) {
        processed_data = ApplyScaling(processed_data, data.scaling_method);
    }

    // Step 6: Prepare hyperparameters
    unordered_map<string, string> train_params = data.parameters;

    auto &registry = AlgorithmRegistry::Instance();
    const auto *algo_info = registry.GetAlgorithm(data.algorithm);

    if (algo_info) {
        for (const auto &[key, value] : algo_info->default_params) {
            if (train_params.find(key) == train_params.end()) {
                train_params[key] = value;
            }
        }

        if (problem_type == "binary_classification") {
            train_params["objective"] = "binary";
            if (train_params.find("metric") == train_params.end()) {
                train_params["metric"] = "binary_logloss,auc";
            }
        } else if (problem_type == "multiclass") {
            train_params["objective"] = "multiclass";
            std::unordered_set<float> classes(training_data.labels.begin(), training_data.labels.end());
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

    if (data.training_options.early_stopping) {
        train_params["early_stopping_round"] = std::to_string(data.training_options.early_stopping_rounds);
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

    // Step 7: Apply hyperparameter tuning if requested
    if (data.tuning_options.tune_hyperparameters) {
        auto tuned_params = TuneHyperparameters(processed_data, data.algorithm, problem_type,
                                                data.tuning_options, feature_descriptors,
                                                data.training_options.seed);

        for (const auto &[key, value] : tuned_params) {
            if (train_params.find(key) == train_params.end()) {
                train_params[key] = value;
            }
        }
    }

    // Step 8: Split data into train/validation
    size_t split_idx = processed_data.features.size() * 0.8;
    esql::TrainingData::SplitData train_data, val_data;

    for (size_t i = 0; i < processed_data.features.size(); ++i) {
        if (i < split_idx) {
            train_data.features.push_back(processed_data.features[i]);
            train_data.labels.push_back(processed_data.labels[i]);
        } else {
            val_data.features.push_back(processed_data.features[i]);
            val_data.labels.push_back(processed_data.labels[i]);
        }
    }
    train_data.size = train_data.features.size();
    val_data.size = val_data.features.size();

    // Step 9: Create schema and train model
    esql::ModelSchema model_schema;
    model_schema.model_id = data.model_name;
    model_schema.description = "Model created from table: " + data.source_table;
    model_schema.target_column = data.target_column;
    model_schema.algorithm = data.algorithm;
    model_schema.problem_type = problem_type;
    model_schema.created_at = std::chrono::system_clock::now();
    model_schema.last_updated = model_schema.created_at;
    model_schema.training_samples = train_data.size;
    model_schema.features = feature_descriptors;

    model_schema.metadata["source_table"] = data.source_table;
    model_schema.metadata["target_column"] = data.target_column;
    model_schema.metadata["algorithm"] = data.algorithm;
    model_schema.metadata["problem_type"] = problem_type;

    if (problem_type == "multiclass" && train_params.find("num_class") != train_params.end()) {
        model_schema.metadata["num_classes"] = train_params["num_class"];
    }

    auto model = std::make_unique<esql::AdaptiveLightGBMModel>(model_schema);

    bool training_success;
    if (!val_data.Empty()) {
        training_success = model->TrainWithSplits(train_data, val_data, train_params,
                                                     data.training_options.early_stopping_rounds);
    } else {
        training_success = model->Train(train_data.features, train_data.labels, train_params);
    }

    if (!training_success) {
        throw std::runtime_error("Model training failed");
    }

    // Step 10: Save model to disk
    string model_path = duckdb::ModelRegistry::GetModelsDirectory() + "/" + data.model_name + ".lgbm";
    model->Save(model_path);

    // Save schema - need to convert between esql::ModelSchema and duckdb::ModelSchema
    duckdb::ModelSchema duckdb_schema;
    duckdb_schema.model_id = model_schema.model_id;
    duckdb_schema.description = model_schema.description;
    duckdb_schema.target_column = model_schema.target_column;
    duckdb_schema.algorithm = model_schema.algorithm;
    duckdb_schema.problem_type = model_schema.problem_type;
    duckdb_schema.created_at = model_schema.created_at;
    duckdb_schema.last_updated = model_schema.last_updated;
    duckdb_schema.training_samples = model_schema.training_samples;
    duckdb_schema.accuracy = model_schema.accuracy;
    duckdb_schema.drift_score = model_schema.drift_score;
    duckdb_schema.metadata = model_schema.metadata;

    // Convert features
    for (const auto& f : model_schema.features) {
        duckdb::FeatureDescriptor df;
        df.name = f.name;
        df.db_column = f.db_column;
        df.data_type = f.data_type;
        df.transformation = f.transformation;
        df.default_value = f.default_value;
        df.required = f.required;
        df.is_categorical = f.is_categorical;
        df.categories = f.categories;
        df.min_value = f.min_value;
        df.max_value = f.max_value;
        df.mean_value = f.mean_value;
        df.std_value = f.std_value;
        duckdb_schema.features.push_back(df);
    }

    duckdb_schema.stats.total_predictions = model_schema.stats.total_predictions;
    duckdb_schema.stats.failed_predictions = model_schema.stats.failed_predictions;
    duckdb_schema.stats.avg_confidence = model_schema.stats.avg_confidence;
    duckdb_schema.stats.avg_inference_time = model_schema.stats.avg_inference_time;

    duckdb::ModelRegistry::SaveModelSchema(context, data.model_name, duckdb_schema);

    // Step 11: Create metadata for registry
    esql::ModelMetadata metadata;
    metadata.name = data.model_name;
    metadata.algorithm = data.algorithm;
    metadata.problem_type = problem_type;
    metadata.feature_names = feature_columns;
    metadata.target_column = data.target_column;
    metadata.parameters = train_params;
    metadata.created_at = model_schema.created_at;
    metadata.training_samples = train_data.size;
    metadata.accuracy = model_schema.accuracy;
    metadata.model_path = model_path;
    // Note: schema assignment would need conversion

    auto get_float = [&](const string &key, float default_val) {
        auto it = model_schema.metadata.find(key);
        if (it != model_schema.metadata.end()) {
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

    // Need to register with duckdb::ModelRegistry - convert metadata
    duckdb::ModelMetadata duckdb_metadata;
    duckdb_metadata.name = metadata.name;
    duckdb_metadata.algorithm = metadata.algorithm;
    duckdb_metadata.problem_type = metadata.problem_type;
    duckdb_metadata.feature_names = metadata.feature_names;
    duckdb_metadata.target_column = metadata.target_column;
    duckdb_metadata.parameters = metadata.parameters;
    duckdb_metadata.created_at = metadata.created_at;
    duckdb_metadata.training_samples = metadata.training_samples;
    duckdb_metadata.accuracy = metadata.accuracy;
    duckdb_metadata.precision = metadata.precision;
    duckdb_metadata.recall = metadata.recall;
    duckdb_metadata.f1_score = metadata.f1_score;
    duckdb_metadata.auc_score = metadata.auc_score;
    duckdb_metadata.r2_score = metadata.r2_score;
    duckdb_metadata.rmse = metadata.rmse;
    duckdb_metadata.mae = metadata.mae;
    duckdb_metadata.model_path = metadata.model_path;
    // duckdb_metadata.schema would need conversion

    duckdb::ModelRegistry::RegisterModel(context, data.model_name, duckdb_metadata);

    // Step 12: Create output table if specified
    if (!data.output_table.empty()) {
        string create_sql = "CREATE TABLE " + data.output_table + " AS SELECT * FROM __model_registry WHERE model_name = '" + data.model_name + "'";
        context.Query(create_sql, QueryParameters());
    }

    // Step 13: Fill output
    output.SetCardinality(1);

    output.SetValue(0, 0, Value(data.model_name));
    output.SetValue(1, 0, Value(data.algorithm));
    output.SetValue(2, 0, Value("trained"));
    output.SetValue(3, 0, Value(model_schema.accuracy));

    output.SetValue(4, 0, Value::UBIGINT(metadata.training_samples));

    output.SetValue(5, 0, Value::TIMESTAMP(Timestamp::FromEpochMicroSeconds(
        std::chrono::duration_cast<std::chrono::microseconds>(
            metadata.created_at.time_since_epoch()).count())));

    output.SetValue(6, 0, Value(model_path));
    output.SetValue(7, 0, Value(problem_type));

    float primary_metric = model_schema.accuracy;
    if (problem_type == "binary_classification") {
        primary_metric = metadata.f1_score > 0 ? metadata.f1_score : metadata.accuracy;
    } else if (problem_type == "multiclass") {
        primary_metric = metadata.f1_score > 0 ? metadata.f1_score : metadata.accuracy;
    } else {
        primary_metric = metadata.r2_score > 0 ? metadata.r2_score : metadata.accuracy;
    }
    output.SetValue(8, 0, Value(primary_metric));

    output.SetValue(9, 0, Value(StringUtil::Join(feature_columns, ", ")));

    data.completed = true;
}

void RegisterTrainModelFunction(DatabaseInstance &db) {
    TableFunction train_model("train_model",
        {
            LogicalType::VARCHAR,
            LogicalType::VARCHAR,
            LogicalType::LIST(LogicalType::VARCHAR),
            LogicalType::VARCHAR,
            LogicalType::VARCHAR,
            LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR),
            LogicalType::VARCHAR
        },
        TrainModelFunction,
        TrainModelBind
    );

    train_model.named_parameters["target_type"] = LogicalType::VARCHAR;

    Connection conn(db);
    auto &context = *conn.context;
    auto &catalog = Catalog::GetSystemCatalog(db);
    CreateTableFunctionInfo train_model_info(train_model);
    catalog.CreateTableFunction(context, train_model_info);
}

} // namespace duckdb
