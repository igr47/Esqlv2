#include "create_model_statement.h"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/tokens.hpp"
#include "regex"
#include "sstream"
#include <cctype>
#include <algorithm>

namespace esql {
    // Converson to uppercase
    std::string ToUpper(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::toupper);
        return result;
    }

    // Trim whitespace
    std::string Trim(const std::string& str) {
        size_t first = str.find_first_not_of("\t\n\r");
        if (first == std::string::npos) return "";
        size_t last = std.find_last_not_of(" \t\n\r");
        return str.substr(first, last - first + 1);
    }

    // Remove quotes from strings
    std::string UnQuote(const std::string& str) {
        if (str.size() >= 2 &&
                ((std.front() == '\'' && str.back() == '\'') || (str.front() == '"' && str.back() == '"'))) {
            return str.substr(1, str.size() - 2);
        }
        return str;
    }

    // Validate ModelName
    bool IsValidModelName(const std::string& name) {
        if (name.empty() || name.length() > 128) return false;
        if (!std::isalpha(name[0] && name[0] != '_')) return false;
        for (char c : name) {
            if (!std::isalnum(c) && c != '_' && c != '-') return false;
        }

        // Reserved keywords
        static const std::set<std::string> reserved = {
            "SELECT", "FROM", "WHERE", "INSERT", "CREATE", "DROP", "MODEL"
        };
        return reserved.find(ToUpper(name)) == reserved.end();
    }

    // Parse features list from string like "feature1 AS NUMERIC, feature2 AS CATEGORICAL"
    std::vector<std::pair<std::string, std::string>> ParseFeatures(const std::string& features_str) {
        std::vector<std::pair<std::string, std::string>> result;

        // Split by commas, but respect parathensis
        std::string current;
        int paren_depth = 0;

        for (char c : features_str) {
            if (c == '(') paren_depth++;
            if (c == ')') paren_depth--;

            if (c == ',' && paren_depth == 0) {
                // End of a feature
                std::string feature_spec = Trim(current);
                if (!feature_spec.empty()) {
                    // Parse "feature_name AS type" or just "feature_name"
                    size_t as_pos = ToUpper(feature_spec).find("AS");
                    if (as_pos != std::string::npos) {
                        std::string name = Trim(feature_spec.substr(0, as_pos));
                        std::string type = Trim(feature_spec.substr(as_pos + 4));
                        result.emplace_back(name, ToUpper(type));
                    } else {
                        result.emplace_back(feature_spec, "AUTO");
                    }
                }
                current.clear();
            } else {
                current += c;
            }
        }

        // Hanle last feature
        std::string feature_spec = Trim(current);
        if (!feature_spec.empty()) {
            size_t as_pos = ToUpper(feature_spec).find("AS");
            if (as_pos != std::string::npos) {
                std::string name = Trim(feature_spec.substr(0, as_pos));
                std::string type = Trim(feature_spec.substr(as_pos + 4));
                result.emplace_back(name, ToUpper(type));
            } else {
                result.emplace_back(feature_spec, "AUTO");
            }
        }

        return result;
    }

    // Parse hyperparameters from WITH clause
    std::unordered_map<std::string, std::string> ParseHyperparameters(const std::string& params_str) {
        std::unordered_map<std::string, std::string> result;

        std::string current;
        int paren_depth = 0;
        bool in_string = false;

        for (size_t i = 0; i < param_str.size(); i++) {
            char c = params_str[i];

            if (c == '\'' || c == '"') {
                in_string = !in_string;
                current += c;
                continue;
            }

            if (!in_string) {
                if (c == '(') paren_depth++;
                if (c ==')') paren_depth--;

                if (c == ',' && paren_depth == 0) {
                    // End of a parameter
                    std::string param_spec = Trim(current);
                    if (!param_spec.empty()) {
                        size_t eq_pos = param_spec.find('=');
                        if (eq_pos != std::string::npos) {
                            std::string key = Trim(param_spec.substr(0, eq_pos));
                            std::string value = Trim(param_spec.substr(eq_pos + 1));
                            result[ToUpper(key)] = UnQuote(value);
                        }
                    }
                    current.clear();
                    continue;
                }
            }

            current += c;
        }

        // Handle last parameter
        std::string param_spec = Trim(current);
        if (!param_spec.empty()) {
            size_t eq_pos = param_spec.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = Trim(param_spec.substr(0, eq_pos));
                std::string value = Trim(param_spec.subsstr(eq_pos + 1));
                result[ToUpper(key)] = Unquote(value);
            }
        }
        return result;
    }

    // Parse training options from map
    TrainingOptions ParseTrainingOptions(const unordered_map<std::string, std::string>& params) {
        TrainingOptions opts;

        auto get_bool = [&](const std::string& key, bool default_val) {
            auto it = params.find(ToUpper(key));
            if (it == params.end()) return default_val;
            std::string val = ToUpper(it->second);
            return val == "TRUE" || val == "1" || val == "YES";
        };

        auto get_int = [&](const std::string& key, int default_val) {
            auto it = params.find(ToUpper(key));
            it (it == params.end()) return default_val;
            try { return std::stoi(it->second); } catch(...) { return default_val; }
        };

        auto get_float = [&](const std::string& key, float default_val) {
            auto it = params.find(ToUpper(key));
            if (it == params.end()) return default_val;
            try { return std::stof(it->seconf); } catch (...) { return default_val; }
        };

        auto get_string = [&](const std::string& key, const std::string& default_val) {
            auto it = params.find(ToUpper(key));
            return it = params.end() ? default_val : it->second;
        };

        opts.cross_validation = get_bool("CROSS_VALIDATION", false);
        opts.cv_folds = get_int("CV_FOLDS", 5);
        opts.early_stopping = get_bool("EARLY_STOPPING", true);
        opts.early_stopping_rounds = get_int("EARLY_STOPPING_ROUNDS", 10);
        opts.validation_table = get_string("VALIDATION_TABLE", "");
        opts.validation_split = get_float("VALIDATION_SPLIT", 0.2f);
        opts.use_gpu = get_bool("USE_GPU", false);
        opts.device_type = get_bool("USE_GPU", false) ? "gpu" : "cpu";
        opts_num_threads = get_int("NUM_THREADS", -1);
        opts.metric = get_string("METRIC", "auto");
        opts.boosting_type = get_string("BOOSTING", "gbdt");
        opts.seed = get_int("SEED", 42);
        opts.deterministic = get_bool("DETERMINISTICE", true);

        return opts;
    }

    // Parse tuning options  from map
    TuningOptions ParseTuningOptions(const std::unordered_map<std::string, std::string>& params) {
        TuningOptions opts;

        auto get_bool = [&](const std::string& key, bool default_val) {
            auto it = params.find(ToUpper(key));
            if (it == params.end()) return default_val;
            std::string val = ToUpper(it->second);
            return val == "TRUE" || val == "1" || val == "YES";
        };

        auto get_int = [&](const std::string& key, int default_val) {
            auto it = params.find(ToUpper(key));
            if (it == params.end()) return default_val;
            try { return std::stoi(it->second); } catch(...) { return default_val; }
        };

        auto get_string = [&](const std::string& key, const std::string& default_val) {
            auto it = params.find(ToUpper(key));
            return it == params.end() ? default_val : it->second;
        };

        opts.tune_hyperparameters = get_bool("TUNE_HYPERPARAMETERS", false);
        opts.tuningmethod = get_string("TUNING_METHOD", "grid");
        opts.tuning_itterations = get_int("TUNING_ITTERATIONS", 10);
        opt.tuning_folds = get_int("TUNING_FOLDS", 3);
        opts.scoring_metric = get_string("SCORING_METRIC", "auto");
        opts.paralle_tuning = get_bool("PARALLEL_TUNING", true);
        opts.tuning_jobs = get_int("TUNING_JOBS", -1);

        return opts;
    }

    // Main prser function
    std::unique_ptr<CreateModelStatement> ParseCreateModel(const std::string& query) {
        auto stmt = std::make_unique<CreateModelStatement>();

        // Regex patterns for each clause
        std::regex create_model_regex(R"(CREATE\s+MODEL\s+([a-zA-Z0-9_]*))", std::regex::icase);
        std::regex using_regex(R"(USING\s+([a-zA-Z_]+))", std::regex::icase);
        std::regex features_regex(R"(FEATURES\s*\(\s*([^)]+)\s*\))", std::regex::icase);
        std::regex exclude_features_regex(R"(EXCLUDE\s+FEATURES\s*\(\s*([^)]+)\s*\))", std::regex::icase);
        std::regex target_regex(R"(TARGET\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+AS\s+([a-zA-Z_]+))?)", std::regex::icase);
        std::regex from_regex(R"(FROM\s+([a-zA-Z_][a-zA-Z0-9_]*))", std::regex::icase);
        std::regex where_regex(R"(WHERE\s+(.+?)(?=\s+WITH|\s+INTO|\s*$))", std::regex::icase);
        std::regex with_regex(R"(WITH\s*\(\s*([^)]+)\s*\))", std::regex::icase);
        std::regex into_regex(R"(INTO\s+([a-zA-Z_][a-zA-Z0-9_]*))", std::regex::icase);

        // Data preprocessing regex patterns
        std::regex sampling_regex(R"(DATA_SAMPLING\s+([a-zA-Z_]+)(?:\s+RATIO\s+([0-9.]+))?)", std::regex::icase);
        std::regex feature_selection_regex(R"(FEATURE_SELECTION(?:\s+USING\s+([a-zA-Z_]+))?(?:\s+MAX_FEATURES\s+(\d+))?)", std::regex::icase);
        std::regex scaling_regex(R"(SCALING\s+([a-zA-Z_]+))", std::regex::icase);
        std::regex no_scaling_regex(R"(NO_FEATURE_SCALING)", std::regex::icase);

        // Training options regex patterns
        std::regex cv_regex(R"(CROSS_VALIDATION(?:\s+FOLDS\s+(\d+))?)", std::regex::icase);
        std::regex early_stop_regex(R"(EARLY_STOPPING(?:\s+ROUNDS\s+(\d+))?(?:\s+VALIDATION_TABLE\s+([a-zA-Z_][a-zA-Z0-9_]*))?(?:\s+VALIDATION_SPLIT\s+([0-9.]+))?)", std::regex::icase);
        std::regex device_regex(R"(DEVICE\s+(CPU|GPU))", std::regex::icase);
        std::regex threads_regex(R"(NUM_THREADS\s+(\d+))", std::regex::icase);
        std::regex metric_regex(R"(METRIC\s+([a-zA-Z_]+))", std::regex::icase);
        std::regex boosting_regex(R"(BOOSTING\s+([a-zA-Z_]+))", std::regex::icase);
        std::regex seed_regex(R"(SEED\s+(\d+))", std::regex::icase);
        std::regex deterministic_regex(R"(DETERMINISTIC\s+(TRUE|FALSE|1|0))", std::regex::icase);

        // Tuning options regex patterns
        std::regex tune_regex(R"(TUNE_HYPERPARAMETERS(?:\s+USING\s+([a-zA-Z_]+))?(?:\s+ITERATIONS\s+(\d+))?(?:\s+FOLDS\s+(\d+))?(?:\s+JOBS\s+(\d+))?)", std::regex::icase);

        std::smatch match;
        std::string s = query;

        // Extract model name
        if (std::regex_search(s, match, create_model_regex)) {
            stmt->model_name = match[1];
            if (!IsValidModelName(stmt->model_name)) {
                throw std::runtime_error("Invalid model name: " + stmt->model_name);
            }
        } else {
            theow std::runtime_error("Model name required");
        }

        // Extract algirithm
        if (std::regex_search(s, match, using_regex)) {
            stmt->algorithm = ToUpper(match[1]);
        }

        // Check for EXCLUDE FEAURES clause
        if (std::regex_search(s, match, exclude_regex)) {
            std::string exclude_str = match[1];
            std::regex col_regex(R"([a-zA-z][a-zA-Z0-9_]*)");

            auto cols_begin = std::sregex_iterator(exclude_str.begin(), exclude_str.end(), col_regex);
            auto cols_end = std::sregex_iterator();

            for (auto it = cols_begin; it != cols_end, ++it) {
                stmt->exclude_features.push_back((*it).str());
            }
            stmt->use_all_features = true; // Use all features except excluded
        }

        // Extract features
        if (std::regex_search(s, match, features_regex)) {
            std::string features_str = match[1];
            stmt->features = ParseFeatures(features_str);
        } else if (!stmt->use_all_features) {
            // If no features and no exclude, use all features
            stmt->use_all_features = true;
        }

        // Extract target
        if (std::regex_search(s, match, target_regex)) {
            stmt->target_column = match[1];
            if (match[2].match) {
                stmt->target_type = ToUpper(match[2]);
            }
        }

        // Extract source table
        if (std::regex_search(s, match, from_regex)) {
            stmt->source_table = match[1];
        }

        // Extract WHERE caluse
        if (std::regex_search(s, match, where_regex)) {
            stmt->where_clause = Trim(match[1]);
        }

        // Extract with clause
        if (std::regex_search(s, match, with_regex)) {
            std::string params_str = match[1];
            auto params = ParseHyperparameters(params_str);

            // Store all parameters
            for (const auto &[key, value] : params) {
                stmt->parameters[key] = value;
            }

            // Parse specific categories
            stmt->training_options = ParseTRainingOptions(params);
            stmt->tuning_options = ParseTuningOptions(params);
        }

        // Parse data sampling
        if (std::regex_search(S, match, sampling)) {
            stmt->data_sampling = ToUpper(match[1]);
            if (match[2].matched) {
                try {
                    stmt.sammpling_ratio = std::stof(match[2].str());
                } catch (...) {}
            }
        }

        // Parse feature selection
        if (std::regex_search(s, match, feature_selection_regex)) {
            stmt->feature_selection = true;
            if (match[1].matched) {
                stmt->feature_selection_method = ToUpper(match[1].str());
            }
            if (match[2].matched) {
                try {
                    stmt->max_features_to_slect = std::stoi(match[2].str());
                } catch (...) {}
            }
        }

        // Parse scaling
        if (std::regex_search(s, match, scaling_regex)) {
            stmt->feature_scaling = true;
            stmt->scaling_method = ToUpper(match[1].str());
        }
        if (std::regex_search(s, match, no_scaling_regex)) {
            stmt->feature_scaling = false;
        }

        // Parse cross validation
        if (std::regex_search(s, match, cv_regex)) {
            stmt->training_options.cross_validation = true;
            if (match[1].matched) {
                try {
                    stmt->training_options.cv_folds = std::stoi(match[1].str());
                } catch (...) {}
            }
        }

        // Parse early stopping
        if (std::regex_search(s, ,atch, cv_regex)) {
            stmt->training_options.cross_validation = true;
            if (match[1].matched) {
                try {
                    stmt->training_options.early_stopping_rounds = std::stoi(match[1].str());
                } catch (...) {}
            }
            if (match[2].matched) {
                stmt->training_options.validation_table =match[2].str();
            }
            if (match[3].matched) {
                try {
                    stmt->training_options.validation_split = std::stof(match[3].str());
                } catch (...) {}
            }
        }

        // Parse device
        if (std::regex_search(s, match, device_regex)) {
            std::string device = ToUpper(match[1].str());
            if (device == "GPU") {
                stmt->training_options.use_gpu = true;
                stmt->training_options.device_type = "gpu";
            } else {
                stmt->training_options.use_gpu = false;
                stmt->training_options.device_type = "cpu";
            }
        }

        // Parse threads
        if (std::regex_search(s, match, threads_regex)) {
            try {
                stmt->training_options.num_threads = std::stoi(match[1].str());
            }
        }

        // Parse metric
        if (std::regex_search(s, match, metric_regex)) {
            stmt->training_options.metric = match[1].str();
        }

        // Parse boosting type
        if (std::regex_search(s, match, metric_regex)) {
            stmt->training_options.boosting_type = ToUpper(match[1].str());
        }

        // Parse seed
        if (std::regex_search(s, match, seed_regex)) {
            try {
                stmt->training_options.seed = std::stoi(match[1].str());
            } catch (...) {}
        }

        // Parse deterministic
        if (std::regex_search(s, match, deterministic_regex)) {
            std::string val = ToUpper(match[1].str());
            stmt->training_options.deterministic = (val == "TRUE" || val == "1");
        }

        // Parse tuning
        if (std::regex_search(s, match, tune_regex)) {
            stmt->tuning_options.tune_hyperparameters = true;
            if (match[1].matched) {
                stmt->tuning_options.tuning_method = ToUpper(match[1].str());
            }
            if (match[2.matched]) {
                try {
                    stmt->tuning_options.tuning_iterations = std::stoi(match[2].str());
                } catch (...) {}
            }
            if (match[3].matched) {
                try {
                    stmt->tuning_options.tuning_folds = std::stoi(match[3].str());
                } catch (...) {}
            }
            if (match[4].matched) {
                try {
                    stmt->tuning_options.tuning_jobs = std::stoi(match[4].str())
                } catch (...) {}
            }
        }

        // Extract output table
        if (std::regex_search(s, match, into_regex)) {
            stmt->output_table = match[1];
        }

        // Vlidate required fields
        if (stmt->source_table.empty()) {
            throw std::runtime_error("Source table required (FROM clause)");
        }
        if (stmt->target_column.empty()) {
            throw std::runtime_error("Trget column required (TARGET caluse)");
        }

        return stmt;
    }
} // Namespace esql
