#pragma once
#include "duckdb.hpp"
#include "duckdb/parser/parsed_data/create_info.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/common/string.hpp"

#include <unordered_map>
#include <string>
#include <vector>
#include <set>
#include <optional>

namespace duckdb {
    // Struct training options(The struct cantains the optons that can be used with the model
    // creation statement)
    struct TrainingOptions {
        bool cross_validation = false;
        int cv_folds = 5;
        bool early_stopping_rounds = 10;
		bool early_stopping = false;
        std::string validation_table;
        float validation_split = 0.2f;
        bool use_gpu = false;
        std::string device_type = "cpu";
        int num_threads = -1;
        std::string metric = "auto";
        std::string boosting_type = "gbdt";
        int seed = 42;
        bool deterministic = true;

        // Serialization
        std::string ToString() const;
        static TrainingOptions FromMap(const std::unordered_map<std::string, std::string> &params);
    };

    // Hyper parameter tuning options
    struct TuningOptions {
        bool tune_hyperparameters = false;
        std::string tuning_method = "grid"; // grid, random, byessian
        int tuning_iterations = 10;
        int tuning_folds = 3;
        std::string scoring_metric = "auto";
        std::unordered_map<std::string, std::vector<std::string>> param_grid;
        std::unordered_map<std::string, std::pair<float, float>> param_ranges;
        bool parallel_tuning = true;
        int tuning_jobs = -1;

        // Serialization
        std::string ToString() const;
        static TuningOptions FromMap(const std::unordered_map<std::string, std::string> &params);
    };

    // Main CREATE MODEL statement
    class CreateModelStatement : public ParserExtensionParseData/*SQLStatement*/ {
        public:
            //CreateModelStatement() : SQLStatement(StatementType::EXTENSION_STATEMENT) {}

            // Core properties
            std::string model_name;
            std::string algorithm = "LIGHTGBM";

            //Features: name -> type (NUMERIC, CATEGORICAL, AUTO)
            std::vector<std::pair<std::string, std::string>> features;

            // Columns to exclude from features(if use_all_features = true)
            std::vector<std::string> exclude_features;
            bool use_all_features = false;

            // Target specification
            std::string target_column;
            std::string target_type; // BINARY, MULTICLASS, REGRESSION, etc.

            // Source data
            std::string source_table;
            std::string where_clause;

            // Parameters
            std::unordered_map<std::string, std::string> parameters;

            // Training configuration
            TrainingOptions training_options;
            TuningOptions tuning_options;

            // Data preprocessing
            std::string data_sampling = "none"; // none, oversample, undersample
            std::string feature_selection_method = "auto";
            std::string scaling_method = "standard";
            float sampling_ratio = 1.0f;
            bool feature_selection = false;
            bool feature_scaling = true;
            int max_features_to_select = -1;

            //Output
            std::string output_table; // Where to store model metadata

            // Flags
            bool if_not_exists = false;
            bool replace = false;
        public:
            // Serialization
            std::string ToString() const override {
				return "CREATE MODEL " + model_name;
			}
			unique_ptr<ParserExtensionParseData> Copy() const override {
				return make_uniq<CreateModelStatement>(*this);
			}
			void RegisterTrainModelFunction(DatabaseInstance &db);
            //std::unique_ptr<SQLStatement> copy() const;

            // Helper methods
            bool IsValid() const;
            std::string GetFeatureLIst() const;
            std::string GetParameterString();
    };
} // namespace duckdb
