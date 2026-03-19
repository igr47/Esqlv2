#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "include/esql/create_model_statement.h"
#include <regex>

namespace duckdb {

// Forward declarations
bool IsCreateModelStatement(const string &query);
unique_ptr<CreateModelStatement> ParseCreateModel(const string &query);

class CreateModelParserExtension : public ParserExtension {
public:
    CreateModelParserExtension() {
        // Set the parse function
        parse_function = [](ParserExtensionInfo *info, const string &query) -> ParserExtensionParseResult {
            try {
                if (!IsCreateModelStatement(query)) {
                    // Not our statement – let DuckDB handle it
                    return ParserExtensionParseResult();
                }

                auto stmt = ParseCreateModel(query);
                return ParserExtensionParseResult(std::move(stmt));
            } catch (const std::exception &ex) {
                return ParserExtensionParseResult(ex.what());
            }
        };

        // Set the plan function
        plan_function = [](ParserExtensionInfo *info, ClientContext &context,
                           unique_ptr<ParserExtensionParseData> parse_data) -> ParserExtensionPlanResult {
            auto &stmt = dynamic_cast<CreateModelStatement &>(*parse_data);

            // Build the argument list for the train_model table function
            vector<Value> parameters;

            // model_name
            parameters.emplace_back(stmt.model_name);

            // algorithm
            parameters.emplace_back(stmt.algorithm);

            // features list (as a LIST of VARCHAR)
            vector<Value> feature_values;
            for (const auto &feature : stmt.features) {
                feature_values.emplace_back(feature.first + ":" + feature.second);
            }
            parameters.emplace_back(Value::LIST(feature_values));

            // target_column
            parameters.emplace_back(stmt.target_column);

            // target_type (optional)
            if (!stmt.target_type.empty()) {
                parameters.emplace_back(stmt.target_type);
            } else {
                parameters.emplace_back();
            }

            // Build parameters map (as a MAP(VARCHAR, VARCHAR))
            vector<Value> param_keys, param_values;

            auto add_param = [&](const string &key, const string &value) {
                param_keys.emplace_back(key);
                param_values.emplace_back(value);
            };

            // User parameters
            for (const auto &p : stmt.parameters) {
                add_param(p.first, p.second);
            }

            // Training options
            add_param("CROSS_VALIDATION", stmt.training_options.cross_validation ? "true" : "false");
            add_param("CV_FOLDS", std::to_string(stmt.training_options.cv_folds));
            add_param("EARLY_STOPPING", stmt.training_options.early_stopping ? "true" : "false");
            add_param("EARLY_STOPPING_ROUNDS", std::to_string(stmt.training_options.early_stopping_rounds));
            add_param("VALIDATION_TABLE", stmt.training_options.validation_table);
            add_param("VALIDATION_SPLIT", std::to_string(stmt.training_options.validation_split));
            add_param("USE_GPU", stmt.training_options.use_gpu ? "true" : "false");
            add_param("NUM_THREADS", std::to_string(stmt.training_options.num_threads));
            add_param("METRIC", stmt.training_options.metric);
            add_param("BOOSTING", stmt.training_options.boosting_type);
            add_param("SEED", std::to_string(stmt.training_options.seed));
            add_param("DETERMINISTIC", stmt.training_options.deterministic ? "true" : "false");

            // Tuning options
            add_param("TUNE_HYPERPARAMETERS", stmt.tuning_options.tune_hyperparameters ? "true" : "false");
            add_param("TUNING_METHOD", stmt.tuning_options.tuning_method);
            add_param("TUNING_ITERATIONS", std::to_string(stmt.tuning_options.tuning_iterations));
            add_param("TUNING_FOLDS", std::to_string(stmt.tuning_options.tuning_folds));
            add_param("SCORING_METRIC", stmt.tuning_options.scoring_metric);
            add_param("PARALLEL_TUNING", stmt.tuning_options.parallel_tuning ? "true" : "false");
            add_param("TUNING_JOBS", std::to_string(stmt.tuning_options.tuning_jobs));

            // Data preprocessing
            add_param("DATA_SAMPLING", stmt.data_sampling);
            add_param("SAMPLING_RATIO", std::to_string(stmt.sampling_ratio));
            add_param("FEATURE_SELECTION", stmt.feature_selection ? "true" : "false");
            add_param("FEATURE_SELECTION_METHOD", stmt.feature_selection_method);
            add_param("MAX_FEATURES_TO_SELECT", std::to_string(stmt.max_features_to_select));
            add_param("NO_FEATURE_SCALING", !stmt.feature_scaling ? "true" : "false");
            add_param("SCALING_METHOD", stmt.scaling_method);
            add_param("OUTPUT_TABLE", stmt.output_table);

            // Create MAP value
            parameters.emplace_back(Value::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR, param_keys, param_values));

            // source_table
            parameters.emplace_back(stmt.source_table);

            ParserExtensionPlanResult result;
			// Look up te existing train_model table function from the catalog
			auto &catalog = Catalog::GetSystemCatalog(context);
			auto &func_entry = catalog.GetEntry<TableFunctionCatalogEntry>(context, DEFAULT_SCHEMA, "train_model");
			auto &func_set = func_entry.functions;
			result.function = func_set.functions.at(0);
            //result.function = TableFunction("train_model", {});
            result.parameters = std::move(parameters);
            result.requires_valid_transaction = false;
            result.return_type = StatementReturnType::QUERY_RESULT;

            return result;
        };
    }
};

bool IsCreateModelStatement(const string &query) {
    std::regex pattern(R"(^\s*CREATE\s+MODEL\s+([a-zA-Z_][a-zA-Z0-9_]*))",
                       std::regex::icase | std::regex::optimize);
    return std::regex_search(query, pattern);
}

} // namespace duckdb
