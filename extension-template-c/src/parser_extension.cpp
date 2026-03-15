#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "create_model_statement.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/paser/tableref/table_function_ref.hpp"
#include <memory>
#include <string>
#include <regex>

namespace esql {
    // Foward declararions
    std::unique_pre<CreateModelStatement> ParseCReateModel(const std::string& query);
    bool IsCreateModelStatement(const std::string& query);

    std::unique_ptr<ParserExtensionParseResult> ParseStatement(ParserExtensionInfo *info, const std::string &query) override {

        auto result = std::make_unique<ParserExtensionParseResult>();

        try {
            if (!IsCreateModelStatement(query)) {
                result->type = ParserExtensionErrorType::UNRECOGNIZED_PARSED_STATEMENT;
                return result;
            }

            auto stmt = ParseCreateModel(query);

            result->type = ParserExtensionErrorType::SUCCESS;
            result->parse_data = std::move(stmt);

        } catch (const std::exception &ex) {
            result->type = ParserExtensionErrorType::SYNTAX_ERROR;
            result->error = ex.what();
        }

        return result;
    }


    std::unique_ptr<ParserExtensionPlanResult> PlanStatement(ParserExtensionInfo *info,ClientContext &context,std::unique_ptr<ParserExtensionParseResult> parse_result) override {
        auto result = std::make_unique<ParserExtensionPlanResult>();

        auto &stmt = dynamic_cast<CreateModelStatement &>(*parse_result->parse_data);

        // Convert to table function call
        std::vector<unique_ptr<ParsedExpression>> children;

// model_name
        children.push_back(std::make_unique<ConstantExpression>(Value(stmt.model_name)));

        // algorithm
        children.push_back(std::make_unique<ConstantExpression>(Value(stmt.algorithm)));

        // features list with types
        std::vector<Value> feature_values;
        for (const auto &feature : stmt.features) {
            feature_values.push_back(Value(feature.first + ":" + feature.second));
        }
        children.push_back(std::make_unique<ConstantExpression>(Value::LIST(feature_values)));

        // target_column
        children.push_back(make_uniq<ConstantExpression>(Value(stmt.target_column)));

        // target_type (optional)
        if (!stmt.target_type.empty()) {
            children.push_back(std::make_unique<ConstantExpression>(Value(stmt.target_type)));
        } else {
            children.push_back(std::make_unique<ConstantExpression>(Value()));
        }

        // parameters map
        std::vector<Value> param_keys, param_values;
        for (const auto &param : stmt.parameters) {
            param_keys.push_back(Value(param.first));
            param_values.push_back(Value(param.second));
        }

        // Add training options to parameters
        auto add_param = [&](const std::string &key, const std::string &value) {
            param_keys.push_back(Value(key));
            param_values.push_back(Value(value));
        };

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

        children.push_back(std::make_unique<ConstantExpression>(
            Value::MAP(Value::LIST(param_keys), Value::LIST(param_values))));

        // source_table
        children.push_back(std::make_unique<ConstantExpression>(Value(stmt.source_table)));

        // Create function call
        auto function = std::make_unique<FunctionExpression>("train_model", std::move(children));

        // Create table function reference
        auto table_function = std::make_unique<TableFunctionRef>();
        table_function->function = std::move(function);

        result->plan = std::move(table_function);
        result->requires_valid_transaction = false;
        result->return_type = StatementReturnType::QUERY_RESULT;

        return result;
    }

    bool IsCreateModelStatement(const std::string &query) {
        // Simple regex to detect CREATE MODEL
        std::regex pattern(R"(^\s*CREATE\s+MODEL\s+([a-zA-Z_][a-zA-Z0-9_]*))",
                       std::regex::icase | std::regex::optimize);
        return std::regex_search(query, pattern);
    }

} // namespace esql

