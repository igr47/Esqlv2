#include "duckdb.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "include/esql/algorithm_registry.hpp"
#include "include/ai_extension_extension.hpp"
#include "include/esql/model_registry.hpp"
#include "include/esql/lightgbm_model.h"
#include "include/esql/create_model_statement.h"
#include "include/esql/show_models_function.hpp"
#include "include/esql/predict_function.hpp"
#include <filesystem>

namespace duckdb {

// Forward declarations
bool IsCreateModelStatement(const string &query);
unique_ptr<CreateModelStatement> ParseCreateModel(const string &query);

// Forward declarations for the table function implementations
void TrainModelFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output);
unique_ptr<FunctionData> TrainModelBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<string> &names);
void RegisterTrainModelFunction(DatabaseInstance &db);

void AiExtensionExtension::Load(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
    // ========================================================================
    // 1. Register table functions FIRST (so they exist when planning)
    // ========================================================================
	//
	RegisterTrainModelFunction(db);

    // Register train_model table function
    loader.RegisterFunction(
        TableFunction("train_model",
            {
                LogicalType::VARCHAR,                     // model_name
                LogicalType::VARCHAR,                     // algorithm
                LogicalType::LIST(LogicalType::VARCHAR),  // features
                LogicalType::VARCHAR,                     // target_column
                LogicalType::VARCHAR,                     // target_type (optional)
                LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR), // parameters
                LogicalType::VARCHAR                      // source_table
            },
            TrainModelFunction, TrainModelBind)
    );

    // predict table function
    loader.RegisterFunction(
        TableFunction("predict",
            {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
            [](ClientContext &context, TableFunctionInput &data, DataChunk &output) {
                output.SetCardinality(1);
                output.SetValue(0, 0, Value(0.0));
            },
            [](ClientContext &context, TableFunctionBindInput &input,
               vector<LogicalType> &return_types, vector<string> &names) -> unique_ptr<FunctionData> {
                return_types.push_back(LogicalType::DOUBLE);
                names.push_back("prediction");
                return nullptr;
            })
    );

    // show_models table function
    /*loader.RegisterFunction(
        TableFunction("show_models", {},
            [](ClientContext &context, TableFunctionInput &data, DataChunk &output) {
                // TODO: implement
                output.SetCardinality(0);
            })
    );*/

	// Register show_models table function - list all models
    loader.RegisterFunction(
        TableFunction("show_models", {},
            ShowModelsFunction, ShowModelsBind)
    );

	// Register show_model table function - detailed view of a single model
    loader.RegisterFunction(
        TableFunction("show_model",
            {LogicalType::VARCHAR},  // model_name
            ShowModelFunction, ShowModelBind)
    );

	    loader.RegisterFunction(
        TableFunction("predict_table",
            {
                LogicalType::VARCHAR,  // model_name
                LogicalType::VARCHAR,  // input_table
                LogicalType::VARCHAR   // where_clause (optional)
            },
            PredictTableFunction, PredictTableBind)
    );

    // ========================================================================
    // 7. Register scalar predict functions
    // ========================================================================

    // ai_predict - returns prediction value
    loader.RegisterFunction(
        ScalarFunction("ai_predict",
            {LogicalType::VARCHAR},  // model_name
            LogicalType::DOUBLE,
            PredictScalarFunction)
    );
    // Note: We need to make this variadic - will handle in registration

    // ai_predict_class - returns class label
    loader.RegisterFunction(
        ScalarFunction("ai_predict_class",
            {LogicalType::VARCHAR},  // model_name
            LogicalType::VARCHAR,
            PredictClassFunction)
    );

    // ai_predict_proba - returns probability
    loader.RegisterFunction(
        ScalarFunction("ai_predict_proba",
            {LogicalType::VARCHAR},  // model_name
            LogicalType::DOUBLE,
            PredictProbaFunction)
    );

    // ai_predict_probas - returns all probabilities as JSON
    loader.RegisterFunction(
        ScalarFunction("ai_predict_probas",
            {LogicalType::VARCHAR},  // model_name
            LogicalType::VARCHAR,
            PredictProbasFunction)
    );


    // ========================================================================
    // 2. Register scalar functions
    // ========================================================================
    loader.RegisterFunction(
        ScalarFunction("predict",
            {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
            LogicalType::DOUBLE,
            [](DataChunk &args, ExpressionState &state, Vector &result) {
                result.SetValue(0, Value(0.0));
            })
    );

    loader.RegisterFunction(
        ScalarFunction("predict_class",
            {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
            LogicalType::VARCHAR,
            [](DataChunk &args, ExpressionState &state, Vector &result) {
                result.SetValue(0, Value("0"));
            })
    );

    loader.RegisterFunction(
        ScalarFunction("predict_proba",
            {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
            LogicalType::DOUBLE,
            [](DataChunk &args, ExpressionState &state, Vector &result) {
                result.SetValue(0, Value(0.0));
            })
    );

    // ========================================================================
    // 3. Create and register parser extension for CREATE MODEL
    // ========================================================================
    auto parser_ext = make_uniq<ParserExtension>();

    // Create parser info (can be null for now)
    parser_ext->parser_info = nullptr;

    // Set up the parse function
    parser_ext->parse_function = [](ParserExtensionInfo *info, const string &query) -> ParserExtensionParseResult {
        try {
            if (!IsCreateModelStatement(query)) {
                // Not our statement - let DuckDB handle it
                return ParserExtensionParseResult();
            }

            auto stmt = ParseCreateModel(query);
            return ParserExtensionParseResult(std::move(stmt));
        } catch (const std::exception &ex) {
            return ParserExtensionParseResult(ex.what());
        }
    };

    // Set up the plan function - note: first parameter is ParserExtensionInfo*, not ParserExtension*
    parser_ext->plan_function = [](ParserExtensionInfo *info, ClientContext &context,
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
            parameters.emplace_back(Value(LogicalType::VARCHAR));
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
        if (!param_keys.empty()) {
            parameters.emplace_back(Value::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR, param_keys, param_values));
        } else {
            parameters.emplace_back(Value(LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR)));
        }

        // source_table
        parameters.emplace_back(stmt.source_table);

        ParserExtensionPlanResult result;

        // Create a TableFunction that will be bound later
        // We can't look up from catalog because it's not yet registered during parsing
        // Instead, create the table function directly
        result.function = TableFunction("train_model",
            {
                LogicalType::VARCHAR,                     // model_name
                LogicalType::VARCHAR,                     // algorithm
                LogicalType::LIST(LogicalType::VARCHAR),  // features
                LogicalType::VARCHAR,                     // target_column
                LogicalType::VARCHAR,                     // target_type (optional)
                LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR), // parameters
                LogicalType::VARCHAR                      // source_table
            },
            TrainModelFunction, TrainModelBind);

        result.parameters = std::move(parameters);
        result.requires_valid_transaction = false;
        result.return_type = StatementReturnType::QUERY_RESULT;

        return result;
    };

    // Register the parser extension
    //auto &db = loader.GetDatabaseInstance();
    ParserExtension::Register(db.config, std::move(*parser_ext));

    // ========================================================================
    // 4. Initialize algorithm registry (no database access)
    // ========================================================================
    AlgorithmRegistry::Instance();

    // ========================================================================
    // 5. Defer table creation and directory setup to first use.
    //    This is done inside TrainModelBind via a lazy initializer.
    // ========================================================================
}

std::string AiExtensionExtension::Name() {
    return "ai_extension";
}

std::string AiExtensionExtension::Version() const {
    return "0.1.0";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void ai_extension_init(duckdb::ExtensionLoader &loader) {
    duckdb::AiExtensionExtension extension;
    extension.Load(loader);
}

DUCKDB_EXTENSION_API const char *ai_extension_version() {
    return "0.1.0";
}

} // extern "C"
