#include "duckdb.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "include/esql/create_model_statement.h"
#include "include/esql/algorithm_registry.hpp"
#include "include/esql/model_registry.hpp"
#include <filesystem>

namespace duckdb {

// Forward declarations – these need to be implemented elsewhere
class CreateModelParserExtension : public ParserExtension {
public:
    CreateModelParserExtension() : ParserExtension() {
        // parse_function: (ParserExtensionInfo*, const string&) -> ParserExtensionParseResult
        parse_function = [](ParserExtensionInfo *info, const string &query)
            -> ParserExtensionParseResult {
            // TODO: Implement actual parsing of CREATE MODEL statements
            // Return an error to fall back to the default parser
            return ParserExtensionParseResult("CREATE MODEL syntax not yet implemented");
        };

        // plan_function: (ParserExtensionInfo*, ClientContext&, unique_ptr<ParserExtensionParseData>)
        // -> ParserExtensionPlanResult
        plan_function = [](ParserExtensionInfo *info, ClientContext &context,
                           unique_ptr<ParserExtensionParseData> parse_data)
            -> ParserExtensionPlanResult {
            // TODO: Implement planning
            ParserExtensionPlanResult result;
            result.requires_valid_transaction = false;
            result.return_type = StatementReturnType::QUERY_RESULT;
            return result;
        };
    }
};

// External declarations
void RegisterTrainModelFunction(DatabaseInstance &db);
void RegisterPredictFunction(DatabaseInstance &db);
void RegisterShowModelsFunction(DatabaseInstance &db);
void RegisterDropModelFunction(DatabaseInstance &db);
void RegisterDescribeModelFunction(DatabaseInstance &db);
void RegisterPredictionScalarFunctions(DatabaseInstance &db);
void CreateModelRegistryTables(DatabaseInstance &db);
void RegisterSyntaxMacros(ClientContext &context);

class AIExtension : public Extension {
public:
    // Load(ExtensionLoader&) is the correct override
    void Load(ExtensionLoader &loader) override {
        // Get the database instance from the loader
        auto &db_instance = loader.GetDatabaseInstance();   // Assumes ExtensionLoader has GetDatabase()
        auto &config = DBConfig::GetConfig(db_instance);

        // 1. Register parser extension for custom syntax
        //config.parser_extensions.push_back(
            //make_uniq<CreateModelParserExtension>());
		//config.options.extensions->RegisterParserExtension(make_uniq<CreateModelParserExtension>());
		//config.options.parser_extensions.push_back(make_uniq<CreateModelParserExtension>());


        // 2. Register table functions
        RegisterTrainModelFunction(db_instance);
        RegisterPredictFunction(db_instance);
        RegisterShowModelsFunction(db_instance);
        RegisterDropModelFunction(db_instance);
        RegisterDescribeModelFunction(db_instance);

        // 3. Register scalar functions for predictions
        RegisterPredictionScalarFunctions(db_instance);

        // 4. Create system tables for model registry
        CreateModelRegistryTables(db_instance);

        // 5. Ensure models directory exists
        std::filesystem::create_directories(ModelRegistry::GetModelsDirectory());

        // 6. Initialize algorithm registry
        AlgorithmRegistry::Instance();

        // 7. Register macros for syntactic sugar
        Connection conn(db_instance);
        RegisterSyntaxMacros(*conn.context);
    }

    std::string Name() override {
        return "ai_extension";
    }

    std::string Version() const override {
        return "0.1.0";
    }
};

void CreateModelRegistryTables(DatabaseInstance &db) {
    Connection conn(db);

    // Use single-argument Query (no QueryParameters) to avoid template issues
    auto result = conn.Query(R"(
        CREATE TABLE IF NOT EXISTS __model_registry (
            model_name VARCHAR PRIMARY KEY,
            algorithm VARCHAR,
            problem_type VARCHAR,
            feature_names VARCHAR[],
            target_column VARCHAR,
            parameters MAP(VARCHAR, VARCHAR),
            created_at TIMESTAMP,
            training_samples UBIGINT,
            accuracy DOUBLE,
            precision DOUBLE,
            recall DOUBLE,
            f1_score DOUBLE,
            auc_score DOUBLE,
            r2_score DOUBLE,
            rmse DOUBLE,
            mae DOUBLE,
            model_path VARCHAR,
            is_active BOOLEAN DEFAULT true,
            prediction_count UBIGINT DEFAULT 0,
            drift_score DOUBLE DEFAULT 0.0,
            last_used TIMESTAMP
        )
    )");

    if (result->HasError()) {
        throw std::runtime_error(result->GetError());
    }

    auto index_result = conn.Query(
        "CREATE INDEX IF NOT EXISTS idx_model_registry_name ON __model_registry(model_name)");
    if (index_result->HasError()) {
        throw std::runtime_error(index_result->GetError());
    }
}

void RegisterPredictionScalarFunctions(DatabaseInstance &db) {
    auto &catalog = Catalog::GetSystemCatalog(db);
    Connection conn(db);
    auto &context = *conn.context;

    // PREDICT(model_name, feature1, feature2, ...)
    ScalarFunction predict_func(
        "predict",
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::DOUBLE,
        [](DataChunk &args, ExpressionState &state, Vector &result) {
            // TODO: Implement prediction logic
            result.SetValue(0, Value(0.0));
        }
    );

    CreateScalarFunctionInfo predict_info(predict_func);
    catalog.CreateFunction(context, predict_info);

    // PREDICT_CLASS(model_name, feature1, feature2, ...)
    ScalarFunction predict_class_func(
        "predict_class",
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::VARCHAR,
        [](DataChunk &args, ExpressionState &state, Vector &result) {
            // TODO: Implement class prediction logic
            result.SetValue(0, Value("0"));
        }
    );

    CreateScalarFunctionInfo predict_class_info(predict_class_func);
    catalog.CreateFunction(context, predict_class_info);

    // PREDICT_PROBA(model_name, feature1, feature2, ...)
    ScalarFunction predict_proba_func(
        "predict_proba",
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::DOUBLE,
        [](DataChunk &args, ExpressionState &state, Vector &result) {
            // TODO: Implement probability prediction logic
            result.SetValue(0, Value(0.0));
        }
    );

    CreateScalarFunctionInfo predict_proba_info(predict_proba_func);
    catalog.CreateFunction(context, predict_proba_info);
}

void RegisterShowModelsFunction(DatabaseInstance &db) {
    TableFunction show_models("show_models", {},
        [](ClientContext &context, TableFunctionInput &data, DataChunk &output) {
            // TODO: Implement show models logic
        });

    CreateTableFunctionInfo show_models_info(show_models);
    Connection conn(db);
    auto &context = *conn.context;
    Catalog::GetSystemCatalog(db).CreateTableFunction(context, show_models_info);
}

void RegisterDropModelFunction(DatabaseInstance &db) {
    (void)db; // suppress unused warning
}

void RegisterDescribeModelFunction(DatabaseInstance &db) {
    (void)db;
}

void RegisterSyntaxMacros(ClientContext &context) {
    // ClientContext::Query requires a QueryParameters argument
    auto result = context.Query(R"(
        CREATE OR REPLACE MACRO "CREATE MODEL"(model_name, algorithm, features, target, source_table) AS TABLE
        SELECT * FROM train_model(
            model_name,
            algorithm,
            list_value(features),
            target,
            '',
            map([], []),
            source_table
        )
    )", QueryParameters());

    if (result->HasError()) {
        throw std::runtime_error(result->GetError());
    }

    auto macro_result = context.Query(R"(
        CREATE OR REPLACE MACRO "PREDICT_USING"(model_name, features) AS
        SELECT predict(model_name, features)
    )", QueryParameters());

    if (macro_result->HasError()) {
        throw std::runtime_error(macro_result->GetError());
    }
}

} // namespace duckdb

extern "C" {

// Use the C++ extension entry point macro (as in quack_extension.cpp)
DUCKDB_CPP_EXTENSION_ENTRY(ai_extension, loader) {
    duckdb::AIExtension extension;
    extension.Load(loader);
}

DUCKDB_EXTENSION_API const char *ai_extension_version() {
    return "0.1.0";
}

} // extern "C"
