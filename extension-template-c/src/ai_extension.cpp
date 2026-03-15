#include "duckdb.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "create_model_statement.hpp"
#include "algorithm_registry.hpp"
#include "model_registry.hpp"
#include <filesystem>

namespace esql {

// External declarations
void RegisterTrainModelFunction(DatabaseInstance &db);
void RegisterPredictFunction(DatabaseInstance &db);
void RegisterShowModelsFunction(DatabaseInstance &db);
void RegisterDropModelFunction(DatabaseInstance &db);
void RegisterDescribeModelFunction(DatabaseInstance &db);

class AIExtension : public Extension {
public:
    void Load(DuckDB &db) override {
        auto &db_instance = db.GetDatabase();
        auto &config = DBConfig::GetConfig(*db_instance);

        // 1. Register parser extension for custom syntax
        config.parser_extensions.push_back(
            make_uniq<CreateModelParserExtension>());

        // 2. Register table functions
        RegisterTrainModelFunction(*db_instance);
        RegisterPredictFunction(*db_instance);
        RegisterShowModelsFunction(*db_instance);
        RegisterDropModelFunction(*db_instance);
        RegisterDescribeModelFunction(*db_instance);

        // 3. Register scalar functions for predictions
        RegisterPredictionScalarFunctions(*db_instance);

        // 4. Create system tables for model registry
        CreateModelRegistryTables(*db_instance);

        // 5. Ensure models directory exists
        std::filesystem::create_directories(ModelRegistry::GetModelsDirectory());

        // 6. Initialize algorithm registry
        AlgorithmRegistry::Instance();

        // 7. Register macros for syntactic sugar
        RegisterSyntaxMacros(*db_instance);
    }

    std::string Name() override {
        return "ai_extension";
    }
};

void CreateModelRegistryTables(DatabaseInstance &db) {
    Connection conn(db);

    // Create models table if not exists
    conn.Query(R"(
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

    // Create index on model_name
    conn.Query("CREATE INDEX IF NOT EXISTS idx_model_registry_name ON __model_registry(model_name)");
}

void RegisterPredictionScalarFunctions(DatabaseInstance &db) {
    // PREDICT(model_name, feature1, feature2, ...)
    // This is a simplified version - you'd implement the full logic
    ExtensionUtil::RegisterFunction(db, ScalarFunction(
        "predict",
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::DOUBLE,
        [](ClientContext &context, ScalarFunction::BindInput &input,
           vector<Value> &arguments, Value &result) {
            // Implementation would load model and predict
            result = Value(0.0);
        }
    ));

    // PREDICT_CLASS(model_name, feature1, feature2, ...)
    ExtensionUtil::RegisterFunction(db, ScalarFunction(
        "predict_class",
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::VARCHAR,
        [](ClientContext &context, ScalarFunction::BindInput &input,
           vector<Value> &arguments, Value &result) {
            result = Value("0");
        }
    ));

    // PREDICT_PROBA(model_name, feature1, feature2, ...)
    ExtensionUtil::RegisterFunction(db, ScalarFunction(
        "predict_proba",
        {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::DOUBLE,
        [](ClientContext &context, ScalarFunction::BindInput &input,
           vector<Value> &arguments, Value &result) {
            result = Value(0.0);
        }
    ));
}

void RegisterShowModelsFunction(DatabaseInstance &db) {
    TableFunction show_models("show_models", {},
        [](ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
            // Implementation
        });

    ExtensionUtil::RegisterFunction(db, show_models);
}

void RegisterDropModelFunction(DatabaseInstance &db) {
    // Drop model function
}

void RegisterDescribeModelFunction(DatabaseInstance &db) {
    // Describe model function
}

void RegisterSyntaxMacros(DatabaseInstance &db) {
    Connection conn(db);

    // Create macro that transforms CREATE MODEL syntax
    conn.Query(R"(
        CREATE MACRO "CREATE MODEL"(model_name, algorithm, features, target, source_table) AS TABLE
        SELECT * FROM train_model(
            model_name,
            algorithm,
            list_value(features),
            target,
            '',
            map([], []),
            source_table
        )
    )");

    // Create macro for PREDICT_USING_model_name syntax
    conn.Query(R"(
        CREATE MACRO "PREDICT_USING"(model_name, features) AS
        SELECT predict(model_name, features)
    )");
}

} // namespace duckdb

// Extension entry point
extern "C" {
DUCKDB_EXTENSION_API void ai_extension_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::AIExtension>();
}

DUCKDB_EXTENSION_API const char *ai_extension_version() {
    return "v0.1.0";
}
}
