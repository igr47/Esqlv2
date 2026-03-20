#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

namespace duckdb {

class MinimalExtension : public Extension {
public:
    void Load(ExtensionLoader &loader) override {
        auto &db = loader.GetDatabaseInstance();
        Connection conn(db);
        auto &context = *conn.context;

        // Create scalar function
        auto test_func = ScalarFunction(
            "minimal_test",
            {},
            LogicalType::VARCHAR,
            [](DataChunk &args, ExpressionState &state, Vector &result) {
                result.SetValue(0, Value("Minimal extension loaded!"));
            }
        );

        // Register via catalog
        CreateScalarFunctionInfo func_info(test_func);
        Catalog::GetSystemCatalog(db).CreateFunction(context, func_info);
    }

    std::string Name() override { return "minimal"; }
    std::string Version() const override { return "0.1.0"; }
};

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void minimal_extension_init(duckdb::ExtensionLoader &loader) {
    duckdb::MinimalExtension extension;
    extension.Load(loader);
}

DUCKDB_EXTENSION_API const char *minimal_extension_version() {
    return "0.1.0";
}

}
