#pragma once
//#include "duckdb.hpp"
#include "lightgbm_isolated.h"
#include <unordered_map>
#include <vector>
#include <set>

namespace duckdb {

enum class AlgorithmCategory {
    REGRESSION,
    CLASSIFICATION,
    RANKING,
    CUSTOM
};

struct AlgorithmInfo {
    std::string name;                 // User-facing name
    std::string lightgbm_objective;   // LightGBM objective string
    AlgorithmCategory category;
    std::string description;
    std::vector<std::string> suitable_problems;
    std::unordered_map<std::string, std::string> default_params;
    bool requires_num_classes = false;
    bool supports_probability = false;

    bool IsSuitableFor(const string &target_type, size_t num_classes = 0) const;
};

class AlgorithmRegistry {
private:
    std::unordered_map<std::string, AlgorithmInfo> algorithms_;
    std::unordered_map<AlgorithmCategory, vector<std::string>> by_category_;

    AlgorithmRegistry(); // Private constructor for singleton

public:
    static AlgorithmRegistry &Instance();

    // Core methods
    bool RegisterAlgorithm(const AlgorithmInfo &info);
    const AlgorithmInfo *GetAlgorithm(const string &name) const;
    vector<string> GetSupportedAlgorithms() const;
    vector<string> GetAlgorithmsByCategory(AlgorithmCategory category) const;

    // Helper methods
    string SuggestAlgorithm(const string &problem_type,
                           const vector<float> &sample_labels) const;

    bool IsAlgorithmSupported(const string &name) const;
    bool ValidateAlgorithmChoice(const string &algorithm_name,
                                 const string &target_type,
                                 size_t num_classes = 0) const;

private:
    void InitializeDefaultAlgorithms();
};

} // namespace duckdb
