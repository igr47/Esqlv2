#pragma once
#include "duckdb.hpp"
#include <unordered_map>
#include <vector>
#include <set>

namespace esql {
    enum class AlgorithmCategory {
        REGRESSION,
        CLASSIFICATION,
        RANKING,
        CUSTOM
    };

    struct AlgorithmInfo {
        std::string name;  // User-facing name
        std::string lightgbm_objective; // LightGBM objective string
        AlgorithmCategory category;
        std::string describtion;
        std::vector<std::string> suitable_problems;
        std::unordered_map<std::string, std::string> default_params;
        bool required_num_classes = false;
        bool support_probabilility = false;
        bool IsSuitbleFor(const std::string& target_type, size_t num_classes = 0) const;
    };

    class AlgorithmRegistry {
        private:
            std::unordered_map<std::string, AlgorithmInfor> algorithms_;
            std::unordered_map<AlgoritmCategory, std::vector<std::string>> by_category_;

            AlgorithmRegistry();

        public:
            static AlgorithmRegistry &Instance();

            // Core methods
            bool RegisterAlgorithm(const AlgoInfo& info);
            const AlgorithmInfo *GetAlgorithm(const std::string& name) const;
            std::vector<std::string> GetSupportedAlgorithms() const;
            std::vector<std::string> GetAlgorithmsByCategory(AlgorithmCategory category) const;

            // Helper methods
            std::string SuggestAlgorithm(const std::string& problem_type, const std::vector<float>& sample_lables) const;
            bool IsAlgorithmSupported(const std::string& name) const;
            bool ValidateAlgorithmChoise(const string& algorithm_name, const std::string& target_type, size_t num_classes = 0) const;

        private:
            void InitializeDefaultAlgorithms();
    };
} // Namespace esql
#endif
