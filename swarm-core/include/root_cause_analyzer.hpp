#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>

// Forward declarations for dependency graph
struct DependencyLink;
class DependencyGraph;

// Data structures for root cause analysis
struct AgentDependency {
    std::string agent_id;
    float confidence_score;  // 0.0 to 1.0
    std::string failure_mode;
    std::vector<std::string> symptoms;
};

struct MitigationRecommendation {
    std::string action;
    std::string priority;  // "HIGH", "MEDIUM", "LOW"
    std::string rationale;
    float expected_impact;  // Percentage improvement expected
};

struct RootCauseResult {
    std::vector<AgentDependency> dependency_chain;
    std::string primary_root_cause;
    std::vector<MitigationRecommendation> recommendations;
    float analysis_confidence;
    std::chrono::milliseconds analysis_duration;
    bool analysis_complete;  // True if completed within timeouts, False if terminated early
};

enum class AnalysisResult {
    SUCCESS,
    CYCLE_DETECTED,
    DEPTH_LIMIT_EXCEEDED,
    TIMEOUT_EXCEEDED,
    MEMORY_LIMIT_EXCEEDED,
    INVALID_INPUT
};

// Configurations matching Swarm-100 architecture
struct AnalysisConfig {
    int max_recursion_depth = 50;
    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000);  // 5 seconds
    size_t max_memory_mb = 10;  // Memory usage limit
    int max_dependency_chain = 25;  // Maximum agents in chain
    float min_confidence_threshold = 0.1f;  // Minimum confidence to continue analysis
};

// Simplified dependency graph representation
using AdjacencyList = std::unordered_map<std::string, std::vector<DependencyLink>>;

struct DependencyLink {
    std::string target_agent;
    float strength;  // 0.0 to 1.0 - how strongly this agent depends on target
    std::string relationship_type;  // "communication", "resource", "gossip", etc.
};

/**
 * Root Cause Analysis Engine for Swarm-100
 *
 * Implements recursive testing loops with proper loop tracking and reasonable limits
 * to identify root causes of multi-agent coordination failures.
 *
 * Safety features:
 * - Stack depth monitoring to prevent infinite recursion
 * - Timeout protection for long-running analyses
 * - Cycle detection in dependency graphs
 * - Memory usage limits
 */
class RootCauseAnalyzer {
private:
    // Analysis configuration
    AnalysisConfig config_;

    // Loop tracking and safety
    std::unordered_set<std::string> visited_nodes_;
    std::stack<std::string> recursion_stack_;
    std::unordered_map<std::string, int> dependency_depths_;
    std::chrono::steady_clock::time_point analysis_start_time_;
    size_t current_memory_usage_;  // In bytes

    // Analysis state
    std::unique_ptr<RootCauseResult> current_result_;
    const AdjacencyList* current_graph_;

    // Pattern recognition for common failure modes
    std::unordered_map<std::string, std::vector<std::string>> failure_patterns_;

public:
    /**
     * Constructor
     * @param config Analysis configuration parameters
     */
    explicit RootCauseAnalyzer(const AnalysisConfig& config = AnalysisConfig());

    /**
     * Main analysis method - recursive root cause detection
     * @param target_agent The agent where failure was observed
     * @param symptoms List of observed symptoms/error messages
     * @param graph The dependency graph of agent relationships
     * @return Root cause analysis result
     */
    RootCauseResult analyze_dependency_chain(
        const std::string& target_agent,
        const std::vector<std::string>& symptoms,
        const AdjacencyList& graph
    );

    /**
     * Check if dependency graph contains cycles
     * @param graph The dependency graph to analyze
     * @return True if cycles detected
     */
    bool detect_cycles(const AdjacencyList& graph);

    /**
     * Reset internal state for new analysis
     * Clears loop tracking, recursion stack, and memory usage counters
     */
    void reset_tracking();

    /**
     * Get current memory usage estimate
     * @return Memory usage in bytes
     */
    size_t get_current_memory_usage() const { return current_memory_usage_; }

private:
    /**
     * Recursive analysis function with stack protection
     * @param current_agent Current agent being analyzed
     * @param symptoms Symptoms propagated from caller
     * @param depth Current recursion depth
     * @return Analysis result indicating success or termination reason
     */
    AnalysisResult analyze_recursive(
        const std::string& current_agent,
        std::vector<std::string> symptoms,
        int depth = 0
    );

    /**
     * Check if current path contains cycles
     * @return True if cycle detected in current recursion path
     */
    bool has_cycle() const;

    /**
     * Check if analysis timed out
     * @return True if analysis exceeded timeout
     */
    bool has_timed_out() const;

    /**
     * Check if current memory usage exceeds limits
     * @return True if memory usage exceeded
     */
    bool memory_limit_exceeded() const;

    /**
     * Add agent to current result's dependency chain
     * @param agent_id Agent that contributes to failure
     * @param confidence Confidence in this agent's role
     * @param failure_mode How this agent is failing
     * @param symptoms Evidence of failure
     */
    void add_to_dependency_chain(
        const std::string& agent_id,
        float confidence,
        const std::string& failure_mode,
        const std::vector<std::string>& symptoms
    );

    /**
     * Identify failure mode from symptoms and agent behavior
     * @param symptoms Observed symptoms
     * @param agent_id Agent being analyzed
     * @return Failure mode classification
     */
    std::string identify_failure_mode(
        const std::vector<std::string>& symptoms,
        const std::string& agent_id
    ) const;

    /**
     * Calculate confidence score for agent's role in failure
     * @param symptoms Evidence against this agent
     * @param dependencies Agent's relationships to others
     * @return Confidence score (0.0 to 1.0)
     */
    float calculate_confidence_score(
        const std::vector<std::string>& symptoms,
        const std::vector<DependencyLink>& dependencies
    ) const;

    /**
     * Generate mitigation recommendations based on identified failures
     * @param failure_mode Type of failure detected
     * @param dependency_chain Chain of failing agents
     * @return List of recommended actions
     */
    std::vector<MitigationRecommendation> generate_recommendations(
        const std::string& failure_mode,
        const std::vector<AgentDependency>& dependency_chain
    ) const;

    /**
     * Initialize failure pattern recognition
     * Maps symptom patterns to failure modes
     */
    void initialize_failure_patterns();

    /**
     * Estimate memory usage for analysis state
     * @return Estimated memory usage in bytes
     */
    size_t estimate_memory_usage() const;

    /**
     * Trust-based fault containment methods
     * Methods for incorporating agent trust scores into root cause analysis
     */

    /**
     * Analyze with trust-based filtering
     * @param target_agent The agent where failure was observed
     * @param symptoms List of observed symptoms/error messages
     * @param graph The dependency graph of agent relationships
     * @param trust_scores Map of agent_id -> trust_score (0.0 to 1.0)
     * @return Root cause analysis result with trust-based filtering
     */
    RootCauseResult analyze_with_trust_filtering(
        const std::string& target_agent,
        const std::vector<std::string>& symptoms,
        const AdjacencyList& graph,
        const std::unordered_map<std::string, float>& trust_scores
    );

    /**
     * Set trust-based analysis parameters
     * @param min_trust_threshold Minimum trust score to consider agent reliable (default 0.3)
     * @param trust_decay_factor How much to reduce confidence for low-trust agents (default 0.5)
     */
    void configure_trust_analysis(float min_trust_threshold = 0.3f, float trust_decay_factor = 0.5f);

    /**
     * Filter dependency chain based on trust scores
     * @param chain Original dependency chain
     * @param trust_scores Agent trust scores
     * @return Filtered chain with low-trust agents potentially removed
     */
    std::vector<AgentDependency> filter_chain_by_trust(
        const std::vector<AgentDependency>& chain,
        const std::unordered_map<std::string, float>& trust_scores
    ) const;

    /**
     * Generate trust-based mitigation recommendations
     * @param failure_mode Type of failure detected
     * @param dependency_chain Chain of failing agents
     * @param trust_scores Agent trust scores
     * @return List of recommended actions considering trust levels
     */
    std::vector<MitigationRecommendation> generate_trust_based_recommendations(
        const std::string& failure_mode,
        const std::vector<AgentDependency>& dependency_chain,
        const std::unordered_map<std::string, float>& trust_scores
    ) const;

private:
    // Trust-based analysis parameters
    float min_trust_threshold_;
    float trust_decay_factor_;
    const std::unordered_map<std::string, float>* current_trust_scores_;
};

// Utility functions for result formatting
std::string analysis_result_to_string(AnalysisResult result);
std::string format_root_cause_result(const RootCauseResult& result);

// Exception classes for error handling
class RootCauseAnalysisException : public std::runtime_error {
public:
    explicit RootCauseAnalysisException(const std::string& message)
        : std::runtime_error(message) {}
};

class AnalysisTimeoutException : public RootCauseAnalysisException {
public:
    explicit AnalysisTimeoutException()
        : RootCauseAnalysisException("Root cause analysis timed out") {}
};

class CycleDetectionException : public RootCauseAnalysisException {
public:
    explicit CycleDetectionException(const std::string& cycle_description)
        : RootCauseAnalysisException("Cycle detected in dependency graph: " + cycle_description) {}
};
