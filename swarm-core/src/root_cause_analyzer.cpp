#include "root_cause_analyzer.hpp"
#include <algorithm>
#include <sstream>
#include <iterator>
#include <cmath>

// Implementation of RootCauseAnalyzer methods

RootCauseAnalyzer::RootCauseAnalyzer(const AnalysisConfig& config)
    : config_(config), current_memory_usage_(0) {
    initialize_failure_patterns();
}

RootCauseResult RootCauseAnalyzer::analyze_dependency_chain(
    const std::string& target_agent,
    const std::vector<std::string>& symptoms,
    const AdjacencyList& graph
) {
    // Reset tracking for new analysis
    reset_tracking();

    // Initialize analysis state
    analysis_start_time_ = std::chrono::steady_clock::now();
    current_result_ = std::make_unique<RootCauseResult>();
    current_result_->analysis_complete = false;
    current_graph_ = &graph;

    try {
        // Start recursive analysis
        AnalysisResult result = analyze_recursive(target_agent, symptoms, 0);

        // Calculate final metrics
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        current_result_->analysis_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - analysis_start_time_
        );

        current_result_->analysis_complete = (result == AnalysisResult::SUCCESS);

        // Determine primary root cause from dependency chain
        if (!current_result_->dependency_chain.empty()) {
            // Sort by confidence and find highest
            std::sort(current_result_->dependency_chain.begin(),
                     current_result_->dependency_chain.end(),
                     [](const AgentDependency& a, const AgentDependency& b) {
                         return a.confidence_score > b.confidence_score;
                     });

            current_result_->primary_root_cause = current_result_->dependency_chain[0].failure_mode;

            // Calculate overall analysis confidence as average
            float total_confidence = 0.0f;
            for (const auto& dep : current_result_->dependency_chain) {
                total_confidence += dep.confidence_score;
            }
            current_result_->analysis_confidence = total_confidence / current_result_->dependency_chain.size();

            // Generate mitigation recommendations
            current_result_->recommendations = generate_recommendations(
                current_result_->primary_root_cause,
                current_result_->dependency_chain
            );
        } else {
            current_result_->analysis_confidence = 0.0f;
            current_result_->primary_root_cause = "No dependencies identified";
        }

    } catch (const std::exception& e) {
        // Handle any unexpected errors
        current_result_->primary_root_cause = "Analysis failed: " + std::string(e.what());
        current_result_->analysis_confidence = 0.0f;
        current_result_->analysis_complete = false;
    }

    return *current_result_;
}

bool RootCauseAnalyzer::detect_cycles(const AdjacencyList& graph) {
    // Simple cycle detection using DFS
    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> recursion_stack;

    auto has_cycle_helper = [&](auto& self, const std::string& node) -> bool {
        if (recursion_stack.count(node)) return true;
        if (visited.count(node)) return false;

        visited.insert(node);
        recursion_stack.insert(node);

        auto it = graph.find(node);
        if (it != graph.end()) {
            for (const auto& link : it->second) {
                if (self(self, link.target_agent)) {
                    return true;
                }
            }
        }

        recursion_stack.erase(node);
        return false;
    };

    for (const auto& [node, _] : graph) {
        if (has_cycle_helper(has_cycle_helper, node)) {
            return true;
        }
    }

    return false;
}

void RootCauseAnalyzer::reset_tracking() {
    visited_nodes_.clear();
    while (!recursion_stack_.empty()) {
        recursion_stack_.pop();
    }
    dependency_depths_.clear();
    current_memory_usage_ = 0;
}

AnalysisResult RootCauseAnalyzer::analyze_recursive(
    const std::string& current_agent,
    std::vector<std::string> symptoms,
    int depth
) {
    // Safety checks - these are the recursive loop protections the user requested
    if (has_cycle()) {
        return AnalysisResult::CYCLE_DETECTED;
    }

    if (depth > config_.max_recursion_depth) {
        return AnalysisResult::DEPTH_LIMIT_EXCEEDED;
    }

    if (has_timed_out()) {
        throw AnalysisTimeoutException();
    }

    if (memory_limit_exceeded()) {
        return AnalysisResult::MEMORY_LIMIT_EXCEEDED;
    }

    // Mark as visited to prevent revisits
    visited_nodes_.insert(current_agent);
    recursion_stack_.push(current_agent);
    dependency_depths_[current_agent] = depth;

    try {
        // RAII pattern - stack will be popped when function exits
        auto stack_guard = [&]() noexcept {
            recursion_stack_.pop();
        };
        auto cleanup = std::unique_ptr<void, decltype(stack_guard)>((void*)1, stack_guard);

        // Identify failure mode for this agent
        std::string failure_mode = identify_failure_mode(symptoms, current_agent);

        // Calculate confidence in this agent's role
        auto graph_it = current_graph_->find(current_agent);
        std::vector<DependencyLink> dependencies;
        if (graph_it != current_graph_->end()) {
            dependencies = graph_it->second;
        }

        float confidence = calculate_confidence_score(symptoms, dependencies);

        // Add to dependency chain if confidence is sufficient
        if (confidence >= config_.min_confidence_threshold) {
            add_to_dependency_chain(current_agent, confidence, failure_mode, symptoms);

            // Check if we've hit the maximum chain length
            if (current_result_->dependency_chain.size() >= static_cast<size_t>(config_.max_dependency_chain)) {
                return AnalysisResult::SUCCESS;  // Success with truncated chain
            }
        }

        // Recursive exploration of dependencies - this is the recursive testing loop
        if (graph_it != current_graph_->end()) {
            for (const auto& dependency : graph_it->second) {
                if (visited_nodes_.find(dependency.target_agent) == visited_nodes_.end()) {
                    // Propagate symptoms to dependent agent (possibly modified)
                    std::vector<std::string> propagated_symptoms = symptoms;
                    propagated_symptoms.push_back("Dependency failure: " + current_agent);

                    AnalysisResult dep_result = analyze_recursive(
                        dependency.target_agent,
                        propagated_symptoms,
                        depth + 1
                    );

                    // Handle different recursive results
                    if (dep_result != AnalysisResult::SUCCESS) {
                        // Could be cycle, timeout, etc. - continue with other dependencies
                        continue;
                    }
                }
            }
        }

        return AnalysisResult::SUCCESS;

    } catch (...) {
        return AnalysisResult::INVALID_INPUT;
    }
}

bool RootCauseAnalyzer::has_cycle() const {
    // Check if current node is already in recursion stack
    std::string current_node = recursion_stack_.top();
    std::stack<std::string> temp_stack = recursion_stack_;

    while (!temp_stack.empty()) {
        if (temp_stack.top() == current_node && temp_stack.size() > 1) {
            return true;  // Found duplicate below top (indicates cycle)
        }
        temp_stack.pop();
    }

    return false;
}

bool RootCauseAnalyzer::has_timed_out() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - analysis_start_time_
    );
    return elapsed > config_.timeout;
}

bool RootCauseAnalyzer::memory_limit_exceeded() const {
    size_t estimated_usage = estimate_memory_usage();
    size_t max_bytes = config_.max_memory_mb * 1024 * 1024;  // Convert MB to bytes
    return estimated_usage > max_bytes;
}

void RootCauseAnalyzer::add_to_dependency_chain(
    const std::string& agent_id,
    float confidence,
    const std::string& failure_mode,
    const std::vector<std::string>& symptoms
) {
    if (current_result_) {
        current_result_->dependency_chain.push_back({
            agent_id, confidence, failure_mode, symptoms
        });
    }

    // Update memory usage estimate
    current_memory_usage_ += sizeof(AgentDependency) +
                            agent_id.capacity() +
                            failure_mode.capacity();
    for (const auto& symptom : symptoms) {
        current_memory_usage_ += symptom.capacity();
    }
}

std::string RootCauseAnalyzer::identify_failure_mode(
    const std::vector<std::string>& symptoms,
    const std::string& agent_id
) const {
    // Pattern matching against symptoms
    for (const auto& [failure_mode, patterns] : failure_patterns_) {
        for (const auto& symptom : symptoms) {
            for (const auto& pattern : patterns) {
                if (symptom.find(pattern) != std::string::npos) {
                    return failure_mode;
                }
            }
        }
    }

    // Default based on agent ID if no patterns match
    if (agent_id.find("ollama") != std::string::npos ||
        symptoms.end() != std::find_if(symptoms.begin(), symptoms.end(),
            [](const std::string& s) { return s.find("ollama") != std::string::npos; })) {
        return "ollama_failure";
    }

    return "unknown_failure";
}

float RootCauseAnalyzer::calculate_confidence_score(
    const std::vector<std::string>& symptoms,
    const std::vector<DependencyLink>& dependencies
) const {
    if (symptoms.empty()) return 0.0f;

    float confidence = 0.0f;

    // Base confidence from symptoms
    confidence += std::min(0.4f, static_cast<float>(symptoms.size()) * 0.1f);

    // Add confidence based on dependency patterns
    for (const auto& dep : dependencies) {
        if (dep.strength > 0.5f) {
            confidence += 0.2f;  // Strong dependencies indicate higher importance
        }
    }

    // Normalize to 0-1 range
    return std::min(1.0f, confidence);
}

std::vector<MitigationRecommendation> RootCauseAnalyzer::generate_recommendations(
    const std::string& failure_mode,
    const std::vector<AgentDependency>& dependency_chain
) const {
    std::vector<MitigationRecommendation> recommendations;

    // Generate recommendations based on failure mode and chain
    if (failure_mode == "ollama_overload") {
        recommendations.push_back({
            "Deploy separate Ollama instances per GPU (ports 11434-11437)",
            "HIGH",
            "Current single instance cannot handle 100 concurrent agents",
            80.0f
        });
        recommendations.push_back({
            "Implement request queuing with priority classification",
            "MEDIUM",
            "Reduces memory pressure and provides better service isolation",
            60.0f
        });
    }

    if (failure_mode == "gossip_isolation") {
        recommendations.push_back({
            "Implement p2p messaging layer with gossip_hops:4 and fanout:5",
            "HIGH",
            "Agents cannot discover peers for coordinated behavior",
            90.0f
        });
        recommendations.push_back({
            "Add neighbor discovery bootstrap from entry nodes",
            "MEDIUM",
            "Ensures initial connectivity for isolated GPUs",
            70.0f
        });
    }

    if (failure_mode == "memory_cascade") {
        recommendations.push_back({
            "Reduce gpu_memory_fraction from 0.95 to 0.85",
            "HIGH",
            "Prevents GPU OOM on single agent memory spikes",
            85.0f
        });
        recommendations.push_back({
            "Implement memory quota enforcement per agent",
            "HIGH",
            "Direct per-agent memory limits prevent cascading failures",
            75.0f
        });
    }

    if (failure_mode == "timeout_failure") {
        recommendations.push_back({
            "Increase request timeouts from 30s to 60s",
            "MEDIUM",
            "Reduces timeout failures in high-latency scenarios",
            55.0f
        });
        recommendations.push_back({
            "Implement exponential backoff retry logic",
            "MEDIUM",
            "Provides resilience to transient network issues",
            65.0f
        });
    }

    // Add general recommendations based on chain length
    if (dependency_chain.size() > 10) {
        recommendations.push_back({
            "Implement circuit breaker pattern for cascading failures",
            "HIGH",
            "Prevents system-wide failure cascades",
            80.0f
        });
    }

    // Sort by expected impact
    std::sort(recommendations.begin(), recommendations.end(),
              [](const MitigationRecommendation& a, const MitigationRecommendation& b) {
                  return a.expected_impact > b.expected_impact;
              });

    return recommendations;
}

void RootCauseAnalyzer::initialize_failure_patterns() {
    failure_patterns_["ollama_failure"] = {
        "ollama", "query failed", "model not found", "out of memory"
    };

    failure_patterns_["gossip_failure"] = {
        "no neighbors", "gossip timeout", "peer discovery failed"
    };

    failure_patterns_["memory_failure"] = {
        "out of memory", "cuda error", "allocation failed", "gpu memory"
    };

    failure_patterns_["timeout_failure"] = {
        "timed out", "deadline exceeded", "response timeout"
    };

    failure_patterns_["network_failure"] = {
        "connection refused", "network unreachable", "socket error"
    };

    failure_patterns_["consensus_failure"] = {
        "consensus failed", "quorum not reached", "leader election"
    };
}

size_t RootCauseAnalyzer::estimate_memory_usage() const {
    size_t usage = sizeof(*this);  // Base object

    // Estimate memory for containers
    usage += visited_nodes_.size() * (sizeof(std::string) + 32);  // String overhead

    for (const auto& [key, value] : dependency_depths_) {
        usage += key.capacity() + sizeof(int);
    }

    // Recursion stack approximation
    usage += recursion_stack_.size() * sizeof(std::string*);

    // Current result approximation
    if (current_result_) {
        usage += sizeof(RootCauseResult);
        for (const auto& dep : current_result_->dependency_chain) {
            usage += sizeof(AgentDependency) + dep.agent_id.capacity() +
                    dep.failure_mode.capacity();
            for (const auto& symptom : dep.symptoms) {
                usage += symptom.capacity();
            }
        }
    }

    return usage;
}

// Utility function implementations
std::string analysis_result_to_string(AnalysisResult result) {
    switch (result) {
        case AnalysisResult::SUCCESS: return "SUCCESS";
        case AnalysisResult::CYCLE_DETECTED: return "CYCLE_DETECTED";
        case AnalysisResult::DEPTH_LIMIT_EXCEEDED: return "DEPTH_LIMIT_EXCEEDED";
        case AnalysisResult::TIMEOUT_EXCEEDED: return "TIMEOUT_EXCEEDED";
        case AnalysisResult::MEMORY_LIMIT_EXCEEDED: return "MEMORY_LIMIT_EXCEEDED";
        case AnalysisResult::INVALID_INPUT: return "INVALID_INPUT";
        default: return "UNKNOWN";
    }
}

std::string format_root_cause_result(const RootCauseResult& result) {
    std::stringstream ss;

    ss << "Root Cause Analysis Result:\n";
    ss << "Primary Root Cause: " << result.primary_root_cause << "\n";
    ss << "Analysis Confidence: " << result.analysis_confidence << "\n";
    ss << "Analysis Duration: " << result.analysis_duration.count() << "ms\n";
    ss << "Analysis Complete: " << (result.analysis_complete ? "YES" : "NO") << "\n";

    ss << "\nDependency Chain (" << result.dependency_chain.size() << " agents):\n";
    for (size_t i = 0; i < result.dependency_chain.size(); ++i) {
        const auto& dep = result.dependency_chain[i];
        ss << i+1 << ". Agent " << dep.agent_id
           << " - " << dep.failure_mode
           << " (confidence: " << dep.confidence_score
           << ", symptoms: " << dep.symptoms.size() << ")\n";
    }

    ss << "\nMitigation Recommendations:\n";
    for (const auto& rec : result.recommendations) {
        ss << "- " << rec.priority << ": " << rec.action
           << " (expected impact: " << rec.expected_impact << "%)\n";
    }

    return ss.str();
}
