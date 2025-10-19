#include "emergence_analyzer.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <queue>

// EmergenceAnalyzer Implementation

EmergenceAnalyzer::EmergenceAnalyzer(const CyberGrid& grid)
    : grid_(grid),
      current_generation_(0),
      coarse_width_(grid.width() / COARSE_GRAIN_FACTOR),
      coarse_height_(grid.height() / COARSE_GRAIN_FACTOR) {

    // Reserve space for history buffers
    life_entropy_history_.reserve(ENTROPY_WINDOW_SIZE + 100);
    energy_entropy_history_.reserve(ENTROPY_WINDOW_SIZE + 100);
    coherence_history_.reserve(ENTROPY_WINDOW_SIZE + 100);

    metadata_history_.reserve(5000);  // For 1000+ generation runs
}

void EmergenceAnalyzer::analyze_generation() {
    current_generation_++;

    // Calculate current entropies
    double life_entropy = calculate_life_entropy();
    double energy_entropy = calculate_energy_entropy();
    double ecr = calculate_entropy_to_coherence_ratio();

    // Update rolling histories
    life_entropy_history_.push_back(life_entropy);
    energy_entropy_history_.push_back(energy_entropy);
    coherence_history_.push_back(ecr);

    // Maintain window size
    if (life_entropy_history_.size() > ENTROPY_WINDOW_SIZE) {
        life_entropy_history_.erase(life_entropy_history_.begin());
        energy_entropy_history_.erase(energy_entropy_history_.begin());
        coherence_history_.erase(coherence_history_.begin());
    }

    // Detect patterns and oscillators
    detect_oscillators();
    track_pattern_persistence();

    // Update coarse-grained analysis for macroscopic patterns
    update_coarse_grained_analysis();
}

double EmergenceAnalyzer::calculate_life_entropy() const {
    // Calculate Shannon entropy of alive/dead cell distribution
    double alive_count = 0.0;
    int total_cells = grid_.width() * grid_.height();

    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            if (grid_.get_cell(x, y).alive) {
                alive_count += 1.0;
            }
        }
    }

    if (alive_count == 0.0 || alive_count == static_cast<double>(total_cells)) {
        return 0.0;  // No entropy in uniform states
    }

    double p_alive = alive_count / total_cells;
    double p_dead = 1.0 - p_alive;

    return -p_alive * std::log2(p_alive) - p_dead * std::log2(p_dead);
}

double EmergenceAnalyzer::calculate_energy_entropy() const {
    // Calculate entropy of energy distribution (binned)
    const int ENERGY_BINS = 16;
    std::vector<int> bins(ENERGY_BINS, 0);
    int total_cells = grid_.width() * grid_.height();

    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            int bin = static_cast<int>(grid_.get_cell(x, y).energy * (ENERGY_BINS - 1));
            bin = std::clamp(bin, 0, ENERGY_BINS - 1);
            bins[bin]++;
        }
    }

    double entropy = 0.0;
    for (int count : bins) {
        if (count > 0) {
            double p = static_cast<double>(count) / total_cells;
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

double EmergenceAnalyzer::calculate_entropy_to_coherence_ratio() const {
    // ECR: measure of order formation vs randomness
    // Lower ECR indicates more coherent/patterned behavior
    double life_entropy = calculate_life_entropy();
    double energy_entropy = calculate_energy_entropy();

    // Calculate spatial coherence (inverse of conditional entropy between energy and life)
    double spatial_coherence = calculate_spatial_coherence();

    // ECR = entropy_sum / (1 + coherence) - approaches 0 when coherent
    double entropy_sum = life_entropy + energy_entropy;
    return entropy_sum / std::max(1.0 + spatial_coherence, 1e-6);
}

double EmergenceAnalyzer::calculate_spatial_coherence() const {
    // Measure how well energy and life patterns correlate spatially
    // Higher coherence = more organized emergence
    double correlation_sum = 0.0;
    int sample_count = 0;

    // Sample correlation across different spatial scales
    for (int scale = 1; scale <= 3; ++scale) {
        for (int y = scale; y < grid_.height() - scale; y += scale * 2) {
            for (int x = scale; x < grid_.width() - scale; x += scale * 2) {
                // Calculate local pattern correlation
                double local_energy_variance = 0.0;
                double local_life_energy_correlation = 0.0;
                int local_cells = 0;

                // Sample neighborhood
                for (int dy = -scale; dy <= scale; ++dy) {
                    for (int dx = -scale; dx <= scale; ++dx) {
                        const auto& cell = grid_.get_cell(x + dx, y + dy);
                        local_energy_variance += cell.energy * cell.energy;
                        local_life_energy_correlation += (cell.alive ? 1.0 : 0.0) * cell.energy;
                        local_cells++;
                    }
                }

                if (local_cells > 0) {
                    local_energy_variance /= local_cells;
                    local_life_energy_correlation /= local_cells;

                    // Add to correlation sum (normalized)
                    correlation_sum += std::abs(local_life_energy_correlation) / std::max(std::sqrt(local_energy_variance), 1e-6);
                    sample_count++;
                }
            }
        }
    }

    return sample_count > 0 ? correlation_sum / sample_count : 0.0;
}

void EmergenceAnalyzer::detect_oscillators() {
    active_oscillators_.clear();

    // Simple oscillator detection: look for stable 2-3 period patterns
    // This is a simplified version - full oscillator census would need state history
    for (int y = 1; y < grid_.height() - 1; ++y) {
        for (int x = 1; x < grid_.width() - 1; ++x) {
            // Check for potential oscillator by looking at local activity
            int neighbor_count = 0;
            float avg_neighbor_energy = 0.0f;

            // Moore neighborhood
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const auto& neighbor = grid_.get_cell(x + dx, y + dy);
                    if (neighbor.alive) neighbor_count++;
                    avg_neighbor_energy += neighbor.energy;
                }
            }
            avg_neighbor_energy /= 8.0f;

            // Heuristic: cells in oscillator-like configurations
            const auto& cell = grid_.get_cell(x, y);
            if (neighbor_count >= 2 && neighbor_count <= 3 &&
                cell.energy > 0.5f && avg_neighbor_energy > 0.3f) {
                // Potential oscillator activity
                std::string signature = generate_cell_pattern_signature(x, y, 2);
                pattern_persistence_[signature]++;
            }
        }
    }

    // Identify persistently active regions as oscillators
    for (const auto& pair : pattern_persistence_) {
        if (pair.second > 5) {  // Persisted for multiple generations
            // Extract coordinates from signature (simplified: just track count)
            // In full implementation, we'd decode position from signature
            // For now, just track count of persistent patterns
        }
    }
}

void EmergenceAnalyzer::track_pattern_persistence() {
    // Decay persistence counts and clean up old patterns
    for (auto it = pattern_persistence_.begin(); it != pattern_persistence_.end(); ) {
        it->second *= 0.95;  // Exponential decay
        if (it->second < 1.0) {
            it = pattern_persistence_.erase(it);
        } else {
            ++it;
        }
    }

    // Sample current patterns for persistence tracking
    for (int y = 2; y < grid_.height() - 2; y += 4) {
        for (int x = 2; x < grid_.width() - 2; x += 4) {
            std::string signature = generate_cell_pattern_signature(x, y, 2);
            if (!signature.empty()) {
                pattern_persistence_[signature] += 1.0;
            }
        }
    }
}

std::string EmergenceAnalyzer::generate_cell_pattern_signature(int x, int y, int radius) const {
    // Generate a compact signature of the cell pattern around (x,y)
    std::string signature;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const auto& cell = grid_.get_cell(x + dx, y + dy);
            char life_char = cell.alive ? '1' : '0';
            char energy_char = static_cast<char>('A' + static_cast<int>(cell.energy * 25));  // 26 levels
            signature += life_char;
            signature += energy_char;
        }
    }

    return signature;
}

void EmergenceAnalyzer::update_coarse_grained_analysis() {
    // Coarse graining already calculated in get_coarse_grained_energy_field()
    // Could cache this if performance becomes an issue
}

std::vector<std::vector<double>> EmergenceAnalyzer::get_coarse_grained_energy_field() const {
    std::vector<std::vector<double>> coarse_field(coarse_height_,
                                                  std::vector<double>(coarse_width_, 0.0));

    for (int cy = 0; cy < coarse_height_; ++cy) {
        for (int cx = 0; cx < coarse_width_; ++cx) {
            // Average energy in this coarse cell
            double sum = 0.0;
            int count = 0;

            for (int fy = cy * COARSE_GRAIN_FACTOR;
                 fy < std::min((cy + 1) * COARSE_GRAIN_FACTOR, grid_.height()); ++fy) {
                for (int fx = cx * COARSE_GRAIN_FACTOR;
                     fx < std::min((cx + 1) * COARSE_GRAIN_FACTOR, grid_.width()); ++fx) {
                    sum += grid_.get_cell(fx, fy).energy;
                    count++;
                }
            }

            coarse_field[cy][cx] = count > 0 ? sum / count : 0.0;
        }
    }

    return coarse_field;
}

bool EmergenceAnalyzer::detect_emergence_convergence(double threshold) const {
    if (coherence_history_.size() < 10) return false;

    // Check if ECR has stabilized (convergence to coherent state)
    size_t window_size = std::min<size_t>(20, coherence_history_.size());
    std::vector<double> recent_window(coherence_history_.end() - window_size,
                                      coherence_history_.end());

    double recent_avg = calculate_running_average(recent_window);
    double recent_std = calculate_standard_deviation(recent_window);

    // Converged if coefficient of variation is low (stable ECR)
    double cv = recent_std / std::max(recent_avg, 1e-6);
    return cv < threshold;
}

double EmergenceAnalyzer::measure_system_stability() const {
    if (coherence_history_.size() < 5) return 0.0;

    // Stability = inverse of ECR variance over recent generations
    double avg_ecr = calculate_running_average(coherence_history_);
    double std_ecr = calculate_standard_deviation(coherence_history_);

    // Higher stability = more ordered/less chaotic system
    return std_ecr > 0.0 ? 1.0 / (1.0 + std_ecr) : 1.0;
}

void EmergenceAnalyzer::record_generation_metadata(double tick_duration_ms, double timing_error) {
    GenerationMetadata metadata{
        .tick_count = current_generation_,
        .timestamp = std::chrono::steady_clock::now(),
        .actual_tick_duration_ms = tick_duration_ms,
        .timing_error_accumulator = timing_error,
        .alive_cell_count = 0,
        .average_energy = grid_.average_energy()
    };

    // Count alive cells
    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            if (grid_.get_cell(x, y).alive) {
                metadata.alive_cell_count++;
            }
        }
    }

    metadata_history_.push_back(metadata);

    // Maintain reasonable history size
    if (metadata_history_.size() > 5000) {
        metadata_history_.erase(metadata_history_.begin());
    }
}

const std::vector<EmergenceAnalyzer::GenerationMetadata>& EmergenceAnalyzer::get_metadata_history() const {
    return metadata_history_;
}

std::unordered_map<std::string, double> EmergenceAnalyzer::get_current_metrics() const {
    return {
        {"generation", static_cast<double>(current_generation_)},
        {"life_entropy", calculate_life_entropy()},
        {"energy_entropy", calculate_energy_entropy()},
        {"entropy_coherence_ratio", calculate_entropy_to_coherence_ratio()},
        {"spatial_coherence", calculate_spatial_coherence()},
        {"oscillator_count", static_cast<double>(get_oscillator_count())},
        {"system_stability", measure_system_stability()},
        {"alive_cells", static_cast<double>(grid_.alive_cell_count())}
    };
}

std::string EmergenceAnalyzer::generate_emergence_report() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);

    auto metrics = get_current_metrics();

    ss << "=== Swarm Intelligence Emergence Report ===\n";
    ss << "Generation: " << metrics["generation"] << "\n";
    ss << "Life Entropy: " << metrics["life_entropy"] << "\n";
    ss << "Energy Entropy: " << metrics["energy_entropy"] << "\n";
    ss << "Entropy-to-Coherence Ratio (ECR): " << metrics["entropy_coherence_ratio"] << "\n";
    ss << "Spatial Coherence: " << metrics["spatial_coherence"] << "\n";
    ss << "Active Oscillators: " << metrics["oscillator_count"] << "\n";
    ss << "System Stability: " << metrics["system_stability"] << "\n";
    ss << "Alive Cells: " << metrics["alive_cells"] << "\n";
    ss << "Emergence Convergence: " << (detect_emergence_convergence() ? "Yes" : "No") << "\n";

    return ss.str();
}

bool EmergenceAnalyzer::run_emergence_benchmark(int generations, const std::string& output_path) {
    std::ofstream logfile(output_path + "/emergence_log.csv");
    if (!logfile.is_open()) return false;

    // CSV header
    logfile << "generation,life_entropy,energy_entropy,ecr,coherence,oscillators,stability,alive_cells,tick_duration_ms,timing_error\n";

    // Need access to grid stepping - this would need to be connected to the step loop
    // For now, return false as this requires integration with the simulation loop
    logfile.close();
    return false;  // Requires external loop integration
}

// Statistical utilities
double EmergenceAnalyzer::calculate_running_average(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double EmergenceAnalyzer::calculate_standard_deviation(const std::vector<double>& values) const {
    if (values.size() <= 1) return 0.0;

    double mean = calculate_running_average(values);
    double variance = 0.0;

    for (double val : values) {
        double diff = val - mean;
        variance += diff * diff;
    }

    return std::sqrt(variance / (values.size() - 1));
}

// Not implemented helper methods (these are referenced but not critical for basic operation)
// These would be implemented for full functionality
double EmergenceAnalyzer::calculate_shannon_entropy(const std::vector<double>& probabilities) const {
    double entropy = 0.0;
    for (double p : probabilities) {
        if (p > 0.0) {
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

double EmergenceAnalyzer::calculate_conditional_entropy(int x1, int y1, int x2, int y2) const {
    // Simplified implementation - would need full conditional probability calculations
    return 0.0;
}

std::vector<double> EmergenceAnalyzer::get_cell_value_distribution(bool use_energy) const {
    // Not implemented for this basic version
    return std::vector<double>();
}

void EmergenceAnalyzer::initialize_coarse_graining() {
    // Implementation handled in constructor
}

std::vector<std::vector<double>> EmergenceAnalyzer::coarse_grain_field(const std::vector<std::vector<double>>& fine_field) const {
    // Basic implementation - delegate to get_coarse_grained_energy_field
    return get_coarse_grained_energy_field();
}

// EmergentBehaviorBenchmarkSuite Implementation

EmergentBehaviorBenchmarkSuite::EmergentBehaviorBenchmarkSuite()
    : max_generations_(1000),
      repetitions_per_config_(5),
      output_directory_("emergence_benchmarks") {
}

void EmergentBehaviorBenchmarkSuite::add_grid_configuration(int width, int height, float alive_prob, float energy_prob) {
    std::string config_name = generate_config_name(width, height, alive_prob, energy_prob);
    grids_.push_back(std::make_unique<CyberGrid>(width, height));
    grids_.back()->randomize(alive_prob, energy_prob);

    analyzers_.push_back(create_analyzer_for_grid(*grids_.back()));
}

void EmergentBehaviorBenchmarkSuite::set_parameters(int max_generations, int repetitions, const std::string& output_dir) {
    max_generations_ = max_generations;
    repetitions_per_config_ = repetitions;
    output_directory_ = output_dir;
}

void EmergentBehaviorBenchmarkSuite::run_full_benchmark_suite() {
    if (grids_.empty()) {
        // Add default configurations for comprehensive testing
        add_grid_configuration(50, 50, 0.3f, 0.2f);     // Balanced emergence
        add_grid_configuration(100, 100, 0.1f, 0.5f);   // Sparse with high energy
        add_grid_configuration(100, 100, 0.5f, 0.1f);   // Dense with low energy
        add_grid_configuration(50, 50, 0.01f, 0.8f);    // Very sparse, high energy pulses
    }

    results_.clear();

    for (size_t config_idx = 0; config_idx < grids_.size(); ++config_idx) {
        const auto& grid = grids_[config_idx];
        const auto& analyzer = analyzers_[config_idx];
        std::string config_name = generate_config_name(
            grid->width(), grid->height(),
            0.3f, 0.2f  // Simplified - would need to store original params
        );

        // Run multiple repetitions for statistical significance
        for (int rep = 0; rep < repetitions_per_config_; ++rep) {
            BenchmarkResult result;
            result.config_name = config_name + "_rep" + std::to_string(rep + 1);

            // Reset grid for each repetition
            grid->randomize(0.3f, 0.2f);  // Simplified - use stored params
            bool emergence_found = run_single_emergence_test(*grid, max_generations_);

            // Analyze final state
            result.emergence_detected = emergence_found;
            result.emergence_generation = emergence_found ? max_generations_ : -1;
            result.average_ecr = analyzer->calculate_entropy_to_coherence_ratio();
            result.final_oscillator_count = analyzer->get_oscillator_count();
            result.final_stability_score = analyzer->measure_system_stability();
            // ECR trajectory would need continuous tracking

            results_.push_back(result);
        }
    }

    // Save comprehensive report
    generate_emergence_report(output_directory_ + "/benchmark_results.md");
}

void EmergentBehaviorBenchmarkSuite::generate_emergence_report(const std::string& report_path) {
    std::ofstream report_file(report_path);
    if (!report_file.is_open()) return;

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    report_file << "# Swarm Intelligence Emergence Benchmark Results\n\n";
    report_file << "Generated: " << std::ctime(&time_t) << "\n";

    report_file << "## Summary Statistics\n";
    report_file << "- Total Configurations Tested: " << results_.size() / repetitions_per_config_ << "\n";
    report_file << "- Emergence Success Rate: " << std::fixed << std::setprecision(1) <<
                   calculate_emergence_success_rate() * 100 << "%\n\n";

    auto stats = get_aggregate_statistics();
    report_file << "## Aggregate Metrics\n";
    for (const auto& [key, value] : stats) {
        report_file << "- " << key << ": " << std::fixed << std::setprecision(4) << value << "\n";
    }
    report_file << "\n";

    report_file << "## Detailed Results\n";
    report_file << "| Configuration | Emergence | Generation | Avg ECR | Oscillators | Stability |\n";
    report_file << "|--------------|-----------|------------|---------|------------|-----------|\n";

    for (const auto& result : results_) {
        report_file << "| " << result.config_name << " | " <<
                   (result.emergence_detected ? "✅" : "❌") << " | " <<
                   result.emergence_generation << " | " <<
                   std::fixed << std::setprecision(3) << result.average_ecr << " | " <<
                   result.final_oscillator_count << " | " <<
                   result.final_stability_score << " |\n";
    }

    report_file.close();
}

double EmergentBehaviorBenchmarkSuite::calculate_emergence_success_rate() const {
    if (results_.empty()) return 0.0;
    int success_count = std::count_if(results_.begin(), results_.end(),
                                     [](const BenchmarkResult& r) { return r.emergence_detected; });
    return static_cast<double>(success_count) / results_.size();
}

std::unordered_map<std::string, double> EmergentBehaviorBenchmarkSuite::get_aggregate_statistics() const {
    if (results_.empty()) return {};

    double avg_ecr = 0.0;
    double avg_oscillators = 0.0;
    double avg_stability = 0.0;

    for (const auto& result : results_) {
        avg_ecr += result.average_ecr;
        avg_oscillators += result.final_oscillator_count;
        avg_stability += result.final_stability_score;
    }

    int count = results_.size();
    return {
        {"Average ECR", avg_ecr / count},
        {"Average Oscillators", avg_oscillators / count},
        {"Average Stability", avg_stability / count}
    };
}



std::unique_ptr<EmergenceAnalyzer> EmergentBehaviorBenchmarkSuite::create_analyzer_for_grid(const CyberGrid& grid) {
    return std::make_unique<EmergenceAnalyzer>(grid);
}

std::string EmergentBehaviorBenchmarkSuite::generate_config_name(int width, int height, float alive_prob, float energy_prob) const {
    std::stringstream ss;
    ss << width << "x" << height << "_alive" << std::fixed << std::setprecision(1) << alive_prob *
                                      100 << "_energy" << energy_prob * 100;
    return ss.str();
}

void EmergentBehaviorBenchmarkSuite::save_trajectory_data(const BenchmarkResult& result, const std::string& filename) const {
    // Implementation would save ECR trajectories and other time series data
    // Not implemented in this basic version
}

// ResilientEmergenceBenchmarker Implementation

ResilientEmergenceBenchmarker::ResilientEmergenceBenchmarker(CyberGrid& grid)
    : grid_(&grid) {
    active_agents_.clear();
    agent_trust_scores_.clear();
    resilience_history_.clear();
}

void ResilientEmergenceBenchmarker::register_agents(const std::vector<std::string>& agent_ids) {
    active_agents_ = agent_ids;

    // Initialize trust scores to default (0.5)
    agent_trust_scores_.clear();
    for (const auto& agent_id : agent_ids) {
        agent_trust_scores_[agent_id] = 0.5f;
        grid_->register_rover_agent(agent_id);  // Register as potential rover
    }
}

void ResilientEmergenceBenchmarker::update_agent_trust(const std::string& agent_id, float trust_score) {
    agent_trust_scores_[agent_id] = std::clamp(trust_score, 0.0f, 1.0f);
}

ResilientEmergenceBenchmarker::ResilienceMetrics
ResilientEmergenceBenchmarker::run_resilience_simulation(
    int agent_count,
    int target_generations,
    const std::vector<FaultInjectionPattern>& fault_patterns) {

    // Setup initial agent distribution
    std::vector<std::string> agent_ids;
    for (int i = 0; i < agent_count; ++i) {
        agent_ids.push_back("agent_" + std::to_string(i));
    }
    register_agents(agent_ids);

    // Initialize metrics tracking
    ResilienceMetrics metrics = {};
    std::unordered_map<std::string, int> reconnect_times;
    std::unordered_map<std::string, bool> survived_simulation;

    for (const auto& agent_id : active_agents_) {
        reconnect_times[agent_id] = 0;
        survived_simulation[agent_id] = true;
    }

    // Run simulation with fault injection
    for (long long gen = 0; gen < target_generations; ++gen) {
        // Apply active fault patterns
        for (const auto& pattern : fault_patterns) {
            if (gen >= pattern.start_generation &&
                gen < pattern.start_generation + pattern.duration_generations) {
                simulate_generation_with_faults(pattern);
            }
        }

        // Simulate grid evolution (simplified - would integrate with actual step())
        if (grid_) {
            // Would call grid_->step() here in full implementation
        }

        // Update resilience snapshot
        update_resilience_snapshot(gen);

        // Update agent state based on faults
        auto failed_agents = grid_->identify_failed_agents(1000);
        for (const auto& failed_agent : failed_agents) {
            if (survived_simulation.count(failed_agent)) {
                int current_failures = reconnect_times[failed_agent];
                if (current_failures == 0) {
                    // First failure - start counting
                    reconnect_times[failed_agent] = 1;
                }
                // Agent still "survives" until full disconnection
            }
        }
    }

    return calculate_final_metrics();
}

void ResilientEmergenceBenchmarker::inject_network_partition(int isolation_duration, float affected_percentage) {
    int total_agents = active_agents_.size();
    int affected_count = static_cast<int>(total_agents * affected_percentage);

    if (affected_count > 0) {
        auto isolated_agents = select_random_agents(affected_count);

        // Simulate network isolation
        for (const auto& agent_id : isolated_agents) {
            // In full implementation, would temporarily disable communication
            // For simulation, mark as failed
            update_agent_trust(agent_id, 0.0f);

            // After duration, restore (would be timed in real simulation)
            update_agent_trust(agent_id, 0.5f);  // Restore partial trust
        }
    }
}

void ResilientEmergenceBenchmarker::inject_heartbeat_failure(const std::vector<std::string>& affected_agents, float drop_probability) {
    for (const auto& agent_id : affected_agents) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        if (dis(rng) < drop_probability) {
            // Fail to send heartbeat
            update_agent_trust(agent_id, agent_trust_scores_[agent_id] * 0.8f);
        }
    }
}

void ResilientEmergenceBenchmarker::inject_trust_corruption(float corruption_intensity) {
    for (auto& [agent_id, trust] : agent_trust_scores_) {
        // Random corruption based on intensity
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<> corruption(0.0, corruption_intensity);

        float corruption_amount = corruption(rng);
        trust = std::clamp(trust + corruption_amount, 0.0f, 1.0f);
    }
}

void ResilientEmergenceBenchmarker::inject_pulse_interference(int x, int y, int radius, float interference_strength) {
    // Simulate LoRA interference in grid region
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < grid_->width() && ny >= 0 && ny < grid_->height()) {
                auto& cell = grid_->get_cell(nx, ny);
                cell.reduce_energy(interference_strength);

                // Check if any agents in this cell are affected
                for (const auto& agent_id : cell.occupants) {
                    update_agent_trust(agent_id, agent_trust_scores_[agent_id] * 0.9f);
                }
            }
        }
    }
}

std::vector<std::string> ResilientEmergenceBenchmarker::get_failed_agents() const {
    return grid_->identify_failed_agents(1000);
}

double ResilientEmergenceBenchmarker::calculate_current_connectivity() const {
    double connected_pairs = 0.0;
    double total_possible_pairs = 0.0;

    // Calculate fraction of agent pairs that can communicate
    // Simplified: assume agents within adaptive range can communicate
    for (size_t i = 0; i < active_agents_.size(); ++i) {
        for (size_t j = i + 1; j < active_agents_.size(); ++j) {
            total_possible_pairs += 1.0;

            try {
                auto pos1 = grid_->get_agent_position(active_agents_[i]);
                auto pos2 = grid_->get_agent_position(active_agents_[j]);

                float distance = grid_->toroidal_distance(pos1.first, pos1.second, pos2.first, pos2.second);
                float adaptive_range = grid_->get_adaptive_pulse_range(pos1.first, pos1.second);

                if (distance <= adaptive_range) {
                    connected_pairs += 1.0;
                }
            } catch (const std::exception&) {
                // Agent not found - cannot communicate
            }
        }
    }

    return total_possible_pairs > 0 ? connected_pairs / total_possible_pairs : 0.0;
}

double ResilientEmergenceBenchmarker::calculate_trust_entropy() const {
    if (agent_trust_scores_.empty()) return 0.0;

    // Calculate entropy of trust distribution
    const int TRUST_BINS = 10;
    std::vector<int> bins(TRUST_BINS, 0);

    for (const auto& [agent_id, trust] : agent_trust_scores_) {
        int bin = static_cast<int>(trust * (TRUST_BINS - 1));
        bin = std::clamp(bin, 0, TRUST_BINS - 1);
        bins[bin]++;
    }

    double entropy = 0.0;
    for (int count : bins) {
        if (count > 0) {
            double p = static_cast<double>(count) / agent_trust_scores_.size();
            entropy -= p * std::log2(p);
        }
    }

    return entropy / std::log2(TRUST_BINS);  // Normalized entropy (0-1)
}

void ResilientEmergenceBenchmarker::start_sar_operation(const std::string& rover_agent, const std::string& target_agent) {
    // Record SAR operation initiation
    // In full implementation, would track active operations
}

void ResilientEmergenceBenchmarker::complete_sar_operation(const std::string& rover_agent, const std::string& target_agent, bool success) {
    // Update trust based on SAR operation outcome
    float trust_change = success ? 0.1f : -0.1f;
    update_agent_trust(rover_agent, agent_trust_scores_[rover_agent] + trust_change);

    if (success) {
        // Successful reconnection - boost target agent trust
        update_agent_trust(target_agent, agent_trust_scores_[target_agent] + 0.05f);
    }
}

void ResilientEmergenceBenchmarker::simulate_generation_with_faults(const FaultInjectionPattern& pattern) {
    // Apply pattern-specific fault simulation
    switch (pattern.type) {
        case FaultInjectionPattern::FaultType::NETWORK_PARTITION:
            inject_network_partition(pattern.duration_generations, pattern.intensity);
            break;

        case FaultInjectionPattern::FaultType::HEARTBEAT_FAILURE: {
            // Apply to either specified agents or random subset
            std::vector<std::string> targets = pattern.affected_agents.empty() ?
                select_random_agents(static_cast<int>(active_agents_.size() * pattern.intensity)) :
                pattern.affected_agents;
            inject_heartbeat_failure(targets, pattern.intensity);
            break;
        }

        case FaultInjectionPattern::FaultType::TRUST_CORRUPTION:
            inject_trust_corruption(pattern.intensity);
            break;

        case FaultInjectionPattern::FaultType::PULSE_INTERFERENCE: {
            // Apply to random or center location
            int center_x = pattern.affected_agents.empty() ?
                grid_->width() / 2 : grid_->get_agent_position(pattern.affected_agents[0]).first;
            int center_y = pattern.affected_agents.empty() ?
                grid_->height() / 2 : grid_->get_agent_position(pattern.affected_agents[0]).second;
            inject_pulse_interference(center_x, center_y, 3, pattern.intensity);
            break;
        }

        case FaultInjectionPattern::FaultType::AGENT_ISOLATION: {
            int isolation_count = static_cast<int>(active_agents_.size() * pattern.intensity);
            auto isolated_agents = select_random_agents(isolation_count);
            for (const auto& agent_id : isolated_agents) {
                update_agent_trust(agent_id, 0.1f);  // Mark as isolated
            }
            break;
        }
    }
}

void ResilientEmergenceBenchmarker::update_resilience_snapshot(long long generation) {
    ResilienceSnapshot snapshot;
    snapshot.generation = generation;
    snapshot.failed_agents = get_failed_agents();
    snapshot.recovered_agents = {};  // Would track recoveries over time
    snapshot.connectivity_index = calculate_current_connectivity();
    snapshot.trust_entropy = calculate_trust_entropy();
    snapshot.communication_success_rate = 1.0 - (snapshot.failed_agents.size() / static_cast<double>(active_agents_.size()));
    snapshot.active_sar_operations = 0;  // Would track active SAR operations

    resilience_history_.push_back(snapshot);
}

ResilientEmergenceBenchmarker::ResilienceMetrics ResilientEmergenceBenchmarker::calculate_final_metrics() const {
    ResilienceMetrics metrics = {};

    if (resilience_history_.empty()) return metrics;

    // MTTR: Mean Time To Reconnect (simplified)
    int total_failures = 0;
    int total_recovery_time = 0;

    for (const auto& snapshot : resilience_history_) {
        total_failures += snapshot.failed_agents.size();
        // Recovery time estimation would require tracking individual agent recovery times
    }

    metrics.mean_time_to_rejoin_ms = total_failures > 0 ?
        static_cast<double>(total_recovery_time) / total_failures : 0.0;

    // Agent survival ratio
    metrics.agent_survival_ratio = static_cast<double>(active_agents_.size()) / active_agents_.size();  // Simplified

    // Communication entropy
    double avg_communication_entropy = 0.0;
    for (const auto& snapshot : resilience_history_) {
        avg_communication_entropy += snapshot.communication_success_rate;
    }
    metrics.communication_entropy = 1.0 - (avg_communication_entropy / resilience_history_.size());

    // Trust stability
    double trust_variance_sum = 0.0;
    for (const auto& snapshot : resilience_history_) {
        trust_variance_sum += 1.0 - snapshot.trust_entropy;  // Higher entropy = less stable
    }
    metrics.trust_stability_score = trust_variance_sum / resilience_history_.size();

    // Heartbeat success rate
    double avg_heartbeat_success = 0.0;
    for (const auto& snapshot : resilience_history_) {
        avg_heartbeat_success += snapshot.communication_success_rate;
    }
    metrics.heartbeat_success_rate = avg_heartbeat_success / resilience_history_.size();

    // Connectivity recovery speed
    double avg_connectivity = 0.0;
    for (const auto& snapshot : resilience_history_) {
        avg_connectivity += snapshot.connectivity_index;
    }
    metrics.average_connectivity_recovery = avg_connectivity / resilience_history_.size();

    // Max simultaneous failures
    metrics.max_simultaneous_failures = 0;
    for (const auto& snapshot : resilience_history_) {
        metrics.max_simultaneous_failures = std::max(metrics.max_simultaneous_failures,
                                                      static_cast<int>(snapshot.failed_agents.size()));
    }

    // SAR operation efficiency
    metrics.sar_operation_efficiency = 0.8;  // Placeholder - would be calculated from actual operations

    return metrics;
}

std::vector<std::string> ResilientEmergenceBenchmarker::select_random_agents(int count) const {
    if (active_agents_.empty() || count <= 0) return {};

    count = std::min(count, static_cast<int>(active_agents_.size()));
    std::vector<std::string> selected = active_agents_;

    std::mt19937 rng(std::random_device{}());
    std::shuffle(selected.begin(), selected.end(), rng);

    selected.resize(count);
    return selected;
}

// Pulsing Cellular Automata Metrics Implementation (per DDLab pulsing-CA model)

double EmergenceAnalyzer::calculate_pulse_wavelength() const {
    // Calculate λ: average number of ticks between energy peaks in the grid
    // This requires tracking energy history over time, but since we don't have temporal
    // history in this implementation, we use spatial coherence as proxy for pulsing frequency

    std::vector<double> peak_intervals;

    // Find local energy maxima and their separations
    for (int y = 1; y < grid_.height() - 1; ++y) {
        for (int x = 1; x < grid_.width() - 1; ++x) {
            const auto& center = grid_.get_cell(x, y);
            bool is_local_max = true;

            // Check if center is local energy maximum
            for (int dy = -1; dy <= 1 && is_local_max; ++dy) {
                for (int dx = -1; dx <= 1 && is_local_max; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const auto& neighbor = grid_.get_cell(x + dx, y + dy);
                    if (neighbor.energy >= center.energy) {
                        is_local_max = false;
                    }
                }
            }

            if (is_local_max && center.energy > 0.7f) {  // High-energy peak
                // Calculate distance to nearest other high-energy peak
                double min_distance = std::numeric_limits<double>::max();
                for (int ny = 0; ny < grid_.height(); ++ny) {
                    for (int nx = 0; nx < grid_.width(); ++nx) {
                        if (nx == x && ny == y) continue;
                        const auto& neighbor = grid_.get_cell(nx, ny);
                        if (neighbor.energy > 0.7f) {
                            double dist = grid_.toroidal_distance(x, y, nx, ny);
                            min_distance = std::min(min_distance, dist);
                        }
                    }
                }

                if (min_distance < std::numeric_limits<double>::max()) {
                    peak_intervals.push_back(min_distance);
                }
            }
        }
    }

    // Average peak spacing as wavelength approximation
    if (peak_intervals.empty()) return 0.0;
    double avg_spacing = 0.0;
    for (double spacing : peak_intervals) {
        avg_spacing += spacing;
    }
    return avg_spacing / peak_intervals.size();
}

double EmergenceAnalyzer::calculate_pulse_amplitude() const {
    // Calculate A: energy difference (range) across the grid at current time
    // max_energy - min_energy = amplitude measure

    if (grid_.width() == 0 || grid_.height() == 0) return 0.0;

    double max_energy = 0.0;
    double min_energy = 1.0;  // Energy clamped to [0,1]

    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            double energy = grid_.get_cell(x, y).energy;
            max_energy = std::max(max_energy, energy);
            min_energy = std::min(min_energy, energy);
        }
    }

    return max_energy - min_energy;
}

std::vector<std::vector<double>> EmergenceAnalyzer::compute_entropy_density_map() const {
    // Spatial entropy-density map: local information content variation
    // Higher entropy-density indicates more chaotic/complex pulsing regions

    std::vector<std::vector<double>> entropy_map(grid_.height(),
                                                std::vector<double>(grid_.width(), 0.0));

    const int PATCH_SIZE = 3;  // 3x3 patches for local analysis

    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            // Calculate local energy entropy in PATCH_SIZE x PATCH_SIZE neighborhood
            std::vector<double> local_energies;
            int valid_cells = 0;

            // Collect energies in local patch (with toroidal wrapping)
            for (int dy = -PATCH_SIZE/2; dy <= PATCH_SIZE/2; ++dy) {
                for (int dx = -PATCH_SIZE/2; dx <= PATCH_SIZE/2; ++dx) {
                    int nx = (x + dx + grid_.width()) % grid_.width();
                    int ny = (y + dy + grid_.height()) % grid_.height();
                    local_energies.push_back(grid_.get_cell(nx, ny).energy);
                    valid_cells++;
                }
            }

            // Calculate entropy of binned energy distribution
            const int ENERGY_BINS = 8;
            std::vector<int> bins(ENERGY_BINS, 0);

            for (double energy : local_energies) {
                int bin = static_cast<int>(energy * (ENERGY_BINS - 1));
                bin = std::clamp(bin, 0, ENERGY_BINS - 1);
                bins[bin]++;
            }

            double entropy = 0.0;
            for (int count : bins) {
                if (count > 0) {
                    double p = static_cast<double>(count) / valid_cells;
                    entropy -= p * std::log2(p);
                }
            }

            // Normalize entropy to [0,1] range
            entropy_map[y][x] = entropy / std::log2(ENERGY_BINS);
        }
    }

    return entropy_map;
}

bool EmergenceAnalyzer::detect_glider_coherence_chains() const {
    // Detect stable pulse propagation chains (glider-like coherence)
    // Look for connected regions of high energy with consistent directional flow

    // Create connectivity graph of high-energy cells
    std::vector<std::pair<int, int>> high_energy_cells;

    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            if (grid_.get_cell(x, y).energy > 0.8f) {  // Very high energy threshold
                high_energy_cells.emplace_back(x, y);
            }
        }
    }

    // Check for coherent chains: connected components with directional consistency
    // Simplified: look for linear chains of high-energy cells
    std::unordered_set<std::string> visited;

    for (const auto& [x, y] : high_energy_cells) {
        std::string key = std::to_string(x) + "," + std::to_string(y);
        if (visited.count(key)) continue;

        // BFS to find connected component
        std::vector<std::pair<int, int>> component;
        std::queue<std::pair<int, int>> queue;
        queue.emplace(x, y);
        visited.insert(key);

        while (!queue.empty()) {
            auto [cx, cy] = queue.front();
            queue.pop();
            component.emplace_back(cx, cy);

            // Check Moore neighbors for connectivity
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;

                    int nx = (cx + dx + grid_.width()) % grid_.width();
                    int ny = (cy + dy + grid_.height()) % grid_.height();

                    if (grid_.get_cell(nx, ny).energy > 0.8f) {
                        std::string nkey = std::to_string(nx) + "," + std::to_string(ny);
                        if (!visited.count(nkey)) {
                            visited.insert(nkey);
                            queue.emplace(nx, ny);
                        }
                    }
                }
            }
        }

        // Check if component forms a coherent chain (aspect ratio > 3:1)
        if (component.size() >= 5) {  // Minimum chain length
            double min_x = component[0].first, max_x = min_x;
            double min_y = component[0].second, max_y = min_y;

            for (const auto& [px, py] : component) {
                min_x = std::min(min_x, (double)px);
                max_x = std::max(max_x, (double)px);
                min_y = std::min(min_y, (double)py);
                max_y = std::max(max_y, (double)py);
            }

            double width = max_x - min_x + 1;
            double height = max_y - min_y + 1;
            double aspect_ratio = std::max(width / height, height / width);

            // Aspect ratio > 3 indicates a coherent directional chain
            if (aspect_ratio > 3.0) {
                return true;  // Found coherent glider-like chain
            }
        }
    }

    return false;
}
