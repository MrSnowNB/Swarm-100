#include "cyber_grid.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <chrono>
#include <thread>

// CyberGrid Implementation

CyberGrid::CyberGrid(int width, int height)
    : width_(width),
      height_(height),
      grid_(static_cast<size_t>(width) * height),
      pulse_phase_ratio_(4),  // Initial pulse phase ratio
      adaptive_attenuation_k_(ATTENUATION_K),
      adaptive_global_damping_(GLOBAL_DAMPING),
      adaptive_activation_threshold_(ACTIVATION_THRESHOLD),
      rng_(std::random_device{}()),
      noise_dist_(0.0f, 1.0f),
      jitter_dist_(0.8f, 1.2f),  // Heartbeat jitter: ±20%
      tick_count_(0),
      last_tick_time_(std::chrono::steady_clock::now()),
      timing_error_accumulator_(0.0) {

    // Initialize cells with coordinates
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            size_t index = linear_index(x, y);
            grid_[index] = Cell(x, y);
        }
    }
}

size_t CyberGrid::linear_index(int x, int y) const {
    int tx = toroidal_x(x);
    int ty = toroidal_y(y);
    return static_cast<size_t>(ty) * width_ + tx;
}

Cell& CyberGrid::get_cell(int x, int y) {
    return grid_[linear_index(x, y)];
}

const Cell& CyberGrid::get_cell(int x, int y) const {
    return grid_[linear_index(x, y)];
}

bool CyberGrid::place_agent(int x, int y, const std::string& agent_id) {
    int tx = toroidal_x(x);
    int ty = toroidal_y(y);

    if (!grid_[linear_index(tx, ty)].add_agent(agent_id)) {
        throw CellOccupancyException(x, y);
    }

    // Add initial energy pulse to signify agent arrival
    grid_[linear_index(tx, ty)].add_energy(0.2f);

    return true;
}

bool CyberGrid::remove_agent(int x, int y, const std::string& agent_id) {
    int tx = toroidal_x(x);
    int ty = toroidal_y(y);

    return grid_[linear_index(tx, ty)].remove_agent(agent_id);
}

Cell& CyberGrid::find_agent(const std::string& agent_id) {
    for (auto& cell : grid_) {
        for (const auto& occupant : cell.occupants) {
            if (occupant == agent_id) {
                return cell;
            }
        }
    }
    throw std::runtime_error("Agent " + agent_id + " not found in grid");
}

std::pair<int, int> CyberGrid::get_agent_position(const std::string& agent_id) const {
    for (const auto& cell : grid_) {
        for (const auto& occupant : cell.occupants) {
            if (occupant == agent_id) {
                return {cell.x, cell.y};
            }
        }
    }
    throw std::runtime_error("Agent " + agent_id + " not found in grid");
}

void CyberGrid::step() {
    // Hardware-locked fixed timestep execution
    static constexpr std::chrono::duration<double> TICK_PERIOD(1.0 / TICK_FREQUENCY_HZ);

    auto t_start = std::chrono::steady_clock::now();

    // Execute Conway rules every tick
    apply_conway_rules();

    // Apply LoRA pulses as phase-locked subharmonic (every pulse_phase_ratio_ ticks)
    if (tick_count_ % pulse_phase_ratio_ == 0) {
        apply_lora_pulses();
    }

    tick_count_++;

    // Calculate timing error and accumulate for drift correction
    auto t_elapsed = std::chrono::steady_clock::now() - t_start;
    double actual_duration = t_elapsed.count();
    double expected_duration = TICK_PERIOD.count();
    double error = actual_duration - expected_duration;

    timing_error_accumulator_ += error;

    // Sleep until next tick, accounting for accumulated error
    auto t_next = last_tick_time_ + (tick_count_ * TICK_PERIOD);

    // Apply correction if error exceeds threshold
    if (std::abs(timing_error_accumulator_) > CORRECTION_THRESHOLD) {
        t_next -= std::chrono::duration<double>(timing_error_accumulator_ * 0.1);  // Gentle correction
        timing_error_accumulator_ *= 0.9;  // Decay accumulator
    }

    std::this_thread::sleep_until(t_next);
    last_tick_time_ = std::chrono::steady_clock::now();

    // TODO: Add agent movement logic when swarm behavior is integrated
}

int CyberGrid::apply_conway_rules() {
    int changes = 0;

    // Create next state grid to avoid modifying while reading
    std::vector<Cell> next_grid = grid_;

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Cell& current = get_cell(x, y);
            Cell& next = next_grid[linear_index(x, y)];

            int live_neighbors = count_living_neighbors(x, y);
            float current_energy = current.energy;

            bool was_alive = current.alive;

            // Modified Conway's Game of Life with enhanced LoRA energy coupling
            if (current.alive) {
                // Alive cell survival rules
                if (live_neighbors < 2) {
                    // Underpopulation - dies (standard Conway)
                    next.alive = false;
                    changes++;
                } else if (live_neighbors <= 3) {
                    // Survival zone (modified by energy)
                    if (current_energy > 0.8f) {
                        // High energy keeps it alive even outside normal range
                        next.alive = true;
                    } else {
                        next.alive = true;  // Standard survival
                    }
                } else {
                    // Overpopulation
                    if (current_energy > 0.3f) {
                        // High energy allows survival with reduced energy
                        next.alive = true;
                        next.energy = current_energy * 0.7f;  // Energy reduction
                    } else {
                        // Standard overpopulation death
                        next.alive = false;
                        changes++;
                    }
                }
            } else {
                // Dead cell birth rules
                if (live_neighbors == 3) {
                    // Standard birth
                    next.alive = true;
                    changes++;
                } else if (calculate_energy_diffusion(x, y) > 2.0f) {
                    // Energy coupling allows birth in energized zones
                    next.alive = true;
                    changes++;
                }
                // Dead cells maintain their death state
            }

            // Cross-diffusion coupling: life changes create energy pulses
            if (was_alive != next.alive) {
                // Death → burst of energy release, Birth → energy consumption
                float life_change_pulse = next.alive ? -0.1f : 0.3f;  // Birth consumes, death releases
                next.energy += life_change_pulse;
            }

            // Constructive interference: life gradient affects energy
            float life_gradient_energy = 0.0f;
            auto neighbors = get_moore_neighbors(x, y);
            for (const auto& [nx, ny] : neighbors) {
                const Cell& neighbor = get_cell(nx, ny);
                if (neighbor.alive && !current.alive) {
                    // Alive neighbor enhances birth potential in dead cells
                    life_gradient_energy += 0.05f;
                }
            }
            next.energy += life_gradient_energy;

            // Clamp energy and natural decay
            next.energy = std::max(0.0f, std::min(1.0f, next.energy * 0.98f));
        }
    }

    // Apply the next state
    grid_ = std::move(next_grid);
    return changes;
}

void CyberGrid::apply_lora_pulses() {
    // Create next energy state grid
    std::vector<float> next_energy(grid_.size(), 0.0f);

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            size_t index = linear_index(x, y);
            float diffused_energy = calculate_energy_diffusion(x, y);

            // Apply global damping to natural energy decay
            float current_energy = get_cell(x, y).energy * GLOBAL_DAMPING;

            // Add diffused energy from neighbors (with log-distance attenuation)
            next_energy[index] = std::min(1.0f, current_energy + diffused_energy);

            // Clamp to valid range
            next_energy[index] = std::max(0.0f, std::min(1.0f, next_energy[index]));
        }
    }

    // Apply computed energies back to cells
    for (size_t i = 0; i < grid_.size(); ++i) {
        grid_[i].energy = next_energy[i];
    }
}

int CyberGrid::count_living_neighbors(int x, int y) const {
    int count = 0;

    // Moore neighborhood: 8 surrounding cells
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;  // Skip self

            const Cell& neighbor = get_cell(x + dx, y + dy);
            if (neighbor.alive) {
                count++;
            }
        }
    }

    return count;
}

float CyberGrid::calculate_energy_diffusion(int x, int y) const {
    float total_diffusion = 0.0f;

    // Sample Moore neighborhood
    std::vector<std::pair<int, int>> neighbors = get_moore_neighbors(x, y);

    for (const auto& [nx, ny] : neighbors) {
        const Cell& neighbor = get_cell(nx, ny);

        if (neighbor.energy > 0.0f) {
            // Calculate toroidal distance for attenuation
            float distance = toroidal_distance(x, y, nx, ny);

            // Apply log-distance path loss: E ∝ e^(-k⋅d)
            // Clamp distance to avoid extreme attenuation
            distance = std::max(1.0f, std::min(static_cast<float>(MAX_PULSE_RANGE * 2), distance));
            float attenuation = std::exp(-ATTENUATION_K * distance);

            total_diffusion += neighbor.energy * attenuation;
        }
    }

    // Scale down diffusion to prevent runaway energy growth
    return total_diffusion * 0.1f;  // Conservative diffusion factor
}

std::vector<std::pair<int, int>> CyberGrid::get_moore_neighbors(int x, int y) const {
    std::vector<std::pair<int, int>> neighbors;

    // All 8 Moore neighborhood positions
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;  // Skip center

            neighbors.emplace_back(x + dx, y + dy);
        }
    }

    return neighbors;
}

float CyberGrid::toroidal_distance(int x1, int y1, int x2, int y2) const {
    // Calculate minimum distance in toroidal topology
    int dx = std::abs(x2 - x1);
    int dy = std::abs(y2 - y1);

    // Use shorter toroidal distance
    dx = std::min(dx, width_ - dx);
    dy = std::min(dy, height_ - dy);

    return std::sqrt(static_cast<float>(dx * dx + dy * dy));
}

void CyberGrid::reset() {
    for (auto& cell : grid_) {
        cell.alive = false;
        cell.energy = 0.0f;
        cell.occupants.clear();
    }
}

void CyberGrid::randomize(float alive_probability, float initial_energy) {
    reset();

    // Random initialization following probabilities
    for (auto& cell : grid_) {
        cell.alive = noise_dist_(rng_) < alive_probability;
        cell.energy = noise_dist_(rng_) * initial_energy;
    }
}

size_t CyberGrid::alive_cell_count() const {
    return std::count_if(grid_.begin(), grid_.end(),
                        [](const Cell& c) { return c.alive; });
}

size_t CyberGrid::occupied_cell_count() const {
    return std::count_if(grid_.begin(), grid_.end(),
                        [](const Cell& c) { return !c.is_empty(); });
}

float CyberGrid::average_energy() const {
    if (grid_.empty()) return 0.0f;

    float total = 0.0f;
    for (const auto& cell : grid_) {
        total += cell.energy;
    }

    return total / grid_.size();
}

void CyberGrid::print_grid(std::ostream& os, int start_x, int start_y, int w, int h) const {
    // Print a section of the grid for debugging
    for (int y = start_y; y < std::min(start_y + h, height_); ++y) {
        for (int x = start_x; x < std::min(start_x + w, width_); ++x) {
            const Cell& cell = get_cell(x, y);

            if (cell.alive) {
                // Show energy level for alive cells
                int energy_char = '0' + static_cast<int>(cell.energy * 9);
                os << static_cast<char>(energy_char);
            } else if (cell.energy > 0.5f) {
                os << '*';  // Pulsing dead cell
            } else if (cell.energy > 0.0f) {
                os << '.';  // Low energy dead cell
            } else if (!cell.is_empty()) {
                os << static_cast<char>('a' + cell.agent_count() - 1);  // Agents but no life/energy
            } else {
                os << ' ';  // Empty dead cell
            }
        }
        os << '\n';
    }
}

std::string CyberGrid::grid_to_string(int zoom_factor) const {
    std::stringstream ss;

    for (int y = 0; y < height_; y += zoom_factor) {
        for (int x = 0; x < width_; x += zoom_factor) {
            const Cell& cell = get_cell(x, y);

            if (cell.alive) {
                int level = static_cast<int>(cell.energy * 9);
                ss << level;
            } else if (cell.energy > 0.0f) {
                ss << '.';
            } else {
                ss << ' ';
            }
        }
        ss << '\n';
    }

    return ss.str();
}

void CyberGrid::set_lora_parameters(float attenuation_k, float global_damping, float activation_threshold) {
    // Note: In this implementation, constants are compile-time.
    // Runtime configuration would require refactoring to instance variables.
    (void)attenuation_k;      // Suppress unused parameter warning
    (void)global_damping;     // Could be added as instance variables
    (void)activation_threshold;
}

// CyberGridFactory Implementation

CyberGrid CyberGridFactory::create_standard_grid() {
    return CyberGrid(100, 100);  // Default Swarm-100 size
}

CyberGrid CyberGridFactory::create_random_grid(int width, int height, float alive_prob, float energy_prob) {
    CyberGrid grid(width, height);
    grid.randomize(alive_prob, energy_prob);
    return grid;
}

CyberGrid CyberGridFactory::create_from_config(const CyberGridConfig& config) {
    CyberGrid grid(config.width, config.height);

    if (config.alive_probability > 0.0f || config.initial_energy > 0.0f) {
        grid.randomize(config.alive_probability, config.initial_energy);
    }

    return grid;
}

// SpatialRootCauseAnalyzer Implementation

std::vector<SpatialDependencyLink> SpatialRootCauseAnalyzer::build_spatial_dependencies() const {
    std::vector<SpatialDependencyLink> dependencies;

    // Create agent position lookup
    std::unordered_map<std::string, std::pair<int, int>> agent_positions;

    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            const Cell& cell = grid_.get_cell(x, y);
            for (const auto& agent_id : cell.occupants) {
                agent_positions[agent_id] = {x, y};
            }
        }
    }

    // Calculate dependencies between agents
    for (const auto& [agent_id, pos1] : agent_positions) {
        const auto& [x1, y1] = pos1;

        for (const auto& [other_id, pos2] : agent_positions) {
            if (agent_id == other_id) continue;

            const auto& [x2, y2] = pos2;

            float distance = grid_.toroidal_distance(x1, y1, x2, y2);
            float strength = std::max(0.0f, 1.0f - distance / 10.0f);  // Proximity strength

            // Energy correlation
            const Cell& cell1 = grid_.get_cell(x1, y1);
            const Cell& cell2 = grid_.get_cell(x2, y2);
            float energy_corr = 1.0f - std::abs(cell1.energy - cell2.energy);

            std::string relationship;
            if (distance <= 1.5f) {
                relationship = "adjacent";
            } else if (energy_corr > 0.7f) {
                relationship = "close_energy";
            } else {
                relationship = "distant";
            }

            dependencies.push_back({
                other_id,
                strength,
                energy_corr,
                static_cast<int>(distance),
                relationship
            });
        }
    }

    return dependencies;
}

std::vector<SpatialDependencyLink> SpatialRootCauseAnalyzer::find_spatial_cascades() const {
    std::vector<SpatialDependencyLink> cascades;

    // Look for patterns where energy or life propagates in chains
    for (int y = 0; y < grid_.height(); ++y) {
        for (int x = 0; x < grid_.width(); ++x) {
            const Cell& cell = grid_.get_cell(x, y);

            if (cell.energy > 0.8f && cell.alive) {
                // Check if this creates a chain of high-energy alive cells
                auto neighbors = grid_.get_moore_neighbors(x, y);

                for (const auto& [nx, ny] : neighbors) {
                    const Cell& neighbor = grid_.get_cell(nx, ny);

                    if (neighbor.energy > 0.6f && neighbor.alive) {
                        // Potential cascade link
                        cascades.push_back({
                            "cascade_cell",
                            0.8f,
                            0.9f,  // High energy correlation
                            1,
                            "pulse_chain"
                        });
                    }
                }
            }
        }
    }

    return cascades;
}

float SpatialRootCauseAnalyzer::calculate_spatial_clustering(const std::vector<SpatialDependencyLink>& chain) const {
    if (chain.empty()) return 0.0f;

    // Calculate average distance - lower means more clustered
    float total_distance = 0.0f;
    for (const auto& link : chain) {
        total_distance += link.distance;
    }

    float avg_distance = total_distance / chain.size();

    // Convert to clustering score (0-1, higher = more clustered)
    return std::max(0.0f, 1.0f - avg_distance / 10.0f);
}

// Adaptive heartbeat and resilience methods implementation

float CyberGrid::calculate_local_density(int x, int y, int radius) const {
    int agent_count = 0;
    int cell_count = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const Cell& cell = get_cell(x + dx, y + dy);
            agent_count += cell.agent_count();
            cell_count++;
        }
    }

    return static_cast<float>(agent_count) / cell_count;
}

void CyberGrid::update_adaptive_parameters() {
    // Calculate average density across grid
    float total_density = 0.0f;
    int sample_count = 0;

    for (int y = 0; y < height_; y += 5) {  // Sample every 5th cell for performance
        for (int x = 0; x < width_; x += 5) {
            total_density += calculate_local_density(x, y, 3);
            sample_count++;
        }
    }

    float avg_density = total_density / sample_count;

    // Adaptive pulse phase ratio: faster pulses in high density areas
    if (avg_density > 0.7f) {
        pulse_phase_ratio_ = 2;  // Faster pulses when crowded
    } else if (avg_density < 0.3f) {
        pulse_phase_ratio_ = 6;  // Slower pulses when sparse
    } else {
        pulse_phase_ratio_ = 4;  // Default pulse rate
    }

    // Adaptive attenuation: stronger signals in sparse areas
    if (avg_density < 0.3f) {
        adaptive_attenuation_k_ = ATTENUATION_K * 0.7f;  // Longer range when sparse
    } else if (avg_density > 0.7f) {
        adaptive_attenuation_k_ = ATTENUATION_K * 1.3f;  // Shorter range when crowded
    } else {
        adaptive_attenuation_k_ = ATTENUATION_K;
    }

    // Trust decay for agents with low activity
    auto now = std::chrono::steady_clock::now();
    for (auto& [agent_id, trust] : agent_trust_scores_) {
        if (last_heartbeat_.count(agent_id)) {
            auto time_since_heartbeat = now - last_heartbeat_[agent_id];
            if (time_since_heartbeat > std::chrono::milliseconds(5000)) {  // 5 seconds
                trust = std::max(0.0f, trust - 0.01f);  // Slow decay
            }
        }
    }
}

int CyberGrid::get_adaptive_heartbeat_interval(const std::string& agent_id) const {
    // Base interval modulated by local density and trust
    int base_interval = 1000;  // 1 second base

    if (last_heartbeat_.count(agent_id)) {
        auto [x, y] = get_agent_position(agent_id);
        float local_density = calculate_local_density(x, y, 3);

        // Denser areas get faster heartbeats (shorter intervals)
        float density_factor = 1.0f - (local_density * 0.5f);  // 0.5 to 1.0
        base_interval = static_cast<int>(base_interval * density_factor);

        // Trust factor: less trusted agents heartbeat more frequently
        float trust_factor = 1.0f;
        if (agent_trust_scores_.count(agent_id)) {
            trust_factor = 2.0f - agent_trust_scores_.at(agent_id);  // 1.0 to 2.0
        }
        base_interval = static_cast<int>(base_interval * trust_factor);

        // Apply jitter: ±20%
        float jitter = jitter_dist_(rng_);
        base_interval = static_cast<int>(base_interval * jitter);
    }

    return std::max(200, std::min(5000, base_interval));  // Clamp to reasonable range
}

std::vector<std::string> CyberGrid::identify_failed_agents(int timeout_ms) const {
    std::vector<std::string> failed_agents;
    auto now = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(timeout_ms);

    for (const auto& [agent_id, last_time] : last_heartbeat_) {
        auto time_since_heartbeat = now - last_time;
        if (time_since_heartbeat > timeout) {
            failed_agents.push_back(agent_id);
        }
    }

    return failed_agents;
}

void CyberGrid::initiate_sar_search(const std::string& failed_agent, const std::string& rover_agent) {
    // SAR mechanism: designated rover agent searches for failed agent
    if (std::find(rover_agents_.begin(), rover_agents_.end(), rover_agent) != rover_agents_.end()) {
        // Record SAR initiation (logging/monitoring)
        record_heartbeat(failed_agent);  // Reset failed agent's heartbeat?
        update_agent_trust(failed_agent, false);  // Mark as temporarily failed
    }
}

void CyberGrid::update_agent_trust(const std::string& agent_id, bool successful_communication) {
    if (!agent_trust_scores_.count(agent_id)) {
        agent_trust_scores_[agent_id] = 0.5f;  // Default trust
    }

    float& trust = agent_trust_scores_[agent_id];

    if (successful_communication) {
        trust = std::min(1.0f, trust + 0.1f);  // Trust recovery
    } else {
        trust = std::max(0.0f, trust - 0.05f);  // Trust decay on failure
    }
}

float CyberGrid::get_agent_trust(const std::string& agent_id) const {
    if (agent_trust_scores_.count(agent_id)) {
        return agent_trust_scores_.at(agent_id);
    }
    return 0.5f;  // Default trust score
}

void CyberGrid::register_rover_agent(const std::string& agent_id) {
    if (std::find(rover_agents_.begin(), rover_agents_.end(), agent_id) == rover_agents_.end()) {
        rover_agents_.push_back(agent_id);
    }
}

std::vector<std::string> CyberGrid::get_rover_agents() const {
    return rover_agents_;
}

float CyberGrid::get_adaptive_pulse_range(int x, int y) const {
    float local_density = calculate_local_density(x, y, 2);

    // Adaptive range: longer in sparse areas, shorter in dense areas
    if (local_density < 0.3f) {
        return MAX_PULSE_RANGE * 1.5f;  // Extended range in sparse areas
    } else if (local_density > 0.7f) {
        return MAX_PULSE_RANGE * 0.7f;  // Reduced range in dense areas
    }

    return MAX_PULSE_RANGE;
}

void CyberGrid::record_heartbeat(const std::string& agent_id) {
    last_heartbeat_[agent_id] = std::chrono::steady_clock::now();
}

std::chrono::steady_clock::time_point CyberGrid::get_last_heartbeat(const std::string& agent_id) const {
    if (last_heartbeat_.count(agent_id)) {
        return last_heartbeat_.at(agent_id);
    }
    return std::chrono::steady_clock::time_point{};  // Default/empty time point
}
