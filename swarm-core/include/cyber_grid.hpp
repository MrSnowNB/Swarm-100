#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>       // for std::pair
#include <algorithm>     // for std::find
#include <cmath>
#include <random>
#include <iostream>

// Forward declarations
class CyberGrid;
struct AgentID;

// Core Cell Structure for CyberGrid
struct Cell {
    // State variables
    bool alive;                    // Conway's Game of Life state
    float energy;                  // LoRA pulse intensity (0.0 to 1.0)

    // Multi-agent occupancy
    std::vector<std::string> occupants;  // Agent IDs in this cell (up to 4)

    // Spatial metadata
    int x, y;                      // Grid coordinates (for debugging)

    // Initialization
    Cell() : alive(false), energy(0.0f), occupants(), x(0), y(0) {}
    Cell(int px, int py) : alive(false), energy(0.0f), occupants(), x(px), y(py) {}

    // Utility methods
    size_t agent_count() const { return occupants.size(); }
    bool is_empty() const { return occupants.empty(); }
    bool can_occupy() const { return occupants.size() < 4; }  // Max 4 agents per cell

    // Add/remove agents
    bool add_agent(const std::string& agent_id) {
        if (!can_occupy()) return false;
        occupants.push_back(agent_id);
        return true;
    }

    bool remove_agent(const std::string& agent_id) {
        auto it = std::find(occupants.begin(), occupants.end(), agent_id);
        if (it != occupants.end()) {
            occupants.erase(it);
            return true;
        }
        return false;
    }

    // Energy manipulation
    void add_energy(float amount) {
        energy = std::min(1.0f, energy + amount);
    }

    void reduce_energy(float amount) {
        energy = std::max(0.0f, energy - amount);
    }

    // Check activation threshold
    bool is_pulsing() const { return energy > 0.65f; }
};

/**
 * CyberGrid - 2D Toroidal Swarm Substrate
 *
 * Implements Conway's Game of Life cellular automata combined with LoRA pulse
 * propagation in a toroidal (wraparound) 2D grid. Serves as the spatial foundation
 * for "Alice in CyberLand" emergent swarm intelligence.
 */
class CyberGrid {
private:
    static constexpr int DEFAULT_WIDTH = 100;
    static constexpr int DEFAULT_HEIGHT = 100;

    // Grid dimensions
    int width_;
    int height_;
    std::vector<Cell> grid_;        // 1D vector for contiguous memory

    // LoRA pulse mechanics constants
    static constexpr float ATTENUATION_K = 0.15f;  // Energy halves every ~4 cells
    static constexpr float GLOBAL_DAMPING = 0.95f;  // Temporal decay per step
    static constexpr float ACTIVATION_THRESHOLD = 0.65f;
    static constexpr int MAX_PULSE_RANGE = 4;        // Effective LoRA range

    // Random number generator for noise/probabilistic behavior
    std::mt19937 rng_;
    std::uniform_real_distribution<float> noise_dist_;

public:
    // Constructors
    explicit CyberGrid(int width = DEFAULT_WIDTH, int height = DEFAULT_HEIGHT);
    CyberGrid(const CyberGrid&) = delete;  // Prevent copying large grids
    CyberGrid& operator=(const CyberGrid&) = delete;
    CyberGrid(CyberGrid&&) = default;
    CyberGrid& operator=(CyberGrid&&) = default;

    // Grid properties
    int width() const { return width_; }
    int height() const { return height_; }
    size_t cell_count() const { return grid_.size(); }

    // Toroidal indexing (wraparound coordinates)
    int toroidal_x(int x) const { return (x + width_) % width_; }
    int toroidal_y(int y) const { return (y + height_) % height_; }
    size_t linear_index(int x, int y) const;

    // Cell access with toroidal wraparound
    Cell& get_cell(int x, int y);
    const Cell& get_cell(int x, int y) const;

    // Agent management
    bool place_agent(int x, int y, const std::string& agent_id);
    bool remove_agent(int x, int y, const std::string& agent_id);
    Cell& find_agent(const std::string& agent_id);
    std::pair<int, int> get_agent_position(const std::string& agent_id);

    // Simulation engines

    /**
     * Execute one full simulation step combining:
     * 1. Conway's Game of Life rules (modified by energy)
     * 2. LoRA pulse propagation and diffusion
     * 3. Agent movement (if enabled)
     */
    void step();

    /**
     * Apply Conway's Game of Life rules with LoRA energy coupling
     * Returns count of cells that changed state
     */
    int apply_conway_rules();

    /**
     * Propagate LoRA-style energy pulses through the grid
     * Uses log-distance path loss model: E = αΣE⋅e^(-k⋅d)
     */
    void apply_lora_pulses();

    /**
     * Count living neighbors using Moore neighborhood (8 directions)
     */
    int count_living_neighbors(int x, int y) const;

    /**
     * Calculate energy diffusion from neighboring cells
     * Excludes center cell, uses toroidal distance
     */
    float calculate_energy_diffusion(int x, int y) const;

    // Utilities
    void reset();                           // Clear all cells
    void randomize(float alive_probability, float initial_energy);
    size_t alive_cell_count() const;        // Conway's "alive" count
    size_t occupied_cell_count() const;     // Cells with agents
    float average_energy() const;           // Mean energy across grid

    // Spatial analysis helpers
    std::vector<std::pair<int, int>> get_moore_neighbors(int x, int y) const;

    // Distance calculations (toroidal)
    float toroidal_distance(int x1, int y1, int x2, int y2) const;

    // Debugging and serialization
    void print_grid(std::ostream& os, int start_x = 0, int start_y = 0, int w = 20, int h = 20) const;
    std::string grid_to_string(int zoom_factor = 1) const;

    // Configuration
    void set_lora_parameters(float attenuation_k = ATTENUATION_K,
                           float global_damping = GLOBAL_DAMPING,
                           float activation_threshold = ACTIVATION_THRESHOLD);
};

// Spatial extension for RootCauseAnalyzer
struct SpatialDependencyLink {
    std::string agent_id;
    float strength;              // 0.0 to 1.0 - spatial closeness/proximity
    float energy_correlation;    // How correlated their energy fields are
    int distance;               // Manhattan distance in grid coordinates
    std::string relationship_type;  // "adjacent", "close_energy", "pulse_chain"
};

/**
 * Grid-based spatial analysis extending RootCauseAnalyzer
 */
class SpatialRootCauseAnalyzer {
private:
    const CyberGrid& grid_;

public:
    explicit SpatialRootCauseAnalyzer(const CyberGrid& grid) : grid_(grid) {}

    /**
     * Generate spatial dependency graph from current grid state
     * Agents closer in space or energy fields have stronger dependencies
     */
    std::vector<SpatialDependencyLink> build_spatial_dependencies() const;

    /**
     * Analyze cascade patterns in 2D space (glider-like failure propagation)
     */
    std::vector<SpatialDependencyLink> find_spatial_cascades() const;

    /**
     * Calculate spatial clustering metrics for agents in failure chains
     */
    float calculate_spatial_clustering(const std::vector<SpatialDependencyLink>& chain) const;
};

// CyberGrid Configuration and Factory
struct CyberGridConfig {
    int width = 100;
    int height = 100;
    float initial_energy = 0.0f;
    float alive_probability = 0.0f;  // For random initialization
    bool toroidal = true;           // Always toroidal for Swarm-100
};

class CyberGridFactory {
public:
    static CyberGrid create_standard_grid();
    static CyberGrid create_random_grid(int width, int height, float alive_prob, float energy_prob);
    static CyberGrid create_from_config(const CyberGridConfig& config);
};

// Exception classes
class CyberGridException : public std::runtime_error {
public:
    explicit CyberGridException(const std::string& message)
        : std::runtime_error(message) {}
};

class InvalidPositionException : public CyberGridException {
public:
    InvalidPositionException(int x, int y, int max_x, int max_y)
        : CyberGridException("Invalid position (" + std::to_string(x) + "," + std::to_string(y) +
                           ") for grid size " + std::to_string(max_x) + "x" + std::to_string(max_y)) {}
};

class CellOccupancyException : public CyberGridException {
public:
    explicit CellOccupancyException(int x, int y)
        : CyberGridException("Cell (" + std::to_string(x) + "," + std::to_string(y) + ") at capacity") {}
};
