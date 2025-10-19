#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "root_cause_analyzer.hpp"
#include "cyber_grid.hpp"

namespace py = pybind11;

// Python bindings for the Swarm-100 C++ core modules
PYBIND11_MODULE(swarm_core, m) {
    m.doc() = "Swarm-100 high-performance C++ core modules";

    // Bind AnalysisResult enum
    py::enum_<AnalysisResult>(m, "AnalysisResult")
        .value("SUCCESS", AnalysisResult::SUCCESS)
        .value("CYCLE_DETECTED", AnalysisResult::CYCLE_DETECTED)
        .value("DEPTH_LIMIT_EXCEEDED", AnalysisResult::DEPTH_LIMIT_EXCEEDED)
        .value("TIMEOUT_EXCEEDED", AnalysisResult::TIMEOUT_EXCEEDED)
        .value("MEMORY_LIMIT_EXCEEDED", AnalysisResult::MEMORY_LIMIT_EXCEEDED)
        .value("INVALID_INPUT", AnalysisResult::INVALID_INPUT);

    // Bind AnalysisConfig struct
    py::class_<AnalysisConfig>(m, "AnalysisConfig")
        .def(py::init<>())
        .def_readwrite("max_recursion_depth", &AnalysisConfig::max_recursion_depth)
        .def_readwrite("timeout", &AnalysisConfig::timeout)
        .def_readwrite("max_memory_mb", &AnalysisConfig::max_memory_mb)
        .def_readwrite("max_dependency_chain", &AnalysisConfig::max_dependency_chain)
        .def_readwrite("min_confidence_threshold", &AnalysisConfig::min_confidence_threshold);

    // Bind data structures
    py::class_<AgentDependency>(m, "AgentDependency")
        .def_readonly("agent_id", &AgentDependency::agent_id)
        .def_readonly("confidence_score", &AgentDependency::confidence_score)
        .def_readonly("failure_mode", &AgentDependency::failure_mode)
        .def_readonly("symptoms", &AgentDependency::symptoms);

    py::class_<MitigationRecommendation>(m, "MitigationRecommendation")
        .def_readonly("action", &MitigationRecommendation::action)
        .def_readonly("priority", &MitigationRecommendation::priority)
        .def_readonly("rationale", &MitigationRecommendation::rationale)
        .def_readonly("expected_impact", &MitigationRecommendation::expected_impact);

    // Bind RootCauseResult struct with proper chrono handling
    py::class_<RootCauseResult>(m, "RootCauseResult")
        .def_readonly("dependency_chain", &RootCauseResult::dependency_chain)
        .def_readonly("primary_root_cause", &RootCauseResult::primary_root_cause)
        .def_readonly("recommendations", &RootCauseResult::recommendations)
        .def_readonly("analysis_confidence", &RootCauseResult::analysis_confidence)
        .def_property_readonly("analysis_duration_ms",
            [](const RootCauseResult& r) { return r.analysis_duration.count(); })
        .def_readonly("analysis_complete", &RootCauseResult::analysis_complete);

    // Bind DependencyLink struct
    py::class_<DependencyLink>(m, "DependencyLink")
        .def(py::init<std::string, float, std::string>())
        .def_readwrite("target_agent", &DependencyLink::target_agent)
        .def_readwrite("strength", &DependencyLink::strength)
        .def_readwrite("relationship_type", &DependencyLink::relationship_type);

    // Bind the main RootCauseAnalyzer class
    py::class_<RootCauseAnalyzer>(m, "RootCauseAnalyzer")
        .def(py::init<const AnalysisConfig&>(), py::arg("config") = AnalysisConfig())
        .def("analyze_dependency_chain", &RootCauseAnalyzer::analyze_dependency_chain,
             py::arg("target_agent"), py::arg("symptoms"), py::arg("graph"),
             "Perform recursive root cause analysis with loop protection")
        .def("detect_cycles", &RootCauseAnalyzer::detect_cycles,
             py::arg("graph"),
             "Check if dependency graph contains cycles")
        .def("reset_tracking", &RootCauseAnalyzer::reset_tracking,
             "Reset internal state for new analysis")
        .def("get_current_memory_usage", &RootCauseAnalyzer::get_current_memory_usage,
             "Get current memory usage estimate");

    // Bind exception classes
    py::register_exception<RootCauseAnalysisException>(m, "RootCauseAnalysisException");
    py::register_exception<AnalysisTimeoutException>(m, "AnalysisTimeoutException");
    py::register_exception<CycleDetectionException>(m, "CycleDetectionException");

    // Bind utility functions
    m.def("format_root_cause_result", &format_root_cause_result,
          py::arg("result"),
          "Format root cause analysis result for display");

    // Bind CyberGrid classes and structures
    // Bind Cell struct
    py::class_<Cell>(m, "Cell")
        .def_readonly("x", &Cell::x)
        .def_readonly("y", &Cell::y)
        .def_readonly("alive", &Cell::alive)
        .def_readonly("energy", &Cell::energy)
        .def_readonly("occupants", &Cell::occupants)
        .def("agent_count", &Cell::agent_count)
        .def("is_empty", &Cell::is_empty)
        .def("can_occupy", &Cell::can_occupy)
        .def("is_pulsing", &Cell::is_pulsing);

    // Bind CyberGrid class (non-copyable)
    py::class_<CyberGrid>(m, "CyberGrid")
        .def(py::init<int, int>(), py::arg("width") = 100, py::arg("height") = 100)
        .def("width", &CyberGrid::width)
        .def("height", &CyberGrid::height)
        .def("cell_count", &CyberGrid::cell_count)
        .def("step", &CyberGrid::step, "Execute one full simulation cycle")
        .def("apply_conway_rules", &CyberGrid::apply_conway_rules,
             "Apply modified Conway's Game of Life rules with energy coupling")
        .def("apply_lora_pulses", &CyberGrid::apply_lora_pulses,
             "Propagate LoRA-style energy pulses through the grid")
        .def("place_agent", &CyberGrid::place_agent, py::arg("x"), py::arg("y"), py::arg("agent_id"))
        .def("remove_agent", &CyberGrid::remove_agent, py::arg("x"), py::arg("y"), py::arg("agent_id"))
        .def("get_agent_position", &CyberGrid::get_agent_position, py::arg("agent_id"))
        .def("count_living_neighbors", &CyberGrid::count_living_neighbors, py::arg("x"), py::arg("y"))
        .def("reset", &CyberGrid::reset)
        .def("randomize", &CyberGrid::randomize, py::arg("alive_probability"), py::arg("initial_energy"))
        .def("alive_cell_count", &CyberGrid::alive_cell_count)
        .def("occupied_cell_count", &CyberGrid::occupied_cell_count)
        .def("average_energy", &CyberGrid::average_energy)
        .def("grid_to_string", &CyberGrid::grid_to_string, py::arg("zoom_factor") = 1)
        .def("set_lora_parameters", &CyberGrid::set_lora_parameters,
             py::arg("attenuation_k") = 0.15f,
             py::arg("global_damping") = 0.95f,
             py::arg("activation_threshold") = 0.65f);

    // Bind CyberGrid configuration and factory
    py::class_<CyberGridConfig>(m, "CyberGridConfig")
        .def(py::init<>())
        .def_readwrite("width", &CyberGridConfig::width)
        .def_readwrite("height", &CyberGridConfig::height)
        .def_readwrite("initial_energy", &CyberGridConfig::initial_energy)
        .def_readwrite("alive_probability", &CyberGridConfig::alive_probability);

    py::class_<CyberGridFactory>(m, "CyberGridFactory")
        .def_static("create_standard_grid", &CyberGridFactory::create_standard_grid)
        .def_static("create_random_grid", &CyberGridFactory::create_random_grid,
                   py::arg("width"), py::arg("height"), py::arg("alive_prob"), py::arg("energy_prob"))
        .def_static("create_from_config", &CyberGridFactory::create_from_config, py::arg("config"));

    // Bind spatial analysis extension
    py::class_<SpatialDependencyLink>(m, "SpatialDependencyLink")
        .def_readonly("agent_id", &SpatialDependencyLink::agent_id)
        .def_readonly("strength", &SpatialDependencyLink::strength)
        .def_readonly("energy_correlation", &SpatialDependencyLink::energy_correlation)
        .def_readonly("distance", &SpatialDependencyLink::distance)
        .def_readonly("relationship_type", &SpatialDependencyLink::relationship_type);

    py::class_<SpatialRootCauseAnalyzer>(m, "SpatialRootCauseAnalyzer")
        .def(py::init<const CyberGrid&>(), py::arg("grid"))
        .def("build_spatial_dependencies", &SpatialRootCauseAnalyzer::build_spatial_dependencies)
        .def("find_spatial_cascades", &SpatialRootCauseAnalyzer::find_spatial_cascades)
        .def("calculate_spatial_clustering", &SpatialRootCauseAnalyzer::calculate_spatial_clustering,
             py::arg("chain"));

    // Bind exception classes
    py::register_exception<CyberGridException>(m, "CyberGridException");
    py::register_exception<InvalidPositionException>(m, "InvalidPositionException");
    py::register_exception<CellOccupancyException>(m, "CellOccupancyException");

    py::register_exception<RootCauseAnalysisException>(m, "RootCauseAnalysisException");
    py::register_exception<AnalysisTimeoutException>(m, "AnalysisTimeoutException");
    py::register_exception<CycleDetectionException>(m, "CycleDetectionException");

    // Bind utility functions
    m.def("format_root_cause_result", &format_root_cause_result,
          py::arg("result"),
          "Format root cause analysis result for display");

    // Type aliases for cleaner Python interface
    m.attr("AdjacencyList") = m.attr("dict");  // Python dict maps to our AdjacencyList

    // Module metadata
    m.attr("__version__") = "0.1.0";
    m.attr("__doc__") = "Swarm-100 C++ Core Modules - Alice in CyberLand: High-performance root cause analysis, cellular automata, and LoRA pulse simulation";
}
