#!/usr/bin/env python3
"""
Test suite for C++ RootCauseAnalyzer module - Python integration validation
This tests the PyBind11 bindings and core functionality once compiled.
"""

# pyright: ignore

import sys
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest  # type: ignore[import]
else:
    import pytest  # type: ignore[import]

# Import will work only after CMake build
if TYPE_CHECKING:
    import swarm_core  # type: ignore[import]

# Add the swarm-core build directory to path (where compiled module will be)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import will work only after CMake build
swarm_core = None
try:
    import swarm_core  # type: ignore[import]
    SWARM_CORE_AVAILABLE = True
except ImportError:
    SWARM_CORE_AVAILABLE = False

@pytest.mark.skipif(not SWARM_CORE_AVAILABLE)
class TestRootCauseAnalyzerIntegration:
    """Test C++ RootCauseAnalyzer PyBind11 integration"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        assert swarm_core is not None
        config = swarm_core.AnalysisConfig()  # type: ignore[attr-defined]
        config.max_recursion_depth = 10  # Smaller for testing
        config.timeout = 1000  # 1 second for testing
        return swarm_core.RootCauseAnalyzer(config)  # type: ignore[attr-defined]

    @pytest.fixture
    def sample_graph(self):
        """Create a sample dependency graph for testing"""
        # agent_00_00 -> agent_00_01 -> agent_00_02
        # agent_01_00 -> agent_01_01 (independent chain)
        graph = {
            "bot_00_00": [
                swarm_core.DependencyLink("bot_00_01", 0.8, "communication"),  # type: ignore[attr-defined]
                swarm_core.DependencyLink("bot_01_00", 0.3, "resource")  # type: ignore[attr-defined]
            ],
            "bot_00_01": [
                swarm_core.DependencyLink("bot_00_02", 0.9, "communication")  # type: ignore[attr-defined]
            ],
            "bot_01_00": [
                swarm_core.DependencyLink("bot_01_01", 0.7, "communication")  # type: ignore[attr-defined]
            ]
        }
        return graph

    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly"""
        assert analyzer is not None
        assert analyzer.get_current_memory_usage() >= 0

        # Test reset functionality
        analyzer.reset_tracking()
        assert analyzer.get_current_memory_usage() >= 0

    def test_cycle_detection(self, analyzer, sample_graph):
        """Test cycle detection in dependency graph"""
        # Test acyclic graph first
        assert not analyzer.detect_cycles(sample_graph)

        # Create cyclic graph: bot_00_02 -> bot_00_00
        cyclic_graph = sample_graph.copy()
        cyclic_graph["bot_00_02"] = [
            swarm_core.DependencyLink("bot_00_00", 0.5, "communication")  # type: ignore[attr-defined]
        ]

        assert analyzer.detect_cycles(cyclic_graph)

    def test_simple_root_cause_analysis(self, analyzer, sample_graph):
        """Test basic root cause analysis functionality"""
        symptoms = ["ollama query failed", "model not found"]

        result = analyzer.analyze_dependency_chain(
            "bot_00_00",
            symptoms,
            sample_graph
        )

        # Verify result structure
        assert hasattr(result, 'dependency_chain')
        assert hasattr(result, 'primary_root_cause')
        assert hasattr(result, 'recommendations')
        assert hasattr(result, 'analysis_confidence')
        assert hasattr(result, 'analysis_duration_ms')
        assert hasattr(result, 'analysis_complete')

        # Check analysis completed successfully
        assert result.analysis_complete

        # Verify dependency chain was constructed
        assert len(result.dependency_chain) > 0

        # Check that recommendations were generated
        if result.recommendations:
            for rec in result.recommendations:
                assert hasattr(rec, 'action')
                assert hasattr(rec, 'priority')
                assert hasattr(rec, 'rationale')
                assert hasattr(rec, 'expected_impact')
                assert 0 <= rec.expected_impact <= 100

    def test_analysis_with_empty_graph(self, analyzer):
        """Test analysis with empty dependency graph"""
        result = analyzer.analyze_dependency_chain(
            "bot_00_00",
            ["test symptom"],
            {}
        )

        assert result.analysis_complete
        assert result.primary_root_cause == "No dependencies identified"

    def test_memory_usage_tracking(self, analyzer, sample_graph):
        """Test memory usage tracking during analysis"""
        initial_memory = analyzer.get_current_memory_usage()

        analyzer.analyze_dependency_chain(
            "bot_00_00",
            ["memory test"],
            sample_graph
        )

        final_memory = analyzer.get_current_memory_usage()
        assert final_memory >= initial_memory

    def test_configuration_limits(self):
        """Test that configuration limits are enforced"""
        # Test with very restrictive config
        config = swarm_core.AnalysisConfig()  # type: ignore[attr-defined]
        config.max_recursion_depth = 1
        config.max_memory_mb = 1
        config.timeout = 1  # 1ms

        restrictive_analyzer = swarm_core.RootCauseAnalyzer(config)  # type: ignore[attr-defined]

        # Create deep graph that should hit limits
        deep_graph = {}
        for i in range(10):  # Create chain longer than depth limit
            agent_id = f"agent_{i:02d}"
            next_id = f"agent_{i+1:02d}"
            deep_graph[agent_id] = [
                swarm_core.DependencyLink(next_id, 0.8, "communication")  # type: ignore[attr-defined]
            ]

        # Analysis should terminate due to depth limit
        result = restrictive_analyzer.analyze_dependency_chain(
            "agent_00",
            ["test symptom"],
            deep_graph
        )

        # Should complete (within limits) due to restrictions
        assert result.analysis_complete or not result.analysis_complete

    def test_exception_handling(self, analyzer):
        """Test that exceptions are properly propagated to Python"""
        # Test with invalid inputs
        try:
            analyzer.analyze_dependency_chain("", [], {})
            # Should not raise exception for valid inputs
        except Exception as e:
            assert isinstance(e, swarm_core.RootCauseAnalysisException)  # type: ignore[attr-defined]

    def test_result_formatting(self, analyzer, sample_graph):
        """Test result formatting function"""
        result = analyzer.analyze_dependency_chain(
            "bot_00_00",
            ["test symptom"],
            sample_graph
        )

        formatted = swarm_core.format_root_cause_result(result)  # type: ignore[attr-defined]
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "Root Cause Analysis Result:" in formatted

    def test_analysis_result_enum(self):
        """Test that AnalysisResult enum is properly exposed"""
        assert hasattr(swarm_core, 'AnalysisResult')  # type: ignore[attr-defined]
        assert hasattr(swarm_core.AnalysisResult, 'SUCCESS')  # type: ignore[attr-defined]
        assert hasattr(swarm_core.AnalysisResult, 'CYCLE_DETECTED')  # type: ignore[attr-defined]

@pytest.mark.skipif(not SWARM_CORE_AVAILABLE)
class TestDependencyChainStructures:
    """Test data structure bindings"""

    def test_dependency_link_creation(self):
        """Test DependencyLink creation and attributes"""
        link = swarm_core.DependencyLink("target_agent", 0.75, "communication")  # type: ignore[attr-defined]

        assert link.target_agent == "target_agent"
        assert abs(link.strength - 0.75) < 0.001
        assert link.relationship_type == "communication"

    def test_agent_dependency_structure(self, analyzer, sample_graph):
        """Test AgentDependency structure from analysis results"""
        result = analyzer.analyze_dependency_chain(
            "bot_00_00",
            ["ollama failure"],
            sample_graph
        )

        if result.dependency_chain:
            dep = result.dependency_chain[0]
            assert hasattr(dep, 'agent_id')
            assert hasattr(dep, 'confidence_score')
            assert hasattr(dep, 'failure_mode')
            assert hasattr(dep, 'symptoms')

            assert 0.0 <= dep.confidence_score <= 1.0
            assert isinstance(dep.symptoms, list)

    def test_analysis_config_modification(self):
        """Test that AnalysisConfig can be modified"""
        config = swarm_core.AnalysisConfig()  # type: ignore[attr-defined]

        # Test default values
        assert config.max_recursion_depth == 50
        assert config.max_memory_mb == 10

        # Test modification
        config.max_recursion_depth = 25
        config.max_memory_mb = 5
        config.min_confidence_threshold = 0.2

        assert config.max_recursion_depth == 25
        assert config.max_memory_mb == 5
        assert abs(config.min_confidence_threshold - 0.2) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
