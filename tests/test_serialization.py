#!/usr/bin/env python3
"""
Test suite for Swarm-100 serialization functions
Tests the to_serializable function that handles NumPy object conversion for YAML compatibility
"""

import pytest
import numpy as np
import yaml
import json


def to_serializable(val):
    """Copy of the serialization function from perturbation_resilience_test.py"""
    if isinstance(val, (np.generic, np.ndarray)):
        return val.tolist()
    return val


class TestSerialization:
    """Unit tests for NumPy serialization robustness"""

    def test_numpy_scalar_int_conversion(self):
        """Test conversion of numpy scalar integers"""
        original = np.int32(42)
        result = to_serializable(original)

        assert result == 42
        assert isinstance(result, int)
        assert not isinstance(result, np.generic)

    def test_numpy_scalar_float_conversion(self):
        """Test conversion of numpy scalar floats"""
        original = np.float64(3.14159)
        result = to_serializable(original)

        assert abs(result - 3.14159) < 1e-10
        assert isinstance(result, float)
        assert not isinstance(result, np.generic)

    def test_numpy_array_1d_conversion(self):
        """Test conversion of 1D numpy arrays"""
        original = np.array([1, 2, 3, 4, 5])
        result = to_serializable(original)

        assert result == [1, 2, 3, 4, 5]
        assert isinstance(result, list)
        assert not isinstance(result, np.ndarray)

    def test_numpy_array_2d_conversion(self):
        """Test conversion of 2D numpy arrays"""
        original = np.array([[1, 2], [3, 4]])
        result = to_serializable(original)

        assert result == [[1, 2], [3, 4]]
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)

    def test_numpy_float_array_conversion(self):
        """Test conversion of float numpy arrays"""
        original = np.array([1.1, 2.2, 3.3])
        result = to_serializable(original)

        assert len(result) == 3
        assert abs(result[0] - 1.1) < 1e-10
        assert isinstance(result, list)

    def test_regular_python_types_passthrough(self):
        """Test that regular Python types pass through unchanged"""
        test_cases = [
            42,
            3.14,
            "hello",
            [1, 2, 3],
            {"key": "value"},
            None,
            True,
            False
        ]

        for original in test_cases:
            result = to_serializable(original)
            assert result == original
            assert type(result) == type(original)

    def test_mixed_data_structure_conversion(self):
        """Test conversion of complex nested data structures"""
        original = {
            "metadata": {
                "numpy_int": np.int64(100),
                "numpy_float": np.float32(2.5),
                "regular_string": "test"
            },
            "data": {
                "array_1d": np.array([10, 20, 30]),
                "array_2d": np.array([[1, 2], [3, 4]]),
                "mixed_list": [np.int32(5), "string", np.float64(1.23)]
            }
        }

        result = to_serializable(original)

        # Check structure is preserved
        assert isinstance(result, dict)
        assert "metadata" in result
        assert "data" in result

        # Check numpy conversions
        assert result["metadata"]["numpy_int"] == 100
        assert isinstance(result["metadata"]["numpy_int"], int)

        assert abs(result["metadata"]["numpy_float"] - 2.5) < 1e-6
        assert isinstance(result["metadata"]["numpy_float"], float)

        # Check arrays converted
        assert result["data"]["array_1d"] == [10, 20, 30]
        assert result["data"]["array_2d"] == [[1, 2], [3, 4]]

        # Check mixed list
        assert result["data"]["mixed_list"][0] == 5
        assert isinstance(result["data"]["mixed_list"][0], int)
        assert result["data"]["mixed_list"][1] == "string"
        assert abs(result["data"]["mixed_list"][2] - 1.23) < 1e-10

    def test_yaml_safe_dump_compatibility(self):
        """Test that serialized data can be safely dumped to YAML"""
        original_data = {
            "recovery_time": np.int32(27),
            "final_similarity": np.float64(0.960),
            "trajectory": np.array([0.1, 0.3, 0.8, 0.95]),
            "metadata": {
                "agent_count": np.int64(100),
                "config": "baseline"
            }
        }

        # Convert to serializable
        serializable = to_serializable(original_data)

        # Should be able to dump to YAML without errors
        yaml_str = yaml.safe_dump(serializable)

        # Should be able to load back
        loaded = yaml.safe_load(yaml_str)

        # Verify data integrity
        assert loaded["recovery_time"] == 27
        assert abs(loaded["final_similarity"] - 0.960) < 1e-10
        assert loaded["trajectory"] == [0.1, 0.3, 0.8, 0.95]
        assert loaded["metadata"]["agent_count"] == 100

    def test_json_double_conversion_compatibility(self):
        """Test JSON double-conversion approach for numpy handling"""
        original_data = {
            "numpy_val": np.float64(1.234),
            "numpy_array": np.array([1, 2, 3]),
            "nested": {
                "another_numpy": np.int32(42)
            }
        }

        # JSON double-conversion approach
        cleaned_results = json.loads(json.dumps(original_data, default=to_serializable))

        # Verify conversion worked
        assert isinstance(cleaned_results["numpy_val"], float)
        assert cleaned_results["numpy_array"] == [1, 2, 3]
        assert cleaned_results["nested"]["another_numpy"] == 42

        # Should be YAML-safe
        yaml_str = yaml.safe_dump(cleaned_results)
        assert "python/object" not in yaml_str
        assert "!!binary" not in yaml_str

    def test_numpy_dtypes_coverage(self):
        """Test coverage of various numpy dtypes"""
        test_cases = [
            (np.int8(1), int),
            (np.int16(2), int),
            (np.int32(3), int),
            (np.int64(4), int),
            (np.float16(1.5), float),
            (np.float32(2.5), float),
            (np.float64(3.5), float),
        ]

        for original, expected_type in test_cases:
            result = to_serializable(original)
            assert isinstance(result, expected_type)
            assert not isinstance(result, np.generic)

    def test_edge_cases(self):
        """Test edge cases and potential error conditions"""
        # Empty array
        empty_array = np.array([])
        result = to_serializable(empty_array)
        assert result == []

        # Single element array
        single_array = np.array([42])
        result = to_serializable(single_array)
        assert result == [42]

        # Zero-dimensional array
        zero_dim = np.array(5)
        result = to_serializable(zero_dim)
        assert result == 5
        assert isinstance(result, int)
