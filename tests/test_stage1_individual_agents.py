#!/usr/bin/env python3
"""
---
file: test_stage1_individual_agents.py
purpose: Stage 1 - Individual Agent Validation (95% coverage target)
framework: pytest with coverage reporting
status: development
created: 2025-10-18
---
"""

import pytest
import time
import requests
import json
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import yaml
from datetime import datetime
import logging

# Import the system under test
from scripts.bot_worker import BotWorker
from scripts.tracing_setup import SwarmSpan, get_swarm_tracer


class TestBotWorker:
    """Comprehensive tests for individual BotWorker functionality"""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'model': {
                'name': 'granite4:micro-h',
                'temperature': 0.7,
                'top_p': 0.9
            },
            'bot': {
                'base_port': 11400,
                'api_timeout': 30
            }
        }

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing"""
        return Mock()

    @pytest.fixture
    def bot_worker(self, sample_config):
        """Create a test bot worker instance"""
        with patch('scripts.bot_worker.logging.getLogger'), \
             patch('scripts.bot_worker.init_swarm_tracing'):
            bot = BotWorker("test_bot_00", 0, 11400)
            # Override the logger and tracer for testing
            bot.logger = Mock()
            bot.swarm_tracer = Mock()
            return bot

    def test_bot_initialization(self, bot_worker):
        """Test bot worker initialization"""
        assert bot_worker.bot_id == "test_bot_00"
        assert bot_worker.gpu_id == 0
        assert bot_worker.port == 11400
        assert bot_worker.ollama_url == "http://localhost:11434"
        assert bot_worker.stats == {
            'requests': 0,
            'errors': 0,
            'start_time': bot_worker.stats['start_time']
        }
        assert isinstance(bot_worker.memory, list)

    @patch('scripts.bot_worker.requests.post')
    def test_query_ollama_success(self, mock_post, bot_worker):
        """Test successful Ollama query"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Test response'}
        mock_post.return_value = mock_response

        result = bot_worker.query_ollama("Test prompt")

        assert result == "Test response"
        assert bot_worker.stats['requests'] == 1
        assert bot_worker.stats['errors'] == 0
        mock_post.assert_called_once()

        # Verify the call arguments
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"
        json_payload = call_args[1]['json']
        assert json_payload['model'] == 'granite4:micro-h'
        assert json_payload['prompt'] == 'Test prompt'

    @patch('scripts.bot_worker.requests.post')
    def test_query_ollama_http_error(self, mock_post, bot_worker):
        """Test Ollama query with HTTP error"""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = bot_worker.query_ollama("Test prompt")

        assert result is None
        assert bot_worker.stats['requests'] == 0
        assert bot_worker.stats['errors'] == 1
        bot_worker.logger.error.assert_called()

    @patch('scripts.bot_worker.requests.post')
    def test_query_ollama_timeout(self, mock_post, bot_worker):
        """Test Ollama query with timeout"""
        # Mock timeout exception
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        result = bot_worker.query_ollama("Test prompt")

        assert result is None
        assert bot_worker.stats['requests'] == 0
        assert bot_worker.stats['errors'] == 1
        bot_worker.logger.error.assert_called()

    @patch('scripts.bot_worker.requests.post')
    def test_query_ollama_connection_error(self, mock_post, bot_worker):
        """Test Ollama query with connection error"""
        # Mock connection error
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = bot_worker.query_ollama("Test prompt")

        assert result is None
        assert bot_worker.stats['requests'] == 0
        assert bot_worker.stats['errors'] == 1
        bot_worker.logger.error.assert_called()

    @patch.object(BotWorker, 'query_ollama')
    def test_health_check_success(self, mock_query, bot_worker):
        """Test successful health check"""
        mock_query.return_value = "pong"

        result = bot_worker.health_check()

        assert result is True
        bot_worker.logger.info.assert_called()
        mock_query.assert_called_with("ping")

    @patch.object(BotWorker, 'query_ollama')
    def test_health_check_failure(self, mock_query, bot_worker):
        """Test failed health check"""
        mock_query.return_value = None

        result = bot_worker.health_check()

        assert result is False
        bot_worker.logger.error.assert_called()

    def test_stats_tracking(self, bot_worker):
        """Test statistics tracking"""
        initial_requests = bot_worker.stats['requests']
        initial_errors = bot_worker.stats['errors']

        # Simulate some operations
        with patch.object(bot_worker, 'query_ollama') as mock_query:
            # Successful query
            mock_query.return_value = "response"
            bot_worker.query_ollama("test")

            # Failed query
            mock_query.return_value = None
            bot_worker.query_ollama("test")

        assert bot_worker.stats['requests'] == initial_requests + 1
        assert bot_worker.stats['errors'] == initial_errors + 1

    def test_memory_management(self, bot_worker):
        """Test bot memory management"""
        assert len(bot_worker.memory) == 0

        # Add some memory entries
        bot_worker.memory.append("memory item 1")
        bot_worker.memory.append("memory item 2")

        assert len(bot_worker.memory) == 2
        assert bot_worker.memory[0] == "memory item 1"
        assert bot_worker.memory[1] == "memory item 2"

    @patch('scripts.bot_worker.time.sleep')
    @patch.object(BotWorker, 'health_check')
    def test_run_loop_initialization(self, mock_health_check, mock_sleep, bot_worker):
        """Test bot run loop initialization"""
        mock_health_check.return_value = True

        # Mock to stop after initialization
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            try:
                bot_worker.run()
            except KeyboardInterrupt:
                pass

        # Verify initialization sequence
        mock_sleep.assert_called_with(2)  # Initial delay
        mock_health_check.assert_called()

    @patch('scripts.bot_worker.time.sleep')
    @patch.object(BotWorker, 'health_check')
    def test_run_loop_with_failure(self, mock_health_check, mock_sleep, bot_worker):
        """Test bot run loop with health check failure"""
        mock_health_check.return_value = False

        # Should exit early on health check failure
        result = bot_worker.run()

        assert result is None
        bot_worker.logger.error.assert_called()

    def test_tracing_initialization(self):
        """Test that tracing is properly initialized"""
        with patch('scripts.bot_worker.init_swarm_tracing') as mock_tracing:
            bot = BotWorker("test_bot", 0, 11400)
            mock_tracing.assert_called_once_with("bot-test_bot")

    def test_swarm_span_context_manager(self):
        """Test SwarmSpan context manager"""
        tracer = get_swarm_tracer()
        with patch.object(tracer, 'create_swarm_span') as mock_span:
            mock_span_obj = Mock()
            mock_span.return_value = mock_span_obj

            with SwarmSpan("test_operation", bot_id="test_bot", gpu_id=0) as span:
                assert span == mock_span_obj

            mock_span_obj.end.assert_called_once()

    def test_swarm_span_exception_handling(self):
        """Test SwarmSpan handles exceptions correctly"""
        tracer = get_swarm_tracer()
        with patch.object(tracer, 'create_swarm_span') as mock_span:
            mock_span_obj = Mock()
            mock_span.return_value = mock_span_obj

            with pytest.raises(ValueError):
                with SwarmSpan("test_operation") as span:
                    raise ValueError("Test exception")

            mock_span_obj.record_exception.assert_called_once()
            mock_span_obj.set_status.assert_called()
            mock_span_obj.end.assert_called_once()

    @pytest.mark.parametrize("prompt_length,expected_success", [
        (10, True),
        (100, True),
        (1000, True),
        (0, True),  # Empty prompt
    ])
    @patch('scripts.bot_worker.requests.post')
    def test_query_ollama_various_prompt_lengths(self, mock_post, bot_worker, prompt_length, expected_success):
        """Test query handling with various prompt lengths"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'OK'}
        mock_post.return_value = mock_response

        prompt = "A" * prompt_length
        result = bot_worker.query_ollama(prompt)

        if expected_success:
            assert result == "OK"
        else:
            assert result is None


class TestConfigurationValidation:
    """Test configuration validation"""

    def test_config_file_exists(self):
        """Test that configuration file exists"""
        config_path = 'configs/swarm_config.yaml'
        assert os.path.exists(config_path), f"Config file {config_path} does not exist"

    def test_config_file_valid_yaml(self):
        """Test that configuration file is valid YAML"""
        config_path = 'configs/swarm_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict)
        assert 'project' in config
        assert 'version' in config
        assert 'hardware' in config
        assert 'model' in config

    def test_required_config_sections(self):
        """Test that all required configuration sections exist"""
        config_path = 'configs/swarm_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert isinstance(config, dict)

        required_sections = ['project', 'hardware', 'model', 'swarm', 'bot', 'performance', 'logging', 'monitoring']
        for section in required_sections:
            assert section in config, f"Missing required config section: {section}"


class TestLoggingSetup:
    """Test logging configuration"""

    def test_log_directory_structure(self):
        """Test that log directories exist"""
        log_dirs = ['logs', 'logs/gpu0', 'logs/gpu1', 'logs/gpu2', 'logs/gpu3']
        for log_dir in log_dirs:
            assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist"

    @patch('scripts.bot_worker.logging.basicConfig')
    def test_bot_logging_configuration(self, mock_config):
        """Test that bot logging is properly configured"""
        with patch('scripts.bot_worker.init_swarm_tracing'):
            bot = BotWorker("test_bot", 0, 11400)

            # Verify logging was configured
            mock_config.assert_called_once()
            call_args = mock_config.call_args
            assert 'format' in call_args[1]
            assert 'level' in call_args[1]


# Performance benchmarks for individual agents
@pytest.mark.benchmark
class TestBotPerformance:
    """Performance benchmarks for individual bot operations"""

    @pytest.fixture
    def performance_bot(self):
        """Bot instance for performance testing"""
        with patch('scripts.bot_worker.init_swarm_tracing'), \
             patch('scripts.bot_worker.logging.getLogger'):
            return BotWorker("perf_bot", 0, 11400)

    @pytest.mark.benchmark
    def test_query_latency_baseline(self, benchmark, performance_bot):
        """Benchmark query latency"""
        def query():
            return performance_bot.query_ollama("What is 2+2?")

        benchmark(query)

    @pytest.mark.benchmark
    def test_health_check_performance(self, benchmark, performance_bot):
        """Benchmark health check performance"""
        def health():
            return performance_bot.health_check()

        benchmark(health)

    @pytest.mark.benchmark
    def test_memory_operations(self, benchmark, performance_bot):
        """Benchmark memory operations"""
        def memory_op():
            performance_bot.memory.append("test entry")
            return len(performance_bot.memory)

        benchmark(memory_op)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=scripts/bot_worker", "--cov-report=html", "--cov-fail-under=95"])
