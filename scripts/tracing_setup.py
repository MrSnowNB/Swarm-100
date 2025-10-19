#!/usr/bin/env python3
"""
---
script: tracing_setup.py
purpose: OpenTelemetry instrumentation setup for swarm observability
status: development
created: 2025-10-18
---
"""

import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes


class SwarmTracer:
    """Central tracing configuration for the swarm system"""

    def __init__(self, service_name="swarm-agent", service_version="1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
        self.tracer_provider = None
        self.tracer = None
        self._setup_tracing()

    def _setup_tracing(self):
        """Initialize OpenTelemetry tracing"""
        # Create resource
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.SERVICE_VERSION: self.service_version,
            ResourceAttributes.SERVICE_NAMESPACE: "swarm-100",
        })

        # Configure tracer provider
        self.tracer_provider = TracerProvider(resource=resource)

        # Configure OTLP exporter (for Jaeger, Grafana, etc.)
        otlp_exporter = OTLPSpanExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            insecure=True,
        )

        # Add batch span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        self.tracer_provider.add_span_processor(span_processor)

        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

        # Initialize auto-instrumentation
        self._setup_auto_instrumentation()

    def _setup_auto_instrumentation(self):
        """Setup automatic instrumentation for HTTP requests"""
        try:
            RequestsInstrumentor().instrument()
        except Exception as e:
            logging.warning(f"Failed to instrument requests: {e}")

    def get_tracer(self):
        """Get the configured tracer"""
        return self.tracer

    def create_swarm_span(self, operation_name, bot_id=None, gpu_id=None, **attributes):
        """Create a span for swarm operations"""
        span = self.tracer.start_as_span(operation_name)

        # Add standard swarm attributes
        if bot_id:
            span.set_attribute("swarm.bot_id", bot_id)
        if gpu_id:
            span.set_attribute("swarm.gpu_id", gpu_id)

        # Add custom attributes
        for key, value in attributes.items():
            span.set_attribute(f"swarm.{key}", value)

        return span

    def shutdown(self):
        """Shutdown tracing"""
        if self.tracer_provider:
            self.tracer_provider.shutdown()


# Global tracer instance
_swarm_tracer = None

def get_swarm_tracer():
    """Get or create the global swarm tracer"""
    global _swarm_tracer
    if _swarm_tracer is None:
        _swarm_tracer = SwarmTracer()
    return _swarm_tracer

def init_swarm_tracing(service_name="swarm-agent"):
    """Initialize tracing for the swarm system"""
    tracer = get_swarm_tracer()
    logging.info(f"Swarm tracing initialized for {service_name}")
    return tracer


# Context manager for swarm operations
class SwarmSpan:
    """Context manager for tracing swarm operations"""

    def __init__(self, operation_name, bot_id=None, gpu_id=None, **attributes):
        self.operation_name = operation_name
        self.bot_id = bot_id
        self.gpu_id = gpu_id
        self.attributes = attributes
        self.span = None
        self.tracer = get_swarm_tracer()

    def __enter__(self):
        self.span = self.tracer.create_swarm_span(
            self.operation_name,
            bot_id=self.bot_id,
            gpu_id=self.gpu_id,
            **self.attributes
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.record_exception(exc_val)
            self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
        else:
            self.span.set_status(trace.Status.OK)

        self.span.end()


# Utility functions for common swarm operations
def trace_message_flow(message_type, from_bot, to_bot, message_size=None):
    """Trace message flow between bots"""
    return SwarmSpan(
        "message_flow",
        operation="send_message",
        message_type=message_type,
        from_bot=from_bot,
        to_bot=to_bot,
        message_size=message_size
    )

def trace_consensus_operation(operation, bot_id, gpu_id, participants=None):
    """Trace consensus-related operations"""
    return SwarmSpan(
        "consensus_operation",
        bot_id=bot_id,
        gpu_id=gpu_id,
        operation=operation,
        participants=participants or []
    )

def trace_health_check(bot_id, gpu_id, check_type="periodic"):
    """Trace health check operations"""
    return SwarmSpan(
        "health_check",
        bot_id=bot_id,
        gpu_id=gpu_id,
        check_type=check_type
    )

def trace_task_execution(task_id, bot_id, gpu_id, task_type=None):
    """Trace task execution"""
    return SwarmSpan(
        "task_execution",
        bot_id=bot_id,
        gpu_id=gpu_id,
        task_id=task_id,
        task_type=task_type
    )
