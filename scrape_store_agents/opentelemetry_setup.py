"""
OpenTelemetry initialization for tracing and metrics.
This will auto-instrument supported libraries and export traces to OTLP (default localhost:4317).
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
import logging
import os

# Set up tracing
resource = Resource.create({"service.name": "scrape-store-agents"})
provider = TracerProvider(resource=resource)
# Use jaeger service name in Docker, fallback to localhost for development
endpoint = "http://jaeger:4318" if "DOCKER" in os.environ else "http://localhost:4318"
otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)
trace.set_tracer_provider(provider)

# Instrument FastAPI, requests, logging, etc.
FastAPIInstrumentor().instrument()
RequestsInstrumentor().instrument()
LoggingInstrumentor().instrument(set_logging_format=True)
AioHttpClientInstrumentor().instrument()
# Add more instrumentations as needed

# Set up logging export (optional)
logger_provider = LoggerProvider(resource=resource)
otlp_log_exporter = OTLPLogExporter(endpoint=endpoint)
log_record_processor = BatchLogRecordProcessor(otlp_log_exporter)
logger_provider.add_log_record_processor(log_record_processor)
logging.basicConfig(level=logging.INFO, handlers=[LoggingHandler(logger_provider=logger_provider)])

# Usage: import this module at the top of your main.py before app/server startup.
