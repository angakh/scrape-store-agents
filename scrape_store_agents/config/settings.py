"""Configuration management for scrape-store-agents."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)


class ScraperConfig(BaseModel):
    """Configuration for a scraper."""
    
    type: str = Field(..., description="Type of scraper (e.g., 'web')")
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; ScrapeStoreAgent/1.0)",
        description="User agent string"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_content_length: int = Field(
        default=10 * 1024 * 1024, 
        description="Maximum content length to process"
    )
    allowed_domains: List[str] = Field(
        default_factory=list,
        description="List of allowed domains"
    )
    blocked_domains: List[str] = Field(
        default_factory=list,
        description="List of blocked domains"
    )
    extract_links: bool = Field(
        default=False,
        description="Whether to extract and follow links"
    )
    max_depth: int = Field(
        default=1,
        description="Maximum crawling depth"
    )
    content_selectors: List[str] = Field(
        default_factory=lambda: [
            'main', 'article', '.content', '.post-content', '.entry-content',
            '.article-body', '.story-body', '#content'
        ],
        description="CSS selectors for content extraction"
    )
    title_selectors: List[str] = Field(
        default_factory=lambda: [
            'h1', 'title', '.title', '.headline', '.post-title'
        ],
        description="CSS selectors for title extraction"
    )
    remove_selectors: List[str] = Field(
        default_factory=lambda: [
            'script', 'style', 'nav', 'header', 'footer', '.sidebar',
            '.navigation', '.menu', '.ads', '.advertisement', '.social-share'
        ],
        description="CSS selectors for elements to remove"
    )


class VectorStoreConfig(BaseModel):
    """Configuration for a vector store."""
    
    type: str = Field(..., description="Type of vector store (e.g., 'chromadb')")
    collection_name: str = Field(
        default="documents",
        description="Name of the collection"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric for similarity search"
    )
    persist_directory: Optional[str] = Field(
        default=None,
        description="Directory to persist data"
    )
    host: Optional[str] = Field(
        default=None,
        description="Vector store server host"
    )
    port: Optional[int] = Field(
        default=None,
        description="Vector store server port"
    )


class SourceConfig(BaseModel):
    """Configuration for a scraping source."""
    
    name: str = Field(..., description="Name of the source")
    url: str = Field(..., description="URL to scrape")
    scraper_type: str = Field(default="web", description="Type of scraper to use")
    schedule: Optional[str] = Field(
        default=None,
        description="Cron schedule for automatic scraping"
    )
    enabled: bool = Field(default=True, description="Whether source is enabled")
    custom_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom configuration for this source"
    )


class APIConfig(BaseModel):
    """Configuration for the API server."""
    
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins"
    )


class AIConfig(BaseModel):
    """Configuration for AI features."""
    
    provider: str = Field(default="openai", description="AI provider: openai or anthropic")
    model: str = Field(default="gpt-3.5-turbo", description="Model name")
    temperature: float = Field(default=0.1, description="Temperature for AI responses")
    max_tokens: int = Field(default=1500, description="Maximum tokens in AI responses")
    reasoning_agent: bool = Field(default=True, description="Enable AI reasoning agent")
    intelligent_router: bool = Field(default=True, description="Enable intelligent scraper router")
    cache_analyses: bool = Field(default=True, description="Cache AI analyses to reduce API calls")


class APIKeysConfig(BaseModel):
    """Configuration for API keys."""
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file: Optional[str] = Field(
        default=None,
        description="Log file path"
    )


class OpenTelemetryConfig(BaseModel):
    """Configuration for OpenTelemetry observability."""
    
    enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing/metrics")
    service_name: str = Field(default="scrape-store-agents", description="Service name for traces")
    exporter: str = Field(default="otlp", description="Exporter type (otlp, jaeger, console)")
    otlp_endpoint: str = Field(default="http://localhost:4317", description="OTLP collector endpoint")
    jaeger_agent_host: str = Field(default="localhost", description="Jaeger agent host")
    jaeger_agent_port: int = Field(default=6831, description="Jaeger agent port")


class Settings(BaseModel):
    """Main application settings."""
    
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    opentelemetry: OpenTelemetryConfig = Field(default_factory=OpenTelemetryConfig)
    sources: List[SourceConfig] = Field(
        default_factory=list,
        description="List of scraping sources"
    )
    
    @validator('sources')
    def validate_sources(cls, v):
        """Validate that source names are unique."""
        names = [source.name for source in v]
        if len(names) != len(set(names)):
            raise ValueError("Source names must be unique")
        return v


def load_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration from file and environment variables.
    
    Args:
        config_path: Path to configuration file (YAML)
        
    Returns:
        Settings object with loaded configuration
    """
    # Default configuration
    config_data = {}
    
    # Load from file if provided
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config file {config_path}: {e}")
                raise
        else:
            logger.warning(f"Configuration file {config_path} not found")
    
    # Override with environment variables
    env_overrides = _get_env_overrides()
    config_data = _merge_configs(config_data, env_overrides)
    
    # Create Settings object
    try:
        settings = Settings(**config_data)
        logger.info("Configuration loaded successfully")
        return settings
    except Exception as e:
        logger.error(f"Error creating settings: {e}")
        raise


def _get_env_overrides() -> Dict[str, Any]:
    """Get configuration overrides from environment variables."""
    overrides = {}
    
    # API Keys
    if os.getenv('OPENAI_API_KEY'):
        overrides.setdefault('api_keys', {})['openai_api_key'] = os.getenv('OPENAI_API_KEY')
    if os.getenv('ANTHROPIC_API_KEY'):
        overrides.setdefault('api_keys', {})['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
    
    # AI configuration
    if os.getenv('AI_PROVIDER'):
        overrides.setdefault('ai', {})['provider'] = os.getenv('AI_PROVIDER')
    if os.getenv('AI_MODEL'):
        overrides.setdefault('ai', {})['model'] = os.getenv('AI_MODEL')
    if os.getenv('AI_TEMPERATURE'):
        overrides.setdefault('ai', {})['temperature'] = float(os.getenv('AI_TEMPERATURE'))
    if os.getenv('AI_MAX_TOKENS'):
        overrides.setdefault('ai', {})['max_tokens'] = int(os.getenv('AI_MAX_TOKENS'))
    
    # API configuration
    if os.getenv('API_HOST'):
        overrides.setdefault('api', {})['host'] = os.getenv('API_HOST')
    if os.getenv('API_PORT'):
        overrides.setdefault('api', {})['port'] = int(os.getenv('API_PORT'))
    if os.getenv('API_DEBUG'):
        overrides.setdefault('api', {})['debug'] = os.getenv('API_DEBUG').lower() == 'true'
    
    # Vector store configuration
    if os.getenv('VECTOR_STORE_TYPE'):
        overrides.setdefault('vector_store', {})['type'] = os.getenv('VECTOR_STORE_TYPE')
    if os.getenv('VECTOR_STORE_COLLECTION'):
        overrides.setdefault('vector_store', {})['collection_name'] = os.getenv('VECTOR_STORE_COLLECTION')
    if os.getenv('VECTOR_STORE_HOST'):
        overrides.setdefault('vector_store', {})['host'] = os.getenv('VECTOR_STORE_HOST')
    if os.getenv('VECTOR_STORE_PORT'):
        overrides.setdefault('vector_store', {})['port'] = int(os.getenv('VECTOR_STORE_PORT'))
    if os.getenv('VECTOR_STORE_PERSIST_DIR'):
        overrides.setdefault('vector_store', {})['persist_directory'] = os.getenv('VECTOR_STORE_PERSIST_DIR')
    
    # Logging configuration
    if os.getenv('LOG_LEVEL'):
        overrides.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
    if os.getenv('LOG_FILE'):
        overrides.setdefault('logging', {})['file'] = os.getenv('LOG_FILE')
    
    return overrides


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def setup_logging(config: LoggingConfig) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    
    # Configure logging
    logging_config = {
        'level': log_level,
        'format': config.format,
    }
    
    # Add file handler if specified
    if config.file:
        logging_config['filename'] = config.file
        logging_config['filemode'] = 'a'
    
    logging.basicConfig(**logging_config)
    
    # Set specific loggers to avoid noise
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def create_example_config() -> Dict[str, Any]:
    """Create an example configuration dictionary."""
    return {
        'scraper': {
            'type': 'web',
            'timeout': 30,
            'max_depth': 2,
            'extract_links': True,
            'allowed_domains': ['example.com', 'docs.example.com']
        },
        'vector_store': {
            'type': 'chromadb',
            'collection_name': 'my_documents',
            'embedding_model': 'all-MiniLM-L6-v2',
            'persist_directory': './data/chroma'
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False
        },
        'logging': {
            'level': 'INFO',
            'file': 'scrape-store-agents.log'
        },
        'sources': [
            {
                'name': 'example-docs',
                'url': 'https://docs.example.com',
                'scraper_type': 'web',
                'enabled': True,
                'schedule': '0 */6 * * *',  # Every 6 hours
                'custom_config': {
                    'max_depth': 3,
                    'content_selectors': ['.documentation-content']
                }
            },
            {
                'name': 'example-blog',
                'url': 'https://blog.example.com',
                'scraper_type': 'web',
                'enabled': True,
                'custom_config': {
                    'extract_links': True,
                    'title_selectors': ['.post-title', 'h1']
                }
            }
        ]
    }