"""Enhanced configuration loader with AI config support."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .settings import Settings, load_config


logger = logging.getLogger(__name__)


def load_ai_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load AI configuration from config file and environment.
    
    Args:
        config_path: Path to config.yml file
        
    Returns:
        Dictionary with AI configuration for ReasoningAgent and router
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yml"
    
    # Load full settings
    try:
        settings = load_config(str(config_path))
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        logger.info("Using environment variables and defaults")
        settings = Settings()  # Use defaults
    
    # Extract API key based on provider
    api_key = None
    if settings.ai.provider.lower() == 'openai':
        api_key = settings.api_keys.openai_api_key or os.getenv('OPENAI_API_KEY')
    elif settings.ai.provider.lower() == 'anthropic':
        api_key = settings.api_keys.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        logger.warning(f"No API key found for provider '{settings.ai.provider}'. "
                      f"Set it in config.yml or environment variable.")
    
    # Build LLM config
    llm_config = {
        'provider': settings.ai.provider,
        'model': settings.ai.model,
        'api_key': api_key,
        'temperature': settings.ai.temperature,
        'max_tokens': settings.ai.max_tokens
    }
    
    return llm_config


def get_scraper_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Get scraper configuration from config file.
    
    Args:
        config_path: Path to config.yml file
        
    Returns:
        Dictionary with scraper configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yml"
    
    try:
        settings = load_config(str(config_path))
        return settings.scraper.dict()
    except Exception as e:
        logger.warning(f"Could not load scraper config: {e}")
        # Return basic defaults
        return {
            'timeout': 30,
            'max_depth': 2,
            'extract_links': False
        }


def get_vector_store_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Get vector store configuration from config file.
    
    Args:
        config_path: Path to config.yml file
        
    Returns:
        Dictionary with vector store configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yml"
    
    try:
        settings = load_config(str(config_path))
        return settings.vector_store.dict()
    except Exception as e:
        logger.warning(f"Could not load vector store config: {e}")
        # Return basic defaults
        return {
            'type': 'chromadb',
            'collection_name': 'scraped_documents',
            'persist_directory': './data/chroma'
        }


def validate_ai_config(llm_config: Dict[str, Any]) -> bool:
    """Validate AI configuration has required fields.
    
    Args:
        llm_config: LLM configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['provider', 'model', 'api_key']
    
    for field in required_fields:
        if not llm_config.get(field):
            logger.error(f"Missing required AI config field: {field}")
            return False
    
    # Check provider is supported
    if llm_config['provider'].lower() not in ['openai', 'anthropic']:
        logger.error(f"Unsupported AI provider: {llm_config['provider']}")
        return False
    
    return True


def create_config_if_missing() -> None:
    """Create config.yml from example if it doesn't exist."""
    config_dir = Path(__file__).parent.parent.parent / "config"
    config_file = config_dir / "config.yml"
    example_file = config_dir / "config.example.yml"
    
    if not config_file.exists() and example_file.exists():
        try:
            config_file.write_text(example_file.read_text())
            logger.info(f"Created {config_file} from example template")
            logger.info("Please edit config.yml with your API keys and settings")
        except Exception as e:
            logger.error(f"Failed to create config.yml: {e}")
    elif not config_file.exists():
        logger.warning(f"No config.yml found and no example file at {example_file}")
        logger.info("Please create config.yml with your configuration")


def get_config_status() -> Dict[str, Any]:
    """Get status of configuration files and settings.
    
    Returns:
        Dictionary with configuration status information
    """
    config_dir = Path(__file__).parent.parent.parent / "config"
    config_file = config_dir / "config.yml"
    example_file = config_dir / "config.example.yml"
    
    status = {
        'config_exists': config_file.exists(),
        'example_exists': example_file.exists(),
        'config_path': str(config_file),
        'example_path': str(example_file),
        'ai_configured': False,
        'api_keys_found': {}
    }
    
    if config_file.exists():
        try:
            settings = load_config(str(config_file))
            
            # Check AI configuration
            llm_config = load_ai_config(str(config_file))
            status['ai_configured'] = validate_ai_config(llm_config)
            
            # Check which API keys are available
            status['api_keys_found'] = {
                'openai': bool(settings.api_keys.openai_api_key or os.getenv('OPENAI_API_KEY')),
                'anthropic': bool(settings.api_keys.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY'))
            }
            
        except Exception as e:
            status['error'] = str(e)
    
    return status