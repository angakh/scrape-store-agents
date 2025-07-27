"""Tests for configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path

from scrape_store_agents.config.settings import (
    Settings,
    ScraperConfig,
    VectorStoreConfig,
    SourceConfig,
    load_config,
    create_example_config
)


class TestScraperConfig:
    """Test ScraperConfig class."""
    
    def test_default_scraper_config(self):
        """Test default scraper configuration."""
        config = ScraperConfig(type="web")
        
        assert config.type == "web"
        assert config.timeout == 30
        assert config.max_depth == 1
        assert config.extract_links is False
        assert len(config.content_selectors) > 0
        assert len(config.title_selectors) > 0
    
    def test_custom_scraper_config(self):
        """Test custom scraper configuration."""
        config = ScraperConfig(
            type="custom",
            timeout=60,
            max_depth=3,
            extract_links=True,
            allowed_domains=["example.com"]
        )
        
        assert config.type == "custom"
        assert config.timeout == 60
        assert config.max_depth == 3
        assert config.extract_links is True
        assert config.allowed_domains == ["example.com"]


class TestVectorStoreConfig:
    """Test VectorStoreConfig class."""
    
    def test_default_vector_store_config(self):
        """Test default vector store configuration."""
        config = VectorStoreConfig(type="chromadb")
        
        assert config.type == "chromadb"
        assert config.collection_name == "documents"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.distance_metric == "cosine"
        assert config.persist_directory is None
        assert config.host is None
        assert config.port is None
    
    def test_custom_vector_store_config(self):
        """Test custom vector store configuration."""
        config = VectorStoreConfig(
            type="custom",
            collection_name="test_docs",
            host="localhost",
            port=8001,
            persist_directory="/data"
        )
        
        assert config.type == "custom"
        assert config.collection_name == "test_docs"
        assert config.host == "localhost"
        assert config.port == 8001
        assert config.persist_directory == "/data"


class TestSourceConfig:
    """Test SourceConfig class."""
    
    def test_source_config(self):
        """Test source configuration."""
        config = SourceConfig(
            name="test-source",
            url="https://example.com"
        )
        
        assert config.name == "test-source"
        assert config.url == "https://example.com"
        assert config.scraper_type == "web"
        assert config.enabled is True
        assert config.schedule is None
        assert config.custom_config == {}
    
    def test_source_config_with_options(self):
        """Test source configuration with all options."""
        custom_config = {"max_depth": 3}
        
        config = SourceConfig(
            name="test-source",
            url="https://example.com",
            scraper_type="custom",
            enabled=False,
            schedule="0 */6 * * *",
            custom_config=custom_config
        )
        
        assert config.name == "test-source"
        assert config.url == "https://example.com"
        assert config.scraper_type == "custom"
        assert config.enabled is False
        assert config.schedule == "0 */6 * * *"
        assert config.custom_config == custom_config


class TestSettings:
    """Test Settings class."""
    
    def test_default_settings(self):
        """Test default settings."""
        settings = Settings()
        
        assert isinstance(settings.scraper, ScraperConfig)
        assert isinstance(settings.vector_store, VectorStoreConfig)
        assert settings.sources == []
    
    def test_settings_validation_unique_source_names(self):
        """Test that source names must be unique."""
        sources = [
            SourceConfig(name="test", url="https://example1.com"),
            SourceConfig(name="test", url="https://example2.com")
        ]
        
        with pytest.raises(ValueError, match="Source names must be unique"):
            Settings(sources=sources)
    
    def test_settings_validation_unique_names_pass(self):
        """Test that unique source names pass validation."""
        sources = [
            SourceConfig(name="test1", url="https://example1.com"),
            SourceConfig(name="test2", url="https://example2.com")
        ]
        
        settings = Settings(sources=sources)
        assert len(settings.sources) == 2


class TestLoadConfig:
    """Test configuration loading."""
    
    def test_load_config_no_file(self):
        """Test loading config when no file exists."""
        settings = load_config("nonexistent.yaml")
        
        # Should return default settings
        assert isinstance(settings, Settings)
        assert settings.scraper.type == "web"
    
    def test_load_config_from_file(self):
        """Test loading config from file."""
        config_data = {
            "scraper": {
                "type": "web",
                "timeout": 60,
                "max_depth": 3
            },
            "vector_store": {
                "type": "chromadb",
                "collection_name": "test_docs"
            },
            "sources": [
                {
                    "name": "test-source",
                    "url": "https://example.com",
                    "enabled": True
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            settings = load_config(config_path)
            
            assert settings.scraper.timeout == 60
            assert settings.scraper.max_depth == 3
            assert settings.vector_store.collection_name == "test_docs"
            assert len(settings.sources) == 1
            assert settings.sources[0].name == "test-source"
            
        finally:
            Path(config_path).unlink()
    
    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            config_path = f.name
        
        try:
            with pytest.raises(Exception):
                load_config(config_path)
        finally:
            Path(config_path).unlink()


class TestCreateExampleConfig:
    """Test example configuration creation."""
    
    def test_create_example_config(self):
        """Test creating example configuration."""
        config = create_example_config()
        
        assert "scraper" in config
        assert "vector_store" in config
        assert "api" in config
        assert "logging" in config
        assert "sources" in config
        
        assert config["scraper"]["type"] == "web"
        assert config["vector_store"]["type"] == "chromadb"
        assert len(config["sources"]) > 0
        
        # Validate sources have required fields
        for source in config["sources"]:
            assert "name" in source
            assert "url" in source
            assert "enabled" in source