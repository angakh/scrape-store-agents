"""Tests for ReasoningAgent functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
from datetime import datetime

from scrape_store_agents.agents.reasoning import ReasoningAgent
from scrape_store_agents.agents.base import Document


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, response_content: str):
        self.response_content = response_content
    
    def invoke(self, prompt: str):
        class MockResponse:
            def __init__(self, content):
                self.content = content
        return MockResponse(self.response_content)


@pytest.fixture
def mock_scraper():
    """Mock scraper for testing."""
    scraper = Mock()
    scraper.name = "MockScraper"
    scraper.validate_url = Mock(return_value=True)
    scraper.scrape = AsyncMock(return_value=[
        Document(
            content="Test content",
            url="https://example.com",
            title="Test Title",
            metadata={"test": "data"}
        )
    ])
    return scraper


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    store = Mock()
    store.name = "MockVectorStore"
    store.add_documents = AsyncMock()
    store.search = AsyncMock(return_value=[])
    store.get_document_count = AsyncMock(return_value=10)
    store.health_check = AsyncMock(return_value=True)
    return store


@pytest.fixture
def llm_config():
    """LLM configuration for testing."""
    return {
        'provider': 'openai',
        'model': 'gpt-3.5-turbo',
        'api_key': 'test-key',
        'temperature': 0.1,
        'max_tokens': 1000
    }


@pytest.fixture
def reasoning_agent(mock_scraper, mock_vector_store, llm_config):
    """Create reasoning agent for testing."""
    with patch('scrape_store_agents.agents.reasoning.ChatOpenAI') as mock_openai:
        mock_llm = MockLLM('{"test": "response"}')
        mock_openai.return_value = mock_llm
        
        agent = ReasoningAgent(mock_scraper, mock_vector_store, llm_config)
        agent.llm = mock_llm
        return agent


class TestReasoningAgent:
    """Test ReasoningAgent functionality."""
    
    def test_initialization(self, mock_scraper, mock_vector_store, llm_config):
        """Test agent initialization."""
        with patch('scrape_store_agents.agents.reasoning.ChatOpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            agent = ReasoningAgent(mock_scraper, mock_vector_store, llm_config)
            
            assert agent.scraper == mock_scraper
            assert agent.vector_store == mock_vector_store
            assert agent.llm_config == llm_config
            assert agent.memory == []
            assert agent.scraping_strategies == {}
    
    def test_openai_initialization(self, mock_scraper, mock_vector_store):
        """Test OpenAI LLM initialization."""
        config = {
            'provider': 'openai',
            'model': 'gpt-4',
            'api_key': 'test-key',
            'temperature': 0.5,
            'max_tokens': 2000
        }
        
        with patch('scrape_store_agents.agents.reasoning.ChatOpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            agent = ReasoningAgent(mock_scraper, mock_vector_store, config)
            
            mock_openai.assert_called_once_with(
                model='gpt-4',
                openai_api_key='test-key',
                temperature=0.5,
                max_tokens=2000
            )
    
    def test_anthropic_initialization(self, mock_scraper, mock_vector_store):
        """Test Anthropic LLM initialization."""
        config = {
            'provider': 'anthropic',
            'model': 'claude-3-sonnet',
            'api_key': 'test-key',
            'temperature': 0.3,
            'max_tokens': 1500
        }
        
        with patch('scrape_store_agents.agents.reasoning.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = Mock()
            
            agent = ReasoningAgent(mock_scraper, mock_vector_store, config)
            
            mock_anthropic.assert_called_once_with(
                model='claude-3-sonnet',
                anthropic_api_key='test-key',
                temperature=0.3,
                max_tokens_to_sample=1500
            )
    
    def test_unsupported_provider(self, mock_scraper, mock_vector_store):
        """Test error handling for unsupported LLM provider."""
        config = {
            'provider': 'unsupported',
            'model': 'test-model',
            'api_key': 'test-key'
        }
        
        with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
            ReasoningAgent(mock_scraper, mock_vector_store, config)
    
    @pytest.mark.asyncio
    async def test_analyze_url_success(self, reasoning_agent):
        """Test successful URL analysis."""
        url = "https://example.com/article"
        
        # Mock successful LLM response
        analysis_response = {
            "site_type": "news",
            "challenges": ["rate_limiting"],
            "recommended_selectors": ["article", ".content"],
            "rate_limit_strategy": "conservative",
            "scraping_approach": "standard",
            "confidence": 0.85
        }
        
        reasoning_agent.llm = MockLLM(json.dumps(analysis_response))
        
        result = await reasoning_agent.analyze_url(url)
        
        assert result == analysis_response
        assert len(reasoning_agent.memory) == 1
        assert reasoning_agent.memory[0]['action'] == 'url_analysis'
        assert reasoning_agent.memory[0]['url'] == url
        assert reasoning_agent.memory[0]['analysis'] == analysis_response
    
    @pytest.mark.asyncio
    async def test_analyze_url_fallback(self, reasoning_agent):
        """Test URL analysis fallback when LLM fails."""
        url = "https://news.example.com/article"
        
        # Mock LLM failure
        reasoning_agent.llm = MockLLM("Invalid JSON response")
        
        result = await reasoning_agent.analyze_url(url)
        
        assert result['site_type'] == 'news'  # Should detect news site
        assert result['confidence'] == 0.3  # Fallback confidence
        assert 'recommended_selectors' in result
    
    @pytest.mark.asyncio
    async def test_adapt_scraping_strategy(self, reasoning_agent):
        """Test scraping strategy adaptation."""
        url = "https://example.com"
        failed_attempts = [
            {
                'error': 'timeout',
                'strategy': 'standard',
                'timestamp': '2024-01-01T00:00:00'
            },
            {
                'error': 'blocked',
                'strategy': 'retry',
                'timestamp': '2024-01-01T00:01:00'
            }
        ]
        
        adaptation_response = {
            "new_selectors": ["body", ".main-content"],
            "request_modifications": {"headers": {}, "delay": 3.0},
            "alternative_approach": "use different user agent",
            "success_probability": 0.75,
            "rationale": "Previous failures suggest anti-bot measures"
        }
        
        reasoning_agent.llm = MockLLM(json.dumps(adaptation_response))
        
        result = await reasoning_agent.adapt_scraping_strategy(url, failed_attempts)
        
        assert result == adaptation_response
        assert len(reasoning_agent.memory) == 1
        assert reasoning_agent.memory[0]['action'] == 'strategy_adaptation'
        assert reasoning_agent.memory[0]['failed_attempts_count'] == 2
    
    @pytest.mark.asyncio
    async def test_decide_storage_strategy(self, reasoning_agent):
        """Test storage strategy decision."""
        documents = [
            Document(
                content="Long article content " * 100,
                url="https://example.com/article1",
                title="Article 1",
                metadata={"content_type": "article"}
            ),
            Document(
                content="Short content",
                url="https://example.com/article2", 
                title="Article 2",
                metadata={"content_type": "article"}
            )
        ]
        
        storage_response = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "metadata_strategy": "extract_key_fields",
            "embedding_approach": "semantic_chunks",
            "index_organization": "by_domain",
            "rationale": "Medium-length articles benefit from semantic chunking"
        }
        
        reasoning_agent.llm = MockLLM(json.dumps(storage_response))
        
        result = await reasoning_agent.decide_storage_strategy(documents)
        
        assert result == storage_response
    
    @pytest.mark.asyncio
    async def test_decide_storage_strategy_empty_documents(self, reasoning_agent):
        """Test storage strategy with no documents."""
        result = await reasoning_agent.decide_storage_strategy([])
        
        assert result['strategy'] == 'none'
        assert result['rationale'] == 'No documents to store'
    
    @pytest.mark.asyncio
    async def test_get_reasoning_stats(self, reasoning_agent):
        """Test reasoning statistics collection."""
        # Add some memory entries
        reasoning_agent.memory = [
            {'action': 'url_analysis', 'timestamp': datetime.utcnow()},
            {'action': 'strategy_adaptation', 'timestamp': datetime.utcnow()},
            {'action': 'url_analysis', 'timestamp': datetime.utcnow()}
        ]
        
        stats = await reasoning_agent.get_reasoning_stats()
        
        assert stats['memory_entries'] == 3
        assert stats['url_analyses'] == 2
        assert stats['strategy_adaptations'] == 1
        assert stats['llm_provider'] == 'openai'
        assert 'scraper' in stats  # From base class
        assert 'vector_store' in stats  # From base class
    
    def test_clear_memory(self, reasoning_agent):
        """Test memory clearing."""
        reasoning_agent.memory = [
            {'action': 'test', 'timestamp': datetime.utcnow()}
        ]
        
        reasoning_agent.clear_memory()
        
        assert reasoning_agent.memory == []
    
    @pytest.mark.asyncio
    async def test_fallback_url_analysis_blog(self, reasoning_agent):
        """Test fallback analysis for blog URLs."""
        url = "https://blog.example.com/post"
        
        result = reasoning_agent._fallback_url_analysis(url)
        
        assert result['site_type'] == 'blog'
        assert result['confidence'] == 0.3
    
    @pytest.mark.asyncio
    async def test_fallback_url_analysis_docs(self, reasoning_agent):
        """Test fallback analysis for documentation URLs."""
        url = "https://docs.example.com/guide"
        
        result = reasoning_agent._fallback_url_analysis(url)
        
        assert result['site_type'] == 'documentation'
        assert result['confidence'] == 0.3