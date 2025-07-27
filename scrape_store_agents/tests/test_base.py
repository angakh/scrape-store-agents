"""Tests for base classes."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from scrape_store_agents.agents.base import Document, SearchResult, Agent


class TestDocument:
    """Test Document class."""
    
    def test_document_creation(self):
        """Test document creation with required fields."""
        doc = Document(
            content="Test content",
            url="https://example.com"
        )
        
        assert doc.content == "Test content"
        assert doc.url == "https://example.com"
        assert doc.title is None
        assert doc.metadata == {}
        assert isinstance(doc.timestamp, datetime)
    
    def test_document_with_optional_fields(self):
        """Test document creation with all fields."""
        timestamp = datetime.utcnow()
        metadata = {"key": "value"}
        
        doc = Document(
            content="Test content",
            url="https://example.com",
            title="Test Title",
            metadata=metadata,
            timestamp=timestamp
        )
        
        assert doc.content == "Test content"
        assert doc.url == "https://example.com"
        assert doc.title == "Test Title"
        assert doc.metadata == metadata
        assert doc.timestamp == timestamp


class TestSearchResult:
    """Test SearchResult class."""
    
    def test_search_result_creation(self):
        """Test search result creation."""
        doc = Document(content="Test", url="https://example.com")
        result = SearchResult(document=doc, score=0.95)
        
        assert result.document == doc
        assert result.score == 0.95
        assert result.distance is None
    
    def test_search_result_with_distance(self):
        """Test search result with distance."""
        doc = Document(content="Test", url="https://example.com")
        result = SearchResult(document=doc, score=0.95, distance=0.05)
        
        assert result.document == doc
        assert result.score == 0.95
        assert result.distance == 0.05


class TestAgent:
    """Test Agent class."""
    
    @pytest.fixture
    def mock_scraper(self):
        """Create mock scraper."""
        scraper = MagicMock()
        scraper.name = "MockScraper"
        scraper.validate_url = MagicMock(return_value=True)
        scraper.scrape = AsyncMock(return_value=[
            Document(content="Test content", url="https://example.com")
        ])
        return scraper
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = MagicMock()
        store.name = "MockVectorStore"
        store.add_documents = AsyncMock()
        store.search = AsyncMock(return_value=[])
        store.get_document_count = AsyncMock(return_value=10)
        store.health_check = AsyncMock(return_value=True)
        return store
    
    @pytest.fixture
    def agent(self, mock_scraper, mock_vector_store):
        """Create agent with mocked dependencies."""
        return Agent(mock_scraper, mock_vector_store)
    
    @pytest.mark.asyncio
    async def test_scrape_and_store_success(self, agent, mock_scraper, mock_vector_store):
        """Test successful scrape and store operation."""
        url = "https://example.com"
        
        result = await agent.scrape_and_store(url)
        
        assert result == 1
        mock_scraper.validate_url.assert_called_once_with(url)
        mock_scraper.scrape.assert_called_once_with(url)
        mock_vector_store.add_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scrape_and_store_invalid_url(self, agent, mock_scraper):
        """Test scrape and store with invalid URL."""
        url = "https://invalid.com"
        mock_scraper.validate_url.return_value = False
        
        with pytest.raises(ValueError, match="cannot handle URL"):
            await agent.scrape_and_store(url)
    
    @pytest.mark.asyncio
    async def test_scrape_and_store_no_documents(self, agent, mock_scraper, mock_vector_store):
        """Test scrape and store when no documents are found."""
        url = "https://example.com"
        mock_scraper.scrape.return_value = []
        
        result = await agent.scrape_and_store(url)
        
        assert result == 0
        mock_vector_store.add_documents.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_search(self, agent, mock_vector_store):
        """Test search functionality."""
        query = "test query"
        expected_results = [
            SearchResult(
                document=Document(content="Test", url="https://example.com"),
                score=0.95
            )
        ]
        mock_vector_store.search.return_value = expected_results
        
        results = await agent.search(query, limit=5)
        
        assert results == expected_results
        mock_vector_store.search.assert_called_once_with(query, 5, None)
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, agent, mock_vector_store):
        """Test search with filters."""
        query = "test query"
        filters = {"url": "example.com"}
        
        await agent.search(query, limit=10, filters=filters)
        
        mock_vector_store.search.assert_called_once_with(query, 10, filters)
    
    @pytest.mark.asyncio
    async def test_get_stats(self, agent, mock_scraper, mock_vector_store):
        """Test get stats functionality."""
        stats = await agent.get_stats()
        
        expected_stats = {
            "scraper": "MockScraper",
            "vector_store": "MockVectorStore",
            "document_count": 10,
            "vector_store_healthy": True
        }
        
        assert stats == expected_stats
        mock_vector_store.get_document_count.assert_called_once()
        mock_vector_store.health_check.assert_called_once()