"""Abstract base classes for scrapers and vector stores."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    """Represents a scraped document."""
    
    content: str
    url: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """Represents a search result from vector store."""
    
    document: Document
    score: float
    distance: Optional[float] = None


class BaseScraper(ABC):
    """Abstract base class for all scrapers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize scraper with configuration.
        
        Args:
            config: Configuration dictionary containing scraper settings
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def scrape(self, url: str, **kwargs) -> List[Document]:
        """Scrape content from a URL.
        
        Args:
            url: The URL to scrape
            **kwargs: Additional scraping parameters
            
        Returns:
            List of Document objects containing scraped content
        """
        pass
    
    @abstractmethod
    def validate_url(self, url: str) -> bool:
        """Validate if this scraper can handle the given URL.
        
        Args:
            url: The URL to validate
            
        Returns:
            True if scraper can handle this URL, False otherwise
        """
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize vector store with configuration.
        
        Args:
            config: Configuration dictionary containing vector store settings
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        limit: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional filters to apply to search
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
        """
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """Get total number of documents in the store.
        
        Returns:
            Number of documents
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if vector store is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)


class Agent:
    """Main agent orchestrating scraping and storage operations."""
    
    def __init__(self, scraper: BaseScraper, vector_store: BaseVectorStore):
        """Initialize agent with scraper and vector store.
        
        Args:
            scraper: Scraper instance to use for content extraction
            vector_store: Vector store instance for storage and retrieval
        """
        self.scraper = scraper
        self.vector_store = vector_store
    
    async def scrape_and_store(self, url: str, **kwargs) -> int:
        """Scrape content from URL and store in vector database.
        
        Args:
            url: URL to scrape
            **kwargs: Additional parameters for scraping
            
        Returns:
            Number of documents stored
        """
        if not self.scraper.validate_url(url):
            raise ValueError(f"Scraper {self.scraper.name} cannot handle URL: {url}")
        
        documents = await self.scraper.scrape(url, **kwargs)
        if documents:
            await self.vector_store.add_documents(documents)
        
        return len(documents)
    
    async def search(
        self, 
        query: str, 
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for documents similar to query.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional search filters
            
        Returns:
            List of SearchResult objects
        """
        return await self.vector_store.search(query, limit, filters)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dictionary containing agent statistics
        """
        return {
            "scraper": self.scraper.name,
            "vector_store": self.vector_store.name,
            "document_count": await self.vector_store.get_document_count(),
            "vector_store_healthy": await self.vector_store.health_check()
        }