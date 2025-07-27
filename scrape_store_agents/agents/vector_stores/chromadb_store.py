"""ChromaDB vector store implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ..base import BaseVectorStore, Document, SearchResult


logger = logging.getLogger(__name__)


class ChromaDBStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ChromaDB store.
        
        Args:
            config: Configuration containing:
                - collection_name: Name of the ChromaDB collection
                - persist_directory: Directory to persist data (optional)
                - embedding_model: Sentence transformer model name
                - host: ChromaDB server host (for client mode)
                - port: ChromaDB server port (for client mode)
                - distance_metric: Distance metric for similarity search
        """
        super().__init__(config)
        
        self.collection_name = self.get_config('collection_name', 'documents')
        self.embedding_model_name = self.get_config(
            'embedding_model', 
            'all-MiniLM-L6-v2'
        )
        self.distance_metric = self.get_config('distance_metric', 'cosine')
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB client
        self._init_client()
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    def _init_client(self):
        """Initialize ChromaDB client."""
        host = self.get_config('host')
        port = self.get_config('port')
        persist_directory = self.get_config('persist_directory')
        
        if host and port:
            # Client mode - connect to remote ChromaDB server
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Embedded mode - use local persistence
            settings = Settings()
            if persist_directory:
                settings = Settings(persist_directory=persist_directory)
            self.client = chromadb.Client(settings)
    
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
            return collection
        except Exception as e:
            # Collection doesn't exist, create it
            logger.info(f"Collection '{self.collection_name}' not found, creating new one: {e}")
            metadata = {"hnsw:space": self.distance_metric}
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata=metadata
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return
        
        try:
            # Prepare data for ChromaDB
            texts = []
            metadatas = []
            ids = []
            
            for doc in documents:
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                
                # Prepare text content
                content = doc.content
                if doc.title:
                    content = f"{doc.title}\n\n{content}"
                
                texts.append(content)
                
                # Prepare metadata
                metadata = {
                    'url': doc.url,
                    'title': doc.title or '',
                    'timestamp': doc.timestamp.isoformat() if doc.timestamp else '',
                    'content_length': len(doc.content),
                }
                
                # Add custom metadata
                if doc.metadata:
                    for key, value in doc.metadata.items():
                        # ChromaDB requires string values for metadata
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f'custom_{key}'] = str(value)
                
                metadatas.append(metadata)
                ids.append(doc_id)
            
            # Generate embeddings
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                self.embedding_model.encode, 
                texts
            )
            
            # Add to ChromaDB collection
            def add_to_collection():
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas,
                    documents=texts,
                    ids=ids
                )
            
            await loop.run_in_executor(None, add_to_collection)
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    async def search(
        self, 
        query: str, 
        limit: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None, 
                self.embedding_model.encode, 
                [query]
            )
            
            # Prepare ChromaDB query parameters
            query_params = {
                'query_embeddings': query_embedding.tolist(),
                'n_results': limit,
                'include': ['documents', 'metadatas', 'distances']
            }
            
            # Add filters if provided
            if filters:
                chroma_filters = self._convert_filters(filters)
                if chroma_filters:
                    query_params['where'] = chroma_filters
            
            # Execute search
            from functools import partial
            func = partial(self.collection.query, **query_params)
            results = await loop.run_in_executor(None, func)
            
            # Convert results to SearchResult objects
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for doc_text, metadata, distance in zip(documents, metadatas, distances):
                    # Extract title and content
                    lines = doc_text.split('\n\n', 1)
                    if len(lines) == 2 and metadata.get('title'):
                        title = lines[0]
                        content = lines[1]
                    else:
                        title = metadata.get('title')
                        content = doc_text
                    
                    # Reconstruct document metadata
                    doc_metadata = {}
                    for key, value in metadata.items():
                        if key.startswith('custom_'):
                            original_key = key[7:]  # Remove 'custom_' prefix
                            doc_metadata[original_key] = value
                    
                    # Create Document object
                    document = Document(
                        content=content,
                        url=metadata['url'],
                        title=title,
                        metadata=doc_metadata,
                        timestamp=datetime.fromisoformat(metadata['timestamp']) if metadata.get('timestamp') else None
                    )
                    
                    # Calculate similarity score (1 - distance for cosine)
                    score = 1.0 - distance if self.distance_metric == 'cosine' else distance
                    
                    search_results.append(SearchResult(
                        document=document,
                        score=score,
                        distance=distance
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise
    
    def _convert_filters(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert filters to ChromaDB format."""
        chroma_filters = {}
        
        for key, value in filters.items():
            if key == 'url':
                chroma_filters['url'] = value
            elif key == 'title':
                chroma_filters['title'] = value
            elif key == 'content_length_min':
                chroma_filters['content_length'] = {'$gte': value}
            elif key == 'content_length_max':
                chroma_filters['content_length'] = {'$lte': value}
            else:
                # Handle custom metadata filters
                chroma_filters[f'custom_{key}'] = str(value)
        
        return chroma_filters if chroma_filters else None
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.collection.delete,
                document_ids
            )
            logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
            
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            raise
    
    async def get_document_count(self) -> int:
        """Get total number of documents in the store."""
        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(
                None,
                self.collection.count
            )
            return count
            
        except Exception as e:
            logger.error(f"Error getting document count from ChromaDB: {e}")
            return 0
    
    async def health_check(self) -> bool:
        """Check if vector store is healthy and accessible."""
        try:
            # Try to get collection info
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.collection.count
            )
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False
    
    async def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            loop = asyncio.get_event_loop()
            
            # Get all document IDs
            results = await loop.run_in_executor(
                None,
                self.collection.get,
                None,  # ids
                None,  # where
                None,  # limit
                None,  # offset
                ['documents']  # include
            )
            
            if results['ids']:
                await loop.run_in_executor(
                    None,
                    self.collection.delete,
                    results['ids']
                )
                logger.info(f"Cleared {len(results['ids'])} documents from collection")
            
        except Exception as e:
            logger.error(f"Error clearing ChromaDB collection: {e}")
            raise