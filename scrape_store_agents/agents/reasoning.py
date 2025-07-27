"""AI-powered reasoning agent for intelligent web scraping and storage."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from urllib.parse import urlparse

from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_community.llms import Anthropic

from .base import Agent, BaseScraper, BaseVectorStore, Document, SearchResult


logger = logging.getLogger(__name__)


class ReasoningAgent(Agent):
    """AI-powered agent that uses LLM reasoning for intelligent scraping decisions."""
    
    def __init__(
        self, 
        scraper: BaseScraper, 
        vector_store: BaseVectorStore,
        llm_config: Dict[str, Any]
    ):
        """Initialize reasoning agent with LLM capabilities.
        
        Args:
            scraper: Scraper instance for content extraction
            vector_store: Vector store for storage and retrieval
            llm_config: LLM configuration containing:
                - provider: 'openai' or 'anthropic'
                - model: model name (e.g., 'gpt-4', 'claude-3-sonnet')
                - api_key: API key for the provider
                - temperature: sampling temperature (0.0-1.0)
                - max_tokens: maximum tokens in response
        """
        super().__init__(scraper, vector_store)
        self.llm_config = llm_config
        self.llm = self._initialize_llm()
        self.memory: List[Dict[str, Any]] = []
        self.scraping_strategies: Dict[str, Any] = {}
        
    def _initialize_llm(self) -> BaseLLM:
        """Initialize the appropriate LLM based on configuration."""
        provider = self.llm_config.get('provider', 'openai').lower()
        model = self.llm_config.get('model', 'gpt-3.5-turbo')
        api_key = self.llm_config.get('api_key')
        temperature = self.llm_config.get('temperature', 0.1)
        max_tokens = self.llm_config.get('max_tokens', 1000)
        
        if provider == 'openai':
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider == 'anthropic':
            return Anthropic(
                model=model,
                anthropic_api_key=api_key,
                temperature=temperature,
                max_tokens_to_sample=max_tokens
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    async def analyze_url(self, url: str) -> Dict[str, Any]:
        """AI analyzes URL to determine optimal scraping strategy.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary containing analysis results and recommended strategy
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            path = parsed_url.path
            
            prompt = f"""
            Analyze this URL for web scraping strategy: {url}
            
            Domain: {domain}
            Path: {path}
            
            Consider:
            1. What type of website is this likely to be? (news, blog, e-commerce, documentation, etc.)
            2. What content extraction challenges might exist?
            3. What CSS selectors would likely work best?
            4. Are there any rate limiting or anti-bot measures to consider?
            5. What's the recommended scraping approach?
            
            Respond with a JSON object containing:
            {{
                "site_type": "type of website",
                "challenges": ["list", "of", "challenges"],
                "recommended_selectors": ["list", "of", "css", "selectors"],
                "rate_limit_strategy": "approach for rate limiting",
                "scraping_approach": "recommended strategy",
                "confidence": 0.85
            }}
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.invoke(prompt)
            )
            
            # Parse LLM response
            try:
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                # Extract JSON from response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                json_str = content[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                # Store analysis in memory
                self.memory.append({
                    'timestamp': datetime.utcnow(),
                    'action': 'url_analysis',
                    'url': url,
                    'analysis': analysis
                })
                
                return analysis
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response for URL analysis: {e}")
                return self._fallback_url_analysis(url)
                
        except Exception as e:
            logger.error(f"Error in URL analysis: {e}")
            return self._fallback_url_analysis(url)
    
    def _fallback_url_analysis(self, url: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Simple heuristics
        if any(news_indicator in domain for news_indicator in ['news', 'cnn', 'bbc', 'reuters']):
            site_type = 'news'
        elif any(blog_indicator in domain for blog_indicator in ['blog', 'medium', 'wordpress']):
            site_type = 'blog'
        elif any(docs_indicator in domain for docs_indicator in ['docs', 'documentation', 'wiki']):
            site_type = 'documentation'
        else:
            site_type = 'general'
        
        return {
            'site_type': site_type,
            'challenges': ['unknown'],
            'recommended_selectors': ['main', 'article', '.content'],
            'rate_limit_strategy': 'conservative',
            'scraping_approach': 'standard',
            'confidence': 0.3
        }
    
    async def adapt_scraping_strategy(
        self, 
        url: str, 
        failed_attempts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """AI adapts scraping strategy based on previous failures.
        
        Args:
            url: URL that failed to scrape
            failed_attempts: List of previous failed attempts with error details
            
        Returns:
            Dictionary containing adapted strategy
        """
        try:
            failures_summary = []
            for attempt in failed_attempts[-3:]:  # Last 3 attempts
                failures_summary.append({
                    'error': attempt.get('error', 'unknown'),
                    'strategy': attempt.get('strategy', 'unknown'),
                    'timestamp': attempt.get('timestamp', 'unknown')
                })
            
            prompt = f"""
            The scraping of {url} has failed multiple times. Help adapt the strategy.
            
            Previous failed attempts:
            {json.dumps(failures_summary, indent=2)}
            
            Based on these failures, suggest an adapted scraping strategy. Consider:
            1. Different CSS selectors to try
            2. Modified request headers or delays
            3. Alternative scraping approaches
            4. Potential anti-bot countermeasures
            
            Respond with a JSON object:
            {{
                "new_selectors": ["list", "of", "alternative", "selectors"],
                "request_modifications": {{"headers": {{}}, "delay": 2.0}},
                "alternative_approach": "description of new approach",
                "success_probability": 0.75,
                "rationale": "explanation of why this should work"
            }}
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.invoke(prompt)
            )
            
            # Parse response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
                
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            adaptation = json.loads(json_str)
            
            # Store adaptation in memory
            self.memory.append({
                'timestamp': datetime.utcnow(),
                'action': 'strategy_adaptation',
                'url': url,
                'adaptation': adaptation,
                'failed_attempts_count': len(failed_attempts)
            })
            
            return adaptation
            
        except Exception as e:
            logger.error(f"Error in strategy adaptation: {e}")
            return {
                'new_selectors': ['body', '*'],
                'request_modifications': {'delay': 5.0},
                'alternative_approach': 'fallback to basic extraction',
                'success_probability': 0.2,
                'rationale': 'Fallback strategy due to adaptation error'
            }
    
    async def decide_storage_strategy(self, documents: List[Document]) -> Dict[str, Any]:
        """AI decides optimal storage and chunking strategy for documents.
        
        Args:
            documents: List of documents to be stored
            
        Returns:
            Dictionary containing storage strategy
        """
        if not documents:
            return {'strategy': 'none', 'rationale': 'No documents to store'}
        
        try:
            # Analyze documents
            doc_stats = {
                'count': len(documents),
                'avg_length': sum(len(doc.content) for doc in documents) / len(documents),
                'content_types': list(set(doc.metadata.get('content_type', 'text') for doc in documents)),
                'domains': list(set(urlparse(doc.url).netloc for doc in documents))
            }
            
            sample_content = documents[0].content[:500] if documents else ""
            
            prompt = f"""
            Analyze these documents and recommend a storage strategy:
            
            Document Statistics:
            - Count: {doc_stats['count']}
            - Average length: {doc_stats['avg_length']:.0f} characters
            - Content types: {doc_stats['content_types']}
            - Domains: {doc_stats['domains']}
            
            Sample content:
            "{sample_content}..."
            
            Recommend:
            1. Optimal chunk size for vector storage
            2. Metadata extraction strategy
            3. Embedding approach
            4. Index organization
            
            Respond with JSON:
            {{
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "metadata_strategy": "extract_key_fields",
                "embedding_approach": "semantic_chunks",
                "index_organization": "by_domain",
                "rationale": "explanation"
            }}
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.invoke(prompt)
            )
            
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
                
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            strategy = json.loads(json_str)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error in storage strategy decision: {e}")
            return {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'metadata_strategy': 'basic',
                'embedding_approach': 'standard',
                'index_organization': 'flat',
                'rationale': 'Fallback strategy due to decision error'
            }
    
    async def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's reasoning activities."""
        base_stats = await super().get_stats()
        
        reasoning_stats = {
            'memory_entries': len(self.memory),
            'url_analyses': len([m for m in self.memory if m['action'] == 'url_analysis']),
            'strategy_adaptations': len([m for m in self.memory if m['action'] == 'strategy_adaptation']),
            'llm_provider': self.llm_config.get('provider'),
            'llm_model': self.llm_config.get('model')
        }
        
        return {**base_stats, **reasoning_stats}
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory.clear()
        logger.info("Agent memory cleared")