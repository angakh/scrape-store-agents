"""Self-improving agent that learns from extraction quality and adapts strategies over time."""

import asyncio
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict

from langchain.llms.base import BaseLLM

from .base import Document
from .reasoning import ReasoningAgent


logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Represents the result of a content extraction."""
    
    url: str
    strategy: Dict[str, Any]
    documents: List[Document]
    quality_score: float
    execution_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    

@dataclass  
class LearningEntry:
    """Represents a learning entry from extraction results."""
    
    site_pattern: str  # Domain pattern or site type
    strategy: Dict[str, Any]
    avg_quality: float
    success_rate: float
    sample_count: int
    last_updated: datetime
    confidence: float


@dataclass
class QualityMetrics:
    """Quality assessment metrics for extracted content."""
    
    content_length_score: float  # 0-1, appropriate content length
    structure_score: float       # 0-1, proper HTML structure extraction
    coherence_score: float       # 0-1, text coherence and readability
    completeness_score: float    # 0-1, content completeness
    uniqueness_score: float      # 0-1, content uniqueness (not duplicate)
    overall_score: float         # 0-1, weighted average


class SelfImprovingAgent(ReasoningAgent):
    """Agent that learns from extraction quality and continuously improves strategies."""
    
    def __init__(self, scraper, vector_store, llm_config, learning_config=None):
        """Initialize self-improving agent.
        
        Args:
            scraper: Base scraper instance
            vector_store: Vector store instance  
            llm_config: LLM configuration
            learning_config: Configuration for learning behavior
        """
        super().__init__(scraper, vector_store, llm_config)
        
        # Learning configuration
        self.learning_config = learning_config or {
            'min_samples_for_confidence': 3,
            'quality_threshold': 0.7,
            'learning_rate': 0.1,
            'strategy_retention_days': 30,
            'max_learning_entries': 1000
        }
        
        # Learning storage
        self.extraction_history: List[ExtractionResult] = []
        self.learned_strategies: Dict[str, LearningEntry] = {}
        self.quality_patterns: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance tracking
        self.improvement_metrics = {
            'quality_improvements': 0,
            'strategy_adaptations': 0,
            'successful_predictions': 0,
            'total_extractions': 0
        }
    
    def _get_site_pattern(self, url: str) -> str:
        """Extract a generalized pattern from URL for learning."""
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Create patterns for learning
        if 'news' in domain or 'cnn' in domain or 'bbc' in domain:
            return 'news_site'
        elif 'blog' in domain or 'medium' in domain or 'wordpress' in domain:
            return 'blog_site'
        elif 'docs' in domain or 'documentation' in domain:
            return 'docs_site'
        elif 'github' in domain:
            return 'github_site'
        elif 'wiki' in domain:
            return 'wiki_site'
        else:
            # Use domain as pattern for specific learning
            return domain
    
    async def validate_extraction_quality(self, documents: List[Document], url: str) -> QualityMetrics:
        """AI validates the quality of extracted content.
        
        Args:
            documents: Extracted documents to validate
            url: Source URL for context
            
        Returns:
            QualityMetrics with detailed quality assessment
        """
        if not documents:
            return QualityMetrics(0, 0, 0, 0, 0, 0)
        
        try:
            # Prepare content for analysis
            total_content = '\n'.join([doc.content for doc in documents])
            content_sample = total_content[:2000]  # Sample for AI analysis
            
            prompt = f"""
            Analyze the quality of this extracted content from {url}:
            
            Content sample:
            "{content_sample}"
            
            Total documents: {len(documents)}
            Total content length: {len(total_content)} characters
            
            Evaluate these quality aspects (0.0-1.0 scale):
            1. Content Length: Is the content substantial and complete?
            2. Structure: Is the content well-structured and properly extracted?
            3. Coherence: Is the text coherent and readable?
            4. Completeness: Does the content seem complete (not cut off)?
            5. Uniqueness: Is the content unique and not duplicate/boilerplate?
            
            Consider:
            - Very short content (< 100 chars) = low quality
            - Repetitive or boilerplate text = low quality
            - Malformed or garbled text = low quality
            - Rich, substantive content = high quality
            
            Respond with JSON:
            {{
                "content_length_score": 0.85,
                "structure_score": 0.90,
                "coherence_score": 0.80,
                "completeness_score": 0.75,
                "uniqueness_score": 0.95,
                "rationale": "explanation of scoring"
            }}
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.invoke(prompt)
            )
            
            # Parse AI response
            content = response.content if hasattr(response, 'content') else str(response)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            quality_data = json.loads(json_str)
            
            # Calculate weighted overall score
            weights = {
                'content_length_score': 0.25,
                'structure_score': 0.25,
                'coherence_score': 0.20,
                'completeness_score': 0.15,
                'uniqueness_score': 0.15
            }
            
            overall_score = sum(
                quality_data.get(metric, 0) * weight 
                for metric, weight in weights.items()
            )
            
            return QualityMetrics(
                content_length_score=quality_data.get('content_length_score', 0),
                structure_score=quality_data.get('structure_score', 0),
                coherence_score=quality_data.get('coherence_score', 0),
                completeness_score=quality_data.get('completeness_score', 0),
                uniqueness_score=quality_data.get('uniqueness_score', 0),
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Error in quality validation: {e}")
            # Fallback to basic heuristics
            return self._fallback_quality_assessment(documents)
    
    def _fallback_quality_assessment(self, documents: List[Document]) -> QualityMetrics:
        """Fallback quality assessment using basic heuristics."""
        if not documents:
            return QualityMetrics(0, 0, 0, 0, 0, 0)
        
        total_content = '\n'.join([doc.content for doc in documents])
        
        # Basic scoring heuristics
        content_length_score = min(len(total_content) / 1000, 1.0)  # Normalize to 1000 chars
        structure_score = 0.7 if any(doc.title for doc in documents) else 0.4
        coherence_score = 0.6  # Default moderate score
        completeness_score = 0.8 if len(total_content) > 200 else 0.3
        uniqueness_score = 0.7  # Default assumption
        
        overall_score = (content_length_score + structure_score + coherence_score + 
                        completeness_score + uniqueness_score) / 5
        
        return QualityMetrics(
            content_length_score, structure_score, coherence_score,
            completeness_score, uniqueness_score, overall_score
        )
    
    async def learn_from_extraction(self, url: str, strategy: Dict[str, Any], 
                                  documents: List[Document], execution_time: float,
                                  success: bool, error_message: Optional[str] = None):
        """Learn from an extraction result to improve future performance.
        
        Args:
            url: URL that was scraped
            strategy: Strategy used for scraping
            documents: Documents extracted (empty if failed)
            execution_time: Time taken for extraction
            success: Whether extraction succeeded
            error_message: Error message if failed
        """
        try:
            # Validate quality if extraction succeeded
            if success and documents:
                quality_metrics = await self.validate_extraction_quality(documents, url)
                quality_score = quality_metrics.overall_score
            else:
                quality_score = 0.0
            
            # Create extraction result
            result = ExtractionResult(
                url=url,
                strategy=strategy,
                documents=documents,
                quality_score=quality_score,
                execution_time=execution_time,
                timestamp=datetime.now(),
                success=success,
                error_message=error_message
            )
            
            # Store in history
            self.extraction_history.append(result)
            
            # Update learned strategies
            site_pattern = self._get_site_pattern(url)
            await self._update_learned_strategies(site_pattern, result)
            
            # Update performance metrics
            self.improvement_metrics['total_extractions'] += 1
            if success and quality_score > self.learning_config['quality_threshold']:
                self.improvement_metrics['successful_predictions'] += 1
            
            # Store learning in memory for reasoning agent
            self.memory.append({
                'timestamp': datetime.now(),
                'action': 'extraction_learning',
                'url': url,
                'site_pattern': site_pattern,
                'quality_score': quality_score,
                'success': success,
                'strategy': strategy
            })
            
            logger.info(f"Learned from extraction: {url} (quality: {quality_score:.2f})")
            
        except Exception as e:
            logger.error(f"Error in learning from extraction: {e}")
    
    async def _update_learned_strategies(self, site_pattern: str, result: ExtractionResult):
        """Update learned strategies based on extraction result."""
        strategy_key = self._get_strategy_key(result.strategy)
        learning_key = f"{site_pattern}:{strategy_key}"
        
        if learning_key in self.learned_strategies:
            # Update existing learning entry
            entry = self.learned_strategies[learning_key]
            
            # Update averages using exponential moving average
            alpha = self.learning_config['learning_rate']
            entry.avg_quality = (1 - alpha) * entry.avg_quality + alpha * result.quality_score
            entry.success_rate = (1 - alpha) * entry.success_rate + alpha * (1.0 if result.success else 0.0)
            entry.sample_count += 1
            entry.last_updated = datetime.now()
            
            # Update confidence based on sample count
            entry.confidence = min(entry.sample_count / self.learning_config['min_samples_for_confidence'], 1.0)
            
        else:
            # Create new learning entry
            self.learned_strategies[learning_key] = LearningEntry(
                site_pattern=site_pattern,
                strategy=result.strategy,
                avg_quality=result.quality_score,
                success_rate=1.0 if result.success else 0.0,
                sample_count=1,
                last_updated=datetime.now(),
                confidence=0.1  # Low initial confidence
            )
        
        # Clean up old entries
        await self._cleanup_old_strategies()
    
    def _get_strategy_key(self, strategy: Dict[str, Any]) -> str:
        """Generate a key for strategy identification."""
        # Create hash of key strategy parameters
        key_params = {
            'timeout': strategy.get('timeout', 30),
            'max_depth': strategy.get('max_depth', 1),
            'extract_links': strategy.get('extract_links', False),
            'user_agent': strategy.get('user_agent', 'default')[:20]  # Truncate
        }
        
        strategy_str = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(strategy_str.encode()).hexdigest()[:8]
    
    async def _cleanup_old_strategies(self):
        """Remove old learning entries to prevent memory bloat."""
        cutoff_date = datetime.now() - timedelta(days=self.learning_config['strategy_retention_days'])
        
        # Remove old entries
        keys_to_remove = [
            key for key, entry in self.learned_strategies.items()
            if entry.last_updated < cutoff_date
        ]
        
        for key in keys_to_remove:
            del self.learned_strategies[key]
        
        # Limit total entries
        if len(self.learned_strategies) > self.learning_config['max_learning_entries']:
            # Keep the most recent and highest quality entries
            sorted_entries = sorted(
                self.learned_strategies.items(),
                key=lambda x: (x[1].last_updated, x[1].avg_quality),
                reverse=True
            )
            
            # Keep top entries
            self.learned_strategies = dict(sorted_entries[:self.learning_config['max_learning_entries']])
    
    async def get_optimized_strategy(self, url: str, base_strategy: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], float]:
        """Get an optimized strategy based on learning history.
        
        Args:
            url: URL to get strategy for
            base_strategy: Base strategy to optimize
            
        Returns:
            Tuple of (optimized_strategy, confidence_score)
        """
        site_pattern = self._get_site_pattern(url)
        
        # Find relevant learned strategies
        relevant_entries = [
            entry for key, entry in self.learned_strategies.items()
            if key.startswith(site_pattern) and entry.confidence > 0.3
        ]
        
        if not relevant_entries:
            # No learning available, return base strategy
            return base_strategy or self.scraper.config, 0.0
        
        # Find best performing strategy
        best_entry = max(relevant_entries, key=lambda x: x.avg_quality * x.confidence)
        
        if best_entry.avg_quality > self.learning_config['quality_threshold']:
            logger.info(f"Using optimized strategy for {site_pattern} (quality: {best_entry.avg_quality:.2f})")
            self.improvement_metrics['strategy_adaptations'] += 1
            return best_entry.strategy, best_entry.confidence
        
        # Return base strategy if no good learned strategy
        return base_strategy or self.scraper.config, 0.0
    
    async def scrape_and_store_with_learning(self, url: str, **kwargs) -> int:
        """Enhanced scrape and store that includes learning.
        
        Args:
            url: URL to scrape
            **kwargs: Additional scraping parameters
            
        Returns:
            Number of documents stored
        """
        start_time = datetime.now()
        success = False
        documents = []
        error_message = None
        
        try:
            # Get optimized strategy
            base_strategy = {**self.scraper.config, **kwargs}
            optimized_strategy, confidence = await self.get_optimized_strategy(url, base_strategy)
            
            # Apply optimized strategy
            if confidence > 0.5:
                # Use learned strategy
                from .scrapers.web_scraper import WebScraper
                optimized_scraper = WebScraper(optimized_strategy)
                temp_agent = type(self)(optimized_scraper, self.vector_store, self.llm_config)
                documents_count = await temp_agent.scrape_and_store(url)
                documents = await optimized_scraper.scrape(url)
            else:
                # Use standard approach
                documents_count = await self.scrape_and_store(url)
                documents = await self.scraper.scrape(url)
            
            success = True
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Learn from this extraction
            await self.learn_from_extraction(
                url, optimized_strategy, documents, execution_time, success
            )
            
            return documents_count
            
        except Exception as e:
            error_message = str(e)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Learn from failure
            await self.learn_from_extraction(
                url, base_strategy, [], execution_time, False, error_message
            )
            
            raise e
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        base_stats = await self.get_reasoning_stats()
        
        # Calculate learning metrics
        quality_scores = [r.quality_score for r in self.extraction_history if r.success]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        recent_extractions = [
            r for r in self.extraction_history 
            if r.timestamp > datetime.now() - timedelta(days=7)
        ]
        
        learning_stats = {
            'total_extractions': len(self.extraction_history),
            'learned_strategies': len(self.learned_strategies),
            'average_quality': avg_quality,
            'recent_extractions': len(recent_extractions),
            'improvement_metrics': self.improvement_metrics.copy(),
            'site_patterns_learned': len(set(entry.site_pattern for entry in self.learned_strategies.values())),
            'high_confidence_strategies': len([
                entry for entry in self.learned_strategies.values() 
                if entry.confidence > 0.7
            ])
        }
        
        return {**base_stats, **learning_stats}
    
    def clear_learning_data(self) -> Dict[str, int]:
        """Clear learning data and reset improvement metrics."""
        cleared_stats = {
            'extraction_history': len(self.extraction_history),
            'learned_strategies': len(self.learned_strategies),
            'quality_patterns': len(self.quality_patterns)
        }
        
        self.extraction_history.clear()
        self.learned_strategies.clear()
        self.quality_patterns.clear()
        
        # Reset metrics but keep total count
        total_extractions = self.improvement_metrics['total_extractions']
        self.improvement_metrics = {
            'quality_improvements': 0,
            'strategy_adaptations': 0,
            'successful_predictions': 0,
            'total_extractions': total_extractions
        }
        
        # Also clear base memory
        self.clear_memory()
        
        logger.info("Learning data cleared")
        return cleared_stats