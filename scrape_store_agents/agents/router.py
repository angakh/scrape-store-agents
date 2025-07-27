"""Intelligent scraper router that uses AI to select optimal scrapers and generate custom selectors."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse
from dataclasses import dataclass

from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_community.llms import Anthropic

from .base import BaseScraper, Document
from .scrapers.web_scraper import WebScraper


logger = logging.getLogger(__name__)


@dataclass
class ScraperRecommendation:
    """Represents a scraper recommendation from the AI."""
    
    scraper_type: str
    confidence: float
    config: Dict[str, Any]
    rationale: str
    custom_selectors: Optional[Dict[str, List[str]]] = None


@dataclass
class SiteAnalysis:
    """Represents AI analysis of a website."""
    
    site_type: str
    complexity: str
    anti_bot_measures: List[str]
    content_structure: Dict[str, Any]
    recommended_approach: str
    confidence: float


class IntelligentScraperRouter:
    """AI-powered router that selects optimal scrapers and generates custom configurations."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """Initialize the intelligent scraper router.
        
        Args:
            llm_config: LLM configuration containing:
                - provider: 'openai' or 'anthropic'
                - model: model name
                - api_key: API key for the provider
                - temperature: sampling temperature
                - max_tokens: maximum tokens in response
        """
        self.llm_config = llm_config
        self.llm = self._initialize_llm()
        
        # Registry of available scrapers
        self.scraper_registry: Dict[str, Type[BaseScraper]] = {
            'web_scraper': WebScraper,
            # Future scrapers can be registered here
            # 'news_scraper': NewsScraper,
            # 'ecommerce_scraper': EcommerceScraper,
            # 'api_scraper': APIScraper,
        }
        
        # Cache for site analyses to avoid redundant LLM calls
        self.analysis_cache: Dict[str, SiteAnalysis] = {}
        
        # Performance tracking
        self.selection_history: List[Dict[str, Any]] = []
    
    def _initialize_llm(self) -> BaseLLM:
        """Initialize the appropriate LLM based on configuration."""
        provider = self.llm_config.get('provider', 'openai').lower()
        model = self.llm_config.get('model', 'gpt-3.5-turbo')
        api_key = self.llm_config.get('api_key')
        temperature = self.llm_config.get('temperature', 0.1)
        max_tokens = self.llm_config.get('max_tokens', 1500)
        
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
    
    async def analyze_site(self, url: str, sample_html: Optional[str] = None) -> SiteAnalysis:
        """Analyze a website to understand its structure and characteristics.
        
        Args:
            url: URL to analyze
            sample_html: Optional HTML sample for deeper analysis
            
        Returns:
            SiteAnalysis object containing AI insights about the site
        """
        # Check cache first
        domain = urlparse(url).netloc
        if domain in self.analysis_cache:
            logger.info(f"Using cached analysis for {domain}")
            return self.analysis_cache[domain]
        
        try:
            parsed_url = urlparse(url)
            
            # Prepare HTML sample string for prompt
            if sample_html:
                html_sample_str = f"HTML Sample (first 2000 chars):\n{sample_html[:2000]}\n"
            else:
                html_sample_str = ""
            
            # Build analysis prompt
            prompt = f"""
            Analyze this website for optimal web scraping strategy:
            
            URL: {url}
            Domain: {parsed_url.netloc}
            Path: {parsed_url.path}
            
            {html_sample_str}
            
            Please analyze:
            1. Site type (news, blog, e-commerce, documentation, social media, etc.)
            2. Content complexity (simple, moderate, complex, dynamic)
            3. Likely anti-bot measures (rate limiting, captcha, js-required, etc.)
            4. Content structure patterns
            5. Recommended scraping approach
            
            Consider factors like:
            - URL patterns and structure
            - Domain name indicators
            - HTML structure (if provided)
            - Common site patterns
            
            Respond with JSON:
            {{
                "site_type": "news|blog|ecommerce|documentation|social|api|other",
                "complexity": "simple|moderate|complex|dynamic",
                "anti_bot_measures": ["rate_limiting", "captcha", "js_required"],
                "content_structure": {{
                    "main_content_likely": ["article", ".content", "main"],
                    "title_likely": ["h1", ".title", ".headline"],
                    "metadata_indicators": ["time", ".author", ".date"]
                }},
                "recommended_approach": "standard|careful|aggressive|api_preferred",
                "confidence": 0.85,
                "rationale": "Explanation of analysis"
            }}
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.invoke(prompt)
            )
            
            # Parse LLM response
            content = response.content if hasattr(response, 'content') else str(response)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            analysis_data = json.loads(json_str)
            
            analysis = SiteAnalysis(
                site_type=analysis_data.get('site_type', 'other'),
                complexity=analysis_data.get('complexity', 'moderate'),
                anti_bot_measures=analysis_data.get('anti_bot_measures', []),
                content_structure=analysis_data.get('content_structure', {}),
                recommended_approach=analysis_data.get('recommended_approach', 'standard'),
                confidence=analysis_data.get('confidence', 0.5)
            )
            
            # Cache the analysis
            self.analysis_cache[domain] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in site analysis: {e}")
            # Return fallback analysis
            return SiteAnalysis(
                site_type='other',
                complexity='moderate', 
                anti_bot_measures=['unknown'],
                content_structure={},
                recommended_approach='standard',
                confidence=0.2
            )
    
    async def select_scraper(self, url: str, requirements: Optional[Dict[str, Any]] = None) -> ScraperRecommendation:
        """Select the optimal scraper for a given URL.
        
        Args:
            url: URL to scrape
            requirements: Optional specific requirements (e.g., content_types, quality_level)
            
        Returns:
            ScraperRecommendation with scraper type and configuration
        """
        try:
            # First analyze the site
            site_analysis = await self.analyze_site(url)
            
            # Build selection prompt
            available_scrapers = list(self.scraper_registry.keys())
            
            prompt = f"""
            Based on the site analysis, select the optimal scraper and configuration:
            
            URL: {url}
            Site Analysis:
            - Type: {site_analysis.site_type}
            - Complexity: {site_analysis.complexity}
            - Anti-bot measures: {site_analysis.anti_bot_measures}
            - Recommended approach: {site_analysis.recommended_approach}
            
            Available scrapers: {available_scrapers}
            
            Requirements: {json.dumps(requirements or {}, indent=2)}
            
            Consider:
            1. Which scraper type is best suited for this site?
            2. What configuration parameters should be used?
            3. What's the confidence level of this recommendation?
            4. What's the rationale for this choice?
            
            Scraper descriptions:
            - web_scraper: General-purpose HTML scraper with configurable selectors
            
            Respond with JSON:
            {{
                "scraper_type": "web_scraper",
                "confidence": 0.85,
                "config": {{
                    "timeout": 30,
                    "max_depth": 2,
                    "extract_links": true,
                    "user_agent": "custom_agent",
                    "delay_between_requests": 2.0,
                    "max_retries": 3
                }},
                "rationale": "Explanation of why this scraper and config are optimal"
            }}
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.invoke(prompt)
            )
            
            # Parse response
            content = response.content if hasattr(response, 'content') else str(response)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            recommendation_data = json.loads(json_str)
            
            recommendation = ScraperRecommendation(
                scraper_type=recommendation_data.get('scraper_type', 'web_scraper'),
                confidence=recommendation_data.get('confidence', 0.5),
                config=recommendation_data.get('config', {}),
                rationale=recommendation_data.get('rationale', 'Default selection')
            )
            
            # Record this selection for learning
            self.selection_history.append({
                'url': url,
                'site_analysis': site_analysis,
                'recommendation': recommendation,
                'timestamp': asyncio.get_event_loop().time()
            })
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error in scraper selection: {e}")
            # Return fallback recommendation
            return ScraperRecommendation(
                scraper_type='web_scraper',
                confidence=0.2,
                config={'timeout': 30},
                rationale=f'Fallback due to selection error: {e}'
            )
    
    async def generate_custom_selectors(self, url: str, content_goals: List[str], html_sample: str) -> Dict[str, List[str]]:
        """Generate custom CSS selectors for specific content extraction goals.
        
        Args:
            url: URL being scraped
            content_goals: List of content types to extract (e.g., ['title', 'content', 'author', 'date'])
            html_sample: Sample HTML from the page
            
        Returns:
            Dictionary mapping content goals to CSS selector lists
        """
        try:
            prompt = f"""
            Analyze this HTML and generate optimal CSS selectors for content extraction:
            
            URL: {url}
            Content Goals: {content_goals}
            
            HTML Sample (first 3000 chars):
            ```html
            {html_sample[:3000]}
            ```
            
            For each content goal, analyze the HTML structure and provide the best CSS selectors.
            Consider:
            1. Specificity vs. reliability
            2. Common patterns in this type of content
            3. Fallback selectors if primary ones fail
            4. Avoiding overly specific selectors that might break
            
            Respond with JSON mapping each goal to a prioritized list of selectors:
            {{
                "title": ["h1.main-title", "h1", ".title", ".headline"],
                "content": ["article .content", ".post-content", "main", ".article-body"],
                "author": [".author", ".byline", "[rel='author']"],
                "date": ["time", ".date", ".published", "[datetime]"],
                "description": [".description", ".excerpt", ".summary"]
            }}
            
            Only include goals that were requested: {content_goals}
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.invoke(prompt)
            )
            
            # Parse response
            content = response.content if hasattr(response, 'content') else str(response)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            selectors = json.loads(json_str)
            
            return selectors
            
        except Exception as e:
            logger.error(f"Error generating custom selectors: {e}")
            # Return fallback selectors
            fallback_selectors = {
                'title': ['h1', 'title', '.title'],
                'content': ['main', 'article', '.content'],
                'author': ['.author', '.byline'],
                'date': ['time', '.date'],
                'description': ['.description', '.excerpt']
            }
            return {goal: fallback_selectors.get(goal, ['*']) for goal in content_goals}
    
    async def create_scraper(self, recommendation: ScraperRecommendation) -> BaseScraper:
        """Create and configure a scraper instance based on recommendation.
        
        Args:
            recommendation: ScraperRecommendation from select_scraper()
            
        Returns:
            Configured scraper instance
        """
        scraper_class = self.scraper_registry.get(recommendation.scraper_type)
        if not scraper_class:
            logger.warning(f"Unknown scraper type: {recommendation.scraper_type}, falling back to WebScraper")
            scraper_class = WebScraper
        
        # Merge custom selectors into config if provided
        config = recommendation.config.copy()
        if recommendation.custom_selectors:
            config.update(recommendation.custom_selectors)
        
        return scraper_class(config)
    
    async def route_and_scrape(self, url: str, requirements: Optional[Dict[str, Any]] = None) -> Tuple[BaseScraper, List[Document]]:
        """Complete routing and scraping pipeline.
        
        Args:
            url: URL to scrape
            requirements: Optional scraping requirements
            
        Returns:
            Tuple of (scraper_used, documents_extracted)
        """
        try:
            # Select optimal scraper
            recommendation = await self.select_scraper(url, requirements)
            logger.info(f"Selected {recommendation.scraper_type} for {url} (confidence: {recommendation.confidence:.2f})")
            
            # Create scraper instance
            scraper = await self.create_scraper(recommendation)
            
            # Perform scraping
            documents = await scraper.scrape(url)
            
            # Log success
            logger.info(f"Successfully scraped {len(documents)} documents from {url}")
            
            return scraper, documents
            
        except Exception as e:
            logger.error(f"Error in route_and_scrape for {url}: {e}")
            # Fallback to basic web scraper
            fallback_scraper = WebScraper({'timeout': 30})
            try:
                documents = await fallback_scraper.scrape(url)
                return fallback_scraper, documents
            except Exception as fallback_error:
                logger.error(f"Fallback scraper also failed: {fallback_error}")
                return fallback_scraper, []
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get statistics about router performance and decisions."""
        if not self.selection_history:
            return {
                'total_selections': 0,
                'cached_analyses': len(self.analysis_cache),
                'scraper_types_used': {},
                'average_confidence': 0.0
            }
        
        scraper_counts = {}
        confidence_scores = []
        
        for selection in self.selection_history:
            scraper_type = selection['recommendation'].scraper_type
            scraper_counts[scraper_type] = scraper_counts.get(scraper_type, 0) + 1
            confidence_scores.append(selection['recommendation'].confidence)
        
        return {
            'total_selections': len(self.selection_history),
            'cached_analyses': len(self.analysis_cache),
            'scraper_types_used': scraper_counts,
            'average_confidence': sum(confidence_scores) / len(confidence_scores),
            'recent_selections': self.selection_history[-5:]  # Last 5 selections
        }
    
    def clear_cache(self) -> None:
        """Clear analysis cache and selection history."""
        self.analysis_cache.clear()
        self.selection_history.clear()
        logger.info("Router cache and history cleared")
    
    def register_scraper(self, name: str, scraper_class: Type[BaseScraper]) -> None:
        """Register a new scraper type with the router.
        
        Args:
            name: Name to register the scraper under
            scraper_class: BaseScraper subclass
        """
        self.scraper_registry[name] = scraper_class
        logger.info(f"Registered scraper: {name}")