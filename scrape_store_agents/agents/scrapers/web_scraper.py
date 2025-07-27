"""Generic web scraper implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse
import re

import requests
from bs4 import BeautifulSoup, Comment

from ..base import BaseScraper, Document


logger = logging.getLogger(__name__)


class WebScraper(BaseScraper):
    """Generic web scraper for extracting content from web pages."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize web scraper.
        
        Args:
            config: Configuration containing:
                - user_agent: User agent string
                - timeout: Request timeout in seconds
                - max_content_length: Maximum content length to process
                - allowed_domains: List of allowed domains (optional)
                - blocked_domains: List of blocked domains (optional)
                - extract_links: Whether to extract and follow links
                - max_depth: Maximum crawling depth
                - content_selectors: CSS selectors for content extraction
                - title_selectors: CSS selectors for title extraction
        """
        super().__init__(config)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.get_config(
                'user_agent', 
                'Mozilla/5.0 (compatible; ScrapeStoreAgent/1.0)'
            )
        })
        
        self.timeout = self.get_config('timeout', 30)
        self.max_content_length = self.get_config('max_content_length', 10 * 1024 * 1024)
        self.allowed_domains = set(self.get_config('allowed_domains', []))
        self.blocked_domains = set(self.get_config('blocked_domains', []))
        self.extract_links = self.get_config('extract_links', False)
        self.max_depth = self.get_config('max_depth', 1)
        
        # Content extraction selectors
        self.content_selectors = self.get_config('content_selectors', [
            'main', 'article', '.content', '.post-content', '.entry-content',
            '.article-body', '.story-body', '#content'
        ])
        self.title_selectors = self.get_config('title_selectors', [
            'h1', 'title', '.title', '.headline', '.post-title'
        ])
        
        # Elements to remove from content
        self.remove_selectors = self.get_config('remove_selectors', [
            'script', 'style', 'nav', 'header', 'footer', '.sidebar',
            '.navigation', '.menu', '.ads', '.advertisement', '.social-share'
        ])
    
    def validate_url(self, url: str) -> bool:
        """Validate if this scraper can handle the given URL."""
        try:
            parsed = urlparse(url)
            
            # Check if URL has valid scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check allowed domains
            if self.allowed_domains and parsed.netloc not in self.allowed_domains:
                return False
            
            # Check blocked domains
            if parsed.netloc in self.blocked_domains:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating URL {url}: {e}")
            return False
    
    async def scrape(self, url: str, **kwargs) -> List[Document]:
        """Scrape content from a URL.
        
        Args:
            url: The URL to scrape
            **kwargs: Additional parameters:
                - depth: Current crawling depth (for recursive scraping)
                - visited: Set of already visited URLs
        
        Returns:
            List of Document objects containing scraped content
        """
        depth = kwargs.get('depth', 0)
        visited = kwargs.get('visited', set())
        
        if depth > self.max_depth or url in visited:
            return []
        
        visited.add(url)
        documents = []
        
        try:
            # Run the blocking request in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.session.get(url, timeout=self.timeout)
            )
            
            response.raise_for_status()
            
            # Check content length
            if len(response.content) > self.max_content_length:
                logger.warning(f"Content too large for {url}, skipping")
                return documents
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main document
            doc = self._extract_document(soup, url)
            if doc and doc.content.strip():
                documents.append(doc)
            
            # Extract links if enabled
            if self.extract_links and depth < self.max_depth:
                links = self._extract_links(soup, url)
                for link in links:
                    if self.validate_url(link):
                        subdocs = await self.scrape(
                            link, 
                            depth=depth + 1, 
                            visited=visited
                        )
                        documents.extend(subdocs)
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        return documents
    
    def _extract_document(self, soup: BeautifulSoup, url: str) -> Optional[Document]:
        """Extract document content from parsed HTML."""
        try:
            # Remove unwanted elements
            for selector in self.remove_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            content = self._extract_content(soup)
            
            if not content:
                return None
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            return Document(
                content=content,
                url=url,
                title=title,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error extracting document from {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract title from the page."""
        for selector in self.title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title:
                    return title
        return None
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page."""
        content_parts = []
        
        # Try specific content selectors first
        for selector in self.content_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if text and len(text) > 100:  # Minimum content length
                        content_parts.append(text)
                break
        
        # If no specific content found, extract from body
        if not content_parts:
            body = soup.find('body')
            if body:
                # Remove remaining unwanted elements
                for tag in ['script', 'style', 'nav', 'header', 'footer']:
                    for element in body.find_all(tag):
                        element.decompose()
                
                text = body.get_text(separator=' ', strip=True)
                if text:
                    content_parts.append(text)
        
        # Clean and join content
        content = ' '.join(content_parts)
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        return content.strip()
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from the page."""
        metadata = {'source_url': url}
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        # Extract language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')
        
        # Extract word count
        if 'word_count' not in metadata:
            content = soup.get_text()
            metadata['word_count'] = len(content.split())
        
        return metadata
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from the page for recursive scraping."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Basic filtering
            if (absolute_url.startswith(('http://', 'https://')) and 
                absolute_url not in links):
                links.append(absolute_url)
        
        return links[:10]  # Limit number of links to prevent explosion