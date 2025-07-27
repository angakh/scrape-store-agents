#!/usr/bin/env python3
"""
Quickstart example for scrape-store-agents.

This example demonstrates how to get started with the framework in just a few lines of code.
Run this script to scrape a website and perform searches in under 5 minutes.

Requirements:
    pip install scrape-store-agents

Usage:
    python examples/quickstart.py
"""

import asyncio
import logging
from pathlib import Path

# Import the framework components
from scrape_store_agents.agents.base import Agent
from scrape_store_agents.agents.scrapers.web_scraper import WebScraper
from scrape_store_agents.agents.vector_stores.chromadb_store import ChromaDBStore


async def main():
    """Main quickstart example."""
    print("🕷️ Scrape Store Agents - Quickstart Example")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # 1. Configure the web scraper
    print("\n1. Setting up web scraper...")
    scraper_config = {
        "type": "web",
        "timeout": 30,
        "max_depth": 2,  # Follow links up to 2 levels deep
        "extract_links": True,  # Extract and follow links
        "content_selectors": [
            "main", "article", ".content", ".documentation"
        ],
        "title_selectors": [
            "h1", "title", ".title"
        ]
    }
    scraper = WebScraper(scraper_config)
    print("✅ Web scraper configured")
    
    # 2. Configure the vector store (ChromaDB)
    print("\n2. Setting up vector store...")
    vector_store_config = {
        "type": "chromadb",
        "collection_name": "quickstart_docs",
        "embedding_model": "all-MiniLM-L6-v2",
        "persist_directory": "./data/quickstart_chroma"
    }
    
    # Create data directory if it doesn't exist
    Path("./data").mkdir(exist_ok=True)
    
    vector_store = ChromaDBStore(vector_store_config)
    print("✅ Vector store configured")
    
    # 3. Create the agent
    print("\n3. Creating agent...")
    agent = Agent(scraper, vector_store)
    print("✅ Agent created")
    
    # 4. Test vector store health
    print("\n4. Testing vector store connection...")
    healthy = await agent.vector_store.health_check()
    if healthy:
        print("✅ Vector store is healthy")
    else:
        print("❌ Vector store health check failed")
        return
    
    # 5. Scrape a documentation website
    print("\n5. Scraping FastAPI documentation...")
    url = "https://fastapi.tiangolo.com/"
    
    try:
        # Validate URL first
        if not scraper.validate_url(url):
            print(f"❌ Invalid URL: {url}")
            return
        
        # Scrape and store documents
        documents_count = await agent.scrape_and_store(url)
        print(f"✅ Successfully scraped and stored {documents_count} documents")
        
        if documents_count == 0:
            print("⚠️ No documents were scraped. The website might be blocking requests.")
            return
        
    except Exception as e:
        print(f"❌ Error scraping website: {e}")
        return
    
    # 6. Get agent statistics
    print("\n6. Getting agent statistics...")
    stats = await agent.get_stats()
    print(f"📊 Total documents in store: {stats['document_count']}")
    print(f"🔍 Vector store: {stats['vector_store']}")
    print(f"🕷️ Scraper: {stats['scraper']}")
    
    # 7. Perform some example searches
    print("\n7. Performing example searches...")
    
    search_queries = [
        "FastAPI tutorial",
        "async functions",
        "API documentation",
        "Python web framework"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Searching for: '{query}'")
        try:
            results = await agent.search(query, limit=3)
            
            if results:
                print(f"   Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result.score:.3f}")
                    if result.document.title:
                        print(f"      Title: {result.document.title}")
                    print(f"      URL: {result.document.url}")
                    # Show first 150 characters of content
                    content_preview = result.document.content[:150] + "..." if len(result.document.content) > 150 else result.document.content
                    print(f"      Preview: {content_preview}")
                    print()
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   ❌ Search error: {e}")
    
    # 8. Example with filters
    print("\n8. Example search with URL filter...")
    try:
        results = await agent.search(
            "FastAPI", 
            limit=2,
            filters={"url": "fastapi.tiangolo.com"}
        )
        print(f"   Found {len(results)} results matching URL filter")
        
    except Exception as e:
        print(f"   ❌ Filtered search error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Quickstart complete!")
    print("\nNext steps:")
    print("1. Edit config/sources.yaml to add your own websites")
    print("2. Run 'scrape-store serve' to start the API server")
    print("3. Visit http://localhost:8000/docs for API documentation")
    print("4. Use the CLI: 'scrape-store --help'")


if __name__ == "__main__":
    """Run the quickstart example."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ Quickstart interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Quickstart failed: {e}")
        raise