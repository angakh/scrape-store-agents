"""Command line interface for scrape-store-agents."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional
import yaml

import click
import uvicorn

from .agents.base import Agent
from .agents.reasoning import ReasoningAgent
from .agents.router import IntelligentScraperRouter
from .agents.self_improving import SelfImprovingAgent
from .agents.scrapers.web_scraper import WebScraper
from .agents.vector_stores.chromadb_store import ChromaDBStore
from .config.settings import (
    Settings, 
    load_config, 
    setup_logging, 
    create_example_config
)
from .config.config_loader import (
    load_ai_config,
    validate_ai_config,
    get_config_status,
    create_config_if_missing
)
from .api.main import create_app


logger = logging.getLogger(__name__)


@click.group()
@click.option(
    '--config', 
    '-c', 
    default='config/config.yml',
    help='Configuration file path'
)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: str, verbose: bool):
    """Scrape Store Agents - Web scraping and vector storage framework."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    # Setup basic logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@cli.command()
@click.option('--host', default='0.0.0.0', help='API server host')
@click.option('--port', default=8000, help='API server port')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.pass_context
def serve(ctx, host: str, port: int, reload: bool):
    """Start the API server."""
    config_path = ctx.obj['config_path']
    
    try:
        # Load configuration
        settings = load_config(config_path)
        setup_logging(settings.logging)
        
        # Create FastAPI app
        app = create_app(config_path)
        
        # Override host/port if provided
        final_host = host if host != '0.0.0.0' else settings.api.host
        final_port = port if port != 8000 else settings.api.port
        
        click.echo(f"Starting API server on {final_host}:{final_port}")
        click.echo(f"Documentation available at http://{final_host}:{final_port}/docs")
        
        # Run server
        uvicorn.run(
            "scrape_store_agents.api.main:app",
            host=final_host,
            port=final_port,
            reload=reload,
            log_level="info"
        )
        
    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('url')
@click.option('--max-depth', default=1, help='Maximum crawling depth')
@click.option('--extract-links', is_flag=True, help='Extract and follow links')
@click.option('--output', '-o', help='Output file for scraped content (JSON)')
@click.option('--ai/--no-ai', default=None, help='Enable/disable AI features (auto-detect if not specified)')
@click.option('--ai-analyze', is_flag=True, help='Show AI analysis of the URL before scraping')
@click.option('--self-improving', is_flag=True, help='Use self-improving agent that learns from extractions')
@click.pass_context
def scrape(ctx, url: str, max_depth: int, extract_links: bool, output: Optional[str], ai: Optional[bool], ai_analyze: bool, self_improving: bool):
    """Scrape content from a URL and store in vector database."""
    config_path = ctx.obj['config_path']
    
    async def run_scrape():
        try:
            # Load configuration
            settings = load_config(config_path)
            setup_logging(settings.logging)
            
            # Check if AI should be used
            use_ai = ai
            if use_ai is None:  # Auto-detect
                try:
                    llm_config = load_ai_config(config_path)
                    use_ai = validate_ai_config(llm_config) and settings.ai.reasoning_agent
                except:
                    use_ai = False
            
            click.echo(f"Scraping URL: {url}")
            click.echo(f"AI features: {'enabled' if use_ai else 'disabled'}")
            if self_improving and use_ai:
                click.echo("🧠 Self-improving mode: Agent will learn from extraction quality")
            
            # Initialize components
            vector_store = ChromaDBStore(settings.vector_store.dict())
            
            if use_ai:
                try:
                    # Load AI configuration
                    llm_config = load_ai_config(config_path)
                    
                    # Initialize AI router if enabled
                    if settings.ai.intelligent_router:
                        router = IntelligentScraperRouter(llm_config)
                        
                        # Show AI analysis if requested
                        if ai_analyze:
                            click.echo("\n🧠 AI Analysis:")
                            analysis = await router.analyze_site(url)
                            click.echo(f"   Site Type: {analysis.site_type}")
                            click.echo(f"   Complexity: {analysis.complexity}")
                            click.echo(f"   Recommended Approach: {analysis.recommended_approach}")
                            click.echo(f"   Confidence: {analysis.confidence:.2f}")
                            if analysis.anti_bot_measures:
                                click.echo(f"   Anti-bot Measures: {', '.join(analysis.anti_bot_measures)}")
                            click.echo()
                        
                        # Use intelligent routing
                        click.echo("🤖 Using AI-powered scraper selection...")
                        scraper, documents = await router.route_and_scrape(url)
                        
                        # Store documents
                        if documents:
                            await vector_store.add_documents(documents)
                        
                        documents_count = len(documents)
                        
                    else:
                        # Use reasoning agent with manual scraper (or self-improving if requested)
                        scraper_config = settings.scraper.dict()
                        scraper_config.update({
                            'max_depth': max_depth,
                            'extract_links': extract_links
                        })
                        
                        scraper = WebScraper(scraper_config)
                        
                        # Use self-improving agent if requested
                        if self_improving:
                            agent = SelfImprovingAgent(scraper, vector_store, llm_config)
                            click.echo("🤖 Using self-improving agent with learning capabilities...")
                        else:
                            agent = ReasoningAgent(scraper, vector_store, llm_config)
                        
                        # Show AI analysis if requested
                        if ai_analyze:
                            click.echo("\n🧠 AI Analysis:")
                            analysis = await agent.analyze_url(url)
                            click.echo(f"   Site Type: {analysis.get('site_type', 'unknown')}")
                            click.echo(f"   Recommended Approach: {analysis.get('scraping_approach', 'standard')}")
                            click.echo(f"   Confidence: {analysis.get('confidence', 0):.2f}")
                            click.echo()
                        
                        if self_improving:
                            click.echo("🧠 Using self-improving agent with learning...")
                            documents_count = await agent.scrape_and_store_with_learning(url)
                            
                            # Show learning stats after scraping
                            stats = await agent.get_learning_stats()
                            click.echo(f"📊 Learning Stats:")
                            click.echo(f"   - Total extractions: {stats.get('total_extractions', 0)}")
                            click.echo(f"   - Learned strategies: {stats.get('learned_strategies', 0)}")
                            click.echo(f"   - Average quality: {stats.get('average_quality', 0):.2f}")
                        else:
                            click.echo("🤖 Using AI-powered reasoning agent...")
                            documents_count = await agent.scrape_and_store(url)
                    
                except Exception as ai_error:
                    click.echo(f"⚠️  AI features failed: {ai_error}")
                    click.echo("Falling back to standard scraping...")
                    use_ai = False
            
            if not use_ai:
                # Standard scraping without AI
                scraper_config = settings.scraper.dict()
                scraper_config.update({
                    'max_depth': max_depth,
                    'extract_links': extract_links
                })
                
                scraper = WebScraper(scraper_config)
                agent = Agent(scraper, vector_store)
                
                # Validate URL
                if not scraper.validate_url(url):
                    click.echo(f"Invalid or blocked URL: {url}", err=True)
                    return
                
                # Scrape and store
                documents_count = await agent.scrape_and_store(url)
            
            click.echo(f"✅ Successfully scraped and stored {documents_count} documents")
            
            # Optionally save to file
            if output:
                if use_ai and 'documents' in locals():
                    # Use documents from AI scraping
                    save_documents = documents
                else:
                    # Re-scrape for output (without storing)
                    scraper_config = settings.scraper.dict()
                    scraper_config.update({
                        'max_depth': max_depth,
                        'extract_links': extract_links
                    })
                    temp_scraper = WebScraper(scraper_config)
                    save_documents = await temp_scraper.scrape(url)
                
                import json
                output_data = []
                for doc in save_documents:
                    output_data.append({
                        'url': doc.url,
                        'title': doc.title,
                        'content': doc.content,
                        'metadata': doc.metadata,
                        'timestamp': doc.timestamp.isoformat() if doc.timestamp else None
                    })
                
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                click.echo(f"📄 Scraped content saved to {output}")
            
        except Exception as e:
            click.echo(f"❌ Error scraping URL: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(run_scrape())


@cli.command()
@click.argument('query')
@click.option('--limit', default=10, help='Maximum number of results')
@click.option('--url-filter', help='Filter results by URL pattern')
@click.option('--format', 'output_format', default='text', 
              type=click.Choice(['text', 'json']), help='Output format')
@click.pass_context
def search(ctx, query: str, limit: int, url_filter: Optional[str], output_format: str):
    """Search for documents similar to the query."""
    config_path = ctx.obj['config_path']
    
    async def run_search():
        try:
            # Load configuration
            settings = load_config(config_path)
            setup_logging(settings.logging)
            
            # Initialize agent
            scraper = WebScraper(settings.scraper.dict())
            vector_store = ChromaDBStore(settings.vector_store.dict())
            agent = Agent(scraper, vector_store)
            
            # Prepare filters
            filters = {}
            if url_filter:
                filters['url'] = url_filter
            
            click.echo(f"Searching for: {query}")
            
            # Perform search
            results = await agent.search(
                query=query,
                limit=limit,
                filters=filters if filters else None
            )
            
            if not results:
                click.echo("No results found.")
                return
            
            # Display results
            if output_format == 'json':
                import json
                result_data = []
                for result in results:
                    result_data.append({
                        'score': result.score,
                        'distance': result.distance,
                        'url': result.document.url,
                        'title': result.document.title,
                        'content': result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content,
                        'metadata': result.document.metadata
                    })
                
                click.echo(json.dumps(result_data, indent=2))
            else:
                click.echo(f"\nFound {len(results)} results:\n")
                for i, result in enumerate(results, 1):
                    click.echo(f"{i}. Score: {result.score:.3f}")
                    if result.document.title:
                        click.echo(f"   Title: {result.document.title}")
                    click.echo(f"   URL: {result.document.url}")
                    content_preview = result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content
                    click.echo(f"   Content: {content_preview}")
                    click.echo()
            
        except Exception as e:
            click.echo(f"Error searching: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(run_search())


@cli.command()
@click.pass_context
def status(ctx):
    """Show agent status and statistics."""
    config_path = ctx.obj['config_path']
    
    async def run_status():
        try:
            # Load configuration
            settings = load_config(config_path)
            setup_logging(settings.logging)
            
            # Check AI configuration
            config_status = get_config_status()
            ai_available = False
            ai_stats = {}
            
            try:
                llm_config = load_ai_config(config_path)
                ai_available = validate_ai_config(llm_config)
                
                if ai_available and settings.ai.reasoning_agent:
                    # Initialize AI agent for stats (SelfImprovingAgent if available)
                    scraper = WebScraper(settings.scraper.dict())
                    vector_store = ChromaDBStore(settings.vector_store.dict())
                    
                    # Try to use SelfImprovingAgent for more detailed stats
                    try:
                        ai_agent = SelfImprovingAgent(scraper, vector_store, llm_config)
                        ai_stats = await ai_agent.get_learning_stats()
                        ai_stats['self_improving'] = True
                    except:
                        # Fallback to ReasoningAgent
                        ai_agent = ReasoningAgent(scraper, vector_store, llm_config)
                        ai_stats = await ai_agent.get_reasoning_stats()
                        ai_stats['self_improving'] = False
                else:
                    # Initialize basic agent
                    scraper = WebScraper(settings.scraper.dict())
                    vector_store = ChromaDBStore(settings.vector_store.dict())
                    agent = Agent(scraper, vector_store)
                    ai_stats = await agent.get_stats()
            except Exception as e:
                # Fallback to basic agent
                scraper = WebScraper(settings.scraper.dict())
                vector_store = ChromaDBStore(settings.vector_store.dict())
                agent = Agent(scraper, vector_store)
                ai_stats = await agent.get_stats()
            
            click.echo("🕷️  Scrape Store Agents Status")
            click.echo("=" * 35)
            
            # Basic stats
            click.echo(f"Scraper: {ai_stats.get('scraper', 'WebScraper')}")
            click.echo(f"Vector Store: {ai_stats.get('vector_store', 'ChromaDBStore')}")
            click.echo(f"Document Count: {ai_stats.get('document_count', 0)}")
            click.echo(f"Vector Store Healthy: {ai_stats.get('vector_store_healthy', False)}")
            
            # AI Status
            click.echo(f"\n🧠 AI Features:")
            click.echo(f"   Configuration: {'✅ Valid' if config_status['ai_configured'] else '❌ Invalid/Missing'}")
            click.echo(f"   Provider: {settings.ai.provider if ai_available else 'Not configured'}")
            click.echo(f"   Model: {settings.ai.model if ai_available else 'Not configured'}")
            click.echo(f"   Reasoning Agent: {'✅ Enabled' if settings.ai.reasoning_agent and ai_available else '❌ Disabled'}")
            click.echo(f"   Intelligent Router: {'✅ Enabled' if settings.ai.intelligent_router and ai_available else '❌ Disabled'}")
            
            # AI Stats if available
            if 'memory_entries' in ai_stats:
                click.echo(f"\n🤖 AI Activity:")
                click.echo(f"   Memory Entries: {ai_stats['memory_entries']}")
                click.echo(f"   URL Analyses: {ai_stats['url_analyses']}")
                click.echo(f"   Strategy Adaptations: {ai_stats['strategy_adaptations']}")
                
                # Self-improving stats if available
                if ai_stats.get('self_improving'):
                    click.echo(f"\n🧠 Self-Improving Features:")
                    click.echo(f"   Total Extractions: {ai_stats.get('total_extractions', 0)}")
                    click.echo(f"   Learned Strategies: {ai_stats.get('learned_strategies', 0)}")
                    click.echo(f"   Average Quality: {ai_stats.get('average_quality', 0):.2f}")
                    click.echo(f"   Site Patterns Learned: {ai_stats.get('site_patterns_learned', 0)}")
                    
                    improvement_metrics = ai_stats.get('improvement_metrics', {})
                    click.echo(f"   Quality Improvements: {improvement_metrics.get('quality_improvements', 0)}")
                    click.echo(f"   Successful Predictions: {improvement_metrics.get('successful_predictions', 0)}")
            
            # API Keys status
            api_keys = config_status.get('api_keys_found', {})
            click.echo(f"\n🔐 API Keys:")
            click.echo(f"   OpenAI: {'✅ Found' if api_keys.get('openai') else '❌ Not found'}")
            click.echo(f"   Anthropic: {'✅ Found' if api_keys.get('anthropic') else '❌ Not found'}")
            
            # Show configured sources
            if settings.sources:
                click.echo(f"\n📚 Configured Sources ({len(settings.sources)}):")
                for source in settings.sources:
                    status_text = "✅ enabled" if source.enabled else "❌ disabled"
                    click.echo(f"   - {source.name}: {source.url} ({status_text})")
            
            # Configuration file status
            click.echo(f"\n⚙️  Configuration:")
            click.echo(f"   Config file: {'✅ Found' if config_status['config_exists'] else '❌ Missing'}")
            if not config_status['config_exists']:
                click.echo(f"   💡 Run: cp {config_status['example_path']} {config_status['config_path']}")
            
        except Exception as e:
            click.echo(f"❌ Error getting status: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(run_status())


@cli.command()
@click.option('--source', help='Scrape specific configured source')
@click.option('--all', 'scrape_all', is_flag=True, help='Scrape all enabled sources')
@click.pass_context
def run(ctx, source: Optional[str], scrape_all: bool):
    """Run scraping for configured sources."""
    config_path = ctx.obj['config_path']
    
    if not source and not scrape_all:
        click.echo("Please specify --source NAME or --all", err=True)
        sys.exit(1)
    
    async def run_scraping():
        try:
            # Load configuration
            settings = load_config(config_path)
            setup_logging(settings.logging)
            
            # Initialize agent
            scraper = WebScraper(settings.scraper.dict())
            vector_store = ChromaDBStore(settings.vector_store.dict())
            agent = Agent(scraper, vector_store)
            
            # Find sources to scrape
            sources_to_scrape = []
            
            if scrape_all:
                sources_to_scrape = [s for s in settings.sources if s.enabled]
            elif source:
                for s in settings.sources:
                    if s.name == source:
                        if not s.enabled:
                            click.echo(f"Source '{source}' is disabled", err=True)
                            return
                        sources_to_scrape = [s]
                        break
                else:
                    click.echo(f"Source '{source}' not found", err=True)
                    return
            
            if not sources_to_scrape:
                click.echo("No sources to scrape")
                return
            
            total_documents = 0
            
            for source_config in sources_to_scrape:
                click.echo(f"Scraping source: {source_config.name}")
                
                # Create custom scraper config
                custom_config = {**settings.scraper.dict(), **source_config.custom_config}
                custom_scraper = WebScraper(custom_config)
                custom_agent = Agent(custom_scraper, vector_store)
                
                try:
                    documents_count = await custom_agent.scrape_and_store(source_config.url)
                    total_documents += documents_count
                    click.echo(f"  Added {documents_count} documents")
                except Exception as e:
                    click.echo(f"  Error: {e}", err=True)
            
            click.echo(f"\nTotal documents added: {total_documents}")
            
        except Exception as e:
            click.echo(f"Error running scraping: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(run_scraping())


@cli.command()
@click.option('--force', is_flag=True, help='Overwrite existing config.yml file')
def init(force: bool):
    """Initialize configuration from template."""
    config_dir = Path("config")
    config_file = config_dir / "config.yml"
    example_file = config_dir / "config.example.yml"
    
    if config_file.exists() and not force:
        click.echo(f"❌ Configuration file {config_file} already exists. Use --force to overwrite.", err=True)
        sys.exit(1)
    
    if not example_file.exists():
        click.echo(f"❌ Example configuration file {example_file} not found.", err=True)
        sys.exit(1)
    
    try:
        # Create config directory if needed
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy example to config.yml
        config_file.write_text(example_file.read_text())
        
        click.echo(f"✅ Configuration file created: {config_file}")
        click.echo("\n📝 Next steps:")
        click.echo("   1. Edit config.yml with your API keys and settings")
        click.echo("   2. Get API keys from:")
        click.echo("      - OpenAI: https://platform.openai.com/api-keys")
        click.echo("      - Anthropic: https://console.anthropic.com/")
        click.echo("   3. Run 'scrape-store status' to verify configuration")
        
    except Exception as e:
        click.echo(f"❌ Error creating configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('url')
@click.option('--detailed', is_flag=True, help='Show detailed analysis information')
@click.pass_context
def analyze(ctx, url: str, detailed: bool):
    """Analyze a URL using AI to understand optimal scraping strategy."""
    config_path = ctx.obj['config_path']
    
    async def run_analyze():
        try:
            # Load configuration
            settings = load_config(config_path)
            setup_logging(settings.logging)
            
            # Check AI configuration
            llm_config = load_ai_config(config_path)
            if not validate_ai_config(llm_config):
                click.echo("❌ AI configuration is invalid or missing.", err=True)
                click.echo("💡 Run 'scrape-store init' to set up configuration.")
                sys.exit(1)
            
            # Initialize AI router
            router = IntelligentScraperRouter(llm_config)
            
            click.echo(f"🧠 Analyzing URL: {url}")
            click.echo("🔄 Processing...")
            
            # Perform analysis
            analysis = await router.analyze_site(url)
            recommendation = await router.select_scraper(url)
            
            # Display results
            click.echo(f"\n📊 AI Analysis Results:")
            click.echo(f"   🌐 Site Type: {analysis.site_type}")
            click.echo(f"   🔧 Complexity: {analysis.complexity}")
            click.echo(f"   📈 Confidence: {analysis.confidence:.2f}")
            click.echo(f"   🎯 Recommended Approach: {analysis.recommended_approach}")
            
            if analysis.anti_bot_measures:
                click.echo(f"   🛡️  Anti-bot Measures: {', '.join(analysis.anti_bot_measures)}")
            
            click.echo(f"\n🤖 Scraper Recommendation:")
            click.echo(f"   📦 Scraper Type: {recommendation.scraper_type}")
            click.echo(f"   📈 Confidence: {recommendation.confidence:.2f}")
            click.echo(f"   💭 Rationale: {recommendation.rationale}")
            
            if detailed:
                click.echo(f"\n🔧 Recommended Configuration:")
                import json
                click.echo(json.dumps(recommendation.config, indent=4))
                
                if analysis.content_structure:
                    click.echo(f"\n🏗️  Content Structure Analysis:")
                    click.echo(json.dumps(analysis.content_structure, indent=4))
            
            # Show next steps
            click.echo(f"\n💡 Next Steps:")
            click.echo(f"   • Run: scrape-store scrape {url} --ai --ai-analyze")
            click.echo(f"   • Use AI-powered scraping with intelligent routing")
            
        except Exception as e:
            click.echo(f"❌ Error analyzing URL: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(run_analyze())


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear all documents?')
@click.pass_context
def clear(ctx):
    """Clear all documents from the vector store."""
    config_path = ctx.obj['config_path']
    
    async def run_clear():
        try:
            # Load configuration
            settings = load_config(config_path)
            setup_logging(settings.logging)
            
            # Initialize vector store
            vector_store = ChromaDBStore(settings.vector_store.dict())
            
            # Clear documents
            if hasattr(vector_store, 'clear_collection'):
                await vector_store.clear_collection()
                click.echo("✅ All documents cleared successfully.")
            else:
                click.echo("❌ Vector store does not support clearing documents.", err=True)
            
        except Exception as e:
            click.echo(f"❌ Error clearing documents: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(run_clear())


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()