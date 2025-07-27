"""FastAPI application for scrape-store-agents."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path

from ..agents.base import Agent, Document, SearchResult
from ..agents.reasoning import ReasoningAgent
from ..agents.router import IntelligentScraperRouter
from ..agents.self_improving import SelfImprovingAgent
from ..agents.scrapers.web_scraper import WebScraper
from ..agents.vector_stores.chromadb_store import ChromaDBStore
from ..config.settings import Settings, load_config
from ..config.config_loader import (
    load_ai_config,
    validate_ai_config,
    get_config_status
)
from ..web.routes import setup_web_routes


logger = logging.getLogger(__name__)


# Pydantic models for API
class ScrapeRequest(BaseModel):
    """Request model for scraping."""
    url: str = Field(..., description="URL to scrape")
    scraper_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom scraper configuration"
    )
    use_ai: Optional[bool] = Field(
        default=None,
        description="Enable AI features (auto-detect if not specified)"
    )
    analyze_first: bool = Field(
        default=False,
        description="Run AI analysis before scraping"
    )
    self_improving: bool = Field(
        default=False,
        description="Use self-improving agent that learns from extractions"
    )


class SearchRequest(BaseModel):
    """Request model for searching."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional search filters"
    )


class SearchResponse(BaseModel):
    """Response model for search results."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    execution_time_ms: float


class ScrapeResponse(BaseModel):
    """Response model for scraping."""
    url: str
    documents_added: int
    success: bool
    message: str
    execution_time_ms: float
    ai_used: bool = False
    ai_analysis: Optional[Dict[str, Any]] = None
    scraper_used: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """Request model for URL analysis."""
    url: str = Field(..., description="URL to analyze")
    detailed: bool = Field(default=False, description="Include detailed analysis")


class AnalyzeResponse(BaseModel):
    """Response model for URL analysis."""
    url: str
    site_analysis: Dict[str, Any]
    scraper_recommendation: Dict[str, Any]
    execution_time_ms: float


class StatusResponse(BaseModel):
    """Response model for status."""
    status: str
    version: str
    document_count: int
    vector_store_healthy: bool
    uptime_seconds: float
    ai_features: Dict[str, Any] = {}


class AIConfigResponse(BaseModel):
    """Response model for AI configuration status."""
    ai_available: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    reasoning_agent_enabled: bool = False
    intelligent_router_enabled: bool = False
    api_keys_configured: Dict[str, bool] = {}
    config_file_exists: bool = False


# Global variables
app = FastAPI(
    title="Scrape Store Agents API",
    description="API for web scraping and vector storage framework",
    version="0.1.0"
)

# Global state
agent: Optional[Agent] = None
ai_agent: Optional[ReasoningAgent] = None
ai_router: Optional[IntelligentScraperRouter] = None
self_improving_agent: Optional[SelfImprovingAgent] = None
settings: Optional[Settings] = None
ai_config: Optional[Dict[str, Any]] = None
ai_available: bool = False
start_time = datetime.utcnow()


# Add OpenTelemetry setup if enabled
try:
    import scrape_store_agents.opentelemetry_setup
except ImportError:
    pass


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create and configure FastAPI application."""
    global agent, ai_agent, ai_router, self_improving_agent, settings, ai_config, ai_available
    
    # Load configuration
    settings = load_config(config_path)
    
    # Setup CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup static files and web interface
    web_dir = Path(__file__).parent.parent / "web"
    static_dir = web_dir / "static"
    templates_dir = web_dir / "templates"
    
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    if templates_dir.exists():
        setup_web_routes(app, str(templates_dir))
    
    # Initialize basic agent
    scraper = WebScraper(settings.scraper.dict())
    vector_store = ChromaDBStore(settings.vector_store.dict())
    agent = Agent(scraper, vector_store)
    
    # Try to initialize AI features
    try:
        ai_config = load_ai_config(config_path)
        ai_available = validate_ai_config(ai_config)
        
        if ai_available:
            # Initialize AI agent
            if settings.ai.reasoning_agent:
                ai_agent = ReasoningAgent(scraper, vector_store, ai_config)
                logger.info(f"ReasoningAgent initialized with {ai_config['provider']} {ai_config['model']}")
            
            # Initialize AI router
            if settings.ai.intelligent_router:
                ai_router = IntelligentScraperRouter(ai_config)
                logger.info("IntelligentScraperRouter initialized")
            
            # Initialize self-improving agent
            try:
                self_improving_agent = SelfImprovingAgent(scraper, vector_store, ai_config)
                logger.info("SelfImprovingAgent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SelfImprovingAgent: {e}")
                self_improving_agent = None
                
        else:
            logger.warning("AI configuration invalid - AI features disabled")
            
    except Exception as e:
        logger.warning(f"Failed to initialize AI features: {e}")
        ai_available = False
    
    logger.info(f"FastAPI application initialized - AI features: {'enabled' if ai_available else 'disabled'}")
    return app


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    if not agent:
        # Fallback initialization if create_app wasn't called
        global settings
        settings = load_config()
        
        scraper = WebScraper(settings.scraper.dict())
        vector_store = ChromaDBStore(settings.vector_store.dict())
        globals()['agent'] = Agent(scraper, vector_store)
    
    # Test vector store connection
    if agent:
        healthy = await agent.vector_store.health_check()
        if not healthy:
            logger.warning("Vector store health check failed")
        else:
            logger.info("Vector store connection healthy")


@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API info endpoint."""
    return {
        "message": "Scrape Store Agents API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Get stats from AI agent if available, otherwise basic agent
        if ai_agent and ai_available:
            stats = await ai_agent.get_reasoning_stats()
        else:
            stats = await agent.get_stats()
            
        uptime = (datetime.utcnow() - start_time).total_seconds()
        
        # Build AI features status
        ai_features = {
            "available": ai_available,
            "reasoning_agent": ai_agent is not None,
            "intelligent_router": ai_router is not None,
        }
        
        if ai_available and ai_config:
            ai_features.update({
                "provider": ai_config.get("provider"),
                "model": ai_config.get("model"),
            })
        
        return StatusResponse(
            status="healthy",
            version="0.1.0",
            document_count=stats.get("document_count", 0),
            vector_store_healthy=stats.get("vector_store_healthy", False),
            uptime_seconds=uptime,
            ai_features=ai_features
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_url(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks
):
    """Scrape content from a URL and store in vector database."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    start_time_req = datetime.utcnow()
    
    try:
        # Determine if AI should be used
        use_ai = request.use_ai
        if use_ai is None:  # Auto-detect
            use_ai = ai_available and settings.ai.reasoning_agent
        
        ai_analysis = None
        scraper_used = "WebScraper"
        documents_added = 0
        
        if use_ai and ai_available:
            try:
                # Use self-improving agent if requested and available
                if request.self_improving and self_improving_agent:
                    # AI analysis if requested
                    if request.analyze_first:
                        analysis = await self_improving_agent.analyze_url(request.url)
                        ai_analysis = {
                            "site_type": analysis.get("site_type", "unknown"),
                            "confidence": analysis.get("confidence", 0),
                            "recommended_approach": analysis.get("scraping_approach", "standard")
                        }
                    
                    # Apply custom config if provided
                    if request.scraper_config:
                        config = {**agent.scraper.config, **request.scraper_config}
                        temp_scraper = WebScraper(config)
                        temp_agent = SelfImprovingAgent(temp_scraper, agent.vector_store, ai_config)
                        documents_added = await temp_agent.scrape_and_store_with_learning(request.url)
                    else:
                        documents_added = await self_improving_agent.scrape_and_store_with_learning(request.url)
                    
                    scraper_used = "AI-SelfImprovingAgent"
                
                # Use AI-powered scraping
                elif settings.ai.intelligent_router and ai_router:
                    # AI analysis if requested
                    if request.analyze_first:
                        analysis = await ai_router.analyze_site(request.url)
                        recommendation = await ai_router.select_scraper(request.url)
                        
                        ai_analysis = {
                            "site_type": analysis.site_type,
                            "complexity": analysis.complexity,
                            "confidence": analysis.confidence,
                            "recommended_approach": analysis.recommended_approach,
                            "scraper_recommendation": {
                                "type": recommendation.scraper_type,
                                "confidence": recommendation.confidence,
                                "rationale": recommendation.rationale
                            }
                        }
                    
                    # Use intelligent routing
                    scraper_instance, documents = await ai_router.route_and_scrape(request.url)
                    scraper_used = f"AI-{scraper_instance.__class__.__name__}"
                    
                    # Store documents
                    if documents:
                        await agent.vector_store.add_documents(documents)
                    documents_added = len(documents)
                    
                elif ai_agent:
                    # Use reasoning agent with manual scraper
                    if request.analyze_first:
                        analysis = await ai_agent.analyze_url(request.url)
                        ai_analysis = {
                            "site_type": analysis.get("site_type", "unknown"),
                            "confidence": analysis.get("confidence", 0),
                            "recommended_approach": analysis.get("scraping_approach", "standard")
                        }
                    
                    # Apply custom config if provided
                    if request.scraper_config:
                        config = {**agent.scraper.config, **request.scraper_config}
                        temp_scraper = WebScraper(config)
                        ai_agent_temp = ReasoningAgent(temp_scraper, agent.vector_store, ai_config)
                        documents_added = await ai_agent_temp.scrape_and_store(request.url)
                    else:
                        documents_added = await ai_agent.scrape_and_store(request.url)
                    
                    scraper_used = "AI-ReasoningAgent"
                    
            except Exception as ai_error:
                logger.warning(f"AI scraping failed: {ai_error}, falling back to standard scraping")
                use_ai = False
        
        if not use_ai:
            # Standard scraping without AI
            if not agent.scraper.validate_url(request.url):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid or blocked URL: {request.url}"
                )
            
            # Apply custom config if provided
            if request.scraper_config:
                config = {**agent.scraper.config, **request.scraper_config}
                temp_scraper = WebScraper(config)
                temp_agent = Agent(temp_scraper, agent.vector_store)
                documents_added = await temp_agent.scrape_and_store(request.url)
            else:
                documents_added = await agent.scrape_and_store(request.url)
        
        execution_time = (datetime.utcnow() - start_time_req).total_seconds() * 1000
        
        return ScrapeResponse(
            url=request.url,
            documents_added=documents_added,
            success=True,
            message=f"Successfully scraped and stored {documents_added} documents",
            execution_time_ms=execution_time,
            ai_used=use_ai and ai_available,
            ai_analysis=ai_analysis,
            scraper_used=scraper_used
        )
        
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time_req).total_seconds() * 1000
        logger.error(f"Error scraping {request.url}: {e}")
        
        return ScrapeResponse(
            url=request.url,
            documents_added=0,
            success=False,
            message=f"Error scraping URL: {str(e)}",
            execution_time_ms=execution_time,
            ai_used=False
        )


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for documents similar to the query."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    start_time_req = datetime.utcnow()
    
    try:
        # Perform search
        results = await agent.search(
            query=request.query,
            limit=request.limit,
            filters=request.filters
        )
        
        # Convert results to dict format
        result_dicts = []
        for result in results:
            result_dict = {
                "content": result.document.content[:500] + "..." if len(result.document.content) > 500 else result.document.content,
                "url": result.document.url,
                "title": result.document.title,
                "score": result.score,
                "distance": result.distance,
                "metadata": result.document.metadata,
                "timestamp": result.document.timestamp.isoformat() if result.document.timestamp else None
            }
            result_dicts.append(result_dict)
        
        execution_time = (datetime.utcnow() - start_time_req).total_seconds() * 1000
        
        return SearchResponse(
            query=request.query,
            results=result_dicts,
            total_results=len(results),
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error searching for '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search", response_model=SearchResponse)
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    url_filter: Optional[str] = Query(None, description="Filter by URL pattern")
):
    """Search for documents (GET endpoint for simple queries)."""
    filters = {}
    if url_filter:
        filters["url"] = url_filter
    
    request = SearchRequest(
        query=q,
        limit=limit,
        filters=filters if filters else None
    )
    
    return await search_documents(request)


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get agent and vector store statistics."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        stats = await agent.get_stats()
        uptime = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            **stats,
            "api_uptime_seconds": uptime,
            "version": "0.1.0"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Check if vector store supports clearing
        if hasattr(agent.vector_store, 'clear_collection'):
            await agent.vector_store.clear_collection()
            return {"message": "All documents cleared successfully"}
        else:
            raise HTTPException(
                status_code=501,
                detail="Vector store does not support clearing documents"
            )
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@app.get("/sources")
async def list_sources():
    """List configured scraping sources."""
    if not settings:
        raise HTTPException(status_code=503, detail="Settings not loaded")
    
    sources = []
    for source in settings.sources:
        sources.append({
            "name": source.name,
            "url": source.url,
            "scraper_type": source.scraper_type,
            "enabled": source.enabled,
            "schedule": source.schedule
        })
    
    return {"sources": sources}


@app.post("/sources/{source_name}/scrape")
async def scrape_source(source_name: str, background_tasks: BackgroundTasks):
    """Scrape a specific configured source."""
    if not settings or not agent:
        raise HTTPException(status_code=503, detail="Application not initialized")
    
    # Find source configuration
    source_config = None
    for source in settings.sources:
        if source.name == source_name:
            source_config = source
            break
    
    if not source_config:
        raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found")
    
    if not source_config.enabled:
        raise HTTPException(status_code=400, detail=f"Source '{source_name}' is disabled")
    
    # Create scrape request
    scraper_config = {**settings.scraper.dict(), **source_config.custom_config}
    request = ScrapeRequest(
        url=source_config.url,
        scraper_config=scraper_config
    )
    
    return await scrape_url(request, background_tasks)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_url(request: AnalyzeRequest):
    """Analyze a URL using AI to determine optimal scraping strategy."""
    if not ai_available or not ai_router:
        raise HTTPException(
            status_code=503, 
            detail="AI features not available. Check configuration and API keys."
        )
    
    start_time_req = datetime.utcnow()
    
    try:
        # Perform AI analysis
        site_analysis = await ai_router.analyze_site(request.url)
        scraper_recommendation = await ai_router.select_scraper(request.url)
        
        # Convert to dict format
        analysis_dict = {
            "site_type": site_analysis.site_type,
            "complexity": site_analysis.complexity,
            "confidence": site_analysis.confidence,
            "recommended_approach": site_analysis.recommended_approach,
            "anti_bot_measures": site_analysis.anti_bot_measures
        }
        
        recommendation_dict = {
            "scraper_type": scraper_recommendation.scraper_type,
            "confidence": scraper_recommendation.confidence,
            "rationale": scraper_recommendation.rationale,
            "config": scraper_recommendation.config if request.detailed else {}
        }
        
        if request.detailed and site_analysis.content_structure:
            analysis_dict["content_structure"] = site_analysis.content_structure
        
        execution_time = (datetime.utcnow() - start_time_req).total_seconds() * 1000
        
        return AnalyzeResponse(
            url=request.url,
            site_analysis=analysis_dict,
            scraper_recommendation=recommendation_dict,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/ai/config", response_model=AIConfigResponse)
async def get_ai_config():
    """Get AI configuration status."""
    try:
        config_status = get_config_status()
        
        return AIConfigResponse(
            ai_available=ai_available,
            provider=ai_config.get("provider") if ai_config else None,
            model=ai_config.get("model") if ai_config else None,
            reasoning_agent_enabled=ai_agent is not None,
            intelligent_router_enabled=ai_router is not None,
            api_keys_configured=config_status.get("api_keys_found", {}),
            config_file_exists=config_status.get("config_exists", False)
        )
        
    except Exception as e:
        logger.error(f"Error getting AI config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI config: {str(e)}")


@app.get("/ai/stats")
async def get_ai_stats():
    """Get AI agent statistics and performance metrics."""
    if not ai_available:
        raise HTTPException(status_code=503, detail="AI features not available")
    
    try:
        stats = {}
        
        # Get reasoning agent stats
        if ai_agent:
            reasoning_stats = await ai_agent.get_reasoning_stats()
            stats["reasoning_agent"] = reasoning_stats
        
        # Get router stats
        if ai_router:
            router_stats = ai_router.get_router_stats()
            stats["intelligent_router"] = router_stats
        
        # Get self-improving agent stats
        if self_improving_agent:
            learning_stats = await self_improving_agent.get_learning_stats()
            stats["self_improving_agent"] = learning_stats
        
        # Add configuration info
        stats["configuration"] = {
            "provider": ai_config.get("provider") if ai_config else None,
            "model": ai_config.get("model") if ai_config else None,
            "temperature": ai_config.get("temperature") if ai_config else None,
            "self_improving_available": self_improving_agent is not None,
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting AI stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI stats: {str(e)}")


@app.post("/ai/clear-cache")
async def clear_ai_cache():
    """Clear AI analysis cache and memory."""
    if not ai_available:
        raise HTTPException(status_code=503, detail="AI features not available")
    
    try:
        cleared_items = []
        
        # Clear reasoning agent memory
        if ai_agent:
            ai_agent.clear_memory()
            cleared_items.append("reasoning_agent_memory")
        
        # Clear router cache
        if ai_router:
            ai_router.clear_cache()
            cleared_items.append("router_cache")
        
        # Clear self-improving agent learning data
        if self_improving_agent:
            cleared_stats = self_improving_agent.clear_learning_data()
            cleared_items.append("self_improving_learning_data")
            logger.info(f"Cleared self-improving learning data: {cleared_stats}")
        
        return {
            "message": "AI cache cleared successfully",
            "items_cleared": cleared_items
        }
        
    except Exception as e:
        logger.error(f"Error clearing AI cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear AI cache: {str(e)}")


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Load configuration
    config_file = "config/sources.yaml"
    app = create_app(config_file)
    
    # Run server
    uvicorn.run(
        app,
        host=settings.api.host if settings else "0.0.0.0",
        port=settings.api.port if settings else 8000,
        log_level="info"
    )