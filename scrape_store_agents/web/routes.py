"""Web interface routes for scrape-store-agents."""

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


def setup_web_routes(app, templates_dir: str):
    """Setup web interface routes."""
    
    templates = Jinja2Templates(directory=templates_dir)
    
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard(request: Request):
        """Main dashboard page."""
        return templates.TemplateResponse("dashboard.html", {"request": request})
    
    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_alias(request: Request):
        """Dashboard alias."""
        return templates.TemplateResponse("dashboard.html", {"request": request})
    
    return app