# 🌐 Web UI Demo

The Scrape Store Agents framework now includes a modern web dashboard for easy management and interaction!

## 🚀 Quick Start

```bash
# Start the application
docker-compose up -d

# Access the web dashboard
open http://localhost:8000
```

## 📊 Dashboard Features

### **Main Dashboard**
- **Real-time statistics** - Document count, system health, uptime
- **Quick search** - Search your scraped content instantly
- **URL scraping** - Add new URLs to scrape with custom options
- **Source management** - View and manage configured sources
- **Activity log** - See recent scraping and search activity
- **System status** - Monitor vector store health and API status

### **Search Interface**
- **Full-text search** - Search across all scraped content
- **Advanced filtering** - Filter by URL patterns
- **Result scoring** - See relevance scores for each result
- **Content preview** - View snippets of matching content
- **Real-time results** - Instant search with performance metrics

### **Scraping Management**
- **One-click scraping** - Scrape configured sources instantly
- **Custom URL scraping** - Add any URL with custom settings
- **Link extraction** - Control depth and link following
- **Progress tracking** - See scraping results and document counts

## 🎨 Interface Screenshots

### Main Dashboard
```
┌─────────────────────────────────────────────────────┐
│  🕷️ Scrape Store Agents                Documents: 1,234 │
│                                           Status: Healthy │
├─────────────────────────────────────────────────────┤
│  ⚡ Quick Actions                                    │
│  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ 🔍 Search Docs  │  │ ➕ Scrape New URL       │  │
│  │ [search box...] │  │ [url input...]          │  │
│  │ [Search] button │  │ [Scrape] button         │  │
│  └─────────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────┤
│  📚 Configured Sources                              │
│  ✅ Python Docs     [Scrape]                       │
│  ✅ FastAPI Docs    [Scrape]                       │
│  ❌ News Site       [Enable]                       │
└─────────────────────────────────────────────────────┘
```

### Search Results
```
┌─────────────────────────────────────────────────────┐
│  🔍 Search Results for "async functions"            │
│  Found 15 results in 23.4ms                        │
├─────────────────────────────────────────────────────┤
│  📄 Score: 0.95                                     │
│      Python Async Documentation                     │
│      docs.python.org/3/library/asyncio.html        │
│      Async functions are defined using async def... │
│                                                     │
│  📄 Score: 0.87                                     │
│      FastAPI Async Guide                            │
│      fastapi.tiangolo.com/async/                    │
│      You can define path operations as async...     │
└─────────────────────────────────────────────────────┘
```

## 🎯 Key Benefits

### **For Developers**
- **No CLI required** - Manage everything through the web interface
- **Visual feedback** - See scraping progress and results immediately
- **Easy experimentation** - Test different URLs and settings quickly
- **System monitoring** - Check health and performance at a glance

### **For Teams**
- **Shared interface** - Everyone can use the same dashboard
- **No installation** - Just visit the URL in a browser
- **Real-time collaboration** - See activity from all team members
- **Documentation friendly** - Easy to demonstrate and share

### **For Production**
- **Monitoring dashboard** - Keep track of system health
- **Quick troubleshooting** - Identify issues through the UI
- **Operational control** - Start/stop scraping without CLI access
- **User-friendly** - Non-technical users can search content

## 🔧 Technical Details

### **Built With**
- **Frontend**: Modern HTML5, CSS3, Vanilla JavaScript
- **Backend**: FastAPI with Jinja2 templates
- **Styling**: Custom CSS with responsive design
- **Icons**: Font Awesome for professional appearance
- **API Integration**: RESTful endpoints for all functionality

### **Performance**
- **Lightweight** - No heavy JavaScript frameworks
- **Fast loading** - Optimized CSS and minimal dependencies  
- **Responsive** - Works on desktop, tablet, and mobile
- **Real-time** - Live updates without page refreshes

### **Architecture**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │───▶│   FastAPI App    │───▶│  Scrape Engine  │
│   (Dashboard)   │    │  (API + Static)  │    │ (Agents/Stores) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
    User Interface         Web Routes              Core Business Logic
```

## 🚀 Access Points

After starting the application, you can access:

- **Web Dashboard**: http://localhost:8000/
- **Alternative URL**: http://localhost:8000/dashboard
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🎨 Customization

The web interface is fully customizable:

### **Styling**
```css
/* Edit: scrape_store_agents/web/static/css/dashboard.css */
:root {
    --primary-color: #2563eb;    /* Change primary color */
    --success-color: #059669;    /* Change success color */
    /* ... customize all colors and styles */
}
```

### **Templates**
```html
<!-- Edit: scrape_store_agents/web/templates/dashboard.html -->
<!-- Add new sections, modify layout, etc. -->
```

### **Functionality**
```javascript
// Edit: scrape_store_agents/web/static/js/dashboard.js
// Add new features, modify behavior, etc.
```

## 🔄 Development

To modify the web interface during development:

```bash
# Files are mounted as volumes in Docker
# Changes to CSS/JS/HTML are reflected immediately
# No rebuild required for frontend changes

# For backend route changes:
docker-compose restart scrape-store-agents
```

## 🎯 Future Enhancements

The web interface foundation supports easy addition of:
- **User authentication** - Login/logout functionality
- **Advanced search** - More filtering and search options
- **Real-time updates** - WebSocket integration for live updates
- **Batch operations** - Multi-URL scraping, bulk management
- **Analytics dashboard** - Usage statistics and insights
- **Configuration editor** - Edit sources.yaml through the UI
- **Dark mode** - Theme switching
- **Export functionality** - Download search results, configurations

---

**The web interface makes Scrape Store Agents accessible to everyone - from developers to non-technical users!** 🎉