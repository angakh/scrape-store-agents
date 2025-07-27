// Scrape Store Agents Dashboard JavaScript

class Dashboard {
    constructor() {
        this.apiBase = '';
        this.init();
    }

    async init() {
        await this.loadStats();
        await this.loadSources();
        await this.loadAIConfig();
        await this.loadAIStats();
        this.setupEventListeners();
        this.startPeriodicUpdates();
        this.addActivity('Dashboard initialized');
    }

    setupEventListeners() {
        // Enter key support for search
        document.getElementById('search-query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });

        // Enter key support for scrape URL
        document.getElementById('scrape-url').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.scrapeUrl();
            }
        });
    }

    startPeriodicUpdates() {
        // Update stats every 30 seconds
        setInterval(() => this.loadStats(), 30000);
        
        // Update sources every 60 seconds
        setInterval(() => this.loadSources(), 60000);
        
        // Update AI stats every 45 seconds
        setInterval(() => this.loadAIStats(), 45000);
    }

    async loadStats() {
        try {
            const response = await fetch(`${this.apiBase}/stats`);
            const data = await response.json();
            
            document.getElementById('doc-count').textContent = data.document_count;
            document.getElementById('uptime').textContent = this.formatUptime(data.api_uptime_seconds);
            
            // Update health status
            const healthStatus = document.getElementById('health-status');
            const vectorStoreStatus = document.getElementById('vector-store-status');
            
            if (data.vector_store_healthy) {
                healthStatus.textContent = 'Healthy';
                healthStatus.className = 'stat-value';
                vectorStoreStatus.textContent = 'Healthy';
                vectorStoreStatus.className = 'status-indicator status-healthy';
            } else {
                healthStatus.textContent = 'Unhealthy';
                healthStatus.className = 'stat-value';
                vectorStoreStatus.textContent = 'Unhealthy';
                vectorStoreStatus.className = 'status-indicator status-unhealthy';
            }
            
        } catch (error) {
            console.error('Error loading stats:', error);
            this.showToast('Failed to load statistics', 'error');
        }
    }

    async loadSources() {
        try {
            const response = await fetch(`${this.apiBase}/sources`);
            const data = await response.json();
            
            const container = document.getElementById('sources-container');
            
            if (data.sources && data.sources.length > 0) {
                container.innerHTML = data.sources.map(source => `
                    <div class="source-card">
                        <div class="source-header">
                            <div class="source-name">${this.escapeHtml(source.name)}</div>
                            <div class="source-status ${source.enabled ? 'status-enabled' : 'status-disabled'}">
                                ${source.enabled ? 'Enabled' : 'Disabled'}
                            </div>
                        </div>
                        <div class="source-url">${this.escapeHtml(source.url)}</div>
                        <div class="source-actions">
                            <button onclick="dashboard.scrapeSource('${source.name}')" 
                                    class="btn btn-primary btn-small"
                                    ${!source.enabled ? 'disabled' : ''}>
                                <i class="fas fa-download"></i> Scrape
                            </button>
                        </div>
                    </div>
                `).join('');
            } else {
                container.innerHTML = `
                    <div class="text-center">
                        <p>No sources configured.</p>
                        <p class="text-sm">Edit <code>config/sources.yaml</code> to add sources.</p>
                    </div>
                `;
            }
            
        } catch (error) {
            console.error('Error loading sources:', error);
            document.getElementById('sources-container').innerHTML = `
                <div class="text-center">
                    <p>Failed to load sources.</p>
                </div>
            `;
        }
    }

    async loadAIConfig() {
        try {
            const response = await fetch(`${this.apiBase}/ai/config`);
            const data = await response.json();
            
            this.aiConfig = data;
            this.updateAIInterface(data);
            
        } catch (error) {
            console.error('Error loading AI config:', error);
            this.aiConfig = { ai_available: false };
            this.updateAIInterface(this.aiConfig);
        }
    }

    updateAIInterface(config) {
        // Update AI status indicator
        const aiStatus = document.getElementById('ai-status');
        const aiProvider = document.getElementById('ai-provider');
        
        if (aiStatus) {
            if (config.ai_available) {
                aiStatus.textContent = 'Available';
                aiStatus.className = 'status-indicator status-healthy';
                if (aiProvider) {
                    aiProvider.textContent = config.provider ? `${config.provider} (${config.model})` : 'Configured';
                }
            } else {
                aiStatus.textContent = 'Not Available';
                aiStatus.className = 'status-indicator status-unhealthy';
                if (aiProvider) {
                    aiProvider.textContent = 'Not configured';
                }
            }
        }

        // Show/hide AI controls
        const aiControls = document.querySelectorAll('.ai-control');
        aiControls.forEach(control => {
            if (config.ai_available) {
                control.style.display = 'block';
            } else {
                control.style.display = 'none';
            }
        });

        // Update AI checkboxes default state
        const useAICheckbox = document.getElementById('use-ai');
        const analyzeFirstCheckbox = document.getElementById('analyze-first');
        
        if (useAICheckbox) {
            useAICheckbox.checked = config.ai_available && config.reasoning_agent_enabled;
            useAICheckbox.disabled = !config.ai_available;
        }
        
        if (analyzeFirstCheckbox) {
            analyzeFirstCheckbox.disabled = !config.ai_available || !config.intelligent_router_enabled;
        }
    }

    async performSearch() {
        const query = document.getElementById('search-query').value.trim();
        if (!query) {
            this.showToast('Please enter a search query', 'error');
            return;
        }

        const limit = document.getElementById('search-limit').value;
        const urlFilter = document.getElementById('url-filter').value.trim();

        this.showLoading(true);

        try {
            const params = new URLSearchParams({
                q: query,
                limit: limit
            });

            if (urlFilter) {
                params.append('url_filter', urlFilter);
            }

            const response = await fetch(`${this.apiBase}/search?${params}`);
            const data = await response.json();

            this.displaySearchResults(data);
            this.addActivity(`Searched for: "${query}" (${data.total_results} results)`);

        } catch (error) {
            console.error('Error performing search:', error);
            this.showToast('Search failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displaySearchResults(data) {
        const section = document.getElementById('search-section');
        const container = document.getElementById('search-results');

        if (data.results && data.results.length > 0) {
            container.innerHTML = `
                <div class="mb-2">
                    <strong>${data.total_results}</strong> results found in ${data.execution_time_ms.toFixed(1)}ms
                </div>
                ${data.results.map((result, index) => `
                    <div class="result-item">
                        <div class="result-header">
                            <div>
                                ${result.title ? `<div class="result-title">${this.escapeHtml(result.title)}</div>` : ''}
                                <div class="result-url">${this.escapeHtml(result.url)}</div>
                            </div>
                            <div class="result-score">Score: ${result.score.toFixed(3)}</div>
                        </div>
                        <div class="result-content">${this.escapeHtml(result.content)}</div>
                    </div>
                `).join('')}
            `;
            section.style.display = 'block';
        } else {
            container.innerHTML = `
                <div class="text-center">
                    <p>No results found for "${this.escapeHtml(data.query)}"</p>
                </div>
            `;
            section.style.display = 'block';
        }
    }

    async scrapeUrl() {
        const url = document.getElementById('scrape-url').value.trim();
        if (!url) {
            this.showToast('Please enter a URL to scrape', 'error');
            return;
        }

        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            this.showToast('Please enter a valid URL (http:// or https://)', 'error');
            return;
        }

        const extractLinks = document.getElementById('extract-links').checked;
        const maxDepth = parseInt(document.getElementById('max-depth').value);
        
        // AI options
        const useAI = document.getElementById('use-ai') ? document.getElementById('use-ai').checked : false;
        const analyzeFirst = document.getElementById('analyze-first') ? document.getElementById('analyze-first').checked : false;
        const selfImproving = document.getElementById('self-improving') ? document.getElementById('self-improving').checked : false;

        this.showLoading(true);

        try {
            const requestBody = {
                url: url,
                scraper_config: {
                    extract_links: extractLinks,
                    max_depth: maxDepth
                }
            };

            // Add AI options if available
            if (this.aiConfig && this.aiConfig.ai_available) {
                requestBody.use_ai = useAI;
                requestBody.analyze_first = analyzeFirst;
                requestBody.self_improving = selfImproving;
            }

            const response = await fetch(`${this.apiBase}/scrape`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();

            if (data.success) {
                let message = `Successfully scraped ${data.documents_added} documents from ${url}`;
                if (data.ai_used) {
                    message += ` (using ${data.scraper_used})`;
                }
                
                this.showToast(message, 'success');
                
                let activityMessage = `Scraped ${data.documents_added} documents from ${this.truncateUrl(url)}`;
                if (data.ai_used) {
                    activityMessage += ` using AI`;
                }
                this.addActivity(activityMessage);
                
                // Show AI analysis if available
                if (data.ai_analysis) {
                    this.displayAIAnalysis(data.ai_analysis);
                }
                
                document.getElementById('scrape-url').value = '';
                
                // Refresh stats
                await this.loadStats();
            } else {
                this.showToast(`Scraping failed: ${data.message}`, 'error');
                this.addActivity(`Failed to scrape ${this.truncateUrl(url)}: ${data.message}`);
            }

        } catch (error) {
            console.error('Error scraping URL:', error);
            this.showToast('Scraping failed', 'error');
            this.addActivity(`Error scraping ${this.truncateUrl(url)}`);
        } finally {
            this.showLoading(false);
        }
    }

    async scrapeSource(sourceName) {
        this.showLoading(true);

        try {
            const response = await fetch(`${this.apiBase}/sources/${sourceName}/scrape`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                this.showToast(`Successfully scraped ${data.documents_added} documents from ${sourceName}`, 'success');
                this.addActivity(`Scraped source: ${sourceName} (${data.documents_added} docs)`);
                
                // Refresh stats
                await this.loadStats();
            } else {
                this.showToast(`Scraping ${sourceName} failed: ${data.message}`, 'error');
                this.addActivity(`Failed to scrape source: ${sourceName}`);
            }

        } catch (error) {
            console.error('Error scraping source:', error);
            this.showToast(`Failed to scrape ${sourceName}`, 'error');
            this.addActivity(`Error scraping source: ${sourceName}`);
        } finally {
            this.showLoading(false);
        }
    }

    async refreshSources() {
        await this.loadSources();
        this.showToast('Sources refreshed', 'success');
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = show ? 'flex' : 'none';
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div>${this.escapeHtml(message)}</div>
        `;

        container.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }

    addActivity(text) {
        const container = document.getElementById('activity-log');
        const item = document.createElement('div');
        item.className = 'activity-item';
        item.innerHTML = `
            <span class="activity-time">${this.formatTime(new Date())}</span>
            <span class="activity-text">${this.escapeHtml(text)}</span>
        `;

        // Add to top
        container.insertBefore(item, container.firstChild);

        // Keep only last 10 items
        while (container.children.length > 10) {
            container.removeChild(container.lastChild);
        }
    }

    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else if (minutes > 0) {
            return `${minutes}m`;
        } else {
            return `${Math.floor(seconds)}s`;
        }
    }

    formatTime(date) {
        return date.toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }

    truncateUrl(url) {
        if (url.length > 40) {
            return url.substring(0, 37) + '...';
        }
        return url;
    }

    displayAIAnalysis(analysis) {
        const analysisSection = document.getElementById('ai-analysis-section');
        const analysisContainer = document.getElementById('ai-analysis-results');
        
        if (!analysisSection || !analysisContainer) {
            console.warn('AI analysis display elements not found');
            return;
        }

        let analysisHtml = `
            <div class="ai-analysis-result">
                <h4>🧠 AI Analysis Results</h4>
                <div class="analysis-grid">
                    <div class="analysis-item">
                        <strong>Site Type:</strong> ${this.escapeHtml(analysis.site_type)}
                    </div>
                    <div class="analysis-item">
                        <strong>Confidence:</strong> ${(analysis.confidence * 100).toFixed(1)}%
                    </div>
        `;

        if (analysis.complexity) {
            analysisHtml += `
                    <div class="analysis-item">
                        <strong>Complexity:</strong> ${this.escapeHtml(analysis.complexity)}
                    </div>
            `;
        }

        if (analysis.recommended_approach) {
            analysisHtml += `
                    <div class="analysis-item">
                        <strong>Approach:</strong> ${this.escapeHtml(analysis.recommended_approach)}
                    </div>
            `;
        }

        if (analysis.scraper_recommendation) {
            analysisHtml += `
                    <div class="analysis-item">
                        <strong>Recommended Scraper:</strong> ${this.escapeHtml(analysis.scraper_recommendation.type)}
                    </div>
                    <div class="analysis-item">
                        <strong>Rationale:</strong> ${this.escapeHtml(analysis.scraper_recommendation.rationale)}
                    </div>
            `;
        }

        analysisHtml += `
                </div>
            </div>
        `;

        analysisContainer.innerHTML = analysisHtml;
        analysisSection.style.display = 'block';
    }

    async analyzeUrl() {
        const url = document.getElementById('scrape-url').value.trim();
        if (!url) {
            this.showToast('Please enter a URL to analyze', 'error');
            return;
        }

        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            this.showToast('Please enter a valid URL (http:// or https://)', 'error');
            return;
        }

        if (!this.aiConfig || !this.aiConfig.ai_available) {
            this.showToast('AI features not available', 'error');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch(`${this.apiBase}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: url,
                    detailed: true
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Combine site analysis and scraper recommendation
                const combinedAnalysis = {
                    ...data.site_analysis,
                    scraper_recommendation: data.scraper_recommendation
                };
                
                this.displayAIAnalysis(combinedAnalysis);
                this.addActivity(`Analyzed ${this.truncateUrl(url)} with AI`);
                this.showToast('URL analysis completed', 'success');
            } else {
                this.showToast(`Analysis failed: ${data.detail}`, 'error');
            }

        } catch (error) {
            console.error('Error analyzing URL:', error);
            this.showToast('Analysis failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async loadAIStats() {
        if (!this.aiConfig || !this.aiConfig.ai_available) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/ai/stats`);
            const data = await response.json();
            
            // Update AI stats in the UI
            const aiStatsContainer = document.getElementById('ai-stats-container');
            if (aiStatsContainer && data) {
                let statsHtml = '<h4>🤖 AI Statistics</h4>';
                
                if (data.reasoning_agent) {
                    statsHtml += `
                        <div class="stat-item">
                            <span class="stat-label">URL Analyses:</span>
                            <span class="stat-value">${data.reasoning_agent.url_analyses || 0}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Strategy Adaptations:</span>
                            <span class="stat-value">${data.reasoning_agent.strategy_adaptations || 0}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Memory Entries:</span>
                            <span class="stat-value">${data.reasoning_agent.memory_entries || 0}</span>
                        </div>
                    `;
                }
                
                if (data.intelligent_router) {
                    statsHtml += `
                        <div class="stat-item">
                            <span class="stat-label">Router Selections:</span>
                            <span class="stat-value">${data.intelligent_router.total_selections || 0}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Cached Analyses:</span>
                            <span class="stat-value">${data.intelligent_router.cached_analyses || 0}</span>
                        </div>
                    `;
                }
                
                if (data.self_improving_agent) {
                    statsHtml += `
                        <h4>🧠 Self-Improving Agent</h4>
                        <div class="stat-item">
                            <span class="stat-label">Total Extractions:</span>
                            <span class="stat-value">${data.self_improving_agent.total_extractions || 0}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Learned Strategies:</span>
                            <span class="stat-value">${data.self_improving_agent.learned_strategies || 0}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Average Quality:</span>
                            <span class="stat-value">${(data.self_improving_agent.average_quality || 0).toFixed(2)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Site Patterns:</span>
                            <span class="stat-value">${data.self_improving_agent.site_patterns_learned || 0}</span>
                        </div>
                    `;
                }
                
                aiStatsContainer.innerHTML = statsHtml;
            }
            
        } catch (error) {
            console.error('Error loading AI stats:', error);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global functions for button clicks
let dashboard;

function performSearch() {
    dashboard.performSearch();
}

function scrapeUrl() {
    dashboard.scrapeUrl();
}

function analyzeUrl() {
    dashboard.analyzeUrl();
}

function refreshSources() {
    dashboard.refreshSources();
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new Dashboard();
});