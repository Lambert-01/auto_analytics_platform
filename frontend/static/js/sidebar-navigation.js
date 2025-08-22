/**
 * Sidebar Navigation System for NISR Analytics Platform
 * Handles sidebar interactions, content switching, and responsive behavior
 */

class SidebarNavigation {
    constructor() {
        this.currentSection = 'dashboard';
        this.isMobile = window.innerWidth <= 1024;
        this.isCollapsed = false;
        this.contentSections = new Map();
        this.init();
    }

    init() {
        console.log('Initializing Sidebar Navigation...');
        this.setupEventListeners();
        this.setupContentSections();
        this.updateActiveSection('dashboard');
        this.handleResize();
    }

    setupEventListeners() {
        // Navigation link clicks
        document.addEventListener('click', (e) => {
            if (e.target.matches('.nav-link, .nav-link *')) {
                const link = e.target.closest('.nav-link');
                const section = link.dataset.section;
                if (section) {
                    e.preventDefault();
                    this.switchSection(section);
                }
            }
        });

        // Sidebar toggle
        document.addEventListener('click', (e) => {
            if (e.target.matches('.sidebar-toggle-btn, .sidebar-toggle-btn *')) {
                this.toggleSidebar();
            }
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // Close sidebar on overlay click
        document.addEventListener('click', (e) => {
            if (e.target.matches('.sidebar-overlay')) {
                this.closeSidebar();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'b':
                        e.preventDefault();
                        this.toggleSidebar();
                        break;
                    case '1':
                        e.preventDefault();
                        this.switchSection('dashboard');
                        break;
                    case '2':
                        e.preventDefault();
                        this.switchSection('data-upload');
                        break;
                    case '3':
                        e.preventDefault();
                        this.switchSection('analysis');
                        break;
                }
            }
        });
    }

    setupContentSections() {
        // Define content for each section
        this.contentSections.set('dashboard', {
            title: 'Dashboard',
            subtitle: 'Overview of your analytics platform',
            content: this.getDashboardContent()
        });

        this.contentSections.set('data-upload', {
            title: 'Data Upload',
            subtitle: 'Upload and manage your datasets',
            content: this.getDataUploadContent()
        });

        this.contentSections.set('datasets', {
            title: 'Datasets',
            subtitle: 'Browse and manage your data collections',
            content: this.getDatasetsContent()
        });

        this.contentSections.set('analysis', {
            title: 'Analysis',
            subtitle: 'Perform statistical analysis on your data',
            content: this.getAnalysisContent()
        });

        this.contentSections.set('ai-lab', {
            title: 'AI Lab',
            subtitle: 'Advanced machine learning and AI tools',
            content: this.getAILabContent()
        });

        this.contentSections.set('automl', {
            title: 'AutoML',
            subtitle: 'Automated machine learning model training',
            content: this.getAutoMLContent()
        });

        this.contentSections.set('predictions', {
            title: 'Predictions',
            subtitle: 'Generate forecasts and predictions',
            content: this.getPredictionsContent()
        });

        this.contentSections.set('visualizations', {
            title: 'Visualization Studio',
            subtitle: 'Create stunning data visualizations',
            content: this.getVisualizationsContent()
        });

        this.contentSections.set('dashboards', {
            title: 'Dashboards',
            subtitle: 'Custom interactive dashboards',
            content: this.getDashboardsContent()
        });

        this.contentSections.set('reports', {
            title: 'Reports',
            subtitle: 'Generate comprehensive analytical reports',
            content: this.getReportsContent()
        });

        this.contentSections.set('real-time', {
            title: 'Real-time Monitor',
            subtitle: 'Live data monitoring and alerts',
            content: this.getRealTimeContent()
        });

        this.contentSections.set('alerts', {
            title: 'Alerts',
            subtitle: 'Manage notifications and alerts',
            content: this.getAlertsContent()
        });

        this.contentSections.set('census-data', {
            title: 'Census Data',
            subtitle: 'Rwanda population and demographic analytics',
            content: this.getCensusDataContent()
        });

        this.contentSections.set('economic-indicators', {
            title: 'Economic Indicators',
            subtitle: 'Rwanda economic performance metrics',
            content: this.getEconomicIndicatorsContent()
        });

        this.contentSections.set('survey-analysis', {
            title: 'Survey Analysis',
            subtitle: 'Analyze national and household surveys',
            content: this.getSurveyAnalysisContent()
        });

        this.contentSections.set('geo-analysis', {
            title: 'Geo Analysis',
            subtitle: 'Geographic and spatial data analysis',
            content: this.getGeoAnalysisContent()
        });
    }

    switchSection(sectionId) {
        if (sectionId === this.currentSection) return;

        // Update active navigation
        this.updateActiveSection(sectionId);
        
        // Update content
        this.updateContent(sectionId);
        
        // Update breadcrumb
        this.updateBreadcrumb(sectionId);
        
        // Close sidebar on mobile
        if (this.isMobile) {
            this.closeSidebar();
        }

        this.currentSection = sectionId;
        
        // Trigger section change event
        this.triggerSectionChange(sectionId);
    }

    updateActiveSection(sectionId) {
        // Remove active class from all nav links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });

        // Add active class to current section
        const activeLink = document.querySelector(`[data-section="${sectionId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }

    updateContent(sectionId) {
        const contentArea = document.getElementById('content-area');
        const sectionData = this.contentSections.get(sectionId);
        
        if (!sectionData) {
            console.warn(`No content defined for section: ${sectionId}`);
            return;
        }

        // Create content with fade effect
        contentArea.style.opacity = '0';
        contentArea.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            contentArea.innerHTML = `
                <div class="content-section active" id="${sectionId}-section">
                    <div class="section-header">
                        <h1>${sectionData.title}</h1>
                        <p>${sectionData.subtitle}</p>
                    </div>
                    <div class="section-content">
                        ${sectionData.content}
                    </div>
                </div>
            `;
            
            // Fade in new content
            contentArea.style.transition = 'all 0.3s ease-out';
            contentArea.style.opacity = '1';
            contentArea.style.transform = 'translateY(0)';
            
            // Initialize section-specific functionality
            this.initializeSectionFeatures(sectionId);
        }, 150);
    }

    updateBreadcrumb(sectionId) {
        const breadcrumbItem = document.getElementById('current-section');
        const sectionData = this.contentSections.get(sectionId);
        
        if (breadcrumbItem && sectionData) {
            breadcrumbItem.textContent = sectionData.title;
        }
    }

    toggleSidebar() {
        if (this.isMobile) {
            this.toggleMobileSidebar();
        } else {
            this.toggleDesktopSidebar();
        }
    }

    toggleMobileSidebar() {
        const sidebar = document.getElementById('sidebar');
        const overlay = this.getOrCreateOverlay();
        
        sidebar.classList.toggle('show');
        overlay.classList.toggle('show');
    }

    toggleDesktopSidebar() {
        const sidebar = document.getElementById('sidebar');
        this.isCollapsed = !this.isCollapsed;
        sidebar.classList.toggle('collapsed', this.isCollapsed);
    }

    closeSidebar() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.querySelector('.sidebar-overlay');
        
        sidebar.classList.remove('show');
        if (overlay) {
            overlay.classList.remove('show');
        }
    }

    getOrCreateOverlay() {
        let overlay = document.querySelector('.sidebar-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'sidebar-overlay';
            document.body.appendChild(overlay);
        }
        return overlay;
    }

    handleResize() {
        const wasMobile = this.isMobile;
        this.isMobile = window.innerWidth <= 1024;
        
        if (wasMobile !== this.isMobile) {
            const sidebar = document.getElementById('sidebar');
            const overlay = document.querySelector('.sidebar-overlay');
            
            if (this.isMobile) {
                sidebar.classList.remove('collapsed');
                sidebar.classList.remove('show');
                this.isCollapsed = false;
            } else {
                sidebar.classList.remove('show');
                if (overlay) {
                    overlay.classList.remove('show');
                }
            }
        }
    }

    triggerSectionChange(sectionId) {
        const event = new CustomEvent('sectionChange', {
            detail: { section: sectionId }
        });
        document.dispatchEvent(event);
    }

    initializeSectionFeatures(sectionId) {
        // Initialize section-specific features
        switch (sectionId) {
            case 'dashboard':
                this.initializeDashboard();
                break;
            case 'data-upload':
                this.initializeDataUpload();
                break;
            case 'real-time':
                this.initializeRealTime();
                break;
            case 'visualizations':
                this.initializeVisualizations();
                break;
            // Add more cases as needed
        }
    }

    initializeDashboard() {
        // Initialize dashboard charts and widgets
        if (window.initRealTimeDashboard) {
            window.initRealTimeDashboard();
        }
    }

    initializeDataUpload() {
        // Initialize drag and drop functionality
        const dropZone = document.querySelector('.drop-zone');
        if (dropZone) {
            this.setupFileUpload(dropZone);
        }
    }

    initializeRealTime() {
        // Initialize real-time features
        if (window.realTimeSystem) {
            window.realTimeSystem.initializeRealTimeCharts();
        }
    }

    initializeVisualizations() {
        // Initialize visualization tools
        if (window.vizSystem) {
            // Any specific initialization needed
        }
    }

    setupFileUpload(dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            this.handleFileUpload(files);
        });
    }

    handleFileUpload(files) {
        Array.from(files).forEach(file => {
            console.log('Uploading file:', file.name);
            // Implement file upload logic
        });
    }

    // Content generation methods for each section
    getDashboardContent() {
        return `
            <div class="dashboard-grid">
                <div class="dashboard-card">
                    <div class="card-header">
                        <h3>Quick Stats</h3>
                        <i class="fas fa-chart-line"></i>
                    </div>
                    <div class="card-content">
                        <div class="stat-grid">
                            <div class="stat-item">
                                <div class="stat-value">12,847</div>
                                <div class="stat-label">Datasets Processed</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">3,421</div>
                                <div class="stat-label">Models Trained</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">87,653</div>
                                <div class="stat-label">Insights Generated</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="dashboard-card">
                    <div class="card-header">
                        <h3>Recent Activity</h3>
                        <i class="fas fa-clock"></i>
                    </div>
                    <div class="card-content">
                        <div class="activity-list">
                            <div class="activity-item">
                                <div class="activity-icon">
                                    <i class="fas fa-upload"></i>
                                </div>
                                <div class="activity-content">
                                    <div class="activity-title">Census Data 2022 uploaded</div>
                                    <div class="activity-time">2 hours ago</div>
                                </div>
                            </div>
                            <div class="activity-item">
                                <div class="activity-icon">
                                    <i class="fas fa-brain"></i>
                                </div>
                                <div class="activity-content">
                                    <div class="activity-title">AutoML model training completed</div>
                                    <div class="activity-time">4 hours ago</div>
                                </div>
                            </div>
                            <div class="activity-item">
                                <div class="activity-icon">
                                    <i class="fas fa-chart-area"></i>
                                </div>
                                <div class="activity-content">
                                    <div class="activity-title">Economic indicators report generated</div>
                                    <div class="activity-time">6 hours ago</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="dashboard-card chart-card">
                    <div class="card-header">
                        <h3>Performance Overview</h3>
                        <i class="fas fa-tachometer-alt"></i>
                    </div>
                    <div class="card-content">
                        <canvas id="performance-chart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

    getDataUploadContent() {
        return `
            <div class="upload-section">
                <div class="upload-card">
                    <div class="drop-zone" id="drop-zone">
                        <div class="drop-zone-content">
                            <div class="drop-icon">
                                <i class="fas fa-cloud-upload-alt"></i>
                            </div>
                            <h3>Drag & Drop Your Data Files</h3>
                            <p>Support for CSV, Excel, JSON, Parquet files</p>
                            <button class="btn-primary" onclick="document.getElementById('file-input').click()">
                                <i class="fas fa-folder-open"></i>
                                Browse Files
                            </button>
                            <input type="file" id="file-input" multiple accept=".csv,.xlsx,.json,.parquet" style="display: none;">
                        </div>
                    </div>
                </div>
                
                <div class="upload-options">
                    <h3>Data Sources</h3>
                    <div class="source-grid">
                        <div class="source-card">
                            <i class="fas fa-database"></i>
                            <h4>Database Connection</h4>
                            <p>Connect to PostgreSQL, MySQL, MongoDB</p>
                            <button class="btn-secondary">Connect</button>
                        </div>
                        <div class="source-card">
                            <i class="fas fa-link"></i>
                            <h4>API Integration</h4>
                            <p>Import data from REST APIs</p>
                            <button class="btn-secondary">Configure</button>
                        </div>
                        <div class="source-card">
                            <i class="fas fa-cloud"></i>
                            <h4>Cloud Storage</h4>
                            <p>Import from AWS S3, Google Cloud</p>
                            <button class="btn-secondary">Setup</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getDatasetsContent() {
        return `
            <div class="datasets-section">
                <div class="datasets-header">
                    <div class="search-filters">
                        <input type="text" placeholder="Search datasets..." class="search-input">
                        <select class="filter-select">
                            <option>All Types</option>
                            <option>Census Data</option>
                            <option>Economic Data</option>
                            <option>Survey Data</option>
                        </select>
                    </div>
                    <button class="btn-primary">
                        <i class="fas fa-plus"></i>
                        New Dataset
                    </button>
                </div>
                
                <div class="datasets-grid">
                    <div class="dataset-card">
                        <div class="dataset-header">
                            <h4>Rwanda Census 2022</h4>
                            <div class="dataset-status active">Active</div>
                        </div>
                        <div class="dataset-meta">
                            <span><i class="fas fa-table"></i> 2.3M rows</span>
                            <span><i class="fas fa-columns"></i> 45 columns</span>
                            <span><i class="fas fa-calendar"></i> Updated 2 days ago</span>
                        </div>
                        <div class="dataset-actions">
                            <button class="btn-sm">Analyze</button>
                            <button class="btn-sm">Export</button>
                            <button class="btn-sm">Preview</button>
                        </div>
                    </div>
                    
                    <div class="dataset-card">
                        <div class="dataset-header">
                            <h4>Economic Indicators Q3 2024</h4>
                            <div class="dataset-status processing">Processing</div>
                        </div>
                        <div class="dataset-meta">
                            <span><i class="fas fa-table"></i> 156K rows</span>
                            <span><i class="fas fa-columns"></i> 28 columns</span>
                            <span><i class="fas fa-calendar"></i> Updated 1 hour ago</span>
                        </div>
                        <div class="dataset-actions">
                            <button class="btn-sm" disabled>Analyze</button>
                            <button class="btn-sm">Export</button>
                            <button class="btn-sm">Preview</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getAnalysisContent() {
        return `
            <div class="analysis-section">
                <div class="analysis-tools">
                    <h3>Analysis Tools</h3>
                    <div class="tools-grid">
                        <div class="tool-card">
                            <i class="fas fa-chart-bar"></i>
                            <h4>Statistical Analysis</h4>
                            <p>Descriptive statistics, correlation analysis</p>
                            <button class="btn-primary">Start Analysis</button>
                        </div>
                        <div class="tool-card">
                            <i class="fas fa-search-plus"></i>
                            <h4>Exploratory Data Analysis</h4>
                            <p>Automated EDA with insights</p>
                            <button class="btn-primary">Generate EDA</button>
                        </div>
                        <div class="tool-card">
                            <i class="fas fa-microscope"></i>
                            <h4>Hypothesis Testing</h4>
                            <p>Statistical significance testing</p>
                            <button class="btn-primary">Run Tests</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Add more content generation methods for other sections...
    getAILabContent() {
        return `
            <div class="ai-lab-section">
                <h3>AI Laboratory</h3>
                <p>Advanced machine learning and AI capabilities coming soon...</p>
            </div>
        `;
    }

    getAutoMLContent() {
        return `
            <div class="automl-section">
                <h3>AutoML Studio</h3>
                <p>Automated machine learning model training coming soon...</p>
            </div>
        `;
    }

    getPredictionsContent() {
        return `
            <div class="predictions-section">
                <h3>Predictions</h3>
                <p>Generate forecasts and predictions coming soon...</p>
            </div>
        `;
    }

    getVisualizationsContent() {
        return `
            <div class="visualizations-section">
                <h3>Visualization Studio</h3>
                <p>Create stunning data visualizations coming soon...</p>
            </div>
        `;
    }

    getDashboardsContent() {
        return `
            <div class="dashboards-section">
                <h3>Custom Dashboards</h3>
                <p>Interactive dashboard builder coming soon...</p>
            </div>
        `;
    }

    getReportsContent() {
        return `
            <div class="reports-section">
                <h3>Reports</h3>
                <p>Comprehensive analytical reports coming soon...</p>
            </div>
        `;
    }

    getRealTimeContent() {
        return `
            <div class="real-time-section">
                <h3>Real-time Monitor</h3>
                <div class="real-time-dashboard">
                    <div class="dashboard-card">
                        <div class="card-header">
                            <h4>Live Performance</h4>
                            <div class="live-indicator">
                                <span class="live-dot"></span>
                                Live
                            </div>
                        </div>
                        <div class="card-content">
                            <canvas id="performance-chart" width="400" height="200"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getAlertsContent() {
        return `
            <div class="alerts-section">
                <h3>Alerts & Notifications</h3>
                <p>Manage your notifications and alert settings coming soon...</p>
            </div>
        `;
    }

    getCensusDataContent() {
        return `
            <div class="census-data-section">
                <h3>Rwanda Census Data Analytics</h3>
                <p>Comprehensive population and demographic analysis tools coming soon...</p>
            </div>
        `;
    }

    getEconomicIndicatorsContent() {
        return `
            <div class="economic-indicators-section">
                <h3>Economic Indicators</h3>
                <p>Rwanda economic performance metrics and analysis coming soon...</p>
            </div>
        `;
    }

    getSurveyAnalysisContent() {
        return `
            <div class="survey-analysis-section">
                <h3>Survey Analysis</h3>
                <p>National and household survey analysis tools coming soon...</p>
            </div>
        `;
    }

    getGeoAnalysisContent() {
        return `
            <div class="geo-analysis-section">
                <h3>Geographic Analysis</h3>
                <p>Spatial data analysis and mapping tools coming soon...</p>
            </div>
        `;
    }
}

// Global functions for compatibility
function toggleSidebar() {
    if (window.sidebarNav) {
        window.sidebarNav.toggleSidebar();
    }
}

function toggleTheme() {
    document.body.classList.toggle('dark-theme');
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) {
        themeIcon.className = document.body.classList.contains('dark-theme') ? 'fas fa-sun' : 'fas fa-moon';
    }
    localStorage.setItem('theme', document.body.classList.contains('dark-theme') ? 'dark' : 'light');
}

function openNotifications() {
    console.log('Opening notifications...');
    // Implementation for notifications
}

function openHelp() {
    console.log('Opening help...');
    // Implementation for help system
}

function toggleUserMenu() {
    console.log('Toggling user menu...');
    // Implementation for user menu
}

// Initialize sidebar navigation when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.sidebarNav = new SidebarNavigation();
    
    // Load saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        const themeIcon = document.getElementById('theme-icon');
        if (themeIcon) {
            themeIcon.className = 'fas fa-sun';
        }
    }
});

// Export for global access
window.SidebarNavigation = SidebarNavigation;
