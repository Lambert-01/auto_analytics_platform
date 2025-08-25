/**
 * App Core - Main application logic with real backend connections
 * This file handles WebSocket connections, data management, and core functionality
 */

class NISRAnalyticsApp {
    constructor() {
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.currentSection = 'dashboard';
        this.dataCache = new Map();
        this.updateTimers = new Map();
    }

    // Initialize the application
    async init() {
        try {
            console.log('Initializing NISR Analytics Platform...');
            
            // Initialize WebSocket connection
            this.initWebSocket();
            
            // Load initial data
            await this.loadInitialData();
            
            // Setup periodic updates
            this.setupPeriodicUpdates();
            
            // Setup event listeners
            this.setupEventListeners();
            
            console.log('NISR Analytics Platform initialized successfully');
            
        } catch (error) {
            console.error('Error initializing app:', error);
            window.showToast('Application initialization failed', 'error');
        }
    }

    // Initialize WebSocket connection
    initWebSocket() {
        try {
            const wsUrl = window.APP_CONFIG.WS_URL;
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.sendWebSocketMessage('client_connected', { 
                    client_type: 'web_dashboard',
                    timestamp: new Date().toISOString()
                });
            };
            
            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.handleWebSocketReconnect();
            };
            
        } catch (error) {
            console.error('WebSocket initialization failed:', error);
        }
    }

    // Handle WebSocket reconnection
    handleWebSocketReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting WebSocket reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            
            setTimeout(() => {
                this.initWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.error('Max WebSocket reconnection attempts reached');
            window.showToast('Connection lost. Please refresh the page.', 'error');
        }
    }

    // Send WebSocket message
    sendWebSocketMessage(type, data) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: type,
                data: data,
                timestamp: new Date().toISOString()
            }));
        }
    }

    // Handle WebSocket messages
    handleWebSocketMessage(event) {
        try {
            const message = JSON.parse(event.data);
            
            switch (message.type) {
                case 'status_update':
                    this.handleStatusUpdate(message.data);
                    break;
                    
                case 'notification':
                    this.handleNotification(message.data);
                    break;
                    
                case 'data_update':
                    this.handleDataUpdate(message.data);
                    break;
                    
                case 'analysis_complete':
                    this.handleAnalysisComplete(message.data);
                    break;
                    
                case 'model_training_update':
                    this.handleModelTrainingUpdate(message.data);
                    break;
                    
                default:
                    console.log('Unhandled WebSocket message:', message);
            }
            
        } catch (error) {
            console.error('Error handling WebSocket message:', error);
        }
    }

    // Handle status updates
    handleStatusUpdate(data) {
        // Update UI elements based on status
        if (data.component === 'dashboard') {
            this.updateDashboardMetrics(data.metrics);
        } else if (data.component === 'dataset') {
            this.updateDatasetStatus(data.dataset_id, data.status);
        }
    }

    // Handle notifications
    handleNotification(data) {
        // Show notification
        window.showToast(data.message, data.type || 'info');
        
        // Update notification counter
        const notificationCount = document.getElementById('notification-count');
        if (notificationCount) {
            const currentCount = parseInt(notificationCount.textContent) || 0;
            notificationCount.textContent = currentCount + 1;
        }
        
        // Add to notifications dropdown
        this.addNotificationToDropdown(data);
    }

    // Handle data updates
    handleDataUpdate(data) {
        // Invalidate cache and reload relevant sections
        this.dataCache.delete(data.type);
        
        if (this.currentSection === data.type) {
            this.refreshCurrentSection();
        }
    }

    // Handle analysis completion
    handleAnalysisComplete(data) {
        window.showToast(`Analysis completed for dataset: ${data.dataset_name}`, 'success');
        
        // Refresh analyses list if on analysis page
        if (this.currentSection === 'analysis') {
            this.loadAnalyses();
        }
    }

    // Handle model training updates
    handleModelTrainingUpdate(data) {
        if (data.status === 'completed') {
            window.showToast(`Model training completed: ${data.model_name}`, 'success');
        } else if (data.status === 'failed') {
            window.showToast(`Model training failed: ${data.model_name}`, 'error');
        }
        
        // Update progress if on AutoML page
        if (this.currentSection === 'automl') {
            this.updateModelTrainingProgress(data);
        }
    }

    // Load initial application data
    async loadInitialData() {
        try {
            // Load dashboard stats
            await this.loadDashboardStats();
            
            // Load user info
            await this.loadUserInfo();
            
            // Load notifications
            await this.loadNotifications();
            
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }

    // Load dashboard statistics
    async loadDashboardStats() {
        try {
            const response = await fetch('/api/v1/dashboard/stats');
            if (response.ok) {
                const stats = await response.json();
                this.dataCache.set('dashboard_stats', stats);
                window.APP_STATE.dashboardStats = stats;
            }
        } catch (error) {
            console.error('Error loading dashboard stats:', error);
        }
    }

    // Load user information
    async loadUserInfo() {
        try {
            const response = await fetch('/api/v1/user/profile');
            if (response.ok) {
                const user = await response.json();
                window.APP_CONFIG.CURRENT_USER = user;
            }
        } catch (error) {
            console.error('Error loading user info:', error);
        }
    }

    // Load notifications
    async loadNotifications() {
        try {
            const response = await fetch('/api/v1/notifications');
            if (response.ok) {
                const notifications = await response.json();
                this.updateNotificationsDropdown(notifications);
                
                // Update notification count
                const notificationCount = document.getElementById('notification-count');
                if (notificationCount) {
                    notificationCount.textContent = notifications.filter(n => !n.read).length;
                }
            }
        } catch (error) {
            console.error('Error loading notifications:', error);
        }
    }

    // Setup periodic updates
    setupPeriodicUpdates() {
        // Update dashboard every 30 seconds
        this.updateTimers.set('dashboard', setInterval(() => {
            if (this.currentSection === 'dashboard') {
                this.loadDashboardStats();
            }
        }, 30000));
        
        // Update notifications every 60 seconds
        this.updateTimers.set('notifications', setInterval(() => {
            this.loadNotifications();
        }, 60000));
    }

    // Setup event listeners
    setupEventListeners() {
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseUpdates();
            } else {
                this.resumeUpdates();
            }
        });
        
        // Handle beforeunload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    // Pause periodic updates when page is hidden
    pauseUpdates() {
        this.updateTimers.forEach((timer) => {
            clearInterval(timer);
        });
    }

    // Resume periodic updates when page becomes visible
    resumeUpdates() {
        this.setupPeriodicUpdates();
    }

    // Update dashboard metrics
    updateDashboardMetrics(metrics) {
        Object.entries(metrics).forEach(([key, value]) => {
            const element = document.getElementById(key);
            if (element) {
                element.textContent = value;
            }
        });
    }

    // Update dataset status
    updateDatasetStatus(datasetId, status) {
        // Update any dataset elements with this ID
        const statusElements = document.querySelectorAll(`[data-dataset-id="${datasetId}"] .dataset-status`);
        statusElements.forEach(element => {
            element.textContent = status;
            element.className = `badge bg-${this.getStatusColor(status)}`;
        });
    }

    // Get status color for badges
    getStatusColor(status) {
        switch(status) {
            case 'completed': return 'success';
            case 'processing': return 'warning';
            case 'failed': return 'danger';
            default: return 'secondary';
        }
    }

    // Add notification to dropdown
    addNotificationToDropdown(notification) {
        const dropdown = document.getElementById('notifications-dropdown');
        if (!dropdown) return;
        
        const notificationItem = document.createElement('li');
        notificationItem.innerHTML = `
            <div class="dropdown-item-text">
                <div class="d-flex justify-content-between">
                    <small class="fw-bold">${notification.title || 'Notification'}</small>
                    <small class="text-muted">${new Date().toLocaleTimeString()}</small>
                </div>
                <div class="mt-1">${notification.message}</div>
            </div>
        `;
        
        // Insert after header and divider
        const divider = dropdown.querySelector('.dropdown-divider');
        if (divider) {
            divider.parentNode.insertBefore(notificationItem, divider.nextSibling);
        }
    }

    // Update notifications dropdown
    updateNotificationsDropdown(notifications) {
        const dropdown = document.getElementById('notifications-dropdown');
        if (!dropdown) return;
        
        // Clear existing notifications (keep header and divider)
        const items = dropdown.querySelectorAll('li:not(:has(.dropdown-header)):not(:has(.dropdown-divider))');
        items.forEach(item => item.remove());
        
        if (notifications.length === 0) {
            const noNotifications = document.createElement('li');
            noNotifications.innerHTML = '<span class="dropdown-item-text">No new notifications</span>';
            dropdown.appendChild(noNotifications);
            return;
        }
        
        notifications.slice(0, 5).forEach(notification => {
            this.addNotificationToDropdown(notification);
        });
    }

    // Refresh current section
    refreshCurrentSection() {
        if (typeof loadSection === 'function') {
            loadSection(this.currentSection);
        }
    }

    // Cleanup resources
    cleanup() {
        // Clear timers
        this.updateTimers.forEach((timer) => {
            clearInterval(timer);
        });
        
        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
        }
        
        // Clear cache
        this.dataCache.clear();
    }

    // API Helper Methods
    async makeRequest(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const mergedOptions = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, mergedOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Get cached data or fetch from API
    async getCachedData(key, fetchFn, ttl = 300000) { // 5 minutes default TTL
        const cached = this.dataCache.get(key);
        
        if (cached && (Date.now() - cached.timestamp) < ttl) {
            return cached.data;
        }
        
        try {
            const data = await fetchFn();
            this.dataCache.set(key, {
                data: data,
                timestamp: Date.now()
            });
            return data;
        } catch (error) {
            // Return cached data if available, even if expired
            if (cached) {
                console.warn('Using expired cache due to fetch error:', error);
                return cached.data;
            }
            throw error;
        }
    }
}

// Global app instance
window.NISRApp = new NISRAnalyticsApp();

// Global helper functions for backward compatibility
window.initializeWebSocket = function() {
    // Already handled by NISRApp.init()
};

window.loadInitialData = function() {
    return window.NISRApp.loadInitialData();
};

// Initialize app when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.NISRApp.init();
    });
} else {
    // DOM already loaded
    window.NISRApp.init();
}
