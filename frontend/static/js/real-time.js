/**
 * Real-time Dashboard and Live Data System
 * Handles streaming data, live updates, and real-time monitoring
 */

class RealTimeSystem {
    constructor() {
        this.connections = new Map();
        this.charts = new Map();
        this.alerts = [];
        this.metrics = {};
        this.isConnected = false;
        this.init();
    }

    init() {
        console.log('Initializing Real-time System...');
        this.setupWebSocket();
        this.initializeRealTimeCharts();
        this.startMetricsCollection();
        this.setupEventListeners();
    }

    setupWebSocket() {
        try {
            const wsUrl = `ws://${window.location.host}/ws/realtime`;
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                console.log('Real-time connection established');
                this.updateConnectionStatus(true);
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleRealTimeData(data);
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                console.log('Real-time connection closed');
                this.updateConnectionStatus(false);
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.setupWebSocket(), 5000);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
        } catch (error) {
            console.error('WebSocket setup error:', error);
            // Simulate real-time data for demo
            this.startSimulation();
        }
    }

    setupEventListeners() {
        // Start/stop monitoring buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('.start-monitoring')) {
                this.startMonitoring(e.target.dataset.source);
            }
            if (e.target.matches('.stop-monitoring')) {
                this.stopMonitoring(e.target.dataset.source);
            }
        });
    }

    initializeRealTimeCharts() {
        this.initPerformanceChart();
        this.initMetricsChart();
        this.initAlertsPanel();
    }

    initPerformanceChart() {
        const ctx = document.getElementById('performance-chart');
        if (!ctx) {
            console.warn('Performance chart canvas not found');
            return;
        }

        // Clear existing chart if any
        if (this.performanceChart) {
            this.performanceChart.destroy();
        }

        // Ensure canvas context is valid
        const canvas = ctx.getContext ? ctx : ctx.querySelector('canvas');
        if (!canvas) {
            console.warn('Invalid canvas element for performance chart');
            return;
        }

        try {
            this.performanceChart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Performance',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                animation: {
                    duration: 0
                },
                font: {
                    family: "'Inter', 'Helvetica Neue', Arial, sans-serif"
                }
            }
        });
        } catch (error) {
            console.error('Failed to create performance chart:', error);
            this.performanceChart = null;
        }
    }

    initMetricsChart() {
        // Initialize additional metrics charts as needed
    }

    initAlertsPanel() {
        const alertsContainer = document.querySelector('.alerts-panel');
        if (!alertsContainer) {
            const panel = document.createElement('div');
            panel.className = 'alerts-panel';
            panel.innerHTML = `
                <div class="panel-header">
                    <h3>Live Alerts</h3>
                    <div class="alert-status">
                        <span class="status-indicator"></span>
                        <span class="status-text">Monitoring</span>
                    </div>
                </div>
                <div class="alerts-list" id="alerts-list">
                    <!-- Alerts will be populated here -->
                </div>
            `;
            document.body.appendChild(panel);
        }
    }

    handleRealTimeData(data) {
        switch (data.type) {
            case 'metrics':
                this.updateMetrics(data.payload);
                break;
            case 'alert':
                this.showAlert(data.payload);
                break;
            case 'performance':
                this.updatePerformanceChart(data.payload);
                break;
            case 'prediction':
                this.updatePredictions(data.payload);
                break;
        }
    }

    updateMetrics(metrics) {
        this.metrics = { ...this.metrics, ...metrics };
        
        // Update live indicators
        Object.entries(metrics).forEach(([key, value]) => {
            const element = document.querySelector(`[data-metric="${key}"]`);
            if (element) {
                element.textContent = this.formatMetricValue(value);
                element.classList.add('updated');
                setTimeout(() => element.classList.remove('updated'), 1000);
            }
        });
    }

    updatePerformanceChart(data) {
        if (!this.performanceChart || !this.performanceChart.data) {
            console.warn('Performance chart not initialized');
            return;
        }

        const chart = this.performanceChart;
        const now = new Date().toLocaleTimeString();
        
        // Add new data point
        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(data.value);
        
        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update('none');
    }

    showAlert(alert) {
        this.alerts.unshift(alert);
        
        // Show toast notification
        showToast(alert.message, alert.severity);
        
        // Update alerts panel
        this.updateAlertsPanel();
        
        // Keep only last 50 alerts
        if (this.alerts.length > 50) {
            this.alerts = this.alerts.slice(0, 50);
        }
    }

    updateAlertsPanel() {
        const alertsList = document.getElementById('alerts-list');
        if (!alertsList) return;

        alertsList.innerHTML = this.alerts.map(alert => `
            <div class="alert-item ${alert.severity}">
                <div class="alert-icon">
                    <i class="fas fa-${this.getAlertIcon(alert.severity)}"></i>
                </div>
                <div class="alert-content">
                    <div class="alert-message">${alert.message}</div>
                    <div class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</div>
                </div>
                <button class="alert-dismiss" onclick="realTimeSystem.dismissAlert('${alert.id}')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `).join('');
    }

    getAlertIcon(severity) {
        const icons = {
            info: 'info-circle',
            warning: 'exclamation-triangle',
            error: 'exclamation-circle',
            success: 'check-circle'
        };
        return icons[severity] || 'info-circle';
    }

    updatePredictions(predictions) {
        const predictionsContainer = document.querySelector('.predictions-list');
        if (!predictionsContainer) return;

        predictionsContainer.innerHTML = predictions.map(pred => `
            <div class="prediction-item">
                <span class="prediction-label">${pred.label}</span>
                <span class="prediction-value ${pred.trend}">${pred.value}</span>
            </div>
        `).join('');
    }

    updateConnectionStatus(connected) {
        const statusElements = document.querySelectorAll('.connection-status');
        statusElements.forEach(element => {
            element.classList.toggle('connected', connected);
            element.classList.toggle('disconnected', !connected);
            element.textContent = connected ? 'Connected' : 'Disconnected';
        });

        const liveIndicators = document.querySelectorAll('.live-indicator');
        liveIndicators.forEach(indicator => {
            indicator.style.display = connected ? 'flex' : 'none';
        });
    }

    startMonitoring(source) {
        if (this.isConnected && this.ws) {
            this.ws.send(JSON.stringify({
                action: 'start_monitoring',
                source: source
            }));
        }
        showToast(`Started monitoring ${source}`, 'success');
    }

    stopMonitoring(source) {
        if (this.isConnected && this.ws) {
            this.ws.send(JSON.stringify({
                action: 'stop_monitoring',
                source: source
            }));
        }
        showToast(`Stopped monitoring ${source}`, 'info');
    }

    dismissAlert(alertId) {
        this.alerts = this.alerts.filter(alert => alert.id !== alertId);
        this.updateAlertsPanel();
    }

    formatMetricValue(value) {
        if (typeof value === 'number') {
            if (value > 1000000) {
                return (value / 1000000).toFixed(1) + 'M';
            } else if (value > 1000) {
                return (value / 1000).toFixed(1) + 'K';
            }
            return value.toLocaleString();
        }
        return value;
    }

    startMetricsCollection() {
        // Simulate real-time metrics collection
        setInterval(() => {
            this.collectSystemMetrics();
        }, 2000);
    }

    collectSystemMetrics() {
        // Simulate system metrics
        const metrics = {
            cpu_usage: Math.random() * 100,
            memory_usage: Math.random() * 100,
            disk_usage: Math.random() * 100,
            network_io: Math.random() * 1000,
            active_users: Math.floor(Math.random() * 500),
            api_requests: Math.floor(Math.random() * 1000)
        };

        this.updateMetrics(metrics);
        
        // Update performance chart
        this.updatePerformanceChart({
            value: metrics.cpu_usage,
            timestamp: Date.now()
        });
    }

    startSimulation() {
        console.log('Starting real-time simulation...');
        
        // Simulate data streams
        setInterval(() => {
            this.simulateDataStream();
        }, 1000);
        
        // Simulate alerts
        setInterval(() => {
            this.simulateAlert();
        }, 10000);
    }

    simulateDataStream() {
        const data = {
            type: 'metrics',
            payload: {
                sales_rate: Math.floor(Math.random() * 1500) + 500,
                user_activity: Math.floor(Math.random() * 1000) + 200,
                system_load: Math.random() * 100
            }
        };
        
        this.handleRealTimeData(data);
    }

    simulateAlert() {
        const alerts = [
            { message: 'High system load detected', severity: 'warning' },
            { message: 'Sales target achieved!', severity: 'success' },
            { message: 'Database connection restored', severity: 'info' },
            { message: 'Unusual traffic pattern detected', severity: 'warning' }
        ];
        
        const alert = alerts[Math.floor(Math.random() * alerts.length)];
        this.showAlert({
            ...alert,
            id: generateId(),
            timestamp: Date.now()
        });
    }

    // Initialize real-time dashboard
    initRealTimeDashboard() {
        // Update stream rates
        const streamItems = document.querySelectorAll('.stream-item');
        streamItems.forEach(item => {
            const rateElement = item.querySelector('.stream-rate');
            if (rateElement) {
                setInterval(() => {
                    const rate = Math.floor(Math.random() * 2000) + 100;
                    rateElement.textContent = `${rate.toLocaleString()}/sec`;
                }, 2000);
            }
        });

        // Initialize live charts
        this.initializeRealTimeCharts();
    }
}

// Initialize Real-time System
const realTimeSystem = new RealTimeSystem();

// Export for global access
window.realTimeSystem = realTimeSystem;
window.initRealTimeDashboard = () => realTimeSystem.initRealTimeDashboard();
