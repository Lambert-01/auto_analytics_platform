/**
 * Advanced Features JavaScript for Auto Analytics Platform
 * Handles ML, AI, and advanced analytics functionality
 */

class AdvancedFeatures {
    constructor() {
        this.initialized = false;
        this.models = new Map();
        this.currentAnalysis = null;
        this.realTimeConnections = new Map();
        this.init();
    }

    init() {
        console.log('Initializing Advanced Features...');
        this.setupEventListeners();
        this.initializeModules();
        this.initialized = true;
    }

    setupEventListeners() {
        // Advanced ML Controls
        document.addEventListener('click', this.handleAdvancedActions.bind(this));
        
        // Real-time data handling
        window.addEventListener('beforeunload', this.cleanupConnections.bind(this));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
    }

    handleAdvancedActions(event) {
        const target = event.target.closest('[data-action]');
        if (!target) return;

        const action = target.dataset.action;
        const params = target.dataset.params ? JSON.parse(target.dataset.params) : {};

        switch (action) {
            case 'start-automl':
                this.startAutoML(params);
                break;
            case 'create-neural-network':
                this.createNeuralNetwork(params);
                break;
            case 'run-deep-analysis':
                this.runDeepAnalysis(params);
                break;
            case 'start-real-time-monitoring':
                this.startRealTimeMonitoring(params);
                break;
            case 'generate-predictions':
                this.generatePredictions(params);
                break;
            case 'export-model':
                this.exportModel(params);
                break;
        }
    }

    handleKeyboardShortcuts(event) {
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case 'k':
                    event.preventDefault();
                    this.openCommandPalette();
                    break;
                case 'n':
                    event.preventDefault();
                    this.createNewAnalysis();
                    break;
                case 'r':
                    event.preventDefault();
                    this.refreshCurrentView();
                    break;
            }
        }
    }

    initializeModules() {
        this.neuralNetworkBuilder = new NeuralNetworkBuilder();
        this.timeSeriesForecaster = new TimeSeriesForecaster();
        this.anomalyDetector = new AnomalyDetector();
        this.recommendationEngine = new RecommendationEngine();
        this.textAnalyzer = new TextAnalyzer();
        this.imageProcessor = new ImageProcessor();
    }

    // AutoML Functionality
    async startAutoML(params) {
        try {
            showToast('Starting AutoML training...', 'info');
            showLoading('Preparing AutoML pipeline...');

            const config = {
                dataset_id: params.datasetId,
                target_column: params.targetColumn,
                task_type: params.taskType || 'auto-detect',
                max_training_time: params.maxTime || 1800,
                algorithms: params.algorithms || ['all'],
                optimization_metric: params.metric || 'auto',
                cross_validation_folds: 5,
                enable_feature_engineering: true,
                enable_ensemble: true,
                enable_neural_networks: true
            };

            const response = await apiClient.post('/models/automl', config);
            
            if (response.success) {
                const sessionId = response.data.session_id;
                this.monitorAutoMLProgress(sessionId);
                showToast('AutoML session started successfully!', 'success');
            } else {
                throw new Error(response.error);
            }

        } catch (error) {
            console.error('AutoML Error:', error);
            showToast(`AutoML failed: ${error.message}`, 'error');
        } finally {
            hideLoading();
        }
    }

    async monitorAutoMLProgress(sessionId) {
        const progressContainer = this.createProgressMonitor(sessionId);
        document.body.appendChild(progressContainer);

        const pollProgress = async () => {
            try {
                const response = await apiClient.get(`/models/automl/${sessionId}/status`);
                if (response.success) {
                    this.updateProgressMonitor(sessionId, response.data);
                    
                    if (response.data.status === 'completed') {
                        this.showAutoMLResults(sessionId, response.data);
                        return;
                    } else if (response.data.status === 'failed') {
                        showToast('AutoML training failed', 'error');
                        return;
                    }
                }
                
                setTimeout(pollProgress, 2000); // Poll every 2 seconds
            } catch (error) {
                console.error('Progress monitoring error:', error);
            }
        };

        pollProgress();
    }

    createProgressMonitor(sessionId) {
        const container = document.createElement('div');
        container.id = `automl-progress-${sessionId}`;
        container.className = 'automl-progress-monitor';
        container.innerHTML = `
            <div class="progress-header">
                <h3>AutoML Training Progress</h3>
                <button onclick="this.parentElement.parentElement.remove()" class="close-btn">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="progress-content">
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progress-bar-${sessionId}"></div>
                </div>
                <div class="progress-details" id="progress-details-${sessionId}">
                    <div class="current-step">Initializing...</div>
                    <div class="models-trained">Models trained: 0</div>
                    <div class="best-score">Best score: -</div>
                    <div class="time-elapsed">Time elapsed: 0s</div>
                </div>
                <div class="model-comparison" id="model-comparison-${sessionId}">
                    <!-- Model comparison chart will be inserted here -->
                </div>
            </div>
        `;
        return container;
    }

    updateProgressMonitor(sessionId, data) {
        const progressBar = document.getElementById(`progress-bar-${sessionId}`);
        const progressDetails = document.getElementById(`progress-details-${sessionId}`);
        
        if (progressBar) {
            progressBar.style.width = `${data.progress}%`;
        }
        
        if (progressDetails) {
            progressDetails.innerHTML = `
                <div class="current-step">${data.current_step}</div>
                <div class="models-trained">Models trained: ${data.models_trained || 0}</div>
                <div class="best-score">Best score: ${data.best_score || '-'}</div>
                <div class="time-elapsed">Time elapsed: ${this.formatTime(data.time_elapsed || 0)}</div>
            `;
        }

        // Update model comparison chart
        if (data.model_scores) {
            this.updateModelComparisonChart(sessionId, data.model_scores);
        }
    }

    // Neural Network Builder
    async createNeuralNetwork(params) {
        try {
            showToast('Opening Neural Network Builder...', 'info');
            
            const modal = this.createNeuralNetworkModal(params);
            document.body.appendChild(modal);
            modal.classList.add('active');

        } catch (error) {
            console.error('Neural Network Error:', error);
            showToast(`Neural Network creation failed: ${error.message}`, 'error');
        }
    }

    createNeuralNetworkModal(params) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay neural-network-modal';
        modal.innerHTML = `
            <div class="neural-network-builder">
                <div class="modal-header">
                    <h2>Neural Network Builder</h2>
                    <button onclick="this.closest('.modal-overlay').remove()" class="modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="network-designer">
                    <div class="layers-panel">
                        <h3>Layers</h3>
                        <div class="layer-types">
                            <button class="layer-btn" data-layer="dense">
                                <i class="fas fa-grip-lines"></i>
                                Dense
                            </button>
                            <button class="layer-btn" data-layer="conv2d">
                                <i class="fas fa-th"></i>
                                Conv2D
                            </button>
                            <button class="layer-btn" data-layer="lstm">
                                <i class="fas fa-wave-square"></i>
                                LSTM
                            </button>
                            <button class="layer-btn" data-layer="dropout">
                                <i class="fas fa-random"></i>
                                Dropout
                            </button>
                            <button class="layer-btn" data-layer="attention">
                                <i class="fas fa-eye"></i>
                                Attention
                            </button>
                        </div>
                        <div class="layer-properties">
                            <h4>Layer Properties</h4>
                            <div id="layer-properties-form">
                                <!-- Properties will be dynamically generated -->
                            </div>
                        </div>
                    </div>
                    <div class="network-canvas">
                        <div class="canvas-header">
                            <h3>Network Architecture</h3>
                            <div class="canvas-controls">
                                <button onclick="advancedFeatures.clearNetwork()">Clear</button>
                                <button onclick="advancedFeatures.autoGenerateNetwork()">Auto Generate</button>
                                <button onclick="advancedFeatures.trainNetwork()">Train Network</button>
                            </div>
                        </div>
                        <div id="network-canvas" class="network-visualization">
                            <!-- Network visualization will be rendered here -->
                        </div>
                    </div>
                </div>
                <div class="training-panel">
                    <h3>Training Configuration</h3>
                    <div class="training-config">
                        <div class="config-group">
                            <label>Optimizer:</label>
                            <select id="optimizer-select">
                                <option value="adam">Adam</option>
                                <option value="sgd">SGD</option>
                                <option value="rmsprop">RMSprop</option>
                                <option value="adagrad">Adagrad</option>
                            </select>
                        </div>
                        <div class="config-group">
                            <label>Learning Rate:</label>
                            <input type="number" id="learning-rate" value="0.001" step="0.0001" min="0.0001" max="1">
                        </div>
                        <div class="config-group">
                            <label>Batch Size:</label>
                            <input type="number" id="batch-size" value="32" min="1" max="1024">
                        </div>
                        <div class="config-group">
                            <label>Epochs:</label>
                            <input type="number" id="epochs" value="100" min="1" max="1000">
                        </div>
                    </div>
                </div>
            </div>
        `;
        return modal;
    }

    // Deep Analysis
    async runDeepAnalysis(params) {
        try {
            showToast('Starting deep analysis...', 'info');
            showLoading('Analyzing data with advanced algorithms...');

            const analysisConfig = {
                dataset_id: params.datasetId,
                analysis_types: [
                    'advanced_statistics',
                    'feature_importance',
                    'correlation_analysis',
                    'outlier_detection',
                    'clustering_analysis',
                    'dimensionality_reduction',
                    'time_series_decomposition',
                    'causal_inference'
                ],
                algorithms: {
                    clustering: ['kmeans', 'dbscan', 'hierarchical', 'gaussian_mixture'],
                    outlier_detection: ['isolation_forest', 'one_class_svm', 'local_outlier_factor'],
                    dimensionality_reduction: ['pca', 'tsne', 'umap', 'lda']
                },
                advanced_options: {
                    generate_insights: true,
                    create_visualizations: true,
                    export_results: true
                }
            };

            const response = await apiClient.post('/analysis/deep', analysisConfig);
            
            if (response.success) {
                this.displayDeepAnalysisResults(response.data);
                showToast('Deep analysis completed!', 'success');
            } else {
                throw new Error(response.error);
            }

        } catch (error) {
            console.error('Deep Analysis Error:', error);
            showToast(`Deep analysis failed: ${error.message}`, 'error');
        } finally {
            hideLoading();
        }
    }

    // Real-time Monitoring
    async startRealTimeMonitoring(params) {
        try {
            showToast('Starting real-time monitoring...', 'info');

            const config = {
                data_source: params.dataSource,
                metrics: params.metrics || ['all'],
                alert_thresholds: params.alertThresholds || {},
                update_interval: params.updateInterval || 1000
            };

            // Create WebSocket connection for real-time data
            const ws = new WebSocket(`ws://localhost:8000/ws/realtime/${params.dataSource}`);
            
            ws.onopen = () => {
                console.log('Real-time connection established');
                ws.send(JSON.stringify(config));
                this.realTimeConnections.set(params.dataSource, ws);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.updateRealTimeDisplay(data);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                showToast('Real-time connection error', 'error');
            };

            ws.onclose = () => {
                console.log('Real-time connection closed');
                this.realTimeConnections.delete(params.dataSource);
            };

        } catch (error) {
            console.error('Real-time Monitoring Error:', error);
            showToast(`Real-time monitoring failed: ${error.message}`, 'error');
        }
    }

    updateRealTimeDisplay(data) {
        // Update real-time charts
        if (data.metrics) {
            this.updateMetricsDisplay(data.metrics);
        }

        // Update alerts
        if (data.alerts) {
            this.showRealTimeAlerts(data.alerts);
        }

        // Update predictions
        if (data.predictions) {
            this.updatePredictionsDisplay(data.predictions);
        }
    }

    // Advanced Predictions
    async generatePredictions(params) {
        try {
            showToast('Generating advanced predictions...', 'info');
            showLoading('Running prediction models...');

            const predictionConfig = {
                model_ids: params.modelIds,
                input_data: params.inputData,
                prediction_types: [
                    'point_forecast',
                    'interval_forecast',
                    'probability_forecast',
                    'scenario_analysis'
                ],
                confidence_levels: [0.8, 0.9, 0.95],
                horizon: params.horizon || 30,
                include_explanations: true
            };

            const response = await apiClient.post('/predictions/advanced', predictionConfig);
            
            if (response.success) {
                this.displayPredictionResults(response.data);
                showToast('Predictions generated successfully!', 'success');
            } else {
                throw new Error(response.error);
            }

        } catch (error) {
            console.error('Prediction Error:', error);
            showToast(`Prediction failed: ${error.message}`, 'error');
        } finally {
            hideLoading();
        }
    }

    // Model Export
    async exportModel(params) {
        try {
            showToast('Exporting model...', 'info');

            const exportConfig = {
                model_id: params.modelId,
                format: params.format || 'onnx',
                include_metadata: true,
                include_preprocessing: true,
                optimize_for_inference: true
            };

            const response = await apiClient.post(`/models/${params.modelId}/export`, exportConfig);
            
            if (response.success) {
                // Download the exported model
                const downloadUrl = response.data.download_url;
                const link = document.createElement('a');
                link.href = downloadUrl;
                link.download = response.data.filename;
                link.click();
                
                showToast('Model exported successfully!', 'success');
            } else {
                throw new Error(response.error);
            }

        } catch (error) {
            console.error('Export Error:', error);
            showToast(`Model export failed: ${error.message}`, 'error');
        }
    }

    // Utility Functions
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }

    openCommandPalette() {
        // Create command palette modal
        const modal = document.createElement('div');
        modal.className = 'modal-overlay command-palette';
        modal.innerHTML = `
            <div class="command-palette-container">
                <div class="command-search">
                    <i class="fas fa-search"></i>
                    <input type="text" placeholder="Type a command..." id="command-input" />
                </div>
                <div class="command-list" id="command-list">
                    <!-- Commands will be populated here -->
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        modal.classList.add('active');
        
        const input = modal.querySelector('#command-input');
        input.focus();
        
        // Populate commands
        this.populateCommandList();
    }

    populateCommandList() {
        const commands = [
            { name: 'Start AutoML', action: 'start-automl', icon: 'fa-robot' },
            { name: 'Create Neural Network', action: 'create-neural-network', icon: 'fa-network-wired' },
            { name: 'Run Deep Analysis', action: 'run-deep-analysis', icon: 'fa-search-plus' },
            { name: 'Generate Predictions', action: 'generate-predictions', icon: 'fa-crystal-ball' },
            { name: 'Start Real-time Monitoring', action: 'start-real-time-monitoring', icon: 'fa-stream' },
            { name: 'Export Model', action: 'export-model', icon: 'fa-download' },
            { name: 'Create Visualization', action: 'create-visualization', icon: 'fa-chart-area' },
            { name: 'Generate Report', action: 'generate-report', icon: 'fa-file-alt' }
        ];

        const commandList = document.getElementById('command-list');
        commandList.innerHTML = commands.map(cmd => `
            <div class="command-item" data-action="${cmd.action}">
                <i class="fas ${cmd.icon}"></i>
                <span>${cmd.name}</span>
            </div>
        `).join('');
    }

    cleanupConnections() {
        // Close all WebSocket connections
        this.realTimeConnections.forEach((ws, key) => {
            ws.close();
        });
        this.realTimeConnections.clear();
    }
}

// Specialized ML Classes
class NeuralNetworkBuilder {
    constructor() {
        this.layers = [];
        this.connections = [];
    }

    addLayer(type, config) {
        const layer = {
            id: generateId(),
            type: type,
            config: config,
            position: { x: 0, y: this.layers.length * 100 }
        };
        this.layers.push(layer);
        return layer;
    }

    removeLayer(layerId) {
        this.layers = this.layers.filter(layer => layer.id !== layerId);
        this.connections = this.connections.filter(conn => 
            conn.from !== layerId && conn.to !== layerId
        );
    }

    connectLayers(fromId, toId) {
        this.connections.push({ from: fromId, to: toId });
    }

    generateArchitecture() {
        return {
            layers: this.layers,
            connections: this.connections
        };
    }
}

class TimeSeriesForecaster {
    constructor() {
        this.models = ['arima', 'lstm', 'prophet', 'transformer'];
    }

    async forecast(data, horizon, model = 'auto') {
        const config = {
            data: data,
            horizon: horizon,
            model: model,
            confidence_intervals: [0.8, 0.95],
            seasonality: 'auto',
            trend: 'auto'
        };

        return await apiClient.post('/forecasting/predict', config);
    }
}

class AnomalyDetector {
    constructor() {
        this.algorithms = ['isolation_forest', 'one_class_svm', 'autoencoder'];
    }

    async detect(data, sensitivity = 0.1) {
        const config = {
            data: data,
            algorithms: this.algorithms,
            sensitivity: sensitivity,
            return_scores: true
        };

        return await apiClient.post('/anomaly/detect', config);
    }
}

class RecommendationEngine {
    constructor() {
        this.methods = ['collaborative', 'content_based', 'hybrid'];
    }

    async recommend(userId, items, method = 'hybrid') {
        const config = {
            user_id: userId,
            items: items,
            method: method,
            num_recommendations: 10
        };

        return await apiClient.post('/recommendations/generate', config);
    }
}

class TextAnalyzer {
    constructor() {
        this.capabilities = ['sentiment', 'entities', 'topics', 'summarization'];
    }

    async analyze(text, analyses = this.capabilities) {
        const config = {
            text: text,
            analyses: analyses,
            language: 'auto'
        };

        return await apiClient.post('/text/analyze', config);
    }
}

class ImageProcessor {
    constructor() {
        this.tasks = ['classification', 'detection', 'segmentation', 'ocr'];
    }

    async process(imageData, task = 'classification') {
        const formData = new FormData();
        formData.append('image', imageData);
        formData.append('task', task);

        return await apiClient.post('/image/process', formData);
    }
}

// Initialize Advanced Features
const advancedFeatures = new AdvancedFeatures();

// Export for global access
window.advancedFeatures = advancedFeatures;
