/**
 * API Client for Auto Analytics Platform
 * Handles all communication with the backend REST API
 */

class ApiClient {
    constructor(baseUrl = '/api/v1') {
        this.baseUrl = baseUrl;
        this.defaultHeaders = {
            'Content-Type': 'application/json',
        };
    }

    /**
     * Make HTTP request
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Request options
     * @returns {Promise<Object>} - Response data
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const config = {
            method: options.method || 'GET',
            headers: {
                ...this.defaultHeaders,
                ...options.headers
            },
            ...options
        };

        // Don't set Content-Type for FormData
        if (options.body instanceof FormData) {
            delete config.headers['Content-Type'];
        } else if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }

        try {
            const response = await fetch(url, config);
            
            // Handle different content types
            const contentType = response.headers.get('content-type');
            let data;
            
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
            } else {
                data = await response.text();
            }

            if (!response.ok) {
                throw new Error(data.detail || data.message || `HTTP ${response.status}: ${response.statusText}`);
            }

            return {
                success: true,
                data: data,
                status: response.status,
                headers: response.headers
            };

        } catch (error) {
            console.error('API Request failed:', error);
            return {
                success: false,
                error: error.message,
                status: 0
            };
        }
    }

    /**
     * GET request
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @returns {Promise<Object>} - Response data
     */
    async get(endpoint, params = {}) {
        const searchParams = new URLSearchParams(params);
        const url = searchParams.toString() ? `${endpoint}?${searchParams}` : endpoint;
        return this.request(url);
    }

    /**
     * POST request
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body
     * @param {Object} options - Additional options
     * @returns {Promise<Object>} - Response data
     */
    async post(endpoint, data = {}, options = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: data,
            ...options
        });
    }

    /**
     * PUT request
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body
     * @returns {Promise<Object>} - Response data
     */
    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: data
        });
    }

    /**
     * DELETE request
     * @param {string} endpoint - API endpoint
     * @returns {Promise<Object>} - Response data
     */
    async delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }

    // ===== Dataset API Methods =====

    /**
     * Upload dataset file
     * @param {File} file - File to upload
     * @param {Function} onProgress - Progress callback
     * @returns {Promise<Object>} - Upload response
     */
    async uploadDataset(file, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);

        const config = {
            method: 'POST',
            body: formData
        };

        // Add progress tracking if callback provided
        if (onProgress && typeof onProgress === 'function') {
            const xhr = new XMLHttpRequest();
            
            return new Promise((resolve, reject) => {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        onProgress(percentComplete);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            resolve({ success: true, data: response });
                        } catch (error) {
                            reject(new Error('Invalid JSON response'));
                        }
                    } else {
                        reject(new Error(`Upload failed: ${xhr.statusText}`));
                    }
                });

                xhr.addEventListener('error', () => {
                    reject(new Error('Upload failed'));
                });

                xhr.open('POST', `${this.baseUrl}/upload`);
                xhr.send(formData);
            });
        }

        return this.request('/upload', config);
    }

    /**
     * Get list of datasets
     * @param {Object} params - Query parameters
     * @returns {Promise<Object>} - Dataset list
     */
    async getDatasets(params = {}) {
        return this.get('/datasets', params);
    }

    /**
     * Get dataset details
     * @param {string} datasetId - Dataset ID
     * @returns {Promise<Object>} - Dataset details
     */
    async getDatasetDetails(datasetId) {
        return this.get(`/datasets/${datasetId}`);
    }

    /**
     * Delete dataset
     * @param {string} datasetId - Dataset ID
     * @returns {Promise<Object>} - Deletion response
     */
    async deleteDataset(datasetId) {
        return this.delete(`/datasets/${datasetId}`);
    }

    /**
     * Get dataset quality report
     * @param {string} datasetId - Dataset ID
     * @returns {Promise<Object>} - Quality report
     */
    async getDatasetQuality(datasetId) {
        return this.get(`/datasets/${datasetId}/quality`);
    }

    /**
     * Preprocess dataset
     * @param {string} datasetId - Dataset ID
     * @param {Object} options - Preprocessing options
     * @returns {Promise<Object>} - Preprocessing response
     */
    async preprocessDataset(datasetId, options) {
        return this.post(`/datasets/${datasetId}/preprocess`, options);
    }

    // ===== Analysis API Methods =====

    /**
     * Start data analysis
     * @param {Object} request - Analysis request
     * @returns {Promise<Object>} - Analysis response
     */
    async startAnalysis(request) {
        return this.post('/analysis', request);
    }

    /**
     * Get analysis results
     * @param {string} analysisId - Analysis ID
     * @returns {Promise<Object>} - Analysis results
     */
    async getAnalysisResults(analysisId) {
        return this.get(`/analysis/${analysisId}`);
    }

    /**
     * Get analysis status
     * @param {string} analysisId - Analysis ID
     * @returns {Promise<Object>} - Analysis status
     */
    async getAnalysisStatus(analysisId) {
        return this.get(`/analysis/${analysisId}/status`);
    }

    /**
     * Get list of analyses
     * @param {Object} params - Query parameters
     * @returns {Promise<Object>} - Analysis list
     */
    async getAnalyses(params = {}) {
        return this.get('/analysis', params);
    }

    /**
     * Perform quick analysis
     * @param {string} datasetId - Dataset ID
     * @returns {Promise<Object>} - Quick analysis results
     */
    async quickAnalysis(datasetId) {
        return this.post(`/analysis/quick/${datasetId}`);
    }

    // ===== ML Model API Methods =====

    /**
     * Train ML model
     * @param {Object} request - Training request
     * @returns {Promise<Object>} - Training response
     */
    async trainModel(request) {
        return this.post('/models/train', request);
    }

    /**
     * Get list of models
     * @param {Object} params - Query parameters
     * @returns {Promise<Object>} - Model list
     */
    async getModels(params = {}) {
        return this.get('/models', params);
    }

    /**
     * Get model details
     * @param {string} modelId - Model ID
     * @returns {Promise<Object>} - Model details
     */
    async getModelDetails(modelId) {
        return this.get(`/models/${modelId}`);
    }

    /**
     * Make prediction
     * @param {string} modelId - Model ID
     * @param {Object} request - Prediction request
     * @returns {Promise<Object>} - Prediction response
     */
    async makePrediction(modelId, request) {
        return this.post(`/models/${modelId}/predict`, request);
    }

    /**
     * Get model status
     * @param {string} modelId - Model ID
     * @returns {Promise<Object>} - Model status
     */
    async getModelStatus(modelId) {
        return this.get(`/models/${modelId}/status`);
    }

    /**
     * Compare models
     * @param {Object} request - Comparison request
     * @returns {Promise<Object>} - Comparison results
     */
    async compareModels(request) {
        return this.post('/models/compare', request);
    }

    /**
     * Start AutoML
     * @param {Object} request - AutoML request
     * @returns {Promise<Object>} - AutoML response
     */
    async startAutoML(request) {
        return this.post('/models/automl', request);
    }

    // ===== Visualization API Methods =====

    /**
     * Generate visualization
     * @param {Object} request - Visualization request
     * @returns {Promise<Object>} - Visualization response
     */
    async generateVisualization(request) {
        return this.post('/visualizations/generate', null, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams(request)
        });
    }

    /**
     * Auto-generate visualizations
     * @param {string} datasetId - Dataset ID
     * @returns {Promise<Object>} - Generated visualizations
     */
    async autoGenerateVisualizations(datasetId) {
        return this.post(`/visualizations/auto-generate/${datasetId}`);
    }

    /**
     * Get list of visualizations
     * @param {Object} params - Query parameters
     * @returns {Promise<Object>} - Visualization list
     */
    async getVisualizations(params = {}) {
        return this.get('/visualizations', params);
    }

    /**
     * Get visualization details
     * @param {string} chartId - Chart ID
     * @returns {Promise<Object>} - Visualization details
     */
    async getVisualizationDetails(chartId) {
        return this.get(`/visualizations/${chartId}`);
    }

    /**
     * Get available chart types
     * @returns {Promise<Object>} - Available chart types
     */
    async getChartTypes() {
        return this.get('/visualizations/types/available');
    }

    // ===== Report API Methods =====

    /**
     * Generate report
     * @param {Object} request - Report request
     * @returns {Promise<Object>} - Report response
     */
    async generateReport(request) {
        return this.post('/reports/generate', request);
    }

    /**
     * Get list of reports
     * @param {Object} params - Query parameters
     * @returns {Promise<Object>} - Report list
     */
    async getReports(params = {}) {
        return this.get('/reports', params);
    }

    /**
     * Get report details
     * @param {string} reportId - Report ID
     * @returns {Promise<Object>} - Report details
     */
    async getReportDetails(reportId) {
        return this.get(`/reports/${reportId}`);
    }

    /**
     * Get report status
     * @param {string} reportId - Report ID
     * @returns {Promise<Object>} - Report status
     */
    async getReportStatus(reportId) {
        return this.get(`/reports/${reportId}/status`);
    }

    /**
     * Download report
     * @param {string} reportId - Report ID
     * @returns {Promise<Object>} - Download response
     */
    async downloadReport(reportId) {
        return this.get(`/reports/${reportId}/download`);
    }

    /**
     * Generate custom report
     * @param {Object} request - Custom report request
     * @returns {Promise<Object>} - Report response
     */
    async generateCustomReport(request) {
        return this.post('/reports/custom', request);
    }

    /**
     * Get report templates
     * @param {Object} params - Query parameters
     * @returns {Promise<Object>} - Template list
     */
    async getReportTemplates(params = {}) {
        return this.get('/reports/templates', params);
    }

    /**
     * Get report analytics
     * @returns {Promise<Object>} - Report analytics
     */
    async getReportAnalytics() {
        return this.get('/reports/analytics');
    }

    // ===== System API Methods =====

    /**
     * Get API status
     * @returns {Promise<Object>} - API status
     */
    async getApiStatus() {
        return this.get('/status');
    }

    /**
     * Get health check
     * @returns {Promise<Object>} - Health status
     */
    async getHealth() {
        return this.get('/health');
    }
}

// Create global API client instance
const apiClient = new ApiClient();

// Export for use in other files
window.apiClient = apiClient;
window.ApiClient = ApiClient;
