/**
 * Section Templates - Real Backend Connected HTML Templates
 * This file contains all the HTML templates for different sections that connect to backend models
 */

// Dashboard Section Template
function getDashboardHTML() {
    return `
        <div class="row mb-4">
            <div class="col-12">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h1 class="display-6 fw-bold text-primary">NISR Rwanda Analytics Dashboard</h1>
                        <p class="text-muted">Real-time insights from Rwanda's national statistics</p>
                    </div>
                    <button class="btn btn-primary btn-modern" onclick="refreshDashboard()">
                        <i class="fas fa-sync-alt me-2"></i>Refresh Data
                    </button>
                </div>
            </div>
        </div>

        <!-- Key Metrics Cards -->
        <div class="row mb-4" id="dashboard-metrics">
            <div class="col-md-3">
                <div class="card card-modern text-center">
                    <div class="card-body">
                        <div class="text-primary mb-2">
                            <i class="fas fa-database fa-2x"></i>
                        </div>
                        <h3 class="fw-bold" id="total-datasets">-</h3>
                        <p class="text-muted mb-0">Total Datasets</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card card-modern text-center">
                    <div class="card-body">
                        <div class="text-success mb-2">
                            <i class="fas fa-chart-bar fa-2x"></i>
                        </div>
                        <h3 class="fw-bold" id="total-analyses">-</h3>
                        <p class="text-muted mb-0">Analyses Completed</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card card-modern text-center">
                    <div class="card-body">
                        <div class="text-warning mb-2">
                            <i class="fas fa-brain fa-2x"></i>
                        </div>
                        <h3 class="fw-bold" id="total-models">-</h3>
                        <p class="text-muted mb-0">ML Models</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card card-modern text-center">
                    <div class="card-body">
                        <div class="text-info mb-2">
                            <i class="fas fa-file-invoice fa-2x"></i>
                        </div>
                        <h3 class="fw-bold" id="total-reports">-</h3>
                        <p class="text-muted mb-0">Reports Generated</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Recent Activity -->
        <div class="row">
            <div class="col-md-8">
                <div class="card card-modern">
                    <div class="card-header">
                        <h5 class="mb-0">Recent Activity</h5>
                    </div>
                    <div class="card-body">
                        <div id="recent-activity-list">
                            <div class="text-center text-muted">
                                <i class="fas fa-spinner fa-spin"></i> Loading recent activity...
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card card-modern">
                    <div class="card-header">
                        <h5 class="mb-0">Quick Actions</h5>
                    </div>
                    <div class="card-body">
                        <div class="d-grid gap-2">
                            <button class="btn btn-outline-primary" onclick="navigateToSection('data-upload')">
                                <i class="fas fa-upload me-2"></i>Upload New Dataset
                            </button>
                            <button class="btn btn-outline-success" onclick="navigateToSection('analysis')">
                                <i class="fas fa-chart-bar me-2"></i>Start Analysis
                            </button>
                            <button class="btn btn-outline-info" onclick="navigateToSection('ai-chat')">
                                <i class="fas fa-comments me-2"></i>AI Assistant
                            </button>
                            <button class="btn btn-outline-warning" onclick="navigateToSection('automl')">
                                <i class="fas fa-brain me-2"></i>Train ML Model
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Load dashboard data from backend
            async function loadDashboardData() {
                try {
                    const response = await fetch('/api/v1/dashboard/stats');
                    if (response.ok) {
                        const data = await response.json();
                        document.getElementById('total-datasets').textContent = data.total_datasets || 0;
                        document.getElementById('total-analyses').textContent = data.total_analyses || 0;
                        document.getElementById('total-models').textContent = data.total_models || 0;
                        document.getElementById('total-reports').textContent = data.total_reports || 0;
                    }
                } catch (error) {
                    console.error('Error loading dashboard data:', error);
                }
            }

            async function loadRecentActivity() {
                try {
                    const response = await fetch('/api/v1/dashboard/recent-activity');
                    if (response.ok) {
                        const activities = await response.json();
                        const activityList = document.getElementById('recent-activity-list');
                        
                        if (activities.length === 0) {
                            activityList.innerHTML = '<div class="text-muted">No recent activity</div>';
                            return;
                        }
                        
                        activityList.innerHTML = activities.map(activity => \`
                            <div class="d-flex align-items-center mb-3">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-\${activity.icon} text-\${activity.type}"></i>
                                </div>
                                <div class="flex-grow-1 ms-3">
                                    <h6 class="mb-1">\${activity.title}</h6>
                                    <p class="text-muted mb-0">\${activity.description}</p>
                                    <small class="text-muted">\${activity.timestamp}</small>
                                </div>
                            </div>
                        \`).join('');
                    }
                } catch (error) {
                    console.error('Error loading recent activity:', error);
                    document.getElementById('recent-activity-list').innerHTML = 
                        '<div class="text-danger">Error loading activity</div>';
                }
            }

            function refreshDashboard() {
                loadDashboardData();
                loadRecentActivity();
                window.showToast('Dashboard refreshed', 'success');
            }

            // Auto-load data when dashboard renders
            setTimeout(() => {
                loadDashboardData();
                loadRecentActivity();
            }, 100);
        </script>
    `;
}

// Data Upload Section Template
function getDataUploadHTML() {
    return `
        <div class="row">
            <div class="col-12">
                <h2 class="fw-bold mb-4">Upload Dataset</h2>
            </div>
        </div>

        <div class="row">
            <div class="col-md-8">
                <div class="card card-modern">
                    <div class="card-header">
                        <h5 class="mb-0">Upload New Dataset</h5>
                    </div>
                    <div class="card-body">
                        <form id="upload-form" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="file-input" class="form-label">Select File</label>
                                <input type="file" class="form-control form-control-modern" id="file-input" 
                                       accept=".csv,.xlsx,.xls,.json,.parquet" required>
                                <div class="form-text">Supported formats: CSV, Excel, JSON, Parquet (Max: 500MB)</div>
                            </div>
                            
                            <div class="mb-3">
                                <label for="dataset-name" class="form-label">Dataset Name</label>
                                <input type="text" class="form-control form-control-modern" id="dataset-name" 
                                       placeholder="Enter dataset name" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="description" class="form-label">Description</label>
                                <textarea class="form-control form-control-modern" id="description" rows="3" 
                                          placeholder="Describe your dataset..."></textarea>
                            </div>
                            
                            <div class="mb-3">
                                <label for="data-source" class="form-label">Data Source</label>
                                <select class="form-select form-control-modern" id="data-source">
                                    <option value="">Select source type</option>
                                    <option value="census">Census Data</option>
                                    <option value="survey">Survey Data</option>
                                    <option value="economic">Economic Indicators</option>
                                    <option value="administrative">Administrative Records</option>
                                    <option value="external">External Source</option>
                                    <option value="other">Other</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="geographic-coverage" class="form-label">Geographic Coverage</label>
                                <select class="form-select form-control-modern" id="geographic-coverage" multiple>
                                    <option value="national">National</option>
                                    <option value="kigali">Kigali City</option>
                                    <option value="eastern">Eastern Province</option>
                                    <option value="northern">Northern Province</option>
                                    <option value="southern">Southern Province</option>
                                    <option value="western">Western Province</option>
                                </select>
                                <div class="form-text">Hold Ctrl to select multiple provinces</div>
                            </div>
                            
                            <div class="d-flex gap-2">
                                <button type="submit" class="btn btn-primary btn-modern">
                                    <i class="fas fa-upload me-2"></i>Upload Dataset
                                </button>
                                <button type="button" class="btn btn-outline-secondary btn-modern" onclick="resetUploadForm()">
                                    <i class="fas fa-undo me-2"></i>Reset
                                </button>
                            </div>
                        </form>
                        
                        <!-- Upload Progress -->
                        <div id="upload-progress" class="mt-4" style="display: none;">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span>Uploading...</span>
                                <span id="upload-percentage">0%</span>
                            </div>
                            <div class="progress progress-modern">
                                <div class="progress-bar" id="upload-progress-bar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card card-modern">
                    <div class="card-header">
                        <h5 class="mb-0">Upload Guidelines</h5>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-info">
                            <h6>File Requirements:</h6>
                            <ul class="mb-0">
                                <li>Maximum file size: 500MB</li>
                                <li>Supported formats: CSV, Excel, JSON, Parquet</li>
                                <li>First row should contain column headers</li>
                                <li>Avoid special characters in column names</li>
                            </ul>
                        </div>
                        
                        <div class="alert alert-warning">
                            <h6>Data Quality Tips:</h6>
                            <ul class="mb-0">
                                <li>Remove empty rows at the end</li>
                                <li>Use consistent date formats</li>
                                <li>Handle missing values appropriately</li>
                                <li>Ensure numeric columns contain only numbers</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('upload-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('file-input');
                const file = fileInput.files[0];
                
                if (!file) {
                    window.showToast('Please select a file', 'error');
                    return;
                }
                
                // Show upload progress
                document.getElementById('upload-progress').style.display = 'block';
                
                formData.append('file', file);
                formData.append('dataset_name', document.getElementById('dataset-name').value);
                formData.append('description', document.getElementById('description').value);
                formData.append('data_source', document.getElementById('data-source').value);
                
                const geoCoverage = Array.from(document.getElementById('geographic-coverage').selectedOptions)
                    .map(option => option.value);
                formData.append('geographic_coverage', JSON.stringify(geoCoverage));
                
                try {
                    const response = await fetch('/api/v1/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        window.showToast('Dataset uploaded successfully!', 'success');
                        resetUploadForm();
                        
                        // Navigate to datasets page
                        setTimeout(() => {
                            navigateToSection('datasets');
                        }, 2000);
                    } else {
                        const error = await response.json();
                        window.showToast(error.detail || 'Upload failed', 'error');
                    }
                } catch (error) {
                    console.error('Upload error:', error);
                    window.showToast('Upload failed', 'error');
                } finally {
                    document.getElementById('upload-progress').style.display = 'none';
                }
            });
            
            function resetUploadForm() {
                document.getElementById('upload-form').reset();
                document.getElementById('upload-progress').style.display = 'none';
            }
        </script>
    `;
}

// Datasets Section Template
function getDatasetsHTML() {
    return `
        <div class="row mb-4">
            <div class="col-12">
                <div class="d-flex justify-content-between align-items-center">
                    <h2 class="fw-bold">Datasets</h2>
                    <button class="btn btn-primary btn-modern" onclick="navigateToSection('data-upload')">
                        <i class="fas fa-plus me-2"></i>Add Dataset
                    </button>
                </div>
            </div>
        </div>

        <!-- Filters -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card card-modern">
                    <div class="card-body">
                        <div class="row g-3">
                            <div class="col-md-3">
                                <input type="text" class="form-control form-control-modern" 
                                       id="search-datasets" placeholder="Search datasets...">
                            </div>
                            <div class="col-md-3">
                                <select class="form-select form-control-modern" id="filter-source">
                                    <option value="">All Sources</option>
                                    <option value="census">Census</option>
                                    <option value="survey">Survey</option>
                                    <option value="economic">Economic</option>
                                    <option value="administrative">Administrative</option>
                                    <option value="external">External</option>
                                </select>
                            </div>
                            <div class="col-md-3">
                                <select class="form-select form-control-modern" id="filter-status">
                                    <option value="">All Status</option>
                                    <option value="completed">Completed</option>
                                    <option value="processing">Processing</option>
                                    <option value="failed">Failed</option>
                                </select>
                            </div>
                            <div class="col-md-3">
                                <button class="btn btn-outline-secondary btn-modern w-100" onclick="loadDatasets()">
                                    <i class="fas fa-sync-alt me-2"></i>Refresh
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Datasets Table -->
        <div class="row">
            <div class="col-12">
                <div class="card card-modern">
                    <div class="card-header">
                        <h5 class="mb-0">Dataset Collection</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover table-modern" id="datasets-table">
                                <thead class="table-light">
                                    <tr>
                                        <th>Name</th>
                                        <th>Source</th>
                                        <th>Rows</th>
                                        <th>Columns</th>
                                        <th>Size</th>
                                        <th>Status</th>
                                        <th>Uploaded</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody id="datasets-tbody">
                                    <tr>
                                        <td colspan="8" class="text-center text-muted">
                                            <i class="fas fa-spinner fa-spin"></i> Loading datasets...
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function loadDatasets() {
                try {
                    const response = await fetch('/api/v1/datasets');
                    if (response.ok) {
                        const datasets = await response.json();
                        renderDatasets(datasets);
                    } else {
                        document.getElementById('datasets-tbody').innerHTML = 
                            '<tr><td colspan="8" class="text-center text-danger">Error loading datasets</td></tr>';
                    }
                } catch (error) {
                    console.error('Error loading datasets:', error);
                    document.getElementById('datasets-tbody').innerHTML = 
                        '<tr><td colspan="8" class="text-center text-danger">Error loading datasets</td></tr>';
                }
            }

            function renderDatasets(datasets) {
                const tbody = document.getElementById('datasets-tbody');
                
                if (datasets.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No datasets found</td></tr>';
                    return;
                }

                tbody.innerHTML = datasets.map(dataset => \`
                    <tr>
                        <td>
                            <div class="fw-semibold">\${dataset.filename}</div>
                            <small class="text-muted">\${dataset.dataset_id}</small>
                        </td>
                        <td><span class="badge bg-secondary">\${dataset.data_source || 'Unknown'}</span></td>
                        <td>\${dataset.rows ? dataset.rows.toLocaleString() : '-'}</td>
                        <td>\${dataset.columns || '-'}</td>
                        <td>\${formatFileSize(dataset.file_size)}</td>
                        <td>
                            <span class="badge bg-\${getStatusColor(dataset.status)}">\${dataset.status}</span>
                        </td>
                        <td>\${formatDate(dataset.upload_timestamp)}</td>
                        <td>
                            <div class="btn-group btn-group-sm">
                                <button class="btn btn-outline-primary" onclick="viewDataset('\${dataset.dataset_id}')" title="View">
                                    <i class="fas fa-eye"></i>
                                </button>
                                <button class="btn btn-outline-success" onclick="analyzeDataset('\${dataset.dataset_id}')" title="Analyze">
                                    <i class="fas fa-chart-bar"></i>
                                </button>
                                <button class="btn btn-outline-danger" onclick="deleteDataset('\${dataset.dataset_id}')" title="Delete">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </td>
                    </tr>
                \`).join('');
            }

            function getStatusColor(status) {
                switch(status) {
                    case 'completed': return 'success';
                    case 'processing': return 'warning';
                    case 'failed': return 'danger';
                    default: return 'secondary';
                }
            }

            function formatFileSize(bytes) {
                if (!bytes) return '-';
                const sizes = ['B', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(1024));
                return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + sizes[i];
            }

            function formatDate(dateString) {
                if (!dateString) return '-';
                return new Date(dateString).toLocaleDateString();
            }

            async function deleteDataset(datasetId) {
                if (!confirm('Are you sure you want to delete this dataset?')) return;
                
                try {
                    const response = await fetch(\`/api/v1/datasets/\${datasetId}\`, {
                        method: 'DELETE'
                    });
                    
                    if (response.ok) {
                        window.showToast('Dataset deleted successfully', 'success');
                        loadDatasets();
                    } else {
                        window.showToast('Failed to delete dataset', 'error');
                    }
                } catch (error) {
                    window.showToast('Error deleting dataset', 'error');
                }
            }

            function viewDataset(datasetId) {
                window.APP_CONFIG.CURRENT_DATASET = datasetId;
                // Navigate to dataset details view
                window.showToast('Dataset details view - coming soon', 'info');
            }

            function analyzeDataset(datasetId) {
                window.APP_CONFIG.CURRENT_DATASET = datasetId;
                navigateToSection('analysis');
            }

            // Auto-load datasets
            setTimeout(loadDatasets, 100);
        </script>
    `;
}

// AI Chat Section Template
function getAIChatHTML() {
    return `
        <div class="row">
            <div class="col-12">
                <h2 class="fw-bold mb-4">AI Assistant</h2>
            </div>
        </div>

        <div class="row">
            <div class="col-md-8">
                <div class="card card-modern h-100">
                    <div class="card-header bg-primary text-white">
                        <div class="d-flex justify-content-between align-items-center">
                            <h5 class="mb-0">
                                <i class="fas fa-robot me-2"></i>NISR AI Assistant
                            </h5>
                            <button class="btn btn-sm btn-outline-light" onclick="clearChat()">
                                <i class="fas fa-trash"></i> Clear
                            </button>
                        </div>
                    </div>
                    <div class="card-body p-0" style="height: 500px; display: flex; flex-direction: column;">
                        <!-- Chat Messages -->
                        <div class="flex-grow-1 p-3" id="chat-messages" style="overflow-y: auto; max-height: 400px;">
                            <div class="chat-message assistant">
                                <div class="d-flex align-items-start mb-3">
                                    <div class="bg-primary text-white rounded-circle p-2 me-3">
                                        <i class="fas fa-robot"></i>
                                    </div>
                                    <div class="flex-grow-1">
                                        <div class="bg-light rounded p-3">
                                            <p class="mb-0">Hello! I'm your NISR AI Assistant. I can help you with data analysis, insights, and answering questions about your datasets. How can I assist you today?</p>
                                        </div>
                                        <small class="text-muted">Just now</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Chat Input -->
                        <div class="border-top p-3">
                            <form id="chat-form" class="d-flex gap-2">
                                <select class="form-select" id="chat-dataset" style="max-width: 200px;">
                                    <option value="">All datasets</option>
                                </select>
                                <input type="text" class="form-control form-control-modern" 
                                       id="chat-input" placeholder="Ask me anything about your data..." required>
                                <button type="submit" class="btn btn-primary">
                                    <i class="fas fa-paper-plane"></i>
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card card-modern mb-3">
                    <div class="card-header">
                        <h6 class="mb-0">Quick Questions</h6>
                    </div>
                    <div class="card-body">
                        <div class="d-grid gap-2">
                            <button class="btn btn-outline-primary btn-sm" onclick="askQuickQuestion('What datasets do I have?')">
                                What datasets do I have?
                            </button>
                            <button class="btn btn-outline-primary btn-sm" onclick="askQuickQuestion('Show me summary statistics')">
                                Show me summary statistics
                            </button>
                            <button class="btn btn-outline-primary btn-sm" onclick="askQuickQuestion('Find correlations in my data')">
                                Find correlations in my data
                            </button>
                            <button class="btn btn-outline-primary btn-sm" onclick="askQuickQuestion('Generate a visualization')">
                                Generate a visualization
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="card card-modern">
                    <div class="card-header">
                        <h6 class="mb-0">AI Capabilities</h6>
                    </div>
                    <div class="card-body">
                        <ul class="list-unstyled">
                            <li><i class="fas fa-check text-success me-2"></i>Data exploration</li>
                            <li><i class="fas fa-check text-success me-2"></i>Statistical analysis</li>
                            <li><i class="fas fa-check text-success me-2"></i>Trend identification</li>
                            <li><i class="fas fa-check text-success me-2"></i>Visualization suggestions</li>
                            <li><i class="fas fa-check text-success me-2"></i>Rwanda-specific insights</li>
                            <li><i class="fas fa-check text-success me-2"></i>NISR data standards</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let chatSessionId = null;

            document.getElementById('chat-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const input = document.getElementById('chat-input');
                const message = input.value.trim();
                const datasetId = document.getElementById('chat-dataset').value;
                
                if (!message) return;
                
                // Add user message to chat
                addChatMessage('user', message);
                input.value = '';
                
                // Show typing indicator
                showTypingIndicator();
                
                try {
                    const response = await fetch('/api/v1/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            message: message,
                            dataset_id: datasetId,
                            session_id: chatSessionId
                        })
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        chatSessionId = result.session_id;
                        
                        hideTypingIndicator();
                        addChatMessage('assistant', result.response);
                        
                        // If response includes data or charts, render them
                        if (result.data) {
                            renderChatData(result.data);
                        }
                    } else {
                        hideTypingIndicator();
                        addChatMessage('assistant', 'Sorry, I encountered an error. Please try again.');
                    }
                } catch (error) {
                    console.error('Chat error:', error);
                    hideTypingIndicator();
                    addChatMessage('assistant', 'Sorry, I encountered a connection error. Please try again.');
                }
            });

            function addChatMessage(sender, message) {
                const messagesContainer = document.getElementById('chat-messages');
                const isUser = sender === 'user';
                
                const messageDiv = document.createElement('div');
                messageDiv.className = \`chat-message \${sender}\`;
                messageDiv.innerHTML = \`
                    <div class="d-flex align-items-start mb-3 \${isUser ? 'justify-content-end' : ''}">
                        \${!isUser ? \`
                            <div class="bg-primary text-white rounded-circle p-2 me-3">
                                <i class="fas fa-robot"></i>
                            </div>
                        \` : ''}
                        <div class="flex-grow-1 \${isUser ? 'text-end' : ''}">
                            <div class="\${isUser ? 'bg-primary text-white' : 'bg-light'} rounded p-3" style="max-width: 80%; \${isUser ? 'margin-left: auto;' : ''}">
                                <p class="mb-0">\${message}</p>
                            </div>
                            <small class="text-muted">\${new Date().toLocaleTimeString()}</small>
                        </div>
                        \${isUser ? \`
                            <div class="bg-secondary text-white rounded-circle p-2 ms-3">
                                <i class="fas fa-user"></i>
                            </div>
                        \` : ''}
                    </div>
                \`;
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }

            function showTypingIndicator() {
                const messagesContainer = document.getElementById('chat-messages');
                const typingDiv = document.createElement('div');
                typingDiv.id = 'typing-indicator';
                typingDiv.innerHTML = \`
                    <div class="d-flex align-items-start mb-3">
                        <div class="bg-primary text-white rounded-circle p-2 me-3">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="bg-light rounded p-3">
                            <div class="typing-dots">
                                <span></span><span></span><span></span>
                            </div>
                        </div>
                    </div>
                \`;
                
                messagesContainer.appendChild(typingDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }

            function hideTypingIndicator() {
                const typingIndicator = document.getElementById('typing-indicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            }

            function askQuickQuestion(question) {
                document.getElementById('chat-input').value = question;
                document.getElementById('chat-form').dispatchEvent(new Event('submit'));
            }

            function clearChat() {
                document.getElementById('chat-messages').innerHTML = \`
                    <div class="chat-message assistant">
                        <div class="d-flex align-items-start mb-3">
                            <div class="bg-primary text-white rounded-circle p-2 me-3">
                                <i class="fas fa-robot"></i>
                            </div>
                            <div class="flex-grow-1">
                                <div class="bg-light rounded p-3">
                                    <p class="mb-0">Chat cleared. How can I help you?</p>
                                </div>
                                <small class="text-muted">Just now</small>
                            </div>
                        </div>
                    </div>
                \`;
                chatSessionId = null;
            }

            // Load available datasets for chat context
            async function loadChatDatasets() {
                try {
                    const response = await fetch('/api/v1/datasets');
                    if (response.ok) {
                        const datasets = await response.json();
                        const select = document.getElementById('chat-dataset');
                        
                        datasets.forEach(dataset => {
                            const option = document.createElement('option');
                            option.value = dataset.dataset_id;
                            option.textContent = dataset.filename;
                            select.appendChild(option);
                        });
                    }
                } catch (error) {
                    console.error('Error loading datasets for chat:', error);
                }
            }

            // Auto-load datasets
            setTimeout(loadChatDatasets, 100);
        </script>

        <style>
            .typing-dots {
                display: flex;
                gap: 4px;
            }
            
            .typing-dots span {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #6c757d;
                animation: typing 1.4s infinite ease-in-out both;
            }
            
            .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
            .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
            
            @keyframes typing {
                0%, 80%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                40% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
        </style>
    `;
}

// Default template for other sections
function getDefaultHTML(section) {
    const sectionNames = {
        'analysis': 'Data Analysis',
        'automl': 'Automated Machine Learning',
        'visualizations': 'Data Visualizations',
        'reports': 'Reports & Documents',
        'census-data': 'Census Data Analysis',
        'economic-indicators': 'Economic Indicators',
        'geo-analysis': 'Geographic Analysis'
    };
    
    const name = sectionNames[section] || section.charAt(0).toUpperCase() + section.slice(1);
    
    return `
        <div class="row">
            <div class="col-12">
                <div class="card card-modern">
                    <div class="card-body text-center py-5">
                        <i class="fas fa-cog fa-3x text-muted mb-3"></i>
                        <h3>${name}</h3>
                        <p class="text-muted">This section is under development with real backend integration.</p>
                        <button class="btn btn-primary btn-modern" onclick="navigateToSection('dashboard')">
                            <i class="fas fa-arrow-left me-2"></i>Return to Dashboard
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Navigation helper function
function navigateToSection(section) {
    // Find and click the corresponding nav link
    const navLink = document.querySelector(`[data-section="${section}"]`);
    if (navLink) {
        navLink.click();
    }
}
