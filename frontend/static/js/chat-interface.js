/**
 * AI Chat Interface JavaScript
 * Handles the conversational AI interface for data analysis
 */

class ChatInterface {
    constructor() {
        this.sessionId = null;
        this.currentDataset = null;
        this.isTyping = false;
        this.messageHistory = [];
        this.settings = {
            responseStyle: 'concise',
            autoVisualizations: true,
            showSuggestions: true
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeSession();
        this.loadDatasets();
        this.loadSettings();
    }
    
    setupEventListeners() {
        // Chat input handling
        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
            chatInput.addEventListener('input', () => {
                this.updateSendButtonState();
                this.autoResizeInput();
            });
            
            chatInput.addEventListener('keydown', (e) => {
                this.handleKeyDown(e);
            });
        }
        
        // Dataset selection
        const datasetSelect = document.getElementById('chat-dataset-select');
        if (datasetSelect) {
            datasetSelect.addEventListener('change', () => {
                this.updateChatContext();
            });
        }
        
        // Settings
        const settingsInputs = document.querySelectorAll('#chat-settings input, #chat-settings select');
        settingsInputs.forEach(input => {
            input.addEventListener('change', () => {
                this.updateSettings();
            });
        });
    }
    
    initializeSession() {
        this.sessionId = `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        document.getElementById('chat-session-id').textContent = `Session: ${this.sessionId.slice(-8)}`;
    }
    
    async loadDatasets() {
        try {
            const response = await window.apiClient.get('/api/v1/datasets');
            const datasets = response.datasets || [];
            
            const select = document.getElementById('chat-dataset-select');
            select.innerHTML = '<option value="">Select a dataset...</option>';
            
            datasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset.dataset_id;
                option.textContent = `${dataset.filename} (${dataset.rows} rows)`;
                select.appendChild(option);
            });
            
        } catch (error) {
            console.error('Error loading datasets:', error);
            this.showError('Failed to load datasets');
        }
    }
    
    updateChatContext() {
        const select = document.getElementById('chat-dataset-select');
        this.currentDataset = select.value;
        
        if (this.currentDataset) {
            this.addSystemMessage(`📊 Active dataset changed to: ${select.options[select.selectedIndex].text}`);
            this.updateSuggestions();
        }
    }
    
    updateSuggestions() {
        const suggestionsContainer = document.querySelector('.suggestions-list');
        
        if (this.currentDataset && this.settings.showSuggestions) {
            const datasetSuggestions = [
                `Describe the ${this.currentDataset} dataset`,
                `Show me column information for this data`,
                `Generate summary statistics`,
                `Create visualizations for key variables`,
                `Find missing values and data quality issues`,
                `Identify outliers in the data`,
                `Show correlation between variables`
            ];
            
            suggestionsContainer.innerHTML = '';
            datasetSuggestions.forEach(suggestion => {
                const btn = document.createElement('button');
                btn.className = 'suggestion-btn';
                btn.innerHTML = `<i class="fas fa-lightbulb"></i> ${suggestion}`;
                btn.onclick = () => this.sendQuickMessage(suggestion);
                suggestionsContainer.appendChild(btn);
            });
        } else {
            // Default suggestions
            this.resetDefaultSuggestions();
        }
    }
    
    resetDefaultSuggestions() {
        const suggestionsContainer = document.querySelector('.suggestions-list');
        const defaultSuggestions = [
            { icon: 'database', text: 'What datasets are available?' },
            { icon: 'chart-bar', text: 'Show me a data summary' },
            { icon: 'chart-area', text: 'Generate visualizations' },
            { icon: 'project-diagram', text: 'Find data correlations' },
            { icon: 'exclamation-triangle', text: 'Detect outliers' }
        ];
        
        suggestionsContainer.innerHTML = '';
        defaultSuggestions.forEach(suggestion => {
            const btn = document.createElement('button');
            btn.className = 'suggestion-btn';
            btn.innerHTML = `<i class="fas fa-${suggestion.icon}"></i> ${suggestion.text}`;
            btn.onclick = () => this.sendQuickMessage(suggestion.text);
            suggestionsContainer.appendChild(btn);
        });
    }
    
    handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
    
    updateSendButtonState() {
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        
        if (input && sendBtn) {
            sendBtn.disabled = !input.value.trim() || this.isTyping;
        }
    }
    
    autoResizeInput() {
        const input = document.getElementById('chat-input');
        if (input) {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        }
    }
    
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message || this.isTyping) return;
        
        // Add user message to chat
        this.addUserMessage(message);
        
        // Clear input
        input.value = '';
        this.updateSendButtonState();
        this.autoResizeInput();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            // Send message to API
            const response = await window.apiClient.post('/api/v1/chat', {
                message: message,
                session_id: this.sessionId,
                dataset_id: this.currentDataset,
                context: {
                    settings: this.settings,
                    previous_messages: this.messageHistory.slice(-5) // Last 5 messages for context
                }
            });
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Add AI response
            this.addAssistantMessage(response.response, response.data_results, response.visualizations);
            
            // Update suggestions if provided
            if (response.suggestions && response.suggestions.length > 0) {
                this.updateDynamicSuggestions(response.suggestions);
            }
            
            // Store message in history
            this.messageHistory.push(
                { role: 'user', content: message, timestamp: new Date() },
                { role: 'assistant', content: response.response, timestamp: new Date() }
            );
            
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            this.addErrorMessage('Sorry, I encountered an error. Please try again.');
        }
    }
    
    sendQuickMessage(message) {
        const input = document.getElementById('chat-input');
        input.value = message;
        this.updateSendButtonState();
        this.sendMessage();
    }
    
    addUserMessage(content) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageElement = this.createMessageElement('user', content);
        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    addAssistantMessage(content, dataResults = null, visualizations = []) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageElement = this.createMessageElement('assistant', content, dataResults, visualizations);
        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    addSystemMessage(content) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageElement = this.createSystemMessageElement(content);
        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    addErrorMessage(content) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageElement = this.createErrorMessageElement(content);
        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    createMessageElement(role, content, dataResults = null, visualizations = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        const avatarIcon = document.createElement('div');
        avatarIcon.className = 'avatar-icon';
        avatarIcon.innerHTML = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        
        avatarDiv.appendChild(avatarIcon);
        messageDiv.appendChild(avatarDiv);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.innerHTML = this.formatMessageContent(content);
        
        bubbleDiv.appendChild(textDiv);
        
        // Add data results if present
        if (dataResults) {
            const dataDiv = this.createDataResultsElement(dataResults);
            bubbleDiv.appendChild(dataDiv);
        }
        
        // Add visualizations if present
        if (visualizations && visualizations.length > 0) {
            const vizDiv = this.createVisualizationsElement(visualizations);
            bubbleDiv.appendChild(vizDiv);
        }
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString();
        
        bubbleDiv.appendChild(timeDiv);
        contentDiv.appendChild(bubbleDiv);
        messageDiv.appendChild(contentDiv);
        
        return messageDiv;
    }
    
    createSystemMessageElement(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system-message';
        messageDiv.innerHTML = `
            <div class="system-message-content">
                <i class="fas fa-info-circle"></i>
                <span>${content}</span>
            </div>
        `;
        return messageDiv;
    }
    
    createErrorMessageElement(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message error-message';
        messageDiv.innerHTML = `
            <div class="error-message-content">
                <i class="fas fa-exclamation-triangle"></i>
                <span>${content}</span>
            </div>
        `;
        return messageDiv;
    }
    
    createDataResultsElement(dataResults) {
        const dataDiv = document.createElement('div');
        dataDiv.className = 'data-results';
        
        if (dataResults.type === 'dataset_info') {
            dataDiv.innerHTML = `
                <div class="data-card">
                    <h4><i class="fas fa-info-circle"></i> Dataset Information</h4>
                    <div class="data-grid">
                        <div class="data-item">
                            <span class="data-label">Rows:</span>
                            <span class="data-value">${dataResults.rows.toLocaleString()}</span>
                        </div>
                        <div class="data-item">
                            <span class="data-label">Columns:</span>
                            <span class="data-value">${dataResults.columns}</span>
                        </div>
                    </div>
                </div>
            `;
        } else if (dataResults.type === 'summary_statistics') {
            dataDiv.innerHTML = this.createSummaryStatsHTML(dataResults.summary);
        } else if (dataResults.type === 'filtered_data') {
            dataDiv.innerHTML = this.createDataTableHTML(dataResults.sample_data);
        }
        
        return dataDiv;
    }
    
    createVisualizationsElement(visualizations) {
        const vizDiv = document.createElement('div');
        vizDiv.className = 'chat-visualizations';
        
        visualizations.forEach((viz, index) => {
            const vizCard = document.createElement('div');
            vizCard.className = 'viz-card';
            vizCard.innerHTML = `
                <div class="viz-header">
                    <h5><i class="fas fa-chart-bar"></i> ${viz.title || 'Visualization'}</h5>
                    <button class="viz-expand-btn" onclick="openVisualizationModal('${viz.file_path}')">
                        <i class="fas fa-expand"></i>
                    </button>
                </div>
                <div class="viz-content">
                    <iframe src="${viz.file_path}" width="100%" height="300" frameborder="0"></iframe>
                </div>
            `;
            vizDiv.appendChild(vizCard);
        });
        
        return vizDiv;
    }
    
    formatMessageContent(content) {
        // Convert markdown-like formatting
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
        content = content.replace(/`(.*?)`/g, '<code>$1</code>');
        content = content.replace(/\n/g, '<br>');
        
        return content;
    }
    
    createSummaryStatsHTML(summary) {
        let html = '<div class="data-card"><h4><i class="fas fa-chart-line"></i> Summary Statistics</h4>';
        
        for (const [column, stats] of Object.entries(summary)) {
            html += `
                <div class="stats-column">
                    <h5>${column}</h5>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-label">Mean:</span>
                            <span class="stat-value">${stats.mean?.toFixed(2) || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Std:</span>
                            <span class="stat-value">${stats.std?.toFixed(2) || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Min:</span>
                            <span class="stat-value">${stats.min || 'N/A'}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Max:</span>
                            <span class="stat-value">${stats.max || 'N/A'}</span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        return html;
    }
    
    createDataTableHTML(data) {
        if (!data || data.length === 0) return '<p>No data to display</p>';
        
        const columns = Object.keys(data[0]);
        let html = '<div class="data-table-container"><table class="data-table">';
        
        // Header
        html += '<thead><tr>';
        columns.forEach(col => {
            html += `<th>${col}</th>`;
        });
        html += '</tr></thead>';
        
        // Body
        html += '<tbody>';
        data.slice(0, 10).forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                html += `<td>${row[col] !== null ? row[col] : 'N/A'}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table>';
        
        if (data.length > 10) {
            html += `<p class="table-note">Showing first 10 of ${data.length} rows</p>`;
        }
        
        html += '</div>';
        return html;
    }
    
    showTypingIndicator() {
        this.isTyping = true;
        const typingDiv = document.getElementById('chat-typing');
        if (typingDiv) {
            typingDiv.style.display = 'block';
        }
        this.updateSendButtonState();
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.isTyping = false;
        const typingDiv = document.getElementById('chat-typing');
        if (typingDiv) {
            typingDiv.style.display = 'none';
        }
        this.updateSendButtonState();
    }
    
    scrollToBottom() {
        const messagesContainer = document.querySelector('.chat-messages-container');
        if (messagesContainer) {
            setTimeout(() => {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }, 100);
        }
    }
    
    updateDynamicSuggestions(suggestions) {
        if (!this.settings.showSuggestions) return;
        
        const suggestionsContainer = document.querySelector('.suggestions-list');
        suggestionsContainer.innerHTML = '';
        
        suggestions.forEach(suggestion => {
            const btn = document.createElement('button');
            btn.className = 'suggestion-btn';
            btn.innerHTML = `<i class="fas fa-magic"></i> ${suggestion}`;
            btn.onclick = () => this.sendQuickMessage(suggestion);
            suggestionsContainer.appendChild(btn);
        });
    }
    
    clearChatHistory() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            const messagesContainer = document.getElementById('chat-messages');
            // Keep only the welcome message
            const welcomeMessage = messagesContainer.querySelector('.welcome-message');
            messagesContainer.innerHTML = '';
            if (welcomeMessage) {
                messagesContainer.appendChild(welcomeMessage);
            }
            
            this.messageHistory = [];
            this.initializeSession();
        }
    }
    
    async downloadChatHistory() {
        try {
            const chatData = {
                session_id: this.sessionId,
                timestamp: new Date().toISOString(),
                dataset: this.currentDataset,
                messages: this.messageHistory
            };
            
            const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `chat_history_${this.sessionId.slice(-8)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
        } catch (error) {
            console.error('Error downloading chat history:', error);
            this.showError('Failed to download chat history');
        }
    }
    
    toggleChatSettings() {
        const settingsPanel = document.getElementById('chat-settings');
        if (settingsPanel) {
            settingsPanel.style.display = settingsPanel.style.display === 'none' ? 'block' : 'none';
        }
    }
    
    updateSettings() {
        const responseStyle = document.getElementById('response-style')?.value;
        const autoVisualizations = document.getElementById('auto-visualizations')?.checked;
        const showSuggestions = document.getElementById('show-suggestions')?.checked;
        
        this.settings = {
            responseStyle: responseStyle || 'concise',
            autoVisualizations: autoVisualizations !== false,
            showSuggestions: showSuggestions !== false
        };
        
        // Save to localStorage
        localStorage.setItem('chatSettings', JSON.stringify(this.settings));
        
        // Update UI based on settings
        if (!this.settings.showSuggestions) {
            document.getElementById('chat-suggestions').style.display = 'none';
        } else {
            document.getElementById('chat-suggestions').style.display = 'block';
        }
    }
    
    loadSettings() {
        try {
            const saved = localStorage.getItem('chatSettings');
            if (saved) {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
                
                // Apply settings to UI
                if (document.getElementById('response-style')) {
                    document.getElementById('response-style').value = this.settings.responseStyle;
                }
                if (document.getElementById('auto-visualizations')) {
                    document.getElementById('auto-visualizations').checked = this.settings.autoVisualizations;
                }
                if (document.getElementById('show-suggestions')) {
                    document.getElementById('show-suggestions').checked = this.settings.showSuggestions;
                }
                
                this.updateSettings();
            }
        } catch (error) {
            console.error('Error loading chat settings:', error);
        }
    }
    
    refreshDatasets() {
        this.loadDatasets();
        this.addSystemMessage('📊 Dataset list refreshed');
    }
    
    showError(message) {
        // Show error notification
        console.error('Chat Interface Error:', message);
    }
}

// Global functions for HTML event handlers
window.sendChatMessage = () => window.chatInterface?.sendMessage();
window.sendQuickMessage = (message) => window.chatInterface?.sendQuickMessage(message);
window.updateChatContext = () => window.chatInterface?.updateChatContext();
window.clearChatHistory = () => window.chatInterface?.clearChatHistory();
window.downloadChatHistory = () => window.chatInterface?.downloadChatHistory();
window.toggleChatSettings = () => window.chatInterface?.toggleChatSettings();
window.refreshDatasets = () => window.chatInterface?.refreshDatasets();
window.handleChatKeyDown = (event) => window.chatInterface?.handleKeyDown(event);
window.autoResizeChatInput = (input) => window.chatInterface?.autoResizeInput();

// Additional utility functions
window.attachFile = () => {
    console.log('File attachment feature not yet implemented');
};

window.toggleVoiceInput = () => {
    console.log('Voice input feature not yet implemented');
};

window.openVisualizationModal = (filePath) => {
    // Open visualization in a modal or new window
    window.open(filePath, '_blank', 'width=800,height=600');
};

// Initialize chat interface when the AI chat section is active
document.addEventListener('DOMContentLoaded', () => {
    // Initialize only if we're on the chat section
    if (document.getElementById('ai-chat-section')) {
        window.chatInterface = new ChatInterface();
    }
});

// Initialize when switching to chat section
document.addEventListener('sectionChanged', (event) => {
    if (event.detail.section === 'ai-chat') {
        if (!window.chatInterface) {
            window.chatInterface = new ChatInterface();
        }
    }
});
