/**
 * AI Chat System for Auto Analytics Platform
 * Handles natural language queries and AI-powered assistance
 */

class AIChatSystem {
    constructor() {
        this.isOpen = false;
        this.messages = [];
        this.isTyping = false;
        this.context = {
            currentDataset: null,
            currentAnalysis: null,
            userPreferences: {}
        };
        this.init();
    }

    init() {
        console.log('Initializing AI Chat System...');
        this.setupEventListeners();
        this.loadChatHistory();
        this.initializeNLP();
    }

    setupEventListeners() {
        // Chat input handling
        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // Chat toggle
        document.addEventListener('click', (e) => {
            if (e.target.matches('.chat-toggle, .chat-toggle *')) {
                this.toggleChat();
            }
        });

        // Suggestion buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('.suggestion-btn')) {
                const question = e.target.textContent.trim();
                this.askQuestion(question);
            }
        });
    }

    toggleChat() {
        const modal = document.getElementById('ai-chat-modal');
        if (modal) {
            this.isOpen = !this.isOpen;
            modal.classList.toggle('active', this.isOpen);
            
            if (this.isOpen) {
                this.focusChatInput();
                this.loadWelcomeMessage();
            }
        }
    }

    askQuestion(question) {
        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
            chatInput.value = question;
            this.toggleChat();
            setTimeout(() => this.sendMessage(), 300);
        }
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;

        // Add user message
        this.addMessage(message, 'user');
        input.value = '';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Process the message with NLP
            const response = await this.processNaturalLanguageQuery(message);
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Add AI response
            this.addMessage(response.text, 'ai', response.data);
            
            // Execute any actions if needed
            if (response.actions) {
                await this.executeActions(response.actions);
            }

        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage(
                "I apologize, but I encountered an error processing your request. Please try rephrasing your question.",
                'ai'
            );
            console.error('Chat Error:', error);
        }
    }

    addMessage(text, sender, data = null) {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}-message`;
        
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-${sender === 'user' ? 'user' : 'robot'}"></i>
            </div>
            <div class="message-content">
                <div class="message-text">${this.formatMessageText(text)}</div>
                ${data ? this.renderMessageData(data) : ''}
                <div class="message-timestamp">${timestamp}</div>
                ${sender === 'ai' ? this.renderMessageActions() : ''}
            </div>
        `;

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Store message
        this.messages.push({
            text: text,
            sender: sender,
            timestamp: Date.now(),
            data: data
        });

        // Auto-save chat history
        this.saveChatHistory();
    }

    formatMessageText(text) {
        // Handle markdown-like formatting
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    renderMessageData(data) {
        if (!data) return '';

        let html = '';

        // Render charts
        if (data.chart) {
            html += `
                <div class="message-chart">
                    <div id="chart-${generateId()}" class="embedded-chart"></div>
                </div>
            `;
        }

        // Render tables
        if (data.table) {
            html += this.renderDataTable(data.table);
        }

        // Render insights
        if (data.insights) {
            html += `
                <div class="message-insights">
                    <h4><i class="fas fa-lightbulb"></i> Key Insights</h4>
                    <ul>
                        ${data.insights.map(insight => `<li>${insight}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Render recommendations
        if (data.recommendations) {
            html += `
                <div class="message-recommendations">
                    <h4><i class="fas fa-thumbs-up"></i> Recommendations</h4>
                    <ul>
                        ${data.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        return html;
    }

    renderDataTable(tableData) {
        const headers = Object.keys(tableData[0] || {});
        
        return `
            <div class="message-table">
                <table class="data-table-small">
                    <thead>
                        <tr>
                            ${headers.map(header => `<th>${header}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${tableData.slice(0, 5).map(row => `
                            <tr>
                                ${headers.map(header => `<td>${row[header] || '-'}</td>`).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
                ${tableData.length > 5 ? `<div class="table-footer">Showing 5 of ${tableData.length} rows</div>` : ''}
            </div>
        `;
    }

    renderMessageActions() {
        return `
            <div class="message-actions">
                <button class="action-btn" onclick="aiChat.copyMessage(this)">
                    <i class="fas fa-copy"></i>
                </button>
                <button class="action-btn" onclick="aiChat.likeMessage(this)">
                    <i class="fas fa-thumbs-up"></i>
                </button>
                <button class="action-btn" onclick="aiChat.shareMessage(this)">
                    <i class="fas fa-share"></i>
                </button>
            </div>
        `;
    }

    showTypingIndicator() {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;

        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message ai-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-animation">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        this.isTyping = true;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        this.isTyping = false;
    }

    async processNaturalLanguageQuery(message) {
        // Analyze the intent and entities
        const analysis = await this.analyzeIntent(message);
        
        // Generate appropriate response based on intent
        switch (analysis.intent) {
            case 'data_overview':
                return await this.handleDataOverviewQuery(analysis);
            case 'statistical_analysis':
                return await this.handleStatisticalQuery(analysis);
            case 'visualization_request':
                return await this.handleVisualizationQuery(analysis);
            case 'prediction_request':
                return await this.handlePredictionQuery(analysis);
            case 'model_training':
                return await this.handleModelTrainingQuery(analysis);
            case 'data_quality':
                return await this.handleDataQualityQuery(analysis);
            case 'comparison':
                return await this.handleComparisonQuery(analysis);
            case 'trend_analysis':
                return await this.handleTrendAnalysisQuery(analysis);
            case 'help':
                return await this.handleHelpQuery(analysis);
            default:
                return await this.handleGeneralQuery(analysis);
        }
    }

    async analyzeIntent(message) {
        // Simple intent classification (in production, this would use a trained NLP model)
        const intents = {
            data_overview: ['overview', 'summary', 'describe', 'show me', 'what is'],
            statistical_analysis: ['mean', 'average', 'median', 'correlation', 'statistics', 'std', 'variance'],
            visualization_request: ['plot', 'chart', 'graph', 'visualize', 'show chart'],
            prediction_request: ['predict', 'forecast', 'future', 'estimate', 'what will'],
            model_training: ['train', 'model', 'machine learning', 'ml', 'build model'],
            data_quality: ['quality', 'missing', 'outliers', 'duplicates', 'clean'],
            comparison: ['compare', 'difference', 'vs', 'versus', 'better'],
            trend_analysis: ['trend', 'pattern', 'over time', 'temporal', 'seasonal'],
            help: ['help', 'how to', 'tutorial', 'guide', 'assist']
        };

        const entities = this.extractEntities(message);
        let intent = 'general';
        let confidence = 0;

        // Find best matching intent
        for (const [intentName, keywords] of Object.entries(intents)) {
            const score = keywords.reduce((acc, keyword) => {
                return acc + (message.toLowerCase().includes(keyword) ? 1 : 0);
            }, 0) / keywords.length;
            
            if (score > confidence) {
                confidence = score;
                intent = intentName;
            }
        }

        return {
            intent: intent,
            confidence: confidence,
            entities: entities,
            originalMessage: message
        };
    }

    extractEntities(message) {
        // Extract column names, numbers, and other entities
        const entities = {
            columns: [],
            numbers: [],
            dates: [],
            datasets: []
        };

        // Extract numbers
        const numbers = message.match(/\d+\.?\d*/g);
        if (numbers) {
            entities.numbers = numbers.map(n => parseFloat(n));
        }

        // Extract potential column names (words in quotes or common column patterns)
        const quotedWords = message.match(/"([^"]+)"|'([^']+)'/g);
        if (quotedWords) {
            entities.columns = quotedWords.map(w => w.replace(/['"]/g, ''));
        }

        // Extract common column patterns
        const columnPatterns = ['price', 'sales', 'revenue', 'cost', 'amount', 'date', 'time', 'category', 'type'];
        columnPatterns.forEach(pattern => {
            if (message.toLowerCase().includes(pattern)) {
                entities.columns.push(pattern);
            }
        });

        return entities;
    }

    async handleDataOverviewQuery(analysis) {
        try {
            // Get current dataset info
            const datasetId = this.context.currentDataset || 'sample_dataset';
            const response = await apiClient.get(`/datasets/${datasetId}`);
            
            if (response.success) {
                const data = response.data;
                return {
                    text: `Here's an overview of your dataset:\n\n**${data.metadata.filename}**\n\n• **Rows:** ${data.metadata.rows.toLocaleString()}\n• **Columns:** ${data.metadata.columns}\n• **Data Quality Score:** ${data.basic_statistics?.quality_score || 'N/A'}\n• **Upload Date:** ${new Date(data.metadata.upload_timestamp).toLocaleDateString()}\n\nThe dataset contains ${data.metadata.columns} features with ${data.metadata.rows.toLocaleString()} observations. Would you like me to show you specific statistics or create visualizations?`,
                    data: {
                        table: data.sample_data.slice(0, 5),
                        insights: [
                            `Dataset has ${data.metadata.rows.toLocaleString()} rows and ${data.metadata.columns} columns`,
                            `Data quality appears ${data.basic_statistics?.quality_score > 80 ? 'good' : 'moderate'}`,
                            `${data.metadata.missing_values_percentage.toFixed(1)}% missing values overall`
                        ]
                    }
                };
            }
        } catch (error) {
            console.error('Data overview error:', error);
        }

        return {
            text: "I don't have access to a specific dataset right now. Please upload a dataset first, and I'll be happy to provide an overview!"
        };
    }

    async handleStatisticalQuery(analysis) {
        const columns = analysis.entities.columns;
        
        if (columns.length === 0) {
            return {
                text: "I'd be happy to provide statistical analysis! Could you specify which columns you're interested in? For example, 'show me statistics for sales_amount' or 'calculate the mean of price'."
            };
        }

        try {
            const datasetId = this.context.currentDataset || 'sample_dataset';
            const response = await apiClient.post(`/analysis/quick/${datasetId}`, {
                columns: columns,
                statistics: ['mean', 'median', 'std', 'min', 'max', 'quantiles']
            });

            if (response.success) {
                const stats = response.data.statistics;
                let text = `Here are the statistics for ${columns.join(', ')}:\n\n`;
                
                columns.forEach(col => {
                    if (stats[col]) {
                        text += `**${col}:**\n`;
                        text += `• Mean: ${stats[col].mean?.toFixed(2) || 'N/A'}\n`;
                        text += `• Median: ${stats[col].median?.toFixed(2) || 'N/A'}\n`;
                        text += `• Std Dev: ${stats[col].std?.toFixed(2) || 'N/A'}\n`;
                        text += `• Range: ${stats[col].min?.toFixed(2) || 'N/A'} - ${stats[col].max?.toFixed(2) || 'N/A'}\n\n`;
                    }
                });

                return {
                    text: text,
                    data: {
                        chart: {
                            type: 'histogram',
                            column: columns[0]
                        },
                        insights: [
                            `${columns[0]} has a mean of ${stats[columns[0]]?.mean?.toFixed(2) || 'N/A'}`,
                            `The distribution shows ${stats[columns[0]]?.skew > 0 ? 'right' : 'left'} skewness`,
                            `Standard deviation indicates ${stats[columns[0]]?.std > stats[columns[0]]?.mean ? 'high' : 'moderate'} variability`
                        ]
                    }
                };
            }
        } catch (error) {
            console.error('Statistical analysis error:', error);
        }

        return {
            text: "I apologize, but I couldn't retrieve the statistical information right now. Please make sure you have a dataset loaded and try again."
        };
    }

    async handleVisualizationQuery(analysis) {
        const columns = analysis.entities.columns;
        
        return {
            text: `I'll create a visualization for you! ${columns.length > 0 ? `I'll focus on the ${columns.join(', ')} column(s).` : 'Let me suggest the best chart type based on your data.'}`,
            data: {
                chart: {
                    type: 'auto',
                    columns: columns
                }
            },
            actions: [
                {
                    type: 'create_visualization',
                    params: { columns: columns }
                }
            ]
        };
    }

    async handlePredictionQuery(analysis) {
        return {
            text: "I can help you make predictions! Let me set up a prediction model based on your data. This will involve training a machine learning model on your dataset.",
            data: {
                insights: [
                    "I'll analyze your data to determine the best prediction approach",
                    "Multiple algorithms will be tested to find the most accurate model",
                    "You'll get confidence intervals and prediction explanations"
                ]
            },
            actions: [
                {
                    type: 'start_prediction_workflow',
                    params: analysis.entities
                }
            ]
        };
    }

    async handleModelTrainingQuery(analysis) {
        return {
            text: "I'll help you train a machine learning model! I can set up AutoML to automatically find the best model for your data, or we can build a custom neural network together.",
            data: {
                recommendations: [
                    "Start with AutoML for quick, high-quality results",
                    "Use custom neural networks for specialized tasks",
                    "Consider ensemble methods for maximum accuracy"
                ]
            },
            actions: [
                {
                    type: 'open_automl_wizard',
                    params: analysis.entities
                }
            ]
        };
    }

    async handleHelpQuery(analysis) {
        const helpTopics = {
            "data upload": "Upload CSV, Excel, JSON, or Parquet files using the drag-and-drop interface.",
            "analysis": "Ask me questions like 'show me statistics for sales' or 'create a chart of revenue over time'.",
            "models": "I can help you train machine learning models with AutoML or build custom neural networks.",
            "predictions": "Request predictions by saying 'predict future sales' or 'forecast next quarter revenue'.",
            "visualizations": "Ask for charts like 'plot sales by category' or 'show correlation heatmap'."
        };

        return {
            text: `I'm here to help! Here are some things you can ask me about:\n\n${Object.entries(helpTopics).map(([topic, desc]) => `**${topic.charAt(0).toUpperCase() + topic.slice(1)}:** ${desc}`).join('\n\n')}\n\nJust ask me questions in natural language, and I'll do my best to help!`,
            data: {
                insights: [
                    "I understand natural language queries",
                    "I can perform complex data analysis automatically",
                    "I learn from your interactions to provide better assistance"
                ]
            }
        };
    }

    async handleGeneralQuery(analysis) {
        const responses = [
            "I'm here to help with your data analysis needs! Could you be more specific about what you'd like to know?",
            "I can assist with data analysis, visualizations, machine learning, and predictions. What would you like to explore?",
            "Let me know how I can help with your data! I can create charts, run analysis, train models, and much more."
        ];

        return {
            text: responses[Math.floor(Math.random() * responses.length)]
        };
    }

    async executeActions(actions) {
        for (const action of actions) {
            try {
                switch (action.type) {
                    case 'create_visualization':
                        await this.createVisualization(action.params);
                        break;
                    case 'start_prediction_workflow':
                        await this.startPredictionWorkflow(action.params);
                        break;
                    case 'open_automl_wizard':
                        await this.openAutoMLWizard(action.params);
                        break;
                }
            } catch (error) {
                console.error(`Action execution error (${action.type}):`, error);
            }
        }
    }

    async createVisualization(params) {
        // Implementation for creating visualizations
        showToast('Creating visualization...', 'info');
    }

    async startPredictionWorkflow(params) {
        // Implementation for prediction workflow
        showToast('Starting prediction workflow...', 'info');
    }

    async openAutoMLWizard(params) {
        // Implementation for AutoML wizard
        showToast('Opening AutoML wizard...', 'info');
    }

    loadWelcomeMessage() {
        if (this.messages.length === 0) {
            this.addMessage(
                "Hello! I'm your AI analytics assistant. I can help you analyze data, create visualizations, train machine learning models, and answer questions about your datasets. What would you like to explore today?",
                'ai',
                {
                    insights: [
                        "Ask me questions in plain English",
                        "I can analyze any dataset you upload",
                        "I'll suggest the best visualizations and models for your data"
                    ]
                }
            );
        }
    }

    focusChatInput() {
        setTimeout(() => {
            const input = document.getElementById('chat-input');
            if (input) input.focus();
        }, 100);
    }

    saveChatHistory() {
        localStorage.setItem('ai_chat_history', JSON.stringify(this.messages));
    }

    loadChatHistory() {
        const saved = localStorage.getItem('ai_chat_history');
        if (saved) {
            this.messages = JSON.parse(saved);
            this.renderChatHistory();
        }
    }

    renderChatHistory() {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer || this.messages.length === 0) return;

        messagesContainer.innerHTML = '';
        this.messages.forEach(message => {
            this.addMessage(message.text, message.sender, message.data);
        });
    }

    initializeNLP() {
        // Initialize any NLP libraries or models
        console.log('NLP system initialized');
    }

    // Message action handlers
    copyMessage(button) {
        const messageText = button.closest('.message-content').querySelector('.message-text').textContent;
        copyToClipboard(messageText);
        showToast('Message copied to clipboard', 'success');
    }

    likeMessage(button) {
        button.classList.toggle('liked');
        button.querySelector('i').className = button.classList.contains('liked') ? 'fas fa-thumbs-up' : 'far fa-thumbs-up';
        showToast(button.classList.contains('liked') ? 'Thanks for the feedback!' : 'Feedback removed', 'info');
    }

    shareMessage(button) {
        const messageText = button.closest('.message-content').querySelector('.message-text').textContent;
        if (navigator.share) {
            navigator.share({
                title: 'AI Analytics Insight',
                text: messageText
            });
        } else {
            copyToClipboard(`AI Analytics Insight: ${messageText}`);
            showToast('Message copied for sharing', 'success');
        }
    }
}

// Initialize AI Chat System
const aiChat = new AIChatSystem();

// Export for global access
window.aiChat = aiChat;
