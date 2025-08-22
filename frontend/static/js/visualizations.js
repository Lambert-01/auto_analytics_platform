/**
 * Advanced Visualizations System for Auto Analytics Platform
 * Handles creation and management of interactive charts and 3D visualizations
 */

class AdvancedVisualizationSystem {
    constructor() {
        this.charts = new Map();
        this.chartCounter = 0;
        this.defaultConfig = {
            responsive: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#667eea',
                    borderWidth: 1
                }
            }
        };
        this.init();
    }

    init() {
        console.log('Initializing Advanced Visualization System...');
        this.setupEventListeners();
        this.initializeChartThemes();
        this.setupChartControls();
    }

    setupEventListeners() {
        // Visualization creation buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-viz-type]')) {
                const vizType = e.target.dataset.vizType;
                const params = e.target.dataset.vizParams ? JSON.parse(e.target.dataset.vizParams) : {};
                this.createVisualization(vizType, params);
            }
        });

        // Chart export buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('.export-chart')) {
                const chartId = e.target.dataset.chartId;
                const format = e.target.dataset.format || 'png';
                this.exportChart(chartId, format);
            }
        });

        // Chart fullscreen toggle
        document.addEventListener('click', (e) => {
            if (e.target.matches('.fullscreen-chart')) {
                const chartId = e.target.dataset.chartId;
                this.toggleFullscreen(chartId);
            }
        });
    }

    initializeChartThemes() {
        this.themes = {
            default: {
                backgroundColor: '#ffffff',
                borderColor: '#667eea',
                pointBackgroundColor: '#667eea',
                gridColor: '#e2e8f0'
            },
            dark: {
                backgroundColor: '#1a202c',
                borderColor: '#4299e1',
                pointBackgroundColor: '#4299e1',
                gridColor: '#2d3748'
            },
            neon: {
                backgroundColor: '#0f0f23',
                borderColor: '#00d4ff',
                pointBackgroundColor: '#00d4ff',
                gridColor: '#1a1a2e'
            }
        };

        this.currentTheme = 'default';
    }

    setupChartControls() {
        // Create chart control panel
        const controlPanel = document.createElement('div');
        controlPanel.className = 'chart-control-panel';
        controlPanel.innerHTML = `
            <div class="control-header">
                <h3>Visualization Controls</h3>
                <button class="control-toggle">
                    <i class="fas fa-cog"></i>
                </button>
            </div>
            <div class="control-content">
                <div class="control-group">
                    <label>Theme:</label>
                    <select id="chart-theme-select">
                        <option value="default">Default</option>
                        <option value="dark">Dark</option>
                        <option value="neon">Neon</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Animation:</label>
                    <input type="checkbox" id="chart-animation" checked>
                </div>
                <div class="control-group">
                    <label>Grid:</label>
                    <input type="checkbox" id="chart-grid" checked>
                </div>
                <div class="control-group">
                    <button onclick="vizSystem.exportAllCharts()">Export All</button>
                    <button onclick="vizSystem.clearAllCharts()">Clear All</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(controlPanel);
    }

    async createVisualization(type, params = {}) {
        try {
            showToast(`Creating ${type} visualization...`, 'info');

            const chartId = `chart_${++this.chartCounter}`;
            const container = this.createChartContainer(chartId, type);
            
            // Get data for visualization
            let data = params.data;
            if (!data && params.datasetId) {
                data = await this.fetchDataForVisualization(params.datasetId, params);
            }

            if (!data) {
                throw new Error('No data available for visualization');
            }

            // Create the specific visualization
            let chart;
            switch (type) {
                case 'scatter3d':
                    chart = await this.createScatter3D(chartId, data, params);
                    break;
                case 'heatmap':
                    chart = await this.createHeatmap(chartId, data, params);
                    break;
                case 'violin':
                    chart = await this.createViolinPlot(chartId, data, params);
                    break;
                case 'network':
                    chart = await this.createNetworkGraph(chartId, data, params);
                    break;
                case 'sankey':
                    chart = await this.createSankeyDiagram(chartId, data, params);
                    break;
                case 'treemap':
                    chart = await this.createTreemap(chartId, data, params);
                    break;
                case 'candlestick':
                    chart = await this.createCandlestickChart(chartId, data, params);
                    break;
                case 'radar':
                    chart = await this.createRadarChart(chartId, data, params);
                    break;
                case 'waterfall':
                    chart = await this.createWaterfallChart(chartId, data, params);
                    break;
                case 'funnel':
                    chart = await this.createFunnelChart(chartId, data, params);
                    break;
                default:
                    chart = await this.createBasicChart(chartId, data, type, params);
            }

            this.charts.set(chartId, {
                chart: chart,
                type: type,
                data: data,
                params: params
            });

            showToast(`${type} visualization created successfully!`, 'success');
            return chartId;

        } catch (error) {
            console.error('Visualization creation error:', error);
            showToast(`Failed to create ${type} visualization: ${error.message}`, 'error');
            throw error;
        }
    }

    createChartContainer(chartId, type) {
        const container = document.createElement('div');
        container.className = 'advanced-chart-container';
        container.innerHTML = `
            <div class="chart-header">
                <h3 class="chart-title">${this.formatChartTitle(type)}</h3>
                <div class="chart-actions">
                    <button class="chart-btn" data-chart-id="${chartId}" onclick="vizSystem.refreshChart('${chartId}')">
                        <i class="fas fa-sync"></i>
                    </button>
                    <button class="chart-btn fullscreen-chart" data-chart-id="${chartId}">
                        <i class="fas fa-expand"></i>
                    </button>
                    <div class="chart-export-dropdown">
                        <button class="chart-btn">
                            <i class="fas fa-download"></i>
                        </button>
                        <div class="export-menu">
                            <button class="export-chart" data-chart-id="${chartId}" data-format="png">PNG</button>
                            <button class="export-chart" data-chart-id="${chartId}" data-format="jpg">JPG</button>
                            <button class="export-chart" data-chart-id="${chartId}" data-format="svg">SVG</button>
                            <button class="export-chart" data-chart-id="${chartId}" data-format="pdf">PDF</button>
                        </div>
                    </div>
                    <button class="chart-btn" onclick="vizSystem.removeChart('${chartId}')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
            <div class="chart-body">
                <div id="${chartId}" class="chart-canvas"></div>
                <div class="chart-loading hidden">
                    <i class="fas fa-spinner fa-spin"></i>
                    <span>Generating visualization...</span>
                </div>
            </div>
            <div class="chart-footer">
                <div class="chart-info">
                    <span class="chart-type">${type}</span>
                    <span class="chart-points">Data points: <span id="${chartId}-points">-</span></span>
                </div>
                <div class="chart-controls">
                    <!-- Chart-specific controls will be added here -->
                </div>
            </div>
        `;

        // Add to visualization gallery or dashboard
        const gallery = document.querySelector('.viz-gallery') || document.body;
        gallery.appendChild(container);

        return container;
    }

    formatChartTitle(type) {
        return type.split(/(?=[A-Z])/).map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    async fetchDataForVisualization(datasetId, params) {
        try {
            const response = await apiClient.get(`/datasets/${datasetId}/sample`, {
                limit: params.sampleSize || 1000,
                columns: params.columns
            });

            if (response.success) {
                return response.data.data;
            }
            throw new Error(response.error);
        } catch (error) {
            console.error('Data fetch error:', error);
            throw error;
        }
    }

    // 3D Scatter Plot
    async createScatter3D(chartId, data, params) {
        const xCol = params.xColumn || Object.keys(data[0])[0];
        const yCol = params.yColumn || Object.keys(data[0])[1];
        const zCol = params.zColumn || Object.keys(data[0])[2];
        const colorCol = params.colorColumn;

        const trace = {
            x: data.map(row => row[xCol]),
            y: data.map(row => row[yCol]),
            z: data.map(row => row[zCol]),
            mode: 'markers',
            marker: {
                size: 5,
                color: colorCol ? data.map(row => row[colorCol]) : '#667eea',
                colorscale: 'Viridis',
                showscale: !!colorCol
            },
            type: 'scatter3d',
            name: '3D Scatter'
        };

        const layout = {
            scene: {
                xaxis: { title: xCol },
                yaxis: { title: yCol },
                zaxis: { title: zCol }
            },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };

        return Plotly.newPlot(chartId, [trace], layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    // Advanced Heatmap
    async createHeatmap(chartId, data, params) {
        // Calculate correlation matrix if not provided
        const numericColumns = this.getNumericColumns(data);
        const correlationMatrix = this.calculateCorrelationMatrix(data, numericColumns);

        const trace = {
            z: correlationMatrix.values,
            x: correlationMatrix.columns,
            y: correlationMatrix.columns,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmid: 0,
            showscale: true,
            hoverongaps: false
        };

        const layout = {
            title: 'Correlation Heatmap',
            xaxis: { side: 'bottom' },
            yaxis: { autorange: 'reversed' },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };

        return Plotly.newPlot(chartId, [trace], layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    // Violin Plot
    async createViolinPlot(chartId, data, params) {
        const yCol = params.yColumn || this.getNumericColumns(data)[0];
        const groupCol = params.groupColumn;

        let traces;
        if (groupCol) {
            const groups = [...new Set(data.map(row => row[groupCol]))];
            traces = groups.map(group => ({
                y: data.filter(row => row[groupCol] === group).map(row => row[yCol]),
                type: 'violin',
                name: group,
                box: { visible: true },
                meanline: { visible: true }
            }));
        } else {
            traces = [{
                y: data.map(row => row[yCol]),
                type: 'violin',
                name: yCol,
                box: { visible: true },
                meanline: { visible: true }
            }];
        }

        const layout = {
            title: `Distribution of ${yCol}`,
            yaxis: { title: yCol },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };

        return Plotly.newPlot(chartId, traces, layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    // Network Graph
    async createNetworkGraph(chartId, data, params) {
        // Convert data to network format
        const nodes = this.extractNodes(data, params);
        const edges = this.extractEdges(data, params);

        const nodeTrace = {
            x: nodes.map(n => n.x),
            y: nodes.map(n => n.y),
            mode: 'markers+text',
            marker: {
                size: nodes.map(n => n.size || 10),
                color: nodes.map(n => n.color || '#667eea'),
                line: { width: 2, color: '#ffffff' }
            },
            text: nodes.map(n => n.label),
            textposition: 'middle center',
            type: 'scatter',
            name: 'Nodes'
        };

        const edgeTraces = edges.map(edge => ({
            x: [edge.x0, edge.x1, null],
            y: [edge.y0, edge.y1, null],
            mode: 'lines',
            line: { width: edge.width || 1, color: edge.color || '#cccccc' },
            type: 'scatter',
            showlegend: false
        }));

        const layout = {
            title: 'Network Graph',
            showlegend: false,
            xaxis: { showgrid: false, zeroline: false, showticklabels: false },
            yaxis: { showgrid: false, zeroline: false, showticklabels: false },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };

        return Plotly.newPlot(chartId, [nodeTrace, ...edgeTraces], layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    // Sankey Diagram
    async createSankeyDiagram(chartId, data, params) {
        const sourceCol = params.sourceColumn || 'source';
        const targetCol = params.targetColumn || 'target';
        const valueCol = params.valueColumn || 'value';

        // Extract unique nodes
        const sources = [...new Set(data.map(row => row[sourceCol]))];
        const targets = [...new Set(data.map(row => row[targetCol]))];
        const allNodes = [...new Set([...sources, ...targets])];

        const trace = {
            type: 'sankey',
            node: {
                pad: 15,
                thickness: 30,
                line: { color: 'black', width: 0.5 },
                label: allNodes,
                color: allNodes.map((_, i) => `hsl(${(i * 360) / allNodes.length}, 70%, 60%)`)
            },
            link: {
                source: data.map(row => allNodes.indexOf(row[sourceCol])),
                target: data.map(row => allNodes.indexOf(row[targetCol])),
                value: data.map(row => row[valueCol])
            }
        };

        const layout = {
            title: 'Flow Diagram',
            font: { size: 10 },
            paper_bgcolor: 'transparent'
        };

        return Plotly.newPlot(chartId, [trace], layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    // Treemap
    async createTreemap(chartId, data, params) {
        const labelCol = params.labelColumn || 'label';
        const valueCol = params.valueColumn || 'value';
        const parentCol = params.parentColumn || 'parent';

        const trace = {
            type: 'treemap',
            labels: data.map(row => row[labelCol]),
            values: data.map(row => row[valueCol]),
            parents: data.map(row => row[parentCol] || ''),
            textinfo: 'label+value',
            textfont: { size: 12 },
            marker: {
                colorscale: 'Viridis',
                showscale: true
            }
        };

        const layout = {
            title: 'Hierarchical Data Visualization',
            paper_bgcolor: 'transparent'
        };

        return Plotly.newPlot(chartId, [trace], layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    // Candlestick Chart
    async createCandlestickChart(chartId, data, params) {
        const dateCol = params.dateColumn || 'date';
        const openCol = params.openColumn || 'open';
        const highCol = params.highColumn || 'high';
        const lowCol = params.lowColumn || 'low';
        const closeCol = params.closeColumn || 'close';

        const trace = {
            x: data.map(row => row[dateCol]),
            open: data.map(row => row[openCol]),
            high: data.map(row => row[highCol]),
            low: data.map(row => row[lowCol]),
            close: data.map(row => row[closeCol]),
            type: 'candlestick',
            name: 'OHLC'
        };

        const layout = {
            title: 'Candlestick Chart',
            xaxis: { type: 'date' },
            yaxis: { title: 'Price' },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };

        return Plotly.newPlot(chartId, [trace], layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    // Radar Chart
    async createRadarChart(chartId, data, params) {
        const categories = params.categories || Object.keys(data[0]).filter(key => typeof data[0][key] === 'number');
        const groupCol = params.groupColumn;

        let traces;
        if (groupCol) {
            const groups = [...new Set(data.map(row => row[groupCol]))];
            traces = groups.map(group => {
                const groupData = data.filter(row => row[groupCol] === group);
                const avgValues = categories.map(cat => 
                    groupData.reduce((sum, row) => sum + (row[cat] || 0), 0) / groupData.length
                );
                
                return {
                    type: 'scatterpolar',
                    r: [...avgValues, avgValues[0]],
                    theta: [...categories, categories[0]],
                    fill: 'toself',
                    name: group
                };
            });
        } else {
            traces = [{
                type: 'scatterpolar',
                r: [...categories.map(cat => data[0][cat]), data[0][categories[0]]],
                theta: [...categories, categories[0]],
                fill: 'toself',
                name: 'Data'
            }];
        }

        const layout = {
            polar: {
                radialaxis: { visible: true }
            },
            title: 'Radar Chart',
            paper_bgcolor: 'transparent'
        };

        return Plotly.newPlot(chartId, traces, layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    // Waterfall Chart
    async createWaterfallChart(chartId, data, params) {
        const categoryCol = params.categoryColumn || 'category';
        const valueCol = params.valueColumn || 'value';

        const trace = {
            type: 'waterfall',
            x: data.map(row => row[categoryCol]),
            y: data.map(row => row[valueCol]),
            connector: { line: { color: '#667eea' } },
            increasing: { marker: { color: '#10b981' } },
            decreasing: { marker: { color: '#ef4444' } },
            totals: { marker: { color: '#6366f1' } }
        };

        const layout = {
            title: 'Waterfall Chart',
            xaxis: { title: categoryCol },
            yaxis: { title: valueCol },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };

        return Plotly.newPlot(chartId, [trace], layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    // Funnel Chart
    async createFunnelChart(chartId, data, params) {
        const labelCol = params.labelColumn || 'stage';
        const valueCol = params.valueColumn || 'value';

        const trace = {
            type: 'funnel',
            y: data.map(row => row[labelCol]),
            x: data.map(row => row[valueCol]),
            textinfo: 'value+percent initial'
        };

        const layout = {
            title: 'Funnel Chart',
            paper_bgcolor: 'transparent'
        };

        return Plotly.newPlot(chartId, [trace], layout, {
            responsive: true,
            displayModeBar: true
        });
    }

    // Utility Functions
    getNumericColumns(data) {
        if (!data || data.length === 0) return [];
        
        const firstRow = data[0];
        return Object.keys(firstRow).filter(key => 
            typeof firstRow[key] === 'number' && !isNaN(firstRow[key])
        );
    }

    calculateCorrelationMatrix(data, columns) {
        const n = columns.length;
        const values = Array(n).fill().map(() => Array(n).fill(0));
        
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i === j) {
                    values[i][j] = 1;
                } else {
                    values[i][j] = this.calculateCorrelation(
                        data.map(row => row[columns[i]]),
                        data.map(row => row[columns[j]])
                    );
                }
            }
        }

        return { values, columns };
    }

    calculateCorrelation(x, y) {
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        return denominator === 0 ? 0 : numerator / denominator;
    }

    extractNodes(data, params) {
        // Simple node extraction - customize based on data structure
        const nodeCol = params.nodeColumn || 'node';
        const nodes = [...new Set(data.map(row => row[nodeCol]))];
        
        return nodes.map((node, i) => ({
            id: node,
            label: node,
            x: Math.cos(2 * Math.PI * i / nodes.length),
            y: Math.sin(2 * Math.PI * i / nodes.length),
            size: 10,
            color: `hsl(${(i * 360) / nodes.length}, 70%, 60%)`
        }));
    }

    extractEdges(data, params) {
        // Simple edge extraction - customize based on data structure
        const sourceCol = params.sourceColumn || 'source';
        const targetCol = params.targetColumn || 'target';
        const nodes = this.extractNodes(data, params);
        
        return data.map(row => {
            const sourceNode = nodes.find(n => n.id === row[sourceCol]);
            const targetNode = nodes.find(n => n.id === row[targetCol]);
            
            return {
                x0: sourceNode?.x || 0,
                y0: sourceNode?.y || 0,
                x1: targetNode?.x || 0,
                y1: targetNode?.y || 0,
                width: 1,
                color: '#cccccc'
            };
        });
    }

    // Chart Management Functions
    refreshChart(chartId) {
        const chartData = this.charts.get(chartId);
        if (chartData) {
            this.createVisualization(chartData.type, chartData.params);
        }
    }

    removeChart(chartId) {
        const chartElement = document.getElementById(chartId);
        if (chartElement) {
            chartElement.closest('.advanced-chart-container').remove();
        }
        this.charts.delete(chartId);
    }

    exportChart(chartId, format) {
        const chartElement = document.getElementById(chartId);
        if (chartElement) {
            Plotly.downloadImage(chartElement, {
                format: format,
                width: 1200,
                height: 800,
                filename: `chart_${chartId}`
            });
            showToast(`Chart exported as ${format.toUpperCase()}`, 'success');
        }
    }

    toggleFullscreen(chartId) {
        const container = document.getElementById(chartId).closest('.advanced-chart-container');
        container.classList.toggle('fullscreen');
        
        // Resize chart after fullscreen toggle
        setTimeout(() => {
            Plotly.Plots.resize(chartId);
        }, 100);
    }

    exportAllCharts() {
        this.charts.forEach((_, chartId) => {
            this.exportChart(chartId, 'png');
        });
        showToast(`Exported ${this.charts.size} charts`, 'success');
    }

    clearAllCharts() {
        this.charts.forEach((_, chartId) => {
            this.removeChart(chartId);
        });
        showToast('All charts cleared', 'info');
    }

    changeTheme(theme) {
        this.currentTheme = theme;
        const themeConfig = this.themes[theme];
        
        // Update all existing charts with new theme
        this.charts.forEach((chartData, chartId) => {
            // Re-render chart with new theme
            this.refreshChart(chartId);
        });
        
        showToast(`Theme changed to ${theme}`, 'success');
    }
}

// Initialize Visualization System
const vizSystem = new AdvancedVisualizationSystem();

// Export for global access
window.vizSystem = vizSystem;
window.createVisualization = (type, params) => vizSystem.createVisualization(type, params);
