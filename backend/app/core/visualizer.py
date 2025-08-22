"""Automated visualization generation engine."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import plot
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    default_width: int = 800
    default_height: int = 600
    color_palette: str = "viridis"
    theme: str = "plotly_white"
    max_categories_for_bar: int = 20
    max_points_for_scatter: int = 10000
    save_format: str = "html"  # html, png, svg
    interactive: bool = True


class AutoVisualizer:
    """Automated visualization generation engine."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.logger = get_logger(__name__)
        
        # Set up matplotlib and seaborn styles
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        
        # Create charts directory if it doesn't exist
        self.charts_dir = Path("data/charts")
        self.charts_dir.mkdir(parents=True, exist_ok=True)
    
    def auto_generate_visualizations(self, df: pd.DataFrame, dataset_id: str) -> List[Dict[str, Any]]:
        """Automatically generate appropriate visualizations for a dataset.
        
        Args:
            df: Dataset to visualize
            dataset_id: Unique dataset identifier
            
        Returns:
            List of generated visualization metadata
        """
        try:
            self.logger.info(f"Auto-generating visualizations for dataset: {dataset_id}")
            
            visualizations = []
            
            # Analyze data types
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            # Generate distribution plots for numerical columns
            for col in numerical_cols[:5]:  # Limit to first 5 numerical columns
                viz = self._create_distribution_plot(df, col, dataset_id)
                if viz:
                    visualizations.append(viz)
            
            # Generate bar charts for categorical columns
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                viz = self._create_categorical_plot(df, col, dataset_id)
                if viz:
                    visualizations.append(viz)
            
            # Generate correlation heatmap if we have numerical columns
            if len(numerical_cols) >= 2:
                viz = self._create_correlation_heatmap(df, numerical_cols, dataset_id)
                if viz:
                    visualizations.append(viz)
            
            # Generate scatter plots for numerical column pairs
            if len(numerical_cols) >= 2:
                # Create scatter plot for first two numerical columns
                viz = self._create_scatter_plot(df, numerical_cols[0], numerical_cols[1], dataset_id)
                if viz:
                    visualizations.append(viz)
            
            # Generate time series plots if datetime columns exist
            if datetime_cols and numerical_cols:
                viz = self._create_time_series_plot(df, datetime_cols[0], numerical_cols[0], dataset_id)
                if viz:
                    visualizations.append(viz)
            
            # Generate box plots for numerical vs categorical
            if numerical_cols and categorical_cols:
                viz = self._create_box_plot(df, numerical_cols[0], categorical_cols[0], dataset_id)
                if viz:
                    visualizations.append(viz)
            
            self.logger.info(f"Generated {len(visualizations)} visualizations for dataset {dataset_id}")
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error auto-generating visualizations: {e}")
            return []
    
    def create_custom_visualization(self, df: pd.DataFrame, chart_type: str, 
                                  x_column: Optional[str] = None, y_column: Optional[str] = None,
                                  color_column: Optional[str] = None, title: Optional[str] = None,
                                  dataset_id: str = None) -> Optional[Dict[str, Any]]:
        """Create a custom visualization based on user specifications.
        
        Args:
            df: Dataset
            chart_type: Type of chart to create
            x_column: Column for x-axis
            y_column: Column for y-axis
            color_column: Column for color encoding
            title: Chart title
            dataset_id: Dataset identifier
            
        Returns:
            Visualization metadata
        """
        try:
            self.logger.info(f"Creating custom {chart_type} visualization")
            
            chart_creators = {
                'histogram': self._create_histogram,
                'bar': self._create_bar_chart,
                'line': self._create_line_chart,
                'scatter': self._create_scatter_plot,
                'box': self._create_box_plot,
                'heatmap': self._create_heatmap,
                'pie': self._create_pie_chart,
                'violin': self._create_violin_plot
            }
            
            if chart_type not in chart_creators:
                self.logger.warning(f"Unsupported chart type: {chart_type}")
                return None
            
            creator_func = chart_creators[chart_type]
            
            # Call appropriate creator function based on chart type
            if chart_type in ['histogram']:
                return creator_func(df, x_column or df.columns[0], dataset_id, title)
            elif chart_type in ['bar', 'pie']:
                return creator_func(df, x_column or df.columns[0], dataset_id, title)
            elif chart_type in ['scatter', 'line']:
                y_col = y_column or (df.select_dtypes(include=[np.number]).columns[1] 
                                   if len(df.select_dtypes(include=[np.number]).columns) > 1 
                                   else df.columns[1])
                return creator_func(df, x_column or df.columns[0], y_col, dataset_id, title, color_column)
            elif chart_type in ['box', 'violin']:
                return creator_func(df, y_column or df.columns[0], x_column, dataset_id, title)
            elif chart_type == 'heatmap':
                return self._create_correlation_heatmap(df, df.select_dtypes(include=[np.number]).columns.tolist(), 
                                                      dataset_id, title)
            
        except Exception as e:
            self.logger.error(f"Error creating custom visualization: {e}")
            return None
    
    def _create_distribution_plot(self, df: pd.DataFrame, column: str, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Create distribution plot for a numerical column.
        
        Args:
            df: Dataset
            column: Column name
            dataset_id: Dataset identifier
            
        Returns:
            Visualization metadata
        """
        try:
            if column not in df.columns:
                return None
            
            # Remove missing values
            clean_data = df[column].dropna()
            if len(clean_data) == 0:
                return None
            
            fig = px.histogram(
                x=clean_data,
                nbins=50,
                title=f"Distribution of {column}",
                labels={'x': column, 'y': 'Frequency'},
                template=self.config.theme
            )
            
            fig.update_layout(
                width=self.config.default_width,
                height=self.config.default_height,
                showlegend=False
            )
            
            # Save the plot
            chart_id = f"dist_{dataset_id}_{column}_{int(datetime.now().timestamp())}"
            file_path = self._save_plot(fig, chart_id)
            
            return {
                'chart_id': chart_id,
                'chart_type': 'histogram',
                'title': f"Distribution of {column}",
                'description': f"Histogram showing the distribution of values in {column}",
                'columns': [column],
                'file_path': file_path,
                'importance': 'high',
                'insights': self._analyze_distribution(clean_data, column)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating distribution plot for {column}: {e}")
            return None
    
    def _create_categorical_plot(self, df: pd.DataFrame, column: str, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Create bar chart for a categorical column.
        
        Args:
            df: Dataset
            column: Column name
            dataset_id: Dataset identifier
            
        Returns:
            Visualization metadata
        """
        try:
            if column not in df.columns:
                return None
            
            # Count values and limit to top categories
            value_counts = df[column].value_counts().head(self.config.max_categories_for_bar)
            
            if len(value_counts) == 0:
                return None
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {column}",
                labels={'x': column, 'y': 'Count'},
                template=self.config.theme
            )
            
            fig.update_layout(
                width=self.config.default_width,
                height=self.config.default_height,
                xaxis_tickangle=-45
            )
            
            # Save the plot
            chart_id = f"cat_{dataset_id}_{column}_{int(datetime.now().timestamp())}"
            file_path = self._save_plot(fig, chart_id)
            
            return {
                'chart_id': chart_id,
                'chart_type': 'bar',
                'title': f"Distribution of {column}",
                'description': f"Bar chart showing the frequency of categories in {column}",
                'columns': [column],
                'file_path': file_path,
                'importance': 'medium',
                'insights': self._analyze_categorical(value_counts, column)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating categorical plot for {column}: {e}")
            return None
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, columns: List[str], 
                                  dataset_id: str, title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create correlation heatmap for numerical columns.
        
        Args:
            df: Dataset
            columns: List of numerical column names
            dataset_id: Dataset identifier
            title: Optional title override
            
        Returns:
            Visualization metadata
        """
        try:
            if len(columns) < 2:
                return None
            
            # Calculate correlation matrix
            corr_matrix = df[columns].corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Features", y="Features", color="Correlation"),
                x=columns,
                y=columns,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title=title or "Feature Correlation Matrix",
                template=self.config.theme
            )
            
            fig.update_layout(
                width=self.config.default_width,
                height=self.config.default_height
            )
            
            # Save the plot
            chart_id = f"corr_{dataset_id}_{int(datetime.now().timestamp())}"
            file_path = self._save_plot(fig, chart_id)
            
            return {
                'chart_id': chart_id,
                'chart_type': 'correlation_heatmap',
                'title': title or "Feature Correlation Matrix",
                'description': "Heatmap showing correlations between numerical features",
                'columns': columns,
                'file_path': file_path,
                'importance': 'high',
                'insights': self._analyze_correlations(corr_matrix)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {e}")
            return None
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_column: str, y_column: str, 
                           dataset_id: str, title: Optional[str] = None, 
                           color_column: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create scatter plot for two numerical columns.
        
        Args:
            df: Dataset
            x_column: X-axis column
            y_column: Y-axis column
            dataset_id: Dataset identifier
            title: Optional title override
            color_column: Optional color encoding column
            
        Returns:
            Visualization metadata
        """
        try:
            if x_column not in df.columns or y_column not in df.columns:
                return None
            
            # Sample data if too many points
            plot_df = df[[x_column, y_column]].copy()
            if color_column and color_column in df.columns:
                plot_df[color_column] = df[color_column]
            
            if len(plot_df) > self.config.max_points_for_scatter:
                plot_df = plot_df.sample(n=self.config.max_points_for_scatter)
            
            fig = px.scatter(
                plot_df,
                x=x_column,
                y=y_column,
                color=color_column,
                title=title or f"{y_column} vs {x_column}",
                template=self.config.theme
            )
            
            fig.update_layout(
                width=self.config.default_width,
                height=self.config.default_height
            )
            
            # Save the plot
            chart_id = f"scatter_{dataset_id}_{x_column}_{y_column}_{int(datetime.now().timestamp())}"
            file_path = self._save_plot(fig, chart_id)
            
            columns_used = [x_column, y_column]
            if color_column:
                columns_used.append(color_column)
            
            return {
                'chart_id': chart_id,
                'chart_type': 'scatter',
                'title': title or f"{y_column} vs {x_column}",
                'description': f"Scatter plot showing relationship between {x_column} and {y_column}",
                'columns': columns_used,
                'file_path': file_path,
                'importance': 'medium',
                'insights': self._analyze_scatter_relationship(df, x_column, y_column)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {e}")
            return None
    
    def _create_box_plot(self, df: pd.DataFrame, y_column: str, x_column: Optional[str], 
                        dataset_id: str, title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create box plot for numerical column, optionally grouped by categorical column.
        
        Args:
            df: Dataset
            y_column: Numerical column for y-axis
            x_column: Optional categorical column for grouping
            dataset_id: Dataset identifier
            title: Optional title override
            
        Returns:
            Visualization metadata
        """
        try:
            if y_column not in df.columns:
                return None
            
            if x_column and x_column in df.columns:
                # Grouped box plot
                fig = px.box(
                    df,
                    x=x_column,
                    y=y_column,
                    title=title or f"{y_column} Distribution by {x_column}",
                    template=self.config.theme
                )
                columns_used = [y_column, x_column]
                description = f"Box plot showing {y_column} distribution across {x_column} categories"
            else:
                # Single box plot
                fig = px.box(
                    df,
                    y=y_column,
                    title=title or f"{y_column} Distribution",
                    template=self.config.theme
                )
                columns_used = [y_column]
                description = f"Box plot showing {y_column} distribution and outliers"
            
            fig.update_layout(
                width=self.config.default_width,
                height=self.config.default_height
            )
            
            # Save the plot
            chart_id = f"box_{dataset_id}_{y_column}_{int(datetime.now().timestamp())}"
            file_path = self._save_plot(fig, chart_id)
            
            return {
                'chart_id': chart_id,
                'chart_type': 'box_plot',
                'title': title or f"{y_column} Distribution" + (f" by {x_column}" if x_column else ""),
                'description': description,
                'columns': columns_used,
                'file_path': file_path,
                'importance': 'medium',
                'insights': self._analyze_box_plot(df, y_column, x_column)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating box plot: {e}")
            return None
    
    def _create_time_series_plot(self, df: pd.DataFrame, date_column: str, value_column: str, 
                               dataset_id: str) -> Optional[Dict[str, Any]]:
        """Create time series plot.
        
        Args:
            df: Dataset
            date_column: Date/time column
            value_column: Value column
            dataset_id: Dataset identifier
            
        Returns:
            Visualization metadata
        """
        try:
            if date_column not in df.columns or value_column not in df.columns:
                return None
            
            # Ensure date column is datetime
            plot_df = df[[date_column, value_column]].copy()
            plot_df[date_column] = pd.to_datetime(plot_df[date_column])
            plot_df = plot_df.sort_values(date_column)
            
            fig = px.line(
                plot_df,
                x=date_column,
                y=value_column,
                title=f"{value_column} Over Time",
                template=self.config.theme
            )
            
            fig.update_layout(
                width=self.config.default_width,
                height=self.config.default_height
            )
            
            # Save the plot
            chart_id = f"time_{dataset_id}_{value_column}_{int(datetime.now().timestamp())}"
            file_path = self._save_plot(fig, chart_id)
            
            return {
                'chart_id': chart_id,
                'chart_type': 'time_series',
                'title': f"{value_column} Over Time",
                'description': f"Time series plot showing {value_column} trends over time",
                'columns': [date_column, value_column],
                'file_path': file_path,
                'importance': 'high',
                'insights': self._analyze_time_series(plot_df, date_column, value_column)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating time series plot: {e}")
            return None
    
    def _save_plot(self, fig, chart_id: str) -> str:
        """Save plot to file.
        
        Args:
            fig: Plotly figure
            chart_id: Unique chart identifier
            
        Returns:
            File path
        """
        if self.config.save_format == 'html':
            file_path = self.charts_dir / f"{chart_id}.html"
            fig.write_html(str(file_path))
        elif self.config.save_format == 'png':
            file_path = self.charts_dir / f"{chart_id}.png"
            fig.write_image(str(file_path))
        elif self.config.save_format == 'svg':
            file_path = self.charts_dir / f"{chart_id}.svg"
            fig.write_image(str(file_path))
        else:
            # Default to HTML
            file_path = self.charts_dir / f"{chart_id}.html"
            fig.write_html(str(file_path))
        
        return str(file_path)
    
    def _analyze_distribution(self, data: pd.Series, column: str) -> List[str]:
        """Analyze distribution and generate insights."""
        insights = []
        
        mean_val = data.mean()
        median_val = data.median()
        skewness = data.skew()
        
        if abs(skewness) < 0.5:
            insights.append(f"{column} shows a relatively normal distribution")
        elif skewness > 0.5:
            insights.append(f"{column} is right-skewed with a longer tail on the right")
        else:
            insights.append(f"{column} is left-skewed with a longer tail on the left")
        
        if abs(mean_val - median_val) / data.std() > 0.5:
            insights.append(f"Mean ({mean_val:.2f}) and median ({median_val:.2f}) differ significantly")
        
        return insights
    
    def _analyze_categorical(self, value_counts: pd.Series, column: str) -> List[str]:
        """Analyze categorical distribution and generate insights."""
        insights = []
        
        total_categories = len(value_counts)
        top_category = value_counts.index[0]
        top_percentage = (value_counts.iloc[0] / value_counts.sum()) * 100
        
        insights.append(f"{column} has {total_categories} unique categories")
        insights.append(f"'{top_category}' is the most frequent category ({top_percentage:.1f}%)")
        
        if top_percentage > 50:
            insights.append("Distribution is heavily skewed towards one category")
        elif top_percentage < 10:
            insights.append("Categories are relatively evenly distributed")
        
        return insights
    
    def _analyze_correlations(self, corr_matrix: pd.DataFrame) -> List[str]:
        """Analyze correlation matrix and generate insights."""
        insights = []
        
        # Find strong correlations
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if strong_corrs:
            insights.append(f"Found {len(strong_corrs)} strong correlations (>0.7)")
            for col1, col2, corr in strong_corrs[:3]:  # Show top 3
                direction = "positive" if corr > 0 else "negative"
                insights.append(f"Strong {direction} correlation between {col1} and {col2} ({corr:.2f})")
        else:
            insights.append("No strong correlations found between features")
        
        return insights
    
    def _analyze_scatter_relationship(self, df: pd.DataFrame, x_col: str, y_col: str) -> List[str]:
        """Analyze scatter plot relationship and generate insights."""
        insights = []
        
        # Calculate correlation
        correlation = df[x_col].corr(df[y_col])
        
        if abs(correlation) > 0.7:
            direction = "positive" if correlation > 0 else "negative"
            insights.append(f"Strong {direction} correlation ({correlation:.3f}) between {x_col} and {y_col}")
        elif abs(correlation) > 0.3:
            direction = "positive" if correlation > 0 else "negative"
            insights.append(f"Moderate {direction} correlation ({correlation:.3f}) between {x_col} and {y_col}")
        else:
            insights.append(f"Weak correlation ({correlation:.3f}) between {x_col} and {y_col}")
        
        return insights
    
    def _analyze_box_plot(self, df: pd.DataFrame, y_col: str, x_col: Optional[str]) -> List[str]:
        """Analyze box plot and generate insights."""
        insights = []
        
        if x_col and x_col in df.columns:
            # Grouped analysis
            group_stats = df.groupby(x_col)[y_col].agg(['mean', 'median', 'std'])
            highest_mean_group = group_stats['mean'].idxmax()
            insights.append(f"'{highest_mean_group}' group has the highest average {y_col}")
            
            if group_stats['std'].max() / group_stats['std'].min() > 2:
                insights.append(f"Significant variation in {y_col} spread across {x_col} groups")
        else:
            # Single variable analysis
            Q1 = df[y_col].quantile(0.25)
            Q3 = df[y_col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[y_col] < Q1 - 1.5*IQR) | (df[y_col] > Q3 + 1.5*IQR)).sum()
            
            if outlier_count > 0:
                outlier_percentage = (outlier_count / len(df)) * 100
                insights.append(f"{outlier_count} potential outliers detected ({outlier_percentage:.1f}%)")
        
        return insights
    
    def _analyze_time_series(self, df: pd.DataFrame, date_col: str, value_col: str) -> List[str]:
        """Analyze time series and generate insights."""
        insights = []
        
        # Calculate basic trend
        df_sorted = df.sort_values(date_col)
        first_val = df_sorted[value_col].iloc[0]
        last_val = df_sorted[value_col].iloc[-1]
        
        if last_val > first_val * 1.1:
            insights.append(f"{value_col} shows an overall increasing trend")
        elif last_val < first_val * 0.9:
            insights.append(f"{value_col} shows an overall decreasing trend")
        else:
            insights.append(f"{value_col} remains relatively stable over time")
        
        # Calculate volatility
        daily_changes = df_sorted[value_col].pct_change().dropna()
        volatility = daily_changes.std()
        
        if volatility > 0.1:
            insights.append(f"{value_col} shows high volatility")
        elif volatility > 0.05:
            insights.append(f"{value_col} shows moderate volatility")
        else:
            insights.append(f"{value_col} shows low volatility")
        
        return insights


# Global visualizer instance
auto_visualizer = AutoVisualizer()
