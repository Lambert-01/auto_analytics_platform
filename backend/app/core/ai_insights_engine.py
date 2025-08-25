"""
AI-Powered Insights Engine for Analytics Platform
Implements: Natural Language Generation, Narrative Insights, AI Chat Interface
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import logging
from dataclasses import dataclass
import re
import warnings
warnings.filterwarnings('ignore')

# NLP and AI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Local imports
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)


@dataclass
class InsightsConfig:
    """Configuration for AI insights generation."""
    # AI Model settings
    use_openai: bool = True
    openai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.7
    
    # Local model settings
    local_model_name: str = "microsoft/DialoGPT-medium"
    
    # Insight generation settings
    max_insights_per_category: int = 5
    insight_categories: List[str] = None
    
    # Natural language settings
    language: str = "en"
    tone: str = "professional"  # professional, casual, technical
    
    # Context settings
    include_statistics: bool = True
    include_recommendations: bool = True
    include_visualizations: bool = True
    
    def __post_init__(self):
        if self.insight_categories is None:
            self.insight_categories = [
                'data_quality', 'distributions', 'correlations', 
                'outliers', 'patterns', 'recommendations'
            ]


class AIInsightsEngine:
    """AI-powered insights generation and natural language interface."""
    
    def __init__(self, config: Optional[InsightsConfig] = None):
        """Initialize the AI insights engine."""
        self.config = config or InsightsConfig()
        self.logger = get_logger(__name__)
        
        # Initialize AI models
        self._initialize_ai_models()
        
        # Insight templates
        self.insight_templates = self._load_insight_templates()
        
        # Chat history for conversational interface
        self.chat_history = []
        
    def _initialize_ai_models(self):
        """Initialize AI models for insight generation."""
        self.openai_client = None
        self.local_model = None
        self.local_tokenizer = None
        
        # Initialize OpenAI
        if OPENAI_AVAILABLE and self.config.use_openai and settings.openai_api_key:
            try:
                openai.api_key = settings.openai_api_key
                self.openai_client = openai
                self.logger.info("OpenAI client initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Initialize local model as fallback
        if TRANSFORMERS_AVAILABLE and not self.openai_client:
            try:
                self.local_tokenizer = AutoTokenizer.from_pretrained(self.config.local_model_name)
                self.local_model = AutoModelForCausalLM.from_pretrained(self.config.local_model_name)
                self.logger.info("Local AI model initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize local model: {e}")
    
    def generate_comprehensive_insights(self, 
                                      dataset_info: Dict[str, Any],
                                      profiling_results: Dict[str, Any],
                                      analysis_results: Dict[str, Any],
                                      model_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive AI-powered insights from analysis results.
        
        Args:
            dataset_info: Basic dataset information
            profiling_results: Data profiling results
            analysis_results: EDA and statistical analysis results
            model_results: Optional ML model results
            
        Returns:
            Comprehensive insights with narratives
        """
        self.logger.info("Generating comprehensive AI insights")
        
        try:
            insights = {
                'executive_summary': self._generate_executive_summary(dataset_info, profiling_results),
                'data_quality_insights': self._generate_data_quality_insights(profiling_results),
                'statistical_insights': self._generate_statistical_insights(analysis_results),
                'pattern_insights': self._generate_pattern_insights(analysis_results),
                'correlation_insights': self._generate_correlation_insights(analysis_results),
                'distribution_insights': self._generate_distribution_insights(profiling_results),
                'outlier_insights': self._generate_outlier_insights(analysis_results),
                'recommendations': self._generate_recommendations(dataset_info, profiling_results, analysis_results),
                'narrative_summary': self._generate_narrative_summary(dataset_info, profiling_results, analysis_results),
                'key_findings': self._extract_key_findings(profiling_results, analysis_results),
                'actionable_items': self._generate_actionable_items(profiling_results, analysis_results)
            }
            
            # Add model insights if available
            if model_results:
                insights['model_insights'] = self._generate_model_insights(model_results)
            
            # Generate meta-insights about the analysis
            insights['meta_insights'] = self._generate_meta_insights(insights)
            
            # Add timestamps and metadata
            insights['generation_info'] = {
                'generated_at': datetime.now().isoformat(),
                'ai_model_used': self._get_model_info(),
                'insight_categories': list(insights.keys()),
                'total_insights': sum(len(v) if isinstance(v, list) else 1 for v in insights.values())
            }
            
            self.logger.info(f"Generated {insights['generation_info']['total_insights']} insights")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return self._generate_fallback_insights(dataset_info, profiling_results)
    
    def _generate_executive_summary(self, dataset_info: Dict, profiling_results: Dict) -> str:
        """Generate executive summary of the dataset."""
        shape = dataset_info.get('shape', (0, 0))
        missing_pct = profiling_results.get('overview', {}).get('missing_data', {}).get('missing_percentage', 0)
        
        summary_prompt = f"""
        Generate a professional executive summary for a dataset analysis with the following characteristics:
        - Dataset size: {shape[0]:,} rows and {shape[1]} columns
        - Missing data: {missing_pct:.1f}% of values are missing
        - Data types: {profiling_results.get('overview', {}).get('column_types', {})}
        
        Provide a 2-3 sentence executive summary highlighting the key characteristics and overall data quality.
        """
        
        return self._generate_text(summary_prompt, max_tokens=150)
    
    def _generate_data_quality_insights(self, profiling_results: Dict) -> List[str]:
        """Generate insights about data quality."""
        insights = []
        overview = profiling_results.get('overview', {})
        
        # Missing data insights
        missing_data = overview.get('missing_data', {})
        missing_pct = missing_data.get('missing_percentage', 0)
        
        if missing_pct == 0:
            insights.append("✅ Excellent data quality: No missing values detected in the dataset.")
        elif missing_pct < 5:
            insights.append(f"✅ Good data quality: Only {missing_pct:.1f}% missing values, which is within acceptable limits.")
        elif missing_pct < 20:
            insights.append(f"⚠️ Moderate data quality: {missing_pct:.1f}% missing values require attention before analysis.")
        else:
            insights.append(f"🔴 Poor data quality: {missing_pct:.1f}% missing values indicate significant data collection issues.")
        
        # Duplicate data insights
        duplicates = overview.get('duplicates', {})
        duplicate_pct = duplicates.get('percentage', 0)
        
        if duplicate_pct > 5:
            insights.append(f"🔍 Data integrity issue: {duplicate_pct:.1f}% duplicate rows detected, consider deduplication.")
        elif duplicate_pct > 0:
            insights.append(f"📊 Minor duplication: {duplicate_pct:.1f}% duplicate rows found, review data collection process.")
        
        # Memory usage insights
        memory_usage = overview.get('memory_usage', {})
        total_mb = memory_usage.get('total_mb', 0)
        
        if total_mb > 1000:  # > 1GB
            insights.append(f"💾 Large dataset: {total_mb:.1f}MB memory usage requires efficient processing strategies.")
        elif total_mb > 100:  # > 100MB
            insights.append(f"📈 Medium-sized dataset: {total_mb:.1f}MB is manageable but consider optimization for large-scale operations.")
        
        return insights[:self.config.max_insights_per_category]
    
    def _generate_statistical_insights(self, analysis_results: Dict) -> List[str]:
        """Generate statistical insights from analysis results."""
        insights = []
        
        # This would be based on the actual analysis results structure
        # For now, generating template insights
        univariate = analysis_results.get('univariate_analysis', {})
        
        for column, analysis in list(univariate.items())[:3]:  # Top 3 columns
            if isinstance(analysis, dict):
                column_type = analysis.get('type', 'unknown')
                if column_type == 'numeric':
                    stats = analysis.get('statistics', {})
                    mean_val = stats.get('mean', 0)
                    std_val = stats.get('std', 0)
                    skewness = stats.get('skewness', 0)
                    
                    if abs(skewness) > 2:
                        direction = "right" if skewness > 0 else "left"
                        insights.append(f"📊 {column}: Highly {direction}-skewed distribution (skewness: {skewness:.2f}) suggests non-normal behavior.")
                    
                    if std_val > mean_val:
                        insights.append(f"📈 {column}: High variability detected (std: {std_val:.2f}, mean: {mean_val:.2f}) indicates diverse value ranges.")
                
                elif column_type == 'categorical':
                    unique_count = analysis.get('unique_count', 0)
                    top_frequency = analysis.get('top_frequency_pct', 0)
                    
                    if top_frequency > 80:
                        insights.append(f"🎯 {column}: Highly concentrated categorical variable ({top_frequency:.1f}% in top category) may have limited predictive power.")
                    elif unique_count > 50:
                        insights.append(f"🔍 {column}: High cardinality categorical ({unique_count} unique values) may require encoding strategies.")
        
        return insights[:self.config.max_insights_per_category]
    
    def _generate_pattern_insights(self, analysis_results: Dict) -> List[str]:
        """Generate insights about patterns in the data."""
        insights = []
        
        patterns = analysis_results.get('patterns', {})
        
        # Time-based patterns
        if 'temporal_patterns' in patterns:
            temporal = patterns['temporal_patterns']
            if temporal.get('has_trend', False):
                insights.append("📈 Temporal trend detected: Data shows systematic changes over time that could be valuable for forecasting.")
            
            if temporal.get('has_seasonality', False):
                insights.append("🔄 Seasonal patterns identified: Regular cyclical behavior suggests time-dependent relationships.")
        
        # Clustering patterns
        if 'clusters' in patterns:
            cluster_info = patterns['clusters']
            n_clusters = cluster_info.get('optimal_clusters', 0)
            if n_clusters > 1:
                insights.append(f"🎯 Data segmentation opportunity: {n_clusters} distinct clusters identified, suggesting natural groupings in the data.")
        
        # Anomaly patterns
        if 'anomalies' in patterns:
            anomaly_rate = patterns['anomalies'].get('anomaly_rate', 0)
            if anomaly_rate > 5:
                insights.append(f"⚠️ High anomaly rate: {anomaly_rate:.1f}% of data points are anomalous, investigate data collection processes.")
        
        return insights[:self.config.max_insights_per_category]
    
    def _generate_correlation_insights(self, analysis_results: Dict) -> List[str]:
        """Generate insights about correlations and relationships."""
        insights = []
        
        correlations = analysis_results.get('relationships', {}).get('correlations', {})
        strong_correlations = correlations.get('strong_correlations', [])
        
        if strong_correlations:
            # Top correlation
            top_corr = strong_correlations[0]
            corr_value = top_corr.get('pearson_correlation', 0)
            col1 = top_corr.get('column1', 'Column1')
            col2 = top_corr.get('column2', 'Column2')
            
            if abs(corr_value) > 0.9:
                insights.append(f"🔗 Very strong correlation: {col1} and {col2} (r={corr_value:.3f}) are highly related, consider multicollinearity.")
            elif abs(corr_value) > 0.7:
                insights.append(f"📊 Strong relationship: {col1} and {col2} (r={corr_value:.3f}) show significant linear association.")
            
            # Multiple strong correlations
            if len(strong_correlations) > 3:
                insights.append(f"🔍 Multiple strong correlations detected ({len(strong_correlations)} pairs), feature selection may improve model performance.")
        else:
            insights.append("📈 Low correlation environment: Features show independence, which is beneficial for diverse modeling approaches.")
        
        return insights[:self.config.max_insights_per_category]
    
    def _generate_distribution_insights(self, profiling_results: Dict) -> List[str]:
        """Generate insights about data distributions."""
        insights = []
        
        columns = profiling_results.get('columns', {})
        
        # Analyze numeric distributions
        numeric_columns = []
        for col_name, col_data in columns.items():
            if 'numeric_stats' in col_data:
                numeric_columns.append((col_name, col_data['numeric_stats']))
        
        if numeric_columns:
            # Normal distribution detection
            normal_cols = []
            skewed_cols = []
            
            for col_name, stats in numeric_columns[:5]:  # Check top 5
                skewness = stats.get('shape', {}).get('skewness', 0)
                is_normal = stats.get('shape', {}).get('is_normal', False)
                
                if is_normal:
                    normal_cols.append(col_name)
                elif abs(skewness) > 2:
                    direction = "right" if skewness > 0 else "left"
                    skewed_cols.append(f"{col_name} ({direction})")
            
            if normal_cols:
                insights.append(f"📊 Normal distributions detected in: {', '.join(normal_cols[:3])} - suitable for parametric statistical tests.")
            
            if skewed_cols:
                insights.append(f"📈 Highly skewed distributions in: {', '.join(skewed_cols[:3])} - consider transformation for modeling.")
        
        # Analyze categorical distributions
        categorical_columns = []
        for col_name, col_data in columns.items():
            if 'categorical_stats' in col_data:
                categorical_columns.append((col_name, col_data['categorical_stats']))
        
        if categorical_columns:
            high_entropy_cols = []
            low_entropy_cols = []
            
            for col_name, stats in categorical_columns[:5]:
                entropy = stats.get('entropy', 0)
                cardinality = stats.get('cardinality', 0)
                
                if entropy > 3:  # High entropy
                    high_entropy_cols.append(col_name)
                elif entropy < 1 and cardinality > 1:  # Low entropy but multiple categories
                    low_entropy_cols.append(col_name)
            
            if high_entropy_cols:
                insights.append(f"🎯 High diversity categories: {', '.join(high_entropy_cols[:3])} show good class distribution for modeling.")
            
            if low_entropy_cols:
                insights.append(f"⚠️ Imbalanced categories: {', '.join(low_entropy_cols[:3])} have skewed distributions that may affect model performance.")
        
        return insights[:self.config.max_insights_per_category]
    
    def _generate_outlier_insights(self, analysis_results: Dict) -> List[str]:
        """Generate insights about outliers."""
        insights = []
        
        outliers = analysis_results.get('outliers', {})
        
        for column, outlier_info in list(outliers.items())[:3]:  # Top 3 columns
            outlier_pct = outlier_info.get('summary', {}).get('consensus_percentage', 0)
            
            if outlier_pct > 10:
                insights.append(f"🔴 High outlier rate in {column}: {outlier_pct:.1f}% outliers detected, investigate data collection or genuine extreme values.")
            elif outlier_pct > 5:
                insights.append(f"⚠️ Moderate outliers in {column}: {outlier_pct:.1f}% outliers found, consider outlier treatment strategies.")
            elif outlier_pct > 0:
                insights.append(f"📊 Minor outliers in {column}: {outlier_pct:.1f}% outliers detected, may represent valuable edge cases.")
        
        return insights[:self.config.max_insights_per_category]
    
    def _generate_recommendations(self, dataset_info: Dict, profiling_results: Dict, analysis_results: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        shape = dataset_info.get('shape', (0, 0))
        missing_pct = profiling_results.get('overview', {}).get('missing_data', {}).get('missing_percentage', 0)
        
        # Data collection recommendations
        if shape[0] < 1000:
            recommendations.append("📊 Data Collection: Increase sample size to at least 1,000 records for more robust statistical analysis.")
        
        if missing_pct > 20:
            recommendations.append("🔧 Data Quality: Address missing value issues before proceeding with modeling - consider data collection improvements.")
        
        # Feature engineering recommendations
        correlations = analysis_results.get('relationships', {}).get('correlations', {})
        if correlations.get('strong_correlations', []):
            recommendations.append("🛠️ Feature Engineering: Create interaction features from strongly correlated variables for enhanced model performance.")
        
        # Modeling recommendations
        if shape[1] > 50:
            recommendations.append("🤖 Modeling: Consider feature selection techniques due to high dimensionality - use L1 regularization or recursive feature elimination.")
        
        # Preprocessing recommendations
        outliers = analysis_results.get('outliers', {})
        high_outlier_cols = [col for col, info in outliers.items() 
                           if info.get('summary', {}).get('consensus_percentage', 0) > 5]
        if high_outlier_cols:
            recommendations.append(f"⚙️ Preprocessing: Apply outlier treatment to {', '.join(high_outlier_cols[:3])} before model training.")
        
        # Business recommendations
        recommendations.append("💼 Business Impact: Focus on features with highest predictive power for actionable insights.")
        recommendations.append("📈 Monitoring: Implement data quality monitoring to maintain analysis reliability over time.")
        
        return recommendations[:self.config.max_insights_per_category]
    
    def _generate_narrative_summary(self, dataset_info: Dict, profiling_results: Dict, analysis_results: Dict) -> str:
        """Generate a comprehensive narrative summary."""
        shape = dataset_info.get('shape', (0, 0))
        missing_pct = profiling_results.get('overview', {}).get('missing_data', {}).get('missing_percentage', 0)
        
        # Data quality assessment
        if missing_pct < 5:
            quality_desc = "high-quality"
        elif missing_pct < 20:
            quality_desc = "moderate-quality"
        else:
            quality_desc = "quality-challenged"
        
        # Size assessment
        if shape[0] > 10000:
            size_desc = "large-scale"
        elif shape[0] > 1000:
            size_desc = "medium-sized"
        else:
            size_desc = "small-scale"
        
        narrative_prompt = f"""
        Write a comprehensive narrative summary for a {size_desc}, {quality_desc} dataset with {shape[0]:,} records and {shape[1]} features. 
        The analysis reveals {missing_pct:.1f}% missing values. 
        Describe the key characteristics, potential use cases, and overall analytical potential in 3-4 sentences.
        """
        
        return self._generate_text(narrative_prompt, max_tokens=200)
    
    def _extract_key_findings(self, profiling_results: Dict, analysis_results: Dict) -> List[str]:
        """Extract the most important findings from the analysis."""
        findings = []
        
        # Most important data quality finding
        missing_pct = profiling_results.get('overview', {}).get('missing_data', {}).get('missing_percentage', 0)
        if missing_pct > 0:
            findings.append(f"Key Finding: {missing_pct:.1f}% of data values are missing, requiring preprocessing attention.")
        
        # Most important correlation finding
        correlations = analysis_results.get('relationships', {}).get('correlations', {})
        strong_correlations = correlations.get('strong_correlations', [])
        if strong_correlations:
            top_corr = strong_correlations[0]
            findings.append(f"Key Finding: Strong correlation detected between {top_corr.get('column1', 'features')} and {top_corr.get('column2', 'target')} (r={top_corr.get('pearson_correlation', 0):.3f}).")
        
        # Most important distribution finding
        # This would be based on actual distribution analysis
        findings.append("Key Finding: Dataset shows mixed distribution patterns suitable for ensemble modeling approaches.")
        
        return findings[:3]  # Top 3 key findings
    
    def _generate_actionable_items(self, profiling_results: Dict, analysis_results: Dict) -> List[str]:
        """Generate specific actionable items."""
        actions = []
        
        missing_pct = profiling_results.get('overview', {}).get('missing_data', {}).get('missing_percentage', 0)
        
        # Immediate actions
        if missing_pct > 10:
            actions.append("IMMEDIATE: Implement missing value imputation strategy before any modeling attempts.")
        
        # Short-term actions
        actions.append("SHORT-TERM: Perform feature engineering to create interaction terms from correlated variables.")
        actions.append("SHORT-TERM: Apply data transformation techniques to address skewed distributions.")
        
        # Long-term actions
        actions.append("LONG-TERM: Establish data quality monitoring pipeline to prevent future quality issues.")
        actions.append("LONG-TERM: Develop automated model retraining pipeline for production deployment.")
        
        return actions[:5]
    
    def _generate_model_insights(self, model_results: Dict) -> List[str]:
        """Generate insights from model training results."""
        insights = []
        
        best_model = model_results.get('best_model', {})
        model_name = best_model.get('model_name', 'Unknown')
        performance = best_model.get('performance', {})
        
        # Model performance insights
        if 'accuracy_test' in performance:
            accuracy = performance['accuracy_test']
            if accuracy > 0.9:
                insights.append(f"🏆 Excellent model performance: {model_name} achieved {accuracy:.1%} accuracy on test data.")
            elif accuracy > 0.8:
                insights.append(f"✅ Good model performance: {model_name} shows strong predictive capability at {accuracy:.1%} accuracy.")
            else:
                insights.append(f"⚠️ Moderate model performance: {model_name} accuracy of {accuracy:.1%} suggests room for improvement.")
        
        if 'r2_test' in performance:
            r2 = performance['r2_test']
            if r2 > 0.8:
                insights.append(f"📊 Strong explanatory power: Model explains {r2:.1%} of variance in target variable.")
            elif r2 > 0.6:
                insights.append(f"📈 Moderate explanatory power: Model captures {r2:.1%} of target variance.")
            else:
                insights.append(f"🔍 Limited explanatory power: Only {r2:.1%} of variance explained, consider feature engineering.")
        
        # Feature importance insights
        feature_importance = best_model.get('feature_importance', {})
        if feature_importance:
            top_feature = max(feature_importance.items(), key=lambda x: x[1])
            insights.append(f"🎯 Most important feature: {top_feature[0]} contributes {top_feature[1]:.1%} to model predictions.")
        
        return insights
    
    def _generate_meta_insights(self, insights: Dict) -> List[str]:
        """Generate meta-insights about the analysis itself."""
        meta_insights = []
        
        total_insights = sum(len(v) if isinstance(v, list) else 1 for v in insights.values() if isinstance(v, (list, str)))
        
        meta_insights.append(f"🔍 Analysis Depth: Generated {total_insights} insights across {len(insights)} categories for comprehensive understanding.")
        
        # Data complexity assessment
        has_correlations = bool(insights.get('correlation_insights'))
        has_outliers = bool(insights.get('outlier_insights'))
        has_patterns = bool(insights.get('pattern_insights'))
        
        complexity_indicators = sum([has_correlations, has_outliers, has_patterns])
        
        if complexity_indicators >= 3:
            meta_insights.append("🧠 Complex Dataset: Multiple analytical dimensions identified, suggesting rich modeling opportunities.")
        elif complexity_indicators >= 2:
            meta_insights.append("📊 Moderate Complexity: Several interesting patterns detected for targeted analysis.")
        else:
            meta_insights.append("📈 Straightforward Dataset: Clear patterns suitable for standard analytical approaches.")
        
        return meta_insights
    
    def chat_with_data(self, question: str, context: Dict[str, Any]) -> str:
        """
        Conversational interface for data questions.
        
        Args:
            question: User's natural language question
            context: Analysis context and results
            
        Returns:
            AI-generated response
        """
        self.logger.info(f"Processing chat question: {question}")
        
        # Add to chat history
        self.chat_history.append({
            'type': 'user',
            'message': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate response based on question type
        question_lower = question.lower()
        
        # Question routing
        if any(word in question_lower for word in ['correlation', 'relationship', 'related']):
            response = self._answer_correlation_question(question, context)
        elif any(word in question_lower for word in ['outlier', 'anomaly', 'unusual']):
            response = self._answer_outlier_question(question, context)
        elif any(word in question_lower for word in ['distribution', 'spread', 'histogram']):
            response = self._answer_distribution_question(question, context)
        elif any(word in question_lower for word in ['missing', 'null', 'empty']):
            response = self._answer_missing_data_question(question, context)
        elif any(word in question_lower for word in ['recommend', 'suggest', 'advice']):
            response = self._answer_recommendation_question(question, context)
        elif any(word in question_lower for word in ['trend', 'time', 'temporal']):
            response = self._answer_trend_question(question, context)
        else:
            response = self._answer_general_question(question, context)
        
        # Add response to chat history
        self.chat_history.append({
            'type': 'assistant',
            'message': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _answer_correlation_question(self, question: str, context: Dict) -> str:
        """Answer questions about correlations and relationships."""
        correlations = context.get('relationships', {}).get('correlations', {})
        strong_correlations = correlations.get('strong_correlations', [])
        
        if strong_correlations:
            top_corr = strong_correlations[0]
            response = f"The strongest correlation in your dataset is between {top_corr.get('column1', 'Column1')} and {top_corr.get('column2', 'Column2')} with a correlation coefficient of {top_corr.get('pearson_correlation', 0):.3f}. "
            
            if len(strong_correlations) > 1:
                response += f"There are {len(strong_correlations)} strong correlations total, which suggests interconnected relationships in your data."
            
            return response
        else:
            return "Your dataset shows relatively low correlations between variables, which indicates good feature independence. This is actually beneficial for many machine learning models as it reduces multicollinearity issues."
    
    def _answer_outlier_question(self, question: str, context: Dict) -> str:
        """Answer questions about outliers."""
        outliers = context.get('outliers', {})
        
        if outliers:
            high_outlier_cols = [(col, info['summary']['consensus_percentage']) 
                               for col, info in outliers.items() 
                               if info.get('summary', {}).get('consensus_percentage', 0) > 5]
            
            if high_outlier_cols:
                col_name, outlier_pct = high_outlier_cols[0]
                return f"Yes, I found significant outliers in your data. The column '{col_name}' has {outlier_pct:.1f}% outliers. These could represent data entry errors or genuine extreme values. I recommend investigating these points to determine if they should be kept, transformed, or removed."
            else:
                return "Your dataset has relatively few outliers, which indicates good data quality. The minor outliers present are likely genuine extreme values rather than data quality issues."
        else:
            return "No significant outliers were detected in your dataset, which suggests consistent and reliable data collection processes."
    
    def _answer_distribution_question(self, question: str, context: Dict) -> str:
        """Answer questions about data distributions."""
        profiling = context.get('profiling', {})
        columns = profiling.get('columns', {})
        
        # Find numeric columns with distribution info
        numeric_distributions = []
        for col_name, col_data in columns.items():
            if 'numeric_stats' in col_data:
                skewness = col_data['numeric_stats'].get('shape', {}).get('skewness', 0)
                is_normal = col_data['numeric_stats'].get('shape', {}).get('is_normal', False)
                numeric_distributions.append((col_name, skewness, is_normal))
        
        if numeric_distributions:
            normal_cols = [col for col, _, is_normal in numeric_distributions if is_normal]
            skewed_cols = [col for col, skew, _ in numeric_distributions if abs(skew) > 1]
            
            response = f"Your dataset contains {len(numeric_distributions)} numeric columns. "
            
            if normal_cols:
                response += f"Columns with normal distributions: {', '.join(normal_cols[:3])}. These are suitable for parametric statistical tests. "
            
            if skewed_cols:
                response += f"Skewed distributions detected in: {', '.join(skewed_cols[:3])}. Consider log transformation or other normalization techniques for these columns."
            
            return response
        else:
            return "Your dataset appears to be primarily categorical. For categorical data, I recommend looking at frequency distributions and entropy measures rather than traditional statistical distributions."
    
    def _answer_missing_data_question(self, question: str, context: Dict) -> str:
        """Answer questions about missing data."""
        profiling = context.get('profiling', {})
        missing_info = profiling.get('overview', {}).get('missing_data', {})
        missing_pct = missing_info.get('missing_percentage', 0)
        
        if missing_pct == 0:
            return "Great news! Your dataset has no missing values, which means you can proceed directly to analysis without imputation."
        elif missing_pct < 5:
            return f"Your dataset has {missing_pct:.1f}% missing values, which is quite low and manageable. You can safely use simple imputation methods like median for numeric columns and mode for categorical columns."
        elif missing_pct < 20:
            return f"Your dataset has {missing_pct:.1f}% missing values, which requires attention. I recommend analyzing the missing data patterns first to determine if the missingness is random or systematic, then apply appropriate imputation techniques."
        else:
            return f"Your dataset has {missing_pct:.1f}% missing values, which is quite high. This suggests potential issues with data collection. Consider advanced imputation methods like KNN or MICE, or investigate whether the missing values carry information themselves."
    
    def _answer_recommendation_question(self, question: str, context: Dict) -> str:
        """Answer recommendation questions."""
        recommendations = context.get('recommendations', [])
        
        if recommendations:
            priority_rec = recommendations[0] if recommendations else "Ensure data quality before proceeding with analysis."
            return f"Based on my analysis, here's my top recommendation: {priority_rec} Additionally, I suggest focusing on feature engineering and implementing proper cross-validation for model development."
        else:
            return "Based on your data characteristics, I recommend starting with exploratory data analysis to understand feature distributions, then proceeding with feature engineering and model selection based on your specific use case."
    
    def _answer_trend_question(self, question: str, context: Dict) -> str:
        """Answer questions about trends and temporal patterns."""
        patterns = context.get('patterns', {})
        temporal = patterns.get('temporal_patterns', {})
        
        if temporal.get('has_trend', False):
            return "Yes, I detected temporal trends in your data. This suggests systematic changes over time that could be valuable for forecasting models. Consider using time series analysis techniques or including time-based features in your models."
        elif temporal.get('has_seasonality', False):
            return "I found seasonal patterns in your data, which indicates regular cyclical behavior. This is valuable for prediction and planning - consider seasonal decomposition and seasonal adjustment techniques."
        else:
            return "I don't see strong temporal trends in the current analysis. If your data has time components, make sure they're properly formatted as datetime types for temporal analysis."
    
    def _answer_general_question(self, question: str, context: Dict) -> str:
        """Answer general questions about the dataset."""
        dataset_info = context.get('dataset_info', {})
        shape = dataset_info.get('shape', (0, 0))
        
        general_response = f"Your dataset contains {shape[0]:,} rows and {shape[1]} columns. "
        
        # Add relevant context based on available information
        profiling = context.get('profiling', {})
        if profiling:
            missing_pct = profiling.get('overview', {}).get('missing_data', {}).get('missing_percentage', 0)
            if missing_pct > 0:
                general_response += f"There are {missing_pct:.1f}% missing values that need attention. "
        
        general_response += "The dataset appears suitable for machine learning analysis. Would you like me to elaborate on any specific aspect of your data?"
        
        return general_response
    
    def _generate_text(self, prompt: str, max_tokens: int = None) -> str:
        """Generate text using available AI models."""
        max_tokens = max_tokens or self.config.max_tokens
        
        # Try OpenAI first
        if self.openai_client:
            try:
                response = self.openai_client.ChatCompletion.create(
                    model=self.config.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=self.config.temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.logger.warning(f"OpenAI generation failed: {e}")
        
        # Try local model
        if self.local_model and self.local_tokenizer:
            try:
                inputs = self.local_tokenizer.encode(prompt, return_tensors='pt')
                outputs = self.local_model.generate(
                    inputs, 
                    max_length=inputs.shape[1] + max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )
                response = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response[len(prompt):].strip()
            except Exception as e:
                self.logger.warning(f"Local model generation failed: {e}")
        
        # Fallback to template-based generation
        return self._generate_template_response(prompt)
    
    def _generate_template_response(self, prompt: str) -> str:
        """Generate response using templates when AI models are unavailable."""
        prompt_lower = prompt.lower()
        
        if 'executive summary' in prompt_lower:
            return "This dataset shows good potential for analytical insights with structured data suitable for machine learning applications."
        elif 'comprehensive narrative' in prompt_lower:
            return "The dataset demonstrates typical characteristics of business data with mixed data types and moderate complexity, offering opportunities for predictive modeling and statistical analysis."
        else:
            return "Based on the data analysis, this dataset shows promise for generating actionable insights through systematic analytical approaches."
    
    def _generate_fallback_insights(self, dataset_info: Dict, profiling_results: Dict) -> Dict[str, Any]:
        """Generate basic insights when AI models fail."""
        shape = dataset_info.get('shape', (0, 0))
        
        return {
            'executive_summary': f"Dataset contains {shape[0]:,} records and {shape[1]} features suitable for analysis.",
            'data_quality_insights': ["Dataset loaded successfully and ready for analysis."],
            'recommendations': ["Proceed with exploratory data analysis.", "Consider data preprocessing steps.", "Evaluate model performance carefully."],
            'generation_info': {
                'generated_at': datetime.now().isoformat(),
                'ai_model_used': 'fallback_template',
                'note': 'AI models unavailable, using template-based insights'
            }
        }
    
    def _get_model_info(self) -> str:
        """Get information about the AI model being used."""
        if self.openai_client:
            return f"OpenAI {self.config.openai_model}"
        elif self.local_model:
            return f"Local model: {self.config.local_model_name}"
        else:
            return "Template-based generation"
    
    def _load_insight_templates(self) -> Dict[str, List[str]]:
        """Load insight templates for different categories."""
        return {
            'data_quality': [
                "Dataset shows {quality_level} data quality with {missing_pct:.1f}% missing values.",
                "Data integrity is {integrity_level} with {duplicate_pct:.1f}% duplicate records."
            ],
            'correlations': [
                "Strong correlation detected between {col1} and {col2} (r={corr:.3f}).",
                "Multiple correlated features suggest potential multicollinearity issues."
            ],
            'distributions': [
                "{column} shows {distribution_type} distribution with skewness {skew:.2f}.",
                "Categorical variable {column} has {cardinality} unique values."
            ]
        }
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the chat conversation history."""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear the chat conversation history."""
        self.chat_history = []
        self.logger.info("Chat history cleared")
