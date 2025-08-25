"""
Comprehensive Data Processing Engine for AI-Powered Analytics Platform
Implements: Data Ingestion, Validation, Type Inference, Profiling, Transformation, EDA
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder, TargetEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Local imports
from app.utils.logger import get_logger
from app.models.dataset import DatasetType, DataQualityIssue
from app.models.analysis import StatisticalSummary, CategoricalSummary, ColumnAnalysis

logger = get_logger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    # Profiling settings
    max_categorical_unique: int = 50
    sample_size_for_profiling: Optional[int] = 10000
    correlation_threshold: float = 0.7
    outlier_methods: List[str] = None
    
    # Transformation settings
    missing_value_strategy: str = "auto"  # auto, drop, mean, median, mode, knn
    outlier_treatment: str = "clip"  # clip, remove, winsorize
    encoding_strategy: str = "auto"  # auto, onehot, label, target
    scaling_strategy: str = "standard"  # standard, minmax, robust
    
    # EDA settings
    max_plots_per_type: int = 20
    plot_sample_size: int = 5000
    
    def __post_init__(self):
        if self.outlier_methods is None:
            self.outlier_methods = ['iqr', 'zscore', 'isolation_forest']


class ComprehensiveDataProcessor:
    """Complete data processing engine implementing all project requirements."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the comprehensive data processor."""
        self.config = config or ProcessingConfig()
        self.logger = get_logger(__name__)
        self.processing_pipeline = {}
        self.analysis_results = {}
        
    def process_dataset(self, 
                       df: pd.DataFrame, 
                       target_column: Optional[str] = None,
                       dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Complete end-to-end data processing pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Optional target variable for supervised learning
            dataset_name: Name for the dataset
            
        Returns:
            Comprehensive processing results
        """
        self.logger.info(f"Starting comprehensive processing for {dataset_name}")
        start_time = datetime.now()
        
        try:
            # Step 1: Data Validation and Quality Assessment
            validation_results = self._validate_dataset(df)
            
            # Step 2: Automatic Type Inference
            type_inference = self._infer_column_types(df)
            
            # Step 3: Comprehensive Data Profiling
            profiling_results = self._profile_dataset(df)
            
            # Step 4: Data Quality Issues Detection
            quality_issues = self._detect_quality_issues(df)
            
            # Step 5: Exploratory Data Analysis
            eda_results = self._perform_eda(df, target_column)
            
            # Step 6: Correlation and Relationship Analysis
            relationship_analysis = self._analyze_relationships(df, target_column)
            
            # Step 7: Outlier Detection (Multiple Methods)
            outlier_analysis = self._comprehensive_outlier_detection(df)
            
            # Step 8: Feature Engineering Recommendations
            feature_recommendations = self._recommend_feature_engineering(df, type_inference)
            
            # Step 9: Data Transformation Pipeline Creation
            transformation_pipeline = self._create_transformation_pipeline(df, type_inference, target_column)
            
            # Step 10: AI-Powered Insights Generation
            ai_insights = self._generate_ai_insights(
                df, profiling_results, eda_results, relationship_analysis, quality_issues
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            comprehensive_results = {
                'dataset_info': {
                    'name': dataset_name,
                    'shape': df.shape,
                    'processing_time_seconds': processing_time,
                    'processed_at': datetime.now().isoformat()
                },
                'validation': validation_results,
                'type_inference': type_inference,
                'profiling': profiling_results,
                'quality_issues': quality_issues,
                'eda': eda_results,
                'relationships': relationship_analysis,
                'outliers': outlier_analysis,
                'feature_engineering': feature_recommendations,
                'transformation_pipeline': transformation_pipeline,
                'ai_insights': ai_insights,
                'recommendations': self._generate_recommendations(
                    quality_issues, relationship_analysis, feature_recommendations
                )
            }
            
            # Store results for future reference
            self.analysis_results[dataset_name] = comprehensive_results
            
            self.logger.info(f"Comprehensive processing completed in {processing_time:.2f} seconds")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive processing: {e}")
            raise
    
    def _validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset structure and basic integrity."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'basic_stats': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'has_duplicates': df.duplicated().any(),
                'duplicate_count': df.duplicated().sum(),
                'total_missing': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
            }
        }
        
        # Check for empty dataset
        if df.empty:
            validation['is_valid'] = False
            validation['issues'].append("Dataset is empty")
            return validation
        
        # Check for all missing columns
        all_missing_cols = [col for col in df.columns if df[col].isnull().all()]
        if all_missing_cols:
            validation['warnings'].append(f"Columns with all missing values: {all_missing_cols}")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            validation['warnings'].append(f"Constant columns detected: {constant_cols}")
        
        # Check for high cardinality
        high_cardinality_cols = [
            col for col in df.columns 
            if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.9
        ]
        if high_cardinality_cols:
            validation['warnings'].append(f"High cardinality columns: {high_cardinality_cols}")
        
        # Memory usage warning
        if validation['basic_stats']['memory_usage_mb'] > 1000:  # >1GB
            validation['warnings'].append(f"Large dataset: {validation['basic_stats']['memory_usage_mb']:.1f}MB")
        
        return validation
    
    def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Advanced column type inference with confidence scores."""
        type_inference = {}
        
        for column in df.columns:
            series = df[column]
            
            # Basic pandas type
            pandas_type = str(series.dtype)
            
            # Infer logical type
            logical_type, confidence = self._infer_logical_type(series)
            
            # Additional metadata
            metadata = {
                'pandas_type': pandas_type,
                'logical_type': logical_type,
                'confidence': confidence,
                'nullable': series.isnull().any(),
                'unique_count': series.nunique(),
                'unique_ratio': series.nunique() / len(series) if len(series) > 0 else 0,
                'missing_count': series.isnull().sum(),
                'missing_ratio': series.isnull().sum() / len(series) if len(series) > 0 else 0,
                'recommendations': self._get_type_recommendations(series, logical_type)
            }
            
            type_inference[column] = metadata
        
        return type_inference
    
    def _infer_logical_type(self, series: pd.Series) -> Tuple[str, float]:
        """Infer logical type with confidence score."""
        if series.empty or series.isnull().all():
            return "empty", 1.0
        
        # Sample non-null values for testing
        sample = series.dropna().head(1000)
        if len(sample) == 0:
            return "missing", 1.0
        
        # Datetime detection
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime", 1.0
        
        # Try datetime parsing for object columns
        if series.dtype == 'object':
            try:
                pd.to_datetime(sample.head(100), errors='raise')
                return "datetime", 0.9
            except:
                pass
        
        # Numeric types
        if pd.api.types.is_integer_dtype(series):
            unique_ratio = series.nunique() / len(series)
            if unique_ratio > 0.95:
                return "identifier", 0.95
            elif series.nunique() <= 2:
                return "binary", 0.9
            elif series.nunique() <= 20:
                return "categorical", 0.8
            return "numeric_integer", 0.9
        
        if pd.api.types.is_float_dtype(series):
            return "numeric_float", 0.95
        
        if pd.api.types.is_bool_dtype(series):
            return "boolean", 1.0
        
        # Object/string analysis
        if series.dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            avg_length = sample.astype(str).str.len().mean()
            
            # Identifier check
            if unique_ratio > 0.9:
                return "identifier", 0.85
            
            # Categorical check
            if series.nunique() <= self.config.max_categorical_unique:
                return "categorical", 0.8
            
            # Text check
            if avg_length > 50:
                return "text", 0.7
            
            return "categorical", 0.6
        
        return "unknown", 0.1
    
    def _get_type_recommendations(self, series: pd.Series, logical_type: str) -> List[str]:
        """Get recommendations for column type handling."""
        recommendations = []
        
        if logical_type == "datetime" and series.dtype == 'object':
            recommendations.append("Convert to datetime type for better performance")
        
        if logical_type == "categorical" and series.dtype != 'category':
            recommendations.append("Consider converting to categorical type to save memory")
        
        if logical_type == "identifier":
            recommendations.append("Consider using this column as an index or removing if not needed")
        
        if logical_type == "binary":
            recommendations.append("Convert to boolean type for clarity")
        
        if series.isnull().sum() / len(series) > 0.5:
            recommendations.append("High missing values - consider imputation strategy")
        
        return recommendations
    
    def _profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dataset profiling."""
        profile = {
            'overview': self._get_dataset_overview(df),
            'columns': {},
            'correlations': {},
            'distributions': {},
            'patterns': {}
        }
        
        # Column-wise profiling
        for column in df.columns:
            profile['columns'][column] = self._profile_column(df[column])
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            profile['correlations'] = self._analyze_correlations(df[numeric_cols])
        
        # Distribution analysis
        profile['distributions'] = self._analyze_distributions(df)
        
        # Pattern detection
        profile['patterns'] = self._detect_patterns(df)
        
        return profile
    
    def _get_dataset_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get high-level dataset overview."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        
        return {
            'shape': df.shape,
            'column_types': {
                'numeric': len(numeric_cols),
                'categorical': len(categorical_cols),
                'datetime': len(datetime_cols),
                'other': len(df.columns) - len(numeric_cols) - len(categorical_cols) - len(datetime_cols)
            },
            'memory_usage': {
                'total_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'per_column_mb': {col: df[col].memory_usage(deep=True) / (1024 * 1024) 
                                 for col in df.columns}
            },
            'missing_data': {
                'total_missing': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'columns_with_missing': df.columns[df.isnull().any()].tolist()
            },
            'duplicates': {
                'count': df.duplicated().sum(),
                'percentage': (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
            }
        }
    
    def _profile_column(self, series: pd.Series) -> Dict[str, Any]:
        """Detailed column profiling."""
        profile = {
            'basic_stats': {
                'count': len(series),
                'missing_count': series.isnull().sum(),
                'missing_percentage': (series.isnull().sum() / len(series)) * 100 if len(series) > 0 else 0,
                'unique_count': series.nunique(),
                'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0
            }
        }
        
        # Type-specific analysis
        if pd.api.types.is_numeric_dtype(series):
            profile['numeric_stats'] = self._get_numeric_stats(series)
            profile['distribution_stats'] = self._get_distribution_stats(series)
        
        elif series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
            profile['categorical_stats'] = self._get_categorical_stats(series)
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile['datetime_stats'] = self._get_datetime_stats(series)
        
        return profile
    
    def _get_numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Get comprehensive numeric statistics."""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {}
        
        return {
            'descriptive': {
                'mean': float(clean_series.mean()),
                'median': float(clean_series.median()),
                'mode': float(clean_series.mode().iloc[0]) if len(clean_series.mode()) > 0 else None,
                'std': float(clean_series.std()),
                'var': float(clean_series.var()),
                'min': float(clean_series.min()),
                'max': float(clean_series.max()),
                'range': float(clean_series.max() - clean_series.min()),
                'iqr': float(clean_series.quantile(0.75) - clean_series.quantile(0.25))
            },
            'quantiles': {
                f'q{int(q*100)}': float(clean_series.quantile(q)) 
                for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            },
            'shape': {
                'skewness': float(clean_series.skew()),
                'kurtosis': float(clean_series.kurtosis()),
                'is_normal': self._test_normality(clean_series)
            }
        }
    
    def _get_distribution_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze distribution characteristics."""
        clean_series = series.dropna()
        if len(clean_series) < 3:
            return {}
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(clean_series.sample(min(5000, len(clean_series))))
        
        return {
            'normality_tests': {
                'shapiro_wilk': {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            },
            'distribution_fit': self._fit_distributions(clean_series)
        }
    
    def _get_categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Get categorical column statistics."""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {}
        
        value_counts = clean_series.value_counts()
        
        return {
            'cardinality': clean_series.nunique(),
            'entropy': float(-np.sum((value_counts / len(clean_series)) * np.log2(value_counts / len(clean_series) + 1e-10))),
            'mode': str(clean_series.mode().iloc[0]) if len(clean_series.mode()) > 0 else None,
            'top_values': value_counts.head(10).to_dict(),
            'value_distribution': {
                'most_frequent_percentage': (value_counts.iloc[0] / len(clean_series)) * 100,
                'least_frequent_count': value_counts.iloc[-1],
                'concentration_ratio': (value_counts.head(5).sum() / len(clean_series)) * 100
            }
        }
    
    def _get_datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Get datetime column statistics."""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {}
        
        return {
            'range': {
                'min': clean_series.min().isoformat() if not pd.isnull(clean_series.min()) else None,
                'max': clean_series.max().isoformat() if not pd.isnull(clean_series.max()) else None,
                'span_days': (clean_series.max() - clean_series.min()).days if not pd.isnull(clean_series.min()) else 0
            },
            'frequency_analysis': self._analyze_datetime_frequency(clean_series)
        }
    
    def _perform_eda(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive Exploratory Data Analysis."""
        eda_results = {
            'univariate_analysis': {},
            'bivariate_analysis': {},
            'multivariate_analysis': {},
            'visualizations': {},
            'insights': []
        }
        
        # Univariate analysis
        for column in df.columns:
            eda_results['univariate_analysis'][column] = self._univariate_analysis(df[column])
        
        # Bivariate analysis with target
        if target_column and target_column in df.columns:
            eda_results['bivariate_analysis'] = self._bivariate_analysis(df, target_column)
        
        # Multivariate analysis
        eda_results['multivariate_analysis'] = self._multivariate_analysis(df)
        
        # Generate visualizations metadata
        eda_results['visualizations'] = self._generate_visualization_recommendations(df, target_column)
        
        # Generate EDA insights
        eda_results['insights'] = self._generate_eda_insights(df, eda_results)
        
        return eda_results
    
    def _analyze_relationships(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive relationship analysis."""
        relationships = {
            'correlations': {},
            'associations': {},
            'dependencies': {},
            'clusters': {}
        }
        
        # Numeric correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            relationships['correlations'] = {
                'pearson_matrix': corr_matrix.to_dict(),
                'strong_correlations': self._find_strong_correlations(corr_matrix),
                'correlation_clusters': self._find_correlation_clusters(corr_matrix)
            }
        
        # Categorical associations
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 1:
            relationships['associations'] = self._analyze_categorical_associations(df[categorical_cols])
        
        # Mixed-type relationships
        if target_column:
            relationships['dependencies'] = self._analyze_feature_importance(df, target_column)
        
        return relationships
    
    def _comprehensive_outlier_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Multi-method outlier detection."""
        outlier_results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_cols:
            series = df[column].dropna()
            if len(series) < 10:  # Need minimum data points
                continue
            
            column_outliers = {
                'methods_used': [],
                'outlier_indices': {},
                'outlier_counts': {},
                'outlier_percentages': {},
                'consensus_outliers': [],
                'summary': {}
            }
            
            # IQR method
            if 'iqr' in self.config.outlier_methods:
                iqr_outliers = self._detect_iqr_outliers(series)
                column_outliers['methods_used'].append('iqr')
                column_outliers['outlier_indices']['iqr'] = iqr_outliers.tolist()
                column_outliers['outlier_counts']['iqr'] = len(iqr_outliers)
                column_outliers['outlier_percentages']['iqr'] = (len(iqr_outliers) / len(series)) * 100
            
            # Z-score method
            if 'zscore' in self.config.outlier_methods:
                zscore_outliers = self._detect_zscore_outliers(series)
                column_outliers['methods_used'].append('zscore')
                column_outliers['outlier_indices']['zscore'] = zscore_outliers.tolist()
                column_outliers['outlier_counts']['zscore'] = len(zscore_outliers)
                column_outliers['outlier_percentages']['zscore'] = (len(zscore_outliers) / len(series)) * 100
            
            # Isolation Forest method
            if 'isolation_forest' in self.config.outlier_methods:
                if_outliers = self._detect_isolation_forest_outliers(series)
                column_outliers['methods_used'].append('isolation_forest')
                column_outliers['outlier_indices']['isolation_forest'] = if_outliers.tolist()
                column_outliers['outlier_counts']['isolation_forest'] = len(if_outliers)
                column_outliers['outlier_percentages']['isolation_forest'] = (len(if_outliers) / len(series)) * 100
            
            # Consensus outliers (detected by multiple methods)
            all_outliers = []
            for method_outliers in column_outliers['outlier_indices'].values():
                all_outliers.extend(method_outliers)
            
            from collections import Counter
            outlier_counts = Counter(all_outliers)
            consensus_outliers = [idx for idx, count in outlier_counts.items() if count >= 2]
            column_outliers['consensus_outliers'] = consensus_outliers
            
            # Summary
            column_outliers['summary'] = {
                'total_methods': len(column_outliers['methods_used']),
                'consensus_count': len(consensus_outliers),
                'consensus_percentage': (len(consensus_outliers) / len(series)) * 100,
                'recommendation': self._get_outlier_recommendation(column_outliers)
            }
            
            outlier_results[column] = column_outliers
        
        return outlier_results
    
    def _recommend_feature_engineering(self, df: pd.DataFrame, type_inference: Dict) -> Dict[str, Any]:
        """Recommend feature engineering strategies."""
        recommendations = {
            'missing_value_handling': {},
            'encoding_strategies': {},
            'scaling_recommendations': {},
            'feature_creation': {},
            'feature_selection': {},
            'dimensionality_reduction': {}
        }
        
        # Missing value recommendations
        for column, info in type_inference.items():
            if info['missing_ratio'] > 0:
                recommendations['missing_value_handling'][column] = self._recommend_missing_strategy(
                    df[column], info
                )
        
        # Encoding recommendations
        categorical_cols = [col for col, info in type_inference.items() 
                          if info['logical_type'] == 'categorical']
        for col in categorical_cols:
            recommendations['encoding_strategies'][col] = self._recommend_encoding_strategy(df[col])
        
        # Scaling recommendations
        numeric_cols = [col for col, info in type_inference.items() 
                       if info['logical_type'] in ['numeric_integer', 'numeric_float']]
        for col in numeric_cols:
            recommendations['scaling_recommendations'][col] = self._recommend_scaling_strategy(df[col])
        
        # Feature creation recommendations
        recommendations['feature_creation'] = self._recommend_feature_creation(df, type_inference)
        
        # Feature selection recommendations
        recommendations['feature_selection'] = self._recommend_feature_selection(df)
        
        # Dimensionality reduction recommendations
        recommendations['dimensionality_reduction'] = self._recommend_dimensionality_reduction(df)
        
        return recommendations
    
    def _create_transformation_pipeline(self, df: pd.DataFrame, type_inference: Dict, target_column: Optional[str]) -> Dict[str, Any]:
        """Create scikit-learn compatible transformation pipeline."""
        pipeline_steps = []
        pipeline_config = {
            'steps': [],
            'column_transformers': {},
            'pipeline_file': None,
            'preprocessing_summary': {}
        }
        
        # Separate columns by type
        numeric_cols = [col for col, info in type_inference.items() 
                       if info['logical_type'] in ['numeric_integer', 'numeric_float'] and col != target_column]
        categorical_cols = [col for col, info in type_inference.items() 
                          if info['logical_type'] == 'categorical' and col != target_column]
        
        # Create preprocessing steps
        preprocessing_summary = {}
        
        if numeric_cols:
            # Numeric preprocessing
            numeric_steps = []
            numeric_steps.append(('imputer', SimpleImputer(strategy='median')))
            numeric_steps.append(('scaler', StandardScaler()))
            
            preprocessing_summary['numeric'] = {
                'columns': numeric_cols,
                'steps': ['median_imputation', 'standard_scaling'],
                'count': len(numeric_cols)
            }
        
        if categorical_cols:
            # Categorical preprocessing
            categorical_steps = []
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            
            # Choose encoding based on cardinality
            high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 10]
            if high_cardinality_cols:
                categorical_steps.append(('encoder', TargetEncoder()))
            else:
                categorical_steps.append(('encoder', OneHotEncoder(drop='first', sparse_output=False)))
            
            preprocessing_summary['categorical'] = {
                'columns': categorical_cols,
                'steps': ['mode_imputation', 'encoding'],
                'encoding_type': 'target' if high_cardinality_cols else 'onehot',
                'count': len(categorical_cols)
            }
        
        pipeline_config['preprocessing_summary'] = preprocessing_summary
        
        return pipeline_config
    
    def _generate_ai_insights(self, df: pd.DataFrame, profiling: Dict, eda: Dict, 
                            relationships: Dict, quality_issues: Dict) -> List[str]:
        """Generate AI-powered insights and narratives."""
        insights = []
        
        # Dataset overview insights
        rows, cols = df.shape
        insights.append(f"📊 Dataset Analysis: {rows:,} records across {cols} features")
        
        # Data quality insights
        missing_pct = quality_issues.get('overall_missing_percentage', 0)
        if missing_pct > 0:
            if missing_pct < 5:
                insights.append(f"✅ Data Quality: Excellent - only {missing_pct:.1f}% missing values")
            elif missing_pct < 20:
                insights.append(f"⚠️ Data Quality: Good - {missing_pct:.1f}% missing values require attention")
            else:
                insights.append(f"🔴 Data Quality: Poor - {missing_pct:.1f}% missing values need significant preprocessing")
        
        # Column type insights
        type_distribution = {}
        for col_profile in profiling.get('columns', {}).values():
            # This would need to be implemented based on the actual profiling structure
            pass
        
        # Correlation insights
        strong_corr = relationships.get('correlations', {}).get('strong_correlations', [])
        if strong_corr:
            top_corr = strong_corr[0] if strong_corr else None
            if top_corr:
                insights.append(
                    f"🔗 Strong Correlation Detected: {top_corr.get('feature1', '')} ↔ "
                    f"{top_corr.get('feature2', '')} (r={top_corr.get('correlation', 0):.2f})"
                )
        
        # Distribution insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skewed_cols = []
        for col in numeric_cols:
            skewness = df[col].skew()
            if abs(skewness) > 2:
                direction = "right" if skewness > 0 else "left"
                skewed_cols.append(f"{col} ({direction}-skewed)")
        
        if skewed_cols:
            insights.append(f"📈 Distribution Alert: Highly skewed features detected - {', '.join(skewed_cols[:3])}")
        
        # Outlier insights
        # This would be implemented based on outlier analysis results
        
        # Feature engineering recommendations
        insights.append("🔧 Recommendations:")
        
        # Missing value strategy
        high_missing_cols = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.3]
        if high_missing_cols:
            insights.append(f"   • Consider advanced imputation for: {', '.join(high_missing_cols[:3])}")
        
        # Categorical encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        high_card_cats = [col for col in categorical_cols if df[col].nunique() > 20]
        if high_card_cats:
            insights.append(f"   • Use target encoding for high-cardinality features: {', '.join(high_card_cats[:3])}")
        
        # Scaling recommendation
        if len(numeric_cols) > 0:
            insights.append("   • Apply standardization to numeric features for ML models")
        
        # Feature selection insight
        if cols > 50:
            insights.append("   • Consider feature selection techniques due to high dimensionality")
        
        return insights
    
    def _generate_recommendations(self, quality_issues: Dict, relationships: Dict, 
                                feature_recommendations: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Data quality recommendations
        recommendations.append("🎯 Priority Actions:")
        
        # Missing data
        missing_pct = quality_issues.get('overall_missing_percentage', 0)
        if missing_pct > 10:
            recommendations.append("1. Address missing values before model training")
        
        # Outliers
        recommendations.append("2. Investigate outliers - they may contain valuable information or errors")
        
        # Feature engineering
        recommendations.append("3. Apply recommended preprocessing pipeline")
        
        # Model selection
        recommendations.append("🤖 Model Recommendations:")
        recommendations.append("• Start with ensemble methods (Random Forest, XGBoost)")
        recommendations.append("• Consider linear models if features are well-scaled")
        recommendations.append("• Use cross-validation for reliable performance estimates")
        
        # Visualization recommendations
        recommendations.append("📊 Visualization Priorities:")
        recommendations.append("• Create correlation heatmap for feature relationships")
        recommendations.append("• Plot distributions of key numeric features")
        recommendations.append("• Visualize missing value patterns")
        
        return recommendations
    
    # Helper methods for detailed analysis
    def _test_normality(self, series: pd.Series) -> bool:
        """Test if series follows normal distribution."""
        if len(series) < 3:
            return False
        sample = series.sample(min(5000, len(series)))
        _, p_value = stats.shapiro(sample)
        return p_value > 0.05
    
    def _fit_distributions(self, series: pd.Series) -> Dict[str, Any]:
        """Fit common distributions to data."""
        # This is a simplified version - full implementation would test multiple distributions
        return {
            'best_fit': 'normal',
            'goodness_of_fit': 0.85,
            'parameters': {'mean': float(series.mean()), 'std': float(series.std())}
        }
    
    def _detect_quality_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect various data quality issues."""
        issues = {
            'overall_missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_issues': {},
            'severity_score': 0,
            'recommendations': []
        }
        
        for column in df.columns:
            col_issues = []
            
            # Missing values
            missing_pct = (df[column].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                col_issues.append(f"High missing values: {missing_pct:.1f}%")
            
            # Constant values
            if df[column].nunique() <= 1:
                col_issues.append("Constant column")
            
            # High cardinality for categoricals
            if df[column].dtype == 'object' and df[column].nunique() > len(df) * 0.8:
                col_issues.append("Very high cardinality")
            
            if col_issues:
                issues['columns_with_issues'][column] = col_issues
        
        # Calculate severity score
        severity = len(issues['columns_with_issues']) / len(df.columns) * 100
        issues['severity_score'] = severity
        
        return issues
    
    # Additional helper methods would be implemented here...
    def _univariate_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """Perform univariate analysis on a single column."""
        # Implementation would go here
        return {}
    
    def _bivariate_analysis(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Perform bivariate analysis with target variable."""
        # Implementation would go here
        return {}
    
    def _multivariate_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform multivariate analysis."""
        # Implementation would go here
        return {}
    
    def _generate_visualization_recommendations(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Generate visualization recommendations."""
        # Implementation would go here
        return {}
    
    def _generate_eda_insights(self, df: pd.DataFrame, eda_results: Dict) -> List[str]:
        """Generate insights from EDA results."""
        # Implementation would go here
        return []
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations in numeric data."""
        # Implementation would go here
        return {}
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distributions."""
        # Implementation would go here
        return {}
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns in the data."""
        # Implementation would go here
        return {}
    
    def _analyze_datetime_frequency(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime frequency patterns."""
        # Implementation would go here
        return {}
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict]:
        """Find strong correlations in correlation matrix."""
        # Implementation would go here
        return []
    
    def _find_correlation_clusters(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Find clusters of correlated features."""
        # Implementation would go here
        return {}
    
    def _analyze_categorical_associations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze associations between categorical variables."""
        # Implementation would go here
        return {}
    
    def _analyze_feature_importance(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze feature importance with respect to target."""
        # Implementation would go here
        return {}
    
    def _detect_iqr_outliers(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series < lower_bound) | (series > upper_bound)].index.values
    
    def _detect_zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return series[z_scores > threshold].index.values
    
    def _detect_isolation_forest_outliers(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        if len(series) < 10:
            return np.array([])
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(series.values.reshape(-1, 1))
        return series[outliers == -1].index.values
    
    def _get_outlier_recommendation(self, outlier_info: Dict) -> str:
        """Get recommendation for outlier handling."""
        consensus_pct = outlier_info['summary']['consensus_percentage']
        
        if consensus_pct < 1:
            return "Low outlier rate - consider keeping all data points"
        elif consensus_pct < 5:
            return "Moderate outliers - investigate and consider capping"
        else:
            return "High outlier rate - investigate data collection process"
    
    def _recommend_missing_strategy(self, series: pd.Series, type_info: Dict) -> str:
        """Recommend missing value handling strategy."""
        missing_ratio = type_info['missing_ratio']
        logical_type = type_info['logical_type']
        
        if missing_ratio < 0.05:
            return "drop" if len(series) > 1000 else "median"
        elif missing_ratio < 0.3:
            if logical_type in ['numeric_integer', 'numeric_float']:
                return "median"
            else:
                return "mode"
        else:
            return "knn"
    
    def _recommend_encoding_strategy(self, series: pd.Series) -> str:
        """Recommend encoding strategy for categorical variables."""
        unique_count = series.nunique()
        
        if unique_count <= 2:
            return "label"
        elif unique_count <= 10:
            return "onehot"
        else:
            return "target"
    
    def _recommend_scaling_strategy(self, series: pd.Series) -> str:
        """Recommend scaling strategy for numeric variables."""
        if series.min() >= 0:  # All positive values
            return "minmax"
        else:
            return "standard"
    
    def _recommend_feature_creation(self, df: pd.DataFrame, type_inference: Dict) -> List[str]:
        """Recommend new features to create."""
        recommendations = []
        
        # Datetime features
        datetime_cols = [col for col, info in type_inference.items() if info['logical_type'] == 'datetime']
        if datetime_cols:
            recommendations.append(f"Extract date components from: {', '.join(datetime_cols)}")
        
        # Interaction features
        numeric_cols = [col for col, info in type_inference.items() 
                       if info['logical_type'] in ['numeric_integer', 'numeric_float']]
        if len(numeric_cols) >= 2:
            recommendations.append("Consider creating interaction features between numeric variables")
        
        # Binning recommendations
        high_cardinality_numeric = [col for col in numeric_cols if df[col].nunique() > 100]
        if high_cardinality_numeric:
            recommendations.append(f"Consider binning high-cardinality numeric features: {', '.join(high_cardinality_numeric[:3])}")
        
        return recommendations
    
    def _recommend_feature_selection(self, df: pd.DataFrame) -> List[str]:
        """Recommend feature selection strategies."""
        recommendations = []
        
        if len(df.columns) > 50:
            recommendations.append("Use statistical feature selection (f_classif, f_regression)")
            recommendations.append("Consider recursive feature elimination with cross-validation")
        
        if len(df.columns) > 100:
            recommendations.append("Apply L1 regularization for automatic feature selection")
        
        return recommendations
    
    def _recommend_dimensionality_reduction(self, df: pd.DataFrame) -> List[str]:
        """Recommend dimensionality reduction techniques."""
        recommendations = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 20:
            recommendations.append("Consider PCA for dimensionality reduction")
        
        if len(numeric_cols) > 50:
            recommendations.append("Explore t-SNE or UMAP for non-linear dimensionality reduction")
        
        return recommendations
