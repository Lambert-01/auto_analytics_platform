"""Automated data profiling and exploratory data analysis engine."""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from app.models.dataset import ColumnInfo, DatasetType, DataQualityIssue
from app.models.analysis import (
    StatisticalSummary, CategoricalSummary, ColumnAnalysis,
    CorrelationAnalysis, OutlierDetectionResult
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProfilerConfig:
    """Configuration for data profiling."""
    max_unique_values_for_categorical: int = 50
    correlation_threshold: float = 0.7
    outlier_methods: List[str] = None
    sample_size_for_profiling: Optional[int] = None
    
    def __post_init__(self):
        if self.outlier_methods is None:
            self.outlier_methods = ['iqr', 'zscore']


class DataProfiler:
    """Automated data profiling and analysis engine."""
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        """Initialize the data profiler.
        
        Args:
            config: Profiler configuration
        """
        self.config = config or ProfilerConfig()
        self.logger = get_logger(__name__)
    
    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data profiling.
        
        Args:
            df: Dataset to profile
            
        Returns:
            Complete profiling results
        """
        try:
            self.logger.info(f"Starting data profiling for dataset with shape: {df.shape}")
            
            # Basic dataset information
            basic_info = self._get_basic_info(df)
            
            # Column-wise analysis
            column_analyses = self._analyze_columns(df)
            
            # Dataset type classification
            dataset_type = self._classify_dataset_type(df)
            
            # Data quality assessment
            quality_issues = self._assess_data_quality(df)
            
            # Correlation analysis (for numerical columns)
            correlation_analysis = self._analyze_correlations(df)
            
            # Outlier detection
            outlier_analysis = self._detect_outliers(df)
            
            # Generate insights and recommendations
            insights = self._generate_insights(df, column_analyses, correlation_analysis)
            recommendations = self._generate_recommendations(quality_issues, insights)
            
            profiling_results = {
                'basic_info': basic_info,
                'dataset_type': dataset_type,
                'column_analyses': column_analyses,
                'correlation_analysis': correlation_analysis,
                'outlier_analysis': outlier_analysis,
                'quality_issues': quality_issues,
                'insights': insights,
                'recommendations': recommendations,
                'profiling_timestamp': datetime.now(),
                'profiler_version': '1.0.0'
            }
            
            self.logger.info("Data profiling completed successfully")
            return profiling_results
            
        except Exception as e:
            self.logger.error(f"Error during data profiling: {e}")
            raise
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information.
        
        Args:
            df: Dataset
            
        Returns:
            Basic dataset information
        """
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'total_missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict()
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> List[ColumnAnalysis]:
        """Analyze each column in the dataset.
        
        Args:
            df: Dataset
            
        Returns:
            List of column analyses
        """
        column_analyses = []
        
        for column in df.columns:
            try:
                analysis = self._analyze_single_column(df, column)
                column_analyses.append(analysis)
            except Exception as e:
                self.logger.warning(f"Error analyzing column {column}: {e}")
                # Create a basic analysis with error info
                column_analyses.append(ColumnAnalysis(
                    column_name=column,
                    data_type=str(df[column].dtype),
                    is_numerical=False,
                    is_categorical=False,
                    is_datetime=False,
                    missing_count=df[column].isnull().sum(),
                    missing_percentage=(df[column].isnull().sum() / len(df)) * 100,
                    unique_count=df[column].nunique(),
                    unique_percentage=(df[column].nunique() / len(df)) * 100
                ))
        
        return column_analyses
    
    def _analyze_single_column(self, df: pd.DataFrame, column: str) -> ColumnAnalysis:
        """Analyze a single column.
        
        Args:
            df: Dataset
            column: Column name
            
        Returns:
            Column analysis
        """
        col_data = df[column]
        
        # Basic information
        is_numerical = pd.api.types.is_numeric_dtype(col_data)
        is_datetime = pd.api.types.is_datetime64_any_dtype(col_data)
        is_categorical = (
            pd.api.types.is_categorical_dtype(col_data) or
            pd.api.types.is_object_dtype(col_data) or
            col_data.nunique() <= self.config.max_unique_values_for_categorical
        )
        
        missing_count = col_data.isnull().sum()
        missing_percentage = (missing_count / len(col_data)) * 100
        unique_count = col_data.nunique()
        unique_percentage = (unique_count / len(col_data)) * 100
        
        # Statistical summaries
        numerical_summary = None
        categorical_summary = None
        
        if is_numerical and not col_data.empty:
            numerical_summary = self._get_numerical_summary(col_data)
        
        if is_categorical and not col_data.empty:
            categorical_summary = self._get_categorical_summary(col_data)
        
        # Outlier detection for numerical columns
        outlier_count = 0
        outlier_percentage = 0.0
        outlier_method = None
        
        if is_numerical:
            outliers = self._detect_column_outliers(col_data)
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(col_data)) * 100
            outlier_method = 'iqr'
        
        return ColumnAnalysis(
            column_name=column,
            data_type=str(col_data.dtype),
            is_numerical=is_numerical,
            is_categorical=is_categorical,
            is_datetime=is_datetime,
            numerical_summary=numerical_summary,
            categorical_summary=categorical_summary,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            outlier_count=outlier_count,
            outlier_percentage=outlier_percentage,
            outlier_method=outlier_method
        )
    
    def _get_numerical_summary(self, col_data: pd.Series) -> StatisticalSummary:
        """Get statistical summary for numerical column.
        
        Args:
            col_data: Column data
            
        Returns:
            Statistical summary
        """
        # Remove missing values for calculations
        clean_data = col_data.dropna()
        
        if len(clean_data) == 0:
            # Return zeros if no valid data
            return StatisticalSummary(
                count=0, mean=0, std=0, min=0, q25=0, q50=0, q75=0, max=0,
                skewness=0, kurtosis=0, variance=0
            )
        
        return StatisticalSummary(
            count=len(clean_data),
            mean=float(clean_data.mean()),
            std=float(clean_data.std()),
            min=float(clean_data.min()),
            q25=float(clean_data.quantile(0.25)),
            q50=float(clean_data.median()),
            q75=float(clean_data.quantile(0.75)),
            max=float(clean_data.max()),
            skewness=float(clean_data.skew()),
            kurtosis=float(clean_data.kurtosis()),
            variance=float(clean_data.var())
        )
    
    def _get_categorical_summary(self, col_data: pd.Series) -> CategoricalSummary:
        """Get summary for categorical column.
        
        Args:
            col_data: Column data
            
        Returns:
            Categorical summary
        """
        # Remove missing values for calculations
        clean_data = col_data.dropna()
        
        if len(clean_data) == 0:
            return CategoricalSummary(
                count=0, unique=0, top="", freq=0, mode="",
                entropy=0.0, value_counts={}
            )
        
        value_counts = clean_data.value_counts()
        top_value = value_counts.index[0] if len(value_counts) > 0 else ""
        top_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
        
        # Calculate entropy
        probabilities = value_counts / len(clean_data)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return CategoricalSummary(
            count=len(clean_data),
            unique=clean_data.nunique(),
            top=str(top_value),
            freq=int(top_freq),
            mode=str(clean_data.mode().iloc[0] if len(clean_data.mode()) > 0 else ""),
            entropy=float(entropy),
            value_counts=value_counts.head(20).to_dict()  # Top 20 values
        )
    
    def _classify_dataset_type(self, df: pd.DataFrame) -> DatasetType:
        """Classify the overall dataset type.
        
        Args:
            df: Dataset
            
        Returns:
            Dataset type classification
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        
        if len(datetime_cols) > 0 and len(datetime_cols) / len(df.columns) > 0.2:
            return DatasetType.TIME_SERIES
        elif len(numerical_cols) == len(df.columns):
            return DatasetType.NUMERICAL
        elif len(categorical_cols) == len(df.columns):
            return DatasetType.CATEGORICAL
        elif len(categorical_cols) > len(numerical_cols):
            return DatasetType.MIXED
        else:
            return DatasetType.MIXED
    
    def _assess_data_quality(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Assess data quality and identify issues.
        
        Args:
            df: Dataset
            
        Returns:
            List of data quality issues
        """
        issues = []
        
        # Check for missing values
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100
                severity = self._get_missing_value_severity(missing_percentage)
                
                issues.append(DataQualityIssue(
                    type="missing_values",
                    column=column,
                    count=missing_count,
                    percentage=missing_percentage,
                    description=f"Column '{column}' has {missing_count} missing values ({missing_percentage:.1f}%)",
                    severity=severity,
                    suggested_action=self._suggest_missing_value_action(missing_percentage, df[column].dtype)
                ))
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            issues.append(DataQualityIssue(
                type="duplicates",
                column=None,
                count=duplicate_count,
                percentage=duplicate_percentage,
                description=f"Dataset contains {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)",
                severity="medium" if duplicate_percentage > 5 else "low",
                suggested_action="Review and remove duplicate rows if appropriate"
            ))
        
        # Check for outliers in numerical columns
        for column in df.select_dtypes(include=[np.number]).columns:
            outliers = self._detect_column_outliers(df[column])
            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(df)) * 100
                issues.append(DataQualityIssue(
                    type="outliers",
                    column=column,
                    count=len(outliers),
                    percentage=outlier_percentage,
                    description=f"Column '{column}' contains {len(outliers)} potential outliers ({outlier_percentage:.1f}%)",
                    severity="low" if outlier_percentage < 5 else "medium",
                    suggested_action="Review outliers for data entry errors or valid extreme values"
                ))
        
        return issues
    
    def _get_missing_value_severity(self, percentage: float) -> str:
        """Determine severity of missing values.
        
        Args:
            percentage: Percentage of missing values
            
        Returns:
            Severity level
        """
        if percentage < 1:
            return "low"
        elif percentage < 5:
            return "medium"
        elif percentage < 20:
            return "high"
        else:
            return "critical"
    
    def _suggest_missing_value_action(self, percentage: float, dtype) -> str:
        """Suggest action for handling missing values.
        
        Args:
            percentage: Percentage of missing values
            dtype: Column data type
            
        Returns:
            Suggested action
        """
        if percentage > 50:
            return "Consider dropping this column due to excessive missing values"
        elif percentage > 20:
            return "Investigate missing value patterns and consider imputation or dropping rows"
        elif pd.api.types.is_numeric_dtype(dtype):
            return "Consider imputation with mean, median, or mode"
        else:
            return "Consider imputation with mode or a default category"
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Optional[CorrelationAnalysis]:
        """Analyze correlations between numerical columns.
        
        Args:
            df: Dataset
            
        Returns:
            Correlation analysis results
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return None
        
        try:
            corr_matrix = df[numerical_cols].corr()
            
            # Find strong correlations
            strong_correlations = []
            for i, col1 in enumerate(numerical_cols):
                for j, col2 in enumerate(numerical_cols):
                    if i < j:  # Avoid duplicates and self-correlation
                        correlation = corr_matrix.loc[col1, col2]
                        if abs(correlation) >= self.config.correlation_threshold:
                            strong_correlations.append({
                                'column1': col1,
                                'column2': col2,
                                'correlation': correlation,
                                'strength': 'strong' if abs(correlation) >= 0.8 else 'moderate'
                            })
            
            return CorrelationAnalysis(
                method="pearson",
                correlation_matrix=corr_matrix.to_dict(),
                strong_correlations=strong_correlations,
                correlation_threshold=self.config.correlation_threshold
            )
            
        except Exception as e:
            self.logger.warning(f"Error in correlation analysis: {e}")
            return None
    
    def _detect_outliers(self, df: pd.DataFrame) -> Optional[OutlierDetectionResult]:
        """Detect outliers in numerical columns.
        
        Args:
            df: Dataset
            
        Returns:
            Outlier detection results
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return None
        
        try:
            outliers_by_column = {}
            outlier_summary = {}
            total_outliers = 0
            
            for column in numerical_cols:
                outliers = self._detect_column_outliers(df[column])
                outliers_by_column[column] = outliers.tolist()
                outlier_summary[column] = len(outliers)
                total_outliers += len(outliers)
            
            outlier_percentage = (total_outliers / len(df)) * 100
            
            return OutlierDetectionResult(
                method="iqr",
                total_outliers=total_outliers,
                outlier_percentage=outlier_percentage,
                outliers_by_column=outliers_by_column,
                outlier_summary=outlier_summary
            )
            
        except Exception as e:
            self.logger.warning(f"Error in outlier detection: {e}")
            return None
    
    def _detect_column_outliers(self, col_data: pd.Series) -> pd.Index:
        """Detect outliers in a single column using IQR method.
        
        Args:
            col_data: Column data
            
        Returns:
            Index of outlier rows
        """
        clean_data = col_data.dropna()
        
        if len(clean_data) < 4:  # Need at least 4 values for quartiles
            return pd.Index([])
        
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        return col_data[outlier_mask].index
    
    def _generate_insights(self, df: pd.DataFrame, column_analyses: List[ColumnAnalysis], 
                          correlation_analysis: Optional[CorrelationAnalysis]) -> List[str]:
        """Generate insights about the dataset.
        
        Args:
            df: Dataset
            column_analyses: Column analysis results
            correlation_analysis: Correlation analysis results
            
        Returns:
            List of insights
        """
        insights = []
        
        # Dataset size insights
        rows, cols = df.shape
        insights.append(f"Dataset contains {rows:,} rows and {cols} columns")
        
        # Data type distribution
        numerical_count = sum(1 for col in column_analyses if col.is_numerical)
        categorical_count = sum(1 for col in column_analyses if col.is_categorical)
        
        if numerical_count > categorical_count:
            insights.append(f"Dataset is primarily numerical with {numerical_count} numerical and {categorical_count} categorical columns")
        elif categorical_count > numerical_count:
            insights.append(f"Dataset is primarily categorical with {categorical_count} categorical and {numerical_count} numerical columns")
        else:
            insights.append(f"Dataset has a balanced mix of numerical ({numerical_count}) and categorical ({categorical_count}) columns")
        
        # Missing value insights
        high_missing_cols = [col for col in column_analyses if col.missing_percentage > 10]
        if high_missing_cols:
            insights.append(f"{len(high_missing_cols)} columns have more than 10% missing values")
        
        # Correlation insights
        if correlation_analysis and correlation_analysis.strong_correlations:
            strong_count = len(correlation_analysis.strong_correlations)
            insights.append(f"Found {strong_count} strong correlations between numerical features")
        
        # Outlier insights
        outlier_cols = [col for col in column_analyses if col.outlier_percentage > 5]
        if outlier_cols:
            insights.append(f"{len(outlier_cols)} columns contain significant outliers (>5%)")
        
        return insights
    
    def _generate_recommendations(self, quality_issues: List[DataQualityIssue], 
                                insights: List[str]) -> List[str]:
        """Generate recommendations based on analysis.
        
        Args:
            quality_issues: Data quality issues
            insights: Generated insights
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Missing value recommendations
        critical_missing = [issue for issue in quality_issues 
                          if issue.type == "missing_values" and issue.severity == "critical"]
        if critical_missing:
            recommendations.append("Consider removing columns with excessive missing values (>50%)")
        
        high_missing = [issue for issue in quality_issues 
                       if issue.type == "missing_values" and issue.severity in ["high", "medium"]]
        if high_missing:
            recommendations.append("Implement appropriate imputation strategies for columns with missing values")
        
        # Duplicate recommendations
        duplicates = [issue for issue in quality_issues if issue.type == "duplicates"]
        if duplicates:
            recommendations.append("Review and remove duplicate rows to improve data quality")
        
        # Outlier recommendations
        outlier_issues = [issue for issue in quality_issues if issue.type == "outliers"]
        if outlier_issues:
            recommendations.append("Investigate outliers to distinguish between errors and valid extreme values")
        
        # General recommendations
        recommendations.append("Perform feature engineering to create new meaningful variables")
        recommendations.append("Consider feature scaling for machine learning applications")
        
        return recommendations


# Global profiler instance
data_profiler = DataProfiler()
