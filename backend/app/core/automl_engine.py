"""
Advanced AutoML Engine for AI-Powered Analytics Platform
Implements: Model Selection, Training, Hyperparameter Optimization, Ensemble Methods
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

# Advanced ML
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
except ImportError:
    cb = None

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, classification_report, confusion_matrix
)

# Model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Local imports
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AutoMLConfig:
    """Configuration for AutoML operations."""
    # Time limits
    max_training_time_minutes: int = 30
    max_models_to_try: int = 20
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Hyperparameter optimization
    n_trials: int = 100
    optimization_direction: str = "maximize"  # for classification
    
    # Model selection
    include_ensemble: bool = True
    include_neural_networks: bool = False
    include_deep_learning: bool = False
    
    # Feature selection
    max_features_ratio: float = 0.8
    feature_selection_method: str = "auto"  # auto, univariate, recursive
    
    # Performance thresholds
    min_accuracy_threshold: float = 0.6
    min_r2_threshold: float = 0.5
    
    # Interpretability
    generate_feature_importance: bool = True
    generate_shap_values: bool = True
    
    # Ensemble settings
    ensemble_size: int = 5
    ensemble_methods: List[str] = None
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ['voting', 'stacking']


class AutoMLEngine:
    """Advanced AutoML engine with model selection and optimization."""
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        """Initialize the AutoML engine."""
        self.config = config or AutoMLConfig()
        self.logger = get_logger(__name__)
        
        # Model registries
        self.classification_models = self._initialize_classification_models()
        self.regression_models = self._initialize_regression_models()
        
        # Results storage
        self.results = {}
        self.best_models = {}
        self.model_performances = {}
        
    def train_automl(self, 
                     X: pd.DataFrame, 
                     y: pd.Series, 
                     problem_type: str = "auto",
                     target_metric: str = "auto") -> Dict[str, Any]:
        """
        Complete AutoML training pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            problem_type: 'classification', 'regression', or 'auto'
            target_metric: Metric to optimize for
            
        Returns:
            Complete AutoML results
        """
        self.logger.info(f"Starting AutoML training on dataset with shape {X.shape}")
        start_time = datetime.now()
        
        try:
            # Step 1: Problem type detection
            if problem_type == "auto":
                problem_type = self._detect_problem_type(y)
            
            self.logger.info(f"Detected problem type: {problem_type}")
            
            # Step 2: Target metric selection
            if target_metric == "auto":
                target_metric = self._select_target_metric(problem_type, y)
            
            # Step 3: Data preprocessing and validation
            X_processed, y_processed = self._preprocess_for_automl(X, y, problem_type)
            
            # Step 4: Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, 
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y_processed if problem_type == "classification" else None
            )
            
            # Step 5: Model selection and training
            models_results = self._train_multiple_models(
                X_train, y_train, X_test, y_test, problem_type, target_metric
            )
            
            # Step 6: Hyperparameter optimization for best models
            optimized_results = self._optimize_hyperparameters(
                X_train, y_train, X_test, y_test, 
                models_results, problem_type, target_metric
            )
            
            # Step 7: Ensemble creation
            ensemble_results = {}
            if self.config.include_ensemble:
                ensemble_results = self._create_ensembles(
                    X_train, y_train, X_test, y_test,
                    optimized_results, problem_type, target_metric
                )
            
            # Step 8: Model interpretation
            interpretation_results = self._interpret_models(
                X_train, optimized_results, problem_type
            )
            
            # Step 9: Final model selection
            best_model_info = self._select_best_model(
                optimized_results, ensemble_results, target_metric
            )
            
            # Step 10: Generate comprehensive results
            training_time = (datetime.now() - start_time).total_seconds()
            
            automl_results = {
                'training_info': {
                    'problem_type': problem_type,
                    'target_metric': target_metric,
                    'training_time_seconds': training_time,
                    'dataset_shape': X.shape,
                    'target_distribution': self._analyze_target_distribution(y_processed, problem_type),
                    'features_used': X_processed.columns.tolist(),
                    'preprocessing_applied': self._get_preprocessing_summary()
                },
                'model_results': {
                    'baseline_models': models_results,
                    'optimized_models': optimized_results,
                    'ensemble_models': ensemble_results,
                    'total_models_trained': len(models_results) + len(optimized_results) + len(ensemble_results)
                },
                'best_model': best_model_info,
                'model_interpretation': interpretation_results,
                'performance_summary': self._create_performance_summary(
                    models_results, optimized_results, ensemble_results
                ),
                'recommendations': self._generate_automl_recommendations(
                    best_model_info, problem_type, X.shape
                )
            }
            
            # Store results
            self.results = automl_results
            
            self.logger.info(f"AutoML training completed in {training_time:.2f} seconds")
            self.logger.info(f"Best model: {best_model_info['model_name']} with {target_metric}: {best_model_info['performance'][target_metric]:.4f}")
            
            return automl_results
            
        except Exception as e:
            self.logger.error(f"Error in AutoML training: {e}")
            raise
    
    def _detect_problem_type(self, y: pd.Series) -> str:
        """Automatically detect if problem is classification or regression."""
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(y):
            unique_values = y.nunique()
            total_values = len(y)
            
            # If few unique values relative to total, likely classification
            if unique_values <= 2:
                return "classification"
            elif unique_values < 20 and unique_values / total_values < 0.05:
                return "classification"
            else:
                return "regression"
        else:
            # Non-numeric is classification
            return "classification"
    
    def _select_target_metric(self, problem_type: str, y: pd.Series) -> str:
        """Select appropriate target metric based on problem type."""
        if problem_type == "classification":
            unique_classes = y.nunique()
            if unique_classes == 2:
                return "roc_auc"
            else:
                return "f1_weighted"
        else:  # regression
            return "r2"
    
    def _preprocess_for_automl(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data for AutoML training."""
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Handle missing values in features
        for column in X_processed.columns:
            if X_processed[column].isnull().any():
                if pd.api.types.is_numeric_dtype(X_processed[column]):
                    X_processed[column].fillna(X_processed[column].median(), inplace=True)
                else:
                    X_processed[column].fillna(X_processed[column].mode().iloc[0] if len(X_processed[column].mode()) > 0 else 'missing', inplace=True)
        
        # Encode categorical features
        categorical_columns = X_processed.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            if X_processed[column].nunique() <= 10:  # One-hot encode low cardinality
                dummies = pd.get_dummies(X_processed[column], prefix=column, drop_first=True)
                X_processed = pd.concat([X_processed.drop(column, axis=1), dummies], axis=1)
            else:  # Label encode high cardinality
                le = LabelEncoder()
                X_processed[column] = le.fit_transform(X_processed[column].astype(str))
        
        # Handle target variable
        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_processed):
            le = LabelEncoder()
            y_processed = pd.Series(le.fit_transform(y_processed), index=y_processed.index)
        
        # Remove any remaining non-numeric columns
        numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
        X_processed = X_processed[numeric_columns]
        
        self.logger.info(f"Preprocessing completed. Features: {X_processed.shape[1]}")
        return X_processed, y_processed
    
    def _initialize_classification_models(self) -> Dict[str, Any]:
        """Initialize classification models with default parameters."""
        models = {
            'dummy_classifier': {
                'model': DummyClassifier(strategy='most_frequent'),
                'params': {},
                'description': 'Baseline dummy classifier'
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.config.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs']
                },
                'description': 'Logistic regression with L2 regularization'
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'description': 'Random forest ensemble classifier'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'description': 'Gradient boosting classifier'
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=self.config.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9]
                },
                'description': 'XGBoost classifier'
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=self.config.random_state, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                },
                'description': 'LightGBM classifier'
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20]
                },
                'description': 'Extra trees classifier'
            }
        }
        
        # Add CatBoost if available
        if cb is not None:
            models['catboost'] = {
                'model': cb.CatBoostClassifier(random_state=self.config.random_state, verbose=False),
                'params': {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 6, 9]
                },
                'description': 'CatBoost classifier'
            }
        
        return models
    
    def _initialize_regression_models(self) -> Dict[str, Any]:
        """Initialize regression models with default parameters."""
        models = {
            'dummy_regressor': {
                'model': DummyRegressor(strategy='mean'),
                'params': {},
                'description': 'Baseline dummy regressor'
            },
            'linear_regression': {
                'model': LinearRegression(),
                'params': {},
                'description': 'Ordinary least squares linear regression'
            },
            'ridge_regression': {
                'model': Ridge(random_state=self.config.random_state),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                'description': 'Ridge regression with L2 regularization'
            },
            'lasso_regression': {
                'model': Lasso(random_state=self.config.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                'description': 'Lasso regression with L1 regularization'
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'description': 'Random forest ensemble regressor'
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'description': 'Gradient boosting regressor'
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9]
                },
                'description': 'XGBoost regressor'
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor(random_state=self.config.random_state, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                },
                'description': 'LightGBM regressor'
            },
            'extra_trees': {
                'model': ExtraTreesRegressor(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20]
                },
                'description': 'Extra trees regressor'
            }
        }
        
        # Add CatBoost if available
        if cb is not None:
            models['catboost'] = {
                'model': cb.CatBoostRegressor(random_state=self.config.random_state, verbose=False),
                'params': {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 6, 9]
                },
                'description': 'CatBoost regressor'
            }
        
        return models
    
    def _train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series,
                              problem_type: str, target_metric: str) -> Dict[str, Any]:
        """Train multiple baseline models."""
        models_dict = self.classification_models if problem_type == "classification" else self.regression_models
        results = {}
        
        self.logger.info(f"Training {len(models_dict)} baseline models")
        
        for model_name, model_config in models_dict.items():
            try:
                self.logger.info(f"Training {model_name}")
                
                # Train model
                model = model_config['model']
                model.fit(X_train, y_train)
                
                # Evaluate model
                performance = self._evaluate_model(
                    model, X_train, y_train, X_test, y_test, problem_type
                )
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'performance': performance,
                    'description': model_config['description'],
                    'model_type': 'baseline',
                    'hyperparameters': model.get_params(),
                    'feature_importance': self._get_feature_importance(model, X_train.columns) if hasattr(model, 'feature_importances_') else None
                }
                
                self.logger.info(f"{model_name} {target_metric}: {performance.get(target_metric, 'N/A'):.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {model_name}: {e}")
                continue
        
        return results
    
    def _evaluate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        performance = {}
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        if problem_type == "classification":
            # Classification metrics
            performance.update({
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'accuracy_test': accuracy_score(y_test, y_pred_test),
                'precision_train': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'precision_test': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall_train': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'recall_test': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1_train': f1_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'f1_test': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            })
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    performance['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                except:
                    pass
            
        else:  # regression
            # Regression metrics
            performance.update({
                'mse_train': mean_squared_error(y_train, y_pred_train),
                'mse_test': mean_squared_error(y_test, y_pred_test),
                'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae_train': mean_absolute_error(y_train, y_pred_train),
                'mae_test': mean_absolute_error(y_test, y_pred_test),
                'r2_train': r2_score(y_train, y_pred_train),
                'r2_test': r2_score(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            })
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.config.cv_folds,
                scoring='accuracy' if problem_type == "classification" else 'r2'
            )
            performance['cv_mean'] = cv_scores.mean()
            performance['cv_std'] = cv_scores.std()
        except:
            pass
        
        return performance
    
    def _optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 baseline_results: Dict, problem_type: str, 
                                 target_metric: str) -> Dict[str, Any]:
        """Optimize hyperparameters for best performing models."""
        # Select top models for optimization
        sorted_models = sorted(
            baseline_results.items(),
            key=lambda x: x[1]['performance'].get(target_metric, 0),
            reverse=True
        )
        
        top_models = sorted_models[:min(5, len(sorted_models))]  # Top 5 models
        optimized_results = {}
        
        self.logger.info(f"Optimizing hyperparameters for top {len(top_models)} models")
        
        for model_name, model_info in top_models:
            if model_name.startswith('dummy'):  # Skip dummy models
                continue
                
            try:
                self.logger.info(f"Optimizing {model_name}")
                
                # Get model configuration
                models_dict = self.classification_models if problem_type == "classification" else self.regression_models
                model_config = models_dict[model_name]
                
                # Optimize using Optuna
                optimized_model, best_params, best_score = self._optimize_with_optuna(
                    model_config['model'], model_config['params'],
                    X_train, y_train, problem_type, target_metric
                )
                
                # Evaluate optimized model
                performance = self._evaluate_model(
                    optimized_model, X_train, y_train, X_test, y_test, problem_type
                )
                
                optimized_results[f"{model_name}_optimized"] = {
                    'model': optimized_model,
                    'performance': performance,
                    'description': f"Hyperparameter optimized {model_config['description']}",
                    'model_type': 'optimized',
                    'hyperparameters': best_params,
                    'optimization_score': best_score,
                    'feature_importance': self._get_feature_importance(optimized_model, X_train.columns) if hasattr(optimized_model, 'feature_importances_') else None
                }
                
                self.logger.info(f"{model_name}_optimized {target_metric}: {performance.get(target_metric, 'N/A'):.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to optimize {model_name}: {e}")
                continue
        
        return optimized_results
    
    def _optimize_with_optuna(self, base_model, param_distributions: Dict,
                             X_train: pd.DataFrame, y_train: pd.Series,
                             problem_type: str, target_metric: str) -> Tuple[Any, Dict, float]:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            # Suggest hyperparameters
            params = {}
            for param_name, param_values in param_distributions.items():
                if isinstance(param_values, list):
                    if all(isinstance(x, (int, np.integer)) for x in param_values):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    elif all(isinstance(x, (float, np.floating)) for x in param_values):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Create model with suggested parameters
            model = base_model.__class__(**{**base_model.get_params(), **params})
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=3,  # Reduced for speed
                scoring='accuracy' if problem_type == "classification" else 'r2'
            )
            
            return cv_scores.mean()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.random_state)
        )
        
        # Optimize
        study.optimize(objective, n_trials=min(self.config.n_trials, 50))  # Reduced for speed
        
        # Create best model
        best_model = base_model.__class__(**{**base_model.get_params(), **study.best_params})
        best_model.fit(X_train, y_train)
        
        return best_model, study.best_params, study.best_value
    
    def _create_ensembles(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         model_results: Dict, problem_type: str, 
                         target_metric: str) -> Dict[str, Any]:
        """Create ensemble models from best performing models."""
        if not self.config.include_ensemble:
            return {}
        
        # Select best models for ensemble
        sorted_models = sorted(
            model_results.items(),
            key=lambda x: x[1]['performance'].get(target_metric, 0),
            reverse=True
        )
        
        top_models = sorted_models[:min(self.config.ensemble_size, len(sorted_models))]
        ensemble_results = {}
        
        if len(top_models) < 2:
            self.logger.warning("Not enough models for ensemble creation")
            return {}
        
        self.logger.info(f"Creating ensembles from top {len(top_models)} models")
        
        # Voting ensemble
        if 'voting' in self.config.ensemble_methods:
            try:
                estimators = [(name, info['model']) for name, info in top_models]
                
                if problem_type == "classification":
                    ensemble = VotingClassifier(estimators=estimators, voting='soft')
                else:
                    ensemble = VotingRegressor(estimators=estimators)
                
                ensemble.fit(X_train, y_train)
                performance = self._evaluate_model(
                    ensemble, X_train, y_train, X_test, y_test, problem_type
                )
                
                ensemble_results['voting_ensemble'] = {
                    'model': ensemble,
                    'performance': performance,
                    'description': f'Voting ensemble of top {len(top_models)} models',
                    'model_type': 'ensemble',
                    'ensemble_members': [name for name, _ in top_models],
                    'hyperparameters': ensemble.get_params()
                }
                
                self.logger.info(f"Voting ensemble {target_metric}: {performance.get(target_metric, 'N/A'):.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to create voting ensemble: {e}")
        
        return ensemble_results
    
    def _interpret_models(self, X_train: pd.DataFrame, model_results: Dict, 
                         problem_type: str) -> Dict[str, Any]:
        """Generate model interpretations and explanations."""
        interpretation = {
            'feature_importance': {},
            'shap_values': {},
            'model_explanations': {}
        }
        
        # Feature importance for tree-based models
        for model_name, model_info in model_results.items():
            model = model_info['model']
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = self._get_feature_importance(model, X_train.columns)
                interpretation['feature_importance'][model_name] = feature_importance
        
        # SHAP values (if available and requested)
        if SHAP_AVAILABLE and self.config.generate_shap_values:
            try:
                # Get best model for SHAP analysis
                best_model_name = max(model_results.keys(), 
                                    key=lambda k: model_results[k]['performance'].get('accuracy_test', 0) 
                                    if problem_type == "classification" else model_results[k]['performance'].get('r2_test', 0))
                
                best_model = model_results[best_model_name]['model']
                
                # Create SHAP explainer
                sample_size = min(100, len(X_train))  # Limit for performance
                X_sample = X_train.sample(n=sample_size, random_state=self.config.random_state)
                
                if hasattr(best_model, 'predict_proba'):
                    explainer = shap.TreeExplainer(best_model)
                else:
                    explainer = shap.Explainer(best_model, X_sample)
                
                shap_values = explainer.shap_values(X_sample)
                
                interpretation['shap_values'][best_model_name] = {
                    'values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                    'feature_names': X_train.columns.tolist(),
                    'sample_size': sample_size
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to generate SHAP values: {e}")
        
        return interpretation
    
    def _select_best_model(self, optimized_results: Dict, ensemble_results: Dict, 
                          target_metric: str) -> Dict[str, Any]:
        """Select the best performing model overall."""
        all_results = {**optimized_results, **ensemble_results}
        
        if not all_results:
            raise ValueError("No models available for selection")
        
        best_model_name = max(
            all_results.keys(),
            key=lambda k: all_results[k]['performance'].get(target_metric, 0)
        )
        
        best_model_info = all_results[best_model_name].copy()
        best_model_info['model_name'] = best_model_name
        best_model_info['target_metric'] = target_metric
        
        return best_model_info
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            importance_values = model.feature_importances_
            return dict(zip(feature_names, importance_values.tolist()))
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficient values
            coef_values = np.abs(model.coef_)
            if coef_values.ndim > 1:
                coef_values = coef_values.mean(axis=0)
            return dict(zip(feature_names, coef_values.tolist()))
        else:
            return {}
    
    def _analyze_target_distribution(self, y: pd.Series, problem_type: str) -> Dict[str, Any]:
        """Analyze target variable distribution."""
        if problem_type == "classification":
            value_counts = y.value_counts()
            return {
                'type': 'classification',
                'classes': len(value_counts),
                'distribution': value_counts.to_dict(),
                'is_balanced': (value_counts.max() / value_counts.min()) < 3 if len(value_counts) > 1 else True
            }
        else:
            return {
                'type': 'regression',
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max()),
                'skewness': float(y.skew()),
                'kurtosis': float(y.kurtosis())
            }
    
    def _get_preprocessing_summary(self) -> List[str]:
        """Get summary of preprocessing steps applied."""
        return [
            "Missing value imputation (median for numeric, mode for categorical)",
            "Categorical encoding (one-hot for low cardinality, label for high cardinality)",
            "Feature selection (numeric features only)"
        ]
    
    def _create_performance_summary(self, baseline_results: Dict, 
                                  optimized_results: Dict, 
                                  ensemble_results: Dict) -> Dict[str, Any]:
        """Create comprehensive performance summary."""
        all_results = {**baseline_results, **optimized_results, **ensemble_results}
        
        summary = {
            'total_models': len(all_results),
            'baseline_models': len(baseline_results),
            'optimized_models': len(optimized_results),
            'ensemble_models': len(ensemble_results),
            'model_types': {
                model_name: info['model_type'] 
                for model_name, info in all_results.items()
            }
        }
        
        return summary
    
    def _generate_automl_recommendations(self, best_model_info: Dict, 
                                       problem_type: str, 
                                       dataset_shape: Tuple[int, int]) -> List[str]:
        """Generate recommendations based on AutoML results."""
        recommendations = []
        
        # Model-specific recommendations
        model_name = best_model_info['model_name']
        performance = best_model_info['performance']
        
        recommendations.append(f"🏆 Best Model: {model_name}")
        
        # Performance recommendations
        if problem_type == "classification":
            accuracy = performance.get('accuracy_test', 0)
            if accuracy < 0.7:
                recommendations.append("⚠️ Model accuracy is below 70% - consider feature engineering")
            elif accuracy > 0.9:
                recommendations.append("✅ Excellent model performance achieved")
        else:  # regression
            r2 = performance.get('r2_test', 0)
            if r2 < 0.5:
                recommendations.append("⚠️ Model explains less than 50% of variance - consider more features")
            elif r2 > 0.8:
                recommendations.append("✅ Model explains most of the variance in target")
        
        # Data size recommendations
        rows, cols = dataset_shape
        if rows < 1000:
            recommendations.append("📊 Small dataset - consider collecting more data for better models")
        if cols > rows / 10:
            recommendations.append("🔍 High dimensionality - consider feature selection or regularization")
        
        # Model improvement suggestions
        recommendations.append("🚀 Next Steps:")
        recommendations.append("• Collect more training data if possible")
        recommendations.append("• Try feature engineering (interactions, transformations)")
        recommendations.append("• Consider deep learning for complex patterns")
        recommendations.append("• Implement model monitoring in production")
        
        return recommendations
    
    def save_model(self, model_path: str, model_name: str = None) -> str:
        """Save the best model to disk."""
        if not self.results:
            raise ValueError("No model results available. Run train_automl first.")
        
        best_model = self.results['best_model']['model']
        
        # Save model
        joblib.dump(best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name or self.results['best_model']['model_name'],
            'problem_type': self.results['training_info']['problem_type'],
            'target_metric': self.results['training_info']['target_metric'],
            'performance': self.results['best_model']['performance'],
            'features_used': self.results['training_info']['features_used'],
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> Tuple[Any, Dict]:
        """Load a saved model and its metadata."""
        model = joblib.load(model_path)
        
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}
        
        return model, metadata
