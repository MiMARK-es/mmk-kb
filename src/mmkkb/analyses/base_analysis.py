"""
Base analysis module for MMK Knowledge Base.
Provides common functionality for all analysis types including cross-validation.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation methods."""
    enable_loo: bool = True
    enable_bootstrap: bool = True
    bootstrap_iterations: int = 200
    bootstrap_validation_size: float = 0.2  # 20% of data for validation


@dataclass
class CrossValidationResults:
    """Results from cross-validation analysis."""
    loo_auc_mean: Optional[float] = None
    loo_auc_std: Optional[float] = None
    loo_aucs: Optional[List[float]] = None
    bootstrap_auc_mean: Optional[float] = None
    bootstrap_auc_std: Optional[float] = None
    bootstrap_aucs: Optional[List[float]] = None


@dataclass
class ROCCurvePoint:
    """Base class for ROC curve points - shared across all analysis types."""
    model_id: int
    fpr: float  # False Positive Rate
    tpr: float  # True Positive Rate
    threshold: float
    created_at: Optional[datetime] = None
    id: Optional[int] = None

@dataclass 
class ROCMetrics:
    """ROC Metrics for specific sensitivity thresholds - shared across all ROC analysis types."""
    model_id: int
    threshold_type: str  # 'se_97', 'se_95', 'max_sum'
    threshold: float
    sensitivity: float
    specificity: float
    npv: float
    ppv: float
    created_at: Optional[datetime] = None
    id: Optional[int] = None


class BaseAnalyzer(ABC):
    """Base class for all analysis types with common functionality."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
    
    def _calculate_threshold_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray,
                                   target_sensitivity: float, prevalence: float) -> Optional[Dict[str, float]]:
        """Calculate metrics for a target sensitivity threshold."""
        # Find threshold that achieves target sensitivity
        valid_indices = tpr >= target_sensitivity
        if not any(valid_indices):
            return None
        
        # Get the index with maximum specificity among those meeting sensitivity requirement
        valid_tpr = tpr[valid_indices]
        valid_fpr = fpr[valid_indices]
        valid_thresholds = thresholds[valid_indices]
        
        # Find minimum FPR (maximum specificity)
        min_fpr_idx = np.argmin(valid_fpr)
        
        chosen_tpr = valid_tpr[min_fpr_idx]
        chosen_fpr = valid_fpr[min_fpr_idx]
        chosen_threshold = valid_thresholds[min_fpr_idx]
        
        sensitivity = chosen_tpr
        specificity = 1 - chosen_fpr
        
        # Calculate PPV and NPV
        ppv = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
        npv = (specificity * (1 - prevalence)) / ((1 - sensitivity) * prevalence + specificity * (1 - prevalence))
        
        return {
            'threshold': float(chosen_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'npv': float(npv),
            'ppv': float(ppv)
        }
    
    def _calculate_max_sum_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                 fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray,
                                 prevalence: float) -> Dict[str, float]:
        """Calculate metrics for maximum (sensitivity + specificity)."""
        # Calculate sensitivity + specificity for each threshold
        sensitivity = tpr
        specificity = 1 - fpr
        sum_se_sp = sensitivity + specificity
        
        # Find threshold with maximum sum
        max_idx = np.argmax(sum_se_sp)
        
        chosen_sensitivity = sensitivity[max_idx]
        chosen_specificity = specificity[max_idx]
        chosen_threshold = thresholds[max_idx]
        
        # Calculate PPV and NPV
        ppv = (chosen_sensitivity * prevalence) / (chosen_sensitivity * prevalence + (1 - chosen_specificity) * (1 - prevalence))
        npv = (chosen_specificity * (1 - prevalence)) / ((1 - chosen_sensitivity) * prevalence + chosen_specificity * (1 - prevalence))
        
        return {
            'threshold': float(chosen_threshold),
            'sensitivity': float(chosen_sensitivity),
            'specificity': float(chosen_specificity),
            'npv': float(npv),
            'ppv': float(ppv)
        }
    
    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                cv_config: CrossValidationConfig) -> CrossValidationResults:
        """
        Perform cross-validation analysis on the data.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_config: Cross-validation configuration
            
        Returns:
            CrossValidationResults with LOO and Bootstrap results
        """
        results = CrossValidationResults()
        
        # Leave-One-Out Cross-Validation
        if cv_config.enable_loo:
            loo_aucs = self._perform_loo_cv(X, y)
            if loo_aucs:
                results.loo_aucs = loo_aucs
                results.loo_auc_mean = float(np.mean(loo_aucs))
                results.loo_auc_std = float(np.std(loo_aucs))
        
        # Bootstrap Cross-Validation
        if cv_config.enable_bootstrap:
            bootstrap_aucs = self._perform_bootstrap_cv(
                X, y, cv_config.bootstrap_iterations, cv_config.bootstrap_validation_size
            )
            if bootstrap_aucs:
                results.bootstrap_aucs = bootstrap_aucs
                results.bootstrap_auc_mean = float(np.mean(bootstrap_aucs))
                results.bootstrap_auc_std = float(np.std(bootstrap_aucs))
        
        return results
    
    def _perform_loo_cv(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Perform Leave-One-Out cross-validation."""
        loo = LeaveOneOut()
        aucs = []
        
        for train_idx, test_idx in loo.split(X):
            try:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Skip if we don't have both classes in training set
                if len(np.unique(y_train)) < 2:
                    continue
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Fit model
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate AUC (for single point, we can't calculate AUC directly)
                # Instead, we'll use the probability as a proxy or skip single-point folds
                if len(np.unique(y_test)) == 1:
                    # Single class in test set - use probability
                    aucs.append(float(y_pred_proba[0]) if y_test[0] == 1 else float(1 - y_pred_proba[0]))
                else:
                    # Multiple classes - calculate AUC
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    aucs.append(float(auc(fpr, tpr)))
                    
            except Exception:
                # Skip failed folds
                continue
        
        return aucs
    
    def _perform_bootstrap_cv(self, X: np.ndarray, y: np.ndarray, 
                            n_iterations: int, validation_size: float) -> List[float]:
        """Perform Bootstrap cross-validation."""
        aucs = []
        n_samples = len(X)
        n_validation = int(n_samples * validation_size)
        
        for _ in range(n_iterations):
            try:
                # Random sampling with replacement for training
                train_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                
                # Remaining samples for validation (without replacement)
                all_indices = set(range(n_samples))
                train_indices_set = set(train_indices)
                validation_candidates = list(all_indices - train_indices_set)
                
                # If not enough validation candidates, sample from all data
                if len(validation_candidates) < n_validation:
                    validation_indices = np.random.choice(n_samples, size=n_validation, replace=False)
                else:
                    validation_indices = np.random.choice(validation_candidates, size=n_validation, replace=False)
                
                X_train, X_val = X[train_indices], X[validation_indices]
                y_train, y_val = y[train_indices], y[validation_indices]
                
                # Skip if we don't have both classes
                if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                    continue
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Fit model
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                
                # Calculate AUC
                fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
                aucs.append(float(auc(fpr, tpr)))
                
            except Exception:
                # Skip failed iterations
                continue
        
        return aucs
    
    @abstractmethod
    def _prepare_experiment_data(self, experiment_id: int, **kwargs) -> Optional[Dict[str, Any]]:
        """Prepare experiment data for analysis. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _generate_biomarker_combinations(self, biomarker_versions: List[int], 
                                       max_size: int) -> List[List[int]]:
        """Generate biomarker combinations. Must be implemented by subclasses."""
        pass