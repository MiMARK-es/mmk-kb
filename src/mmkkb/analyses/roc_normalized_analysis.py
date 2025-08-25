"""
ROC Normalized analysis module for MMK Knowledge Base.
Provides ROC curve analysis with biomarker normalization, where one biomarker is used 
to normalize all others (divide by normalizer) before performing ROC analysis.
"""
import sqlite3
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from ..config import get_database_path
from ..experiments import ExperimentDatabase
from ..samples import SampleDatabase
from .base_analysis import BaseAnalyzer, CrossValidationConfig, CrossValidationResults

@dataclass
class ROCNormalizedAnalysis:
    """ROC Normalized Analysis model representing an analysis run."""
    name: str
    description: str
    experiment_id: int
    normalizer_biomarker_version_id: int  # The biomarker used for normalization
    prevalence: float  # Expected prevalence for PPV/NPV calculations
    max_combination_size: int  # Maximum number of biomarkers per model
    cross_validation_config: Optional[CrossValidationConfig] = None
    created_at: Optional[datetime] = None
    id: Optional[int] = None

@dataclass
class ROCNormalizedModel:
    """ROC Normalized Model representing a specific biomarker combination model."""
    analysis_id: int
    biomarker_combination: List[int]  # List of biomarker_version_ids (normalized)
    normalizer_biomarker_version_id: int  # The normalizer biomarker
    auc: float
    coefficients: Dict[str, float]  # Model coefficients including intercept
    cross_validation_results: Optional[CrossValidationResults] = None
    created_at: Optional[datetime] = None
    id: Optional[int] = None

@dataclass
class ROCNormalizedMetrics:
    """ROC Normalized Metrics for specific sensitivity thresholds."""
    model_id: int
    threshold_type: str  # 'se_97', 'se_95', 'max_sum'
    threshold: float
    sensitivity: float
    specificity: float
    npv: float
    ppv: float
    created_at: Optional[datetime] = None
    id: Optional[int] = None

@dataclass
class ROCNormalizedCurvePoint:
    """Individual point on ROC Normalized curve."""
    model_id: int
    fpr: float  # False Positive Rate
    tpr: float  # True Positive Rate
    threshold: float
    created_at: Optional[datetime] = None
    id: Optional[int] = None

class ROCNormalizedAnalysisDatabase:
    """Database operations for ROC Normalized analysis."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_database_path()
        self.init_database()
    
    def init_database(self):
        """Initialize database with ROC Normalized analysis tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # ROC normalized analyses table - add cross-validation configuration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roc_normalized_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    experiment_id INTEGER NOT NULL,
                    normalizer_biomarker_version_id INTEGER NOT NULL,
                    prevalence REAL NOT NULL CHECK (prevalence > 0 AND prevalence < 1),
                    max_combination_size INTEGER NOT NULL CHECK (max_combination_size > 0),
                    cross_validation_config TEXT,  -- JSON config for cross-validation
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE,
                    FOREIGN KEY (normalizer_biomarker_version_id) REFERENCES biomarker_versions (id) ON DELETE CASCADE
                )
            """)
            
            # ROC normalized models table - add cross-validation results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roc_normalized_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER NOT NULL,
                    biomarker_combination TEXT NOT NULL,  -- JSON array of biomarker_version_ids
                    normalizer_biomarker_version_id INTEGER NOT NULL,
                    auc REAL NOT NULL,
                    coefficients TEXT NOT NULL,  -- JSON object with coefficients
                    cross_validation_results TEXT,  -- JSON object with CV results
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES roc_normalized_analyses (id) ON DELETE CASCADE,
                    FOREIGN KEY (normalizer_biomarker_version_id) REFERENCES biomarker_versions (id) ON DELETE CASCADE
                )
            """)
            
            # ROC normalized metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roc_normalized_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    threshold_type TEXT NOT NULL CHECK (threshold_type IN ('se_97', 'se_95', 'max_sum')),
                    threshold REAL NOT NULL,
                    sensitivity REAL NOT NULL,
                    specificity REAL NOT NULL,
                    npv REAL NOT NULL,
                    ppv REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES roc_normalized_models (id) ON DELETE CASCADE
                )
            """)
            
            # ROC normalized curve points table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roc_normalized_curve_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    fpr REAL NOT NULL,
                    tpr REAL NOT NULL,
                    threshold REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES roc_normalized_models (id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_roc_normalized_analyses_experiment_id ON roc_normalized_analyses (experiment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_roc_normalized_models_analysis_id ON roc_normalized_models (analysis_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_roc_normalized_metrics_model_id ON roc_normalized_metrics (model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_roc_normalized_curve_points_model_id ON roc_normalized_curve_points (model_id)")
            
            conn.commit()
    
    def _get_connection(self):
        """Get database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    # ROC Normalized Analysis operations
    def create_roc_normalized_analysis(self, analysis: ROCNormalizedAnalysis) -> ROCNormalizedAnalysis:
        """Create a new ROC Normalized analysis."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO roc_normalized_analyses (name, description, experiment_id, normalizer_biomarker_version_id, prevalence, max_combination_size, cross_validation_config)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (analysis.name, analysis.description, analysis.experiment_id, 
                  analysis.normalizer_biomarker_version_id, analysis.prevalence, analysis.max_combination_size, json.dumps(asdict(analysis.cross_validation_config)) if analysis.cross_validation_config else None))
            
            analysis.id = cursor.lastrowid
            
            row = conn.execute(
                "SELECT created_at FROM roc_normalized_analyses WHERE id = ?", 
                (analysis.id,)
            ).fetchone()
            if row:
                analysis.created_at = datetime.fromisoformat(row[0])
            
            conn.commit()
        return analysis
    
    def get_roc_normalized_analysis_by_id(self, analysis_id: int) -> Optional[ROCNormalizedAnalysis]:
        """Get ROC Normalized analysis by ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM roc_normalized_analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
            
            if row:
                return ROCNormalizedAnalysis(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    experiment_id=row["experiment_id"],
                    normalizer_biomarker_version_id=row["normalizer_biomarker_version_id"],
                    prevalence=row["prevalence"],
                    max_combination_size=row["max_combination_size"],
                    cross_validation_config=json.loads(row["cross_validation_config"]) if row["cross_validation_config"] else None,
                    created_at=datetime.fromisoformat(row["created_at"])
                )
        return None
    
    def list_roc_normalized_analyses(self, experiment_id: Optional[int] = None) -> List[ROCNormalizedAnalysis]:
        """List ROC Normalized analyses, optionally filtered by experiment."""
        analyses = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            
            if experiment_id:
                query = "SELECT * FROM roc_normalized_analyses WHERE experiment_id = ? ORDER BY created_at DESC"
                rows = conn.execute(query, (experiment_id,)).fetchall()
            else:
                query = "SELECT * FROM roc_normalized_analyses ORDER BY created_at DESC"
                rows = conn.execute(query).fetchall()
            
            for row in rows:
                analyses.append(ROCNormalizedAnalysis(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    experiment_id=row["experiment_id"],
                    normalizer_biomarker_version_id=row["normalizer_biomarker_version_id"],
                    prevalence=row["prevalence"],
                    max_combination_size=row["max_combination_size"],
                    cross_validation_config=json.loads(row["cross_validation_config"]) if row["cross_validation_config"] else None,
                    created_at=datetime.fromisoformat(row["created_at"])
                ))
        return analyses
    
    # ROC Normalized Model operations
    def create_roc_normalized_model(self, model: ROCNormalizedModel) -> ROCNormalizedModel:
        """Create a new ROC Normalized model."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO roc_normalized_models (analysis_id, biomarker_combination, normalizer_biomarker_version_id, auc, coefficients, cross_validation_results)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model.analysis_id, json.dumps(model.biomarker_combination), 
                  model.normalizer_biomarker_version_id, model.auc, json.dumps(model.coefficients), json.dumps(asdict(model.cross_validation_results)) if model.cross_validation_results else None))
            
            model.id = cursor.lastrowid
            
            row = conn.execute(
                "SELECT created_at FROM roc_normalized_models WHERE id = ?", 
                (model.id,)
            ).fetchone()
            if row:
                model.created_at = datetime.fromisoformat(row[0])
            
            conn.commit()
        return model
    
    def get_roc_normalized_models_by_analysis(self, analysis_id: int) -> List[ROCNormalizedModel]:
        """Get all ROC Normalized models for an analysis."""
        models = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM roc_normalized_models WHERE analysis_id = ? ORDER BY auc DESC",
                (analysis_id,)
            ).fetchall()
            
            for row in rows:
                models.append(ROCNormalizedModel(
                    id=row["id"],
                    analysis_id=row["analysis_id"],
                    biomarker_combination=json.loads(row["biomarker_combination"]),
                    normalizer_biomarker_version_id=row["normalizer_biomarker_version_id"],
                    auc=row["auc"],
                    coefficients=json.loads(row["coefficients"]),
                    cross_validation_results=json.loads(row["cross_validation_results"]) if row["cross_validation_results"] else None,
                    created_at=datetime.fromisoformat(row["created_at"])
                ))
        return models
    
    # ROC Normalized Metrics operations
    def create_roc_normalized_metrics(self, metrics: ROCNormalizedMetrics) -> ROCNormalizedMetrics:
        """Create ROC Normalized metrics."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO roc_normalized_metrics (model_id, threshold_type, threshold, sensitivity, specificity, npv, ppv)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (metrics.model_id, metrics.threshold_type, metrics.threshold,
                  metrics.sensitivity, metrics.specificity, metrics.npv, metrics.ppv))
            
            metrics.id = cursor.lastrowid
            
            row = conn.execute(
                "SELECT created_at FROM roc_normalized_metrics WHERE id = ?", 
                (metrics.id,)
            ).fetchone()
            if row:
                metrics.created_at = datetime.fromisoformat(row[0])
            
            conn.commit()
        return metrics
    
    def get_roc_normalized_metrics_by_model(self, model_id: int) -> List[ROCNormalizedMetrics]:
        """Get all metrics for a model."""
        metrics = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM roc_normalized_metrics WHERE model_id = ?",
                (model_id,)
            ).fetchall()
            
            for row in rows:
                metrics.append(ROCNormalizedMetrics(
                    id=row["id"],
                    model_id=row["model_id"],
                    threshold_type=row["threshold_type"],
                    threshold=row["threshold"],
                    sensitivity=row["sensitivity"],
                    specificity=row["specificity"],
                    npv=row["npv"],
                    ppv=row["ppv"],
                    created_at=datetime.fromisoformat(row["created_at"])
                ))
        return metrics
    
    # ROC Normalized Curve Points operations
    def create_roc_normalized_curve_points(self, points: List[ROCNormalizedCurvePoint]) -> List[ROCNormalizedCurvePoint]:
        """Create multiple ROC Normalized curve points."""
        with self._get_connection() as conn:
            for point in points:
                cursor = conn.execute("""
                    INSERT INTO roc_normalized_curve_points (model_id, fpr, tpr, threshold)
                    VALUES (?, ?, ?, ?)
                """, (point.model_id, point.fpr, point.tpr, point.threshold))
                
                point.id = cursor.lastrowid
                
                row = conn.execute(
                    "SELECT created_at FROM roc_normalized_curve_points WHERE id = ?", 
                    (point.id,)
                ).fetchone()
                if row:
                    point.created_at = datetime.fromisoformat(row[0])
            
            conn.commit()
        return points
    
    def get_roc_normalized_curve_points_by_model(self, model_id: int) -> List[ROCNormalizedCurvePoint]:
        """Get all ROC Normalized curve points for a model."""
        points = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM roc_normalized_curve_points WHERE model_id = ? ORDER BY fpr",
                (model_id,)
            ).fetchall()
            
            for row in rows:
                points.append(ROCNormalizedCurvePoint(
                    id=row["id"],
                    model_id=row["model_id"],
                    fpr=row["fpr"],
                    tpr=row["tpr"],
                    threshold=row["threshold"],
                    created_at=datetime.fromisoformat(row["created_at"])
                ))
        return points

class ROCNormalizedAnalyzer(BaseAnalyzer):
    """Main class for performing ROC Normalized analysis on biomarker data."""
    
    def __init__(self, db_path: Optional[str] = None):
        super().__init__(db_path)
        self.db_path = db_path or get_database_path()
        self.roc_norm_db = ROCNormalizedAnalysisDatabase(self.db_path)
        self.exp_db = ExperimentDatabase(self.db_path)
        self.sample_db = SampleDatabase(self.db_path)
    
    def run_roc_normalized_analysis(self, analysis: ROCNormalizedAnalysis) -> Dict[str, Any]:
        """
        Run complete ROC Normalized analysis for an experiment.
        
        Args:
            analysis: ROC Normalized analysis configuration
            
        Returns:
            Dictionary with analysis results and summary statistics
        """
        # Create the analysis record
        created_analysis = self.roc_norm_db.create_roc_normalized_analysis(analysis)
        
        # Get experiment data
        experiment_data = self._prepare_experiment_data(analysis.experiment_id, analysis.normalizer_biomarker_version_id)
        if experiment_data is None:
            raise ValueError(f"No valid data found for experiment {analysis.experiment_id}")
        
        # Generate all biomarker combinations (excluding normalizer)
        available_biomarkers = [bv for bv in experiment_data['biomarker_versions'] 
                              if bv != analysis.normalizer_biomarker_version_id]
        
        if not available_biomarkers:
            raise ValueError("No biomarkers available for analysis after excluding normalizer")
        
        biomarker_combinations = self._generate_biomarker_combinations(
            available_biomarkers, 
            analysis.max_combination_size
        )
        
        results = {
            'analysis_id': created_analysis.id,
            'normalizer_biomarker': analysis.normalizer_biomarker_version_id,
            'total_combinations': len(biomarker_combinations),
            'models_created': 0,
            'successful_models': [],
            'failed_models': []
        }
        
        # Process each combination
        for combination in biomarker_combinations:
            try:
                model_result = self._analyze_biomarker_combination(
                    created_analysis.id,
                    combination,
                    analysis.normalizer_biomarker_version_id,
                    experiment_data,
                    analysis.prevalence,
                    analysis.cross_validation_config
                )
                results['successful_models'].append(model_result)
                results['models_created'] += 1
                
            except Exception as e:
                results['failed_models'].append({
                    'combination': combination,
                    'error': str(e)
                })
        
        return results
    
    def _prepare_experiment_data(self, experiment_id: int, normalizer_bv_id: int) -> Optional[Dict[str, Any]]:
        """Prepare experiment data for analysis with normalization."""
        # Get experiment measurements
        measurements = self.exp_db.get_measurements_by_experiment(experiment_id)
        if not measurements:
            return None
        
        # Get sample data with diagnosis information
        sample_ids = list(set(m.sample_id for m in measurements))
        samples_data = {}
        
        for sample_id in sample_ids:
            sample = self.sample_db.get_sample_by_id(sample_id)
            if sample:
                samples_data[sample_id] = {
                    'code': sample.code,
                    'dx': sample.dx  # 0 = benign, 1 = disease
                }
        
        # Organize data by biomarker version
        biomarker_data = {}
        biomarker_versions = set()
        
        for measurement in measurements:
            if measurement.sample_id not in samples_data:
                continue  # Skip if no sample data
                
            bv_id = measurement.biomarker_version_id
            biomarker_versions.add(bv_id)
            
            if bv_id not in biomarker_data:
                biomarker_data[bv_id] = {}
            
            biomarker_data[bv_id][measurement.sample_id] = measurement.value
        
        # Check if normalizer biomarker is available
        if normalizer_bv_id not in biomarker_versions:
            raise ValueError(f"Normalizer biomarker version {normalizer_bv_id} not found in experiment")
        
        # Create DataFrame for analysis with normalization
        df_data = []
        for sample_id, sample_info in samples_data.items():
            # Check if normalizer value exists and is not zero
            if (normalizer_bv_id not in biomarker_data or 
                sample_id not in biomarker_data[normalizer_bv_id] or
                biomarker_data[normalizer_bv_id][sample_id] == 0):
                continue  # Skip samples without normalizer or with zero normalizer
            
            normalizer_value = biomarker_data[normalizer_bv_id][sample_id]
            
            row = {'sample_id': sample_id, 'dx': sample_info['dx']}
            
            # Add normalized biomarker values
            skip_sample = False
            for bv_id in biomarker_versions:
                if bv_id == normalizer_bv_id:
                    continue  # Skip the normalizer itself
                    
                if bv_id in biomarker_data and sample_id in biomarker_data[bv_id]:
                    # Normalize by dividing by normalizer
                    normalized_value = biomarker_data[bv_id][sample_id] / normalizer_value
                    row[f'biomarker_{bv_id}'] = normalized_value
                else:
                    skip_sample = True
                    break
            
            if not skip_sample:
                df_data.append(row)
        
        if not df_data:
            return None
        
        df = pd.DataFrame(df_data)
        available_biomarkers = [bv for bv in biomarker_versions if bv != normalizer_bv_id]
        
        return {
            'dataframe': df,
            'biomarker_versions': available_biomarkers,
            'normalizer_biomarker_version_id': normalizer_bv_id,
            'sample_count': len(df),
            'positive_cases': len(df[df['dx'] == 1]),
            'negative_cases': len(df[df['dx'] == 0])
        }
    
    def _generate_biomarker_combinations(self, biomarker_versions: List[int], 
                                       max_size: int) -> List[List[int]]:
        """Generate all possible biomarker combinations up to max_size."""
        all_combinations = []
        
        for size in range(1, min(max_size + 1, len(biomarker_versions) + 1)):
            for combo in combinations(biomarker_versions, size):
                all_combinations.append(list(combo))
        
        return all_combinations
    
    def _analyze_biomarker_combination(self, analysis_id: int, 
                                     biomarker_combination: List[int],
                                     normalizer_bv_id: int,
                                     experiment_data: Dict[str, Any],
                                     prevalence: float,
                                     cv_config: Optional[CrossValidationConfig] = None) -> Dict[str, Any]:
        """Analyze a specific normalized biomarker combination."""
        df = experiment_data['dataframe'].copy()
        
        # Prepare features and target
        feature_cols = [f'biomarker_{bv_id}' for bv_id in biomarker_combination]
        X = df[feature_cols].values
        y = df['dx'].values
        
        # Perform cross-validation if configured
        cv_results = None
        if cv_config:
            cv_results = self._perform_cross_validation(X, y, cv_config)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit logistic regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Store model coefficients (including intercept and feature scaling info)
        coefficients = {
            'intercept': float(model.intercept_[0]),
            'coef': [float(c) for c in model.coef_[0]],
            'biomarker_version_ids': biomarker_combination,
            'normalizer_biomarker_version_id': normalizer_bv_id,
            'scaler_mean': [float(m) for m in scaler.mean_],
            'scaler_scale': [float(s) for s in scaler.scale_]
        }
        
        # Create ROC Normalized model
        roc_model = ROCNormalizedModel(
            analysis_id=analysis_id,
            biomarker_combination=biomarker_combination,
            normalizer_biomarker_version_id=normalizer_bv_id,
            auc=roc_auc,
            coefficients=coefficients,
            cross_validation_results=cv_results
        )
        created_model = self.roc_norm_db.create_roc_normalized_model(roc_model)
        
        # Calculate metrics for different thresholds
        metrics_results = []
        
        # Calculate metrics for se_97 (97% sensitivity)
        se_97_metrics = self._calculate_threshold_metrics(
            y, y_pred_proba, fpr, tpr, thresholds, 
            target_sensitivity=0.97, prevalence=prevalence
        )
        if se_97_metrics:
            metrics = ROCNormalizedMetrics(
                model_id=created_model.id,
                threshold_type='se_97',
                **se_97_metrics
            )
            self.roc_norm_db.create_roc_normalized_metrics(metrics)
            metrics_results.append(('se_97', se_97_metrics))
        
        # Calculate metrics for se_95 (95% sensitivity)
        se_95_metrics = self._calculate_threshold_metrics(
            y, y_pred_proba, fpr, tpr, thresholds,
            target_sensitivity=0.95, prevalence=prevalence
        )
        if se_95_metrics:
            metrics = ROCNormalizedMetrics(
                model_id=created_model.id,
                threshold_type='se_95',
                **se_95_metrics
            )
            self.roc_norm_db.create_roc_normalized_metrics(metrics)
            metrics_results.append(('se_95', se_95_metrics))
        
        # Calculate metrics for max(sensitivity + specificity)
        max_sum_metrics = self._calculate_max_sum_metrics(
            y, y_pred_proba, fpr, tpr, thresholds, prevalence
        )
        if max_sum_metrics:
            metrics = ROCNormalizedMetrics(
                model_id=created_model.id,
                threshold_type='max_sum',
                **max_sum_metrics
            )
            self.roc_norm_db.create_roc_normalized_metrics(metrics)
            metrics_results.append(('max_sum', max_sum_metrics))
        
        # Store ROC curve points
        roc_points = []
        for i in range(len(fpr)):
            point = ROCNormalizedCurvePoint(
                model_id=created_model.id,
                fpr=float(fpr[i]),
                tpr=float(tpr[i]),
                threshold=float(thresholds[i]) if i < len(thresholds) else float('inf')
            )
            roc_points.append(point)
        
        self.roc_norm_db.create_roc_normalized_curve_points(roc_points)
        
        return {
            'model_id': created_model.id,
            'biomarker_combination': biomarker_combination,
            'normalizer_biomarker_version_id': normalizer_bv_id,
            'auc': roc_auc,
            'cross_validation_results': cv_results,
            'metrics': metrics_results,
            'roc_points_count': len(roc_points)
        }
    
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
    
    def generate_analysis_report(self, analysis_id: int) -> pd.DataFrame:
        """Generate a comprehensive report for ROC Normalized analysis."""
        # Get analysis info
        analysis = self.roc_norm_db.get_roc_normalized_analysis_by_id(analysis_id)
        if not analysis:
            raise ValueError(f"Analysis {analysis_id} not found")
        
        # Get all models for this analysis
        models = self.roc_norm_db.get_roc_normalized_models_by_analysis(analysis_id)
        
        report_data = []
        
        # Get normalizer biomarker name
        normalizer_bv = self.exp_db.get_biomarker_version_by_id(analysis.normalizer_biomarker_version_id)
        normalizer_name = "Unknown"
        if normalizer_bv:
            normalizer_biomarker = self.exp_db.get_biomarker_by_id(normalizer_bv.biomarker_id)
            if normalizer_biomarker:
                normalizer_name = f"{normalizer_biomarker.name}_{normalizer_bv.version}"
        
        for model in models:
            # Get biomarker names for this model
            biomarker_names = []
            for bv_id in model.biomarker_combination:
                bv = self.exp_db.get_biomarker_version_by_id(bv_id)
                if bv:
                    biomarker = self.exp_db.get_biomarker_by_id(bv.biomarker_id)
                    if biomarker:
                        biomarker_names.append(f"{biomarker.name}_{bv.version}")
            
            # Get metrics for this model
            metrics = self.roc_norm_db.get_roc_normalized_metrics_by_model(model.id)
            metrics_dict = {m.threshold_type: m for m in metrics}
            
            # Create row data
            row = {
                'Model_ID': model.id,
                'AUC': model.auc,
                'Normalizer': normalizer_name,
                'Prevalence': analysis.prevalence
            }
            
            # Add biomarker columns (as ratios)
            for i, name in enumerate(biomarker_names, 1):
                row[f'Biomarker_{i}'] = f"{name}/{normalizer_name}"
            
            # Add metrics for each threshold type
            for threshold_type in ['se_97', 'se_95', 'max_sum']:
                if threshold_type in metrics_dict:
                    m = metrics_dict[threshold_type]
                    prefix = threshold_type
                    row[f'{prefix}_Threshold'] = m.threshold
                    row[f'{prefix}_Sensitivity'] = m.sensitivity
                    row[f'{prefix}_Specificity'] = m.specificity
                    row[f'{prefix}_NPV'] = m.npv
                    row[f'{prefix}_PPV'] = m.ppv
                else:
                    # Fill with NaN if metrics not available
                    prefix = threshold_type
                    for suffix in ['Threshold', 'Sensitivity', 'Specificity', 'NPV', 'PPV']:
                        row[f'{prefix}_{suffix}'] = np.nan
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)