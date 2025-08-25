"""
ROC Ratios analysis module for MMK Knowledge Base.
Provides ROC curve analysis with biomarker ratios, where all possible ratios 
between biomarkers are computed and used as features for diagnostic modeling.
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
class ROCRatiosAnalysis:
    """ROC Ratios Analysis model representing an analysis run."""
    name: str
    description: str
    experiment_id: int
    prevalence: float  # Expected prevalence for PPV/NPV calculations
    max_combination_size: int  # Maximum number of ratios per model
    cross_validation_config: Optional[CrossValidationConfig] = None
    created_at: Optional[datetime] = None
    id: Optional[int] = None


@dataclass
class ROCRatiosModel:
    """ROC Ratios Model representing a specific ratio combination model."""
    analysis_id: int
    ratio_combination: List[Tuple[int, int]]  # List of (numerator_bv_id, denominator_bv_id) tuples
    auc: float
    coefficients: Dict[str, float]  # Model coefficients including intercept
    cross_validation_results: Optional[CrossValidationResults] = None
    created_at: Optional[datetime] = None
    id: Optional[int] = None


@dataclass
class ROCRatiosMetrics:
    """ROC Ratios Metrics for specific sensitivity thresholds."""
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
class ROCRatiosCurvePoint:
    """Individual point on ROC Ratios curve."""
    model_id: int
    fpr: float  # False Positive Rate
    tpr: float  # True Positive Rate
    threshold: float
    created_at: Optional[datetime] = None
    id: Optional[int] = None


class ROCRatiosAnalysisDatabase:
    """Database operations for ROC Ratios analysis."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_database_path()
        self.init_database()
    
    def init_database(self):
        """Initialize database with ROC Ratios analysis tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # ROC ratios analyses table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roc_ratios_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    experiment_id INTEGER NOT NULL,
                    prevalence REAL NOT NULL CHECK (prevalence > 0 AND prevalence < 1),
                    max_combination_size INTEGER NOT NULL CHECK (max_combination_size > 0),
                    cross_validation_config TEXT,  -- JSON config for cross-validation
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
                )
            """)
            
            # ROC ratios models table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roc_ratios_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER NOT NULL,
                    ratio_combination TEXT NOT NULL,  -- JSON array of [numerator_bv_id, denominator_bv_id] pairs
                    auc REAL NOT NULL,
                    coefficients TEXT NOT NULL,  -- JSON object with coefficients
                    cross_validation_results TEXT,  -- JSON object with CV results
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES roc_ratios_analyses (id) ON DELETE CASCADE
                )
            """)
            
            # ROC ratios metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roc_ratios_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    threshold_type TEXT NOT NULL CHECK (threshold_type IN ('se_97', 'se_95', 'max_sum')),
                    threshold REAL NOT NULL,
                    sensitivity REAL NOT NULL,
                    specificity REAL NOT NULL,
                    npv REAL NOT NULL,
                    ppv REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES roc_ratios_models (id) ON DELETE CASCADE
                )
            """)
            
            # ROC ratios curve points table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roc_ratios_curve_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    fpr REAL NOT NULL,
                    tpr REAL NOT NULL,
                    threshold REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES roc_ratios_models (id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_roc_ratios_analyses_experiment_id ON roc_ratios_analyses (experiment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_roc_ratios_models_analysis_id ON roc_ratios_models (analysis_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_roc_ratios_metrics_model_id ON roc_ratios_metrics (model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_roc_ratios_curve_points_model_id ON roc_ratios_curve_points (model_id)")
            
            conn.commit()
    
    def _get_connection(self):
        """Get database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def create_roc_ratios_analysis(self, analysis: ROCRatiosAnalysis) -> ROCRatiosAnalysis:
        """Create a new ROC Ratios analysis."""
        with self._get_connection() as conn:
            cv_config_json = None
            if analysis.cross_validation_config:
                cv_config_json = json.dumps(asdict(analysis.cross_validation_config))
            
            cursor = conn.execute("""
                INSERT INTO roc_ratios_analyses (name, description, experiment_id, prevalence, max_combination_size, cross_validation_config)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (analysis.name, analysis.description, analysis.experiment_id, 
                  analysis.prevalence, analysis.max_combination_size, cv_config_json))
            
            analysis.id = cursor.lastrowid
            
            row = conn.execute(
                "SELECT created_at FROM roc_ratios_analyses WHERE id = ?", 
                (analysis.id,)
            ).fetchone()
            if row:
                analysis.created_at = datetime.fromisoformat(row[0])
            
            conn.commit()
        return analysis
    
    def get_roc_ratios_analysis_by_id(self, analysis_id: int) -> Optional[ROCRatiosAnalysis]:
        """Get ROC Ratios analysis by ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM roc_ratios_analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
            
            if row:
                cv_config = None
                if row["cross_validation_config"]:
                    cv_config = CrossValidationConfig(**json.loads(row["cross_validation_config"]))
                
                return ROCRatiosAnalysis(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    experiment_id=row["experiment_id"],
                    prevalence=row["prevalence"],
                    max_combination_size=row["max_combination_size"],
                    cross_validation_config=cv_config,
                    created_at=datetime.fromisoformat(row["created_at"])
                )
        return None
    
    def list_roc_ratios_analyses(self, experiment_id: Optional[int] = None) -> List[ROCRatiosAnalysis]:
        """List ROC Ratios analyses, optionally filtered by experiment."""
        analyses = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            
            if experiment_id:
                query = "SELECT * FROM roc_ratios_analyses WHERE experiment_id = ? ORDER BY created_at DESC"
                rows = conn.execute(query, (experiment_id,)).fetchall()
            else:
                query = "SELECT * FROM roc_ratios_analyses ORDER BY created_at DESC"
                rows = conn.execute(query).fetchall()
            
            for row in rows:
                cv_config = None
                if row["cross_validation_config"]:
                    cv_config = CrossValidationConfig(**json.loads(row["cross_validation_config"]))
                
                analyses.append(ROCRatiosAnalysis(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    experiment_id=row["experiment_id"],
                    prevalence=row["prevalence"],
                    max_combination_size=row["max_combination_size"],
                    cross_validation_config=cv_config,
                    created_at=datetime.fromisoformat(row["created_at"])
                ))
        return analyses
    
    def create_roc_ratios_model(self, model: ROCRatiosModel) -> ROCRatiosModel:
        """Create a new ROC Ratios model."""
        with self._get_connection() as conn:
            cv_results_json = None
            if model.cross_validation_results:
                cv_results_json = json.dumps(asdict(model.cross_validation_results))
            
            cursor = conn.execute("""
                INSERT INTO roc_ratios_models (analysis_id, ratio_combination, auc, coefficients, cross_validation_results)
                VALUES (?, ?, ?, ?, ?)
            """, (model.analysis_id, json.dumps(model.ratio_combination), 
                  model.auc, json.dumps(model.coefficients), cv_results_json))
            
            model.id = cursor.lastrowid
            
            row = conn.execute(
                "SELECT created_at FROM roc_ratios_models WHERE id = ?", 
                (model.id,)
            ).fetchone()
            if row:
                model.created_at = datetime.fromisoformat(row[0])
            
            conn.commit()
        return model
    
    def get_roc_ratios_models_by_analysis(self, analysis_id: int) -> List[ROCRatiosModel]:
        """Get all ROC Ratios models for an analysis."""
        models = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM roc_ratios_models WHERE analysis_id = ? ORDER BY auc DESC",
                (analysis_id,)
            ).fetchall()
            
            for row in rows:
                cv_results = None
                if row["cross_validation_results"]:
                    cv_data = json.loads(row["cross_validation_results"])
                    cv_results = CrossValidationResults(**cv_data)
                
                models.append(ROCRatiosModel(
                    id=row["id"],
                    analysis_id=row["analysis_id"],
                    ratio_combination=[tuple(pair) for pair in json.loads(row["ratio_combination"])],
                    auc=row["auc"],
                    coefficients=json.loads(row["coefficients"]),
                    cross_validation_results=cv_results,
                    created_at=datetime.fromisoformat(row["created_at"])
                ))
        return models
    
    def create_roc_ratios_metrics(self, metrics: ROCRatiosMetrics) -> ROCRatiosMetrics:
        """Create ROC Ratios metrics."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO roc_ratios_metrics (model_id, threshold_type, threshold, sensitivity, specificity, npv, ppv)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (metrics.model_id, metrics.threshold_type, metrics.threshold,
                  metrics.sensitivity, metrics.specificity, metrics.npv, metrics.ppv))
            
            metrics.id = cursor.lastrowid
            
            row = conn.execute(
                "SELECT created_at FROM roc_ratios_metrics WHERE id = ?", 
                (metrics.id,)
            ).fetchone()
            if row:
                metrics.created_at = datetime.fromisoformat(row[0])
            
            conn.commit()
        return metrics
    
    def get_roc_ratios_metrics_by_model(self, model_id: int) -> List[ROCRatiosMetrics]:
        """Get all metrics for a model."""
        metrics = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM roc_ratios_metrics WHERE model_id = ?",
                (model_id,)
            ).fetchall()
            
            for row in rows:
                metrics.append(ROCRatiosMetrics(
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
    
    def create_roc_ratios_curve_points(self, points: List[ROCRatiosCurvePoint]) -> List[ROCRatiosCurvePoint]:
        """Create multiple ROC Ratios curve points."""
        with self._get_connection() as conn:
            for point in points:
                cursor = conn.execute("""
                    INSERT INTO roc_ratios_curve_points (model_id, fpr, tpr, threshold)
                    VALUES (?, ?, ?, ?)
                """, (point.model_id, point.fpr, point.tpr, point.threshold))
                
                point.id = cursor.lastrowid
                
                row = conn.execute(
                    "SELECT created_at FROM roc_ratios_curve_points WHERE id = ?", 
                    (point.id,)
                ).fetchone()
                if row:
                    point.created_at = datetime.fromisoformat(row[0])
            
            conn.commit()
        return points
    
    def get_roc_ratios_curve_points_by_model(self, model_id: int) -> List[ROCRatiosCurvePoint]:
        """Get all ROC Ratios curve points for a model."""
        points = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM roc_ratios_curve_points WHERE model_id = ? ORDER BY fpr",
                (model_id,)
            ).fetchall()
            
            for row in rows:
                points.append(ROCRatiosCurvePoint(
                    id=row["id"],
                    model_id=row["model_id"],
                    fpr=row["fpr"],
                    tpr=row["tpr"],
                    threshold=row["threshold"],
                    created_at=datetime.fromisoformat(row["created_at"])
                ))
        return points


class ROCRatiosAnalyzer(BaseAnalyzer):
    """Main class for performing ROC Ratios analysis on biomarker data."""
    
    def __init__(self, db_path: Optional[str] = None):
        super().__init__(db_path)
        self.db_path = db_path or get_database_path()
        self.roc_ratios_db = ROCRatiosAnalysisDatabase(self.db_path)
        self.exp_db = ExperimentDatabase(self.db_path)
        self.sample_db = SampleDatabase(self.db_path)
    
    def run_roc_ratios_analysis(self, analysis: ROCRatiosAnalysis) -> Dict[str, Any]:
        """
        Run complete ROC Ratios analysis for an experiment.
        
        Args:
            analysis: ROC Ratios analysis configuration
            
        Returns:
            Dictionary with analysis results and summary statistics
        """
        # Get experiment data first to validate experiment exists and has data
        experiment_data = self._prepare_experiment_data(analysis.experiment_id)
        if experiment_data is None:
            raise ValueError(f"No valid data found for experiment {analysis.experiment_id}")
        
        # Create the analysis record after validation
        created_analysis = self.roc_ratios_db.create_roc_ratios_analysis(analysis)
        
        # Generate all possible ratios between biomarkers
        available_biomarkers = experiment_data['biomarker_versions']
        
        if len(available_biomarkers) < 2:
            raise ValueError("At least 2 biomarkers are required for ratio analysis")
        
        # Generate all possible ratio combinations
        ratio_combinations = self._generate_ratio_combinations(
            available_biomarkers, 
            analysis.max_combination_size
        )
        
        results = {
            'analysis_id': created_analysis.id,
            'available_biomarkers': available_biomarkers,
            'total_combinations': len(ratio_combinations),
            'models_created': 0,
            'successful_models': [],
            'failed_models': []
        }
        
        # Process each combination
        for combination in ratio_combinations:
            try:
                model_result = self._analyze_ratio_combination(
                    created_analysis.id,
                    combination,
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
    
    def _prepare_experiment_data(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Prepare experiment data for ratio analysis."""
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
        
        # Create DataFrame for analysis
        df_data = []
        for sample_id, sample_info in samples_data.items():
            row = {'sample_id': sample_id, 'dx': sample_info['dx']}
            
            # Add biomarker values
            skip_sample = False
            for bv_id in biomarker_versions:
                if bv_id in biomarker_data and sample_id in biomarker_data[bv_id]:
                    row[f'biomarker_{bv_id}'] = biomarker_data[bv_id][sample_id]
                else:
                    skip_sample = True
                    break
            
            if not skip_sample:
                df_data.append(row)
        
        if not df_data:
            return None
        
        df = pd.DataFrame(df_data)
        
        return {
            'dataframe': df,
            'biomarker_versions': list(biomarker_versions),
            'sample_count': len(df),
            'positive_cases': len(df[df['dx'] == 1]),
            'negative_cases': len(df[df['dx'] == 0])
        }
    
    def _generate_ratio_combinations(self, biomarker_versions: List[int], 
                                   max_size: int) -> List[List[Tuple[int, int]]]:
        """Generate all possible ratio combinations up to max_size."""
        # First generate all possible ratios between biomarkers
        all_ratios = []
        for i in range(len(biomarker_versions)):
            for j in range(len(biomarker_versions)):
                if i != j:  # Don't create ratios of biomarker with itself
                    numerator = biomarker_versions[i]
                    denominator = biomarker_versions[j]
                    all_ratios.append((numerator, denominator))
        
        # Now generate combinations of these ratios
        all_combinations = []
        
        for size in range(1, min(max_size + 1, len(all_ratios) + 1)):
            for combo in combinations(all_ratios, size):
                all_combinations.append(list(combo))
        
        return all_combinations
    
    def _analyze_ratio_combination(self, analysis_id: int, 
                                 ratio_combination: List[Tuple[int, int]],
                                 experiment_data: Dict[str, Any],
                                 prevalence: float,
                                 cv_config: Optional[CrossValidationConfig] = None) -> Dict[str, Any]:
        """Analyze a specific ratio combination."""
        df = experiment_data['dataframe'].copy()
        
        # Create ratio features
        ratio_features = []
        feature_names = []
        
        for i, (numerator_bv_id, denominator_bv_id) in enumerate(ratio_combination):
            numerator_col = f'biomarker_{numerator_bv_id}'
            denominator_col = f'biomarker_{denominator_bv_id}'
            ratio_col = f'ratio_{i}'
            
            # Calculate ratio, handling division by zero
            denominator_values = df[denominator_col]
            # Replace zeros with a small value to avoid division by zero
            denominator_values = denominator_values.replace(0, 1e-10)
            
            ratio_values = df[numerator_col] / denominator_values
            ratio_features.append(ratio_values)
            feature_names.append(ratio_col)
        
        # Create feature matrix
        X = np.column_stack(ratio_features)
        y = df['dx'].values
        
        # Check for invalid values (inf, nan)
        if np.any(~np.isfinite(X)):
            raise ValueError("Invalid values (inf or nan) in ratio calculations")
        
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
            'ratio_combination': ratio_combination,
            'scaler_mean': [float(m) for m in scaler.mean_],
            'scaler_scale': [float(s) for s in scaler.scale_]
        }
        
        # Create ROC Ratios model
        roc_model = ROCRatiosModel(
            analysis_id=analysis_id,
            ratio_combination=ratio_combination,
            auc=roc_auc,
            coefficients=coefficients,
            cross_validation_results=cv_results
        )
        created_model = self.roc_ratios_db.create_roc_ratios_model(roc_model)
        
        # Calculate metrics for different thresholds
        metrics_results = []
        
        # Calculate metrics for se_97 (97% sensitivity)
        se_97_metrics = self._calculate_threshold_metrics(
            y, y_pred_proba, fpr, tpr, thresholds, 
            target_sensitivity=0.97, prevalence=prevalence
        )
        if se_97_metrics:
            metrics = ROCRatiosMetrics(
                model_id=created_model.id,
                threshold_type='se_97',
                **se_97_metrics
            )
            self.roc_ratios_db.create_roc_ratios_metrics(metrics)
            metrics_results.append(('se_97', se_97_metrics))
        
        # Calculate metrics for se_95 (95% sensitivity)
        se_95_metrics = self._calculate_threshold_metrics(
            y, y_pred_proba, fpr, tpr, thresholds,
            target_sensitivity=0.95, prevalence=prevalence
        )
        if se_95_metrics:
            metrics = ROCRatiosMetrics(
                model_id=created_model.id,
                threshold_type='se_95',
                **se_95_metrics
            )
            self.roc_ratios_db.create_roc_ratios_metrics(metrics)
            metrics_results.append(('se_95', se_95_metrics))
        
        # Calculate metrics for max(sensitivity + specificity)
        max_sum_metrics = self._calculate_max_sum_metrics(
            y, y_pred_proba, fpr, tpr, thresholds, prevalence
        )
        if max_sum_metrics:
            metrics = ROCRatiosMetrics(
                model_id=created_model.id,
                threshold_type='max_sum',
                **max_sum_metrics
            )
            self.roc_ratios_db.create_roc_ratios_metrics(metrics)
            metrics_results.append(('max_sum', max_sum_metrics))
        
        # Store ROC curve points
        roc_points = []
        for i in range(len(fpr)):
            point = ROCRatiosCurvePoint(
                model_id=created_model.id,
                fpr=float(fpr[i]),
                tpr=float(tpr[i]),
                threshold=float(thresholds[i]) if i < len(thresholds) else float('inf')
            )
            roc_points.append(point)
        
        self.roc_ratios_db.create_roc_ratios_curve_points(roc_points)
        
        return {
            'model_id': created_model.id,
            'ratio_combination': ratio_combination,
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
        ppv_denominator = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
        if ppv_denominator > 0:
            ppv = (sensitivity * prevalence) / ppv_denominator
        else:
            ppv = 0.0
            
        npv_denominator = (1 - sensitivity) * prevalence + specificity * (1 - prevalence)
        if npv_denominator > 0:
            npv = (specificity * (1 - prevalence)) / npv_denominator
        else:
            npv = 1.0
        
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
        
        # Calculate PPV and NPV with safe division
        ppv_denominator = chosen_sensitivity * prevalence + (1 - chosen_specificity) * (1 - prevalence)
        if ppv_denominator > 0:
            ppv = (chosen_sensitivity * prevalence) / ppv_denominator
        else:
            ppv = 0.0
            
        npv_denominator = (1 - chosen_sensitivity) * prevalence + chosen_specificity * (1 - prevalence)
        if npv_denominator > 0:
            npv = (chosen_specificity * (1 - prevalence)) / npv_denominator
        else:
            npv = 1.0
        
        return {
            'threshold': float(chosen_threshold),
            'sensitivity': float(chosen_sensitivity),
            'specificity': float(chosen_specificity),
            'npv': float(npv),
            'ppv': float(ppv)
        }
    
    def generate_analysis_report(self, analysis_id: int) -> pd.DataFrame:
        """Generate a comprehensive report for ROC Ratios analysis."""
        # Get analysis info
        analysis = self.roc_ratios_db.get_roc_ratios_analysis_by_id(analysis_id)
        if not analysis:
            raise ValueError(f"Analysis {analysis_id} not found")
        
        # Get all models for this analysis
        models = self.roc_ratios_db.get_roc_ratios_models_by_analysis(analysis_id)
        
        report_data = []
        
        for model in models:
            # Get biomarker names for this model's ratios
            ratio_names = []
            for numerator_bv_id, denominator_bv_id in model.ratio_combination:
                # Get numerator biomarker name
                num_bv = self.exp_db.get_biomarker_version_by_id(numerator_bv_id)
                num_name = "Unknown"
                if num_bv:
                    num_biomarker = self.exp_db.get_biomarker_by_id(num_bv.biomarker_id)
                    if num_biomarker:
                        num_name = f"{num_biomarker.name}_{num_bv.version}"
                
                # Get denominator biomarker name
                den_bv = self.exp_db.get_biomarker_version_by_id(denominator_bv_id)
                den_name = "Unknown"
                if den_bv:
                    den_biomarker = self.exp_db.get_biomarker_by_id(den_bv.biomarker_id)
                    if den_biomarker:
                        den_name = f"{den_biomarker.name}_{den_bv.version}"
                
                ratio_names.append(f"{num_name}/{den_name}")
            
            # Get metrics for this model
            metrics = self.roc_ratios_db.get_roc_ratios_metrics_by_model(model.id)
            metrics_dict = {m.threshold_type: m for m in metrics}
            
            # Create row data
            row = {
                'Model_ID': model.id,
                'AUC': model.auc,
                'Prevalence': analysis.prevalence
            }
            
            # Add ratio columns
            for i, ratio_name in enumerate(ratio_names, 1):
                row[f'Ratio_{i}'] = ratio_name
            
            # Add cross-validation results if available
            if model.cross_validation_results:
                cv = model.cross_validation_results
                if hasattr(cv, 'loo_auc_mean') and cv.loo_auc_mean is not None:
                    row['CV_LOO_AUC_Mean'] = cv.loo_auc_mean
                    row['CV_LOO_AUC_Std'] = cv.loo_auc_std
                if hasattr(cv, 'bootstrap_auc_mean') and cv.bootstrap_auc_mean is not None:
                    row['CV_Bootstrap_AUC_Mean'] = cv.bootstrap_auc_mean
                    row['CV_Bootstrap_AUC_Std'] = cv.bootstrap_auc_std
            
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
    
    def _generate_biomarker_combinations(self, biomarker_versions: List[int], 
                                       max_size: int) -> List[List[int]]:
        """Generate biomarker combinations for ratio analysis (not used in ratios analysis)."""
        # This method is required by BaseAnalyzer but not used in ratios analysis
        # since we generate ratio combinations instead
        return []