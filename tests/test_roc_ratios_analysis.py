"""
Tests for ROC Ratios analysis functionality.
"""
import pytest
import tempfile
import os
import sqlite3
import numpy as np
from datetime import datetime

from src.mmkkb.analyses.roc_ratios_analysis import (
    ROCRatiosAnalyzer, ROCRatiosAnalysis, ROCRatiosAnalysisDatabase,
    ROCRatiosModel
)
from src.mmkkb.analyses.base_analysis import CrossValidationConfig, ROCCurvePoint, ROCMetrics
from src.mmkkb.projects import ProjectDatabase, Project
from src.mmkkb.samples import SampleDatabase, Sample
from src.mmkkb.experiments import ExperimentDatabase, Experiment, Biomarker, BiomarkerVersion, Measurement


class TestROCRatiosAnalysisDatabase:
    """Test ROC Ratios analysis database operations."""
    
    @pytest.fixture
    def db_path(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)
    
    @pytest.fixture
    def roc_ratios_db(self, db_path):
        """Create ROC ratios analysis database instance."""
        # Initialize all required database schemas
        from src.mmkkb.projects import ProjectDatabase
        from src.mmkkb.samples import SampleDatabase  
        from src.mmkkb.experiments import ExperimentDatabase
        
        # Initialize all databases to create required tables
        project_db = ProjectDatabase(db_path)
        sample_db = SampleDatabase(db_path)
        exp_db = ExperimentDatabase(db_path)
        
        # Create a dummy experiment for testing
        from src.mmkkb.projects import Project
        from src.mmkkb.experiments import Biomarker, BiomarkerVersion, Experiment
        
        # Create project
        project = Project(
            code="TEST",
            name="Test Project", 
            description="Test",
            creator="tester"
        )
        created_project = project_db.create_project(project)
        
        # Create biomarker and version
        biomarker = Biomarker(name="TestBio", description="Test")
        created_bio = exp_db.create_biomarker(biomarker)
        
        bio_version = BiomarkerVersion(
            biomarker_id=created_bio.id,
            version="1",
            description="Test"
        )
        created_version = exp_db.create_biomarker_version(bio_version)
        
        # Create experiment
        experiment = Experiment(
            name="Test Exp",
            description="Test",
            project_id=created_project.id
        )
        created_exp = exp_db.create_experiment(experiment)
        
        # Now create the ROC ratios database
        roc_ratios_db = ROCRatiosAnalysisDatabase(db_path)
        
        # Store experiment ID for tests
        roc_ratios_db._test_experiment_id = created_exp.id
        return roc_ratios_db
    
    def test_init_database(self, roc_ratios_db):
        """Test database initialization."""
        # Database should be initialized without errors
        assert roc_ratios_db.db_path is not None
        
        # Check tables exist
        import sqlite3
        conn = sqlite3.connect(roc_ratios_db.db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = [
            'roc_ratios_analyses', 'roc_ratios_models', 
            'roc_ratios_metrics', 'roc_ratios_curve_points'
        ]
        for table in expected_tables:
            assert table in tables
    
    def test_create_and_get_analysis(self, roc_ratios_db):
        """Test creating and retrieving ROC ratios analysis."""
        cv_config = CrossValidationConfig(
            enable_loo=True,
            enable_bootstrap=True,
            bootstrap_iterations=100
        )
        
        analysis = ROCRatiosAnalysis(
            name="Test Ratios Analysis",
            description="Test description",
            experiment_id=roc_ratios_db._test_experiment_id,
            prevalence=0.3,
            max_combination_size=2,
            cross_validation_config=cv_config
        )
        
        # Create analysis
        created = roc_ratios_db.create_roc_ratios_analysis(analysis)
        assert created.id is not None
        assert created.created_at is not None
        
        # Retrieve analysis
        retrieved = roc_ratios_db.get_roc_ratios_analysis_by_id(created.id)
        assert retrieved is not None
        assert retrieved.name == "Test Ratios Analysis"
        assert retrieved.experiment_id == roc_ratios_db._test_experiment_id
        assert retrieved.prevalence == 0.3
        assert retrieved.max_combination_size == 2
        assert retrieved.cross_validation_config is not None
        assert retrieved.cross_validation_config.enable_loo == True
    
    def test_list_analyses(self, roc_ratios_db):
        """Test listing ROC ratios analyses."""
        # Create multiple analyses
        for i in range(3):
            analysis = ROCRatiosAnalysis(
                name=f"Analysis {i}",
                description=f"Description {i}",
                experiment_id=roc_ratios_db._test_experiment_id,
                prevalence=0.2 + i * 0.1,
                max_combination_size=2
            )
            roc_ratios_db.create_roc_ratios_analysis(analysis)
        
        # List all analyses
        all_analyses = roc_ratios_db.list_roc_ratios_analyses()
        assert len(all_analyses) == 3
        
        # List by experiment
        exp_analyses = roc_ratios_db.list_roc_ratios_analyses(experiment_id=roc_ratios_db._test_experiment_id)
        assert len(exp_analyses) == 3
        # Check that we have all the analyses (order might vary)
        analysis_names = {analysis.name for analysis in exp_analyses}
        assert analysis_names == {"Analysis 0", "Analysis 1", "Analysis 2"}
    
    def test_create_and_get_models(self, roc_ratios_db):
        """Test creating and retrieving ROC ratios models."""
        # First create an analysis
        analysis = ROCRatiosAnalysis(
            name="Test Analysis",
            description="Test",
            experiment_id=roc_ratios_db._test_experiment_id,
            prevalence=0.3,
            max_combination_size=2
        )
        created_analysis = roc_ratios_db.create_roc_ratios_analysis(analysis)
        
        # Create model
        model = ROCRatiosModel(
            analysis_id=created_analysis.id,
            ratio_combination=[(1, 2), (3, 4)],
            auc=0.85,
            coefficients={
                'intercept': 0.5,
                'coef': [0.3, 0.7],
                'ratio_combination': [(1, 2), (3, 4)]
            }
        )
        
        created_model = roc_ratios_db.create_roc_ratios_model(model)
        assert created_model.id is not None
        assert created_model.created_at is not None
        
        # Retrieve models by analysis
        models = roc_ratios_db.get_roc_ratios_models_by_analysis(created_analysis.id)
        assert len(models) == 1
        assert models[0].ratio_combination == [(1, 2), (3, 4)]
        assert models[0].auc == 0.85
    
    def test_create_metrics(self, roc_ratios_db):
        """Test creating ROC ratios metrics."""
        # Create analysis and model first
        analysis = ROCRatiosAnalysis(
            name="Test Analysis",
            description="Test",
            experiment_id=roc_ratios_db._test_experiment_id,
            prevalence=0.3,
            max_combination_size=2
        )
        created_analysis = roc_ratios_db.create_roc_ratios_analysis(analysis)
        
        model = ROCRatiosModel(
            analysis_id=created_analysis.id,
            ratio_combination=[(1, 2)],
            auc=0.8,
            coefficients={'intercept': 0.1}
        )
        created_model = roc_ratios_db.create_roc_ratios_model(model)
        
        # Create metrics
        metrics = ROCMetrics(
            model_id=created_model.id,
            threshold_type='se_97',
            threshold=0.5,
            sensitivity=0.97,
            specificity=0.7,
            npv=0.95,
            ppv=0.75
        )
        
        created_metrics = roc_ratios_db.create_roc_ratios_metrics(metrics)
        assert created_metrics.id is not None
        
        # Retrieve metrics
        model_metrics = roc_ratios_db.get_roc_ratios_metrics_by_model(created_model.id)
        assert len(model_metrics) == 1
        assert model_metrics[0].threshold_type == 'se_97'
        assert model_metrics[0].sensitivity == 0.97
    
    def test_create_curve_points(self, roc_ratios_db):
        """Test creating ROC curve points."""
        # Create analysis and model first
        analysis = ROCRatiosAnalysis(
            name="Test Analysis",
            description="Test",
            experiment_id=roc_ratios_db._test_experiment_id,
            prevalence=0.3,
            max_combination_size=2
        )
        created_analysis = roc_ratios_db.create_roc_ratios_analysis(analysis)
        
        model = ROCRatiosModel(
            analysis_id=created_analysis.id,
            ratio_combination=[(1, 2)],
            auc=0.8,
            coefficients={'intercept': 0.1}
        )
        created_model = roc_ratios_db.create_roc_ratios_model(model)
        
        # Create curve points
        points = [
            ROCCurvePoint(
                model_id=created_model.id,
                fpr=0.0,
                tpr=0.0,
                threshold=1.0
            ),
            ROCCurvePoint(
                model_id=created_model.id,
                fpr=0.2,
                tpr=0.8,
                threshold=0.5
            )
        ]
        
        created_points = roc_ratios_db.create_roc_ratios_curve_points(points)
        assert len(created_points) == 2
        assert all(p.id is not None for p in created_points)
        
        # Retrieve curve points
        model_points = roc_ratios_db.get_roc_ratios_curve_points_by_model(created_model.id)
        assert len(model_points) == 2
        assert model_points[0].fpr == 0.0
        assert model_points[1].fpr == 0.2


class TestROCRatiosAnalyzer:
    """Test ROC Ratios analyzer functionality."""
    
    @pytest.fixture
    def db_path(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)
    
    @pytest.fixture
    def analyzer(self, db_path):
        """Create analyzer with test data."""
        analyzer = ROCRatiosAnalyzer(db_path)
        
        # Set up test data
        self._setup_test_data(analyzer, db_path)
        
        return analyzer
    
    def _setup_test_data(self, analyzer, db_path):
        """Set up test data for analysis."""
        # Create project
        project_db = ProjectDatabase(db_path)
        project = Project(
            code="TEST001",
            name="Test Project",
            description="Test project for ratios analysis",
            creator="tester"
        )
        created_project = project_db.create_project(project)
        
        # Create samples
        sample_db = SampleDatabase(db_path)
        samples_data = [
            # Positive cases (dx=1)
            {"code": "S001", "dx": 1},
            {"code": "S002", "dx": 1},
            {"code": "S003", "dx": 1},
            {"code": "S004", "dx": 1},
            {"code": "S005", "dx": 1},
            # Negative cases (dx=0)
            {"code": "S006", "dx": 0},
            {"code": "S007", "dx": 0},
            {"code": "S008", "dx": 0},
            {"code": "S009", "dx": 0},
            {"code": "S010", "dx": 0},
        ]
        
        sample_ids = []
        for sample_data in samples_data:
            sample = Sample(
                code=sample_data["code"],
                age=50,  # Default age
                bmi=25.0,  # Default BMI
                dx=sample_data["dx"],
                dx_origin="Test",
                collection_center="Test Center",
                processing_time=24,
                project_id=created_project.id
            )
            created_sample = sample_db.create_sample(sample)
            sample_ids.append(created_sample.id)
        
        # Create experiment with biomarkers
        exp_db = ExperimentDatabase(db_path)
        
        # Create biomarkers
        biomarkers = [
            Biomarker(name="Biomarker_A", description="Test biomarker A"),
            Biomarker(name="Biomarker_B", description="Test biomarker B"),
            Biomarker(name="Biomarker_C", description="Test biomarker C")
        ]
        
        biomarker_ids = []
        for biomarker in biomarkers:
            created = exp_db.create_biomarker(biomarker)
            biomarker_ids.append(created.id)
        
        # Create biomarker versions
        biomarker_version_ids = []
        for i, biomarker_id in enumerate(biomarker_ids):
            version = BiomarkerVersion(
                biomarker_id=biomarker_id,
                version="1",
                description=f"Test version for biomarker {i+1}"
            )
            created = exp_db.create_biomarker_version(version)
            biomarker_version_ids.append(created.id)
        
        # Create experiment
        experiment = Experiment(
            name="Test Ratios Experiment",
            description="Test experiment for ratios analysis",
            project_id=created_project.id
        )
        created_experiment = exp_db.create_experiment(experiment)
        
        # Create measurements with synthetic data that should show some signal
        np.random.seed(42)  # For reproducible results
        
        for i, sample_id in enumerate(sample_ids):
            is_positive = samples_data[i]["dx"] == 1
            
            for j, bv_id in enumerate(biomarker_version_ids):
                # Create synthetic values with some separation between classes
                if is_positive:
                    # Positive cases: higher values for biomarker A, lower for B, variable for C
                    if j == 0:  # Biomarker A
                        value = np.random.normal(10, 2)
                    elif j == 1:  # Biomarker B  
                        value = np.random.normal(5, 1)
                    else:  # Biomarker C
                        value = np.random.normal(8, 1.5)
                else:
                    # Negative cases: lower values for biomarker A, higher for B, variable for C
                    if j == 0:  # Biomarker A
                        value = np.random.normal(6, 2)
                    elif j == 1:  # Biomarker B
                        value = np.random.normal(9, 1)
                    else:  # Biomarker C
                        value = np.random.normal(7, 1.5)
                
                # Ensure positive values
                value = max(value, 0.1)
                
                measurement = Measurement(
                    experiment_id=created_experiment.id,
                    sample_id=sample_id,
                    biomarker_version_id=bv_id,
                    value=value
                )
                exp_db.create_measurement(measurement)
        
        # Store experiment ID for tests
        analyzer._test_experiment_id = created_experiment.id
    
    def test_generate_ratio_combinations(self, analyzer):
        """Test ratio combination generation."""
        biomarker_versions = [1, 2, 3]
        
        # Test max_size = 1
        combinations = analyzer._generate_ratio_combinations(biomarker_versions, 1)
        # Should be 3*2 = 6 individual ratios (A/B, A/C, B/A, B/C, C/A, C/B)
        assert len(combinations) == 6
        
        # Test max_size = 2
        combinations = analyzer._generate_ratio_combinations(biomarker_versions, 2)
        # Should be 6 individual ratios + combinations of 2 ratios
        # C(6,2) = 15, so total = 6 + 15 = 21
        assert len(combinations) == 21
        
        # Check that all combinations are valid
        for combo in combinations:
            assert isinstance(combo, list)
            for ratio in combo:
                assert isinstance(ratio, tuple)
                assert len(ratio) == 2
                assert ratio[0] != ratio[1]  # No self-ratios
    
    def test_prepare_experiment_data(self, analyzer):
        """Test experiment data preparation."""
        experiment_data = analyzer._prepare_experiment_data(analyzer._test_experiment_id)
        
        assert experiment_data is not None
        assert 'dataframe' in experiment_data
        assert 'biomarker_versions' in experiment_data
        assert 'sample_count' in experiment_data
        assert 'positive_cases' in experiment_data
        assert 'negative_cases' in experiment_data
        
        df = experiment_data['dataframe']
        assert len(df) == 10  # 10 samples
        assert len(experiment_data['biomarker_versions']) == 3  # 3 biomarkers
        assert experiment_data['positive_cases'] == 5
        assert experiment_data['negative_cases'] == 5
        
        # Check that all required columns are present
        expected_cols = ['sample_id', 'dx'] + [f'biomarker_{bv_id}' for bv_id in experiment_data['biomarker_versions']]
        for col in expected_cols:
            assert col in df.columns
    
    def test_run_roc_ratios_analysis(self, analyzer):
        """Test running complete ROC ratios analysis."""
        analysis = ROCRatiosAnalysis(
            name="Test Ratios Analysis",
            description="Test analysis",
            experiment_id=analyzer._test_experiment_id,
            prevalence=0.3,
            max_combination_size=1  # Keep it simple for testing
        )
        
        results = analyzer.run_roc_ratios_analysis(analysis)
        
        assert 'analysis_id' in results
        assert 'total_combinations' in results
        assert 'models_created' in results
        assert 'successful_models' in results
        assert 'failed_models' in results
        
        assert results['analysis_id'] is not None
        assert results['total_combinations'] > 0
        assert results['models_created'] >= 0
        
        # Check that some models were created successfully
        if results['models_created'] > 0:
            assert len(results['successful_models']) > 0
            
            # Check first successful model
            first_model = results['successful_models'][0]
            assert 'model_id' in first_model
            assert 'ratio_combination' in first_model
            assert 'auc' in first_model
            assert first_model['auc'] >= 0 and first_model['auc'] <= 1
    
    def test_run_roc_ratios_analysis_with_cv(self, analyzer):
        """Test running ROC ratios analysis with cross-validation."""
        cv_config = CrossValidationConfig(
            enable_loo=True,
            enable_bootstrap=True,
            bootstrap_iterations=10  # Small number for testing
        )
        
        analysis = ROCRatiosAnalysis(
            name="Test CV Ratios Analysis",
            description="Test with CV",
            experiment_id=analyzer._test_experiment_id,
            prevalence=0.3,
            max_combination_size=1,
            cross_validation_config=cv_config
        )
        
        results = analyzer.run_roc_ratios_analysis(analysis)
        
        assert results['models_created'] > 0
        
        # Check that CV results are included
        if results['successful_models']:
            first_model = results['successful_models'][0]
            if first_model['cross_validation_results']:
                cv_results = first_model['cross_validation_results']
                # Should have LOO and/or bootstrap results
                assert (hasattr(cv_results, 'loo_auc_mean') and cv_results.loo_auc_mean is not None) or \
                       (hasattr(cv_results, 'bootstrap_auc_mean') and cv_results.bootstrap_auc_mean is not None)
    
    def test_generate_analysis_report(self, analyzer):
        """Test generating analysis report."""
        # First run an analysis
        analysis = ROCRatiosAnalysis(
            name="Test Report Analysis",
            description="Test for report",
            experiment_id=analyzer._test_experiment_id,
            prevalence=0.3,
            max_combination_size=1
        )
        
        results = analyzer.run_roc_ratios_analysis(analysis)
        analysis_id = results['analysis_id']
        
        # Generate report
        df = analyzer.generate_analysis_report(analysis_id)
        
        assert df is not None
        assert len(df) > 0
        
        # Check expected columns
        expected_cols = ['Model_ID', 'AUC', 'Prevalence']
        for col in expected_cols:
            assert col in df.columns
        
        # Check that ratio columns are present
        ratio_cols = [col for col in df.columns if col.startswith('Ratio_')]
        assert len(ratio_cols) > 0
        
        # Check threshold metrics columns
        threshold_types = ['se_97', 'se_95', 'max_sum']
        metric_types = ['Threshold', 'Sensitivity', 'Specificity', 'NPV', 'PPV']
        
        for threshold_type in threshold_types:
            for metric_type in metric_types:
                col_name = f"{threshold_type}_{metric_type}"
                assert col_name in df.columns
    
    def test_error_handling(self, analyzer):
        """Test error handling for invalid inputs."""
        # Test with non-existent experiment
        analysis = ROCRatiosAnalysis(
            name="Invalid Analysis",
            description="Should fail",
            experiment_id=9999,  # Non-existent
            prevalence=0.3,
            max_combination_size=1
        )
        
        # This should fail with either a foreign key error or "No valid data found" error
        # Both are valid error conditions for a non-existent experiment
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            analyzer.run_roc_ratios_analysis(analysis)
        
        # Test with experiment that has only one biomarker
        # Create a valid analysis first, then mock the experiment data
        valid_analysis = ROCRatiosAnalysis(
            name="Valid Analysis",
            description="For biomarker validation test",
            experiment_id=analyzer._test_experiment_id,  # Use existing experiment
            prevalence=0.3,
            max_combination_size=1
        )
        
        experiment_data = {
            'biomarker_versions': [1],  # Only one biomarker
            'dataframe': None,
            'sample_count': 0,
            'positive_cases': 0,
            'negative_cases': 0
        }
        
        # Mock the _prepare_experiment_data method temporarily
        original_method = analyzer._prepare_experiment_data
        analyzer._prepare_experiment_data = lambda exp_id: experiment_data
        
        try:
            with pytest.raises(ValueError, match="At least 2 biomarkers are required"):
                analyzer.run_roc_ratios_analysis(valid_analysis)
        finally:
            # Restore original method
            analyzer._prepare_experiment_data = original_method


if __name__ == "__main__":
    pytest.main([__file__])