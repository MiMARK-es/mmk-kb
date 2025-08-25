"""
Tests for ROC Normalized Analysis functionality.
"""
import pytest
import tempfile
import os
import pandas as pd
from datetime import datetime

from src.mmkkb.analyses.roc_normalized_analysis import (
    ROCNormalizedAnalysisDatabase, ROCNormalizedAnalyzer, ROCNormalizedAnalysis,
    ROCNormalizedModel, ROCNormalizedMetrics, ROCNormalizedCurvePoint
)
from src.mmkkb.projects import ProjectDatabase, Project
from src.mmkkb.samples import SampleDatabase, Sample
from src.mmkkb.experiments import ExperimentDatabase, Experiment, Biomarker, BiomarkerVersion, Measurement


class TestROCNormalizedAnalysisDatabase:
    """Test ROC Normalized Analysis database operations."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)
    
    @pytest.fixture
    def roc_norm_db(self, temp_db):
        """Create ROC Normalized Analysis database instance."""
        return ROCNormalizedAnalysisDatabase(temp_db)
    
    def test_init_database(self, roc_norm_db):
        """Test database initialization creates all tables."""
        # Test should not raise any exceptions
        roc_norm_db.init_database()
        
        # Verify tables exist by attempting to query them
        with roc_norm_db._get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'roc_normalized_analyses' in tables
            assert 'roc_normalized_models' in tables
            assert 'roc_normalized_metrics' in tables
            assert 'roc_normalized_curve_points' in tables
    
    def test_create_roc_normalized_analysis(self, roc_norm_db, temp_db):
        """Test creating ROC Normalized analysis."""
        # Setup dependencies
        proj_db = ProjectDatabase(temp_db)
        exp_db = ExperimentDatabase(temp_db)
        
        # Create project and experiment
        project = Project(code="TEST001", name="Test Project", description="Test", creator="Test")
        proj_db.create_project(project)
        
        experiment = Experiment(name="Test Experiment", description="Test", project_id=project.id)
        exp_db.create_experiment(experiment)
        
        # Create biomarker and version for normalizer
        biomarker = Biomarker(name="TestBiomarker", description="Test")
        exp_db.create_biomarker(biomarker)
        
        bv = BiomarkerVersion(biomarker_id=biomarker.id, version="v1", description="Test")
        exp_db.create_biomarker_version(bv)
        
        # Create analysis
        analysis = ROCNormalizedAnalysis(
            name="Test Analysis",
            description="Test Description",
            experiment_id=experiment.id,
            normalizer_biomarker_version_id=bv.id,
            prevalence=0.3,
            max_combination_size=2
        )
        
        result = roc_norm_db.create_roc_normalized_analysis(analysis)
        
        assert result.id is not None
        assert result.name == "Test Analysis"
        assert result.normalizer_biomarker_version_id == bv.id
        assert result.created_at is not None
    
    def test_get_roc_normalized_analysis_by_id(self, roc_norm_db, temp_db):
        """Test retrieving ROC Normalized analysis by ID."""
        # Setup and create analysis
        proj_db = ProjectDatabase(temp_db)
        exp_db = ExperimentDatabase(temp_db)
        
        project = Project(code="TEST001", name="Test Project", description="Test", creator="Test")
        proj_db.create_project(project)
        
        experiment = Experiment(name="Test Experiment", description="Test", project_id=project.id)
        exp_db.create_experiment(experiment)
        
        biomarker = Biomarker(name="TestBiomarker", description="Test")
        exp_db.create_biomarker(biomarker)
        
        bv = BiomarkerVersion(biomarker_id=biomarker.id, version="v1", description="Test")
        exp_db.create_biomarker_version(bv)
        
        analysis = ROCNormalizedAnalysis(
            name="Test Analysis",
            description="Test Description",
            experiment_id=experiment.id,
            normalizer_biomarker_version_id=bv.id,
            prevalence=0.3,
            max_combination_size=2
        )
        
        created = roc_norm_db.create_roc_normalized_analysis(analysis)
        retrieved = roc_norm_db.get_roc_normalized_analysis_by_id(created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "Test Analysis"
        assert retrieved.normalizer_biomarker_version_id == bv.id
    
    def test_list_roc_normalized_analyses(self, roc_norm_db, temp_db):
        """Test listing ROC Normalized analyses."""
        # Setup and create multiple analyses
        proj_db = ProjectDatabase(temp_db)
        exp_db = ExperimentDatabase(temp_db)
        
        project = Project(code="TEST001", name="Test Project", description="Test", creator="Test")
        proj_db.create_project(project)
        
        experiment = Experiment(name="Test Experiment", description="Test", project_id=project.id)
        exp_db.create_experiment(experiment)
        
        biomarker = Biomarker(name="TestBiomarker", description="Test")
        exp_db.create_biomarker(biomarker)
        
        bv = BiomarkerVersion(biomarker_id=biomarker.id, version="v1", description="Test")
        exp_db.create_biomarker_version(bv)
        
        # Create two analyses
        analysis1 = ROCNormalizedAnalysis(
            name="Analysis 1", description="Test 1", experiment_id=experiment.id,
            normalizer_biomarker_version_id=bv.id, prevalence=0.3, max_combination_size=2
        )
        analysis2 = ROCNormalizedAnalysis(
            name="Analysis 2", description="Test 2", experiment_id=experiment.id,
            normalizer_biomarker_version_id=bv.id, prevalence=0.4, max_combination_size=3
        )
        
        roc_norm_db.create_roc_normalized_analysis(analysis1)
        roc_norm_db.create_roc_normalized_analysis(analysis2)
        
        analyses = roc_norm_db.list_roc_normalized_analyses()
        assert len(analyses) == 2
        
        # Test filtering by experiment
        filtered = roc_norm_db.list_roc_normalized_analyses(experiment.id)
        assert len(filtered) == 2
    
    def test_create_roc_normalized_model(self, roc_norm_db, temp_db):
        """Test creating ROC Normalized model."""
        # Setup analysis first
        proj_db = ProjectDatabase(temp_db)
        exp_db = ExperimentDatabase(temp_db)
        
        project = Project(code="TEST001", name="Test Project", description="Test", creator="Test")
        proj_db.create_project(project)
        
        experiment = Experiment(name="Test Experiment", description="Test", project_id=project.id)
        exp_db.create_experiment(experiment)
        
        biomarker = Biomarker(name="TestBiomarker", description="Test")
        exp_db.create_biomarker(biomarker)
        
        bv = BiomarkerVersion(biomarker_id=biomarker.id, version="v1", description="Test")
        exp_db.create_biomarker_version(bv)
        
        analysis = ROCNormalizedAnalysis(
            name="Test Analysis", description="Test", experiment_id=experiment.id,
            normalizer_biomarker_version_id=bv.id, prevalence=0.3, max_combination_size=2
        )
        created_analysis = roc_norm_db.create_roc_normalized_analysis(analysis)
        
        # Create model
        model = ROCNormalizedModel(
            analysis_id=created_analysis.id,
            biomarker_combination=[bv.id],
            normalizer_biomarker_version_id=bv.id,
            auc=0.75,
            coefficients={"intercept": 0.1, "coef": [0.5], "biomarker_version_ids": [bv.id]}
        )
        
        result = roc_norm_db.create_roc_normalized_model(model)
        
        assert result.id is not None
        assert result.auc == 0.75
        assert result.normalizer_biomarker_version_id == bv.id
        assert result.created_at is not None


class TestROCNormalizedAnalyzer:
    """Test ROC Normalized Analyzer functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)
    
    @pytest.fixture
    def setup_test_data(self, temp_db):
        """Setup test data for analysis."""
        # Create databases
        proj_db = ProjectDatabase(temp_db)
        sample_db = SampleDatabase(temp_db)
        exp_db = ExperimentDatabase(temp_db)
        
        # Create project
        project = Project(code="TEST001", name="Test Project", description="Test", creator="Test")
        proj_db.create_project(project)
        
        # Create samples
        samples = []
        for i in range(20):
            sample = Sample(
                code=f"S{i:03d}",
                age=30 + i,
                bmi=25.0,
                dx=i % 2,  # Alternating diagnosis
                dx_origin="Test",
                collection_center="Test Center",
                processing_time=120,
                project_id=project.id
            )
            samples.append(sample_db.create_sample(sample))
        
        # Create experiment
        experiment = Experiment(name="Test Experiment", description="Test", project_id=project.id)
        experiment = exp_db.create_experiment(experiment)
        
        # Create biomarkers
        biomarkers = []
        biomarker_versions = []
        
        for i in range(3):
            biomarker = Biomarker(name=f"Biomarker_{i}", description=f"Test biomarker {i}")
            biomarker = exp_db.create_biomarker(biomarker)
            biomarkers.append(biomarker)
            
            bv = BiomarkerVersion(biomarker_id=biomarker.id, version="v1", description="Version 1")
            bv = exp_db.create_biomarker_version(bv)
            biomarker_versions.append(bv)
        
        # Create measurements
        for i, sample in enumerate(samples):
            for j, bv in enumerate(biomarker_versions):
                # Create different patterns for different biomarkers
                if j == 0:  # Normalizer - always positive values
                    value = 10.0 + (i * 0.5)
                elif j == 1:  # Correlated with diagnosis
                    value = 5.0 + (sample.dx * 3.0) + (i * 0.2)
                else:  # Random-ish values
                    value = 2.0 + (i * 0.1)
                
                measurement = Measurement(
                    experiment_id=experiment.id,
                    sample_id=sample.id,
                    biomarker_version_id=bv.id,
                    value=value
                )
                exp_db.create_measurement(measurement)
        
        return {
            'temp_db': temp_db,
            'experiment': experiment,
            'biomarker_versions': biomarker_versions,
            'samples': samples
        }
    
    def test_run_roc_normalized_analysis(self, setup_test_data):
        """Test running complete ROC Normalized analysis."""
        temp_db = setup_test_data['temp_db']
        experiment = setup_test_data['experiment']
        biomarker_versions = setup_test_data['biomarker_versions']
        
        analyzer = ROCNormalizedAnalyzer(temp_db)
        
        # Use first biomarker as normalizer
        normalizer_bv_id = biomarker_versions[0].id
        
        analysis = ROCNormalizedAnalysis(
            name="Test Normalized Analysis",
            description="Test normalized analysis",
            experiment_id=experiment.id,
            normalizer_biomarker_version_id=normalizer_bv_id,
            prevalence=0.5,
            max_combination_size=2
        )
        
        results = analyzer.run_roc_normalized_analysis(analysis)
        
        assert 'analysis_id' in results
        assert 'normalizer_biomarker' in results
        assert results['normalizer_biomarker'] == normalizer_bv_id
        assert results['models_created'] >= 0
        assert 'successful_models' in results
        assert 'failed_models' in results
    
    def test_prepare_experiment_data_with_normalization(self, setup_test_data):
        """Test data preparation with normalization."""
        temp_db = setup_test_data['temp_db']
        experiment = setup_test_data['experiment']
        biomarker_versions = setup_test_data['biomarker_versions']
        
        analyzer = ROCNormalizedAnalyzer(temp_db)
        normalizer_bv_id = biomarker_versions[0].id
        
        # Test private method
        experiment_data = analyzer._prepare_experiment_data(experiment.id, normalizer_bv_id)
        
        assert experiment_data is not None
        assert 'dataframe' in experiment_data
        assert 'biomarker_versions' in experiment_data
        assert 'normalizer_biomarker_version_id' in experiment_data
        assert experiment_data['normalizer_biomarker_version_id'] == normalizer_bv_id
        
        # Check that normalizer is excluded from available biomarkers
        assert normalizer_bv_id not in experiment_data['biomarker_versions']
        
        # Check that data is normalized (values should be different from original)
        df = experiment_data['dataframe']
        assert len(df) > 0
        assert 'dx' in df.columns
        
        # Should have normalized biomarker columns (excluding normalizer)
        expected_cols = [f'biomarker_{bv.id}' for bv in biomarker_versions if bv.id != normalizer_bv_id]
        for col in expected_cols:
            assert col in df.columns
    
    def test_generate_analysis_report(self, setup_test_data):
        """Test generating analysis report."""
        temp_db = setup_test_data['temp_db']
        experiment = setup_test_data['experiment']
        biomarker_versions = setup_test_data['biomarker_versions']
        
        analyzer = ROCNormalizedAnalyzer(temp_db)
        normalizer_bv_id = biomarker_versions[0].id
        
        # Run a small analysis first
        analysis = ROCNormalizedAnalysis(
            name="Test Report Analysis",
            description="Test",
            experiment_id=experiment.id,
            normalizer_biomarker_version_id=normalizer_bv_id,
            prevalence=0.5,
            max_combination_size=1  # Keep it small for testing
        )
        
        results = analyzer.run_roc_normalized_analysis(analysis)
        
        # Generate report
        report_df = analyzer.generate_analysis_report(results['analysis_id'])
        
        assert isinstance(report_df, pd.DataFrame)
        
        if not report_df.empty:
            # Check expected columns
            expected_columns = ['Model_ID', 'AUC', 'Normalizer', 'Prevalence']
            for col in expected_columns:
                assert col in report_df.columns
            
            # Check that normalizer information is included
            if len(report_df) > 0:
                assert 'Normalizer' in report_df.columns
                # Should contain biomarker ratios
                biomarker_cols = [col for col in report_df.columns if col.startswith('Biomarker_')]
                for col in biomarker_cols:
                    if pd.notna(report_df[col].iloc[0]):
                        assert '/' in str(report_df[col].iloc[0])  # Should contain ratio notation


class TestROCNormalizedCLI:
    """Test ROC Normalized CLI commands integration."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)
    
    def test_roc_normalized_commands_available(self):
        """Test that ROC Normalized commands are available in CLI."""
        from src.mmkkb.cli.roc_normalized_commands import ROCNormalizedAnalysisCommandHandler
        
        handler = ROCNormalizedAnalysisCommandHandler()
        
        # Test that handler exists and has required methods
        assert hasattr(handler, 'add_commands')
        assert hasattr(handler, 'handle_command')
        assert callable(handler.add_commands)
        assert callable(handler.handle_command)