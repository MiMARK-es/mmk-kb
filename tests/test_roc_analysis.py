"""
Tests for ROC analysis functionality.
"""
import pytest
import tempfile
import os
import numpy as np
import pandas as pd
from src.mmkkb.roc_analysis import ROCAnalysisDatabase, ROCAnalyzer, ROCAnalysis
from src.mmkkb.experiments import ExperimentDatabase, Experiment, Measurement
from src.mmkkb.samples import SampleDatabase, Sample
from src.mmkkb.projects import ProjectDatabase, Project


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def setup_test_data(temp_db):
    """Set up test data for ROC analysis."""
    # Create databases
    project_db = ProjectDatabase(temp_db)
    sample_db = SampleDatabase(temp_db)
    exp_db = ExperimentDatabase(temp_db)
    
    # Create project
    project = Project(
        code="TEST_ROC",
        name="Test ROC Project",
        description="Project for ROC testing",
        creator="Test Creator"
    )
    project = project_db.create_project(project)
    
    # Create samples (mix of cases and controls)
    samples = []
    
    # Cases (dx=1)
    for i in range(1, 26):  # 25 cases
        sample = Sample(
            code=f"CASE_{i:03d}",
            age=45 + i,
            bmi=25.0 + i * 0.1,
            dx=True,  # disease
            dx_origin="biopsy",
            collection_center="Hospital_A",
            processing_time=120,
            project_id=project.id
        )
        samples.append(sample_db.create_sample(sample))
    
    # Controls (dx=0)
    for i in range(1, 26):  # 25 controls
        sample = Sample(
            code=f"CTRL_{i:03d}",
            age=45 + i,
            bmi=25.0 + i * 0.1,
            dx=False,  # benign
            dx_origin="screening",
            collection_center="Hospital_B",
            processing_time=90,
            project_id=project.id
        )
        samples.append(sample_db.create_sample(sample))
    
    # Create experiment
    experiment = Experiment(
        name="Test Biomarker Study",
        description="Test experiment for ROC analysis",
        project_id=project.id,
        csv_filename="test_data.csv"
    )
    experiment = exp_db.create_experiment(experiment)
    
    # Create biomarkers and measurements
    biomarker_versions = []
    
    # Create 3 biomarkers
    for i, biomarker_name in enumerate(['Biomarker_A', 'Biomarker_B', 'Biomarker_C'], 1):
        bv = exp_db.create_biomarker_with_version(
            biomarker_name=biomarker_name,
            version="v1.0",
            biomarker_description=f"Test biomarker {biomarker_name}",
            category="test"
        )
        biomarker_versions.append(bv)
        
        # Create measurements for each sample
        np.random.seed(42 + i)  # Reproducible random data
        
        for sample in samples:
            # Make cases generally have different values than controls
            if sample.dx:  # Cases
                base_value = 2.0 + i * 0.5
                noise = np.random.normal(0, 0.5)
            else:  # Controls
                base_value = 4.0 + i * 0.5
                noise = np.random.normal(0, 0.5)
            
            measurement_value = max(0.1, base_value + noise)  # Ensure positive
            
            measurement = Measurement(
                experiment_id=experiment.id,
                sample_id=sample.id,
                biomarker_version_id=bv.id,
                value=measurement_value
            )
            exp_db.create_measurement(measurement)
    
    return {
        'db_path': temp_db,
        'project': project,
        'experiment': experiment,
        'samples': samples,
        'biomarker_versions': biomarker_versions
    }


class TestROCAnalysisDatabase:
    """Test ROC analysis database operations."""
    
    def test_init_database(self, temp_db):
        """Test database initialization."""
        roc_db = ROCAnalysisDatabase(temp_db)
        
        # Check tables exist
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'roc_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
        expected_tables = ['roc_analyses', 'roc_models', 'roc_metrics', 'roc_curve_points']
        for table in expected_tables:
            assert table in tables
    
    def test_create_roc_analysis(self, temp_db, setup_test_data):
        """Test creating ROC analysis."""
        data = setup_test_data
        roc_db = ROCAnalysisDatabase(temp_db)
        
        analysis = ROCAnalysis(
            name="Test Analysis",
            description="Test ROC analysis",
            experiment_id=data['experiment'].id,
            prevalence=0.3,
            max_combination_size=2
        )
        
        created_analysis = roc_db.create_roc_analysis(analysis)
        
        assert created_analysis.id is not None
        assert created_analysis.name == "Test Analysis"
        assert created_analysis.prevalence == 0.3
        assert created_analysis.max_combination_size == 2
        assert created_analysis.created_at is not None
    
    def test_get_roc_analysis_by_id(self, temp_db, setup_test_data):
        """Test retrieving ROC analysis by ID."""
        data = setup_test_data
        roc_db = ROCAnalysisDatabase(temp_db)
        
        analysis = ROCAnalysis(
            name="Test Analysis",
            description="Test ROC analysis",
            experiment_id=data['experiment'].id,
            prevalence=0.3,
            max_combination_size=2
        )
        
        created_analysis = roc_db.create_roc_analysis(analysis)
        retrieved_analysis = roc_db.get_roc_analysis_by_id(created_analysis.id)
        
        assert retrieved_analysis is not None
        assert retrieved_analysis.id == created_analysis.id
        assert retrieved_analysis.name == created_analysis.name


class TestROCAnalyzer:
    """Test ROC analyzer functionality."""
    
    def test_prepare_experiment_data(self, setup_test_data):
        """Test experiment data preparation."""
        data = setup_test_data
        analyzer = ROCAnalyzer(data['db_path'])
        
        experiment_data = analyzer._prepare_experiment_data(data['experiment'].id)
        
        assert experiment_data is not None
        assert 'dataframe' in experiment_data
        assert 'biomarker_versions' in experiment_data
        assert 'sample_count' in experiment_data
        assert 'positive_cases' in experiment_data
        assert 'negative_cases' in experiment_data
        
        # Check data structure
        df = experiment_data['dataframe']
        assert len(df) == 50  # 25 cases + 25 controls
        assert 'dx' in df.columns
        assert 'sample_id' in df.columns
        
        # Check biomarker columns exist
        for bv_id in experiment_data['biomarker_versions']:
            assert f'biomarker_{bv_id}' in df.columns
        
        # Check class distribution
        assert experiment_data['positive_cases'] == 25
        assert experiment_data['negative_cases'] == 25
    
    def test_generate_biomarker_combinations(self, setup_test_data):
        """Test biomarker combination generation."""
        data = setup_test_data
        analyzer = ROCAnalyzer(data['db_path'])
        
        biomarker_versions = [1, 2, 3]
        
        # Test max size 1
        combinations = analyzer._generate_biomarker_combinations(biomarker_versions, 1)
        assert len(combinations) == 3
        assert [1] in combinations
        assert [2] in combinations
        assert [3] in combinations
        
        # Test max size 2
        combinations = analyzer._generate_biomarker_combinations(biomarker_versions, 2)
        assert len(combinations) == 6  # 3 single + 3 pairs
        assert [1, 2] in combinations
        assert [1, 3] in combinations
        assert [2, 3] in combinations
        
        # Test max size 3
        combinations = analyzer._generate_biomarker_combinations(biomarker_versions, 3)
        assert len(combinations) == 7  # 3 single + 3 pairs + 1 triplet
        assert [1, 2, 3] in combinations
    
    def test_run_roc_analysis(self, setup_test_data):
        """Test running complete ROC analysis."""
        data = setup_test_data
        analyzer = ROCAnalyzer(data['db_path'])
        
        analysis = ROCAnalysis(
            name="Complete Test Analysis",
            description="Test complete ROC analysis",
            experiment_id=data['experiment'].id,
            prevalence=0.3,
            max_combination_size=2
        )
        
        results = analyzer.run_roc_analysis(analysis)
        
        assert 'analysis_id' in results
        assert 'total_combinations' in results
        assert 'models_created' in results
        assert 'successful_models' in results
        assert 'failed_models' in results
        
        # Should create 6 models (3 single + 3 pairs)
        assert results['total_combinations'] == 6
        assert results['models_created'] > 0
        
        # Check some models were successful
        assert len(results['successful_models']) > 0
        
        # Check model structure
        for model in results['successful_models']:
            assert 'model_id' in model
            assert 'biomarker_combination' in model
            assert 'auc' in model
            assert 'metrics' in model
            assert 'roc_points_count' in model
            
            # AUC should be between 0 and 1
            assert 0 <= model['auc'] <= 1
    
    def test_generate_analysis_report(self, setup_test_data):
        """Test generating analysis report."""
        data = setup_test_data
        analyzer = ROCAnalyzer(data['db_path'])
        
        # Run analysis first
        analysis = ROCAnalysis(
            name="Report Test Analysis",
            description="Test analysis for report generation",
            experiment_id=data['experiment'].id,
            prevalence=0.3,
            max_combination_size=2
        )
        
        results = analyzer.run_roc_analysis(analysis)
        
        # Generate report
        report_df = analyzer.generate_analysis_report(results['analysis_id'])
        
        assert not report_df.empty
        assert 'Model_ID' in report_df.columns
        assert 'AUC' in report_df.columns
        assert 'Prevalence' in report_df.columns
        assert 'Biomarker_1' in report_df.columns
        
        # Check threshold metrics columns exist
        threshold_types = ['se_97', 'se_95', 'max_sum']
        metrics = ['Threshold', 'Sensitivity', 'Specificity', 'NPV', 'PPV']
        
        for threshold_type in threshold_types:
            for metric in metrics:
                col_name = f'{threshold_type}_{metric}'
                assert col_name in report_df.columns
        
        # Check data values
        assert all(0 <= auc <= 1 for auc in report_df['AUC'])
        assert all(prev == 0.3 for prev in report_df['Prevalence'])


class TestROCAnalysisIntegration:
    """Integration tests for ROC analysis."""
    
    def test_full_workflow(self, setup_test_data):
        """Test complete ROC analysis workflow."""
        data = setup_test_data
        
        # 1. Create analysis
        analyzer = ROCAnalyzer(data['db_path'])
        analysis = ROCAnalysis(
            name="Integration Test",
            description="Full workflow test",
            experiment_id=data['experiment'].id,
            prevalence=0.4,
            max_combination_size=3
        )
        
        # 2. Run analysis
        results = analyzer.run_roc_analysis(analysis)
        assert results['models_created'] > 0
        
        # 3. Get analysis details
        roc_db = ROCAnalysisDatabase(data['db_path'])
        retrieved_analysis = roc_db.get_roc_analysis_by_id(results['analysis_id'])
        assert retrieved_analysis is not None
        
        # 4. Get models
        models = roc_db.get_roc_models_by_analysis(results['analysis_id'])
        assert len(models) > 0
        
        # 5. Get metrics for first model
        first_model = models[0]
        metrics = roc_db.get_roc_metrics_by_model(first_model.id)
        assert len(metrics) > 0
        
        # 6. Get ROC curve points
        roc_points = roc_db.get_roc_curve_points_by_model(first_model.id)
        assert len(roc_points) > 0
        
        # 7. Generate report
        report_df = analyzer.generate_analysis_report(results['analysis_id'])
        assert not report_df.empty
        
        # Check report has expected structure
        assert len(report_df) == len(models)
        assert all(model_id in report_df['Model_ID'].values for model_id in [m.id for m in models])
    
    def test_edge_cases(self, setup_test_data):
        """Test edge cases and error handling."""
        data = setup_test_data
        analyzer = ROCAnalyzer(data['db_path'])
        
        # Test with non-existent experiment - should fail at data preparation stage
        # We'll test this by directly calling the data preparation method
        experiment_data = analyzer._prepare_experiment_data(9999)
        assert experiment_data is None
        
        # Test report generation with non-existent analysis
        with pytest.raises(ValueError, match="Analysis .* not found"):
            analyzer.generate_analysis_report(9999)


if __name__ == "__main__":
    pytest.main([__file__])