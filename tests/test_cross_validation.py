"""
Test cross-validation functionality for MMK Knowledge Base analyses.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.mmkkb.analyses.base_analysis import BaseAnalyzer, CrossValidationConfig, CrossValidationResults
from src.mmkkb.analyses.roc_analysis import ROCAnalyzer, ROCAnalysis
from src.mmkkb.analyses.roc_normalized_analysis import ROCNormalizedAnalyzer, ROCNormalizedAnalysis


class TestCrossValidation:
    """Test cross-validation functionality."""
    
    def test_cross_validation_config_defaults(self):
        """Test default cross-validation configuration."""
        config = CrossValidationConfig()
        assert config.enable_loo is True
        assert config.enable_bootstrap is True
        assert config.bootstrap_iterations == 200
        assert config.bootstrap_validation_size == 0.2
    
    def test_cross_validation_config_custom(self):
        """Test custom cross-validation configuration."""
        config = CrossValidationConfig(
            enable_loo=False,
            enable_bootstrap=True,
            bootstrap_iterations=100,
            bootstrap_validation_size=0.3
        )
        assert config.enable_loo is False
        assert config.enable_bootstrap is True
        assert config.bootstrap_iterations == 100
        assert config.bootstrap_validation_size == 0.3
    
    def test_cross_validation_results_structure(self):
        """Test cross-validation results structure."""
        results = CrossValidationResults(
            loo_auc_mean=0.85,
            loo_auc_std=0.05,
            loo_aucs=[0.8, 0.85, 0.9],
            bootstrap_auc_mean=0.82,
            bootstrap_auc_std=0.08,
            bootstrap_aucs=[0.75, 0.82, 0.89]
        )
        
        assert results.loo_auc_mean == 0.85
        assert results.loo_auc_std == 0.05
        assert len(results.loo_aucs) == 3
        assert results.bootstrap_auc_mean == 0.82
        assert results.bootstrap_auc_std == 0.08
        assert len(results.bootstrap_aucs) == 3
    
    def test_loo_cross_validation(self):
        """Test Leave-One-Out cross-validation."""
        # Create mock analyzer
        class MockAnalyzer(BaseAnalyzer):
            def _prepare_experiment_data(self, experiment_id, **kwargs):
                return None
            def _generate_biomarker_combinations(self, biomarker_versions, max_size):
                return []
        
        analyzer = MockAnalyzer()
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 50
        X = np.random.randn(n_samples, 2)
        y = np.random.choice([0, 1], n_samples)
        
        # Perform LOO CV
        loo_aucs = analyzer._perform_loo_cv(X, y)
        
        # Check results
        assert isinstance(loo_aucs, list)
        assert len(loo_aucs) > 0
        assert all(isinstance(auc, float) for auc in loo_aucs)
        assert all(0 <= auc <= 1 for auc in loo_aucs)
    
    def test_bootstrap_cross_validation(self):
        """Test Bootstrap cross-validation."""
        # Create mock analyzer
        class MockAnalyzer(BaseAnalyzer):
            def _prepare_experiment_data(self, experiment_id, **kwargs):
                return None
            def _generate_biomarker_combinations(self, biomarker_versions, max_size):
                return []
        
        analyzer = MockAnalyzer()
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = np.random.choice([0, 1], n_samples)
        
        # Perform Bootstrap CV
        bootstrap_aucs = analyzer._perform_bootstrap_cv(X, y, n_iterations=50, validation_size=0.2)
        
        # Check results
        assert isinstance(bootstrap_aucs, list)
        assert len(bootstrap_aucs) > 0
        assert all(isinstance(auc, float) for auc in bootstrap_aucs)
        assert all(0 <= auc <= 1 for auc in bootstrap_aucs)
    
    def test_perform_cross_validation_both_enabled(self):
        """Test cross-validation with both LOO and Bootstrap enabled."""
        # Create mock analyzer
        class MockAnalyzer(BaseAnalyzer):
            def _prepare_experiment_data(self, experiment_id, **kwargs):
                return None
            def _generate_biomarker_combinations(self, biomarker_versions, max_size):
                return []
        
        analyzer = MockAnalyzer()
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = np.random.choice([0, 1], 30)
        
        # Configure CV
        cv_config = CrossValidationConfig(
            enable_loo=True,
            enable_bootstrap=True,
            bootstrap_iterations=20,
            bootstrap_validation_size=0.25
        )
        
        # Perform CV
        results = analyzer._perform_cross_validation(X, y, cv_config)
        
        # Check results
        assert isinstance(results, CrossValidationResults)
        assert results.loo_auc_mean is not None
        assert results.loo_auc_std is not None
        assert results.loo_aucs is not None
        assert results.bootstrap_auc_mean is not None
        assert results.bootstrap_auc_std is not None
        assert results.bootstrap_aucs is not None
        
        assert 0 <= results.loo_auc_mean <= 1
        assert results.loo_auc_std >= 0
        assert 0 <= results.bootstrap_auc_mean <= 1
        assert results.bootstrap_auc_std >= 0
    
    def test_perform_cross_validation_loo_only(self):
        """Test cross-validation with only LOO enabled."""
        # Create mock analyzer
        class MockAnalyzer(BaseAnalyzer):
            def _prepare_experiment_data(self, experiment_id, **kwargs):
                return None
            def _generate_biomarker_combinations(self, biomarker_versions, max_size):
                return []
        
        analyzer = MockAnalyzer()
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(25, 2)
        y = np.random.choice([0, 1], 25)
        
        # Configure CV
        cv_config = CrossValidationConfig(
            enable_loo=True,
            enable_bootstrap=False
        )
        
        # Perform CV
        results = analyzer._perform_cross_validation(X, y, cv_config)
        
        # Check results
        assert isinstance(results, CrossValidationResults)
        assert results.loo_auc_mean is not None
        assert results.loo_auc_std is not None
        assert results.loo_aucs is not None
        assert results.bootstrap_auc_mean is None
        assert results.bootstrap_auc_std is None
        assert results.bootstrap_aucs is None
    
    def test_perform_cross_validation_bootstrap_only(self):
        """Test cross-validation with only Bootstrap enabled."""
        # Create mock analyzer
        class MockAnalyzer(BaseAnalyzer):
            def _prepare_experiment_data(self, experiment_id, **kwargs):
                return None
            def _generate_biomarker_combinations(self, biomarker_versions, max_size):
                return []
        
        analyzer = MockAnalyzer()
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(40, 2)
        y = np.random.choice([0, 1], 40)
        
        # Configure CV
        cv_config = CrossValidationConfig(
            enable_loo=False,
            enable_bootstrap=True,
            bootstrap_iterations=30
        )
        
        # Perform CV
        results = analyzer._perform_cross_validation(X, y, cv_config)
        
        # Check results
        assert isinstance(results, CrossValidationResults)
        assert results.loo_auc_mean is None
        assert results.loo_auc_std is None
        assert results.loo_aucs is None
        assert results.bootstrap_auc_mean is not None
        assert results.bootstrap_auc_std is not None
        assert results.bootstrap_aucs is not None


class TestROCAnalysisWithCrossValidation:
    """Test ROC analysis with cross-validation integration."""
    
    @patch('src.mmkkb.analyses.roc_analysis.ExperimentDatabase')
    @patch('src.mmkkb.analyses.roc_analysis.SampleDatabase')
    def test_roc_analysis_with_cv_config(self, mock_sample_db, mock_exp_db):
        """Test ROC analysis creation with cross-validation configuration."""
        # Setup mock data
        analyzer = ROCAnalyzer()
        
        cv_config = CrossValidationConfig(
            enable_loo=True,
            enable_bootstrap=True,
            bootstrap_iterations=50
        )
        
        analysis = ROCAnalysis(
            name="Test Analysis",
            description="Test with CV",
            experiment_id=1,
            prevalence=0.3,
            max_combination_size=2,
            cross_validation_config=cv_config
        )
        
        # Check that analysis includes CV config
        assert analysis.cross_validation_config is not None
        assert analysis.cross_validation_config.enable_loo is True
        assert analysis.cross_validation_config.enable_bootstrap is True
        assert analysis.cross_validation_config.bootstrap_iterations == 50
    
    def test_roc_analysis_database_cv_storage(self):
        """Test that cross-validation config is properly stored in database."""
        from src.mmkkb.analyses.roc_analysis import ROCAnalysisDatabase
        
        # Create temporary database and setup full schema
        import tempfile
        import sqlite3
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            # Setup minimal schema for testing
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                # Create experiments table (minimal for foreign key)
                conn.execute("""
                    CREATE TABLE experiments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL
                    )
                """)
                # Insert test experiment
                conn.execute("INSERT INTO experiments (name) VALUES ('Test Experiment')")
                conn.commit()
            
            db = ROCAnalysisDatabase(db_path)
            
            cv_config = CrossValidationConfig(
                enable_loo=True,
                enable_bootstrap=False,
                bootstrap_iterations=100
            )
            
            analysis = ROCAnalysis(
                name="CV Test",
                description="Test CV storage",
                experiment_id=1,
                prevalence=0.25,
                max_combination_size=3,
                cross_validation_config=cv_config
            )
            
            # Store analysis
            created_analysis = db.create_roc_analysis(analysis)
            
            # Retrieve analysis
            retrieved_analysis = db.get_roc_analysis_by_id(created_analysis.id)
            
            # Check CV config was stored and retrieved correctly
            assert retrieved_analysis is not None
            assert retrieved_analysis.cross_validation_config is not None
            assert retrieved_analysis.cross_validation_config['enable_loo'] is True
            assert retrieved_analysis.cross_validation_config['enable_bootstrap'] is False
            assert retrieved_analysis.cross_validation_config['bootstrap_iterations'] == 100
            
        finally:
            import os
            os.unlink(db_path)


class TestCLIIntegration:
    """Test CLI integration with cross-validation parameters."""
    
    def test_cli_cv_parameters_parsing(self):
        """Test that CLI correctly parses cross-validation parameters."""
        import argparse
        from src.mmkkb.cli.analysis_commands import AnalysisCommandHandler
        
        handler = AnalysisCommandHandler()
        
        # Create parser
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers(dest='command')
        handler.add_commands(subparsers)
        
        # Test ROC analysis with CV parameters
        args = main_parser.parse_args([
            'analysis', 'roc-run', '1', 'Test Analysis', '0.3',
            '--enable-cv',
            '--bootstrap-iterations', '150',
            '--bootstrap-validation-size', '0.25',
            '--disable-loo'
        ])
        
        assert args.command == 'analysis'
        assert args.analysis_command == 'roc-run'
        assert args.experiment_id == 1
        assert args.name == 'Test Analysis'
        assert args.prevalence == 0.3
        assert args.enable_cv is True
        assert args.bootstrap_iterations == 150
        assert args.bootstrap_validation_size == 0.25
        assert args.disable_loo is True
    
    def test_cli_default_cv_parameters(self):
        """Test CLI default cross-validation parameters."""
        import argparse
        from src.mmkkb.cli.analysis_commands import AnalysisCommandHandler
        
        handler = AnalysisCommandHandler()
        
        # Create parser
        main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers(dest='command')
        handler.add_commands(subparsers)
        
        # Test ROC analysis with minimal parameters
        args = main_parser.parse_args([
            'analysis', 'roc-run', '1', 'Test Analysis', '0.3'
        ])
        
        assert args.bootstrap_iterations == 200  # Default
        assert args.bootstrap_validation_size == 0.2  # Default
        assert args.enable_cv is False  # Default
        assert args.disable_loo is False  # Default
        assert args.disable_bootstrap is False  # Default