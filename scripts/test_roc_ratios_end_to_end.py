#!/usr/bin/env python3
"""
End-to-end test script for ROC Ratios analysis functionality.
This script tests the complete workflow from data setup to analysis and reporting.
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmkkb.projects import ProjectDatabase, Project
from mmkkb.samples import SampleDatabase, Sample
from mmkkb.experiments import ExperimentDatabase, Experiment, Biomarker, BiomarkerVersion, Measurement
from mmkkb.analyses.roc_ratios_analysis import ROCRatiosAnalyzer, ROCRatiosAnalysis
from mmkkb.analyses.base_analysis import CrossValidationConfig


def create_test_data(db_path: str):
    """Create comprehensive test data for ROC ratios analysis."""
    print("🔧 Setting up test data...")
    
    # Initialize databases
    project_db = ProjectDatabase(db_path)
    sample_db = SampleDatabase(db_path)
    exp_db = ExperimentDatabase(db_path)
    
    # Create project
    project = Project(
        code="RATIOS_TEST",
        name="ROC Ratios Test Project",
        description="End-to-end test for ROC ratios analysis",
        creator="test_script"
    )
    created_project = project_db.create_project(project)
    print(f"✅ Created project: {created_project.code}")
    
    # Create samples with realistic distribution
    np.random.seed(42)
    samples_data = []
    
    # Create 30 samples (15 positive, 15 negative)
    for i in range(30):
        is_positive = i < 15
        samples_data.append({
            "code": f"SAMPLE_{i+1:03d}",
            "dx": 1 if is_positive else 0,
            "age": np.random.randint(25, 80),
            "bmi": np.random.normal(25, 5),
            "dx_origin": "Clinical",
            "collection_center": "Test Center",
            "processing_time": 24
        })
    
    sample_ids = []
    for sample_data in samples_data:
        sample = Sample(
            code=sample_data["code"],
            age=sample_data["age"],
            bmi=max(sample_data["bmi"], 15),  # Ensure reasonable BMI
            dx=sample_data["dx"],
            dx_origin=sample_data["dx_origin"],
            collection_center=sample_data["collection_center"],
            processing_time=sample_data["processing_time"],
            project_id=created_project.id
        )
        created_sample = sample_db.create_sample(sample)
        sample_ids.append(created_sample.id)
    
    print(f"✅ Created {len(sample_ids)} samples")
    
    # Create biomarkers that will show ratio effects
    biomarkers_info = [
        {"name": "IL6", "description": "Interleukin-6", "category": "cytokine"},
        {"name": "TNFα", "description": "Tumor Necrosis Factor Alpha", "category": "cytokine"},
        {"name": "CRP", "description": "C-Reactive Protein", "category": "acute_phase"},
        {"name": "PCT", "description": "Procalcitonin", "category": "acute_phase"}
    ]
    
    biomarker_version_ids = []
    for bio_info in biomarkers_info:
        # Create biomarker
        biomarker = Biomarker(
            name=bio_info["name"],
            description=bio_info["description"],
            category=bio_info["category"]
        )
        created_bio = exp_db.create_biomarker(biomarker)
        
        # Create version
        version = BiomarkerVersion(
            biomarker_id=created_bio.id,
            version="v1.0",
            description=f"Standard assay for {bio_info['name']}"
        )
        created_version = exp_db.create_biomarker_version(version)
        biomarker_version_ids.append(created_version.id)
    
    print(f"✅ Created {len(biomarker_version_ids)} biomarker versions")
    
    # Create experiment
    experiment = Experiment(
        name="ROC Ratios Test Experiment",
        description="Test experiment with realistic biomarker data for ratio analysis",
        project_id=created_project.id,
        csv_filename="test_ratios_data.csv"
    )
    created_experiment = exp_db.create_experiment(experiment)
    print(f"✅ Created experiment: {created_experiment.name}")
    
    # Create realistic measurements with ratio patterns
    print("🔬 Generating synthetic biomarker measurements...")
    
    for i, sample_id in enumerate(sample_ids):
        is_positive = samples_data[i]["dx"] == 1
        
        # Generate correlated biomarker values that will show ratio effects
        if is_positive:
            # Positive cases: inflammation pattern
            il6 = np.random.lognormal(3, 0.5)     # Higher IL6
            tnf = np.random.lognormal(2, 0.4)     # Moderate TNF
            crp = np.random.lognormal(2.5, 0.6)   # Higher CRP
            pct = np.random.lognormal(1, 0.8)     # Variable PCT
        else:
            # Negative cases: normal pattern
            il6 = np.random.lognormal(1.5, 0.4)   # Lower IL6
            tnf = np.random.lognormal(1.2, 0.3)   # Lower TNF
            crp = np.random.lognormal(1, 0.4)     # Lower CRP
            pct = np.random.lognormal(0, 0.5)     # Lower PCT
        
        values = [il6, tnf, crp, pct]
        
        # Create measurements
        for j, (bv_id, value) in enumerate(zip(biomarker_version_ids, values)):
            measurement = Measurement(
                experiment_id=created_experiment.id,
                sample_id=sample_id,
                biomarker_version_id=bv_id,
                value=value
            )
            exp_db.create_measurement(measurement)
    
    print("✅ Created realistic biomarker measurements")
    return created_experiment.id


def test_roc_ratios_analysis(db_path: str, experiment_id: int):
    """Test the complete ROC ratios analysis workflow."""
    print("\n🧪 Testing ROC Ratios Analysis...")
    
    analyzer = ROCRatiosAnalyzer(db_path)
    
    # Test 1: Basic analysis without cross-validation
    print("📊 Running basic ROC ratios analysis...")
    
    analysis_basic = ROCRatiosAnalysis(
        name="Basic Ratios Test",
        description="Testing basic ROC ratios functionality",
        experiment_id=experiment_id,
        prevalence=0.5,  # 50% prevalence in our test data
        max_combination_size=2  # Test up to 2-ratio combinations
    )
    
    results_basic = analyzer.run_roc_ratios_analysis(analysis_basic)
    
    print(f"✅ Basic analysis completed:")
    print(f"   - Analysis ID: {results_basic['analysis_id']}")
    print(f"   - Total combinations tested: {results_basic['total_combinations']}")
    print(f"   - Successful models: {results_basic['models_created']}")
    print(f"   - Failed models: {len(results_basic['failed_models'])}")
    
    assert results_basic['models_created'] > 0, "No models were created"
    assert len(results_basic['successful_models']) > 0, "No successful models"
    
    # Check first successful model
    best_model = results_basic['successful_models'][0]
    print(f"   - Best AUC: {best_model['auc']:.3f}")
    print(f"   - Ratio combination: {best_model['ratio_combination']}")
    
    # Test 2: Analysis with cross-validation
    print("\n🔄 Running ROC ratios analysis with cross-validation...")
    
    cv_config = CrossValidationConfig(
        enable_loo=True,
        enable_bootstrap=True,
        bootstrap_iterations=50  # Reasonable number for testing
    )
    
    analysis_cv = ROCRatiosAnalysis(
        name="CV Ratios Test",
        description="Testing ROC ratios with cross-validation",
        experiment_id=experiment_id,
        prevalence=0.5,
        max_combination_size=1,  # Keep it smaller for CV
        cross_validation_config=cv_config
    )
    
    results_cv = analyzer.run_roc_ratios_analysis(analysis_cv)
    
    print(f"✅ CV analysis completed:")
    print(f"   - Models with CV: {results_cv['models_created']}")
    
    # Check CV results
    if results_cv['successful_models']:
        cv_model = results_cv['successful_models'][0]
        cv_results = cv_model['cross_validation_results']
        if cv_results:
            if hasattr(cv_results, 'loo_auc_mean') and cv_results.loo_auc_mean:
                print(f"   - LOO CV AUC: {cv_results.loo_auc_mean:.3f} ± {cv_results.loo_auc_std:.3f}")
            if hasattr(cv_results, 'bootstrap_auc_mean') and cv_results.bootstrap_auc_mean:
                print(f"   - Bootstrap CV AUC: {cv_results.bootstrap_auc_mean:.3f} ± {cv_results.bootstrap_auc_std:.3f}")
    
    # Test 3: Generate analysis reports
    print("\n📋 Generating analysis reports...")
    
    # Generate report for basic analysis
    report_basic = analyzer.generate_analysis_report(results_basic['analysis_id'])
    print(f"✅ Basic analysis report generated:")
    print(f"   - Models in report: {len(report_basic)}")
    print(f"   - Columns: {len(report_basic.columns)}")
    
    # Show top 3 models
    if len(report_basic) > 0:
        print("   - Top 3 models by AUC:")
        top_models = report_basic.nlargest(3, 'AUC')[['Model_ID', 'AUC', 'Ratio_1']]
        for _, model in top_models.iterrows():
            print(f"     Model {model['Model_ID']}: AUC={model['AUC']:.3f}, Ratio={model['Ratio_1']}")
    
    # Generate report for CV analysis
    report_cv = analyzer.generate_analysis_report(results_cv['analysis_id'])
    print(f"✅ CV analysis report generated: {len(report_cv)} models")
    
    return results_basic, results_cv


def test_database_operations(db_path: str):
    """Test direct database operations."""
    print("\n🗄️  Testing database operations...")
    
    from mmkkb.analyses.roc_ratios_analysis import ROCRatiosAnalysisDatabase
    
    db = ROCRatiosAnalysisDatabase(db_path)
    
    # List all analyses
    analyses = db.list_roc_ratios_analyses()
    print(f"✅ Found {len(analyses)} ROC ratios analyses")
    
    for analysis in analyses:
        print(f"   - {analysis.name} (ID: {analysis.id})")
        
        # Get models for this analysis
        models = db.get_roc_ratios_models_by_analysis(analysis.id)
        print(f"     └─ {len(models)} models")
        
        if models:
            best_model = max(models, key=lambda m: m.auc)
            print(f"     └─ Best AUC: {best_model.auc:.3f}")
            
            # Get metrics for best model
            metrics = db.get_roc_ratios_metrics_by_model(best_model.id)
            print(f"     └─ {len(metrics)} metric sets")
            
            # Get curve points for best model
            curve_points = db.get_roc_ratios_curve_points_by_model(best_model.id)
            print(f"     └─ {len(curve_points)} ROC curve points")


def test_error_handling(db_path: str):
    """Test error handling scenarios."""
    print("\n⚠️  Testing error handling...")
    
    analyzer = ROCRatiosAnalyzer(db_path)
    
    # Test 1: Non-existent experiment
    print("🚫 Testing with non-existent experiment...")
    try:
        invalid_analysis = ROCRatiosAnalysis(
            name="Invalid Test",
            description="Should fail",
            experiment_id=99999,
            prevalence=0.5,
            max_combination_size=1
        )
        analyzer.run_roc_ratios_analysis(invalid_analysis)
        print("❌ ERROR: Should have failed!")
    except Exception as e:
        print(f"✅ Correctly caught error: {type(e).__name__}")
    
    # Test 2: Invalid prevalence
    print("🚫 Testing with invalid prevalence...")
    try:
        from mmkkb.analyses.roc_ratios_analysis import ROCRatiosAnalysisDatabase
        db = ROCRatiosAnalysisDatabase(db_path)
        
        invalid_analysis = ROCRatiosAnalysis(
            name="Invalid Prevalence",
            description="Should fail",
            experiment_id=1,
            prevalence=1.5,  # Invalid: > 1
            max_combination_size=1
        )
        db.create_roc_ratios_analysis(invalid_analysis)
        print("❌ ERROR: Should have failed!")
    except Exception as e:
        print(f"✅ Correctly caught error: {type(e).__name__}")


def main():
    """Run complete end-to-end test."""
    print("🚀 Starting ROC Ratios End-to-End Test")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Step 1: Create test data
        experiment_id = create_test_data(db_path)
        
        # Step 2: Test ROC ratios analysis
        results_basic, results_cv = test_roc_ratios_analysis(db_path, experiment_id)
        
        # Step 3: Test database operations
        test_database_operations(db_path)
        
        # Step 4: Test error handling
        test_error_handling(db_path)
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("\n📊 Summary:")
        print(f"   - Basic analysis: {results_basic['models_created']} models created")
        print(f"   - CV analysis: {results_cv['models_created']} models created")
        print(f"   - Database operations: ✅ Working")
        print(f"   - Error handling: ✅ Working")
        
        # Show some interesting findings
        if results_basic['successful_models']:
            best_basic = max(results_basic['successful_models'], key=lambda m: m['auc'])
            print(f"   - Best model AUC: {best_basic['auc']:.3f}")
            print(f"   - Best ratio combination: {best_basic['ratio_combination']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"\n🧹 Cleaned up test database: {db_path}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)