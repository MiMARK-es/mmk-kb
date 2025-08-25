#!/usr/bin/env python3
"""
Demonstration script for the new cross-validation features in MMK-KB.
This script shows how to use the enhanced analysis capabilities.
"""
import os
import sys
import tempfile
import pandas as pd
import numpy as np

# Add the src directory to the path so we can import mmkkb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mmkkb.projects import ProjectDatabase, Project
from mmkkb.samples import SampleDatabase, Sample  
from mmkkb.experiments import ExperimentDatabase, Experiment, Biomarker, BiomarkerVersion, Measurement
from mmkkb.analyses.roc_analysis import ROCAnalyzer, ROCAnalysis
from mmkkb.analyses.roc_normalized_analysis import ROCNormalizedAnalyzer, ROCNormalizedAnalysis
from mmkkb.analyses.base_analysis import CrossValidationConfig

def create_demo_data(db_path):
    """Create demo data for testing cross-validation features."""
    print("🔄 Creating demo data...")
    
    # Initialize databases
    proj_db = ProjectDatabase(db_path)
    sample_db = SampleDatabase(db_path)
    exp_db = ExperimentDatabase(db_path)
    
    # Create project
    project = Project(
        code="CV_DEMO",
        name="Cross-Validation Demo",
        description="Demonstration of cross-validation features",
        creator="Demo Script"
    )
    created_project = proj_db.create_project(project)
    
    # Create samples with balanced classes
    samples = []
    for i in range(50):
        sample = Sample(
            project_id=created_project.id,
            code=f"SAMPLE_{i:03d}",
            dx=i % 2,  # Alternating 0/1 for balanced classes
            age=30 + (i % 40),
            bmi=22.0 + (i % 10),
            dx_origin="clinical",
            collection_center="Demo Hospital",
            processing_time=120
        )
        samples.append(sample_db.create_sample(sample))
    
    # Create experiment
    experiment = Experiment(
        project_id=created_project.id,
        name="Biomarker Panel",
        description="Demo biomarker panel for CV testing"
    )
    created_experiment = exp_db.create_experiment(experiment)
    
    # Create biomarkers
    biomarkers = []
    biomarker_names = ["Protein_A", "Protein_B", "Protein_C", "Total_Protein"]
    
    for name in biomarker_names:
        biomarker = Biomarker(name=name, description=f"Demo {name}")
        created_biomarker = exp_db.create_biomarker(biomarker)
        
        biomarker_version = BiomarkerVersion(
            biomarker_id=created_biomarker.id,
            version="v1.0",
            description=f"Demo assay for {name}"
        )
        created_bv = exp_db.create_biomarker_version(biomarker_version)
        biomarkers.append(created_bv)
    
    # Generate synthetic measurements
    np.random.seed(42)  # For reproducible results
    
    measurements = []
    for sample in samples:
        for bv in biomarkers:
            # Create realistic biomarker data with some signal
            if bv.id == biomarkers[-1].id:  # Total_Protein (normalizer)
                value = 100 + np.random.normal(0, 10)  # Base level around 100
            else:
                # Other proteins - add some disease signal
                base_value = 50 + np.random.normal(0, 15)
                if sample.dx == 1:  # Disease samples
                    base_value += 20 + np.random.normal(0, 10)  # Higher in disease
                value = max(0.1, base_value)  # Ensure positive values
            
            measurement = Measurement(
                experiment_id=created_experiment.id,
                sample_id=sample.id,
                biomarker_version_id=bv.id,
                value=value
            )
            measurements.append(exp_db.create_measurement(measurement))
    
    print(f"✅ Created demo data:")
    print(f"   - Project: {created_project.code}")
    print(f"   - Samples: {len(samples)} (balanced classes)")
    print(f"   - Biomarkers: {len(biomarkers)}")
    print(f"   - Measurements: {len(measurements)}")
    
    return created_experiment.id, biomarkers[-1].id  # Return experiment ID and normalizer ID

def demo_roc_analysis_with_cv(db_path, experiment_id):
    """Demonstrate ROC analysis with cross-validation."""
    print("\n🔄 Running ROC Analysis with Cross-Validation...")
    
    analyzer = ROCAnalyzer(db_path)
    
    # Configure cross-validation
    cv_config = CrossValidationConfig(
        enable_loo=True,
        enable_bootstrap=True,
        bootstrap_iterations=100,  # Reduced for demo speed
        bootstrap_validation_size=0.25
    )
    
    # Create analysis
    analysis = ROCAnalysis(
        name="Demo CV Analysis",
        description="ROC analysis with cross-validation demonstration",
        experiment_id=experiment_id,
        prevalence=0.3,
        max_combination_size=2,
        cross_validation_config=cv_config
    )
    
    # Run analysis
    results = analyzer.run_roc_analysis(analysis)
    
    print(f"✅ ROC Analysis completed:")
    print(f"   - Analysis ID: {results['analysis_id']}")
    print(f"   - Models created: {results['models_created']}")
    print(f"   - Failed models: {len(results['failed_models'])}")
    
    # Show top models with CV results
    models = analyzer.roc_db.get_roc_models_by_analysis(results['analysis_id'])
    print(f"\n📊 Top 5 models by AUC:")
    for i, model in enumerate(models[:5], 1):
        cv_summary = ""
        if model.cross_validation_results:
            if model.cross_validation_results.get('loo_auc_mean'):
                cv_summary += f" LOO: {model.cross_validation_results['loo_auc_mean']:.3f}±{model.cross_validation_results.get('loo_auc_std', 0):.3f}"
            if model.cross_validation_results.get('bootstrap_auc_mean'):
                cv_summary += f" Bootstrap: {model.cross_validation_results['bootstrap_auc_mean']:.3f}±{model.cross_validation_results.get('bootstrap_auc_std', 0):.3f}"
        
        print(f"   {i}. Model {model.id}: AUC = {model.auc:.3f} [CV:{cv_summary}]")
    
    return results['analysis_id']

def demo_roc_normalized_analysis_with_cv(db_path, experiment_id, normalizer_id):
    """Demonstrate ROC normalized analysis with cross-validation."""
    print("\n🔄 Running ROC Normalized Analysis with Cross-Validation...")
    
    analyzer = ROCNormalizedAnalyzer(db_path)
    
    # Configure cross-validation (Bootstrap only for speed)
    cv_config = CrossValidationConfig(
        enable_loo=False,
        enable_bootstrap=True,
        bootstrap_iterations=50,  # Reduced for demo speed
        bootstrap_validation_size=0.2
    )
    
    # Create analysis
    analysis = ROCNormalizedAnalysis(
        name="Demo Normalized CV Analysis",
        description="ROC normalized analysis with cross-validation demonstration",
        experiment_id=experiment_id,
        normalizer_biomarker_version_id=normalizer_id,
        prevalence=0.3,
        max_combination_size=2,
        cross_validation_config=cv_config
    )
    
    # Run analysis
    results = analyzer.run_roc_normalized_analysis(analysis)
    
    print(f"✅ ROC Normalized Analysis completed:")
    print(f"   - Analysis ID: {results['analysis_id']}")
    print(f"   - Normalizer: {results['normalizer_biomarker']}")
    print(f"   - Models created: {results['models_created']}")
    print(f"   - Failed models: {len(results['failed_models'])}")
    
    # Show top models with CV results
    models = analyzer.roc_norm_db.get_roc_normalized_models_by_analysis(results['analysis_id'])
    print(f"\n📊 Top 3 normalized models by AUC:")
    for i, model in enumerate(models[:3], 1):
        cv_summary = ""
        if model.cross_validation_results and model.cross_validation_results.get('bootstrap_auc_mean'):
            cv_summary = f" Bootstrap: {model.cross_validation_results['bootstrap_auc_mean']:.3f}±{model.cross_validation_results.get('bootstrap_auc_std', 0):.3f}"
        
        print(f"   {i}. Model {model.id}: AUC = {model.auc:.3f} [CV:{cv_summary}]")
    
    return results['analysis_id']

def demo_cli_usage():
    """Show CLI usage examples."""
    print("\n📋 CLI Usage Examples:")
    print("=" * 50)
    
    print("\n# Standard ROC analysis with cross-validation:")
    print("mmk-kb analysis roc-run 1 'CV Analysis' 0.3 --enable-cv")
    
    print("\n# ROC analysis with custom bootstrap parameters:")
    print("mmk-kb analysis roc-run 1 'Custom Bootstrap' 0.3 --enable-cv \\")
    print("  --disable-loo --bootstrap-iterations 500 --bootstrap-validation-size 0.25")
    
    print("\n# ROC normalized analysis with cross-validation:")
    print("mmk-kb analysis roc-norm-run 1 4 'Normalized CV' 0.3 --enable-cv")
    
    print("\n# View analysis results:")
    print("mmk-kb analysis roc-list")
    print("mmk-kb analysis roc-show 1")
    print("mmk-kb analysis roc-report 1 --output results.csv")

def main():
    """Main demonstration function."""
    print("🚀 MMK-KB Cross-Validation Demo")
    print("=" * 40)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create demo data
        experiment_id, normalizer_id = create_demo_data(db_path)
        
        # Demo ROC analysis with CV
        roc_analysis_id = demo_roc_analysis_with_cv(db_path, experiment_id)
        
        # Demo ROC normalized analysis with CV
        norm_analysis_id = demo_roc_normalized_analysis_with_cv(db_path, experiment_id, normalizer_id)
        
        # Show CLI usage
        demo_cli_usage()
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Key features demonstrated:")
        print(f"   - Cross-validation with LOO and Bootstrap methods")
        print(f"   - Modular analysis structure with BaseAnalyzer")
        print(f"   - Enhanced CLI with grouped commands")
        print(f"   - Database storage of CV configurations and results")
        print(f"   - Both ROC and ROC normalized analyses with CV")
        
    finally:
        # Clean up temporary database
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"\n🧹 Cleaned up temporary database")

if __name__ == "__main__":
    main()