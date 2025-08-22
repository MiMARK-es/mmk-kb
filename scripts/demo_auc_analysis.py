#!/usr/bin/env python3
"""
Demonstration of AUC analysis using the MMK Knowledge Base biomarker structure.
Shows how the unified biomarker system makes bioinformatics analysis efficient.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from mmkkb.experiments import ExperimentDatabase
from mmkkb.samples import SampleDatabase
from mmkkb.config import set_environment, Environment


def calculate_biomarker_auc(biomarker_id: int, db_path: str = None):
    """Calculate AUC for a biomarker using all available measurements across versions."""
    set_environment(Environment.TESTING)
    
    experiment_db = ExperimentDatabase(db_path)
    sample_db = SampleDatabase(db_path)
    
    # Get all data for this biomarker across all versions and experiments
    analysis_data = experiment_db.get_biomarker_data_for_analysis(biomarker_id)
    
    if not analysis_data or not analysis_data['measurements']:
        return None, f"No measurements found for biomarker ID {biomarker_id}"
    
    biomarker = analysis_data['biomarker']
    measurements = analysis_data['measurements']
    
    # Prepare data for AUC calculation
    values = []
    labels = []
    sample_info = []
    
    for measurement in measurements:
        # Get sample information to determine case/control status
        sample = sample_db.get_sample_by_id(measurement['sample_id'])
        if sample:
            values.append(measurement['value'])
            labels.append(1 if sample.dx else 0)  # 1 for case, 0 for control
            sample_info.append({
                'sample_code': measurement['sample_code'],
                'value': measurement['value'],
                'label': 1 if sample.dx else 0,
                'version': measurement['version'],
                'experiment': measurement['experiment_name']
            })
    
    if len(set(labels)) < 2:
        return None, f"Need both cases and controls for AUC calculation"
    
    # Calculate AUC
    auc = roc_auc_score(labels, values)
    
    # Create results summary
    results = {
        'biomarker_name': biomarker.name,
        'biomarker_id': biomarker_id,
        'auc': auc,
        'total_measurements': len(values),
        'cases': sum(labels),
        'controls': len(labels) - sum(labels),
        'versions_used': analysis_data['versions_used'],
        'experiments': analysis_data['unique_experiments'],
        'mean_case_value': np.mean([v for v, l in zip(values, labels) if l == 1]),
        'mean_control_value': np.mean([v for v, l in zip(values, labels) if l == 0]),
        'sample_data': sample_info
    }
    
    return results, None


def run_auc_analysis_demo():
    """Run AUC analysis on all biomarkers to demonstrate the system."""
    set_environment(Environment.TESTING)
    
    experiment_db = ExperimentDatabase()
    biomarkers = experiment_db.list_biomarkers()
    
    print("🧬 AUC Analysis Demo: Biomarker Performance Across All Versions")
    print("=" * 70)
    
    results_summary = []
    
    for biomarker in biomarkers:
        results, error = calculate_biomarker_auc(biomarker.id)
        
        if error:
            print(f"⚠️  {biomarker.name}: {error}")
            continue
        
        results_summary.append(results)
        
        print(f"\n📊 {results['biomarker_name']} (ID: {results['biomarker_id']})")
        print(f"   AUC: {results['auc']:.3f}")
        print(f"   Measurements: {results['total_measurements']} ({results['cases']} cases, {results['controls']} controls)")
        print(f"   Versions: {', '.join(results['versions_used'])}")
        print(f"   Experiments: {results['experiments']}")
        print(f"   Mean Values: Cases={results['mean_case_value']:.2f}, Controls={results['mean_control_value']:.2f}")
        
        # Show performance interpretation
        if results['auc'] >= 0.8:
            performance = "🟢 Excellent"
        elif results['auc'] >= 0.7:
            performance = "🟡 Good"
        elif results['auc'] >= 0.6:
            performance = "🟠 Fair"
        else:
            performance = "🔴 Poor"
        
        print(f"   Performance: {performance}")
    
    # Summary table
    print(f"\n📋 AUC Summary Table:")
    print(f"{'Biomarker':<15} {'AUC':<6} {'N':<4} {'Versions':<12} {'Performance':<12}")
    print("-" * 65)
    
    # Sort by AUC descending
    results_summary.sort(key=lambda x: x['auc'], reverse=True)
    
    for result in results_summary:
        versions_str = ','.join(result['versions_used'])[:11]
        if result['auc'] >= 0.8:
            perf = "Excellent"
        elif result['auc'] >= 0.7:
            perf = "Good"  
        elif result['auc'] >= 0.6:
            perf = "Fair"
        else:
            perf = "Poor"
        
        print(f"{result['biomarker_name']:<15} {result['auc']:<6.3f} {result['total_measurements']:<4} {versions_str:<12} {perf:<12}")
    
    print(f"\n✅ Analysis completed for {len(results_summary)} biomarkers")
    print(f"💡 Key Insight: Each biomarker was analyzed using ALL available measurements")
    print(f"   across different versions and experiments automatically!")


if __name__ == "__main__":
    run_auc_analysis_demo()