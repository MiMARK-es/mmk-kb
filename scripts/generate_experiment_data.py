#!/usr/bin/env python3
"""
Generate realistic experiment datasets for MMK Knowledge Base testing.
Creates CSV files with biomarker measurements suitable for AUC analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def generate_biomarker_experiment_1():
    """Generate first biomarker experiment with realistic case/control distributions."""
    np.random.seed(42)  # For reproducible results
    
    # Create sample codes (mix of cases and controls)
    cases = [f"CASE_{i:03d}" for i in range(1, 26)]  # 25 cases
    controls = [f"CTRL_{i:03d}" for i in range(1, 26)]  # 25 controls
    samples = cases + controls
    
    # Define biomarkers with different effect sizes for AUC analysis
    biomarkers = {
        'Biomarker_A': {'control_mean': 2.1, 'control_std': 0.8, 'case_mean': 4.2, 'case_std': 1.2},  # High discriminatory power
        'Biomarker_B': {'control_mean': 3.5, 'control_std': 1.0, 'case_mean': 5.8, 'case_std': 1.5},  # Good discriminatory power
        'Biomarker_C': {'control_mean': 1.8, 'control_std': 0.6, 'case_mean': 2.9, 'case_std': 0.9},  # Moderate discriminatory power
        'Biomarker_D': {'control_mean': 2.3, 'control_std': 0.7, 'case_mean': 2.6, 'case_std': 0.8},  # Low discriminatory power
        'Biomarker_E': {'control_mean': 1.5, 'control_std': 0.5, 'case_mean': 3.1, 'case_std': 1.1},  # High discriminatory power
    }
    
    # Generate data
    data = {'sample': samples}
    
    for biomarker, params in biomarkers.items():
        # Controls (first 25 samples)
        control_values = np.random.normal(
            params['control_mean'], 
            params['control_std'], 
            25
        )
        # Cases (last 25 samples) 
        case_values = np.random.normal(
            params['case_mean'], 
            params['case_std'], 
            25
        )
        
        # Ensure all values are positive
        control_values = np.abs(control_values)
        case_values = np.abs(case_values)
        
        values = np.concatenate([control_values, case_values])
        data[biomarker] = np.round(values, 2)
    
    return pd.DataFrame(data)


def generate_biomarker_experiment_2():
    """Generate second biomarker experiment with different sample set."""
    np.random.seed(123)
    
    # Create sample codes with some overlap and some new samples
    cases = [f"CASE_{i:03d}" for i in range(10, 31)]  # 21 cases (some overlap with exp1)
    controls = [f"CTRL_{i:03d}" for i in range(10, 30)]  # 20 controls (some overlap with exp1)
    samples = cases + controls
    
    # Define different biomarkers for this experiment
    biomarkers = {
        'Biomarker_F': {'control_mean': 150.5, 'control_std': 25.0, 'case_mean': 95.2, 'case_std': 20.0},
        'Biomarker_G': {'control_mean': 12.3, 'control_std': 4.2, 'case_mean': 28.9, 'case_std': 8.5},
        'Biomarker_H': {'control_mean': 8.5, 'control_std': 2.1, 'case_mean': 22.3, 'case_std': 6.2},
        'Biomarker_A': {'control_mean': 2.0, 'control_std': 0.9, 'case_mean': 4.5, 'case_std': 1.3},  # Same biomarker, different version
    }
    
    # Generate data
    data = {'sample': samples}
    
    for biomarker, params in biomarkers.items():
        # Controls
        control_values = np.random.normal(
            params['control_mean'], 
            params['control_std'], 
            20
        )
        # Cases
        case_values = np.random.normal(
            params['case_mean'], 
            params['case_std'], 
            21
        )
        
        # Ensure positive values
        control_values = np.abs(control_values)
        case_values = np.abs(case_values)
        
        values = np.concatenate([control_values, case_values])
        data[biomarker] = np.round(values, 1)
    
    return pd.DataFrame(data)


def generate_biomarker_experiment_3():
    """Generate third biomarker experiment focusing on high-discrimination biomarkers."""
    np.random.seed(456)
    
    # Create sample codes
    cases = [f"CASE_{i:03d}" for i in range(1, 16)]  # 15 cases
    controls = [f"CTRL_{i:03d}" for i in range(1, 16)]  # 15 controls
    samples = cases + controls
    
    # Define biomarkers with clinical-like performance
    biomarkers = {
        'Biomarker_I': {'control_mean': 0.02, 'control_std': 0.01, 'case_mean': 0.85, 'case_std': 0.45},  # Very high discrimination
        'Biomarker_J': {'control_mean': 85.3, 'control_std': 25.8, 'case_mean': 1250.7, 'case_std': 680.2},  # High discrimination
        'Biomarker_K': {'control_mean': 1.8, 'control_std': 0.6, 'case_mean': 12.5, 'case_std': 4.2},  # Good discrimination
        'Biomarker_B': {'control_mean': 3.2, 'control_std': 1.1, 'case_mean': 6.1, 'case_std': 1.6},  # Same biomarker as exp1, different version
    }
    
    # Generate data
    data = {'sample': samples}
    
    for biomarker, params in biomarkers.items():
        # Controls
        control_values = np.random.normal(
            params['control_mean'], 
            params['control_std'], 
            15
        )
        # Cases
        case_values = np.random.normal(
            params['case_mean'], 
            params['case_std'], 
            15
        )
        
        # Ensure positive values
        control_values = np.abs(control_values)
        case_values = np.abs(case_values)
        
        values = np.concatenate([control_values, case_values])
        data[biomarker] = np.round(values, 3)
    
    return pd.DataFrame(data)


def create_sample_info_file():
    """Create a file documenting the sample codes and their case/control status."""
    sample_info = []
    
    # Get all unique sample codes from all experiments
    all_samples = set()
    
    # From experiment 1
    for i in range(1, 26):
        all_samples.add(f'CASE_{i:03d}')
        all_samples.add(f'CTRL_{i:03d}')
    
    # From experiment 2 (overlapping range)
    for i in range(10, 31):
        all_samples.add(f'CASE_{i:03d}')
    for i in range(10, 30):
        all_samples.add(f'CTRL_{i:03d}')
    
    # Convert to sorted list and create info
    for sample_code in sorted(all_samples):
        if sample_code.startswith('CASE_'):
            sample_info.append({
                'sample_code': sample_code,
                'diagnosis': 'case',
                'description': 'Disease case sample'
            })
        else:
            sample_info.append({
                'sample_code': sample_code,
                'diagnosis': 'control',
                'description': 'Healthy control sample'
            })
    
    return pd.DataFrame(sample_info)


def main():
    """Generate all sample datasets."""
    # Create experiment data directory structure
    base_dir = Path("experiment_data")
    example_dir = base_dir / "example_data"
    example_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment datasets
    exp1_data = generate_biomarker_experiment_1()
    exp1_data.to_csv(example_dir / "biomarker_study_1.csv", index=False)
    print(f"✅ Generated experiment 1: {len(exp1_data)} samples, {len(exp1_data.columns)-1} biomarkers")
    
    exp2_data = generate_biomarker_experiment_2()
    exp2_data.to_csv(example_dir / "biomarker_study_2.csv", index=False)
    print(f"✅ Generated experiment 2: {len(exp2_data)} samples, {len(exp2_data.columns)-1} biomarkers")
    
    exp3_data = generate_biomarker_experiment_3()
    exp3_data.to_csv(example_dir / "biomarker_study_3.csv", index=False)
    print(f"✅ Generated experiment 3: {len(exp3_data)} samples, {len(exp3_data.columns)-1} biomarkers")
    
    # Generate sample information file
    sample_info = create_sample_info_file()
    sample_info.to_csv(base_dir / "sample_information.csv", index=False)
    print(f"✅ Generated sample information file: {len(sample_info)} sample records")
    
    # Create README file
    readme_content = """# Experiment Data Directory

This directory contains example datasets for testing the MMK Knowledge Base experiment functionality.

## Dataset Overview

### Example Data Folder (example_data/)

Contains three biomarker experiments with realistic case/control distributions suitable for AUC analysis:

#### 1. biomarker_study_1.csv
- **Samples**: 50 (25 cases: CASE_001-CASE_025, 25 controls: CTRL_001-CTRL_025)
- **Biomarkers**: 5 biomarkers (Biomarker_A through Biomarker_E)
- **Expected AUCs**: Range from 0.55 to 0.85 (different discriminatory powers)

#### 2. biomarker_study_2.csv  
- **Samples**: 41 (21 cases: CASE_010-CASE_030, 20 controls: CTRL_010-CTRL_029)
- **Biomarkers**: 4 biomarkers (Biomarker_F, G, H, and Biomarker_A)
- **Note**: Some sample overlap with study 1, Biomarker_A measured with different version

#### 3. biomarker_study_3.csv
- **Samples**: 30 (15 cases: CASE_001-CASE_015, 15 controls: CTRL_001-CTRL_015)
- **Biomarkers**: 4 biomarkers (Biomarker_I, J, K, and Biomarker_B)
- **Note**: High-discrimination biomarkers, some sample overlap with study 1

## Key Features for Testing

1. **Sample Overlap**: Some samples appear in multiple experiments
2. **Biomarker Versions**: Same biomarkers (A, B) measured in different experiments with different versions
3. **AUC Analysis Ready**: Data generated with realistic case/control distributions
4. **Case/Control Labels**: Clear sample naming convention for outcome analysis

## Sample Code Convention

- **CASE_XXX**: Disease case samples
- **CTRL_XXX**: Healthy control samples
- Sample numbers range from 001 to 030 across all experiments

## Data Generation Details

- Controls: Lower biomarker concentrations with normal biological variation
- Cases: Elevated biomarker concentrations with disease-related variation  
- Different effect sizes to simulate various diagnostic performances
- Positive values ensured for all biomarker measurements

## Usage with MMK-KB

1. Create a project for biomarker studies
2. Add samples using codes from sample_information.csv
3. Upload experiments using the CSV files with different biomarker versions:
   - Study 1: version "RUO" 
   - Study 2: version "proprietary"
   - Study 3: version "clinical"
4. Analyze biomarker performance across experiments and versions
"""
    
    with open(example_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Generated README.md with dataset documentation")
    print(f"\n📁 All experiment data saved to: {base_dir.absolute()}")
    print(f"📁 Example datasets in: {example_dir.absolute()}")


if __name__ == "__main__":
    main()