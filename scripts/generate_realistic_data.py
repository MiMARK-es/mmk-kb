#!/usr/bin/env python3
"""
Generate realistic biomarker data for ROC analysis testing.
Creates data with varying discriminative power to demonstrate ROC analysis features.
"""

import numpy as np
import pandas as pd
import os

def generate_realistic_biomarker_data():
    """Generate biomarker data with realistic AUC distributions."""
    
    np.random.seed(42)  # For reproducible results
    
    # Create sample codes
    case_samples = [f"CASE_{i:03d}" for i in range(1, 51)]  # 50 cases
    ctrl_samples = [f"CTRL_{i:03d}" for i in range(1, 51)]  # 50 controls
    all_samples = case_samples + ctrl_samples
    
    # Define biomarkers with different discriminative powers
    biomarkers = {
        'IL6': {
            'case_mean': 15.2, 'case_std': 4.8,
            'ctrl_mean': 8.1, 'ctrl_std': 2.9,
            'description': 'Excellent discriminator - AUC ~0.85-0.90'
        },
        'TNFa': {
            'case_mean': 12.7, 'case_std': 5.2,
            'ctrl_mean': 7.3, 'ctrl_std': 3.1,
            'description': 'Good discriminator - AUC ~0.75-0.80'
        },
        'IL1b': {
            'case_mean': 9.8, 'case_std': 4.1,
            'ctrl_mean': 6.2, 'ctrl_std': 3.7,
            'description': 'Fair discriminator - AUC ~0.65-0.70'
        },
        'IL10': {
            'case_mean': 5.4, 'case_std': 3.2,
            'ctrl_mean': 4.1, 'ctrl_std': 2.8,
            'description': 'Poor discriminator - AUC ~0.55-0.60'
        },
        'CRP': {
            'case_mean': 8.9, 'case_std': 2.7,
            'ctrl_mean': 3.2, 'ctrl_std': 1.8,
            'description': 'Very good discriminator - AUC ~0.80-0.85'
        },
        'IFNg': {
            'case_mean': 7.2, 'case_std': 4.9,
            'ctrl_mean': 6.8, 'ctrl_std': 4.2,
            'description': 'Very poor discriminator - AUC ~0.50-0.55'
        }
    }
    
    # Generate data
    data = {'sample': all_samples}
    
    for biomarker, params in biomarkers.items():
        values = []
        
        # Generate case values
        case_values = np.random.normal(
            params['case_mean'], 
            params['case_std'], 
            50
        )
        # Ensure positive values
        case_values = np.maximum(case_values, 0.1)
        
        # Generate control values  
        ctrl_values = np.random.normal(
            params['ctrl_mean'], 
            params['ctrl_std'], 
            50
        )
        # Ensure positive values
        ctrl_values = np.maximum(ctrl_values, 0.1)
        
        # Combine and round to 2 decimal places
        all_values = np.concatenate([case_values, ctrl_values])
        data[biomarker] = np.round(all_values, 2)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df, biomarkers

def generate_sample_metadata():
    """Generate sample metadata with clinical information."""
    
    np.random.seed(42)
    
    samples = []
    
    # Cases
    for i in range(1, 51):
        sample = {
            'code': f"CASE_{i:03d}",
            'age': np.random.randint(45, 85),
            'bmi': np.round(np.random.normal(27.5, 4.2), 1),
            'dx': 1,  # disease
            'dx_origin': np.random.choice(['biopsy', 'pathology', 'surgical']),
            'collection_center': np.random.choice(['Hospital_A', 'Hospital_B', 'Hospital_C']),
            'processing_time': np.random.randint(90, 180)
        }
        samples.append(sample)
    
    # Controls
    for i in range(1, 51):
        sample = {
            'code': f"CTRL_{i:03d}",
            'age': np.random.randint(35, 75),
            'bmi': np.round(np.random.normal(25.2, 3.8), 1),
            'dx': 0,  # benign
            'dx_origin': 'screening',
            'collection_center': np.random.choice(['Hospital_A', 'Hospital_B', 'Hospital_C']),
            'processing_time': np.random.randint(60, 120)
        }
        samples.append(sample)
    
    return pd.DataFrame(samples)

def main():
    """Generate and save realistic test data."""
    
    # Create output directory
    output_dir = "experiment_data/realistic_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🧬 Generating realistic biomarker data...")
    
    # Generate biomarker data
    biomarker_df, biomarker_info = generate_realistic_biomarker_data()
    biomarker_file = os.path.join(output_dir, "realistic_biomarker_study.csv")
    biomarker_df.to_csv(biomarker_file, index=False)
    
    print(f"✅ Biomarker data saved to: {biomarker_file}")
    print(f"   - Samples: {len(biomarker_df)}")
    print(f"   - Biomarkers: {len(biomarker_df.columns) - 1}")
    
    # Generate sample metadata
    sample_df = generate_sample_metadata()
    sample_file = os.path.join(output_dir, "realistic_samples.csv")
    sample_df.to_csv(sample_file, index=False)
    
    print(f"✅ Sample data saved to: {sample_file}")
    print(f"   - Cases: {len(sample_df[sample_df['dx'] == 1])}")
    print(f"   - Controls: {len(sample_df[sample_df['dx'] == 0])}")
    
    # Create README with biomarker descriptions
    readme_content = f"""# Realistic Biomarker Test Data

This dataset contains realistic biomarker measurements designed to demonstrate the ROC analysis functionality with meaningful AUC distributions.

## Dataset Characteristics

- **Samples**: 100 total (50 cases, 50 controls)
- **Biomarkers**: {len(biomarker_info)}
- **Design**: Simulated inflammatory biomarkers with varying discriminative power

## Biomarker Profiles

"""
    
    for biomarker, params in biomarker_info.items():
        readme_content += f"""### {biomarker}
- **Cases**: μ={params['case_mean']}, σ={params['case_std']}
- **Controls**: μ={params['ctrl_mean']}, σ={params['ctrl_std']}
- **Expected Performance**: {params['description']}

"""
    
    readme_content += """## Usage

```bash
# Create project and upload data
mmk-kb create "REALISTIC_TEST" "Realistic ROC Analysis" "Test with meaningful AUCs" "ROC Developer"
mmk-kb use "REALISTIC_TEST"

# Upload sample metadata
mmk-kb sample-upload experiment_data/realistic_data/realistic_samples.csv

# Upload biomarker data
mmk-kb experiment-upload experiment_data/realistic_data/realistic_biomarker_study.csv \\
  "Realistic Biomarker Study" "Multi-biomarker panel with varying discriminative power"

# Run ROC analysis
mmk-kb roc-run 1 "Realistic ROC Analysis" 0.3 --max-combinations 3 \\
  --description "Analysis with realistic biomarker performance"

# View results
mmk-kb roc-report 1 --top 15
```

## Expected Results

This dataset should produce:
- **High-performing models**: IL6, CRP-based combinations (AUC 0.80-0.90)
- **Medium-performing models**: TNFa, IL1b combinations (AUC 0.65-0.80)
- **Poor-performing models**: IL10, IFNg combinations (AUC 0.50-0.65)

This distribution allows testing of threshold optimization and demonstrates real-world biomarker analysis scenarios.
"""
    
    readme_file = os.path.join(output_dir, "README.md")
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Documentation saved to: {readme_file}")
    print("\n🎯 Expected AUC ranges:")
    for biomarker, params in biomarker_info.items():
        print(f"   - {biomarker}: {params['description']}")
    
    print(f"\n📊 Data generation completed!")
    print(f"Use the commands in {readme_file} to test with realistic data.")

if __name__ == "__main__":
    main()