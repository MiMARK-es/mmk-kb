# Realistic Biomarker Test Data

This dataset contains realistic biomarker measurements designed to demonstrate the ROC analysis functionality with meaningful AUC distributions.

## Dataset Characteristics

- **Samples**: 100 total (50 cases, 50 controls)
- **Biomarkers**: 6
- **Design**: Simulated inflammatory biomarkers with varying discriminative power

## Biomarker Profiles

### IL6
- **Cases**: μ=15.2, σ=4.8
- **Controls**: μ=8.1, σ=2.9
- **Expected Performance**: Excellent discriminator - AUC ~0.85-0.90

### TNFa
- **Cases**: μ=12.7, σ=5.2
- **Controls**: μ=7.3, σ=3.1
- **Expected Performance**: Good discriminator - AUC ~0.75-0.80

### IL1b
- **Cases**: μ=9.8, σ=4.1
- **Controls**: μ=6.2, σ=3.7
- **Expected Performance**: Fair discriminator - AUC ~0.65-0.70

### IL10
- **Cases**: μ=5.4, σ=3.2
- **Controls**: μ=4.1, σ=2.8
- **Expected Performance**: Poor discriminator - AUC ~0.55-0.60

### CRP
- **Cases**: μ=8.9, σ=2.7
- **Controls**: μ=3.2, σ=1.8
- **Expected Performance**: Very good discriminator - AUC ~0.80-0.85

### IFNg
- **Cases**: μ=7.2, σ=4.9
- **Controls**: μ=6.8, σ=4.2
- **Expected Performance**: Very poor discriminator - AUC ~0.50-0.55

## Usage

```bash
# Create project and upload data
mmk-kb create "REALISTIC_TEST" "Realistic ROC Analysis" "Test with meaningful AUCs" "ROC Developer"
mmk-kb use "REALISTIC_TEST"

# Upload sample metadata
mmk-kb sample-upload experiment_data/realistic_data/realistic_samples.csv

# Upload biomarker data
mmk-kb experiment-upload experiment_data/realistic_data/realistic_biomarker_study.csv \
  "Realistic Biomarker Study" "Multi-biomarker panel with varying discriminative power"

# Run ROC analysis
mmk-kb roc-run 1 "Realistic ROC Analysis" 0.3 --max-combinations 3 \
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
