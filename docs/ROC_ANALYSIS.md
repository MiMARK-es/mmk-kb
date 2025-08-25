# ROC Analysis Guide

## Overview

The ROC (Receiver Operating Characteristic) analysis module provides comprehensive biomarker performance evaluation through logistic regression modeling and ROC curve analysis. This feature enables systematic evaluation of single biomarkers and biomarker combinations to identify optimal diagnostic panels.

## Key Features

- **Flexible Biomarker Combinations**: Analyze single biomarkers, pairs, triplets, or any combination size
- **Comprehensive Metrics**: Calculate AUC, sensitivity, specificity, PPV, NPV at multiple thresholds
- **Model Storage**: Store complete model coefficients for future predictions
- **ROC Curve Data**: Store all ROC curve points for plotting and analysis
- **Customizable Prevalence**: Calculate PPV/NPV based on expected disease prevalence
- **Performance Thresholds**: Evaluate models at 97% sensitivity, 95% sensitivity, and optimal (max sensitivity+specificity)

## Quick Start

### 1. Run ROC Analysis

```bash
# Basic analysis with default settings
mmk-kb roc-run <experiment_id> "Analysis Name" <prevalence>

# Advanced analysis with custom parameters
mmk-kb roc-run 1 "Comprehensive Analysis" 0.3 \
  --max-combinations 3 \
  --description "Multi-biomarker diagnostic panel evaluation"
```

### 2. View Results

```bash
# List all ROC analyses
mmk-kb roc-list

# Show analysis details
mmk-kb roc-show <analysis_id>

# Generate comprehensive report
mmk-kb roc-report <analysis_id> --top 10
```

### 3. Examine Specific Models

```bash
# Show model details with coefficients
mmk-kb roc-model <model_id>

# Export ROC curve data
mmk-kb roc-curve <model_id> --output model_curve.csv
```

## Command Reference

### `roc-run` - Execute ROC Analysis

Runs comprehensive ROC analysis on experiment data, testing all biomarker combinations up to the specified maximum size.

**Syntax:**
```bash
mmk-kb roc-run <experiment_id> <name> <prevalence> [options]
```

**Parameters:**
- `experiment_id`: ID of experiment containing biomarker measurements
- `name`: Descriptive name for the analysis
- `prevalence`: Expected disease prevalence (0.0-1.0) for PPV/NPV calculations

**Options:**
- `--max-combinations N`: Maximum biomarkers per model (default: 3)
- `--description TEXT`: Analysis description

**Example:**
```bash
mmk-kb roc-run 1 "Cytokine Panel Study" 0.25 --max-combinations 4
```

### `roc-list` - List ROC Analyses

Lists all ROC analyses, optionally filtered by experiment.

**Syntax:**
```bash
mmk-kb roc-list [--experiment <experiment_id>]
```

**Example:**
```bash
mmk-kb roc-list --experiment 1
```

### `roc-show` - Analysis Details

Shows comprehensive details of a specific ROC analysis including top-performing models.

**Syntax:**
```bash
mmk-kb roc-show <analysis_id>
```

### `roc-report` - Generate Report

Generates detailed performance report with all metrics for all models in an analysis.

**Syntax:**
```bash
mmk-kb roc-report <analysis_id> [options]
```

**Options:**
- `--output FILE`: Save report to CSV file
- `--top N`: Show only top N models by AUC

**Example:**
```bash
mmk-kb roc-report 1 --output results.csv --top 15
```

### `roc-model` - Model Details

Shows detailed information about a specific model including coefficients and all performance metrics.

**Syntax:**
```bash
mmk-kb roc-model <model_id>
```

### `roc-curve` - Export Curve Data

Exports ROC curve coordinates for plotting and further analysis.

**Syntax:**
```bash
mmk-kb roc-curve <model_id> [--output <file>]
```

**Example:**
```bash
mmk-kb roc-curve 5 --output biomarker_a_curve.csv
```

## Analysis Process

### 1. Data Preparation

The system automatically:
- Extracts biomarker measurements from the specified experiment
- Links measurements to sample diagnosis information (dx field)
- Creates a complete dataset with all biomarker values per sample
- Excludes samples with missing biomarker data

### 2. Model Generation

For each biomarker combination:
- Standardizes biomarker values using StandardScaler
- Fits logistic regression model
- Calculates ROC curve and AUC
- Stores model coefficients and scaling parameters

### 3. Performance Evaluation

Each model is evaluated at three key thresholds:

**97% Sensitivity Threshold (`se_97_*`)**
- Finds threshold achieving ≥97% sensitivity
- Maximizes specificity among qualifying thresholds
- Clinical focus: minimize false negatives

**95% Sensitivity Threshold (`se_95_*`)**
- Finds threshold achieving ≥95% sensitivity
- Maximizes specificity among qualifying thresholds
- Balanced clinical performance

**Optimal Threshold (`max_sum_*`)**
- Maximizes (sensitivity + specificity)
- Youden's J statistic optimization
- Statistical optimum

## Report Columns

ROC analysis reports include the following columns:

### Core Metrics
- `Model_ID`: Unique model identifier
- `AUC`: Area Under the ROC Curve (0-1)
- `Prevalence`: Expected disease prevalence used for calculations

### Biomarker Identification
- `Biomarker_1`, `Biomarker_2`, `Biomarker_3`, etc.: Biomarkers in the model

### Performance at 97% Sensitivity
- `se_97_Threshold`: Probability threshold
- `se_97_Sensitivity`: Achieved sensitivity
- `se_97_Specificity`: Corresponding specificity
- `se_97_NPV`: Negative Predictive Value
- `se_97_PPV`: Positive Predictive Value

### Performance at 95% Sensitivity
- `se_95_Threshold`: Probability threshold
- `se_95_Sensitivity`: Achieved sensitivity
- `se_95_Specificity`: Corresponding specificity
- `se_95_NPV`: Negative Predictive Value
- `se_95_PPV`: Positive Predictive Value

### Optimal Performance
- `max_sum_Threshold`: Probability threshold
- `max_sum_Sensitivity`: Sensitivity at optimal point
- `max_sum_Specificity`: Specificity at optimal point
- `max_sum_NPV`: Negative Predictive Value
- `max_sum_PPV`: Positive Predictive Value

## Database Storage

### ROC Analyses Table
Stores analysis metadata:
- Analysis name and description
- Experiment reference
- Prevalence setting
- Maximum combination size
- Creation timestamp

### ROC Models Table
Stores individual models:
- Biomarker combination (JSON array)
- AUC value
- Complete model coefficients (JSON object)
- Scaling parameters for feature standardization

### ROC Metrics Table
Stores performance metrics:
- Model reference
- Threshold type (se_97, se_95, max_sum)
- All performance metrics

### ROC Curve Points Table
Stores complete ROC curve:
- False Positive Rate (FPR)
- True Positive Rate (TPR)
- Decision threshold

## Best Practices

### Analysis Design
1. **Prevalence Selection**: Use realistic disease prevalence for your target population
2. **Combination Size**: Start with smaller combinations (2-3) for initial screening
3. **Sample Size**: Ensure adequate cases and controls for stable models
4. **Data Quality**: Verify complete biomarker measurements across samples

### Interpretation
1. **AUC Interpretation**:
   - 0.9-1.0: Excellent discrimination
   - 0.8-0.9: Good discrimination
   - 0.7-0.8: Fair discrimination
   - 0.6-0.7: Poor discrimination
   - 0.5-0.6: Fail/Random

2. **Threshold Selection**:
   - Use `se_97_*` for screening applications (minimize false negatives)
   - Use `se_95_*` for balanced clinical performance
   - Use `max_sum_*` for research/comparison purposes

3. **Clinical Context**:
   - Consider PPV/NPV in context of disease prevalence
   - Evaluate cost of false positives vs false negatives
   - Validate findings in independent datasets

### Model Application
Use stored model coefficients to predict on new data:
```python
# Model coefficients are stored with scaling parameters
# Apply same standardization: (X - mean) / scale
# Prediction: probability = 1 / (1 + exp(-(intercept + sum(coef_i * X_i))))
```

## Examples

### Single Biomarker Screening
```bash
# Quick screening of individual biomarkers
mmk-kb roc-run 1 "Individual Biomarker Screen" 0.2 --max-combinations 1
mmk-kb roc-report 1 --top 5
```

### Comprehensive Panel Development
```bash
# Full combination analysis
mmk-kb roc-run 1 "Complete Panel Analysis" 0.3 --max-combinations 4 \
  --description "Comprehensive evaluation for diagnostic panel development"

# Generate full report
mmk-kb roc-report 1 --output comprehensive_results.csv

# Examine top model
mmk-kb roc-model $(mmk-kb roc-report 1 --top 1 | grep -E "^\s*[0-9]+" | awk '{print $1}' | head -1)
```

### Model Validation Preparation
```bash
# Export ROC curves for top models
for model_id in 5 7 12; do
  mmk-kb roc-curve $model_id --output "model_${model_id}_curve.csv"
done
```

## Integration with Analysis Pipelines

The ROC analysis module integrates seamlessly with the existing MMK-KB workflow:

1. **Project Setup**: Use existing project management
2. **Sample Import**: Standard sample CSV upload
3. **Biomarker Data**: Standard experiment upload
4. **ROC Analysis**: New ROC analysis commands
5. **Results Export**: CSV reports for external analysis

## Troubleshooting

### Common Issues

**"No valid data found for experiment"**
- Verify experiment contains measurements
- Check that samples have diagnosis information (dx field)
- Ensure biomarker measurements exist for samples

**"Foreign key constraint failed"**
- Verify experiment ID exists
- Check database integrity

**Models with poor AUC**
- Review data quality and sample size
- Consider biomarker preprocessing
- Validate diagnosis accuracy

### Performance Considerations

- Large combination sizes (>4) can generate many models
- Analysis time scales exponentially with biomarker count
- Consider filtering biomarkers before analysis
- Use appropriate hardware for large datasets