# ROC Normalized Analysis

## Overview

ROC Normalized Analysis extends the standard ROC analysis by normalizing biomarker values against a selected reference biomarker before performing the analysis. This technique is useful when you want to control for variations in one biomarker and examine the diagnostic performance of ratios of biomarkers.

## Key Concepts

### Normalization Process
- **Normalizer Biomarker**: A selected biomarker used as the denominator for all calculations
- **Normalized Features**: All other biomarkers are divided by the normalizer biomarker
- **Analysis**: ROC analysis is performed on the normalized ratios (e.g., `biomarker1/normalizer`, `biomarker2/normalizer`)

### Use Cases
- **Reference Standardization**: When one biomarker serves as a stable reference (e.g., total protein, housekeeping genes)
- **Ratio Analysis**: When diagnostic value comes from biomarker ratios rather than absolute values
- **Variation Control**: To reduce technical or biological variation in one dimension

## Command Reference

### Run ROC Normalized Analysis
```bash
mmk-kb roc-norm-run <experiment_id> <normalizer_biomarker_version_id> <name> <prevalence> [options]
```

**Arguments:**
- `experiment_id`: ID of the experiment to analyze
- `normalizer_biomarker_version_id`: ID of the biomarker version to use as normalizer
- `name`: Name for the analysis
- `prevalence`: Expected disease prevalence (0.0-1.0)

**Options:**
- `--max-combinations N`: Maximum biomarker combinations to test (default: 3)
- `--description TEXT`: Analysis description

**Example:**
```bash
# Use biomarker version 5 as normalizer for experiment 1
mmk-kb roc-norm-run 1 5 "Cytokine Ratio Analysis" 0.25 --max-combinations 2 --description "IL-6 normalized analysis"
```

### List ROC Normalized Analyses
```bash
mmk-kb roc-norm-list [options]
```

**Options:**
- `--experiment ID`: Filter by experiment ID

**Example:**
```bash
mmk-kb roc-norm-list --experiment 1
```

### Show Analysis Details
```bash
mmk-kb roc-norm-show <analysis_id>
```

**Example:**
```bash
mmk-kb roc-norm-show 1
```

### Generate Analysis Report
```bash
mmk-kb roc-norm-report <analysis_id> [options]
```

**Options:**
- `--output FILE`: Save report to CSV file
- `--top N`: Show only top N models by AUC

**Example:**
```bash
mmk-kb roc-norm-report 1 --output normalized_results.csv --top 10
```

### Show Model Details
```bash
mmk-kb roc-norm-model <model_id>
```

### Export ROC Curve Data
```bash
mmk-kb roc-norm-curve <model_id> [options]
```

**Options:**
- `--output FILE`: Save curve data to CSV file

## Workflow Example

### 1. Prepare Data
Ensure you have:
- An experiment with multiple biomarker measurements
- Samples with diagnostic information (dx field)
- A suitable normalizer biomarker

### 2. Identify Normalizer Biomarker
```bash
# List biomarker versions to find normalizer
mmk-kb biomarker-versions --experiment 1
```

### 3. Run Analysis
```bash
# Run normalized analysis using biomarker version 3 as normalizer
mmk-kb roc-norm-run 1 3 "Protein Ratio Analysis" 0.3 --max-combinations 2
```

### 4. Review Results
```bash
# Show analysis overview
mmk-kb roc-norm-show 1

# Generate detailed report
mmk-kb roc-norm-report 1 --top 5
```

### 5. Examine Best Models
```bash
# Look at specific model details
mmk-kb roc-norm-model 15

# Export ROC curve for plotting
mmk-kb roc-norm-curve 15 --output model_15_curve.csv
```

## Understanding Results

### Report Columns
- **Model_ID**: Unique model identifier
- **AUC**: Area Under the ROC Curve
- **Normalizer**: Name of the normalizer biomarker
- **Biomarker_N**: Normalized biomarker combinations (shown as ratios)
- **Prevalence**: Disease prevalence used for calculations
- **Threshold Metrics**: Performance at 97% sensitivity, 95% sensitivity, and optimal thresholds

### Model Coefficients
Coefficients represent the logistic regression weights for normalized features:
- **Intercept**: Model intercept term
- **Feature Coefficients**: Weights for each normalized biomarker ratio

### Performance Metrics
For each threshold type:
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate  
- **PPV**: Positive predictive value (depends on prevalence)
- **NPV**: Negative predictive value (depends on prevalence)

## Technical Notes

### Data Requirements
- All samples must have values for the normalizer biomarker
- Normalizer values must be non-zero
- Samples without complete data are excluded from analysis

### Normalization Formula
```
normalized_value = original_biomarker_value / normalizer_biomarker_value
```

### Model Training
1. Biomarker values are normalized by division
2. Normalized values are standardized (mean=0, std=1)
3. Logistic regression is fitted to predict diagnosis
4. ROC curves and metrics are calculated

## Comparison with Standard ROC Analysis

| Aspect | Standard ROC | Normalized ROC |
|--------|-------------|----------------|
| Features | Absolute biomarker values | Biomarker ratios |
| Interpretation | Individual biomarker performance | Relative biomarker performance |
| Use Case | Direct biomarker evaluation | Ratio-based diagnostics |
| Normalization | None | Division by reference biomarker |

## Best Practices

### Choosing a Normalizer
- Select a stable, well-measured biomarker
- Avoid biomarkers that correlate strongly with diagnosis
- Consider biological relevance (e.g., housekeeping proteins)
- Ensure normalizer has no zero values

### Analysis Parameters
- Set appropriate prevalence based on your study population
- Start with smaller max-combinations for exploration
- Review failed models to understand data issues

### Interpretation
- Focus on AUC for overall model ranking
- Consider sensitivity/specificity trade-offs for clinical application
- Examine coefficients to understand biomarker contributions
- Validate findings in independent datasets

## Troubleshooting

### Common Issues
1. **No models created**: Check that normalizer biomarker exists in experiment
2. **All models failed**: Verify normalizer has non-zero values for all samples
3. **Poor performance**: Consider different normalizer or check data quality
4. **Missing samples**: Ensure samples have both normalizer and target biomarker values

### Error Messages
- `"Normalizer biomarker version not found"`: Check biomarker version ID
- `"No valid data found"`: Verify experiment has measurements
- `"No biomarkers available after excluding normalizer"`: Need additional biomarkers beyond normalizer