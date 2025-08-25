# ROC Ratios Analysis

ROC Ratios Analysis performs diagnostic modeling using biomarker ratios as features. This analysis generates all possible ratios between biomarkers and creates logistic regression models to find the best combinations for disease classification.

## Overview

The ROC Ratios Analysis differs from standard ROC analysis by:
- Using **ratios between biomarkers** as features instead of individual biomarker values
- Testing **combinations of multiple ratios** in single models
- Providing **comprehensive cross-validation** options
- Calculating **clinical metrics** (PPV, NPV) at relevant sensitivity thresholds

## Key Features

### Ratio Generation
- Creates all possible ratios between available biomarkers (A/B, A/C, B/A, B/C, etc.)
- Supports combinations of multiple ratios in single models
- Configurable maximum combination size to control complexity

### Model Training
- Uses logistic regression with standardized features
- Handles edge cases (division by zero, infinite values)
- Provides detailed model coefficients and scaling parameters

### Cross-Validation
- Leave-One-Out (LOO) cross-validation
- Bootstrap cross-validation with configurable iterations
- Statistical measures (mean, standard deviation) for model stability

### Clinical Metrics
- **SE_97**: Metrics at 97% sensitivity threshold
- **SE_95**: Metrics at 95% sensitivity threshold  
- **MAX_SUM**: Metrics at maximum (sensitivity + specificity)
- PPV and NPV calculations using provided prevalence

## CLI Usage

### Run ROC Ratios Analysis

```bash
# Basic analysis
mmkkb analysis roc-ratios-run \
  --experiment-id 1 \
  --name "Inflammation Ratios" \
  --description "Testing cytokine ratios for sepsis diagnosis" \
  --prevalence 0.3 \
  --max-combination-size 2

# With cross-validation
mmkkb analysis roc-ratios-run \
  --experiment-id 1 \
  --name "CV Ratios Analysis" \
  --description "Ratios with cross-validation" \
  --prevalence 0.3 \
  --max-combination-size 1 \
  --enable-loo \
  --enable-bootstrap \
  --bootstrap-iterations 100
```

### List Analyses

```bash
# List all ROC ratios analyses
mmkkb analysis roc-ratios-list

# List analyses for specific experiment
mmkkb analysis roc-ratios-list --experiment-id 1
```

### Show Analysis Details

```bash
# Show detailed analysis information
mmkkb analysis roc-ratios-show --analysis-id 1

# Include model details
mmkkb analysis roc-ratios-show --analysis-id 1 --include-models
```

### Generate Reports

```bash
# Generate comprehensive analysis report
mmkkb analysis roc-ratios-report --analysis-id 1 --output results.csv

# Generate report with specific format
mmkkb analysis roc-ratios-report --analysis-id 1 --format excel --output analysis.xlsx
```

## Python API Usage

### Basic Analysis

```python
from mmkkb.analyses.roc_ratios_analysis import ROCRatiosAnalyzer, ROCRatiosAnalysis

# Initialize analyzer
analyzer = ROCRatiosAnalyzer("path/to/database.db")

# Create analysis configuration
analysis = ROCRatiosAnalysis(
    name="Biomarker Ratios Study",
    description="Testing IL6/PCT and TNF/CRP ratios",
    experiment_id=1,
    prevalence=0.25,  # 25% disease prevalence
    max_combination_size=2  # Test up to 2 ratios per model
)

# Run analysis
results = analyzer.run_roc_ratios_analysis(analysis)

print(f"Created {results['models_created']} models")
print(f"Best AUC: {max(m['auc'] for m in results['successful_models']):.3f}")
```

### Cross-Validation Analysis

```python
from mmkkb.analyses.base_analysis import CrossValidationConfig

# Configure cross-validation
cv_config = CrossValidationConfig(
    enable_loo=True,
    enable_bootstrap=True,
    bootstrap_iterations=200
)

# Create analysis with CV
analysis = ROCRatiosAnalysis(
    name="CV Ratios Analysis",
    description="Ratios analysis with comprehensive validation",
    experiment_id=1,
    prevalence=0.3,
    max_combination_size=1,
    cross_validation_config=cv_config
)

results = analyzer.run_roc_ratios_analysis(analysis)

# Check CV results
for model in results['successful_models']:
    if model['cross_validation_results']:
        cv = model['cross_validation_results']
        print(f"LOO CV AUC: {cv.loo_auc_mean:.3f} ± {cv.loo_auc_std:.3f}")
        print(f"Bootstrap CV AUC: {cv.bootstrap_auc_mean:.3f} ± {cv.bootstrap_auc_std:.3f}")
```

### Generate Analysis Report

```python
# Generate comprehensive report
report_df = analyzer.generate_analysis_report(analysis_id=1)

# Display top models
top_models = report_df.nlargest(5, 'AUC')
print(top_models[['Model_ID', 'AUC', 'Ratio_1', 'se_97_Sensitivity', 'se_97_Specificity']])

# Save to file
report_df.to_csv('roc_ratios_report.csv', index=False)
```

## Database Operations

### Direct Database Access

```python
from mmkkb.analyses.roc_ratios_analysis import ROCRatiosAnalysisDatabase

# Initialize database
db = ROCRatiosAnalysisDatabase("path/to/database.db")

# List all analyses
analyses = db.list_roc_ratios_analyses()
for analysis in analyses:
    print(f"{analysis.name}: {analysis.prevalence:.1%} prevalence")

# Get models for an analysis
models = db.get_roc_ratios_models_by_analysis(analysis_id=1)
best_model = max(models, key=lambda m: m.auc)
print(f"Best model AUC: {best_model.auc:.3f}")
print(f"Ratio combination: {best_model.ratio_combination}")

# Get metrics for a model
metrics = db.get_roc_ratios_metrics_by_model(best_model.id)
for metric in metrics:
    print(f"{metric.threshold_type}: Se={metric.sensitivity:.3f}, Sp={metric.specificity:.3f}")
```

## Understanding Results

### Model Output Structure

Each successful model in the results contains:

```python
{
    'model_id': 42,
    'ratio_combination': [(1, 2), (3, 4)],  # (numerator_bv_id, denominator_bv_id)
    'auc': 0.875,
    'cross_validation_results': CrossValidationResults(...),  # If CV enabled
    'metrics': [
        ('se_97', {'threshold': 0.3, 'sensitivity': 0.97, 'specificity': 0.65, ...}),
        ('se_95', {'threshold': 0.4, 'sensitivity': 0.95, 'specificity': 0.72, ...}),
        ('max_sum', {'threshold': 0.5, 'sensitivity': 0.88, 'specificity': 0.85, ...})
    ],
    'roc_points_count': 25
}
```

### Report Columns

The generated report includes:

- **Model_ID**: Unique identifier for the model
- **AUC**: Area Under the ROC Curve
- **Prevalence**: Expected disease prevalence used for PPV/NPV
- **Ratio_1, Ratio_2, ...**: Biomarker ratios used (e.g., "IL6_v1.0/PCT_v1.0")
- **CV_LOO_AUC_Mean/Std**: Leave-one-out cross-validation AUC statistics
- **CV_Bootstrap_AUC_Mean/Std**: Bootstrap cross-validation AUC statistics
- **se_97_***: Metrics at 97% sensitivity (Threshold, Sensitivity, Specificity, NPV, PPV)
- **se_95_***: Metrics at 95% sensitivity
- **max_sum_***: Metrics at maximum (sensitivity + specificity)

### Interpreting Ratios

Biomarker ratios can reveal important biological relationships:

- **IL6/PCT**: Inflammation vs. bacterial infection markers
- **TNF/CRP**: Acute vs. systemic inflammatory response
- **Biomarker_A/Biomarker_B**: Relative expression patterns

High-performing ratios often indicate:
- Complementary biomarkers with opposing patterns
- Normalization effects reducing technical variation
- Biologically meaningful relationships

## Best Practices

### Analysis Design

1. **Start Small**: Begin with `max_combination_size=1` to identify best individual ratios
2. **Increase Gradually**: Expand to larger combinations based on initial results
3. **Use Cross-Validation**: Always enable CV for robust performance estimates
4. **Set Appropriate Prevalence**: Use expected disease prevalence in target population

### Performance Evaluation

1. **Primary Metric**: Use AUC for overall discriminative ability
2. **Clinical Metrics**: Focus on SE_97 or SE_95 for high-sensitivity screening
3. **Cross-Validation**: Check CV results for model stability
4. **Multiple Models**: Compare top models for consistent patterns

### Interpretation Guidelines

1. **Biological Plausibility**: Ensure ratio combinations make biological sense
2. **Clinical Relevance**: Consider practical implementation of identified ratios
3. **Validation**: Test top models in independent datasets when possible
4. **Threshold Selection**: Choose thresholds based on clinical requirements

## Troubleshooting

### Common Issues

**No Models Created**
- Check that experiment has ≥2 biomarkers
- Verify samples have complete biomarker measurements
- Ensure sufficient sample size (recommend n≥20)

**Low AUC Values**
- May indicate weak biomarker relationships
- Try different biomarker combinations
- Check data quality and outliers

**Cross-Validation Warnings**
- Small sample sizes can cause CV instability
- Reduce bootstrap iterations for faster computation
- Consider LOO-only for very small datasets

**Memory Issues**
- Reduce `max_combination_size` for large biomarker sets
- Process analyses in smaller batches
- Use single-ratio analysis first to identify candidates

### Performance Optimization

- Use `max_combination_size=1` for exploratory analysis
- Enable CV selectively for final model evaluation
- Process large experiments in smaller biomarker subsets
- Consider parallel processing for multiple experiments

## Related Documentation

- [ROC Analysis](ROC_ANALYSIS.md) - Standard single-biomarker ROC analysis
- [ROC Normalized Analysis](ROC_NORMALIZED_ANALYSIS.md) - Normalized biomarker ROC analysis
- [Cross Validation](CROSS_VALIDATION.md) - Cross-validation methods and interpretation
- [Experiment Management](EXPERIMENT_MANAGEMENT.md) - Managing biomarker experiments