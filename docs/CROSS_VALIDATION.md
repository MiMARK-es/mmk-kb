# Cross-Validation Analysis Guide

This guide covers the new cross-validation features added to MMK-KB analyses.

## Overview

Cross-validation has been added to both ROC Analysis and ROC Normalized Analysis to provide more robust model evaluation. Two types of cross-validation are supported:

1. **Leave-One-Out (LOO) Cross-Validation**: Tests each sample individually
2. **Bootstrap Cross-Validation**: Uses random sampling with replacement

## Features

### Cross-Validation Types

#### Leave-One-Out (LOO)
- Trains on n-1 samples, tests on 1 sample
- Repeats for each sample in the dataset
- Provides conservative estimate of model performance
- Enabled by default when cross-validation is activated

#### Bootstrap
- Uses random sampling with replacement for training
- Configurable number of iterations (default: 200)
- Configurable validation set size (default: 20% of data)
- Provides robust estimate with confidence intervals

### Configuration Options

Cross-validation is configured using the following parameters:

- `--enable-cv`: Enable cross-validation (required to activate)
- `--disable-loo`: Disable Leave-One-Out validation
- `--disable-bootstrap`: Disable Bootstrap validation
- `--bootstrap-iterations`: Number of bootstrap iterations (default: 200)
- `--bootstrap-validation-size`: Fraction of data for validation (default: 0.2)

## Usage Examples

### Basic ROC Analysis with Cross-Validation

```bash
# Enable both LOO and Bootstrap cross-validation
mmk-kb analysis roc-run 1 "CV Analysis" 0.3 --enable-cv

# Enable only LOO cross-validation
mmk-kb analysis roc-run 1 "LOO Analysis" 0.3 --enable-cv --disable-bootstrap

# Enable only Bootstrap with custom parameters
mmk-kb analysis roc-run 1 "Bootstrap Analysis" 0.3 --enable-cv --disable-loo \
  --bootstrap-iterations 500 --bootstrap-validation-size 0.25
```

### ROC Normalized Analysis with Cross-Validation

```bash
# Enable cross-validation for normalized analysis
mmk-kb analysis roc-norm-run 1 5 "Normalized CV Analysis" 0.3 --enable-cv

# Custom bootstrap configuration
mmk-kb analysis roc-norm-run 1 5 "Custom Bootstrap" 0.3 --enable-cv \
  --bootstrap-iterations 100 --bootstrap-validation-size 0.3
```

### Viewing Cross-Validation Results

```bash
# List analyses with CV indicators
mmk-kb analysis roc-list

# Show detailed analysis with CV results
mmk-kb analysis roc-show 1

# Generate report including CV metrics
mmk-kb analysis roc-report 1 --output cv_results.csv
```

## Output Interpretation

### Cross-Validation Metrics

When cross-validation is enabled, each model includes:

- **LOO AUC Mean**: Average AUC across all LOO folds
- **LOO AUC Std**: Standard deviation of LOO AUCs
- **Bootstrap AUC Mean**: Average AUC across bootstrap iterations
- **Bootstrap AUC Std**: Standard deviation of bootstrap AUCs

### CLI Output

The CLI will show cross-validation information:

```
🔄 Running ROC analysis 'CV Analysis' on experiment 1...
📊 Cross-validation enabled: LOO, Bootstrap(200 iter)
✅ Analysis completed!
   Analysis ID: 1
   Total combinations tested: 15
   Successful models: 15
   Failed models: 0
```

### Model Display

When viewing analysis details, CV results are shown:

```
Top 10 models by AUC:
   1. Model 5: AUC = 0.850 [CV: LOO: 0.823±0.045, Bootstrap: 0.841±0.032]
   2. Model 3: AUC = 0.832 [CV: LOO: 0.801±0.052, Bootstrap: 0.825±0.038]
```

## Technical Details

### Database Storage

Cross-validation configurations and results are stored in the database:

- Analysis table includes `cross_validation_config` JSON field
- Model table includes `cross_validation_results` JSON field
- Backward compatible with existing analyses

### Performance Considerations

- LOO CV time complexity: O(n × training_time)
- Bootstrap CV time complexity: O(iterations × training_time)
- Memory usage scales with dataset size
- Consider reducing iterations for large datasets

### Best Practices

1. **Enable CV for final models**: Use for publication-ready results
2. **Start with defaults**: 200 bootstrap iterations usually sufficient
3. **Consider dataset size**: 
   - Small datasets (<100 samples): Use LOO
   - Large datasets (>1000 samples): Use Bootstrap only
4. **Validate assumptions**: Ensure balanced classes in validation sets

## Troubleshooting

### Common Issues

**Error: "No valid data found"**
- Check that experiment has sufficient samples
- Ensure both positive and negative cases exist

**Warning: "Some models failed"**
- Normal for small datasets or extreme biomarker combinations
- Check failed_models list for specific errors

**Slow performance**
- Reduce bootstrap iterations for initial exploration
- Disable LOO for large datasets
- Use `--max-combinations` to limit complexity

### Performance Optimization

```bash
# Fast exploration (reduced CV)
mmk-kb analysis roc-run 1 "Quick Test" 0.3 --enable-cv \
  --disable-loo --bootstrap-iterations 50

# Production analysis (comprehensive CV)
mmk-kb analysis roc-run 1 "Production" 0.3 --enable-cv \
  --bootstrap-iterations 1000
```

## Migration from Legacy Commands

The old command structure still works but is deprecated:

```bash
# Old (deprecated)
mmk-kb roc-run 1 "Analysis" 0.3

# New (recommended)
mmk-kb analysis roc-run 1 "Analysis" 0.3
```

Cross-validation is only available with the new command structure.

## API Usage

For programmatic access:

```python
from src.mmkkb.analyses.roc_analysis import ROCAnalyzer, ROCAnalysis
from src.mmkkb.analyses.base_analysis import CrossValidationConfig

# Configure cross-validation
cv_config = CrossValidationConfig(
    enable_loo=True,
    enable_bootstrap=True,
    bootstrap_iterations=200,
    bootstrap_validation_size=0.2
)

# Create analysis with CV
analysis = ROCAnalysis(
    name="API Analysis",
    description="Analysis with CV",
    experiment_id=1,
    prevalence=0.3,
    max_combination_size=3,
    cross_validation_config=cv_config
)

# Run analysis
analyzer = ROCAnalyzer()
results = analyzer.run_roc_analysis(analysis)
```