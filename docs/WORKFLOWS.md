# Data Workflows

## Overview

This guide covers common research workflows and usage patterns for MMK-KB, from basic data entry to complex multi-study analyses.

## Basic Workflows

### Single Study Workflow

**Scenario**: Single biomarker study with clinical samples

```bash
# 1. Create and setup project
mmk-kb create "PILOT_2024" "Pilot Biomarker Study" "Initial validation study" "Dr. Research"
mmk-kb use "PILOT_2024"

# 2. Upload sample data
mmk-kb sample-preview clinical_samples.csv
mmk-kb sample-upload clinical_samples.csv

# 3. Verify sample upload
mmk-kb samples
# Expected: List of uploaded samples with clinical metadata

# 4. Upload biomarker measurements
mmk-kb csv-preview cytokine_data.csv
mmk-kb experiment-upload cytokine_data.csv "Cytokine Panel" "Initial cytokine screening"

# 5. Review results
mmk-kb experiments
mmk-kb measurements-summary
mmk-kb biomarkers

# 6. Backup completed study
mmk-kb backup --dir study_backups
```

### Multi-Experiment Workflow

**Scenario**: Multiple experiments on the same sample set

```bash
# Setup project and samples (as above)
mmk-kb create "MULTI_EXP_2024" "Multi-Experiment Study" "Comprehensive analysis" "Research Team"
mmk-kb use "MULTI_EXP_2024"
mmk-kb sample-upload clinical_data.csv

# Upload multiple experiments
mmk-kb experiment-upload cytokine_panel.csv "Cytokine Analysis" "Pro-inflammatory cytokines" --version "v1.0"
mmk-kb experiment-upload chemokine_panel.csv "Chemokine Analysis" "Chemokine profiling" --version "v1.0"
mmk-kb experiment-upload growth_factors.csv "Growth Factor Panel" "Growth factor measurements" --version "v1.0"

# Compare across experiments
mmk-kb experiments
mmk-kb measurements-summary
mmk-kb biomarker-analysis 1  # Analyze specific biomarker across experiments
```

## Advanced Workflows

### Method Validation Workflow

**Scenario**: Comparing different assay versions or methods

```bash
# Setup project
mmk-kb create "METHOD_VALIDATION" "Assay Validation Study" "Comparing v1.0 vs v2.0 assays" "QA Team"
mmk-kb use "METHOD_VALIDATION"
mmk-kb sample-upload validation_samples.csv

# Upload same biomarkers with different methods
mmk-kb experiment-upload method_v1_data.csv "Method v1.0" "Original assay method" --version "v1.0"
mmk-kb experiment-upload method_v2_data.csv "Method v2.0" "Improved assay method" --version "v2.0"

# Analyze method differences
mmk-kb biomarker-versions --biomarker "IL-6"
mmk-kb biomarker-analysis 1  # Shows data from both versions

# Export for statistical analysis
# (Programmatic export to compare methods in R/Python)
```

### Multi-Center Study Workflow

**Scenario**: Data from multiple collection centers

```bash
# Create main project
mmk-kb create "MULTICENTER_2024" "Multi-Center Study" "5-site biomarker study" "Principal Investigator"
mmk-kb use "MULTICENTER_2024"

# Upload samples from each center (with collection_center field)
mmk-kb sample-upload site_a_samples.csv
mmk-kb sample-upload site_b_samples.csv
mmk-kb sample-upload site_c_samples.csv

# Upload experiments (can be center-specific or combined)
mmk-kb experiment-upload site_a_biomarkers.csv "Site A Analysis" "Biomarker data from Site A"
mmk-kb experiment-upload site_b_biomarkers.csv "Site B Analysis" "Biomarker data from Site B"
mmk-kb experiment-upload combined_analysis.csv "Combined Analysis" "Pooled biomarker analysis"

# Review by center
mmk-kb samples  # Shows collection_center for each sample
mmk-kb measurements-summary
```

### Longitudinal Study Workflow

**Scenario**: Time-series biomarker measurements

```bash
# Setup longitudinal project
mmk-kb create "LONGITUDINAL_2024" "Longitudinal Biomarker Study" "6-month follow-up study" "Clinical Team"
mmk-kb use "LONGITUDINAL_2024"

# Upload baseline samples
mmk-kb sample-upload baseline_samples.csv

# Upload time-point experiments
mmk-kb experiment-upload baseline_biomarkers.csv "Baseline" "Baseline measurements" --version "baseline"
mmk-kb experiment-upload month_3_biomarkers.csv "Month 3" "3-month follow-up" --version "month_3"
mmk-kb experiment-upload month_6_biomarkers.csv "Month 6" "6-month follow-up" --version "month_6"

# Track biomarker changes over time
mmk-kb biomarker-analysis 1  # Shows temporal progression
```

## Environment-Based Workflows

### Development to Production Pipeline

**Development Phase:**
```bash
# Work in development environment
mmk-kb setenv development
mmk-kb create "DEV_STUDY" "Development Study" "Testing workflows" "Developer"
mmk-kb use "DEV_STUDY"

# Test with small datasets
mmk-kb sample-upload test_samples.csv
mmk-kb experiment-upload test_biomarkers.csv "Test Experiment" "Testing upload process"

# Verify functionality
mmk-kb measurements-summary
```

**Staging Phase:**
```bash
# Move to staging for validation
mmk-kb setenv staging
mmk-kb copy development staging

# Validate in staging environment
mmk-kb list
mmk-kb use "DEV_STUDY"
mmk-kb measurements-summary

# Upload larger validation datasets
mmk-kb sample-upload validation_samples.csv
mmk-kb experiment-upload validation_biomarkers.csv "Validation Experiment" "QA validation"
```

**Production Deployment:**
```bash
# Deploy to production
mmk-kb setenv production
mmk-kb copy staging production

# Backup production before use
mmk-kb backup --dir production_backups

# Production operations
mmk-kb create "PROD_STUDY_2024" "Production Study" "Live research study" "PI"
# ... continue with production workflows
```

### Quality Control Workflow

**Pre-Upload QC:**
```bash
# Use staging environment for QC
mmk-kb setenv staging

# Preview and validate all data files
mmk-kb sample-preview samples.csv --rows 10
mmk-kb csv-preview experiment_data.csv --rows 10

# Test upload with subset
head -20 samples.csv > test_samples.csv
mmk-kb sample-upload test_samples.csv

# Verify upload worked correctly
mmk-kb samples
```

**Post-Upload QC:**
```bash
# Check data integrity
mmk-kb measurements-summary
mmk-kb biomarkers

# Validate expected sample counts
mmk-kb samples | wc -l  # Should match expected count

# Check for missing biomarker links
mmk-kb experiment-show 1  # Review any missing sample warnings

# If QC passes, move to production
mmk-kb copy staging production
```

## Data Integration Workflows

### Programmatic Analysis Workflow

```python
# Python script for integrated analysis
from mmkkb.projects import ProjectDatabase
from mmkkb.samples import SampleDatabase
from mmkkb.experiments import ExperimentDatabase
from mmkkb.csv_processor import CSVProcessor
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Initialize connections
project_db = ProjectDatabase()
sample_db = SampleDatabase()
experiment_db = ExperimentDatabase()
csv_processor = CSVProcessor()

# Load project data
project = project_db.get_project_by_code("ANALYSIS_2024")
samples = sample_db.list_samples(project_id=project.id)
experiments = experiment_db.list_experiments(project_id=project.id)

print(f"Loaded {len(samples)} samples and {len(experiments)} experiments")

# Extract experiment data for analysis
experiment_dfs = []
for exp in experiments:
    df = csv_processor.get_experiment_data_as_dataframe(exp.id)
    if df is not None:
        df['experiment'] = exp.name
        experiment_dfs.append(df)

# Combine all experiment data
if experiment_dfs:
    combined_df = pd.concat(experiment_dfs, ignore_index=True)
    
    # Merge with sample metadata
    sample_meta = pd.DataFrame([{
        'sample': s.code,
        'age': s.age,
        'bmi': s.bmi,
        'dx': s.dx,
        'collection_center': s.collection_center
    } for s in samples])
    
    analysis_df = combined_df.merge(sample_meta, on='sample')
    
    # Perform statistical analysis
    biomarkers = [col for col in analysis_df.columns 
                 if col not in ['sample', 'experiment', 'age', 'bmi', 'dx', 'collection_center']]
    
    for biomarker in biomarkers:
        case_values = analysis_df[analysis_df['dx'] == True][biomarker].dropna()
        control_values = analysis_df[analysis_df['dx'] == False][biomarker].dropna()
        
        if len(case_values) > 0 and len(control_values) > 0:
            t_stat, p_value = stats.ttest_ind(case_values, control_values)
            print(f"{biomarker}: t={t_stat:.3f}, p={p_value:.3f}")
```

### R Integration Workflow

```r
# R script for statistical analysis
library(DBI)
library(RSQLite)
library(dplyr)
library(ggplot2)

# Connect to MMK-KB database
con <- dbConnect(SQLite(), "mmk_kb.db")

# Query data directly from database
query <- "
SELECT 
    s.code as sample_code,
    s.age, s.bmi, s.dx,
    b.name as biomarker,
    bv.version,
    m.value,
    e.name as experiment
FROM measurements m
JOIN samples s ON m.sample_id = s.id
JOIN experiments e ON m.experiment_id = e.id
JOIN biomarker_versions bv ON m.biomarker_version_id = bv.id
JOIN biomarkers b ON bv.biomarker_id = b.id
WHERE s.project_id = 1
"

data <- dbGetQuery(con, query)
dbDisconnect(con)

# Statistical analysis
results <- data %>%
  group_by(biomarker) %>%
  summarise(
    case_mean = mean(value[dx == 1], na.rm = TRUE),
    control_mean = mean(value[dx == 0], na.rm = TRUE),
    case_n = sum(dx == 1),
    control_n = sum(dx == 0),
    t_test_p = t.test(value[dx == 1], value[dx == 0])$p.value,
    .groups = 'drop'
  )

print(results)

# Visualization
ggplot(data, aes(x = factor(dx), y = value, fill = factor(dx))) +
  geom_boxplot() +
  facet_wrap(~biomarker, scales = "free_y") +
  labs(x = "Diagnosis", y = "Biomarker Value", fill = "Disease Status") +
  theme_minimal()
```

## Backup and Recovery Workflows

### Daily Backup Workflow

```bash
#!/bin/bash
# Daily backup script

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/daily"

# Backup all environments
for env in development staging production; do
    echo "Backing up $env environment..."
    mmk-kb --env $env backup --dir "$BACKUP_DIR/$DATE"
done

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -type d -mtime +30 -exec rm -rf {} \;

echo "Daily backup completed: $DATE"
```

### Disaster Recovery Workflow

**Recovery Scenario**: Production database corruption

```bash
# 1. Assess damage
mmk-kb --env production list
# If this fails, database is corrupted

# 2. Find latest backup
ls -la /backups/daily/*/mmk_kb_production_*.db | tail -5

# 3. Restore from backup
LATEST_BACKUP="/backups/daily/20240821/mmk_kb_production_20240821_140000.db"
mmk-kb --env production restore "$LATEST_BACKUP" --no-confirm

# 4. Verify restoration
mmk-kb --env production list
mmk-kb --env production env

# 5. Resume operations
echo "Production restored from backup: $LATEST_BACKUP"
```

### Migration Workflow

**Scenario**: Moving from development to new production server

```bash
# On source server
mmk-kb --env production backup --dir migration_backup
scp migration_backup/mmk_kb_production_*.db newserver:/tmp/

# On destination server
mmk-kb --env production restore /tmp/mmk_kb_production_*.db --no-confirm
mmk-kb --env production vacuum  # Optimize after migration
mmk-kb --env production list    # Verify migration
```

## Troubleshooting Workflows

### Data Validation Workflow

```bash
# Check for common issues
mmk-kb env  # Verify environment and database status

# Project-level checks
mmk-kb list
mmk-kb show "PROJECT_CODE"

# Sample-level checks
mmk-kb samples --project "PROJECT_CODE" | wc -l  # Count samples
mmk-kb samples | head -10  # Check sample format

# Experiment-level checks
mmk-kb experiments
mmk-kb measurements-summary

# Biomarker checks
mmk-kb biomarkers
mmk-kb biomarker-versions
```

### Performance Optimization Workflow

```bash
# Database maintenance
mmk-kb vacuum --env production

# Check database sizes
mmk-kb env

# Cleanup test data
mmk-kb clean-tests --no-confirm

# Archive old projects (manual process)
# 1. Export old project data
# 2. Delete from active database
# 3. Store archives separately
```

## Best Practices Summary

### Data Organization
- Use descriptive project codes and names
- Implement consistent sample naming conventions
- Document biomarker versions and methods
- Maintain data dictionaries

### Environment Management
- Develop in development environment
- Test in staging environment  
- Deploy to production environment
- Use separate environments for different studies

### Quality Control
- Always preview CSV files before upload
- Validate data in staging before production
- Regular backups before major operations
- Monitor database growth and performance

### Documentation
- Document workflow procedures
- Maintain change logs for methods
- Record data provenance information
- Share workflows with team members