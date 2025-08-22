# Experiment Management

## Overview

Experiments contain biomarker measurements linked to samples within projects. The system supports sophisticated biomarker versioning and bulk data upload via CSV files.

## Data Model

### Experiments
- **Name**: Experiment identifier
- **Description**: Detailed experiment description
- **Project**: Parent project (automatically linked)
- **Upload Date**: When data was uploaded
- **CSV Filename**: Source data file reference
- **Timestamps**: Creation and update tracking

### Biomarkers & Versions
- **Biomarker**: Biological entity (e.g., "IL-6", "TNF-alpha")
  - Name, description, category
- **Biomarker Version**: Specific implementation
  - Version identifier (e.g., "v1.0", "RUO", "proprietary")
  - Version-specific description
- **Measurements**: Individual values linking experiment, sample, and biomarker version

## Listing Experiments

```bash
# List experiments in current project
mmk-kb experiments

# List experiments in specific project
mmk-kb experiments --project "PROJ001"

# Example output:
🧬 Found 3 experiments in project PROJ001:

ID  Name                    Description              Samples  Biomarkers  Upload Date      
----------------------------------------------------------------------------------------
1   Cytokine Panel         Initial cytokine screen  25       8           2024-08-22 10:30
2   Chemokine Analysis     Chemokine measurements   25       6           2024-08-22 11:45
3   Validation Study       Biomarker validation     20       5           2024-08-22 14:20
```

## Viewing Experiment Details

```bash
# Show detailed experiment information
mmk-kb experiment-show 1

# Example output:
🧬 Experiment Details:
   ID: 1
   Name: Cytokine Panel
   Description: Initial cytokine screening
   Project: PROJ001 (Biomarker Study)
   Upload Date: 2024-08-22 10:30:15
   CSV File: cytokine_panel.csv
   
📊 Experiment Statistics:
   Samples: 25
   Biomarkers: 8 (IL-6, TNF-alpha, CRP, IL-1beta, IL-10, IFN-gamma, IL-8, IL-12)
   Biomarker Versions: 8
   Measurements: 200
   
🧪 Sample Coverage:
   CASE_001: 8 measurements
   CASE_002: 8 measurements
   CTRL_001: 8 measurements
   ...
```

## CSV Data Upload

### CSV Format Requirements

The experiment CSV must have:
- **sample column**: Contains sample codes that exist in the project
- **Biomarker columns**: One column per biomarker (numeric values)

### Example CSV Format

```csv
sample,IL-6,TNF-alpha,CRP,IL-1beta
CASE_001,15.2,8.5,12.3,4.7
CASE_002,22.1,12.8,18.5,6.2
CTRL_001,5.8,3.2,4.1,2.1
CTRL_002,6.9,4.1,5.2,2.8
```

### CSV Upload Process

```bash
# Preview CSV before upload
mmk-kb csv-preview experiment_data.csv

# Preview with more rows
mmk-kb csv-preview experiment_data.csv --rows 10

# Upload experiment data
mmk-kb experiment-upload experiment_data.csv "Cytokine Panel" "Initial cytokine screening"

# Upload to specific project with version
mmk-kb experiment-upload data.csv "Study Name" "Description" --project "PROJ001" --version "v2.0"

# Example output:
✅ Experiment 'Cytokine Panel' created successfully
🧬 4 biomarker versions processed
🧪 200 measurements created
⚠️  2 sample codes not found in project: MISSING_001, MISSING_002
```

### Biomarker Versioning

When uploading experiments, you can specify a version for all biomarkers:

```bash
# Different version identifiers
mmk-kb experiment-upload data.csv "Experiment" "Description" --version "v1.0"
mmk-kb experiment-upload data.csv "Experiment" "Description" --version "RUO"
mmk-kb experiment-upload data.csv "Experiment" "Description" --version "proprietary"
```

**Version Benefits:**
- Track different assay implementations
- Compare results across assay versions
- Maintain data provenance
- Support method validation studies

## Biomarker Management

### Listing Biomarkers

```bash
# List all biomarkers
mmk-kb biomarkers

# Example output:
🧬 Found 12 biomarkers:

ID  Name         Category    Description                        Versions
------------------------------------------------------------------------
1   IL-6         cytokine    Interleukin-6 inflammatory...     v1.0, RUO
2   TNF-alpha    cytokine    Tumor necrosis factor alpha       v1.0
3   CRP          protein     C-reactive protein                v1.0, v2.0
4   IL-1beta     cytokine    Interleukin-1 beta                v1.0
```

### Biomarker Versions

```bash
# List all biomarker versions
mmk-kb biomarker-versions

# List versions for specific biomarker
mmk-kb biomarker-versions --biomarker "IL-6"

# Example output:
🧬 Biomarker versions for IL-6:

Version  Description                           Experiments  Measurements
------------------------------------------------------------------------
v1.0     Initial assay version                3            150
RUO      Research use only version            1            25
```

### Detailed Biomarker Analysis

```bash
# Get comprehensive biomarker analysis
mmk-kb biomarker-analysis 1

# Example output:
🧬 Biomarker Analysis: IL-6
Description: Interleukin-6 inflammatory cytokine
Category: cytokine

📊 Summary:
   - Total measurements: 245
   - Unique experiments: 3
   - Unique samples: 82
   - Versions used: v1.0, RUO

📋 Measurements by Version:
   v1.0: 220 measurements (89.8%)
   RUO:  25 measurements (10.2%)

📈 Value Statistics:
   Mean: 12.5 ± 8.2
   Range: 2.1 - 45.8
   Median: 10.3

🧪 Recent Measurements (latest 10):
Experiment           Sample     Value   Version   Date
------------------------------------------------------
Cytokine Panel       CASE_001   15.20   v1.0     2024-08-22
Cytokine Panel       CASE_002   22.10   v1.0     2024-08-22
Validation Study     CTRL_001   5.80    RUO      2024-08-22
```

## Measurement Operations

### Summary Statistics

```bash
# Get measurement summary for current project
mmk-kb measurements-summary

# Get summary for specific project
mmk-kb measurements-summary --project "PROJ001"

# Example output:
📊 Measurement Summary for PROJ001:

📈 Overall Statistics:
   Total Experiments: 3
   Total Samples: 25
   Total Biomarkers: 12
   Total Biomarker Versions: 14
   Total Measurements: 1,250

📋 By Experiment:
   Cytokine Panel (ID: 1):      200 measurements (8 biomarkers, 25 samples)
   Chemokine Analysis (ID: 2):  150 measurements (6 biomarkers, 25 samples)
   Validation Study (ID: 3):    100 measurements (5 biomarkers, 20 samples)

🧬 Top Biomarkers:
   IL-6:        245 measurements across 3 experiments
   TNF-alpha:   180 measurements across 2 experiments
   CRP:         156 measurements across 2 experiments
```

## Data Export

### Export Experiment Data

```bash
# Export specific experiment to DataFrame (programmatic)
# Available via Python API only
```

**Python API Example:**
```python
from mmkkb.csv_processor import CSVProcessor

processor = CSVProcessor()
df = processor.get_experiment_data_as_dataframe(experiment_id=1)

# DataFrame structure:
#   sample    IL-6_v1.0  TNF-alpha_v1.0  CRP_v1.0
#   CASE_001  15.2       8.5             12.3
#   CASE_002  22.1       12.8            18.5
```

## Programmatic Usage

### Python API

```python
from mmkkb.experiments import ExperimentDatabase, Experiment, Biomarker, BiomarkerVersion, Measurement
from mmkkb.csv_processor import CSVProcessor

# Initialize
experiment_db = ExperimentDatabase()
csv_processor = CSVProcessor()

# Create experiment
experiment = Experiment(
    name="API Experiment",
    description="Created via Python API",
    project_id=project_id
)
created_experiment = experiment_db.create_experiment(experiment)

# Create biomarker with version
biomarker_version = experiment_db.create_biomarker_with_version(
    biomarker_name="IL-6",
    version="v2.0",
    biomarker_description="Interleukin-6 cytokine",
    version_description="High-sensitivity assay",
    category="cytokine"
)

# Create measurement
measurement = Measurement(
    experiment_id=created_experiment.id,
    sample_id=sample_id,
    biomarker_version_id=biomarker_version.id,
    value=15.2
)
created_measurement = experiment_db.create_measurement(measurement)

# Query operations
experiments = experiment_db.list_experiments(project_id=project_id)
biomarkers = experiment_db.list_biomarkers()
measurements = experiment_db.get_measurements_by_experiment(experiment_id)

# Advanced analysis
analysis_data = experiment_db.get_biomarker_data_for_analysis(biomarker_id=1)
summary = experiment_db.get_measurement_summary(project_id=project_id)
```

### CSV Processing API

```python
# Validate CSV structure
is_valid, error_msg, biomarker_columns = csv_processor.validate_csv_structure("data.csv")

# Preview CSV
success, message, preview_data = csv_processor.preview_csv("data.csv", num_rows=5)

# Process upload
success, message, experiment = csv_processor.process_csv_upload(
    csv_path="data.csv",
    experiment_name="API Upload",
    experiment_description="Uploaded via API",
    project_id=project_id,
    biomarker_version="v1.0"
)
```

## Data Validation

### CSV Validation Rules

**Required Structure:**
- Must contain 'sample' column
- Must have at least one biomarker column
- Sample column cannot contain empty values
- Biomarker columns must contain numeric values

**Sample Linking:**
- Sample codes must exist in the target project
- Non-existent samples are reported but don't stop upload
- Case-sensitive sample code matching

**Value Validation:**
- Numeric values required for biomarker columns
- NaN/empty values are skipped (not stored)
- Non-numeric values cause validation errors

### Error Handling

**Missing Sample Codes:**
```
⚠️ 3 sample codes not found in project: MISSING_001, MISSING_002, INVALID_003
```

**Invalid CSV Structure:**
```
❌ CSV must contain a 'sample' column
❌ Column 'IL-6' contains non-numeric values
```

**Duplicate Measurements:**
- Same experiment + sample + biomarker version = duplicate
- Duplicates are automatically skipped
- No error generated for duplicates

## Integration Workflows

### Complete Research Workflow

```bash
# 1. Create project and upload samples
mmk-kb create "STUDY_2024" "Biomarker Study" "Description" "Researcher"
mmk-kb use "STUDY_2024"
mmk-kb sample-upload clinical_data.csv

# 2. Upload initial experiment
mmk-kb experiment-upload cytokine_panel.csv "Cytokine Panel" "Initial screening" --version "v1.0"

# 3. Upload follow-up experiment with different version
mmk-kb experiment-upload cytokine_panel_v2.csv "Cytokine Panel v2" "Updated assay" --version "v2.0"

# 4. Analyze results
mmk-kb experiments
mmk-kb biomarkers
mmk-kb biomarker-analysis 1
mmk-kb measurements-summary
```

### Method Validation Workflow

```bash
# Upload same samples with different assay versions
mmk-kb experiment-upload assay_v1.csv "Method Validation v1" "Original method" --version "v1.0"
mmk-kb experiment-upload assay_v2.csv "Method Validation v2" "Improved method" --version "v2.0"

# Compare biomarker versions
mmk-kb biomarker-versions --biomarker "IL-6"
mmk-kb biomarker-analysis 1  # Shows data from both versions
```

## Best Practices

### Experiment Design
- Use descriptive experiment names
- Include method details in descriptions
- Document assay versions consistently
- Plan biomarker naming conventions

### Data Upload
- Validate CSV format before upload
- Preview data with `csv-preview`
- Use consistent biomarker version identifiers
- Upload samples before experiments

### Quality Control
- Check for missing sample links
- Validate measurement counts
- Review biomarker version usage
- Monitor duplicate detection

### Version Management
- Use semantic versioning (v1.0, v1.1, v2.0)
- Document version differences
- Track assay changes over time
- Plan migration strategies

## Troubleshooting

### Experiment Upload Issues

**Sample Not Found:**
```bash
mmk-kb experiment-upload data.csv "Test" "Description"
# ⚠️ 5 sample codes not found in project: SAMPLE_001, SAMPLE_002...
```
*Solution*: Upload samples first or fix sample codes in CSV

**No Current Project:**
```bash
mmk-kb experiments
# ❌ No current project set
```
*Solution*: Use `mmk-kb use "PROJECT_CODE"` or specify `--project`

**CSV Format Issues:**
```bash
mmk-kb csv-preview invalid.csv
# ❌ CSV must contain a 'sample' column
```
*Solution*: Fix CSV format and retry

### Performance Considerations

**Large CSV Files:**
- System handles large files efficiently
- Memory usage scales with file size
- Progress reporting for bulk operations

**Database Growth:**
- Regular vacuum operations recommended
- Monitor database file size
- Use backup strategies for large datasets