# Sample Management

## Overview

Samples represent patient or specimen data with clinical metadata. Each sample belongs to a project and contains clinical attributes needed for biomarker analysis.

## Sample Data Model

Each sample contains:
- **Code**: Sample identifier (unique within project)
- **Age**: Patient age in years (1-150)
- **BMI**: Body Mass Index (0-100)
- **Diagnosis (dx)**: Disease status (0=benign/control, 1=disease/case)
- **Diagnosis Origin**: Source of diagnosis (e.g., "biopsy", "screening")
- **Collection Center**: Sample collection facility
- **Processing Time**: Processing time in minutes
- **Project**: Parent project (automatically linked)
- **Timestamps**: Creation and update tracking

## Individual Sample Operations

### Creating Samples

```bash
# Create individual sample
mmk-kb sample-create "CASE001" 45 25.3 1 "biopsy" "Hospital_A" 120

# Create in specific project
mmk-kb sample-create "CTRL001" 48 26.1 0 "screening" "Hospital_B" 85 --project "PROJ001"

# Create with disease designation
mmk-kb sample-create "SAMPLE001" 52 28.7 disease "pathology" "Lab_C" 90
```

**Parameters:**
- `code`: Sample identifier (string)
- `age`: Age in years (integer, 1-150)
- `bmi`: Body Mass Index (float, 0-100)
- `dx`: Diagnosis (0/1, false/true, benign/disease, control/case)
- `dx_origin`: Diagnosis source (string)
- `collection_center`: Collection facility (string)
- `processing_time`: Processing time in minutes (integer, ≥0)

### Viewing Samples

```bash
# Show specific sample details
mmk-kb sample-show "CASE001"

# Example output:
🧪 Sample Details:
   Code: CASE001
   Age: 45 years
   BMI: 25.3
   Diagnosis: Disease (1)
   Diagnosis Origin: biopsy
   Collection Center: Hospital_A
   Processing Time: 120 minutes
   Project: PROJ001 (Biomarker Study)
   Created: 2024-08-22 10:30:15
   Updated: 2024-08-22 10:30:15
```

### Listing Samples

```bash
# List samples in current project
mmk-kb samples

# List samples in specific project
mmk-kb samples --project "PROJ001"

# Example output:
🧪 Found 5 samples in project PROJ001:

Code      Age  BMI   Dx  Origin     Center      Time  Created           
----------------------------------------------------------------------
CASE001   45   25.3  1   biopsy     Hospital_A  120   2024-08-22 10:30
CASE002   52   28.7  1   pathology  Hospital_B  90    2024-08-22 10:31
CTRL001   48   26.1  0   screening  Hospital_A  85    2024-08-22 10:32
CTRL002   55   29.2  0   screening  Hospital_C  95    2024-08-22 10:33
CTRL003   43   24.8  0   screening  Hospital_A  80    2024-08-22 10:34
```

### Updating Samples

```bash
# Update sample attributes
mmk-kb sample-update "CASE001" --age 46 --bmi 25.5

# Update diagnosis
mmk-kb sample-update "CASE001" --dx 0  # Change to control

# Update multiple fields
mmk-kb sample-update "CASE001" --age 46 --bmi 25.5 --dx-origin "re-biopsy"
```

### Deleting Samples

```bash
# Delete sample with confirmation
mmk-kb sample-delete "CASE001"

# Example output:
⚠️  Delete sample 'CASE001'? This will also delete associated measurements. (y/N): y
✅ Sample 'CASE001' deleted successfully
```

## Bulk CSV Operations

For larger datasets, use CSV upload functionality for efficient bulk operations.

### CSV Format Requirements

Required columns (in any order):
- `code`: Sample identifier
- `age`: Patient age (1-150)
- `bmi`: Body Mass Index (0-100)
- `dx`: Diagnosis status
- `dx_origin`: Diagnosis source
- `collection_center`: Collection facility
- `processing_time`: Processing time in minutes

### Sample CSV Format

```csv
code,age,bmi,dx,dx_origin,collection_center,processing_time
CASE_001,45,25.3,1,biopsy,Hospital_A,120
CASE_002,52,28.7,disease,pathology,Hospital_B,90
CTRL_001,48,26.1,0,screening,Hospital_A,85
CTRL_002,55,29.2,control,screening,Hospital_C,95
CTRL_003,43,24.8,false,screening,Hospital_D,80
```

### Diagnosis Value Mapping

The `dx` column accepts various formats:
- **Disease/Case**: `1`, `true`, `disease`, `case` → `True`
- **Control/Benign**: `0`, `false`, `benign`, `control` → `False`

### CSV Upload Operations

```bash
# Preview CSV before upload
mmk-kb sample-preview samples.csv

# Preview with more rows
mmk-kb sample-preview samples.csv --rows 10

# Upload samples to current project
mmk-kb sample-upload samples.csv

# Upload to specific project
mmk-kb sample-upload samples.csv --project "PROJ001"

# Handle duplicates (default: skip)
mmk-kb sample-upload samples.csv --fail-on-duplicates
```

### CSV Export

```bash
# Export samples from current project
mmk-kb sample-export output_samples.csv

# Export from specific project
mmk-kb sample-export output_samples.csv --project "PROJ001"
```

## Data Validation

### Age Validation
- Must be numeric integer
- Range: 1-150 years
- Required field

### BMI Validation
- Must be numeric (float or integer)
- Range: 0-100
- Required field

### Diagnosis Validation
- Accepts: 0, 1, false, true, benign, disease, control, case
- Case-insensitive
- Required field

### Processing Time Validation
- Must be non-negative integer
- Represents minutes
- Required field

### Code Validation
- Must be unique within project
- Cannot be empty or whitespace-only
- Recommended: alphanumeric with underscores/hyphens

## Working with Current Project

### Setting Project Context

```bash
# Set current project for sample operations
mmk-kb use "PROJ001"

# Verify current project
mmk-kb current
```

### Project-Scoped Operations

Once a current project is set:
```bash
# These use current project automatically
mmk-kb samples
mmk-kb sample-create "NEW001" 45 25.0 1 "biopsy" "Hospital" 120
mmk-kb sample-upload data.csv
```

## Programmatic Usage

### Python API

```python
from mmkkb.samples import Sample, SampleDatabase, CurrentProjectManager
from mmkkb.projects import ProjectDatabase

# Initialize
sample_db = SampleDatabase()
project_db = ProjectDatabase()

# Get project
project = project_db.get_project_by_code("PROJ001")

# Create sample
sample = Sample(
    code="API_SAMPLE_001",
    age=45,
    bmi=25.3,
    dx=True,  # Disease
    dx_origin="biopsy",
    collection_center="API Hospital",
    processing_time=120,
    project_id=project.id
)
created_sample = sample_db.create_sample(sample)

# Query samples
samples = sample_db.list_samples(project_id=project.id)
sample = sample_db.get_sample_by_code("API_SAMPLE_001", project.id)

# Update sample
sample.age = 46
sample.bmi = 25.5
updated_sample = sample_db.update_sample(sample)

# Delete sample
success = sample_db.delete_sample_by_code("API_SAMPLE_001", project.id)

# Count samples
count = sample_db.count_samples(project_id=project.id)
```

### CSV Processing API

```python
from mmkkb.sample_csv_processor import SampleCSVProcessor

# Initialize processor
processor = SampleCSVProcessor()

# Validate CSV
is_valid, error_msg, columns = processor.validate_csv_structure("samples.csv")

# Preview CSV
success, message, preview_data = processor.preview_csv("samples.csv", num_rows=5)

# Process upload
success, message, created_samples = processor.process_csv_upload(
    csv_path="samples.csv",
    project_id=project.id,
    skip_duplicates=True
)

# Export samples
success, message = processor.export_samples_to_csv(
    project_id=project.id,
    output_path="exported_samples.csv"
)
```

## Integration with Experiments

Samples uploaded via CSV can be immediately used in experiment data uploads. The experiment CSV processor will automatically find and link to samples by their codes within the same project.

### Workflow Example

```bash
# 1. Upload samples
mmk-kb sample-upload clinical_data.csv

# 2. Verify samples
mmk-kb samples

# 3. Upload experiment data (references sample codes)
mmk-kb experiment-upload biomarker_data.csv "Study 1" "Cytokine analysis"

# 4. System automatically links samples to measurements
```

## Error Handling

### Common Upload Errors

**Missing Required Columns:**
```
❌ Missing required columns: age, dx
```

**Invalid Data Types:**
```
❌ Age column contains non-numeric values: "forty-five"
❌ BMI column contains non-numeric values: "normal"
```

**Range Validation Errors:**
```
❌ Invalid age value: 200. Age must be between 1 and 150.
❌ Invalid BMI value: -5.0. BMI must be between 0 and 100.
```

**Duplicate Sample Codes:**
```
⚠️ 3 samples skipped (duplicates): CASE_001, CASE_002, CTRL_001
```

### Recovery Strategies

**Fix CSV Data:**
- Correct data types and ranges
- Ensure required columns exist
- Validate diagnosis values

**Handle Duplicates:**
- Use `--fail-on-duplicates` for strict validation
- Default behavior skips duplicates
- Update existing samples via programmatic API

## Best Practices

### Sample Naming
- Use consistent naming conventions
- Include case/control indicators
- Use meaningful prefixes (CASE_, CTRL_, PATIENT_)

### Data Quality
- Validate data before upload
- Use preview functionality
- Maintain data dictionaries
- Document collection protocols

### Backup Strategy
- Backup before bulk uploads
- Test with small datasets first
- Use staging environment for validation

## Troubleshooting

### Sample Not Found
```bash
mmk-kb sample-show "MISSING"
# Error: Sample with code 'MISSING' not found in current project
```
*Solution*: Check sample code or project context

### No Current Project
```bash
mmk-kb samples
# Error: No current project set
```
*Solution*: Use `mmk-kb use "PROJECT_CODE"` or specify `--project`

### CSV Validation Errors
```bash
mmk-kb sample-upload invalid.csv
# Error: Missing required columns: age, dx
```
*Solution*: Fix CSV format and retry upload