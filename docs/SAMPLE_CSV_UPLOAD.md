# Sample CSV Upload Functionality

This document explains how to use the CSV upload functionality for samples in the MMK Knowledge Base.

## CSV Format Requirements

The sample CSV file must contain the following columns in any order:

- **code**: Unique sample identifier (string)
- **age**: Patient age in years (integer, 1-150)
- **bmi**: Body Mass Index (float, 0-100)
- **dx**: Diagnosis status (0/1, false/true, benign/disease, control/case)
- **dx_origin**: Source of diagnosis (string, e.g., "biopsy", "screening", "pathology")
- **collection_center**: Sample collection center (string, e.g., "Hospital_A")
- **processing_time**: Processing time in minutes (integer, ≥0)

## Sample CSV Format

```csv
code,age,bmi,dx,dx_origin,collection_center,processing_time
CASE_001,45,25.3,1,biopsy,Hospital_A,120
CASE_002,52,28.7,disease,pathology,Hospital_B,90
CTRL_001,48,26.1,0,screening,Hospital_A,85
CTRL_002,55,29.2,control,screening,Hospital_C,95
```

## Available Commands

### Preview CSV File
Preview the structure and content of a CSV file before uploading:

```bash
mmk-kb sample-preview path/to/samples.csv
mmk-kb sample-preview path/to/samples.csv --rows 10  # Preview more rows
```

### Upload Samples from CSV
Upload samples from a CSV file to the current or specified project:

```bash
# Upload to current project
mmk-kb sample-upload path/to/samples.csv

# Upload to specific project
mmk-kb sample-upload path/to/samples.csv --project PROJECT_CODE

# Fail on duplicate sample codes (default: skip duplicates)
mmk-kb sample-upload path/to/samples.csv --fail-on-duplicates
```

### Export Samples to CSV
Export existing samples from a project to a CSV file:

```bash
# Export from current project
mmk-kb sample-export output_samples.csv

# Export from specific project
mmk-kb sample-export output_samples.csv --project PROJECT_CODE
```

## Diagnosis (dx) Value Mapping

The `dx` column accepts various formats that are automatically converted:

- **Disease/Case**: `1`, `true`, `disease`, `case` → `True`
- **Control/Benign**: `0`, `false`, `benign`, `control` → `False`

## Error Handling

The system provides comprehensive validation and error reporting:

- **Missing columns**: Reports which required columns are missing
- **Invalid data types**: Validates age, BMI, and processing_time are numeric
- **Range validation**: Ensures age (1-150), BMI (0-100), processing_time (≥0)
- **Duplicate handling**: Can skip or report duplicate sample codes
- **Invalid dx values**: Reports unrecognized diagnosis values

## Example Workflow

1. **Prepare your CSV file** with the required columns and format
2. **Preview the file** to check structure and data distribution:
   ```bash
   mmk-kb sample-preview my_samples.csv
   ```
3. **Set current project** (if not already set):
   ```bash
   mmk-kb use MY_PROJECT
   ```
4. **Upload the samples**:
   ```bash
   mmk-kb sample-upload my_samples.csv
   ```
5. **Verify upload** by listing samples:
   ```bash
   mmk-kb samples
   ```

## Sample Files

Example CSV files are available in the `experiment_data/` directory:
- `sample_data_example.csv`: Complete sample data with all required fields

## Integration with Experiments

Samples uploaded via CSV can be used immediately in experiment data uploads. The experiment CSV upload functionality will automatically find and link to samples by their codes within the same project.