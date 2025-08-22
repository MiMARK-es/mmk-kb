# API Reference

## Overview

MMK-KB provides a comprehensive Python API for programmatic access to all functionality. The API is organized into modules that mirror the CLI structure.

## Core Modules

### Configuration Management

```python
from mmkkb.config import Environment, set_environment, get_current_environment, get_database_path

# Environment management
set_environment(Environment.STAGING)
current_env = get_current_environment()
db_path = get_database_path(Environment.PRODUCTION)

# Available environments
Environment.DEVELOPMENT
Environment.STAGING  
Environment.TESTING
Environment.PRODUCTION
```

### Project Operations

```python
from mmkkb.projects import Project, ProjectDatabase

# Initialize database
project_db = ProjectDatabase()
# Or with custom path
project_db = ProjectDatabase("/path/to/database.db")

# Create project
project = Project(
    code="API_PROJECT",
    name="API Test Project", 
    description="Created via Python API",
    creator="Python Script"
)
created_project = project_db.create_project(project)

# Retrieve projects
project = project_db.get_project_by_code("API_PROJECT")
project_by_id = project_db.get_project_by_id(1)
all_projects = project_db.list_projects()

# Update project
project.description = "Updated description"
updated_project = project_db.update_project(project)

# Delete project
success = project_db.delete_project("API_PROJECT")
```

### Sample Operations

```python
from mmkkb.samples import Sample, SampleDatabase, CurrentProjectManager

# Initialize
sample_db = SampleDatabase()
project_manager = CurrentProjectManager()

# Set current project context
project_manager.use_project("PROJ001")
current_project_id = project_manager.get_current_project_id()

# Create sample
sample = Sample(
    code="API_SAMPLE_001",
    age=45,
    bmi=25.3,
    dx=True,  # Disease/case
    dx_origin="biopsy",
    collection_center="API Hospital",
    processing_time=120,
    project_id=current_project_id
)
created_sample = sample_db.create_sample(sample)

# Query samples
sample = sample_db.get_sample_by_code("API_SAMPLE_001", current_project_id)
sample_by_id = sample_db.get_sample_by_id(1)
project_samples = sample_db.list_samples(project_id=current_project_id)
all_samples = sample_db.list_samples()

# Update sample
sample.age = 46
sample.bmi = 25.5
updated_sample = sample_db.update_sample(sample)

# Delete sample
success = sample_db.delete_sample(sample.id)
success = sample_db.delete_sample_by_code("API_SAMPLE_001", current_project_id)

# Count samples
count = sample_db.count_samples(project_id=current_project_id)
```

### Experiment and Biomarker Operations

```python
from mmkkb.experiments import (
    ExperimentDatabase, Experiment, Biomarker, BiomarkerVersion, Measurement
)

# Initialize
experiment_db = ExperimentDatabase()

# Create biomarker
biomarker = Biomarker(
    name="IL-6",
    description="Interleukin-6 inflammatory cytokine",
    category="cytokine"
)
created_biomarker = experiment_db.create_biomarker(biomarker)

# Create biomarker version
biomarker_version = BiomarkerVersion(
    biomarker_id=created_biomarker.id,
    version="v2.0",
    description="High-sensitivity assay"
)
created_version = experiment_db.create_biomarker_version(biomarker_version)

# Convenience method for biomarker + version
biomarker_version = experiment_db.create_biomarker_with_version(
    biomarker_name="TNF-alpha",
    version="v1.0",
    biomarker_description="Tumor necrosis factor alpha",
    version_description="Standard assay",
    category="cytokine"
)

# Create experiment
experiment = Experiment(
    name="API Experiment",
    description="Created via Python API",
    project_id=project_id,
    csv_filename="api_data.csv"
)
created_experiment = experiment_db.create_experiment(experiment)

# Create measurement
measurement = Measurement(
    experiment_id=created_experiment.id,
    sample_id=sample_id,
    biomarker_version_id=biomarker_version.id,
    value=15.2
)
created_measurement = experiment_db.create_measurement(measurement)

# Query operations
biomarkers = experiment_db.list_biomarkers()
biomarker = experiment_db.get_biomarker_by_name("IL-6")
versions = experiment_db.list_biomarker_versions(biomarker_id=biomarker.id)
experiments = experiment_db.list_experiments(project_id=project_id)
measurements = experiment_db.get_measurements_by_experiment(experiment_id)

# Advanced analysis
analysis_data = experiment_db.get_biomarker_data_for_analysis(biomarker_id=1)
summary = experiment_db.get_measurement_summary(project_id=project_id)
```

## CSV Processing

### Sample CSV Processing

```python
from mmkkb.sample_csv_processor import SampleCSVProcessor

# Initialize
processor = SampleCSVProcessor()

# Validate CSV structure
is_valid, error_msg, required_columns = processor.validate_csv_structure("samples.csv")
if not is_valid:
    print(f"Validation error: {error_msg}")

# Preview CSV content
success, message, preview_data = processor.preview_csv("samples.csv", num_rows=5)
if success:
    print(f"Total rows: {preview_data['total_rows']}")
    print(f"Columns: {preview_data['columns']}")
    print(f"Preview: {preview_data['preview_rows']}")

# Process CSV upload
success, message, created_samples = processor.process_csv_upload(
    csv_path="samples.csv",
    project_id=project_id,
    skip_duplicates=True
)
print(f"Created {len(created_samples)} samples")

# Export samples to CSV
success, message = processor.export_samples_to_csv(
    project_id=project_id,
    output_path="exported_samples.csv"
)
```

### Experiment CSV Processing

```python
from mmkkb.csv_processor import CSVProcessor

# Initialize
processor = CSVProcessor()

# Validate experiment CSV
is_valid, error_msg, biomarker_columns = processor.validate_csv_structure("experiment.csv")
if is_valid:
    print(f"Found biomarkers: {biomarker_columns}")

# Preview experiment data
success, message, preview_data = processor.preview_csv("experiment.csv", num_rows=5)

# Process experiment upload
success, message, experiment = processor.process_csv_upload(
    csv_path="experiment.csv",
    experiment_name="API Experiment",
    experiment_description="Uploaded via API",
    project_id=project_id,
    biomarker_version="v1.0"
)

if success:
    print(f"Created experiment: {experiment.name}")
    print(message)

# Export experiment data to DataFrame
import pandas as pd
df = processor.get_experiment_data_as_dataframe(experiment_id=1)
if df is not None:
    print(df.head())
    # Save to CSV
    df.to_csv("experiment_export.csv", index=False)
```

## Database Utilities

```python
from mmkkb.db_utils import DatabaseUtils
from mmkkb.config import Environment

# Backup operations
backup_path = DatabaseUtils.backup_database(Environment.DEVELOPMENT)
print(f"Backup created: {backup_path}")

# Backup with custom settings
backup_path = DatabaseUtils.backup_database(
    source_env=Environment.STAGING,
    backup_dir="custom_backups",
    include_timestamp=True
)

# Restore database
success = DatabaseUtils.restore_database(
    backup_path="backup_file.db",
    target_env=Environment.DEVELOPMENT,
    confirm=False  # Skip confirmation
)

# Database maintenance
DatabaseUtils.vacuum_database(Environment.DEVELOPMENT)
DatabaseUtils.clean_database(Environment.TESTING, confirm=False)
cleaned_count = DatabaseUtils.clean_all_test_databases(confirm=False)

# Copy between environments
success = DatabaseUtils.copy_database(
    source_env=Environment.DEVELOPMENT,
    target_env=Environment.STAGING,
    confirm=False
)

# Get database status
status = DatabaseUtils.list_database_status()
for env_name, env_status in status.items():
    print(f"{env_name}: {env_status['project_count']} projects, {env_status['size_bytes']} bytes")
```

## Data Models

### Project Model

```python
@dataclass
class Project:
    code: str                           # Unique identifier
    name: str                          # Human-readable name  
    description: str                   # Project description
    creator: str                       # Creator name
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None

# Usage
project = Project(
    code="PROJ001",
    name="Test Project",
    description="Test description", 
    creator="Test User"
)
```

### Sample Model

```python
@dataclass
class Sample:
    code: str                          # Sample identifier
    age: int                           # Age in years (1-150)
    bmi: float                         # Body Mass Index (0-100)
    dx: bool                           # Diagnosis (True=disease, False=control)
    dx_origin: str                     # Diagnosis source
    collection_center: str             # Collection facility
    processing_time: int               # Processing time in minutes
    project_id: int                    # Parent project ID
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None

# Usage
sample = Sample(
    code="SAMPLE001",
    age=45,
    bmi=25.3,
    dx=True,  # Disease
    dx_origin="biopsy",
    collection_center="Hospital A",
    processing_time=120,
    project_id=1
)
```

### Experiment Models

```python
@dataclass
class Biomarker:
    name: str                          # Biomarker name (unique)
    description: Optional[str] = None   # Description
    category: Optional[str] = None      # Category (e.g., "cytokine")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None

@dataclass 
class BiomarkerVersion:
    biomarker_id: int                  # Parent biomarker ID
    version: str                       # Version identifier
    description: Optional[str] = None   # Version description
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None

@dataclass
class Experiment:
    name: str                          # Experiment name
    description: str                   # Experiment description
    project_id: int                    # Parent project ID
    upload_date: Optional[datetime] = None
    csv_filename: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None

@dataclass
class Measurement:
    experiment_id: int                 # Experiment ID
    sample_id: int                     # Sample ID
    biomarker_version_id: int          # Biomarker version ID
    value: float                       # Measurement value
    created_at: Optional[datetime] = None
    id: Optional[int] = None
```

## Error Handling

### Exception Types

```python
# Database connection errors
try:
    project_db = ProjectDatabase("/invalid/path.db")
except Exception as e:
    print(f"Database error: {e}")

# Validation errors
try:
    sample = Sample(
        code="TEST",
        age=200,  # Invalid age
        bmi=25.0,
        dx=True,
        dx_origin="test",
        collection_center="test", 
        processing_time=120,
        project_id=1
    )
    sample_db.create_sample(sample)
except Exception as e:
    print(f"Validation error: {e}")

# CSV processing errors
try:
    success, message, samples = processor.process_csv_upload(
        csv_path="invalid.csv",
        project_id=1
    )
    if not success:
        print(f"Upload failed: {message}")
except Exception as e:
    print(f"Processing error: {e}")
```

### Validation Helpers

```python
# Check if project exists
project = project_db.get_project_by_code("PROJ001")
if project is None:
    print("Project not found")

# Check if sample exists
sample = sample_db.get_sample_by_code("SAMPLE001", project_id)
if sample is None:
    print("Sample not found")

# Validate CSV before processing
is_valid, error_msg, columns = processor.validate_csv_structure("data.csv")
if not is_valid:
    print(f"Invalid CSV: {error_msg}")
    exit(1)
```

## Advanced Usage Patterns

### Batch Operations

```python
# Batch create samples
samples_data = [
    {"code": "CASE001", "age": 45, "bmi": 25.3, "dx": True, ...},
    {"code": "CASE002", "age": 52, "bmi": 28.1, "dx": True, ...},
    {"code": "CTRL001", "age": 48, "bmi": 26.0, "dx": False, ...},
]

created_samples = []
for sample_data in samples_data:
    sample = Sample(project_id=project_id, **sample_data)
    created_sample = sample_db.create_sample(sample)
    created_samples.append(created_sample)

print(f"Created {len(created_samples)} samples")
```

### Data Analysis Integration

```python
import pandas as pd
import numpy as np

# Get experiment data as DataFrame
df = processor.get_experiment_data_as_dataframe(experiment_id=1)

# Basic analysis
print(f"Shape: {df.shape}")
print(f"Biomarkers: {df.columns.tolist()[1:]}")  # Exclude 'sample' column

# Statistical analysis
biomarker_cols = [col for col in df.columns if col != 'sample']
stats = df[biomarker_cols].describe()
print(stats)

# Correlation analysis
correlation_matrix = df[biomarker_cols].corr()
print(correlation_matrix)
```

### Transaction-like Operations

```python
# Atomic operations with error handling
try:
    # Create project
    project = project_db.create_project(Project(...))
    
    # Upload samples
    success, message, samples = processor.process_csv_upload(...)
    if not success:
        raise Exception(f"Sample upload failed: {message}")
    
    # Upload experiments
    success, message, experiment = csv_processor.process_csv_upload(...)
    if not success:
        raise Exception(f"Experiment upload failed: {message}")
        
    print("All operations completed successfully")
    
except Exception as e:
    print(f"Operation failed: {e}")
    # Could implement rollback logic here
```

### Custom Analysis Functions

```python
def analyze_biomarker_trends(biomarker_name: str, project_id: int):
    """Custom analysis function for biomarker trends."""
    
    # Get biomarker data
    biomarker = experiment_db.get_biomarker_by_name(biomarker_name)
    if not biomarker:
        return None
        
    analysis_data = experiment_db.get_biomarker_data_for_analysis(biomarker.id)
    
    # Convert to DataFrame for analysis
    measurements = analysis_data['measurements']
    df = pd.DataFrame(measurements)
    
    # Calculate statistics by experiment
    experiment_stats = df.groupby('experiment_name')['value'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    
    return {
        'biomarker': analysis_data['biomarker'],
        'total_measurements': analysis_data['total_measurements'],
        'experiment_statistics': experiment_stats.to_dict('index'),
        'overall_mean': df['value'].mean(),
        'overall_std': df['value'].std()
    }

# Usage
results = analyze_biomarker_trends("IL-6", project_id=1)
if results:
    print(f"Analysis for {results['biomarker'].name}:")
    print(f"Total measurements: {results['total_measurements']}")
    print(f"Overall mean: {results['overall_mean']:.2f}")
```

## Integration Examples

### Jupyter Notebook Integration

```python
# Notebook cell 1: Setup
from mmkkb.projects import ProjectDatabase
from mmkkb.samples import SampleDatabase  
from mmkkb.experiments import ExperimentDatabase
from mmkkb.csv_processor import CSVProcessor
import pandas as pd
import matplotlib.pyplot as plt

# Initialize databases
project_db = ProjectDatabase()
sample_db = SampleDatabase()
experiment_db = ExperimentDatabase()

# Notebook cell 2: Load and explore data
project = project_db.get_project_by_code("STUDY_2024")
samples = sample_db.list_samples(project_id=project.id)
experiments = experiment_db.list_experiments(project_id=project.id)

print(f"Project: {project.name}")
print(f"Samples: {len(samples)}")
print(f"Experiments: {len(experiments)}")

# Notebook cell 3: Analysis and visualization
processor = CSVProcessor()
df = processor.get_experiment_data_as_dataframe(experiment_id=1)

# Plot biomarker distributions
biomarkers = [col for col in df.columns if col != 'sample']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, biomarker in enumerate(biomarkers[:4]):
    ax = axes[i//2, i%2]
    df[biomarker].hist(ax=ax, bins=20)
    ax.set_title(biomarker)
plt.tight_layout()
plt.show()
```

### Web Application Integration

```python
from flask import Flask, jsonify, request
from mmkkb.projects import ProjectDatabase
from mmkkb.samples import SampleDatabase

app = Flask(__name__)
project_db = ProjectDatabase()
sample_db = SampleDatabase()

@app.route('/api/projects')
def get_projects():
    projects = project_db.list_projects()
    return jsonify([{
        'code': p.code,
        'name': p.name,
        'creator': p.creator,
        'created_at': p.created_at.isoformat()
    } for p in projects])

@app.route('/api/projects/<project_code>/samples')
def get_project_samples(project_code):
    project = project_db.get_project_by_code(project_code)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
        
    samples = sample_db.list_samples(project_id=project.id)
    return jsonify([{
        'code': s.code,
        'age': s.age,
        'bmi': s.bmi,
        'dx': s.dx
    } for s in samples])

if __name__ == '__main__':
    app.run(debug=True)
```