# Project Management

## Overview

Projects are the top-level organizational units in MMK-KB that contain samples and experiments. Each project has a unique code, metadata, and serves as a container for related research data.

## Project Structure

Each project contains:
- **Unique Code**: User-defined identifier (e.g., "CANCER_2024")
- **Name**: Human-readable project name
- **Description**: Detailed project description
- **Creator**: Project creator name
- **Timestamps**: Automatic creation and update tracking

## Basic Operations

### Creating Projects

```bash
# Create a new project
mmk-kb create "PROJ001" "Biomarker Study" "IL-6 analysis in cancer patients" "Dr. Smith"

# Create with complex names (use quotes)
mmk-kb create "CANCER_2024" "Multi-center Cancer Study" "Comprehensive biomarker analysis across multiple sites" "Research Team"
```

### Listing Projects

```bash
# List all projects
mmk-kb list

# Example output:
📋 Found 3 projects:

Code         Name                     Creator         Created            
------------------------------------------------------------------------
CANCER_2024  Multi-center Cancer      Research Team   2024-08-22 10:30
PROJ001      Biomarker Study          Dr. Smith       2024-08-21 15:45
VALID_001    Validation Study         Dr. Brown       2024-08-20 09:15
```

### Viewing Project Details

```bash
# Show detailed project information
mmk-kb show "PROJ001"

# Example output:
📋 Project Details:
   Code: PROJ001
   Name: Biomarker Study
   Description: IL-6 analysis in cancer patients
   Creator: Dr. Smith
   Created: 2024-08-21 15:45:30
   Updated: 2024-08-21 15:45:30
   
📊 Project Statistics:
   Samples: 45
   Experiments: 3
   Biomarkers: 12
   Measurements: 540
```

### Updating Projects

```bash
# Update project details (programmatic API only)
# CLI updates coming in future versions
```

### Deleting Projects

```bash
# Delete a project (with confirmation)
mmk-kb delete "PROJ001"

# Example output:
⚠️  This will permanently delete project 'PROJ001' and ALL associated data:
   - 45 samples
   - 3 experiments  
   - 540 measurements
   
   Type 'PROJ001' to confirm deletion: PROJ001
✅ Project 'PROJ001' deleted successfully
```

## Current Project Management

### Setting Current Project

For convenience, you can set a "current" project to avoid specifying the project code in every command:

```bash
# Set current project
mmk-kb use "PROJ001"

# Verify current project
mmk-kb current

# Example output:
📋 Current Project: PROJ001 (Biomarker Study)
```

### Using Current Project

Once set, sample and experiment commands will use the current project by default:

```bash
# These commands use the current project
mmk-kb samples                    # List samples in current project
mmk-kb experiments               # List experiments in current project
mmk-kb sample-upload data.csv    # Upload to current project
```

### Clearing Current Project

```bash
# Clear current project setting
mmk-kb clear

# Verify
mmk-kb current
# Output: No current project set
```

## Working with Multiple Projects

### Project-Specific Commands

You can always specify a project explicitly, even when a current project is set:

```bash
# Work with specific project
mmk-kb samples --project "OTHER_PROJ"
mmk-kb experiments --project "VALIDATION_001"
mmk-kb sample-upload data.csv --project "PROJ001"
```

### Environment-Specific Projects

Projects are isolated by environment. The same project code can exist in different environments:

```bash
# Development environment
mmk-kb create "TEST_001" "Test Project" "Development testing" "Dev Team"

# Switch to staging
mmk-kb setenv staging
mmk-kb create "TEST_001" "Test Project" "Staging testing" "QA Team"

# These are separate projects in different databases
```

## Project Validation and Constraints

### Code Requirements
- **Unique**: Project codes must be unique within an environment
- **Non-empty**: Cannot be empty or whitespace-only
- **Recommended format**: Use alphanumeric characters, underscores, hyphens

### Field Validation
- **Name**: Required, non-empty string
- **Description**: Required, non-empty string  
- **Creator**: Required, non-empty string

### Example Valid Projects
```bash
mmk-kb create "CANCER_2024" "Cancer Study" "Biomarker analysis" "Dr. Smith"
mmk-kb create "PILOT-001" "Pilot Study" "Initial validation" "Research Team"
mmk-kb create "VALIDATION_v2" "Validation Study v2" "Second validation round" "QA Team"
```

## Programmatic Usage

### Python API

```python
from mmkkb.projects import Project, ProjectDatabase
from mmkkb.config import set_environment, Environment

# Set environment
set_environment(Environment.DEVELOPMENT)

# Initialize database
project_db = ProjectDatabase()

# Create project
project = Project(
    code="API_001",
    name="API Test Project",
    description="Created via Python API",
    creator="Python Script"
)
created_project = project_db.create_project(project)

# Retrieve project
project = project_db.get_project_by_code("API_001")
print(f"Project: {project.name}, Created: {project.created_at}")

# List all projects
projects = project_db.list_projects()
for p in projects:
    print(f"{p.code}: {p.name}")

# Update project
project.description = "Updated description"
updated_project = project_db.update_project(project)

# Delete project
success = project_db.delete_project("API_001")
```

## Best Practices

### Project Naming
- Use descriptive, meaningful project codes
- Include year or version for longitudinal studies
- Use consistent naming conventions across your organization

```bash
# Good examples
mmk-kb create "BREAST_CANCER_2024" "Breast Cancer Biomarkers" "..." "..."
mmk-kb create "VALIDATION_COHORT_2" "Validation Cohort 2" "..." "..."
mmk-kb create "PILOT_IL6_STUDY" "IL-6 Pilot Study" "..." "..."

# Avoid
mmk-kb create "proj1" "project" "test" "user"
```

### Project Organization
- One project per distinct research study
- Group related experiments within the same project
- Use environments to separate development, testing, and production data

### Data Management
- Always backup before major operations
- Use staging environment for testing workflows
- Document project purpose in the description field

## Troubleshooting

### Common Issues

**Project Code Already Exists:**
```bash
mmk-kb create "EXISTING" "..." "..." "..."
# Error: Project with code 'EXISTING' already exists
```
*Solution*: Choose a different project code or delete the existing project

**No Current Project Set:**
```bash
mmk-kb samples
# Error: No project specified and no current project set
```
*Solution*: Use `mmk-kb use "PROJECT_CODE"` or specify `--project` flag

**Project Not Found:**
```bash
mmk-kb show "NONEXISTENT"
# Error: Project with code 'NONEXISTENT' not found
```
*Solution*: Check project code spelling or list projects with `mmk-kb list`

### Recovery Operations

**Restore Deleted Project:**
If you have a backup, you can restore it:
```bash
mmk-kb restore backup_file.db
```

**List Projects in All Environments:**
```bash
mmk-kb --env development list
mmk-kb --env staging list
mmk-kb --env production list
```