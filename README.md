# MMK Knowledge Base

A Python-based knowledge management system for storing and managing projects with SQLite database backend.

## Features

- **Project Management**: CRUD operations for projects with unique codes
- **SQLite Database**: Lightweight, file-based database storage
- **Environment Management**: Support for development, staging, testing, and production databases
- **Database Utilities**: Backup, restore, cleanup, and management tools
- **Command Line Interface**: Full-featured CLI for all operations
- **Full Test Coverage**: Comprehensive pytest test suite
- **Clean Architecture**: Well-structured codebase following Python best practices

## Project Structure

```
mmk-kb/
├── src/mmkkb/           # Main package
│   ├── __init__.py
│   ├── projects.py      # Project model and database operations
│   ├── config.py        # Environment configuration
│   ├── db_utils.py      # Database utilities
│   └── cli.py           # Command line interface
├── tests/               # Test suite
│   └── test_projects.py
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## Project Model

Each project contains:
- **code**: Unique identifier (user-defined string)
- **name**: Project name
- **description**: Project description
- **creator**: Project creator name
- **created_at**: Automatic timestamp
- **updated_at**: Automatic timestamp

## Setup

### Prerequisites
- Python 3.12
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mmk-kb
```

2. Create and activate virtual environment:
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Command Line Interface

The CLI provides comprehensive project and database management capabilities:

```bash
# Show all available commands
mmk-kb --help

# Project operations
mmk-kb list                                    # List all projects
mmk-kb create "PRJ001" "My Project" "Description" "Creator"  # Create project
mmk-kb show "PRJ001"                          # Show project details
mmk-kb delete "PRJ001"                        # Delete project

# Environment management
mmk-kb env                                     # Show environment status
mmk-kb setenv staging                          # Set current environment
mmk-kb --env production list                   # Use specific environment for command

# Database management
mmk-kb backup                                  # Backup current environment database
mmk-kb backup --env staging --dir backups     # Backup specific environment
mmk-kb restore backup_file.db                 # Restore from backup
mmk-kb clean --env staging                     # Clean staging database
mmk-kb clean-tests                             # Clean all test databases
mmk-kb copy development staging                # Copy between environments
mmk-kb vacuum                                  # Optimize database
```

### Programmatic Usage

```python
from mmkkb.projects import Project, ProjectDatabase
from mmkkb.config import Environment, set_environment

# Set environment (optional - defaults to development)
set_environment(Environment.STAGING)

# Initialize database (uses current environment)
db = ProjectDatabase()

# Create a project
project = Project(
    code="PRJ001",
    name="My First Project",
    description="This is a test project",
    creator="John Doe"
)
created_project = db.create_project(project)

# Read projects
project = db.get_project_by_code("PRJ001")
all_projects = db.list_projects()

# Update project
project.description = "Updated description"
updated_project = db.update_project(project)

# Delete project
success = db.delete_project("PRJ001")
```

### Environment Management

The system supports multiple database environments:

- **development**: Default environment for local development (`mmk_kb.db`)
- **staging**: For testing and examples (`mmk_kb_staging.db`)
- **testing**: For automated tests (`test_mmk_kb.db`)
- **production**: For production use (`mmk_kb_production.db`)

Set environment via:
```bash
export MMK_KB_ENV=staging  # Environment variable
mmk-kb setenv staging      # CLI command
```

### Database Utilities

```python
from mmkkb.db_utils import DatabaseUtils
from mmkkb.config import Environment

# Backup operations
backup_path = DatabaseUtils.backup_database(Environment.STAGING)
DatabaseUtils.restore_database(backup_path, Environment.DEVELOPMENT)

# Cleanup operations
DatabaseUtils.clean_database(Environment.STAGING)
DatabaseUtils.clean_all_test_databases()

# Database management
DatabaseUtils.copy_database(Environment.DEVELOPMENT, Environment.STAGING)
DatabaseUtils.vacuum_database(Environment.DEVELOPMENT)

# Status information
status = DatabaseUtils.list_database_status()
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/mmkkb

# Run specific test file
pytest tests/test_projects.py

# Run with verbose output
pytest -v
```

## Development

### Code Formatting

The project uses black and isort for code formatting:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/
```

### Database

The SQLite database file (`mmk_kb.db`) will be created automatically in the project root when you first use the database operations.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

[Add your license here]