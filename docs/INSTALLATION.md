# Installation & Setup

## Prerequisites

- **Python 3.12+** - Required for modern Python features
- **pip** - Package installer for Python
- **Git** - For cloning the repository

## Installation Steps

### 1. Clone Repository
```bash
git clone <repository-url>
cd mmk-kb
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 4. Verify Installation
```bash
# Test CLI is working
mmk-kb --help

# Check environment status
mmk-kb env
```

## Environment Variables

Set optional environment variables for configuration:

```bash
# Set default database environment
export MMK_KB_ENV=development  # Options: development, staging, testing, production

# Set custom database base directory (optional)
export MMK_KB_BASE_DIR=/path/to/databases
```

## Database Initialization

Databases are created automatically when first accessed:

```bash
# Initialize development database (default)
mmk-kb list

# Initialize specific environment
mmk-kb --env staging list
```

## Troubleshooting

### Common Issues

**Python Version Error:**
```bash
# Check Python version
python --version
# Ensure you're using Python 3.12+
```

**Module Not Found:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
# Reinstall in development mode
pip install -e .
```

**Permission Errors:**
```bash
# Check file permissions
ls -la mmk_kb*.db
# Ensure write permissions to directory
```

### Development Setup

For development work:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

## Next Steps

After installation:
1. Read [Project Management](PROJECT_MANAGEMENT.md) to create your first project
2. Follow [Data Workflows](WORKFLOWS.md) for common usage patterns
3. Check [CLI Reference](CLI_REFERENCE.md) for complete command documentation