# MMK Knowledge Base

A comprehensive Python-based biomarker data management system for storing, managing, and analyzing experimental data with a focus on biomarker research workflows.

## 🎯 Overview

MMK-KB provides:
- **Project-based organization** of research data
- **Sample management** with clinical metadata  
- **Biomarker experiment tracking** with versioning
- **ROC analysis and diagnostic modeling** for biomarker evaluation
- **CSV-based data import/export** workflows
- **Multi-environment database support** (development, staging, testing, production)
- **Full command-line interface** for all operations
- **Programmatic API** for integration with analysis pipelines

## 🚀 Quick Start

```bash
# Install
git clone <repository-url>
cd mmk-kb
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Basic usage
mmk-kb create "PROJ001" "My Study" "Description" "Creator"
mmk-kb use "PROJ001"
mmk-kb sample-upload samples.csv
mmk-kb experiment-upload data.csv "Experiment Name" "Description"

# ROC analysis
mmk-kb roc-run 1 "Analysis Name" 0.3 --max-combinations 3
mmk-kb roc-report 1 --top 10
```

## 📚 Documentation

| Topic | Description |
|-------|-------------|
| [Installation & Setup](docs/INSTALLATION.md) | Complete installation guide and environment setup |
| [Architecture & Implementation](docs/ARCHITECTURE.md) | System design, database schema, and implementation details |
| [Project Management](docs/PROJECT_MANAGEMENT.md) | Creating and managing research projects |
| [Sample Management](docs/SAMPLE_MANAGEMENT.md) | Working with clinical samples and metadata |
| [Sample CSV Upload](docs/SAMPLE_CSV_UPLOAD.md) | Bulk sample data upload via CSV |
| [Experiment Management](docs/EXPERIMENT_MANAGEMENT.md) | Managing biomarker experiments and measurements |
| [ROC Analysis](docs/ROC_ANALYSIS.md) | **NEW**: Comprehensive ROC analysis and diagnostic modeling |
| [CLI Reference](docs/CLI_REFERENCE.md) | Complete command-line interface documentation |
| [API Reference](docs/API_REFERENCE.md) | Programmatic usage and Python API |
| [Data Workflows](docs/WORKFLOWS.md) | Common research workflows and examples |
| [Environment Management](docs/ENVIRONMENTS.md) | Database environments and deployment |

## 🔧 Key Features

- **SQLite backend** with foreign key constraints and ACID compliance
- **Modular CLI system** with specialized command handlers
- **CSV processors** for bulk data import/export with validation
- **Biomarker versioning** for tracking different assay implementations
- **ROC analysis engine** with logistic regression and performance metrics
- **Environment isolation** for development, staging, and production
- **Comprehensive backup/restore** utilities

## 📊 ROC Analysis Features

- **Multi-biomarker modeling**: Test single biomarkers and combinations
- **Comprehensive metrics**: AUC, sensitivity, specificity, PPV, NPV
- **Multiple thresholds**: 97% sensitivity, 95% sensitivity, and optimal performance
- **Model storage**: Complete coefficients for future predictions
- **ROC curve data**: Full curve coordinates for plotting
- **Flexible analysis**: User-defined prevalence and combination limits

## 📊 Example Workflow

```bash
# 1. Create and use project
mmk-kb create "CANCER_2024" "Cancer Study" "Multi-center analysis" "Dr. Research"
mmk-kb use "CANCER_2024"

# 2. Upload sample data
mmk-kb sample-upload clinical_data.csv

# 3. Upload experiment data
mmk-kb experiment-upload cytokine_panel.csv "Cytokine Analysis" "Initial screening"

# 4. Run ROC analysis
mmk-kb roc-run 1 "Diagnostic Panel Study" 0.25 --max-combinations 3

# 5. Review results
mmk-kb roc-show 1
mmk-kb roc-report 1 --output results.csv --top 15
mmk-kb roc-model 5  # Examine specific model
```

## 🧪 Testing

```bash
pytest                    # Run all tests
pytest --cov=src/mmkkb   # Run with coverage
pytest tests/test_roc_analysis.py  # Test ROC analysis specifically
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

See [Contributing Guide](docs/CONTRIBUTING.md) for details.

## 📄 License

[Add your license information here]