# MMK Knowledge Base

A comprehensive Python-based biomarker data management system for storing, managing, and analyzing experimental data with a focus on biomarker research workflows.

## 🎯 Overview

MMK-KB provides:
- **Project-based organization** of research data
- **Sample management** with clinical metadata  
- **Biomarker experiment tracking** with versioning
- **ROC analysis and diagnostic modeling** for biomarker evaluation
- **ROC normalized analysis** for ratio-based biomarker diagnostics
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

# ROC normalized analysis (NEW)
mmk-kb roc-norm-run 1 5 "Normalized Analysis" 0.3 --max-combinations 2
mmk-kb roc-norm-report 1 --top 10
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
| [ROC Analysis](docs/ROC_ANALYSIS.md) | Comprehensive ROC analysis and diagnostic modeling |
| [ROC Normalized Analysis](docs/ROC_NORMALIZED_ANALYSIS.md) | **NEW**: Ratio-based biomarker analysis with normalization |
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
- **ROC normalized analysis** for ratio-based diagnostics and reference standardization
- **Environment isolation** for development, staging, and production
- **Comprehensive backup/restore** utilities

## 📊 Analysis Features

### ROC Analysis
- **Multi-biomarker modeling**: Test single biomarkers and combinations
- **Comprehensive metrics**: AUC, sensitivity, specificity, PPV, NPV
- **Multiple thresholds**: 97% sensitivity, 95% sensitivity, and optimal performance
- **Model storage**: Complete coefficients for future predictions
- **ROC curve data**: Full curve coordinates for plotting
- **Flexible analysis**: User-defined prevalence and combination limits

### ROC Normalized Analysis ⭐ NEW
- **Ratio-based analysis**: Normalize biomarkers against a reference biomarker
- **Reference standardization**: Control for variations in housekeeping biomarkers
- **Same comprehensive metrics**: All ROC analysis features applied to normalized ratios
- **Normalizer tracking**: Full traceability of which biomarker was used for normalization
- **Clinical applications**: Ideal for protein ratios, gene expression normalization, etc.

## 📊 Example Workflow

```bash
# 1. Create and use project
mmk-kb create "CANCER_2024" "Cancer Study" "Multi-center analysis" "Dr. Research"
mmk-kb use "CANCER_2024"

# 2. Upload sample data
mmk-kb sample-upload clinical_data.csv

# 3. Upload experiment data
mmk-kb experiment-upload cytokine_panel.csv "Cytokine Analysis" "Initial screening"

# 4. Run standard ROC analysis
mmk-kb roc-run 1 "Diagnostic Panel Study" 0.25 --max-combinations 3

# 5. Run normalized ROC analysis (using total protein as normalizer)
mmk-kb biomarker-versions --experiment 1  # Find total protein biomarker version ID
mmk-kb roc-norm-run 1 8 "Protein Ratio Analysis" 0.25 --max-combinations 2

# 6. Review results
mmk-kb roc-show 1
mmk-kb roc-norm-show 1
mmk-kb roc-report 1 --output standard_results.csv --top 15
mmk-kb roc-norm-report 1 --output normalized_results.csv --top 15
```

## 🧪 Testing

```bash
pytest                    # Run all tests
pytest --cov=src/mmkkb   # Run with coverage
pytest tests/test_roc_analysis.py  # Test ROC analysis specifically
pytest tests/test_roc_normalized_analysis.py  # Test ROC normalized analysis
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

See [Contributing Guide](docs/CONTRIBUTING.md) for details.

## 📄 License

[Add your license information here]