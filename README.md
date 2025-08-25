# MMK Knowledge Base

A comprehensive Python-based biomarker data management system for storing, managing, and analyzing experimental data with a focus on biomarker research workflows.

## 🎯 Overview

MMK-KB provides:
- **Project-based organization** of research data
- **Sample management** with clinical metadata  
- **Biomarker experiment tracking** with versioning
- **ROC analysis and diagnostic modeling** with cross-validation support
- **ROC normalized analysis** for ratio-based biomarker diagnostics
- **CSV-based data import/export** workflows
- **Multi-environment database support** (development, staging, testing, production)
- **Modular command-line interface** with grouped analysis commands
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

# ROC analysis (new grouped command structure)
mmk-kb analysis roc-run 1 "Analysis Name" 0.3 --max-combinations 3
mmk-kb analysis roc-report 1 --top 10

# ROC analysis with cross-validation
mmk-kb analysis roc-run 1 "CV Analysis" 0.3 --enable-cv --bootstrap-iterations 500

# ROC normalized analysis
mmk-kb analysis roc-norm-run 1 5 "Normalized Analysis" 0.3 --max-combinations 2
mmk-kb analysis roc-norm-report 1 --top 10
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
| [ROC Normalized Analysis](docs/ROC_NORMALIZED_ANALYSIS.md) | Ratio-based biomarker analysis with normalization |
| [Cross-Validation](docs/CROSS_VALIDATION.md) | **NEW**: Cross-validation features for robust model evaluation |
| [CLI Reference](docs/CLI_REFERENCE.md) | Complete command-line interface documentation |
| [API Reference](docs/API_REFERENCE.md) | Programmatic usage and Python API |
| [Data Workflows](docs/WORKFLOWS.md) | Common research workflows and examples |
| [Environment Management](docs/ENVIRONMENTS.md) | Database environments and deployment |

## 🔧 Key Features

- **SQLite backend** with foreign key constraints and ACID compliance
- **Modular CLI system** with grouped analysis commands
- **Cross-validation support** with LOO and Bootstrap methods
- **CSV processors** for bulk data import/export with validation
- **Biomarker versioning** for tracking different assay implementations
- **ROC analysis engine** with logistic regression and performance metrics
- **ROC normalized analysis** for ratio-based diagnostics and reference standardization
- **Environment isolation** for development, staging, and production
- **Comprehensive backup/restore** utilities

## 📊 Analysis Features

### ROC Analysis with Cross-Validation ⭐ ENHANCED
- **Multi-biomarker modeling**: Test single biomarkers and combinations
- **Cross-validation support**: Leave-One-Out (LOO) and Bootstrap validation
- **Comprehensive metrics**: AUC, sensitivity, specificity, PPV, NPV with CV statistics
- **Multiple thresholds**: 97% sensitivity, 95% sensitivity, and optimal performance
- **Model storage**: Complete coefficients for future predictions
- **ROC curve data**: Full curve coordinates for plotting
- **Flexible analysis**: User-defined prevalence and combination limits

### ROC Normalized Analysis with Cross-Validation ⭐ ENHANCED
- **Ratio-based analysis**: Normalize biomarkers against a reference biomarker
- **Cross-validation support**: All CV features available for normalized analyses
- **Reference standardization**: Control for variations in housekeeping biomarkers
- **Same comprehensive metrics**: All ROC analysis features applied to normalized ratios
- **Normalizer tracking**: Full traceability of which biomarker was used for normalization
- **Clinical applications**: Ideal for protein ratios, gene expression normalization, etc.

### New Command Structure
- **Grouped commands**: All analysis commands under `mmk-kb analysis`
- **Modular design**: Easy to add new analysis types
- **Backward compatibility**: Legacy commands still supported
- **Help organization**: Cleaner help output with logical grouping

## 📊 Example Workflow

```bash
# 1. Create and use project
mmk-kb create "CANCER_2024" "Cancer Study" "Multi-center analysis" "Dr. Research"
mmk-kb use "CANCER_2024"

# 2. Upload sample data
mmk-kb sample-upload clinical_data.csv

# 3. Upload experiment data
mmk-kb experiment-upload cytokine_panel.csv "Cytokine Analysis" "Initial screening"

# 4. Run standard ROC analysis with cross-validation
mmk-kb analysis roc-run 1 "Diagnostic Panel Study" 0.25 --max-combinations 3 --enable-cv

# 5. Run normalized ROC analysis with custom CV parameters
mmk-kb biomarker-versions --experiment 1  # Find total protein biomarker version ID
mmk-kb analysis roc-norm-run 1 8 "Protein Ratio Analysis" 0.25 --max-combinations 2 \
  --enable-cv --bootstrap-iterations 500

# 6. Review results with cross-validation metrics
mmk-kb analysis roc-show 1
mmk-kb analysis roc-norm-show 1
mmk-kb analysis roc-report 1 --output cv_results.csv --top 15
mmk-kb analysis roc-norm-report 1 --output normalized_cv_results.csv --top 15
```

## 🧪 Testing

```bash
pytest                    # Run all tests
pytest --cov=src/mmkkb   # Run with coverage
pytest tests/test_roc_analysis.py  # Test ROC analysis specifically
pytest tests/test_cross_validation.py  # Test cross-validation features
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