# Experiment Data Directory

This directory contains example datasets for testing the MMK Knowledge Base experiment functionality.

## Dataset Overview

### Example Data Folder (example_data/)

Contains three biomarker experiments with realistic case/control distributions suitable for AUC analysis:

#### 1. biomarker_study_1.csv
- **Samples**: 50 (25 cases: CASE_001-CASE_025, 25 controls: CTRL_001-CTRL_025)
- **Biomarkers**: 5 biomarkers (Biomarker_A through Biomarker_E)
- **Expected AUCs**: Range from 0.55 to 0.85 (different discriminatory powers)

#### 2. biomarker_study_2.csv  
- **Samples**: 41 (21 cases: CASE_010-CASE_030, 20 controls: CTRL_010-CTRL_029)
- **Biomarkers**: 4 biomarkers (Biomarker_F, G, H, and Biomarker_A)
- **Note**: Some sample overlap with study 1, Biomarker_A measured with different version

#### 3. biomarker_study_3.csv
- **Samples**: 30 (15 cases: CASE_001-CASE_015, 15 controls: CTRL_001-CTRL_015)
- **Biomarkers**: 4 biomarkers (Biomarker_I, J, K, and Biomarker_B)
- **Note**: High-discrimination biomarkers, some sample overlap with study 1

## Key Features for Testing

1. **Sample Overlap**: Some samples appear in multiple experiments
2. **Biomarker Versions**: Same biomarkers (A, B) measured in different experiments with different versions
3. **AUC Analysis Ready**: Data generated with realistic case/control distributions
4. **Case/Control Labels**: Clear sample naming convention for outcome analysis

## Sample Code Convention

- **CASE_XXX**: Disease case samples
- **CTRL_XXX**: Healthy control samples
- Sample numbers range from 001 to 030 across all experiments

## Data Generation Details

- Controls: Lower biomarker concentrations with normal biological variation
- Cases: Elevated biomarker concentrations with disease-related variation  
- Different effect sizes to simulate various diagnostic performances
- Positive values ensured for all biomarker measurements

## Usage with MMK-KB

1. Create a project for biomarker studies
2. Add samples using codes from sample_information.csv
3. Upload experiments using the CSV files with different biomarker versions:
   - Study 1: version "RUO" 
   - Study 2: version "proprietary"
   - Study 3: version "clinical"
4. Analyze biomarker performance across experiments and versions
