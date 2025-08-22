"""
CSV processing utilities for experiment data uploads.
Handles parsing CSV files with sample columns and biomarker measurements.
"""
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from .experiments import ExperimentDatabase, Experiment, Biomarker, BiomarkerVersion, Measurement
from .samples import SampleDatabase


class CSVProcessor:
    """Processes CSV files containing experiment data."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize CSV processor with database connections."""
        self.experiment_db = ExperimentDatabase(db_path)
        self.sample_db = SampleDatabase(db_path)
    
    def validate_csv_structure(self, csv_path: str) -> Tuple[bool, str, List[str]]:
        """
        Validate CSV structure and return biomarker columns.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (is_valid, error_message, biomarker_columns)
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Check if 'sample' column exists
            if 'sample' not in df.columns:
                return False, "CSV must contain a 'sample' column", []
            
            # Get biomarker columns (all columns except 'sample')
            biomarker_columns = [col for col in df.columns if col != 'sample']
            
            if not biomarker_columns:
                return False, "CSV must contain at least one biomarker column", []
            
            # Check for empty values in sample column
            if df['sample'].isna().any():
                return False, "Sample column cannot contain empty values", []
            
            # Check for non-numeric values in biomarker columns
            for col in biomarker_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # Try to convert to numeric, allowing for some NaN values
                    try:
                        pd.to_numeric(df[col], errors='coerce')
                    except:
                        return False, f"Column '{col}' contains non-numeric values", []
            
            return True, "", biomarker_columns
            
        except Exception as e:
            return False, f"Error reading CSV: {str(e)}", []
    
    def process_csv_upload(
        self, 
        csv_path: str, 
        experiment_name: str,
        experiment_description: str,
        project_id: int,
        biomarker_version: str = "v1.0",
        sample_code_column: str = "sample"
    ) -> Tuple[bool, str, Optional[Experiment]]:
        """
        Process CSV upload and create experiment with measurements.
        
        Args:
            csv_path: Path to CSV file
            experiment_name: Name for the experiment
            experiment_description: Description for the experiment
            project_id: ID of the project this experiment belongs to
            biomarker_version: Version identifier for biomarkers in this experiment
            sample_code_column: Name of the column containing sample codes
            
        Returns:
            Tuple of (success, message, experiment_object)
        """
        # Validate CSV structure
        is_valid, error_msg, biomarker_columns = self.validate_csv_structure(csv_path)
        if not is_valid:
            return False, error_msg, None
        
        try:
            # Read CSV data
            df = pd.read_csv(csv_path)
            csv_filename = Path(csv_path).name
            
            # Create experiment
            experiment = Experiment(
                name=experiment_name,
                description=experiment_description,
                project_id=project_id,
                csv_filename=csv_filename
            )
            created_experiment = self.experiment_db.create_experiment(experiment)
            
            # Process biomarkers and measurements
            measurements_created = 0
            samples_not_found = []
            biomarker_versions_created = {}
            
            # Create biomarker versions (this will create biomarkers if they don't exist)
            for biomarker_name in biomarker_columns:
                biomarker_version_obj = self.experiment_db.create_biomarker_with_version(
                    biomarker_name=biomarker_name,
                    version=biomarker_version,
                    biomarker_description=f"Biomarker from experiment {experiment_name}",
                    version_description=f"Version {biomarker_version} used in {experiment_name}"
                )
                biomarker_versions_created[biomarker_name] = biomarker_version_obj
            
            # Process each row (sample)
            for _, row in df.iterrows():
                sample_code = str(row[sample_code_column]).strip()
                
                # Find sample in database
                sample = self.sample_db.get_sample_by_code(sample_code, project_id)
                if not sample:
                    samples_not_found.append(sample_code)
                    continue
                
                # Create measurements for each biomarker
                for biomarker_name in biomarker_columns:
                    value = row[biomarker_name]
                    
                    # Skip NaN values
                    if pd.isna(value):
                        continue
                    
                    measurement = Measurement(
                        experiment_id=created_experiment.id,
                        sample_id=sample.id,
                        biomarker_version_id=biomarker_versions_created[biomarker_name].id,
                        value=float(value)
                    )
                    
                    try:
                        self.experiment_db.create_measurement(measurement)
                        measurements_created += 1
                    except Exception as e:
                        # Handle duplicate measurements (same experiment, sample, biomarker version)
                        if "UNIQUE constraint failed" in str(e):
                            continue
                        else:
                            raise e
            
            # Prepare result message
            message_parts = [
                f"✅ Experiment '{experiment_name}' created successfully",
                f"🧬 {len(biomarker_versions_created)} biomarker versions processed",
                f"🧪 {measurements_created} measurements created"
            ]
            
            if samples_not_found:
                message_parts.append(f"⚠️  {len(samples_not_found)} sample codes not found in project: {', '.join(samples_not_found[:5])}")
                if len(samples_not_found) > 5:
                    message_parts.append(f"   ... and {len(samples_not_found) - 5} more")
            
            return True, "\n".join(message_parts), created_experiment
            
        except Exception as e:
            return False, f"Error processing CSV: {str(e)}", None
    
    def preview_csv(self, csv_path: str, num_rows: int = 5) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Preview CSV file contents and structure.
        
        Args:
            csv_path: Path to CSV file
            num_rows: Number of rows to preview
            
        Returns:
            Tuple of (success, message, preview_data)
        """
        try:
            df = pd.read_csv(csv_path)
            
            preview_data = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "sample_column_exists": "sample" in df.columns,
                "biomarker_columns": [col for col in df.columns if col != "sample"],
                "preview_rows": df.head(num_rows).to_dict(orient="records"),
                "sample_codes": df["sample"].tolist() if "sample" in df.columns else [],
                "missing_values": df.isnull().sum().to_dict()
            }
            
            return True, "CSV preview generated successfully", preview_data
            
        except Exception as e:
            return False, f"Error previewing CSV: {str(e)}", None
    
    def get_experiment_data_as_dataframe(self, experiment_id: int) -> Optional[pd.DataFrame]:
        """
        Export experiment data as a pandas DataFrame.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            DataFrame with sample codes as rows and biomarkers as columns
        """
        try:
            # Get experiment measurements
            measurements = self.experiment_db.get_measurements_by_experiment(experiment_id)
            
            if not measurements:
                return None
            
            # Build data structure
            sample_ids = set()
            biomarker_version_ids = set()
            
            # Collect all sample and biomarker version IDs
            for measurement in measurements:
                sample_ids.add(measurement.sample_id)
                biomarker_version_ids.add(measurement.biomarker_version_id)
            
            # Get sample and biomarker information
            samples = {}
            for sample_id in sample_ids:
                sample = self.sample_db.get_sample_by_id(sample_id)
                if sample:
                    samples[sample_id] = sample.code
            
            biomarker_versions = {}
            for version_id in biomarker_version_ids:
                version = self.experiment_db.get_biomarker_version_by_id(version_id)
                if version:
                    biomarker = self.experiment_db.get_biomarker_by_id(version.biomarker_id)
                    if biomarker:
                        # Use biomarker name with version for column name
                        biomarker_versions[version_id] = f"{biomarker.name}_{version.version}"
                    else:
                        biomarker_versions[version_id] = f"biomarker_version_{version_id}"
                else:
                    biomarker_versions[version_id] = f"biomarker_version_{version_id}"
            
            # Build DataFrame data
            df_data = []
            for sample_id in sample_ids:
                row_data = {"sample": samples.get(sample_id, f"sample_{sample_id}")}
                
                for measurement in measurements:
                    if measurement.sample_id == sample_id:
                        biomarker_name = biomarker_versions.get(
                            measurement.biomarker_version_id, 
                            f"biomarker_version_{measurement.biomarker_version_id}"
                        )
                        row_data[biomarker_name] = measurement.value
                
                df_data.append(row_data)
            
            return pd.DataFrame(df_data)
            
        except Exception:
            return None