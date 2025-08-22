"""
CSV processing utilities for sample data uploads.
Handles parsing CSV files with sample information for bulk upload.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from .samples import Sample, SampleDatabase


class SampleCSVProcessor:
    """Processes CSV files containing sample data."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize CSV processor with database connection."""
        self.sample_db = SampleDatabase(db_path)
    
    def validate_csv_structure(self, csv_path: str) -> Tuple[bool, str, List[str]]:
        """
        Validate CSV file structure for sample data.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Tuple of (is_valid, error_message, required_columns)
        """
        required_columns = ['code', 'age', 'bmi', 'dx', 'dx_origin', 'collection_center', 'processing_time']
        
        try:
            df = pd.read_csv(csv_path)
            
            # Check if file is empty
            if df.empty:
                return False, "CSV file is empty", []
            
            # Check for required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}", []
            
            # Validate data types and values
            for _, row in df.iterrows():
                # Check age is numeric and positive
                try:
                    age = pd.to_numeric(row['age'])
                    if age <= 0 or age > 150:
                        return False, f"Invalid age value: {age}. Age must be between 1 and 150.", []
                except (ValueError, TypeError):
                    return False, f"Age column contains non-numeric values: {row['age']}", []
                
                # Check BMI is numeric and reasonable
                try:
                    bmi = pd.to_numeric(row['bmi'])
                    if bmi <= 0 or bmi > 100:
                        return False, f"Invalid BMI value: {bmi}. BMI must be between 0 and 100.", []
                except (ValueError, TypeError):
                    return False, f"BMI column contains non-numeric values: {row['bmi']}", []
                
                # Check dx is valid (0, 1, false, true, benign, disease, etc.)
                dx_str = str(row['dx']).lower().strip()
                if dx_str not in ['0', '1', 'false', 'true', 'benign', 'disease', 'control', 'case']:
                    return False, f"Invalid dx value: {row['dx']}. Must be 0/1, false/true, benign/disease, or control/case.", []
                
                # Check processing_time is numeric and positive
                try:
                    processing_time = pd.to_numeric(row['processing_time'])
                    if processing_time < 0:
                        return False, f"Invalid processing_time value: {processing_time}. Must be non-negative.", []
                except (ValueError, TypeError):
                    return False, f"Processing_time column contains non-numeric values: {row['processing_time']}", []
            
            return True, "", required_columns
            
        except Exception as e:
            return False, f"Error reading CSV: {str(e)}", []
    
    def preview_csv(self, csv_path: str, num_rows: int = 5) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Preview CSV file structure and content.
        
        Args:
            csv_path: Path to the CSV file
            num_rows: Number of rows to preview
            
        Returns:
            Tuple of (success, message, preview_data)
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Basic structure info
            preview_data = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns),
                'preview_rows': df.head(num_rows).to_dict('records'),
                'missing_values': df.isnull().sum().to_dict(),
                'dx_distribution': {}
            }
            
            # Analyze diagnosis distribution if dx column exists
            if 'dx' in df.columns:
                dx_counts = df['dx'].value_counts()
                preview_data['dx_distribution'] = dx_counts.to_dict()
            
            return True, "CSV preview generated successfully", preview_data
            
        except Exception as e:
            return False, f"Error previewing CSV: {str(e)}", None
    
    def process_csv_upload(
        self, 
        csv_path: str, 
        project_id: int,
        skip_duplicates: bool = True
    ) -> Tuple[bool, str, List[Sample]]:
        """
        Process CSV upload and create samples.
        
        Args:
            csv_path: Path to CSV file
            project_id: ID of the project to add samples to
            skip_duplicates: Whether to skip samples with duplicate codes
            
        Returns:
            Tuple of (success, message, created_samples)
        """
        # Validate CSV structure
        is_valid, error_msg, _ = self.validate_csv_structure(csv_path)
        if not is_valid:
            return False, error_msg, []
        
        try:
            # Read CSV data
            df = pd.read_csv(csv_path)
            csv_filename = Path(csv_path).name
            
            created_samples = []
            skipped_samples = []
            error_samples = []
            
            # Process each row (sample)
            for index, row in df.iterrows():
                try:
                    sample_code = str(row['code']).strip()
                    
                    # Check if sample already exists
                    existing_sample = self.sample_db.get_sample_by_code(sample_code, project_id)
                    if existing_sample:
                        if skip_duplicates:
                            skipped_samples.append(sample_code)
                            continue
                        else:
                            error_samples.append(f"{sample_code}: already exists")
                            continue
                    
                    # Convert dx to boolean
                    dx_str = str(row['dx']).lower().strip()
                    if dx_str in ['1', 'true', 'disease', 'case']:
                        dx_bool = True
                    elif dx_str in ['0', 'false', 'benign', 'control']:
                        dx_bool = False
                    else:
                        error_samples.append(f"{sample_code}: invalid dx value '{row['dx']}'")
                        continue
                    
                    # Create sample object
                    sample = Sample(
                        code=sample_code,
                        age=int(row['age']),
                        bmi=float(row['bmi']),
                        dx=dx_bool,
                        dx_origin=str(row['dx_origin']).strip(),
                        collection_center=str(row['collection_center']).strip(),
                        processing_time=int(row['processing_time']),
                        project_id=project_id
                    )
                    
                    # Create sample in database
                    created_sample = self.sample_db.create_sample(sample)
                    created_samples.append(created_sample)
                    
                except Exception as e:
                    error_samples.append(f"{row.get('code', f'row {index + 1}')}: {str(e)}")
            
            # Prepare result message
            message_parts = [
                f"✅ Sample CSV upload completed from {csv_filename}",
                f"🧪 {len(created_samples)} samples created successfully"
            ]
            
            if skipped_samples:
                message_parts.append(f"⚠️  {len(skipped_samples)} samples skipped (duplicates): {', '.join(skipped_samples[:5])}")
                if len(skipped_samples) > 5:
                    message_parts.append(f"   ... and {len(skipped_samples) - 5} more")
            
            if error_samples:
                message_parts.append(f"❌ {len(error_samples)} samples failed:")
                for error in error_samples[:5]:
                    message_parts.append(f"   - {error}")
                if len(error_samples) > 5:
                    message_parts.append(f"   ... and {len(error_samples) - 5} more errors")
            
            success = len(created_samples) > 0 or (len(error_samples) == 0 and len(skipped_samples) > 0)
            return success, "\n".join(message_parts), created_samples
            
        except Exception as e:
            return False, f"Error processing CSV: {str(e)}", []
    
    def export_samples_to_csv(self, project_id: int, output_path: str) -> Tuple[bool, str]:
        """
        Export samples from a project to CSV file.
        
        Args:
            project_id: ID of the project
            output_path: Path where to save the CSV file
            
        Returns:
            Tuple of (success, message)
        """
        try:
            samples = self.sample_db.list_samples(project_id)
            
            if not samples:
                return False, "No samples found in the project"
            
            # Convert samples to DataFrame
            data = []
            for sample in samples:
                data.append({
                    'code': sample.code,
                    'age': sample.age,
                    'bmi': sample.bmi,
                    'dx': 1 if sample.dx else 0,
                    'dx_origin': sample.dx_origin,
                    'collection_center': sample.collection_center,
                    'processing_time': sample.processing_time
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
            return True, f"✅ Exported {len(samples)} samples to {output_path}"
            
        except Exception as e:
            return False, f"Error exporting samples: {str(e)}"