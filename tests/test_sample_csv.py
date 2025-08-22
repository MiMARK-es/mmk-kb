"""
Tests for sample CSV processing functionality.
"""
import tempfile
import pytest
from pathlib import Path
from src.mmkkb.sample_csv_processor import SampleCSVProcessor
from src.mmkkb.samples import Sample, SampleDatabase
from src.mmkkb.projects import Project, ProjectDatabase


@pytest.fixture
def sample_csv_processor():
    """Create a sample CSV processor for testing."""
    return SampleCSVProcessor()


@pytest.fixture
def project_for_samples():
    """Create a test project for samples."""
    import uuid
    project_db = ProjectDatabase()
    unique_code = f"TEST_SAMPLES_{uuid.uuid4().hex[:8]}"
    project = Project(
        code=unique_code,
        name="Test Samples Project",
        description="Project for testing sample CSV functionality",
        creator="test_user"
    )
    return project_db.create_project(project)


def test_sample_csv_validation():
    """Test CSV validation for sample data."""
    csv_processor = SampleCSVProcessor()
    
    # Create temporary CSV file with valid sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("code,age,bmi,dx,dx_origin,collection_center,processing_time\n")
        f.write("S001,45,25.3,1,biopsy,Hospital_A,120\n")
        f.write("S002,52,28.7,0,screening,Hospital_B,90\n")
        f.write("S003,38,22.1,disease,pathology,Hospital_A,150\n")
        csv_path = f.name
    
    try:
        is_valid, error_msg, columns = csv_processor.validate_csv_structure(csv_path)
        
        assert is_valid is True
        assert error_msg == ""
        assert len(columns) == 7
        assert "code" in columns
        assert "age" in columns
        assert "dx" in columns
    finally:
        Path(csv_path).unlink()


def test_sample_csv_validation_missing_columns():
    """Test CSV validation with missing required columns."""
    csv_processor = SampleCSVProcessor()
    
    # Create CSV with missing columns
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("code,age,bmi\n")
        f.write("S001,45,25.3\n")
        csv_path = f.name
    
    try:
        is_valid, error_msg, columns = csv_processor.validate_csv_structure(csv_path)
        
        assert is_valid is False
        assert "Missing required columns" in error_msg
        assert "dx" in error_msg
    finally:
        Path(csv_path).unlink()


def test_sample_csv_validation_invalid_age():
    """Test CSV validation with invalid age values."""
    csv_processor = SampleCSVProcessor()
    
    # Create CSV with invalid age
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("code,age,bmi,dx,dx_origin,collection_center,processing_time\n")
        f.write("S001,200,25.3,1,biopsy,Hospital_A,120\n")  # Invalid age
        csv_path = f.name
    
    try:
        is_valid, error_msg, columns = csv_processor.validate_csv_structure(csv_path)
        
        assert is_valid is False
        assert "Invalid age value" in error_msg
    finally:
        Path(csv_path).unlink()


def test_sample_csv_preview():
    """Test CSV preview functionality."""
    csv_processor = SampleCSVProcessor()
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("code,age,bmi,dx,dx_origin,collection_center,processing_time\n")
        f.write("S001,45,25.3,1,biopsy,Hospital_A,120\n")
        f.write("S002,52,28.7,0,screening,Hospital_B,90\n")
        f.write("S003,38,22.1,case,pathology,Hospital_A,150\n")
        csv_path = f.name
    
    try:
        success, message, preview_data = csv_processor.preview_csv(csv_path, 2)
        
        assert success is True
        assert preview_data['total_rows'] == 3
        assert preview_data['total_columns'] == 7
        assert len(preview_data['preview_rows']) == 2
        assert 'dx_distribution' in preview_data
        
        # Check dx distribution (values are stored as they appear in CSV)
        dx_dist = preview_data['dx_distribution']
        assert 1 in dx_dist or '1' in dx_dist  # One disease case
        assert 0 in dx_dist or '0' in dx_dist  # One control
        assert 'case' in dx_dist  # One case
    finally:
        Path(csv_path).unlink()


def test_sample_csv_upload(project_for_samples):
    """Test complete CSV upload process for samples."""
    project = project_for_samples
    csv_processor = SampleCSVProcessor()
    
    # Create temporary CSV file with sample data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("code,age,bmi,dx,dx_origin,collection_center,processing_time\n")
        f.write("S001,45,25.3,1,biopsy,Hospital_A,120\n")
        f.write("S002,52,28.7,0,screening,Hospital_B,90\n")
        f.write("S003,38,22.1,disease,pathology,Hospital_A,150\n")
        f.write("S004,60,30.5,control,screening,Hospital_C,100\n")
        csv_path = f.name
    
    try:
        success, message, samples = csv_processor.process_csv_upload(
            csv_path=csv_path,
            project_id=project.id,
            skip_duplicates=True
        )
        
        assert success is True
        assert len(samples) == 4
        assert "4 samples created successfully" in message
        
        # Verify samples were created correctly
        sample_db = SampleDatabase()
        created_samples = sample_db.list_samples(project.id)
        assert len(created_samples) == 4
        
        # Check specific sample data
        s001 = sample_db.get_sample_by_code("S001", project.id)
        assert s001 is not None
        assert s001.age == 45
        assert s001.bmi == 25.3
        assert s001.dx is True  # Disease case
        assert s001.dx_origin == "biopsy"
        assert s001.collection_center == "Hospital_A"
        assert s001.processing_time == 120
        
        s002 = sample_db.get_sample_by_code("S002", project.id)
        assert s002 is not None
        assert s002.dx is False  # Control
        
        s003 = sample_db.get_sample_by_code("S003", project.id)
        assert s003 is not None
        assert s003.dx is True  # Disease case (from "disease" string)
        
        s004 = sample_db.get_sample_by_code("S004", project.id)
        assert s004 is not None
        assert s004.dx is False  # Control (from "control" string)
    finally:
        Path(csv_path).unlink()


def test_sample_csv_upload_duplicates(project_for_samples):
    """Test CSV upload with duplicate sample codes."""
    project = project_for_samples
    csv_processor = SampleCSVProcessor()
    sample_db = SampleDatabase()
    
    # First, create a sample manually
    existing_sample = Sample(
        code="S001",
        age=40,
        bmi=24.0,
        dx=False,
        dx_origin="existing",
        collection_center="Hospital_X",
        processing_time=100,
        project_id=project.id
    )
    sample_db.create_sample(existing_sample)
    
    # Create CSV with duplicate and new samples
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("code,age,bmi,dx,dx_origin,collection_center,processing_time\n")
        f.write("S001,45,25.3,1,biopsy,Hospital_A,120\n")  # Duplicate
        f.write("S002,52,28.7,0,screening,Hospital_B,90\n")   # New
        csv_path = f.name
    
    try:
        # Test with skip_duplicates=True
        success, message, samples = csv_processor.process_csv_upload(
            csv_path=csv_path,
            project_id=project.id,
            skip_duplicates=True
        )
        
        assert success is True
        assert len(samples) == 1  # Only S002 should be created
        assert "1 samples created successfully" in message
        assert "1 samples skipped (duplicates)" in message
        assert "S001" in message
        
        # Test with skip_duplicates=False
        success, message, samples = csv_processor.process_csv_upload(
            csv_path=csv_path,
            project_id=project.id,
            skip_duplicates=False
        )
        
        assert "already exists" in message
    finally:
        Path(csv_path).unlink()


def test_sample_csv_export(project_for_samples):
    """Test exporting samples to CSV."""
    project = project_for_samples
    csv_processor = SampleCSVProcessor()
    sample_db = SampleDatabase()
    
    # Create test samples
    samples_data = [
        ("S001", 45, 25.3, True, "biopsy", "Hospital_A", 120),
        ("S002", 52, 28.7, False, "screening", "Hospital_B", 90),
        ("S003", 38, 22.1, True, "pathology", "Hospital_A", 150),
    ]
    
    for code, age, bmi, dx, dx_origin, center, proc_time in samples_data:
        sample = Sample(
            code=code,
            age=age,
            bmi=bmi,
            dx=dx,
            dx_origin=dx_origin,
            collection_center=center,
            processing_time=proc_time,
            project_id=project.id
        )
        sample_db.create_sample(sample)
    
    # Export to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        output_path = f.name
    
    try:
        success, message = csv_processor.export_samples_to_csv(project.id, output_path)
        
        assert success is True
        assert "Exported 3 samples" in message
        
        # Verify exported content
        import pandas as pd
        df = pd.read_csv(output_path)
        
        assert len(df) == 3
        assert list(df.columns) == ['code', 'age', 'bmi', 'dx', 'dx_origin', 'collection_center', 'processing_time']
        
        # Check data integrity
        s001_row = df[df['code'] == 'S001'].iloc[0]
        assert s001_row['age'] == 45
        assert s001_row['bmi'] == 25.3
        assert s001_row['dx'] == 1  # True converted to 1
        assert s001_row['dx_origin'] == "biopsy"
    finally:
        Path(output_path).unlink()


def test_sample_csv_invalid_dx_values():
    """Test CSV validation with invalid dx values."""
    csv_processor = SampleCSVProcessor()
    
    # Create CSV with invalid dx value
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("code,age,bmi,dx,dx_origin,collection_center,processing_time\n")
        f.write("S001,45,25.3,invalid_dx,biopsy,Hospital_A,120\n")
        csv_path = f.name
    
    try:
        is_valid, error_msg, columns = csv_processor.validate_csv_structure(csv_path)
        
        assert is_valid is False
        assert "Invalid dx value" in error_msg
        assert "invalid_dx" in error_msg
    finally:
        Path(csv_path).unlink()