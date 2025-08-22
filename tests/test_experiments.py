import pytest
import tempfile
import pandas as pd
from pathlib import Path
from mmkkb.config import Environment, set_environment
from mmkkb.projects import Project, ProjectDatabase
from mmkkb.samples import Sample, SampleDatabase
from mmkkb.experiments import Experiment, Biomarker, BiomarkerVersion, Measurement, ExperimentDatabase
from mmkkb.csv_processor import CSVProcessor


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set test environment for all tests."""
    set_environment(Environment.TESTING)
    yield
    # Cleanup after tests
    from mmkkb.db_utils import DatabaseUtils
    DatabaseUtils.clean_all_test_databases(confirm=False)


@pytest.fixture
def project_with_samples():
    """Fixture to create a test project with samples."""
    # Create project
    project_db = ProjectDatabase()
    project = Project(
        code="EXP001",
        name="Experiment Test Project",
        description="A test project for experiments.",
        creator="Tester",
    )
    created_project = project_db.create_project(project)
    
    # Create samples
    sample_db = SampleDatabase()
    samples = []
    for i in range(5):
        sample = Sample(
            code=f"S{i+1:03d}",
            age=30 + i,
            bmi=22.0 + i,
            dx=i % 2 == 0,
            dx_origin="Clinical",
            collection_center="Hospital A",
            processing_time=100 + i * 10,
            project_id=created_project.id,
        )
        created_sample = sample_db.create_sample(sample)
        samples.append(created_sample)
    
    return created_project, samples


@pytest.fixture
def experiment_db():
    """Fixture to provide experiment database."""
    return ExperimentDatabase()


def test_create_biomarker(experiment_db):
    """Test creating biomarkers."""
    biomarker = Biomarker(
        name="IL-6",
        description="Interleukin 6",
        category="cytokine"
    )
    
    created_biomarker = experiment_db.create_biomarker(biomarker)
    
    assert created_biomarker.id is not None
    assert created_biomarker.name == "IL-6"
    assert created_biomarker.category == "cytokine"
    assert created_biomarker.created_at is not None


def test_biomarker_unique_name(experiment_db):
    """Test that biomarkers are unique by name only."""
    biomarker1 = Biomarker(name="IL-6", category="cytokine")
    biomarker2 = Biomarker(name="IL-6", category="interleukin")  # Different category, same name
    
    created1 = experiment_db.create_biomarker(biomarker1)
    created2 = experiment_db.create_biomarker(biomarker2)  # Should return existing
    
    assert created1.id == created2.id  # Same biomarker returned


def test_biomarker_versions(experiment_db):
    """Test biomarker versions with different versions of same biomarker."""
    # Create biomarker
    biomarker = Biomarker(name="IL-6", category="cytokine")
    created_biomarker = experiment_db.create_biomarker(biomarker)
    
    # Create different versions
    version1 = BiomarkerVersion(
        biomarker_id=created_biomarker.id,
        version="RUO",
        description="Research use only"
    )
    version2 = BiomarkerVersion(
        biomarker_id=created_biomarker.id,
        version="proprietary",
        description="Proprietary assay"
    )
    version3 = BiomarkerVersion(
        biomarker_id=created_biomarker.id,
        version="RUO",  # Duplicate
        description="Research use only"
    )
    
    created_v1 = experiment_db.create_biomarker_version(version1)
    created_v2 = experiment_db.create_biomarker_version(version2)
    created_v3 = experiment_db.create_biomarker_version(version3)  # Should return existing
    
    assert created_v1.id != created_v2.id  # Different versions
    assert created_v1.id == created_v3.id  # Same version returns existing
    assert created_v1.biomarker_id == created_v2.biomarker_id  # Same biomarker


def test_create_biomarker_with_version(experiment_db):
    """Test convenience method for creating biomarker with version."""
    version = experiment_db.create_biomarker_with_version(
        biomarker_name="TNF-alpha",
        version="v2.0",
        biomarker_description="Tumor necrosis factor alpha",
        version_description="Version 2.0 of the assay",
        category="cytokine"
    )
    
    assert version.id is not None
    assert version.version == "v2.0"
    
    # Verify biomarker was created
    biomarker = experiment_db.get_biomarker_by_id(version.biomarker_id)
    assert biomarker.name == "TNF-alpha"
    assert biomarker.category == "cytokine"


def test_create_experiment(experiment_db, project_with_samples):
    """Test creating experiments."""
    project, samples = project_with_samples
    
    experiment = Experiment(
        name="Cytokine Panel",
        description="Measurement of inflammatory cytokines",
        project_id=project.id,
        csv_filename="cytokines.csv"
    )
    
    created_experiment = experiment_db.create_experiment(experiment)
    
    assert created_experiment.id is not None
    assert created_experiment.name == "Cytokine Panel"
    assert created_experiment.project_id == project.id
    assert created_experiment.upload_date is not None


def test_create_measurements(experiment_db, project_with_samples):
    """Test creating measurements with biomarker versions."""
    project, samples = project_with_samples
    
    # Create experiment
    experiment = Experiment(
        name="Test Experiment",
        description="Test measurements",
        project_id=project.id
    )
    created_experiment = experiment_db.create_experiment(experiment)
    
    # Create biomarker version
    biomarker_version = experiment_db.create_biomarker_with_version(
        biomarker_name="TNF-alpha",
        version="v1.0",
        category="cytokine"
    )
    
    # Create measurements
    measurements = []
    for i, sample in enumerate(samples):
        measurement = Measurement(
            experiment_id=created_experiment.id,
            sample_id=sample.id,
            biomarker_version_id=biomarker_version.id,
            value=10.0 + i * 2.5
        )
        created_measurement = experiment_db.create_measurement(measurement)
        measurements.append(created_measurement)
    
    assert len(measurements) == 5
    assert all(m.id is not None for m in measurements)
    
    # Test retrieval
    retrieved_measurements = experiment_db.get_measurements_by_experiment(created_experiment.id)
    assert len(retrieved_measurements) == 5


def test_measurement_unique_constraint(experiment_db, project_with_samples):
    """Test that measurements are unique per experiment+sample+biomarker_version."""
    project, samples = project_with_samples
    
    # Create experiment and biomarker version
    experiment = Experiment(
        name="Unique Test",
        description="Test unique constraint",
        project_id=project.id
    )
    created_experiment = experiment_db.create_experiment(experiment)
    
    biomarker_version = experiment_db.create_biomarker_with_version(
        biomarker_name="IL-1beta",
        version="v1.0"
    )
    
    # Create first measurement
    measurement1 = Measurement(
        experiment_id=created_experiment.id,
        sample_id=samples[0].id,
        biomarker_version_id=biomarker_version.id,
        value=15.0
    )
    experiment_db.create_measurement(measurement1)
    
    # Try to create duplicate measurement
    measurement2 = Measurement(
        experiment_id=created_experiment.id,
        sample_id=samples[0].id,
        biomarker_version_id=biomarker_version.id,
        value=20.0
    )
    
    with pytest.raises(Exception):  # Should raise unique constraint violation
        experiment_db.create_measurement(measurement2)


def test_csv_processor_validation():
    """Test CSV validation."""
    csv_processor = CSVProcessor()
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("sample,IL-6,TNF-alpha,IL-1beta\n")
        f.write("S001,10.5,15.2,8.3\n")
        f.write("S002,12.1,18.7,9.1\n")
        f.write("S003,9.8,14.3,7.8\n")
        csv_path = f.name
    
    try:
        is_valid, error_msg, biomarker_columns = csv_processor.validate_csv_structure(csv_path)
        
        assert is_valid is True
        assert error_msg == ""
        assert biomarker_columns == ["IL-6", "TNF-alpha", "IL-1beta"]
    finally:
        Path(csv_path).unlink()


def test_csv_processor_upload(project_with_samples):
    """Test complete CSV upload process."""
    project, samples = project_with_samples
    csv_processor = CSVProcessor()
    
    # Create temporary CSV file with existing sample codes
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("sample,IL-6,TNF-alpha,IL-1beta\n")
        for i, sample in enumerate(samples[:3]):
            f.write(f"{sample.code},{10.5 + i},{15.2 + i*2},{8.3 + i*0.5}\n")
        csv_path = f.name
    
    try:
        success, message, experiment = csv_processor.process_csv_upload(
            csv_path=csv_path,
            experiment_name="CSV Test Experiment",
            experiment_description="Test CSV upload functionality",
            project_id=project.id,
            biomarker_version="test_v1.0"
        )
        
        assert success is True
        assert experiment is not None
        assert "3 biomarker versions processed" in message
        assert "9 measurements created" in message  # 3 samples * 3 biomarkers
        
        # Verify data was created
        experiment_db = ExperimentDatabase()
        measurements = experiment_db.get_measurements_by_experiment(experiment.id)
        assert len(measurements) == 9
        
        # Verify biomarkers and versions were created
        biomarkers = experiment_db.list_biomarkers()
        il6_biomarker = next((b for b in biomarkers if b.name == "IL-6"), None)
        assert il6_biomarker is not None
        
        versions = experiment_db.list_biomarker_versions(il6_biomarker.id)
        test_versions = [v for v in versions if v.version == "test_v1.0"]
        assert len(test_versions) == 1
        
    finally:
        Path(csv_path).unlink()


def test_measurement_summary(experiment_db, project_with_samples):
    """Test measurement summary statistics."""
    project, samples = project_with_samples
    
    # Create experiment with measurements
    experiment = Experiment(
        name="Summary Test",
        description="Test summary functionality",
        project_id=project.id
    )
    created_experiment = experiment_db.create_experiment(experiment)
    
    # Create biomarker versions and measurements
    biomarker_versions = []
    for biomarker_name in ["IL-6", "TNF-alpha"]:
        version = experiment_db.create_biomarker_with_version(
            biomarker_name=biomarker_name,
            version="summary_test"
        )
        biomarker_versions.append(version)
    
    # Create measurements (2 biomarker versions * 3 samples = 6 measurements)
    for version in biomarker_versions:
        for sample in samples[:3]:
            measurement = Measurement(
                experiment_id=created_experiment.id,
                sample_id=sample.id,
                biomarker_version_id=version.id,
                value=10.0
            )
            experiment_db.create_measurement(measurement)
    
    # Test summary
    summary = experiment_db.get_measurement_summary(project.id)
    
    assert summary["experiment_count"] >= 1
    assert summary["sample_count"] >= 3
    assert summary["biomarker_count"] >= 2  # 2 unique biomarkers
    assert summary["biomarker_version_count"] >= 2  # 2 versions
    assert summary["measurement_count"] >= 6


def test_cross_experiment_sample_biomarker():
    """Test that same sample and biomarker can be in multiple experiments with different versions."""
    # Create project and sample
    project_db = ProjectDatabase()
    project = Project(code="CROSS001", name="Cross Test", description="Test", creator="Tester")
    created_project = project_db.create_project(project)
    
    sample_db = SampleDatabase()
    sample = Sample(
        code="CROSS_S001", age=30, bmi=22.0, dx=True,
        dx_origin="Clinical", collection_center="Hospital",
        processing_time=100, project_id=created_project.id
    )
    created_sample = sample_db.create_sample(sample)
    
    experiment_db = ExperimentDatabase()
    
    # Create biomarker with two versions
    version1 = experiment_db.create_biomarker_with_version(
        biomarker_name="IL-6",
        version="RUO"
    )
    version2 = experiment_db.create_biomarker_with_version(
        biomarker_name="IL-6",
        version="proprietary"
    )
    
    # Create two experiments
    exp1 = Experiment(name="Exp1", description="First", project_id=created_project.id)
    exp2 = Experiment(name="Exp2", description="Second", project_id=created_project.id)
    created_exp1 = experiment_db.create_experiment(exp1)
    created_exp2 = experiment_db.create_experiment(exp2)
    
    # Create measurements in both experiments for same sample+biomarker but different versions
    measurement1 = Measurement(
        experiment_id=created_exp1.id,
        sample_id=created_sample.id,
        biomarker_version_id=version1.id,
        value=15.0
    )
    measurement2 = Measurement(
        experiment_id=created_exp2.id,
        sample_id=created_sample.id,
        biomarker_version_id=version2.id,
        value=18.5
    )
    
    created_m1 = experiment_db.create_measurement(measurement1)
    created_m2 = experiment_db.create_measurement(measurement2)
    
    assert created_m1.id != created_m2.id
    assert created_m1.value == 15.0
    assert created_m2.value == 18.5
    
    # Verify sample has measurements from both experiments
    sample_measurements = experiment_db.get_measurements_by_sample(created_sample.id)
    assert len(sample_measurements) == 2
    assert set(m.experiment_id for m in sample_measurements) == {created_exp1.id, created_exp2.id}
    
    # Verify biomarker analysis groups data correctly
    biomarker_id = version1.biomarker_id  # Same biomarker for both versions
    analysis_data = experiment_db.get_biomarker_data_for_analysis(biomarker_id)
    
    assert analysis_data is not None
    assert analysis_data["total_measurements"] == 2
    assert analysis_data["unique_experiments"] == 2
    assert analysis_data["unique_samples"] == 1
    assert set(analysis_data["versions_used"]) == {"RUO", "proprietary"}


def test_biomarker_data_for_analysis(experiment_db, project_with_samples):
    """Test the biomarker analysis functionality."""
    project, samples = project_with_samples
    
    # Create biomarker versions
    version1 = experiment_db.create_biomarker_with_version(
        biomarker_name="IL-6",
        version="v1.0",
        category="cytokine"
    )
    version2 = experiment_db.create_biomarker_with_version(
        biomarker_name="IL-6",
        version="v2.0",
        category="cytokine"
    )
    
    # Create experiments
    exp1 = Experiment(name="Exp1", description="First", project_id=project.id)
    exp2 = Experiment(name="Exp2", description="Second", project_id=project.id)
    created_exp1 = experiment_db.create_experiment(exp1)
    created_exp2 = experiment_db.create_experiment(exp2)
    
    # Create measurements with different versions
    for i, sample in enumerate(samples[:2]):
        measurement1 = Measurement(
            experiment_id=created_exp1.id,
            sample_id=sample.id,
            biomarker_version_id=version1.id,
            value=10.0 + i
        )
        measurement2 = Measurement(
            experiment_id=created_exp2.id,
            sample_id=sample.id,
            biomarker_version_id=version2.id,
            value=15.0 + i
        )
        experiment_db.create_measurement(measurement1)
        experiment_db.create_measurement(measurement2)
    
    # Test analysis data
    analysis_data = experiment_db.get_biomarker_data_for_analysis(version1.biomarker_id)
    
    assert analysis_data is not None
    assert analysis_data["biomarker"].name == "IL-6"
    assert analysis_data["total_measurements"] == 4  # 2 samples * 2 versions
    assert analysis_data["unique_experiments"] == 2
    assert analysis_data["unique_samples"] == 2
    assert set(analysis_data["versions_used"]) == {"v1.0", "v2.0"}
    
    # Check measurement details
    measurements = analysis_data["measurements"]
    assert len(measurements) == 4
    assert all("sample_code" in m for m in measurements)
    assert all("experiment_name" in m for m in measurements)
    assert all("version" in m for m in measurements)