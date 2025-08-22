import pytest
from mmkkb.config import Environment, set_environment
from mmkkb.projects import Project, ProjectDatabase
from mmkkb.samples import Sample, SampleDatabase, CurrentProjectManager


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set test environment for all tests."""
    set_environment(Environment.TESTING)
    yield
    # Cleanup after tests
    from mmkkb.db_utils import DatabaseUtils
    DatabaseUtils.clean_all_test_databases(confirm=False)


@pytest.fixture
def project_db():
    """Fixture to provide a fresh project database for each test."""
    return ProjectDatabase()


@pytest.fixture
def sample_db():
    """Fixture to provide a fresh sample database for each test."""
    return SampleDatabase()


@pytest.fixture
def test_project(project_db):
    """Fixture to create a test project."""
    project = Project(
        code="P001",
        name="Test Project",
        description="A test project for samples.",
        creator="Tester",
    )
    return project_db.create_project(project)


@pytest.fixture
def current_project_manager():
    """Fixture to provide a fresh current project manager."""
    return CurrentProjectManager()


def test_create_sample(sample_db, test_project):
    """Test creating a sample."""
    sample = Sample(
        code="S001",
        age=45,
        bmi=23.5,
        dx=True,
        dx_origin="Clinical",
        collection_center="Hospital A",
        processing_time=120,
        project_id=test_project.id,
    )
    
    created_sample = sample_db.create_sample(sample)
    
    assert created_sample.id is not None
    assert created_sample.code == "S001"
    assert created_sample.age == 45
    assert created_sample.bmi == 23.5
    assert created_sample.dx is True
    assert created_sample.dx_origin == "Clinical"
    assert created_sample.collection_center == "Hospital A"
    assert created_sample.processing_time == 120
    assert created_sample.project_id == test_project.id
    assert created_sample.created_at is not None
    assert created_sample.updated_at is not None


def test_get_sample_by_id(sample_db, test_project):
    """Test retrieving a sample by ID."""
    sample = Sample(
        code="S002",
        age=30,
        bmi=22.0,
        dx=False,
        dx_origin="Screening",
        collection_center="Clinic B",
        processing_time=90,
        project_id=test_project.id,
    )
    created_sample = sample_db.create_sample(sample)
    
    retrieved_sample = sample_db.get_sample_by_id(created_sample.id)
    
    assert retrieved_sample is not None
    assert retrieved_sample.id == created_sample.id
    assert retrieved_sample.code == "S002"
    assert retrieved_sample.dx is False


def test_get_sample_by_code(sample_db, test_project):
    """Test retrieving a sample by code within a project."""
    sample = Sample(
        code="S003",
        age=55,
        bmi=28.3,
        dx=True,
        dx_origin="Biopsy",
        collection_center="Hospital C",
        processing_time=150,
        project_id=test_project.id,
    )
    sample_db.create_sample(sample)
    
    retrieved_sample = sample_db.get_sample_by_code("S003", test_project.id)
    
    assert retrieved_sample is not None
    assert retrieved_sample.code == "S003"
    assert retrieved_sample.project_id == test_project.id


def test_sample_code_unique_per_project(sample_db, test_project, project_db):
    """Test that sample codes are unique within a project but can be reused across projects."""
    # Create another project
    project2 = Project(
        code="P002",
        name="Second Project",
        description="Another test project.",
        creator="Tester",
    )
    created_project2 = project_db.create_project(project2)
    
    # Create sample in first project
    sample1 = Sample(
        code="S001",
        age=40,
        bmi=25.0,
        dx=False,
        dx_origin="Clinical",
        collection_center="Hospital A",
        processing_time=100,
        project_id=test_project.id,
    )
    sample_db.create_sample(sample1)
    
    # Try to create another sample with same code in same project (should fail)
    sample_duplicate = Sample(
        code="S001",
        age=35,
        bmi=24.0,
        dx=True,
        dx_origin="Clinical",
        collection_center="Hospital A",
        processing_time=110,
        project_id=test_project.id,
    )
    
    with pytest.raises(Exception):  # Should raise integrity constraint violation
        sample_db.create_sample(sample_duplicate)
    
    # Create sample with same code in different project (should succeed)
    sample_different_project = Sample(
        code="S001",
        age=50,
        bmi=26.0,
        dx=True,
        dx_origin="Screening",
        collection_center="Clinic B",
        processing_time=95,
        project_id=created_project2.id,
    )
    created_sample = sample_db.create_sample(sample_different_project)
    
    assert created_sample.id is not None
    assert created_sample.code == "S001"
    assert created_sample.project_id == created_project2.id


def test_list_samples_by_project(sample_db, test_project, project_db):
    """Test listing samples filtered by project."""
    # Create another project
    project2 = Project(
        code="P002",
        name="Second Project",
        description="Another test project.",
        creator="Tester",
    )
    created_project2 = project_db.create_project(project2)
    
    # Create samples in first project
    for i in range(3):
        sample = Sample(
            code=f"S00{i+1}",
            age=30 + i,
            bmi=22.0 + i,
            dx=i % 2 == 0,
            dx_origin="Clinical",
            collection_center="Hospital A",
            processing_time=100 + i * 10,
            project_id=test_project.id,
        )
        sample_db.create_sample(sample)
    
    # Create sample in second project
    sample = Sample(
        code="S004",
        age=40,
        bmi=25.0,
        dx=True,
        dx_origin="Screening",
        collection_center="Clinic B",
        processing_time=90,
        project_id=created_project2.id,
    )
    sample_db.create_sample(sample)
    
    # List samples for first project
    samples_p1 = sample_db.list_samples(test_project.id)
    assert len(samples_p1) == 3
    assert all(s.project_id == test_project.id for s in samples_p1)
    
    # List samples for second project
    samples_p2 = sample_db.list_samples(created_project2.id)
    assert len(samples_p2) == 1
    assert samples_p2[0].project_id == created_project2.id
    
    # List all samples
    all_samples = sample_db.list_samples()
    assert len(all_samples) == 4


def test_update_sample(sample_db, test_project):
    """Test updating a sample."""
    sample = Sample(
        code="S005",
        age=35,
        bmi=23.0,
        dx=False,
        dx_origin="Clinical",
        collection_center="Hospital A",
        processing_time=100,
        project_id=test_project.id,
    )
    created_sample = sample_db.create_sample(sample)
    
    # Update the sample
    created_sample.age = 36
    created_sample.bmi = 24.5
    created_sample.dx = True
    created_sample.dx_origin = "Biopsy"
    
    updated_sample = sample_db.update_sample(created_sample)
    
    assert updated_sample.age == 36
    assert updated_sample.bmi == 24.5
    assert updated_sample.dx is True
    assert updated_sample.dx_origin == "Biopsy"
    # Check that updated_at is greater than or equal to created_at (timestamps may be identical due to second precision)
    assert updated_sample.updated_at >= updated_sample.created_at


def test_delete_sample(sample_db, test_project):
    """Test deleting a sample."""
    sample = Sample(
        code="S006",
        age=45,
        bmi=27.0,
        dx=True,
        dx_origin="Clinical",
        collection_center="Hospital B",
        processing_time=120,
        project_id=test_project.id,
    )
    created_sample = sample_db.create_sample(sample)
    
    # Delete by ID
    success = sample_db.delete_sample(created_sample.id)
    assert success
    
    # Verify it's gone
    retrieved_sample = sample_db.get_sample_by_id(created_sample.id)
    assert retrieved_sample is None


def test_delete_sample_by_code(sample_db, test_project):
    """Test deleting a sample by code and project."""
    sample = Sample(
        code="S007",
        age=50,
        bmi=26.0,
        dx=False,
        dx_origin="Screening",
        collection_center="Clinic C",
        processing_time=80,
        project_id=test_project.id,
    )
    sample_db.create_sample(sample)
    
    # Delete by code and project
    success = sample_db.delete_sample_by_code("S007", test_project.id)
    assert success
    
    # Verify it's gone
    retrieved_sample = sample_db.get_sample_by_code("S007", test_project.id)
    assert retrieved_sample is None


def test_count_samples(sample_db, test_project, project_db):
    """Test counting samples."""
    # Create another project
    project2 = Project(
        code="P002",
        name="Second Project",
        description="Another test project.",
        creator="Tester",
    )
    created_project2 = project_db.create_project(project2)
    
    # Initially no samples
    assert sample_db.count_samples() == 0
    assert sample_db.count_samples(test_project.id) == 0
    assert sample_db.count_samples(created_project2.id) == 0
    
    # Create samples in first project
    for i in range(2):
        sample = Sample(
            code=f"S00{i+1}",
            age=30 + i,
            bmi=22.0 + i,
            dx=i % 2 == 0,
            dx_origin="Clinical",
            collection_center="Hospital A",
            processing_time=100 + i * 10,
            project_id=test_project.id,
        )
        sample_db.create_sample(sample)
    
    # Create sample in second project
    sample = Sample(
        code="S003",
        age=40,
        bmi=25.0,
        dx=True,
        dx_origin="Screening",
        collection_center="Clinic B",
        processing_time=90,
        project_id=created_project2.id,
    )
    sample_db.create_sample(sample)
    
    # Check counts
    assert sample_db.count_samples() == 3
    assert sample_db.count_samples(test_project.id) == 2
    assert sample_db.count_samples(created_project2.id) == 1


def test_current_project_manager(current_project_manager, test_project):
    """Test the current project manager functionality."""
    # Clear any existing project state first
    current_project_manager.clear_current_project()
    
    # Initially no project is active
    assert not current_project_manager.is_project_active()
    assert current_project_manager.get_current_project_id() is None
    assert current_project_manager.get_current_project_code() is None
    
    # Set current project
    success = current_project_manager.use_project(test_project.code)
    assert success
    assert current_project_manager.is_project_active()
    assert current_project_manager.get_current_project_id() == test_project.id
    assert current_project_manager.get_current_project_code() == test_project.code
    
    # Try to set non-existent project
    success = current_project_manager.use_project("NONEXISTENT")
    assert not success
    # Should still have the previous project active
    assert current_project_manager.is_project_active()
    assert current_project_manager.get_current_project_code() == test_project.code
    
    # Clear current project
    current_project_manager.clear_current_project()
    assert not current_project_manager.is_project_active()
    assert current_project_manager.get_current_project_id() is None
    assert current_project_manager.get_current_project_code() is None


def test_dx_boolean_handling(sample_db, test_project):
    """Test that dx field properly handles boolean values."""
    # Test with True (disease)
    sample_disease = Sample(
        code="S_DISEASE",
        age=45,
        bmi=25.0,
        dx=True,
        dx_origin="Clinical",
        collection_center="Hospital A",
        processing_time=100,
        project_id=test_project.id,
    )
    created_disease = sample_db.create_sample(sample_disease)
    
    # Test with False (benign)
    sample_benign = Sample(
        code="S_BENIGN",
        age=50,
        bmi=24.0,
        dx=False,
        dx_origin="Screening",
        collection_center="Clinic B",
        processing_time=90,
        project_id=test_project.id,
    )
    created_benign = sample_db.create_sample(sample_benign)
    
    # Retrieve and verify
    retrieved_disease = sample_db.get_sample_by_id(created_disease.id)
    retrieved_benign = sample_db.get_sample_by_id(created_benign.id)
    
    assert retrieved_disease.dx is True
    assert retrieved_benign.dx is False
    assert isinstance(retrieved_disease.dx, bool)
    assert isinstance(retrieved_benign.dx, bool)


def test_foreign_key_constraint(sample_db):
    """Test that foreign key constraint prevents orphaned samples."""
    # Try to create sample with non-existent project_id
    sample = Sample(
        code="S_ORPHAN",
        age=40,
        bmi=25.0,
        dx=True,
        dx_origin="Clinical",
        collection_center="Hospital A",
        processing_time=100,
        project_id=99999,  # Non-existent project
    )
    
    with pytest.raises(Exception):  # Should raise foreign key constraint violation
        sample_db.create_sample(sample)