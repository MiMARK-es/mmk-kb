import os

import pytest

from mmkkb.config import Environment, set_environment
from mmkkb.projects import Project, ProjectDatabase


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set test environment for all tests."""
    set_environment(Environment.TESTING)
    yield
    # Cleanup after tests
    from mmkkb.db_utils import DatabaseUtils

    DatabaseUtils.clean_all_test_databases(confirm=False)


@pytest.fixture
def db():
    """Fixture to provide a fresh database for each test."""
    # The database will automatically use the test environment path
    db = ProjectDatabase()
    yield db
    # Cleanup is handled by the autouse fixture


def test_create_project(db):
    """Test creating a project."""
    project = Project(
        code="P001",
        name="Test Project",
        description="A test project.",
        creator="Tester",
    )
    created_project = db.create_project(project)

    assert created_project.id is not None
    assert created_project.code == "P001"
    assert created_project.name == "Test Project"


def test_get_project_by_code(db):
    """Test retrieving a project by code."""
    project = Project(
        code="P002",
        name="Another Project",
        description="Another test project.",
        creator="Tester",
    )
    db.create_project(project)

    retrieved_project = db.get_project_by_code("P002")
    assert retrieved_project is not None
    assert retrieved_project.code == "P002"


def test_update_project(db):
    """Test updating a project."""
    project = Project(
        code="P003",
        name="Update Test",
        description="Before update.",
        creator="Tester",
    )
    db.create_project(project)

    project.name = "Updated Name"
    updated_project = db.update_project(project)

    assert updated_project.name == "Updated Name"


def test_delete_project(db):
    """Test deleting a project."""
    project = Project(
        code="P004", name="Delete Test", description="To be deleted.", creator="Tester"
    )
    db.create_project(project)

    success = db.delete_project("P004")
    assert success
    assert db.get_project_by_code("P004") is None
