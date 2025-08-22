"""
Sample model and database operations for MMK Knowledge Base.
Samples are linked to projects and contain patient/specimen data.
"""
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from .config import get_database_path


@dataclass
class Sample:
    """Sample model with patient/specimen data linked to a project."""
    code: str
    age: int
    bmi: float
    dx: bool  # 0 = benign, 1 = disease
    dx_origin: str
    collection_center: str
    processing_time: int
    project_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None


class SampleDatabase:
    """Database operations for samples."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.
        Args:
            db_path: Path to database file. If None, uses current environment's database.
        """
        self.db_path = db_path or get_database_path()
        self.init_database()
    
    def init_database(self):
        """Initialize the database with the samples table."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    bmi REAL NOT NULL,
                    dx INTEGER NOT NULL CHECK (dx IN (0, 1)),
                    dx_origin TEXT NOT NULL,
                    collection_center TEXT NOT NULL,
                    processing_time INTEGER NOT NULL,
                    project_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE,
                    UNIQUE (code, project_id)
                )
                """
            )
            # Create index for better performance on project queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_samples_project_id 
                ON samples (project_id)
                """
            )
            conn.commit()
    
    def _get_connection(self):
        """Get a database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def create_sample(self, sample: Sample) -> Sample:
        """Create a new sample."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO samples (code, age, bmi, dx, dx_origin, collection_center, 
                                   processing_time, project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample.code,
                    sample.age,
                    sample.bmi,
                    int(sample.dx),
                    sample.dx_origin,
                    sample.collection_center,
                    sample.processing_time,
                    sample.project_id,
                ),
            )
            sample.id = cursor.lastrowid
            
            # Get the created timestamps
            row = conn.execute(
                """
                SELECT created_at, updated_at FROM samples WHERE id = ?
                """,
                (sample.id,),
            ).fetchone()
            if row:
                sample.created_at = datetime.fromisoformat(row[0])
                sample.updated_at = datetime.fromisoformat(row[1])
            conn.commit()
        return sample
    
    def get_sample_by_id(self, sample_id: int) -> Optional[Sample]:
        """Get a sample by its ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM samples WHERE id = ?
                """,
                (sample_id,),
            ).fetchone()
            if row:
                return Sample(
                    id=row["id"],
                    code=row["code"],
                    age=row["age"],
                    bmi=row["bmi"],
                    dx=bool(row["dx"]),
                    dx_origin=row["dx_origin"],
                    collection_center=row["collection_center"],
                    processing_time=row["processing_time"],
                    project_id=row["project_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None
    
    def get_sample_by_code(self, code: str, project_id: int) -> Optional[Sample]:
        """Get a sample by its code within a specific project."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM samples WHERE code = ? AND project_id = ?
                """,
                (code, project_id),
            ).fetchone()
            if row:
                return Sample(
                    id=row["id"],
                    code=row["code"],
                    age=row["age"],
                    bmi=row["bmi"],
                    dx=bool(row["dx"]),
                    dx_origin=row["dx_origin"],
                    collection_center=row["collection_center"],
                    processing_time=row["processing_time"],
                    project_id=row["project_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None
    
    def list_samples(self, project_id: Optional[int] = None) -> List[Sample]:
        """Get all samples, optionally filtered by project."""
        samples = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            if project_id is not None:
                query = """
                    SELECT * FROM samples 
                    WHERE project_id = ? 
                    ORDER BY created_at DESC
                """
                rows = conn.execute(query, (project_id,)).fetchall()
            else:
                query = """
                    SELECT * FROM samples 
                    ORDER BY created_at DESC
                """
                rows = conn.execute(query).fetchall()
            
            for row in rows:
                samples.append(
                    Sample(
                        id=row["id"],
                        code=row["code"],
                        age=row["age"],
                        bmi=row["bmi"],
                        dx=bool(row["dx"]),
                        dx_origin=row["dx_origin"],
                        collection_center=row["collection_center"],
                        processing_time=row["processing_time"],
                        project_id=row["project_id"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
        return samples
    
    def update_sample(self, sample: Sample) -> Sample:
        """Update an existing sample."""
        import time
        # Add a small delay to ensure updated_at is different from created_at
        time.sleep(0.01)
        
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE samples 
                SET code = ?, age = ?, bmi = ?, dx = ?, dx_origin = ?, 
                    collection_center = ?, processing_time = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    sample.code,
                    sample.age,
                    sample.bmi,
                    int(sample.dx),
                    sample.dx_origin,
                    sample.collection_center,
                    sample.processing_time,
                    sample.id,
                ),
            )
            
            # Get the updated timestamp
            row = conn.execute(
                """
                SELECT updated_at FROM samples WHERE id = ?
                """,
                (sample.id,),
            ).fetchone()
            if row:
                sample.updated_at = datetime.fromisoformat(row[0])
            conn.commit()
        return sample
    
    def delete_sample(self, sample_id: int) -> bool:
        """Delete a sample by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM samples WHERE id = ?
                """,
                (sample_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_sample_by_code(self, code: str, project_id: int) -> bool:
        """Delete a sample by code within a specific project."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM samples WHERE code = ? AND project_id = ?
                """,
                (code, project_id),
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def count_samples(self, project_id: Optional[int] = None) -> int:
        """Count samples, optionally filtered by project."""
        with self._get_connection() as conn:
            if project_id is not None:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM samples WHERE project_id = ?",
                    (project_id,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM samples")
            return cursor.fetchone()[0]


class CurrentProjectManager:
    """Manages the current active project for sample operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the current project manager.
        Args:
            db_path: Path to database file. If None, uses current environment's database.
        """
        self.db_path = db_path or get_database_path()
        self._current_project_id = None
        self._current_project_code = None
    
    def use_project(self, project_code: str) -> bool:
        """
        Set the current active project by code.
        Returns True if project exists and was set, False otherwise.
        """
        from .projects import ProjectDatabase
        
        project_db = ProjectDatabase(self.db_path)
        project = project_db.get_project_by_code(project_code)
        
        if project:
            self._current_project_id = project.id
            self._current_project_code = project.code
            return True
        return False
    
    def get_current_project_id(self) -> Optional[int]:
        """Get the current project ID."""
        return self._current_project_id
    
    def get_current_project_code(self) -> Optional[str]:
        """Get the current project code."""
        return self._current_project_code
    
    def clear_current_project(self):
        """Clear the current project."""
        self._current_project_id = None
        self._current_project_code = None
    
    def is_project_active(self) -> bool:
        """Check if a project is currently active."""
        return self._current_project_id is not None


# Global instance for CLI usage
current_project_manager = CurrentProjectManager()