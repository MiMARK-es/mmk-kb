"""
Project model and database operations for MMK Knowledge Base.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from .config import get_database_path


@dataclass
class Project:
    """Project model with code, name, description, and creator."""

    code: str
    name: str
    description: str
    creator: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None


class ProjectDatabase:
    """Database operations for projects."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to database file. If None, uses current environment's database.
        """
        self.db_path = db_path or get_database_path()
        self.init_database()

    def init_database(self):
        """Initialize the database with the projects table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    creator TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def create_project(self, project: Project) -> Project:
        """Create a new project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO projects (code, name, description, creator)
                VALUES (?, ?, ?, ?)
            """,
                (project.code, project.name, project.description, project.creator),
            )
            project.id = cursor.lastrowid
            # Get the created timestamps
            row = conn.execute(
                """
                SELECT created_at, updated_at FROM projects WHERE id = ?
            """,
                (project.id,),
            ).fetchone()
            if row:
                project.created_at = datetime.fromisoformat(row[0])
                project.updated_at = datetime.fromisoformat(row[1])
            conn.commit()
        return project

    def get_project_by_code(self, code: str) -> Optional[Project]:
        """Get a project by its code."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM projects WHERE code = ?
            """,
                (code,),
            ).fetchone()
            if row:
                return Project(
                    id=row["id"],
                    code=row["code"],
                    name=row["name"],
                    description=row["description"],
                    creator=row["creator"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None

    def get_project_by_id(self, project_id: int) -> Optional[Project]:
        """Get a project by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM projects WHERE id = ?
            """,
                (project_id,),
            ).fetchone()
            if row:
                return Project(
                    id=row["id"],
                    code=row["code"],
                    name=row["name"],
                    description=row["description"],
                    creator=row["creator"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None

    def list_projects(self) -> List[Project]:
        """Get all projects."""
        projects = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM projects ORDER BY created_at DESC
            """
            ).fetchall()
            for row in rows:
                projects.append(
                    Project(
                        id=row["id"],
                        code=row["code"],
                        name=row["name"],
                        description=row["description"],
                        creator=row["creator"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
        return projects

    def update_project(self, project: Project) -> Project:
        """Update an existing project."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE projects 
                SET name = ?, description = ?, creator = ?, updated_at = CURRENT_TIMESTAMP
                WHERE code = ?
            """,
                (project.name, project.description, project.creator, project.code),
            )
            # Get the updated timestamp
            row = conn.execute(
                """
                SELECT updated_at FROM projects WHERE code = ?
            """,
                (project.code,),
            ).fetchone()
            if row:
                project.updated_at = datetime.fromisoformat(row[0])
            conn.commit()
        return project

    def delete_project(self, code: str) -> bool:
        """Delete a project by code."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM projects WHERE code = ?
            """,
                (code,),
            )
            conn.commit()
            return cursor.rowcount > 0
