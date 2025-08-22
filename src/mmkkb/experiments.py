"""
Experiment and biomarker models and database operations for MMK Knowledge Base.
Experiments contain biomarker measurements for samples from projects.
"""
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from .config import get_database_path


@dataclass
class Biomarker:
    """Biomarker model representing the biological entity."""
    name: str
    description: Optional[str] = None
    category: Optional[str] = None  # e.g., "cytokine", "chemokine", "growth_factor"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None


@dataclass
class BiomarkerVersion:
    """Version information for a biomarker in specific experiments."""
    biomarker_id: int
    version: str  # RUO, proprietary, peptide_sequence, etc.
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None


@dataclass 
class Experiment:
    """Experiment model linked to a project."""
    name: str
    description: str
    project_id: int
    upload_date: Optional[datetime] = None
    csv_filename: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None


@dataclass
class Measurement:
    """Measurement model linking sample, biomarker version, and experiment with a value."""
    experiment_id: int
    sample_id: int
    biomarker_version_id: int  # References the specific version used
    value: float
    created_at: Optional[datetime] = None
    id: Optional[int] = None


class ExperimentDatabase:
    """Database operations for experiments, biomarkers, and measurements."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.
        Args:
            db_path: Path to database file. If None, uses current environment's database.
        """
        self.db_path = db_path or get_database_path()
        self.init_database()
    
    def init_database(self):
        """Initialize the database with experiment-related tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Biomarkers table - the core biological entity
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS biomarkers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
            # Biomarker versions table - different versions/implementations of the same biomarker
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS biomarker_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    biomarker_id INTEGER NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (biomarker_id) REFERENCES biomarkers (id) ON DELETE CASCADE,
                    UNIQUE (biomarker_id, version)
                )
                """
            )
            
            # Experiments table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    project_id INTEGER NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    csv_filename TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
                )
                """
            )
            
            # Measurements table - references biomarker version instead of biomarker directly
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    sample_id INTEGER NOT NULL,
                    biomarker_version_id INTEGER NOT NULL,
                    value REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE,
                    FOREIGN KEY (sample_id) REFERENCES samples (id) ON DELETE CASCADE,
                    FOREIGN KEY (biomarker_version_id) REFERENCES biomarker_versions (id) ON DELETE CASCADE,
                    UNIQUE (experiment_id, sample_id, biomarker_version_id)
                )
                """
            )
            
            # Create indexes for better performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_project_id ON experiments (project_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_biomarker_versions_biomarker_id ON biomarker_versions (biomarker_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_measurements_experiment_id ON measurements (experiment_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_measurements_sample_id ON measurements (sample_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_measurements_biomarker_version_id ON measurements (biomarker_version_id)"
            )
            conn.commit()
    
    def _get_connection(self):
        """Get a database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    # Biomarker operations
    def create_biomarker(self, biomarker: Biomarker) -> Biomarker:
        """Create a new biomarker or get existing one."""
        with self._get_connection() as conn:
            # Try to get existing biomarker first
            existing = self.get_biomarker_by_name(biomarker.name)
            if existing:
                return existing
            
            cursor = conn.execute(
                """
                INSERT INTO biomarkers (name, description, category)
                VALUES (?, ?, ?)
                """,
                (biomarker.name, biomarker.description, biomarker.category),
            )
            biomarker.id = cursor.lastrowid
            
            # Get the created timestamps
            row = conn.execute(
                "SELECT created_at, updated_at FROM biomarkers WHERE id = ?",
                (biomarker.id,),
            ).fetchone()
            if row:
                biomarker.created_at = datetime.fromisoformat(row[0])
                biomarker.updated_at = datetime.fromisoformat(row[1])
            conn.commit()
        return biomarker
    
    def get_biomarker_by_name(self, name: str) -> Optional[Biomarker]:
        """Get a biomarker by name."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM biomarkers WHERE name = ?",
                (name,),
            ).fetchone()
            if row:
                return Biomarker(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    category=row["category"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None
    
    def get_biomarker_by_id(self, biomarker_id: int) -> Optional[Biomarker]:
        """Get a biomarker by its ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM biomarkers WHERE id = ?",
                (biomarker_id,),
            ).fetchone()
            if row:
                return Biomarker(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    category=row["category"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None
    
    def list_biomarkers(self) -> List[Biomarker]:
        """Get all biomarkers."""
        biomarkers = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM biomarkers ORDER BY name"
            ).fetchall()
            
            for row in rows:
                biomarkers.append(
                    Biomarker(
                        id=row["id"],
                        name=row["name"],
                        description=row["description"],
                        category=row["category"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
        return biomarkers
    
    # Biomarker version operations
    def create_biomarker_version(self, biomarker_version: BiomarkerVersion) -> BiomarkerVersion:
        """Create a new biomarker version or get existing one."""
        with self._get_connection() as conn:
            # Try to get existing version first
            existing = self.get_biomarker_version_by_biomarker_and_version(
                biomarker_version.biomarker_id, biomarker_version.version
            )
            if existing:
                return existing
            
            cursor = conn.execute(
                """
                INSERT INTO biomarker_versions (biomarker_id, version, description)
                VALUES (?, ?, ?)
                """,
                (biomarker_version.biomarker_id, biomarker_version.version, biomarker_version.description),
            )
            biomarker_version.id = cursor.lastrowid
            
            # Get the created timestamps
            row = conn.execute(
                "SELECT created_at, updated_at FROM biomarker_versions WHERE id = ?",
                (biomarker_version.id,),
            ).fetchone()
            if row:
                biomarker_version.created_at = datetime.fromisoformat(row[0])
                biomarker_version.updated_at = datetime.fromisoformat(row[1])
            conn.commit()
        return biomarker_version
    
    def get_biomarker_version_by_biomarker_and_version(self, biomarker_id: int, version: str) -> Optional[BiomarkerVersion]:
        """Get a biomarker version by biomarker ID and version."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM biomarker_versions WHERE biomarker_id = ? AND version = ?",
                (biomarker_id, version),
            ).fetchone()
            if row:
                return BiomarkerVersion(
                    id=row["id"],
                    biomarker_id=row["biomarker_id"],
                    version=row["version"],
                    description=row["description"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None
    
    def get_biomarker_version_by_id(self, version_id: int) -> Optional[BiomarkerVersion]:
        """Get a biomarker version by its ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM biomarker_versions WHERE id = ?",
                (version_id,),
            ).fetchone()
            if row:
                return BiomarkerVersion(
                    id=row["id"],
                    biomarker_id=row["biomarker_id"],
                    version=row["version"],
                    description=row["description"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None
    
    def list_biomarker_versions(self, biomarker_id: Optional[int] = None) -> List[BiomarkerVersion]:
        """Get all biomarker versions, optionally filtered by biomarker."""
        versions = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            if biomarker_id is not None:
                query = "SELECT * FROM biomarker_versions WHERE biomarker_id = ? ORDER BY version"
                rows = conn.execute(query, (biomarker_id,)).fetchall()
            else:
                query = "SELECT * FROM biomarker_versions ORDER BY biomarker_id, version"
                rows = conn.execute(query).fetchall()
            
            for row in rows:
                versions.append(
                    BiomarkerVersion(
                        id=row["id"],
                        biomarker_id=row["biomarker_id"],
                        version=row["version"],
                        description=row["description"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
        return versions
    
    # Convenience method to create biomarker and version together
    def create_biomarker_with_version(self, biomarker_name: str, version: str, 
                                    biomarker_description: str = None, 
                                    version_description: str = None,
                                    category: str = None) -> BiomarkerVersion:
        """Create or get biomarker and create/get its version."""
        # Create or get biomarker
        biomarker = Biomarker(
            name=biomarker_name,
            description=biomarker_description,
            category=category
        )
        created_biomarker = self.create_biomarker(biomarker)
        
        # Create or get version
        biomarker_version = BiomarkerVersion(
            biomarker_id=created_biomarker.id,
            version=version,
            description=version_description
        )
        created_version = self.create_biomarker_version(biomarker_version)
        
        return created_version

    # Experiment operations
    def create_experiment(self, experiment: Experiment) -> Experiment:
        """Create a new experiment."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments (name, description, project_id, csv_filename)
                VALUES (?, ?, ?, ?)
                """,
                (experiment.name, experiment.description, experiment.project_id, experiment.csv_filename),
            )
            experiment.id = cursor.lastrowid
            
            # Get the created timestamps
            row = conn.execute(
                "SELECT upload_date, created_at, updated_at FROM experiments WHERE id = ?",
                (experiment.id,),
            ).fetchone()
            if row:
                experiment.upload_date = datetime.fromisoformat(row[0])
                experiment.created_at = datetime.fromisoformat(row[1])
                experiment.updated_at = datetime.fromisoformat(row[2])
            conn.commit()
        return experiment
    
    def get_experiment_by_id(self, experiment_id: int) -> Optional[Experiment]:
        """Get an experiment by its ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
            if row:
                return Experiment(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    project_id=row["project_id"],
                    upload_date=datetime.fromisoformat(row["upload_date"]),
                    csv_filename=row["csv_filename"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None
    
    def list_experiments(self, project_id: Optional[int] = None) -> List[Experiment]:
        """Get all experiments, optionally filtered by project."""
        experiments = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            if project_id is not None:
                query = "SELECT * FROM experiments WHERE project_id = ? ORDER BY upload_date DESC"
                rows = conn.execute(query, (project_id,)).fetchall()
            else:
                query = "SELECT * FROM experiments ORDER BY upload_date DESC"
                rows = conn.execute(query).fetchall()
            
            for row in rows:
                experiments.append(
                    Experiment(
                        id=row["id"],
                        name=row["name"],
                        description=row["description"],
                        project_id=row["project_id"],
                        upload_date=datetime.fromisoformat(row["upload_date"]),
                        csv_filename=row["csv_filename"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
        return experiments
    
    # Measurement operations
    def create_measurement(self, measurement: Measurement) -> Measurement:
        """Create a new measurement."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO measurements (experiment_id, sample_id, biomarker_version_id, value)
                VALUES (?, ?, ?, ?)
                """,
                (measurement.experiment_id, measurement.sample_id, 
                 measurement.biomarker_version_id, measurement.value),
            )
            measurement.id = cursor.lastrowid
            
            # Get the created timestamp
            row = conn.execute(
                "SELECT created_at FROM measurements WHERE id = ?",
                (measurement.id,),
            ).fetchone()
            if row:
                measurement.created_at = datetime.fromisoformat(row[0])
            conn.commit()
        return measurement
    
    def get_measurements_by_experiment(self, experiment_id: int) -> List[Measurement]:
        """Get all measurements for an experiment."""
        measurements = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM measurements WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchall()
            
            for row in rows:
                measurements.append(
                    Measurement(
                        id=row["id"],
                        experiment_id=row["experiment_id"],
                        sample_id=row["sample_id"],
                        biomarker_version_id=row["biomarker_version_id"],
                        value=row["value"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
        return measurements
    
    def get_measurements_by_sample(self, sample_id: int) -> List[Measurement]:
        """Get all measurements for a sample across all experiments."""
        measurements = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM measurements WHERE sample_id = ?",
                (sample_id,),
            ).fetchall()
            
            for row in rows:
                measurements.append(
                    Measurement(
                        id=row["id"],
                        experiment_id=row["experiment_id"],
                        sample_id=row["sample_id"],
                        biomarker_version_id=row["biomarker_version_id"],
                        value=row["value"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
        return measurements
    
    def get_measurements_by_biomarker(self, biomarker_id: int) -> List[Measurement]:
        """Get all measurements for a biomarker across all experiments and versions."""
        measurements = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT m.* FROM measurements m
                JOIN biomarker_versions bv ON m.biomarker_version_id = bv.id
                WHERE bv.biomarker_id = ?
                """,
                (biomarker_id,),
            ).fetchall()
            
            for row in rows:
                measurements.append(
                    Measurement(
                        id=row["id"],
                        experiment_id=row["experiment_id"],
                        sample_id=row["sample_id"],
                        biomarker_version_id=row["biomarker_version_id"],
                        value=row["value"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
        return measurements
    
    def get_measurements_by_biomarker_version(self, biomarker_version_id: int) -> List[Measurement]:
        """Get all measurements for a specific biomarker version."""
        measurements = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM measurements WHERE biomarker_version_id = ?",
                (biomarker_version_id,),
            ).fetchall()
            
            for row in rows:
                measurements.append(
                    Measurement(
                        id=row["id"],
                        experiment_id=row["experiment_id"],
                        sample_id=row["sample_id"],
                        biomarker_version_id=row["biomarker_version_id"],
                        value=row["value"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
        return measurements
    
    def get_measurement_summary(self, project_id: Optional[int] = None) -> Dict[str, Any]:
        """Get summary statistics about measurements."""
        with self._get_connection() as conn:
            if project_id is not None:
                query = """
                    SELECT 
                        COUNT(DISTINCT e.id) as experiment_count,
                        COUNT(DISTINCT m.sample_id) as sample_count,
                        COUNT(DISTINCT bv.biomarker_id) as biomarker_count,
                        COUNT(DISTINCT m.biomarker_version_id) as biomarker_version_count,
                        COUNT(m.id) as measurement_count
                    FROM measurements m
                    JOIN experiments e ON m.experiment_id = e.id
                    JOIN biomarker_versions bv ON m.biomarker_version_id = bv.id
                    WHERE e.project_id = ?
                """
                row = conn.execute(query, (project_id,)).fetchone()
            else:
                query = """
                    SELECT 
                        COUNT(DISTINCT experiment_id) as experiment_count,
                        COUNT(DISTINCT sample_id) as sample_count,
                        COUNT(DISTINCT bv.biomarker_id) as biomarker_count,
                        COUNT(DISTINCT biomarker_version_id) as biomarker_version_count,
                        COUNT(m.id) as measurement_count
                    FROM measurements m
                    JOIN biomarker_versions bv ON m.biomarker_version_id = bv.id
                """
                row = conn.execute(query).fetchone()
            
            return {
                "experiment_count": row[0],
                "sample_count": row[1], 
                "biomarker_count": row[2],
                "biomarker_version_count": row[3],
                "measurement_count": row[4]
            }
    
    def get_biomarker_data_for_analysis(self, biomarker_id: int) -> Dict[str, Any]:
        """
        Get all measurement data for a biomarker across all versions for bioinformatics analysis.
        This groups data by the biomarker regardless of version.
        """
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            
            # Get biomarker info
            biomarker = self.get_biomarker_by_id(biomarker_id)
            if not biomarker:
                return None
            
            # Get all measurements for this biomarker across versions
            rows = conn.execute(
                """
                SELECT 
                    m.id, m.experiment_id, m.sample_id, m.value, m.created_at,
                    bv.version, bv.description as version_description,
                    s.code as sample_code,
                    e.name as experiment_name
                FROM measurements m
                JOIN biomarker_versions bv ON m.biomarker_version_id = bv.id
                JOIN samples s ON m.sample_id = s.id
                JOIN experiments e ON m.experiment_id = e.id
                WHERE bv.biomarker_id = ?
                ORDER BY m.experiment_id, m.sample_id
                """,
                (biomarker_id,),
            ).fetchall()
            
            measurements_data = []
            for row in rows:
                measurements_data.append({
                    "measurement_id": row["id"],
                    "experiment_id": row["experiment_id"],
                    "experiment_name": row["experiment_name"],
                    "sample_id": row["sample_id"],
                    "sample_code": row["sample_code"],
                    "value": row["value"],
                    "version": row["version"],
                    "version_description": row["version_description"],
                    "created_at": row["created_at"]
                })
            
            return {
                "biomarker": biomarker,
                "measurements": measurements_data,
                "total_measurements": len(measurements_data),
                "unique_experiments": len(set(m["experiment_id"] for m in measurements_data)),
                "unique_samples": len(set(m["sample_id"] for m in measurements_data)),
                "versions_used": list(set(m["version"] for m in measurements_data))
            }