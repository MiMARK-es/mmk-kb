"""
Database utilities for MMK Knowledge Base.
Provides utilities for backup, cleanup, and database management.
"""

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import Environment, config, get_database_path


class DatabaseUtils:
    """Utilities for database management, backup, and cleanup."""

    @staticmethod
    def backup_database(
        source_env: Optional[Environment] = None,
        backup_dir: str = "backups",
        include_timestamp: bool = True,
    ) -> str:
        """
        Create a backup of the specified database.

        Args:
            source_env: Environment to backup. Defaults to current environment.
            backup_dir: Directory to store backups in.
            include_timestamp: Whether to include timestamp in backup filename.

        Returns:
            Path to the created backup file.

        Raises:
            FileNotFoundError: If source database doesn't exist.
        """
        source_path = Path(get_database_path(source_env))

        if not source_path.exists():
            raise FileNotFoundError(f"Database file not found: {source_path}")

        # Create backup directory if it doesn't exist
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)

        # Generate backup filename
        env_name = (source_env or config.current_environment).value
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"mmk_kb_{env_name}_{timestamp}.db"
        else:
            backup_filename = f"mmk_kb_{env_name}_backup.db"

        backup_file = backup_path / backup_filename

        # Create backup using SQLite backup API for consistency
        with sqlite3.connect(str(source_path)) as source_conn:
            with sqlite3.connect(str(backup_file)) as backup_conn:
                source_conn.backup(backup_conn)

        return str(backup_file)

    @staticmethod
    def restore_database(
        backup_path: str, target_env: Optional[Environment] = None, confirm: bool = True
    ) -> bool:
        """
        Restore database from backup.

        Args:
            backup_path: Path to backup file.
            target_env: Environment to restore to. Defaults to current environment.
            confirm: Whether to ask for confirmation before overwriting.

        Returns:
            True if restoration was successful.

        Raises:
            FileNotFoundError: If backup file doesn't exist.
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        target_path = Path(get_database_path(target_env))

        if confirm and target_path.exists():
            response = input(f"⚠️  This will overwrite {target_path}. Continue? (y/N): ")
            if response.lower() != "y":
                print("❌ Restoration cancelled.")
                return False

        # Restore using SQLite backup API
        with sqlite3.connect(str(backup_file)) as backup_conn:
            with sqlite3.connect(str(target_path)) as target_conn:
                backup_conn.backup(target_conn)

        print(f"✅ Database restored from {backup_path} to {target_path}")
        return True

    @staticmethod
    def clean_database(env: Optional[Environment] = None, confirm: bool = True) -> bool:
        """
        Clean (delete) database file for specified environment.

        Args:
            env: Environment to clean. Defaults to current environment.
            confirm: Whether to ask for confirmation before deletion.

        Returns:
            True if database was deleted.
        """
        db_path = Path(get_database_path(env))
        env_name = (env or config.current_environment).value

        if not db_path.exists():
            print(f"ℹ️  Database file doesn't exist: {db_path}")
            return False

        if confirm:
            response = input(f"⚠️  Delete {env_name} database ({db_path})? (y/N): ")
            if response.lower() != "y":
                print("❌ Deletion cancelled.")
                return False

        db_path.unlink()
        print(f"✅ Deleted {env_name} database: {db_path}")
        return True

    @staticmethod
    def clean_all_test_databases(confirm: bool = True) -> int:
        """
        Clean all test-related database files.

        Args:
            confirm: Whether to ask for confirmation before deletion.

        Returns:
            Number of files deleted.
        """
        patterns = ["test_*.db", "*_test.db", "test*.db"]
        deleted_count = 0

        for pattern in patterns:
            for db_file in Path.cwd().glob(pattern):
                if confirm:
                    response = input(f"⚠️  Delete test database {db_file}? (y/N): ")
                    if response.lower() != "y":
                        continue

                db_file.unlink()
                print(f"✅ Deleted test database: {db_file}")
                deleted_count += 1

        return deleted_count

    @staticmethod
    def list_database_status() -> dict:
        """
        Get status of all database environments.

        Returns:
            Dictionary with environment status information.
        """
        status = {}
        db_files = config.list_database_files()

        for env, (path, exists) in db_files.items():
            file_path = Path(path)

            env_status = {
                "path": path,
                "exists": exists,
                "size_bytes": file_path.stat().st_size if exists else 0,
                "modified": (
                    datetime.fromtimestamp(file_path.stat().st_mtime)
                    if exists
                    else None
                ),
            }

            if exists:
                # Get row count from projects table
                try:
                    with sqlite3.connect(path) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM projects")
                        env_status["project_count"] = cursor.fetchone()[0]
                except sqlite3.Error:
                    env_status["project_count"] = "Error"
            else:
                env_status["project_count"] = 0

            status[env.value] = env_status

        return status

    @staticmethod
    def copy_database(
        source_env: Environment, target_env: Environment, confirm: bool = True
    ) -> bool:
        """
        Copy database from one environment to another.

        Args:
            source_env: Source environment.
            target_env: Target environment.
            confirm: Whether to ask for confirmation before overwriting.

        Returns:
            True if copy was successful.
        """
        source_path = Path(get_database_path(source_env))
        target_path = Path(get_database_path(target_env))

        if not source_path.exists():
            print(f"❌ Source database doesn't exist: {source_path}")
            return False

        if confirm and target_path.exists():
            response = input(
                f"⚠️  Overwrite {target_env.value} database with {source_env.value} data? (y/N): "
            )
            if response.lower() != "y":
                print("❌ Copy cancelled.")
                return False

        # Use SQLite backup API for consistency
        with sqlite3.connect(str(source_path)) as source_conn:
            with sqlite3.connect(str(target_path)) as target_conn:
                source_conn.backup(target_conn)

        print(f"✅ Copied {source_env.value} database to {target_env.value}")
        return True

    @staticmethod
    def vacuum_database(env: Optional[Environment] = None) -> bool:
        """
        Vacuum database to reclaim space and optimize.

        Args:
            env: Environment to vacuum. Defaults to current environment.

        Returns:
            True if vacuum was successful.
        """
        db_path = get_database_path(env)
        env_name = (env or config.current_environment).value

        if not Path(db_path).exists():
            print(f"❌ Database doesn't exist: {db_path}")
            return False

        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("VACUUM")
            print(f"✅ Vacuumed {env_name} database")
            return True
        except sqlite3.Error as e:
            print(f"❌ Failed to vacuum database: {e}")
            return False
