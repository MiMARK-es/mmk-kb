"""
Database configuration and environment management for MMK Knowledge Base.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional


class Environment(Enum):
    """Database environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    TESTING = "testing"
    PRODUCTION = "production"


class DatabaseConfig:
    """Database configuration manager."""

    # Default database paths for different environments
    DEFAULT_PATHS = {
        Environment.DEVELOPMENT: "mmk_kb.db",
        Environment.STAGING: "mmk_kb_staging.db",
        Environment.TESTING: "test_mmk_kb.db",
        Environment.PRODUCTION: "mmk_kb_production.db",
    }

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize database configuration.

        Args:
            base_dir: Base directory for database files. Defaults to current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._current_env = self._detect_environment()

    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables."""
        env_name = os.getenv("MMK_KB_ENV", "development").lower()
        try:
            return Environment(env_name)
        except ValueError:
            return Environment.DEVELOPMENT

    def get_db_path(self, env: Optional[Environment] = None) -> str:
        """
        Get database path for specified environment.

        Args:
            env: Environment to get path for. Defaults to current environment.

        Returns:
            Full path to database file.
        """
        env = env or self._current_env
        filename = self.DEFAULT_PATHS[env]
        return str(self.base_dir / filename)

    def set_environment(self, env: Environment) -> None:
        """Set current environment."""
        self._current_env = env
        os.environ["MMK_KB_ENV"] = env.value

    @property
    def current_environment(self) -> Environment:
        """Get current environment."""
        return self._current_env

    def list_database_files(self) -> dict[Environment, tuple[str, bool]]:
        """
        List all database files and their existence status.

        Returns:
            Dictionary mapping environment to (path, exists) tuples.
        """
        result = {}
        for env in Environment:
            path = self.get_db_path(env)
            exists = Path(path).exists()
            result[env] = (path, exists)
        return result


# Global configuration instance
config = DatabaseConfig()


def get_database_path(env: Optional[Environment] = None) -> str:
    """
    Get database path for specified environment.

    Args:
        env: Environment to get path for. Defaults to current environment.

    Returns:
        Full path to database file.
    """
    return config.get_db_path(env)


def set_environment(env: Environment) -> None:
    """Set current database environment."""
    config.set_environment(env)


def get_current_environment() -> Environment:
    """Get current database environment."""
    return config.current_environment
