"""
Base command interface for CLI handlers.
"""
from abc import ABC, abstractmethod
import argparse
from typing import Optional


class BaseCommandHandler(ABC):
    """Base class for command handlers."""
    
    @abstractmethod
    def add_commands(self, subparsers) -> None:
        """Add command parsers to the subparsers."""
        pass
    
    @abstractmethod
    def handle_command(self, args, db_path: str) -> bool:
        """Handle the command execution."""
        pass
    
    def get_project_id_from_args(self, args, current_project_manager, db_path: str) -> Optional[int]:
        """Helper to get project ID from args or current project."""
        from ..projects import ProjectDatabase
        
        if hasattr(args, 'project') and args.project:
            # Use specified project
            project_db = ProjectDatabase(db_path)
            project = project_db.get_project_by_code(args.project)
            if not project:
                print(f"❌ Project with code '{args.project}' not found.")
                return None
            return project.id
        elif current_project_manager.is_project_active():
            # Use current project
            return current_project_manager.get_current_project_id()
        else:
            print("❌ No project specified and no current project set. Use 'mmk-kb use <project_code>' first.")
            return None