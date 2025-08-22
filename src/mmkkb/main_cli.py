#!/usr/bin/env python3
"""
Modular Command Line Interface for MMK Knowledge Base
"""
import argparse
import sys
from typing import List

from .config import Environment, get_database_path, set_environment
from .cli.project_commands import ProjectCommandHandler
from .cli.sample_commands import SampleCommandHandler
from .cli.environment_commands import EnvironmentCommandHandler, DatabaseCommandHandler


class CLIManager:
    """Main CLI manager that coordinates all command handlers."""
    
    def __init__(self):
        """Initialize CLI manager with all command handlers."""
        self.handlers = [
            ProjectCommandHandler(),
            SampleCommandHandler(), 
            EnvironmentCommandHandler(),
            DatabaseCommandHandler(),
        ]
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(description="MMK Knowledge Base CLI")
        parser.add_argument(
            "--db", help="Database file path (overrides environment-based path)"
        )
        parser.add_argument(
            "--env",
            choices=[e.value for e in Environment],
            help="Set environment for this command",
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Let each handler add its commands
        for handler in self.handlers:
            handler.add_commands(subparsers)
        
        return parser
    
    def execute_command(self, args) -> bool:
        """Execute the command using the appropriate handler."""
        # Set temporary environment if specified
        if args.env:
            set_environment(Environment(args.env))

        # Determine database path
        db_path = args.db or get_database_path()

        # Find the handler that can process this command
        for handler in self.handlers:
            try:
                if handler.handle_command(args, db_path):
                    return True
            except Exception as e:
                print(f"❌ Error in {handler.__class__.__name__}: {e}")
                return False
        
        print(f"❌ Unknown command: {args.command}")
        return False


def main():
    """Main CLI entry point."""
    cli_manager = CLIManager()
    parser = cli_manager.create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        success = cli_manager.execute_command(args)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()