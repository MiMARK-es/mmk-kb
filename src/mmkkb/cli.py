#!/usr/bin/env python3
"""
Command Line Interface for MMK Knowledge Base
"""
import argparse
import sys
from datetime import datetime

from .config import (Environment, get_current_environment, get_database_path,
                     set_environment)
from .db_utils import DatabaseUtils
from .projects import Project, ProjectDatabase


def list_projects(db_path):
    """List all projects."""
    db = ProjectDatabase(db_path)
    projects = db.list_projects()

    if not projects:
        print("No projects found.")
        return

    print(f"\n📋 Found {len(projects)} projects:\n")
    print(f"{'Code':<10} {'Name':<25} {'Creator':<15} {'Created':<19}")
    print("-" * 70)

    for project in projects:
        created_str = (
            project.created_at.strftime("%Y-%m-%d %H:%M")
            if project.created_at
            else "Unknown"
        )
        print(
            f"{project.code:<10} {project.name[:24]:<25} {project.creator[:14]:<15} {created_str:<19}"
        )


def create_project(db_path, code, name, description, creator):
    """Create a new project."""
    db = ProjectDatabase(db_path)

    # Check if project already exists
    existing = db.get_project_by_code(code)
    if existing:
        print(f"❌ Project with code '{code}' already exists.")
        return False

    project = Project(code=code, name=name, description=description, creator=creator)
    created_project = db.create_project(project)

    print(f"✅ Created project: {created_project.code} - {created_project.name}")
    return True


def show_project(db_path, code):
    """Show details of a specific project."""
    db = ProjectDatabase(db_path)
    project = db.get_project_by_code(code)

    if not project:
        print(f"❌ Project with code '{code}' not found.")
        return False

    print(f"\n📝 Project Details:")
    print(f"Code: {project.code}")
    print(f"Name: {project.name}")
    print(f"Creator: {project.creator}")
    print(f"Description: {project.description}")
    print(f"Created: {project.created_at}")
    print(f"Updated: {project.updated_at}")
    return True


def delete_project(db_path, code):
    """Delete a project."""
    db = ProjectDatabase(db_path)

    # Check if project exists
    project = db.get_project_by_code(code)
    if not project:
        print(f"❌ Project with code '{code}' not found.")
        return False

    # Confirm deletion
    response = input(
        f"⚠️  Are you sure you want to delete project '{code}' - '{project.name}'? (y/N): "
    )
    if response.lower() != "y":
        print("❌ Deletion cancelled.")
        return False

    success = db.delete_project(code)
    if success:
        print(f"✅ Deleted project: {code}")
    else:
        print(f"❌ Failed to delete project: {code}")

    return success


def show_environment():
    """Show current environment and database status."""
    current_env = get_current_environment()
    current_path = get_database_path()

    print(f"🌍 Current Environment: {current_env.value}")
    print(f"📁 Database Path: {current_path}")
    print()

    # Show status of all environments
    status = DatabaseUtils.list_database_status()

    print("📊 Database Status:")
    print(f"{'Environment':<12} {'Status':<8} {'Projects':<9} {'Size':<10} {'Path'}")
    print("-" * 75)

    for env_name, info in status.items():
        status_icon = "✅" if info["exists"] else "❌"
        size_str = f"{info['size_bytes']} B" if info["exists"] else "N/A"

        print(
            f"{env_name:<12} {status_icon:<8} {info['project_count']:<9} {size_str:<10} {info['path']}"
        )


def set_env(env_name):
    """Set current environment."""
    try:
        env = Environment(env_name.lower())
        set_environment(env)
        print(f"✅ Environment set to: {env.value}")
        print(f"📁 Database path: {get_database_path()}")
    except ValueError:
        valid_envs = [e.value for e in Environment]
        print(f"❌ Invalid environment. Valid options: {', '.join(valid_envs)}")


def backup_db(env_name, backup_dir):
    """Backup database."""
    try:
        env = Environment(env_name.lower()) if env_name else None
        backup_path = DatabaseUtils.backup_database(env, backup_dir)
        print(f"✅ Database backup created: {backup_path}")
    except Exception as e:
        print(f"❌ Backup failed: {e}")


def restore_db(backup_path, env_name):
    """Restore database from backup."""
    try:
        env = Environment(env_name.lower()) if env_name else None
        success = DatabaseUtils.restore_database(backup_path, env)
        if success:
            print(f"✅ Database restored successfully")
    except Exception as e:
        print(f"❌ Restore failed: {e}")


def clean_db(env_name):
    """Clean database."""
    try:
        env = Environment(env_name.lower()) if env_name else None
        DatabaseUtils.clean_database(env)
    except ValueError:
        valid_envs = [e.value for e in Environment]
        print(f"❌ Invalid environment. Valid options: {', '.join(valid_envs)}")


def clean_test_dbs():
    """Clean all test databases."""
    count = DatabaseUtils.clean_all_test_databases()
    print(f"✅ Cleaned {count} test database files")


def copy_db(source_env, target_env):
    """Copy database between environments."""
    try:
        source = Environment(source_env.lower())
        target = Environment(target_env.lower())
        DatabaseUtils.copy_database(source, target)
    except ValueError:
        valid_envs = [e.value for e in Environment]
        print(f"❌ Invalid environment. Valid options: {', '.join(valid_envs)}")


def main():
    """Main CLI entry point."""
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

    # Project commands
    list_parser = subparsers.add_parser("list", help="List all projects")

    create_parser = subparsers.add_parser("create", help="Create a new project")
    create_parser.add_argument("code", help="Project code (unique identifier)")
    create_parser.add_argument("name", help="Project name")
    create_parser.add_argument("description", help="Project description")
    create_parser.add_argument("creator", help="Project creator")

    show_parser = subparsers.add_parser("show", help="Show project details")
    show_parser.add_argument("code", help="Project code")

    delete_parser = subparsers.add_parser("delete", help="Delete a project")
    delete_parser.add_argument("code", help="Project code")

    # Environment commands
    env_parser = subparsers.add_parser("env", help="Show environment status")

    setenv_parser = subparsers.add_parser("setenv", help="Set current environment")
    setenv_parser.add_argument("environment", choices=[e.value for e in Environment])

    # Database management commands
    backup_parser = subparsers.add_parser("backup", help="Backup database")
    backup_parser.add_argument("--env", help="Environment to backup (default: current)")
    backup_parser.add_argument("--dir", default="backups", help="Backup directory")

    restore_parser = subparsers.add_parser(
        "restore", help="Restore database from backup"
    )
    restore_parser.add_argument("backup_path", help="Path to backup file")
    restore_parser.add_argument("--env", help="Target environment (default: current)")

    clean_parser = subparsers.add_parser("clean", help="Clean database")
    clean_parser.add_argument("--env", help="Environment to clean (default: current)")

    subparsers.add_parser("clean-tests", help="Clean all test databases")

    copy_parser = subparsers.add_parser(
        "copy", help="Copy database between environments"
    )
    copy_parser.add_argument("source", choices=[e.value for e in Environment])
    copy_parser.add_argument("target", choices=[e.value for e in Environment])

    vacuum_parser = subparsers.add_parser("vacuum", help="Vacuum database to optimize")
    vacuum_parser.add_argument("--env", help="Environment to vacuum (default: current)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Set temporary environment if specified
    if args.env:
        set_environment(Environment(args.env))

    # Determine database path
    db_path = args.db or get_database_path()

    try:
        # Project commands
        if args.command == "list":
            list_projects(db_path)
        elif args.command == "create":
            create_project(
                db_path, args.code, args.name, args.description, args.creator
            )
        elif args.command == "show":
            show_project(db_path, args.code)
        elif args.command == "delete":
            delete_project(db_path, args.code)

        # Environment commands
        elif args.command == "env":
            show_environment()
        elif args.command == "setenv":
            set_env(args.environment)

        # Database management commands
        elif args.command == "backup":
            backup_db(args.env, args.dir)
        elif args.command == "restore":
            restore_db(args.backup_path, args.env)
        elif args.command == "clean":
            clean_db(args.env)
        elif args.command == "clean-tests":
            clean_test_dbs()
        elif args.command == "copy":
            copy_db(args.source, args.target)
        elif args.command == "vacuum":
            env = Environment(args.env) if args.env else None
            DatabaseUtils.vacuum_database(env)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
