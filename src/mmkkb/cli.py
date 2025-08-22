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
from .samples import Sample, SampleDatabase, current_project_manager


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


def use_project(db_path, code):
    """Set the current active project."""
    if current_project_manager.use_project(code):
        print(f"✅ Now using project: {code}")
        return True
    else:
        print(f"❌ Project with code '{code}' not found.")
        return False


def show_current_project():
    """Show the current active project."""
    if current_project_manager.is_project_active():
        code = current_project_manager.get_current_project_code()
        project_id = current_project_manager.get_current_project_id()
        
        # Get project details
        db = ProjectDatabase()
        project = db.get_project_by_id(project_id)
        
        if project:
            print(f"🎯 Current Project: {project.code} - {project.name}")
            
            # Show sample count
            sample_db = SampleDatabase()
            sample_count = sample_db.count_samples(project_id)
            print(f"📊 Samples: {sample_count}")
        else:
            print("❌ Current project not found in database.")
    else:
        print("ℹ️  No project currently active. Use 'mmk-kb use <project_code>' to set one.")


def clear_current_project():
    """Clear the current active project."""
    if current_project_manager.is_project_active():
        code = current_project_manager.get_current_project_code()
        current_project_manager.clear_current_project()
        print(f"✅ Cleared current project: {code}")
    else:
        print("ℹ️  No project currently active.")


def list_samples(db_path, project_code=None):
    """List samples for current or specified project."""
    sample_db = SampleDatabase(db_path)
    
    if project_code:
        # Use specified project
        project_db = ProjectDatabase(db_path)
        project = project_db.get_project_by_code(project_code)
        if not project:
            print(f"❌ Project with code '{project_code}' not found.")
            return False
        project_id = project.id
        project_name = f"{project.code} - {project.name}"
    elif current_project_manager.is_project_active():
        # Use current project
        project_id = current_project_manager.get_current_project_id()
        project_name = current_project_manager.get_current_project_code()
    else:
        print("❌ No project specified and no current project set. Use 'mmk-kb use <project_code>' first.")
        return False
    
    samples = sample_db.list_samples(project_id)
    if not samples:
        print(f"No samples found in project {project_name}.")
        return True
    
    print(f"\n🧪 Found {len(samples)} samples in project {project_name}:\n")
    print(f"{'Code':<12} {'Age':<4} {'BMI':<6} {'Dx':<3} {'Origin':<15} {'Center':<15} {'Processing':<10}")
    print("-" * 80)
    
    for sample in samples:
        dx_str = "Dis" if sample.dx else "Ben"
        print(
            f"{sample.code:<12} {sample.age:<4} {sample.bmi:<6.1f} {dx_str:<3} "
            f"{sample.dx_origin[:14]:<15} {sample.collection_center[:14]:<15} {sample.processing_time:<10}"
        )
    return True


def create_sample(db_path, code, age, bmi, dx, dx_origin, collection_center, processing_time, project_code=None):
    """Create a new sample."""
    sample_db = SampleDatabase(db_path)
    
    if project_code:
        # Use specified project
        project_db = ProjectDatabase(db_path)
        project = project_db.get_project_by_code(project_code)
        if not project:
            print(f"❌ Project with code '{project_code}' not found.")
            return False
        project_id = project.id
    elif current_project_manager.is_project_active():
        # Use current project
        project_id = current_project_manager.get_current_project_id()
    else:
        print("❌ No project specified and no current project set. Use 'mmk-kb use <project_code>' first.")
        return False
    
    # Check if sample already exists in this project
    existing = sample_db.get_sample_by_code(code, project_id)
    if existing:
        print(f"❌ Sample with code '{code}' already exists in this project.")
        return False
    
    # Convert dx string to boolean
    dx_bool = dx.lower() in ('1', 'true', 'disease', 'dis')
    
    sample = Sample(
        code=code,
        age=int(age),
        bmi=float(bmi),
        dx=dx_bool,
        dx_origin=dx_origin,
        collection_center=collection_center,
        processing_time=int(processing_time),
        project_id=project_id
    )
    
    try:
        created_sample = sample_db.create_sample(sample)
        dx_str = "Disease" if created_sample.dx else "Benign"
        print(f"✅ Created sample: {created_sample.code} (Age: {created_sample.age}, BMI: {created_sample.bmi}, Dx: {dx_str})")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample: {e}")
        return False


def show_sample(db_path, code, project_code=None):
    """Show details of a specific sample."""
    sample_db = SampleDatabase(db_path)
    
    if project_code:
        # Use specified project
        project_db = ProjectDatabase(db_path)
        project = project_db.get_project_by_code(project_code)
        if not project:
            print(f"❌ Project with code '{project_code}' not found.")
            return False
        project_id = project.id
    elif current_project_manager.is_project_active():
        # Use current project
        project_id = current_project_manager.get_current_project_id()
    else:
        print("❌ No project specified and no current project set. Use 'mmk-kb use <project_code>' first.")
        return False
    
    sample = sample_db.get_sample_by_code(code, project_id)
    if not sample:
        print(f"❌ Sample with code '{code}' not found in current project.")
        return False
    
    dx_str = "Disease" if sample.dx else "Benign"
    print(f"\n🧪 Sample Details:")
    print(f"Code: {sample.code}")
    print(f"Age: {sample.age}")
    print(f"BMI: {sample.bmi}")
    print(f"Diagnosis: {dx_str}")
    print(f"Diagnosis Origin: {sample.dx_origin}")
    print(f"Collection Center: {sample.collection_center}")
    print(f"Processing Time: {sample.processing_time}")
    print(f"Project ID: {sample.project_id}")
    print(f"Created: {sample.created_at}")
    print(f"Updated: {sample.updated_at}")
    return True


def update_sample(db_path, code, project_code=None, **kwargs):
    """Update an existing sample."""
    sample_db = SampleDatabase(db_path)
    
    if project_code:
        # Use specified project
        project_db = ProjectDatabase(db_path)
        project = project_db.get_project_by_code(project_code)
        if not project:
            print(f"❌ Project with code '{project_code}' not found.")
            return False
        project_id = project.id
    elif current_project_manager.is_project_active():
        # Use current project
        project_id = current_project_manager.get_current_project_id()
    else:
        print("❌ No project specified and no current project set. Use 'mmk-kb use <project_code>' first.")
        return False
    
    sample = sample_db.get_sample_by_code(code, project_id)
    if not sample:
        print(f"❌ Sample with code '{code}' not found in current project.")
        return False
    
    # Update fields if provided
    if 'age' in kwargs and kwargs['age'] is not None:
        sample.age = int(kwargs['age'])
    if 'bmi' in kwargs and kwargs['bmi'] is not None:
        sample.bmi = float(kwargs['bmi'])
    if 'dx' in kwargs and kwargs['dx'] is not None:
        sample.dx = kwargs['dx'].lower() in ('1', 'true', 'disease', 'dis')
    if 'dx_origin' in kwargs and kwargs['dx_origin'] is not None:
        sample.dx_origin = kwargs['dx_origin']
    if 'collection_center' in kwargs and kwargs['collection_center'] is not None:
        sample.collection_center = kwargs['collection_center']
    if 'processing_time' in kwargs and kwargs['processing_time'] is not None:
        sample.processing_time = int(kwargs['processing_time'])
    
    try:
        updated_sample = sample_db.update_sample(sample)
        print(f"✅ Updated sample: {updated_sample.code}")
        return True
    except Exception as e:
        print(f"❌ Failed to update sample: {e}")
        return False


def delete_sample(db_path, code, project_code=None):
    """Delete a sample."""
    sample_db = SampleDatabase(db_path)
    
    if project_code:
        # Use specified project
        project_db = ProjectDatabase(db_path)
        project = project_db.get_project_by_code(project_code)
        if not project:
            print(f"❌ Project with code '{project_code}' not found.")
            return False
        project_id = project.id
    elif current_project_manager.is_project_active():
        # Use current project
        project_id = current_project_manager.get_current_project_id()
    else:
        print("❌ No project specified and no current project set. Use 'mmk-kb use <project_code>' first.")
        return False
    
    # Check if sample exists
    sample = sample_db.get_sample_by_code(code, project_id)
    if not sample:
        print(f"❌ Sample with code '{code}' not found in current project.")
        return False
    
    # Confirm deletion
    dx_str = "Disease" if sample.dx else "Benign"
    response = input(
        f"⚠️  Are you sure you want to delete sample '{code}' (Age: {sample.age}, Dx: {dx_str})? (y/N): "
    )
    if response.lower() != "y":
        print("❌ Deletion cancelled.")
        return False
    
    success = sample_db.delete_sample_by_code(code, project_id)
    if success:
        print(f"✅ Deleted sample: {code}")
    else:
        print(f"❌ Failed to delete sample: {code}")
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

    # Project selection commands
    use_parser = subparsers.add_parser("use", help="Set current active project")
    use_parser.add_argument("code", help="Project code")

    current_parser = subparsers.add_parser("current", help="Show current active project")

    clear_parser = subparsers.add_parser("clear", help="Clear current active project")

    # Sample commands
    samples_list_parser = subparsers.add_parser("samples", help="List samples in current/specified project")
    samples_list_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")

    sample_create_parser = subparsers.add_parser("sample-create", help="Create a new sample")
    sample_create_parser.add_argument("code", help="Sample code")
    sample_create_parser.add_argument("age", type=int, help="Patient age")
    sample_create_parser.add_argument("bmi", type=float, help="Patient BMI")
    sample_create_parser.add_argument("dx", help="Diagnosis (0/benign or 1/disease)")
    sample_create_parser.add_argument("dx_origin", help="Diagnosis origin")
    sample_create_parser.add_argument("collection_center", help="Collection center")
    sample_create_parser.add_argument("processing_time", type=int, help="Processing time")
    sample_create_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")

    sample_show_parser = subparsers.add_parser("sample-show", help="Show sample details")
    sample_show_parser.add_argument("code", help="Sample code")
    sample_show_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")

    sample_update_parser = subparsers.add_parser("sample-update", help="Update a sample")
    sample_update_parser.add_argument("code", help="Sample code")
    sample_update_parser.add_argument("--age", type=int, help="Patient age")
    sample_update_parser.add_argument("--bmi", type=float, help="Patient BMI")
    sample_update_parser.add_argument("--dx", help="Diagnosis (0/benign or 1/disease)")
    sample_update_parser.add_argument("--dx-origin", help="Diagnosis origin")
    sample_update_parser.add_argument("--collection-center", help="Collection center")
    sample_update_parser.add_argument("--processing-time", type=int, help="Processing time")
    sample_update_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")

    sample_delete_parser = subparsers.add_parser("sample-delete", help="Delete a sample")
    sample_delete_parser.add_argument("code", help="Sample code")
    sample_delete_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")

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

        # Project selection commands
        elif args.command == "use":
            use_project(db_path, args.code)
        elif args.command == "current":
            show_current_project()
        elif args.command == "clear":
            clear_current_project()

        # Sample commands
        elif args.command == "samples":
            list_samples(db_path, args.project)
        elif args.command == "sample-create":
            create_sample(
                db_path, args.code, args.age, args.bmi, args.dx, 
                args.dx_origin, args.collection_center, args.processing_time, args.project
            )
        elif args.command == "sample-show":
            show_sample(db_path, args.code, args.project)
        elif args.command == "sample-update":
            update_sample(
                db_path, args.code, args.project,
                age=args.age, bmi=args.bmi, dx=args.dx,
                dx_origin=getattr(args, 'dx_origin', None),
                collection_center=getattr(args, 'collection_center', None),
                processing_time=getattr(args, 'processing_time', None)
            )
        elif args.command == "sample-delete":
            delete_sample(db_path, args.code, args.project)

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
