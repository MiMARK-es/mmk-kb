"""
Project management commands for the CLI.
"""
from .base import BaseCommandHandler
from ..projects import Project, ProjectDatabase
from ..samples import SampleDatabase


class ProjectCommandHandler(BaseCommandHandler):
    """Handler for project-related commands."""
    
    def add_commands(self, subparsers) -> None:
        """Add project command parsers."""
        # List projects
        list_parser = subparsers.add_parser("list", help="List all projects")
        
        # Create project
        create_parser = subparsers.add_parser("create", help="Create a new project")
        create_parser.add_argument("code", help="Project code (unique identifier)")
        create_parser.add_argument("name", help="Project name")
        create_parser.add_argument("description", help="Project description")
        create_parser.add_argument("creator", help="Project creator")
        
        # Show project
        show_parser = subparsers.add_parser("show", help="Show project details")
        show_parser.add_argument("code", help="Project code")
        
        # Delete project
        delete_parser = subparsers.add_parser("delete", help="Delete a project")
        delete_parser.add_argument("code", help="Project code")
        
        # Project selection commands
        use_parser = subparsers.add_parser("use", help="Set current active project")
        use_parser.add_argument("code", help="Project code")
        
        current_parser = subparsers.add_parser("current", help="Show current active project")
        
        clear_parser = subparsers.add_parser("clear", help="Clear current active project")
    
    def handle_command(self, args, db_path: str) -> bool:
        """Handle project commands."""
        from ..samples import current_project_manager
        
        if args.command == "list":
            return self._list_projects(db_path)
        elif args.command == "create":
            return self._create_project(db_path, args.code, args.name, args.description, args.creator)
        elif args.command == "show":
            return self._show_project(db_path, args.code)
        elif args.command == "delete":
            return self._delete_project(db_path, args.code)
        elif args.command == "use":
            return self._use_project(db_path, args.code)
        elif args.command == "current":
            return self._show_current_project()
        elif args.command == "clear":
            return self._clear_current_project()
        return False
    
    def _list_projects(self, db_path: str) -> bool:
        """List all projects."""
        db = ProjectDatabase(db_path)
        projects = db.list_projects()

        if not projects:
            print("No projects found.")
            return True

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
        return True
    
    def _create_project(self, db_path: str, code: str, name: str, description: str, creator: str) -> bool:
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
    
    def _show_project(self, db_path: str, code: str) -> bool:
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
    
    def _delete_project(self, db_path: str, code: str) -> bool:
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
    
    def _use_project(self, db_path: str, code: str) -> bool:
        """Set the current active project."""
        from ..samples import current_project_manager
        
        if current_project_manager.use_project(code):
            print(f"✅ Now using project: {code}")
            return True
        else:
            print(f"❌ Project with code '{code}' not found.")
            return False
    
    def _show_current_project(self) -> bool:
        """Show the current active project."""
        from ..samples import current_project_manager
        
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
        return True
    
    def _clear_current_project(self) -> bool:
        """Clear the current active project."""
        from ..samples import current_project_manager
        
        if current_project_manager.is_project_active():
            code = current_project_manager.get_current_project_code()
            current_project_manager.clear_current_project()
            print(f"✅ Cleared current project: {code}")
        else:
            print("ℹ️  No project currently active.")
        return True