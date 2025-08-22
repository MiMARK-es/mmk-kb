"""
Sample management commands for the CLI.
"""
from .base import BaseCommandHandler
from ..projects import ProjectDatabase
from ..samples import Sample, SampleDatabase


class SampleCommandHandler(BaseCommandHandler):
    """Handler for sample-related commands."""
    
    def add_commands(self, subparsers) -> None:
        """Add sample command parsers."""
        # List samples
        samples_list_parser = subparsers.add_parser("samples", help="List samples in current/specified project")
        samples_list_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")

        # Create sample
        sample_create_parser = subparsers.add_parser("sample-create", help="Create a new sample")
        sample_create_parser.add_argument("code", help="Sample code")
        sample_create_parser.add_argument("age", type=int, help="Patient age")
        sample_create_parser.add_argument("bmi", type=float, help="Patient BMI")
        sample_create_parser.add_argument("dx", help="Diagnosis (0/benign or 1/disease)")
        sample_create_parser.add_argument("dx_origin", help="Diagnosis origin")
        sample_create_parser.add_argument("collection_center", help="Collection center")
        sample_create_parser.add_argument("processing_time", type=int, help="Processing time")
        sample_create_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")

        # Show sample
        sample_show_parser = subparsers.add_parser("sample-show", help="Show sample details")
        sample_show_parser.add_argument("code", help="Sample code")
        sample_show_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")

        # Update sample
        sample_update_parser = subparsers.add_parser("sample-update", help="Update a sample")
        sample_update_parser.add_argument("code", help="Sample code")
        sample_update_parser.add_argument("--age", type=int, help="Patient age")
        sample_update_parser.add_argument("--bmi", type=float, help="Patient BMI")
        sample_update_parser.add_argument("--dx", help="Diagnosis (0/benign or 1/disease)")
        sample_update_parser.add_argument("--dx-origin", help="Diagnosis origin")
        sample_update_parser.add_argument("--collection-center", help="Collection center")
        sample_update_parser.add_argument("--processing-time", type=int, help="Processing time")
        sample_update_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")

        # Delete sample
        sample_delete_parser = subparsers.add_parser("sample-delete", help="Delete a sample")
        sample_delete_parser.add_argument("code", help="Sample code")
        sample_delete_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")
    
    def handle_command(self, args, db_path: str) -> bool:
        """Handle sample commands."""
        if args.command == "samples":
            return self._list_samples(db_path, args.project)
        elif args.command == "sample-create":
            return self._create_sample(db_path, args)
        elif args.command == "sample-show":
            return self._show_sample(db_path, args.code, args.project)
        elif args.command == "sample-update":
            return self._update_sample(db_path, args)
        elif args.command == "sample-delete":
            return self._delete_sample(db_path, args.code, args.project)
        return False
    
    def _list_samples(self, db_path: str, project_code: str = None) -> bool:
        """List samples for current or specified project."""
        from ..samples import CurrentProjectManager
        
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
        else:
            # Use current project with correct database path
            current_project_manager = CurrentProjectManager(db_path)
            if current_project_manager.is_project_active():
                project_id = current_project_manager.get_current_project_id()
                project_code = current_project_manager.get_current_project_code()
                
                # Get project details
                project_db = ProjectDatabase(db_path)
                project = project_db.get_project_by_id(project_id)
                if project:
                    project_name = f"{project.code} - {project.name}"
                else:
                    project_name = project_code
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
    
    def _create_sample(self, db_path: str, args) -> bool:
        """Create a new sample."""
        from ..samples import current_project_manager
        
        sample_db = SampleDatabase(db_path)
        project_id = self.get_project_id_from_args(args, current_project_manager, db_path)
        
        if project_id is None:
            return False
        
        # Check if sample already exists in this project
        existing = sample_db.get_sample_by_code(args.code, project_id)
        if existing:
            print(f"❌ Sample with code '{args.code}' already exists in this project.")
            return False
        
        # Convert dx string to boolean
        dx_bool = args.dx.lower() in ('1', 'true', 'disease', 'dis')
        
        sample = Sample(
            code=args.code,
            age=int(args.age),
            bmi=float(args.bmi),
            dx=dx_bool,
            dx_origin=args.dx_origin,
            collection_center=args.collection_center,
            processing_time=int(args.processing_time),
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
    
    def _show_sample(self, db_path: str, code: str, project_code: str = None) -> bool:
        """Show details of a specific sample."""
        from ..samples import current_project_manager
        
        sample_db = SampleDatabase(db_path)
        
        # Create a mock args object for the helper
        class MockArgs:
            def __init__(self, project):
                self.project = project
        
        args = MockArgs(project_code)
        project_id = self.get_project_id_from_args(args, current_project_manager, db_path)
        
        if project_id is None:
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
    
    def _update_sample(self, db_path: str, args) -> bool:
        """Update an existing sample."""
        from ..samples import current_project_manager
        
        sample_db = SampleDatabase(db_path)
        project_id = self.get_project_id_from_args(args, current_project_manager, db_path)
        
        if project_id is None:
            return False
        
        sample = sample_db.get_sample_by_code(args.code, project_id)
        if not sample:
            print(f"❌ Sample with code '{args.code}' not found in current project.")
            return False
        
        # Update fields if provided
        if args.age is not None:
            sample.age = int(args.age)
        if args.bmi is not None:
            sample.bmi = float(args.bmi)
        if args.dx is not None:
            sample.dx = args.dx.lower() in ('1', 'true', 'disease', 'dis')
        if getattr(args, 'dx_origin', None) is not None:
            sample.dx_origin = args.dx_origin
        if getattr(args, 'collection_center', None) is not None:
            sample.collection_center = args.collection_center
        if getattr(args, 'processing_time', None) is not None:
            sample.processing_time = int(args.processing_time)
        
        try:
            updated_sample = sample_db.update_sample(sample)
            print(f"✅ Updated sample: {updated_sample.code}")
            return True
        except Exception as e:
            print(f"❌ Failed to update sample: {e}")
            return False
    
    def _delete_sample(self, db_path: str, code: str, project_code: str = None) -> bool:
        """Delete a sample."""
        from ..samples import current_project_manager
        
        sample_db = SampleDatabase(db_path)
        
        # Create a mock args object for the helper
        class MockArgs:
            def __init__(self, project):
                self.project = project
        
        args = MockArgs(project_code)
        project_id = self.get_project_id_from_args(args, current_project_manager, db_path)
        
        if project_id is None:
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