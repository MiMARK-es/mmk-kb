"""
Experiment management commands for the CLI.
"""
from .base import BaseCommandHandler
from ..experiments import ExperimentDatabase
from ..csv_processor import CSVProcessor
from ..projects import ProjectDatabase


class ExperimentCommandHandler(BaseCommandHandler):
    """Handler for experiment-related commands."""
    
    def add_commands(self, subparsers) -> None:
        """Add experiment command parsers."""
        # List experiments
        experiments_list_parser = subparsers.add_parser("experiments", help="List experiments in current/specified project")
        experiments_list_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")

        # Upload CSV experiment
        experiment_upload_parser = subparsers.add_parser("experiment-upload", help="Upload experiment from CSV file")
        experiment_upload_parser.add_argument("csv_path", help="Path to CSV file")
        experiment_upload_parser.add_argument("name", help="Experiment name")
        experiment_upload_parser.add_argument("description", help="Experiment description")
        experiment_upload_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")
        experiment_upload_parser.add_argument("--version", default="v1.0", help="Biomarker version (default: v1.0)")

        # Preview CSV
        csv_preview_parser = subparsers.add_parser("csv-preview", help="Preview CSV file structure")
        csv_preview_parser.add_argument("csv_path", help="Path to CSV file")
        csv_preview_parser.add_argument("--rows", type=int, default=5, help="Number of rows to preview (default: 5)")

        # Show experiment details
        experiment_show_parser = subparsers.add_parser("experiment-show", help="Show experiment details")
        experiment_show_parser.add_argument("experiment_id", type=int, help="Experiment ID")

        # List biomarkers
        biomarkers_list_parser = subparsers.add_parser("biomarkers", help="List all biomarkers")

        # List biomarker versions
        biomarker_versions_parser = subparsers.add_parser("biomarker-versions", help="List biomarker versions")
        biomarker_versions_parser.add_argument("--biomarker", help="Biomarker name to filter versions")

        # Show biomarker analysis data
        biomarker_analysis_parser = subparsers.add_parser("biomarker-analysis", help="Show biomarker data for analysis")
        biomarker_analysis_parser.add_argument("biomarker_id", type=int, help="Biomarker ID")

        # Show measurement summary
        measurements_summary_parser = subparsers.add_parser("measurements-summary", help="Show measurement summary")
        measurements_summary_parser.add_argument("--project", help="Project code (optional, uses current if not specified)")
    
    def handle_command(self, args, db_path: str) -> bool:
        """Handle experiment commands."""
        if args.command == "experiments":
            return self._list_experiments(db_path, args.project)
        elif args.command == "experiment-upload":
            return self._upload_experiment(db_path, args)
        elif args.command == "csv-preview":
            return self._preview_csv(db_path, args.csv_path, args.rows)
        elif args.command == "experiment-show":
            return self._show_experiment(db_path, args.experiment_id)
        elif args.command == "biomarkers":
            return self._list_biomarkers(db_path)
        elif args.command == "biomarker-versions":
            return self._list_biomarker_versions(db_path, getattr(args, 'biomarker', None))
        elif args.command == "biomarker-analysis":
            return self._show_biomarker_analysis(db_path, args.biomarker_id)
        elif args.command == "measurements-summary":
            return self._show_measurements_summary(db_path, args.project)
        return False
    
    def _list_experiments(self, db_path: str, project_code: str = None) -> bool:
        """List experiments for current or specified project."""
        from ..samples import current_project_manager
        
        experiment_db = ExperimentDatabase(db_path)
        
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
        
        experiments = experiment_db.list_experiments(project_id)
        if not experiments:
            print(f"No experiments found in project {project_name}.")
            return True
        
        print(f"\n🔬 Found {len(experiments)} experiments in project {project_name}:\n")
        print(f"{'ID':<4} {'Name':<25} {'Upload Date':<19} {'CSV File':<20} {'Measurements':<12}")
        print("-" * 85)
        
        for experiment in experiments:
            upload_date_str = experiment.upload_date.strftime("%Y-%m-%d %H:%M") if experiment.upload_date else "Unknown"
            csv_filename = experiment.csv_filename or "N/A"
            
            # Get measurement count
            measurements = experiment_db.get_measurements_by_experiment(experiment.id)
            measurement_count = len(measurements)
            
            print(
                f"{experiment.id:<4} {experiment.name[:24]:<25} {upload_date_str:<19} "
                f"{csv_filename[:19]:<20} {measurement_count:<12}"
            )
        return True
    
    def _upload_experiment(self, db_path: str, args) -> bool:
        """Upload experiment from CSV file."""
        from ..samples import current_project_manager
        
        csv_processor = CSVProcessor(db_path)
        project_id = self.get_project_id_from_args(args, current_project_manager, db_path)
        
        if project_id is None:
            return False
        
        # Process CSV upload
        success, message, experiment = csv_processor.process_csv_upload(
            csv_path=args.csv_path,
            experiment_name=args.name,
            experiment_description=args.description,
            project_id=project_id,
            biomarker_version=args.version
        )
        
        print(message)
        return success
    
    def _preview_csv(self, db_path: str, csv_path: str, num_rows: int) -> bool:
        """Preview CSV file structure."""
        csv_processor = CSVProcessor(db_path)
        
        success, message, preview_data = csv_processor.preview_csv(csv_path, num_rows)
        
        if not success:
            print(f"❌ {message}")
            return False
        
        print(f"📄 CSV Preview: {csv_path}\n")
        print(f"📊 Structure:")
        print(f"   - Total rows: {preview_data['total_rows']}")
        print(f"   - Total columns: {preview_data['total_columns']}")
        print(f"   - Sample column exists: {'✅' if preview_data['sample_column_exists'] else '❌'}")
        print(f"   - Biomarker columns: {len(preview_data['biomarker_columns'])}")
        
        if preview_data['biomarker_columns']:
            print(f"\n🧬 Biomarker columns:")
            for i, col in enumerate(preview_data['biomarker_columns'][:10], 1):
                print(f"   {i}. {col}")
            if len(preview_data['biomarker_columns']) > 10:
                print(f"   ... and {len(preview_data['biomarker_columns']) - 10} more")
        
        print(f"\n📋 First {num_rows} rows:")
        for i, row in enumerate(preview_data['preview_rows'], 1):
            print(f"   Row {i}: {dict(list(row.items())[:3])}{'...' if len(row) > 3 else ''}")
        
        if any(count > 0 for count in preview_data['missing_values'].values()):
            print(f"\n⚠️  Missing values:")
            for col, count in preview_data['missing_values'].items():
                if count > 0:
                    print(f"   - {col}: {count} missing")
        
        return True
    
    def _show_experiment(self, db_path: str, experiment_id: int) -> bool:
        """Show details of a specific experiment."""
        experiment_db = ExperimentDatabase(db_path)
        
        experiment = experiment_db.get_experiment_by_id(experiment_id)
        if not experiment:
            print(f"❌ Experiment with ID {experiment_id} not found.")
            return False
        
        # Get project details
        project_db = ProjectDatabase(db_path)
        project = project_db.get_project_by_id(experiment.project_id)
        project_name = f"{project.code} - {project.name}" if project else f"Project ID {experiment.project_id}"
        
        # Get measurement statistics
        measurements = experiment_db.get_measurements_by_experiment(experiment_id)
        unique_samples = len(set(m.sample_id for m in measurements))
        unique_biomarkers = len(set(m.biomarker_id for m in measurements))
        
        print(f"\n🔬 Experiment Details:")
        print(f"ID: {experiment.id}")
        print(f"Name: {experiment.name}")
        print(f"Description: {experiment.description}")
        print(f"Project: {project_name}")
        print(f"Upload Date: {experiment.upload_date}")
        print(f"CSV File: {experiment.csv_filename or 'N/A'}")
        print(f"Created: {experiment.created_at}")
        print(f"Updated: {experiment.updated_at}")
        print(f"\n📊 Measurements:")
        print(f"   - Total measurements: {len(measurements)}")
        print(f"   - Unique samples: {unique_samples}")
        print(f"   - Unique biomarkers: {unique_biomarkers}")
        
        return True
    
    def _list_biomarkers(self, db_path: str) -> bool:
        """List all biomarkers."""
        experiment_db = ExperimentDatabase(db_path)
        biomarkers = experiment_db.list_biomarkers()
        
        if not biomarkers:
            print("No biomarkers found.")
            return True
        
        print(f"\n🧬 Found {len(biomarkers)} biomarkers:\n")
        print(f"{'ID':<4} {'Name':<30} {'Category':<15} {'Created':<19}")
        print("-" * 70)
        
        for biomarker in biomarkers:
            created_str = biomarker.created_at.strftime("%Y-%m-%d %H:%M") if biomarker.created_at else "Unknown"
            category = biomarker.category or "N/A"
            print(
                f"{biomarker.id:<4} {biomarker.name[:29]:<30} {category[:14]:<15} {created_str:<19}"
            )
        return True
    
    def _list_biomarker_versions(self, db_path: str, biomarker_name: str = None) -> bool:
        """List biomarker versions."""
        experiment_db = ExperimentDatabase(db_path)
        
        biomarker_id = None
        if (biomarker_name):
            biomarker = experiment_db.get_biomarker_by_name(biomarker_name)
            if not biomarker:
                print(f"❌ Biomarker '{biomarker_name}' not found.")
                return False
            biomarker_id = biomarker.id
            print(f"\n🔬 Versions for biomarker '{biomarker_name}':")
        else:
            print(f"\n🔬 All biomarker versions:")
        
        versions = experiment_db.list_biomarker_versions(biomarker_id)
        
        if not versions:
            print("No biomarker versions found.")
            return True
        
        print(f"\n{'ID':<4} {'Biomarker':<25} {'Version':<15} {'Created':<19}")
        print("-" * 65)
        
        for version in versions:
            biomarker = experiment_db.get_biomarker_by_id(version.biomarker_id)
            biomarker_name_display = biomarker.name if biomarker else f"ID:{version.biomarker_id}"
            created_str = version.created_at.strftime("%Y-%m-%d %H:%M") if version.created_at else "Unknown"
            
            print(
                f"{version.id:<4} {biomarker_name_display[:24]:<25} {version.version:<15} {created_str:<19}"
            )
        return True
    
    def _show_biomarker_analysis(self, db_path: str, biomarker_id: int) -> bool:
        """Show biomarker data for analysis."""
        experiment_db = ExperimentDatabase(db_path)
        
        analysis_data = experiment_db.get_biomarker_data_for_analysis(biomarker_id)
        if not analysis_data:
            print(f"❌ Biomarker with ID {biomarker_id} not found or has no measurements.")
            return False
        
        biomarker = analysis_data["biomarker"]
        measurements = analysis_data["measurements"]
        
        print(f"\n🧬 Biomarker Analysis: {biomarker.name}")
        print(f"Description: {biomarker.description or 'N/A'}")
        print(f"Category: {biomarker.category or 'N/A'}")
        print(f"\n📊 Summary:")
        print(f"   - Total measurements: {analysis_data['total_measurements']}")
        print(f"   - Unique experiments: {analysis_data['unique_experiments']}")
        print(f"   - Unique samples: {analysis_data['unique_samples']}")
        print(f"   - Versions used: {', '.join(analysis_data['versions_used'])}")
        
        if measurements:
            print(f"\n📋 Recent measurements (showing first 10):")
            print(f"{'Experiment':<20} {'Sample':<10} {'Value':<12} {'Version':<12}")
            print("-" * 56)
            
            for measurement in measurements[:10]:
                print(
                    f"{measurement['experiment_name'][:19]:<20} "
                    f"{measurement['sample_code']:<10} "
                    f"{measurement['value']:<12.2f} "
                    f"{measurement['version']:<12}"
                )
            
            if len(measurements) > 10:
                print(f"   ... and {len(measurements) - 10} more measurements")
        
        return True
    
    def _show_measurements_summary(self, db_path: str, project_code: str = None) -> bool:
        """Show measurement summary statistics."""
        from ..samples import current_project_manager
        
        experiment_db = ExperimentDatabase(db_path)
        
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
            # Show global summary
            project_id = None
            project_name = "All projects"
        
        summary = experiment_db.get_measurement_summary(project_id)
        
        print(f"\n📊 Measurement Summary for {project_name}:")
        print(f"   - Experiments: {summary['experiment_count']}")
        print(f"   - Samples with measurements: {summary['sample_count']}")
        print(f"   - Unique biomarkers: {summary['biomarker_count']}")
        print(f"   - Biomarker versions: {summary['biomarker_version_count']}")
        print(f"   - Total measurements: {summary['measurement_count']}")
        
        return True