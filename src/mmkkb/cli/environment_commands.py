"""
Environment and database management commands for the CLI.
"""
from .base import BaseCommandHandler
from ..config import Environment, get_current_environment, get_database_path, set_environment
from ..db_utils import DatabaseUtils


class EnvironmentCommandHandler(BaseCommandHandler):
    """Handler for environment-related commands."""
    
    def add_commands(self, subparsers) -> None:
        """Add environment command parsers."""
        # Environment status
        env_parser = subparsers.add_parser("env", help="Show environment status")

        # Set environment
        setenv_parser = subparsers.add_parser("setenv", help="Set current environment")
        setenv_parser.add_argument("environment", choices=[e.value for e in Environment])
    
    def handle_command(self, args, db_path: str) -> bool:
        """Handle environment commands."""
        if args.command == "env":
            return self._show_environment()
        elif args.command == "setenv":
            return self._set_environment(args.environment)
        return False
    
    def _show_environment(self) -> bool:
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
        return True
    
    def _set_environment(self, env_name: str) -> bool:
        """Set current environment."""
        try:
            env = Environment(env_name.lower())
            set_environment(env)
            print(f"✅ Environment set to: {env.value}")
            print(f"📁 Database path: {get_database_path()}")
            return True
        except ValueError:
            valid_envs = [e.value for e in Environment]
            print(f"❌ Invalid environment. Valid options: {', '.join(valid_envs)}")
            return False


class DatabaseCommandHandler(BaseCommandHandler):
    """Handler for database management commands."""
    
    def add_commands(self, subparsers) -> None:
        """Add database management command parsers."""
        # Backup database
        backup_parser = subparsers.add_parser("backup", help="Backup database")
        backup_parser.add_argument("--env", help="Environment to backup (default: current)")
        backup_parser.add_argument("--dir", default="backups", help="Backup directory")

        # Restore database
        restore_parser = subparsers.add_parser("restore", help="Restore database from backup")
        restore_parser.add_argument("backup_path", help="Path to backup file")
        restore_parser.add_argument("--env", help="Target environment (default: current)")

        # Clean database
        clean_parser = subparsers.add_parser("clean", help="Clean database")
        clean_parser.add_argument("--env", help="Environment to clean (default: current)")

        # Clean test databases
        subparsers.add_parser("clean-tests", help="Clean all test databases")

        # Copy database
        copy_parser = subparsers.add_parser("copy", help="Copy database between environments")
        copy_parser.add_argument("source", choices=[e.value for e in Environment])
        copy_parser.add_argument("target", choices=[e.value for e in Environment])

        # Vacuum database
        vacuum_parser = subparsers.add_parser("vacuum", help="Vacuum database to optimize")
        vacuum_parser.add_argument("--env", help="Environment to vacuum (default: current)")
    
    def handle_command(self, args, db_path: str) -> bool:
        """Handle database management commands."""
        if args.command == "backup":
            return self._backup_database(args.env, args.dir)
        elif args.command == "restore":
            return self._restore_database(args.backup_path, args.env)
        elif args.command == "clean":
            return self._clean_database(args.env)
        elif args.command == "clean-tests":
            return self._clean_test_databases()
        elif args.command == "copy":
            return self._copy_database(args.source, args.target)
        elif args.command == "vacuum":
            return self._vacuum_database(args.env)
        return False
    
    def _backup_database(self, env_name: str = None, backup_dir: str = "backups") -> bool:
        """Backup database."""
        try:
            env = Environment(env_name.lower()) if env_name else None
            backup_path = DatabaseUtils.backup_database(env, backup_dir)
            print(f"✅ Database backup created: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def _restore_database(self, backup_path: str, env_name: str = None) -> bool:
        """Restore database from backup."""
        try:
            env = Environment(env_name.lower()) if env_name else None
            success = DatabaseUtils.restore_database(backup_path, env)
            if success:
                print(f"✅ Database restored successfully")
            return success
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False
    
    def _clean_database(self, env_name: str = None) -> bool:
        """Clean database."""
        try:
            env = Environment(env_name.lower()) if env_name else None
            success = DatabaseUtils.clean_database(env)
            return success
        except ValueError:
            valid_envs = [e.value for e in Environment]
            print(f"❌ Invalid environment. Valid options: {', '.join(valid_envs)}")
            return False
    
    def _clean_test_databases(self) -> bool:
        """Clean all test databases."""
        count = DatabaseUtils.clean_all_test_databases()
        print(f"✅ Cleaned {count} test database files")
        return True
    
    def _copy_database(self, source_env: str, target_env: str) -> bool:
        """Copy database between environments."""
        try:
            source = Environment(source_env.lower())
            target = Environment(target_env.lower())
            success = DatabaseUtils.copy_database(source, target)
            return success
        except ValueError:
            valid_envs = [e.value for e in Environment]
            print(f"❌ Invalid environment. Valid options: {', '.join(valid_envs)}")
            return False
    
    def _vacuum_database(self, env_name: str = None) -> bool:
        """Vacuum database to optimize."""
        env = Environment(env_name) if env_name else None
        success = DatabaseUtils.vacuum_database(env)
        return success