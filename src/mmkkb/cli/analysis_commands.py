"""
Analysis commands module for MMK Knowledge Base CLI.
Groups all analysis-related commands under a single 'analysis' command.
"""
import argparse
from typing import List, Optional
from ..analyses.roc_analysis import ROCAnalyzer, ROCAnalysis
from ..analyses.roc_normalized_analysis import ROCNormalizedAnalyzer, ROCNormalizedAnalysis
from ..analyses.base_analysis import CrossValidationConfig
from .base import BaseCommandHandler


class AnalysisCommandHandler(BaseCommandHandler):
    """Handler for analysis commands grouped under 'analysis'."""
    
    def add_commands(self, subparsers):
        """Add analysis commands to the parser."""
        analysis_parser = subparsers.add_parser(
            'analysis', 
            help='Analysis operations (ROC, ROC-normalized, etc.)'
        )
        analysis_subparsers = analysis_parser.add_subparsers(
            dest='analysis_command', 
            help='Available analysis commands'
        )
        
        # ROC Analysis commands
        self._add_roc_commands(analysis_subparsers)
        
        # ROC Normalized Analysis commands
        self._add_roc_normalized_commands(analysis_subparsers)
    
    def _add_roc_commands(self, subparsers):
        """Add ROC analysis commands."""
        # ROC run command
        roc_run_parser = subparsers.add_parser(
            'roc-run',
            help='Run ROC analysis on experiment data'
        )
        roc_run_parser.add_argument(
            'experiment_id', type=int,
            help='Experiment ID to analyze'
        )
        roc_run_parser.add_argument(
            'name', type=str,
            help='Analysis name'
        )
        roc_run_parser.add_argument(
            'prevalence', type=float,
            help='Expected disease prevalence (0-1) for PPV/NPV calculations'
        )
        roc_run_parser.add_argument(
            '--description', type=str, default='',
            help='Analysis description'
        )
        roc_run_parser.add_argument(
            '--max-combinations', type=int, default=3,
            help='Maximum number of biomarkers per combination (default: 3)'
        )
        roc_run_parser.add_argument(
            '--enable-cv', action='store_true',
            help='Enable cross-validation'
        )
        roc_run_parser.add_argument(
            '--disable-loo', action='store_true',
            help='Disable Leave-One-Out cross-validation'
        )
        roc_run_parser.add_argument(
            '--disable-bootstrap', action='store_true',
            help='Disable Bootstrap cross-validation'
        )
        roc_run_parser.add_argument(
            '--bootstrap-iterations', type=int, default=200,
            help='Number of bootstrap iterations (default: 200)'
        )
        roc_run_parser.add_argument(
            '--bootstrap-validation-size', type=float, default=0.2,
            help='Bootstrap validation set size as fraction (default: 0.2)'
        )
        
        # ROC list command
        roc_list_parser = subparsers.add_parser(
            'roc-list',
            help='List ROC analyses'
        )
        roc_list_parser.add_argument(
            '--experiment', type=int,
            help='Filter by experiment ID'
        )
        
        # ROC show command
        roc_show_parser = subparsers.add_parser(
            'roc-show',
            help='Show ROC analysis details'
        )
        roc_show_parser.add_argument(
            'analysis_id', type=int,
            help='Analysis ID to show'
        )
        
        # ROC report command
        roc_report_parser = subparsers.add_parser(
            'roc-report',
            help='Generate ROC analysis report'
        )
        roc_report_parser.add_argument(
            'analysis_id', type=int,
            help='Analysis ID for report'
        )
        roc_report_parser.add_argument(
            '--output', type=str,
            help='Output CSV file path'
        )
        roc_report_parser.add_argument(
            '--top', type=int,
            help='Show only top N models by AUC'
        )
    
    def _add_roc_normalized_commands(self, subparsers):
        """Add ROC normalized analysis commands."""
        # ROC normalized run command
        roc_norm_run_parser = subparsers.add_parser(
            'roc-norm-run',
            help='Run ROC normalized analysis on experiment data'
        )
        roc_norm_run_parser.add_argument(
            'experiment_id', type=int,
            help='Experiment ID to analyze'
        )
        roc_norm_run_parser.add_argument(
            'normalizer_biomarker_version_id', type=int,
            help='Biomarker version ID to use as normalizer'
        )
        roc_norm_run_parser.add_argument(
            'name', type=str,
            help='Analysis name'
        )
        roc_norm_run_parser.add_argument(
            'prevalence', type=float,
            help='Expected disease prevalence (0-1) for PPV/NPV calculations'
        )
        roc_norm_run_parser.add_argument(
            '--description', type=str, default='',
            help='Analysis description'
        )
        roc_norm_run_parser.add_argument(
            '--max-combinations', type=int, default=3,
            help='Maximum number of biomarkers per combination (default: 3)'
        )
        roc_norm_run_parser.add_argument(
            '--enable-cv', action='store_true',
            help='Enable cross-validation'
        )
        roc_norm_run_parser.add_argument(
            '--disable-loo', action='store_true',
            help='Disable Leave-One-Out cross-validation'
        )
        roc_norm_run_parser.add_argument(
            '--disable-bootstrap', action='store_true',
            help='Disable Bootstrap cross-validation'
        )
        roc_norm_run_parser.add_argument(
            '--bootstrap-iterations', type=int, default=200,
            help='Number of bootstrap iterations (default: 200)'
        )
        roc_norm_run_parser.add_argument(
            '--bootstrap-validation-size', type=float, default=0.2,
            help='Bootstrap validation set size as fraction (default: 0.2)'
        )
        
        # ROC normalized list command
        roc_norm_list_parser = subparsers.add_parser(
            'roc-norm-list',
            help='List ROC normalized analyses'
        )
        roc_norm_list_parser.add_argument(
            '--experiment', type=int,
            help='Filter by experiment ID'
        )
        
        # ROC normalized show command
        roc_norm_show_parser = subparsers.add_parser(
            'roc-norm-show',
            help='Show ROC normalized analysis details'
        )
        roc_norm_show_parser.add_argument(
            'analysis_id', type=int,
            help='Analysis ID to show'
        )
        
        # ROC normalized report command
        roc_norm_report_parser = subparsers.add_parser(
            'roc-norm-report',
            help='Generate ROC normalized analysis report'
        )
        roc_norm_report_parser.add_argument(
            'analysis_id', type=int,
            help='Analysis ID for report'
        )
        roc_norm_report_parser.add_argument(
            '--output', type=str,
            help='Output CSV file path'
        )
        roc_norm_report_parser.add_argument(
            '--top', type=int,
            help='Show only top N models by AUC'
        )
    
    def handle_command(self, args, db_path: str) -> bool:
        """Handle analysis commands."""
        if not hasattr(args, 'analysis_command') or not args.analysis_command:
            print("❌ No analysis command specified")
            return False
        
        # ROC Analysis commands
        if args.analysis_command.startswith('roc-') and not args.analysis_command.startswith('roc-norm-'):
            return self._handle_roc_command(args, db_path)
        
        # ROC Normalized Analysis commands
        elif args.analysis_command.startswith('roc-norm-'):
            return self._handle_roc_normalized_command(args, db_path)
        
        else:
            print(f"❌ Unknown analysis command: {args.analysis_command}")
            return False
    
    def _handle_roc_command(self, args, db_path: str) -> bool:
        """Handle ROC analysis commands."""
        analyzer = ROCAnalyzer(db_path)
        
        if args.analysis_command == 'roc-run':
            return self._handle_roc_run(args, analyzer)
        elif args.analysis_command == 'roc-list':
            return self._handle_roc_list(args, analyzer)
        elif args.analysis_command == 'roc-show':
            return self._handle_roc_show(args, analyzer)
        elif args.analysis_command == 'roc-report':
            return self._handle_roc_report(args, analyzer)
        else:
            print(f"❌ Unknown ROC command: {args.analysis_command}")
            return False
    
    def _handle_roc_normalized_command(self, args, db_path: str) -> bool:
        """Handle ROC normalized analysis commands."""
        analyzer = ROCNormalizedAnalyzer(db_path)
        
        if args.analysis_command == 'roc-norm-run':
            return self._handle_roc_norm_run(args, analyzer)
        elif args.analysis_command == 'roc-norm-list':
            return self._handle_roc_norm_list(args, analyzer)
        elif args.analysis_command == 'roc-norm-show':
            return self._handle_roc_norm_show(args, analyzer)
        elif args.analysis_command == 'roc-norm-report':
            return self._handle_roc_norm_report(args, analyzer)
        else:
            print(f"❌ Unknown ROC normalized command: {args.analysis_command}")
            return False
    
    def _handle_roc_run(self, args, analyzer: ROCAnalyzer) -> bool:
        """Handle ROC run command."""
        try:
            # Configure cross-validation if enabled
            cv_config = None
            if args.enable_cv:
                cv_config = CrossValidationConfig(
                    enable_loo=not args.disable_loo,
                    enable_bootstrap=not args.disable_bootstrap,
                    bootstrap_iterations=args.bootstrap_iterations,
                    bootstrap_validation_size=args.bootstrap_validation_size
                )
            
            analysis = ROCAnalysis(
                name=args.name,
                description=args.description,
                experiment_id=args.experiment_id,
                prevalence=args.prevalence,
                max_combination_size=args.max_combinations,
                cross_validation_config=cv_config
            )
            
            print(f"🔄 Running ROC analysis '{args.name}' on experiment {args.experiment_id}...")
            
            if cv_config:
                cv_info = []
                if cv_config.enable_loo:
                    cv_info.append("LOO")
                if cv_config.enable_bootstrap:
                    cv_info.append(f"Bootstrap({cv_config.bootstrap_iterations} iter)")
                print(f"📊 Cross-validation enabled: {', '.join(cv_info)}")
            
            results = analyzer.run_roc_analysis(analysis)
            
            print(f"✅ Analysis completed!")
            print(f"   Analysis ID: {results['analysis_id']}")
            print(f"   Total combinations tested: {results['total_combinations']}")
            print(f"   Successful models: {results['models_created']}")
            print(f"   Failed models: {len(results['failed_models'])}")
            
            if results['failed_models']:
                print("⚠️  Some models failed:")
                for failure in results['failed_models'][:5]:  # Show first 5
                    print(f"   - Combination {failure['combination']}: {failure['error']}")
                if len(results['failed_models']) > 5:
                    print(f"   ... and {len(results['failed_models']) - 5} more")
            
            return True
            
        except Exception as e:
            print(f"❌ Error running ROC analysis: {e}")
            return False
    
    def _handle_roc_list(self, args, analyzer: ROCAnalyzer) -> bool:
        """Handle ROC list command."""
        try:
            analyses = analyzer.roc_db.list_roc_analyses(args.experiment)
            
            if not analyses:
                if args.experiment:
                    print(f"No ROC analyses found for experiment {args.experiment}")
                else:
                    print("No ROC analyses found")
                return True
            
            print("\n📊 ROC Analyses:")
            print("-" * 80)
            
            for analysis in analyses:
                cv_info = ""
                if analysis.cross_validation_config:
                    cv_parts = []
                    if analysis.cross_validation_config.get('enable_loo'):
                        cv_parts.append("LOO")
                    if analysis.cross_validation_config.get('enable_bootstrap'):
                        cv_parts.append("Bootstrap")
                    if cv_parts:
                        cv_info = f" [CV: {', '.join(cv_parts)}]"
                
                print(f"ID: {analysis.id}")
                print(f"Name: {analysis.name}{cv_info}")
                print(f"Experiment: {analysis.experiment_id}")
                print(f"Prevalence: {analysis.prevalence:.3f}")
                print(f"Max combinations: {analysis.max_combination_size}")
                print(f"Created: {analysis.created_at}")
                print("-" * 40)
            
            return True
            
        except Exception as e:
            print(f"❌ Error listing ROC analyses: {e}")
            return False
    
    def _handle_roc_show(self, args, analyzer: ROCAnalyzer) -> bool:
        """Handle ROC show command."""
        try:
            analysis = analyzer.roc_db.get_roc_analysis_by_id(args.analysis_id)
            if not analysis:
                print(f"❌ ROC analysis {args.analysis_id} not found")
                return False
            
            models = analyzer.roc_db.get_roc_models_by_analysis(args.analysis_id)
            
            print(f"\n📊 ROC Analysis Details (ID: {analysis.id})")
            print("=" * 60)
            print(f"Name: {analysis.name}")
            print(f"Description: {analysis.description}")
            print(f"Experiment ID: {analysis.experiment_id}")
            print(f"Prevalence: {analysis.prevalence:.3f}")
            print(f"Max combination size: {analysis.max_combination_size}")
            print(f"Created: {analysis.created_at}")
            
            if analysis.cross_validation_config:
                print("\n🔄 Cross-Validation Configuration:")
                cv_config = analysis.cross_validation_config
                if cv_config.get('enable_loo'):
                    print("  ✓ Leave-One-Out enabled")
                if cv_config.get('enable_bootstrap'):
                    print(f"  ✓ Bootstrap enabled ({cv_config.get('bootstrap_iterations', 200)} iterations)")
            
            print(f"\n📈 Models: {len(models)}")
            if models:
                print("\nTop 10 models by AUC:")
                for i, model in enumerate(models[:10], 1):
                    cv_summary = ""
                    if model.cross_validation_results:
                        cv_parts = []
                        if model.cross_validation_results.get('loo_auc_mean'):
                            cv_parts.append(f"LOO: {model.cross_validation_results['loo_auc_mean']:.3f}±{model.cross_validation_results.get('loo_auc_std', 0):.3f}")
                        if model.cross_validation_results.get('bootstrap_auc_mean'):
                            cv_parts.append(f"Bootstrap: {model.cross_validation_results['bootstrap_auc_mean']:.3f}±{model.cross_validation_results.get('bootstrap_auc_std', 0):.3f}")
                        if cv_parts:
                            cv_summary = f" [CV: {', '.join(cv_parts)}]"
                    
                    print(f"  {i:2d}. Model {model.id}: AUC = {model.auc:.3f}{cv_summary}")
                    print(f"      Biomarkers: {model.biomarker_combination}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error showing ROC analysis: {e}")
            return False
    
    def _handle_roc_report(self, args, analyzer: ROCAnalyzer) -> bool:
        """Handle ROC report command."""
        try:
            df = analyzer.generate_analysis_report(args.analysis_id)
            
            if args.top:
                df = df.head(args.top)
            
            if args.output:
                df.to_csv(args.output, index=False)
                print(f"✅ Report saved to {args.output}")
            else:
                print("\n📊 ROC Analysis Report:")
                print("=" * 100)
                print(df.to_string(index=False))
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating ROC report: {e}")
            return False
    
    # Complete the ROC normalized command handlers
    def _handle_roc_norm_run(self, args, analyzer: ROCNormalizedAnalyzer) -> bool:
        """Handle ROC normalized run command."""
        try:
            # Configure cross-validation if enabled
            cv_config = None
            if args.enable_cv:
                cv_config = CrossValidationConfig(
                    enable_loo=not args.disable_loo,
                    enable_bootstrap=not args.disable_bootstrap,
                    bootstrap_iterations=args.bootstrap_iterations,
                    bootstrap_validation_size=args.bootstrap_validation_size
                )
            
            analysis = ROCNormalizedAnalysis(
                name=args.name,
                description=args.description,
                experiment_id=args.experiment_id,
                normalizer_biomarker_version_id=args.normalizer_biomarker_version_id,
                prevalence=args.prevalence,
                max_combination_size=args.max_combinations,
                cross_validation_config=cv_config
            )
            
            print(f"🔄 Running ROC normalized analysis '{args.name}' on experiment {args.experiment_id}...")
            print(f"📊 Using normalizer biomarker version ID: {args.normalizer_biomarker_version_id}")
            
            if cv_config:
                cv_info = []
                if cv_config.enable_loo:
                    cv_info.append("LOO")
                if cv_config.enable_bootstrap:
                    cv_info.append(f"Bootstrap({cv_config.bootstrap_iterations} iter)")
                print(f"📊 Cross-validation enabled: {', '.join(cv_info)}")
            
            results = analyzer.run_roc_normalized_analysis(analysis)
            
            print(f"✅ Analysis completed!")
            print(f"   Analysis ID: {results['analysis_id']}")
            print(f"   Normalizer biomarker: {results['normalizer_biomarker']}")
            print(f"   Total combinations tested: {results['total_combinations']}")
            print(f"   Successful models: {results['models_created']}")
            print(f"   Failed models: {len(results['failed_models'])}")
            
            if results['failed_models']:
                print("⚠️  Some models failed:")
                for failure in results['failed_models'][:5]:  # Show first 5
                    print(f"   - Combination {failure['combination']}: {failure['error']}")
                if len(results['failed_models']) > 5:
                    print(f"   ... and {len(results['failed_models']) - 5} more")
            
            return True
            
        except Exception as e:
            print(f"❌ Error running ROC normalized analysis: {e}")
            return False
    
    def _handle_roc_norm_list(self, args, analyzer: ROCNormalizedAnalyzer) -> bool:
        """Handle ROC normalized list command."""
        try:
            analyses = analyzer.roc_norm_db.list_roc_normalized_analyses(args.experiment)
            
            if not analyses:
                if args.experiment:
                    print(f"No ROC normalized analyses found for experiment {args.experiment}")
                else:
                    print("No ROC normalized analyses found")
                return True
            
            print("\n📊 ROC Normalized Analyses:")
            print("-" * 80)
            
            for analysis in analyses:
                cv_info = ""
                if analysis.cross_validation_config:
                    cv_parts = []
                    if analysis.cross_validation_config.get('enable_loo'):
                        cv_parts.append("LOO")
                    if analysis.cross_validation_config.get('enable_bootstrap'):
                        cv_parts.append("Bootstrap")
                    if cv_parts:
                        cv_info = f" [CV: {', '.join(cv_parts)}]"
                
                print(f"ID: {analysis.id}")
                print(f"Name: {analysis.name}{cv_info}")
                print(f"Experiment: {analysis.experiment_id}")
                print(f"Normalizer: {analysis.normalizer_biomarker_version_id}")
                print(f"Prevalence: {analysis.prevalence:.3f}")
                print(f"Max combinations: {analysis.max_combination_size}")
                print(f"Created: {analysis.created_at}")
                print("-" * 40)
            
            return True
            
        except Exception as e:
            print(f"❌ Error listing ROC normalized analyses: {e}")
            return False
    
    def _handle_roc_norm_show(self, args, analyzer: ROCNormalizedAnalyzer) -> bool:
        """Handle ROC normalized show command."""
        try:
            analysis = analyzer.roc_norm_db.get_roc_normalized_analysis_by_id(args.analysis_id)
            if not analysis:
                print(f"❌ ROC normalized analysis {args.analysis_id} not found")
                return False
            
            models = analyzer.roc_norm_db.get_roc_normalized_models_by_analysis(args.analysis_id)
            
            print(f"\n📊 ROC Normalized Analysis Details (ID: {analysis.id})")
            print("=" * 60)
            print(f"Name: {analysis.name}")
            print(f"Description: {analysis.description}")
            print(f"Experiment ID: {analysis.experiment_id}")
            print(f"Normalizer biomarker version ID: {analysis.normalizer_biomarker_version_id}")
            print(f"Prevalence: {analysis.prevalence:.3f}")
            print(f"Max combination size: {analysis.max_combination_size}")
            print(f"Created: {analysis.created_at}")
            
            if analysis.cross_validation_config:
                print("\n🔄 Cross-Validation Configuration:")
                cv_config = analysis.cross_validation_config
                if cv_config.get('enable_loo'):
                    print("  ✓ Leave-One-Out enabled")
                if cv_config.get('enable_bootstrap'):
                    print(f"  ✓ Bootstrap enabled ({cv_config.get('bootstrap_iterations', 200)} iterations)")
            
            print(f"\n📈 Models: {len(models)}")
            if models:
                print("\nTop 10 models by AUC:")
                for i, model in enumerate(models[:10], 1):
                    cv_summary = ""
                    if model.cross_validation_results:
                        cv_parts = []
                        if model.cross_validation_results.get('loo_auc_mean'):
                            cv_parts.append(f"LOO: {model.cross_validation_results['loo_auc_mean']:.3f}±{model.cross_validation_results.get('loo_auc_std', 0):.3f}")
                        if model.cross_validation_results.get('bootstrap_auc_mean'):
                            cv_parts.append(f"Bootstrap: {model.cross_validation_results['bootstrap_auc_mean']:.3f}±{model.cross_validation_results.get('bootstrap_auc_std', 0):.3f}")
                        if cv_parts:
                            cv_summary = f" [CV: {', '.join(cv_parts)}]"
                    
                    print(f"  {i:2d}. Model {model.id}: AUC = {model.auc:.3f}{cv_summary}")
                    print(f"      Biomarkers: {model.biomarker_combination}")
                    print(f"      Normalizer: {model.normalizer_biomarker_version_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error showing ROC normalized analysis: {e}")
            return False
    
    def _handle_roc_norm_report(self, args, analyzer: ROCNormalizedAnalyzer) -> bool:
        """Handle ROC normalized report command."""
        try:
            df = analyzer.generate_analysis_report(args.analysis_id)
            
            if args.top:
                df = df.head(args.top)
            
            if args.output:
                df.to_csv(args.output, index=False)
                print(f"✅ Report saved to {args.output}")
            else:
                print("\n📊 ROC Normalized Analysis Report:")
                print("=" * 100)
                print(df.to_string(index=False))
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating ROC normalized report: {e}")
            return False