"""
ROC Normalized analysis commands for the CLI.
"""
import os
from .base import BaseCommandHandler
from ..analyses.roc_normalized_analysis import ROCNormalizedAnalysisDatabase, ROCNormalizedAnalyzer, ROCNormalizedAnalysis
from ..experiments import ExperimentDatabase
from ..projects import ProjectDatabase

class ROCNormalizedAnalysisCommandHandler(BaseCommandHandler):
    """Handler for ROC Normalized analysis commands."""
    
    def add_commands(self, subparsers) -> None:
        """Add ROC Normalized analysis command parsers."""
        # Run ROC Normalized analysis
        roc_norm_run_parser = subparsers.add_parser(
            "roc-norm-run", 
            help="Run ROC Normalized analysis on experiment data"
        )
        roc_norm_run_parser.add_argument("experiment_id", type=int, help="Experiment ID to analyze")
        roc_norm_run_parser.add_argument("normalizer_biomarker_version_id", type=int, help="Biomarker version ID to use as normalizer")
        roc_norm_run_parser.add_argument("name", help="Analysis name")
        roc_norm_run_parser.add_argument("prevalence", type=float, help="Expected prevalence (0.0-1.0)")
        roc_norm_run_parser.add_argument(
            "--max-combinations", 
            type=int, 
            default=3, 
            help="Maximum biomarker combinations (default: 3)"
        )
        roc_norm_run_parser.add_argument("--description", default="", help="Analysis description")
        
        # List ROC Normalized analyses
        roc_norm_list_parser = subparsers.add_parser(
            "roc-norm-list", 
            help="List ROC Normalized analyses"
        )
        roc_norm_list_parser.add_argument("--experiment", type=int, help="Filter by experiment ID")
        
        # Show ROC Normalized analysis details
        roc_norm_show_parser = subparsers.add_parser(
            "roc-norm-show", 
            help="Show ROC Normalized analysis details"
        )
        roc_norm_show_parser.add_argument("analysis_id", type=int, help="Analysis ID")
        
        # Generate ROC Normalized analysis report
        roc_norm_report_parser = subparsers.add_parser(
            "roc-norm-report", 
            help="Generate ROC Normalized analysis report"
        )
        roc_norm_report_parser.add_argument("analysis_id", type=int, help="Analysis ID")
        roc_norm_report_parser.add_argument(
            "--output", 
            help="Output CSV file path (optional, prints to console if not specified)"
        )
        roc_norm_report_parser.add_argument(
            "--top", 
            type=int, 
            help="Show only top N models by AUC"
        )
        
        # Show ROC Normalized model details
        roc_norm_model_parser = subparsers.add_parser(
            "roc-norm-model", 
            help="Show ROC Normalized model details"
        )
        roc_norm_model_parser.add_argument("model_id", type=int, help="Model ID")
        
        # Export ROC Normalized curve data
        roc_norm_curve_parser = subparsers.add_parser(
            "roc-norm-curve", 
            help="Export ROC Normalized curve data for a model"
        )
        roc_norm_curve_parser.add_argument("model_id", type=int, help="Model ID")
        roc_norm_curve_parser.add_argument(
            "--output", 
            help="Output CSV file path (optional, prints to console if not specified)"
        )
    
    def handle_command(self, args, db_path: str) -> bool:
        """Handle ROC Normalized analysis commands."""
        if args.command == "roc-norm-run":
            return self._run_roc_normalized_analysis(db_path, args)
        elif args.command == "roc-norm-list":
            return self._list_roc_normalized_analyses(db_path, args.experiment)
        elif args.command == "roc-norm-show":
            return self._show_roc_normalized_analysis(db_path, args.analysis_id)
        elif args.command == "roc-norm-report":
            return self._generate_roc_normalized_report(db_path, args)
        elif args.command == "roc-norm-model":
            return self._show_roc_normalized_model(db_path, args.model_id)
        elif args.command == "roc-norm-curve":
            return self._export_roc_normalized_curve(db_path, args)
        return False
    
    def _run_roc_normalized_analysis(self, db_path: str, args) -> bool:
        """Run ROC Normalized analysis on experiment data."""
        # Validate inputs
        if not (0 < args.prevalence < 1):
            print("❌ Prevalence must be between 0 and 1 (exclusive).")
            return False
        
        if args.max_combinations < 1:
            print("❌ Max combinations must be at least 1.")
            return False
        
        # Check if experiment exists
        exp_db = ExperimentDatabase(db_path)
        experiment = exp_db.get_experiment_by_id(args.experiment_id)
        if not experiment:
            print(f"❌ Experiment with ID {args.experiment_id} not found.")
            return False
        
        # Check if normalizer biomarker version exists
        normalizer_bv = exp_db.get_biomarker_version_by_id(args.normalizer_biomarker_version_id)
        if not normalizer_bv:
            print(f"❌ Normalizer biomarker version with ID {args.normalizer_biomarker_version_id} not found.")
            return False
        
        # Get normalizer biomarker name
        normalizer_biomarker = exp_db.get_biomarker_by_id(normalizer_bv.biomarker_id)
        normalizer_name = f"{normalizer_biomarker.name}_{normalizer_bv.version}" if normalizer_biomarker else f"BV{args.normalizer_biomarker_version_id}"
        
        print(f"🔬 Starting ROC Normalized analysis for experiment: {experiment.name}")
        print(f"   - Normalizer: {normalizer_name}")
        print(f"   - Prevalence: {args.prevalence}")
        print(f"   - Max combinations: {args.max_combinations}")
        
        try:
            # Create analysis configuration
            analysis = ROCNormalizedAnalysis(
                name=args.name,
                description=args.description,
                experiment_id=args.experiment_id,
                normalizer_biomarker_version_id=args.normalizer_biomarker_version_id,
                prevalence=args.prevalence,
                max_combination_size=args.max_combinations
            )
            
            # Run analysis
            analyzer = ROCNormalizedAnalyzer(db_path)
            results = analyzer.run_roc_normalized_analysis(analysis)
            
            print(f"\n✅ ROC Normalized analysis completed!")
            print(f"   - Analysis ID: {results['analysis_id']}")
            print(f"   - Normalizer biomarker ID: {results['normalizer_biomarker']}")
            print(f"   - Total combinations tested: {results['total_combinations']}")
            print(f"   - Successful models: {results['models_created']}")
            print(f"   - Failed models: {len(results['failed_models'])}")
            
            if results['failed_models']:
                print(f"\n⚠️  Failed combinations:")
                for i, failed in enumerate(results['failed_models'][:5], 1):
                    print(f"   {i}. {failed['combination']}: {failed['error']}")
                if len(results['failed_models']) > 5:
                    print(f"   ... and {len(results['failed_models']) - 5} more")
            
            if results['successful_models']:
                # Show top 5 models by AUC
                top_models = sorted(results['successful_models'], 
                                  key=lambda x: x['auc'], reverse=True)[:5]
                print(f"\n🏆 Top 5 models by AUC:")
                for i, model in enumerate(top_models, 1):
                    biomarker_info = self._get_biomarker_names(db_path, model['biomarker_combination'], model['normalizer_biomarker_version_id'])
                    print(f"   {i}. AUC: {model['auc']:.3f} - {biomarker_info}")
            
            print(f"\nUse 'mmk-kb roc-norm-report {results['analysis_id']}' to see detailed results.")
            
        except Exception as e:
            print(f"❌ Error running ROC Normalized analysis: {str(e)}")
            return False
        
        return True
    
    def _list_roc_normalized_analyses(self, db_path: str, experiment_id: int = None) -> bool:
        """List ROC Normalized analyses."""
        roc_norm_db = ROCNormalizedAnalysisDatabase(db_path)
        analyses = roc_norm_db.list_roc_normalized_analyses(experiment_id)
        
        if not analyses:
            filter_text = f" for experiment {experiment_id}" if experiment_id else ""
            print(f"No ROC Normalized analyses found{filter_text}.")
            return True
        
        print(f"\n📊 Found {len(analyses)} ROC Normalized analyses:\n")
        print(f"{'ID':<4} {'Name':<25} {'Experiment':<12} {'Normalizer':<15} {'Prevalence':<10} {'Max Comb':<8} {'Created':<19}")
        print("-" * 100)
        
        for analysis in analyses:
            # Get normalizer name
            exp_db = ExperimentDatabase(db_path)
            normalizer_bv = exp_db.get_biomarker_version_by_id(analysis.normalizer_biomarker_version_id)
            normalizer_name = "Unknown"
            if normalizer_bv:
                normalizer_biomarker = exp_db.get_biomarker_by_id(normalizer_bv.biomarker_id)
                if normalizer_biomarker:
                    normalizer_name = f"{normalizer_biomarker.name[:10]}_{normalizer_bv.version}"
            
            created_str = analysis.created_at.strftime("%Y-%m-%d %H:%M") if analysis.created_at else "Unknown"
            print(
                f"{analysis.id:<4} {analysis.name[:24]:<25} {analysis.experiment_id:<12} {normalizer_name[:14]:<15} "
                f"{analysis.prevalence:<10.3f} {analysis.max_combination_size:<8} {created_str:<19}"
            )
        
        return True
    
    def _show_roc_normalized_analysis(self, db_path: str, analysis_id: int) -> bool:
        """Show details of a specific ROC Normalized analysis."""
        roc_norm_db = ROCNormalizedAnalysisDatabase(db_path)
        analysis = roc_norm_db.get_roc_normalized_analysis_by_id(analysis_id)
        
        if not analysis:
            print(f"❌ ROC Normalized analysis with ID {analysis_id} not found.")
            return False
        
        # Get experiment details
        exp_db = ExperimentDatabase(db_path)
        experiment = exp_db.get_experiment_by_id(analysis.experiment_id)
        
        # Get normalizer details
        normalizer_bv = exp_db.get_biomarker_version_by_id(analysis.normalizer_biomarker_version_id)
        normalizer_name = "Unknown"
        if normalizer_bv:
            normalizer_biomarker = exp_db.get_biomarker_by_id(normalizer_bv.biomarker_id)
            if normalizer_biomarker:
                normalizer_name = f"{normalizer_biomarker.name}_{normalizer_bv.version}"
        
        # Get models count
        models = roc_norm_db.get_roc_normalized_models_by_analysis(analysis_id)
        
        print(f"\n📊 ROC Normalized Analysis Details:")
        print(f"ID: {analysis.id}")
        print(f"Name: {analysis.name}")
        print(f"Description: {analysis.description or 'N/A'}")
        print(f"Experiment: {experiment.name if experiment else f'ID {analysis.experiment_id}'}")
        print(f"Normalizer: {normalizer_name}")
        print(f"Prevalence: {analysis.prevalence}")
        print(f"Max Combination Size: {analysis.max_combination_size}")
        print(f"Created: {analysis.created_at}")
        print(f"Total Models: {len(models)}")
        
        if models:
            # Show top 10 models by AUC
            top_models = sorted(models, key=lambda x: x.auc, reverse=True)[:10]
            print(f"\n🏆 Top {len(top_models)} models by AUC:")
            print(f"{'Model ID':<8} {'AUC':<6} {'Normalized Biomarkers'}")
            print("-" * 60)
            
            for model in top_models:
                biomarker_info = self._get_biomarker_names(db_path, model.biomarker_combination, model.normalizer_biomarker_version_id)
                print(f"{model.id:<8} {model.auc:<6.3f} {biomarker_info}")
        
        return True
    
    def _generate_roc_normalized_report(self, db_path: str, args) -> bool:
        """Generate comprehensive ROC Normalized analysis report."""
        try:
            analyzer = ROCNormalizedAnalyzer(db_path)
            report_df = analyzer.generate_analysis_report(args.analysis_id)
            
            if report_df.empty:
                print(f"❌ No models found for analysis {args.analysis_id}.")
                return False
            
            # Apply top N filter if specified
            if args.top:
                report_df = report_df.nlargest(args.top, 'AUC')
            
            if args.output:
                # Save to file
                report_df.to_csv(args.output, index=False)
                print(f"✅ Report saved to: {args.output}")
                print(f"📊 Generated report with {len(report_df)} models.")
            else:
                # Print to console
                print(f"\n📊 ROC Normalized Analysis Report (Analysis ID: {args.analysis_id})")
                print(f"{'='*80}")
                
                # Show summary
                print(f"Total Models: {len(report_df)}")
                print(f"Best AUC: {report_df['AUC'].max():.3f}")
                print(f"Average AUC: {report_df['AUC'].mean():.3f}")
                if 'Normalizer' in report_df.columns:
                    print(f"Normalizer: {report_df['Normalizer'].iloc[0] if len(report_df) > 0 else 'N/A'}")
                
                # Show top models with key metrics
                display_cols = ['Model_ID', 'AUC', 'Biomarker_1', 'se_97_Sensitivity', 'se_97_Specificity', 'se_97_PPV']
                if 'Biomarker_2' in report_df.columns:
                    display_cols.insert(3, 'Biomarker_2')
                
                available_cols = [col for col in display_cols if col in report_df.columns]
                
                print(f"\n🏆 Top models:")
                print(report_df[available_cols].head(10).to_string(index=False, float_format='%.3f'))
                
                if len(report_df) > 10:
                    print(f"\n... and {len(report_df) - 10} more models")
                    print(f"\nUse --output to save full report to CSV file.")
            
        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")
            return False
        
        return True
    
    def _show_roc_normalized_model(self, db_path: str, model_id: int) -> bool:
        """Show details of a specific ROC Normalized model."""
        roc_norm_db = ROCNormalizedAnalysisDatabase(db_path)
        
        # Get model details (we need to find it by searching through analyses)
        analyses = roc_norm_db.list_roc_normalized_analyses()
        model = None
        analysis = None
        
        for a in analyses:
            models = roc_norm_db.get_roc_normalized_models_by_analysis(a.id)
            for m in models:
                if m.id == model_id:
                    model = m
                    analysis = a
                    break
            if model:
                break
        
        if not model:
            print(f"❌ ROC Normalized model with ID {model_id} not found.")
            return False
        
        # Get biomarker names
        biomarker_names = self._get_biomarker_names(db_path, model.biomarker_combination, model.normalizer_biomarker_version_id)
        
        # Get metrics
        metrics = roc_norm_db.get_roc_normalized_metrics_by_model(model_id)
        metrics_dict = {m.threshold_type: m for m in metrics}
        
        # Get ROC curve points count
        roc_points = roc_norm_db.get_roc_normalized_curve_points_by_model(model_id)
        
        print(f"\n🎯 ROC Normalized Model Details:")
        print(f"Model ID: {model.id}")
        print(f"Analysis: {analysis.name} (ID: {analysis.id})")
        print(f"Normalized Biomarkers: {biomarker_names}")
        print(f"AUC: {model.auc:.4f}")
        print(f"ROC Curve Points: {len(roc_points)}")
        print(f"Created: {model.created_at}")
        
        print(f"\n📈 Model Coefficients:")
        coeffs = model.coefficients
        print(f"Intercept: {coeffs['intercept']:.4f}")
        for i, (bv_id, coef) in enumerate(zip(coeffs['biomarker_version_ids'], coeffs['coef'])):
            bv_name = self._get_single_biomarker_name(db_path, bv_id)
            normalizer_name = self._get_single_biomarker_name(db_path, coeffs['normalizer_biomarker_version_id'])
            print(f"{bv_name}/{normalizer_name}: {coef:.4f}")
        
        if metrics:
            print(f"\n📊 Performance Metrics:")
            print(f"{'Threshold Type':<12} {'Threshold':<10} {'Sensitivity':<11} {'Specificity':<11} {'PPV':<8} {'NPV':<8}")
            print("-" * 70)
            
            for threshold_type in ['se_97', 'se_95', 'max_sum']:
                if threshold_type in metrics_dict:
                    m = metrics_dict[threshold_type]
                    print(f"{threshold_type:<12} {m.threshold:<10.4f} {m.sensitivity:<11.3f} "
                          f"{m.specificity:<11.3f} {m.ppv:<8.3f} {m.npv:<8.3f}")
        
        return True
    
    def _export_roc_normalized_curve(self, db_path: str, args) -> bool:
        """Export ROC Normalized curve data for a model."""
        roc_norm_db = ROCNormalizedAnalysisDatabase(db_path)
        points = roc_norm_db.get_roc_normalized_curve_points_by_model(args.model_id)
        
        if not points:
            print(f"❌ No ROC Normalized curve data found for model {args.model_id}.")
            return False
        
        # Create DataFrame
        import pandas as pd
        curve_data = pd.DataFrame([{
            'FPR': point.fpr,
            'TPR': point.tpr,
            'Threshold': point.threshold
        } for point in points])
        
        if args.output:
            # Save to file
            curve_data.to_csv(args.output, index=False)
            print(f"✅ ROC Normalized curve data saved to: {args.output}")
            print(f"📊 Exported {len(points)} curve points.")
        else:
            # Print to console
            print(f"\n📈 ROC Normalized Curve Data (Model ID: {args.model_id})")
            print(f"Total points: {len(points)}")
            print(f"\nFirst 10 points:")
            print(curve_data.head(10).to_string(index=False, float_format='%.4f'))
            
            if len(points) > 10:
                print(f"\n... and {len(points) - 10} more points")
                print(f"Use --output to save full curve data to CSV file.")
        
        return True
    
    def _get_biomarker_names(self, db_path: str, biomarker_version_ids: list, normalizer_bv_id: int) -> str:
        """Get human-readable biomarker names from version IDs with normalization info."""
        exp_db = ExperimentDatabase(db_path)
        names = []
        
        # Get normalizer name
        normalizer_bv = exp_db.get_biomarker_version_by_id(normalizer_bv_id)
        normalizer_name = "Unknown"
        if normalizer_bv:
            normalizer_biomarker = exp_db.get_biomarker_by_id(normalizer_bv.biomarker_id)
            if normalizer_biomarker:
                normalizer_name = f"{normalizer_biomarker.name}_{normalizer_bv.version}"
        
        for bv_id in biomarker_version_ids:
            bv = exp_db.get_biomarker_version_by_id(bv_id)
            if bv:
                biomarker = exp_db.get_biomarker_by_id(bv.biomarker_id)
                if biomarker:
                    names.append(f"{biomarker.name}_{bv.version}/{normalizer_name}")
                else:
                    names.append(f"BV{bv_id}/{normalizer_name}")
            else:
                names.append(f"BV{bv_id}/{normalizer_name}")
        
        return " + ".join(names)
    
    def _get_single_biomarker_name(self, db_path: str, biomarker_version_id: int) -> str:
        """Get human-readable biomarker name from version ID."""
        exp_db = ExperimentDatabase(db_path)
        bv = exp_db.get_biomarker_version_by_id(biomarker_version_id)
        if bv:
            biomarker = exp_db.get_biomarker_by_id(bv.biomarker_id)
            if biomarker:
                return f"{biomarker.name}_{bv.version}"
            else:
                return f"BV{biomarker_version_id}"
        else:
            return f"BV{biomarker_version_id}"