#!/usr/bin/env python3
"""
Batch KZ Article Analysis Script (Non-Interactive)

Workflow:
1. Load config files (main config with tuned parameters, camera settings, defect config)
2. For each dataset:
   a. Load raw data from specified day/sequences
   b. Apply quality filtering using multishot_lib
   c. Run waterfall_v2 analysis with KZ_det_scan mode
   d. Save all plots to iteration subfolder
3. Save all configs for reproducibility

Usage:
    python batch_kz_article_analysis.py /path/to/config.yaml camera_settings.yaml defect_config.yaml [--iteration N] [--silent]

The config.yaml should already contain tuned parameters (affine_correction, x_min_integration, 
x_max_integration, domain_balance_fit_window) from a previous interactive run or manual editing.
"""

import sys
import os
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import logging
import getpass
import traceback

sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib')
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall')

from multishot_lib import get_and_filter_shots

# Import waterfall_plot from waterfall directory
import importlib.util
spec = importlib.util.spec_from_file_location("waterfall_lib", 
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/waterfall_lib.py")
waterfall_lib_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(waterfall_lib_module)
waterfall_plot = waterfall_lib_module.waterfall_plot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchKZArticleAnalyzer:
    """Batch analyzer for KZ article datasets (non-interactive)"""
    
    def __init__(self, config_path, camera_settings_path, defect_config_path, 
                 iteration=None, silent=False):
        """
        Initialize batch analyzer with config files.
        
        Parameters
        ----------
        config_path : str
            Path to main config.yaml (with tuned parameters)
        camera_settings_path : str
            Path to camera_settings.yaml
        defect_config_path : str
            Path to defect_config.yaml
        iteration : int, optional
            Specific iteration to run (default: auto-increment from today's count)
        silent : bool
            If True, save plots without displaying them
        """
        self.config = self._load_yaml(config_path)
        self.camera_settings = self._load_yaml(camera_settings_path)
        self.defect_config = self._load_yaml(defect_config_path)
        
        self.config_path = config_path
        self.camera_settings_path = camera_settings_path
        self.defect_config_path = defect_config_path
        self.silent = silent
        
        # Determine iteration number
        if iteration is None:
            self.iteration = self._get_next_iteration()
        else:
            self.iteration = iteration
        
        # Setup results directory
        self.results_dir = self._setup_results_dir()
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    @staticmethod
    def _load_yaml(path):
        """Load YAML config file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_next_iteration(self):
        """Determine next iteration number for today"""
        results_base = Path(__file__).parent / "results"
        today = datetime.now()
        year_month_day = today.strftime('%Y/%m/%d')
        day_dir = results_base / year_month_day
        
        if not day_dir.exists():
            return 1
        
        # Find highest iteration number
        max_iter = 0
        for folder in day_dir.iterdir():
            if folder.is_dir() and folder.name.startswith('iteration_'):
                try:
                    iter_num = int(folder.name.split('_')[1])
                    max_iter = max(max_iter, iter_num)
                except (ValueError, IndexError):
                    pass
        
        return max_iter + 1
    
    def _setup_results_dir(self):
        """Setup results directory with year/month/day/iteration_N structure"""
        results_base = Path(__file__).parent / "results"
        today = datetime.now()
        year_month_day = today.strftime('%Y/%m/%d')
        iteration_dir = results_base / year_month_day / f"iteration_{self.iteration}"
        
        iteration_dir.mkdir(parents=True, exist_ok=True)
        return iteration_dir
    
    def run_analysis(self):
        """Run batch analysis on all datasets"""
        logger.info(f"\n{'='*80}")
        logger.info(f"KZ ARTICLE BATCH ANALYSIS - Iteration {self.iteration}")
        logger.info(f"{'='*80}\n")
        
        dataset_results = {}
        failed_datasets = []
        
        for dataset_config in self.config['datasets']:
            dataset_id = dataset_config['dataset_id']
            logger.info(f"\n{'='*80}")
            logger.info(f"DATASET: {dataset_id}")
            logger.info(f"{'='*80}")
            
            try:
                # Load and analyze dataset
                success = self._analyze_dataset(dataset_config)
                
                if success:
                    dataset_results[dataset_id] = dataset_config
                else:
                    failed_datasets.append(dataset_id)
                    logger.error(f"Analysis failed for {dataset_id}")
            
            except Exception as e:
                logger.error(f"Exception during analysis of {dataset_id}: {e}")
                logger.error(traceback.format_exc())
                failed_datasets.append(dataset_id)
        
        # Save results
        self._save_results(dataset_results, failed_datasets)
    
    def _analyze_dataset(self, dataset_config):
        """
        Analyze single dataset without user prompts.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        dataset_id = dataset_config['dataset_id']
        date_cfg = dataset_config['date']
        seqs = dataset_config['sequences']
        camera_name = self.camera_settings['camera_settings']['camera_name']
        
        # Create dataset-specific results directory
        dataset_dir = self.results_dir / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        
        logger.info(f"Loading data for {dataset_id}...")
        
        # Load raw data using get_and_filter_shots which applies quality filters
        hdf_config = {
            'raw_or_processed': 'raw',
            'year': date_cfg['year'],
            'month': date_cfg['month'],
            'day': date_cfg['day'],
        }
        
        try:
            df_filtered = get_and_filter_shots(hdf_config, camera_name, seqs=seqs)
        except Exception as e:
            logger.error(f"Failed to load/filter data: {e}")
            return False
        
        if df_filtered is None or df_filtered.empty:
            logger.error(f"No data after filtering")
            return False
        
        logger.info(f"✓ Loaded {len(df_filtered)} filtered shots")
        
        # Prepare waterfall analysis configuration
        mode_config = {
            'scan': 'KZ_det_scan',
            'seqs': seqs,
            'data_origin': self.camera_settings['camera_settings']['data_origin'],
            'magnetization_modality': self.camera_settings['camera_settings']['magnetization_modality'],
            'shot_filter': self.config['shared_params']['SHOT_FILTER_CONFIG'],
            'magnetization_modalities': self.camera_settings['modality_config'],
            'defect_analysis_global': self.defect_config['defect_analysis'],
            # Dataset-specific parameters
            'affine_correction': dataset_config['affine_correction'],
            'x_min_integration': dataset_config['x_min_integration'],
            'x_max_integration': dataset_config['x_max_integration'],
            'domain_balance_fit_window': dataset_config['domain_balance_fit_window'],
        }
        
        # Shared params
        params = self.config['shared_params']
        
        # Run waterfall analysis
        logger.info(f"Running waterfall_plot analysis for KZ_det_scan...")
        
        try:
            # Suppress plot display if silent mode
            if self.silent:
                plt.ioff()  # Turn off interactive mode
            
            waterfall_plot(
                df=df_filtered,
                seqs=seqs,
                scan='KZ_det_scan',
                mode_cfg=mode_config,
                params=params,
                data_origin=mode_config['data_origin'],
            )
            
            # Save figures to dataset directory
            for fignum in plt.get_fignums():
                fig = plt.figure(fignum)
                fig_title = "figure"
                if fig.suptitle:
                    fig_title = str(fig.suptitle).replace(" ", "_").replace("/", "_").lower()[:50]
                fig_name = f"{fignum:02d}_{fig_title}.png"
                save_path = dataset_dir / fig_name
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"  Saved: {save_path.name}")
                
                if self.silent:
                    plt.close(fig)
            
            logger.info(f"✓ Analysis complete for {dataset_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed during waterfall_plot: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _save_results(self, dataset_results, failed_datasets):
        """Save all configs and results to iteration folder"""
        logger.info(f"\n{'='*80}")
        logger.info("SAVING RESULTS")
        logger.info(f"{'='*80}\n")
        
        # Save config files
        config_copy = dict(self.config)
        config_copy['datasets'] = list(dataset_results.values())
        
        with open(self.results_dir / 'config.yaml', 'w') as f:
            yaml.dump(config_copy, f, default_flow_style=False)
        logger.info(f"✓ Saved config.yaml")
        
        with open(self.results_dir / 'camera_settings.yaml', 'w') as f:
            yaml.dump(self.camera_settings, f, default_flow_style=False)
        logger.info(f"✓ Saved camera_settings.yaml")
        
        with open(self.results_dir / 'defect_config.yaml', 'w') as f:
            yaml.dump(self.defect_config, f, default_flow_style=False)
        logger.info(f"✓ Saved defect_config.yaml")
        
        # Save analysis metadata
        metadata = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'user': getpass.getuser(),
            'mode': 'batch_analysis',
            'silent_mode': self.silent,
            'num_datasets': len(dataset_results),
            'datasets_analyzed': list(dataset_results.keys()),
            'datasets_failed': failed_datasets,
            'num_failed': len(failed_datasets),
        }
        with open(self.results_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved metadata.json")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYSIS SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Iteration: {self.iteration}")
        logger.info(f"Datasets analyzed: {len(dataset_results)}/{len(dataset_results) + len(failed_datasets)}")
        if failed_datasets:
            logger.warning(f"Failed datasets: {failed_datasets}")
        logger.info(f"Results saved to: {self.results_dir}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch KZ Article Analysis (Non-Interactive)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run batch analysis with new iteration
  python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml
  
  # Re-run specific iteration
  python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --iteration 3
  
  # Run silently (save plots without displaying)
  python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --silent
        """
    )
    
    parser.add_argument('config', help='Path to config.yaml (with tuned parameters)')
    parser.add_argument('camera_settings', help='Path to camera_settings.yaml')
    parser.add_argument('defect_config', help='Path to defect_config.yaml')
    parser.add_argument('--iteration', type=int, default=None, 
                        help='Specific iteration number (default: auto-increment)')
    parser.add_argument('--silent', action='store_true',
                        help='Save plots without displaying them')
    
    args = parser.parse_args()
    
    # Verify files exist
    for fpath in [args.config, args.camera_settings, args.defect_config]:
        if not os.path.exists(fpath):
            print(f"ERROR: File not found: {fpath}")
            sys.exit(1)
    
    analyzer = BatchKZArticleAnalyzer(
        args.config,
        args.camera_settings,
        args.defect_config,
        iteration=args.iteration,
        silent=args.silent
    )
    analyzer.run_analysis()
