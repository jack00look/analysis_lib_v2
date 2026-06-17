"""
Waterfall v2 - Multishot analysis using refactored v2 libraries

This script demonstrates how to use the new multishot_lib to:
1. Load shots from a specific day
2. Check parameter consistency across all shots
3. Filter shots by quality metrics (saturation, background, norm_err, DMD update)
4. Return a DataFrame ready for further analysis

Example usage:
    python waterfall_v2_example.py
"""

import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add analysislib_v2 to path
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import multishot_lib
from multishot_lib import get_and_filter_shots, get_day_data

# Import main config
from waterfall_v2.waterfall_config import ACTIVE_MODE, SEQS, HDF_CONFIG

# Import common config
from waterfall_v2.common_config import MAGNETIZATION_MODALITIES

# Auto-load mode config based on ACTIVE_MODE
import importlib
mode_config_module = importlib.import_module(f'waterfall_v2.mode_configs.{ACTIVE_MODE}')
MODE_CONFIG = mode_config_module.MODE_CONFIG

# Import plotting library
from waterfall.waterfall_lib import waterfall_plot
from waterfall_v2.common_config import PARAMS


def main():
    """
    Main waterfall analysis function.
    
    For KZ_det_scan_window / KZ_window_moving modes, also performs:
    1. Domain extraction
    2. Sliding window analysis
    3. Result plotting
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"WATERFALL V2 - MULTISHOT ANALYSIS")
    logger.info(f"{'='*80}\n")
    
    logger.info(f"Mode: {ACTIVE_MODE}")
    logger.info(f"Scan: {MODE_CONFIG['scan']}")
    
    # Get magnetization modality
    mag_modality_name = MODE_CONFIG['magnetization_modality']
    mag_modality = MAGNETIZATION_MODALITIES[mag_modality_name]
    camera_name = mag_modality['camera_name']
    atoms_images = mag_modality['atoms_images']
    
    logger.info(f"Camera: {camera_name}")
    logger.info(f"Atom images: {atoms_images}")
    
    # Determine sequences to load
    seqs_to_load = SEQS if SEQS is not None else None
    if seqs_to_load:
        logger.info(f"Sequences: {seqs_to_load}")
    else:
        logger.info("Sequences: All available")
    
    # Load shots using get_day_data (same as waterfall_FLAT_v4)
    # This loads raw data without quality filtering, for comparison with waterfall_FLAT_v4
    logger.info(f"\n{'='*80}")
    logger.info("LOADING SHOTS (RAW DATA - NO QUALITY FILTERING)")
    logger.info(f"{'='*80}\n")
    
    df = get_day_data(HDF_CONFIG)
    
    if df is None or df.empty:
        logger.error("\nNo shots loaded. Exiting.")
        return
    
    logger.info(f"\n✓ Loaded {len(df)} total shots from HDF files")
    
    logger.info(f"\n{'='*80}")
    logger.info("SHOTS LOADED SUCCESSFULLY")
    logger.info(f"{'='*80}\n")
    
    # Show summary
    logger.info(f"Shape of DataFrame: {df.shape}")
    
    # Display shot information if available
    shot_name_col = None
    for col in df.columns:
        if isinstance(col, tuple) and col[1] == 'shot_name':
            shot_name_col = col
            break
        elif col == 'shot_name':
            shot_name_col = col
            break
    
    if shot_name_col:
        logger.info(f"\nShot names (first 10):")
        for shot_name in df[shot_name_col].head(10).values:
            logger.info(f"  - {shot_name}")
        if len(df) > 10:
            logger.info(f"  ... and {len(df) - 10} more")
    else:
        logger.info(f"\nLoaded {len(df)} shots from the DataFrame")
    
    # At this point, the multishot script can use df to extract:
    # - n1D projections (if available in results)
    # - Processing parameters (cam_vert1_param_*)
    # - Quality metrics (cam_vert1_*_cnt_rel_atoms_sat, etc.)
    # - Any other analysis results stored in HDF5
    
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING WATERFALL PLOTS")
    logger.info(f"{'='*80}\n")
    
    try:
        logger.info("Preparing mode config...")
        
        # Set seqs in mode config (required by run_mode)
        MODE_CONFIG['seqs'] = seqs_to_load
        
        # Inject shared configs into mode runtime payload (like waterfall_FLAT_v4 does)
        from waterfall_v2.common_config import SHOT_FILTER_CONFIG, DEFECT_ANALYSIS_PARAMS
        MODE_CONFIG['shot_filter'] = dict(SHOT_FILTER_CONFIG)
        MODE_CONFIG['magnetization_modalities'] = dict(MAGNETIZATION_MODALITIES)
        MODE_CONFIG['defect_analysis_global'] = dict(DEFECT_ANALYSIS_PARAMS)
        
        logger.info("Calling run_mode to generate waterfall plots...")
        from waterfall.waterfall_lib import run_mode
        run_mode(df, MODE_CONFIG, PARAMS)
        logger.info("✓ Waterfall plots generated successfully!")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # KZ_det_scan_window specific analysis. KZ_window_moving is an alias
    # whose MODE_CONFIG['scan'] points at the same workflow.
    if MODE_CONFIG.get('scan') == 'KZ_det_scan_window':
        logger.info(f"\n{'='*80}")
        logger.info("KZ_det_scan_window EXTENDED ANALYSIS")
        logger.info(f"{'='*80}\n")
        
        try:
            if seqs_to_load is not None:
                df_window = df[df['sequence_index'].isin(seqs_to_load)].copy().reset_index(drop=True)
            else:
                df_window = df.copy().reset_index(drop=True)
            logger.info(f"Window analysis shots: {len(df_window)}")

            # Import domain and window analysis modules
            from waterfall.domain_extraction_lib import create_domain_info_per_shot
            from waterfall.window_analysis_lib import (
                sliding_window_analysis,
                save_window_analysis_results,
                save_window_analysis_csv,
                results_to_dataframe
            )
            from plot_domains_validation import plot_domains_validation
            
            # Step 1: Extract domain information
            logger.info("Step 1: Extracting domain information for all shots...")
            domain_info_dict = create_domain_info_per_shot(
                df_window,
                seqs=seqs_to_load,
                scan='KZ_det_scan_window',
                data_origin='show_ODs_v2',
                mode_cfg=MODE_CONFIG,
                params=PARAMS
            )
            
            if domain_info_dict:
                logger.info(f"✓ Extracted domain info for {len(domain_info_dict)} shots\n")                
                # Plot domain validation FIRST - before any further analysis
                logger.info("\nStep 1b: Creating domain validation plot...")
                try:
                    output_dir = './results/window_analysis'
                    fig_val = plot_domains_validation(
                        df_window, domain_info_dict, PARAMS, MODE_CONFIG,
                        output_dir=output_dir,
                        figname='domains_validation.png'
                    )
                    logger.info("✓ Domain validation plot created\n")
                    logger.info("EXAMINE THE DOMAIN VALIDATION PLOT TO VERIFY DOMAIN DETECTION\n")
                except Exception as e:
                    logger.warning(f"Could not create domain validation plot: {e}\n")
            else:
                logger.warning("No domain info extracted, skipping window analysis.\n")
                return df
            
            # Step 2: Perform sliding window analysis
            logger.info("Step 2: Performing sliding window analysis...")
            window_size_um = float(MODE_CONFIG.get('window_size', 80.0))
            window_step_um = float(MODE_CONFIG.get('window_step', 5.0))
            x_min_um = float(PARAMS.get('X_MIN_INTEGRATION', 920.0))
            x_max_um = float(PARAMS.get('X_MAX_INTEGRATION', 1180.0))
            
            logger.info(f"  Window: size={window_size_um:.1f} μm, step={window_step_um:.1f} μm")
            logger.info(f"  Range: [{x_min_um:.1f}, {x_max_um:.1f}] μm\n")
            
            results = sliding_window_analysis(
                df_window,
                x_min_um=x_min_um,
                x_max_um=x_max_um,
                window_size_um=window_size_um,
                window_step_um=window_step_um,
                mode_cfg=MODE_CONFIG,
                params=PARAMS,
                domain_info_dict=domain_info_dict
            )
            
            if results:
                logger.info(f"\n✓ Analyzed {len(results)} windows\n")
            else:
                logger.warning("No valid windows analyzed.\n")
                return df
            
            # Step 3: Save results
            logger.info("Step 3: Saving analysis results...")
            output_dir = './results/window_analysis'
            save_window_analysis_results(results, output_dir=output_dir)
            save_window_analysis_csv(results, output_dir=output_dir)
            
            # Step 4: Create plots
            logger.info("\nStep 4: Creating result plots...")
            try:
                from plot_window_analysis import plot_window_analysis, plot_window_analysis_with_domains, plot_raw_magnetization_by_field
                
                fig1, _ = plot_window_analysis(results, window_size_um, 
                                              output_dir=output_dir,
                                              figname='window_analysis_twin_axis.png')
                
                fig2, _ = plot_window_analysis_with_domains(results, window_size_um,
                                                           output_dir=output_dir,
                                                           figname='window_analysis_detailed.png')
                
                # Plot raw magnetization grouped by field
                # Build field_summary from results
                field_summary = {}
                for result in results:
                    for field_val_str, field_data in result['field_summary'].items():
                        try:
                            field_val = float(field_val_str)
                            if field_val not in field_summary:
                                field_summary[field_val] = field_data
                        except (ValueError, TypeError):
                            continue
                
                if field_summary:
                    fig3 = plot_raw_magnetization_by_field(df_window, field_summary, domain_info_dict, PARAMS,
                                                          output_dir=output_dir,
                                                          figname='raw_magnetization_by_field.png')
                
                logger.info("✓ Plots created successfully!\n")
                
            except Exception as e:
                logger.warning(f"Could not create plots: {e}\n")
            
            logger.info(f"\n{'='*80}")
            logger.info("KZ_det_scan_window ANALYSIS COMPLETE")
            logger.info(f"Results saved to: {output_dir}/")
            logger.info(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Error in KZ_det_scan_window analysis: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*80}")
    logger.info("Ready for further analysis (e.g., waterfall plots, fitting, etc.)")
    logger.info(f"{'='*80}\n")
    
    return df


if __name__ == '__main__':
    df = main()
