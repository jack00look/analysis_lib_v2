# KZ Article Analysis Framework - Implementation Summary

## Overview

A complete framework for interactive and batch analysis of KZ detuning scan datasets for publication-quality research. Enables parameter tuning through visual inspection, reproducible batch reruns, and complete provenance tracking.

## Created Files

### Configuration Files

1. **`config.yaml`** - Main configuration
   - Article metadata
   - Shared shot filtering parameters (background, saturation, probe_norm_err, n_tot thresholds)
   - Dataset list with per-dataset tunable parameters:
     - date (YYYY, MM, DD)
     - sequences (list of sequence numbers)
     - affine_correction (a1, b1, c1, a2, b2, c2)
     - x_min_integration, x_max_integration (micrometers)
     - domain_balance_fit_window ([x_min, x_max])

2. **`camera_settings.yaml`** - Camera configuration (shared across article)
   - Magnetization modality (vert1_two_component)
   - Camera name (cam_vert1)
   - Atom images (PTAI_m1, PTAI_m2)
   - Data origin (show_ODs_v2)
   - Apply n_tot check flag (always True for this article)

3. **`defect_config.yaml`** - Domain wall defect-finding algorithm parameters
   - Gaussian smoothing sigma
   - Peak detection thresholds
   - Peak quality filter parameters
   - Geometric cleanup parameters
   - Histogram control parameters

### Main Scripts

1. **`run_kz_article_analysis.py`** - Interactive Analysis Script
   - **Purpose**: Initial parameter tuning and exploration
   - **Workflow**:
     1. For each dataset:
        - Load raw data from NAS
        - Apply shot quality filtering (via multishot_lib)
        - Display 5 random m1/m2 profiles from different detunings
        - Prompt user to adjust affine correction coefficients
        - Prompt user to adjust integration limits (x_min, x_max)
        - Run waterfall_v2 analysis (KZ_det_scan mode)
        - Display domain wall plots
        - Prompt user to adjust domain_balance_fit_window
        - Ask for permission to continue to next dataset
     2. Save iteration folder with all configs and analysis results
   - **Output**: `results/YYYY/MM/DD/iteration_N/`
   - **Reproducibility**: All configs saved for exact reproduction

2. **`batch_kz_article_analysis.py`** - Batch Analysis Script (Non-Interactive)
   - **Purpose**: Automated reanalysis with fixed parameters
   - **Features**:
     - No user prompts (uses tuned parameters from config.yaml)
     - Can run silently (`--silent` flag) for batch processing
     - Accepts specific iteration number to re-run
     - Auto-increments iteration if not specified
   - **Output**: `results/YYYY/MM/DD/iteration_N/`
   - **Usage Examples**:
     ```bash
     # New iteration
     python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml
     
     # Silent mode (save plots without displaying)
     python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --silent
     
     # Re-run specific iteration
     python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --iteration 3
     ```

### Documentation

1. **`README.md`** - Comprehensive documentation
   - Framework structure and workflow
   - Configuration setup instructions
   - Interactive and batch analysis workflows
   - Shot quality filtering details
   - Reproducibility guidelines
   - Parameter tuning tips
   - Common issues and solutions

2. **`QUICKSTART.md`** - Quick reference guide
   - Adding datasets to config.yaml
   - First analysis workflow
   - Batch reruns
   - Checking results
   - File structure for reproducibility
   - Next steps checklist

## Key Features

### 1. Shot Quality Filtering
- Applied during data loading via `get_and_filter_shots()`
- Filters based on:
  - Background contamination (`back_threshold: 0.001`)
  - Saturation (`sat_threshold: 1.0`)
  - Probe norm error (`probe_norm_err_threshold: 0.1`)
  - N_total (`ntot_threshold: 7.0e5`)
- Consistent across all datasets in article

### 2. Interactive Parameter Tuning
- Visual inspection of 5 random profiles per dataset
- Manual adjustment of:
  - Affine correction coefficients (gain balancing)
  - Integration limits (x_min, x_max in micrometers)
  - Domain wall fit window
- Real-time waterfall plot generation with feedback

### 3. Batch Processing
- Non-interactive reruns with fixed parameters
- Silent mode for automated analysis
- Configurable iteration numbering
- Perfect for publication-ready final runs

### 4. Full Reproducibility
- All configurations saved to iteration folder
- Can re-run exact same analysis from saved configs
- Metadata tracking (timestamp, user, num_datasets)
- Version control friendly

### 5. Organized Output
```
results/
└── YYYY/MM/DD/
    ├── iteration_1/
    │   ├── config.yaml
    │   ├── camera_settings.yaml
    │   ├── defect_config.yaml
    │   ├── metadata.json
    │   └── dataset_001/
    │       ├── 01_waterfall.png
    │       ├── 02_domain_wall.png
    │       └── ...
    ├── iteration_2/
    └── ...
```

## Data Pipeline

```
Raw HDF5 Data (NAS)
        ↓
get_and_filter_shots() [multishot_lib]
        ↓
Quality Filtering (background, saturation, norm_err, n_tot)
        ↓
waterfall_plot() [waterfall/waterfall_lib.py]
        ↓
KZ_det_scan Analysis with:
  - Affine correction
  - Integration limits
  - Domain wall defect finding
        ↓
Saved Plots + Configs
```

## Usage Workflow

### First Time: Interactive Tuning
```bash
python run_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml
# Results: results/2026/05/08/iteration_1/
```

### Subsequent Iterations: Batch Reruns
```bash
# Adjust parameters in config.yaml

# Run batch analysis (silent)
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --silent
# Results: results/2026/05/08/iteration_2/
```

### Perfect Reproduction
```bash
# Copy iteration configs
cp results/2026/05/08/iteration_1/*.yaml ./

# Exact reproduction
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml
# Results: identical to iteration_1
```

## Integration with Existing Tools

- **multishot_lib**: Used for shot loading and quality filtering
- **waterfall_lib**: Uses the newer version from `waterfall/` directory (not `waterfall_v2/`)
- **lyse/HDF5**: Reads raw shot data from NAS542_dataBEC2

## Customization

To adapt for different experiments (e.g., different scan types):

1. Create new folder: `article_analysis/YourExperiment/`
2. Copy config templates and scripts
3. Modify:
   - `scan` parameter in MODE_CONFIG (e.g., 'bubbles_evolution', 'ARP_Forward')
   - Camera settings and defect parameters as needed
   - Dataset list and initial tuning values

## Notes

- Scripts use `matplotlib.use('Agg')` compatible code (can run headless)
- All paths use absolute file paths for maximum portability
- Iteration numbering auto-increments by day (resets at midnight)
- Shot filtering thresholds are global but can be edited per-article in config.yaml
- N_tot check is always enabled (configured in camera_settings.yaml)
