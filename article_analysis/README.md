# Article Analysis Framework

A comprehensive framework for publishing reproducible scientific analysis of atomic physics experiments. Designed for parameter tuning, batch processing, and complete provenance tracking.

## Purpose

This framework enables:
1. **Interactive parameter tuning** with visual feedback
2. **Automated batch reanalysis** with fixed parameters
3. **Perfect reproducibility** through config file storage
4. **Publication-ready organization** with full documentation

## Structure

```
article_analysis/
├── KZ_article/                          # Example article (KZ detuning scan)
│   ├── config.yaml                      # Dataset list and parameters
│   ├── camera_settings.yaml             # Camera configuration
│   ├── defect_config.yaml               # Algorithm parameters
│   ├── run_kz_article_analysis.py       # Interactive analysis script
│   ├── batch_kz_article_analysis.py     # Batch analysis script
│   ├── README.md                        # Comprehensive guide
│   ├── QUICKSTART.md                    # Quick reference
│   ├── IMPLEMENTATION.md                # Technical details
│   └── results/
│       └── YYYY/MM/DD/
│           ├── iteration_1/             # First run results
│           ├── iteration_2/             # Refined parameters
│           └── ...
├── setup_new_article.sh                 # Template for new articles
└── README.md                            # This file
```

## Quick Start

### For KZ Article (Example):

```bash
cd KZ_article

# Interactive tuning (first time)
python run_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml

# Results saved to: results/YYYY/MM/DD/iteration_1/

# Batch reruns (with automatic parameter tweaks)
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --silent

# Results saved to: results/YYYY/MM/DD/iteration_2/
```

### Create New Article:

```bash
bash setup_new_article.sh MyArticleName

cd MyArticleName

# Edit configs
nano config.yaml
nano camera_settings.yaml
nano defect_config.yaml

# Run analysis
python run_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml
```

## Article Components

### 1. Configuration Files

**`config.yaml`** - Main configuration
- Article metadata
- Shared shot filtering parameters
- Dataset list with per-dataset tunable parameters

**`camera_settings.yaml`** - Camera setup
- Camera name, atom images, data_origin
- Magnetization modality
- Typically shared across all datasets

**`defect_config.yaml`** - Algorithm parameters
- Domain wall detection parameters
- Shared across all datasets
- Can be customized per article

### 2. Analysis Scripts

**Interactive**: `run_*_article_analysis.py`
- Visual parameter tuning
- Shows sample profiles
- Prompts for user input
- Ideal for initial exploration

**Batch**: `batch_*_article_analysis.py`
- Non-interactive processing
- Fixed parameters from config
- Silent plot saving
- Ideal for publication runs

### 3. Results Organization

```
results/YYYY/MM/DD/iteration_N/
├── config.yaml                  # Used configurations
├── camera_settings.yaml
├── defect_config.yaml
├── metadata.json                # Analysis metadata
└── dataset_ID/
    ├── 01_waterfall.png         # Analysis plots
    ├── 02_domain_wall.png
    └── ...
```

## Key Features

### Reproducibility
- All configs saved with each iteration
- Can re-run exact same analysis from iteration folder
- Perfect for publishing supplementary data

### Flexibility
- Easy to add new datasets to config
- Parameter tuning through visual inspection
- Batch processing for multiple datasets

### Quality Control
- Shot quality filtering via multishot_lib
- Configurable filtering thresholds
- Transparent rejection statistics

### Organization
- Automatic iteration numbering (per-day)
- Per-dataset result folders
- Complete metadata tracking

## Workflow Patterns

### Pattern 1: New Article from Scratch
```
1. Copy KZ_article template
2. Edit config.yaml with your datasets
3. Run interactive script
4. Review and adjust parameters
5. Save to version control
6. Batch rerun for final publication
```

### Pattern 2: Different Experiment Type
```
1. Run setup_new_article.sh
2. Modify scripts for your scan type (e.g., bubbles_evolution → ARP_Forward)
3. Adjust camera_settings.yaml
4. Adjust defect_config.yaml
5. Run analysis
```

### Pattern 3: Publication Reruns
```
1. Find iteration folder from previous run
2. Copy config files
3. Run batch script with copied configs
4. Produces identical results for publication
```

## Shot Filtering

All analyses include quality filtering:

- **Background Contamination**: `cnt_rel_atoms_back < back_threshold`
- **Saturation**: `cnt_rel_atoms_sat < sat_threshold`
- **Probe Norm Error**: `probe_norm_err < probe_norm_err_threshold`
- **N_total**: `n_tot > ntot_threshold` (always enabled)

Thresholds configured globally in `config.yaml` for consistency.

## Integration

### With multishot_lib
- `get_and_filter_shots()` loads and filters raw data
- Quality metrics applied automatically
- Consistent across all analyses

### With waterfall_lib
- Uses newer version from `waterfall/` directory
- Provides KZ_det_scan, bubbles_evolution, ARP_Forward modes
- Complete magnetization and domain wall analysis

### With Lyse/HDF5
- Reads raw shot data from NAS542_dataBEC2
- Automatic date-based directory navigation
- Supports both raw and reanalyzed data

## Directory Tree

```
article_analysis/
├── KZ_article/
│   ├── config.yaml
│   ├── camera_settings.yaml
│   ├── defect_config.yaml
│   ├── run_kz_article_analysis.py (main script for KZ_det_scan)
│   ├── batch_kz_article_analysis.py
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── IMPLEMENTATION.md
│   └── results/
│       └── 2026/05/08/
│           ├── iteration_1/
│           ├── iteration_2/
│           └── ...
├── BEC_formation/                       # Can create new articles
│   ├── config.yaml
│   ├── camera_settings.yaml
│   ├── defect_config.yaml
│   ├── run_article_analysis.py
│   ├── batch_article_analysis.py
│   ├── README.md
│   └── results/
├── setup_new_article.sh
└── README.md (this file)
```

## Tips

1. **Initial Setup**: Start with KZ_article as template, don't modify directly
2. **Parameter Tuning**: Use interactive script for visual feedback
3. **Batch Runs**: Use batch script with `--silent` for final publication
4. **Reproducibility**: Always keep iteration folder configs with published results
5. **Version Control**: Add KZ_article to git, exclude results/ folder

## Common Use Cases

### Use Case 1: Publication Figure
```bash
# Interactive tuning
cd KZ_article
python run_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml

# Review iteration_1/ results
# Adjust parameters if needed

# Final batch run
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --silent

# Publish with configs from iteration_N/
```

### Use Case 2: Verify Results from Paper
```bash
# Get configs from supplementary materials
cp paper_supplementary/iteration_N/*.yaml ./

# Reproduce exact same analysis
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml

# Results will be identical to published figures
```

### Use Case 3: New Experiment, Similar Protocol
```bash
bash setup_new_article.sh MyExperiment
cd MyExperiment

# Edit configs for your data
nano config.yaml

# Run analysis with new parameters
python run_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml
```

## Troubleshooting

**Script won't run**: Check that Python environment has required packages
```bash
conda activate labscr_py38
python -c "import lyse, h5py, pandas, numpy, matplotlib, yaml"
```

**No data loaded**: Verify NAS path and date/sequences in config
```bash
ls /home/rick/NAS542_dataBEC2/2026/05/07/
```

**Plots not saving**: Check write permissions in results directory
```bash
touch article_analysis/test_write.txt && rm article_analysis/test_write.txt
```

**Iteration numbering reset**: Each calendar day creates new iteration 1
```bash
# Tomorrow creates results/2026/05/09/iteration_1/
# Today creates results/2026/05/08/iteration_1/
```

## Documentation

- **README.md** (this file) - Overview and quick reference
- **KZ_article/QUICKSTART.md** - Quick start guide
- **KZ_article/README.md** - Comprehensive KZ article documentation
- **KZ_article/IMPLEMENTATION.md** - Technical implementation details

## Next Steps

1. Start with KZ_article example
2. Read KZ_article/QUICKSTART.md
3. Run interactive script with example data
4. Create new article for your experiment
5. Integrate into your publication workflow

