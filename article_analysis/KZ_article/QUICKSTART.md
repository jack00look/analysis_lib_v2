# Quick Start Guide for KZ Article Analysis

## Adding More Datasets

Edit `config.yaml` and add entries to the `datasets:` section:

```yaml
  - dataset_id: "dataset_003"
    description: "Your description here"
    date:
      year: 2026
      month: 5
      day: 9
    sequences: [15, 16, 17, 18]  # Your sequence numbers
    affine_correction:
      a1: 1.0
      b1: 0.0
      c1: 0.0
      a2: 1.0
      b2: 0.0
      c2: 0.0
    x_min_integration: 900
    x_max_integration: 1200
    domain_balance_fit_window: [910, 1150]
```

## The Analysis Workflow

### Step 1: Interactive Analysis (integrated preprocessing + tuning)

```bash
cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/article_analysis/KZ_article

# Make scripts executable
chmod +x run_kz_article_analysis.py

# Run interactive analysis
python run_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml
```

The script will guide you through the complete workflow, including preprocessing:

**For the first dataset**, you'll be prompted:
```
PREPROCESSING REQUIRED
Would you like to run preprocessing now? [y/n]:
```

Choose `y` to automatically preprocess the raw HDF5 files for that dataset's sequences. The script will:
- Call `run_show_ODs_batch.run_analysis()` programmatically
- Analyze raw HDF5 files from NAS
- Create consolidated HDF files with show_ODs_v2 data
- Then automatically continue to tuning

**For each dataset**, you'll then:
1. See 5 random m1/m2 density profiles from different detunings
2. Tune **affine correction coefficients** (a1, b1, c1, a2, b2, c2)
3. Adjust **integration limits** (x_min, x_max in μm)
4. Review waterfall analysis showing domain walls
5. Set **domain fit window** for velocity extraction
6. Continue to next dataset or finish

**Results** are saved to:
```
results/2026/05/08/iteration_1/
├── config.yaml              # Updated configs with tuned parameters
├── camera_settings.yaml
├── defect_config.yaml
└── metadata.json            # Timestamp, user, iteration number
```

## Subsequent Reruns

After your first analysis iteration, you can:

### Re-run with Same Parameters (Batch Mode)
```bash
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml

# Creates: results/2026/05/08/iteration_2/
# Uses parameters from previous iteration
```

### Re-run with Manual Tuning (Another Interactive Session)
```bash
python run_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml

# Creates: results/2026/05/08/iteration_2/
# Allows you to adjust parameters interactively again
```

## Check Results

```bash
ls results/2026/05/08/iteration_1/
# config.yaml  camera_settings.yaml  defect_config.yaml  metadata.json  dataset_001/

ls results/2026/05/08/iteration_1/dataset_001/
# 01_waterfall.png  02_domain_wall.png  ... (all analysis plots)
```

## Modify Parameters for Next Iteration

1. Edit `config.yaml` with new parameters
2. Run batch script again (new iteration created automatically)

Or modify and rerun specific iteration:

```bash
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --iteration 1
```

## Shot Filtering Details

All analyses apply filtering from multishot_lib:

**Background Contamination**: `cnt_rel_atoms_back < 0.001`
**Saturation**: `cnt_rel_atoms_sat < 1.0`
**Probe Norm Error**: `probe_norm_err < 0.1`
**N_total**: `n_tot > 7.0e5`

These parameters are stored in `config.yaml` under `shared_params.SHOT_FILTER_CONFIG` and can be adjusted globally for all datasets.

## File Structure for Reproducibility

To perfectly reproduce an analysis:

```bash
# Copy these 3 files to new location
cp results/2026/05/08/iteration_1/config.yaml ./
cp results/2026/05/08/iteration_1/camera_settings.yaml ./
cp results/2026/05/08/iteration_1/defect_config.yaml ./

# Run batch analysis - produces identical results
python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml
```

All configuration is self-contained in the iteration folder, so you can always reproduce results exactly.

## Next Steps

- [ ] Edit `config.yaml` with your datasets
- [ ] Edit `camera_settings.yaml` if needed (usually not required)
- [ ] Edit `defect_config.yaml` if needed for different domain wall algorithm
- [ ] Run interactive script: `python run_kz_article_analysis.py ...`
- [ ] Review results in `results/2026/05/08/iteration_1/`
- [ ] Run batch reruns with adjusted parameters as needed
