# Automated DMD Feedback Update Workflow

This **complete, all-in-one** workflow automates your entire feedback process. Once your ARP_Backward scan completes and waterfall analysis finishes, you can run a single command to:

1. Load the base DMD profile (from h5 or txt)
2. Load the new sigmoid from waterfall
3. Apply feedback correction
4. Save the updated profile with auto-increment naming
5. **Visualize** old vs new profile with error decomposition
6. Update `plot_sigmoid_txt.py` configuration
7. **Automatically load to the DMD server**

## Quick Start

### 1. Configure Base Behavior

Edit `feedback_automation_config.py` to set:

```python
# Choose profile source (h5 or txt)
USE_H5_PROFILE = True

# If using h5, specify the reference shot
H5_SHOT_YEAR = 2026
H5_SHOT_MONTH = 3
H5_SHOT_DAY = 30
H5_SHOT_SEQUENCE = 2
H5_SHOT_ITERATION = 8
H5_SHOT_REPETITION = 38

# Or use txt file (set USE_H5_PROFILE = False)
PROFILE_TXT_FILE = 'dmd_profile_20260331_103346.txt'

# Load to DMD automatically (can skip with --no-server)
LOAD_TO_DMD = True
```

### 2. Run After Scan

After your ARP_Backward scan completes:

```bash
cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD

# Complete workflow: save, plot, and load to DMD
python auto_feedback_update.py

# Override parameters
python auto_feedback_update.py --kp 1.5 --sigma 5.0

# Save and plot, but don't load to DMD (for review)
python auto_feedback_update.py --no-server

# Then view all profiles together
python plot_sigmoid_txt.py
```

## What It Does

### Step-by-Step Workflow

1. **Load Base Profile** - From h5 file or txt file
2. **Load New Sigmoid** - From waterfall analysis output
3. **Apply Correction** - Extend sigmoid, apply Gaussian smoothing, combine with gains
4. **Save Profile** - Auto-increment naming: `sigmoid_center_interpolation_update_0.txt`
5. **Show Visualization** - Displays old vs new profile comparison with error decomposition
6. **Update Plot Config** - Adds new profile to `plot_sigmoid_txt.py` automatically
7. **Load to DMD** - Sends profile to DMD server (can skip with `--no-server`)

### Visualization

The script automatically shows:
- **Top plot**: Old DMD profile (blue) vs New DMD profile (red)
- **Bottom plot**: Raw sigmoid (light), smoothed sigmoid (medium), applied correction (dark)
- Feedback region and wall boundaries marked

## Configuration Options

### In `feedback_automation_config.py`:

- **USE_H5_PROFILE**: Load base profile from h5 file or txt
- **H5_SHOT_***: Reference shot parameters (if using h5)
- **PROFILE_TXT_FILE**: Txt file with base profile (if not using h5)
- **NEW_SIGMOID_PATH**: Path to new sigmoid from waterfall
- **DEFAULT_SMOOTHING_SIGMA**: Gaussian blur sigma
- **NEW_PROFILE_KP**: Proportional gain for feedback
- **PROFILE_KP_MULTIPLICITY**: List of gains to apply (e.g., `[0.5, 0.5]` for two passes)
- **AUTO_UPDATE_PLOT_CONFIG**: Auto-update `plot_sigmoid_txt.py`?
- **LOAD_TO_DMD**: Auto-load to DMD server?
- **CREATE_BACKUPS**: Create backups of modified files

### Command-Line Overrides:

```bash
--kp VALUE           Override NEW_PROFILE_KP
--sigma VALUE        Override DEFAULT_SMOOTHING_SIGMA
--no-server          Don't load to DMD server (just save & plot)
--no-plot-update     Don't update plot configuration
```

## Workflow Examples

### Scenario 1: Quick Feedback Iteration

```python
# feedback_automation_config.py
USE_H5_PROFILE = False
PROFILE_TXT_FILE = 'sigmoid_center_interpolation0.txt'
LOAD_TO_DMD = True  # Auto-load for quick iteration
```

```bash
# Run and everything is done
python auto_feedback_update.py --kp 1.0 --sigma 2.0
# → Saves profile, shows plot, loads to DMD automatically
```

### Scenario 2: Conservative Approach (Review Before Loading)

```python
# feedback_automation_config.py
LOAD_TO_DMD = False  # Manual review step
```

```bash
# Step 1: Generate and review
python auto_feedback_update.py --kp 1.0 --sigma 2.0
# → Shows plot for inspection

# Step 2: If satisfied, load to server
python auto_feedback_update.py --kp 1.0 --sigma 2.0  # Will ask to confirm or just add --server
```

### Scenario 3: Multi-Iteration Campaign

**Day 1:**
```bash
python auto_feedback_update.py --kp 1.0 --sigma 2.0
# Creates: sigmoid_center_interpolation_update_0.txt
```

**Day 2 (update config and run again):**
```python
# feedback_automation_config.py
PROFILE_TXT_FILE = 'sigmoid_center_interpolation_update_0.txt'
```

```bash
python auto_feedback_update.py --kp 1.2 --sigma 3.0
# Creates: sigmoid_center_interpolation_update_1.txt
```

## Output Files

After running the script:

- **`sigmoid_center_interpolation_update_N.txt`**: New DMD profile sent to server
- **`plot_sigmoid_txt.py.bak`**: Backup of plot config (if enabled)
- **Console output**: Complete workflow log

## Integration with Existing Tools

The automation workflow **integrates seamlessly** with your existing scripts:

- **`feedback_manual_update.py`**: For complex multi-profile corrections
- **`plot_sigmoid_txt.py`**: Automatically updated to show new profile
- **DMD Server**: Auto-loads profiles (or manual control with `--no-server`)

## Troubleshooting

### "Could not find new sigmoid file"
- Ensure waterfall scan completed
- Check `NEW_SIGMOID_PATH` in config
- Usually: `/path/to/waterfall/sigmoid_center_interpolation.txt`

### "Could not load base DMD profile"
- If using h5: verify shot numbers
- If using txt: verify file exists with 2 columns
- Check file has header or `skiprows=1` setting

### "Failed to load to DMD server"
- Verify DMD server is running
- Check IP/port in `feedback_config.py`
- Use `--no-server` to skip if needed

### Plot not updating
- Set `AUTO_UPDATE_PLOT_CONFIG = True` in config
- Check write permissions on `plot_sigmoid_txt.py`
- Inspect `.bak` backup if needed

## Advanced Features

### Multiple Gain Passes

Apply the same sigmoid multiple times with different gains:

```python
# feedback_automation_config.py
PROFILE_KP_MULTIPLICITY = [0.5, 0.5]  # Apply twice with 0.5 each
```

### Custom Output Directory

```python
# feedback_automation_config.py
DMD_PROFILES_FOLDER = '/custom/path/to/dmd/profiles'  # Or None for DMD folder
```

### Dry Run (Save Only)

```bash
python auto_feedback_update.py --no-server --no-plot-update
# Just saves the profile, nothing else
```

## Performance

Typical execution time: **< 5 seconds**

- Profile loading: ~0.5s
- Correction calculation: ~1s
- File I/O: ~0.5s
- DMD server communication: ~1-2s (network dependent)
- Plotting: ~2-3s
