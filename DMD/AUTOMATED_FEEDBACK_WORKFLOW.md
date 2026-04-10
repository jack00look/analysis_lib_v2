# Automated Feedback Workflow - Complete Implementation

## Overview

This implementation provides a **persistent profile accumulation system** for automated DMD feedback. Unlike a single-profile approach, this system maintains a list of all sigmoids used during a feedback campaign, allowing each run to add to the accumulated corrections.

## Core Components

### 1. **feedback_sigmoids_list.py** - Persistent Profile Storage
The central configuration file that maintains the list of all sigmoids used in the feedback workflow.

**Structure:**
```python
SIGMOID_PROFILES = [
    {
        'filename': 'sigmoid_center_interpolation0.txt',      # File to load
        'kp': 1.0,                                            # Strength parameter for this sigmoid
        'smoothing_sigma': 2.0,                               # Gaussian smoothing width
        'description': 'Initial profile - baseline',
    },
    # ... add more profiles as feedback runs accumulate
]

DEFAULT_NEW_SIGMA = 2.0   # Default smoothing for new sigmoids
DEFAULT_NEW_KP = 0.8      # Default strength for new sigmoids
```

**Usage:**
- Automatically updated by `auto_feedback_update.py` after each run
- Manually editable to adjust `kp` or `smoothing_sigma` between runs
- Can be modified to remove profiles or change parameters
- Each profile applied independently with its own kp and sigma

### 2. **feedback_automation_config.py** - Automation Settings
Central configuration for the automation system.

**Key Settings:**
```python
# Base profile source (None to auto-detect, or explicit path)
DMD_PROFILES_FOLDER = None  # Will use script directory if None

# Sigmoid source from waterfall analysis
NEW_SIGMOID_PATH = "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/sigmoid_center_interpolation.txt"

# Whether to automatically load profile to DMD server on startup
LOAD_TO_DMD = True

# Whether to use h5 or txt for base profile
USE_H5_PROFILE = False
PROFILE_H5_FILE = None
PROFILE_TXT_FILE = None  # Will auto-detect sigmoid_center_interpolation.txt

# Feedback wall configuration (from feedback_config)
WALL_TYPE = 'hard'  # or 'soft' or None
```

### 3. **auto_feedback_update.py** - Main Automation Script
Complete streamlined automation with profile accumulation.

## Workflow - Step by Step

Each time you run `auto_feedback_update.py`:

### Step 1: Load Base Profile
- Loads the initial DMD profile (from h5 or txt)
- This is the starting point for corrections

### Step 2: Load New Sigmoid
- Loads the sigmoid from waterfall analysis: `sigmoid_center_interpolation.txt`
- This represents the new error to be corrected

### Step 3: Save and Register New Sigmoid
- Auto-saves the sigmoid with incrementing name: `sigmoid_center_interpolation_update_0.txt`, `_update_1.txt`, etc.
- Adds the new sigmoid to `SIGMOID_PROFILES` list in `feedback_sigmoids_list.py`
- Uses DEFAULT_NEW_KP and DEFAULT_NEW_SIGMA (or user-provided parameters)

### Step 4: Load ALL Sigmoids from List
- Loads every sigmoid in `SIGMOID_PROFILES`
- Applies each with its individual kp and smoothing_sigma parameters
- **Accumulates corrections**: `new_profile = old_profile + Σ(kpi * smoothed_sigmoidi)`

### Step 5: Visualize
- Shows comparison plots:
  - **Top**: Old vs New DMD profile
  - **Bottom**: All sigmoid contributions overlaid

### Step 6: Update Plot Config
- Automatically updates `plot_sigmoid_txt.py` configuration

### Step 7: Load to DMD Server
- Sends the combined profile to the DMD server (default behavior, can disable with `--no-server`)

## Usage

### Basic Usage
```bash
# Run with defaults (uses config values for kp and sigma)
python auto_feedback_update.py

# Run with custom kp and sigma
python auto_feedback_update.py --kp 0.9 --sigma 1.5

# Run without loading to DMD server (preview mode)
python auto_feedback_update.py --no-server

# Run without updating plot config
python auto_feedback_update.py --no-plot-update
```

### Manual Workflow
1. Run ARP_Backward scan
2. Analyze in waterfall → saves sigmoid_center_interpolation.txt
3. Run `python auto_feedback_update.py` → saves sigmoid, updates list, applies all corrections
4. Check plot visualization
5. *Optional*: Manually edit `feedback_sigmoids_list.py` to adjust kp/sigma for next run
6. Repeat steps 1-5

### Manual Parameter Adjustment
Edit `feedback_sigmoids_list.py` to fine-tune between runs:

```python
SIGMOID_PROFILES = [
    {
        'filename': 'sigmoid_center_interpolation0.txt',
        'kp': 0.8,        # <-- Adjust strength
        'smoothing_sigma': 2.0,  # <-- Adjust smoothing
        'description': 'Initial profile',
    },
    {
        'filename': 'sigmoid_center_interpolation_update_0.txt',
        'kp': 0.9,        # <-- Increased strength for this one
        'smoothing_sigma': 1.5,  # <-- Narrower smoothing
        'description': 'First feedback iteration',
    },
]
```

### Removing a Profile
Simply comment out or delete the entry:
```python
SIGMOID_PROFILES = [
    # {
    #     'filename': 'sigmoid_center_interpolation_update_0.txt',
    #     'kp': 0.9,
    #     'smoothing_sigma': 1.5,
    #     'description': 'First feedback iteration - disabled',
    # },
    {
        'filename': 'sigmoid_center_interpolation_update_1.txt',
        'kp': 1.0,
        'smoothing_sigma': 2.0,
        'description': 'Second iteration',
    },
]
```

## Key Features

✅ **Profile Accumulation**: Each run adds to the list, not replaces it
✅ **Independent Parameters**: Each sigmoid has its own kp and sigma
✅ **Manual Control**: Edit feedback_sigmoids_list.py to adjust anytime
✅ **Persistent History**: All sigmoids used are tracked and saved
✅ **Visualization**: Multi-sigmoid plots show all contributions
✅ **Automatic Numbering**: New sigmoids auto-numbered (0, 1, 2, ...)
✅ **DMD Integration**: Auto-loads to server by default
✅ **Graceful Fallbacks**: Handles missing files and h5 support optionally

## File Structure

```
analysislib_v2/DMD/
├── auto_feedback_update.py              # Main automation script
├── feedback_automation_config.py         # Automation settings
├── feedback_sigmoids_list.py             # Persistent profile list
├── feedback_manual_update.py             # Original manual script (reference)
├── plot_sigmoid_txt.py                   # Visualization helper
│
├── sigmoid_center_interpolation0.txt     # Initial base profile
├── sigmoid_center_interpolation_update_0.txt   # First accumulated sigmoid
├── sigmoid_center_interpolation_update_1.txt   # Second accumulated sigmoid
└── ...
```

## Important Notes

1. **List Persistence**: The `SIGMOID_PROFILES` list is automatically updated by the script. Don't manually add entries unless you're creating them manually.

2. **File Locations**: All sigmoid files should be in the DMD folder (where `auto_feedback_update.py` is located).

3. **Waterfall Integration**: The waterfall analysis must produce `sigmoid_center_interpolation.txt` in the path specified by `NEW_SIGMOID_PATH` (or in the DMD folder).

4. **Backwards Compatibility**: The original `feedback_manual_update.py` still works for complex custom corrections not suitable for automation.

5. **Configuration Editing**: Edit `feedback_automation_config.py` to change default paths, enable h5 support, or adjust wall parameters.

## Troubleshooting

**Error: "No sigmoids found in feedback_sigmoids_list.py"**
- Solution: Add at least one entry to SIGMOID_PROFILES with an existing sigmoid file

**Error: "Could not load sigmoid_center_interpolation.txt"**
- Solution: Run waterfall analysis first to generate the sigmoid file

**DMD not receiving profile**
- Check: Is `LOAD_TO_DMD = True` in feedback_automation_config.py?
- Check: Is the zerorpc DMD server running?
- Try: `python auto_feedback_update.py --no-server` to skip DMD loading (preview mode)

**List file gets corrupted**
- The script uses robust regex-based modification. If corruption occurs, restore from backup or manually recreate the list.

## Advanced: Combining with feedback_manual_update.py

You can use both scripts in the same workflow:
- Use `auto_feedback_update.py` for standard feedback iterations
- Use `feedback_manual_update.py` when you need to add multiple sigmoids manually with custom parameters
- The list-based approach means both scripts can work with the same accumulated profile list
