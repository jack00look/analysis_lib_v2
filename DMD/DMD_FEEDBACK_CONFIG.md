# DMD Feedback Configuration and Parameters

## Overview

All DMD feedback parameters have been centralized into a configuration file for easier management and consistency between scripts.

## Files Created/Modified

### 1. **feedback_config.py** (NEW)
Central configuration file containing all feedback parameters:

```python
# Server connection
DMD_SERVER_IP = '192.168.1.154'
DMD_SERVER_PORT = '6001'

# Separate configurations for density and magnetization feedback
DENSITY_CONFIG = {
    'program_name': 'Do_BEC_DMD_feedback',
    'x_center': 1040,              # µm
    'feedback_width': 150,          # µm
    'n_threshold': 1e6,
    'kp': 1e-10,
    'smoothing_sigma': 20.0,
}

MAGNETIZATION_CONFIG = {
    'program_name': 'Do_BEC_DMD_magnetization_feedback',
    'x_center': 1040,
    'feedback_width': 150,
    'n_threshold': 1e6,
    'kp': 1e-10,
    'smoothing_sigma': 10.0,
}

# Shared wall configuration
WALL_CONFIG = {
    'wall_type': 'soft',            # 'none', 'hard', or 'soft'
    'soft_wall_width': 50,          # µm
}

# Initial profile configuration
INITIAL_PROFILE_CONFIG = {
    'background_value': 1.0,
    'feedback_value': 0.2,
    'x_center': 1040,
    'feedback_width': 150,
    'wall_type': 'soft',
    'soft_wall_width': 50,
}
```

### 2. **feedback_atoms.py** (MODIFIED)
- Removed hardcoded parameters
- Now loads from `DENSITY_CONFIG` in `feedback_config.py`
- Loads wall configuration from `WALL_CONFIG`

### 3. **feedback_magnetization.py** (MODIFIED)
- Removed hardcoded parameters
- Now loads from `MAGNETIZATION_CONFIG` in `feedback_config.py`
- Loads wall configuration from `WALL_CONFIG`

### 4. **load_initial_profile.py** (NEW)
Generates and loads initial DMD profile with configurable walls:

```bash
python load_initial_profile.py
```

Features:
- Creates profile with feedback region and background
- Supports hard walls (immediate jump to background)
- Supports soft walls (cos² smooth transition)
- Plots current DMD image and generated profile
- Shows configuration summary

## How to Configure

### Changing Feedback Parameters

Edit `feedback_config.py`:

```python
DENSITY_CONFIG = {
    'x_center': 1000,           # Change center position
    'feedback_width': 200,      # Change feedback region width
    'kp': 5e-10,                # Adjust gain
    'smoothing_sigma': 15.0,    # Change error smoothing
    'n_threshold': 2e6,         # Change atom number threshold
}
```

### Changing Wall Configuration

Edit `feedback_config.py`:

```python
WALL_CONFIG = {
    'wall_type': 'soft',        # Use 'hard', 'soft', or 'none'
    'soft_wall_width': 100,     # Width of transition region (µm)
}
```

### Changing Initial Profile

Edit `INITIAL_PROFILE_CONFIG` in `feedback_config.py`:

```python
INITIAL_PROFILE_CONFIG = {
    'background_value': 1.0,    # Intensity outside feedback region
    'feedback_value': 0.1,      # Intensity in feedback region
    'wall_type': 'soft',
    'soft_wall_width': 50,
}
```

## Wall Types

### 1. **No Walls** (`'none'`)
- No walls applied
- Profile only has feedback and background regions

### 2. **Hard Walls** (`'hard'`)
- Immediate jump from feedback region to background value
- Good for sharp confinement
- Can cause edges in optical potential

### 3. **Soft Walls** (`'soft'`)
- Smooth cos² transition over `soft_wall_width`
- Starts at feedback region edge value
- Reaches background value at `soft_wall_width` distance
- Prevents artificial edges and heating

## Usage Workflow

1. **Configure parameters** in `feedback_config.py`

2. **Load initial profile** to set up DMD:
   ```bash
   python load_initial_profile.py
   ```

3. **Run density feedback** (automatic in lyse):
   - Will use `DENSITY_CONFIG` from config file

4. **Run magnetization feedback** (automatic in lyse):
   - Will use `MAGNETIZATION_CONFIG` from config file

## Independent Parameter Control

Since density and magnetization feedback have separate config dictionaries, you can:
- Use different feedback gains (kp) for each
- Use different feedback regions
- Use different thresholds
- Use different smoothing

But they share the same wall configuration. To use different walls for different feedback types, you would need to extend the config structure.
