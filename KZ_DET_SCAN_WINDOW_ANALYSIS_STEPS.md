# KZ_det_scan_window Analysis: Step-by-Step Explanation

## Overview
The KZ_det_scan_window analysis is a sliding window approach to analyze magnetization and domain structure as a function of spatial position and applied field.

## Step 1: Domain Extraction for All Shots
**What it does:**
- Takes each shot's magnetization profile (1D array of magnetization values)
- Detects domain boundaries by finding where magnetization changes sign (crosses zero)
- Assigns a sign to each domain (+1 for positive, -1 for negative magnetization)
- Records the spatial extent (x_start, x_end) of each domain
- Counts total number of domains in each shot

**Why it matters:**
- Validates that domain detection is working correctly before windowed analysis
- Each domain represents a region where spins are polarized in the same direction

**Output:**
- `domain_info_dict`: Dictionary indexed by shot number containing:
  - `domains`: List of detected domains with boundaries and signs
  - `n_domains`: Total number of domains
  - `field`: Applied field value for that shot
  - Visualization: `domains_validation.png` showing all shots with domains overlaid

## Step 2: Sliding Window Analysis
**What it does across spatial windows:**

For each window position (from x_min=920 μm to x_max=1180 μm, in steps of 5 μm):

### 2a: Window Data Collection
- Extracts the magnetization profile segment in that window for each shot
- Computes average magnetization in the window
- Counts defects (domain walls) in that window region
- Counts number of domains in that window
- Groups all this data by field value

### 2b: Linear Fit to Find Zero Crossing
- Selects only field points where |avg_magnetization| ≤ domain_balance_fit_window (0.4)
- Fits a line through these points: mag_mean = slope × field + intercept
- Solves: field_at_zero = -intercept / slope (where magnetization crosses zero)
- This is more robust than just finding the closest point

### 2c: Interpolate Defect Density at Zero Crossing
- Finds the two bracketing field points (just above and just below the zero-crossing field)
- Linearly interpolates defect density between these two points
- Stores both the value and uncertainty (standard error of mean)

**Why it matters:**
- Shows how magnetization and defect count vary with position
- The field_at_zero tells you where the domain balance is zero (transition between + and - magnetized regions)
- Spatial variation in field_at_zero indicates spatial structure in the sample

**Output:**
- `window_analysis.json`: Detailed results for each window including:
  - `window_center_um`: Center position of window
  - `field_at_zero`: Field where magnetization crosses zero (from linear fit)
  - `domain_density_at_zero`: Interpolated defect density at zero field
  - `domain_density_error_at_zero`: Error on that density
  - `fit_r_squared`: Quality of the linear fit
- `window_analysis.csv`: Summary table format

## Step 3: Visualization

### 3a: Twin-Axis Plot
- Left y-axis: Defect density vs window position
- Right y-axis: Field at zero magnetization vs window position
- Shows spatial variation of both quantities

### 3b: Detailed 3-Panel Plot
- Panel 1: Defect density with error bars
- Panel 2: Field at zero magnetization
- Panel 3: Domain count at zero field

### 3c: Raw Magnetization with Domains
- All shots' profiles ordered by field value
- Colored by field (red = low, blue = high)
- Green bands = positive domains
- Red bands = negative domains
- Allows visual validation of domain extraction quality

## Key Parameters (in common_config.py and KZ_det_scan_window.py)

```python
# Integration region (spatial limits)
'X_MIN_INTEGRATION': 920,     # Left boundary [μm]
'X_MAX_INTEGRATION': 1180,    # Right boundary [μm]

# Windowing parameters
'window_size': 80.0,          # Width of each analysis window [μm]
'window_step': 5.0,           # Distance between window centers [μm]

# Fitting parameter
'domain_balance_fit_window': 0.4  # |magnetization| threshold for linear fit points
```

## Expected Results

### Domain Validation Plot
- Should show clear domain boundaries as colored bands
- Green and red bands should align with positive/negative magnetization regions
- Domain count should vary with field (fewer domains at extreme fields, more at intermediate)

### Window Analysis Results
- Defect density should show spatial variation (could be higher/lower in certain regions)
- Field at zero should indicate position of domain balance transition
- If sample has spatial structure, field_at_zero will vary across windows

### Raw Magnetization Plot
- Organized by field value (low to high)
- Domains should be clearly visible and consistent across similar fields
- Shows experimental scatter (shot-to-shot variation)
