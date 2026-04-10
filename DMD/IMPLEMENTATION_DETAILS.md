# Implementation Details - Profile Accumulation System

## Architecture Overview

The automated feedback system implements a **profile accumulation model** with three coordinated components:

```
┌─────────────────────────────────────────────────────────────┐
│  User runs: python auto_feedback_update.py                   │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    Config Import   List Loading    Sigmoid Loading
         │               │               │
    feedback_        feedback_        waterfall
    automation_      sigmoids_        sigmoid
    config.py        list.py          .txt
         │               │               │
         └───────────────┴───────────────┘
                         │
                         ▼
            ┌─────────────────────────────┐
            │  Step 1: Load Base Profile  │
            └─────────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────┐
            │  Step 2: Load New Sigmoid   │
            │  (from waterfall)           │
            └─────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────┐
    │  Step 3: Save & Register New Sigmoid      │
    │  - Auto-name: _update_N.txt               │
    │  - Add to SIGMOID_PROFILES list           │
    └───────────────────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────┐
    │  Step 4: Load ALL Sigmoids from List      │
    │  - Fetch each sigmoid from file           │
    │  - Apply individual kp and sigma          │
    │  - Accumulate: profile += kp_i * sig_i   │
    └───────────────────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────┐
    │  Step 5: Visualize                        │
    │  - Multi-sigmoid plot (all contributions) │
    │  - Old vs New DMD profile                 │
    └───────────────────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────┐
    │  Step 6-7: Update Config & Load to DMD    │
    │  - Update plot_sigmoid_txt.py config      │
    │  - Send combined profile to DMD server    │
    └───────────────────────────────────────────┘
```

## File Interactions

### 1. auto_feedback_update.py (Main Script)

**Key Functions:**

#### `load_sigmoid_list(dmd_folder) → list`
```python
# Dynamically imports feedback_sigmoids_list.py
# Returns: List of dicts with {filename, kp, smoothing_sigma, description}
# Handles missing file gracefully, returns empty list
```

#### `add_sigmoid_to_list(dmd_folder, new_filename, kp, sigma) → bool`
```python
# Modifies feedback_sigmoids_list.py using regex pattern
# Pattern: r"(SIGMOID_PROFILES\s*=\s*\[)(.*?)(\])"
# Inserts new entry before closing bracket
# Returns: True on success, False on failure
```

#### `load_dmd_profile_from_txt(filepath) → tuple(x_profile, profile_data)`
```python
# Loads DMD profile from txt file
# Handles header row auto-detection
# Falls back with skiprows=1 if "could not convert string" error
# Returns: (x_array, profile_array) or (None, None) on failure
```

#### `main(use_h5=False, kp=None, sigma=None, load_to_dmd=True, update_plot=True) → bool`
```python
# 7-step workflow:
# 1. Load base profile
# 2. Load new sigmoid from waterfall
# 3. Save with auto-increment name + add to list
# 4. Load ALL sigmoids from list and accumulate corrections
# 5. Visualize with plot_profile_comparison_multi()
# 6. Update plot config
# 7. Load to DMD server (if enabled)
```

**Command-line Interface:**
```
--kp VALUE                Set kp for new sigmoid
--sigma VALUE            Set smoothing sigma for new sigmoid
--no-server              Skip loading to DMD (preview mode)
--no-plot-update         Skip updating plot config
```

### 2. feedback_sigmoids_list.py (Persistent State)

**Structure:**
```python
SIGMOID_PROFILES = [
    {
        'filename': str,          # Name of sigmoid file
        'kp': float,              # Strength parameter for this sigmoid
        'smoothing_sigma': float, # Gaussian smoothing width
        'description': str,       # Human-readable label (optional)
    },
    # ... more profiles
]

DEFAULT_NEW_KP = float            # Default kp for new sigmoids
DEFAULT_NEW_SIGMA = float         # Default sigma for new sigmoids
```

**Key Properties:**
- Plain Python file (human-readable and editable)
- Imported as module by auto_feedback_update.py
- Modified via regex pattern matching (not parsed/AST)
- Each profile can have independent parameters

### 3. feedback_automation_config.py (Configuration)

**Key Settings:**
```python
DMD_PROFILES_FOLDER = None or str  # Base path for sigmoids
USE_H5_PROFILE = bool              # Use h5 or txt for base
PROFILE_H5_FILE = str or None      # Explicit h5 path
PROFILE_TXT_FILE = str or None     # Explicit txt path
NEW_SIGMOID_PATH = str             # Waterfall sigmoid location
LOAD_TO_DMD = bool                 # Auto-load to server (default True)

# Imported from feedback_config
WALL_TYPE = 'hard' or 'soft' or None
X_CENTER = float
FEEDBACK_WIDTH = float
SOFT_WALL_WIDTH = float
```

## Workflow Mechanics

### Profile Accumulation Algorithm

**Initial State:**
```
old_profile = load_base_profile()  # From h5 or txt
```

**For each sigmoid in SIGMOID_PROFILES:**
```
1. Load sigmoid from file: sigmoid_data = load(profile['filename'])
2. Extend to full range: extended = extend_to_dmd_range(sigmoid_data)
3. Smooth: smoothed = gaussian_filter(extended, sigma=profile['smoothing_sigma'])
4. Scale: scaled = profile['kp'] * smoothed
5. Accumulate: old_profile += scaled
```

**Result:**
```
new_profile = old_profile + Σ(kp_i * gaussian_filter(sigmoid_i, sigma_i))
              for all i in SIGMOID_PROFILES
```

### Auto-Increment Naming

**Algorithm:**
1. Scan DMD folder for files matching pattern: `sigmoid_center_interpolation_update_*.txt`
2. Extract number from each: `int(filename.replace('sigmoid_center_interpolation_update_', '').replace('.txt', ''))`
3. Find max: `max_num = max(extracted_numbers)`
4. New number: `new_num = max_num + 1`
5. New filename: `sigmoid_center_interpolation_update_{new_num}.txt`

**Examples:**
- First run: No update files → creates `sigmoid_center_interpolation_update_0.txt`
- Second run: Finds update_0 → creates `sigmoid_center_interpolation_update_1.txt`
- Third run: Finds update_0, update_1 → creates `sigmoid_center_interpolation_update_2.txt`

### List Modification via Regex

**Pattern for add_sigmoid_to_list():**
```python
# Match: SIGMOID_PROFILES = [...]
pattern = r"(SIGMOID_PROFILES\s*=\s*\[)(.*?)(\])"

# Replacement function:
def replacer(match):
    opening = match.group(1)      # "SIGMOID_PROFILES = ["
    content = match.group(2)      # existing entries
    closing = match.group(3)      # "]"
    
    new_entry = format_new_profile_dict(filename, kp, sigma)
    
    # Insert comma if content not empty
    if content.strip():
        new_content = content.rstrip() + ",\n    " + new_entry
    else:
        new_content = "\n    " + new_entry + "\n"
    
    return opening + new_content + closing

# Apply regex substitution
new_file_content = re.sub(pattern, replacer, file_content, flags=re.DOTALL)
```

### Visualization: Multi-Sigmoid Plotting

**plot_profile_comparison_multi() function:**

**Top Subplot:**
- Blue line: Old DMD profile (reference)
- Red line: New DMD profile (after all corrections)
- Green lines: Feedback region boundaries
- Title shows number of sigmoids applied

**Bottom Subplot:**
- One color per sigmoid (using matplotlib tableau colors)
- Thin dashed line: Raw extended sigmoid
- Medium line: Smoothed sigmoid (with individual sigma)
- Thick black line: Total correction (sum of all kp_i * smoothed_i)
- Zero line reference
- Legend shows each sigmoid's filename

**Features:**
- Handles variable number of sigmoids (loops through list)
- Color recycling for >10 sigmoids
- Shows which sigmoids contribute most to correction
- Individual sigma smoothing visible for each profile

## Data Flow Example

**Run 1:**
```
Input:
  - sigmoid_center_interpolation0.txt (base profile)
  - sigmoid_center_interpolation.txt (new sigmoid from waterfall)
  - config: DEFAULT_NEW_KP=0.8, DEFAULT_NEW_SIGMA=2.0

Steps:
  1. base = load(sigmoid_center_interpolation0.txt)
  2. new_sig = load(sigmoid_center_interpolation.txt)
  3. Save: sigmoid_center_interpolation_update_0.txt
  4. feedback_sigmoids_list.py:
     SIGMOID_PROFILES = [
         {filename: sigmoid_center_interpolation0.txt, kp: 1.0, sigma: 2.0},
         {filename: sigmoid_center_interpolation_update_0.txt, kp: 0.8, sigma: 2.0},  ← NEW
     ]
  5. Load both, apply:
     final = base + 1.0*smooth(base) + 0.8*smooth(update_0)
  6. Plot shows 2 sigmoids
  7. Load to DMD

Output:
  - New file: sigmoid_center_interpolation_update_0.txt
  - Updated: feedback_sigmoids_list.py (2 entries now)
  - DMD receives: combined profile with both sigmoids
```

**Run 2:**
```
Input:
  - sigmoid_center_interpolation.txt (new sigmoid from waterfall)
  - feedback_sigmoids_list.py now has 2 entries

Steps:
  1. Save: sigmoid_center_interpolation_update_1.txt
  2. feedback_sigmoids_list.py:
     SIGMOID_PROFILES = [
         {filename: sigmoid_center_interpolation0.txt, kp: 1.0, sigma: 2.0},
         {filename: sigmoid_center_interpolation_update_0.txt, kp: 0.8, sigma: 2.0},
         {filename: sigmoid_center_interpolation_update_1.txt, kp: 0.8, sigma: 2.0},  ← NEW
     ]
  3. Load all 3, apply:
     final = base + 1.0*smooth(base) + 0.8*smooth(update_0) + 0.8*smooth(update_1)
  4. Plot shows 3 sigmoids
  5. Load to DMD

Output:
  - New file: sigmoid_center_interpolation_update_1.txt
  - Updated: feedback_sigmoids_list.py (3 entries now)
  - DMD receives: combined profile with all 3 sigmoids
```

**Manual Edit Before Run 3:**
```python
# User edits feedback_sigmoids_list.py:
SIGMOID_PROFILES = [
    {filename: sigmoid_center_interpolation0.txt, kp: 0.5, sigma: 2.0},  # REDUCED
    # {filename: sigmoid_center_interpolation_update_0.txt, ...},  # COMMENTED OUT
    {filename: sigmoid_center_interpolation_update_1.txt, kp: 1.0, sigma: 1.5},  # MODIFIED
]

# Run 3:
  - Creates: sigmoid_center_interpolation_update_2.txt
  - Loads 3 sigmoids but only 2 will be used (base is always used)
  - final = base + 0.5*smooth(base) + 1.0*smooth(update_1) + 0.8*smooth(update_2)
  - DMD receives new combined profile
```

## Error Handling

**File Not Found:**
- Base profile missing → Log error and exit
- New sigmoid missing → Log warning, try alternative path
- Sigmoid in list missing → Log warning, skip that profile

**Import Failures:**
- h5py unavailable → Set h5_support=False, use txt only
- imaging_analysis_lib unavailable → Log warning, continue with defaults
- zerorpc unavailable → Log warning, skip DMD loading

**Regex Modification Failures:**
- Pattern doesn't match → Log error, don't modify file
- File write fails → Log error, don't corrupt original

**Visualization:**
- Matplotlib not available → Try fallback or skip plotting
- Empty sigmoid list → Plot shows old vs new only

## Performance Considerations

**Memory:**
- Typical DMD profile: ~1MB (10,000 points)
- Each sigmoid accumulation: In-place numpy addition (fast)
- Gaussian filtering: scipy.ndimage (optimized)
- Multiple sigmoids scale linearly: O(n_sigmoids * profile_size)

**Speed:**
- Load profile: ~10ms
- Load/extend sigmoid: ~5ms per sigmoid
- Gaussian smoothing: ~50ms per sigmoid
- Plot generation: ~500ms
- DMD communication: ~100-500ms
- **Total per run: ~1-2 seconds typical**

**Storage:**
- Each sigmoid file: ~1MB
- 10 sigmoids: ~10MB
- Minimal additional storage needed

## Testing Recommendations

1. **Unit Tests:**
   - Test load_sigmoid_list() with valid/invalid files
   - Test add_sigmoid_to_list() regex modification
   - Test load_dmd_profile_from_txt() with various formats

2. **Integration Tests:**
   - Run with single sigmoid (baseline)
   - Run with multiple sigmoids (accumulation)
   - Run with manual list edits between iterations

3. **Validation:**
   - Verify accumulated profile isn't > 2x base profile (sanity check)
   - Verify DMD receives correct profile
   - Verify list file remains valid Python after modification

## Future Enhancements

1. **Undo/Version Control:** Keep backup of previous SIGMOID_PROFILES states
2. **Parameter Optimization:** Auto-tune kp/sigma based on DMD feedback
3. **Visualization Export:** Save plots to disk for documentation
4. **Status Dashboard:** Real-time feedback about profile quality
5. **Interactive Editor:** GUI for editing SIGMOID_PROFILES
