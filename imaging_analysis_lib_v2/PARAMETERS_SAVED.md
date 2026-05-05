# Processing Parameters Saved in show_ODs_v2

## Overview

The refactored `show_ODs_v2.py` now saves all relevant parameters used to compute n1D projections. This enables complete reproducibility - if you look at an already-analyzed image, you know exactly how to obtain the same n1D values.

## Parameters Saved

For each camera and each image, the following are stored:

### 1. **Camera Configuration Parameters**
These define HOW the image is processed:

```
<camera_name>_param_roi
    NumPy slice defining the region extracted from raw images
    Example: "np.s_[:, :]" (full image)
    
<camera_name>_param_roi_back
    Background reference region for normalization
    Example: "np.s_[110:, -300:]" (bottom-right corner)
    
<camera_name>_param_roi_integration
    Region used for 1D projections
    Example: "np.s_[50:85, :]" (middle rows only)
    
<camera_name>_param_alpha
    Absorption imaging correction factor
    Example: 2.33 (unitless)
    
<camera_name>_param_chi_sat
    Saturation parameter of atomic cross-section
    Example: 4.7e7 (1/intensity)
    
<camera_name>_param_pulse_time
    Duration of imaging light pulse
    Example: 5e-6 (seconds, 5 microseconds)
    
<camera_name>_param_px_size
    Physical size of camera pixel
    Example: 1.019e-6 (meters, ~1 micrometer)
    
<camera_name>_param_method
    Imaging method used
    Example: 'absorption' or 'phase_contrast'
    
<camera_name>_param_background_std_multiplier
    Threshold for background detection
    Used as: back_max = mean(background) + this_value * std(background)
    Default: 3.0
    
<camera_name>_param_saturation_threshold_multiplier
    Threshold for saturation detection
    Used as: sat_min = dynamic_range * this_value
    Default: 0.95
    
<camera_name>_param_atomic_cross_section_sigma
    Rb-87 atomic cross-section (physical constant)
    Value: 1.656425e-13 (m²)
```

### 2. **Image Metadata** (For each image key)
Information about image quality and processing:

```
<image_key>_OD_log_bare_sum
    Sum of raw logarithmic OD (before alpha correction)
    
<image_key>_OD_lin_sum
    Sum of linear OD contribution
    
<image_key>_OD_log_sum
    Sum of corrected logarithmic OD (after alpha)
    
<image_key>_probe_norm_err
    Normalization error metric (quality indicator)
    Lower is better. Used to assess background ROI quality.
    
<image_key>_cnt_rel_probe_back
    Fraction of probe pixels in background region
    (pixels below back_max threshold)
    
<image_key>_cnt_rel_atoms_back
    Fraction of atom image pixels in background
    (pixels below back_max threshold)
    
<image_key>_cnt_rel_probe_sat
    Fraction of saturated probe pixels
    (pixels above saturation threshold)
    
<image_key>_cnt_rel_atoms_sat
    Fraction of saturated atom image pixels
    (pixels above saturation threshold)
```

### 3. **1D Projection Data** (For each image key)
The actual computed results:

```
<image_key>_n1D_x
    1D projection along x-axis
    Shape: (N,) where N = image width
    Units: atoms/m (column density)
    
<image_key>_n1D_y
    1D projection along y-axis
    Shape: (M,) where M = image height
    Units: atoms/m (column density)
    
<image_key>_com_x
    Center of mass in x direction
    Units: pixels (relative to image array)
    
<image_key>_com_y
    Center of mass in y direction
    Units: pixels (relative to image array)
```

### 4. **SVD-Specific Parameters** (If SVD probes are used)
Same as above but with prefix:
```
<camera_name>_svd_param_roi
<camera_name>_svd_param_roi_back
<camera_name>_svd_param_roi_integration
... etc
```

## Example: Recreating n1D from Saved Parameters

If you have an already-analyzed image with all these parameters saved, you can recreate the exact same n1D by:

```python
import h5py
import numpy as np
from imaging_analysis_lib_v2.main import calculate_OD, get_n1Ds

# Load parameters from lyse results
lyse_path = "path/to/experiment/results.h5"
with h5py.File(lyse_path, 'r') as f:
    # Retrieve saved parameters
    roi = np.s_[:, :]  # Parse from saved string
    roi_back = np.s_[110:, -300:]
    roi_integration = np.s_[50:85, :]
    alpha = 2.33
    chi_sat = 4.7e7
    pulse_time = 5e-6
    px_size = 1.019e-6
    
    # Create camera config dict
    cam = {
        'roi': roi,
        'roi_back': roi_back,
        'roi_integration': roi_integration,
        'alpha': alpha,
        'chi_sat': chi_sat,
        'pulse_time': pulse_time,
        'px_size': px_size,
        'method': 'absorption',
        'dynamic_range': 10000,
        'h5group': 'data/cam_vert1/images',
        'name': 'vert1',
        'atoms_images': ['PTAI_0'],
        'probe_image': 'PTAI_probe',
        'background_image': 'PTAI_back',
        'roi_integration': roi_integration,
    }
    
    # Load raw images
    atoms = f['data/cam_vert1/images/PTAI_0'][()]
    probe = f['data/cam_vert1/images/PTAI_probe'][()]
    back = f['data/cam_vert1/images/PTAI_back'][()]
    
    # Recompute exactly the same way
    n_2D, metadata, OD_log_bare = calculate_OD(atoms, probe, cam, back)
    n1D_x, n1D_y = get_n1Ds(n_2D[roi_integration], cam)
    
    # These should match the saved n1D_x and n1D_y exactly!
```

## Comparison: Old vs New

### Old show_ODs.py
```python
# Saves only:
key_n1D_x              # 1D projections
key_n1D_y
key_com_x              # Center of mass
key_com_y
key_OD_log_bare_sum    # Metadata
key_OD_lin_sum
... (metadata fields)

# No way to recreate analysis!
# If you want to recompute n1D with different roi_integration,
# you don't know what roi_integration was originally used.
```

### New show_ODs_v2.py
```python
# Saves all of above PLUS:
vert1_param_roi              # All input parameters
vert1_param_roi_back
vert1_param_roi_integration  # Critical! You know what ROI was used
vert1_param_alpha
vert1_param_chi_sat
vert1_param_pulse_time
vert1_param_px_size
vert1_param_method
vert1_param_background_std_multiplier
vert1_param_saturation_threshold_multiplier
vert1_param_atomic_cross_section_sigma

# Complete reproducibility! You can now:
# 1. Load the raw image
# 2. Read the saved parameters
# 3. Recompute n1D exactly the same way
# 4. Modify parameters and compare results
```

## Removing Flags (As Requested)

The original code saved quality flags like:
```
flag_atoms_sat          # Was image saturated?
flag_probe_sat
```

These are **NOT** saved in v2. Instead:
- You have the raw percentages: `cnt_rel_atoms_sat`, `cnt_rel_probe_sat`
- You can set your own thresholds for what constitutes "too much saturation"
- More flexible than hardcoded flags

To detect saturation issues in v2:
```python
if cnt_rel_atoms_sat > 0.001:  # More than 0.1% saturated
    print("WARNING: Image is saturated!")
```

## Notes on Parameter Strings

Due to HDF5 limitations, complex objects like slices are saved as strings:
```
'roi_back': 'np.s_[110:, -300:]'  # Stored as string, not evaluable
```

When recreating, you need to parse this manually or use:
```python
# Safe parsing
import ast
roi_str = 'np.s_[110:, -300:]'
# Note: this is for documentation only
# In practice, you'd keep a reference to the original camera_settings.py

# Or better: always have camera_settings.py as your source of truth
from camera_settings import cameras
cam = cameras['cam_vert1']
```

## Summary

The new system provides **complete traceability**:
✅ Know exactly what ROI was used  
✅ Know what alpha/chi_sat were used  
✅ Know detection thresholds  
✅ Can recreate analysis exactly  
✅ Can modify parameters and compare  
✅ No black-box analysis results  

This is especially important for:
- Debugging analysis issues
- Comparing results with different settings
- Publishing results (full reproducibility)
- Long-term data archival
