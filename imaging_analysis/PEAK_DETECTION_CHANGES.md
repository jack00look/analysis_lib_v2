# Zero Crossing Peak Detection Implementation

## Overview
Added functionality to detect and mark peaks in the derivative signal of the z1d (magnetization profile) above a threshold. These peaks represent zero crossings in the magnetization, which are signatures of domain transitions.

## Changes Made

### 1. New Helper Function: `find_peaks_above_threshold(x, y, threshold)`
**Location:** [show_magnetization_v2.py](show_magnetization_v2.py#L630)

Finds local maxima in the derivative signal that exceed the threshold.

**Algorithm:**
- Iterates through the data
- Identifies points where: `y[i] > y[i-1]` AND `y[i] > y[i+1]` AND `y[i] > threshold`
- Returns both indices and peak values

**Returns:** `(peak_indices, peak_values)` - numpy arrays of peak positions and magnitudes

### 2. Enhanced Function: `plot_z1d_derivative()`
**Location:** [show_magnetization_v2.py](show_magnetization_v2.py#L644)

**Key Improvements:**
- **Derivative Analysis:** Computes absolute value of the smoothed derivative
- **Threshold Marking:** Draws horizontal line showing the detection threshold
- **Peak Detection:** Uses `find_peaks_above_threshold()` to identify zero crossings
- **Visualization:** 
  - Red scatter plot marks detected peaks with 'x' markers
  - Red dashed vertical lines highlight peak locations
  - Title displays total number of detected zero crossings
  - Debug output prints peak statistics (count, positions, values)

**Parameters:**
- `derivative_threshold=0.02` (default) - Controls sensitivity; adjust based on noise level

**Returns:** 
- `fig` - matplotlib figure
- `nb_zero_crossings` - count of peaks above threshold
- `peak_indices` - array of peak positions

### 3. Updated Main Function: `show_magnetization_v2()`
**Location:** [show_magnetization_v2.py](show_magnetization_v2.py#L800)

**Changes:**
- Captures return values from `plot_z1d_derivative()`
- Stores zero crossing count in results: `infos_analysis['nb_zero_crossings']`
- Result is saved to lyse database for further analysis

## Usage Example

```python
# In the analysis workflow:
fig_deriv, nb_zero_crossings, peak_indices = plot_z1d_derivative(
    Z, z1d, y_roi, title, 
    derivative_threshold=0.02  # Adjust sensitivity here
)

print(f"Detected {nb_zero_crossings} zero crossings")
print(f"Peak indices: {peak_indices}")
```

## How It Works

1. **Compute Derivative:** Gradient of smoothed z1d signal
2. **Take Absolute Value:** Both positive and negative transitions are detected
3. **Apply Threshold:** Only peaks exceeding the threshold are counted
4. **Identify Local Maxima:** Peaks must be higher than neighbors
5. **Visualize:** Plot shows all peaks with markers and vertical lines
6. **Count & Store:** Total count saved to analysis results

## Visualization Components

The derivative plot now shows:
- **Blue line:** Raw derivative of z1d
- **Red dashed line:** Absolute value of smoothed derivative
- **Orange dashed line:** Detection threshold
- **Red 'x' markers:** Detected peaks (local maxima above threshold)
- **Red vertical lines:** Positions of detected zero crossings
- **Title:** "Derivative of z1d - N Zero Crossings Detected"

## Adjusting Sensitivity

To modify detection sensitivity, adjust `derivative_threshold` parameter:
- **Lower values (e.g., 0.01):** More sensitive, may detect noise
- **Higher values (e.g., 0.05):** Less sensitive, only strong transitions
- **Default (0.02):** Balanced for typical magnetization profiles

## Output Saved to Lyse

The following result is now saved:
```python
infos_analysis['nb_zero_crossings'] = int(nb_zero_crossings)
```

This allows tracking zero crossing statistics across multiple shots in an experiment.

## Physical Interpretation

Each peak in the derivative represents a zero crossing in the magnetization signal, indicating:
- Magnetic domain boundary transitions
- Changes in magnetization sign (spin flip)
- Domain wall motion or creation

The count provides a direct measure of domain dynamics in the system.
