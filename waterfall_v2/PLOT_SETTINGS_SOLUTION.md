## Solution: Plot Settings Persistence for Bubbles Evolution Mode

### Problem Statement
The bubbles_evolution mode dynamically creates different numbers of plots based on available data/shots. When the number of plots changes, matplotlib's automatic figure numbering shifts, causing previously saved plot settings (axis ranges, colormap limits, etc.) to be lost or applied to wrong figures.

### Root Cause
The original code used `plt.subplots()` which creates auto-numbered figures (1, 2, 3...). When the total number of plots changed, the figure numbers shifted, breaking the association between settings and specific plots.

### Solution Overview
Implemented a **named figure system with persistent settings storage**:

1. **Plot Settings Manager** (`plot_settings_manager.py`)
   - Manages plot configurations indexed by plot name instead of figure number
   - Stores settings (axis limits, etc.) in a JSON file (`.waterfall_plot_settings.json`)
   - Provides automatic autosave on user interaction

2. **Named Figures** 
   - Evolution plots now use explicit figure names:
     - `'bubbles_evolution_fluctuations'` - Local z fluctuations waterfall
     - `'bubbles_evolution_magnetization'` - Magnetization evolution waterfall
     - `'bubbles_evolution_domain_wall'` - Domain wall velocity analysis
   - Named figures persist across runs and always have the same identity

3. **Automatic Settings Application**
   - Settings are loaded and applied when plots are created
   - User modifications trigger autosave via matplotlib event callbacks
   - No manual saving required

### Key Components

#### PlotSettingsManager Class
```python
from plot_settings_manager import get_plot_manager

manager = get_plot_manager()

# Create a named figure
fig = plt.figure('my_plot_id')
manager.register_figure('my_plot_id', fig)

# Apply saved settings
manager.apply_settings('my_plot_id', ax)

# Enable autosave on user interaction
manager.enable_autosave('my_plot_id', ax)

# Manual save if needed
manager.save_settings('my_plot_id', ax)
```

#### How It Works

1. **On Plot Creation:**
   ```python
   fig = plt.figure('bubbles_evolution_fluctuations')  # Named figure
   manager.apply_settings('bubbles_evolution_fluctuations', ax)  # Load saved settings
   manager.enable_autosave('bubbles_evolution_fluctuations', ax)  # Setup autosave
   ```

2. **On User Interaction:**
   - User adjusts axis limits via GUI (zoom, pan, etc.)
   - Matplotlib event callbacks trigger
   - Settings are automatically extracted and saved to JSON

3. **On Next Run:**
   - Settings are loaded from `.waterfall_plot_settings.json`
   - Applied to the correctly-named figures
   - Settings persist regardless of how many plots are created

### Files Modified

1. **`plot_settings_manager.py`** (NEW)
   - 280+ lines of documented code
   - Handles JSON persistence and matplotlib integration
   - Supports autosave via event callbacks

2. **`waterfall_lib.py`**
   - Import added: `from waterfall_v2.plot_settings_manager import get_plot_manager`
   - `plot_evolution_analysis()` function updated to:
     - Use named figures instead of auto-numbered ones
     - Register figures with the manager
     - Apply saved settings after plot creation
     - Enable autosave for user interactions

### Settings Storage

Settings are stored in `.waterfall_plot_settings.json` in the current working directory:

```json
{
  "bubbles_evolution_fluctuations": {
    "xlim": [0.0, 0.0003],
    "ylim": [0.0, 0.002]
  },
  "bubbles_evolution_magnetization": {
    "xlim": [0.0, 0.0003],
    "ylim": [0.0, 0.002]
  },
  "bubbles_evolution_domain_wall": {
    "xlim": [900.0, 1200.0],
    "ylim": [0.5, 2.5]
  }
}
```

### Benefits

✅ **Settings Persist** - Axis ranges survive shot count changes  
✅ **Named Plots** - Plot identity is stable and meaningful  
✅ **Automatic Saving** - No manual save required  
✅ **Backward Compatible** - Uses matplotlib's existing figure naming  
✅ **Extensible** - Easy to add more settings (colormaps, line colors, etc.)  
✅ **Persistent Across Sessions** - JSON file survives program restarts  

### Usage

No changes to user workflow! The system works automatically:

1. Open bubbles_evolution analysis
2. Adjust plot axes as needed with the GUI
3. Close/reopen analysis
4. **Previously adjusted axes remain as set** ✓

### Troubleshooting

**Settings not persisting?**
- Check that `.waterfall_plot_settings.json` is created in the working directory
- Verify you're using the same working directory for runs
- Check console for any "Warning" messages from the manager

**Reset all settings:**
```python
from plot_settings_manager import get_plot_manager
manager = get_plot_manager()
manager.clear_settings()  # Clears all plots
# or
manager.clear_settings('bubbles_evolution_fluctuations')  # Clear one plot
```

**Check saved plots:**
```python
manager = get_plot_manager()
print(manager.list_saved_plots())
print(manager.get_settings('bubbles_evolution_fluctuations'))
```

### Future Enhancements

Possible extensions to save more settings:
- Colormap selection
- Line colors and styles
- Grid visibility
- Tick formatting
- Data range limits (beyond just display limits)
- Custom plot annotations
