# All-Shots Waterfall - Quick Start Guide

## Enable the Feature

Edit: `waterfall_v2/mode_configs/bubbles_evolution.py`

```python
'plots': {
    # ... other settings ...
    'all_shots_waterfall': True,  # Change False to True
}
```

## What You'll See

### Plot 1: All Shots Waterfall
- **Each row**: One individual shot
- **Color**: Magnetization value (red = +1, blue = -1)
- **Green circles**: Left domain wall for each shot
- **Violet squares**: Right domain wall for each shot
- **Lime lines**: Integration region boundaries
- **White separators**: Groups by bubbles_time (or other scan variable)
- **Labels**: Scan variable values on left and right sides

### Plot 2: Domain Wall Evolution
- **X-axis**: Scan variable (bubbles_time, etc.)
- **Y-axis**: Position in micrometers
- **Scatter dots**: Individual measurements per shot
- **Error bars**: Mean ± Standard Error
- **Green**: Left domain walls
- **Violet**: Right domain walls

## How to Use

1. **Set ACTIVE_MODE** to `'bubbles_evolution'` in waterfall_config.py
2. **Set SEQS** to your desired sequences
3. **Enable all_shots_waterfall** in mode config (set to True)
4. **Run analysis**: `python waterfall_v2_example.py`
5. **View plots**: Two new figures will appear

## Customize

### Disable Again
Simply set `'all_shots_waterfall': False` to turn off (default)

### Adjust Integration Region
In `waterfall_config.py` PARAMS:
```python
'X_MIN_INTEGRATION': 900,   # Adjust left boundary (micrometers)
'X_MAX_INTEGRATION': 1200,  # Adjust right boundary (micrometers)
```

### Change Colormap/Display
Edit the `plot_all_shots_waterfall()` function in waterfall_lib.py

## Features

✅ Shows all individual shots (not averaged)
✅ Automatic domain wall detection per shot
✅ Evolution plot shows trends
✅ Separators group shots by scan variable
✅ Plot settings auto-saved (axis ranges persist)
✅ Works with other bubbles_evolution plots

## Keyboard Shortcuts in Plot

Standard matplotlib:
- **Mouse scroll**: Zoom in/out
- **Left click + drag**: Pan
- **Right click**: Reset zoom
- **S**: Save figure

## Troubleshooting

**Plots not appearing?**
- Check `all_shots_waterfall: True` in config
- Check bubbles_evolution has data loaded
- Verify SEQS point to valid data

**Domain walls not visible?**
- May not detect walls if magnetization is uniform
- Try adjusting integration region
- Check PARAMS['X_MIN_INTEGRATION'] and ['X_MAX_INTEGRATION']

**Want to reset plot settings?**
Delete: `.waterfall_plot_settings.json`

## See Also

- [ALL_SHOTS_WATERFALL_FEATURE.md](ALL_SHOTS_WATERFALL_FEATURE.md) - Full technical details
- [PLOT_SETTINGS_SOLUTION.md](PLOT_SETTINGS_SOLUTION.md) - Settings persistence system
- [README_PLOT_SETTINGS.md](README_PLOT_SETTINGS.md) - General plot management
