## Quick Start: Plot Settings Persistence

### What's Fixed
✅ Plot axis ranges now **stay the same** even when shot count changes  
✅ No more "figure 3 becomes figure 2" chaos  
✅ Your zoom/pan adjustments are automatically saved and restored  

### How It Works (Automatically)

1. **First run with bubbles_evolution:**
   - Plots display with default ranges
   - Adjust axes as needed via GUI (zoom, pan, etc.)

2. **Settings auto-save:**
   - System detects your changes
   - Settings saved to `.waterfall_plot_settings.json`
   - You'll see: `"Plot settings autosave enabled for 'bubbles_evolution_fluctuations'"`

3. **Next run:**
   - Same working directory
   - Plots load with **your saved axis ranges**
   - Different shot counts? No problem - settings stick!

### For Developers

Access the manager directly:

```python
from waterfall_v2.plot_settings_manager import get_plot_manager

manager = get_plot_manager()

# View what's saved
print(manager.list_saved_plots())
# Output: ['bubbles_evolution_fluctuations', 'bubbles_evolution_magnetization', 'bubbles_evolution_domain_wall']

# Check specific settings
print(manager.get_settings('bubbles_evolution_fluctuations'))
# Output: {'xlim': [0.0, 0.0003], 'ylim': [0.0, 0.002]}

# Clear settings for a plot
manager.clear_settings('bubbles_evolution_fluctuations')

# Clear all settings
manager.clear_settings()
```

### Settings File Location

`.waterfall_plot_settings.json` in your current working directory

Structure:
```json
{
  "plot_id_1": {"xlim": [x_min, x_max], "ylim": [y_min, y_max]},
  "plot_id_2": {"xlim": [x_min, x_max], "ylim": [y_min, y_max]},
  ...
}
```

### Plot IDs for bubbles_evolution

- `bubbles_evolution_fluctuations` - Local z fluctuations
- `bubbles_evolution_magnetization` - Magnetization evolution  
- `bubbles_evolution_domain_wall` - Domain wall velocity

### Troubleshooting

**Q: Where does `.waterfall_plot_settings.json` get saved?**  
A: Current working directory when Python is run. Check with `os.getcwd()`

**Q: Can I manually edit the JSON?**  
A: Yes, but must follow the format exactly. Backup before editing.

**Q: Settings not loading?**  
A: Verify:
  - Same working directory as original run
  - File exists: `.waterfall_plot_settings.json`
  - JSON is valid (no syntax errors)
  - Check console for warning messages

**Q: How do I reset plot ranges?**  
A: Delete `.waterfall_plot_settings.json` or use:
```python
from waterfall_v2.plot_settings_manager import get_plot_manager
get_plot_manager().clear_settings('plot_name')
```
