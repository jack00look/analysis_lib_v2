# Fix: Bubbles Evolution Plot Settings Persistence

## 🎯 Problem
When bubbles_evolution mode creates different numbers of plots (based on available shots), the plot figure numbers shift, causing loss of user-made axis adjustments from the previous run.

**Example:**
- Run 1: 5 shots → Figures 1-5, user adjusts Figure 3's axes
- Run 2: 3 shots → Figures 1-3, user's settings lost (Figure 3 no longer exists)

## ✅ Solution
Implemented **named figure system with persistent JSON-based settings**.

Plots now use stable identifiers:
- `bubbles_evolution_fluctuations`
- `bubbles_evolution_magnetization`
- `bubbles_evolution_domain_wall`

Settings are automatically saved and restored regardless of total plot count.

---

## 📦 What Was Implemented

### New Files
- **`plot_settings_manager.py`** - Core settings management system
  - Persists to `.waterfall_plot_settings.json`
  - Automatic autosave on user interaction
  - 280+ lines of well-documented code

### Modified Files
- **`waterfall_lib.py`** 
  - Added import: `from waterfall_v2.plot_settings_manager import get_plot_manager`
  - Refactored `plot_evolution_analysis()` to use named figures
  - Added autosave callbacks to all three evolution plots

### Documentation
- `PLOT_SETTINGS_SOLUTION.md` - Technical deep dive
- `PLOT_SETTINGS_QUICKSTART.md` - User/dev quick reference  
- `IMPLEMENTATION_COMPLETE.md` - Comprehensive overview
- `CHANGES_SUMMARY.md` - Detailed change log

---

## 🚀 How It Works

```
User runs bubbles_evolution
    ↓
Named figures created (not auto-numbered)
    ↓
Settings loaded from .waterfall_plot_settings.json
    ↓
Settings applied to axes
    ↓
Autosave enabled on user interactions
    ↓
User adjusts axes (zoom, pan, etc.)
    ↓
Settings automatically saved to JSON
    ↓
Next run: Settings restored automatically ✓
```

---

## 🎮 Usage

**For Users:** Just use bubbles_evolution normally!
```bash
# Run analysis
python waterfall_v2_example.py  # with ACTIVE_MODE = 'bubbles_evolution'

# Adjust plot axes as needed
# Close analysis

# Run again (same/different shots)
python waterfall_v2_example.py

# Axes appear with your previous adjustments ✓
```

**For Developers:**
```python
from waterfall_v2.plot_settings_manager import get_plot_manager

manager = get_plot_manager()
print(manager.list_saved_plots())  # See all saved plots
print(manager.get_settings('plot_id'))  # Get specific settings
manager.clear_settings('plot_id')  # Reset specific plot
manager.clear_settings()  # Reset all
```

---

## 📁 Settings Storage

**Location:** `.waterfall_plot_settings.json` in current working directory

**Format:**
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

---

## ✨ Benefits

| Feature | Benefit |
|---------|---------|
| Named figures | Stable plot identity across runs |
| Automatic saving | No manual intervention needed |
| JSON persistence | Settings survive restarts |
| Shot-count agnostic | Works with any number of plots |
| Backward compatible | No breaking changes |
| Extensible | Easy to add more settings |
| Zero configuration | Works out of the box |

---

## 🔍 Testing

### Quick Test
```bash
cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2

# Set ACTIVE_MODE = 'bubbles_evolution' in waterfall_config.py
python waterfall_v2_example.py

# Adjust axis ranges on plots (zoom, pan)
# Close plots
# Run again
# Verify axes retained your adjustments ✓

# Check settings file
cat .waterfall_plot_settings.json
```

### Verify Settings File
```bash
# After first run:
ls -la .waterfall_plot_settings.json

# View contents:
cat .waterfall_plot_settings.json | python -m json.tool
```

---

## 📋 Files Modified

### waterfall_lib.py
```python
# Line ~14: Added import
from waterfall_v2.plot_settings_manager import get_plot_manager

# Lines ~1717-1878: plot_evolution_analysis() function
# Changes:
# 1. Create named figures instead of auto-numbered ones
# 2. Register figures with manager
# 3. Apply saved settings on creation
# 4. Enable autosave callbacks
```

---

## 🐛 Troubleshooting

**Q: Settings not persisting?**
- Check: `.waterfall_plot_settings.json` exists in current directory
- Check: JSON file is valid (no syntax errors)
- Check: Same working directory as original run
- Solution: Delete file and start fresh

**Q: Want to reset settings?**
```python
from waterfall_v2.plot_settings_manager import get_plot_manager
get_plot_manager().clear_settings()  # Reset all
```

**Q: Can't find `.waterfall_plot_settings.json`?**
```python
import os
print(os.getcwd())  # This is where settings file should be
```

---

## 🔄 Backward Compatibility

✅ **Fully backward compatible**
- No breaking changes
- No new dependencies  
- Works with existing code
- Automatic opt-in (no config needed)
- Can be disabled by deleting settings file

---

## 📚 Documentation

- **[PLOT_SETTINGS_SOLUTION.md](PLOT_SETTINGS_SOLUTION.md)** - Technical details
- **[PLOT_SETTINGS_QUICKSTART.md](PLOT_SETTINGS_QUICKSTART.md)** - Quick reference
- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - Full overview
- **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)** - Detailed changes

---

## ✅ Implementation Checklist

- [x] Plot settings manager created
- [x] JSON persistence implemented
- [x] Autosave callbacks connected
- [x] waterfall_lib.py updated
- [x] Named figures implemented
- [x] Settings applied on creation
- [x] Documentation complete
- [x] Code tested and verified
- [x] No syntax errors
- [x] Backward compatible

---

## 🎉 Result

**Problem:** Plot adjustments lost when shot count changes ❌  
**Solution:** Automatic persistent settings by plot name ✅

Users can now adjust bubbles_evolution plots once and have the settings stick across all future runs!
