# AUTOMATED DMD FEEDBACK SYSTEM - Master Index

**Location:** `/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD/`

**Status:** ✓ COMPLETE AND READY FOR USE

## 📋 Quick Navigation

### 🚀 START HERE
- **README_FEEDBACK_AUTOMATION.md** (4.7 KB) - Overview and quick summary
- **QUICK_START.md** (5.1 KB) - Get running in 5 minutes

### 📖 Full Documentation  
- **AUTOMATED_FEEDBACK_WORKFLOW.md** (8.3 KB) - Complete workflow guide with usage examples
- **IMPLEMENTATION_DETAILS.md** (15 KB) - Technical architecture and algorithms
- **SYSTEM_ARCHITECTURE.md** (22 KB) - Visual diagrams and data flow

### 💻 Core Script Files

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| **auto_feedback_update.py** | 788 | 32 KB | Main automation script (REWRITTEN) |
| **feedback_sigmoids_list.py** | 48 | 1.8 KB | Persistent profile registry (CREATED) |
| **feedback_automation_config.py** | 112 | 4.2 KB | Central configuration (READY) |

## 🎯 What This System Does

**Profile Accumulation Automation for DMD Feedback**

Each time you run:
```bash
python auto_feedback_update.py
```

The system automatically:
1. Loads your base DMD profile
2. Loads new sigmoid from waterfall analysis
3. Saves the sigmoid with auto-incrementing name
4. Adds it to a persistent SIGMOID_PROFILES registry
5. Loads ALL sigmoids from the registry
6. Combines all corrections mathematically
7. Visualizes all sigmoids contributing
8. Loads the combined profile to your DMD server

## 📁 File Organization

### Core Components (What to Use)
```
auto_feedback_update.py              ← Run this command
feedback_automation_config.py         ← Edit for paths/settings
feedback_sigmoids_list.py             ← Edit to tune parameters
```

### Sigmoid Files (Auto-managed)
```
sigmoid_center_interpolation0.txt     ← Base profile (initial)
sigmoid_center_interpolation_update_0.txt  ← First feedback
sigmoid_center_interpolation_update_1.txt  ← Second feedback
sigmoid_center_interpolation_update_2.txt  ← Third feedback
... (accumulates over time)
```

### Documentation (Reference)
```
README_FEEDBACK_AUTOMATION.md         ← Start here
QUICK_START.md                        ← Quick reference
AUTOMATED_FEEDBACK_WORKFLOW.md        ← Workflow guide
IMPLEMENTATION_DETAILS.md             ← Technical details
SYSTEM_ARCHITECTURE.md                ← Diagrams & data flow
IMPLEMENTATION_COMPLETE.md            ← Completion summary
```

## 🔑 Key Features

✅ **Profile Accumulation** - Each run adds to list (not replaces)
✅ **Persistent State** - SIGMOID_PROFILES survives between runs
✅ **Independent Parameters** - Each sigmoid has own kp and sigma
✅ **Manual Control** - Edit feedback_sigmoids_list.py to tune
✅ **Auto-Numbering** - Sigmoids saved as update_0.txt, update_1.txt, etc.
✅ **Multi-Sigmoid Visualization** - See all sigmoids and their contributions
✅ **Automatic Registration** - New sigmoids auto-added to persistent list
✅ **DMD Integration** - Auto-loads combined profile to server
✅ **Error Handling** - Graceful failures and optional h5 support
✅ **Flexible CLI** - Options for custom parameters and preview mode

## 📊 Mathematical Model

```
new_profile = old_profile + Σ(kp_i × gaussian_filter(sigmoid_i, sigma_i))

Where:
  • Each sigmoid has independent kp (strength) and sigma (smoothing)
  • All sigmoids summed together
  • Results fully reproducible
  • Manual tuning possible between runs
```

## 🎬 Typical Usage Pattern

### First Run
```bash
cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD
python auto_feedback_update.py
```
Creates: `sigmoid_center_interpolation_update_0.txt`

### Second Run
```bash
python auto_feedback_update.py
```
Creates: `sigmoid_center_interpolation_update_1.txt`
Now 2 sigmoids combined!

### With Custom Parameters
```bash
python auto_feedback_update.py --kp 0.9 --sigma 1.5
```

### Preview Without DMD Loading
```bash
python auto_feedback_update.py --no-server
```

### Manual Tuning Between Runs
Edit `feedback_sigmoids_list.py`:
- Adjust `kp` values (0.5 to 1.5 typical)
- Adjust `smoothing_sigma` values
- Comment out profiles to disable
- Change descriptions for tracking

## 📝 Documentation Guide

### For Getting Started
1. Read: **README_FEEDBACK_AUTOMATION.md** (2 min)
2. Read: **QUICK_START.md** (5 min)
3. Run: `python auto_feedback_update.py`

### For Understanding the Workflow
1. Read: **AUTOMATED_FEEDBACK_WORKFLOW.md** (15 min)
2. Review examples in **QUICK_START.md**
3. Try manual editing examples

### For Technical Details
1. Read: **IMPLEMENTATION_DETAILS.md** (30 min)
2. Study: **SYSTEM_ARCHITECTURE.md** diagrams (20 min)
3. Reference: Function docstrings in auto_feedback_update.py

### For Troubleshooting
1. Check: **QUICK_START.md** → "Troubleshooting" section
2. Review: **AUTOMATED_FEEDBACK_WORKFLOW.md** → "Troubleshooting"
3. Verify: Settings in feedback_automation_config.py

## 🛠️ Configuration

### Main Settings (feedback_automation_config.py)

```python
DMD_PROFILES_FOLDER = None              # Auto-detected if None
USE_H5_PROFILE = False                  # Use txt instead of h5
NEW_SIGMOID_PATH = "..."                # Waterfall output path
LOAD_TO_DMD = True                      # Auto-load (default)
WALL_TYPE = 'soft'                      # Feedback wall type
```

### Parameter Defaults (feedback_sigmoids_list.py)

```python
DEFAULT_NEW_KP = 0.8                    # Strength for new sigmoids
DEFAULT_NEW_SIGMA = 2.0                 # Smoothing for new sigmoids
```

## ✅ Verification Checklist

**All components verified:**
- ✓ auto_feedback_update.py - 788 lines, all functions defined
- ✓ feedback_sigmoids_list.py - Persistent registry structure valid
- ✓ feedback_automation_config.py - All settings importable
- ✓ plot_profile_comparison_multi() - Multi-sigmoid visualization implemented
- ✓ Accumulation algorithm - Mathematically correct
- ✓ Auto-increment naming - Logic verified
- ✓ Error handling - Graceful failures
- ✓ DMD integration - Ready to use
- ✓ Documentation - Complete (6 guides)

## 🎓 System Architecture Summary

```
User runs: python auto_feedback_update.py
                    ↓
          [Read: config + persistent list]
                    ↓
    [1. Load base  2. Load new  3. Save new]
                    ↓
    [4. Load ALL from list  5. Combine corrections]
                    ↓
        [6. Visualize  7. Update config]
                    ↓
    [8. Load to DMD  9. Update persistent list]
                    ↓
              [Repeat next iteration]
```

## 📞 Support

**Questions or issues?**
1. Check **QUICK_START.md** troubleshooting
2. Review **AUTOMATED_FEEDBACK_WORKFLOW.md** examples
3. Study **SYSTEM_ARCHITECTURE.md** diagrams
4. Reference function docstrings in code

## 🎉 Summary

You now have:
- ✓ Fully automated feedback workflow
- ✓ Profile accumulation system
- ✓ Manual parameter tuning capability
- ✓ Multi-sigmoid visualization
- ✓ Persistent state management
- ✓ Comprehensive documentation
- ✓ Complete error handling
- ✓ Production-ready code

**Ready to use immediately!**

---

**Last Updated:** March 31, 2024
**Version:** 1.0 - Profile Accumulation Model
**Status:** ✓ COMPLETE AND TESTED
