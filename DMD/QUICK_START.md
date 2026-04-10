# Quick Start Guide - Automated DMD Feedback

## The System
You now have a **persistent profile accumulation system** that automatically manages your DMD feedback workflow.

## Three Key Files

| File | Purpose |
|------|---------|
| `auto_feedback_update.py` | Main automation script - run this each iteration |
| `feedback_automation_config.py` | Settings (base profile path, new sigmoid path, etc.) |
| `feedback_sigmoids_list.py` | **Persistent list of all sigmoids used** - edit manually to adjust |

## Standard Workflow (Each Iteration)

1. **Run ARP_Backward scan** - generates new data
2. **Analyze in waterfall** - produces sigmoid_center_interpolation.txt
3. **Run automation** - one command:
   ```bash
   cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD
   python auto_feedback_update.py
   ```
4. **Check visualization** - plots show all sigmoids and combined correction
5. **DMD automatically updated** - profile loaded to server
6. *Optional:* **Adjust parameters** - edit feedback_sigmoids_list.py for next iteration

## Command Options

```bash
# Standard run (uses defaults from config)
python auto_feedback_update.py

# Custom parameters
python auto_feedback_update.py --kp 0.9 --sigma 1.5

# Preview without loading to DMD
python auto_feedback_update.py --no-server

# Skip plot config update
python auto_feedback_update.py --no-plot-update
```

## What Happens Automatically

Each run:
1. ✓ Saves new sigmoid with auto-incrementing name (update_0, update_1, ...)
2. ✓ Adds to `SIGMOID_PROFILES` list
3. ✓ Loads ALL sigmoids from list
4. ✓ Applies each with its own kp and sigma
5. ✓ Combines all corrections
6. ✓ Visualizes results
7. ✓ Updates plot config
8. ✓ Loads to DMD server

## Manual Profile Tuning

Edit `feedback_sigmoids_list.py` to adjust between runs:

```python
SIGMOID_PROFILES = [
    {
        'filename': 'sigmoid_center_interpolation0.txt',
        'kp': 0.8,              # ← Adjust strength (try 0.5 to 1.5)
        'smoothing_sigma': 2.0, # ← Adjust smoothing (lower = sharper)
        'description': 'Initial profile',
    },
]
```

## How the Accumulation Works

**Run 1:** Base profile + sigmoid_0 with kp₀ and σ₀
**Run 2:** Base profile + sigmoid_0 + sigmoid_1 (each with own kp, σ)
**Run 3:** Base profile + sigmoid_0 + sigmoid_1 + sigmoid_2
... etc

So corrections **accumulate** rather than replace each other.

## File Organization

```
DMD/
├── auto_feedback_update.py              ← Run this
├── feedback_automation_config.py         ← Edit for paths/settings
├── feedback_sigmoids_list.py             ← Edit to tune profiles
│
├── sigmoid_center_interpolation0.txt     ← Base profile (initial)
├── sigmoid_center_interpolation_update_0.txt  ← First feedback result
├── sigmoid_center_interpolation_update_1.txt  ← Second feedback result
└── sigmoid_center_interpolation_update_2.txt  ← etc.
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No sigmoids found" | Add a sigmoid filename to SIGMOID_PROFILES |
| "Could not load sigmoid" | Verify waterfall created sigmoid_center_interpolation.txt |
| DMD not updated | Check LOAD_TO_DMD = True in config, verify server is running |
| Script crashes | Run with `--no-server` to skip DMD loading and debug |

## Examples

### Example 1: Adjust kp for next iteration
```python
# Edit feedback_sigmoids_list.py
SIGMOID_PROFILES = [
    {
        'filename': 'sigmoid_center_interpolation0.txt',
        'kp': 0.5,  # Changed from 1.0 (weaker)
        'smoothing_sigma': 2.0,
        'description': 'Initial profile - reduced strength',
    },
]
# Run again - will apply with new kp value
```

### Example 2: Make feedback smoother
```python
# Edit feedback_sigmoids_list.py
SIGMOID_PROFILES = [
    {
        'filename': 'sigmoid_center_interpolation0.txt',
        'kp': 1.0,
        'smoothing_sigma': 5.0,  # Changed from 2.0 (much smoother)
        'description': 'Initial profile - smoother',
    },
]
```

### Example 3: Remove a problematic profile
```python
# Comment out the problematic line
SIGMOID_PROFILES = [
    # {
    #     'filename': 'sigmoid_center_interpolation_update_0.txt',
    #     'kp': 0.9,
    #     'smoothing_sigma': 1.5,
    #     'description': 'Removed - caused oscillations',
    # },
    {
        'filename': 'sigmoid_center_interpolation_update_1.txt',
        'kp': 1.0,
        'smoothing_sigma': 2.0,
        'description': 'Working well',
    },
]
```

## Integration with feedback_manual_update.py

If you need to add multiple sigmoids manually at once:
1. Use `feedback_manual_update.py` as before
2. But with automation, the list-based approach means both work together
3. All profiles contribute to the final DMD profile

## Questions?

- **Where do sigmoids go?** In the DMD folder (auto-saved there)
- **Can I remove a profile?** Yes, comment it out in feedback_sigmoids_list.py
- **Can I add profiles manually?** Yes, just add the line to SIGMOID_PROFILES
- **What if I want different kp for different runs?** Edit feedback_sigmoids_list.py before each run
- **Does the old feedback_manual_update.py still work?** Yes, both can coexist
