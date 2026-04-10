# System Architecture & Data Flow Diagrams

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AUTOMATED FEEDBACK SYSTEM                         │
│                      (Profile Accumulation Model)                        │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │   Waterfall Analysis Output      │
                    │ sigmoid_center_interpolation.txt │
                    └────────────────┬─────────────────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────┐
         │   auto_feedback_update.py (MAIN SCRIPT)          │
         │                                                   │
         │  import feedback_automation_config              │
         │  import feedback_sigmoids_list                  │
         │                                                   │
         │  7-Step Accumulation Workflow:                   │
         │  1. Load base profile                            │
         │  2. Load new sigmoid from waterfall              │
         │  3. Save + auto-register new sigmoid             │
         │  4. Load ALL sigmoids from list                  │
         │  5. Combine corrections (accumulate)             │
         │  6. Visualize with multi-sigmoid plots           │
         │  7. Load to DMD server                           │
         └───┬───────────────────────────────────────────┬──┘
             │                                           │
    ┌────────▼─────────────┐                   ┌────────▼─────────────┐
    │ feedback_            │                   │ feedback_            │
    │ sigmoid_             │                   │ automation_          │
    │ list.py              │                   │ config.py            │
    │                      │                   │                      │
    │ Persistent Registry: │                   │ Settings:            │
    │ - Profile list       │                   │ - Base profile path  │
    │ - kp for each        │                   │ - New sigmoid path   │
    │ - sigma for each     │                   │ - DMD server config  │
    │ - Manual editable    │                   │ - Wall settings      │
    └──────────────────────┘                   └──────────────────────┘
             │
             │ (updated each run)
             │
    ┌────────▼──────────────────────────────────────────────────────┐
    │ DMD Profile Output & Visualization                            │
    │                                                               │
    │ ├─ Combined DMD profile (sent to hardware)                   │
    │ ├─ Multi-sigmoid visualization (matplotlib)                  │
    │ └─ Updated plot_sigmoid_txt.py config                        │
    └────────────────────────────────────────────────────────────────┘
```

## Data Flow - Single Iteration

```
RUN 1: python auto_feedback_update.py
═════════════════════════════════════════════════════════════════

INPUT:
  sigmoid_center_interpolation.txt  (from waterfall)
  sigmoid_center_interpolation0.txt (base profile, in DMD folder)
  feedback_automation_config.py     (settings: kp, sigma, etc.)
  feedback_sigmoids_list.py         (current: only base profile)

PROCESSING:
  Step 1: Load base profile
    sigmoid_center_interpolation0.txt → [x_vals, profile_vals]

  Step 2: Load new sigmoid
    sigmoid_center_interpolation.txt → [x_sigmoid, sigmoid_vals]

  Step 3: Save & register
    Save as: sigmoid_center_interpolation_update_0.txt
    Add to: feedback_sigmoids_list.py:
      {filename: 'sigmoid_center_interpolation_update_0.txt',
       kp: 0.8, sigma: 2.0}

  Step 4-5: Load all & accumulate
    SIGMOID_PROFILES = [
      {filename: 'sigmoid_center_interpolation0.txt', kp: 1.0, σ: 2.0},
      {filename: 'sigmoid_center_interpolation_update_0.txt', kp: 0.8, σ: 2.0},
    ]

    new_profile = base
    for each profile in list:
      extended = extend_to_dmd_range(load(filename))
      smoothed = gaussian_filter(extended, sigma)
      scaled = kp * smoothed
      new_profile += scaled

  Step 6: Visualize
    Plot 1 (top): old_profile (blue) vs new_profile (red)
    Plot 2 (bot): All 2 sigmoids + total correction

  Step 7: Load to DMD

OUTPUT:
  ├─ sigmoid_center_interpolation_update_0.txt (saved)
  ├─ feedback_sigmoids_list.py (updated with new entry)
  ├─ DMD receives new_profile
  ├─ matplotlib shows visualization
  └─ plot_sigmoid_txt.py config updated


RUN 2: python auto_feedback_update.py
═════════════════════════════════════════════════════════════════

INPUT:
  sigmoid_center_interpolation.txt  (NEW from waterfall)
  sigmoid_center_interpolation0.txt (base profile)
  feedback_sigmoids_list.py         (NOW has 2 entries!)

PROCESSING:
  Step 3: Save & register
    Save as: sigmoid_center_interpolation_update_1.txt  ← Increment!
    Add to: feedback_sigmoids_list.py:
      {filename: 'sigmoid_center_interpolation_update_1.txt',
       kp: 0.8, sigma: 2.0}

  Step 4-5: Load all & accumulate
    SIGMOID_PROFILES = [
      {filename: 'sigmoid_center_interpolation0.txt', kp: 1.0, σ: 2.0},
      {filename: 'sigmoid_center_interpolation_update_0.txt', kp: 0.8, σ: 2.0},
      {filename: 'sigmoid_center_interpolation_update_1.txt', kp: 0.8, σ: 2.0},  ← NEW
    ]

    new_profile accumulates from ALL 3 profiles now!

  Step 6: Visualize
    Plot 1 (top): old_profile (blue) vs new_profile (red)
    Plot 2 (bot): All 3 sigmoids + total correction

OUTPUT:
  ├─ sigmoid_center_interpolation_update_1.txt (saved)
  ├─ feedback_sigmoids_list.py (NOW has 3 entries)
  ├─ DMD receives combined_profile (from all 3 sigmoids)
  ├─ matplotlib shows 3 sigmoids in visualization
  └─ plot_sigmoid_txt.py config updated


RUN 3 (with manual editing): Edit feedback_sigmoids_list.py
═════════════════════════════════════════════════════════════════

BEFORE RUN 3 - User edits file:
  SIGMOID_PROFILES = [
    {filename: 'sigmoid_center_interpolation0.txt', kp: 0.5, σ: 2.0},     # ← REDUCED
    # {filename: 'sigmoid_center_interpolation_update_0.txt', ...},         # ← COMMENTED OUT
    {filename: 'sigmoid_center_interpolation_update_1.txt', kp: 1.0, σ: 1.5},  # ← MODIFIED
  ]

PROCESSING:
  Step 3: Save & register
    Save as: sigmoid_center_interpolation_update_2.txt
    Add new entry (update_1 already in list)

  Step 4-5: Load all & accumulate
    new_profile accumulates from 3 profiles:
      - base (kp=0.5, was 1.0) ← WEAKER
      - update_1 (kp=1.0, σ=1.5) ← SHARPER
      - update_2 (kp=0.8, σ=2.0) ← NEW
    NOTE: update_0 is NOT applied (commented out)

OUTPUT:
  ├─ sigmoid_center_interpolation_update_2.txt (saved)
  ├─ DMD receives combined_profile with 3 sigmoids (different params than before!)
  └─ Manual editing worked! Parameters took effect
```

## Persistent List Evolution

```
INITIAL STATE (feedback_sigmoids_list.py):
────────────────────────────────────────────────────────────

SIGMOID_PROFILES = [
    {
        'filename': 'sigmoid_center_interpolation0.txt',
        'kp': 1.0,
        'smoothing_sigma': 2.0,
        'description': 'Initial profile - baseline',
    },
]


AFTER RUN 1:
────────────────────────────────────────────────────────────

SIGMOID_PROFILES = [
    {
        'filename': 'sigmoid_center_interpolation0.txt',
        'kp': 1.0,
        'smoothing_sigma': 2.0,
        'description': 'Initial profile - baseline',
    },
    {
        'filename': 'sigmoid_center_interpolation_update_0.txt',  ← NEW
        'kp': 0.8,
        'smoothing_sigma': 2.0,
        'description': 'Feedback iteration 1',
    },
]


AFTER RUN 2:
────────────────────────────────────────────────────────────

SIGMOID_PROFILES = [
    {
        'filename': 'sigmoid_center_interpolation0.txt',
        'kp': 1.0,
        'smoothing_sigma': 2.0,
        'description': 'Initial profile - baseline',
    },
    {
        'filename': 'sigmoid_center_interpolation_update_0.txt',
        'kp': 0.8,
        'smoothing_sigma': 2.0,
        'description': 'Feedback iteration 1',
    },
    {
        'filename': 'sigmoid_center_interpolation_update_1.txt',  ← NEW
        'kp': 0.8,
        'smoothing_sigma': 2.0,
        'description': 'Feedback iteration 2',
    },
]


AFTER MANUAL EDIT + RUN 3:
────────────────────────────────────────────────────────────

SIGMOID_PROFILES = [
    {
        'filename': 'sigmoid_center_interpolation0.txt',
        'kp': 0.5,                                    # ← EDITED
        'smoothing_sigma': 2.0,
        'description': 'Initial profile - reduced',   # ← EDITED
    },
    # COMMENTED OUT - no longer applied
    # {
    #     'filename': 'sigmoid_center_interpolation_update_0.txt',
    #     'kp': 0.8,
    #     'smoothing_sigma': 2.0,
    #     'description': 'Feedback iteration 1',
    # },
    {
        'filename': 'sigmoid_center_interpolation_update_1.txt',
        'kp': 1.0,                                    # ← EDITED
        'smoothing_sigma': 1.5,                       # ← EDITED
        'description': 'Feedback iteration 2 - tuned', # ← EDITED
    },
    {
        'filename': 'sigmoid_center_interpolation_update_2.txt',  ← NEW
        'kp': 0.8,
        'smoothing_sigma': 2.0,
        'description': 'Feedback iteration 3',
    },
]

    RESULT: Only 3 profiles applied (update_0 disabled by comment)
```

## Accumulation Process - Mathematical View

```
Base Profile:    [—————————————————————————————————]
                         base_profile(x)

Sigmoid 1:       [=======▁▁▁▁▁▁▁======]  × kp₁ = [==▂▂▂▂▂▂▂==]
(smoothed)       smoothed_1(x)              sigmoid_1_scaled(x)

Sigmoid 2:       [===▂▂▁▁▁▁▁▂▂===]  × kp₂ = [=▁▁▂▂▂▂▂▁▁=]
(smoothed)       smoothed_2(x)              sigmoid_2_scaled(x)

Sigmoid 3:       [=▁▁▂▂▃▃▂▂▁▁=]  × kp₃ = [▁▂▂▃▃▄▄▃▃▂▂▁]
(smoothed)       smoothed_3(x)              sigmoid_3_scaled(x)

                            ├─ Add all together
                            ▼

Combined Effect: [==▃▃▄▅▆▆▅▆▄▄==]

Final Profile:   [————▃▄▅▆▇▇▆▇▅▅————]
=base + sigmoid_1_scaled + sigmoid_2_scaled + sigmoid_3_scaled
```

## Visualization Output Layout

```
┌─────────────────────────────────────────────────────┐
│           DMD Profile Comparison - 3 sigmoids       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Intensity │                                        │
│      ▲     │  ╱╲─ New Profile (red)                │
│      │     │ ╱  ╲                                  │
│      │     │╱    ╲___  Old Profile (blue)          │
│      │     ├ ─ ─ Feedback Region                   │
│      │ ───┼─────────────────────────────────────→  │
│      │    │                                    x   │
│      └────┴─────────────────────────────────────   │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Correction │       ▁▁▁▂▂▂▂▂▁▁  (colored lines)     │
│         ▲   │      ╱            ╲                   │
│         │   │     ╱   Sigmoid 1   ╲     (color 1)  │
│         │   │    ╱ ╱ ╲ Sigmoid 2 ╱ ╲   (color 2)  │
│         │   │   ╱▂▂▃▃▃▃▄▄▄▃▃▂▂╲  (color 3)      │
│      ───┼───┼──┼────────────────────────────       │
│         │   │  │                  Total (black)    │
│         └───┴──┴─────────────────────────────→ x  │
│                                                     │
│  Legend:                                           │
│  ─ Sigmoid 1 (σ=2.0, kp=0.8)                      │
│  ─ Sigmoid 2 (σ=1.5, kp=1.0)                      │
│  ─ Sigmoid 3 (σ=2.0, kp=0.8)                      │
│  ─ Total applied correction                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## File System Organization

```
/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD/

├── auto_feedback_update.py                    ← Main script (RUN THIS)
├── feedback_automation_config.py              ← Settings (edit if needed)
├── feedback_sigmoids_list.py                  ← Profiles (edit to tune)
├── feedback_manual_update.py                  ← Original (still works)
│
├── sigmoid_center_interpolation0.txt          ← Base profile (initial)
├── sigmoid_center_interpolation_update_0.txt  ← First feedback (auto-saved)
├── sigmoid_center_interpolation_update_1.txt  ← Second feedback (auto-saved)
├── sigmoid_center_interpolation_update_2.txt  ← Third feedback (auto-saved)
│
├── plot_sigmoid_txt.py                        ← Visualization helper
├── plot_sigmoid_txt_config.py                 ← Auto-updated by script
│
├── QUICK_START.md                             ← Start here!
├── AUTOMATED_FEEDBACK_WORKFLOW.md             ← Full guide
├── IMPLEMENTATION_DETAILS.md                  ← Technical deep-dive
├── IMPLEMENTATION_COMPLETE.md                 ← This summary
└── SYSTEM_ARCHITECTURE.md                     ← This file
```

## State Machine View

```
                    ┌─────────────┐
                    │   INITIAL   │
                    └──────┬──────┘
                           │ feedback_sigmoids_list.py exists
                           │ with 1 base profile
                           ▼
                 ┌──────────────────┐
              ┌─►│  READY TO RUN    │◄─┐
              │  └──────────┬───────┘  │
              │             │ Run script
              │             ▼          │
              │        ┌─────────────────────────────┐
              │        │ 1. Load base profile        │
              │        │ 2. Load new sigmoid         │
              │        │ 3. Save + register          │
              │        │ 4. Combine all corrections  │
              │        │ 5. Visualize                │
              │        │ 6. Update config            │
              │        │ 7. Load to DMD              │
              │        └─────────────┬───────────────┘
              │                      │ Script completes
              │                      ▼
              │        ┌─────────────────────────────┐
              │        │ LIST UPDATED                │
              │        │ feedback_sigmoids_list.py   │
              │        │ now has N+1 profiles        │
              │        └────────────┬────────────────┘
              │                     │ Optional: user edits
              │                     │ parameters in file
              │                     ▼
              │          ┌──────────────────┐
              └──────────│ READY TO RUN     │
                         └──────────────────┘
                              ▲
                              │ Repeat for each
                              │ waterfall iteration

   Manual Edit Cycle:
   ┌─────────────────────────────────────────────────┐
   │ Edit feedback_sigmoids_list.py:                 │
   │  - Change kp values                             │
   │  - Change sigma values                          │
   │  - Comment out profiles to disable              │
   │  - Add new manual entries                       │
   └────────────┬────────────────────────────────────┘
                │
                ▼
   ┌─────────────────────────────────────────────────┐
   │ Next run applies updated parameters             │
   │ Changes take effect immediately                 │
   └────────────┬────────────────────────────────────┘
                │
                └──► Back to READY TO RUN state
```

## Command Execution Flow

```
$ cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD
$ python auto_feedback_update.py [OPTIONS]

┌──────────────────────────────────────┐
│ Parse Command Line Arguments         │
│  --kp VALUE                          │
│  --sigma VALUE                       │
│  --no-server                         │
│  --no-plot-update                    │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Import Configuration Modules         │
│  ✓ feedback_automation_config        │
│  ✓ feedback_sigmoids_list            │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Execute main() with Options          │
│  use_h5=config.USE_H5_PROFILE        │
│  kp=args.kp or config.DEFAULT_NEW_KP │
│  sigma=args.sigma or config.DEFAULT  │
│  load_to_dmd=not args.no_server      │
│  update_plot=not args.no_plot_update │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Return: Success/Failure              │
│  Exit code 0 = Success               │
│  Exit code 1 = Failure               │
└──────────────────────────────────────┘
```
