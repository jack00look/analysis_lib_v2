# Complete File Listing and Purposes

## Master Directory: `article_analysis/`

### Top-Level Files

| File | Type | Purpose | Editable |
|------|------|---------|----------|
| `README.md` | Markdown | Master overview of entire framework | No |
| `IMPLEMENTATION_SUMMARY.md` | Markdown | Complete technical documentation | No |
| `FILES.md` | Markdown | This file - quick reference | No |
| `setup_new_article.sh` | Bash Script | Template for creating new articles | No (just run it) |

### Generated Directories

| Directory | Purpose | Auto-Created |
|-----------|---------|--------------|
| `KZ_article/` | Example article (KZ detuning scan) | YES, included |
| `results/` | (Created inside each article folder) | YES, on first run |

---

## KZ Article: `KZ_article/`

### Configuration Files (Edit These!)

| File | Type | Purpose | Editable |
|------|------|---------|----------|
| `config.yaml` | YAML Config | Main configuration with dataset list and parameters | **YES** |
| `camera_settings.yaml` | YAML Config | Camera setup and image processing | Optional |
| `defect_config.yaml` | YAML Config | Domain wall algorithm parameters | Optional |

#### `config.yaml` Details
- **What to edit**: Dataset list (date, sequences, affine_correction, integration limits)
- **When to edit**: Initial setup and parameter tuning iterations
- **Never change**: shared_params (unless adjusting filtering thresholds)
- **Format**: YAML with nested structure
- **Saved with each iteration** for reproducibility

#### `camera_settings.yaml` Details
- **What to edit**: Usually nothing (rarely needs changes)
- **When to edit**: If using different camera or data_origin
- **Typical uses**: Shared across all datasets in article
- **Format**: YAML defining camera, modality, atom images
- **Saved with each iteration**

#### `defect_config.yaml` Details
- **What to edit**: Algorithm parameters (optional fine-tuning)
- **When to edit**: If domain wall detection needs adjustment
- **Typical values**: Pre-tuned for standard experiments
- **Format**: YAML with algorithm parameters
- **Saved with each iteration**

### Main Scripts (Run These!)

| File | Type | Purpose | When to Use |
|------|------|---------|------------|
| `run_kz_article_analysis.py` | Python Script | Interactive analysis with parameter tuning | **First-time analysis** |
| `batch_kz_article_analysis.py` | Python Script | Batch analysis (non-interactive) | **Reruns, publication** |

#### `run_kz_article_analysis.py` Details
- **Purpose**: Visual parameter tuning with user prompts
- **Usage**: `python run_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml`
- **Interactive**: YES (requires user input)
- **Output location**: `results/YYYY/MM/DD/iteration_N/`
- **When to run**: Initial exploration, parameter tuning
- **Time per dataset**: ~15-30 seconds (includes user think time)

#### `batch_kz_article_analysis.py` Details
- **Purpose**: Automated analysis with fixed parameters
- **Usage**: 
  - `python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml`
  - `python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --silent`
  - `python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml --iteration 1`
- **Interactive**: NO (fully automated)
- **Output location**: `results/YYYY/MM/DD/iteration_N/`
- **When to run**: Publication reruns, parameter sweeps, reproducible analysis
- **Time per dataset**: ~10 seconds

### Documentation Files (Read These!)

| File | Type | Purpose | Audience |
|------|------|---------|----------|
| `README.md` | Markdown | Comprehensive KZ article guide | New users |
| `QUICKSTART.md` | Markdown | Quick reference and checklist | Quick reference |
| `IMPLEMENTATION.md` | Markdown | Technical implementation details | Advanced users |

#### `README.md` Details
- **Contains**: Full workflow, configuration, parameters, troubleshooting
- **Read time**: 15-20 minutes
- **Start here** if new to framework

#### `QUICKSTART.md` Details
- **Contains**: Quick reference, key commands, next steps
- **Read time**: 5-10 minutes
- **Quick lookup** for common tasks

#### `IMPLEMENTATION.md` Details
- **Contains**: Technical details, customization, code structure
- **Read time**: 10-15 minutes
- **Reference** for advanced customization

### Results Directory (Auto-Created): `results/`

```
results/
└── YYYY/MM/DD/                 # Calendar date of analysis
    ├── iteration_1/            # First analysis this day
    │   ├── config.yaml         # Saved configuration
    │   ├── camera_settings.yaml
    │   ├── defect_config.yaml
    │   ├── metadata.json       # Analysis metadata
    │   └── dataset_001/        # Per-dataset results
    │       ├── 01_waterfall.png
    │       ├── 02_domain_wall.png
    │       └── ...
    ├── iteration_2/            # Second analysis this day
    └── ...
```

---

## File Purposes Quick Lookup

### I want to...

**...start analyzing KZ data**
1. Read: `KZ_article/QUICKSTART.md`
2. Edit: `KZ_article/config.yaml`
3. Run: `python run_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml`

**...understand the full workflow**
1. Read: `README.md` (master) OR `KZ_article/README.md`
2. Reference: `KZ_article/IMPLEMENTATION.md` for details

**...create a new article**
1. Run: `bash setup_new_article.sh MyArticleName`
2. Edit: `MyArticleName/config.yaml`
3. Run: `python run_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml`

**...reproduce published results**
1. Get configs from publication supplementary
2. Run: `python batch_kz_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml`

**...understand my results**
1. Check: `results/YYYY/MM/DD/iteration_N/metadata.json`
2. View: `results/YYYY/MM/DD/iteration_N/dataset_XXX/*.png`

**...adjust parameters**
1. Edit: `config.yaml` with new values
2. Run batch script (new iteration created automatically)
3. Compare results between iterations

**...fix a problem**
1. Check: `KZ_article/README.md` "Common Issues" section
2. Or check: This `FILES.md` and `IMPLEMENTATION_SUMMARY.md`

---

## File Dependencies

### To Run Interactive Analysis
Requires:
- `config.yaml` (with datasets defined)
- `camera_settings.yaml`
- `defect_config.yaml`
- `run_kz_article_analysis.py`
- All supporting Python libraries (lyse, h5py, pandas, etc.)

### To Run Batch Analysis
Requires:
- `config.yaml` (with tuned parameters)
- `camera_settings.yaml`
- `defect_config.yaml`
- `batch_kz_article_analysis.py`
- All supporting Python libraries

### To Reproduce Results
Requires:
- `config.yaml` from iteration folder
- `camera_settings.yaml` from iteration folder
- `defect_config.yaml` from iteration folder
- `batch_kz_article_analysis.py`
- Same Python environment/versions

---

## Summary Table

| File | Type | Edit | Read | Run |
|------|------|------|------|-----|
| `config.yaml` | YAML | ✅ YES | ✅ YES | - |
| `camera_settings.yaml` | YAML | ⚠️ Optional | ✅ YES | - |
| `defect_config.yaml` | YAML | ⚠️ Optional | ✅ YES | - |
| `run_kz_article_analysis.py` | Python | ❌ NO | - | ✅ First time |
| `batch_kz_article_analysis.py` | Python | ❌ NO | - | ✅ Reruns |
| `README.md` (master) | Markdown | ❌ NO | ✅ START | - |
| `README.md` (KZ) | Markdown | ❌ NO | ✅ YES | - |
| `QUICKSTART.md` | Markdown | ❌ NO | ✅ QUICK | - |
| `IMPLEMENTATION.md` | Markdown | ❌ NO | ✅ DETAILS | - |
| `IMPLEMENTATION_SUMMARY.md` | Markdown | ❌ NO | ✅ COMPLETE | - |
| `setup_new_article.sh` | Bash | ❌ NO | - | ✅ Create |

---

## File Locations

```
/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/
└── article_analysis/
    ├── README.md                              # Master README
    ├── IMPLEMENTATION_SUMMARY.md              # This implementation
    ├── FILES.md                               # This file
    ├── setup_new_article.sh                   # Setup script
    │
    └── KZ_article/                            # Example article
        ├── config.yaml                        # Edit this
        ├── camera_settings.yaml               # Usually don't edit
        ├── defect_config.yaml                 # Usually don't edit
        ├── run_kz_article_analysis.py         # Run first time
        ├── batch_kz_article_analysis.py       # Run for reruns
        ├── README.md                          # Read this
        ├── QUICKSTART.md                      # Quick reference
        ├── IMPLEMENTATION.md                  # Tech details
        └── results/
            └── (auto-created on first run)
```

---

## Version Control Recommendations

### Commit to Git
```
article_analysis/
├── README.md
├── IMPLEMENTATION_SUMMARY.md
├── FILES.md
├── setup_new_article.sh
└── KZ_article/
    ├── config.yaml
    ├── camera_settings.yaml
    ├── defect_config.yaml
    ├── run_kz_article_analysis.py
    ├── batch_kz_article_analysis.py
    ├── README.md
    ├── QUICKSTART.md
    └── IMPLEMENTATION.md
```

### Exclude from Git
```
article_analysis/
└── KZ_article/
    └── results/          # ← Add to .gitignore
```

This keeps scripts/configs under version control while excluding large plot files.

