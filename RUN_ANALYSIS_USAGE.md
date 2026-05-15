# Using run_show_ODs_batch as a Function

The `run_show_ODs_batch.py` script has been refactored to support both:
1. **Command-line usage** (interactive prompts)
2. **Programmatic usage** (import and call as a function)

## Programmatic Usage

### Simple Example

```python
from run_show_ODs_batch import run_analysis

# Preprocess specific sequences
success = run_analysis(
    year=2026,
    month=3,
    day=24,
    sequences=[33, 34, 35, 36, 37, 38, 43, 44, 45, 46, 47],
    verbose=True
)

if success:
    print("Preprocessing complete!")
```

### Function Signature

```python
def run_analysis(year, month, day, sequences=None, verbose=True):
    """
    Programmatic entry point for running show_ODs_v2 batch analysis.
    
    Parameters
    ----------
    year : int
        Year (e.g., 2026)
    month : int
        Month (1-12)
    day : int
        Day (1-31)
    sequences : list, optional
        List of sequence indices to process (e.g., [26, 30]).
        If None, will process all available sequences for that date.
    verbose : bool
        If True, print status messages. If False, suppress output.
    
    Returns
    -------
    bool
        True if analysis completed successfully, False otherwise.
    """
```

## Usage Examples

### Example 1: Process specific sequences with output

```python
from run_show_ODs_batch import run_analysis

success = run_analysis(2026, 5, 7, sequences=[29, 30, 34, 35], verbose=True)
```

### Example 2: Process all sequences for a date (silent mode)

```python
from run_show_ODs_batch import run_analysis

# sequences=None means all available sequences
success = run_analysis(2026, 3, 24, sequences=None, verbose=False)
```

### Example 3: In article analysis scripts

```python
#!/usr/bin/env python3
from run_show_ODs_batch import run_analysis

# Your article analysis setup
datasets = [
    {'date': (2026, 3, 24), 'sequences': [33, 34, 35]},
    {'date': (2026, 3, 25), 'sequences': [10, 11, 12]},
]

for dataset in datasets:
    year, month, day = dataset['date']
    seqs = dataset['sequences']
    
    print(f"Preprocessing {year}-{month:02d}-{day:02d}, sequences {seqs}")
    success = run_analysis(year, month, day, sequences=seqs)
    
    if not success:
        print(f"Failed for {year}-{month:02d}-{day:02d}!")
        sys.exit(1)

print("All datasets preprocessed successfully!")
```

## Integration with Article Analysis

The `run_kz_article_analysis.py` script automatically uses this feature:

1. When you run it, it checks if preprocessing is needed
2. It asks: "Would you like to run preprocessing now?"
3. If you say yes (y), it calls `run_analysis()` automatically
4. After preprocessing completes, it continues with parameter tuning

## Command-Line Usage (unchanged)

The script still works from the terminal as before:

```bash
cd /home/rick/labscript-suite/userlib/analysislib/analysislib_v2
python run_show_ODs_batch.py
# Then follow interactive prompts
```

Or with configuration set in the script:

```python
# At the top of run_show_ODs_batch.py, set:
SEQS = [33, 34, 35, 36, 37, 38, 43, 44, 45, 46, 47]
HDF_CONFIG = {'today': False, 'year': 2026, 'month': 3, 'day': 24}
```

Then run:
```bash
python run_show_ODs_batch.py
```

## Benefits

- **Flexibility**: Use in scripts, notebooks, or interactively
- **Automation**: Integrate preprocessing into larger workflows
- **Consistency**: Same logic whether called from CLI or programmatically
- **Reliability**: No need to manually switch between terminal and scripts

