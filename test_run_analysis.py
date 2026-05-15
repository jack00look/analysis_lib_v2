#!/usr/bin/env python3
"""
Example script showing how to call run_show_ODs_batch.run_analysis() programmatically.

Instead of running run_show_ODs_batch.py from the terminal,
you can import it and call run_analysis() with specific parameters.
"""

from run_show_ODs_batch import run_analysis

# Example 1: Preprocess specific sequences from a specific date
print("="*70)
print("EXAMPLE: Preprocessing KZ dataset (2026-03-24, sequences 33-38, 43-47)")
print("="*70)

success = run_analysis(
    year=2026,
    month=3,
    day=24,
    sequences=[33, 34, 35, 36, 37, 38, 43, 44, 45, 46, 47],
    verbose=True
)

if success:
    print("\n✓ Preprocessing complete! Data is ready for article analysis.")
else:
    print("\n✗ Preprocessing failed!")

# Example 2: Process all available sequences for a date (sequences=None)
# success = run_analysis(2026, 5, 10, sequences=None, verbose=True)

# Example 3: Silent mode for automated workflows
# success = run_analysis(2026, 3, 24, sequences=[29, 30], verbose=False)
