"""
Feedback Sigmoids List Configuration

This file maintains the persistent list of all sigmoid profiles used for feedback.

When auto_feedback_update.py runs, it:
1. Saves the new sigmoid from waterfall analysis
2. Automatically adds it to SIGMOID_PROFILES below
3. Loads all sigmoids in this list and applies them for feedback

You can manually edit this file to:
- Adjust kp and smoothing_sigma values per profile
- Reorder profiles (they're applied sequentially)
- Remove profiles from feedback
- Comment out profiles without deleting them

Default values for new profiles are set in feedback_automation_config.py:
- DEFAULT_SMOOTHING_SIGMA = 2.0
- NEW_PROFILE_KP = 0.8
"""

# =============================================================================
# SIGMOID PROFILES FOR FEEDBACK
# =============================================================================

# List of sigmoid profiles to apply for feedback
# Each profile will be loaded, mean-subtracted, extended to full DMD x-range, 
# smoothed with chosen sigma, scaled by kp, and added to the DMD profile
SIGMOID_PROFILES = [
    
    {
        'filename': 'sigmoid_center_interpolation_update_0.txt',
        'kp': 0.3,
        'smoothing_sigma': 1.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'filename': 'sigmoid_center_interpolation_update_1.txt',
        'kp': 0.2,
        'smoothing_sigma': 1.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'filename': 'sigmoid_center_interpolation_update_2.txt',
        'kp': 0.4,
        'smoothing_sigma': 1.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'filename': 'sigmoid_center_interpolation_update_3.txt',
        'kp': 0.3,
        'smoothing_sigma': 1.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'filename': 'sigmoid_center_interpolation_update_4.txt',
        'kp': 0.3,
        'smoothing_sigma': 1.0,
        'description': 'Auto-added from waterfall',
    },
    ]
