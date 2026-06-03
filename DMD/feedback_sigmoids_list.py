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
        'sigmoid_filename': 'sigmoid_center_interpolation_update_0.txt',
        'density_filename': 'density_error_profile_update_0.txt',
        'kp_sigmoid': 0.8,
        'smoothing_sigma_sigmoid': 5.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_1.txt',
        'density_filename': 'density_error_profile_update_1.txt',
        'kp_sigmoid': 0.6,
        'smoothing_sigma_sigmoid': 5.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_2.txt',
        'density_filename': 'density_error_profile_update_2.txt',
        'kp_sigmoid': 0.6,
        'smoothing_sigma_sigmoid': 5.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_3.txt',
        'density_filename': 'density_error_profile_update_3.txt',
        'kp_sigmoid': 0.6,
        'smoothing_sigma_sigmoid': 2.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_4.txt',
        'density_filename': 'density_error_profile_update_4.txt',
        'kp_sigmoid': 0.5,
        'smoothing_sigma_sigmoid': 2.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_5.txt',
        'density_filename': 'density_error_profile_update_5.txt',
        'kp_sigmoid': 0.4,
        'smoothing_sigma_sigmoid': 2.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_6.txt',
        'density_filename': 'density_error_profile_update_6.txt',
        'kp_sigmoid': 0.5,
        'smoothing_sigma_sigmoid': 2.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_7.txt',
        'density_filename': 'density_error_profile_update_7.txt',
        'kp_sigmoid': 0.5,
        'smoothing_sigma_sigmoid': 2.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_8.txt',
        'density_filename': 'density_error_profile_update_8.txt',
        'kp_sigmoid': 0.5,
        'smoothing_sigma_sigmoid': 2.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_9.txt',
        'density_filename': 'density_error_profile_update_9.txt',
        'kp_sigmoid': 0.5,
        'smoothing_sigma_sigmoid': 2.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_10.txt',
        'density_filename': 'density_error_profile_update_10.txt',
        'kp_sigmoid': 0.6,
        'smoothing_sigma_sigmoid': 2.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    
    {
        'sigmoid_filename': 'sigmoid_center_interpolation_update_11.txt',
        'density_filename': 'density_error_profile_update_11.txt',
        'kp_sigmoid': 0.6,
        'smoothing_sigma_sigmoid': 2.0,
        'kp_density': 0.0,
        'smoothing_sigma_density': 3.0,
        'description': 'Auto-added from waterfall',
    },
    ]
