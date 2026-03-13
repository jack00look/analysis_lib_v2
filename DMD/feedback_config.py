"""
Configuration file for DMD feedback scripts.
Centralizes all feedback parameters in one place.
"""

# =============================================================================
# DMD SERVER PARAMETERS
# =============================================================================
DMD_SERVER_IP = '192.168.1.154'
DMD_SERVER_PORT = '6001'

# =============================================================================
# SHARED FEEDBACK REGION AND WALL CONFIGURATION
# =============================================================================
# These parameters apply to ALL feedback scripts
X_CENTER =1041                    # Center position (µm)
FEEDBACK_WIDTH = 150                # Half-width of feedback region (µm)
WALL_TYPE = 'soft'                  # Options: 'none', 'hard', 'soft'
SOFT_WALL_WIDTH = 30                # Width of soft wall transition region (µm)

# =============================================================================
# DENSITY FEEDBACK PARAMETERS
# =============================================================================
DENSITY_CONFIG = {
    'program_name': 'Do_BEC_DMD_feedback',
    'n_threshold': 2e5,             # Minimum atom number for feedback
    'kp': 0.5e-4,                    # Proportional gain
    'smoothing_sigma': 10.,        # Gaussian smoothing sigma for error signal
}

# =============================================================================
# MAGNETIZATION FEEDBACK PARAMETERS
# =============================================================================
MAGNETIZATION_CONFIG = {
    'program_name': 'Do_ARP_Forward',
    'n_threshold': 2e5,             # Minimum atom number for feedback
    'kp': -.1e-1,                    # Proportional gain
    'smoothing_sigma': 3.0,        # Gaussian smoothing sigma for error signal
}

# =============================================================================
# INITIAL PROFILE CONFIGURATION
# =============================================================================
# Used by load_initial_profile.py to create the initial DMD profPile
INITIAL_PROFILE_CONFIG = {
    'background_value': 1.0,        # Background intensity outside feedback region
    'feedback_value': 0.2,          # Intensity in the feedback region
}

# =============================================================================
# MULTI-SHOT AVERAGING PARAMETERS
# =============================================================================
N_SHOTS_TO_AVERAGE = 1              # Number of shots to average (1 = no averaging)
MAX_LOOKBACK_TIME = 300.0           # Maximum time to look back for shots (seconds)
STRICT_MODE = True                  # If True, don't reuse shots that already triggered feedback
                                    # If False, allow overlapping windows

# =============================================================================
# UTILITY PARAMETERS
# =============================================================================
# Pixel size and array dimensions (from imaging system calibration)
PIXEL_SIZE_UM = 1.019  # µm per pixel on DMD
DMD_PIXELS = 2048      # Total number of pixels in profile

