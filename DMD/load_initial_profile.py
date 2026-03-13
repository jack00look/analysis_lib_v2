"""
Load initial DMD profile with walls.

This script generates and loads an initial profile consisting of:
- Background intensity outside the feedback region
- Feedback intensity in the feedback region
- Optional hard or soft walls for confinement

Similar to load_profile_test.py but using centralized configuration.
"""

import zerorpc
import numpy as np
import matplotlib.pyplot as plt
import sys

# Import configuration
spec_config = __import__('importlib.util').util.spec_from_file_location(
    "feedback_config", 
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD/feedback_config.py"
)
feedback_config = __import__('importlib.util').util.module_from_spec(spec_config)
spec_config.loader.exec_module(feedback_config)

# Connect to DMD server
client = zerorpc.Client()
client.connect(f"tcp://{feedback_config.DMD_SERVER_IP}:{feedback_config.DMD_SERVER_PORT}")
print('Server response:', client.hello())

# ============================================================================
# Load configuration
# ============================================================================
config = feedback_config.INITIAL_PROFILE_CONFIG
pixel_size = feedback_config.PIXEL_SIZE_UM
n_pixels = feedback_config.DMD_PIXELS

bg_value = config['background_value']
fb_value = config['feedback_value']
x_center = feedback_config.X_CENTER
fb_width = feedback_config.FEEDBACK_WIDTH
wall_type = feedback_config.WALL_TYPE
wall_width = feedback_config.SOFT_WALL_WIDTH

# ============================================================================
# Generate profile
# ============================================================================
profile_x = np.arange(n_pixels) * pixel_size
profile_y = np.ones_like(profile_x) * bg_value

# Define feedback region
x_min = x_center - fb_width
x_max = x_center + fb_width

# Set feedback region
ind_feedback = (profile_x >= x_min) & (profile_x <= x_max)
profile_y[ind_feedback] = fb_value

# Apply walls
if wall_type == 'hard':
    # Hard walls: already have background outside region, just ensure it
    profile_y[~ind_feedback] = bg_value
    print(f'Applied hard walls at {x_center} ± {fb_width} µm')
    
elif wall_type == 'soft':
    # Soft walls: cos^2 transition from feedback region edge to background
    left_edge = x_center - fb_width
    right_edge = x_center + fb_width
    
    # Left wall transition
    left_wall_start = left_edge - wall_width
    ind_left_wall = (profile_x >= left_wall_start) & (profile_x < left_edge)
    if np.any(ind_left_wall):
        # Distance from left edge (0 at left_edge, wall_width at left_wall_start)
        dist = left_edge - profile_x[ind_left_wall]
        # cos^2 transition: fb_value at edge, bg_value at wall_start
        transition = np.cos(np.pi/2 * (1 - dist/wall_width))**2
        profile_y[ind_left_wall] = fb_value + (bg_value - fb_value) * transition
    
    # Right wall transition
    right_wall_end = right_edge + wall_width
    ind_right_wall = (profile_x > right_edge) & (profile_x <= right_wall_end)
    if np.any(ind_right_wall):
        # Distance from right edge (0 at right_edge, wall_width at right_wall_end)
        dist = profile_x[ind_right_wall] - right_edge
        # cos^2 transition: fb_value at edge, bg_value at wall_end
        transition = np.cos(np.pi/2 * (1 - dist/wall_width))**2
        profile_y[ind_right_wall] = fb_value + (bg_value - fb_value) * transition
    
    # Set background beyond soft wall regions
    profile_y[profile_x < left_wall_start] = bg_value
    profile_y[profile_x > right_wall_end] = bg_value
    print(f'Applied soft walls at {x_center} ± {fb_width} µm with {wall_width} µm transition')

else:
    print('No walls applied (wall_type = none)')

# ============================================================================
# Retrieve and plot current state
# ============================================================================
last_2d_image = client.get_last_2d_image()
fig_2d, ax_2d = plt.subplots(figsize=(10, 4))
ax_2d.imshow(last_2d_image, cmap='viridis', aspect='auto')
ax_2d.set_title('Last 2D Image from DMD Server')
ax_2d.set_xlabel('Pixel X')
ax_2d.set_ylabel('Pixel Y')
plt.colorbar(ax_2d.imshow(last_2d_image, cmap='viridis', aspect='auto'), 
             label='Intensity', ax=ax_2d)

# Try to get last atoms profile
last_atoms_profile = client.get_last_atoms_profile()
try:
    x_last = np.array(last_atoms_profile['x'])
    y_last = np.array(last_atoms_profile['y'])
    print('Retrieved last atoms profile')
except Exception as e:
    print('Could not retrieve last atoms profile:', e)
    x_last, y_last = None, None

# Plot 1D profiles
fig_1d, ax_1d = plt.subplots(figsize=(12, 6))

if x_last is not None and y_last is not None:
    ax_1d.plot(x_last, y_last, label='Last Atoms Profile', linewidth=2, alpha=0.7)

ax_1d.plot(profile_x, profile_y, 'r-', label='Generated Profile', linewidth=2)
ax_1d.set_xlabel('Position (µm)')
ax_1d.set_ylabel('Intensity (a.u.)')
ax_1d.set_title('Initial Load Profile for DMD')
ax_1d.legend(fontsize=12)
ax_1d.grid(alpha=0.3)

# Mark regions on plot
ax_1d.axvline(x_min, color='green', linestyle=':', alpha=0.5, label='Feedback Region')
ax_1d.axvline(x_max, color='green', linestyle=':', alpha=0.5)

if wall_type == 'soft':
    ax_1d.axvline(x_min - wall_width, color='orange', linestyle=':', alpha=0.5, label='Wall Boundary')
    ax_1d.axvline(x_max + wall_width, color='orange', linestyle=':', alpha=0.5)

ax_1d.legend(fontsize=11)
plt.tight_layout()
plt.show()

# ============================================================================
# Send profile to DMD
# ============================================================================
print('\nSending profile to DMD server...')
status = client.load_1d_profile(profile_x.tolist(), profile_y.tolist())
print(f'Status: {status}')

# Print configuration summary
print('\n' + '='*60)
print('INITIAL PROFILE SUMMARY')
print('='*60)
print(f'Background value: {bg_value}')
print(f'Feedback value: {fb_value}')
print(f'Feedback region: {x_center} ± {fb_width} µm')
print(f'Wall type: {wall_type}')
if wall_type == 'soft':
    print(f'Wall width: {wall_width} µm')
print('='*60)

client.close()
