"""
Edit and Load DMD Profile Script

This script allows you to:
1. Load a DMD profile from a txt file
2. Interactively edit the profile within the feedback walls
3. Send the modified profile to the DMD server

Usage:
    python edit_and_load_profile.py <path_to_profile.txt>
    
    OR run without arguments to open a file dialog.
"""

import zerorpc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import h5py
import importlib
import os
import sys
from scipy.interpolate import interp1d
from datetime import datetime

# Import feedback configuration
spec = importlib.util.spec_from_file_location("feedback_config", 
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/DMD/feedback_config.py")
feedback_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback_config)

# =============================================================================
# CONFIGURATION
# =============================================================================

# DMD server connection
DMD_SERVER_IP = feedback_config.DMD_SERVER_IP
DMD_SERVER_PORT = feedback_config.DMD_SERVER_PORT

# Wall configuration
X_CENTER = feedback_config.X_CENTER
FEEDBACK_WIDTH = feedback_config.FEEDBACK_WIDTH
WALL_TYPE = feedback_config.WALL_TYPE
SOFT_WALL_WIDTH = feedback_config.SOFT_WALL_WIDTH

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================


class ProfileEditor:
    """Interactive profile editor with wall boundaries."""
    
    def __init__(self, x_um, profile_y):
        """
        Initialize the editor with profile data.
        
        Args:
            x_um: x-axis positions in micrometers
            profile_y: profile values
        """
        self.x_um = np.array(x_um)
        self.profile_y = np.array(profile_y)
        self.original_profile = np.copy(profile_y)
        
        # Define wall regions
        self.left_wall = X_CENTER - FEEDBACK_WIDTH
        self.right_wall = X_CENTER + FEEDBACK_WIDTH
        
        # Identify editable region (inside walls)
        self.editable_mask = (self.x_um >= self.left_wall) & (self.x_um <= self.right_wall)
        
        # Create figure and axes
        self.fig, (self.ax_profile, self.ax_slider) = plt.subplots(
            2, 1, figsize=(14, 10), 
            gridspec_kw={'height_ratios': [4, 1]},
            tight_layout=True
        )
        
        # Plot original profile
        self.line_orig, = self.ax_profile.plot(
            self.x_um, self.original_profile, 'b--', linewidth=1.5, 
            alpha=0.5, label='Original Profile'
        )
        
        # Plot editable profile
        self.line_edit, = self.ax_profile.plot(
            self.x_um, self.profile_y, 'r-', linewidth=2.5, 
            marker='o', markersize=3, alpha=0.8, label='Current Profile'
        )
        
        # Highlight editable region
        self._draw_wall_regions()
        
        # Setup axis labels and title
        self.ax_profile.set_xlabel('Position (μm)', fontsize=12)
        self.ax_profile.set_ylabel('DMD Intensity (a.u.)', fontsize=12)
        self.ax_profile.set_title('DMD Profile Editor - Modify values inside the walls', 
                                 fontsize=14, fontweight='bold')
        self.ax_profile.legend(loc='upper right')
        self.ax_profile.grid(True, alpha=0.3)
        self.ax_profile.set_ylim(bottom=-0.1)
        
        # Add statistics text
        self.stats_text = self.ax_profile.text(
            0.02, 0.98, '', transform=self.ax_profile.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8), fontsize=10, family='monospace'
        )
        
        # Setup sliders for editing
        self._setup_sliders()
        
        # Setup mouse event for clicking on profile
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Instructions
        instruction_text = (
            "INSTRUCTIONS:\n"
            "• Click on the profile to select a point (or region) for editing\n"
            "• Use the Value slider to set the intensity of the selected point(s)\n"
            "• Use the Selection Width slider to broaden/narrow the selection region\n"
            "• Click 'Apply & Load' to send to DMD\n"
            "• Click 'Reset' to restore original profile"
        )
        print(instruction_text)

        self.selected_indices = []
        self.click_x_um = None
        self.selection_span = None
        self._updating_selection = False
        self.update_stats()
    
    def _draw_wall_regions(self):
        """Draw wall regions on the plot."""
        # Feedback region (inside walls)
        feedback_rect = patches.Rectangle(
            (self.left_wall, 0), FEEDBACK_WIDTH * 2, 
            self.ax_profile.get_ylim()[1],
            linewidth=2, edgecolor='green', facecolor='green', 
            alpha=0.1, label='Editable Region'
        )
        self.ax_profile.add_patch(feedback_rect)
        
        if WALL_TYPE == 'soft':
            # Soft wall regions
            left_soft_start = self.left_wall - SOFT_WALL_WIDTH
            right_soft_end = self.right_wall + SOFT_WALL_WIDTH
            
            left_soft_rect = patches.Rectangle(
                (left_soft_start, 0), SOFT_WALL_WIDTH,
                self.ax_profile.get_ylim()[1],
                linewidth=1, edgecolor='orange', facecolor='orange',
                alpha=0.05, label='Soft Wall Region'
            )
            self.ax_profile.add_patch(left_soft_rect)
            
            right_soft_rect = patches.Rectangle(
                (self.right_wall, 0), SOFT_WALL_WIDTH,
                self.ax_profile.get_ylim()[1],
                linewidth=1, edgecolor='orange', facecolor='orange',
                alpha=0.05
            )
            self.ax_profile.add_patch(right_soft_rect)
        
        # Vertical lines at wall boundaries
        self.ax_profile.axvline(self.left_wall, color='green', linestyle=':', 
                               linewidth=2, alpha=0.7)
        self.ax_profile.axvline(self.right_wall, color='green', linestyle=':', 
                               linewidth=2, alpha=0.7)
    
    def _setup_sliders(self):
        """Setup sliders for profile editing."""
        slider_height = 0.03

        # Value slider
        ax_value = plt.axes([0.15, 0.08, 0.55, slider_height])
        self.slider_value = Slider(
            ax_value, 'Value', 0, 1.0, valinit=0.5, valstep=0.01
        )
        self.slider_value.on_changed(self.on_slider_changed)

        # Selection width (resolution) slider
        ax_resolution = plt.axes([0.15, 0.03, 0.55, slider_height])
        self.slider_resolution = Slider(
            ax_resolution, 'Selection Width (µm)', 0, 100, valinit=0, valstep=1
        )
        self.slider_resolution.on_changed(self.on_resolution_changed)

        # Buttons
        ax_apply = plt.axes([0.73, 0.075, 0.10, 0.04])
        self.btn_apply = plt.Button(ax_apply, 'Apply & Load', hovercolor='0.975')
        self.btn_apply.on_clicked(self.on_apply)

        ax_reset = plt.axes([0.85, 0.075, 0.08, 0.04])
        self.btn_reset = plt.Button(ax_reset, 'Reset', hovercolor='0.975')
        self.btn_reset.on_clicked(self.on_reset)
    
    def _update_selection(self, x_clicked):
        """Update selected indices based on click position and current selection width."""
        half_width = self.slider_resolution.val / 2.0

        editable_indices = np.where(self.editable_mask)[0]
        if len(editable_indices) == 0:
            self.selected_indices = []
            return

        editable_x = self.x_um[editable_indices]

        if half_width < 0.5:
            # Select single nearest point
            nearest = np.argmin(np.abs(editable_x - x_clicked))
            self.selected_indices = [editable_indices[nearest]]
        else:
            within = np.abs(editable_x - x_clicked) <= half_width
            indices_within = editable_indices[within]
            if len(indices_within) == 0:
                nearest = np.argmin(np.abs(editable_x - x_clicked))
                self.selected_indices = [editable_indices[nearest]]
            else:
                self.selected_indices = list(indices_within)

        # Update selection span visual
        if self.selection_span is not None:
            self.selection_span.remove()
            self.selection_span = None

        if self.selected_indices:
            x_sel = self.x_um[self.selected_indices]
            x_min_sel, x_max_sel = x_sel.min(), x_sel.max()
            if x_min_sel == x_max_sel:
                dx = abs(self.x_um[1] - self.x_um[0]) if len(self.x_um) > 1 else 1.0
                x_min_sel -= dx * 0.5
                x_max_sel += dx * 0.5
            self.selection_span = self.ax_profile.axvspan(
                x_min_sel, x_max_sel, alpha=0.3, color='yellow', zorder=2
            )
            mean_val = float(np.mean(self.profile_y[self.selected_indices]))
            self._updating_selection = True
            self.slider_value.set_val(np.clip(mean_val, 0.0, 1.0))
            self._updating_selection = False

        self.update_stats()
        self.fig.canvas.draw()

    def on_resolution_changed(self, val):
        """Re-apply selection when resolution slider changes."""
        if self.click_x_um is not None:
            self._update_selection(self.click_x_um)

    def on_click(self, event):
        """Handle mouse click on profile."""
        if event.inaxes != self.ax_profile or event.xdata is None:
            return

        self.click_x_um = event.xdata
        self._update_selection(event.xdata)

        if self.selected_indices:
            n = len(self.selected_indices)
            mean_val = float(np.mean(self.profile_y[self.selected_indices]))
            print(f"Selected {n} point(s) around x={event.xdata:.2f} µm, "
                  f"mean value={mean_val:.4f}")
    
    def on_slider_changed(self, val):
        """Handle value slider change: apply to all selected indices."""
        if self._updating_selection:
            return
        if self.selected_indices:
            self.profile_y[self.selected_indices] = val
            self.line_edit.set_ydata(self.profile_y)
            self.update_stats()
            self.fig.canvas.draw()
    
    def on_apply(self, event):
        """Apply changes and send to DMD server."""
        plt.close(self.fig)
        print("\nApplying profile changes and loading to DMD...")
    
    def on_reset(self, event):
        """Reset to original profile."""
        self.profile_y[:] = self.original_profile
        self.slider_value.set_val(0.5)
        self.slider_resolution.set_val(0)
        self.selected_indices = []
        self.click_x_um = None
        if self.selection_span is not None:
            self.selection_span.remove()
            self.selection_span = None
        self.line_edit.set_ydata(self.profile_y)
        self.update_stats()
        self.fig.canvas.draw()
        print("Profile reset to original.")
    
    def update_stats(self):
        """Update statistics display."""
        mean_val = np.mean(self.profile_y[self.editable_mask])
        max_val = np.max(self.profile_y[self.editable_mask])
        min_val = np.min(self.profile_y[self.editable_mask])

        if self.selected_indices:
            n = len(self.selected_indices)
            sel_mean = float(np.mean(self.profile_y[self.selected_indices]))
            selected_text = f"Selected ({n} pts): {sel_mean:.4f}"
        else:
            selected_text = "Click to select"

        stats_text = (
            f"Editable Region Stats:\n"
            f"Mean: {mean_val:.4f}\n"
            f"Max: {max_val:.4f}\n"
            f"Min: {min_val:.4f}\n"
            f"{selected_text}"
        )
        self.stats_text.set_text(stats_text)
    
    def get_modified_profile(self):
        """Return the modified profile."""
        return self.x_um, self.profile_y


def load_profile_from_txt(txt_path):
    """
    Load DMD profile from txt file.
    
    Expected format: two columns (x_um, profile_y)
    
    Returns:
        (x_um, profile_y) or (None, None) if failed
    """
    try:
        if not os.path.exists(txt_path):
            print(f"Error: File not found: {txt_path}")
            return None, None
        
        data = np.loadtxt(txt_path, comments='#', skiprows=1)
        
        if data.ndim != 2 or data.shape[1] != 2:
            print(f"Error: Expected 2 columns in {txt_path}, got shape {data.shape}")
            return None, None
        
        x_um = data[:, 0]
        profile_y = data[:, 1]
        
        print(f"Loaded profile from {txt_path}:")
        print(f"  {len(x_um)} points")
        print(f"  x range: {x_um.min():.2f} to {x_um.max():.2f} um")
        print(f"  profile range: {profile_y.min():.4f} to {profile_y.max():.4f}")
        
        return x_um, profile_y
        
    except Exception as e:
        print(f"Error loading profile from txt file: {e}")
        return None, None


def send_profile_to_dmd(x_um, profile_y):
    """
    Send modified profile to DMD server.
    
    Args:
        x_um: x-axis positions in micrometers
        profile_y: profile values
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\nConnecting to DMD server at {DMD_SERVER_IP}:{DMD_SERVER_PORT}...")
        client = zerorpc.Client()
        client.connect(f"tcp://{DMD_SERVER_IP}:{DMD_SERVER_PORT}")
        
        print("Testing connection...")
        hello_response = client.hello()
        print(f"Server response: {hello_response}")
        
        print("Sending profile to DMD...")
        status = client.load_1d_profile(x_um.tolist(), profile_y.tolist())
        print(f"DMD update status: {status}")
        
        client.close()
        print("\nProfile successfully sent to DMD!")
        return True
        
    except Exception as e:
        print(f"\nERROR: Failed to send profile to DMD: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_modified_profile(x_um, profile_y, output_filename=None):
    """
    Save the modified profile to a txt file.
    
    Args:
        x_um: x-axis positions
        profile_y: profile values
        output_filename: optional custom filename
    
    Returns:
        Path to saved file
    """
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"dmd_profile_edited_{timestamp}.txt"
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    header = "x_um\tprofile_y"
    data = np.column_stack((x_um, profile_y))
    
    np.savetxt(output_path, data, header=header, delimiter='\t', 
               fmt='%.6f', comments='')
    
    print(f"\nModified profile saved to: {output_path}")
    return output_path


def get_input_file_path():
    """Get input file path from command line or user."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    # Try to open file dialog
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Select DMD profile txt file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=OUTPUT_DIR
        )
        
        if file_path:
            return file_path
    except Exception as e:
        print(f"File dialog not available: {e}")
    
    # Fallback to default
    default_file = os.path.join(OUTPUT_DIR, "dmd_last.txt")
    if os.path.exists(default_file):
        print(f"Using default file: {default_file}")
        return default_file
    
    print("Error: No input file specified and no default file found.")
    return None


def main():
    """Main execution function."""
    
    print("="*70)
    print("DMD Profile Editor and Loader")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1) Get input file
    # -------------------------------------------------------------------------
    input_file = get_input_file_path()
    if input_file is None:
        print("ERROR: Could not determine input file path.")
        return
    
    print(f"\nInput file: {input_file}")
    
    # -------------------------------------------------------------------------
    # 2) Load profile from txt
    # -------------------------------------------------------------------------
    x_um, profile_y = load_profile_from_txt(input_file)
    if x_um is None:
        print("ERROR: Could not load profile from txt file.")
        return
    
    # -------------------------------------------------------------------------
    # 3) Open interactive editor
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("Opening interactive profile editor...")
    print("="*70)
    
    editor = ProfileEditor(x_um, profile_y)
    plt.show()
    
    # Check if user applied changes
    x_modified, profile_modified = editor.get_modified_profile()
    
    # -------------------------------------------------------------------------
    # 4) Save modified profile
    # -------------------------------------------------------------------------
    save_modified_profile(x_modified, profile_modified)
    
    # -------------------------------------------------------------------------
    # 5) Send to DMD server
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    user_input = input("Send modified profile to DMD? (y/n): ").strip().lower()
    
    if user_input == 'y':
        success = send_profile_to_dmd(x_modified, profile_modified)
        if success:
            print("\n" + "="*70)
            print("Profile successfully loaded to DMD!")
            print("="*70)
    else:
        print("\nProfile not sent to DMD. Exiting.")
        print("="*70)


if __name__ == '__main__':
    main()
