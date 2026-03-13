"""
Get Last DMD Profile Script

This script retrieves the last profile that was sent to the DMD,
displays it in a plot, and saves it to a txt file.
"""

import zerorpc
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import importlib

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

# Output directory for saving profile
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================


def get_last_profile_from_dmd():
    """
    Connect to DMD server and retrieve the last profile that was sent to atoms.
    
    Returns:
        (x_um, profile_y) tuple or (None, None) if unsuccessful
    """
    try:
        print(f"Connecting to DMD server at {DMD_SERVER_IP}:{DMD_SERVER_PORT}...")
        client = zerorpc.Client()
        client.connect(f"tcp://{DMD_SERVER_IP}:{DMD_SERVER_PORT}")
        
        print("Testing connection...")
        hello_response = client.hello()
        print(f"Server response: {hello_response}")
        
        print("Retrieving last atoms profile from DMD...")
        result = client.get_last_atoms_profile()
        
        if isinstance(result, dict) and 'x' in result and 'y' in result:
            x_um = np.array(result['x'])
            profile_y = np.array(result['y'])
            print(f"Retrieved profile: {len(x_um)} points")
            return x_um, profile_y
        else:
            print(f"ERROR: Unexpected response format: {type(result)}")
            if isinstance(result, dict):
                print(f"Available keys: {result.keys()}")
            return None, None
        
        client.close()
        
    except Exception as e:
        print(f"ERROR: Failed to get profile from DMD server: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_profile_to_txt(x_um, profile_y, output_filename=None):
    """
    Save profile to a text file.
    
    Args:
        x_um: x-axis values in micrometers
        profile_y: profile values
        output_filename: optional custom filename (default: timestamp-based)
    
    Returns:
        Full path to saved file
    """
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"dmd_profile_{timestamp}.txt"
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Save as two-column text file
    header = "x_um\tprofile_y"
    data = np.column_stack((x_um, profile_y))
    
    np.savetxt(output_path, data, header=header, delimiter='\t', 
               fmt='%.6f', comments='')
    
    print(f"\nProfile saved to: {output_path}")
    print(f"  Columns: x_um, profile_y")
    print(f"  Data points: {len(x_um)}")
    
    return output_path


def plot_profile(x_um, profile_y):
    """
    Plot the DMD profile.
    
    Args:
        x_um: x-axis values in micrometers
        profile_y: profile values
    """
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    
    ax.plot(x_um, profile_y, 'b-', linewidth=2, marker='o', markersize=3, alpha=0.8)
    ax.fill_between(x_um, profile_y, alpha=0.2)
    
    ax.set_xlabel('Position (μm)', fontsize=12)
    ax.set_ylabel('DMD Intensity (a.u.)', fontsize=12)
    ax.set_title('Last DMD Profile Sent to Device', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Add some statistics as text
    mean_val = np.mean(profile_y)
    max_val = np.max(profile_y)
    min_val = np.min(profile_y)
    
    stats_text = f"Mean: {mean_val:.4f}\nMax: {max_val:.4f}\nMin: {min_val:.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8), fontsize=10, family='monospace')
    
    plt.show()


def main():
    """Main execution function."""
    
    print("="*70)
    print("Get Last DMD Profile Script")
    print("="*70)
    
    # Retrieve profile from DMD server
    print()
    x_um, profile_y = get_last_profile_from_dmd()
    
    if x_um is None or profile_y is None:
        print("\nERROR: Could not retrieve profile from DMD server.")
        return
    
    print(f"\nProfile retrieved successfully:")
    print(f"  x range: {x_um.min():.2f} to {x_um.max():.2f} um")
    print(f"  {len(x_um)} data points")
    print(f"  Profile range: {profile_y.min():.4f} to {profile_y.max():.4f}")
    
    # Plot the profile
    print("\nDisplaying profile plot...")
    plot_profile(x_um, profile_y)
    
    # Save to txt file
    print()
    save_profile_to_txt(x_um, profile_y)
    
    print("\n" + "="*70)
    print("Script completed successfully.")
    print("="*70)


if __name__ == '__main__':
    main()
