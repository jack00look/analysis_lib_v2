#!/usr/bin/env python3
"""
View all sigmoid feedback profiles and RMS analysis.

Loads all sigmoid profiles (x vs field) from feedback_sigmoids_list.py 
plus the latest sigmoid from waterfall folder.

Creates 2 figures:
1. All sigmoid profiles overlaid with RMS region shaded
2. RMS scatter plot for each profile in the region

No modifications to any files.

Usage:
    python view_feedback_profiles.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import feedback_automation_config as config
except ImportError:
    print("ERROR: Could not import feedback_automation_config")
    sys.exit(1)


# Configuration
BIAS_FIELD_OFFSET_MG = 129.8
MG_TO_HZ = 2.1e3
MG_TO_UG = 1e3

REGION_START_UM = 900
REGION_END_UM = 1200


def log(msg, level=0):
    """Simple logging function."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    indent = "  " * level
    print(f"[{timestamp}] {indent}{msg}")


def load_sigmoid_txt(file_path):
    """Load 2-column txt file: x_um, y (field value)."""
    try:
        data = np.loadtxt(file_path, comments='#')
        if data.ndim != 2 or data.shape[1] < 2:
            return None, None
        x = data[:, 0]
        y = data[:, 1]
        return x, y
    except Exception as e:
        log(f"ERROR loading {file_path}: {e}", level=1)
        return None, None


def load_sigmoid_list():
    """Load SIGMOID_PROFILES from feedback_sigmoids_list.py"""
    try:
        import importlib.util
        dmd_folder = os.path.dirname(os.path.abspath(__file__))
        list_file = os.path.join(dmd_folder, 'feedback_sigmoids_list.py')
        
        spec = importlib.util.spec_from_file_location("sigmoid_list", list_file)
        if spec is None or spec.loader is None:
            return []
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'SIGMOID_PROFILES'):
            return module.SIGMOID_PROFILES
        else:
            return []
    except Exception as e:
        log(f"ERROR loading sigmoid list: {e}")
        return []


def main():
    log("="*70)
    log("Sigmoid Profiles Viewer")
    log("="*70)
    
    dmd_folder = os.path.dirname(os.path.abspath(__file__))
    sigmoid_folder = os.path.join(dmd_folder, config.MAGNETIZATION_FEEDBACK_FOLDER)
    
    # Load sigmoid list
    sigmoid_list = load_sigmoid_list()
    if not sigmoid_list:
        log("ERROR: No sigmoids in list")
        return False
    
    log(f"Found {len(sigmoid_list)} profiles in feedback_sigmoids_list.py\n")
    
    # Load all profiles
    loaded_data = []
    
    for i, profile_spec in enumerate(sigmoid_list):
        filename = profile_spec['filename']
        description = profile_spec.get('description', filename)
        
        filepath = os.path.join(sigmoid_folder, filename)
        
        if not os.path.exists(filepath):
            log(f"  [{i+1}/{len(sigmoid_list)}] {filename} - NOT FOUND", level=1)
            continue
        
        x, y = load_sigmoid_txt(filepath)
        if x is None:
            log(f"  [{i+1}/{len(sigmoid_list)}] {filename} - ERROR", level=1)
            continue
        
        # Convert to Hz
        y_hz = (y - BIAS_FIELD_OFFSET_MG) * MG_TO_HZ
        
        loaded_data.append({
            'x': x,
            'y': y_hz,
            'filename': filename,
            'label': description,
        })
        
        log(f"  [{i+1}/{len(sigmoid_list)}] {description}")
    
    # Also try to load latest waterfall sigmoid
    waterfall_path = config.NEW_SIGMOID_PATH
    if os.path.exists(waterfall_path):
        x_wf, y_wf = load_sigmoid_txt(waterfall_path)
        if x_wf is not None:
            y_wf_hz = (y_wf - BIAS_FIELD_OFFSET_MG) * MG_TO_HZ
            loaded_data.append({
                'x': x_wf,
                'y': y_wf_hz,
                'filename': os.path.basename(waterfall_path),
                'label': 'Waterfall (New)',
            })
            log(f"  [+] Waterfall (New)")
    
    if not loaded_data:
        log("ERROR: No profiles loaded")
        return False
    
    log(f"\nPlotting {len(loaded_data)} sigmoid profiles\n")
    
    # =========================================================================
    # FIGURE 1: All sigmoid profiles with RMS region shaded
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6), tight_layout=True)
    
    for i, data in enumerate(loaded_data):
        color = f'C{i % 10}'
        
        ax1.plot(data['x'], data['y'], linestyle='-', linewidth=2.3, alpha=0.95,
                label=data['label'], color=color)
        ax1.scatter(data['x'], data['y'], s=10, alpha=0.25, c=color)
    
    ax1.set_title('Sigmoid Feedback Profiles', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Position [μm]', fontsize=11)
    ax1.set_ylabel('Field Detuning [Hz]', fontsize=11)
    
    # Secondary axis: Bias field in mG
    secax1 = ax1.secondary_yaxis(
        'right',
        functions=(
            lambda hz: (hz / MG_TO_HZ) + BIAS_FIELD_OFFSET_MG,
            lambda mg: (mg - BIAS_FIELD_OFFSET_MG) * MG_TO_HZ,
        ),
    )
    secax1.set_ylabel('Bias Field [mG]', fontsize=11)
    
    # Shade RMS region
    region_min = min(REGION_START_UM, REGION_END_UM)
    region_max = max(REGION_START_UM, REGION_END_UM)
    ax1.axvspan(region_min, region_max, color='gray', alpha=0.12, zorder=0)
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # =========================================================================
    # FIGURE 2: RMS scatter plot for each profile in region
    # =========================================================================
    profile_labels = []
    profile_rms_hz = []
    
    for item in loaded_data:
        x = item['x']
        y_hz = item['y']
        mask = (x >= region_min) & (x <= region_max)
        
        if np.count_nonzero(mask) == 0:
            log(f"WARNING: No data in RMS region for {item['label']}", level=1)
            continue
        
        y_region = y_hz[mask]
        rms_hz = np.sqrt(np.mean((y_region - np.mean(y_region)) ** 2))
        
        profile_labels.append(item['label'])
        profile_rms_hz.append(rms_hz)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6), tight_layout=True)
    x_idx = np.arange(len(profile_labels))
    
    ax2.scatter(x_idx, profile_rms_hz, s=100, color='tab:green', edgecolors='k',
               linewidth=0.8, zorder=3)
    
    # Y-axis limits with headroom
    y_min = float(np.min(profile_rms_hz))
    y_max = float(np.max(profile_rms_hz))
    y_span = max(y_max - y_min, 1.0)
    y_bottom = y_min - 0.12 * y_span
    y_top = y_max + 0.20 * y_span
    ax2.set_ylim(y_bottom, y_top)
    
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(profile_labels, rotation=20, ha='right', fontsize=10)
    ax2.set_ylabel('RMS around mean in region [Hz]', fontsize=11)
    ax2.set_xlabel('Sigmoid profile', fontsize=11)
    ax2.set_title(f'Profile RMS around mean in x-region [{region_min:.1f}, {region_max:.1f}] μm',
                 fontsize=12, fontweight='bold')
    
    # Secondary axis: RMS in μG
    secax2 = ax2.secondary_yaxis(
        'right',
        functions=(
            lambda hz_rms: (hz_rms / MG_TO_HZ) * MG_TO_UG,
            lambda ug_rms: (ug_rms / MG_TO_UG) * MG_TO_HZ,
        ),
    )
    secax2.set_ylabel('RMS around mean in region [μG]', fontsize=11)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # FIGURE 3: Feedback effectiveness - error profile correlation
    # =========================================================================
    # Show how well each correction is moving towards the goal
    # For each pair (profile_i, profile_i+1):
    #   Plot: profile_i (mean-subtracted) vs (profile_i+1 - profile_i)
    #   If feedback is working, they should be correlated
    
    if len(loaded_data) >= 2:
        log("\nComputing feedback effectiveness correlations...")
        
        fig3, axes3 = plt.subplots(len(loaded_data)-1, 1, figsize=(10, 3.5*(len(loaded_data)-1)), 
                                   tight_layout=True)
        
        # Handle single subplot case
        if len(loaded_data) == 2:
            axes3 = [axes3]
        
        for pair_idx in range(len(loaded_data) - 1):
            ax = axes3[pair_idx]
            
            profile_i = loaded_data[pair_idx]
            profile_i_plus_1 = loaded_data[pair_idx + 1]
            
            x_i = profile_i['x']
            y_i_hz = profile_i['y']
            
            x_i_plus_1 = profile_i_plus_1['x']
            y_i_plus_1_hz = profile_i_plus_1['y']
            
            # Interpolate profile_i+1 onto profile_i's grid
            interp_func = interp1d(x_i_plus_1, y_i_plus_1_hz, kind='linear', 
                                  bounds_error=False, fill_value=np.nan)
            y_i_plus_1_interp = interp_func(x_i)
            
            # Compute error profile (mean-subtracted) - this is the correction applied
            y_i_error = y_i_hz - np.mean(y_i_hz)
            
            # Compute the difference (change from profile i to i+1)
            y_diff = y_i_plus_1_interp - y_i_hz
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(y_i_error) | np.isnan(y_diff))
            y_i_error_valid = y_i_error[valid_mask]
            y_diff_valid = y_diff[valid_mask]
            
            if len(y_i_error_valid) > 0:
                # Linear regression
                slope, intercept, r_value, p_value, std_err = linregress(y_i_error_valid, y_diff_valid)
                
                # Scatter plot
                ax.scatter(y_i_error_valid, y_diff_valid, s=40, color='purple', alpha=0.5,
                          edgecolors='k', linewidth=0.3)
                
                # Fit line
                x_fit = np.array([y_i_error_valid.min(), y_i_error_valid.max()])
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r-', linewidth=2.5, alpha=0.8,
                       label=f'Linear fit: R²={r_value**2:.4f}, slope={slope:.4f}')
                
                # Diagnostic text for kp assessment
                kp_diagnosis = "GOOD"
                kp_color = 'green'
                if slope > 1.2:
                    kp_diagnosis = f"kp TOO HIGH (slope={slope:.3f} > 1.2)"
                    kp_color = 'red'
                elif slope < 0.8:
                    kp_diagnosis = f"kp TOO LOW (slope={slope:.3f} < 0.8)"
                    kp_color = 'orange'
                else:
                    kp_diagnosis = f"kp GOOD (slope={slope:.3f} ≈ 1.0)"
                
                ax.text(0.05, 0.95, kp_diagnosis, transform=ax.transAxes, 
                       fontsize=11, fontweight='bold', color=kp_color,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.8))
                
                ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
                ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
                
                ax.set_xlabel(f'{profile_i["label"]} error (mean-subtracted) [Hz]', fontsize=10)
                ax.set_ylabel(f'{profile_i_plus_1["label"]} - {profile_i["label"]} [Hz]', fontsize=10)
                ax.set_title(f'Feedback Effectiveness: {profile_i["label"]} → {profile_i_plus_1["label"]}',
                            fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower right', fontsize=9)
    
    # =========================================================================
    # FIGURE 4: Overcorrection detection
    # =========================================================================
    # Check if error profiles oscillate in sign (overcorrection = opposite signs)
    
    log("\nAnalyzing for overcorrection...")
    
    if len(loaded_data) >= 2:
        fig4, ax4 = plt.subplots(figsize=(10, 5), tight_layout=True)
        
        overcorrection_info = []
        
        for pair_idx in range(len(loaded_data) - 1):
            profile_i = loaded_data[pair_idx]
            profile_i_plus_1 = loaded_data[pair_idx + 1]
            
            x_i = profile_i['x']
            y_i_hz = profile_i['y']
            
            x_i_plus_1 = profile_i_plus_1['x']
            y_i_plus_1_hz = profile_i_plus_1['y']
            
            # Interpolate profile_i+1 onto profile_i's grid
            interp_func = interp1d(x_i_plus_1, y_i_plus_1_hz, kind='linear', 
                                  bounds_error=False, fill_value=np.nan)
            y_i_plus_1_interp = interp_func(x_i)
            
            # Error profiles (mean-subtracted)
            y_i_error = y_i_hz - np.mean(y_i_hz)
            y_i_plus_1_error = y_i_plus_1_interp - np.mean(y_i_plus_1_interp)
            
            # Remove NaNs
            valid_mask = ~(np.isnan(y_i_error) | np.isnan(y_i_plus_1_error))
            y_i_error_valid = y_i_error[valid_mask]
            y_i_plus_1_error_valid = y_i_plus_1_error[valid_mask]
            
            # Check sign correlation
            sign_correlation = np.mean(np.sign(y_i_error_valid) * np.sign(y_i_plus_1_error_valid))
            
            overcorrection_info.append({
                'label': f'{profile_i["label"]} → {profile_i_plus_1["label"]}',
                'sign_corr': sign_correlation,
                'pair_idx': pair_idx,
            })
        
        x_oc = np.arange(len(overcorrection_info))
        sign_corrs = [oc['sign_corr'] for oc in overcorrection_info]
        colors_oc = ['green' if sc > 0.5 else 'orange' if sc > -0.2 else 'red' for sc in sign_corrs]
        labels_oc = [oc['label'] for oc in overcorrection_info]
        
        ax4.scatter(x_oc, sign_corrs, s=150, color=colors_oc, edgecolors='k', linewidth=1)
        ax4.axhline(0.5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (> 0.5)')
        ax4.axhline(-0.2, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Fair (-0.2 to 0.5)')
        ax4.axhline(-1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Overcorrecting (< -0.2)')
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax4.set_xticks(x_oc)
        ax4.set_xticklabels(labels_oc, rotation=20, ha='right', fontsize=10)
        ax4.set_ylabel('Sign Correlation', fontsize=11)
        ax4.set_xlabel('Profile Pair', fontsize=11)
        ax4.set_title('Overcorrection Detection (Sign Oscillation of Error Profiles)',
                     fontsize=12, fontweight='bold')
        ax4.set_ylim(-1.1, 1.1)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation text
        for idx, oc_info in enumerate(overcorrection_info):
            sc = oc_info['sign_corr']
            if sc > 0.5:
                text = "Stable"
                color = 'green'
            elif sc > -0.2:
                text = "Slightly oscillating"
                color = 'orange'
            else:
                text = "OVERCORRECTING!"
                color = 'red'
            ax4.text(idx, sc + 0.15, text, ha='center', fontsize=9, fontweight='bold', color=color)
    
    # =========================================================================
    # FIGURE 5: Frequency analysis - low vs high frequency performance
    # =========================================================================
    # Split into low frequency (macro structures) and high frequency (details)
    
    log("\nAnalyzing feedback at different frequency scales...")
    
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)
    
    profile_labels_freq = []
    rms_low_freq = []
    rms_high_freq = []
    
    for item in loaded_data:
        x = item['x']
        y_hz = item['y']
        
        profile_labels_freq.append(item['label'])
        
        # Compute FFT
        N = len(y_hz)
        dt = x[1] - x[0] if len(x) > 1 else 1.0
        freqs = np.fft.fftfreq(N, dt)
        
        # Only use positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        fft_result = np.fft.fft(np.array(y_hz))
        fft_mag = np.abs(fft_result)[pos_mask]
        
        if len(freqs_pos) > 2:
            # Split at median frequency
            median_freq = np.median(freqs_pos)
            
            # Low frequency (macro structures)
            low_freq_mask = freqs_pos < median_freq
            rms_lf = np.sqrt(np.mean(fft_mag[low_freq_mask] ** 2)) if np.any(low_freq_mask) else 0
            rms_low_freq.append(rms_lf)
            
            # High frequency (details)
            high_freq_mask = freqs_pos >= median_freq
            rms_hf = np.sqrt(np.mean(fft_mag[high_freq_mask] ** 2)) if np.any(high_freq_mask) else 0
            rms_high_freq.append(rms_hf)
        else:
            rms_low_freq.append(0)
            rms_high_freq.append(0)
    
    # Plot 5a: Low vs High frequency RMS for each profile
    x_idx_freq = np.arange(len(profile_labels_freq))
    width = 0.35
    
    ax5a.bar(x_idx_freq - width/2, rms_low_freq, width, label='Low Freq (Macro)',
            color='blue', alpha=0.7, edgecolor='k')
    ax5a.bar(x_idx_freq + width/2, rms_high_freq, width, label='High Freq (Details)',
            color='red', alpha=0.7, edgecolor='k')
    
    ax5a.set_xticks(x_idx_freq)
    ax5a.set_xticklabels(profile_labels_freq, rotation=20, ha='right', fontsize=10)
    ax5a.set_ylabel('RMS Amplitude (frequency domain)', fontsize=11)
    ax5a.set_xlabel('Sigmoid profile', fontsize=11)
    ax5a.set_title('Feedback Performance: Macro vs Detail Structures', fontsize=12, fontweight='bold')
    ax5a.legend(fontsize=10)
    ax5a.grid(True, alpha=0.3, axis='y')
    
    # Plot 5b: Ratio of high to low frequency (shows if details are being corrected)
    freq_ratios = [rms_high_freq[i] / (rms_low_freq[i] + 1e-10) for i in range(len(profile_labels_freq))]
    colors_freq = ['green' if fr < 0.5 else 'orange' if fr < 1.0 else 'red' for fr in freq_ratios]
    
    ax5b.scatter(x_idx_freq, freq_ratios, s=150, color=colors_freq, edgecolors='k', linewidth=1)
    ax5b.axhline(0.5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (< 0.5)')
    ax5b.axhline(1.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Fair (0.5-1.0)')
    ax5b.axhline(1.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Poor (> 1.0)')
    
    ax5b.set_xticks(x_idx_freq)
    ax5b.set_xticklabels(profile_labels_freq, rotation=20, ha='right', fontsize=10)
    ax5b.set_ylabel('High Freq / Low Freq Ratio', fontsize=11)
    ax5b.set_xlabel('Sigmoid profile', fontsize=11)
    ax5b.set_title('Detail-to-Structure Ratio (Lower = Better Correction)', fontsize=12, fontweight='bold')
    ax5b.legend(fontsize=10)
    ax5b.grid(True, alpha=0.3, axis='y')
    
    plt.show()
    
    # Print summary
    log("\n" + "="*70)
    log("SUMMARY")
    log("="*70)
    for i, (label, rms) in enumerate(zip(profile_labels, profile_rms_hz)):
        log(f"  {i+1}. {label}: RMS = {rms:.3f} Hz ({rms/MG_TO_HZ*MG_TO_UG:.3f} μG)")
    log("="*70 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
