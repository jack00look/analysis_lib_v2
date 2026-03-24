import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
PLOT_FILES = [
    {
        'filename': 'sigmoid_center_interpolation1.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': 'Initial Sigmoid Profile',
    },
    {
        'filename': 'sigmoid_center_interpolation2.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '1st Iteration',
    },
    {
        'filename': 'sigmoid_center_interpolation3.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '2nd Iteration',
    },
    {
        'filename': 'sigmoid_center_interpolation4.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '3rd Iteration',
    },
    {
        'filename': 'sigmoid_center_interpolation5.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '4th Iteration',
    },
    {
        'filename': 'sigmoid_center_interpolation6.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '5th Iteration',
    },
    {
        'filename': 'sigmoid_center_interpolation7.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '6th Iteration',
    },
    {
        'filename': 'sigmoid_center_interpolation8.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '7th Iteration',
    },
    {
        'filename': 'sigmoid_center_interpolation9.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '8th Iteration',
    },
    {
        'filename': 'sigmoid_center_interpolation10.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '9th Iteration',
    },
    {
        'filename': 'sigmoid_center_interpolation11.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '10th Iteration',
    },
    {
        'filename': 'sigmoid_center_interpolation12.txt',
        'line_style': '-',
        'line_color': 'k',
        'scatter_color': '0.15',
        'label': '11th Iteration',
    },
    {
        'filename': '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/sigmoid_center_interpolation.txt',
        'line_style': 'c-',
        'scatter_color': 'b',
        'label': 'Sigmoid interpolation New',
    },
]

# Backward-compatible single-file config fallback.
TXT_FILENAME = 'sigmoid_center_interpolation.txt'
TXT_FILENAME_2 = 'sigmoid_center_interpolation2.txt'


# Conversion between bias field (mG) and detuning (Hz)
BIAS_FIELD_OFFSET_MG = 129.8
MG_TO_HZ = 2.1e3
MG_TO_UG = 1e3

# Region (in x [um]) used for RMS calculation in the third figure.
REGION_START_UM = 950
REGION_END_UM = 1150


def load_sigmoid_txt(file_path):
    """Load 2-column txt file: x_um, y."""
    data = np.loadtxt(file_path, comments='#')
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f'Expected at least 2 columns in {file_path}, got shape {data.shape}')
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def get_plot_specs():
    """Return normalized plot specifications."""
    if 'PLOT_FILES' in globals() and PLOT_FILES is not None:
        specs = PLOT_FILES
    else:
        specs = []
        if 'TXT_FILENAME' in globals() and TXT_FILENAME:
            specs.append({
                'filename': TXT_FILENAME,
                'line_style': 'm-',
                'scatter_color': 'k',
                'label': 'Sigmoid interpolation',
                'points_label': 'Points from txt',
            })
        if 'TXT_FILENAME_2' in globals() and TXT_FILENAME_2:
            specs.append({
                'filename': TXT_FILENAME_2,
                'line_style': 'c-',
                'scatter_color': 'b',
                'label': 'Sigmoid interpolation 2',
                'points_label': 'Points from txt 2',
            })

    normalized_specs = []
    for index, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            raise ValueError(f'PLOT_FILES[{index - 1}] must be a dict.')
        if 'filename' not in spec:
            raise ValueError(f'PLOT_FILES[{index - 1}] is missing "filename".')

        normalized_specs.append({
            'filename': spec['filename'],
            'line_style': spec.get('line_style', '-'),
            'line_color': spec.get('line_color', None),
            'scatter_color': spec.get('scatter_color', None),
            'label': spec.get('label', f'Sigmoid interpolation {index}'),
            'points_label': spec.get('points_label', f'Points from txt {index}'),
        })

    if len(normalized_specs) == 0:
        raise ValueError('No plot files configured.')

    return normalized_specs


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_specs = get_plot_specs()

    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)

    loaded_data = []
    for i, spec in enumerate(plot_specs):
        txt_path = os.path.join(script_dir, spec['filename'])
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f'Txt file not found: {txt_path}')

        x, y = load_sigmoid_txt(txt_path)
        y = (y - BIAS_FIELD_OFFSET_MG) * MG_TO_HZ
        loaded_data.append({'x': x, 'y': y, 'spec': spec})

        color = f'C{i % 10}'

        plot_kwargs = {
            'linestyle': '-',
            'linewidth': 2.3,
            'alpha': 0.95,
            'label': spec['label'],
            'color': color,
        }
        ax.plot(x, y, **plot_kwargs)

        scatter_kwargs = {'s': 10, 'alpha': 0.25, 'c': color}
        ax.scatter(x, y, **scatter_kwargs)

    ax.set_title('Magnetization Feedback')
    ax.set_xlabel('x [um]')
    ax.set_ylabel('Field Detuning [Hz]')

    # Secondary y-axis: show corresponding bias field in mG.
    secax = ax.secondary_yaxis(
        'right',
        functions=(
            lambda hz: (hz / MG_TO_HZ) + BIAS_FIELD_OFFSET_MG,
            lambda mg: (mg - BIAS_FIELD_OFFSET_MG) * MG_TO_HZ,
        ),
    )
    secax.set_ylabel('Bias Field [mG]')

    # Show selected RMS region on the main plot.
    region_min = min(REGION_START_UM, REGION_END_UM)
    region_max = max(REGION_START_UM, REGION_END_UM)
    ax.axvspan(region_min, region_max, color='gray', alpha=0.12, zorder=0)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # If we have at least 2 files, create a second figure showing difference vs residual
    if len(loaded_data) >= 2:
        from scipy.interpolate import interp1d
        from scipy.stats import linregress
        
        # Use first profile as reference
        x1 = loaded_data[0]['x']
        y1 = loaded_data[0]['y']
        
        # Load second profile
        x2 = loaded_data[1]['x']
        y2 = loaded_data[1]['y']
        
        # Interpolate y2 onto x1 grid for comparison
        interp_func = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value='extrapolate')
        y2_interp = interp_func(x1)
        
        # Compute residuals and differences
        y1_residual = y1 - np.mean(y1)
        y_diff = y2_interp - y1
        
        # Compute correlation
        slope, intercept, r_value, p_value, std_err = linregress(y1_residual, y_diff)
        
        # Create second figure with scatter plot and fit line
        fig2, ax2 = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax2.scatter(y1_residual, y_diff, s=50, color='purple', alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Plot fit line
        x_fit = np.array([y1_residual.min(), y1_residual.max()])
        y_fit = slope * x_fit + intercept
        ax2.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Linear fit: R²={r_value**2:.4f}')
        
        ax2.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        ax2.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax2.set_xlabel(f'{loaded_data[0]["spec"]["label"]} - mean({loaded_data[0]["spec"]["label"]})')
        ax2.set_ylabel(f'{loaded_data[1]["spec"]["label"]} - {loaded_data[0]["spec"]["label"]}')
        ax2.set_title(f'Difference vs Residual (R²={r_value**2:.4f}, slope={slope:.4f})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    # Third figure: RMS in chosen x-region for each sigmoid profile.
    profile_labels = []
    profile_rms_hz = []
    for item in loaded_data:
        x = item['x']
        y_hz = item['y']
        mask = (x >= region_min) & (x <= region_max)
        if np.count_nonzero(mask) == 0:
            raise ValueError(
                f'No data points found in RMS region [{region_min}, {region_max}] um '
                f'for profile "{item["spec"]["label"]}".'
            )

        y_region = y_hz[mask]
        rms_hz = np.sqrt(np.mean((y_region - np.mean(y_region)) ** 2))
        profile_labels.append(item['spec']['label'])
        profile_rms_hz.append(rms_hz)

    fig3, ax3 = plt.subplots(figsize=(8, 5), tight_layout=True)
    x_idx = np.arange(len(profile_labels))
    ax3.scatter(x_idx, profile_rms_hz, s=70, color='tab:green', edgecolors='k', linewidth=0.6)

    # Keep labels inside axes by setting explicit y-limits with headroom.
    y_min = float(np.min(profile_rms_hz))
    y_max = float(np.max(profile_rms_hz))
    y_span = max(y_max - y_min, 1.0)
    y_bottom = y_min - 0.12 * y_span
    y_top = y_max + 0.20 * y_span
    ax3.set_ylim(y_bottom, y_top)

    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(profile_labels, rotation=20, ha='right')
    ax3.set_ylabel('RMS around mean in region [Hz]')
    ax3.set_xlabel('Sigmoid profile')
    ax3.set_title(f'Profile RMS around mean in x-region [{region_min:.1f}, {region_max:.1f}] um')

    # Secondary y-axis: RMS in uG (mean-subtracted, so no offset term).
    secax3 = ax3.secondary_yaxis(
        'right',
        functions=(
            lambda hz_rms: (hz_rms / MG_TO_HZ) * MG_TO_UG,
            lambda ug_rms: (ug_rms / MG_TO_UG) * MG_TO_HZ,
        ),
    )
    secax3.set_ylabel('RMS around mean in region [uG]')

    ax3.grid(True, alpha=0.3)

    plt.show()


if __name__ == '__main__':
    main()
