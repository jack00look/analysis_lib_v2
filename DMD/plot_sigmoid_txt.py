import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
PLOT_FILES = [
    {
        'filename': 'sigmoid_center_interpolation3.txt',
        'line_style': 'm-',
        'scatter_color': 'k',
        'label': 'Sigmoid interpolation 1',
        'points_label': 'Points from txt',
    },
    {
        'filename': '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall/sigmoid_center_interpolation.txt',
        'line_style': 'c-',
        'scatter_color': 'b',
        'label': 'Sigmoid interpolation 2',
        'points_label': 'Points from txt 2',
    },
]

# Backward-compatible single-file config fallback.
TXT_FILENAME = 'sigmoid_center_interpolation.txt'
TXT_FILENAME_2 = 'sigmoid_center_interpolation2.txt'


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
    for spec in plot_specs:
        txt_path = os.path.join(script_dir, spec['filename'])
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f'Txt file not found: {txt_path}')

        x, y = load_sigmoid_txt(txt_path)
        loaded_data.append({'x': x, 'y': y, 'spec': spec})
        ax.plot(x, y, spec['line_style'], linewidth=2, label=spec['label'])
        scatter_kwargs = {'s': 12, 'alpha': 0.35, 'label': spec['points_label']}
        if spec['scatter_color'] is not None:
            scatter_kwargs['c'] = spec['scatter_color']
        ax.scatter(x, y, **scatter_kwargs)

    ax.set_title('Sigmoid Center Interpolation (from txt)')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('sigmoid center')
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

    plt.show()


if __name__ == '__main__':
    main()
