import h5py
import importlib
import matplotlib.pyplot as plt
import numpy as np
from fitting.models1D import BimodalBose1DModel2Centers, Gaussian1DModel
from lmfit_lyse_interface import parameters_dict_with_errors
import traceback
import time
import subprocess
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import f
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

cameras = camera_settings.cameras

# Configuration parameters
GAUSSIAN_BLUR_UM = 4.0  # Gaussian blur width in micrometers
VERTICAL_LINE_START_UM = 900.0  # Starting position for vertical lines in micrometers
VERTICAL_LINE_END_UM = 1200.0  # Ending position for vertical lines in micrometers
VERTICAL_LINE_SPACING_UM = 5.0  # Spacing between vertical lines in micrometers
F_TEST_THRESHOLD = 20.0  # F-test threshold for choosing best fit model (lowered for speed)

# Fit bounds (in micrometers)
TF_CENTER_MIN_UM = 60.0  # Minimum TF center position
TF_CENTER_MAX_UM = 80.0  # Maximum TF center position
GAUSS_CENTER_MIN_UM = 60.0  # Minimum Gaussian center position
GAUSS_CENTER_MAX_UM = 80.0  # Maximum Gaussian center position
TF_RADIUS_MIN_UM = 200.0  # Minimum TF radius
TF_RADIUS_MAX_UM = 500.0  # Maximum TF radius
GAUSS_WIDTH_MIN_UM = 100.0  # Minimum Gaussian width (sigma)
GAUSS_WIDTH_MAX_UM = 500.0  # Maximum Gaussian width (sigma)


def reconstruct_1d_profiles(fit_results_dict, y_vals_full, x_vals_full, model_selections):
    """
    Reconstruct 2D images from 1D fit results.
    
    Returns:
    --------
    tf_2d : 2D array
        Thomas-Fermi (condensed) component 2D image
    thermal_2d : 2D array
        Thermal component 2D image
    """
    n_y = len(y_vals_full)
    n_x = len(x_vals_full)
    
    tf_2d = np.zeros((n_y, n_x))
    thermal_2d = np.zeros((n_y, n_x))
    
    for x_um, result in fit_results_dict.items():
        model_name = result['model']
        x_pos_m = x_um / 1e6
        # Find closest x index
        x_idx = np.argmin(np.abs(x_vals_full - x_pos_m))
        
        if x_idx >= n_x:
            x_idx = n_x - 1
        
        best_out = result.get('fit_result')
        if best_out is None:
            continue
        
        try:
            # Compute fit profile
            y_fit = best_out.best_fit
            
            if model_name == 'bimodal':
                # Extract TF (condensed) and thermal components from bimodal fit
                params = best_out.params.valuesdict()
                # BimodalBose model has center1 (TF), center2 (thermal)
                # For bimodal: compute TF component using model
                try:
                    # Extract bimodal parameters and reconstruct components
                    # This is simplified - you may need to adjust based on your model structure
                    tf_2d[:, x_idx] = y_fit  # Full bimodal profile as TF for now
                    thermal_2d[:, x_idx] = 0  # Placeholder
                except:
                    pass
            else:
                # Gaussian - all thermal
                thermal_2d[:, x_idx] = y_fit
                tf_2d[:, x_idx] = 0
        except:
            pass
    
    return tf_2d, thermal_2d

def choose_best_model_1D(y_vals, x_vals, prefix_gauss, prefix_bimodal, ax):
    """
    Fit 1D data with both Gaussian and BimodalBose1DModel2Centers models.
    Use F-test to choose the best model.
    
    Returns:
    --------
    best_model_name : str
        Name of the best model ('gaussian' or 'bimodal')
    best_out : lmfit result object
        Fit results from the best model
    F_test_value : float
        F-test statistic
    """
    try:
        # Fit with Gaussian model
        gaussian_model = Gaussian1DModel(prefix=prefix_gauss)
        try:
            p0_gauss = gaussian_model.guess(y_vals, X=x_vals)
        except Exception as e:
            print(f"  Warning: Gaussian guess failed, skipping this line: {e}")
            return None, None, None
        
        if p0_gauss is None or len(p0_gauss) == 0:
            print(f"  Warning: Gaussian guess returned empty parameters")
            return None, None, None
        
        try:
            out_gauss = gaussian_model.fit(y_vals, p0_gauss, X=x_vals)
        except Exception as e:
            print(f"  Warning: Gaussian fit failed: {e}")
            return None, None, None
        
        chisqr_gauss = out_gauss.chisqr
        nfree_gauss = out_gauss.nfree
        
        # Fit with BimodalBose1DModel2Centers
        bimodal_model = BimodalBose1DModel2Centers(prefix=prefix_bimodal)
        try:
            p0_bimodal = bimodal_model.guess(y_vals, X=x_vals)
        except Exception as e:
            print(f"  Warning: Bimodal guess failed: {e}")
            # Fall back to Gaussian if bimodal guess fails
            best_model_name = 'gaussian'
            best_out = out_gauss
            best_model = gaussian_model
            ax.plot(x_vals, y_vals, 'o', label='data', markersize=4)
            ax.plot(x_vals, gaussian_model.eval(out_gauss.params, X=x_vals), 'g-', linewidth=2, label='gaussian (bimodal failed)')
            ax.legend(fontsize=8)
            ax.set_title('Bimodal fit unavailable')
            return best_model_name, best_out, np.inf
        
        if p0_bimodal is None or len(p0_bimodal) == 0:
            print(f"  Warning: Bimodal guess returned empty parameters")
            # Fall back to Gaussian
            best_model_name = 'gaussian'
            best_out = out_gauss
            best_model = gaussian_model
            ax.plot(x_vals, y_vals, 'o', label='data', markersize=4)
            ax.plot(x_vals, gaussian_model.eval(out_gauss.params, X=x_vals), 'g-', linewidth=2, label='gaussian (bimodal empty)')
            ax.legend(fontsize=8)
            ax.set_title('Bimodal parameters empty')
            return best_model_name, best_out, np.inf
        
        try:
            out_bimodal = bimodal_model.fit(y_vals, p0_bimodal, X=x_vals)
        except Exception as e:
            print(f"  Warning: Bimodal fit failed: {e}")
            # Fall back to Gaussian if bimodal fit fails
            best_model_name = 'gaussian'
            best_out = out_gauss
            best_model = gaussian_model
            ax.plot(x_vals, y_vals, 'o', label='data', markersize=4)
            ax.plot(x_vals, gaussian_model.eval(out_gauss.params, X=x_vals), 'g-', linewidth=2, label='gaussian (bimodal fit failed)')
            ax.legend(fontsize=8)
            ax.set_title('Bimodal fit failed')
            return best_model_name, best_out, np.inf
        
        chisqr_bimodal = out_bimodal.chisqr
        nfree_bimodal = out_bimodal.nfree
        
        # F-test: compare Gaussian (simpler) vs Bimodal (more complex)
        # If bimodal is significantly better, F_test will be large
        if nfree_bimodal > 0 and chisqr_bimodal > 0:
            F_test = (chisqr_gauss - chisqr_bimodal) / (nfree_gauss - nfree_bimodal) / (chisqr_bimodal / nfree_bimodal)
        else:
            F_test = 0
        
        # Choose model based on F-test threshold
        if F_test > F_TEST_THRESHOLD:
            best_model_name = 'bimodal'
            best_out = out_bimodal
            best_model = bimodal_model
        else:
            best_model_name = 'gaussian'
            best_out = out_gauss
            best_model = gaussian_model
        
        # Plot both models for reference
        ax.plot(x_vals, y_vals, 'o', label='data', markersize=4)
        ax.plot(x_vals, gaussian_model.eval(out_gauss.params, X=x_vals), 'b--', label='gaussian', alpha=0.7)
        ax.plot(x_vals, bimodal_model.eval(out_bimodal.params, X=x_vals), 'r-', label='bimodal', alpha=0.7)
        ax.plot(x_vals, best_model.eval(best_out.params, X=x_vals), 'g-', linewidth=2, label=f'best fit ({best_model_name})')
        ax.legend(fontsize=8)
        ax.set_title(f'F_test={F_test:.2f}, threshold={F_TEST_THRESHOLD}')
        
        return best_model_name, best_out, F_test
        
    except Exception as e:
        print(f"Error in fitting vertical line: {e}")
        print(traceback.format_exc())
        return None, None, None


def choose_best_model_1D_fast(y_vals, x_vals, prefix_gauss, prefix_bimodal, px_size=1.0):
    """
    Fast version of model selection - uses leastsq method with reduced iterations.
    Returns only the result, no plotting (plotting done after parallel execution).
    
    Returns:
    --------
    best_model_name : str
        Name of the best model ('gaussian' or 'bimodal')
    best_out : lmfit result object
        Fit results from the best model
    F_test_value : float
        F-test statistic
    """
    try:
        # Fit with Gaussian model - faster method
        gaussian_model = Gaussian1DModel(prefix=prefix_gauss, pixel_size=px_size)
        try:
            p0_gauss = gaussian_model.guess(y_vals, X=x_vals)
        except Exception as e:
            return None, None, None
        
        if p0_gauss is None or len(p0_gauss) == 0:
            return None, None, None
        
        try:
            out_gauss = gaussian_model.fit(y_vals, p0_gauss, X=x_vals, method='leastsq', max_nfev=300)
        except Exception as e:
            return None, None, None
        
        chisqr_gauss = out_gauss.chisqr
        nfree_gauss = out_gauss.nfree
        
        # Fit with BimodalBose1DModel2Centers - faster method
        bimodal_model = BimodalBose1DModel2Centers(prefix=prefix_bimodal, pixel_size=px_size)
        try:
            p0_bimodal = bimodal_model.guess(y_vals, X=x_vals)
        except Exception as e:
            # Fall back to Gaussian
            return 'gaussian', out_gauss, np.inf
        
        if p0_bimodal is None or len(p0_bimodal) == 0:
            return 'gaussian', out_gauss, np.inf
        
        try:
            out_bimodal = bimodal_model.fit(y_vals, p0_bimodal, X=x_vals, method='leastsq', max_nfev=300)
        except Exception as e:
            # Fall back to Gaussian
            return 'gaussian', out_gauss, np.inf
        
        chisqr_bimodal = out_bimodal.chisqr
        nfree_bimodal = out_bimodal.nfree
        
        # F-test with safety checks
        if nfree_bimodal > 0 and chisqr_bimodal > 0:
            F_test = (chisqr_gauss - chisqr_bimodal) / (nfree_gauss - nfree_bimodal) / (chisqr_bimodal / nfree_bimodal)
        else:
            F_test = 0
        
        # Choose model based on F-test threshold
        if F_test > F_TEST_THRESHOLD:
            best_model_name = 'bimodal'
            best_out = out_bimodal
        else:
            best_model_name = 'gaussian'
            best_out = out_gauss
        
        return best_model_name, best_out, F_test
        
    except Exception as e:
        return None, None, None


def fit_single_line_parallel(line_data, y_vals_full, line_idx, px_size=1.0):
    """
    Fit a single vertical line - designed for parallel execution.
    
    Returns:
    --------
    tuple: (line_idx, line_name, best_model_name, best_out, F_test, x_um, y_vals_1D, gaussian_out, bimodal_out)
    """
    try:
        vertical_line = line_data['data']
        x_um = line_data['x_position_um']
        
        # Use y-axis values for fitting
        x_vals_1D = y_vals_full
        y_vals_1D = vertical_line
        
        # Fit with both models using fast method
        prefix_gauss = f'line{line_idx}_gauss_'
        prefix_bimodal = f'line{line_idx}_bimodal_'
        
        best_model_name, best_out, F_test = choose_best_model_1D_fast(
            y_vals_1D, x_vals_1D, prefix_gauss, prefix_bimodal, px_size
        )
        
        # Also get individual model results for plotting
        gaussian_model = Gaussian1DModel(prefix=prefix_gauss, pixel_size=px_size)
        bimodal_model = BimodalBose1DModel2Centers(prefix=prefix_bimodal, pixel_size=px_size)
        
        try:
            p0_gauss = gaussian_model.guess(y_vals_1D, X=x_vals_1D)
            out_gauss = gaussian_model.fit(y_vals_1D, p0_gauss, X=x_vals_1D, method='leastsq', max_nfev=300)
        except:
            out_gauss = None
        
        try:
            p0_bimodal = bimodal_model.guess(y_vals_1D, X=x_vals_1D)
            out_bimodal = bimodal_model.fit(y_vals_1D, p0_bimodal, X=x_vals_1D, method='leastsq', max_nfev=300)
        except:
            out_bimodal = None
        
        return (line_idx, best_model_name, best_out, F_test, x_um, y_vals_1D, x_vals_1D, out_gauss, out_bimodal)
    
    except Exception as e:
        print(f"Error in parallel fit line {line_idx}: {e}")
        return (line_idx, None, None, None, None, None, None, None, None)


def compute_cached_initial_guesses(n2D_blurred, y_vals_full, x_indices_subset):
    """
    Pre-compute initial guess parameters for Gaussian and BimodalBose models
    for a subset of x positions. This caches guesses to speed up fitting.
    
    Returns:
    --------
    cache_gauss : dict
        Cached Gaussian initial parameters indexed by x_pixel
    cache_bimodal : dict
        Cached BimodalBose initial parameters indexed by x_pixel
    """
    cache_gauss = {}
    cache_bimodal = {}
    
    try:
        gaussian_model_ref = Gaussian1DModel(prefix='ref_gauss_')
        bimodal_model_ref = BimodalBose1DModel2Centers(prefix='ref_bimodal_')
        
        # Sample every 3rd line to compute averaged initial guesses
        sample_indices = x_indices_subset[::max(1, len(x_indices_subset)//3)]
        if len(sample_indices) == 0:
            sample_indices = x_indices_subset
        
        averaged_y_vals = None
        for x_pix in sample_indices:
            vertical_line = n2D_blurred[:, x_pix]
            if averaged_y_vals is None:
                averaged_y_vals = vertical_line.copy()
            else:
                averaged_y_vals += vertical_line
        
        if averaged_y_vals is not None:
            averaged_y_vals /= len(sample_indices)
            
            # Compute initial guesses from averaged data
            p0_gauss_avg = gaussian_model_ref.guess(averaged_y_vals, X=y_vals_full)
            p0_bimodal_avg = bimodal_model_ref.guess(averaged_y_vals, X=y_vals_full)
            
            # Store the parameter values (not the full Parameters object with prefix)
            gauss_vals = p0_gauss_avg.valuesdict()
            bimodal_vals = p0_bimodal_avg.valuesdict()
            
            # Cache for all lines
            for x_pix in x_indices_subset:
                cache_gauss[x_pix] = gauss_vals.copy()
                cache_bimodal[x_pix] = bimodal_vals.copy()
    
    except Exception as e:
        print(f"Warning: Could not pre-compute cached guesses: {e}")
    
    return cache_gauss, cache_bimodal


def show_local_thermal_fraction(h5file, show=True):
    """
    Show local thermal fraction analysis from 2D OD images.
    
    For each 2D image:
    1. Apply Gaussian blur along x-direction
    2. Extract vertical lines every 100 um
    3. Fit each line with BimodalBose1DModel2Centers and Gaussian1DModel
    4. Choose best fit using F-test criterion
    5. Plot results
    
    Parameters:
    -----------
    h5file : h5py file object
        HDF5 file containing the image data
    show : bool, optional
        Whether to display plots (default: True)
    
    Returns:
    --------
    infos_analysis : dict
        Dictionary containing analysis results and metadata
    """
    infos_analysis = {}
    
    try:
        camera_loop_start = time.time()
        for cam_entry in cameras:
            cam = cameras[cam_entry]

            dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file, cam)
            if dict_images is None:
                continue

            images = dict_images['images']
            infos_images = dict_images['infos']
            axes = dict_images['axes']

            keys = list(images.keys())
            if len(keys) == 0:
                continue
            
            # Filter out SVD keys for now, process normal OD images
            normal_keys = [key for key in keys if 'SVD' not in key]
            L = len(normal_keys)
            
            # Dictionary to store fit results per key for second figure
            fit_results_by_key = {}

            # Get axes information
            x_vals_full = axes[list(axes.keys())[0]]  # x-axis values in meters
            y_vals_full = axes[list(axes.keys())[1]]  # y-axis values in meters
            px_size = cam['px_size']
            
            # Convert blur width and line spacing from micrometers to pixels
            blur_sigma_um = GAUSSIAN_BLUR_UM  # micrometers
            blur_sigma_pixels = blur_sigma_um / 1e6 / px_size  # convert to pixels
            
            line_spacing_um = VERTICAL_LINE_SPACING_UM  # micrometers
            line_spacing_pixels = int(line_spacing_um / 1e6 / px_size)
            
            print(f"Camera: {cam['name']}")
            print(f"  Blur sigma: {blur_sigma_pixels:.2f} pixels ({blur_sigma_um} um)")
            print(f"  Line spacing: {line_spacing_pixels} pixels ({line_spacing_um} um)")

            # Create figure for thermal fraction analysis
            # Show 2D image with F-test results and BEC fraction for each key (m1, m2, etc.)
            fig, ax = plt.subplots(L, 3, figsize=(18, 4*L), constrained_layout=True)
            if L == 1:
                ax = ax.reshape(1, -1)

            for i, key in enumerate(normal_keys):
                print(f'\nProcessing key: {key}')
                
                # Get the 2D image
                n2D = images[key]
                
                # Apply Gaussian blur along x-direction (axis=1)
                n2D_blurred = gaussian_filter(n2D, sigma=(0, blur_sigma_pixels))
                
                # Extract vertical lines using configuration parameters
                start_pos_um = VERTICAL_LINE_START_UM
                end_pos_um = VERTICAL_LINE_END_UM
                step_um = VERTICAL_LINE_SPACING_UM
                target_positions_um = np.arange(start_pos_um, end_pos_um + step_um, step_um).tolist()
                target_positions_m = np.array(target_positions_um) / 1e6
                
                # Find the closest pixel indices for each target position
                x_pixel_indices = []
                for target_pos in target_positions_m:
                    if target_pos >= x_vals_full[0] and target_pos <= x_vals_full[-1]:
                        closest_idx = np.argmin(np.abs(x_vals_full - target_pos))
                        x_pixel_indices.append(closest_idx)
                
                x_pixel_indices = np.array(x_pixel_indices)
                
                n_pixels_x = n2D_blurred.shape[1]
                print(f"  Extracting {len(x_pixel_indices)} vertical lines at positions: {target_positions_um[:len(x_pixel_indices)]} um")
                
                results_lines = {}
                for line_idx, x_pix in enumerate(x_pixel_indices):
                    x_pix = min(x_pix, n_pixels_x - 1)  # Ensure within bounds
                    vertical_line = n2D_blurred[:, x_pix]
                    x_um = x_vals_full[x_pix] * 1e6
                    
                    results_lines[f'line_{line_idx}'] = {
                        'x_position_um': x_um,
                        'x_pixel': x_pix,
                        'data': vertical_line
                    }
                
                # Dictionary to store fit results for this key
                fit_results_dict = {}
                
                
                # Parallel fitting using ProcessPoolExecutor (true parallelism)
                num_workers = min(8, max(1, multiprocessing.cpu_count() - 1))
                print(f"  Parallel fitting {len(results_lines)} lines with {num_workers} workers...")
                fit_start = time.time()
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    future_to_line = {
                        executor.submit(fit_single_line_parallel, line_data, y_vals_full, idx, px_size): line_name
                        for idx, (line_name, line_data) in enumerate(results_lines.items())
                    }
                    
                    completed = 0
                    for future in as_completed(future_to_line):
                        line_name = future_to_line[future]
                        completed += 1
                        
                        try:
                            line_idx, best_model_name, best_out, F_test, x_um, y_vals_1D, x_vals_1D, out_gauss, out_bimodal = future.result()
                            
                            if best_out is not None:
                                # Compute and store the fitted profile
                                y_fit = best_out.eval(params=best_out.params, X=x_vals_1D)
                                
                                fit_results_dict[x_um] = {
                                    'model': best_model_name,
                                    'F_test': F_test,
                                    'x_um': x_um,
                                    'fit_result': best_out,
                                    'y_vals': y_vals_1D,
                                    'x_vals': x_vals_1D,
                                    'y_fit': y_fit
                                }
                                
                                # Save results to infos_analysis
                                infos_analysis[f'{key}_{line_name}_model'] = best_model_name
                                infos_analysis[f'{key}_{line_name}_F_test'] = F_test
                                infos_analysis[f'{key}_{line_name}_x_position_um'] = x_um
                                
                                # Save fit parameters
                                for param_name, param_value in best_out.params.valuesdict().items():
                                    infos_analysis[f'{key}_{line_name}_{param_name}'] = param_value
                        
                        except Exception as e:
                            print(f"Error processing result for {line_name}: {e}")
                
                fit_elapsed = time.time() - fit_start
                print(f"  Parallel fitting completed in {fit_elapsed:.2f}s")
                
                # Store fit results for this key for second figure
                fit_results_by_key[key] = fit_results_dict
                
                # Prepare F-test data outside of plotting
                f_test_values = [result['F_test'] for result in fit_results_dict.values()]
                f_test_x_positions = [result['x_um'] for result in fit_results_dict.values()]
                
                # Prepare BEC fraction data
                bec_fractions = []
                bec_x_positions = []
                for x_um, result in fit_results_dict.items():
                    model_name = result['model']
                    best_out = result['fit_result']
                    
                    if model_name == 'bimodal' and best_out is not None:
                        # Calculate N_bec and N_th manually from the fit parameters
                        # Using formulas: N_th = amp_g * sqrt(2*pi) * sx * 1.2020569 * pixel_size**2 / cross_section
                        #                N_bec = 16/15 * rx * amp_tf * pixel_size**2 / cross_section
                        try:
                            params_dict = best_out.params.valuesdict()
                            
                            # Find the prefix (it varies by line)
                            prefix = ''
                            for key in params_dict.keys():
                                if 'amp_g' in key:
                                    prefix = key.replace('amp_g', '')
                                    break
                            
                            # Extract fitting parameters
                            amp_g = params_dict.get(prefix + 'amp_g', 0)
                            amp_tf = params_dict.get(prefix + 'amp_tf', 0)
                            sx = params_dict.get(prefix + 'sx', 0)
                            rx = params_dict.get(prefix + 'rx', 0)
                            pixel_size = params_dict.get(prefix + 'pixel_size', 1.0)
                            cross_section = params_dict.get(prefix + 'cross_section', 1.656425e-13)
                            
                            # Calculate N_th and N_bec
                            n_th = amp_g * np.sqrt(2*np.pi) * sx * 1.2020569 * pixel_size**2 / cross_section
                            n_bec = 16/15 * rx * amp_tf * pixel_size**2 / cross_section
                            
                            if len(bec_fractions) == 0:
                                print(f"  Bimodal fit params: amp_g={amp_g:.2e}, amp_tf={amp_tf:.2e}, sx={sx:.2e}, rx={rx:.2e}")
                                print(f"  Calculated: N_th={n_th:.2e}, N_bec={n_bec:.2e}")
                            
                            if n_th + n_bec > 0:
                                bec_frac = n_bec / (n_bec + n_th)
                            else:
                                bec_frac = 0
                        except Exception as e:
                            print(f"  Error computing BEC fraction: {e}")
                            bec_frac = 0
                    else:
                        # Gaussian fit - no BEC component
                        bec_frac = 0
                    
                    bec_fractions.append(bec_frac)
                    bec_x_positions.append(x_um)
                
                plot_start = time.time()
                
                # Plot blurred image on left with colored vertical lines
                im_left = ax[i, 0].pcolormesh(x_vals_full*1e6, y_vals_full*1e6, n2D_blurred, cmap='viridis', shading='auto')
                ax[i, 0].set_title(f'{key} (Blurred)')
                ax[i, 0].set_xlabel('x (μm)')
                ax[i, 0].set_ylabel('y (μm)')
                cbar = plt.colorbar(im_left, ax=ax[i, 0])
                cbar.set_label('OD')
                
                # Color-code vertical lines based on fit results
                for x_um, result in fit_results_dict.items():
                    model_name = result['model']
                    color = 'cyan' if model_name == 'bimodal' else 'lime'
                    linestyle = '-' if model_name == 'bimodal' else '--'
                    ax[i, 0].axvline(x=x_um, color=color, linewidth=1.5, linestyle=linestyle, alpha=0.8)
                
                # Plot F-test results on middle column
                ax[i, 1].scatter(f_test_x_positions, f_test_values, s=50, alpha=0.7, color='blue')
                ax[i, 1].axhline(y=F_TEST_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold: {F_TEST_THRESHOLD}')
                ax[i, 1].set_xlabel('x position (μm)')
                ax[i, 1].set_ylabel('F-test statistic')
                ax[i, 1].set_title(f'{key} - F-test Results')
                ax[i, 1].grid(True, alpha=0.3)
                ax[i, 1].legend()
                
                # Plot BEC fraction on right column
                ax[i, 2].scatter(bec_x_positions, bec_fractions, s=50, alpha=0.7, color='purple')
                ax[i, 2].set_xlabel('x position (μm)')
                ax[i, 2].set_ylabel('BEC Fraction')
                ax[i, 2].set_title(f'{key} - Local BEC Fraction')
                ax[i, 2].set_ylim([-0.05, 1.05])
                ax[i, 2].grid(True, alpha=0.3)
                
                plot_elapsed = time.time() - plot_start
                print(f"  Plotting completed in {plot_elapsed:.2f}s")

            title = general_lib_mod.get_title(h5file, cam)
            fig.suptitle(title, fontsize=12)
            
            figure_complete_time = time.time() - camera_loop_start
            print(f"Figure saved for camera {cam['name']}")
            print(f"  Camera processing total: {figure_complete_time:.2f}s")
            
            # Create second figure with reconstructed 2D profiles from fits
            fig2, ax2 = plt.subplots(L, 2, figsize=(12, 4*L), constrained_layout=True)
            if L == 1:
                ax2 = ax2.reshape(1, -1)
            
            for i, key in enumerate(normal_keys):
                fit_results_dict = fit_results_by_key.get(key, {})
                
                # Reconstruct 2D images from fit results
                n_y = len(y_vals_full)
                n_x = len(x_vals_full)
                tf_2d = np.zeros((n_y, n_x))
                thermal_2d = np.zeros((n_y, n_x))
                
                for x_um, result in fit_results_dict.items():
                    x_pos_m = x_um / 1e6
                    x_idx = np.argmin(np.abs(x_vals_full - x_pos_m))
                    if x_idx >= n_x:
                        x_idx = n_x - 1
                    
                    best_out = result.get('fit_result')
                    model_name = result.get('model')
                    x_vals_1D = result.get('x_vals')
                    y_fit = result.get('y_fit')  # Use the stored fitted profile
                    if best_out is None or x_vals_1D is None or y_fit is None:
                        continue
                    
                    try:
                        if model_name == 'bimodal':
                            # Extract TF and thermal components separately using eval_bimodal_components
                            # Get the prefix from the fitted parameters
                            param_names = list(best_out.params.keys())
                            # Find the prefix by looking for amp_g parameter
                            prefix = ''
                            for pname in param_names:
                                if 'amp_g' in pname:
                                    prefix = pname.replace('amp_g', '')
                                    break
                            
                            if prefix:
                                bimodal_model = BimodalBose1DModel2Centers(prefix=prefix)
                                therm, bec = bimodal_model.eval_bimodal_components(best_out.params, x_vals_1D)
                                tf_2d[:, x_idx] = bec
                                thermal_2d[:, x_idx] = therm
                        else:
                            # Gaussian - entire fit is thermal, no TF component
                            thermal_2d[:, x_idx] = y_fit
                            tf_2d[:, x_idx] = 0
                    except Exception as e:
                        print(f"Error reconstructing components at x={x_um}: {e}")
                
                # Interpolate missing columns from adjacent columns
                for col in range(tf_2d.shape[1]):
                    if np.all(tf_2d[:, col] == 0) and np.all(thermal_2d[:, col] == 0):
                        # This column has no data, find adjacent non-zero column
                        # Try left neighbor first, then right neighbor
                        if col > 0 and (np.any(tf_2d[:, col-1] != 0) or np.any(thermal_2d[:, col-1] != 0)):
                            tf_2d[:, col] = tf_2d[:, col-1]
                            thermal_2d[:, col] = thermal_2d[:, col-1]
                        elif col < tf_2d.shape[1] - 1 and (np.any(tf_2d[:, col+1] != 0) or np.any(thermal_2d[:, col+1] != 0)):
                            tf_2d[:, col] = tf_2d[:, col+1]
                            thermal_2d[:, col] = thermal_2d[:, col+1]
                
                # Find fitted x-range for axis limits
                fitted_x_um = np.array(list(fit_results_dict.keys()))
                x_min_fit = np.min(fitted_x_um)
                x_max_fit = np.max(fitted_x_um)
                
                # Convert to indices
                x_idx_min = np.argmin(np.abs(x_vals_full*1e6 - x_min_fit))
                x_idx_max = np.argmin(np.abs(x_vals_full*1e6 - x_max_fit))
                
                # Crop to fitted region
                tf_2d_cropped = tf_2d[:, x_idx_min:x_idx_max+1]
                thermal_2d_cropped = thermal_2d[:, x_idx_min:x_idx_max+1]
                x_vals_cropped = x_vals_full[x_idx_min:x_idx_max+1]
                
                # Plot TF profile on left
                im_tf = ax2[i, 0].pcolormesh(x_vals_cropped*1e6, y_vals_full*1e6, tf_2d_cropped, cmap='hot', shading='auto', vmax=1e14)
                ax2[i, 0].set_title(f'{key} - Thomas-Fermi Profile')
                ax2[i, 0].set_xlabel('x (μm)')
                ax2[i, 0].set_ylabel('y (μm)')
                ax2[i, 0].set_xlim([x_min_fit, x_max_fit])
                ax2[i, 0].set_ylim([y_vals_full.min()*1e6, y_vals_full.max()*1e6])
                cbar_tf = plt.colorbar(im_tf, ax=ax2[i, 0])
                cbar_tf.set_label('OD')
                
                # Plot thermal fraction profile on right
                im_thermal = ax2[i, 1].pcolormesh(x_vals_cropped*1e6, y_vals_full*1e6, thermal_2d_cropped, cmap='cool', shading='auto')
                ax2[i, 1].set_title(f'{key} - Thermal Fraction Profile')
                ax2[i, 1].set_xlabel('x (μm)')
                ax2[i, 1].set_ylabel('y (μm)')
                ax2[i, 1].set_xlim([x_min_fit, x_max_fit])
                ax2[i, 1].set_ylim([y_vals_full.min()*1e6, y_vals_full.max()*1e6])
                cbar_thermal = plt.colorbar(im_thermal, ax=ax2[i, 1])
                cbar_thermal.set_label('OD')
            
            fig2.suptitle(f'{title} - Fit Reconstructed Profiles', fontsize=12)
            
            # Create separate fig3 for each image key with spatial profiles of fit parameters
            for i, key in enumerate(normal_keys):
                fig3, ax3 = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
                
                fit_results_dict = fit_results_by_key.get(key, {})
                
                tf_centers = []
                gauss_centers = []
                tf_radii = []
                gauss_sigmas = []
                x_positions = []
                
                for x_um, result in fit_results_dict.items():
                    model_name = result['model']
                    best_out = result['fit_result']
                    
                    if best_out is None:
                        continue
                    
                    x_positions.append(x_um)  # Already in micrometers
                    params_dict = best_out.params.valuesdict()
                    
                    # Find the prefix
                    prefix = ''
                    for key_param in params_dict.keys():
                        if 'amp_g' in key_param:
                            prefix = key_param.replace('amp_g', '')
                            break
                    
                    if model_name == 'bimodal':
                        # Extract bimodal parameters (already computed from params, should be in physical units)
                        mtfx = params_dict.get(prefix + 'mtfx', np.nan)
                        mgx = params_dict.get(prefix + 'mgx', np.nan)
                        rx = params_dict.get(prefix + 'rx', 0)
                        sx = params_dict.get(prefix + 'sx', 0)
                        px_size = params_dict.get(prefix + 'pixel_size', 1.0)
                        
                        # Convert from pixel units to micrometers
                        tf_centers.append(mtfx * px_size * 1e6)
                        gauss_centers.append(mgx * px_size * 1e6)
                        tf_radii.append(rx * px_size * 1e6)
                        gauss_sigmas.append(sx * px_size * 1e6)
                    else:
                        # Gaussian fit - only has gaussian center and sigma
                        m = params_dict.get(prefix + 'm', np.nan)
                        s = params_dict.get(prefix + 's', 0)
                        px_size = params_dict.get(prefix + 'pixel_size', 1.0)
                        
                        tf_centers.append(np.nan)
                        gauss_centers.append(m * px_size * 1e6)
                        tf_radii.append(np.nan)
                        gauss_sigmas.append(s * px_size * 1e6)
                
                x_positions = np.array(x_positions)
                tf_centers = np.array(tf_centers)
                gauss_centers = np.array(gauss_centers)
                tf_radii = np.array(tf_radii)
                gauss_sigmas = np.array(gauss_sigmas)
                
                # Plot TF center vs x position
                valid_tf = ~np.isnan(tf_centers)
                ax3[0, 0].scatter(x_positions[valid_tf], tf_centers[valid_tf], s=50, alpha=0.7, color='red')
                ax3[0, 0].set_xlabel('x position (μm)')
                ax3[0, 0].set_ylabel('TF Center (μm)')
                ax3[0, 0].set_title(f'{key} - Thomas-Fermi Center')
                ax3[0, 0].grid(True, alpha=0.3)
                
                # Plot Gaussian center vs x position
                valid_gauss = ~np.isnan(gauss_centers)
                ax3[0, 1].scatter(x_positions[valid_gauss], gauss_centers[valid_gauss], s=50, alpha=0.7, color='blue')
                ax3[0, 1].set_xlabel('x position (μm)')
                ax3[0, 1].set_ylabel('Gaussian Center (μm)')
                ax3[0, 1].set_title(f'{key} - Thermal (Gaussian) Center')
                ax3[0, 1].grid(True, alpha=0.3)
                
                # Plot TF radius vs x position
                valid_rad = ~np.isnan(tf_radii)
                ax3[1, 0].scatter(x_positions[valid_rad], tf_radii[valid_rad], s=50, alpha=0.7, color='darkred')
                ax3[1, 0].set_xlabel('x position (μm)')
                ax3[1, 0].set_ylabel('TF Radius (μm)')
                ax3[1, 0].set_title(f'{key} - Thomas-Fermi Radius')
                ax3[1, 0].grid(True, alpha=0.3)
                
                # Plot Gaussian sigma vs x position
                valid_sig = ~np.isnan(gauss_sigmas)
                ax3[1, 1].scatter(x_positions[valid_sig], gauss_sigmas[valid_sig], s=50, alpha=0.7, color='darkblue')
                ax3[1, 1].set_xlabel('x position (μm)')
                ax3[1, 1].set_ylabel('Gaussian σ (μm)')
                ax3[1, 1].set_title(f'{key} - Thermal (Gaussian) Width')
                ax3[1, 1].grid(True, alpha=0.3)
                
                fig3.suptitle(f'{title} - {key} - Spatial Profiles of Fit Parameters', fontsize=12)

        print(f"Analysis function returning results...")
        return infos_analysis
        
    except Exception as e:
        print(traceback.format_exc())
        return infos_analysis


if __name__ == '__main__':
    time_0 = time.time()
    try:
        import lyse
        run = lyse.Run(lyse.path)
        dict = None
        
        print("Starting analysis...")
        analysis_time = time.time()
        with h5py.File(lyse.path, 'r+') as h5file:
            dict = show_local_thermal_fraction(h5file)
        analysis_elapsed = time.time() - analysis_time
        print(f"[show_local_thermal_fraction] Analysis completed in {analysis_elapsed:.2f}s")
        
        if dict is not None:
            print(f"Results generated: {len(dict)} parameters")
            # Lyse saving temporarily disabled
            # for key in dict:
            #     try:
            #         run.save_result(key, dict[key])
            #     except Exception as e:
            #         print(f"Error saving result {key}: {e}")
        
        print('Total elapsed time: {:.2f} s'.format(time.time()-time_0))
    except Exception as e:
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
        print(traceback.format_exc())
