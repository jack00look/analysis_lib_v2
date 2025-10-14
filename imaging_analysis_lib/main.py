import numpy as np
import h5py
import time
import importlib
from numba import jit
from scipy.ndimage import gaussian_filter
import traceback
import matplotlib.patches as patches
from fitting.models import J_Flat2DModel
from scipy.stats import f

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

fontsize_mine = 12

def extract_images_parts(dict_images,cam,key):
    n2D = dict_images['images'][key]
    axes = dict_images['axes']
    x_vals, y_vals, X_name, Y_name = get_axes(axes)
    n1D_x, n1D_y = get_n1Ds(n2D,cam)
    return n2D, n1D_x, n1D_y, x_vals, y_vals

def do_fit_1D(y_vals,x_vals,MODEL,prefix,ax,invert = False):

    model = MODEL(prefix=prefix)

    p0 = model.guess(y_vals, X=x_vals)
    if not invert:
        ax.plot(x_vals, model.eval(p0, X=x_vals), 'k--', label='initial guess')
    else:
        ax.plot(model.eval(p0, X=x_vals), x_vals, 'k--', label='initial guess')
    
    out = model.fit(y_vals, p0, X=x_vals)
    model.plot_best(out.params,x_vals,ax,invert)

    ax.legend()

    return out

def do_fit_2D(Z_vals,X_vals,Y_vals,MODEL,prefix,ax,i,j):

    flat_model = J_Flat2DModel(prefix=prefix)
    p0_flat = flat_model.guess(Z_vals, X=X_vals, Y=Y_vals)
    out_flat = flat_model.fit(Z_vals, p0_flat, X=X_vals, Y=Y_vals)
    chisqr_flat = out_flat.chisqr
    nfree_flat = out_flat.nfree

    ax_surface = ax[i,j]
    ax_x = ax[i+1,j]
    ax_y = ax[i,j-1]

    x_vals = X_vals[0,:]
    y_vals = Y_vals[:,0]
    Z_vals_x = np.sum(Z_vals, axis=0)
    Z_vals_y = np.sum(Z_vals, axis=1)

    model = MODEL(prefix=prefix)

    p0 = model.guess(Z_vals, X=X_vals, Y=Y_vals)
    guess_fit = model.eval(p0, X=X_vals, Y=Y_vals)
    guess_fit_x = np.sum(guess_fit, axis=0)
    guess_fit_y = np.sum(guess_fit, axis=1)
    
    fit_results = model.fit(Z_vals, p0, X=X_vals, Y=Y_vals)

    chisqr = fit_results.chisqr
    nfree = fit_results.nfree

    F_test = (chisqr_flat - chisqr)/(nfree_flat - nfree) / (chisqr / nfree)
    p_val = 1 - f.cdf(F_test, nfree_flat - nfree, nfree)
    print('nfree flat: ',nfree_flat)
    print('nfree model: ',nfree)
    print('F_test value: ',F_test)
    print('p-value: ',p_val)

    F_test_threshold = 150.

    is_flat = F_test < F_test_threshold

    best_fit = model.eval(fit_results.params, X=X_vals, Y=Y_vals)
    best_fit_x = np.sum(best_fit, axis=0)
    best_fit_y = np.sum(best_fit, axis=1)

    fig_surface = ax_surface.pcolormesh(X_vals, Y_vals, Z_vals - best_fit)
    
    ax_x.plot(x_vals, Z_vals_x)
    ax_x.plot(x_vals, guess_fit_x, 'k--', label='initial guess')
    ax_x.plot(x_vals, best_fit_x, 'r-', label='best fit')
    ax_x.legend()

    ax_y.plot(Z_vals_y, y_vals)
    ax_y.plot(guess_fit_y, y_vals, 'k--', label='initial guess')
    ax_y.plot(best_fit_y, y_vals, 'r-', label='best fit')
    ax_y.legend()

    ax[i+1,j-1].axis('off')


    return fit_results, is_flat

def get_axes(axes):
    axes_names = list(axes.keys())
    X_name = axes_names[0]
    Y_name = axes_names[1]
    x_vals = axes[X_name]
    y_vals = axes[Y_name]
    return x_vals, y_vals, X_name, Y_name

def get_n1Ds(n_2D,cam):
    OD_int_hor = np.sum(n_2D,axis=0)*cam['px_size']
    OD_int_vert = np.sum(n_2D,axis=1)*cam['px_size']
    return OD_int_hor, OD_int_vert

def get_roiback_edges(cam,x_vals,y_vals):
    X,Y = np.meshgrid(x_vals,y_vals)
    axes_infos = get_axes_infos(cam)
    roi_back = cam['roi_back']
    roi_back_xmin = 0
    roi_back_xmax = 0
    roi_back_ymin = 0
    roi_back_ymax = 0
    if axes_infos['invert_1']:
        roi_back_xmin = X[:,::-1][roi_back].min()
        roi_back_xmax = X[:,::-1][roi_back].max()
    else:
        roi_back_xmin = X[roi_back].min()
        roi_back_xmax = X[roi_back].max()
    if axes_infos['invert_2']:
        roi_back_ymin = Y[::-1][roi_back].min()
        roi_back_ymax = Y[::-1][roi_back].max()
    else:
        roi_back_ymin = Y[roi_back].min()
        roi_back_ymax = Y[roi_back].max()
    return roi_back_xmin, roi_back_xmax, roi_back_ymin, roi_back_ymax

def get_roi_integration(cam,x_vals,y_vals):
    X,Y = np.meshgrid(x_vals,y_vals)
    axes_infos = get_axes_infos(cam)
    roi_integration = cam['roi_integration']
    roi_integration_xmin = 0
    roi_integration_xmax = 0
    roi_integration_ymin = 0
    roi_integration_ymax = 0
    if axes_infos['invert_1']:
        roi_integration_xmin = X[:,::-1][roi_integration].min()
        roi_integration_xmax = X[:,::-1][roi_integration].max()
    else:
        roi_integration_xmin = X[roi_integration].min()
        roi_integration_xmax = X[roi_integration].max()
    if axes_infos['invert_2']:
        roi_integration_ymin = Y[::-1][roi_integration].min()
        roi_integration_ymax = Y[::-1][roi_integration].max()
    else:
        roi_integration_ymin = Y[roi_integration].min()
        roi_integration_ymax = Y[roi_integration].max()
    return roi_integration_xmin, roi_integration_xmax, roi_integration_ymin, roi_integration_ymax

def plot_OD(dict_images,key,cam,fig,ax,i,j):

    ODlog = dict_images['images_ODlog'][key]
    n2D = dict_images['images'][key]
    roi_integration = cam['roi_integration']
    n1D_x, n1D_y = get_n1Ds(n2D[roi_integration],cam)
    axes = dict_images['axes']
    x_vals, y_vals, X_name, Y_name = get_axes(axes)
    y_vals_1D = y_vals.copy()
    x_vals_1D = x_vals.copy()
    X_1D,Y_1D = np.meshgrid(x_vals_1D,y_vals_1D)
    X_1D = X_1D[roi_integration]
    Y_1D = Y_1D[roi_integration]
    x_vals_1D = X_1D[0,:]
    y_vals_1D = Y_1D[:,0]
    roi_back_xmin, roi_back_xmax, roi_back_ymin, roi_back_ymax = get_roiback_edges(cam,x_vals,y_vals)
    roi_int_xmin, roi_int_xmax, roi_int_ymin, roi_int_ymax = get_roi_integration(cam,x_vals,y_vals)
    X,Y = np.meshgrid(x_vals,y_vals)
    X_name += ' [m]'
    Y_name += ' [m]'

    fig_temp = ax[2*i,j+1].pcolormesh(X,Y,ODlog,vmin = 0,vmax = 1.5,cmap = 'gist_stern')
    cbar = fig.colorbar(fig_temp,ax=ax[2*i,j+1])
    cbar.set_label('ln(I_in/I_out)',fontsize=fontsize_mine)
    rect = patches.Rectangle((roi_back_xmin,roi_back_ymin),roi_back_xmax-roi_back_xmin,roi_back_ymax-roi_back_ymin,linewidth=3,edgecolor='g',facecolor='none',linestyle='--')
    rect_int = patches.Rectangle((roi_int_xmin,roi_int_ymin),roi_int_xmax-roi_int_xmin,roi_int_ymax-roi_int_ymin,linewidth=3,edgecolor='b',facecolor='none',linestyle='--')
    ax[2*i,j+1].add_patch(rect)
    ax[2*i,j+1].add_patch(rect_int)
    ax[2*i,j+1].set_title(key)
    ax[2*i,j+1].ticklabel_format(axis='both',style='sci',scilimits=(0,0))
    
    ax[2*i+1,j+1].plot(x_vals_1D,n1D_x)
    ax[2*i+1,j+1].set_xlabel(X_name,fontsize=fontsize_mine)
    ax[2*i+1,j+1].set_ylabel('n1D [m$^{-1}$]',fontsize=fontsize_mine)
    ax[2*i+1,j+1].sharex(ax[2*i,j+1])
    ax[2*i,j+1].ticklabel_format(axis='both',style='sci',scilimits=(0,0))

    ax[2*i,j].plot(n1D_y,y_vals_1D)
    ax[2*i,j].set_ylabel(Y_name,fontsize=fontsize_mine)
    ax[2*i,j].set_xlabel('n1D [m$^{-1}$]',fontsize=fontsize_mine)
    ax[2*i,j].sharey(ax[2*i,j+1])
    ax[2*i,j].ticklabel_format(axis='both',style='sci',scilimits=(0,0))

    ax[2*i+1,j].axis('off')

    return n1D_x,n1D_y

def plot_dmd(dict_images,key,cam,fig,ax,i,j,a = None,b = None):
    n2D = dict_images['images'][key]
    n1D_x, n1D_y = get_n1Ds(n2D,cam)
    axes = dict_images['axes']
    x_vals, y_vals, X_name, Y_name = get_axes(axes)
    dmd_raw_data = dict_images['raws_dmd']
    dmd_image = dmd_raw_data['image1D']
    if a is None:
        a = dmd_raw_data['calibration'][0]
    if b is None:
        b = dmd_raw_data['calibration'][1]
    dmd_x_arr = a*np.arange(0,1920) + b
    dmd_y_arr = np.zeros((np.size(dmd_x_arr)))
    for k in range(1920):
        dmd_y_arr[k] = dmd_image[k]
    n1D_x_smooth = gaussian_filter(n1D_x, 10)
    dmd_y_arr = dmd_y_arr*np.max(n1D_x_smooth)
    ind_dmd_in_atoms = np.where((dmd_x_arr > x_vals[0]) & (dmd_x_arr < x_vals[-1]))[0]
    dmd_y_arr = dmd_y_arr[ind_dmd_in_atoms]
    dmd_x_arr = dmd_x_arr[ind_dmd_in_atoms]
    ax[i,j].plot(dmd_x_arr,dmd_y_arr)

def calculate_OD(atoms, probe, cam, back = None, roi_background=None,alpha = None, chi_sat = None, pulse_time = None,detuning = 0):

    # sigma is the cross section of the atoms, used to compute the 2D density
    sigma = 1.656425e-13 #m2

    # clip the images in their ROI
    roi = cam['roi']
    atoms = atoms[roi]
    probe = probe[roi]
    back = back[roi]

    # now compute the figures of merit for the images regarding saturation (too much light) and background (too little light)
    px_number = np.size(atoms.flatten())
    back_max = np.mean(back) + 3*np.std(back)
    sat_min = cam['dynamic_range']*0.95
    cnt_probe_back = np.sum(probe<back_max)/px_number
    cnt_atoms_back = np.sum(atoms<back_max)/px_number
    cnt_probe_sat = np.sum(probe>sat_min)/px_number
    cnt_atoms_sat = np.sum(atoms>sat_min)/px_number

    # subtract the background
    atoms = atoms - back
    probe = probe - back
    

    # normalize the probe to the atoms
    roi_background = cam['roi_back']
    probe = probe * (atoms[roi_background].mean()/probe[roi_background].mean())
    probe = probe.clip(1)
    atoms = atoms.clip(1)

    arr = np.zeros_like(atoms)
    arr[roi_background] = 1
    roi_background_size = np.sum(arr[roi_background])
    
    
    # compute figure of merit for the normalization, if roi_background is good then error_norm should be small
    error_norm = np.sqrt(np.sum(((atoms[roi_background]-probe[roi_background])**2/(np.sqrt(2)*probe[roi_background]))))/roi_background_size

    # get the parameters of the camera
    alpha = cam['alpha']
    chi_sat = cam['chi_sat']
    pulse_time = cam['pulse_time']

    # compute the 2D density
    OD_log_bare = np.log( (np.array(probe).clip(1)) / (np.array(atoms).clip(1))  )
    OD_log = OD_log_bare * alpha * (1 + (2*detuning)**2)
    OD_lin = (probe -  atoms) / (chi_sat * pulse_time)
    n_2D = (OD_log + OD_lin)/sigma

    OD_lin_sum = np.sum(OD_lin)
    OD_log_sum = np.sum(OD_log)
    OD_log_bare_sum = np.sum(OD_log_bare)

    # compute the number of atoms
    pixel_size = cam['px_size']
    N = np.sum(n_2D)*pixel_size**2

    dict_infos = {'OD_log_bare_sum':OD_log_bare_sum,
                  'OD_lin_sum':OD_lin_sum,
                  'N_atoms':N,
                  'probe_norm_err':error_norm,
                  'OD_log_sum':OD_log_sum,
                  'cnt_rel_probe_back':cnt_probe_back,
                  'cnt_rel_atoms_back':cnt_atoms_back,
                  'cnt_rel_probe_sat':cnt_probe_sat,
                  'cnt_rel_atoms_sat':cnt_atoms_sat
                  }
    
    return n_2D,dict_infos,OD_log_bare

def get_raws_camera(h5file,cam,show_errs = False):
    try:
        # check if the camera is in the h5file
        if cam['h5group'] not in h5file:
            if show_errs:
                print(h5file.filename + " has no " + cam['name'] + " images")
            return
        # get the images with their names
        imgs = h5file[cam['h5group']]
        img_names = [s.decode('utf-8') for s in imgs.attrs['img_names']]
        images = dict(zip(img_names, imgs))
        return images
    except Exception as e:
        print('Error in getting raws from cam' + cam['name'] + ':', e)
        print(traceback.format_exc())


def get_axes_infos(cam):

    # get matrix from camera_settings.py that transforms axis of camera image to true axis of the experiment
    axis_matrix = cam['axis_matrix'].copy()

    axis_names = ['x','y','z']
    x_ax_cam_array = np.array([1,0])
    y_ax_cam_array = np.array([0,1])
    img_ax_1 = np.dot(axis_matrix,x_ax_cam_array)
    img_ax_2 = np.dot(axis_matrix,y_ax_cam_array)
    ind_1 = np.where(np.abs(img_ax_1) != 0)[0][0]
    ind_2 = np.where(np.abs(img_ax_2) != 0)[0][0]
    name_1 = axis_names[ind_1]
    name_2 = axis_names[ind_2]
    invert_1 = img_ax_1[ind_1] < 0
    invert_2 = img_ax_2[ind_2] < 0
    dict = {'name_1':name_1,'name_2':name_2,'invert_1':invert_1,'invert_2':invert_2}
    return dict

def get_images_camera_waxes(h5file,cam,show_errs = False):
    try:
        atoms_keys = cam['atoms_images'].copy()
        probe_key = cam['probe_image']
        background_key = cam['background_image']
        
        images_raws = get_raws_camera(h5file,cam)
        if images_raws is None:
            return
        try:
            probe_det_0 = h5file['globals'].attrs['probe_detuning_0']
        except:
            probe_det_0 = 0
            print('No probe_detuning_0 attribute found in globals, setting it to 0.')
        detuning = (h5file['globals'].attrs['probe_detuning_debug'] - probe_det_0)/9.7946
        images = {}
        images_ODlog = {}
        infos = {}
        keys_in_images = list(images_raws.keys())
        for key in atoms_keys:
            if key not in images_raws:
                continue
            try:
                n_2D,dict_infos,OD_log_bare = calculate_OD(images_raws[key], images_raws[probe_key], cam, images_raws[background_key],detuning=detuning)
            except Exception as e:
                print("Error in calculating OD for " + key + " in cam " + cam['name'], e)
                print(traceback.format_exc())
                continue
            images[key] = n_2D
            infos[key] = dict_infos
            images_ODlog[key] = OD_log_bare
            
        
        px_size = cam['px_size']
        x_ax_cam = np.arange(images[keys_in_images[0]].shape[1])*px_size
        y_ax_cam = np.arange(images[keys_in_images[0]].shape[0])*px_size
        dict_axes_infos = get_axes_infos(cam)
        axis_image = {}
        axis_image[dict_axes_infos['name_1']] = x_ax_cam
        axis_image[dict_axes_infos['name_2']] = y_ax_cam
        for key in images:
            if dict_axes_infos['invert_1']:
                images[key] = images[key][:,::-1]
                images_ODlog[key] = images_ODlog[key][:,::-1]
            if dict_axes_infos['invert_2']:
                images[key] = images[key][::-1,:]
                images_ODlog[key] = images_ODlog[key][::-1,:]
        dmd_raw_data = get_raws_dmd(h5file,show_errs)
        dict = {'images':images,'axes':axis_image,'infos':infos,'images_ODlog':images_ODlog,'raws_dmd':dmd_raw_data}
        return dict
    except Exception as e:
        print("Error in getting images with axes from cam" + cam['name'] + ": " + str(e))
        print(traceback.format_exc())
        return

def get_raws_dmd(h5file,show_errs = False):
    try:
        data = {}
        if 'data/dmd/dmd_alive' not in h5file:
            if show_errs:
                print(h5file.filename + " has no DMD data")
            return
        else:
            for name in h5file['data/dmd']:
                data[name] = h5file['data/dmd/' + name]
        return data
    except Exception as e:
        if show_errs:
            raise("Error in getting raw data from DMD: " + str(e))
        return

def get_image_dmd(h5file,show_errs=False):
    dmd_raw_data = get_raws_dmd(h5file,show_errs)
    if dmd_raw_data is None:
        return
    calibration_parameters = dmd_raw_data['calibration']
    image = dmd_raw_data['image']
    a_cal = calibration_parameters[0]
    b_cal = calibration_parameters[1]
    image = gaussian_filter(image, 1/a_cal)
    x_min = 0*a_cal + b_cal
    x_max = 1920*a_cal + b_cal
    x_array = np.arange(x_min,x_max,1)
    y_array = np.arange(0,1080,1)
    image_calibrated = np.zeros((1080,np.size(x_array)))
    for i in range(len(x_array)):
        x = x_array[i]
        x_dmd = int(np.round((x - b_cal)/a_cal))
        for j in range(len(y_array)):
            y = y_array[j]
            image_calibrated[j,i] = image[y,x_dmd]
    return image_calibrated, x_array, y_array


