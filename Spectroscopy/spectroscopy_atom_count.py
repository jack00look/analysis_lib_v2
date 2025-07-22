import h5py
import traceback
import importlib
import matplotlib.pyplot as plt
import numpy as np
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, Gaussian1DModel,J_BimodalBose1DModel2Centers
from fitting.models import J_Gaussian2DModel
from lmfit_lyse_interface import parameters_dict_with_errors
from lmfit import minimize, Parameters
import scipy.ndimage
import time
from matplotlib.patches import Circle, Rectangle

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

cameras = camera_settings.cameras






def spectroscopy_atom_count(h5file,show=True):

    cam_name = 'cam_hor1'
    key = 'atoms'
    cam = cameras[cam_name]

    # get images from camera if any
    dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file,cam)
    if dict_images is None:
        print('No images found for ',cam_name)
        return

    fig,ax = plt.subplots(2,2,figsize=(10,10),constrained_layout=True)
    fig_fits,ax_fits = plt.subplots(ncols=2,nrows=4,figsize=(10,5),constrained_layout=True)
    dict = {}
    # get the image and the ODlog image for the key
    n2D,n1D_x,n1D_y,x_vals,y_vals = imaging_analysis_lib_mod.extract_images_parts(dict_images,cam,key)
    # plot the image do the fits
    imaging_analysis_lib_mod.plot_OD(dict_images,key,cam,fig,ax,0,0)

    centers = {
        'm1': (3.35e-3,4.2e-3),
        'm2': (3.4e-3,1.84e-3)
    }

    side_halflength = 0.9e-3

    N_m1_fit = 0
    N_m2_fit = 0
    N_m1_sum = 0
    N_m2_sum = 0

    dict = {}

    for (key_state,i) in zip(centers,range(len(centers))):

        x0 = centers[key_state][0]
        y0 = centers[key_state][1]

        mask_x = (x_vals > x0-side_halflength) & (x_vals < x0+side_halflength)
        mask_y = (y_vals > y0-side_halflength) & (y_vals < y0+side_halflength)
        ind_x = np.where(mask_x)[0]
        ind_y = np.where(mask_y)[0]
        ind_x_min = int(ind_x.min())
        ind_x_max = int(ind_x.max())
        ind_y_min = int(ind_y.min())
        ind_y_max = int(ind_y.max())

        # get the number of atoms in the state
        n_atoms = np.sum(n2D[ind_y_min:ind_y_max+1,ind_x_min:ind_x_max+1])
        dict.update({key_state+'_Natoms': n_atoms})


        x_fit = x_vals[ind_x_min:ind_x_max+1]
        y_fit = y_vals[ind_y_min:ind_y_max+1]
        n2D_fit = n2D[ind_y_min:ind_y_max+1,ind_x_min:ind_x_max+1]
        X, Y = np.meshgrid(x_fit, y_fit)
        out = imaging_analysis_lib_mod.do_fit_2D(n2D_fit, X, Y, J_Gaussian2DModel, key_state+'_', ax_fits,2*i,1)

        params_dict = out.params.valuesdict()
        if key_state == 'm1':
            N_m1_fit = params_dict[key_state+'_N']
            N_m1_sum = n_atoms
        elif key_state == 'm2':
            N_m2_fit = params_dict[key_state+'_N']
            N_m2_sum = n_atoms


        dict.update({key_state+'_fit': out.params.valuesdict()[key_state+'_N']})
        dict.update({key_state+'_sx': out.params.valuesdict()[key_state+'_sx']})
        dict.update({key_state+'_sy': out.params.valuesdict()[key_state+'_sy']})
        dict.update({key_state+'_var': out.params.valuesdict()[key_state+'_var']})
        dict.update({key_state+'_redchi': out.result.redchi})
        ax[0,1].add_patch(Rectangle((x0-side_halflength,y0-side_halflength),2*side_halflength,2*side_halflength,fill=False,zorder=10,color = 'blue',ls = '--',lw=4))

    M_sum = (N_m2_sum - N_m1_sum) / (N_m2_sum + N_m1_sum)
    dict.update({'M_sum': M_sum})
    M_fit = (N_m2_fit - N_m1_fit) / (N_m2_fit + N_m1_fit)
    dict.update({'M_fit': M_fit})

    
    return dict


if __name__ == '__main__':
    try:
        time_0 = time.time()
        import lyse
        run = lyse.Run(lyse.path)
        dict = None
        with h5py.File(lyse.path) as h5file:
            dict = spectroscopy_atom_count(h5file)
            if dict is not None:
                for key in dict:
                    run.save_result(key,dict[key])
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
    except Exception as e:
        print(traceback.format_exc())
        print('Failed to count the atoms in the three spinor states')


