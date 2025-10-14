import h5py
import traceback
import importlib
import matplotlib.pyplot as plt
import numpy as np
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, Gaussian1DModel,J_BimodalBose1DModel2Centers
from fitting.models import J_Gaussian2DModel,J_BimodalBose2DModel2Centers,J_Flat2DModel
from lmfit_lyse_interface import parameters_dict_with_errors
from lmfit import minimize, Parameters
import scipy.ndimage
import time
from matplotlib.patches import Circle,Rectangle

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

cameras = camera_settings.cameras






def spinor_atom_count(h5file,show=True):

    cam_name = 'cam_hor1'
    key = 'atoms'
    cam = cameras[cam_name]

    # get images from camera if any
    dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file,cam)
    if dict_images is None:
        print('No images found for ',cam_name)
        return

    fig,ax = plt.subplots(2,2,figsize=(10,10),constrained_layout=True)
    fig_fits,ax_fits = plt.subplots(ncols=2,nrows=6,figsize=(5,10),constrained_layout=True)
    dict = {}
    # get the image and the ODlog image for the key
    n2D,n1D_x,n1D_y,x_vals,y_vals = imaging_analysis_lib_mod.extract_images_parts(dict_images,cam,key)
    # plot the image do the fits
    imaging_analysis_lib_mod.plot_OD(dict_images,key,cam,fig,ax,0,0)

    centers = {
        'm1': (3.407e-3,4.177e-3),
        '_0': (3.431e-3,3.000e-3),
        'p1': (3.462e-3,1.952e-3)
    }

    side_halflength_hor = 0.8e-3
    side_halflength_vert = 0.4e-3
    dict = {}

    for (key_state,i) in zip(centers,range(len(centers))):

        x0 = centers[key_state][0]
        y0 = centers[key_state][1]

        x_mask = (x_vals > x0-side_halflength_hor) & (x_vals < x0+side_halflength_hor)
        x_mask_ind = np.where(x_mask)[0]
        x_ind_min = int(x_mask_ind.min())
        x_ind_max = int(x_mask_ind.max())
        y_mask = (y_vals > y0-side_halflength_vert) & (y_vals < y0+side_halflength_vert)
        y_mask_ind = np.where(y_mask)[0]
        y_ind_min = int(y_mask_ind.min())
        y_ind_max = int(y_mask_ind.max())

        x_fit = x_vals[x_ind_min:x_ind_max+1]
        y_fit = y_vals[y_ind_min:y_ind_max+1]
        n2D_fit = n2D[y_ind_min:y_ind_max+1,x_ind_min:x_ind_max+1]
        # get the number of atoms in the state
        dict.update({key_state+'_Nsum': n2D_fit.sum()})

        X,Y = np.meshgrid(x_fit,y_fit)
        out, is_flat = imaging_analysis_lib_mod.do_fit_2D(n2D_fit, X, Y, J_BimodalBose2DModel2Centers, key_state+'_', ax_fits,2*i,1)
        fit_results = out.params.valuesdict()
        if is_flat:
            dict.update({key_state+'_Nfit': 0.})
        else:
            dict.update({key_state+'_Nfit': fit_results[key_state+'_N']})

        ax[0,1].add_patch(Rectangle((x0-side_halflength_hor,y0-side_halflength_vert),2*side_halflength_hor,2*side_halflength_vert,fill=False,zorder=10,color = 'g',ls = '--',lw=2))
    return dict


if __name__ == '__main__':
    try:
        time_0 = time.time()
        import lyse
        run = lyse.Run(lyse.path)
        dict = None
        with h5py.File(lyse.path) as h5file:
            dict = spinor_atom_count(h5file)
            if dict is not None:
                for key in dict:
                    run.save_result(key,dict[key])
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
    except Exception as e:
        print(traceback.format_exc())
        print('Failed to count the atoms in the three spinor states')


