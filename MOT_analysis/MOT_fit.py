import h5py
import lyse
import importlib
import matplotlib.pyplot as plt
import numpy as np
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, J_Gaussian1DModel
from lmfit_lyse_interface import parameters_dict_with_errors
from lmfit import minimize, Parameters
import scipy.ndimage
import traceback
import time

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

cameras = camera_settings.cameras


cam_name = 'cam_hor3'
key = 'atoms'

# cam_name = 'cam_vert1'
# key = 'PTAI_m1'

cam = cameras[cam_name]


def fit_MOT(h5file,show=True):

    # get images from camera if any
    dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file,cam)
    if dict_images is None:
        print('No images found for ',cam_name)
        return
    
    n2D,n1D_x,n1D_y,x_vals,y_vals = imaging_analysis_lib_mod.extract_images_parts(dict_images,cam,key)

    # ymin = 0.
    # ymax = 2.5
    # y_ind = (y_vals > ymin) & (y_vals < ymax)
    # y_vals = y_vals[y_ind]
    # n1D_y = n1D_y[y_ind]

    # plot the image do the fits
    fig,ax = plt.subplots(2,2,figsize=(10,10),constrained_layout=True)
    imaging_analysis_lib_mod.plot_OD(dict_images,key,cam,fig,ax,0,0)
    out_x = imaging_analysis_lib_mod.do_fit_1D(n1D_x,x_vals,J_Gaussian1DModel,'xfit_',ax[1,1])
    out_y = imaging_analysis_lib_mod.do_fit_1D(n1D_y,y_vals,J_Gaussian1DModel,'yfit_',ax[0,0],invert=True)

    # get the parameters from the fits in a dictionary
    dict = out_x.params.valuesdict()
    dict.update(out_y.params.valuesdict())
    keys_dict_list = list(dict.keys())
    for i in range(len(keys_dict_list)):
        key_dict = keys_dict_list[i]
        ax[1,0].text(0.5, 1-i*0.07, f'{key_dict}: {dict[key_dict]:.2f}', fontsize = 12,transform=ax[1,0].transAxes,
                     horizontalalignment='center', verticalalignment='center')
    return dict

if __name__ == '__main__':
    try:
        time_0 = time.time()
        import lyse
        run = lyse.Run(lyse.path)
        dict = None 
        with h5py.File(run.h5_path) as h5file:
            dict = fit_MOT(h5file)
        if dict is not None:
            for key in dict:
                run.save_result(key,dict[key])
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
    except Exception as e:
        print(traceback.format_exc())
        print('Failed to fit MOT')



