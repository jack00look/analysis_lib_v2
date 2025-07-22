import h5py
import traceback
import importlib
import matplotlib.pyplot as plt
import numpy as np
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, Gaussian1DModel,J_BimodalBose1DModel2Centers
from lmfit_lyse_interface import parameters_dict_with_errors
from lmfit import minimize, Parameters
import scipy.ndimage
import time

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

cameras = camera_settings.cameras



cam_name = 'cam_vert1'
keys = ['PTAI_m1']

cam_name = 'cam_hor1'
keys = ['atoms']


cam = cameras[cam_name]

def BEC_fit_hor(h5file,show=True):

    # get images from camera if any
    dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file,cam)
    if dict_images is None:
        print('No images found for ',cam_name)
        return

    fig,ax = plt.subplots(2*len(keys),2,figsize=(10*len(keys),10),constrained_layout=True)
    dict = {}
    for i,key in enumerate(keys):
        # get the image and the ODlog image for the key
        n2D,n1D_x,n1D_y,x_vals,y_vals = imaging_analysis_lib_mod.extract_images_parts(dict_images,cam,key)

        # plot the image do the fits
        imaging_analysis_lib_mod.plot_OD(dict_images,key,cam,fig,ax,2*i,0)
        out_x = imaging_analysis_lib_mod.do_fit_1D(n1D_x,x_vals,J_BimodalBose1DModel2Centers,key + 'xfit_',ax[2*i+1,1])
        out_y = imaging_analysis_lib_mod.do_fit_1D(n1D_y,y_vals,J_BimodalBose1DModel2Centers,key + 'yfit_',ax[2*i,0],invert=True)

        # get the parameters from the fits in a dictionary
        dict.update(out_x.params.valuesdict())
        dict.update(out_y.params.valuesdict())

    return dict


if __name__ == '__main__':
    try:
        time_0 = time.time()
        import lyse
        run = lyse.Run(lyse.path)
        dict = None
        with h5py.File(lyse.path) as h5file:
            dict = BEC_fit_hor(h5file)
            if dict is not None:
                for key in dict:
                    run.save_result(key,dict[key])
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
    except Exception as e:
        print(traceback.format_exc())
        print('Failed to fit BEC')


