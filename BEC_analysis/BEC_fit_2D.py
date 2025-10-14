import h5py
import traceback
import importlib
import matplotlib.pyplot as plt
import numpy as np
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, Gaussian1DModel,J_BimodalBose1DModel2Centers
from fitting.models import J_Gaussian2DModel,J_BimodalBose2DModel2Centers
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

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

cameras = camera_settings.cameras






def BEC_fit_2D(h5file,show=True):

    for cam_entry in cameras:
            cam = cameras[cam_entry]

            dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file,cam)
            if dict_images is None:
                continue

            images = dict_images['images']
            infos_images = dict_images['infos']

            keys = list(images.keys())
            if len(keys) == 0:
                continue
            print('keys:',keys)

            fig,ax = plt.subplots(ncols=2,nrows=2*len(keys),figsize=(10,10),constrained_layout=True)

            fig.suptitle(general_lib_mod.get_title(h5file,cam),fontsize=15)

            fig_fits,ax_fits = plt.subplots(ncols=2,nrows=2*len(keys),figsize=(10,5*len(keys)),constrained_layout=True)

            dict = {}

            for (key,i) in zip(keys,range(len(keys))):
                print('Fitting key:',key)
                print('i:',i)
                n2D,n1D_x,n1D_y,x_vals,y_vals = imaging_analysis_lib_mod.extract_images_parts(dict_images,cam,key)
                imaging_analysis_lib_mod.plot_OD(dict_images,key,cam,fig,ax,i,0)
                X,Y = np.meshgrid(x_vals,y_vals)
                out, is_flat = imaging_analysis_lib_mod.do_fit_2D(n2D, X, Y, J_BimodalBose2DModel2Centers, key+'_', ax_fits,2*i,1)
                params_dict = out.params.valuesdict()
                if is_flat:
                    dict.update({key+'_Nfit': 0.})
                else:
                    dict.update({key+'_Nfit': params_dict[key+'_N']})
                    for param_key in params_dict:
                        if param_key != key+'_N':
                            dict.update({param_key: params_dict[param_key]})
            return dict


if __name__ == '__main__':
    try:
        time_0 = time.time()
        import lyse
        run = lyse.Run(lyse.path)
        dict = None
        with h5py.File(lyse.path) as h5file:
            dict = BEC_fit_2D(h5file)
            if dict is not None:
                for key in dict:
                    run.save_result(key,dict[key])
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
    except Exception as e:
        print(traceback.format_exc())
        print('Failed to do the 2D BEC fit')


