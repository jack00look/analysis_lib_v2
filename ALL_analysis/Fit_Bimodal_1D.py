import h5py
import lyse
import importlib
import matplotlib.pyplot as plt
import numpy as np
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, J_Gaussian1DModel,J_BimodalBose1DModel2Centers
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

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

cameras = camera_settings.cameras

def Fit_Bimodal_1D(h5file,show=True):

    important_infos_dict = {
        'N': {
            'unit':'MLN',
            'factor':1e-6
        }
    }

    # check all cameras
    for (cam_entry,j) in zip(cameras,range(len(cameras))):
            
            #select camera
            cam = cameras[cam_entry]

            # get all image related stuff
            dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file,cam)

            # check if there is in fact image related stuff
            if dict_images is None:
                continue
            
            # extract images and infos about images
            images = dict_images['images']
            infos_images = dict_images['infos']

            # list all keys corresponding to images
            keys = list(images.keys())

            # check if there is in fact images
            if len(keys) == 0:
                continue
            
            # create plot
            fig,ax = plt.subplots(ncols=2,nrows=2*len(keys),figsize=(10,10*len(keys)),constrained_layout=True)

            # dict to save all infos I will want to save in Lyse
            dict_infos = None

            #range in all images
            for (key,i) in zip(keys,range(len(keys))):

                # extract various parts of the image
                n2D,n1D_x,n1D_y,x_vals,y_vals = imaging_analysis_lib_mod.extract_images_parts(dict_images,cam,key)

                # plot the OD of the image
                imaging_analysis_lib_mod.plot_OD(dict_images,key,cam,fig,ax,2*i,0)

                # perform the 1D fits in x and y
                try:
                    out_x = imaging_analysis_lib_mod.do_fit_1D(n1D_x,x_vals,J_BimodalBose1DModel2Centers,key+'_xfit_',ax[2*i+1,1])
                    out_y = imaging_analysis_lib_mod.do_fit_1D(n1D_y,y_vals,J_BimodalBose1DModel2Centers,key+'_yfit_',ax[2*i,0],invert=True)
                except Exception as e:
                    print('Failed to fit 1D bimodal for camera ',cam_entry,' key ',key)
                    print(traceback.format_exc())
                    continue

                # save results of the fits in a temp dictionary related to this image
                dict_temp = out_x.params.valuesdict()
                dict_temp.update(out_y.params.valuesdict())

                # get important infos text to display on the plot
                INFO_text = general_lib_mod.get_important_infos_text(important_infos_dict,dict_temp)
                # add the text to the plot
                ax[2*i+1,0].text(0.5, 0.5,INFO_text,fontsize=12,transform=ax[1,0].transAxes,ha='center',va='center',bbox=dict(boxstyle="round", fc="white", ec="black"))

                # update the main dictionary with the temp one
                if dict_infos is None:
                    dict_infos = dict_temp
                else:
                    dict_infos.update(dict_temp)

            # add a title to the plot
            title = general_lib_mod.get_title(h5file,cam)
            fig.suptitle(title,fontsize=12)
                

    return dict_infos

if __name__ == '__main__':
    try:
        time_0 = time.time()
        import lyse
        run = lyse.Run(lyse.path)
        dict_save = None 
        with h5py.File(run.h5_path) as h5file:
            dict_save = Fit_Bimodal_1D(h5file)
        if dict_save is not None:
            for key in dict_save:
                run.save_result(key,dict_save[key])
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
    except Exception as e:
        print(traceback.format_exc())
        print('Failed to fit Gaussian 1D')



