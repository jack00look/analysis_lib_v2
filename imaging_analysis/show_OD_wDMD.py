import h5py
import lyse
import importlib
import matplotlib.pyplot as plt
import numpy as np
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, Gaussian1DModel
from lmfit_lyse_interface import parameters_dict_with_errors

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

cameras = camera_settings.cameras

run = lyse.Run(lyse.path)


vmin = -1
vmax = 8

def plot_OD(img,key,ax,i,j):
    fig_temp = ax[2*i,j+1].imshow(img,vmin=vmin,vmax = vmax,aspect='auto')
    fig.colorbar(fig_temp,ax=ax[2*i,j+1])
    ax[2*i,j+1].set_title(key)
    OD_int_hor = np.sum(img[0:120,:],axis=0)                     #Range of integration along vertical axis
    OD_int_vert = np.sum(img[:,800:1000],axis=1)                 #Range of integration along horizontal axis
    ax[2*i+1,j+1].plot(OD_int_hor)
    ax[2*i,j].plot(np.arange(len(OD_int_vert)), OD_int_vert)
    #ax[2*i,j].invert_xaxis()
    ax[2*i+1,j+1].sharex(ax[2*i,j+1])
    #ax[2*i,j].sharey(ax[2*i,j+1])
    ax[2*i+1,j].remove()
    return ax, OD_int_vert



with h5py.File(lyse.path) as h5file:
    for cam in cameras:
        try:
            images = imaging_analysis_lib_mod.get_images_camera(h5file,cam)
            if images is None:
                continue
            image_dmd, x_dmd, y_dmd = imaging_analysis_lib_mod.get_image_dmd(h5file)
            keys = list(images.keys())
            if len(keys) == 0:
                continue
            normal_keys = [key for key in keys if 'SVD' not in key]

            L = len(normal_keys)
            L_x_camera = 0
            fig,ax = plt.subplots(L*2+2,2,figsize=(L*6,6),constrained_layout=True,gridspec_kw={'width_ratios': [1, 2]})
            for i in range(L):
                j=0
                img = images[normal_keys[i]]
                L_x_camera = np.shape(img)[1]
                key = normal_keys[i]
                ax, OD_int_vert = plot_OD(img,key,ax,i,j)
                if i == 0:
                    model = Bimodal1DModel(pixel_size = cam['px_size'])
                    params = model.guess(OD_int_vert, X=np.arange(len(OD_int_vert)))
                    params['sx'].set(value = 100)
                    params['rx'].set(value = 2, max = 40)
                    params['offset'].set(value = 0, vary=True)
                    result = model.fit(OD_int_vert, X=np.arange(len(OD_int_vert)), params=params)
                    result.plot_fit(ax = ax[0, 0], show_init = True)
                    # ax[0, 0].plot(np.arange(len(OD_int_vert)), result.best_fit)
                    print(result.fit_report())
                    run.save_results_dict(parameters_dict_with_errors(result.params))
            
            image_dmd_full = np.zeros((1080,L_x_camera))
            for i in range(len(x_dmd)):
                x = x_dmd[i]
                x_dmd_int = int(np.round(x))
                for j in range(len(y_dmd)):
                    y = y_dmd[j]
                    y_dmd_int = int(np.round(y))
                    image_dmd_full[y_dmd_int,x_dmd_int] = image_dmd[j,i]
            ax[L*2,1].imshow(image_dmd_full,vmin=0,vmax=1,aspect='auto',cmap='binary')
            ax[L*2,1].set_title("DMD")
            ax[L*2,1].sharex(ax[L*2-1,1])
            img_dmd_full_int = np.sum(image_dmd_full,axis=0)
            ax[L*2+1,1].plot(img_dmd_full_int)
            ax[L*2+1,1].sharex(ax[L*2,1])
            ax[L*2,0].remove()
            ax[L*2+1,0].remove()

            fig.suptitle(cam['name'])
        except Exception as e:
            print(e)
            pass