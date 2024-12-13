import h5py
import lyse
import importlib
import matplotlib.pyplot as plt
import numpy as np

import camera_settings
importlib.reload(camera_settings)
from camera_settings import cameras

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

vmin = -1
vmax = 10

def plot_OD(img,key,ax,i,j):
    fig_temp = ax[2*i,j+1].imshow(img,vmin=vmin,vmax = vmax,aspect='auto')
    fig.colorbar(fig_temp,ax=ax[2*i,j+1])
    ax[2*i,j+1].set_title(key)
    OD_int_hor = np.sum(img,axis=0)
    OD_int_vert = np.sum(img,axis=1)
    ax[2*i+1,j+1].plot(OD_int_hor)
    ax[2*i,j].plot(OD_int_vert,np.arange(len(OD_int_vert)))
    ax[2*i,j].invert_xaxis()
    ax[2*i+1,j+1].sharex(ax[2*i,j+1])
    ax[2*i,j].sharey(ax[2*i,j+1])
    ax[2*i+1,j].remove()
    return ax

with h5py.File(lyse.path) as h5file:
    for cam in cameras:
        try:
            images = imaging_analysis_lib_mod.get_images_camera(h5file,cam)
            if images is None:
                continue
            keys = list(images.keys())
            if len(keys) == 0:
                continue
            SVD_keys = [key for key in keys if 'SVD' in key]
            normal_keys = [key for key in keys if 'SVD' not in key]

            L = len(normal_keys)
            if SVD_keys != []:
                fig,ax = plt.subplots(L*2,4,figsize=(L*6,12),constrained_layout=True,gridspec_kw={'width_ratios': [1, 2,1,2]})
            else:
                fig,ax = plt.subplots(L*2,2,figsize=(L*6,6),constrained_layout=True,gridspec_kw={'width_ratios': [1, 2]})
            for i in range(L):
                j=0
                img = images[normal_keys[i]]
                key = normal_keys[i]
                plot_OD(img,key,ax,i,j)
                if i>0:
                    ax[2*i,1].sharex(ax[0,1])
                    ax[2*i,1].sharey(ax[0,1])
                if len(SVD_keys) > i:
                    j=2
                    img = images[SVD_keys[i]]
                    key = SVD_keys[i]
                    plot_OD(img,key,ax,i,j)
                    if i>0:
                        ax[2*i,3].sharex(ax[0,3])
                        ax[2*i,3].sharey(ax[0,3])
            
            fig.suptitle(cam['name'])
        except Exception as e:
            pass