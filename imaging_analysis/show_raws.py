import h5py
import lyse
import importlib
import matplotlib.pyplot as plt

import camera_settings
importlib.reload(camera_settings)
from camera_settings import cameras

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)



with h5py.File(lyse.path) as h5file:
    for cam in cameras:
        try:
            images = imaging_analysis_lib_mod.get_raws_camera(h5file,cam)
            if images is None:
                continue
            keys = list(images.keys())
            L = len(keys)
            fig,ax = plt.subplots(L,2,figsize=(20,5),constrained_layout=True)
            for i in range(L):
                #fig_temp = ax[i,0].imshow(images[keys[i]],vmin=0,vmax=cam['dynamic_range']-1,aspect='auto')
                fig_temp = ax[i,0].imshow(images[keys[i]],vmin=0,vmax=cam['dynamic_range'],aspect='auto')
                fig.colorbar(fig_temp,ax=ax[i,0])
                ax[i,0].set_title(keys[i])
                ax[i,1].hist(images[keys[i]].flatten(),bins=int(cam['dynamic_range']/100))
                ax[i,1].set_xlim(0,cam['dynamic_range'])
                if i>0:
                    ax[i,0].sharex(ax[0,0])
                    ax[i,0].sharey(ax[0,0])
                    ax[i,1].sharex(ax[0,1])
            
            fig.suptitle(cam['name'])
        except Exception as e:
            pass
    
