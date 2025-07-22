import h5py
import lyse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import time
import traceback

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

cameras = camera_settings.cameras

time_start = time.time()

def show_raws(h5file):
    dict = None
    for cam_entry in cameras:
        cam = cameras[cam_entry]
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
    dmd_raw_data = imaging_analysis_lib_mod.get_raws_dmd(h5file,True)
    dmd_raw_data_keys = list(dmd_raw_data.keys())
    if 'image' in dmd_raw_data_keys:
        img_dmd = dmd_raw_data['image']
        L = 1
        img_mask = None
        if 'mask' in dmd_raw_data_keys:
            L = 2
            img_mask = dmd_raw_data['mask']
        fig,ax = plt.subplots(L,1,figsize=(20,5),constrained_layout=True)
        if img_mask is not None:
            fig_temp = ax[0].imshow(img_mask,vmin=0,vmax=1,aspect='auto',cmap='binary')
            fig.colorbar(fig_temp,ax=ax[0])
            ax[0].set_title("Mask")
            ax[0].sharex(ax[1])
            ax[0].sharey(ax[1])
            ax[1].set_title("Image")
            fig_temp = ax[1].imshow(img_dmd,vmin=0,vmax=1,aspect='auto',cmap='binary')
            fig.colorbar(fig_temp,ax=ax[1])
            fig.suptitle("DMD")
        else:
            fig_temp = ax.imshow(img_dmd,vmin=0,vmax=1,aspect='auto',cmap='binary')
            fig.colorbar(fig_temp,ax=ax)
            ax.set_title("DMD")

    return dict

if __name__ == '__main__':
    time_0 = time.time()
    try:
        import lyse
        run = lyse.Run(lyse.path)
        dict = None
        with h5py.File(lyse.path,'r+') as h5file:
            dict = show_raws(h5file)
        if dict is not None:
            for key in dict:
                run.save_result(key,dict[key])
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
    except Exception as e:
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
        print(traceback.format_exc())


with h5py.File(lyse.path) as h5file:
    for cam_entry in cameras:
        cam = cameras[cam_entry]
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
    dmd_raw_data = imaging_analysis_lib_mod.get_raws_dmd(h5file,True)
    dmd_raw_data_keys = list(dmd_raw_data.keys())
    if 'image' in dmd_raw_data_keys:
        img_dmd = dmd_raw_data['image']
        L = 1
        img_mask = None
        if 'mask' in dmd_raw_data_keys:
            L = 2
            img_mask = dmd_raw_data['mask']
        fig,ax = plt.subplots(L,1,figsize=(20,5),constrained_layout=True)
        if img_mask is not None:
            fig_temp = ax[0].imshow(img_mask,vmin=0,vmax=1,aspect='auto',cmap='binary')
            fig.colorbar(fig_temp,ax=ax[0])
            ax[0].set_title("Mask")
            ax[0].sharex(ax[1])
            ax[0].sharey(ax[1])
            ax[1].set_title("Image")
            fig_temp = ax[1].imshow(img_dmd,vmin=0,vmax=1,aspect='auto',cmap='binary')
            fig.colorbar(fig_temp,ax=ax[1])
            fig.suptitle("DMD")
        else:
            fig_temp = ax.imshow(img_dmd,vmin=0,vmax=1,aspect='auto',cmap='binary')
            fig.colorbar(fig_temp,ax=ax)
            ax.set_title("DMD")

time_end = time.time()



        
    
