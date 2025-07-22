import importlib
import lyse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import traceback

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

run = lyse.Run(lyse.path)

cam = camera_settings.cam_vert1


def show_magnetization(h5filepath,SVD,run):
    with h5py.File(h5filepath) as h5file:
        try:
            images = imaging_analysis_lib_mod.get_images_camera(h5file,cam)

            x_axis = np.arange(0,images[cam['atoms_images'][0]].shape[1])*cam['px_size']*1e6
            y_axis = np.arange(0,images[cam['atoms_images'][0]].shape[0])*cam['px_size']*1e6
            X,Y = np.meshgrid(x_axis,y_axis)

            if images is None:
                print('No images found')
            
            atoms_keys = cam['atoms_images'].copy()
            if SVD:
                atoms_keys = [(key+'_SVD') for key in atoms_keys]

            int_images = {}
            for key in atoms_keys:
                int_images[key] = np.sum(images[key],axis=0)
            
            fig,ax = plt.subplots(3,2,figsize=(12,12),constrained_layout=True,sharex='all')

            for i in range(2):
                img = images[atoms_keys[i]]
                int = int_images[atoms_keys[i]]
                img_plot = ax[i,0].pcolormesh(X,Y,img,shading='auto',vmin=-1,vmax=10)
                plt.colorbar(img_plot,ax=ax[i,0])
                ax[i,0].set_title(atoms_keys[i])
                ax[i,1].plot(x_axis,int)
                ax[i,1].set_title('Integrated '+atoms_keys[i])

            n_m1_1D = int_images[atoms_keys[0]]
            n_m2_1D = int_images[atoms_keys[1]]
            N = n_m1_1D + n_m2_1D
            Z = (n_m1_1D - n_m2_1D)/N

            ax[2,0].plot(x_axis,Z)
            ax[2,0].set_title('Magnetization')
            ax[2,0].set_xlabel('x [um]')
            ax[2,0].set_ylabel('Magnetization')
            ax[2,0].set_ylim(-1,1)

            ax[2,1].plot(x_axis,N)
            ax[2,1].set_title('Total density 1D')
            ax[2,1].set_xlabel('x [um]')
            ax[2,1].set_ylabel('Total density')
        except Exception as e:
            print('Error in show_magnetization: ', traceback.format_exc())
    
try:
    SVD = True
    run = lyse.Run(lyse.path)
    show_magnetization(lyse.path,SVD,run)

except Exception as e:
    print('Error in show_magnetization: ', traceback.format_exc())   
