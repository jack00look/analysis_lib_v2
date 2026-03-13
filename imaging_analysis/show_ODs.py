import h5py
import importlib
import matplotlib.pyplot as plt
import numpy as np
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, Gaussian1DModel
from lmfit_lyse_interface import parameters_dict_with_errors
import traceback
import time
import subprocess
import sys

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

warning_script_name = "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/open_warning_window.py"

cameras = camera_settings.cameras

def show_ODs(h5file,show=True,show_SVD=True):
    """
    Show OD images from h5 file.
    
    Parameters:
    -----------
    show_SVD : bool, optional
        If True, also show OD images computed with SVD-reconstructed probes (default: True)
    """
    infos_analysis = {}
    try:
        for cam_entry in cameras:
            cam = cameras[cam_entry]

            dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file,cam)
            if dict_images is None:
                continue

            # Also get SVD images if available and requested
            dict_images_SVD = None
            if show_SVD:
                dict_images_SVD = imaging_analysis_lib_mod.get_images_camera_waxes_SVD(h5file,cam)
                if dict_images_SVD is not None:
                    print(f"Found SVD images: {list(dict_images_SVD['images'].keys())}")
                else:
                    print(f"No SVD images found for camera {cam['name']}")

            images = dict_images['images']
            infos_images = dict_images['infos']

            keys = list(images.keys())
            if len(keys) == 0:
                continue
            normal_keys = [key for key in keys if 'SVD' not in key]
            L = len(normal_keys)

            # Create figure for normal ODs
            fig,ax = plt.subplots(L*2,2,figsize=(L*6,6),constrained_layout=True,gridspec_kw={'width_ratios': [1, 1]})
            print(f"Creating normal OD figure with {L} images, {L*2} rows of subplots")
            for i in range(L):
                key = normal_keys[i]
                print('Processing key:',key)
                j=0
                for key_infos in infos_images[key]:
                    infos_analysis[key+'_'+key_infos] = infos_images[key][key_infos]
                n1D_x,n1D_y = imaging_analysis_lib_mod.plot_OD(dict_images,key,cam,fig,ax,i,j)
                infos_analysis[key+'_n1D_x'] = n1D_x
                x_vals = np.arange(len(n1D_x))
                y_vals = np.arange(len(n1D_y))
                com_x = np.sum(n1D_x*x_vals)/np.sum(n1D_x)
                com_y = np.sum(n1D_y*y_vals)/np.sum(n1D_y)
                infos_analysis[key+'_com_x'] = com_x
                infos_analysis[key+'_com_y'] = com_y

                infos_analysis[key[5:]+'_1d'] = n1D_x

                infos_analysis[key+'_n1D_y'] = n1D_y

                if i>0:
                    ax[2*i,1].sharex(ax[0,1])
                    ax[2*i,1].sharey(ax[0,1])
                              
            title = general_lib_mod.get_title(h5file,cam)
            fig.suptitle(title,fontsize=12)
            
            # Create separate figure for SVD ODs if available
            if dict_images_SVD is not None:
                images_svd = dict_images_SVD['images']
                infos_images_svd = dict_images_SVD['infos']
                svd_keys = list(images_svd.keys())
                L_svd = len(svd_keys)
                
                if L_svd > 0:
                    fig_svd,ax_svd = plt.subplots(L_svd*2,2,figsize=(L_svd*6,6),constrained_layout=True,gridspec_kw={'width_ratios': [1, 1]})
                    print(f"Creating SVD OD figure with {L_svd} images, {L_svd*2} rows of subplots")
                    
                    for i in range(L_svd):
                        key = svd_keys[i]
                        print('Processing SVD key:',key)
                        j=0
                        for key_infos in infos_images_svd[key]:
                            infos_analysis[key+'_'+key_infos] = infos_images_svd[key][key_infos]
                        n1D_x,n1D_y = imaging_analysis_lib_mod.plot_OD(dict_images_SVD,key,cam,fig_svd,ax_svd,i,j)
                        infos_analysis[key+'_n1D_x'] = n1D_x
                        x_vals = np.arange(len(n1D_x))
                        y_vals = np.arange(len(n1D_y))
                        com_x = np.sum(n1D_x*x_vals)/np.sum(n1D_x)
                        com_y = np.sum(n1D_y*y_vals)/np.sum(n1D_y)
                        infos_analysis[key+'_com_x'] = com_x
                        infos_analysis[key+'_com_y'] = com_y

                        infos_analysis[key+'_n1D_x'] = n1D_x
                        infos_analysis[key+'_n1D_y'] = n1D_y

                        if i>0:
                            ax_svd[2*i,1].sharex(ax_svd[0,1])
                            ax_svd[2*i,1].sharey(ax_svd[0,1])
                    
                    title_svd = general_lib_mod.get_title(h5file,cam) + "\n[SVD Reconstructed Probes]"
                    fig_svd.suptitle(title_svd,fontsize=12)

        for key in infos_analysis:
            if 'flag' in key:
                warning_message = key.replace(" ", "_")
                #subprocess.Popen([sys.executable, warning_script_name,warning_message])

        return infos_analysis
    except Exception as e:
        print(traceback.format_exc())
        return

if __name__ == '__main__':
    time_0 = time.time()
    try:
        import lyse
        run = lyse.Run(lyse.path)
        dict = None
        with h5py.File(lyse.path,'r+') as h5file:
            dict = show_ODs(h5file)
        if dict is not None:
            for key in dict:
                run.save_result(key,dict[key])
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
    except Exception as e:
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
        print(traceback.format_exc())