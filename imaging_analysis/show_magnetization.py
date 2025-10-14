import h5py
import importlib
import matplotlib.pyplot as plt
import numpy as np
from fitting.models1D import ThomasFermi1DModel, Bimodal1DModel, Gaussian1DModel
from lmfit_lyse_interface import parameters_dict_with_errors
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

def show_magnetization(h5file,show=True):

    infos_analysis = {}

    try:

        cam = cameras['cam_vert1']

        dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file,cam)
        if dict_images is None:
            return None

        images = dict_images['images']
        images_ODlog = dict_images['images_ODlog']
        infos_images = dict_images['infos']

        keys = list(images.keys())
        if len(keys) == 0:
            return None
        normal_keys = [key for key in keys if 'SVD' not in key]
        L = len(normal_keys)


        img_m1 = images['PTAI_m1']
        img_m2 = images['PTAI_m2']
        
        fig,ax = plt.subplots(6,1,figsize=(L*6,6),constrained_layout=True, sharex=True)
        ax[0].imshow(images_ODlog['PTAI_m1'],vmin = 0,vmax = 1.5,cmap = 'gist_stern')
        ax[0].set_title('m1')

        ax[1].imshow(images_ODlog['PTAI_m2'],vmin = 0,vmax = 1.5,cmap = 'gist_stern')
        ax[1].set_title('m2')

        D_tot = img_m1 + img_m2
        Z = (img_m1 - img_m2)/(D_tot)
        ax[2].imshow(Z,vmin = -1,vmax = 1,cmap = 'RdBu')
        ax[2].set_title('Z = (m1-m2)/(m1+m2)')

        ax[3].imshow(images_ODlog['PTAI_m1'] + images_ODlog['PTAI_m2'],vmin = 0,vmax = 1.5,cmap = 'gist_stern')
        ax[3].set_title('m1 + m2')

        n1d_m1 = np.sum(img_m1,axis=0)
        n1d_m2 = np.sum(img_m2,axis=0)

        n1d_tot = n1d_m1 + n1d_m2
        z1d = (n1d_m1 - n1d_m2)/(n1d_tot)

        ax[4].plot(n1d_tot,label='m1')
        ax[4].set_title('n1D')

        ax[5].plot(z1d,label='m2')
        ax[5].set_title('z1D')
        ax[5].set_ylim([-1.1,1.1])
    
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
            dict = show_magnetization(h5file)
        if dict is not None:
            for key in dict:
                run.save_result(key,dict[key])
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
    except Exception as e:
        print('Elapsed time: {:.2f} s'.format(time.time()-time_0))
        print(traceback.format_exc())