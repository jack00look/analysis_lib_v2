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

def show_ODs(h5file,show=True):
    infos_analysis = {}
    try:
        cam_entry = 'cam_vert1'
        cam = cameras[cam_entry]

        dict_images = imaging_analysis_lib_mod.get_images_camera_waxes(h5file,cam)
        if dict_images is None:
            return

        images = dict_images['images']
        infos_images = dict_images['infos']
        dmd_raw_data = dict_images['raws_dmd']

        keys = list(images.keys())
        if len(keys) == 0:
            return
        normal_keys = [key for key in keys if 'SVD' not in key]
        L = len(normal_keys)

        fig,ax = plt.subplots(L*2,2,figsize=(L*6,6),constrained_layout=True,gridspec_kw={'width_ratios': [1, 1]})
        for i in range(L):
            key = normal_keys[i]
            n2D,n1D_x,n1D_y,x_vals,y_vals = imaging_analysis_lib_mod.extract_images_parts(dict_images,cam,key)
            j=0
            for key_infos in infos_images[key]:
                infos_analysis[key+'_'+key_infos] = infos_images[key][key_infos]
            imaging_analysis_lib_mod.plot_OD(dict_images,key,cam,fig,ax,i,j)
            imaging_analysis_lib_mod.plot_dmd(dict_images,key,cam,fig,ax,i+1,j+1,a=3.5e-7,b=6.8e-4)
            #ax[2*i, 1].set_aspect(1)

            if i>0:
                ax[2*i,1].sharex(ax[0,1])
                ax[2*i,1].sharey(ax[0,1])
        
        detuning = h5file['globals'].attrs['probe_detuning_debug']/9.7946
        tof = h5file['globals'].attrs['tof_debug']
        title = 'CAM' + cam['name']
        title += '\nDetuning. {:.2f} [Gamma], TOF: {:.2f} ms'.format(detuning,tof)
        h5file_infos = general_lib_mod.get_h5file_infos(h5file.filename)
        title += '\n' + h5file_infos['year'] + '/' + h5file_infos['month'] + '/' + h5file_infos['day']
        title += '\n' + 'seq:' + h5file_infos['seq'] + '   ' + 'it:' + h5file_infos['it'] + '   ' + 'rep:' + h5file_infos['rep']
        title += '\n' + h5file_infos['program_name']
        fig.suptitle(title,fontsize=15)

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