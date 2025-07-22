import importlib
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt


spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

cameras = camera_settings.cameras

vmin = -1
vmax = 15

w_x = 1200
w_y = 300

def plot_OD(img,key,fig,ax,i,j):
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

def get_ClinClog(h5file,show=True):
    for cam in cameras:
        try:
            raws = imaging_analysis_lib_mod.get_raws_camera(h5file,cam)
            images = imaging_analysis_lib_mod.get_images_camera(h5file,cam)
            if images is None:
                continue
            keys = list(images.keys())
            if len(keys) == 0:
                continue
            normal_keys = [key for key in keys if 'SVD' not in key]
            key = normal_keys[0]
            image = images[normal_keys[0]]
            atom_raw = raws[normal_keys[0]]
            probe_raw = raws[cam['probe_image']]
            back_raw = raws[cam['background_image']]
            smooth_image = gaussian_filter(image,10)
            center = np.where(smooth_image == np.amax(smooth_image))

            dict = {}

            y_c = center[0][0]
            x_c = center[1][0]
            dict['y_center'] = y_c
            dict['x_center'] = x_c
            dict['x_width'] = w_x
            dict['y_width'] = w_y
            ym = y_c-w_y//2
            yM = y_c+w_y//2
            xm = x_c-w_x//2
            xM = x_c+w_x//2
            roi = np.s_[ym:yM,xm:xM]


            pulse_time = cam['pulse_time']
            dict['pulse_time'] = pulse_time
            roi_back = cam['roi_back']
            dict['roi_back'] = np.array(roi_back)
            probe = probe_raw - back_raw
            atom = atom_raw - back_raw
            factor = np.sum(atom[roi_back])/np.sum(probe[roi_back])
            probe_rescaled = np.copy(probe)
            probe_rescaled = probe_rescaled*factor
            w_y_array = np.arange(2,62,2)
            Clog_array = np.zeros(len(w_y_array))
            Clin_array = np.zeros(len(w_y_array))
            Clin_int_array = np.zeros(len(w_y_array))
            Clog_int_array = np.zeros(len(w_y_array))
            for i in range(len(w_y_array)):
                w_y_val = w_y_array[i]
                w_x_val = w_y_val*5
                probe_int = np.mean(probe_rescaled[y_c-w_y_val//2:y_c+w_y_val//2,x_c-w_x_val//2:x_c+w_x_val//2])
                atom_int = np.mean(atom[y_c-w_y_val//2:y_c+w_y_val//2,x_c-w_x_val//2:x_c+w_x_val//2])
                log_array = np.log(probe_rescaled/atom)[y_c-w_y_val//2:y_c+w_y_val//2,x_c-w_x_val//2:x_c+w_x_val//2]
                Clog_int_array[i] = np.mean(log_array)
                Clin_int_array[i] = np.mean((probe_rescaled-atom)[y_c-w_y_val//2:y_c+w_y_val//2,x_c-w_x_val//2:x_c+w_x_val//2]/pulse_time)
                Clog_array[i] = np.log(probe_int/atom_int)
                Clin_array[i] = (probe_int-atom_int)/pulse_time
            
            dict['Clog_int_array'] = Clog_int_array
            dict['Clin_int_array'] = Clin_int_array
            dict['Clog_array'] = Clog_array
            dict['Clin_array'] = Clin_array
            Clog_array = np.log((np.clip(probe_rescaled[roi],1,None))/ (np.clip(atom[roi],1,None)))
            Clog = np.sum(Clog_array)
            Clin_array = (probe_rescaled[roi]-atom[roi])/pulse_time
            Clin = np.sum(Clin_array)
            
            dict['w_y_array'] = w_y_array
            dict['Clog'] = Clog
            dict['Clin'] = Clin

            if show:
                fig,ax = plt.subplots(2,2,figsize=(6,6),constrained_layout=True,gridspec_kw={'width_ratios': [1, 2]})
                plot_OD(image,key,fig,ax,0,0)
                ax[0,1].plot([xm,xm],[ym,yM],'r')
                ax[0,1].plot([xM,xM],[ym,yM],'r')
                ax[0,1].plot([xm,xM],[ym,ym],'r')
                ax[0,1].plot([xm,xM],[yM,yM],'r')
                fig.suptitle(h5file.filename + '\n' + cam['name'])
                plt.show()

            return dict
        except Exception as e:
            raise(e)

if __name__ == "__main__":
    import lyse

    run = lyse.Run(lyse.path)
    with h5py.File(lyse.path,'r+') as h5file:
        dict = get_ClinClog(h5file)
    if dict is not None:
        for key in dict.keys():
            run.save_result(key, dict[key])

    