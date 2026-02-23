import h5py
import lyse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import time
import traceback
import sys
import subprocess

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

cameras = camera_settings.cameras

time_start = time.time()

def fit_radial_gaussian(image):
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        return np.nan, np.nan
    img_min = np.nanmin(img)
    weights = img - img_min
    weights[weights < 0] = 0
    total = np.nansum(weights)
    if total <= 0:
        return np.nan, np.nan
    yy, xx = np.indices(img.shape)
    x0 = np.nansum(xx * weights) / total
    y0 = np.nansum(yy * weights) / total
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    mask = weights > 0
    r2_vals = r2[mask].ravel()
    i_vals = weights[mask].ravel()
    if r2_vals.size < 5:
        return np.nan, np.nan
    log_i = np.log(i_vals)
    slope, intercept = np.polyfit(r2_vals, log_i, 1)
    if slope >= 0:
        return np.nan, np.nan
    w = np.sqrt(-2.0 / slope)
    A = np.exp(intercept)
    return A, w

def show_raws(h5file):
    res_dict = {}
    for cam_entry in cameras:
        cam = cameras[cam_entry]
        try:
            images = imaging_analysis_lib_mod.get_raws_camera(h5file,cam,show_errs=False)
            if images is None:
                continue
            keys = list(images.keys())
            print('Camera:',cam['name'])
            print('Keys:',keys)
            print('Len(keys)=',len(keys))
            L = len(keys)
            fig,ax = plt.subplots(L*2,3,figsize=(20,5),constrained_layout=True)
            # if L==1:
            #     ax = np.array([ax])
            for i in range(L):
                # fig_temp = ax[i,0].imshow(images[keys[i]],vmin=0,vmax=cam['dynamic_range']-1,aspect='auto')
                fig_temp = ax[i*2,1].imshow(images[keys[i]],vmin=0,vmax=cam['dynamic_range'],aspect='auto')
                res_dict[keys[i] + '_sum'] = np.sum(images[keys[i]])
                A_fit, w_fit = fit_radial_gaussian(images[keys[i]])
                res_dict[keys[i] + '_gauss_A'] = A_fit
                res_dict[keys[i] + '_gauss_w'] = w_fit
                print(f"{cam['name']} | {keys[i]} | Gaussian fit: A={A_fit:.3g}, w={w_fit:.3g} px")
                integrated_profile_x = np.sum(images[keys[i]],axis=0)
                integrated_profile_y = np.sum(images[keys[i]],axis=1)
                ax[i*2+1,1].plot(np.arange(len(integrated_profile_x)),integrated_profile_x)
                ax[i*2,0].plot(integrated_profile_y, np.arange(len(integrated_profile_y)))
                ax[i*2+1,1].sharex(ax[i*2,1])
                ax[i*2,0].sharey(ax[i*2,1])
                fig.colorbar(fig_temp,ax=ax[i*2,1])
                ax[i*2,1].set_title(keys[i])
                ax[i*2,2].hist(images[keys[i]].flatten(),bins=int(cam['dynamic_range']/100))
                ax[i*2,2].set_xlim(0,cam['dynamic_range'])
                ax[i*2+1,0].remove()
                ax[i*2+1,2].remove()
                if i>0:
                    ax[i*2,1].sharex(ax[0,1])
                    ax[i*2,1].sharey(ax[0,1])
                    ax[i*2,2].sharex(ax[0,2])
            
            fig.suptitle(cam['name'])
        except Exception as e:
            print(traceback.format_exc())
    

    return res_dict

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

        
    
