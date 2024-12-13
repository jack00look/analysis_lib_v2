import numpy as np
import h5py
import time
import importlib
from numba import jit

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

def calculate_OD(atoms, probe, cam, back = None, roi_background=None):
    if roi_background is not None:
        probe = probe * (atoms[roi_background].mean()/probe[roi_background].mean())
    

    if back is not None:
        atoms = atoms - back
        probe = probe - back

    alpha = cam['alpha']
    chi_sat = cam['chi_sat']
    pulse_time = cam['pulse_time']

    OD = alpha*np.log( (np.array(probe).clip(1)) / (np.array(atoms).clip(1))  ) + (probe -  atoms) / (chi_sat * pulse_time)

    return OD

def get_raws_camera(h5file,cam,show_errs = False):
    try:
        if cam['h5group'] not in h5file:
            if show_errs:
                print(h5file.filename + " has no " + cam['name'] + " images")
            return
        imgs = h5file[cam['h5group']]
        img_names = [s.decode('utf-8') for s in imgs.attrs['img_names']]
        images = dict(zip(img_names, imgs))
        return images
    except Exception as e:
        if show_errs:
            raise("Error in getting raws from cam" + cam['name'] + ": " + str(e))
        return

def get_images_camera(h5file,cam,show_errs = False):
    try:
        atoms_keys = cam['atoms_images'].copy()
        probe_key = cam['probe_image']
        background_key = cam['background_image']
        
        images_raws = get_raws_camera(h5file,cam)
        images = {} 
        for key in atoms_keys:
            try:
                images[key] = calculate_OD(images_raws[key], images_raws[probe_key], cam, images_raws[background_key], cam['roi_back'])
            except Exception as e:
                #print("Error in calculating OD for " + key + " in cam " + cam['name'], e)
                pass
            try:
                probe_SVD = h5file['results/raw/' + cam['name'] + '/' + key + '_SVDprobe']
                images[key+'_SVD'] = calculate_OD(images_raws[key], probe_SVD, cam, images_raws[background_key])
            except Exception as e:
                #print("Error in calculating OD for " + key + "_SVD in cam " + cam['name'], e)
                pass
        return images
    except Exception as e:
        if show_errs:
            raise("Error in getting imagges from cam" + cam['name'] + ": " + str(e))
        return

