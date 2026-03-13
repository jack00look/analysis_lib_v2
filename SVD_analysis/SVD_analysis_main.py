import importlib
import numpy as np
import h5py
from pathlib import Path
import json
import inspect
import os

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/main.py")
imaging_analysis_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imaging_analysis_lib_mod)

spec = importlib.util.spec_from_file_location("SVD_analysis.SVD_analysis_reduce", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/SVD_analysis/SVD_analysis_reduce.py")
SVD_analysis_reduce_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVD_analysis_reduce_mod)

spec = importlib.util.spec_from_file_location("general_lib.settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py")
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)

reduce_image_func = SVD_analysis_reduce_mod.get_two_lateral_regions

def SVD_training(year,month,day,sequences, camera,path=settings.bec2path,what_images = None,show_errs = False,reduce_func_name='get_two_lateral_regions',reduce_func_params=None):
    
    # Select the reduce function
    if reduce_func_name == 'get_corners_custom':
        if reduce_func_params is None:
            reduce_func_params = {'vert_size': 20, 'hor_size': 100}
        reduce_image_func = lambda img: SVD_analysis_reduce_mod.get_corners_custom(img, **reduce_func_params)
    elif reduce_func_name == 'get_two_lateral_regions':
        reduce_image_func = SVD_analysis_reduce_mod.get_two_lateral_regions
    elif reduce_func_name == 'get_lateral_region':
        reduce_image_func = SVD_analysis_reduce_mod.get_lateral_region
    elif reduce_func_name == 'get_corners':
        reduce_image_func = SVD_analysis_reduce_mod.get_corners
    else:
        raise ValueError(f"Unknown reduce_func_name: {reduce_func_name}")
    
    filenames_sequence = general_lib_mod.get_ordered_multiple_sequences(year,month,day,sequences,path,show_reps=True,show_errs=show_errs)

    all_probes = []
    all_probes_full = []
    shape_x =0
    shape_y = 0

    if what_images is None:
        what_images = [camera['probe_image']]

    filenames_list = []
    keys_list = []

    shape_found = False

    for filename in filenames_sequence:
        try:
            raws = None
            with h5py.File(filename,'r') as h5file:
                raws = imaging_analysis_lib_mod.get_raws_camera(h5file,camera)
            for what_image in what_images:

                probe = raws[what_image]

                # check if probe intensity is enough (should avoid 'lost' images)
                if np.sum(probe>5000) < 100000:
                    pass

                # renormalize probe images so that they all have the same total intensity, thus being weighted equally in the SVD
                probe = probe/np.sum(probe)

                # checks if all probe images have the same shape
                if not shape_found:
                    shape_y, shape_x = np.shape(probe)
                    shape_found = True
                    print(f'Training image shape: {shape_y} x {shape_x} pixels')
                if shape_found:
                    if np.shape(probe) != (shape_y,shape_x):
                        print('Error in probe shape in file ' + filename)
                        continue

                # get the portion of the probe image which will be used for the SVD
                probe_cut = reduce_image_func(probe)

                all_probes.append(probe_cut.flatten())
                all_probes_full.append(probe.flatten())
                filenames_list.append(filename)
                keys_list.append(what_image)

        except:
            pass

    all_probes = np.array(all_probes)
    all_probes_full = np.array(all_probes_full)
    filenames_list = np.array(filenames_list)
    keys_list = np.array(keys_list)

    # perform SVD
    L = np.shape(all_probes)[0]

    print('Starting SVD')
    U, S, Vh = np.linalg.svd(all_probes, full_matrices=False)
    print('SVD done')

    # truncate SVD
    U = U[:,:L]
    S = S[:L]
    Vh = Vh[:L,:]

    # invert U and S for the reconstruction
    print('Inverting U and S')
    invS = np.linalg.inv(np.diag(S))
    invU = np.linalg.inv(U)
    print('Inversion done')

    return invU, invS, Vh, all_probes_full,shape_x,shape_y,filenames_list,keys_list,reduce_image_func,reduce_func_name,reduce_func_params

def make_SVD_savepath(year,month,day,path):
    path = Path(path)
    if not(path.exists()):
        print("get_SVD_savepath: main_path does not exist (main_path = " + str(path) + ")")
        return
    path = path / str(year)
    if not(path.exists()):
        print("get_SVD_savepath: Year folder does not exist")
        return
    path = path / str(month).zfill(2)
    if not(path.exists()):
        print("get_SVD_savepath: Month folder does not exist")
        return
    path = path / str(day).zfill(2)
    if not(path.exists()):
        print("get_SVD_savepath: Day folder does not exist")
        return
    path = path / 'SVD'
    if not(path.exists()):
        path.mkdir()
    
    folder_list  = np.array([f for f in path.iterdir() if f.is_dir()])
    if np.shape(folder_list)[0] == 0:
        folder_number = 0
    else:
        folder_number = np.max([int(f.name) for f in folder_list]) + 1
    path = path / str(folder_number)
    print('Saving in ', path)
    path.mkdir()
    return path

def reconstruct_probe(img_cut,invU,invS,Vh,all_probes_full, img_shape):
    proj = np.dot(img_cut.flatten(),np.transpose(Vh))
    probe = np.reshape(np.dot(np.transpose(proj), np.dot(invS,np.dot(invU,all_probes_full))), img_shape)
    return probe

def analyze_SVD_h5file(h5file,cam,reduce_image_func,invU,invS,Vh,all_probes_full,training_shape,alpha=0.0):
    """
    Analyze h5 file and reconstruct probe images using SVD.
    
    Parameters:
    -----------
    training_shape : tuple
        Shape (height, width) of training images for proper reconstruction
    alpha : float, optional
        Blending weight between SVD reconstruction and actual probe
        0.0 = pure SVD (default), 1.0 = pure actual probe
        Recommended: 0.2-0.3 for hybrid approach
    """
    # get raw images
    images = imaging_analysis_lib_mod.get_raws_camera(h5file,cam)

    # if no raw images, end analysis
    if images is None:
        return
    
    # get the names of the atoms images and probe image
    atoms_names = cam['atoms_images'].copy()
    probe_name = cam['probe_image']

    # get actual probe image if available and alpha > 0
    actual_probe = None
    if alpha > 0 and probe_name in images.keys():
        actual_probe = np.array(images[probe_name])
        # Check if current probe shape matches training shape
        if actual_probe.shape != training_shape:
            print(f"Warning: Current probe shape {actual_probe.shape} != training shape {training_shape}")
            print("Cannot use hybrid mode with mismatched shapes. Using pure SVD (alpha=0).")
            alpha = 0
            actual_probe = None
        else:
            # normalize by total intensity (same as in training)
            actual_probe = actual_probe / np.sum(actual_probe)

    # for each atoms image, reconstruct the probe and save it
    OD = {}
    for atoms_name in atoms_names:

        # if the atoms image is not in the raw images, skip it
        if atoms_name not in images.keys():
            continue

        try:
            img = np.array(images[atoms_name])
            
            # Check if current image shape matches training shape
            if img.shape != training_shape:
                print(f"Warning: Image {atoms_name} shape {img.shape} != training shape {training_shape}")
                print(f"Skipping {atoms_name} - cannot apply SVD with mismatched image sizes.")
                continue
            
            # Extract same corner regions as used in training
            img_cut = reduce_image_func(img)
            
            # SVD reconstruction
            proj = np.dot(img_cut.flatten(),np.transpose(Vh))
            probe_svd = np.reshape(np.dot(np.transpose(proj), np.dot(invS,np.dot(invU,all_probes_full))), training_shape)
            
            # Blend with actual probe if alpha > 0
            if actual_probe is not None and alpha > 0:
                probe = (1 - alpha) * probe_svd + alpha * actual_probe
            else:
                probe = probe_svd
            
            # Debug: Compare SVD probe with actual probe
            if probe_name in images.keys():
                actual_probe_full = np.array(images[probe_name])
                # Normalize both for comparison
                probe_norm = probe / np.sum(probe)
                actual_norm = actual_probe_full / np.sum(actual_probe_full)
                diff = np.abs(probe_norm - actual_norm)
                diff_percent = np.mean(diff) / np.mean(actual_norm) * 100
                max_diff_percent = np.max(diff) / np.mean(actual_norm) * 100
                print(f"  {atoms_name}: SVD probe differs from actual by {diff_percent:.2f}% avg, {max_diff_percent:.2f}% max")
            
            # Note: probe is normalized (sum ~= 1/N_training_images), but calculate_OD 
            # will renormalize it to match atoms intensity using background region, so this is fine
            
            general_lib_mod.save_data(h5file, 'results/raw/' + cam['name'] + '/' + atoms_name + '_SVDprobe' , probe)
        except Exception as e:
            print("Error in processing image " + atoms_name, '(' + str(e) + ')')

def get_probes(SVD_path,NAS_path,filenames_list,keys_list,cam):
    probes = []
    for i in range(len(filenames_list)):
        with h5py.File(filenames_list[i]) as h5file:
            images = imaging_analysis_lib_mod.get_raws_camera(h5file,cam)
            probe = images[keys_list[i]]
            probe = probe/np.sum(probe)
            probes.append(probe.flatten())
    probes = np.array(probes)
    np.save(SVD_path / 'all_probes_full.npy', probes)
    with open(NAS_path + '/SVD_allprobefiles_list.txt', 'a') as f:
        f.write(str(SVD_path / 'all_probes_full.npy') + '\n')
    return probes