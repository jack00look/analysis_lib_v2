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


reduce_image_func = SVD_analysis_reduce_mod.get_lateral_region

def SVD_training(year,month,day,sequences, camera,path=settings.bec2path,what_images = None,show_errs = False):
    
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

    return invU, invS, Vh, all_probes_full,shape_x,shape_y,filenames_list,keys_list,reduce_image_func

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

def analyze_SVD_h5file(h5file,cam,reduce_image_func,invU,invS,Vh,all_probes_full):
    images = imaging_analysis_lib_mod.get_raws_camera(h5file,cam)
    atoms_names = cam['atoms_images'].copy()
    OD = {}
    for atoms_name in atoms_names:
        if atoms_name not in images.keys():
            #print("Image " + atoms_name + " not found")
            continue
        try:
            img = images[atoms_name]
            img_cut = reduce_image_func(img)
            probe = reconstruct_probe(img_cut,invU,invS,Vh,all_probes_full, np.shape(img))
            general_lib_mod.save_data(h5file, 'results/raw/' + cam['name'] + '/' + atoms_name + '_SVDprobe' , probe)
        except Exception as e:
            raise("Error in processing image " + atoms_name, '(' + str(e) + ')')

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