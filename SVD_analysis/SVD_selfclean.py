import importlib
from pathlib import Path
import numpy as np
import json
import sys
import h5py

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

spec = importlib.util.spec_from_file_location("SVD_analysis.SVD_analysis_main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/SVD_analysis/SVD_analysis_main.py")
SVD_main_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVD_main_mod)

spec = importlib.util.spec_from_file_location("SVD_analysis.SVD_training", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/SVD_analysis/SVD_training.py")
SVD_training_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVD_training_mod)

spec = importlib.util.spec_from_file_location("general_lib.settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py")
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)



def get_last_SVDsavepath(year,month,day,path):
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
        print("get_SVD_savepath: SVD folder does not exist")
        return
    folder_list  = np.array([f for f in path.iterdir() if f.is_dir()])
    if np.shape(folder_list)[0] == 0:
        print("get_SVD_savepath: No SVD folder found")
        return
    folder_number = np.max([int(f.name) for f in folder_list])
    path = path / str(folder_number)
    return path



def SVD_selfclean(year,month,day,sequences,cam,path=settings.bec2path,what_images=None):
    SVD_training_mod.SVD_training_save(year,month,day,sequences,cam,path,what_images)
    SVD_path = get_last_SVDsavepath(year,month,day,path)

    infos = json.load(open(SVD_path / 'infos.json'))

    spec = importlib.util.spec_from_file_location("reduce_image_module.reduce_image_func", str(SVD_path / 'reduce_image_module'/'reduce_image_func.py'))
    reduce_image_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reduce_image_mod)

    if infos['camera'] != cam['name']:
        print('Camera name does not match')
        return

    invU = np.load(SVD_path / 'invU.npy')
    invS = np.load(SVD_path / 'invS.npy')
    Vh = np.load(SVD_path / 'Vh.npy')

    all_probes_full_path = Path(SVD_path / 'all_probes_full.npy')
    all_probes_full = None
    if all_probes_full_path.exists():
        all_probes_full = np.load(SVD_path / 'all_probes_full.npy')
    else:
        filenames_list = np.load(SVD_path / 'filename_list.npy',allow_pickle=True)
        keys_list = np.load(SVD_path / 'keys_list.npy',allow_pickle=True)
        all_probes_full = SVD_main_mod.get_probes(SVD_path,path,filenames_list,keys_list,cam)
    
    filenames = general_lib_mod.get_ordered_multiple_sequences(year,month,day,sequences,path,show_reps=True)
    L = len(filenames)
    cnt = 0
    for filename in filenames:
        with h5py.File(filename) as h5file:
            try:
                SVD_main_mod.analyze_SVD_h5file(h5file,cam,reduce_image_mod.reduce_image_func,invU,invS,Vh,all_probes_full)
                cam_raws_h5name = 'results/raw/' + cam['name']
                if cam_raws_h5name in h5file:
                    h5file[cam_raws_h5name].attrs['comment'] = 'SVD done with ' + str(SVD_path) + ' SVD dataset'

            except Exception as e:
                raise("Error in SVD_OD_analysis for " + cam['name'] + ": " + str(e))
        cnt += 1
        print('Done with ' + str(cnt) + ' out of ' + str(L) + ' files', end="\r", flush=True)
    return
    
