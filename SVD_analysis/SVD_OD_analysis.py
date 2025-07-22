import numpy as np
import h5py
import json
import datetime
import importlib
import time
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("general_lib.main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/main.py")
general_lib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(general_lib_mod)

spec = importlib.util.spec_from_file_location("general_lib.settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py")
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)

spec = importlib.util.spec_from_file_location("SVD_analysis.SVD_analysis_main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/SVD_analysis/SVD_analysis_main.py")
SVD_main_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVD_main_mod)



def get_SVD_savepath(year,month,day,SVD_number,path):
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
    path = path / str(SVD_number)
    if not(path.exists()):
        print("get_SVD_savepath: SVD number folder does not exist")
        return
    return path


def SVD_OD_analysis(h5filepath,cam,year,month,day,SVD_number,path = settings.bec2path):

    SVD_path = get_SVD_savepath(year,month,day,SVD_number,path)

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
    

    with h5py.File(h5filepath) as h5file:
        try:
            SVD_main_mod.analyze_SVD_h5file(h5file,cam,reduce_image_mod.reduce_image_func,invU,invS,Vh,all_probes_full)
            cam_raws_h5name = 'results/raw/' + cam['name']
            if cam_raws_h5name in h5file:
                h5file[cam_raws_h5name].attrs['comment'] = 'SVD done with ' + str(SVD_path) + ' SVD dataset'

        except Exception as e:
            raise("Error in SVD_OD_analysis for " + cam['name'] + ": " + str(e))
            return
    return

