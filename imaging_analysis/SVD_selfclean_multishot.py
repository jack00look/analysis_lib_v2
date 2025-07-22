import importlib
import importlib.util
import numpy as np
import h5py
from pathlib import PurePath

spec = importlib.util.spec_from_file_location("SVD_analysis.SVD_selfclean", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/SVD_analysis/SVD_selfclean.py")
SVD_selfclean_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVD_selfclean_mod)


spec = importlib.util.spec_from_file_location("general_lib.settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py")
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

cam_vert1 = camera_settings.cam_vert1
cam_hor1 = camera_settings.cam_hor1

try:
    year = 2024
    month = 12
    day = 17
    sequences = [9,]
    cam = cam_vert1
    SVD_selfclean_mod.SVD_selfclean(year,month,day,sequences,cam,settings.bec2path,what_images=['PTAI_probe'])
except Exception as e:
    raise(e)




