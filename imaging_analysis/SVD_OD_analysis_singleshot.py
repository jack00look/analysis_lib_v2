import importlib
import lyse

spec = importlib.util.spec_from_file_location("SVD_analysis.SVD_OD_analysis", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/SVD_analysis/SVD_OD_analysis.py")
SVD_OD_analysis_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVD_OD_analysis_mod)

import camera_settings
importlib.reload(camera_settings)
from camera_settings import cam_vert1,cam_hor1

try:
    year = 2024
    month = 12
    day = 2
    SVD_sequence = 5
    SVD_OD_analysis_mod.SVD_OD_analysis(lyse.path, cam_vert1, year, month, day, SVD_sequence)
except Exception as e:
    raise('Error in SVD analysis for cam', cam_vert1['name'], ':', e)

